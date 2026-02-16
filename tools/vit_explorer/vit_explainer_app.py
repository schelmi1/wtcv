#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Gradio is required. Install with: pip install gradio") from exc


_MODEL: torch.nn.Module | None = None
PATCH_SIZE = 14
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def _load_dino() -> torch.nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    # Hub name requested by user.
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=True)
    model.eval()
    _MODEL = model
    return model


def _resize_to_patch_grid(img: Image.Image, max_side: int = 560) -> Image.Image:
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    nw = max(PATCH_SIZE, int(round(w * scale)))
    nh = max(PATCH_SIZE, int(round(h * scale)))
    nw = max(PATCH_SIZE, (nw // PATCH_SIZE) * PATCH_SIZE)
    nh = max(PATCH_SIZE, (nh // PATCH_SIZE) * PATCH_SIZE)
    return img.resize((nw, nh), Image.BILINEAR)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(DINO_MEAN, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(DINO_STD, dtype=x.dtype).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def _flatten_tensors(obj: Any) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if torch.is_tensor(obj):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_tensors(v))
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_tensors(v))
    return out


def _extract_patch_tokens_from_obj(obj: Any, expected_patches: int) -> torch.Tensor:
    tensors = _flatten_tensors(obj)
    best: torch.Tensor | None = None
    for t in tensors:
        if t.ndim == 4:
            b, c, h, w = t.shape
            if b == 1 and (h * w) == expected_patches:
                cand = t.flatten(2).transpose(1, 2)  # [1,N,C]
                return cand
        if t.ndim == 3:
            b, n, c = t.shape
            if b != 1:
                continue
            if n == expected_patches:
                return t
            if n > expected_patches:
                # For reg models token order is cls/reg tokens first, patch tokens at tail.
                return t[:, n - expected_patches :, :]
            if best is None or n > best.shape[1]:
                best = t
    if best is None:
        raise RuntimeError("Could not locate patch-token tensor in model outputs.")
    if best.shape[1] < expected_patches:
        raise RuntimeError(
            f"Found token tensor with only {best.shape[1]} tokens, expected {expected_patches} patches."
        )
    return best[:, best.shape[1] - expected_patches :, :]


def _get_all_layer_patch_tokens(model: torch.nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    expected = int((x.shape[-2] // PATCH_SIZE) * (x.shape[-1] // PATCH_SIZE))
    calls: Sequence[Dict[str, Any]] = (
        {"n": 12, "return_class_token": True},
        {"n": 12, "return_class_token": False},
        {"n": 12},
        {"n": list(range(12)), "return_class_token": True},
        {"n": list(range(12)), "return_class_token": False},
        {"n": list(range(12))},
    )
    last_err: Exception | None = None
    outputs: Any = None
    for kwargs in calls:
        try:
            outputs = model.get_intermediate_layers(x, **kwargs)
            last_err = None
            break
        except Exception as exc:  # pragma: no cover - depends on installed dino API variant
            last_err = exc
    if last_err is not None:
        raise RuntimeError(f"Failed to fetch intermediate layers from dinov2 model: {last_err}")

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    tokens: List[torch.Tensor] = []
    for out in outputs:
        pt = _extract_patch_tokens_from_obj(out, expected)
        tokens.append(pt.detach().cpu())
    if len(tokens) != 12:
        raise RuntimeError(f"Expected 12 layer outputs, got {len(tokens)}.")
    return tokens


def _norm01(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def _heatmap_unsigned(m: np.ndarray) -> np.ndarray:
    u = _norm01(m.astype(np.float32))
    r = (u * 255.0).astype(np.uint8)
    g = (u * 220.0).astype(np.uint8)
    b = ((1.0 - u) * 70.0).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _heatmap_signed(sim: np.ndarray) -> np.ndarray:
    u = np.clip((sim.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    lo = np.array([20.0, 90.0, 220.0], dtype=np.float32)   # blue
    mid = np.array([245.0, 245.0, 245.0], dtype=np.float32)  # white
    hi = np.array([220.0, 60.0, 40.0], dtype=np.float32)   # red
    rgb = np.zeros((*u.shape, 3), dtype=np.float32)
    left = u <= 0.5
    right = ~left
    ul = (u[left] * 2.0)[:, None]
    ur = ((u[right] - 0.5) * 2.0)[:, None]
    if ul.size > 0:
        rgb[left] = lo[None, :] * (1.0 - ul) + mid[None, :] * ul
    if ur.size > 0:
        rgb[right] = mid[None, :] * (1.0 - ur) + hi[None, :] * ur
    return rgb.clip(0, 255).astype(np.uint8)


def _upsample_map(m: np.ndarray, out_wh: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(m).resize(out_wh, Image.BILINEAR))


def _draw_patch_grid(img_np: np.ndarray, patch: int = PATCH_SIZE) -> np.ndarray:
    img = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(img)
    w, h = img.size
    grid_color = (255, 255, 255)
    for x in range(0, w, patch):
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
    for y in range(0, h, patch):
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)
    return np.asarray(img)


def _draw_selection(
    img_np: np.ndarray,
    mode: str,
    px: int,
    py: int,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
) -> np.ndarray:
    img = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(img)
    if mode == "point":
        r = 6
        draw.ellipse([(px - r, py - r), (px + r, py + r)], outline=(255, 0, 0), width=2)
    else:
        x1, x2 = sorted((int(bx1), int(bx2)))
        y1, y2 = sorted((int(by1), int(by2)))
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
    return np.asarray(img)


def _token_norm_gallery(tokens: List[torch.Tensor], gh: int, gw: int, out_wh: Tuple[int, int]) -> List[Tuple[np.ndarray, str]]:
    items: List[Tuple[np.ndarray, str]] = []
    for i, t in enumerate(tokens, start=1):
        x = t[0].numpy().reshape(gh, gw, -1)
        n = np.linalg.norm(x, axis=-1)
        hm = _heatmap_unsigned(n)
        up = _upsample_map(hm, out_wh)
        items.append((up, f"Layer {i}: patch-token L2 norm map"))
    return items


def _patch_centers(gh: int, gw: int, patch: int = PATCH_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    ys = (np.arange(gh, dtype=np.float32) + 0.5) * float(patch)
    xs = (np.arange(gw, dtype=np.float32) + 0.5) * float(patch)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy, xx


def _similarity_maps(
    tokens: List[torch.Tensor],
    gh: int,
    gw: int,
    out_wh: Tuple[int, int],
    mode: str,
    px: int,
    py: int,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
) -> Tuple[str, List[Tuple[np.ndarray, str]]]:
    patch_x = int(np.clip(px // PATCH_SIZE, 0, gw - 1))
    patch_y = int(np.clip(py // PATCH_SIZE, 0, gh - 1))
    idx = patch_y * gw + patch_x

    yy, xx = _patch_centers(gh, gw, patch=PATCH_SIZE)
    x1, x2 = sorted((int(bx1), int(bx2)))
    y1, y2 = sorted((int(by1), int(by2)))
    mask = (xx >= float(x1)) & (xx <= float(x2)) & (yy >= float(y1)) & (yy <= float(y2))
    mask_flat = mask.reshape(-1)

    maps: List[Tuple[np.ndarray, str]] = []
    for i, t in enumerate(tokens, start=1):
        z = t[0].float()  # [N, C]
        if mode == "point":
            ref = z[idx : idx + 1]
            ref_label = f"point pixel=({px},{py}), patch=({patch_x},{patch_y})"
        else:
            if int(mask_flat.sum()) == 0:
                raise gr.Error("BBox did not include any patch centers. Increase bbox size.")
            ref = z[mask_flat].mean(dim=0, keepdim=True)
            ref_label = f"bbox pixels=({x1},{y1})-({x2},{y2}), selected_patches={int(mask_flat.sum())}"

        sim = F.cosine_similarity(z, ref.expand_as(z), dim=1).numpy().reshape(gh, gw)
        hm = _heatmap_signed(sim)
        up = _upsample_map(hm, out_wh)
        maps.append((up, f"Layer {i}: cosine similarity"))

    expl = (
        f"Reference: {ref_label}\n"
        "For each layer, similarity is computed between selected reference embedding and every patch token."
    )
    return expl, maps


def _default_bbox(w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(w * 0.30)
    y1 = int(h * 0.30)
    x2 = int(w * 0.70)
    y2 = int(h * 0.70)
    return x1, y1, x2, y2


def extract_tokens(image_path: str):
    if image_path is None or str(image_path).strip() == "":
        raise gr.Error("Please provide an image path.")

    p = Path(image_path).expanduser()
    if not p.is_file():
        raise gr.Error(f"Image path does not exist: {p}")

    try:
        model = _load_dino()
    except Exception as exc:
        raise gr.Error(f"Failed to load dinov2_vits14_reg: {exc}") from exc

    try:
        image = Image.open(p).convert("RGB")
    except Exception as exc:
        raise gr.Error(f"Could not read image: {exc}") from exc

    proc = _resize_to_patch_grid(image)
    x = _to_tensor(proc)

    try:
        tokens = _get_all_layer_patch_tokens(model, x)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    w, h = proc.size
    gw = w // PATCH_SIZE
    gh = h // PATCH_SIZE
    proc_np = np.asarray(proc)
    overlay_grid = _draw_patch_grid(proc_np)
    norm_gallery = _token_norm_gallery(tokens, gh, gw, out_wh=(w, h))

    px, py = w // 2, h // 2
    bx1, by1, bx2, by2 = _default_bbox(w, h)
    sel_preview = _draw_selection(overlay_grid, "point", px, py, bx1, by1, bx2, by2)

    state = {
        "w": w,
        "h": h,
        "gw": gw,
        "gh": gh,
        "image": proc_np,
        "grid_image": overlay_grid,
        "tokens": tokens,
    }
    summary = (
        "Model: `dinov2_vits14_reg` (12 layers)\n"
        f"Input image: `{p}`\n"
        f"Processed size: `{w}x{h}` (`{gw}x{gh}` patch grid, patch size={PATCH_SIZE})\n"
        "Patch tokens extracted from all layers. You can now choose point/bbox and compute cosine similarity maps."
    )
    selection_info = f"Default point selected at pixel=({px},{py})."
    return (
        summary,
        overlay_grid,
        norm_gallery,
        state,
        px,
        py,
        bx1,
        by1,
        bx2,
        by2,
        sel_preview,
        selection_info,
    )


def on_click_set_point(
    state: Dict[str, Any] | None,
    mode: str,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
    evt: gr.SelectData,
):
    if state is None:
        raise gr.Error("Load tokens first.")
    if not isinstance(evt.index, (tuple, list)) or len(evt.index) < 2:
        raise gr.Error("Click event did not provide image coordinates.")
    x = int(np.clip(evt.index[0], 0, int(state["w"]) - 1))
    y = int(np.clip(evt.index[1], 0, int(state["h"]) - 1))
    preview = _draw_selection(state["grid_image"], mode, x, y, int(bx1), int(by1), int(bx2), int(by2))
    return x, y, preview, f"Point set by click at pixel=({x},{y})."


def update_selection_preview(
    state: Dict[str, Any] | None,
    mode: str,
    px: int,
    py: int,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
):
    if state is None:
        return None, "Load tokens first."
    x = int(np.clip(px, 0, int(state["w"]) - 1))
    y = int(np.clip(py, 0, int(state["h"]) - 1))
    preview = _draw_selection(state["grid_image"], mode, x, y, int(bx1), int(by1), int(bx2), int(by2))
    info = (
        f"Mode=point, pixel=({x},{y})."
        if mode == "point"
        else f"Mode=bbox, box=({int(bx1)},{int(by1)})-({int(bx2)},{int(by2)})."
    )
    return preview, info


def compute_similarity(
    state: Dict[str, Any] | None,
    mode: str,
    px: int,
    py: int,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
):
    if state is None:
        raise gr.Error("Load tokens first.")
    w = int(state["w"])
    h = int(state["h"])
    x = int(np.clip(px, 0, w - 1))
    y = int(np.clip(py, 0, h - 1))
    x1 = int(np.clip(bx1, 0, w - 1))
    y1 = int(np.clip(by1, 0, h - 1))
    x2 = int(np.clip(bx2, 0, w - 1))
    y2 = int(np.clip(by2, 0, h - 1))
    expl, sim_gallery = _similarity_maps(
        tokens=state["tokens"],
        gh=int(state["gh"]),
        gw=int(state["gw"]),
        out_wh=(w, h),
        mode=mode,
        px=x,
        py=y,
        bx1=x1,
        by1=y1,
        bx2=x2,
        by2=y2,
    )
    preview = _draw_selection(state["grid_image"], mode, x, y, x1, y1, x2, y2)
    return expl, sim_gallery, preview


def make_app() -> gr.Blocks:
    with gr.Blocks(title="ViT Explainer - DINOv2 ViT-S/14 Reg") as app:
        gr.Markdown(
            """
            # ViT Explainer (DINOv2)
            Model: **`dinov2_vits14_reg`**  
            Workflow:
            1. Load image + extract patch tokens from all 12 layers.
            2. Inspect per-layer patch-token maps.
            3. Choose **point** or **bbox**.
            4. Compute cosine similarity maps vs all patches for every layer.
            """
        )

        state = gr.State(value=None)

        with gr.Row():
            image_path = gr.Textbox(label="Image Path", placeholder="/path/to/image.jpg", lines=1)
            load_btn = gr.Button("Extract Patch Tokens", variant="primary")

        summary = gr.Markdown(label="Summary")
        image_with_grid = gr.Image(label="Model Input + Patch Grid (click to set point)", type="numpy", interactive=True)
        token_gallery = gr.Gallery(label="All 12 Layers: Patch-Token Maps", columns=4, rows=3, height=520)

        with gr.Row():
            mode = gr.Radio(choices=["point", "bbox"], value="point", label="Reference Selection Mode")
            compute_btn = gr.Button("Compute Cosine Similarity (12 Layers)", variant="primary")

        with gr.Row():
            px = gr.Number(value=0, label="Point X (pixel)", precision=0)
            py = gr.Number(value=0, label="Point Y (pixel)", precision=0)
            bx1 = gr.Number(value=0, label="BBox X1", precision=0)
            by1 = gr.Number(value=0, label="BBox Y1", precision=0)
            bx2 = gr.Number(value=0, label="BBox X2", precision=0)
            by2 = gr.Number(value=0, label="BBox Y2", precision=0)

        selection_preview = gr.Image(label="Current Selection Overlay", type="numpy")
        selection_info = gr.Markdown(label="Selection Info")
        sim_expl = gr.Markdown(label="Cosine Similarity Explanation")
        sim_gallery = gr.Gallery(label="All 12 Layers: Cosine Similarity Maps", columns=4, rows=3, height=520)

        load_btn.click(
            fn=extract_tokens,
            inputs=[image_path],
            outputs=[
                summary,
                image_with_grid,
                token_gallery,
                state,
                px,
                py,
                bx1,
                by1,
                bx2,
                by2,
                selection_preview,
                selection_info,
            ],
        )

        image_with_grid.select(
            fn=on_click_set_point,
            inputs=[state, mode, bx1, by1, bx2, by2],
            outputs=[px, py, selection_preview, selection_info],
        )

        for inp in [mode, px, py, bx1, by1, bx2, by2]:
            inp.change(
                fn=update_selection_preview,
                inputs=[state, mode, px, py, bx1, by1, bx2, by2],
                outputs=[selection_preview, selection_info],
            )

        compute_btn.click(
            fn=compute_similarity,
            inputs=[state, mode, px, py, bx1, by1, bx2, by2],
            outputs=[sim_expl, sim_gallery, selection_preview],
        )

    return app


def main() -> None:
    app = make_app()
    app.queue().launch(server_name="127.0.0.1", server_port=7862, share=False)


if __name__ == "__main__":
    main()
