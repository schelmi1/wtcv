#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Gradio is required. Install with: pip install gradio") from exc

try:
    from torchvision.models import ResNet18_Weights, resnet18
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required. Install with: pip install torchvision") from exc


_MODEL: nn.Module | None = None
_WEIGHTS: ResNet18_Weights | None = None


def _load_resnet18() -> Tuple[nn.Module, ResNet18_Weights]:
    global _MODEL, _WEIGHTS
    if _MODEL is not None and _WEIGHTS is not None:
        return _MODEL, _WEIGHTS
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    _MODEL = model
    _WEIGHTS = weights
    return model, weights


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def _make_grid(images: List[np.ndarray], cols: int = 8, pad: int = 2) -> np.ndarray:
    if not images:
        return np.zeros((16, 16, 3), dtype=np.uint8)
    h, w = images[0].shape[:2]
    ch = 1 if images[0].ndim == 2 else images[0].shape[2]
    rows = (len(images) + cols - 1) // cols
    grid_h = rows * h + (rows - 1) * pad
    grid_w = cols * w + (cols - 1) * pad
    if ch == 1:
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    else:
        grid = np.zeros((grid_h, grid_w, ch), dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = img
    if grid.ndim == 2:
        grid = np.stack([grid, grid, grid], axis=-1)
    return grid


def _conv1_kernels_image(model: nn.Module) -> np.ndarray:
    w = model.conv1.weight.detach().cpu().numpy()  # [64, 3, 7, 7]
    imgs: List[np.ndarray] = []
    for i in range(w.shape[0]):
        k = w[i]  # [3,7,7]
        k = np.transpose(k, (1, 2, 0))  # [7,7,3]
        k = _normalize_map(k)
        up = np.array(Image.fromarray(k).resize((56, 56), Image.NEAREST))
        imgs.append(up)
    return _make_grid(imgs, cols=8, pad=2)


def _denorm_image(x: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    z = (x[0] * s + m).clamp(0.0, 1.0)
    arr = (z.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def _shape_tuple(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(v) for v in t.shape)


def _spatial(t: torch.Tensor) -> Tuple[int, int]:
    if t.ndim < 4:
        return (-1, -1)
    return int(t.shape[-2]), int(t.shape[-1])


def _add_shape_row(
    rows: List[Dict[str, object]],
    step: str,
    operation: str,
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    kernel: str = "",
    stride: str = "",
    padding: str = "",
    note: str = "",
) -> None:
    h_in, w_in = _spatial(x_in)
    h_out, w_out = _spatial(x_out)
    spatial_change = "-"
    if h_in > 0 and h_out > 0:
        spatial_change = f"{h_in}x{w_in} -> {h_out}x{w_out}"
    rows.append(
        {
            "step": step,
            "operation": operation,
            "in_shape": str(_shape_tuple(x_in)),
            "out_shape": str(_shape_tuple(x_out)),
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "spatial_change": spatial_change,
            "note": note,
        }
    )


def _forward_with_explanations(
    model: nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], pd.DataFrame, pd.DataFrame, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    acts: Dict[str, torch.Tensor] = {}
    shape_rows: List[Dict[str, object]] = []
    residual_rows: List[Dict[str, object]] = []
    residual_compare: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    with torch.no_grad():
        z0 = model.conv1(x)
        _add_shape_row(
            shape_rows,
            "stem.conv1",
            "Conv2d",
            x,
            z0,
            kernel="7x7",
            stride="2",
            padding="3",
            note="64 kernels, early edge/texture extraction",
        )
        acts["stem.conv1"] = z0.detach().cpu()

        z1 = model.bn1(z0)
        z2 = model.relu(z1)
        _add_shape_row(
            shape_rows,
            "stem.bn1_relu",
            "BatchNorm + ReLU",
            z0,
            z2,
            note="normalize channels then non-linearity",
        )

        z3 = model.maxpool(z2)
        _add_shape_row(
            shape_rows,
            "stem.maxpool",
            "MaxPool2d",
            z2,
            z3,
            kernel="3x3",
            stride="2",
            padding="1",
            note="keeps strongest local responses",
        )
        acts["stem.maxpool"] = z3.detach().cpu()

        x_stage = z3
        for s_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], start=1):
            for b_idx, block in enumerate(layer, start=1):
                block_name = f"stage{s_idx}.block{b_idx}"
                x_in = x_stage
                identity = x_stage

                b1 = block.conv1(x_stage)
                _add_shape_row(
                    shape_rows,
                    f"{block_name}.conv1",
                    "Conv2d",
                    x_stage,
                    b1,
                    kernel="3x3",
                    stride=str(block.conv1.stride[0]),
                    padding=str(block.conv1.padding[0]),
                    note="first conv in residual branch",
                )
                b1 = block.bn1(b1)
                b1 = block.relu(b1)

                b2 = block.conv2(b1)
                _add_shape_row(
                    shape_rows,
                    f"{block_name}.conv2",
                    "Conv2d",
                    b1,
                    b2,
                    kernel="3x3",
                    stride=str(block.conv2.stride[0]),
                    padding=str(block.conv2.padding[0]),
                    note="second conv in residual branch",
                )
                b2 = block.bn2(b2)

                used_proj = block.downsample is not None
                if used_proj:
                    proj = block.downsample(x_stage)
                    _add_shape_row(
                        shape_rows,
                        f"{block_name}.skip_proj",
                        "Skip Projection (1x1 conv + bn)",
                        x_stage,
                        proj,
                        kernel="1x1",
                        stride=str(block.downsample[0].stride[0]),
                        padding="0",
                        note="match shape/channels for skip path",
                    )
                    identity = proj

                pre_add = b2
                added = pre_add + identity
                post_add = block.relu(added)
                _add_shape_row(
                    shape_rows,
                    f"{block_name}.add_relu",
                    "Residual Add + ReLU",
                    pre_add,
                    post_add,
                    note="output = ReLU(F(x) + skip(x))",
                )

                residual_rows.append(
                    {
                        "block": block_name,
                        "projection_skip": bool(used_proj),
                        "input_shape": str(_shape_tuple(x_in)),
                        "residual_shape": str(_shape_tuple(pre_add)),
                        "skip_shape": str(_shape_tuple(identity)),
                        "output_shape": str(_shape_tuple(post_add)),
                        "pre_add_mean": float(pre_add.mean().item()),
                        "pre_add_std": float(pre_add.std(unbiased=False).item()),
                        "post_add_mean": float(post_add.mean().item()),
                        "post_add_std": float(post_add.std(unbiased=False).item()),
                    }
                )

                if b_idx == 1:
                    residual_compare[f"stage{s_idx}.block1"] = (
                        pre_add.detach().cpu(),
                        post_add.detach().cpu(),
                    )

                acts[block_name] = post_add.detach().cpu()
                x_stage = post_add

            acts[f"stage{s_idx}.out"] = x_stage.detach().cpu()

        avg = model.avgpool(x_stage)
        _add_shape_row(
            shape_rows,
            "head.avgpool",
            "AdaptiveAvgPool2d",
            x_stage,
            avg,
            note="global spatial averaging to 1x1",
        )
        acts["head.avgpool"] = avg.detach().cpu()

        flat = torch.flatten(avg, 1)
        _add_shape_row(
            shape_rows,
            "head.flatten",
            "Flatten",
            avg,
            flat,
            note="convert [N,C,1,1] -> [N,C]",
        )
        logits = model.fc(flat)
        _add_shape_row(
            shape_rows,
            "head.fc",
            "Linear",
            flat,
            logits,
            note="1000 ImageNet logits (unnormalized scores)",
        )

    return (
        logits.detach().cpu(),
        acts,
        pd.DataFrame(shape_rows),
        pd.DataFrame(residual_rows),
        residual_compare,
    )


def _activation_gallery_items(acts: Dict[str, torch.Tensor], max_layers: int = 16) -> List[Tuple[np.ndarray, str]]:
    preferred = [
        "stem.conv1",
        "stem.maxpool",
        "stage1.block1",
        "stage1.block2",
        "stage2.block1",
        "stage2.block2",
        "stage3.block1",
        "stage3.block2",
        "stage4.block1",
        "stage4.block2",
        "stage1.out",
        "stage2.out",
        "stage3.out",
        "stage4.out",
    ]
    items: List[Tuple[np.ndarray, str]] = []
    for name in preferred[:max_layers]:
        if name not in acts:
            continue
        t = acts[name]
        if t.ndim != 4 or t.shape[0] == 0:
            continue
        feat = t[0].numpy()  # [C,H,W]
        if feat.ndim != 3 or feat.shape[0] < 1:
            continue

        stds = feat.reshape(feat.shape[0], -1).std(axis=1)
        topk = int(min(9, feat.shape[0]))
        top_idx = np.argsort(-stds)[:topk]
        maps = [_normalize_map(feat[c]) for c in top_idx]
        vis = _make_grid(maps, cols=3, pad=2)
        items.append((vis, f"{name} | top-{topk} channels by std"))
    return items


def _stage_stats(acts: Dict[str, torch.Tensor]) -> pd.DataFrame:
    rows = []
    order = [
        "stem.conv1",
        "stem.maxpool",
        "stage1.block1",
        "stage1.block2",
        "stage1.out",
        "stage2.block1",
        "stage2.block2",
        "stage2.out",
        "stage3.block1",
        "stage3.block2",
        "stage3.out",
        "stage4.block1",
        "stage4.block2",
        "stage4.out",
        "head.avgpool",
    ]
    for name in order:
        if name not in acts:
            continue
        t = acts[name].float()
        rows.append(
            {
                "layer": name,
                "shape": str(tuple(int(v) for v in t.shape)),
                "min": float(t.min().item()),
                "max": float(t.max().item()),
                "mean": float(t.mean().item()),
                "std": float(t.std(unbiased=False).item()),
            }
        )
    return pd.DataFrame(rows)


def _top5(logits: torch.Tensor, categories: List[str]) -> pd.DataFrame:
    probs = torch.softmax(logits[0], dim=0)
    vals, idxs = torch.topk(probs, k=5)
    rows = []
    for rank, (v, i) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        cls = categories[i] if 0 <= i < len(categories) else f"class_{i}"
        rows.append(
            {
                "rank": rank,
                "class_idx": int(i),
                "class_name": cls,
                "logit": float(logits[0, i].item()),
                "prob": float(v),
            }
        )
    return pd.DataFrame(rows)


def _conv_math_example(
    model: nn.Module,
    x: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> Tuple[str, pd.DataFrame, np.ndarray]:
    with torch.no_grad():
        y = model.conv1(x)  # [1,64,H',W']
    out_h = int(y.shape[-2])
    out_w = int(y.shape[-1])
    oy = out_h // 2
    ox = out_w // 2
    stride = int(model.conv1.stride[0])
    pad = int(model.conv1.padding[0])
    kernel_size = int(model.conv1.kernel_size[0])

    x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
    sy = oy * stride
    sx = ox * stride
    patch = x_pad[0, :, sy : sy + kernel_size, sx : sx + kernel_size]  # [3,7,7]
    kernel = model.conv1.weight[0]  # [3,7,7]
    products = patch * kernel
    summed = float(products.sum().item())
    actual = float(y[0, 0, oy, ox].item())

    m = torch.tensor(mean, dtype=patch.dtype, device=patch.device).view(3, 1, 1)
    s = torch.tensor(std, dtype=patch.dtype, device=patch.device).view(3, 1, 1)
    patch_rgb = ((patch * s + m).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    patch_vis = np.array(Image.fromarray(patch_rgb).resize((140, 140), Image.NEAREST))

    contrib = products.detach().cpu().numpy().reshape(-1)
    p_vals = patch.detach().cpu().numpy().reshape(-1)
    k_vals = kernel.detach().cpu().numpy().reshape(-1)
    idx = np.argsort(-np.abs(contrib))[:12]
    rows = []
    for j in idx.tolist():
        c = j // (kernel_size * kernel_size)
        rem = j % (kernel_size * kernel_size)
        ry = rem // kernel_size
        rx = rem % kernel_size
        rows.append(
            {
                "c": int(c),
                "y": int(ry),
                "x": int(rx),
                "patch_value": float(p_vals[j]),
                "kernel_weight": float(k_vals[j]),
                "product": float(contrib[j]),
            }
        )
    contrib_df = pd.DataFrame(rows)

    md = (
        "Single-location conv example (conv1, filter=0):\n"
        f"- output position: `(y={oy}, x={ox})`\n"
        f"- patch size: `{kernel_size}x{kernel_size}x3`, stride=`{stride}`, padding=`{pad}`\n"
        f"- dot product: `sum(patch * kernel) = {summed:.6f}`\n"
        f"- model output at same location: `{actual:.6f}`\n"
        f"- absolute difference: `{abs(summed - actual):.6e}`"
    )
    return md, contrib_df, patch_vis


def _residual_compare_gallery(
    residual_compare: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> List[Tuple[np.ndarray, str]]:
    items: List[Tuple[np.ndarray, str]] = []
    for name in ["stage1.block1", "stage2.block1", "stage3.block1", "stage4.block1"]:
        if name not in residual_compare:
            continue
        pre, post = residual_compare[name]
        f_pre = pre[0].numpy()
        f_post = post[0].numpy()
        stds = f_pre.reshape(f_pre.shape[0], -1).std(axis=1)
        topk = int(min(4, f_pre.shape[0]))
        top_idx = np.argsort(-stds)[:topk]
        pre_maps = [_normalize_map(f_pre[c]) for c in top_idx]
        post_maps = [_normalize_map(f_post[c]) for c in top_idx]
        g1 = _make_grid(pre_maps, cols=2, pad=2)
        g2 = _make_grid(post_maps, cols=2, pad=2)
        spacer = np.full((g1.shape[0], 12, 3), 20, dtype=np.uint8)
        side = np.concatenate([g1, spacer, g2], axis=1)
        items.append((side, f"{name}: left=pre-add F(x), right=post-add ReLU(F(x)+skip)"))
    return items


def _downsample_events(shape_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in shape_df.iterrows():
        sc = str(r["spatial_change"])
        if "->" not in sc:
            continue
        left, right = [s.strip() for s in sc.split("->", 1)]
        if left != right:
            rows.append(
                {
                    "step": r["step"],
                    "operation": r["operation"],
                    "spatial_change": sc,
                    "why": "stride/pooling reduced resolution while increasing abstraction",
                }
            )
    return pd.DataFrame(rows)


def _pooling_explainer(acts: Dict[str, torch.Tensor]) -> str:
    maxpool = acts["stem.maxpool"]
    stage4 = acts["stage4.out"]
    avg = acts["head.avgpool"]
    c0_mean = float(stage4[0, 0].mean().item())
    c0_avg = float(avg[0, 0, 0, 0].item())
    return (
        "Pooling behavior:\n"
        f"- MaxPool: `stem.maxpool` keeps strongest local responses. Shape `{tuple(maxpool.shape)}`.\n"
        f"- GlobalAvgPool: `head.avgpool` averages each channel map to one value. "
        f"Shape `{tuple(stage4.shape)}` -> `{tuple(avg.shape)}`.\n"
        f"- Example channel-0: mean(stage4[0,0,:,:]) = `{c0_mean:.6f}`, avgpool output = `{c0_avg:.6f}`."
    )


def _receptive_field_df(model: nn.Module) -> pd.DataFrame:
    seq: List[Tuple[str, int, int]] = [
        ("stem.conv1", 7, int(model.conv1.stride[0])),
        ("stem.maxpool", 3, int(model.maxpool.stride)),
        ("stage1.block1.conv1", 3, int(model.layer1[0].conv1.stride[0])),
        ("stage1.block1.conv2", 3, int(model.layer1[0].conv2.stride[0])),
        ("stage1.block2.conv1", 3, int(model.layer1[1].conv1.stride[0])),
        ("stage1.block2.conv2", 3, int(model.layer1[1].conv2.stride[0])),
        ("stage2.block1.conv1", 3, int(model.layer2[0].conv1.stride[0])),
        ("stage2.block1.conv2", 3, int(model.layer2[0].conv2.stride[0])),
        ("stage2.block2.conv1", 3, int(model.layer2[1].conv1.stride[0])),
        ("stage2.block2.conv2", 3, int(model.layer2[1].conv2.stride[0])),
        ("stage3.block1.conv1", 3, int(model.layer3[0].conv1.stride[0])),
        ("stage3.block1.conv2", 3, int(model.layer3[0].conv2.stride[0])),
        ("stage3.block2.conv1", 3, int(model.layer3[1].conv1.stride[0])),
        ("stage3.block2.conv2", 3, int(model.layer3[1].conv2.stride[0])),
        ("stage4.block1.conv1", 3, int(model.layer4[0].conv1.stride[0])),
        ("stage4.block1.conv2", 3, int(model.layer4[0].conv2.stride[0])),
        ("stage4.block2.conv1", 3, int(model.layer4[1].conv1.stride[0])),
        ("stage4.block2.conv2", 3, int(model.layer4[1].conv2.stride[0])),
    ]

    rf = 1
    jump = 1
    rows = []
    for name, k, s in seq:
        rf = rf + (k - 1) * jump
        jump = jump * s
        rows.append({"layer": name, "kernel": k, "stride": s, "effective_stride_to_input": jump, "receptive_field_px": rf})
    return pd.DataFrame(rows)


def _preprocess_explainer(weights: ResNet18_Weights, image: Image.Image, x: torch.Tensor) -> str:
    t = weights.transforms()
    crop_size = getattr(t, "crop_size", None)
    resize_size = getattr(t, "resize_size", None)
    interpolation = getattr(t, "interpolation", None)
    mean = getattr(t, "mean", None)
    std = getattr(t, "std", None)
    raw_w, raw_h = image.size
    proc_h = int(x.shape[-2])
    proc_w = int(x.shape[-1])
    return (
        "Input preprocessing (must match pretrained weights):\n"
        f"- Original image size: `{raw_w}x{raw_h}`\n"
        f"- Transform resize target: `{resize_size}` then center-crop: `{crop_size}`\n"
        f"- Interpolation: `{interpolation}`\n"
        f"- Normalize by ImageNet mean/std:\n"
        f"  - mean={mean}\n"
        f"  - std={std}\n"
        f"- Model input tensor shape after batch add: `{tuple(x.shape)}` ([N,C,H,W], here `{proc_h}x{proc_w}` spatial)."
    )


def run_explainer(image_path: str):
    if image_path is None or str(image_path).strip() == "":
        raise gr.Error("Please provide an image path.")

    p = Path(image_path).expanduser()
    if not p.is_file():
        raise gr.Error(f"Image path does not exist: {p}")

    try:
        model, weights = _load_resnet18()
    except Exception as exc:
        raise gr.Error(f"Failed to load pretrained resnet18: {exc}") from exc

    try:
        image = Image.open(p).convert("RGB")
    except Exception as exc:
        raise gr.Error(f"Could not read image: {exc}") from exc

    preprocess = weights.transforms()
    x = preprocess(image).unsqueeze(0)
    logits, acts, shape_df, residual_df, residual_compare = _forward_with_explanations(model, x)

    input_np = np.array(image)
    mean = list(getattr(preprocess, "mean", [0.485, 0.456, 0.406]))
    std = list(getattr(preprocess, "std", [0.229, 0.224, 0.225]))
    preprocessed_preview = _denorm_image(x, mean, std)
    kernels = _conv1_kernels_image(model)
    stats_df = _stage_stats(acts)
    classes = list(weights.meta.get("categories", []))
    top5_df = _top5(logits, classes)
    gallery = _activation_gallery_items(acts)
    conv_md, conv_df, patch_img = _conv_math_example(model, x, mean, std)
    residual_gallery = _residual_compare_gallery(residual_compare)
    downsample_df = _downsample_events(shape_df)
    pooling_md = _pooling_explainer(acts)
    rf_df = _receptive_field_df(model)
    preprocess_md = _preprocess_explainer(weights, image, x)

    summary = (
        "Model: torchvision `resnet18` (pretrained ImageNet-1K)\n"
        f"Input file: `{p}`\n"
        f"Input size: `{image.size[0]}x{image.size[1]}`\n"
        "Forward-pass explainer includes preprocessing, conv math at one location, shape flow, "
        "downsampling, residual internals, pooling, receptive field growth, and logits vs probabilities."
    )
    channel_caveat_md = (
        "Channel interpretation caveat:\n"
        "- A single feature map channel is not a direct object mask.\n"
        "- CNN features are distributed; semantics emerge across many channels and layers.\n"
        "- High activation means strong response to learned pattern, not guaranteed object presence."
    )
    residual_md = (
        "Residual block internals:\n"
        "- Each block computes `F(x)` with two 3x3 convolutions.\n"
        "- Output is `ReLU(F(x) + skip(x))`.\n"
        "- In blocks with stride-2/channel change (stage2/3/4 block1), skip uses a 1x1 projection."
    )
    logits_md = (
        "Logits vs probabilities:\n"
        "- `fc` outputs raw logits (unnormalized scores).\n"
        "- `softmax(logits)` converts scores to probabilities that sum to 1."
    )
    return (
        summary,
        preprocess_md,
        input_np,
        preprocessed_preview,
        kernels,
        conv_md,
        patch_img,
        conv_df,
        shape_df,
        downsample_df,
        residual_md,
        residual_df,
        residual_gallery,
        stats_df,
        pooling_md,
        rf_df,
        logits_md,
        top5_df,
        channel_caveat_md,
        gallery,
    )


def make_app() -> gr.Blocks:
    with gr.Blocks(title="CNN Explainer Mockup - ResNet18") as app:
        gr.Markdown(
            """
            # CNN Explainer (Mockup)
            Fixed backbone: **pretrained ResNet18**.
            Provide an image path, run one forward pass, then inspect kernels, pooling, residual blocks, and stage activations.
            """
        )
        with gr.Row():
            image_path = gr.Textbox(
                label="Image Path",
                placeholder="/path/to/image.jpg",
                lines=1,
            )
            run_btn = gr.Button("Run Forward Pass", variant="primary")

        summary = gr.Markdown(label="Summary")
        preprocess_md = gr.Markdown(label="1) Input Preprocessing")
        with gr.Row():
            input_img = gr.Image(label="Input Image", type="numpy")
            preprocessed_preview = gr.Image(label="Preprocessed Tensor (denormalized preview)", type="numpy")

        kernel_img = gr.Image(label="Conv1 Learned Kernels (64 filters)", type="numpy")
        conv_md = gr.Markdown(label="2) Conv Math at One Location")
        with gr.Row():
            patch_img = gr.Image(label="Selected 7x7 RGB Patch (upsampled)", type="numpy")
            conv_df = gr.Dataframe(label="Top Contribution Terms (patch * kernel)", interactive=False)

        shape_df = gr.Dataframe(label="3) Shape Flow Through Forward Pass", interactive=False)
        downsample_df = gr.Dataframe(label="4) Downsampling Events", interactive=False)

        residual_md = gr.Markdown(label="5) Residual Block Internals + 6) Pre/Post Add")
        residual_df = gr.Dataframe(label="Residual Block Stats", interactive=False)
        residual_gallery = gr.Gallery(
            label="Pre-add vs Post-add Activations (left vs right)",
            columns=2,
            rows=2,
            height=360,
        )

        stats_df = gr.Dataframe(label="Layer Activation Stats", interactive=False)
        pooling_md = gr.Markdown(label="7) Pooling Explanation")
        rf_df = gr.Dataframe(label="8) Receptive Field Growth (approximate)", interactive=False)

        logits_md = gr.Markdown(label="9) Logits vs Probabilities")
        top5_df = gr.Dataframe(label="Top-5 Predictions", interactive=False)

        channel_caveat_md = gr.Markdown(label="10) Channel Interpretation Caveat")
        gallery = gr.Gallery(label="Activation Map Grids", columns=4, rows=4, height=520)

        run_btn.click(
            fn=run_explainer,
            inputs=[image_path],
            outputs=[
                summary,
                preprocess_md,
                input_img,
                preprocessed_preview,
                kernel_img,
                conv_md,
                patch_img,
                conv_df,
                shape_df,
                downsample_df,
                residual_md,
                residual_df,
                residual_gallery,
                stats_df,
                pooling_md,
                rf_df,
                logits_md,
                top5_df,
                channel_caveat_md,
                gallery,
            ],
        )

    return app


def main() -> None:
    app = make_app()
    app.queue().launch(server_name="127.0.0.1", server_port=7861, share=False)


if __name__ == "__main__":
    main()
