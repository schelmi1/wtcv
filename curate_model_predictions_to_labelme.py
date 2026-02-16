#!/usr/bin/env python3
import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF

from models import Stage1SegNet, load_stage1_state_dict_compat
from wtcv_utils.tiling import crop_with_pad, tile_origins


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive model curation to LabelMe pairs")
    ap.add_argument("--input-dir", type=Path, required=True, help="Folder with source images")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("data/labelme_curated_from_model"))

    ap.add_argument("--label", type=str, default="vehicle")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--tile-stride", type=int, default=128)
    ap.add_argument("--seg-out-stride", type=int, default=4)
    ap.add_argument("--pred-threshold", type=float, default=0.5)

    ap.add_argument("--use-tile-cls-gating", action="store_true", default=True)
    ap.add_argument("--no-use-tile-cls-gating", action="store_false", dest="use_tile_cls_gating")
    ap.add_argument("--tile-cls-threshold", type=float, default=0.5)
    ap.add_argument("--tile-cls-mode", type=str, choices=["hard", "multiply"], default="hard")

    ap.add_argument("--min-poly-area", type=float, default=20.0)
    ap.add_argument("--poly-epsilon-frac", type=float, default=0.002)

    ap.add_argument("--max-images", type=int, default=0, help="0 means all")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--save-preview", action="store_true", default=False)

    return ap.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> Tuple[Stage1SegNet, Dict]:
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    except Exception as e:
        warnings.warn(
            "weights_only=True failed; falling back to weights_only=False. "
            f"Use trusted checkpoints only. Error: {e}"
        )
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    fusion_channels = int(ckpt_cfg.get("fusion_channels", 256))
    dino_upsampler = str(ckpt_cfg.get("dino_upsampler_type", "learned"))
    dino_layers = str(ckpt_cfg.get("dino_layers", "last"))
    anyup_q_chunk_size = int(ckpt_cfg.get("anyup_q_chunk_size", 256))
    local_backbone = str(ckpt_cfg.get("local_backbone", "resnet18"))
    head_type = str(ckpt_cfg.get("head_type", "pointwise"))
    use_tile_cls_head = bool(ckpt_cfg.get("use_tile_cls_head", False))
    use_zoom_cls_head = bool(ckpt_cfg.get("use_zoom_cls_head", False))

    model = Stage1SegNet(
        channels=fusion_channels,
        trust_repo=True,
        dino_upsampler_type=dino_upsampler,
        dino_layers=dino_layers,
        anyup_q_chunk_size=anyup_q_chunk_size,
        local_backbone=local_backbone,
        head_type=head_type,
        use_tile_cls_head=use_tile_cls_head,
        use_zoom_cls_head=use_zoom_cls_head,
    ).to(device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_stage1_state_dict_compat(
        model,
        state,
        strict=False,
        interpolate_mismatch=True,
        verbose=True,
    )
    model.eval()

    info = {
        "fusion_channels": fusion_channels,
        "dino_upsampler": dino_upsampler,
        "dino_layers": dino_layers,
        "anyup_q_chunk_size": anyup_q_chunk_size,
        "local_backbone": local_backbone,
        "head_type": head_type,
        "use_tile_cls_head": use_tile_cls_head,
        "use_zoom_cls_head": use_zoom_cls_head,
    }
    return model, info


def infer_prob_map(
    model: Stage1SegNet,
    image_np: np.ndarray,
    tile_size: int,
    stride: int,
    seg_out_stride: int,
    device: torch.device,
    use_tile_cls_gating: bool,
    tile_cls_threshold: float,
    tile_cls_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    img = Image.fromarray(image_np).convert("RGB")
    W, H = img.size

    gh, gw = H // seg_out_stride, W // seg_out_stride
    accum = np.zeros((gh, gw), dtype=np.float32)
    count = np.zeros((gh, gw), dtype=np.float32)
    cls_accum = np.zeros((gh, gw), dtype=np.float32)

    norm = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    tile_cls_probs: List[float] = []
    tile_cls_used = False

    for x0, y0 in tile_origins(W, H, tile_size, stride):
        tile = crop_with_pad(img, x0, y0, tile_size)
        x = norm(TF.to_tensor(tile)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)
            prob = torch.sigmoid(pred["seg_logit"])[0, 0].cpu().numpy()

            tile_cls_prob = 1.0
            if "tile_logit" in pred:
                tile_cls_prob = float(torch.sigmoid(pred["tile_logit"])[0, 0].item())
                tile_cls_used = True
            tile_cls_probs.append(tile_cls_prob)

            if use_tile_cls_gating and tile_cls_used:
                if tile_cls_mode == "hard":
                    if tile_cls_prob < tile_cls_threshold:
                        prob = np.zeros_like(prob, dtype=np.float32)
                else:
                    prob = prob * float(tile_cls_prob)

        th, tw = prob.shape
        gx0, gy0 = x0 // seg_out_stride, y0 // seg_out_stride
        gx1, gy1 = min(gw, gx0 + tw), min(gh, gy0 + th)
        pw = gx1 - gx0
        ph = gy1 - gy0
        if pw <= 0 or ph <= 0:
            continue

        patch = prob[:ph, :pw]
        accum[gy0:gy1, gx0:gx1] += patch
        count[gy0:gy1, gx0:gx1] += 1.0
        cls_accum[gy0:gy1, gx0:gx1] += float(tile_cls_prob)

    prob_lr = np.divide(accum, np.maximum(count, 1e-6))
    cls_lr = np.divide(cls_accum, np.maximum(count, 1e-6))

    stats = {
        "tile_cls_used": float(1.0 if tile_cls_used else 0.0),
        "tile_cls_mean": float(np.mean(tile_cls_probs)) if tile_cls_probs else float("nan"),
        "tile_cls_min": float(np.min(tile_cls_probs)) if tile_cls_probs else float("nan"),
        "tile_cls_max": float(np.max(tile_cls_probs)) if tile_cls_probs else float("nan"),
    }
    return prob_lr, cls_lr, stats


def mask_to_polygons(mask_u8: np.ndarray, min_area: float, epsilon_frac: float) -> List[List[List[float]]]:
    m = (mask_u8 > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[List[List[float]]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, float(epsilon_frac) * float(peri))
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2).astype(float)
        if pts.shape[0] < 3:
            continue
        poly = [[float(x), float(y)] for x, y in pts]
        polys.append(poly)
    return polys


def make_labelme_json(image_name: str, h: int, w: int, polys: List[List[List[float]]], label: str) -> Dict:
    shapes = []
    for poly in polys:
        shapes.append(
            {
                "label": label,
                "points": poly,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )

    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def build_zoom_panel(img_bgr: np.ndarray, pred_mask: np.ndarray, max_items: int = 6) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    panel = np.zeros_like(img_bgr)
    panel[:] = 20

    contours, _ = cv2.findContours((pred_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 1 or bh <= 1:
            continue
        boxes.append((x, y, bw, bh, bw * bh))
    boxes.sort(key=lambda t: t[4], reverse=True)
    boxes = boxes[:max_items]

    rows, cols = 2, 3
    pad = 8
    tile_w = max(1, (w - (cols + 1) * pad) // cols)
    tile_h = max(1, (h - (rows + 1) * pad) // rows)

    for i, b in enumerate(boxes):
        r = i // cols
        c = i % cols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)
        x, y, bw, bh, _ = b

        # Expand crop slightly for context.
        mx = int(max(2, bw * 0.2))
        my = int(max(2, bh * 0.2))
        cx0 = max(0, x - mx)
        cy0 = max(0, y - my)
        cx1 = min(w, x + bw + mx)
        cy1 = min(h, y + bh + my)
        crop = img_bgr[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        panel[y0:y0 + tile_h, x0:x0 + tile_w] = crop
        cv2.rectangle(panel, (x0, y0), (x0 + tile_w, y0 + tile_h), (0, 255, 255), 1)
        cv2.putText(panel, f"#{i+1}", (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(panel, "Zoomed detections", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    return panel


def build_preview(
    img_bgr: np.ndarray,
    prob_full: np.ndarray,
    pred_mask: np.ndarray,
    polys: List[List[List[float]]],
    stats: Dict[str, float],
    idx: int,
    total: int,
    accepted: int,
    rejected: int,
) -> np.ndarray:

    prob_u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_MAGMA)
    heat_overlay = cv2.addWeighted(img_bgr, 0.65, heat, 0.35, 0.0)

    zoom_panel = build_zoom_panel(img_bgr, pred_mask, max_items=6)

    contour_view = img_bgr.copy()
    for poly in polys:
        arr = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(contour_view, [arr], isClosed=True, color=(0, 255, 255), thickness=2)

    top = np.hstack([img_bgr, heat_overlay])
    bot = np.hstack([zoom_panel, contour_view])
    grid = np.vstack([top, bot])

    panel_h, panel_w = grid.shape[:2]
    info_h = 180
    canvas = np.zeros((panel_h + info_h, panel_w, 3), dtype=np.uint8)
    canvas[:panel_h] = grid

    txt = [
        f"image {idx+1}/{total}",
        f"accepted={accepted} rejected={rejected}",
        f"pred_pixels={int(pred_mask.sum())} polys={len(polys)}",
        f"tile_cls_mean={stats.get('tile_cls_mean', float('nan')):.3f}",
        "keys: [a]=accept/save  [d]=reject  [n]=next  [p]=prev  [g]=goto index  [q]=quit",
    ]
    y = panel_h + 30
    for t in txt:
        cv2.putText(canvas, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        y += 32

    if panel_w > 2200:
        scale = 2200.0 / panel_w
        canvas = cv2.resize(canvas, (int(panel_w * scale), int((panel_h + info_h) * scale)), interpolation=cv2.INTER_AREA)

    return canvas


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {args.input_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, m_info = load_model(args.checkpoint, device)
    print("device:", device)
    print("model:", m_info)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    images = [p for p in sorted(args.input_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if args.start_index > 0:
        images = images[args.start_index:]
    if args.max_images > 0:
        images = images[: args.max_images]

    if not images:
        print("No images found.")
        return

    print(f"images_to_process={len(images)}")

    accepted_set = set()
    rejected_set = set()
    cache: Dict[int, Dict] = {}

    win = "model_curation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i = 0
    while 0 <= i < len(images):
        img_path = images[i]

        if i in cache:
            image_rgb = cache[i]["image_rgb"]
            image_bgr = cache[i]["image_bgr"]
            h, w = image_rgb.shape[:2]
            prob_full = cache[i]["prob_full"]
            pred_mask = cache[i]["pred_mask"]
            polys = cache[i]["polys"]
            stats = cache[i]["stats"]
        else:
            image_rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            h, w = image_rgb.shape[:2]
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            prob_lr, cls_lr, stats = infer_prob_map(
                model=model,
                image_np=image_rgb,
                tile_size=args.tile_size,
                stride=args.tile_stride,
                seg_out_stride=args.seg_out_stride,
                device=device,
                use_tile_cls_gating=args.use_tile_cls_gating,
                tile_cls_threshold=args.tile_cls_threshold,
                tile_cls_mode=args.tile_cls_mode,
            )

            prob_full = F.interpolate(
                torch.from_numpy(prob_lr).float().unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            pred_mask = (prob_full >= args.pred_threshold).astype(np.uint8)
            polys = mask_to_polygons(pred_mask, min_area=args.min_poly_area, epsilon_frac=args.poly_epsilon_frac)

            cache[i] = {
                "image_rgb": image_rgb,
                "image_bgr": image_bgr,
                "prob_full": prob_full,
                "pred_mask": pred_mask,
                "polys": polys,
                "stats": stats,
            }

        preview = build_preview(
            image_bgr,
            prob_full,
            pred_mask,
            polys,
            stats,
            i,
            len(images),
            len(accepted_set),
            len(rejected_set),
        )
        cv2.imshow(win, preview)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord("a"), ord("A")):
                out_img = args.output_dir / img_path.name
                out_json = args.output_dir / f"{img_path.stem}.json"

                shutil.copy2(img_path, out_img)
                labelme = make_labelme_json(out_img.name, h, w, polys, args.label)
                out_json.write_text(json.dumps(labelme, ensure_ascii=False, indent=2))

                if args.save_preview:
                    cv2.imwrite(str(args.output_dir / f"{img_path.stem}__preview.jpg"), preview)

                accepted_set.add(i)
                rejected_set.discard(i)
                print(f"[accept] {img_path.name} polygons={len(polys)}")
                i += 1
                break

            if k in (ord("d"), ord("D")):
                rejected_set.add(i)
                accepted_set.discard(i)
                print(f"[reject] {img_path.name}")
                i += 1
                break

            if k in (ord("n"), ord("N"), 83):  # right arrow may map to 83 on some builds
                i += 1
                break

            if k in (ord("p"), ord("P"), 81):  # left arrow may map to 81 on some builds
                i = max(0, i - 1)
                break

            if k in (ord("g"), ord("G")):
                try:
                    raw = input(f"goto index [1..{len(images)}]: ").strip()
                    if raw != "":
                        j = int(raw) - 1
                        if 0 <= j < len(images):
                            i = j
                        else:
                            print(f"index out of range: {raw}")
                except Exception as e:
                    print(f"invalid goto input: {e}")
                break

            if k in (ord("q"), ord("Q"), 27):
                print("quit requested")
                cv2.destroyAllWindows()
                print(f"done accepted={len(accepted_set)} rejected={len(rejected_set)}")
                return

    cv2.destroyAllWindows()
    print(f"done accepted={len(accepted_set)} rejected={len(rejected_set)}")


if __name__ == "__main__":
    main()
