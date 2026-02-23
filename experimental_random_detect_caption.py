#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from curate_model_predictions_to_labelme import infer_prob_map, load_model, mask_to_polygons
from wtcv_utils.labelme import IMG_EXTS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Experimental: random image -> adapter detections -> center-crop detections -> HF VQA"
    )
    ap.add_argument("--input-dir", type=Path, required=True, help="Folder with images")
    ap.add_argument("--image-path", type=Path, default=None, help="Optional direct image path; if set, skips random sampling.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Stage1 adapter checkpoint")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/experimental_detect_caption"))
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pred-threshold", type=float, default=0.35)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--tile-stride", type=int, default=512)
    ap.add_argument("--seg-out-stride", type=int, default=4)
    ap.add_argument("--use-tile-cls-gating", action="store_true", default=True)
    ap.add_argument("--no-use-tile-cls-gating", action="store_false", dest="use_tile_cls_gating")
    ap.add_argument("--tile-cls-threshold", type=float, default=0.5)
    ap.add_argument("--tile-cls-mode", type=str, choices=["hard", "multiply"], default="hard")
    ap.add_argument("--min-poly-area", type=float, default=14.0)
    ap.add_argument("--poly-epsilon-frac", type=float, default=0.002)

    ap.add_argument("--min-det-area", type=float, default=64.0, help="Minimum detection bbox area in px")
    ap.add_argument("--min-det-side", type=int, default=8, help="Minimum detection bbox side in px")
    ap.add_argument("--crop-context", type=float, default=1.6, help="Square crop context around det bbox")
    ap.add_argument("--caption-crop-size", type=int, default=224, help="Resize crops for VQA model")
    ap.add_argument("--vqa-model", type=str, default="Salesforce/blip-vqa-base")
    ap.add_argument(
        "--vqa-prompt",
        type=str,
        default="What is the main military object in this image? Answer with a short noun phrase.",
    )
    ap.add_argument("--device", type=str, default="", help="cuda|cpu, blank=auto")
    return ap.parse_args()


def _discover_images(input_dir: Path) -> List[Path]:
    exts = {str(e).lower() for e in IMG_EXTS}
    return [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _safe_crop_square(
    image_rgb: np.ndarray,
    cx: float,
    cy: float,
    side: int,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    side = int(max(8, side))
    x0 = int(round(cx - 0.5 * side))
    y0 = int(round(cy - 0.5 * side))
    x1 = x0 + side
    y1 = y0 + side
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)
    out = np.zeros((side, side, 3), dtype=np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        crop = image_rgb[sy0:sy1, sx0:sx1]
        ox = sx0 - x0
        oy = sy0 - y0
        out[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    return out


def _load_vqa_pipeline(model_name: str, device: torch.device):
    from transformers import pipeline  # lazy import

    hf_device = 0 if str(device).startswith("cuda") and torch.cuda.is_available() else -1
    errors = []
    for task in ("visual-question-answering", "vqa"):
        try:
            return pipeline(task, model=model_name, device=hf_device)
        except Exception as e:
            errors.append(f"{task}: {e}")
    raise RuntimeError(
        "Failed to initialize VQA pipeline for visual-question-answering/vqa. "
        + " | ".join(errors)
    )


def _answer_crop(vqa_pipe, crop_rgb: np.ndarray, prompt: str) -> str:
    img = Image.fromarray(crop_rgb)
    q = str(prompt).strip() or "What is in this image?"
    out = None
    # Handle common VQA pipeline call shapes.
    for call in (
        lambda: vqa_pipe(image=img, question=q),
        lambda: vqa_pipe({"image": img, "question": q}),
        lambda: vqa_pipe(img, q),
    ):
        try:
            out = call()
            break
        except Exception:
            continue
    if out is None:
        return "(no answer)"

    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        txt = out[0].get("answer", "")
        if txt is None or str(txt).strip() == "":
            txt = out[0].get("generated_text", "")
    else:
        txt = out

    t = str(txt).strip()
    # Remove common prompt-echo patterns.
    for pfx in (
        q,
        q.lower(),
        "answer:",
        "Answer:",
    ):
        if t.startswith(pfx):
            t = t[len(pfx) :].strip(" :.-\n\t")
    # Very short/empty fallback to avoid blank UI rows.
    if t == "":
        t = "(no answer)"
    return t


def _mask_stats_for_poly(prob_full: np.ndarray, poly: List[List[float]]) -> Tuple[float, float, int]:
    h, w = prob_full.shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return float("nan"), float("nan"), 0
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
    cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    sel = prob_full[m > 0]
    if sel.size <= 0:
        return float("nan"), float("nan"), 0
    return float(sel.mean()), float(sel.max()), int(sel.size)


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {args.input_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    image_path_raw = ""
    if args.image_path is not None:
        image_path_raw = str(args.image_path).strip()
    if image_path_raw.startswith("file://"):
        image_path_raw = image_path_raw[len("file://") :].strip()
    if image_path_raw.lower() == "none":
        image_path_raw = ""

    ip: Path
    selection_mode = "random"
    if image_path_raw != "":
        ip = Path(image_path_raw).expanduser()
        if not ip.exists() or not ip.is_file():
            raise FileNotFoundError(f"Missing image-path file: {ip}")
        selection_mode = "direct"
    else:
        images = _discover_images(args.input_dir)
        if len(images) == 0:
            raise RuntimeError(f"No images found in: {args.input_dir}")
        rng = random.Random(int(args.seed))
        ip = images[rng.randrange(len(images))]
    image_rgb = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
    h, w = image_rgb.shape[:2]

    device = torch.device(args.device.strip() if args.device.strip() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_info = load_model(args.checkpoint, device)

    prob_lr, _cls_lr, stats = infer_prob_map(
        model=model,
        image_np=image_rgb,
        tile_size=int(args.tile_size),
        stride=int(args.tile_stride),
        seg_out_stride=int(args.seg_out_stride),
        device=device,
        use_tile_cls_gating=bool(args.use_tile_cls_gating),
        tile_cls_threshold=float(args.tile_cls_threshold),
        tile_cls_mode=str(args.tile_cls_mode),
    )
    prob_full = F.interpolate(
        torch.from_numpy(prob_lr).float().unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    pred_mask = (prob_full >= float(args.pred_threshold)).astype(np.uint8)
    polys = mask_to_polygons(
        pred_mask,
        min_area=float(args.min_poly_area),
        epsilon_frac=float(args.poly_epsilon_frac),
    )

    vqa_pipe = _load_vqa_pipeline(str(args.vqa_model), device=device)
    output_rows = []

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    preview = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    kept = 0
    for i, poly in enumerate(polys):
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        area = float(bw * bh)
        if area < float(args.min_det_area):
            continue
        if bw < int(args.min_det_side) or bh < int(args.min_det_side):
            continue
        pred_mean, pred_max, mask_area_px = _mask_stats_for_poly(prob_full, poly)

        cx = float(x + 0.5 * bw)
        cy = float(y + 0.5 * bh)
        side = int(max(bw, bh) * float(args.crop_context))
        crop = _safe_crop_square(image_rgb, cx=cx, cy=cy, side=side)
        crop = np.array(Image.fromarray(crop).resize((int(args.caption_crop_size), int(args.caption_crop_size)), Image.BILINEAR))

        answer = _answer_crop(vqa_pipe, crop, prompt=str(args.vqa_prompt))
        crop_name = f"{ip.stem}__det{i:03d}.jpg"
        crop_path = crops_dir / crop_name
        Image.fromarray(crop).save(crop_path, quality=95)

        kept += 1
        color = (0, 255, 255)
        cv2.rectangle(preview, (int(x), int(y)), (int(x + bw), int(y + bh)), color, 2)
        cv2.putText(
            preview,
            f"{kept}: {answer[:60]}",
            (int(x), max(18, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        output_rows.append(
            {
                "det_index": int(i),
                "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
                "bbox_area": float(area),
                "mask_area_px": int(mask_area_px),
                "adapter_pred_mean": float(pred_mean),
                "adapter_pred_max": float(pred_max),
                "crop_path": str(crop_path),
                "question": str(args.vqa_prompt),
                "answer": str(answer),
            }
        )

    preview_path = out_dir / f"{ip.stem}__preview.jpg"
    cv2.imwrite(str(preview_path), preview)
    result = {
        "input_dir": str(args.input_dir),
        "selection_mode": str(selection_mode),
        "image_path_arg": str(image_path_raw),
        "picked_image": str(ip),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "model_info": model_info,
        "vqa_model": str(args.vqa_model),
        "vqa_prompt": str(args.vqa_prompt),
        "pred_threshold": float(args.pred_threshold),
        "min_det_area": float(args.min_det_area),
        "min_det_side": int(args.min_det_side),
        "num_polygons_total": int(len(polys)),
        "num_kept_for_caption": int(kept),
        "tile_cls_stats": stats,
        "preview_path": str(preview_path),
        "detections": output_rows,
    }
    print(f"selection_mode={selection_mode}")
    print(f"image_path_arg={image_path_raw}")
    print(f"picked_image={ip}")
    print("RESULT_JSON_BEGIN")
    print(json.dumps(result, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
