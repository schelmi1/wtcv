from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from curate_model_predictions_to_labelme import infer_prob_map, load_model, make_labelme_json, mask_to_polygons
from wtcv_app.common import as_bool


@dataclass
class CachedModel:
    model: torch.nn.Module
    info: Dict
    mtime_ns: int
    device: str


_MODEL_CACHE: Dict[str, CachedModel] = {}


def _draw_polygons(image_bgr: np.ndarray, polys: List[List[List[float]]], color: Tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy()
    for poly in polys:
        if len(poly) < 3:
            continue
        arr = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [arr], True, color, 2, cv2.LINE_AA)
    return out


def _get_cached_model(checkpoint: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    key = str(checkpoint.resolve())
    mtime = checkpoint.stat().st_mtime_ns
    cached = _MODEL_CACHE.get(key)
    if cached is not None and cached.mtime_ns == mtime and cached.device == str(device):
        return cached.model, cached.info
    model, info = load_model(checkpoint, device)
    _MODEL_CACHE[key] = CachedModel(model=model, info=info, mtime_ns=mtime, device=str(device))
    return model, info


def single_image_infer(
    image_path: str,
    checkpoint: str,
    label: str,
    tile_size: int,
    tile_stride: int,
    seg_out_stride: int,
    pred_threshold: float,
    use_tile_cls_gating: bool,
    tile_cls_threshold: float,
    tile_cls_mode: str,
    min_poly_area: float,
    poly_epsilon_frac: float,
):
    use_tile_cls_gating = as_bool(use_tile_cls_gating)
    ip = Path(image_path.strip()) if image_path else Path("")
    cp = Path(checkpoint.strip()) if checkpoint else Path("")
    if not ip.exists():
        return None, None, None, f"Missing image: {ip}"
    if not cp.exists():
        return None, None, None, f"Missing checkpoint: {cp}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = _get_cached_model(cp, device)

    image_rgb = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]

    prob_lr, _cls_lr, stats = infer_prob_map(
        model=model,
        image_np=image_rgb,
        tile_size=int(tile_size),
        stride=int(tile_stride),
        seg_out_stride=int(seg_out_stride),
        device=device,
        use_tile_cls_gating=bool(use_tile_cls_gating),
        tile_cls_threshold=float(tile_cls_threshold),
        tile_cls_mode=str(tile_cls_mode),
    )

    prob_full = F.interpolate(
        torch.from_numpy(prob_lr).float().unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    pred_mask = (prob_full >= float(pred_threshold)).astype(np.uint8)
    polys = mask_to_polygons(pred_mask, min_area=float(min_poly_area), epsilon_frac=float(poly_epsilon_frac))

    heat_u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_MAGMA)
    heat_overlay = cv2.addWeighted(image_bgr, 0.60, heat, 0.40, 0.0)

    poly_view = _draw_polygons(image_bgr, polys, (0, 255, 255))
    mask_vis = (pred_mask * 255).astype(np.uint8)

    labelme = make_labelme_json(ip.name, h, w, polys, label)
    report = {
        "device": str(device),
        "checkpoint": str(cp),
        "image": str(ip),
        "model_info": info,
        "stats": stats,
        "polygons": len(polys),
        "mask_pixels": int(pred_mask.sum()),
        "labelme_preview": labelme,
    }
    return poly_view[:, :, ::-1], heat_overlay[:, :, ::-1], mask_vis, json.dumps(report, indent=2)


def build_tab() -> None:
    with gr.Tab("Single Image Inference"):
        with gr.Row():
            infer_image_path = gr.Textbox(value="", label="Image Path", info="Path to the single image to run inference on.")
            infer_ckpt = gr.Textbox(value="", label="Checkpoint", info="Path to a trained model checkpoint (.pt) to load for inference/evaluation.")
            infer_label = gr.Textbox(value="vehicle", label="Label", info="Class label name used for output polygons and evaluation target.")
        with gr.Row():
            infer_tile = gr.Number(value=256, precision=0, label="Tile Size", info="Side length of each square inference/training tile in pixels.")
            infer_stride = gr.Number(value=128, precision=0, label="Tile Stride", info="Step size between tile origins; lower values add overlap and compute cost.")
            infer_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride", info="Output stride of segmentation logits relative to tile resolution.")
            infer_thr = gr.Number(value=0.5, label="Pred Threshold", info="Probability threshold used to convert logits/probabilities into a binary mask.")
        with gr.Row():
            infer_gate = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Gating", info="If on, tile classification score gates segmentation output per tile.")
            infer_tile_cls_thr = gr.Number(value=0.5, label="Tile Cls Threshold", info="Minimum tile classification confidence required for gating.")
            infer_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile Cls Mode", info="Gating behavior: hard masking or probability multiplication.")
            infer_min_poly = gr.Number(value=20.0, label="Min Poly Area", info="Minimum polygon area kept during mask-to-polygon conversion.")
            infer_eps = gr.Number(value=0.002, label="Poly Epsilon", info="Polygon simplification epsilon fraction used by contour approximation.")
        infer_btn = gr.Button("Run Inference", variant="primary")
        with gr.Row():
            infer_poly_img = gr.Image(label="Polygons Overlay", type="numpy")
            infer_heat_img = gr.Image(label="Heatmap Overlay", type="numpy")
            infer_mask_img = gr.Image(label="Pred Mask", type="numpy")
        infer_report = gr.Textbox(label="Inference Report", lines=22, elem_classes=["mono"])
        infer_btn.click(
            fn=single_image_infer,
            inputs=[
                infer_image_path,
                infer_ckpt,
                infer_label,
                infer_tile,
                infer_stride,
                infer_seg_stride,
                infer_thr,
                infer_gate,
                infer_tile_cls_thr,
                infer_tile_cls_mode,
                infer_min_poly,
                infer_eps,
            ],
            outputs=[infer_poly_img, infer_heat_img, infer_mask_img, infer_report],
        )
