from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from wtcv_utils.labelme import polygon_area as polygon_area, shape_to_points as shape_to_points
from wtcv_utils.records import load_labelme_pairs


def _draw_polygons(image_bgr: np.ndarray, polys: List[List[List[float]]], color: Tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy()
    for poly in polys:
        if len(poly) < 3:
            continue
        arr = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [arr], True, color, 2, cv2.LINE_AA)
    return out


def dataset_peek(
    data_dir: str,
    label: str,
    max_images: int,
    sample_count: int,
    seed: int,
):
    dd = Path(data_dir.strip()) if data_dir else Path("")
    if not dd.exists():
        return f"Missing dataset dir: {dd}", []

    rng = random.Random(int(seed))
    pairs = load_labelme_pairs(
        dd,
        load_workers=8,
        max_images=int(max_images),
        random_sample=bool(int(max_images) > 0),
        sample_seed=int(seed),
        progress_desc="dataset_peek",
        progress_leave=False,
    )

    label_cf = label.strip().casefold()
    n_images = 0
    n_objects = 0
    ratios: List[float] = []
    labels_seen = set()

    for pair in pairs:
        ip = pair.image_path
        d = pair.json_data
        w = int(d.get("imageWidth", 0) or 0)
        h = int(d.get("imageHeight", 0) or 0)
        if w <= 0 or h <= 0:
            try:
                with Image.open(ip) as im:
                    w, h = im.size
            except Exception:
                continue
        if w <= 0 or h <= 0:
            continue
        n_images += 1
        img_area = float(max(1, w * h))
        for s in d.get("shapes", []) or []:
            labels_seen.add(str(s.get("label", "")))
            if str(s.get("label", "")).strip().casefold() != label_cf:
                continue
            pts = shape_to_points(s, min_poly_points=3)
            if pts is None:
                continue
            a = polygon_area(pts)
            if a <= 0:
                continue
            n_objects += 1
            ratios.append(float(a / img_area))

    if n_images == 0:
        return "No readable image/json pairs found.", []

    arr = np.array(ratios, dtype=np.float64) if ratios else np.array([], dtype=np.float64)

    def pct(x: float) -> str:
        return f"{100.0 * x:.4f}%"

    summary = []
    summary.append(f"Dataset: `{dd}`")
    summary.append(f"Images: **{n_images}**")
    summary.append(f"Objects with label `{label}`: **{n_objects}**")
    summary.append(f"All labels seen: {sorted(labels_seen)}")
    if arr.size > 0:
        summary.append(f"Mean area ratio: {pct(float(arr.mean()))}")
        summary.append(f"Median area ratio: {pct(float(np.median(arr)))}")
        summary.append(f"p90 area ratio: {pct(float(np.quantile(arr, 0.90)))}")
        summary.append(f"<1% area: {int((arr < 0.01).sum())} / {arr.size}")
        summary.append(f"<0.5% area: {int((arr < 0.005).sum())} / {arr.size}")
        summary.append(f"<0.1% area: {int((arr < 0.001).sum())} / {arr.size}")
    else:
        summary.append("No valid objects for the selected label.")

    gallery: List[Tuple[np.ndarray, str]] = []
    sample_pairs = list(pairs)
    rng.shuffle(sample_pairs)
    for pair in sample_pairs:
        if len(gallery) >= int(sample_count):
            break
        ip = pair.image_path
        d = pair.json_data
        img_bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        polys = []
        for s in d.get("shapes", []) or []:
            if str(s.get("label", "")).strip().casefold() != label_cf:
                continue
            pts = shape_to_points(s, min_poly_points=3)
            if pts is None:
                continue
            if len(pts) >= 3:
                polys.append(pts)
        if not polys:
            continue
        view = _draw_polygons(img_bgr, polys, (0, 255, 255))
        gallery.append((view[:, :, ::-1], f"{ip.name} | objs={len(polys)}"))

    return "\n".join(summary), gallery


def build_tab(root: Path) -> None:
    with gr.Tab("Dataset Peek"):
        with gr.Row():
            peek_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Dataset Dir")
            peek_label = gr.Textbox(value="vehicle", label="Label")
            peek_max = gr.Number(value=0, precision=0, label="Max Images for Stats (0=all)")
            peek_sample = gr.Number(value=9, precision=0, label="Gallery Samples")
            peek_seed = gr.Number(value=42, precision=0, label="Seed")
        peek_btn = gr.Button("Analyze Dataset", variant="primary")
        peek_report = gr.Markdown()
        peek_gallery = gr.Gallery(label="Random Samples (with overlay)", columns=3, height=550)
        peek_btn.click(
            fn=dataset_peek,
            inputs=[peek_dir, peek_label, peek_max, peek_sample, peek_seed],
            outputs=[peek_report, peek_gallery],
        )
