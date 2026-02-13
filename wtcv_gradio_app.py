#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "Gradio is required to run this app. Install with: pip install gradio"
    ) from exc

from curate_model_predictions_to_labelme import (
    infer_prob_map,
    load_model,
    make_labelme_json,
    mask_to_polygons,
)


ROOT = Path(__file__).resolve().parent
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp"}
MAX_LOG_CHARS = 250_000

THEME_INIT_JS = r"""
() => {
  const THEMES = ["default", "ocean", "forest", "ember"];
  const MODE_KEY = "wtcv_mode";
  const THEME_KEY = "wtcv_theme";
  const body = document.body;
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ""; }
  function getMode() { return localStorage.getItem(MODE_KEY) || "light"; }
  function getTheme() {
    const t = localStorage.getItem(THEME_KEY) || "default";
    return THEMES.includes(t) ? t : "default";
  }
  function setBtnText(id, text) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.tagName && el.tagName.toLowerCase() === "button") {
      el.textContent = text;
      return;
    }
    const btn = el.querySelector("button");
    if (btn) btn.textContent = text;
  }
  function applyUi() {
    const mode = getMode();
    const theme = getTheme();
    const root = document.documentElement;
    const container = document.querySelector(".gradio-container");
    const nodes = [body, root, container].filter(Boolean);
    nodes.forEach((n) => {
      n.classList.toggle("wtcv-dark", mode === "dark");
      n.classList.remove("wtcv-theme-ocean", "wtcv-theme-forest", "wtcv-theme-ember");
      if (theme !== "default") n.classList.add("wtcv-theme-" + theme);
    });
    setBtnText("wtcv-mode-btn", "Mode: " + cap(mode));
    setBtnText("wtcv-theme-btn", "Theme: " + cap(theme));
  }
  function toggleMode() {
    const next = getMode() === "dark" ? "light" : "dark";
    localStorage.setItem(MODE_KEY, next);
    applyUi();
  }
  function cycleTheme() {
    const cur = getTheme();
    const idx = THEMES.indexOf(cur);
    const next = THEMES[(idx + 1 + THEMES.length) % THEMES.length];
    localStorage.setItem(THEME_KEY, next);
    applyUi();
  }
  function install() {
    applyUi();
    const modeBtn = document.getElementById("wtcv-mode-btn");
    const themeBtn = document.getElementById("wtcv-theme-btn");
    if (modeBtn && !modeBtn.dataset.bound) {
      modeBtn.dataset.bound = "1";
      modeBtn.addEventListener("click", toggleMode);
    }
    if (themeBtn && !themeBtn.dataset.bound) {
      themeBtn.dataset.bound = "1";
      themeBtn.addEventListener("click", cycleTheme);
    }
  }
  install();
  setTimeout(install, 120);
  setTimeout(install, 800);
}
"""

@dataclass
class CachedModel:
    model: torch.nn.Module
    info: Dict
    mtime_ns: int
    device: str


_MODEL_CACHE: Dict[str, CachedModel] = {}


def _find_image_for_json(data_dir: Path, stem: str) -> Optional[Path]:
    for p in data_dir.glob(f"{stem}.*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def _shape_to_points(shape: Dict) -> List[List[float]]:
    st = str(shape.get("shape_type", "")).lower()
    pts = shape.get("points", []) or []
    if st == "rectangle" and len(pts) >= 2:
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        lx, rx = min(float(x0), float(x1)), max(float(x0), float(x1))
        ty, by = min(float(y0), float(y1)), max(float(y0), float(y1))
        return [[lx, ty], [rx, ty], [rx, by], [lx, by]]
    return [[float(p[0]), float(p[1])] for p in pts]


def _polygon_area(points: List[List[float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points], dtype=np.float32)
    y = np.array([p[1] for p in points], dtype=np.float32)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _draw_polygons(image_bgr: np.ndarray, polys: List[List[List[float]]], color: Tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy()
    for poly in polys:
        if len(poly) < 3:
            continue
        arr = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [arr], True, color, 2, cv2.LINE_AA)
    return out


def _trim_logs(text: str) -> str:
    if len(text) <= MAX_LOG_CHARS:
        return text
    return text[-MAX_LOG_CHARS:]


def _looks_like_tqdm_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "\x1b[" in s:
        return True
    if re.search(r"\d+%\|", s) is not None:
        return True
    if ("it/s" in s and "|" in s) or ("s/it" in s and "|" in s):
        return True
    if re.search(r"\b\d+/\d+\b", s) is not None and "|" in s:
        return True
    return False


def _stream_command(cmd: Sequence[str], cwd: Optional[Path] = None) -> Generator[Tuple[str, str], None, None]:
    cmd_list = [str(x) for x in cmd]
    cmd_str = shlex.join(cmd_list)
    base_logs = f"$ {cmd_str}\n\n"
    current_line = ""
    live_tqdm_line = ""
    yield cmd_str, base_logs

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        env=env,
    )

    assert proc.stdout is not None

    def _commit_line(line: str) -> None:
        nonlocal base_logs, live_tqdm_line
        if line == "":
            return
        if _looks_like_tqdm_line(line):
            live_tqdm_line = line
        else:
            if live_tqdm_line:
                base_logs += live_tqdm_line + "\n"
                live_tqdm_line = ""
            base_logs += line + "\n"

    while True:
        chunk = proc.stdout.read(2048)
        if not chunk:
            if proc.poll() is not None:
                break
            continue

        text = chunk.decode("utf-8", errors="replace")
        for ch in text:
            if ch == "\r":
                # tqdm updates line in place; keep as transient live line.
                if current_line:
                    live_tqdm_line = current_line
                    current_line = ""
            elif ch == "\n":
                _commit_line(current_line)
                current_line = ""
            else:
                current_line += ch

        transient = live_tqdm_line if live_tqdm_line else current_line
        yield cmd_str, _trim_logs(base_logs + transient)

    rc = proc.wait()
    _commit_line(current_line)
    if live_tqdm_line:
        base_logs += live_tqdm_line + "\n"
        live_tqdm_line = ""
    logs = _trim_logs(base_logs + f"\n[process_exit_code={rc}]\n")
    yield cmd_str, logs


def _build_bool_arg(flag_true: str, flag_false: str, value: bool) -> List[str]:
    return [flag_true if value else flag_false]


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "on", "enable", "enabled", "amp"}


def run_train(
    data_dir: str,
    output_dir: str,
    run_name: str,
    resume_checkpoint: str,
    epochs: int,
    subset_size: int,
    label: str,
    fp_label: str,
    tile_size: int,
    tile_stride: int,
    tile_scales: str,
    batch_size: int,
    num_workers: int,
    dino_upsampler: str,
    anyup_q_chunk_size: int,
    head_type: str,
    use_tile_cls_head: bool,
    tile_cls_weight: float,
    balance_train_50_50: bool,
    balance_val_50_50: bool,
    augment_low_vis: bool,
    hard_negative_mining: bool,
    hnm_hard_ratio: float,
    hnm_pool_frac: float,
    lr: float,
    lr_scheduler: str,
    lr_min: float,
    weight_decay: float,
    mcc_weight: float,
    mcc_warmup_epochs: int,
    bce_weight: float,
    boundary_weight: float,
    use_fp_supervision: bool,
    fp_neg_weight: float,
    fp_neg_ratio: float,
    val_interval: int,
    image_log_interval: int,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    use_tile_cls_head = _as_bool(use_tile_cls_head)
    balance_train_50_50 = _as_bool(balance_train_50_50)
    balance_val_50_50 = _as_bool(balance_val_50_50)
    augment_low_vis = _as_bool(augment_low_vis)
    hard_negative_mining = _as_bool(hard_negative_mining)
    use_fp_supervision = _as_bool(use_fp_supervision)
    trust_torch_hub_repo = _as_bool(trust_torch_hub_repo)

    cmd = [
        sys.executable,
        str(ROOT / "train_stage1_seg.py"),
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--epochs",
        str(int(epochs)),
        "--subset-size",
        str(int(subset_size)),
        "--label",
        label,
        "--fp-label",
        fp_label,
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--tile-scales",
        tile_scales,
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--dino-upsampler",
        dino_upsampler,
        "--anyup-q-chunk-size",
        str(int(anyup_q_chunk_size)),
        "--head-type",
        head_type,
        "--tile-cls-weight",
        str(float(tile_cls_weight)),
        "--hnm-hard-ratio",
        str(float(hnm_hard_ratio)),
        "--hnm-pool-frac",
        str(float(hnm_pool_frac)),
        "--lr",
        str(float(lr)),
        "--lr-scheduler",
        lr_scheduler,
        "--lr-min",
        str(float(lr_min)),
        "--weight-decay",
        str(float(weight_decay)),
        "--mcc-weight",
        str(float(mcc_weight)),
        "--mcc-warmup-epochs",
        str(int(mcc_warmup_epochs)),
        "--bce-weight",
        str(float(bce_weight)),
        "--boundary-weight",
        str(float(boundary_weight)),
        "--fp-neg-weight",
        str(float(fp_neg_weight)),
        "--fp-neg-ratio",
        str(float(fp_neg_ratio)),
        "--val-interval",
        str(int(val_interval)),
        "--image-log-interval",
        str(int(image_log_interval)),
    ]
    if run_name.strip():
        cmd += ["--run-name", run_name.strip()]
    if resume_checkpoint.strip():
        cmd += ["--resume-checkpoint", resume_checkpoint.strip()]

    cmd += _build_bool_arg("--use-tile-cls-head", "--no-use-tile-cls-head", bool(use_tile_cls_head))
    cmd += _build_bool_arg("--balance-train-50-50", "--no-balance-train-50-50", bool(balance_train_50_50))
    cmd += _build_bool_arg("--balance-val-50-50", "--no-balance-val-50-50", bool(balance_val_50_50))
    if augment_low_vis:
        cmd += ["--augment-low-vis"]
    cmd += _build_bool_arg("--hard-negative-mining", "--no-hard-negative-mining", bool(hard_negative_mining))
    cmd += _build_bool_arg("--use-fp-supervision", "--no-use-fp-supervision", bool(use_fp_supervision))
    cmd += _build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))

    yield from _stream_command(cmd)


def run_eval(
    data_dir: str,
    checkpoint: str,
    label_name: str,
    tile_size: int,
    tile_stride: int,
    seg_out_stride: int,
    pred_threshold: float,
    num_workers: int,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    trust_torch_hub_repo = _as_bool(trust_torch_hub_repo)
    cmd = [
        sys.executable,
        str(ROOT / "eval_stage1_seg.py"),
        "--data-dir",
        data_dir,
        "--checkpoint",
        checkpoint,
        "--label-name",
        label_name,
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--seg-out-stride",
        str(int(seg_out_stride)),
        "--pred-threshold",
        str(float(pred_threshold)),
        "--num-workers",
        str(int(num_workers)),
    ]
    cmd += _build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    yield from _stream_command(cmd)


def run_sam_convert(
    input_dir: str,
    output_dir: str,
    model_id: str,
    device: str,
    image_batch_size: int,
    prompt_mode: str,
    load_workers: int,
    min_poly_area: float,
    poly_epsilon_frac: float,
    max_images: int,
    overwrite: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite = _as_bool(overwrite)
    cmd = [
        sys.executable,
        str(ROOT / "sam1_box_to_poly_batched.py"),
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--model-id",
        model_id,
        "--device",
        device,
        "--image-batch-size",
        str(int(image_batch_size)),
        "--prompt-mode",
        str(prompt_mode).strip().lower(),
        "--load-workers",
        str(int(load_workers)),
        "--min-poly-area",
        str(float(min_poly_area)),
        "--poly-epsilon-frac",
        str(float(poly_epsilon_frac)),
        "--max-images",
        str(int(max_images)),
    ]
    if overwrite:
        cmd += ["--overwrite"]
    yield from _stream_command(cmd)


def run_augment(
    target_dir: str,
    donor_dir: str,
    output_dir: str,
    target_label: str,
    donor_label: str,
    seed: int,
    min_pastes_per_image: int,
    max_pastes_per_image: int,
    max_images: int,
    donor_max_images: int,
    placement_horizon_frac: float,
    max_placement_tries: int,
    max_overlap_iou: float,
    size_min_ratio: float,
    size_max_ratio: float,
    min_poly_area: float,
    poly_epsilon_frac: float,
    feather_radius: int,
    jpeg_quality_min: int,
    jpeg_quality_max: int,
    occlusion_prob: float,
    overwrite: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite = _as_bool(overwrite)
    cmd = [
        sys.executable,
        str(ROOT / "augment_record_pairs_with_polygons.py"),
        "--target-dir",
        target_dir,
        "--donor-dir",
        donor_dir,
        "--output-dir",
        output_dir,
        "--target-label",
        target_label,
        "--donor-label",
        donor_label,
        "--seed",
        str(int(seed)),
        "--min-pastes-per-image",
        str(int(min_pastes_per_image)),
        "--max-pastes-per-image",
        str(int(max_pastes_per_image)),
        "--max-images",
        str(int(max_images)),
        "--donor-max-images",
        str(int(donor_max_images)),
        "--placement-horizon-frac",
        str(float(placement_horizon_frac)),
        "--max-placement-tries",
        str(int(max_placement_tries)),
        "--max-overlap-iou",
        str(float(max_overlap_iou)),
        "--size-min-ratio",
        str(float(size_min_ratio)),
        "--size-max-ratio",
        str(float(size_max_ratio)),
        "--min-poly-area",
        str(float(min_poly_area)),
        "--poly-epsilon-frac",
        str(float(poly_epsilon_frac)),
        "--feather-radius",
        str(int(feather_radius)),
        "--jpeg-quality-min",
        str(int(jpeg_quality_min)),
        "--jpeg-quality-max",
        str(int(jpeg_quality_max)),
        "--occlusion-prob",
        str(float(occlusion_prob)),
    ]
    if overwrite:
        cmd += ["--overwrite"]
    yield from _stream_command(cmd)


def run_curation(
    input_dir: str,
    checkpoint: str,
    output_dir: str,
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
    max_images: int,
    start_index: int,
    save_preview: bool,
) -> Generator[Tuple[str, str], None, None]:
    use_tile_cls_gating = _as_bool(use_tile_cls_gating)
    save_preview = _as_bool(save_preview)
    cmd = [
        sys.executable,
        str(ROOT / "curate_model_predictions_to_labelme.py"),
        "--input-dir",
        input_dir,
        "--checkpoint",
        checkpoint,
        "--output-dir",
        output_dir,
        "--label",
        label,
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--seg-out-stride",
        str(int(seg_out_stride)),
        "--pred-threshold",
        str(float(pred_threshold)),
        "--tile-cls-threshold",
        str(float(tile_cls_threshold)),
        "--tile-cls-mode",
        tile_cls_mode,
        "--min-poly-area",
        str(float(min_poly_area)),
        "--poly-epsilon-frac",
        str(float(poly_epsilon_frac)),
        "--max-images",
        str(int(max_images)),
        "--start-index",
        str(int(start_index)),
    ]
    cmd += _build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    if save_preview:
        cmd += ["--save-preview"]
    yield from _stream_command(cmd)


def run_extract_frames(
    video_path: str,
    output_dir: str,
    fps: float,
    overwrite: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite = _as_bool(overwrite)
    video = Path(video_path)
    if not video.exists():
        cmd_str = "ffmpeg (not executed)"
        logs = f"Missing video file: {video_path}\n"
        yield cmd_str, logs
        return
    out_dir = Path(output_dir)
    if overwrite and out_dir.exists():
        for p in out_dir.glob("*"):
            if p.is_file():
                p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pat = str(out_dir / f"{video.stem}_f%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(video),
        "-vf",
        f"fps={float(fps)}",
        out_pat,
    ]
    yield from _stream_command(cmd)


def run_object_umap(
    input_dir: str,
    dataset_name: str,
    output_dir: str,
    label_filter: str,
    max_objects: int,
    tile_size: int,
    tile_context_scale: float,
    batch_size: int,
    dino_model: str,
    device: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    num_clusters: int,
    seed: int,
    overwrite_dataset: bool,
    launch: bool,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite_dataset = _as_bool(overwrite_dataset)
    launch = _as_bool(launch)
    trust_torch_hub_repo = _as_bool(trust_torch_hub_repo)

    cmd = [
        sys.executable,
        str(ROOT / "fiftyone_object_umap.py"),
        "--input-dir",
        input_dir,
        "--dataset-name",
        dataset_name,
        "--output-dir",
        output_dir,
        "--label-filter",
        label_filter,
        "--max-objects",
        str(int(max_objects)),
        "--tile-size",
        str(int(tile_size)),
        "--tile-context-scale",
        str(float(tile_context_scale)),
        "--batch-size",
        str(int(batch_size)),
        "--dino-model",
        dino_model,
        "--device",
        str(device),
        "--umap-n-neighbors",
        str(int(umap_n_neighbors)),
        "--umap-min-dist",
        str(float(umap_min_dist)),
        "--umap-metric",
        umap_metric,
        "--num-clusters",
        str(int(num_clusters)),
        "--seed",
        str(int(seed)),
    ]
    cmd += _build_bool_arg("--overwrite-dataset", "--no-overwrite-dataset", bool(overwrite_dataset))
    cmd += _build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    if launch:
        cmd += ["--launch"]
    yield from _stream_command(cmd)


def run_export_tagged_labelme(
    dataset_name: str,
    output_dir: str,
    tag_labels: str,
    overwrite: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite = _as_bool(overwrite)
    cmd = [
        sys.executable,
        str(ROOT / "fiftyone_export_tagged_to_labelme.py"),
        "--dataset-name",
        dataset_name,
        "--output-dir",
        output_dir,
        "--tag-labels",
        tag_labels,
    ]
    cmd += _build_bool_arg("--overwrite", "--no-overwrite", bool(overwrite))
    yield from _stream_command(cmd)


def run_live_screen(
    checkpoint: str,
    label: str,
    monitor_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
    tile_size: int,
    tile_stride: int,
    seg_out_stride: int,
    pred_threshold: float,
    use_tile_cls_gating: bool,
    tile_cls_threshold: float,
    tile_cls_mode: str,
    min_poly_area: float,
    poly_epsilon_frac: float,
    max_fps: float,
    infer_every: int,
    amp_mode: str,
    auto_save_persistent: bool,
    persist_infers: int,
    persist_iou_threshold: float,
    persist_max_miss: int,
    persist_save_cooldown_infers: int,
    output_dir: str,
    save_preview: bool,
    print_monitors: bool,
) -> Generator[Tuple[str, str], None, None]:
    use_tile_cls_gating = _as_bool(use_tile_cls_gating)
    auto_save_persistent = _as_bool(auto_save_persistent)
    save_preview = _as_bool(save_preview)
    print_monitors = _as_bool(print_monitors)
    cmd = [
        sys.executable,
        str(ROOT / "live_screen_inference_cv2.py"),
        "--checkpoint",
        checkpoint,
        "--label",
        label,
        "--monitor-index",
        str(int(monitor_index)),
        "--x",
        str(int(x)),
        "--y",
        str(int(y)),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--seg-out-stride",
        str(int(seg_out_stride)),
        "--pred-threshold",
        str(float(pred_threshold)),
        "--tile-cls-threshold",
        str(float(tile_cls_threshold)),
        "--tile-cls-mode",
        tile_cls_mode,
        "--min-poly-area",
        str(float(min_poly_area)),
        "--poly-epsilon-frac",
        str(float(poly_epsilon_frac)),
        "--max-fps",
        str(float(max_fps)),
        "--infer-every",
        str(int(infer_every)),
        "--persist-infers",
        str(int(persist_infers)),
        "--persist-iou-threshold",
        str(float(persist_iou_threshold)),
        "--persist-max-miss",
        str(int(persist_max_miss)),
        "--persist-save-cooldown-infers",
        str(int(persist_save_cooldown_infers)),
        "--output-dir",
        output_dir,
    ]
    cmd += _build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    if str(amp_mode).strip().lower() == "amp":
        cmd += ["--amp"]
    else:
        cmd += ["--no-amp"]
    if auto_save_persistent:
        cmd += ["--auto-save-persistent"]
    if save_preview:
        cmd += ["--save-preview"]
    if print_monitors:
        cmd += ["--print-monitors"]
    yield from _stream_command(cmd)


def run_media_source(
    checkpoint: str,
    input_path: str,
    label: str,
    output_dir: str,
    tile_size: int,
    tile_stride: int,
    seg_out_stride: int,
    pred_threshold: float,
    use_tile_cls_gating: bool,
    tile_cls_threshold: float,
    tile_cls_mode: str,
    min_poly_area: float,
    poly_epsilon_frac: float,
    max_fps: float,
    infer_every: int,
    amp_mode: str,
    ui_mode: str,
    auto_save: bool,
    save_empty: bool,
    save_preview: bool,
    start_index: int,
    max_items: int,
) -> Generator[Tuple[str, str], None, None]:
    use_tile_cls_gating = _as_bool(use_tile_cls_gating)
    auto_save = _as_bool(auto_save)
    save_empty = _as_bool(save_empty)
    save_preview = _as_bool(save_preview)
    ui_mode = str(ui_mode).strip().lower()

    cmd = [
        sys.executable,
        str(ROOT / "media_source_inference_cv2.py"),
        "--checkpoint",
        checkpoint,
        "--input-path",
        input_path,
        "--label",
        label,
        "--output-dir",
        output_dir,
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--seg-out-stride",
        str(int(seg_out_stride)),
        "--pred-threshold",
        str(float(pred_threshold)),
        "--tile-cls-threshold",
        str(float(tile_cls_threshold)),
        "--tile-cls-mode",
        tile_cls_mode,
        "--min-poly-area",
        str(float(min_poly_area)),
        "--poly-epsilon-frac",
        str(float(poly_epsilon_frac)),
        "--max-fps",
        str(float(max_fps)),
        "--infer-every",
        str(int(infer_every)),
        "--start-index",
        str(int(start_index)),
        "--max-items",
        str(int(max_items)),
    ]
    cmd += _build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    cmd += _build_bool_arg("--auto-save", "--no-auto-save", bool(auto_save))
    if save_empty:
        cmd += ["--save-empty"]
    if save_preview:
        cmd += ["--save-preview"]
    if str(amp_mode).strip().lower() == "amp":
        cmd += ["--amp"]
    else:
        cmd += ["--no-amp"]
    if ui_mode == "on":
        cmd += ["--ui"]
    else:
        cmd += ["--no-ui"]
    yield from _stream_command(cmd)


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
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    use_tile_cls_gating = _as_bool(use_tile_cls_gating)
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


def dataset_peek(
    data_dir: str,
    label: str,
    max_images: int,
    sample_count: int,
    seed: int,
) -> Tuple[str, List[Tuple[np.ndarray, str]]]:
    dd = Path(data_dir.strip()) if data_dir else Path("")
    if not dd.exists():
        return f"Missing dataset dir: {dd}", []

    rng = random.Random(int(seed))
    jfs = sorted(dd.glob("*.json"))
    if int(max_images) > 0 and int(max_images) < len(jfs):
        jfs = rng.sample(jfs, int(max_images))

    label_cf = label.strip().casefold()
    n_images = 0
    n_objects = 0
    ratios: List[float] = []
    labels_seen = set()

    for jf in jfs:
        ip = _find_image_for_json(dd, jf.stem)
        if ip is None:
            continue
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
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
            pts = _shape_to_points(s)
            a = _polygon_area(pts)
            if a <= 0:
                continue
            n_objects += 1
            ratios.append(float(a / img_area))

    if n_images == 0:
        return "No readable image/json pairs found.", []

    arr = np.array(ratios, dtype=np.float64) if ratios else np.array([], dtype=np.float64)
    pct = lambda x: f"{100.0 * x:.4f}%"
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

    # Build random visualization samples.
    gallery: List[Tuple[np.ndarray, str]] = []
    sample_jfs = sorted(dd.glob("*.json"))
    rng.shuffle(sample_jfs)
    for jf in sample_jfs:
        if len(gallery) >= int(sample_count):
            break
        ip = _find_image_for_json(dd, jf.stem)
        if ip is None:
            continue
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        img_bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        polys = []
        for s in d.get("shapes", []) or []:
            if str(s.get("label", "")).strip().casefold() != label_cf:
                continue
            pts = _shape_to_points(s)
            if len(pts) >= 3:
                polys.append(pts)
        if not polys:
            continue
        view = _draw_polygons(img_bgr, polys, (0, 255, 255))
        gallery.append((view[:, :, ::-1], f"{ip.name} | objs={len(polys)}"))

    return "\n".join(summary), gallery


def make_app() -> gr.Blocks:
    with gr.Blocks(
        title="WTCV Studio",
        theme=gr.themes.Soft(),
        css="""
        :root {
            --wtcv-bg: #f6f8fb;
            --wtcv-panel: #ffffff;
            --wtcv-text: #171c26;
            --wtcv-muted: #6b7280;
            --wtcv-accent: #2b6df8;
            --wtcv-accent-2: #1447b8;
            --wtcv-border: #d6dce7;
            --wtcv-input: #ffffff;
            --wtcv-tab: #eff3f9;
            --wtcv-radius: 12px;
            --wtcv-shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
            --wtcv-atmo-1: rgba(43, 109, 248, 0.08);
            --wtcv-atmo-2: rgba(20, 71, 184, 0.04);
        }
        body.wtcv-dark {
            --wtcv-bg: #0f1420;
            --wtcv-panel: #121b2b;
            --wtcv-text: #e7ecf6;
            --wtcv-muted: #9aa8bd;
            --wtcv-accent: #6ea8ff;
            --wtcv-accent-2: #9ec2ff;
            --wtcv-border: #24334d;
            --wtcv-input: #0d1626;
            --wtcv-tab: #1a2437;
            --wtcv-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
            --wtcv-atmo-1: rgba(110, 168, 255, 0.12);
            --wtcv-atmo-2: rgba(27, 56, 107, 0.18);
        }
        :is(body, html, .gradio-container).wtcv-theme-ocean {
            --wtcv-accent: #0ea5b7;
            --wtcv-accent-2: #0a7c89;
            --wtcv-radius: 16px;
            --wtcv-atmo-1: rgba(14, 165, 183, 0.16);
            --wtcv-atmo-2: rgba(10, 124, 137, 0.14);
        }
        :is(body, html, .gradio-container).wtcv-theme-forest {
            --wtcv-accent: #2f9e44;
            --wtcv-accent-2: #227338;
            --wtcv-radius: 8px;
            --wtcv-atmo-1: rgba(47, 158, 68, 0.16);
            --wtcv-atmo-2: rgba(34, 115, 56, 0.14);
        }
        :is(body, html, .gradio-container).wtcv-theme-ember {
            --wtcv-accent: #e65f2b;
            --wtcv-accent-2: #b43f17;
            --wtcv-radius: 14px;
            --wtcv-atmo-1: rgba(230, 95, 43, 0.18);
            --wtcv-atmo-2: rgba(180, 63, 23, 0.14);
        }
        :is(body, html, .gradio-container).wtcv-theme-ocean:not(.wtcv-dark) {
            --wtcv-bg: #eefbfe;
            --wtcv-panel: #f8fdff;
            --wtcv-text: #0f2a31;
            --wtcv-muted: #4e6f77;
            --wtcv-border: #b8e3ea;
            --wtcv-input: #ffffff;
            --wtcv-tab: #e5f7fb;
        }
        :is(body, html, .gradio-container).wtcv-dark.wtcv-theme-ocean {
            --wtcv-bg: #06161e;
            --wtcv-panel: #0d2430;
            --wtcv-text: #dbf3f8;
            --wtcv-muted: #8cb5be;
            --wtcv-border: #1f4653;
            --wtcv-input: #0a1f29;
            --wtcv-tab: #12313d;
        }
        :is(body, html, .gradio-container).wtcv-theme-forest:not(.wtcv-dark) {
            --wtcv-bg: #f3faf2;
            --wtcv-panel: #fcfffb;
            --wtcv-text: #1d2d1f;
            --wtcv-muted: #5f7561;
            --wtcv-border: #cbe5cf;
            --wtcv-input: #ffffff;
            --wtcv-tab: #eaf6ec;
        }
        :is(body, html, .gradio-container).wtcv-dark.wtcv-theme-forest {
            --wtcv-bg: #101811;
            --wtcv-panel: #172419;
            --wtcv-text: #e2f3e4;
            --wtcv-muted: #9ab89e;
            --wtcv-border: #2f4633;
            --wtcv-input: #121e14;
            --wtcv-tab: #213124;
        }
        :is(body, html, .gradio-container).wtcv-theme-ember:not(.wtcv-dark) {
            --wtcv-bg: #fff4ef;
            --wtcv-panel: #fffdfc;
            --wtcv-text: #3a1f15;
            --wtcv-muted: #8b6252;
            --wtcv-border: #efc7b7;
            --wtcv-input: #ffffff;
            --wtcv-tab: #ffe9df;
        }
        :is(body, html, .gradio-container).wtcv-dark.wtcv-theme-ember {
            --wtcv-bg: #1b100c;
            --wtcv-panel: #281611;
            --wtcv-text: #ffe7dd;
            --wtcv-muted: #cba08f;
            --wtcv-border: #5d3528;
            --wtcv-input: #22130f;
            --wtcv-tab: #341d16;
        }
        body, .gradio-container {
            background: var(--wtcv-bg) !important;
            color: var(--wtcv-text) !important;
            transition: background-color .2s ease, color .2s ease;
        }
        .gradio-container::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
              radial-gradient(900px 500px at 15% -10%, var(--wtcv-atmo-1), transparent 60%),
              radial-gradient(900px 520px at 110% 0%, var(--wtcv-atmo-2), transparent 62%);
        }
        .gradio-container > * {
            position: relative;
            z-index: 1;
        }
        .gradio-container {
            --body-background-fill: var(--wtcv-bg) !important;
            --background-fill-primary: var(--wtcv-bg) !important;
            --background-fill-secondary: var(--wtcv-panel) !important;
            --block-background-fill: var(--wtcv-panel) !important;
            --block-border-color: var(--wtcv-border) !important;
            --body-text-color: var(--wtcv-text) !important;
            --body-text-color-subdued: var(--wtcv-muted) !important;
            --input-background-fill: var(--wtcv-input) !important;
            --input-border-color: var(--wtcv-border) !important;
            --button-primary-background-fill: var(--wtcv-accent) !important;
            --button-primary-background-fill-hover: var(--wtcv-accent-2) !important;
            --button-primary-border-color: var(--wtcv-accent) !important;
            --button-primary-text-color: #ffffff !important;
            --button-secondary-background-fill: var(--wtcv-panel) !important;
            --button-secondary-border-color: var(--wtcv-border) !important;
            --button-secondary-text-color: var(--wtcv-text) !important;
            --color-accent: var(--wtcv-accent) !important;
            --color-accent-soft: color-mix(in srgb, var(--wtcv-accent) 20%, transparent) !important;
        }
        .gradio-container .block,
        .gradio-container .gr-box,
        .gradio-container [class*="panel"],
        .gradio-container [class*="block"] {
            background: var(--wtcv-panel) !important;
            border-color: var(--wtcv-border) !important;
            color: var(--wtcv-text) !important;
            border-radius: var(--wtcv-radius) !important;
            box-shadow: var(--wtcv-shadow) !important;
        }
        .gradio-container input,
        .gradio-container textarea,
        .gradio-container select {
            background: var(--wtcv-input) !important;
            color: var(--wtcv-text) !important;
            border-color: var(--wtcv-border) !important;
            border-radius: calc(var(--wtcv-radius) - 4px) !important;
        }
        .gradio-container input[type="checkbox"],
        .gradio-container input[type="radio"] {
            accent-color: var(--wtcv-accent) !important;
            cursor: pointer;
        }
        .gradio-container button.primary,
        .gradio-container .gr-button-primary {
            background: var(--wtcv-accent) !important;
            border-color: var(--wtcv-accent) !important;
            color: #fff !important;
        }
        .gradio-container button.primary:hover,
        .gradio-container .gr-button-primary:hover {
            background: var(--wtcv-accent-2) !important;
            border-color: var(--wtcv-accent-2) !important;
        }
        .gradio-container [role="tab"],
        .gradio-container .tabitem {
            background: var(--wtcv-tab) !important;
            color: var(--wtcv-text) !important;
            border-color: var(--wtcv-border) !important;
            border-radius: calc(var(--wtcv-radius) - 6px) !important;
        }
        .gradio-container [role="tab"][aria-selected="true"] {
            border-bottom-color: var(--wtcv-accent) !important;
            color: var(--wtcv-accent) !important;
        }
        #top-title {
            font-size: 30px;
            font-weight: 700;
            color: var(--wtcv-accent);
            letter-spacing: .2px;
        }
        .wtcv-theme-controls {
            position: fixed;
            right: 16px;
            top: 12px;
            z-index: 10050;
            display: flex;
            gap: 8px;
            align-items: center;
            width: max-content !important;
            background: color-mix(in srgb, var(--wtcv-panel) 88%, transparent);
            border: 1px solid color-mix(in srgb, var(--wtcv-accent) 28%, #9aa5b1);
            border-radius: 999px;
            padding: 6px 8px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.16);
            backdrop-filter: blur(6px);
            pointer-events: none;
        }
        .wtcv-theme-controls > * { pointer-events: auto; }
        .wtcv-theme-controls .gr-button, .wtcv-theme-controls button {
            border: 1px solid color-mix(in srgb, var(--wtcv-accent) 45%, #8b9aad);
            background: var(--wtcv-panel);
            color: var(--wtcv-text);
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
        }
        .wtcv-theme-controls .gr-button:hover, .wtcv-theme-controls button:hover {
            border-color: var(--wtcv-accent);
            color: var(--wtcv-accent-2);
        }
        .wtcv-theme-controls .wtcv-chip {
            font-size: 12px;
            color: var(--wtcv-muted);
            padding: 0 6px 0 2px;
            user-select: none;
        }
        .mono textarea {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;}
        """,
    ) as app:
        gr.HTML(
            """
            <div class="wtcv-theme-controls" id="wtcv-theme-controls">
              <span class="wtcv-chip">UI</span>
              <button id="wtcv-mode-btn" type="button">Mode: Light</button>
              <button id="wtcv-theme-btn" type="button">Theme: Default</button>
            </div>
            """
        )

        app.load(fn=None, inputs=None, outputs=None, js=THEME_INIT_JS)

        gr.Markdown(
            """
            <div id="top-title">WTCV Studio</div>
            Unified control panel for training, evaluation, dataset conversion, augmentation, and inference preview.
            """,
        )

        with gr.Tabs():
            with gr.Tab("Train"):
                with gr.Row():
                    data_dir = gr.Textbox(value=str(ROOT / "data/record_pairs"), label="Data Dir")
                    output_dir = gr.Textbox(value=str(ROOT / "runs"), label="Output Dir")
                with gr.Row():
                    run_name = gr.Textbox(value="", label="Run Name")
                    resume_ckpt = gr.Textbox(value="", label="Resume Checkpoint (optional)")
                with gr.Row():
                    epochs = gr.Number(value=5, precision=0, label="Epochs")
                    subset_size = gr.Number(value=0, precision=0, label="Subset Size (0=all)")
                    label = gr.Textbox(value="vehicle", label="Label")
                    fp_label = gr.Textbox(value="fp", label="FP Label")
                with gr.Row():
                    tile_size = gr.Number(value=224, precision=0, label="Tile Size")
                    tile_stride = gr.Number(value=112, precision=0, label="Tile Stride")
                    tile_scales = gr.Textbox(value="1.0", label="Tile Scales")
                    seg_out_stride_train = gr.Number(value=4, precision=0, label="Seg Out Stride (fixed in script)")
                with gr.Row():
                    batch_size = gr.Number(value=8, precision=0, label="Batch Size")
                    num_workers = gr.Number(value=8, precision=0, label="Num Workers")
                    lr = gr.Number(value=1e-4, label="LR")
                    lr_min = gr.Number(value=1e-5, label="LR Min")
                    lr_scheduler = gr.Dropdown(choices=["none", "cosine"], value="cosine", label="LR Scheduler")
                with gr.Row():
                    dino_upsampler = gr.Dropdown(choices=["learned", "anyup"], value="learned", label="DINO Upsampler")
                    anyup_q_chunk_size = gr.Number(value=256, precision=0, label="AnyUp q_chunk_size")
                    head_type = gr.Dropdown(choices=["pointwise", "dwsep", "residual"], value="pointwise", label="Head Type")
                    weight_decay = gr.Number(value=1e-4, label="Weight Decay")
                with gr.Row():
                    use_tile_cls_head = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Head")
                    balance_train_50_50 = gr.Dropdown(choices=["on", "off"], value="on", label="Balance Train 50/50")
                    balance_val_50_50 = gr.Dropdown(choices=["on", "off"], value="on", label="Balance Val 50/50")
                    augment_low_vis = gr.Dropdown(choices=["on", "off"], value="off", label="Low-Vis Augment")
                    hard_negative_mining = gr.Dropdown(choices=["on", "off"], value="on", label="Hard Negative Mining")
                    trust_torch_hub_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo")
                with gr.Row():
                    tile_cls_weight = gr.Number(value=0.3, label="Tile Cls Weight")
                    hnm_hard_ratio = gr.Number(value=0.3, label="HNM Hard Ratio")
                    hnm_pool_frac = gr.Number(value=0.2, label="HNM Pool Frac")
                    val_interval = gr.Number(value=5, precision=0, label="Val Interval")
                    image_log_interval = gr.Number(value=1, precision=0, label="Image Log Interval")
                with gr.Row():
                    mcc_weight = gr.Number(value=0.4, label="MCC Weight")
                    mcc_warmup_epochs = gr.Number(value=3, precision=0, label="MCC Warmup Epochs")
                    bce_weight = gr.Number(value=0.5, label="BCE Weight")
                    boundary_weight = gr.Number(value=0.2, label="Boundary Weight")
                    use_fp_supervision = gr.Dropdown(choices=["on", "off"], value="on", label="Use FP Supervision")
                    fp_neg_weight = gr.Number(value=0.3, label="FP Neg Weight")
                    fp_neg_ratio = gr.Number(value=0.5, label="FP Neg Ratio (balanced neg)")
                train_btn = gr.Button("Run Training", variant="primary")
                train_cmd = gr.Textbox(label="Command", interactive=False)
                train_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)

                train_btn.click(
                    fn=run_train,
                    inputs=[
                        data_dir,
                        output_dir,
                        run_name,
                        resume_ckpt,
                        epochs,
                        subset_size,
                        label,
                        fp_label,
                        tile_size,
                        tile_stride,
                        tile_scales,
                        batch_size,
                        num_workers,
                        dino_upsampler,
                        anyup_q_chunk_size,
                        head_type,
                        use_tile_cls_head,
                        tile_cls_weight,
                        balance_train_50_50,
                        balance_val_50_50,
                        augment_low_vis,
                        hard_negative_mining,
                        hnm_hard_ratio,
                        hnm_pool_frac,
                        lr,
                        lr_scheduler,
                        lr_min,
                        weight_decay,
                        mcc_weight,
                        mcc_warmup_epochs,
                        bce_weight,
                        boundary_weight,
                        use_fp_supervision,
                        fp_neg_weight,
                        fp_neg_ratio,
                        val_interval,
                        image_log_interval,
                        trust_torch_hub_repo,
                    ],
                    outputs=[train_cmd, train_logs],
                )

            with gr.Tab("Evaluate"):
                with gr.Row():
                    eval_data_dir = gr.Textbox(value=str(ROOT / "data/record_pairs"), label="Data Dir")
                    eval_ckpt = gr.Textbox(value="", label="Checkpoint")
                    eval_label = gr.Textbox(value="vehicle", label="Label")
                with gr.Row():
                    eval_tile_size = gr.Number(value=224, precision=0, label="Tile Size")
                    eval_tile_stride = gr.Number(value=112, precision=0, label="Tile Stride")
                    eval_seg_out_stride = gr.Number(value=4, precision=0, label="Seg Out Stride")
                    eval_pred_threshold = gr.Number(value=0.5, label="Pred Threshold")
                    eval_workers = gr.Number(value=8, precision=0, label="Num Workers")
                    eval_trust_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo")
                eval_btn = gr.Button("Run Evaluation", variant="primary")
                eval_cmd = gr.Textbox(label="Command", interactive=False)
                eval_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)
                eval_btn.click(
                    fn=run_eval,
                    inputs=[
                        eval_data_dir,
                        eval_ckpt,
                        eval_label,
                        eval_tile_size,
                        eval_tile_stride,
                        eval_seg_out_stride,
                        eval_pred_threshold,
                        eval_workers,
                        eval_trust_repo,
                    ],
                    outputs=[eval_cmd, eval_logs],
                )

            with gr.Tab("SAM bbox->poly"):
                with gr.Row():
                    sam_input_dir = gr.Textbox(value=str(ROOT / "data/war_thunder_v1_test1_labelme_pairs"), label="Input Dir")
                    sam_output_dir = gr.Textbox(value=str(ROOT / "data/sam_box_to_poly"), label="Output Dir")
                    sam_model_id = gr.Textbox(value="facebook/sam-vit-base", label="Model ID")
                with gr.Row():
                    sam_device = gr.Textbox(value="cuda", label="Device")
                    sam_batch = gr.Number(value=4, precision=0, label="Image Batch Size")
                    sam_prompt_mode = gr.Dropdown(
                        choices=["bbox", "point"],
                        value="bbox",
                        label="Prompt Mode",
                    )
                    sam_load_workers = gr.Number(value=8, precision=0, label="Load Workers")
                    sam_min_poly = gr.Number(value=20.0, label="Min Poly Area")
                    sam_poly_eps = gr.Number(value=0.002, label="Poly Epsilon Frac")
                    sam_max_images = gr.Number(value=0, precision=0, label="Max Images (0=all)")
                    sam_overwrite = gr.Dropdown(choices=["on", "off"], value="off", label="Overwrite Output")
                sam_btn = gr.Button("Run SAM Conversion", variant="primary")
                sam_cmd = gr.Textbox(label="Command", interactive=False)
                sam_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)
                sam_btn.click(
                    fn=run_sam_convert,
                    inputs=[
                        sam_input_dir,
                        sam_output_dir,
                        sam_model_id,
                        sam_device,
                        sam_batch,
                        sam_prompt_mode,
                        sam_load_workers,
                        sam_min_poly,
                        sam_poly_eps,
                        sam_max_images,
                        sam_overwrite,
                    ],
                    outputs=[sam_cmd, sam_logs],
                )

            with gr.Tab("Augment"):
                with gr.Row():
                    aug_target_dir = gr.Textbox(value=str(ROOT / "data/record_pairs"), label="Target Dir")
                    aug_donor_dir = gr.Textbox(value=str(ROOT / "data/sam_box_to_poly"), label="Donor Dir")
                    aug_output_dir = gr.Textbox(value=str(ROOT / "data/record_pairs_augmented"), label="Output Dir")
                with gr.Row():
                    aug_target_label = gr.Textbox(value="vehicle", label="Target Label")
                    aug_donor_label = gr.Textbox(value="vehicle", label="Donor Label")
                    aug_seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    aug_min_paste = gr.Number(value=1, precision=0, label="Min Pastes / Image")
                    aug_max_paste = gr.Number(value=3, precision=0, label="Max Pastes / Image")
                    aug_max_images = gr.Number(value=0, precision=0, label="Max Target Images (0=all)")
                    aug_donor_max_images = gr.Number(value=0, precision=0, label="Donor Max Images (0=all)")
                with gr.Row():
                    aug_horizon = gr.Number(value=0.35, label="Placement Horizon Frac")
                    aug_tries = gr.Number(value=40, precision=0, label="Max Placement Tries")
                    aug_overlap = gr.Number(value=0.15, label="Max Overlap IoU")
                    aug_size_min = gr.Number(value=0.0001, label="Size Min Ratio")
                    aug_size_max = gr.Number(value=0.05, label="Size Max Ratio")
                with gr.Row():
                    aug_min_poly = gr.Number(value=12.0, label="Min Poly Area")
                    aug_poly_eps = gr.Number(value=0.0, label="Poly Epsilon Frac (0=raw contour)")
                    aug_feather = gr.Number(value=3, precision=0, label="Feather Radius")
                    aug_jpeg_min = gr.Number(value=55, precision=0, label="JPEG Q Min")
                    aug_jpeg_max = gr.Number(value=92, precision=0, label="JPEG Q Max")
                    aug_occ = gr.Number(value=0.45, label="Occlusion Prob")
                    aug_overwrite = gr.Dropdown(choices=["on", "off"], value="off", label="Overwrite Output")
                aug_btn = gr.Button("Run Augmentation", variant="primary")
                aug_cmd = gr.Textbox(label="Command", interactive=False)
                aug_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)
                aug_btn.click(
                    fn=run_augment,
                    inputs=[
                        aug_target_dir,
                        aug_donor_dir,
                        aug_output_dir,
                        aug_target_label,
                        aug_donor_label,
                        aug_seed,
                        aug_min_paste,
                        aug_max_paste,
                        aug_max_images,
                        aug_donor_max_images,
                        aug_horizon,
                        aug_tries,
                        aug_overlap,
                        aug_size_min,
                        aug_size_max,
                        aug_min_poly,
                        aug_poly_eps,
                        aug_feather,
                        aug_jpeg_min,
                        aug_jpeg_max,
                        aug_occ,
                        aug_overwrite,
                    ],
                    outputs=[aug_cmd, aug_logs],
                )

            with gr.Tab("Curation (CV2 UI)"):
                gr.Markdown(
                    "Launches `curate_model_predictions_to_labelme.py`. "
                    "This opens an OpenCV window on your desktop (`a/d/n/p/g/q` controls)."
                )
                with gr.Row():
                    cur_input = gr.Textbox(value=str(ROOT / "videos"), label="Input Dir")
                    cur_ckpt = gr.Textbox(value="", label="Checkpoint")
                    cur_out = gr.Textbox(value=str(ROOT / "data/helo_curated_labelme"), label="Output Dir")
                with gr.Row():
                    cur_label = gr.Textbox(value="vehicle", label="Label")
                    cur_tile = gr.Number(value=224, precision=0, label="Tile Size")
                    cur_stride = gr.Number(value=112, precision=0, label="Tile Stride")
                    cur_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride")
                    cur_thr = gr.Number(value=0.5, label="Pred Threshold")
                with gr.Row():
                    cur_gating = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Gating")
                    cur_tile_cls_thr = gr.Number(value=0.5, label="Tile Cls Threshold")
                    cur_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile Cls Mode")
                    cur_min_poly = gr.Number(value=20.0, label="Min Poly Area")
                    cur_eps = gr.Number(value=0.002, label="Poly Epsilon")
                    cur_max = gr.Number(value=0, precision=0, label="Max Images")
                    cur_start = gr.Number(value=0, precision=0, label="Start Index")
                    cur_save_preview = gr.Dropdown(choices=["on", "off"], value="off", label="Save Preview")
                cur_btn = gr.Button("Launch Curation", variant="primary")
                cur_cmd = gr.Textbox(label="Command", interactive=False)
                cur_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)
                cur_btn.click(
                    fn=run_curation,
                    inputs=[
                        cur_input,
                        cur_ckpt,
                        cur_out,
                        cur_label,
                        cur_tile,
                        cur_stride,
                        cur_seg_stride,
                        cur_thr,
                        cur_gating,
                        cur_tile_cls_thr,
                        cur_tile_cls_mode,
                        cur_min_poly,
                        cur_eps,
                        cur_max,
                        cur_start,
                        cur_save_preview,
                    ],
                    outputs=[cur_cmd, cur_logs],
                )

            with gr.Tab("Single Image Inference"):
                with gr.Row():
                    infer_image_path = gr.Textbox(value="", label="Image Path")
                    infer_ckpt = gr.Textbox(value="", label="Checkpoint")
                    infer_label = gr.Textbox(value="vehicle", label="Label")
                with gr.Row():
                    infer_tile = gr.Number(value=224, precision=0, label="Tile Size")
                    infer_stride = gr.Number(value=112, precision=0, label="Tile Stride")
                    infer_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride")
                    infer_thr = gr.Number(value=0.5, label="Pred Threshold")
                with gr.Row():
                    infer_gate = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Gating")
                    infer_tile_cls_thr = gr.Number(value=0.5, label="Tile Cls Threshold")
                    infer_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile Cls Mode")
                    infer_min_poly = gr.Number(value=20.0, label="Min Poly Area")
                    infer_eps = gr.Number(value=0.002, label="Poly Epsilon")
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

            with gr.Tab("Dataset Peek"):
                with gr.Row():
                    peek_dir = gr.Textbox(value=str(ROOT / "data/record_pairs"), label="Dataset Dir")
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

            with gr.Tab("Object UMAP (FiftyOne)"):
                gr.Markdown(
                    "Creates one sample per object from LabelMe pairs, computes masked DINO object embeddings on object-centric tiles, then runs UMAP + KMeans and writes a FiftyOne dataset."
                )
                with gr.Row():
                    fo_input_dir = gr.Textbox(value=str(ROOT / "data/record_pairs"), label="Input LabelMe Dir")
                    fo_dataset_name = gr.Textbox(value="wtcv_object_umap", label="FiftyOne Dataset Name")
                    fo_output_dir = gr.Textbox(value=str(ROOT / "outputs/fiftyone_object_umap"), label="Output Dir")
                with gr.Row():
                    fo_labels = gr.Textbox(value="vehicle,fp", label="Label Filter (comma-separated)")
                    fo_max_objects = gr.Number(value=0, precision=0, label="Max Objects (0=all)")
                    fo_tile_size = gr.Number(value=448, precision=0, label="Tile Size")
                    fo_context = gr.Number(value=2.0, label="Tile Context Scale")
                    fo_batch = gr.Number(value=12, precision=0, label="DINO Batch Size")
                with gr.Row():
                    fo_dino_model = gr.Textbox(value="dinov2_vits14", label="DINO Model")
                    fo_device = gr.Textbox(value="", label="Device (blank=auto)")
                    fo_umap_neighbors = gr.Number(value=30, precision=0, label="UMAP n_neighbors")
                    fo_umap_min_dist = gr.Number(value=0.05, label="UMAP min_dist")
                    fo_umap_metric = gr.Textbox(value="cosine", label="UMAP metric")
                with gr.Row():
                    fo_clusters = gr.Number(value=20, precision=0, label="KMeans Clusters")
                    fo_seed = gr.Number(value=42, precision=0, label="Seed")
                    fo_overwrite = gr.Dropdown(choices=["on", "off"], value="on", label="Overwrite Dataset")
                    fo_launch = gr.Dropdown(choices=["on", "off"], value="on", label="Launch FiftyOne App")
                    fo_trust_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo")
                fo_btn = gr.Button("Run Object UMAP + Build FiftyOne Dataset", variant="primary")
                fo_cmd = gr.Textbox(label="Command", interactive=False)
                fo_logs = gr.Textbox(label="Live Logs", lines=22, elem_classes=["mono"], interactive=False)
                fo_btn.click(
                    fn=run_object_umap,
                    inputs=[
                        fo_input_dir,
                        fo_dataset_name,
                        fo_output_dir,
                        fo_labels,
                        fo_max_objects,
                        fo_tile_size,
                        fo_context,
                        fo_batch,
                        fo_dino_model,
                        fo_device,
                        fo_umap_neighbors,
                        fo_umap_min_dist,
                        fo_umap_metric,
                        fo_clusters,
                        fo_seed,
                        fo_overwrite,
                        fo_launch,
                        fo_trust_repo,
                    ],
                    outputs=[fo_cmd, fo_logs],
                )

                gr.Markdown("Export reviewed/taged FiftyOne samples back to merged LabelMe image/json pairs (grouped by source image).")
                with gr.Row():
                    fo_exp_dataset_name = gr.Textbox(value="wtcv_object_umap", label="Dataset Name")
                    fo_exp_output_dir = gr.Textbox(value=str(ROOT / "data/umap_filtered_dataset"), label="Output LabelMe Dir")
                    fo_exp_tag_labels = gr.Textbox(value="vehicle,fp", label="Tag Labels (priority order)")
                    fo_exp_overwrite = gr.Dropdown(choices=["on", "off"], value="on", label="Overwrite Output")
                fo_exp_btn = gr.Button("Export Tagged -> LabelMe", variant="primary")
                fo_exp_cmd = gr.Textbox(label="Export Command", interactive=False)
                fo_exp_logs = gr.Textbox(label="Export Logs", lines=12, elem_classes=["mono"], interactive=False)
                fo_exp_btn.click(
                    fn=run_export_tagged_labelme,
                    inputs=[fo_exp_dataset_name, fo_exp_output_dir, fo_exp_tag_labels, fo_exp_overwrite],
                    outputs=[fo_exp_cmd, fo_exp_logs],
                )

            with gr.Tab("Video -> Frames"):
                with gr.Row():
                    vid_path = gr.Textbox(value="", label="Video Path")
                    vid_out = gr.Textbox(value=str(ROOT / "videos/frames_2fps"), label="Output Frames Dir")
                    vid_fps = gr.Number(value=2.0, label="FPS")
                    vid_overwrite = gr.Dropdown(choices=["on", "off"], value="off", label="Overwrite Existing Frames")
                vid_btn = gr.Button("Extract Frames", variant="primary")
                vid_cmd = gr.Textbox(label="Command", interactive=False)
                vid_logs = gr.Textbox(label="Live Logs", lines=20, elem_classes=["mono"], interactive=False)
                vid_btn.click(
                    fn=run_extract_frames,
                    inputs=[vid_path, vid_out, vid_fps, vid_overwrite],
                    outputs=[vid_cmd, vid_logs],
                )

            with gr.Tab("Media Source (CV2/Headless)"):
                gr.Markdown(
                    "Run tiled inference on either an image folder or a video file. "
                    "UI is optional and defaults to off for headless systems."
                )
                with gr.Row():
                    media_ckpt = gr.Textbox(value="", label="Checkpoint")
                    media_input = gr.Textbox(value="", label="Input Path (image folder or video file)")
                    media_label = gr.Textbox(value="vehicle", label="Label")
                    media_out = gr.Textbox(value=str(ROOT / "data/media_inference_labelme"), label="Output Dir")
                with gr.Row():
                    media_tile = gr.Number(value=448, precision=0, label="Tile Size")
                    media_stride = gr.Number(value=448, precision=0, label="Tile Stride")
                    media_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride")
                    media_thr = gr.Number(value=0.5, label="Pred Threshold")
                    media_fps = gr.Number(value=30.0, label="Max FPS")
                    media_infer_every = gr.Number(value=2, precision=0, label="Infer Every N")
                with gr.Row():
                    media_amp_mode = gr.Dropdown(choices=["amp", "no_amp"], value="amp", label="AMP Mode")
                    media_ui_mode = gr.Dropdown(choices=["off", "on"], value="off", label="UI (default off)")
                    media_gate = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Gating")
                    media_tile_cls_thr = gr.Number(value=0.5, label="Tile Cls Threshold")
                    media_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile Cls Mode")
                    media_min_poly = gr.Number(value=20.0, label="Min Poly Area")
                    media_eps = gr.Number(value=0.002, label="Poly Epsilon")
                with gr.Row():
                    media_auto_save = gr.Dropdown(choices=["on", "off"], value="on", label="Auto Save")
                    media_save_empty = gr.Dropdown(choices=["on", "off"], value="off", label="Save Empty")
                    media_save_preview = gr.Dropdown(choices=["on", "off"], value="off", label="Save Preview")
                    media_start = gr.Number(value=0, precision=0, label="Start Index")
                    media_max_items = gr.Number(value=0, precision=0, label="Max Items (0=all)")
                media_btn = gr.Button("Run Media Source Inference", variant="primary")
                media_cmd = gr.Textbox(label="Command", interactive=False)
                media_logs = gr.Textbox(label="Live Logs", lines=20, elem_classes=["mono"], interactive=False)
                media_btn.click(
                    fn=run_media_source,
                    inputs=[
                        media_ckpt,
                        media_input,
                        media_label,
                        media_out,
                        media_tile,
                        media_stride,
                        media_seg_stride,
                        media_thr,
                        media_gate,
                        media_tile_cls_thr,
                        media_tile_cls_mode,
                        media_min_poly,
                        media_eps,
                        media_fps,
                        media_infer_every,
                        media_amp_mode,
                        media_ui_mode,
                        media_auto_save,
                        media_save_empty,
                        media_save_preview,
                        media_start,
                        media_max_items,
                    ],
                    outputs=[media_cmd, media_logs],
                )

            with gr.Tab("Live Screen (CV2 UI)"):
                gr.Markdown(
                    "Launches `live_screen_inference_cv2.py` for live screen capture inference. "
                    "Controls in OpenCV window: `q`, `space`, `+/-`, `[ ]`, `g`, `m`, `a`."
                )
                with gr.Row():
                    live_ckpt = gr.Textbox(value="", label="Checkpoint")
                    live_label = gr.Textbox(value="vehicle", label="Label")
                    live_output_dir = gr.Textbox(value=str(ROOT / "data/live_screen_labelme"), label="Output Dir (for key 'a')")
                with gr.Row():
                    live_monitor_idx = gr.Number(value=1, precision=0, label="Monitor Index")
                    live_x = gr.Number(value=0, precision=0, label="Region X")
                    live_y = gr.Number(value=0, precision=0, label="Region Y")
                    live_w = gr.Number(value=0, precision=0, label="Region Width (0=full)")
                    live_h = gr.Number(value=0, precision=0, label="Region Height (0=full)")
                with gr.Row():
                    live_tile = gr.Number(value=448, precision=0, label="Tile Size")
                    live_stride = gr.Number(value=448, precision=0, label="Tile Stride")
                    live_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride")
                    live_thr = gr.Number(value=0.5, label="Pred Threshold")
                    live_fps = gr.Number(value=30.0, label="Max FPS")
                    live_infer_every = gr.Number(value=2, precision=0, label="Infer Every N Frames")
                with gr.Row():
                    live_amp_mode = gr.Dropdown(
                        choices=["amp", "no_amp"],
                        value="amp",
                        label="AMP Mode",
                    )
                    live_gate = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Gating")
                    live_tile_cls_thr = gr.Number(value=0.5, label="Tile Cls Threshold")
                    live_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile Cls Mode")
                    live_min_poly = gr.Number(value=20.0, label="Min Poly Area")
                    live_eps = gr.Number(value=0.002, label="Poly Epsilon")
                    live_save_preview = gr.Dropdown(choices=["on", "off"], value="off", label="Save Preview on key 'a'")
                    live_print_monitors = gr.Dropdown(choices=["on", "off"], value="off", label="Print Monitors & Exit")
                with gr.Row():
                    live_auto_save_persistent = gr.Dropdown(choices=["on", "off"], value="off", label="Auto-save Persistent Detections")
                    live_persist_infers = gr.Number(value=3, precision=0, label="Persistent N (inference steps)")
                    live_persist_iou = gr.Number(value=0.25, label="Persistence IoU Threshold")
                    live_persist_max_miss = gr.Number(value=1, precision=0, label="Persistence Max Miss")
                    live_persist_cooldown = gr.Number(value=8, precision=0, label="Auto-save Cooldown (inferences)")
                live_btn = gr.Button("Launch Live Screen Inference", variant="primary")
                live_cmd = gr.Textbox(label="Command", interactive=False)
                live_logs = gr.Textbox(label="Live Logs", lines=20, elem_classes=["mono"], interactive=False)
                live_btn.click(
                    fn=run_live_screen,
                    inputs=[
                        live_ckpt,
                        live_label,
                        live_monitor_idx,
                        live_x,
                        live_y,
                        live_w,
                        live_h,
                        live_tile,
                        live_stride,
                        live_seg_stride,
                        live_thr,
                        live_gate,
                        live_tile_cls_thr,
                        live_tile_cls_mode,
                        live_min_poly,
                        live_eps,
                        live_fps,
                        live_infer_every,
                        live_amp_mode,
                        live_auto_save_persistent,
                        live_persist_infers,
                        live_persist_iou,
                        live_persist_max_miss,
                        live_persist_cooldown,
                        live_output_dir,
                        live_save_preview,
                        live_print_monitors,
                    ],
                    outputs=[live_cmd, live_logs],
                )

    return app


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WTCV Studio Gradio app")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", default=False)
    ap.add_argument("--no-queue", action="store_true", default=False)
    return ap.parse_args()


def main() -> None:
    args = parse_cli()
    app = make_app()
    if args.no_queue:
        app.launch(server_name=args.host, server_port=args.port, share=args.share)
    else:
        app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
