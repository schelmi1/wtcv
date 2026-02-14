from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


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
    use_tile_cls_gating = as_bool(use_tile_cls_gating)
    save_preview = as_bool(save_preview)
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
    cmd += build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    if save_preview:
        cmd += ["--save-preview"]
    yield from stream_command(cmd)


def build_tab(root: Path) -> None:
    with gr.Tab("Curation (CV2 UI)"):
        gr.Markdown(
            "Launches `curate_model_predictions_to_labelme.py`. "
            "This opens an OpenCV window on your desktop (`a/d/n/p/g/q` controls)."
        )
        with gr.Row():
            cur_input = gr.Textbox(value=str(root / "videos"), label="Input Dir")
            cur_ckpt = gr.Textbox(value="", label="Checkpoint")
            cur_out = gr.Textbox(value=str(root / "data/helo_curated_labelme"), label="Output Dir")
        with gr.Row():
            cur_label = gr.Textbox(value="vehicle", label="Label")
            cur_tile = gr.Number(value=256, precision=0, label="Tile Size")
            cur_stride = gr.Number(value=128, precision=0, label="Tile Stride")
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
