from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


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
    use_tile_cls_gating = as_bool(use_tile_cls_gating)
    auto_save = as_bool(auto_save)
    save_empty = as_bool(save_empty)
    save_preview = as_bool(save_preview)
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
    cmd += build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    cmd += build_bool_arg("--auto-save", "--no-auto-save", bool(auto_save))
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
    yield from stream_command(cmd)


def build_tab(root: Path) -> None:
    with gr.Tab("Media Source (CV2/Headless)"):
        gr.Markdown(
            "Run tiled inference on either an image folder or a video file. "
            "UI is optional and defaults to off for headless systems."
        )
        with gr.Row():
            media_ckpt = gr.Textbox(value="", label="Checkpoint")
            media_input = gr.Textbox(value="", label="Input Path (image folder or video file)")
            media_label = gr.Textbox(value="vehicle", label="Label")
            media_out = gr.Textbox(value=str(root / "data/media_inference_labelme"), label="Output Dir")
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
