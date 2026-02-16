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
            media_ckpt = gr.Textbox(value="", label="Checkpoint", info="Path to a trained model checkpoint (.pt) to load for inference/evaluation.")
            media_input = gr.Textbox(value="", label="Input Path (image folder or video file)", info="Input source path: image directory or single video file.")
            media_label = gr.Textbox(value="vehicle", label="Label", info="Class label name used for output polygons and evaluation target.")
            media_out = gr.Textbox(value=str(root / "data/media_inference_labelme"), label="Output Dir", info="Directory where generated outputs are written.")
        with gr.Row():
            media_tile = gr.Number(value=448, precision=0, label="Tile Size", info="Side length of each square inference/training tile in pixels.")
            media_stride = gr.Number(value=448, precision=0, label="Tile Stride", info="Step size between tile origins; lower values add overlap and compute cost.")
            media_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride", info="Output stride of segmentation logits relative to tile resolution.")
            media_thr = gr.Number(value=0.5, label="Pred Threshold", info="Probability threshold used to convert logits/probabilities into a binary mask.")
            media_fps = gr.Number(value=30.0, label="Max FPS", info="Upper bound on processing/display frame rate.")
            media_infer_every = gr.Number(value=2, precision=0, label="Infer Every N", info="Run inference once every N source frames/images.")
        with gr.Row():
            media_amp_mode = gr.Dropdown(choices=["amp", "no_amp"], value="amp", label="AMP Mode", info="Use mixed precision (amp) or full precision (no_amp).")
            media_ui_mode = gr.Dropdown(choices=["off", "on"], value="off", label="UI (default off)", info="Enable interactive OpenCV UI; off runs headless.")
            media_gate = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Gating", info="If on, tile classification score gates segmentation output per tile.")
            media_tile_cls_thr = gr.Number(value=0.5, label="Tile Cls Threshold", info="Minimum tile classification confidence required for gating.")
            media_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile Cls Mode", info="Gating behavior: hard masking or probability multiplication.")
            media_min_poly = gr.Number(value=20.0, label="Min Poly Area", info="Minimum polygon area kept during mask-to-polygon conversion.")
            media_eps = gr.Number(value=0.002, label="Poly Epsilon", info="Polygon simplification epsilon fraction used by contour approximation.")
        with gr.Row():
            media_auto_save = gr.Dropdown(choices=["on", "off"], value="on", label="Auto Save", info="Automatically write LabelMe outputs for processed inputs.")
            media_save_empty = gr.Dropdown(choices=["on", "off"], value="off", label="Save Empty", info="Also save outputs for frames/images with no detections.")
            media_save_preview = gr.Dropdown(choices=["on", "off"], value="off", label="Save Preview", info="Save visualization preview images alongside LabelMe outputs.")
            media_start = gr.Number(value=0, precision=0, label="Start Index", info="Start processing from this item/frame index.")
            media_max_items = gr.Number(value=0, precision=0, label="Max Items (0=all)", info="Maximum items/frames to process; 0 means all available.")
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
