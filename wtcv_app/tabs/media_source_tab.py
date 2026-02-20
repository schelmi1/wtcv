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
    save_preview: bool,
    auto_save_persistent: bool,
    persist_infers: int,
    persist_iou_threshold: float,
    persist_max_miss: int,
    persist_save_cooldown_infers: int,
    start_index: int,
    max_items: int,
) -> Generator[Tuple[str, str], None, None]:
    use_tile_cls_gating = as_bool(use_tile_cls_gating)
    save_preview = as_bool(save_preview)
    auto_save_persistent = as_bool(auto_save_persistent)
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
        "--persist-infers",
        str(int(persist_infers)),
        "--persist-iou-threshold",
        str(float(persist_iou_threshold)),
        "--persist-max-miss",
        str(int(persist_max_miss)),
        "--persist-save-cooldown-infers",
        str(int(persist_save_cooldown_infers)),
    ]
    cmd += build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    if save_preview:
        cmd += ["--save-preview"]
    if auto_save_persistent:
        cmd += ["--auto-save-persistent"]
    if str(amp_mode).strip().lower() == "amp":
        cmd += ["--amp"]
    else:
        cmd += ["--no-amp"]
    if ui_mode == "on":
        cmd += ["--ui"]
    else:
        cmd += ["--no-ui"]
    yield from stream_command(cmd)


def build_content(root: Path) -> None:
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
        media_save_preview = gr.Dropdown(choices=["on", "off"], value="off", label="Save Preview", info="Save visualization preview images alongside LabelMe outputs.")
        media_auto_save_persistent = gr.Dropdown(choices=["on", "off"], value="off", label="Auto-save Persistent Detections", info="Automatically save detections that persist across inference steps.")
        media_persist_infers = gr.Number(value=3, precision=0, label="Persistent N (inference steps)", info="Minimum consecutive inference hits required before auto-save.")
        media_persist_iou = gr.Number(value=0.25, label="Persistence IoU Threshold", info="IoU needed to match detections across inference steps.")
        media_persist_max_miss = gr.Number(value=1, precision=0, label="Persistence Max Miss", info="Max unmatched inference steps before a persistent track is dropped.")
        media_persist_cooldown = gr.Number(value=8, precision=0, label="Auto-save Cooldown (inferences)", info="Inference-step cooldown between automatic persistent saves.")
    with gr.Row():
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
            media_save_preview,
            media_auto_save_persistent,
            media_persist_infers,
            media_persist_iou,
            media_persist_max_miss,
            media_persist_cooldown,
            media_start,
            media_max_items,
        ],
        outputs=[media_cmd, media_logs],
    )


def build_tab(root: Path, nested: bool = False) -> None:
    if nested:
        build_content(root)
        return
    with gr.Tab("Media Source (CV2/Headless)"):
        build_content(root)
