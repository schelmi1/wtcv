from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


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
    use_tile_cls_gating = as_bool(use_tile_cls_gating)
    auto_save_persistent = as_bool(auto_save_persistent)
    save_preview = as_bool(save_preview)
    print_monitors = as_bool(print_monitors)
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
    cmd += build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
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
    yield from stream_command(cmd)


def build_content(root: Path) -> None:
    gr.Markdown(
        "Launches `live_screen_inference_cv2.py` for live screen capture inference. "
        "Controls in OpenCV window: `q`, `space`, `+/-`, `[ ]`, `g`, `m`, `a`."
    )
    with gr.Row():
        live_ckpt = gr.Textbox(value="", label="Checkpoint", info="Path to a trained model checkpoint (.pt) to load for inference/evaluation.")
        live_label = gr.Textbox(value="vehicle", label="Label", info="Class label name used for output polygons and evaluation target.")
        live_output_dir = gr.Textbox(value=str(root / "data/live_screen_labelme"), label="Output Dir (for key 'a')", info="Directory used when saving detections via key \"a\" in live UI.")
    with gr.Row():
        live_monitor_idx = gr.Number(value=1, precision=0, label="Monitor Index", info="Index of monitor to capture for live screen inference.")
        live_x = gr.Number(value=0, precision=0, label="Region X", info="Left coordinate of screen capture region in pixels.")
        live_y = gr.Number(value=0, precision=0, label="Region Y", info="Top coordinate of screen capture region in pixels.")
        live_w = gr.Number(value=0, precision=0, label="Region Width (0=full)", info="Capture width in pixels; 0 uses full monitor width.")
        live_h = gr.Number(value=0, precision=0, label="Region Height (0=full)", info="Capture height in pixels; 0 uses full monitor height.")
    with gr.Row():
        live_tile = gr.Number(value=512, precision=0, label="Tile Size", info="Side length of each square inference/training tile in pixels.")
        live_stride = gr.Number(value=512, precision=0, label="Tile Stride", info="Step size between tile origins; lower values add overlap and compute cost.")
        live_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride", info="Output stride of segmentation logits relative to tile resolution.")
        live_thr = gr.Number(value=0.5, label="Pred Threshold", info="Probability threshold used to convert logits/probabilities into a binary mask.")
        live_fps = gr.Number(value=30.0, label="Max FPS", info="Upper bound on processing/display frame rate.")
        live_infer_every = gr.Number(value=2, precision=0, label="Infer Every N Frames", info="Run model inference once every N captured frames.")
    with gr.Row():
        live_amp_mode = gr.Dropdown(choices=["amp", "no_amp"], value="amp", label="AMP Mode", info="Use mixed precision (amp) or full precision (no_amp).")
        live_gate = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Gating", info="If on, tile classification score gates segmentation output per tile.")
        live_tile_cls_thr = gr.Number(value=0.5, label="Tile Cls Threshold", info="Minimum tile classification confidence required for gating.")
        live_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile Cls Mode", info="Gating behavior: hard masking or probability multiplication.")
        live_min_poly = gr.Number(value=20.0, label="Min Poly Area", info="Minimum polygon area kept during mask-to-polygon conversion.")
        live_eps = gr.Number(value=0.002, label="Poly Epsilon", info="Polygon simplification epsilon fraction used by contour approximation.")
        live_save_preview = gr.Dropdown(choices=["on", "off"], value="off", label="Save Preview on key 'a'", info="If on, also save overlay preview images when pressing \"a\".")
        live_print_monitors = gr.Dropdown(choices=["on", "off"], value="off", label="Print Monitors & Exit", info="Print monitor geometry list then exit without inference.")
    with gr.Row():
        live_auto_save_persistent = gr.Dropdown(choices=["on", "off"], value="off", label="Auto-save Persistent Detections", info="Automatically save detections that persist across inference steps.")
        live_persist_infers = gr.Number(value=3, precision=0, label="Persistent N (inference steps)", info="Minimum consecutive inference hits required before auto-save.")
        live_persist_iou = gr.Number(value=0.25, label="Persistence IoU Threshold", info="IoU needed to match detections across inference steps.")
        live_persist_max_miss = gr.Number(value=1, precision=0, label="Persistence Max Miss", info="Max unmatched inference steps before a persistent track is dropped.")
        live_persist_cooldown = gr.Number(value=8, precision=0, label="Auto-save Cooldown (inferences)", info="Inference-step cooldown between automatic saves.")
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


def build_tab(root: Path, nested: bool = False) -> None:
    if nested:
        build_content(root)
        return
    with gr.Tab("Live Screen (CV2 UI)"):
        build_content(root)
