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


def build_tab(root: Path) -> None:
    with gr.Tab("Live Screen (CV2 UI)"):
        gr.Markdown(
            "Launches `live_screen_inference_cv2.py` for live screen capture inference. "
            "Controls in OpenCV window: `q`, `space`, `+/-`, `[ ]`, `g`, `m`, `a`."
        )
        with gr.Row():
            live_ckpt = gr.Textbox(value="", label="Checkpoint")
            live_label = gr.Textbox(value="vehicle", label="Label")
            live_output_dir = gr.Textbox(value=str(root / "data/live_screen_labelme"), label="Output Dir (for key 'a')")
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
            live_amp_mode = gr.Dropdown(choices=["amp", "no_amp"], value="amp", label="AMP Mode")
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
