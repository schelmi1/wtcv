from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


def run_batched_image_infer(
    checkpoint: str,
    input_path: str,
    output_dir: str,
    label: str,
    tile_size: int,
    tile_stride: int,
    seg_out_stride: int,
    tile_batch_size: int,
    num_workers: int,
    pred_threshold: float,
    use_tile_cls_gating: bool,
    tile_cls_threshold: float,
    tile_cls_mode: str,
    min_poly_area: float,
    poly_epsilon_frac: float,
    start_index: int,
    max_images: int,
    recursive: bool,
    amp_mode: str,
    save_empty: bool,
    overwrite: bool,
) -> Generator[Tuple[str, str], None, None]:
    use_tile_cls_gating = as_bool(use_tile_cls_gating)
    recursive = as_bool(recursive)
    save_empty = as_bool(save_empty)
    overwrite = as_bool(overwrite)

    cmd = [
        sys.executable,
        str(ROOT / "batch_image_folder_inference.py"),
        "--checkpoint",
        checkpoint,
        "--input-path",
        input_path,
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
        "--tile-batch-size",
        str(int(tile_batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--pred-threshold",
        str(float(pred_threshold)),
        "--tile-cls-threshold",
        str(float(tile_cls_threshold)),
        "--tile-cls-mode",
        str(tile_cls_mode),
        "--min-poly-area",
        str(float(min_poly_area)),
        "--poly-epsilon-frac",
        str(float(poly_epsilon_frac)),
        "--start-index",
        str(int(start_index)),
        "--max-images",
        str(int(max_images)),
    ]
    cmd += build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    if recursive:
        cmd += ["--recursive"]
    if str(amp_mode).strip().lower() == "amp":
        cmd += ["--amp"]
    else:
        cmd += ["--no-amp"]
    if save_empty:
        cmd += ["--save-empty"]
    if overwrite:
        cmd += ["--overwrite"]
    yield from stream_command(cmd)


def build_content(root: Path) -> None:
    gr.Markdown(
        "High-throughput image-folder inference using a tile-stream dataloader. "
        "Tiles from multiple images are batched together for faster GPU utilization. "
        "No OpenCV UI."
    )
    with gr.Row():
        bi_ckpt = gr.Textbox(value="", label="Checkpoint", info="Path to a trained model checkpoint (.pt).")
        bi_input = gr.Textbox(value=str(root / "data/record_pairs"), label="Input Path", info="Image directory (or single image file).")
        bi_out = gr.Textbox(value=str(root / "data/batch_inference_labelme"), label="Output Dir", info="Directory where LabelMe outputs and summary files are written.")
        bi_label = gr.Textbox(value="vehicle", label="Label", info="Label name used for saved polygons.")
    with gr.Row():
        bi_tile = gr.Number(value=512, precision=0, label="Tile Size", info="Tile side length. Must be a multiple of 256.")
        bi_stride = gr.Number(value=512, precision=0, label="Tile Stride", info="Tile step between origins.")
        bi_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride", info="Model output stride for stitching.")
        bi_tile_batch = gr.Number(value=32, precision=0, label="Tile Batch Size", info="Number of tiles per GPU forward pass.")
        bi_workers = gr.Number(value=8, precision=0, label="Num Workers", info="DataLoader workers for image decode/tiling.")
        bi_thr = gr.Number(value=0.5, label="Pred Threshold", info="Binary threshold for final masks.")
    with gr.Row():
        bi_gate = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Gating", info="Apply tile classifier gating, if available in checkpoint.")
        bi_tile_cls_thr = gr.Number(value=0.5, label="Tile Cls Threshold", info="Tile classifier threshold for hard gating.")
        bi_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile Cls Mode", info="Gating behavior.")
        bi_min_poly = gr.Number(value=20.0, label="Min Poly Area", info="Minimum polygon area to keep.")
        bi_eps = gr.Number(value=0.002, label="Poly Epsilon", info="Contour simplification epsilon fraction.")
        bi_amp = gr.Dropdown(choices=["amp", "no_amp"], value="amp", label="AMP Mode", info="Use mixed precision or full precision.")
    with gr.Row():
        bi_start = gr.Number(value=0, precision=0, label="Start Index", info="Start from this image index.")
        bi_max = gr.Number(value=0, precision=0, label="Max Images (0=all)", info="Maximum images to process.")
        bi_recursive = gr.Dropdown(choices=["on", "off"], value="off", label="Recursive", info="If on, search subdirectories recursively.")
        bi_save_empty = gr.Dropdown(choices=["on", "off"], value="off", label="Save Empty Outputs", info="If on, save images/json with no polygons too.")
        bi_overwrite = gr.Dropdown(choices=["on", "off"], value="off", label="Overwrite Existing", info="If on, existing outputs can be replaced.")
    bi_btn = gr.Button("Run Batched Image Inference", variant="primary")
    bi_cmd = gr.Textbox(label="Command", interactive=False)
    bi_logs = gr.Textbox(label="Live Logs", lines=22, elem_classes=["mono"], interactive=False)
    bi_btn.click(
        fn=run_batched_image_infer,
        inputs=[
            bi_ckpt,
            bi_input,
            bi_out,
            bi_label,
            bi_tile,
            bi_stride,
            bi_seg_stride,
            bi_tile_batch,
            bi_workers,
            bi_thr,
            bi_gate,
            bi_tile_cls_thr,
            bi_tile_cls_mode,
            bi_min_poly,
            bi_eps,
            bi_start,
            bi_max,
            bi_recursive,
            bi_amp,
            bi_save_empty,
            bi_overwrite,
        ],
        outputs=[bi_cmd, bi_logs],
    )


def build_tab(root: Path, nested: bool = False) -> None:
    if nested:
        build_content(root)
        return
    with gr.Tab("Batched Image Inference"):
        build_content(root)
