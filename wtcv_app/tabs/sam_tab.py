from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, stream_command


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
    overwrite = as_bool(overwrite)
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
    yield from stream_command(cmd)


def build_tab(root: Path) -> None:
    with gr.Tab("SAM bbox->poly"):
        with gr.Row():
            sam_input_dir = gr.Textbox(value=str(root / "data/war_thunder_v1_test1_labelme_pairs"), label="Input Dir")
            sam_output_dir = gr.Textbox(value=str(root / "data/sam_box_to_poly"), label="Output Dir")
            sam_model_id = gr.Textbox(value="facebook/sam-vit-base", label="Model ID")
        with gr.Row():
            sam_device = gr.Textbox(value="cuda", label="Device")
            sam_batch = gr.Number(value=4, precision=0, label="Image Batch Size")
            sam_prompt_mode = gr.Dropdown(choices=["bbox", "point"], value="bbox", label="Prompt Mode")
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
