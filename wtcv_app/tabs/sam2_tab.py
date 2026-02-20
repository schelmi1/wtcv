from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, stream_command


def run_sam2_convert(
    input_dir: str,
    output_dir: str,
    label_filter: str,
    model_id: str,
    device: str,
    image_batch_size: int,
    prompt_mode: str,
    inference_mode: str,
    crop_size: int,
    load_workers: int,
    min_poly_area: float,
    poly_epsilon_frac: float,
    max_images: int,
    overwrite: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite = as_bool(overwrite)
    cmd = [
        sys.executable,
        str(ROOT / "sam2_box_to_poly_batched.py"),
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--label-filter",
        str(label_filter),
        "--model-id",
        model_id,
        "--device",
        device,
        "--image-batch-size",
        str(int(image_batch_size)),
        "--prompt-mode",
        str(prompt_mode).strip().lower(),
        "--inference-mode",
        str(inference_mode).strip().lower(),
        "--crop-size",
        str(int(crop_size)),
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
    with gr.Tab("SAM2 bbox->poly"):
        with gr.Row():
            sam_input_dir = gr.Textbox(
                value=str(root / "data/war_thunder_v1_test1_labelme_pairs"),
                label="Input Dir",
                info="Input directory read by the underlying script.",
            )
            sam_output_dir = gr.Textbox(
                value=str(root / "data/sam2_box_to_poly"),
                label="Output Dir",
                info="Directory where generated outputs are written.",
            )
            sam_label_filter = gr.Textbox(
                value="",
                label="Label Filter (blank=all)",
                info="Only refine these labels (comma-separated, case-insensitive).",
            )
            sam_model_id = gr.Textbox(
                value="facebook/sam2-hiera-small",
                label="Model ID",
                info="Hugging Face SAM2 model identifier loaded for conversion.",
            )
        with gr.Row():
            sam_device = gr.Textbox(value="cuda", label="Device", info="Execution device string (e.g. cuda, cpu).")
            sam_batch = gr.Number(value=4, precision=0, label="Image Batch Size", info="Number of images processed per SAM forward pass.")
            sam_prompt_mode = gr.Dropdown(choices=["bbox", "point"], value="point", label="Prompt Mode", info="Prompt type passed to SAM2 (bounding box or point prompt).")
            sam_inference_mode = gr.Dropdown(choices=["whole", "crop"], value="whole", label="Inference Mode", info="Run SAM2 on whole image or object-centered fixed-size crop.")
            sam_crop_size = gr.Number(value=512, precision=0, label="Crop Size", info="Square crop side used when Inference Mode is 'crop'.")
            sam_load_workers = gr.Number(value=8, precision=0, label="Load Workers", info="Number of worker threads/processes for reading dataset items.")
            sam_min_poly = gr.Number(value=20.0, label="Min Poly Area", info="Minimum polygon area kept during mask-to-polygon conversion.")
            sam_poly_eps = gr.Number(value=0.002, label="Poly Epsilon Frac", info="Contour simplification factor applied before writing polygons.")
            sam_max_images = gr.Number(value=0, precision=0, label="Max Images (0=all)", info="Maximum images to process; 0 means process the full dataset.")
            sam_overwrite = gr.Dropdown(choices=["on", "off"], value="off", label="Overwrite Output", info="If on, existing output files/directories may be replaced.")
        sam_btn = gr.Button("Run SAM2 Conversion", variant="primary")
        sam_cmd = gr.Textbox(label="Command", interactive=False)
        sam_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)
        sam_btn.click(
            fn=run_sam2_convert,
            inputs=[
                sam_input_dir,
                sam_output_dir,
                sam_label_filter,
                sam_model_id,
                sam_device,
                sam_batch,
                sam_prompt_mode,
                sam_inference_mode,
                sam_crop_size,
                sam_load_workers,
                sam_min_poly,
                sam_poly_eps,
                sam_max_images,
                sam_overwrite,
            ],
            outputs=[sam_cmd, sam_logs],
        )
