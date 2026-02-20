from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, stream_command


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
    overwrite = as_bool(overwrite)
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
    yield from stream_command(cmd)


def build_content(root: Path) -> None:
    with gr.Row():
        aug_target_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Target Dir", info="Target dataset directory whose images will receive pasted objects.")
        aug_donor_dir = gr.Textbox(value=str(root / "data/sam_box_to_poly"), label="Donor Dir", info="Donor dataset directory used to source polygon objects.")
        aug_output_dir = gr.Textbox(value=str(root / "data/record_pairs_augmented"), label="Output Dir", info="Directory where generated outputs are written.")
    with gr.Row():
        aug_target_label = gr.Textbox(value="vehicle", label="Target Label", info="Label in target dataset used for placement constraints/statistics.")
        aug_donor_label = gr.Textbox(value="vehicle", label="Donor Label", info="Label class sampled from donor polygons.")
        aug_seed = gr.Number(value=42, precision=0, label="Seed", info="Random seed for reproducible sampling and clustering behavior.")
    with gr.Row():
        aug_min_paste = gr.Number(value=1, precision=0, label="Min Pastes / Image", info="Minimum number of pasted objects per augmented target image.")
        aug_max_paste = gr.Number(value=3, precision=0, label="Max Pastes / Image", info="Maximum number of pasted objects per augmented target image.")
        aug_max_images = gr.Number(value=0, precision=0, label="Max Target Images (0=all)", info="Maximum target images to augment; 0 means all.")
        aug_donor_max_images = gr.Number(value=0, precision=0, label="Donor Max Images (0=all)", info="Maximum donor images to scan; 0 means all.")
    with gr.Row():
        aug_horizon = gr.Number(value=0.35, label="Placement Horizon Frac", info="Preferred vertical placement bias as image-height fraction.")
        aug_tries = gr.Number(value=40, precision=0, label="Max Placement Tries", info="Maximum random placement attempts per paste candidate.")
        aug_overlap = gr.Number(value=0.15, label="Max Overlap IoU", info="Maximum allowed IoU overlap between pasted and existing objects.")
        aug_size_min = gr.Number(value=0.0001, label="Size Min Ratio", info="Minimum pasted object area ratio relative to target image area.")
        aug_size_max = gr.Number(value=0.05, label="Size Max Ratio", info="Maximum pasted object area ratio relative to target image area.")
    with gr.Row():
        aug_min_poly = gr.Number(value=12.0, label="Min Poly Area", info="Minimum polygon area kept during mask-to-polygon conversion.")
        aug_poly_eps = gr.Number(value=0.0, label="Poly Epsilon Frac (0=raw contour)", info="Contour simplification factor; 0 keeps raw contour points.")
        aug_feather = gr.Number(value=3, precision=0, label="Feather Radius", info="Alpha feather radius in pixels for smoother paste blending.")
        aug_jpeg_min = gr.Number(value=55, precision=0, label="JPEG Q Min", info="Minimum JPEG recompression quality used for appearance perturbation.")
        aug_jpeg_max = gr.Number(value=92, precision=0, label="JPEG Q Max", info="Maximum JPEG recompression quality used for appearance perturbation.")
        aug_occ = gr.Number(value=0.45, label="Occlusion Prob", info="Probability of applying synthetic occlusion to pasted objects.")
        aug_overwrite = gr.Dropdown(choices=["on", "off"], value="off", label="Overwrite Output", info="If on, existing output files/directories may be replaced.")
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


def build_tab(root: Path, nested: bool = False) -> None:
    if nested:
        build_content(root)
        return
    with gr.Tab("Augment"):
        build_content(root)
