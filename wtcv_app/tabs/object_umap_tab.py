from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


def run_object_umap(
    input_dir: str,
    dataset_name: str,
    output_dir: str,
    label_filter: str,
    max_objects: int,
    tile_size: int,
    tile_context_scale: float,
    batch_size: int,
    dino_model: str,
    device: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    num_clusters: int,
    seed: int,
    overwrite_dataset: bool,
    launch: bool,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite_dataset = as_bool(overwrite_dataset)
    launch = as_bool(launch)
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)

    cmd = [
        sys.executable,
        str(ROOT / "fiftyone_object_umap.py"),
        "--input-dir",
        input_dir,
        "--dataset-name",
        dataset_name,
        "--output-dir",
        output_dir,
        "--label-filter",
        label_filter,
        "--max-objects",
        str(int(max_objects)),
        "--tile-size",
        str(int(tile_size)),
        "--tile-context-scale",
        str(float(tile_context_scale)),
        "--batch-size",
        str(int(batch_size)),
        "--dino-model",
        dino_model,
        "--device",
        str(device),
        "--umap-n-neighbors",
        str(int(umap_n_neighbors)),
        "--umap-min-dist",
        str(float(umap_min_dist)),
        "--umap-metric",
        umap_metric,
        "--num-clusters",
        str(int(num_clusters)),
        "--seed",
        str(int(seed)),
    ]
    cmd += build_bool_arg("--overwrite-dataset", "--no-overwrite-dataset", bool(overwrite_dataset))
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    if launch:
        cmd += ["--launch"]
    yield from stream_command(cmd)


def run_export_tagged_labelme(
    dataset_name: str,
    output_dir: str,
    tag_labels: str,
    overwrite: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite = as_bool(overwrite)
    cmd = [
        sys.executable,
        str(ROOT / "fiftyone_export_tagged_to_labelme.py"),
        "--dataset-name",
        dataset_name,
        "--output-dir",
        output_dir,
        "--tag-labels",
        tag_labels,
    ]
    cmd += build_bool_arg("--overwrite", "--no-overwrite", bool(overwrite))
    yield from stream_command(cmd)


def build_tab(root: Path) -> None:
    with gr.Tab("Object UMAP (FiftyOne)"):
        gr.Markdown(
            "Creates one sample per object from LabelMe pairs, computes masked DINO object embeddings on object-centric tiles, then runs UMAP + KMeans and writes a FiftyOne dataset."
        )
        with gr.Row():
            fo_input_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Input LabelMe Dir")
            fo_dataset_name = gr.Textbox(value="wtcv_object_umap", label="FiftyOne Dataset Name")
            fo_output_dir = gr.Textbox(value=str(root / "outputs/fiftyone_object_umap"), label="Output Dir")
        with gr.Row():
            fo_labels = gr.Textbox(value="vehicle,fp", label="Label Filter (comma-separated)")
            fo_max_objects = gr.Number(value=0, precision=0, label="Max Objects (0=all)")
            fo_tile_size = gr.Number(value=448, precision=0, label="Tile Size")
            fo_context = gr.Number(value=2.0, label="Tile Context Scale")
            fo_batch = gr.Number(value=12, precision=0, label="DINO Batch Size")
        with gr.Row():
            fo_dino_model = gr.Textbox(value="dinov2_vits14", label="DINO Model")
            fo_device = gr.Textbox(value="", label="Device (blank=auto)")
            fo_umap_neighbors = gr.Number(value=30, precision=0, label="UMAP n_neighbors")
            fo_umap_min_dist = gr.Number(value=0.05, label="UMAP min_dist")
            fo_umap_metric = gr.Textbox(value="cosine", label="UMAP metric")
        with gr.Row():
            fo_clusters = gr.Number(value=20, precision=0, label="KMeans Clusters")
            fo_seed = gr.Number(value=42, precision=0, label="Seed")
            fo_overwrite = gr.Dropdown(choices=["on", "off"], value="on", label="Overwrite Dataset")
            fo_launch = gr.Dropdown(choices=["on", "off"], value="on", label="Launch FiftyOne App")
            fo_trust_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo")
        fo_btn = gr.Button("Run Object UMAP + Build FiftyOne Dataset", variant="primary")
        fo_cmd = gr.Textbox(label="Command", interactive=False)
        fo_logs = gr.Textbox(label="Live Logs", lines=22, elem_classes=["mono"], interactive=False)
        fo_btn.click(
            fn=run_object_umap,
            inputs=[
                fo_input_dir,
                fo_dataset_name,
                fo_output_dir,
                fo_labels,
                fo_max_objects,
                fo_tile_size,
                fo_context,
                fo_batch,
                fo_dino_model,
                fo_device,
                fo_umap_neighbors,
                fo_umap_min_dist,
                fo_umap_metric,
                fo_clusters,
                fo_seed,
                fo_overwrite,
                fo_launch,
                fo_trust_repo,
            ],
            outputs=[fo_cmd, fo_logs],
        )

        gr.Markdown("Export reviewed/taged FiftyOne samples back to merged LabelMe image/json pairs (grouped by source image).")
        with gr.Row():
            fo_exp_dataset_name = gr.Textbox(value="wtcv_object_umap", label="Dataset Name")
            fo_exp_output_dir = gr.Textbox(value=str(root / "data/umap_filtered_dataset"), label="Output LabelMe Dir")
            fo_exp_tag_labels = gr.Textbox(value="vehicle,fp", label="Tag Labels (priority order)")
            fo_exp_overwrite = gr.Dropdown(choices=["on", "off"], value="on", label="Overwrite Output")
        fo_exp_btn = gr.Button("Export Tagged -> LabelMe", variant="primary")
        fo_exp_cmd = gr.Textbox(label="Export Command", interactive=False)
        fo_exp_logs = gr.Textbox(label="Export Logs", lines=12, elem_classes=["mono"], interactive=False)
        fo_exp_btn.click(
            fn=run_export_tagged_labelme,
            inputs=[fo_exp_dataset_name, fo_exp_output_dir, fo_exp_tag_labels, fo_exp_overwrite],
            outputs=[fo_exp_cmd, fo_exp_logs],
        )
