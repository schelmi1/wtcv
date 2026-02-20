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
    feature_backend: str,
    adapter_checkpoint: str,
    adapter_feature_key: str,
    adapter_input_size: int,
    device: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    run_kmeans: bool,
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
        "--feature-backend",
        str(feature_backend),
        "--adapter-feature-key",
        str(adapter_feature_key),
        "--adapter-input-size",
        str(int(adapter_input_size)),
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
    cmd += build_bool_arg("--run-kmeans", "--no-run-kmeans", bool(as_bool(run_kmeans)))
    adapter_checkpoint = str(adapter_checkpoint).strip()
    if adapter_checkpoint:
        cmd += ["--adapter-checkpoint", adapter_checkpoint]
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
            fo_input_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Input LabelMe Dir", info="Directory of input LabelMe image/json pairs.")
            fo_dataset_name = gr.Textbox(value="wtcv_object_umap", label="FiftyOne Dataset Name", info="Name of the FiftyOne dataset to create or overwrite.")
            fo_output_dir = gr.Textbox(value=str(root / "outputs/fiftyone_object_umap"), label="Output Dir", info="Directory where generated outputs are written.")
        with gr.Row():
            fo_labels = gr.Textbox(value="vehicle,fp", label="Label Filter (comma-separated)", info="Comma-separated labels to include when building object samples.")
            fo_max_objects = gr.Number(value=0, precision=0, label="Max Objects (0=all)", info="Maximum number of objects to include; 0 means all objects.")
            fo_tile_size = gr.Number(value=448, precision=0, label="Tile Size", info="Side length of each square inference/training tile in pixels.")
            fo_context = gr.Number(value=2.0, label="Tile Context Scale", info="Object crop context multiplier relative to object bbox size.")
            fo_batch = gr.Number(value=12, precision=0, label="DINO Batch Size", info="Batch size used while extracting DINO embeddings.")
        with gr.Row():
            fo_feature_backend = gr.Dropdown(
                choices=["auto", "dino", "adapter"],
                value="auto",
                label="Feature Backend",
                info="auto uses adapter only when a checkpoint is provided; otherwise raw DINO.",
            )
            fo_adapter_ckpt = gr.Textbox(
                value="",
                label="Adapter Checkpoint (optional)",
                info="If set, Object UMAP can use Stage1 adapter features for masked pooling.",
            )
            fo_adapter_feature_key = gr.Dropdown(
                choices=["feat_adapted", "feat_dino"],
                value="feat_adapted",
                label="Adapter Feature Key",
                info="Feature map key from Stage1 model to pool object embeddings from.",
            )
            fo_adapter_input_size = gr.Number(
                value=0,
                precision=0,
                label="Adapter Input Size (0=tile size)",
                info="Optional square resize before adapter forward. Must be multiple of 256 when > 0.",
            )
        with gr.Row():
            fo_device = gr.Textbox(value="", label="Device (blank=auto)", info="Compute device override; leave blank for automatic selection.")
            fo_umap_neighbors = gr.Number(value=30, precision=0, label="UMAP n_neighbors", info="UMAP neighborhood size controlling local/global manifold balance.")
            fo_umap_min_dist = gr.Number(value=0.05, label="UMAP min_dist", info="UMAP minimum embedding distance; lower values produce tighter clusters.")
            fo_umap_metric = gr.Textbox(value="cosine", label="UMAP metric", info="Distance metric used by UMAP for embedding computation.")
        with gr.Row():
            fo_run_kmeans = gr.Dropdown(
                choices=["on", "off"],
                value="off",
                label="Run KMeans",
                info="Optional. If off, no cluster fit/labels are created.",
            )
            fo_clusters = gr.Number(value=20, precision=0, label="KMeans Clusters", info="Number of KMeans clusters in embedding space.")
            fo_seed = gr.Number(value=42, precision=0, label="Seed", info="Random seed for reproducible sampling and clustering behavior.")
            fo_overwrite = gr.Dropdown(choices=["on", "off"], value="on", label="Overwrite Dataset", info="If on, delete and recreate an existing FiftyOne dataset name.")
            fo_launch = gr.Dropdown(choices=["on", "off"], value="on", label="Launch FiftyOne App", info="If on, open the FiftyOne app after dataset creation.")
            fo_trust_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo", info="Allow torch.hub to trust and execute repository code without prompt.")
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
                fo_feature_backend,
                fo_adapter_ckpt,
                fo_adapter_feature_key,
                fo_adapter_input_size,
                fo_device,
                fo_umap_neighbors,
                fo_umap_min_dist,
                fo_umap_metric,
                fo_run_kmeans,
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
            fo_exp_dataset_name = gr.Textbox(value="wtcv_object_umap", label="Dataset Name", info="Name of the existing FiftyOne dataset to export from.")
            fo_exp_output_dir = gr.Textbox(value=str(root / "data/umap_filtered_dataset"), label="Output LabelMe Dir", info="Destination directory for merged LabelMe exports.")
            fo_exp_tag_labels = gr.Textbox(value="vehicle,fp", label="Tag Labels (priority order)", info="Comma-separated tags mapped to output labels in priority order.")
            fo_exp_overwrite = gr.Dropdown(choices=["on", "off"], value="on", label="Overwrite Output", info="If on, existing output files/directories may be replaced.")
        fo_exp_btn = gr.Button("Export Tagged -> LabelMe", variant="primary")
        fo_exp_cmd = gr.Textbox(label="Export Command", interactive=False)
        fo_exp_logs = gr.Textbox(label="Export Logs", lines=12, elem_classes=["mono"], interactive=False)
        fo_exp_btn.click(
            fn=run_export_tagged_labelme,
            inputs=[fo_exp_dataset_name, fo_exp_output_dir, fo_exp_tag_labels, fo_exp_overwrite],
            outputs=[fo_exp_cmd, fo_exp_logs],
        )
