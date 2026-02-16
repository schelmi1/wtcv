from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


def run_embedding_bank(
    input_dir: str,
    output_dir: str,
    label_filter: str,
    max_objects: int,
    tile_size: int,
    tile_context_scale: float,
    min_poly_points: int,
    batch_size: int,
    device: str,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)
    cmd = [
        sys.executable,
        str(ROOT / "build_embedding_bank.py"),
        "--input-dir",
        input_dir,
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
        "--min-poly-points",
        str(int(min_poly_points)),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        str(device),
    ]
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    yield from stream_command(cmd)


def run_similarity_report(
    input_dir: str,
    output_dir: str,
    label_filter: str,
    max_objects: int,
    tile_size: int,
    tile_context_scale: float,
    min_poly_points: int,
    batch_size: int,
    device: str,
    top_k: int,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)
    cmd = [
        sys.executable,
        str(ROOT / "report_object_cosine_similarity.py"),
        "--input-dir",
        input_dir,
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
        "--min-poly-points",
        str(int(min_poly_points)),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        str(device),
        "--top-k",
        str(int(top_k)),
    ]
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    yield from stream_command(cmd)


def build_tab(root: Path) -> None:
    with gr.Tab("Embedding Bank"):
        gr.Markdown(
            "Builds masked object embeddings from LabelMe image/json pairs and writes a reusable embedding bank "
            "(`embedding_bank.npz`, `embedding_bank_meta.jsonl`, `embedding_prototypes.npz`)."
        )
        with gr.Row():
            bank_input_dir = gr.Textbox(
                value=str(root / "data/record_pairs"),
                label="Input LabelMe Dir",
                info="Directory of input LabelMe image/json pairs.",
            )
            bank_output_dir = gr.Textbox(
                value=str(root / "outputs/embedding_bank"),
                label="Output Dir",
                info="Directory where embedding bank files are written.",
            )
            bank_labels = gr.Textbox(
                value="vehicle",
                label="Label Filter (comma-separated, blank=all)",
                info="Only these labels are embedded. Leave blank to include all labels.",
            )
        with gr.Row():
            bank_max_objects = gr.Number(
                value=0,
                precision=0,
                label="Max Objects (0=all)",
                info="Maximum number of objects to include; 0 means all objects.",
            )
            bank_tile_size = gr.Number(
                value=448,
                precision=0,
                label="Tile Size",
                info="Side length of each square object-centric tile.",
            )
            bank_context = gr.Number(
                value=2.0,
                label="Tile Context Scale",
                info="Crop side = max(object_w, object_h) * context scale.",
            )
            bank_min_poly_points = gr.Number(
                value=3,
                precision=0,
                label="Min Poly Points",
                info="Minimum number of polygon points required to keep an object.",
            )
            bank_batch = gr.Number(
                value=12,
                precision=0,
                label="DINO Batch Size",
                info="Batch size used while extracting DINO embeddings.",
            )
        with gr.Row():
            bank_device = gr.Textbox(
                value="",
                label="Device (blank=auto)",
                info="Compute device override; leave blank for automatic selection.",
            )
            bank_trust_repo = gr.Dropdown(
                choices=["on", "off"],
                value="on",
                label="Trust torch.hub repo",
                info="Allow torch.hub to trust and execute repository code without prompt.",
            )
        bank_btn = gr.Button("Build Embedding Bank", variant="primary")
        bank_cmd = gr.Textbox(label="Command", interactive=False)
        bank_logs = gr.Textbox(label="Live Logs", lines=22, elem_classes=["mono"], interactive=False)
        bank_btn.click(
            fn=run_embedding_bank,
            inputs=[
                bank_input_dir,
                bank_output_dir,
                bank_labels,
                bank_max_objects,
                bank_tile_size,
                bank_context,
                bank_min_poly_points,
                bank_batch,
                bank_device,
                bank_trust_repo,
            ],
            outputs=[bank_cmd, bank_logs],
        )

        gr.Markdown("Analyze a LabelMe folder and report highest and lowest cosine-similarity object pairs.")
        with gr.Row():
            sim_input_dir = gr.Textbox(
                value=str(root / "data/record_pairs"),
                label="Similarity Input LabelMe Dir",
                info="Directory of input LabelMe image/json pairs to analyze.",
            )
            sim_output_dir = gr.Textbox(
                value=str(root / "outputs/embedding_similarity"),
                label="Similarity Output Dir",
                info="Directory where similarity report files are written.",
            )
            sim_labels = gr.Textbox(
                value="vehicle",
                label="Label Filter (comma-separated, blank=all)",
                info="Only these labels are embedded for similarity scoring.",
            )
        with gr.Row():
            sim_max_objects = gr.Number(
                value=0,
                precision=0,
                label="Max Objects (0=all)",
                info="Maximum number of objects to include; 0 means all objects.",
            )
            sim_top_k = gr.Number(
                value=25,
                precision=0,
                label="Top K High/Low Pairs",
                info="Number of most similar and least similar pairs to report.",
            )
            sim_tile_size = gr.Number(
                value=448,
                precision=0,
                label="Tile Size",
                info="Side length of each square object-centric tile.",
            )
            sim_context = gr.Number(
                value=2.0,
                label="Tile Context Scale",
                info="Crop side = max(object_w, object_h) * context scale.",
            )
            sim_min_poly_points = gr.Number(
                value=3,
                precision=0,
                label="Min Poly Points",
                info="Minimum number of polygon points required to keep an object.",
            )
            sim_batch = gr.Number(
                value=12,
                precision=0,
                label="DINO Batch Size",
                info="Batch size used while extracting DINO embeddings.",
            )
        with gr.Row():
            sim_device = gr.Textbox(
                value="",
                label="Device (blank=auto)",
                info="Compute device override; leave blank for automatic selection.",
            )
            sim_trust_repo = gr.Dropdown(
                choices=["on", "off"],
                value="on",
                label="Trust torch.hub repo",
                info="Allow torch.hub to trust and execute repository code without prompt.",
            )
        sim_btn = gr.Button("Run Cosine Similarity Report", variant="primary")
        sim_cmd = gr.Textbox(label="Similarity Command", interactive=False)
        sim_logs = gr.Textbox(label="Similarity Logs", lines=20, elem_classes=["mono"], interactive=False)
        sim_btn.click(
            fn=run_similarity_report,
            inputs=[
                sim_input_dir,
                sim_output_dir,
                sim_labels,
                sim_max_objects,
                sim_tile_size,
                sim_context,
                sim_min_poly_points,
                sim_batch,
                sim_device,
                sim_top_k,
                sim_trust_repo,
            ],
            outputs=[sim_cmd, sim_logs],
        )
