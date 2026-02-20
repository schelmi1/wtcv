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
    feature_backend: str,
    adapter_checkpoint: str,
    adapter_feature_key: str,
    adapter_input_size: int,
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
        "--feature-backend",
        str(feature_backend).strip().lower(),
        "--adapter-feature-key",
        str(adapter_feature_key).strip(),
        "--adapter-input-size",
        str(int(adapter_input_size)),
        "--device",
        str(device),
    ]
    if str(adapter_checkpoint).strip():
        cmd += ["--adapter-checkpoint", str(adapter_checkpoint).strip()]
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
    max_image_similarity: float,
    image_embed_size: int,
    image_embed_batch_size: int,
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
        "--max-image-similarity",
        str(float(max_image_similarity)),
        "--image-embed-size",
        str(int(image_embed_size)),
        "--image-embed-batch-size",
        str(int(image_embed_batch_size)),
    ]
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    yield from stream_command(cmd)


def run_unique_vs_bank(
    input_dir: str,
    bank_npz: str,
    output_dir: str,
    label_filter: str,
    max_images: int,
    min_poly_points: int,
    scene_knn: int,
    unique_keep_count: int,
    unique_keep_ratio: float,
    scene_embed_size: int,
    scene_embed_batch_size: int,
    tile_size: int,
    tile_context_scale: float,
    max_objects: int,
    batch_size: int,
    bank_topk: int,
    feature_backend: str,
    adapter_checkpoint: str,
    adapter_feature_key: str,
    adapter_input_size: int,
    device: str,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)
    cmd = [
        sys.executable,
        str(ROOT / "unique_images_vs_embedding_bank.py"),
        "--input-dir",
        input_dir,
        "--bank-npz",
        bank_npz,
        "--output-dir",
        output_dir,
        "--label-filter",
        label_filter,
        "--max-images",
        str(int(max_images)),
        "--min-poly-points",
        str(int(min_poly_points)),
        "--scene-knn",
        str(int(scene_knn)),
        "--unique-keep-count",
        str(int(unique_keep_count)),
        "--unique-keep-ratio",
        str(float(unique_keep_ratio)),
        "--scene-embed-size",
        str(int(scene_embed_size)),
        "--scene-embed-batch-size",
        str(int(scene_embed_batch_size)),
        "--tile-size",
        str(int(tile_size)),
        "--tile-context-scale",
        str(float(tile_context_scale)),
        "--max-objects",
        str(int(max_objects)),
        "--batch-size",
        str(int(batch_size)),
        "--bank-topk",
        str(int(bank_topk)),
        "--feature-backend",
        str(feature_backend).strip().lower(),
        "--adapter-feature-key",
        str(adapter_feature_key).strip(),
        "--adapter-input-size",
        str(int(adapter_input_size)),
        "--device",
        str(device),
    ]
    if str(adapter_checkpoint).strip():
        cmd += ["--adapter-checkpoint", str(adapter_checkpoint).strip()]
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
        with gr.Row():
            bank_feature_backend = gr.Dropdown(
                choices=["dino", "adapter"],
                value="dino",
                label="Feature Backend",
                info="`dino` uses raw DINO patch-token embeddings. `adapter` uses a Stage1 checkpoint feature map.",
            )
            bank_adapter_ckpt = gr.Textbox(
                value="",
                label="Adapter Checkpoint (for backend=adapter)",
                info="Path to Stage1 checkpoint to extract adapted features from.",
            )
            bank_adapter_feature_key = gr.Dropdown(
                choices=["feat_adapted", "feat_dino"],
                value="feat_adapted",
                label="Adapter Feature Key",
                info="Feature map key returned by Stage1 model when backend=adapter.",
            )
            bank_adapter_input_size = gr.Number(
                value=0,
                precision=0,
                label="Adapter Input Size (0=tile size)",
                info="Optional square resize before adapter forward. Must be multiple of 256 for adapter backend.",
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
                bank_feature_backend,
                bank_adapter_ckpt,
                bank_adapter_feature_key,
                bank_adapter_input_size,
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
            sim_max_scene = gr.Number(
                value=0.92,
                label="Max Scene Similarity",
                info="Discard candidate pairs if source-image cosine similarity is above this value.",
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
            sim_img_embed_size = gr.Number(
                value=448,
                precision=0,
                label="Scene Embed Size",
                info="Resize for scene-level image embeddings used by the scene similarity filter.",
            )
            sim_img_embed_batch = gr.Number(
                value=12,
                precision=0,
                label="Scene Embed Batch",
                info="Batch size for scene-level embedding extraction.",
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
                sim_max_scene,
                sim_img_embed_size,
                sim_img_embed_batch,
                sim_trust_repo,
            ],
            outputs=[sim_cmd, sim_logs],
        )

        gr.Markdown(
            "Unique-scenes-first workflow: rank scene uniqueness within new records, keep the most unique images, "
            "then score their objects against an existing embedding bank."
        )
        with gr.Row():
            uvb_input_dir = gr.Textbox(
                value=str(root / "data/record_pairs"),
                label="Unique-vs-Bank Input LabelMe Dir",
                info="Directory of new LabelMe image/json pairs to mine from.",
            )
            uvb_bank_npz = gr.Textbox(
                value=str(root / "outputs/embedding_bank/embedding_bank.npz"),
                label="Embedding Bank NPZ",
                info="Path to `embedding_bank.npz` generated by Build Embedding Bank.",
            )
            uvb_output_dir = gr.Textbox(
                value=str(root / "outputs/unique_vs_bank"),
                label="Unique-vs-Bank Output Dir",
                info="Directory where unique-scene and object-vs-bank reports are written.",
            )
        with gr.Row():
            uvb_labels = gr.Textbox(
                value="vehicle",
                label="Label Filter (comma-separated, blank=all)",
                info="Target labels for object extraction and novelty scoring.",
            )
            uvb_max_images = gr.Number(
                value=0,
                precision=0,
                label="Max Candidate Images (0=all)",
                info="Optional cap on candidate images before uniqueness ranking.",
            )
            uvb_min_poly = gr.Number(
                value=3,
                precision=0,
                label="Min Poly Points",
                info="Minimum polygon points for valid object annotations.",
            )
            uvb_scene_knn = gr.Number(
                value=5,
                precision=0,
                label="Scene KNN",
                info="K neighbors used to compute scene density/uniqueness.",
            )
        with gr.Row():
            uvb_keep_count = gr.Number(
                value=0,
                precision=0,
                label="Keep Unique Count (0=ratio)",
                info="If >0, keep exactly this many most-unique images.",
            )
            uvb_keep_ratio = gr.Number(
                value=0.30,
                label="Keep Unique Ratio",
                info="Used when keep-count is 0; fraction of most-unique images to keep.",
            )
            uvb_scene_size = gr.Number(
                value=448,
                precision=0,
                label="Scene Embed Size",
                info="Resize size used for scene embeddings.",
            )
            uvb_scene_batch = gr.Number(
                value=12,
                precision=0,
                label="Scene Embed Batch",
                info="Batch size for scene embedding extraction.",
            )
        with gr.Row():
            uvb_tile_size = gr.Number(
                value=448,
                precision=0,
                label="Object Tile Size",
                info="Object crop size used for object embedding extraction.",
            )
            uvb_tile_context = gr.Number(
                value=2.0,
                label="Object Tile Context",
                info="Object crop context scale.",
            )
            uvb_max_objects = gr.Number(
                value=0,
                precision=0,
                label="Max Objects (0=all)",
                info="Optional object cap after unique scene selection.",
            )
            uvb_obj_batch = gr.Number(
                value=12,
                precision=0,
                label="Object Embed Batch",
                info="Batch size for object embedding extraction.",
            )
            uvb_bank_topk = gr.Number(
                value=5,
                precision=0,
                label="Bank Top-K Mean",
                info="Top-K bank neighbors used for mean object-vs-bank score.",
            )
        with gr.Row():
            uvb_device = gr.Textbox(
                value="",
                label="Device (blank=auto)",
                info="Compute device override; leave blank for automatic selection.",
            )
            uvb_trust_repo = gr.Dropdown(
                choices=["on", "off"],
                value="on",
                label="Trust torch.hub repo",
                info="Allow torch.hub to trust and execute repository code without prompt.",
            )
        with gr.Row():
            uvb_feature_backend = gr.Dropdown(
                choices=["dino", "adapter"],
                value="dino",
                label="Object Feature Backend",
                info="Backend used for object-vs-bank embeddings on selected unique scenes.",
            )
            uvb_adapter_ckpt = gr.Textbox(
                value="",
                label="Adapter Checkpoint (backend=adapter)",
                info="Stage1 checkpoint path used when Object Feature Backend is adapter.",
            )
            uvb_adapter_feature_key = gr.Dropdown(
                choices=["feat_adapted", "feat_dino"],
                value="feat_adapted",
                label="Adapter Feature Key",
                info="Feature map key used for masked pooling in adapter backend.",
            )
            uvb_adapter_input_size = gr.Number(
                value=0,
                precision=0,
                label="Adapter Input Size (0=tile size)",
                info="Optional adapter forward resize; must be multiple of 256.",
            )
        uvb_btn = gr.Button("Run Unique Scenes -> Object-vs-Bank", variant="primary")
        uvb_cmd = gr.Textbox(label="Unique-vs-Bank Command", interactive=False)
        uvb_logs = gr.Textbox(label="Unique-vs-Bank Logs", lines=20, elem_classes=["mono"], interactive=False)
        uvb_btn.click(
            fn=run_unique_vs_bank,
            inputs=[
                uvb_input_dir,
                uvb_bank_npz,
                uvb_output_dir,
                uvb_labels,
                uvb_max_images,
                uvb_min_poly,
                uvb_scene_knn,
                uvb_keep_count,
                uvb_keep_ratio,
                uvb_scene_size,
                uvb_scene_batch,
                uvb_tile_size,
                uvb_tile_context,
                uvb_max_objects,
                uvb_obj_batch,
                uvb_bank_topk,
                uvb_feature_backend,
                uvb_adapter_ckpt,
                uvb_adapter_feature_key,
                uvb_adapter_input_size,
                uvb_device,
                uvb_trust_repo,
            ],
            outputs=[uvb_cmd, uvb_logs],
        )
