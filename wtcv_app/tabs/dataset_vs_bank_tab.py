from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


def run_dataset_vs_bank(
    input_dir: str,
    checkpoint: str,
    embedding_bank: str,
    output_dir: str,
    pred_threshold: float,
    min_poly_area: float,
    tile_size: int,
    tile_stride: int,
    seg_out_stride: int,
    use_tile_cls_gating: bool,
    tile_cls_threshold: float,
    tile_cls_mode: str,
    feature_backend: str,
    adapter_feature_key: str,
    adapter_input_size: int,
    positive_labels: str,
    bank_topk: int,
    accept_score: float,
    dedup_iou: float,
    vehicle_label: str,
    device: str,
    use_amp: bool,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    use_tile_cls_gating = as_bool(use_tile_cls_gating)
    use_amp = as_bool(use_amp)
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)
    cmd = [
        sys.executable,
        str(ROOT / "score_dataset_with_embedding_bank.py"),
        "--input-dir",
        input_dir,
        "--embedding-bank",
        embedding_bank,
        "--output-dir",
        output_dir,
        "--pred-threshold",
        str(float(pred_threshold)),
        "--min-poly-area",
        str(float(min_poly_area)),
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--seg-out-stride",
        str(int(seg_out_stride)),
        "--tile-cls-threshold",
        str(float(tile_cls_threshold)),
        "--tile-cls-mode",
        str(tile_cls_mode),
        "--feature-backend",
        str(feature_backend).strip().lower(),
        "--adapter-feature-key",
        str(adapter_feature_key).strip(),
        "--adapter-input-size",
        str(int(adapter_input_size)),
        "--positive-labels",
        positive_labels,
        "--bank-topk",
        str(int(bank_topk)),
        "--accept-score",
        str(float(accept_score)),
        "--dedup-iou",
        str(float(dedup_iou)),
        "--vehicle-label",
        str(vehicle_label),
        "--device",
        str(device),
    ]
    ckpt = str(checkpoint).strip()
    if ckpt != "":
        cmd += ["--checkpoint", ckpt]
    cmd += build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    if bool(use_amp):
        cmd += ["--use-amp"]
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    yield from stream_command(cmd)


def build_tab(root: Path, nested: bool = False) -> None:
    container = gr.Column if nested else gr.Tab
    kwargs = {} if nested else {"label": "Dataset vs Bank"}
    with container(**kwargs):
        gr.Markdown(
            "Run detector inference over a full image dataset, score each detected polygon against an embedding bank, "
            "keep original labels, and add only new positive candidates plus a JSONL report."
        )
        with gr.Row():
            dvb_input_dir = gr.Textbox(
                value=str(root / "data/record_pairs"),
                label="Dataset Images Dir",
                info="Directory of images to run detections on.",
            )
            dvb_ckpt = gr.Textbox(
                value="",
                label="Checkpoint",
                info="Optional model checkpoint. If blank, runs in no-checkpoint mode with vanilla DINO scoring on existing LabelMe shapes.",
            )
            dvb_bank = gr.Textbox(
                value=str(root / "outputs/embedding_bank/embedding_bank.npz"),
                label="Embedding Bank NPZ",
                info="Path to `embedding_bank.npz` for similarity scoring.",
            )
            dvb_output_dir = gr.Textbox(
                value=str(root / "outputs/dataset_vs_embedding_bank"),
                label="Output Dir",
                info="Writes scored JSONL report + LabelMe exports.",
            )
        with gr.Row():
            dvb_pred_thr = gr.Number(value=0.35, label="Pred Threshold", info="Mask threshold before polygon extraction.")
            dvb_min_area = gr.Number(value=14.0, label="Min Poly Area", info="Drops very small predicted polygons.")
            dvb_tile_size = gr.Number(value=448, precision=0, label="Tile Size", info="Inference tile size.")
            dvb_tile_stride = gr.Number(value=448, precision=0, label="Tile Stride", info="Inference tile stride.")
            dvb_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride", info="Segmentation output stride.")
            dvb_use_gate = gr.Dropdown(
                choices=["on", "off"],
                value="on",
                label="Use Tile CLS Gating",
                info="Apply tile-classification gating during inference.",
            )
            dvb_tile_cls_thr = gr.Number(value=0.5, label="Tile CLS Threshold", info="Tile gate threshold.")
            dvb_tile_cls_mode = gr.Dropdown(
                choices=["hard", "multiply"],
                value="hard",
                label="Tile CLS Mode",
                info="Hard drop or probability multiplication for gating.",
            )
        with gr.Row():
            dvb_pos_labels = gr.Textbox(
                value="vehicle",
                label="Positive Labels (bank subset)",
                info="Comma-separated bank labels used as positive reference.",
            )
            dvb_topk = gr.Number(value=5, precision=0, label="Bank Top-K", info="Top-K cosine neighbors for scoring.")
            dvb_accept = gr.Number(value=0.35, label="Accept Score", info="Add candidate when pos_topk_mean >= this value.")
            dvb_dedup_iou = gr.Number(value=0.30, label="Dedup IoU", info="Skip candidate if IoU with existing positive >= this value.")
            dvb_vehicle_label = gr.Textbox(value="vehicle", label="Added Label Name", info="Label assigned to accepted added candidates.")
        with gr.Row():
            dvb_feature_backend = gr.Dropdown(
                choices=["auto", "dino", "adapter"],
                value="auto",
                label="Feature Backend",
                info="Embedding backend for candidate-vs-bank similarity.",
            )
            dvb_adapter_feature_key = gr.Dropdown(
                choices=["feat_adapted", "feat_dino"],
                value="feat_adapted",
                label="Adapter Feature Key",
                info="Feature map used for masked pooling in adapter backend.",
            )
            dvb_adapter_input_size = gr.Number(
                value=0,
                precision=0,
                label="Adapter Input Size (0=auto)",
                info="Optional adapter forward resize; must be multiple of 256.",
            )
        with gr.Row():
            dvb_device = gr.Textbox(value="", label="Device (blank=auto)", info="Compute device override.")
            dvb_amp = gr.Dropdown(choices=["on", "off"], value="off", label="AMP", info="Enable mixed precision inference.")
            dvb_trust = gr.Dropdown(
                choices=["on", "off"],
                value="on",
                label="Trust torch.hub repo",
                info="Allow torch.hub to trust and execute repository code without prompt.",
            )
        dvb_btn = gr.Button("Run Full Dataset vs Embedding Bank", variant="primary")
        dvb_cmd = gr.Textbox(label="Dataset-vs-Bank Command", interactive=False)
        dvb_logs = gr.Textbox(label="Dataset-vs-Bank Logs", lines=20, elem_classes=["mono"], interactive=False)
        dvb_btn.click(
            fn=run_dataset_vs_bank,
            inputs=[
                dvb_input_dir,
                dvb_ckpt,
                dvb_bank,
                dvb_output_dir,
                dvb_pred_thr,
                dvb_min_area,
                dvb_tile_size,
                dvb_tile_stride,
                dvb_seg_stride,
                dvb_use_gate,
                dvb_tile_cls_thr,
                dvb_tile_cls_mode,
                dvb_feature_backend,
                dvb_adapter_feature_key,
                dvb_adapter_input_size,
                dvb_pos_labels,
                dvb_topk,
                dvb_accept,
                dvb_dedup_iou,
                dvb_vehicle_label,
                dvb_device,
                dvb_amp,
                dvb_trust,
            ],
            outputs=[dvb_cmd, dvb_logs],
        )
