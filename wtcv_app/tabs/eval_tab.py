from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


def run_eval(
    data_dir: str,
    checkpoint: str,
    label_name: str,
    tile_size: int,
    tile_stride: int,
    seg_out_stride: int,
    pred_threshold: float,
    num_workers: int,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)
    cmd = [
        sys.executable,
        str(ROOT / "eval_stage1_seg.py"),
        "--data-dir",
        data_dir,
        "--checkpoint",
        checkpoint,
        "--label-name",
        label_name,
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--seg-out-stride",
        str(int(seg_out_stride)),
        "--pred-threshold",
        str(float(pred_threshold)),
        "--num-workers",
        str(int(num_workers)),
    ]
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    yield from stream_command(cmd)


def run_linear_probe(
    data_dir: str,
    output_dir: str,
    run_name: str,
    checkpoint: str,
    label_name: str,
    fp_label: str,
    tile_size: int,
    tile_stride: int,
    min_poly_points: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    pred_threshold: float,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)
    cmd = [
        sys.executable,
        str(ROOT / "linear_probe_seg.py"),
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--run-name",
        run_name,
        "--label",
        label_name,
        "--fp-label",
        fp_label,
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--min-poly-points",
        str(int(min_poly_points)),
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--epochs",
        str(int(epochs)),
        "--lr",
        str(float(lr)),
        "--weight-decay",
        str(float(weight_decay)),
        "--pred-threshold",
        str(float(pred_threshold)),
    ]
    if checkpoint.strip():
        cmd += ["--checkpoint", checkpoint.strip()]
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    yield from stream_command(cmd)


def build_tab(root: Path) -> None:
    with gr.Tab("Evaluate"):
        with gr.Tabs():
            with gr.Tab("Standard Eval"):
                with gr.Row():
                    eval_data_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Data Dir", info="Directory containing training/evaluation LabelMe data pairs.")
                    eval_ckpt = gr.Textbox(value="", label="Checkpoint", info="Path to a trained model checkpoint (.pt) to load for inference/evaluation.")
                    eval_label = gr.Textbox(value="vehicle", label="Label", info="Class label name used for output polygons and evaluation target.")
                with gr.Row():
                    eval_tile_size = gr.Number(value=256, precision=0, label="Tile Size", info="Side length of each square inference/training tile in pixels.")
                    eval_tile_stride = gr.Number(value=128, precision=0, label="Tile Stride", info="Step size between tile origins; lower values add overlap and compute cost.")
                    eval_seg_out_stride = gr.Number(value=4, precision=0, label="Seg Out Stride", info="Output stride of segmentation logits relative to tile resolution.")
                    eval_pred_threshold = gr.Number(value=0.5, label="Pred Threshold", info="Probability threshold used to convert logits/probabilities into a binary mask.")
                    eval_workers = gr.Number(value=8, precision=0, label="Num Workers", info="Number of data-loader workers for parallel sample preparation.")
                    eval_trust_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo", info="Allow torch.hub to trust and execute repository code without prompt.")
                eval_btn = gr.Button("Run Evaluation", variant="primary")
                eval_cmd = gr.Textbox(label="Command", interactive=False)
                eval_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)
                eval_btn.click(
                    fn=run_eval,
                    inputs=[
                        eval_data_dir,
                        eval_ckpt,
                        eval_label,
                        eval_tile_size,
                        eval_tile_stride,
                        eval_seg_out_stride,
                        eval_pred_threshold,
                        eval_workers,
                        eval_trust_repo,
                    ],
                    outputs=[eval_cmd, eval_logs],
                )

            with gr.Tab("Linear Probing"):
                with gr.Row():
                    lp_data_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Data Dir", info="LabelMe dataset used for probe training/evaluation.")
                    lp_output_dir = gr.Textbox(value=str(root / "runs"), label="Output Dir", info="Directory where linear-probe runs are saved.")
                    lp_run_name = gr.Textbox(value="", label="Run Name", info="Optional run-name suffix for the output folder.")
                with gr.Row():
                    lp_ckpt = gr.Textbox(value="", label="Checkpoint (optional)", info="Optional feature checkpoint: SSL pretrain or Stage1. Empty uses vanilla DINO.")
                    lp_label = gr.Textbox(value="vehicle", label="Label", info="Positive class label used for segmentation probe.")
                    lp_fp_label = gr.Textbox(value="", label="FP Label", info="Optional false-positive label name in dataset annotations. Leave empty to ignore FP labels.")
                with gr.Row():
                    lp_tile_size = gr.Number(value=512, precision=0, label="Tile Size", info="Tile size used to build probe samples.")
                    lp_tile_stride = gr.Number(value=512, precision=0, label="Tile Stride", info="Tile stride used to build probe samples.")
                    lp_min_poly_points = gr.Number(value=3, precision=0, label="Min Poly Points", info="Minimum polygon points required to keep an annotation.")
                    lp_pred_thr = gr.Number(value=0.5, label="Pred Threshold", info="Threshold used for IoU metric binarization.")
                with gr.Row():
                    lp_batch = gr.Number(value=16, precision=0, label="Batch Size", info="Probe training batch size.")
                    lp_workers = gr.Number(value=8, precision=0, label="Num Workers", info="DataLoader worker processes.")
                    lp_epochs = gr.Number(value=5, precision=0, label="Epochs", info="Number of probe training epochs.")
                    lp_lr = gr.Number(value=1e-3, label="LR", info="Learning rate for linear probe head.")
                    lp_wd = gr.Number(value=1e-4, label="Weight Decay", info="Weight decay for linear probe optimizer.")
                    lp_trust_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo", info="Allow torch.hub model loading without prompt.")
                lp_btn = gr.Button("Run Linear Probing", variant="primary")
                lp_cmd = gr.Textbox(label="Command", interactive=False)
                lp_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)
                lp_btn.click(
                    fn=run_linear_probe,
                    inputs=[
                        lp_data_dir,
                        lp_output_dir,
                        lp_run_name,
                        lp_ckpt,
                        lp_label,
                        lp_fp_label,
                        lp_tile_size,
                        lp_tile_stride,
                        lp_min_poly_points,
                        lp_batch,
                        lp_workers,
                        lp_epochs,
                        lp_lr,
                        lp_wd,
                        lp_pred_thr,
                        lp_trust_repo,
                    ],
                    outputs=[lp_cmd, lp_logs],
                )
