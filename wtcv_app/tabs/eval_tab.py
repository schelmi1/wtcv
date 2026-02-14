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


def build_tab(root: Path) -> None:
    with gr.Tab("Evaluate"):
        with gr.Row():
            eval_data_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Data Dir")
            eval_ckpt = gr.Textbox(value="", label="Checkpoint")
            eval_label = gr.Textbox(value="vehicle", label="Label")
        with gr.Row():
            eval_tile_size = gr.Number(value=256, precision=0, label="Tile Size")
            eval_tile_stride = gr.Number(value=128, precision=0, label="Tile Stride")
            eval_seg_out_stride = gr.Number(value=4, precision=0, label="Seg Out Stride")
            eval_pred_threshold = gr.Number(value=0.5, label="Pred Threshold")
            eval_workers = gr.Number(value=8, precision=0, label="Num Workers")
            eval_trust_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo")
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
