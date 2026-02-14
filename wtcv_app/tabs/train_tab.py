from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


def run_train(
    data_dir: str,
    output_dir: str,
    run_name: str,
    resume_checkpoint: str,
    epochs: int,
    subset_size: int,
    label: str,
    fp_label: str,
    tile_size: int,
    tile_stride: int,
    tile_scales: str,
    batch_size: int,
    num_workers: int,
    dino_upsampler: str,
    dino_layers: str,
    anyup_q_chunk_size: int,
    head_type: str,
    use_tile_cls_head: bool,
    tile_cls_weight: float,
    balance_train_50_50: bool,
    balance_val_50_50: bool,
    augment_low_vis: bool,
    hard_negative_mining: bool,
    hnm_hard_ratio: float,
    hnm_pool_frac: float,
    lr: float,
    lr_scheduler: str,
    lr_min: float,
    weight_decay: float,
    mcc_weight: float,
    mcc_warmup_epochs: int,
    bce_weight: float,
    boundary_weight: float,
    training_strategy: str,
    preserve_weight: float,
    preserve_warmup_epochs: int,
    preserve_bg_weight: float,
    preserve_fg_weight: float,
    var_weight: float,
    var_gamma: float,
    use_fp_supervision: bool,
    fp_neg_weight: float,
    fp_neg_ratio: float,
    val_interval: int,
    image_log_interval: int,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    use_tile_cls_head = as_bool(use_tile_cls_head)
    balance_train_50_50 = as_bool(balance_train_50_50)
    balance_val_50_50 = as_bool(balance_val_50_50)
    augment_low_vis = as_bool(augment_low_vis)
    hard_negative_mining = as_bool(hard_negative_mining)
    use_fp_supervision = as_bool(use_fp_supervision)
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)

    cmd = [
        sys.executable,
        str(ROOT / "train_stage1_seg.py"),
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--epochs",
        str(int(epochs)),
        "--subset-size",
        str(int(subset_size)),
        "--label",
        label,
        "--fp-label",
        fp_label,
        "--tile-size",
        str(int(tile_size)),
        "--tile-stride",
        str(int(tile_stride)),
        "--tile-scales",
        tile_scales,
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--dino-upsampler",
        dino_upsampler,
        "--dino-layers",
        str(dino_layers),
        "--anyup-q-chunk-size",
        str(int(anyup_q_chunk_size)),
        "--head-type",
        head_type,
        "--tile-cls-weight",
        str(float(tile_cls_weight)),
        "--hnm-hard-ratio",
        str(float(hnm_hard_ratio)),
        "--hnm-pool-frac",
        str(float(hnm_pool_frac)),
        "--lr",
        str(float(lr)),
        "--lr-scheduler",
        lr_scheduler,
        "--lr-min",
        str(float(lr_min)),
        "--weight-decay",
        str(float(weight_decay)),
        "--mcc-weight",
        str(float(mcc_weight)),
        "--mcc-warmup-epochs",
        str(int(mcc_warmup_epochs)),
        "--bce-weight",
        str(float(bce_weight)),
        "--boundary-weight",
        str(float(boundary_weight)),
        "--training-strategy",
        str(training_strategy),
        "--preserve-weight",
        str(float(preserve_weight)),
        "--preserve-warmup-epochs",
        str(int(preserve_warmup_epochs)),
        "--preserve-bg-weight",
        str(float(preserve_bg_weight)),
        "--preserve-fg-weight",
        str(float(preserve_fg_weight)),
        "--var-weight",
        str(float(var_weight)),
        "--var-gamma",
        str(float(var_gamma)),
        "--fp-neg-weight",
        str(float(fp_neg_weight)),
        "--fp-neg-ratio",
        str(float(fp_neg_ratio)),
        "--val-interval",
        str(int(val_interval)),
        "--image-log-interval",
        str(int(image_log_interval)),
    ]
    if run_name.strip():
        cmd += ["--run-name", run_name.strip()]
    if resume_checkpoint.strip():
        cmd += ["--resume-checkpoint", resume_checkpoint.strip()]

    cmd += build_bool_arg("--use-tile-cls-head", "--no-use-tile-cls-head", bool(use_tile_cls_head))
    cmd += build_bool_arg("--balance-train-50-50", "--no-balance-train-50-50", bool(balance_train_50_50))
    cmd += build_bool_arg("--balance-val-50-50", "--no-balance-val-50-50", bool(balance_val_50_50))
    if augment_low_vis:
        cmd += ["--augment-low-vis"]
    cmd += build_bool_arg("--hard-negative-mining", "--no-hard-negative-mining", bool(hard_negative_mining))
    cmd += build_bool_arg("--use-fp-supervision", "--no-use-fp-supervision", bool(use_fp_supervision))
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))

    yield from stream_command(cmd)


def build_tab(root: Path) -> None:
    def _toggle_semantic_controls(strategy: str):
        show = str(strategy).strip().lower() == "semantic_preserve"
        return gr.update(visible=show)

    with gr.Tab("Train"):
        with gr.Row():
            data_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Data Dir")
            output_dir = gr.Textbox(value=str(root / "runs"), label="Output Dir")
        with gr.Row():
            run_name = gr.Textbox(value="", label="Run Name")
            resume_ckpt = gr.Textbox(value="", label="Resume Checkpoint (optional)")
        with gr.Row():
            epochs = gr.Number(value=5, precision=0, label="Epochs")
            subset_size = gr.Number(value=0, precision=0, label="Subset Size (0=all)")
            label = gr.Textbox(value="vehicle", label="Label")
            fp_label = gr.Textbox(value="fp", label="FP Label")
        with gr.Row():
            tile_size = gr.Number(value=256, precision=0, label="Tile Size")
            tile_stride = gr.Number(value=128, precision=0, label="Tile Stride")
            tile_scales = gr.Textbox(value="1.0", label="Tile Scales")
            gr.Number(value=4, precision=0, label="Seg Out Stride (fixed in script)", interactive=False)
        with gr.Row():
            batch_size = gr.Number(value=8, precision=0, label="Batch Size")
            num_workers = gr.Number(value=8, precision=0, label="Num Workers")
            lr = gr.Number(value=1e-4, label="LR")
            lr_min = gr.Number(value=1e-5, label="LR Min")
            lr_scheduler = gr.Dropdown(choices=["none", "cosine"], value="cosine", label="LR Scheduler")
        with gr.Row():
            dino_upsampler = gr.Dropdown(choices=["learned", "pixelshuffle", "anyup"], value="learned", label="DINO Upsampler")
            dino_layers = gr.Textbox(value="last", label="DINO Layers (last or 1-based csv)")
            anyup_q_chunk_size = gr.Number(value=256, precision=0, label="AnyUp q_chunk_size")
            head_type = gr.Dropdown(choices=["pointwise", "dwsep", "residual"], value="pointwise", label="Head Type")
            weight_decay = gr.Number(value=1e-4, label="Weight Decay")
        with gr.Row():
            use_tile_cls_head = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Head")
            balance_train_50_50 = gr.Dropdown(choices=["on", "off"], value="on", label="Balance Train 50/50")
            balance_val_50_50 = gr.Dropdown(choices=["on", "off"], value="on", label="Balance Val 50/50")
            augment_low_vis = gr.Dropdown(choices=["on", "off"], value="off", label="Low-Vis Augment")
            hard_negative_mining = gr.Dropdown(choices=["on", "off"], value="on", label="Hard Negative Mining")
            trust_torch_hub_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo")
        with gr.Row():
            tile_cls_weight = gr.Number(value=0.3, label="Tile Cls Weight")
            hnm_hard_ratio = gr.Number(value=0.3, label="HNM Hard Ratio")
            hnm_pool_frac = gr.Number(value=0.2, label="HNM Pool Frac")
            val_interval = gr.Number(value=5, precision=0, label="Val Interval")
            image_log_interval = gr.Number(value=1, precision=0, label="Image Log Interval")
        with gr.Row():
            mcc_weight = gr.Number(value=0.4, label="MCC Weight")
            mcc_warmup_epochs = gr.Number(value=3, precision=0, label="MCC Warmup Epochs")
            bce_weight = gr.Number(value=0.5, label="BCE Weight")
            boundary_weight = gr.Number(value=0.2, label="Boundary Weight")
            training_strategy = gr.Dropdown(
                choices=["task_only", "semantic_preserve"],
                value="task_only",
                label="Training Strategy",
            )
            use_fp_supervision = gr.Dropdown(choices=["on", "off"], value="on", label="Use FP Supervision")
            fp_neg_weight = gr.Number(value=0.3, label="FP Neg Weight")
            fp_neg_ratio = gr.Number(value=0.5, label="FP Neg Ratio (balanced neg)")
        with gr.Group(visible=False) as semantic_preserve_controls:
            gr.Markdown(
                "Semantic-preserve weights: `Preserve Weight` scales feature-preservation loss; "
                "`Preserve Warmup` ramps it over epochs; `Preserve BG/FG Weight` control token weighting "
                "(background vs object regions); `Var Weight` scales anti-collapse variance regularizer; "
                "`Var Gamma` is the target per-channel token std threshold for that regularizer."
            )
            with gr.Row():
                preserve_weight = gr.Number(value=0.10, label="Preserve Weight")
                preserve_warmup_epochs = gr.Number(value=3, precision=0, label="Preserve Warmup")
                preserve_bg_weight = gr.Number(value=1.0, label="Preserve BG Weight")
                preserve_fg_weight = gr.Number(value=0.25, label="Preserve FG Weight")
                var_weight = gr.Number(value=0.01, label="Var Weight")
                var_gamma = gr.Number(value=0.5, label="Var Gamma")

        training_strategy.change(
            fn=_toggle_semantic_controls,
            inputs=[training_strategy],
            outputs=[semantic_preserve_controls],
        )
        train_btn = gr.Button("Run Training", variant="primary")
        train_cmd = gr.Textbox(label="Command", interactive=False)
        train_logs = gr.Textbox(label="Live Logs", lines=24, elem_classes=["mono"], interactive=False)

        train_btn.click(
            fn=run_train,
            inputs=[
                data_dir,
                output_dir,
                run_name,
                resume_ckpt,
                epochs,
                subset_size,
                label,
                fp_label,
                tile_size,
                tile_stride,
                tile_scales,
                batch_size,
                num_workers,
                dino_upsampler,
                dino_layers,
                anyup_q_chunk_size,
                head_type,
                use_tile_cls_head,
                tile_cls_weight,
                balance_train_50_50,
                balance_val_50_50,
                augment_low_vis,
                hard_negative_mining,
                hnm_hard_ratio,
                hnm_pool_frac,
                lr,
                lr_scheduler,
                lr_min,
                weight_decay,
                mcc_weight,
                mcc_warmup_epochs,
                bce_weight,
                boundary_weight,
                training_strategy,
                preserve_weight,
                preserve_warmup_epochs,
                preserve_bg_weight,
                preserve_fg_weight,
                var_weight,
                var_gamma,
                use_fp_supervision,
                fp_neg_weight,
                fp_neg_ratio,
                val_interval,
                image_log_interval,
                trust_torch_hub_repo,
            ],
            outputs=[train_cmd, train_logs],
        )
