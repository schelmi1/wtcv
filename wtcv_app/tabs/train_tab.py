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
    local_backbone: str,
    local_unfreeze: str,
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
        "--local-backbone",
        str(local_backbone),
        "--local-unfreeze",
        str(local_unfreeze),
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
            data_dir = gr.Textbox(value=str(root / "data/record_pairs"), label="Data Dir", info="Directory containing training/evaluation LabelMe data pairs.")
            output_dir = gr.Textbox(value=str(root / "runs"), label="Output Dir", info="Directory where generated outputs are written.")
        with gr.Row():
            run_name = gr.Textbox(value="", label="Run Name", info="Optional run subfolder name; leave blank for auto-generated name.")
            resume_ckpt = gr.Textbox(value="", label="Resume Checkpoint (optional)", info="Optional checkpoint path to resume training state from.")
        with gr.Row():
            epochs = gr.Number(value=5, precision=0, label="Epochs", info="Number of full training passes over the dataset.")
            subset_size = gr.Number(value=0, precision=0, label="Subset Size (0=all)", info="Optional sample cap for quicker experiments; 0 uses the full dataset.")
            label = gr.Textbox(value="vehicle", label="Label", info="Class label name used for output polygons and evaluation target.")
            fp_label = gr.Textbox(value="fp", label="FP Label", info="Label name treated as false-positive/background supervision.")
        with gr.Row():
            tile_size = gr.Number(value=256, precision=0, label="Tile Size", info="Side length of each square inference/training tile in pixels.")
            tile_stride = gr.Number(value=128, precision=0, label="Tile Stride", info="Step size between tile origins; lower values add overlap and compute cost.")
            tile_scales = gr.Textbox(value="1.0", label="Tile Scales", info="Comma-separated tile scale multipliers for multi-scale training crops.")
            gr.Number(value=4, precision=0, label="Seg Out Stride (fixed in script)", interactive=False)
        with gr.Row():
            batch_size = gr.Number(value=8, precision=0, label="Batch Size", info="Mini-batch size used during training.")
            num_workers = gr.Number(value=8, precision=0, label="Num Workers", info="Number of data-loader workers for parallel sample preparation.")
            lr = gr.Number(value=1e-4, label="LR", info="Initial optimizer learning rate.")
            lr_min = gr.Number(value=1e-5, label="LR Min", info="Minimum learning rate floor (used by cosine schedule).")
            lr_scheduler = gr.Dropdown(choices=["none", "cosine"], value="cosine", label="LR Scheduler", info="Learning-rate scheduling strategy used during training.")
        with gr.Row():
            dino_upsampler = gr.Dropdown(choices=["learned", "pixelshuffle", "anyup"], value="learned", label="DINO Upsampler", info="Upsampling head used to project DINO tokens to dense feature maps.")
            dino_layers = gr.Textbox(value="last", label="DINO Layers (last or 1-based csv)", info="DINO transformer layers to use (\"last\" or comma-separated 1-based indices).")
            anyup_q_chunk_size = gr.Number(value=256, precision=0, label="AnyUp q_chunk_size", info="Chunk size used by AnyUp attention upsampler to limit memory.")
            local_backbone = gr.Dropdown(choices=["resnet18", "resnet34", "resnet50"], value="resnet18", label="Local ResNet Backbone", info="Select local CNN backbone used before fusion. ResNet50 has higher capacity and more channels.")
            local_unfreeze = gr.Dropdown(choices=["none", "l1", "stem+1"], value="none", label="Local ResNet Unfreeze", info="Unfreeze local ResNet blocks: none (frozen), l1 (layer1 only), stem+1 (stem and layer1).")
            head_type = gr.Dropdown(choices=["pointwise", "dwsep", "residual"], value="pointwise", label="Head Type", info="Segmentation decoder head architecture variant.")
            weight_decay = gr.Number(value=1e-4, label="Weight Decay", info="L2-style regularization strength in the optimizer.")
        with gr.Row():
            use_tile_cls_head = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile Cls Head", info="Enable auxiliary tile-level classification head during training.")
            balance_train_50_50 = gr.Dropdown(choices=["on", "off"], value="on", label="Balance Train 50/50", info="Balance train sampling between positive and negative tiles.")
            balance_val_50_50 = gr.Dropdown(choices=["on", "off"], value="on", label="Balance Val 50/50", info="Balance validation sampling between positive and negative tiles.")
            augment_low_vis = gr.Dropdown(choices=["on", "off"], value="off", label="Low-Vis Augment", info="Enable low-visibility image augmentations during training.")
            hard_negative_mining = gr.Dropdown(choices=["on", "off"], value="on", label="Hard Negative Mining", info="Enable hard-negative mining to focus on difficult negatives.")
            trust_torch_hub_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo", info="Allow torch.hub to trust and execute repository code without prompt.")
        with gr.Row():
            tile_cls_weight = gr.Number(value=0.3, label="Tile Cls Weight", info="Loss weight for auxiliary tile classification objective.")
            hnm_hard_ratio = gr.Number(value=0.3, label="HNM Hard Ratio", info="Fraction of hard negatives mixed into each mined negative batch.")
            hnm_pool_frac = gr.Number(value=0.2, label="HNM Pool Frac", info="Fraction of negative pool considered when selecting hard examples.")
            val_interval = gr.Number(value=5, precision=0, label="Val Interval", info="Run validation every N training epochs.")
            image_log_interval = gr.Number(value=1, precision=0, label="Image Log Interval", info="Log visual prediction examples every N epochs.")
        with gr.Row():
            mcc_weight = gr.Number(value=0.4, label="MCC Weight", info="Loss weight for Matthews correlation coefficient term.")
            mcc_warmup_epochs = gr.Number(value=3, precision=0, label="MCC Warmup Epochs", info="Epochs used to ramp in MCC loss contribution.")
            bce_weight = gr.Number(value=0.5, label="BCE Weight", info="Loss weight for binary cross-entropy segmentation term.")
            boundary_weight = gr.Number(value=0.2, label="Boundary Weight", info="Loss weight for boundary-focused segmentation term.")
            training_strategy = gr.Dropdown(
                choices=["task_only", "semantic_preserve"],
                value="task_only",
                label="Training Strategy",
                info="Select pure task loss or semantic-preservation regularized training.",
            )
            use_fp_supervision = gr.Dropdown(choices=["on", "off"], value="on", label="Use FP Supervision", info="If on, include fp-labeled objects in negative supervision terms.")
            fp_neg_weight = gr.Number(value=0.3, label="FP Neg Weight", info="Loss weight applied to false-positive negative supervision.")
            fp_neg_ratio = gr.Number(value=0.5, label="FP Neg Ratio (balanced neg)", info="Target ratio of fp negatives among sampled negatives.")
        with gr.Group(visible=False) as semantic_preserve_controls:
            gr.Markdown(
                "Semantic-preserve weights: `Preserve Weight` scales feature-preservation loss; "
                "`Preserve Warmup` ramps it over epochs; `Preserve BG/FG Weight` control token weighting "
                "(background vs object regions); `Var Weight` scales anti-collapse variance regularizer; "
                "`Var Gamma` is the target per-channel token std threshold for that regularizer."
            )
            with gr.Row():
                preserve_weight = gr.Number(value=0.10, label="Preserve Weight", info="Global weight for semantic feature preservation loss.")
                preserve_warmup_epochs = gr.Number(value=3, precision=0, label="Preserve Warmup", info="Epochs used to warm up semantic preservation loss weight.")
                preserve_bg_weight = gr.Number(value=1.0, label="Preserve BG Weight", info="Relative preservation weight for background tokens.")
                preserve_fg_weight = gr.Number(value=0.25, label="Preserve FG Weight", info="Relative preservation weight for foreground/object tokens.")
                var_weight = gr.Number(value=0.01, label="Var Weight", info="Weight of variance regularizer used to avoid feature collapse.")
                var_gamma = gr.Number(value=0.5, label="Var Gamma", info="Target token standard-deviation threshold for variance regularization.")

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
                local_backbone,
                local_unfreeze,
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
