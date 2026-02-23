from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


def run_pretraining(
    input_dir: str,
    output_dir: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    image_size: int,
    local_crop_size: int,
    num_local_crops: int,
    global_min_scale: float,
    local_min_scale: float,
    dino_model: str,
    out_dim: int,
    proj_hidden_dim: int,
    proj_bottleneck_dim: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    teacher_momentum: float,
    teacher_temp: float,
    student_temp: float,
    center_momentum: float,
    dino_weight: float,
    ibot_weight: float,
    ibot_mask_ratio: float,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_targets: str,
    head_only_warmup_epochs: int,
    warmup_use_vanilla_backbone: bool,
    seed: int,
    save_every: int,
    debug_pca_every_steps: int,
    lora_log_every_steps: int,
    device: str,
    trust_torch_hub_repo: bool,
) -> Generator[Tuple[str, str], None, None]:
    trust_torch_hub_repo = as_bool(trust_torch_hub_repo)

    cmd = [
        sys.executable,
        str(ROOT / "pretrain_dino_lora_ssl.py"),
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--image-size",
        str(int(image_size)),
        "--local-crop-size",
        str(int(local_crop_size)),
        "--num-local-crops",
        str(int(num_local_crops)),
        "--global-min-scale",
        str(float(global_min_scale)),
        "--local-min-scale",
        str(float(local_min_scale)),
        "--dino-model",
        str(dino_model),
        "--out-dim",
        str(int(out_dim)),
        "--proj-hidden-dim",
        str(int(proj_hidden_dim)),
        "--proj-bottleneck-dim",
        str(int(proj_bottleneck_dim)),
        "--lr",
        str(float(lr)),
        "--min-lr",
        str(float(min_lr)),
        "--weight-decay",
        str(float(weight_decay)),
        "--teacher-momentum",
        str(float(teacher_momentum)),
        "--teacher-temp",
        str(float(teacher_temp)),
        "--student-temp",
        str(float(student_temp)),
        "--center-momentum",
        str(float(center_momentum)),
        "--dino-weight",
        str(float(dino_weight)),
        "--ibot-weight",
        str(float(ibot_weight)),
        "--ibot-mask-ratio",
        str(float(ibot_mask_ratio)),
        "--lora-rank",
        str(int(lora_rank)),
        "--lora-alpha",
        str(float(lora_alpha)),
        "--lora-dropout",
        str(float(lora_dropout)),
        "--lora-targets",
        str(lora_targets),
        "--head-only-warmup-epochs",
        str(int(head_only_warmup_epochs)),
        "--seed",
        str(int(seed)),
        "--save-every",
        str(int(save_every)),
        "--debug-pca-every-steps",
        str(int(debug_pca_every_steps)),
        "--lora-log-every-steps",
        str(int(lora_log_every_steps)),
        "--device",
        str(device),
    ]
    if str(run_name).strip():
        cmd += ["--run-name", str(run_name).strip()]
    cmd += build_bool_arg(
        "--warmup-use-vanilla-backbone",
        "--no-warmup-use-vanilla-backbone",
        bool(as_bool(warmup_use_vanilla_backbone)),
    )
    cmd += build_bool_arg("--trust-torch-hub-repo", "--no-trust-torch-hub-repo", bool(trust_torch_hub_repo))
    yield from stream_command(cmd)


def build_tab(root: Path, nested: bool = False) -> None:
    container = gr.Group if nested else gr.Tab
    container_kwargs = {} if nested else {"label": "LoRA SSL Pretrain"}

    with container(**container_kwargs):
        gr.Markdown(
            "LoRA-based self-supervised DINO/iBOT-style pretraining. "
            "Student backbone gets LoRA adapters, teacher is EMA-updated."
        )
        with gr.Row():
            input_dir = gr.Textbox(value=str(root / "data"), label="Input Image Dir(s)", info="Recursive image scan root(s) for unlabeled pretraining frames. Supports comma-separated directories.")
            output_dir = gr.Textbox(value=str(root / "runs"), label="Output Dir", info="Run folders are created here.")
            run_name = gr.Textbox(value="", label="Run Name (optional)", info="Optional suffix for run directory.")
        with gr.Row():
            epochs = gr.Number(value=10, precision=0, label="Epochs")
            batch_size = gr.Number(value=16, precision=0, label="Batch Size")
            num_workers = gr.Number(value=8, precision=0, label="Num Workers")
            seed = gr.Number(value=42, precision=0, label="Seed")
            save_every = gr.Number(value=1, precision=0, label="Save Every (epochs)")
            debug_pca_every_steps = gr.Number(value=0, precision=0, label="Debug PCA Every N Steps (0=off)")
            lora_log_every_steps = gr.Number(value=20, precision=0, label="LoRA Log Every N Steps (0=off)")
        with gr.Row():
            image_size = gr.Number(value=224, precision=0, label="Global Crop Size")
            local_crop_size = gr.Number(value=96, precision=0, label="Local Crop Size")
            num_local_crops = gr.Number(value=4, precision=0, label="Num Local Crops")
            global_min_scale = gr.Number(value=0.4, label="Global Min Scale")
            local_min_scale = gr.Number(value=0.08, label="Local Min Scale")
        with gr.Row():
            dino_model = gr.Textbox(value="dinov2_vits14_reg", label="DINO Model")
            out_dim = gr.Number(value=65536, precision=0, label="Output Dim")
            proj_hidden_dim = gr.Number(value=2048, precision=0, label="Proj Hidden Dim")
            proj_bottleneck_dim = gr.Number(value=256, precision=0, label="Proj Bottleneck Dim")
        with gr.Row():
            lr = gr.Number(value=1e-3, label="LR")
            min_lr = gr.Number(value=1e-5, label="Min LR")
            weight_decay = gr.Number(value=1e-4, label="Weight Decay")
            teacher_momentum = gr.Number(value=0.996, label="Teacher Momentum")
            center_momentum = gr.Number(value=0.9, label="Center Momentum")
        with gr.Row():
            teacher_temp = gr.Number(value=0.04, label="Teacher Temp")
            student_temp = gr.Number(value=0.1, label="Student Temp")
            dino_weight = gr.Number(value=1.0, label="DINO Loss Weight")
            ibot_weight = gr.Number(value=1.0, label="iBOT Loss Weight")
            ibot_mask_ratio = gr.Number(value=0.3, label="iBOT Mask Ratio")
        with gr.Row():
            lora_rank = gr.Number(value=8, precision=0, label="LoRA Rank")
            lora_alpha = gr.Number(value=16.0, label="LoRA Alpha")
            lora_dropout = gr.Number(value=0.0, label="LoRA Dropout")
            lora_targets = gr.Textbox(value="attn.qkv,attn.proj", label="LoRA Targets")
            head_only_warmup_epochs = gr.Number(value=1, precision=0, label="Head-only Warmup Epochs")
            warmup_use_vanilla_backbone = gr.Dropdown(choices=["on", "off"], value="on", label="Warmup uses vanilla DINO")
        with gr.Row():
            device = gr.Textbox(value="", label="Device (blank=auto)")
            trust_torch_hub_repo = gr.Dropdown(choices=["on", "off"], value="on", label="Trust torch.hub repo")

        btn = gr.Button("Run LoRA SSL Pretraining", variant="primary")
        cmd_out = gr.Textbox(label="Command", interactive=False)
        logs = gr.Textbox(label="Live Logs", lines=22, elem_classes=["mono"], interactive=False)

        btn.click(
            fn=run_pretraining,
            inputs=[
                input_dir,
                output_dir,
                run_name,
                epochs,
                batch_size,
                num_workers,
                image_size,
                local_crop_size,
                num_local_crops,
                global_min_scale,
                local_min_scale,
                dino_model,
                out_dim,
                proj_hidden_dim,
                proj_bottleneck_dim,
                lr,
                min_lr,
                weight_decay,
                teacher_momentum,
                teacher_temp,
                student_temp,
                center_momentum,
                dino_weight,
                ibot_weight,
                ibot_mask_ratio,
                lora_rank,
                lora_alpha,
                lora_dropout,
                lora_targets,
                head_only_warmup_epochs,
                warmup_use_vanilla_backbone,
                seed,
                save_every,
                debug_pca_every_steps,
                lora_log_every_steps,
                device,
                trust_torch_hub_repo,
            ],
            outputs=[cmd_out, logs],
        )
