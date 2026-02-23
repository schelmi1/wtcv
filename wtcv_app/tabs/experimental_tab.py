from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import ROOT, as_bool, build_bool_arg, stream_command


def run_experimental_detect_caption(
    input_dir: str,
    image_path: str,
    checkpoint: str,
    output_dir: str,
    seed: int,
    pred_threshold: float,
    tile_size: int,
    tile_stride: int,
    seg_out_stride: int,
    use_tile_cls_gating: bool,
    tile_cls_threshold: float,
    tile_cls_mode: str,
    min_poly_area: float,
    poly_epsilon_frac: float,
    min_det_area: float,
    min_det_side: int,
    crop_context: float,
    caption_crop_size: int,
    vqa_model_choice: str,
    vqa_model_custom: str,
    vqa_prompt: str,
    device: str,
) -> Generator[Tuple[str, str, str, str | None], None, None]:
    use_tile_cls_gating = as_bool(use_tile_cls_gating)
    vqa_model_choice = str(vqa_model_choice).strip()
    vqa_model_custom = str(vqa_model_custom).strip()
    vqa_model = vqa_model_custom if vqa_model_choice == "custom" and vqa_model_custom else vqa_model_choice
    cmd = [
        sys.executable,
        str(ROOT / "experimental_random_detect_caption.py"),
        "--input-dir",
        str(input_dir),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(int(seed)),
        "--pred-threshold",
        str(float(pred_threshold)),
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
        "--min-poly-area",
        str(float(min_poly_area)),
        "--poly-epsilon-frac",
        str(float(poly_epsilon_frac)),
        "--min-det-area",
        str(float(min_det_area)),
        "--min-det-side",
        str(int(min_det_side)),
        "--crop-context",
        str(float(crop_context)),
        "--caption-crop-size",
        str(int(caption_crop_size)),
        "--vqa-model",
        str(vqa_model),
        "--vqa-prompt",
        str(vqa_prompt),
        "--device",
        str(device),
    ]
    image_path = str(image_path).strip()
    if image_path:
        cmd += ["--image-path", image_path]
    cmd += build_bool_arg("--use-tile-cls-gating", "--no-use-tile-cls-gating", bool(use_tile_cls_gating))
    cmd_text = ""
    logs_text = ""
    for cmd_text, logs_text in stream_command(cmd):
        yield cmd_text, logs_text, "", None

    summary = ""
    preview_value: str | None = None
    try:
        run_ok = "[process_exit_code=0]" in logs_text
        if run_ok and ("RESULT_JSON_BEGIN" in logs_text) and ("RESULT_JSON_END" in logs_text):
            s0 = logs_text.rfind("RESULT_JSON_BEGIN")
            s1 = logs_text.rfind("RESULT_JSON_END")
            payload = logs_text[s0 + len("RESULT_JSON_BEGIN") : s1].strip()
            d = json.loads(payload)
            lines = []
            lines.append(f"selection_mode: {d.get('selection_mode', '')}")
            lines.append(f"image_path_arg: {d.get('image_path_arg', '')}")
            lines.append(f"picked_image: {d.get('picked_image', '')}")
            lines.append(f"num_kept_for_caption: {d.get('num_kept_for_caption', 0)}")
            lines.append(f"vqa_model: {d.get('vqa_model', '')}")
            lines.append(f"vqa_prompt: {d.get('vqa_prompt', '')}")
            lines.append(f"preview_path: {d.get('preview_path', '')}")
            pp = str(d.get("preview_path", "")).strip()
            if pp:
                preview_value = pp
            lines.append("")
            dets = d.get("detections", []) or []
            if len(dets) == 0:
                lines.append("No detections passed filtering.")
            else:
                lines.append("VQA answers:")
                for i, row in enumerate(dets, start=1):
                    ans = str(row.get("answer", "")).strip()
                    bbox = row.get("bbox_xywh", [])
                    pmean = row.get("adapter_pred_mean", float("nan"))
                    pmax = row.get("adapter_pred_max", float("nan"))
                    barea = row.get("bbox_area", 0.0)
                    marea = row.get("mask_area_px", 0)
                    lines.append(
                        f"{i:02d}. bbox={bbox} bbox_area={barea:.1f} mask_area={marea} "
                        f"pred_mean={pmean:.4f} pred_max={pmax:.4f}  answer={ans}"
                    )
            summary = "\n".join(lines)
        elif not run_ok:
            summary = (
                "Run failed; no results parsed.\n"
                "Check Logs above for the exact error.\n"
                "Expected RESULT_JSON_BEGIN/END in stdout."
            )
        else:
            summary = (
                "No structured result block produced by this run.\n"
                "Expected RESULT_JSON_BEGIN/END in stdout."
            )
    except Exception as e:
        summary = f"Failed to parse caption results from stdout: {e}"

    yield cmd_text, logs_text, summary, preview_value


def build_tab(root: Path, nested: bool = False) -> None:
    container = gr.Column if nested else gr.Tab
    kwargs = {} if nested else {"label": "Random Detect + Caption"}
    with container(**kwargs):
        gr.Markdown(
            "Experimental pipeline: pick one random image from a folder, run adapter detections, "
            "center-crop kept detections, and answer a VQA prompt on each crop with a Hugging Face VQA model."
        )
        with gr.Row():
            exp_input_dir = gr.Textbox(
                value=str(root / "data/record_pairs"),
                label="Input Image Folder",
                info="Folder to sample one random image from.",
            )
            exp_image_path = gr.Textbox(
                value="",
                label="Direct Image Path (optional)",
                info="If set, uses this image directly instead of random folder sampling.",
            )
            exp_ckpt = gr.Textbox(
                value="",
                label="Adapter Checkpoint",
                info="Stage1 checkpoint used for detection.",
            )
            exp_output_dir = gr.Textbox(
                value=str(root / "outputs/experimental_detect_caption"),
                label="Output Dir",
                info="Writes preview image and per-detection crops.",
            )
        with gr.Row():
            exp_seed = gr.Number(value=42, precision=0, label="Seed")
            exp_pred_thr = gr.Number(value=0.35, label="Pred Threshold")
            exp_min_poly_area = gr.Number(value=14.0, label="Min Poly Area")
            exp_poly_eps = gr.Number(value=0.002, label="Poly Epsilon Frac")
            exp_min_det_area = gr.Number(value=64.0, label="Min Detection Area")
            exp_min_det_side = gr.Number(value=8, precision=0, label="Min Detection Side")
        with gr.Row():
            exp_tile_size = gr.Number(value=512, precision=0, label="Tile Size")
            exp_tile_stride = gr.Number(value=512, precision=0, label="Tile Stride")
            exp_seg_stride = gr.Number(value=4, precision=0, label="Seg Out Stride")
            exp_use_gate = gr.Dropdown(choices=["on", "off"], value="on", label="Use Tile CLS Gating")
            exp_tile_cls_thr = gr.Number(value=0.5, label="Tile CLS Threshold")
            exp_tile_cls_mode = gr.Dropdown(choices=["hard", "multiply"], value="hard", label="Tile CLS Mode")
        with gr.Row():
            exp_crop_context = gr.Number(value=1.6, label="Detection Crop Context")
            exp_caption_crop_size = gr.Number(value=224, precision=0, label="Caption Crop Size")
            exp_vqa_model_choice = gr.Dropdown(
                choices=[
                    "microsoft/kosmos-2-patch14-224",
                    "Salesforce/blip-vqa-base",
                    "Salesforce/blip-vqa-capfilt-large",
                    "custom",
                ],
                value="Salesforce/blip-vqa-base",
                label="VQA Model",
                info="Choose a VQA preset model or select custom.",
            )
            exp_vqa_model_custom = gr.Textbox(
                value="",
                label="Custom VQA Model (optional)",
                info="Used only when VQA Model = custom.",
            )
            exp_vqa_prompt = gr.Textbox(
                value="What is the main military object in this image? Answer with a short noun phrase.",
                label="VQA Prompt",
                info="Question asked per detection crop.",
            )
            exp_device = gr.Textbox(value="", label="Device (blank=auto)")

        exp_btn = gr.Button("Run Experimental Detect + Caption", variant="primary")
        exp_cmd = gr.Textbox(label="Command", interactive=False)
        exp_logs = gr.Textbox(label="Logs", lines=20, elem_classes=["mono"], interactive=False)
        exp_caption_summary = gr.Textbox(
            label="Caption Results",
            lines=12,
            elem_classes=["mono"],
            interactive=False,
            info="Parsed directly from script stdout after run completes.",
        )
        exp_preview = gr.Image(
            label="Detection Preview (heatmap + bboxes + captions)",
            type="filepath",
            interactive=False,
        )
        exp_btn.click(
            fn=run_experimental_detect_caption,
            inputs=[
                exp_input_dir,
                exp_image_path,
                exp_ckpt,
                exp_output_dir,
                exp_seed,
                exp_pred_thr,
                exp_tile_size,
                exp_tile_stride,
                exp_seg_stride,
                exp_use_gate,
                exp_tile_cls_thr,
                exp_tile_cls_mode,
                exp_min_poly_area,
                exp_poly_eps,
                exp_min_det_area,
                exp_min_det_side,
                exp_crop_context,
                exp_caption_crop_size,
                exp_vqa_model_choice,
                exp_vqa_model_custom,
                exp_vqa_prompt,
                exp_device,
            ],
            outputs=[exp_cmd, exp_logs, exp_caption_summary, exp_preview],
        )
