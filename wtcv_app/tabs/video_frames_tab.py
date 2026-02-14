from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple

import gradio as gr

from wtcv_app.common import as_bool, stream_command


def run_extract_frames(
    video_path: str,
    output_dir: str,
    fps: float,
    overwrite: bool,
) -> Generator[Tuple[str, str], None, None]:
    overwrite = as_bool(overwrite)
    video = Path(video_path)
    if not video.exists():
        cmd_str = "ffmpeg (not executed)"
        logs = f"Missing video file: {video_path}\n"
        yield cmd_str, logs
        return
    out_dir = Path(output_dir)
    if overwrite and out_dir.exists():
        for p in out_dir.glob("*"):
            if p.is_file():
                p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pat = str(out_dir / f"{video.stem}_f%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(video),
        "-vf",
        f"fps={float(fps)}",
        out_pat,
    ]
    yield from stream_command(cmd)


def build_tab(root: Path) -> None:
    with gr.Tab("Video -> Frames"):
        with gr.Row():
            vid_path = gr.Textbox(value="", label="Video Path")
            vid_out = gr.Textbox(value=str(root / "videos/frames_2fps"), label="Output Frames Dir")
            vid_fps = gr.Number(value=2.0, label="FPS")
            vid_overwrite = gr.Dropdown(choices=["on", "off"], value="off", label="Overwrite Existing Frames")
        vid_btn = gr.Button("Extract Frames", variant="primary")
        vid_cmd = gr.Textbox(label="Command", interactive=False)
        vid_logs = gr.Textbox(label="Live Logs", lines=20, elem_classes=["mono"], interactive=False)
        vid_btn.click(
            fn=run_extract_frames,
            inputs=[vid_path, vid_out, vid_fps, vid_overwrite],
            outputs=[vid_cmd, vid_logs],
        )
