#!/usr/bin/env python3
from __future__ import annotations

import argparse

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("Gradio is required to run this app. Install with: pip install gradio") from exc

from wtcv_app.common import ROOT
from wtcv_app.tabs.augment_tab import build_tab as build_augment_tab
from wtcv_app.tabs.curation_tab import build_tab as build_curation_tab
from wtcv_app.tabs.dataset_peek_tab import build_tab as build_dataset_peek_tab
from wtcv_app.tabs.eval_tab import build_tab as build_eval_tab
from wtcv_app.tabs.live_screen_tab import build_tab as build_live_screen_tab
from wtcv_app.tabs.media_source_tab import build_tab as build_media_source_tab
from wtcv_app.tabs.object_umap_tab import build_tab as build_object_umap_tab
from wtcv_app.tabs.sam_tab import build_tab as build_sam_tab
from wtcv_app.tabs.single_image_tab import build_tab as build_single_image_tab
from wtcv_app.tabs.train_tab import build_tab as build_train_tab
from wtcv_app.tabs.video_frames_tab import build_tab as build_video_frames_tab


THEME_INIT_JS = r"""
() => {
  const THEMES = ["default", "ocean", "forest", "ember"];
  const MODE_KEY = "wtcv_mode";
  const THEME_KEY = "wtcv_theme";
  const body = document.body;
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ""; }
  function getMode() { return localStorage.getItem(MODE_KEY) || "light"; }
  function getTheme() {
    const t = localStorage.getItem(THEME_KEY) || "default";
    return THEMES.includes(t) ? t : "default";
  }
  function setBtnText(id, text) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.tagName && el.tagName.toLowerCase() === "button") {
      el.textContent = text;
      return;
    }
    const btn = el.querySelector("button");
    if (btn) btn.textContent = text;
  }
  function applyUi() {
    const mode = getMode();
    const theme = getTheme();
    const root = document.documentElement;
    const container = document.querySelector(".gradio-container");
    const nodes = [body, root, container].filter(Boolean);
    nodes.forEach((n) => {
      n.classList.toggle("wtcv-dark", mode === "dark");
      n.classList.remove("wtcv-theme-ocean", "wtcv-theme-forest", "wtcv-theme-ember");
      if (theme !== "default") n.classList.add("wtcv-theme-" + theme);
    });
    setBtnText("wtcv-mode-btn", "Mode: " + cap(mode));
    setBtnText("wtcv-theme-btn", "Theme: " + cap(theme));
  }
  function toggleMode() {
    const next = getMode() === "dark" ? "light" : "dark";
    localStorage.setItem(MODE_KEY, next);
    applyUi();
  }
  function cycleTheme() {
    const cur = getTheme();
    const idx = THEMES.indexOf(cur);
    const next = THEMES[(idx + 1 + THEMES.length) % THEMES.length];
    localStorage.setItem(THEME_KEY, next);
    applyUi();
  }
  function install() {
    applyUi();
    const modeBtn = document.getElementById("wtcv-mode-btn");
    const themeBtn = document.getElementById("wtcv-theme-btn");
    if (modeBtn && !modeBtn.dataset.bound) {
      modeBtn.dataset.bound = "1";
      modeBtn.addEventListener("click", toggleMode);
    }
    if (themeBtn && !themeBtn.dataset.bound) {
      themeBtn.dataset.bound = "1";
      themeBtn.addEventListener("click", cycleTheme);
    }
  }
  install();
  setTimeout(install, 120);
  setTimeout(install, 800);
}
"""

APP_CSS = """
:root {
  --wtcv-bg: #f6f8fb;
  --wtcv-panel: #ffffff;
  --wtcv-text: #171c26;
  --wtcv-muted: #6b7280;
  --wtcv-accent: #2b6df8;
  --wtcv-accent-2: #1447b8;
  --wtcv-border: #d6dce7;
}
body.wtcv-dark {
  --wtcv-bg: #0f1420;
  --wtcv-panel: #121b2b;
  --wtcv-text: #e7ecf6;
  --wtcv-muted: #9aa8bd;
  --wtcv-accent: #6ea8ff;
  --wtcv-accent-2: #9ec2ff;
  --wtcv-border: #24334d;
}
.gradio-container {
  background: var(--wtcv-bg) !important;
  color: var(--wtcv-text) !important;
}
.gradio-container .block,
.gradio-container .gr-box,
.gradio-container [class*="panel"],
.gradio-container [class*="block"] {
  background: var(--wtcv-panel) !important;
  border-color: var(--wtcv-border) !important;
  color: var(--wtcv-text) !important;
}
#top-title {
  font-size: 30px;
  font-weight: 700;
  color: var(--wtcv-accent);
}
.wtcv-theme-controls {
  position: fixed;
  right: 16px;
  top: 12px;
  z-index: 10050;
  display: flex;
  gap: 8px;
  align-items: center;
  width: max-content !important;
  background: color-mix(in srgb, var(--wtcv-panel) 88%, transparent);
  border: 1px solid color-mix(in srgb, var(--wtcv-accent) 28%, #9aa5b1);
  border-radius: 999px;
  padding: 6px 8px;
}
.wtcv-theme-controls > * { pointer-events: auto; }
.wtcv-theme-controls .wtcv-chip {
  font-size: 12px;
  color: var(--wtcv-muted);
  padding: 0 6px 0 2px;
}
.mono textarea {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;}
"""


def make_app() -> gr.Blocks:
    with gr.Blocks(title="WTCV Studio", theme=gr.themes.Soft(), css=APP_CSS) as app:
        gr.HTML(
            """
            <div class="wtcv-theme-controls" id="wtcv-theme-controls">
              <span class="wtcv-chip">UI</span>
              <button id="wtcv-mode-btn" type="button">Mode: Light</button>
              <button id="wtcv-theme-btn" type="button">Theme: Default</button>
            </div>
            """
        )

        app.load(fn=None, inputs=None, outputs=None, js=THEME_INIT_JS)

        gr.Markdown(
            """
            <div id="top-title">WTCV Studio</div>
            Unified control panel for training, evaluation, dataset conversion, augmentation, and inference preview.
            """
        )

        with gr.Tabs():
            build_train_tab(ROOT)
            build_eval_tab(ROOT)
            build_sam_tab(ROOT)
            build_augment_tab(ROOT)
            build_curation_tab(ROOT)
            build_single_image_tab()
            build_dataset_peek_tab(ROOT)
            build_object_umap_tab(ROOT)
            build_video_frames_tab(ROOT)
            build_media_source_tab(ROOT)
            build_live_screen_tab(ROOT)

    return app


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WTCV Studio Gradio app")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", default=False)
    ap.add_argument("--no-queue", action="store_true", default=False)
    return ap.parse_args()


def main() -> None:
    args = parse_cli()
    app = make_app()
    if args.no_queue:
        app.launch(server_name=args.host, server_port=args.port, share=args.share)
    else:
        app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
