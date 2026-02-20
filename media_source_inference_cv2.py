#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from curate_model_predictions_to_labelme import (
    build_zoom_panel,
    infer_prob_map,
    load_model,
    make_labelme_json,
    mask_to_polygons,
)
from wtcv_utils.labelme import IMG_EXTS


VID_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def ensure_cv2_highgui() -> None:
    missing = []
    if not hasattr(cv2, "namedWindow"):
        missing.append("namedWindow")
    if not hasattr(cv2, "imshow"):
        missing.append("imshow")
    if not hasattr(cv2, "waitKey"):
        missing.append("waitKey")
    if not hasattr(cv2, "destroyAllWindows"):
        missing.append("destroyAllWindows")
    if not missing:
        return
    cv2_file = getattr(cv2, "__file__", None)
    raise RuntimeError(
        "OpenCV HighGUI is not available in this environment. "
        f"missing={missing} cv2.__file__={cv2_file}. "
        "Install GUI OpenCV (`opencv-python`) and remove headless builds."
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run tiled model inference on image folders or video files (optional OpenCV UI)")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input-path", type=Path, required=True, help="Image folder or video file")
    ap.add_argument("--label", type=str, default="vehicle")
    ap.add_argument("--output-dir", type=Path, default=Path("data/media_inference_labelme"))

    ap.add_argument("--tile-size", type=int, default=448)
    ap.add_argument("--tile-stride", type=int, default=448)
    ap.add_argument("--seg-out-stride", type=int, default=4)
    ap.add_argument("--pred-threshold", type=float, default=0.5)
    ap.add_argument("--use-tile-cls-gating", action="store_true", default=True)
    ap.add_argument("--no-use-tile-cls-gating", action="store_false", dest="use_tile_cls_gating")
    ap.add_argument("--tile-cls-threshold", type=float, default=0.5)
    ap.add_argument("--tile-cls-mode", type=str, choices=["hard", "multiply"], default="hard")
    ap.add_argument("--min-poly-area", type=float, default=20.0)
    ap.add_argument("--poly-epsilon-frac", type=float, default=0.002)
    ap.add_argument("--max-fps", type=float, default=30.0)
    ap.add_argument("--infer-every", type=int, default=2, help="Run inference every Nth frame/image")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=0, help="Max inferred items (0=all)")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")

    ap.add_argument("--ui", action="store_true", default=False, help="Enable OpenCV window UI")
    ap.add_argument("--no-ui", action="store_false", dest="ui")
    ap.add_argument("--window-name", type=str, default="media_source_inference")
    ap.add_argument("--save-preview", action="store_true", default=False)
    ap.add_argument("--auto-save-persistent", action="store_true", default=False)
    ap.add_argument("--persist-infers", type=int, default=3, help="Require N matched inference steps before auto-save")
    ap.add_argument("--persist-iou-threshold", type=float, default=0.25, help="IoU threshold for matching detections across inferences")
    ap.add_argument("--persist-max-miss", type=int, default=1, help="Allow up to this many missed inference steps before track drop")
    ap.add_argument("--persist-save-cooldown-infers", type=int, default=8, help="Minimum inference-step gap between auto-saves")
    return ap.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    return [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def poly_bbox(poly: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    ba = max(1.0, (bx1 - bx0) * (by1 - by0))
    return float(inter / (aa + ba - inter))


def update_tracks(
    tracks: List[Dict],
    detections: List[Dict],
    infer_idx: int,
    iou_threshold: float,
    max_miss: int,
    next_track_id: int,
) -> Tuple[List[Dict], int]:
    pairs = []
    for ti, t in enumerate(tracks):
        for di, d in enumerate(detections):
            iou = bbox_iou(t["bbox"], d["bbox"])
            if iou >= iou_threshold:
                pairs.append((iou, ti, di))
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_t = set()
    matched_d = set()
    for _iou, ti, di in pairs:
        if ti in matched_t or di in matched_d:
            continue
        matched_t.add(ti)
        matched_d.add(di)
        det = detections[di]
        tr = tracks[ti]
        tr["bbox"] = det["bbox"]
        tr["poly"] = det["poly"]
        tr["miss"] = 0
        tr["hits"] += 1
        tr["last_infer"] = infer_idx

    for ti, tr in enumerate(tracks):
        if ti not in matched_t:
            tr["miss"] += 1

    for di, det in enumerate(detections):
        if di in matched_d:
            continue
        tracks.append(
            {
                "id": int(next_track_id),
                "bbox": det["bbox"],
                "poly": det["poly"],
                "hits": 1,
                "miss": 0,
                "last_infer": infer_idx,
            }
        )
        next_track_id += 1

    tracks = [t for t in tracks if int(t["miss"]) <= int(max_miss)]
    return tracks, next_track_id


def build_preview(
    img_bgr: np.ndarray,
    prob_full: np.ndarray,
    pred_mask: np.ndarray,
    polys: List[List[List[float]]],
    stats: Dict[str, float],
    source_name: str,
    idx: int,
    total: int,
    fps: float,
    threshold: float,
    gating: bool,
    tile_cls_thr: float,
    tile_cls_mode: str,
    paused: bool,
    auto_save_persistent: bool,
    persistent_count: int,
    persist_infers: int,
    auto_saved_count: int,
    saved: int,
) -> np.ndarray:
    prob_u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_MAGMA)
    merged = cv2.addWeighted(img_bgr, 0.66, heat, 0.34, 0.0)
    for poly in polys:
        arr = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(merged, [arr], isClosed=True, color=(0, 255, 255), thickness=2)
    zoom_panel = build_zoom_panel(img_bgr, pred_mask, max_items=6)
    grid = np.hstack([merged, zoom_panel])

    ph, pw = grid.shape[:2]
    info_h = 170
    canvas = np.zeros((ph + info_h, pw, 3), dtype=np.uint8)
    canvas[:ph] = grid
    status = "PAUSED" if paused else "RUN"
    lines = [
        f"{status} fps={fps:.2f} src={source_name} item={idx+1}/{total} saved={saved}",
        f"pred_thr={threshold:.3f} polys={len(polys)} pred_pixels={int(pred_mask.sum())}",
        f"tile_cls_gate={gating} tile_cls_thr={tile_cls_thr:.3f} mode={tile_cls_mode} tile_cls_mean={stats.get('tile_cls_mean', float('nan')):.3f}",
        f"persistent={persistent_count} (N={persist_infers}) auto_save_persistent={auto_save_persistent} auto_saved={auto_saved_count}",
        "keys: [q]=quit [space]=pause [+/ -]=pred_thr [[/]]=tile_cls_thr [g]=gate [m]=mode [t]=persist-auto [a]=save",
    ]
    y = ph + 30
    for t in lines:
        cv2.putText(canvas, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
        y += 34
    if pw > 2400:
        s = 2400.0 / pw
        canvas = cv2.resize(canvas, (int(pw * s), int((ph + info_h) * s)), interpolation=cv2.INTER_AREA)
    return canvas


def save_output(
    out_dir: Path,
    image_bgr: np.ndarray,
    polys: List[List[List[float]]],
    label: str,
    source_stem: str,
    idx: int,
    preview_bgr: Optional[np.ndarray] = None,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source_stem}_{idx:06d}"
    out_img = out_dir / f"{stem}.jpg"
    out_json = out_dir / f"{stem}.json"
    cv2.imwrite(str(out_img), image_bgr)
    h, w = image_bgr.shape[:2]
    d = make_labelme_json(out_img.name, h, w, polys, label)
    out_json.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    if preview_bgr is not None:
        cv2.imwrite(str(out_dir / f"{stem}__preview.jpg"), preview_bgr)
    return out_img, out_json


def infer_frame(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, List[List[List[float]]], Dict[str, float]]:
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
    with torch.inference_mode():
        with autocast_ctx:
            prob_lr, _cls_lr, stats = infer_prob_map(
                model=model,
                image_np=image_rgb,
                tile_size=int(args.tile_size),
                stride=int(args.tile_stride),
                seg_out_stride=int(args.seg_out_stride),
                device=device,
                use_tile_cls_gating=bool(args.use_tile_cls_gating),
                tile_cls_threshold=float(args.tile_cls_threshold),
                tile_cls_mode=str(args.tile_cls_mode),
            )
        prob_full = F.interpolate(
            torch.from_numpy(prob_lr).float().unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
    pred_mask = (prob_full >= float(args.pred_threshold)).astype(np.uint8)
    polys = mask_to_polygons(pred_mask, min_area=float(args.min_poly_area), epsilon_frac=float(args.poly_epsilon_frac))
    return prob_full, pred_mask, polys, stats


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Missing input path: {args.input_path}")
    if bool(args.ui):
        ensure_cv2_highgui()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    model, info = load_model(args.checkpoint, device)
    print("device:", device)
    print("amp:", use_amp)
    print("model:", info)
    print("ui:", bool(args.ui))
    print("auto_save_persistent:", bool(args.auto_save_persistent))

    is_video = args.input_path.is_file() and (args.input_path.suffix.lower() in VID_EXTS)
    is_dir = args.input_path.is_dir()
    if not is_video and not is_dir:
        raise RuntimeError("input-path must be a video file or image directory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_items = max(0, int(args.max_items))
    infer_every = max(1, int(args.infer_every))
    min_dt = 0.0 if float(args.max_fps) <= 0 else (1.0 / float(args.max_fps))
    paused = False
    auto_save_persistent = bool(args.auto_save_persistent)
    persist_infers = max(1, int(args.persist_infers))
    persist_iou_threshold = float(args.persist_iou_threshold)
    persist_max_miss = max(0, int(args.persist_max_miss))
    persist_save_cooldown_infers = max(0, int(args.persist_save_cooldown_infers))
    saved_count = 0
    auto_saved_count = 0
    fps_smooth = 0.0
    last_t = time.time()
    item_idx = 0
    total = 0
    source_name = args.input_path.name
    infer_idx = 0
    tracks: List[Dict] = []
    next_track_id = 1
    last_auto_save_infer = -10**9
    last_persistent_polys: List[List[List[float]]] = []

    if is_video:
        cap = cv2.VideoCapture(str(args.input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.input_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(1, total_frames)
        if bool(args.ui):
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        frame_idx = -1
        last_preview = None
        last_frame = None
        last_polys: List[List[List[float]]] = []
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx < int(args.start_index):
                    continue
                if (frame_idx - int(args.start_index)) % infer_every != 0:
                    continue
                item_idx += 1
                now = time.time()
                dt = now - last_t
                if dt > 0:
                    fps_inst = 1.0 / dt
                    fps_smooth = fps_inst if fps_smooth <= 0 else (0.90 * fps_smooth + 0.10 * fps_inst)
                last_t = now

                prob_full, pred_mask, polys, stats = infer_frame(model, frame, device, use_amp, args)
                last_frame = frame
                last_polys = polys
                infer_idx += 1
                detections = [{"poly": p, "bbox": poly_bbox(p)} for p in polys if len(p) >= 3]
                tracks, next_track_id = update_tracks(
                    tracks=tracks,
                    detections=detections,
                    infer_idx=infer_idx,
                    iou_threshold=persist_iou_threshold,
                    max_miss=persist_max_miss,
                    next_track_id=next_track_id,
                )
                persistent_tracks = [
                    t for t in tracks
                    if int(t["hits"]) >= persist_infers and int(t["miss"]) == 0 and int(t["last_infer"]) == infer_idx
                ]
                last_persistent_polys = [t["poly"] for t in persistent_tracks if "poly" in t and len(t["poly"]) >= 3]

                if (
                    auto_save_persistent
                    and len(last_persistent_polys) > 0
                    and (infer_idx - last_auto_save_infer) >= persist_save_cooldown_infers
                ):
                    save_output(
                        out_dir=args.output_dir / "persistent_auto",
                        image_bgr=frame,
                        polys=last_persistent_polys,
                        label=args.label,
                        source_stem=args.input_path.stem,
                        idx=item_idx,
                        preview_bgr=None,
                    )
                    saved_count += 1
                    auto_saved_count += 1
                    last_auto_save_infer = infer_idx
                if bool(args.ui):
                    last_preview = build_preview(
                        img_bgr=frame,
                        prob_full=prob_full,
                        pred_mask=pred_mask,
                        polys=polys,
                        stats=stats,
                        source_name=source_name,
                        idx=max(0, frame_idx),
                        total=total,
                        fps=fps_smooth,
                        threshold=float(args.pred_threshold),
                        gating=bool(args.use_tile_cls_gating),
                        tile_cls_thr=float(args.tile_cls_threshold),
                        tile_cls_mode=str(args.tile_cls_mode),
                        paused=paused,
                        auto_save_persistent=auto_save_persistent,
                        persistent_count=len(last_persistent_polys),
                        persist_infers=persist_infers,
                        auto_saved_count=auto_saved_count,
                        saved=saved_count,
                    )
            if bool(args.ui) and last_preview is not None:
                cv2.imshow(args.window_name, last_preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key in (ord("+"), ord("=")):
                    args.pred_threshold = min(0.99, float(args.pred_threshold) + 0.02)
                elif key in (ord("-"), ord("_")):
                    args.pred_threshold = max(0.01, float(args.pred_threshold) - 0.02)
                elif key == ord("["):
                    args.tile_cls_threshold = max(0.01, float(args.tile_cls_threshold) - 0.02)
                elif key == ord("]"):
                    args.tile_cls_threshold = min(0.99, float(args.tile_cls_threshold) + 0.02)
                elif key in (ord("g"), ord("G")):
                    args.use_tile_cls_gating = not bool(args.use_tile_cls_gating)
                elif key in (ord("m"), ord("M")):
                    args.tile_cls_mode = "multiply" if str(args.tile_cls_mode) == "hard" else "hard"
                elif key in (ord("t"), ord("T")):
                    auto_save_persistent = not auto_save_persistent
                    print(f"[toggle] auto_save_persistent={auto_save_persistent}")
                elif key in (ord("a"), ord("A")) and last_preview is not None and last_frame is not None:
                    # manual save last processed frame
                    save_output(
                        out_dir=args.output_dir,
                        image_bgr=last_frame,
                        polys=last_polys,
                        label=args.label,
                        source_stem=args.input_path.stem,
                        idx=item_idx,
                        preview_bgr=last_preview if bool(args.save_preview) else None,
                    )
                    saved_count += 1
            if max_items > 0 and item_idx >= max_items:
                break
            now2 = time.time()
            sleep_s = min_dt - (now2 - last_t)
            if sleep_s > 0:
                time.sleep(sleep_s)
        cap.release()
    else:
        paths = list_images(args.input_path)
        if int(args.start_index) > 0:
            paths = paths[int(args.start_index):]
        if infer_every > 1:
            paths = [p for i, p in enumerate(paths) if (i % infer_every == 0)]
        if max_items > 0:
            paths = paths[:max_items]
        total = len(paths)
        if total == 0:
            print("No images to process")
            return
        if bool(args.ui):
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        for i, ip in enumerate(paths):
            item_idx = i + 1
            frame = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps_inst = 1.0 / dt
                fps_smooth = fps_inst if fps_smooth <= 0 else (0.90 * fps_smooth + 0.10 * fps_inst)
            last_t = now
            prob_full, pred_mask, polys, stats = infer_frame(model, frame, device, use_amp, args)
            infer_idx += 1
            detections = [{"poly": p, "bbox": poly_bbox(p)} for p in polys if len(p) >= 3]
            tracks, next_track_id = update_tracks(
                tracks=tracks,
                detections=detections,
                infer_idx=infer_idx,
                iou_threshold=persist_iou_threshold,
                max_miss=persist_max_miss,
                next_track_id=next_track_id,
            )
            persistent_tracks = [
                t for t in tracks
                if int(t["hits"]) >= persist_infers and int(t["miss"]) == 0 and int(t["last_infer"]) == infer_idx
            ]
            last_persistent_polys = [t["poly"] for t in persistent_tracks if "poly" in t and len(t["poly"]) >= 3]

            if (
                auto_save_persistent
                and len(last_persistent_polys) > 0
                and (infer_idx - last_auto_save_infer) >= persist_save_cooldown_infers
            ):
                save_output(
                    out_dir=args.output_dir / "persistent_auto",
                    image_bgr=frame,
                    polys=last_persistent_polys,
                    label=args.label,
                    source_stem=ip.stem,
                    idx=item_idx,
                    preview_bgr=None,
                )
                saved_count += 1
                auto_saved_count += 1
                last_auto_save_infer = infer_idx
            if bool(args.ui):
                preview = build_preview(
                    img_bgr=frame,
                    prob_full=prob_full,
                    pred_mask=pred_mask,
                    polys=polys,
                    stats=stats,
                    source_name=ip.name,
                    idx=i,
                    total=total,
                    fps=fps_smooth,
                    threshold=float(args.pred_threshold),
                    gating=bool(args.use_tile_cls_gating),
                    tile_cls_thr=float(args.tile_cls_threshold),
                    tile_cls_mode=str(args.tile_cls_mode),
                    paused=False,
                    auto_save_persistent=auto_save_persistent,
                    persistent_count=len(last_persistent_polys),
                    persist_infers=persist_infers,
                    auto_saved_count=auto_saved_count,
                    saved=saved_count,
                )
                cv2.imshow(args.window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("+"), ord("=")):
                    args.pred_threshold = min(0.99, float(args.pred_threshold) + 0.02)
                elif key in (ord("-"), ord("_")):
                    args.pred_threshold = max(0.01, float(args.pred_threshold) - 0.02)
                elif key == ord("["):
                    args.tile_cls_threshold = max(0.01, float(args.tile_cls_threshold) - 0.02)
                elif key == ord("]"):
                    args.tile_cls_threshold = min(0.99, float(args.tile_cls_threshold) + 0.02)
                elif key in (ord("g"), ord("G")):
                    args.use_tile_cls_gating = not bool(args.use_tile_cls_gating)
                elif key in (ord("m"), ord("M")):
                    args.tile_cls_mode = "multiply" if str(args.tile_cls_mode) == "hard" else "hard"
                elif key in (ord("t"), ord("T")):
                    auto_save_persistent = not auto_save_persistent
                    print(f"[toggle] auto_save_persistent={auto_save_persistent}")
                elif key in (ord("a"), ord("A")):
                    save_output(
                        out_dir=args.output_dir,
                        image_bgr=frame,
                        polys=polys,
                        label=args.label,
                        source_stem=ip.stem,
                        idx=item_idx,
                        preview_bgr=preview if bool(args.save_preview) else None,
                    )
                    saved_count += 1
            now2 = time.time()
            sleep_s = min_dt - (now2 - last_t)
            if sleep_s > 0:
                time.sleep(sleep_s)

    if bool(args.ui):
        cv2.destroyAllWindows()
    print(f"done processed={item_idx} saved={saved_count} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
