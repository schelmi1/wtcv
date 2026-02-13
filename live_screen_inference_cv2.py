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
    infer_prob_map,
    load_model,
    make_labelme_json,
    mask_to_polygons,
)

try:
    import mss  # type: ignore
except Exception:
    mss = None

try:
    from PIL import ImageGrab  # type: ignore
except Exception:
    ImageGrab = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live screen inference with OpenCV UI")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--label", type=str, default="vehicle")

    ap.add_argument("--monitor-index", type=int, default=1, help="mss monitor index (1..N); 0 means full virtual screen")
    ap.add_argument("--x", type=int, default=0, help="Capture region x offset relative to selected monitor")
    ap.add_argument("--y", type=int, default=0, help="Capture region y offset relative to selected monitor")
    ap.add_argument("--width", type=int, default=0, help="Capture region width; 0 means full monitor width")
    ap.add_argument("--height", type=int, default=0, help="Capture region height; 0 means full monitor height")
    ap.add_argument("--print-monitors", action="store_true", default=False)

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
    ap.add_argument("--infer-every", type=int, default=2, help="Run inference every N captured frames")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")

    ap.add_argument("--window-name", type=str, default="live_screen_inference")
    ap.add_argument("--output-dir", type=Path, default=Path("data/live_screen_labelme"))
    ap.add_argument("--save-preview", action="store_true", default=False)

    ap.add_argument("--auto-save-persistent", action="store_true", default=False)
    ap.add_argument("--persist-infers", type=int, default=3, help="Require N matched inference steps before auto-save")
    ap.add_argument("--persist-iou-threshold", type=float, default=0.25, help="IoU threshold for matching detections across inferences")
    ap.add_argument("--persist-max-miss", type=int, default=1, help="Allow up to this many missed inference steps before track drop")
    ap.add_argument("--persist-save-cooldown-infers", type=int, default=8, help="Minimum inference-step gap between auto-saves")
    return ap.parse_args()


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
    # Greedy IoU matching between existing tracks and current detections.
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

    # Miss bookkeeping for unmatched tracks.
    for ti, tr in enumerate(tracks):
        if ti not in matched_t:
            tr["miss"] += 1

    # Spawn tracks for unmatched detections.
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

    # Drop stale tracks.
    tracks = [t for t in tracks if int(t["miss"]) <= int(max_miss)]
    return tracks, next_track_id


class ScreenGrabber:
    def __init__(self, monitor_index: int, x: int, y: int, width: int, height: int) -> None:
        self.use_mss = mss is not None
        self.sct = None
        self.region = None
        self.bbox = None

        if self.use_mss:
            self.sct = mss.mss()
            monitors = self.sct.monitors
            if len(monitors) <= 1:
                monitor_index = 0
            monitor_index = max(0, min(int(monitor_index), len(monitors) - 1))
            base = monitors[monitor_index]
            left = int(base["left"]) + int(x)
            top = int(base["top"]) + int(y)
            full_w = int(base["width"])
            full_h = int(base["height"])
            w = full_w if int(width) <= 0 else int(width)
            h = full_h if int(height) <= 0 else int(height)
            w = max(1, min(w, full_w))
            h = max(1, min(h, full_h))
            self.region = {"left": left, "top": top, "width": w, "height": h}
            return

        if ImageGrab is None:
            raise RuntimeError(
                "No screen capture backend available. Install `mss` (recommended): pip install mss"
            )

        # Best-effort ImageGrab fallback.
        left = int(x)
        top = int(y)
        w = int(width)
        h = int(height)
        if w <= 0 or h <= 0:
            raise RuntimeError(
                "ImageGrab fallback requires explicit --width/--height when mss is not available."
            )
        self.bbox = (left, top, left + w, top + h)

    def grab_bgr(self) -> np.ndarray:
        if self.use_mss and self.sct is not None and self.region is not None:
            shot = np.array(self.sct.grab(self.region), dtype=np.uint8)  # BGRA
            return shot[:, :, :3].copy()
        assert self.bbox is not None
        img = ImageGrab.grab(bbox=self.bbox)
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def print_monitors() -> None:
        if mss is None:
            print("mss not available; monitor listing unavailable.")
            return
        with mss.mss() as sct:
            for i, mon in enumerate(sct.monitors):
                print(f"monitor[{i}] left={mon['left']} top={mon['top']} width={mon['width']} height={mon['height']}")


def build_live_preview(
    img_bgr: np.ndarray,
    prob_full: np.ndarray,
    pred_mask: np.ndarray,
    polys,
    zoom_scores: List[Optional[float]],
    stats: Dict[str, float],
    fps: float,
    threshold: float,
    gating: bool,
    tile_cls_thr: float,
    tile_cls_mode: str,
    paused: bool,
    persistent_count: int,
    auto_saved_count: int,
    auto_save_persistent: bool,
    persist_infers: int,
) -> np.ndarray:
    prob_u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_MAGMA)
    merged = cv2.addWeighted(img_bgr, 0.66, heat, 0.34, 0.0)
    for i, poly in enumerate(polys):
        arr = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(merged, [arr], isClosed=True, color=(0, 255, 255), thickness=2)
        if i < len(zoom_scores) and zoom_scores[i] is not None:
            bx0, by0, _bx1, _by1 = poly_bbox(poly)
            txt = f"z={float(zoom_scores[i]):.2f}"
            tx = int(max(0, bx0))
            ty = int(max(18, by0 - 6))
            cv2.putText(merged, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # Keep zoomed target panel, but avoid extra multi-panel composition.
    zoom_panel = build_zoom_panel_scored(img_bgr, polys, zoom_scores, max_items=6)
    grid = np.hstack([merged, zoom_panel])

    panel_h, panel_w = grid.shape[:2]
    info_h = 170
    canvas = np.zeros((panel_h + info_h, panel_w, 3), dtype=np.uint8)
    canvas[:panel_h] = grid

    status = "PAUSED" if paused else "LIVE"
    zoom_valid = [v for v in zoom_scores if v is not None]
    zoom_mean = float(np.mean(zoom_valid)) if len(zoom_valid) > 0 else float("nan")
    lines = [
        f"{status}  fps={fps:.2f}  pred_thr={threshold:.3f}  polygons={len(polys)}  pred_pixels={int(pred_mask.sum())}",
        f"tile_cls_gate={gating}  tile_cls_thr={tile_cls_thr:.3f}  mode={tile_cls_mode}  tile_cls_mean={stats.get('tile_cls_mean', float('nan')):.3f}",
        f"zoom_cls_mean={zoom_mean:.3f}  zoom_cls_count={len(zoom_valid)}",
        f"persistent={persistent_count} (N={persist_infers})  auto_save={auto_save_persistent}  auto_saved={auto_saved_count}",
        "keys: [q]=quit  [space]=pause/resume  [+/-]=pred threshold  [[/]]=tile-cls threshold",
        "keys: [g]=toggle tile-cls gating  [m]=toggle hard/multiply  [t]=toggle auto-save  [a]=save frame+json",
    ]
    y = panel_h + 30
    for t in lines:
        cv2.putText(canvas, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
        y += 36

    if panel_w > 2400:
        scale = 2400.0 / panel_w
        canvas = cv2.resize(canvas, (int(panel_w * scale), int((panel_h + info_h) * scale)), interpolation=cv2.INTER_AREA)
    return canvas


def save_current(
    out_dir: Path,
    frame_bgr: np.ndarray,
    polys,
    label: str,
    frame_idx: int,
    preview_bgr: Optional[np.ndarray],
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    stem = f"screen_{ts}_{frame_idx:06d}"
    out_img = out_dir / f"{stem}.jpg"
    out_json = out_dir / f"{stem}.json"
    cv2.imwrite(str(out_img), frame_bgr)
    h, w = frame_bgr.shape[:2]
    d = make_labelme_json(out_img.name, h, w, polys, label)
    out_json.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    if preview_bgr is not None:
        cv2.imwrite(str(out_dir / f"{stem}__preview.jpg"), preview_bgr)
    return out_img, out_json


def build_zoom_panel_scored(
    img_bgr: np.ndarray,
    polys: List[List[List[float]]],
    zoom_scores: List[Optional[float]],
    max_items: int = 6,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    panel = np.zeros_like(img_bgr)
    panel[:] = 20

    boxes = []
    for i, poly in enumerate(polys):
        if len(poly) < 3:
            continue
        x0, y0, x1, y1 = poly_bbox(poly)
        bw = max(1, int(round(x1 - x0)))
        bh = max(1, int(round(y1 - y0)))
        if bw <= 1 or bh <= 1:
            continue
        boxes.append((i, int(round(x0)), int(round(y0)), bw, bh, bw * bh))
    boxes.sort(key=lambda t: t[5], reverse=True)
    boxes = boxes[: max(1, int(max_items))]

    rows, cols = 2, 3
    pad = 8
    tile_w = max(1, (w - (cols + 1) * pad) // cols)
    tile_h = max(1, (h - (rows + 1) * pad) // rows)

    for j, b in enumerate(boxes):
        i, x, y, bw, bh, _ = b
        r = j // cols
        c = j % cols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)

        mx = int(max(2, bw * 0.2))
        my = int(max(2, bh * 0.2))
        cx0 = max(0, x - mx)
        cy0 = max(0, y - my)
        cx1 = min(w, x + bw + mx)
        cy1 = min(h, y + bh + my)
        crop = img_bgr[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        panel[y0:y0 + tile_h, x0:x0 + tile_w] = crop
        cv2.rectangle(panel, (x0, y0), (x0 + tile_w, y0 + tile_h), (0, 255, 255), 1)
        score = zoom_scores[i] if i < len(zoom_scores) else None
        stxt = f"z={float(score):.2f}" if score is not None else "z=n/a"
        cv2.putText(panel, stxt, (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Zoomed detections + zoom_cls", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    return panel


def _normalize_tile_uint8_rgb(tile_rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype).view(3, 1, 1)
    return (t - mean) / std


def compute_zoom_scores_for_polys(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    polys: List[List[List[float]]],
    tile_size: int,
    device: torch.device,
    use_amp: bool,
    max_polys: int = 24,
) -> List[Optional[float]]:
    # If checkpoint/model has no zoom head, return Nones.
    if getattr(model, "zoom_cls_head", None) is None:
        return [None for _ in polys]
    if len(polys) == 0:
        return []

    h, w = image_rgb.shape[:2]
    scores: List[Optional[float]] = [None for _ in polys]

    entries = []
    for i, poly in enumerate(polys):
        if len(poly) < 3:
            continue
        bx0, by0, bx1, by1 = poly_bbox(poly)
        bw = max(1.0, float(bx1 - bx0))
        bh = max(1.0, float(by1 - by0))
        mx = max(2.0, bw * 0.2)
        my = max(2.0, bh * 0.2)
        cx0 = int(max(0.0, np.floor(bx0 - mx)))
        cy0 = int(max(0.0, np.floor(by0 - my)))
        cx1 = int(min(float(w), np.ceil(bx1 + mx)))
        cy1 = int(min(float(h), np.ceil(by1 + my)))
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        entries.append((i, cx0, cy0, cx1, cy1, bx0, by0, bx1, by1, bw * bh))

    if len(entries) == 0:
        return scores

    entries.sort(key=lambda x: x[-1], reverse=True)
    entries = entries[: max(1, int(max_polys))]

    tiles = []
    boxes = []
    keep_indices = []
    for e in entries:
        i, cx0, cy0, cx1, cy1, bx0, by0, bx1, by1, _a = e
        crop = image_rgb[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        if ch < 2 or cw < 2:
            continue
        interp = cv2.INTER_AREA if (cw > tile_size or ch > tile_size) else cv2.INTER_LINEAR
        crop_rs = cv2.resize(crop, (tile_size, tile_size), interpolation=interp)
        tiles.append(_normalize_tile_uint8_rgb(crop_rs))

        sx = float(tile_size) / float(cw)
        sy = float(tile_size) / float(ch)
        zx0 = float((bx0 - cx0) * sx)
        zy0 = float((by0 - cy0) * sy)
        zx1 = float((bx1 - cx0) * sx)
        zy1 = float((by1 - cy0) * sy)
        zx0 = max(0.0, min(float(tile_size - 1), zx0))
        zy0 = max(0.0, min(float(tile_size - 1), zy0))
        zx1 = max(zx0 + 1.0, min(float(tile_size), zx1))
        zy1 = max(zy0 + 1.0, min(float(tile_size), zy1))
        boxes.append([zx0, zy0, zx1, zy1])
        keep_indices.append(i)

    if len(tiles) == 0:
        return scores

    x = torch.stack(tiles, dim=0).to(device)
    zoom_boxes = torch.tensor(boxes, dtype=torch.float32, device=device)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
    with torch.inference_mode():
        with autocast_ctx:
            pred = model(x, zoom_boxes=zoom_boxes)
            zl = pred.get("zoom_logit", None)
            if zl is None:
                return scores
            zp = torch.sigmoid(zl[:, 0]).detach().float().cpu().tolist()
    for i, p in zip(keep_indices, zp):
        scores[i] = float(p)
    return scores


def main() -> None:
    args = parse_args()
    if args.print_monitors:
        ScreenGrabber.print_monitors()
        return
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = load_model(args.checkpoint, device)
    print("device:", device)
    print("amp:", bool(args.amp and device.type == "cuda"))
    print("model:", info)
    print("zoom_cls_head:", bool(info.get("use_zoom_cls_head", False)))

    grabber = ScreenGrabber(
        monitor_index=args.monitor_index,
        x=args.x,
        y=args.y,
        width=args.width,
        height=args.height,
    )
    if grabber.region is not None:
        print(
            "capture_region:",
            f"left={grabber.region['left']} top={grabber.region['top']} "
            f"width={grabber.region['width']} height={grabber.region['height']}",
        )

    pred_threshold = float(args.pred_threshold)
    tile_cls_threshold = float(args.tile_cls_threshold)
    use_tile_cls_gating = bool(args.use_tile_cls_gating)
    tile_cls_mode = str(args.tile_cls_mode)
    use_amp = bool(args.amp and device.type == "cuda")
    infer_every = max(1, int(args.infer_every))
    min_dt = 0.0 if args.max_fps <= 0 else (1.0 / float(args.max_fps))
    auto_save_persistent = bool(args.auto_save_persistent)
    persist_infers = max(1, int(args.persist_infers))
    persist_iou_threshold = float(args.persist_iou_threshold)
    persist_max_miss = max(0, int(args.persist_max_miss))
    persist_save_cooldown_infers = max(0, int(args.persist_save_cooldown_infers))

    last_frame = None
    last_prob_full = None
    last_pred_mask = None
    last_polys = []
    last_zoom_scores: List[Optional[float]] = []
    last_stats = {"tile_cls_mean": float("nan")}
    frame_idx = 0
    paused = False
    last_t = time.time()
    fps_smooth = 0.0
    infer_idx = 0
    tracks: List[Dict] = []
    next_track_id = 1
    last_persistent_polys: List[List[List[float]]] = []
    auto_saved_count = 0
    last_auto_save_infer = -10**9

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    while True:
        now = time.time()
        dt = now - last_t
        if dt < min_dt:
            time.sleep(max(0.0, min_dt - dt))
            now = time.time()
            dt = now - last_t
        last_t = now
        if dt > 0:
            fps_inst = 1.0 / dt
            fps_smooth = fps_inst if fps_smooth <= 0 else (0.90 * fps_smooth + 0.10 * fps_inst)

        if not paused:
            frame_bgr = grabber.grab_bgr()
            frame_idx += 1
            if frame_bgr is None or frame_bgr.size == 0:
                continue
            last_frame = frame_bgr

            if (frame_idx % infer_every == 0) or (last_prob_full is None):
                infer_idx += 1
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w = image_rgb.shape[:2]
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp
                    else nullcontext()
                )
                with torch.inference_mode():
                    with autocast_ctx:
                        prob_lr, _cls_lr, stats = infer_prob_map(
                            model=model,
                            image_np=image_rgb,
                            tile_size=args.tile_size,
                            stride=args.tile_stride,
                            seg_out_stride=args.seg_out_stride,
                            device=device,
                            use_tile_cls_gating=use_tile_cls_gating,
                            tile_cls_threshold=tile_cls_threshold,
                            tile_cls_mode=tile_cls_mode,
                        )
                    prob_full = F.interpolate(
                        torch.from_numpy(prob_lr).float().unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0].numpy()
                pred_mask = (prob_full >= pred_threshold).astype(np.uint8)
                polys = mask_to_polygons(
                    pred_mask,
                    min_area=float(args.min_poly_area),
                    epsilon_frac=float(args.poly_epsilon_frac),
                )
                zoom_scores = compute_zoom_scores_for_polys(
                    model=model,
                    image_rgb=image_rgb,
                    polys=polys,
                    tile_size=int(args.tile_size),
                    device=device,
                    use_amp=use_amp,
                    max_polys=24,
                )
                last_prob_full = prob_full
                last_pred_mask = pred_mask
                last_polys = polys
                last_zoom_scores = zoom_scores
                last_stats = stats

                # Temporal persistence tracking on polygon detections.
                detections = []
                for poly in polys:
                    if len(poly) < 3:
                        continue
                    detections.append({"poly": poly, "bbox": poly_bbox(poly)})

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
                    out_img, out_json = save_current(
                        out_dir=args.output_dir / "persistent_auto",
                        frame_bgr=last_frame,
                        polys=last_persistent_polys,
                        label=args.label,
                        frame_idx=frame_idx,
                        preview_bgr=None,
                    )
                    auto_saved_count += 1
                    last_auto_save_infer = infer_idx
                    print(
                        f"[auto-saved persistent] {out_img.name} / {out_json.name} "
                        f"polygons={len(last_persistent_polys)} infer_idx={infer_idx}"
                    )

        if last_frame is None or last_prob_full is None or last_pred_mask is None:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            continue

        preview = build_live_preview(
            img_bgr=last_frame,
            prob_full=last_prob_full,
            pred_mask=last_pred_mask,
            polys=last_polys,
            zoom_scores=last_zoom_scores,
            stats=last_stats,
            fps=fps_smooth,
            threshold=pred_threshold,
            gating=use_tile_cls_gating,
            tile_cls_thr=tile_cls_threshold,
            tile_cls_mode=tile_cls_mode,
            paused=paused,
            persistent_count=len(last_persistent_polys),
            auto_saved_count=auto_saved_count,
            auto_save_persistent=auto_save_persistent,
            persist_infers=persist_infers,
        )
        cv2.imshow(args.window_name, preview)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            pred_threshold = min(0.99, pred_threshold + 0.02)
        elif key in (ord("-"), ord("_")):
            pred_threshold = max(0.01, pred_threshold - 0.02)
        elif key == ord("["):
            tile_cls_threshold = max(0.01, tile_cls_threshold - 0.02)
        elif key == ord("]"):
            tile_cls_threshold = min(0.99, tile_cls_threshold + 0.02)
        elif key in (ord("g"), ord("G")):
            use_tile_cls_gating = not use_tile_cls_gating
        elif key in (ord("m"), ord("M")):
            tile_cls_mode = "multiply" if tile_cls_mode == "hard" else "hard"
        elif key in (ord("t"), ord("T")):
            auto_save_persistent = not auto_save_persistent
            print(f"[toggle] auto_save_persistent={auto_save_persistent}")
        elif key in (ord("a"), ord("A")):
            out_img, out_json = save_current(
                out_dir=args.output_dir,
                frame_bgr=last_frame,
                polys=last_polys,
                label=args.label,
                frame_idx=frame_idx,
                preview_bgr=preview if args.save_preview else None,
            )
            print(f"[saved] {out_img.name} / {out_json.name} polygons={len(last_polys)}")

    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    main()
