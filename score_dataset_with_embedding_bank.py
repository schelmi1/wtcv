#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from build_embedding_bank import (
    _token_grid_shape,
    load_crop_tensor,
    masked_pool_from_tokens,
    polygon_mask,
    square_crop_with_pad,
)
from curate_model_predictions_to_labelme import infer_prob_map, load_model, mask_to_polygons
from wtcv_utils.labelme import IMG_EXTS, shape_to_points

FIXED_DINO_MODEL = "dinov2_vits14_reg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run model detections over a dataset, score each detection with embedding-bank similarity, and export LabelMe candidates"
    )
    ap.add_argument("--input-dir", type=Path, required=True, help="Folder containing images")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional Stage1 model checkpoint (.pt). If omitted, falls back to vanilla DINO scoring mode.",
    )
    ap.add_argument("--embedding-bank", type=Path, required=True, help="Path to embedding_bank.npz")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_vs_embedding_bank"))

    ap.add_argument("--pred-threshold", type=float, default=0.35)
    ap.add_argument("--min-poly-area", type=float, default=14.0)
    ap.add_argument("--poly-epsilon-frac", type=float, default=0.002)
    ap.add_argument("--tile-size", type=int, default=448)
    ap.add_argument("--tile-stride", type=int, default=448)
    ap.add_argument("--seg-out-stride", type=int, default=4)
    ap.add_argument("--use-tile-cls-gating", action="store_true", default=True)
    ap.add_argument("--no-use-tile-cls-gating", action="store_false", dest="use_tile_cls_gating")
    ap.add_argument("--tile-cls-threshold", type=float, default=0.5)
    ap.add_argument("--tile-cls-mode", type=str, choices=["hard", "multiply"], default="hard")
    ap.add_argument("--use-amp", action="store_true", default=False)

    ap.add_argument("--dino-model", type=str, default="", help="Blank uses bank manifest config or fallback")
    ap.add_argument(
        "--feature-backend",
        type=str,
        choices=["auto", "dino", "adapter"],
        default="auto",
        help="Embedding backend for candidate scoring.",
    )
    ap.add_argument(
        "--adapter-feature-key",
        type=str,
        choices=["feat_adapted", "feat_dino"],
        default="feat_adapted",
        help="Feature key used when --feature-backend=adapter.",
    )
    ap.add_argument(
        "--adapter-input-size",
        type=int,
        default=0,
        help="Optional square resize for adapter backend before forward (0 auto). Must be multiple of 256.",
    )
    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")
    ap.add_argument("--embed-batch-size", type=int, default=12)
    ap.add_argument("--obj-tile-size", type=int, default=0, help="0 uses bank manifest tile_size")
    ap.add_argument("--obj-context-scale", type=float, default=0.0, help="0 uses bank manifest tile_context_scale")

    ap.add_argument("--positive-labels", type=str, default="vehicle", help="Comma labels from bank for positive subset")
    ap.add_argument("--negative-labels", type=str, default="fp", help="Unused in additive-positive mode (kept for CLI compatibility)")
    ap.add_argument("--bank-topk", type=int, default=5)
    ap.add_argument(
        "--use-faiss",
        action="store_true",
        default=False,
        help="Use FAISS IndexFlatIP for top-k similarity search against positive bank (default: off).",
    )
    ap.add_argument("--no-use-faiss", action="store_false", dest="use_faiss")
    ap.add_argument("--neg-weight", type=float, default=1.0, help="Unused in additive-positive mode (kept for CLI compatibility)")
    ap.add_argument("--accept-score", type=float, default=0.35, help="Positive add threshold on pos_topk_mean")
    ap.add_argument("--fp-score", type=float, default=0.40, help="Unused in additive-positive mode (kept for CLI compatibility)")
    ap.add_argument("--fp-label", type=str, default="fp", help="Unused in additive-positive mode (kept for CLI compatibility)")
    ap.add_argument("--vehicle-label", type=str, default="vehicle", help="Base label for added candidates; `_auto` is appended automatically")
    ap.add_argument("--export-fp", action="store_true", default=True, help="Unused in additive-positive mode (kept for CLI compatibility)")
    ap.add_argument("--no-export-fp", action="store_false", dest="export_fp")
    ap.add_argument("--dedup-iou", type=float, default=0.30, help="Skip candidate if IoU with existing/added positive polygon >= this value")

    ap.add_argument("--max-images", type=int, default=0, help="0 means all")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu, blank=auto")
    return ap.parse_args()


def _parse_labels(raw: str) -> List[str]:
    out = [x.strip().casefold() for x in str(raw).split(",") if x.strip()]
    return sorted(set(out))


def _discover_images(input_dir: Path, max_images: int) -> List[Path]:
    exts = {str(e).lower() for e in IMG_EXTS}
    images = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if max_images > 0:
        images = images[: int(max_images)]
    return images


def _load_bank_manifest_defaults(bank_npz: Path) -> Dict:
    m = bank_npz.parent / "embedding_bank_manifest.json"
    if not m.exists():
        return {}
    try:
        d = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def _load_bank_with_subsets(
    bank_npz: Path,
    pos_labels_cf: Sequence[str],
    neg_labels_cf: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    data = np.load(bank_npz, allow_pickle=False)
    if "embeddings" not in data:
        raise RuntimeError(f"Missing embeddings in bank npz: {bank_npz}")
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    label_table = None
    label_ids = None
    if "labels" in data and "label_ids" in data:
        try:
            label_table = [str(x).casefold() for x in np.asarray(data["labels"]).tolist()]
            label_ids = np.asarray(data["label_ids"], dtype=np.int64)
            if len(label_ids) != emb.shape[0]:
                label_table = None
                label_ids = None
        except Exception:
            label_table = None
            label_ids = None

    idx_all = np.arange(emb.shape[0], dtype=np.int64)
    pos_idx = idx_all
    neg_idx = np.zeros((0,), dtype=np.int64)

    if label_table is not None and label_ids is not None:
        if len(pos_labels_cf) > 0:
            pos_mask = np.array(
                [label_table[int(lid)] in set(pos_labels_cf) for lid in label_ids],
                dtype=bool,
            )
            if bool(pos_mask.any()):
                pos_idx = np.where(pos_mask)[0]
        if len(neg_labels_cf) > 0:
            neg_mask = np.array(
                [label_table[int(lid)] in set(neg_labels_cf) for lid in label_ids],
                dtype=bool,
            )
            if bool(neg_mask.any()):
                neg_idx = np.where(neg_mask)[0]

    info = {
        "num_total": int(emb.shape[0]),
        "num_pos": int(pos_idx.shape[0]),
        "num_neg": int(neg_idx.shape[0]),
        "has_label_subsets": bool(label_table is not None and label_ids is not None),
    }
    return emb[pos_idx], emb[neg_idx], info


def _poly_bbox(poly: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _poly_iou(a: List[List[float]], b: List[List[float]]) -> float:
    ax0, ay0, ax1, ay1 = _poly_bbox(a)
    bx0, by0, bx1, by1 = _poly_bbox(b)
    x0 = int(math.floor(min(ax0, bx0)))
    y0 = int(math.floor(min(ay0, by0)))
    x1 = int(math.ceil(max(ax1, bx1)))
    y1 = int(math.ceil(max(ay1, by1)))
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    ma = np.zeros((h, w), dtype=np.uint8)
    mb = np.zeros((h, w), dtype=np.uint8)
    pa = np.array([[(float(p[0]) - x0), (float(p[1]) - y0)] for p in a], dtype=np.float32).reshape(-1, 2)
    pb = np.array([[(float(p[0]) - x0), (float(p[1]) - y0)] for p in b], dtype=np.float32).reshape(-1, 2)
    if pa.shape[0] < 3 or pb.shape[0] < 3:
        return 0.0
    cv2.fillPoly(ma, [pa.astype(np.int32)], 1)
    cv2.fillPoly(mb, [pb.astype(np.int32)], 1)
    inter = int(((ma > 0) & (mb > 0)).sum())
    union = int(((ma > 0) | (mb > 0)).sum())
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _make_crop_and_mask(
    image_rgb: np.ndarray,
    poly: List[List[float]],
    tile_size: int,
    context_scale: float,
) -> Tuple[torch.Tensor, np.ndarray] | None:
    x0, y0, x1, y1 = _poly_bbox(poly)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    side = int(max(16.0, math.ceil(max(bw, bh) * float(context_scale))))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    tx0 = int(round(cx - 0.5 * side))
    ty0 = int(round(cy - 0.5 * side))

    crop = square_crop_with_pad(image_rgb, tx0, ty0, side)
    crop_pil = Image.fromarray(crop).resize((tile_size, tile_size), Image.BILINEAR)
    crop_np = np.array(crop_pil, dtype=np.uint8)

    sx = float(tile_size) / float(side)
    sy = float(tile_size) / float(side)
    cpts = [[(float(p[0]) - float(tx0)) * sx, (float(p[1]) - float(ty0)) * sy] for p in poly]
    cpts = [[max(0.0, min(float(tile_size - 1), p[0])), max(0.0, min(float(tile_size - 1), p[1]))] for p in cpts]
    if len(cpts) < 3:
        return None
    mask = polygon_mask(cpts, w=tile_size, h=tile_size)
    if int(mask.sum()) <= 0:
        return None
    return load_crop_tensor(crop_np), mask


def _embed_polygons_dino(
    image_rgb: np.ndarray,
    polys: Sequence[List[List[float]]],
    dino: torch.nn.Module,
    device: torch.device,
    tile_size: int,
    context_scale: float,
    batch_size: int,
) -> List[np.ndarray | None]:
    out: List[np.ndarray | None] = [None] * len(polys)
    pending_t: List[torch.Tensor] = []
    pending_m: List[np.ndarray] = []
    pending_i: List[int] = []

    def flush() -> None:
        nonlocal pending_t, pending_m, pending_i
        if not pending_t:
            return
        x = torch.stack(pending_t, dim=0).to(device)
        with torch.inference_mode():
            feats = dino.forward_features(x)
            tok = feats["x_norm_patchtokens"]
            b, n, c = tok.shape
            gh, gw = _token_grid_shape(int(n))
            fmap = tok.reshape(b, gh, gw, c).permute(0, 3, 1, 2).contiguous()
        for j in range(fmap.shape[0]):
            v = masked_pool_from_tokens(fmap[j], pending_m[j])
            v = F.normalize(v, dim=0)
            out[int(pending_i[j])] = v.detach().cpu().numpy().astype(np.float32)
        pending_t = []
        pending_m = []
        pending_i = []

    for i, poly in enumerate(polys):
        cm = _make_crop_and_mask(image_rgb, poly=poly, tile_size=tile_size, context_scale=context_scale)
        if cm is None:
            continue
        t, m = cm
        pending_t.append(t)
        pending_m.append(m)
        pending_i.append(int(i))
        if len(pending_t) >= max(1, int(batch_size)):
            flush()
    flush()
    return out


def _embed_polygons_adapter(
    image_rgb: np.ndarray,
    polys: Sequence[List[List[float]]],
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int,
    context_scale: float,
    batch_size: int,
    feature_key: str,
    adapter_input_size: int,
) -> List[np.ndarray | None]:
    out: List[np.ndarray | None] = [None] * len(polys)
    pending_t: List[torch.Tensor] = []
    pending_m: List[np.ndarray] = []
    pending_i: List[int] = []

    def flush() -> None:
        nonlocal pending_t, pending_m, pending_i
        if not pending_t:
            return
        x = torch.stack(pending_t, dim=0)
        if int(adapter_input_size) > 0 and int(adapter_input_size) != int(x.shape[-1]):
            x = F.interpolate(x, size=(int(adapter_input_size), int(adapter_input_size)), mode="bilinear", align_corners=False)
        x = x.to(device)
        with torch.inference_mode():
            pred = model(x, return_features=True)
            if feature_key not in pred:
                raise RuntimeError(
                    f"Adapter feature key '{feature_key}' not found. Available keys: {sorted(list(pred.keys()))}"
                )
            fmap = pred[feature_key]
        for j in range(int(fmap.shape[0])):
            v = masked_pool_from_tokens(fmap[j], pending_m[j])
            v = F.normalize(v, dim=0)
            out[int(pending_i[j])] = v.detach().cpu().numpy().astype(np.float32)
        pending_t = []
        pending_m = []
        pending_i = []

    for i, poly in enumerate(polys):
        cm = _make_crop_and_mask(image_rgb, poly=poly, tile_size=tile_size, context_scale=context_scale)
        if cm is None:
            continue
        t, m = cm
        pending_t.append(t)
        pending_m.append(m)
        pending_i.append(int(i))
        if len(pending_t) >= max(1, int(batch_size)):
            flush()
    flush()
    return out


def _topk_scores(query: np.ndarray, bank: np.ndarray, k: int) -> Tuple[float, float]:
    if bank.shape[0] <= 0:
        return float("nan"), float("nan")
    sims = np.matmul(bank, query.astype(np.float32))
    if sims.size == 0:
        return float("nan"), float("nan")
    kk = max(1, min(int(k), int(sims.shape[0])))
    idx = np.argpartition(sims, -kk)[-kk:]
    topk = sims[idx]
    return float(np.mean(topk)), float(np.max(topk))


def _build_faiss_index_ip(bank: np.ndarray):
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS requested with --use-faiss but import failed. "
            "Install with `pip install faiss-cpu` (or faiss-gpu where supported)."
        ) from e
    if bank.ndim != 2 or int(bank.shape[0]) <= 0 or int(bank.shape[1]) <= 0:
        raise RuntimeError(f"Invalid bank shape for FAISS: {tuple(bank.shape)}")
    d = int(bank.shape[1])
    index = faiss.IndexFlatIP(d)
    bank_c = np.ascontiguousarray(bank.astype(np.float32))
    index.add(bank_c)
    return index


def _topk_scores_faiss(query: np.ndarray, faiss_index, k: int) -> Tuple[float, float]:
    ntotal = int(faiss_index.ntotal)
    if ntotal <= 0:
        return float("nan"), float("nan")
    qq = np.ascontiguousarray(query.astype(np.float32).reshape(1, -1))
    kk = max(1, min(int(k), ntotal))
    sims, _idx = faiss_index.search(qq, kk)
    if sims.size == 0:
        return float("nan"), float("nan")
    topk = sims[0]
    return float(np.mean(topk)), float(np.max(topk))


def _mask_stats_for_poly(prob_full: np.ndarray, poly: List[List[float]]) -> Tuple[float, float, int]:
    h, w = prob_full.shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return float("nan"), float("nan"), 0
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
    cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    sel = prob_full[m > 0]
    if sel.size <= 0:
        return float("nan"), float("nan"), 0
    return float(sel.mean()), float(sel.max()), int(sel.size)


def _make_labelme_json(image_name: str, h: int, w: int, shapes: List[Dict]) -> Dict:
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(image_name),
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def _load_existing_labelme_for_image(ip: Path, h: int, w: int) -> Dict:
    jp = ip.with_suffix(".json")
    if jp.exists():
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return _make_labelme_json(image_name=ip.name, h=h, w=w, shapes=[])


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {args.input_dir}")
    if not args.embedding_bank.exists():
        raise FileNotFoundError(f"Missing embedding bank: {args.embedding_bank}")

    checkpoint: Path | None = None
    if args.checkpoint is not None:
        ckpt_str = str(args.checkpoint).strip()
        # Handles UI-provided empty string -> Path(".") edge-case.
        if ckpt_str not in {"", "."}:
            checkpoint = Path(args.checkpoint)
            if (not checkpoint.exists()) or (not checkpoint.is_file()):
                raise FileNotFoundError(f"Missing checkpoint file: {checkpoint}")

    manifest = _load_bank_manifest_defaults(args.embedding_bank)
    cfg_m = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    dino_model_name = str(args.dino_model).strip() or str(cfg_m.get("dino_model", FIXED_DINO_MODEL))
    obj_tile_size = int(args.obj_tile_size) if int(args.obj_tile_size) > 0 else int(cfg_m.get("tile_size", 448))
    obj_context_scale = (
        float(args.obj_context_scale) if float(args.obj_context_scale) > 0 else float(cfg_m.get("tile_context_scale", 2.0))
    )

    pos_labels = _parse_labels(args.positive_labels)
    pos_label_set = set(pos_labels) if len(pos_labels) > 0 else {str(args.vehicle_label).strip().casefold()}
    neg_labels = _parse_labels(args.negative_labels)
    pos_bank, _neg_bank_unused, bank_info = _load_bank_with_subsets(
        args.embedding_bank,
        pos_labels_cf=pos_labels,
        neg_labels_cf=neg_labels,
    )
    if pos_bank.shape[0] <= 0:
        raise RuntimeError("Positive embedding-bank subset is empty. Check --positive-labels or bank content.")
    faiss_index = None
    if bool(args.use_faiss):
        faiss_index = _build_faiss_index_ip(pos_bank)

    device = torch.device(args.device.strip() if args.device.strip() else ("cuda" if torch.cuda.is_available() else "cpu"))
    images = _discover_images(args.input_dir, max_images=int(args.max_images))
    if len(images) <= 0:
        raise RuntimeError(f"No images found in: {args.input_dir}")

    feature_backend = str(args.feature_backend).strip().lower()
    if feature_backend == "auto":
        fb_m = str(cfg_m.get("feature_backend", "dino")).strip().lower()
        feature_backend = fb_m if fb_m in {"dino", "adapter"} else "dino"
        print(f"feature_backend_auto_resolved={feature_backend}")

    if checkpoint is None:
        # No adapter checkpoint available: force vanilla DINO embedding path.
        if feature_backend != "dino":
            print("note: no checkpoint specified; forcing feature_backend=dino")
        feature_backend = "dino"

    if feature_backend == "dino":
        if int(obj_tile_size) % 14 != 0:
            raise ValueError(
                f"DINO backend requires object tile size divisible by 14, got obj_tile_size={int(obj_tile_size)}. "
                "Use --obj-tile-size divisible by 14 (e.g. 448 or 504) or set --feature-backend=adapter."
            )
    if feature_backend == "adapter":
        if int(args.adapter_input_size) > 0 and (int(args.adapter_input_size) % 256) != 0:
            raise ValueError("--adapter-input-size must be multiple of 256")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    out_labelme = out / "labelme_scored"
    out_labelme.mkdir(parents=True, exist_ok=True)
    out_report = out / "detections_scored.jsonl"

    model = None
    info: Dict = {}
    if checkpoint is not None:
        model, info = load_model(checkpoint, device)
    else:
        print("note: running without checkpoint; using existing LabelMe shapes as candidates")
    dino = None
    embed_model = None
    if feature_backend == "dino":
        dino = torch.hub.load("facebookresearch/dinov2", dino_model_name, trust_repo=bool(args.trust_torch_hub_repo)).to(device).eval()
        for p in dino.parameters():
            p.requires_grad = False
    else:
        if model is None:
            raise RuntimeError("Adapter backend requires a checkpoint model.")
        embed_model = model
        print("adapter_embedding_model_checkpoint=<checkpoint>")
        if int(args.adapter_input_size) <= 0:
            if int(obj_tile_size) % 256 == 0:
                args.adapter_input_size = int(obj_tile_size)
            elif int(args.tile_size) % 256 == 0:
                args.adapter_input_size = int(args.tile_size)
            else:
                raise ValueError(
                    "Adapter backend requires adapter input multiple of 256, but both obj_tile_size and tile_size are not multiples of 256. "
                    "Set --adapter-input-size explicitly (e.g. 512)."
                )

    print(f"device={device}")
    print(f"checkpoint={checkpoint if checkpoint is not None else 'None'}")
    print(f"feature_backend={feature_backend}")
    if feature_backend == "dino":
        print(f"dino_model={dino_model_name}")
    else:
        print(f"adapter_feature_key={args.adapter_feature_key}")
        print(f"adapter_input_size={args.adapter_input_size}")
    print(f"input_images={len(images)}")
    print(f"bank_total={bank_info['num_total']} bank_pos={bank_info['num_pos']} bank_neg={bank_info['num_neg']} (neg currently unused)")
    print(f"use_faiss={bool(args.use_faiss)}")
    if faiss_index is not None:
        print(f"faiss_index=IndexFlatIP dim={int(pos_bank.shape[1])} ntotal={int(faiss_index.ntotal)}")
    print(f"det_tile={args.tile_size}/{args.tile_stride} obj_tile={obj_tile_size} obj_context={obj_context_scale}")
    print(f"use_tile_cls_gating={args.use_tile_cls_gating} tile_cls_threshold={args.tile_cls_threshold} mode={args.tile_cls_mode}")
    added_label = str(args.vehicle_label).strip()
    if not added_label:
        added_label = "vehicle"
    if not added_label.endswith("_auto"):
        added_label = f"{added_label}_auto"

    print(f"accept_score={args.accept_score} dedup_iou={args.dedup_iou} vehicle_label={args.vehicle_label} added_label={added_label}")
    print(f"model_info={info if checkpoint is not None else 'N/A (no checkpoint mode)'}")
    print("strategy=keep original labels, add positive candidates only")

    total_polys = 0
    total_original_shapes = 0
    total_added = 0
    total_skipped_score = 0
    total_skipped_duplicate = 0

    with out_report.open("w", encoding="utf-8") as f_report:
        for ip in tqdm(images, desc="dataset vs bank", unit="img"):
            image_rgb = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
            h, w = image_rgb.shape[:2]
            src_labelme = _load_existing_labelme_for_image(ip, h=h, w=w)
            src_shapes = deepcopy(src_labelme.get("shapes", []) or [])
            total_original_shapes += int(len(src_shapes))
            existing_pos_polys: List[List[List[float]]] = []
            for s in src_shapes:
                lab = str(s.get("label", "")).strip().casefold()
                if lab not in pos_label_set:
                    continue
                pts = shape_to_points(s, min_poly_points=3)
                if pts is None:
                    continue
                existing_pos_polys.append(pts)

            stats = {
                "tile_cls_used": float("nan"),
                "tile_cls_mean": float("nan"),
            }
            if checkpoint is not None:
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
                polys = mask_to_polygons(
                    pred_mask,
                    min_area=float(args.min_poly_area),
                    epsilon_frac=float(args.poly_epsilon_frac),
                )
            else:
                # No detector checkpoint: score existing non-positive shapes as candidates.
                prob_full = np.zeros((h, w), dtype=np.float32)
                polys = []
                for s in src_shapes:
                    lab = str(s.get("label", "")).strip().casefold()
                    if lab in pos_label_set:
                        continue
                    pts = shape_to_points(s, min_poly_points=3)
                    if pts is None:
                        continue
                    polys.append(pts)
            if feature_backend == "dino":
                emb_list = _embed_polygons_dino(
                    image_rgb=image_rgb,
                    polys=polys,
                    dino=dino,
                    device=device,
                    tile_size=int(obj_tile_size),
                    context_scale=float(obj_context_scale),
                    batch_size=int(args.embed_batch_size),
                )
            else:
                emb_list = _embed_polygons_adapter(
                    image_rgb=image_rgb,
                    polys=polys,
                    model=embed_model,
                    device=device,
                    tile_size=int(obj_tile_size),
                    context_scale=float(obj_context_scale),
                    batch_size=int(args.embed_batch_size),
                    feature_key=str(args.adapter_feature_key),
                    adapter_input_size=int(args.adapter_input_size),
                )

            added_shapes: List[Dict] = []
            accepted_polys: List[List[List[float]]] = list(existing_pos_polys)
            for di, poly in enumerate(polys):
                total_polys += 1
                e = emb_list[di]
                conf_mean, conf_max, area_px = _mask_stats_for_poly(prob_full, poly)
                pos_mean = float("nan")
                pos_max = float("nan")
                action = "skip_low_score"
                skip_reason = "score"
                if e is not None:
                    if int(e.shape[0]) != int(pos_bank.shape[1]):
                        raise RuntimeError(
                            f"Embedding dim mismatch: query={int(e.shape[0])} bank={int(pos_bank.shape[1])}. "
                            "Use a bank built with the same feature backend/checkpoint."
                        )
                    if faiss_index is not None:
                        pos_mean, pos_max = _topk_scores_faiss(e, faiss_index, int(args.bank_topk))
                    else:
                        pos_mean, pos_max = _topk_scores(e, pos_bank, int(args.bank_topk))
                    if (not math.isnan(pos_mean)) and float(pos_mean) >= float(args.accept_score):
                        dup = False
                        for ep in accepted_polys:
                            if _poly_iou(poly, ep) >= float(args.dedup_iou):
                                dup = True
                                break
                        if dup:
                            action = "skip_duplicate"
                            skip_reason = "dedup_iou"
                            total_skipped_duplicate += 1
                        else:
                            action = "add_positive"
                            skip_reason = ""
                            total_added += 1
                            accepted_polys.append(poly)
                            added_shapes.append(
                                {
                                    "label": str(added_label),
                                    "points": poly,
                                    "group_id": None,
                                    "shape_type": "polygon",
                                    "flags": {},
                                }
                            )
                    else:
                        total_skipped_score += 1
                else:
                    total_skipped_score += 1

                row = {
                    "image_path": str(ip),
                    "det_index": int(di),
                    "action": action,
                    "skip_reason": skip_reason,
                    "area_px": int(area_px),
                    "pred_mean": conf_mean,
                    "pred_max": conf_max,
                    "score_pos_topk_mean": pos_mean,
                    "score_pos_topk_max": pos_max,
                    "accept_score": float(args.accept_score),
                    "polygon": poly,
                    "tile_cls_used": float(stats.get("tile_cls_used", float("nan"))),
                    "tile_cls_mean": float(stats.get("tile_cls_mean", float("nan"))),
                }
                f_report.write(json.dumps(row, ensure_ascii=True) + "\n")

            out_img = out_labelme / ip.name
            out_json = out_labelme / f"{ip.stem}.json"
            shutil.copy2(ip, out_img)
            merged_shapes = src_shapes + added_shapes
            d = deepcopy(src_labelme)
            d["imagePath"] = out_img.name
            d["imageData"] = None
            d["imageHeight"] = int(h)
            d["imageWidth"] = int(w)
            d["shapes"] = merged_shapes
            out_json.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

            image_row = {
                "image_path": str(ip),
                "action": "image_summary",
                "original_shapes": int(len(src_shapes)),
                "added_positive_shapes": int(len(added_shapes)),
                "merged_shapes": int(len(merged_shapes)),
                "tile_cls_used": float(stats.get("tile_cls_used", float("nan"))),
                "tile_cls_mean": float(stats.get("tile_cls_mean", float("nan"))),
            }
            f_report.write(json.dumps(image_row, ensure_ascii=True) + "\n")

    summary = {
        "input_dir": str(args.input_dir),
        "checkpoint": str(args.checkpoint),
        "embedding_bank": str(args.embedding_bank),
        "output_dir": str(out),
        "num_images": int(len(images)),
        "num_predicted_polygons_total": int(total_polys),
        "num_original_shapes_total": int(total_original_shapes),
        "num_added_positive_shapes": int(total_added),
        "num_skipped_low_score": int(total_skipped_score),
        "num_skipped_duplicate": int(total_skipped_duplicate),
        "dino_model": str(dino_model_name),
        "feature_backend": str(feature_backend),
        "adapter_feature_key": str(args.adapter_feature_key),
        "adapter_input_size": int(args.adapter_input_size),
        "obj_tile_size": int(obj_tile_size),
        "obj_context_scale": float(obj_context_scale),
        "use_faiss": bool(args.use_faiss),
        "accept_score": float(args.accept_score),
        "dedup_iou": float(args.dedup_iou),
        "vehicle_label": str(args.vehicle_label),
        "added_label": str(added_label),
        "positive_labels": sorted(pos_label_set),
        "bank_subset_info": bank_info,
        "strategy": "preserve_original_labels_and_add_positive_candidates_only",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
