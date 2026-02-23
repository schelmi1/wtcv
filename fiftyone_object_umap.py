#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
from curate_model_predictions_to_labelme import load_model

try:
    import umap  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import `umap-learn` (install with `pip install umap-learn`, "
        "and verify numba cache/runtime is healthy)."
    ) from e

try:
    from sklearn.cluster import KMeans
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency `scikit-learn`. Install with: pip install scikit-learn") from e

try:
    import fiftyone as fo
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency `fiftyone`. Install with: pip install fiftyone") from e
try:
    import fiftyone.brain as fob
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency `fiftyone-brain`. Install with: pip install fiftyone-brain") from e

from wtcv_utils.labelme import polygon_area, polygon_bbox, shape_to_points
from wtcv_utils.records import load_labelme_pairs

DEFAULT_DINO_MODEL = "dinov2_vits14_reg"
DEFAULT_SAM1_MODEL = "facebook/sam-vit-base"
DEFAULT_SAM2_MODEL = "facebook/sam2-hiera-tiny"


def _infer_backend_from_backbone_model(model_name: str) -> Optional[str]:
    m = str(model_name).strip().lower()
    if not m:
        return None
    if "sam2" in m or "hiera" in m:
        return "sam2"
    if "/sam-" in m or m.startswith("sam-") or m.startswith("facebook/sam"):
        return "sam1"
    if "dinov2" in m:
        return "dino"
    return None


@dataclass
class ObjMeta:
    source_image: Path
    source_json: Path
    source_label: str
    source_label_cf: str
    source_obj_idx: int
    image_w: int
    image_h: int
    points: List[List[float]]
    bbox_xyxy: Tuple[float, float, float, float]
    area_ratio: float
    crop_path: Path
    crop_points: List[List[float]]


def _build_crops_for_pair(
    pair_idx: int,
    image_path: str,
    json_path: str,
    json_data: Dict,
    label_set: Sequence[str],
    tile_size: int,
    tile_context_scale: float,
    min_poly_points: int,
    crops_dir: str,
) -> List[Dict]:
    ip = Path(image_path)
    jf = Path(json_path)
    try:
        img = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
    except Exception:
        return []

    h, w = img.shape[:2]
    img_area = float(max(1, w * h))
    out: List[Dict] = []
    shapes = json_data.get("shapes", []) or []
    label_set_cf = {str(x).strip().casefold() for x in label_set if str(x).strip()}
    crops_root = Path(crops_dir)

    for sidx, s in enumerate(shapes):
        lab = str(s.get("label", "")).strip()
        lab_cf = lab.casefold()
        if label_set_cf and (lab_cf not in label_set_cf):
            continue
        pts = shape_to_points(s, min_poly_points=min_poly_points)
        if pts is None:
            continue
        x0, y0, x1, y1 = polygon_bbox(pts)
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        side = int(max(16.0, np.ceil(max(bw, bh) * float(tile_context_scale))))
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        tx0 = int(round(cx - 0.5 * side))
        ty0 = int(round(cy - 0.5 * side))

        crop = square_crop_with_pad(img, tx0, ty0, side)
        interp = Image.BILINEAR
        crop_pil = Image.fromarray(crop).resize((tile_size, tile_size), interp)
        crop_np = np.array(crop_pil, dtype=np.uint8)

        sx = float(tile_size) / float(side)
        sy = float(tile_size) / float(side)
        cpts = [[(float(p[0]) - float(tx0)) * sx, (float(p[1]) - float(ty0)) * sy] for p in pts]
        cpts = [[max(0.0, min(float(tile_size - 1), p[0])), max(0.0, min(float(tile_size - 1), p[1]))] for p in cpts]

        stem = f"{pair_idx:06d}_{ip.stem}__obj{sidx:04d}"
        crop_path = crops_root / f"{stem}.jpg"
        Image.fromarray(crop_np).save(crop_path, quality=95)

        out.append(
            {
                "source_image": str(ip),
                "source_json": str(jf),
                "source_label": lab if lab else "unknown",
                "source_label_cf": lab_cf if lab else "unknown",
                "source_obj_idx": int(sidx),
                "image_w": int(w),
                "image_h": int(h),
                "points": pts,
                "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                "area_ratio": float(max(0.0, polygon_area(pts)) / img_area),
                "crop_path": str(crop_path),
                "crop_points": cpts,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build object-level FiftyOne dataset with DINO mask-pooled embeddings + UMAP clustering")
    ap.add_argument("--input-dir", type=Path, required=True, help="LabelMe image/json pairs")
    ap.add_argument("--dataset-name", type=str, default="wtcv_object_umap")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/fiftyone_object_umap"))
    ap.add_argument("--label-filter", type=str, default="vehicle,fp", help="Comma-separated labels to include, case-insensitive")
    ap.add_argument("--max-objects", type=int, default=0, help="0 means all")

    ap.add_argument("--tile-size", type=int, default=448)
    ap.add_argument("--tile-context-scale", type=float, default=2.0, help="Crop side = max(w,h) * scale around object bbox")
    ap.add_argument("--min-poly-points", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument(
        "--crop-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel workers for object crop building (default: CPU/2, min 1).",
    )
    ap.add_argument(
        "--feature-backend",
        type=str,
        choices=["auto", "dino", "adapter", "sam1", "sam2"],
        default="auto",
        help="Feature extractor backend. auto=adapter when checkpoint provided, else dino.",
    )
    ap.add_argument(
        "--adapter-checkpoint",
        type=str,
        default="",
        help="Optional Stage1 checkpoint path. If set (and backend=auto), adapter features are used.",
    )
    ap.add_argument(
        "--adapter-feature-key",
        type=str,
        choices=["feat_adapted", "feat_dino"],
        default="feat_adapted",
        help="Feature map key from Stage1 model used for masked pooling.",
    )
    ap.add_argument(
        "--adapter-input-size",
        type=int,
        default=0,
        help="Optional square resize before adapter forward (0 keeps tile_size). Must be multiple of 256.",
    )

    ap.add_argument("--dino-model", type=str, default=DEFAULT_DINO_MODEL)
    ap.add_argument(
        "--backbone-model",
        type=str,
        default="",
        help=(
            "Optional explicit backbone model string. For dino backend: torch.hub model "
            "(e.g. dinov2_vits14_reg, dinov2_vitb14). For sam1/sam2: Hugging Face model id "
            "(e.g. facebook/sam-vit-base, facebook/sam2-hiera-tiny)."
        ),
    )
    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu; default auto")

    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.05)
    ap.add_argument("--umap-metric", type=str, default="cosine")
    ap.add_argument("--num-clusters", type=int, default=20)
    ap.add_argument("--run-kmeans", action="store_true", default=False, help="If set, run KMeans and write cluster labels.")
    ap.add_argument("--no-run-kmeans", action="store_false", dest="run_kmeans")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--overwrite-dataset", action="store_true", default=True)
    ap.add_argument("--no-overwrite-dataset", action="store_false", dest="overwrite_dataset")
    ap.add_argument("--brain-key", type=str, default="obj_umap")
    ap.add_argument("--compute-visualization", action="store_true", default=True)
    ap.add_argument("--no-compute-visualization", action="store_false", dest="compute_visualization")
    ap.add_argument("--launch", action="store_true", default=False, help="Launch FiftyOne App at the end")
    return ap.parse_args()


def square_crop_with_pad(img_rgb: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x1, y1 = x0 + side, y0 + side
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    crop = img_rgb[sy0:sy1, sx0:sx1]
    out = np.zeros((side, side, 3), dtype=np.uint8)
    ox, oy = sx0 - x0, sy0 - y0
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def polygon_mask(points: List[List[float]], w: int, h: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(m)
    if len(points) >= 3:
        draw.polygon([(float(p[0]), float(p[1])) for p in points], fill=1)
    return np.array(m, dtype=np.uint8)


def build_object_crops(
    input_dir: Path,
    output_dir: Path,
    label_filter: Sequence[str],
    tile_size: int,
    tile_context_scale: float,
    min_poly_points: int,
    max_objects: int,
    crop_workers: int,
) -> List[ObjMeta]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    label_set = {str(x).strip().casefold() for x in label_filter if str(x).strip()}
    metas: List[ObjMeta] = []
    pairs = load_labelme_pairs(
        input_dir,
        load_workers=8,
        progress_desc="scan labelme",
        progress_leave=True,
    )
    tasks = [
        (
            i,
            str(pair.image_path),
            str(pair.json_path),
            pair.json_data,
            tuple(label_set),
            int(tile_size),
            float(tile_context_scale),
            int(min_poly_points),
            str(crops_dir),
        )
        for i, pair in enumerate(pairs)
    ]
    if len(tasks) == 0:
        return metas
    workers = int(crop_workers)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) // 2)
    workers = max(1, workers)
    pbar = tqdm(total=len(tasks), desc="build object crops")
    if workers == 1:
        for t in tasks:
            rows = _build_crops_for_pair(*t)
            for r in rows:
                metas.append(
                    ObjMeta(
                        source_image=Path(r["source_image"]),
                        source_json=Path(r["source_json"]),
                        source_label=str(r["source_label"]),
                        source_label_cf=str(r["source_label_cf"]),
                        source_obj_idx=int(r["source_obj_idx"]),
                        image_w=int(r["image_w"]),
                        image_h=int(r["image_h"]),
                        points=r["points"],
                        bbox_xyxy=tuple(r["bbox_xyxy"]),
                        area_ratio=float(r["area_ratio"]),
                        crop_path=Path(r["crop_path"]),
                        crop_points=r["crop_points"],
                    )
                )
            pbar.update(1)
            if (len(metas) % 100) == 0:
                pbar.set_postfix(objects=len(metas))
            if max_objects > 0 and len(metas) >= max_objects:
                pbar.close()
                return metas[: int(max_objects)]
    else:
        t_cols = list(zip(*tasks))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rows in ex.map(_build_crops_for_pair, *t_cols, chunksize=8):
                for r in rows:
                    metas.append(
                        ObjMeta(
                            source_image=Path(r["source_image"]),
                            source_json=Path(r["source_json"]),
                            source_label=str(r["source_label"]),
                            source_label_cf=str(r["source_label_cf"]),
                            source_obj_idx=int(r["source_obj_idx"]),
                            image_w=int(r["image_w"]),
                            image_h=int(r["image_h"]),
                            points=r["points"],
                            bbox_xyxy=tuple(r["bbox_xyxy"]),
                            area_ratio=float(r["area_ratio"]),
                            crop_path=Path(r["crop_path"]),
                            crop_points=r["crop_points"],
                        )
                    )
                pbar.update(1)
                if (len(metas) % 100) == 0:
                    pbar.set_postfix(objects=len(metas))
                if max_objects > 0 and len(metas) >= max_objects:
                    break
    pbar.close()
    if max_objects > 0:
        return metas[: int(max_objects)]
    return metas


def load_crop_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = TF.to_tensor(img)
    norm = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return norm(x)


def masked_pool_from_tokens(
    token_map: torch.Tensor,  # (C,H,W)
    mask_u8: np.ndarray,      # (tile, tile)
) -> torch.Tensor:
    c, h, w = token_map.shape
    m = torch.from_numpy(mask_u8.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m = F.interpolate(m, size=(h, w), mode="area")[0, 0].to(token_map.device).clamp(0.0, 1.0)
    den = m.sum()
    if float(den.item()) < 1e-6:
        # Fallback: global mean if mask collapses at token resolution.
        return token_map.mean(dim=(1, 2))
    vec = (token_map * m.unsqueeze(0)).sum(dim=(1, 2)) / den
    return vec


def compute_embeddings_dino(
    metas: List[ObjMeta],
    dino_model_name: str,
    device: torch.device,
    batch_size: int,
    trust_repo: bool,
) -> np.ndarray:
    model = torch.hub.load("facebookresearch/dinov2", dino_model_name, trust_repo=trust_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    embs: List[np.ndarray] = []
    pending_x: List[torch.Tensor] = []
    pending_masks: List[np.ndarray] = []

    def flush() -> None:
        nonlocal pending_x, pending_masks
        if len(pending_x) == 0:
            return
        x = torch.stack(pending_x, dim=0).to(device)
        with torch.inference_mode():
            feats = model.forward_features(x)
            tok = feats["x_norm_patchtokens"]  # (B,N,C)
            b, n, c = tok.shape
            hw = int(np.sqrt(n))
            fmap = tok.reshape(b, hw, hw, c).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        for i in range(fmap.shape[0]):
            v = masked_pool_from_tokens(fmap[i], pending_masks[i])
            v = F.normalize(v, dim=0)
            embs.append(v.detach().cpu().numpy().astype(np.float32))
        pending_x = []
        pending_masks = []

    for m in tqdm(metas, desc="dino masked pool"):
        x = load_crop_tensor(m.crop_path)
        mask = polygon_mask(m.crop_points, w=x.shape[-1], h=x.shape[-2])
        pending_x.append(x)
        pending_masks.append(mask)
        if len(pending_x) >= max(1, int(batch_size)):
            flush()
    flush()

    if len(embs) == 0:
        raise RuntimeError("No embeddings computed")
    return np.stack(embs, axis=0)


def compute_embeddings_sam(
    metas: List[ObjMeta],
    backend: str,
    model_name: str,
    device: torch.device,
    batch_size: int,
    tile_size: int,
) -> np.ndarray:
    backend = str(backend).strip().lower()
    if backend == "sam1":
        from transformers import SamModel, SamProcessor

        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(device).eval()
    elif backend == "sam2":
        from transformers import Sam2Model, Sam2Processor

        processor = Sam2Processor.from_pretrained(model_name)
        model = Sam2Model.from_pretrained(model_name).to(device).eval()
    else:
        raise ValueError(f"Unsupported SAM backend: {backend}")

    for p in model.parameters():
        p.requires_grad = False
    if backend == "sam2":
        print(f"sam2_forced_input_size={int(tile_size)}x{int(tile_size)}")
    elif backend == "sam1":
        print("sam1_forced_input_size=1024x1024 (model constraint)")

    embs: List[np.ndarray] = []
    pending_images: List[np.ndarray] = []
    pending_masks: List[np.ndarray] = []

    def _pick_feature_map(raw_embeddings: object) -> torch.Tensor:
        if isinstance(raw_embeddings, torch.Tensor):
            if raw_embeddings.ndim != 4:
                raise RuntimeError(f"SAM image embedding tensor must be 4D, got shape={tuple(raw_embeddings.shape)}")
            return raw_embeddings
        if isinstance(raw_embeddings, (list, tuple)):
            cands: List[torch.Tensor] = []
            for x in raw_embeddings:
                if isinstance(x, torch.Tensor) and x.ndim == 4:
                    cands.append(x)
            if not cands:
                raise RuntimeError("SAM returned list/tuple embeddings but no 4D tensor feature maps were found")
            # Prefer the highest spatial resolution map for object-level masked pooling.
            cands.sort(key=lambda t: int(t.shape[-2]) * int(t.shape[-1]), reverse=True)
            return cands[0]
        raise RuntimeError(f"Unsupported SAM image embedding type: {type(raw_embeddings)}")

    def _run_model_batch(images_batch: List[np.ndarray], masks_batch: List[np.ndarray]) -> None:
        if backend == "sam2":
            inputs = processor(
                images=images_batch,
                do_resize=True,
                size={"height": int(tile_size), "width": int(tile_size)},
                mask_size={"height": int(tile_size), "width": int(tile_size)},
                return_tensors="pt",
            )
        elif backend == "sam1":
            # SAM1 image encoder is configured for 1024x1024; keep processor defaults.
            inputs = processor(images=images_batch, return_tensors="pt")
        else:
            inputs = processor(images=images_batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            try:
                raw = model.get_image_embeddings(pixel_values=pixel_values)
            except RuntimeError as e:
                if backend == "sam2" and "view size is not compatible" in str(e):
                    # Workaround for transformers SAM2 get_image_embeddings using view() on non-contiguous tensors.
                    image_outputs = model.get_image_features(pixel_values, return_dict=True)
                    feature_maps = list(image_outputs.fpn_hidden_states)
                    if len(feature_maps) == 0:
                        raise RuntimeError("SAM2 fallback: empty feature maps from get_image_features") from e
                    feature_maps[-1] = feature_maps[-1] + model.no_memory_embedding
                    raw = []
                    for feat, feat_size in zip(feature_maps, model.backbone_feature_sizes):
                        t = feat.permute(1, 2, 0).contiguous().reshape(pixel_values.shape[0], -1, *feat_size)
                        raw.append(t)
                else:
                    raise
            fmap = _pick_feature_map(raw)
        for i in range(int(fmap.shape[0])):
            v = masked_pool_from_tokens(fmap[i], masks_batch[i])
            v = F.normalize(v, dim=0)
            embs.append(v.detach().cpu().numpy().astype(np.float32))

    def _run_with_oom_split(images_batch: List[np.ndarray], masks_batch: List[np.ndarray]) -> None:
        try:
            _run_model_batch(images_batch, masks_batch)
        except RuntimeError as e:
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("cuda out of memory" in msg)
            if (not is_oom) or len(images_batch) <= 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            mid = len(images_batch) // 2
            _run_with_oom_split(images_batch[:mid], masks_batch[:mid])
            _run_with_oom_split(images_batch[mid:], masks_batch[mid:])

    def flush() -> None:
        nonlocal pending_images, pending_masks
        if len(pending_images) == 0:
            return
        _run_with_oom_split(pending_images, pending_masks)
        pending_images = []
        pending_masks = []

    for m in tqdm(metas, desc=f"{backend} masked pool"):
        img = np.array(Image.open(m.crop_path).convert("RGB"), dtype=np.uint8)
        mask = polygon_mask(m.crop_points, w=img.shape[1], h=img.shape[0])
        pending_images.append(img)
        pending_masks.append(mask)
        if len(pending_images) >= max(1, int(batch_size)):
            flush()
    flush()

    if len(embs) == 0:
        raise RuntimeError("No embeddings computed")
    return np.stack(embs, axis=0)


def compute_embeddings_adapter(
    metas: List[ObjMeta],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    feature_key: str,
    adapter_input_size: int,
) -> np.ndarray:
    model, info = load_model(checkpoint=checkpoint, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"adapter_model_info={info}")
    print(f"adapter_feature_key={feature_key}")
    print(f"adapter_input_size={(adapter_input_size if adapter_input_size > 0 else 'tile_size')}")

    embs: List[np.ndarray] = []
    pending_x: List[torch.Tensor] = []
    pending_masks: List[np.ndarray] = []

    def flush() -> None:
        nonlocal pending_x, pending_masks
        if len(pending_x) == 0:
            return
        x = torch.stack(pending_x, dim=0)
        if int(adapter_input_size) > 0 and int(adapter_input_size) != int(x.shape[-1]):
            x = F.interpolate(
                x,
                size=(int(adapter_input_size), int(adapter_input_size)),
                mode="bilinear",
                align_corners=False,
            )
        x = x.to(device)
        with torch.inference_mode():
            pred = model(x, return_features=True)
            if feature_key not in pred:
                raise RuntimeError(
                    f"Feature key '{feature_key}' not returned by model. "
                    f"Available keys: {sorted(list(pred.keys()))}"
                )
            fmap = pred[feature_key]
            if fmap.ndim != 4:
                raise RuntimeError(f"Expected 4D feature map for {feature_key}, got shape={tuple(fmap.shape)}")
        for i in range(int(fmap.shape[0])):
            v = masked_pool_from_tokens(fmap[i], pending_masks[i])
            v = F.normalize(v, dim=0)
            embs.append(v.detach().cpu().numpy().astype(np.float32))
        pending_x = []
        pending_masks = []

    for m in tqdm(metas, desc="adapter masked pool"):
        x = load_crop_tensor(m.crop_path)
        mask = polygon_mask(m.crop_points, w=x.shape[-1], h=x.shape[-2])
        pending_x.append(x)
        pending_masks.append(mask)
        if len(pending_x) >= max(1, int(batch_size)):
            flush()
    flush()

    if len(embs) == 0:
        raise RuntimeError("No embeddings computed")
    return np.stack(embs, axis=0)


def add_to_fiftyone(
    dataset_name: str,
    overwrite_dataset: bool,
    metas: List[ObjMeta],
    embeddings: np.ndarray,
    umap_xy: np.ndarray,
    cluster_ids: Optional[np.ndarray],
    launch: bool,
) -> fo.Dataset:
    if fo.dataset_exists(dataset_name):
        if overwrite_dataset:
            fo.delete_dataset(dataset_name)
        else:
            raise RuntimeError(f"FiftyOne dataset already exists: {dataset_name}. Use --overwrite-dataset.")

    ds = fo.Dataset(dataset_name)
    add_batch_size = 512
    samples: List[fo.Sample] = []
    pbar_build = tqdm(total=len(metas), desc="fiftyone sample build")
    pbar_add = tqdm(total=len(metas), desc="fiftyone add_samples")
    for i, m in enumerate(metas):
        cp = Image.open(m.crop_path)
        cw, ch = cp.size
        cp.close()
        px = [float(p[0]) / float(max(1, cw)) for p in m.crop_points]
        py = [float(p[1]) / float(max(1, ch)) for p in m.crop_points]
        poly_norm = [[list(xy) for xy in zip(px, py)]]

        bx0, by0, bx1, by1 = polygon_bbox(m.crop_points)
        bw, bh = max(1.0, bx1 - bx0), max(1.0, by1 - by0)
        det = fo.Detection(
            label=m.source_label,
            bounding_box=[
                float(max(0.0, bx0) / max(1, cw)),
                float(max(0.0, by0) / max(1, ch)),
                float(min(float(cw), bw) / max(1, cw)),
                float(min(float(ch), bh) / max(1, ch)),
            ],
        )
        pl = fo.Polyline(label=m.source_label, points=poly_norm, closed=True, filled=False)

        s = fo.Sample(filepath=str(m.crop_path))
        s["source_image_path"] = str(m.source_image)
        s["source_json_path"] = str(m.source_json)
        # Keep UI-facing index 1-based, and store explicit 0-based index for robust programmatic lookup.
        s["source_obj_idx"] = int(m.source_obj_idx) + 1
        s["source_obj_idx0"] = int(m.source_obj_idx)
        s["source_obj_num"] = int(m.source_obj_idx) + 1
        s["source_label"] = str(m.source_label)
        s["area_ratio"] = float(m.area_ratio)
        s["object_bbox"] = fo.Detections(detections=[det])
        s["object_poly"] = fo.Polylines(polylines=[pl])
        s["embedding"] = embeddings[i].astype(np.float32).tolist()
        s["umap"] = umap_xy[i].astype(np.float32).tolist()
        if cluster_ids is not None:
            s["cluster"] = int(cluster_ids[i])
        samples.append(s)
        pbar_build.update(1)

        if len(samples) >= add_batch_size:
            ds.add_samples(samples)
            pbar_add.update(len(samples))
            samples = []

    if len(samples) > 0:
        ds.add_samples(samples)
        pbar_add.update(len(samples))
    pbar_build.close()
    pbar_add.close()

    ds.persistent = True
    print(f"fiftyone_dataset={ds.name} samples={len(ds)}")
    return ds


def compute_brain_visualization(
    ds: fo.Dataset,
    embeddings: np.ndarray,
    umap_xy: np.ndarray,
    brain_key: str,
) -> None:
    key = str(brain_key).strip() if str(brain_key).strip() else "obj_umap"
    if key in ds.list_brain_runs():
        ds.delete_brain_run(key)
    # Use precomputed UMAP points so the Embeddings tab is immediately available.
    fob.compute_visualization(
        ds,
        points=umap_xy.astype(np.float32),
        embeddings=embeddings.astype(np.float32),
        brain_key=key,
        method="umap",
        num_dims=2,
    )
    print(f"brain_visualization_key={key}")


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {args.input_dir}")

    adapter_ckpt_raw = str(args.adapter_checkpoint).strip()
    requested_backend = str(args.feature_backend)
    if requested_backend == "auto":
        feature_backend = "adapter" if adapter_ckpt_raw else "dino"
    else:
        feature_backend = requested_backend
    backbone_model_raw = str(args.backbone_model).strip()
    inferred_backend = _infer_backend_from_backbone_model(backbone_model_raw)
    if inferred_backend in {"sam1", "sam2"} and requested_backend == "auto":
        feature_backend = inferred_backend
    if inferred_backend in {"sam1", "sam2"} and feature_backend == "dino":
        raise ValueError(
            f"--feature-backend=dino is incompatible with --backbone-model={backbone_model_raw!r}. "
            f"Use --feature-backend={inferred_backend} (or auto) for SAM model ids."
        )
    if inferred_backend == "dino" and feature_backend in {"sam1", "sam2"}:
        raise ValueError(
            f"--feature-backend={feature_backend} is incompatible with DINO backbone model {backbone_model_raw!r}. "
            "Choose a SAM model id for SAM backends, or switch backend to dino."
        )

    adapter_checkpoint: Optional[Path] = None
    if feature_backend == "adapter":
        if not adapter_ckpt_raw:
            raise ValueError("--adapter-checkpoint is required when --feature-backend=adapter")
        adapter_checkpoint = Path(adapter_ckpt_raw)
        if not adapter_checkpoint.exists() or not adapter_checkpoint.is_file():
            raise FileNotFoundError(f"Missing adapter checkpoint file: {adapter_checkpoint}")
        if int(args.adapter_input_size) > 0 and (int(args.adapter_input_size) % 256) != 0:
            raise ValueError("--adapter-input-size must be multiple of 256")
        if int(args.adapter_input_size) <= 0 and (int(args.tile_size) % 256) != 0:
            raise ValueError(
                "Adapter backend requires model input size multiple of 256. "
                "Set --tile-size to multiple of 256 or use --adapter-input-size."
            )

    if feature_backend == "dino":
        dino_model_name = backbone_model_raw if backbone_model_raw else str(args.dino_model).strip()
    else:
        dino_model_name = str(args.dino_model).strip()
    if feature_backend == "sam1":
        sam_model_name = backbone_model_raw if backbone_model_raw else DEFAULT_SAM1_MODEL
    elif feature_backend == "sam2":
        sam_model_name = backbone_model_raw if backbone_model_raw else DEFAULT_SAM2_MODEL
    else:
        sam_model_name = ""

    device = torch.device(args.device) if str(args.device).strip() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = [x.strip() for x in str(args.label_filter).split(",") if x.strip()]

    print(f"device={device}")
    print(f"input_dir={args.input_dir}")
    print(f"labels={labels if labels else 'ALL'}")
    print(f"tile_size={args.tile_size} tile_context_scale={args.tile_context_scale}")
    print(f"crop_workers={int(args.crop_workers)}")
    print(f"feature_backend={feature_backend}")
    if feature_backend == "adapter":
        print(f"adapter_checkpoint={adapter_checkpoint}")
        print(f"adapter_feature_key={args.adapter_feature_key}")
        print(f"adapter_input_size={(int(args.adapter_input_size) if int(args.adapter_input_size) > 0 else 'tile_size')}")
    elif feature_backend == "dino":
        print(f"dino_model={dino_model_name}")
    elif feature_backend in {"sam1", "sam2"}:
        print(f"sam_model={sam_model_name}")
    print(f"run_kmeans={bool(args.run_kmeans)}")
    if bool(args.run_kmeans):
        print(f"num_clusters={int(args.num_clusters)}")

    metas = build_object_crops(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        label_filter=labels,
        tile_size=int(args.tile_size),
        tile_context_scale=float(args.tile_context_scale),
        min_poly_points=int(args.min_poly_points),
        max_objects=int(args.max_objects),
        crop_workers=int(args.crop_workers),
    )
    if len(metas) == 0:
        raise RuntimeError("No valid objects found from LabelMe pairs")
    print(f"objects={len(metas)}")

    print("stage=compute_embeddings")
    if feature_backend == "adapter":
        emb = compute_embeddings_adapter(
            metas=metas,
            checkpoint=adapter_checkpoint,
            device=device,
            batch_size=int(args.batch_size),
            feature_key=str(args.adapter_feature_key),
            adapter_input_size=int(args.adapter_input_size),
        )
    elif feature_backend == "dino":
        emb = compute_embeddings_dino(
            metas=metas,
            dino_model_name=dino_model_name,
            device=device,
            batch_size=int(args.batch_size),
            trust_repo=bool(args.trust_torch_hub_repo),
        )
    elif feature_backend in {"sam1", "sam2"}:
        emb = compute_embeddings_sam(
            metas=metas,
            backend=feature_backend,
            model_name=sam_model_name,
            device=device,
            batch_size=int(args.batch_size),
            tile_size=int(args.tile_size),
        )
    else:
        raise ValueError(f"Unsupported feature backend: {feature_backend}")

    print("stage=umap_fit_transform")
    reducer = umap.UMAP(
        n_neighbors=max(2, int(args.umap_n_neighbors)),
        min_dist=float(args.umap_min_dist),
        metric=str(args.umap_metric),
        random_state=int(args.seed),
    )
    um = reducer.fit_transform(emb)
    cl: Optional[np.ndarray] = None
    if bool(args.run_kmeans):
        print("stage=kmeans_fit_predict")
        k = max(2, int(args.num_clusters))
        if len(metas) < k:
            k = max(2, min(len(metas), 8))
        km = KMeans(n_clusters=k, random_state=int(args.seed), n_init=10)
        cl = km.fit_predict(um).astype(np.int32)
    else:
        print("stage=kmeans_fit_predict (skipped)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if cl is None:
        np.savez_compressed(
            args.output_dir / "embeddings_umap.npz",
            embedding=emb.astype(np.float32),
            umap=um.astype(np.float32),
        )
    else:
        np.savez_compressed(
            args.output_dir / "embeddings_umap.npz",
            embedding=emb.astype(np.float32),
            umap=um.astype(np.float32),
            cluster=cl.astype(np.int32),
        )

    ds = add_to_fiftyone(
        dataset_name=str(args.dataset_name),
        overwrite_dataset=bool(args.overwrite_dataset),
        metas=metas,
        embeddings=emb,
        umap_xy=um,
        cluster_ids=cl,
        launch=bool(args.launch),
    )
    if bool(args.compute_visualization):
        compute_brain_visualization(
            ds=ds,
            embeddings=emb,
            umap_xy=um,
            brain_key=str(args.brain_key),
        )
    if bool(args.launch):
        session = fo.launch_app(ds)
        print("FiftyOne app launched; press Ctrl+C to close")
        session.wait()
    print(f"saved_npz={args.output_dir / 'embeddings_umap.npz'}")


if __name__ == "__main__":
    main()
