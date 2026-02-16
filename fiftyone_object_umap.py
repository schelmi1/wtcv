#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF

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

FIXED_DINO_MODEL = "dinov2_vits14_reg"


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

    ap.add_argument("--dino-model", type=str, default=FIXED_DINO_MODEL)
    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu; default auto")

    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.05)
    ap.add_argument("--umap-metric", type=str, default="cosine")
    ap.add_argument("--num-clusters", type=int, default=20)
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
    for pair in pairs:
        jf = pair.json_path
        ip = pair.image_path
        d = pair.json_data

        try:
            img = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        h, w = img.shape[:2]
        img_area = float(max(1, w * h))

        shapes = d.get("shapes", []) or []
        for sidx, s in enumerate(shapes):
            lab = str(s.get("label", "")).strip()
            lab_cf = lab.casefold()
            if label_set and (lab_cf not in label_set):
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

            stem = f"{ip.stem}__obj{sidx:04d}"
            crop_path = crops_dir / f"{stem}.jpg"
            Image.fromarray(crop_np).save(crop_path, quality=95)

            metas.append(
                ObjMeta(
                    source_image=ip,
                    source_json=jf,
                    source_label=lab if lab else "unknown",
                    source_label_cf=lab_cf if lab else "unknown",
                    source_obj_idx=int(sidx),
                    image_w=int(w),
                    image_h=int(h),
                    points=pts,
                    bbox_xyxy=(x0, y0, x1, y1),
                    area_ratio=float(max(0.0, polygon_area(pts)) / img_area),
                    crop_path=crop_path,
                    crop_points=cpts,
                )
            )
            if max_objects > 0 and len(metas) >= max_objects:
                return metas
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


def compute_embeddings(
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


def add_to_fiftyone(
    dataset_name: str,
    overwrite_dataset: bool,
    metas: List[ObjMeta],
    embeddings: np.ndarray,
    umap_xy: np.ndarray,
    cluster_ids: np.ndarray,
    launch: bool,
) -> fo.Dataset:
    if fo.dataset_exists(dataset_name):
        if overwrite_dataset:
            fo.delete_dataset(dataset_name)
        else:
            raise RuntimeError(f"FiftyOne dataset already exists: {dataset_name}. Use --overwrite-dataset.")

    ds = fo.Dataset(dataset_name)
    samples: List[fo.Sample] = []
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
        s["cluster"] = int(cluster_ids[i])
        samples.append(s)

    ds.add_samples(samples)
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
    if str(args.dino_model).strip() != FIXED_DINO_MODEL:
        print(f"forcing_dino_model={FIXED_DINO_MODEL} (requested={args.dino_model})")
        args.dino_model = FIXED_DINO_MODEL

    device = torch.device(args.device) if str(args.device).strip() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = [x.strip() for x in str(args.label_filter).split(",") if x.strip()]

    print(f"device={device}")
    print(f"input_dir={args.input_dir}")
    print(f"labels={labels if labels else 'ALL'}")
    print(f"tile_size={args.tile_size} tile_context_scale={args.tile_context_scale}")

    metas = build_object_crops(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        label_filter=labels,
        tile_size=int(args.tile_size),
        tile_context_scale=float(args.tile_context_scale),
        min_poly_points=int(args.min_poly_points),
        max_objects=int(args.max_objects),
    )
    if len(metas) == 0:
        raise RuntimeError("No valid objects found from LabelMe pairs")
    print(f"objects={len(metas)}")

    emb = compute_embeddings(
        metas=metas,
        dino_model_name=str(args.dino_model),
        device=device,
        batch_size=int(args.batch_size),
        trust_repo=bool(args.trust_torch_hub_repo),
    )

    reducer = umap.UMAP(
        n_neighbors=max(2, int(args.umap_n_neighbors)),
        min_dist=float(args.umap_min_dist),
        metric=str(args.umap_metric),
        random_state=int(args.seed),
    )
    um = reducer.fit_transform(emb)
    k = max(2, int(args.num_clusters))
    if len(metas) < k:
        k = max(2, min(len(metas), 8))
    km = KMeans(n_clusters=k, random_state=int(args.seed), n_init=10)
    cl = km.fit_predict(um).astype(np.int32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
