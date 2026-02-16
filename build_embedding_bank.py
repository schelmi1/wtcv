#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF

from wtcv_utils.labelme import polygon_area, polygon_bbox, shape_to_points
from wtcv_utils.records import load_labelme_pairs

FIXED_DINO_MODEL = "dinov2_vits14_reg"


@dataclass
class ObjectTile:
    tile_tensor: torch.Tensor
    mask_u8: np.ndarray
    meta: Dict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an object embedding bank from LabelMe image/json pairs")
    ap.add_argument("--input-dir", type=Path, required=True, help="LabelMe image/json folder")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/embedding_bank"), help="Output directory")
    ap.add_argument(
        "--label-filter",
        type=str,
        default="vehicle",
        help="Comma-separated labels to include, case-insensitive; blank means all",
    )
    ap.add_argument("--max-objects", type=int, default=0, help="0 means all")
    ap.add_argument("--tile-size", type=int, default=448, help="Square object tile size")
    ap.add_argument("--tile-context-scale", type=float, default=2.0, help="Crop side scale around object bbox")
    ap.add_argument("--min-poly-points", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--load-workers", type=int, default=8)

    ap.add_argument("--dino-model", type=str, default=FIXED_DINO_MODEL)
    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu; blank means auto")
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


def load_crop_tensor(img_np: np.ndarray) -> torch.Tensor:
    x = TF.to_tensor(Image.fromarray(img_np))
    norm = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return norm(x)


def masked_pool_from_tokens(
    token_map: torch.Tensor,  # (C,H,W)
    mask_u8: np.ndarray,  # (tile, tile)
) -> torch.Tensor:
    c, h, w = token_map.shape
    m = torch.from_numpy(mask_u8.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m = F.interpolate(m, size=(h, w), mode="area")[0, 0].to(token_map.device).clamp(0.0, 1.0)
    den = m.sum()
    if float(den.item()) < 1e-6:
        return token_map.mean(dim=(1, 2))
    return (token_map * m.unsqueeze(0)).sum(dim=(1, 2)) / den


def iter_object_tiles(
    input_dir: Path,
    label_filter: Sequence[str],
    tile_size: int,
    tile_context_scale: float,
    min_poly_points: int,
    max_objects: int,
    load_workers: int,
) -> Iterable[ObjectTile]:
    label_set = {str(x).strip().casefold() for x in label_filter if str(x).strip()}
    pairs = load_labelme_pairs(
        input_dir,
        load_workers=max(1, int(load_workers)),
        progress_desc="scan labelme",
        progress_leave=True,
    )

    emitted = 0
    for pair in pairs:
        ip = pair.image_path
        jf = pair.json_path
        d = pair.json_data

        try:
            img = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
        except Exception:
            continue

        h, w = img.shape[:2]
        img_area = float(max(1, h * w))
        shapes = d.get("shapes", []) or []
        for sidx, s in enumerate(shapes):
            label = str(s.get("label", "")).strip()
            label_cf = label.casefold()
            if label_set and label_cf not in label_set:
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
            crop_pil = Image.fromarray(crop).resize((tile_size, tile_size), Image.BILINEAR)
            crop_np = np.array(crop_pil, dtype=np.uint8)

            sx = float(tile_size) / float(side)
            sy = float(tile_size) / float(side)
            cpts = [[(float(p[0]) - float(tx0)) * sx, (float(p[1]) - float(ty0)) * sy] for p in pts]
            cpts = [[max(0.0, min(float(tile_size - 1), p[0])), max(0.0, min(float(tile_size - 1), p[1]))] for p in cpts]

            meta = {
                "source_image_path": str(ip),
                "source_json_path": str(jf),
                "source_label": label if label else "unknown",
                "source_label_cf": label_cf if label else "unknown",
                "source_obj_idx0": int(sidx),
                "source_obj_num": int(sidx) + 1,
                "source_obj_idx": int(sidx) + 1,
                "image_w": int(w),
                "image_h": int(h),
                "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                "area_ratio": float(max(0.0, polygon_area(pts)) / img_area),
            }
            yield ObjectTile(
                tile_tensor=load_crop_tensor(crop_np),
                mask_u8=polygon_mask(cpts, w=tile_size, h=tile_size),
                meta=meta,
            )

            emitted += 1
            if max_objects > 0 and emitted >= max_objects:
                return


def _token_grid_shape(num_tokens: int) -> Tuple[int, int]:
    gh = int(round(float(np.sqrt(num_tokens))))
    if gh <= 0:
        raise RuntimeError(f"Invalid patch token count: {num_tokens}")
    if num_tokens % gh != 0:
        raise RuntimeError(f"Patch token count {num_tokens} does not factor into a rectangular grid")
    return gh, int(num_tokens // gh)


def compute_bank(
    tiles: Iterable[ObjectTile],
    dino_model_name: str,
    device: torch.device,
    batch_size: int,
    trust_repo: bool,
) -> Tuple[np.ndarray, List[Dict]]:
    model = torch.hub.load("facebookresearch/dinov2", dino_model_name, trust_repo=trust_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    embeddings: List[np.ndarray] = []
    metadata: List[Dict] = []
    pending_tiles: List[torch.Tensor] = []
    pending_masks: List[np.ndarray] = []
    pending_meta: List[Dict] = []

    def flush() -> None:
        nonlocal pending_tiles, pending_masks, pending_meta
        if not pending_tiles:
            return
        x = torch.stack(pending_tiles, dim=0).to(device)
        with torch.inference_mode():
            feats = model.forward_features(x)
            tok = feats["x_norm_patchtokens"]  # (B,N,C)
            b, n, c = tok.shape
            gh, gw = _token_grid_shape(int(n))
            fmap = tok.reshape(b, gh, gw, c).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        for i in range(fmap.shape[0]):
            v = masked_pool_from_tokens(fmap[i], pending_masks[i])
            v = F.normalize(v, dim=0)
            embeddings.append(v.detach().cpu().numpy().astype(np.float32))
            metadata.append(pending_meta[i])
        pending_tiles = []
        pending_masks = []
        pending_meta = []

    for t in tqdm(tiles, desc="dino masked pool"):
        pending_tiles.append(t.tile_tensor)
        pending_masks.append(t.mask_u8)
        pending_meta.append(t.meta)
        if len(pending_tiles) >= max(1, int(batch_size)):
            flush()
    flush()

    if not embeddings:
        raise RuntimeError("No embeddings computed. Check --input-dir and --label-filter.")
    return np.stack(embeddings, axis=0), metadata


def build_label_prototypes(embeddings: np.ndarray, metadata: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = sorted({str(m.get("source_label", "unknown")) for m in metadata})
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    label_ids = np.array([label_to_id[str(m.get("source_label", "unknown"))] for m in metadata], dtype=np.int32)

    protos: List[np.ndarray] = []
    counts: List[int] = []
    for i, lab in enumerate(labels):
        mask = label_ids == int(i)
        vec = embeddings[mask].mean(axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(vec) + 1e-12)
        protos.append((vec / nrm).astype(np.float32))
        counts.append(int(mask.sum()))

    return np.array(labels, dtype=np.str_), np.stack(protos, axis=0), np.array(counts, dtype=np.int32)


def save_bank_artifacts(
    output_dir: Path,
    embeddings: np.ndarray,
    metadata: List[Dict],
    labels: np.ndarray,
    label_ids: np.ndarray,
    prototypes: np.ndarray,
    prototype_counts: np.ndarray,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    bank_npz = output_dir / "embedding_bank.npz"
    meta_jsonl = output_dir / "embedding_bank_meta.jsonl"
    proto_npz = output_dir / "embedding_prototypes.npz"
    manifest_json = output_dir / "embedding_bank_manifest.json"

    np.savez_compressed(
        bank_npz,
        embeddings=embeddings.astype(np.float32),
        labels=labels,
        label_ids=label_ids.astype(np.int32),
        source_obj_idx0=np.array([int(m["source_obj_idx0"]) for m in metadata], dtype=np.int32),
        source_obj_idx=np.array([int(m["source_obj_idx"]) for m in metadata], dtype=np.int32),
        area_ratio=np.array([float(m["area_ratio"]) for m in metadata], dtype=np.float32),
    )

    with meta_jsonl.open("w", encoding="utf-8") as f:
        for i, m in enumerate(metadata):
            row = dict(m)
            row["embedding_index"] = int(i)
            row["label_id"] = int(label_ids[i])
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    np.savez_compressed(
        proto_npz,
        labels=labels,
        prototypes=prototypes.astype(np.float32),
        counts=prototype_counts.astype(np.int32),
    )

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_labels": int(labels.shape[0]),
        "labels": labels.tolist(),
        "files": {
            "embedding_bank": str(bank_npz.name),
            "metadata_jsonl": str(meta_jsonl.name),
            "prototypes": str(proto_npz.name),
        },
        "config": {
            "label_filter": str(args.label_filter),
            "max_objects": int(args.max_objects),
            "tile_size": int(args.tile_size),
            "tile_context_scale": float(args.tile_context_scale),
            "min_poly_points": int(args.min_poly_points),
            "batch_size": int(args.batch_size),
            "load_workers": int(args.load_workers),
            "dino_model": str(args.dino_model),
            "device": str(args.device),
            "trust_torch_hub_repo": bool(args.trust_torch_hub_repo),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"saved={bank_npz}")
    print(f"saved={meta_jsonl}")
    print(f"saved={proto_npz}")
    print(f"saved={manifest_json}")


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
    print(f"output_dir={args.output_dir}")
    print(f"labels={labels if labels else 'ALL'}")
    print(f"tile_size={args.tile_size} tile_context_scale={args.tile_context_scale}")
    print(f"dino_model={args.dino_model}")

    tiles = iter_object_tiles(
        input_dir=args.input_dir,
        label_filter=labels,
        tile_size=int(args.tile_size),
        tile_context_scale=float(args.tile_context_scale),
        min_poly_points=int(args.min_poly_points),
        max_objects=int(args.max_objects),
        load_workers=int(args.load_workers),
    )

    embeddings, metadata = compute_bank(
        tiles=tiles,
        dino_model_name=str(args.dino_model),
        device=device,
        batch_size=int(args.batch_size),
        trust_repo=bool(args.trust_torch_hub_repo),
    )
    print(f"objects={len(metadata)} embedding_dim={embeddings.shape[1]}")

    labels_arr, prototypes, proto_counts = build_label_prototypes(embeddings, metadata)
    label_to_id = {str(labels_arr[i]): i for i in range(len(labels_arr))}
    label_ids = np.array([label_to_id[str(m.get('source_label', 'unknown'))] for m in metadata], dtype=np.int32)

    save_bank_artifacts(
        output_dir=args.output_dir,
        embeddings=embeddings,
        metadata=metadata,
        labels=labels_arr,
        label_ids=label_ids,
        prototypes=prototypes,
        prototype_counts=proto_counts,
        args=args,
    )


if __name__ == "__main__":
    main()
