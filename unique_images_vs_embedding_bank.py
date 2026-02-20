#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from torchvision.transforms import functional as TF

from build_embedding_bank import compute_bank, compute_bank_with_adapter, iter_object_tiles
from wtcv_utils.labelme import shape_to_points
from wtcv_utils.records import load_labelme_pairs

FIXED_DINO_MODEL = "dinov2_vits14_reg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select unique scenes first, then score object novelty vs embedding bank")
    ap.add_argument("--input-dir", type=Path, required=True, help="LabelMe image/json folder")
    ap.add_argument("--bank-npz", type=Path, required=True, help="Path to embedding_bank.npz")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/unique_vs_bank"), help="Output directory")

    ap.add_argument("--label-filter", type=str, default="vehicle", help="Comma-separated labels to include, blank=all")
    ap.add_argument("--min-poly-points", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=0, help="Optional cap of candidate images (0=all)")
    ap.add_argument("--load-workers", type=int, default=8)

    ap.add_argument("--scene-knn", type=int, default=5, help="Neighbor count for scene-density uniqueness score")
    ap.add_argument("--unique-keep-count", type=int, default=0, help="If >0, keep exactly this many unique images")
    ap.add_argument("--unique-keep-ratio", type=float, default=0.30, help="If keep-count is 0, keep this fraction")
    ap.add_argument("--scene-embed-size", type=int, default=448)
    ap.add_argument("--scene-embed-batch-size", type=int, default=12)

    ap.add_argument("--tile-size", type=int, default=448)
    ap.add_argument("--tile-context-scale", type=float, default=2.0)
    ap.add_argument("--max-objects", type=int, default=0, help="Optional cap on object extraction after scene selection")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--bank-topk", type=int, default=5, help="Top-k bank matches used for mean score")
    ap.add_argument(
        "--feature-backend",
        type=str,
        choices=["dino", "adapter"],
        default="dino",
        help="Object embedding backend used for selected-scene object-vs-bank scoring.",
    )
    ap.add_argument(
        "--adapter-checkpoint",
        type=str,
        default="",
        help="Checkpoint path required when --feature-backend=adapter.",
    )
    ap.add_argument(
        "--adapter-feature-key",
        type=str,
        choices=["feat_adapted", "feat_dino"],
        default="feat_adapted",
        help="Stage1 feature map to masked-pool in adapter backend.",
    )
    ap.add_argument(
        "--adapter-input-size",
        type=int,
        default=0,
        help="Optional square resize for adapter backend before forward (0 keeps tile_size). Must be multiple of 256.",
    )

    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu; blank means auto")
    return ap.parse_args()


def _discover_candidate_images(
    input_dir: Path,
    label_filter: Sequence[str],
    min_poly_points: int,
    max_images: int,
    load_workers: int,
) -> List[Dict]:
    label_set = {str(x).strip().casefold() for x in label_filter if str(x).strip()}
    out: List[Dict] = []
    pairs = load_labelme_pairs(
        input_dir,
        load_workers=max(1, int(load_workers)),
        progress_desc="discover candidate images",
        progress_leave=True,
    )
    for pair in pairs:
        d = pair.json_data
        obj_count = 0
        for s in d.get("shapes", []) or []:
            lab = str(s.get("label", "")).strip().casefold()
            if label_set and lab not in label_set:
                continue
            pts = shape_to_points(s, min_poly_points=int(min_poly_points))
            if pts is None:
                continue
            obj_count += 1
        if obj_count <= 0:
            continue
        out.append(
            {
                "image_path": str(pair.image_path),
                "json_path": str(pair.json_path),
                "object_count": int(obj_count),
            }
        )
    out.sort(key=lambda r: (Path(str(r["image_path"])).name, str(r["image_path"])))
    if max_images > 0:
        out = out[: int(max_images)]
    return out


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as im:
        img = im.convert("RGB").resize((int(image_size), int(image_size)), Image.BILINEAR)
    x = TF.to_tensor(img)
    norm = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return norm(x)


def _compute_scene_embeddings(
    image_paths: Sequence[str],
    device: torch.device,
    trust_repo: bool,
    image_size: int,
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    model = torch.hub.load("facebookresearch/dinov2", FIXED_DINO_MODEL, trust_repo=trust_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    vecs: List[np.ndarray] = []
    keys: List[str] = []
    pending_x: List[torch.Tensor] = []
    pending_k: List[str] = []

    def flush() -> None:
        nonlocal pending_x, pending_k
        if not pending_x:
            return
        x = torch.stack(pending_x, dim=0).to(device)
        with torch.inference_mode():
            feats = model.forward_features(x)
            tok = feats["x_norm_patchtokens"]  # [B,N,C]
            v = tok.mean(dim=1)
            v = F.normalize(v, dim=1)
        arr = v.detach().cpu().numpy().astype(np.float32)
        for i, k in enumerate(pending_k):
            vecs.append(arr[i])
            keys.append(k)
        pending_x = []
        pending_k = []

    for p in tqdm(image_paths, desc="scene dino embeds", unit="img"):
        ip = Path(str(p))
        if not ip.exists():
            continue
        try:
            t = _load_image_tensor(ip, image_size=int(image_size))
        except Exception:
            continue
        pending_x.append(t)
        pending_k.append(str(ip))
        if len(pending_x) >= max(1, int(batch_size)):
            flush()
    flush()
    if not vecs:
        raise RuntimeError("No scene embeddings were computed")
    return np.stack(vecs, axis=0), keys


def _rank_unique_images(scene_emb: np.ndarray, keys: List[str], scene_knn: int) -> Tuple[List[Dict], np.ndarray]:
    z = scene_emb.astype(np.float32)
    sim = np.matmul(z, z.T)
    n = int(sim.shape[0])
    rows: List[Dict] = []
    for i in range(n):
        vals = sim[i].copy()
        vals[i] = -np.inf
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            density = 0.0
        else:
            k = max(1, min(int(scene_knn), int(finite.size)))
            topk = np.partition(finite, -k)[-k:]
            density = float(topk.mean())
        uniqueness = float(1.0 - density)
        rows.append(
            {
                "image_path": str(keys[i]),
                "scene_density": float(density),
                "scene_uniqueness": float(uniqueness),
            }
        )
    rows.sort(key=lambda r: float(r["scene_uniqueness"]), reverse=True)
    for idx, r in enumerate(rows, start=1):
        r["rank"] = int(idx)
    return rows, sim


def _select_unique(rows: Sequence[Dict], keep_count: int, keep_ratio: float) -> List[Dict]:
    n = len(rows)
    if n <= 0:
        return []
    if keep_count > 0:
        k = min(n, int(keep_count))
    else:
        k = max(1, min(n, int(math.ceil(float(keep_ratio) * n))))
    return list(rows[:k])


def _copy_relinked_labelme_pair(dst_dir: Path, rank: int, image_path: Path, json_path: Path) -> None:
    if not image_path.exists() or not json_path.exists():
        return
    stem = f"{int(rank):04d}_{image_path.stem}"
    dst_img = dst_dir / f"{stem}{image_path.suffix.lower()}"
    shutil.copy2(image_path, dst_img)
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw["imagePath"] = dst_img.name
    # Keep LabelMe JSON lightweight: do not inline base64 imageData.
    raw["imageData"] = None
    try:
        with Image.open(dst_img) as im:
            w, h = im.size
        raw["imageWidth"] = int(w)
        raw["imageHeight"] = int(h)
    except Exception:
        pass
    (dst_dir / f"{stem}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_selected_object_tiles(
    input_dir: Path,
    selected_images: set[str],
    label_filter: Sequence[str],
    tile_size: int,
    tile_context_scale: float,
    min_poly_points: int,
    max_objects: int,
    load_workers: int,
) -> Iterable:
    for t in iter_object_tiles(
        input_dir=input_dir,
        label_filter=label_filter,
        tile_size=int(tile_size),
        tile_context_scale=float(tile_context_scale),
        min_poly_points=int(min_poly_points),
        max_objects=int(max_objects),
        load_workers=int(load_workers),
    ):
        ip = str(t.meta.get("source_image_path", ""))
        if ip in selected_images:
            yield t


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {args.input_dir}")
    if not args.bank_npz.exists():
        raise FileNotFoundError(f"Missing bank npz: {args.bank_npz}")

    device = torch.device(args.device) if str(args.device).strip() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = [x.strip() for x in str(args.label_filter).split(",") if x.strip()]
    print(f"device={device}")
    print(f"dino_model={FIXED_DINO_MODEL}")
    print(f"input_dir={args.input_dir}")
    print(f"bank_npz={args.bank_npz}")
    print(f"labels={labels if labels else 'ALL'}")
    print(f"feature_backend={args.feature_backend}")
    if str(args.feature_backend) == "adapter":
        adapter_ckpt_raw = str(args.adapter_checkpoint).strip()
        if not adapter_ckpt_raw:
            raise ValueError("--adapter-checkpoint is required when --feature-backend=adapter")
        args.adapter_checkpoint = Path(adapter_ckpt_raw)
        if not args.adapter_checkpoint.exists() or not args.adapter_checkpoint.is_file():
            raise FileNotFoundError(f"Missing adapter checkpoint file: {args.adapter_checkpoint}")
        if int(args.adapter_input_size) > 0 and (int(args.adapter_input_size) % 256) != 0:
            raise ValueError("--adapter-input-size must be multiple of 256")
        if int(args.adapter_input_size) <= 0 and (int(args.tile_size) % 256) != 0:
            raise ValueError(
                "Adapter backend requires model input size multiple of 256. "
                "Set --tile-size to multiple of 256 or use --adapter-input-size."
            )
        print(f"adapter_checkpoint={args.adapter_checkpoint}")
        print(f"adapter_feature_key={args.adapter_feature_key}")
        print(f"adapter_input_size={(int(args.adapter_input_size) if int(args.adapter_input_size) > 0 else 'tile_size')}")

    candidates = _discover_candidate_images(
        input_dir=args.input_dir,
        label_filter=labels,
        min_poly_points=int(args.min_poly_points),
        max_images=int(args.max_images),
        load_workers=int(args.load_workers),
    )
    if not candidates:
        raise RuntimeError("No candidate images with valid target objects")
    image_paths = [str(r["image_path"]) for r in candidates]
    print(f"candidate_images={len(image_paths)}")

    scene_emb, scene_keys = _compute_scene_embeddings(
        image_paths=image_paths,
        device=device,
        trust_repo=bool(args.trust_torch_hub_repo),
        image_size=int(args.scene_embed_size),
        batch_size=int(args.scene_embed_batch_size),
    )
    ranked_rows, sim_mx = _rank_unique_images(scene_emb=scene_emb, keys=scene_keys, scene_knn=int(args.scene_knn))
    selected_rows = _select_unique(
        rows=ranked_rows,
        keep_count=int(args.unique_keep_count),
        keep_ratio=float(args.unique_keep_ratio),
    )
    selected_images = {str(r["image_path"]) for r in selected_rows}
    print(f"selected_unique_images={len(selected_images)}")

    tiles = _iter_selected_object_tiles(
        input_dir=args.input_dir,
        selected_images=selected_images,
        label_filter=labels,
        tile_size=int(args.tile_size),
        tile_context_scale=float(args.tile_context_scale),
        min_poly_points=int(args.min_poly_points),
        max_objects=int(args.max_objects),
        load_workers=int(args.load_workers),
    )
    if str(args.feature_backend) == "adapter":
        obj_emb, obj_meta = compute_bank_with_adapter(
            tiles=tiles,
            checkpoint=args.adapter_checkpoint,
            device=device,
            batch_size=int(args.batch_size),
            feature_key=str(args.adapter_feature_key),
            adapter_input_size=int(args.adapter_input_size),
            total_objects=(int(args.max_objects) if int(args.max_objects) > 0 else None),
        )
    else:
        obj_emb, obj_meta = compute_bank(
            tiles=tiles,
            dino_model_name=FIXED_DINO_MODEL,
            device=device,
            batch_size=int(args.batch_size),
            trust_repo=bool(args.trust_torch_hub_repo),
            total_objects=(int(args.max_objects) if int(args.max_objects) > 0 else None),
        )
    print(f"selected_objects={len(obj_meta)}")

    bank_blob = np.load(args.bank_npz)
    if "embeddings" not in bank_blob:
        raise RuntimeError(f"Bank file has no `embeddings` array: {args.bank_npz}")
    bank_emb = np.array(bank_blob["embeddings"], dtype=np.float32)
    if bank_emb.ndim != 2 or bank_emb.shape[0] <= 0:
        raise RuntimeError("Invalid bank embeddings shape")
    if obj_emb.shape[1] != bank_emb.shape[1]:
        raise RuntimeError(
            f"Embedding dim mismatch: objects={obj_emb.shape[1]} bank={bank_emb.shape[1]} "
            f"(bank={args.bank_npz})"
        )

    obj_emb = obj_emb.astype(np.float32)
    bank_emb = bank_emb.astype(np.float32)
    obj_emb /= np.clip(np.linalg.norm(obj_emb, axis=1, keepdims=True), 1e-12, None)
    bank_emb /= np.clip(np.linalg.norm(bank_emb, axis=1, keepdims=True), 1e-12, None)

    sim = np.matmul(obj_emb, bank_emb.T)  # [M,B]
    max_bank = sim.max(axis=1)
    k = max(1, min(int(args.bank_topk), int(bank_emb.shape[0])))
    topk_mean = np.partition(sim, -k, axis=1)[:, -k:].mean(axis=1)

    object_rows: List[Dict] = []
    for i, m in enumerate(obj_meta):
        max_cos = float(max_bank[i])
        object_rows.append(
            {
                "source_image_path": str(m.get("source_image_path", "")),
                "source_json_path": str(m.get("source_json_path", "")),
                "source_label": str(m.get("source_label", "unknown")),
                "source_obj_idx": int(m.get("source_obj_idx", -1)),
                "area_ratio": float(m.get("area_ratio", 0.0)),
                "max_bank_cos": max_cos,
                "topk_mean_bank_cos": float(topk_mean[i]),
                "novelty_score": float(1.0 - max_cos),
            }
        )
    object_rows.sort(key=lambda r: float(r["max_bank_cos"]))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sel_dir = out / "selected_unique_labelme"
    if sel_dir.exists():
        shutil.rmtree(sel_dir)
    sel_dir.mkdir(parents=True, exist_ok=True)

    # Write scene ranking and selected list.
    scene_csv = out / "scene_uniqueness.csv"
    with scene_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["rank", "image_path", "scene_uniqueness", "scene_density"])
        wr.writeheader()
        for r in ranked_rows:
            wr.writerow(r)

    selected_txt = out / "selected_unique_images.txt"
    selected_txt.write_text("\n".join(sorted(selected_images)) + ("\n" if selected_images else ""), encoding="utf-8")

    # Copy selected unique image/json pairs as valid LabelMe pairs.
    rank_map = {str(r["image_path"]): int(r["rank"]) for r in ranked_rows}
    json_map = {str(r["image_path"]): str(r["json_path"]) for r in candidates}
    for ip in sorted(selected_images):
        jp = json_map.get(ip, "")
        if not jp:
            continue
        _copy_relinked_labelme_pair(sel_dir, rank=rank_map.get(ip, 999999), image_path=Path(ip), json_path=Path(jp))

    # Save scene embeddings/similarity for audit.
    np.savez_compressed(out / "scene_embeddings.npz", image_paths=np.array(scene_keys, dtype=np.str_), embeddings=scene_emb)
    np.savez_compressed(out / "scene_similarity.npz", image_paths=np.array(scene_keys, dtype=np.str_), similarity=sim_mx.astype(np.float32))

    # Save object-vs-bank report.
    obj_csv = out / "objects_vs_bank.csv"
    with obj_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "source_image_path",
                "source_json_path",
                "source_label",
                "source_obj_idx",
                "area_ratio",
                "max_bank_cos",
                "topk_mean_bank_cos",
                "novelty_score",
            ],
        )
        wr.writeheader()
        for r in object_rows:
            wr.writerow(r)

    top_novel_json = out / "top_novel_objects.json"
    top_novel_json.write_text(json.dumps(object_rows[: min(500, len(object_rows))], indent=2), encoding="utf-8")

    summary = {
        "input_dir": str(args.input_dir),
        "bank_npz": str(args.bank_npz),
        "dino_model": FIXED_DINO_MODEL,
        "feature_backend": str(args.feature_backend),
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "adapter_feature_key": str(args.adapter_feature_key),
        "adapter_input_size": int(args.adapter_input_size),
        "candidate_images": int(len(image_paths)),
        "selected_unique_images": int(len(selected_images)),
        "selected_objects": int(len(object_rows)),
        "scene_knn": int(args.scene_knn),
        "unique_keep_count": int(args.unique_keep_count),
        "unique_keep_ratio": float(args.unique_keep_ratio),
        "scene_embed_size": int(args.scene_embed_size),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nsummary")
    print("-------")
    print(json.dumps(summary, indent=2))
    print(f"saved={scene_csv}")
    print(f"saved={selected_txt}")
    print(f"saved={sel_dir}")
    print(f"saved={obj_csv}")
    print(f"saved={top_novel_json}")


if __name__ == "__main__":
    main()
