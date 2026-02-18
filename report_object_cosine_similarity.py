#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import shutil
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF

from build_embedding_bank import compute_bank, iter_object_tiles

FIXED_DINO_MODEL = "dinov2_vits14_reg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Report high/low cosine-similarity object pairs from LabelMe records")
    ap.add_argument("--input-dir", type=Path, required=True, help="LabelMe image/json folder")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/embedding_similarity"), help="Output directory")
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
    ap.add_argument("--top-k", type=int, default=25, help="How many high/low pairs to report")
    ap.add_argument(
        "--max-image-similarity",
        type=float,
        default=0.92,
        help="Drop candidate pairs whose source-image cosine similarity is above this threshold",
    )
    ap.add_argument("--image-embed-size", type=int, default=448, help="Square resize used for scene embeddings")
    ap.add_argument("--image-embed-batch-size", type=int, default=12, help="Batch size for scene embedding extraction")
    return ap.parse_args()


def _build_pair_reports(
    embeddings: np.ndarray,
    metadata: List[Dict],
    top_k: int,
    image_embeddings: Dict[str, np.ndarray],
    max_image_similarity: float,
) -> Tuple[List[Dict], List[Dict], Dict]:
    n = int(embeddings.shape[0])
    if n < 2:
        raise RuntimeError("Need at least two objects to compute cosine similarity pairs")

    z = embeddings.astype(np.float32)
    sim = np.matmul(z, z.T)
    np.fill_diagonal(sim, np.nan)

    grouped: Dict[Tuple[str, str, str, str], Dict] = {}
    for i in range(n):
        mi = metadata[i]
        i_img = str(mi.get("source_image_path", ""))
        i_json = str(mi.get("source_json_path", ""))
        i_lab = str(mi.get("source_label", "unknown"))
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if not np.isfinite(s):
                continue
            mj = metadata[j]
            j_img = str(mj.get("source_image_path", ""))
            j_json = str(mj.get("source_json_path", ""))
            j_lab = str(mj.get("source_label", "unknown"))
            # Skip same source sample; ranking should compare different source images.
            if i_img == j_img and i_json == j_json:
                continue

            left = (i_img, i_json, i_lab)
            right = (j_img, j_json, j_lab)
            if left <= right:
                key = (left[0], left[1], right[0], right[1])
                a_lab, b_lab = left[2], right[2]
            else:
                key = (right[0], right[1], left[0], left[1])
                a_lab, b_lab = right[2], left[2]

            g = grouped.get(key)
            if g is None:
                g = {
                    "sum": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                    "a_labels": set(),
                    "b_labels": set(),
                }
                grouped[key] = g
            g["sum"] += s
            g["count"] += 1
            g["min"] = min(float(g["min"]), s)
            g["max"] = max(float(g["max"]), s)
            g["a_labels"].add(a_lab)
            g["b_labels"].add(b_lab)

    if not grouped:
        raise RuntimeError("No valid similarity pairs found")

    rows: List[Dict] = []
    pre_filter_count = 0
    for key, g in grouped.items():
        a_img, a_json, b_img, b_json = key
        cnt = int(g["count"])
        ea = image_embeddings.get(a_img)
        eb = image_embeddings.get(b_img)
        scene_score = float(np.dot(ea, eb)) if (ea is not None and eb is not None) else float("nan")
        pre_filter_count += 1
        if np.isfinite(scene_score) and scene_score > float(max_image_similarity):
            continue
        rows.append(
            {
                "score": float(g["sum"] / max(1, cnt)),  # mean cosine across all object-object pairs for this source pair
                "pair_count": cnt,
                "score_min": float(g["min"]),
                "score_max": float(g["max"]),
                "scene_score": scene_score,
                "a_image": a_img,
                "a_json": a_json,
                "b_image": b_img,
                "b_json": b_json,
                "a_label": ",".join(sorted(str(x) for x in g["a_labels"])),
                "b_label": ",".join(sorted(str(x) for x in g["b_labels"])),
            }
        )
    if not rows:
        raise RuntimeError(
            f"No pairs left after scene filter (max_image_similarity={float(max_image_similarity):.3f}). "
            "Try a higher threshold."
        )

    rows.sort(key=lambda r: float(r["score"]))
    k = max(1, min(int(top_k), len(rows)))
    low = rows[:k]
    high = list(reversed(rows[-k:]))

    all_scores = np.array([float(r["score"]) for r in rows], dtype=np.float32)
    summary = {
        "num_objects": n,
        "num_pairs": int(len(rows)),
        "score_min": float(all_scores.min()),
        "score_mean": float(all_scores.mean()),
        "score_median": float(np.median(all_scores)),
        "score_max": float(all_scores.max()),
        "num_pairs_pre_scene_filter": int(pre_filter_count),
        "num_pairs_post_scene_filter": int(len(rows)),
        "max_image_similarity": float(max_image_similarity),
    }
    return high, low, summary


def _select_unique_source_rows(
    high_rows: Sequence[Dict],
    low_rows: Sequence[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    used_sources: set[str] = set()

    def _select(rows: Sequence[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for row in rows:
            a = str(row.get("a_image", ""))
            b = str(row.get("b_image", ""))
            if not a or not b:
                continue
            if a == b:
                continue
            if a in used_sources or b in used_sources:
                continue
            out.append(row)
            used_sources.add(a)
            used_sources.add(b)
        return out

    # Keep bottom-pair coverage first, then fill remaining budget from top pairs.
    low_sel = _select(low_rows)
    high_sel = _select(high_rows)
    return high_sel, low_sel


def _print_rows(title: str, rows: Sequence[Dict]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for i, r in enumerate(rows, start=1):
        a_img = Path(str(r["a_image"])).name
        b_img = Path(str(r["b_image"])).name
        print(
            f"[{i:02d}] mean_cos={float(r['score']):.4f} n={int(r['pair_count'])} "
            f"(min={float(r['score_min']):.4f}, max={float(r['score_max']):.4f}, scene={float(r['scene_score']):.4f}) | "
            f"A: {a_img} ({r['a_label']}) | "
            f"B: {b_img} ({r['b_label']})"
        )


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as im:
        img = im.convert("RGB").resize((int(image_size), int(image_size)), Image.BILINEAR)
    x = TF.to_tensor(img)
    norm = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return norm(x)


def _compute_image_embeddings(
    image_paths: Sequence[str],
    dino_model_name: str,
    device: torch.device,
    trust_repo: bool,
    image_size: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model = torch.hub.load("facebookresearch/dinov2", dino_model_name, trust_repo=trust_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    out: Dict[str, np.ndarray] = {}
    pending_t: List[torch.Tensor] = []
    pending_k: List[str] = []

    def flush() -> None:
        nonlocal pending_t, pending_k
        if not pending_t:
            return
        x = torch.stack(pending_t, dim=0).to(device)
        with torch.inference_mode():
            feats = model.forward_features(x)
            tok = feats["x_norm_patchtokens"]  # (B,N,C)
            vec = tok.mean(dim=1)
            vec = F.normalize(vec, dim=1)
        arr = vec.detach().cpu().numpy().astype(np.float32)
        for i, k in enumerate(pending_k):
            out[k] = arr[i]
        pending_t = []
        pending_k = []

    for p in image_paths:
        ip = Path(str(p))
        if not ip.exists():
            continue
        try:
            t = _load_image_tensor(ip, image_size=int(image_size))
        except Exception:
            continue
        pending_t.append(t)
        pending_k.append(str(ip))
        if len(pending_t) >= max(1, int(batch_size)):
            flush()
    flush()
    return out


def _copy_labelme_pair_files(
    dst_dir: Path,
    out_stem: str,
    image_path: Path,
    json_path: Path,
) -> Dict:
    out: Dict = {
        "source_image_path": str(image_path),
        "source_json_path": str(json_path),
        "copied_image_path": "",
        "copied_json_path": "",
        "missing_image": False,
        "missing_json": False,
        "labelme_relinked": False,
    }
    if not image_path.exists():
        out["missing_image"] = True
        out["missing_json"] = not json_path.exists()
        return out

    dst_img = dst_dir / f"{out_stem}{image_path.suffix.lower()}"
    shutil.copy2(image_path, dst_img)
    out["copied_image_path"] = str(dst_img)

    if not json_path.exists():
        out["missing_json"] = True
        return out

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw["imagePath"] = dst_img.name
    with dst_img.open("rb") as f:
        raw["imageData"] = base64.b64encode(f.read()).decode("ascii")
    try:
        with Image.open(dst_img) as im:
            w, h = im.size
        raw["imageWidth"] = int(w)
        raw["imageHeight"] = int(h)
    except Exception:
        pass

    dst_json = dst_dir / f"{out_stem}.json"
    dst_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    out["copied_json_path"] = str(dst_json)
    out["labelme_relinked"] = True
    return out


def _save_labelme_pair_exports(
    output_dir: Path,
    high_rows: Sequence[Dict],
    low_rows: Sequence[Dict],
) -> None:
    top_dir = output_dir / "top_pairs"
    bottom_dir = output_dir / "bottom_pairs"
    if top_dir.exists():
        shutil.rmtree(top_dir)
    if bottom_dir.exists():
        shutil.rmtree(bottom_dir)
    top_dir.mkdir(parents=True, exist_ok=True)
    bottom_dir.mkdir(parents=True, exist_ok=True)

    def _write_rows(rows: Sequence[Dict], dst: Path) -> None:
        for rank, row in enumerate(rows, start=1):
            row_stem = f"{rank:03d}_mean_{float(row['score']):+.4f}_n{int(row['pair_count'])}"

            _copy_labelme_pair_files(
                dst_dir=dst,
                out_stem=f"{row_stem}_a",
                image_path=Path(str(row.get("a_image", ""))),
                json_path=Path(str(row.get("a_json", ""))),
            )
            _copy_labelme_pair_files(
                dst_dir=dst,
                out_stem=f"{row_stem}_b",
                image_path=Path(str(row.get("b_image", ""))),
                json_path=Path(str(row.get("b_json", ""))),
            )

    _write_rows(high_rows, top_dir)
    _write_rows(low_rows, bottom_dir)


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

    unique_images = sorted({str(m.get("source_image_path", "")) for m in metadata if str(m.get("source_image_path", ""))})
    image_embeddings = _compute_image_embeddings(
        image_paths=unique_images,
        dino_model_name=str(args.dino_model),
        device=device,
        trust_repo=bool(args.trust_torch_hub_repo),
        image_size=int(args.image_embed_size),
        batch_size=int(args.image_embed_batch_size),
    )
    print(f"scene_embeddings={len(image_embeddings)}")

    high_rows, low_rows, summary = _build_pair_reports(
        embeddings=embeddings,
        metadata=metadata,
        top_k=int(args.top_k),
        image_embeddings=image_embeddings,
        max_image_similarity=float(args.max_image_similarity),
    )
    high_rows, low_rows = _select_unique_source_rows(high_rows=high_rows, low_rows=low_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "cosine_similarity_report.json"
    csv_path = args.output_dir / "cosine_similarity_pairs.csv"
    emb_path = args.output_dir / "cosine_similarity_embeddings.npz"

    payload = {
        "summary": summary,
        "top_k": int(args.top_k),
        "selected_top_pairs": int(len(high_rows)),
        "selected_bottom_pairs": int(len(low_rows)),
        "high_similarity_pairs": high_rows,
        "low_similarity_pairs": low_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "score",
                "pair_count",
                "score_min",
                "score_max",
                "scene_score",
                "a_label",
                "b_label",
                "a_image",
                "a_json",
                "b_image",
                "b_json",
            ],
        )
        writer.writeheader()
        for row in low_rows + high_rows:
            writer.writerow(row)

    np.savez_compressed(
        emb_path,
        embeddings=embeddings.astype(np.float32),
    )
    _save_labelme_pair_exports(
        output_dir=args.output_dir,
        high_rows=high_rows,
        low_rows=low_rows,
    )

    print("\nsummary")
    print("-------")
    print(json.dumps(summary, indent=2))
    _print_rows("Top high-similarity pairs", high_rows)
    _print_rows("Top low-similarity pairs", low_rows)
    print(f"\nsaved={json_path}")
    print(f"saved={csv_path}")
    print(f"saved={emb_path}")
    print(f"saved={args.output_dir / 'top_pairs'}")
    print(f"saved={args.output_dir / 'bottom_pairs'}")


if __name__ == "__main__":
    main()
