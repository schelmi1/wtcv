#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm.auto import tqdm

try:
    import fiftyone as fo
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency `fiftyone`. Install with: pip install fiftyone") from e


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export tagged FiftyOne object samples back to merged LabelMe image/json pairs")
    ap.add_argument("--dataset-name", type=str, required=True, help="FiftyOne dataset created by fiftyone_object_umap.py")
    ap.add_argument("--output-dir", type=Path, default=Path("data/umap_filtered_dataset"))
    ap.add_argument("--tag-labels", type=str, default="vehicle,fp", help="Comma-separated allowed tags -> labels")
    ap.add_argument("--overwrite", action="store_true", default=True)
    ap.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    return ap.parse_args()


def shape_to_points(shape: Dict) -> Optional[List[List[float]]]:
    st = str(shape.get("shape_type", "")).strip().lower()
    pts = shape.get("points", []) or []
    if st == "rectangle":
        if len(pts) < 2:
            return None
        x0, y0 = float(pts[0][0]), float(pts[0][1])
        x1, y1 = float(pts[1][0]), float(pts[1][1])
        lx, rx = min(x0, x1), max(x0, x1)
        ty, by = min(y0, y1), max(y0, y1)
        return [[lx, ty], [rx, ty], [rx, by], [lx, by]]
    if len(pts) < 3:
        return None
    return [[float(p[0]), float(p[1])] for p in pts]


def stable_image_name(src_path: Path) -> str:
    # Preserve readability while avoiding collisions from different folders.
    h = hashlib.sha1(str(src_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{src_path.stem}__{h}{src_path.suffix.lower()}"


def choose_label_from_tags(tags: Sequence[str], allowed_in_order: Sequence[str]) -> Optional[str]:
    tag_set = {str(t).strip().casefold() for t in (tags or []) if str(t).strip()}
    for lab in allowed_in_order:
        if lab.casefold() in tag_set:
            return lab
    return None


def sample_get(sample: fo.Sample, field: str, default=None):
    try:
        return sample.get_field(field)
    except Exception:
        return default


def read_points_from_source_json(source_json: Path, source_obj_idx: int) -> Optional[List[List[float]]]:
    try:
        d = json.loads(source_json.read_text())
    except Exception:
        return None
    shapes = d.get("shapes", []) or []
    if source_obj_idx < 0 or source_obj_idx >= len(shapes):
        return None
    return shape_to_points(shapes[source_obj_idx])


def main() -> None:
    args = parse_args()
    labels_order = [x.strip() for x in str(args.tag_labels).split(",") if x.strip()]
    if len(labels_order) == 0:
        raise RuntimeError("--tag-labels must contain at least one label")

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = fo.load_dataset(args.dataset_name)
    print(f"dataset={ds.name} samples={len(ds)}")
    print(f"tag_labels={labels_order}")

    # Group selected objects by original source image.
    grouped: Dict[str, Dict] = {}
    json_cache: Dict[str, Dict] = {}
    selected = 0
    skipped_unlabeled = 0
    skipped_invalid = 0

    for s in tqdm(ds.iter_samples(progress=False), total=len(ds), desc="collect tagged objects"):
        out_label = choose_label_from_tags(getattr(s, "tags", []) or [], labels_order)
        if out_label is None:
            skipped_unlabeled += 1
            continue

        source_image_path = str(sample_get(s, "source_image_path", "") or "")
        source_json_path = str(sample_get(s, "source_json_path", "") or "")
        source_obj_idx = int(sample_get(s, "source_obj_idx", -1) or -1)
        if source_image_path == "" or source_json_path == "" or source_obj_idx < 0:
            skipped_invalid += 1
            continue

        src_img = Path(source_image_path)
        src_json = Path(source_json_path)
        if (not src_img.exists()) or (not src_json.exists()):
            skipped_invalid += 1
            continue

        points = None
        source_points = sample_get(s, "source_points", None)
        if isinstance(source_points, list) and len(source_points) >= 3:
            try:
                points = [[float(p[0]), float(p[1])] for p in source_points]
            except Exception:
                points = None
        if points is None:
            # Fallback: recover geometry from original source json + object index.
            cache_key = str(src_json.resolve())
            if cache_key not in json_cache:
                try:
                    json_cache[cache_key] = json.loads(src_json.read_text())
                except Exception:
                    json_cache[cache_key] = {}
            d = json_cache[cache_key]
            shapes = d.get("shapes", []) or []
            if source_obj_idx < 0 or source_obj_idx >= len(shapes):
                skipped_invalid += 1
                continue
            points = shape_to_points(shapes[source_obj_idx])
        if points is None or len(points) < 3:
            skipped_invalid += 1
            continue

        gkey = str(src_img.resolve())
        if gkey not in grouped:
            grouped[gkey] = {
                "source_image": src_img,
                "source_json": src_json,
                "items": [],
                "seen": set(),
            }
        dedup_key = (int(source_obj_idx), str(out_label))
        if dedup_key in grouped[gkey]["seen"]:
            continue
        grouped[gkey]["seen"].add(dedup_key)
        grouped[gkey]["items"].append(
            {
                "label": out_label,
                "points": points,
                "source_obj_idx": int(source_obj_idx),
            }
        )
        selected += 1

    exported_images = 0
    exported_shapes = 0
    for g in tqdm(grouped.values(), desc="write labelme"):
        src_img: Path = g["source_image"]
        items: List[Dict] = g["items"]
        if len(items) == 0:
            continue

        with Image.open(src_img) as im:
            w, h = im.size

        out_img_name = stable_image_name(src_img)
        out_img = args.output_dir / out_img_name
        out_json = args.output_dir / f"{Path(out_img_name).stem}.json"
        shutil.copy2(src_img, out_img)

        shapes = []
        for it in items:
            shapes.append(
                {
                    "label": str(it["label"]),
                    "points": it["points"],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                }
            )

        d = {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": out_img.name,
            "imageData": None,
            "imageHeight": int(h),
            "imageWidth": int(w),
        }
        out_json.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        exported_images += 1
        exported_shapes += len(shapes)

    print(
        "done",
        f"selected={selected}",
        f"exported_images={exported_images}",
        f"exported_shapes={exported_shapes}",
        f"skipped_unlabeled={skipped_unlabeled}",
        f"skipped_invalid={skipped_invalid}",
        f"output_dir={args.output_dir}",
    )


if __name__ == "__main__":
    main()
