#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm.auto import tqdm

try:
    import fiftyone as fo
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency `fiftyone`. Install with: pip install fiftyone") from e

from wtcv_utils.labelme import shape_to_points


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export tagged FiftyOne object samples back to merged LabelMe image/json pairs")
    ap.add_argument("--dataset-name", type=str, required=True, help="FiftyOne dataset created by fiftyone_object_umap.py")
    ap.add_argument("--output-dir", type=Path, default=Path("data/umap_filtered_dataset"))
    ap.add_argument("--tag-labels", type=str, default="vehicle,fp", help="Comma-separated allowed tags -> labels")
    ap.add_argument(
        "--debug-missing-limit",
        type=int,
        default=8,
        help="How many invalid-sample examples to print when required fields are missing",
    )
    ap.add_argument("--overwrite", action="store_true", default=True)
    ap.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    return ap.parse_args()


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


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    if isinstance(v, (list, tuple, dict, set)):
        return len(v) == 0
    return False


def sample_get_any(sample: fo.Sample, fields: Sequence[str], default=None) -> Tuple[Any, Optional[str]]:
    for f in fields:
        v = sample_get(sample, f, None)
        if not is_missing(v):
            return v, f
    return default, None


def coerce_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return int(default)


def points_from_any(v: Any) -> Optional[List[List[float]]]:
    if not isinstance(v, list):
        return None
    if len(v) >= 3 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in v):
        try:
            return [[float(p[0]), float(p[1])] for p in v]
        except Exception:
            return None
    # Some sources store polygon points as a wrapped list: [[[x,y], ...]]
    if len(v) == 1 and isinstance(v[0], list):
        inner = v[0]
        if len(inner) >= 3 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in inner):
            try:
                return [[float(p[0]), float(p[1])] for p in inner]
            except Exception:
                return None
    return None


def read_points_from_source_json(source_json: Path, source_obj_idx: int) -> Optional[List[List[float]]]:
    try:
        d = json.loads(source_json.read_text())
    except Exception:
        return None
    shapes = d.get("shapes", []) or []
    if source_obj_idx < 0 or source_obj_idx >= len(shapes):
        return None
    return shape_to_points(shapes[source_obj_idx], min_poly_points=3)


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
    field_hits: Dict[str, Counter] = {
        "source_image_path": Counter(),
        "source_json_path": Counter(),
        "source_obj_idx0": Counter(),
        "source_points": Counter(),
    }
    missing_reasons: Counter = Counter()
    invalid_examples: List[str] = []
    selected = 0
    skipped_unlabeled = 0
    skipped_invalid = 0

    source_image_aliases = (
        "source_image_path",
        "source.image_path",
        "source.filepath",
        "source.path",
        "source_image",
        "source",
    )
    source_json_aliases = (
        "source_json_path",
        "source.json_path",
        "source.labelme_json_path",
        "source_json",
    )
    source_obj_idx0_aliases = (
        "source_obj_idx0",
        "source.obj_idx0",
        "source.object_idx0",
    )
    source_obj_idx_aliases = (
        "source_obj_idx",
        "source.obj_idx",
        "source.object_idx",
        "object.idx",
        "object.index",
        "object_id",
    )
    source_obj_num_aliases = (
        "source_obj_num",
        "source.obj_num",
        "source.object_num",
    )
    source_points_aliases = (
        "source_points",
        "source.points",
        "object.points",
    )

    for s in tqdm(ds.iter_samples(progress=False), total=len(ds), desc="collect tagged objects"):
        out_label = choose_label_from_tags(getattr(s, "tags", []) or [], labels_order)
        if out_label is None:
            skipped_unlabeled += 1
            continue

        source_image_raw, source_image_field = sample_get_any(s, source_image_aliases, default="")
        source_json_raw, source_json_field = sample_get_any(s, source_json_aliases, default="")
        source_obj_idx0_raw, source_obj_idx0_field = sample_get_any(s, source_obj_idx0_aliases, default=None)
        source_obj_idx_raw, source_obj_idx_field = sample_get_any(s, source_obj_idx_aliases, default=None)
        source_obj_num_raw, source_obj_num_field = sample_get_any(s, source_obj_num_aliases, default=None)

        source_image_path = str(source_image_raw or "")
        source_json_path = str(source_json_raw or "")
        source_obj_idx = -1
        source_obj_idx_field_used = None
        if source_obj_idx0_field is not None:
            source_obj_idx = coerce_int(source_obj_idx0_raw, default=-1)
            source_obj_idx_field_used = source_obj_idx0_field
        elif source_obj_idx_field is not None:
            # Backward compatibility: legacy datasets use 0-based source_obj_idx.
            source_obj_idx = coerce_int(source_obj_idx_raw, default=-1)
            source_obj_idx_field_used = source_obj_idx_field
        elif source_obj_num_field is not None:
            # 1-based human-readable object number.
            source_obj_idx = coerce_int(source_obj_num_raw, default=0) - 1
            source_obj_idx_field_used = source_obj_num_field

        if source_image_field is not None:
            field_hits["source_image_path"][source_image_field] += 1
        if source_json_field is not None:
            field_hits["source_json_path"][source_json_field] += 1
        if source_obj_idx_field_used is not None:
            field_hits["source_obj_idx0"][source_obj_idx_field_used] += 1

        miss = []
        if source_image_path == "":
            miss.append("source_image_path")
        if source_json_path == "":
            miss.append("source_json_path")
        if source_obj_idx < 0:
            miss.append("source_obj_idx")
        if len(miss) > 0:
            for m in miss:
                missing_reasons[m] += 1
            skipped_invalid += 1
            if len(invalid_examples) < max(0, int(args.debug_missing_limit)):
                invalid_examples.append(f"id={s.id} missing={','.join(miss)}")
            continue

        src_img = Path(source_image_path)
        src_json = Path(source_json_path)
        if not src_img.exists():
            missing_reasons["source_image_missing_on_disk"] += 1
            skipped_invalid += 1
            if len(invalid_examples) < max(0, int(args.debug_missing_limit)):
                invalid_examples.append(f"id={s.id} missing_on_disk=source_image_path path={src_img}")
            continue
        if not src_json.exists():
            missing_reasons["source_json_missing_on_disk"] += 1
            skipped_invalid += 1
            if len(invalid_examples) < max(0, int(args.debug_missing_limit)):
                invalid_examples.append(f"id={s.id} missing_on_disk=source_json_path path={src_json}")
            continue

        points = None
        source_points_raw, source_points_field = sample_get_any(s, source_points_aliases, default=None)
        if source_points_field is not None:
            field_hits["source_points"][source_points_field] += 1
        points = points_from_any(source_points_raw)
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
                missing_reasons["source_obj_idx_out_of_range"] += 1
                skipped_invalid += 1
                if len(invalid_examples) < max(0, int(args.debug_missing_limit)):
                    invalid_examples.append(
                        f"id={s.id} source_obj_idx_out_of_range idx={source_obj_idx} shapes={len(shapes)} json={src_json}"
                    )
                continue
            points = shape_to_points(shapes[source_obj_idx], min_poly_points=3)
        if points is None or len(points) < 3:
            missing_reasons["invalid_polygon_points"] += 1
            skipped_invalid += 1
            if len(invalid_examples) < max(0, int(args.debug_missing_limit)):
                invalid_examples.append(f"id={s.id} invalid_polygon_points idx={source_obj_idx} json={src_json}")
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
                "source_obj_num": int(source_obj_idx) + 1,
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
    for canonical, ctr in field_hits.items():
        if len(ctr) == 0:
            continue
        parts = [f"{k}:{v}" for k, v in ctr.most_common()]
        print(f"field_resolution {canonical} -> {', '.join(parts)}")
    if len(missing_reasons) > 0:
        parts = [f"{k}:{v}" for k, v in missing_reasons.most_common()]
        print(f"skip_reasons {', '.join(parts)}")
    if len(invalid_examples) > 0:
        print("invalid_examples")
        for ex in invalid_examples:
            print(ex)


if __name__ == "__main__":
    main()
