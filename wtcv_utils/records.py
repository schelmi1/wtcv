from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from wtcv_utils.labelme import IMG_EXTS, polygon_area, polygon_bbox, shape_to_points


@dataclass
class LabelmePair:
    image_path: Path
    json_path: Path
    json_data: Dict


def _build_image_index(data_dir: Path, image_exts: Sequence[str]) -> Dict[str, Path]:
    pref = {str(ext).lower(): i for i, ext in enumerate(image_exts)}
    idx: Dict[str, Tuple[int, Path]] = {}
    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        sfx = p.suffix.lower()
        if sfx not in pref:
            continue
        rank = pref[sfx]
        prev = idx.get(p.stem)
        if prev is None or rank < prev[0]:
            idx[p.stem] = (rank, p)
    return {k: v[1] for k, v in idx.items()}


def _load_pair_task(task: Tuple[str, str]) -> Optional[Tuple[str, str, Dict]]:
    json_path_str, image_path_str = task
    try:
        d = json.loads(Path(json_path_str).read_text())
    except Exception:
        return None
    return (json_path_str, image_path_str, d)


def load_labelme_pairs(
    data_dir: Path,
    load_workers: int = 8,
    max_images: int = 0,
    random_sample: bool = False,
    sample_seed: int = 42,
    progress_desc: str = "load_records",
    progress_leave: bool = True,
) -> List[LabelmePair]:
    image_index = _build_image_index(data_dir, IMG_EXTS)
    json_files = sorted(data_dir.glob("*.json"))
    if max_images > 0:
        n = int(max_images)
        if random_sample and n < len(json_files):
            rr = random.Random(int(sample_seed))
            json_files = rr.sample(json_files, n)
            json_files.sort()
        else:
            json_files = json_files[:n]

    tasks: List[Tuple[str, str]] = []
    for jf in json_files:
        ip = image_index.get(jf.stem)
        if ip is None:
            continue
        tasks.append((str(jf), str(ip)))

    out: List[LabelmePair] = []
    workers = max(1, int(load_workers))

    if workers == 1:
        for t in tqdm(tasks, desc=progress_desc, leave=progress_leave):
            item = _load_pair_task(t)
            if item is None:
                continue
            js, is_, d = item
            out.append(LabelmePair(image_path=Path(is_), json_path=Path(js), json_data=d))
        return out

    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            it = ex.map(_load_pair_task, tasks, chunksize=32)
            for item in tqdm(it, total=len(tasks), desc=progress_desc, leave=progress_leave):
                if item is None:
                    continue
                js, is_, d = item
                out.append(LabelmePair(image_path=Path(is_), json_path=Path(js), json_data=d))
    except Exception:
        # Robust fallback in constrained environments.
        out = []
        for t in tqdm(tasks, desc=progress_desc, leave=progress_leave):
            item = _load_pair_task(t)
            if item is None:
                continue
            js, is_, d = item
            out.append(LabelmePair(image_path=Path(is_), json_path=Path(js), json_data=d))

    out.sort(key=lambda p: p.json_path.name)
    return out


def polygon_center(points: List[List[float]]) -> List[float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [float(np.mean(xs)), float(np.mean(ys))]


def load_labelme_records(
    data_dir: Path,
    label_name: str,
    min_poly_points: int,
    include_fp: bool = False,
    fp_label: str = "fp",
    load_workers: int = 8,
) -> List[Dict]:
    """
    Load LabelMe image/json pairs into training/eval record dicts.

    Output schema intentionally matches the legacy `train_stage1_seg.load_records`
    format so downstream code can stay unchanged during refactors.
    """
    wanted_label = str(label_name).strip().casefold()
    fp_label_cf = str(fp_label).strip().casefold()

    records: List[Dict] = []
    for pair in load_labelme_pairs(data_dir, load_workers=load_workers, progress_desc="load_records"):
        jf = pair.json_path
        img_path = pair.image_path
        d = pair.json_data

        w = int(d.get("imageWidth") or 0)
        h = int(d.get("imageHeight") or 0)
        if w <= 0 or h <= 0:
            try:
                with Image.open(img_path) as im:
                    w, h = map(int, im.size)
            except Exception:
                continue
        if w <= 0 or h <= 0:
            continue

        shapes = d.get("shapes", []) or []
        objects = []
        for s in shapes:
            got_label = str(s.get("label", "")).strip().casefold()
            is_pos = got_label == wanted_label
            is_fp = include_fp and (got_label == fp_label_cf) and (got_label != wanted_label)
            if (not is_pos) and (not is_fp):
                continue

            stype = str(s.get("shape_type", "")).strip().lower()
            pts = shape_to_points(s, min_poly_points=min_poly_points)
            if pts is None:
                continue

            x0, y0, x1, y1 = polygon_bbox(pts)
            cx, cy = polygon_center(pts)
            if stype == "rectangle":
                pa = float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
            else:
                pa = float(polygon_area(pts))

            objects.append(
                {
                    "label_cf": got_label,
                    "is_fp": bool(is_fp),
                    "points": pts,
                    "shape_type": stype if stype else "polygon",
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "center_xy": [cx, cy],
                    "poly_area": pa,
                }
            )

        records.append(
            {
                "image_path": str(img_path),
                "json_path": str(jf),
                "width": w,
                "height": h,
                "objects": objects,
            }
        )
    return records


def discover_labels(data_dir: Path) -> List[str]:
    labels = set()
    for pair in load_labelme_pairs(data_dir, load_workers=8, progress_desc="discover_labels"):
        d = pair.json_data
        for s in d.get("shapes", []) or []:
            lab = s.get("label")
            if isinstance(lab, str) and lab.strip():
                labels.add(lab.strip())
    return sorted(labels)
