from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp")


def find_image_for_json(data_dir: Path, stem: str, image_exts: Sequence[str] = IMG_EXTS) -> Optional[Path]:
    cands: List[Path] = []
    for p in data_dir.glob(f"{stem}.*"):
        if p.is_file() and p.suffix.lower() in image_exts:
            cands.append(p)
    if not cands:
        return None
    pref = {e: i for i, e in enumerate(image_exts)}
    cands.sort(key=lambda p: pref.get(p.suffix.lower(), 999))
    return cands[0]


def shape_to_points(shape: Dict, min_poly_points: int = 3) -> Optional[List[List[float]]]:
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
    if len(pts) < int(min_poly_points):
        return None
    return [[float(p[0]), float(p[1])] for p in pts]


def polygon_bbox(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([float(p[0]) for p in points], dtype=np.float32)
    y = np.array([float(p[1]) for p in points], dtype=np.float32)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def label_casefold(label: str) -> str:
    return str(label).strip().casefold()


def label_matches(label: str, target: str) -> bool:
    return label_casefold(label) == label_casefold(target)

