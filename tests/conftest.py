from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from PIL import Image


def _write_image(path: Path, w: int = 64, h: int = 64, value: int = 128) -> None:
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_labelme_json(
    path: Path,
    image_name: str,
    w: Optional[int],
    h: Optional[int],
    shapes: List[Dict[str, Any]],
) -> None:
    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
    }
    if w is not None:
        payload["imageWidth"] = int(w)
    if h is not None:
        payload["imageHeight"] = int(h)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


@pytest.fixture()
def labelme_pairs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pairs"
    d.mkdir(parents=True, exist_ok=True)

    # Pair A: polygon vehicle + polygon FP
    _write_image(d / "a.png", w=64, h=64, value=100)
    _write_labelme_json(
        d / "a.json",
        image_name="a.png",
        w=64,
        h=64,
        shapes=[
            {
                "label": "Vehicle",
                "points": [[10, 10], [25, 10], [25, 22], [10, 22]],
                "shape_type": "polygon",
                "flags": {},
            },
            {
                "label": "FP",
                "points": [[35, 35], [45, 35], [45, 45], [35, 45]],
                "shape_type": "polygon",
                "flags": {},
            },
        ],
    )

    # Pair B: rectangle vehicle with missing imageWidth/Height -> fallback to image read
    _write_image(d / "b.jpg", w=80, h=48, value=180)
    _write_labelme_json(
        d / "b.json",
        image_name="b.jpg",
        w=None,
        h=None,
        shapes=[
            {
                "label": "vehicle",
                "points": [[5, 6], [22, 28]],
                "shape_type": "rectangle",
                "flags": {},
            }
        ],
    )

    # JSON without image pair: should be ignored by pair loader.
    _write_labelme_json(
        d / "c.json",
        image_name="c.png",
        w=64,
        h=64,
        shapes=[],
    )

    # Invalid JSON: should be ignored.
    (d / "bad.json").write_text("{ bad json")
    return d


@pytest.fixture()
def seg_dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "seg_pairs"
    d.mkdir(parents=True, exist_ok=True)

    # Positive tile with vehicle bbox fully inside.
    _write_image(d / "pos.png", w=224, h=224, value=80)
    _write_labelme_json(
        d / "pos.json",
        image_name="pos.png",
        w=224,
        h=224,
        shapes=[
            {
                "label": "vehicle",
                "shape_type": "rectangle",
                "points": [[80, 90], [120, 130]],
                "flags": {},
            }
        ],
    )

    # FP-only negative.
    _write_image(d / "fp.png", w=224, h=224, value=90)
    _write_labelme_json(
        d / "fp.json",
        image_name="fp.png",
        w=224,
        h=224,
        shapes=[
            {
                "label": "fp",
                "shape_type": "rectangle",
                "points": [[70, 70], [100, 100]],
                "flags": {},
            }
        ],
    )

    # Pure negatives.
    _write_image(d / "neg1.png", w=224, h=224, value=110)
    _write_labelme_json(d / "neg1.json", image_name="neg1.png", w=224, h=224, shapes=[])

    _write_image(d / "neg2.png", w=224, h=224, value=120)
    _write_labelme_json(d / "neg2.json", image_name="neg2.png", w=224, h=224, shapes=[])

    return d

