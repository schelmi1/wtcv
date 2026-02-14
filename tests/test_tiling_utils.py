from __future__ import annotations

import numpy as np
from PIL import Image

from wtcv_utils.tiling import crop_with_pad, tile_origins


def test_tile_origins_covers_right_bottom_edges() -> None:
    # width/height not divisible by stride; final origin must include right/bottom edge anchor.
    origins = tile_origins(width=500, height=380, tile=224, stride=112)
    assert (500 - 224, 380 - 224) in origins
    assert (0, 0) in origins


def test_crop_with_pad_keeps_size_and_pads() -> None:
    arr = np.full((10, 10, 3), 255, dtype=np.uint8)
    img = Image.fromarray(arr)
    crop = crop_with_pad(img, x0=-5, y0=-5, size=20)
    out = np.array(crop)
    assert out.shape == (20, 20, 3)
    # top-left is padded black
    assert int(out[0, 0, 0]) == 0
    # source content appears in the pasted region
    assert int(out[8, 8, 0]) == 255

