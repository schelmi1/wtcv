from __future__ import annotations

from typing import List, Tuple

from PIL import Image


def tile_origins(width: int, height: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    xs = list(range(0, max(1, width - tile + 1), stride))
    ys = list(range(0, max(1, height - tile + 1), stride))

    if len(xs) == 0 or xs[-1] != max(0, width - tile):
        xs.append(max(0, width - tile))
    if len(ys) == 0 or ys[-1] != max(0, height - tile):
        ys.append(max(0, height - tile))

    seen = set()
    out = []
    for y in ys:
        for x in xs:
            if (x, y) not in seen:
                out.append((x, y))
                seen.add((x, y))
    return out


def crop_with_pad(img: Image.Image, x0: int, y0: int, size: int) -> Image.Image:
    w, h = img.size
    x1, y1 = x0 + size, y0 + size

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)

    crop = img.crop((sx0, sy0, sx1, sy1))
    out = Image.new("RGB", (size, size), (0, 0, 0))
    out.paste(crop, (sx0 - x0, sy0 - y0))
    return out

