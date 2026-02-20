from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wtcv_utils.sam_masks import post_process_masks_compat, resolve_postprocess_sizes


class _Sam1StyleImageProcessor:
    def post_process_masks(self, masks, original_sizes, reshaped_input_sizes, **kwargs):
        _ = (masks, kwargs)
        return [("sam1", original_sizes, reshaped_input_sizes)]


class _Sam2StyleImageProcessor:
    def post_process_masks(self, masks, original_sizes, mask_threshold=0.0, **kwargs):
        _ = (masks, mask_threshold, kwargs)
        return [("sam2", original_sizes)]


def test_post_process_masks_compat_sam1_signature() -> None:
    pred = torch.randn(2, 1, 3, 8, 8)
    original = torch.tensor([[512, 512], [512, 512]], dtype=torch.int64)
    reshaped = torch.tensor([[256, 256], [256, 256]], dtype=torch.int64)
    proc = _Sam1StyleImageProcessor()

    out = post_process_masks_compat(proc, pred, original, reshaped)
    assert out[0][0] == "sam1"
    assert torch.equal(out[0][1], original)
    assert torch.equal(out[0][2], reshaped)


def test_post_process_masks_compat_sam2_signature() -> None:
    pred = torch.randn(2, 1, 3, 8, 8)
    original = torch.tensor([[512, 512], [512, 512]], dtype=torch.int64)
    reshaped = torch.tensor([[256, 256], [256, 256]], dtype=torch.int64)
    proc = _Sam2StyleImageProcessor()

    out = post_process_masks_compat(proc, pred, original, reshaped)
    assert out[0][0] == "sam2"
    assert torch.equal(out[0][1], original)


def test_resolve_postprocess_sizes_fallbacks() -> None:
    inputs = {"pixel_values": torch.randn(2, 3, 256, 384)}
    pil_img = Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8))
    np_img = np.zeros((480, 640, 3), dtype=np.uint8)

    original, reshaped = resolve_postprocess_sizes(inputs, [pil_img, np_img])
    assert original.shape == (2, 2)
    assert reshaped.shape == (2, 2)
    assert original.tolist() == [[720, 1280], [480, 640]]
    assert reshaped.tolist() == [[256, 384], [256, 384]]


def test_resolve_postprocess_sizes_uses_existing_keys() -> None:
    original = torch.tensor([[100, 200]], dtype=torch.int64)
    reshaped = torch.tensor([[256, 256]], dtype=torch.int64)
    inputs = {
        "pixel_values": torch.randn(1, 3, 256, 256),
        "original_sizes": original,
        "reshaped_input_sizes": reshaped,
    }
    pil_img = Image.fromarray(np.zeros((50, 60, 3), dtype=np.uint8))

    got_original, got_reshaped = resolve_postprocess_sizes(inputs, [pil_img])
    assert torch.equal(got_original, original.cpu())
    assert torch.equal(got_reshaped, reshaped.cpu())

