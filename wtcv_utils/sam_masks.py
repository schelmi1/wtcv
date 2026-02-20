from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def resolve_postprocess_sizes(
    inputs: Dict[str, torch.Tensor],
    batch_images: List[Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build robust original/reshaped size tensors for SAM post-processing.
    Handles processor variants that may omit one or both keys.
    """
    original_sizes = inputs.get("original_sizes")
    reshaped_sizes = inputs.get("reshaped_input_sizes")

    if original_sizes is None:
        orig_hw: List[List[int]] = []
        for im in batch_images:
            if hasattr(im, "size") and not isinstance(im, np.ndarray):
                # PIL image: size=(W,H)
                w, h = im.size
                orig_hw.append([int(h), int(w)])
            else:
                # numpy image: shape=(H,W,C)
                h, w = int(im.shape[0]), int(im.shape[1])
                orig_hw.append([h, w])
        original_sizes = torch.tensor(orig_hw, dtype=torch.int64, device=inputs["pixel_values"].device)

    if reshaped_sizes is None:
        ph = int(inputs["pixel_values"].shape[-2])
        pw = int(inputs["pixel_values"].shape[-1])
        reshaped_sizes = torch.tensor(
            [[ph, pw] for _ in range(len(batch_images))],
            dtype=torch.int64,
            device=inputs["pixel_values"].device,
        )

    return original_sizes.detach().cpu(), reshaped_sizes.detach().cpu()


def post_process_masks_compat(
    image_processor: Any,
    pred_masks_cpu: torch.Tensor,
    original_sizes_cpu: torch.Tensor,
    reshaped_sizes_cpu: torch.Tensor,
):
    """
    Call `post_process_masks` across SAM API variants:
    - SAM1-style: (masks, original_sizes, reshaped_input_sizes, ...)
    - SAM2-fast style: (masks, original_sizes, ...)
    """
    post_fn = image_processor.post_process_masks
    params = inspect.signature(post_fn).parameters
    if "reshaped_input_sizes" in params:
        return post_fn(pred_masks_cpu, original_sizes_cpu, reshaped_sizes_cpu)
    return post_fn(pred_masks_cpu, original_sizes_cpu)

