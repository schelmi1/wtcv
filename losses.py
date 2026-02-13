#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def mcc_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()

    probs = probs.view(probs.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    tp = (probs * targets).sum(dim=1)
    tn = ((1 - probs) * (1 - targets)).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = numerator / (denominator + eps)
    return 1.0 - mcc.mean()


def boundary_map(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W), values in [0,1]
    dil = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return (dil - ero).clamp(0.0, 1.0)


def mcc_bce_boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mcc_weight: float,
    bce_weight: float,
    boundary_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    targets = targets.float()
    mcc = mcc_loss_with_logits(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probs = torch.sigmoid(logits)
    pred_b = boundary_map(probs)
    tgt_b = boundary_map(targets)
    bnd = F.binary_cross_entropy(pred_b, tgt_b)

    total = mcc_weight * mcc + bce_weight * bce + boundary_weight * bnd
    parts = {
        "mcc": float(mcc.detach().item()),
        "bce": float(bce.detach().item()),
        "boundary": float(bnd.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, parts


def segmentation_metrics(
    pred_logit: torch.Tensor,
    seg_target: torch.Tensor,
    thr: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    probs = torch.sigmoid(pred_logit).float()
    tgt = seg_target.float()

    probs_f = probs.view(probs.shape[0], -1)
    tgt_f = tgt.view(tgt.shape[0], -1)

    inter = (probs_f * tgt_f).sum(dim=1)
    union = (probs_f + tgt_f - probs_f * tgt_f).sum(dim=1)
    valid_union = union > eps
    if bool(valid_union.any().item()):
        soft_iou = float((inter[valid_union] / (union[valid_union] + eps)).mean().item())
    else:
        soft_iou = float("nan")

    pos_mask = tgt_f.sum(dim=1) > 0
    if bool(pos_mask.any().item()):
        pos_inter = inter[pos_mask]
        pos_union = union[pos_mask]
        pos_iou = float((pos_inter / (pos_union + eps)).mean().item())
    else:
        pos_iou = float("nan")

    pred_bin = (probs_f > thr).float()
    neg_mask = ~pos_mask
    if bool(neg_mask.any().item()):
        neg_has_fp = (pred_bin[neg_mask].sum(dim=1) > 0).float()
        neg_fp_rate = float(neg_has_fp.mean().item())
    else:
        neg_fp_rate = float("nan")

    return {
        "soft_iou": soft_iou,
        "pos_iou": pos_iou,
        "neg_fp_rate": neg_fp_rate,
    }

