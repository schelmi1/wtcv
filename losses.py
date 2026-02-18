#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Optional, Tuple

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


def focal_bce_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    targets = targets.float()
    probs = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = -alpha_t * torch.pow(1.0 - pt, gamma) * torch.log(pt)
    return focal.mean()


def tversky_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    targets = targets.float()
    probs = torch.sigmoid(logits)
    probs_f = probs.view(probs.shape[0], -1)
    targets_f = targets.view(targets.shape[0], -1)
    tp = (probs_f * targets_f).sum(dim=1)
    fp = (probs_f * (1.0 - targets_f)).sum(dim=1)
    fn = ((1.0 - probs_f) * targets_f).sum(dim=1)
    score = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return 1.0 - score.mean()


def focal_tversky_boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focal_weight: float,
    tversky_weight: float,
    boundary_weight: float,
    focal_alpha: float,
    focal_gamma: float,
    tversky_alpha: float,
    tversky_beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    targets = targets.float()
    focal = focal_bce_loss_with_logits(
        logits,
        targets,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )
    tversky = tversky_loss_with_logits(
        logits,
        targets,
        alpha=tversky_alpha,
        beta=tversky_beta,
    )
    probs = torch.sigmoid(logits)
    pred_b = boundary_map(probs)
    tgt_b = boundary_map(targets)
    bnd = F.binary_cross_entropy(pred_b, tgt_b)
    total = focal_weight * focal + tversky_weight * tversky + boundary_weight * bnd
    parts = {
        "mcc": float("nan"),
        "bce": float("nan"),
        "focal": float(focal.detach().item()),
        "tversky": float(tversky.detach().item()),
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


def semantic_preserve_losses(
    feat_dino: torch.Tensor,
    feat_adapted: torch.Tensor,
    seg_target: torch.Tensor,
    bg_weight: float,
    fg_weight: float,
    var_gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_hw = (int(seg_target.shape[-2]), int(seg_target.shape[-1]))
    d = F.interpolate(feat_dino.detach(), size=target_hw, mode="bilinear", align_corners=False)
    a = F.interpolate(feat_adapted, size=target_hw, mode="bilinear", align_corners=False)

    d_n = F.normalize(d, dim=1, eps=1e-6)
    a_n = F.normalize(a, dim=1, eps=1e-6)
    cos = (a_n * d_n).sum(dim=1)
    tok_loss = 1.0 - cos

    fg = (seg_target[:, 0] > 0.5).float()
    w = float(bg_weight) * (1.0 - fg) + float(fg_weight) * fg
    preserve = (w * tok_loss).sum() / torch.clamp(w.sum(), min=1.0)

    # Small anti-collapse variance term on adapted normalized features.
    a_flat = a_n.flatten(2)
    std = a_flat.std(dim=2, unbiased=False)
    var = torch.relu(float(var_gamma) - std).mean()
    return preserve, var


def compose_training_loss(
    pred: Dict[str, torch.Tensor],
    seg_target: torch.Tensor,
    tile_target: torch.Tensor,
    zoom_target: Optional[torch.Tensor],
    fp_target: Optional[torch.Tensor],
    segmentation_loss: str,
    mcc_weight: float,
    bce_weight: float,
    focal_weight: float,
    tversky_weight: float,
    focal_alpha: float,
    focal_gamma: float,
    tversky_alpha: float,
    tversky_beta: float,
    boundary_weight: float,
    tile_cls_weight: float,
    zoom_cls_weight: float,
    use_fp_supervision: bool,
    fp_neg_weight: float,
    training_strategy: str,
    preserve_weight: float,
    preserve_bg_weight: float,
    preserve_fg_weight: float,
    var_weight: float,
    var_gamma: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    seg_key = str(segmentation_loss).strip().lower().replace(" ", "").replace("+", "_")
    if seg_key in {"bce_mcc", "mcc_bce"}:
        loss, parts = mcc_bce_boundary_loss(
            pred["seg_logit"],
            seg_target,
            mcc_weight=mcc_weight,
            bce_weight=bce_weight,
            boundary_weight=boundary_weight,
        )
        parts["focal"] = float("nan")
        parts["tversky"] = float("nan")
    elif seg_key in {"focalbce_tversky", "focal_bce_tversky", "focaltversky"}:
        loss, parts = focal_tversky_boundary_loss(
            pred["seg_logit"],
            seg_target,
            focal_weight=focal_weight,
            tversky_weight=tversky_weight,
            boundary_weight=boundary_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
        )
    else:
        raise ValueError(
            f"Unsupported segmentation_loss '{segmentation_loss}'. "
            "Use one of: bce_mcc, focal_bce_tversky"
        )

    tile_cls_val = float("nan")
    if ("tile_logit" in pred) and (tile_cls_weight > 0):
        tile_bce = F.binary_cross_entropy_with_logits(pred["tile_logit"], tile_target)
        loss = loss + tile_cls_weight * tile_bce
        tile_cls_val = float(tile_bce.detach().item())
    parts["tile_cls"] = tile_cls_val

    zoom_cls_val = float("nan")
    if ("zoom_logit" in pred) and (zoom_cls_weight > 0) and (zoom_target is not None):
        zoom_bce = F.binary_cross_entropy_with_logits(pred["zoom_logit"], zoom_target)
        loss = loss + zoom_cls_weight * zoom_bce
        zoom_cls_val = float(zoom_bce.detach().item())
    parts["zoom_cls"] = zoom_cls_val

    fp_sup_val = float("nan")
    if use_fp_supervision and (fp_neg_weight > 0) and (fp_target is not None):
        fp_mask = fp_target > 0.5
        if bool(fp_mask.any().item()):
            fp_bce = F.binary_cross_entropy_with_logits(
                pred["seg_logit"][fp_mask],
                torch.zeros_like(pred["seg_logit"][fp_mask]),
            )
            loss = loss + fp_neg_weight * fp_bce
            fp_sup_val = float(fp_bce.detach().item())
    parts["fp_sup"] = fp_sup_val

    preserve_val = float("nan")
    var_val = float("nan")
    use_semantic_preserve = str(training_strategy).lower() == "semantic_preserve"
    if use_semantic_preserve:
        feat_dino = pred.get("feat_dino", None)
        feat_adapted = pred.get("feat_adapted", None)
        if feat_dino is not None and feat_adapted is not None:
            l_preserve, l_var = semantic_preserve_losses(
                feat_dino=feat_dino,
                feat_adapted=feat_adapted,
                seg_target=seg_target,
                bg_weight=preserve_bg_weight,
                fg_weight=preserve_fg_weight,
                var_gamma=var_gamma,
            )
            if preserve_weight > 0:
                loss = loss + float(preserve_weight) * l_preserve
            if var_weight > 0:
                loss = loss + float(var_weight) * l_var
            preserve_val = float(l_preserve.detach().item())
            var_val = float(l_var.detach().item())
    parts["preserve"] = preserve_val
    parts["var"] = var_val
    parts["total"] = float(loss.detach().item())

    return loss, parts


def get_epoch_mcc_weight(target_weight: float, warmup_epochs: int, epoch: int) -> float:
    if warmup_epochs <= 0:
        return float(target_weight)
    if epoch <= 0:
        return 0.0
    if epoch >= warmup_epochs:
        return float(target_weight)
    return float(target_weight) * (float(epoch) / float(warmup_epochs))
