#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from backbones_adapters import DinoLearnedUpsampler, DinoPixelShuffleUpsampler
from backbones_adapters import ResNetLocalBranch
from wtcv_utils.tiling import crop_with_pad, tile_origins


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Self-supervised DINO/iBOT-style LoRA pretraining")
    ap.add_argument("--input-dir", type=str, required=True, help="Directory with images (recursive), or comma-separated directories")
    ap.add_argument("--output-dir", type=Path, default=Path("runs"), help="Base output directory")
    ap.add_argument("--run-name", type=str, default="", help="Optional run name suffix")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--tile-size", type=int, default=0, help="Optional pretraining tile size (0 disables tiling).")
    ap.add_argument("--tile-stride", type=int, default=0, help="Optional pretraining tile stride (0 => tile_size).")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--local-crop-size", type=int, default=96)
    ap.add_argument("--num-local-crops", type=int, default=4)
    ap.add_argument("--global-min-scale", type=float, default=0.4)
    ap.add_argument("--local-min-scale", type=float, default=0.08)

    ap.add_argument("--dino-model", type=str, default="dinov2_vits14_reg")
    ap.add_argument("--out-dim", type=int, default=65536)
    ap.add_argument("--proj-hidden-dim", type=int, default=2048)
    ap.add_argument("--proj-bottleneck-dim", type=int, default=256)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)

    ap.add_argument("--teacher-momentum", type=float, default=0.996)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--student-temp", type=float, default=0.1)
    ap.add_argument("--center-momentum", type=float, default=0.9)

    ap.add_argument("--ibot-weight", type=float, default=1.0)
    ap.add_argument("--dino-weight", type=float, default=1.0)
    ap.add_argument("--ibot-mask-ratio", type=float, default=0.3)
    ap.add_argument("--koleo-weight", type=float, default=0.0, help="KoLeo uniformity weight on student global CLS features.")

    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-targets", type=str, default="attn.qkv,attn.proj", help="Comma separated name substrings")
    ap.add_argument("--head-only-warmup-epochs", type=int, default=1, help="Train only heads for first N epochs before enabling LoRA updates.")
    ap.add_argument("--warmup-use-vanilla-backbone", action="store_true", default=True, help="During head warmup, feed frozen vanilla DINO features into heads.")
    ap.add_argument("--no-warmup-use-vanilla-backbone", action="store_false", dest="warmup_use_vanilla_backbone")
    ap.add_argument("--lora-log-every-steps", type=int, default=20, help="Compute/log LoRA movement metrics every N steps (0=off).")
    ap.add_argument("--pretrain-mode", type=str, choices=["lora", "upscaling"], default="lora")
    ap.add_argument("--upscale-type", type=str, choices=["learned", "pixelshuffle"], default="learned")
    ap.add_argument("--edge-consistency-weight", type=float, default=0.05, help="Weight for edge consistency regularization in upscaling mode.")
    ap.add_argument("--upscale-gate-init", type=float, default=0.0, help="Initial value for learnable upscaling residual gate (pre-sigmoid).")
    ap.add_argument("--upscale-local-gain", type=float, default=1.0, help="Fixed multiplier for local CNN branch in upscaling mode.")
    ap.add_argument("--upscale-dino-gain", type=float, default=1.0, help="Fixed multiplier for DINO upsample branch in upscaling mode.")
    ap.add_argument("--upscale-refine-gain", type=float, default=1.0, help="Residual gain for post-refine block in upscaling mode.")
    ap.add_argument("--upscale-sharpen-gain", type=float, default=0.0, help="Optional unsharp gain on fused map in upscaling mode.")
    ap.add_argument("--semantic-smooth-weight", type=float, default=0.0, help="Weight for semantic smoothing regularization in upscaling mode.")
    ap.add_argument("--semantic-smooth-tau", type=float, default=0.2, help="Temperature for semantic affinity in smoothing term.")

    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--debug-pca-every-steps", type=int, default=0, help="If >0, save token PCA debug image every N optimization steps.")
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageFolderRecursive(Dataset):
    def __init__(self, roots: Sequence[Path]):
        self.roots = [Path(r) for r in roots]
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths: List[Path] = []
        for root in self.roots:
            self.paths.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Image.Image:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return img


class TiledImageFolderRecursive(Dataset):
    def __init__(self, roots: Sequence[Path], tile_size: int, tile_stride: int):
        self.roots = [Path(r) for r in roots]
        self.tile_size = int(tile_size)
        self.tile_stride = int(tile_stride) if int(tile_stride) > 0 else int(tile_size)
        if self.tile_size <= 0:
            raise ValueError("tile_size must be > 0 for tiled dataset")
        if self.tile_stride <= 0:
            raise ValueError("tile_stride must be > 0 for tiled dataset")
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths: List[Path] = []
        for root in self.roots:
            self.paths.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        self.samples: List[Tuple[Path, int, int]] = []
        for p in self.paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            for x0, y0 in tile_origins(w, h, self.tile_size, self.tile_stride):
                self.samples.append((p, int(x0), int(y0)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Image.Image:
        p, x0, y0 = self.samples[idx]
        img = Image.open(p).convert("RGB")
        tile = crop_with_pad(img, int(x0), int(y0), int(self.tile_size))
        return tile


class MultiCropAug:
    def __init__(self, image_size: int, local_crop_size: int, num_local_crops: int, global_min_scale: float, local_min_scale: float):
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.global_t = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(global_min_scale, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        self.local_t = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=(local_min_scale, global_min_scale), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        self.num_local_crops = int(num_local_crops)

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        crops = [self.global_t(img), self.global_t(img)]
        for _ in range(self.num_local_crops):
            crops.append(self.local_t(img))
        return crops


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.rank = int(rank)
        self.scale = float(alpha) / float(max(1, rank))
        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.drop(self.lora_a(x))) * self.scale


def apply_lora(model: nn.Module, target_substrings: Sequence[str], rank: int, alpha: float, dropout: float) -> int:
    names = [n for n, _ in model.named_modules()]
    replaced = 0
    for full_name in names:
        if not any(s in full_name for s in target_substrings):
            continue
        parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        child_name = full_name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        child = getattr(parent, child_name, None)
        if isinstance(child, nn.Linear):
            mod = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            mod = mod.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(parent, child_name, mod)
            replaced += 1
    return replaced


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.last = nn.Linear(bottleneck_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)


@dataclass
class SSLState:
    center_cls: torch.Tensor
    center_patch: torch.Tensor


class Stage1UpscaleTokenAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        upscale_type: str = "learned",
        local_backbone: str = "resnet18",
        gate_init: float = 0.0,
        local_gain: float = 1.0,
        dino_gain: float = 1.0,
        refine_gain: float = 1.0,
        sharpen_gain: float = 0.0,
        semantic_smooth_tau: float = 0.2,
    ):
        super().__init__()
        self.channels = int(channels)
        t = str(upscale_type).strip().lower()
        self.upscale_type = t
        if t == "pixelshuffle":
            self.up = DinoPixelShuffleUpsampler(self.channels)
        else:
            self.up = DinoLearnedUpsampler(self.channels)
        self.local = ResNetLocalBranch(backbone=str(local_backbone), out_channels=self.channels)
        self.local_proj = nn.Conv2d(int(self.local.out_channels), self.channels, kernel_size=1)
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.local_gain = float(local_gain)
        self.dino_gain = float(dino_gain)
        self.refine_gain = float(refine_gain)
        self.sharpen_gain = float(sharpen_gain)
        self.semantic_smooth_tau = float(max(1e-4, semantic_smooth_tau))
        self.post_refine = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
        )

    def _reshape_tokens(self, tokens: torch.Tensor, image_hw: Tuple[int, int], patch_size: int) -> Tuple[torch.Tensor, int, int]:
        b, n, c = tokens.shape
        h, w = int(image_hw[0]), int(image_hw[1])
        gh = max(1, h // int(patch_size))
        gw = max(1, w // int(patch_size))
        if gh * gw != int(n):
            # Fallback: infer a valid factor pair closest to image aspect ratio.
            target_ar = float(w) / float(max(1, h))
            best: Tuple[float, int, int] | None = None
            nn = int(n)
            for d in range(1, int(math.sqrt(float(nn))) + 1):
                if (nn % d) != 0:
                    continue
                a, b2 = int(d), int(nn // d)
                for hh, ww in ((a, b2), (b2, a)):
                    ar = float(ww) / float(max(1, hh))
                    score = abs(ar - target_ar)
                    if best is None or score < best[0]:
                        best = (score, hh, ww)
            if best is None:
                raise RuntimeError(f"Cannot factor token count N={n} into grid")
            gh, gw = int(best[1]), int(best[2])
        fmap = tokens.transpose(1, 2).reshape(b, c, gh, gw).contiguous()
        return fmap, gh, gw

    @staticmethod
    def _edge_consistency_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_m = a.mean(dim=1, keepdim=True)
        b_m = b.mean(dim=1, keepdim=True)
        ax = a_m[..., :, 1:] - a_m[..., :, :-1]
        ay = a_m[..., 1:, :] - a_m[..., :-1, :]
        bx = b_m[..., :, 1:] - b_m[..., :, :-1]
        by = b_m[..., 1:, :] - b_m[..., :-1, :]
        return (ax - bx).abs().mean() + (ay - by).abs().mean()

    @staticmethod
    def _pairwise_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # a,b: [B,C,H,W] -> cosine [B,1,H,W]
        an = F.normalize(a, dim=1, eps=eps)
        bn = F.normalize(b, dim=1, eps=eps)
        return (an * bn).sum(dim=1, keepdim=True)

    def _semantic_smoothness_loss(self, dino_low: torch.Tensor, adapted_low: torch.Tensor) -> torch.Tensor:
        # dino_low/adapted_low: [B,C,H,W] at token-grid resolution.
        tau = self.semantic_smooth_tau
        ax = adapted_low[..., :, 1:] - adapted_low[..., :, :-1]
        ay = adapted_low[..., 1:, :] - adapted_low[..., :-1, :]
        simx = self._pairwise_cos(dino_low[..., :, 1:], dino_low[..., :, :-1])
        simy = self._pairwise_cos(dino_low[..., 1:, :], dino_low[..., :-1, :])
        wx = torch.sigmoid(simx / tau)
        wy = torch.sigmoid(simy / tau)
        lx = (wx * (ax * ax).mean(dim=1, keepdim=True)).mean()
        ly = (wy * (ay * ay).mean(dim=1, keepdim=True)).mean()
        return lx + ly

    def _forward_core(
        self,
        tokens: torch.Tensor,
        image: torch.Tensor,
        patch_size: int,
        return_upsampled_tokens: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        image_hw = (int(image.shape[-2]), int(image.shape[-1]))
        fmap, gh, gw = self._reshape_tokens(tokens, image_hw=image_hw, patch_size=patch_size)
        b, c = fmap.shape[0], fmap.shape[1]
        local = self.local(image)
        target_hw = (int(local.shape[-2]), int(local.shape[-1]))
        local_p = self.local_proj(local)
        up = self.up(fmap, target_hw=target_hw)
        if up.shape[-2:] != target_hw:
            up = F.interpolate(up, size=target_hw, mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.residual_gate)
        fused = self.local_gain * local_p + (self.dino_gain * gate) * up
        fused = fused + self.refine_gain * self.post_refine(fused)
        if self.sharpen_gain > 0.0:
            blur = F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)
            fused = fused + self.sharpen_gain * (fused - blur)
        edge_loss = self._edge_consistency_loss(fused, local_p)
        up_tokens: torch.Tensor | None = None
        if return_upsampled_tokens:
            up_tokens = fused.reshape(b, c, int(fused.shape[-2] * fused.shape[-1])).transpose(1, 2).contiguous()
        down = F.interpolate(fused, size=(gh, gw), mode="area")
        sem_loss = self._semantic_smoothness_loss(fmap, down)
        out = down.reshape(b, c, gh * gw).transpose(1, 2).contiguous()
        return out, up_tokens, edge_loss, sem_loss

    def forward_with_upsampled_tokens(
        self,
        tokens: torch.Tensor,
        image: torch.Tensor,
        patch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out, up_tokens, edge_loss, sem_loss = self._forward_core(
            tokens=tokens,
            image=image,
            patch_size=patch_size,
            return_upsampled_tokens=True,
        )
        if up_tokens is None:
            raise RuntimeError("Expected upsampled tokens but got None")
        return out, up_tokens, edge_loss, sem_loss

    def forward_with_edge_loss(
        self,
        tokens: torch.Tensor,
        image: torch.Tensor,
        patch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _up_tokens, edge_loss, sem_loss = self._forward_core(
            tokens=tokens,
            image=image,
            patch_size=patch_size,
            return_upsampled_tokens=False,
        )
        return out, edge_loss, sem_loss

    def forward(self, tokens: torch.Tensor, image: torch.Tensor, patch_size: int) -> torch.Tensor:
        out, _, _ = self.forward_with_edge_loss(tokens=tokens, image=image, patch_size=patch_size)
        return out


def count_params(module: nn.Module) -> Tuple[int, int]:
    total = int(sum(p.numel() for p in module.parameters()))
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return trainable, total


def print_trainable_summary(
    student_backbone: nn.Module,
    student_cls_head: nn.Module,
    student_patch_head: nn.Module,
    epoch1_warmup: bool,
    pretrain_mode: str,
    student_token_adapter: nn.Module | None = None,
) -> None:
    sb_train, sb_total = count_params(student_backbone)
    ad_train, ad_total = (0, 0)
    if student_token_adapter is not None:
        ad_train, ad_total = count_params(student_token_adapter)
    cls_train, cls_total = count_params(student_cls_head)
    patch_train, patch_total = count_params(student_patch_head)
    epoch1_trainable = int(sb_train + ad_train + cls_train + patch_train)

    lora_max = int(sum(p.numel() for n, p in student_backbone.named_parameters() if "lora_" in n))
    lora_trainable_now = int(sum(p.numel() for n, p in student_backbone.named_parameters() if ("lora_" in n and p.requires_grad)))
    adapter_max = int(ad_total)
    max_trainable = int((lora_max if str(pretrain_mode) == "lora" else adapter_max) + cls_total + patch_total)
    total_params = int(sb_total + ad_total + cls_total + patch_total)

    module_flags: List[str] = []
    if epoch1_warmup:
        module_flags += ["student_cls_head", "student_patch_head"]
    else:
        if str(pretrain_mode) == "lora":
            module_flags += ["student_backbone.lora", "student_cls_head", "student_patch_head"]
        else:
            module_flags += ["student_token_adapter", "student_cls_head", "student_patch_head"]
    module_msg = ", ".join(module_flags)

    print(
        f"trainable_modules: {module_msg} | trainable_params={epoch1_trainable:,} "
        f"| lora_trainable_params(epoch1)={lora_trainable_now:,} lora_params(max)={lora_max:,} "
        f"| adapter_trainable_params(epoch1)={ad_train:,} adapter_params(max)={ad_total:,}"
    )
    print("trainable module breakdown:")
    print(
        f"  - {'student_backbone':18s} trainable={sb_train:,} "
        f"frozen={sb_total - sb_train:,} total={sb_total:,}"
    )
    if student_token_adapter is not None:
        print(
            f"  - {'student_token_adapter':18s} trainable={ad_train:,} "
            f"frozen={ad_total - ad_train:,} total={ad_total:,}"
        )
    print(
        f"  - {'student_cls_head':18s} trainable={cls_train:,} "
        f"frozen={cls_total - cls_train:,} total={cls_total:,}"
    )
    print(
        f"  - {'student_patch_head':18s} trainable={patch_train:,} "
        f"frozen={patch_total - patch_train:,} total={patch_total:,}"
    )
    print(
        f"  - {'TOTAL':18s} trainable(epoch1)={epoch1_trainable:,} "
        f"trainable(max)={max_trainable:,} frozen={total_params - epoch1_trainable:,} total={total_params:,}"
    )


def set_lora_trainable(student_backbone: nn.Module, train_lora: bool) -> None:
    for n, p in student_backbone.named_parameters():
        if "lora_" in n:
            p.requires_grad = bool(train_lora)


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(trainable)


def snapshot_lora_params(student_backbone: nn.Module) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for n, p in student_backbone.named_parameters():
        if "lora_" in n:
            snap[n] = p.detach().clone()
    return snap


def snapshot_module_params(module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for n, p in module.named_parameters():
        snap[f"{prefix}{n}"] = p.detach().clone()
    return snap


@torch.no_grad()
def lora_movement_metrics(student_backbone: nn.Module, lora_init: Dict[str, torch.Tensor]) -> Dict[str, float]:
    sum_sq = 0.0
    delta_sq = 0.0
    for n, p in student_backbone.named_parameters():
        if "lora_" not in n:
            continue
        w = p.detach()
        sum_sq += float((w * w).sum().item())
        w0 = lora_init.get(n, None)
        if w0 is not None:
            d = w - w0
            delta_sq += float((d * d).sum().item())
    lora_norm = float(math.sqrt(max(0.0, sum_sq)))
    lora_delta = float(math.sqrt(max(0.0, delta_sq)))
    lora_rel = float(lora_delta / (lora_norm + 1e-12))
    return {
        "lora_norm": lora_norm,
        "lora_delta_norm": lora_delta,
        "lora_delta_rel": lora_rel,
    }


@torch.no_grad()
def module_movement_metrics(module: nn.Module, init_snap: Dict[str, torch.Tensor], prefix: str = "") -> Dict[str, float]:
    sum_sq = 0.0
    delta_sq = 0.0
    for n, p in module.named_parameters():
        k = f"{prefix}{n}"
        w = p.detach()
        sum_sq += float((w * w).sum().item())
        w0 = init_snap.get(k, None)
        if w0 is not None:
            d = w - w0
            delta_sq += float((d * d).sum().item())
    norm = float(math.sqrt(max(0.0, sum_sq)))
    delta = float(math.sqrt(max(0.0, delta_sq)))
    rel = float(delta / (norm + 1e-12))
    return {
        "norm": norm,
        "delta_norm": delta,
        "delta_rel": rel,
    }


@torch.no_grad()
def update_teacher(student_backbone: nn.Module, teacher_backbone: nn.Module, momentum: float) -> None:
    for ps, pt in zip(student_backbone.parameters(), teacher_backbone.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


@torch.no_grad()
def update_center(old_center: torch.Tensor, teacher_logits: torch.Tensor, momentum: float) -> torch.Tensor:
    batch_center = teacher_logits.mean(dim=0, keepdim=True)
    return old_center * momentum + batch_center * (1.0 - momentum)


def dino_ce(student_logits: torch.Tensor, teacher_logits: torch.Tensor, student_temp: float, teacher_temp: float, center: torch.Tensor) -> torch.Tensor:
    s = F.log_softmax(student_logits / float(student_temp), dim=-1)
    t = F.softmax((teacher_logits - center) / float(teacher_temp), dim=-1)
    return -(t * s).sum(dim=-1).mean()


def random_token_mask(batch: int, tokens: int, ratio: float, device: torch.device) -> torch.Tensor:
    keep = max(1, int(tokens * max(0.0, min(1.0, 1.0 - ratio))))
    idx = torch.rand(batch, tokens, device=device).argsort(dim=1)
    mask = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, idx[:, :keep], True)
    return mask


def koleo_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # KoLeo-style nearest-neighbor spread on normalized features.
    # z: [B, C]
    if z.ndim != 2 or z.shape[0] < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.t()  # [B, B]
    b = int(sim.shape[0])
    sim.fill_diagonal_(-1.0)
    nn_sim, _ = sim.max(dim=1)  # closest neighbor by cosine
    # Convert cosine to distance proxy in [0, 2].
    nn_dist = (1.0 - nn_sim).clamp_min(eps)
    return (-torch.log(nn_dist)).mean()


def cosine_warmup_cosine_decay(step: int, total_steps: int, warmup_steps: int, base: float, min_v: float) -> float:
    if step < warmup_steps:
        return base * float(step + 1) / float(max(1, warmup_steps))
    t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_v + 0.5 * (base - min_v) * (1.0 + math.cos(math.pi * t))


def make_run_dir(output_dir: Path, run_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{run_name.strip()}" if run_name.strip() else "-ssl-pretrain"
    rd = output_dir / f"{ts}{suffix}"
    (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
    return rd


def _normalize_crop_size(size: int, patch: int) -> int:
    s = int(size)
    p = max(1, int(patch))
    eff = (s // p) * p
    if eff < p:
        eff = p
    return eff


def _prepare_dino_scaled_input(x: torch.Tensor, patch: int, scale_num: int = 14, scale_den: int = 16) -> torch.Tensor:
    h, w = int(x.shape[-2]), int(x.shape[-1])
    dh = int(round(float(h) * float(scale_num) / float(scale_den)))
    dw = int(round(float(w) * float(scale_num) / float(scale_den)))
    dh = max(int(patch), int((dh // int(patch)) * int(patch)))
    dw = max(int(patch), int((dw // int(patch)) * int(patch)))
    if dh == h and dw == w:
        return x
    return F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)


def _token_grid_shape(num_tokens: int) -> Tuple[int, int]:
    gh = int(round(math.sqrt(float(num_tokens))))
    if gh <= 0:
        raise RuntimeError(f"Invalid token count: {num_tokens}")
    if num_tokens % gh == 0:
        return gh, num_tokens // gh
    for d in range(gh, 0, -1):
        if num_tokens % d == 0:
            return d, num_tokens // d
    return 1, num_tokens


def _tokens_to_pca_rgb(tokens: torch.Tensor) -> np.ndarray:
    # tokens: (N, C)
    t = tokens.detach().float()
    t = t - t.mean(dim=0, keepdim=True)
    n, c = t.shape
    q = min(3, n, c)
    if q <= 0:
        raise RuntimeError("Empty token tensor in PCA conversion")
    _, _, v = torch.pca_lowrank(t, q=q)
    p = t @ v[:, :q]
    if q < 3:
        p = torch.cat([p, torch.zeros(n, 3 - q, device=p.device, dtype=p.dtype)], dim=1)
    gh, gw = _token_grid_shape(int(n))
    p = p.reshape(gh, gw, 3)
    p = p - p.amin(dim=(0, 1), keepdim=True)
    p = p / (p.amax(dim=(0, 1), keepdim=True) + 1e-6)
    arr = (p.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def _tokens_to_kmeans_rgb(tokens: torch.Tensor, k: int = 8) -> np.ndarray:
    t = tokens.detach().float().cpu().numpy()
    n = int(t.shape[0])
    if n <= 0:
        raise RuntimeError("Empty token tensor in KMeans conversion")
    k_eff = max(1, min(int(k), n))
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
    labels = km.fit_predict(t)
    gh, gw = _token_grid_shape(n)

    cmap = matplotlib.colormaps["tab20c"]
    # Sample 8 distinct colors from tab20c; fallback wraps when k_eff > 8.
    palette = (np.array([cmap(i / 7.0)[:3] for i in range(8)], dtype=np.float32) * 255.0).astype(np.uint8)
    rgb = palette[labels % 8]
    return rgb.reshape(gh, gw, 3)


def _denorm_to_rgb_u8(x: torch.Tensor) -> np.ndarray:
    # x: (3,H,W), normalized with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0.0, 1.0)
    return (y.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def write_debug_pca_panel(
    out_path: Path,
    img_chw_norm: torch.Tensor,
    vanilla_tokens: torch.Tensor,
    adapted_tokens: torch.Tensor,
) -> None:
    orig = _denorm_to_rgb_u8(img_chw_norm)
    vh, vw = orig.shape[:2]
    v_rgb = _tokens_to_pca_rgb(vanilla_tokens)
    a_rgb = _tokens_to_pca_rgb(adapted_tokens)
    v_img = np.array(Image.fromarray(v_rgb).resize((vw, vh), Image.NEAREST), dtype=np.uint8)
    a_img = np.array(Image.fromarray(a_rgb).resize((vw, vh), Image.NEAREST), dtype=np.uint8)
    panel = np.concatenate([orig, v_img, a_img], axis=1)
    canvas = Image.fromarray(panel)
    draw = ImageDraw.Draw(canvas)
    col_w = vw
    titles = ["Original", "Vanilla DINO PCA", "Adapted PCA (upscaled when enabled)"]
    for i, t in enumerate(titles):
        x0 = i * col_w
        draw.rectangle([x0, 0, x0 + col_w, 22], fill=(0, 0, 0))
        draw.text((x0 + 6, 5), t, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


@torch.no_grad()
def write_debug_kmeans_panel(
    out_path: Path,
    img_chw_norm: torch.Tensor,
    vanilla_tokens: torch.Tensor,
    adapted_tokens: torch.Tensor,
    k: int = 8,
) -> None:
    orig = _denorm_to_rgb_u8(img_chw_norm)
    vh, vw = orig.shape[:2]
    v_rgb = _tokens_to_kmeans_rgb(vanilla_tokens, k=k)
    a_rgb = _tokens_to_kmeans_rgb(adapted_tokens, k=k)
    v_img = np.array(Image.fromarray(v_rgb).resize((vw, vh), Image.NEAREST), dtype=np.uint8)
    a_img = np.array(Image.fromarray(a_rgb).resize((vw, vh), Image.NEAREST), dtype=np.uint8)

    panel = np.concatenate([orig, v_img, a_img], axis=1)
    canvas = Image.fromarray(panel)
    draw = ImageDraw.Draw(canvas)
    col_w = vw
    titles = [f"Original", f"Vanilla KMeans(k={k})", f"Adapted KMeans(k={k}) (upscaled when enabled)"]
    for i, t in enumerate(titles):
        x0 = i * col_w
        draw.rectangle([x0, 0, x0 + col_w, 22], fill=(0, 0, 0))
        draw.text((x0 + 6, 5), t, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    device = torch.device(args.device) if str(args.device).strip() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dirs = [Path(s.strip()) for s in str(args.input_dir).split(",") if s.strip()]
    if len(input_dirs) == 0:
        raise RuntimeError("--input-dir is empty; provide one or more directories")
    missing = [str(p) for p in input_dirs if (not p.exists() or not p.is_dir())]
    if missing:
        raise FileNotFoundError(f"Missing input directories: {missing}")

    use_tiling = int(args.tile_size) > 0
    if use_tiling:
        ds = TiledImageFolderRecursive(
            input_dirs,
            tile_size=int(args.tile_size),
            tile_stride=(int(args.tile_stride) if int(args.tile_stride) > 0 else int(args.tile_size)),
        )
    else:
        ds = ImageFolderRecursive(input_dirs)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in input directories: {[str(p) for p in input_dirs]}")

    run_dir = make_run_dir(args.output_dir, args.run_name)
    cfg = vars(args).copy()
    for k, v in list(cfg.items()):
        if isinstance(v, Path):
            cfg[k] = str(v)
    cfg["dataset_size"] = len(ds)
    cfg["device"] = str(device)
    cfg["input_dirs_resolved"] = [str(p) for p in input_dirs]
    cfg["tiling_enabled"] = bool(use_tiling)
    cfg["tile_size_effective"] = int(args.tile_size) if use_tiling else 0
    cfg["tile_stride_effective"] = (int(args.tile_stride) if int(args.tile_stride) > 0 else int(args.tile_size)) if use_tiling else 0
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"device={device}")
    print(f"input_dirs={[str(p) for p in input_dirs]}")
    if use_tiling:
        n_images = len(getattr(ds, "paths", []))
        print(f"dataset_images={n_images}")
        print(f"dataset_tiles={len(ds)} tile_size={cfg['tile_size_effective']} tile_stride={cfg['tile_stride_effective']}")
    else:
        print(f"dataset_images={len(ds)}")
    print(f"run_dir={run_dir}")

    student_backbone = torch.hub.load("facebookresearch/dinov2", str(args.dino_model), trust_repo=bool(args.trust_torch_hub_repo)).to(device)
    teacher_backbone = torch.hub.load("facebookresearch/dinov2", str(args.dino_model), trust_repo=bool(args.trust_torch_hub_repo)).to(device)
    vanilla_backbone = None
    need_vanilla = int(args.debug_pca_every_steps) > 0 or bool(args.warmup_use_vanilla_backbone)
    if need_vanilla:
        vanilla_backbone = torch.hub.load("facebookresearch/dinov2", str(args.dino_model), trust_repo=bool(args.trust_torch_hub_repo)).to(device).eval()
        for p in vanilla_backbone.parameters():
            p.requires_grad = False
    patch = int(student_backbone.patch_embed.patch_size[0]) if isinstance(student_backbone.patch_embed.patch_size, tuple) else int(student_backbone.patch_embed.patch_size)
    eff_global_size = _normalize_crop_size(int(args.image_size), patch)
    eff_local_size = _normalize_crop_size(int(args.local_crop_size), patch)
    if eff_global_size != int(args.image_size):
        print(f"adjusted_global_crop_size={eff_global_size} (requested={int(args.image_size)}, patch={patch})")
    if eff_local_size != int(args.local_crop_size):
        print(f"adjusted_local_crop_size={eff_local_size} (requested={int(args.local_crop_size)}, patch={patch})")
    debug_image_size = int(eff_global_size * 2)
    cfg["debug_effective_image_size"] = int(debug_image_size)
    cfg["patch_size"] = int(patch)
    cfg["effective_image_size"] = int(eff_global_size)
    cfg["effective_local_crop_size"] = int(eff_local_size)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    aug = MultiCropAug(
        image_size=eff_global_size,
        local_crop_size=eff_local_size,
        num_local_crops=int(args.num_local_crops),
        global_min_scale=float(args.global_min_scale),
        local_min_scale=float(args.local_min_scale),
    )

    def collate_pil(batch: List[Image.Image]) -> List[List[torch.Tensor]]:
        return [aug(img) for img in batch]

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pil,
        drop_last=True,
    )

    with torch.inference_mode():
        probe = torch.randn(1, 3, eff_global_size, eff_global_size, device=device)
        fo = student_backbone.forward_features(probe)
        cls_dim = int(fo["x_norm_clstoken"].shape[-1])

    pretrain_mode = str(args.pretrain_mode).strip().lower()
    student_token_adapter: nn.Module | None = None
    n_lora = 0
    if pretrain_mode == "lora":
        targets = [s.strip() for s in str(args.lora_targets).split(",") if s.strip()]
        n_lora = apply_lora(student_backbone, target_substrings=targets, rank=int(args.lora_rank), alpha=float(args.lora_alpha), dropout=float(args.lora_dropout))
        _ = apply_lora(teacher_backbone, target_substrings=targets, rank=int(args.lora_rank), alpha=float(args.lora_alpha), dropout=float(args.lora_dropout))
        teacher_backbone.load_state_dict(student_backbone.state_dict(), strict=True)
    elif pretrain_mode == "upscaling":
        student_token_adapter = Stage1UpscaleTokenAdapter(
            channels=cls_dim,
            upscale_type=str(args.upscale_type),
            gate_init=float(args.upscale_gate_init),
            local_gain=float(args.upscale_local_gain),
            dino_gain=float(args.upscale_dino_gain),
            refine_gain=float(args.upscale_refine_gain),
            sharpen_gain=float(args.upscale_sharpen_gain),
            semantic_smooth_tau=float(args.semantic_smooth_tau),
        ).to(device)
    else:
        raise RuntimeError(f"Unsupported pretrain_mode={pretrain_mode}")

    teacher_backbone.eval()
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    if pretrain_mode == "lora":
        for n, p in student_backbone.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        lora_init = snapshot_lora_params(student_backbone)
        adapter_init: Dict[str, torch.Tensor] = {}
    else:
        for p in student_backbone.parameters():
            p.requires_grad = False
        lora_init = {}
        adapter_init = snapshot_module_params(student_token_adapter, prefix="adapter.") if student_token_adapter is not None else {}

    student_cls_head = MLPHead(cls_dim, int(args.proj_hidden_dim), int(args.proj_bottleneck_dim), int(args.out_dim)).to(device)
    teacher_cls_head = MLPHead(cls_dim, int(args.proj_hidden_dim), int(args.proj_bottleneck_dim), int(args.out_dim)).to(device)
    teacher_cls_head.load_state_dict(student_cls_head.state_dict(), strict=True)
    teacher_cls_head.eval()
    for p in teacher_cls_head.parameters():
        p.requires_grad = False

    student_patch_head = nn.Linear(cls_dim, int(args.out_dim)).to(device)
    teacher_patch_head = nn.Linear(cls_dim, int(args.out_dim)).to(device)
    teacher_patch_head.load_state_dict(student_patch_head.state_dict(), strict=True)
    teacher_patch_head.eval()
    for p in teacher_patch_head.parameters():
        p.requires_grad = False

    params: List[torch.nn.Parameter] = [p for p in student_backbone.parameters() if p.requires_grad]
    if student_token_adapter is not None:
        params += [p for p in student_token_adapter.parameters() if p.requires_grad]
    params += list(student_cls_head.parameters()) + list(student_patch_head.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    total_steps = int(args.epochs) * max(1, len(loader))
    warmup_steps = max(10, int(0.05 * total_steps))
    step = 0

    state = SSLState(
        center_cls=torch.zeros(1, int(args.out_dim), device=device),
        center_patch=torch.zeros(1, int(args.out_dim), device=device),
    )

    history: List[Dict] = []
    print(f"pretrain_mode={pretrain_mode}")
    print(f"lora_modules={n_lora}")
    epoch1_warmup = bool(int(args.head_only_warmup_epochs) >= 1)
    if pretrain_mode == "lora":
        set_lora_trainable(student_backbone, train_lora=not epoch1_warmup)
    if student_token_adapter is not None:
        set_module_trainable(student_token_adapter, trainable=not epoch1_warmup)
    print_trainable_summary(
        student_backbone=student_backbone,
        student_token_adapter=student_token_adapter,
        student_cls_head=student_cls_head,
        student_patch_head=student_patch_head,
        epoch1_warmup=epoch1_warmup,
        pretrain_mode=pretrain_mode,
    )

    for epoch in range(1, int(args.epochs) + 1):
        in_head_warmup = epoch <= int(args.head_only_warmup_epochs)
        if pretrain_mode == "lora":
            set_lora_trainable(student_backbone, train_lora=not in_head_warmup)
        if student_token_adapter is not None:
            set_module_trainable(student_token_adapter, trainable=not in_head_warmup)
        student_backbone.train()
        if student_token_adapter is not None:
            student_token_adapter.train()
        student_cls_head.train()
        student_patch_head.train()

        loss_meter = 0.0
        dino_meter = 0.0
        ibot_meter = 0.0
        koleo_meter = 0.0
        proto_entropy_meter = 0.0
        proto_top1_meter = 0.0
        proto_active_frac_meter = 0.0
        edge_consistency_meter = 0.0
        semantic_smooth_meter = 0.0
        lora_delta_meter = 0.0
        lora_rel_meter = 0.0
        lora_norm_last = 0.0
        lora_measure_count = 0
        up_delta_meter = 0.0
        up_rel_meter = 0.0
        up_norm_last = 0.0
        up_measure_count = 0

        pbar = tqdm(loader, desc=f"epoch {epoch}/{int(args.epochs)}")
        for batch in pbar:
            # batch: list length B, each element list[crops]
            num_crops = 2 + int(args.num_local_crops)
            crops: List[torch.Tensor] = []
            for cidx in range(num_crops):
                crops.append(torch.stack([sample[cidx] for sample in batch], dim=0).to(device, non_blocking=True))

            student_cls_logits: List[torch.Tensor] = []
            student_patch_logits: List[torch.Tensor] = []
            edge_consistency_terms: List[torch.Tensor] = []
            semantic_smooth_terms: List[torch.Tensor] = []
            for c in crops:
                dino_c = _prepare_dino_scaled_input(c, patch=patch) if pretrain_mode == "upscaling" else c
                if in_head_warmup and bool(args.warmup_use_vanilla_backbone) and (vanilla_backbone is not None):
                    with torch.no_grad():
                        fs = vanilla_backbone.forward_features(dino_c)
                else:
                    fs = student_backbone.forward_features(dino_c)
                cls = fs["x_norm_clstoken"]
                tok = fs["x_norm_patchtokens"]
                if (student_token_adapter is not None) and (not in_head_warmup):
                    tok, e_loss, s_loss = student_token_adapter.forward_with_edge_loss(tok, image=c, patch_size=patch)
                    edge_consistency_terms.append(e_loss)
                    semantic_smooth_terms.append(s_loss)
                    cls = tok.mean(dim=1)
                student_cls_logits.append(student_cls_head(cls))
                pt = student_patch_head(tok.reshape(-1, tok.shape[-1])).reshape(tok.shape[0], tok.shape[1], -1)
                student_patch_logits.append(pt)

            with torch.no_grad():
                teacher_cls_logits: List[torch.Tensor] = []
                teacher_patch_logits: List[torch.Tensor] = []
                for c in crops[:2]:  # global only
                    dino_c = _prepare_dino_scaled_input(c, patch=patch) if pretrain_mode == "upscaling" else c
                    ft = teacher_backbone.forward_features(dino_c)
                    cls_t = ft["x_norm_clstoken"]
                    tok_t = ft["x_norm_patchtokens"]
                    teacher_cls_logits.append(teacher_cls_head(cls_t))
                    pt_t = teacher_patch_head(tok_t.reshape(-1, tok_t.shape[-1])).reshape(tok_t.shape[0], tok_t.shape[1], -1)
                    teacher_patch_logits.append(pt_t)

            dino_loss = 0.0
            count = 0
            for sidx in range(len(student_cls_logits)):
                for tidx in range(len(teacher_cls_logits)):
                    if sidx == tidx:
                        continue
                    dino_loss = dino_loss + dino_ce(
                        student_logits=student_cls_logits[sidx],
                        teacher_logits=teacher_cls_logits[tidx].detach(),
                        student_temp=float(args.student_temp),
                        teacher_temp=float(args.teacher_temp),
                        center=state.center_cls,
                    )
                    count += 1
            dino_loss = dino_loss / float(max(1, count))

            ibot_loss = 0.0
            for gidx in range(2):
                spt = student_patch_logits[gidx]
                tpt = teacher_patch_logits[gidx].detach()
                bsz, ntok, dim = spt.shape
                mask = random_token_mask(bsz, ntok, ratio=float(args.ibot_mask_ratio), device=device)

                sflat = spt[mask]
                tflat = tpt[mask]
                s_log = F.log_softmax(sflat / float(args.student_temp), dim=-1)
                t_prob = F.softmax((tflat - state.center_patch) / float(args.teacher_temp), dim=-1)
                ibot_loss = ibot_loss + (-(t_prob * s_log).sum(dim=-1).mean())
            ibot_loss = ibot_loss / 2.0

            if len(edge_consistency_terms) > 0:
                edge_consistency_loss = torch.stack(edge_consistency_terms).mean()
            else:
                edge_consistency_loss = torch.zeros((), device=device)
            if len(semantic_smooth_terms) > 0:
                semantic_smooth_loss = torch.stack(semantic_smooth_terms).mean()
            else:
                semantic_smooth_loss = torch.zeros((), device=device)
            if len(student_cls_logits) >= 2:
                cls_global = torch.cat([student_cls_logits[0], student_cls_logits[1]], dim=0)
            else:
                cls_global = student_cls_logits[0]
            koleo = koleo_loss(cls_global)
            total_loss = (
                float(args.dino_weight) * dino_loss
                + float(args.ibot_weight) * ibot_loss
                + float(args.edge_consistency_weight) * edge_consistency_loss
                + float(args.semantic_smooth_weight) * semantic_smooth_loss
                + float(args.koleo_weight) * koleo
            )

            lr_now = cosine_warmup_cosine_decay(step=step, total_steps=total_steps, warmup_steps=warmup_steps, base=float(args.lr), min_v=float(args.min_lr))
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            with torch.no_grad():
                mom = 1.0 - (1.0 - float(args.teacher_momentum)) * (math.cos(math.pi * step / max(1, total_steps)) + 1.0) / 2.0
                if pretrain_mode == "lora":
                    update_teacher(student_backbone, teacher_backbone, momentum=mom)
                for ps, pt in zip(student_cls_head.parameters(), teacher_cls_head.parameters()):
                    pt.data.mul_(mom).add_(ps.data, alpha=1.0 - mom)
                for ps, pt in zip(student_patch_head.parameters(), teacher_patch_head.parameters()):
                    pt.data.mul_(mom).add_(ps.data, alpha=1.0 - mom)

                all_teacher_cls = torch.cat(teacher_cls_logits, dim=0)
                all_teacher_patch = torch.cat([x.reshape(-1, x.shape[-1]) for x in teacher_patch_logits], dim=0)
                tprob = F.softmax((all_teacher_cls - state.center_cls) / float(args.teacher_temp), dim=-1)
                proto = tprob.mean(dim=0)
                proto_entropy = float((-(proto * (proto + 1e-9).log()).sum()).detach().cpu())
                proto_top1 = float(proto.max().detach().cpu())
                proto_active_frac = float((proto > (1.0 / float(proto.numel()))).float().mean().detach().cpu())
                state.center_cls = update_center(state.center_cls, all_teacher_cls, momentum=float(args.center_momentum))
                state.center_patch = update_center(state.center_patch, all_teacher_patch, momentum=float(args.center_momentum))

            loss_meter += float(total_loss.detach().cpu())
            dino_meter += float(dino_loss.detach().cpu())
            ibot_meter += float(ibot_loss.detach().cpu())
            koleo_meter += float(koleo.detach().cpu())
            edge_consistency_meter += float(edge_consistency_loss.detach().cpu())
            semantic_smooth_meter += float(semantic_smooth_loss.detach().cpu())
            proto_entropy_meter += proto_entropy
            proto_top1_meter += proto_top1
            proto_active_frac_meter += proto_active_frac
            step += 1

            lora_log_n = int(args.lora_log_every_steps)
            if pretrain_mode == "lora" and lora_log_n > 0 and (step % lora_log_n == 0):
                lm = lora_movement_metrics(student_backbone, lora_init)
                lora_norm_last = float(lm["lora_norm"])
                lora_delta_meter += float(lm["lora_delta_norm"])
                lora_rel_meter += float(lm["lora_delta_rel"])
                lora_measure_count += 1
            if (
                pretrain_mode == "upscaling"
                and student_token_adapter is not None
                and lora_log_n > 0
                and (step % lora_log_n == 0)
            ):
                am = module_movement_metrics(student_token_adapter, adapter_init, prefix="adapter.")
                up_norm_last = float(am["norm"])
                up_delta_meter += float(am["delta_norm"])
                up_rel_meter += float(am["delta_rel"])
                up_measure_count += 1

            if (
                int(args.debug_pca_every_steps) > 0
                and (step % int(args.debug_pca_every_steps) == 0)
                and (vanilla_backbone is not None)
                and (not in_head_warmup)
            ):
                try:
                    dbg_in = crops[0][:1]  # first global crop, first sample
                    if dbg_in.shape[-1] != debug_image_size or dbg_in.shape[-2] != debug_image_size:
                        dbg_in = F.interpolate(
                            dbg_in,
                            size=(debug_image_size, debug_image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                    with torch.no_grad():
                        dbg_dino = _prepare_dino_scaled_input(dbg_in, patch=patch) if pretrain_mode == "upscaling" else dbg_in
                        v_feat = vanilla_backbone.forward_features(dbg_dino)
                        s_feat = student_backbone.forward_features(dbg_dino)
                        v_tok = v_feat["x_norm_patchtokens"][0]
                        s_tok = s_feat["x_norm_patchtokens"][0]
                        if student_token_adapter is not None:
                            _, s_tok_up, _dbg_edge, _dbg_sem = student_token_adapter.forward_with_upsampled_tokens(
                                s_tok.unsqueeze(0),
                                image=dbg_in,
                                patch_size=patch,
                            )
                            s_tok = s_tok_up[0]
                    dbg_path = run_dir / "debug_pca" / f"step_{step:07d}_e{epoch:03d}.png"
                    write_debug_pca_panel(
                        out_path=dbg_path,
                        img_chw_norm=dbg_in[0],
                        vanilla_tokens=v_tok,
                        adapted_tokens=s_tok,
                    )
                    dbg_km_path = run_dir / "debug_kmeans" / f"step_{step:07d}_e{epoch:03d}.png"
                    write_debug_kmeans_panel(
                        out_path=dbg_km_path,
                        img_chw_norm=dbg_in[0],
                        vanilla_tokens=v_tok,
                        adapted_tokens=s_tok,
                        k=8,
                    )
                    print(f"debug_pca_saved={dbg_path}")
                    print(f"debug_kmeans_saved={dbg_km_path}")
                except Exception as dbg_e:
                    print(f"debug_pca_failed_step={step} error={dbg_e}")
            step_in_epoch = max(1, step - (epoch - 1) * len(loader))
            postfix = {
                "loss": f"{(loss_meter/step_in_epoch):.4f}",
                "dino": f"{(dino_meter/step_in_epoch):.4f}",
                "ibot": f"{(ibot_meter/step_in_epoch):.4f}",
                "koleo": f"{(koleo_meter/step_in_epoch):.4f}",
                "edge": f"{(edge_consistency_meter/step_in_epoch):.4f}",
                "ssmth": f"{(semantic_smooth_meter/step_in_epoch):.4f}",
                "pent": f"{(proto_entropy_meter/step_in_epoch):.2f}",
                "ptop1": f"{(proto_top1_meter/step_in_epoch):.3f}",
                "pact": f"{(proto_active_frac_meter/step_in_epoch):.3f}",
                "lr": f"{lr_now:.2e}",
            }
            if pretrain_mode == "lora":
                postfix["lora_d"] = (
                    f"{(lora_delta_meter/max(1, lora_measure_count)):.3f}"
                    if int(args.lora_log_every_steps) > 0
                    else "off"
                )
                postfix["lora_rel"] = (
                    f"{(lora_rel_meter/max(1, lora_measure_count)):.4f}"
                    if int(args.lora_log_every_steps) > 0
                    else "off"
                )
            else:
                postfix["up_d"] = (
                    f"{(up_delta_meter/max(1, up_measure_count)):.3f}"
                    if int(args.lora_log_every_steps) > 0
                    else "off"
                )
                postfix["up_rel"] = (
                    f"{(up_rel_meter/max(1, up_measure_count)):.4f}"
                    if int(args.lora_log_every_steps) > 0
                    else "off"
                )
            pbar.set_postfix(postfix)

        epoch_steps = max(1, len(loader))
        row = {
            "epoch": epoch,
            "loss": loss_meter / epoch_steps,
            "dino_loss": dino_meter / epoch_steps,
            "ibot_loss": ibot_meter / epoch_steps,
            "koleo_loss": koleo_meter / epoch_steps,
            "edge_consistency_loss": edge_consistency_meter / epoch_steps,
            "semantic_smooth_loss": semantic_smooth_meter / epoch_steps,
            "lr": lr_now,
            "head_only_warmup": bool(in_head_warmup),
            "proto_entropy": proto_entropy_meter / epoch_steps,
            "proto_top1": proto_top1_meter / epoch_steps,
            "proto_active_frac": proto_active_frac_meter / epoch_steps,
            "lora_norm_last": float(lora_norm_last) if pretrain_mode == "lora" else 0.0,
            "lora_delta_norm_avg": (
                float(lora_delta_meter / max(1, lora_measure_count))
                if (pretrain_mode == "lora" and int(args.lora_log_every_steps) > 0)
                else 0.0
            ),
            "lora_delta_rel_avg": (
                float(lora_rel_meter / max(1, lora_measure_count))
                if (pretrain_mode == "lora" and int(args.lora_log_every_steps) > 0)
                else 0.0
            ),
            "up_norm_last": float(up_norm_last) if pretrain_mode == "upscaling" else 0.0,
            "up_delta_norm_avg": (
                float(up_delta_meter / max(1, up_measure_count))
                if (pretrain_mode == "upscaling" and int(args.lora_log_every_steps) > 0)
                else 0.0
            ),
            "up_delta_rel_avg": (
                float(up_rel_meter / max(1, up_measure_count))
                if (pretrain_mode == "upscaling" and int(args.lora_log_every_steps) > 0)
                else 0.0
            ),
        }
        history.append(row)
        if pretrain_mode == "lora":
            print(
                f"epoch={epoch:02d} loss={row['loss']:.4f} dino={row['dino_loss']:.4f} "
                f"ibot={row['ibot_loss']:.4f} koleo={row['koleo_loss']:.4f} "
                f"edge={row['edge_consistency_loss']:.4f} ssmth={row['semantic_smooth_loss']:.4f} "
                f"pent={row['proto_entropy']:.2f} "
                f"ptop1={row['proto_top1']:.4f} pact={row['proto_active_frac']:.3f} "
                f"lora_norm={row['lora_norm_last']:.3f} "
                f"lora_d={row['lora_delta_norm_avg']:.3f} "
                f"lora_rel={row['lora_delta_rel_avg']:.4f} "
                f"warmup={int(row['head_only_warmup'])} lr={row['lr']:.6e}"
            )
        else:
            print(
                f"epoch={epoch:02d} loss={row['loss']:.4f} dino={row['dino_loss']:.4f} "
                f"ibot={row['ibot_loss']:.4f} koleo={row['koleo_loss']:.4f} "
                f"edge={row['edge_consistency_loss']:.4f} ssmth={row['semantic_smooth_loss']:.4f} "
                f"pent={row['proto_entropy']:.2f} "
                f"ptop1={row['proto_top1']:.4f} pact={row['proto_active_frac']:.3f} "
                f"up_norm={row['up_norm_last']:.3f} "
                f"up_d={row['up_delta_norm_avg']:.3f} "
                f"up_rel={row['up_delta_rel_avg']:.4f} "
                f"warmup={int(row['head_only_warmup'])} lr={row['lr']:.6e}"
            )

        if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0 or epoch == int(args.epochs)):
            ckpt = {
                "epoch": epoch,
                "student_backbone": student_backbone.state_dict(),
                "teacher_backbone": teacher_backbone.state_dict(),
                "student_token_adapter": student_token_adapter.state_dict() if student_token_adapter is not None else None,
                "student_cls_head": student_cls_head.state_dict(),
                "teacher_cls_head": teacher_cls_head.state_dict(),
                "student_patch_head": student_patch_head.state_dict(),
                "teacher_patch_head": teacher_patch_head.state_dict(),
                "optimizer": opt.state_dict(),
                "args": vars(args),
                "history": history,
            }
            torch.save(ckpt, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")

    torch.save({
        "student_backbone": student_backbone.state_dict(),
        "student_token_adapter": student_token_adapter.state_dict() if student_token_adapter is not None else None,
        "student_cls_head": student_cls_head.state_dict(),
        "student_patch_head": student_patch_head.state_dict(),
        "args": vars(args),
        "history": history,
    }, run_dir / "checkpoints" / "final.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved_final={run_dir / 'checkpoints' / 'final.pt'}")


if __name__ == "__main__":
    main()
