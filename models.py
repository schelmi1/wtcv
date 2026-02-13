#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FrozenDinoTokenBranch(nn.Module):
    def __init__(self, out_channels: int = 256, trust_repo: bool = True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=trust_repo
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.proj = nn.Conv2d(384, out_channels, kernel_size=1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        feats = self.backbone.forward_features(x)
        tokens = feats["x_norm_patchtokens"]
        b, n, c = tokens.shape
        h = w = int(math.sqrt(n))
        fmap = tokens.transpose(1, 2).reshape(b, c, h, w)
        return self.proj(fmap)


class DinoLearnedUpsampler(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.act1(self.conv1(x))
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.act2(self.conv2(x))
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x


class AnyUpFeatureUpsampler(nn.Module):
    def __init__(self, q_chunk_size: int = 256, trust_repo: bool = True):
        super().__init__()
        self.q_chunk_size = q_chunk_size
        self.upsampler = torch.hub.load(
            "wimmerth/anyup",
            "anyup",
            verbose=False,
            trust_repo=trust_repo,
        )
        self.upsampler.eval()
        for p in self.upsampler.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, hr_image: torch.Tensor, lr_features: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        self.upsampler.eval()
        x = self.upsampler(hr_image, lr_features, q_chunk_size=self.q_chunk_size)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x


class ResNet18LocalBranch(nn.Module):
    def __init__(self, out_channels: int = 256):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1 = m.layer1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.l1(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.seg_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"seg_logit": self.seg_head(x)}


class AnyUpPointwiseSegHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        mid = max(1, in_channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"seg_logit": self.net(x)}


class AnyUpDwSepSegHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        mid = max(1, in_channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid),
            nn.Conv2d(mid, mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid),
            nn.Conv2d(mid, mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"seg_logit": self.net(x)}


class AnyUpResidualPointwiseSegHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        mid = max(1, in_channels // 2)
        self.in_proj = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=1),
        )
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(mid, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.act(self.in_proj(x))
        x = self.act(x + self.block1(x))
        x = self.act(x + self.block2(x))
        return {"seg_logit": self.out_proj(x)}


class TileClassifierHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        hid = max(8, in_channels // 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(x).flatten(1)
        z = self.act(self.fc1(z))
        return self.fc2(z)


class ZoomRoiClassifierHead(nn.Module):
    def __init__(self, in_channels: int = 256, roi_size: int = 8):
        super().__init__()
        self.roi_size = int(roi_size)
        hid = max(16, in_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hid, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hid, 1)

    def forward(self, feat: torch.Tensor, zoom_boxes: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
        b, _c, hf, wf = feat.shape
        in_h, in_w = int(input_hw[0]), int(input_hw[1])
        z = zoom_boxes.float()
        if z.ndim == 3 and z.shape[1] == 1:
            z = z[:, 0, :]
        if z.ndim != 2 or z.shape[1] != 4:
            raise ValueError(f"zoom_boxes must be (B,4), got {tuple(z.shape)}")
        if z.shape[0] != b:
            raise ValueError(f"zoom_boxes batch {z.shape[0]} != feat batch {b}")

        x0 = z[:, 0].clamp(0.0, max(0.0, float(in_w - 1)))
        y0 = z[:, 1].clamp(0.0, max(0.0, float(in_h - 1)))
        x1 = z[:, 2].clamp(1.0, float(in_w))
        y1 = z[:, 3].clamp(1.0, float(in_h))
        x1 = torch.maximum(x1, x0 + 1.0)
        y1 = torch.maximum(y1, y0 + 1.0)

        sx = float(wf) / float(max(1, in_w))
        sy = float(hf) / float(max(1, in_h))
        rois = torch.zeros((b, 5), dtype=torch.float32, device=feat.device)
        rois[:, 0] = torch.arange(0, b, device=feat.device, dtype=torch.float32)
        rois[:, 1] = x0 * sx
        rois[:, 2] = y0 * sy
        rois[:, 3] = x1 * sx
        rois[:, 4] = y1 * sy

        pooled = torchvision.ops.roi_align(
            feat,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            aligned=True,
        )
        zf = self.conv(pooled)
        zf = self.pool(zf).flatten(1)
        return self.fc(zf)


class Stage1SegNet(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        trust_repo: bool = True,
        dino_upsampler_type: str = "learned",
        anyup_q_chunk_size: int = 256,
        head_type: str = "pointwise",
        use_tile_cls_head: bool = False,
        use_zoom_cls_head: bool = False,
    ):
        super().__init__()
        if dino_upsampler_type not in {"learned", "anyup"}:
            raise ValueError(f"Unsupported dino_upsampler_type={dino_upsampler_type}")
        if head_type not in {"pointwise", "dwsep", "residual"}:
            raise ValueError(f"Unsupported head_type={head_type}")
        self.dino_upsampler_type = dino_upsampler_type
        self.head_type = head_type
        self.use_tile_cls_head = bool(use_tile_cls_head)
        self.use_zoom_cls_head = bool(use_zoom_cls_head)
        self.dino = FrozenDinoTokenBranch(channels, trust_repo=trust_repo)
        self.dino_up = DinoLearnedUpsampler(channels)
        self.dino_anyup: Optional[AnyUpFeatureUpsampler] = None
        if self.dino_upsampler_type == "anyup":
            self.dino_anyup = AnyUpFeatureUpsampler(
                q_chunk_size=anyup_q_chunk_size,
                trust_repo=trust_repo,
            )
        if self.dino_upsampler_type == "anyup":
            self.local = None
            self.fuse_1x1 = None
            if self.head_type == "pointwise":
                self.anyup_head = AnyUpPointwiseSegHead(channels)
            elif self.head_type == "dwsep":
                self.anyup_head = AnyUpDwSepSegHead(channels)
            else:
                self.anyup_head = AnyUpResidualPointwiseSegHead(channels)
            self.head = None
            self.tile_cls_head = TileClassifierHead(channels) if self.use_tile_cls_head else None
            self.zoom_cls_head = ZoomRoiClassifierHead(channels) if self.use_zoom_cls_head else None
        else:
            self.local = ResNet18LocalBranch(channels)
            self.fuse_1x1 = nn.Conv2d(channels + 64, channels, kernel_size=1)
            self.head = SegmentationHead(channels)
            self.anyup_head = None
            self.tile_cls_head = TileClassifierHead(channels) if self.use_tile_cls_head else None
            self.zoom_cls_head = ZoomRoiClassifierHead(channels) if self.use_zoom_cls_head else None

    def forward(self, x: torch.Tensor, zoom_boxes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        fdino = self.dino(x)
        if self.dino_upsampler_type == "anyup" and self.dino_anyup is not None:
            fdino_up = self.dino_anyup(x, fdino, target_hw=x.shape[-2:])
            out = self.anyup_head(fdino_up)
            cls_feat = fdino_up
        else:
            flocal = self.local(x)
            fdino_up = self.dino_up(fdino, target_hw=flocal.shape[-2:])
            fused = torch.cat([fdino_up, flocal], dim=1)
            fused = self.fuse_1x1(fused)
            out = self.head(fused)
            cls_feat = fused

        target_hw = (x.shape[-2] // 4, x.shape[-1] // 4)
        out["seg_logit"] = F.interpolate(out["seg_logit"], size=target_hw, mode="bilinear", align_corners=False)
        if self.tile_cls_head is not None:
            out["tile_logit"] = self.tile_cls_head(cls_feat)
        if self.zoom_cls_head is not None and zoom_boxes is not None:
            out["zoom_logit"] = self.zoom_cls_head(
                cls_feat,
                zoom_boxes=zoom_boxes,
                input_hw=(x.shape[-2], x.shape[-1]),
            )
        return out

