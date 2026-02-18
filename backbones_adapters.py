#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FrozenDinoTokenBranch(nn.Module):
    @staticmethod
    def _parse_layer_spec(
        spec: Union[str, Sequence[int], None],
        depth: int,
    ) -> Tuple[int, ...]:
        if spec is None:
            return (depth - 1,)
        if isinstance(spec, (list, tuple)):
            vals = [int(v) for v in spec]
        else:
            s = str(spec).strip().lower()
            if s in {"", "last", "final"}:
                return (depth - 1,)
            vals = []
            for part in s.split(","):
                p = part.strip()
                if not p:
                    continue
                vals.append(int(p))
        if len(vals) == 0:
            return (depth - 1,)

        out = []
        for v in vals:
            idx = v - 1 if v >= 1 else v
            if idx < 0 or idx >= depth:
                raise ValueError(f"Invalid dino layer '{v}' for depth={depth}. Expected 1..{depth}.")
            out.append(int(idx))
        return tuple(out)

    def __init__(
        self,
        out_channels: int = 256,
        trust_repo: bool = True,
        model_name: str = "dinov2_vits14_reg",
        layer_spec: Union[str, Sequence[int], None] = "last",
        layer_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", str(model_name), trust_repo=trust_repo
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.depth = int(len(self.backbone.blocks))
        self.model_name = str(model_name)
        if layer_indices is None:
            self.layer_indices = self._parse_layer_spec(layer_spec, depth=self.depth)
        else:
            idx = [int(v) for v in layer_indices]
            if len(idx) == 0:
                raise ValueError("layer_indices must contain at least one layer index")
            for v in idx:
                if v < 0 or v >= self.depth:
                    raise ValueError(f"layer index {v} out of range [0,{self.depth - 1}]")
            self.layer_indices = tuple(idx)
        in_channels = 384 * int(len(self.layer_indices))
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        h = w = int(math.sqrt(n))
        if h * w != n:
            raise ValueError(f"Non-square token grid; got N={n}")
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        if len(self.layer_indices) == 1 and self.layer_indices[0] == (self.depth - 1):
            feats = self.backbone.forward_features(x)
            tokens = feats["x_norm_patchtokens"]
            fmap = self._tokens_to_map(tokens)
        else:
            inter = self.backbone.get_intermediate_layers(
                x,
                n=list(self.layer_indices),
                reshape=False,
                return_class_token=False,
                norm=True,
            )
            maps = [self._tokens_to_map(t) for t in inter]
            fmap = torch.cat(maps, dim=1)
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


class DinoPixelShuffleUpsampler(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.pre1 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.act1 = nn.GELU()
        self.ref1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.pre2 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.ps2 = nn.PixelShuffle(2)
        self.act2 = nn.GELU()
        self.ref2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.ps1(self.pre1(x))
        x = self.act1(self.ref1(x))
        x = self.ps2(self.pre2(x))
        x = self.act2(self.ref2(x))
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


class ResNetLocalBranch(nn.Module):
    def __init__(self, backbone: str = "resnet18", out_channels: int = 256):
        super().__init__()
        _ = out_channels  # kept for interface compatibility
        bb = str(backbone).strip().lower()
        if bb == "resnet18":
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            l1_out = 64
        elif bb == "resnet34":
            m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
            l1_out = 64
        elif bb == "resnet50":
            m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            l1_out = 256
        else:
            raise ValueError(f"Unsupported local backbone: {backbone}. Use one of: resnet18, resnet34, resnet50")

        self.backbone_name = bb
        self.out_channels = int(l1_out)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1 = m.layer1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.l1(x)
        return x


class ResNet18LocalBranch(ResNetLocalBranch):
    # Backward-compatible wrapper for existing imports/call-sites.
    def __init__(self, out_channels: int = 256):
        super().__init__(backbone="resnet18", out_channels=out_channels)


__all__ = [
    "FrozenDinoTokenBranch",
    "DinoLearnedUpsampler",
    "DinoPixelShuffleUpsampler",
    "AnyUpFeatureUpsampler",
    "ResNetLocalBranch",
    "ResNet18LocalBranch",
]
