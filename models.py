#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones_adapters import (
    AnyUpFeatureUpsampler,
    DinoLearnedUpsampler,
    DinoPixelShuffleUpsampler,
    FrozenDinoTokenBranch,
    ResNetLocalBranch,
)
from heads import (
    AnyUpDwSepSegHead,
    AnyUpPointwiseSegHead,
    AnyUpResidualPointwiseSegHead,
    SegmentationHead,
    TileClassifierHead,
    ZoomRoiClassifierHead,
)


def parse_dino_layers_spec(
    spec: Union[str, Sequence[int], None],
    depth: int = 12,
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

    out: List[int] = []
    for v in vals:
        # User-facing format is 1-based layer ids (e.g. 6,9,12).
        idx = v - 1 if v >= 1 else v
        if idx < 0 or idx >= depth:
            raise ValueError(f"Invalid dino layer '{v}' for depth={depth}. Expected 1..{depth}.")
        out.append(int(idx))
    return tuple(out)


def _interp_1d(src: torch.Tensor, out_len: int) -> torch.Tensor:
    x = src.reshape(1, 1, -1)
    y = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
    return y.reshape(out_len)


def _interp_2d(src: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    x = src.reshape(1, 1, int(src.shape[0]), int(src.shape[1]))
    y = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return y.reshape(out_hw[0], out_hw[1])


def _maybe_interpolate_tensor(src: torch.Tensor, target_shape: torch.Size) -> Optional[torch.Tensor]:
    tgt = tuple(int(v) for v in target_shape)
    if tuple(int(v) for v in src.shape) == tgt:
        return src

    # 1D vectors (biases, affine parameters): resize length.
    if src.ndim == 1 and len(tgt) == 1:
        return _interp_1d(src, tgt[0])

    # 2D matrices (linear weights): resize on whichever/both axes changed.
    if src.ndim == 2 and len(tgt) == 2:
        return _interp_2d(src, (tgt[0], tgt[1]))

    # 3D tensors (e.g. [1, N, C] style embeddings): interpolate sequence dim if C matches.
    if src.ndim == 3 and len(tgt) == 3:
        if src.shape[0] == tgt[0] and src.shape[2] == tgt[2]:
            x = src.transpose(1, 2)  # [B, C, N]
            y = F.interpolate(x, size=tgt[1], mode="linear", align_corners=False)
            return y.transpose(1, 2)
        return None

    # 4D tensors (conv kernels/feature maps): interpolate only spatial dims when channels match.
    if src.ndim == 4 and len(tgt) == 4:
        if src.shape[0] == tgt[0] and src.shape[1] == tgt[1]:
            b = int(src.shape[0] * src.shape[1])
            x = src.reshape(b, 1, int(src.shape[2]), int(src.shape[3]))
            y = F.interpolate(x, size=(tgt[2], tgt[3]), mode="bilinear", align_corners=False)
            return y.reshape(tgt[0], tgt[1], tgt[2], tgt[3])
        return None

    return None


def load_stage1_state_dict_compat(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
    interpolate_mismatch: bool = True,
    verbose: bool = True,
) -> Dict[str, List]:
    model_state = model.state_dict()
    adapted: Dict[str, torch.Tensor] = {}

    missing_in_ckpt: List[str] = []
    unexpected_in_ckpt: List[str] = []
    interpolated: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    skipped_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for k in state_dict.keys():
        if k not in model_state:
            unexpected_in_ckpt.append(k)

    for k, tgt in model_state.items():
        if k not in state_dict:
            missing_in_ckpt.append(k)
            continue

        src = state_dict[k]
        if isinstance(src, nn.Parameter):
            src = src.detach()

        if tuple(src.shape) == tuple(tgt.shape):
            adapted[k] = src
            continue

        if interpolate_mismatch:
            resized = _maybe_interpolate_tensor(src, tgt.shape)
            if resized is not None:
                adapted[k] = resized.to(dtype=tgt.dtype)
                interpolated.append((k, tuple(src.shape), tuple(tgt.shape)))
                continue

        skipped_mismatch.append((k, tuple(src.shape), tuple(tgt.shape)))

    model.load_state_dict(adapted, strict=False)

    if verbose:
        if interpolated:
            print("checkpoint compat: interpolated tensors:")
            for k, src_shape, tgt_shape in interpolated:
                print(f"  - {k}: {src_shape} -> {tgt_shape}")
        if skipped_mismatch:
            print("checkpoint compat: skipped mismatched tensors:")
            for k, src_shape, tgt_shape in skipped_mismatch:
                print(f"  - {k}: {src_shape} != {tgt_shape}")
        if missing_in_ckpt:
            print(f"checkpoint compat: missing tensors in checkpoint={len(missing_in_ckpt)}")
        if unexpected_in_ckpt:
            print(f"checkpoint compat: unexpected tensors in checkpoint={len(unexpected_in_ckpt)}")

    if strict and (missing_in_ckpt or unexpected_in_ckpt or skipped_mismatch):
        raise RuntimeError(
            "Strict compatible checkpoint load failed with missing/unexpected/mismatched tensors. "
            f"missing={len(missing_in_ckpt)} unexpected={len(unexpected_in_ckpt)} "
            f"skipped_mismatch={len(skipped_mismatch)}"
        )

    return {
        "missing_in_ckpt": missing_in_ckpt,
        "unexpected_in_ckpt": unexpected_in_ckpt,
        "interpolated": interpolated,
        "skipped_mismatch": skipped_mismatch,
    }


class Stage1SegNet(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        trust_repo: bool = True,
        dino_upsampler_type: str = "learned",
        anyup_q_chunk_size: int = 256,
        local_backbone: str = "resnet18",
        head_type: str = "pointwise",
        dino_model_name: str = "dinov2_vits14_reg",
        dino_layers: Union[str, Sequence[int], None] = "last",
        use_tile_cls_head: bool = False,
        use_zoom_cls_head: bool = False,
    ):
        super().__init__()
        if dino_upsampler_type not in {"learned", "pixelshuffle", "anyup"}:
            raise ValueError(f"Unsupported dino_upsampler_type={dino_upsampler_type}")
        if str(local_backbone).strip().lower() not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"Unsupported local_backbone={local_backbone}")
        if head_type not in {"pointwise", "dwsep", "residual"}:
            raise ValueError(f"Unsupported head_type={head_type}")
        self.dino_upsampler_type = dino_upsampler_type
        self.local_backbone = str(local_backbone).strip().lower()
        self.head_type = head_type
        self.dino_model_name = str(dino_model_name)
        self.dino_layers = dino_layers
        self.use_tile_cls_head = bool(use_tile_cls_head)
        self.use_zoom_cls_head = bool(use_zoom_cls_head)
        # DINOv2 uses patch size 14. We feed a 14/16 downscaled view so that
        # for 256k inputs we get token grids of 16k (power-of-two).
        self.dino_input_scale = 14.0 / 16.0

        self.dino = FrozenDinoTokenBranch(
            channels,
            trust_repo=trust_repo,
            model_name=self.dino_model_name,
            layer_spec=dino_layers,
        )
        if self.dino_upsampler_type == "pixelshuffle":
            self.dino_up = DinoPixelShuffleUpsampler(channels)
        else:
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
            self.local = ResNetLocalBranch(backbone=self.local_backbone, out_channels=channels)
            self.fuse_1x1 = nn.Conv2d(channels + int(self.local.out_channels), channels, kernel_size=1)
            if self.head_type == "pointwise":
                self.head = SegmentationHead(channels)
            elif self.head_type == "dwsep":
                self.head = AnyUpDwSepSegHead(channels)
            else:
                self.head = AnyUpResidualPointwiseSegHead(channels)
            self.anyup_head = None
            self.tile_cls_head = TileClassifierHead(channels) if self.use_tile_cls_head else None
            self.zoom_cls_head = ZoomRoiClassifierHead(channels) if self.use_zoom_cls_head else None

    def _prepare_dino_input(self, x: torch.Tensor) -> torch.Tensor:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if (h % 256) != 0 or (w % 256) != 0:
            raise ValueError(
                f"Stage1SegNet expects input H/W to be multiples of 256, got {(h, w)}. "
                "Use tile-size 256 or multiples (e.g. 512)."
            )

        dino_h = int(round(h * self.dino_input_scale))
        dino_w = int(round(w * self.dino_input_scale))
        if (dino_h % 14) != 0 or (dino_w % 14) != 0:
            raise ValueError(
                f"DINO input {(dino_h, dino_w)} must be divisible by patch size 14."
            )
        if dino_h == h and dino_w == w:
            return x
        return F.interpolate(x, size=(dino_h, dino_w), mode="bilinear", align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        zoom_boxes: Optional[torch.Tensor] = None,
        trace_shapes: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        def _trace(name: str, t: Optional[torch.Tensor]) -> None:
            if not trace_shapes:
                return
            if t is None:
                print(f"[shape] {name}: None")
                return
            print(f"[shape] {name}: {tuple(t.shape)}")

        _trace("input", x)
        x_dino = self._prepare_dino_input(x)
        _trace("dino_input_scaled", x_dino)
        fdino = self.dino(x_dino)
        _trace("dino_tokens_proj", fdino)
        if self.dino_upsampler_type == "anyup" and self.dino_anyup is not None:
            fdino_up = self.dino_anyup(x, fdino, target_hw=x.shape[-2:])
            _trace("anyup_features", fdino_up)
            out = self.anyup_head(fdino_up)
            _trace("head_seg_logit_raw", out.get("seg_logit", None))
            cls_feat = fdino_up
        else:
            flocal = self.local(x)
            _trace("resnet_l1", flocal)
            fdino_up = self.dino_up(fdino, target_hw=flocal.shape[-2:])
            _trace("dino_up_features", fdino_up)
            fused = torch.cat([fdino_up, flocal], dim=1)
            _trace("fuse_concat", fused)
            fused = self.fuse_1x1(fused)
            _trace("fuse_1x1", fused)
            out = self.head(fused)
            _trace("head_seg_logit_raw", out.get("seg_logit", None))
            cls_feat = fused

        target_hw = (x.shape[-2] // 4, x.shape[-1] // 4)
        out["seg_logit"] = F.interpolate(out["seg_logit"], size=target_hw, mode="bilinear", align_corners=False)
        _trace("seg_logit_out", out["seg_logit"])
        if self.tile_cls_head is not None:
            out["tile_logit"] = self.tile_cls_head(cls_feat)
            _trace("tile_logit", out["tile_logit"])
        if self.zoom_cls_head is not None and zoom_boxes is not None:
            _trace("zoom_boxes", zoom_boxes)
            out["zoom_logit"] = self.zoom_cls_head(
                cls_feat,
                zoom_boxes=zoom_boxes,
                input_hw=(x.shape[-2], x.shape[-1]),
            )
            _trace("zoom_logit", out["zoom_logit"])
        if return_features:
            # Auxiliary features for advanced training objectives.
            out["feat_dino"] = fdino
            out["feat_adapted"] = cls_feat
        return out


__all__ = ["Stage1SegNet", "load_stage1_state_dict_compat"]
