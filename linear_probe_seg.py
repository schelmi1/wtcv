#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from models import Stage1SegNet, load_stage1_state_dict_compat
from pretrain_dino_lora_ssl import Stage1UpscaleTokenAdapter
from wtcv_utils.records import load_labelme_records
from wtcv_utils.tiling import crop_with_pad, tile_origins


def module_param_stats(module: nn.Module) -> Dict[str, int]:
    total = int(sum(p.numel() for p in module.parameters()))
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return {
        "total": total,
        "trainable": trainable,
        "frozen": int(total - trainable),
    }


def print_param_line(name: str, stats: Dict[str, int]) -> None:
    print(
        f"  - {name:24s} trainable={stats['trainable']:,} "
        f"frozen={stats['frozen']:,} total={stats['total']:,}"
    )


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
        parent_name = full_name.rsplit('.', 1)[0] if '.' in full_name else ''
        child_name = full_name.split('.')[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        child = getattr(parent, child_name, None)
        if isinstance(child, nn.Linear):
            mod = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            mod = mod.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(parent, child_name, mod)
            replaced += 1
    return replaced


def _prepare_dino_scaled_input(x: torch.Tensor, patch: int = 14, scale_num: int = 14, scale_den: int = 16) -> torch.Tensor:
    h, w = int(x.shape[-2]), int(x.shape[-1])
    dh = int(round(float(h) * float(scale_num) / float(scale_den)))
    dw = int(round(float(w) * float(scale_num) / float(scale_den)))
    dh = max(int(patch), int((dh // int(patch)) * int(patch)))
    dw = max(int(patch), int((dw // int(patch)) * int(patch)))
    if (dh, dw) == (h, w):
        return x
    return F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)


def _infer_grid_from_tokens(n_tokens: int, h: int, w: int, patch: int = 14) -> Tuple[int, int]:
    gh = max(1, int(h // patch))
    gw = max(1, int(w // patch))
    if gh * gw == int(n_tokens):
        return gh, gw
    target_ar = float(w) / float(max(1, h))
    best = None
    nnn = int(n_tokens)
    for d in range(1, int(math.sqrt(float(nnn))) + 1):
        if (nnn % d) != 0:
            continue
        a, b = int(d), int(nnn // d)
        for hh, ww in ((a, b), (b, a)):
            ar = float(ww) / float(max(1, hh))
            score = abs(ar - target_ar)
            if best is None or score < best[0]:
                best = (score, hh, ww)
    if best is None:
        raise RuntimeError(f"Cannot infer token grid for N={n_tokens}")
    return int(best[1]), int(best[2])


class FeatureExtractorBase(nn.Module):
    out_channels: int

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VanillaOrSSLExtractor(FeatureExtractorBase):
    def __init__(self, backbone: nn.Module, adapter: Optional[Stage1UpscaleTokenAdapter]):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.patch_size = 14
        p0 = next(self.backbone.parameters())
        probe = torch.randn(1, 3, 224, 224, device=p0.device, dtype=p0.dtype)
        with torch.inference_mode():
            cdim = int(self.backbone.forward_features(probe)["x_norm_clstoken"].shape[-1])
        self.out_channels = cdim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_dino = _prepare_dino_scaled_input(x, patch=self.patch_size)
        feats = self.backbone.forward_features(x_dino)
        tok = feats["x_norm_patchtokens"]
        if self.adapter is not None:
            tok = self.adapter(tokens=tok, image=x, patch_size=self.patch_size)
        b, n, c = tok.shape
        gh, gw = _infer_grid_from_tokens(int(n), h=int(x_dino.shape[-2]), w=int(x_dino.shape[-1]), patch=self.patch_size)
        fmap = tok.transpose(1, 2).reshape(b, c, gh, gw).contiguous()
        return fmap


class Stage1FeatureExtractor(FeatureExtractorBase):
    def __init__(self, model: Stage1SegNet):
        super().__init__()
        self.model = model
        self.out_channels = int(self.model.fuse_1x1.out_channels) if self.model.fuse_1x1 is not None else 256

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, return_features=True)
        return out["feat_adapted"]


@dataclass
class LPConfig:
    data_dir: Path
    output_dir: Path
    run_name: str
    checkpoint: Optional[Path]
    label: str
    fp_label: str
    tile_size: int
    tile_stride: int
    min_poly_points: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    pred_threshold: float
    seed: int
    trust_torch_hub_repo: bool


def parse_args() -> LPConfig:
    ap = argparse.ArgumentParser(description="Linear probing segmentation on frozen features")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("runs"))
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional SSL or Stage1 checkpoint. Empty => vanilla DINO")
    ap.add_argument("--label", type=str, default="vehicle")
    ap.add_argument("--fp-label", type=str, default="")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--tile-stride", type=int, default=512)
    ap.add_argument("--min-poly-points", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pred-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")
    a = ap.parse_args()
    return LPConfig(
        data_dir=a.data_dir,
        output_dir=a.output_dir,
        run_name=a.run_name,
        checkpoint=a.checkpoint,
        label=a.label,
        fp_label=a.fp_label,
        tile_size=int(a.tile_size),
        tile_stride=int(a.tile_stride),
        min_poly_points=int(a.min_poly_points),
        batch_size=int(a.batch_size),
        num_workers=int(a.num_workers),
        epochs=int(a.epochs),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        pred_threshold=float(a.pred_threshold),
        seed=int(a.seed),
        trust_torch_hub_repo=bool(a.trust_torch_hub_repo),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tile_intersects_obj(obj: Dict, x0: int, y0: int, ts: int) -> bool:
    bx0, by0, bx1, by1 = obj["bbox_xyxy"]
    return not (bx1 <= x0 or by1 <= y0 or bx0 >= x0 + ts or by0 >= y0 + ts)


def _build_tile_mask(objects: List[Dict], x0: int, y0: int, ts: int) -> np.ndarray:
    m = Image.new("L", (ts, ts), 0)
    dr = ImageDraw.Draw(m)
    for o in objects:
        if bool(o.get("is_fp", False)):
            continue
        if not _tile_intersects_obj(o, x0, y0, ts):
            continue
        pts = o.get("points", []) or []
        if len(pts) >= 3:
            spts = [[float(p[0]) - float(x0), float(p[1]) - float(y0)] for p in pts]
            dr.polygon(spts, fill=1)
        else:
            bx0, by0, bx1, by1 = o["bbox_xyxy"]
            dr.rectangle([bx0 - x0, by0 - y0, bx1 - x0, by1 - y0], fill=1)
    return np.array(m, dtype=np.float32)


class LinearProbeTileDataset(Dataset):
    def __init__(self, records: List[Dict], tile_size: int, tile_stride: int):
        self.records = records
        self.tile_size = int(tile_size)
        self.items: List[Tuple[int, int, int]] = []
        for ridx, r in enumerate(records):
            for x0, y0 in tile_origins(r["width"], r["height"], self.tile_size, int(tile_stride)):
                self.items.append((ridx, int(x0), int(y0)))
        self.norm = torch.nn.Sequential()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ridx, x0, y0 = self.items[idx]
        r = self.records[ridx]
        img = Image.open(r["image_path"]).convert("RGB")
        tile = crop_with_pad(img, x0, y0, self.tile_size)
        x = TF.to_tensor(tile)
        x = TF.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        y = _build_tile_mask(r.get("objects", []), x0, y0, self.tile_size)
        y = torch.from_numpy(y).unsqueeze(0)
        return {"x": x, "y": y}


def _safe_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def build_feature_extractor(cfg: LPConfig, device: torch.device) -> Tuple[FeatureExtractorBase, Dict[str, object]]:
    if cfg.checkpoint is None or str(cfg.checkpoint).strip() == "":
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", trust_repo=cfg.trust_torch_hub_repo).to(device).eval()
        for p in backbone.parameters():
            p.requires_grad = False
        return VanillaOrSSLExtractor(backbone=backbone, adapter=None).to(device).eval(), {
            "feature_source": "vanilla_dino",
            "dino_model": "dinov2_vits14_reg",
            "load_summary": {
                "backbone_loaded_tensors": 0,
                "backbone_loaded_params": 0,
                "adapter_loaded_tensors": 0,
                "adapter_loaded_params": 0,
            },
        }

    ckpt = _safe_load(cfg.checkpoint, device)

    if isinstance(ckpt, dict) and ("student_backbone" in ckpt) and ("model" not in ckpt):
        args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
        pretrain_mode = str(args.get("pretrain_mode", "lora")).strip().lower()
        dino_model = str(args.get("dino_model", "dinov2_vits14_reg"))
        backbone = torch.hub.load("facebookresearch/dinov2", dino_model, trust_repo=cfg.trust_torch_hub_repo).to(device).eval()
        if pretrain_mode in {"lora", "lora_upscaling"}:
            rank = int(args.get("lora_rank", 8))
            alpha = float(args.get("lora_alpha", 16.0))
            dropout = float(args.get("lora_dropout", 0.0))
            targets = [s.strip() for s in str(args.get("lora_targets", "attn.qkv,attn.proj")).split(",") if s.strip()]
            _ = apply_lora(backbone, target_substrings=targets, rank=rank, alpha=alpha, dropout=dropout)
        bb_state = ckpt["student_backbone"]
        backbone.load_state_dict(bb_state, strict=True)
        bb_loaded_tensors = int(len(bb_state))
        bb_loaded_params = int(sum(int(v.numel()) for v in bb_state.values()))
        adapter = None
        ad_loaded_tensors = 0
        ad_loaded_params = 0
        if pretrain_mode in {"upscaling", "lora_upscaling"} and ckpt.get("student_token_adapter", None) is not None:
            probe = torch.randn(1, 3, 224, 224, device=device)
            with torch.inference_mode():
                cdim = int(backbone.forward_features(probe)["x_norm_clstoken"].shape[-1])
            adapter = Stage1UpscaleTokenAdapter(
                channels=cdim,
                upscale_type=str(args.get("upscale_type", "learned")),
                gate_init=float(args.get("upscale_gate_init", 0.0)),
                local_gain=float(args.get("upscale_local_gain", 1.0)),
                dino_gain=float(args.get("upscale_dino_gain", 1.0)),
                refine_gain=float(args.get("upscale_refine_gain", 1.0)),
                sharpen_gain=float(args.get("upscale_sharpen_gain", 0.0)),
                semantic_smooth_tau=float(args.get("semantic_smooth_tau", 0.2)),
            ).to(device)
            ad_state = ckpt["student_token_adapter"]
            ad_model_state = adapter.state_dict()
            for k, v in ad_state.items():
                if k in ad_model_state and tuple(v.shape) == tuple(ad_model_state[k].shape):
                    ad_loaded_tensors += 1
                    ad_loaded_params += int(v.numel())
            adapter.load_state_dict(ad_state, strict=False)
            adapter.eval()
            for p in adapter.parameters():
                p.requires_grad = False
        for p in backbone.parameters():
            p.requires_grad = False
        return VanillaOrSSLExtractor(backbone=backbone, adapter=adapter).to(device).eval(), {
            "feature_source": "ssl_checkpoint",
            "pretrain_mode": pretrain_mode,
            "dino_model": dino_model,
            "has_adapter": bool(adapter is not None),
            "load_summary": {
                "backbone_loaded_tensors": int(bb_loaded_tensors),
                "backbone_loaded_params": int(bb_loaded_params),
                "adapter_loaded_tensors": int(ad_loaded_tensors),
                "adapter_loaded_params": int(ad_loaded_params),
            },
        }

    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    fusion_channels = int(ckpt_cfg.get("fusion_channels", 256))
    dino_upsampler = str(ckpt_cfg.get("dino_upsampler_type", "learned"))
    dino_layers = str(ckpt_cfg.get("dino_layers", "last"))
    anyup_q_chunk_size = int(ckpt_cfg.get("anyup_q_chunk_size", 256))
    local_backbone = str(ckpt_cfg.get("local_backbone", "resnet18"))
    head_type = str(ckpt_cfg.get("head_type", "pointwise"))
    use_tile_cls_head = bool(ckpt_cfg.get("use_tile_cls_head", False))
    use_zoom_cls_head = bool(ckpt_cfg.get("use_zoom_cls_head", False))

    model = Stage1SegNet(
        channels=fusion_channels,
        trust_repo=cfg.trust_torch_hub_repo,
        dino_upsampler_type=dino_upsampler,
        dino_layers=dino_layers,
        anyup_q_chunk_size=anyup_q_chunk_size,
        local_backbone=local_backbone,
        head_type=head_type,
        use_tile_cls_head=use_tile_cls_head,
        use_zoom_cls_head=use_zoom_cls_head,
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    compat = load_stage1_state_dict_compat(model, state, strict=False, interpolate_mismatch=True, verbose=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model_state = model.state_dict()
    loaded_exact_tensors = 0
    loaded_exact_params = 0
    for k, v in state.items():
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
            loaded_exact_tensors += 1
            loaded_exact_params += int(v.numel())
    loaded_interp_tensors = int(len(compat.get("interpolated", [])))
    loaded_interp_params = 0
    for k, _src, _tgt in compat.get("interpolated", []):
        if k in model_state:
            loaded_interp_params += int(model_state[k].numel())
    return Stage1FeatureExtractor(model).to(device).eval(), {
        "feature_source": "stage1_checkpoint",
        "dino_upsampler": dino_upsampler,
        "dino_layers": dino_layers,
        "load_summary": {
            "stage1_loaded_exact_tensors": int(loaded_exact_tensors),
            "stage1_loaded_exact_params": int(loaded_exact_params),
            "stage1_loaded_interpolated_tensors": int(loaded_interp_tensors),
            "stage1_loaded_interpolated_params": int(loaded_interp_params),
            "stage1_missing_tensors": int(len(compat.get("missing_in_ckpt", []))),
            "stage1_unexpected_tensors": int(len(compat.get("unexpected_in_ckpt", []))),
            "stage1_skipped_mismatch_tensors": int(len(compat.get("skipped_mismatch", []))),
        },
    }


@torch.no_grad()
def evaluate(extractor: FeatureExtractorBase, probe_head: nn.Module, loader: DataLoader, device: torch.device, pred_thr: float) -> Dict[str, float]:
    probe_head.eval()
    inter = 0.0
    union = 0.0
    loss_sum = 0.0
    n = 0
    bce = nn.BCEWithLogitsLoss()
    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        f = extractor.forward_features(x)
        logit = probe_head(f)
        y_lr = F.interpolate(y, size=logit.shape[-2:], mode="nearest")
        loss = bce(logit, y_lr)
        prob = torch.sigmoid(logit)
        pred = (prob >= float(pred_thr)).float()
        t = (y_lr >= 0.5).float()
        inter += float((pred * t).sum().item())
        union += float(((pred + t) > 0).float().sum().item())
        loss_sum += float(loss.item())
        n += 1
        pbar.set_postfix(
            {
                "loss": f"{(loss_sum / max(1, n)):.4f}",
                "iou": f"{(inter / max(1e-6, union)):.4f}",
            }
        )
    return {
        "loss": float(loss_sum / max(1, n)),
        "iou": float(inter / max(1e-6, union)),
    }


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {cfg.data_dir}")
    if cfg.checkpoint is not None and (not cfg.checkpoint.exists()):
        raise FileNotFoundError(f"Missing checkpoint: {cfg.checkpoint}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = f"{stamp}-{cfg.run_name}" if cfg.run_name else f"{stamp}-linear-probe"
    run_dir = cfg.output_dir / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    records = load_labelme_records(
        cfg.data_dir,
        cfg.label,
        cfg.min_poly_points,
        include_fp=bool(str(cfg.fp_label).strip()),
        fp_label=cfg.fp_label,
        load_workers=cfg.num_workers,
    )
    if len(records) == 0:
        raise RuntimeError("No records loaded")

    rng = np.random.default_rng(cfg.seed)
    order = np.arange(len(records))
    rng.shuffle(order)
    records = [records[i] for i in order]

    split = int(0.95 * len(records))
    train_records = records[:split]
    val_records = records[split:]

    train_ds = LinearProbeTileDataset(train_records, cfg.tile_size, cfg.tile_stride)
    val_ds = LinearProbeTileDataset(val_records, cfg.tile_size, cfg.tile_stride)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    extractor, info = build_feature_extractor(cfg, device)
    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()

    probe_head = nn.Conv2d(int(extractor.out_channels), 1, kernel_size=1).to(device)
    opt = torch.optim.AdamW(probe_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    print("linear_probe_setup:")
    print(f"  records_train={len(train_records)} records_val={len(val_records)}")
    print(f"  tiles_train={len(train_ds)} tiles_val={len(val_ds)}")
    print(f"  feature_source={info.get('feature_source')}")
    print(f"  feature_channels={extractor.out_channels}")
    if cfg.checkpoint is not None:
        print(f"  checkpoint={cfg.checkpoint}")
    for k, v in info.items():
        if k == "feature_source":
            continue
        print(f"  {k}={v}")

    if isinstance(info.get("load_summary", None), dict):
        print("module_load_summary:")
        for k, v in info["load_summary"].items():
            print(f"  {k}={v}")

    ext_stats = module_param_stats(extractor)
    probe_stats = module_param_stats(probe_head)
    print("module_param_breakdown:")
    print_param_line("feature_extractor(total)", ext_stats)
    if isinstance(extractor, VanillaOrSSLExtractor):
        print_param_line("feature_extractor.backbone", module_param_stats(extractor.backbone))
        if extractor.adapter is not None:
            print_param_line("feature_extractor.adapter", module_param_stats(extractor.adapter))
        else:
            print("  - feature_extractor.adapter   trainable=0 frozen=0 total=0")
    elif isinstance(extractor, Stage1FeatureExtractor):
        m = extractor.model
        for mod_name, mod in m.named_children():
            print_param_line(f"feature_extractor.{mod_name}", module_param_stats(mod))
    print_param_line("probe_head", probe_stats)

    history = []
    best_iou = -1.0
    best_path = run_dir / "linear_probe_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        probe_head.train()
        loss_sum = 0.0
        inter = 0.0
        union = 0.0
        n = 0
        train_pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in train_pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            with torch.inference_mode():
                f = extractor.forward_features(x)
            # Features produced in inference_mode are special tensors that cannot
            # be captured by autograd in downstream trainable ops; convert them.
            f = f.detach().clone()
            logit = probe_head(f)
            y_lr = F.interpolate(y, size=logit.shape[-2:], mode="nearest")
            loss = bce(logit, y_lr)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                prob = torch.sigmoid(logit)
                pred = (prob >= float(cfg.pred_threshold)).float()
                t = (y_lr >= 0.5).float()
                inter += float((pred * t).sum().item())
                union += float(((pred + t) > 0).float().sum().item())
            loss_sum += float(loss.item())
            n += 1
            train_pbar.set_postfix(
                {
                    "loss": f"{(loss_sum / max(1, n)):.4f}",
                    "iou": f"{(inter / max(1e-6, union)):.4f}",
                    "lr": f"{opt.param_groups[0]['lr']:.2e}",
                }
            )

        train_loss = float(loss_sum / max(1, n))
        train_iou = float(inter / max(1e-6, union))
        val_stats = evaluate(extractor, probe_head, val_loader, device, cfg.pred_threshold)

        row = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": float(val_stats["loss"]),
            "val_iou": float(val_stats["iou"]),
        }
        history.append(row)
        (run_dir / f"stats_epoch_{epoch:03d}.json").write_text(
            json.dumps(
                {
                    "epoch": int(epoch),
                    "metrics": row,
                    "best_val_iou_so_far": float(max(best_iou, row["val_iou"])),
                    "feature_info": info,
                    "run_dir": str(run_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"epoch {epoch:03d}/{cfg.epochs:03d} "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"val_loss={row['val_loss']:.4f} val_iou={row['val_iou']:.4f}"
        )

        if row["val_iou"] > best_iou:
            best_iou = row["val_iou"]
            torch.save(
                {
                    "probe_head": probe_head.state_dict(),
                    "config": asdict(cfg),
                    "feature_info": info,
                    "best_val_iou": float(best_iou),
                    "epoch": int(epoch),
                },
                best_path,
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
                "feature_info": info,
                "best_val_iou": float(best_iou),
                "best_checkpoint": str(best_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done: run_dir={run_dir}")
    print(f"best_val_iou={best_iou:.4f}")
    print(f"best_checkpoint={best_path}")


if __name__ == "__main__":
    main()
