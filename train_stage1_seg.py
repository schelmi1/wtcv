#!/usr/bin/env python3
import argparse
import json
import math
import random
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.transforms import functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


@dataclass
class Cfg:
    data_dir: Path
    output_dir: Path
    run_name: str

    tile_size: int = 224
    tile_stride: int = 112

    label_name: str = "vehicle"
    min_poly_points: int = 3

    seg_out_stride: int = 4

    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 5
    lr: float = 2e-4
    weight_decay: float = 1e-4

    fusion_channels: int = 256
    dino_upsampler_type: str = "learned"  # learned | anyup
    anyup_q_chunk_size: int = 256

    balance_train_50_50: bool = True
    balance_val_50_50: bool = False

    seed: int = 42
    subset_size: int = 0  # 0 means use all records

    val_interval: int = 1
    train_example_items: int = 3
    val_example_items: int = 3
    image_log_interval: int = 1

    iou_threshold: float = 0.5

    trust_torch_hub_repo: bool = True


def _yaml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def write_hparams_yaml(path: Path, cfg: Cfg) -> None:
    d = asdict(cfg)
    lines = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, Path):
            v = str(v)
        lines.append(f"{k}: {_yaml_scalar(v)}")
    path.write_text("\n".join(lines) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polygon_area(points: List[List[float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points], dtype=np.float32)
    y = np.array([p[1] for p in points], dtype=np.float32)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_center(points: List[List[float]]) -> Tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.mean(xs)), float(np.mean(ys))


def load_records(data_dir: Path, label_name: str, min_poly_points: int) -> List[Dict]:
    allowed_exts = (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff")
    records = []
    for jf in sorted(data_dir.glob("*.json")):
        # Resolve image file by shared stem across common extensions.
        candidates = []
        for p in data_dir.glob(f"{jf.stem}.*"):
            if p.is_file() and p.suffix.lower() in allowed_exts:
                candidates.append(p)
        if not candidates:
            continue
        # Prefer a stable extension order if multiple files exist for one stem.
        pref = {ext: i for i, ext in enumerate(allowed_exts)}
        candidates.sort(key=lambda p: pref.get(p.suffix.lower(), 999))
        img_path = candidates[0]

        d = json.loads(jf.read_text())
        w = int(d.get("imageWidth"))
        h = int(d.get("imageHeight"))
        shapes = d.get("shapes", []) or []

        objects = []
        for s in shapes:
            if s.get("label") != label_name:
                continue
            pts = s.get("points") or []
            stype = str(s.get("shape_type", "")).lower()
            if stype == "rectangle":
                if len(pts) < 2:
                    continue
            else:
                if len(pts) < min_poly_points:
                    continue
            x0, y0, x1, y1 = polygon_bbox(pts)
            cx, cy = polygon_center(pts)
            objects.append(
                {
                    "points": pts,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "center_xy": [cx, cy],
                    "poly_area": polygon_area(pts),
                }
            )

        records.append(
            {
                "image_path": str(img_path),
                "json_path": str(jf),
                "width": w,
                "height": h,
                "objects": objects,
            }
        )

    return records


def discover_labels(data_dir: Path) -> List[str]:
    labels = set()
    for jf in data_dir.glob("*.json"):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        for s in d.get("shapes", []) or []:
            lab = s.get("label")
            if isinstance(lab, str) and lab.strip():
                labels.add(lab.strip())
    return sorted(labels)


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


def objects_in_tile(objs: List[Dict], x0: int, y0: int, size: int) -> List[Dict]:
    x1, y1 = x0 + size, y0 + size
    out = []
    for o in objs:
        bx0, by0, bx1, by1 = o["bbox_xyxy"]
        # Keep only objects whose bbox is fully contained in the tile.
        fully_within = (bx0 >= x0) and (by0 >= y0) and (bx1 <= x1) and (by1 <= y1)
        if not fully_within:
            continue
        # Keep shifted bbox only (triplet pipeline uses bbox->mask).
        sbx0 = float(max(0.0, min(size, bx0 - x0)))
        sby0 = float(max(0.0, min(size, by0 - y0)))
        sbx1 = float(max(0.0, min(size, bx1 - x0)))
        sby1 = float(max(0.0, min(size, by1 - y0)))
        if sbx1 <= sbx0 or sby1 <= sby0:
            continue
        out.append(
            {
                "bbox_xyxy": [sbx0, sby0, sbx1, sby1],
                "poly_area": o["poly_area"],
            }
        )
    return out


def build_seg_target(tile_size: int, objs: List[Dict], out_stride: int) -> np.ndarray:
    full = Image.new("L", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(full)
    for o in objs:
        bx0, by0, bx1, by1 = o.get("bbox_xyxy", [0, 0, 0, 0])
        if bx1 > bx0 and by1 > by0:
            draw.rectangle([bx0, by0, bx1, by1], fill=1)

    full_np = np.array(full, dtype=np.float32)
    t = torch.from_numpy(full_np).unsqueeze(0).unsqueeze(0)
    # Preserve tiny positives: if any pixel in a stride block is positive,
    # the output cell stays positive.
    t = F.max_pool2d(t, kernel_size=out_stride, stride=out_stride)
    return t[0, 0].numpy().astype(np.float32)


class SegTileDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        tile_size: int,
        stride: int,
        seg_out_stride: int,
        seed: int,
        balance_50_50: bool,
        dataset_name: str = "dataset",
    ):
        self.records = records
        self.tile_size = tile_size
        self.stride = stride
        self.seg_out_stride = seg_out_stride
        self.balance_50_50 = balance_50_50

        # Triplets: (ridx, x0, y0, objs_in_tile, is_object)
        all_triplets = []
        obj_triplets = []
        non_obj_triplets = []

        for ridx, r in tqdm(
            enumerate(records),
            total=len(records),
            desc=f"Building {dataset_name} tile index",
            leave=False,
        ):
            for x0, y0 in tile_origins(r["width"], r["height"], tile_size, stride):
                objs = objects_in_tile(r["objects"], x0, y0, tile_size)
                item = {
                    "ridx": ridx,
                    "x0": x0,
                    "y0": y0,
                    "objs": objs,
                    "is_object": int(len(objs) > 0),
                }
                all_triplets.append(item)
                if item["is_object"] == 1:
                    obj_triplets.append(item)
                else:
                    non_obj_triplets.append(item)

        if balance_50_50 and len(obj_triplets) > 0 and len(non_obj_triplets) > 0:
            n = min(len(obj_triplets), len(non_obj_triplets))
            rng = np.random.default_rng(seed)
            pos_sel = rng.choice(len(obj_triplets), size=n, replace=False)
            neg_sel = rng.choice(len(non_obj_triplets), size=n, replace=False)
            self.samples = [obj_triplets[i] for i in pos_sel] + [non_obj_triplets[i] for i in neg_sel]
            rng.shuffle(self.samples)
        else:
            self.samples = all_triplets

        # Dataset-level positive/negative index lists (relative to self.samples).
        self.pos_dataset_indices = [i for i, s in enumerate(self.samples) if s["is_object"] == 1]
        self.neg_dataset_indices = [i for i, s in enumerate(self.samples) if s["is_object"] == 0]

        self.normalize = torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        ridx = item["ridx"]
        x0 = item["x0"]
        y0 = item["y0"]
        r = self.records[ridx]

        img = Image.open(r["image_path"]).convert("RGB")
        tile = crop_with_pad(img, x0, y0, self.tile_size)

        seg_t = build_seg_target(self.tile_size, item["objs"], self.seg_out_stride)

        x = TF.to_tensor(tile)
        x = self.normalize(x)

        return {
            "image": x,
            "seg_target": torch.from_numpy(seg_t).unsqueeze(0),
            "meta": {
                "image_path": r["image_path"],
                "tile_origin": (x0, y0),
                "num_objects": len(item["objs"]),
                "is_positive_tile": int(item["is_object"]),
            },
        }


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


class Stage1SegNet(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        trust_repo: bool = True,
        dino_upsampler_type: str = "learned",
        anyup_q_chunk_size: int = 256,
    ):
        super().__init__()
        if dino_upsampler_type not in {"learned", "anyup"}:
            raise ValueError(f"Unsupported dino_upsampler_type={dino_upsampler_type}")
        self.dino_upsampler_type = dino_upsampler_type
        self.dino = FrozenDinoTokenBranch(channels, trust_repo=trust_repo)
        self.dino_up = DinoLearnedUpsampler(channels)
        self.dino_anyup: Optional[AnyUpFeatureUpsampler] = None
        if self.dino_upsampler_type == "anyup":
            self.dino_anyup = AnyUpFeatureUpsampler(
                q_chunk_size=anyup_q_chunk_size,
                trust_repo=trust_repo,
            )
        self.local = ResNet18LocalBranch(channels)
        self.fuse_1x1 = nn.Conv2d(channels + 64, channels, kernel_size=1)
        self.head = SegmentationHead(channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        fdino = self.dino(x)
        flocal = self.local(x)
        if self.dino_upsampler_type == "anyup" and self.dino_anyup is not None:
            fdino_up = self.dino_anyup(x, fdino, target_hw=flocal.shape[-2:])
        else:
            fdino_up = self.dino_up(fdino, target_hw=flocal.shape[-2:])

        fused = torch.cat([fdino_up, flocal], dim=1)
        fused = self.fuse_1x1(fused)
        out = self.head(fused)

        target_hw = (x.shape[-2] // 4, x.shape[-1] // 4)
        out["seg_logit"] = F.interpolate(out["seg_logit"], size=target_hw, mode="bilinear", align_corners=False)
        return out


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


def mask_iou_at_threshold(pred_logit: torch.Tensor, seg_target: torch.Tensor, thr: float = 0.5, eps: float = 1e-7) -> float:
    pred = (torch.sigmoid(pred_logit) > thr).float()
    tgt = seg_target.float()

    pred = pred.view(pred.shape[0], -1)
    tgt = tgt.view(tgt.shape[0], -1)

    inter = (pred * tgt).sum(dim=1)
    union = ((pred + tgt) > 0).float().sum(dim=1)
    iou = torch.where(union > 0, inter / (union + eps), torch.ones_like(union))
    return float(iou.mean().item())


def unnormalize_image(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1)
    return y.permute(1, 2, 0).cpu().numpy()


def upsample_map_to_image(map_2d: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(map_2d).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=out_hw, mode="nearest")
    return t[0, 0].numpy()


def make_sample_figure(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    max_items: int,
    thr: float,
    title_prefix: str,
    positives_only: bool = False,
):
    model.eval()
    n = min(max_items, len(dataset))
    rng = np.random.default_rng()

    # Prefer informative sampling.
    pos_idx = getattr(dataset, "pos_dataset_indices", [])
    neg_idx = getattr(dataset, "neg_dataset_indices", [])
    if positives_only and len(pos_idx) > 0:
        n_pos = min(len(pos_idx), n)
        idxs = rng.choice(pos_idx, size=n_pos, replace=False).tolist()
        if len(idxs) < n:
            remaining = n - len(idxs)
            if len(neg_idx) > 0:
                add = rng.choice(neg_idx, size=min(remaining, len(neg_idx)), replace=False).tolist()
                idxs.extend(add)
            if len(idxs) < n:
                all_idx = list(range(len(dataset)))
                rng.shuffle(all_idx)
                for i in all_idx:
                    if i not in idxs:
                        idxs.append(i)
                    if len(idxs) >= n:
                        break
    elif len(pos_idx) > 0 and len(neg_idx) > 0 and n >= 2:
        n_pos = min(len(pos_idx), max(1, n // 2))
        n_neg = min(len(neg_idx), n - n_pos)
        pick_pos = rng.choice(pos_idx, size=n_pos, replace=False).tolist()
        pick_neg = rng.choice(neg_idx, size=n_neg, replace=False).tolist()
        idxs = pick_pos + pick_neg
        rng.shuffle(idxs)
    else:
        idxs = rng.choice(len(dataset), size=n, replace=False).tolist()

    images = []
    targets = []
    metas = []
    for idx in idxs:
        s = dataset[int(idx)]
        images.append(s["image"])
        targets.append(s["seg_target"])
        metas.append(s["meta"])

    x = torch.stack(images, dim=0).to(device)
    seg_t = torch.stack(targets, dim=0)

    with torch.no_grad():
        pred = model(x)
        seg_p = torch.sigmoid(pred["seg_logit"]).detach().cpu()

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        img = unnormalize_image(x[i].detach().cpu())
        gt = seg_t[i, 0].numpy()
        pr = seg_p[i, 0].numpy()
        gt_up = upsample_map_to_image(gt, img.shape[:2])
        pr_up = upsample_map_to_image(pr, img.shape[:2])
        lb = (pr_up > thr).astype(np.float32)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(
            f"{title_prefix} tile\nobjs={metas[i]['num_objects']} fg={int((gt_up > 0).any())}"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_up, cmap="magma", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title("GT seg")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pr_up, cmap="magma", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("Pred prob")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(lb, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 3].set_title(f"Pred mask thr={thr}")
        axes[i, 3].axis("off")

    plt.tight_layout()
    return fig


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iou_threshold: float,
    train: bool,
    epoch: int,
    split_name: str,
):
    model.train(train)
    if not train:
        model.eval()

    total_losses = []
    iou_scores = []

    pbar = tqdm(loader, desc=f"{split_name} epoch {epoch:02d}", leave=False)
    for batch in pbar:
        x = batch["image"].to(device)
        seg_t = batch["seg_target"].to(device)

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = mcc_loss_with_logits(pred["seg_logit"], seg_t)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        loss_val = float(loss.detach().cpu())
        iou_val = mask_iou_at_threshold(pred["seg_logit"].detach(), seg_t, thr=iou_threshold)
        total_losses.append(loss_val)
        iou_scores.append(iou_val)

        pbar.set_postfix({"loss": f"{loss_val:.4f}", "iou": f"{iou_val:.3f}"})

    return {
        "total_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
        "mask_iou": float(np.mean(iou_scores)) if iou_scores else float("nan"),
    }


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser(description="Stage-1 segmentation training (frozen DINO + local ResNet)")

    ap.add_argument("--data-dir", type=Path, default=Path("data/record_pairs"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs"))
    ap.add_argument("--run-name", type=str, default="")

    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--tile-stride", type=int, default=112)

    ap.add_argument("--label-name", type=str, default="vehicle")
    ap.add_argument("--min-poly-points", type=int, default=3)

    ap.add_argument("--seg-out-stride", type=int, default=4)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--fusion-channels", type=int, default=256)
    ap.add_argument("--dino-upsampler-type", type=str, choices=["learned", "anyup"], default="learned")
    ap.add_argument("--anyup-q-chunk-size", type=int, default=256)

    ap.add_argument("--balance-train-50-50", action="store_true", default=True)
    ap.add_argument("--no-balance-train-50-50", action="store_false", dest="balance_train_50_50")
    ap.add_argument("--balance-val-50-50", action="store_true", default=False)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset-size", type=int, default=0, help="Number of records (images) to train/eval split on. 0 = all.")

    ap.add_argument("--val-interval", type=int, default=1)
    ap.add_argument("--train-example-items", type=int, default=3)
    ap.add_argument("--val-example-items", type=int, default=3)
    ap.add_argument("--image-log-interval", type=int, default=1)

    ap.add_argument("--iou-threshold", type=float, default=0.5)

    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")

    a = ap.parse_args()

    return Cfg(
        data_dir=a.data_dir,
        output_dir=a.output_dir,
        run_name=a.run_name,
        tile_size=a.tile_size,
        tile_stride=a.tile_stride,
        label_name=a.label_name,
        min_poly_points=a.min_poly_points,
        seg_out_stride=a.seg_out_stride,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        fusion_channels=a.fusion_channels,
        dino_upsampler_type=a.dino_upsampler_type,
        anyup_q_chunk_size=a.anyup_q_chunk_size,
        balance_train_50_50=a.balance_train_50_50,
        balance_val_50_50=a.balance_val_50_50,
        seed=a.seed,
        subset_size=a.subset_size,
        val_interval=a.val_interval,
        train_example_items=a.train_example_items,
        val_example_items=a.val_example_items,
        image_log_interval=a.image_log_interval,
        iou_threshold=a.iou_threshold,
        trust_torch_hub_repo=a.trust_torch_hub_repo,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {cfg.data_dir}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_slug = stamp if cfg.run_name == "" else f"{stamp}-{cfg.run_name}"
    run_dir = cfg.output_dir / run_slug
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, indent=2)
    )
    write_hparams_yaml(run_dir / "hparams.yaml", cfg)

    writer = SummaryWriter(log_dir=str(run_dir))

    records = load_records(cfg.data_dir, cfg.label_name, cfg.min_poly_points)
    if len(records) == 0:
        raise RuntimeError("No records found.")

    rng = np.random.default_rng(cfg.seed)
    order = np.arange(len(records))
    rng.shuffle(order)
    records = [records[i] for i in order]
    print(f"Loaded records={len(records)}")
    print("Effective flags:")
    print(f"  balance_train_50_50={cfg.balance_train_50_50}")
    print(f"  balance_val_50_50={cfg.balance_val_50_50}")
    print(f"  subset_size={cfg.subset_size} (applied after balancing/shuffle)")
    print(f"  tile_size={cfg.tile_size}, tile_stride={cfg.tile_stride}")
    print(f"  seg_out_stride={cfg.seg_out_stride}")
    print(f"  batch_size={cfg.batch_size}, num_workers={cfg.num_workers}")
    print(f"  dino_upsampler_type={cfg.dino_upsampler_type}")
    if cfg.dino_upsampler_type == "anyup":
        print(f"  anyup_q_chunk_size={cfg.anyup_q_chunk_size}")
    print(f"  val_interval={cfg.val_interval}, image_log_interval={cfg.image_log_interval}")
    print(f"  iou_threshold={cfg.iou_threshold}")

    split = int(0.95 * len(records))
    train_records = records[:split]
    val_records = records[split:]

    train_ds = SegTileDataset(
        train_records,
        tile_size=cfg.tile_size,
        stride=cfg.tile_stride,
        seg_out_stride=cfg.seg_out_stride,
        seed=cfg.seed,
        balance_50_50=cfg.balance_train_50_50,
        dataset_name="train",
    )
    val_ds = SegTileDataset(
        val_records,
        tile_size=cfg.tile_size,
        stride=cfg.tile_stride,
        seg_out_stride=cfg.seg_out_stride,
        seed=cfg.seed,
        balance_50_50=cfg.balance_val_50_50,
        dataset_name="val",
    )

    # Apply subset at the end: balancing has already been applied in dataset generation.
    if cfg.subset_size > 0:
        n = min(cfg.subset_size, len(train_ds))
        ds_rng = np.random.default_rng(cfg.seed)
        keep_idx = ds_rng.choice(len(train_ds), size=n, replace=False).tolist()
        train_ds.samples = [train_ds.samples[i] for i in keep_idx]
        train_ds.pos_dataset_indices = [i for i, s in enumerate(train_ds.samples) if s["is_object"] == 1]
        train_ds.neg_dataset_indices = [i for i, s in enumerate(train_ds.samples) if s["is_object"] == 0]
        print(
            f"Applied subset_size={cfg.subset_size} after balancing. "
            f"effective_train_tiles={len(train_ds)}"
        )
    else:
        print(f"Using full train set after dataset balancing step. effective_train_tiles={len(train_ds)}")

    print(
        f"train_tile_balance: pos={len(getattr(train_ds, 'pos_dataset_indices', []))} "
        f"neg={len(getattr(train_ds, 'neg_dataset_indices', []))}"
    )
    if len(getattr(train_ds, "pos_dataset_indices", [])) == 0:
        found = discover_labels(cfg.data_dir)
        raise RuntimeError(
            "No positive train tiles found after dataset build. "
            f"Current --label-name='{cfg.label_name}'. "
            f"Discovered labels in dataset: {found}. "
            "Set --label-name to the correct class (e.g. 'Tank' or 'enemy')."
        )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"train_tiles={len(train_ds)} val_tiles={len(val_ds)}")

    model = Stage1SegNet(
        channels=cfg.fusion_channels,
        trust_repo=cfg.trust_torch_hub_repo,
        dino_upsampler_type=cfg.dino_upsampler_type,
        anyup_q_chunk_size=cfg.anyup_q_chunk_size,
    ).to(device)

    # Freeze both backbones to reduce overfitting.
    for p in model.dino.parameters():
        p.requires_grad = False
    for p in model.local.parameters():
        p.requires_grad = False
    # In anyup mode, keep the learned upsampler frozen and use AnyUp instead.
    if cfg.dino_upsampler_type == "anyup":
        for p in model.dino_up.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.dino_upsampler_type == "anyup":
        module_msg = "fuse_1x1, head (AnyUp + backbones frozen)"
    else:
        module_msg = "dino_up, fuse_1x1, head"
    print(f"trainable modules: {module_msg} | trainable_params={sum(p.numel() for p in trainable)}")

    history = []
    best_val_iou = -1.0

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            iou_threshold=cfg.iou_threshold,
            train=True,
            epoch=epoch,
            split_name="train",
        )

        va = {"total_loss": float("nan"), "mask_iou": float("nan")}
        if epoch % cfg.val_interval == 0:
            va = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                iou_threshold=cfg.iou_threshold,
                train=False,
                epoch=epoch,
                split_name="val",
            )

        row = {
            "epoch": epoch,
            "train_total": tr["total_loss"],
            "train_iou": tr["mask_iou"],
            "val_total": va["total_loss"],
            "val_iou": va["mask_iou"],
        }
        history.append(row)

        print(
            f"epoch={epoch:02d} "
            f"train_total={row['train_total']:.4f} train_iou={row['train_iou']:.4f} "
            f"val_total={row['val_total']:.4f} val_iou={row['val_iou']:.4f}"
        )

        writer.add_scalar("loss/train_total", row["train_total"], epoch)
        writer.add_scalar("metric/train_mask_iou", row["train_iou"], epoch)
        if epoch % cfg.val_interval == 0:
            writer.add_scalar("loss/val_total", row["val_total"], epoch)
            writer.add_scalar("metric/val_mask_iou", row["val_iou"], epoch)

        if epoch % cfg.image_log_interval == 0:
            train_fig = make_sample_figure(
                model=model,
                dataset=train_ds,
                device=device,
                max_items=cfg.train_example_items,
                thr=cfg.iou_threshold,
                title_prefix="Train",
                positives_only=False,
            )
            writer.add_figure("examples/train", train_fig, global_step=epoch)
            plt.close(train_fig)

            val_fig = make_sample_figure(
                model=model,
                dataset=val_ds,
                device=device,
                max_items=cfg.val_example_items,
                thr=cfg.iou_threshold,
                title_prefix="Val",
                positives_only=False,
            )
            writer.add_figure("examples/val", val_fig, global_step=epoch)
            plt.close(val_fig)

        if epoch % cfg.val_interval == 0 and row["val_iou"] > best_val_iou:
            best_val_iou = row["val_iou"]
            best_path = ckpt_dir / "best_val_iou.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "cfg": asdict(cfg),
                    "best_val_iou": best_val_iou,
                },
                best_path,
            )

    final_ckpt = ckpt_dir / "final.pt"
    torch.save(
        {
            "epoch": cfg.epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "cfg": asdict(cfg),
            "best_val_iou": best_val_iou,
        },
        final_ckpt,
    )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    writer.close()

    print(f"Saved final checkpoint: {final_ckpt}")
    print(f"TensorBoard logdir: {run_dir}")


if __name__ == "__main__":
    main()
