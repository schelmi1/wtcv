#!/usr/bin/env python3
import argparse
import io
import json
import math
import random
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.transforms import functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from losses import mcc_bce_boundary_loss, segmentation_metrics
from models import Stage1SegNet
from wtcv_utils.records import discover_labels, load_labelme_records
from wtcv_utils.tiling import crop_with_pad, tile_origins

@dataclass
class Cfg:
    data_dir: Path
    output_dir: Path
    run_name: str
    resume_checkpoint: Optional[Path] = None

    tile_size: int = 224
    tile_stride: int = 112
    tile_scales: str = "1.0"

    label_name: str = "vehicle"
    fp_label: str = "fp"
    min_poly_points: int = 3

    seg_out_stride: int = 4

    batch_size: int = 8
    num_workers: int = 8
    dataloader_verbose: bool = True
    epochs: int = 5
    lr: float = 1e-4
    lr_scheduler: str = "cosine"  # none | cosine
    lr_min: float = 1e-5
    weight_decay: float = 1e-4

    fusion_channels: int = 256
    dino_upsampler_type: str = "learned"  # learned | anyup
    anyup_q_chunk_size: int = 256
    head_type: str = "pointwise"  # pointwise | dwsep | residual
    use_tile_cls_head: bool = True
    tile_cls_weight: float = 0.3
    use_zoom_cls_head: bool = True
    zoom_cls_weight: float = 0.2
    use_fp_supervision: bool = True
    fp_neg_weight: float = 0.3
    fp_neg_ratio: float = 0.5

    balance_train_50_50: bool = True
    balance_val_50_50: bool = True
    augment_low_vis: bool = False
    hard_negative_mining: bool = True
    hnm_hard_ratio: float = 0.3
    hnm_pool_frac: float = 0.2

    seed: int = 42
    subset_size: int = 0  # 0 means use all records

    val_interval: int = 5
    train_example_items: int = 3
    val_example_items: int = 3
    image_log_interval: int = 1

    iou_threshold: float = 0.5
    mcc_weight: float = 0.4
    mcc_warmup_epochs: int = 3
    bce_weight: float = 0.5
    boundary_weight: float = 0.2

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


def objects_in_tile(objs: List[Dict], x0: int, y0: int, size: int) -> List[Dict]:
    x1, y1 = x0 + size, y0 + size
    out = []
    for o in objs:
        bx0, by0, bx1, by1 = o["bbox_xyxy"]
        # Keep only objects whose bbox is fully contained in the tile.
        fully_within = (bx0 >= x0) and (by0 >= y0) and (bx1 <= x1) and (by1 <= y1)
        if not fully_within:
            continue
        # Keep shifted geometry in tile-local coordinates.
        sbx0 = float(max(0.0, min(size, bx0 - x0)))
        sby0 = float(max(0.0, min(size, by0 - y0)))
        sbx1 = float(max(0.0, min(size, bx1 - x0)))
        sby1 = float(max(0.0, min(size, by1 - y0)))
        if sbx1 <= sbx0 or sby1 <= sby0:
            continue
        spts = []
        for p in o.get("points", []):
            px = float(max(0.0, min(size, float(p[0]) - x0)))
            py = float(max(0.0, min(size, float(p[1]) - y0)))
            spts.append([px, py])
        out.append(
            {
                "label_cf": str(o.get("label_cf", "")),
                "is_fp": bool(o.get("is_fp", False)),
                "bbox_xyxy": [sbx0, sby0, sbx1, sby1],
                "points": spts,
                "shape_type": o.get("shape_type", "polygon"),
                "poly_area": o["poly_area"],
            }
        )
    return out


def build_seg_target(tile_size: int, objs: List[Dict], out_stride: int) -> np.ndarray:
    full = Image.new("L", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(full)
    for o in objs:
        pts = o.get("points", []) or []
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
            continue
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
        tile_configs: List[Tuple[int, int]],
        seg_out_stride: int,
        seed: int,
        balance_50_50: bool,
        fp_neg_ratio: float = 0.5,
        augment_low_vis: bool = False,
        is_train: bool = False,
        dataset_name: str = "dataset",
        verbose: bool = False,
    ):
        self.records = records
        self.tile_size = tile_size
        self.tile_configs = tile_configs
        self.seg_out_stride = seg_out_stride
        self.balance_50_50 = balance_50_50
        self.fp_neg_ratio = float(max(0.0, min(1.0, fp_neg_ratio)))
        self.augment_low_vis = augment_low_vis
        self.is_train = is_train
        self.verbose = verbose
        self.seed = int(seed)

        # Triplets: (ridx, x0, y0, objs_in_tile, is_object)
        all_triplets = []
        obj_triplets = []
        non_obj_triplets = []
        fp_non_obj_triplets = []
        pure_non_obj_triplets = []

        for ridx, r in tqdm(
            enumerate(records),
            total=len(records),
            desc=f"Building {dataset_name} tile index",
            leave=verbose,
        ):
            for src_tile_size, src_stride in self.tile_configs:
                for x0, y0 in tile_origins(r["width"], r["height"], src_tile_size, src_stride):
                    objs = objects_in_tile(r["objects"], x0, y0, src_tile_size)
                    pos_objs = [o for o in objs if not bool(o.get("is_fp", False))]
                    fp_objs = [o for o in objs if bool(o.get("is_fp", False))]
                    item = {
                        "ridx": ridx,
                        "x0": x0,
                        "y0": y0,
                        "src_tile_size": src_tile_size,
                        "objs": objs,
                        "pos_objs": pos_objs,
                        "fp_objs": fp_objs,
                        "is_object": int(len(pos_objs) > 0),
                        "has_fp": int(len(fp_objs) > 0),
                    }
                    all_triplets.append(item)
                    if item["is_object"] == 1:
                        obj_triplets.append(item)
                    else:
                        non_obj_triplets.append(item)
                        if item["has_fp"] == 1:
                            fp_non_obj_triplets.append(item)
                        else:
                            pure_non_obj_triplets.append(item)

        if balance_50_50 and len(obj_triplets) > 0 and len(non_obj_triplets) > 0:
            n = min(len(obj_triplets), len(non_obj_triplets))
            rng = np.random.default_rng(seed)
            pos_sel = rng.choice(len(obj_triplets), size=n, replace=False)
            n_fp = min(len(fp_non_obj_triplets), int(round(n * self.fp_neg_ratio)))
            n_rest = n - n_fp
            sel_neg = []
            if n_fp > 0:
                fp_sel = rng.choice(len(fp_non_obj_triplets), size=n_fp, replace=False)
                sel_neg.extend([fp_non_obj_triplets[i] for i in fp_sel])
            if n_rest > 0:
                pool = pure_non_obj_triplets
                if len(pool) >= n_rest:
                    rest_sel = rng.choice(len(pool), size=n_rest, replace=False)
                    sel_neg.extend([pool[i] for i in rest_sel])
                else:
                    if len(pool) > 0:
                        sel_neg.extend(pool)
                    rem = n_rest - len(pool)
                    if rem > 0:
                        pool2 = non_obj_triplets
                        rep = rem > len(pool2)
                        fill_sel = rng.choice(len(pool2), size=rem, replace=rep)
                        sel_neg.extend([pool2[i] for i in fill_sel])
            self.samples = [obj_triplets[i] for i in pos_sel] + sel_neg
            rng.shuffle(self.samples)
        else:
            self.samples = all_triplets

        # Dataset-level positive/negative index lists (relative to self.samples).
        self.pos_dataset_indices = [i for i, s in enumerate(self.samples) if s["is_object"] == 1]
        self.neg_dataset_indices = [i for i, s in enumerate(self.samples) if s["is_object"] == 0]
        self.fp_neg_dataset_indices = [i for i, s in enumerate(self.samples) if (s["is_object"] == 0 and s["has_fp"] == 1)]
        self.neg_hard_scores = np.zeros(len(self.samples), dtype=np.float32)

        self.normalize = torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        if self.verbose:
            print(
                f"{dataset_name}_tiles built: total={len(self.samples)} "
                f"pos={len(self.pos_dataset_indices)} neg={len(self.neg_dataset_indices)} "
                f"fp_neg={len(self.fp_neg_dataset_indices)} "
                f"(balanced={self.balance_50_50} fp_neg_ratio={self.fp_neg_ratio:.2f})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def update_hard_negative_scores(self, score_by_index: Dict[int, float]) -> None:
        for idx, score in score_by_index.items():
            i = int(idx)
            if i < 0 or i >= len(self.samples):
                continue
            if self.samples[i]["is_object"] == 0:
                self.neg_hard_scores[i] = max(float(self.neg_hard_scores[i]), float(score))

    def build_epoch_indices_for_hnm(
        self,
        epoch: int,
        seed: int,
        hard_ratio: float,
        pool_frac: float,
    ) -> List[int]:
        if len(self.pos_dataset_indices) == 0 or len(self.neg_dataset_indices) == 0:
            idxs = list(range(len(self.samples)))
            rng = np.random.default_rng(seed + epoch)
            rng.shuffle(idxs)
            return idxs

        rng = np.random.default_rng(seed + epoch)
        pos = list(self.pos_dataset_indices)
        n_pos = len(pos)
        n_neg = n_pos

        neg = np.array(self.neg_dataset_indices, dtype=np.int64)
        neg_scores = self.neg_hard_scores[neg]
        pool_n = max(1, int(len(neg) * max(0.0, min(1.0, pool_frac))))
        hard_order = np.argsort(-neg_scores)
        hard_pool = neg[hard_order[:pool_n]]

        hard_take = min(len(hard_pool), int(n_neg * max(0.0, min(1.0, hard_ratio))))
        hard_sel = rng.choice(hard_pool, size=hard_take, replace=False).tolist() if hard_take > 0 else []

        remaining = n_neg - len(hard_sel)
        neg_set = set(neg.tolist())
        hard_set = set(hard_pool.tolist())
        rest_pool = np.array(sorted(list(neg_set - hard_set)), dtype=np.int64)
        if len(rest_pool) == 0:
            rest_pool = neg
        rest_take = min(len(rest_pool), remaining)
        rest_sel = rng.choice(rest_pool, size=rest_take, replace=False).tolist() if rest_take > 0 else []

        if len(hard_sel) + len(rest_sel) < n_neg:
            fill = n_neg - (len(hard_sel) + len(rest_sel))
            extra = rng.choice(neg, size=fill, replace=(fill > len(neg))).tolist()
            rest_sel.extend(extra)

        idxs = pos + hard_sel + rest_sel
        rng.shuffle(idxs)
        return idxs

    def _low_vis_augment(self, tile: Image.Image) -> Image.Image:
        # Strong low-visibility simulation: blur/haze/contrast/gamma/jpeg artifacts.
        img = tile
        if random.random() < 0.9:
            bf = random.uniform(0.6, 1.35)
            cf = random.uniform(0.55, 1.35)
            sf = random.uniform(0.5, 1.2)
            hf = random.uniform(-0.03, 0.03)
            img = TF.adjust_brightness(img, bf)
            img = TF.adjust_contrast(img, cf)
            img = TF.adjust_saturation(img, sf)
            img = TF.adjust_hue(img, hf)
        if random.random() < 0.45:
            img = TF.gaussian_blur(img, kernel_size=[3, 3], sigma=[0.1, 1.4])
        if random.random() < 0.35:
            gamma = random.uniform(0.7, 1.5)
            img = TF.adjust_gamma(img, gamma)
        if random.random() < 0.4:
            # JPEG compression artifacts
            buf = io.BytesIO()
            q = random.randint(25, 65)
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        ridx = item["ridx"]
        x0 = item["x0"]
        y0 = item["y0"]
        src_tile_size = int(item["src_tile_size"])
        r = self.records[ridx]

        img = Image.open(r["image_path"]).convert("RGB")
        tile = crop_with_pad(img, x0, y0, src_tile_size)
        if src_tile_size != self.tile_size:
            tile = tile.resize((self.tile_size, self.tile_size), Image.BILINEAR)
        if self.is_train and self.augment_low_vis:
            tile = self._low_vis_augment(tile)

        scale = float(self.tile_size) / float(max(src_tile_size, 1))
        pos_objs_scaled = []
        fp_objs_scaled = []
        for o in item["objs"]:
            bx0, by0, bx1, by1 = o["bbox_xyxy"]
            spts = [[float(px) * scale, float(py) * scale] for px, py in (o.get("points", []) or [])]
            rec = {
                "bbox_xyxy": [bx0 * scale, by0 * scale, bx1 * scale, by1 * scale],
                "points": spts,
                "shape_type": o.get("shape_type", "polygon"),
                "poly_area": o["poly_area"],
            }
            if bool(o.get("is_fp", False)):
                fp_objs_scaled.append(rec)
            else:
                pos_objs_scaled.append(rec)

        seg_t = build_seg_target(self.tile_size, pos_objs_scaled, self.seg_out_stride)
        fp_t = build_seg_target(self.tile_size, fp_objs_scaled, self.seg_out_stride)

        # Build a zoom ROI classification target:
        # positives use largest object bbox with context; negatives use deterministic random background crop.
        if len(pos_objs_scaled) > 0:
            best = None
            best_area = -1.0
            for o in pos_objs_scaled:
                bx0, by0, bx1, by1 = o["bbox_xyxy"]
                a = float(max(0.0, bx1 - bx0) * max(0.0, by1 - by0))
                if a > best_area:
                    best_area = a
                    best = (float(bx0), float(by0), float(bx1), float(by1))
            assert best is not None
            bx0, by0, bx1, by1 = best
            cx = 0.5 * (bx0 + bx1)
            cy = 0.5 * (by0 + by1)
            bw = max(10.0, (bx1 - bx0) * 1.8)
            bh = max(10.0, (by1 - by0) * 1.8)
            zx0 = max(0.0, cx - bw * 0.5)
            zy0 = max(0.0, cy - bh * 0.5)
            zx1 = min(float(self.tile_size), cx + bw * 0.5)
            zy1 = min(float(self.tile_size), cy + bh * 0.5)
            zoom_target = 1.0
        elif len(fp_objs_scaled) > 0:
            # Negative zoom: prefer annotated false-positive regions as hard negatives.
            best = None
            best_area = -1.0
            for o in fp_objs_scaled:
                bx0, by0, bx1, by1 = o["bbox_xyxy"]
                a = float(max(0.0, bx1 - bx0) * max(0.0, by1 - by0))
                if a > best_area:
                    best_area = a
                    best = (float(bx0), float(by0), float(bx1), float(by1))
            assert best is not None
            bx0, by0, bx1, by1 = best
            cx = 0.5 * (bx0 + bx1)
            cy = 0.5 * (by0 + by1)
            bw = max(10.0, (bx1 - bx0) * 1.8)
            bh = max(10.0, (by1 - by0) * 1.8)
            zx0 = max(0.0, cx - bw * 0.5)
            zy0 = max(0.0, cy - bh * 0.5)
            zx1 = min(float(self.tile_size), cx + bw * 0.5)
            zy1 = min(float(self.tile_size), cy + bh * 0.5)
            zoom_target = 0.0
        else:
            # Deterministic pseudo-random negative crop by dataset index.
            rr = np.random.default_rng(self.seed * 1_000_003 + int(idx))
            bw = float(rr.uniform(0.18, 0.45) * self.tile_size)
            bh = float(rr.uniform(0.18, 0.45) * self.tile_size)
            cx = float(rr.uniform(bw * 0.5, self.tile_size - bw * 0.5))
            cy = float(rr.uniform(bh * 0.5, self.tile_size - bh * 0.5))
            zx0 = max(0.0, cx - bw * 0.5)
            zy0 = max(0.0, cy - bh * 0.5)
            zx1 = min(float(self.tile_size), cx + bw * 0.5)
            zy1 = min(float(self.tile_size), cy + bh * 0.5)
            zoom_target = 0.0
        # Ensure valid box extents.
        if zx1 <= zx0:
            zx1 = min(float(self.tile_size), zx0 + 2.0)
        if zy1 <= zy0:
            zy1 = min(float(self.tile_size), zy0 + 2.0)

        x = TF.to_tensor(tile)
        x = self.normalize(x)

        return {
            "image": x,
            "seg_target": torch.from_numpy(seg_t).unsqueeze(0),
            "fp_target": torch.from_numpy(fp_t).unsqueeze(0),
            "tile_target": torch.tensor([float(item["is_object"])], dtype=torch.float32),
            "zoom_target": torch.tensor([float(zoom_target)], dtype=torch.float32),
            "zoom_box": torch.tensor([zx0, zy0, zx1, zy1], dtype=torch.float32),
            "meta": {
                "image_path": r["image_path"],
                "tile_origin": (x0, y0),
                "source_tile_size": src_tile_size,
                "num_objects": len(item["pos_objs"]),
                "num_fp_objects": len(item["fp_objs"]),
                "is_positive_tile": int(item["is_object"]),
                "sample_index": int(idx),
            },
        }


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
    mcc_weight: float,
    bce_weight: float,
    boundary_weight: float,
    tile_cls_weight: float,
    zoom_cls_weight: float,
    use_fp_supervision: bool,
    fp_neg_weight: float,
    train: bool,
    epoch: int,
    split_name: str,
    collect_hard_scores: bool = False,
    writer: SummaryWriter | None = None,
    global_step_start: int = 0,
    dataloader_verbose: bool = False,
):
    model.train(train)
    if not train:
        model.eval()

    total_losses = []
    mcc_losses = []
    bce_losses = []
    boundary_losses = []
    tile_cls_losses = []
    zoom_cls_losses = []
    fp_sup_losses = []
    soft_iou_scores = []
    pos_iou_scores = []
    neg_fp_rates = []
    fp_activation_scores = []
    hard_scores: Dict[int, float] = {}

    pbar = tqdm(loader, desc=f"{split_name} epoch {epoch:02d}", leave=dataloader_verbose)
    step_count = 0
    last_step_time = time.time()
    for batch in pbar:
        now = time.time()
        data_time = now - last_step_time
        x = batch["image"].to(device)
        seg_t = batch["seg_target"].to(device)
        fp_t = batch.get("fp_target", None)
        tile_t = batch["tile_target"].to(device)
        zoom_t = batch.get("zoom_target", None)
        zoom_box = batch.get("zoom_box", None)
        if zoom_t is not None:
            zoom_t = zoom_t.to(device)
        if zoom_box is not None:
            zoom_box = zoom_box.to(device)
        if fp_t is not None:
            fp_t = fp_t.to(device)
        grad_norm_val: Optional[float] = None

        with torch.set_grad_enabled(train):
            pred = model(x, zoom_boxes=zoom_box)
            loss, parts = mcc_bce_boundary_loss(
                pred["seg_logit"],
                seg_t,
                mcc_weight=mcc_weight,
                bce_weight=bce_weight,
                boundary_weight=boundary_weight,
            )
            tile_cls_val = float("nan")
            if ("tile_logit" in pred) and (tile_cls_weight > 0):
                tile_bce = F.binary_cross_entropy_with_logits(pred["tile_logit"], tile_t)
                loss = loss + tile_cls_weight * tile_bce
                tile_cls_val = float(tile_bce.detach().item())
            parts["tile_cls"] = tile_cls_val
            zoom_cls_val = float("nan")
            if ("zoom_logit" in pred) and (zoom_cls_weight > 0) and (zoom_t is not None):
                zoom_bce = F.binary_cross_entropy_with_logits(pred["zoom_logit"], zoom_t)
                loss = loss + zoom_cls_weight * zoom_bce
                zoom_cls_val = float(zoom_bce.detach().item())
            parts["zoom_cls"] = zoom_cls_val
            fp_sup_val = float("nan")
            if use_fp_supervision and (fp_neg_weight > 0) and (fp_t is not None):
                fp_mask = fp_t > 0.5
                if bool(fp_mask.any().item()):
                    fp_bce = F.binary_cross_entropy_with_logits(
                        pred["seg_logit"][fp_mask],
                        torch.zeros_like(pred["seg_logit"][fp_mask]),
                    )
                    loss = loss + fp_neg_weight * fp_bce
                    fp_sup_val = float(fp_bce.detach().item())
            parts["fp_sup"] = fp_sup_val
            parts["total"] = float(loss.detach().item())
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                total_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    param_norm = float(g.norm(2).item())
                    total_norm_sq += param_norm * param_norm
                grad_norm_val = math.sqrt(total_norm_sq)
                optimizer.step()

        loss_val = float(loss.detach().cpu())
        metric_vals = segmentation_metrics(pred["seg_logit"].detach(), seg_t, thr=iou_threshold)
        soft_iou_val = metric_vals["soft_iou"]
        pos_iou_val = metric_vals["pos_iou"]
        neg_fp_val = metric_vals["neg_fp_rate"]
        fp_activation_val = float("nan")
        if fp_t is not None:
            fp_mask_eval = fp_t > 0.5
            if bool(fp_mask_eval.any().item()):
                fp_activation_val = float(torch.sigmoid(pred["seg_logit"].detach())[fp_mask_eval].mean().item())
        total_losses.append(loss_val)
        mcc_losses.append(parts["mcc"])
        bce_losses.append(parts["bce"])
        boundary_losses.append(parts["boundary"])
        tile_cls_losses.append(parts["tile_cls"])
        zoom_cls_losses.append(parts["zoom_cls"])
        fp_sup_losses.append(parts["fp_sup"])
        soft_iou_scores.append(soft_iou_val)
        pos_iou_scores.append(pos_iou_val)
        neg_fp_rates.append(neg_fp_val)
        fp_activation_scores.append(fp_activation_val)

        if collect_hard_scores:
            probs = torch.sigmoid(pred["seg_logit"].detach())
            neg_mask = (seg_t.view(seg_t.shape[0], -1).sum(dim=1) == 0)
            meta = batch.get("meta", {})
            idxs = meta.get("sample_index", None)
            if idxs is not None:
                if torch.is_tensor(idxs):
                    idx_list = idxs.detach().cpu().tolist()
                elif isinstance(idxs, list):
                    idx_list = [int(v) for v in idxs]
                else:
                    idx_list = []
                for bi, ds_idx in enumerate(idx_list):
                    if bi >= probs.shape[0]:
                        break
                    if bool(neg_mask[bi].item()):
                        hs = float(probs[bi, 0].max().item())
                        prev = hard_scores.get(int(ds_idx), 0.0)
                        if hs > prev:
                            hard_scores[int(ds_idx)] = hs

        if writer is not None:
            gs = global_step_start + step_count
            writer.add_scalar(f"loss_step/{split_name}_total", loss_val, gs)
            writer.add_scalar(f"loss_step/{split_name}_mcc", parts["mcc"], gs)
            writer.add_scalar(f"loss_step/{split_name}_bce", parts["bce"], gs)
            writer.add_scalar(f"loss_step/{split_name}_boundary", parts["boundary"], gs)
            if not math.isnan(parts["tile_cls"]):
                writer.add_scalar(f"loss_step/{split_name}_tile_cls", parts["tile_cls"], gs)
            if not math.isnan(parts["zoom_cls"]):
                writer.add_scalar(f"loss_step/{split_name}_zoom_cls", parts["zoom_cls"], gs)
            if not math.isnan(parts["fp_sup"]):
                writer.add_scalar(f"loss_step/{split_name}_fp_sup", parts["fp_sup"], gs)
            writer.add_scalar(f"metric_step/{split_name}_soft_iou", soft_iou_val, gs)
            writer.add_scalar(f"metric_step/{split_name}_pos_iou", pos_iou_val, gs)
            writer.add_scalar(f"metric_step/{split_name}_neg_fp_rate", neg_fp_val, gs)
            if not math.isnan(fp_activation_val):
                writer.add_scalar(f"metric_step/{split_name}_fp_activation", fp_activation_val, gs)
            # Backward-compatible alias: mask_iou now reports soft_iou.
            writer.add_scalar(f"metric_step/{split_name}_mask_iou", soft_iou_val, gs)
            if grad_norm_val is not None:
                writer.add_scalar(f"grad_step/{split_name}_l2_norm", grad_norm_val, gs)
        step_count += 1

        postfix = {
            "loss": f"{loss_val:.4f}",
            "soft_iou": f"{soft_iou_val:.3f}",
            "pos_iou": f"{pos_iou_val:.3f}",
            "neg_fp": f"{neg_fp_val:.3f}",
        }
        if grad_norm_val is not None:
            postfix["grad"] = f"{grad_norm_val:.3f}"
        if not math.isnan(parts["fp_sup"]):
            postfix["fp_sup"] = f"{parts['fp_sup']:.3f}"
        if dataloader_verbose:
            step_time = time.time() - now
            postfix["data_s"] = f"{data_time:.3f}"
            postfix["step_s"] = f"{step_time:.3f}"
        pbar.set_postfix(postfix)
        last_step_time = time.time()

    def _safe_nanmean(vals: List[float]) -> float:
        if len(vals) == 0:
            return float("nan")
        arr = np.array(vals, dtype=np.float32)
        if not np.any(~np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    return {
        "total_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
        "mcc_loss": float(np.mean(mcc_losses)) if mcc_losses else float("nan"),
        "bce_loss": float(np.mean(bce_losses)) if bce_losses else float("nan"),
        "boundary_loss": float(np.mean(boundary_losses)) if boundary_losses else float("nan"),
        "tile_cls_loss": _safe_nanmean(tile_cls_losses),
        "zoom_cls_loss": _safe_nanmean(zoom_cls_losses),
        "fp_sup_loss": _safe_nanmean(fp_sup_losses),
        "soft_iou": _safe_nanmean(soft_iou_scores),
        "pos_iou": _safe_nanmean(pos_iou_scores),
        "neg_fp_rate": _safe_nanmean(neg_fp_rates),
        "fp_activation": _safe_nanmean(fp_activation_scores),
        # Backward-compatible alias: mask_iou now reports soft_iou.
        "mask_iou": _safe_nanmean(soft_iou_scores),
        "hard_scores": hard_scores,
        "num_steps": int(step_count),
    }


def parse_tile_configs(tile_size: int, tile_stride: int, tile_scales: str) -> List[Tuple[int, int]]:
    vals = []
    for tok in str(tile_scales).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        vals.append(float(tok))
    if len(vals) == 0:
        vals = [1.0]

    out = []
    seen = set()
    for s in vals:
        ts = max(14, int(round(tile_size * s)))
        st = max(1, int(round(tile_stride * s)))
        key = (ts, st)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def get_epoch_mcc_weight(target_weight: float, warmup_epochs: int, epoch: int) -> float:
    if warmup_epochs <= 0:
        return float(target_weight)
    if epoch <= 0:
        return 0.0
    if epoch >= warmup_epochs:
        return float(target_weight)
    return float(target_weight) * (float(epoch) / float(warmup_epochs))


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: Cfg):
    if cfg.lr_scheduler == "none":
        return None
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.epochs),
            eta_min=cfg.lr_min,
        )
    raise ValueError(f"Unsupported lr_scheduler: {cfg.lr_scheduler}")


def parse_args() -> Cfg:
    ap = argparse.ArgumentParser(description="Stage-1 segmentation training (frozen DINO + local ResNet)")

    ap.add_argument("--data-dir", type=Path, default=Path("data/record_pairs"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs"))
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--resume-checkpoint", type=Path, default=None)

    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--tile-stride", type=int, default=112)
    ap.add_argument("--tile-scales", type=str, default="1.0", help="Comma-separated scale factors for mixed tiling, e.g. 1.0,1.5")

    ap.add_argument("--label", dest="label_name", type=str, default="vehicle")
    ap.add_argument("--label-name", dest="label_name", type=str, help=argparse.SUPPRESS)
    ap.add_argument("--fp-label", type=str, default="fp")
    ap.add_argument("--min-poly-points", type=int, default=3)

    ap.add_argument("--seg-out-stride", type=int, default=4)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dataloader-verbose", action="store_true", default=True)
    ap.add_argument("--no-dataloader-verbose", action="store_false", dest="dataloader_verbose")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-scheduler", type=str, choices=["none", "cosine"], default="cosine")
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--fusion-channels", type=int, default=256)
    ap.add_argument("--dino-upsampler", dest="dino_upsampler_type", type=str, choices=["learned", "anyup"], default="learned")
    ap.add_argument("--dino-upsampler-type", dest="dino_upsampler_type", type=str, choices=["learned", "anyup"], help=argparse.SUPPRESS)
    ap.add_argument("--anyup-q-chunk-size", type=int, default=256)
    ap.add_argument("--head-type", type=str, choices=["pointwise", "dwsep", "residual"], default="pointwise")
    ap.add_argument("--use-tile-cls-head", action="store_true", default=True)
    ap.add_argument("--no-use-tile-cls-head", action="store_false", dest="use_tile_cls_head")
    ap.add_argument("--tile-cls-weight", type=float, default=0.3)
    ap.add_argument("--use-zoom-cls-head", action="store_true", default=True)
    ap.add_argument("--no-use-zoom-cls-head", action="store_false", dest="use_zoom_cls_head")
    ap.add_argument("--zoom-cls-weight", type=float, default=0.2)
    ap.add_argument("--use-fp-supervision", action="store_true", default=True)
    ap.add_argument("--no-use-fp-supervision", action="store_false", dest="use_fp_supervision")
    ap.add_argument("--fp-neg-weight", type=float, default=0.3)
    ap.add_argument("--fp-neg-ratio", type=float, default=0.5)

    ap.add_argument("--balance-train-50-50", action="store_true", default=True)
    ap.add_argument("--no-balance-train-50-50", action="store_false", dest="balance_train_50_50")
    ap.add_argument("--balance-val-50-50", action="store_true", default=True)
    ap.add_argument("--no-balance-val-50-50", action="store_false", dest="balance_val_50_50")
    ap.add_argument("--augment-low-vis", action="store_true", default=False)
    ap.add_argument("--hard-negative-mining", action="store_true", default=True)
    ap.add_argument("--no-hard-negative-mining", action="store_false", dest="hard_negative_mining")
    ap.add_argument("--hnm-hard-ratio", type=float, default=0.3, help="Fraction of sampled negatives drawn from hard pool.")
    ap.add_argument("--hnm-pool-frac", type=float, default=0.2, help="Top fraction of negative samples considered hard pool.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset-size", type=int, default=0, help="Number of records (images) to train/eval split on. 0 = all.")

    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--train-example-items", type=int, default=3)
    ap.add_argument("--val-example-items", type=int, default=3)
    ap.add_argument("--image-log-interval", type=int, default=1)

    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--mcc-weight", type=float, default=0.4)
    ap.add_argument("--mcc-warmup-epochs", type=int, default=3)
    ap.add_argument("--bce-weight", type=float, default=0.5)
    ap.add_argument("--boundary-weight", type=float, default=0.2)

    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")

    a = ap.parse_args()

    return Cfg(
        data_dir=a.data_dir,
        output_dir=a.output_dir,
        run_name=a.run_name,
        resume_checkpoint=a.resume_checkpoint,
        tile_size=a.tile_size,
        tile_stride=a.tile_stride,
        tile_scales=a.tile_scales,
        label_name=a.label_name,
        fp_label=a.fp_label,
        min_poly_points=a.min_poly_points,
        seg_out_stride=a.seg_out_stride,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        dataloader_verbose=a.dataloader_verbose,
        epochs=a.epochs,
        lr=a.lr,
        lr_scheduler=a.lr_scheduler,
        lr_min=a.lr_min,
        weight_decay=a.weight_decay,
        fusion_channels=a.fusion_channels,
        dino_upsampler_type=a.dino_upsampler_type,
        anyup_q_chunk_size=a.anyup_q_chunk_size,
        head_type=a.head_type,
        use_tile_cls_head=a.use_tile_cls_head,
        tile_cls_weight=a.tile_cls_weight,
        use_zoom_cls_head=a.use_zoom_cls_head,
        zoom_cls_weight=a.zoom_cls_weight,
        use_fp_supervision=a.use_fp_supervision,
        fp_neg_weight=a.fp_neg_weight,
        fp_neg_ratio=a.fp_neg_ratio,
        balance_train_50_50=a.balance_train_50_50,
        balance_val_50_50=a.balance_val_50_50,
        augment_low_vis=a.augment_low_vis,
        hard_negative_mining=a.hard_negative_mining,
        hnm_hard_ratio=a.hnm_hard_ratio,
        hnm_pool_frac=a.hnm_pool_frac,
        seed=a.seed,
        subset_size=a.subset_size,
        val_interval=a.val_interval,
        train_example_items=a.train_example_items,
        val_example_items=a.val_example_items,
        image_log_interval=a.image_log_interval,
        iou_threshold=a.iou_threshold,
        mcc_weight=a.mcc_weight,
        mcc_warmup_epochs=a.mcc_warmup_epochs,
        bce_weight=a.bce_weight,
        boundary_weight=a.boundary_weight,
        trust_torch_hub_repo=a.trust_torch_hub_repo,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {cfg.data_dir}")

    resume_ckpt = cfg.resume_checkpoint
    resume_blob = None
    if resume_ckpt is not None:
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_ckpt}")
        try:
            resume_blob = torch.load(resume_ckpt, map_location="cpu", weights_only=True)
        except Exception:
            # Older checkpoints may require full unpickling.
            resume_blob = torch.load(resume_ckpt, map_location="cpu", weights_only=False)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg.run_name == "":
        run_suffix = "resume" if resume_ckpt is not None else ""
        run_slug = f"{stamp}-{run_suffix}" if run_suffix else stamp
    else:
        run_slug = f"{stamp}-{cfg.run_name}"
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

    records = load_labelme_records(
        cfg.data_dir,
        cfg.label_name,
        cfg.min_poly_points,
        include_fp=True,
        fp_label=cfg.fp_label,
        load_workers=cfg.num_workers,
    )
    if len(records) == 0:
        raise RuntimeError("No records found.")

    rng = np.random.default_rng(cfg.seed)
    order = np.arange(len(records))
    rng.shuffle(order)
    records = [records[i] for i in order]
    print(f"Loaded records={len(records)}")
    print("Effective flags:")
    print(f"  resume_checkpoint={str(cfg.resume_checkpoint) if cfg.resume_checkpoint else 'None'}")
    if cfg.resume_checkpoint is not None:
        print("  note: --epochs is interpreted as additional epochs when resuming")
        print("  note: resume always writes to a new run directory")
        print("  note: resume resets LR/scheduler to cfg.lr/cfg.lr_min for the new run")
    print(f"  balance_train_50_50={cfg.balance_train_50_50}")
    print(f"  balance_val_50_50={cfg.balance_val_50_50}")
    print(f"  subset_size={cfg.subset_size} (applied after balancing/shuffle)")
    print(f"  tile_size={cfg.tile_size}, tile_stride={cfg.tile_stride}")
    print(f"  tile_scales={cfg.tile_scales}")
    print(f"  seg_out_stride={cfg.seg_out_stride}")
    print(f"  batch_size={cfg.batch_size}, num_workers={cfg.num_workers}")
    print(f"  dataloader_verbose={cfg.dataloader_verbose}")
    print(f"  dino_upsampler_type={cfg.dino_upsampler_type}")
    if cfg.dino_upsampler_type == "anyup":
        print(f"  anyup_q_chunk_size={cfg.anyup_q_chunk_size}")
        print(f"  head_type={cfg.head_type}")
    print(f"  use_tile_cls_head={cfg.use_tile_cls_head} tile_cls_weight={cfg.tile_cls_weight}")
    print(f"  use_zoom_cls_head={cfg.use_zoom_cls_head} zoom_cls_weight={cfg.zoom_cls_weight}")
    print(
        f"  use_fp_supervision={cfg.use_fp_supervision} fp_label='{cfg.fp_label}' "
        f"fp_neg_weight={cfg.fp_neg_weight} fp_neg_ratio={cfg.fp_neg_ratio}"
    )
    print(f"  val_interval={cfg.val_interval}, image_log_interval={cfg.image_log_interval}")
    print(f"  iou_threshold={cfg.iou_threshold}")
    print(
        f"  loss={cfg.mcc_weight}*mcc + {cfg.bce_weight}*bce + "
        f"{cfg.boundary_weight}*boundary + {cfg.tile_cls_weight}*tile_cls + "
        f"{cfg.zoom_cls_weight}*zoom_cls + {cfg.fp_neg_weight}*fp_sup(if enabled)"
    )
    print(f"  mcc_warmup_epochs={cfg.mcc_warmup_epochs}")
    print(f"  lr={cfg.lr} lr_scheduler={cfg.lr_scheduler} lr_min={cfg.lr_min}")
    print(f"  augment_low_vis={cfg.augment_low_vis}")
    print(
        f"  hard_negative_mining={cfg.hard_negative_mining} "
        f"(hard_ratio={cfg.hnm_hard_ratio}, pool_frac={cfg.hnm_pool_frac})"
    )

    split = int(0.95 * len(records))
    train_records = records[:split]
    val_records = records[split:]

    tile_configs = parse_tile_configs(cfg.tile_size, cfg.tile_stride, cfg.tile_scales)
    print(f"  tile_configs={tile_configs}")

    train_ds = SegTileDataset(
        train_records,
        tile_size=cfg.tile_size,
        tile_configs=tile_configs,
        seg_out_stride=cfg.seg_out_stride,
        seed=cfg.seed,
        balance_50_50=cfg.balance_train_50_50,
        fp_neg_ratio=cfg.fp_neg_ratio,
        augment_low_vis=cfg.augment_low_vis,
        is_train=True,
        dataset_name="train",
        verbose=cfg.dataloader_verbose,
    )
    val_ds = SegTileDataset(
        val_records,
        tile_size=cfg.tile_size,
        tile_configs=tile_configs,
        seg_out_stride=cfg.seg_out_stride,
        seed=cfg.seed,
        balance_50_50=cfg.balance_val_50_50,
        fp_neg_ratio=cfg.fp_neg_ratio,
        augment_low_vis=False,
        is_train=False,
        dataset_name="val",
        verbose=cfg.dataloader_verbose,
    )

    # Apply subset at the end: balancing has already been applied in dataset generation.
    if cfg.subset_size > 0:
        n = min(cfg.subset_size, len(train_ds))
        ds_rng = np.random.default_rng(cfg.seed)
        keep_idx = ds_rng.choice(len(train_ds), size=n, replace=False).tolist()
        train_ds.samples = [train_ds.samples[i] for i in keep_idx]
        train_ds.pos_dataset_indices = [i for i, s in enumerate(train_ds.samples) if s["is_object"] == 1]
        train_ds.neg_dataset_indices = [i for i, s in enumerate(train_ds.samples) if s["is_object"] == 0]
        train_ds.fp_neg_dataset_indices = [
            i for i, s in enumerate(train_ds.samples) if (s["is_object"] == 0 and s.get("has_fp", 0) == 1)
        ]
        train_ds.neg_hard_scores = np.zeros(len(train_ds.samples), dtype=np.float32)
        print(
            f"Applied subset_size={cfg.subset_size} after balancing. "
            f"effective_train_tiles={len(train_ds)}"
        )
    else:
        print(f"Using full train set after dataset balancing step. effective_train_tiles={len(train_ds)}")
        train_ds.neg_hard_scores = np.zeros(len(train_ds.samples), dtype=np.float32)

    print(
        f"train_tile_balance: pos={len(getattr(train_ds, 'pos_dataset_indices', []))} "
        f"neg={len(getattr(train_ds, 'neg_dataset_indices', []))} "
        f"fp_neg={len(getattr(train_ds, 'fp_neg_dataset_indices', []))}"
    )
    if len(getattr(train_ds, "pos_dataset_indices", [])) == 0:
        found = discover_labels(cfg.data_dir)
        raise RuntimeError(
            "No positive train tiles found after dataset build. "
            f"Current --label='{cfg.label_name}'. "
            f"Discovered labels in dataset: {found}. "
            "Set --label to the correct class (e.g. 'Tank' or 'enemy')."
        )

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"train_tiles={len(train_ds)} val_tiles={len(val_ds)}")

    model = Stage1SegNet(
        channels=cfg.fusion_channels,
        trust_repo=cfg.trust_torch_hub_repo,
        dino_upsampler_type=cfg.dino_upsampler_type,
        anyup_q_chunk_size=cfg.anyup_q_chunk_size,
        head_type=cfg.head_type,
        use_tile_cls_head=cfg.use_tile_cls_head,
        use_zoom_cls_head=cfg.use_zoom_cls_head,
    ).to(device)

    # Freeze both backbones to reduce overfitting.
    for p in model.dino.parameters():
        p.requires_grad = False
    if model.local is not None:
        for p in model.local.parameters():
            p.requires_grad = False
    # In anyup mode, keep the learned upsampler frozen and use AnyUp directly.
    if cfg.dino_upsampler_type == "anyup":
        for p in model.dino_up.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_lr_scheduler(optimizer, cfg)

    if cfg.dino_upsampler_type == "anyup":
        module_msg = f"anyup_head[{cfg.head_type}] (AnyUp + backbones frozen)"
    else:
        module_msg = "dino_up, fuse_1x1, head"
    if cfg.use_tile_cls_head:
        module_msg = f"{module_msg}, tile_cls_head"
    if cfg.use_zoom_cls_head:
        module_msg = f"{module_msg}, zoom_cls_head"
    print(f"trainable modules: {module_msg} | trainable_params={sum(p.numel() for p in trainable)}")

    history = []
    best_val_iou = -1.0
    train_global_step = 0
    val_global_step = 0
    start_epoch = 1
    resume_base_epoch = 0
    target_end_epoch = cfg.epochs

    if resume_blob is not None:
        state = resume_blob["model"] if isinstance(resume_blob, dict) and "model" in resume_blob else resume_blob
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print(
                "warning: strict resume load failed; retrying with strict=False "
                f"(likely architecture delta such as zoom head). error={e}"
            )
            model.load_state_dict(state, strict=False)
        resume_cfg = resume_blob.get("cfg", {}) if isinstance(resume_blob, dict) else {}
        resume_has_zoom_head = bool(resume_cfg.get("use_zoom_cls_head", False))
        optimizer_loaded = False
        if isinstance(resume_blob, dict) and "optimizer" in resume_blob:
            # If architecture changed (e.g., zoom head added), optimizer param groups may mismatch.
            if resume_has_zoom_head != bool(cfg.use_zoom_cls_head):
                print(
                    "warning: resume checkpoint optimizer state skipped due to model head mismatch "
                    f"(checkpoint use_zoom_cls_head={resume_has_zoom_head}, "
                    f"current use_zoom_cls_head={cfg.use_zoom_cls_head}). "
                    "Using freshly initialized optimizer state."
                )
            else:
                try:
                    optimizer.load_state_dict(resume_blob["optimizer"])
                    optimizer_loaded = True
                except Exception as e:
                    print(
                        "warning: failed to load optimizer state from resume checkpoint "
                        f"(likely parameter-group mismatch after architecture change): {e}. "
                        "Using freshly initialized optimizer state."
                    )
        # Resume policy: start each resumed run with a fresh LR schedule from cfg.lr -> cfg.lr_min.
        # Loading an old CosineAnnealingLR state at/after T_max can make LR rise again.
        if optimizer_loaded:
            print("note: optimizer state loaded; resetting LR to cfg.lr and restarting scheduler for this resumed run.")
        else:
            print("note: using fresh optimizer state; initializing LR/scheduler from cfg.")
        for pg in optimizer.param_groups:
            pg["lr"] = float(cfg.lr)
            pg["initial_lr"] = float(cfg.lr)
        scheduler = build_lr_scheduler(optimizer, cfg)

        if isinstance(resume_blob, dict):
            history = list(resume_blob.get("history", []))
            best_val_iou = float(resume_blob.get("best_val_iou", best_val_iou))
            resume_base_epoch = int(resume_blob.get("epoch", 0))
            start_epoch = resume_base_epoch + 1
            train_global_step = int(resume_blob.get("train_global_step", 0))
            val_global_step = int(resume_blob.get("val_global_step", 0))

        target_end_epoch = resume_base_epoch + cfg.epochs
        print(
            f"Resumed from {resume_ckpt} | start_epoch={start_epoch} "
            f"target_end_epoch={target_end_epoch} "
            f"best_val_iou={best_val_iou:.4f} train_global_step={train_global_step} "
            f"val_global_step={val_global_step}"
        )
    else:
        target_end_epoch = cfg.epochs

    if start_epoch > target_end_epoch:
        print(
            f"Nothing to train: start_epoch={start_epoch} is greater than target_end_epoch={target_end_epoch}. "
            "Increase --epochs to continue training."
        )
        return

    for epoch in range(start_epoch, target_end_epoch + 1):
        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_mcc_weight = get_epoch_mcc_weight(
            target_weight=cfg.mcc_weight,
            warmup_epochs=cfg.mcc_warmup_epochs,
            epoch=epoch,
        )
        if cfg.hard_negative_mining:
            epoch_indices = train_ds.build_epoch_indices_for_hnm(
                epoch=epoch,
                seed=cfg.seed,
                hard_ratio=cfg.hnm_hard_ratio,
                pool_frac=cfg.hnm_pool_frac,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                sampler=SubsetRandomSampler(epoch_indices),
                num_workers=cfg.num_workers,
            )
        else:
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        if cfg.dataloader_verbose:
            print(
                f"epoch={epoch:02d} loader_info: "
                f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
                f"batch_size={cfg.batch_size} workers={cfg.num_workers}"
            )

        tr = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            iou_threshold=cfg.iou_threshold,
            mcc_weight=epoch_mcc_weight,
            bce_weight=cfg.bce_weight,
            boundary_weight=cfg.boundary_weight,
            tile_cls_weight=cfg.tile_cls_weight,
            zoom_cls_weight=cfg.zoom_cls_weight,
            use_fp_supervision=cfg.use_fp_supervision,
            fp_neg_weight=cfg.fp_neg_weight,
            train=True,
            epoch=epoch,
            split_name="train",
            collect_hard_scores=cfg.hard_negative_mining,
            writer=writer,
            global_step_start=train_global_step,
            dataloader_verbose=cfg.dataloader_verbose,
        )
        train_global_step += int(tr.get("num_steps", 0))
        if cfg.hard_negative_mining:
            train_ds.update_hard_negative_scores(tr.get("hard_scores", {}))

        va = {
            "total_loss": float("nan"),
            "mcc_loss": float("nan"),
            "bce_loss": float("nan"),
            "boundary_loss": float("nan"),
            "tile_cls_loss": float("nan"),
            "zoom_cls_loss": float("nan"),
            "fp_sup_loss": float("nan"),
            "soft_iou": float("nan"),
            "pos_iou": float("nan"),
            "neg_fp_rate": float("nan"),
            "fp_activation": float("nan"),
            "mask_iou": float("nan"),
        }
        if epoch % cfg.val_interval == 0:
            va = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                iou_threshold=cfg.iou_threshold,
                mcc_weight=epoch_mcc_weight,
                bce_weight=cfg.bce_weight,
                boundary_weight=cfg.boundary_weight,
                tile_cls_weight=cfg.tile_cls_weight,
                zoom_cls_weight=cfg.zoom_cls_weight,
                use_fp_supervision=cfg.use_fp_supervision,
                fp_neg_weight=cfg.fp_neg_weight,
                train=False,
                epoch=epoch,
                split_name="val",
                collect_hard_scores=False,
                writer=writer,
                global_step_start=val_global_step,
                dataloader_verbose=cfg.dataloader_verbose,
            )
            val_global_step += int(va.get("num_steps", 0))

        row = {
            "epoch": epoch,
            "train_total": tr["total_loss"],
            "train_tile_cls_loss": tr["tile_cls_loss"],
            "train_zoom_cls_loss": tr["zoom_cls_loss"],
            "train_fp_sup_loss": tr["fp_sup_loss"],
            "train_iou": tr["soft_iou"],
            "train_pos_iou": tr["pos_iou"],
            "train_neg_fp_rate": tr["neg_fp_rate"],
            "train_fp_activation": tr["fp_activation"],
            "val_total": va["total_loss"],
            "val_tile_cls_loss": va["tile_cls_loss"],
            "val_zoom_cls_loss": va["zoom_cls_loss"],
            "val_fp_sup_loss": va["fp_sup_loss"],
            "val_iou": va["soft_iou"],
            "val_pos_iou": va["pos_iou"],
            "val_neg_fp_rate": va["neg_fp_rate"],
            "val_fp_activation": va["fp_activation"],
        }
        history.append(row)

        print(
            f"epoch={epoch:02d} "
            f"lr={lr_now:.8f} "
            f"mcc_w={epoch_mcc_weight:.4f} "
            f"train_total={row['train_total']:.4f} train_tile_cls={row['train_tile_cls_loss']:.4f} "
            f"train_zoom_cls={row['train_zoom_cls_loss']:.4f} train_fp_sup={row['train_fp_sup_loss']:.4f} "
            f"train_iou={row['train_iou']:.4f} "
            f"train_pos_iou={row['train_pos_iou']:.4f} train_neg_fp={row['train_neg_fp_rate']:.4f} "
            f"train_fp_act={row['train_fp_activation']:.4f} "
            f"val_total={row['val_total']:.4f} val_tile_cls={row['val_tile_cls_loss']:.4f} "
            f"val_zoom_cls={row['val_zoom_cls_loss']:.4f} val_fp_sup={row['val_fp_sup_loss']:.4f} "
            f"val_iou={row['val_iou']:.4f} "
            f"val_pos_iou={row['val_pos_iou']:.4f} val_neg_fp={row['val_neg_fp_rate']:.4f} "
            f"val_fp_act={row['val_fp_activation']:.4f}"
        )

        writer.add_scalar("lr/epoch", lr_now, epoch)
        writer.add_scalar("loss_cfg/mcc_weight", epoch_mcc_weight, epoch)
        writer.add_scalar("metric/train_soft_iou", row["train_iou"], epoch)
        writer.add_scalar("metric/train_pos_iou", row["train_pos_iou"], epoch)
        writer.add_scalar("metric/train_neg_fp_rate", row["train_neg_fp_rate"], epoch)
        if not math.isnan(row["train_fp_activation"]):
            writer.add_scalar("metric/train_fp_activation", row["train_fp_activation"], epoch)
        if not math.isnan(row["train_tile_cls_loss"]):
            writer.add_scalar("loss_epoch/train_tile_cls", row["train_tile_cls_loss"], epoch)
        if not math.isnan(row["train_zoom_cls_loss"]):
            writer.add_scalar("loss_epoch/train_zoom_cls", row["train_zoom_cls_loss"], epoch)
        if not math.isnan(row["train_fp_sup_loss"]):
            writer.add_scalar("loss_epoch/train_fp_sup", row["train_fp_sup_loss"], epoch)
        # Backward-compatible alias: mask_iou now reports soft_iou.
        writer.add_scalar("metric/train_mask_iou", row["train_iou"], epoch)
        if epoch % cfg.val_interval == 0:
            writer.add_scalar("metric/val_soft_iou", row["val_iou"], epoch)
            writer.add_scalar("metric/val_pos_iou", row["val_pos_iou"], epoch)
            writer.add_scalar("metric/val_neg_fp_rate", row["val_neg_fp_rate"], epoch)
            if not math.isnan(row["val_fp_activation"]):
                writer.add_scalar("metric/val_fp_activation", row["val_fp_activation"], epoch)
            if not math.isnan(row["val_tile_cls_loss"]):
                writer.add_scalar("loss_epoch/val_tile_cls", row["val_tile_cls_loss"], epoch)
            if not math.isnan(row["val_zoom_cls_loss"]):
                writer.add_scalar("loss_epoch/val_zoom_cls", row["val_zoom_cls_loss"], epoch)
            if not math.isnan(row["val_fp_sup_loss"]):
                writer.add_scalar("loss_epoch/val_fp_sup", row["val_fp_sup_loss"], epoch)
            # Backward-compatible alias: mask_iou now reports soft_iou.
            writer.add_scalar("metric/val_mask_iou", row["val_iou"], epoch)

        if scheduler is not None:
            scheduler.step()

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
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "history": history,
                    "cfg": asdict(cfg),
                    "best_val_iou": best_val_iou,
                    "train_global_step": train_global_step,
                    "val_global_step": val_global_step,
                },
                best_path,
            )

    final_ckpt = ckpt_dir / "final.pt"
    torch.save(
        {
            "epoch": target_end_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
            "cfg": asdict(cfg),
            "best_val_iou": best_val_iou,
            "train_global_step": train_global_step,
            "val_global_step": val_global_step,
        },
        final_ckpt,
    )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    writer.close()

    print(f"Saved final checkpoint: {final_ckpt}")
    print(f"TensorBoard logdir: {run_dir}")


if __name__ == "__main__":
    main()
