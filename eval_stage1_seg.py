#!/usr/bin/env python3
import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from models import Stage1SegNet
from wtcv_utils.records import load_labelme_records
from wtcv_utils.tiling import crop_with_pad, tile_origins


@dataclass
class EvalCfg:
    data_dir: Path
    checkpoint: Path
    tile_size: int = 224
    tile_stride: int = 112
    seg_out_stride: int = 4
    label_name: str = "vehicle"
    min_poly_points: int = 3
    fusion_channels: int = 256
    pred_threshold: float = 0.5
    iou_thresholds: Tuple[float, ...] = tuple(np.arange(0.5, 0.96, 0.05).tolist())
    area_small: float = 32.0 * 32.0
    area_medium: float = 96.0 * 96.0
    trust_torch_hub_repo: bool = True
    num_workers: int = 8


def parse_args() -> EvalCfg:
    ap = argparse.ArgumentParser(description="Evaluate stage1 segmentation checkpoint")
    ap.add_argument("--data-dir", type=Path, default=Path("data/record_pairs"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--tile-stride", type=int, default=112)
    ap.add_argument("--seg-out-stride", type=int, default=4)
    ap.add_argument("--label-name", type=str, default="vehicle")
    ap.add_argument("--min-poly-points", type=int, default=3)
    ap.add_argument("--fusion-channels", type=int, default=256)
    ap.add_argument("--pred-threshold", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--trust-torch-hub-repo", action="store_true", default=True)
    ap.add_argument("--no-trust-torch-hub-repo", action="store_false", dest="trust_torch_hub_repo")
    a = ap.parse_args()

    return EvalCfg(
        data_dir=a.data_dir,
        checkpoint=a.checkpoint,
        tile_size=a.tile_size,
        tile_stride=a.tile_stride,
        seg_out_stride=a.seg_out_stride,
        label_name=a.label_name,
        min_poly_points=a.min_poly_points,
        fusion_channels=a.fusion_channels,
        pred_threshold=a.pred_threshold,
        num_workers=a.num_workers,
        trust_torch_hub_repo=a.trust_torch_hub_repo,
    )


class EvalImageDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        image_path = Path(rec["image_path"])
        img = Image.open(image_path).convert("RGB")
        return {
            "record": rec,
            "image_id": image_path.name,
            "image_np": np.array(img, dtype=np.uint8),
        }


def _collate_eval(batch: List[Dict]) -> Dict:
    # Batch size is 1 for full-image evaluation.
    return batch[0]


def infer_segmentation_on_image(
    model: Stage1SegNet,
    image_np: np.ndarray,
    tile_size: int,
    stride: int,
    seg_out_stride: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    img = Image.fromarray(image_np).convert("RGB")
    W, H = img.size

    gh, gw = H // seg_out_stride, W // seg_out_stride
    accum = np.zeros((gh, gw), dtype=np.float32)
    count = np.zeros((gh, gw), dtype=np.float32)

    norm = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for x0, y0 in tile_origins(W, H, tile_size, stride):
        tile = crop_with_pad(img, x0, y0, tile_size)
        x = norm(TF.to_tensor(tile)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)
            prob = torch.sigmoid(pred["seg_logit"])[0, 0].cpu().numpy()

        th, tw = prob.shape
        gx0, gy0 = x0 // seg_out_stride, y0 // seg_out_stride
        gx1, gy1 = min(gw, gx0 + tw), min(gh, gy0 + th)

        accum[gy0:gy1, gx0:gx1] += prob[: gy1 - gy0, : gx1 - gx0]
        count[gy0:gy1, gx0:gx1] += 1.0

    merged = accum / np.maximum(count, 1e-6)
    return merged


def polygon_to_mask(points: List[List[float]], w: int, h: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(m)
    if len(points) >= 3:
        draw.polygon([(float(p[0]), float(p[1])) for p in points], fill=1)
    return np.array(m, dtype=np.uint8)


def build_gt_instances(record: Dict, seg_out_stride: int) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
    w, h = int(record["width"]), int(record["height"])
    gw, gh = w // seg_out_stride, h // seg_out_stride

    inst_masks = []
    inst_areas = []
    union_full = np.zeros((h, w), dtype=np.uint8)

    for obj in record["objects"]:
        pts = obj.get("points", [])
        m = polygon_to_mask(pts, w, h)
        if m.sum() == 0:
            continue
        union_full = np.maximum(union_full, m)

        t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(gh, gw), mode="nearest")
        ms = (t[0, 0].numpy() > 0.5).astype(np.uint8)
        if ms.sum() == 0:
            continue

        inst_masks.append(ms)
        inst_areas.append(float(m.sum()))

    gt_union = (union_full > 0).astype(np.uint8)
    return inst_masks, inst_areas, gt_union


def connected_components(binary: np.ndarray) -> List[np.ndarray]:
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=np.uint8)
    comps = []
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    ys, xs = np.where(binary > 0)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = 1
        coords = []

        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or binary[ny, nx] == 0:
                    continue
                visited[ny, nx] = 1
                stack.append((ny, nx))

        comp = np.zeros_like(binary, dtype=np.uint8)
        yy, xx = zip(*coords)
        comp[np.array(yy), np.array(xx)] = 1
        comps.append(comp)

    return comps


def mask_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    inter = float(np.logical_and(a > 0, b > 0).sum())
    union = float(np.logical_or(a > 0, b > 0).sum())
    if union <= 0:
        return 0.0
    return inter / (union + eps)


def ap_from_pr(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / max(n_gt, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)

    rec_levels = np.linspace(0.0, 1.0, 101)
    prec_interp = np.zeros_like(rec_levels)
    for i, r in enumerate(rec_levels):
        mask = recall >= r
        prec_interp[i] = np.max(precision[mask]) if np.any(mask) else 0.0
    return float(np.mean(prec_interp))


def compute_ap_for_area(
    detections: List[Dict],
    gt_by_image: Dict[str, Dict],
    iou_thr: float,
    area_range: Tuple[float, float],
) -> float:
    area_min, area_max = area_range

    gt_match_flags = {}
    n_gt = 0
    for img_id, g in gt_by_image.items():
        keep = []
        for a in g["areas"]:
            in_range = (a >= area_min) and (a < area_max)
            keep.append(in_range)
            if in_range:
                n_gt += 1
        gt_match_flags[img_id] = np.zeros(len(g["masks"]), dtype=np.uint8)
        gt_match_flags[img_id + "__keep"] = np.array(keep, dtype=np.uint8)

    if n_gt == 0:
        return float("nan")

    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    tp = []
    fp = []

    for d in dets:
        img_id = d["image_id"]
        pm = d["mask"]

        if img_id not in gt_by_image:
            tp.append(0)
            fp.append(1)
            continue

        g = gt_by_image[img_id]
        masks = g["masks"]
        keep = gt_match_flags[img_id + "__keep"]
        matched = gt_match_flags[img_id]

        best_iou = -1.0
        best_j = -1
        for j, gm in enumerate(masks):
            if keep[j] == 0:
                continue
            i = mask_iou(pm, gm)
            if i > best_iou:
                best_iou = i
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr and matched[best_j] == 0:
            matched[best_j] = 1
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    return ap_from_pr(np.array(tp, dtype=np.float32), np.array(fp, dtype=np.float32), n_gt)


def main() -> None:
    cfg = parse_args()

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {cfg.data_dir}")
    if not cfg.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {cfg.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    except Exception as e:
        warnings.warn(
            "weights_only=True failed while loading checkpoint; falling back to "
            "weights_only=False. Use only with trusted checkpoints. "
            f"Original error: {e}"
        )
        ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)

    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    model_channels = int(ckpt_cfg.get("fusion_channels", cfg.fusion_channels))
    dino_upsampler_type = str(ckpt_cfg.get("dino_upsampler_type", "learned"))
    anyup_q_chunk_size = int(ckpt_cfg.get("anyup_q_chunk_size", 256))
    head_type = str(ckpt_cfg.get("head_type", "pointwise"))
    use_tile_cls_head = bool(ckpt_cfg.get("use_tile_cls_head", False))
    use_zoom_cls_head = bool(ckpt_cfg.get("use_zoom_cls_head", False))

    model = Stage1SegNet(
        channels=model_channels,
        trust_repo=cfg.trust_torch_hub_repo,
        dino_upsampler_type=dino_upsampler_type,
        anyup_q_chunk_size=anyup_q_chunk_size,
        head_type=head_type,
        use_tile_cls_head=use_tile_cls_head,
        use_zoom_cls_head=use_zoom_cls_head,
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    records = load_labelme_records(
        cfg.data_dir,
        cfg.label_name,
        cfg.min_poly_points,
        load_workers=cfg.num_workers,
    )
    if len(records) == 0:
        raise RuntimeError("No records found")
    eval_ds = EvalImageDataset(records)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_eval,
        persistent_workers=cfg.num_workers > 0,
    )

    gt_by_image: Dict[str, Dict] = {}
    detections: List[Dict] = []

    semantic_inter = 0.0
    semantic_union = 0.0
    semantic_iou_per_img = []

    for sample in tqdm(eval_loader):
        rec = sample["record"]
        image_id = sample["image_id"]
        image_np = sample["image_np"]

        prob = infer_segmentation_on_image(
            model=model,
            image_np=image_np,
            tile_size=cfg.tile_size,
            stride=cfg.tile_stride,
            seg_out_stride=cfg.seg_out_stride,
            device=device,
        )

        pred_bin = (prob >= cfg.pred_threshold).astype(np.uint8)
        pred_components = connected_components(pred_bin)

        inst_masks, inst_areas, gt_union_full = build_gt_instances(rec, cfg.seg_out_stride)
        gt_by_image[image_id] = {"masks": inst_masks, "areas": inst_areas}

        # semantic IoU at full resolution
        w, h = int(rec["width"]), int(rec["height"])
        t = torch.from_numpy(pred_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(h, w), mode="nearest")
        pred_full = (t[0, 0].numpy() > 0.5).astype(np.uint8)

        inter = float(np.logical_and(pred_full > 0, gt_union_full > 0).sum())
        union = float(np.logical_or(pred_full > 0, gt_union_full > 0).sum())
        iou = (inter / union) if union > 0 else 1.0
        semantic_iou_per_img.append(iou)
        semantic_inter += inter
        semantic_union += union

        # predicted instances with scores
        for comp in pred_components:
            score = float(prob[comp > 0].mean()) if np.any(comp > 0) else 0.0
            detections.append({"image_id": image_id, "score": score, "mask": comp})

    # AP metrics
    area_all = (0.0, float("inf"))
    area_small = (0.0, cfg.area_small)
    area_medium = (cfg.area_small, cfg.area_medium)
    area_large = (cfg.area_medium, float("inf"))

    ap_all = []
    ap_small = []
    ap_medium = []
    ap_large = []

    for thr in cfg.iou_thresholds:
        ap_all.append(compute_ap_for_area(detections, gt_by_image, thr, area_all))
        ap_small.append(compute_ap_for_area(detections, gt_by_image, thr, area_small))
        ap_medium.append(compute_ap_for_area(detections, gt_by_image, thr, area_medium))
        ap_large.append(compute_ap_for_area(detections, gt_by_image, thr, area_large))

    def nanmean(x):
        arr = np.array(x, dtype=np.float32)
        return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")

    ap50 = compute_ap_for_area(detections, gt_by_image, 0.50, area_all)
    ap75 = compute_ap_for_area(detections, gt_by_image, 0.75, area_all)

    out = {
        "num_images": len(records),
        "num_detections": len(detections),
        "semantic_iou_mean_per_image": float(np.mean(semantic_iou_per_img)),
        "semantic_iou_global": (semantic_inter / semantic_union) if semantic_union > 0 else 1.0,
        "mAP_50_95": nanmean(ap_all),
        "mAP_50": float(ap50),
        "mAP_75": float(ap75),
        "mAP_small": nanmean(ap_small),
        "mAP_medium": nanmean(ap_medium),
        "mAP_large": nanmean(ap_large),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
