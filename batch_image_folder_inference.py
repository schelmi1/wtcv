#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from curate_model_predictions_to_labelme import load_model, make_labelme_json, mask_to_polygons
from wtcv_utils.labelme import IMG_EXTS
from wtcv_utils.tiling import tile_origins


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="High-throughput tiled inference on image folders. "
        "Tiles from multiple images are batched together for faster GPU utilization."
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input-path", type=Path, required=True, help="Image folder or single image path")
    ap.add_argument("--output-dir", type=Path, default=Path("data/batch_inference_labelme"))
    ap.add_argument("--label", type=str, default="vehicle")

    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--tile-stride", type=int, default=512)
    ap.add_argument("--seg-out-stride", type=int, default=4)
    ap.add_argument("--tile-batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--pred-threshold", type=float, default=0.5)
    ap.add_argument("--use-tile-cls-gating", action="store_true", default=True)
    ap.add_argument("--no-use-tile-cls-gating", action="store_false", dest="use_tile_cls_gating")
    ap.add_argument("--tile-cls-threshold", type=float, default=0.5)
    ap.add_argument("--tile-cls-mode", type=str, choices=["hard", "multiply"], default="hard")
    ap.add_argument("--min-poly-area", type=float, default=20.0)
    ap.add_argument("--poly-epsilon-frac", type=float, default=0.002)

    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0, help="0 means all")
    ap.add_argument("--recursive", action="store_true", default=False)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--save-empty", action="store_true", default=False, help="Also write empty-json samples")
    ap.add_argument("--overwrite", action="store_true", default=False)
    return ap.parse_args()


def list_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() in IMG_EXTS:
        return [input_path]
    if not input_path.is_dir():
        return []
    if recursive:
        files = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in sorted(input_path.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return files


def crop_with_pad_np(image_rgb: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1, y1 = x0 + size, y0 + size
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)

    out = np.zeros((size, size, 3), dtype=np.uint8)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out_y0 = sy0 - y0
    out_x0 = sx0 - x0
    out[out_y0 : out_y0 + (sy1 - sy0), out_x0 : out_x0 + (sx1 - sx0)] = image_rgb[sy0:sy1, sx0:sx1]
    return out


def probe_image_sizes(paths: List[Path]) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    for p in tqdm(paths, desc="probe image sizes"):
        with Image.open(p) as im:
            w, h = map(int, im.size)
        sizes.append((w, h))
    return sizes


class MultiImageTileDataset(IterableDataset):
    def __init__(
        self,
        image_paths: List[Path],
        image_sizes: List[Tuple[int, int]],
        tile_origins_by_image: List[List[Tuple[int, int]]],
        tile_size: int,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.image_sizes = image_sizes
        self.tile_origins_by_image = tile_origins_by_image
        self.tile_size = int(tile_size)

    def _iter_image_indices(self) -> Iterable[int]:
        info = get_worker_info()
        n = len(self.image_paths)
        if info is None:
            yield from range(n)
            return
        wid = int(info.id)
        wnum = int(info.num_workers)
        for idx in range(wid, n, wnum):
            yield idx

    def __iter__(self):
        for image_idx in self._iter_image_indices():
            img_path = self.image_paths[image_idx]
            try:
                with Image.open(img_path) as im:
                    image_rgb = np.array(im.convert("RGB"), dtype=np.uint8)
            except Exception:
                continue

            for x0, y0 in self.tile_origins_by_image[image_idx]:
                tile = crop_with_pad_np(image_rgb, int(x0), int(y0), self.tile_size)
                tile_t = torch.from_numpy(tile).permute(2, 0, 1).contiguous()  # uint8 [3,H,W]
                yield {
                    "tile": tile_t,
                    "image_idx": int(image_idx),
                    "x0": int(x0),
                    "y0": int(y0),
                }


def collate_tiles(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "tile": torch.stack([b["tile"] for b in batch], dim=0),
        "image_idx": torch.tensor([b["image_idx"] for b in batch], dtype=torch.long),
        "x0": torch.tensor([b["x0"] for b in batch], dtype=torch.long),
        "y0": torch.tensor([b["y0"] for b in batch], dtype=torch.long),
    }


def finalize_image(
    image_idx: int,
    states: List[Dict],
    image_paths: List[Path],
    image_sizes: List[Tuple[int, int]],
    args: argparse.Namespace,
    output_dir: Path,
    summary_rows: List[Dict],
) -> None:
    st = states[image_idx]
    if st.get("finalized", False):
        return
    st["finalized"] = True

    w, h = image_sizes[image_idx]
    prob_lr = np.divide(st["accum"], np.maximum(st["count"], 1e-6))
    prob_full = (
        F.interpolate(
            torch.from_numpy(prob_lr).float().unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .cpu()
        .numpy()
    )

    pred_mask = (prob_full >= float(args.pred_threshold)).astype(np.uint8)
    polys = mask_to_polygons(
        pred_mask,
        min_area=float(args.min_poly_area),
        epsilon_frac=float(args.poly_epsilon_frac),
    )

    img_path = image_paths[image_idx]
    wrote = False
    out_img = output_dir / img_path.name
    out_json = output_dir / f"{img_path.stem}.json"
    if polys or bool(args.save_empty):
        if out_img.exists() and not bool(args.overwrite):
            pass
        else:
            shutil.copy2(img_path, out_img)
            d = make_labelme_json(
                image_name=out_img.name,
                h=h,
                w=w,
                polys=polys,
                label=str(args.label),
            )
            out_json.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            wrote = True

    tile_cls_mean = float(st["tile_cls_sum"] / max(st["seen"], 1))
    summary_rows.append(
        {
            "image": img_path.name,
            "image_path": str(img_path),
            "num_polygons": int(len(polys)),
            "pred_fg_pixels": int(pred_mask.sum()),
            "tile_cls_mean": tile_cls_mean,
            "output_json": str(out_json) if (polys or bool(args.save_empty)) else "",
            "wrote_output": bool(wrote),
        }
    )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if int(args.tile_size) <= 0:
        raise ValueError("--tile-size must be > 0")
    if int(args.tile_stride) <= 0:
        raise ValueError("--tile-stride must be > 0")
    if int(args.seg_out_stride) <= 0:
        raise ValueError("--seg-out-stride must be > 0")
    if int(args.tile_size) % 256 != 0:
        raise ValueError("--tile-size must be a multiple of 256 for current Stage1SegNet")

    image_paths = list_images(args.input_path, recursive=bool(args.recursive))
    if int(args.start_index) > 0:
        image_paths = image_paths[int(args.start_index) :]
    if int(args.max_images) > 0:
        image_paths = image_paths[: int(args.max_images)]
    if not image_paths:
        raise RuntimeError(f"No images found at: {args.input_path}")

    image_sizes = probe_image_sizes(image_paths)
    tile_lists = [
        tile_origins(w, h, tile=int(args.tile_size), stride=int(args.tile_stride))
        for (w, h) in tqdm(image_sizes, desc="build tile origins")
    ]
    tile_counts = [len(v) for v in tile_lists]
    total_tiles = int(sum(tile_counts))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_jsonl = output_dir / "batch_infer_summary.jsonl"
    summary_json = output_dir / "batch_infer_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    model, info = load_model(args.checkpoint, device)
    print("device:", device)
    print("amp:", use_amp)
    print("model:", info)
    print(f"images={len(image_paths)} total_tiles={total_tiles}")
    print(
        f"tile_size={args.tile_size} tile_stride={args.tile_stride} "
        f"tile_batch_size={args.tile_batch_size} num_workers={args.num_workers}"
    )
    print(
        f"use_tile_cls_gating={args.use_tile_cls_gating} tile_cls_threshold={args.tile_cls_threshold} "
        f"tile_cls_mode={args.tile_cls_mode}"
    )

    states: List[Dict] = []
    for (w, h), ntiles in zip(image_sizes, tile_counts):
        gh = max(1, h // int(args.seg_out_stride))
        gw = max(1, w // int(args.seg_out_stride))
        states.append(
            {
                "accum": np.zeros((gh, gw), dtype=np.float32),
                "count": np.zeros((gh, gw), dtype=np.float32),
                "seen": 0,
                "total": int(ntiles),
                "tile_cls_sum": 0.0,
                "finalized": False,
            }
        )

    ds = MultiImageTileDataset(
        image_paths=image_paths,
        image_sizes=image_sizes,
        tile_origins_by_image=tile_lists,
        tile_size=int(args.tile_size),
    )

    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.tile_batch_size)),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=bool(device.type == "cuda"),
        collate_fn=collate_tiles,
        persistent_workers=bool(int(args.num_workers) > 0),
        prefetch_factor=4 if int(args.num_workers) > 0 else None,
    )

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp and device.type == "cuda" else nullcontext()
    )

    summary_rows: List[Dict] = []
    pbar = tqdm(total=total_tiles, desc="tile infer")
    images_done = 0
    with torch.inference_mode():
        for batch in loader:
            x = batch["tile"].to(device, non_blocking=True).float().div_(255.0)
            x = (x - mean) / std
            with autocast_ctx:
                pred = model(x)
                probs = torch.sigmoid(pred["seg_logit"][:, 0])  # [B,h,w]

                if bool(args.use_tile_cls_gating) and ("tile_logit" in pred):
                    tile_prob = torch.sigmoid(pred["tile_logit"][:, 0]).view(-1, 1, 1)
                    if str(args.tile_cls_mode) == "hard":
                        probs = torch.where(tile_prob >= float(args.tile_cls_threshold), probs, torch.zeros_like(probs))
                    else:
                        probs = probs * tile_prob
                    tile_prob_1d = tile_prob[:, 0, 0]
                else:
                    tile_prob_1d = torch.ones((probs.shape[0],), dtype=probs.dtype, device=probs.device)

            probs_np = probs.float().cpu().numpy()
            tile_prob_np = tile_prob_1d.float().cpu().numpy()
            image_idx_np = batch["image_idx"].cpu().numpy()
            x0_np = batch["x0"].cpu().numpy()
            y0_np = batch["y0"].cpu().numpy()

            for bi in range(probs_np.shape[0]):
                img_idx = int(image_idx_np[bi])
                st = states[img_idx]
                prob = probs_np[bi]
                x0 = int(x0_np[bi])
                y0 = int(y0_np[bi])

                th, tw = prob.shape
                gx0, gy0 = x0 // int(args.seg_out_stride), y0 // int(args.seg_out_stride)
                gh, gw = st["accum"].shape
                gx1, gy1 = min(gw, gx0 + tw), min(gh, gy0 + th)
                pw = gx1 - gx0
                ph = gy1 - gy0
                if pw > 0 and ph > 0:
                    st["accum"][gy0:gy1, gx0:gx1] += prob[:ph, :pw]
                    st["count"][gy0:gy1, gx0:gx1] += 1.0

                st["seen"] += 1
                st["tile_cls_sum"] += float(tile_prob_np[bi])
                if st["seen"] >= st["total"]:
                    finalize_image(
                        image_idx=img_idx,
                        states=states,
                        image_paths=image_paths,
                        image_sizes=image_sizes,
                        args=args,
                        output_dir=output_dir,
                        summary_rows=summary_rows,
                    )
                    images_done += 1

            pbar.update(int(probs_np.shape[0]))
            pbar.set_postfix(images_done=images_done)
    pbar.close()

    for img_idx, st in enumerate(states):
        if not st["finalized"]:
            finalize_image(
                image_idx=img_idx,
                states=states,
                image_paths=image_paths,
                image_sizes=image_sizes,
                args=args,
                output_dir=output_dir,
                summary_rows=summary_rows,
            )

    summary_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in summary_rows) + ("\n" if summary_rows else "")
    )
    summary = {
        "images_total": len(image_paths),
        "tiles_total": total_tiles,
        "images_with_polygons": int(sum(1 for r in summary_rows if int(r["num_polygons"]) > 0)),
        "outputs_written": int(sum(1 for r in summary_rows if bool(r["wrote_output"]))),
        "output_dir": str(output_dir),
        "summary_jsonl": str(summary_jsonl),
        "checkpoint": str(args.checkpoint),
        "label": str(args.label),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
