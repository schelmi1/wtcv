#!/usr/bin/env python3
import argparse
import shutil
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
from transformers import Sam2Model, Sam2Processor

from wtcv_utils.records import LabelmePair, load_labelme_pairs
from wtcv_utils.sam_masks import post_process_masks_compat, resolve_postprocess_sizes

@dataclass
class Record:
    image_path: Path
    json_path: Path
    json_data: Dict
    bbox_shape_indices: List[int]
    point_shape_indices: List[int]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert LabelMe bbox annotations to polygons with batched SAM2")
    ap.add_argument("--input-dir", type=Path, required=True, help="Folder with LabelMe image/json pairs")
    ap.add_argument("--output-dir", type=Path, default=Path("data/sam2_box_to_poly"))
    ap.add_argument("--model-id", type=str, default="facebook/sam2-hiera-small")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image-batch-size", type=int, default=4, help="How many images per SAM forward pass")
    ap.add_argument("--prompt-mode", type=str, choices=["bbox", "point"], default="point")
    ap.add_argument("--inference-mode", type=str, choices=["whole", "crop"], default="whole")
    ap.add_argument("--crop-size", type=int, default=512, help="Crop side in pixels for inference-mode=crop")
    ap.add_argument("--min-poly-area", type=float, default=20.0)
    ap.add_argument("--poly-epsilon-frac", type=float, default=0.002)
    ap.add_argument(
        "--label-filter",
        type=str,
        default="",
        help="Comma-separated labels to refine; blank means all labels.",
    )
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--max-images", type=int, default=0, help="0 means all")
    ap.add_argument(
        "--load-workers",
        type=int,
        default=8,
        help="Workers for JSON/image pair discovery in load_records",
    )
    return ap.parse_args()


def _build_record_from_pair(pair: LabelmePair, label_set: set[str]) -> Record:
    d = pair.json_data
    shapes = d.get("shapes", []) or []
    bbox_idx: List[int] = []
    point_idx: List[int] = []
    for i, s in enumerate(shapes):
        lab = str(s.get("label", "")).strip().casefold()
        if label_set and lab not in label_set:
            continue
        st = str(s.get("shape_type", "")).lower()
        pts = s.get("points", []) or []
        if st == "rectangle" and len(pts) >= 2:
            bbox_idx.append(i)
        elif len(pts) >= 3:
            point_idx.append(i)
    return Record(
        image_path=pair.image_path,
        json_path=pair.json_path,
        json_data=d,
        bbox_shape_indices=bbox_idx,
        point_shape_indices=point_idx,
    )


def load_records(input_dir: Path, load_workers: int, label_filter: str = "", max_images: int = 0) -> List[Record]:
    label_set = {x.strip().casefold() for x in str(label_filter).split(",") if x.strip()}
    pairs = load_labelme_pairs(
        input_dir,
        load_workers=load_workers,
        max_images=max_images,
        progress_desc="load_records",
        progress_leave=True,
    )
    return [_build_record_from_pair(p, label_set=label_set) for p in pairs]


def shape_rect_to_box_and_point(shape: Dict) -> Tuple[List[float], List[float]] | None:
    pts = shape.get("points", []) or []
    if len(pts) < 2:
        return None
    x0, y0 = float(pts[0][0]), float(pts[0][1])
    x1, y1 = float(pts[1][0]), float(pts[1][1])
    lx, rx = min(x0, x1), max(x0, x1)
    ty, by = min(y0, y1), max(y0, y1)
    if rx <= lx or by <= ty:
        return None
    box = [lx, ty, rx, by]
    point = [0.5 * (lx + rx), 0.5 * (ty + by)]
    return box, point


def polygon_centroid(points: List[List[float]]) -> List[float]:
    if len(points) < 3:
        if len(points) == 0:
            return [0.0, 0.0]
        x = float(np.mean([float(p[0]) for p in points]))
        y = float(np.mean([float(p[1]) for p in points]))
        return [x, y]
    xs = np.array([float(p[0]) for p in points], dtype=np.float64)
    ys = np.array([float(p[1]) for p in points], dtype=np.float64)
    x2 = np.roll(xs, -1)
    y2 = np.roll(ys, -1)
    cross = xs * y2 - x2 * ys
    a2 = np.sum(cross)  # 2A
    if abs(a2) < 1e-9:
        return [float(xs.mean()), float(ys.mean())]
    cx = np.sum((xs + x2) * cross) / (3.0 * a2)
    cy = np.sum((ys + y2) * cross) / (3.0 * a2)
    return [float(cx), float(cy)]


def shape_to_cog_point(shape: Dict) -> List[float] | None:
    st = str(shape.get("shape_type", "")).lower()
    pts = shape.get("points", []) or []
    if st == "rectangle":
        bp = shape_rect_to_box_and_point(shape)
        if bp is None:
            return None
        _, p = bp
        return p
    if len(pts) < 3:
        return None
    poly = [[float(p[0]), float(p[1])] for p in pts]
    return polygon_centroid(poly)


def shape_bounds(shape: Dict) -> List[float] | None:
    st = str(shape.get("shape_type", "")).lower()
    pts = shape.get("points", []) or []
    if st == "rectangle":
        bp = shape_rect_to_box_and_point(shape)
        if bp is None:
            return None
        box, _ = bp
        return box
    if len(pts) == 0:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    lx, rx = min(xs), max(xs)
    ty, by = min(ys), max(ys)
    if rx <= lx or by <= ty:
        return None
    return [lx, ty, rx, by]


def make_square_crop(img: Image.Image, cx: float, cy: float, crop_size: int) -> Tuple[Image.Image, int, int]:
    cs = int(max(8, crop_size))
    left = int(round(float(cx) - 0.5 * cs))
    top = int(round(float(cy) - 0.5 * cs))
    right = left + cs
    bottom = top + cs

    out = Image.new("RGB", (cs, cs), (0, 0, 0))
    iw, ih = img.size
    sx0 = max(0, left)
    sy0 = max(0, top)
    sx1 = min(iw, right)
    sy1 = min(ih, bottom)
    if sx1 > sx0 and sy1 > sy0:
        patch = img.crop((sx0, sy0, sx1, sy1))
        dx0 = sx0 - left
        dy0 = sy0 - top
        out.paste(patch, (dx0, dy0))
    return out, left, top


def clamp_box_to_crop(box: List[float], left: int, top: int, crop_size: int) -> List[float] | None:
    cs = float(crop_size)
    x0 = float(box[0]) - float(left)
    y0 = float(box[1]) - float(top)
    x1 = float(box[2]) - float(left)
    y1 = float(box[3]) - float(top)
    x0 = float(max(0.0, min(cs - 1.0, x0)))
    y0 = float(max(0.0, min(cs - 1.0, y0)))
    x1 = float(max(0.0, min(cs - 1.0, x1)))
    y1 = float(max(0.0, min(cs - 1.0, y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def clamp_point_to_crop(point: List[float], left: int, top: int, crop_size: int) -> List[float]:
    cs = float(crop_size)
    x = float(point[0]) - float(left)
    y = float(point[1]) - float(top)
    x = float(max(0.0, min(cs - 1.0, x)))
    y = float(max(0.0, min(cs - 1.0, y)))
    return [x, y]


def shift_and_clip_poly_to_image(poly: List[List[float]], left: int, top: int, width: int, height: int) -> List[List[float]]:
    out: List[List[float]] = []
    max_x = float(max(0, width - 1))
    max_y = float(max(0, height - 1))
    for p in poly:
        x = float(p[0]) + float(left)
        y = float(p[1]) + float(top)
        x = float(max(0.0, min(max_x, x)))
        y = float(max(0.0, min(max_y, y)))
        out.append([x, y])
    return out


def mask_to_polygons(mask_u8: np.ndarray, min_area: float, epsilon_frac: float) -> List[List[List[float]]]:
    m = (mask_u8 > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, epsilon_frac * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2).astype(float)
        if pts.shape[0] < 3:
            continue
        polys.append([[float(x), float(y)] for x, y in pts])
    return polys


def normalize_mask_candidates(mask_tensor: torch.Tensor, n_boxes: int) -> np.ndarray:
    # target shape: (n_boxes, n_masks, H, W)
    arr = mask_tensor.detach().cpu().numpy()
    while arr.ndim > 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]

    if arr.ndim != 4:
        raise RuntimeError(f"Unexpected mask tensor shape after squeeze: {arr.shape}")

    if arr.shape[0] == n_boxes:
        return arr
    if arr.shape[1] == n_boxes:
        return np.transpose(arr, (1, 0, 2, 3))

    if n_boxes == 1 and arr.shape[0] != 1:
        return arr[:1]
    raise RuntimeError(f"Cannot align masks with boxes: arr={arr.shape}, n_boxes={n_boxes}")


def normalize_scores(score_tensor: torch.Tensor, n_boxes: int) -> np.ndarray:
    # target shape: (n_boxes, n_masks)
    s = score_tensor.detach().cpu().numpy()
    while s.ndim > 2 and s.shape[0] == 1:
        s = s[0]

    if s.ndim == 1:
        s = s[np.newaxis, ...]

    if s.ndim != 2:
        raise RuntimeError(f"Unexpected score tensor shape after squeeze: {s.shape}")

    if s.shape[0] == n_boxes:
        return s
    if s.shape[1] == n_boxes:
        return s.T

    if n_boxes == 1 and s.shape[0] != 1:
        return s[:1]
    raise RuntimeError(f"Cannot align scores with boxes: score={s.shape}, n_boxes={n_boxes}")


def chunked(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {args.input_dir}")

    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(f"Output dir exists: {args.output_dir}. Use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading SAM2: {args.model_id} on {device}")
    processor = Sam2Processor.from_pretrained(args.model_id)
    sam = Sam2Model.from_pretrained(args.model_id).to(device).eval()

    records = load_records(args.input_dir, args.load_workers, label_filter=args.label_filter, max_images=args.max_images)

    print(f"records_total={len(records)}")
    print(f"image_batch_size={args.image_batch_size}")
    print(f"label_filter={args.label_filter if str(args.label_filter).strip() else 'ALL'}")

    written = 0
    total_prompts = 0
    converted_prompts = 0
    fallback_prompts = 0

    # Process records in batches.
    for batch in tqdm(list(chunked(records, args.image_batch_size)), desc="SAM batched bbox->poly"):
        inference_mode = str(args.inference_mode).strip().lower()
        if inference_mode == "whole":
            batch_images: List[Image.Image] = []
            batch_boxes: List[List[List[float]]] = []
            batch_points: List[List[List[float]]] = []
            batch_labels: List[List[int]] = []
            batch_records: List[Record] = []
            batch_shape_indices: List[List[int]] = []
            batch_true_prompt_counts: List[int] = []

            for rec in batch:
                img = Image.open(rec.image_path).convert("RGB")
                boxes_i: List[List[float]] = []
                points_i: List[List[float]] = []
                labels_i: List[int] = []
                shape_idx_i = []

                if args.prompt_mode == "bbox":
                    for si in rec.bbox_shape_indices:
                        shape = rec.json_data.get("shapes", [])[si]
                        bp = shape_rect_to_box_and_point(shape)
                        if bp is None:
                            continue
                        box, _ = bp
                        boxes_i.append(box)
                        shape_idx_i.append(si)
                    total_prompts += len(rec.bbox_shape_indices)
                else:
                    for si in rec.point_shape_indices:
                        shape = rec.json_data.get("shapes", [])[si]
                        p = shape_to_cog_point(shape)
                        if p is None:
                            continue
                        points_i.append(p)
                        labels_i.append(1)
                        shape_idx_i.append(si)
                    total_prompts += len(rec.point_shape_indices)

                if len(shape_idx_i) == 0:
                    out_img = args.output_dir / rec.image_path.name
                    out_json = args.output_dir / rec.json_path.name
                    shutil.copy2(rec.image_path, out_img)
                    out_json.write_text(json.dumps(rec.json_data, ensure_ascii=False, indent=2))
                    written += 1
                    continue

                batch_images.append(img)
                batch_boxes.append(boxes_i)
                batch_points.append(points_i)
                batch_labels.append(labels_i)
                batch_records.append(rec)
                batch_shape_indices.append(shape_idx_i)
                batch_true_prompt_counts.append(len(shape_idx_i))

            if len(batch_images) == 0:
                continue

            max_prompts = max(batch_true_prompt_counts)
            if args.prompt_mode == "bbox":
                padded_batch_boxes: List[List[List[float]]] = []
                for boxes_i in batch_boxes:
                    if len(boxes_i) < max_prompts:
                        pad_box = boxes_i[-1]
                        boxes_i = boxes_i + [pad_box] * (max_prompts - len(boxes_i))
                    padded_batch_boxes.append(boxes_i)
                inputs = processor(images=batch_images, input_boxes=padded_batch_boxes, return_tensors="pt")
            else:
                padded_points: List[List[List[List[float]]]] = []
                padded_labels: List[List[List[int]]] = []
                for pts_i, lbl_i in zip(batch_points, batch_labels):
                    cur_pts = [[[float(p[0]), float(p[1])]] for p in pts_i]
                    cur_lbl = [[int(l)] for l in lbl_i]
                    if len(cur_pts) < max_prompts:
                        pad_p = cur_pts[-1]
                        pad_l = cur_lbl[-1]
                        cur_pts = cur_pts + [pad_p] * (max_prompts - len(cur_pts))
                        cur_lbl = cur_lbl + [pad_l] * (max_prompts - len(cur_lbl))
                    padded_points.append(cur_pts)
                    padded_labels.append(cur_lbl)
                inputs = processor(images=batch_images, input_points=padded_points, input_labels=padded_labels, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = sam(**inputs, multimask_output=True)

            orig_sizes_cpu, reshaped_sizes_cpu = resolve_postprocess_sizes(inputs, batch_images)
            post_masks = post_process_masks_compat(
                image_processor=processor.image_processor,
                pred_masks_cpu=out.pred_masks.detach().cpu(),
                original_sizes_cpu=orig_sizes_cpu,
                reshaped_sizes_cpu=reshaped_sizes_cpu,
            )

            for bi, rec in enumerate(batch_records):
                n_prompts = int(batch_true_prompt_counts[bi])

                masks_all = normalize_mask_candidates(post_masks[bi], n_boxes=max_prompts)
                scores_all = normalize_scores(out.iou_scores[bi], n_boxes=max_prompts)
                masks_i = masks_all[:n_prompts]
                scores_i = scores_all[:n_prompts]

                d_new = deepcopy(rec.json_data)
                repl: Dict[int, List[Dict]] = {}

                for j in range(n_prompts):
                    best_idx = int(np.argmax(scores_i[j]))
                    best_mask = (masks_i[j, best_idx] > 0).astype(np.uint8)
                    polys = mask_to_polygons(best_mask, min_area=args.min_poly_area, epsilon_frac=args.poly_epsilon_frac)

                    si = batch_shape_indices[bi][j]
                    src_shape = d_new["shapes"][si]
                    label = src_shape.get("label", "vehicle")
                    flags = src_shape.get("flags", {}) or {}

                    if len(polys) == 0:
                        fallback_prompts += 1
                        repl[si] = [src_shape]
                        continue

                    converted_prompts += 1
                    ns = []
                    for poly in polys:
                        ns.append(
                            {
                                "label": label,
                                "points": poly,
                                "group_id": src_shape.get("group_id"),
                                "shape_type": "polygon",
                                "flags": {**flags, "source": "sam2_box_to_poly"},
                            }
                        )
                    repl[si] = ns

                final_shapes: List[Dict] = []
                for si, s in enumerate(d_new.get("shapes", [])):
                    if si in repl:
                        final_shapes.extend(repl[si])
                    else:
                        final_shapes.append(s)

                d_new["shapes"] = final_shapes
                out_img = args.output_dir / rec.image_path.name
                out_json = args.output_dir / rec.json_path.name
                shutil.copy2(rec.image_path, out_img)
                out_json.write_text(json.dumps(d_new, ensure_ascii=False, indent=2))
                written += 1
            continue

        # Crop mode: run each prompt on an object-centered fixed-size crop.
        prompt_tasks: List[Dict] = []
        rec_repl: Dict[int, Dict[int, List[Dict]]] = {}
        rec_prompt_idxs: Dict[int, List[int]] = {}

        for rec in batch:
            img = Image.open(rec.image_path).convert("RGB")
            iw, ih = img.size
            rec_key = id(rec)
            rec_repl[rec_key] = {}

            prompt_idxs = rec.bbox_shape_indices if args.prompt_mode == "bbox" else rec.point_shape_indices
            rec_prompt_idxs[rec_key] = list(prompt_idxs)
            total_prompts += len(prompt_idxs)

            if len(prompt_idxs) == 0:
                out_img = args.output_dir / rec.image_path.name
                out_json = args.output_dir / rec.json_path.name
                shutil.copy2(rec.image_path, out_img)
                out_json.write_text(json.dumps(rec.json_data, ensure_ascii=False, indent=2))
                written += 1
                continue

            for si in prompt_idxs:
                src_shape = rec.json_data.get("shapes", [])[si]
                label = src_shape.get("label", "vehicle")
                flags = src_shape.get("flags", {}) or {}
                b = shape_bounds(src_shape)
                if b is None:
                    fallback_prompts += 1
                    rec_repl[rec_key][si] = [src_shape]
                    continue

                cx = 0.5 * (float(b[0]) + float(b[2]))
                cy = 0.5 * (float(b[1]) + float(b[3]))
                crop, left, top = make_square_crop(img, cx, cy, int(args.crop_size))

                if args.prompt_mode == "bbox":
                    bp = shape_rect_to_box_and_point(src_shape)
                    if bp is None:
                        fallback_prompts += 1
                        rec_repl[rec_key][si] = [src_shape]
                        continue
                    box_abs, _ = bp
                    box_crop = clamp_box_to_crop(box_abs, left, top, int(args.crop_size))
                    if box_crop is None:
                        fallback_prompts += 1
                        rec_repl[rec_key][si] = [src_shape]
                        continue
                    prompt_tasks.append(
                        {
                            "rec": rec,
                            "shape_idx": si,
                            "src_shape": src_shape,
                            "label": label,
                            "flags": flags,
                            "crop_img": crop,
                            "crop_left": left,
                            "crop_top": top,
                            "image_width": iw,
                            "image_height": ih,
                            "box": box_crop,
                        }
                    )
                else:
                    p = shape_to_cog_point(src_shape)
                    if p is None:
                        fallback_prompts += 1
                        rec_repl[rec_key][si] = [src_shape]
                        continue
                    p_crop = clamp_point_to_crop(p, left, top, int(args.crop_size))
                    prompt_tasks.append(
                        {
                            "rec": rec,
                            "shape_idx": si,
                            "src_shape": src_shape,
                            "label": label,
                            "flags": flags,
                            "crop_img": crop,
                            "crop_left": left,
                            "crop_top": top,
                            "image_width": iw,
                            "image_height": ih,
                            "point": p_crop,
                        }
                    )

        for task_batch in chunked(prompt_tasks, args.image_batch_size):
            if len(task_batch) == 0:
                continue
            imgs = [t["crop_img"] for t in task_batch]
            if args.prompt_mode == "bbox":
                input_boxes = [[t["box"]] for t in task_batch]
                inputs = processor(images=imgs, input_boxes=input_boxes, return_tensors="pt")
            else:
                input_points = [[ [t["point"]] ] for t in task_batch]
                input_labels = [[[1]] for _ in task_batch]
                inputs = processor(images=imgs, input_points=input_points, input_labels=input_labels, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = sam(**inputs, multimask_output=True)

            orig_sizes_cpu, reshaped_sizes_cpu = resolve_postprocess_sizes(inputs, imgs)
            post_masks = post_process_masks_compat(
                image_processor=processor.image_processor,
                pred_masks_cpu=out.pred_masks.detach().cpu(),
                original_sizes_cpu=orig_sizes_cpu,
                reshaped_sizes_cpu=reshaped_sizes_cpu,
            )

            for bi, task in enumerate(task_batch):
                masks_i = normalize_mask_candidates(post_masks[bi], n_boxes=1)
                scores_i = normalize_scores(out.iou_scores[bi], n_boxes=1)
                best_idx = int(np.argmax(scores_i[0]))
                best_mask = (masks_i[0, best_idx] > 0).astype(np.uint8)
                polys = mask_to_polygons(best_mask, min_area=args.min_poly_area, epsilon_frac=args.poly_epsilon_frac)

                rec = task["rec"]
                rec_key = id(rec)
                si = int(task["shape_idx"])
                src_shape = task["src_shape"]
                label = task["label"]
                flags = task["flags"]
                left = int(task["crop_left"])
                top = int(task["crop_top"])

                if len(polys) == 0:
                    fallback_prompts += 1
                    rec_repl[rec_key][si] = [src_shape]
                    continue

                converted_prompts += 1
                iw = int(task["image_width"])
                ih = int(task["image_height"])
                ns = []
                for poly in polys:
                    full_poly = shift_and_clip_poly_to_image(poly, left, top, iw, ih)
                    ns.append(
                        {
                            "label": label,
                            "points": full_poly,
                            "group_id": src_shape.get("group_id"),
                            "shape_type": "polygon",
                            "flags": {**flags, "source": "sam2_box_to_poly"},
                        }
                    )
                rec_repl[rec_key][si] = ns

        for rec in batch:
            rec_key = id(rec)
            prompt_idxs = rec_prompt_idxs.get(rec_key, [])
            if len(prompt_idxs) == 0:
                continue

            d_new = deepcopy(rec.json_data)
            repl = rec_repl.get(rec_key, {})
            for si in prompt_idxs:
                if si not in repl:
                    repl[si] = [d_new["shapes"][si]]

            final_shapes: List[Dict] = []
            for si, s in enumerate(d_new.get("shapes", [])):
                if si in repl:
                    final_shapes.extend(repl[si])
                else:
                    final_shapes.append(s)

            d_new["shapes"] = final_shapes
            out_img = args.output_dir / rec.image_path.name
            out_json = args.output_dir / rec.json_path.name
            shutil.copy2(rec.image_path, out_img)
            out_json.write_text(json.dumps(d_new, ensure_ascii=False, indent=2))
            written += 1

    print("done")
    print(f"output_dir={args.output_dir}")
    print(f"prompt_mode={args.prompt_mode}")
    print(f"inference_mode={args.inference_mode}")
    if str(args.inference_mode).strip().lower() == "crop":
        print(f"crop_size={args.crop_size}")
    print(f"records_written={written}")
    print(f"prompts_total={total_prompts}")
    print(f"prompts_converted={converted_prompts}")
    print(f"prompts_fallback={fallback_prompts}")


if __name__ == "__main__":
    main()
