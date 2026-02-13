#!/usr/bin/env python3
import argparse
from copy import deepcopy
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class TargetRecord:
    image_path: Path
    json_path: Path
    json_data: Dict
    width: int
    height: int
    objects: List[Dict]


@dataclass
class DonorObject:
    image_rgba: np.ndarray  # HxWx4 (BGRA)
    mask: np.ndarray  # HxW uint8 (0/1)
    area: float
    bbox_hw: Tuple[int, int]
    label: str


@dataclass
class PasteResult:
    poly_points: List[List[float]]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Augment record_pairs by pasting polygon objects from donor dataset")
    ap.add_argument("--target-dir", type=Path, default=Path("data/record_pairs"))
    ap.add_argument("--donor-dir", type=Path, default=Path("data/sam_box_to_poly"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/record_pairs_augmented"))

    ap.add_argument("--target-label", type=str, default="vehicle")
    ap.add_argument("--donor-label", type=str, default="vehicle")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-pastes-per-image", type=int, default=1)
    ap.add_argument("--max-pastes-per-image", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=0, help="0 means all target images")
    ap.add_argument(
        "--donor-max-images",
        type=int,
        default=0,
        help="0 means all donor pairs; >0 limits donor json/image pairs loaded",
    )

    ap.add_argument("--placement-horizon-frac", type=float, default=0.35, help="Only place below this y-fraction")
    ap.add_argument("--max-placement-tries", type=int, default=40)
    ap.add_argument("--max-overlap-iou", type=float, default=0.15)

    ap.add_argument("--size-min-ratio", type=float, default=0.0001)
    ap.add_argument("--size-max-ratio", type=float, default=0.05)
    ap.add_argument("--min-poly-area", type=float, default=12.0)

    ap.add_argument("--feather-radius", type=int, default=3)
    ap.add_argument("--jpeg-quality-min", type=int, default=55)
    ap.add_argument("--jpeg-quality-max", type=int, default=92)
    ap.add_argument("--occlusion-prob", type=float, default=0.45)

    ap.add_argument("--overwrite", action="store_true", default=False)
    return ap.parse_args()


def find_image_for_json(data_dir: Path, stem: str) -> Optional[Path]:
    cands = [p for p in data_dir.glob(f"{stem}.*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not cands:
        return None
    cands.sort(key=lambda p: p.name)
    return cands[0]


def polygon_area(points: List[List[float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points], dtype=np.float32)
    y = np.array([p[1] for p in points], dtype=np.float32)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def shape_to_points(shape: Dict) -> List[List[float]]:
    st = str(shape.get("shape_type", "")).lower()
    pts = shape.get("points", []) or []
    if st == "rectangle" and len(pts) >= 2:
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        lx, rx = min(float(x0), float(x1)), max(float(x0), float(x1))
        ty, by = min(float(y0), float(y1)), max(float(y0), float(y1))
        return [[lx, ty], [rx, ty], [rx, by], [lx, by]]
    return [[float(p[0]), float(p[1])] for p in pts]


def load_target_records(data_dir: Path, label: str) -> List[TargetRecord]:
    label_cf = label.strip().casefold()
    recs: List[TargetRecord] = []
    for jf in tqdm(sorted(data_dir.glob("*.json")), desc="load_target_records", leave=True):
        ip = find_image_for_json(data_dir, jf.stem)
        if ip is None:
            continue
        d = json.loads(jf.read_text())
        w = int(d.get("imageWidth", 0) or 0)
        h = int(d.get("imageHeight", 0) or 0)
        if w <= 0 or h <= 0:
            continue

        objs = []
        for s in d.get("shapes", []) or []:
            if str(s.get("label", "")).strip().casefold() != label_cf:
                continue
            pts = shape_to_points(s)
            if len(pts) < 3:
                continue
            a = polygon_area(pts)
            if a <= 0:
                continue
            x0, y0, x1, y1 = polygon_bbox(pts)
            objs.append({"points": pts, "area": a, "bbox": [x0, y0, x1, y1]})

        recs.append(TargetRecord(ip, jf, d, w, h, objs))
    return recs


def mask_to_polygons(mask_u8: np.ndarray, min_area: float, epsilon_frac: float = 0.002) -> List[List[List[float]]]:
    m = (mask_u8 > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[List[float]]] = []
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


def build_donor_pool(
    data_dir: Path,
    label: str,
    min_poly_area: float,
    donor_max_images: int = 0,
    sample_seed: int = 42,
) -> List[DonorObject]:
    label_cf = label.strip().casefold()
    pool: List[DonorObject] = []
    donor_jsons = sorted(data_dir.glob("*.json"))
    if donor_max_images > 0:
        r = random.Random(sample_seed)
        if donor_max_images < len(donor_jsons):
            donor_jsons = r.sample(donor_jsons, donor_max_images)
        else:
            r.shuffle(donor_jsons)

    for jf in tqdm(donor_jsons, desc="build_donor_pool", leave=True):
        ip = find_image_for_json(data_dir, jf.stem)
        if ip is None:
            continue
        img_bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        d = json.loads(jf.read_text())
        h, w = img_bgr.shape[:2]
        best: Optional[DonorObject] = None

        for s in d.get("shapes", []) or []:
            if str(s.get("label", "")).strip().casefold() != label_cf:
                continue
            pts = shape_to_points(s)
            if len(pts) < 3:
                continue

            poly = np.array(pts, dtype=np.float32)
            x0 = max(0, int(math.floor(poly[:, 0].min())))
            y0 = max(0, int(math.floor(poly[:, 1].min())))
            x1 = min(w, int(math.ceil(poly[:, 0].max())))
            y1 = min(h, int(math.ceil(poly[:, 1].max())))
            if x1 <= x0 or y1 <= y0:
                continue

            local = poly.copy()
            local[:, 0] -= x0
            local[:, 1] -= y0

            crop = img_bgr[y0:y1, x0:x1]
            m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillPoly(m, [local.astype(np.int32)], 1)
            area = float(m.sum())
            if area < min_poly_area:
                continue

            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = (m * 255).astype(np.uint8)

            cand = DonorObject(
                image_rgba=rgba,
                mask=m,
                area=area,
                bbox_hw=(y1 - y0, x1 - x0),
                label=label,
            )
            if best is None or cand.area > best.area:
                best = cand

        # If multiple polygons exist in one donor json, keep only the largest one.
        if best is not None:
            pool.append(best)

    return pool


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (ax1 - ax0) * (ay1 - ay0))
    ba = max(1, (bx1 - bx0) * (by1 - by0))
    return float(inter / (aa + ba - inter))


def augment_object_appearance(obj_rgba: np.ndarray, rng: random.Random, jpeg_q_range: Tuple[int, int]) -> np.ndarray:
    out = obj_rgba.copy()
    bgr = out[:, :, :3].astype(np.float32)

    # Brightness/contrast and slight color shift.
    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.uniform(-18.0, 18.0)  # brightness
    bgr = bgr * alpha + beta

    # HSV jitter.
    hsv = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-5, 5)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.2), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.85, 1.2), 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Blur/noise.
    if rng.random() < 0.5:
        k = 3
        bgr = cv2.GaussianBlur(bgr, (k, k), sigmaX=rng.uniform(0.1, 1.1))
    if rng.random() < 0.4:
        noise = np.random.normal(0, rng.uniform(1.0, 5.0), bgr.shape).astype(np.float32)
        bgr = bgr + noise

    bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    # JPEG artifacts.
    qmin, qmax = jpeg_q_range
    if qmax >= qmin and rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.randint(qmin, qmax))])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                bgr = dec

    out[:, :, :3] = bgr
    return out


def feather_alpha(alpha_u8: np.ndarray, radius: int) -> np.ndarray:
    a = alpha_u8.astype(np.float32) / 255.0
    if radius <= 0:
        return a
    k = radius * 2 + 1
    a = cv2.GaussianBlur(a, (k, k), sigmaX=max(0.1, radius * 0.5))
    return np.clip(a, 0.0, 1.0)


def choose_paste_count(rng: random.Random, lo: int, hi: int) -> int:
    lo2 = max(0, min(lo, hi))
    hi2 = max(lo2, hi)
    return rng.randint(lo2, hi2)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if not args.target_dir.exists():
        raise FileNotFoundError(f"Missing target-dir: {args.target_dir}")
    if not args.donor_dir.exists():
        raise FileNotFoundError(f"Missing donor-dir: {args.donor_dir}")

    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(f"Output exists: {args.output_dir}. Use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_recs = load_target_records(args.target_dir, args.target_label)
    if args.max_images > 0:
        target_recs = target_recs[: args.max_images]

    if len(target_recs) == 0:
        raise RuntimeError("No target records found")

    donor_pool = build_donor_pool(
        args.donor_dir,
        args.donor_label,
        min_poly_area=args.min_poly_area,
        donor_max_images=args.donor_max_images,
        sample_seed=args.seed,
    )
    if len(donor_pool) == 0:
        raise RuntimeError("No donor polygon objects found")

    # Match target size distribution via annotation area ratios.
    ratios: List[float] = []
    obj_count_dist: List[int] = []
    for r in target_recs:
        img_area = float(max(1, r.width * r.height))
        obj_count_dist.append(len(r.objects))
        for o in r.objects:
            rr = float(o["area"]) / img_area
            if args.size_min_ratio <= rr <= args.size_max_ratio:
                ratios.append(rr)

    if len(ratios) == 0:
        # fallback safe tiny-object distribution
        ratios = [0.001, 0.002, 0.003, 0.005, 0.008]

    print(f"target_records={len(target_recs)} donor_objects={len(donor_pool)} size_ratio_samples={len(ratios)}")

    written = 0
    pasted_total = 0

    for rec in tqdm(target_recs, desc="augment_record_pairs", leave=True):
        img_bgr = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        base = img_bgr.copy()

        existing_boxes: List[Tuple[int, int, int, int]] = []
        for o in rec.objects:
            x0, y0, x1, y1 = o["bbox"]
            existing_boxes.append((int(x0), int(y0), int(x1), int(y1)))

        out_shapes = deepcopy(rec.json_data.get("shapes", []) or [])

        n_paste = choose_paste_count(rng, args.min_pastes_per_image, args.max_pastes_per_image)
        pasted_here = 0

        for _ in range(n_paste):
            d = donor_pool[rng.randrange(len(donor_pool))]

            # Sample target area ratio from real distribution.
            area_ratio = ratios[rng.randrange(len(ratios))]
            target_area = float(W * H) * area_ratio
            if target_area <= 1:
                continue

            scale = math.sqrt(target_area / max(1.0, d.area))
            new_h = max(3, int(round(d.bbox_hw[0] * scale)))
            new_w = max(3, int(round(d.bbox_hw[1] * scale)))
            if new_h >= H or new_w >= W:
                continue

            # Resize donor patch/mask.
            obj_rgba = cv2.resize(d.image_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            obj_rgba = augment_object_appearance(obj_rgba, rng, (args.jpeg_quality_min, args.jpeg_quality_max))
            mask = (obj_rgba[:, :, 3] > 0).astype(np.uint8)

            # Placement tries with plausible constraints.
            placed = False
            y_min = int(max(0, min(H - new_h, round(args.placement_horizon_frac * H))))
            for _try in range(args.max_placement_tries):
                x0 = rng.randint(0, max(0, W - new_w))
                y0 = rng.randint(y_min, max(y_min, H - new_h))
                x1, y1 = x0 + new_w, y0 + new_h

                # Keep away from extreme bright patches (likely sky/UI) even below horizon.
                roi = base[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                if float(roi.mean()) > 240.0:
                    continue

                cand_box = (x0, y0, x1, y1)
                if any(bbox_iou(cand_box, eb) > args.max_overlap_iou for eb in existing_boxes):
                    continue

                # Alpha blend with feathered boundary.
                alpha = feather_alpha((mask * 255).astype(np.uint8), radius=args.feather_radius)
                alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)

                obj_rgb = obj_rgba[:, :, :3].astype(np.float32)
                dst = base[y0:y1, x0:x1].astype(np.float32)
                comp = obj_rgb * alpha3 + dst * (1.0 - alpha3)
                base[y0:y1, x0:x1] = np.clip(comp, 0, 255).astype(np.uint8)

                # Optional occlusion: overlay a background patch segment on top.
                m_work = mask.copy()
                if rng.random() < args.occlusion_prob:
                    occ_w = max(2, int(round(new_w * rng.uniform(0.15, 0.45))))
                    occ_h = max(2, int(round(new_h * rng.uniform(0.15, 0.45))))
                    ox = rng.randint(0, max(0, new_w - occ_w))
                    oy = rng.randint(0, max(0, new_h - occ_h))
                    # bring back original background in occluder area
                    base[y0 + oy:y0 + oy + occ_h, x0 + ox:x0 + ox + occ_w] = img_bgr[y0 + oy:y0 + oy + occ_h, x0 + ox:x0 + ox + occ_w]
                    m_work[oy:oy + occ_h, ox:ox + occ_w] = 0

                # Convert pasted mask to polygon(s) in global coords.
                polys_local = mask_to_polygons(m_work, min_area=args.min_poly_area)
                if len(polys_local) == 0:
                    placed = True
                    existing_boxes.append(cand_box)
                    break

                for poly in polys_local:
                    poly_g = [[float(px + x0), float(py + y0)] for px, py in poly]
                    out_shapes.append(
                        {
                            "label": args.target_label,
                            "points": poly_g,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {"source": "paste_augment"},
                        }
                    )

                existing_boxes.append(cand_box)
                pasted_here += 1
                placed = True
                break

            if not placed:
                continue

        out_img_name = rec.image_path.name
        out_json_name = rec.json_path.name

        out_img = args.output_dir / out_img_name
        out_json = args.output_dir / out_json_name

        cv2.imwrite(str(out_img), base)

        d_new = deepcopy(rec.json_data)
        d_new["imagePath"] = out_img_name
        d_new["shapes"] = out_shapes
        out_json.write_text(json.dumps(d_new, ensure_ascii=False, indent=2))

        pasted_total += pasted_here
        written += 1

    print("done")
    print(f"output_dir={args.output_dir}")
    print(f"images_written={written}")
    print(f"pasted_objects_total={pasted_total}")


if __name__ == "__main__":
    main()
