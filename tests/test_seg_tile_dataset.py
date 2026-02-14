from __future__ import annotations

from pathlib import Path

from train_stage1_seg import SegTileDataset, objects_in_tile
from wtcv_utils.records import load_labelme_records


def test_objects_in_tile_requires_full_bbox_containment() -> None:
    objs = [
        {
            "label_cf": "vehicle",
            "is_fp": False,
            "points": [[200, 200], [230, 200], [230, 230], [200, 230]],
            "shape_type": "polygon",
            "bbox_xyxy": [200.0, 200.0, 230.0, 230.0],
            "center_xy": [215.0, 215.0],
            "poly_area": 900.0,
        }
    ]
    # Tile is 0..224 => object extends outside; should be excluded.
    got = objects_in_tile(objs, x0=0, y0=0, size=224)
    assert got == []


def test_seg_tile_dataset_balancing_and_mask_targets(seg_dataset_dir: Path) -> None:
    records = load_labelme_records(
        seg_dataset_dir,
        label_name="vehicle",
        min_poly_points=3,
        include_fp=True,
        fp_label="fp",
        load_workers=1,
    )
    ds = SegTileDataset(
        records=records,
        tile_size=224,
        tile_configs=[(224, 224)],
        seg_out_stride=4,
        seed=42,
        balance_50_50=True,
        fp_neg_ratio=1.0,
        augment_low_vis=False,
        is_train=True,
        dataset_name="test",
        verbose=False,
    )

    # One positive + one sampled negative in balanced mode.
    assert len(ds.pos_dataset_indices) == 1
    assert len(ds.neg_dataset_indices) == 1
    assert len(ds) == 2

    # At least one negative should come from FP-only tile when fp_neg_ratio=1.0.
    neg_samples = [ds.samples[i] for i in ds.neg_dataset_indices]
    assert any(int(s.get("has_fp", 0)) == 1 for s in neg_samples)

    # Positive sample has non-empty seg target.
    pos_idx = ds.pos_dataset_indices[0]
    pos_item = ds[pos_idx]
    assert float(pos_item["tile_target"].item()) == 1.0
    assert float(pos_item["seg_target"].sum().item()) > 0.0

    # FP-negative sample has empty seg target and non-empty fp target.
    neg_idx = ds.neg_dataset_indices[0]
    neg_item = ds[neg_idx]
    assert float(neg_item["tile_target"].item()) == 0.0
    assert float(neg_item["seg_target"].sum().item()) == 0.0
    assert float(neg_item["fp_target"].sum().item()) > 0.0

