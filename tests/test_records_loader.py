from __future__ import annotations

from pathlib import Path

from wtcv_utils.records import discover_labels, load_labelme_pairs, load_labelme_records


def test_load_labelme_pairs_filters_missing_and_bad_json(labelme_pairs_dir: Path) -> None:
    pairs = load_labelme_pairs(labelme_pairs_dir, load_workers=2, progress_leave=False)
    names = [p.json_path.name for p in pairs]
    assert names == ["a.json", "b.json"]


def test_load_labelme_pairs_max_images_random_sample(labelme_pairs_dir: Path) -> None:
    pairs = load_labelme_pairs(
        labelme_pairs_dir,
        load_workers=1,
        max_images=1,
        random_sample=True,
        sample_seed=123,
        progress_leave=False,
    )
    assert len(pairs) == 1
    assert pairs[0].json_path.name in {"a.json", "b.json"}


def test_load_labelme_records_case_insensitive_and_fp(labelme_pairs_dir: Path) -> None:
    recs = load_labelme_records(
        labelme_pairs_dir,
        label_name="vehicle",
        min_poly_points=3,
        include_fp=True,
        fp_label="fp",
        load_workers=1,
    )
    assert len(recs) == 2
    # a.json has one vehicle + one fp
    rec_a = next(r for r in recs if r["json_path"].endswith("a.json"))
    assert len(rec_a["objects"]) == 2
    assert any(o["is_fp"] for o in rec_a["objects"])
    assert any(not o["is_fp"] for o in rec_a["objects"])


def test_load_labelme_records_without_fp(labelme_pairs_dir: Path) -> None:
    recs = load_labelme_records(
        labelme_pairs_dir,
        label_name="vehicle",
        min_poly_points=3,
        include_fp=False,
        load_workers=1,
    )
    rec_a = next(r for r in recs if r["json_path"].endswith("a.json"))
    assert all(not o["is_fp"] for o in rec_a["objects"])
    assert len(rec_a["objects"]) == 1


def test_discover_labels(labelme_pairs_dir: Path) -> None:
    labels = discover_labels(labelme_pairs_dir)
    assert "Vehicle" in labels
    assert "FP" in labels
    assert "vehicle" in labels

