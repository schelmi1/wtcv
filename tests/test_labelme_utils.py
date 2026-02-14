from __future__ import annotations

from wtcv_utils.labelme import polygon_area, polygon_bbox, shape_to_points


def test_shape_to_points_rectangle() -> None:
    shape = {
        "shape_type": "rectangle",
        "points": [[8, 12], [20, 28]],
    }
    pts = shape_to_points(shape, min_poly_points=3)
    assert pts == [[8.0, 12.0], [20.0, 12.0], [20.0, 28.0], [8.0, 28.0]]


def test_shape_to_points_polygon_min_points() -> None:
    shape = {"shape_type": "polygon", "points": [[1, 1], [2, 2]]}
    assert shape_to_points(shape, min_poly_points=3) is None


def test_polygon_area_and_bbox() -> None:
    pts = [[10, 10], [20, 10], [20, 25], [10, 25]]
    assert polygon_bbox(pts) == (10.0, 10.0, 20.0, 25.0)
    assert polygon_area(pts) == 150.0

