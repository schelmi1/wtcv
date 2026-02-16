from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from fastapi import HTTPException
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.vit_explorer.backend import main as vmain


@pytest.fixture(autouse=True)
def _reset_vit_backend_state() -> None:
    vmain.SESSIONS.clear()
    vmain._ACTIVE_ADAPTER_PATH = None
    vmain._STAGE1_ADAPTERS.clear()


def _make_session() -> vmain.SessionData:
    # 2x2 patch grid over a 28x28 image.
    z = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    z = F.normalize(z, p=2, dim=1)
    return vmain.SessionData(
        session_id="s1",
        created_ts=0.0,
        width=28,
        height=28,
        grid_w=2,
        grid_h=2,
        image_data_url="data:image/png;base64,xx",
        token_norm_maps=[[1.0, 2.0, 3.0, 4.0]],
        tokens_normed=[z],
        adapter_name="none",
        adapter_checkpoint="",
        patch_count=4,
    )


def test_health_endpoint() -> None:
    assert vmain.health() == {"status": "ok"}


def test_adapter_current_default_unloaded() -> None:
    r = vmain.adapter_current()
    assert r["loaded"] is False
    assert r["adapter_name"] == "none"


def test_similarity_requires_existing_session() -> None:
    with pytest.raises(HTTPException) as ex:
        vmain.similarity(
            vmain.SimilarityRequest(
                session_id="missing",
                mode="point",
                point_x=2,
                point_y=3,
            )
        )
    assert ex.value.status_code == 404
    assert "Session not found" in str(ex.value.detail)


def test_similarity_point_and_bbox() -> None:
    s = _make_session()
    vmain.SESSIONS[s.session_id] = s

    point = vmain.similarity(
        vmain.SimilarityRequest(
            session_id=s.session_id,
            mode="point",
            point_x=1,
            point_y=1,
        )
    )
    assert point["session_id"] == s.session_id
    assert len(point["similarity_maps"]) == 1
    assert len(point["similarity_maps"][0]) == 4
    assert "mode=point" in point["info"]

    bbox = vmain.similarity(
        vmain.SimilarityRequest(
            session_id=s.session_id,
            mode="bbox",
            bbox_x1=6,
            bbox_y1=6,
            bbox_x2=20,
            bbox_y2=20,
        )
    )
    assert len(bbox["similarity_maps"]) == 1
    assert len(bbox["similarity_maps"][0]) == 4
    assert "selected_patches=" in bbox["info"]


def test_similarity_bbox_without_selected_patches_returns_400() -> None:
    s = _make_session()
    vmain.SESSIONS[s.session_id] = s

    with pytest.raises(HTTPException) as ex:
        vmain.similarity(
            vmain.SimilarityRequest(
                session_id=s.session_id,
                mode="bbox",
                bbox_x1=0,
                bbox_y1=0,
                bbox_x2=0,
                bbox_y2=0,
            )
        )
    assert ex.value.status_code == 400
    assert "BBox does not include any patch centers" in str(ex.value.detail)


def test_encode_path_uses_build_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    img_path = tmp_path / "img.png"
    Image.new("RGB", (40, 30), color=(10, 20, 30)).save(img_path)

    fake = _make_session()
    fake.width = 42
    fake.height = 56
    fake.grid_w = 3
    fake.grid_h = 4
    fake.patch_count = 12
    fake.token_norm_maps = [[0.1] * 12]

    def _fake_build_session(img, max_side: int, use_loaded_adapter: bool):
        assert img.mode == "RGB"
        assert max_side == 560
        assert use_loaded_adapter is False
        return fake

    monkeypatch.setattr(vmain, "_build_session", _fake_build_session)
    body = vmain.encode_path(vmain.EncodePathRequest(image_path=str(img_path)))
    assert body["session_id"] == fake.session_id
    assert body["image_width"] == 42
    assert body["image_height"] == 56
    assert body["grid_w"] == 3
    assert body["grid_h"] == 4
    assert body["layer_count"] == 1
    assert body["patch_count"] == 12


def test_encode_path_missing_file_returns_404(tmp_path: Path) -> None:
    p = tmp_path / "missing.png"
    with pytest.raises(HTTPException) as ex:
        vmain.encode_path(vmain.EncodePathRequest(image_path=str(p)))
    assert ex.value.status_code == 404
