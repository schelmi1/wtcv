from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Stage1SegNet, load_stage1_state_dict_compat


def _print_headline(adapter: str, head: str, tile_cls: bool, zoom_cls: bool) -> None:
    print(
        "\n"
        + "=" * 88
        + f"\nTEST CASE | adapter={adapter} | head={head} | tile_cls={tile_cls} | zoom_cls={zoom_cls}\n"
        + "=" * 88
    )


class _FakeDinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [nn.Identity() for _ in range(12)]

    def forward_features(self, x: torch.Tensor):
        b, _c, h, w = x.shape
        gh = max(1, h // 14)
        gw = max(1, w // 14)
        n = gh * gw
        tokens = torch.randn(b, n, 384, device=x.device, dtype=x.dtype)
        return {"x_norm_patchtokens": tokens}

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n=1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        _ = (reshape, return_class_token, norm)
        b, _c, h, w = x.shape
        gh = max(1, h // 14)
        gw = max(1, w // 14)
        nn_tokens = gh * gw
        if isinstance(n, int):
            idxs = list(range(max(0, 12 - int(n)), 12))
        else:
            idxs = [int(v) for v in n]
        out = []
        for i in idxs:
            t = torch.randn(b, nn_tokens, 384, device=x.device, dtype=x.dtype) + (float(i) * 1e-3)
            out.append(t)
        return tuple(out)


class _FakeAnyUp(nn.Module):
    def forward(self, hr_image: torch.Tensor, lr_features: torch.Tensor, q_chunk_size: int = 256):
        _ = q_chunk_size
        return F.interpolate(lr_features, size=hr_image.shape[-2:], mode="bilinear", align_corners=False)


class _FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


@pytest.fixture()
def mock_model_deps(monkeypatch: pytest.MonkeyPatch):
    ba = sys.modules["backbones_adapters"]

    def fake_hub_load(repo_or_dir, model, *args, **kwargs):
        _ = (args, kwargs)
        if repo_or_dir == "facebookresearch/dinov2":
            assert model == "dinov2_vits14_reg"
            return _FakeDinoBackbone()
        if repo_or_dir == "wimmerth/anyup":
            assert model == "anyup"
            return _FakeAnyUp()
        raise AssertionError(f"Unexpected torch.hub.load call: repo={repo_or_dir}, model={model}")

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    monkeypatch.setattr(ba.torchvision.models, "resnet18", lambda *a, **k: _FakeResNet())


@pytest.mark.parametrize("dino_upsampler_type", ["learned", "pixelshuffle"])
def test_stage1segnet_learned_forward_shapes(dino_upsampler_type: str, mock_model_deps) -> None:
    _print_headline(
        adapter=dino_upsampler_type,
        head="segmentation_head",
        tile_cls=True,
        zoom_cls=True,
    )
    model = Stage1SegNet(
        channels=64,
        dino_upsampler_type=dino_upsampler_type,
        use_tile_cls_head=True,
        use_zoom_cls_head=True,
    ).eval()

    x = torch.randn(2, 3, 256, 256)
    zoom_boxes = torch.tensor([[20.0, 30.0, 90.0, 110.0], [50.0, 60.0, 140.0, 170.0]], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x, zoom_boxes=zoom_boxes, trace_shapes=True)

    assert pred["seg_logit"].shape == (2, 1, 64, 64)
    assert pred["tile_logit"].shape == (2, 1)
    assert pred["zoom_logit"].shape == (2, 1)


@pytest.mark.parametrize("head_type", ["pointwise", "dwsep", "residual"])
def test_stage1segnet_anyup_forward_shapes(head_type: str, mock_model_deps) -> None:
    _print_headline(adapter="anyup", head=head_type, tile_cls=True, zoom_cls=True)
    model = Stage1SegNet(
        channels=64,
        dino_upsampler_type="anyup",
        head_type=head_type,
        use_tile_cls_head=True,
        use_zoom_cls_head=True,
    ).eval()

    x = torch.randn(2, 3, 256, 256)
    zoom_boxes = torch.tensor([[10.0, 10.0, 80.0, 80.0], [30.0, 25.0, 130.0, 150.0]], dtype=torch.float32)

    with torch.no_grad():
        pred = model(x, zoom_boxes=zoom_boxes, trace_shapes=True)

    assert pred["seg_logit"].shape == (2, 1, 64, 64)
    assert pred["tile_logit"].shape == (2, 1)
    assert pred["zoom_logit"].shape == (2, 1)


def test_stage1segnet_zoom_head_requires_boxes(mock_model_deps) -> None:
    _print_headline(adapter="learned", head="segmentation_head", tile_cls=True, zoom_cls=True)
    model = Stage1SegNet(
        channels=32,
        dino_upsampler_type="learned",
        use_tile_cls_head=True,
        use_zoom_cls_head=True,
    ).eval()
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        pred = model(x, zoom_boxes=None, trace_shapes=True)

    assert "zoom_logit" not in pred
    assert pred["seg_logit"].shape == (1, 1, 64, 64)
    assert pred["tile_logit"].shape == (1, 1)


def test_stage1segnet_rejects_non_256_multiple_input(mock_model_deps) -> None:
    _print_headline(adapter="learned", head="segmentation_head", tile_cls=False, zoom_cls=False)
    model = Stage1SegNet(
        channels=32,
        dino_upsampler_type="learned",
        use_tile_cls_head=False,
        use_zoom_cls_head=False,
    ).eval()
    x = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError, match="multiples of 256"):
        _ = model(x)


def test_compat_loader_interpolates_mismatched_spatial_weights(mock_model_deps) -> None:
    _print_headline(adapter="learned", head="segmentation_head", tile_cls=False, zoom_cls=False)
    model = Stage1SegNet(
        channels=32,
        dino_upsampler_type="learned",
        use_tile_cls_head=False,
        use_zoom_cls_head=False,
    ).eval()

    state = model.state_dict()
    # Simulate an "old style" conv kernel shape.
    state["head.seg_head.weight"] = torch.randn(1, 32, 3, 3)
    report = load_stage1_state_dict_compat(
        model,
        state,
        strict=False,
        interpolate_mismatch=True,
        verbose=False,
    )

    interp_keys = [k for (k, _src, _tgt) in report["interpolated"]]
    assert "head.seg_head.weight" in interp_keys
    assert model.head.seg_head.weight.shape == (1, 32, 1, 1)


def test_stage1segnet_multilayer_dino_forward_shapes(mock_model_deps) -> None:
    _print_headline(adapter="learned", head="segmentation_head", tile_cls=False, zoom_cls=False)
    model = Stage1SegNet(
        channels=64,
        dino_upsampler_type="learned",
        dino_layers="6,9,12",
        use_tile_cls_head=False,
        use_zoom_cls_head=False,
    ).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        pred = model(x, trace_shapes=True)
    assert pred["seg_logit"].shape == (1, 1, 64, 64)
