# Tests

This folder contains the lightweight unit/integration tests for the WTCV codebase.

## Run tests

From repo root:

```bash
pytest -q tests
```

Run a single file:

```bash
pytest -q tests/test_models_forward.py
```

Show model shape traces during forward tests:

```bash
pytest -q -s tests/test_models_forward.py
```

## Test files

- `tests/test_labelme_utils.py`
  - Validates LabelMe shape parsing and geometry helpers.
  - Covers polygon area, bbox extraction, rectangle -> polygon conversion.

- `tests/test_records_loader.py`
  - Validates fast LabelMe pair loading and record building.
  - Covers missing/bad JSON handling, random sampling, case-insensitive labels, FP inclusion/exclusion, label discovery.

- `tests/test_tiling_utils.py`
  - Validates tiling helper behavior.
  - Covers right/bottom edge coverage and padded crops.

- `tests/test_seg_tile_dataset.py`
  - Validates segmentation tile dataset construction.
  - Covers full-bbox tile containment, 50/50 balancing behavior, and positive/negative/FP mask targets.

- `tests/test_models_forward.py`
  - Validates `Stage1SegNet` forward pass and output shapes.
  - Covers learned + anyup paths, tile/zoom heads, 256-multiple input constraint, and checkpoint compatibility interpolation.
  - Uses mocked `torch.hub.load` + mocked ResNet to avoid external downloads and keep tests fast.

- `tests/conftest.py`
  - Shared fixtures for synthetic LabelMe datasets and temporary test files.

## Notes

- Tests are designed to be CPU-friendly and deterministic enough for local development.
- Model tests intentionally use random tensors and fake backbones/upsamplers, so they check wiring and tensor shapes, not model quality.
