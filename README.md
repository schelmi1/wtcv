# WTCV

War Thunder computer vision workspace for small-object vehicle segmentation with tiled inference.

## WTCV Studio (Gradio App)

Script: `wtcv_gradio_app.py`

Purpose:
- Browser UI that unifies the main workflows:
  - Train
  - Evaluate
  - SAM bbox->polygon conversion
  - Dataset augmentation
  - Curation launcher (OpenCV window)
  - Single-image tiled inference preview
  - Dataset peek/stats
  - Video -> frames extraction

Install:
```bash
pip install gradio
```

Run:
```bash
python wtcv_gradio_app.py --host 127.0.0.1 --port 7860
```

## Main Entry Points

### 1) Train stage-1 segmentation
Script: `train_stage1_seg.py`

Purpose:
- Train the stage-1 segmentation model on LabelMe image/json pairs.
- Supports learned upsampler or AnyUp, tile classification head, balancing, HNM, and TensorBoard logging.

Minimal example:
```bash
python train_stage1_seg.py \
  --data-dir /home/schelli/git/wtcv/data/vehicle_pretrain \
  --run-name pretrain_run
```

Common args:
- `--data-dir`: LabelMe pairs folder.
- `--output-dir`: run output root (default `runs` layout).
- `--run-name`: run name suffix.
- `--resume-checkpoint`: resume from checkpoint.
- `--epochs`: number of epochs (resume adds on top of loaded epoch).
- `--label`: target label (case-insensitive).
- `--subset-size`: limit records before train/val split.
- `--batch-size`, `--num-workers`.
- `--tile-size`, `--tile-stride`, `--tile-scales`.
- `--dino-upsampler {learned,anyup}`.
- `--head-type {pointwise,dwsep,residual}`.
- `--use-tile-cls-head` / `--no-use-tile-cls-head`.
- `--balance-train-50-50`, `--balance-val-50-50`.
- `--hard-negative-mining`, `--hnm-hard-ratio`, `--hnm-pool-frac`.
- `--lr`, `--lr-scheduler {none,cosine}`, `--lr-min`, `--weight-decay`.
- `--mcc-weight`, `--mcc-warmup-epochs`, `--bce-weight`, `--boundary-weight`.

---

### 2) Evaluate stage-1 checkpoint
Script: `eval_stage1_seg.py`

Purpose:
- Evaluate trained checkpoint on LabelMe dataset.
- Reports IoU/AP-style metrics by object scales.

Example:
```bash
python eval_stage1_seg.py \
  --data-dir /home/schelli/git/wtcv/data/record_pairs \
  --checkpoint /home/schelli/git/wtcv/runs/<run>/checkpoints/last.pt \
  --label-name vehicle
```

Common args:
- `--data-dir`: evaluation dataset.
- `--checkpoint`: model checkpoint path (required).
- `--label-name`: target label (case-insensitive).
- `--tile-size`, `--tile-stride`, `--seg-out-stride`.
- `--pred-threshold`: mask threshold.
- `--num-workers`: dataloader workers.

---

### 3) Interactive curation from model predictions
Script: `curate_model_predictions_to_labelme.py`

Purpose:
- Run tiled inference on image folders and manually accept/reject detections.
- Save accepted predictions as LabelMe image/json pairs.

Example:
```bash
python curate_model_predictions_to_labelme.py \
  --input-dir /home/schelli/git/wtcv/videos/<frames_folder> \
  --checkpoint /home/schelli/git/wtcv/runs/<run>/checkpoints/last.pt \
  --output-dir /home/schelli/git/wtcv/data/helo_curated_labelme \
  --label vehicle
```

Common args:
- `--input-dir`: source image folder (required).
- `--checkpoint`: trained model (required).
- `--output-dir`: accepted LabelMe pairs output.
- `--pred-threshold`.
- `--use-tile-cls-gating`, `--tile-cls-threshold`, `--tile-cls-mode {hard,multiply}`.
- `--min-poly-area`, `--poly-epsilon-frac`.
- `--start-index`, `--max-images`.

---

### 4) Batched SAM1 bbox -> polygon conversion
Script: `sam1_box_to_poly_batched.py`

Purpose:
- Convert LabelMe rectangle annotations into polygon masks using SAM1 in batches.

Example:
```bash
python sam1_box_to_poly_batched.py \
  --input-dir /home/schelli/git/wtcv/data/war_thunder_v1_test1_labelme_pairs \
  --output-dir /home/schelli/git/wtcv/data/sam_box_to_poly \
  --image-batch-size 4 \
  --load-workers 8 \
  --overwrite
```

Common args:
- `--input-dir`: LabelMe pairs with rectangle shapes.
- `--output-dir`: converted pairs destination.
- `--model-id`: HF SAM model id (default `facebook/sam-vit-base`).
- `--device`: `cuda` or `cpu`.
- `--image-batch-size`: images per SAM forward.
- `--load-workers`: parallel JSON/image pair discovery.
- `--min-poly-area`, `--poly-epsilon-frac`.
- `--max-images`: debug subset.

---

### 5) Synthetic augmentation by object pasting
Script: `augment_record_pairs_with_polygons.py`

Purpose:
- Paste donor polygon objects onto `record_pairs`-style targets.
- Keeps target dataset untouched and writes augmented LabelMe pairs.

Example:
```bash
python augment_record_pairs_with_polygons.py \
  --target-dir /home/schelli/git/wtcv/data/record_pairs \
  --donor-dir /home/schelli/git/wtcv/data/sam_box_to_poly \
  --output-dir /home/schelli/git/wtcv/data/record_pairs_augmented \
  --target-label vehicle \
  --donor-label vehicle \
  --overwrite
```

Common args:
- `--target-dir`: base dataset to augment.
- `--donor-dir`: polygon donor dataset.
- `--output-dir`: augmented output.
- `--target-label`, `--donor-label` (case-insensitive).
- `--min-pastes-per-image`, `--max-pastes-per-image`.
- `--size-min-ratio`, `--size-max-ratio`: pasted object area ratio limits.
- `--placement-horizon-frac`, `--max-overlap-iou`, `--max-placement-tries`.
- `--occlusion-prob`, `--feather-radius`, JPEG quality args.

## Notebooks

Primary notebook entry points:
- `record_pairs_single_image_tiled_inference.ipynb`: single image tiled inference and reconstruction.
- `sam1_bbox_to_polygon_labelme.ipynb`: exploratory SAM bbox-to-polygon workflow.
- `war_thunder_bbox_distribution.ipynb`: bbox-area distribution analysis.

Recommendation: use scripts for production runs and notebooks for exploration/visual QA.

## Data and Outputs

- Datasets: `data/...` (LabelMe pairs and converted COCO datasets).
- Training runs: `runs/<timestamp>-<name>/`
  - `config.json`
  - `history.json`
  - checkpoints
  - TensorBoard event files

## Quick Help

For full flag list:
```bash
python train_stage1_seg.py --help
python eval_stage1_seg.py --help
python curate_model_predictions_to_labelme.py --help
python sam1_box_to_poly_batched.py --help
python augment_record_pairs_with_polygons.py --help
```
