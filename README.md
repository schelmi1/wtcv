# WTCV

War Thunder computer vision workspace for small-object vehicle segmentation with tiled inference.

## Codebase Map (Refactor Pass)

Shared utilities now live in:
- `wtcv_utils/labelme.py`
  - `IMG_EXTS`
  - `find_image_for_json(...)`
  - `shape_to_points(...)` (rectangle + polygon handling)
  - `polygon_area(...)`, `polygon_bbox(...)`
  - label casefold/match helpers
- `wtcv_utils/records.py`
  - `load_labelme_pairs(...)` (shared fast/multiprocess LabelMe pair loader)
  - `load_labelme_records(...)`
  - `discover_labels(...)`
- `wtcv_utils/tiling.py`
  - `tile_origins(...)`
  - `crop_with_pad(...)`

Core model/loss modules:
- `models.py`
  - `Stage1SegNet`
  - checkpoint compatibility loader `load_stage1_state_dict_compat(...)`
  - supports DINO upsamplers: `learned`, `pixelshuffle`, `anyup`
  - supports DINO layer selection (`last` or e.g. `6,9,12`)
- `backbones_adapters.py`
  - frozen DINO token branch (single-layer and multi-layer channel fusion)
  - learned/pixelshuffle/AnyUp upsamplers
  - local ResNet branch
- `heads.py`
  - segmentation heads (`pointwise`, `dwsep`, `residual`)
  - tile classifier head
  - zoom ROI classifier head
- `losses.py`
  - `mcc_bce_boundary_loss(...)`
  - `segmentation_metrics(...)`
  - MCC/boundary helper losses

Scripts migrated to use these shared helpers:
- `wtcv_gradio_app.py`
- `augment_record_pairs_with_polygons.py`
- `sam1_box_to_poly_batched.py`
- `fiftyone_object_umap.py`
- `build_embedding_bank.py`
- `fiftyone_export_tagged_to_labelme.py`
- `media_source_inference_cv2.py`
- `train_stage1_seg.py`
- `eval_stage1_seg.py`
- `curate_model_predictions_to_labelme.py`

This removes duplicated geometry/path logic, centralizes model/loss definitions, and keeps app + data tools consistent.

## WTCV Studio (Gradio App)

Script: `wtcv_gradio_app.py`

Purpose:
- Browser UI that unifies the main workflows:
  - Train
  - Evaluate
  - SAM bbox->polygon conversion
  - Dataset augmentation
  - Curation launcher (OpenCV window)
  - Media-source inference (folder/video, optional UI)
  - Live screen inference (OpenCV UI)
  - Object UMAP + FiftyOne export
  - Embedding bank builder (from LabelMe folders)
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

## ViT Explorer Tool (Isolated)

Location:
- `tools/vit_explorer/`

Purpose:
- Interactive ViT token explorer (FastAPI backend + React/Vite frontend).
- Supports vanilla `dinov2_vits14_reg` tokens and optional Stage1 fused adapter features from WTCV checkpoints.

Install and run:
```bash
# backend
cd tools/vit_explorer/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

```bash
# frontend (new terminal)
cd tools/vit_explorer/frontend
npm install
npm run dev
```

Open:
- Frontend: `http://127.0.0.1:5173`
- Backend: `http://127.0.0.1:8000`

Notes:
- Kept isolated from existing WTCV training/inference code paths to avoid regressions.
- `node_modules` and frontend build artifacts are ignored by `.gitignore`.

## Main Entry Points

### 1) Train stage-1 segmentation
Script: `train_stage1_seg.py`

Purpose:
- Train the stage-1 segmentation model on LabelMe image/json pairs.
- Supports multi-layer DINO tokens, learned/pixelshuffle/AnyUp upsamplers, tile/zoom heads, balancing, HNM, and TensorBoard logging.

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
- `--fp-label`: false-positive annotation label (default `fp`, case-insensitive).
- `--subset-size`: final train tile subset size (applied after tile build + balancing).
- `--batch-size`, `--num-workers`.
- `--tile-size`, `--tile-stride`, `--tile-scales`.
  - Current default is `256/128`.
  - Model expects tile height/width to be multiples of `256` (DINO path uses internal `14/16` downscale).
- `--dino-upsampler {learned,pixelshuffle,anyup}`.
- `--dino-layers`: `last` (default) or comma-separated 1-based layers like `6,9,12`.
- `--head-type {pointwise,dwsep,residual}`.
- `--use-tile-cls-head` / `--no-use-tile-cls-head`.
- `--use-fp-supervision` / `--no-use-fp-supervision`.
- `--fp-neg-weight`: weight for FP-region suppression loss.
- `--fp-neg-ratio`: in 50/50 balancing, target fraction of negatives drawn from `fp` tiles.
- `--balance-train-50-50`, `--balance-val-50-50`.
- `--hard-negative-mining`, `--hnm-hard-ratio`, `--hnm-pool-frac`.
- `--lr`, `--lr-scheduler {none,cosine}`, `--lr-min`, `--weight-decay`.
- `--mcc-weight`, `--mcc-warmup-epochs`, `--bce-weight`, `--boundary-weight`.
- `--training-strategy {task_only,semantic_preserve}`.
- Semantic-preserve knobs:
  - `--preserve-weight`, `--preserve-warmup-epochs`
  - `--preserve-bg-weight`, `--preserve-fg-weight`
  - `--var-weight`, `--var-gamma`

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
- `--prompt-mode {bbox,point}`: bbox prompts or center-of-gravity point prompts (for polygon datasets).
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
- `--poly-epsilon-frac`: polygon simplification for pasted masks (`0` keeps raw contours).
- `--occlusion-prob`, `--feather-radius`, JPEG quality args.

---

### 6) Object Embeddings + UMAP in FiftyOne
Script: `fiftyone_object_umap.py`

Purpose:
- Build one sample per object from LabelMe pairs.
- Compute DINO masked object embeddings on object-centric tiles.
- Run UMAP + KMeans and write a FiftyOne dataset for fast clustered review.

Example:
```bash
python fiftyone_object_umap.py \
  --input-dir /home/schelli/git/wtcv/data/record_pairs \
  --dataset-name wtcv_object_umap \
  --output-dir /home/schelli/git/wtcv/outputs/fiftyone_object_umap \
  --label-filter vehicle,fp \
  --tile-size 448 \
  --tile-context-scale 2.0 \
  --num-clusters 20
```

Common args:
- `--input-dir`: LabelMe pairs folder.
- `--dataset-name`: target FiftyOne dataset name.
- `--output-dir`: saved object crops + `embeddings_umap.npz`.
- `--label-filter`: comma-separated labels, case-insensitive.
- `--tile-size`, `--tile-context-scale`.
- DINO backbone is fixed to `dinov2_vits14_reg` in app/script workflows, plus `--batch-size`, `--device`.
- `--umap-n-neighbors`, `--umap-min-dist`, `--umap-metric`.
- `--num-clusters`: KMeans clusters in UMAP space.
- `--overwrite-dataset`, `--launch`.

---

### 7) Export Tagged FiftyOne Objects -> LabelMe
Script: `fiftyone_export_tagged_to_labelme.py`

Purpose:
- Read object-level FiftyOne samples (from `fiftyone_object_umap.py`).
- Use `sample.tags` as labels (`vehicle`, `fp`, etc.).
- Merge objects back by source image and export LabelMe image/json pairs.

Example:
```bash
python fiftyone_export_tagged_to_labelme.py \
  --dataset-name wtcv_object_umap \
  --output-dir /home/schelli/git/wtcv/data/umap_filtered_dataset \
  --tag-labels vehicle,fp
```

Common args:
- `--dataset-name`: source FiftyOne dataset name.
- `--output-dir`: merged LabelMe output folder.
- `--tag-labels`: allowed tags in priority order (comma-separated).
- `--overwrite`: replace existing output folder.

---

### 8) Build Embedding Bank (LabelMe folder -> reusable vectors)
Script: `build_embedding_bank.py`

Purpose:
- Build masked DINO object embeddings from LabelMe image/json pairs.
- Save reusable bank artifacts for similarity search / reference matching.

Example:
```bash
python build_embedding_bank.py \
  --input-dir /home/schelli/git/wtcv/data/record_pairs \
  --output-dir /home/schelli/git/wtcv/outputs/embedding_bank \
  --label-filter vehicle \
  --tile-size 448 \
  --tile-context-scale 2.0
```

Outputs:
- `embedding_bank.npz` (embeddings + numeric label/object fields)
- `embedding_bank_meta.jsonl` (per-object metadata rows)
- `embedding_prototypes.npz` (mean normalized prototype per label)
- `embedding_bank_manifest.json` (run config + file index)

---

### 9) Media Source Inference (folder or video)
Script: `media_source_inference_cv2.py`

Purpose:
- Run tiled inference on either:
  - an image folder, or
  - a video file
- Supports optional OpenCV UI or headless mode (`--no-ui`, default).

Example:
```bash
python media_source_inference_cv2.py \
  --checkpoint /home/schelli/git/wtcv/runs/<run>/checkpoints/final.pt \
  --input-path /home/schelli/git/wtcv/videos/frames_2fps \
  --output-dir /home/schelli/git/wtcv/data/media_inference_labelme \
  --no-ui
```

Common args:
- `--input-path`: image folder or video file.
- `--ui` / `--no-ui` (default off).
- `--auto-save`, `--save-empty`, `--save-preview`.
- `--tile-size`, `--tile-stride`, `--pred-threshold`.
- `--use-tile-cls-gating`, `--tile-cls-threshold`, `--tile-cls-mode`.
- `--infer-every`, `--max-fps`, `--start-index`, `--max-items`.

---

### 10) Object Cosine Similarity Report (LabelMe folder)
Script: `report_object_cosine_similarity.py`

Purpose:
- Compute masked DINO embeddings for objects in a LabelMe folder.
- Report highest and lowest cosine-similarity object pairs.

Example:
```bash
python report_object_cosine_similarity.py \
  --input-dir /home/schelli/git/wtcv/data/record_pairs \
  --output-dir /home/schelli/git/wtcv/outputs/embedding_similarity \
  --label-filter vehicle \
  --top-k 25
```

Outputs:
- `cosine_similarity_report.json` (summary + high/low pairs)
- `cosine_similarity_pairs.csv` (tabular high/low pairs)
- `cosine_similarity_embeddings.npz` (computed embeddings)
- `top_pairs/` (only copied LabelMe image/json pairs for highest-cosine pairs)
- `bottom_pairs/` (only copied LabelMe image/json pairs for lowest-cosine pairs)

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
python fiftyone_object_umap.py --help
python build_embedding_bank.py --help
python report_object_cosine_similarity.py --help
python fiftyone_export_tagged_to_labelme.py --help
python media_source_inference_cv2.py --help
python live_screen_inference_cv2.py --help
```
