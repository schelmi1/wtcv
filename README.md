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
- `sam2_box_to_poly_batched.py`
- `fiftyone_object_umap.py`
- `build_embedding_bank.py`
- `unique_images_vs_embedding_bank.py`
- `score_dataset_with_embedding_bank.py`
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
  - Refinement (SAM1 + SAM2 bbox->polygon conversion)
  - Dataset augmentation
  - Data Curation Toolkit:
    - Curation launcher (OpenCV window)
    - Single-image inference
    - Dataset peek
    - Video -> frames
    - Full Dataset vs Embedding Bank
  - Media Inference (super tab):
    - Batched image-folder inference (tile dataloader, no UI)
    - Media-source inference (folder/video, optional UI)
    - Live screen inference (OpenCV UI)
  - Object UMAP + FiftyOne export
  - Embedding bank builder (from LabelMe folders)

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

FP balancing behavior (important):
- With `--balance-train-50-50`, train tiles are balanced so `pos_tiles == neg_tiles`.
- `--fp-neg-ratio` is applied only inside the selected negative half.
- Expected FP-negative count is approximately:
  `train_fp_neg_tiles ~= min(fp_neg_pool, round(train_neg_tiles * fp_neg_ratio))`
- Example: if `train_tiles=23k`, then `train_neg_tiles=11.5k`. With `--fp-neg-ratio 0.5`, expected `train_fp_neg_tiles` is about `5.7k` (assuming FP-negative pool is large enough).
- `train_fp_neg_tiles` counts tiles with `is_object=0` and `has_fp=1`. Tiles that contain both a true vehicle and `fp` annotations are counted as positive tiles, not FP-negative tiles.

Current dataset logging includes:
- pre-balance pools (`obj_pool`, `neg_pool`, `fp_neg_pool`, `pure_neg_pool`)
- selected FP negatives after balancing
- `train_fp_neg_frac_of_neg`

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
- `--prompt-mode {bbox,point}`: bbox prompts or center-of-gravity point prompts (for polygon datasets). Current default is `point`.
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
- Build masked object embeddings from LabelMe image/json pairs.
- Save reusable bank artifacts for similarity search / reference matching.
- Supports two backends:
  - `dino`: raw DINO patch-token features (fixed `dinov2_vits14_reg`)
  - `adapter`: Stage1 checkpoint features (`feat_adapted` or `feat_dino`)

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

Important args:
- `--feature-backend {dino,adapter}`
- `--adapter-checkpoint` (required when backend is `adapter`)
- `--adapter-feature-key {feat_adapted,feat_dino}`
- `--adapter-input-size` (optional; must be multiple of 256 for adapter backend)
- `--max-objects`
  - If `>0`, both `tile stream` and `dino masked pool iters` progress bars run with exact totals.
  - If `0`, script pre-counts valid objects to set progress total.

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

### 9b) Batched Image Folder Inference (tile dataloader)
Script: `batch_image_folder_inference.py`

Purpose:
- High-throughput inference for image folders by batching tiles across multiple images.
- Uses `torch.utils.data.DataLoader` + worker processes to decode/tiling in parallel.
- Stitches per-tile probabilities back to per-image maps and exports LabelMe image/json outputs.

Example:
```bash
python batch_image_folder_inference.py \
  --checkpoint /home/schelli/git/wtcv/runs/<run>/checkpoints/final.pt \
  --input-path /home/schelli/git/wtcv/data/record_pairs \
  --output-dir /home/schelli/git/wtcv/data/batch_inference_labelme \
  --tile-size 512 \
  --tile-stride 512 \
  --tile-batch-size 32 \
  --num-workers 8 \
  --amp
```

Common args:
- `--input-path`: image directory or single image file.
- `--tile-batch-size`: number of tiles per model forward pass.
- `--num-workers`: dataloader workers for image decode + tile emission.
- `--tile-size` (must be multiple of 256 for current model), `--tile-stride`, `--seg-out-stride`.
- `--pred-threshold`, `--min-poly-area`, `--poly-epsilon-frac`.
- `--use-tile-cls-gating`, `--tile-cls-threshold`, `--tile-cls-mode`.
- `--start-index`, `--max-images`, `--recursive`.
- `--save-empty` (also write empty-json outputs), `--overwrite`.

Outputs:
- LabelMe image/json pairs in `--output-dir`
- `batch_infer_summary.jsonl` (per-image stats)
- `batch_infer_summary.json` (run-level summary)

---

### 10) Object Cosine Similarity Report (LabelMe folder)
Script: `report_object_cosine_similarity.py`

Purpose:
- Compute masked DINO embeddings for objects in a LabelMe folder.
- Report highest and lowest cosine-similarity object pairs.
- Reduce near-duplicate scene bias by filtering image-pair candidates using scene-level cosine similarity.

Pipeline:
1. Extract and embed all filtered objects in the input folder.
2. Compute object-to-object cosine similarities.
3. Aggregate scores into source image-pair rows.
4. Compute scene/image embeddings for source images.
5. Filter image-pair rows with scene cosine above `--max-image-similarity`.
6. Export top/bottom pair reports and copied LabelMe pairs.

Example:
```bash
python report_object_cosine_similarity.py \
  --input-dir /home/schelli/git/wtcv/data/record_pairs \
  --output-dir /home/schelli/git/wtcv/outputs/embedding_similarity \
  --label-filter vehicle \
  --max-image-similarity 0.92 \
  --top-k 25
```

Outputs:
- `cosine_similarity_report.json` (summary + high/low pairs)
- `cosine_similarity_pairs.csv` (tabular high/low pairs)
- `cosine_similarity_embeddings.npz` (computed embeddings)
- `top_pairs/` (only copied LabelMe image/json pairs for highest-cosine pairs)
- `bottom_pairs/` (only copied LabelMe image/json pairs for lowest-cosine pairs)

Notes:
- This command does not preselect images before object embedding.
- Scene similarity is a post-filter on image-pair candidates after object embeddings are computed.
- `--max-image-similarity` controls that post-filter (lower = more scene diversity).
- If you want scene preselection first, use `unique_images_vs_embedding_bank.py` (section 11).

---

### 11) Unique Scenes -> Object-vs-Bank
Script: `unique_images_vs_embedding_bank.py`

Purpose:
- Rank new scenes by image-level uniqueness.
- Keep the most unique images.
- Score objects from these images against an existing embedding bank.

Pipeline:
1. Discover candidate images containing the requested labels.
2. Compute one scene embedding per candidate image.
3. Rank scene uniqueness using `--scene-knn`.
4. Keep the top unique subset (`--unique-keep-count` or `--unique-keep-ratio`).
5. Extract and embed objects only from selected images.
6. Score object novelty versus bank embeddings (`1 - max_bank_cos`).

Object embedding backend:
- `--feature-backend {dino,adapter}` for the object-vs-bank step.
- Scene uniqueness embeddings remain raw DINO image embeddings.

Example:
```bash
python unique_images_vs_embedding_bank.py \
  --input-dir /home/schelli/git/wtcv/data/record_pairs \
  --bank-npz /home/schelli/git/wtcv/outputs/embedding_bank/embedding_bank.npz \
  --output-dir /home/schelli/git/wtcv/outputs/unique_vs_bank \
  --label-filter vehicle \
  --scene-knn 5 \
  --unique-keep-ratio 0.30
```

Outputs:
- `scene_uniqueness.csv` (scene ranking by uniqueness)
- `selected_unique_images.txt`
- `selected_unique_labelme/` (copied valid LabelMe pairs for selected unique images)
- `objects_vs_bank.csv` (object novelty vs bank)
- `top_novel_objects.json`

---

### 12) Full Dataset vs Embedding Bank (Auto-add positives)
Script: `score_dataset_with_embedding_bank.py`

Purpose:
- Run model detections over a full image dataset.
- Score each detected polygon against a positive embedding-bank subset.
- Keep original LabelMe labels unchanged.
- Add only accepted positive candidates as new shapes.

Current strategy:
1. Load existing image/json pair (or create empty LabelMe json if missing).
2. Run tiled inference and polygon extraction.
3. Embed each predicted polygon (backend selectable):
   - `--feature-backend auto|dino|adapter`
   - `auto` resolves from embedding-bank manifest config.
4. Compute top-k cosine similarity vs positive bank subset.
5. Add candidate if score passes threshold and is not duplicate by IoU.

Label behavior:
- Added shapes are always written with `_auto` suffix:
  - e.g. `vehicle -> vehicle_auto`
- Existing labels are preserved.

Example:
```bash
python score_dataset_with_embedding_bank.py \
  --input-dir /home/schelli/git/wtcv/data/record_pairs \
  --checkpoint /home/schelli/git/wtcv/runs/<run>/checkpoints/final.pt \
  --embedding-bank /home/schelli/git/wtcv/outputs/embedding_bank/embedding_bank.npz \
  --output-dir /home/schelli/git/wtcv/outputs/dataset_vs_embedding_bank \
  --feature-backend auto \
  --positive-labels vehicle,vehicle_auto \
  --accept-score 0.35 \
  --dedup-iou 0.30
```

Key args:
- `--feature-backend {auto,dino,adapter}`
  - `adapter` uses the same `--checkpoint` as adapter source.
  - no separate adapter checkpoint in this workflow.
- `--adapter-feature-key {feat_adapted,feat_dino}`
- `--adapter-input-size` (optional; multiple of 256)
- `--positive-labels`
- `--bank-topk`
- `--accept-score`
- `--dedup-iou`
- `--vehicle-label` (base label before `_auto` suffixing)

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
python unique_images_vs_embedding_bank.py --help
python score_dataset_with_embedding_bank.py --help
python report_object_cosine_similarity.py --help
python fiftyone_export_tagged_to_labelme.py --help
python media_source_inference_cv2.py --help
python live_screen_inference_cv2.py --help
```
