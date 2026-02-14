# CNN / ViT Explainer

This repo contains:
- `app.py`: CNN explainer (Gradio, ResNet18 walkthrough)
- `vit_explainer_app.py`: older Gradio ViT explorer
- `backend/main.py` + `frontend/`: main interactive ViT explorer (FastAPI + React/Vite)

## Repo Layout
- `backend/`: API for token extraction + similarity
- `frontend/`: UI for hover/bbox similarity exploration
- `app.py`, `vit_explainer_app.py`: standalone Gradio tools

## Prerequisites
- Python 3.10+
- Node 20+ and npm
- Internet on first run (Torch Hub model download)
- Optional for adapter mode: local WTCV repo at `/home/schelli/git/wtcv`

## Backend Install
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Frontend Install
```bash
cd frontend
npm install
```

## Start (Main App)
Run both services in separate terminals.

Terminal 1 (backend):
```bash
cd backend
source .venv/bin/activate  # if using venv
uvicorn main:app --host 127.0.0.1 --port 8000
```

Terminal 2 (frontend):
```bash
cd frontend
npm run dev
```

Open:
- Frontend: `http://127.0.0.1:5173`
- Backend: `http://127.0.0.1:8000`

## Adapter Workflow (Stage1SegNet)
1. In UI, set adapter checkpoint path, for example:
   - `/home/schelli/git/wtcv/runs/20260214-084834-pretrain2/checkpoints/final.pt`
2. Click `Load Adapter`
3. Enable `Apply loaded adapter during extraction`
4. Click `Extract From Path` or `Extract From Upload`

Notes:
- Without adapter: uses vanilla `dinov2_vits14_reg` (12 layers).
- With adapter: uses Stage1 fused feature (`fused = fuse_1x1(...)`) and returns one layer.
- Adapter mode can have denser grid (higher patch count).

## Run Gradio Apps (Optional)
CNN explainer:
```bash
python app.py
```
URL: `http://127.0.0.1:7861`

ViT Gradio explainer:
```bash
python vit_explainer_app.py
```
URL: `http://127.0.0.1:7862`

## Source-Only Push Checklist
This repo ignores generated folders/caches via `.gitignore`:
- `frontend/node_modules/`
- `frontend/dist/`
- `__pycache__/`
- Python cache/test artifacts

Before pushing:
```bash
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
rm -rf frontend/node_modules frontend/dist
```

Then commit only source/config/docs.

## If You Already Tracked Build Artifacts
If `node_modules`/`dist` were already committed once, untrack them:
```bash
git rm -r --cached frontend/node_modules frontend/dist
```
Then commit `.gitignore` and push.
