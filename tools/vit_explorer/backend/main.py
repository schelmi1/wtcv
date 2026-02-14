#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

PATCH_SIZE = 14
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
MAX_SESSIONS = 12
# Resolve the wtcv repo root from this file location:
# tools/vit_explorer/backend/main.py -> parents[3] == repo root.
WTCV_REPO = Path(__file__).resolve().parents[3]

_MODEL: torch.nn.Module | None = None
_STAGE1_ADAPTERS: Dict[str, "Stage1Adapter"] = {}
_ACTIVE_ADAPTER_PATH: Optional[str] = None


@dataclass
class SessionData:
    session_id: str
    created_ts: float
    width: int
    height: int
    grid_w: int
    grid_h: int
    image_data_url: str
    token_norm_maps: List[List[float]]
    tokens_normed: List[torch.Tensor]  # each [N, C], l2-normalized
    adapter_name: str
    adapter_checkpoint: str
    patch_count: int


SESSIONS: Dict[str, SessionData] = {}


class EncodePathRequest(BaseModel):
    image_path: str = Field(..., min_length=1)
    max_side: int = Field(560, ge=224, le=1600)
    apply_loaded_adapter: bool = False


class SimilarityRequest(BaseModel):
    session_id: str
    mode: str = Field("point", pattern="^(point|bbox)$")
    point_x: int = 0
    point_y: int = 0
    bbox_x1: int = 0
    bbox_y1: int = 0
    bbox_x2: int = 0
    bbox_y2: int = 0


class AdapterLoadRequest(BaseModel):
    checkpoint_path: str = Field(..., min_length=1)


@dataclass
class Stage1Adapter:
    checkpoint_path: str
    adapter_name: str
    model: torch.nn.Module
    concat_layers: int


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _import_wtcv_models():
    repo = str(WTCV_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from models import Stage1SegNet, load_stage1_state_dict_compat  # type: ignore

    return Stage1SegNet, load_stage1_state_dict_compat


def _load_dino_reg() -> torch.nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=True)
    model.eval()
    _MODEL = model
    return model


def _load_stage1_adapter(checkpoint_path: str) -> Stage1Adapter:
    ckp = str(Path(checkpoint_path).expanduser().resolve())
    if ckp in _STAGE1_ADAPTERS:
        return _STAGE1_ADAPTERS[ckp]

    p = Path(ckp)
    if not p.is_file():
        raise RuntimeError(f"Adapter checkpoint does not exist: {p}")

    Stage1SegNet, load_stage1_state_dict_compat = _import_wtcv_models()

    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported checkpoint format: expected dict.")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint format: missing model state dict.")
    cfg = ckpt.get("cfg", {})

    model = Stage1SegNet(
        channels=int(_cfg_get(cfg, "fusion_channels", 256)),
        trust_repo=bool(_cfg_get(cfg, "trust_torch_hub_repo", True)),
        dino_upsampler_type=str(_cfg_get(cfg, "dino_upsampler_type", "learned")),
        anyup_q_chunk_size=int(_cfg_get(cfg, "anyup_q_chunk_size", 256)),
        head_type=str(_cfg_get(cfg, "head_type", "pointwise")),
        dino_layers=_cfg_get(cfg, "dino_layers", "last"),
        use_tile_cls_head=bool(_cfg_get(cfg, "use_tile_cls_head", False)),
        use_zoom_cls_head=bool(_cfg_get(cfg, "use_zoom_cls_head", False)),
    )
    load_stage1_state_dict_compat(
        model,
        state,
        strict=False,
        interpolate_mismatch=True,
        verbose=False,
    )
    model.eval()
    for p_model in model.parameters():
        p_model.requires_grad = False

    if getattr(model, "local", None) is None or getattr(model, "fuse_1x1", None) is None:
        raise RuntimeError(
            "This checkpoint/config uses a Stage1 path without local+fuse features. "
            "The explainer expects fused features before segmentation head."
        )
    if str(getattr(model, "dino_upsampler_type", "")) == "anyup":
        raise RuntimeError(
            "anyup Stage1 config is not supported in this explainer for fused-token extraction. "
            "Use a checkpoint with learned/pixelshuffle dino_upsampler_type."
        )

    in_ch = int(model.dino.proj.in_channels)
    if in_ch % 384 != 0:
        raise RuntimeError(f"Unexpected dino.proj.in_channels={in_ch}; expected multiple of 384.")
    concat_layers = in_ch // 384
    out_ch = int(model.dino.proj.out_channels)
    adapter_name = f"Stage1 fused ({in_ch}->{out_ch}, k={concat_layers}, up={model.dino_upsampler_type})"

    ad = Stage1Adapter(
        checkpoint_path=ckp,
        adapter_name=adapter_name,
        model=model,
        concat_layers=concat_layers,
    )
    _STAGE1_ADAPTERS[ckp] = ad
    return ad


def _resize_to_patch_grid(img: Image.Image, max_side: int = 560) -> Image.Image:
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    nw = max(PATCH_SIZE, int(round(w * scale)))
    nh = max(PATCH_SIZE, int(round(h * scale)))
    nw = max(PATCH_SIZE, (nw // PATCH_SIZE) * PATCH_SIZE)
    nh = max(PATCH_SIZE, (nh // PATCH_SIZE) * PATCH_SIZE)
    return img.resize((nw, nh), Image.BILINEAR)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(DINO_MEAN, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(DINO_STD, dtype=x.dtype).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def _flatten_tensors(obj: Any) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if torch.is_tensor(obj):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_tensors(v))
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_tensors(v))
    return out


def _extract_patch_tokens_from_obj(obj: Any, expected_patches: int) -> torch.Tensor:
    tensors = _flatten_tensors(obj)
    best: torch.Tensor | None = None
    for t in tensors:
        if t.ndim == 4:
            b, _c, h, w = t.shape
            if b == 1 and (h * w) == expected_patches:
                return t.flatten(2).transpose(1, 2)  # [1, N, C]
        if t.ndim == 3:
            b, n, _c = t.shape
            if b != 1:
                continue
            if n == expected_patches:
                return t
            if n > expected_patches:
                # For reg models: cls/reg tokens come first, patch tokens at tail.
                return t[:, n - expected_patches :, :]
            if best is None or n > best.shape[1]:
                best = t
    if best is None or best.shape[1] < expected_patches:
        raise RuntimeError("Could not locate patch-token tensor in intermediate layer output.")
    return best[:, best.shape[1] - expected_patches :, :]


def _get_all_layer_patch_tokens(model: torch.nn.Module, x: torch.Tensor) -> Tuple[List[torch.Tensor], int, int]:
    gh = int(x.shape[-2] // PATCH_SIZE)
    gw = int(x.shape[-1] // PATCH_SIZE)
    expected = int(gh * gw)
    calls: Sequence[Dict[str, Any]] = (
        {"n": 12, "return_class_token": True},
        {"n": 12, "return_class_token": False},
        {"n": 12},
        {"n": list(range(12)), "return_class_token": True},
        {"n": list(range(12)), "return_class_token": False},
        {"n": list(range(12))},
    )
    outputs: Any = None
    last_err: Exception | None = None
    for kwargs in calls:
        try:
            with torch.no_grad():
                outputs = model.get_intermediate_layers(x, **kwargs)
            last_err = None
            break
        except Exception as exc:  # pragma: no cover
            last_err = exc
    if last_err is not None:
        raise RuntimeError(f"Failed to query intermediate layers: {last_err}")

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    out: List[torch.Tensor] = []
    for layer_out in outputs:
        pt = _extract_patch_tokens_from_obj(layer_out, expected)
        out.append(pt.detach().cpu())
    if len(out) != 12:
        raise RuntimeError(f"Expected 12 layer token sets, got {len(out)}.")
    return out, gh, gw


def _tokens_to_map(tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    b, n, c = tokens.shape
    if int(n) != int(gh * gw):
        raise RuntimeError(f"Token count mismatch: n={n}, expected {gh*gw} ({gh}x{gw}).")
    return tokens.transpose(1, 2).reshape(b, c, gh, gw)


def _stage1_dino_projected_map(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Re-implements FrozenDinoTokenBranch path without square-grid assumption.
    backbone = m.dino.backbone
    layer_indices = tuple(int(v) for v in getattr(m.dino, "layer_indices", (len(backbone.blocks) - 1,)))
    depth = int(len(backbone.blocks))
    gh = int(x.shape[-2] // PATCH_SIZE)
    gw = int(x.shape[-1] // PATCH_SIZE)

    with torch.no_grad():
        if len(layer_indices) == 1 and layer_indices[0] == (depth - 1):
            feats = backbone.forward_features(x)
            toks = feats["x_norm_patchtokens"]  # [B,N,384]
            cat_map = _tokens_to_map(toks, gh, gw)
        else:
            inter = backbone.get_intermediate_layers(
                x,
                n=list(layer_indices),
                reshape=False,
                return_class_token=False,
                norm=True,
            )
            maps = [_tokens_to_map(t, gh, gw) for t in inter]
            cat_map = torch.cat(maps, dim=1)
        proj = m.dino.proj(cat_map)
    return proj


def _get_stage1_fused_tokens(adapter: Stage1Adapter, x: torch.Tensor) -> Tuple[List[torch.Tensor], int, int]:
    m = adapter.model
    with torch.no_grad():
        # Stage1 path up to fused = self.fuse_1x1(fused): this is a single output map.
        flocal = m.local(x)
        local_hw = (int(flocal.shape[-2]), int(flocal.shape[-1]))
        fdino = _stage1_dino_projected_map(m, x)
        fdino_up = m.dino_up(fdino, target_hw=local_hw)
        fused = torch.cat([fdino_up, flocal], dim=1)
        fused = m.fuse_1x1(fused)
        tok = fused.flatten(2).transpose(1, 2).detach().cpu()
    gh = int(local_hw[0])
    gw = int(local_hw[1])
    return [tok], gh, gw


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _token_norm_maps(tokens: List[torch.Tensor], gh: int, gw: int) -> List[List[float]]:
    maps: List[List[float]] = []
    for t in tokens:
        z = t[0].float()  # [N, C]
        if int(z.shape[0]) != int(gh * gw):
            raise RuntimeError(f"Token/grid mismatch: N={int(z.shape[0])}, grid={gh}x{gw}")
        n = torch.linalg.vector_norm(z, ord=2, dim=1).reshape(gh, gw)
        maps.append(n.reshape(-1).tolist())
    return maps


def _l2_normalize_tokens(tokens: List[torch.Tensor]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for t in tokens:
        z = t[0].float()  # [N, C]
        out.append(F.normalize(z, p=2, dim=1))
    return out


def _default_bbox(w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(w * 0.30)
    y1 = int(h * 0.30)
    x2 = int(w * 0.70)
    y2 = int(h * 0.70)
    return x1, y1, x2, y2


def _evict_if_needed() -> None:
    if len(SESSIONS) <= MAX_SESSIONS:
        return
    ids_sorted = sorted(SESSIONS.values(), key=lambda s: s.created_ts)
    to_remove = len(SESSIONS) - MAX_SESSIONS
    for i in range(to_remove):
        sid = ids_sorted[i].session_id
        SESSIONS.pop(sid, None)


def _xy_to_patch_idx(x: int, y: int, w: int, h: int, gw: int, gh: int) -> int:
    px = int(np.clip((float(x) / max(1.0, float(w))) * gw, 0, gw - 1))
    py = int(np.clip((float(y) / max(1.0, float(h))) * gh, 0, gh - 1))
    return py * gw + px


def _bbox_mask(w: int, h: int, gw: int, gh: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    xa, xb = sorted((x1, x2))
    ya, yb = sorted((y1, y2))
    cell_w = float(w) / float(gw)
    cell_h = float(h) / float(gh)
    ys = (np.arange(gh, dtype=np.float32) + 0.5) * cell_h
    xs = (np.arange(gw, dtype=np.float32) + 0.5) * cell_w
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return ((xx >= float(xa)) & (xx <= float(xb)) & (yy >= float(ya)) & (yy <= float(yb))).reshape(-1)


def _compute_similarity(
    session: SessionData,
    mode: str,
    point_x: int,
    point_y: int,
    bbox_x1: int,
    bbox_y1: int,
    bbox_x2: int,
    bbox_y2: int,
) -> Tuple[str, List[List[float]]]:
    w, h, gw, gh = session.width, session.height, session.grid_w, session.grid_h
    x = int(np.clip(point_x, 0, w - 1))
    y = int(np.clip(point_y, 0, h - 1))
    x1 = int(np.clip(bbox_x1, 0, w - 1))
    y1 = int(np.clip(bbox_y1, 0, h - 1))
    x2 = int(np.clip(bbox_x2, 0, w - 1))
    y2 = int(np.clip(bbox_y2, 0, h - 1))
    idx = _xy_to_patch_idx(x, y, w, h, gw, gh)
    mask = _bbox_mask(w, h, gw, gh, x1, y1, x2, y2)

    sim_maps: List[List[float]] = []
    for z in session.tokens_normed:
        if mode == "point":
            ref = z[idx]
        else:
            if int(mask.sum()) == 0:
                raise HTTPException(status_code=400, detail="BBox does not include any patch centers.")
            ref = z[mask].mean(dim=0)
            ref = F.normalize(ref[None, :], p=2, dim=1).squeeze(0)
        sim = torch.matmul(z, ref)  # [N], cosine due to normalization
        sim_maps.append(sim.cpu().tolist())

    if mode == "point":
        patch_x = int(np.clip((float(x) / max(1.0, float(w))) * gw, 0, gw - 1))
        patch_y = int(np.clip((float(y) / max(1.0, float(h))) * gh, 0, gh - 1))
        info = f"Reference mode=point, pixel=({x},{y}), patch=({patch_x},{patch_y})."
    else:
        info = (
            f"Reference mode=bbox, box=({min(x1, x2)},{min(y1, y2)})-({max(x1, x2)},{max(y1, y2)}), "
            f"selected_patches={int(mask.sum())}."
        )
    return info, sim_maps


def _build_session(img: Image.Image, max_side: int, use_loaded_adapter: bool) -> SessionData:
    proc = _resize_to_patch_grid(img, max_side=max_side)
    x = _to_tensor(proc)

    adapter_name = "none"
    adapter_ckpt = ""
    if use_loaded_adapter:
        if _ACTIVE_ADAPTER_PATH is None:
            raise HTTPException(status_code=400, detail="No adapter is loaded. Click 'Load Adapter' first.")
        ad = _load_stage1_adapter(_ACTIVE_ADAPTER_PATH)
        tokens, gh, gw = _get_stage1_fused_tokens(ad, x)
        adapter_name = ad.adapter_name
        adapter_ckpt = ad.checkpoint_path
    else:
        dino = _load_dino_reg()
        tokens, gh, gw = _get_all_layer_patch_tokens(dino, x)

    w, h = proc.size
    patch_count = int(gh * gw)
    norm_maps = _token_norm_maps(tokens, gh, gw)
    sid = uuid.uuid4().hex

    session = SessionData(
        session_id=sid,
        created_ts=time.time(),
        width=w,
        height=h,
        grid_w=int(gw),
        grid_h=int(gh),
        image_data_url=_image_to_data_url(proc),
        token_norm_maps=norm_maps,
        tokens_normed=_l2_normalize_tokens(tokens),
        adapter_name=adapter_name,
        adapter_checkpoint=adapter_ckpt,
        patch_count=patch_count,
    )
    SESSIONS[sid] = session
    _evict_if_needed()
    return session


app = FastAPI(title="ViT Explainer API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/adapter/current")
def adapter_current() -> Dict[str, Any]:
    if _ACTIVE_ADAPTER_PATH is None:
        return {"loaded": False, "adapter_name": "none", "adapter_checkpoint": ""}
    try:
        ad = _load_stage1_adapter(_ACTIVE_ADAPTER_PATH)
    except Exception as exc:
        return {"loaded": False, "adapter_name": "none", "adapter_checkpoint": "", "error": str(exc)}
    return {
        "loaded": True,
        "adapter_name": ad.adapter_name,
        "adapter_checkpoint": ad.checkpoint_path,
    }


@app.post("/api/adapter/load")
def adapter_load(req: AdapterLoadRequest) -> Dict[str, Any]:
    global _ACTIVE_ADAPTER_PATH
    try:
        ad = _load_stage1_adapter(req.checkpoint_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load adapter: {exc}") from exc
    _ACTIVE_ADAPTER_PATH = ad.checkpoint_path
    return {
        "loaded": True,
        "adapter_name": ad.adapter_name,
        "adapter_checkpoint": ad.checkpoint_path,
        "info": "Stage1 adapter loaded and cached. Enable 'apply loaded adapter' for extraction.",
    }


@app.post("/api/adapter/clear")
def adapter_clear() -> Dict[str, Any]:
    global _ACTIVE_ADAPTER_PATH
    _ACTIVE_ADAPTER_PATH = None
    return {"loaded": False, "adapter_name": "none", "adapter_checkpoint": "", "info": "Adapter cleared."}


@app.post("/api/encode_path")
def encode_path(req: EncodePathRequest) -> Dict[str, Any]:
    p = Path(req.image_path).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Image path does not exist: {p}")
    try:
        img = Image.open(p).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}") from exc

    try:
        session = _build_session(img, max_side=req.max_side, use_loaded_adapter=bool(req.apply_loaded_adapter))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract tokens: {exc}") from exc
    px, py = session.width // 2, session.height // 2
    bx1, by1, bx2, by2 = _default_bbox(session.width, session.height)
    return {
        "session_id": session.session_id,
        "image_width": session.width,
        "image_height": session.height,
        "grid_w": session.grid_w,
        "grid_h": session.grid_h,
        "layer_count": len(session.token_norm_maps),
        "patch_size": PATCH_SIZE,
        "image_data_url": session.image_data_url,
        "token_norm_maps": session.token_norm_maps,
        "default_point_x": px,
        "default_point_y": py,
        "default_bbox_x1": bx1,
        "default_bbox_y1": by1,
        "default_bbox_x2": bx2,
        "default_bbox_y2": by2,
        "info": f"Extracted patch tokens from {len(session.token_norm_maps)} layer(s). Grid={session.grid_w}x{session.grid_h}.",
        "adapter_name": session.adapter_name,
        "adapter_checkpoint": session.adapter_checkpoint,
        "patch_count": session.patch_count,
    }


@app.post("/api/encode_upload")
async def encode_upload(
    file: UploadFile = File(...),
    max_side: int = 560,
    apply_loaded_adapter: bool = False,
) -> Dict[str, Any]:
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image: {exc}") from exc

    try:
        session = _build_session(img, max_side=max_side, use_loaded_adapter=bool(apply_loaded_adapter))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract tokens: {exc}") from exc
    px, py = session.width // 2, session.height // 2
    bx1, by1, bx2, by2 = _default_bbox(session.width, session.height)
    return {
        "session_id": session.session_id,
        "image_width": session.width,
        "image_height": session.height,
        "grid_w": session.grid_w,
        "grid_h": session.grid_h,
        "layer_count": len(session.token_norm_maps),
        "patch_size": PATCH_SIZE,
        "image_data_url": session.image_data_url,
        "token_norm_maps": session.token_norm_maps,
        "default_point_x": px,
        "default_point_y": py,
        "default_bbox_x1": bx1,
        "default_bbox_y1": by1,
        "default_bbox_x2": bx2,
        "default_bbox_y2": by2,
        "info": f"Extracted patch tokens from {len(session.token_norm_maps)} layer(s). Grid={session.grid_w}x{session.grid_h}.",
        "adapter_name": session.adapter_name,
        "adapter_checkpoint": session.adapter_checkpoint,
        "patch_count": session.patch_count,
    }


@app.post("/api/similarity")
def similarity(req: SimilarityRequest) -> Dict[str, Any]:
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Re-run token extraction.")

    info, maps = _compute_similarity(
        session=session,
        mode=req.mode,
        point_x=req.point_x,
        point_y=req.point_y,
        bbox_x1=req.bbox_x1,
        bbox_y1=req.bbox_y1,
        bbox_x2=req.bbox_x2,
        bbox_y2=req.bbox_y2,
    )
    return {"session_id": session.session_id, "similarity_maps": maps, "info": info}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
