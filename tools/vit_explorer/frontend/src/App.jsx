import { useEffect, useMemo, useRef, useState } from "react";
import { HeatmapCanvas } from "./heatmap.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "";

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function parseJsonResponse(resp) {
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data?.detail || data?.error || `HTTP ${resp.status}`);
  }
  return data;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function drawGridAndSelection(canvas, img, session, mode, point, bbox, dragBox) {
  if (!canvas || !img || !session) return;
  const w = session.image_width;
  const h = session.image_height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  const cellW = w / Math.max(1, session.grid_w || 1);
  const cellH = h / Math.max(1, session.grid_h || 1);
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= session.grid_w; i += 1) {
    const x = Math.round(i * cellW);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let j = 0; j <= session.grid_h; j += 1) {
    const y = Math.round(j * cellH);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#ff3b30";
  ctx.fillStyle = "#ff3b30";
  ctx.lineWidth = 2.5;
  if (mode === "point") {
    const x = clamp(Math.round(point.x), 0, w - 1);
    const y = clamp(Math.round(point.y), 0, h - 1);
    const r = 6;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 10, y);
    ctx.lineTo(x + 10, y);
    ctx.moveTo(x, y - 10);
    ctx.lineTo(x, y + 10);
    ctx.stroke();
  } else {
    const x1 = Math.min(bbox.x1, bbox.x2);
    const y1 = Math.min(bbox.y1, bbox.y2);
    const x2 = Math.max(bbox.x1, bbox.x2);
    const y2 = Math.max(bbox.y1, bbox.y2);
    ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
  }

  if (dragBox) {
    ctx.strokeStyle = "#ffd60a";
    ctx.lineWidth = 2;
    const x1 = Math.min(dragBox.x0, dragBox.x1);
    const y1 = Math.min(dragBox.y0, dragBox.y1);
    const x2 = Math.max(dragBox.x0, dragBox.x1);
    const y2 = Math.max(dragBox.y0, dragBox.y1);
    ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
  }
}

function LayerStrip({ title, maps, session, selectedLayer, onSelect, kind }) {
  if (!session || maps.length === 0) return null;
  const thumbW = 150;
  const thumbH = Math.round((thumbW * session.image_height) / session.image_width);
  return (
    <section className="panel">
      <h3>{title}</h3>
      <div className="layer-strip">
        {maps.map((m, idx) => (
          <button
            key={`${title}-${idx}`}
            className={`layer-card ${selectedLayer === idx + 1 ? "active" : ""}`}
            onClick={() => onSelect(idx + 1)}
            type="button"
          >
            <div className="layer-title">Layer {idx + 1}</div>
            <HeatmapCanvas
              map={m}
              gw={session.grid_w}
              gh={session.grid_h}
              width={thumbW}
              height={thumbH}
              kind={kind}
              className="thumb-canvas"
            />
          </button>
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const HOVER_DEBOUNCE_MS = 70;
  const [imagePath, setImagePath] = useState("");
  const [adapterCheckpoint, setAdapterCheckpoint] = useState("");
  const [loadedAdapter, setLoadedAdapter] = useState({ loaded: false, adapter_name: "none", adapter_checkpoint: "" });
  const [applyLoadedAdapter, setApplyLoadedAdapter] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [session, setSession] = useState(null);
  const [tokenMaps, setTokenMaps] = useState([]);
  const [simMaps, setSimMaps] = useState([]);
  const [mode, setMode] = useState("point");
  const [point, setPoint] = useState({ x: 0, y: 0 });
  const [bbox, setBbox] = useState({ x1: 0, y1: 0, x2: 0, y2: 0 });
  const [layer, setLayer] = useState(1);
  const [status, setStatus] = useState("Load an image path or upload a file.");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [simBusy, setSimBusy] = useState(false);
  const [dragBox, setDragBox] = useState(null);

  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const hoverTimerRef = useRef(null);
  const simAbortRef = useRef(null);
  const simReqIdRef = useRef(0);

  useEffect(() => {
    if (!session?.image_data_url) {
      imgRef.current = null;
      return;
    }
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      drawGridAndSelection(canvasRef.current, img, session, mode, point, bbox, dragBox);
    };
    img.src = session.image_data_url;
  }, [session]);

  useEffect(() => {
    drawGridAndSelection(canvasRef.current, imgRef.current, session, mode, point, bbox, dragBox);
  }, [session, mode, point, bbox, dragBox]);

  useEffect(() => {
    return () => {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      if (simAbortRef.current) {
        simAbortRef.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const resp = await fetch(apiUrl("/api/adapter/current"));
        const data = await parseJsonResponse(resp);
        if (active) {
          setLoadedAdapter({
            loaded: !!data.loaded,
            adapter_name: data.adapter_name || "none",
            adapter_checkpoint: data.adapter_checkpoint || "",
          });
        }
      } catch (_err) {
        // Non-fatal for first paint.
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!session) return;
    scheduleSimilarity(session.session_id, mode, point, bbox, 0);
  }, [mode]);

  const selectedNormMap = useMemo(() => tokenMaps[layer - 1] || null, [tokenMaps, layer]);
  const selectedSimMap = useMemo(() => simMaps[layer - 1] || null, [simMaps, layer]);
  const layerMax = useMemo(() => Math.max(1, tokenMaps.length, simMaps.length), [tokenMaps, simMaps]);

  useEffect(() => {
    if (layer > layerMax) {
      setLayer(layerMax);
    }
  }, [layer, layerMax]);

  function eventToImageXY(evt) {
    const canvas = canvasRef.current;
    if (!canvas || !session) return null;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const x = clamp(Math.round((evt.clientX - rect.left) * sx), 0, session.image_width - 1);
    const y = clamp(Math.round((evt.clientY - rect.top) * sy), 0, session.image_height - 1);
    return { x, y };
  }

  function onCanvasMouseDown(evt) {
    if (!session || mode !== "bbox") return;
    const xy = eventToImageXY(evt);
    if (!xy) return;
    setDragBox({ x0: xy.x, y0: xy.y, x1: xy.x, y1: xy.y });
  }

  function onCanvasMouseMove(evt) {
    if (!session) return;
    const xy = eventToImageXY(evt);
    if (!xy) return;
    if (mode === "point") {
      setPoint(xy);
      scheduleSimilarity(session.session_id, "point", xy, bbox, HOVER_DEBOUNCE_MS);
      return;
    }
    if (!dragBox) return;
    const x1 = Math.min(dragBox.x0, xy.x);
    const y1 = Math.min(dragBox.y0, xy.y);
    const x2 = Math.max(dragBox.x0, xy.x);
    const y2 = Math.max(dragBox.y0, xy.y);
    const box = { x1, y1, x2, y2 };
    setDragBox((prev) => ({ ...prev, x1: xy.x, y1: xy.y }));
    setBbox(box);
    scheduleSimilarity(session.session_id, "bbox", point, box, HOVER_DEBOUNCE_MS);
  }

  function onCanvasMouseUp(evt) {
    if (!session || !dragBox || mode !== "bbox") return;
    const xy = eventToImageXY(evt);
    if (!xy) return;
    const x1 = Math.min(dragBox.x0, xy.x);
    const y1 = Math.min(dragBox.y0, xy.y);
    const x2 = Math.max(dragBox.x0, xy.x);
    const y2 = Math.max(dragBox.y0, xy.y);
    const box = { x1, y1, x2, y2 };
    setBbox(box);
    setDragBox(null);
    scheduleSimilarity(session.session_id, "bbox", point, box, 0);
  }

  async function runSimilarityRequest(sid, m, p, b) {
    const reqId = simReqIdRef.current + 1;
    simReqIdRef.current = reqId;
    if (simAbortRef.current) {
      simAbortRef.current.abort();
    }
    const controller = new AbortController();
    simAbortRef.current = controller;
    setSimBusy(true);
    try {
      const resp = await fetch(apiUrl("/api/similarity"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sid,
          mode: m,
          point_x: p.x,
          point_y: p.y,
          bbox_x1: b.x1,
          bbox_y1: b.y1,
          bbox_x2: b.x2,
          bbox_y2: b.y2,
        }),
        signal: controller.signal,
      });
      const data = await parseJsonResponse(resp);
      if (reqId !== simReqIdRef.current) return;
      setSimMaps(data.similarity_maps || []);
      setStatus(data.info || "Similarity updated.");
    } catch (err) {
      if (err?.name === "AbortError") return;
      setError(String(err.message || err));
    } finally {
      if (reqId === simReqIdRef.current) {
        setSimBusy(false);
      }
    }
  }

  function scheduleSimilarity(sid, m, p, b, debounceMs = 0) {
    if (!sid) return;
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    if (debounceMs <= 0) {
      runSimilarityRequest(sid, m, p, b);
      return;
    }
    hoverTimerRef.current = setTimeout(() => {
      hoverTimerRef.current = null;
      runSimilarityRequest(sid, m, p, b);
    }, debounceMs);
  }

  async function applyExtractResponse(data) {
    setSession(data);
    setTokenMaps(data.token_norm_maps || []);
    setLayer(1);
    setMode("point");
    const p = { x: data.default_point_x, y: data.default_point_y };
    const b = {
      x1: data.default_bbox_x1,
      y1: data.default_bbox_y1,
      x2: data.default_bbox_x2,
      y2: data.default_bbox_y2,
    };
    setPoint(p);
    setBbox(b);
    setStatus(data.info || "Tokens extracted.");
    await runSimilarityRequest(data.session_id, "point", p, b);
  }

  async function onExtractFromPath() {
    if (!imagePath.trim()) {
      setError("Provide an image path.");
      return;
    }
    try {
      setBusy(true);
      setError("");
      setStatus("Extracting patch tokens...");
      const resp = await fetch(apiUrl("/api/encode_path"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: imagePath,
          max_side: 560,
          apply_loaded_adapter: applyLoadedAdapter,
        }),
      });
      const data = await parseJsonResponse(resp);
      await applyExtractResponse(data);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function onExtractUpload() {
    if (!uploadFile) {
      setError("Choose an image file first.");
      return;
    }
    try {
      setBusy(true);
      setError("");
      setStatus("Uploading image and extracting patch tokens...");
      const fd = new FormData();
      fd.append("file", uploadFile);
      const query = new URLSearchParams({
        max_side: "560",
        apply_loaded_adapter: applyLoadedAdapter ? "true" : "false",
      });
      const resp = await fetch(apiUrl(`/api/encode_upload?${query.toString()}`), {
        method: "POST",
        body: fd,
      });
      const data = await parseJsonResponse(resp);
      await applyExtractResponse(data);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function onLoadAdapter() {
    if (!adapterCheckpoint.trim()) {
      setError("Provide an adapter checkpoint path.");
      return;
    }
    try {
      setBusy(true);
      setError("");
      setStatus("Loading adapter checkpoint...");
      const resp = await fetch(apiUrl("/api/adapter/load"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint_path: adapterCheckpoint.trim() }),
      });
      const data = await parseJsonResponse(resp);
      setLoadedAdapter({
        loaded: !!data.loaded,
        adapter_name: data.adapter_name || "none",
        adapter_checkpoint: data.adapter_checkpoint || "",
      });
      setStatus(data.info || "Adapter loaded.");
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function onClearAdapter() {
    try {
      setBusy(true);
      setError("");
      const resp = await fetch(apiUrl("/api/adapter/clear"), {
        method: "POST",
      });
      const data = await parseJsonResponse(resp);
      setLoadedAdapter({
        loaded: !!data.loaded,
        adapter_name: data.adapter_name || "none",
        adapter_checkpoint: data.adapter_checkpoint || "",
      });
      setApplyLoadedAdapter(false);
      setStatus(data.info || "Adapter cleared.");
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <h1>ViT Explainer</h1>
        <p>DINOv2 ViT-S/14 Reg. Select point or bbox, then compare cosine similarity across available layers.</p>
      </header>

      <section className="panel controls">
        <div className="control-row">
          <label>
            Optional Adapter Checkpoint (backend path)
            <input
              type="text"
              value={adapterCheckpoint}
              onChange={(e) => setAdapterCheckpoint(e.target.value)}
              placeholder="/home/schelli/git/wtcv/runs/.../checkpoints/best_val_iou.pt"
            />
          </label>
          <div className="mode-row">
            <button type="button" onClick={onLoadAdapter} disabled={busy}>
              Load Adapter
            </button>
            <button type="button" onClick={onClearAdapter} disabled={busy}>
              Clear Adapter
            </button>
          </div>
        </div>
        <div className="control-row">
          <label className="mode-row">
            <input
              type="checkbox"
              checked={applyLoadedAdapter}
              onChange={(e) => setApplyLoadedAdapter(e.target.checked)}
              disabled={!loadedAdapter.loaded}
            />
            Apply loaded adapter during extraction
          </label>
          <div className="hint">
            Adapter is loaded explicitly via button. Extraction uses it only when this toggle is enabled.
          </div>
        </div>
        <div className="control-row">
          <label>
            Image Path (backend local path)
            <input
              type="text"
              value={imagePath}
              onChange={(e) => setImagePath(e.target.value)}
              placeholder="/absolute/path/to/image.jpg"
            />
          </label>
          <button type="button" onClick={onExtractFromPath} disabled={busy}>
            Extract From Path
          </button>
        </div>
        <div className="control-row">
          <label>
            Upload Image (browser)
            <input type="file" accept="image/*" onChange={(e) => setUploadFile(e.target.files?.[0] || null)} />
          </label>
          <button type="button" onClick={onExtractUpload} disabled={busy}>
            Extract From Upload
          </button>
        </div>
      </section>

      {error ? <div className="error">{error}</div> : null}
      <div className="status">{busy ? "Working..." : simBusy ? "Updating similarity..." : status}</div>
      <div className="status">
        Loaded adapter:{" "}
        <code>{loadedAdapter.loaded ? loadedAdapter.adapter_name : "none"}</code>
        {loadedAdapter.loaded && loadedAdapter.adapter_checkpoint ? ` | ${loadedAdapter.adapter_checkpoint}` : ""}
      </div>
      {session?.adapter_name ? (
        <div className="status">
          Session adapter: <code>{session.adapter_name}</code>
          {session.adapter_checkpoint ? ` | ${session.adapter_checkpoint}` : ""}
          {typeof session.patch_count === "number" ? ` | patches=${session.patch_count}` : ""}
        </div>
      ) : null}

      <section className="panel two-col">
        <div>
          <h3>Selection Canvas</h3>
          <canvas
            ref={canvasRef}
            className="selection-canvas"
            onMouseDown={onCanvasMouseDown}
            onMouseMove={onCanvasMouseMove}
            onMouseUp={onCanvasMouseUp}
            onMouseLeave={onCanvasMouseUp}
          />
          <div className="hint">Point mode: hover to update. BBox mode: drag box to update live.</div>
        </div>
        <div className="selection-controls">
          <h3>Reference Selection</h3>
          <div className="mode-row">
            <label>
              <input type="radio" name="mode" checked={mode === "point"} onChange={() => setMode("point")} />
              Point
            </label>
            <label>
              <input type="radio" name="mode" checked={mode === "bbox"} onChange={() => setMode("bbox")} />
              BBox
            </label>
          </div>
          <div className="number-grid">
            <label>
              Point X
              <input
                type="number"
                value={point.x}
                onChange={(e) => {
                  const next = { ...point, x: Number(e.target.value) || 0 };
                  setPoint(next);
                  if (session) scheduleSimilarity(session.session_id, mode, next, bbox, HOVER_DEBOUNCE_MS);
                }}
              />
            </label>
            <label>
              Point Y
              <input
                type="number"
                value={point.y}
                onChange={(e) => {
                  const next = { ...point, y: Number(e.target.value) || 0 };
                  setPoint(next);
                  if (session) scheduleSimilarity(session.session_id, mode, next, bbox, HOVER_DEBOUNCE_MS);
                }}
              />
            </label>
            <label>
              BBox X1
              <input
                type="number"
                value={bbox.x1}
                onChange={(e) => {
                  const next = { ...bbox, x1: Number(e.target.value) || 0 };
                  setBbox(next);
                  if (session) scheduleSimilarity(session.session_id, mode, point, next, HOVER_DEBOUNCE_MS);
                }}
              />
            </label>
            <label>
              BBox Y1
              <input
                type="number"
                value={bbox.y1}
                onChange={(e) => {
                  const next = { ...bbox, y1: Number(e.target.value) || 0 };
                  setBbox(next);
                  if (session) scheduleSimilarity(session.session_id, mode, point, next, HOVER_DEBOUNCE_MS);
                }}
              />
            </label>
            <label>
              BBox X2
              <input
                type="number"
                value={bbox.x2}
                onChange={(e) => {
                  const next = { ...bbox, x2: Number(e.target.value) || 0 };
                  setBbox(next);
                  if (session) scheduleSimilarity(session.session_id, mode, point, next, HOVER_DEBOUNCE_MS);
                }}
              />
            </label>
            <label>
              BBox Y2
              <input
                type="number"
                value={bbox.y2}
                onChange={(e) => {
                  const next = { ...bbox, y2: Number(e.target.value) || 0 };
                  setBbox(next);
                  if (session) scheduleSimilarity(session.session_id, mode, point, next, HOVER_DEBOUNCE_MS);
                }}
              />
            </label>
          </div>
          <div className="hint">No button needed. Similarity updates automatically from hover/drag/inputs.</div>
        </div>
      </section>

      <section className="panel">
        <div className="layer-header">
          <h3>Layer Inspector</h3>
          <label>
            Layer {layer} / {layerMax}
            <input
              type="range"
              min="1"
              max={layerMax}
              value={layer}
              onChange={(e) => setLayer(Number(e.target.value))}
              disabled={!session}
            />
          </label>
        </div>
        <div className="map-row">
          <div>
            <h4>Patch-Token Norm (Layer {layer})</h4>
            {session && selectedNormMap ? (
              <HeatmapCanvas
                map={selectedNormMap}
                gw={session.grid_w}
                gh={session.grid_h}
                width={session.image_width}
                height={session.image_height}
                kind="norm"
                className="big-map"
              />
            ) : (
              <div className="empty">No map yet.</div>
            )}
          </div>
          <div>
            <h4>Cosine Similarity (Layer {layer})</h4>
            {session && selectedSimMap ? (
              <HeatmapCanvas
                map={selectedSimMap}
                gw={session.grid_w}
                gh={session.grid_h}
                width={session.image_width}
                height={session.image_height}
                kind="sim"
                className="big-map"
              />
            ) : (
              <div className="empty">Compute similarity to populate this map.</div>
            )}
          </div>
        </div>
      </section>

      <LayerStrip title={`Token Norm Maps (${tokenMaps.length} layer${tokenMaps.length === 1 ? "" : "s"})`} maps={tokenMaps} session={session} selectedLayer={layer} onSelect={setLayer} kind="norm" />
      <LayerStrip title={`Cosine Similarity Maps (${simMaps.length} layer${simMaps.length === 1 ? "" : "s"})`} maps={simMaps} session={session} selectedLayer={layer} onSelect={setLayer} kind="sim" />
    </div>
  );
}
