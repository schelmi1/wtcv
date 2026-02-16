import { useEffect, useRef } from "react";

function normalize01(values) {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi - lo < 1e-12) {
    return values.map(() => 0);
  }
  const span = hi - lo;
  return values.map((v) => (v - lo) / span);
}

function colorUnsigned(u) {
  const r = Math.round(255 * u);
  const g = Math.round(220 * u);
  const b = Math.round(70 * (1 - u));
  return [r, g, b];
}

function colorSigned(v) {
  const u = Math.max(0, Math.min(1, (v + 1) * 0.5));
  const lo = [20, 90, 220];
  const mid = [245, 245, 245];
  const hi = [220, 60, 40];
  if (u <= 0.5) {
    const t = u * 2;
    return [
      Math.round(lo[0] * (1 - t) + mid[0] * t),
      Math.round(lo[1] * (1 - t) + mid[1] * t),
      Math.round(lo[2] * (1 - t) + mid[2] * t),
    ];
  }
  const t = (u - 0.5) * 2;
  return [
    Math.round(mid[0] * (1 - t) + hi[0] * t),
    Math.round(mid[1] * (1 - t) + hi[1] * t),
    Math.round(mid[2] * (1 - t) + hi[2] * t),
  ];
}

export function drawMapToCanvas(canvas, map, gw, gh, outW, outH, kind) {
  if (!canvas || !map || gw <= 0 || gh <= 0 || map.length !== gw * gh) return;
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const src = document.createElement("canvas");
  src.width = gw;
  src.height = gh;
  const sctx = src.getContext("2d");
  if (!sctx) return;

  const img = sctx.createImageData(gw, gh);
  const data = img.data;
  const norm = kind === "norm" ? normalize01(map) : null;
  for (let i = 0; i < map.length; i += 1) {
    const [r, g, b] = kind === "sim" ? colorSigned(map[i]) : colorUnsigned(norm[i]);
    const j = i * 4;
    data[j + 0] = r;
    data[j + 1] = g;
    data[j + 2] = b;
    data[j + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);

  ctx.clearRect(0, 0, outW, outH);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(src, 0, 0, outW, outH);
}

export function HeatmapCanvas({ map, gw, gh, width, height, kind, className }) {
  const ref = useRef(null);

  useEffect(() => {
    drawMapToCanvas(ref.current, map, gw, gh, width, height, kind);
  }, [map, gw, gh, width, height, kind]);

  return <canvas ref={ref} className={className} />;
}
