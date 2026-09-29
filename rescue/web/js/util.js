// Small helpers shared by every module.
import * as THREE from "three";

export const $ = id => document.getElementById(id);
export const hex = c => "#" + c.toString(16).padStart(6, "0");
export const rgb = a => new THREE.Color(a[0] / 255, a[1] / 255, a[2] / 255);
export const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
export const fmtS = v => v === null || v === undefined ? "–" : v + " s";
export const sleep = ms => new Promise(r => setTimeout(r, ms));

// deterministic random numbers, so a building looks the same every time it is drawn
export function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

// the "ironbow" palette of thermal cameras: black -> purple -> red -> orange -> yellow -> white
const IRONBOW = [[0, 0, 0], [28, 0, 72], [120, 0, 150], [200, 30, 90], [240, 100, 20], [255, 180, 0], [255, 235, 110], [255, 255, 255]];
export function ironbow(u) {
  u = Math.min(1, Math.max(0, u)) * (IRONBOW.length - 1);
  const i = Math.min(IRONBOW.length - 2, Math.floor(u)), f = u - i;
  return IRONBOW[i].map((c, k) => Math.round(c * (1 - f) + IRONBOW[i + 1][k] * f));
}

// temperature (°C) of a cell on the team's thermal map, or null if nobody has looked there
export function cellTemp(thermalMap, i) {
  const code = thermalMap.charCodeAt(i) - 48;
  return code > 0 ? 5 + (code - 1) / 2 : null;
}

export function canvasTexture(size, draw) {
  const c = document.createElement("canvas"); c.width = c.height = size;
  draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8;
  return t;
}

export function concrete(base) {
  return canvasTexture(128, (g, n) => {
    const rnd = mulberry32(7);
    g.fillStyle = base; g.fillRect(0, 0, n, n);
    for (let i = 0; i < 900; i++) {
      g.fillStyle = rnd() < 0.5 ? `rgba(0,0,0,${0.04 + 0.07 * rnd()})` : `rgba(255,255,255,${0.04 + 0.06 * rnd()})`;
      g.fillRect(rnd() * n, rnd() * n, 1 + 2 * rnd(), 1 + 2 * rnd());
    }
    g.strokeStyle = "rgba(60,55,50,0.4)";
    for (let k = 0; k < 3; k++) { let x = rnd() * n, y = rnd() * n; g.beginPath(); g.moveTo(x, y);
      for (let s = 0; s < 6; s++) { x += (rnd() - 0.5) * 30; y += rnd() * 22; g.lineTo(x, y); } g.stroke(); }
    g.fillStyle = "rgba(0,0,0,0.16)"; g.fillRect(0, n - 10, n, 10);
  });
}

// a rounded text label that always faces the camera
export function labelSprite(text, bg, fg = "#fff", scale = 1) {
  const font = "bold 32px system-ui", probe = document.createElement("canvas").getContext("2d");
  probe.font = font;
  const w = probe.measureText(text).width + 36, cw = Math.max(220, Math.ceil(w) + 4);   // long labels get a wider canvas
  const c = document.createElement("canvas"); c.width = cw; c.height = 64;
  const g = c.getContext("2d"); g.font = font;
  g.fillStyle = bg; g.beginPath(); g.roundRect((cw - w) / 2, 4, w, 56, 14); g.fill();
  g.fillStyle = fg; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, cw / 2, 34);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, fog: false }));
  s.scale.set(1.5 * scale * cw / 220, 0.44 * scale, 1); s.renderOrder = 10;
  return s;
}

// size a canvas for the screen's pixel density and return a context in CSS pixels
export function fitCanvas(canvas, height) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth || 320;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(height * dpr); canvas.style.height = height + "px";
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h: height };
}
