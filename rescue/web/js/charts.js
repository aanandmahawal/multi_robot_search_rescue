// The three small drawings of the details panel: the victim map, the LiDAR scan of one robot,
// and the progress chart.
import { FALSE_ALARM, ROBOT_COLORS, STATUS } from "./config.js";
import { store, truthVisible } from "./store.js";
import { $, cellTemp, fitCanvas, hex, ironbow } from "./util.js";

// ------------------------------------------------------------------ victim map (top-down, the team's own picture)
export function drawVictimMap() {
  const { world, state } = store;
  const canvas = $("vmap"), W = world.W, H = world.H;
  const s = (canvas.clientWidth || 320) / W, m2p = s / world.cell;
  const { g } = fitCanvas(canvas, Math.round(s * H));
  const [tLo, tHi] = world.thermal_scale;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, k = state.known[i];
    g.fillStyle = k === "0" ? "#0e131a" : k === "2" ? "#566378" : state.searched[i] === "1" ? "#2b3b52" : "#1a2433";
    g.fillRect(x * s, y * s, s + 0.5, s + 0.5);
    const T = cellTemp(state.thermal_map, i);
    if (T !== null && T - world.ambient >= 2.5) { const [r, gg, b] = ironbow((T - tLo) / (tHi - tLo)); g.fillStyle = `rgb(${r},${gg},${b})`; g.fillRect(x * s, y * s, s + 0.5, s + 0.5); }
  }
  g.fillStyle = "#2cc3cc"; for (const [x, y] of state.frontiers) g.fillRect(x * s, y * s, s + 0.5, s + 0.5);

  const dot = (x, y, r, fill, stroke) => { g.beginPath(); g.arc(x * m2p, y * m2p, r, 0, 7); g.fillStyle = fill; g.fill(); if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.5; g.stroke(); } };
  const mark = (x, y, text, color) => { g.font = "bold 10px system-ui"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = color; g.fillText(text, x * m2p, y * m2p + 0.5); };
  const truth = truthVisible();
  if (truth) for (const st of state.victim_status) {
    const v = state.victims.find(b => b.id === st.id);
    if (v && st.status !== "found") dot(v.x, v.y, 7, "rgba(0,0,0,0)", STATUS[st.status].css);
  }
  state.robots.forEach((r, i) => dot(r.x, r.y, 3.5, hex(ROBOT_COLORS[i % ROBOT_COLORS.length]), "#fff"));
  for (const c of state.candidates) {
    const px = c.x * m2p, py = c.y * m2p;
    if (c.status === "rejected") { g.strokeStyle = "#8b99a8"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px - 3.5, py - 3.5); g.lineTo(px + 3.5, py + 3.5); g.moveTo(px + 3.5, py - 3.5); g.lineTo(px - 3.5, py + 3.5); g.stroke(); continue; }
    if (c.status === "confirmed") {
      const real = c.truth.real;
      dot(c.x, c.y, 6.5, real ? STATUS.found.css : FALSE_ALARM.css, "#0b1016"); mark(c.x, c.y, real ? "✓" : "✕", real ? "#04210d" : "#fff");
      continue;
    }
    dot(c.x, c.y, 5, STATUS.sighted.css, "#0b1016");
    const label = Math.round(100 * c.p) + "%";
    g.font = "bold 10px system-ui"; g.textAlign = "left"; g.textBaseline = "middle";
    g.lineWidth = 3; g.strokeStyle = "#0b1016"; g.strokeText(label, px + 8, py); g.fillStyle = STATUS.sighted.css; g.fillText(label, px + 8, py);
  }

  const rep = state.reports, key = (cls, bg, inner, text) => `<span><i class="${cls}" style="background:${bg}">${inner}</i>${text}</span>`;
  const keys = [];
  if (world.has_cameras) keys.push(key("", STATUS.found.css, "✓", "confirmed: a real victim"), key("", FALSE_ALARM.css, "✕", "confirmed: false alarm"),
                                   key("", STATUS.sighted.css, "", "suspected, with probability"), key("sq", "transparent", "<b style='color:#8b99a8'>✕</b>", "rejected after a closer look"),
                                   key("sq", "linear-gradient(90deg,#c81e5a,#ffb400)", "", "warm spot (thermal camera)"), key("sq", "#2b3b52", "", "searched by a camera"));
  keys.push(key("sq", "#1a2433", "", world.has_cameras ? "mapped, not searched yet" : "free floor on the map"), key("sq", "#566378", "", "obstacle"),
            key("sq", "#2cc3cc", "", world.has_cameras ? "edge of the searched area" : "edge of the map"), key("", "#4f9cf9", "", "robot"));
  if (truth) keys.push(key("", "transparent; border:2px solid " + STATUS.hidden.css, "", "truth: not found yet"), key("", "transparent; border:2px solid " + STATUS.buried.css, "", "truth: buried, undetectable"));
  $("vmapKeys").innerHTML = keys.join("");
  $("vmapNote").textContent = !world.has_cameras
    ? "These robots carry only a LiDAR: the map shows the building, but nobody can be found without a camera."
    : `The robots confirmed ${rep.confirmed} ${rep.confirmed === 1 ? "person" : "people"}: ${rep.real} real victim${rep.real === 1 ? "" : "s"} and ${rep.false} false alarm${rep.false === 1 ? "" : "s"}. ` +
      `${rep.suspected} more ${rep.suspected === 1 ? "sighting is" : "sightings are"} being checked. The robots cannot tell a false alarm from a victim; the simulator can, and marks it here.`;
}

// ------------------------------------------------------------------ LiDAR scan of one robot, robot facing up
export function drawLidarPlot() {
  const { world, state } = store, cfg = world.config;
  const r = state.robots[store.selected], color = hex(ROBOT_COLORS[store.selected % ROBOT_COLORS.length]);
  const { g, w, h } = fitCanvas($("lidarPlot"), 250);
  const cx = w / 2, cy = h / 2, range = cfg.lidar_range, k = (h / 2 - 14) / range;
  g.fillStyle = "#070b10"; g.fillRect(0, 0, w, h);

  g.strokeStyle = "#1d2a3a"; g.fillStyle = "#5d6d80"; g.font = "10px system-ui"; g.textAlign = "left"; g.textBaseline = "middle"; g.lineWidth = 1;
  for (let m = 2; m <= range + 0.01; m += 2) { g.beginPath(); g.arc(cx, cy, m * k, 0, 7); g.stroke(); g.fillText(m + " m", cx + m * k + 3, cy); }
  g.beginPath(); g.moveTo(cx, cy - range * k); g.lineTo(cx, cy + range * k); g.moveTo(cx - range * k, cy); g.lineTo(cx + range * k, cy); g.stroke();

  if (world.has_cameras) {                              // what the cameras cover, for comparison: a wedge straight ahead
    const half = cfg.camera_fov * Math.PI / 360;
    g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, cfg.camera_range * k, -Math.PI / 2 - half, -Math.PI / 2 + half); g.closePath();
    g.fillStyle = "rgba(79,156,249,0.13)"; g.fill(); g.strokeStyle = "rgba(79,156,249,0.45)"; g.stroke();
    g.fillStyle = "#7f93aa"; g.textAlign = "left"; g.textBaseline = "top"; g.fillText("blue wedge: what the cameras see", 8, 8);
  }
  const scan = r.scan || [], cs = Math.cos(r.heading), sn = Math.sin(r.heading);
  let near = Infinity, far = 0;
  g.fillStyle = color;
  for (const [x, y] of scan) {
    const dx = x - r.x, dy = y - r.y, fwd = dx * cs + dy * sn, right = -dx * sn + dy * cs, d = Math.hypot(dx, dy);
    near = Math.min(near, d); far = Math.max(far, d);
    g.fillRect(cx + right * k - 1.3, cy - fwd * k - 1.3, 2.6, 2.6);
  }
  g.beginPath(); g.moveTo(cx, cy - 8); g.lineTo(cx + 5.5, cy + 6); g.lineTo(cx - 5.5, cy + 6); g.closePath();   // the robot
  g.fillStyle = "#fff"; g.fill(); g.strokeStyle = color; g.lineWidth = 2; g.stroke();

  $("lidarInfo").textContent = scan.length
    ? `${scan.length} of ${cfg.lidar_rays} beams came back · nearest surface ${near.toFixed(1)} m · farthest ${far.toFixed(1)} m. ` +
      `The laser sweeps ${Math.round(cfg.lidar_height * 100)} cm above the floor, so a person lying on the floor does not appear here.`
    : "Waiting for the first scan: press Start.";
}

// ------------------------------------------------------------------ progress over time
export function drawChart() {
  const { world, state } = store, hs = state.history;
  const { g, w, h } = fitCanvas($("chart"), 150), pad = 26;
  g.clearRect(0, 0, w, h);
  g.fillStyle = "#8b99a8"; g.font = "11px system-ui"; g.textBaseline = "alphabetic"; g.textAlign = "left";
  if (hs.length < 2) { g.fillText("No data yet: press Start.", 8, h / 2); return; }
  const tMax = Math.max(60, hs[hs.length - 1].t);
  const top = world.config.victims_known ? world.config.n_victims : Math.max(1, ...hs.map(p => p.found));
  const X = t => pad + (w - pad - 8) * t / tMax, Y = v => h - pad + 4 - (h - pad - 30) * v;
  g.strokeStyle = "#223044"; g.lineWidth = 1; g.beginPath();
  for (const v of [0, 0.5, 1]) { g.moveTo(pad, Y(v)); g.lineTo(w - 8, Y(v)); } g.stroke();
  const line = (vals, color) => { g.strokeStyle = color; g.lineWidth = 2; g.beginPath(); hs.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, X(p.t), Y(vals(p)))); g.stroke(); };
  line(p => p.mapped ?? p.coverage, "#a78bfa"); line(p => p.coverage, "#39d0d8"); line(p => p.found / top, "#3fd46b");
  let x = pad;
  for (const [text, color] of [["mapped", "#a78bfa"], ["searched", "#39d0d8"], [`victims found (of ${top})`, "#3fd46b"]]) {
    g.fillStyle = color; g.fillRect(x, 8, 10, 3); g.fillText(text, x + 14, 14); x += g.measureText(text).width + 30;
  }
  g.fillStyle = "#8b99a8"; g.textAlign = "right";
  g.fillText("100%", pad - 2, Y(1) + 4); g.fillText("0", pad - 2, Y(0) + 4); g.fillText(`${tMax} s`, w - 8, h - 4);
}
