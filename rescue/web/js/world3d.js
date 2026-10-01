// Turns the mission data into the 3-D picture: builds the building once per mission, then
// updates floor, colours, victims, sightings, robots and LiDAR scans after every step.
//
// The same building can be drawn as three pictures (store.viewAs):
//   combined   the building as it is, plus the laser's hits and the warm spots the cameras found
//   lidar      what the laser knows: obstacles it hit in violet, everything else dark
//   thermal    what the thermal cameras measured: the temperature of every cell they looked at
import * as THREE from "three";
import { animateVictim, makeBase, makeFalseAlarm, makeRobot, makeVictim, paintVictim, setRobotLabel, setScan, setVictimStatus } from "./actors.js";
import { createBuilder } from "./builders.js";
import { ROBOT_COLORS, VIEWS } from "./config.js";
import { lightBuilding, onFrame, scene, setFollowTarget, setLook, setView } from "./scene.js";
import { animateSensorFx, animateSweep, buildSensorFx, lidarMaterial, makeSweep, updateSensorFx } from "./sensorfx.js";
import { store, truthVisible } from "./store.js";
import { $, cellTemp, hex, ironbow, labelSprite, mulberry32 } from "./util.js";

const CELL_PX = 16;                       // floor texture: pixels per navigation cell
const UNSEEN = [9, 9, 14], LASER_HIT = [126, 96, 255], LASER_MISS = [20, 27, 40], LASER_DARK = [8, 11, 16];
let worldGroup = null, dynGroup = null, candGroup = null, searchGroup = null;
let floor, floorLit, floorFlat, floorCtx, floorTex, cellNoise;
let shaded = [], singles = [], robots3d = [], victims3d = {}, zoneLabels = [], base = null, lidarMat = null;
let things = new Map();                   // map cell -> { name, height, note } of what stands there (for the inspector)

export function initWorld() {
  setFollowTarget(() => robots3d.length ? robots3d[store.selected % robots3d.length].group : null);
  onFrame(frame);
}

// the picture that can be shown with the sensors these robots carry
export function availableView(key) {
  const v = VIEWS.find(v => v.key === key);
  return v && (!v.needs || (store.world && store.world[v.needs])) ? key : "combined";
}

// ------------------------------------------------------------------ build (once per mission)
export function buildWorld() {
  const { world, state } = store;
  for (const g of [worldGroup, dynGroup, candGroup]) if (g) scene.remove(g);
  worldGroup = new THREE.Group(); dynGroup = new THREE.Group(); candGroup = new THREE.Group();
  scene.add(worldGroup, dynGroup, candGroup);
  victims3d = {};
  const Wm = world.W * world.cell, Hm = world.H * world.cell;
  const rand = mulberry32(world.config.seed * 7919 + 13);
  lightBuilding(Wm, Hm);

  const canvas = document.createElement("canvas"); canvas.width = world.W * CELL_PX; canvas.height = world.H * CELL_PX;
  floorCtx = canvas.getContext("2d");
  floorTex = new THREE.CanvasTexture(canvas); floorTex.colorSpace = THREE.SRGBColorSpace; floorTex.anisotropy = 8;
  cellNoise = Array.from({ length: world.W * world.H }, () => rand());
  floorLit = new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.9 });
  floorFlat = new THREE.MeshBasicMaterial({ map: floorTex, toneMapped: false });        // sensor pictures: exact colours
  floor = new THREE.Mesh(new THREE.PlaneGeometry(Wm, Hm), floorLit);
  floor.rotation.x = -Math.PI / 2; floor.position.set(Wm / 2, 0, Hm / 2); floor.receiveShadow = true; worldGroup.add(floor);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: 0x161c24, roughness: 1 }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(Wm / 2, -0.02, Hm / 2); ground.receiveShadow = true; worldGroup.add(ground);
  buildSensorFx(worldGroup, ground);

  const b = createBuilder(world, worldGroup);
  b.structure(rand);
  b.rubble(rand);
  for (const it of world.furniture) b.furniture(it, rand);
  b.flush();
  world.decoys.forEach(b.decoy);
  world.warm_objects.forEach(b.warmObject);
  shaded = b.shaded; singles = b.singles;
  lidarMat = lidarMaterial();
  for (const s of shaded) s.base = s.mesh.material;
  for (const o of singles) o.lidar = lidarMaterial();
  things = describeCells(world);
  base = makeBase(world, worldGroup);

  zoneLabels = Object.entries(world.zone_of).map(([rid, z]) => {
    let sx = 0, sy = 0, n = 0;
    for (let i = 0; i < world.zones.length; i++) if (+world.zones[i] === z) { sx += i % world.W; sy += Math.floor(i / world.W); n++; }
    const s = labelSprite(`Zone of R${rid}`, hex(ROBOT_COLORS[rid % ROBOT_COLORS.length]), "#0b1016", 1.1);
    s.position.set((sx / n + 0.5) * world.cell, 3.2, (sy / n + 0.5) * world.cell); worldGroup.add(s);
    return s;
  });
  robots3d = state.robots.map((r, i) => {
    const R = makeRobot(r, i, world.config, dynGroup);
    R.group.userData.robot = i;
    if (world.has_lidar) { R.sweep = makeSweep(world.config); R.group.add(R.sweep); }
    return R;
  });
  store.selected = Math.min(store.selected, state.robots.length - 1);
  store.viewAs = availableView(store.viewAs);
  if (!store.keepCamera) setView(store.viewMode === "free" ? "overview" : store.viewMode);   // obstacle edits keep the view
  store.keepCamera = false;
}

// ------------------------------------------------------------------ colours of the sensor pictures
// thermal picture: colour of a temperature; the cold end is lifted a little so that "seen and cold"
// can be told from "not seen"
function heat(T) {
  const [lo, hi] = store.world.thermal_scale;
  return ironbow(Math.max(0.24, (T - lo) / (hi - lo)));
}
const cellOf = (x, y) => Math.floor(y / store.world.cell) * store.world.W + Math.floor(x / store.world.cell);
const thermalAt = i => { const T = cellTemp(store.state.thermal_map, i); return T === null ? UNSEEN : heat(T); };
const laserAt = i => store.state.known[i] === "2" ? LASER_HIT : store.state.known[i] === "1" ? LASER_MISS : LASER_DARK;

// ------------------------------------------------------------------ floor texture
function drawFloor() {
  const mode = store.viewAs, want = mode === "combined" ? floorLit : floorFlat;
  if (floor.material !== want) floor.material = want;
  ({ combined: floorScene, lidar: floorLidar, thermal: floorThermal })[mode]();
  floorTex.needsUpdate = true;
}

function floorLidar() {
  const { world, state } = store, W = world.W, H = world.H, P = CELL_PX, g = floorCtx;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const k = state.known[y * W + x], px = x * P, py = y * P;
    g.fillStyle = k === "2" ? "#3a2b86" : k === "1" ? "#0e1830" : "#03050a"; g.fillRect(px, py, P, P);
    if (k === "1") { g.fillStyle = "rgba(120,140,255,0.07)"; g.fillRect(px, py, P, 1); g.fillRect(px, py, 1, P); }
  }
}

// like a real thermal image: temperatures blend softly into each other (a blurred copy of the
// cell colours), then the cells no camera has looked at are hatched on top, sharp
let blurCanvas = null;
function floorThermal() {
  const { world, state } = store, W = world.W, H = world.H, P = CELL_PX, g = floorCtx, cv = g.canvas;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const T = cellTemp(state.thermal_map, y * W + x);
    const [r, gg, b] = T === null ? [10, 8, 18] : heat(T);
    g.fillStyle = `rgb(${r},${gg},${b})`; g.fillRect(x * P, y * P, P, P);
  }
  if (!blurCanvas || blurCanvas.width !== cv.width || blurCanvas.height !== cv.height) {
    blurCanvas = document.createElement("canvas"); blurCanvas.width = cv.width; blurCanvas.height = cv.height;
  }
  const bg = blurCanvas.getContext("2d");
  bg.filter = `blur(${Math.round(P * 0.45)}px)`; bg.clearRect(0, 0, cv.width, cv.height); bg.drawImage(cv, 0, 0);
  g.drawImage(blurCanvas, 0, 0);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (cellTemp(state.thermal_map, y * W + x) !== null) continue;
    const px = x * P, py = y * P;                         // no thermal camera has looked here: hatched
    g.fillStyle = "#0a0a10"; g.fillRect(px, py, P, P);
    g.fillStyle = "rgba(255,255,255,0.06)"; for (let k = 0; k < P; k += 4) g.fillRect(px + k, py + (P - 4 - k + P) % P, 2, 2);
  }
}

function floorScene() {
  const { world, state, layers } = store;
  const known = state.known, searched = state.searched, W = world.W, H = world.H, P = CELL_PX, g = floorCtx;
  const b = world.config.building, pal = world.floor_palette, [tLo, tHi] = world.thermal_scale;
  const fog = layers.fog && state.t > 0;                 // before Start the building is shown as it is
  const zonesOn = (layers.zones || layers.strategy) && (world.config.strategy === "partition" || !!world.regions);
  const zoneColor = {};
  for (const [rid, z] of Object.entries(world.zone_of)) zoneColor[z] = hex(ROBOT_COLORS[rid % ROBOT_COLORS.length]);
  const frontier = new Set(state.frontiers.map(c => c[1] * W + c[0]));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, n = cellNoise[i], ft = +world.floor[i], px = x * P, py = y * P;
    if (frontier.has(i)) { g.fillStyle = "#2cc3cc"; g.fillRect(px, py, P, P); continue; }
    if (known[i] === "0" && fog) {                        // not on the map yet
      g.fillStyle = "#171c23"; g.fillRect(px, py, P, P);
      if (zonesOn) { g.fillStyle = zoneColor[world.zones[i]] + "30"; g.fillRect(px, py, P, P); }
      continue;
    }
    const c = pal[ft], v = (n - 0.5) * 14;
    g.fillStyle = `rgb(${c[0] + v},${c[1] + v},${c[2] + v})`; g.fillRect(px, py, P, P);
    // floor pattern per building / area
    const wood = (b === "apartments" && ft === 0) || (b === "school" && ft === 2);
    if (wood) { g.fillStyle = "rgba(40,25,10,0.35)"; for (let k = 0; k < P; k += 4) g.fillRect(px, py + k, P, 1);
                if (n > 0.5) g.fillRect(px + Math.floor(n * P), py, 1, P); }
    else if (b === "parking") { if (n > 0.75) { g.fillStyle = "rgba(0,0,0,0.15)"; g.fillRect(px + 3, py + 5, 6, 4); } }
    else if (b === "office" && ft === 0) { if (n > 0.6) { g.fillStyle = "rgba(255,255,255,0.05)"; g.fillRect(px, py, P, P); } }
    else { g.fillStyle = "rgba(30,34,40,0.35)"; g.fillRect(px, py, P, 1); g.fillRect(px, py, 1, P);
           if (b === "warehouse" && (x % 8 === 0 || y % 8 === 0)) { g.fillStyle = "rgba(0,0,0,0.35)"; g.fillRect(px, py, x % 8 === 0 ? 2 : P, y % 8 === 0 ? 2 : P); } }
    if (fog && searched[i] === "0") {                     // on the map, but no camera has looked here
      g.fillStyle = "rgba(12,16,28,0.55)"; g.fillRect(px, py, P, P);
    }
    const T = cellTemp(state.thermal_map, i), warm = T === null ? 0 : T - world.ambient;      // warm spots the cameras found
    if (warm >= 2.5) { const [r, gg, bb] = ironbow((T - tLo) / (tHi - tLo)); g.fillStyle = `rgba(${r},${gg},${bb},${Math.min(0.92, 0.35 + warm / 12)})`; g.fillRect(px, py, P, P); }
    if (zonesOn) { g.fillStyle = zoneColor[world.zones[i]] + (world.regions ? "22" : "3a"); g.fillRect(px, py, P, P); }
    if (layers.visits && state.visits) {                  // driven through more than once: the more often, the redder
      const v = +state.visits[i];
      if (v > 1) { g.fillStyle = `rgba(255,60,110,${Math.min(0.65, 0.13 * (v - 1))})`; g.fillRect(px, py, P, P); }
    }
  }
  // painted markings (parking bays, gym court; the test ground's 2 m grid is kept faint)
  g.lineWidth = b === "plain" ? 1.5 : 3;
  g.strokeStyle = b === "parking" ? "rgba(245,245,235,0.8)" : b === "plain" ? "rgba(250,250,250,0.35)" : "rgba(250,250,250,0.75)";
  const s = P / world.cell;
  for (const [x1, y1, x2, y2] of world.markings) { g.beginPath(); g.moveTo(x1 * s, y1 * s); g.lineTo(x2 * s, y2 * s); g.stroke(); }
  drawPatterns(g, s);
  drawStrategy(g, s);
}

// the team strategy, drawn on the floor (Layers: strategy):
//   zones          partition, and the rectangles of a lawnmower / spiral: a border in each robot's colour
//   robot -> goal  a line in the robot's colour: solid = won in the auction or a zone / pattern goal,
//                  dashed = greedy (nearest), dotted = random
//   claims         coordinated: around each robot's goal a filled 2 m circle (no teammate goes there) and
//                  a dashed ring of the discount radius (teammates' goals inside it lose value)
function drawStrategy(g, s) {
  const { world, state, layers } = store;
  if (!layers.strategy) return;
  const strat = world.config.strategy, W = world.W, H = world.H, P = CELL_PX;
  const colOf = i => hex(ROBOT_COLORS[i % ROBOT_COLORS.length]);
  if (strat === "partition" || world.regions) {
    const zc = {};
    for (const [rid, z] of Object.entries(world.zone_of)) zc[z] = colOf(+rid);
    g.lineWidth = 3; g.setLineDash([]);
    const edge = (x1, y1, x2, y2, col) => { g.strokeStyle = col; g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); };
    for (let y = 1; y < H - 1; y++) for (let x = 4; x < W - 1; x++) {
      const z = world.zones[y * W + x];
      if (x + 1 < W - 1 && world.zones[y * W + x + 1] !== z) {             // vertical border: each side in its own colour
        edge((x + 1) * P - 2, y * P, (x + 1) * P - 2, (y + 1) * P, zc[z]);
        edge((x + 1) * P + 2, y * P, (x + 1) * P + 2, (y + 1) * P, zc[world.zones[y * W + x + 1]]);
      }
      if (y + 1 < H - 1 && world.zones[(y + 1) * W + x] !== z) {           // horizontal border
        edge(x * P, (y + 1) * P - 2, (x + 1) * P, (y + 1) * P - 2, zc[z]);
        edge(x * P, (y + 1) * P + 2, (x + 1) * P, (y + 1) * P + 2, zc[world.zones[(y + 1) * W + x]]);
      }
    }
  }
  const rad = world.config.utility_discount_radius || 4;
  state.robots.forEach((r, i) => {
    const goal = r.goal;
    if (!goal || goal.kind === "home" || goal.kind === "report") return;
    const col = colOf(i), gx = goal.x * s, gy = goal.y * s, rx = r.x * s, ry = r.y * s;
    if (strat === "coordinated" && (goal.kind === "frontier" || goal.kind === "verify")) {
      g.setLineDash([]); g.fillStyle = col + "2e"; g.beginPath(); g.arc(gx, gy, 2 * s, 0, 2 * Math.PI); g.fill();
      g.setLineDash([7, 6]); g.lineWidth = 2; g.strokeStyle = col + "cc"; g.beginPath(); g.arc(gx, gy, rad * s, 0, 2 * Math.PI); g.stroke();
    }
    const sweep = goal.kind === "sweep" || goal.kind === "transit";
    g.setLineDash(sweep ? [] : strat === "greedy" ? [12, 7] : strat === "random" ? [2, 7] : []);
    g.lineWidth = 4; g.strokeStyle = "#0b1016aa"; g.beginPath(); g.moveTo(rx, ry); g.lineTo(gx, gy); g.stroke();
    g.lineWidth = 2.5; g.strokeStyle = col; g.beginPath(); g.moveTo(rx, ry); g.lineTo(gx, gy); g.stroke();
    g.setLineDash([]);
    const a = Math.atan2(gy - ry, gx - rx), L = 14;                                   // arrow head at the goal
    if (Math.hypot(gx - rx, gy - ry) > 2 * L) {
      g.fillStyle = col; g.beginPath(); g.moveTo(gx, gy);
      g.lineTo(gx - L * Math.cos(a - 0.4), gy - L * Math.sin(a - 0.4)); g.lineTo(gx - L * Math.cos(a + 0.4), gy - L * Math.sin(a + 0.4)); g.fill();
    }
    g.lineWidth = 3; g.strokeStyle = col; g.beginPath(); g.arc(gx, gy, 9, 0, 2 * Math.PI); g.stroke();
  });
}

// a label above each robot's goal: why it goes there ("bid 4.6 = 6.0 − 0.5×2.7", "nearest · 1.4 m", "zone Z1 · 4.1 m")
function setGoalLabel(R, r, i, show) {
  const goal = r.goal, text = show && goal && goal.tag && goal.kind !== "home" ? `R${r.id} · ${goal.tag}` : "";
  if (R.goalText !== text) {
    R.goalText = text;
    if (R.goalLabel) { dynGroup.remove(R.goalLabel); R.goalLabel.material.map.dispose(); R.goalLabel.material.dispose(); R.goalLabel = null; }
    if (text) { R.goalLabel = labelSprite(text, hex(ROBOT_COLORS[i % ROBOT_COLORS.length]), "#0b1016", 0.62); dynGroup.add(R.goalLabel); }
  }
  if (R.goalLabel) R.goalLabel.position.set(goal.x, 1.45, goal.y);
}

// the card that explains the strategy on the floor
const STRAT_TEXT = {
  coordinated: ["Coordinated: an auction", "Every robot that needs a goal bids <b>utility = value − 0.5 × distance (m)</b> on each frontier or sighting; the best bid wins, then the next robot bids on what is left.",
    [["solid", "a robot and the goal it won (label: its winning bid)"], ["fill", "2 m round a won goal: no teammate goes there"], ["ring", "discount radius: teammates' goals inside lose value, so the team spreads out"]]],
  greedy: ["Greedy: nearest goal", "Every robot drives to its own nearest frontier or sighting and <b>ignores its teammates</b>. Watch them bunch up on the same opening.",
    [["dash", "a robot and its nearest goal (label: distance)"]]],
  partition: ["Partition: one zone per robot", "The building is split into zones; <b>each robot only searches inside its own zone</b> and never helps elsewhere.",
    [["zone", "zone borders, in each owner's colour"], ["solid", "a robot and the nearest goal in its zone"]]],
  random: ["Random: the baseline", "Every robot picks <b>any reachable goal at random</b>. It shows what the other strategies gain.",
    [["dot", "a robot and its random goal"]]],
};
const KEY_SVG = {
  solid: '<svg viewBox="0 0 34 14"><line x1="2" y1="7" x2="28" y2="7" stroke="#4f9cf9" stroke-width="3"/><path d="M26 2 L33 7 L26 12 z" fill="#4f9cf9"/></svg>',
  dash: '<svg viewBox="0 0 34 14"><line x1="2" y1="7" x2="28" y2="7" stroke="#4f9cf9" stroke-width="3" stroke-dasharray="7 4"/><path d="M26 2 L33 7 L26 12 z" fill="#4f9cf9"/></svg>',
  dot: '<svg viewBox="0 0 34 14"><line x1="2" y1="7" x2="28" y2="7" stroke="#4f9cf9" stroke-width="3" stroke-dasharray="1.5 4"/><path d="M26 2 L33 7 L26 12 z" fill="#4f9cf9"/></svg>',
  fill: '<svg viewBox="0 0 34 14"><circle cx="17" cy="7" r="6.5" fill="#4f9cf955"/></svg>',
  ring: '<svg viewBox="0 0 34 14"><circle cx="17" cy="7" r="6" fill="none" stroke="#4f9cf9" stroke-width="1.6" stroke-dasharray="3 2"/></svg>',
  zone: '<svg viewBox="0 0 34 14"><rect x="1" y="1" width="15" height="12" fill="#4f9cf930"/><rect x="18" y="1" width="15" height="12" fill="#f0883e30"/><line x1="15.5" y1="1" x2="15.5" y2="13" stroke="#4f9cf9" stroke-width="2"/><line x1="18.5" y1="1" x2="18.5" y2="13" stroke="#f0883e" stroke-width="2"/></svg>',
};
let stratKey = "";
function renderStratCard() {
  const { world, layers } = store, card = $("stratCard");
  card.hidden = !layers.strategy || !world;
  if (card.hidden) { stratKey = ""; return; }
  const strat = world.config.strategy, cov = world.config.coverage, key = strat + cov;
  if (key === stratKey) return;
  stratKey = key;
  const [title, text, keys] = STRAT_TEXT[strat];
  const pattern = cov !== "frontier"
    ? `<p style="margin-top:6px">Coverage is <b>${cov === "spiral" ? "a spiral" : "a lawnmower"}</b>: each robot first sweeps its own rectangle (coloured borders, label "${cov === "spiral" ? "spiral" : "lane"} wp n/N"); the ${strat} strategy only decides the mop-up afterwards.</p>` : "";
  const rows = (cov !== "frontier" && strat !== "partition" ? [["zone", "each robot's pattern rectangle"], ...keys] : keys)
    .map(([k, t]) => `${KEY_SVG[k]}<span>${t}</span>`).join("");
  card.innerHTML = `<button class="close x" title="Hide (Layers: strategy)">✕</button><b class="t">${title}</b>${text}${pattern}<div class="k">${rows}</div>`;
  card.querySelector(".x").onclick = () => {
    store.layers.strategy = false;
    const box = document.querySelector('#layers input[data-layer="strategy"]'); if (box) box.checked = false;
    updateWorld();
  };
}

// every robot's coverage pattern (dashed: dot = start, ring = next waypoint, red cross = skipped
// waypoint) and the track it has really driven (solid), painted onto the floor
function drawPatterns(g, s) {
  const { state, layers } = store;
  const line = pts => { g.beginPath(); pts.forEach(([x, y], k) => k ? g.lineTo(x * s, y * s) : g.moveTo(x * s, y * s)); g.stroke(); };
  g.lineCap = "round"; g.lineJoin = "round";
  state.robots.forEach((r, i) => {
    const col = hex(ROBOT_COLORS[i % ROBOT_COLORS.length]), sw = r.sweep;
    if (layers.pattern && sw) {
      g.setLineDash([10, 7]); g.lineWidth = 4; g.strokeStyle = col; line(sw.corners); g.setLineDash([]);
      g.fillStyle = col; g.beginPath(); g.arc(sw.corners[0][0] * s, sw.corners[0][1] * s, 9, 0, 2 * Math.PI); g.fill();
      if (sw.next) { g.lineWidth = 3; g.beginPath(); g.arc(sw.next[0] * s, sw.next[1] * s, 10, 0, 2 * Math.PI); g.stroke(); }
      g.strokeStyle = "#ff3b3b"; g.lineWidth = 3;
      for (const [x, y] of sw.skipped) { const d = 8; line([[x - d / s, y - d / s], [x + d / s, y + d / s]]); line([[x - d / s, y + d / s], [x + d / s, y - d / s]]); }
    }
    if (layers.trail && r.track && r.track.length > 1) {
      g.globalAlpha = 0.9; g.lineWidth = 3; g.strokeStyle = "#0b1016"; line(r.track);
      g.lineWidth = 1.8; g.strokeStyle = col; line(r.track); g.globalAlpha = 1;
    }
  });
}

// ------------------------------------------------------------------ objects: real colours or sensor colours
function paintObjects() {
  const { world, state, layers } = store, mode = store.viewAs, real = mode === "combined";
  const known = state.known, searched = state.searched, fog = layers.fog && state.t > 0, col = new THREE.Color();
  const sensor = mode === "thermal" ? thermalAt : laserAt;
  for (const s of shaded) {                              // walls, furniture, rubble (instanced)
    const want = mode === "lidar" ? lidarMat : s.base, map = real ? s.map : null;
    if (s.mesh.material !== want) s.mesh.material = want;
    if (want === s.base && s.base.map !== map) { s.base.map = map; s.base.needsUpdate = true; }
    for (let i = 0; i < s.colors.length; i++) {
      const k = s.nav[i];
      if (real) col.copy(s.colors[i]).multiplyScalar(!fog ? 1 : known[k] === "0" ? 0.38 : searched[k] === "0" ? 0.7 : 1);
      else { const [r, g, b] = sensor(k); col.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace); }
      s.mesh.setColorAt(i, col);
    }
    s.mesh.instanceColor.needsUpdate = true;
  }
  for (const o of singles) {                             // look-alikes and warm objects
    const flat = mode === "lidar" ? o.lidar : o.flat;
    if (!real) { const [r, g, b] = sensor(cellOf(o.x, o.y)); flat.color.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace); }
    for (const m of o.meshes) m.material = real ? m.userData.base : flat;
  }
  for (const vv of Object.values(victims3d)) {           // people
    const v = vv.v, i = cellOf(v.x, v.y);
    if (real) paintVictim(vv, null);
    else if (mode === "lidar") paintVictim(vv, { skin: LASER_MISS, cloth: LASER_MISS, cover: v.cover === "thin" ? LASER_MISS : laserAt(i) });
    else {
      const T = cellTemp(state.thermal_map, i);
      if (T === null) paintVictim(vv, { skin: UNSEEN, cloth: UNSEEN, cover: UNSEEN });
      else if (v.cover === "thin") paintVictim(vv, { skin: heat(T), cloth: heat(T), cover: heat(T) });       // faint, through the debris
      else paintVictim(vv, { skin: heat(v.skin_temp), cloth: heat(v.skin_temp - 5.5), cover: heat(world.ambient) });
    }
  }
}

// ------------------------------------------------------------------ update (after every step, or when the picture changes)
export function updateWorld() {
  const { world, state, layers } = store;
  if (!world || !state) return;
  const mode = store.viewAs = availableView(store.viewAs);
  setLook(mode);
  drawFloor();
  updateSensorFx(mode);
  renderHud();
  zoneLabels.forEach(l => l.visible = mode === "combined" && (layers.zones || layers.strategy) && (world.config.strategy === "partition" || !!world.regions));
  renderStratCard();

  // victims appear when the page first hears of them (unknown count: only once found)
  for (const v of state.victims) if (!victims3d[v.id]) {
    victims3d[v.id] = makeVictim(v, world, worldGroup);
    victims3d[v.id].group.userData.victim = v.id;
  }
  const visible = new Set(state.victim_status.map(s => s.id));
  for (const s of state.victim_status) setVictimStatus(victims3d[s.id], s.status);
  for (const [id, vv] of Object.entries(victims3d)) vv.group.visible = layers.flags && visible.has(+id);
  paintObjects();

  // what the robots believe: sightings they are unsure about, places they rejected, and false alarms
  scene.remove(candGroup); candGroup = new THREE.Group(); scene.add(candGroup);
  if (layers.sightings) {
    for (const c of state.candidates) {
      if (c.status === "confirmed") { if (!c.truth.real) makeFalseAlarm(c, candGroup); continue; }
      if (c.status === "rejected") { const x = labelSprite("✕ rejected", "#3a4350", "#c9d1d9", 0.55); x.position.set(c.x, 0.6, c.y); candGroup.add(x); continue; }
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.45, 0.6, 32), new THREE.MeshBasicMaterial({ color: 0xffb020, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
      ring.rotation.x = -Math.PI / 2; ring.position.set(c.x, 0.06, c.y); candGroup.add(ring);
      const temp = c.temp != null ? ` · ${Math.round(c.temp)}°C` : "";
      const q = labelSprite(`? ${Math.round(100 * c.p)}%${temp}`, "#ffb020", "#1a1300", temp ? 0.85 : 0.6); q.position.set(c.x, 1.0, c.y); candGroup.add(q);
    }
  }

  drawSearch();

  state.robots.forEach((r, i) => {
    const R = robots3d[i];
    const pts = [new THREE.Vector3(r.x, 0.08, r.y), ...r.path.map(p => new THREE.Vector3(p[0], 0.08, p[1]))];
    R.path.geometry.dispose(); R.path.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    R.path.visible = layers.paths && r.path.length > 0;
    R.goal.visible = layers.paths && !!r.goal && r.goal.kind !== "home";
    if (r.goal) R.goal.position.set(r.goal.x, 0.6, r.goal.y);
    R.cone.visible = layers.fov && world.has_cameras && mode !== "lidar";
    setRobotLabel(R, r, i);
    setGoalLabel(R, r, i, layers.strategy && mode === "combined");
    setScan(R, r, r.scan || [], world.config.lidar_height, world.has_lidar && mode !== "thermal");
  });
}

// what the selected robot's planner looked at for its current route: the cells Dijkstra / A* expanded,
// the RRT* trees, or the cells the ants left pheromone on
function drawSearch() {
  const { world, state, layers } = store;
  if (searchGroup) { scene.remove(searchGroup); searchGroup.traverse(o => o.geometry && o.geometry.dispose()); }
  searchGroup = new THREE.Group(); scene.add(searchGroup);
  const r = state.robots[store.selected];
  if (!layers.search || !r || !r.search || !r.search.length || !r.plan) return;
  const c = world.cell, col = new THREE.Color(ROBOT_COLORS[store.selected % ROBOT_COLORS.length]);
  if (r.plan.algorithm === "rrtstar") {
    const pos = new Float32Array(r.search.length * 6);
    r.search.forEach(([x1, y1, x2, y2], k) => pos.set([x1 * c, 0.12, y1 * c, x2 * c, 0.12, y2 * c], k * 6));
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    searchGroup.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: col.clone().lerp(new THREE.Color(0xffffff), 0.4),
      transparent: true, opacity: 0.55, depthWrite: false })));
    return;
  }
  const pos = new Float32Array(r.search.length * 3);
  r.search.forEach(([x, y], k) => pos.set([(x + 0.5) * c, 0.1, (y + 0.5) * c], k * 3));
  const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const ants = r.plan.algorithm === "aco";
  searchGroup.add(new THREE.Points(geo, new THREE.PointsMaterial({ color: ants ? 0xffc14d : col, size: ants ? 0.22 : 0.3,
    transparent: true, opacity: ants ? 0.8 : 0.35, depthWrite: false })));
}

// ------------------------------------------------------------------ every frame
const lerpAngle = (a, b, t) => { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * t; };

function frame(time) {
  const { state, prevState } = store;
  if (!state || !robots3d.length) return;
  const a = Math.min(1, (performance.now() - store.lastAt) / store.frameDur);
  const running = store.phase === "running";
  state.robots.forEach((r, i) => {
    const p = prevState && prevState.robots[i], R = robots3d[i];
    if (!R) return;
    R.group.position.set(p ? p.x + (r.x - p.x) * a : r.x, 0, p ? p.y + (r.y - p.y) * a : r.y);
    R.group.rotation.y = -(p ? lerpAngle(p.heading, r.heading, a) : r.heading);
    R.beacon.material.emissiveIntensity = 1 + Math.sin(time * 6 + i) * 0.9;
    R.goal.rotation.y = time * 2;
    if (R.spinner && running) R.spinner.rotation.y = time * 12;       // the laser turns while the mission runs
    R.label.visible = !(store.viewMode === "follow" && i === store.selected);
    if (R.sweep) {
      R.sweep.visible = running && store.viewAs !== "thermal";
      if (R.sweep.visible) animateSweep(R.sweep, time, i * 0.37);
    }
  });
  animateSensorFx(time);
  for (const vv of Object.values(victims3d)) animateVictim(vv, time);
  if (base) base.light.material.emissiveIntensity = 1 + Math.sin(time * 4);
}

// ------------------------------------------------------------------ what the inspector can ask
// the groups a double-click can hit, and what stands in a map cell
export const pickables = () => [worldGroup, dynGroup].filter(Boolean);
export const thingAt = i => things.get(i) || null;
// the map cell of one instance of the instanced walls, furniture and rubble (-1 if the mesh is not one of them)
export function cellOfInstance(mesh, id) {
  const s = shaded.find(s => s.mesh === mesh);
  return s && id != null ? s.nav[id] : -1;
}

const NAMES = {
  desk: "Office desk", table: "Table", school_desk: "School desk", bed: "Bed", hospital_bed: "Hospital bed", sofa: "Sofa",
  counter: "Kitchen counter", shelf: "Bookshelf", cabinet: "Filing cabinet", car: "Parked car", bleachers: "Bleachers",
  crate: "Low crate (under the LiDAR)",
};
const nice = k => NAMES[k] || k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, " ");

function describeCells(world) {
  const m = new Map(), W = world.W;
  const put = (i, name, height, note = "") => m.set(i, { name, height, note });
  for (const [x, y, h] of world.rubble) put(y * W + x, "Rubble pile", h, "Collapsed material. Cameras cannot see through thick rubble.");
  for (const it of world.furniture) {
    const [x0, y0, w, h] = it.nav;
    for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) put(y * W + x, nice(it.kind), it.height || 0.8);
  }
  const kind = { 1: "Wall", 2: "Pillar", 3: "Storage rack" };
  for (const [x, y, t, h] of world.structure) put(y * W + x, kind[t] || "Structure", t === 3 ? 2.0 : h);
  for (const d of world.decoys) put(cellOf(d.x, d.y), nice(d.kind), d.height || 0.3, "A look-alike: a colour camera can mistake it for a person.");
  for (const d of world.warm_objects) put(cellOf(d.x, d.y), nice(d.kind), d.height || 0.3, "Warm: a thermal camera can mistake it for a person.");
  return m;
}

// ------------------------------------------------------------------ instrument read-out of the sensor pictures
export function renderHud() {
  const { world, state } = store, el = $("hud"), mode = store.viewAs, cfg = world.config;
  el.hidden = mode === "combined";
  el.className = "hud " + mode;
  if (mode === "lidar") {
    const pct = Math.round(100 * state.metrics.mapped);
    el.innerHTML = `<b>LiDAR</b><span>2-D laser scanner</span>
      <dl><dt>scan plane</dt><dd>${Math.round(cfg.lidar_height * 100)} cm</dd><dt>range</dt><dd>${cfg.lidar_range} m</dd>
      <dt>beams / turn</dt><dd>${cfg.lidar_rays}</dd><dt>mapped</dt><dd>${pct}%</dd></dl><i class="bar"><i style="width:${pct}%"></i></i>`;
  } else if (mode === "thermal") {
    const [lo, hi] = world.thermal_scale;
    let top = null;
    for (let i = 0; i < world.W * world.H; i++) { const T = cellTemp(state.thermal_map, i); if (T !== null && (top === null || T > top)) top = T; }
    el.innerHTML = `<b>THERMAL</b><span>ironbow · °C</span>
      <div class="tscale"><i></i><em>${hi}+</em><em>${Math.round((lo + hi) / 2)}</em><em>${Math.round(lo)}</em></div>
      <dl><dt>air</dt><dd>${world.ambient} °C</dd><dt>hottest seen</dt><dd>${top === null ? "–" : top.toFixed(1) + " °C"}</dd></dl>`;
  }
}

// ------------------------------------------------------------------ legend
export function renderLegend() {
  const { world, state, layers } = store;
  const el = $("legend");
  el.hidden = !store.legend || !world;
  if (el.hidden) return;
  const sw = ([c, t]) => `<span><i class="sw" style="background:${c}"></i>${t}</span>`;
  const people = truthVisible()
    ? [["var(--found)", "victim found"], ["var(--hidden)", "victim not found yet"], ["var(--sighted)", "spotted, being checked"],
       ["var(--buried)", "buried under thick rubble: undetectable"]]
    : [["var(--found)", "victim found by the robots"]];
  if (world.has_cameras && state && state.reports.false) people.push(["var(--hidden)", "✕ false alarm: confirmed, but nobody is there"]);
  let html;
  if (store.viewAs === "thermal") {
    const [lo, hi] = world.thermal_scale;
    html = `<span class="scale"><i></i><b><em>${Math.round(lo)} °C</em><em>${Math.round((lo + hi) / 2)} °C</em><em>${hi} °C and hotter</em></b></span>` +
      [["repeating-linear-gradient(45deg,#0a0a10 0 3px,#2a2a33 3px 5px)", "no thermal camera has looked here yet"], ...people].map(sw).join("");
  } else if (store.viewAs === "lidar") {
    html = [["#7e60ff", "the laser hit something here"], ["#101c2b", "free, as far as the laser can tell"], ["#05080d", "not on the map yet"],
            ["var(--lidar)", "dots: where the latest beams ended"], ...people].map(sw).join("");
  } else {
    const items = [...people, ["var(--frontier)", "edge of the searched area"]];
    if (layers.pattern && world.regions) items.push(["repeating-linear-gradient(90deg,#4f9cf9 0 4px,transparent 4px 7px)", "coverage pattern (dot = start, ring = next waypoint, red × = waypoint skipped: inside an obstacle)"]);
    if (layers.trail) items.push(["#3a78c4", "where each robot has really driven (solid line)"]);
    if (layers.search) items.push(["#9cc8ff", "planner search of the selected robot: cells examined, RRT* tree, or ant trails (yellow)"]);
    if (layers.visits) items.push(["rgba(255,60,110,0.6)", "driven through more than once (redder = more often)"]);
    if (layers.fog) {
      if (world.has_lidar) items.push(["#4a5568", "mapped, not searched yet"]);
      items.push(["#171c23", "not on the map yet"]);
    }
    if (world.has_lidar) items.push(["var(--lidar)", "LiDAR: where the laser beams ended"]);
    if (world.has_cameras) items.push(["linear-gradient(90deg,#78008c,#f06414,#ffe66e)", "warm spot found by a thermal camera"]);
    html = items.map(sw).join("");
  }
  el.innerHTML = html;
}
