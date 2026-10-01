// Open test ground only: a layout editor for your own obstacles.
//
//   Edit layout   turns editing on. Editing is only possible before the mission starts: pressing Start
//                 ends it, and while a mission runs the editor is locked (Restart to edit again).
//   Add tool      a click on free floor adds the chosen obstacle there (centred on the cell). A green ghost
//                 shows where it will go; a red one means the spot is taken.
//   Remove tool   the obstacle under the mouse is outlined in red; a click removes it. Obstacles are picked
//                 on their 3-D shape, not on the floor behind them, so a tall block can be clicked anywhere.
//   The list      every obstacle with its size and position; ✕ removes it, hovering it highlights it.
//   Undo, Default, Clear all, and Esc to stop editing.
// A click only counts if the mouse did not move (a drag still orbits the view), and double-clicks are
// ignored while editing, so nothing is ever added twice. Every change rebuilds the ground with the new
// layout; the server refuses an obstacle that would seal off part of the ground, and the note says so.
import * as THREE from "three";
import { camera, renderer, scene } from "./scene.js";
import { store } from "./store.js";
import { $, esc } from "./util.js";
import { cellOfInstance, pickables } from "./world3d.js";

const HEIGHT = { cabinet: 1.3, shelf: 1.8, counter: 0.95, crate: 0.3 };
const NAME = { cabinet: "Tall block", shelf: "Wall", counter: "Counter", crate: "Low crate" };
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let ghost = null, onChange = () => {}, history = [], listHover = -1;

export function initEditor(change) {
  onChange = change;
  const geo = new THREE.BoxGeometry(1, 1, 1); geo.translate(0, 0.5, 0);
  ghost = new THREE.Group();
  ghost.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x3fd46b, transparent: true, opacity: 0.35, depthWrite: false })));
  ghost.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x3fd46b, depthTest: false })));
  ghost.visible = false; ghost.renderOrder = 15; scene.add(ghost);

  const canvas = renderer.domElement;
  let down = null;
  canvas.addEventListener("pointerdown", e => { down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null; });
  canvas.addEventListener("click", e => {
    const d = down; down = null;
    if (!editing() || e.detail > 1 || !d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return;   // drags orbit, double-clicks are ignored
    act(target(e));
  });
  canvas.addEventListener("pointermove", e => { if (!e.buttons && editing()) show(target(e)); });
  canvas.addEventListener("pointerleave", () => { if (listHover < 0) ghost.visible = false; });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && store.editObstacles) setEditing(false); });

  $("obsEdit").onclick = () => setEditing(!store.editObstacles);
  document.querySelectorAll("#obsToolSeg button").forEach(b => b.onclick = () => setTool(b.dataset.tool));
  $("obsKind").addEventListener("change", () => { if (store.editTool !== "add") setTool("add"); else syncEditor(); });
  $("obsUndo").onclick = () => { if (history.length) commit(history.pop(), false); };
  $("obsDefault").onclick = () => commit(null);
  $("obsClear").onclick = () => commit([]);
  $("obsRestart").onclick = () => { store.editAfterReset = true; onChange(); };
  $("obsList").addEventListener("click", e => {
    const b = e.target.closest("button[data-i]");
    if (b) { const list = current().filter((_, k) => k !== +b.dataset.i); commit(list); }
  });
  $("obsList").addEventListener("pointerover", e => { const row = e.target.closest("[data-row]"); highlight(row ? +row.dataset.row : -1); });
  $("obsList").addEventListener("pointerleave", () => highlight(-1));
}

// ------------------------------------------------------------------ state
const plain = () => !!store.world && store.world.config.building === "plain";
export const canEdit = () => plain() && store.phase === "ready";
export const editing = () => store.editObstacles && canEdit();

export function setEditing(on) {
  if (on && plain() && store.phase !== "ready") { store.editAfterReset = true; onChange(); return; }   // Edit = restart first
  store.editObstacles = !!on && canEdit();
  if (!store.editObstacles) ghost.visible = false;
  syncEditor();
}

function setTool(tool) { store.editTool = tool; ghost.visible = false; syncEditor(); }

// the obstacles as [kind, x, y, w, h] in cells: the user's list, or what the ground has now
function current() {
  if (store.obstacles) return store.obstacles;
  return (store.world ? store.world.furniture : []).map(f => [f.kind, ...f.nav]);
}

function commit(list, remember = true) {
  if (remember) history.push(store.obstacles ? store.obstacles.map(o => o.slice()) : null);
  if (history.length > 50) history.shift();
  store.obstacles = list;
  store.keepCamera = true;                       // the ground is rebuilt: keep looking at the same spot
  store.editAfterReset = store.editObstacles;    // ... and keep editing if we were
  ghost.visible = false;
  onChange();
}

// ------------------------------------------------------------------ what is under the mouse
const inside = (o, cx, cy) => cx >= o[1] && cx < o[1] + o[3] && cy >= o[2] && cy < o[2] + o[4];
const overlaps = (a, b) => a[1] < b[1] + b[3] && b[1] < a[1] + a[3] && a[2] < b[2] + b[4] && b[2] < a[2] + a[4];

// { obstacle: index or -1, cell: [x, y] on the floor or null }
function target(e) {
  const r = renderer.domElement.getBoundingClientRect(), w = store.world, list = current();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  for (const h of ray.intersectObjects(pickables(), true)) {             // the first obstacle hit in 3-D wins
    if (!h.object.isInstancedMesh) continue;
    const c = cellOfInstance(h.object, h.instanceId);
    if (c < 0) continue;
    const k = list.findIndex(o => inside(o, c % w.W, Math.floor(c / w.W)));
    if (k >= 0) return { obstacle: k, cell: null };
    break;                                                               // a wall or something else in front
  }
  const p = new THREE.Vector3();
  if (!ray.ray.intersectPlane(floor, p)) return { obstacle: -1, cell: null };
  const cx = Math.floor(p.x / w.cell), cy = Math.floor(p.z / w.cell);
  const k = list.findIndex(o => inside(o, cx, cy));
  if (k >= 0) return { obstacle: k, cell: null };
  return { obstacle: -1, cell: cx >= 4 && cx < w.W - 1 && cy >= 1 && cy < w.H - 1 ? [cx, cy] : null };
}

function proposal([cx, cy]) {
  const [kind, w, h] = $("obsKind").value.split(",");
  return [kind, cx - Math.floor((+w - 1) / 2), cy - Math.floor((+h - 1) / 2), +w, +h];
}

function fits(o) {
  const w = store.world;
  return o[1] >= 4 && o[2] >= 1 && o[1] + o[3] <= w.W - 1 && o[2] + o[4] <= w.H - 1 && !current().some(b => overlaps(o, b));
}

// ------------------------------------------------------------------ act and preview
function act(t) {
  if (store.editTool === "remove") {
    if (t.obstacle >= 0) commit(current().filter((_, k) => k !== t.obstacle));
    return;
  }
  if (!t.cell) return;                                                   // add: only on free floor
  const o = proposal(t.cell);
  if (fits(o)) commit([...current(), o]);
  else note("That spot is taken or too close to the wall. Pick free floor, or switch to Remove.", true);
}

function show(t) {
  const list = current();
  if (store.editTool === "remove") {
    if (t.obstacle >= 0) place(list[t.obstacle], 0xff5a5c); else ghost.visible = false;
    return;
  }
  if (t.obstacle >= 0) { place(list[t.obstacle], 0xff5a5c); return; }    // add mode over an obstacle: taken
  if (!t.cell) { ghost.visible = false; return; }
  const o = proposal(t.cell);
  place(o, fits(o) ? 0x3fd46b : 0xff5a5c);
}

function place(o, color) {
  const c = store.world.cell, [kind, x, y, w, h] = o;
  ghost.scale.set(w * c + 0.04, (HEIGHT[kind] || 1) + 0.04, h * c + 0.04);
  ghost.position.set((x + w / 2) * c, 0, (y + h / 2) * c);
  ghost.children[0].material.color.set(color); ghost.children[1].material.color.set(color);
  ghost.visible = true;
}

function highlight(k) {
  listHover = k;
  document.querySelectorAll("#obsList [data-row]").forEach(r => r.classList.toggle("hot", +r.dataset.row === k));
  if (k >= 0 && current()[k]) place(current()[k], 0xffb020); else if (!editing()) ghost.visible = false;
}

function note(text, warn = false) { const n = $("obsNote"); n.textContent = text; n.classList.toggle("warn", warn); }

// ------------------------------------------------------------------ the panel
export function syncEditor() {
  if (!$("plainTools")) return;
  const on = store.editObstacles, ok = canEdit(), list = current(), c = store.world ? store.world.cell : 0.5;
  $("obsEdit").classList.toggle("on", on);
  $("obsEdit").textContent = on ? "✓ Done" : "✎ Edit layout";
  $("obsEdit").hidden = !ok && plain();
  $("obsRestart").hidden = ok || !plain();
  document.querySelectorAll("#obsToolSeg button").forEach(b => { b.classList.toggle("on", b.dataset.tool === store.editTool); b.disabled = !on; });
  for (const id of ["obsKind", "obsDefault", "obsClear"]) $(id).disabled = !ok;
  $("obsUndo").disabled = !ok || !history.length;
  $("view").classList.toggle("editing", on);
  $("editBanner").hidden = !on;
  $("editBanner").innerHTML = store.editTool === "remove"
    ? "<b>Editing layout · Remove</b> click an obstacle to delete it (it turns red under the mouse) · <kbd>Esc</kbd> done"
    : `<b>Editing layout · Add</b> click free floor to place a ${esc($("obsKind").selectedOptions[0].text.toLowerCase())} · <kbd>Esc</kbd> done`;
  $("obsCount").textContent = `${list.length} obstacle${list.length === 1 ? "" : "s"}`;
  $("obsList").innerHTML = list.map(([kind, x, y, w, h], k) =>
    `<div class="orow" data-row="${k}"><i class="sw ${kind}"></i><span>${NAME[kind] || kind} ${(w * c).toFixed(1)} × ${(h * c).toFixed(1)} m` +
    `<small>at ${((x + w / 2) * c).toFixed(1)}, ${((y + h / 2) * c).toFixed(1)} m</small></span>` +
    (ok ? `<button class="close" data-i="${k}" title="Remove this obstacle">✕</button>` : "") + `</div>`).join("")
    || `<p class="fine">No obstacles: an empty hall.</p>`;
  if (!ok && plain()) note(LOCKED);
  else if ($("obsNote").textContent === LOCKED) note("");
}
const LOCKED = "The layout is locked while a mission runs. Press Restart to edit it again.";

// after a new ground was built: keep exactly what was placed, say if something was refused, reopen the editor
export function afterWorld() {
  if (!plain()) { store.obstacles = null; store.editObstacles = false; history = []; note(""); syncEditor(); return; }
  if (store.obstacles) {
    const wanted = store.obstacles.length, placed = store.world.furniture.length;
    store.obstacles = store.world.furniture.map(f => [f.kind, ...f.nav]);
    note(placed < wanted ? "That obstacle was not placed: it would seal off part of the ground." : "", placed < wanted);
  } else note("");
  if (store.editAfterReset) { store.editAfterReset = false; store.editObstacles = true; }
  syncEditor();
}

// the obstacles for the next mission, as the server wants them ("none" = an empty ground; absent = default)
export const obstacleParam = () => !store.obstacles ? {}
  : { obstacles: store.obstacles.length ? store.obstacles.map(o => o.join(",")).join(";") : "none" };
