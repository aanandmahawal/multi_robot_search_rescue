// Entry point: starts a mission, runs the playback loop, and connects the controls to it.
//
// A mission goes through these phases:
//   loading -> ready (waits for Start) -> running <-> paused -> finished
import { api, REQUEST_MS } from "./api.js";
import { initControls, missionParams, setLayer, syncTransport, syncViewAs } from "./controls.js";
import { clearSelection, initInspector, refreshInspector } from "./inspect.js";
import { initPanels, resetPanels, selectRobot, setOnSelect, showTab, updatePanels } from "./panels.js";
import { camera, initScene, lookAt, refit, renderer, setView } from "./scene.js";
import { store } from "./store.js";
import { $, sleep } from "./util.js";
import { availableView, buildWorld, initWorld, renderLegend, updateWorld } from "./world3d.js";

function showError(msg) { $("error").hidden = !msg; $("error").textContent = msg || ""; }
function setPhase(p) { store.phase = p; syncTransport(); }
async function whenIdle() { while (store.busy) await sleep(20); }

// one request to the server at a time
async function withServer(work) {
  await whenIdle();
  store.busy = true;
  try { await work(); showError(""); return true; }
  catch (e) { showError(e.message); console.error(e.stack || e); return false; }
  finally { store.busy = false; }
}

function redraw() { if (store.state) { updateWorld(); renderLegend(); syncViewAs(); refreshInspector(); } }

async function newMission({ start = false } = {}) {
  setPhase("loading");
  const ok = await withServer(async () => {
    const j = await api.reset(missionParams());
    Object.assign(store, { world: j.world, state: j.state, prevState: null, lastAt: performance.now() });
    clearSelection(); buildWorld(); redraw(); resetPanels(); updatePanels(true);
  });
  setPhase(!ok ? "ready" : start ? "running" : "ready");
}

async function advance(n) {
  if (store.busy || !store.world) return;
  const ok = await withServer(async () => {
    const next = await api.step(n, store.layers.search ? store.selected : null);
    // robots glide from the previous state to the new one (in a straight line: fine for a few cells)
    Object.assign(store, { prevState: n <= 4 ? store.state : null, state: next, lastAt: performance.now() });
    updateWorld(); updatePanels(); refreshInspector();
  });
  if (!ok) setPhase("paused");
  else if (store.state.finished) setPhase("finished");
}

function play() {
  const p = store.phase;
  if (p === "ready" || p === "paused") setPhase("running");
  else if (p === "running") setPhase("paused");
  else if (p === "finished") newMission({ start: true });
}

async function stepOnce() {
  if (store.phase === "loading" || store.phase === "finished") return;
  setPhase("paused");
  await whenIdle();
  await advance(1);
}

async function reveal() {
  await withServer(async () => { store.state = await api.reveal(); redraw(); updatePanels(true); });
}

// which picture of the building is drawn: combined | lidar | thermal
function setViewAs(key) {
  store.viewAs = availableView(key);
  redraw();
  if (!store.state) syncViewAs();
}

// the playback loop: while the mission is running, ask the server for the next second(s)
async function tick() {
  // seconds per request: enough that the request's fixed cost is shared by several steps
  const n = Math.min(30, Math.max(1, Math.round(store.speed / 8), Math.round(store.speed * REQUEST_MS / 1000)));
  store.frameDur = 1000 * n / store.speed;
  const t0 = performance.now();
  if (store.phase === "running") await advance(n);
  setTimeout(tick, Math.max(10, store.frameDur - (performance.now() - t0)));
}

// ------------------------------------------------------------------ start
initScene($("view"));
initWorld();
initInspector();
initPanels({ onReveal: reveal });
initControls({
  onNewMission: () => newMission(),
  onPlay: play,
  onStep: stepOnce,
  onView: setView,
  onViewAs: setViewAs,
  onLayers: redraw,
  onResize: refit,
});
showTab(store.tab);
// a newly selected robot: ask the server for its planner search right away (one step of 0 s is not possible,
// so the search appears with the next step; while paused, redraw what we have)
setOnSelect(async () => {
  if (store.state && store.layers.search && !store.busy && store.phase !== "running")
    await withServer(async () => { store.state = await api.state(store.selected); });
  redraw();
});

// for automated checks and screenshots (see README): drive the page from the browser console
window.rescueDebug = {
  advance: async n => { if (store.phase === "ready") setPhase("paused"); await whenIdle(); await advance(n); },
  play, reveal, setView, clearSelection, setViewAs, lookAt, selectRobot, showTab,
  pause: () => { if (store.phase === "running") setPhase("paused"); },
  setLayer: (k, on) => { setLayer(k, on); redraw(); },
  get state() { return store.state; }, get world() { return store.world; }, get phase() { return store.phase; },
  get viewAs() { return store.viewAs; },
  // where a point of the building (metres) appears on the page, e.g. to double-click it in a test
  screenOf(x, y, h = 0.3) {
    const p = camera.position.clone().set(x, h, y).project(camera), r = renderer.domElement.getBoundingClientRect();
    return [r.left + (p.x + 1) / 2 * r.width, r.top + (1 - p.y) / 2 * r.height];
  },
};

newMission({ start: store.autostart }).then(tick);
