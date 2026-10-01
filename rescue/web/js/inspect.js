// Double-click anything in the 3-D view to select it: the camera glides to it, it becomes the
// centre of rotation and zoom, a marker shows the spot, and a card tells what is there and what
// each sensor of the team knows about it. Also: the zoom buttons and the mouse-help chip.
import * as THREE from "three";
import { STATE_TEXT, STATUS } from "./config.js";
import { selectRobot } from "./panels.js";
import { camera, focusOn, onFrame, renderer, scene, setView, zoomBy } from "./scene.js";
import { store } from "./store.js";
import { $, cellTemp, esc } from "./util.js";
import { cellOfInstance, pickables, thingAt } from "./world3d.js";

const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let marker = null, pick = null;             // pick: { point, robot?, victim? } of the selection

export function initInspector() {
  marker = makeMarker(); scene.add(marker.group);
  const canvas = renderer.domElement;
  canvas.addEventListener("dblclick", e => {
    if (store.editObstacles) return;                                  // clicks place obstacles while editing
    const hit = pickAt(e);
    if (hit) select(hit); else clearSelection();
  });
  // the cursor shows whether a double-click would pick something (checked at most 8 times a second)
  let lastMove = 0;
  canvas.style.cursor = "grab";
  canvas.addEventListener("pointermove", e => {
    if (e.buttons || performance.now() - lastMove < 120) return;
    lastMove = performance.now();
    canvas.style.cursor = pickAt(e) ? "crosshair" : "grab";
  });
  canvas.addEventListener("pointerdown", () => { canvas.style.cursor = "grabbing"; });
  canvas.addEventListener("pointerup", () => { canvas.style.cursor = "grab"; });

  $("zoomIn").onclick = () => zoomBy(0.65);
  $("zoomOut").onclick = () => zoomBy(1.5);
  $("zoomFit").onclick = () => { clearSelection(); setView("overview"); };
  $("inspClose").onclick = clearSelection;
  makeDraggable($("inspector"), $("inspHandle"));
  $("inspCenter").onclick = () => pick && focusOn(pick.point, 5);
  $("inspFollow").onclick = () => { if (pick && pick.robot != null) { selectRobot(pick.robot); clearSelection(); setView("follow"); } };

  const seen = (() => { try { return localStorage.getItem("rt-navhelp") === "off"; } catch { return false; } })();
  $("navHelp").hidden = seen;
  $("navHelpClose").onclick = () => { $("navHelp").hidden = true; try { localStorage.setItem("rt-navhelp", "off"); } catch { /* private window */ } };
  $("navHelpBtn").onclick = () => { $("navHelp").hidden = !$("navHelp").hidden; };

  document.addEventListener("keydown", e => {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === "Escape") clearSelection();
    if (e.key === "+" || e.key === "=") zoomBy(0.75);
    if (e.key === "-" || e.key === "_") zoomBy(1.33);
    if (e.key === "f" || e.key === "F") { if (pick) focusOn(pick.point, 5); else setView("overview"); }
  });

  onFrame(time => {
    if (!marker.group.visible) return;
    const s = 1 + 0.12 * Math.sin(time * 4);
    marker.ring.scale.set(s, s, 1);
    marker.box.material.opacity = 0.65 + 0.3 * Math.sin(time * 4);
    if (pick && pick.robot != null) {                                // a robot moves: the marker goes with it
      const R = pick.object; marker.group.position.set(R.position.x, 0, R.position.z);
    }
  });
}

// the first visible, solid thing under the mouse
function pickAt(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  for (const h of ray.intersectObjects(pickables(), true)) {
    const o = h.object;
    if (o.isSprite || o.isPoints || o.isLine || !shown(o)) continue;
    const m = Array.isArray(o.material) ? o.material[0] : o.material;
    if (m && ((m.transparent && m.opacity < 0.5) || m.blending === THREE.AdditiveBlending)) continue;   // camera cones, sweeps, glows
    let robot = null, victim = null, owner = null;
    for (let p = o; p; p = p.parent) {
      if (p.userData.robot != null) { robot = p.userData.robot; owner = p; break; }
      if (p.userData.victim != null) { victim = p.userData.victim; break; }
    }
    return { point: h.point.clone(), robot, victim, object: owner, cell: o.isInstancedMesh ? cellOfInstance(o, h.instanceId) : -1 };
  }
  return null;
}
const shown = o => { for (; o; o = o.parent) if (!o.visible) return false; return true; };

function select(hit) {
  pick = hit;
  const { world } = store, c = world.cell;
  let cx = Math.floor(hit.point.x / c), cy = Math.floor(hit.point.z / c);
  if (hit.cell >= 0) { cx = hit.cell % world.W; cy = Math.floor(hit.cell / world.W); }    // the object itself, not the floor under the hit
  pick.cell = cx >= 0 && cy >= 0 && cx < world.W && cy < world.H ? cy * world.W + cx : -1;
  if (hit.robot != null) selectRobot(hit.robot);

  // marker: a ring on the floor, and the outline of the cell up to the height of what stands there
  const thing = pick.cell >= 0 ? thingAt(pick.cell) : null;
  const h = hit.robot != null ? 0.7 : thing ? Math.max(0.25, thing.height) : 0.15;
  const small = hit.robot != null || hit.victim != null;
  marker.group.position.set(hit.robot != null ? hit.object.position.x : (cx + 0.5) * c, 0, hit.robot != null ? hit.object.position.z : (cy + 0.5) * c);
  if (hit.victim != null) marker.group.position.set(hit.point.x, 0, hit.point.z);
  marker.box.scale.set(small ? 0.9 : c * 1.04, h + 0.04, small ? 0.9 : c * 1.04);
  marker.ring.position.y = 0.03;
  marker.group.visible = true;

  const close = hit.robot != null || hit.victim != null ? 4.5 : 8;
  focusOn(new THREE.Vector3(marker.group.position.x, Math.min(hit.point.y, 1), marker.group.position.z), close);
  refreshInspector();
}

export function clearSelection() {
  pick = null;
  if (marker) marker.group.visible = false;
  $("inspector").hidden = true;
}

// the card: what is at the selected spot and what each sensor of the team knows about it
export function refreshInspector() {
  const card = $("inspector");
  if (!pick || !store.state) { card.hidden = true; return; }
  const { world, state } = store, i = pick.cell;
  let title, sub = "", note = "";
  if (pick.robot != null) {
    const r = state.robots[pick.robot];
    title = `Robot R${r.id}`; sub = STATE_TEXT[r.state] || r.state; note = r.reason || "";
  } else if (pick.victim != null) {
    const s = state.victim_status.find(v => v.id === pick.victim), v = state.victims.find(v => v.id === pick.victim);
    title = `Victim ${pick.victim}`; sub = s ? STATUS[s.status].text + (s.status === "found" ? ` at ${s.t} s` : "") : "";
    note = v ? v.describe : "";
  } else {
    const thing = i >= 0 ? thingAt(i) : null;
    title = i < 0 ? "Outside the building" : thing ? thing.name : "Floor";
    note = thing ? thing.note : "";
  }
  const x = pick.point.x.toFixed(1), y = pick.point.z.toFixed(1);
  $("inspTitle").textContent = title;
  $("inspSub").textContent = [sub, `at ${x} m, ${y} m`].filter(Boolean).join(" · ");
  $("inspNote").textContent = note; $("inspNote").hidden = !note;
  $("inspFollow").hidden = pick.robot == null;

  const rows = [];
  if (i >= 0) {
    const k = state.known[i];
    rows.push(["lidar", "LiDAR map", !world.has_lidar ? "no LiDAR on these robots"
      : k === "2" ? "obstacle: a beam hit something here" : k === "1" ? "free: beams passed through" : "not mapped yet"]);
    const T = cellTemp(state.thermal_map, i);
    rows.push(["thermal", "Thermal", !world.has_cameras ? "no thermal camera on these robots"
      : T === null ? "no thermal camera has looked here" : `${T.toFixed(1)} °C · ` + (T - world.ambient >= 2.5 ? `${(T - world.ambient).toFixed(1)} °C warmer than the air`
        : T - world.ambient <= -1.5 ? `${(world.ambient - T).toFixed(1)} °C colder than the air` : "about air temperature")]);
    rows.push(["camera", "Cameras", !world.has_cameras ? "no camera on these robots"
      : state.searched[i] === "1" ? "searched: a camera has looked here" : "not searched yet"]);
  }
  $("inspRows").innerHTML = rows.map(([cls, k, v]) => `<div class="irow ${cls}"><i></i><span>${k}</span><b>${esc(v)}</b></div>`).join("");
  card.hidden = false;
}

// the card can be dragged anywhere over the view by its title; it stays where you leave it
function makeDraggable(card, handle) {
  const view = $("view");
  const place = (x, y) => {
    const v = view.getBoundingClientRect(), w = card.offsetWidth, h = card.offsetHeight;
    x = Math.min(Math.max(0, x), v.width - w); y = Math.min(Math.max(0, y), v.height - h);
    card.style.left = x + "px"; card.style.top = y + "px"; card.style.bottom = "auto";
    return [x, y];
  };
  try { const p = JSON.parse(localStorage.getItem("rs-inspector") || "null"); if (p) requestAnimationFrame(() => place(...p)); } catch { /* no storage */ }
  handle.addEventListener("pointerdown", e => {
    if (e.target.closest("button")) return;
    e.preventDefault();
    const c = card.getBoundingClientRect(), dx = e.clientX - c.left, dy = e.clientY - c.top;
    card.classList.add("dragging");
    const move = ev => { const v = view.getBoundingClientRect(); place(ev.clientX - v.left - dx, ev.clientY - v.top - dy); };
    const up = ev => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
      card.classList.remove("dragging");
      const v = view.getBoundingClientRect();
      try { localStorage.setItem("rs-inspector", JSON.stringify(place(ev.clientX - v.left - dx, ev.clientY - v.top - dy))); } catch { /* no storage */ }
    };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  });
}

function makeMarker() {
  const group = new THREE.Group(); group.visible = false;
  const col = 0x4f9cf9;
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.52, 48),
    new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthTest: false, toneMapped: false }));
  ring.rotation.x = -Math.PI / 2; ring.renderOrder = 20;
  const boxGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)); boxGeo.translate(0, 0.5, 0);
  const box = new THREE.LineSegments(boxGeo, new THREE.LineBasicMaterial({ color: 0x9cc8ff, transparent: true, depthTest: false, toneMapped: false }));
  box.renderOrder = 21;
  group.add(ring, box);
  return { group, ring, box };
}
