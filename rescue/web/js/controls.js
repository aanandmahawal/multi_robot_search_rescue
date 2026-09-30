// Everything the user can press: mission setup, the transport bar on the 3-D view, the picture
// switch (Combined / LiDAR / Thermal), the Layers menu, the panel switches, keyboard shortcuts,
// and presets in the address bar.
import { EXPLAIN, LAYERS, PHASE_TEXT, SENSORS, SPEEDS, VIEWS } from "./config.js";
import { store } from "./store.js";
import { $ } from "./util.js";

const SELECTS = ["building", "damage", "strategy", "planner", "coverage", "avoidance", "battery"];
const SLIDERS = ["robots", "victims", "rspeed", "camrange", "lidarrange", "density"];
const SEGS = [["knownSeg", "known"], ["varySeg", "vary"], ["revisitSeg", "revisit"], ["deadendSeg", "deadend"], ["rechargeSeg", "recharge"]];
const EACH_WH = ["0", "10", "5", "3", "2", "1.5", "1"];      // choices for one robot's battery ("0" = unlimited)

// Battery = "Different per robot": one small choice per robot
function batteryEachList() {
  const n = +$("robots").value;
  while (store.batteryEach.length < n) store.batteryEach.push(["3", "2", "1.5", "1"][store.batteryEach.length % 4]);
  return store.batteryEach.slice(0, n);
}

let onBatteryChange = () => {};
function renderBatteryEach(onChange = onBatteryChange) {
  const box = $("batteryEach"), each = $("battery").value === "each";
  box.hidden = !each;
  $("rechargeRow").hidden = $("battery").value === "0";
  if (!each) return;
  box.innerHTML = batteryEachList().map((v, i) => `<label>R${i}<select data-i="${i}">` +
    EACH_WH.map(w => `<option value="${w}"${w === v ? " selected" : ""}>${w === "0" ? "∞" : w + " Wh"}</option>`).join("") +
    `</select></label>`).join("");
  box.querySelectorAll("select").forEach(s => s.onchange = () => { store.batteryEach[+s.dataset.i] = s.value; onChange(); });
}

// the settings of the next mission, as the server wants them
export const missionParams = () => ({
  building: $("building").value, damage: $("damage").value, strategy: $("strategy").value, sensors: store.sensors,
  robots: $("robots").value, victims: $("victims").value, known: store.known, seed: store.seed,
  planner: $("planner").value, coverage: $("coverage").value, avoidance: $("avoidance").value,
  speed: $("rspeed").value, battery: $("battery").value === "each" ? 0 : $("battery").value, recharge: store.recharge,
  ...($("battery").value === "each" ? { battery_each: batteryEachList().join(",") } : {}),
  camrange: $("camrange").value, lidarrange: $("lidarrange").value,
  density: $("density").value, revisit: store.revisit, deadend: store.deadend,
  // "Vary each run": new random numbers for every run in the same building
  ...(store.vary === "1" ? { run: Math.floor(Math.random() * 1e6) } : {}),
  ...store.preset,
});

export function refreshHints() {
  $("hint-building").textContent = EXPLAIN.building[$("building").value];
  $("hint-damage").textContent = EXPLAIN.damage[$("damage").value];
  $("hint-known").textContent = EXPLAIN.known[store.known];
  $("hint-strategy").textContent = EXPLAIN.strategy[$("strategy").value];
  $("hint-density").textContent = EXPLAIN.density;
  $("hint-vary").textContent = EXPLAIN.vary[store.vary];
  $("hint-planner").textContent = EXPLAIN.planner[$("planner").value];
  $("hint-coverage").textContent = EXPLAIN.coverage[$("coverage").value];
  $("hint-recharge").textContent = EXPLAIN.recharge[store.recharge];
  renderBatteryEach();
  $("hint-avoidance").textContent = EXPLAIN.avoidance[$("avoidance").value];
  $("hint-revisit").textContent = EXPLAIN.revisit[store.revisit];
  $("hint-deadend").textContent = EXPLAIN.deadend[store.deadend];
  $("hint-speed").textContent = EXPLAIN.speed;
  $("hint-battery").textContent = EXPLAIN.battery;
  $("hint-ranges").textContent = EXPLAIN.ranges;
  $("countRow").hidden = store.known !== "1";
  for (const id of SLIDERS) $(id + "Out").textContent = $(id).value;
  $("densityOut").textContent = $("density").value + "×";
  $("rspeedOut").textContent = $("rspeed").value;
  for (const [seg, key] of SEGS)
    document.querySelectorAll(`#${seg} button`).forEach(b => b.classList.toggle("on", b.dataset[key] === store[key]));
  document.querySelectorAll("#sensorOptions .option").forEach(b => b.classList.toggle("on", b.dataset.sensors === store.sensors));
}

// the transport bar follows the phase of the mission
export function syncTransport() {
  const p = store.phase;
  $("status").className = "status " + p;
  $("statusText").textContent = PHASE_TEXT[p];
  $("play").textContent = { loading: "…", ready: "▶ Start", running: "⏸ Pause", paused: "▶ Resume", finished: "↻ Run again" }[p];
  $("play").disabled = p === "loading";
  $("stepBtn").disabled = p === "loading" || p === "finished";
  $("restart").disabled = p === "loading";
  $("callout").hidden = p !== "ready";
}

// the picture switch: a picture is only offered when the robots carry the sensor for it
export function syncViewAs() {
  const world = store.world;
  document.querySelectorAll("#viewAsSeg button").forEach((b, i) => {
    const v = VIEWS.find(v => v.key === b.dataset.as), ok = !v.needs || !world || world[v.needs];
    b.disabled = !ok; b.classList.toggle("on", v.key === store.viewAs);
    b.setAttribute("aria-pressed", v.key === store.viewAs);
    b.title = ok ? `${v.help} (key ${i + 1})` : `Not available: these robots carry no ${v.key === "lidar" ? "LiDAR" : "camera"}`;
  });
  $("view").className = "look-" + store.viewAs;
  const extras = [world && world.has_lidar && "the laser's hits", world && world.has_cameras && "the warm spots the thermal cameras found"].filter(Boolean);
  $("viewAsHelp").textContent = store.viewAs === "combined" ? `The building as it is, with ${extras.join(" and ") || "what the robots sensed"}.`
                                                            : VIEWS.find(v => v.key === store.viewAs).help;
  $("viewAsHelp").hidden = !store.legend && store.viewAs === "combined";      // the sensor pictures always say what they show
}

function setKnown(v) {
  store.known = v;
  document.querySelectorAll("#knownSeg button").forEach(b => b.classList.toggle("on", b.dataset.known === v));
}

// the segmented switches of the setup panel (Runs, Revisits, Dead ends): store key -> value
function setSeg(key, v) { store[key] = v; refreshHints(); }

export function setSpeed(v) {
  store.speed = v;
  document.querySelectorAll("#speedSeg button").forEach(b => b.classList.toggle("on", +b.dataset.speed === v));
}

export function setLayer(key, on) {
  store.layers[key] = on;
  const box = document.querySelector(`#layers input[data-layer="${key}"]`);
  if (box) box.checked = on;
}

export function initControls(h) {
  // ---- mission setup: any change prepares a new mission, which then waits for Start
  let timer = null;
  const changed = newBuilding => {
    if (newBuilding) store.seed = Math.floor(Math.random() * 100000);
    refreshHints(); clearTimeout(timer); timer = setTimeout(h.onNewMission, 300);
  };
  for (const id of ["building", "damage", "density"]) $(id).addEventListener("change", () => changed(true));
  for (const id of ["strategy", "planner", "coverage", "avoidance", "battery", "robots", "victims", "rspeed", "camrange", "lidarrange"])
    $(id).addEventListener("change", () => changed(false));
  onBatteryChange = () => changed(false);
  const battery = () => renderBatteryEach();
  $("battery").addEventListener("change", battery);
  $("robots").addEventListener("change", battery);
  battery();
  for (const [seg, key] of SEGS.slice(1))
    document.querySelectorAll(`#${seg} button`).forEach(b => b.onclick = () => { setSeg(key, b.dataset[key]); changed(false); });
  for (const id of SLIDERS) $(id).addEventListener("input", refreshHints);
  document.querySelectorAll("#knownSeg button").forEach(b => b.onclick = () => { setKnown(b.dataset.known); changed(false); });
  $("newBuilding").onclick = () => changed(true);
  document.querySelectorAll(".info").forEach(b => b.onclick = () => {
    const open = $(b.dataset.hint).classList.toggle("open");
    b.classList.toggle("on", open);
  });
  $("sensorOptions").innerHTML = SENSORS.map(s =>
    `<button class="option" data-sensors="${s.key}"><i></i><b>${s.title}</b><span>${s.text}</span></button>`).join("");
  document.querySelectorAll("#sensorOptions .option").forEach(b => b.onclick = () => {
    if (store.sensors === b.dataset.sensors) return;
    store.sensors = b.dataset.sensors; store.preset = {};          // a choice made here replaces address-bar presets
    changed(false);
  });

  // ---- transport bar
  $("speedSeg").innerHTML = SPEEDS.map(s => `<button data-speed="${s}">${s}×</button>`).join("");
  document.querySelectorAll("#speedSeg button").forEach(b => b.onclick = () => setSpeed(+b.dataset.speed));
  setSpeed(store.speed);
  $("play").onclick = h.onPlay;
  $("stepBtn").onclick = h.onStep;
  $("restart").onclick = h.onNewMission;

  // ---- view: picture, camera presets, layers, legend
  $("viewAsSeg").innerHTML = VIEWS.map((v, i) =>
    `<button class="pic" data-as="${v.key}"><i class="ico"></i><span><b>${v.title}</b><small>${v.sub}</small></span><kbd>${i + 1}</kbd></button>`).join("");
  document.querySelectorAll("#viewAsSeg button").forEach(b => b.onclick = () => h.onViewAs(b.dataset.as));
  document.querySelectorAll("#viewSeg button").forEach(b => b.onclick = () => h.onView(b.dataset.view));
  $("layers").innerHTML = LAYERS.map(([k, label]) =>
    `<label><input type="checkbox" data-layer="${k}"${store.layers[k] ? " checked" : ""}>${label}</label>`).join("");
  document.querySelectorAll("#layers input").forEach(box => box.onchange = () => { store.layers[box.dataset.layer] = box.checked; h.onLayers(); });
  $("layersBtn").onclick = e => { e.stopPropagation(); $("layers").hidden = !$("layers").hidden; $("layersBtn").classList.toggle("on", !$("layers").hidden); };
  $("layers").onclick = e => e.stopPropagation();
  document.addEventListener("click", () => { $("layers").hidden = true; $("layersBtn").classList.remove("on"); });
  $("legendBtn").onclick = () => { store.legend = !store.legend; $("legendBtn").classList.toggle("on", store.legend); syncViewAs(); h.onLayers(); };

  // ---- side panels
  const togglePanel = (btn, cls) => $(btn).onclick = () => {
    const off = $("app").classList.toggle(cls); $(btn).classList.toggle("on", !off);
    setTimeout(h.onResize, 60);                          // the 3-D view has a new width
  };
  togglePanel("toggleSetup", "no-setup"); togglePanel("toggleDetails", "no-details");
  $("toggleSetup").classList.add("on"); $("toggleDetails").classList.add("on");

  // ---- keyboard: space = start / pause, full stop = one second, 1 2 3 = picture
  document.addEventListener("keydown", e => {
    if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); h.onPlay(); }
    if (e.key === ".") h.onStep();
    if (["1", "2", "3"].includes(e.key)) h.onViewAs(VIEWS[+e.key - 1].key);
  });

  applyPresets();
  refreshHints();
  syncTransport();
  syncViewAs();
}

// presets in the address bar, e.g.
//   ?building=parking&damage=severe&sensors=cameras&show=thermal&known=1&robots=4&seed=42&view=top&tab=sensors&speed=12
function applyPresets() {
  const q = new URLSearchParams(location.search);
  for (const id of [...SELECTS, ...SLIDERS]) if (q.has(id)) $(id).value = q.get(id);
  if (SENSORS.some(s => s.key === q.get("sensors"))) store.sensors = q.get("sensors");
  for (const k of ["vision", "lidar"]) if (q.has(k)) store.preset[k] = q.get(k);       // for tests: e.g. vision=ideal
  if (q.has("known")) setKnown(q.get("known") === "1" ? "1" : "0");
  if (q.has("vary")) store.vary = q.get("vary") === "1" ? "1" : "0";
  if (q.has("revisit")) store.revisit = q.get("revisit") === "0" ? "0" : "0.3";
  if (q.has("deadend")) store.deadend = q.get("deadend") === "0" ? "0" : "1";
  if (q.has("recharge")) store.recharge = q.get("recharge") === "0" ? "0" : "1";
  if (q.has("battery_each")) { $("battery").value = "each"; store.batteryEach = q.get("battery_each").split(","); }
  if (q.has("seed")) store.seed = +q.get("seed");
  if (q.has("view")) store.viewMode = q.get("view");
  if (VIEWS.some(v => v.key === q.get("show"))) store.viewAs = q.get("show");
  if (q.has("tab")) store.tab = q.get("tab");
  if (q.has("speed") && SPEEDS.includes(+q.get("speed"))) setSpeed(+q.get("speed"));
  if (q.get("setup") === "0") { $("app").classList.add("no-setup"); $("toggleSetup").classList.remove("on"); }
  if (q.get("details") === "0") { $("app").classList.add("no-details"); $("toggleDetails").classList.remove("on"); }
  if (q.get("legend") === "1") { store.legend = true; $("legendBtn").classList.add("on"); }
  store.autostart = q.get("autostart") === "1";
}
