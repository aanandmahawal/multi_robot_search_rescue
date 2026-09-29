// The top bar numbers, the mission summary, and the details panel (one tab at a time).
// Only the open tab is redrawn after a step, which keeps the page light.
//
// Every count of people comes from one place, state.reports:
//   confirmed by the robots = real victims + false alarms
import { api } from "./api.js";
import { drawChart, drawLidarPlot, drawVictimMap } from "./charts.js";
import { CAMERA_CAPTION, PLANNER_NAMES, ROBOT_COLORS, STATE_TEXT, STATUS } from "./config.js";
import { missionParams } from "./controls.js";
import { lookAt, snapFollow } from "./scene.js";
import { store, truthVisible } from "./store.js";
import { $, esc, fmtS, hex } from "./util.js";

let camTick = 0, bannerClosed = false, onReveal = () => {};
const robotColor = i => hex(ROBOT_COLORS[i % ROBOT_COLORS.length]);
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const falseAlarms = () => store.state.candidates.filter(c => c.status === "confirmed" && !c.truth.real);

export function initPanels(handlers) {
  onReveal = handlers.onReveal;
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
  $("cmpRun").onclick = startCompare;
  $("exportMap").onclick = () => store.state && api.downloadVictimMap(`victim_map_${store.world.config.building}_t${store.state.t}s.json`);
}

export function showTab(name) {
  store.tab = name;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.hidden = t.id !== "tab-" + name);
  if (store.state) renderTab(true);
}

export function selectRobot(i) {
  store.selected = i; snapFollow();
  if (store.state) renderTab(true);
  onSelect();
}
let onSelect = () => {};
export const setOnSelect = fn => { onSelect = fn; };

// a new mission: forget what belonged to the old one
export function resetPanels() {
  bannerClosed = false; camTick = 0;
  const chips = $("robotChips"); chips.innerHTML = "";
  store.state.robots.forEach((r, i) => {
    const b = document.createElement("button");
    b.textContent = "R" + r.id; b.style.setProperty("--c", robotColor(i));
    b.onclick = () => selectRobot(i);
    chips.appendChild(b);
  });
  for (const id of ["camThermal", "camColour"]) $(id).removeAttribute("src");
}

export function updatePanels(force = false) {
  renderKpis();
  renderBanner();
  renderTab(force);
}

// ------------------------------------------------------------------ top bar
function renderKpis() {
  const { world, state } = store, m = state.metrics, rep = state.reports, known = world.config.victims_known;
  $("k-time").textContent = state.t + " s";
  if (!world.has_cameras) {
    // the user chose "LiDAR" under Sensors on the robots: this team carries a laser and no camera
    $("k-found").textContent = "–"; $("k-found-sub").textContent = "LiDAR-only team: no camera to find people"; $("k-false").hidden = true;
  } else {
    $("k-found").textContent = known ? `${rep.real} / ${world.config.n_victims}` : `${rep.real}`;
    $("k-found-sub").textContent = known ? "victims found" : "victims found (total unknown)";
    $("k-false").hidden = !rep.false;
    $("k-false").textContent = `+ ${plural(rep.false, "false alarm")}`;
  }
  const noCam = "You chose LiDAR under Sensors on the robots: this team carries a laser and no camera. " +
                "A laser measures distances; it cannot recognise people. Choose Combined or Vision + Thermal to search for victims.";
  $("k-found-box").title = world.has_cameras ? "Locations the robots confirmed, checked against the truth: real victims and false alarms" : noCam;
  $("k-searched-box").title = world.has_cameras ? "Share of the reachable floor a camera has looked at" : noCam;
  $("k-searched-sub").textContent = world.has_cameras ? "searched by cameras" : "searched: no camera fitted";
  $("k-searched").textContent = world.has_cameras ? Math.round(100 * m.coverage) + "%" : "–";
  $("k-searched-bar").style.width = (100 * m.coverage) + "%";
  $("k-mapped").textContent = Math.round(100 * m.mapped) + "%";
  $("k-mapped-bar").style.width = (100 * m.mapped) + "%";
  $("badge-victims").textContent = rep.real || "";
}

// ------------------------------------------------------------------ mission summary
function renderBanner() {
  const { world, state } = store, m = state.metrics, rep = state.reports, cfg = world.config, banner = $("banner");
  banner.hidden = !state.finished || bannerClosed;
  if (banner.hidden) return;
  let html = `<button class="close" id="bannerClose" title="Hide">✕</button><b>Mission complete at ${state.t} s</b><br>`;
  if (!world.has_cameras) {
    html += `${Math.round(100 * m.mapped)}% of the reachable building mapped by LiDAR.<br>` +
            `You chose <b>LiDAR</b> only, so these robots carry no camera, and a laser cannot recognise people: nobody could be found. ` +
            `Choose <b>Combined</b> or <b>Vision + Thermal</b> under <i>Sensors on the robots</i> to search for victims.`;
  } else {
    html += `${Math.round(100 * m.coverage)}% of the reachable building searched · the robots reported ${rep.confirmed} ${rep.confirmed === 1 ? "person" : "people"}: ` +
            `<b>${rep.real}</b> real victim${rep.real === 1 ? "" : "s"}` + (rep.false ? ` and ${plural(rep.false, "false alarm")}` : "");
    if (cfg.victims_known) html += (rep.real >= cfg.n_victims ? " · all expected victims accounted for" : ` · ${cfg.n_victims - rep.real} still missing`) +
                                   (state.truth_summary ? `<br>${esc(state.truth_summary)}` : "");
    else if (!state.revealed) html += `<br><button id="revealBtn">Reveal how many people were really inside (simulation only)</button>`;
    else html += `<br>${esc(state.truth_summary || "")}`;
  }
  banner.innerHTML = html;
  $("bannerClose").onclick = () => { bannerClosed = true; banner.hidden = true; };
  if ($("revealBtn")) $("revealBtn").onclick = () => { bannerClosed = false; onReveal(); };
}

// ------------------------------------------------------------------ tabs
function renderTab(force) {
  ({ victims: renderVictims, robots: renderRobots, sensors: () => renderSensors(force), map: drawVictimMap,
     stats: renderStats, log: renderLog })[store.tab]();
}

function renderVictims() {
  const { world, state } = store, rep = state.reports, known = world.config.victims_known, truth = truthVisible();
  $("victimTitle").textContent = known ? `${world.config.n_victims} people are known to be inside`
                                 : state.revealed ? "Ground truth revealed" : "Victims found so far";
  $("victimNote").textContent = !world.has_cameras ? "These robots carry only a LiDAR. A laser cannot recognise people, so nobody will be found."
    : truth ? "You can see every victim; the robots still have to find them. Click one to fly there."
    : "Nobody knows how many people are inside, so only victims the robots have found are listed. Click one to fly there.";
  $("reportSum").hidden = !world.has_cameras;
  $("reportSum").innerHTML = `<span><b>${rep.confirmed}</b>confirmed by the robots</span><span class="real"><b>${rep.real}</b>real</span>` +
    `<span class="false"><b>${rep.false}</b>false</span><span class="open"><b>${rep.suspected}</b>being checked</span>`;
  $("repeatNote").hidden = !rep.repeats;
  $("repeatNote").textContent = `${plural(rep.repeats, "repeated report")} of a person already on the list ${rep.repeats === 1 ? "was" : "were"} merged ` +
    `(head and legs of one body can look like two warm shapes).`;

  const bodies = Object.fromEntries(state.victims.map(v => [v.id, v]));
  $("victimList").innerHTML = state.victim_status.map(s => {
    const st = STATUS[s.status], v = bodies[s.id];
    const right = s.status === "found" ? `Found at ${s.t} s<br><span>by R${s.by.join(", R")}</span>` : st.text;
    return `<div class="vrow" data-id="${s.id}" style="--c:${st.css}"><i class="flag"></i>
            <div><div class="name">Victim ${s.id}</div><div class="sub">${esc(v.describe)}</div></div><div class="st">${right}</div></div>`;
  }).join("") || (world.has_cameras ? `<p class="fine">No victims found yet.</p>` : "");
  document.querySelectorAll(".vrow").forEach(el => el.onclick = () => { const v = bodies[+el.dataset.id]; lookAt(v.x, v.y); });

  const wrong = falseAlarms();
  $("falseTitle").hidden = $("falseNote").hidden = !wrong.length;
  $("falseTitle").textContent = `${plural(wrong.length, "false alarm")}`;
  $("falseList").innerHTML = wrong.map((c, i) => {
    const temp = c.temp != null ? ` · measured ${Math.round(c.temp)} °C` : "";
    return `<div class="frow" data-i="${i}"><i>✕</i><div><div class="name">Really ${esc(c.truth.text)}</div>
            <div class="sub">at (${c.x.toFixed(1)}, ${c.y.toFixed(1)}) m · confirmed at ${c.confirmed_at} s by R${c.by.join(", R")}${temp}</div></div></div>`;
  }).join("");
  document.querySelectorAll(".frow").forEach(el => el.onclick = () => { const c = wrong[+el.dataset.i]; lookAt(c.x, c.y); });
}

function renderRobots() {
  const { world, state } = store;
  $("robotList").innerHTML = state.robots.map((r, i) => {
    const zone = world.config.strategy === "partition" ? `zone Z${r.zone}` : `${r.distance} m driven`;
    const p = r.plan;
    const route = p ? `${PLANNER_NAMES[p.algorithm] || p.algorithm} route ${p.length_m} m · ${p.work} ${p.algorithm === "rrtstar" ? "samples" : p.algorithm === "aco" ? "ant steps" : "cells expanded"} · ${p.ms} ms` +
                      (p.note ? ` · <span class="warn">${esc(p.note)}</span>` : "") : "no route planned yet";
    const bat = r.battery == null ? "" : `<span class="batt${r.battery < 0.2 ? " low" : ""}"><i style="width:${Math.round(100 * r.battery)}%"></i></span>${Math.round(100 * r.battery)}%`;
    const repeat = r.moves ? Math.round(100 * r.repeat_moves / r.moves) : 0;
    return `<div class="rrow${i === store.selected ? " sel" : ""}" data-i="${i}" style="--c:${robotColor(i)}">
      <div class="top">R${r.id} · ${STATE_TEXT[r.state] || r.state}<span>${zone}</span></div>
      <div class="why">${esc(r.reason || "")}</div>
      <div class="nav">${route}</div>
      <div class="nav">${r.energy_wh} Wh used ${bat} · ${repeat}% of moves over cells I had crossed · ${r.replans} replans · ${r.deadends} dead ends · ${r.conflicts} waits</div></div>`;
  }).join("");
  document.querySelectorAll(".rrow").forEach(el => el.onclick = () => selectRobot(+el.dataset.i));
}

function renderSensors(force) {
  const { world, state } = store, r = state.robots[store.selected], vision = world.config.vision;
  [...$("robotChips").children].forEach((b, i) => b.classList.toggle("on", i === store.selected));
  $("camSection").hidden = !world.has_cameras; $("noCamera").hidden = world.has_cameras;
  $("lidarSection").hidden = !world.has_lidar; $("noLidar").hidden = world.has_lidar;
  if (world.has_cameras) {
    if (force || !(camTick++ % 2)) {                     // the camera images are the costly part: every second update
      for (const [id, kind] of [["camThermal", "thermal"], ["camColour", "colour"]]) {
        api.cameraImage(store.selected, kind, state.t).then(src => {
          const img = new Image(); img.onload = () => { $(id).src = img.src; };
          img.src = src;
        }).catch(() => {});
      }
    }
    const hot = r.too_hot ? ` ${plural(r.too_hot, "detection")} dropped as too hot for a person (above 40 °C).` : "";
    $("thermalInfo").textContent = `Dark = cold, bright = warm; the crosshair marks the hottest pixel. ${CAMERA_CAPTION.thermal[vision]}${hot}`;
    $("camInfo").textContent = `R${r.id} is ${STATE_TEXT[r.state] || r.state}: ${plural(r.detections, "detection")} so far. ${CAMERA_CAPTION.colour[vision]}`;
  }
  if (world.has_lidar) drawLidarPlot();
}

function renderStats() {
  const { world, state } = store, m = state.metrics, rep = state.reports, cams = world.has_cameras;
  const rows = [
    [cams ? rep.confirmed : "–", "locations confirmed by the robots"],
    [cams ? rep.real : "–", "of them real victims"],
    [cams ? rep.false : "–", "of them false alarms", rep.false > 0],
    [cams && rep.confirmed ? Math.round(100 * rep.real / rep.confirmed) + "%" : "–", "confirmed reports that were real people"],
    [fmtS(m.time_first_victim), "first victim found at"],
    [fmtS(m.time_last_find), "latest victim found at"],
    [fmtS(m.time_90pct_mapped), "90% of the building mapped at"],
    [cams ? fmtS(m.time_90pct_coverage) : "–", "90% of the building searched at"],
    [Math.round(m.distance_m) + " m", "distance driven (team)"],
    [m.bumps, "collisions: bumper hits (obstacles not on the map)"],
    [m.energy_wh.toFixed(1) + " Wh", "energy used (team)" + (m.batteries_emptied ? `, ${m.batteries_emptied} battery empty` : "")],
    [Math.round(100 * m.repeat_ratio) + "%", "moves over cells already driven through", m.repeat_ratio > 0.5],
    [m.robot_conflicts, "waits because a teammate was in the way"],
    [m.dead_ends, "dead ends / loops recovered from"],
    [m.replans, "routes replanned: new obstacle on the route"],
    [m.plan_ms_mean.toFixed(1) + " ms", `planning time per route (${m.routes_planned} routes)`],
    [cams && m.detection_rate != null ? Math.round(100 * m.detection_rate) + "%" : "–", "detection rate: found / findable victims" + (m.detection_rate == null ? " (shown once the truth is revealed)" : "")],
    [cams ? m.redundancy.toFixed(2) : "–", "robots that looked at each cell"],
    [Math.round(100 * m.goal_overlap) + "%", "robot pairs aiming at the same spot"],
  ];
  $("metrics").innerHTML = rows.map(([v, k, bad]) => `<div class="metric${bad ? " bad" : ""}"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  drawChart();
}

// ------------------------------------------------------------------ compare algorithms
const CMP_COLS = [
  ["mission_time", "time", v => v + " s", "min"],
  ["coverage", "searched", v => Math.round(100 * v) + "%", "max"],
  ["found", "victims", v => v, "max"],
  ["false_alarms", "false", v => v, "min"],
  ["distance_m", "distance", v => Math.round(v) + " m", "min"],
  ["energy_wh", "energy", v => v.toFixed(1) + " Wh", "min"],
  ["collisions", "bumps", v => v, "min"],
  ["repeat_ratio", "repeats", v => Math.round(100 * v) + "%", "min"],
  ["robot_conflicts", "waits", v => v, "min"],
  ["dead_ends", "dead ends", v => v, "min"],
  ["plan_ms_mean", "plan", v => v.toFixed(1) + " ms", "min"],
];
const OPTION_NAMES = { ...PLANNER_NAMES, frontier: "Frontier", boustrophedon: "Lawnmower", spiral: "Spiral", none: "Follow route",
  dwa: "DWA", coordinated: "Coordinated", partition: "Partition", greedy: "Greedy", random: "Random" };

async function startCompare() {
  const vary = $("cmpDim").value, fast = $("cmpFast").checked ? "1" : "0";
  const params = { ...missionParams(), vary, fast };
  if (!params.run) params.run = store.seed;                     // identical random choices for every option
  $("cmpRun").disabled = true;
  $("cmpOut").innerHTML = `<p class="fine">Starting…</p>`;
  try {
    const { job } = await api.compare(params);
    store.compareJob = job;
    pollCompare(job);
  } catch (e) { $("cmpOut").innerHTML = `<p class="note">${esc(e.message)}</p>`; $("cmpRun").disabled = false; }
}

async function pollCompare(job) {
  if (store.compareJob !== job) return;
  let s;
  try { s = await api.compareStatus(job); } catch (e) { $("cmpOut").innerHTML = `<p class="note">${esc(e.message)}</p>`; $("cmpRun").disabled = false; return; }
  renderCompare(s);
  if (s.done < s.total) setTimeout(() => pollCompare(job), 1500);
  else $("cmpRun").disabled = false;
}

function renderCompare(s) {
  const done = s.rows.filter(r => !r.pending && !r.error);
  const best = {};
  for (const [k, , , dir] of CMP_COLS) {
    const vals = done.map(r => r[k]);
    if (vals.length > 1) best[k] = dir === "min" ? Math.min(...vals) : Math.max(...vals);
  }
  const head = `<tr><th></th>${CMP_COLS.map(([, label]) => `<th>${label}</th>`).join("")}</tr>`;
  const rows = s.rows.map(r => {
    const name = OPTION_NAMES[r.option] || r.option;
    if (r.pending) return `<tr><td>${name}</td><td colspan="${CMP_COLS.length}" class="pending">running…</td></tr>`;
    if (r.error) return `<tr><td>${name}</td><td colspan="${CMP_COLS.length}" class="warn">${esc(r.error)}</td></tr>`;
    return `<tr><td>${name}${r.all_home ? "" : ' <span class="warn" title="not every robot got back to base">!</span>'}</td>` +
      CMP_COLS.map(([k, , f]) => `<td class="${best[k] !== undefined && r[k] === best[k] ? "best" : ""}">${f(r[k])}</td>`).join("") + "</tr>";
  }).join("");
  const st = s.settings;
  $("cmpOut").innerHTML = `<p class="fine">${s.done} / ${s.total} done · ${s.seconds} s · ${st.building}, ${st.damage} damage, seed ${st.seed}, ` +
    `${st.n_robots} robots${s.fast ? ", perfect-eyes vision" : ""}. Everything else as set on the left.</p>` +
    `<div class="cmp-wrap"><table class="cmp">${head}${rows}</table></div>`;
}

function renderLog() {
  $("log").innerHTML = store.state.events.slice().reverse().map(([t, e]) => {
    const cls = e.startsWith("CONFIRMED") || e.startsWith("ALL") ? "confirmed" : e.includes("rejected") ? "rejected" : e.includes("too hot") ? "hot"
      : e.includes("heat signature") ? "thermal" : e.includes("spotted") ? "spotted" : e.includes("closer look") ? "verify" : e.includes("returns to base") ? "base" : "";
    return `<div class="${cls}"><span class="t">${t}s</span>${esc(e)}</div>`;
  }).join("");
}
