// Everything the page asks of the server. Every tab has its own mission, named by a random id.
//
// Two ways to reach the Python side, chosen automatically:
//   http       python -m rescue serve: plain fetch() calls to /api/... on the same server
//   streamlit  the page runs as a Streamlit component (streamlit_app.py): each call is sent to
//              Python as the component's value, Streamlit reruns the script, and the answer comes
//              back in the component's arguments. One call at a time, in order.
const SID = Math.random().toString(36).slice(2, 12);
const STREAMLIT = new URLSearchParams(location.search).has("streamlitUrl");
// fixed cost of one request: a Streamlit request reruns the script (~0.4 s), so ask for more steps at once there
export const REQUEST_MS = STREAMLIT ? 400 : 0;

// ------------------------------------------------------------------ http
async function httpGet(path) {
  const r = await fetch("api/" + path + (path.includes("?") ? "&" : "?") + "sid=" + SID);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

// ------------------------------------------------------------------ streamlit (component protocol, no npm package needed)
let nextId = 1, pending = null, queue = Promise.resolve();
const toStreamlit = (type, data = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");

if (STREAMLIT) {
  window.addEventListener("message", e => {
    const d = e.data;
    if (!d || d.type !== "streamlit:render") return;
    const res = d.args && d.args.response;
    if (pending && res && res.id === pending.id) {
      const p = pending; pending = null;
      res.ok ? p.resolve(res.result) : p.reject(new Error(res.error));
    }
  });
  toStreamlit("streamlit:componentReady", { apiVersion: 1 });
  // fill the browser window: the component's frame is as tall as the page around it allows
  const fit = () => { let h = 860; try { h = Math.max(640, window.parent.innerHeight - 8); } catch { /* other origin */ }
                      toStreamlit("streamlit:setFrameHeight", { height: h }); };
  fit(); window.addEventListener("resize", fit);
  try { window.parent.addEventListener("resize", fit); } catch { /* other origin */ }
}

function rpc(method, params = {}) {
  const run = () => new Promise((resolve, reject) => {
    pending = { id: nextId++, resolve, reject };
    toStreamlit("streamlit:setComponentValue", { value: { id: pending.id, method, params }, dataType: "json" });
  });
  const p = queue.then(run, run);
  queue = p.catch(() => {});
  return p;
}

// ------------------------------------------------------------------ one API, whichever way
const qs = o => new URLSearchParams(o).toString();
export const api = STREAMLIT ? {
  reset: params => rpc("reset", params),
  step: (n, plan) => rpc("step", { n, plan }),
  compare: params => rpc("compare", params),
  compareStatus: job => rpc("compare_status", { job }),
  reveal: () => rpc("reveal"),
  state: plan => rpc("state", { plan }),
  cameraImage: (robot, kind) => rpc("camera", { robot, kind }),          // -> data: URL
  async downloadVictimMap(name) { save(name, JSON.stringify(await rpc("victim_map"), null, 2)); },
} : {
  reset: params => httpGet("reset?" + qs(params)),                       // -> { world, state }
  // plan: the robot whose planner search (cells looked at, RRT* tree, ant trails) should come back too
  step: (n, plan) => httpGet("step?n=" + n + (plan == null ? "" : "&plan=" + plan)),   // -> state
  compare: params => httpGet("compare?" + qs(params)),                   // -> { job }
  compareStatus: job => httpGet("compare?job=" + job),                   // -> rows so far
  reveal: () => httpGet("reveal"),                                       // -> state
  state: plan => httpGet("state" + (plan == null ? "" : "?plan=" + plan)),   // -> state, without a step
  // kind: "colour" | "thermal"
  cameraImage: async (robot, kind, t) => `api/camera?robot=${robot}&kind=${kind}&sid=${SID}&t=${t}-${Math.random()}`,
  async downloadVictimMap(name) {
    const r = await fetch("api/victim_map?download=1&sid=" + SID);
    save(name, await r.blob());
  },
};

function save(name, data) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(data instanceof Blob ? data : new Blob([data], { type: "application/json" }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
