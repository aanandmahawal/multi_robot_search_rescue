// The one place that holds what the page currently knows. Modules read and write it directly.
import { DEFAULT_SPEED, LAYERS } from "./config.js";

export const store = {
  world: null,          // the building, as sent by /api/reset (does not change during a mission)
  state: null,          // the latest mission state from /api/step
  prevState: null,      // the state before it (robots glide from one to the other)
  lastAt: 0,            // when `state` arrived (ms)
  frameDur: 150,        // how long the glide between two states takes (ms)

  phase: "loading",     // loading | ready | running | paused | finished
  busy: false,          // a request to the server is under way
  speed: DEFAULT_SPEED, // simulated seconds per real second

  sensors: "both",      // what the robots carry: both | cameras | lidar
  known: "0",           // "1" = the number of victims is known
  vary: "0",            // "1" = every run draws new random numbers (same building, different paths)
  revisit: "0",         // route cost per earlier visit of a cell ("0" = shortest routes)
  deadend: "1",         // dead-end / loop recovery on
  recharge: "1",        // "1" = a robot back on low battery recharges and goes out again
  batteryEach: [],      // per-robot capacities (Wh) when Battery = "Different per robot"
  compareJob: null,     // the running algorithm comparison (Stats tab)
  seed: Math.floor(Math.random() * 100000),
  preset: {},           // extra settings given in the address bar (e.g. vision=ideal)

  selected: 0,          // the robot shown in the Sensors tab and followed by the Follow view
  viewMode: "overview", // camera: overview | top | follow | free
  viewAs: "combined",   // picture: combined | lidar | thermal
  tab: "victims",       // the open tab of the details panel
  layers: Object.fromEntries(LAYERS.map(([k]) => [k, k !== "visits"])),
  legend: false,        // the colour key of the 3-D view (Legend button)
};

export const truthVisible = () => !!(store.world && (store.world.config.victims_known || (store.state && store.state.revealed)));
