// The 3-D stage: renderer, lights, camera, and the three ways of looking at the building.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { store } from "./store.js";

export let renderer, scene, camera, controls, sun, sky;

const frameHooks = [];              // called every frame with the elapsed time in seconds
let followTarget = () => null;      // returns the 3-D object of the robot to follow
let focus = null, snap = false;
const clock = new THREE.Clock();

// lighting per picture: daylight with shadows for the real scene, flat light for sensor pictures
const LOOKS = {
  combined: { background: 0x0b1118, sky: 1.45, sun: 2.4, shadows: true },
  lidar:    { background: 0x04070c, sky: 1.7, sun: 1.0, shadows: false },
  thermal:  { background: 0x07040d, sky: 1.7, sun: 1.0, shadows: false },
};

export function initScene(container) {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.domElement.className = "scene";
  container.prepend(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(LOOKS.combined.background);
  scene.fog = new THREE.Fog(LOOKS.combined.background, 50, 120);
  camera = new THREE.PerspectiveCamera(46, 1, 0.1, 400);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.09; controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 1.2; controls.maxDistance = 160; controls.zoomSpeed = 1.1;
  controls.screenSpacePanning = false;                         // right-drag slides along the floor, not up into the sky
  controls.addEventListener("start", () => { focus = null; if (store.viewMode !== "follow") leavePresets(); });

  sky = new THREE.HemisphereLight(0xe6eef8, 0x3a3e44, LOOKS.combined.sky);
  sun = new THREE.DirectionalLight(0xfff0d8, LOOKS.combined.sun);
  sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0005;
  scene.add(sky, sun, sun.target);

  const resize = () => {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);
  resize();
  animate();
}

export const onFrame = fn => frameHooks.push(fn);
export const setFollowTarget = fn => { followTarget = fn; };
export const snapFollow = () => { snap = true; };

// light and background for the chosen picture (combined | lidar | thermal)
export function setLook(mode) {
  const l = LOOKS[mode] || LOOKS.combined;
  scene.background.set(l.background); scene.fog.color.set(l.background);
  sky.intensity = l.sky; sun.intensity = l.sun; sun.castShadow = l.shadows;
}

// place the sun so that the whole building casts shadows
export function lightBuilding(Wm, Hm) {
  sun.position.set(Wm * 0.3, 30, -Hm * 0.2); sun.target.position.set(Wm / 2, 0, Hm / 2);
  Object.assign(sun.shadow.camera, { left: -Wm * 0.75, right: Wm * 0.75, top: Hm * 0.9, bottom: -Hm * 0.9, near: 1, far: 90 });
  sun.shadow.camera.updateProjectionMatrix();
}

export function flyTo(pos, target, ms = 900) {
  focus = { fromPos: camera.position.clone(), fromTarget: controls.target.clone(), pos, target, t0: performance.now(), ms };
}

// the user moved the camera: none of the preset buttons describes the view any more
function leavePresets() {
  if (store.viewMode === "free") return;
  store.viewMode = "free";
  document.querySelectorAll("#viewSeg button").forEach(b => b.classList.remove("on"));
}

// where the camera will end up (the end of a running flight, or where it is now)
const goal = () => focus ? { pos: focus.pos.clone(), target: focus.target.clone() } : { pos: camera.position.clone(), target: controls.target.clone() };

// zoom in (k < 1) or out (k > 1) towards the point the camera orbits
export function zoomBy(k) {
  if (store.viewMode === "follow") setView("overview");
  leavePresets();
  const g = goal(), off = g.pos.clone().sub(g.target);
  const d = THREE.MathUtils.clamp(off.length() * k, controls.minDistance, controls.maxDistance);
  flyTo(g.target.clone().add(off.setLength(d)), g.target, 350);
}

// make a point the centre of rotation and zoom, and move closer to it (keeps the viewing direction)
export function focusOn(point, dist = 7) {
  if (store.viewMode === "follow") controls.enabled = true;
  leavePresets();
  const g = goal(), dir = g.pos.clone().sub(g.target);
  if (dir.y < 0.5) dir.y = 0.5;
  const d = Math.min(dir.length(), dist);
  flyTo(point.clone().add(dir.setLength(d)), point.clone(), 700);
}

export function setView(mode) {
  store.viewMode = mode;
  document.querySelectorAll("#viewSeg button").forEach(b => b.classList.toggle("on", b.dataset.view === mode));
  const world = store.world;
  if (!world) return;
  const Wm = world.W * world.cell, Hm = world.H * world.cell;
  controls.enabled = mode !== "follow"; snap = mode === "follow";
  // move the camera back until the whole building (and the base in front of it) fits the width of the view
  const tanH = Math.tan(camera.fov * Math.PI / 360) * camera.aspect, tanV = Math.tan(camera.fov * Math.PI / 360);
  if (mode === "overview") {
    const target = new THREE.Vector3(Wm * 0.5 - 1, 0, Hm * 0.52), dir = new THREE.Vector3(0, Wm * 0.62, Hm * 0.9);
    const k = Math.max(1, (Wm * 0.5 + 3.5) / tanH / dir.length() * 1.08);
    flyTo(target.clone().addScaledVector(dir, k), target);
  }
  if (mode === "top") {
    const height = Math.max((Wm * 0.5 + 3) / tanH, (Hm * 0.5 + 1.5) / tanV);
    flyTo(new THREE.Vector3(Wm * 0.5 - 1, height, Hm * 0.5 + 0.01), new THREE.Vector3(Wm * 0.5 - 1, 0, Hm * 0.5));
  }
}

// the view changed size (a side panel was opened or closed): frame the building again
export function refit() { if (store.viewMode === "overview" || store.viewMode === "top") setView(store.viewMode); }

// leave the preset views and look at one spot (used when a victim is clicked)
export function lookAt(x, y) {
  leavePresets(); controls.enabled = true;
  flyTo(new THREE.Vector3(x + 4, 6.5, y + 6), new THREE.Vector3(x, 0.3, y));
}

function animate() {
  requestAnimationFrame(animate);
  const time = clock.getElapsedTime();
  for (const fn of frameHooks) fn(time);
  const R = store.viewMode === "follow" ? followTarget() : null;
  if (R) {
    const hd = -R.rotation.y;
    const target = new THREE.Vector3(R.position.x + Math.cos(hd), 0, R.position.z + Math.sin(hd));
    const eye = new THREE.Vector3(R.position.x - Math.cos(hd) * 2.6, 9, R.position.z - Math.sin(hd) * 2.6);
    camera.position.lerp(eye, snap ? 1 : 0.06); controls.target.lerp(target, snap ? 1 : 0.1); snap = false;
  } else if (focus) {
    const k = Math.min(1, (performance.now() - focus.t0) / focus.ms), e = k * k * (3 - 2 * k);
    camera.position.lerpVectors(focus.fromPos, focus.pos, e); controls.target.lerpVectors(focus.fromTarget, focus.target, e);
    if (k >= 1) focus = null;
  }
  controls.update();
  renderer.render(scene, camera);
}
