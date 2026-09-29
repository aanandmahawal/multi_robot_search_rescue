// The looks that make the LiDAR and Thermal pictures read like the real instruments.
//
//   LiDAR    a 2-D laser scanner only measures one horizontal slice (the scan plane, 40 cm up).
//            Objects the laser hit are drawn as violet "extrusions" of the map with a bright line
//            at the scan plane, the walls it traced are an outline of points (the point cloud a
//            real scanner produces), a grid shows the metric map, and a ring sweeps out from each
//            robot while the laser turns.
//   Thermal  what the thermal cameras measured, smoothed like a real thermal image, with a glow
//            around everything warmer than the air.
// Nothing here adds knowledge: every colour still comes from the team's map and thermal map.
import * as THREE from "three";
import { store } from "./store.js";
import { cellTemp, ironbow } from "./util.js";

const SHARED = { uScanH: { value: 0.4 }, uTime: { value: 0 } };
let grid = null, outline = null, glow = null, ground = null, dotTex = null;

// ------------------------------------------------------------------ LiDAR material
// flat colour (the map state of the cell, from the instance colour) plus: faint horizontal layers,
// and a bright line where the object crosses the scan plane, on objects the laser has hit
export function lidarMaterial() {
  const m = new THREE.MeshBasicMaterial({ toneMapped: false });
  m.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, SHARED);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vLidarW;")
      .replace("#include <project_vertex>", `#include <project_vertex>
        vec4 lidarW = vec4( transformed, 1.0 );
        #ifdef USE_INSTANCING
          lidarW = instanceMatrix * lidarW;
        #endif
        vLidarW = ( modelMatrix * lidarW ).xyz;`);
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vLidarW;\nuniform float uScanH;\nuniform float uTime;")
      .replace("#include <opaque_fragment>", `
        float y = vLidarW.y;
        vec3 nrm = normalize( cross( dFdx( vLidarW ), dFdy( vLidarW ) ) );          // face direction, for a little shape
        float shade = 0.45 + 0.55 * abs( dot( nrm, normalize( vec3( 0.45, 0.8, 0.35 ) ) ) );
        float hit = step( 0.12, max( diffuseColor.r, max( diffuseColor.g, diffuseColor.b ) ) );
        float band = exp( -pow( ( y - uScanH ) / 0.04, 2.0 ) );
        float layers = smoothstep( 0.86, 1.0, fract( y * 6.0 ) ) * 0.35;
        float fade = 1.0 - smoothstep( uScanH, uScanH + 2.4, y ) * 0.5;               // above the plane the laser knows nothing
        outgoingLight = outgoingLight * ( shade + layers ) * fade + hit * band * vec3( 0.85, 0.9, 1.0 ) * ( 1.6 + 0.4 * sin( uTime * 5.0 ) );
        #include <opaque_fragment>`);
  };
  return m;
}

// a soft round dot, used for the point cloud and the thermal glow
function dot() {
  if (dotTex) return dotTex;
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d"), r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.35, "rgba(255,255,255,0.55)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 64, 64);
  dotTex = new THREE.CanvasTexture(c);
  return dotTex;
}

// ------------------------------------------------------------------ build (once per mission)
export function buildSensorFx(group, groundMesh) {
  const { world } = store, Wm = world.W * world.cell, Hm = world.H * world.cell;
  ground = groundMesh;
  SHARED.uScanH.value = world.config.lidar_height;

  const size = Math.ceil(Math.max(Wm, Hm) + 16);                    // 1 m grid of the metric map
  grid = new THREE.GridHelper(size, size, 0x5b4bb5, 0x241d4d);
  grid.position.set(Wm / 2, 0.012, Hm / 2); grid.material.transparent = true; grid.material.opacity = 0.55;
  grid.material.depthWrite = false;
  group.add(grid);

  outline = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
    map: dot(), size: 0.16, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }));
  outline.frustumCulled = false; group.add(outline);

  glow = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
    map: dot(), size: 1.5, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    toneMapped: false, opacity: 0.8 }));
  glow.frustumCulled = false; group.add(glow);
}

// ------------------------------------------------------------------ update (after every step)
export function updateSensorFx(mode) {
  if (!grid) return;
  const { world, state } = store;
  grid.visible = mode === "lidar";
  ground.material.color.set({ combined: 0x161c24, lidar: 0x04060b, thermal: 0x0c0612 }[mode]);
  outline.visible = mode === "lidar" && world.has_lidar;
  glow.visible = mode === "thermal";
  if (outline.visible) fillOutline(world, state);
  if (glow.visible) fillGlow(world, state);
}

// the edges between a cell the laser hit and a free cell next to it: where the beams ended
function fillOutline(world, state) {
  const { W, H, cell } = world, k = state.known, h = world.config.lidar_height, pts = [], cols = [];
  const cA = new THREE.Color(0xd8ccff), cB = new THREE.Color(0x8f6bff), c = new THREE.Color();
  const put = (x, z, t) => { pts.push(x, h, z); c.copy(cA).lerp(cB, t); cols.push(c.r, c.g, c.b); };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (k[y * W + x] !== "2") continue;
    const t = ((x * 7 + y * 13) % 10) / 10;                         // a little variation, like beam intensity
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H || k[ny * W + nx] !== "1") continue;
      for (let s = 0; s < 4; s++) {
        const a = (s + 0.5) / 4;
        const px = dx ? (dx > 0 ? x + 1 : x) : x + a, pz = dy ? (dy > 0 ? y + 1 : y) : y + a;
        put(px * cell, pz * cell, t);
      }
    }
  }
  setPoints(outline, pts, cols);
}

// a glow over every cell a thermal camera measured as warmer than the air
function fillGlow(world, state) {
  const { W, H, cell } = world, [lo, hi] = world.thermal_scale, pts = [], cols = [];
  for (let i = 0; i < W * H && pts.length < 3 * 2500; i++) {
    const T = cellTemp(state.thermal_map, i);
    if (T === null || T - world.ambient < 2.5) continue;
    const [r, g, b] = ironbow((T - lo) / (hi - lo)), s = Math.min(1, (T - world.ambient) / 12);
    pts.push((i % W + 0.5) * cell, 0.35, (Math.floor(i / W) + 0.5) * cell);
    cols.push(r / 255 * s, g / 255 * s, b / 255 * s);
  }
  setPoints(glow, pts, cols);
}

function setPoints(obj, pts, cols) {
  obj.geometry.dispose();
  obj.geometry = new THREE.BufferGeometry();
  obj.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  obj.geometry.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
}

// ------------------------------------------------------------------ every frame
export function animateSensorFx(time) { SHARED.uTime.value = time; }

// a ring that sweeps out from a robot to the laser's range while the laser turns
export function makeSweep(cfg) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.94, 1, 64), new THREE.MeshBasicMaterial({
    color: 0xc4b5fd, transparent: true, opacity: 0.5, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    blending: THREE.AdditiveBlending }));
  ring.rotation.x = -Math.PI / 2; ring.position.y = cfg.lidar_height; ring.userData.range = cfg.lidar_range;
  return ring;
}

export function animateSweep(ring, time, phase) {
  const f = (time * 0.9 + phase) % 1;
  ring.scale.setScalar(0.2 + f * ring.userData.range);
  ring.material.opacity = 0.55 * (1 - f) * (1 - f);
}
