// The building itself: walls, racks, rubble, furniture, look-alikes and warm objects.
//
// Furniture, racks and rubble are made of many small boxes and cylinders. They are collected
// into lists and drawn as a few instanced meshes (fast). Every instance remembers the map cell
// it stands in, so parts of the building nobody has seen yet can be drawn darker.
import * as THREE from "three";
import { concrete, mulberry32, rgb } from "./util.js";

const BOX = new THREE.BoxGeometry(1, 1, 1); BOX.translate(0, 0.5, 0);
const CYL = new THREE.CylinderGeometry(1, 1, 1, 16); CYL.translate(0, 0.5, 0);
const ROCK = new THREE.DodecahedronGeometry(1, 0);
const TEX = { wall: concrete("#bdb7ab"), pillar: concrete("#8f9094") };
const WALL_TINT = { office: 0xe6e9ee, apartments: 0xf5ead8, hospital: 0xe9f4ef, school: 0xf7ecd2, parking: 0xb9bcc0, warehouse: 0xffffff, plain: 0xe8e8e4 };

export function createBuilder(world, group) {
  const shaded = [];                                   // { mesh, colors, nav, map } per instanced mesh
  const singles = [];                                  // look-alikes and warm objects: { meshes, flat, x, y }
  const batch = { box: [], cyl: [], rock: [] };
  // box with its bottom at y, centred at (x, z), size (sx, sy, sz), optional rotation about y
  const pushBox = (x, y, z, sx, sy, sz, color, nav, rotY = 0) => batch.box.push([x, y, z, sx, sy, sz, color, nav, rotY]);
  const pushCyl = (x, y, z, r, h, color, nav, axis = "y") => batch.cyl.push([x, y, z, r, h, color, nav, axis]);

  function instanced(geo, mat, items, place) {
    if (!items.length) return;
    const mesh = new THREE.InstancedMesh(geo, mat, items.length);
    mesh.castShadow = true; mesh.receiveShadow = true;
    const m = new THREE.Matrix4(), col = new THREE.Color(), colors = [], nav = [];
    items.forEach((it, i) => { const r = place(it, m); mesh.setMatrixAt(i, m); col.set(r.color); mesh.setColorAt(i, col); colors.push(col.clone()); nav.push(r.nav); });
    group.add(mesh); shaded.push({ mesh, colors, nav, map: mat.map || null });
  }

  function flush() {
    const q = new THREE.Quaternion(), e = new THREE.Euler(), pos = new THREE.Vector3(), scl = new THREE.Vector3();
    instanced(BOX, new THREE.MeshStandardMaterial({ roughness: 0.75 }), batch.box, (b, m) => {
      q.setFromEuler(e.set(0, b[8], 0)); m.compose(pos.set(b[0], b[1], b[2]), q, scl.set(b[3], b[4], b[5]));
      return { color: b[6], nav: b[7] };
    });
    instanced(CYL, new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.3 }), batch.cyl, (c, m) => {
      if (c[7] === "x") q.setFromEuler(e.set(0, 0, Math.PI / 2)); else if (c[7] === "z") q.setFromEuler(e.set(Math.PI / 2, 0, 0)); else q.identity();
      const p = c[7] === "x" ? pos.set(c[0] + c[4] / 2, c[1], c[2]) : c[7] === "z" ? pos.set(c[0], c[1], c[2] - c[4] / 2) : pos.set(c[0], c[1], c[2]);
      m.compose(p, q, scl.set(c[3], c[4], c[3]));
      return { color: c[5], nav: c[6] };
    });
    instanced(ROCK, new THREE.MeshStandardMaterial({ roughness: 1, flatShading: true }), batch.rock, (k, m) => {
      q.setFromEuler(e.set(k[4] * 6, k[4] * 11, k[4] * 3));
      m.compose(pos.set(k[0], k[1], k[2]), q, scl.set(k[3] * 1.2, k[3] * 0.8, k[3]));
      return { color: k[5], nav: k[6] };
    });
  }

  // ---------------------------------------------------------------- walls, pillars, racks
  function structure(rand) {
    const c = world.cell, W = world.W, tint = WALL_TINT[world.config.building] || 0xffffff;
    const walls = world.structure.filter(s => s[2] === 1), pillars = world.structure.filter(s => s[2] === 2),
          racks = world.structure.filter(s => s[2] === 3);
    const q = new THREE.Quaternion(), pos = new THREE.Vector3(), scl = new THREE.Vector3();
    const textured = (items, tex, scale, color) => instanced(BOX, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }), items, (s, m) => {
      m.compose(pos.set((s[0] + 0.5) * c, 0, (s[1] + 0.5) * c), q, scl.set(c * scale, s[3], c * scale));
      return { color, nav: s[1] * W + s[0] };
    });
    textured(walls, TEX.wall, 1.0, tint);
    textured(pillars, TEX.pillar, 0.9, 0xffffff);
    for (const s of pillars) pushBox((s[0] + 0.5) * c, 0, (s[1] + 0.5) * c, c * 0.94, 0.35, c * 0.94, 0xf5c518, s[1] * W + s[0]);
    for (const s of racks) {                          // pallet racking: orange uprights, blue beams, cartons
      const x0 = s[0] * c, z = (s[1] + 0.5) * c, nav = s[1] * W + s[0];
      for (const dz of [-0.2, 0.2]) pushBox(x0 + 0.03, 0, z + dz, 0.05, 2.0, 0.05, 0xe07b22, nav);
      for (const h of [0.08, 0.7, 1.3, 1.9]) pushBox(x0 + c / 2, h, z, c, 0.04, 0.46, 0x3f6aa8, nav);
      for (const h of [0.12, 0.74, 1.34]) if (rand() < 0.75)
        pushBox(x0 + c / 2 + (rand() - 0.5) * 0.08, h, z, 0.38, 0.25 + rand() * 0.2, 0.34, rand() < 0.8 ? 0xb88a55 : 0xd9d4c7, nav);
    }
  }

  function rubble(rand) {
    const c = world.cell, W = world.W, tones = [0x8d8a84, 0x7a746c, 0x9c9589, 0x6f6b66, 0x8a7f70];
    for (const [x, y, h] of world.rubble) {
      const nav = y * W + x;
      for (let i = 0, n = 4 + Math.floor(h * 5); i < n; i++) {
        const r = 0.08 + rand() * 0.16;
        batch.rock.push([(x + 0.15 + 0.7 * rand()) * c, Math.max(r * 0.7, Math.min(h - r * 0.5, rand() * h * 0.9)), (y + 0.15 + 0.7 * rand()) * c,
                         r, rand(), tones[Math.floor(rand() * tones.length)], nav]);
      }
      pushBox((x + 0.5) * c, h * 0.5, (y + 0.5) * c, 0.42, 0.07, 0.3, 0xa7a197, nav, rand() * 6);
    }
    const f = world.fine;
    for (const d of world.debris) pushBox((d[0] + 0.5) * f, 0, (d[1] + 0.5) * f, 0.08 + rand() * 0.1, 0.03, 0.05 + rand() * 0.08,
                                          new THREE.Color(d[2] / 255, d[3] / 255, d[4] / 255).getHex(), (d[1] >> 1) * W + (d[0] >> 1), rand() * 6);
  }

  // ---------------------------------------------------------------- furniture
  function furniture(it, rand) {
    const { x, y: z, sx, sy } = it, nav = it.nav[1] * world.W + it.nav[0];
    const col = new THREE.Color(it.color[0] / 255, it.color[1] / 255, it.color[2] / 255).getHex();
    const dir = { n: [0, -1], s: [0, 1], e: [1, 0], w: [-1, 0] }[it.facing] || [0, -1];     // where the front points
    const legs = (h, w, d, inset, c2) => { for (const [a, b] of [[-1, -1], [1, -1], [-1, 1], [1, 1]])
      pushBox(x + a * (w / 2 - inset), 0, z + b * (d / 2 - inset), 0.05, h, 0.05, c2, nav); };
    const chair = (cx, cz, face, c2 = 0x2b2f36) => {
      pushBox(cx, 0, cz, 0.05, 0.42, 0.05, 0x555a61, nav);
      pushBox(cx, 0.42, cz, 0.42, 0.06, 0.42, c2, nav);
      pushBox(cx - face[0] * 0.2, 0.48, cz - face[1] * 0.2, face[0] ? 0.05 : 0.42, 0.42, face[0] ? 0.42 : 0.05, c2, nav);
    };
    switch (it.kind) {
      case "desk": case "table": case "school_desk": {
        const h = it.kind === "school_desk" ? 0.68 : 0.72, top = it.kind === "table" ? 0x6e4b31 : it.kind === "school_desk" ? 0xc9a36b : 0xd8c29a;
        pushBox(x, h, z, sx * 0.94, 0.04, sy * 0.9, top, nav);
        legs(h, sx * 0.94, sy * 0.9, 0.05, 0x3c3f45);
        if (it.kind === "desk") {                     // monitor, keyboard and chair
          const back = [-dir[0], -dir[1]];
          pushBox(x + back[0] * sy * 0.25, h + 0.04, z + back[1] * sy * 0.25, 0.1, 0.05, 0.1, 0x222222, nav);
          pushBox(x + back[0] * sy * 0.25, h + 0.09, z + back[1] * sy * 0.25, dir[0] ? 0.03 : 0.55, 0.33, dir[0] ? 0.55 : 0.03, 0x15181c, nav);
          pushBox(x, h + 0.04, z + dir[1] * 0.08, 0.4, 0.02, 0.14, 0x2a2d33, nav);
          chair(x + dir[0] * sy * 0.35, z + dir[1] * sy * 0.35, dir);
        } else if (it.kind === "school_desk") {
          chair(x + dir[0] * 0.45 * -1, z - dir[1] * 0.45, [-dir[0], -dir[1]], 0x3a6fb0);
        } else {                                      // chairs tucked along the long sides
          const alongX = sx >= sy, n = Math.max(1, Math.floor((alongX ? sx : sy) / 0.8));
          for (let i = 0; i < n; i++) {
            const t = -((alongX ? sx : sy) / 2) + (i + 0.5) * (alongX ? sx : sy) / n;
            for (const s of [-1, 1]) {
              if (alongX) chair(x + t, z + s * (sy / 2 - 0.1), [0, -s]); else chair(x + s * (sx / 2 - 0.1), z + t, [-s, 0]);
            }
          }
        }
        break;
      }
      case "bed": case "hospital_bed": {
        const hosp = it.kind === "hospital_bed";
        const head = [x + dir[0] * (Math.max(sx, sy) / 2 - 0.15), z + dir[1] * (Math.max(sx, sy) / 2 - 0.15)];
        pushBox(x, 0, z, sx * 0.95, hosp ? 0.45 : 0.3, sy * 0.95, hosp ? 0x9aa3ab : 0x6b4a33, nav);
        pushBox(x, hosp ? 0.45 : 0.3, z, sx * 0.9, 0.16, sy * 0.92, 0xf2f2f0, nav);
        const bx = x - dir[0] * sy * 0.12, bz = z - dir[1] * sy * 0.12;            // blanket over the foot end
        pushBox(bx, hosp ? 0.6 : 0.45, bz, dir[0] ? sx * 0.6 : sx * 0.92, 0.05, dir[0] ? sy * 0.92 : sy * 0.62,
                hosp ? 0x7cc6b3 : [0x4a6fa5, 0xa54a4a, 0x5b8c5a, 0x8b6fb0][Math.floor(rand() * 4)], nav);
        pushBox(head[0] - dir[0] * 0.15, hosp ? 0.6 : 0.45, head[1] - dir[1] * 0.15, dir[0] ? 0.25 : sx * 0.6, 0.1, dir[0] ? sx * 0.6 : 0.25, 0xffffff, nav);
        pushBox(head[0] + dir[0] * 0.1, 0, head[1] + dir[1] * 0.1, dir[0] ? 0.06 : sx * 0.95, hosp ? 1.0 : 0.9, dir[0] ? sx * 0.95 : 0.06, hosp ? 0xc8ced4 : 0x5a3e2b, nav);
        if (hosp) {
          for (const s of [-1, 1]) pushBox(x + (dir[0] ? 0 : s * sx * 0.47), 0.62, z + (dir[0] ? s * sy * 0.47 : 0), dir[0] ? sy * 0.5 : 0.03, 0.18, dir[0] ? 0.03 : sy * 0.5, 0xc8ced4, nav);
          pushCyl(head[0] + (dir[0] ? 0 : sx * 0.6), 0, head[1] + (dir[0] ? sy * 0.6 : 0), 0.02, 1.8, 0xb0b8c0, nav);
          pushBox(head[0] + (dir[0] ? 0 : sx * 0.6), 1.6, head[1] + (dir[0] ? sy * 0.6 : 0), 0.12, 0.2, 0.06, 0xdff2ff, nav);
        }
        break;
      }
      case "sofa": {
        pushBox(x, 0, z, sx * 0.95, 0.42, sy * 0.95, col, nav);
        pushBox(x - dir[0] * sx * 0.38, 0.42, z - dir[1] * sy * 0.38, dir[0] ? sx * 0.22 : sx * 0.95, 0.4, dir[0] ? sy * 0.95 : sy * 0.22, col, nav);
        for (const s of [-1, 1]) pushBox(x + (dir[0] ? 0 : s * sx * 0.44), 0.42, z + (dir[0] ? s * sy * 0.44 : 0), dir[0] ? sx * 0.95 : sx * 0.1, 0.18, dir[0] ? sy * 0.1 : sy * 0.95, col, nav);
        for (const s of [-0.25, 0.25]) pushBox(x + (dir[0] ? 0.05 : s * sx), 0.42, z + (dir[0] ? s * sy : 0.05), dir[0] ? sx * 0.5 : sx * 0.42, 0.12, dir[0] ? sy * 0.42 : sy * 0.5, new THREE.Color(col).offsetHSL(0, 0, 0.08).getHex(), nav);
        break;
      }
      case "counter": {
        pushBox(x, 0, z, sx * 0.95, 0.88, sy * 0.95, 0xe9e6df, nav);
        pushBox(x, 0.88, z, sx, 0.05, sy, 0x2e3136, nav);
        pushBox(x, 0.93, z + (sy > sx ? -sy * 0.2 : 0), sx > sy ? 0.5 : sx * 0.55, 0.02, sx > sy ? sy * 0.55 : 0.4, 0x9aa4ae, nav);
        break;
      }
      case "shelf": {
        pushBox(x, 0, z, sx * 0.95, 1.8, sy * 0.95, 0x7a5535, nav);
        const alongX = sx >= sy;
        for (let k = 0; k < 5; k++) {
          const h = 0.1 + k * 0.36, len = alongX ? sx * 0.85 : sy * 0.85;
          for (let t = -len / 2; t < len / 2 - 0.05;) {
            const w = 0.04 + rand() * 0.05, bh = 0.22 + rand() * 0.08;
            const c2 = [0xc0392b, 0x2e86c1, 0x27ae60, 0xf1c40f, 0x8e44ad, 0xecf0f1, 0xd35400][Math.floor(rand() * 7)];
            if (alongX) pushBox(x + t + w / 2, h, z + (sy / 2) * 0.55, w, bh, 0.16, c2, nav);
            else pushBox(x + (sx / 2) * 0.55, h, z + t + w / 2, 0.16, bh, w, c2, nav);
            t += w + 0.01;
          }
        }
        break;
      }
      case "crate": {                                  // low wooden crate: under the LiDAR's scan plane
        pushBox(x, 0, z, sx * 0.9, it.height, sy * 0.9, 0xb08c5c, nav);
        for (const t of [-0.3, 0, 0.3]) pushBox(x, it.height - 0.005, z + t * sy * 0.9, sx * 0.92, 0.012, 0.03, 0x7a5c36, nav);
        break;
      }
      case "cabinet": {
        pushBox(x, 0, z, sx * 0.92, it.height, sy * 0.92, 0x9ea5ad, nav);
        const alongX = sx >= sy, n = Math.max(2, Math.round((alongX ? sx : sy) / 0.4));
        for (let k = 1; k < n; k++) {
          const t = -((alongX ? sx : sy) / 2) + k * (alongX ? sx : sy) / n;
          pushBox(x + (alongX ? t : 0), 0.05, z + (alongX ? 0 : t), alongX ? 0.02 : sx * 0.95, it.height - 0.1, alongX ? sy * 0.95 : 0.02, 0x5f666e, nav);
        }
        break;
      }
      case "car": {
        const len = Math.max(sx, sy), wid = Math.min(sx, sy), along = sy >= sx ? "z" : "x", f = dir;   // bonnet points towards the lane
        pushBox(x, 0.22, z, along === "z" ? wid * 0.92 : len * 0.95, 0.62, along === "z" ? len * 0.95 : wid * 0.92, col, nav);
        const cabX = x - f[0] * len * 0.08, cabZ = z - f[1] * len * 0.08;
        pushBox(cabX, 0.84, cabZ, along === "z" ? wid * 0.82 : len * 0.5, 0.5, along === "z" ? len * 0.5 : wid * 0.82, 0x26303b, nav);
        pushBox(cabX, 1.33, cabZ, along === "z" ? wid * 0.78 : len * 0.44, 0.06, along === "z" ? len * 0.44 : wid * 0.78, col, nav);
        for (const a of [-0.32, 0.32]) for (const s of [-1, 1]) {
          const wx = along === "z" ? x + s * wid * 0.4 : x + a * len, wz = along === "z" ? z + a * len : z + s * wid * 0.4;
          pushCyl(wx - (along === "z" ? 0.11 : 0), 0.33, wz + (along === "z" ? 0 : 0.11), 0.33, 0.22, 0x151515, nav, along === "z" ? "x" : "z");
        }
        for (const s of [-1, 1]) {                                // headlights and tail lights
          const px = along === "z" ? x + s * wid * 0.3 : x, pz = along === "z" ? z : z + s * wid * 0.3;
          pushBox(px + f[0] * len * 0.47, 0.55, pz + f[1] * len * 0.47, 0.25, 0.1, 0.05, 0xfff1b8, nav);
          pushBox(px - f[0] * len * 0.47, 0.55, pz - f[1] * len * 0.47, 0.25, 0.1, 0.05, 0xc0141a, nav);
        }
        break;
      }
      case "bleachers": {
        for (let k = 0; k < 3; k++) {                  // three rows, rising towards the wall they face
          const depth = sy / 3, zz = z + dir[1] * (-(sy / 2) + depth * (k + 0.5));
          pushBox(x, 0, zz, sx * 0.98, 0.4 * (k + 1), depth, [0x3c5a96, 0x4a6aa8, 0x5877b8][k], nav);
          for (let s = -sx / 2 + 0.3; s < sx / 2; s += 0.6) pushBox(x + s, 0.4 * (k + 1), zz, 0.45, 0.04, depth * 0.6, 0xe8e2d0, nav);
        }
        break;
      }
    }
  }

  // ---------------------------------------------------------------- single objects
  function single(d, roughness, build) {
    const g = new THREE.Group();
    const [sx, sy] = d.size, L = Math.max(sx, sy), Wd = Math.min(sx, sy);
    const mat = (c, extra = {}) => new THREE.MeshStandardMaterial({ color: c, roughness, ...extra });
    const add = (geo, m, x, y, z) => { const mesh = new THREE.Mesh(geo, m); mesh.position.set(x, y, z); mesh.castShadow = true; g.add(mesh); return mesh; };
    build({ add, mat, L, Wd, col: rgb(d.color) });
    g.position.set(d.x, 0, d.y); g.rotation.y = sx >= sy ? 0 : Math.PI / 2;
    group.add(g);
    // for the LiDAR and thermal pictures the object is repainted in one flat colour
    const meshes = g.children.filter(o => o.isMesh);
    for (const o of meshes) o.userData.base = o.material;
    singles.push({ meshes, flat: new THREE.MeshBasicMaterial({ toneMapped: false }), x: d.x, y: d.y });
  }

  // things a colour camera could mistake for a person
  const decoy = d => single(d, 0.85, ({ add, mat, L, Wd, col }) => {
    if (d.kind === "jacket") {
      add(new THREE.BoxGeometry(L * 0.7, 0.05, Wd * 0.9), mat(col), 0, 0.03, 0);
      for (const s of [-1, 1]) add(new THREE.BoxGeometry(L * 0.18, 0.04, Wd * 0.8), mat(col), s * L * 0.42, 0.025, 0.08 * s).rotation.y = 0.5 * s;
    } else if (d.kind === "bag") {
      add(new THREE.CapsuleGeometry(Wd * 0.3, L * 0.3, 4, 12), mat(col), 0, Wd * 0.3, 0).rotation.z = Math.PI / 2;
      add(new THREE.TorusGeometry(0.09, 0.015, 6, 16, Math.PI), mat(0x222222), 0, Wd * 0.58, 0);
    } else if (d.kind === "cardboard box") {
      add(new THREE.BoxGeometry(L * 0.9, d.height, Wd * 0.9), mat(0xb88a55), 0, d.height / 2, 0);
      add(new THREE.BoxGeometry(L * 0.92, 0.01, 0.08), mat(0xd8c7a0), 0, d.height + 0.005, 0);
    } else if (d.kind === "wooden beam") {
      add(new THREE.BoxGeometry(L * 0.95, 0.18, 0.22), mat(0x7a5230), 0, 0.09, 0).rotation.y = 0.05;
    } else {                                                         // barrel
      const r = Wd * 0.45;
      add(new THREE.CylinderGeometry(r, r, d.height, 20), mat(col, { roughness: 0.5, metalness: 0.2 }), 0, d.height / 2, 0);
      for (const h of [0.15, 0.4]) add(new THREE.TorusGeometry(r, 0.012, 6, 20), mat(0x333333), 0, h, 0).rotation.x = Math.PI / 2;
    }
  });

  // things a thermal camera could mistake for a person
  const warmObject = d => single(d, 0.8, ({ add, mat, L, Wd, col }) => {
    if (d.kind === "space heater") {
      add(new THREE.BoxGeometry(L * 0.9, 0.5, Wd * 0.7), mat(0xd8d8dc, { metalness: 0.3, roughness: 0.4 }), 0, 0.28, 0);
      for (let i = -3; i <= 3; i++) add(new THREE.BoxGeometry(0.02, 0.42, Wd * 0.72), mat(0x9a9aa0), i * L * 0.11, 0.28, 0);
      add(new THREE.SphereGeometry(0.02, 8, 6), new THREE.MeshStandardMaterial({ color: 0xff3020, emissive: 0xff3020, emissiveIntensity: 2 }), L * 0.38, 0.5, Wd * 0.36);
      for (const s of [-1, 1]) add(new THREE.BoxGeometry(0.06, 0.04, Wd * 0.9), mat(0x333333), s * L * 0.35, 0.02, 0);
    } else if (d.kind === "laptop left running") {
      add(new THREE.BoxGeometry(L * 0.85, 0.03, Wd * 0.9), mat(0x2b2d33, { metalness: 0.4, roughness: 0.5 }), 0, 0.015, 0);
      add(new THREE.BoxGeometry(L * 0.85, Wd * 0.85, 0.015), mat(0x2b2d33), 0, 0.03 + Wd * 0.4, -Wd * 0.42).rotation.x = -0.35;
      add(new THREE.PlaneGeometry(L * 0.75, Wd * 0.7), new THREE.MeshBasicMaterial({ color: 0x8fd0ff }), 0, 0.035 + Wd * 0.4, -Wd * 0.42 + 0.012).rotation.x = -0.35;
    } else if (d.kind === "hot-water leak") {
      const p = add(new THREE.CircleGeometry(0.5, 24), new THREE.MeshStandardMaterial({ color: 0x3a4a66, roughness: 0.15, metalness: 0.6, transparent: true, opacity: 0.85 }), 0, 0.006, 0);
      p.rotation.x = -Math.PI / 2; p.scale.set(L * 0.95, Wd * 0.9, 1); p.castShadow = false;
      add(new THREE.CylinderGeometry(0.04, 0.04, 0.5, 10), mat(0xb87333, { metalness: 0.7, roughness: 0.3 }), -L * 0.3, 0.05, 0).rotation.z = Math.PI / 2;
    } else if (d.kind === "dog") {
      const fur = mat(col);
      add(new THREE.CapsuleGeometry(0.13, 0.36, 4, 10), fur, 0, 0.15, 0).rotation.z = Math.PI / 2;
      add(new THREE.SphereGeometry(0.1, 12, 10), fur, 0.36, 0.2, 0);
      add(new THREE.BoxGeometry(0.12, 0.06, 0.07), fur, 0.46, 0.17, 0);
      for (const s of [-1, 1]) add(new THREE.BoxGeometry(0.06, 0.09, 0.03), fur, 0.34, 0.29, s * 0.07).rotation.z = 0.4;
      for (const [dx, dz] of [[0.18, 0.1], [0.18, -0.1], [-0.18, 0.1], [-0.18, -0.1]]) add(new THREE.CapsuleGeometry(0.035, 0.18, 3, 6), fur, dx, 0.05, dz).rotation.z = Math.PI / 2;
      add(new THREE.CapsuleGeometry(0.025, 0.25, 3, 6), fur, -0.4, 0.14, 0.05).rotation.z = Math.PI / 2 + 0.5;
    } else {                                                          // sun-warmed rubble: looks like any other pile
      const rnd = mulberry32(Math.round(d.x * 100 + d.y * 7)), tones = [0x8d8a84, 0x7a746c, 0x9c9589, 0x8a7f70];
      for (let i = 0; i < 12; i++) {
        const r = 0.07 + rnd() * 0.13;
        const m = add(new THREE.DodecahedronGeometry(r, 0), new THREE.MeshStandardMaterial({ color: tones[i % 4], roughness: 1, flatShading: true }),
                      (rnd() - 0.5) * L * 0.8, r * 0.7 + rnd() * 0.2, (rnd() - 0.5) * Wd * 0.8);
        m.rotation.set(rnd() * 6, rnd() * 6, rnd() * 6); m.scale.set(1.2, 0.8, 1);
      }
    }
  });

  return { shaded, singles, structure, rubble, furniture, decoy, warmObject, flush };
}
