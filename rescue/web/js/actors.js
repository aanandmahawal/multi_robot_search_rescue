// Everything that acts or changes during a mission: victims with their flags, robots with their
// camera cone and LiDAR scan, and the rescue team's base.
import * as THREE from "three";
import { FALSE_ALARM, ROBOT_COLORS, STATUS } from "./config.js";
import { hex, labelSprite, mulberry32, rgb } from "./util.js";

const MAX_BEAMS = 360;

// ------------------------------------------------------------------ victims: body, rubble on it, flag
export function makeVictim(v, world, group) {
  const g = new THREE.Group(), body = new THREE.Group();
  const parts = [];                                                  // meshes that are repainted for the sensor pictures
  const part = (mesh, kind) => { mesh.userData.part = kind; mesh.userData.base = mesh.material; parts.push(mesh); return mesh; };
  const mat = c => new THREE.MeshStandardMaterial({ color: rgb(c), roughness: 0.8 });
  const skin = mat(v.colors.skin), shirt = mat(v.colors.shirt), pants = mat(v.colors.pants);
  const cap = (r, len, m, x, y, z) => { const mesh = part(new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 6, 12), m), "cloth");
    mesh.rotation.z = Math.PI / 2; mesh.position.set(x, y, z); mesh.castShadow = true; body.add(mesh); };
  cap(0.075, 0.55, pants, 0.36, 0.08, 0.09); cap(0.075, 0.55, pants, 0.36, 0.08, -0.09);
  cap(0.16, 0.32, shirt, 0.98, 0.14, 0);
  cap(0.055, 0.42, shirt, 0.98, 0.07, 0.25); cap(0.055, 0.42, shirt, 0.98, 0.07, -0.25);
  for (const z of [0.25, -0.25]) { const h = part(new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), skin), "skin"); h.position.set(0.68, 0.07, z); body.add(h); }
  const head = part(new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 12), skin), "skin"); head.position.set(1.47, 0.13, 0); head.castShadow = true; body.add(head);
  const L = Math.hypot(v.head[0] - v.feet[0], v.head[1] - v.feet[1]) + 0.25;
  body.scale.set(L / 1.6, 1, 1); body.position.set(-L / 2, 0, 0);
  const holder = new THREE.Group(); holder.add(body);
  holder.rotation.y = -Math.atan2(v.head[1] - v.feet[1], v.head[0] - v.feet[0]);
  holder.position.set((v.head[0] + v.feet[0]) / 2, 0, (v.head[1] + v.feet[1]) / 2);
  holder.visible = v.cover !== "thick";                              // under thick rubble nothing of the body shows
  g.add(holder);

  if (v.covered && v.covered.length) {                               // rubble or debris lying on the body
    const f = world.fine, thick = v.cover !== "thin";
    const rockMat = new THREE.MeshStandardMaterial({ color: thick ? 0x8a847c : 0xa79f93, roughness: 1, flatShading: true });
    const rnd = mulberry32(v.id * 131 + 7);
    for (const [fx, fy] of v.covered) {
      const cx = (fx + 0.5) * f, cz = (fy + 0.5) * f;
      if (thick) {
        for (let i = 0; i < 4; i++) {
          const r = 0.09 + rnd() * 0.1, m = part(new THREE.Mesh(new THREE.DodecahedronGeometry(r, 0), rockMat), "cover");
          m.position.set(cx + (rnd() - 0.5) * 0.18, r * 0.6 + rnd() * 0.25, cz + (rnd() - 0.5) * 0.18);
          m.rotation.set(rnd() * 6, rnd() * 6, rnd() * 6); m.scale.set(1.2, 0.8, 1); m.castShadow = true; g.add(m);
        }
      } else {
        const m = part(new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.05 + rnd() * 0.05, 0.24), rockMat), "cover");
        m.position.set(cx, 0.22 + rnd() * 0.05, cz); m.rotation.y = rnd() * 0.6; m.castShadow = true; g.add(m);
      }
    }
  }

  const fx = v.head[0], fz = v.head[1], s = STATUS.hidden;
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 3.0, 8), new THREE.MeshStandardMaterial({ color: 0xd9dee4, metalness: 0.6, roughness: 0.35 }));
  pole.position.set(fx, 1.5, fz); pole.castShadow = true; g.add(pole);
  const clothGeo = new THREE.PlaneGeometry(1.0, 0.62, 12, 4); clothGeo.translate(0.5, 0, 0);
  const cloth = new THREE.Mesh(clothGeo, new THREE.MeshStandardMaterial({ color: s.color, side: THREE.DoubleSide, emissive: s.color, emissiveIntensity: 0.35, roughness: 0.6 }));
  cloth.position.set(fx + 0.03, 2.65, fz); cloth.castShadow = true; g.add(cloth);
  const ring = new THREE.Mesh(new THREE.RingGeometry(1.0, 1.18, 40),
    new THREE.MeshBasicMaterial({ color: s.color, transparent: true, opacity: 0.85, side: THREE.DoubleSide, fog: false }));
  ring.rotation.x = -Math.PI / 2; ring.position.set(v.x, 0.05, v.y); g.add(ring);
  const label = labelSprite(`V${v.id}  !`, s.css); label.position.set(fx, 3.35, fz); g.add(label);
  group.add(g);
  const flat = Object.fromEntries(["skin", "cloth", "cover"].map(k => [k, new THREE.MeshBasicMaterial({ toneMapped: false })]));
  return { group: g, cloth, clothGeo, base: clothGeo.attributes.position.array.slice(), ring, label, status: "hidden", v, parts, flat };
}

// repaint a victim for a sensor picture. colors = { skin, cloth, cover } as [r, g, b] (0-255), or null for the real colours
export function paintVictim(vv, colors) {
  for (const k of Object.keys(vv.flat)) if (colors) vv.flat[k].color.setRGB(colors[k][0] / 255, colors[k][1] / 255, colors[k][2] / 255, THREE.SRGBColorSpace);
  for (const m of vv.parts) m.material = colors ? vv.flat[m.userData.part] : m.userData.base;
}

// a place the robots confirmed although nobody is there
export function makeFalseAlarm(c, group) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.66, 32),
    new THREE.MeshBasicMaterial({ color: FALSE_ALARM.color, side: THREE.DoubleSide, transparent: true, opacity: 0.9, fog: false }));
  ring.rotation.x = -Math.PI / 2; ring.position.set(c.x, 0.06, c.y);
  // a person already counted, confirmed again further than 1.6 m from the first report, is not a phantom
  const twice = c.truth && c.truth.kind === "duplicate";
  const label = labelSprite(twice ? "⧉ counted twice" : "✕ false alarm", twice ? "#f5a524" : FALSE_ALARM.css, twice ? "#1a1300" : "#fff", 0.72);
  label.position.set(c.x, 1.25, c.y);
  if (twice) ring.material.color.set(0xf5a524);
  group.add(ring, label);
}

export function setVictimStatus(vv, status) {
  if (vv.status === status) return;
  vv.status = status;
  const s = STATUS[status];
  vv.cloth.material.color.set(s.color); vv.cloth.material.emissive.set(s.color); vv.ring.material.color.set(s.color);
  const old = vv.label; vv.group.remove(old); old.material.map.dispose();
  vv.label = labelSprite(`V${vv.v.id}  ${s.icon}`, s.css, status === "sighted" ? "#1a1300" : "#fff");
  vv.label.position.copy(old.position); vv.group.add(vv.label);
}

// the flag waves, the ring of a victim nobody has found yet pulses
export function animateVictim(vv, time) {
  const pos = vv.clothGeo.attributes.position;
  for (let k = 0; k < pos.count; k++) { const x = vv.base[k * 3]; pos.array[k * 3 + 2] = Math.sin(x * 5 - time * 4 + vv.v.id) * 0.08 * x; }
  pos.needsUpdate = true;
  vv.ring.scale.setScalar(vv.status === "hidden" ? 1 + 0.08 * Math.sin(time * 3 + vv.v.id) : 1);
}

// ------------------------------------------------------------------ robots
export function makeRobot(r, i, cfg, group) {
  const col = ROBOT_COLORS[i % ROBOT_COLORS.length];
  const g = new THREE.Group();
  const paint = new THREE.MeshStandardMaterial({ color: col, roughness: 0.4, metalness: 0.2 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1b1f24, roughness: 0.7 });
  const add = (geo, m, x, y, z) => { const mesh = new THREE.Mesh(geo, m); mesh.position.set(x, y, z); mesh.castShadow = true; g.add(mesh); return mesh; };
  add(new THREE.BoxGeometry(0.62, 0.22, 0.46), paint, 0, 0.2, 0);
  add(new THREE.BoxGeometry(0.44, 0.08, 0.36), dark, -0.04, 0.35, 0);
  const wheelGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.08, 18); wheelGeo.rotateX(Math.PI / 2);
  for (const [dx, dz] of [[0.2, 0.26], [-0.2, 0.26], [0.2, -0.26], [-0.2, -0.26]]) add(wheelGeo, dark, dx, 0.11, dz);
  // camera mast: colour + depth camera and thermal camera side by side
  add(new THREE.CylinderGeometry(0.025, 0.025, 0.3), new THREE.MeshStandardMaterial({ color: 0x9aa4ae }), 0.18, 0.5, 0);
  add(new THREE.BoxGeometry(0.12, 0.09, 0.2), dark, 0.2, cfg.camera_height + 0.1, 0);
  add(new THREE.CircleGeometry(0.035, 16), new THREE.MeshBasicMaterial({ color: 0x9fe7ff }), 0.265, cfg.camera_height + 0.1, 0.045).rotation.y = Math.PI / 2;
  add(new THREE.CircleGeometry(0.03, 16), new THREE.MeshBasicMaterial({ color: 0xff8c1a }), 0.265, cfg.camera_height + 0.1, -0.045).rotation.y = Math.PI / 2;
  // LiDAR: a small spinning drum on the deck
  let drum = null;
  if (cfg.lidar) {
    add(new THREE.CylinderGeometry(0.075, 0.085, 0.05, 20), dark, -0.1, cfg.lidar_height - 0.045, 0);
    drum = add(new THREE.CylinderGeometry(0.065, 0.065, 0.06, 20), new THREE.MeshStandardMaterial({ color: 0x2a2350, emissive: 0xa78bfa, emissiveIntensity: 0.9, roughness: 0.3 }), -0.1, cfg.lidar_height + 0.01, 0);
    add(new THREE.BoxGeometry(0.02, 0.03, 0.1), new THREE.MeshBasicMaterial({ color: 0xffffff }), -0.1, cfg.lidar_height + 0.01, 0).userData.spin = true;
  }
  const beacon = add(new THREE.SphereGeometry(0.05, 12, 10), new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 2 }), -0.26, 0.42, 0);
  const fov = cfg.camera_fov * Math.PI / 180;
  const cone = new THREE.Mesh(new THREE.CircleGeometry(cfg.camera_range, 32, -fov / 2, fov),
    new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
  cone.rotation.x = -Math.PI / 2; cone.position.y = 0.04; g.add(cone);
  const label = labelSprite("R" + r.id, hex(col), "#0b1016", 0.7); label.position.y = 1.15; g.add(label);
  group.add(g);

  const path = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.9 }));
  const goal = new THREE.Mesh(new THREE.OctahedronGeometry(0.2), new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.6 }));
  group.add(path, goal);

  // the latest LiDAR scan, in world coordinates: a dot where every beam ended, and faint beams
  const light = new THREE.Color(col).lerp(new THREE.Color(0xffffff), 0.45);
  const dots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: light, size: 0.11, sizeAttenuation: true, transparent: true, opacity: 0.95, depthWrite: false }));
  dots.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAX_BEAMS * 3), 3));
  const beams = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: light, transparent: true, opacity: 0.1, depthWrite: false }));
  beams.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAX_BEAMS * 6), 3));
  dots.frustumCulled = beams.frustumCulled = false;
  dots.visible = beams.visible = false;
  group.add(dots, beams);
  return { group: g, beacon, cone, label, path, goal, dots, beams, drum, spinner: g.children.find(c => c.userData.spin) };
}

// the name tag above a robot; with a battery it also shows the charge: "R0 · 62%" (green / amber / red),
// "⚡" while charging, "empty" once flat
export function setRobotLabel(R, r, i) {
  const col = hex(ROBOT_COLORS[i % ROBOT_COLORS.length]);
  let text = "R" + r.id, bg = col, fg = "#0b1016";
  if (r.battery != null) {
    const pct = Math.round(100 * r.battery);
    text += r.depleted ? " · empty" : r.state === "charging" ? ` · ⚡${pct}%` : ` · ${pct}%`;
    if (r.depleted || pct < 20) { bg = "#e5484d"; fg = "#fff"; }
    else if (r.low_battery || pct < 45) { bg = "#f5a524"; fg = "#1a1300"; }
    else if (r.state === "charging") { bg = "#3fd46b"; }
  }
  if (R.labelText === text + bg) return;
  R.labelText = text + bg;
  const old = R.label; R.group.remove(old); old.material.map.dispose(); old.material.dispose();
  R.label = labelSprite(text, bg, fg, 0.7); R.label.position.y = 1.15; R.group.add(R.label);
}

// draw a robot's LiDAR scan: `scan` is a list of [x, y] points in metres
export function setScan(R, r, scan, height, visible) {
  const n = Math.min(scan.length, MAX_BEAMS);
  const d = R.dots.geometry.attributes.position, b = R.beams.geometry.attributes.position;
  for (let k = 0; k < n; k++) {
    const [x, y] = scan[k];
    d.array[k * 3] = x; d.array[k * 3 + 1] = height; d.array[k * 3 + 2] = y;
    b.array[k * 6] = r.x; b.array[k * 6 + 1] = height; b.array[k * 6 + 2] = r.y;
    b.array[k * 6 + 3] = x; b.array[k * 6 + 4] = height; b.array[k * 6 + 5] = y;
  }
  d.needsUpdate = b.needsUpdate = true;
  R.dots.geometry.setDrawRange(0, n); R.beams.geometry.setDrawRange(0, n * 2);
  R.dots.visible = R.beams.visible = visible && n > 0;
}

// ------------------------------------------------------------------ the rescue team's base
export function makeBase(world, group) {
  const g = new THREE.Group(), [bx, bz] = world.base;
  const tent = new THREE.Mesh(new THREE.ConeGeometry(1.2, 1.4, 4, 1), new THREE.MeshStandardMaterial({ color: 0x2f6fd0, roughness: 0.8 }));
  tent.rotation.y = Math.PI / 4; tent.position.set(bx - 1.6, 0.7, bz); tent.castShadow = true; g.add(tent);
  const pad = new THREE.Mesh(new THREE.CircleGeometry(1.6, 40), new THREE.MeshStandardMaterial({ color: 0x1e3a5f }));
  pad.rotation.x = -Math.PI / 2; pad.position.set(bx - 1.2, 0.015, bz); g.add(pad);
  const light = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 10), new THREE.MeshStandardMaterial({ color: 0xff3b3b, emissive: 0xff3b3b, emissiveIntensity: 2 }));
  light.position.set(bx - 1.6, 1.5, bz); g.add(light);
  const label = labelSprite("BASE · rescue team", "#2f6fd0", "#fff", 1.1); label.position.set(bx - 1.2, 2.4, bz); g.add(label);
  group.add(g);
  return { group: g, light };
}
