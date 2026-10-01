"""Thermal model of the disaster zone: what a thermal camera would see.

Every fine cell (0.25 m) gets a **surface temperature** in degrees Celsius:

* the building sits at an **ambient** temperature that depends on the building type and
  drops when collapsed walls let outside air in, with gentle spatial drift and noise;
* **people are warm**: exposed skin 33-36 °C, clothing 27-31 °C (cloth insulates), and a
  person trapped for hours may have cooled by a few degrees (hypothermia);
* **warm objects that are not people** (space heaters, laptops on battery, hot-water leaks,
  pets, sun-warmed rubble, car engines that ran recently) give a thermal camera the same
  kind of false alarms that jackets and bags give a colour camera;
* **rubble on top of a person blocks the heat.** Warmth leaking through a cover falls off as
  exp(-thickness / 0.10 m): a thin layer of dust and light debris (6-14 cm) still shows a
  faint warm patch, but 40-80 cm of concrete rubble shows nothing at all.

That last point is the key limitation of thermal search: a person **buried under thick
rubble has no thermal signature** and cannot be found by any camera. The simulation creates
such victims on purpose (share depends on the damage level) and reports them separately, so
the results never pretend that cameras can see through concrete.

All randomness here uses its own random stream (seed + 7), so adding the thermal layer did
not change any building layout, victim or decoy position of the original simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .world import BUILDINGS, FINE, OBJ_WALL, World, _try_place

THERMAL_BASE = 200          # object id of warm object m in the fine grid = THERMAL_BASE + m

# ambient temperature per building (°C) and how much collapsed walls cool it down
AMBIENT = {"office": 20.0, "apartments": 21.0, "hospital": 22.0, "school": 19.0, "parking": 13.0, "warehouse": 15.0,
           "plain": 20.0}
COLLAPSE_COOLING = {"light": 0.5, "moderate": 1.5, "severe": 3.0}

# human surface temperatures as a thermal camera sees them (°C)
SKIN = (33.0, 36.0)         # face and hands
CLOTHING_DROP = (4.0, 7.0)  # clothing shows this much cooler than the skin under it
HYPOTHERMIA = 4.0           # a trapped person may have cooled by up to this much
HUMAN_MAX = 40.0            # nothing on a living person is hotter than this

# rubble on a body
COVER_THIN = (0.06, 0.14)   # metres of dust / light debris: some heat still leaks through
COVER_THICK = (0.40, 0.80)  # metres of concrete rubble: no heat reaches the surface
HEAT_LEAK_DEPTH = 0.10      # warmth through a cover falls as exp(-thickness / this)
COVER_VISIBLE = 0.20        # a body part under less cover than this still has a (faint) signature
# share of victims whose WHOLE body is covered, by damage level: (thin debris, thick rubble)
FULLY_BURIED = {"light": (0.05, 0.0), "moderate": (0.10, 0.10), "severe": (0.10, 0.20)}

# warm objects: kind -> (length, width in fine cells, height m, colour, (t_lo, t_hi) °C, buildings, blocks the way?)
WARM_OBJECTS = {
    "space heater": (3, 1, 0.55, (205, 205, 210), (42.0, 60.0), ("office", "apartments", "school", "hospital"), True),
    "laptop left running": (2, 1, 0.25, (40, 42, 48), (34.0, 42.0), ("office", "school", "hospital", "warehouse"), True),
    "hot-water leak": (4, 3, 0.01, (58, 66, 84), (28.0, 38.0), ("apartments", "hospital", "office", "school"), False),
    "dog": (3, 1, 0.30, None, (31.0, 36.0), ("apartments", "parking", "warehouse", "school"), True),
    "sun-warmed rubble": (3, 2, 0.35, (125, 118, 108), (28.0, 36.0), BUILDINGS, True),
}
FUR = [(120, 80, 40), (40, 35, 30), (230, 225, 210), (180, 140, 90)]
WARM_ENGINE_SHARE = 0.4     # share of parked cars whose engine ran recently
WARM_ENGINE = (40.0, 65.0)  # °C of the bonnet and front of such a car


@dataclass
class ThermalDecoy:
    """A warm object that is not a person: the thermal camera's version of a jacket on the floor."""
    id: int
    kind: str
    x: float
    y: float
    temp: float
    size: tuple[float, float] = (0.5, 0.5)
    height: float = 0.2
    color: list = field(default_factory=list)
    nav_cells: list = field(default_factory=list)


# ------------------------------------------------------------------ build
def add_thermal(world: World, rng: np.random.Generator) -> None:
    """Give the world a temperature field, warm objects and buried victims."""
    cfg = world.cfg
    Hf, Wf = world.height.shape
    building = cfg.building if cfg.building in AMBIENT else "office"
    amb = AMBIENT[building] - (0.0 if building == "plain" else COLLAPSE_COOLING.get(cfg.damage, 1.5))
    world.ambient = float(amb)
    drift = ndimage.gaussian_filter(rng.normal(0, 1, (Hf, Wf)), sigma=6)     # draughts, sunlit areas
    drift *= 0.8 / max(float(drift.std()), 1e-6)
    temp = amb + drift + rng.normal(0, 0.15, (Hf, Wf))
    temp[world.obj == OBJ_WALL] -= 0.6                                          # concrete and steel feel cooler
    if world.cover is None:
        world.cover = np.zeros((Hf, Wf))

    _bury_and_warm_victims(world, rng, temp)
    _place_warm_objects(world, rng, temp)
    _warm_car_engines(world, rng, temp)

    world.temp = temp.astype(np.float32)
    world.nav_temp = world.temp.reshape(world.H, FINE, world.W, FINE).max(axis=(1, 3))


def _bury_and_warm_victims(world: World, rng, temp) -> None:
    cfg = world.cfg
    thin_share, thick_share = (0.0, 0.0) if cfg.building == "plain" else FULLY_BURIED.get(cfg.damage, FULLY_BURIED["moderate"])
    for v in world.victims:
        cooled = rng.uniform(0, HYPOTHERMIA) if rng.random() < 0.5 else 0.0
        v.skin_temp = float(rng.uniform(*SKIN) - cooled)
        u = rng.random()
        if u < thick_share + thin_share:                       # the whole body disappears under a cover
            kind = "thick" if u < thick_share else "thin"
            lo, hi = COVER_THICK if kind == "thick" else COVER_THIN
            tone = np.array((118, 112, 104)) if kind == "thick" else np.array((156, 148, 136))
            v.covered = list(v.fine_cells)
            for fx, fy in v.fine_cells:
                h = float(rng.uniform(lo, hi))
                world.height[fy, fx] = h if kind == "thick" else world.height[fy, fx] + h
                world.cover[fy, fx] = h
                world.color[fy, fx] = np.clip(tone + rng.normal(0, 12, 3), 0, 255)
            v.cover, v.buried = kind, 1.0
        for li, (fx, fy) in zip(v.segments, v.fine_cells):
            body = v.skin_temp if li == 0 else v.skin_temp - rng.uniform(*CLOTHING_DROP)
            leak = float(np.exp(-world.cover[fy, fx] / HEAT_LEAK_DEPTH))
            temp[fy, fx] += (body - temp[fy, fx]) * leak


def _place_warm_objects(world: World, rng, temp) -> None:
    cfg = world.cfg
    building = cfg.building if cfg.building in AMBIENT else "office"
    kinds = [k for k, spec in WARM_OBJECTS.items() if building in spec[5]]
    keep_from = [(v.x, v.y) for v in world.victims]
    f = cfg.cell / FINE
    for j in range(0 if building == "plain" else cfg.n_warm_objects):   # the test ground has no warm look-alikes
        kind = kinds[j % len(kinds)]
        L, Wd, h, col, (t_lo, t_hi), _, blocks = WARM_OBJECTS[kind]
        placed = _try_place(world, rng, L, Wd, keep_from=keep_from, min_gap=2.0)
        if placed is None:
            continue
        cells, navs = placed
        col = np.array(col if col is not None else FUR[int(rng.integers(len(FUR)))], dtype=float)
        t = float(rng.uniform(t_lo, t_hi))
        m = len(world.thermal_decoys)
        for fx, fy in cells:
            world.height[fy, fx] = h * (rng.uniform(0.55, 1.2) if kind == "sun-warmed rubble" else rng.uniform(0.9, 1.05))
            world.color[fy, fx] = np.clip(col + rng.normal(0, 10, 3), 0, 255)
            world.obj[fy, fx] = THERMAL_BASE + m
            temp[fy, fx] = t + rng.normal(0, 0.8)
        if not blocks:                                          # a puddle does not stop a robot
            for nx, ny in navs:
                world.blocked[ny, nx] = False
        xs, ys = [c[0] for c in cells], [c[1] for c in cells]
        world.thermal_decoys.append(ThermalDecoy(m, kind, (np.mean(xs) + 0.5) * f, (np.mean(ys) + 0.5) * f, t,
                                                 size=((max(xs) - min(xs) + 1) * f, (max(ys) - min(ys) + 1) * f),
                                                 height=float(h), color=[int(c) for c in col], nav_cells=navs))


def _warm_car_engines(world: World, rng, temp) -> None:
    for it in world.furniture:
        if it.kind != "car" or rng.random() >= WARM_ENGINE_SHARE:
            continue
        it.temp = float(rng.uniform(*WARM_ENGINE))
        cells = it.cells()
        axis = 1 if it.facing in ("n", "s") else 0                # the bonnet points towards 'facing'
        coords = sorted({c[axis] for c in cells})
        n = max(1, len(coords) // 3)
        front = set(coords[:n]) if it.facing in ("n", "w") else set(coords[-n:])
        for x, y in cells:
            if (y if axis else x) in front:
                sy, sx = slice(y * FINE, (y + 1) * FINE), slice(x * FINE, (x + 1) * FINE)
                temp[sy, sx] = it.temp + rng.normal(0, 1.5, (FINE, FINE))


# ------------------------------------------------------------------ display
# "ironbow" palette used by thermal cameras: black -> purple -> red -> orange -> yellow -> white
IRONBOW = np.array([(0, 0, 0), (28, 0, 72), (120, 0, 150), (200, 30, 90), (240, 100, 20),
                    (255, 180, 0), (255, 235, 110), (255, 255, 255)], dtype=float)
SCALE_BELOW, SCALE_TOP = 3.0, 40.0    # images show ambient - 3 °C (black) ... 40 °C (white)


def to_unit(temp: np.ndarray, ambient: float) -> np.ndarray:
    lo = ambient - SCALE_BELOW
    return np.clip((np.asarray(temp, dtype=float) - lo) / (SCALE_TOP - lo), 0, 1)


def palette(unit: np.ndarray) -> np.ndarray:
    """0..1 -> uint8 RGB in the ironbow palette."""
    v = np.clip(unit, 0, 1) * (len(IRONBOW) - 1)
    i = np.clip(np.floor(v).astype(int), 0, len(IRONBOW) - 2)
    f = (v - i)[..., None]
    return np.clip(IRONBOW[i] * (1 - f) + IRONBOW[i + 1] * f, 0, 255).astype(np.uint8)


def describe_cover(v) -> str:
    """Plain words for how a victim is (not) visible."""
    return {"none": "lying in the open",
            "partial": f"partly buried ({round(100 * v.buried)}% under rubble)",
            "thin": "whole body under a thin layer of debris: a faint heat signature only the thermal camera can sense",
            "thick": "buried under thick rubble: no heat reaches the surface, no camera can detect this person"}[v.cover]
