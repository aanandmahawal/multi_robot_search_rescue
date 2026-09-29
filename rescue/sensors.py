"""Robot sensors.

``visible_cells``  what the RGB-D camera covers on the navigation map (used for mapping and
                   for measuring search coverage). Walls and rubble taller than the camera
                   block the view; low objects are seen but do not block it.
``render``         a real perspective camera image of the 3-D scene, made by ray-marching
                   through the fine heightfield. Returns colour, depth, the **thermal image**
                   (the surface temperature every pixel looks at, as a thermal camera bolted
                   next to the colour camera would measure it) and, for training only, which
                   object every pixel shows and how much rubble lies on it.

The thermal camera needs no light: it works in the dark and through dust. It is blurrier
than the colour camera and its readings carry sensor noise (about 0.25 °C) plus a small
per-frame calibration drift, like a real uncooled thermal core.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .world import FINE, World

THERMAL_BLUR = 0.8      # pixels: thermal optics are soft
THERMAL_NOISE = 0.25    # °C per pixel (sensor noise)
THERMAL_DRIFT = 0.3     # °C per frame (calibration drift of the whole image)


# ------------------------------------------------------------------ mapping sensor
def visible_cells(world: World, x: float, y: float, heading: float,
                  fov_deg: float | None = None, max_range: float | None = None) -> np.ndarray:
    """Navigation cells inside the camera frustum with a clear line of sight.

    Returns an int array of shape (n, 2) with unique (cx, cy) pairs.
    """
    cfg = world.cfg
    fov = np.radians(fov_deg if fov_deg is not None else cfg.camera_fov)
    rng_max = max_range if max_range is not None else cfg.camera_range
    n_rays = max(9, int(fov / np.radians(1.5)))
    angles = heading + np.linspace(-fov / 2, fov / 2, n_rays)
    t = np.arange(0.0, rng_max, cfg.cell * 0.35)
    px = x + np.cos(angles)[:, None] * t[None, :]
    py = y + np.sin(angles)[:, None] * t[None, :]
    cx = np.floor(px / cfg.cell).astype(int)
    cy = np.floor(py / cfg.cell).astype(int)
    inside = (cx >= 0) & (cx < world.W) & (cy >= 0) & (cy < world.H)
    cxc, cyc = np.clip(cx, 0, world.W - 1), np.clip(cy, 0, world.H - 1)
    stop = world.blocks_view()[cyc, cxc] | ~inside
    stop[:, 0] = False                                   # the robot's own cell never blocks
    first = np.where(stop.any(axis=1), stop.argmax(axis=1), t.size)
    keep = (np.arange(t.size)[None, :] <= first[:, None]) & inside
    cells = np.stack([cx[keep], cy[keep]], axis=1)
    return np.unique(cells, axis=0) if len(cells) else cells.reshape(0, 2)


def line_of_sight(world: World, a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if nothing taller than the camera lies between two points (metres)."""
    d = np.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(d / (world.cfg.cell * 0.35)))
    xs = np.linspace(a[0], b[0], n)[1:-1]
    ys = np.linspace(a[1], b[1], n)[1:-1]
    cx = np.floor(xs / world.cfg.cell).astype(int)
    cy = np.floor(ys / world.cfg.cell).astype(int)
    blocks = world.blocks_view()
    own = world.to_cell(*b)
    return not any(blocks[y_, x_] and (x_, y_) != own for x_, y_ in zip(cx, cy))


# ------------------------------------------------------------------ camera
@dataclass
class CameraFrame:
    rgb: np.ndarray        # float32 [H, W, 3] in 0..1
    depth: np.ndarray      # float32 [H, W] metres along the ray (max range where nothing was hit)
    obj: np.ndarray        # int16  [H, W] object id per pixel, -1 = nothing (ground truth, training only)
    points: np.ndarray     # float32 [H, W, 3] 3-D point each pixel sees (what a depth camera gives)
    pose: tuple            # (x, y, heading)
    thermal: np.ndarray | None = None   # float32 [H, W] surface temperature in °C (the thermal camera image)
    cover: np.ndarray | None = None     # float32 [H, W] rubble on the object each pixel sees, metres (training only)


_RAY_CACHE: dict = {}


def _camera_rays(cfg) -> np.ndarray:
    """Unit ray directions in the camera frame: (forward, right, up) per pixel."""
    key = (cfg.image_w, cfg.image_h, cfg.camera_fov, cfg.camera_pitch)
    if key not in _RAY_CACHE:
        W, H = cfg.image_w, cfg.image_h
        f = (W / 2) / np.tan(np.radians(cfg.camera_fov) / 2)
        u = (np.arange(W) + 0.5 - W / 2) / f
        v = (np.arange(H) + 0.5 - H / 2) / f
        uu, vv = np.meshgrid(u, v)
        p = np.radians(cfg.camera_pitch)
        fwd, right, up = np.ones_like(uu), uu, -vv
        horiz = fwd * np.cos(p) - up * np.sin(p)
        z = fwd * np.sin(p) + up * np.cos(p)
        d = np.stack([horiz, right, z], axis=-1)
        _RAY_CACHE[key] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return _RAY_CACHE[key]


def render(world: World, x: float, y: float, heading: float,
           rng: np.random.Generator | None = None) -> CameraFrame:
    """Render what a robot's camera at (x, y) facing ``heading`` sees."""
    cfg = world.cfg
    cam = _camera_rays(cfg)                                   # [H, W, 3]
    c, s = np.cos(heading), np.sin(heading)
    dx = cam[..., 0] * c - cam[..., 1] * s
    dy = cam[..., 0] * s + cam[..., 1] * c
    dz = cam[..., 2]
    t_max = cfg.camera_range + 2.0
    t = np.arange(0.12, t_max, 0.07, dtype=np.float32)
    px = x + dx[..., None] * t
    py = y + dy[..., None] * t
    pz = cfg.camera_height + dz[..., None] * t

    fcell = cfg.cell / FINE
    fx = np.floor(px / fcell).astype(np.int32)
    fy = np.floor(py / fcell).astype(np.int32)
    Hf, Wf = world.height.shape
    inside = (fx >= 0) & (fx < Wf) & (fy >= 0) & (fy < Hf)
    fxc, fyc = np.clip(fx, 0, Wf - 1), np.clip(fy, 0, Hf - 1)
    ground = world.height[fyc, fxc]
    hit = inside & ((pz <= ground) | (pz <= 0.0))

    any_hit = hit.any(axis=-1)
    k = hit.argmax(axis=-1)
    rows, cols = np.indices(k.shape)
    hx, hy = fxc[rows, cols, k], fyc[rows, cols, k]
    depth = np.where(any_hit, t[k], t_max).astype(np.float32)
    obj = np.where(any_hit, world.obj[hy, hx], -1).astype(np.int16)

    # shading: top faces bright, side faces darker, headlamp light falls off with distance
    kp = np.maximum(k - 1, 0)
    same_cell = (fxc[rows, cols, kp] == hx) & (fyc[rows, cols, kp] == hy)
    shade = np.where(same_cell | (pz[rows, cols, k] <= 0.02), 1.0, 0.7)
    light = 0.04 + 1.15 / (1.0 + 0.09 * depth ** 2)   # robot headlamp in a dark building
    rgb = world.color[hy, hx].astype(np.float32) / 255.0 * (shade * light)[..., None]
    rgb[~any_hit] = (0.03, 0.03, 0.035)

    points = np.stack([px[rows, cols, k], py[rows, cols, k], pz[rows, cols, k]], axis=-1).astype(np.float32)

    # thermal camera: the temperature of whatever each pixel looks at (nothing in range = ambient)
    thermal = cover = None
    if world.temp is not None:
        thermal = np.where(any_hit, world.temp[hy, hx], world.ambient).astype(np.float32)
        thermal += (world.ambient - thermal) * 0.06 * (1.0 - np.exp(-depth / 4.0))   # dust between camera and object
        thermal = ndimage.gaussian_filter(thermal, THERMAL_BLUR)
        cover = np.where(any_hit, world.cover[hy, hx], 0.0).astype(np.float32)

    if rng is not None:   # sensor imperfections: dust haze, exposure, noise
        haze = rng.uniform(0.0, 0.25) * (1.0 - np.exp(-depth / 4.0))
        rgb = rgb * (1 - haze[..., None]) + haze[..., None] * np.array([0.30, 0.28, 0.25])
        rgb = rgb * rng.uniform(0.8, 1.15) + rng.normal(0, 0.025, rgb.shape)
        if thermal is not None:
            thermal = (thermal + rng.normal(0, THERMAL_NOISE, thermal.shape) + rng.normal(0, THERMAL_DRIFT)).astype(np.float32)
    return CameraFrame(np.clip(rgb, 0, 1).astype(np.float32), depth, obj, points, (x, y, heading),
                       thermal=thermal, cover=cover)
