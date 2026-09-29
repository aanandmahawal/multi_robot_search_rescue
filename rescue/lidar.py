"""2-D LiDAR: a spinning laser range finder on top of every robot.

Once per second the laser makes a full turn and measures, in 240 directions (1.5 degrees
apart), the distance to the nearest surface that crosses its **scan plane** 0.40 m above the
floor, up to 8 m away.

What LiDAR is good at
    Geometry. It sees in every direction at once, in complete darkness, further than the
    cameras (8 m against 5 m), and its distances are accurate to about 2 cm. That makes it the
    right sensor for building the map the robots drive on.

What LiDAR cannot do
    * It only sees what crosses the scan plane. Anything lower (a person lying on the floor,
      a jacket, a bag, low rubble, a puddle) is invisible to it: the beam passes over the top.
      The depth camera sees those inside its narrower field of view, and the bumper catches
      what both missed.
    * It measures distance, nothing else. It cannot tell a person from a box: no colour, no
      warmth. **LiDAR maps the building; the cameras find the people.**

The readings are noisy (2 cm, plus about 1 % of beams that return nothing), and a beam that
clips the corner of a wall ends in a cell that is mostly free. So inside one scan a cell only
counts as "hit" when more beams ended in it than passed through it, and one scan is never
trusted on its own: ``mapping.apply_lidar`` adds every scan to a probabilistic occupancy grid
that weighs up repeated evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RescueConfig
from .world import FINE, World

STEP = 0.08          # metres between samples along a beam
SUB = 8              # ...refined to STEP / SUB (1 cm) where the beam meets a surface
MARGIN = 0.06        # inverse sensor model: a beam proves free space up to (range - margin), so 2 cm of
                     # noise rarely frees the cell in front of a wall
HIT_DEPTH = 0.03     # ...and the hit is placed this far behind the measured surface: deep enough for 2 cm of noise,
                     # shallow enough that a beam grazing a corner does not come out in the empty cell beyond


@dataclass
class LidarScan:
    angles: np.ndarray        # [n] beam directions, radians in the world frame
    ranges: np.ndarray        # [n] measured distance in metres (noisy); inf = nothing came back
    points: np.ndarray        # [m, 2] where the returning beams ended, metres (for display)
    free_cells: np.ndarray    # [k, 2] navigation cells (x, y) the beams passed through
    hit_cells: np.ndarray     # [j, 2] navigation cells (x, y) in which a beam ended on a surface
    pose: tuple               # (x, y) of the scanner


def _cells(flat: np.ndarray, W: int) -> np.ndarray:
    return np.stack([flat % W, flat // W], axis=1).astype(int)


def scan(world: World, x: float, y: float, cfg: RescueConfig, rng: np.random.Generator | None = None) -> LidarScan:
    """One revolution of the laser at (x, y). Pass ``rng`` for a realistic (noisy) sensor."""
    n = cfg.lidar_rays
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    t = np.arange(STEP, cfg.lidar_range + STEP, STEP)                       # [T]
    ca, sa = np.cos(angles)[:, None], np.sin(angles)[:, None]
    px, py = x + ca * t, y + sa * t                                         # [n, T]

    f = cfg.cell / FINE
    Hf, Wf = world.height.shape
    fx, fy = np.floor(px / f).astype(int), np.floor(py / f).astype(int)
    inside = (fx >= 0) & (fx < Wf) & (fy >= 0) & (fy < Hf)
    tall = world.height[np.clip(fy, 0, Hf - 1), np.clip(fx, 0, Wf - 1)] > cfg.lidar_height
    stop = (tall & inside) | ~inside
    first = np.where(stop.any(axis=1), stop.argmax(axis=1), t.size - 1)
    rows = np.arange(n)
    returned = stop.any(axis=1) & inside[rows, first]          # ended on a surface, not at the edge of the map
    # the first sample inside a surface can lie up to one STEP behind it: find the surface to 1 cm, as a
    # real laser measures it (otherwise a beam grazing a corner "ends" in the empty cell beyond)
    sub = t[first][:, None] - STEP + np.arange(1, SUB + 1)[None, :] * (STEP / SUB)          # [n, SUB]
    sx, sy = np.floor((x + ca * sub) / f).astype(int), np.floor((y + sa * sub) / f).astype(int)
    s_in = (sx >= 0) & (sx < Wf) & (sy >= 0) & (sy < Hf)
    s_stop = ((world.height[np.clip(sy, 0, Hf - 1), np.clip(sx, 0, Wf - 1)] > cfg.lidar_height) & s_in) | ~s_in
    surface = sub[rows, np.where(s_stop.any(axis=1), s_stop.argmax(axis=1), SUB - 1)]
    measured = np.where(returned, surface, np.inf)
    reach = np.where(stop.any(axis=1), surface, cfg.lidar_range)           # how far the beam really travelled

    if rng is not None:
        measured = measured + rng.normal(0, cfg.lidar_noise, n)
        lost = rng.random(n) < cfg.lidar_dropout                           # a beam that returned nothing tells nothing
        measured[lost], reach[lost], returned = np.inf, 0.0, returned & ~lost
        reach = np.where(returned, measured, reach)

    # inverse sensor model: count, for every cell, the beams that passed through and the beams that ended in it
    W, n_cells = world.W, world.W * world.H
    free = (t[None, :] < (reach - MARGIN)[:, None]) & inside
    cell_id = (np.floor(py / cfg.cell).astype(np.int64) * W + np.floor(px / cfg.cell).astype(np.int64))
    beam = np.broadcast_to(rows[:, None], cell_id.shape)
    pairs = np.unique(beam[free] * n_cells + cell_id[free])                 # one count per (beam, cell)
    passes = np.bincount(pairs % n_cells, minlength=n_cells)
    end = measured[returned] + HIT_DEPTH
    ex, ey = x + np.cos(angles[returned]) * end, y + np.sin(angles[returned]) * end
    hx = np.clip(np.floor(ex / cfg.cell).astype(np.int64), 0, W - 1)
    hy = np.clip(np.floor(ey / cfg.cell).astype(np.int64), 0, world.H - 1)
    hits = np.bincount(hy * W + hx, minlength=n_cells)
    is_hit = hits > passes
    hit_cells = _cells(np.flatnonzero(is_hit), W)
    free_cells = _cells(np.flatnonzero((passes > 0) & ~is_hit), W)

    points = np.stack([x + np.cos(angles[returned]) * measured[returned],
                       y + np.sin(angles[returned]) * measured[returned]], axis=1).astype(np.float32)
    return LidarScan(angles, measured.astype(np.float32), points, free_cells, hit_cells, (x, y))
