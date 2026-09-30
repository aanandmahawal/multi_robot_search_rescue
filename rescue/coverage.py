"""Coverage patterns: the path a robot follows to search its part of the building.

Three patterns (``cfg.coverage``):

* **frontier**       Frontier-based exploration (Yamauchi 1997). No fixed path: a robot drives to
                     the edge of the searched area (a *frontier*), looks, and picks the next one.
                     Which frontier is chosen is the team strategy's job (``coordination.py``).
* **boustrophedon**  Lawnmower / "ox-turning" coverage (Choset & Pignon 1997). The robot's region is
                     swept in parallel lanes, alternating direction, joined by short turns.
* **spiral**         Spiral coverage: the region is swept from the outside in along rectangular
                     rings that shrink by one lane spacing per pass.

**Regions.** Rescuers know the outline of the building, not what is inside. For the two fixed
patterns the building's rectangle is split into n equal rectangles, one per robot: an r x c grid
(r c = n) chosen so the rectangles are as square as possible (min over factorisations of the
largest aspect ratio). Robots are matched to regions so the total distance from where they stand
to where their pattern starts is smallest (Hungarian method).

**Lane spacing.** A robot searches with its forward camera: range R, field of view F. A cell at
sideways offset e from the lane is seen once the robot is at forward distance f >= e (inside the
cone) with sqrt(f^2 + e^2) <= R, so the swept band has half-width R sin(F/2). Lanes are spaced

        w = 2 R sin(F/2) (1 - overlap),      overlap = 0.25

(5.3 m for the default 5 m, 90 degree camera). A LiDAR-only team maps all around: w = 2 R_lidar
(1 - overlap).

**Lawnmower.** Lanes run parallel to the region's *longer* side: that minimises the number of lanes
and so the number of turns, the costly part of a sweep (Huang 2001: the best sweep direction is
the one with the smallest width across it). With D the region's width across the lanes,

        K = ceil(D / w) lanes,     lane k at  a0 + (k + 1/2) D / K,      k = 0 .. K-1

so neighbouring lanes are D/K <= w apart and the outer lanes lie D/(2K) <= w/2 from the edge.
Even-numbered lanes run one way, odd ones the other way.

**Spiral.** Passes are D_x / K_x apart across x and D_y / K_y apart across y (K = ceil(D / w) as
above). Starting in a corner, the robot drives the top edge, the right edge, the bottom edge and
the left edge, and every edge moves one spacing inwards once it has been driven (the classic
"spiral matrix" order). It stops when two opposite edges have crossed. Every horizontal pass
is one of the K_y lane lines, every vertical one one of the K_x lines: the same spacing
guarantee as the lawnmower.

Both patterns start in the region corner nearest the robot (the pattern is mirrored to put it
there; mirroring once turns a clockwise spiral counter-clockwise). They are then cut into
waypoints at most ``WAYPOINT_STEP`` apart. A robot drives from waypoint to waypoint with the
chosen route planner; unknown cells are assumed free (the free-space assumption) and a waypoint
the robot finds inside an obstacle, or cannot reach at all, is skipped. When a robot's pattern is
done, it helps with whatever the pattern missed (shadows behind obstacles) by frontier
exploration, so the search still ends only when nothing reachable is left unsearched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

PATTERNS = ("frontier", "boustrophedon", "spiral")
OVERLAP = 0.25            # share of a lane's width that neighbouring lanes overlap
WAYPOINT_STEP = 1.5       # metres between waypoints along a pattern
END_INSET = 1.0           # metres: lane ends stay this far from the edge of the region


def lane_width(cfg) -> float:
    """Distance w (metres) between neighbouring lanes (see the module doc)."""
    if cfg.vision != "none":
        half = cfg.camera_range * np.sin(np.radians(cfg.camera_fov) / 2)
    else:
        half = cfg.lidar_range
    return float(2 * half * (1 - OVERLAP))


def building_box(world) -> tuple[float, float, float, float]:
    """The inside of the building's outer wall in metres (x0, y0, x1, y1): all rescuers know."""
    c = world.cfg.cell
    return 4 * c, 1 * c, (world.W - 1) * c, (world.H - 1) * c        # the outer wall is at x = 3, W-1 and y = 0, H-1


def grid_shape(n: int, width: float, height: float) -> tuple[int, int]:
    """(rows, cols) with rows x cols = n whose rectangles are the most square."""
    best = None
    for rows in range(1, n + 1):
        if n % rows:
            continue
        cols = n // rows
        w, h = width / cols, height / rows
        aspect = max(w / h, h / w)
        if best is None or aspect < best[0] - 1e-9:
            best = (aspect, rows, cols)
    return best[1], best[2]


def regions(world, n: int) -> list[tuple[float, float, float, float]]:
    """The building's rectangle split into n equal rectangles (x0, y0, x1, y1), row by row."""
    x0, y0, x1, y1 = building_box(world)
    rows, cols = grid_shape(n, x1 - x0, y1 - y0)
    xs, ys = np.linspace(x0, x1, cols + 1), np.linspace(y0, y1, rows + 1)
    return [(float(xs[j]), float(ys[i]), float(xs[j + 1]), float(ys[i + 1])) for i in range(rows) for j in range(cols)]


def region_map(world, boxes) -> np.ndarray:
    """Navigation cell -> index of the region it lies in (cells outside the building: the nearest)."""
    c = world.cfg.cell
    gy, gx = np.mgrid[0:world.H, 0:world.W]
    px, py = (gx + 0.5) * c, (gy + 0.5) * c
    d = np.stack([np.hypot(np.clip(px, b[0], b[2]) - px, np.clip(py, b[1], b[3]) - py) for b in boxes])
    return np.argmin(d, axis=0)


# ------------------------------------------------------------------ the two patterns
def _mirror(pts, box, flip_x, flip_y):
    x0, y0, x1, y1 = box
    return [((x0 + x1 - x) if flip_x else x, (y0 + y1 - y) if flip_y else y) for x, y in pts]


def _corner_flips(box, start) -> tuple[bool, bool]:
    """Mirror the pattern so that it starts in the corner nearest ``start``."""
    x0, y0, x1, y1 = box
    return abs(start[0] - x1) < abs(start[0] - x0), abs(start[1] - y1) < abs(start[1] - y0)


def lanes(lo: float, hi: float, w: float) -> list[float]:
    """K = ceil(D / w) lane lines across [lo, hi], D / K apart, the outer ones D / 2K from the edges."""
    D = hi - lo
    K = max(1, int(np.ceil(D / w - 1e-9)))
    return [lo + (k + 0.5) * D / K for k in range(K)]


def lawnmower(box, w: float) -> tuple[list, str]:
    """Corner points of a boustrophedon sweep of ``box``, starting at its top-left corner."""
    x0, y0, x1, y1 = box
    horizontal = (x1 - x0) >= (y1 - y0)                               # lanes along the longer side
    if horizontal:
        e = min(END_INSET, (x1 - x0) / 2)
        pts = []
        for k, y in enumerate(lanes(y0, y1, w)):
            a, b = (x0 + e, x1 - e) if k % 2 == 0 else (x1 - e, x0 + e)
            pts += [(a, y), (b, y)]
    else:
        e = min(END_INSET, (y1 - y0) / 2)
        pts = []
        for k, x in enumerate(lanes(x0, x1, w)):
            a, b = (y0 + e, y1 - e) if k % 2 == 0 else (y1 - e, y0 + e)
            pts += [(x, a), (x, b)]
    n = len(pts) // 2
    return pts, f"{n} {'horizontal' if horizontal else 'vertical'} lane{'s' if n != 1 else ''}"


def spiral(box, w: float) -> tuple[list, str]:
    """Corner points of a rectangular spiral over ``box``, from the outside in, clockwise on the
    screen (y points down), starting at its top-left corner."""
    x0, y0, x1, y1 = box
    xs, ys = lanes(x0, x1, w), lanes(y0, y1, w)
    left, right, top, bottom = 0, len(xs) - 1, 0, len(ys) - 1            # indices into the lane lines
    pts = [(xs[left], ys[top])]
    rings = 0                                                             # each ring starts with a top edge
    while True:
        rings += 1
        pts.append((xs[right], ys[top])); top += 1                        # top edge, left to right
        if top > bottom:
            break
        pts.append((xs[right], ys[bottom])); right -= 1                   # right edge, downwards
        if left > right:
            break
        pts.append((xs[left], ys[bottom])); bottom -= 1                   # bottom edge, right to left
        if top > bottom:
            break
        pts.append((xs[left], ys[top])); left += 1                        # left edge, upwards
        if left > right:
            break
    out = [pts[0]]
    for p in pts[1:]:                                                     # drop zero-length legs
        if np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return out, f"{rings} ring{'s' if rings != 1 else ''}"


def densify(pts: list, step: float = WAYPOINT_STEP) -> list:
    """Cut every leg into pieces at most ``step`` long (keeping the corners)."""
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / step - 1e-9)))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


def path_length(pts: list) -> float:
    return float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])))


# ------------------------------------------------------------------ one robot's sweep
@dataclass
class Sweep:
    """A robot's coverage pattern and how far along it is."""
    pattern: str
    region: int
    box: tuple
    corners: list                       # the pattern's corner points (metres), in driving order
    points: list                        # waypoints (metres), at most WAYPOINT_STEP apart
    cells: list                         # the same waypoints as navigation cells
    describe: str                       # e.g. "3 horizontal lanes" / "2 rings, clockwise"
    index: int = 0                      # next waypoint to drive to
    reached: int = 0
    skipped: list = field(default_factory=list)    # waypoints dropped: inside an obstacle / out of reach
    start_move: int | None = None       # the robot's move count when it reached its first waypoint...
    end_move: int | None = None         # ...and when it finished the pattern (to measure how well it kept to it)
    end_t: int | None = None            # mission time when the pattern was finished

    @property
    def done(self) -> bool:
        return self.index >= len(self.cells)

    def arrive(self, moves: int) -> None:
        """The robot stands on the next waypoint (``moves`` = cells driven so far)."""
        self.index += 1
        self.reached += 1
        if self.start_move is None:
            self.start_move = moves

    def where(self, i: int | None = None) -> str:
        """Plain words for waypoint i (default: the next one)."""
        i = self.index if i is None else i
        return f"waypoint {i + 1} of {len(self.cells)} of my {self.pattern_name} ({self.describe})"

    @property
    def pattern_name(self) -> str:
        return {"boustrophedon": "lawnmower sweep", "spiral": "spiral"}[self.pattern]


def make_sweep(pattern: str, world, box, region: int, start_xy) -> Sweep:
    w = lane_width(world.cfg)
    pts, what = (lawnmower if pattern == "boustrophedon" else spiral)(box, w)
    fx, fy = _corner_flips(box, start_xy)
    pts = _mirror(pts, box, fx, fy)
    if pattern == "spiral":
        what += ", " + ("counter-clockwise" if fx != fy else "clockwise")
    what += f", {w:.1f} m apart"
    points = densify(pts)
    cells, keep = [], []
    for p in points:                                                 # neighbouring waypoints in the same cell: once
        c = world.to_cell(*p)
        c = (min(max(c[0], 0), world.W - 1), min(max(c[1], 0), world.H - 1))
        if not cells or cells[-1] != c:
            cells.append(c)
            keep.append(p)
    return Sweep(pattern, region, tuple(box), pts, keep, cells, what)


def assign(starts_xy: list, boxes: list, pattern: str, world) -> list[int]:
    """Region index for every robot: the matching with the smallest total distance from each robot
    to the first waypoint of its pattern (Hungarian method; exact search for small teams)."""
    n = len(starts_xy)
    cost = np.zeros((n, n))
    for j, b in enumerate(boxes):
        for i, s in enumerate(starts_xy):
            first = make_sweep(pattern, world, b, j, s).points[0]
            cost[i, j] = np.hypot(first[0] - s[0], first[1] - s[1])
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        out = [0] * n
        for i, j in zip(rows, cols):
            out[int(i)] = int(j)
        return out
    except ImportError:                                               # pragma: no cover - scipy is a dependency
        return list(min(permutations(range(n)), key=lambda p: sum(cost[i, p[i]] for i in range(n))))
