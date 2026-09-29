"""Occupancy maps, frontiers and path planning.

Each robot keeps its own map of the building. Every cell is ``UNKNOWN``, ``FREE`` or
``OBSTACLE``; maps are merged whenever robots can talk to each other.

**Two sensors write the map.** The LiDAR sees far and all around but is noisy and blind below
its scan plane; the depth camera sees only ahead but also the floor and low obstacles. They
are combined in an **evidence grid** (a probabilistic occupancy grid, Moravec & Elfes 1985):
every cell holds the log-odds that it is occupied. A laser beam passing through a cell lowers
it, a beam ending in a cell raises it, and a cell is only called FREE or OBSTACLE once the
evidence is clear. What the camera saw from close by is final: a jacket on the floor stays an
obstacle even though every laser beam passes over it.

**Two different questions.** "Do I know what is there?" (the map: can I drive through) and
"Has anyone looked there for people?" (the *searched* layer: only cameras can answer that).
A **frontier** is a searched free cell next to a cell no camera has looked at yet, whatever the
laser says about that cell: the edge of the searched area. Driving to frontiers is the classic autonomous exploration strategy (Yamauchi,
1997); neighbouring frontier cells are grouped into clusters and each cluster is one possible
goal. Its size estimates how much new area a robot would see there. Without LiDAR the map and
the searched layer are the same thing, and this reduces to plain frontier exploration.

**When the sensors cannot see further.** Sometimes a robot stands on a frontier and still
cannot resolve the unknown cell next to it: a table top blocks the laser, a door frame makes the
laser call a passable cell an obstacle, a camera cannot see into a corner. Then the robot
*probes*: it drives slowly into that cell. Either it gets in (the cell is free: a robot knows the
floor it stands on) or the bumper stops it (an obstacle). A cell that is still unresolved after a
probe is *given up* and no longer counts as a frontier, so the search always ends.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

UNKNOWN, FREE, OBSTACLE = 0, 1, 2
MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]

# evidence grid: log-odds that a cell is occupied
L_HIT = 0.9        # a laser beam ended in this cell
L_PASS = 0.45      # a laser beam passed through this cell
L_LIDAR_MAX = 2.5  # the laser alone never gets more certain than this...
L_MAX = 4.0        # ...so what the camera saw from close by, or the bumper felt, always wins
L_DECIDE = 0.8     # |evidence| needed to call a cell FREE or OBSTACLE


@dataclass
class Frontier:
    id: int
    cell: tuple[int, int]        # goal cell (a frontier cell near the cluster's centre)
    size: int                    # number of frontier cells = expected information gain
    cells: np.ndarray


# ------------------------------------------------------------------ writing the map
def apply_lidar(evidence: np.ndarray, solid: np.ndarray, free_cells: np.ndarray, hit_cells: np.ndarray) -> None:
    """Add one LiDAR scan. Cells the camera or the bumper proved solid are never freed again:
    the laser passes over low obstacles without noticing them."""
    if len(free_cells):
        fx, fy = free_cells[:, 0], free_cells[:, 1]
        keep = ~solid[fy, fx]
        fx, fy = fx[keep], fy[keep]
        ev = evidence[fy, fx]
        evidence[fy, fx] = np.maximum(ev - L_PASS, np.minimum(ev, -L_LIDAR_MAX))
    if len(hit_cells):
        hx, hy = hit_cells[:, 0], hit_cells[:, 1]
        ev = evidence[hy, hx]
        evidence[hy, hx] = np.minimum(ev + L_HIT, np.maximum(ev, L_LIDAR_MAX))


def apply_camera(evidence: np.ndarray, solid: np.ndarray, cells: np.ndarray, blocked: np.ndarray) -> None:
    """The depth camera sees the floor and everything on it inside its field of view."""
    if len(cells) == 0:
        return
    xs, ys = cells[:, 0], cells[:, 1]
    b = blocked[ys, xs]
    evidence[ys, xs] = np.where(b, L_MAX, -L_MAX)
    solid[ys[b], xs[b]] = True


def classify(evidence: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Evidence grid -> UNKNOWN / FREE / OBSTACLE."""
    known = np.zeros(evidence.shape, np.int8) if out is None else out
    known[...] = UNKNOWN
    known[evidence <= -L_DECIDE] = FREE
    known[evidence >= L_DECIDE] = OBSTACLE
    return known


def merge_evidence(grids: list[np.ndarray], solid: np.ndarray | None = None) -> np.ndarray:
    """Team map: for every cell keep the strongest opinion anyone has (sharing the same
    observation twice must not count twice, so evidence is not added up). Cells somebody
    proved solid stay obstacles whatever the lasers say."""
    merged = grids[0].copy()
    for g in grids[1:]:
        stronger = np.abs(g) > np.abs(merged)
        merged[stronger] = g[stronger]
    if solid is not None:
        merged[solid] = L_MAX
    return merged


def update_map(known: np.ndarray, cells: np.ndarray, blocked: np.ndarray) -> int:
    """Mark observed cells free / obstacle directly (no evidence grid). Returns how many were new."""
    if len(cells) == 0:
        return 0
    xs, ys = cells[:, 0], cells[:, 1]
    new = int((known[ys, xs] == UNKNOWN).sum())
    known[ys, xs] = np.where(blocked[ys, xs], OBSTACLE, FREE)
    return new


def explore_map(known: np.ndarray, searched: np.ndarray, gaveup: np.ndarray | None = None) -> np.ndarray:
    """The map as the *search* sees it: a cell is FREE or OBSTACLE only once a camera has
    looked at it, and UNKNOWN (still to be searched) otherwise. That includes obstacles only the
    laser has seen: a heap the laser bounced off may be rubble lying on a person. Cells the team
    gave up on (still unresolved after a probe) are treated as obstacles: nobody goes back."""
    out = np.zeros(known.shape, np.int8)
    out[(known == FREE) & searched] = FREE
    out[(known == OBSTACLE) & searched] = OBSTACLE
    if gaveup is not None:
        out[gaveup & (out == UNKNOWN)] = OBSTACLE
    return out


# ------------------------------------------------------------------ reading the map
def frontiers(known: np.ndarray, min_size: int = 1) -> list[Frontier]:
    free = known == FREE
    unknown = known == UNKNOWN
    near_unknown = ndimage.binary_dilation(unknown, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    fmask = free & near_unknown
    labels, n = ndimage.label(fmask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        cells = np.argwhere(labels == k)[:, ::-1]        # (x, y)
        if len(cells) < min_size:
            continue
        centre = cells.mean(axis=0)
        goal = cells[np.argmin(((cells - centre) ** 2).sum(axis=1))]
        out.append(Frontier(len(out), (int(goal[0]), int(goal[1])), len(cells), cells))
    return out


def dijkstra(known: np.ndarray, start: tuple[int, int], avoid: set | None = None, optimistic: bool = False):
    """Shortest path distances (in cells) over known free space, 8-connected, no corner cutting.

    ``optimistic=True`` also allows unknown cells (assume they are free until seen otherwise);
    used to head towards a region the robot has not mapped yet.
    Returns (dist array with inf for unreachable, parent dict).
    """
    H, W = known.shape
    passable = (known != OBSTACLE) if optimistic else (known == FREE)
    dist = np.full((H, W), np.inf)
    dist[start[1], start[0]] = 0.0
    parent: dict = {}
    heap = [(0.0, start)]
    avoid = avoid or set()
    while heap:
        d, (x, y) = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dx, dy, cost in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H) or not passable[ny, nx] or (nx, ny) in avoid:
                continue
            if dx and dy and not (passable[y, nx] and passable[ny, x]):
                continue                                   # do not cut corners
            nd = d + cost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                parent[(nx, ny)] = (x, y)
                heapq.heappush(heap, (nd, (nx, ny)))
    return dist, parent


def extract_path(parent: dict, start, goal) -> list[tuple[int, int]]:
    if goal == start:
        return []
    if goal not in parent:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.pop()
    return path[::-1]
