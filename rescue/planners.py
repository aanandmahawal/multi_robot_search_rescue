"""Route planners: how a robot gets from where it is to the goal it has chosen.

Choosing *which* goal to go to is done elsewhere (``coordination.py``); it uses one Dijkstra
distance field from the robot to every cell, because it must compare many goals at once. Once a
goal is chosen, the route to it is planned by one of four algorithms (``cfg.planner``). All four
work on the robot's own map and obey the same rules, so they can be compared fairly:

* a robot drives only through cells its map calls FREE;
* it moves in 8 directions, a straight step costs 1 cell (0.5 m), a diagonal step sqrt(2);
* it never cuts the corner of an obstacle (a diagonal step needs both side cells free);
* optionally, a cell costs ``extra[y, x]`` more to enter (e.g. a penalty for cells the team has
  already driven through, so routes prefer fresh ground).

**Dijkstra** (1959) expands cells in order of their cost g(n) from the start: it is optimal and
explores in all directions evenly.

**A*** (Hart, Nilsson, Raphael 1968) expands in order of f(n) = g(n) + h(n), where h is the
*octile distance* to the goal, h = max(dx, dy) + (sqrt(2) - 1) min(dx, dy): the exact cost on an
empty 8-connected grid. It never overestimates (admissible) and never drops by more than one
step's cost (consistent), so A* finds the same optimal cost as Dijkstra while expanding fewer
cells, because it heads towards the goal.

**RRT*** (Karaman & Frazzoli 2011) grows a tree of straight segments from the start by random
sampling. Each new sample is joined to the tree through the neighbour that gives it the cheapest
cost, and neighbours are *rewired* through it when that shortens their cost. With radius
r(n) = gamma * sqrt(log n / n) the tree's best route converges to the optimum as samples grow. It
does not need a grid, so its routes are made of long straight lines at any angle; here they are
then traced back onto grid cells. Randomised: the route differs from run to run.

**Ant Colony Optimisation** (Dorigo 1996) sends m ants from the start. At each cell an ant picks
the next cell j with probability p_j proportional to tau_j^alpha * eta_j^beta, where tau is the
pheromone on j and eta_j = 1 / (step cost + octile distance from j to the goal) is how promising j
looks. Ants that reach the goal lay pheromone Q / L on their route (shorter routes lay more),
pheromone evaporates by a factor (1 - rho) each round, and the best route so far is reinforced
(elitist ant system). After a few rounds the colony converges on a short route. Also randomised.

Every planner returns the route as a list of grid cells plus how much work it did, so the
dashboard can show how the algorithms differ.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from .mapping import FREE, MOVES, extract_path

PLANNERS = ("dijkstra", "astar", "rrtstar", "aco")
SQ2 = 1.4142                         # the same diagonal cost as mapping.MOVES, so A*'s heuristic stays admissible


@dataclass
class Plan:
    path: list                       # cells (x, y) from the first step to the goal (start not included)
    cost: float                      # route cost in cells (including extra costs)
    length: float                    # route length in cells (geometry only)
    expanded: int                    # cells expanded / samples drawn / ant steps: how much work was done
    ms: float                        # planning time
    algorithm: str
    note: str = ""                   # e.g. "no route found, fell back to A*"
    search: list = field(default_factory=list)   # for display: cells looked at, or tree / trail segments


def octile(ax, ay, bx, by) -> float:
    dx, dy = abs(ax - bx), abs(ay - by)
    return max(dx, dy) + (SQ2 - 1.0) * min(dx, dy)


def route_length(start, path) -> float:
    pts = [start, *path]
    return float(sum(SQ2 if (a[0] != b[0] and a[1] != b[1]) else 1.0 for a, b in zip(pts, pts[1:])))


def valid_route(free: np.ndarray, start, path) -> bool:
    """Every step goes to a neighbouring free cell and never cuts an obstacle's corner."""
    prev = start
    for c in path:
        dx, dy = c[0] - prev[0], c[1] - prev[1]
        if max(abs(dx), abs(dy)) != 1 or not free[c[1], c[0]]:
            return False
        if dx and dy and not (free[prev[1], c[0]] and free[c[1], prev[0]]):
            return False
        prev = c
    return True


# ------------------------------------------------------------------ Dijkstra and A*
def grid_search(free: np.ndarray, start, goal, extra: np.ndarray | None = None, avoid: set | None = None,
                heuristic: bool = True) -> tuple[list, float, int, list]:
    """Best-first search on the grid: A* with the octile heuristic, or Dijkstra without it.
    Returns (path, cost, expanded, closed cells)."""
    H, W = free.shape
    gx, gy = goal
    g = {start: 0.0}
    parent: dict = {}
    closed = set()
    order = []
    h0 = octile(*start, gx, gy) if heuristic else 0.0
    heap = [(h0, -0.0, start)]
    avoid = avoid or set()
    while heap:
        f, neg_g, cur = heapq.heappop(heap)
        gc = -neg_g
        if cur in closed:
            continue
        closed.add(cur)
        order.append(cur)
        if cur == goal:
            return extract_path(parent, start, goal), gc, len(closed), order
        x, y = cur
        for dx, dy, cost in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H) or not free[ny, nx] or (nx, ny) in avoid or (nx, ny) in closed:
                continue
            if dx and dy and not (free[y, nx] and free[ny, x]):
                continue
            ng = gc + cost + (0.0 if extra is None else float(extra[ny, nx]))
            if ng < g.get((nx, ny), np.inf):
                g[(nx, ny)] = ng
                parent[(nx, ny)] = cur
                h = octile(nx, ny, gx, gy) if heuristic else 0.0
                # ties on f: prefer the deeper node (larger g), which reaches the goal sooner
                heapq.heappush(heap, (ng + h, -ng, (nx, ny)))
    return [], np.inf, len(closed), order


# ------------------------------------------------------------------ RRT*
def _trace(free: np.ndarray, pts: list) -> list:
    """Turn a polyline (cell-centre coordinates) into a chain of 8-connected cells without corner cutting."""
    cells = []
    cur = (int(pts[0][0]), int(pts[0][1]))
    for a, b in zip(pts, pts[1:]):
        # walk along the segment in steps of 0.1 cell and record every cell the robot's centre enters
        n = max(1, int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / 0.1)))
        for k in range(1, n + 1):
            nx, ny = int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n)
            if (nx, ny) == cur:
                continue
            dx, dy = nx - cur[0], ny - cur[1]
            if dx and dy and not (free[cur[1], nx] and free[ny, cur[0]]):
                # the line crossed exactly at a corner: step through the free side cell first
                cells.append((nx, cur[1]) if free[cur[1], nx] else (cur[0], ny))
            cells.append((nx, ny))
            cur = (nx, ny)
    return cells


SUBCELL = 5                          # collision checks use a 5x finer grid of "safe" points


def safe_mask(free: np.ndarray) -> np.ndarray:
    """Fine grid (SUBCELL points per cell side): a point is safe when its cell is free and, if it lies
    within 0.3 cell of a border, the cell across that border is free too. A segment whose samples are
    all safe never squeezes diagonally between two obstacles."""
    H, W = free.shape
    k = SUBCELL
    f = (np.arange(k) + 0.5) / k                                    # position of the sub-points inside a cell
    pad = np.pad(free, 1, constant_values=False)
    safe = np.repeat(np.repeat(free, k, 0), k, 1).copy()
    for axis, (lo, hi) in enumerate([(f < 0.3, f > 0.7)] * 2):
        for side, sel in ((-1, lo), (1, hi)):
            nb = pad[1 + (side if axis == 0 else 0):H + 1 + (side if axis == 0 else 0),
                     1 + (side if axis == 1 else 0):W + 1 + (side if axis == 1 else 0)]
            nbf = np.repeat(np.repeat(nb, k, 0), k, 1)
            sub = np.tile(sel, H) if axis == 0 else np.tile(sel, W)
            if axis == 0:
                safe[sub, :] &= nbf[sub, :]
            else:
                safe[:, sub] &= nbf[:, sub]
    return safe


def _segment_free(safe: np.ndarray, a, b) -> bool:
    """Is the straight segment a-b (cell coordinates) safe? Sampled every 0.1 cell on the fine grid."""
    Hs, Ws = safe.shape
    d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    n = max(2, int(np.ceil(d / 0.1)) + 1)
    t = np.linspace(0.0, 1.0, n)
    sx = np.floor((a[0] + (b[0] - a[0]) * t) * SUBCELL).astype(int)
    sy = np.floor((a[1] + (b[1] - a[1]) * t) * SUBCELL).astype(int)
    if sx.min() < 0 or sy.min() < 0 or sx.max() >= Ws or sy.max() >= Hs:
        return False
    return bool(safe[sy, sx].all())


class _Tree:
    def __init__(self, root, size):
        self.p = np.zeros((size, 2))
        self.parent = np.full(size, -1)
        self.cost = np.zeros(size)
        self.p[0], self.n = root, 1

    def add(self, q, parent, cost):
        self.p[self.n], self.parent[self.n], self.cost[self.n] = q, parent, cost
        self.n += 1
        return self.n - 1

    def branch(self, i):
        out = []
        while i >= 0:
            out.append((self.p[i, 0], self.p[i, 1]))
            i = self.parent[i]
        return out                                                    # from node i back to the root


def _extend(tree: _Tree, safe, q, step, gamma):
    """RRT* extension of `tree` towards q: steer, choose the cheapest collision-free parent among the
    neighbours within r(n), add the node, rewire the neighbours through it. Returns the new index or -1."""
    n = tree.n
    d = np.hypot(tree.p[:n, 0] - q[0], tree.p[:n, 1] - q[1])
    i0 = int(np.argmin(d))
    if d[i0] > step:
        q = tree.p[i0] + (q - tree.p[i0]) * (step / d[i0])
        d = np.hypot(tree.p[:n, 0] - q[0], tree.p[:n, 1] - q[1])
    if d.min() < 1e-6:
        return -1
    radius = max(min(step * 1.5, gamma * np.sqrt(np.log(n + 1) / (n + 1))), float(d.min()) + 1e-9)
    near = np.flatnonzero(d <= radius)
    best = -1
    for i in near[np.argsort(tree.cost[near] + d[near])]:            # cheapest parent first
        if _segment_free(safe, tree.p[i], q):
            best = int(i)
            break
    if best < 0:
        return -1
    new = tree.add(q, best, tree.cost[best] + d[best])
    for i in near:                                                      # rewire through the new node
        c = tree.cost[new] + d[i]
        if i != best and c < tree.cost[i] - 1e-9 and _segment_free(safe, q, tree.p[i]):
            tree.parent[i], tree.cost[i] = new, c
    return new


def rrt_star(free: np.ndarray, start, goal, rng: np.random.Generator, reach: np.ndarray | None = None,
             max_samples: int = 700, step: float = 5.0, extra_after_goal: int = 150) -> tuple[list, int, list]:
    """Bidirectional RRT*: one tree grows from the robot, one from the goal (the RRT-Connect idea,
    Kuffner & LaValle 2000, with RRT* parent choice and rewiring). After each extension the other tree
    tries to reach the new node in straight steps; when they meet, a route exists, and the remaining
    samples keep improving it. Once a route of cost c is known, samples are drawn only where they can
    still improve it: points p with |p - start| + |p - goal| <= c, an ellipse around the two ends
    (Informed RRT*, Gammell et al. 2014). Returns (polyline, samples drawn, tree segments)."""
    safe = safe_mask(free)
    s = np.array([start[0] + 0.5, start[1] + 0.5])
    g = np.array([goal[0] + 0.5, goal[1] + 0.5])
    pool = np.argwhere(reach if reach is not None else free)[:, ::-1].astype(float)
    focal = np.hypot(pool[:, 0] + 0.5 - s[0], pool[:, 1] + 0.5 - s[1]) + np.hypot(pool[:, 0] + 0.5 - g[0], pool[:, 1] + 0.5 - g[1])
    informed = pool
    size = 2 * max_samples + 4
    trees = [_Tree(s, size), _Tree(g, size)]
    gamma = 2.0 * np.sqrt(len(pool) / np.pi) if len(pool) else 10.0
    best = None                                                           # (cost, node in tree 0, node in tree 1)
    drawn, since = 0, 0
    if _segment_free(safe, s, g):                                         # straight line of sight: done
        return [tuple(s), tuple(g)], 0, []
    while drawn < max_samples and (best is None or since < extra_after_goal):
        drawn += 1
        a, b = (0, 1) if drawn % 2 else (1, 0)                            # take turns
        q = informed[int(rng.integers(len(informed)))] + rng.random(2)
        na = _extend(trees[a], safe, q, step, gamma)
        if na < 0:
            continue
        qa = trees[a].p[na]
        # the other tree reaches towards the new node (connect)
        for _ in range(4):
            tb = trees[b]
            d = np.hypot(tb.p[:tb.n, 0] - qa[0], tb.p[:tb.n, 1] - qa[1])
            nb = int(np.argmin(d))
            if d[nb] <= step and _segment_free(safe, tb.p[nb], qa):
                c = trees[a].cost[na] + d[nb] + tb.cost[nb]
                ends = (na, nb) if a == 0 else (nb, na)
                if best is None or c < best[0]:
                    best = (c, *ends)
                    inside = pool[focal <= c + 1.0]                      # + 1 cell: sample whole cells around the ellipse
                    informed = inside if len(inside) else pool
                break
            if _extend(tb, safe, qa, step, gamma) < 0:
                break
        if best is not None:
            since += 1
    tree = [[float(t.p[i, 0]), float(t.p[i, 1]), float(t.p[t.parent[i], 0]), float(t.p[t.parent[i], 1])]
            for t in trees for i in range(1, t.n) if t.parent[i] >= 0]
    if best is None:
        return [], drawn, tree
    _, i0, i1 = best
    poly = trees[0].branch(i0)[::-1] + trees[1].branch(i1)          # start ... meeting point ... goal
    return _smooth(safe, poly), drawn, tree


def _smooth(safe: np.ndarray, poly: list) -> list:
    """Line-of-sight shortcutting, the usual last step after RRT: from each point jump to the
    furthest later point that can be reached in a straight, collision-free line."""
    out, i = [poly[0]], 0
    while i < len(poly) - 1:
        j = len(poly) - 1
        while j > i + 1 and not _segment_free(safe, np.array(poly[i]), np.array(poly[j])):
            j -= 1
        out.append(poly[j])
        i = j
    return out


# ------------------------------------------------------------------ Ant Colony Optimisation
def ant_colony(free: np.ndarray, start, goal, rng: np.random.Generator, extra: np.ndarray | None = None,
               ants: int = 10, rounds: int = 12, alpha: float = 1.0, beta: float = 4.0, rho: float = 0.3,
               q: float = 10.0, patience: int = 3) -> tuple[list, int, list]:
    """Elitist ant system for a shortest route on the grid. Returns (path, ant steps, pheromone cells)."""
    H, W = free.shape
    gx, gy = goal
    tau = np.ones((H, W))
    best, best_len, steps_total, calm = None, np.inf, 0, 0
    limit = int(4 * octile(*start, gx, gy) + 60)                               # an ant gives up after this many steps
    for _ in range(rounds):
        found = []
        for _ in range(ants):
            cur, walk, seen, L = start, [], {start}, 0.0
            for _ in range(limit):
                x, y = cur
                opts, w = [], []
                for dx, dy, c in MOVES:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < W and 0 <= ny < H) or not free[ny, nx] or (nx, ny) in seen:
                        continue
                    if dx and dy and not (free[y, nx] and free[ny, x]):
                        continue
                    step = c + (0.0 if extra is None else float(extra[ny, nx]))
                    eta = 1.0 / (step + octile(nx, ny, gx, gy) + 1e-6)          # how promising: short step, close to goal
                    opts.append(((nx, ny), step))
                    w.append(tau[ny, nx] ** alpha * eta ** beta)
                steps_total += 1
                if not opts:                                                     # dead end: step back and try elsewhere
                    if not walk:
                        break
                    last = walk.pop()
                    cur = walk[-1] if walk else start
                    L -= SQ2 if (last[0] != cur[0] and last[1] != cur[1]) else 1.0
                    if extra is not None:
                        L -= float(extra[last[1], last[0]])
                    continue
                w = np.asarray(w)
                k = int(rng.choice(len(opts), p=w / w.sum()))
                cur, step = opts[k]
                walk.append(cur)
                seen.add(cur)
                L += step
                if cur == goal:
                    found.append((walk, L))
                    break
        tau *= (1.0 - rho)                                                       # evaporation
        improved = False
        for walk, L in found:
            for x, y in walk:
                tau[y, x] += q / L
            if L < best_len - 1e-9:
                best, best_len, improved = walk, L, True
        calm = 0 if improved or best is None else calm + 1
        if calm >= patience:                                                     # converged: the best route stopped improving
            break
        if best is not None:                                                     # elitist reinforcement
            for x, y in best:
                tau[y, x] += q / best_len
    trail = np.argwhere(tau > 1.5)[:, ::-1]
    return (best or []), steps_total, [[int(x), int(y)] for x, y in trail[:600]]


def _shortcut(free: np.ndarray, start, path: list) -> list:
    """Remove loops from a route (an ant can wander): if the route comes back next to a cell it
    passed earlier, jump straight there."""
    pts = [start, *path]
    out, i = [start], 0
    index = {c: k for k, c in enumerate(pts)}          # last time each cell is visited
    while pts[i] != pts[-1]:
        i = index[pts[i]]
        cur = pts[i]
        j = i + 1
        for k in range(len(pts) - 1, i + 1, -1):        # furthest later cell that is a neighbour
            c = pts[k]
            dx, dy = c[0] - cur[0], c[1] - cur[1]
            if max(abs(dx), abs(dy)) == 1 and (not (dx and dy) or (free[cur[1], c[0]] and free[c[1], cur[0]])):
                j = k
                break
        out.append(pts[j])
        i = j
    return out[1:]


# ------------------------------------------------------------------ one entry point
def plan(algorithm: str, known: np.ndarray, start, goal, rng: np.random.Generator,
         extra: np.ndarray | None = None, avoid: set | None = None, reach: np.ndarray | None = None) -> Plan:
    """Plan a route from start to goal on a robot's map with the chosen algorithm."""
    t0 = time.perf_counter()
    free = known == FREE
    free = free.copy()
    free[start[1], start[0]] = True                      # the robot stands here
    if avoid:
        blocked_free = free.copy()
        for x, y in avoid:
            blocked_free[y, x] = False
        blocked_free[start[1], start[0]] = True
    else:
        blocked_free = free
    note, search = "", []
    if start == goal:
        return Plan([], 0.0, 0.0, 0, 0.0, algorithm)
    if algorithm in ("dijkstra", "astar"):
        path, cost, expanded, order = grid_search(blocked_free, start, goal, extra, None, heuristic=algorithm == "astar")
        search = [list(c) for c in order[:1500]]
    elif algorithm == "rrtstar":
        poly, expanded, tree = rrt_star(blocked_free, start, goal, rng, reach)
        search = tree[:500]
        path = _trace(blocked_free, poly) if poly else []
        if path and path[0] == start:
            path = path[1:]
        if path and path[-1] != goal:                    # the tree ended next to the goal cell: take the last step(s)
            tail, *_ = grid_search(blocked_free, path[-1], goal, extra, None, heuristic=True)
            path = path + tail
        if poly and not valid_route(blocked_free, start, path):
            path = []
        if not path:
            path, _, e2, _ = grid_search(blocked_free, start, goal, extra, None, heuristic=True)
            expanded += e2
            note = "RRT* found no route within its sample budget: fell back to A*"
    elif algorithm == "aco":
        walk, expanded, trail = ant_colony(blocked_free, start, goal, rng, extra)
        search = trail
        path = _shortcut(blocked_free, start, walk) if walk else []
        if not path:
            path, _, e2, _ = grid_search(blocked_free, start, goal, extra, None, heuristic=True)
            expanded += e2
            note = "no ant reached the goal: fell back to A*"
    else:
        raise ValueError(f"unknown planner {algorithm!r}; choose from {PLANNERS}")
    length = route_length(start, path) if path else np.inf
    cost = length + (0.0 if extra is None or not path else float(sum(extra[y, x] for x, y in path)))
    return Plan(path, cost, length, int(expanded), (time.perf_counter() - t0) * 1000.0, algorithm, note, search)
