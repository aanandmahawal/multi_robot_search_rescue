"""Multi-robot decision making: which robot explores / checks what.

Every robot sees two kinds of possible goals on its own map:

* **frontier goals**: go to the edge of the area the cameras have searched. Value = number of
  frontier cells, i.e. roughly how much new area it should reveal. Long frontiers are cut into
  pieces of about 10 cells so several robots can work along one big opening. (The LiDAR map
  usually reaches further than the searched area: robots plan their routes on the map, but only
  a camera can search a cell for people.)
* **verify goals**: go close to an uncertain victim sighting and take a better look.

The strategies differ only in how goals are chosen:

  random       pick any reachable goal at random                         (baseline)
  greedy       every robot takes its nearest goal, ignoring teammates     (Yamauchi 1998)
  partition    the building is split into one zone per robot; each robot
               only explores its own zone (heading towards it while it is
               still out of reach) and helps elsewhere once it is done    (area division)
  coordinated  sequential auction on utility = value - 0.5 x distance.
               When a robot wins a goal, goals near it lose value for the
               others, so the team spreads out; a sighting is only ever
               checked by one robot                                       (Burgard et al. 2005)

With a fixed coverage pattern (``cfg.coverage`` = boustrophedon or spiral, see ``coverage.py``)
every robot first drives its own pattern, waypoint by waypoint; it only leaves it to check a
sighting less than VERIFY_DETOUR metres away (one robot per sighting). Once its pattern is done,
it joins the strategy above to search whatever the patterns missed.

Every chosen goal carries a plain-language ``reason`` that the dashboard shows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .mapping import FREE, OBSTACLE, extract_path, frontiers
from .sensors import line_of_sight

if TYPE_CHECKING:
    from .simulator import RobotAgent, Simulator

STRATEGIES = ("random", "greedy", "partition", "coordinated")
DIST_COST = 0.5          # utility points per metre travelled
PIECE = 10               # frontier cells per goal piece
VERIFY_DETOUR = 6.0      # metres: a robot on a coverage pattern leaves it to check sightings this close


@dataclass
class Goal:
    kind: str                     # frontier | verify | sweep | home | report | transit | probe | backtrack
    cell: tuple[int, int]
    value: float
    distance: float               # metres from the robot along its path
    candidate: int | None = None  # registry id for verify goals
    look_at: tuple | None = None  # point to face on arrival
    reason: str = ""
    path: list | None = None      # precomputed path (transit goals plan through unexplored space)
    waypoint: int | None = None   # sweep goals: index of the waypoint in the robot's coverage pattern
    tag: str = ""                 # short label of why this goal won, drawn in the 3-D view (strategy layer)
    utility: float | None = None  # coordinated: the winning bid


def _pieces(cells: np.ndarray) -> list[np.ndarray]:
    """Cut a long frontier into chunks of ~PIECE cells along its main direction."""
    k = int(np.ceil(len(cells) / PIECE))
    if k <= 1:
        return [cells]
    centred = cells - cells.mean(axis=0)
    axis = np.linalg.svd(centred.astype(float), full_matrices=False)[2][0]
    order = np.argsort(centred @ axis)
    return [cells[idx] for idx in np.array_split(order, k)]


def robot_goals(sim: "Simulator", r: "RobotAgent", dist: np.ndarray) -> list[Goal]:
    """All goals robot ``r`` can reach according to its own knowledge."""
    cfg, world = sim.cfg, sim.world
    goals = []
    tabu = [c for c, until in getattr(r, "tabu", {}).items() if until > sim.t]
    for f in frontiers(sim.explore_view(r)):
        for cells in _pieces(f.cells):
            d_cells = dist[cells[:, 1], cells[:, 0]]
            ok = np.isfinite(d_cells)
            if not ok.any():
                continue
            # goal = the reachable cell closest to the piece's centre (the centre itself can be
            # unreachable, e.g. a diagonal strip seen through a doorway)
            centre = cells.mean(axis=0)
            cand = cells[ok]
            k = int(np.argmin(((cand - centre) ** 2).sum(axis=1)))
            cell = (int(cand[k][0]), int(cand[k][1]))
            if any(max(abs(cell[0] - t[0]), abs(cell[1] - t[1])) <= 1 for t in tabu):
                continue                               # a dead end I backed out of: not again for a while
            d = float(dist[cell[1], cell[0]]) * cfg.cell
            if not sim.affordable(r, cell, d):
                r.unaffordable += 1                    # not enough battery to get there and back
                continue
            goals.append(Goal("frontier", cell, float(len(cells)), d))
    free = r.known == FREE
    for c in r.registry.open_candidates():
        cx, cy = world.to_cell(c.x, c.y)
        rad = int(np.ceil(cfg.verify_distance / cfg.cell))
        best = None
        for y in range(max(0, cy - rad), min(world.H, cy + rad + 1)):
            for x in range(max(0, cx - rad), min(world.W, cx + rad + 1)):
                if not free[y, x] or not np.isfinite(dist[y, x]):
                    continue
                px, py = world.to_metres((x, y))
                e = np.hypot(px - c.x, py - c.y)
                if 0.7 <= e <= cfg.verify_distance - 0.3 and (best is None or dist[y, x] < best[0]):
                    # an inconclusive look is retried from a different side
                    if any(np.hypot(px - sx, py - sy) < 1.2 for sx, sy in c.tried_spots):
                        continue
                    if line_of_sight(world, (px, py), (c.x, c.y)):
                        best = (dist[y, x], (x, y))
        if best is not None and any(max(abs(best[1][0] - t[0]), abs(best[1][1] - t[1])) <= 1 for t in tabu):
            best = None
        if best is not None and not sim.affordable(r, best[1], float(best[0]) * cfg.cell):
            r.unaffordable += 1
            best = None
        if best is not None:
            # a sighting is worth more the more likely it is to be a real person
            belief = 1.0 / (1.0 + np.exp(-c.logodds)) if cfg.verify_by_belief else 1.0
            goals.append(Goal("verify", best[1], cfg.verify_value * belief, float(best[0]) * cfg.cell,
                              candidate=c.id, look_at=(c.x, c.y)))
    return goals


def _describe(g: Goal, why: str, sim=None) -> Goal:
    what = (f"check possible victim at ({g.look_at[0]:.1f}, {g.look_at[1]:.1f}) m" if g.kind == "verify"
            else f"explore frontier of {g.value:.0f} cells")
    g.reason = f"{what}, {g.distance:.1f} m away: {why}"
    return g


def next_waypoint(sim: "Simulator", r: "RobotAgent") -> Goal | None:
    """The next waypoint of r's coverage pattern that r can drive to. A waypoint r already stands on
    counts as reached; one inside a known obstacle, out of reach even through unmapped space, or at
    a dead end r backed out of is skipped (and remembered as skipped)."""
    sw, cfg = r.sweep, sim.cfg
    tabu = [c for c, until in r.tabu.items() if until > sim.t]
    while not sw.done:
        c = sw.cells[sw.index]
        x, y = c
        if c == r.cell:
            sw.arrive(len(r.track) - 1)
            continue
        why = None
        if r.known[y, x] == OBSTACLE or r.gaveup[y, x]:
            why = "inside an obstacle"
        elif not np.isfinite(r.opt_dist[y, x]):
            why = "cannot be reached"
        elif any(max(abs(x - t[0]), abs(y - t[1])) <= 1 for t in tabu):
            why = "a dead end I backed out of"
        if why is None:
            d = float(r.opt_dist[y, x]) * cfg.cell
            if not sim.affordable(r, c, d, work_s=0.0):
                r.unaffordable += 1                    # keep the waypoint: recharge first, then continue from here
                return None
            return Goal("sweep", c, 0.0, d, waypoint=sw.index, reason=f"following {sw.where()}, {d:.1f} m away",
                        tag=f"{'spiral' if sw.pattern == 'spiral' else 'lane'} wp {sw.index + 1}/{len(sw.cells)}")
        sw.skipped.append((sw.index, why))
        sw.index += 1
    return None


def _sweep(sim, deciding, options) -> dict[int, Goal]:
    """Robots still on their coverage pattern: the next waypoint, or a sighting very close by."""
    out: dict[int, Goal] = {}
    for r in deciding:
        if r.sweep is None or r.sweep.end_t is not None:
            continue
        # sightings a teammate is already going to check are left to it
        checking = [o.goal.look_at for o in sim.teammates(r) if o.goal is not None and o.goal.kind == "verify"]
        checking += [g.look_at for g in out.values() if g.kind == "verify"]
        near = [g for g in options[r.id] if g.kind == "verify" and g.distance <= VERIFY_DETOUR
                and all(np.hypot(g.look_at[0] - a[0], g.look_at[1] - a[1]) >= 1.3 for a in checking)]
        if near:
            g = min(near, key=lambda g: g.distance)
            g.tag = "check sighting"
            out[r.id] = _describe(g, f"a sighting within {VERIFY_DETOUR:g} m of my {r.sweep.pattern_name}: "
                                     f"checking it, then continuing with {r.sweep.where()}")
            continue
        g = None if r.sweep.done else next_waypoint(sim, r)
        if g is not None:
            out[r.id] = g
        elif not r.sweep.done:                         # the battery cannot pay for the next waypoint and the way back
            g = sim._battery_blocked(r)
            if g is not None:
                out[r.id] = g                          # recharge, then continue the pattern where I left it
            else:                                      # no recharging: the rest of my pattern is out of reach
                r.sweep.end_move, r.sweep.end_t = len(r.track) - 1, sim.t
                sim._log(f"R{r.id} stops its {r.sweep.pattern_name} at {r.sweep.where()}: {r.finished_reason}.")
        else:
            r.sweep.end_move, r.sweep.end_t = len(r.track) - 1, sim.t
            sim._log(f"R{r.id} has finished its {r.sweep.pattern_name} ({r.sweep.reached} waypoints reached, "
                     f"{len(r.sweep.skipped)} skipped): now searching what the patterns missed "
                     f"({sim.cfg.strategy} strategy).")
    return out


def choose(sim: "Simulator", deciding: list["RobotAgent"]) -> dict[int, Goal | None]:
    options = {r.id: robot_goals(sim, r, r.dist) for r in deciding}
    out: dict[int, Goal | None] = {}
    if sim.cfg.coverage != "frontier":
        out.update(_sweep(sim, deciding, options))
        deciding = [r for r in deciding if r.id not in out]
        if not deciding:
            return out
    out.update(_choose_frontier(sim, deciding, options))
    return out


def _choose_frontier(sim: "Simulator", deciding: list["RobotAgent"], options: dict) -> dict[int, Goal | None]:
    """Frontier-based exploration: the team strategy picks among the frontier and verify goals."""
    strategy = sim.cfg.strategy
    out: dict[int, Goal | None] = {}
    if strategy == "random":
        for r in deciding:
            opts = options[r.id]
            g = opts[int(sim.rng.integers(len(opts)))] if opts else None
            if g is not None:
                g.tag = f"random (1 of {len(opts)})"
            out[r.id] = _describe(g, f"picked at random from {len(opts)} goals", sim) if g else None
        return out
    if strategy == "greedy":
        for r in deciding:
            g = min(options[r.id], key=lambda g: (g.distance + sim.tie_break(), -g.value), default=None)
            if g is not None:
                g.tag = f"nearest · {g.distance:.1f} m"
            out[r.id] = _describe(g, "nearest goal (teammates ignored)", sim) if g else None
        return out
    if strategy == "partition":
        return _partition(sim, deciding, options)
    if strategy == "coordinated":
        return _auction(sim, deciding, options)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")


def _partition(sim, deciding, options) -> dict[int, Goal | None]:
    """Strict area division: a robot only ever explores (and checks sightings) inside its own
    zone. While its zone is out of reach on its map it heads straight for it, planning through
    unexplored space; when its zone is fully searched it goes back to base."""
    from .mapping import UNKNOWN, dijkstra, extract_path
    out = {}
    for r in deciding:
        zone = r.region
        mine = [g for g in options[r.id] if sim.region[g.cell[1], g.cell[0]] == zone]
        if mine:
            g = min(mine, key=lambda g: (g.distance, -g.value))
            g.tag = f"zone Z{zone} · {g.distance:.1f} m"
            out[r.id] = _describe(g, f"nearest goal inside my zone Z{zone}", sim)
            continue
        # nothing reachable in my zone on my map: is there still unsearched space in it?
        dist, parent = dijkstra(r.known, r.cell, optimistic=True)
        todo = (sim.region == zone) & (sim.explore_view(r) == UNKNOWN) & np.isfinite(dist)
        if not todo.any():
            out[r.id] = None           # zone finished (or sealed off): partition robots do not help elsewhere
            r.finished_reason = f"my zone Z{zone} is fully searched (partition robots stay out of other zones)"
            continue
        ys, xs = np.nonzero(todo)
        k = int(np.argmin(dist[ys, xs]))
        cell = (int(xs[k]), int(ys[k]))
        g = Goal("transit", cell, 0.0, float(dist[ys[k], xs[k]]) * sim.cfg.cell,
                 reason=f"driving to my zone Z{zone} ({float(dist[ys[k], xs[k]]) * sim.cfg.cell:.1f} m), "
                        f"only passing through other zones", tag=f"to zone Z{zone}")
        g.path = extract_path(parent, r.cell, cell)
        out[r.id] = g
    return out


def _auction(sim, deciding, options) -> dict[int, Goal | None]:
    cfg = sim.cfg
    radius = cfg.utility_discount_radius
    # goals already held by teammates I can talk to
    claimed = [(o.goal, o) for r in deciding for o in sim.teammates(r)
               if o.goal is not None and o not in deciding and o.goal.kind in ("frontier", "verify")]

    def discounted(g: Goal, claims) -> tuple[float, str]:
        if g.kind == "verify":
            taken = any(c.kind == "verify" and np.hypot(c.look_at[0] - g.look_at[0], c.look_at[1] - g.look_at[1]) < 1.3
                        for c, _ in claims)
            return (-np.inf, "") if taken else (g.value, "")
        v, note = g.value, ""
        for c, o in claims:
            if c.kind != "frontier":
                continue
            d = np.hypot(c.cell[0] - g.cell[0], c.cell[1] - g.cell[1]) * cfg.cell
            if d < 2.0:
                return -np.inf, ""                         # never send two robots to (almost) the same spot
            if d < radius:
                v *= d / radius                            # a teammate will already see most of this area
                note = f" (value cut from {g.value:.0f} because R{o.id} is heading nearby)"
        return v, note

    result: dict[int, Goal | None] = {}
    pending = list(deciding)
    while pending:
        best = None
        for r in pending:
            for g in options[r.id]:
                v, note = discounted(g, claimed)
                u = v - DIST_COST * g.distance + sim.tie_break()      # tie-break: 0 unless runs are set to vary
                if best is None or u > best[0]:
                    best = (u, r, g, v, note)
        if best is None or best[0] == -np.inf:
            # more robots than distinct goals: spread out as far as possible from claimed goals
            for r in pending:
                fr = [g for g in options[r.id] if g.kind == "frontier"]
                if not fr:
                    result[r.id] = None
                    continue

                def spread(g):
                    near = min((np.hypot(c.cell[0] - g.cell[0], c.cell[1] - g.cell[1]) for c, _ in claimed), default=99)
                    return near * cfg.cell - 0.2 * g.distance
                g = max(fr, key=spread)
                g.tag = "spread out"
                result[r.id] = _describe(g, "every goal is taken, so going where I overlap least with teammates", sim)
                claimed.append((g, r))
            break
        u, r, g, v, note = best
        g.tag, g.utility = f"bid {u:.1f} = {v:.1f} − 0.5×{g.distance:.1f}", float(u)
        result[r.id] = _describe(g, f"won the auction, utility {u:.1f} = value {v:.1f} - 0.5 x {g.distance:.1f} m{note}", sim)
        claimed.append((g, r))
        pending.remove(r)
    return result


def partition_regions(world, n: int, rng) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Split the building footprint into n compact zones (k-means on cell coordinates).

    Only the outline of the building is used (rescuers know how big it is), never the
    hidden layout inside."""
    ys, xs = np.mgrid[1:world.H - 1, 4:world.W - 1]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    centres = pts[order[np.linspace(0, len(pts) - 1, n).astype(int)]] + rng.normal(0, 0.1, (n, 2))
    for _ in range(30):
        lab = np.argmin(((pts[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        centres = np.array([pts[lab == k].mean(axis=0) if (lab == k).any() else centres[k] for k in range(n)])
    gy, gx = np.mgrid[0:world.H, 0:world.W]
    all_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
    region = np.argmin(((all_pts[:, None, :] - centres[None]) ** 2).sum(-1), axis=1).reshape(world.H, world.W)
    return region, [tuple(c) for c in centres]


def path_to(r: "RobotAgent", goal: Goal) -> list:
    if goal.path is not None:
        return list(goal.path)
    return extract_path(r.parent, r.cell, goal.cell)
