"""Tests for route planning, coverage patterns, motion (kinematics, energy, DWA) and dead-end recovery."""
from dataclasses import replace

import numpy as np
import pytest

from rescue.config import RescueConfig
from rescue.coverage import building_box, grid_shape, lane_width, lanes, lawnmower, regions, spiral
from rescue.mapping import FREE, OBSTACLE, dijkstra
from rescue.metrics import summarize
from rescue.motion import (dwa_step, clearance_map, energy_home, move_time, rejoin, stuck_in_loop, turn_energy,
                           TRAIL_LEN)
from rescue.planners import PLANNERS, bresenham, octile, plan, route_length, valid_route
from rescue.simulator import Simulator
from rescue.world import generate

CFG = RescueConfig(vision="ideal", building="office", seed=3)


@pytest.fixture(scope="module")
def truth_map():
    w = generate(CFG)
    known = np.where(w.blocked, OBSTACLE, FREE).astype(np.int8)
    return w, known


def _pairs(w, n, seed=0):
    cells = np.argwhere(w.reachable)[:, ::-1]
    rng = np.random.default_rng(seed)
    return [(tuple(map(int, cells[rng.integers(len(cells))])), tuple(map(int, cells[rng.integers(len(cells))])))
            for _ in range(n)]


# ------------------------------------------------------------------ planners
def test_astar_is_optimal_and_expands_less(truth_map):
    w, known = truth_map
    for s, g in _pairs(w, 12):
        d = plan("dijkstra", known, s, g, np.random.default_rng(0))
        a = plan("astar", known, s, g, np.random.default_rng(0))
        dist, _ = dijkstra(known, s)
        assert d.length == pytest.approx(dist[g[1], g[0]], abs=1e-6)          # Dijkstra: the true shortest route
        assert a.length == pytest.approx(d.length, abs=1e-6)                   # A*: the same length...
        assert a.expanded <= d.expanded                                         # ...looking at fewer cells


def test_octile_heuristic_is_admissible(truth_map):
    w, known = truth_map
    s = w.starts[0]
    dist, _ = dijkstra(known, s)
    ys, xs = np.nonzero(np.isfinite(dist))
    h = np.array([octile(x, y, *s) for x, y in zip(xs, ys)])
    assert (h <= dist[ys, xs] + 1e-9).all()                                    # never overestimates


@pytest.mark.parametrize("algorithm", PLANNERS)
def test_every_planner_returns_a_valid_route(truth_map, algorithm):
    w, known = truth_map
    for s, g in _pairs(w, 5, seed=4):
        p = plan(algorithm, known, s, g, np.random.default_rng(1), reach=w.reachable)
        assert p.path and p.path[-1] == g
        assert valid_route(known == FREE, s, p.path)                           # free cells, no corner cutting
        opt = plan("astar", known, s, g, np.random.default_rng(0)).length
        assert p.length >= opt - 1e-6                                          # nothing beats the optimum


def test_planners_respect_extra_cost():
    known = np.full((9, 9), FREE, np.int8)
    extra = np.zeros((9, 9))
    extra[1:8, 4] = 50.0                                                        # an expensive column in the middle
    p = plan("astar", known, (0, 4), (8, 4), np.random.default_rng(0), extra=extra)
    assert all(c[0] != 4 or c[1] in (0, 8) for c in p.path)                     # goes round the ends of it


# ------------------------------------------------------------------ coverage patterns
def test_lane_spacing_follows_the_camera_footprint():
    cfg = RescueConfig(camera_range=5.0, camera_fov=90.0)
    assert lane_width(cfg) == pytest.approx(2 * 5.0 * np.sin(np.pi / 4) * 0.75)    # 2 R sin(F/2) (1 - overlap)
    assert lane_width(replace(cfg, vision="none", lidar_range=8.0)) == pytest.approx(2 * 8.0 * 0.75)   # LiDAR only


def test_lanes_are_at_most_w_apart_and_half_w_from_the_edges():
    for D, w in ((19.0, 5.3), (10.0, 5.0), (3.0, 5.3), (25.5, 2.0)):
        ys = lanes(0.0, D, w)
        assert len(ys) == int(np.ceil(D / w - 1e-9))
        assert ys[0] <= w / 2 + 1e-9 and D - ys[-1] <= w / 2 + 1e-9
        assert all(0 < b - a <= w + 1e-9 for a, b in zip(ys, ys[1:]))


def test_lawnmower_runs_along_the_longer_side_in_alternate_directions():
    pts, what = lawnmower((0.0, 0.0, 20.0, 10.0), 4.0)                          # wide: horizontal lanes
    assert what.startswith("3 horizontal")
    for k in range(3):
        (xa, ya), (xb, yb) = pts[2 * k], pts[2 * k + 1]
        assert ya == yb                                                         # one lane = one straight line
        assert (xb > xa) == (k % 2 == 0)                                        # even lanes one way, odd the other
    pts, what = lawnmower((0.0, 0.0, 6.0, 20.0), 4.0)                           # tall: vertical lanes
    assert what.startswith("2 vertical") and pts[0][0] == pts[1][0]


def test_spiral_goes_round_from_the_outside_in():
    box = (0.0, 0.0, 20.0, 20.0)
    pts, _ = spiral(box, 4.0)
    xs, ys = lanes(0, 20, 4.0), lanes(0, 20, 4.0)
    assert pts[:5] == [(xs[0], ys[0]), (xs[-1], ys[0]), (xs[-1], ys[-1]), (xs[0], ys[-1]), (xs[0], ys[1])]
    edge = [min(x, 20 - x, y, 20 - y) for x, y in pts]                          # distance to the region's edge...
    assert all(b >= a - 1e-9 for a, b in zip(edge, edge[1:]))                   # ...never decreases: outside in
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert x0 == x1 or y0 == y1                                             # only straight legs
    turns = [np.sign((pts[i][0] - pts[i - 1][0]) * (pts[i + 1][1] - pts[i][1])
                     - (pts[i][1] - pts[i - 1][1]) * (pts[i + 1][0] - pts[i][0])) for i in range(1, len(pts) - 1)]
    assert all(t > 0 for t in turns)                                            # always turning the same way (clockwise)


def test_regions_split_the_building_into_equal_squarish_rectangles():
    w = generate(replace(CFG, building="plain"))
    for n in (1, 2, 3, 4, 6):
        boxes = regions(w, n)
        x0, y0, x1, y1 = building_box(w)
        area = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        assert len(boxes) == n and sum(area) == pytest.approx((x1 - x0) * (y1 - y0))
        assert max(area) == pytest.approx(min(area))
    assert grid_shape(4, 25.5, 19.0) == (2, 2) and grid_shape(2, 25.5, 19.0) == (1, 2)


@pytest.mark.parametrize("pattern", ["boustrophedon", "spiral"])
def test_robots_drive_their_pattern_on_the_test_ground(pattern):
    cfg = replace(CFG, building="plain", coverage=pattern, n_robots=2, max_steps=1500)
    sim = Simulator(cfg)
    while not sim.finished:
        sim.step()
    c = cfg.cell
    for r in sim.robots:
        sw = r.sweep
        assert sw.done and sw.end_t is not None
        assert sw.reached + len(sw.skipped) == len(sw.cells)
        assert all(why == "inside an obstacle" for _, why in sw.skipped)       # nothing else is ever skipped here
        assert len(sw.skipped) <= 5
        assert all(sim.world.blocked[sw.cells[i][1], sw.cells[i][0]] for i, _ in sw.skipped)   # really on a block or a body
        # every waypoint reached was really driven through
        driven = set(r.track)
        skipped = {i for i, _ in sw.skipped}
        assert all(cell in driven for i, cell in enumerate(sw.cells) if i not in skipped)
        # the track keeps to the pattern: mostly within one cell of it
        pts = np.array([[(x + 0.5) * c, (y + 0.5) * c] for x, y in r.track[sw.start_move:sw.end_move + 1]])
        dev = np.array([min(_seg_dist(p, a, b) for a, b in zip(sw.corners, sw.corners[1:])) for p in pts])
        assert np.median(dev) <= 0.36 and np.mean(dev <= 1.0) > 0.85
    assert sim.coverage() > 0.99 and all(r.state == "done" for r in sim.robots)


def _seg_dist(p, a, b):
    a, b = np.asarray(a), np.asarray(b)
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    return float(np.hypot(*(a + t * ab - p)))


def test_bresenham_lines_have_octile_length():
    for b in ((10, 3), (-7, 7), (0, -5), (4, -9)):
        cells = bresenham((0, 0), b)
        assert cells[-1] == b and len(cells) == max(abs(b[0]), abs(b[1]))
        assert route_length((0, 0), cells) == pytest.approx(octile(0, 0, *b))


def test_planners_are_near_optimal_on_open_ground():
    w = generate(replace(CFG, building="plain"))
    known = np.where(w.blocked, OBSTACLE, FREE).astype(np.int8)
    for algorithm in ("rrtstar", "aco"):
        ratios = []
        for s, g in _pairs(w, 8, seed=2):
            if s == g:
                continue
            opt = plan("astar", known, s, g, np.random.default_rng(0)).length
            ratios.append(plan(algorithm, known, s, g, np.random.default_rng(3), reach=w.reachable).length / opt)
        assert np.mean(ratios) < 1.06, (algorithm, ratios)


# ------------------------------------------------------------------ motion
def test_kinematics_and_energy():
    cfg = RescueConfig(speed=0.5, turn_rate=180.0)
    assert move_time(cfg, 0.5, 0.0) == pytest.approx(1.0)
    assert move_time(cfg, 0.5 * 2 ** 0.5, 0.0) == pytest.approx(2 ** 0.5)     # a diagonal takes sqrt(2) as long
    assert move_time(cfg, 0.5, np.pi / 2) == pytest.approx(1.5)               # a 90 degree turn costs 0.5 s
    assert turn_energy(np.pi) == pytest.approx(2 * turn_energy(np.pi / 2))
    assert energy_home(cfg, 20) > energy_home(cfg, 10) > 0


def test_dwa_follows_the_route_in_the_open():
    known = np.full((7, 7), FREE, np.int8)
    path = [(4, 3), (5, 3), (6, 3)]
    cell, _ = dwa_step((3, 3), 0.0, path, known, [], clearance_map(known, []))
    assert cell == (4, 3)


def test_dwa_steps_around_a_teammate_and_never_into_obstacles():
    known = np.full((7, 7), FREE, np.int8)
    known[1, :] = OBSTACLE
    path = [(4, 3), (5, 3), (6, 3)]
    others = [(4, 3)]                                                           # a teammate on my route
    cell, rows = dwa_step((3, 3), 0.0, path, known, others, clearance_map(known, others))
    assert cell is not None and cell != (4, 3) and known[cell[1], cell[0]] == FREE
    assert all(known[c[1], c[0]] == FREE and c not in others for _, c, *__ in rows)
    assert rejoin(path, cell) != [] or cell not in path                         # it can continue its route from there


def test_loop_detection():
    trail = [(1, 1), (2, 1)] * (TRAIL_LEN // 2)
    assert stuck_in_loop(trail)
    assert not stuck_in_loop([(i, 1) for i in range(TRAIL_LEN)])


# ------------------------------------------------------------------ in the simulator
def test_robots_never_drive_faster_than_their_speed():
    for speed in (0.3, 0.5, 1.0):
        sim = Simulator(replace(CFG, speed=speed))
        for _ in range(60):
            sim.step()
        assert max(r.distance for r in sim.robots) <= speed * sim.t + 0.71
        assert max(r.distance for r in sim.robots) >= 0.5 * speed * sim.t      # and do drive


def test_same_settings_same_mission_unless_varied():
    def cells(cfg):
        sim = Simulator(cfg)
        for _ in range(60):
            sim.step()
        return [r.cell for r in sim.robots], sim
    a, _ = cells(CFG)
    b, _ = cells(CFG)
    c, sc = cells(replace(CFG, run_seed=11))
    d, sd = cells(replace(CFG, run_seed=12))
    assert a == b
    assert c != d and np.array_equal(sc.world.blocked, sd.world.blocked)       # same building, different paths


def test_low_battery_robots_come_home_before_it_is_empty():
    sim = Simulator(replace(CFG, battery_wh=3.0))
    sim.run()
    m = summarize(sim)
    assert m["batteries_emptied"] == 0
    assert all(r.cell in sim.world.starts for r in sim.robots)
    assert all(r.energy_j <= 3.0 * 3600 for r in sim.robots)
    # it turns home in time: either the energy check refuses the next task, or the battery runs low on the way
    assert any("returns to base" in e and "battery" in e for _, e in sim.events)


def test_dead_end_recovery_backs_off_and_avoids_the_goal():
    sim = Simulator(CFG)
    for _ in range(40):
        sim.step()
    r = next(r for r in sim.robots if r.goal is not None and r.goal.kind == "frontier" and len(r.history) >= 6)
    goal = r.goal.cell
    a, b = r.cell, r.history[-2]
    r.trail.extend([a, b] * TRAIL_LEN)                                          # pretend it has been going back and forth
    sim._recover(r)
    assert r.goal.kind == "backtrack" and r.state == "backtracking"
    assert max(abs(r.goal.cell[0] - a[0]), abs(r.goal.cell[1] - a[1])) >= 4
    assert r.tabu[goal] > sim.t
    from rescue.coordination import robot_goals
    r.dist, _ = dijkstra(r.known, r.cell)
    assert all(max(abs(g.cell[0] - goal[0]), abs(g.cell[1] - goal[1])) > 1 for g in robot_goals(sim, r, r.dist))


@pytest.mark.parametrize("kw", [{"planner": "astar"}, {"avoidance": "dwa"}, {"coverage": "boustrophedon"},
                                {"coverage": "spiral"}, {"revisit_penalty": 0.3}])
def test_every_navigation_option_searches_the_whole_building(kw):
    sim = Simulator(replace(CFG, **kw)).run()
    assert sim.coverage() > 0.99
    assert all(r.state == "done" for r in sim.robots)


def test_rrtstar_tree_costs_stay_consistent_after_rewiring():
    from rescue.planners import _Tree, _extend, safe_mask
    free = np.ones((30, 30), bool)
    free[10:20, 14] = False
    safe, rng = safe_mask(free), np.random.default_rng(5)
    tree = _Tree(np.array([2.5, 2.5]), 700)
    for _ in range(600):
        _extend(tree, safe, rng.random(2) * 30, 5.0, 2.0 * np.sqrt(free.sum() / np.pi))
    for i in range(1, tree.n):
        p = tree.parent[i]
        assert tree.cost[i] == pytest.approx(tree.cost[p] + np.hypot(*(tree.p[i] - tree.p[p])))


# ------------------------------------------------------------------ batteries
def test_each_robot_can_have_its_own_battery():
    sim = Simulator(replace(CFG, n_robots=3, battery_wh=5.0, battery_each=(1.0, 0.0, 3.0)))
    assert [sim.capacity(r) for r in sim.robots] == [3600.0, np.inf, 3.0 * 3600]
    assert sim.charge(sim.robots[1]) is None and sim.charge(sim.robots[0]) == 1.0


def test_robots_recharge_and_continue_their_pattern():
    sim = Simulator(replace(CFG, building="plain", coverage="boustrophedon", n_robots=1, battery_wh=2.0, max_steps=3000)).run()
    r = sim.robots[0]
    assert r.charges >= 1 and not r.depleted                                    # it went home to charge, never ran flat
    assert r.sweep.done and r.sweep.reached + len(r.sweep.skipped) == len(r.sweep.cells)   # and finished its pattern
    assert all(why == "inside an obstacle" for _, why in r.sweep.skipped)        # no waypoint dropped for lack of battery
    assert sim.coverage() > 0.99 and r.state == "done" and sim.t < 3000
    assert r.energy_used > 2.0 * 3600                                           # it used more than one battery's worth


def test_no_robot_takes_a_task_its_battery_cannot_pay_for():
    for each in ((1.0, 1.0), (1.0, 2.0)):
        sim = Simulator(replace(CFG, building="plain", n_robots=2, battery_each=each, recharge=False)).run()
        assert all(not r.depleted and r.cell in sim.world.starts for r in sim.robots)
        assert all(r.charges == 0 for r in sim.robots)


def test_a_battery_too_small_for_any_task_ends_the_mission_cleanly():
    sim = Simulator(replace(CFG, building="plain", n_robots=2, battery_wh=0.4, max_steps=2000)).run()
    assert sim.t < 400 and all(r.state == "done" and not r.depleted for r in sim.robots)
    assert all(r.charges <= 2 for r in sim.robots)                               # no endless docking
    assert any("cannot reach" in e for _, e in sim.events) or any("even a full" in r.finished_reason for r in sim.robots)


def test_run_until_empty_never_turns_home_and_stops_where_the_battery_dies():
    sim = Simulator(replace(CFG, building="plain", n_robots=2, battery_wh=1.0, drain=True)).run()
    assert all(r.depleted and r.state == "done" for r in sim.robots)          # both ran flat...
    assert all(r.cell not in sim.world.starts for r in sim.robots)           # ...out in the hall, not at the base
    assert all(r.charges == 0 and not r.low_battery for r in sim.robots)
    assert not any("returns to base" in e for _, e in sim.events)


# ------------------------------------------------------------------ open test ground
def test_your_own_obstacles_on_the_test_ground():
    from rescue.world import generate
    own = (("cabinet", 20, 10, 2, 2), ("crate", 30, 25, 1, 1), ("shelf", 4, 18, 1, 4))   # the shelf would block the entrance
    w = generate(replace(CFG, building="plain", obstacles=own))
    assert [(f.kind, f.x0, f.y0) for f in w.furniture] == [("cabinet", 20, 10), ("crate", 30, 25)]
    assert w.blocked[25, 30] and w.nav_height[25, 30] < 0.4                  # a low crate: under the LiDAR's scan plane
    assert not generate(replace(CFG, building="plain", obstacles=())).furniture
    assert all(v.cover == "none" for v in w.victims)                          # the test ground is undamaged


def test_editing_the_test_ground_does_not_move_the_victims():
    from rescue.world import generate
    plain = replace(CFG, building="plain")
    w = generate(plain)
    layout = tuple((f.kind, f.x0, f.y0, f.w, f.h) for f in w.furniture)                    # the default layout, as the editor sends it
    before = [(v.x, v.y) for v in w.victims]
    for extra in [("crate", 30, 30, 1, 1), ("cabinet", 40, 10, 2, 2)]:
        if any(extra[1] <= x / CFG.cell < extra[1] + extra[3] + 1 and extra[2] <= y / CFG.cell < extra[2] + extra[4] + 1 for x, y in before):
            continue                                                          # a block on a victim does move that victim
        assert [(v.x, v.y) for v in generate(replace(plain, obstacles=layout + (extra,))).victims] == before
