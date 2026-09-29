"""Tests for route planning, coverage patterns, motion (kinematics, energy, DWA) and dead-end recovery."""
from dataclasses import replace

import numpy as np
import pytest

from rescue.config import RescueConfig
from rescue.coverage import order_scores, pattern_key
from rescue.mapping import FREE, OBSTACLE, dijkstra
from rescue.metrics import summarize
from rescue.motion import (dwa_step, clearance_map, energy_home, move_time, rejoin, stuck_in_loop, turn_energy,
                           TRAIL_LEN)
from rescue.planners import PLANNERS, octile, plan, valid_route
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
def test_boustrophedon_sweeps_lanes_in_alternate_directions(truth_map):
    w, _ = truth_map
    lane = int(round(CFG.camera_range / CFG.cell))
    a, b = (10, 1), (40, 1)                                                     # lane 0: left to right
    c, d = (10, lane + 1), (40, lane + 1)                                       # lane 1: right to left
    k = lambda cell: pattern_key("boustrophedon", w, cell)
    assert k(a) < k(b) < k(d) < k(c)


def test_spiral_goes_outside_in(truth_map):
    w, _ = truth_map
    outer, inner = (5, 2), (w.W // 2, w.H // 2)
    assert pattern_key("spiral", w, outer) < pattern_key("spiral", w, inner)
    s = order_scores("spiral", w, [inner, outer])
    assert s == [1.0, 0.0]
    assert order_scores("frontier", w, [inner, outer]) is None


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
    assert any("battery low" in e for _, e in sim.events)


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
