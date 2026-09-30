from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rescue import lidar
from rescue.config import RescueConfig
from rescue.coordination import Goal, _auction
from rescue.mapping import (FREE, OBSTACLE, UNKNOWN, apply_camera, apply_lidar, classify, dijkstra, explore_map,
                            extract_path, frontiers, merge_evidence)
from rescue.perception import CNNDetector, frame_to_label, model_path
from rescue.sensors import line_of_sight, render, visible_cells
from rescue.simulator import Simulator
from rescue.thermal import HUMAN_MAX, THERMAL_BASE, WARM_OBJECTS
from rescue.victims import SAME_PERSON, Registry, Report, distinct
from rescue.world import DECOY_BASE, VICTIM_BASE, generate

CFG = RescueConfig(vision="ideal")
BURIED_CFG = replace(CFG, building="apartments", damage="severe", seed=5)   # has victims under thin and thick cover


@pytest.fixture(scope="module")
def world():
    return generate(CFG)


@pytest.fixture(scope="module")
def buried_world():
    return generate(BURIED_CFG)


def _view_of(w, target, lo=1.5, hi=3.0):
    """A camera frame looking at a point from 1.5-3 m with a clear line of sight."""
    for cy, cx in np.argwhere(w.reachable):
        px, py = w.to_metres((cx, cy))
        if lo < np.hypot(px - target[0], py - target[1]) < hi and line_of_sight(w, (px, py), target):
            return render(w, px, py, np.arctan2(target[1] - py, target[0] - px), np.random.default_rng(1))
    pytest.skip("no viewpoint")


# ------------------------------------------------------------------- world
def test_world_has_reachable_victims(world):
    assert len(world.victims) == CFG.n_victims
    for v in world.victims:
        assert all(world.blocked[y, x] for x, y in v.nav_cells)          # people are obstacles
        near = [(x + dx, y + dy) for x, y in v.nav_cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        assert any(world.in_bounds(c) and world.reachable[c[1], c[0]] for c in near)
    for c in world.starts:
        assert world.reachable[c[1], c[0]]


def test_generation_is_deterministic():
    a, b = generate(CFG), generate(CFG)
    assert np.array_equal(a.blocked, b.blocked)
    assert [(v.x, v.y) for v in a.victims] == [(v.x, v.y) for v in b.victims]
    assert np.array_equal(a.temp, b.temp) and [d.kind for d in a.thermal_decoys] == [d.kind for d in b.thermal_decoys]


# ----------------------------------------------------------------- thermal
def test_people_are_warm_and_rubble_hides_the_heat(buried_world):
    w = buried_world
    kinds = {v.cover for v in w.victims}
    assert {"thin", "thick"} <= kinds
    for v in w.victims:
        temps = np.array([w.temp[fy, fx] for fx, fy in v.fine_cells])
        if v.cover == "none":
            assert temps.max() > w.ambient + 8            # exposed skin
        elif v.cover == "thin":
            assert w.ambient + 1.5 < temps.max() < w.ambient + 9   # faint signature through light debris
            assert not v.detectable is False              # still detectable in principle
        elif v.cover == "thick":
            assert temps.max() < w.ambient + 1.5          # nothing reaches the surface
            assert not v.detectable
            assert all(w.cover[fy, fx] >= 0.4 for fx, fy in v.fine_cells)


def test_thermal_camera_sees_a_warm_body_in_the_dark(buried_world):
    w = buried_world
    v = next(v for v in w.victims if v.cover == "none")
    fr = _view_of(w, (v.x, v.y))
    on = (fr.obj >= VICTIM_BASE) & (fr.obj < DECOY_BASE)
    assert fr.thermal.shape == fr.depth.shape and on.sum() > 5
    assert fr.thermal[on].max() > np.median(fr.thermal[~on]) + 6
    assert frame_to_label(fr, "thermal").sum() > 0 and frame_to_label(fr, "cnn").sum() > 0


def test_thick_rubble_has_no_signature_and_is_never_labelled(buried_world):
    w = buried_world
    v = next(v for v in w.victims if v.cover == "thick")
    fr = _view_of(w, (v.x, v.y), 1.2, 3.5)
    on = (fr.obj >= VICTIM_BASE) & (fr.obj < DECOY_BASE)
    assert on.sum() > 5
    assert fr.thermal[on].max() < w.ambient + 2.5
    assert frame_to_label(fr, "thermal").sum() == 0 and frame_to_label(fr, "cnn").sum() == 0
    thin = next(v for v in w.victims if v.cover == "thin")
    fr = _view_of(w, (thin.x, thin.y), 1.2, 3.5)
    assert frame_to_label(fr, "thermal").sum() > 0 and frame_to_label(fr, "cnn").sum() == 0   # thermal-only case


def test_warm_objects_are_hot_but_not_people(buried_world):
    w = buried_world
    assert len(w.thermal_decoys) == CFG.n_warm_objects
    for d in w.thermal_decoys:
        lo, hi = WARM_OBJECTS[d.kind][4]
        assert lo <= d.temp <= hi
        fx, fy = int(d.x / w.cfg.cell * 2), int(d.y / w.cfg.cell * 2)
        assert w.obj[fy, fx] >= THERMAL_BASE
        assert all(np.hypot(d.x - v.x, d.y - v.y) >= 2.0 for v in w.victims)
    sim = Simulator(BURIED_CFG)
    d = w.thermal_decoys[0]
    assert sim.truth_kind_at(d.x, d.y) == "warm" and "not a person" in sim._truth_at(d.x, d.y)


def test_parking_cars_can_have_warm_engines():
    w = generate(replace(CFG, building="parking", seed=2))
    warm = [f for f in w.furniture if f.kind == "car" and f.temp > 0]
    assert warm
    car = warm[0]
    cells = car.cells()
    temps = [w.nav_temp[y, x] for x, y in cells]
    assert max(temps) > 38 and min(temps) < w.ambient + 3          # hot bonnet, cold boot


def test_too_hot_detections_are_flagged(buried_world):
    w = buried_world
    heater = next(d for d in w.thermal_decoys if d.kind == "space heater")
    fr = _view_of(w, (heater.x, heater.y), 1.0, 3.0)
    det = CNNDetector.__new__(CNNDetector)                    # no network needed for the physics check
    det.cfg, det.modality, det.uses_thermal = replace(CFG, vision="thermal"), "thermal", True
    heat = np.zeros((12, 16), np.float32)
    iy, ix = np.unravel_index(int(np.argmax(fr.thermal)), fr.thermal.shape)
    heat[iy // 4, ix // 4] = 0.95
    out = det.detect(fr, heat)
    assert len(out) == 1 and out[0].temp > HUMAN_MAX and out[0].too_hot


# ----------------------------------------------------------------- sensors
def test_camera_cannot_see_through_walls(world):
    ys, xs = np.nonzero(world.nav_height > 2.0)
    for x, y in zip(xs, ys):                              # find a wall with free space west of it
        if 2 < x < world.W - 3 and not world.blocked[y, x - 1] and not world.blocked[y, x + 1] and world.nav_height[y, x + 1] < 0.1:
            px, py = world.to_metres((x - 1, y))
            cells = {tuple(c) for c in visible_cells(world, px, py, 0.0, fov_deg=10)}
            assert (x, y) in cells and (x + 1, y) not in cells
            return
    pytest.skip("no suitable wall")


def test_render_shows_victim_when_facing_it(world):
    v = next(v for v in world.victims if v.cover == "none")
    frame = _view_of(world, (v.x, v.y))
    assert frame.rgb.shape == (CFG.image_h, CFG.image_w, 3)
    assert ((frame.obj >= VICTIM_BASE) & (frame.obj < DECOY_BASE)).sum() > 5
    assert frame.depth.min() > 0 and frame.depth.max() <= CFG.camera_range + 2


# ------------------------------------------------------------------- lidar
def test_lidar_measures_walls_all_around(world):
    x, y = world.to_metres(world.starts[0])
    s = lidar.scan(world, x, y, CFG)                        # no noise
    assert len(s.ranges) == CFG.lidar_rays and np.isfinite(s.ranges).any()
    assert s.ranges[np.isfinite(s.ranges)].max() <= CFG.lidar_range + 0.1
    real = [bool(world.blocked[cy, cx]) for cx, cy in s.hit_cells]
    assert len(real) > 10 and np.mean(real) > 0.95                       # beams end on real obstacles
    assert not {tuple(c) for c in s.free_cells} & {tuple(c) for c in s.hit_cells}
    noisy = lidar.scan(world, x, y, CFG, np.random.default_rng(0))
    both = np.isfinite(s.ranges) & np.isfinite(noisy.ranges)
    assert both.sum() > 20 and 0 < np.abs(s.ranges[both] - noisy.ranges[both]).mean() < 0.05


def test_lidar_cannot_see_a_person_on_the_floor(world):
    v = next(v for v in world.victims if v.cover == "none")
    for cy, cx in np.argwhere(world.reachable):
        px, py = world.to_metres((cx, cy))
        if 1.0 < np.hypot(px - v.x, py - v.y) < 2.5 and line_of_sight(world, (px, py), (v.x, v.y)):
            break
    s = lidar.scan(world, px, py, CFG)
    body = set(v.nav_cells)
    assert not body & {tuple(c) for c in s.hit_cells}       # the beams pass over a lying person...
    assert body & {tuple(c) for c in s.free_cells}          # ...so the laser calls those cells free
    ev, solid = np.zeros_like(world.blocked, dtype=np.float32), np.zeros_like(world.blocked)
    for _ in range(10):
        apply_lidar(ev, solid, s.free_cells, s.hit_cells)
    bx, by = next(iter(body & {tuple(c) for c in s.free_cells}))
    assert classify(ev)[by, bx] == FREE                     # wrong, as far as the laser can tell
    apply_camera(ev, solid, np.array([[bx, by]]), world.blocked)
    for _ in range(10):
        apply_lidar(ev, solid, s.free_cells, s.hit_cells)
    assert classify(ev)[by, bx] == OBSTACLE                 # what the camera saw is final
    other = np.full_like(ev, -2.5)
    assert classify(merge_evidence([other, ev], solid))[by, bx] == OBSTACLE


def test_lidar_map_is_accurate_despite_noise():
    sim = Simulator(replace(CFG, max_steps=400)).run()
    known, truth = sim.team_map(), sim.world.blocked
    seen = known != UNKNOWN
    assert sim.mapped() > 0.95
    wrong_obstacle = (known == OBSTACLE) & ~truth
    assert wrong_obstacle.sum() <= 0.01 * seen.sum()        # noise must not wall the robots in
    wrong_free = (known == FREE) & truth & sim.coverage_union
    assert wrong_free.sum() == 0                            # nothing a camera looked at is misjudged


def test_lidar_alone_maps_the_building_but_finds_nobody():
    sim = Simulator(replace(CFG, vision="none", lidar=True, seed=3)).run()
    assert sim.cfg.sensors == "lidar"
    assert sim.finished and sim.t < CFG.max_steps          # the robots explore, finish and come home
    assert sim.mapped() > 0.9
    assert sim.coverage() == 0 and not sim.found_at and not sim.reports      # no camera: nothing searched, nobody found
    assert np.isnan(sim.thermal_map).all()
    assert "no camera" in sim.truth_summary() and any("cannot recognise people" in e for _, e in sim.events)
    with pytest.raises(ValueError):
        Simulator(replace(CFG, vision="none", lidar=False))


def test_three_sensor_sets():
    from rescue.config import SENSOR_SETS
    from rescue.server import _cfg_from
    assert {k: replace(CFG, vision=v, lidar=l).sensors for k, (v, l) in SENSOR_SETS.items()} == \
        {"lidar": "lidar", "cameras": "cameras", "both": "both"}
    cfg = _cfg_from({"sensors": ["cameras"]}, CFG)
    assert (cfg.vision, cfg.lidar) == ("fusion", False)
    cfg = _cfg_from({"sensors": ["lidar"]}, CFG)
    assert (cfg.vision, cfg.lidar) == ("none", True)
    cfg = _cfg_from({"sensors": ["both"], "vision": ["ideal"]}, CFG)       # presets may still name a vision mode
    assert (cfg.vision, cfg.lidar) == ("ideal", True)
    with pytest.raises(ValueError):
        _cfg_from({"sensors": ["sonar"]}, CFG)


def test_the_search_is_not_finished_where_only_the_laser_has_looked():
    """A heap the laser bounced off may be rubble on a person: a camera has to look at it."""
    known = np.full((5, 8), FREE, np.int8)
    known[2, 5] = OBSTACLE                                  # seen by the laser only
    searched = np.zeros((5, 8), bool)
    searched[:, :5] = True                                  # the cameras have covered the left part
    e = explore_map(known, searched)
    assert e[2, 5] == UNKNOWN and e[2, 4] == FREE
    assert any((4, 2) in map(tuple, f.cells.tolist()) for f in frontiers(e))     # so the cell next to it is a goal
    searched[2, 5] = True
    assert explore_map(known, searched)[2, 5] == OBSTACLE
    # the mission in which this went wrong: six robots, perfect eyes, one victim partly under rubble
    sim = Simulator(replace(CFG, strategy="coordinated", n_robots=6, building="school", seed=201)).run()
    assert len(sim.found_at) == sum(v.detectable for v in sim.world.victims)


def test_lidar_maps_ahead_of_the_cameras():
    with_lidar = Simulator(replace(CFG, seed=3, max_steps=60)).run()
    without = Simulator(replace(CFG, seed=3, lidar=False, max_steps=60)).run()
    assert with_lidar.mapped() > with_lidar.coverage() + 0.03       # the laser sees further than the cameras
    assert abs(without.mapped() - without.coverage()) < 0.02         # camera only: mapped = searched
    assert with_lidar.robots[0].scan is not None and without.robots[0].scan is None
    e = explore_map(with_lidar.team_map(), np.logical_or.reduce([r.searched for r in with_lidar.robots]))
    assert ((e == UNKNOWN) & (with_lidar.team_map() == FREE)).sum() > 0   # mapped, but still to be searched


# ----------------------------------------------------------------- mapping
def test_frontiers_and_dijkstra():
    known = np.full((5, 8), UNKNOWN, np.int8)
    known[:, :4] = FREE
    known[2, 2] = OBSTACLE
    fr = frontiers(known)
    assert len(fr) == 1 and all(c[0] == 3 for c in fr[0].cells)
    dist, parent = dijkstra(known, (0, 2))
    assert dist[2, 3] == pytest.approx(4.414, abs=0.01)            # detour, no corner cutting
    assert np.isinf(dist[0, 5])                                      # unknown is never planned through
    path = extract_path(parent, (0, 2), (3, 2))
    assert path[-1] == (3, 2) and (2, 2) not in path


def test_no_corner_cutting():
    known = np.full((3, 3), FREE, np.int8)
    known[0, 1] = known[1, 0] = OBSTACLE
    dist, _ = dijkstra(known, (0, 0))
    assert np.isinf(dist[1, 1])


# ---------------------------------------------------------------- registry
def _rep(i, pos, x=5.0, y=5.0, p=0.9, d=1.5, temp=None):
    return Report(i, i, 0, pos, x, y, p, d, temp)


def test_close_confident_sightings_confirm():
    reg = Registry(RescueConfig())
    for i in range(5):
        reg.apply(_rep(i, True, x=5.0 + 0.05 * i, p=0.95, temp=30 + i))
    reg.settle(5)
    assert reg.candidates[0].status == "candidate"          # 5 frames are not enough yet (6 needed)
    reg.apply(_rep(5, True, x=5.2, p=0.95))
    assert reg.candidates[0].status == "candidate"          # the verdict is only taken in settle()
    assert [ch for _, ch in reg.settle(6)] == ["confirmed"]
    assert len(reg.candidates) == 1 and reg.candidates[0].status == "confirmed"
    assert reg.candidates[0].temp == 34                     # warmest reading is kept


def test_negative_looks_reject_false_alarm():
    reg = Registry(RescueConfig())
    reg.apply(_rep(0, True, p=0.65, d=4.0))
    reg.settle(0)
    assert reg.candidates[0].status == "candidate"
    for i in range(1, 4):
        reg.apply(_rep(i, False, d=1.5))
    reg.settle(4)
    assert reg.candidates[0].status == "rejected"


def test_robots_that_share_their_reports_always_agree():
    """With the radio reaching everyone, every robot holds the same sightings with the same verdict
    at every second: one robot can never confirm what another rejected on the same evidence."""
    sim = Simulator(replace(CFG, vision="thermal", n_robots=3, building="parking", seed=5))
    for _ in range(160):
        sim.step()
        views = [[(round(c.x, 6), round(c.y, 6), c.status) for c in r.registry.candidates] for r in sim.robots]
        assert all(v == views[0] for v in views)
    assert any(c.status != "candidate" for c in sim.robots[0].registry.candidates)     # verdicts were taken


def test_one_person_is_reported_once():
    """Head and legs of one body can be confirmed as two sightings; they count as one person."""
    reg = Registry(RescueConfig())
    i = 0
    for x in (5.0, 6.5, 12.0):                              # 1.5 m apart (same body), then someone else
        for _ in range(6):
            reg.apply(Report(i, 10 * i, 0, True, x, 5.0, 0.95, 1.5)); i += 1
    reg.settle(i)
    assert [c.status for c in reg.candidates] == ["confirmed"] * 3
    people = reg.people()
    assert len(people) == 2 and [round(c.x) for c in people] == [5, 12]        # the first report of a body is kept
    assert all(np.hypot(a.x - b.x, a.y - b.y) >= SAME_PERSON for a in people for b in people if a is not b)
    assert distinct([]) == []


def test_reports_are_idempotent():
    reg = Registry(RescueConfig())
    r = _rep(0, True, p=0.7, d=4)
    reg.apply(r); reg.apply(r)
    assert reg.candidates[0].positives == 1


# ------------------------------------------------------------ coordination
def test_auction_spreads_robots():
    sim = Simulator(replace(CFG, n_robots=2, strategy="coordinated"))
    sim._groups = sim.groups()
    a, b = sim.robots
    same = [Goal("frontier", (10, 10), 10.0, 5.0), Goal("frontier", (40, 30), 8.0, 20.0)]
    choice = _auction(sim, [a, b], {a.id: list(same), b.id: list(same)})
    assert choice[a.id].cell != choice[b.id].cell


# -------------------------------------------------------------- simulation
@pytest.mark.parametrize("strategy", ["greedy", "partition", "coordinated"])
def test_mission_finds_victims_safely(strategy):
    sim = Simulator(replace(CFG, strategy=strategy, max_steps=600))
    while not sim.finished:
        sim.step()
        cells = [r.cell for r in sim.robots]
        assert len(set(cells)) == len(cells), "two robots in one cell"
        assert all(not sim.world.blocked[c[1], c[0]] for c in cells), "robot inside an obstacle"
    detectable = sum(v.detectable for v in sim.world.victims)
    assert len(sim.found_at) >= detectable - 1
    assert sim.coverage() > 0.9


def test_even_perfect_eyes_cannot_see_through_thick_rubble():
    sim = Simulator(replace(BURIED_CFG, strategy="coordinated")).run()
    deep = {v.id for v in sim.world.victims if not v.detectable}
    assert deep and not (deep & set(sim.found_at))
    assert sim.coverage() > 0.95
    from rescue.metrics import summarize
    m = summarize(sim)
    assert m["victims_buried_deep"] == len(deep) and m["found"] == m["victims_detectable"]
    assert m["recall_detectable"] == 1.0 and m["recall"] < 1.0
    assert m["found_thin_cover"] == m["victims_thin_cover"] > 0    # faint signatures were found
    assert "buried under thick rubble" in sim.truth_summary()
    assert not any(f"/{len(sim.world.victims)}" in e for _, e in sim.events)   # the log never leaks the total


def test_thermal_map_covers_what_the_team_saw():
    sim = Simulator(replace(CFG, max_steps=60)).run()
    seen = sim.coverage_union
    assert np.array_equal(~np.isnan(sim.thermal_map), seen)
    hot = sim.thermal_map[seen] > sim.world.ambient + 8
    assert hot.sum() > 0                                            # a warm body or object was mapped


def test_limited_radio_reports_reach_base():
    sim = Simulator(replace(CFG, comm_range=8.0, max_steps=900)).run()
    detectable = sum(v.detectable for v in sim.world.victims)
    assert len(sim.found_at) >= detectable - 1
    assert len(sim.reported_at) == len(sim.found_at)          # every find eventually reaches the base
    assert all(sim.reported_at[v] >= sim.found_at[v] for v in sim.reported_at)


def test_radio_is_blocked_by_walls():
    sim = Simulator(replace(CFG, comm_range=10.0))
    w = sim.world
    ys, xs = np.nonzero(w.nav_height > 2.0)
    for x, y in zip(xs, ys):
        a, b = (x - 4, y), (x + 4, y)                          # 4 m apart, wall in between
        if 0 <= a[0] and b[0] < w.W and not w.blocked[a[1], a[0]] and not w.blocked[b[1], b[0]]:
            sim.robots[0].cell, sim.robots[1].cell = a, b
            assert not sim.can_talk(sim.robots[0], sim.robots[1])   # 4 m > 35 % of 10 m through a wall
            sim.robots[1].cell = (x - 1, y)                        # same side, 1.5 m, clear line of sight
            assert sim.can_talk(sim.robots[0], sim.robots[1])
            return
    pytest.skip("no suitable wall")


@pytest.mark.parametrize("building", ["office", "apartments", "hospital", "school", "parking", "warehouse"])
@pytest.mark.parametrize("damage", ["light", "severe"])
def test_every_building_type_is_searchable(building, damage):
    w = generate(replace(CFG, building=building, damage=damage, seed=9))
    assert len(w.victims) == CFG.n_victims
    assert w.searchable > 1200
    for a in w.victims:                                        # people are distinct (>= 2 m apart)
        assert all(np.hypot(a.x - b.x, a.y - b.y) >= 2.0 for b in w.victims if b is not a)
    assert w.temp.shape == w.height.shape and np.isfinite(w.temp).all()


def test_known_victim_count_ends_mission_early():
    seed = next(s for s in range(7, 40) if all(v.detectable for v in generate(replace(CFG, seed=s)).victims))
    unknown = Simulator(replace(CFG, seed=seed)).run()
    known = Simulator(replace(CFG, seed=seed, victims_known=True)).run()
    assert len(known.found_at) == CFG.n_victims
    assert known.t < unknown.t
    assert any("expected victims are confirmed" in e for _, e in known.events)


def test_coordination_reduces_duplicate_goals():
    overlap = {}
    for strategy in ("greedy", "coordinated"):
        sims = [Simulator(replace(CFG, strategy=strategy, seed=s)).run() for s in (3, 4)]
        overlap[strategy] = sum(s.overlap_pairs for s in sims) / sum(s.overlap_samples for s in sims)
    assert overlap["coordinated"] < 0.5 * overlap["greedy"]


@pytest.mark.parametrize("building", ["office", "warehouse"])
def test_partition_robots_only_search_their_own_zone(building):
    sim = Simulator(replace(CFG, strategy="partition", building=building, seed=3))
    inside = total = 0
    while not sim.finished:
        sim.step()
        for r in sim.robots:
            total += 1
            inside += sim.region[r.cell[1], r.cell[0]] == r.region
            if r.goal is not None and r.goal.kind in ("frontier", "verify"):
                assert sim.region[r.goal.cell[1], r.goal.cell[0]] == r.region, "explored a teammate's zone"
    assert inside / total > 0.6              # the rest is driving to / from the zone
    detectable = sum(v.detectable for v in sim.world.victims)
    assert sim.coverage() > 0.95 and len(sim.found_at) >= detectable - 1


def test_furniture_blocks_robots_and_view():
    w = generate(replace(CFG, building="parking", seed=2))
    cars = [f for f in w.furniture if f.kind == "car"]
    assert cars
    for x, y in cars[0].cells():
        assert w.blocked[y, x] and w.blocks_view()[y, x]      # a car is taller than the camera


def test_one_confirmation_counts_one_victim():
    sim = Simulator(CFG)
    v = sim.world.victims[0]
    assert len(sim.match([(v.x, v.y, 0), (v.x + 0.2, v.y, 0)])) == 1


def test_simulation_is_deterministic():
    a = Simulator(replace(CFG, max_steps=150)).run()
    b = Simulator(replace(CFG, max_steps=150)).run()
    assert [r.cell for r in a.robots] == [r.cell for r in b.robots]


# ----------------------------------------------------------------- dashboard
def test_dashboard_hides_the_truth_until_revealed():
    from rescue.server import Session, encode_thermal_map
    s = Session(replace(BURIED_CFG, max_steps=40))
    s.sim.run()
    st = s.state_payload()
    assert st["metrics"]["victims"] is None and st["metrics"]["victims_buried_deep"] is None and st["truth_summary"] is None
    assert all(v["status"] == "found" for v in st["victim_status"])
    assert "ground_truth_simulation_only" not in s.victim_map()
    assert len(st["thermal_map"]) == s.sim.world.W * s.sim.world.H == len(st["searched"]) == len(st["known"])
    assert len(st["robots"][0]["scan"]) > 20 and st["metrics"]["mapped"] >= st["metrics"]["coverage"]
    s.reveal = True
    st = s.state_payload()
    assert {v["status"] for v in st["victim_status"]} >= {"buried"} and "thick rubble" in st["truth_summary"]
    vm = s.victim_map()
    assert "ground_truth_simulation_only" in vm and "limitations" in vm
    assert all(k in vm for k in ("confirmed_victims", "suspected_victims", "unexplained_warm_spots"))
    png = s.camera_png(0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = encode_thermal_map(np.array([[np.nan, 5.0, 20.0, 45.0]]))
    assert decoded[0] == "0" and decoded[1] == "1" and decoded[3] == chr(48 + 74)


@pytest.mark.skipif(not Path(RescueConfig().fusion_detector_path).exists(), reason="fusion detector not trained")
def test_reports_on_the_dashboard_add_up():
    """Confirmed by the robots = real victims + false alarms + people counted twice, the same numbers everywhere."""
    from rescue.server import Session
    s = Session(RescueConfig(vision="fusion", building="warehouse", seed=4, max_steps=400))
    s.sim.run()
    st = s.state_payload()
    rep, m = st["reports"], st["metrics"]
    confirmed = [c for c in st["candidates"] if c["status"] == "confirmed"]
    assert rep["confirmed"] == len(confirmed) == rep["real"] + rep["false"] + rep["twice"]
    assert rep["repeats"] == m["repeated_reports"] and not any(c["status"] == "repeat" for c in st["candidates"])
    assert all(np.hypot(a["x"] - b["x"], a["y"] - b["y"]) >= SAME_PERSON for a in confirmed for b in confirmed if a is not b)
    assert rep["real"] == m["found"] == len([v for v in st["victim_status"] if v["status"] == "found"])
    assert rep["false"] + rep["twice"] == m["false_alarms"] and m["confirmed_reports"] == rep["confirmed"]
    assert rep["twice"] == m["false_alarms_duplicate"]
    assert all(c["truth"]["text"] for c in confirmed) and all(c["truth"] is None for c in st["candidates"] if c["status"] == "candidate")
    assert all("not a person" in c["truth"]["text"] or "nothing there" in c["truth"]["text"] or "already counted" in c["truth"]["text"]
               for c in confirmed if not c["truth"]["real"])
    for kind, width in (("colour", 64 * 6), ("thermal", 64 * 6), ("both", (64 * 2 + 2) * 5)):
        from PIL import Image
        import io
        assert Image.open(io.BytesIO(s.camera_png(0, kind))).size[0] == width


def test_every_browser_tab_has_its_own_mission():
    from rescue.server import MAX_SESSIONS, Sessions
    sessions = Sessions(replace(CFG, max_steps=50))
    a = sessions.get("tab-a", replace(CFG, building="office", seed=1))
    b = sessions.get("tab-b", replace(CFG, building="school", seed=2))
    a.sim.step(); a.sim.step()
    assert a is not b and (a.sim.t, b.sim.t) == (2, 0)
    assert sessions.get("tab-a") is a and sessions.get("tab-a").sim.cfg.building == "office"
    assert sessions.get("tab-a", replace(CFG, building="parking")).sim.t == 0          # a reset starts a new mission
    for i in range(MAX_SESSIONS + 3):
        sessions.get(f"tab-{i}")
    assert len(sessions.by_id) == MAX_SESSIONS and "tab-b" not in sessions.by_id


def test_lidar_only_dashboard_has_no_camera():
    from rescue.server import Session
    s = Session(replace(CFG, vision="none", max_steps=30))
    s.sim.run()
    w, st = s.world_payload(), s.state_payload()
    assert w["sensors"] == "lidar" and w["has_lidar"] and not w["has_cameras"]
    assert set(st["searched"]) == {"0"} and "1" in st["known"] and st["reports"]["confirmed"] == 0
    assert len(st["robots"][0]["scan"]) > 20
    with pytest.raises(ValueError):
        s.camera_png(0, "thermal")


@pytest.mark.parametrize("modality", ["cnn", "thermal", "fusion"])
def test_learned_detector_missions_run(modality):
    cfg = RescueConfig(vision=modality, max_steps=120)
    if not Path(model_path(cfg, modality)).exists():
        pytest.skip(f"{modality} detector not trained")
    sim = Simulator(cfg).run()
    assert sim.robots[0].last_heat is not None
    assert sim.coverage() > 0.2
    if modality != "cnn":
        assert sim.robots[0].last_frame.thermal is not None
