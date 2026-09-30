"""Multi-robot search-and-rescue simulation loop.

One step = one second. Each step every robot:
  1. SENSE        its LiDAR measures the distance to the walls all around and adds the scan
                  to its map; its depth camera adds what lies on the floor ahead; its thermal
                  camera adds the warmest reading of every visible cell to the team's thermal
                  map. The cells the cameras covered count as *searched*.
  2. DETECT       its victim detector looks at the camera image(s); sightings become reports
                  (a "person" hotter than 40 °C is dropped: heaters and engines are not people)
  3. COMMUNICATE  robots within radio range merge their maps and victim reports
  4. DECIDE       robots without a goal pick one (the next waypoint of their coverage pattern /
                  a frontier to explore / a sighting to verify)
  5. ACT          drive (turn on the spot, then drive at cfg.speed: see motion.py), spin to scan,
                  or turn to look at a sighting; every move and every second costs energy

Robots only ever use their *own* knowledge. The ground truth (where victims really are) is
used by the simulator to render images and to score performance, never to decide.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import heapq

import numpy as np

from .config import RescueConfig
from . import coverage, lidar
from .coordination import Goal, choose, partition_regions, path_to
from .mapping import (FREE, L_LIDAR_MAX, L_MAX, MOVES, OBSTACLE, UNKNOWN, apply_camera, apply_lidar, classify, dijkstra, explore_map, extract_path,
                      merge_evidence)
from .motion import (BACKOFF, BLOCKED_STEPS, RESERVE, SAFETY, TABU_S, TRAIL_LEN, base_power, clearance_map, drive_energy_per_m,
                     dwa_step, energy_home, move_time, rejoin, stuck_in_loop, turn_energy, wrap)
from .planners import Plan, plan, route_length
from .sensors import line_of_sight, render, visible_cells
from .thermal import HUMAN_MAX, describe_cover
from .victims import MERGE_RADIUS, Registry, Report, distinct
from .world import generate

HEADINGS = {(1, 0): 0.0, (1, 1): np.pi / 4, (0, 1): np.pi / 2, (-1, 1): 3 * np.pi / 4,
            (-1, 0): np.pi, (-1, -1): -3 * np.pi / 4, (0, -1): -np.pi / 2, (1, -1): -np.pi / 4}
REDECIDE_EVERY = 12
MAX_FAILED_BREAKS = 4    # a robot gives up driving through laser-only obstacles after this many real ones
MATCH_RADIUS = 1.5     # metres: a confirmed location this close to a real victim counts as found


@dataclass
class RobotAgent:
    @property
    def energy_used(self) -> float:
        """Joules used over the whole mission (energy_j is only what the battery is down by now)."""
        return self.energy_j + self.energy_charged

    id: int
    cell: tuple[int, int]
    heading: float
    known: np.ndarray
    seen: np.ndarray
    registry: Registry
    state: str = "idle"           # idle | exploring | scanning | verifying | returning | done
    goal: Goal | None = None
    path: list = field(default_factory=list)
    scan_left: int = 0
    look_left: int = 0
    stuck: int = 0
    last_decision: int = -10**9
    region: int = 0
    dist: np.ndarray | None = None
    parent: dict | None = None
    distance: float = 0.0
    verify_trips: int = 0
    detections: int = 0
    last_frame: object = None
    last_heat: np.ndarray | None = None
    reason: str = "starting up"
    finished_reason: str = ""
    too_hot: int = 0              # detections dropped because they were hotter than any person
    evidence: np.ndarray | None = None   # log-odds that each cell is occupied (LiDAR + camera + bumper)
    solid: np.ndarray | None = None      # cells the camera or the bumper proved to be in the way
    searched: np.ndarray | None = None   # cells a camera (mine or a teammate's) has looked at
    scan: object = None                  # my latest LiDAR scan
    bumps: int = 0                       # times the bumper found an obstacle the map did not show
    probed: np.ndarray | None = None     # cells I drove into because my sensors could not resolve them
    gaveup: np.ndarray | None = None     # cells still unresolved after a probe: no longer a frontier (shared)
    scanned_at: tuple | None = None      # where I last made a 360-degree camera scan
    probes: int = 0                      # times I drove into a cell my sensors could not resolve
    energy_j: float = 0.0                # joules the battery is down by (0 = full); see energy_used
    depleted: bool = False               # battery empty: stopped where I am
    low_battery: bool = False            # heading home to recharge (without recharging: stays home)
    low_battery_reason: str = ""
    charges: int = 0                     # times I recharged at the base
    energy_charged: float = 0.0          # joules the charger put back (energy used = energy_j + this)
    charge_s: int = 0                    # seconds spent on the charger
    unaffordable: int = 0                # goals I had to leave out this decision: not enough battery to get there and back
    dist_home: np.ndarray | None = None  # distance (cells) from the base to every cell (battery checks)
    credit: float = 0.0                  # seconds of driving time available (see motion.py)
    conflicts: int = 0                   # times I had to wait because a teammate was in the way
    deadends: int = 0                    # dead ends / loops detected and backed out of
    replans: int = 0                     # routes replanned because a new obstacle appeared on them
    moves: int = 0                       # cells driven into
    repeat_moves: int = 0                # ...of which I had driven through before
    plan: object = None                  # the latest route plan (planners.Plan)
    plans: int = 0                       # routes planned
    plan_ms: float = 0.0                 # total planning time
    plan_work: int = 0                   # total cells expanded / samples / ant steps
    dist_ms: float = 0.0                 # time of the last distance field (used to choose goals)
    route_checked: bool = False          # replan when a new obstacle blocks this route
    visits: np.ndarray | None = None     # how often I drove through each cell
    trail: object = None                 # my last positions (dead-end / loop detection)
    history: object = None               # the cells I came through, most recent last (to back off along)
    tabu: dict = field(default_factory=dict)   # goal cell -> time until which I avoid it (dead end)
    failed_breaks: int = 0               # attempts to drive through a laser-only obstacle that really was one
    sweep: object = None                 # my coverage pattern (coverage.Sweep), None for frontier exploration
    opt_dist: np.ndarray | None = None   # distances with unmapped cells assumed free (to reach pattern waypoints)
    opt_parent: dict | None = None
    track: list = field(default_factory=list)   # every cell I drove into, in order (for display)
    outbox: list = field(default_factory=list)  # ids of my reports this second (applied in _communicate)


@dataclass
class BaseStation:
    """The rescue team's command post at the entrance. A victim only counts as *reported*
    once news of it reaches here (directly or relayed through other robots)."""
    id: int
    cell: tuple[int, int]
    known: np.ndarray
    registry: Registry
    evidence: np.ndarray | None = None
    solid: np.ndarray | None = None
    searched: np.ndarray | None = None
    gaveup: np.ndarray | None = None


class Simulator:
    def __init__(self, cfg: RescueConfig, world=None):
        self.cfg = cfg
        # the building comes from cfg.seed; every random choice in the mission (sensor noise, tie-breaks,
        # RRT* samples, ants) from cfg.run_seed if set, so the same building can be run differently
        run = cfg.seed if cfg.run_seed is None else cfg.run_seed
        self.rng = np.random.default_rng(run + 1)
        self.cam_rng = np.random.default_rng(run + 2)
        self.lidar_rng = np.random.default_rng(run + 3)
        self.plan_rng = np.random.default_rng(run + 4)
        self._tie_rng = np.random.default_rng(run + 5)
        self.world = world or generate(cfg)
        W, H = self.world.W, self.world.H
        starts = self.world.starts
        self.robots = [RobotAgent(i, starts[(i * 3) % len(starts)], 0.0,
                                  np.zeros((H, W), np.int8), np.zeros((H, W), bool), Registry(cfg))
                       for i in range(cfg.n_robots)]
        self.region, self.region_centres = partition_regions(self.world, cfg.n_robots, self.rng)
        for i, r in enumerate(sorted(self.robots, key=lambda r: r.cell[1])):
            order = np.argsort([c[1] for c in self.region_centres])
            r.region = int(order[i])
        self.region_boxes = None
        if cfg.coverage != "frontier":
            # a fixed coverage pattern: one rectangle of the building per robot, each swept with the pattern
            boxes = coverage.regions(self.world, cfg.n_robots)
            starts_xy = [self.pos(r) for r in self.robots]
            for r, k in zip(self.robots, coverage.assign(starts_xy, boxes, cfg.coverage, self.world)):
                r.region = k
                r.sweep = coverage.make_sweep(cfg.coverage, self.world, boxes[k], k, self.pos(r))
            self.region = coverage.region_map(self.world, boxes)
            self.region_boxes = boxes
            self.region_centres = [((b[0] + b[2]) / 2 / cfg.cell, (b[1] + b[3]) / 2 / cfg.cell) for b in boxes]
        if cfg.vision == "none" and not cfg.lidar:
            raise ValueError("the robots need at least one sensor: a camera (vision) or the LiDAR")
        self.detector = None
        if cfg.vision in ("cnn", "thermal", "fusion"):
            from .perception import CNNDetector
            self.detector = CNNDetector(cfg)
        # the team's thermal map: warmest temperature seen in every cell (display and export only)
        self.thermal_map = np.full((H, W), np.nan, dtype=np.float32)
        self.reports: list[Report] = []
        self.t = 0
        self.finished = False
        self.events: list[tuple[int, str]] = []
        self.found_at: dict[int, int] = {}          # victim id -> time first confirmed
        self.announced: list[tuple[float, float, str]] = []
        self.links: set = set()
        self.messages = 0
        self.history: list[dict] = []
        self.coverage_union = np.zeros((H, W), bool)
        mid = self.world.H // 2
        self.base = BaseStation(-1, (1, mid), np.zeros((H, W), np.int8), Registry(cfg))
        for node in [*self.robots, self.base]:
            node.evidence = np.zeros((H, W), np.float32)
            node.solid = np.zeros((H, W), bool)
            node.searched = np.zeros((H, W), bool)
            node.gaveup = np.zeros((H, W), bool)
        self.visits = np.zeros((H, W), np.int32)          # how often any robot drove through each cell
        self.team_repeat = 0                                # moves into a cell some robot had already driven through
        for r in self.robots:
            r.probed = np.zeros((H, W), bool)
            r.visits = np.zeros((H, W), np.int32)
            r.trail, r.history = deque(maxlen=TRAIL_LEN), deque([r.cell], maxlen=80)
            r.visits[r.cell[1], r.cell[0]] += 1
            self.visits[r.cell[1], r.cell[0]] += 1
            r.track.append(r.cell)
        self.reported_at: dict[int, int] = {}       # victim id -> time the base learned about it
        self.overlap_pairs = 0
        self.overlap_samples = 0
        self._groups = [self.robots + [self.base]]
        radio = "unlimited" if not cfg.comm_range else f"{cfg.comm_range:g} m (walls cut it to {0.35 * cfg.comm_range:.1f} m)"
        count = f"{cfg.n_victims} victims expected" if cfg.victims_known else "number of victims unknown: search everything"
        eyes = {"cnn": "AI on the colour camera", "thermal": "AI on the thermal camera",
                "fusion": "AI on colour + thermal cameras", "ideal": "perfect eyes",
                "none": "none (no camera on board: the robots cannot recognise people)"}[cfg.vision]
        maps = f"LiDAR ({cfg.lidar_range:g} m, all around) + depth camera" if cfg.lidar else "depth camera only (no LiDAR)"
        where = "an open test ground" if cfg.building == "plain" else f"a {cfg.damage}ly damaged {cfg.building} building"
        self._log(f"{cfg.n_robots} robots enter {where} "
                  f"(about {self.world.ambient:.0f} °C inside). Strategy: {cfg.strategy}. Vision: {eyes}. "
                  f"Mapping: {maps}. Radio: {radio}. {count}.")
        for r in self.robots:
            if r.sweep is not None:
                b = r.sweep.box
                self._log(f"R{r.id}: {r.sweep.pattern_name} of region {r.region + 1} (x {b[0]:.1f}-{b[2]:.1f} m, "
                          f"y {b[1]:.1f}-{b[3]:.1f} m): {r.sweep.describe}, {len(r.sweep.cells)} waypoints, "
                          f"{coverage.path_length(r.sweep.corners):.0f} m long.")

    # ------------------------------------------------------------------ helpers
    def _log(self, text: str) -> None:
        self.events.append((self.t, text))

    def pos(self, r: RobotAgent) -> tuple[float, float]:
        return self.world.to_metres(r.cell)

    def can_talk(self, a, b) -> bool:
        """Radio link between two nodes (robots or the base). Concrete walls weaken the
        signal: through a wall the range drops to 35 %."""
        rng = self.cfg.comm_range
        if not rng:
            return True
        pa, pb = self.pos(a), self.pos(b)
        d = np.hypot(pa[0] - pb[0], pa[1] - pb[1])
        if d <= 0.35 * rng:
            return True
        return d <= rng and line_of_sight(self.world, pa, pb)

    def groups(self) -> list[list]:
        """Nodes that can exchange data, directly or relayed through each other."""
        nodes = self.robots + [self.base]
        if not self.cfg.comm_range:
            return [nodes]
        parent = list(range(len(nodes)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, a in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                if self.can_talk(a, nodes[j]):
                    parent[find(i)] = find(j)
        comps: dict[int, list] = {}
        for i, n in enumerate(nodes):
            comps.setdefault(find(i), []).append(n)
        return list(comps.values())

    def teammates(self, r: RobotAgent) -> list[RobotAgent]:
        """Robots r can currently talk to (directly or relayed)."""
        for g in self._groups:
            if r in g:
                return [o for o in g if o is not r and isinstance(o, RobotAgent)]
        return []

    def connected_to_base(self, r: RobotAgent) -> bool:
        return any(r in g and self.base in g for g in self._groups)

    def tie_break(self) -> float:
        """A small random amount added to goal utilities when runs are set to vary (cfg.run_seed);
        0 otherwise, so the same settings always give the same mission."""
        return float(self._tie_rng.uniform(0.0, 0.3)) if self.cfg.run_seed is not None else 0.0

    def capacity(self, r) -> float:
        """Joules r's battery holds when full (inf = unlimited). cfg.battery_each overrides cfg.battery_wh."""
        each = self.cfg.battery_each
        wh = each[r.id] if r.id < len(each) else self.cfg.battery_wh
        return wh * 3600.0 if wh and wh > 0 else np.inf

    def battery_left(self, r: RobotAgent) -> float:
        """Joules left in r's battery (inf when the battery is unlimited)."""
        return self.capacity(r) - r.energy_j

    def charge(self, r) -> float | None:
        """Share of r's battery left, 0..1 (None: unlimited)."""
        cap = self.capacity(r)
        return None if not np.isfinite(cap) else max(0.0, 1.0 - r.energy_j / cap)

    def _battery_low(self, r: RobotAgent) -> bool:
        """Quick check every second: is it time to head home? (straight-line distance x 1.5 for detours)"""
        if not np.isfinite(self.capacity(r)) or r.depleted or r.cell in self.world.starts:
            return False
        if r.low_battery:
            return True
        x, y = r.cell
        d = 1.5 * min(np.hypot(x - sx, y - sy) for sx, sy in self.world.starts) * self.cfg.cell
        return self.battery_left(r) <= energy_home(self.cfg, d) + RESERVE * self.capacity(r)

    def affordable(self, r: RobotAgent, cell, dist_m: float, work_s: float = 5.0) -> bool:
        """Energy-aware task choice: can r drive dist_m to ``cell``, work there for work_s seconds (a scan,
        a close look) and still get back to base with the safety factor and the reserve?
            E_go   = 1.3 (e_drive d + P_base (d / v + work_s))
            E_back = 1.3 (e_drive d_home + P_base d_home / v)
            affordable  <=>  battery left >= E_go + E_back + 5 % of capacity
        d_home comes from a distance field grown from the base on r's own map."""
        cap = self.capacity(r)
        if not np.isfinite(cap) or r.dist_home is None:
            return True
        d_home = float(r.dist_home[cell[1], cell[0]]) * self.cfg.cell
        if not np.isfinite(d_home):
            d_home = dist_m + float(np.hypot(*np.subtract(self.pos(r), self.world.to_metres(self.base.cell))))
        cfg = self.cfg
        go = SAFETY * (drive_energy_per_m(cfg) * dist_m + base_power(cfg) * (dist_m / cfg.speed + work_s))
        return self.battery_left(r) >= go + energy_home(cfg, d_home) + RESERVE * cap

    def recharge_goal(self, r: RobotAgent, why: str) -> Goal:
        """Go home to recharge (the reason is kept until the robot is charged again)."""
        home = min(self.world.starts, key=lambda c: r.dist[c[1], c[0]])
        r.low_battery, r.low_battery_reason = True, why
        return Goal("home", home, 0.0, float(r.dist[home[1], home[0]]) * self.cfg.cell, reason=why)

    def _route_blocked(self, r: RobotAgent) -> bool:
        """Has a new obstacle appeared on the route I am following?"""
        return r.route_checked and bool(r.path) and any(r.known[y, x] == OBSTACLE for x, y in r.path[:20])

    # --------------------------------------------------------------------- step
    def run(self, steps: int | None = None) -> "Simulator":
        for _ in range(steps or self.cfg.max_steps):
            if self.finished:
                break
            self.step()
        return self

    def step(self) -> None:
        if self.finished:
            return
        for r in self.robots:
            self._sense(r)
        self._groups = self.groups()
        self._communicate()
        self._decide()
        self._act()
        self._score()
        self.t += 1
        if self.t >= self.cfg.max_steps or all(r.state == "done" for r in self.robots):
            self.finished = True
            of = f" of {self.cfg.n_victims} expected" if self.cfg.victims_known else ""
            if self.cfg.vision == "none":
                self._log(f"Mission ended at t={self.t}s: {100 * self.mapped():.0f}% of the building mapped by LiDAR. "
                          f"Nobody was found: a laser measures distances, it cannot recognise people.")
            else:
                self._log(f"Mission ended at t={self.t}s: {len(self.found_at)} victims found{of}, "
                          f"{100 * self.coverage():.0f}% of the building searched.")

    def truth_summary(self) -> str:
        """Ground truth in plain words (shown on request only when the victim count is unknown)."""
        vs = self.world.victims
        deep = [v for v in vs if not v.detectable]
        thin = [v for v in vs if v.cover == "thin"]
        missed = [v for v in vs if v.detectable and v.id not in self.found_at]
        parts = [f"{len(vs)} people were really inside; the robots found {len(self.found_at)}."]
        if self.cfg.vision == "none":
            parts.append("These robots carried a LiDAR but no camera: they mapped the building, but a laser cannot "
                         "tell a person from a box and sees nothing lying on the floor.")
        if deep:
            parts.append(f"{len(deep)} {'was' if len(deep) == 1 else 'were'} buried under thick rubble: no heat reaches "
                         f"the surface, so no camera (thermal or colour) could ever detect them. Finding them needs "
                         f"listening devices, search dogs or rescuers digging.")
        if thin:
            found_thin = sum(v.id in self.found_at for v in thin)
            parts.append(f"{len(thin)} {'was' if len(thin) == 1 else 'were'} covered by thin debris with only a faint "
                         f"heat signature; {found_thin} of them found.")
        if missed:
            parts.append(f"{len(missed)} visible {'victim was' if len(missed) == 1 else 'victims were'} missed.")
        return " ".join(parts)

    # ------------------------------------------------------------------- sense
    def _sense(self, r: RobotAgent) -> None:
        world, cfg = self.world, self.cfg
        x, y = self.pos(r)
        if cfg.lidar:                       # the laser maps the walls all around (noisy, blind below its plane)
            for _ in range(2 if self.t == 0 else 1):        # two turns before the robots set off
                r.scan = lidar.scan(world, x, y, cfg, self.lidar_rng)
                apply_lidar(r.evidence, r.solid, r.scan.free_cells, r.scan.hit_cells)
        cx, cy = r.cell                     # the floor I stand on is free: I am standing on it
        r.evidence[cy, cx], r.solid[cy, cx] = -L_MAX, False
        if cfg.vision == "none":            # no camera on board: the robot maps, but it cannot search for people
            classify(r.evidence, out=r.known)
            r.searched |= r.known != UNKNOWN                 # exploration then simply follows the map
            return
        cells = visible_cells(world, x, y, r.heading)
        apply_camera(r.evidence, r.solid, cells, world.blocked)     # the depth camera: floor and low obstacles ahead
        classify(r.evidence, out=r.known)
        r.seen[cells[:, 1], cells[:, 0]] = True
        r.searched[cells[:, 1], cells[:, 0]] = True
        self.coverage_union[cells[:, 1], cells[:, 0]] = True
        if world.nav_temp is not None:      # thermal map: warmest reading per cell the team has ever taken
            cy, cx = cells[:, 1], cells[:, 0]
            self.thermal_map[cy, cx] = np.fmax(self.thermal_map[cy, cx], world.nav_temp[cy, cx])

        capture = cfg.vision == "ideal" or r.state in ("scanning", "verifying") or (self.t + r.id) % 2 == 0
        if not capture:
            return
        if cfg.vision == "ideal":
            vis = set(map(tuple, cells.tolist()))
            dets = []
            for v in world.victims:
                d = np.hypot(v.x - x, v.y - y)
                # perfect eyes still cannot see through thick rubble: no camera can
                if v.detectable and d <= cfg.camera_range and any(c in vis for c in v.nav_cells):
                    dets.append((v.x + self.rng.normal(0, 0.15), v.y + self.rng.normal(0, 0.15), 0.97, d, None))
        else:
            frame = render(world, x, y, r.heading, self.cam_rng)
            heat = self.detector.heatmap(frame)
            r.last_frame, r.last_heat = frame, heat
            dets = []
            for det in self.detector.detect(frame, heat):
                if det.too_hot:            # physics check: nothing on a living person is hotter than 40 °C
                    r.too_hot += 1
                    self._announce(det.x, det.y, "hot", f"R{r.id}'s thermal camera sees a {det.temp:.0f} °C object at "
                                   f"({det.x:.1f}, {det.y:.1f}) m: far too hot for a person, ignored "
                                   f"(ground truth: {self._truth_at(det.x, det.y)}).")
                    continue
                dets.append((det.x, det.y, det.confidence, det.distance, det.temp))
        for dx, dy, p, d, temp in dets:
            self._report(r, True, dx, dy, p, d, temp)
            r.detections += 1
        # negative evidence: candidates in clear view that the detector did not fire on
        vis_cells = set(map(tuple, cells.tolist()))
        for c in list(r.registry.candidates):
            if c.status != "candidate":
                continue
            d = np.hypot(c.x - x, c.y - y)
            if d > cfg.verify_distance + 0.5 or world.to_cell(c.x, c.y) not in vis_cells:
                continue
            if not any(np.hypot(c.x - dx, c.y - dy) < MERGE_RADIUS for dx, dy, *_ in dets):
                self._report(r, False, c.x, c.y, 0.0, d)

    def _report(self, r: RobotAgent, positive: bool, x, y, p, d, temp=None) -> None:
        rep = Report(len(self.reports), self.t, r.id, positive, float(x), float(y), float(p), float(d),
                     None if temp is None else float(temp))
        self.reports.append(rep)
        r.outbox.append(rep.id)             # applied with everybody else's reports in _communicate

    def _new_candidate_event(self, rep: Report) -> None:
        what = (f"R{rep.robot}'s thermal camera picked up a {rep.temp:.0f} °C heat signature" if rep.temp is not None
                else f"R{rep.robot} spotted a possible victim")
        self._announce(rep.x, rep.y, "candidate", f"{what} at ({rep.x:.1f}, {rep.y:.1f}) m, confidence "
                                                  f"{rep.confidence:.2f} from {rep.distance:.1f} m away")

    def _truth_at(self, x, y) -> str:
        for v in self.world.victims:
            if np.hypot(v.x - x, v.y - y) <= MATCH_RADIUS:
                return "a real victim, " + describe_cover(v)
        for dcy in self.world.decoys:
            if np.hypot(dcy.x - x, dcy.y - y) <= MATCH_RADIUS:
                return f"a {dcy.kind}, not a person"
        for w in self.world.thermal_decoys:
            if np.hypot(w.x - x, w.y - y) <= MATCH_RADIUS:
                return f"a {w.kind} at {w.temp:.0f} °C, not a person"
        for f in self.world.furniture:
            if f.temp > 0 and any(np.hypot((cx + 0.5) * self.cfg.cell - x, (cy + 0.5) * self.cfg.cell - y) <= MATCH_RADIUS
                                  for cx, cy in f.cells()):
                return f"a parked car with a warm engine ({f.temp:.0f} °C), not a person"
        return "nothing there (false alarm)"

    def truth_kind_at(self, x, y) -> str:
        """victim | warm (a warm object or engine) | lookalike (a visual decoy) | nothing"""
        if any(np.hypot(v.x - x, v.y - y) <= MATCH_RADIUS for v in self.world.victims):
            return "victim"
        if any(np.hypot(w.x - x, w.y - y) <= MATCH_RADIUS for w in self.world.thermal_decoys) or any(
                f.temp > 0 and any(np.hypot((cx + 0.5) * self.cfg.cell - x, (cy + 0.5) * self.cfg.cell - y) <= MATCH_RADIUS
                                   for cx, cy in f.cells()) for f in self.world.furniture):
            return "warm"
        if any(np.hypot(d.x - x, d.y - y) <= MATCH_RADIUS for d in self.world.decoys):
            return "lookalike"
        return "nothing"

    def _announce(self, x, y, kind, text) -> bool:
        for ax, ay, k in self.announced:
            if k == kind and np.hypot(ax - x, ay - y) < MERGE_RADIUS:
                return False
        self.announced.append((x, y, kind))
        self._log(text)
        return True

    def _status_event(self, c, change) -> None:
        if change == "confirmed":
            heat = f", {c.temp:.0f} °C" if c.temp is not None else ""
            self._announce(c.x, c.y, "confirmed",
                           f"CONFIRMED victim at ({c.x:.1f}, {c.y:.1f}) m after {c.positives} sightings{heat} "
                           f"by R{', R'.join(map(str, sorted(c.reporters)))}. Ground truth: {self._truth_at(c.x, c.y)}.")
        elif change == "rejected":
            self._announce(c.x, c.y, "rejected",
                           f"Sighting at ({c.x:.1f}, {c.y:.1f}) m rejected after a closer look. "
                           f"Ground truth: {self._truth_at(c.x, c.y)}.")

    # -------------------------------------------------------------- communicate
    def _communicate(self) -> None:
        for g in self._groups:
            if len(g) < 2:
                continue
            solid = np.logical_or.reduce([r.solid for r in g])
            evidence = merge_evidence([r.evidence for r in g], solid)
            searched = np.logical_or.reduce([r.searched for r in g])
            gaveup = np.logical_or.reduce([r.gaveup for r in g])
            merged = classify(evidence)
            for r in g:
                if (r.known != merged).any() or (r.searched != searched).any():
                    self.messages += 1
                r.evidence[:], r.solid[:], r.searched[:], r.known[:] = evidence, solid, searched, merged
                r.gaveup[:] = gaveup
        # victim reports: everyone in a group applies the same new reports in the same order (by id), then
        # takes its verdicts, so robots holding the same evidence always agree (see victims.py)
        for g in self._groups:
            ids = set().union(*(r.registry.known | set(getattr(r, "outbox", ())) for r in g))
            for r in g:
                for rid in sorted(ids - r.registry.known):
                    if r.registry.apply(self.reports[rid]) == "new" and r is not self.base:
                        self._new_candidate_event(self.reports[rid])
                for c, change in r.registry.settle(self.t):
                    if r is not self.base:
                        self._status_event(c, change)
        for r in self.robots:
            r.outbox.clear()
        links = {(a.id, b.id) for g in self._groups for a in g for b in g if a.id < b.id}
        self.links = links if self.cfg.comm_range else set()

    # ------------------------------------------------------------------- decide
    def _goal_still_valid(self, r: RobotAgent) -> bool:
        g = r.goal
        if g is None:
            return False
        if g.kind == "frontier":               # still a searched cell at the edge of the unsearched area?
            x, y = g.cell
            k = self.explore_view(r)
            return k[y, x] == FREE and any(
                0 <= x + dx < k.shape[1] and 0 <= y + dy < k.shape[0] and k[y + dy, x + dx] == UNKNOWN
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if g.kind == "verify":
            return any(c.id == g.candidate for c in r.registry.open_candidates())
        if g.kind == "sweep":                      # the waypoint I am heading for, and not inside an obstacle
            x, y = g.cell
            return r.sweep is not None and r.sweep.index == g.waypoint and r.known[y, x] != OBSTACLE
        if g.kind == "report":
            return not self.connected_to_base(r)
        if g.kind in ("probe", "backtrack"):
            return bool(r.path)
        if g.kind == "transit":                    # keep going until I am inside my zone
            return self.region[r.cell[1], r.cell[0]] != r.region and bool(r.path)
        return True

    def _unreported(self, r: RobotAgent) -> list:
        """Victims r has confirmed that the base has not heard about yet."""
        at_base = self.base.registry.confirmed()
        return [c for c in r.registry.confirmed()
                if all(np.hypot(c.x - b.x, c.y - b.y) >= MERGE_RADIUS for b in at_base)]

    def _special_goal(self, r: RobotAgent) -> Goal | None:
        """Goals that override exploring: all known victims found, or news to deliver."""
        cfg = self.cfg
        home = min(self.world.starts, key=lambda c: r.dist[c[1], c[0]])
        d = float(r.dist[home[1], home[0]]) * cfg.cell
        cap = self.capacity(r)
        if np.isfinite(cap) and np.isfinite(d):
            left = self.battery_left(r)
            if r.low_battery or left <= energy_home(cfg, d) + RESERVE * cap:
                if not r.low_battery:           # the decision sticks until the robot has recharged
                    r.low_battery = True
                    r.low_battery_reason = (f"battery low ({100 * left / cap:.0f}% left): just enough to drive the "
                                            f"{d:.0f} m back to base" + (" and recharge" if cfg.recharge else ""))
                return Goal("home", home, 0.0, d, reason=r.low_battery_reason)
        if cfg.victims_known and len(r.registry.people()) >= cfg.n_victims:
            return Goal("home", home, 0.0, d, reason=f"all {cfg.n_victims} expected victims are confirmed: mission complete")
        if cfg.comm_range and not self.connected_to_base(r):
            late = [c for c in self._unreported(r) if self.t - (c.confirmed_at or self.t) >= cfg.report_timeout]
            if cfg.strategy == "coordinated":
                # one courier per victim is enough: the robot that first saw it (every robot can
                # work this out from the shared reports, even when they are out of touch)
                late = [c for c in late if min(c.reporters) == r.id]
            if late:
                return Goal("report", home, 0.0, d,
                            reason=f"base has not heard about {len(late)} victim(s) I found: driving back into radio range")
        return None

    def _decide(self) -> None:
        deciding = []
        for r in self.robots:
            if r.state in ("scanning", "verifying", "charging") or r.depleted:
                continue
            periodic = self.t - r.last_decision >= REDECIDE_EVERY
            if r.state == "done" and not periodic:
                continue
            blocked_route = self._route_blocked(r)
            if blocked_route:                     # the map changed under my route: plan again before I run into it
                r.replans += 1
                c = next(c for c in r.path[:20] if r.known[c[1], c[0]] == OBSTACLE)
                m = self.world.to_metres(c)
                self._log(f"R{r.id} sees an obstacle on its route at ({m[0]:.1f}, {m[1]:.1f}) m: replanning.")
            homing = r.goal is not None and r.goal.kind == "home"
            if (r.goal is None or not self._goal_still_valid(r) or periodic or (not r.path and r.cell != r.goal.cell)
                    or blocked_route or (not homing and self._battery_low(r))):
                deciding.append(r)
        if not deciding:
            return
        for r in deciding:
            avoid = {o.cell for o in self.robots if o is not r} if r.stuck >= 3 else None
            t0 = time.perf_counter()
            r.dist, r.parent = dijkstra(r.known, r.cell, avoid)
            r.dist_ms = (time.perf_counter() - t0) * 1000.0
            r.last_decision = self.t
            r.unaffordable = 0
            if np.isfinite(self.capacity(r)):                # how far every cell is from the base (battery checks)
                r.dist_home, _ = dijkstra(r.known, self.base.cell, optimistic=r.sweep is not None and not r.sweep.done)
            if r.sweep is not None and not r.sweep.done:       # waypoints may lie in space not mapped yet
                r.opt_dist, r.opt_parent = dijkstra(r.known, r.cell, avoid, optimistic=True)
        special = {r.id: self._special_goal(r) for r in deciding}
        choices = choose(self, [r for r in deciding if special[r.id] is None])
        choices.update({k: v for k, v in special.items() if v is not None})
        for r in deciding:
            g = choices.get(r.id)
            if g is None and r.unaffordable:
                g = self._battery_blocked(r)
            if g is None:
                g = self._breakthrough(r)
            if g is None:
                home = min(self.world.starts, key=lambda c: r.dist[c[1], c[0]])
                g = Goal("home", home, 0.0, float(r.dist[home[1], home[0]]) * self.cfg.cell,
                         reason=r.finished_reason or "no unexplored area or unchecked sighting left on my map")
            if g.kind == "home" and r.cell not in self.world.starts and not np.isfinite(r.dist[g.cell[1], g.cell[0]]):
                g = self._way_home(r, g)
            if g.kind in ("home", "report"):
                if r.state not in ("returning", "done", "reporting"):
                    self._log(f"R{r.id} returns to base: {g.reason}.")
                r.reason = g.reason
                if g.kind == "home" and r.cell == g.cell:
                    r.goal, r.path = g, []
                    self._at_base(r)
                    continue
                r.state = "returning" if g.kind == "home" else "reporting"
            else:
                changed = r.goal is None or r.goal.kind != g.kind or r.goal.cell != g.cell
                if g.kind == "verify" and changed:
                    r.verify_trips += 1
                    self._log(f"R{r.id} goes to take a closer look at the sighting near "
                              f"({g.look_at[0]:.1f}, {g.look_at[1]:.1f}) m ({g.distance:.0f} m away).")
                r.state = {"frontier": "exploring", "verify": "verifying_trip", "transit": "transit", "probe": "exploring",
                           "backtrack": "backtracking", "sweep": "sweeping"}[g.kind]
                if g.kind == "probe":
                    self._log(f"R{r.id}: {g.reason}.")
                r.reason = g.reason
            r.path = self._route(r, g)
            r.goal = g
            if r.cell == g.cell and g.kind == "frontier" and (self.cfg.vision == "none" or r.scanned_at == r.cell):
                self._probe(r)              # I have looked from here already: my sensors cannot see further
            elif r.cell == g.cell and g.kind != "home":
                self._arrived(r)            # already standing on the goal

    # ---------------------------------------------------------------------- act
    def _route(self, r: RobotAgent, g: Goal) -> list:
        """The route to goal g, planned with the chosen algorithm (cfg.planner, see planners.py)."""
        cfg = self.cfg
        if g.path is not None:                   # transit, probe, careful way home: the goal brings its own route
            r.route_checked = g.kind == "transit"
            return list(g.path)
        r.route_checked = True
        if r.cell == g.cell:
            return []
        # the same goal as before and its route is still clear: keep it (a slow planner need not run again)
        if cfg.planner in ("rrtstar", "aco") and r.path and r.path[-1] == g.cell and r.goal is not None \
                and r.goal.cell == g.cell and not self._route_blocked(r):
            return r.path
        extra = cfg.revisit_penalty * np.minimum(self.visits, 5).astype(float) if cfg.revisit_penalty else None
        # a pattern waypoint may lie beyond the mapped area: plan as if unmapped cells were free (the free-space
        # assumption) and replan as soon as the sensors show an obstacle on the route (_route_blocked)
        optimistic = g.kind == "sweep"
        known = np.where(r.known == UNKNOWN, FREE, r.known).astype(np.int8) if optimistic else r.known
        dist, parent = (r.opt_dist, r.opt_parent) if optimistic else (r.dist, r.parent)
        if cfg.planner == "dijkstra" and extra is None:
            # the distance field computed to choose the goal already holds the shortest route
            path = extract_path(parent, r.cell, g.cell)
            p = Plan(path, route_length(r.cell, path), route_length(r.cell, path), int(np.isfinite(dist).sum()),
                     r.dist_ms, "dijkstra")
        else:
            avoid = {o.cell for o in self.robots if o is not r} if r.stuck >= 3 else None
            p = plan(cfg.planner, known, r.cell, g.cell, self.plan_rng, extra, avoid, reach=np.isfinite(dist))
            if not p.path:
                p.path = extract_path(parent, r.cell, g.cell)
        r.plan, r.plans, r.plan_ms, r.plan_work = p, r.plans + 1, r.plan_ms + p.ms, r.plan_work + p.expanded
        return list(p.path)

    # ---------------------------------------------------------------------- act
    def _act(self) -> None:
        cfg = self.cfg
        movers = []
        for r in self.robots:
            if r.depleted:
                continue
            if r.state == "scanning":
                r.heading += np.pi / 2
                r.energy_j += turn_energy(np.pi / 2)
                r.scan_left -= 1
                if r.scan_left <= 0:
                    r.state, r.goal = "idle", None
                continue
            if r.state == "charging":                     # docked: the charger refills the battery
                r.energy_charged += min(r.energy_j, cfg.charge_power)
                r.energy_j = max(0.0, r.energy_j - cfg.charge_power)
                r.charge_s += 1
                if r.energy_j <= 0.0:
                    r.low_battery, r.low_battery_reason = False, ""
                    r.state, r.goal, r.last_decision = "idle", None, -10**9
                    self._log(f"R{r.id} is fully charged after {r.charge_s} s on the charger in total: back to work.")
                continue
            if r.state == "verifying":
                gx, gy = r.goal.look_at
                x, y = self.pos(r)
                new = float(np.arctan2(gy - y, gx - x))
                r.energy_j += turn_energy(wrap(new - r.heading))
                r.heading = new
                r.look_left -= 1
                if r.look_left <= 0:
                    # a completed close look counts as one attempt for everyone r can talk to
                    for o in [r, *self.teammates(r)]:
                        for c in o.registry.candidates:
                            if np.hypot(c.x - gx, c.y - gy) < MERGE_RADIUS:
                                c.verify_attempts += 1
                                c.tried_spots.append((x, y))
                    r.state, r.goal = "idle", None
                continue
            r.credit = min(r.credit + 1.0, 1.0)          # one more second of driving time (no hoarding while waiting)
            if r.path:
                movers.append(r)

        # drive: in rounds, so a fast robot can take several cells per second; in every round each robot
        # with driving time left states the cell it wants, and conflicts are resolved as before
        waiting: set[int] = set()
        for _ in range(8):
            active = [r for r in movers if r.path and r.credit > 1e-9 and r.id not in waiting and not r.depleted]
            if not active:
                break
            intents: dict[int, tuple[int, int]] = {}
            for r in active:
                nxt = self._next_cell(r)
                if nxt is None:                           # DWA: every move leads away or is blocked: wait
                    r.stuck += 1
                    r.conflicts += 1
                    waiting.add(r.id)
                    continue
                intents[r.id] = nxt
            claimed, blocked = self._resolve(intents)
            for i in blocked:
                self.robots[i].stuck += 1
                self.robots[i].conflicts += 1
                waiting.add(i)
            for tgt, i in claimed.items():
                self._move(self.robots[i], tgt)
        for r in self.robots:
            if r.stuck >= 3 and r.path:
                r.path, r.goal = [], None        # blocked by a teammate for a while: replan around it

        # every second: base power, battery, dead ends
        for r in self.robots:
            if r.depleted:
                continue
            if r.state not in ("done", "charging"):
                r.energy_j += base_power(cfg)
            if r.energy_j >= self.capacity(r):
                r.depleted, r.state, r.path, r.goal = True, "done", [], None
                m = self.pos(r)
                r.reason = f"battery empty at ({m[0]:.1f}, {m[1]:.1f}) m: stopped, waiting to be recovered"
                self._log(f"R{r.id}: {r.reason}.")
                continue
            if r.state in ("exploring", "sweeping", "verifying_trip", "transit", "returning", "reporting", "backtracking"):
                r.trail.append(r.cell)
                if cfg.deadend_recovery and r.goal is not None and r.goal.kind != "backtrack" and \
                        (stuck_in_loop(r.trail) or r.stuck >= BLOCKED_STEPS):
                    self._recover(r)

    def _next_cell(self, r: RobotAgent):
        """The cell r drives into next: the route's next cell, or the DWA choice (motion.dwa_step)."""
        if self.cfg.avoidance != "dwa":
            return r.path[0]
        others = [o.cell for o in self.robots if o is not r]
        cell, _ = dwa_step(r.cell, r.heading, r.path, r.known, others, clearance_map(r.known, others))
        return cell

    def _resolve(self, intents: dict) -> tuple[dict, set]:
        """No two robots in one cell, no swapping through each other. Repeated until stable, because
        a blocked robot also blocks whoever wanted its cell. Returns (cell -> robot, blocked robots)."""
        occupied = {r.cell: r.id for r in self.robots}
        order = sorted(intents, key=lambda i: (self.robots[i].state != "verifying_trip", i))
        blocked: set[int] = set()
        while True:
            claimed: dict = {}
            newly = None
            for i in order:
                if i in blocked:
                    continue
                tgt = intents[i]
                occ = occupied.get(tgt)
                swap = occ is not None and intents.get(occ) == self.robots[i].cell
                if tgt in claimed or (occ is not None and (occ not in intents or occ in blocked or swap)):
                    newly = i
                    break
                claimed[tgt] = i
            if newly is None:
                return claimed, blocked
            blocked.add(newly)

    def _move(self, r: RobotAgent, tgt) -> None:
        """Turn towards tgt and drive into it: costs time (motion.move_time) and energy."""
        cfg = self.cfg
        dx, dy = tgt[0] - r.cell[0], tgt[1] - r.cell[1]
        new_heading = HEADINGS[(dx, dy)]
        dtheta = wrap(new_heading - r.heading)
        if self.world.blocked[tgt[1], tgt[0]]:        # bumper: map was wrong, replan
            if r.goal is not None and r.goal.kind == "probe":
                r.failed_breaks += 1
            r.known[tgt[1], tgt[0]] = 2
            r.evidence[tgt[1], tgt[0]], r.solid[tgt[1], tgt[0]] = L_MAX, True
            r.bumps += 1
            r.path, r.goal = [], None
            r.heading = new_heading
            r.credit -= move_time(cfg, 0.1, dtheta)   # turned, touched it, stopped
            r.energy_j += turn_energy(dtheta)
            return
        dist = cfg.cell * (1.4142 if dx and dy else 1.0)
        r.credit -= move_time(cfg, dist, dtheta)
        r.energy_j += drive_energy_per_m(cfg) * dist + turn_energy(dtheta)
        r.heading = new_heading
        r.distance += dist
        r.cell = tgt
        r.moves += 1
        if r.visits[tgt[1], tgt[0]]:
            r.repeat_moves += 1
        if self.visits[tgt[1], tgt[0]]:
            self.team_repeat += 1
        r.visits[tgt[1], tgt[0]] += 1
        self.visits[tgt[1], tgt[0]] += 1
        r.track.append(tgt)
        if not r.history or r.history[-1] != tgt:
            r.history.append(tgt)
        if r.path and r.path[0] == tgt:
            r.path.pop(0)
        else:
            r.path = rejoin(r.path, tgt)              # DWA stepped off the route: continue from here, or plan again
        r.stuck = 0
        if not r.path and r.goal is not None and r.cell == r.goal.cell:
            self._arrived(r)

    def _recover(self, r: RobotAgent) -> None:
        """Dead end or loop: back off along my own trail and avoid that goal for TABU_S seconds."""
        looping = stuck_in_loop(r.trail)
        g = r.goal
        r.deadends += 1
        if g.kind in ("frontier", "verify", "probe", "transit", "sweep"):
            r.tabu[g.cell] = self.t + TABU_S
        others = {o.cell for o in self.robots if o is not r}
        dist, parent = dijkstra(r.known, r.cell, others)
        x, y = r.cell
        target = next((c for c in reversed(r.history)
                       if max(abs(c[0] - x), abs(c[1] - y)) >= BACKOFF and np.isfinite(dist[c[1], c[0]])), None)
        r.trail.clear()
        r.stuck = 0
        m = self.pos(r)
        what = "going back and forth" if looping else f"blocked for {BLOCKED_STEPS} s"
        avoid = f"; avoiding that goal for {TABU_S} s" if g.kind != "home" else ""
        if target is None:
            r.state, r.goal, r.path = "idle", None, []
            self._log(f"R{r.id}: dead end at ({m[0]:.1f}, {m[1]:.1f}) m ({what}){avoid}.")
            return
        path = extract_path(parent, r.cell, target)
        back = route_length(r.cell, path) * self.cfg.cell
        r.reason = f"dead end at ({m[0]:.1f}, {m[1]:.1f}) m ({what}): backing off {back:.1f} m along my trail{avoid}"
        r.goal = Goal("backtrack", target, 0.0, back, reason=r.reason, path=path)
        r.path, r.state, r.route_checked = path, "backtracking", False
        self._log(f"R{r.id}: {r.reason}.")

    def _arrived(self, r: RobotAgent) -> None:
        g = r.goal
        if g.kind in ("frontier", "probe"):
            if self.cfg.scan_on_arrival and self.cfg.vision != "none":     # look around with the cameras
                r.state, r.scan_left, r.scanned_at = "scanning", 3, r.cell
            else:
                r.state, r.goal = "idle", None
        elif g.kind == "verify":
            r.state, r.look_left = "verifying", 2
        elif g.kind == "sweep":                    # waypoint reached: on to the next one (no 360 degree scan:
            sw = r.sweep                           # the pattern's lane spacing already covers the ground)
            if sw.index == g.waypoint:
                sw.arrive(len(r.track) - 1)
            r.state, r.goal = "idle", None
        elif g.kind == "home":
            self._at_base(r)
        elif g.kind in ("report", "transit", "backtrack"):
            r.state, r.goal = "idle", None

    def _at_base(self, r: RobotAgent) -> None:
        """Arrived at the base: recharge if I came back on low battery, otherwise my work is done."""
        if r.low_battery and self.cfg.recharge and np.isfinite(self.capacity(r)):
            if r.energy_j > 0.0:
                r.state, r.charges = "charging", r.charges + 1
                r.reason = f"charging at the base ({100 * self.charge(r):.0f}% now, {self.cfg.charge_power:g} W charger)"
                self._log(f"R{r.id} docks at the base with {100 * self.charge(r):.0f}% battery: charging.")
                return
        r.state = "done"
        self._log(f"R{r.id} is back at base" + (": its battery is empty of useful range, it stays here." if r.low_battery else "."))

    def _battery_blocked(self, r: RobotAgent) -> Goal | None:
        """Work is left on my map, but none of it is within reach of my battery (there and back).
        Go and recharge; with a full battery that still cannot reach it, stay at the base."""
        cap = self.capacity(r)
        full = r.energy_j <= 0.02 * cap + 5 * base_power(self.cfg)      # as good as full: just off the charger
        at_base = r.cell in self.world.starts
        if self.cfg.recharge and not (full and at_base):
            return self.recharge_goal(r, f"{r.unaffordable} task(s) left, but with {100 * self.charge(r):.0f}% "
                                         f"battery I could not get there and back: going to recharge")
        r.finished_reason = (f"{r.unaffordable} task(s) left, but even a full {cap / 3600:g} Wh battery cannot reach them "
                             f"and come back" if full else
                             f"{r.unaffordable} task(s) left, beyond the reach of my {100 * self.charge(r):.0f}% battery "
                             f"(recharging is off)")
        return None

    def _probe(self, r: RobotAgent) -> None:
        """I stand on a frontier, but my sensors cannot resolve the unknown cell next to it (a table
        top blocks the laser, a door frame looks solid, a camera cannot see into a corner). Drive
        into that cell: either I get in (then it is free) or the bumper stops me (an obstacle).
        A cell still unresolved after one probe is given up, so this always ends."""
        k, (x, y) = self.explore_view(r), r.cell
        H, W = k.shape
        around = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= x + dx < W and 0 <= y + dy < H and k[y + dy, x + dx] == UNKNOWN]
        for c in around:
            if r.probed[c[1], c[0]]:
                r.gaveup[c[1], c[0]] = True              # tried already and still unresolved: give up
        fresh = [c for c in around if not r.probed[c[1], c[0]]]
        if not fresh:
            r.state, r.goal, r.path = "idle", None, []
            return
        c = min(fresh, key=lambda c: r.known[c[1], c[0]] == OBSTACLE)          # try what the map calls free first
        r.probed[c[1], c[0]] = True
        r.probes += 1
        m = self.world.to_metres(c)
        r.reason = (f"my sensors cannot see past this spot: driving slowly into the unmapped cell at "
                    f"({m[0]:.1f}, {m[1]:.1f}) m to find out (the bumper stops me if it is blocked)")
        r.state, r.goal, r.path = "exploring", Goal("probe", c, 0.0, self.cfg.cell, reason=r.reason, path=[c]), [c]
        r.route_checked = False

    def _breakthrough(self, r: RobotAgent) -> Goal | None:
        """Nothing left that I can reach, but my map still has unsearched space sealed off by cells
        only the laser calls obstacles (beams grazing the jambs of a narrow doorway can "end" inside
        the opening, and then no robot drives through to correct it). Before giving up, try the
        cheapest way through such cells, preferring weak evidence: the bumper settles each cell for
        good (free: I get through; blocked: it becomes a proven obstacle, never tried again). A robot
        stops trying after MAX_FAILED_BREAKS real obstacles, so the search always ends."""
        if not self.cfg.lidar or r.failed_breaks >= MAX_FAILED_BREAKS:
            return None
        k = self.explore_view(r)
        H, W = k.shape
        near_unknown = np.zeros((H, W), bool)
        u = k == UNKNOWN
        near_unknown[1:] |= u[:-1]; near_unknown[:-1] |= u[1:]; near_unknown[:, 1:] |= u[:, :-1]; near_unknown[:, :-1] |= u[:, 1:]
        targets = (k == FREE) & near_unknown & ~np.isfinite(r.dist)          # frontier cells I cannot reach...
        if not targets.any():
            return None
        targets &= ~np.isfinite(dijkstra(r.known, r.cell)[0])                 # ...even ignoring teammates in the way
        found = self._hopeful_path(r, targets) if targets.any() else None
        if found is None:
            return None
        path, d, crossing = found
        m = self.world.to_metres(crossing)
        reason = (f"unexplored space is left behind ({m[0]:.1f}, {m[1]:.1f}) m, which only the laser calls blocked: "
                  f"driving through to check (the bumper stops me if it is really blocked)")
        return Goal("probe", path[-1], 0.0, d * self.cfg.cell, reason=reason, path=path)

    def _way_home(self, r: RobotAgent, g: Goal) -> Goal:
        """My map shows no way back to base (e.g. the way in was a cell only the laser called
        blocked): try the careful way through such cells. If there is none at all, stay here and
        say so, rather than waiting for a path that will never come."""
        starts = np.zeros(r.known.shape, bool)
        for c in self.world.starts:
            starts[c[1], c[0]] = True
        found = self._hopeful_path(r, starts)
        if found is not None:
            path, d, _ = found
            return Goal("home", path[-1], 0.0, d * self.cfg.cell, reason=g.reason, path=path)
        return Goal("home", r.cell, 0.0, 0.0, reason="my map shows no way back to base: waiting here for the rescue team")

    def _hopeful_path(self, r: RobotAgent, targets: np.ndarray):
        """Shortest way to any target cell that may also cross cells only the laser calls obstacles
        (never proven by the camera or the bumper); each such cell costs as much as 5-10 m of driving,
        more if the laser is surer. Diagonal moves never cut the corner of such a cell, so the robot
        really enters it and proves it free, and can always come back the same way.
        Returns (path, cost in cells, first doubtful cell) or None."""
        H, W = r.known.shape
        free = r.known == FREE
        doubtful = (r.known == OBSTACLE) & ~r.solid
        passable = free | doubtful
        extra = 10.0 + 10.0 * np.clip(r.evidence, 0, L_LIDAR_MAX) / L_LIDAR_MAX
        dist = np.full((H, W), np.inf)
        dist[r.cell[1], r.cell[0]] = 0.0
        parent, heap = {}, [(0.0, r.cell)]
        while heap:
            d, (x, y) = heapq.heappop(heap)
            if d > dist[y, x]:
                continue
            if targets[y, x] and (x, y) != r.cell:
                path = [(x, y)]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                path = path[-2::-1]
                return path, d, next((c for c in path if doubtful[c[1], c[0]]), path[-1])
            for dx, dy, cost in MOVES:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H) or not passable[ny, nx]:
                    continue
                if dx and dy and not (free[y, nx] and free[ny, x]):
                    continue                                   # do not cut corners
                nd = d + cost + (extra[ny, nx] if doubtful[ny, nx] else 0.0)
                if nd < dist[ny, nx]:
                    dist[ny, nx], parent[(nx, ny)] = nd, (x, y)
                    heapq.heappush(heap, (nd, (nx, ny)))
        return None

    def explore_view(self, r) -> np.ndarray:
        """The map r explores on: cells still to be searched are UNKNOWN (see mapping.explore_map)."""
        return explore_map(r.known, r.searched, r.gaveup)

    # ------------------------------------------------------------------- scoring
    def coverage(self) -> float:
        """Share of the reachable floor a camera has looked at (searched)."""
        reach = self.world.reachable
        return float((self.coverage_union & reach).sum() / reach.sum())

    def team_map(self) -> np.ndarray:
        solid = np.logical_or.reduce([r.solid for r in self.robots])
        return classify(merge_evidence([r.evidence for r in self.robots], solid))

    def mapped(self) -> float:
        """Share of the reachable floor that is on the team's map (LiDAR or camera)."""
        reach = self.world.reachable
        return float(((self.team_map() != UNKNOWN) & reach).sum() / reach.sum())

    def confirmed_sightings(self) -> list:
        """Every confirmed sighting of the team (the same sighting held by several robots once)."""
        out: list = []
        for r in self.robots:
            for c in r.registry.confirmed():
                if all(np.hypot(c.x - s.x, c.y - s.y) >= MERGE_RADIUS for s in out):
                    out.append(c)
        return out

    def confirmed_people(self) -> list:
        """The people the team believes it has found: its confirmed sightings, each person once."""
        return distinct(self.confirmed_sightings())

    def confirmed_locations(self) -> list[tuple[float, float, int]]:
        """Where the team says people are (x, y, time), each person once."""
        return [(c.x, c.y, c.confirmed_at) for c in self.confirmed_people()]

    def repeated_reports(self) -> int:
        """Confirmed sightings that are a second report of a person already counted."""
        return len(self.confirmed_sightings()) - len(self.confirmed_people())

    def match(self, locations) -> dict[int, tuple[float, float]]:
        """Pair confirmed locations with real victims one-to-one (closest pairs first)."""
        pairs = sorted((np.hypot(v.x - x, v.y - y), v.id, i) for i, (x, y, *_) in enumerate(locations)
                       for v in self.world.victims)
        used_v, used_l, out = set(), set(), {}
        for d, vid, i in pairs:
            if d <= MATCH_RADIUS and vid not in used_v and i not in used_l:
                used_v.add(vid); used_l.add(i)
                out[vid] = locations[i][:2]
        return out

    def _score(self) -> None:
        base_conf = [(c.x, c.y) for c in self.base.registry.confirmed()]
        for vid, (x, y) in self.match(base_conf).items():
            if vid not in self.reported_at:
                self.reported_at[vid] = self.t + 1
                if self.cfg.comm_range:
                    self._log(f"BASE RECEIVED the location of victim {vid} at ({x:.1f}, {y:.1f}) m.")
        goals = [r.goal.cell for r in self.robots if r.goal is not None and r.goal.kind == "frontier"]
        for i in range(len(goals)):
            for j in range(i + 1, len(goals)):
                self.overlap_samples += 1
                self.overlap_pairs += np.hypot(goals[i][0] - goals[j][0], goals[i][1] - goals[j][1]) * self.cfg.cell < 2.0
        conf = self.confirmed_locations()
        matched = self.match(conf)
        for vid in matched:
            if vid not in self.found_at:
                self.found_at[vid] = self.t + 1
                n = len(self.found_at)
                if n == len(self.world.victims):
                    self._log(f"ALL {n} victims have been found at t={self.t + 1}s.")
        false = len(conf) - len(matched)
        self.history.append({"t": self.t + 1, "coverage": round(self.coverage(), 4), "mapped": round(self.mapped(), 4),
                             "energy_wh": round(sum(r.energy_used for r in self.robots) / 3600.0, 3),
                             "found": len(self.found_at), "false": false})
