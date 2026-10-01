"""Local web server for the 3-D search-and-rescue dashboard (standard library only).

    python -m rescue serve   ->  http://localhost:8001

The dashboard itself is plain files in ``rescue/web`` (index.html, css/, js/, vendor/). The
browser asks for batches of simulation steps (GET /api/step?n=K) and draws the result with
Three.js.

    /api/reset?building=..&sensors=..  start a new mission (it waits until the first /api/step)
                                       sensors = lidar | cameras | both
    /api/step?n=K                      advance K seconds, return the new state
    /api/camera?robot=i&kind=thermal   what one of robot i's cameras sees (kind = colour | thermal |
                                       both), with the detector's heatmap and detections on top
    /api/victim_map                    the team's map of suspected victim locations as JSON
    /api/reveal                        show the ground truth (unknown victim count only)

Every browser tab runs its own mission: the page sends a random ``sid`` with each request and
the server keeps one Session per sid (the eight most recently used).
"""
from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import SENSOR_SETS, VISION_MODES, RescueConfig
from .coordination import STRATEGIES
from .coverage import PATTERNS
from .planners import PLANNERS
from .mapping import explore_map, frontiers
from .metrics import summarize
from .simulator import Simulator
from .thermal import SCALE_BELOW, SCALE_TOP, describe_cover, palette, to_unit
from .world import BUILDINGS, DAMAGE, FLOORS, OBJ_DEBRIS, PLAIN_KINDS

WEB = Path(__file__).parent / "web"
STATIC_DIRS = ("/vendor/", "/js/", "/css/")
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".html": "text/html; charset=utf-8", ".json": "application/json", ".png": "image/png",
                ".svg": "image/svg+xml"}
MAX_STEPS = 30
MAX_SESSIONS = 8
# metrics that would give away how many people are inside while the count is unknown
TRUTH_METRICS = ("victims", "recall", "time_50pct_victims", "time_all_victims", "time_all_detectable", "flagged_unconfirmed",
                 "victims_detectable", "victims_buried_deep", "victims_thin_cover", "recall_detectable", "found_thin_cover",
                 "detection_rate")
WARM_SPOT = 6.0     # °C above ambient: a cell on the thermal map worth pointing out


def _clean(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def encode_thermal_map(tmap: np.ndarray) -> str:
    """One character per cell: '0' = never seen, otherwise 5 °C + (code - 1) / 2 for '1'..'z'."""
    q = np.where(np.isnan(tmap), 0, np.clip(np.round((tmap - 5.0) * 2), 1, 74)).astype(int)
    return "".join(chr(48 + v) for v in q.ravel().tolist())


def _bits(a: np.ndarray) -> str:
    return "".join("1" if v else "0" for v in a.ravel().tolist())


class Session:
    def __init__(self, cfg: RescueConfig):
        self.lock = threading.Lock()
        self.reset(cfg)

    def reset(self, cfg: RescueConfig) -> None:
        self.sim = Simulator(cfg)
        self.reveal = False          # unknown victim count: ground truth only on explicit request

    def _truth_visible(self) -> bool:
        return self.sim.cfg.victims_known or self.reveal

    @staticmethod
    def _victim_body(v) -> dict:
        return {"id": v.id, "x": v.x, "y": v.y, "buried": round(v.buried, 2), "cover": v.cover,
                "detectable": v.detectable, "describe": describe_cover(v), "skin_temp": round(v.skin_temp, 1),
                "covered": [list(c) for c in v.covered], "head": v.head, "feet": v.feet, "colors": v.colors}

    # ---------------------------------------------------------------- payloads
    def world_payload(self) -> dict:
        w, sim = self.sim.world, self.sim
        f = w.cfg.cell / 2
        ys, xs = np.nonzero(w.structure)
        structure = [[int(x), int(y), int(w.structure[y, x]), round(float(w.nav_height[y, x]), 2)] for y, x in zip(ys, xs)]
        ys, xs = np.nonzero(w.rubble)
        rubble = [[int(x), int(y), round(float(w.rubble[y, x]), 2)] for y, x in zip(ys, xs)]
        ys, xs = np.nonzero(w.obj == OBJ_DEBRIS)
        debris = [[int(x), int(y), *map(int, w.color[y, x])] for y, x in zip(ys, xs)]
        c = w.cfg.cell
        furniture = [{"kind": it.kind, "x": (it.x0 + it.w / 2) * c, "y": (it.y0 + it.h / 2) * c,
                      "sx": it.w * c, "sy": it.h * c, "height": it.height, "color": list(it.color),
                      "facing": it.facing, "nav": [it.x0, it.y0, it.w, it.h]} for it in w.furniture]
        return {
            "W": w.W, "H": w.H, "cell": w.cfg.cell, "fine": f,
            "structure": structure, "rubble": rubble, "debris": debris, "furniture": furniture,
            "floor": "".join(map(str, w.floor.ravel().tolist())), "markings": w.markings,
            "floor_palette": [list(x) for x in FLOORS[w.cfg.building]],
            "decoys": [{"kind": d.kind, "x": d.x, "y": d.y, "size": d.size, "color": d.color, "height": d.height}
                       for d in w.decoys],
            "warm_objects": [{"kind": d.kind, "x": d.x, "y": d.y, "size": d.size, "color": d.color, "height": d.height}
                             for d in w.thermal_decoys],
            "ambient": round(w.ambient, 1),
            "thermal_scale": [round(w.ambient - SCALE_BELOW, 1), SCALE_TOP],
            "base": sim.pos(sim.base),
            "zones": "".join(map(str, sim.region.ravel().tolist())),
            # coverage patterns: one rectangle per robot (x0, y0, x1, y1 metres), None for frontier exploration
            "regions": None if sim.region_boxes is None else [list(map(float, b)) for b in sim.region_boxes],
            "zone_of": {r.id: r.region for r in sim.robots},
            "reachable": int(w.searchable),
            # what the robots carry: lidar | cameras | both
            "sensors": w.cfg.sensors, "has_lidar": w.cfg.lidar, "has_cameras": w.cfg.vision != "none",
            "config": {**w.cfg.to_dict(), "n_victims": w.cfg.n_victims if self._truth_visible() else None},
        }

    def _team_candidates(self) -> list[dict]:
        """Every sighting the team knows about, merged across robots (1.3 m), each confirmed or
        rejected one labelled with what it really was. The robots do not know these labels; the
        simulator does, and the dashboard shows them so that a false alarm is never mistaken for
        a victim."""
        sim = self.sim

        def row(c):
            return {"x": c.x, "y": c.y, "p": 1 / (1 + np.exp(-c.logodds)), "status": c.status, "temp": c.temp,
                    "rank": {"confirmed": 2, "candidate": 1, "rejected": 0}[c.status],
                    "sightings": c.positives, "close_looks": c.verify_attempts, "by": sorted(c.reporters),
                    "first_seen": c.first_seen, "confirmed_at": c.confirmed_at}

        # one person, one report: the simulator's own list of confirmed people (repeats already merged)
        cands = [row(c) for c in sim.confirmed_people()]
        taken = [(c.x, c.y) for c in sim.confirmed_sightings()]
        others: list = []
        for r in sim.robots:
            for c in r.registry.candidates:
                if c.status == "confirmed" or any(np.hypot(x - c.x, y - c.y) < 1.3 for x, y in taken):
                    continue
                same = next((d for d in others if np.hypot(d["x"] - c.x, d["y"] - c.y) < 1.3), None)
                new = row(c)
                if same is None:
                    others.append(new)
                elif new["rank"] > same["rank"] or (new["rank"] == same["rank"] and new["p"] > same["p"]):
                    same.update(new)
        cands += others
        confirmed = [c for c in cands if c["status"] == "confirmed"]
        real = {tuple(xy) for xy in sim.match([(c["x"], c["y"], 0) for c in confirmed]).values()}   # one report per person
        for c in cands:
            c["truth"] = None
            if c["status"] == "candidate":
                continue
            kind, text = sim.truth_kind_at(c["x"], c["y"]), sim._truth_at(c["x"], c["y"])
            if c["status"] == "confirmed":
                if (c["x"], c["y"]) in real:
                    kind = "victim"
                elif kind == "victim":
                    kind, text = "duplicate", "a second report of a person who is already counted"
            c["truth"] = {"kind": kind, "real": kind == "victim", "text": text}
        return cands

    def _warm_spots(self, cands) -> list[dict]:
        """Clusters of warm cells on the thermal map with no sighting nearby (unexplained heat)."""
        from scipy import ndimage
        sim, w = self.sim, self.sim.world
        hot = np.nan_to_num(sim.thermal_map, nan=-99) >= w.ambient + WARM_SPOT
        labels, n = ndimage.label(hot)
        out = []
        for k in range(1, n + 1):
            ys, xs = np.nonzero(labels == k)
            x, y = (xs.mean() + 0.5) * w.cfg.cell, (ys.mean() + 0.5) * w.cfg.cell
            if any(np.hypot(c["x"] - x, c["y"] - y) < 1.5 for c in cands):
                continue
            out.append({"x": round(float(x), 2), "y": round(float(y), 2), "cells": int(len(xs)),
                        "temperature_c": round(float(sim.thermal_map[ys, xs].max()), 1)})
        return out

    def state_payload(self, plan_robot: int | None = None) -> dict:
        sim = self.sim
        team = sim.team_map()
        fr = frontiers(explore_map(team, np.logical_or.reduce([r.searched for r in sim.robots]), np.logical_or.reduce([r.gaveup for r in sim.robots])))
        cands = self._team_candidates()
        confirmed = [c for c in cands if c["status"] == "confirmed"]
        # ground-truth status of the victims. When nobody knows how many people are inside,
        # only the ones the robots have found are sent (unless the user asks to reveal).
        truth = self._truth_visible()
        victim_status, bodies = [], []
        for v in sim.world.victims:
            who = sorted({rid for r in sim.robots for c in r.registry.confirmed()
                          if np.hypot(c.x - v.x, c.y - v.y) <= 1.5 for rid in c.reporters})
            if v.id in sim.found_at:
                st = {"id": v.id, "status": "found", "t": sim.found_at[v.id], "by": who}
            elif not truth:
                continue
            elif not v.detectable:
                st = {"id": v.id, "status": "buried"}
            elif any(d["status"] == "candidate" and np.hypot(d["x"] - v.x, d["y"] - v.y) <= 1.5 for d in cands):
                st = {"id": v.id, "status": "sighted"}
            else:
                st = {"id": v.id, "status": "hidden"}
            victim_status.append(st)
            bodies.append(self._victim_body(v))
        links = []
        if sim.cfg.comm_range:
            for g in sim.groups():
                links += [[a.id, b.id] for a in g for b in g if a.id < b.id and sim.can_talk(a, b)]
        m = summarize(sim)
        if not truth:   # nothing that would give away how many people are inside
            for k in TRUTH_METRICS:
                m[k] = None
        m["time_last_find"] = max(sim.found_at.values()) if sim.found_at else None
        return {
            "t": sim.t,
            "finished": sim.finished,
            "robots": [{
                "id": r.id, "x": sim.pos(r)[0], "y": sim.pos(r)[1], "heading": r.heading,
                "state": r.state, "distance": round(r.distance, 1), "detections": r.detections, "too_hot": r.too_hot,
                "reason": r.reason, "zone": r.region, "linked": sim.connected_to_base(r),
                "goal": None if r.goal is None else {"kind": r.goal.kind,
                                                     "x": sim.world.to_metres(r.goal.cell)[0],
                                                     "y": sim.world.to_metres(r.goal.cell)[1],
                                                     "tag": r.goal.tag, "utility": r.goal.utility},
                "path": [sim.world.to_metres(c) for c in r.path[:200]],
                # where the robot has driven (metres; long tracks thinned to at most ~600 points)
                "track": [sim.world.to_metres(c) for c in r.track[::max(1, len(r.track) // 600)] + r.track[-1:]],
                # its coverage pattern: corners (metres), waypoints done, skipped ones
                "sweep": None if r.sweep is None else {
                    "pattern": r.sweep.pattern, "describe": r.sweep.describe, "region": r.sweep.region,
                    "corners": [[round(x, 2), round(y, 2)] for x, y in r.sweep.corners],
                    "next": None if r.sweep.done else [round(v, 2) for v in r.sweep.points[r.sweep.index]],
                    "index": r.sweep.index, "total": len(r.sweep.cells), "reached": r.sweep.reached,
                    "skipped": [[round(v, 2) for v in r.sweep.points[i]] + [why] for i, why in r.sweep.skipped],
                    "finished_at": r.sweep.end_t},
                "bumps": r.bumps,
                # navigation: the latest route plan, energy, dead ends (planners.py, motion.py)
                "plan": None if r.plan is None else {
                    "algorithm": r.plan.algorithm, "length_m": round(r.plan.length * sim.cfg.cell, 1),
                    "work": r.plan.expanded, "ms": round(r.plan.ms, 1), "note": r.plan.note},
                "plans": r.plans, "plan_ms": round(r.plan_ms, 1), "replans": r.replans, "deadends": r.deadends,
                "conflicts": r.conflicts, "moves": r.moves, "repeat_moves": r.repeat_moves,
                "energy_wh": round(r.energy_used / 3600.0, 2),
                "battery": None if sim.charge(r) is None else round(sim.charge(r), 3), "depleted": r.depleted,
                "capacity_wh": None if not np.isfinite(sim.capacity(r)) else round(sim.capacity(r) / 3600.0, 2),
                "charges": r.charges, "charge_s": r.charge_s, "low_battery": r.low_battery,
                # what the planner looked at for this route (only for the robot the page asks about)
                "search": (r.plan.search if (r.plan is not None and plan_robot == i) else None),
                # the latest LiDAR scan: where the beams ended (metres)
                "scan": [] if r.scan is None else np.round(r.scan.points, 2).tolist(),
            } for i, r in enumerate(sim.robots)],
            "visits": "".join(map(str, np.minimum(sim.visits, 9).ravel().tolist())),   # times driven through, 0-9
            "known": "".join(map(str, team.ravel().tolist())),          # the map: 0 unknown, 1 free, 2 obstacle
            "searched": _bits(sim.coverage_union),                       # cells a camera has looked at
            "thermal_map": encode_thermal_map(sim.thermal_map),
            "frontiers": [list(map(int, c)) for f in fr for c in f.cells.tolist()],
            "candidates": [{k: c[k] for k in ("x", "y", "p", "status", "rank", "temp", "sightings", "by", "confirmed_at", "truth")}
                           for c in cands],
            # what the robots reported, checked against the truth: confirmed = real + false
            "reports": {"confirmed": len(confirmed), "real": sum(c["truth"]["real"] for c in confirmed),
                        # confirmed where nobody is (a dog, a heater, a jacket, nothing) ...
                        "false": sum(c["truth"]["kind"] not in ("victim", "duplicate") for c in confirmed),
                        # ... and a person already counted, confirmed a second time (> 1.6 m from the first report)
                        "twice": sum(c["truth"]["kind"] == "duplicate" for c in confirmed),
                        "suspected": sum(c["status"] == "candidate" for c in cands),
                        "repeats": sim.repeated_reports()},      # second reports of the same person, merged into the first
            "victim_status": victim_status,
            "victims": bodies,
            "revealed": self.reveal,
            "truth_summary": sim.truth_summary() if truth else None,
            "links": links,
            "metrics": m,
            "history": sim.history[::5][-400:] + sim.history[-1:],
            "victims_known": sim.cfg.victims_known,
            "events": [[t, e] for t, e in sim.events[-80:]],
        }

    def victim_map(self) -> dict:
        """The team's map of suspected victim locations: what the robots would hand to rescuers."""
        sim, w, cfg = self.sim, self.sim.world, self.sim.cfg
        cands = self._team_candidates()

        def row(c, confirmed):
            d = {"x": round(c["x"], 2), "y": round(c["y"], 2), "probability": round(c["p"], 3),
                 "sightings": c["sightings"], "seen_by_robots": c["by"], "first_seen_s": c["first_seen"],
                 "temperature_c": None if c["temp"] is None else round(c["temp"], 1)}
            if confirmed:
                d["confirmed_s"] = c["confirmed_at"]
                if self._truth_visible():
                    d["ground_truth_simulation_only"] = c["truth"]["text"]
            else:
                d["close_looks_taken"] = c["close_looks"]
            return d

        out = {
            "mission": {"building": cfg.building, "damage": cfg.damage, "robots": cfg.n_robots, "strategy": cfg.strategy,
                        "sensors": cfg.sensors, "vision": cfg.vision, "lidar": cfg.lidar, "mission_time_s": sim.t,
                        "finished": sim.finished, "building_mapped": round(sim.mapped(), 3),
                        "building_searched": round(sim.coverage(), 3), "ambient_c": round(w.ambient, 1)},
            "coordinates": "metres from the top-left corner of the map: x to the right, y downwards",
            "confirmed_victims": [row(c, True) for c in cands if c["status"] == "confirmed"],
            "suspected_victims": [row(c, False) for c in cands if c["status"] == "candidate"],
            "rejected_sightings": [{"x": round(c["x"], 2), "y": round(c["y"], 2), "sightings": c["sightings"]}
                                   for c in cands if c["status"] == "rejected"],
            "unexplained_warm_spots": self._warm_spots(cands),
            "limitations": ("Cameras, thermal or colour, cannot detect a person buried under thick rubble or "
                            "concrete: no heat reaches the surface. A LiDAR cannot recognise people at all. "
                            "Confirmed locations can be false alarms. Unsearched cells and confirmed locations "
                            "should be checked by rescuers with listening devices or search dogs."),
        }
        if self._truth_visible():
            out["ground_truth_simulation_only"] = [
                {"id": v.id, "x": round(v.x, 2), "y": round(v.y, 2), "cover": v.cover, "detectable_by_cameras": v.detectable,
                 "found": v.id in sim.found_at} for v in w.victims]
        return out

    def camera_png(self, rid: int, kind: str = "both") -> bytes:
        """What robot ``rid`` sees. kind: colour | thermal | both (side by side)."""
        from PIL import Image, ImageDraw

        from .sensors import render
        sim, w = self.sim, self.sim.world
        mode = sim.cfg.vision
        if mode == "none":
            raise ValueError("these robots carry no camera")
        if kind not in ("colour", "thermal", "both"):
            raise ValueError("kind must be colour, thermal or both")
        r = sim.robots[rid % len(sim.robots)]
        frame, heat = r.last_frame, r.last_heat
        if frame is None or tuple(frame.pose[:2]) != sim.pos(r):
            x, y = sim.pos(r)
            frame = render(w, x, y, r.heading, np.random.default_rng(sim.t))
            heat = sim.detector.heatmap(frame) if sim.detector else None
        H, W = frame.depth.shape
        rgb = (frame.rgb * 255).astype(np.float32)
        therm = palette(to_unit(frame.thermal, w.ambient)).astype(np.float32)
        ai_colour, ai_thermal = mode in ("cnn", "fusion"), mode in ("thermal", "fusion")
        if heat is not None:      # the detector's probability map, tinted on the image(s) it looks at
            up = np.kron(heat, np.ones((H // heat.shape[0], W // heat.shape[1])))[..., None]
            if ai_colour:
                rgb = rgb * (1 - 0.55 * up) + np.array([255, 40, 40]) * 0.55 * up
            if ai_thermal:
                therm = therm * (1 - 0.5 * up) + np.array([40, 255, 120]) * 0.5 * up
        panels = {"colour": [("colour", rgb, ai_colour)], "thermal": [("thermal", therm, ai_thermal)],
                  "both": [("colour", rgb, ai_colour), ("thermal", therm, ai_thermal)]}[kind]
        gap, scale = 2, 5 if kind == "both" else 6
        strips = []
        for i, (_, img, _) in enumerate(panels):
            strips += ([np.full((H, gap, 3), 24.0)] if i else []) + [img]
        img = np.concatenate(strips, axis=1)
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((img.shape[1] * scale, H * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        for i, (name, _, ai_reads_it) in enumerate(panels):
            off = i * (W + gap) * scale
            if heat is not None and ai_reads_it:
                b = scale * (H // heat.shape[0])
                for by, bx in np.argwhere(heat >= sim.cfg.detect_threshold):
                    d.rectangle([off + bx * b, by * b, off + (bx + 1) * b - 1, (by + 1) * b - 1], outline=(255, 230, 0), width=2)
            if kind == "both":
                d.text((off + 5, 4), f"{name} camera", fill="white")
            if name != "thermal":
                continue
            # hottest spot and its temperature, and the temperature scale
            iy, ix = np.unravel_index(int(np.argmax(frame.thermal)), frame.thermal.shape)
            cx, cy = off + (ix + 0.5) * scale, (iy + 0.5) * scale
            for x0, y0, x1, y1 in ((-9, 0, -3, 0), (3, 0, 9, 0), (0, -9, 0, -3), (0, 3, 0, 9)):
                d.line([cx + x0, cy + y0, cx + x1, cy + y1], fill="white", width=1)
            tx = cx + 10 if cx < off + W * scale - 60 else cx - 58
            d.text((tx, cy - 14 if cy > 20 else cy + 6), f"{float(frame.thermal[iy, ix]):.1f} °C", fill="white")
            bar = Image.fromarray(palette(np.linspace(0, 1, 100))[None].repeat(6, axis=0))
            im.paste(bar, (off + W * scale - 106, H * scale - 12))
            d.text((off + W * scale - 106, H * scale - 24), f"{w.ambient - SCALE_BELOW:.0f} °C", fill="white")
            d.text((off + W * scale - 34, H * scale - 24), f"{SCALE_TOP:.0f} °C", fill="white")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def _obstacles(q, building: str):
    """Open test ground: the user's obstacles, "kind,x,y,w,h;..." in cells ("" = none at all, absent = default)."""
    if building != "plain" or "obstacles" not in q:
        return None
    out = []
    for part in q["obstacles"][0].split(";"):
        bits = part.split(",")
        if len(bits) != 5 or bits[0] not in PLAIN_KINDS:
            continue
        x, y, w, h = (int(float(b)) for b in bits[1:])
        if 1 <= w <= 12 and 1 <= h <= 12:
            out.append((bits[0], x, y, w, h))
    return tuple(out[:80])


def _cfg_from(q, base: RescueConfig) -> RescueConfig:
    g = lambda k, cast, dflt: cast(q[k][0]) if k in q else dflt
    strategy = g("strategy", str, base.strategy)
    # the dashboard sends one choice, "sensors"; "vision" and "lidar" remain for presets in the address bar
    vision, lidar = base.vision, base.lidar
    if "sensors" in q:
        if q["sensors"][0] not in SENSOR_SETS:
            raise ValueError("sensors must be lidar, cameras or both")
        vision, lidar = SENSOR_SETS[q["sensors"][0]]
    vision = g("vision", str, vision)
    lidar = g("lidar", str, "1" if lidar else "0") == "1"
    if strategy not in STRATEGIES or vision not in VISION_MODES:
        raise ValueError("bad strategy or vision")
    building = g("building", str, base.building)
    damage = g("damage", str, base.damage)
    if building not in BUILDINGS or damage not in DAMAGE:
        raise ValueError("bad building or damage level")
    seed = g("seed", int, base.seed)
    known = g("known", str, "1" if base.victims_known else "0") == "1"
    # (the dashboard no longer offers limited radio; the CLI still does: --comm-range)
    # when nobody knows how many people are inside, the simulator hides a random number (4-12)
    n = g("victims", int, base.n_victims) if known else int(np.random.default_rng(seed + 99).integers(4, 13))
    clamp = lambda v, lo, hi: max(lo, min(hi, v))
    planner = g("planner", str, base.planner)
    coverage = g("coverage", str, base.coverage)
    each = tuple(clamp(float(v), 0.0, 500.0) for v in g("battery_each", str, "").split(",") if v.strip())
    avoidance = g("avoidance", str, base.avoidance)
    if planner not in PLANNERS or coverage not in PATTERNS or avoidance not in ("none", "dwa"):
        raise ValueError("bad planner, coverage pattern or obstacle avoidance")
    run = g("run", str, "")
    return replace(base, strategy=strategy, vision=vision, building=building, damage=damage,
                   n_robots=max(1, min(8, g("robots", int, base.n_robots))),
                   n_victims=max(1, min(16, n)), victims_known=known, lidar=lidar,
                   comm_range=0.0, seed=seed,
                   planner=planner, coverage=coverage, avoidance=avoidance,
                   battery_each=each, recharge=g("recharge", str, "1" if base.recharge else "0") == "1",
                   drain=g("recharge", str, "") == "drain", obstacles=_obstacles(q, building),
                   speed=clamp(g("speed", float, base.speed), 0.2, 1.5),
                   battery_wh=clamp(g("battery", float, base.battery_wh), 0.0, 500.0),
                   camera_range=clamp(g("camrange", float, base.camera_range), 2.0, 8.0),
                   lidar_range=clamp(g("lidarrange", float, base.lidar_range), 3.0, 20.0),
                   revisit_penalty=clamp(g("revisit", float, base.revisit_penalty), 0.0, 2.0),
                   deadend_recovery=g("deadend", str, "1" if base.deadend_recovery else "0") == "1",
                   run_seed=int(run) if run.lstrip("-").isdigit() else None)


def _compare_one(cfg: RescueConfig) -> dict:
    """One mission of a comparison, run to the end (top level: runs in a worker process)."""
    from .simulator import Simulator
    sim = Simulator(cfg).run()
    m = summarize(sim)
    keep = ("mission_time", "coverage", "mapped", "found", "false_alarms", "victims", "victims_detectable",
            "detection_rate", "distance_m", "energy_wh", "collisions", "robot_conflicts", "repeated_visits",
            "repeat_ratio", "dead_ends", "replans", "routes_planned", "plan_ms_mean", "plan_work_mean",
            "batteries_emptied", "time_90pct_coverage", "time_first_victim")
    out = {k: m[k] for k in keep}
    out["all_home"] = all(r.state == "done" and not r.depleted for r in sim.robots)
    return out


COMPARE_DIMENSIONS = {"planner": PLANNERS, "coverage": PATTERNS, "avoidance": ("none", "dwa"), "strategy": STRATEGIES}


class Compare:
    """Runs the same mission (same building, same random choices) once per option of one setting,
    in parallel worker processes, so algorithms can be compared under identical conditions."""

    def __init__(self, workers: int | None = None):
        import os
        self.lock, self.jobs, self.pool = threading.Lock(), {}, None
        # parallel missions: RESCUE_WORKERS, or up to 4 (fewer on a small machine)
        self.workers = workers or int(os.environ.get("RESCUE_WORKERS", 0)) or max(1, min(4, os.cpu_count() or 1))

    def start(self, cfg: RescueConfig, dimension: str, fast: bool) -> str:
        from concurrent.futures import ProcessPoolExecutor
        if dimension not in COMPARE_DIMENSIONS:
            raise ValueError(f"can compare only {', '.join(COMPARE_DIMENSIONS)}")
        if fast and cfg.vision != "none":
            cfg = replace(cfg, vision="ideal")                    # perfect eyes: same navigation, ~6x faster
        if cfg.run_seed is None:
            cfg = replace(cfg, run_seed=cfg.seed)                 # the same random choices for every option
        options = COMPARE_DIMENSIONS[dimension]
        with self.lock:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.workers)
            job = f"j{len(self.jobs) + 1}"
            futures = {o: self.pool.submit(_compare_one, replace(cfg, **{dimension: o})) for o in options}
            self.jobs[job] = {"dimension": dimension, "fast": fast, "futures": futures, "started": time.time(),
                              "settings": {k: getattr(cfg, k) for k in ("building", "damage", "seed", "run_seed", "n_robots",
                                                                        "strategy", "planner", "coverage", "avoidance",
                                                                        "vision", "lidar", "speed", "battery_wh",
                                                                        "battery_each", "recharge")}}
        return job

    def status(self, job: str) -> dict:
        j = self.jobs.get(job)
        if j is None:
            raise ValueError("unknown comparison")
        rows, done = [], 0
        for o, f in j["futures"].items():
            if f.done():
                done += 1
                err = f.exception()
                rows.append({"option": o, "error": None if err is None else str(err), **({} if err else f.result())})
            else:
                rows.append({"option": o, "pending": True})
        return {"dimension": j["dimension"], "fast": j["fast"], "done": done, "total": len(rows), "rows": rows,
                "settings": j["settings"], "seconds": round(time.time() - j["started"], 1)}


class Sessions:
    """One mission per browser tab, the least recently used one is dropped when there are too many."""

    def __init__(self, base: RescueConfig):
        self.base, self.lock, self.by_id = base, threading.Lock(), {}

    def get(self, sid: str, cfg: RescueConfig | None = None) -> Session:
        with self.lock:
            s = self.by_id.pop(sid, None)
            if s is None:
                s = Session(cfg or self.base)
            elif cfg is not None:
                with s.lock:
                    s.reset(cfg)
            self.by_id[sid] = s                          # most recently used last
            while len(self.by_id) > MAX_SESSIONS:
                self.by_id.pop(next(iter(self.by_id)))
            return s


def make_handler(sessions: Sessions, base: RescueConfig, compare: "Compare | None" = None):
    compare = compare or Compare()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, status=200, extra=()):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=200, extra=()):
            self._send(json.dumps(_clean(obj), allow_nan=False, indent=None if not extra else 2).encode(),
                       "application/json", status, extra)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            sid = q.get("sid", ["default"])[0][:40]
            try:
                session = None if not url.path.startswith("/api/") or url.path in ("/api/reset", "/api/compare") \
                    else sessions.get(sid)
                pr = q.get("plan", [""])[0]
                plan_robot = int(pr) if pr.isdigit() else None
                if url.path in ("/", "/index.html"):
                    self._send((WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path.startswith(STATIC_DIRS):
                    f = (WEB / url.path.lstrip("/")).resolve()
                    if WEB.resolve() not in f.parents or not f.is_file() or f.suffix not in STATIC_TYPES:
                        return self._json({"error": "not found"}, 404)
                    self._send(f.read_bytes(), STATIC_TYPES[f.suffix])
                elif url.path == "/api/reset":
                    session = sessions.get(sid, _cfg_from(q, base))
                    with session.lock:
                        self._json({"world": session.world_payload(), "state": session.state_payload()})
                elif url.path == "/api/world":
                    with session.lock:
                        self._json({"world": session.world_payload(), "state": session.state_payload()})
                elif url.path == "/api/reveal":
                    with session.lock:
                        session.reveal = True
                        self._json(session.state_payload())
                elif url.path == "/api/step":
                    n = max(1, min(MAX_STEPS, int(q.get("n", ["1"])[0])))
                    with session.lock:
                        for _ in range(n):
                            session.sim.step()
                        self._json(session.state_payload(plan_robot))
                elif url.path == "/api/state":
                    with session.lock:
                        self._json(session.state_payload(plan_robot))
                elif url.path == "/api/compare":
                    if "job" in q:
                        self._json(compare.status(q["job"][0]))
                    else:
                        job = compare.start(_cfg_from(q, base), q.get("vary", ["planner"])[0], q.get("fast", ["1"])[0] == "1")
                        self._json({"job": job})
                elif url.path == "/api/camera":
                    with session.lock:
                        self._send(session.camera_png(int(q.get("robot", ["0"])[0]), q.get("kind", ["both"])[0]), "image/png")
                elif url.path == "/api/victim_map":
                    with session.lock:
                        name = f"victim_map_{session.sim.cfg.building}_t{session.sim.t}s.json"
                        self._json(session.victim_map(), extra=[("Content-Disposition", f'attachment; filename="{name}"')]
                                   if "download" in q else ())
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    return H


def serve(base: RescueConfig, host: str = "127.0.0.1", port: int = 8001) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(Sessions(base), base, Compare()))
    print(f"Search-and-rescue dashboard running at http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
