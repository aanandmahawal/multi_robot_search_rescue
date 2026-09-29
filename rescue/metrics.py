"""How well did the team do? Every number here is computed against the ground truth."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .simulator import Simulator

MATCH_RADIUS = 1.5


def _first(history, key, value):
    return next((h["t"] for h in history if h[key] >= value), None)


def summarize(sim: "Simulator") -> dict:
    world, cfg, h = sim.world, sim.cfg, sim.history
    n = len(world.victims)
    conf = sim.confirmed_locations()
    matched = sim.match(conf)
    true_pos = len(matched)
    vic = {v.id: v for v in world.victims}
    errors = [np.hypot(vic[k].x - x, vic[k].y - y) for k, (x, y) in matched.items()]
    # real victims the team did not confirm but still lists as "possible victim here"
    open_spots = [(c.x, c.y) for r in sim.robots for c in r.registry.candidates if c.status == "candidate"]
    flagged = sum(1 for v in world.victims if v.id not in sim.found_at and
                  any(np.hypot(v.x - x, v.y - y) <= MATCH_RADIUS for x, y in open_spots))
    seen_by = np.sum([r.seen & world.reachable for r in sim.robots], axis=0)
    covered = seen_by > 0
    detectable = [v for v in world.victims if v.detectable]
    thin = [v for v in world.victims if v.cover == "thin"]
    good = set(map(tuple, matched.values()))                 # confirmed locations that are real victims
    kinds = [sim.truth_kind_at(x, y) for x, y, *_ in conf if (x, y) not in good]
    return {
        "strategy": cfg.strategy,
        "building": cfg.building,
        "damage": cfg.damage,
        "victims_known": cfg.victims_known,
        "sensors": cfg.sensors,
        "vision": cfg.vision,
        "lidar": cfg.lidar,
        "robots": cfg.n_robots,
        "comm_range": cfg.comm_range,
        "seed": cfg.seed,
        "victims": n,
        "victims_detectable": len(detectable),               # not buried under thick rubble
        "victims_buried_deep": n - len(detectable),          # no heat signature: no camera can find them
        "victims_thin_cover": len(thin),                     # faint heat signature: thermal camera only
        "found": len(sim.found_at),
        "found_thin_cover": sum(v.id in sim.found_at for v in thin),
        "recall": len(sim.found_at) / max(1, n),
        "recall_detectable": len(sim.found_at) / max(1, len(detectable)),
        "confirmed_reports": len(conf),                      # people the team reports = found + false alarms
        "repeated_reports": sim.repeated_reports(),          # second reports of the same person (merged, not counted)
        "precision": true_pos / max(1, len(conf)),
        "false_alarms": len(conf) - true_pos,
        "false_alarms_warm": kinds.count("warm"),            # heaters, pets, hot water, engines
        "false_alarms_lookalike": kinds.count("lookalike"),  # jackets, bags, boxes...
        "too_hot_dropped": sum(r.too_hot for r in sim.robots),
        "flagged_unconfirmed": flagged,
        "localization_error_m": float(np.mean(errors)) if errors else None,
        "time_first_victim": min(sim.found_at.values()) if sim.found_at else None,
        "time_50pct_victims": _first(h, "found", n / 2),
        "time_all_victims": _first(h, "found", n),                                  # never reached if someone is buried deep
        "time_all_detectable": _first(h, "found", len(detectable)) if detectable else None,   # every victim a camera could see
        "reported": len(sim.reported_at),
        "time_all_reported": max(sim.reported_at.values()) if len(sim.reported_at) == n and n else None,
        "goal_overlap": sim.overlap_pairs / max(1, sim.overlap_samples),
        "time_90pct_coverage": _first(h, "coverage", 0.9),
        "coverage": sim.coverage(),
        "time_90pct_mapped": _first(h, "mapped", 0.9),
        "mapped": sim.mapped(),
        "bumps": sum(r.bumps for r in sim.robots),
        "mission_time": sim.t,
        "distance_m": sum(r.distance for r in sim.robots),
        "redundancy": float(seen_by[covered].mean()) if covered.any() else 0.0,
        "verify_trips": sum(r.verify_trips for r in sim.robots),
        "reports": len(sim.reports),
        "map_syncs": sim.messages,
        # navigation (planners.py, coverage.py, motion.py)
        "planner": cfg.planner,
        "coverage_pattern": cfg.coverage,
        "avoidance": cfg.avoidance,
        "speed": cfg.speed,
        "battery_wh": cfg.battery_wh,
        "energy_wh": sum(r.energy_j for r in sim.robots) / 3600.0,
        "energy_wh_per_robot": [round(r.energy_j / 3600.0, 2) for r in sim.robots],
        "batteries_emptied": sum(r.depleted for r in sim.robots),
        "collisions": sum(r.bumps for r in sim.robots),                 # bumper contacts with obstacles
        "robot_conflicts": sum(r.conflicts for r in sim.robots),        # waits because a teammate was in the way
        "moves": sum(r.moves for r in sim.robots),
        "repeated_visits": sim.team_repeat,                            # moves into a cell some robot had driven through
        "repeat_ratio": sim.team_repeat / max(1, sum(r.moves for r in sim.robots)),
        "dead_ends": sum(r.deadends for r in sim.robots),
        "replans": sum(r.replans for r in sim.robots),                   # routes replanned because of a new obstacle
        "routes_planned": sum(r.plans for r in sim.robots),
        "plan_ms_mean": sum(r.plan_ms for r in sim.robots) / max(1, sum(r.plans for r in sim.robots)),
        "plan_work_mean": sum(r.plan_work for r in sim.robots) / max(1, sum(r.plans for r in sim.robots)),
        "detection_rate": len(sim.found_at) / max(1, len(detectable)),    # victims found / victims a camera could find
    }


def format_summary(m: dict) -> str:
    fmt = lambda v, suf="s": "not reached" if v is None else f"{v}{suf}"
    rows = [
        ("Building", f"{m['damage']} damage {m['building']}, "
                     f"{'victim count known' if m['victims_known'] else 'victim count unknown'}"),
        ("Team", f"{m['robots']} robots, strategy '{m['strategy']}', "
                 + {"lidar": "LiDAR only (no camera)", "cameras": f"cameras only (vision '{m['vision']}', no LiDAR)",
                    "both": f"LiDAR + cameras (vision '{m['vision']}')"}[m["sensors"]]
                 + f", radio {'unlimited' if not m['comm_range'] else str(m['comm_range']) + ' m'}"),
        ("Confirmed by the robots", f"{m['confirmed_reports']} people = {m['found']} real victims + {m['false_alarms']} false alarms "
                                    f"({m['repeated_reports']} repeated reports of the same person merged)"),
        ("Victims found", f"{m['found']} / {m['victims']}  (recall {100 * m['recall']:.0f}%)"),
        ("Buried under thick rubble", f"{m['victims_buried_deep']} victims with no heat signature: no camera can detect them "
                                      f"(recall on the {m['victims_detectable']} detectable victims: {100 * m['recall_detectable']:.0f}%)"),
        ("Under thin debris", f"{m['found_thin_cover']} / {m['victims_thin_cover']} found (faint heat signature only)"),
        ("False alarms confirmed", f"{m['false_alarms']}  (precision {100 * m['precision']:.0f}%): "
                                   f"{m['false_alarms_warm']} on warm objects, {m['false_alarms_lookalike']} on look-alikes"),
        ("Too hot to be a person", f"{m['too_hot_dropped']} detections dropped (heaters, engines)"),
        ("Missed but flagged", f"{m['flagged_unconfirmed']} real victims left as 'possible victim, check by hand'"),
        ("Location error", "-" if m["localization_error_m"] is None else f"{m['localization_error_m']:.2f} m"),
        ("First victim found", fmt(m["time_first_victim"])),
        ("Half of victims found", fmt(m["time_50pct_victims"])),
        ("All detectable victims found", fmt(m["time_all_detectable"])),
        ("Robots aiming at the same spot", f"{100 * m['goal_overlap']:.0f}% of robot pairs (lower = better teamwork)"),
        ("90% of building mapped", fmt(m["time_90pct_mapped"])),
        ("90% of building searched", fmt(m["time_90pct_coverage"])),
        ("Building mapped / searched", f"{100 * m['mapped']:.1f}% on the map, {100 * m['coverage']:.1f}% looked at by a camera"),
        ("Bumper hits", f"{m['bumps']} (obstacles too low for the LiDAR that no camera had seen yet)"),
        ("Mission time", f"{m['mission_time']}s"),
        ("Distance driven (team)", f"{m['distance_m']:.0f} m"),
        ("Redundancy", f"{m['redundancy']:.2f} robots saw each searched cell (lower = better teamwork)"),
        ("Verification trips", f"{m['verify_trips']}"),
        ("Navigation", f"planner {m['planner']}, coverage {m['coverage_pattern']}, avoidance {m['avoidance']}, "
                       f"{m['speed']} m/s" + (f", battery {m['battery_wh']:g} Wh" if m["battery_wh"] else "")),
        ("Energy used (team)", f"{m['energy_wh']:.1f} Wh" + (f", {m['batteries_emptied']} battery empty" if m["batteries_emptied"] else "")),
        ("Repeated visits", f"{100 * m['repeat_ratio']:.0f}% of moves over cells already driven through"),
        ("Waits / dead ends / replans", f"{m['robot_conflicts']} waits for a teammate, {m['dead_ends']} dead ends, "
                                        f"{m['replans']} routes replanned for a new obstacle"),
        ("Route planning", f"{m['routes_planned']} routes, {m['plan_ms_mean']:.1f} ms and {m['plan_work_mean']:.0f} "
                           f"cells / samples / ant steps each on average"),
    ]
    if m["comm_range"]:
        rows.insert(8, ("Reported to base", f"{m['reported']} victims (all by {fmt(m['time_all_reported'])})"))
    w = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k:<{w}}  {v}" for k, v in rows)


def run_and_summarize(cfg) -> dict:
    """Top-level so benchmark worker processes can import it. Includes the time series."""
    from .simulator import Simulator
    sim = Simulator(cfg).run()
    out = summarize(sim)
    out["curve_found"] = [h["found"] for h in sim.history]
    out["curve_coverage"] = [h["coverage"] for h in sim.history]
    return out
