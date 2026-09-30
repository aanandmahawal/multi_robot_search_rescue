"""How reliable is the victim search? Many missions, every verdict checked against the ground truth.

    python scripts/reliability.py [--seeds 2] [--robots 4]

For every sensor set (colour camera AI, thermal camera AI, both fused) the same missions are run
in all six damaged buildings. Each place the team *confirmed* as a person is classified by what
is really there: a real person, a person already counted (reported a second time), a warm object
(dog, heater, laptop, hot water, sun-warmed rubble, car engine), a look-alike (jacket, bag, box,
beam, barrel) or nothing. Each sighting the team *rejected* is classified the same way, so a real
person wrongly rejected shows up. Victims never found are split by how they lay (in the open,
partly buried, under thin debris, under thick rubble where no camera can see them). Finally the
robots' registries are compared every second: two robots holding the same reports must never
disagree on a verdict.

Written to results/reliability/report.md.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rescue.config import RescueConfig          # noqa: E402
from rescue.simulator import MATCH_RADIUS, Simulator   # noqa: E402
from rescue.world import BUILDINGS             # noqa: E402

OUT = Path("results/reliability")
VISIONS = {"cnn": "colour camera AI", "thermal": "thermal camera AI", "fusion": "colour + thermal fused"}


def what_is_there(sim: Simulator, x: float, y: float) -> str:
    """Fine-grained ground truth at a reported location."""
    w, c = sim.world, sim.cfg.cell
    if any(np.hypot(v.x - x, v.y - y) <= MATCH_RADIUS for v in w.victims):
        return "person"
    for d in w.thermal_decoys:
        if np.hypot(d.x - x, d.y - y) <= MATCH_RADIUS:
            return d.kind
    for f in w.furniture:
        if f.temp > 0 and any(np.hypot((cx + 0.5) * c - x, (cy + 0.5) * c - y) <= MATCH_RADIUS for cx, cy in f.cells()):
            return "car engine"
    for d in w.decoys:
        if np.hypot(d.x - x, d.y - y) <= MATCH_RADIUS:
            return d.kind
    return "nothing"


def run(cfg: RescueConfig) -> dict:
    sim = Simulator(cfg)
    conflicts = 0
    while not sim.finished:
        sim.step()
        views = [{(round(c.x, 6), round(c.y, 6)): c.status for c in r.registry.candidates} for r in sim.robots]
        conflicts += any(v != views[0] for v in views[1:])
    people = sim.confirmed_people()
    matched = sim.match([(c.x, c.y, 0) for c in people])
    real = {tuple(xy) for xy in matched.values()}
    confirmed = []
    for c in people:
        kind = what_is_there(sim, c.x, c.y)
        confirmed.append("person" if (c.x, c.y) in real else "person counted twice" if kind == "person" else kind)
    rejected = Counter()
    for c in sim.robots[0].registry.candidates:
        if c.status == "rejected":
            rejected[what_is_there(sim, c.x, c.y)] += 1
    missed = Counter(("under thick rubble" if v.cover == "thick" else "under thin debris" if v.cover == "thin"
                      else "partly buried" if v.cover == "partial" else "in the open")
                     for v in sim.world.victims if v.id not in sim.found_at)
    lying = Counter(("under thick rubble" if v.cover == "thick" else "under thin debris" if v.cover == "thin"
                     else "partly buried" if v.cover == "partial" else "in the open") for v in sim.world.victims)
    return {"vision": cfg.vision, "confirmed": Counter(confirmed), "rejected": rejected, "missed": missed,
            "lying": lying, "victims": len(sim.world.victims), "found": len(sim.found_at),
            "detectable": sum(v.detectable for v in sim.world.victims), "conflict_s": conflicts, "t": sim.t}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--robots", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=300)
    a = ap.parse_args()
    buildings = [b for b in BUILDINGS if b != "plain"]
    base = RescueConfig(n_robots=a.robots, strategy="coordinated", n_victims=10)
    jobs = [replace(base, vision=v, building=b, seed=a.first_seed + k) for v in VISIONS for b in buildings for k in range(a.seeds)]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run, jobs))

    n_missions = len(jobs) // len(VISIONS)
    lines = [f"# Reliability of the victim search", "",
             f"{n_missions} missions per sensor set ({len(buildings)} damaged buildings x {a.seeds} seeds), "
             f"{a.robots} robots, coordinated strategy, 10 victims, 12 look-alikes, 6 warm objects per building. "
             f"Every number is checked against the ground truth, which the robots never see.", ""]
    lines += ["## Summary", "",
              "| sensors | victims found / findable | confirmed reports | of them real | person counted twice | "
              "false alarms (nobody there) | precision | rejected sightings at a real person* | seconds with robots disagreeing |",
              "|---|---|---|---|---|---|---|---|---|"]
    for v, name in VISIONS.items():
        rs = [r for r in rows if r["vision"] == v]
        conf = sum((r["confirmed"] for r in rs), Counter())
        rej = sum((r["rejected"] for r in rs), Counter())
        total = sum(conf.values())
        false = total - conf["person"] - conf["person counted twice"]
        lines.append(f"| {name} | {sum(r['found'] for r in rs)} / {sum(r['detectable'] for r in rs)} | {total} | "
                     f"{conf['person']} | {conf['person counted twice']} | {false} | "
                     f"{100 * conf['person'] / max(1, total):.0f}% | {rej['person']} | {sum(r['conflict_s'] for r in rs)} |")
    lines += ["", "* usually a second, weaker sighting of a person who was confirmed from another sighting; the "
                  "missed victims are listed per sensor set below."]
    for v, name in VISIONS.items():
        rs = [r for r in rows if r["vision"] == v]
        conf = sum((r["confirmed"] for r in rs), Counter())
        rej = sum((r["rejected"] for r in rs), Counter())
        missed = sum((r["missed"] for r in rs), Counter())
        lying = sum((r["lying"] for r in rs), Counter())
        lines += ["", f"## {name}", "",
                  "What the confirmed reports really were: " + ", ".join(f"{k} {n}" for k, n in conf.most_common()) + ".", "",
                  "What the rejected sightings really were: " + (", ".join(f"{k} {n}" for k, n in rej.most_common()) or "none") + ".", "",
                  "Victims missed, by how they lay (missed / all): " +
                  ", ".join(f"{k} {missed[k]} / {lying[k]}" for k in ("in the open", "partly buried", "under thin debris",
                                                                       "under thick rubble") if lying[k]) + "."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
