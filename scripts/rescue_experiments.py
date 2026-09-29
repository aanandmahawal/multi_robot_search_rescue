"""Extra studies for the search-and-rescue README (ideal vision, so only teamwork is measured).

    python scripts/rescue_experiments.py

Team size: how does time-to-find scale with 1, 2, 4, 6, 8 robots (greedy vs coordinated),
averaged over all six building types?
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rescue.config import RescueConfig          # noqa: E402
from rescue.metrics import run_and_summarize    # noqa: E402

SEEDS = range(200, 212)
OUT = Path("results")


def mean_time(rows, key, cap=900):
    return float(np.mean([r[key] if r[key] is not None else cap for r in rows]))


def main():
    base = RescueConfig(vision="ideal")
    from rescue.world import BUILDINGS
    team = [(n, s, replace(base, strategy=s, n_robots=n, building=BUILDINGS[k % len(BUILDINGS)], seed=k))
            for n in (1, 2, 4, 6, 8) for s in ("greedy", "coordinated") for k in SEEDS]
    with ProcessPoolExecutor() as pool:
        team_rows = list(pool.map(run_and_summarize, [c for *_, c in team]))

    lines = ["### Team size (perfect vision, 12 buildings of all six types)", "",
             "| robots | greedy: all found (s) | coordinated: all found (s) | coordinated: redundancy | speed-up vs 1 robot |",
             "|---|---|---|---|---|"]
    solo = None
    for n in (1, 2, 4, 6, 8):
        g = [r for (nn, s, _), r in zip(team, team_rows) if nn == n and s == "greedy"]
        c = [r for (nn, s, _), r in zip(team, team_rows) if nn == n and s == "coordinated"]
        tc = mean_time(c, "time_all_detectable")
        solo = solo or tc
        lines.append(f"| {n} | {mean_time(g, 'time_all_detectable'):.0f} | {tc:.0f} | "
                     f"{np.mean([r['redundancy'] for r in c]):.2f} | {solo / tc:.1f}x |")
    lines += ["", "(Time to find every victim a camera could see; victims buried under thick rubble are excluded. "
              "Missions that never found them all count as 900 s, the time limit.)"]
    text = "\n".join(lines)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "experiments.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
