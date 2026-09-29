"""Command line.

    python -m rescue serve                         # 3-D dashboard at http://localhost:8001
    python -m rescue simulate --strategy coordinated
    python -m rescue train                         # train the victim detector (CNN)
    python -m rescue benchmark --seeds 8           # compare strategies, write results/
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import SENSOR_SETS, VISION_MODES, RescueConfig
from .coordination import STRATEGIES
from .coverage import PATTERNS
from .planners import PLANNERS
from .metrics import format_summary, run_and_summarize, summarize
from .world import BUILDINGS, DAMAGE


def _cfg(a) -> RescueConfig:
    building = a.building if a.building != "all" else "office"
    vision, lidar = (a.vision if a.vision != "all" else "fusion"), not a.no_lidar
    if a.sensors:                                   # one word for both settings
        vision, lidar = SENSOR_SETS[a.sensors]
    return RescueConfig(n_robots=a.robots, n_victims=a.victims, seed=a.seed, vision=vision,
                        comm_range=a.comm_range, max_steps=a.max_steps, detector_path=a.detector,
                        building=building, damage=a.damage, victims_known=a.known, lidar=lidar,
                        planner=a.planner, coverage=a.coverage, avoidance=a.avoidance, speed=a.speed,
                        battery_wh=a.battery, obstacle_density=a.density, revisit_penalty=a.revisit,
                        deadend_recovery=not a.no_deadend, run_seed=a.run_seed)


def cmd_simulate(a):
    from .simulator import Simulator
    cfg = replace(_cfg(a), strategy=a.strategy)
    t0 = time.perf_counter()
    sim = Simulator(cfg).run()
    print(f"Simulated {sim.t} s of mission in {time.perf_counter() - t0:.1f} s of compute\n")
    if a.events:
        for t, e in sim.events:
            print(f"  [t={t:4d}s] {e}")
        print()
    print(format_summary(summarize(sim)))


def cmd_train(a):
    from .perception import MODALITY_NAMES, train_detectors
    modalities = ("cnn", "thermal", "fusion") if a.modality == "all" else (a.modality,)
    reports = train_detectors(_cfg(a), modalities, n_worlds=a.worlds, views=a.views, epochs=a.epochs)
    for m, r in reports.items():
        print(f"\n{MODALITY_NAMES[m].capitalize()} detector on images from disaster zones it never saw:")
        print(f"  images            train {r['train_images']}, test {r['test_images']} ({r['images_with_victims']} test images contain a victim)")
        print(f"  image precision   {100 * r['image_precision']:.1f}%   (when it says 'person here', how often it is right)")
        print(f"  image recall      {100 * r['image_recall']:.1f}%   (of images showing a person, how many it flags)")
        print(f"  image ROC-AUC     {r['image_auc']:.3f}")
        print(f"  heatmap AP        {r['block_average_precision']:.3f}")
        print(f"  saved -> {r['path']}  ({r['seconds']} s)")


def cmd_benchmark(a):
    base = _cfg(a)
    buildings = BUILDINGS if a.building == "all" else (base.building,)
    configs = [replace(base, strategy=s, building=b, seed=a.seed + k)
               for s in a.strategies for b in buildings for k in range(a.seeds)]
    visions = [v for v in VISION_MODES if v != "none"] if a.vision == "all" else (base.vision,)
    configs = [replace(c, vision=v) for c in configs for v in visions]
    print(f"Running {len(configs)} missions ({a.robots} robots, building={a.building}, vision={a.vision}, "
          f"radio={'unlimited' if not a.comm_range else a.comm_range})...", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(run_and_summarize, configs))
    print(f"done in {time.perf_counter() - t0:.0f} s\n")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in r.items() if not k.startswith("curve_")} for r in rows]
    with open(out / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    def cell(vals, fmt, pct=False):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        m, s = np.mean(vals), np.std(vals)
        return (fmt.format(100 * m) + "%") if pct else f"{fmt.format(m)} ± {fmt.format(s)}"

    pct = ("recall", "recall_detectable", "precision")
    cols = [("time_50pct_victims", "50% found (s)", "{:.0f}"), ("time_all_detectable", "all found (s)", "{:.0f}"),
            ("recall", "recall", "{:.0f}"), ("recall_detectable", "recall (detectable)", "{:.0f}"),
            ("precision", "precision", "{:.0f}"), ("false_alarms_warm", "false alarms: warm objects", "{:.1f}"),
            ("false_alarms_lookalike", "false alarms: look-alikes", "{:.1f}"),
            ("time_90pct_mapped", "90% mapped (s)", "{:.0f}"),
            ("time_90pct_coverage", "90% searched (s)", "{:.0f}"), ("distance_m", "distance (m)", "{:.0f}"),
            ("redundancy", "redundancy", "{:.2f}")]
    groups = [("vision", v, "strategy", s) for v in visions for s in a.strategies]
    lines = ["| vision | strategy | " + " | ".join(c[1] for c in cols) + " | all found in |",
             "|---|---|" + "---|" * (len(cols) + 1)]
    for _, v, _, s in groups:
        sub = [r for r in rows if r["strategy"] == s and r["vision"] == v]
        cells = [cell([r[k] for r in sub], f, pct=k in pct) for k, _, f in cols]
        done = sum(r["time_all_detectable"] is not None for r in sub)
        lines.append(f"| {v} | {s} | " + " | ".join(cells) + f" | {done}/{len(sub)} runs |")
    table = "\n".join(lines)
    print(table)
    (out / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    _plot(rows, a.strategies, out / "benchmark.png", a.max_steps, visions)
    print(f"\nresults -> {out / 'benchmark.csv'}, {out / 'benchmark.md'}, {out / 'benchmark.png'}")


def _plot(rows, strategies, path, horizon, visions=("cnn",)):
    """One curve per strategy (single vision mode) or per vision mode (several modes, first strategy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"random": "#8b98a5", "greedy": "#e8743b", "partition": "#3ba55d", "coordinated": "#2f7ed8",
              "cnn": "#e8743b", "thermal": "#c0392b", "fusion": "#3ba55d", "ideal": "#8b98a5"}
    by_vision = len(visions) > 1
    series = [(v, [r for r in rows if r["vision"] == v and r["strategy"] == strategies[0]]) for v in visions] if by_vision \
        else [(s, [r for r in rows if r["strategy"] == s]) for s in strategies]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, sub in series:
        if not sub:
            continue
        for key, ax, norm in (("curve_found", axes[0], "victims_detectable"), ("curve_coverage", axes[1], None)):
            curves = np.zeros((len(sub), horizon))
            for i, r in enumerate(sub):
                c = np.array(r[key], dtype=float)
                if norm:
                    c = c / max(1, r[norm])
                curves[i, :len(c)] = c
                curves[i, len(c):] = c[-1] if len(c) else 0
            m = 100 * curves.mean(axis=0)
            ax.plot(np.arange(1, horizon + 1), m, label=name, color=colors.get(name), lw=2)
    axes[0].set_title("Victims found (% of the victims a camera could see)")
    axes[1].set_title("Building searched (% of reachable area)")
    for ax in axes:
        ax.set_xlabel("mission time (s)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title=f"vision ({strategies[0]} strategy)" if by_vision else "strategy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cmd_serve(a):
    from .server import serve
    serve(_cfg(a), port=a.port)


def main(argv=None):
    import sys
    if hasattr(sys.stdout, "reconfigure"):          # log lines contain "°C"; never crash on a narrow console encoding
        sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(prog="rescue", description="RescueThermal AI: multi-robot search-and-rescue simulator")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = RescueConfig()

    def common(sp):
        sp.add_argument("--robots", type=int, default=d.n_robots)
        sp.add_argument("--victims", type=int, default=d.n_victims)
        sp.add_argument("--seed", type=int, default=d.seed)
        sp.add_argument("--vision", default=d.vision, choices=[*VISION_MODES, "all"],
                        help="what the AI looks at: cnn = colour camera, thermal, fusion = both, ideal = perfect eyes "
                             "('all' = every mode, benchmark only)")
        sp.add_argument("--comm-range", type=float, default=d.comm_range, help="radio range in metres (0 = unlimited)")
        sp.add_argument("--max-steps", type=int, default=d.max_steps)
        sp.add_argument("--detector", default=d.detector_path)
        sp.add_argument("--building", default=d.building, choices=[*BUILDINGS, "all"],
                        help="building type ('all' = every type, benchmark only)")
        sp.add_argument("--damage", default=d.damage, choices=list(DAMAGE))
        sp.add_argument("--known", action="store_true", help="robots know how many victims are inside")
        sp.add_argument("--no-lidar", action="store_true", help="robots map with the depth camera only (no LiDAR)")
        sp.add_argument("--sensors", default=None, choices=list(SENSOR_SETS),
                        help="shortcut: lidar = laser only (maps, finds nobody), cameras = computer vision + thermal "
                             "imaging without LiDAR, both = everything (overrides --vision and --no-lidar)")
        # navigation (README section 9a)
        sp.add_argument("--planner", default=d.planner, choices=PLANNERS, help="route planner")
        sp.add_argument("--coverage", default=d.coverage, choices=PATTERNS, help="order of searching")
        sp.add_argument("--avoidance", default=d.avoidance, choices=("none", "dwa"), help="local obstacle avoidance")
        sp.add_argument("--speed", type=float, default=d.speed, help="driving speed in m/s")
        sp.add_argument("--battery", type=float, default=d.battery_wh, help="battery capacity in Wh (0 = unlimited)")
        sp.add_argument("--density", type=float, default=d.obstacle_density, help="rubble multiplier (1 = normal)")
        sp.add_argument("--revisit", type=float, default=d.revisit_penalty,
                        help="extra route cost per earlier visit of a cell (0 = shortest routes, 0.3 = avoid revisits)")
        sp.add_argument("--no-deadend", action="store_true", help="switch off dead-end / loop recovery")
        sp.add_argument("--run-seed", type=int, default=None,
                        help="vary the mission in the same building (sensor noise, tie-breaks, RRT*, ants)")

    s = sub.add_parser("simulate", help="run one mission and print the results")
    common(s)
    s.add_argument("--strategy", default=d.strategy, choices=STRATEGIES)
    s.add_argument("--events", action="store_true", help="print the mission log")
    s.set_defaults(func=cmd_simulate)

    t = sub.add_parser("train", help="train the victim detector(s) on rendered camera images")
    common(t)
    t.add_argument("--modality", default="all", choices=["cnn", "thermal", "fusion", "all"],
                   help="which detector to train: colour camera, thermal camera, both cameras, or all three")
    t.add_argument("--worlds", type=int, default=200)
    t.add_argument("--views", type=int, default=100)
    t.add_argument("--epochs", type=int, default=20)
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("benchmark", help="compare strategies over many random disaster zones")
    common(b)
    b.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    b.add_argument("--seeds", type=int, default=8)
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--out", default="results")
    b.set_defaults(func=cmd_benchmark)

    v = sub.add_parser("serve", help="3-D live dashboard in the browser")
    common(v)
    v.add_argument("--port", type=int, default=8001)
    v.set_defaults(func=cmd_serve)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
