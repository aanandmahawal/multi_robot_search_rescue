"""Check the navigation algorithms on the open test ground (building "plain").

    python scripts/check_algorithms.py [--robots 1] [--planner astar] [--seed 0]

1. Route planners, on the true map of the test ground between 40 random pairs of cells:
   Dijkstra must find the exact shortest route (checked against an independent distance field),
   A* the same length while expanding fewer cells, and RRT* and the ant colony valid routes
   (free cells, no corner cutting) that are never shorter than the optimum.
2. Coverage patterns: one mission per pattern with perfect vision (so only navigation is
   measured). For each robot: its pattern, the waypoints it reached or skipped, how far its
   track strayed from the pattern while following it, and the share of the ground searched
   when the patterns were finished. Written to results/algorithms/check.md and a picture of every
   robot's pattern (dashed) and track (solid): results/algorithms/coverage.png.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rescue import coverage                                   # noqa: E402
from rescue.config import RescueConfig                        # noqa: E402
from rescue.mapping import FREE, OBSTACLE, dijkstra           # noqa: E402
from rescue.planners import PLANNERS, plan, valid_route       # noqa: E402
from rescue.simulator import Simulator                        # noqa: E402
from rescue.world import generate                             # noqa: E402

OUT = Path("results/algorithms")
COLORS = ["#4f9cf9", "#f0883e", "#c084fc", "#2dd4bf", "#f472b6", "#facc15", "#60a5fa", "#a3e635"]


def check_planners(cfg: RescueConfig, pairs: int = 40) -> list[str]:
    w = generate(cfg)
    known = np.where(w.blocked, OBSTACLE, FREE).astype(np.int8)
    cells = np.argwhere(w.reachable)[:, ::-1]
    rng = np.random.default_rng(1)
    rows = {a: {"ratio": [], "work": [], "ms": [], "valid": 0, "fallback": 0} for a in PLANNERS}
    for _ in range(pairs):
        s, g = (tuple(map(int, cells[rng.integers(len(cells))])) for _ in range(2))
        if s == g:
            continue
        dist, _ = dijkstra(known, s)
        opt = float(dist[g[1], g[0]])
        for a in PLANNERS:
            p = plan(a, known, s, g, np.random.default_rng(7), reach=w.reachable)
            ok = bool(p.path) and p.path[-1] == g and valid_route(known == FREE, s, p.path) and p.length >= opt - 1e-6
            r = rows[a]
            r["valid"] += ok
            r["fallback"] += bool(p.note)
            r["ratio"].append(p.length / opt)
            r["work"].append(p.expanded)
            r["ms"].append(p.ms)
            if a == "dijkstra":
                assert abs(p.length - opt) < 1e-6, "Dijkstra is not optimal"
            if a == "astar":
                assert abs(p.length - opt) < 1e-6, "A* is not optimal"
    n = len(rows["dijkstra"]["ratio"])
    out = [f"### Route planners ({n} random routes on the true map of the test ground)", "",
           "| planner | valid routes | length / optimum (mean, worst) | work (cells / samples / ant steps) | time (ms) | fell back to A* |",
           "|---|---|---|---|---|---|"]
    for a in PLANNERS:
        r = rows[a]
        out.append(f"| {a} | {r['valid']}/{n} | {np.mean(r['ratio']):.3f}, {np.max(r['ratio']):.3f} | "
                   f"{np.mean(r['work']):.0f} | {np.mean(r['ms']):.1f} | {r['fallback']} |")
    return out


def _dist_to_polyline(p, pts) -> float:
    best = np.inf
    for a, b in zip(pts, pts[1:]):
        a, b = np.asarray(a), np.asarray(b)
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
        best = min(best, float(np.hypot(*(a + t * ab - p))))
    return best


def run_pattern(cfg: RescueConfig) -> tuple[Simulator, float | None]:
    sim = Simulator(cfg)
    at_end = None
    while not sim.finished:
        sim.step()
        if at_end is None and all(r.sweep is None or r.sweep.done for r in sim.robots) and cfg.coverage != "frontier":
            at_end = sim.coverage()
    return sim, at_end


def check_coverage(cfg: RescueConfig) -> tuple[list[str], list]:
    out = [f"### Coverage patterns ({cfg.n_robots} robot{'s' if cfg.n_robots > 1 else ''}, planner {cfg.planner}, "
           f"lane spacing {coverage.lane_width(cfg):.2f} m, perfect vision)", "",
           "| pattern | mission (s) | searched when patterns done | searched at the end | driven (m) | repeat moves |",
           "|---|---|---|---|---|---|"]
    detail, sims = [], []
    for pattern in coverage.PATTERNS:
        sim, at_end = run_pattern(replace(cfg, coverage=pattern))
        sims.append((pattern, sim))
        out.append(f"| {pattern} | {sim.t} | {'-' if at_end is None else f'{100 * at_end:.1f}%'} | "
                   f"{100 * sim.coverage():.1f}% | {sum(r.distance for r in sim.robots):.0f} | {sim.team_repeat} |")
        for r in sim.robots:
            sw = r.sweep
            if sw is None:
                continue
            c = sim.cfg.cell
            track = [np.array([(x + 0.5) * c, (y + 0.5) * c]) for x, y in r.track[sw.start_move or 0:(sw.end_move or len(r.track)) + 1]]
            dev = np.array([_dist_to_polyline(p, sw.corners) for p in track]) if track else np.zeros(1)
            skipped = ", ".join(f"#{i + 1} {why}" for i, why in sw.skipped) or "none"
            detail.append(f"| {pattern} | R{r.id} | {sw.region + 1} | {sw.describe} | "
                          f"{coverage.path_length(sw.corners):.1f} | {sw.reached}/{len(sw.cells)} | {skipped} | "
                          f"{np.mean(dev):.2f} / {np.percentile(dev, 95):.2f} / {dev.max():.2f} | {sw.end_t} |")
    out += ["", "| pattern | robot | region | pattern | length (m) | waypoints reached | skipped | "
                "distance from the pattern (m): mean / 95% / max | pattern done at (s) |",
            "|---|---|---|---|---|---|---|---|---|", *detail]
    return out, sims


def draw(sims, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(sims), figsize=(6.2 * len(sims), 5.2))
    for ax, (pattern, sim) in zip(np.atleast_1d(axes), sims):
        w, c = sim.world, sim.cfg.cell
        ax.imshow(np.where(w.blocked, 0.25, 0.95), cmap="gray", vmin=0, vmax=1,
                  extent=(0, w.W * c, w.H * c, 0), interpolation="nearest")
        for b in sim.region_boxes or []:
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ls=":", lw=0.8, ec="#888"))
        for r in sim.robots:
            col = COLORS[r.id % len(COLORS)]
            if r.sweep is not None:
                xs, ys = zip(*r.sweep.corners)
                ax.plot(xs, ys, "--", color=col, lw=1.2, alpha=0.9)
                ax.plot(xs[0], ys[0], "o", color=col, ms=5)
                for i, _ in r.sweep.skipped:
                    ax.plot(*r.sweep.points[i], "x", color="red", ms=6)
            tx = [(x + 0.5) * c for x, _ in r.track]
            ty = [(y + 0.5) * c for _, y in r.track]
            ax.plot(tx, ty, "-", color=col, lw=1.6, alpha=0.75)
        ax.set_title(f"{pattern}: {sim.t} s, {100 * sim.coverage():.0f}% searched")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle("Dashed: each robot's coverage pattern (o = start, x = skipped waypoint). Solid: where it drove.")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robots", type=int, default=1)
    ap.add_argument("--planner", default="dijkstra", choices=PLANNERS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = RescueConfig(building="plain", vision="ideal", n_robots=a.robots, planner=a.planner,
                       seed=a.seed, max_steps=2500)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = check_planners(cfg)
    print("\n".join(lines), "\n", flush=True)
    cov, sims = check_coverage(cfg)
    print("\n".join(cov), flush=True)
    draw(sims, OUT / "coverage.png")
    (OUT / "check.md").write_text("\n".join(lines + [""] + cov) + "\n", encoding="utf-8")
    print(f"\nwritten: {OUT / 'check.md'}, {OUT / 'coverage.png'}")


if __name__ == "__main__":
    main()
