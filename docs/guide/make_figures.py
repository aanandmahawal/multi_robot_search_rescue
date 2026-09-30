"""Figures for the project guide (docs/guide). Every figure is computed from the simulator itself.

    python docs/guide/make_figures.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rescue import coverage, lidar                                      # noqa: E402
from rescue.config import RescueConfig                                  # noqa: E402
from rescue.mapping import FREE, OBSTACLE, UNKNOWN, frontiers          # noqa: E402
from rescue.planners import plan                                       # noqa: E402
from rescue.sensors import line_of_sight, render                       # noqa: E402
from rescue.simulator import Simulator                                 # noqa: E402
from rescue.thermal import palette, to_unit                            # noqa: E402
from rescue.world import BUILDINGS, generate                           # noqa: E402

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans", "savefig.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})
ROBOT = ["#2f7de0", "#e8702a", "#9b59d0", "#16a38a", "#e0508f", "#c9a400"]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def layer_image(w, walls=True, furniture=True, rubble=True, people=True):
    """RGB picture of a world's navigation grid, layer by layer."""
    img = np.ones((w.H, w.W, 3)) * np.array([0.93, 0.93, 0.91])
    img[:, :3] = (0.80, 0.86, 0.95)                                   # staging area outside the entrance
    if furniture:
        for f in w.furniture:
            for x, y in f.cells():
                img[y, x] = np.array(f.color) / 255 * 0.8 + 0.15
    if rubble:
        img[w.rubble > 0] = (0.55, 0.45, 0.35)
    if walls:
        img[w.structure == 1] = (0.25, 0.25, 0.28)
        img[w.structure == 2] = (0.45, 0.45, 0.5)
        img[w.structure == 3] = (0.2, 0.35, 0.6)
    if people:
        for d in w.decoys:
            for x, y in d.nav_cells:
                img[y, x] = (0.95, 0.75, 0.2)
        for d in w.thermal_decoys:
            for x, y in d.nav_cells:
                img[y, x] = (1.0, 0.45, 0.1)
        for v in w.victims:
            for x, y in v.nav_cells:
                img[y, x] = (0.85, 0.1, 0.15)
    return img


def show(ax, img, w, title=""):
    c = w.cfg.cell
    ax.imshow(img, extent=(0, w.W * c, w.H * c, 0), interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


# ------------------------------------------------------------------ 1. the seven buildings
def fig_buildings():
    fig, axes = plt.subplots(2, 4, figsize=(12, 4.6))
    for ax, b in zip(axes.ravel(), BUILDINGS):
        w = generate(RescueConfig(building=b, seed=2))
        show(ax, layer_image(w), w, {"plain": "open test ground"}.get(b, b))
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.legend(handles=[Patch(color=(0.25, 0.25, 0.28), label="wall"), Patch(color=(0.45, 0.45, 0.5), label="pillar"),
                       Patch(color=(0.2, 0.35, 0.6), label="storage rack"), Patch(color=(0.55, 0.45, 0.35), label="rubble"),
                       Patch(color=(0.7, 0.6, 0.5), label="furniture"), Patch(color=(0.85, 0.1, 0.15), label="victim"),
                       Patch(color=(0.95, 0.75, 0.2), label="look-alike"), Patch(color=(1.0, 0.45, 0.1), label="warm object"),
                       Patch(color=(0.80, 0.86, 0.95), label="staging area (base)")], loc="center", frameon=False)
    save(fig, "buildings.png")


def fig_layers():
    w = generate(RescueConfig(building="office", seed=3))
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6))
    steps = [dict(walls=True, furniture=False, rubble=False, people=False), dict(walls=True, furniture=True, rubble=False, people=False),
             dict(walls=True, furniture=True, rubble=True, people=False), dict()]
    titles = ["1. walls and doors\n(some collapsed)", "2. + furniture\n(never cuts off a room)", "3. + rubble piles",
              "4. + victims, look-alikes,\nwarm objects"]
    for ax, st, t in zip(axes, steps, titles):
        show(ax, layer_image(w, **st), w, t)
    save(fig, "layers.png")


# ------------------------------------------------------------------ 2. thermal
def fig_thermal():
    w = generate(RescueConfig(building="apartments", seed=4))
    fig = plt.figure(figsize=(12, 3.6))
    ax = fig.add_axes([0.0, 0.05, 0.5, 0.9])
    ax.imshow(palette(to_unit(w.temp, w.ambient)), extent=(0, w.W * 0.5, w.H * 0.5, 0), interpolation="nearest")
    ax.set_title(f"surface temperature of every 25 cm cell (ambient {w.ambient:.0f} °C)")
    ax.set_xticks([]); ax.set_yticks([])
    ax2 = fig.add_axes([0.56, 0.16, 0.2, 0.7])
    skin = [v.skin_temp for v in w.victims]
    cloth = [float(w.temp[fy, fx]) for v in w.victims for li, (fx, fy) in zip(v.segments, v.fine_cells) if li > 0 and v.cover == "none"]
    warm = [d.temp for d in w.thermal_decoys]
    ax2.hist([np.full(400, w.ambient) + np.random.default_rng(0).normal(0, 0.8, 400), cloth, skin, warm],
             bins=np.arange(12, 62, 1.5), stacked=False, label=["floor / walls", "clothing", "exposed skin", "warm objects"],
             color=["#9aa0a6", "#8e44ad", "#e67e22", "#c0392b"], density=True, histtype="stepfilled", alpha=0.55)
    ax2.axvline(40, color="k", ls="--", lw=1)
    ax2.text(40.5, ax2.get_ylim()[1] * 0.8, "40 °C: hotter\nthan any person", fontsize=7)
    ax2.set_xlabel("°C"); ax2.set_yticks([]); ax2.legend(fontsize=7, frameon=False)
    ax2.set_title("who is how warm")
    ax3 = fig.add_axes([0.81, 0.16, 0.18, 0.7])
    d = np.linspace(0, 0.8, 100)
    ax3.plot(d * 100, np.exp(-d / 0.10), color="#c0392b")
    ax3.axvspan(6, 14, color="#f5b041", alpha=0.3, label="thin debris")
    ax3.axvspan(40, 80, color="#7f8c8d", alpha=0.3, label="thick rubble")
    ax3.set_xlabel("rubble on the body (cm)"); ax3.set_ylabel("share of body heat reaching the surface")
    ax3.legend(fontsize=7, frameon=False); ax3.set_title("heat through rubble: exp(-d / 10 cm)")
    save(fig, "thermal.png")


# ------------------------------------------------------------------ 3. LiDAR scan and the map it builds
def fig_lidar():
    cfg = RescueConfig(building="office", seed=3)
    w = generate(cfg)
    x, y = w.to_metres((20, 20))
    s = lidar.scan(w, x, y, cfg, np.random.default_rng(1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    show(ax, layer_image(w, people=True), w, "one turn of the laser: 240 beams, 8 m, 40 cm above the floor")
    for a, r in zip(s.angles, s.ranges):
        rr = r if np.isfinite(r) else cfg.lidar_range
        ax.plot([x, x + np.cos(a) * rr], [y, y + np.sin(a) * rr], color="#7d5cff", lw=0.25, alpha=0.5)
    ax.scatter(s.points[:, 0], s.points[:, 1], s=2, color="#4b2bd6")
    ax.plot(x, y, "o", color="#e8702a", ms=6)
    ax = axes[1]
    img = np.ones((w.H, w.W, 3)) * 0.15
    for cx, cy in s.free_cells:
        img[cy, cx] = (0.85, 0.92, 0.85)
    for cx, cy in s.hit_cells:
        img[cy, cx] = (0.45, 0.3, 0.9)
    show(ax, img, w, "what that scan says: green = beams passed (free), violet = beams ended (hit)")
    ax.plot(x, y, "o", color="#e8702a", ms=6)
    save(fig, "lidar.png")


def fig_map_growth():
    sim = Simulator(RescueConfig(building="office", seed=3, vision="ideal", n_robots=3))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    snaps = [3, 30, 90]
    k = 0
    while sim.t < snaps[-1]:
        sim.step()
        if sim.t in snaps:
            ax = axes[k]; k += 1
            team = sim.team_map()
            searched = np.logical_or.reduce([r.searched for r in sim.robots])
            img = np.zeros((sim.world.H, sim.world.W, 3))
            img[team == UNKNOWN] = (0.12, 0.13, 0.16)
            img[team == FREE] = (0.62, 0.66, 0.72)
            img[(team == FREE) & searched] = (0.93, 0.94, 0.95)
            img[team == OBSTACLE] = (0.42, 0.3, 0.85)
            view = sim.explore_view(sim.robots[0])
            for f in frontiers(view):
                for cx, cy in f.cells:
                    img[cy, cx] = (0.17, 0.76, 0.8)
            show(ax, img, sim.world, f"t = {sim.t} s: {100 * sim.mapped():.0f}% mapped, {100 * sim.coverage():.0f}% searched")
            for r in sim.robots:
                px, py = sim.pos(r)
                ax.plot(px, py, "o", color=ROBOT[r.id], ms=5, mec="white")
    fig.legend(handles=[Patch(color=(0.12, 0.13, 0.16), label="unknown"), Patch(color=(0.62, 0.66, 0.72), label="mapped by LiDAR, not searched"),
                        Patch(color=(0.93, 0.94, 0.95), label="searched by a camera"), Patch(color=(0.42, 0.3, 0.85), label="obstacle"),
                        Patch(color=(0.17, 0.76, 0.8), label="frontier")], loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    save(fig, "map_growth.png")


# ------------------------------------------------------------------ 4. a camera frame and what the AI makes of it
def fig_camera():
    cfg = RescueConfig(building="apartments", seed=4, vision="fusion")
    w = generate(cfg)
    rng = np.random.default_rng(3)
    v = next(v for v in w.victims if v.cover == "none")
    best = None
    for cy, cx in np.argwhere(w.reachable & ~w.blocked):
        px, py = w.to_metres((cx, cy))
        d = np.hypot(px - v.x, py - v.y)
        if 2.0 < d < 3.2 and line_of_sight(w, (px, py), (v.x, v.y)):
            best = (px, py); break
    px, py = best
    frame = render(w, px, py, np.arctan2(v.y - py, v.x - px), rng)
    from rescue.perception import CNNDetector
    det = CNNDetector(cfg)
    heat = det.heatmap(frame)
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.5))
    axes[0].imshow(frame.rgb); axes[0].set_title("colour image")
    axes[1].imshow(frame.depth, cmap="viridis_r"); axes[1].set_title("depth image")
    axes[2].imshow(palette(to_unit(frame.thermal, w.ambient))); axes[2].set_title("thermal image")
    axes[3].imshow(frame.rgb * 0.5); axes[3].imshow(heat, cmap="Reds", alpha=0.75, extent=(0, 64, 48, 0), vmin=0, vmax=1)
    axes[3].contour(np.kron(heat >= cfg.detect_threshold, np.ones((4, 4))), levels=[0.5], colors="yellow", linewidths=1)
    axes[3].set_title("AI heatmap (yellow = detection)")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    save(fig, "camera.png")


# ------------------------------------------------------------------ 5. the belief filter over time
def fig_belief():
    sim = Simulator(RescueConfig(building="parking", seed=5, vision="thermal", n_robots=3))
    hist = {}
    while not sim.finished and sim.t < 260:
        sim.step()
        for c in sim.robots[0].registry.candidates:
            hist.setdefault(c.id, []).append((sim.t, c.logodds))
    import importlib.util
    spec = importlib.util.spec_from_file_location("rel", ROOT / "scripts" / "reliability.py")
    rel = importlib.util.module_from_spec(spec); spec.loader.exec_module(rel)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    shown = {"person": 0, "other": 0}
    for c in sim.robots[0].registry.candidates:
        what = rel.what_is_there(sim, c.x, c.y)
        key = "person" if what == "person" else "other"
        if shown[key] >= 4 or len(hist[c.id]) < 3:
            continue
        shown[key] += 1
        t, l = zip(*hist[c.id])
        ax.plot(t, l, color="#1e8449" if key == "person" else "#c0392b", lw=1.4,
                label=f"{what} ({c.status})")
    ax.axhline(sim.cfg.confirm_logodds, color="#1e8449", ls="--", lw=1)
    ax.text(sim.t * 0.99, sim.cfg.confirm_logodds + 0.4, "confirm at 9 (and 6 frames, one close look)", ha="right", fontsize=7, color="#1e8449")
    ax.axhline(sim.cfg.reject_logodds, color="#c0392b", ls="--", lw=1)
    ax.text(sim.t * 0.99, sim.cfg.reject_logodds - 1.2, "reject at -1.5", ha="right", fontsize=7, color="#c0392b")
    ax.set_ylim(-4, 26)
    ax.set_xlabel("mission time (s)"); ax.set_ylabel("evidence (log-odds)")
    ax.set_title("how evidence for sightings grows or falls (thermal camera, parking garage)")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    save(fig, "belief.png")


# ------------------------------------------------------------------ 6. the four planners on one route
def fig_planners():
    w = generate(RescueConfig(building="office", seed=3))
    known = np.where(w.blocked, OBSTACLE, FREE).astype(np.int8)
    from rescue.mapping import dijkstra
    s = (5, 20)
    dist, _ = dijkstra(known, s)
    far = np.where(np.isfinite(dist), dist, -1)
    gy, gx = np.unravel_index(np.argmax(far), far.shape)
    g = (int(gx), int(gy))                                              # the farthest reachable cell
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    base = layer_image(w, people=False)
    for ax, alg, name in zip(axes, ("dijkstra", "astar", "rrtstar", "aco"), ("Dijkstra", "A*", "RRT*", "Ant colony")):
        p = plan(alg, known, s, g, np.random.default_rng(2), reach=w.reachable)
        img = base.copy()
        if alg in ("dijkstra", "astar"):
            for x, y in p.search:
                img[y, x] = img[y, x] * 0.4 + np.array([0.55, 0.75, 1.0]) * 0.6
        elif alg == "aco":
            for x, y in p.search:
                img[y, x] = img[y, x] * 0.4 + np.array([1.0, 0.78, 0.3]) * 0.6
        show(ax, img, w, f"{name}\n{p.length * 0.5:.1f} m route, {p.expanded} "
                         f"{'samples' if alg == 'rrtstar' else 'ant steps' if alg == 'aco' else 'cells'}, {p.ms:.0f} ms")
        if alg == "rrtstar":
            for x1, y1, x2, y2 in p.search:
                ax.plot([x1 * 0.5, x2 * 0.5], [y1 * 0.5, y2 * 0.5], color="#5d9cec", lw=0.4)
        pts = [s, *p.path]
        ax.plot([(c[0] + 0.5) * 0.5 for c in pts], [(c[1] + 0.5) * 0.5 for c in pts], color="#c0392b", lw=1.6)
        ax.plot((s[0] + .5) * .5, (s[1] + .5) * .5, "o", color="#27ae60", ms=5)
        ax.plot((g[0] + .5) * .5, (g[1] + .5) * .5, "*", color="#c0392b", ms=9)
    save(fig, "planners.png")


# ------------------------------------------------------------------ 7. coverage patterns for a team
def fig_patterns():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
    for ax, (pat, n) in zip(axes, (("boustrophedon", 1), ("spiral", 1), ("boustrophedon", 4))):
        cfg = RescueConfig(building="plain", seed=0, vision="ideal", n_robots=n, coverage=pat)
        sim = Simulator(cfg)
        show(ax, layer_image(sim.world, people=False), sim.world,
             f"{'lawnmower' if pat == 'boustrophedon' else 'spiral'}, {n} robot{'s' if n > 1 else ''} "
             f"(w = {coverage.lane_width(cfg):.1f} m)")
        for b in sim.region_boxes:
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ls=":", ec="#555"))
        for r in sim.robots:
            xs, ys = zip(*r.sweep.corners)
            ax.plot(xs, ys, "-", color=ROBOT[r.id], lw=1.5)
            ax.plot(xs[0], ys[0], "o", color=ROBOT[r.id], ms=5)
            ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                        arrowprops=dict(arrowstyle="->", color=ROBOT[r.id], lw=1.5))
    save(fig, "patterns.png")


# ------------------------------------------------------------------ 8. strategies compared
def fig_strategies():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for strat, col in zip(("random", "greedy", "partition", "coordinated"), ("#95a5a6", "#e67e22", "#8e44ad", "#2f7de0")):
        sim = Simulator(RescueConfig(building="office", seed=7, vision="ideal", n_robots=4, strategy=strat)).run()
        t = [h["t"] for h in sim.history]; c = [100 * h["coverage"] for h in sim.history]
        ax.plot(t, c, color=col, lw=1.6, label=f"{strat} (done at {sim.t} s)")
    ax.set_xlabel("mission time (s)"); ax.set_ylabel("% of the building searched")
    ax.set_title("the same office, four team strategies (4 robots, perfect eyes)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "strategies.png")


# ------------------------------------------------------------------ 9. batteries
def fig_battery():
    cfg = RescueConfig(building="plain", seed=0, vision="ideal", n_robots=4, coverage="boustrophedon",
                       battery_each=(1.5, 2.0, 3.0, 0.0))
    sim = Simulator(cfg)
    rows = {r.id: [] for r in sim.robots}
    while not sim.finished:
        sim.step()
        for r in sim.robots:
            ch = sim.charge(r)
            rows[r.id].append((sim.t, 100 * (ch if ch is not None else 1.0), r.state))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for r in sim.robots:
        t, c, st = zip(*rows[r.id])
        cap = cfg.battery_each[r.id]
        ax.plot(t, c, color=ROBOT[r.id], lw=1.5, label=f"R{r.id}: {cap:g} Wh" if cap else f"R{r.id}: unlimited")
        charging = [ti for ti, s in zip(t, st) if s == "charging"]
        if charging:
            ax.scatter(charging, [c[t.index(ti)] for ti in charging], s=4, color=ROBOT[r.id])
    ax.set_xlabel("mission time (s)"); ax.set_ylabel("battery (%)")
    ax.set_title("different batteries, one lawnmower sweep: robots turn home in time, recharge (dots) and resume")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")
    ax.set_ylim(0, 105)
    save(fig, "battery.png")


# ------------------------------------------------------------------ 10. reliability (numbers from scripts/reliability.py)
def fig_reliability():
    labels = ["colour\ncamera", "thermal\ncamera", "colour +\nthermal"]
    old = [63, 74, 75]; new = [81, 79, 89]
    fa_old = [54, 34, 30]; fa_new = [21, 25, 13]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    x = np.arange(3)
    for ax, a, b, t in ((axes[0], old, new, "precision: confirmed reports that were real people (%)"),
                        (axes[1], fa_old, fa_new, "false alarms in 12 unseen missions")):
        ax.bar(x - 0.18, a, 0.36, color="#bdc3c7", label="previous rule (0.85 / 6 / 4 frames)")
        ax.bar(x + 0.18, b, 0.36, color="#2f7de0", label="current rule (0.92 / 9 / 6 frames)")
        for i in range(3):
            ax.text(x[i] - 0.18, a[i] + 1, str(a[i]), ha="center", fontsize=7)
            ax.text(x[i] + 0.18, b[i] + 1, str(b[i]), ha="center", fontsize=7)
        ax.set_xticks(x, labels); ax.set_title(t)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "reliability.png")


if __name__ == "__main__":
    which = sys.argv[1:] or [n[4:] for n in list(globals()) if n.startswith("fig_")]
    for n in which:
        globals()["fig_" + n]()
