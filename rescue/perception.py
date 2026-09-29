"""Victim detection from camera images (computer vision).

Three detectors share one architecture (``VictimNet``: a small fully-convolutional network
whose output is a 12 x 16 heatmap, the probability that each 4x4-pixel block of the image
shows part of a person) and differ only in what they look at:

  cnn       colour + depth             (4 channels)   the classic camera
  thermal   temperature + depth        (2 channels)   the thermal camera: warm bodies, even in the dark
  fusion    colour + depth + thermal   (5 channels)   both cameras together

Peaks in the heatmap become detections; the depth reading at the peak turns them into a
position on the map; with a thermal camera the temperature at the peak is attached, and a
"person" hotter than 40 °C (a heater, a car engine) is flagged as too hot to be one.

Training data comes from the simulator itself: thousands of rendered camera views in randomly
generated disaster zones, labelled automatically from the renderer's per-pixel object ids.
Views are biased towards victims, look-alike objects (jackets, bags, boxes, beams, barrels) and
warm objects (heaters, pets, hot water, laptops, warm engines), so each network has to learn
what a person looks like to *its* sensor rather than "anything on the floor" or "anything
warm". Body parts under rubble are only labelled "person" for the thermal detectors, and only
when the cover is thin enough for heat to leak through (< 20 cm): no network is asked to see
through concrete.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import RescueConfig
from .sensors import CameraFrame, line_of_sight, render
from .thermal import COVER_VISIBLE, HUMAN_MAX
from .world import DECOY_BASE, VICTIM_BASE, World, generate

BLOCK = 4   # heatmap cell = 4x4 pixels
# channel slices of the full stack [red, green, blue, depth, thermal]
MODALITIES = {"cnn": (0, 4), "thermal": (3, 5), "fusion": (0, 5)}
MODALITY_NAMES = {"cnn": "colour camera", "thermal": "thermal camera", "fusion": "colour + thermal cameras"}


def model_path(cfg: RescueConfig, modality: str) -> str:
    return {"cnn": cfg.detector_path, "thermal": cfg.thermal_detector_path, "fusion": cfg.fusion_detector_path}[modality]


def n_channels(modality: str) -> int:
    a, b = MODALITIES[modality]
    return b - a


def is_victim(obj: np.ndarray) -> np.ndarray:
    return (obj >= VICTIM_BASE) & (obj < DECOY_BASE)


def thermal_to_unit(thermal: np.ndarray) -> np.ndarray:
    """10 °C -> 0, 50 °C -> 1 (a radiometric camera measures absolute temperatures)."""
    return np.clip((thermal - 10.0) / 40.0, 0, 1)


def frame_to_stack(frame: CameraFrame, cfg: RescueConfig) -> np.ndarray:
    """All five channels, each scaled to 0..1."""
    depth = np.clip(frame.depth / (cfg.camera_range + 2.0), 0, 1)
    thermal = thermal_to_unit(frame.thermal) if frame.thermal is not None else np.zeros_like(depth)
    return np.concatenate([frame.rgb.transpose(2, 0, 1), depth[None], thermal[None]], axis=0).astype(np.float32)


def frame_to_input(frame: CameraFrame, cfg: RescueConfig, modality: str = "cnn") -> np.ndarray:
    a, b = MODALITIES[modality]
    return frame_to_stack(frame, cfg)[a:b]


def frame_to_label(frame: CameraFrame, modality: str = "cnn") -> np.ndarray:
    """Which 4x4 blocks show a person *that this sensor could see*: exposed body parts for
    the colour camera, exposed or thinly covered ones for the thermal detectors."""
    vis = is_victim(frame.obj)
    if frame.cover is not None:
        vis &= frame.cover < (COVER_VISIBLE if modality in ("thermal", "fusion") else 1e-6)
    H, W = frame.obj.shape
    v = vis.reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK).mean(axis=(1, 3))
    return (v >= 0.15).astype(np.float32)


# --------------------------------------------------------------------- network
def build_net(in_channels: int = 4):
    import torch.nn as nn

    def block(i, o, d=1):
        return [nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

    return nn.Sequential(
        *block(in_channels, 16), *block(16, 16), nn.MaxPool2d(2),   # 24 x 32
        *block(16, 32), *block(32, 32), nn.MaxPool2d(2),            # 12 x 16
        *block(32, 48), *block(48, 48, d=2), *block(48, 48, d=4),   # wider context
        nn.Conv2d(48, 1, 1),
    )


# ---------------------------------------------------------------- data generation
def _sample_views(world: World, rng: np.random.Generator, n: int):
    """Camera poses: mostly aimed at victims, look-alikes or warm objects from 1-5 m, some random."""
    free = np.argwhere(world.reachable & ~world.blocked)
    victims = [(v.x, v.y) for v in world.victims]
    lookalikes = [(d.x, d.y) for d in world.decoys] + [(d.x, d.y) for d in world.thermal_decoys]
    c = world.cfg.cell
    for f in world.furniture:                      # the warm bonnet of a car whose engine ran recently
        if f.temp > 0:
            fx, fy = f.x0 + f.w / 2, f.y0 + f.h / 2
            if f.facing in ("n", "s"):
                fy = f.y0 + 0.5 if f.facing == "n" else f.y0 + f.h - 0.5
            else:
                fx = f.x0 + 0.5 if f.facing == "w" else f.x0 + f.w - 0.5
            lookalikes.append((fx * c, fy * c))
    poses = []
    tries = 0
    while len(poses) < n and tries < n * 30:
        tries += 1
        cy, cx = free[int(rng.integers(len(free)))]
        px, py = world.to_metres((cx, cy))
        r = rng.random()
        if r < 0.3 or not (victims or lookalikes):
            poses.append((px, py, rng.uniform(-np.pi, np.pi)))
            continue
        if (r < 0.62 and victims) or not lookalikes:
            tx, ty = victims[int(rng.integers(len(victims)))]          # a person (maybe under rubble)
        else:
            tx, ty = lookalikes[int(rng.integers(len(lookalikes)))]    # a hard negative
        d = np.hypot(tx - px, ty - py)
        if not (0.8 < d < 5.0) or not line_of_sight(world, (px, py), (tx, ty)):
            continue
        poses.append((px, py, np.arctan2(ty - py, tx - px) + rng.normal(0, 0.35)))
    return poses


def _render_world(args):
    base, seed, views = args
    rng = np.random.default_rng(seed)
    from .world import BUILDINGS, DAMAGE
    cfg = replace(base, seed=seed, n_victims=int(rng.integers(8, 15)), n_decoys=int(rng.integers(10, 20)),
                  n_warm_objects=int(rng.integers(4, 10)),
                  building=str(rng.choice(BUILDINGS)), damage=str(rng.choice(list(DAMAGE))))
    world = generate(cfg)
    X, Yc, Yt = [], [], []
    for pose in _sample_views(world, rng, views):
        frame = render(world, *pose, rng=rng)
        X.append(frame_to_stack(frame, cfg))
        Yc.append(frame_to_label(frame, "cnn"))
        Yt.append(frame_to_label(frame, "thermal"))
    return np.stack(X).astype(np.float16), np.stack(Yc).astype(np.uint8), np.stack(Yt).astype(np.uint8)


def make_dataset(base: RescueConfig, n_worlds: int, views_per_world: int, seed0: int, verbose=True):
    """Render ``views_per_world`` labelled images (all five channels) in each of ``n_worlds``
    random disaster zones, in parallel on all CPU cores. Returns X and one label set per
    modality."""
    from concurrent.futures import ProcessPoolExecutor

    X, Yc, Yt = [], [], []
    jobs = [(base, seed0 + i, views_per_world) for i in range(n_worlds)]
    with ProcessPoolExecutor() as pool:
        for i, (x, yc, yt) in enumerate(pool.map(_render_world, jobs)):
            X.append(x); Yc.append(yc); Yt.append(yt)
            if verbose and (i + 1) % 20 == 0:
                print(f"  rendered {i + 1}/{n_worlds} disaster zones ({sum(len(a) for a in X)} images)", flush=True)
    Yc, Yt = np.concatenate(Yc).astype(np.float32), np.concatenate(Yt).astype(np.float32)
    return np.concatenate(X), {"cnn": Yc, "thermal": Yt, "fusion": Yt}   # images kept as float16 to save memory


# ------------------------------------------------------------------- training
def _fit(net, X: np.ndarray, Y: np.ndarray, epochs: int, verbose: bool) -> None:
    import torch
    import torch.nn as nn

    pos_frac = Y.mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(20.0, (1 - pos_frac) / max(pos_frac, 1e-4)) ** 0.5))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr, Ytr = torch.from_numpy(X), torch.from_numpy(Y)[:, None]
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        total, t0 = 0.0, time.time()
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb, yb = Xtr[idx].float(), Ytr[idx]
            if torch.rand(1).item() < 0.5:                     # augmentation: mirror image
                xb, yb = xb.flip(-1), yb.flip(-1)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if verbose:
            print(f"    epoch {ep + 1:2d}/{epochs}  loss {total / len(perm):.4f}  ({time.time() - t0:.0f} s)", flush=True)


def train_detectors(base: RescueConfig, modalities=("cnn", "thermal", "fusion"), n_worlds: int = 200,
                    views: int = 100, epochs: int = 20, verbose: bool = True) -> dict:
    """Render one dataset and train one detector per modality on it. Every model is saved next
    to a .json with its scores on images from disaster zones it never saw."""
    import torch

    t0 = time.time()
    if verbose:
        print(f"Rendering training images from {n_worlds} random disaster zones...")
    X, Y = make_dataset(base, n_worlds, views, seed0=10_000, verbose=verbose)
    if verbose:
        print(f"Rendering test images from {max(10, n_worlds // 5)} unseen disaster zones...")
    Xt, Yt = make_dataset(base, max(10, n_worlds // 5), views, seed0=90_000, verbose=verbose)
    render_seconds = time.time() - t0

    reports = {}
    for m in modalities:
        a, b = MODALITIES[m]
        t1 = time.time()
        if verbose:
            print(f"\nTraining the {MODALITY_NAMES[m]} detector ({b - a} input channels)...")
        torch.manual_seed(0)
        net = build_net(b - a)
        _fit(net, np.ascontiguousarray(X[:, a:b]), Y[m], epochs, verbose)
        report = evaluate(net, np.ascontiguousarray(Xt[:, a:b]), Yt[m])
        report.update(modality=m, channels=b - a, train_images=int(len(X)), test_images=int(len(Xt)),
                      epochs=epochs, seconds=round(time.time() - t1 + render_seconds / len(modalities), 1))
        path = Path(model_path(base, m))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), path)
        path.with_suffix(".json").write_text(json.dumps(report, indent=2))
        report["path"] = str(path)
        reports[m] = report
    return reports


def evaluate(net, X: np.ndarray, Y: np.ndarray, threshold: float = 0.5) -> dict:
    """Image-level and heatmap-level quality on held-out images."""
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    net.eval()
    with torch.no_grad():
        P = torch.sigmoid(torch.cat([net(torch.from_numpy(X[i:i + 256]).float()) for i in range(0, len(X), 256)]))[:, 0].numpy()
    has = Y.reshape(len(Y), -1).max(axis=1) > 0
    score = P.reshape(len(P), -1).max(axis=1)
    pred = score >= threshold
    tp = int((pred & has).sum()); fp = int((pred & ~has).sum()); fn = int((~pred & has).sum())
    return {
        "image_precision": tp / max(1, tp + fp),
        "image_recall": tp / max(1, tp + fn),
        "image_auc": float(roc_auc_score(has, score)) if 0 < has.mean() < 1 else None,
        "block_average_precision": float(average_precision_score(Y.ravel() > 0, P.ravel())),
        "images_with_victims": int(has.sum()),
        "threshold": threshold,
    }


# ------------------------------------------------------------------- inference
@dataclass
class Detection:
    x: float                   # metres
    y: float
    confidence: float
    distance: float            # how far away the robot saw it
    temp: float | None = None  # °C at the detection (thermal camera only)
    too_hot: bool = False      # hotter than any living person: a heater or an engine, not a victim


class CNNDetector:
    def __init__(self, cfg: RescueConfig, modality: str | None = None):
        import torch
        self.modality = modality or cfg.vision
        if self.modality not in MODALITIES:
            raise ValueError(f"no learned detector for vision mode {self.modality!r}")
        path = Path(model_path(cfg, self.modality))
        if not path.exists():
            raise FileNotFoundError(f"no trained detector for the {MODALITY_NAMES[self.modality]} at {path}. "
                                    f"Train it first:  python -m rescue train --modality {self.modality}")
        self.torch = torch
        torch.set_num_threads(1)
        self.net = build_net(n_channels(self.modality))
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()
        self.cfg = cfg
        self.uses_thermal = self.modality in ("thermal", "fusion")

    def heatmap(self, frame: CameraFrame) -> np.ndarray:
        x = self.torch.from_numpy(frame_to_input(frame, self.cfg, self.modality)[None])
        with self.torch.no_grad():
            return self.torch.sigmoid(self.net(x))[0, 0].numpy()

    def detect(self, frame: CameraFrame, heat: np.ndarray | None = None) -> list[Detection]:
        from scipy import ndimage

        heat = self.heatmap(frame) if heat is None else heat
        mask = heat >= self.cfg.detect_threshold
        labels, n = ndimage.label(mask)
        out = []
        for k in range(1, n + 1):
            cells = np.argwhere(labels == k)
            by, bx = cells[np.argmax(heat[labels == k])]
            conf = float(heat[by, bx])
            # the nearest valid depth reading inside the peak block locates the object
            sub = frame.depth[by * BLOCK:(by + 1) * BLOCK, bx * BLOCK:(bx + 1) * BLOCK]
            iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
            d = float(sub[iy, ix])
            if d >= self.cfg.camera_range + 1.5:
                continue
            p = frame.points[by * BLOCK + iy, bx * BLOCK + ix]
            temp = None
            if self.uses_thermal and frame.thermal is not None:
                temp = float(frame.thermal[by * BLOCK:(by + 1) * BLOCK, bx * BLOCK:(bx + 1) * BLOCK].max())
            too_hot = temp is not None and self.cfg.thermal_check and temp > HUMAN_MAX
            out.append(Detection(float(p[0]), float(p[1]), conf, d, temp, too_hot))
        return out
