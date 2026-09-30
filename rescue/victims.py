"""Shared victim registry: turning noisy detections into confirmed victim locations.

Robots never trust a single camera frame. Every observation becomes a *report* (a message
other robots can receive):

  positive report  "I think I see a person at (x, y) with confidence p, from d metres"
                   (with a thermal camera: "... and it is T degrees warm")
  negative report  "I looked at candidate location (x, y) from close by and saw nobody"

Reports near the same spot are fused into a *candidate* holding a log-odds belief that a
victim is there (a standard Bayesian evidence filter). A close look counts more than a
distant one. A candidate is **confirmed** once its evidence crosses ``confirm_logodds`` and
**rejected** when it drops below ``reject_logodds``. Uncertain candidates become
"go and take a closer look" tasks for the team.

**One verdict for the whole team.** Evidence is a plain sum (the order of the reports does not
matter) and the verdict is only taken in ``settle``, once per second, after the robots have
exchanged their reports; every robot applies new reports in the same order (by report id). So
robots that hold the same reports always hold the same verdict: one robot cannot confirm a
sighting that another rejected on the same evidence. A confirmation is final (it has been passed
on to the rescuers); a rejection is not: new, stronger sightings can reopen a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RescueConfig

MERGE_RADIUS = 1.3      # metres: detections closer than this are the same sighting
SAME_PERSON = 1.6       # metres: confirmed sightings closer than this are one person reported twice
                        # (a lying body is 1.75 m long: head and legs can show up as two warm shapes)
PRIOR = -0.6            # log-odds a new sighting starts with (most first glimpses are wrong)
MAX_VERIFY = 4          # close looks (each from a new viewpoint) before giving up on a sighting


@dataclass(frozen=True)
class Report:
    id: int
    t: int
    robot: int
    positive: bool
    x: float
    y: float
    confidence: float
    distance: float
    temp: float | None = None       # °C measured at the sighting (thermal camera only)


@dataclass
class Candidate:
    id: int
    x: float
    y: float
    logodds: float
    first_seen: int
    status: str = "candidate"          # candidate | confirmed | rejected
    confirmed_at: int | None = None
    positives: int = 0
    negatives: int = 0
    verify_attempts: int = 0
    weight: float = 0.0
    reporters: set = field(default_factory=set)
    frames: set = field(default_factory=set)       # (robot, time) of every positive sighting
    closest: float = float("inf")                  # closest distance it was seen from
    tried_spots: list = field(default_factory=list)   # where close looks were taken from
    temp: float | None = None                      # warmest temperature reported here (thermal camera)
    dirty: bool = False                            # new evidence since the last verdict


def distinct(confirmed: list) -> list:
    """Confirmed sightings with repeats of the same person removed: of two that lie closer
    than ``SAME_PERSON`` the one confirmed first is kept."""
    kept: list = []
    for c in sorted(confirmed, key=lambda c: (c.confirmed_at if c.confirmed_at is not None else 10**9, c.first_seen, c.x, c.y)):
        if all(np.hypot(c.x - k.x, c.y - k.y) >= SAME_PERSON for k in kept):
            kept.append(c)
    return kept


def _logit(p: float) -> float:
    p = min(max(p, 0.02), 0.98)
    return float(np.log(p / (1 - p)))


class Registry:
    """One robot's view of where victims are (merged from every report it has received)."""

    def __init__(self, cfg: RescueConfig):
        self.cfg = cfg
        self.candidates: list[Candidate] = []
        self.known: set[int] = set()

    def _near(self, x: float, y: float, radius: float) -> Candidate | None:
        best, best_d = None, radius
        for c in self.candidates:
            d = np.hypot(c.x - x, c.y - y)
            if d <= best_d:
                best, best_d = c, d
        return best

    def apply(self, r: Report) -> str | None:
        """Add one report's evidence (the verdict waits for ``settle``). Returns "new" when the
        report opened a new candidate."""
        if r.id in self.known:
            return None
        self.known.add(r.id)
        close = r.distance <= self.cfg.verify_distance
        w = 1.0 if close else 0.55
        c = self._near(r.x, r.y, MERGE_RADIUS)
        new = None
        if r.positive:
            if c is None:
                c = Candidate(len(self.candidates), r.x, r.y, PRIOR, r.t)
                self.candidates.append(c)
                new = "new"
            c.logodds += w * _logit(r.confidence)
            wt = w * r.confidence
            c.x = (c.x * c.weight + r.x * wt) / (c.weight + wt)
            c.y = (c.y * c.weight + r.y * wt) / (c.weight + wt)
            c.weight += wt
            c.positives += 1
            c.reporters.add(r.robot)
            c.frames.add((r.robot, r.t))
            c.closest = min(c.closest, r.distance)
            if r.temp is not None:
                c.temp = r.temp if c.temp is None else max(c.temp, r.temp)
        else:
            if c is None:
                return None
            c.logodds -= 0.9 * w
            c.negatives += 1
        c.dirty = True
        return new

    def settle(self, t: int) -> list[tuple[Candidate, str]]:
        """Take the verdict on every candidate whose evidence changed. Returns the changes
        [(candidate, "confirmed" | "rejected")]."""
        out = []
        for c in self.candidates:
            if c.dirty:
                c.dirty = False
                change = self._update_status(c, t)
                if change:
                    out.append((c, change))
        return out

    def _update_status(self, c: Candidate, t: int) -> str | None:
        if c.status == "confirmed":
            return None
        close_enough = c.closest <= self.cfg.verify_distance or not self.cfg.confirm_needs_close_look
        if c.logodds >= self.cfg.confirm_logodds and len(c.frames) >= self.cfg.confirm_frames and close_enough:
            c.status, c.confirmed_at = "confirmed", t
            return "confirmed"
        if c.logodds <= self.cfg.reject_logodds:
            changed = c.status != "rejected"
            c.status = "rejected"
            return "rejected" if changed else None
        c.status = "candidate"
        return None

    def open_candidates(self) -> list[Candidate]:
        """Sightings worth a closer look."""
        return [c for c in self.candidates if c.status == "candidate" and c.verify_attempts < MAX_VERIFY]

    def confirmed(self) -> list[Candidate]:
        return [c for c in self.candidates if c.status == "confirmed"]

    def people(self) -> list[Candidate]:
        """The people this robot believes it has found: confirmed sightings, each person once."""
        return distinct(self.confirmed())
