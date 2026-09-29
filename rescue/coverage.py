"""Coverage patterns: in which order a robot works through the space still to be searched.

Every pattern works on the same candidates, the frontier goals of ``coordination.robot_goals``
(the edge of the searched area), so the search always ends only when nothing reachable is left
unsearched. What changes is the *order*, expressed as a score s in [0, 1] for each goal
(0 = next in line, 1 = last):

* **frontier**       no order: the team strategy weighs size against distance (Yamauchi 1997).
* **boustrophedon**  "ox-turning", the lawnmower pattern (Choset 2000): the building is cut into
                     horizontal lanes one sensor footprint wide, w. Lane k = floor(y / w); inside a
                     lane the sweep runs left to right on even lanes and right to left on odd ones:
                         key = k * W + (x if k is even else W - 1 - x)
* **spiral**         from the outside in: ring k = floor(d_edge / w), where d_edge is the distance
                     to the nearest outer wall of the building (its outline is known to rescuers,
                     not its inside), and inside a ring the robot goes round clockwise starting
                     from the entrance side:
                         key = k * 2 pi + ((phi - phi_entrance) mod 2 pi)
The lane width w is the width a robot covers in one pass: its camera range (the camera must see
the people), or 60 % of the LiDAR range for a LiDAR-only team.

How the score enters the decision (``coordination``), in the same units as the strategy uses:
* coordinated (auction): value' = value + ORDER_VALUE * (1 - s)
* greedy / partition:    distance' = distance + ORDER_METRES * s
* random:                ignored (it is the baseline).
"""
from __future__ import annotations

import numpy as np

PATTERNS = ("frontier", "boustrophedon", "spiral")
ORDER_VALUE = 12.0        # auction utility points for being next in the pattern (= 24 m of driving at 0.5 per metre)
ORDER_METRES = 25.0       # greedy / partition: being last in the pattern counts like 25 m extra distance


def lane_width(cfg) -> float:
    """Width (metres) one robot covers in one pass."""
    return cfg.camera_range if cfg.vision != "none" else 0.6 * cfg.lidar_range


def pattern_key(pattern: str, world, cell: tuple[int, int]) -> float:
    """Sort key of a cell in the pattern (smaller = earlier)."""
    cfg, W, H = world.cfg, world.W, world.H
    w = max(1, int(round(lane_width(cfg) / cfg.cell)))            # lane width in cells
    x, y = cell
    if pattern == "boustrophedon":
        k = y // w
        return float(k * W + (x if k % 2 == 0 else W - 1 - x))
    if pattern == "spiral":
        x0 = 3                                                      # the building starts after the staging area
        d_edge = min(x - x0, W - 1 - x, y, H - 1 - y)
        k = max(0, d_edge) // w
        cx, cy = (x0 + W - 1) / 2, (H - 1) / 2
        phi = np.arctan2(y - cy, x - cx)                           # y points down: increasing phi is clockwise on screen
        phi0 = np.pi                                                # the entrance is on the left (west) side
        return float(k * 2 * np.pi + (phi - phi0) % (2 * np.pi))
    return 0.0


def order_scores(pattern: str, world, cells: list) -> list[float] | None:
    """Rank the given goal cells in the pattern: 0 = next, 1 = last. None for the frontier pattern."""
    if pattern == "frontier" or not cells:
        return None
    keys = np.array([pattern_key(pattern, world, c) for c in cells])
    if len(cells) == 1:
        return [0.0]
    rank = np.argsort(np.argsort(keys, kind="stable"), kind="stable")
    return (rank / (len(cells) - 1)).tolist()


def describe(pattern: str, world, cell) -> str:
    """Plain words for where a goal sits in the pattern (for the robot's reason)."""
    cfg = world.cfg
    w = max(1, int(round(lane_width(cfg) / cfg.cell)))
    if pattern == "boustrophedon":
        k = cell[1] // w
        return f"lawnmower lane {k + 1} ({'left to right' if k % 2 == 0 else 'right to left'})"
    if pattern == "spiral":
        d_edge = min(cell[0] - 3, world.W - 1 - cell[0], cell[1], world.H - 1 - cell[1])
        return f"spiral ring {max(0, d_edge) // w + 1} from the outside"
    return ""
