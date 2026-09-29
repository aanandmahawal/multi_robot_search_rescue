"""How a robot physically moves: time, energy, local obstacle avoidance and dead-end detection.

Kinematics (differential drive)
    A robot turns on the spot, then drives straight. A move from one cell to a neighbour at
    distance d (0.5 m straight, 0.71 m diagonal) with a heading change dtheta takes
        t = |dtheta| / omega_max + d / v
    seconds, with v the driving speed and omega_max the turning rate. Every simulated second gives a
    robot one second of "time credit"; a move is started while the credit is positive and its time
    is subtracted (the credit may go negative and is paid back in the next seconds). So over a
    mission the robot drives exactly at speed v, diagonals take sqrt(2) times as long as straight
    steps, and sharp turns cost time.

Energy (a 25 kg tracked robot)
    P_base    electronics and sensors, all the time the robot is working:
              computer 15 W + LiDAR 8 W (if fitted) + cameras 6 W (if fitted)
    E_drive   rolling resistance on a debris-strewn floor, through the drivetrain:
              E = m g C_rr d / eta          per metre   (m = 25 kg, C_rr = 0.05, eta = 0.6)
            + motor copper losses, which grow with the square of speed: P = k v^2, k = 12 W s^2/m^2,
              i.e. E = k v d per metre
    E_turn    turning on the spot scrubs the tracks: E = m g mu_t (b / 2) |dtheta| / eta per radian
              (mu_t = 0.3, track width b = 0.4 m)
    Battery   capacity in Wh (1 Wh = 3600 J). A robot heads home when what is left is no more than
              the energy to drive back (distance x energy per metre, plus base power for the time it
              takes) times a 1.3 safety factor plus a 5 % reserve. At 0 J it stops where it is.

Dynamic Window Approach (Fox, Burgard & Thrun 1997), on the grid
    The global route says where to go; DWA decides the very next move, so the robot keeps clear of
    obstacles and teammates that were not there when the route was planned. The admissible moves
    (the "dynamic window") are the 8 neighbouring cells a robot can reach in the next step: free
    on its map, not occupied by a teammate, no corner cutting; staying put is the fallback. Each
    admissible move c is scored
        G(c) = alpha * heading(c) + beta * clearance(c) + gamma * progress(c) - lambda * turn(c)
    heading    1 - |angle between the move and the direction to the "carrot"| / pi, where the carrot
               is the route cell LOOKAHEAD cells ahead
    clearance  min(distance from c to the nearest obstacle or teammate, 3 cells) / 3
    progress   (distance to the carrot now - distance after the move) / step length, from [-1, 1]
               mapped to [0, 1]
    turn       |heading change| / pi (smoothness)
    with alpha = 0.45, beta = 0.25, gamma = 0.30, lambda = 0.10. The best move wins.

Dead ends and loops
    A robot remembers its last TRAIL_LEN positions. If, while it is driving somewhere, it has
    visited at most LOOP_UNIQUE different cells in that time (going back and forth, or stuck behind
    something), or it has been blocked for BLOCKED_STEPS seconds, it is in a dead end or a loop.
    It then backs off along its own trail to a cell at least BACKOFF cells away and marks the goal
    it was heading for as "tabu" for TABU_S seconds, so it chooses something else.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .mapping import FREE, OBSTACLE

# ---- robot and energy constants
MASS = 25.0              # kg
G = 9.81                 # m/s^2
C_RR = 0.05              # rolling resistance coefficient (tracks on a debris-strewn floor)
ETA = 0.6                # drivetrain efficiency
K_COPPER = 12.0          # W s^2 / m^2: motor losses k v^2
MU_TURN = 0.3            # track scrubbing when turning on the spot
TRACK = 0.4              # m, distance between the tracks
P_COMPUTER, P_LIDAR, P_CAMERAS = 15.0, 8.0, 6.0     # W
SAFETY, RESERVE = 1.3, 0.05

# ---- DWA
LOOKAHEAD = 4
ALPHA, BETA, GAMMA, LAMBDA = 0.45, 0.25, 0.30, 0.10
CLEAR_MAX = 3.0          # cells

# ---- dead ends
TRAIL_LEN, LOOP_UNIQUE, BLOCKED_STEPS, BACKOFF, TABU_S = 24, 4, 8, 4, 60

NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def base_power(cfg) -> float:
    """Watts drawn by electronics and sensors while the robot works."""
    return P_COMPUTER + (P_LIDAR if cfg.lidar else 0.0) + (P_CAMERAS if cfg.vision != "none" else 0.0)


def drive_energy_per_m(cfg) -> float:
    """Joules per metre driven: rolling resistance through the drivetrain + motor losses k v."""
    return MASS * G * C_RR / ETA + K_COPPER * cfg.speed


def turn_energy(dtheta: float) -> float:
    """Joules to turn on the spot by dtheta radians."""
    return MASS * G * MU_TURN * (TRACK / 2) * abs(dtheta) / ETA


def move_time(cfg, dist_m: float, dtheta: float) -> float:
    """Seconds to turn by dtheta, then drive dist_m."""
    return abs(dtheta) / np.radians(cfg.turn_rate) + dist_m / cfg.speed


def energy_home(cfg, dist_m: float) -> float:
    """Joules needed to drive dist_m back to base, with the safety factor."""
    return SAFETY * (drive_energy_per_m(cfg) * dist_m + base_power(cfg) * dist_m / cfg.speed)


def clearance_map(known: np.ndarray, others: list) -> np.ndarray:
    """Distance (cells) from every cell to the nearest obstacle on the map or teammate."""
    solid = known == OBSTACLE
    for x, y in others:
        solid[y, x] = True
    if not solid.any():
        return np.full(known.shape, CLEAR_MAX)
    return ndimage.distance_transform_edt(~solid)


def _cost_to_go(c, path: list, cum: list, total: float) -> float:
    """Local cost-to-go from cell c to the lookahead point: join the route at the best cell j,
    then follow it: min_j octile(c, path[j]) + (total - cum[j])."""
    best = np.inf
    for j, p in enumerate(path):
        dx, dy = abs(c[0] - p[0]), abs(c[1] - p[1])
        best = min(best, max(dx, dy) + 0.41421356 * min(dx, dy) + total - cum[j])
    return best


def dwa_step(cell, heading: float, path: list, known: np.ndarray, others: list, clear: np.ndarray):
    """Choose the next cell with the Dynamic Window Approach. Returns (cell or None, score table)."""
    H, W = known.shape
    x, y = cell
    ahead = path[:LOOKAHEAD]
    cum, prev, acc = [], cell, 0.0                              # route length from the robot to each route cell
    for p in ahead:
        acc += np.hypot(p[0] - prev[0], p[1] - prev[1])
        cum.append(acc)
        prev = p
    total = cum[-1]
    d_now = total                                               # I am on the route: cost-to-go = its length
    aim = ahead[min(1, len(ahead) - 1)]                         # heading target: the route two cells ahead
    to_aim = np.arctan2(aim[1] - y, aim[0] - x)
    occupied = set(map(tuple, others))
    nxt = path[0]
    rows = []
    for dx, dy in NEIGHBOURS:
        c = (x + dx, y + dy)
        if not (0 <= c[0] < W and 0 <= c[1] < H) or c in occupied:
            continue
        if not (known[c[1], c[0]] == FREE or c == nxt):          # the route's next cell may be unexplored (transit)
            continue
        if dx and dy and (known[y, c[0]] == OBSTACLE or known[c[1], x] == OBSTACLE):
            continue                                            # no corner cutting
        step = np.hypot(dx, dy)
        move = np.arctan2(dy, dx)
        heading_s = 1.0 - abs(wrap(move - to_aim)) / np.pi
        clear_s = min(float(clear[c[1], c[0]]), CLEAR_MAX) / CLEAR_MAX
        progress = (d_now - _cost_to_go(c, ahead, cum, total)) / step
        progress_s = (np.clip(progress, -1, 1) + 1) / 2
        turn_s = abs(wrap(move - heading)) / np.pi
        g = ALPHA * heading_s + BETA * clear_s + GAMMA * progress_s - LAMBDA * turn_s
        rows.append((g, c, heading_s, clear_s, progress_s, turn_s))
    if not rows:
        return None, []
    rows.sort(key=lambda r: -r[0])
    best = rows[0]
    if nxt in occupied and best[4] < 0.5:
        return None, rows                                       # a teammate is on my route and every way round leads back: wait
    return best[1], rows


def rejoin(path: list, cell) -> list:
    """After a DWA move: continue the route from where the robot is now, or [] to plan a new one."""
    head = path[:LOOKAHEAD + 2]
    if cell in head:
        return path[head.index(cell) + 1:]
    for j, c in enumerate(head):
        if max(abs(c[0] - cell[0]), abs(c[1] - cell[1])) == 1:
            return path[j:]
    return []


def stuck_in_loop(trail) -> bool:
    """A full trail of positions with at most LOOP_UNIQUE different cells: going nowhere."""
    return len(trail) >= TRAIL_LEN and len(set(trail)) <= LOOP_UNIQUE
