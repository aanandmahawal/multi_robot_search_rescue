"""Procedural disaster zone: a damaged building floor with hidden victims.

Six realistic building types, each with its own layout and furniture:

  office      open-plan desk clusters, meeting rooms and private offices, a lift core
  apartments  a corridor with furnished flats: sofa, dining table, kitchen, bedroom
  hospital    two corridors crossing, wards with hospital beds, a reception desk
  school      classrooms with rows of desks along a corridor, and a gym with bleachers
  parking     a parking garage: rows of parked cars, pillars and painted bays
  warehouse   offices along the top and a hall full of pillars and storage racks
  plain       an open test ground: one hall, a few fixed obstacles (in the middle, in the
              corners, along the walls), no rubble and no look-alikes. For checking how the
              coverage patterns and route planners behave.

Furniture is real: it blocks the robots' way, and tall pieces (desks, shelves, cars,
racks) block the camera's view.

Two resolutions describe the same world:

* **navigation grid** (``cell`` = 0.5 m): what robots plan on. ``blocked`` marks cells a
  robot cannot enter; ``nav_height`` is the tallest thing in the cell.
* **fine grid** (0.25 m): what cameras render. Every fine cell has a height, a colour and an
  object id (a 2.5-D heightfield).

Object ids in the fine grid: 0 floor, 1 wall / structure / furniture, 2 rubble, 3 debris,
10 + k = victim k, 100 + j = decoy j (things that could fool a colour camera),
200 + m = warm object m (things that could fool a thermal camera, see thermal.py).
Body parts of a victim that lie under rubble keep the victim's id; ``World.cover`` holds the
thickness of the rubble on top of them (0 = exposed).
Coordinates: x grows to the right, y grows "down" the map, z is up. Metres unless noted.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import RescueConfig

FINE = 2  # fine cells per navigation cell (per axis)

OBJ_FLOOR, OBJ_WALL, OBJ_RUBBLE, OBJ_DEBRIS = 0, 1, 2, 3
VICTIM_BASE, DECOY_BASE = 10, 100

WALL_HEIGHT = 2.4
SKIN = np.array([(255, 219, 172), (241, 194, 125), (224, 172, 105), (198, 134, 66), (141, 85, 36), (92, 60, 40)])
SHIRTS = np.array([(200, 40, 40), (40, 80, 190), (40, 150, 70), (230, 200, 40), (250, 120, 20),
                   (225, 225, 225), (110, 110, 115), (30, 30, 35), (130, 60, 150)])
PANTS = np.array([(35, 45, 90), (25, 25, 28), (170, 150, 110), (90, 90, 95), (60, 50, 40)])

BUILDINGS = ("office", "apartments", "hospital", "school", "parking", "warehouse", "plain")
# damage level -> (share of walls that collapse, rubble density, share of victims partly buried)
DAMAGE = {"light": (0.10, 0.03, 0.2), "moderate": (0.25, 0.06, 0.4), "severe": (0.45, 0.10, 0.6)}

STRUCT_WALL, STRUCT_PILLAR, STRUCT_RACK = 1, 2, 3
STRUCT_HEIGHT = {STRUCT_WALL: WALL_HEIGHT, STRUCT_PILLAR: 2.6, STRUCT_RACK: 2.0}
STRUCT_COLOR = {STRUCT_WALL: (168, 160, 146), STRUCT_PILLAR: (150, 150, 156), STRUCT_RACK: (72, 102, 150)}

# furniture kind -> (height in metres, default colour)
FURNITURE = {
    "desk": (0.75, (150, 118, 82)),
    "table": (0.75, (112, 80, 54)),
    "bed": (0.55, (225, 225, 230)),
    "hospital_bed": (0.65, (205, 218, 214)),
    "sofa": (0.8, (90, 105, 140)),
    "counter": (0.95, (92, 92, 98)),
    "shelf": (1.8, (130, 95, 60)),
    "cabinet": (1.3, (150, 152, 158)),
    "car": (1.45, (160, 30, 30)),
    "school_desk": (0.7, (176, 146, 104)),
    "bleachers": (1.2, (60, 90, 150)),
}
CAR_COLORS = [(170, 30, 35), (35, 70, 160), (220, 220, 225), (30, 30, 35), (120, 125, 130), (200, 160, 40), (40, 110, 70)]
SOFA_COLORS = [(90, 105, 140), (140, 70, 60), (80, 120, 90), (120, 110, 100)]
# floor colour per building: (main area, corridor / secondary, special area)
FLOORS = {
    "office": ((96, 102, 114), (128, 128, 124), (140, 132, 118)),
    "apartments": ((128, 96, 64), (150, 146, 138), (160, 160, 158)),
    "hospital": ((176, 190, 184), (196, 200, 196), (170, 186, 196)),
    "school": ((164, 150, 122), (150, 150, 146), (178, 132, 80)),
    "parking": ((74, 76, 80), (86, 88, 92), (74, 76, 80)),
    "warehouse": ((96, 94, 90), (110, 108, 104), (104, 100, 96)),
    "plain": ((138, 140, 136), (138, 140, 136), (138, 140, 136)),
}


@dataclass
class Victim:
    id: int
    x: float                    # centre, metres
    y: float
    fine_cells: list[tuple[int, int]]
    buried: float               # fraction of the body covered by rubble
    nav_cells: list[tuple[int, int]] = field(default_factory=list)
    head: tuple[float, float] = (0.0, 0.0)       # metres, for drawing the body in 3-D
    feet: tuple[float, float] = (0.0, 0.0)
    colors: dict = field(default_factory=dict)   # skin / shirt / pants RGB
    segments: list = field(default_factory=list)  # body part of each fine cell: 0 head, 1-3 torso, 4-6 legs
    covered: list = field(default_factory=list)   # fine cells hidden under rubble or debris
    cover: str = "none"          # none | partial (thick rubble on part of the body) | thin (whole body under
                                 # light debris: a faint heat signature leaks through) | thick (whole body
                                 # under thick rubble: no heat signature, no camera can see it)
    skin_temp: float = 0.0       # °C of the exposed skin (set in thermal.py)

    @property
    def detectable(self) -> bool:
        """Could any camera see this person? Not when the whole body is under thick rubble."""
        return self.cover != "thick"


@dataclass
class Decoy:
    id: int
    kind: str
    x: float
    y: float
    nav_cells: list[tuple[int, int]] = field(default_factory=list)
    size: tuple[float, float] = (0.5, 0.5)       # footprint in metres (x, y), for drawing
    color: list = field(default_factory=list)
    height: float = 0.2


@dataclass
class Furniture:
    kind: str
    x0: int                     # navigation-cell rectangle
    y0: int
    w: int
    h: int
    color: tuple
    height: float
    facing: str = ""            # n / s / e / w: where the front (car bonnet, bed head, bleacher top) points
    temp: float = 0.0           # °C: a car whose engine ran recently is warm (a thermal decoy)

    def cells(self):
        return [(x, y) for y in range(self.y0, self.y0 + self.h) for x in range(self.x0, self.x0 + self.w)]


@dataclass
class World:
    cfg: RescueConfig
    blocked: np.ndarray          # bool [H, W] navigation
    nav_height: np.ndarray       # float [H, W]
    height: np.ndarray           # float [H*2, W*2] fine heightfield
    color: np.ndarray            # uint8 [H*2, W*2, 3]
    obj: np.ndarray              # int16 [H*2, W*2]
    victims: list[Victim]
    decoys: list[Decoy]
    starts: list[tuple[int, int]]    # robot start cells (staging area)
    reachable: np.ndarray        # bool [H, W]: free cells connected to the staging area
    structure: np.ndarray | None = None   # int [H, W]: 1 wall, 2 pillar, 3 rack (0 = none)
    rubble: np.ndarray | None = None      # float [H, W]: rubble pile height (0 = none)
    furniture: list = field(default_factory=list)
    floor: np.ndarray | None = None       # int [H, W]: 0 main area, 1 corridor, 2 special area
    markings: list = field(default_factory=list)   # painted floor lines [x1, y1, x2, y2] in metres
    temp: np.ndarray | None = None        # float32 [H*2, W*2]: surface temperature (°C) of every fine cell
    cover: np.ndarray | None = None       # float [H*2, W*2]: rubble / debris lying on a victim, metres (0 = exposed)
    nav_temp: np.ndarray | None = None    # float [H, W]: warmest fine cell in each navigation cell
    ambient: float = 20.0                 # °C of the building itself
    thermal_decoys: list = field(default_factory=list)   # warm objects that are not people

    @property
    def W(self) -> int:
        return self.blocked.shape[1]

    @property
    def H(self) -> int:
        return self.blocked.shape[0]

    def to_metres(self, c: tuple[int, int]) -> tuple[float, float]:
        s = self.cfg.cell
        return ((c[0] + 0.5) * s, (c[1] + 0.5) * s)

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        s = self.cfg.cell
        return (int(np.floor(x / s)), int(np.floor(y / s)))

    def in_bounds(self, c) -> bool:
        return 0 <= c[0] < self.W and 0 <= c[1] < self.H

    def blocks_view(self) -> np.ndarray:
        """Cells that stop the camera: anything taller than the camera itself."""
        return self.nav_height > self.cfg.camera_height + 0.05

    @property
    def searchable(self) -> int:
        """Number of reachable cells: the area that can be searched."""
        return int(self.reachable.sum())


# ------------------------------------------------------------------ layout helpers
class Plan:
    """Walls, floors and furniture requests produced by a building layout."""

    def __init__(self, W, H, rng):
        self.rng = rng
        self.wall = np.zeros((H, W), dtype=bool)
        self.kind = np.zeros((H, W), dtype=np.int8)
        self.floor = np.zeros((H, W), dtype=np.int8)
        self.items: list[Furniture] = []
        self.markings: list = []
        self.bx0, self.bx1, self.by0, self.by1 = 3, W - 1, 0, H - 1
        self.mid = H // 2

    def hwall(self, y, x0, x1, doors=()):
        self.wall[y, x0:x1 + 1] = True
        for d in doors:
            self.wall[y, d:d + 2] = False

    def vwall(self, x, y0, y1, doors=()):
        self.wall[y0:y1 + 1, x] = True
        for d in doors:
            self.wall[d:d + 2, x] = False

    def door(self, lo, hi):
        """A random door position (2 cells wide) between lo and hi."""
        return int(self.rng.integers(lo + 1, max(lo + 2, hi - 2)))

    def add(self, kind, x0, y0, w, h, facing="", color=None):
        height, default = FURNITURE[kind]
        self.items.append(Furniture(kind, int(x0), int(y0), int(w), int(h), tuple(color or default), height, facing))

    def floor_rect(self, x0, y0, x1, y1, value):
        self.floor[y0:y1 + 1, x0:x1 + 1] = value


def _rooms_along(p: Plan, y0, y1, door_row, x_from, x_to, width=(8, 11)):
    """Split a strip into rooms with a door onto ``door_row``. Returns the room rectangles."""
    rooms, x = [], x_from
    while x < x_to:
        nx = x + int(p.rng.integers(*width))
        if nx >= x_to - 5:
            nx = x_to
        if nx < x_to:
            p.vwall(nx, y0, y1)
        d = p.door(x, nx)
        p.wall[door_row, d:d + 2] = False
        rooms.append((x + 1, min(y0, y1) + 1, nx - 1, max(y0, y1) - 1))
        x = nx
    return rooms


def _layout_office(p: Plan):
    """Open-plan office: meeting rooms and offices at top and bottom, desk clusters and a lift core."""
    bx0, bx1, mid, rng = p.bx0, p.bx1, p.mid, p.rng
    p.hwall(8, bx0, bx1)
    p.hwall(31, bx0, bx1)
    for rooms, back in ((_rooms_along(p, 0, 8, 8, bx0, bx1), "top"), (_rooms_along(p, 31, 39, 31, bx0, bx1), "bottom")):
        for (x0, y0, x1, y1) in rooms:
            p.floor_rect(x0, y0, x1, y1, 2)
            w = x1 - x0 + 1
            if rng.random() < 0.5 and w >= 7:                          # meeting room: long table
                p.add("table", x0 + 2, y0 + 2, w - 4, 3)
            else:                                                     # private office: desk + cabinet
                dy = y0 if back == "top" else y1 - 1
                p.add("desk", x0 + 1, dy, 4, 2)
                p.add("cabinet", x1 - 1, y0 + 1 if back == "top" else y1 - 3, 1, 3)
    # lift / stair core in the middle of the floor
    cx0, cx1, cy0, cy1 = 28, 33, 14, 25
    p.hwall(cy0, cx0, cx1); p.hwall(cy1, cx0, cx1); p.vwall(cx0, cy0, cy1); p.vwall(cx1, cy0, cy1)
    p.floor_rect(bx0 + 1, 9, bx1 - 1, 30, 0)
    # desk clusters: 2 x 2 desks facing each other
    for y in (10, 26):
        for x in range(bx0 + 4, bx1 - 6, 9):
            if cx0 - 7 < x < cx1 + 1:
                continue
            for dx in (0, 3):
                for dy in (0, 2):
                    p.add("desk", x + dx, y + dy, 3, 2, facing="n" if dy else "s")
    for x in range(bx0 + 4, bx1 - 6, 9):                             # a second row away from the aisle
        if cx0 - 7 < x < cx1 + 1:
            continue
        if rng.random() < 0.6:
            p.add("cabinet", x, 15, 5, 1)
            p.add("cabinet", x, 24, 5, 1)


def _layout_apartments(p: Plan):
    """Apartment block: a corridor with flats on both sides (living room + bedroom)."""
    bx0, bx1, mid, rng = p.bx0, p.bx1, p.mid, p.rng
    top, bottom = mid - 2, mid + 2
    p.hwall(top, bx0, bx1); p.hwall(bottom, bx0, bx1)
    p.floor_rect(bx0, top + 1, bx1, bottom - 1, 1)
    for y0, y1, cw, side in ((p.by0, top, top, "top"), (bottom, p.by1, bottom, "bottom")):
        x = bx0
        while x < bx1:
            nx = x + int(rng.integers(9, 12))
            if nx >= bx1 - 6:
                nx = bx1
            if nx < bx1:
                p.vwall(nx, y0, y1)
            d = p.door(x, nx)
            p.wall[cw, d:d + 2] = False
            sy = (y0 + y1) // 2                                        # bedroom at the back
            p.hwall(sy, x, nx, doors=[p.door(x, nx)])
            p.floor_rect(x + 1, min(y0, y1) + 1, nx - 1, max(y0, y1) - 1, 0)
            # living room (next to the corridor) and bedroom (at the back)
            if side == "top":
                living, bed_y0, bed_y1 = (sy + 1, cw - 1), y0 + 1, sy - 1
            else:
                living, bed_y0, bed_y1 = (cw + 1, sy - 1), sy + 1, y1 - 1
            lx0, lx1 = x + 1, nx - 1
            sofa_color = SOFA_COLORS[int(rng.integers(len(SOFA_COLORS)))]
            p.add("sofa", lx0, living[0] + 1, 2, 4, facing="e", color=sofa_color)
            p.add("table", lx0 + 4, living[0] + 2, 3, 2)
            p.add("counter", lx1, living[0] + 1, 1, 5)                 # kitchen along the side wall
            bed_y = bed_y0 if side == "top" else bed_y1 - 3
            p.add("bed", lx0 + 2, bed_y, 3, 4, facing="n" if side == "top" else "s")
            p.add("cabinet", lx1, bed_y0 + 1, 1, 3)                    # wardrobe
            x = nx


def _layout_hospital(p: Plan):
    """Hospital: two corridors crossing, wards with beds, a reception desk at the entrance."""
    bx0, bx1, mid, rng = p.bx0, p.bx1, p.mid, p.rng
    top, bottom = mid - 3, mid + 2                                     # corridor rows mid-2 .. mid+1
    vx0, vx1 = 29, 34                                                  # vertical corridor x 30..33
    p.hwall(top, bx0, bx1); p.hwall(bottom, bx0, bx1)
    p.vwall(vx0, p.by0, top); p.vwall(vx1, p.by0, top)
    p.vwall(vx0, bottom, p.by1); p.vwall(vx1, bottom, p.by1)
    p.wall[top, vx0 + 1:vx1] = False
    p.wall[bottom, vx0 + 1:vx1] = False
    p.floor_rect(bx0, top + 1, bx1, bottom - 1, 1)
    p.floor_rect(vx0 + 1, p.by0 + 1, vx1 - 1, p.by1 - 1, 1)
    first = True
    for y0, y1, cw, side in ((p.by0, top, top, "top"), (bottom, p.by1, bottom, "bottom")):
        for xa, xb in ((bx0, vx0), (vx1, bx1)):
            for (x0, ry0, x1, ry1) in _rooms_along(p, y0, y1, cw, xa, xb, width=(10, 13)):
                p.floor_rect(x0, ry0, x1, ry1, 2)
                if first:                                              # reception next to the entrance
                    p.add("counter", x0 + 1, ry1 - 3 if side == "top" else ry0 + 1, 5, 2)
                    p.add("cabinet", x0, ry0 + 1 if side == "top" else ry1 - 2, 4, 1)
                    first = False
                    continue
                n_beds = max(1, (x1 - x0 - 1) // 4)
                for b in range(n_beds):
                    bx = x0 + 1 + b * 4
                    by = ry0 if side == "top" else ry1 - 3
                    p.add("hospital_bed", bx, by, 2, 4, facing="n" if side == "top" else "s")
                    if b % 2 == 0 and bx + 2 <= x1:
                        p.add("cabinet", bx + 2, by if side == "top" else by + 3, 1, 1)


def _layout_school(p: Plan):
    """School: classrooms with rows of desks along a corridor, and a gym hall with bleachers."""
    bx0, bx1, mid, rng = p.bx0, p.bx1, p.mid, p.rng
    gx = 37                                                            # gym from here to the east wall
    top, bottom = mid - 3, mid + 2
    p.hwall(top, bx0, gx); p.hwall(bottom, bx0, gx)
    p.vwall(gx, p.by0, p.by1, doors=[mid - 2, mid])
    p.floor_rect(bx0, top + 1, gx - 1, bottom - 1, 1)
    for y0, y1, cw, side in ((p.by0, top, top, "top"), (bottom, p.by1, bottom, "bottom")):
        for (x0, ry0, x1, ry1) in _rooms_along(p, y0, y1, cw, bx0, gx, width=(10, 12)):
            p.floor_rect(x0, ry0, x1, ry1, 0)
            front = ry0 if side == "top" else ry1                      # teacher's desk and board by the far wall
            p.add("table", x0 + 3, front, 4, 1)
            p.add("shelf", x1, ry0 + 1, 1, 4)
            rows = range(ry0 + 3, ry1 - 1, 3) if side == "top" else range(ry1 - 4, ry0, -3)
            for r in list(rows)[:3]:
                for c in range(x0 + 1, x1 - 1, 3):
                    p.add("school_desk", c, r, 2, 1, facing="n" if side == "top" else "s")
    # gym
    p.floor_rect(gx + 1, p.by0 + 1, bx1 - 1, p.by1 - 1, 2)
    p.add("bleachers", gx + 2, p.by0 + 1, bx1 - gx - 3, 3, facing="n")
    c = 0.5
    x0, x1, y0, y1 = (gx + 2) * c, (bx1 - 1) * c, 6 * c, (p.by1 - 1) * c
    p.markings += [[x0, y0, x1, y0], [x0, y1, x1, y1], [x0, y0, x0, y1], [x1, y0, x1, y1],
                   [x0, (y0 + y1) / 2, x1, (y0 + y1) / 2]]
    cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, 1.6
    ang = np.linspace(0, 2 * np.pi, 25)
    p.markings += [[cx + r * np.cos(a), cy + r * np.sin(a), cx + r * np.cos(b), cy + r * np.sin(b)]
                   for a, b in zip(ang[:-1], ang[1:])]


def _layout_parking(p: Plan):
    """Parking garage: two rows of parking bays with cars, pillars, a driving lane."""
    bx0, bx1, rng = p.bx0, p.bx1, p.rng
    c = 0.5
    p.floor_rect(bx0 + 1, 12, bx1 - 1, 27, 1)                          # driving lane
    for (y0, facing) in ((3, "s"), (28, "n")):
        x = bx0 + 3
        while x + 4 < bx1 - 1:
            p.markings.append([x * c, y0 * c, x * c, (y0 + 9) * c])
            if rng.random() < 0.72:
                p.add("car", x + 1, y0, 3, 9, facing=facing, color=CAR_COLORS[int(rng.integers(len(CAR_COLORS)))])
            x += 5
        p.markings.append([x * c, y0 * c, x * c, (y0 + 9) * c])
    for x in range(bx0 + 3, bx1 - 1, 15):                              # structural pillars on the bay lines
        for y in (12, 27):
            p.wall[y, x] = True
            p.kind[y, x] = STRUCT_PILLAR
    for x in range(bx0 + 4, bx1 - 3, 4):                               # dashed centre line
        p.markings.append([x * c, 20 * c, (x + 2) * c, 20 * c])
    p.hwall(p.by1 - 3, bx1 - 8, bx1)                                   # stairwell in the corner
    p.vwall(bx1 - 8, p.by1 - 3, p.by1, doors=[p.by1 - 2])


def _layout_warehouse(p: Plan):
    """Warehouse: offices along the top, a big hall with pillars and storage racks."""
    bx0, bx1, mid, rng = p.bx0, p.bx1, p.mid, p.rng
    s = p.by0 + 7
    p.hwall(s, bx0, bx1)                                               # office wall first, doors cut next
    for (x0, y0, x1, y1) in _rooms_along(p, p.by0, s, s, bx0, bx1):
        p.floor_rect(x0, y0, x1, y1, 2)
        p.add("desk", x0 + 1, y0, 4, 2)
        p.add("shelf", x1, y0 + 1, 1, 3)
    for py in range(s + 5, p.by1 - 2, 7):
        for px in range(bx0 + 6, bx1 - 2, 8):
            p.wall[py, px] = True
            p.kind[py, px] = STRUCT_PILLAR
    for ry in range(s + 3, p.by1 - 1, 4):
        if abs(ry - mid) <= 1:
            continue
        x = bx0 + 5
        while x < bx1 - 4:
            length = int(rng.integers(6, 11))
            for rx in range(x, min(x + length, bx1 - 3)):
                if not p.wall[ry, rx]:
                    p.wall[ry, rx] = True
                    p.kind[ry, rx] = STRUCT_RACK
            x += length + 3


def _layout_plain(p: Plan):
    """Open test ground: one empty hall with a few fixed obstacles, the same every time.

    In the middle: a 2 x 2 m block at the centre and two 1 x 1 m crates on the diagonal.
    In the four corners: 1.5 x 1.5 m cabinets. Along the walls: a shelf on the top wall, one on the
    bottom wall, a counter on the right wall and one on the left wall (away from the entrance).
    A painted 2 m grid on the floor makes it easy to judge how straight the robots drive."""
    bx0, bx1, by0, by1 = p.bx0, p.bx1, p.by0, p.by1
    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
    p.add("cabinet", cx - 2, cy - 2, 4, 4)                                        # middle
    p.add("cabinet", bx0 + 11, by0 + 9, 2, 2)
    p.add("cabinet", bx1 - 13, by1 - 11, 2, 2)
    for x0, y0 in ((bx0 + 1, by0 + 1), (bx1 - 3, by0 + 1), (bx0 + 1, by1 - 3), (bx1 - 3, by1 - 3)):
        p.add("cabinet", x0, y0, 3, 3)                                            # corners
    p.add("shelf", bx0 + 15, by0 + 1, 8, 1)                                       # along the walls
    p.add("shelf", bx1 - 22, by1 - 1, 8, 1)
    p.add("counter", bx1 - 1, cy - 4, 1, 8)
    p.add("counter", bx0 + 1, by0 + 7, 1, 5)
    c = 0.5
    for x in range(bx0 + 1, bx1 + 1, 4):
        p.markings.append([x * c, (by0 + 1) * c, x * c, by1 * c])
    for y in range(by0 + 1, by1 + 1, 4):
        p.markings.append([(bx0 + 1) * c, y * c, bx1 * c, y * c])


LAYOUTS = {"office": _layout_office, "apartments": _layout_apartments, "hospital": _layout_hospital,
           "school": _layout_school, "parking": _layout_parking, "warehouse": _layout_warehouse,
           "plain": _layout_plain}


# ------------------------------------------------------------------ build
def generate(cfg: RescueConfig) -> World:
    rng = np.random.default_rng(cfg.seed)
    W, H = cfg.width, cfg.height
    building = cfg.building if cfg.building in LAYOUTS else "office"
    collapse, rubble_density, buried = DAMAGE.get(cfg.damage, DAMAGE["moderate"])
    p = Plan(W, H, rng)
    bx0, bx1, by0, by1, mid = p.bx0, p.bx1, p.by0, p.by1, p.mid

    # building shell: staging area x < 3, outer wall around the rest, one entrance
    p.wall[by0, bx0:] = p.wall[by1, bx0:] = True
    p.wall[:, bx0] = p.wall[:, bx1] = True
    LAYOUTS[building](p)
    p.wall[mid - 2:mid + 2, bx0] = False                               # 4-cell-wide main entrance
    wall, kind = p.wall, p.kind

    # partial collapse: interior walls fall and leave rubble around the gap
    rubble_h = np.zeros((H, W))
    interior = wall.copy()
    interior[[by0, by1], :] = False
    interior[:, [bx0, bx1]] = False
    ys, xs = np.nonzero(interior)
    n_collapse = int(collapse * len(xs) / 6)
    for i in rng.choice(len(xs), size=min(n_collapse, len(xs)), replace=False):
        x, y = int(xs[i]), int(ys[i])
        horizontal = wall[y, max(0, x - 1)] or wall[y, min(W - 1, x + 1)]
        for k in range(int(rng.integers(2, 5))):
            cx, cy = (x + k, y) if horizontal else (x, y + k)
            if 0 < cx < W - 1 and 0 < cy < H - 1 and interior[cy, cx]:
                wall[cy, cx] = False
                if rng.random() < 0.35:
                    rubble_h[cy, cx] = rng.uniform(0.4, 1.2)
    kind[~wall] = 0
    kind[wall & (kind == 0)] = STRUCT_WALL
    starts = [(1, y) for y in range(mid - 4, mid + 4)] + [(0, y) for y in range(mid - 4, mid + 4)]

    # furniture: only placed where it does not cut off part of the building
    blocked = wall | (rubble_h > 0)
    furniture: list[Furniture] = []
    furn_mask = np.zeros((H, W), dtype=bool)
    reach = _flood(blocked, starts[0])
    for it in p.items:
        cells = [(x, y) for x, y in it.cells() if 0 < x < W - 1 and 0 < y < H - 1]
        if len(cells) != it.w * it.h or any(blocked[y, x] or not reach[y, x] for x, y in cells):
            continue
        trial = blocked.copy()
        for x, y in cells:
            trial[y, x] = True
        after = _flood(trial, starts[0])
        if after.sum() != reach.sum() - len(cells):
            continue
        blocked, reach = trial, after
        furniture.append(it)
        for x, y in cells:
            furn_mask[y, x] = True

    # rubble piles inside rooms
    free = ~blocked
    free[:, :bx0 + 1] = False
    n_piles = min(int(rubble_density * cfg.obstacle_density * free.sum() / 4), int(free.sum()) // 2)
    if building == "plain":
        n_piles = 0                                                    # the test ground keeps only its fixed obstacles
    fy, fx = np.nonzero(free)
    for i in rng.choice(len(fx), size=n_piles, replace=False):
        x, y = int(fx[i]), int(fy[i])
        for _ in range(int(rng.integers(2, 7))):
            if 0 < x < W - 1 and 0 < y < H - 1 and not wall[y, x] and not furn_mask[y, x] \
                    and abs(y - mid) + abs(x - bx0) > 4:
                rubble_h[y, x] = max(rubble_h[y, x], rng.uniform(0.3, 1.4))
            dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(4))]
            x, y = x + dx, y + dy
    blocked = wall | (rubble_h > 0) | furn_mask
    reachable = _flood(blocked, starts[0])

    # --- fine heightfield (what cameras see) ---------------------------------
    Hf, Wf = H * FINE, W * FINE
    height = np.zeros((Hf, Wf))
    obj = np.zeros((Hf, Wf), dtype=np.int16)
    floor_cols = np.array(FLOORS[building], dtype=float)
    color = floor_cols[np.repeat(np.repeat(p.floor, FINE, 0), FINE, 1)] + rng.normal(0, 7, (Hf, Wf, 1))

    def fine_block(cx, cy):
        return slice(cy * FINE, (cy + 1) * FINE), slice(cx * FINE, (cx + 1) * FINE)

    for y, x in zip(*np.nonzero(wall)):
        sy, sx = fine_block(x, y)
        k = int(kind[y, x])
        height[sy, sx] = STRUCT_HEIGHT[k]
        obj[sy, sx] = OBJ_WALL
        color[sy, sx] = np.array(STRUCT_COLOR[k]) + rng.normal(0, 8, (FINE, FINE, 1))
    for it in furniture:
        for x, y in it.cells():
            sy, sx = fine_block(x, y)
            height[sy, sx] = it.height
            obj[sy, sx] = OBJ_WALL
            color[sy, sx] = np.array(it.color) + rng.normal(0, 8, (FINE, FINE, 1))
    for y, x in zip(*np.nonzero(rubble_h)):
        sy, sx = fine_block(x, y)
        height[sy, sx] = rubble_h[y, x] * rng.uniform(0.6, 1.0, (FINE, FINE))
        obj[sy, sx] = OBJ_RUBBLE
        tone = rng.choice([(120, 115, 108), (135, 120, 100), (100, 98, 95)])
        color[sy, sx] = np.array(tone) + rng.normal(0, 14, (FINE, FINE, 3))
    # light debris on the floor: visual clutter only, robots drive over it
    clutter = (rng.random((Hf, Wf)) < (0.0 if building == "plain" else 0.04)) & (obj == OBJ_FLOOR)
    height[clutter] = rng.uniform(0.03, 0.08, clutter.sum())
    obj[clutter] = OBJ_DEBRIS
    color[clutter] = np.array([110, 95, 75]) + rng.normal(0, 15, (clutter.sum(), 3))
    color = np.clip(color, 0, 255)

    world = World(cfg, blocked, np.zeros((H, W)), height, color.astype(np.uint8), obj, [], [], starts, reachable)
    world.structure, world.rubble, world.furniture = kind, rubble_h, furniture
    world.floor, world.markings = p.floor, p.markings
    world.cover = np.zeros((Hf, Wf))
    _place_people_and_decoys(world, rng, buried)
    from .thermal import add_thermal                                   # (imported here: thermal.py imports this module)
    add_thermal(world, np.random.default_rng(cfg.seed + 7))            # own random stream: layouts stay as they were
    world.nav_height = height.reshape(H, FINE, W, FINE).max(axis=(1, 3))
    world.reachable = _flood(world.blocked, starts[0])
    return world


def _flood(blocked: np.ndarray, start) -> np.ndarray:
    H, W = blocked.shape
    seen = np.zeros_like(blocked)
    q = deque([start])
    seen[start[1], start[0]] = True
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not seen[ny, nx] and not blocked[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny))
    return seen


def _try_place(world: World, rng, length_f: int, width_f: int, min_x: int = 6, keep_from=(), min_gap: float = 0.0):
    """Find a spot for an object of length x width fine cells without cutting off any area.

    Returns (fine cells in body order, nav cells) or None.
    """
    cfg = world.cfg
    before = _flood(world.blocked, world.starts[0])
    cand = np.argwhere(before)
    for _ in range(300):
        cy, cx = cand[int(rng.integers(len(cand)))]
        if cx < min_x:
            continue
        horizontal = rng.random() < 0.5
        fx0, fy0 = cx * FINE + int(rng.integers(FINE)), cy * FINE + int(rng.integers(FINE))
        cells = [(fx0 + i, fy0 + j) if horizontal else (fx0 + j, fy0 + i)
                 for i in range(length_f) for j in range(width_f)]
        if not all(0 <= fx < world.W * FINE and 0 <= fy < world.H * FINE and world.obj[fy, fx] in (0, 3)
                   for fx, fy in cells):
            continue
        navs = sorted({(fx // FINE, fy // FINE) for fx, fy in cells})
        f = cfg.cell / FINE
        cxm, cym = (np.mean([c[0] for c in cells]) + 0.5) * f, (np.mean([c[1] for c in cells]) + 0.5) * f
        if any(np.hypot(cxm - x, cym - y) < min_gap for x, y in keep_from):
            continue
        if any(world.blocked[ny, nx] or not before[ny, nx] for nx, ny in navs):
            continue
        trial = world.blocked.copy()
        for nx, ny in navs:
            trial[ny, nx] = True
        after = _flood(trial, world.starts[0])
        if after.sum() != before.sum() - len(navs):
            continue                      # would wall off part of the building
        for nx, ny in navs:
            world.blocked[ny, nx] = True
        return cells, navs
    return None


def _place_people_and_decoys(world: World, rng, buried_fraction: float) -> None:
    cfg = world.cfg
    for k in range(cfg.n_victims):
        placed = _try_place(world, rng, 7, 2, keep_from=[(v.x, v.y) for v in world.victims], min_gap=2.0)
        if placed is None:
            continue
        cells, navs = placed
        skin, shirt, pants = SKIN[rng.integers(len(SKIN))], SHIRTS[rng.integers(len(SHIRTS))], PANTS[rng.integers(len(PANTS))]
        head_first = rng.random() < 0.5
        body = []  # (length index, fine cell)
        for idx, c in enumerate(cells):
            li = idx // 2
            body.append((li if head_first else 6 - li, c))
        oid = VICTIM_BASE + len(world.victims)
        for li, (fx, fy) in body:
            if li == 0:
                col, h = skin, 0.2
            elif li <= 3:
                col, h = shirt, 0.28
            else:
                col, h = pants, 0.18
            world.height[fy, fx] = h + rng.uniform(-0.02, 0.02)
            world.color[fy, fx] = np.clip(col + rng.normal(0, 10, 3), 0, 255)
            world.obj[fy, fx] = oid
        buried, covered = 0.0, []
        if rng.random() < buried_fraction:
            # cover a contiguous part of the body with rubble; the head often stays visible
            n_cover = int(rng.integers(2, 5))
            start_from_head = rng.random() < 0.25
            covered = [c for li, c in body if (li < n_cover if start_from_head else li >= 7 - n_cover)]
            for fx, fy in covered:
                world.height[fy, fx] = rng.uniform(0.35, 0.6)
                world.color[fy, fx] = np.clip(np.array([125, 118, 108]) + rng.normal(0, 14, 3), 0, 255)
                world.cover[fy, fx] = world.height[fy, fx]           # thick rubble: no heat gets through
            buried = len(covered) / len(cells)
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        f = cfg.cell / FINE
        end = lambda li: tuple(float((np.mean([c[i] for l, c in body if l == li]) + 0.5) * f) for i in (0, 1))
        world.victims.append(Victim(len(world.victims), (np.mean(xs) + 0.5) * f, (np.mean(ys) + 0.5) * f, cells,
                                    buried, navs, head=end(0), feet=end(6),
                                    colors={"skin": skin.tolist(), "shirt": shirt.tolist(), "pants": pants.tolist()},
                                    segments=[li for li, _ in body], covered=covered,
                                    cover="partial" if covered else "none"))

    kinds = {
        "jacket": (3, 2, lambda: SHIRTS[rng.integers(len(SHIRTS))], 0.07),
        "bag": (2, 2, lambda: SHIRTS[rng.integers(len(SHIRTS))], 0.3),
        "cardboard box": (3, 3, lambda: np.array([196, 160, 112]), 0.35),
        "wooden beam": (7, 1, lambda: np.array([120, 80, 45]), 0.2),
        "barrel": (2, 2, lambda: np.array([235, 110, 20]), 0.55),
    }
    names = list(kinds)
    for j in range(0 if cfg.building == "plain" else cfg.n_decoys):     # the test ground has no look-alikes
        kind = names[j % len(names)]
        L, Wd, col_fn, h = kinds[kind]
        placed = _try_place(world, rng, L, Wd)
        if placed is None:
            continue
        cells, navs = placed
        col = col_fn()
        did = len(world.decoys)
        for fx, fy in cells:
            world.height[fy, fx] = h * rng.uniform(0.85, 1.1)
            world.color[fy, fx] = np.clip(col + rng.normal(0, 12, 3), 0, 255)
            world.obj[fy, fx] = DECOY_BASE + did
        f = cfg.cell / FINE
        xs, ys = [c[0] for c in cells], [c[1] for c in cells]
        world.decoys.append(Decoy(did, kind, (np.mean(xs) + 0.5) * f, (np.mean(ys) + 0.5) * f, navs,
                                  size=((max(xs) - min(xs) + 1) * f, (max(ys) - min(ys) + 1) * f),
                                  color=[int(v) for v in col], height=float(h)))
