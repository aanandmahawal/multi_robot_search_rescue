"""All tunable parameters of the search-and-rescue simulation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

# what the robots' AI looks at: the colour camera, the thermal camera, both, perfect (geometric) sight,
# or nothing at all (no camera on board: the robots can map with their LiDAR but cannot find people)
VISION_MODES = ("cnn", "thermal", "fusion", "ideal", "none")

# the three sensor sets the dashboard offers: name -> (vision, lidar)
SENSOR_SETS = {
    "lidar": ("none", True),        # laser only: maps the building, cannot recognise people
    "cameras": ("fusion", False),   # computer vision + thermal imaging; the depth camera does the mapping
    "both": ("fusion", True),       # the laser maps, the cameras find the people
}


@dataclass
class RescueConfig:
    # ---- disaster zone -------------------------------------------------------
    width: int = 56             # navigation cells (each CELL metres wide)
    height: int = 40
    cell: float = 0.5           # metres per navigation cell (the camera renders at cell / 2)
    building: str = "office"    # office | apartments | warehouse
    damage: str = "moderate"    # light | moderate | severe  (collapse, rubble, buried victims)
    n_victims: int = 10         # how many victims are really inside
    victims_known: bool = False # do the robots know that number? (then they stop once all are found)
    n_decoys: int = 12          # clothing piles / debris that look a bit like people (fool a colour camera)
    n_warm_objects: int = 6     # heaters, pets, hot water, laptops... that are warm but not people (fool a thermal camera)

    # ---- robots ----------------------------------------------------------------
    n_robots: int = 4
    camera_range: float = 5.0       # metres the RGB-D camera can see
    camera_fov: float = 90.0        # horizontal field of view, degrees
    camera_height: float = 0.55     # metres above the floor
    camera_pitch: float = -14.0     # degrees (negative = looking down)
    image_w: int = 64
    image_h: int = 48
    scan_on_arrival: bool = True    # spin 360 degrees when a waypoint is reached
    lidar: bool = True              # 360-degree laser scanner that builds the map; off = map with the depth camera only
    lidar_range: float = 8.0        # metres
    lidar_rays: int = 240           # beams per revolution (1.5 degrees apart)
    lidar_height: float = 0.40      # metres: height of the scan plane; anything lower is invisible to the laser
    lidar_noise: float = 0.02       # metres of range noise (1 sigma)
    lidar_dropout: float = 0.01     # share of beams that return nothing
    comm_range: float = 0.0         # radio range in metres; 0 = unlimited. Walls cut it to 35 %.
    report_timeout: float = 60.0    # with limited radio: return to base if a find is unreported this long

    # ---- decision making -------------------------------------------------------
    strategy: str = "coordinated"   # random | greedy | partition | coordinated
    vision: str = "fusion"          # cnn (colour camera) | thermal (thermal camera) | fusion (both) | ideal (perfect,
                                    # geometric) | none (no camera on board)
    detector_path: str = "models/victim_detector.pt"            # AI for the colour camera (RGB + depth)
    thermal_detector_path: str = "models/thermal_detector.pt"   # AI for the thermal camera (temperature + depth)
    fusion_detector_path: str = "models/fusion_detector.pt"     # AI for both cameras together
    thermal_check: bool = True      # physics, not AI: a "person" hotter than 40 °C is a heater or an engine, ignore it
    detect_threshold: float = 0.85  # heatmap probability that counts as a detection
    confirm_logodds: float = 6.0    # evidence needed to declare "victim found"
    confirm_frames: int = 4         # ...seen in at least this many separate camera frames
    confirm_needs_close_look: bool = True  # ...including at least one from within verify_distance
    reject_logodds: float = -1.5    # evidence below which a candidate is dropped
    verify_distance: float = 2.5    # a close look counts as strong evidence
    verify_value: float = 30.0      # utility of going to check a sighting that is surely a person...
    verify_by_belief: bool = True   # ...scaled by the current belief that it is one (weak sightings count less)
    utility_discount_radius: float = 4.0  # coordinated: frontiers near a teammate's goal lose value

    # ---- navigation (rescue/planners.py, rescue/coverage.py, rescue/motion.py) --------
    planner: str = "dijkstra"       # route to the chosen goal: dijkstra | astar | rrtstar | aco
    coverage: str = "frontier"      # order in which unsearched space is visited: frontier | boustrophedon | spiral
    avoidance: str = "none"         # local obstacle avoidance: none (follow the route, wait for teammates) | dwa
    speed: float = 0.5              # m/s driving speed (a diagonal cell, 0.71 m, takes longer than a straight one)
    turn_rate: float = 180.0        # deg/s turning on the spot (a differential-drive robot turns before it drives)
    battery_wh: float = 0.0         # battery capacity in watt-hours; 0 = unlimited
    obstacle_density: float = 1.0   # multiplies the number of rubble piles (1 = as the damage level says)
    revisit_penalty: float = 0.0    # extra route cost (cells) per earlier team visit of a cell, up to 5 visits
    deadend_recovery: bool = True   # detect loops / blocked corridors, back off and avoid that goal for a while

    # ---- run ---------------------------------------------------------------------
    max_steps: int = 900
    seed: int = 0                   # the building (and, unless run_seed is set, every random choice in the mission)
    run_seed: int | None = None     # set to vary a mission in the same building: sensor noise, tie-breaks, RRT*/ACO

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sensors(self) -> str:
        """Which technology the robots carry: lidar | cameras | both."""
        if self.vision == "none":
            return "lidar"
        return "both" if self.lidar else "cameras"
