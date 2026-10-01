// Constants and every piece of explanatory text shown in the dashboard.

export const ROBOT_COLORS = [0x4f9cf9, 0xf0883e, 0xc084fc, 0x2dd4bf, 0xf472b6, 0xfacc15, 0x60a5fa, 0xa3e635];

export const STATUS = {
  hidden:  { color: 0xff5a5c, css: "#ff5a5c", text: "Not found yet", icon: "!" },
  sighted: { color: 0xffb020, css: "#ffb020", text: "Spotted – being checked", icon: "?" },
  found:   { color: 0x3fd46b, css: "#3fd46b", text: "Found", icon: "✓" },
  buried:  { color: 0xb39ddb, css: "#b39ddb", text: "Cannot be detected by any camera", icon: "⛏" },
};
export const FALSE_ALARM = { color: 0xff5a5c, css: "#ff5a5c" };

export const STATE_TEXT = {
  exploring: "exploring", scanning: "scanning 360°", verifying: "looking closely", verifying_trip: "going to check a sighting",
  transit: "driving to its zone", returning: "returning to base", reporting: "driving back into radio range",
  done: "finished, at base", idle: "deciding", backtracking: "backing out of a dead end",
  sweeping: "following its coverage pattern", charging: "charging at the base",
};

export const PLANNER_NAMES = { dijkstra: "Dijkstra", astar: "A*", rrtstar: "RRT*", aco: "Ant colony" };

// simulated seconds per real second offered in the transport bar
export const SPEEDS = [1, 3, 6, 12, 30];
export const DEFAULT_SPEED = 6;

// the three sensor sets a robot team can carry
export const SENSORS = [
  { key: "both", title: "Combined", text: "LiDAR maps the building, cameras find the people" },
  { key: "cameras", title: "Vision + Thermal", text: "Cameras only: AI reads colour and thermal images" },
  { key: "lidar", title: "LiDAR", text: "Laser only, no camera fitted: maps the building, cannot find people" },
];

// which technology's picture of the building the 3-D view draws
export const VIEWS = [
  { key: "combined", title: "Combined", sub: "real scene + findings", needs: null,
    help: "The building as it is, with the laser's hits and the warm spots the thermal cameras found." },
  { key: "lidar", title: "LiDAR", sub: "laser map", needs: "has_lidar",
    help: "What the laser knows: violet is where beams hit something. People on the floor stay dark: the beams pass over them." },
  { key: "thermal", title: "Thermal", sub: "heat image", needs: "has_cameras",
    help: "What the thermal cameras measured: warm things glow. Hatched floor has not been looked at yet." },
];

// what can be switched on and off in the 3-D view (Layers menu)
export const LAYERS = [
  ["flags", "Victim flags and bodies"],
  ["fog", "Shade what is not mapped / not searched"],
  ["paths", "Robot paths and goals"],
  ["strategy", "Strategy: who goes where, and why"],
  ["pattern", "Coverage patterns (lawnmower / spiral)"],
  ["trail", "Where each robot has driven"],
  ["fov", "Camera view cones"],
  ["sightings", "Sightings and false alarms"],
  ["zones", "Zones (partition strategy)"],
  ["search", "Planner search of the selected robot"],
  ["visits", "Repeat visits (cells driven through again)"],
];

export const PHASE_TEXT = { loading: "Loading", ready: "Ready", running: "Running", paused: "Paused", finished: "Finished" };

export const EXPLAIN = {
  obstacles: "Open test ground only, before the mission starts. Press Edit layout, then use Add (click free floor: a green ghost shows where the obstacle goes, red means the spot is taken) or Remove (click an obstacle: it turns red under the mouse). The list shows every obstacle; ✕ removes one, and hovering highlights it in the view. Undo, Default and Clear all work at any time before Start; Esc or Done ends editing. Pressing Start locks the layout; Restart unlocks it. Tall blocks and walls stop the LiDAR and the cameras. A low crate (30 cm) lies under the LiDAR's 40 cm scan plane: the laser cannot see it and the camera looks over it, so robots only notice it with the depth camera close up or the bumper. Obstacles that would seal off part of the ground are refused. Default restores the original layout; Clear all empties the hall.",
  vary: {
    "0": "Repeatable: the same settings give exactly the same mission, move for move (every random choice comes from the building's seed). This is what makes a fair comparison possible.",
    "1": "Vary each run: the building stays the same, but every run draws new random numbers for sensor noise, tie-breaks between equally good goals, RRT* samples and ant choices. Robots then take different paths in the same building.",
  },
  planner: {
    dijkstra: "Dijkstra: expands cells in order of their route cost g from the robot, so it always finds the shortest route, but it searches evenly in all directions. The same distance field also tells the robot how far every goal is, which it needs to choose a goal.",
    astar: "A*: expands cells in order of f = g + h, where h is the octile distance to the goal, max(dx, dy) + 0.41·min(dx, dy): the exact distance on an empty grid. h never overestimates, so A* finds the same shortest route as Dijkstra while looking at far fewer cells.",
    rrtstar: "RRT*: grows random trees of straight lines from the robot and from the goal until they meet, joining every new point through its cheapest neighbour and rewiring neighbours through it when that is shorter (radius γ·√(log n / n)); a rewired point passes its saving on to everything below it. Once a route is found, it samples only inside the ellipse |p−start| + |p−goal| ≤ best cost (Informed RRT*). The straight lines are drawn onto the grid with Bresenham's algorithm, so each costs exactly its octile length. Routes differ from run to run and narrow doors are hard to find by chance; then it falls back to A*.",
    aco: "Ant colony: 10 ants walk from the robot towards the goal. At cell i an ant picks the next cell j with probability ∝ τ_j^α · η_j^β, where τ is pheromone and η_j = 1 / (1 + detour), detour = step + h(j) − h(i) ≥ 0 being how much longer the move makes the best route still possible (h = octile distance to the goal; 0 when heading straight for it). α = 1, β = 4. Ants that arrive lay pheromone Q / L (shorter routes lay more), pheromone evaporates by ρ = 0.3 per round, and the best route is reinforced. It stops when the best route has not improved for 3 rounds.",
  },
  coverage: {
    frontier: "Frontier-based: go to the edge of the searched area where the most unsearched space is, weighed against the distance (value − 0.5 × metres). Adapts to any layout; the default.",
    boustrophedon: "Lawnmower (boustrophedon): the building is split into one rectangle per robot (as square as possible). Each robot sweeps its rectangle in K = ⌈D / w⌉ parallel lanes along the longer side (fewest turns), D / K apart, alternating direction, starting in the corner nearest to it. It drives from waypoint to waypoint (every 1.5 m) with the chosen planner, assumes unmapped ground is free, and skips waypoints inside obstacles. Afterwards it searches what the lanes missed (behind obstacles) by frontier exploration.",
    spiral: "Spiral: the building is split into one rectangle per robot. Each robot drives round its rectangle from the outside in: top edge, right edge, bottom edge, left edge, and every edge moves in by one lane spacing once driven, until the edges meet. Passes are at most w apart, as in the lawnmower. Afterwards the robot searches what the spiral missed by frontier exploration.",
  },
  avoidance: {
    none: "Follow the route: the robot drives cell by cell along its planned route. When the LiDAR or camera shows a new obstacle on the route, it replans at once; when a teammate is in the way it waits, and after 3 s it plans around it.",
    dwa: "Dynamic Window Approach (Fox, Burgard & Thrun 1997): every step the robot scores each reachable neighbouring cell with G = 0.45·heading + 0.25·clearance + 0.30·progress − 0.10·turn, where heading is the alignment with the route ahead, clearance the distance to the nearest obstacle or teammate (up to 3 cells), and progress the drop in route distance. It keeps clear of walls and steers around teammates instead of waiting.",
  },
  revisit: {
    "0": "Allowed: routes are the shortest possible, even if they run along corridors the team has driven many times.",
    "0.3": "Avoid: every cell costs 0.3 cells extra per earlier team visit (up to 5), so routes prefer ground nobody has driven over yet. The robots see new things on the way and get in each other's way less, at the price of slightly longer routes.",
  },
  deadend: {
    "1": "Recover: a robot that has visited at most 4 different cells in 24 s while driving somewhere (going back and forth), or that has been blocked for 8 s, backs off along its own trail by at least 4 cells and avoids that goal for 60 s.",
    "0": "Off: robots only rely on the normal replanning; a robot can stay stuck in a loop.",
  },
  speed: "Driving speed in m/s. A robot turns on the spot (180°/s), then drives: a move takes |turn| / 180°/s + distance / speed, so diagonal steps (0.71 m) take longer than straight ones (0.5 m). Faster robots finish sooner, but the cameras look once per second, so they see less on the way, and motor losses grow with speed².",
  battery: "Battery capacity, the same for every robot or different per robot. Power: computer 15 W + LiDAR 8 W + cameras 6 W all the time, plus driving m·g·C_rr / η = 20 J per metre (25 kg, C_rr 0.05, η 0.6) + motor losses 12·v² W, plus turning: about 42 W while driving at 0.5 m/s, so 1 Wh lasts roughly 85 s. Before taking a task, a robot checks it can drive there, work, and still get back: 1.3 × (energy there + energy back) + 5 % reserve must fit in what is left; tasks that do not fit are left for others. On the way it heads home as soon as only 1.3 × the drive back + 5 % is left. This floor is small (28 × 20 m), so only small batteries (1-3 Wh) run low within a mission; a real robot carries a few hundred Wh for a much larger site.",
  recharge: {
    "1": "Recharge: a robot that came home on low battery docks at the base (60 W charger: 1 Wh per minute), charges to full and goes back out, continuing its coverage pattern where it left it.",
    "0": "Stay: a robot that came home on low battery stays at the base for the rest of the mission; its unfinished area is left to its teammates.",
    drain: "Run empty: the robots ignore their battery. They never turn home and take any task, however far; when a battery is flat the robot stops exactly where it is and waits to be recovered. Compare with Recharge to see what the battery rules are worth.",
  },
  ranges: "How far the sensors reach. The camera range limits how far a robot can search for people (and sets the lawnmower lane width); the LiDAR range how far it maps walls. Longer range = fewer trips.",
  building: {
    office: "Open-plan office: clusters of desks, meeting rooms and private offices along the walls, and a solid lift/stair core in the middle.",
    apartments: "Apartment block: one corridor with flats on both sides. Each flat has a living room (sofa, dining table, kitchen) and a bedroom.",
    hospital: "Hospital: two corridors crossing, wards full of hospital beds, and a reception desk near the entrance.",
    school: "School: classrooms with rows of desks along a corridor, and a large gym with bleachers.",
    parking: "Parking garage: two rows of parked cars, pillars and a wide driving lane. Cars block the cameras and the laser; some still have a warm engine.",
    warehouse: "Warehouse: offices along the top and a big hall with pillars and tall storage racks that block the view.",
    plain: "Open test ground: one undamaged hall with a painted 2 m grid, no rubble, no look-alikes, no warm objects and nobody buried: made to watch how the coverage patterns, strategies and route planners behave. It starts with a few obstacles (a block in the middle, crates, cabinets in the corners, shelves and counters along the walls); place your own with the obstacle editor below."
  },
  damage: {
    light: "Light: few collapsed walls, little rubble. About 20 % of victims are partly buried and 5 % lie under thin debris.",
    moderate: "Moderate: some walls collapsed, rubble piles. About 40 % of victims are partly buried, 10 % lie under thin debris and 10 % are buried under thick rubble, where no camera can detect them.",
    severe: "Severe: many walls collapsed, lots of rubble. About 60 % of victims are partly buried, 10 % lie under thin debris and 20 % are buried under thick rubble, where no camera can detect them.",
  },
  known: {
    "1": "Known: the robots are told the number and stop as soon as they have confirmed that many. Faster, but a false alarm counts too, so a real person can be missed. You see every victim from the start.",
    "0": "Unknown: as in most real disasters. The robots search the whole building and you only see the victims they find. At the end you can reveal how many people were really inside.",
  },
  strategy: {
    coordinated: "Coordinated: the robots hold an auction. Every goal scores value − 0.5 × distance; the best robot-goal pair is assigned first and goals near it lose value for the others, so the team spreads out. Only one robot checks each sighting.",
    partition: "Partition: the building is split into one zone per robot (coloured floor). Each robot searches only its own zone, may drive through another zone to reach it, and goes home when its zone is finished.",
    greedy: "Greedy: every robot drives to its nearest goal and ignores its teammates. Simple, but robots often chase the same spot.",
    random: "Random: every robot picks any reachable goal at random. A baseline that shows how much a real strategy helps.",
  },
};

export const CAMERA_CAPTION = {
  colour: {
    fusion: "The AI reads this image together with the thermal one. Red tint: where it thinks a person is. Yellow boxes: detections.",
    cnn: "The AI reads this image. Red tint: where it thinks a person is. Yellow boxes: detections.",
    thermal: "Shown for you only: in this mission the AI reads the thermal image.",
    ideal: "Shown for you only: perfect-eyes mode does not use images.",
  },
  thermal: {
    fusion: "The AI reads this image together with the colour one. Green tint: where it thinks a person is. Yellow boxes: detections.",
    thermal: "The AI reads this image. Green tint: where it thinks a person is. Yellow boxes: detections.",
    cnn: "Shown for you only: in this mission the AI reads the colour image.",
    ideal: "Shown for you only: perfect-eyes mode does not use images.",
  },
};
