### Route planners (40 random routes on the true map of the test ground)

| planner | valid routes | length / optimum (mean, worst) | work (cells / samples / ant steps) | time (ms) | fell back to A* |
|---|---|---|---|---|---|
| dijkstra | 40/40 | 1.000, 1.000 | 974 | 9.0 | 0 |
| astar | 40/40 | 1.000, 1.000 | 87 | 1.2 | 0 |
| rrtstar | 40/40 | 1.013, 1.085 | 116 | 47.1 | 0 |
| aco | 40/40 | 1.011, 1.089 | 1842 | 101.8 | 0 |

### Coverage patterns (1 robot, planner dijkstra, lane spacing 5.30 m, perfect vision)

| pattern | mission (s) | searched when patterns done | searched at the end | driven (m) | repeat moves |
|---|---|---|---|---|---|
| frontier | 388 | - | 100.0% | 166 | 57 |
| boustrophedon | 434 | 97.3% | 100.0% | 199 | 69 |
| spiral | 375 | 97.5% | 100.0% | 166 | 27 |

| pattern | robot | region | pattern | length (m) | waypoints reached | skipped | distance from the pattern (m): mean / 95% / max | pattern done at (s) |
|---|---|---|---|---|---|---|---|---|
| boustrophedon | R0 | 1 | 4 horizontal lanes, 5.3 m apart | 108.2 | 75/77 | #6 inside an obstacle, #46 inside an obstacle | 0.21 / 0.62 / 1.62 | 307 |
| spiral | R0 | 1 | 2 rings, clockwise, 5.3 m apart | 94.8 | 67/68 | #5 inside an obstacle | 0.29 / 1.12 / 2.38 | 288 |
