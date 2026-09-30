# Reliability of the victim search

12 missions per sensor set (6 damaged buildings x 2 seeds), 4 robots, coordinated strategy, 10 victims, 12 look-alikes, 6 warm objects per building. Every number is checked against the ground truth, which the robots never see.

## Summary

| sensors | victims found / findable | confirmed reports | of them real | person counted twice | false alarms (nobody there) | precision | rejected sightings at a real person* | seconds with robots disagreeing |
|---|---|---|---|---|---|---|---|---|
| colour camera AI | 80 / 102 | 93 | 79 | 1 | 13 | 85% | 6 | 0 |
| thermal camera AI | 99 / 102 | 118 | 98 | 4 | 16 | 83% | 6 | 0 |
| colour + thermal fused | 100 / 102 | 113 | 99 | 2 | 12 | 88% | 3 | 0 |

* usually a second, weaker sighting of a person who was confirmed from another sighting; the missed victims are listed per sensor set below.

## colour camera AI

What the confirmed reports really were: person 79, bag 8, laptop left running 2, jacket 1, person counted twice 1, wooden beam 1, sun-warmed rubble 1.

What the rejected sightings really were: nothing 27, person 6, jacket 6, wooden beam 6, car engine 4, sun-warmed rubble 3, laptop left running 2, space heater 2, bag 2, cardboard box 2, barrel 2, hot-water leak 1.

Victims missed, by how they lay (missed / all): in the open 2 / 52, partly buried 2 / 32, under thin debris 18 / 18, under thick rubble 18 / 18.

## thermal camera AI

What the confirmed reports really were: person 98, sun-warmed rubble 8, person counted twice 4, hot-water leak 2, nothing 2, dog 2, barrel 1, laptop left running 1.

What the rejected sightings really were: nothing 47, dog 8, person 6, hot-water leak 5, laptop left running 4, jacket 4, cardboard box 3, bag 3, barrel 3, wooden beam 2, sun-warmed rubble 1, car engine 1.

Victims missed, by how they lay (missed / all): in the open 0 / 52, partly buried 3 / 32, under thin debris 0 / 18, under thick rubble 18 / 18.

## colour + thermal fused

What the confirmed reports really were: person 99, dog 5, sun-warmed rubble 3, nothing 2, person counted twice 2, jacket 1, hot-water leak 1.

What the rejected sightings really were: nothing 35, barrel 6, sun-warmed rubble 6, jacket 5, hot-water leak 5, dog 4, bag 4, person 3, laptop left running 3, wooden beam 2, car engine 2, space heater 1, cardboard box 1.

Victims missed, by how they lay (missed / all): in the open 0 / 52, partly buried 2 / 32, under thin debris 0 / 18, under thick rubble 18 / 18.
