# Executed habitat receipts

The frozen seed-20260908 scenes are staged outside the repository:

- B2: `/Users/ember/paperbin/chreatures/integration/fly-habitat-v1/seed-20260908-b2`
- B4: `/Users/ember/paperbin/chreatures/integration/fly-habitat-v1/seed-20260908-b4`

Both use an identical 80 x 60 x 30 mm geometry, material, and route payload.
Only the resident count, spawn list, and compiled resident arrays differ. The
plan has 225 habitat geoms, 156 physically bound material regions, 190
connected routes, 28 movable finite food objects, and four elevated colony
attachment surfaces. Spawn centers clear every static and movable primitive by
at least 1.2 mm using capsules, expanded axis-aligned bounds, or a conservative
orientation-independent bound for rotated primitives.

MuJoCo 3.12 advanced each compiled scene by 100 steps with finite body and
sensor state:

| Cohort | qpos | qvel | actuators | bodies | joints | geoms | sensor data |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | 462 | 432 | 180 | 366 | 282 | 367 | 192 |
| B4 | 728 | 696 | 360 | 506 | 536 | 509 | 384 |

The exact current artifact identities are recorded in each staged
`artifact-manifest.json`. The native host completed one joined 0.01 s B4 tick
with 100 MuJoCo substeps, full BODY807 and retinal extraction, 190 measured
route apertures, ecology, and 19,200 active finite blade-element evaluations.
The sealed receipt is `native-host-startup-receipt.json` beside the B4 scene;
its SHA-256 is
`e203af70bf18c3278116a57c19e9c2b718c3ec58cbdaac1994fec1a5d9928fcc`.
It binds the current world, physics, scene, and plan hashes exactly. This
establishes executable coupling, without a locomotor, flight, feeding,
learning, or ecological outcome claim.

The preserved `native-host-startup-receipt.pre-clearance-layout.json` belongs
to an earlier spawn layout and must not be attributed to the current scene.

## Long-horizon correction

The one-tick B4 receipt above did not establish longer physical stability. A
subsequent zero-MOTOR92 native run of the corresponding B2 scene failed at
control tick 92. MuJoCo reported a huge acceleration at DOF 285, the rotational
component of `ecology/grain-03/free`. Grain 03 initially penetrated bark
fragment 19 by 0.288778 model millimetres; grain 08 penetrated bark fragment 22
by 0.169464 mm. After rejecting colliding grain placements, an elevated movable
pod could still slide from its inclined leaf beyond the finite tiled ground and
free-fall. This remained finite for two seconds but its speed grew under model
gravity without a bound.

The corrected scenes are staged in `fly-habitat-v3`. The planner rejects every
ground grain candidate that overlaps existing terrain or another grain and
adds four passive perimeter collision walls around the declared volume. The
walls do not add routes, stores, targets, or controller information. On the
current native host with zero MOTOR92, corrected B2 completed 200 control ticks
(2.0 seconds and 20,000 MuJoCo substeps) and corrected B4 completed 120 ticks
(1.2 seconds and 12,000 substeps). All sampled qpos and qvel values remained
finite; the peak absolute generalized qvel value was 724.337 and declined
after contacts. Generalized qvel mixes free-joint translation in mm/s with
free-joint rotation and hinge velocity in rad/s, so that aggregate maximum has
no single physical unit. The exact scenes and measurements are bound by
`fly-habitat-v3/zero-motor-stability-receipt.json` outside the repository.
