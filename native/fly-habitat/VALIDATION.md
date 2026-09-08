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
`artifact-manifest.json`. A native joined check for the final B4 spawn layout is
pending. The preserved `native-host-startup-receipt.pre-clearance-layout.json`
belongs to an earlier spawn layout and must not be attributed to the current
scene. That earlier run completed one 0.01 s tick with 100 MuJoCo substeps,
BODY807 and retinal extraction, route aperture refresh, ecology, and actual
wing aerodynamics; it establishes executable coupling for that exact archived
scene, without a locomotor, flight, feeding, or learning competence claim.
