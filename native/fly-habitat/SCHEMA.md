# Habitat plan composition contract

`chreatures.fly-habitat-plan.v1` is the single current source for habitat
geometry and its finite material graph.

- `bounds_mm` and `spawns` use the imported MuJoCo model's millimetres.
- `geometries` contains named `box`, `capsule`, `ellipsoid`, and `cylinder`
  primitives. Static primitives compile to named fixed bodies. Dynamic grains
  and pods compile to named free bodies with explicit model-unit mass.
- `regions` uses metre coordinates required by `ecology-core`; every region
  names one physical geometry. Its eight capacities and initial quantities use
  the existing pool order from `finite-garden-v2.json`.
- `routes` only joins named regions. The Rust planner proves that the undirected
  material graph is connected. Route openness remains the native collision-ray
  measurement; the plan does not provide a path to a controller.
- `packets` binds finite stores to movable geometry. `colony_sites` binds living
  colony stores to selected elevated leaf geometry and its containing region.
- `information_boundary` is provenance. Geometry IDs, region IDs, stores,
  routes, seed, and spawn coordinates are not resident observations.

`native/fly-body/tools/compose_ecology_scene.py` requires this plan and compiles
its primitives together with independently prefixed, actual NeuroMechFly
bodies. It rejects any missing or extra compiled habitat geom. The resulting
physics manifest records the exact plan-file SHA-256, parameters, bounds, and
validation counts.

`native/browser-world/export_fixture.py` requires the same plan and checks its
file SHA against the physics manifest. It constructs the conservative ecology
from the existing eight-pool reaction definitions, uses the plan's regions and
routes verbatim, and binds physical geoms to region, packet, or organism stores.

