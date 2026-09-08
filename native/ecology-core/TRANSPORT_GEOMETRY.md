# Physical route aperture contract v1

`transport_geometry.rs` supplies the same planner, measurement reduction and
private cache to the browser/Wasm host and native MuJoCo host. The conservative
material kernel remains the only transport engine. Aperture plans and raw world
geometry are physical-host data and never controller observations.

`RouteGeometryPlan::new(&regions, &routes)` builds 32 equally weighted area
samples for each route. The disk radius is `sqrt(cross_section_m2 / pi)`. A central disk cell and three annular rings contain 1, 7, 12 and 12 cells,
each of area A/32. Ring radial-square midpoints are 4.5/32, 14/32 and 26/32;
the middle 12-cell ring is rotated 15 degrees. The central cell is sampled at
the origin, avoiding the former four-ring central blind region. The perpendicular frame follows the least-aligned coordinate
axis; circle constants are fixed in Rust. Each area sample has two opposed rays
between offset copies of the actual region centers. Geometric ray length is the
actual center separation. The separately reported declared transport length
continues to determine molecular conductance.

The plan exposes:

- `endpoints_m`: globally deduplicated SI endpoint coordinates.
- `rays`: ordered pairs with `origin_endpoint`, `destination_endpoint`,
  `direction`, `max_distance_m`, `route_index`, `sample_index`, and `reverse`.
- `routes`: aperture radius, area, geometric and declared transport lengths, and
  the route's contiguous ray range.
- `sha256`: the complete canonical serialized plan hashed with an empty identity
  field. Plans are rebuilt from the immutable region/route config on restore.

Both hosts supply `free_distance_m` in ray order and `endpoint_inside_solid` once
per unique endpoint. Both query paths must use the same active collision-solid set:
`(geom_contype | geom_conaffinity) != 0`. This includes physical terrain,
constructed branches, movable grains and collision-enabled articulated geometry.
Visual markers and collision-disabled decorative/tissue meshes are excluded.
This is a collision-solid approximation, not permeability of every rendered
tissue. Current fixtures have no explicit contact pairs. If a future fixture
uses explicit pairs that override zero collision masks, both query paths must
include those pair geoms or reject the unsupported fixture. A no-hit or beyond-end
ray is clipped to its declared `max_distance_m`; negative MuJoCo no-hit sentinels
must be converted at the host boundary. Missing, nonfinite, negative, oversized
or wrong-plan measurements are rejected.

Endpoint containment is mandatory. Opposed raycasts alone can miss a solid that
encloses the entire segment, because both exit intersections lie beyond the
segment. Use actual signed MuJoCo `geomDistance` against a reusable point-radius
scratch sphere (`ROUTE_ENDPOINT_PROBE_RADIUS_M = 1e-8`). This conservatively
inflates solids by ten nanometres. Batch the shared endpoints and prune candidate
solids with their actual AABBs/bounding spheres. Compile scratch geometry only
when topology changes; moving a query probe does not require a new model or
`setConst` call. Unsupported containment queries must not be treated as clear.

`plan.evaluate(&plan.sha256, &free_distances, &inside_flags)` returns one fraction
per route. An area sample counts only when neither endpoint is inside a solid
and both directed rays reach their endpoints (1e-9 metre traversal tolerance).
The fraction is the number of clear samples divided by 32. It excludes
`base_open_fraction`; the existing conservative kernel applies that factor once:

```
G_pool = D_pool * cross_section / declared_length
         * base_open_fraction * sampled_open_fraction
```

The existing finite donor/receiver allocation and elemental accounting then
limit actual transfers. Geometry does not create advection. Supply zero flow
unless an explicit physical airflow source supplies a signed volume flow;
concentration differences or blocked apertures are not inferred wind currents.

`RouteGeometryState` owns the private cache. `new(&plan)` starts unmeasured and
closed. `refresh_due(&plan, tick, topology_revision)` requests measurement at
startup, after ten control ticks (0.1 seconds at the pinned 100 Hz), or whenever
physical topology changes. `refresh(...)` validates every measurement before
installing fractions. `openness(&plan, tick, topology_revision)` rejects stale or
unmeasured input. Do this before the next ecological transport step. A committed
construction/removal or collision activation/deactivation invalidates the cache
before the following transport step. Ordinary pose and size changes, including
moving grains and body motion, are observed on the fixed cadence.

Serialize the whole `RouteGeometryState` with the physical checkpoint, including
its plan hash, measured tick, topology revision and fractions. Restore the plan
from the same config, validate the cache, and call `refresh_due` before use. The
host's topology counter and geometry must restore together. Failed physical tick
transactions must also restore the cache; do not attach future measurements to
an older world. No compatibility default should silently insert all-open routes.

This is a finite straight-path aperture approximation to molecular diffusion,
not a diffusion PDE, a porous-material model, or a tortuosity solver. It does not
resolve detours around individual obstacles; alternate declared region routes
can still transport material. Structures thinner than the sampling spacing may
be missed. These limits are shared between hosts, rather than hidden behind a
single center ray or backend-specific openness rule.

The earlier plant GAM campaign ran before this sampler. Its route inputs stayed
at the host's unsampled all-open default; it supplies no physical clearance or
permeability evidence. The original raw receipts remain unchanged, with the
interpretation corrected by the committed
[scope amendment](../../research/fly_ecology_atlas/scope-amendment.json).

The one native check is `cargo run --release --example transport_aperture` in this
crate. It uses explicit analytical sphere-ray/containment inputs, exercises a
thin off-axis obstruction, an enclosing solid, removal, cadence/checkpoint
replay, and actual conservative material transfer. MuJoCo host integration is a
separate joined measurement and is not claimed by that check.
