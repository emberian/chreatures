# Conservative 3D fields

`chreatures.fields.FieldEnvironment` is the chemical ecology layer for the
hollow garden. It is separate from MuJoCo: rigid bodies determine geometry and
moving source positions, while the field solver transports arbitrary scalar
channels through the space. The default channels are the three scents already
used by the resident sensory bridge.

The field is a regular finite-volume grid. Configuration dimensions are written
as `[nx, ny, nz]`; arrays are stored as `[channel, z, y, x]`. Concentrations are
mass per cubic meter. Each interior face computes one flux and applies equal and
opposite changes to the two adjacent cells. Habitat boundaries and faces touching
a solid cell have zero flux. A permeability value from zero to one scales each
remaining face, so porous regions, dense vegetation, baffles, and ventilation
paths can alter transport without inventing a new object kind.

Diffusion for channel `c` is

```text
flux(c, left -> right) = D[c] * permeability * (C[left] - C[right]) / spacing
```

Advection uses the face-centered velocity and the upwind concentration. Decay
and passive uptake use exact exponential factors inside each transport step.
Localized sinks remove a bounded mass in proportion to the available weighted
concentration. Point and Gaussian sources normalize their weights over fluid
cells before depositing, including when a source lies beside a solid. These
choices make the mass ledger explicit:

```text
mass_after = mass_before + injected - decayed - uptake + numerical_residual
```

The residual should stay near floating-point roundoff. The module demo injects
one unit beside a sealed wall and reports `1.0000000000000004` remaining mass, a
`-2.22e-16` residual, and zero concentration on the other side.

## Physical-world integration

Construct the field after the MuJoCo scene exists:

```python
from chreatures.fields import FieldEnvironment

field = FieldEnvironment.from_world(world, {
    "grid": [48, 32, 14],
    "integration_dt": 0.01,
})
```

`from_world` voxelizes actual world-space boxes, spheres, ellipsoids, cylinders,
and capsules from static entities. This includes rotated ramps, the raised walk,
and the compound underpass. Dynamic food and toys remain MuJoCo bodies rather
than frozen obstacles in the grid. Their current positions supply moving
chemical sources:

```python
sources = field.sources_from_world(world)
field.sync_dynamic_barriers(world.diffusion_barriers())
ledger = field.advance(0.05, sources=sources, sinks=local_sinks, flow=air_velocity)
odor_left, odor_right = field.sample([left_antenna_xyz, right_antenna_xyz])
```

`sources_from_world` derives emission from scent components and remaining food
amount. It retains an internal source key so successive positions deposit along
the traveled segment instead of skipping between cells. The sampled result is
only channel concentrations. Entity keys, object IDs, source locations, and
world coordinates never enter the organism's observation.

### Moving thin membranes

Hinged and sliding entities can opt into chemical coupling with a strict
version-1 component:

```json
{
  "type": "diffusion_barrier",
  "version": 1,
  "shape_indices": [0],
  "permeability": 0.0,
  "translation_epsilon": 0.004,
  "rotation_epsilon": 0.012
}
```

The selected shapes must be boxes on a `hinge` or `slide` entity. At the
physical step boundary, `PhysicsWorld.diffusion_barriers()` supplies their
current world poses to `sync_dynamic_barriers`. The field caches three
face-centered factor arrays and recomputes them only after a selected panel
moves by the declared translation or rotation tolerance. An oriented-box
intersection test handles both rotating leaf gates and translating panels.
Overlapping membranes use the least permeable factor.

This is a thin-membrane approximation. A panel scales diffusive and advective
transport across cell-center segments that intersect its current oriented box.
It does not turn cells solid, remove concentration from a newly covered cell,
displace fluid as it moves, generate pressure, or model hydrodynamics. Closing
a gate therefore preserves field mass at the synchronization instant; it only
changes later flux. The regular static solid and cell-permeability grids remain
unchanged.

Only explicitly selected panels participate. Creatures, food, balls, and
ordinary moving mechanisms do not become fluid bodies. Organisms receive only
the resulting local channel samples, with no barrier or assembly identifiers.
`data/habitats/counterweight-chemistry.json` demonstrates a sealed sliding
passage gate coupled to a pressure lift and a partially permeable hinged leaf.

Callers may pass source dictionaries directly. A source has `position`,
`channel`, `rate`, and optional `spread`, `key`, or `previous_position`. A sink
has `position`, `rate`, optional `spread`, and an optional channel; omitting the
channel applies the sink independently to every channel. Sources inject mass per
second and sinks request mass per second. The returned ledger reports what was
actually removed.

Flow may be a constant `[vx, vy, vz]`, a `[3,nz,ny,nx]` array, or a
`[nz,ny,nx,3]` array. It is a physical velocity in meters per second. A sealed
outer boundary remains no-flux even when flow points outward, so field mass does
not disappear through a numerical edge. A caller that wants exchange with an
outside atmosphere should represent it explicitly with boundary-adjacent sinks
and sources.

## Stability guard

Transport is explicit and uses a configured fixed integration step. Before any
mutation, every `advance` checks

```text
CFL = dt * (
    max|ux|/dx + max|uy|/dy + max|uz|/dz
    + 2*max(D)*(1/dx² + 1/dy² + 1/dz²)
)
```

against `max_cfl`, which defaults to 0.82. An unsafe configured step or flow is
rejected instead of silently clipping an unstable result. Outer calls such as
the 0.05-second organism tick are split into the fixed field step, with a bounded
`max_substeps` guard. The default hollow-garden grid and diffusion constants use
a 0.01-second field step.

## Permeability, uptake, and trails

The constructor accepts a scalar or grid-shaped permeability array. Declarative
`permeability_zones` can assign rectangular volumes without changing the solid
geometry:

```python
config = {
    "permeability_zones": [
        {"min": [3.0, 4.8, 0.0], "max": [4.6, 6.6, 1.2], "value": 0.18}
    ]
}
```

This can represent a slowly ventilated hollow while retaining true zero flux at
stone surfaces. Channel `decay` describes chemical loss; channel `uptake`
describes uniform background absorption. Local biological uptake belongs in the
sink list so its removed mass is measured separately.

Channels are arbitrary. A persistent memory trail is simply a slowly diffusing,
slowly decaying channel deposited by local secretion:

```python
trail_config = {
    "channels": [
        *three_scent_definitions,
        {"name": "trail", "diffusion": 0.004, "decay": 0.0008}
    ]
}
field.deposit(body_position, "trail", mass=0.0005, spread=0.08)
```

The trail records where secretion physically occurred. It carries no resident
name or social identity. Another body can sample it only at its own sensor
points, and airflow, walls, decay, and uptake alter it like every other channel.

## Persistence

`snapshot()` stores the normalized configuration, channel arrays, solid and
permeability grids, base flow, clock, moving-source history, most recent source
and sink parameters, cumulative mass ledger, and RNG state. `restore()` checks
array shapes, finiteness, nonnegative concentration, and RNG compatibility.
The arrays use float64 so a same-runtime continuation preserves both transport
state and mass accounting exactly.

Current fields use the Rust finite-volume kernel and write version-3 snapshots
with the `rust-face-v1` transport identity. Older version-1/2 snapshots are
imported as data into this one current engine. The old NumPy equation is retained
only in `scripts/probe_native_transport.py` as an independent reference. Fields
with membranes also save normalized topology, synchronized poses, exact face
factors, and raster update count. Restore verifies those factors against the
saved poses.

Build the required extension with the same interpreter used for the world:
`python native/world-kernels/build_extension.py`. Each field owns reusable Rust
flux/change buffers, and a single native call processes every channel and face
in each substep. Sources, sinks and the mass ledger still use the surrounding
Python/NumPy layer. The migration probe covers random permeability, solids,
advection, diffusion, moving sources, sinks, membranes, mass conservation and
exact restored continuation. The native and reference concentration arrays
matched bit for bit in the executed 48×32×14 three-channel cases.

A focused check operated the chemistry variant with the finite-mass transport
block. The coupled slide gate moved from `0.000 m` to `0.339 m`. After the same
one-unit release and 1.5 seconds of diffusion, measured mass immediately beyond
the passage was `0.000439` closed and `0.06934` open. Synchronizing the closing
gate over an already concentrated region left the complete concentration array
and total mass bit-identical. Saving during subsequent physical motion and
restoring both world and field produced exact 24-step continuation; rotating
the hinged leaf changed 39 cached face factors at the probe grid resolution.

Run the contained conservative-wall demonstration with:

```bash
python -m chreatures.fields
```
