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
ledger = field.advance(0.05, sources=sources, sinks=local_sinks, flow=air_velocity)
odor_left, odor_right = field.sample([left_antenna_xyz, right_antenna_xyz])
```

`sources_from_world` derives emission from scent components and remaining food
amount. It retains an internal source key so successive positions deposit along
the traveled segment instead of skipping between cells. The sampled result is
only channel concentrations. Entity keys, object IDs, source locations, and
world coordinates never enter the organism's observation.

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

Run the contained conservative-wall demonstration with:

```bash
python -m chreatures.fields
```
