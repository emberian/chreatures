# Static topology and conservative chemical fields

Physical growth and removal can change the static geometry compiled into a
`PhysicsWorld`. `FieldEnvironment.sync_static_geometry(world)` synchronizes that
geometry without reconstructing the field or resetting its chemistry.

Call it after the synchronous physical topology commit and before dynamic
membrane synchronization, source deposition, or transport:

```python
physical_transaction.commit()
static_report = field.sync_static_geometry(world)
field.sync_dynamic_barriers(world.diffusion_barriers())
field.advance(dt, sources=field.sources_from_world(world))
```

The method keys its fast path to the persisted `world.model_revision`. If the
revision is unchanged it returns `None` before requesting a world view or
allocating a raster. A changed revision rasterizes all current static shapes
into a candidate mask. Free bodies, residents, hinged parts, and sliding parts
remain outside this mask. Versioned moving membranes continue through their
separate face-permeability mechanism.

## Conservative replacement

The candidate is prepared before any live field array changes. Cells that cease
to be solid become empty fluid cells. Their permeability is restored from the
field's configured base permeability, including heterogeneous permeability
zones. Existing concentration in every other fluid cell is retained.

When a newly built solid covers concentration, the solver moves each affected
cell's complete channel mass to the nearest remaining fluid cell. Distance uses
physical grid spacing in z, y, and x. SciPy's exact Euclidean distance transform
provides a deterministic nearest-cell choice, and `numpy.add.at` combines cells
that share a destination. The operation does not decay, absorb, or invent mass.
It does not advance field time, RNG, moving-source history, source/sink records,
transport diagnostics, or the native transport solver.

The returned report records:

- revision and number of rasterized static shapes;
- newly solid and reopened cell counts;
- displaced mass by channel;
- mass-weighted mean displacement by channel;
- maximum displacement;
- mass before, mass after, and floating-point residual.

The snapshot keeps a bounded cumulative ledger containing total displaced mass,
mass-distance, maximum distance, rasterization count, and the latest report.
It does not retain an unbounded event list.

If a candidate makes every cell solid while any newly covered cell contains
mass, synchronization raises before mutation because no conservative destination
exists. Raster, concentration, permeability, revision, and ledger therefore
remain unchanged. A changed topology is also rejected if the world size differs,
the field exceeds its existing 2.5-million-cell grid limit, or the view contains
more than 65,536 static shapes. The configured grid remains bounded to 4–256
cells on each axis.

## Persistence

Field snapshot version 4 adds the base permeability hidden underneath current
solids, the synchronized world revision, and the static-topology ledger. Restore
checks that visible permeability equals base permeability with exactly the solid
cells zeroed. Versions 1–3 still import. Those older formats did not retain
permeability beneath solids, so migration reconstructs configured defaults in
those cells and preserves every saved fluid-cell value. New version-4 snapshots
restore this state exactly.

`PhysicsWorld` persists `model_revision`; a restored joined checkpoint therefore
takes the no-op synchronization path until another physical topology transaction
commits.

## Focused physical probe

Run:

```bash
.venv/bin/python scripts/probe_static_field_topology.py
```

The probe uses a real `PhysicsWorld` topology transaction to build a sealed wall
through an existing diffusing trace. The wall made 400 cells solid and moved
`0.1399243593` mass by `0.1 m`; total mass remained exactly `1.0`. During the
closed interval, right-side mass stayed bit-identical at `0.0644914171`. Removing
the wall reopened all 400 cells with permeability 1, after which right-side mass
rose to `0.1031511875`. Snapshot restoration was exact, and a mass-bearing
all-solid candidate was rejected without changing the field snapshot.
