# Transactional evolving geometry

`PhysicsWorld.prepare_topology_batch(operations)` validates an ordered batch and
compiles its complete candidate MuJoCo model without changing the active world.
The caller can inspect its own ecological or metabolic budget after preparation
and before calling `transaction.commit()`. A commit checks the expected model
revision, swaps in the already compiled model, and preserves unaffected named
joint state, actuator controls, simulation time, RNG, resident physiology,
hand/grip state, touch state, contact history, and component contents.
Physics may advance between preparation and commit: dynamic state is read from
the live world during commit, so those intervening steps are retained. The
revision guard rejects only a competing topology commit, whose named-state
mapping could differ from the prepared candidate.

Supported operations are `add` with a complete declarative entity, `remove` by
ID, `replace` by ID with a complete entity of the same ID, and `append_shapes`
for an existing entity. A batch may touch each stable entity ID once. Shapes use
the ordinary habitat schema, so capsules, boxes, ellipsoids, and other existing
geometry need no plant-specific physics type.

Stable IDs do not make replacement a partial update. A `replace` takes its
geometry, physical properties, components, and initial resonance from the new
declaration. Live mutable properties are overlaid only for unaffected entities.
Use `append_shapes` when existing authored and runtime properties should remain.

Habitat `limits.entities` and MuJoCo compiler limits remain explicit hardware
and safety capacities. They are not ecological carrying capacities; ecology
must decide affordability before committing. Invalid candidates and stale
transactions leave the active world unchanged. Growth systems should aggregate
branch, leaf, or root changes into one transaction rather than compile per bud.

## Static structure affordances

A static entity can contain many locally positioned and rotated boxes,
ellipsoids, cylinders, spheres, and capsules. Capsule `fromto` endpoints make
branching coral, rails, and bridge members direct to express. Groups of boxes
and capsules can form nonflat floors, ramps, tunnel walls and ceilings, arches,
and bridges, with collision, friction, contact acoustics, occlusion, and sensing
handled by the ordinary world machinery.

The geometry is a union of convex MuJoCo primitives. It cannot subtract a
volume, so a tunnel must be assembled from separate floor, wall, and ceiling
shapes. Habitat entities currently cannot introduce triangle meshes,
heightfields, deformable terrain, breakable members, or per-shape physical
materials; density and friction come from the entity's physical material.
Static bodies do not respond to load, and equality assemblies apply only to the
existing movable joint types. The fixed MuJoCo contact and joint capacities are
compiler hardware limits rather than measures of ecological carrying capacity.
