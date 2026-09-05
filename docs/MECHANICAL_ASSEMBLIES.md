# Passive mechanical assemblies

Habitat version 1 supports passive couplings between authored one-axis entity
joints. This provides a small composition seam for linked platforms, gates,
levers, and counterweights without adding action names or scripted meanings.

## One-axis entities

An entity can use `mobility: "hinge"` or `mobility: "slide"`. Both use the same
joint structure:

```json
{
  "id": "pressure-lift",
  "mobility": "slide",
  "position": [7.5, 3.85, 0.0],
  "joint": {
    "axis": [0.0, 0.0, 1.0],
    "range": [0.0, 0.18],
    "damping": 0.55,
    "initial": 0.18
  },
  "material": "cyan",
  "physical_material": "mechanism-light",
  "shapes": [{"type": "box", "size": [0.6, 0.5, 0.035]}],
  "components": []
}
```

Hinge ranges and initial positions use degrees in the habitat file. Slide
ranges and initial positions use meters. Shape, visual material, physical
material, mobility, and sensory/ecological components remain independent.
MuJoCo supplies mass from geometry and density, gravity, inertia, damping,
contact, joint limits, and all resulting motion.

## Linear joint coupling

The top-level `assemblies` list accepts one deliberately narrow assembly type:

```json
{
  "id": "lifted-passage",
  "type": "joint_coupling",
  "joint_a": "pressure-lift",
  "joint_b": "passage-gate",
  "offset": 0.18,
  "ratio": -0.42857143,
  "solref": [0.012, 1.0],
  "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]
}
```

The constraint is `q_a = offset + ratio * q_b`. `joint_a` and `joint_b` name
entities, not raw MuJoCo identifiers. Each must be a hinge or slide, and a joint
can belong to only one coupling. This prevents accidental cyclic or duplicate
constraints while allowing multiple independent assemblies in one habitat.
The authored initial coordinates must satisfy the equation.

`solref` and `solimp` expose MuJoCo's bounded constraint response parameters.
The schema rejects unknown fields, invalid or duplicate ids, missing and static
entities, zero ratios, nonfinite values, unstable impedance domains, excessive
assembly counts, and inconsistent initial coordinates before compilation.

The compiler emits a native MuJoCo joint equality. No Python callback moves a
part in response to a contact or labels an interaction. Forces transfer through
the solver, so mass ratios, gravity, friction, joint damping, contact position,
and carried objects determine the result.

## Observation and persistence

`PhysicsWorld.view().assemblies` reports both physical coordinates and the
current constraint error for rendering and diagnostics. `sense()` does not
include assembly records, joint ids, or world coordinates. Moving gates and
platforms affect organisms through native collision, retinal occlusion,
contact, proprioception, light, chemical transport, and sound.

Joint coordinates and velocities are part of the existing
`mjSTATE_INTEGRATION` snapshot. Habitat topology and coupling parameters are
covered by the model signature. Mutable equality solver parameters are already
among the exact saved model fields. Specifications with no `assemblies` compile
to the previous XML and restore existing checkpoints without a compatibility
exception.

## Counterweight terrarium

`data/habitats/counterweight-terrarium.json` is a new variant of the terrarium
nursery. It adds a short approach ramp, a vertical pressure lift, a vertical
passage gate and frame, and a transportable block. The lift starts raised while
the gate blocks the direct floor passage. Their negative-ratio coupling makes
added mass on the lift lower the platform and raise the gate. Leaving a block
on the platform preserves the alternate route for another resident; removing
it allows the counterweight gate to descend.

The original terrarium mechanisms, food cycle, and optional
`data/components/terrarium-play.json` acoustic transducers remain compatible
with the variant. Existing terrarium and live habitat files are unchanged.

A focused operation used the bounded caregiver spring to carry the physical
block onto the lift and then released it. Contact weight lowered the lift from
`0.180 m` to `0.035 m` and raised the gate from approximately `0` to `0.339 m`.
A horizontal native geometry ray through the passage changed from occluded
transmission `0.1` to clear transmission `1.0`. Saving at that state, restoring,
and advancing both worlds for another 120 steps produced identical complete
world snapshots.
