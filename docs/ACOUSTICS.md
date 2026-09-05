# Physical acoustics

`chreatures.acoustics.Acoustics` is an optional environment layer for physically
triggered, three-tone sound. It turns bounded contact work and work extracted
from hinges into finite oscillator energy. It does not choose resident actions,
identify an actor, or assign reward.

## Runtime lifecycle

Attach the layer after constructing the physical world:

```python
from chreatures.acoustics import Acoustics
from chreatures.physics import PhysicsWorld

world = PhysicsWorld(seed=7)
acoustics = Acoustics(world, "data/components/acoustic-play.json")

world.advance(actions, dt)
acoustics.advance(dt)
sound = world.sense("pip")["sound"]
```

`PhysicsWorld.advance` supplies contact events and applies acoustic hinge loads
during each MuJoCo substep. Call `Acoustics.advance` once after the corresponding
world interval to decay stored energy, account for radiation, and advance its
clock. With no engine attached, the previous world sound behavior is unchanged.

Save the two states together. Restore the physical world first because acoustic
bindings are checked against its entity topology:

```python
saved = {"world": world.snapshot(), "acoustics": acoustics.snapshot()}

world = PhysicsWorld.restore(saved["world"])
acoustics = Acoustics.restore(world, saved["acoustics"])
```

The acoustic snapshot contains its normalized configuration, topology
signature, oscillator energy, contact cooldowns, event counters, clock, and
energy ledger. Restore rejects nonfinite, inconsistent, or mismatched state and
detaches the failed engine atomically.

## Reusable transducers

An emitter can be declared as an `acoustic_resonator` component on an entity or
bound externally in a version 1 acoustic configuration. External bindings let
one portable experiment instrument an existing stable habitat without changing
that habitat's compiled MuJoCo model. `data/components/acoustic-play.json`
binds four examples: a contact-sounding ball, a contact-and-hinge flap, a
seesaw pressure drum, and a hinge-driven pendulum bell.

Each emitter accepts these properties:

| Property | Meaning |
| --- | --- |
| `id`, `entity` | Unique transducer id and physical entity binding |
| `drive` | `contact`, `hinge`, or `both` |
| `tones` | Relative low, middle, and high oscillator weights |
| `energy_capacity`, `initial_energy` | Finite stored energy bounds |
| `capture_efficiency` | Fraction of eligible mechanical work stored |
| `impact_threshold`, `min_impact_speed`, `cooldown` | Contact trigger filter and refractory interval |
| `hinge_damping`, `max_hinge_torque` | Bounded opposing torque that harvests hinge work |
| `decay_time`, `radiative_fraction` | Oscillator decay and radiated share |
| `reference_energy`, `gain`, `range` | Energy-to-amplitude conversion and distance falloff |
| `occlusion` | Transmission through geometry blocking the direct ray |
| `source_offset` | Local offset from the current physical entity pose |

Only one emitter may bind a physical entity. Configurations are capped at 64
emitters. Unknown fields, invalid hinge bindings, nonfinite values, and values
outside their declared bounds are rejected before attachment.

## Physical and sensory contract

For contact drive, the world publishes an anonymous event containing the
entity, contact point, normal impulse, relative normal speed, and a bounded
impact-work estimate. It does not publish a resident or caregiver identity.
The contact transducer captures a configured fraction of dissipative impact
work up to its capacity.

For hinge drive, the transducer applies torque opposite the current joint
velocity. The captured source is the work removed by that torque during the
substep. This creates a measurable mechanical consequence while preserving the
declarative hinge and shape definitions.

The engine samples every source at its current 3-D pose. Amplitude is
proportional to the square root of tone energy, falls with squared distance,
and is multiplied by direct-ray transmission when geometry occludes the
source. The organism receives only the resulting three local sound values via
`sense`; it receives no source id, world coordinate, event record, or ledger.
Resident signals and legacy resonators use the same occlusion path while the
optional engine is attached.

The ledger records observed contact work, extracted hinge work, captured
energy, transduction loss, rejected capacity, radiated energy, and internal
oscillator loss. `view()` exposes energy and mechanical balance residuals for
environment diagnostics.
