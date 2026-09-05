# Lightweight articulated body

`chreatures.articulated.ArticulatedWorld` is an alternate body implementation for the existing MuJoCo habitat. It preserves the `PhysicsWorld` boundary: `sense(id)`, `advance(actions, dt)`, `command(...)`, `view()`, `snapshot()`, and `restore(...)` have the same roles. Nothing switches the default runtime to this class yet.

The body is an engineered synthetic mechanism described by `data/bodies/hexapod.json`. It is not NeuroMechFly, a fly anatomical reconstruction, or evidence of physiological fidelity. The higher-detail FlyGym route remains available for offline experiments, but the lightweight model is intended for interactive multi-resident simulation.

## Mechanics

Each resident is one free trunk with six two-link legs. Every leg has a hip yaw hinge, a knee hinge, and a colliding spherical tarsus. The compiled body therefore has twelve actuated hinge degrees of freedom, thirteen rigid links, and six independently contacting feet. Two named antenna sites sit at the ends of visible, non-colliding antenna shafts.

Continuous `forward`/`thrust` and `turn`/`yaw` values feed a tripod stance reflex. During stance, a hip sweeps the planted tarsus backward. During recovery, its knee flexes and its hip returns forward. Differential left/right drive produces yaw. Bounded proportional-derivative torques are written to MuJoCo generalized forces on every physics substep. Translation is produced only by joint motion, contact, and friction; the articulated implementation does not apply the base crawler's direct trunk traction. A bounded roll/pitch torque acts as a vestibular stance reflex and cannot translate the body or set its height.

This reflex is supplied body mechanics. It does not choose when or where to move, inspect goals, seek resources, avoid obstacles, or read world positions. A learned controller can later modulate continuous thrust/yaw or replace the twelve joint targets while keeping the same physical body.

The JSON mechanism file owns dimensions, mass densities, contact friction, hinge limits, tripod phases, servo gains, torque limits, and stance timing. The full mechanism definition is embedded into each snapshot. `ArticulatedWorld.restore(snapshot)` reconstructs the same definition and rejects a different compiled model signature.

## Interface

Instantiate it explicitly:

```python
from chreatures.articulated import ArticulatedWorld

world = ArticulatedWorld(seed=7)
sense = world.sense("mica")
outcomes = world.advance({"mica": {"forward": 0.7, "turn": 0.1}}, dt=0.05)
frame = world.view()
```

Inherited sensory fields remain present. The alternate body adds:

- `tarsal_contact`: six normalized contact loads in `lf, lm, lh, rf, rm, rh` order.
- `joint_position` and `joint_velocity`: twelve radians/radians-per-second values, ordered hip then knee for each leg in the same leg order.
- `antenna_position`: world positions of the left and right antenna tips. Odor is sampled at these actual sites.

`view()` retains the ordinary habitat fields and adds `body_model` and `articulations`. Each articulation exposes every link world pose, every resident geom type/size/world pose/color, every hinge angle/velocity/world anchor/world axis/range, and every tarsus or antenna site world pose. Each body entry also contains its corresponding `shapes` and `articulation`, allowing a renderer to update the real MuJoCo transforms without reconstructing the kinematic chain.

## Physical behavior and limits

The body walks forward and backward on the flat garden and turns through differential leg drive. A headless local MuJoCo 3.12 rollout after settling moved the default `mica` body about 0.78 m in 6 s at `forward=0.75`, while maintaining a trunk height near 0.075 m. A separate rollout initialized on the west ramp advanced about 0.66 m uphill and rose about 0.14 m in 8 s at `forward=0.9`. These are smoke measurements, not fixed trajectory promises: contact dynamics, starting pose, fatigue, collisions, and other residents alter the path.

The default west ramp begins above the surrounding ground, so a body approaching from the lower floor encounters its vertical end face rather than a continuous ramp toe. The articulated body can traverse the inclined surface through MuJoCo contact once physically placed on or connected to it; the habitat geometry must provide a reachable transition if autonomous floor-to-platform climbing is desired.

This model omits wings, compliant tendons, adhesive pads, detailed joint anatomy, muscle physiology, and FlyGym's validated NeuroMechFly morphology. Its contact loads and antenna positions are physical sensor sites, while odor remains the habitat's analytic field sampled at those sites. Joint torques and the stance reflex are fabricated motor capabilities and should be identified as such in experiments.
