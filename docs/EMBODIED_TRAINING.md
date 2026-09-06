# Current-life embodied training profile

`chreatures.training_environment` provides an explicit opt-in worker world for
the next training stage.  Existing affordance checkpoints remain legacy-world
v0 and restore through their existing path.  Merely importing this module does
not change a runner default, an existing resident, or a live world.

## Versioned contract

`EmbodiedTrainingProfile.current()` resolves and embeds the complete training
contract:

- articulated lightweight hexapod with the `body-v1` physical camera frame;
- the current five-by-sixteen native ray retina and physical antenna sites;
- finite-volume three-channel diffusion sampled only at those antenna sites;
- the finite terrarium resource reservoirs, producers, turnover, and growth;
- contact- and hinge-driven physical acoustics;
- the finite-energy homeostatic objective v1;
- structural-variation rules and hashes of every source module and asset.

`to_value()` contains the complete profile plus a canonical SHA-256;
`from_value()` verifies it.  A worker snapshot embeds that value rather than a
profile name or a path.  Restore reconstructs physical, diffusion, resource,
acoustic, RNG, and ledger state, verifies an optional expected profile hash,
and rejects clocks that differ.  This prevents a saved legacy environment from
silently acquiring the new sensorium or ecology.

`EmbodiedTrainingProfile.current_v2()` is a new opt-in contract rather than a
mutation of a v1 run.  It records a three-stage egocentric food-bearing
schedule of `[0.75, pi/2, pi]` radians.  A training spec and every v2 world
snapshot record the selected stage and effective half-span.  Held-out specs
always use `pi`, including when the training stage is narrower.  The v2
resource contract also raises conserved ambient material inflow to `0.008`
units/s, reservoir uptake by 3x, and producer growth by 2.5x.  Its three early
foods (`sun-berry`, `shade-nectar`, and `screen-seed`) are all backed by those
renewable reservoirs.  This makes repeated physical acquisition possible over
the declared long episode without creating food, repairing physiology, or
paying an oracle reward.

The learner observes only the existing encoded senses returned by `sense`:
occluded body-frame rays, bilateral local chemical concentrations, physical
sound, touch, illumination, shade, and body-local proprioception.  World
positions, entity kinds, names, identifiers, variation parameters, and
resource ledgers remain outside the policy input.  Positions appear in the
factory and diagnostic snapshots because physics and exact continuation need
them; they are never returned as sensory truth.

## Worker seam

The type is a drop-in replacement for the physical object currently owned by
one training worker:

```python
from chreatures.training_environment import (
    EmbodiedTrainingProfile,
    EmbodiedTrainingWorld,
    embodied_training_spec,
)

profile = EmbodiedTrainingProfile.current()
spec = embodied_training_spec(seed, held_out=False, profile=profile)
world = EmbodiedTrainingWorld(seed, spec, profile)

sense = world.sense(body_id)
outcomes = world.advance(actions, 0.05)
snapshot = world.snapshot()
world = EmbodiedTrainingWorld.restore(snapshot, expected_profile=profile)
```

Select v2 and its curriculum stage only at a new environment boundary:

```python
profile = EmbodiedTrainingProfile.current_v2()
spec = embodied_training_spec(seed, profile=profile, stage=1)  # pi/2
world = EmbodiedTrainingWorld(seed, spec, profile)

probe_spec = embodied_training_spec(
    heldout_seed, profile=profile, stage=1, held_out=True,
)  # always pi
```

The factory signature is `embodied_training_spec(seed, *, held_out=False,
stage=0, profile=None, base_spec=None)`.  Profile v1 accepts only stage zero
and retains its original serialized spec and snapshot shape.  Profile v2
snapshots use `chreatures-embodied-training-world-v2`, embed `stage`, and
restore only when that stage agrees with the physical world's stored spec.

For a process worker, replace only construction and restore:

```python
# reset
world = EmbodiedTrainingWorld(
    payload["seed"], payload["spec"],
    EmbodiedTrainingProfile.from_value(payload["profile"]),
)

# restore
world = EmbodiedTrainingWorld.restore(
    payload["snapshot"], expected_profile=payload["profile_sha256"]
)
```

`.bodies`, `.sense(body_id)`, `.advance(actions, dt)`, `.snapshot()`, and
`.restore(...)` match the existing worker boundary.  `advance` retains each
body's physical `nutrition`, `contact`, `distance`, and `effort`, and adds
`homeostatic_reward` plus named `homeostasis` components.  A runner can train
on that physical reward while logging prediction progress separately.
`last_telemetry` supplies aggregate timing-aligned nutrition, distance,
contacts, effort, homeostatic reward, field conservation, finite-resource
ledger, and acoustic energy/mechanical residuals.

The runner must store `profile.to_value()` and its hash in the top-level cohort
checkpoint and compare it before restoring worker snapshots.  New runs should
also copy the profile into `run.json` and exported motor provenance.  An exact
resume uses the embedded profile; choosing a different profile starts a new
training environment and is not a resume.

## Structural variation

The factory starts from `terrarium-garden.json`, preserves the connected
terraces and ramps, and varies resident headings, initial physiology, movable
food/play-object placement, and the positions of two independent occluders.
It then puts one ordinary finite food body 0.28–0.44 m from each
resident at a randomized egocentric bearing.  This supplies an early physically
reachable acquisition opportunity; eating still requires contact, food can be
moved or exhausted, and replenishment follows the independent ecology ledger.
V1 retains its historical nearby-food tuple, including one nonrenewable authored
food. V2 uses only the three ecology-backed renewable foods and expands their
bearing distribution by the explicit stage schedule above.
There is no resident-ID-aware feeder, oracle controller, or physiology repair.
Held-out worlds use a disjoint seed offset and wider occluder variation.  This
changes physical visibility, contact opportunities, local diffusion, movable
resource histories, and acoustic paths without adding routes, goals, or
teacher coordinates.  Training and held-out worlds keep the same sensor and
action schema.

This first profile still has limitations.  The lightweight articulated body is
engineered rather than a validated NeuroMechFly morphology.  The field grid and
ecology add substantial CPU and memory work to each worker.  The homeostatic
objective uses the world's dimensionless effort proxy rather than measured
positive joint work.  Structural variation is bounded so ramps remain
connected; it does not yet span new terrain topologies.

The profile declares 24,000-step (1,200 model-second) training episodes and
held-out evaluations, telemetry summaries every 1,200 steps (60 seconds), and
full checkpoints every 4,800 steps (240 seconds).  This duration spans
digestion and fatigue recovery many times and exceeds the observed inherited
policy collapse near 1,100 seconds.  These are requirements for a future
hardware run, not an executed result.  Fatigue follows its actual effort/rest
dynamics throughout; no within-episode reset or caregiver action repairs it.
On the existing fixed-width B48 circuit, compute estimates one 24,000-step
trajectory at roughly 22–25 minutes, with sixteen worlds providing more
evidence than four for similar GPU circuit cost.

## Executed worker continuation check

A local reference worker used a held-out structural variant with three actual
articulated residents.  It advanced five 50 ms ticks with nonzero thrust, yaw,
gaze, and posture, serialized the complete world through JSON, restored it,
and advanced both branches once more with identical actions.  The next physical
outcomes, body-v1 senses, diffusion values, resource/acoustic ledgers, finite
energy components, and aggregate telemetry were exactly equal.

At simulated time `0.30 s`, the measured telemetry included mean effort
`0.167288`, distance `0.00929335 m`, zero ingestion, homeostatic reward sum
`-0.00144008`, field CFL `0.0336`, field residuals below `2e-18`, resource mass
residual zero, acoustic energy residual zero, and acoustic mechanical
residual `2.22e-16`.  This establishes the worker and exact-restore seam; it
does not show that a policy learns under the new profile.  End-to-end training
on hardware remains a separate, explicitly selected run.
