# NumPy motor inheritance

`chreatures.motor_inheritance` deploys a trained `PredictiveActorCritic` on a
laptop with NumPy alone.  This is inherited population training, not personal
online learning.  The artifact freezes the learned policy, critic, predictor,
fixed feature/context transforms, and the final calibrated `RunningMoments`.
Each `MotorOrgan` owns its recurrent context, random stream, held macro action,
macro clock, previous normalized features and prediction.  Nothing in this
module changes an existing resident's anatomical substrate.

The controller accepts only a 384-value MaleCNS readout and six body-local
physiology values.  A physiology dictionary is mapped in training order:
`energy`, `gut`, `fatigue`, `tanh(speed/2)`,
`tanh(angular_velocity/4)`, `support`.  Passing a six-value array means it is
already scaled in that order.  Unknown dictionary fields are rejected so world
positions, object kinds, identities, and evidence cannot slip through this
interface.

The eight action coordinates are, in order, `thrust`, `yaw`, `gaze_pitch`,
`grip`, `signal_low`, `signal_mid`, `signal_high`, and `posture`.  The policy's
Gaussian latent is squashed with `tanh`.  As in the training world's adapter,
grip and the three signals are rectified to `[0, 1]` before being returned;
the other channels remain in `[-1, 1]`.  One sampled action is held for exactly
five calls to `tick`.  On the following call, the organ updates context from
the just-observed normalized features and the held action using the PPO model's
exact macro-boundary equation, then chooses the next action.

```python
from chreatures.motor_inheritance import MotorArtifact, MotorOrgan

motor = MotorOrgan("motor-step-0001230.npz", seed=41)
action = motor.tick(neural_features, {
    "energy": body.energy,
    "gut": body.gut,
    "fatigue": body.fatigue,
    "speed": body.speed,
    "angular_velocity": body.angular_velocity,
    "support": neural_support,
}, 0.05)
receipt = motor.snapshot("resident-motor.npz")
motor = MotorOrgan.restore(receipt["path"], expected_sha256=receipt["sha256"])
```

The snapshot is a pure NPZ and is self-contained.  It embeds the immutable
artifact metadata and arrays as well as all private continuation state.  The
artifact identity is a canonical SHA-256 over metadata, array names, dtypes,
shapes, and bytes, independent of ZIP timestamps.  The snapshot receipt also
provides a SHA-256 of the NPZ file.  Restore checks both identities and restores
the NumPy bit-generator state; replay was verified bit-for-bit across macro
boundaries.

Whole-world JSON manifests can store one shared artifact and compact private
resident values without creating one file per resident:

```python
shared = motor.artifact.to_value()
residents = {resident_id: motor.snapshot_value() for resident_id, motor in motors.items()}

artifact = MotorArtifact.from_value(shared)
motors = {
    resident_id: MotorOrgan.restore_value(value, artifact)
    for resident_id, value in residents.items()
}
```

Every encoded array carries its NumPy dtype, shape, and JSON-compatible data.
`snapshot_value()` references the shared artifact by canonical hash by default;
`snapshot_value(include_artifact=True)` creates a standalone embedded value.
`view()` reports the artifact identity, current context, held physical action,
decision count, held-tick position, and predictor error.  These are inherited
policy diagnostics and private working state, not a claim of personal learning.

A contextual selector can replace a macro decision without mutating private
fields.  At initial state, or after `open_macro_boundary(next_normalized)`, call
`forward(normalized, physiology)` and select a physical eight-value action,
then call `commit_macro_action(normalized, hidden, selected, dt)`.  The commit
requires an empty boundary, finite correctly shaped inputs, action values in
`[-1, 1]`, and nonnegative grip/signals.  It records the predictor input,
decision count, action, first held tick, and macro time.  Calls two through five
use `continue_macro_action(dt)`.  After the fifth tick,
`open_macro_boundary(next_normalized)` applies the inherited context equation
and makes the next commit legal.  This three-method seam retains the same
macro bookkeeping as `tick`; the selected physical action itself becomes the
action-conditioned context input.

## Export from a training machine

Torch checkpoints use Python serialization and must be treated as trusted
inputs.  The exporter therefore requires an explicit `--trusted-checkpoint`
flag and should run only against a checkpoint produced by this project.  The
resulting NPZ is loaded with `allow_pickle=False` and does not require Torch.

```sh
PYTHONPATH=. python scripts/export_motor_organ.py \
  --checkpoint /path/to/checkpoints/learner-step-0001230.pt \
  --output /path/to/motor-step-0001230.npz \
  --graph-sha256 48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625 \
  --port-spec-sha256 fffb48c65bdb5bc2503ff8ad7c65b4419e12aa9ef5b58b9f36bc910f64dadb6f \
  --port-bundle-sha256 56fdf4657358628843412c5d72a11c4464eea75127616ad0626bb7bb3f0865b2 \
  --run-record /path/to/run.json \
  --cohort-checkpoint /path/to/checkpoints/cohort-step-0001230.json.gz \
  --trusted-checkpoint
```

The artifact records checkpoint, graph, rich-port specification, port-bundle,
and optional run-record hashes, plus the learner configuration and update and
decision counts.  When a run record is supplied, the exporter refuses a graph
or port-spec mismatch.  A supplied cohort checkpoint also records its hash,
the canonical hash of its embedded physical specification, the native physics
model signature, and the body, sensorium, and chemical interfaces that the
snapshot establishes.  Explicit `--sensorium-interface`, `--body-interface`,
`--chemical-model`, and `--physical-spec-sha256` values are available for older
checkpoints; absent evidence is written as `unknown` rather than guessed.
When present, the run record also supplies the source habitat, physics module,
and sensorium module hashes independently of the randomized checkpoint world.

The step-1230 equivalence-check cohort has no `spec.sensorium` field.  Its artifact is
therefore labeled `legacy-world-v0-default (spec.sensorium absent)` with
`analytic-odor-default (spec.sensorium absent)`.  Its embedded articulated body
is `chreatures-lightweight-hexapod:v1`; this is not the newer `body-v1`
sensorium configuration.  Running this policy under `body-v1` with diffusion
is a cross-environment transfer and should be evaluated as such.

## Executed equivalence check

The exporter was run CPU-only against the then-current trained checkpoint on hbox,
`/tank/chreatures/runs/learning/affordance-16x3-v3-continuation/checkpoints/learner-step-0001230.pt`
(checkpoint SHA-256
`11a4dbf01c1cc041c63e9cd5af5c9eb46140c06f3a3a0f4c24b539e883270bc7`).
It contains 16 PPO updates and 1,078 decisions.  Twelve deterministic forward,
predictor, projection, and recurrent-context steps were compared with Torch.
The declared acceptance tolerance was `2e-6`; observed maximum absolute errors
were:

| Quantity | Maximum absolute error |
|---|---:|
| normalization | 0 |
| policy mean | 1.49e-8 |
| value | 7.45e-8 |
| hidden state | 1.64e-7 |
| predictor | 1.79e-7 |
| fixed projection | 9.09e-7 |
| recurrent context | 3.13e-7 |

This comparison covers the deterministic learned computation over multiple
context transitions.  Stochastic samples are not expected to match Torch
because NumPy and Torch use different random-number implementations; NumPy's
own stochastic stream is exactly restorable.  The deployed organ does no PPO
updates, episodic memory, reward processing, or weight mutation.  It exposes
the trained abstract thrust/yaw/posture interface rather than direct leg-joint
commands, and it does not add the training world's separate local eating
reflex.  Future personal episodic learning belongs in a separate organ and
must not mutate these inherited arrays.

## Nursery 8,000-step artifact

`data/genomes/nursery-8000.npz` is the exported trained checkpoint at 8,000
physical steps, 38 PPO updates, and 2,432 policy decisions.  Its receipt is:

- canonical artifact SHA-256:
  `48fbb38b21c1041b970a3ca793ac47dc25efc01f90d3a0e44d14d39772eddc76`
- NPZ file SHA-256:
  `4156eede91741948944b0c5d75d6917a8f1be77ced0ee2a4368260605a79bf60`
- source learner checkpoint SHA-256:
  `3264442875fbe16eabc2647b7c9cf0a7d07756f5f96eb7bee83568904a38ec59`
- linked cohort checkpoint SHA-256:
  `78a7970a7dabfa0fe1ae25ef2b546d69ab85970d5e3b80a2f14adcbff20bf4d8`
- MaleCNS graph SHA-256:
  `48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`
- rich-port specification SHA-256:
  `fffb48c65bdb5bc2503ff8ad7c65b4419e12aa9ef5b58b9f36bc910f64dadb6f`

In its recorded held-out evaluation, the learned deterministic policy ingested
`0.751397` total nutrition versus `0.671136` for the fixed initialization, an
11.96% increase.  This is a modest result from one evaluation configuration,
not evidence of general foraging competence.  Neural-feature silencing reduced
ingestion to `0.483120`, while other outcomes, including the shaped reward, did
not all move consistently; the experiment does not isolate which learned cues
caused the improvement.

The artifact retains its actual training boundary: legacy-world-v0 default
camera, analytic odor, and `chreatures-lightweight-hexapod:v1`.  Use under the
new body-v1 sensorium and diffusion chemistry is explicitly cross-environment
transfer.  It is suitable as an immutable baseline motor inheritance for new
residents; it has not been installed into or used to replace any live resident.
