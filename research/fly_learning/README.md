# Actual-fly developmental learning

This directory owns the training and corpus boundary for the articulated fly
body introduced after Anatomical CNS V3. It does not reinterpret the earlier
synthetic 12-joint body or its MOTOR34 artifacts. The body morphology, typed
afferents, motor atlas and centered motor-neuron decoder must be identity-bound
in every episode and fitted service.

The collection target is twelve independent physical worlds, each 1,024 control
ticks by four articulated residents. Worlds 0–7 train, 8–9 select checkpoints, and 10–11 are
untouched held-out world instances. Every world uses a fresh process, seeded
native state and exact initial snapshot. The first corpus uses one authenticated
B4 scene layout with counterbalanced histories; it does not claim twelve distinct
geometries. Separate composed pose/object layouts are reserved for the final
matched physical assay and a later expanded corpus.
Within worlds, resident-specific counterbalancing interleaves safe babbling,
author walking references, posture and support transfer, self-righting, forward
motion, turns, stopping, mouth reach and withdrawal, pump/salivary consequences,
and unconstrained consequence collection. Both viable and deliberately difficult
attempts remain in the corpus. The first corpus retains broader phase labels as
sampler strata, but its frozen host does not schedule a screen, point tones, or
physical terrain changes. Those labels are therefore not evidence for light,
acoustic, or terrain competence. The expanded nursery corpus uses actual screen
and point-tone schedules and separately hashed pose/object layouts before those
behaviors are eligible for a learning claim.

Author motor references are offline diagnostic targets. Closed-loop bouts deliver
zero, smooth OU, pulse and reversal context vectors through the annotated
descending-neuron interface; the CNS motor pools then drive the physical body.
Restored matched states provide context counterfactuals without exposing state
truth to the controller. Curriculum phase, geometry, joint targets, rewards and
outcomes are sampler, loss or evaluator values only.

Walking references use the pinned FlyGym `FlyBodyPreprogrammedSteps` experimental
42-joint kinematic trajectories and its six-leg adhesion phase schedule. The
collector consumes an immutable sampled trajectory bank so hbox does not depend
on mutable package import behavior. Turning changes left/right author trajectory
amplitudes; posture, head, antenna, proboscis and abdomen references are separately
labeled engineered diagnostic targets. The authored trajectory never runs in a
resident or production controller.

`data.py` deliberately exposes only mapped MaleCNS retina, typed body afferents, delivered
context and reset through `Episode.model_inputs`. Collected Z512 is an
artifact-bound replay cache. Joint and segment kinematics, contacts, sensory-site
positions, full 6×16 force/torque/position/normal/tangent contact records,
teacher actuation, success/failure and reward are targets. The loader
rejects extra arrays, mixed world identities, invalid control sources, mid-world
resets, context outside signed bounds, and malformed MOTOR92 values. Teacher
references may label CNS-controlled transitions for offline loss, but remain
noncausal targets.

The actual implemented 1,771-site retina is the sole visual array. Collection
does not synthesize or require the author's separate 721-ommatidium-per-eye
renderer. The private controller never reads retina values directly; only the
CNS receives the mapped optic tensor.

`delivered_motor` is the normalized CNS contract: 84 signed tanh outputs, six
adhesion probabilities, pharyngeal pump and salivary drive. `applied_body_control` separately receipts the
actual author-body values: 84 position targets in radians and six adhesion
values. This makes output scaling auditable without turning physical servo state
into another controller input.

Full-CNS training replays each chronology from its exact baseline. Recurrent
state is cached by world and resident; sampled windows burn 40 preceding ticks
before optimizing 64 ticks with checkpointed short segments. Sampling is
balanced across curriculum phase, success/failure and teacher/CNS control source.
The current motor boundary is MOTOR92: 84 named author joint-position targets,
six adhesion outputs, pharyngeal pump and salivary drive, decoded only from all 815 motor neurons through
body-part and side structural masks. It does not treat the 156 specifically
resolved muscle cohorts as an exclusive bottleneck: those annotations remain
evidence and auxiliary analysis views, while incompletely resolved antenna,
proboscis, abdomen and neck families remain available through explicit inferred
adapters. No inverse muscle-cohort target is fabricated from servo commands. The
centered signed decoder fits the actual author motor target. Reference rates and
scales come only from training worlds and remain explicit artifact tensors.

Physics advances at the author model's 0.0001 s step. One CNS/context/MOTOR92
transition spans exactly 100 MuJoCo substeps, giving a 0.01 s control interval.
The private goal horizon is derived as 40 control transitions from 0.4 seconds;
it is not conflated with the eight-phase acquired context-suffix capacity.

Action-conditioned prediction receives Z512, CNS-derived efference and delivered
context, never raw observer state. The canonical runtime ensemble fits every
suffix depth from one through eight control transitions; longer 0.4 s goals use
actually achieved endpoint keys in private memory. The private recurrent context organ receives
Z512, its own previous delivered context and a goal key recalled from achieved
Z history. It emits context12, value, termination and selection values; it cannot
emit muscle activation. Hindsight future keys may condition inverse-context
training, but future outcomes cannot enter the forward predictor or selector.
Physical reward weights offline losses and trains consequence statistics; it is
not a lifetime input.

The private model is the canonical native/Torch resident contract, not a new
research actor: GRU(524→256) consumes Z512 and previous delivered context12;
the goal encoder maps Z512+hidden256 to a normalized key128; four local proposals
come from hidden256+held goal128+previous context12. Three predictor members take
the canonical 908-vector and context suffixes, then predict Z512 deltas while the
shared core rolls forward. Candidate234 remains padded context96 followed by
remaining, phase and total durations in seconds divided by the 0.4 s goal
horizon and clamped to one, recalled flag, log evidence count,
recall score, lower-confidence consequence utility, forecast-valid, goal
progress, combined predictor/empirical uncertainty, and endpoint key128.
Recalled candidates use their actually achieved stored endpoint; local candidates
use the predictor ensemble mean. Per-phase consequence mean, sample variance and
count yield utility minus half an uncertainty term. These statistics affect
selection but never become sensory input.

Final evidence uses matched initialized, motor-bootstrap and learned-context
arms on held-out worlds with randomized task order. Reports include falls,
recovery, pose and trajectory error, antenna/mouth contact, effort, stopping,
termination, and measured context-to-DN-to-MN-to-body effects. This repertoire
does not gate memory, social behavior, development or ecology.

## Commands

Each collection world runs in its own process. The integrated native host
exports a small Python factory implementing the protocols in `collect.py`:

```sh
python -m research.fly_learning.collect \
  --factory research.fly_learning.native_host:create_bundle \
  --world-index 0 \
  --source-revision FULL_GIT_SHA \
  --scene native/fly-body/scenes/training-4/world.json \
  --service /tank/chreatures/runs/malecns-v4/seed/initialized-cns-v4.bin \
  --author-source native/fly-body/assets/author-step-bank-v1/trajectory-bank.npz \
  --output /tank/chreatures/runs/fly-learning/source/episode-00.npz
```

After all twelve independent files exist, seal without rewriting them and run
one substantial fit. The parent resident is the identity-bound initialized
context resident; omitting it deliberately runs only the goal-free CNS stage.

```sh
python -m research.fly_learning.train seal \
  --source /tank/chreatures/runs/fly-learning/source \
  --output /tank/chreatures/runs/fly-learning/corpus

HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m research.fly_learning.train train \
  --corpus /tank/chreatures/runs/fly-learning/corpus \
  --service /tank/chreatures/runs/malecns-v4/seed/initialized-cns-v4.bin \
  --parent-resident /tank/chreatures/runs/malecns-v4/seed/initialized-resident-v4.npz \
  --source-revision FULL_GIT_SHA \
  --run /tank/chreatures/runs/fly-learning/training/development-v1
```

The run writes append-safe progress, portable optimizer checkpoints, train-only
normalization tensors, a fresh CHCNS4 service, final-service latent replays, and
freshly bound resident/control artifacts. Offline validation cannot establish
locomotor competence; the final headless assay replays matched held-out worlds
with the trained service under zero context and under the learned resident.

The physical-stimulus nursery remains a separately sealed corpus because its
twelve scene and layout identities intentionally differ. Passing its manifest
adds the nursery train, validation and held-out worlds to the corresponding
whole-world splits; omitting it preserves the body-bootstrap recipe above.

```sh
python -m research.fly_learning.train train \
  --corpus /tank/chreatures/runs/fly-learning/body-bootstrap \
  --nursery-corpus /tank/chreatures/runs/fly-learning/nursery/nursery-corpus.json \
  --service /tank/chreatures/runs/malecns-v4/seed/initialized-cns-v4.bin \
  --parent-resident /tank/chreatures/runs/malecns-v4/seed/initialized-resident-v4.npz \
  --source-revision FULL_GIT_SHA \
  --run /tank/chreatures/runs/fly-learning/training/nursery-development-v1
```

The nursery loader verifies every episode hash, twelve distinct physical
layouts and initial snapshots, and all sixteen screen/sound bridge
acknowledgements. Stimulus plans and acknowledgements remain corpus provenance;
model ingress is still optic1771, BODY807, delivered context12 and reset only.

To continue from a body-bootstrap child while authenticating caches collected
through its initialized ancestor, name both services explicitly. Continuation
preserves the child's BODY807 and centered motor-neuron normalization so the
existing signed decoder keeps its numerical meaning; every raw nursery and
bootstrap chronology is then replayed through the child during optimization.

```sh
python -m research.fly_learning.train train \
  --corpus /tank/chreatures/runs/fly-learning/body-bootstrap \
  --nursery-corpus /tank/chreatures/runs/fly-learning/nursery/nursery-corpus.json \
  --collection-service /tank/chreatures/runs/malecns-v4/seed/initialized-cns-v4.bin \
  --service /tank/chreatures/runs/fly-learning/training/body-bootstrap/actual-fly-cns-development.bin \
  --parent-resident /tank/chreatures/runs/fly-learning/training/body-bootstrap/fly-context-resident.npz \
  --preserve-parent-normalization \
  --source-revision FULL_GIT_SHA \
  --run /tank/chreatures/runs/fly-learning/training/nursery-continuation-v1
```

## On-policy stability and recovery wave

`recovery.py` collects each trained child's own physical failures, an offline
teacher correction from that exact world state, and a subsequent CNS release
without resetting the fly. Four residents respectively exercise zero, smooth
OU, pulsed, and reversing private context through descending neurons. Two or
four independent B4 native worlds share one batched full-CNS model while
retaining separate recurrent state columns and episode files.

```sh
python -m research.fly_learning.recovery collect \
  --world-start 0 --cohort-width 2 \
  --output /tank/chreatures/runs/fly-learning/recovery/source \
  --scenes /tank/chreatures/runs/fly-learning/nursery/layouts \
  --native-binary /home/ember/chreatures-runs/fly-world/bin/chreatures-fly-world \
  --native-manifest /home/ember/chreatures-runs/fly-world/deployment.json \
  --service /tank/chreatures/runs/fly-learning/training/nursery/actual-fly-cns-development.bin \
  --author-source native/fly-body/assets/author-step-bank-v1/trajectory-bank.npz \
  --body-schema native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/schema.json \
  --motor-atlas research/fly_embodiment/fly-body-neural-atlas-v1.npz \
  --source-revision FULL_GIT_SHA
```

The next fit repeats `--collection-service` for every collection-time CNS and
adds `--recovery-corpus`. Thirty-five percent of batches begin at the real
cold reset and optimize all first 0.4 seconds; ordinary batches retain the
40-tick causal history. Continuous upright, support, stability and progress
measurements weight motor imitation without removing failures from consequence
training. A low-weight action-conditioned stability term differentiates
through the predicted absolute M92 action using frozen predictor parameters.
It does not expose pose or contact labels to CNS inference.

```sh
python -m research.fly_learning.train train \
  --corpus BODY_BOOTSTRAP/corpus.json \
  --nursery-corpus NURSERY/nursery-corpus.json \
  --recovery-corpus RECOVERY/recovery-corpus.json \
  --collection-service INITIALIZED/initialized-cns-v4.bin \
  --collection-service PARENT/actual-fly-cns-development.bin \
  --service PARENT/actual-fly-cns-development.bin \
  --parent-resident PARENT/fly-context-resident.npz \
  --prediction-heads PARENT/physical-prediction-heads.pt \
  --preserve-parent-normalization \
  --reset-prefix-fraction .35 --reset-sequence 40 \
  --viability-weighted-motor --motor-slew-weight .2 \
  --counterfactual-stability-weight .03 --motor-decoder-norm-weight .01 \
  --cns-motor-lr 5e-4 --source-revision FULL_GIT_SHA --run NEW_RUN
```

M92 remains an absolute actuator target. Slew supervision shapes its temporal
changes without adding an action integrator or a controller bypass. Reset
prefixes are drawn from the recovery corpus when it is present, because those
worlds deliberately begin with a stopping/stance counterfactual target rather
than an arbitrary locomotor phase.

The action-conditioned stability auxiliary starts from heads bound to the
exact parent service. A completed older run can export them without replaying
or optimizing anything:

```sh
python -m research.fly_learning.train extract-heads \
  --checkpoint PARENT/cns-checkpoint-000512.pt \
  --result PARENT/result.json \
  --output PARENT/physical-prediction-heads.pt
```
