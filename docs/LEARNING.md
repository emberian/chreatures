# Predictive PPO affordance learning

`chreatures.learning` is a versioned replacement for developmental learning
experiments. It leaves `AdaptiveOrgan` and existing residents unchanged. The
new system addresses a statistical error in that baseline: temporally
correlated exploration was previously evaluated with the score of an
independent Gaussian policy. The new actor samples an independent latent Normal
at each macro decision, applies `tanh`, and stores the latent value and complete
squashed-policy log probability. PPO recomputes the same density, including
the `log(1-a²)` Jacobian, before applying clipped likelihood ratios.

## Model and timescales

The full 351-input retinal-v1 interface drives the complete 165,122-neuron
MaleCNS graph at 20 Hz. Its 384 readouts are standardized with streaming
population moments. A learned 384→128 encoder combines those features with six
body-local values and a private 32-value recurrent context. A two-layer
128-wide trunk feeds the eight-coordinate actor and critic. A separate
action-conditioned head predicts the next change in a fixed 64-dimensional
projection of the standardized neural features.

The actor chooses every five physical steps, or 0.25 seconds, and the command
is held between decisions. Neural state, contact dynamics, digestion, energy,
fatigue and sensing continue at 20 Hz. The eight policy coordinates are thrust,
yaw, gaze pitch, grip, three signals, and posture. Eating remains a local
contact reflex modulated by gut and energy.

Generalized advantage estimates use continuous-time discount and trace
constants. PPO trains the encoder, actor, critic, log standard deviations and
predictor in vectorized GPU minibatches. Prediction learning progress can add a
small intrinsic return; indefinitely surprising input cannot. The external
return contains only improvement in bodily homeostatic potential,
hunger-weighted actual ingestion, and bounded actuator effort. Contact and
travel are recorded outcomes but are not reward terms. No object kind, color,
identifier, position, resource bearing, goal coordinate, or distance teacher
enters the learner.

The learned weights are shared population parameters during research training.
This is an inheritance optimization procedure, not a claim that three
residents in one world share personal learning. Each rollout resident retains
its own recurrent context, prediction-error traces, full MaleCNS state,
physical state and random trajectory. These private arrays are included in
continuation checkpoints and excluded from the exported genome.

## Curriculum and controls

`scripts/learn_affordances.py` uses `ArticulatedSensoriumWorld` with the native
batched retina. Sixteen persistent worker processes each own one MuJoCo world;
the parent process batches all 48 neural states on one GPU and trains the policy
there. This avoids serializing whole worlds on each step and lets native world
work span CPU cores.

The default neural backend is `TritonFusedCircuit`. Its HIP wave32 kernel fuses
CSR recurrence with each rate update while preserving the two global Jacobi
substeps. A full-graph B48 ABBA measurement found 27.286 ms per step, or
1,759.15 resident-steps/s, versus 159.442 ms and 301.05 resident-steps/s for
Torch CSR: a 5.843× kernel speedup. Complete-state parity was within 1.2e-7,
output parity within 2.4e-7, and same-backend checkpoint replay was exact.
`MicrobatchedResidentCircuit` remains available as `--brain-backend
microbatch`; its fastest exact Torch chunk size was three. Inputs enter as one
contiguous `[351,48]` float32 array; 384 features plus three neural physiology
values return in one device-to-host copy.

All episodes begin with adequate energy and nearby physical resources. The
first four episodes use resource bearings within ±0.75 radians as an easy
acquisition curriculum. Later episodes draw the complete ±π egocentric bearing
range. Held-out worlds always use the complete range. Nearby balls and boxes,
terrain, heading and resource placement vary across seeds. The connected ramp
and deck assembly moves as one rigid group so randomization cannot break its
approach geometry.

After training, the runner evaluates three policies on identical held-out world
seeds:

- the fixed initialized network with deterministic mean actions;
- the learned network with deterministic mean actions;
- the learned network with all 384 neural features zeroed before the encoder
  and private context update.

Evaluation reports actual ingestion, contact, travel, energy, effort and the
same bodily return. A useful sensor-dependent capability requires the learned
policy to improve over initialization and degrade under neural silencing. The
runner records failures without relabeling movement or predictor loss as a
skill.

`scripts/evaluate_connectome_sensitivity.py` runs the frozen learned policy and
normalizer through the canonical graph and the degree-matched recurrent rewire
on identical 16-world held-out layouts. It requires exact neuron identity and
ordering, reuses the same sparse input/readout maps, and changes only recurrent
topology. This tests sensitivity to the curated connectome; demonstrating an
anatomical learning advantage would additionally require training the matched
control from the same initialized policy.

## Reproducible hbox run

The isolated ROCm environment is `/tank/chreatures/envs/rocm-dev` with PyTorch
2.9.1+rocm6.3, SciPy 1.18.1 and MuJoCo 3.12.0. Bulk output and caches remain on
`/tank`:

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
PYTHONPATH=/tank/chreatures/learning-src-v3-20260905 \
TRITON_CACHE_DIR=/tank/chreatures/cache/triton \
PYTORCH_KERNEL_CACHE_PATH=/tank/chreatures/cache/torch-learning \
/tank/chreatures/envs/rocm-dev/bin/python \
  /tank/chreatures/learning-src-v3-20260905/scripts/learn_affordances.py \
  --graph /tank/chreatures/data/malecns/derived \
  --port-bundle /tank/chreatures/data/ports/retinal-v1-maps.npz \
  --output /tank/chreatures/runs/learning/affordance-16x3-v3 \
  --worlds 16 --steps 20000 --episode-steps 400 \
  --macro-steps 5 --rollout-decisions 64 --checkpoint-every 4000 \
  --brain-backend triton \
  --eval-worlds 4 --eval-steps 800 --workers 16 --seed 20260906
```

This starts an independent in-process GPU brain and research residents. It does
not call the live services on ports 8765, 8767 or 8768.

## Persistence and export

At update boundaries, a checkpoint consists of four checksum-linked files:

- a compressed full-cohort JSON with every exact articulated MuJoCo world,
  current observation, episode location and private identities;
- a neural NPZ with each resident's rates, adaptation, support and clock;
- a Torch checkpoint with model, optimizer, running moments, resident contexts,
  prediction traces, NumPy RNG and CPU/GPU Torch RNG state;
- a compressed rollout NPZ with every pending macro transition, including
  latent actions and old policy log probabilities.

Resume with the original arguments plus `--resume <cohort-checkpoint.json.gz>`.
Periodic checkpoints occur every 4,000 physical steps, keeping the large neural
state files on `/tank`.

`--first-checkpoint 320` can create an early full checkpoint after one PPO
update. `--restore-audit-only --resume <cohort-checkpoint.json.gz>` then loads
all three parts, constructs every world and verifies graph, port, resident and
checksum identity before exiting. A learner-only checkpoint can instead be
passed with `--warm-start-learner`: this inherits shared model parameters,
optimizer moments and feature normalization while explicitly resetting the
cohort, full neural state, private recurrent contexts and prediction traces.
The run record distinguishes this new-cohort initialization from exact resume.
Legacy version-1 signal checkpoints did not contain a partially filled rollout;
continuing one requires `--resume-drops-pending-rollout`, which records the
training discontinuity while retaining learned parameters, optimizer moments,
worlds and neural state. Version-2 checkpoints include that rollout. Moving an
existing `[B,N]` neural snapshot from the Torch backend to Triton preserves its
stored state exactly; subsequent dynamics agree to the float32 tolerances above
rather than promising bit-identical trajectories across backends.

`initial-genome.npz` pins the fixed comparison policy.
`learned-genome.npz` contains only shared model parameters and fixed feature/
context transforms. `import_genome` validates every name, shape and finite
value while preserving the recipient trainer's private context, normalization,
prediction traces and random state. `run.json` pins the command, graph, port
bundle, device and source hashes. Raw macro and PPO update streams are append-
only JSONL files; `summary.json` adds timing breakdowns, checkpoint receipts and
held-out results.

Continuation summaries distinguish `steps_advanced` and
`resident_steps_advanced` for the timed process from the absolute step
coordinate and cumulative policy exposure. Never divide a cumulative step
coordinate by a resumed process's elapsed time.
