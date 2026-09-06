# Learned body-local sensorimotor skills

## Decision

The next controller should combine **hindsight goal-conditioned supervised
learning** with a **recurrent low-level policy**, using goals that are encoded
from the resident's own future observation histories. The resident should not
receive a world state, packet identity, bearing, distance, or position. A 4 Hz
manager chooses a previously achieved latent observation goal and a duration;
the worker observes and acts at the physical 20 Hz rate.

This is a bounded precursor to a Director-style hierarchy rather than a full
Director implementation. The existing predictive model is not accurate enough
in energy and gut channels to make imagined homeostasis the training substrate,
and the current acquisition data are too sparse to train a hierarchy wholly in
imagination. Real experienced transitions remain authoritative.

## Why this boundary fits the current failure

The chemical-encounter run requested eating on all 960,000 resident steps. It
recorded 36,727 generic contact-while-eating ticks but only 23 mouth-material
contact ticks. Somatic transfer requires an actual contact point within 0.045 m
of the head-front mouth point. Absorption and its physical reward occur in the
same 0.05 s step as ingestion. One mouth-contact tick per 41,739 resident steps
therefore measures sparse exposure to successful head alignment; it does not
measure a 41,739-step credit delay. Temporal credit or goal dependence must be
tested from action-conditioned histories at declared future horizons.

The actor currently chooses one eight-coordinate action every 0.25 s and holds
it for five physics ticks. That is a poor seam for correcting yaw, stopping,
posture, or collision response during the last centimeters of approach. The
new worker therefore runs every physics tick. Its goal is an observation
history, never an analyst coordinate.

## Methods inspected

### Goal-conditioned supervised learning

[Learning to Reach Goals via Iterated Supervised Learning](https://arxiv.org/abs/1912.06088)
turns each trajectory suffix into supervised tuples of current state, achieved
future state, action, and horizon. The
[author implementation](https://github.com/dibyaghosh/gcsl) samples a future
index from the same stored trajectory, uses that observation as the goal, and
trains action likelihood with an optional horizon encoding (`buffer.py` and
`gcsl.py`). This gives useful supervision even when the commanded goal was
missed.

Its published implementation assumes Markov state vectors and environment goal
extractors. Those assumptions are not acceptable here. We retain the hindsight
tuple construction, but replace each state with a recurrent encoding of only
the resident's anonymous observation/action history. Goals come from the same
encoder applied to achieved future histories.

### Visual reinforcement learning with imagined goals

[RIG](https://arxiv.org/abs/1807.04742) learns a visual representation, samples
goals in that representation, and relabels experience with achieved visual
goals. Its [project and code links](https://sites.google.com/site/visualrlwithimaginedgoals/)
show that the policy and reward can operate from images without ground-truth
state. This supports body-local sensory goals, but its VAE prior can generate
unreachable goals and latent Euclidean distance can emphasize appearance over
controllability. The first Chreatures version must therefore select actual
future encodings from replay; it must not sample arbitrary latent vectors or
turn latent distance into organism reward.

### Director

[Director](https://arxiv.org/abs/2206.04114) learns a world model from pixels,
compresses replay representations through a goal autoencoder, lets a manager
select latent goals, and trains a worker to reach them. The
[author repository](https://github.com/danijar/director) switches manager goals
at a configured skill duration, conditions the worker on recurrent world-model
features plus the goal, and trains both policies in imagined trajectories
(`hierarchy.py`; `configs.yaml`). Its egocentric ant-maze experiment does not use
global position or a top-down view, which matches our information boundary.

Director is the target decomposition, but importing its TensorFlow/Dreamer
stack would duplicate the native predictor and current PPO. Its manager also
depends on model reward accuracy. We adopt its manager/worker state ownership
and temporal boundary only after the real-data worker has measurable achieved
goal competence.

### Play-LMP

[Learning Latent Plans from Play](https://arxiv.org/abs/1903.01973) represents a
multimodal action sequence with a latent plan and decodes it with a policy
conditioned on current and goal observations. The
[maintained research implementation](https://github.com/Stanford-ILIAD/plato_sandbox)
uses fixed training windows and periodically replans the latent variable. This
is useful evidence for representing several physical solutions to the same
sensory transition.

Play-LMP is not the first implementation choice. Its results rely on broad
teleoperated play and goal observations that include a stable external camera.
Chreatures has no expert or human play corpus, and an external camera would
violate the body-local boundary. A stochastic plan latent can be added only if
the resident's own exploration data show genuinely multimodal action suffixes.

### LAPO and HILP

[LAPO](https://arxiv.org/abs/2312.10812) jointly trains a quantized inverse
dynamics encoder and latent-action-conditioned forward model, then grounds the
latent codes with action-labelled data. The
[author code](https://github.com/schmidtdominik/LAPO) implements product
quantization and separate inverse/forward stages. Chreatures already records
the exact eight actions, so action-free video recovery adds complexity without
information.

[HILP](https://arxiv.org/abs/2402.15567) learns a temporally structured Hilbert
representation and directional skills. Its
[author code](https://github.com/seohongpark/HILP) samples future observations
geometrically from episodes, but benchmark goal construction includes complete
simulator states and, for ant mazes, explicit XY goal coordinates. We should
not copy that goal interface. Its temporal-distance representation is a later
candidate once a recurrent observation-state encoder has been validated.

## Minimal implementation

### Recorded experience

Add a versioned collector record at every 20 Hz physical boundary:

- the 351 permitted encoded physical sense channels before and after action;
- the six existing local physiology/circuit values;
- the executed physical eight-vector, including the action held by the macro
  policy rather than its pre-squash sample;
- episode/reset boundary, resident-private sequence key, and physical `dt`;
- generic contact and mouth-contact outcomes for evaluation and stratified
  reporting, not as controller inputs or relabeled goals.

Do not store world positions, headings, entity IDs, material labels, target
assignments, or analyst distances in the learning dataset. Circuit readouts may
be stored as a separately identity-bound view, but the first worker should use
physical sense channels so its learned control is not tied to one graph.

### First recurrent achieved-goal worker

The implemented v1 goal autoencoder receives four normalized observations
ending at an achieved future tick and maps the flattened `4 x 357` window through
`1428 -> 256 tanh -> 64`; a symmetric decoder reconstructs the window. It is fit
only on training worlds, then frozen before any worker update. The worker's causal
GRU receives the current 357-channel observation, previous executed eight-vector,
oral command, and real reset boundaries. Its recurrent state summarizes the
available history rather than using a fixed 16-step input window.

Achieved goals come only from the same resident episode at offsets `1..40` ticks
(0.05--2.0 s). The trainer chooses uniformly among five buckets (`1..2`, `3..5`,
`6..10`, `11..20`, `21..40`), then uniformly within the selected bucket, and
passes the normalized log horizon to the policy. It optimizes factorized action
negative log likelihood with half the weight on all valid samples and half on
samples whose quantized action changed, when such changes exist. A loss that
predicts the frozen future code from the current state and recorded action suffix
is still a proposed forward-consistency experiment; it was not used in v1.

### Runtime seam

At a macro boundary, the 4 Hz actor proposes a bounded worker goal code and
duration. During each of the next five physical ticks, the worker consumes a
fresh actual observation and emits an action correction. Apply `tanh` bounds
and the existing nonnegative grip/signal projection. Limit each correction by a
versioned per-axis delta around the manager action during the first deployment.
Observation advances experienced recurrent state once; counterfactual goal
queries operate on copied state and cannot mutate it.

Do not initially let the manager emit arbitrary vectors. Its candidates are
codes from a resident-private replay reservoir of achieved recent windows,
plus the current observed code. Matching the current observation does not
itself guarantee stopping, and a previously achieved goal may be unreachable
from the present situation. The constraint establishes an experienced source
for each candidate, not current feasibility. A later goal autoencoder can compress and generate
goals after reconstruction and reachability tests justify it.

### Training order and acceptance

1. Collect varied anonymous physical trajectories from the existing stochastic
   motor cohort, including terrain, collision, turns, stopping, and packets.
2. Fit the goal autoencoder using training worlds only, freeze it, and then train
   the recurrent hindsight worker. Keep entire resident sequences together
   across truncated backpropagation and reset hidden state only at real episode
   boundaries. Select the worker using validation worlds only.
3. After model selection is complete, open the heldout worlds once to evaluate
   achieved-goal error, arbitrary-bearing turning,
   stopping distance, collision recovery, and the ratio of mouth contacts to
   all contacts. Coordinate-informed metrics may be used by the analyst but
   never passed to the model.
4. Compare the same frozen 4 Hz manager with worker disabled and enabled. A
   promotion requires improved low-level goal attainment and mouth/contact
   precision without increased invalid actions or loss of exact restore.
5. Only then train a manager to select achieved goal codes with the existing
   auditable homeostatic objective. Keep physical reward separate from worker
   supervised losses.

This design does not yet establish better acquisition or regulation. It defines
a trainable information boundary and a testable route from recorded body-local
experience to fast closed-loop motor competence.

## Research provenance

Scry OpenAlex queries on 2026-09-06 resolved the five papers above under record
IDs `0521c8e9-bd59-4054-97f3-8e009a5d5958`,
`f461532c-7ddf-4067-b644-daf2c1f33cd6`,
`f76c8cd8-0596-44cd-aa41-17e89571f4d7`,
`fea390e0-4a81-4f15-b385-41a66b7e9320`, and
`ede30d70-867b-4f63-b590-4f2819d3790d`. The corresponding author repositories
were inspected at their current default-branch heads; no third-party code was
copied into this repository.
