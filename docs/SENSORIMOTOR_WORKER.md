# Sensorimotor worker research results

## Historical v1 result

The first achieved-goal worker was trained offline from 192,000 executed 20 Hz
resident-action steps: two 2,000-step episodes, 16 worlds, and three residents
per world. Whole worlds were separated before normalization or window creation:
12 training worlds, two validation worlds, and two heldout worlds. Model inputs
contain 351 permitted body-local senses, six local physiology/circuit values,
and executed actions; they exclude world coordinates, bearings, distances,
object labels, and identities.

The goal autoencoder completed 384 updates using training worlds only. It maps
four observations ending at an achieved future tick through
`1428 -> 256 tanh -> 64`, then reconstructs them through
`64 -> 256 tanh -> 1428`. The encoder was frozen before the recurrent worker was
created or optimized. Its reported heldout Huber error of `0.13236`, versus
`0.32927` for the training-mean baseline, and latent effective rank of `28.75`
were computed only in the final heldout pass after worker selection.

The recurrent worker completed eight epochs with 64-step TBPTT and selected
epoch eight using validation worlds. Validation action NLL decreased from
`2.74044` to `2.51928`. Training minimized a per-axis recurrent action NLL with
half the available weight assigned to all valid samples and half to quantized
action-change samples when changes were present. Achieved offsets were sampled
by choosing uniformly among five buckets (`1..2`, `3..5`, `6..10`, `11..20`,
`21..40`) and uniformly within each chosen bucket. No forward-consistency loss
was used; that remains a proposed experiment. The action decoder uses 65
categorical bins for signed thrust, yaw, gaze, and posture. Grip and signals use
a joint hurdle distribution with an inactive outcome and 32 positive bins; mode
decoding compares the full inactive probability with the most probable
active-bin joint probability.

The heldout worlds were evaluated once after selection. Mean action NLL was
`2.49668`. Permuting achieved goals across other residents increased NLL in all
five horizon buckets in both episodes, by `0.0276–0.0524`. This historical v1
control mixed goal changes with resident-specific distribution changes, so it
does not establish goal reliance. Signed-axis quantized accuracy remains low
(`0.065–0.245` across the two episodes); rectified-axis accuracy ranges
`0.406–0.616`.

Training ran on an AMD Radeon 890M with Torch `2.10.0+rocm7.0` and HIP
`7.0.51831`, completing in `16.212 s` while the existing VLM services remained
online. A process snapshot at elapsed five seconds reported 160% CPU and 1.9%
host memory. The run finished before the delayed sysfs sampler attached, so no
GPU-busy percentage is claimed.

The artifact remains research-only on persvati at
`/home/ember/chreatures/runs/sensorimotor-worker/v1-seed20260908/worker.pt`.
Its SHA-256 is
`df2a0bddd299758770fba2b3e315bdb4865c76e868f52b107a4ff832b7a44e3a`.
The exact public receipt is
`data/training/sensorimotor-worker-v1.receipt.json`; raw result, identity,
environment, and log records are mirrored under the ignored
`runs/sensorimotor-worker-v1/` directory. Weights remain in bulk storage for the
closed-loop probe.

The worker factorizes action axes, detaches carried recurrent state at TBPTT
boundaries, and was evaluated on only two heldout world slots across two
episodes. Oral action is still supplied by the physiology law rather than
learned by the worker. The paired v2 result below supersedes v1's narrow
goal-permutation interpretation.

## Paired v2 result

V2 trained matched goal-conditioned and goal-free workers at seeds `20260908`,
`20260909`, and `20260910`. Both modes used the same architecture, 192,000
executed actions, 12/2/2 whole-world train/validation/heldout split, 384 goal
autoencoder updates, eight worker epochs, efference-copy inputs, seeds, and
validation-NLL checkpoint rule. The goal-free worker retained the full goal
path and parameter capacity but received a fixed zero goal vector during
training and evaluation. Each selected epoch eight before the single heldout
evaluation.

Mean heldout action NLL was `2.49864` for the goal-conditioned worker and
`2.49085` for the goal-free control. Replacing a goal with another achieved goal
from the same resident and episode at a comparable horizon changed conditioned
NLL by `[-0.000148, 0.000082, 0.000169, 0.000037, 0.000449]` across the five
horizon buckets. The signs vary by seed and the effects are near zero. V2
therefore finds no offline goal reliance and corrects the confounded v1 claim.
It remains possible for physical closed-loop evaluation to find useful control,
but neither result establishes an attained skill.

Executed actions are strongly autoregressive: exact quantized persistence by
axis was `[49.46%, 46.89%, 29.47%, 61.42%, 51.15%, 57.87%, 50.89%, 28.62%]`
for thrust, yaw, gaze pitch, grip, three signals, and posture. Repeat-last-action
RMSE was lower than worker MAP RMSE on natural and action-change samples across
the reported axes. Worker MAP improved most stopping strata, which motivates
the separate closed-loop evaluator but does not overcome the paired negative
control. The achieved-goal construction and high action persistence allow a
worker to model action history while largely ignoring the goal.

The six runs took `16.34–16.68 s` each (conditioned mean `16.536 s`, goal-free
mean `16.485 s`) on an AMD Radeon 890M with Torch `2.10.0+rocm7.0` and HIP
`7.0.51831`. A 0.2-second sampler observed 375 GPU-busy readings: mean `74.9%`,
median `93%`, p95 and maximum `96%`. Existing VLM services remained online.
The raw checkpoints and results remain in ignored bulk paths under
`/home/ember/chreatures/runs/sensorimotor-worker/v2-*`; exact paths and hashes
are recorded in `data/training/sensorimotor-worker-v2.receipt.json`.
