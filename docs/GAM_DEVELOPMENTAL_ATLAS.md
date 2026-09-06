# Developmental GAM atlas

The first developmental atlas is fitted to the bounded observation from
`/tank/chreatures/runs/development/rich-sensorimotor-online-v2-seed20260913`.
That run stopped after update 39. It has no final result receipt: the atlas contains
exactly 119,808 transitions from 39 completed telemetry packets, ticks 0 through
4,991, and labels this source `bounded-interrupted-run`. The source was revision
`dfa4f1a`; telemetry after update 20 reflects learned parameters that were not
durably checkpointed when the process stopped.

[`fit_developmental_atlas.py`](../integrations/gam_mechanisms/fit_developmental_atlas.py)
normally requires `identity.json` and final `result.json`. Its explicit interrupted
mode instead requires `updates.jsonl`, a source revision, contiguous packets, and
agreement between the maximum recorded tick and the last completed update. It
verifies the rich telemetry schema, hashes every rollout packet, and creates a
compact table outside Git. Audit episode, world, and resident slots never become
predictors.

The atlas fits three actual transition responses with pinned native
`gamfit==0.1.259`: realized goal-code distance progress, world-reported physical
effort, and the mean absolute standardized residual of the deployed three-target
body-law bank. The last response measures where inherited body expectations remain
wrong over lived time; it is not a reward or welfare measure.

Predictors summarize permitted current physiology, MaleCNS activity, private GRU
history, previous and executed actions, achieved-goal distance and age, body-local
peripheral/foveal proximity and contrast, and elapsed resident ticks. Tensor smooths
model goal-distance by age and developmental-time by goal-distance interactions.
The pipeline persists the real `.gam` models and direct 25-by-25 prediction surfaces,
rather than drawing illustrative curves.

Only episode zero was observed, so worlds 0–2 train the first atlas and complete
world 3 is held out. The held-out world contains 29,952 transitions. The effort GAM
has RMSE `0.01863` against `0.14744` for the training-mean baseline. Goal-progress
RMSE is `0.63870` against `0.65100`, a small improvement. Only 2,787 held-out rows
are inside every old law domain; its residual GAM fails badly (`3.6375` versus
`0.8191`) and its implausible surfaces are retained as negative evidence, not as an
interpretable developmental effect. Full native models live on bulk storage at
`/tank/chreatures/runs/analysis/developmental-gam-atlas-v2/models`; their SHA-256
identities and the actual 25-by-25 prediction surfaces remain in the compact local
atlas report.

The atlas is descriptive conditional prediction, not a causal analysis. It cannot
silently replace the deployed law bank. A successor candidate requires an explicit
artifact, better complete-unit validation, matching runtime features, and separate
promotion by the integration owner.

## Rich-domain successor law

The separate `rich_body_laws_v2` candidate uses 196,608 completed rich-play
transitions from two fresh episodes at source revision `4c35042`. Physical worlds
0–2 in each episode train the exact native 12-feature body-law contract (147,456
rows); world 3 in both episodes is held out (49,152 rows). Its artifact embeds birth,
MaleCNS graph, sensorium, world-profile, port, physiology, target-unit, timestep and
source hashes. It is a candidate for a new birth and does not rewrite a running
resident.

Held-out RMSE improves over the training-mean baseline for encoded-speed change
(`0.02452` versus `0.02864`), energy cost (`0.00003138` versus `0.00005350`), and
fatigue recovery (`0.00013485` versus `0.00025516`). 42,846 held-out rows (87.17%)
are inside all fitted domains. On the interrupted online contexts, its exact native
feature domains cover 81,687 of 119,808 executed transitions (68.18%), compared
with 9,142 (7.63%) for the original bank. This is measured coverage, not evidence
that candidate selection improves behavior.
