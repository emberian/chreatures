# Developmental GAM atlas

The developmental atlas pipeline is prepared for the completed rich online run at
`/tank/chreatures/runs/development/rich-sensorimotor-online-v1-seed20260913`.
It has not yet produced results because that run's immutable telemetry and final
receipt do not exist yet. No synthetic records or plots stand in for them.

[`fit_developmental_atlas.py`](../integrations/gam_mechanisms/fit_developmental_atlas.py)
first requires both `identity.json` and final `result.json`, verifies the rich
telemetry schema, hashes every rollout packet, and creates a compact table outside
Git. Audit episode, world, and resident slots never become predictors.

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

Episodes whose index is divisible by five are held out completely. Within remaining
episodes, a separate deterministic set of complete physical worlds is reserved for
validation. Reports distinguish row counts from episode/world units and compare
native GAM predictions with the training-response mean.

The atlas is descriptive conditional prediction, not a causal analysis. It cannot
silently replace the deployed law bank. A successor candidate requires an explicit
artifact, better complete-unit validation, matching runtime features, and separate
promotion by the integration owner.
