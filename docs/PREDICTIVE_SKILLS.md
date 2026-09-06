# Action-skilled predictive research organ

This isolated research lane tested a successor to the frozen production predictor without changing
its weights, native runtime, or foresight integration. The question is whether a
model that represents the world's effective action algebra can retain useful
action information for longer and expose model disagreement separately from its
Gaussian residual scale.

## Diagnosis

The retained dataset's scalar physiology changes slowly per 0.25-second macro:
energy, gut, and fatigue delta standard deviations are `9.84e-4`, `1.30e-3`, and
`1.47e-3`. Gut delta has zero median and 99th percentile because intake is a
sparse contact event. In contrast, angular-local delta has standard deviation
`9.62e-2`.

World effort uses magnitudes of signed thrust, yaw, and vertical/gaze controls.
Fatigue delta correlates `0.604` with absolute thrust in the training split, while
its strongest signed-action correlation is `0.178`. The v3 predictor receives
only signed actions, leaving its tanh action encoder to approximate this
sign-symmetric rule and preserve it through recurrent transitions. Grip and
signals are already clipped nonnegative, so duplicating their absolute values
would add no information.

## Research architecture

`research.predictive_skills` contains:

- the eight executed actions plus `abs(thrust)`, `abs(yaw)`, and
  `abs(gaze_pitch)` as an 11-value effective-action basis;
- a posterior GRU and action-only transition GRU;
- learned horizon embeddings for every horizon H1 through H16;
- a physiology decoder conditioned directly on latent state, horizon, and the
  current effective action basis;
- three independently initialized members. Their prediction variance is a model
  disagreement proxy, while each member's Gaussian scale remains conditional
  residual noise plus misfit.

Training retains the immutable whole-world split and fits its H1–H16 delta
normalizer from training worlds only. The fixed run used four epochs, four
960-row episodes, six contiguous chunks per episode, and 96 updates per member.
The three-member ROCm 7 run completed in `127.63 s` on persvati while the two VLM
services remained available.

## Heldout results

The ensemble mean beats persistence on aggregate features at all evaluated
horizons: RMSE is `0.527` versus `0.584` at H1, `0.549` versus `0.561` at H4,
`0.560` versus `0.580` at H8, and `0.564` versus `0.586` at H16. It also slightly
beats persistence on all six physiology channels at H16. The margins for scalar
energy, gut, and fatigue remain small.

Replacing recorded actions with another resident's same-time actions increases
feature MSE by `0.00238` at H1, `0.00379` at H4, `0.00649` at H8, and `0.01218`
at H16. Fatigue MSE increases by `3.96e-8`, `1.99e-7`, `4.75e-7`, and
`1.06e-6`. This is stronger action discrimination than v3, but the successor's
absolute feature RMSE is worse than v3 at shared horizons. Gut mismatch remains
slightly negative, consistent with sparse contact events that action alone cannot
identify.

Feature ensemble variance has Pearson correlation `0.25–0.27` with squared
heldout error. Energy correlation ranges `0.09–0.23`; gut is weak except at H8;
and several H16 motion/physiology correlations are near zero or negative. The
ensemble therefore supplies some useful unsupported-rollout signal, but it is not
calibrated OOD probability and is not uniformly informative by channel.

A repeated full heldout inference pass sampled sysfs GPU busy every 0.1 seconds:
72 samples, mean `38.8%`, median `46%`, p95 `69%`, maximum `72%`. This is a
shared-GPU observation with both VLM services live, not an exclusive throughput
benchmark.

The dataset predates the common-chemistry model and contains only the earlier
scalar E/G/F physiology. These results do not establish prediction of chemical
reserves or regulation. Offline predictive fit and action discrimination also do
not establish behavioral improvement. The successor remains research-only; v3
continues to be the runtime artifact pending root review.

Raw receipts are under `runs/predictive-skills/h16-ensemble-v1/`.
