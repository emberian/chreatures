# Recurrent predictive state organ

`chreatures.predictive_state` is a compact trainable dynamics organ. It closes a
specific gap in the current controller: the existing 32-value context is a fixed
random reservoir and its auxiliary predictor is only one-step. This organ learns
a private recurrent state from lived observations and executed actions, then
rolls candidate action sequences forward for horizons 1 through 8. Policy and
planner coupling remain separate.

## Resident-visible sequence contract

Inputs are the 384 anonymous normalized neural features, six local physiology
values, the eight executed motor actions, and reset/valid masks. There are no
names, scenario labels, rewards, positions, object kinds, account identifiers,
or unobserved world state.

Arrays are time-major. `action[t]` is executed after observation `t` and before
observation `t+1`; `reset[t]` clears recurrent state before consuming row `t`.
Episode directories append the separately recorded terminal observation so the
last executed action has a target. Training uses only contiguous chunks, applies
resets before observations, and excludes a configurable burn-in prefix from the
loss.

The real recorder fixes whole-world splits: world slots 4–15 are training data
and slots 0–3 are held out. Output statistics are fit exclusively on training
worlds. Adding later episode shards is an explicit refit with a new normalizer
identity; heldout data never changes the transform.

## Normalized residual dynamics

The posterior GRU consumes normalized `[features, physiology]`, the previous
executed action, and private recurrent state. A separate transition GRU advances
that state under candidate actions. For every horizon `h = 1..8`, decoder targets
are:

```text
feature_target[h] = (feature[t+h] - feature_mean) / feature_scale
phys_target[h] = ((phys[t+h] - phys[t]) - delta_mean[h]) / delta_scale[h]
```

`feature_mean/scale` and horizon-specific `delta_mean/scale` come from training
rows only. Physiology is therefore learned at its observed delta scale rather
than competing with hundreds of larger neural channels or a large global output
variance floor.

The physical-unit inference adapter returns:

```text
feature_mean[h] = feature_mean + feature_scale * predicted_feature[h]
physiology_mean[h] = observed_phys_anchor + delta_mean[h]
                     + delta_scale[h] * predicted_delta[h]
residual_scale = training_scale * exp(predicted_log_std)
```

It never clips predictions. It returns validity flags, and consumers must reject
invalid forecasts instead of turning them into plausible physiological values.
The observed physiology anchor advances only on an actual `observe` call;
imagination never advances it.

Training batches all valid rollout starts as `[starts * residents, horizon]` and
uses eight batched transition passes. The Gaussian NLL is explicitly balanced
50:50 between the 384 feature channels and six physiology channels. A latent
consistency loss aligns imagined states with detached future posterior states.
The reported residual scale represents conditional noise plus model error. It is
not calibrated epistemic or OOD uncertainty.

## Persistence and native contract

Checkpoints include model and optimizer state, update count, NumPy and Torch RNG,
configuration, and the complete normalizer arrays and identity. Immutable NPZ
exports pin every tensor's name, shape, dtype, and SHA-256 plus feature,
physiology, and action ordering; graph, port bundle, source normalizer, dataset
split, and source hashes when supplied; output-normalizer identity; top-level
`forecast_status`; and PyTorch GRU gate order.

Unknown provenance stays explicitly unknown for research-only artifacts. Native
snapshots bind the immutable artifact, tensor manifest, and input identity, so a
same-shaped different model cannot restore them.

Complete exports embed the upstream PPO `count`, `mean`, and `m2`
in float64. Their artifact and canonical moment hashes must match the recorded
training-input identity before export and again when the native adapter loads the
archive. `normalize_source_features(raw[B,384])` reproduces the original PPO
transform, including `m2 / max(count, 1)`, the `1e-5` variance floor, and
`[-5,5]` clipping, and returns contiguous float32. This lets a fresh resident
start from raw MaleCNS readouts without depending on a coincidentally matching
motor checkpoint. Version-1 artifacts remain readable for recorded normalized
sequences but reject raw-source normalization explicitly.

The current version-3 artifact also binds its cadence directly to the hashed
collector manifest: physics steps are `0.05` seconds, five physics steps form one
predictor observation, and the predictor interval is therefore `0.25` seconds.
The native adapter rejects a version-3 archive if this temporal contract is
missing, differs, or names another dataset-manifest hash.

`NativePredictiveCohort.observe(features, physiology, previous_action, reset)`
updates private experienced state. `query_from_snapshot(snapshot, actions)`
accepts actions shaped `[T,Bq,8]`, with `1 <= Bq <= cohort capacity`; a B1
snapshot is tiled internally. It returns feature and physiology means and
residual scales in physical units, validity, and horizon support without mutating
the experienced cohort.

## Executed first-shard smoke

A CPU smoke used all 960 rows and 36 training residents from episode 000, with
12 heldout residents and no heldout statistics. One epoch produced 12 updates.
Heldout feature NLL was `0.5623` at H1 and `0.5617` at H8; physiology-delta NLL
was `0.5180` at H1 and `0.6312` at H8. This validates the real sequence and
normalization path; it is not the production forecast model.

The production candidate then completed a fixed 96-update ROCm 7 budget over all
four episodes. Whole-world heldout feature NLL at H1/H2/H4/H8 was
`0.0746/0.0919/0.1294/0.1719`; physiology-delta NLL was
`0.1322/0.1323/0.1508/0.2034`. Physical-unit RMSE beat persistence at every
reported horizon: feature RMSE was `0.499/0.503/0.515/0.537` versus
`0.584/0.636/0.561/0.580`, and physiology RMSE was
`0.03954/0.03956/0.03980/0.03992` versus
`0.04044/0.04040/0.04053/0.04058`.

Final Rust comparison had maximum latent error `7.08e-7`; physical physiology
mean and scale errors were `2.98e-8` and `2.24e-8`. Snapshot continuation was
exact and a same-shaped other-model restore was rejected. This is a learned
short-horizon model; its residual scale still must not be treated as epistemic
confidence.

```bash
python scripts/train_predictive_state.py DATASET_ROOT \
  --output /tank/chreatures/predictive-state/run --device cuda
```

## Frozen action-discrimination assay

The final frozen export was evaluated on all four whole-world holdout shards.
Each posterior context was rolled forward three ways: its recorded future
actions, the same-time action sequence from the next heldout resident, and all
zeros. No weights or normalization statistics changed. This tests conditional
action use against the observed future; resident-shifted actions are a
distribution-preserving mismatch, not a causal counterfactual.

Recorded actions improved aggregate feature MSE over resident-shifted actions at
every horizon, from `0.00255` at H1 to `0.00504` at H8. They also improved the
largest action-responsive physiology channels: angular-local MSE by `3.73e-4`
at H1 and `3.41e-4` at H8, and fatigue MSE by `1.31e-8` at H1 and `2.98e-7` at
H8. Energy, gut, and support effects were much smaller; a few short-horizon or
zero-action comparisons were slightly negative. The model therefore carries a
real action-conditioned signal, strongest for neural features and angular
motion, while its homeostatic action discrimination remains weak and should be
weighted cautiously by foresight.

The raw receipt is
`runs/predictive-state/fullgraph-v1/action-discrimination.json`, SHA-256
`0ef6cad36a8a391ed238e30fcd313230ca752a7467e8f2df31611e490b253707`.
