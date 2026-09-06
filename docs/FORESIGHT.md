# Private predictive foresight

`chreatures.foresight.ResidentForesight` is an optional candidate-evidence
organ for a resident. It does not generate unrestricted actions. At a motor
macro boundary it evaluates the existing inherited candidate set with a
recurrent predictive model, then returns bounded score corrections accepted by
`ContextualMotorRefiner.candidate_evidence`.

Each resident owns a batch-one `NativePredictiveCohort` for actual experience,
a separate query cohort sized for at most 9 candidates × 4 branches, an RNG,
the last actual observation/action, and one selected intention tail. Immutable
model weights may be shared at the artifact level; these mutable values are
private and included in the resident snapshot.

## Actual and imagined state

Call `observe(features, physiology, previous_action, reset=...)` exactly once
per actual motor macro boundary. Features must match the predictive artifact's
locked input identity. Physiology is ordered energy, gut, fatigue, local speed,
local angular velocity, neural support. The action is the physical eight-vector
executed before this observation. Only this method advances the experienced
recurrent state.

Pass `foresight.candidate_evidence` as the contextual refiner's evidence
callback. The callback receives the immutable candidate tuple. It creates four
eight-action branches per candidate in one native batch. Every branch's first
action is exactly its candidate. Later actions use smooth AR perturbations in
atanh space with fixed sigma 0.18; grip and the three signal channels are
rectified. When a prior intention exists, branch zero follows that tail after
the new exact first action.

The query cohort's `query_from_snapshot(snapshot, actions)` starts from a frozen
copy of actual state and returns raw physical-unit forecasts in one native
call. It cannot update the experienced cohort. A repeated query with the same
float32 candidates returns the cached report and consumes no RNG. Query noise
is therefore fixed before downstream selection and does not depend on private
actor parameters.

After the real selector acts, call `commit_executed(action)`. If exactly one
queried candidate matches the executed float32 action, the organ retains that
candidate's best valid predicted tail. An externally changed action or an
invalid forecast clears the tail. The next `observe` incorporates what actually
happened; imagined states are never inserted as experience.

```python
foresight = ResidentForesight(predictive_export)
foresight.observe(features, physiology, previous_action, reset=first_macro)

decision = refiner.refine(
    memory, features, local_physiology, policy_mean, policy_log_std,
    candidate_evidence=foresight.candidate_evidence,
)
foresight.commit_executed(decision["action_vector"])
```

## Scoring and evidence limits

The native model returns absolute predicted physiology in physical units.
Foresight rejects a branch when any energy, gut, or fatigue forecast is
nonfinite or outside `[0,1]`; it does not clip an invalid trajectory into a
plausible one. Valid trajectories are scored as discounted consecutive changes
in `FiniteEnergyObjective.potential`, including reserve, fatigue, and gut
overload terms. Scores are averaged over valid branches and converted to a
correction relative to candidate zero, then clipped to ±0.12. This preserves
the inherited and relational scores as independent parts of final selection.

Diagnostics report each candidate's physical forecast score, valid branch
count, mean E/G/F residual scales, horizon support, model artifact hash, and
input identity hash. Decoder residual scale combines stochastic variation and
model error. It is a heuristic residual scale, not calibrated confidence or an
epistemic uncertainty estimate. Model training status is copied into every
report. An untrained or research-scope model can execute the same mechanics,
but its output remains explicitly research status and provides no evidence of
behavioral benefit.

Snapshots embed the versioned foresight and finite-energy configurations,
model and input identities, exact native actual state, RNG, actual observation,
executed action, intention tail, and any frozen query cache. Restore requires
the same immutable export and rejects identity mismatches.

## Validation status

The Python orchestration first passed a deterministic batched seam check using
a candidate-sensitive stand-in. It then ran on hbox against the trained real
episode-000 physical-unit export, artifact SHA-256
`12b0faf597fa603af7da3172239e9242b0975819c44dfb182cf6863cf4b8ed82`.
For three fixed candidates its physical forecast scores were `-0.0158723`,
`-0.0145812`, and `-0.00805243`, producing bounded corrections `0`,
`0.000103292`, and `0.000625593`. All four branches per candidate were valid;
mean E/G/F residual scales were approximately `0.00263`, `0.00339`, and
`0.00172`.

Repeated queries were identical, imagination left the actual native snapshot
unchanged, and a JSON snapshot/restore reproduced both the pending report and
the next observation/query including RNG-generated plans. The export had 12
training updates and a pinned real anonymous episode-000 dataset/input
identity. It predates the explicit `forecast_status` metadata field, so the
report honestly labels its status unknown and uncalibrated. Native-owner parity
for the same export measured maximum absolute differences of `5.96e-7` for
latent state, `1.49e-8` for physical physiology mean, and `7.45e-9` for scale,
with exact snapshot continuation. These are execution and numerical receipts,
not evidence that foresight improves behavior or that residual scales are
calibrated.
