# Private predictive foresight

> **Historical research record.** The Python implementation and operational
> commands described here were retired from the current tree after the native
> rich developmental controller replaced them. Reproduce this experiment from
> Git commit `0caa7ef`; the findings and design rationale below remain part of
> the research record.

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

Call `observe(raw_features, physiology, previous_action, reset=...)` exactly
once per actual motor macro boundary. The complete predictive artifact owns the
frozen source PPO count/mean/M2 transform and converts raw MaleCNS readouts to
the trained feature contract itself. Its checksum is part of the locked input
identity; preprocessing never depends on the currently selected motor genome.
Physiology is ordered energy, gut, fatigue, local speed,
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
foresight.observe(raw_features, physiology, previous_action, reset=first_macro)

decision = refiner.refine(
    memory, features, local_physiology, policy_mean, policy_log_std,
    candidate_evidence=foresight.candidate_evidence,
)
foresight.commit_executed(decision["action_vector"])
```

Fresh 3-D worlds opt in through the existing personal motor path:

```bash
python -m chreatures.server \
  --motor-genome data/genomes/nursery-20000-finite-energy.npz \
  --personal-memory \
  --predictive-model /path/to/complete-predictive-state.npz
```

The runtime observes raw anonymous MaleCNS output and six local physiology
channels only at an actual five-tick motor boundary. It supplies the previously
executed macro action, evaluates candidates, and commits the selected physical
action's intention tail. Held ticks do not advance the predictor. If visual
candidate evidence is also active, the chooser sums both frozen-state reports
before applying its existing ±0.12 external-evidence bound. New checkpoints
store the predictive artifact location and one complete private foresight
snapshot per resident. Checkpoints without these fields restore through the
old controller path and reserialize without foresight fields.

## Scoring and evidence limits

The native model returns absolute predicted physiology in physical units.
Foresight rejects a branch when any energy, gut, or fatigue forecast is
nonfinite or outside `[0,1]`; it does not clip an invalid trajectory into a
plausible one. Valid trajectories are scored as discounted consecutive changes
in `FiniteEnergyObjective.potential`, including reserve, fatigue, and gut
overload terms. Scores are averaged over valid branches and converted to a
correction relative to candidate zero, then clipped to ±0.12. This preserves
the inherited and relational scores as independent parts of final selection.
An all-invalid candidate receives zero correction. Its forecast score, best
branch, residual summary, and support summary are serialized as `null`, keeping
the audit JSON finite without manufacturing substitute physiology.

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
a candidate-sensitive stand-in. The final hbox check used the complete
four-episode physical-unit export, artifact SHA-256
`c7ba729b8a62b7fda5a436a69d3ee8f5036c1c63119da6549443a9ed49f82ca2`.
This artifact preserves the trained predictor and embeds the collector's source
PPO moments (count `412656.0001`, identity `cc9f3d1a…`) so preprocessing is
complete and independent of a deployed motor artifact. Embedded normalization
matched the original RunningMoments calculation bit exactly.
The v3 temporal contract also pins five 50 ms physical steps per 250 ms
predictor observation; construction rejects a missing or different interval.

For three fixed candidates its physical forecast scores were `-0.0127308`,
`-0.00978304`, and `-0.0187044`, producing bounded corrections `0`,
`0.000235823`, and `-0.000477887`. All four branches per candidate were valid.
A fixed inherited-score probe selected candidate zero with foresight disabled
and candidate one with foresight enabled. This is an actual chooser readout
change; it does not establish that the chosen action was better.

Repeated queries were identical, imagination left the actual native snapshot
unchanged, and a JSON snapshot/restore reproduced both the pending report and
the next observation/query including RNG-generated plans. The export pins its
real anonymous four-episode dataset/input identity and explicitly says its
residual scales are not epistemic or OOD calibrated. Native-owner episode-003
parity measured maximum absolute differences of `7.08e-7` for latent state,
`2.98e-8` for physical physiology mean, and `2.24e-8` for scale, with exact
snapshot continuation and different-model restore rejection. These are
execution and numerical receipts, not evidence that foresight improves
behavior. The frozen action-discrimination assay found clear effects in neural
features and angular motion, while energy, gut, and support sensitivity remained
weak; homeostatic planning claims must remain correspondingly modest.

The joined living-motor probe reconstructed a raw readout from held-out
episode-003 row `[1,0]`; the complete artifact recovered its recorded normalized
384-vector with maximum absolute error `0`. Two actual five-tick boundaries
produced two predictor observations and two motor decisions. The first actions
were different across those experienced boundaries, and restoring both private
organs from JSON reproduced the next held action and complete foresight state
exactly. This exercises the runtime macro lifecycle and persistence boundary on
recorded inputs without advancing a live world or presenting it as an embodied
performance result.
