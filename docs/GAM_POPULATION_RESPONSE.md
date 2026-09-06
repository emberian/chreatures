# GAM population response mechanisms

The population response path is an inherited, immutable mechanism bank. A candidate genome records the SHA-256 of one bank and its feature contract. Rust authenticates both before evaluating it. A refit creates an artifact for later births; it does not alter a living resident.

`native/cognitive-core/src/population_response.rs` evaluates fitted `SauersML/gam` additive laws through the existing native `LawBank`. It rejects the complete response when any input is outside the fitted local domain. Positive responses pass through capped softplus, signed responses through bounded tanh, and allocation groups through a joint softmax whose outputs sum to the declared budget. The developmental controller has a next-birth optional seam: an authenticated embedded bank may declare predictive mechanisms, signed weights, training-only normalization scales, and a maximum candidate tilt no larger than 0.5. Candidate predictions are centered and bounded separately from the deployed three-law score. They never become actuator gains or somatic-law changes. With no bank, this path performs no evaluation and the existing controller path is unchanged.

Python exposes the seam as `DevelopmentalResidentCohort(..., population_response_artifact=PATH)`. Omission is the current path. The wrapper hashes the exact file bytes and native Rust authenticates that hash and the embedded feature-contract identity. Snapshots store the external absolute path and both identities; restore requires the caller to supply the same explicit artifact again. Expansion retains the immutable reference and existing private history, while newly appended or hatched residents start with cold history.

The frozen v1 input order is in `integrations/gam_mechanisms/population_feature_contract_v1.json`: physiology12, four private history summaries, and executed actions12. Lineage, candidate, episode, world, and environment keys exist only for audit and splitting. Identities, object labels, and world coordinates are excluded from inference.

`fit_population_response.py` consumes complete-unit NPZ telemetry, a response schema, and the separately hashed feature contract. It requires lineage, environment, candidate, episode, and world unit arrays. Training excludes declared held-out lineages and environments; validation uses whole remaining worlds. It reports held-out lineage, held-out environment, their union, and validation separately, including error after aggregating every complete candidate/episode/world unit. Every response uses native `gamfit==0.1.259` from pinned `SauersML/gam` commit `7c7eca8ac4826de95c8e743a20294bee132a9bcc`. A native fit failure is preserved in `fit_report.json`, and no partial response bank is minted.

The first measured physiology12 artifact is
`integrations/gam_mechanisms/artifacts/population_first_prefix_v1/supported_fit/population_response_bank.json`
(SHA-256 `e787c7936558319b8f07a9f54e308d995fa2ed27007a414e2831e4c2b3b7a426`).
It contains only the two targets whose explicitly reduced native GAM fits certified:
one-step energy-state delta and effort. Rust's authenticated
`PopulationResponseBank` loader accepts the exact artifact and feature-contract
identity. It is a candidate for explicitly opted-in new research births; no
existing or campaign resident was changed.

The source is 315,392 committed transitions (32 lives, 9,856 ticks) from the
complete atomic trace prefix of B0. B0 later ended in one cohort-engine ATP
invariant failure, so these are censored transition observations rather than
completed-life outcomes or archive fitness. All candidates share one founder,
which prevents lineage-generalization claims. The split was frozen before
target inspection: 17,248 target-blind temporally subsampled training rows,
68,992 rows from seven whole candidate/episode/world validation units, and
108,416 rows from eleven whole held-out candidate or environment units. On the
held-out union, energy-delta RMSE is `2.507e-5` versus a training-mean baseline
RMSE of `3.468e-5`; effort RMSE is `0.03269` versus `0.13016`. Aggregate RMSE
over complete candidate/episode/world units is `6.371e-6` and `0.003539`,
respectively. These are predictive associations under executed actions, not
causal effects or evidence of evolutionary improvement.

The original eight-response attempt and an all-eight target-blind retry are
preserved as failed reports. Six laws were not minted after native solver
certification failures. The supported fit uses six declared smooth features
per target while preserving finite validation for all 28 inputs and fitted-range
checks for the smooth features each law actually uses. The
25x25 native-model surfaces and the Rust grid-export approximation are in
`surfaces-25x25.json`; the largest absolute transformed prediction difference
on those complete in-domain grids is `3.420e-4`. This approximation check is
separate from held-out prediction error.

Runtime requires all 28 values to be finite, while range checks apply only to
the smooth-term features actually used by each law. The two-law union uses ten
features and accepts 95.82% of all observed rows and 96.75% of the final
candidate/environment holdout. Checking all 28 fitted ranges would accept
77.87%, but that is only a diagnostic and is not the Rust rule. The runtime
residual RMSE and conservative bound are calibrated solely from validation
worlds (`0.06402`/`0.30136` latent units for energy and
`0.19363`/`0.89146` for effort); final holdouts affect reporting only.

The current laws use no history smooths. Their frozen B0 row artifact used
pre-state samples through `pre[t-1]` for its unused history columns, which is
one step behind native history. The current v2 row recipe instead summarizes
committed post-states through `post[t-1]`; because `post[t-1] == pre[t]` on a
contiguous life, this matches the state available at proposal `t`. For example,
tick 0 is cold and tick 1 contains only committed `post[0]`. This correction
does not change the fitted two-law bank or its hashes because neither law has a
history term. Resets begin cold, and future history-dependent fits must use the
v2 receipt timing.

The native model files remain in bulk storage at
`/tank/chreatures/runs/analysis/population-gam-first-prefix-v1/supported_fit/`:
energy SHA-256 `9830cf73ee47ff95ca9843f1ae247573d6579eb41949b4bb5342db1a2d422f63`
and effort SHA-256
`3da83b7fc8b7e99ee763ddb0623eba3f45d6522dae8d6f7aeb54c8c72bbc6655`.
The compact repository artifacts are sufficient for Rust inference and public
inspection; the bulk models preserve exact `gamfit` reconstruction.

The population evaluator can now record the required row-level evidence with `--gam-trace-chunk-steps 128`. Trace chunks flush before every coherent checkpoint and contain pre/post physiology12, executed action12, outcome8, and measured organ-flow3 arrays. After a completed evaluation, the staged commands are:

```text
python integrations/gam_mechanisms/prepare_population_response_rows.py --evaluation RUN --output RUN/population-gam-rows.npz
python integrations/gam_mechanisms/make_population_response_schema.py --rows RUN/population-gam-rows.npz --feature-contract integrations/gam_mechanisms/population_feature_contract_v1.json --heldout-candidate CANDIDATE_SHA --heldout-environment ENVIRONMENT_SHA --output RUN/population-gam-fit-schema.json
python integrations/gam_mechanisms/fit_population_response.py --data RUN/population-gam-rows.npz --schema RUN/population-gam-fit-schema.json --feature-contract integrations/gam_mechanisms/population_feature_contract_v1.json --output RUN/population-gam-fit
```

The row preparer rejects incomplete or overlapping traces and derives the four
64-tick histories from committed post-states strictly before each proposal;
the first tick has a cold zero history. Native `PopulationHistory` advances
only after a matching physical consequence commits. The schema maker labels
outputs as observed one-step associations under executed actions. It does not
call them causal effects.

The first-generation wave has one shared founder parent, so it cannot estimate lineage generalization. Its split was frozen before outcome inspection in `population_wave_first_fit_split.json`: three whole candidate lives and one whole environment are held out, while a remaining whole world supplies validation. Later generations should use `--heldout-lineage` once multiple real lineages exist. If one of the eight attempted native fits fails, the failed report remains; an explicit `--target` list can mint only a declared, successfully supported subset. Candidate scoring is defined only when both the energy-state-delta and effort laws are present.

The next population collection must record pre-state physiology12, the exact executed action12, bounded history summaries, post-state physiology12, and the somatic debits/transfers supplied by the body mechanism. Candidate, lineage, episode, world, and environment keys must cover each complete physical unit. Allocation targets should be measured funded transfers divided by a declared available budget; recovery targets should be measured post-minus-pre changes with time units; plasticity targets require an actual circuit-side update signal. Until those fields exist, the fitting recipe must produce no bank.
