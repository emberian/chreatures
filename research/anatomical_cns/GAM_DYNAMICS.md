# Anatomical CNS V3 physiology response experiment

**Status: executed on the actual physical V3 corpus.** All 25 settings, a center zero-edge replay, real native `gamfit==0.1.259` fitting, and one sealed additional fullgraph confirmation completed. The confirmation exceeded the prespecified prediction-error threshold; no CNS parameters were promoted. Executed results and limits are below.

`gam_dynamics.py` replays the current `AnatomicalCNS` model, with all 165,122 cells and 25,563,197 anatomical edges. It does not implement another neural update equation or export a production model. It varies shared physiological controls around one frozen trained artifact. Cell-type heterogeneity, body/context interfaces, motor weights, and global readout remain those of that artifact. Full in-graph physiology learning remains the canonical Torch trainer's responsibility.

## Frozen experiment

The design has 25 settings: 16 corners, eight axial points, and the center of a four-dimensional box. Coordinates below are dimensionless and take −1, 0, +1. One later interior confirmation and one center zero-edge replay bring the complete batch to **27 fullgraph runs**. A run uses eight whole physical worlds, 512 chronological neural ticks per world, and three private resident states per world. Default execution batches all eight worlds together (24 private resident states). Batching does not join or reset their private histories.

| Coordinate | Intervention on the trained array | Effective bound retained |
|---|---|---|
| `release_tau` | add coordinate × log(2) to every release-tau logit | 0.05–2 s |
| `release_use` | add coordinate × log(2) to every release-use logit | 0.01–0.5 |
| `mod_gain` | multiply target-type mod-gain raw array by 1 + 0.5 × coordinate | signed coefficient 0.5 tanh(raw) |
| `mod_tau` | add coordinate × log(2) to all three family-tau logits | 0.1–5 s |

Every setting restores the trained arrays before applying its intervention. Logit shifts retain type ordering and hard bounds without clipping. The modulation gain intervention preserves signs; family-tau shifts retain family ordering. These are bounded sensitivity interventions, not estimates of measured fly physiology. Modulation's adaptation coefficients and all other V3 parameters stay fixed. Family-specific or type-specific identifiability cannot be established by four shared controls.

The strict `data.load_corpus` loader verifies the eight episode identities, hashes, physical provenance, array layouts, and whole-world split. Only actual optic RGB, BODY110, and delivered context12 enter the neural model. Each episode starts from the model's exact baseline and receives its t=0 observation once; there is no hidden warmup. The recorded context at t enters the step that returns MOTOR34 at t. Teacher motor, skill, pose, and other teacher labels never enter the CNS. The initial eight ticks are retained in neural history and excluded from scored targets. The final observation at t=512 is a future target, not a fabricated final context step.

## Outcomes and selection

The direct frozen MOTOR34 output is scored against the physically executed teacher motor targets. Error is the mean squared error after scaling the two signed channels by their full range of two; the remaining channels have range one. Reports include per-channel and per-world errors and a training-world mean-motor baseline. No motor decoder is refitted in this experiment.

A fixed ridge probe with penalty 0.01 predicts **four-tick sensory change from current Z512 alone**. Targets are 24 RGB means over eight contiguous atlas-site groups plus BODY110, normalized using training-only current-target standard deviations with floor 0.01. The atlas groups are not claimed to be uniform angular retinal sectors. Probe features are standardized on training worlds with a 1e-6 floor. Its intercept is unpenalized. The probe never sees raw current senses, future context, or executed motor commands. Scores compare against sensory persistence and training-mean change; these measure the scope of this linear decoder under an unconditioned future, and cannot prove absence of information or action-conditioned predictive ability.

Probe targets cover t=8..508 inclusive, with lookahead through t=512. Motor targets cover t=8..511. Constant target dimensions remain in the group MSE; active dimension counts are reported. Optic and body groups receive equal weight through their separate persistence-normalized errors.

Worlds 0–3 fit the inner temporal probe and its statistics; worlds 4–5 supply the **parameter-selection response**. Worlds 0–5 fit the final reporting probe; worlds 6–7 are the untouched heldout-world evaluation. These six training worlds may already have participated in fitting the frozen CNS: the inner split validates the probe and parameter choice, not generalization of the original CNS training. No heldout metric selects a setting, a probe hyperparameter, or a checkpoint. The predeclared scalar response is:

```
inner motor range-normalized MSE
+ 0.25 × mean(optic future MSE / optic persistence MSE,
              body future MSE / body persistence MSE)
```

The 0.25 weight is a fixed research tradeoff, not evidence that one unit of prediction error is interchangeable with physical skill. The components remain separately visible. Predicted antagonist squared activation and paired co-contraction are diagnostics, reported alongside teacher effort. Low activity or low effort alone cannot win the objective. No visual activity or variance objective is used.

## Native GAM and additional fullgraph observation

The analyzer reuses native loading/version checks, stderr capture, and error metrics from `research/dynamics_v2/gam_fit.py`; it calls the real native Rust GAM extension. It fits the joint response with:

```
response ~ te(release_tau,release_use,mod_gain,mod_tau,k=15)
```

The inner objective gets leave-one-entire-setting-out cross-validation, compared with an additive four-smooth GAM and the remaining-settings mean. Separate joint native fits are attempted for motor error, temporal error ratio, and effort where their responses are identifiable. Reports retain nine-point conditional axis slices and all six paired conditional response grids. The paired nonadditive effect is f(a,b) − f(a,0) − f(0,b) + f(0,0), with other controls fixed at their trained center; these are predictions from the fitted native GAM, not additional measured settings. They expose candidate release/modulation interactions, with usefulness constrained by the leave-setting-out evidence. Exactly constant responses and the native solver’s effectively-constant response rejection are reported as unidentifiable; no rescaling, jitter, or surrogate regression creates a spurious fitted result. Saved native models must reproduce their predictions after reload. The four-dimensional model has only 25 observations: it is a regularized local calibration surface, not broad biological identification. Cross-validation failure remains a failure in the report.

The fitted inner-objective surface chooses one unmeasured point from a fixed 4⁴ grid at coordinates −0.75, −0.25, +0.25, +0.75. Its prediction is sealed before any additional replay. The same artifact, source files, corpus, and replay configuration then evaluate that point with the actual fullgraph. The report gives absolute prediction error and whether it is below the leave-setting-out RMSE. A replay status of `actual-fullgraph-confirmed` means the observation was executed; the accuracy flag can be false. If the objective is constant, no meaningful GAM choice is available and no confirmation is proposed.

The final report also requires the center with **all fast and modulatory edge values zero**, retaining the same nontrivial baseline and afferent/body/context/motor interfaces. Its temporal probe is refitted on its own training Z, giving that representation the same decoder capacity. This tests graph dependence of this replay/probe task; it is not an edge-randomization experiment or a claim about biological lesions. Z and motor variability in this ablation are diagnostics for a residual route, not optimized targets.

## Launch and receipts

Run from the repository root in an immutable source copy. Keep the corpus and result bulk on `/tank`. The commands below are reusable path templates. The completed wave used the exact source, paths, identities, and configuration listed in the execution record below.

```sh
python -m research.anatomical_cns.gam_dynamics design --output RUN/design.json
HSA_OVERRIDE_GFX_VERSION=10.3.0 /tank/chreatures/envs/rocm-dev/bin/python -m research.anatomical_cns.gam_dynamics run --service TRAINED_CHCNS3.bin --corpus CORPUS --output RUN/results
HSA_OVERRIDE_GFX_VERSION=10.3.0 /tank/chreatures/envs/rocm-dev/bin/python -m research.anatomical_cns.gam_dynamics run --service TRAINED_CHCNS3.bin --corpus CORPUS --output RUN/results --zero-edge
```

`--start` and `--stop` permit bounded index ranges within the same frozen 25-setting design. A completed setting is reused only when its full provenance matches; otherwise the script refuses it. Each setting is written only after replay and evaluation finish. GPU allocation is capped at 70% of device capacity, with default B=24; this is an allocation limit, not a measured peak or assurance against contention. Preserve existing services and let root schedule the lane. The first actual setting's wall time should inform whether the remainder needs separate scheduled batches; there is no automatic background launch.

Native analysis can run locally after copying the compact setting JSON files, using the already isolated GAM environment:

```sh
integrations/.venv/bin/python -m research.anatomical_cns.gam_dynamics fit --results RUN/results --output RUN/gam
```

Copy `confirmation-proposal.json` back to the same frozen hbox source/run and execute once:

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 /tank/chreatures/envs/rocm-dev/bin/python -m research.anatomical_cns.gam_dynamics run --service TRAINED_CHCNS3.bin --corpus CORPUS --output RUN/results --confirmation RUN/gam/confirmation-proposal.json
integrations/.venv/bin/python -m research.anatomical_cns.gam_dynamics confirm --results RUN/results --fit RUN/gam
```

Each result records service/adapter and corpus identities, every episode hash, replay source hashes, Torch/HIP, batch configuration, fixed probe settings, exact intervention, elapsed wall time, and split-specific metrics. The fit adds native GAM version/source/build, model files, warnings, cross-validation, observed heldout outcomes, and a sealed confirmation proposal. It never replaces a current CNS or resident binding. The conclusions apply to open-loop replay of recorded physical sequences; closed-loop motor competence, interactive learning, and deployment require the integrated wave's own evaluation.


## Executed physical wave, 2026-09-07

The frozen replay lives at `/tank/chreatures/runs/malecns-v3/gam/physical-ede2b03d/`; compact records, source snapshots, native models, and confirmation are mirrored at `/Users/ember/paperbin/chreatures/integration/anatomical-cns-gam-physical-ede2b03d/`. Raw physical arrays and trained weights remain on `/tank`. Repository evidence is sealed in [gam.json](../../docs/receipts/anatomical-cns-v3/gam.json) and [gam-confirmation.json](../../docs/receipts/anatomical-cns-v3/gam-confirmation.json).

- Trained CHCNS3 service SHA: `6e1acf46d18f5db0ff6e0096ed7439ebaf5df722ce34c1f0c56276ba271fccc3`; adapter `53765d4afb16ee084d9d0866066ac1ddc4f879a54af0d4693d97230509688996`.
- Physical corpus SHA: `e020317ae5f94d22c39af191d45b1934419d927d5e9d54492ed72a460d5f7367`, six training and two heldout worlds, 512 chronological CNS steps per world, three residents each. Delivered bootstrap context is zero throughout; this wave cannot identify context-conditioned physiology.
- Frozen replay source SHA: `ede2b03d4734c81219a2dbc6c84fca1158d85763e1b284ac8d5d8bf6fbd94a02`; exact model SHA `55fbdafd8fecf0ef5072cb29729d8e7278ab8241e84939a9c9d6f2e4291b76d5` matches the trainer. The later analysis-source snapshot is separately hashed by the native fit report.
- Two completed B6 pilot settings were preserved after the single approved restart. The repeated setting-00 metric record was **exactly equal** at B6 and B24. The coherent 27-run campaign uses B24 throughout; pilot records do not enter the GAM. The 25 settings took 1,558.56 seconds; zero-edge 61.69 seconds; confirmation 63.58 seconds. Observed total GPU memory was 2.855 GB, with the protected-service baseline restored to 1.06 GB after completion; this is sampled telemetry, not a claimed exhaustive peak.

The real native joint GAM predicted left-out settings better than both benchmarks:

| Inner-objective leave-setting-out model | RMSE | R² |
|---|---:|---:|
| Four-parameter tensor GAM | 0.000015462 | 0.8861 |
| Additive GAM | 0.000023064 | 0.7466 |
| Remaining-settings mean | 0.000047731 | −0.0851 |

The motor-error target was rejected by the native solver as effectively constant: sample standard deviation approximately `1.379e-11`, below its `1e-10` threshold. The first analysis attempt and native objective model are preserved under `gam-first-attempt-native-constant/`. The completed analysis records that rejection and continues other targets without rescaling or fabricated variation. Antagonist effort was also constant at the declared resolution. Objective and temporal-ratio surfaces were genuinely fitted by the native extension.

The measured inner-objective corner contrasts support a release recovery/use interaction: increasing use changes the objective by about `+1.03e-5` at fast recovery and `−8.22e-5` at slow recovery. This is compatible with the release equation coupling recovery time and use in resource depletion; it is not an inferred biological parameter estimate. Modulation-gain contrasts are only `1.08e-7` to `4.79e-7` in magnitude, and family-time contrasts only `1.22e-8` to `1.81e-7`. The sparse 25-setting design does **not** establish the larger interior release/modulation interactions predicted by the four-dimensional surface. Conditional grids are retained so this interpolation weakness is visible.

The sealed interior proposal was `(release_tau, release_use, mod_gain, mod_tau) = (0.75, −0.75, −0.25, −0.25)`. It predicted inner objective `0.26807132195`; the actual fullgraph replay measured `0.26809148957`. Absolute error `0.00002016762` exceeded joint leave-setting-out RMSE `0.00001546191`, so `within_loo_rmse` is **false**. The final status confirms execution, not predictive success.

| Whole-heldout result | Trained center | GAM-proposed setting | Center with zero edges |
|---|---:|---:|---:|
| Motor range-normalized MSE | 0.03073530 | 0.03073530 | approximately 0.03073530 |
| Four-tick optic skill vs persistence | 0.086274 | 0.086319 | 0.001485 |
| Four-tick body skill vs persistence | 0.010385 | 0.010370 | 0.000077 |
| Combined objective | 0.26865293 | 0.26864920 | approximately 0.28054 |

The proposed setting's heldout objective improvement is only `0.00000372819`, with motor error unchanged. Across all 25 settings, heldout motor MSE spans just `1.08e-10`; it is worse than the training-world mean-motor baseline. The measured physiology box therefore does not solve the nearly constant MOTOR34 output. No new CNS artifact, resident lineage, or public model was created.

The zero-edge temporal probes reduce to training-mean change prediction to roundoff, supporting graph dependence of the modest temporal decoding result. A diagnostic caveat is preserved explicitly: the archived replay computed output standard deviations using float32 accumulation, which reports small nonzero values even for repeated constants (`Z` approximately `1.6e-6`, motor approximately `1.3e-5`). These are not evidence of an unmasked sensory route or learned variability. The current launcher uses float64 for these diagnostics; scored probe and motor errors already used float64 and are unchanged. No extra GPU replay was used to replace the archived numbers.

These are unconditioned linear sensory predictions on a shared physical curriculum. They cannot distinguish all stimulus-specific information from neural history correlated with the shared curriculum timing, and they do not test context learning, nonlinear decoding, or closed-loop competence. The decisive result is limited but useful: the graph carries some decodable temporal information, the native GAM captures part of release-parameter sensitivity, and neither its weak motor response nor its failed additional-point criterion justifies physiological parameter promotion.
