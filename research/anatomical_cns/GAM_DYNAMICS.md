# Anatomical CNS V3 physiology response experiment

**Status: implemented launcher and analysis; actual V3 sweep, native GAM fit, and confirmation have not run.** Root schedules the GPU after a trained V3 artifact and the physical BODY110 corpus exist. Formula validation used the installed native `gamfit==0.1.259`; that validation is not a fitted result.

`gam_dynamics.py` replays the current `AnatomicalCNS` model, with all 165,122 cells and 25,563,197 anatomical edges. It does not implement another neural update equation or export a production model. It varies shared physiological controls around one frozen trained artifact. Cell-type heterogeneity, body/context interfaces, motor weights, and global readout remain those of that artifact. Full in-graph physiology learning remains the canonical Torch trainer's responsibility.

## Frozen experiment

The design has 25 settings: 16 corners, eight axial points, and the center of a four-dimensional box. Coordinates below are dimensionless and take −1, 0, +1. One later interior confirmation and one center zero-edge replay bring the complete batch to **27 fullgraph runs**. A run uses eight whole physical worlds, 512 chronological neural ticks per world, and three private resident states per world. Default execution batches two worlds at a time. Batching does not join or reset their private histories.

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

Worlds 0–3 fit the inner temporal probe and its statistics; worlds 4–5 supply the **parameter-selection response**. Worlds 0–5 fit the final reporting probe; worlds 6–7 are the untouched heldout-world evaluation. No heldout metric selects a setting, a probe hyperparameter, or a checkpoint. The predeclared scalar response is:

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

The inner objective gets leave-one-entire-setting-out cross-validation, compared with an additive four-smooth GAM and the remaining-settings mean. Separate joint native surfaces describe motor error, temporal error ratio, and effort. Exactly constant responses are reported as unidentifiable; no jitter or surrogate regression creates a spurious fitted result. Saved native models must reproduce their predictions after reload. The four-dimensional model has only 25 observations: it is a regularized local calibration surface, not broad biological identification. Cross-validation failure remains a failure in the report.

The fitted inner-objective surface chooses one unmeasured point from a fixed 4⁴ grid at coordinates −0.75, −0.25, +0.25, +0.75. Its prediction is sealed before any additional replay. The same artifact, source files, corpus, and replay configuration then evaluate that point with the actual fullgraph. The report gives absolute prediction error and whether it is below the leave-setting-out RMSE. A replay status of `actual-fullgraph-confirmed` means the observation was executed; the accuracy flag can be false. If the objective is constant, no meaningful GAM choice is available and no confirmation is proposed.

The final report also requires the center with **all fast and modulatory edge values zero**, retaining the same nontrivial baseline and afferent/body/context/motor interfaces. Its temporal probe is refitted on its own training Z, giving that representation the same decoder capacity. This tests graph dependence of this replay/probe task; it is not an edge-randomization experiment or a claim about biological lesions. Z and motor variability in this ablation are diagnostics for a residual route, not optimized targets.

## Launch and receipts

Run from the repository root in an immutable source copy. Keep the corpus and result bulk on `/tank`. The commands below are templates with explicit paths to be supplied after root schedules the lane. They have not been executed on V3 data.

```sh
python -m research.anatomical_cns.gam_dynamics design --output RUN/design.json
HSA_OVERRIDE_GFX_VERSION=10.3.0 /tank/chreatures/envs/rocm-dev/bin/python -m research.anatomical_cns.gam_dynamics run --service TRAINED_CHCNS3.bin --corpus CORPUS --output RUN/results
HSA_OVERRIDE_GFX_VERSION=10.3.0 /tank/chreatures/envs/rocm-dev/bin/python -m research.anatomical_cns.gam_dynamics run --service TRAINED_CHCNS3.bin --corpus CORPUS --output RUN/results --zero-edge
```

`--start` and `--stop` permit bounded index ranges within the same frozen 25-setting design. A completed setting is reused only when its full provenance matches; otherwise the script refuses it. Each setting is written only after replay and evaluation finish. GPU allocation is capped at 70% of device capacity, with default B=6; this is an allocation limit, not a measured peak or assurance against contention. Preserve existing services and let root schedule the lane. The first actual setting's wall time should inform whether the remainder needs separate scheduled batches; there is no automatic background launch.

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
