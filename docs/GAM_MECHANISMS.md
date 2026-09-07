# Experienced consequence laws

Chreatures uses `SauersML/gam` to compress experienced nonlinear body responses
into immutable consequence-law banks. These are conditional predictive models:
they do not establish causal effects, biological mechanisms, competence, or a
resident's welfare. They never alter physical chemistry or physiology.

## Runtime contract

The native evaluator in
[`gam_law.rs`](../native/cognitive-core/src/gam_law.rs) is connected to the
[`developmental resident`](../native/cognitive-core/src/developmental.rs).
The learned policy proposes four actions in the same current body/neural/goal
context. Each proposal receives a prediction of its one-tick body consequences.
World coordinates, names, object kinds, resident IDs, and
evidence records are excluded. Audit-only episode/world/resident keys are used to
split fitting data and are absent from the artifact's feature vector.

A law bank declares, for every input, its name, unit, normalization, and fitted
domain. It declares each response unit, its held-out residual RMSE, and sampled
smooth terms. Runtime returns expected consequences and held-out residual errors;
these errors are empirical scales, not calibrated uncertainty intervals.
Any feature outside the experienced domain makes the candidate out of domain and
suppresses its contribution to the selection adjustment. Such a candidate retains
the original policy's support. The bank contains no happiness or privileged reward
score.

The current selector compares predicted energy, fatigue and encoded-speed changes
with the body component of a previously achieved sensory goal. The difference is
divided by the goal's remaining commitment ticks, rather than by the age of the
memory. Standardized errors are bounded, and centered candidate scores alter
resampling weights through a bounded `0.5 * tanh(score - mean_score)` term. This is
an engineered way to use a remembered body state; it does not prove that the whole
sensory goal is controllable or that the old body state is reachable now.

Each resident also owns a small normalized least-mean-squares residual model in
[`personal_consequences.rs`](../native/cognitive-core/src/personal_consequences.rs).
It learns only after a committed physical transition, using the exact executed
action and pre-action physiology. The native boundary checks the pending tick,
action, oral command and physiology before consuming a learning update. Unexecuted
candidates supply no targets. Out-of-domain transitions are counted separately and
do not update the private model. Learning rates, bounded feature coordinates,
innovation limits and per-target correction bounds are part of the artifact.
Private coefficients, update counts, residual summaries, RNG and pending transition
state survive a whole-resident checkpoint.

The rich Torch development policy is trained without this four-candidate
refinement. The current native-v8 export embeds the preceding rich-play consequence bank
with the inherited controller weights, creating a declared deployment variant
rather than an identical copy of the training policy. The earlier v3 export
receipt already pinned the same rich bank SHA alongside the update-160 inherited
controller, while describing that artifact as a future birth. Export and fit
receipts establish identity and wiring; they do not establish that the new
predictor/world launch has occurred or that behavior improved.

`LawBank::fitted_features` is the single v1 extraction contract: physiology indices
0/2/3/5, mean of the 384 permitted neural rates, candidate thrust/yaw/grip/oral,
mean absolute magnitude of the first four motor commands, and the two declared
interaction coordinates. `normalize_private` maps each fitted domain to `[-1,1]`
for resident-private residual learning and separately returns the domain flag.

The current export samples each additive smooth at 97 knots and uses clamped linear
interpolation. Clamping is only an arithmetic safeguard: the out-of-domain flag still
prevents the selection adjustment. The whole-fit report measures exported-grid predictions against
direct native GAM predictions.

## Two runtime law layers

The embedded body consequence bank is
`artifacts/rich_body_laws_v2/body_consequence_laws.json` (SHA-256
`1abb586294db1cbc8db7ba20788eec01d54ef88cf9a642b7e15c4f909223d87b`).
It uses the 12-feature `LawBank::fitted_features` contract and supplies movement,
energy-cost, and fatigue-recovery estimates to the remembered-body-goal
refinement and each resident's private residual learner.

The population response bank is a separate optional layer. The current research
candidate is
`artifacts/population_complete_b2_history_v2/fit/population_response_bank.json`
(SHA-256
`4d535c586c899609a51c14792b6dda6a50c66dc84e2b763c1a99a74842947b92`).
It uses physiology12, private 64-tick history summaries, and hypothetical
action12 to predict energy-state delta, fatigue-state delta, and effort. Its
bounded, centered score contribution is evaluated alongside the body-law score.
It is loaded only when an explicit new-birth artifact is supplied; embedding the
body bank does not implicitly enable this optional layer. Neither bank changes
native body chemistry.

## Current rich-body fit

`integrations/gam_mechanisms/fit_consequence_laws.py` consumes actual 20 Hz
sensorimotor transition packets. The rich-body bank uses 196,608 completed
rich-play transitions across two episodes, four world slots, and six residents
per world. Complete world slot 3 is held out in both episodes: 147,456 rows from
six episode/world units train the fits and 49,152 rows from two complete units
are held out. Prediction rows and independent episode/world units remain distinct.

| One-step target | Held-out GAM RMSE | Training-mean baseline RMSE |
| --- | ---: | ---: |
| Change in encoded speed (`tanh(speed / 2)`) | 0.02452 | 0.02864 |
| Energy cost | 0.00003138 | 0.00005350 |
| Fatigue recovery | 0.0001349 | 0.0002552 |

42,846 of 49,152 held-out rows (87.17%) lie inside all fitted feature domains.
Within those rows, the largest sampled-grid errors relative to direct GAM
predictions are `5.826e-5`, `1.173e-6`, and `2.114e-6`, respectively. The
[fit report](../integrations/gam_mechanisms/artifacts/rich_body_laws_v2/fit_report.json) records the
source, split, domains, bounds and exact artifact identities.

The state/action features are body-local energy, fatigue, speed, circuit support,
mean MaleCNS readout activity, executed thrust/yaw/grip/oral commands, mean motor
magnitude, and explicit thrust-by-fatigue and yaw-by-speed interaction coordinates.
The response laws predict one-tick movement response, energy cost, and fatigue
recovery. The interaction coordinates let smooth terms express context-dependent
effects without giving the controller identity or world geometry.

This completed rich-play collection is the authoritative source for the embedded
bank. Later development and population telemetry support separate atlas and
population-response fits; they do not silently replace this 12-feature body bank.

## Retained historical base fit

The earlier `artifacts/body_consequence_laws.json` bank (SHA-256
`41e222efdbee7f9b018f10764cd51ca97298005ff132bcffa1f96eab0cddbba6`) is
retained as historical evidence. It used 192,000 transitions, with 144,000
training and 48,000 held-out rows, and reported movement, energy, and fatigue
RMSEs of `0.01849`, `0.0000191`, and `0.0000698`. It is not the bank embedded in
the current native-v8 export.

## Queued current rich-v4 refit

The next body-law fit consumes only the completed
`chreatures-sensorimotor-play-rich-v4` campaign with 655,360 delivered physical
transitions. The compact extractor refuses partial progress manifests, dirty
collector source, non-contiguous 512-tick shards, altered receipts, or any split
other than world slots 0–7 for fitting, slot 8 for validation calibration, and
slot 9 for final reporting. It also requires the current native-v8 resident
identity and its authenticated 80-member founding bank. All eight residents
sharing an episode/world remain in the same partition.

Rich-v4 uses the current twelve-action body boundary. The old shortcut over
action columns `0:4` is invalid because column 3 is posture. The current native
LawBank seam maps grip from delivered column 4, oral/eat from column 8, and
computes motor magnitude from columns 0, 1, 2, and 4. The target seam remains
one-tick encoded-speed change, energy cost, and fatigue recovery from physiology
columns 3, 0, and 2. This refresh changes fitted physical associations; it adds
no score, personality, reward, or privileged world feature.

After the authoritative manifest exists, extract the compact bulk intermediate
on persvati beside the relayed collection:

```sh
/home/ember/kaxsim/.venv7/bin/python integrations/gam_mechanisms/prepare_rich_consequence_laws.py \
  --collection /home/ember/chreatures-data/sensorimotor-play/rich-v4-seed20260919 \
  --source-revision aa0093ab4f2c5e26f135bc9d79df448c8ae249a1 \
  --output /home/ember/chreatures-data/sensorimotor-play/rich-v4-seed20260919/gam-body-transitions-v1.npz
```

Transfer only that compact intermediate to the M2 source-frozen integration
checkout whose `integrations/.venv` contains native `gamfit==0.1.259`. Build the
current Rust verifier outside the laptop workspace, then fit into a fresh output
directory:

```sh
CARGO_TARGET_DIR=/Users/ember/paperbin/chreatures/build/gam-mechanisms \
  cargo build --release --manifest-path integrations/gam_mechanisms/Cargo.toml
integrations/.venv/bin/python integrations/gam_mechanisms/fit_consequence_laws.py \
  --compact /path/to/gam-body-transitions-v1.npz \
  --output-dir /path/to/rich-v4-body-laws-v1 \
  --native-evaluator /Users/ember/paperbin/chreatures/build/gam-mechanisms/release/chreatures-gam-mechanisms
```

The fit uses `gamfit.fit_array`, which stays on the package's native Rust path
without materializing 524,288 Python row dictionaries. It saves each certified
native `.gam` model. A failed model or native verification leaves a failure
receipt and any completed model files, mints no `body_consequence_laws.json`, and
does not substitute the preceding bank. A successful run emits the reusable
LawBank, validation and untouched-holdout domain metrics, per-world cohort
checks, native boundary cases, a native evaluation receipt, and an outside-only
analyst receipt. Root still owns artifact promotion and source freeze.

## Historical source and reproduction

The fitting API is the native `gamfit==0.1.259` wheel, built from the official
SauersML/gam commit `7c7eca8ac4826de95c8e743a20294bee132a9bcc` (the upstream
release commit dated 2026-07-23). The formula uses REML-selected univariate smooths.
We intentionally do not constrain observed energy or fatigue curves toward desired
behavior: physical laws remain authoritative, and the fit must reveal what was
experienced.

Reproducing the preceding rich-body bank requires its archived rich-play packets, including
the separate oral-command column. The new twelve-action rich-v3 corpus has its
own declared contract; it is not interchangeable with those fitting inputs.
That historical invocation belongs to pinned Git history. The current extractor
intentionally accepts rich-v4 only. Compact transition matrices remain bulk
intermediates outside Git, while each law bank records the authoritative packet
hashes and collection identities.
