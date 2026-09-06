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
refinement. Exporting it with the consequence bank creates a declared deployment
variant, rather than preserving an identical policy. The joined native mechanism
has executed on an authenticated synthetic integration fixture. A learned rich
resident deployment and its physical outcomes remain a separate step; neither the
fit nor that fixture establishes an improvement in behavior.

`LawBank::fitted_features` is the single v1 extraction contract: physiology indices
0/2/3/5, mean of the 384 permitted neural rates, candidate thrust/yaw/grip/oral,
mean absolute magnitude of the first four motor commands, and the two declared
interaction coordinates. `normalize_private` maps each fitted domain to `[-1,1]`
for resident-private residual learning and separately returns the domain flag.

The current export samples each additive smooth at 97 knots and uses clamped linear
interpolation. Clamping is only an arithmetic safeguard: the out-of-domain flag still
prevents the selection adjustment. The whole-fit report measures exported-grid predictions against
direct native GAM predictions.

## Executed fit

`integrations/gam_mechanisms/fit_consequence_laws.py` consumes the actual 20 Hz
sensorimotor transition packets produced in fresh chemical worlds. Its current
source has 192,000 transitions from two episodes and 32 independently seeded
physical worlds. Twenty-four complete worlds (144,000 rows) train the fits; eight
complete worlds (48,000 rows) are held out. Prediction counts and independent world
counts are reported separately.

| One-step target | Held-out GAM RMSE | Training-mean baseline RMSE |
| --- | ---: | ---: |
| Change in encoded speed (`tanh(speed / 2)`) | 0.01849 | 0.02346 |
| Energy cost | 0.0000191 | 0.0000634 |
| Fatigue recovery | 0.0000698 | 0.000165 |

44,184 of 48,000 held-out rows (92.05%) lie inside all fitted feature domains.
Within those rows, the largest sampled-grid errors relative to direct GAM
predictions are 0.00002503, 0.0000006421 and 0.0000009613 respectively. The
[fit report](../integrations/gam_mechanisms/artifacts/fit_report.json) records the
source, split, domains, bounds and exact artifact identities.

The state/action features are body-local energy, fatigue, speed, circuit support,
mean MaleCNS readout activity, executed thrust/yaw/grip/oral commands, mean motor
magnitude, and explicit thrust-by-fatigue and yaw-by-speed interaction coordinates.
The response laws predict one-tick movement response, energy cost, and fatigue
recovery. The interaction coordinates let smooth terms express context-dependent
effects without giving the controller identity or world geometry.

This collection supplied the achieved histories used to bootstrap development. The
subsequent online joined-development run retained per-update aggregates rather than
per-transition rows, so it is not represented as this fit's source. The next rich
joined run records pre-action recurrent state, goal distance and age, exact actions,
one-step physical outcomes, and contiguous keys for masked 4/10/20-tick targets.
Those rows will replace this bank through a one-way artifact promotion after
complete-world and complete-episode evaluation.

## Source and reproduction

The fitting API is the native `gamfit==0.1.259` wheel, built from the official
SauersML/gam commit `7c7eca8ac4826de95c8e743a20294bee132a9bcc` (the upstream
release commit dated 2026-07-23). The formula uses REML-selected univariate smooths.
We intentionally do not constrain observed energy or fatigue curves toward desired
behavior: physical laws remain authoritative, and the fit must reveal what was
experienced.

Extract compact columns on the bulk node, then fit on the local pinned environment:

```sh
/tank/chreatures/envs/rocm-dev/bin/python fit_consequence_laws.py \
  --episode episode-000.npz --episode episode-001.npz \
  --compact experienced_transitions.npz
integrations/.venv/bin/python integrations/gam_mechanisms/fit_consequence_laws.py \
  --compact experienced_transitions.npz \
  --output-dir integrations/gam_mechanisms/artifacts
cargo run --manifest-path integrations/gam_mechanisms/Cargo.toml -- \
  integrations/gam_mechanisms/artifacts/body_consequence_laws.json FEATURE...
```

The compact transition matrix is bulk intermediate data and stays outside Git. The
law bank records SHA-256 hashes of both authoritative source packets.
