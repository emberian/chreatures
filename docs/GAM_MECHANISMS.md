# Experienced consequence laws

Chreatures uses `SauersML/gam` to compress experienced nonlinear body responses
into immutable consequence-law banks. These are conditional predictive models:
they do not establish causal effects, biological mechanisms, competence, or a
resident's welfare. They never alter physical chemistry or physiology.

## Runtime contract

The callable native evaluator is
[`native/cognitive-core/src/gam_law.rs`](../native/cognitive-core/src/gam_law.rs).
actions. It is not yet wired into the resident selector. Each candidate combines
the same current body/neural/goal context with a
hypothetical action. World coordinates, names, object kinds, resident IDs, and
evidence records are excluded. Audit-only episode/world/resident keys are used to
split fitting data and are absent from the artifact's feature vector.

A law bank declares, for every input, its name, unit, normalization, and fitted
domain. It declares each response unit, its held-out residual RMSE, and sampled
smooth terms. Runtime returns expected consequences and that residual uncertainty.
Any feature outside the experienced domain makes the candidate out of domain and
suppresses its scalar score. The caller supplies bounded private preference weights;
the bank contains no happiness or privileged reward score.

`LawBank::fitted_features` is the single v1 extraction contract: physiology indices
0/2/3/5, mean of the 384 permitted neural rates, candidate thrust/yaw/grip/oral,
mean absolute magnitude of the first four motor commands, and the two declared
interaction coordinates. `normalize_private` maps each fitted domain to `[-1,1]`
for resident-private residual learning and separately returns the domain flag.

The current export samples each additive smooth at 97 knots and uses clamped linear
interpolation. Clamping is only an arithmetic safeguard: the out-of-domain flag still
prevents promotion. The whole-fit report measures exported-grid predictions against
direct native GAM predictions.

## Executed fit

`integrations/gam_mechanisms/fit_consequence_laws.py` consumes the actual 20 Hz
sensorimotor transition packets produced in fresh chemical worlds. Its current
source has 192,000 transitions from two episodes and 32 independently seeded
physical worlds. Twenty-four complete worlds (144,000 rows) train the fits; eight
complete worlds (48,000 rows) are held out. Prediction counts and independent world
counts are reported separately.

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
