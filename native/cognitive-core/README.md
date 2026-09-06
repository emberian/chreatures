# Chreatures cognitive core

This AGPL-3.0-or-later PyO3 extension runs the immutable recurrent predictive
organ without Torch. It uses `matrixmultiply` SGEMM and owns reusable cohort
scratch, private latent state, the previous executed action, and the latest
actually observed physiology anchor. Only returned NumPy arrays allocate at the
FFI boundary.

`observe(features, physiology, previous_action, reset)` performs one batched
posterior update. It normalizes observation inputs with the immutable training
normalizer and records the physical physiology anchor. `imagine(actions[T,B,A])`
copies current state, performs at most eight transition steps, and returns raw
physical feature/physiology means, physical residual scales, validity, and
horizon support. It does not advance experienced state, previous action, or the
anchor.

The Python adapter also exposes
`query_from_snapshot(snapshot, actions[T,Bq,8])`. It accepts
`1 <= Bq <= configured capacity`, tiles a B1 snapshot internally for candidate
queries, and returns only the active `Bq` rows. A separate query cohort can thus
evaluate planning branches without restoring or mutating the experienced
cohort.

Snapshots contain all three private buffers and pin the artifact SHA-256, tensor
manifest SHA-256, and input identity. A same-shaped model with different weights
rejects restoration. Exports additionally pin the output normalizer and source
layout/split identity and carry a top-level `forecast_status`.

Complete version-2 archives include the source PPO normalizer's float64
`count/mean/m2` and both of its recorded hashes. The Python adapter method
`normalize_source_features(raw[B,384])` validates finite floating input and
returns contiguous float32 `[B,384]` using the exact training-time variance
floor and clipping rule. Older archives deliberately raise when this method is
requested.

Version 3 additionally exposes an identity-bound `temporal_contract` derived
from the source collector manifest. Loading requires the trained cadence of five
`0.05`-second physics steps per `0.25`-second observation and the exact source
manifest SHA-256.

The GRU equations match PyTorch's `r,z,n` gate ordering and reset placement:
`n=tanh(x_n + r*h_n)`, followed by `h'=(1-z)*n+z*h`, with both projection biases
included. There are no version fallbacks or resident identity inputs.

On the real episode-000 training split (961 observations, 36 residents), maximum
native/Torch latent error was `5.96e-7`. Six-step physical feature mean/scale
errors were `1.79e-7` and `1.19e-7`; physiology mean/scale errors were `1.49e-8`
and `7.45e-9`. Snapshot continuation was exact. Quiet CPU medians were 203.47 ms
native versus 407.54 ms Torch for the full observe sequence, and 3.99 ms native
versus 2.78 ms Torch for six-step imagination. These timings establish viable
Torch-free inference rather than a universal performance claim.

Build with the intended interpreter:

```bash
python native/cognitive-core/build_extension.py --output-dir .
```

## Joined developmental resident

`DevelopmentalResidentCohort` is now the current-only rich controller and owns
the complete recurring 20 Hz control state:
worker GRU, four-frame causal ring, 128-slot achieved-key reservoir, sticky goal
selection, manager and action RNG streams, selected goal provenance, and the
last actual executed motor vector. A single cohort `step` accepts raw 4,453-value
observations, current 384-value canonical neural readouts, six local physiology
values, actual previous action plus oral channel, a shared model tick, time, and
reset flags. Rows in one cohort must share the model tick; this makes the
ten-tick manager boundary explicit instead of letting one row silently drive
another row's selection clock. The observation order is fixed as direct
`rich-body-v1` RGB-proximity rays `[4096]`, canonical senses `[351]`, then
physiology `[6]`. The authenticated artifact carries the train-only mean and
scale for all 4,453 columns; the native core normalizes and clips each row once.

The visual front reshapes the first 256 rays to `[4,8,32]` and the following
768 rays to `[4,24,32]`. Each branch uses replicated padding on both finite-FOV
axes, 4→16 and 16→24 convolutions with strides `(1,2)` and `(2,2)`, and its own
64-value projection. A separate 357→128 body projection produces a 256-value
frame code. Current control projects that code plus the actual previous nine
outcomes into the GRU. Goal encoding applies the same shared front to four
actual historical frames, then maps 1,024→256→64. There is no 357-input
production branch or synthetic warmup window.

Each current observation passes through the frozen visual/body front exactly
once. A private four-entry ring retains its 256-value frame codes beside the raw
goal-memory ring, so completed windows run only the 1,024→256→64 projection.
Snapshots preserve both rings and reject divergent cursors or counts. Raw
windows remain authoritative for provenance and body-goal refinement. Manager
projections and reservoir scoring run only at the ten-tick selection boundary.

The step first updates the GRU from the current observation. It then pushes the
same actual observation into the ring, encodes and stores a window only after
four real frames, and queries only that resident's valid past keys. Manager
scores are `query·key / 8 * query_gain`. A selection commits for ten ticks. The
worker receives `log1p(remaining_attempt_ticks)/log(41)`; this is time remaining
in the current attempt, not the age of the achieved memory. Before any selected
goal exists, both goal and horizon are zero. Outputs are absolute proposed
actions, clearly separate from the supplied actual previous action.

Action mode is an immutable constructor choice. `sample` draws four signed,
four hurdle, and four positive categorical decisions from snapshotted
per-resident xoshiro streams; its RNG position is independent of whether a
hurdle is active. `map` is deterministic and explicit. The Python wrapper only
authenticates the joined worker/manager artifact, packs it once, converts arrays
at the boundary, and encodes the complete native snapshot as exact base64 for a
JSON world manifest. Restoring requires the same artifact file/content identity,
execution mode, batch size, and native state schema.

The current artifact also authenticates the complete fitted GAM consequence
bank and the fixed refinement configuration. The action heads run once, then
the native private RNG draws four proposals. For a selected achieved goal, the
controller compares each proposal's inherited plus personally learned
one-tick body consequences with the per-tick change toward the last physiology
row in that actual goal window. Errors use the fitted target scales and are
bounded to four scale units. Centered scores receive a bounded `0.5*tanh` tilt;
out-of-domain proposals receive zero tilt and remain in the actor's support.
This is an engineered refinement of the local body component of an achieved
sensory goal, not a reward, welfare signal, or reachability guarantee.

The returned oral command is the native automatic law
`clamp((1-gut)*(1.1-energy),0,1)`. The selected proposal, oral value, fitted
features, inherited prediction, and tick remain pending until the host supplies
the exact executed nine-value receipt plus pre/post physiology through
`observe_consequences`. Receipt validation is cohort-atomic. Only a matching
completed physical transition updates the bounded private normalized-LMS
residual. Pending credit, learned weights, error statistics, and RNG state are
all included in snapshots.

The rich exporter accepts only a self-contained
`chreatures-rich-online-sensorimotor-development-v1` checkpoint. It rejects a
rich bootstrap combined with an unrelated earlier manager. The native rich
front was checked against an independent NumPy implementation at maximum
absolute hidden-state error `2.80e-9`; a 12-step snapshot continuation replayed
bit-exactly. A learned-artifact comparison remains required after the current
rich online run produces its final checkpoint.

On the M2 Max with the authenticated synthetic rich fixture, the complete B3
step plus consequence-receipt path averaged 1.74 ms before caching and 0.704 ms
afterward over 40 ticks. This is execution evidence, not trained-behavior
evidence.
