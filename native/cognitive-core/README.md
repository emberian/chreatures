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
