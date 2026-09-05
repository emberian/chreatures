# Native library integrations

The first integration slice uses each selected upstream library through its
public native API. The checked-in artifacts are execution evidence, not claims
that either fitted structure is a biological mechanism.

## gamfit

The Python integration pins `gamfit==0.1.259`, whose Rust-backed wheel is the
release at upstream commit
[`7c7eca8ac4826de95c8e743a20294bee132a9bcc`](https://github.com/SauersML/gam/commit/7c7eca8ac4826de95c8e743a20294bee132a9bcc).
The upstream head inspected while implementing this adapter was
`4773a6fbed33af359b8a6fff5636385f1a2d8de8`. The primary sources inspected were
the repository's `README_PYPI.md`, `docs/getting-started.md`, and
`docs/persistence.md`. They define `gamfit.fit`, response-scale `predict`,
`Model.save`, and `gamfit.load`; the CLI uses those APIs directly. The package
is AGPL-3.0-or-later.

Install in an isolated environment and run the bounded regression:

```bash
uv venv integrations/.venv
uv pip install --python integrations/.venv/bin/python \
  -r integrations/gamfit-requirements.txt
integrations/.venv/bin/python integrations/gamfit_regression.py
```

The experiment deterministically generates 180 observations from
`0.65*x**2 + sin(2.2*x)` with Gaussian noise, holds out every third point, and
fits `y ~ s(x, k=14)` on the other 120. It compares held-out RMSE with a declared
training-mean null, saves the native `.gam` payload, reloads it, and requires
identical predictions. Outputs live in `integrations/artifacts/gamfit/`.

The run on 5 September 2026 used the macOS arm64 CPython ABI3 wheel. It reported
the native extension available, completed the fit in about 0.18 seconds, scored
held-out RMSE 0.07447 versus null RMSE 1.88108 (ratio 0.03959), and reloaded a
213,188-byte model with maximum prediction delta 0.0. Exact machine-readable
values and build identifiers are in `regression_result.json`.

## Universal Weave

The Rust bridge pins `universal-weave` 0.5.0 by Git revision
[`7a5a0dabb94885e44ad8a6c4355c015d7f38020f`](https://github.com/transkatgirl/universal-weave/commit/7a5a0dabb94885e44ad8a6c4355c015d7f38020f).
The primary sources inspected were its `README.md`, `Cargo.toml`,
`src/independent/mod.rs`, and native independent/archived integration tests.
The crate describes `IndependentWeave` as its DAG implementation and exposes
validated serde and rkyv serialization. The bridge enables the serde feature
and uses `IndependentWeave::insert`, `validate`, graph traversal, serde
serialization, and serde deserialization directly. The package is Unlicense.

Run the bridge from its crate directory:

```bash
cd integrations/weave
cargo run --locked -- --output ../artifacts/weave/evidence.weave.json
```

Without an input file this creates an explicitly labelled two-node synthetic
demonstration. With `--input REQUEST.json`, the bridge imports actual habitat
journal entries as independent episode nodes. Each node retains the source
event's string ID, numeric time, text, and complete JSON object. Optional
evidence records can name journal event IDs as parents. Missing parents and
duplicate IDs are rejected rather than repaired.

The bridge writes the library's own serialized `IndependentWeave`, reads it
back into the same native type, validates the reloaded topology, checks
equality, and prints a JSON receipt. Numeric tensors and checkpoints remain
outside these display records; their `artifact_uri` fields are references. The
serde payload has integration schema metadata but is not promised compatible
with future crate releases, so both the Git revision and generated `Cargo.lock`
stay pinned.

## Observing habitat records

`integrations/observe_habitat.py` is the bounded connection to project records.
It accepts the `chreatures-checkpoint-v1` envelope written by `Habitat.save`, a
JSON object containing `journal` and `history`/`telemetry`, or a CSV with
`time`, `energy`, and `activity` columns. Checkpoint hashes are verified before
any records are read. Non-finite and malformed telemetry rows are counted and
excluded; journal entries are never silently rewritten.

Run it after saving a habitat:

```bash
integrations/.venv/bin/python integrations/observe_habitat.py \
  runs/residents.json --output-dir integrations/artifacts/observatory
```

At least 60 finite rows, 12 distinct model times, eight distinct energy values,
and three distinct activity values are required before fitting
`activity ~ s(energy, k=8) + s(time, k=8)`. If those criteria are not met, the
report says `gamfit.status: skipped` with the exact reason. If adequate data are
present but the pinned native wheel is absent or a fit fails, the report says
`unavailable` or `failed`; it never substitutes another estimator. A completed
fit is explicitly descriptive, uses every fifth time-ordered row as an
interpolation diagnostic, and persists/reloads the native model before it is
reported.

The command writes `observatory_report.json`, the native
`habitat_journal.weave.json`, its explicit `weave_import_request.json`, and a
native `.gam` model only after a successful fit. `telemetry_used.csv` is the
compact, immutable table actually passed to the fitter; its three numeric
fields retain ten significant decimal digits. Input hashes, habitat ID, record
counts, rejected rows, resident IDs, time range, native versions, metrics,
persistence hashes, and limitations are recorded for the observatory.

The checked-in observatory artifact is from the actual immutable
`runs/first-live-checkpoint.json` checkpoint for habitat
`da3ed986-34f1-4b95-a7d3-d4336617aded`, captured through model time 63.4
seconds. Its checkpoint hash
`96108d9a4ed92ae15784bbf67c6661200c512dee52d6a70cbb73f55b2f4eb3e0` was
verified before extraction. All 378 history rows were finite and came from
Mica, Fern, and Pip. Their exact fitter excerpt is a 13,400-byte CSV. The native
fit completed in about 0.15 seconds, with held-out RMSE 0.03762 versus 0.05312
for the training-mean null; this modest predictive comparison remains purely
descriptive. During fitting, gamfit also printed its `RHO uncertainty` warning
with heavy-tail diagnostic `k_hat=0.525`; downstream interpretation should keep
that smoothing-uncertainty warning visible. The orchestrator captures native
standard-error output in the report's `native_messages` rather than discarding
it. The 357,147-byte model reloaded
with prediction delta 0.0. The
native Weave artifact contains the checkpoint's four journal episodes and one
explicitly labelled GAM evidence node, and its 3,476-byte serialization passed
reload equality and topology validation.
