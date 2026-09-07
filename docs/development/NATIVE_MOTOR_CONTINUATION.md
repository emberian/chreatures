# Native resident v8: executed motor continuation

The current native population export/execution and outer/private resident
snapshot formats advance to v8. The inner motor library advances to
`chreatures-private-motor-suffix-v2`. The immutable Torch developmental base
remains v5; the recurrent consequence predictor remains v3, context1560/H8.
Old lives retain their frozen engine; there is no v7 restore fallback.

The v7 controller could propose only the first action of a recalled sequence at
each tick. Selecting that first action also incremented sequence support, even
though its remaining actions had not been executed. Native v8 keeps a private
slot/generation/phase cursor and measured per-phase attempt outcomes. One of the
four recalled proposals is reserved for the active sequence's remaining actions.
The ordinary native scorer re-evaluates that proposal after current sensation;
selecting another proposal interrupts it. Continuation never implicitly restarts
at phase zero and receives no hardcoded persistence bonus.

Only exact, contiguous, acknowledged execution of the complete sequence updates
its support and per-phase empirical outcome means. Initial physical capture
counts as one complete execution. Overrides, interruptions, tick gaps and slot
replacement cancel the attempt without changing its empirical value. Partial
execution still enters the ordinary actual-action capture history. All cursors,
last actual ticks, pending selection flags, attempt outcomes, counters and RNG
are private checkpoint state; births clear them.

Remaining horizons can be one to seven ticks. A predicted achieved-history key
for H1–H3 combines the preceding actual frame codes with predicted future codes,
so the four-frame key never indexes before its forecast. The exporter/loader now
use the existing predictor calibration for all eight horizons. Observations,
immutable tensors and policy input dimensions are unchanged.

The observer seam adds `candidate_suffix_phase[B,8]` (uint8),
`motor_suffix_completed_total[B]` and `motor_suffix_interrupted_total[B]` (uint64).
Existing `candidate_suffix_length[B,8]` reports remaining ticks; slot/generation
and selected candidate identify which experience and phase was proposed.
Support is a complete-execution count, not calibrated confidence.

## Executed validation and limits

The focused native lifecycle scenario
`joined_execution_continuation_outcome_interrupt_and_restore` passed. It covers
all eight executed phases, pending and midsequence exact restoration, complete
outcome updates, local-action interruption, host override, a missing tick,
generation rejection, growth and cold clearing. It supplies action/outcome
receipts directly; it is not a physical behavior assay. The macOS test executable
required the configured CPython dylib to be preloaded after compilation.

`scripts/probe_native_motor_continuation.py` then ran one B8 joined boundary
check using the actual exported model and native predictor/selector on hbox.
Its ten synthetic sensory/outcome boundaries acquire actual supplied-action
sequences. An explicitly analytical checkpoint cursor fixture covers remaining
horizons8..1 in one batch. All66 next-decision fields restore exactly; all17
pending receipt fields and the resulting private snapshot also match. The
forecast tensor is finite and shaped `[8,8,12]`. All77 immutable arrays and the
consequence laws match the prior v7 artifact exactly. The selector chose another
recalled proposal in every phase-sweep row; this verifies interruptibility, not
learned persistence or physical competence. The native lifecycle scenario,
separately, verifies complete-sequence attribution.

Compact executed output: `data/training/native-v8-boundary.receipt.json`.
No physical world or neural service was advanced by this lane.

## Materialized artifact

- Artifact: `/tank/chreatures/scratch/cognitive-v8-integration/artifacts/developmental-resident-population-v8.npz`
- File SHA256: `c633b66057d31f2ddb0e0a3d345a74c746ab60ef087a98ca426d86cafd8787d6`
- Internal identity: `ee1973ec7bad21c8a102088ac62b77bb3c176ac87a4fe2482542cb810235aaa8`
- Native module: `/tank/chreatures/scratch/cognitive-v8-integration/_cognitive_core.cpython-312-x86_64-linux-gnu.so`
- Torch v5 source: `/tank/chreatures/runs/development/population-v5-body-actuator-seed20260917-attempt4/development.pt`
- Predictor v3 source: `/tank/chreatures/data/models/sensorimotor-rich/population-v6-predictor-seed20260918/rich-recurrent-consequence-v3.npz`

The old v7 artifact is unchanged. The isolated build and reexport used the
project-owned `/tank/chreatures/envs/rocm-dev/bin/python`; export ran on CPU and
did not consume a live world's neural/GPU state.

```sh
cd /tank/chreatures/scratch/cognitive-v8-integration
CARGO_TARGET_DIR=/tank/chreatures/scratch/cognitive-v8-integration/target PYO3_PYTHON=/tank/chreatures/envs/rocm-dev/bin/python /tank/chreatures/envs/rocm-dev/bin/python native/cognitive-core/build_extension.py --output-dir .
PYTHONPATH=. /tank/chreatures/envs/rocm-dev/bin/python scripts/export_developmental_resident.py --checkpoint /tank/chreatures/runs/development/population-v5-body-actuator-seed20260917-attempt4/development.pt --predictor /tank/chreatures/data/models/sensorimotor-rich/population-v6-predictor-seed20260918/rich-recurrent-consequence-v3.npz --consequence-laws integrations/gam_mechanisms/artifacts/rich_body_laws_v2/body_consequence_laws.json --output artifacts/developmental-resident-population-v8.npz --trusted-checkpoint
PYTHONPATH=. /tank/chreatures/envs/rocm-dev/bin/python scripts/probe_native_motor_continuation.py --artifact artifacts/developmental-resident-population-v8.npz --prior-artifact /tank/chreatures/scratch/cognitive-v7-integration/artifacts/developmental-resident-population-v7.npz --assignment /tank/chreatures/campaigns/population-v6-b5efde99-seed20260918/plans/plan-0000/batch-0000.json --receipt artifacts/native-v8-boundary.receipt.json
```

## Training boundary

`collect_rich_sensorimotor.py` executes the native composite controller, so its
new corpus includes these continuation proposals and actual chosen outcomes.
The existing `develop_rich_sensorimotor.py` PPO calls the Torch worker directly;
it does not optimize native candidate selection, contextual memory or suffix
selection. A predictor refit on native trajectories therefore learns this
controller's actual outcomes, while the existing PPO cannot be described as
training the complete deployed controller. This mismatch was handed to the
learning pipeline lane; no new RL architecture was added in this repair.
