# MaleCNS observer annotation sidecar

The version 1 sidecar lets a public observer inspect a selected canonical
MaleCNS row without exposing annotation truth to the organism. It is a separate,
optional display asset and is not part of a CNS service, resident checkpoint,
trainer batch, predictor, memory input, or action-scoring interface.

The payload is deterministic gzip JSON. Rows are implicit indices `0..165121`
in the canonical graph order: ascending exact MaleCNS `bodyId`. Body IDs are
little-endian signed 64-bit values encoded with base64 and must be decoded as
JavaScript `BigInt`, avoiding loss through JSON numbers. Annotation strings are
dictionary-coded; empty strings remain missing rather than acquiring inferred
labels. Included fields are superclass, class, subclass, type, MANC type, side,
entry nerve, exit nerve, predicted transmitter, effective transmitter, NT basis,
and prediction confidence.

One byte of flags per row records exact sensory annotation membership, presence
in the actual-fly sensory atlas, descending/context routing, exact motor
annotation membership, MOTOR92 routing, and BODY807 routing. Two CSR structures
map a selected canonical row to named sensory ports and MOTOR92 outputs. These
memberships come from the hash-bound `fly-body-neural-atlas-v1.npz`; its evidence
grades remain attached to the port/output dictionaries. Engineered actuator and
afferent mappings remain engineered when displayed.

The sidecar contains no neural activity, private state, world labels, object
identity, task state, or inferred coordinates. It does not reuse soma coordinates;
the existing observer position atlas owns those measured coordinates separately.

The live inspector joins this sidecar to the selected model's anatomical identity
(`identity.anatomy`, not its distinct retinal `identity.atlas`). It separately reads
the selected model's receptor rows, pointers and site weights: 4,107 retinal rows,
3,936 with supported inputs and 171 without input sites. These are displayed
alongside BODY807, descending-context and MOTOR92 memberships. Unsupported rows
remain inspectable and participate in the full graph; they acquire no fabricated
sensory input.

Click a supplied soma point or search an exact body ID, type or class. The selected
row's sampled state trace uses model time and a fixed display scale. Changing the
resident or field, or restoring a checkpoint, clears the displayed trace. This
display changes neither CNS state nor control, and does not interpret rate-model
values as measured firing frequencies.

Build with:

```sh
.venv/bin/python research/fly_embodiment/export_observer_annotations.py \
  --neurons /tank/chreatures/data/malecns/derived/neurons.npz \
  --atlas research/fly_embodiment/fly-body-neural-atlas-v1.npz \
  --output-dir ~/paperbin/chreatures/integration/fly-observer-annotations-v1
```

The exporter refuses overwrites and verifies these inputs before reading them:

- MaleCNS graph: `48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`
- `neurons.npz`: `0e6706229b93cbdcab48905504bd3b8f6075534488c65ad2c8f844668036ea9a`
- actual-fly atlas: `066136815c53fd7855e12246a60ce05702f44f4ea01ee547f01734418b0bc873`

Dataset source: MaleCNS connectome v1.0, adult male *Drosophila* brain and VNC,
CC BY 4.0, DOI `10.1016/j.cell.2026.08.015`. Transmitter prediction is a dataset
annotation. It does not establish receptor expression, synaptic sign,
cotransmission, release dynamics, or behavioral function.
