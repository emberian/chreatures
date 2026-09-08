# MaleCNS V3 anatomical interface export

`anatomy.py` produces the immutable anatomical supports for the V3 coupled CNS
wave. It reads the canonical MaleCNS v1.0 CSR and row-aligned neuron metadata;
it does not contact, advance, or mutate a neural service.

The artifact contains the frozen dense structural masks `atlas.body_mask`
`[11233,110]` and `atlas.motor_mask` `[34,815]`, plus exact body, descending,
and motor row indices. `graph.channel` labels every neuron as unknown, fast, DA,
OA, or 5HT. `graph.weight` recomputes every canonical edge from measured counts:
fast signed efficacy is normalized by all incoming counts at its target; each
modulator is positive and normalized by same-family incoming counts. Unknown
transmitter edges remain zero. These labels establish anatomical support, not
receptor sign or kinetics.

All sensory supports first select a declared annotation modality. Side and leg
entry nerve are preserved where available. Chemical identity, body axes,
directions, interoceptive signals, and auditory frequency are learned within the
appropriate support because the annotations do not measure those functions. All
16 frequency bands therefore share the typed auditory support. Interoception,
shade, and fatigue mappings are explicitly marked proxies. MANC motor labels
remain dataset annotations; mapping those labels onto the synthetic fixture's
joints is an engineered assumption rather than a measured muscle homology.

The physical fixture order is read from `site/live/fixtures/garden.xml`: LF, LM,
LH, RF, RM, RH, each hip then knee. BODY110 stores all twelve joint angles in
that order, then all velocities, then all loads. Foot contacts use LF through RH;
fatigue repeats the twelve-joint order. MOTOR34 uses the same joint order, each
positive then negative direction, followed by its ten named ancillary outputs.

Run on the bulk host:

```sh
python research/anatomical_cns/anatomy.py \
  /tank/chreatures/data/malecns/derived \
  /tank/chreatures/data/malecns/v3/anatomical-cns-v3-r2.npz
```

The adjacent `.manifest.json` pins the graph, dimensions, per-channel rules and
counts, fast/modulator edge and synapse counts, artifact size, and checksum.
Loading the NPZ must use `allow_pickle=False`.
The exporter refuses to overwrite either an existing artifact or its manifest.

`anatomical-cns-v3-r2.npz` is the current artifact. The un-suffixed first export
and its retained manifest are historical evidence of the rejected ordering and
must not be loaded: it interleaved joint families and imposed unsupported
functional hash partitions.

## Source and interpretation

- MaleCNS connectome v1.0, adult male *Drosophila* brain and VNC, CC BY 4.0.
- Dataset DOI: `10.1016/j.cell.2026.08.015`.
- Canonical dataset SHA-256:
  `48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`.
- Derived metadata SHA-256: `0e6706229b93cbdcab48905504bd3b8f6075534488c65ad2c8f844668036ea9a`.
- Canonical CSR orientation is postsynaptic row with presynaptic `indices`.

The selector design follows the repository's source audit in
`docs/development/ANATOMICAL_MOTOR_BOOTSTRAP.md`. In the inspected metadata,
there are 815 annotated motor neurons, 1,314 descending neurons, 115 auditory
cells (114 sensory plus one annotation-missing cell), and 425 chordotonal cells
(403 VNC sensory plus 22 sensory ascending). The sensory exporter deliberately
excludes the annotation-missing auditory cell because the CNS input contract
requires a typed afferent.

## Torch recurrence and service export

`model.py` is the differentiable full-graph V3 reference used for ROCm training.
It accepts retina `[B,1771,3]`, body `[B,110]`, and the context actually delivered
to the CNS `[B,12]`. A step returns latent `[B,512]`, motor `[B,34]`, and seven
private `[165122,B]` fields: rate, adaptation, support, release availability, and
the dopamine, octopamine, and serotonin target traces. The body and motor weights
are multiplied by the exported structural masks on every forward pass. Context is
injected only at the 1,314 descending rows, and both latent and motor values are
computed after recurrent activity. Raw observations are training targets only.

`export.py` is the sole V2 bridge. It reads a sealed CHCNS2 artifact as an offline
seed, combines it with the current `anatomy.py` output, and writes a fresh CHCNS3
artifact. It copies only optic tuning, the five shared V2 type dynamics, and the
masked latent readout. Its receipt records full source identities and labels the
body110, context12, motor34, release, and family-modulation interfaces untrained.
The V3 runtime has no V2 loader and no private neural state is migrated.

```sh
python -m research.anatomical_cns.export \
  --v2-service /path/to/sealed-v2.bin \
  --anatomy /tank/chreatures/data/malecns/v3/anatomical-cns-v3.npz \
  --output /tank/chreatures/data/malecns/v3/initialized-cns-v3.bin
```

The anatomy file must be the corrected current export for the epoch. Its SHA is
bound into the service metadata; a prior selector draft must not be substituted.
