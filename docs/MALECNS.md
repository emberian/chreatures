# MaleCNS v1.0 graph backend

Chreatures can load the full curated-neuron graph of the adult male fly brain
and ventral nerve cord from the official MaleCNS v1.0 flat connectome. This is
separate from `data/connectome/circuit.npz`, the compact female FlyWire subset
used by the earlier browser/runtime.

The derived graph contains 165,122 traced neurons, 25,563,197 directed
connections, and 124,025,046 synapses. It is the full induced graph of official
body annotation rows whose `status` is exactly `Traced`: no top-k selection and
no edge threshold is added. Both endpoints must be traced. The extractor
explicitly excludes 46,455 non-neuron or non-curated annotation rows with
statuses `Orphan`, `Glia`, `Unimportant`, `Assign`, `Anchor`, or missing. This
keeps the curated brain-plus-cord organism graph instead of pulling in the much
larger set of untraced segmentation fragments.

## Bulk artifacts

The sources and full derived graph live outside the repository at
`hbox:/tank/chreatures/data/malecns`:

```text
malecns/
  source/
    body-annotations-male-cns-v1.0-minconf-0.5.feather
    body-neurotransmitters-male-cns-v1.0.feather
    connectome-weights-male-cns-v1.0-minconf-0.5.feather
  derived/
    indptr.npy
    indices.npy
    counts.npy
    row_synapses.npy
    neurons.npz
    manifest.json
```

`indptr.npy`, `indices.npy`, and `counts.npy` are a canonical CSR graph. Rows
are postsynaptic targets; each `indices` entry is a presynaptic source and the
aligned `counts` entry is its measured synapse count. Sources strictly increase
inside each row. All four uncompressed graph arrays can be memory mapped, so
opening the graph does not unpack the 1 GB Feather source or duplicate 25.6
million target indices. `row_synapses` stores each retained target's total
incoming count.

`neurons.npz` follows ascending `bodyId` order and preserves exact IDs, labels,
types, instances, superclass/class/subclass, soma and root side, soma neuromere,
entry and exit nerve, tracing status, `fru`/`dsx` and dimorphism fields, and the
source transmitter fields. The repository keeps only
`data/malecns/manifest.json`; every bulk artifact and its checksum is pinned
there without checking hundreds of megabytes into git.

## Loading and selecting neurons

Set `CHREATURES_MALECNS_DIR` when the bulk mount has a different path, or pass
the directory directly:

```python
from chreatures.malecns import load_malecns

graph = load_malecns("/tank/chreatures/data/malecns/derived")
print(graph.summary())

visual = graph.select(class_="visual")
left_leg_sensory = graph.select(
    superclass="vnc_sensory",
    soma_neuromere=["T1", "T2", "T3"],
    side="L",
)
afferents = graph.population_indices("afferent")
efferents = graph.population_indices("efferent")
matrix = graph.matrix(normalized=True, signed=True)
```

`matrix` is a SciPy CSR matrix with the anatomical target/source orientation.
Its default values are
`count * sign[source] / max(retained incoming count[target], 1)`. This bounded
rate-model normalization is a Chreatures runtime choice, not a MaleCNS
measurement. Call `matrix(normalized=False, signed=False)` for raw float
synapse counts. Code that does not need SciPy can consume `indptr`, `indices`,
`counts`, and `row_synapses` directly. `edge_arrays()` can materialize explicit
targets for APIs that require COO, but that intentionally allocates another
`int32[E]` array.

Exact metadata selectors do not assume FlyWire or projection-neuron naming.
Scalar criteria mean equality, lists mean membership, and different fields are
ANDed. The `groups` compatibility attribute is the exact MaleCNS `superclass`,
not one of the hand-selected groups from the earlier compact circuit.

## Sparse organism interfaces

`graph.default_input_map` returns `(names, matrix)` where `matrix` has shape
`[165122, 16]` and names follow the existing runtime order:

```text
odor L0, odor L1, odor L2, odor R0, odor R1, odor R2,
obstacle left, obstacle right, red, green, blue,
tone low, tone middle, tone high, shade, contact
```

These are engineered stimulus assignments over annotated sensory populations.
Olfactory afferents are split by anatomical side and a deterministic body-ID
bucket; visual afferents supply the two obstacle and three color channels;
the auditory subclass supplies three tone channels; hygro/thermosensory cells
supply shade; tactile/mechanosensory cells supply contact. Overlap among visual
channels is intentional. This map says where the synthetic organism injects a
signal; it does not claim those neurons encode the fictional sensor values in
the fly.

`graph.default_readout_map` returns an exactly 48 by 165,122 sparse matrix. Its
rows are disjoint efferent populations, each normalized to a population mean.
The first 47 rows are the largest exact
`(superclass, exit-nerve-or-soma-neuromere-or-class, side)` signatures with
lexical tie-breaking; `other_efferent` contains the remainder. The row names
retain those annotation values. This is a stable organism interface and not a
calibrated motor decoder.

Custom sparse maps use exact selectors or explicit local indices and never
form a dense neuron-by-channel matrix:

```python
input_names, input_map = graph.build_input_map({
    "left visual": {"class_": "visual", "side": "L"},
    "T1 sensory": {"superclass": "vnc_sensory", "soma_neuromere": "T1"},
})
readout_names, readout_map = graph.build_readout_map({
    "leg motor": {"superclass": "vnc_motor", "soma_neuromere": ["T1", "T2", "T3"]},
})
```

## Neurotransmitters

The official transmitter table provides per-body ground truth, direct
prediction, type-level aggregation, and consensus values. Chreatures chooses
`ground_truth` first, then a non-`unclear` `consensus_nt`, then a non-`unclear`
`predicted_nt`. All source fields and the chosen `nt_basis` remain available.
For the traced set, 83,496 neurons use ground truth, 78,024 use consensus,
1,078 use direct prediction, and 2,524 remain unavailable.

The optional rate sign maps acetylcholine to `+1`; GABA, glutamate, and
histamine to `-1`; and monoamines or unavailable values to `0`. A neuron-level
transmitter label is not exact physiology: receptor identity, cotransmission,
synapse-specific effects, and modulatory dynamics can change the effect. Raw
counts and transmitter metadata stay separate so another execution model can
make a different choice.

## Rebuilding

The extractor validates the three pinned source SHA-256 checksums, reads Arrow
record batches directly, and makes two connectivity passes: one to size CSR
rows and one to fill them. It does not load the 151,856,684-row edge table into
RAM. On the bulk host:

```bash
/tank/chreatures/data/.venv/bin/python scripts/acquire_malecns.py \
  --root /tank/chreatures/data/malecns \
  --download
```

The official project distributes MaleCNS under CC BY 4.0. Cite *Sexual
dimorphism in the complete Drosophila male central nervous system connectome*,
Cell (2026), doi:10.1016/j.cell.2026.08.015, as requested by the
[MaleCNS project](https://male-cns.janelia.org/), and retain attribution when
redistributing derived data. Source URLs, byte sizes, hashes, schema, selection
rule, exact category counts, and the derived dataset hash are recorded in the
manifest.
