# Connectome data

`data/connectome/circuit.npz` is a compact circuit extracted from the public
FlyWire FAFB materialization 783. It contains 6,789 exact FlyWire neurons,
564,810 directed connections, and 1,367,086 measured synapses. It is an induced
subgraph: every stored edge is present in the pinned source connectivity table,
the presynaptic/postsynaptic direction is unchanged, and no threshold removes
weak connections. There are no generated or randomly filled edges.

The circuit is centered on the bilateral olfactory memory system:

| Group | Selection from annotations | Neurons |
| --- | --- | ---: |
| PN | `cell_class == ALPN` | 685 |
| KC | `cell_class == Kenyon_Cell` | 5,177 |
| MBON | `cell_class == MBON` | 96 |
| DAN | `cell_class == DAN` | 331 |
| CX | 250 `cell_class == CX` neurons with the most measured synapses to/from the core | 250 |
| descending | 250 `super_class == descending` neurons with the most measured synapses to/from the core | 250 |

The first four classes form the 6,289-neuron core. Ranking for the two extension
groups sums source `Connectivity` on edges in either direction between a
candidate and any core neuron. Ties resolve by ascending root ID. Only candidates
with at least one such measured synapse are eligible. After selecting neurons,
all source edges whose endpoints are selected are retained. If a source ever
contains repeated rows for one directed pair, the extractor sums those rows.

## Files and schema

`circuit.npz` can be loaded with `numpy.load(..., allow_pickle=False)`:

| Array | Shape / dtype | Meaning |
| --- | --- | --- |
| `ids` | `Unicode[N]` | Exact FlyWire root IDs as strings; never coerce these through JavaScript numbers |
| `pre`, `post` | `int32[E]` | Directed presynaptic and postsynaptic local indices |
| `count` | `float32[E]` | Positive measured synapse count for that connection |
| `sign` | `float32[N]` | Declared rate-model sign for each presynaptic neuron |
| `labels` | `Unicode[N]` | Cell type, hemibrain type, subclass, class, or ID fallback |
| `type` | `Unicode[N]` | FlyWire `cell_type` annotation |
| `side` | `Unicode[N]` | FlyWire hemisphere annotation (`left`, `right`, or `center`) |
| `group` | `Unicode[N]` | `PN`, `KC`, `MBON`, `DAN`, `CX`, or `descending` |
| `predicted_nt` | `Unicode[N]` | FlyWire `top_nt` prediction |
| `effective_nt` | `Unicode[N]` | Known-transmitter-first value used to assign model sign |
| `nt_basis` | `Unicode[N]` | `known_nt`, `top_nt`, or `unavailable` provenance for `effective_nt` |
| `nt_confidence` | `float32[N]` | `top_nt_conf`, with unavailable values represented as zero |

`neurons.json` is the row-aligned human-readable metadata. It includes the root
ID, local index, group, label, flow, super/class/subclass, cell and hemibrain
types, side, predicted and known transmitter fields, VFB and FBbt identifiers,
and the model sign. `manifest.json` pins revisions, source and artifact SHA-256
digests, selection rules, exact totals, schema, citation, and licensing.

The `sign` array is a modeling choice, not a synaptic sign observed in EM.
Positive small-molecule literature annotations in `known_nt` take precedence;
`top_nt` is the fallback when `known_nt` has only negative evidence, peptides,
or no value. Multiple positive known transmitters are retained, joined by `+`.
Effective acetylcholine is `+1`; effective GABA and glutamate are `-1`; dopamine,
serotonin, octopamine, tyramine, nitric oxide, unavailable, and any future value
are `0`. A mixed acetylcholine plus GABA/glutamate annotation is also `0` because
one neuron-level sign cannot represent it. The raw prediction remains in
`predicted_nt` and the chosen provenance is explicit in `nt_basis`.

This precedence matters here: `top_nt` predicts dopamine for 5,172 of 5,177 KCs,
while the literature-backed `known_nt` field identifies every selected KC as
acetylcholine. The effective array therefore keeps all KCs cholinergic and makes
the measured PN to KC to MBON pathway functional under the declared sign model.

| Group | `+1` | `-1` | `0` |
| --- | ---: | ---: | ---: |
| PN | 474 | 211 | 0 |
| KC | 5,177 | 0 | 0 |
| MBON | 51 | 45 | 0 |
| DAN | 0 | 2 | 329 |
| CX | 48 | 164 | 38 |
| descending | 173 | 45 | 32 |

No propagation class is wholly unsigned. Most DANs are zero because dopamine is
kept as a separate modulatory transmitter instead of being treated as generic
excitation.

No sensory injection or motor readout mapping is included in these data. A
runtime may build synthetic injection/readout mappings from the documented
groups, but must identify those mappings as synthetic and must not describe them
as FlyWire measurements.

## Pinned sources and licensing

The measured pair connectivity and completed-neuron list come from
[`philshiu/Drosophila_brain_model`](https://github.com/philshiu/Drosophila_brain_model)
at commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`:

- `Connectivity_783.parquet`: 15,091,983 rows, SHA-256
  `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347`
- `Completeness_783.csv`: 138,639 rows, SHA-256
  `bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311`

That repository is MIT licensed. Its connectivity is a transformed copy of the
FlyWire v783 data deposited at
[`10.5281/zenodo.10676866`](https://doi.org/10.5281/zenodo.10676866).

Neuron metadata comes from
[`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations)
tag `v3.1.0`, commit `8587524c1748ce5ef2080822a2fc890fc03bf597`:

- `supplemental_files/Supplemental_file1_neuron_annotations.tsv`: 139,248 rows,
  SHA-256 `9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be`

Release `v3.1.0` still annotates FlyWire materialization 783 and adds later
cross-validated typing. The annotation table supplies `root_id`, `flow`,
`super_class`, `cell_class`, `cell_sub_class`, `cell_type`, `hemibrain_type`,
`side`, transmitter predictions/confidence, known-transmitter references, and
ontology IDs used here.

[FlyWire's public-release guidelines](https://flywire.ai/guidelines) specify
CC BY-NC 4.0 for public release 783. The Zenodo connectivity record currently
reports CC BY 4.0 in its metadata; this derived artifact follows the more
restrictive FlyWire guideline, CC BY-NC 4.0. Attribute and cite:

- Dorkenwald et al. (2024), *Neuronal wiring diagram of an adult brain*, Nature
  634, 124-138, doi:10.1038/s41586-024-07558-y.
- Schlegel et al. (2024), *Whole-brain annotation and multi-connectome cell
  typing of Drosophila*, Nature 634, 139-152,
  doi:10.1038/s41586-024-07686-5.

## Reproducing the extraction

Bulk inputs are kept on `hbox:/tank/chreatures/data`, outside the laptop repo.
With pandas and pyarrow installed, the script can download any missing pinned
input, validate every source checksum, and extract in place:

```bash
python scripts/acquire_connectome.py \
  --source-dir /tank/chreatures/data \
  --download \
  --output-dir /tank/chreatures/data/extracted
```

Copy only `circuit.npz`, `neurons.json`, and `manifest.json` from `extracted/`
into `data/connectome/`. The checked-in manifest records the checksums from this
exact run.
