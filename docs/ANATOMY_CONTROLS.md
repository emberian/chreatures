# MaleCNS anatomy controls

An anatomy control asks whether a learned result depends on the measured
MaleCNS wiring rather than graph size, degree alone, or the sensory and readout
interfaces. It is an alternate sparse graph for matched training runs. It is
not another reconstruction and must never be described as biological wiring.

## Matched directed rewiring

`matched-rewire-v1` starts from every curated MaleCNS v1.0 edge. For two edges
`u -> v` and `x -> y`, it proposes `x -> v` and `u -> y`. Candidate edges are
paired only when all of these values are equal:

- exact measured synapse count;
- effective transmitter annotation of the presynaptic neuron;
- exact presynaptic superclass;
- exact postsynaptic superclass.

Equal-weight endpoint swaps preserve every neuron's directed in-degree,
out-degree, incoming synapse total, and outgoing synapse total. The strata
preserve edge and synapse mixing by effective transmitter and superclass. The
transmitter field is an annotation used for matching; it is not an assertion
of exact synaptic physiology.

The source is a simple directed edge table, so proposals that would create a
duplicate pair are reverted. MaleCNS contains self-edges, so the source policy
allows them in the control and reports their counts before and after. Edges in
singleton strata, pairs with a repeated endpoint, and pairs rejected during
duplicate cleanup stay unchanged. The artifact reports the resulting overlap
instead of claiming complete randomization.

The implementation stores only CSR arrays and linear edge work arrays. It does
not allocate a dense neuron-by-neuron matrix.

## Build and load

Build on bulk storage:

```shell
python scripts/build_connectome_controls.py \
  --graph /tank/chreatures/data/malecns/derived \
  --output /tank/chreatures/data/malecns/controls/matched-rewire-v1-seed20260905 \
  --scratch /tank/chreatures/cache/matched-rewire-v1-seed20260905 \
  --seed 20260905 --verify-source
```

The builder first runs the same algorithm on 4,096 real target rows, checks
its invariants, and then performs one full pass. It refuses to replace an
existing output or the source directory. A failed build leaves a `.partial`
directory for diagnosis.

The result has the same memory-mapped CSR and neuron metadata interface as the
canonical graph:

```python
from chreatures.connectome_controls import load_connectome_control
from chreatures.remote_brain import RemoteBrain

control = load_connectome_control(
    "/tank/chreatures/data/malecns/controls/matched-rewire-v1-seed20260905",
    source_dataset_hash=(
        "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625"
    ),
)
brain = RemoteBrain(control, device="cuda")
```

`manifest.json` records the source graph hash and artifact hashes, the complete
control spec and its hash, the output artifact hashes and dataset hash, all
proposal/rejection counts, edge-set overlap, self-edge counts, and measured
invariants. `control-spec.json` is also hashed as part of the output dataset.

## Recurrence controls

`RemoteBrain` accepts two runtime controls without changing the loaded graph:

```python
no_recurrence = RemoteBrain(graph, recurrence=False, device="cuda")
injected = RemoteBrain(graph, recurrence=some_scipy_csr, device="cuda")
```

The injected matrix must be sparse, square, and already contain the desired
normalization and signs. Metadata identifies graph, injected, or disabled
recurrence and hashes an injected CSR. Snapshot version 3 pins that recurrence
identity; legacy snapshots may only restore with the ordinary graph recurrence.
This gives experiments three explicit conditions: measured wiring, matched
rewiring loaded as the graph, and recurrence disabled. Each condition still
needs a separately initialized learning run and a declared seed.

## Full control receipt

The substantive control was built on the full curated graph at:

```text
/tank/chreatures/data/malecns/controls/matched-rewire-v1-seed20260905
```

Its output dataset SHA-256 is
`584b0ad7dd5cfe8b80e024804523baada9367466b42385104553337ce0ecae2a`.
Its control spec SHA-256 is
`921846065dc139b3d8239b7e1b906bfaf96adee154f68ce34454107742817d56`;
the source dataset is
`48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`.

The full pass measured:

| Quantity | Value |
|---|---:|
| neurons | 165,122 |
| directed edges | 25,563,197 |
| measured synapses | 124,025,046 |
| occupied matching strata | 34,791 |
| attempted swap pairs | 12,770,702 |
| accepted swap pairs | 11,951,971 |
| rewired edges | 23,477,604 (91.8414%) |
| untouched source edges | 2,085,593 (8.1586%) |
| duplicate-colliding pairs reverted | 760,166 |
| repeated-source pairs rejected | 50,235 |
| repeated-target pairs rejected | 8,330 |
| singleton-stratum edges untouched | 21,793 |
| source/control self-edges | 101 / 1,052 |

All seven full-report invariants are exact: per-neuron out-degree, per-neuron
outgoing synapse strength, per-target in-degree, per-target incoming synapse
strength, edge synapse-count histogram, transmitter/superclass edge mixing, and
transmitter/superclass synapse mixing. Every output row is sorted and contains
unique presynaptic IDs. The self-edge increase is retained in the report rather
than hidden; self-edges were permitted because the source graph contains them.
