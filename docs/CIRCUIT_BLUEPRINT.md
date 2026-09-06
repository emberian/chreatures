# Circuit blueprints

`CircuitBlueprint` is a small, executable inheritance mechanism for changing
recurrent anatomy between lineages. It compiles a new sparse graph artifact;
it never edits a running brain or the pinned MaleCNS source directory.

The parent graph is the MaleCNS v1.0 reconstruction described in
[`MALECNS.md`](MALECNS.md): 165,122 traced neurons, 25,563,197 directed
connections, and 124,025,046 synapses. Its dataset hash is
`48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`.
The official source is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
and must retain the citation *Sexual dimorphism in the complete Drosophila
male central nervous system connectome*, Cell (2026),
doi:10.1016/j.cell.2026.08.015.

## Executable contract

[`circuit_blueprint.py`](../chreatures/circuit_blueprint.py) exposes:

```python
blueprint = CircuitBlueprint.load("data/circuit-blueprints/example.json")
receipt = compile_blueprint(
    parent_graph,
    parent_port_bundle,
    blueprint,
    "/tank/chreatures/data/circuit-blueprints/example",
    selector_root=repository_root,
    parent_port_sha256=parent_port_file_sha256,
)
derived = DerivedCircuitGraph.load(receipt["path"], mmap=True)
ports = NeuralPortBundle.load(Path(receipt["path"]) / "ports.npz", derived)
```

The derived loader presents the same `n`, `hash`, CSR arrays, neuron metadata,
`matrix()`, and sparse-port boundary consumed by `RemoteBrain` and
`NeuronMajorCircuit`. The compiler also emits the existing dimension-generic
`metal-csr-v2` binary layout. Production `MetalCircuit` intentionally pins the
canonical graph and is not weakened; a derived artifact needs a separately configured
native service.

A blueprint pins the parent graph, parent port semantics, and optionally the
exact parent port file. Each module selects exact ancestral body IDs from a
checksummed NPZ, declares a copy count, and chooses one boundary rule:

- `internal` clones only connections whose two endpoints are in the module.
- `incoming` also clones measured external inputs into duplicate targets.
- `bidirectional` additionally clones duplicate outputs into ancestral targets.

All copied edges retain the measured direction and integer synapse count.
`edits.add`, `edits.remove`, and `edits.reweight` use exact source/target
references. Counts for additions and reweights are positive uint32 integers.
These edits are model operations even when their endpoints descend from real
neurons.

Parent neuron indices and body IDs remain unchanged. Duplicate rows are
appended. Each duplicate has a unique deterministic synthetic body ID and ID,
plus `ancestral_indices`, `ancestral_body_ids`, `origin`, `module`, and
`copy_index`. Its type, side, transmitter provenance, and sign are inherited
from the named ancestral neuron; `origin=synthetic_duplicate` prevents those
copied annotations from being mistaken for another reconstructed neuron.

With `ports=inherit`, an input row is copied to each descendant. Readout
coefficients are split equally across an ancestor and its descendants, which
preserves the aggregate coefficient of each 384-channel population. Interface
names and dimensions remain 351 inputs and 384 readouts, while the port spec is
re-pinned to the derived graph hash.

## Provenance and normalization

`manifest.json` retains the complete source dataset record, source checksums,
license, parent graph hash, canonical blueprint, and operation summary.
`derived-edges.npz` contains one row for every surviving derived edge:

- derived source and target indices;
- ancestral source and target indices;
- compiled integer synapse count;
- a basis code distinguishing cloned measured edges, explicit additions, and
  explicit reweights.

Removed connections remain in `edge-provenance.json`. This makes each changed
edge traceable without attaching a string to all 25.6 million ancestral edges.
The duplicated cells and every cloned or edited connection are synthetic
developmental hypotheses. The source anatomy supports the template and counts;
it does not show that this duplication occurs in a fly.

Incoming normalization is recomputed from the complete derived row. A
bidirectional copy adds new presynaptic sources to some ancestral target rows,
so those rows acquire a larger denominator and the normalized weights of their
unchanged ancestral inputs decrease. The manifest records the number and hash
of affected ancestral rows. The compiler proves before applying explicit edits
that the induced ancestral-to-ancestral edge set and counts are exact; it also
hash-verifies the read-only parent artifacts before and after compilation.

## Gamma1pedc executable example

[`gamma1pedc-duplicate-v1.json`](../data/circuit-blueprints/gamma1pedc-duplicate-v1.json)
duplicates the exact 3,623 connected Kenyon cells, two MBON11 neurons, and two
PPL101 dopaminergic references in the existing checksummed gamma1pedc
substrate. It declares bidirectional boundary cloning and contains no explicit
edge edits. The name identifies the measured template; the duplicate itself is
a synthetic developmental test.

The bulk artifact lives at
`/tank/chreatures/data/circuit-blueprints/gamma1pedc-duplicate-v1`. Compilation
produced 168,749 neurons, 26,557,080 edges, and 126,896,270 synapses. It added
993,883 traceable cloned edges: 584,620 internal, 182,595 incoming, and 226,668
outgoing. All 25,563,197 ancestral-to-ancestral edges and ancestral metadata
rows matched exactly. Outgoing clones changed the normalization denominator of
3,152 ancestral rows, adding 939,080 incoming synapses across those rows
(median increase 6, maximum 99,970).

An actual `NeuronMajorCircuit` run on an AMD RX 6750 XT with PyTorch
2.5.1+rocm6.2 compared twelve 50 ms steps under a 351-channel graded input with
a zero-input control. All 3,627 duplicate cells were active and all had a
nonzero input-conditioned rate difference; the maximum duplicate difference
was 0.106406 and the maximum 384-channel readout difference was 0.610962. The
paired run took 0.198 seconds. This verifies that the compiled anatomy executes
and participates in input-to-readout dynamics. It is not evidence that the
synthetic duplication improves behavior or matches developmental biology.

Run the compiler with:

```bash
python scripts/compile_circuit_blueprint.py \
  data/circuit-blueprints/gamma1pedc-duplicate-v1.json \
  /tank/chreatures/data/circuit-blueprints/gamma1pedc-duplicate-v1 \
  --graph /tank/chreatures/data/malecns/derived \
  --ports data/ports/retinal-v1-maps.npz \
  --selector-root .
```

The focused local test uses a six-neuron derived graph to exercise module
duplication, all three edit operations, port inheritance, source immutability,
provenance, and sparse rate propagation:

```bash
.venv/bin/python -m pytest -q tests/test_circuit_blueprint.py
```
