# Coupled developmental inheritance

`chreatures.developmental_genome` defines a small, immutable genotype that can
change metabolism, physical development, and neural port expression together.
It compiles into the existing `MetabolicWeb` and `GrowthSystem`; it is not a
second simulator or a controller policy.

The allocation coefficients are engineered inheritance constraints. They have
not been measured as developmental tradeoffs in an animal or plant. Their job
is to make changes accountable and resource limited while the simulator gains
the mechanisms needed to test them.

## Immutable boundary and provenance

A genome carries canonical SHA-256 identity for itself and exact identities for
the common chemistry, base growth grammar, measured MaleCNS graph, neural port
specification, and compiled port bundle. Compilation rejects any source that no
longer has the recorded identity. An offspring records the parent and founder
genome SHA-256, mutation operator version, unsigned mutation seed, generation,
and mutation scale.

The schema is deliberately closed. Runtime pools, ATP, growth buds and clocks,
random generator state, neural rates and adaptation, memories, contexts, and
optimizer state are rejected. Those values remain private lifetime state and
cannot silently become inherited data.

The checked nonstructural founder remains at the configured path
`data/development/common-ancestor-v1.json`, and its deterministic example child
remains at `data/development/common-ancestor-offspring-seed37-v1.json`. Their
contents use the one-way `chreatures-developmental-genome-v2` schema; the paths
remain stable for existing callers. Both declare `neural.circuit: null` and
therefore retain the exact current graph. The structural research founder is
`data/development/circuit-common-ancestor-v2.json`. These are synthetic research
genotypes, not reconstructed natural ancestors.

## Shared allocation budget

Four nonnegative allocation fractions sum to exactly one:

- photosynthesis;
- digestion;
- structural synthesis;
- geometry.

The first three divide one bounded total enzyme-activity budget. Expression
weights then divide each category among exact compartment/reaction loci. The
emitted order is `soma`, `gut`, and `allocated_structure`. Digestion is expressed
only in the gut row, while tissue already transferred to the allocated structure
row has turnover but no digestion enzyme. This compartment boundary prevents a
digestive reaction from consuming allocated tissue merely because both use the
same chemical species.

Geometry receives a bounded shared factor relative to the founder allocation.
It modifies actual rule segment length and radius and leaf area and thickness.
The resulting grammar is passed through `GrowthSystem` validation. Since the
native growth request computes capsule volume and leaf mass from those emitted
dimensions, an offspring's geometry allocation changes real soft/tough-tissue
and ATP requests. The compiler receipt reports every final enzyme activity and
dimension multiplier.

Housekeeping respiration and turnover have a separate low ceiling. They remain
bounded inherited expression, but they do not consume the four-way experimental
allocation budget. This keeps the present tradeoff legible; it does not claim
that real housekeeping is cost free.

## Actual constructors

```python
from chreatures.developmental_genome import DevelopmentalGenome

genome = DevelopmentalGenome.load(
    "data/development/common-ancestor-v1.json"
)
phenotype = genome.compile()

# The returned object is the real native-backed GrowthSystem.
growth = phenotype.new_growth(seed=19)

# A Biosphere can concatenate rows from several phenotypes into one shared web.
rows = phenotype.enzyme_rows()  # soma, gut, allocated_structure

# This convenience path constructs a real isolated native MetabolicWeb.
web = phenotype.new_metabolism(
    pools=[
        {"mineral": 3, "inorganic_carbon": 3, "reserve": 3},
        {"soft_tissue": 2, "tough_tissue": 2, "detritus": 1},
        {"soft_tissue": 2, "tough_tissue": 2},
    ],
    atp=[2, 2, 2],
    atp_capacity=[4, 4, 4],
    bulk={"mineral": 3, "inorganic_carbon": 3},
)
```

The convenience web has exactly three rows. A shared ecological runtime should
use `enzyme_rows()` to assemble all colony rows, retain its own row assignments,
and give each living instance its own pools and developmental state.

## Related offspring

```python
child = genome.offspring(seed=37, mutation_scale=0.08)
child_value = child.to_value()
child_phenotype = child.compile()
```

The operator uses NumPy PCG64 and a fixed field order. It perturbs allocation in
log space and projects it back to a simplex with a 0.05 minimum per category.
It perturbs enzyme weights, housekeeping expression, growth multipliers, and
sparse neural gains within their declared bounds. The same parent, seed, and
scale produce the same child SHA-256. A positive mutation scale must produce a
different genome. This creates bounded related variants; it does not implement
selection, recombination, population genetics, or a species concept.

Ordinary `offspring()` retains the parent's active circuit identity. Structural
birth is explicit because it compiles a real bulk artifact:

```python
child, receipt = genome.structural_offspring(
    seed=37,
    parent_graph=graph,
    parent_ports=ports,
    parent_port_path="data/ports/retinal-v1-maps.npz",
    output_directory="/tank/chreatures/data/circuit-blueprints/lineage-g1-seed37",
    mutation_scale=0.08,
    selector_root=repository_root,
)
```

The method verifies that the supplied graph and ports are the active circuit
named by the parent genome. It deterministically materializes exact bounded
changes to newly cloned edges, compiles the graph and inherited ports, updates
the child's active source hashes, embeds the complete blueprint and compilation
identity, and only then computes the child genome hash. A returned structural
child can be the `parent_graph` of the next structural birth.

## Neural scope

Neural expression includes sparse gain at exact named input and readout ports.
Compilation verifies the active graph hash, port spec hash, complete port-bundle
hash, dimensions, and every named locus before emitting readonly float32 gain
arrays. `apply_input_gains` and `apply_readout_gains` perform the executable
elementwise operation while preserving the final channel dimension.

An optional inherited circuit template pins an exact selector artifact,
boundary-cloning rule, and port rule. Compiler limits permit one module copy,
at most 64 removals, at most 64 reweights, and count factors within 0.5..2.
Checked source configuration narrows this to one copy, eight removals, eight
reweights, and factors 0.75..1.25 per structural birth. The mutation operator
samples only edges that the new module copy creates. It never rewrites an
ancestral-to-ancestral measurement. The exact selected edits, rather than a
probabilistic placeholder, become the child's embedded `CircuitBlueprint`.

The measured root MaleCNS graph remains immutable. Every duplicate and edge
variation is labeled as a synthetic developmental hypothesis. Receptor
physiology and active inherited plasticity masks remain outside this mechanism.
Private lifetime neural plasticity, rates, optimizer state, and memories remain
outside the genotype and are recursively rejected if inserted anywhere in it.

## Validation scope

The focused constructor check loads the founder and seed-37 child, recompiles
both against the exact chemistry, grammar, graph, and port bundle, constructs
native `MetabolicWeb` and `GrowthSystem` instances for both, and steps their
chemistry. It verifies finite native ledgers, deterministic child identity,
bounded allocation, soma/gut/structure separation, readonly neural gains, and a
different real growth material request under identical growth seed and local
signals. This demonstrates the coupled mechanism. It does not establish that
either genotype is viable over a lifetime or fitter in an ecosystem.

The nonstructural founder genome identity is
`56d4beeefede48ff40972507870d48eb0d467a78406ff999e8b11f73cc5dcfca`;
the deterministic seed-37 child is
`b9b34b12b741be59ab90b7171efa01a9e4ddb7b06993453196a8c6483eeca19b`.
With growth seed 19 and the same four local receptor values, the founder asked
for `0.0565802650` biomass and `0.153578517` ATP. The child asked for
`0.0664748052` biomass and `0.180434997` ATP, a 17.49% larger material request.
Both native metabolic ledgers remained finite; placing their six emitted rows
in one shared `MetabolicWeb` produced distinct parent/child reaction extents.
A 64-seed sweep, including eight maximum-scale mutations, compiled and
constructed both native subsystems for every child while preserving the
allocation simplex and declared bounds.

The structural founder is
`aaf842da666ab4c0c75faedf9494144be2fa8a25e03ea62daeca29658b4e5cd9`.
Its seed-37 child compiled graph
`5ce567fb10d391b560dc12a00a4d1595a41859ebfabed6d3b4bff1c6307f1e97`;
the seed-40 grandchild compiled graph
`d4d927715b988e00249730dba3cb496210fc442b9fad5288cb2f0d8438331163`.
Each birth added one 3,627-neuron typed module and made one removal and one
reweight among 993,883 eligible cloned edges. The second graph has 172,376
neurons, 27,550,961 directed edges, and 129,767,490 synapses. Every inherited
parent edge matched generation one after filtering the added synthetic sources,
the metadata prefix was exact, body IDs were monotonic and unique, and the graph
manifest carried the measured root plus both direct blueprint transitions.

Current AMD sparse dynamics executed twelve 50 ms steps on this second graph.
All 7,254 synthetic neurons had a nonzero graded-input/control difference; the
maximum rate difference was 0.103316 and maximum readout difference was
0.631365. This proves executable multi-generation inheritance and dynamics. It
does not establish viability, fitness, or biological developmental fidelity.
Exact bulk hashes are recorded in
`data/development/circuit-lineage-two-generation.receipt.json`.
