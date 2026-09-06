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

The checked founder is `data/development/common-ancestor-v1.json`. A related
deterministic example child is
`data/development/common-ancestor-offspring-seed37-v1.json`. These are synthetic
research genotypes, not reconstructed natural ancestors.

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

## Neural scope

The current supported neural expression is sparse gain at exact named input and
readout ports. Compilation verifies the measured graph hash, port spec hash,
complete port-bundle hash, dimensions, and every named locus before emitting
readonly float32 gain arrays. `apply_input_gains` and `apply_readout_gains`
perform the executable elementwise operation while preserving the final channel
dimension.

The measured 165,122-neuron, 25,563,197-edge graph remains unchanged. There is
currently no inherited recurrent edge rewrite, neuron duplication, receptor
physiology, developmental rewiring, or active inherited plasticity mask. Adding
any of those requires a versioned runtime mechanism and an exact graph substrate;
the genome does not emit unused fields that suggest they already work. Private
lifetime neural plasticity and memories remain outside the genotype.

## Validation scope

The focused constructor check loads the founder and seed-37 child, recompiles
both against the exact chemistry, grammar, graph, and port bundle, constructs
native `MetabolicWeb` and `GrowthSystem` instances for both, and steps their
chemistry. It verifies finite native ledgers, deterministic child identity,
bounded allocation, soma/gut/structure separation, readonly neural gains, and a
different real growth material request under identical growth seed and local
signals. This demonstrates the coupled mechanism. It does not establish that
either genotype is viable over a lifetime or fitter in an ecosystem.

The founder genome identity is
`c4fdc0e72d0cc59292d9c0855daefa186c18db5793f2353ad02aeeab737dfc23`;
the deterministic seed-37 child is
`400b385034e94f808aae8f0c18b056a52e6f819fa3c82033cd5f187673d9eb5f`.
With growth seed 19 and the same four local receptor values, the founder asked
for `0.0565802650` biomass and `0.153578517` ATP. The child asked for
`0.0664748052` biomass and `0.180434997` ATP, a 17.49% larger material request.
Both native metabolic ledgers remained finite; placing their six emitted rows
in one shared `MetabolicWeb` produced distinct parent/child reaction extents.
A 64-seed sweep, including eight maximum-scale mutations, compiled and
constructed both native subsystems for every child while preserving the
allocation simplex and declared bounds.
