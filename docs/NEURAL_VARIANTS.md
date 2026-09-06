# Heritable neural variants

`chreatures.neural_genotype` compiles immutable candidate parameters into the
full canonical MaleCNS circuit. It never stores a living resident's rates,
support, adaptation, memories, objective receipts, optimizer, or RNG state.
Candidates that share the same graph, neural ports, controller interface, and
structural blueprint have one compatibility-group hash and may occupy columns
of one hardware cohort. A structural variant has a different graph hash and is
scheduled separately.

The current recipe is
[`neural-variant-canonical-v1.json`](../data/ports/neural-variant-canonical-v1.json),
SHA-256 identity
`352b4c0cd9dacb5d84914a43651268b44073e4df0b8d9a33a1b44db64ed8bd46`.
It pins canonical MaleCNS graph `48ce8c8f…d625`, retinal-v2 port bundle
`933b871f…be53`, port spec `a3182cc5…302e`, 351 inputs, and 384 readouts.

## Measured and engineered parts

Group membership comes from exact curated MaleCNS annotation strings or from
the source records in retinal-v2. On the 165,122-neuron canonical graph the
recipe resolves these nonempty groups:

| Group | Neurons |
| --- | ---: |
| visual pathways | 103,268 |
| olfactory pathways | 3,783 |
| mechanosensory pathways | 5,756 |
| Kenyon cells | 4,064 |
| mushroom-body outputs | 97 |
| DAN annotated | 340 |
| central complex | 2,950 |
| ascending pathways | 2,391 |
| descending pathways | 1,330 |
| VNC intrinsic | 13,151 |
| motor/efferent | 925 |

The port groups resolve 320 retina, six odor, one contact, six contact-normal,
two touch, six linear-proprioceptive, six angular-proprioceptive, three sound,
and one shade inputs. Readouts retain the declared retinal-v2 domains: 160
visual, 96 mushroom-body, 80 navigation, and 48 efferent.

The annotations and source memberships are measured records. Every gain,
bound, overlap rule, mutation, recombination, and structural duplication is an
engineered hypothesis. In particular, `DAN` and transmitter annotations do not
establish receptor action or modulatory sign. These parameters do not imply an
anatomical contribution to behavior.

## Genotype and compilation

The population core supplies six bounded scalar loci:

```text
input_gain, readout_gain, excitability, recurrent_gain,
learning_rate_gain, modulator_gain
```

Compilation starts the whole phenotype from those six scalars. The recurrent
scalar is split as its square root across source and target gains so neutral
annotation factors produce the scalar once. Per-group alleles multiply that
base and are clipped only after all explicitly overlapping groups have been
applied. Group alleles are values in the genotype JSON, so descendants inherit
or recombine actual alleles. They are never regenerated from a candidate hash.

The initial population wave should use `mutation_scale=0.0`; all annotation
group factors then remain one and only the six population loci vary. Enabling
group variation later is an explicit recipe experiment. Its seed and every
resulting allele remain in the genotype. Up to two parents are supported.

```python
from chreatures.neural_genotype import (
    NeuralVariantRecipe,
    compile_population_phenotypes,
    batch_neural_phenotypes,
)

recipe = NeuralVariantRecipe.load(
    "data/ports/neural-variant-canonical-v1.json"
)
phenotypes = compile_population_phenotypes(
    candidates,
    recipe,
    graph,
    ports,
    port_bundle_sha256,
    base_controller_sha256,
)
arrays, phenotype_ids, group = batch_neural_phenotypes(phenotypes)
circuit.bind_neural_phenotypes(
    arrays,
    phenotype_sha256=phenotype_ids,
    compatibility_group=group,
)
```

The helper consumes each exact `candidate.neural_population_loci()` mapping. It
carries the candidate identity, campaign base-controller identity, immutable
population variation seed, six scalar loci, and
`neural_seed = variation_seed ^ 0x4e455552414c5631`. The external cold-birth
receipt records that mapping and separately binds the candidate, recipe,
compiled phenotype, and base controller hashes. The helper requires the pinned
canonical graph and ports, always sets annotation-group mutation to zero for
this population wave, deduplicates repeated immutable candidate identities,
and rejects an identity repeated with different neural birth inputs.

`NeuralPhenotype.save(path)` writes one allow-pickle-false NPZ containing a
canonical JSON metadata scalar and seven C-contiguous float32 arrays. Loading
checks the file receipt, all graph/port/controller identities, ordered port
names, shapes, bounds, per-array hashes, and phenotype hash.
The bounded full-graph persistence receipt is
[`neural-variant-persistence-check-v1.receipt.json`](../data/ports/neural-variant-persistence-check-v1.receipt.json).
It records an exact seven-array round trip, immutable loaded arrays, and
rejection of a mismatched base-controller identity. The sample artifact stays
on `/tank`; population births create their own content-addressed artifacts.

## Runtime equations

For resident `b`, input channel `c`, neuron `i`, source neuron `j`, and readout
`o`, the full graph remains sparse and unchanged:

```text
x'[c,b] = input_gain[c,b] * x[c,b]
q[i,b] = target_gain[i,b] * sum_j(
    W[i,j] * source_gain[j,b] * rate[j,b]
)
a[i,b] = 0.005
       + excitability[i,b] * (drive[i,b] + base_gain*q[i,b])
       - 0.10*adaptation[i,b]
y'[o,b] = readout_gain[o,b] * sum_i(R[o,i] * rate[i,b])
```

The two existing rate substeps, support update, adaptation update, timestep,
and all 25.56 million graph edges remain intact. Torch neuron-major,
resident-major, microbatch, serial-row Triton, and edge-tiled Triton paths use
the same equations. The backend uploads `[value,resident]` gain matrices once
when a cohort is bound; recurring steps contain no Python resident loop or
host-side dense gain copy.

`learning_rate_gain` and `modulator_gain` are identity-bound but inactive in the
generic circuit. A future explicit local plasticity owner may multiply its own
target-neuron delta by `learning_rate_gain`. A declared synthetic or measured
modulator path may multiply its target current by `modulator_gain`. Neither is
silently applied to fixed recurrence or adaptation.

Circuit metadata records the ordered phenotype hashes, compatibility group,
all array hashes, active parameters, and the two inactive parameters. Exported
state includes `neural_variant_state_identity`; restore first binds the same
ordered phenotype columns and then rejects a different state identity.

## Structural births

`materialize_population_structural_variant` in `circuit_blueprint.py` resolves
one exact named structural template from this authenticated recipe, then uses
the existing sparse CircuitBlueprint compiler. The current template is the
measured gamma1pedc KC/MBON11/PPL101 selector with a synthetic duplicate and
bounded edits only on cloned edges. Its resulting graph, derived ports,
blueprint, and manifest hashes enter `structural_identity`; they are separate
from parameter-only candidates and from the immutable canonical ancestor.

## Bounded full-graph checks

The compile receipt is
[`neural-variant-fullgraph-check-v1.receipt.json`](../data/ports/neural-variant-fullgraph-check-v1.receipt.json).
It resolved every group above and produced finite, immutable arrays of shapes
351, 384, and 165,122 on hbox without loading or changing a neural state.

The runtime receipt is
[`neural-variant-runtime-check-v1.receipt.json`](../data/ports/neural-variant-runtime-check-v1.receipt.json).
One three-resident full-graph step compared Torch CSR with edge-tiled Triton on
the RX 6750 XT. Readouts differed by at most `2.98e-8`, rates by `1.16e-7`, and
physiology by zero. Same-backend snapshot replay was exact, and reordered
phenotype identities were rejected. The recorded cold timings include Triton
compilation and are not a performance benchmark. This establishes mechanism
application and replay, not behavioral value.
