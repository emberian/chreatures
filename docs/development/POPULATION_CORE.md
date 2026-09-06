# Native population core

`native/population-core` is the deterministic population variation, selection, transfer-scheduling, and quality-diversity archive engine for the population wave. `chreatures.population.PopulationSearch` is its thin artifact/process boundary. Python supplies complete episode results and performs phenotype construction; it does not mutate or select candidates.

## Candidate boundary

A candidate is `chreatures-population-genome-v1`. Its canonical SHA-256 binds zero, one, or two genetic parent hashes; the variation seed, operator, recipe and mutated loci; graph, ports, base-controller and developmental-base identities; and every bounded scalar locus. Candidate data cannot contain state, memory, context, optimizer, neural rates/support, metabolic pools/ATP, RNG state, or history.

The current neutral recipe includes the existing twelve bounded articulated-body traits, twenty-eight somatic gain loci plus a three-pool anonymous secretion simplex, separate two-part metabolic and three-part developmental allocation simplexes, six neural-expression gains, recurrent/private-learning gains, and a fixed inherited population-adapter row, and per-channel categorical temperature offsets and output gains for the twelve-channel organism interface. Allocation mutation is native and renormalizes the declared simplex. Other loci use reflected Gaussian mutation at their bounds; crossover uses deterministic parent-locus choice with occasional arithmetic interpolation. This is a deliberately local kernel rather than arbitrary graph damage.

Neural scalar loci are inputs to the separate annotation-selector genotype compiler. The neural compiler seed is the recorded candidate variation seed XOR `0x4e455552414c5631`. A materialized phenotype references that compiled neural genotype receipt by SHA; the population genome does not duplicate per-neuron arrays. Structural graph variants carry their compiled circuit receipt and naturally group under a different compatibility identity.

`CandidateGenome.controller_adapter()` returns:

```text
candidate_sha256
loci_sha256
recurrent_gain                         scalar
learning_rate_gain                     scalar
action_gain                            float[12]
action_logit_temperature_offset        float[12]
policy_adapter_index                    integer (row 0 fixed by the baseline selection recipe)
population_adapter_bank_sha256/rank/count
```

Temperature offsets divide each categorical channel's logits by `exp(offset)`. Action gains multiply decoded values before signed/rectified bounds. The recurrent gain applies only to the private residual adapter, and the learning-rate gain applies only to private contextual learning. A fresh birth receives the immutable loci and fresh private state. Reproduction never copies a resident snapshot.

## Ask, transfer, and tell

`compose_population_birth(habitat, biosphere, candidates)` performs the cold physical join after regional generation has assigned five private chemistry rows. It copies both inputs; installs articulated traits; scales current somatic rates, capacities, activity costs and body-local phototrophy; writes the structural/gland/brood and anonymous secretion simplexes; adjusts founder carbon-fixation/digestion expression; and returns the two documents plus a content-addressed receipt binding physical, controller and neural phenotypes. It never performs recurring physiology or mutates a parent artifact.

A durable search state pins its full configuration, RNG, candidate table, environmental genomes, environment cursor, complete success/failure ledger, archive, and descriptor/quality versions. Registered environments bind zero to two environment parents, variation seed/recipe, topology, finite-resource, physical-profile identities, and epoch. The configuration pins the frozen probe-policy panel for the epoch. A changed descriptor, quality recipe, or probe panel starts a new search artifact rather than relabeling an existing archive.

`ask(n)` returns any requested positive population size up to 4096 as candidate/environment assignments. `ask_transfers(n)` schedules archived candidates into other registered environments and persists pending pairs so concurrent work is not duplicated. Environments rotate deterministically, and assignments are marked `direct-transfer`; parameter fine-tuning is outside this engine and may begin only after the coordinator records direct physical evaluation. Compatible hardware grouping uses the candidate's graph, ports, and controller identity and is not a fixed resident count.

`ask_transfers(n)` schedules already archived candidates into registered environments where that candidate has neither completed nor pending direct evaluation. The pending set is persisted, preventing duplicate simultaneous assignments.

`tell(result)` accepts a content-addressed terminal episode record. Its identity includes the anonymous life hash, evaluation seed, committed physical ticks, and trajectory SHA in addition to candidate, environment, status and metrics, so repeated lives remain distinct. Successful evaluations require positive committed ticks; a pre-birth failure may record zero ticks while still pinning its planned life and real failure/snapshot receipt. A success must contain every named physical descriptor and quality component. The native engine computes its bounded weighted quality and descriptor cell, retains up to the configured number of members per cell, and uses quality then candidate hash for deterministic ordering. A failure has null descriptor, cell, and quality, never enters an archive cell, and remains permanently in the evaluation ledger with its reason. Capacity exhaustion is therefore evidence rather than an implicit eviction.

Typical host use:

```python
search = PopulationSearch.initialize("campaign/search.json", config, seed=20260906)
search.register_environment(environment_genome)
assignments = search.ask(48)
# Run complete physical lives elsewhere.
decision = search.tell(terminal_episode_result)
search.validate()
```

Set `CHREATURES_POPULATION_CORE` to the compiled release CLI for campaigns. Without it, the development wrapper invokes Cargo against the crate manifest. State writes are temporary-file replacements. A campaign coordinator must serialize calls to a shared state path.

## Executed check

The focused native check compiled the crate, initialized two identical states, registered the same content-addressed environment, and restored one before `ask(6)`. Both states emitted byte-identical candidate identities and controller adapters. A complete success entered one archive cell; the archived candidate was then scheduled into a second environment, and a second physical-life identity in the first environment remained a distinct retained evaluation. The success had native quality `-0.098`; a `capacity-exhausted` failure was retained with null quality and did not enter the archive. The resulting state validated after both tells. This checks deterministic state continuation and terminal retention, not population fitness or useful evolved behavior.

The environment owner also exercised the cold composer on a native-generated eight-resident regional habitat. It returned eight physical bodies, eight mobile records, eight phenotype receipts, and preserved each assigned five-row block beginning at rows 24–28 and 29–33. The joined birth receipt was `9012e4e831344fbf59c5502940e76868a83f362e9c5d82163c83c321188534f4`. This validates the current generated schema join; it does not advance those founders or evaluate their fitness.
