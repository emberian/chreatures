# Native chemistry and constructed habitat

`Biosphere` joins the native `MetabolicCohort`, resource-funded `GrowthSystem`
and transactional MuJoCo geometry. A colony has private body and allocated
structure compartments, a private developmental state, and per-part chemical
stocks. All colonies use one immutable reaction definition. The six material
pools conserve carbon-equivalent and mineral-equivalent quantities; chemical
energy, ATP, captured photons, heat and exported work have explicit ledgers.
These are synthetic units and constitutive laws, not calibrated plant biology.

## A developmental boundary

At a due growth boundary, local light rays and the colony’s own mineral pool
supply signals to its grammar. A proposal requests exact soft/tough tissue and
assembly work. The runtime prepares a single physical topology transaction,
checks the developmental receipts on unpublished state, transfers material and
pays work, commits geometry, then publishes the resulting developmental state.
Known preparation or funding failures preserve the prior state. Unexpected
failures propagate to the runtime’s failed-tick pause; the implementation does
not promise arbitrary distributed failure rollback.

Named physical state survives recompilation. Added and removed static geometry
also updates the chemical field’s rasterization, conservatively displacing
material covered by new solids. This transport rule does not model pressure
or momentum. Geometry below the transport grid’s resolution may have no field
effect. The growth grammar separately declares a 2 mm physical feature floor.

Tissue turnover updates the chemical stocks associated with each built part.
Dead material retains its scaffold until an explicit removal. `release_parts`
removes selected colliders and transfers their remaining material into a named
compartment. The same transfer machinery now supports acquisition when a mobile
mouth physically contacts a sufficiently small part with ingestion enabled.
Learned harvesting has not been established.

## Reproduce the non-neural experiment

Build the [native world kernels](../native/world-kernels/README.md), then use:

```console
.venv/bin/python scripts/probe_biosphere.py --seconds 180 --output runs/biosphere-reef
```

The script constructs supplied related founders, advances native chemistry,
physics and transport, checks same-runtime continuation, and transfers a leaf
into another compartment’s digestion. The public founder configuration is
[`reef-founders-v1.json`](../data/biosphere/reef-founders-v1.json); the lineage
records are [`reef-lineage-v1.json`](../data/development/reef-lineage-v1.json).
The numerical and geometry receipts from the 180-second run are summarized in
[ECOLOGICAL_COMMONS.md](ECOLOGICAL_COMMONS.md) and the public lab notebook.

The seed terrain contains three differently oriented attachment surfaces. The
founder enzyme allocation, initial material and photon flux are engineered
experimental settings. They have not been selected by evolution. Root growth
can enter supplied support geometry; excavation work and mechanical stress are
not modeled. Bud support currently follows attachment ancestry, and competition
acts through physical shading rather than a separate fitted competition model.

## Run beside mobile residents

Start a separate ordered neural service and use a fresh checkpoint:

```console
uv run chreatures --port 8772 --brain-url http://127.0.0.1:18772 \
  --checkpoint runs/reef-garden.json \
  --habitat data/habitats/reef-garden.json \
  --biosphere data/biosphere/reef-founders-v1.json \
  --motor-genome data/genomes/nursery-embodied-v2-step9695.npz \
  --personal-memory --personal-plasticity
```

The biosphere and legacy resource producer are mutually exclusive options.
Snapshots retain configuration identity, private chemistry, developmental RNG,
part stocks and geometry correspondence. Existing saved worlds retain their
own habitat/body specifications; current source changes do not reconstruct an
already living resident.

Existing saved reef residents retain their separate physiological system. Fresh
birth-v2 worlds can bind every mobile resident to private body/gut compartments
in the same web. `resident_physiology_coupled` then reports true. See
[SOMATIC_PHYSIOLOGY.md](SOMATIC_PHYSIOLOGY.md) for funded activation, physical
mouth contact, digestion and absorption, and [MATERIAL_OBJECTS.md](MATERIAL_OBJECTS.md)
for finite shared stores and geometry boundaries.

Fresh birth-v3 worlds can additionally enable [physical recycling](ECOLOGICAL_EXCHANGE.md):
mobile material leaves through body-local outlets into finite free deposits,
and constructed roots acquire configured resources under actual physical contact.
The native chemical ledger spans all these compartments. The current snapshot
format is v4. It explicitly preserves developmental part iteration order because
floating-point tissue and illumination reductions depend on that order. Earlier
v1/v2 data import with exchange disabled; v3 imports retain the order available
in their saved document. Those earlier artifacts did not record the order before
canonical JSON serialization, so exact recovery of that lost order is not claimed.

Embodied reproduction, autonomous predation, full mass-to-inertia coupling
and conservation of every physical object in the world remain unimplemented.
`whole_food_web` remains false. Odor transport currently carries sensory tracers,
separate from the conserved material pools. The older birth/snapshot formats
have a one-way data import into the current owner, preserving their uncoupled
physiological semantics. Existing live processes have not been upgraded.
