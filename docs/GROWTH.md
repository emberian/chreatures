# Transactional developmental growth

`chreatures.growth.GrowthSystem` is a resource-gated parametric L-system for
slow physical development. Its immutable grammar is shared genotype-level data.
Each instance owns only its seed-derived genotype multipliers, active buds,
three-dimensional turtle frames, parent-part ancestry, path resistance, counters,
clock, transaction state, and private random stream.

`GrowthKernel` is part of the required `_world_kernels` extension. Rebuild it
with `.venv/bin/python native/world-kernels/build_extension.py --output-dir .`
after adding the module.

The kernel does not choose a goal or query world objects. At a developmental
boundary, the light kernel and native developmental surface kernel sample each
current bud against live MuJoCo geometry. Local values include light, nutrient
availability, structural support, crowding distance, the closest supported
static-primitive direction, and gravity in the colony frame. Geometry belonging
to the same colony is excluded from the external attachment query. Meshes and
height fields yield no attachment proof and retain authoritative topology
validation.

Each rule declares bounded guidance weights and a surface reach. These combine
the bud frame, gravity, free-space repulsion, and surface attraction before
geometry is proposed. A segment that reaches a supported static surface
terminates one radius away and records the stable target geom and world contact
point. Removing that target invalidates the reference. No entity name or
waypoint enters the native developmental decision.

Competition is the normalized distance to the nearest eligible external static
surface and is recomputed from live geometry at every due boundary. Support is
one for an initial scaffold bud; descendants use the remaining-to-initial tissue
fraction of their immediate parent part. Losing a recorded external attachment
halves that local support. These are explicit engineered proxies for clearance
and material continuity, not measurements of stress, vascular pressure, or
whole-organ physiology.

A production rewrites one eligible bud into a capsule segment, an optional leaf,
and zero or more successor buds. Successors carry turtle frames and declare a
symbol, branch angle, azimuth, generation-dependent phase, scale, activation
probability, and four local response weights. Those weights modulate successor
probability from actual light, transported nutrient, support, and competition,
so the same inherited grammar can differentiate locally. Segment length, branch
angle, and leaf area include bounded
seed-derived genotype variation and private developmental variation. The sample
grammar has shoot, lateral, and root symbols, but the mechanism does not contain
named plant, vine, coral, bridge, or chamber behavior.

## Exact material request

Geometry determines requested biomass before anything mutates. Capsule biomass
uses the volume represented by the physical capsule:

```text
stem_or_root_mass = density * (pi*r^2*length + 4*pi*r^3/3)
leaf_mass = leaf_area * areal_density
```

The grammar carries a composition vector and ATP cost for each geometry kind.
`data/growth/nursery-plant.json` requests `soft_tissue` and `tough_tissue`:
branches are mostly tough tissue, leaves are mostly soft tissue, and roots have
their own mixture. A proposal reports exact branch, root, and leaf mass, the
resulting resource vector, and ATP. These are allocation requests; the growth
engine cannot mint or consume material itself. The metabolism adapter can move
that composition from a body compartment to an allocated-structure compartment,
retaining carbon, nitrogen, and chemical energy bookkeeping.

Every committed segment adds `length / (conductivity * pi * radius^2)` to its
descendants' path resistance. A bud's nutrient input is attenuated by its
inherited path resistance and the rule's half-resistance, and its cadence-level
biomass is bounded by the same path conductance. This is an engineered transport
law with declared synthetic units. It creates no chemical pool and moves no
inventory; the exact body-to-structure transfer remains the material commit.

## Transaction boundary

```python
from chreatures.growth import GrowthSystem

growth = GrowthSystem("data/growth/nursery-plant.json", seed=19)
growth.elapse(4.0)
receptors = growth.buds()
local_signals = sample_environment_at(receptors)
proposal = growth.propose(local_signals, structural_budget=0.2)
```

`structural_budget` is supplied by the real allocator and bounds the complete
candidate. The native kernel evaluates candidates against copies of bud and RNG
state. If no bud is locally viable or the budget cannot fund a complete rewrite,
it returns `None`.

For a nonempty proposal, use this order:

1. Convert its geometry to one physical batch and call
   `world.prepare_topology_batch(operations)`. This fully validates and compiles
   the candidate world without mutation.
2. Reserve or debit the exact resource vector and ATP in a metabolic transaction.
3. Commit the prepared physical transaction.
4. Call `growth.commit(...)` with the exact accepted resource receipt and
   `physical_committed=True`.

The growth commit rejects a different token, any resource float that differs
from the requested bit pattern, a different ATP amount, or an uncommitted
physical transaction. If resource reservation or physical preparation fails,
`growth.reject(token)` discards the proposal. Proposal plus rejection leaves
buds, counters, and RNG bit-identical. Developmental time cannot advance while a
proposal is pending.

The surrounding adapter must make metabolic reservation reversible if the
already-prepared physical commit fails. Growth validates both receipts but does
not reach into either subsystem to implement a cross-engine rollback.

## Batched physical topology

Growth geometry is local to a shared structural frame. It contains no resident
ID or world-object query. `physical_operations` groups the complete accepted
candidate for the generic topology API:

```python
operations = growth.physical_operations(proposal, {
    "branch": "nursery-plant-branches",
    "root": "nursery-plant-roots",
    "leaf": "nursery-plant-leaves",
})
transaction = world.prepare_topology_batch(operations)
```

Segments become existing-schema capsules with `fromto` endpoints. Leaves become
oriented ellipsoids. One development event appends all shapes in one model
transaction, avoiding a MuJoCo rebuild per bud. The three target entities share
a physical origin but retain independent materials. The seed templates and
bindings in `data/growth/nursery-plant-physics.json` place a sample on an open
terrarium lane.

Those structural entity IDs belong to the habitat, not the organism. Appended
capsules and leaves remain collision and ray-occlusion geometry if the organism
that allocated them is later removed. The same primitive can therefore support
persistent coral-like substrate, overlapping branch spans, sheltered voids, and
routes that change over generations. A grammar must develop such geometry from
local signals and turtle productions; cognition never receives a bridge label,
waypoint, structure ID, or grammar symbol.

## Resolution and terminal growth

The current grammar schema is version 4. Every segment declares its radius
scale exponent and minimum length-to-diameter aspect, while
`resolution.minimum_feature_size` sets the smallest radius or half-extent that
can become physical geometry. A successor outside either boundary is a terminal
outcome and is not stored as another bud. A sampled leaf is emitted only when
all three half-extents meet the same resolution. No geometry or biomass is
clamped: omitted leaves are excluded from requested biomass, resources, and
ATP. Earlier grammar schemas are not imported by the current process.

Every stored bud therefore produces a valid capsule radius and remains valid
under snapshot restore. Restore accepts any finite positive scale satisfying
that physical invariant instead of imposing an unrelated numeric scale floor.
This is a declared simulation resolution boundary, not evidence of adaptive
growth or a generation-specific phenotype.

## Cadence and boundedness

The sample grammar waits four seconds before its first event and six seconds
between accepted batches. A batch has at most 24 new shapes, each rule at most
16 successors, and the instance at most 192 live buds. Engine limits cap a
grammar at 64 rules, 16,384 buds, and 4,096 shapes per batch. A failed or
underfunded event does not schedule another event or consume randomness.
`is_due` reads the native clock so callers can avoid sampling bud-local rays
between developmental events. `capacity()` reports live, maximum, and remaining
bud slots. `proposal_metrics()` and each nonempty proposal report resolution
terminals plus bud-capacity, structural-budget, and shape-limit rejections from
the latest due evaluation. These counters are diagnostics; rejected candidates
do not consume the private random stream. A colony can terminate with zero live
buds at the declared resolution instead of repeatedly proposing an invalid
collider.

This grammar belongs to an environmental colony and can vary through inherited
regional habitat ancestry. The current regional genome mutates bounded multipliers
for guidance, conductivity, half-resistance, and all four successor response
weights, and records them in the environment lineage. Mobile residents use
separate somatic allocation and body genotypes. The current mechanism does not
grow new resident limbs or copy colony genes into offspring.

## Persistence and identity

The normalized grammar is encoded with sorted JSON and hashed with SHA-256 in
both Python and native code. A snapshot records that hash, seed, exact xorshift
state, seed-derived genotype, bud frames, parent-part identities, path resistance,
counters, generation, clock, next due time, and any pending proposal inputs and
result. Restore requires the same
grammar and replays a pending proposal to verify it exactly before accepting the
checkpoint.

The accepted structure itself lives in the ordinary `PhysicsWorld` snapshot.
Growth snapshots do not duplicate physical shapes, and physical snapshots do not
contain private bud or RNG state; the joined habitat checkpoint must store both.

## Focused joined check

The final isolated native build
`f941aebd5a227937b3d5f50451e25ce890c1b3afb78b584aba7d0b5ca2408d3d`
ran one generated `terraced-delta` habitat with eight mobile residents and all
twelve environmental colonies for 90 physical and biosphere ticks (4.5 model
seconds). Its pinned source inputs were Living Reef
`7313420e6f8b507bf45b8ab2ec733064033af127897a3bab135a964fb519fe0e`,
regional configuration
`a7beb842b4027215ed6eb51373d16b45d3a965e75546010180f074d53773f41a`,
and resident bundle
`cfca86973dc039b3bffb4fac6b71e1d0eb92b34f411a7a57fc4f818f8b1e00a6`.

The world committed 32 funded parts in twelve growth events, including twelve
stable surface attachments and eight parts with structural parents. The first
event attached `reef-01:root-1` to the exact hashed shape on
`region-platform-05` and debited 0.0040683102 soft tissue, 0.0078973081 tough
tissue, and 0.0323071696 ATP for 0.0119656184 synthetic biomass. Physical packet
transactions and growth rebuilt the MuJoCo model through revision 9.

The seven-pool chemistry produced fermentate from finite founder material:
whole-world fermentate rose from zero to 0.0214634202 pool units. Action-funded
release committed 0.0005297642 fermentate units into physical packets during the
interval. The first such packet contained 0.0000228382 fermentate units and a
total element-weighted mass of 0.0001982683. All 54 exchange evidence events had
positive committed quantities; blocked or zero transfers emitted no evidence.
This proves production and physical availability for cross-feeding. The short
run did not demonstrate uptake by a second organism, so it is not evidence of a
completed donor-to-recipient exchange.

A bounded inherited environment mutation changed all twelve compiled colony
grammars and recorded both the parent genome and parent environment record. The
final world and biosphere restored bit-exactly. Their canonical snapshot hashes
were `75df818666ec019213b941870f728bf2fb1ae5dc67617f9ba6bcca407fd2e0ad`
and `02d9abccd456226b56670bf4b0e14e3a9ef3557b2e2a344a1695d6a238d523cf`.

This verifies native local cue evaluation, chemical funding, physical geometry
commitment, attachment and ancestry persistence, finite cross-feeding precursor
release, inherited habitat variation, and joined restoration in one actual
world. It does not establish evolutionary benefit, botanical fidelity, a
completed feeding interaction, or growth of mobile anatomy.
