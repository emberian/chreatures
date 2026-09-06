# Transactional developmental growth

`chreatures.growth.GrowthSystem` is a resource-gated parametric L-system for
slow physical development. Its immutable grammar is shared genotype-level data.
Each instance owns only its seed-derived genotype multipliers, active buds,
three-dimensional turtle frames, counters, clock, transaction state, and private
random stream.

`GrowthKernel` is part of the required `_world_kernels` extension. Rebuild it
with `.venv/bin/python native/world-kernels/build_extension.py --output-dir .`
after adding the module.

The kernel does not choose a goal or query world objects. At a developmental
boundary, environment machinery samples four bounded values at each current bud:
local light, nutrient availability, structural support, and local competition.
Each rule declares minimums and weights for those signals. Competition attenuates
local vigor, so nearby shading and crowding change segment length and leaf area
through physical sensing. A root rule can give light zero weight; this is a
shared developmental parameter rather than an organism behavior policy.
The adapter derives support from local contact/structure evidence and
competition from local occupancy or attenuation. It must not substitute entity
names, caregiver identity, or a target waypoint for those measurements.

A production rewrites one eligible bud into a capsule segment, an optional leaf,
and zero or more successor buds. Successors carry turtle frames and declare a
symbol, branch angle, azimuth, generation-dependent phase, scale, and activation
probability. Segment length, branch angle, and leaf area include bounded
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

## Cadence and boundedness

The sample grammar waits four seconds before its first event and six seconds
between accepted batches. A batch has at most 24 new shapes, each rule at most
16 successors, and the instance at most 192 live buds. Engine limits cap a
grammar at 64 rules, 16,384 buds, and 4,096 shapes per batch. A failed or
underfunded event does not schedule another event or consume randomness.

## Persistence and identity

The normalized grammar is encoded with sorted JSON and hashed with SHA-256 in
both Python and native code. A snapshot records that hash, seed, exact xorshift
state, seed-derived genotype, bud frames, counters, generation, clock, next due
time, and any pending proposal inputs and result. Restore requires the same
grammar and replays a pending proposal to verify it exactly before accepting the
checkpoint.

The accepted structure itself lives in the ordinary `PhysicsWorld` snapshot.
Growth snapshots do not duplicate physical shapes, and physical snapshots do not
contain private bud or RNG state; the joined habitat checkpoint must store both.

## Focused physical check

With seed 19 and measured local signals `(light=.82, nutrient=.78,
support=.86, competition=.04)`, the first nursery proposal requested
`0.0565803` biomass: `0.0458109` branch and `0.0107694` root. Its exact request
was `0.0112614` soft tissue, `0.0453189` tough tissue, and `0.153579` ATP.
Rejecting and reproposing returned the identical token, geometry, request, and
RNG state. Pending and committed snapshots restored exactly and produced the
same next proposal.

The proposal was prepared and committed as one real MuJoCo topology batch. A
horizontal native ray that previously reached the east wall hit the new branch
at `0.285674 m`, demonstrating physical occlusion. The complete world snapshot
restored exactly after the append. This verifies the mechanism boundary; the
resource receipt in that isolated probe used the proposal's exact vector, while
the joined runtime supplies it through the common chemistry allocator.
