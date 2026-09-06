# Finite material objects

`MaterialObjects` binds ordinary MuJoCo entities to private compartments in the
same `MetabolicWeb` used by residents, growing structures, and environmental
chemistry. The web row is the complete inventory. The physical entity has no
second `amount` field, and a configured entity may not also have a `food` or
`scent` component.

This mechanism can represent a loose packet, a refillable common store, or
material left in a nursery without assigning an owner or intended use. A
controller receives ordinary contact, retinal colour, and local chemical cues.
Entity identifiers and compartment rows remain on the environment side of that
sensory boundary.

## Integration contract

Construct the adapter after the physical world and shared web:

```python
materials = MaterialObjects(world, biosphere, "data/materials/finite-packet-v1.json")
```

The second argument may be a `MetabolicWeb`, an object whose `web` property
returns one, or a callback returning one. Passing the owner or callback is
preferred because a failed larger transaction can replace its web while
rolling back. When a Biosphere owner is supplied, every integer under a
configuration key ending in `_row`, including
`biosphere.mobility.residents[*].body_row` and `gut_row`, is reserved. Material
rows must be separate. Code that passes a bare web is responsible for choosing
private rows.

For ingestion, use contact samples collected by `PhysicsWorld` during its
physiology phase:

```python
proposals = materials.acquisition_proposals(
    resident_id,
    gut_row,
    contact_samples,
    maximum_mass=bite_credit,
    receiver_capacity=free_gut_mass,
)
receipt = materials.commit(proposals[0])
```

Mass here is the sum of the chemistry's conserved elemental equivalents. The
adapter takes the smaller of bite credit and receiver capacity, then removes a
proportional slice of the contacted object's current mixture. Thus the
physiology gates the quantity while the source chemistry determines what is
acquired. `per_pool_limits={...}` is an alternative for environment machinery
that has already derived an exact pool request. Neither interface accepts a
remote object as evidence of access: acquisition proposals are made only for
entities in the supplied physical contact samples.

`prepare_withdraw(entity, receiver_row, resources)` and
`prepare_deposit(entity, donor_row, resources)` provide exact non-ingestion
transfers for bounded environmental processes. A proposal records both rows,
their relevant before-state, the geometry boundary, model revision, and a
content hash. `commit` rederives and compares all of it before mutation. The
native web transfer is checked for elemental and stored-energy conservation
before a precompiled physical topology transaction is committed. A failed
transfer or topology commit restores the native web state.

For several mouth contacts at one phase boundary, call
`withdraw_batch(requests)`. Each request contains `entity`, `receiver_row`, and
an exact `resources` mapping after physiology has proved access and enforced
aggregate gut capacity. The native batch computes each pool's scarcity factor
from one common pre-state. Competing requests therefore receive proportional
shares of the last material independent of request order. The adapter stages
the same batch on an exact web clone, compiles one topology transaction for all
resulting object boundaries, applies the authoritative native batch, and then
commits that physical transaction. The returned `moved_resources` is an M×K
list aligned with request order and the returned `pools` column names. An empty
request list is a cheap no-op.

Single prepared proposals still commit one object at a time. A boundary
crossing changes topology and invalidates other proposals prepared against the
prior model revision.

## Dormant deposit slots

An object may opt into reusable physical activation by adding a full
`dormant_template` entity to its existing material-object entry. At birth its
web row must be exactly empty and its entity id must be absent from the physical
world. The template must be a free MuJoCo entity with the same id, without
`food` or `scent` components. `remove_when_empty` must be true, and `capacities`
must explicitly name every pool in the shared chemistry, using zero for pools
the slot cannot hold. Construction compiles the dormant entity without
adopting it, so malformed geometry fails before the world starts.

```json
{
  "entity": "shared-packet-0",
  "row": 14,
  "capacities": {
    "mineral": 0.0,
    "inorganic_carbon": 0.0,
    "reserve": 0.3,
    "soft_tissue": 0.5,
    "tough_tissue": 0.0,
    "detritus": 0.2
  },
  "remove_when_empty": true,
  "dormant_template": {
    "id": "shared-packet-0",
    "mobility": "free",
    "material": "packet-0",
    "physical_material": "light",
    "position": [0.0, 0.0, 0.1],
    "shapes": [{"type": "sphere", "size": [0.08]}],
    "components": []
  }
}
```

The entry also includes the normal `content_weights`, `boundaries`, and
`surface` fields omitted above.

Authorized egestion or material release uses one atomic batch:

```python
receipt = materials.deposit_batch([
    {
        "entity": "shared-packet-0",
        "donor_row": gut_row,
        "resources": released_pools,
        "position": rear_contact_position,
    }
])
```

An inactive slot requires a finite position inside the habitat. All requests
targeting that slot in the same batch must give the same position. The position
is installed in its private base entity only when nonzero material actually
moves and the physical spawn commits. Later boundary replacements preserve the
free body's current MuJoCo pose. When the object empties, its geometry is
removed while its last base pose remains available for exact restore; a later
deposit may reactivate the slot at a different supplied position. Supplying a
position for an already active object is rejected before mutation, preventing
material deposit from becoming a teleport action.

Before paying chemistry into a dormant slot, the prepared topology transaction
checks its new geoms against the current poses and mutable geometry of the
developed world. A penetrating spawn is capacity-blocked in the batch receipt;
its requested chemistry remains in the donor row. This is a placement
admissibility check, not a force-free guarantee after ordinary dynamics begin.

For each slot and pool, `deposit_batch` divides the free capacity fairly among
simultaneous requests from the same pre-state. It then passes those limited
requests through native `transfer_batch`, which fairly handles scarcity shared
by donor rows. Structure rows and other material rows cannot be donors. The
receipt aligns its M×K arrays with request order and `pools`; it reports
`moved_resources`, total `blocked_resources`, and the portions blocked by
receiver capacity and donor scarcity. If all preallocated slots are active or
full, the rejected chemistry remains in its donor rows.

Direct `sync_geometry()` will not activate an inactive dormant slot after an
out-of-band web edit because there is no authorized physical position. Such a
deposit must go through `deposit_batch`.

## Physical boundaries and cues

Each object declares capacities and a descending list of content boundaries.
`content_weights` maps the current pool vector to a geometry-control scalar.
Crossing a boundary replaces the entity with a uniformly scaled copy of its
authored shapes. MuJoCo recomputes body mass and inertia from the new geometry.
When `remove_when_empty` is true and every pool is exactly empty, an existing
prepared topology transaction removes the entity. A refillable object instead
remains at its last declared scale.

Geometry updates are deliberately discrete. They avoid rebuilding MuJoCo for
every small bite while making major depletion physically visible and changing
what can be pushed or gripped. `sync_geometry()` applies the same derivation
after another authorized web operation changes a material row.

The `surface` coefficients derive RGB and three odor channels from public pool
amounts. `surface_cues()` includes an entity id so environment machinery can
place those signals, but policy code must receive only the resulting ray RGB
and local field samples. The coefficients reveal chemical surface state, not
an object category, owner, or prescribed value.

## Snapshots

Save the physical world, metabolic web, and material adapter together. Restore
the world and web first, followed by:

```python
materials = MaterialObjects.restore(world, biosphere, saved["materials"])
```

The material snapshot stores the immutable config, authored base geometry,
derived boundary state, and bounded diagnostics. It never stores pool amounts;
those occur once in the web's native snapshot. Restore rejects chemistry or
config identity changes, invalid ledgers, disagreement between the web-derived
boundary and saved state, and disagreement between that state and the actual
physical entity.

## Focused physical probe

`scripts/probe_material_objects.py` builds a three-row common-chemistry web and a
real free MuJoCo packet in contact with a resident. It partially transfers the
packet, checks that its body mass changes by `0.78³`, checkpoints all three
owners, then has two receivers simultaneously request the remainder. Both get
equal native scarcity allocations, and one topology batch removes the drained
packet in both continuations. The probe asserts exact joined continuation and
zero elemental and stored-energy residual.

`scripts/probe_dormant_materials.py` starts with an empty row and no packet
geometry. Two donors overfill the same dormant slot; both receive equal
capacity allocation and the untransferred amounts remain with them. The probe
then performs a partial withdrawal, removes the fully drained object, and
reactivates the same slot elsewhere. It rejects an attempted active-object
teleport atomically and replays the entire depletion and respawn continuation
from a joined physical/web/material checkpoint.

The sample's full and partial physical masses are approximately `0.0750631` and
`0.0356214` MuJoCo mass units. These values are physical consequences of the
sample geometry and its habitat density; they are not treated as chemical
inventory or a calibrated mapping from elemental equivalents to kilograms.

`probe_material_overhead.py` at
[`867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa`](https://github.com/emberian/chreatures/blob/867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa/scripts/probe_material_overhead.py)
profiles the unconditional material calls in the retired fast-articulated
recycling world. Its local research-reference branch emulates the earlier scans
on a disposable instance; it is not a runtime backend. The archived measurement
in `data/performance/material-overhead-v1.receipt.json` supports reduced recurring
material overhead and explicitly makes no whole-world speedup claim because
shared-host wall time was noisy and physical topology compilation still
dominates deposit events.
