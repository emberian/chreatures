# Conserved physical exchange among residents, colonies, and roots

Fresh recycling-reef worlds connect mobile excretion and root acquisition through
finite physical deposits in the same chemical web. A deposit can subsequently be
manipulated, ingested through a mouth, or partially depleted by contacting roots.
The scientific claim is material transport under supplied physiological laws;
learned gardening, waste handling and a stable food web are not established.

This document records the retired recycling-reef assay. Its exact habitat
builder is preserved at
[`867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa`](https://github.com/emberian/chreatures/blob/867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa/scripts/build_chemical_habitat.py).
Current worlds consume the authored birth-v4 Living Reef data directly and do
not migrate this builder's birth-v2/v3 output.

Exchange v2 also closes the photosynthesis-to-food loop. A material emitter
names a non-structural donor compartment, an existing physical attachment and
local offset, release rates and protected reserve floors per chemical pool, an
interval, packet mass bounds, and a finite set of dormant slots. At a due
boundary it can transfer only chemistry already present above every floor.
`MaterialObjects.deposit_batch` performs the native donor debit, receiver
credit, conservation checks, and topology activation. A newly visible packet
begins with exactly the moved chemistry; there is no spawn endowment or second
resource store.

## One inventory, spatial consequences

`EcologicalExchange` owns configuration, interval phases and cumulative transfer
receipts. `MetabolicWeb` owns every resource quantity. `MaterialObjects` owns the
mapping from reusable chemical compartments to actual physical geometry.

At a configured interval, each mobile outlet requests a fraction of its current
gut and body pools, using first-order transport rates gated by funded activity.
The request is bounded by a minimum and maximum deposit mass. Its position is an
offset in the body's complete local orientation, scaled by body radius. This is
an engineered peristaltic law, independent of action preferences. Inorganic carbon
is carried as synthetic material; this is not a model of respiratory gas physics.

The supplied world has 24 initially empty shared deposit slots. A successful
transfer instantiates a free physical object. There is no resource grant at
spawn. Geometry changes at declared content boundaries, and a fully depleted
slot can be used again at a new position. If no slot is free, material stays in
the resident and a capacity event is recorded. Neither overflow nor tiny residue
is silently deleted. The finite allocation is a current simulation capacity,
not a biological population law.

The first v2 profile retains those 24 slots solely for mobile egestion and adds
four explicitly reserved exudate slots for each of three colonies. Reservations
are disjoint across emitters and from mobile slots, so mobile-first update order
cannot consume colony packet capacity. Each emitter persists a round-robin
cursor within its own reservation. This is a declared simulation resource
allocation, not an identity preference or behavior policy. All paths still use
the same material transfer and physical-object mechanism.

An emitter's due phase saturates at one configured interval while material is
below the packet minimum, capacity is full, or its physical attachment is
unavailable. It cannot accumulate an arbitrarily large deferred release right.
Missing attachments retain donor material and increment a separate persisted
counter instead of raising inside a world tick.

If an emitter debits a mobile body or gut compartment, exchange immediately
resynchronizes that body's normalized physiology after the successful material
batch. Colony donors do not require this extra synchronization.

The supplied colony emitter draws reserve from its existing body compartment at
`0.006` pool units/s every two seconds, protects 42 reserve units in the normal
founder profile, and bounds packets to 0.004–0.03 mass units. It attaches beside
the colony's authored branch transform. Photosynthesis, respiration, growth,
and secretion therefore compete for one native inventory. The controller sees
only ordinary physical light, odor, contact, and ingestion consequences; no
emitter identity, pool ledger, object label, or world position enters policy
input.

The earlier exchange-v1 migration behavior belongs to the archived assay source
above. The current chemical training profile accepts authored birth-v4 inputs
and biosphere-v5 snapshots only.

Root acquisition requires a MuJoCo contact between a material object and a
resource-funded root part. Founder attachment placeholders do not acquire
anything. Contact distances above 10 micrometres are excluded. The configured
flux scales with the capsule's physical surface area and remaining soft/tough
tissue fraction; turnover into detritus removes conducting capacity. Multiple
contacts with one part do not multiply its surface. A root contacting several
objects shares its budget among them, and native batch withdrawal resolves
competition for scarce donor pools. Per-colony resource capacities limit intake.

This is passive synthetic root transport. Uptake geometry uses a whole contacted
root part's surface, not a resolved microscopic contact area. Deposits have no
independent microbial metabolism. Body inertia does not yet track all acquired
and expelled chemical mass, and deposit mass changes remain discretized. Nutrient
diffusion through soil, complete recycling and embodied reproduction remain open.

## Executed coupled experiment

The executed probe source is archived at
[`scripts/probe_ecological_exchange.py` from `867cdb8`](https://github.com/emberian/chreatures/blob/867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa/scripts/probe_ecological_exchange.py).

Three mobile residents generated physical deposits from their private chemistry
under supplied quiet actions. In a disposable research branch, one resulting
deposit was positioned against a constructed root; a matched branch placed it
0.8 metres away. After one physical/chemical tick, the contacting root acquired
`0.0004548671064071331` mineral and `0.0006823006596106993` inorganic-carbon units.
The separated control acquired zero. This position intervention is declared;
the probe does not claim the residents learned to deliver nutrients.

Both branches continued exactly after whole-world restore. Element and chemical
energy accounting residuals remained below `1e-10` abstract units. The raw report
at `runs/ecological-exchange-v1/report.json` includes source hashes and the
individual egestion vectors. The separate `probe_dormant_materials.py` exercises
scarcity, empty-slot reuse, changing position, atomic rejection and replay.

## Executed colony exudation assay

The executed probe source is archived at
[`scripts/probe_colony_exudation.py` from `867cdb8`](https://github.com/emberian/chreatures/blob/867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa/scripts/probe_colony_exudation.py).

The causal source assay sets every colony founder reserve to zero. After 20
model-seconds, the daylight branch had captured `1.7501939` photon-energy units
and emitted `0.1359373` newly produced reserve into twelve actual dormant-slot
packets. An otherwise matched near-dark branch captured `1.94e-13` and emitted
zero. A daylight branch with emitters disabled captured `1.7501955`, emitted
zero, and retained produced reserve in colony bodies. This separates secretion
of founder stock from additional reserve funded by the native light-driven
reaction network.

Whole physics, metabolism, material objects, emitter clocks/cursors, and
chemistry restored and continued exactly. Element and stored-energy residuals
remained below `1e-10`. In a separate controlled contact intervention, the
mouth of resident `mica` contacted one emitted packet for two ticks and acquired
`0.008` mass units; packet reserve fell from `0.012` to `0.004`, while gut
reserve rose from zero to `0.00450014` after simultaneous native digestion.
Positioning the packet at the mouth is an engineered assay intervention, not
learned foraging or evidence of autonomous ecological regulation. The receipt
is `runs/colony-exudation-v1/report.json`; its compact tracked receipt is
`data/ecology/colony-exudation-v1.receipt.json`.
