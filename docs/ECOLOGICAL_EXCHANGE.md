# Physical recycling between mobile residents and constructed roots

Fresh recycling-reef worlds connect mobile excretion and root acquisition through
finite physical deposits in the same chemical web. A deposit can subsequently be
manipulated, ingested through a mouth, or partially depleted by contacting roots.
The scientific claim is material transport under supplied physiological laws;
learned gardening, waste handling and a stable food web are not established.

Build a new habitat and birth configuration:

```sh
uv run python scripts/build_chemical_habitat.py --recycling --output runs/recycling-reef-birth
```

Use these `habitat.json` and `biosphere.json` files with a fresh 3D world and its own
ordered neural service. Existing checkpoints import with exchange disabled;
their original mobile physiology and material inventories are preserved.

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

```sh
uv run python scripts/probe_ecological_exchange.py
```

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
