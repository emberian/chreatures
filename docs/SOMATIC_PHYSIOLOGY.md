# Mobile residents in the common chemistry

`SomaticPhysiology` connects each new resident's physical body to two private
rows of the Biosphere's native reaction system: body reserves/ATP and gut
material/enzymes. These are the same pool and reaction definitions used by
constructed colony tissue and finite physical packets. Founder material is
explicit. Normalized `energy` and `gut` are sensory readouts of that chemistry,
not separate editable stores. Fatigue is a bounded activity/recovery dynamic.

Before physics, maintenance and active-output budgets draw from actual body
ATP. Available funding limits articulated/crawler forces, posture, grip and
signal amplitude. Zero funding supplies zero active output. The activation law
is a declared engineering model based on requested activity; it is not a fitted
muscle model or a guarantee of metabolic-to-mechanical energy conservation.
Positive applied mechanical work is measured separately. Gaze currently changes
a retinal parameter without a modeled head joint.

After physics, ingestion requires both an eating action and a measured contact
within the resident's body-local mouth neighborhood. Merely touching material
with another part of the body does not grant access. Bite capacity accumulates
from funded oral activity and is bounded by the supplied maximum bite and
remaining gut capacity. Small developed parts can be removed into a gut in a
single physical/material transaction. Free packets permit partial acquisition;
competing demands are limited simultaneously by the native transfer operator.
Objects shrink at declared content boundaries and disappear when empty.

The shared reactor then advances once. Body ATP can fund digestive reactions,
and bounded transport moves actual liberated reserves from gut to body.
`nutrition` evidence records absorbed reserve, rather than awarding energy for
contact. Private enzyme expression changes what can be digested and how quickly.
Undigested material and mineral remain in the gut; excretion and a complete
spatial recycling system are outstanding mechanisms. Body/inertia changes do
not yet account for every acquired or lost material unit.

## Historical assay

The executed contact probe depended on the retired birth-v2/v3 habitat builder.
Its exact source is preserved as
[`scripts/probe_somatic_contact.py` at `867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa`](https://github.com/emberian/chreatures/blob/867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa/scripts/probe_somatic_contact.py).

The probe creates three independently initialized physical experiments with a
finite soft-tissue packet and an initially empty gut. Quiet supplied actions and
founding placement establish the controls; no neural policy runs in this probe.
The mouth-contact case ingested `0.008` material-equivalent units and absorbed
`0.005613213898396849` reserve units. Side-contact and no-eating controls both
reported zero ingestion and absorption despite actual physical contacts. All
three restored and continued exactly for four subsequent steps. Chemical
accounting residuals were below `1e-10` in the model's abstract units.

A separate joined three-resident run restored physical state, Biosphere state
(including private physiology and material bindings) and the odor field exactly.
The fixed six-leg interface and the normalized neural channel ranges are retained;
this changes physiological meaning, so inherited motor policies remain transfer
baselines until evaluated under these new dynamics.

Historical public birth artifacts:

- [`chemical-reef.json`](../data/habitats/chemical-reef.json): inherited body
  variation, irregular terrain and four finite grippable chemical packets.
- [`chemical-reef-v1.json`](../data/biosphere/chemical-reef-v1.json): three growing
  colonies, private mobile compartments and finite packet rows under one program.
- [`build_chemical_habitat.py` at `867cdb8`](https://github.com/emberian/chreatures/blob/867cdb83a1eba836a4d9f4898f7be9f83f8ab5fa/scripts/build_chemical_habitat.py):
  the retired founding construction. Current worlds use authored birth-v4 data.

The chemistry-derived surface colors reach ordinary retinal rays. Its odor
coefficients emit sensory tracers through the existing transport field. Those
tracers are not material nutrient pools and do not close an elemental food web.
The runtime reports this distinction. Snapshot restore verifies normalized
body readouts against the chemistry, restores private bite capacity and payment
history, and rebinds the transient physics owner. Mid-step snapshots are rejected.

The v2 physiological snapshot also records `mouth_material_contacts`: unique
material entities contacted inside the mouth radius with a positive ingestion
request during each physical tick. This is separate from actual `ingested_mass`
and subsequent absorption. On import of an older snapshot, the counter starts
at zero with an explicit `contacts_since` model-time boundary; missing historical
observations are not interpreted as evidence of no contacts.
