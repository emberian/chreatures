# Ecological interdependence

The interdependence chemistry is a future-world configuration in
`data/metabolism/interdependence-chemistry-v1.json`. Its quantities are
synthetic carbon-equivalent and mineral-equivalent bookkeeping, not calibrated
molecular biochemistry.

`fermentate` carries one carbon equivalent and 0.58 chemical-energy units. The
three added reactions are element and energy balanced:

| Reaction | Material transformation | ATP | Heat |
|---|---|---:|---:|
| reserve fermentation | reserve → fermentate | 0.28 | 0.14 |
| fermentate respiration | fermentate → inorganic carbon | 0.40 | 0.18 |
| detritus hydrolysis | detritus → 4 fermentate + mineral | 0.25 | 0.23 |

Expression is scarce. Each compartment has a maximum coefficient of `0.06`
per reaction and a total enzyme-expression budget of `0.10`. Existing founder
rows use up to `0.08105`, so acquiring a strong added pathway requires a real
allocation tradeoff. Genomes set rates and capacities; they never create
inventory.

Metabolites move between organisms as finite physical material packets. A
release withdraws exact pool quantities from a body or gut row, creates a
packet at the body-local outlet, and another organism must contact and ingest
that packet. Root acquisition likewise requires contact between a constructed
root capsule and a packet. Colony exudates are withdrawn above a protected
donor reserve floor. This permits phototroph → fermenter → fermentate consumer
and detritus → decomposer → fermentate consumer cycles without assigning roles
or targets to controllers.

The three transported scalar fields remain nonconserved sensory tracers. A
secretion pulse can advertise recent material exchange, but field intensity is
not edible mass and never enters the metabolic inventory.

Exchange v4 records each committed material movement with body/entity actors,
named pool quantities in `pool_quantity`, element-weighted mass in
`synthetic_element_sum`, chemical time, and a persisted exchange step index.
The native release kernel calculates all resident × compartment × pool
candidates in one call; Python retains physical placement and atomic material
transactions.

The composed regional B8 check ran 90 steps at 0.05 seconds. Fermentate rose
from zero to 0.0214634202 pool units, including 0.0005297642 committed into
physical packets; the first packet appeared at chemical time 0.25 seconds.
The same run committed 32 constructed parts across 12 attachments, emitted 54
nonzero exchange events, and reproduced both World and Biosphere snapshots
exactly after restore. The identities and raw-result hashes are recorded in
`data/ecology/interdependence-native-v1.receipt.json`.
