# A common substrate for different ways of living

The target is an evolving ecology. The current native substrate joins common
chemistry, resource-funded development, constructed physical geometry and
chemical transport. Mobile residents can have different inherited bodies, but
their existing feeding and physiological system is not yet joined to the
colonies’ common chemistry. There is no developed population lifecycle yet.
More training cannot create missing mechanisms of reproduction or exchange.

The substrate combines implemented mechanisms with remaining lifecycle work:

- Conserved material pools and energetically constrained reactions.
- Acquisition surfaces, digestion, maintenance and allocation to tissue.
- Resource-gated developmental programs that build actual geometry.
- Reproduction by transferring material into offspring, with inherited rules.
- Limited senses and whatever neural/cognitive modules a body expresses.
- Local secretion, consumption and environmental modification that change
  opportunities for other organisms.

Producer, grazer, predator and decomposer describe measured acquisition and
conversion strategies. They are not labels the controller receives or branches
in an action script. Founder parameter sets will be engineered; a changed
descendant population is not automatically evidence that novel niches evolved.

## Common chemistry

`data/metabolism/common-chemistry.json` declares a deliberately synthetic
six-pool chemistry. It conserves two bookkeeping quantities, carbon-equivalent
and mineral-equivalent material. These are abstract units, not a calibrated
CHON molecular model. Pool chemical energy plus free metabolic energy is
accounted against captured photons, exported work and dissipated heat.

The same reaction program can support carbon fixation, respiration, structural
growth and digestion. Organisms differ in enzyme expression, uptake surfaces,
allocation and the structures they can maintain. Digestion has activation costs;
chemical similarity can make different organisms possible food without granting
a controller access to their identity. Both resource scarcity and the costs of
acquiring or maintaining a capability must matter.

Reaction competition is simultaneous: proposed fluxes see the beginning state,
then each resource's total demand bounds reactions consuming that resource.
Mineral scarcity must not turn off respiration merely because growth also lacks
minerals. Transfer and birth move existing quantities; they do not initialize
new material from a fresh energy meter.

The native reactor runs in `native/world-kernels`. Its ledger covers the defined
chemical system, including development’s exported assembly work. Joining it to mobile actuator work, physical
mass, spatial transport and the complete food web requires explicit adapters;
we will not claim whole-world conservation before those transfers are joined.

## Development makes environments

A parametric rewriting grammar supplies buds, branch directions and acquisition
surfaces. Its rules request material and produce geometry proposals. A bud grows
only after the metabolic allocation and physical topology transaction succeed.
Local light and resource signals can affect which rules become active.

This lets inheritance concern a process rather than a completed mesh. Growth
can create shade, supports, obstacles, resource patches and shelters; those
changes alter the developmental opportunities of later organisms. The initial
grammar and founders are supplied mechanisms, not an assertion that plant
development has already emerged.

Physical topology updates are batched at developmental boundaries. Living
MuJoCo state is retained by named joints; failed updates leave both the body
and its material allocation unchanged. Hardware capacity limits are explicit
execution limits, not invisible ecological selection pressure.

## The shared connection

There are several useful, distinct connections to investigate:

1. Common ancestry: shared reaction definitions, developmental primitives and
   inherited neural building blocks.
2. Physical coupling: signals, nutrients, secretions and modified structures
   travel through the environment and have local consequences.
3. Transferable developmental modules: a synthetic carrier could transmit a
   bounded metabolic or developmental capability, with its own material cost,
   receptor compatibility and replication rules.

The third direction takes inspiration from the user's uplift-virus idea. It is
a proposed digital mechanism, not a claim about biological viruses. A carrier
would transmit declared data/modules rather than execute arbitrary software.
It need not transmit adult episodic memories. Shared immutable model weights
are a compute optimization; coupling between organisms must have an explicit
route, latency and consequence in their world.

## Conditions for varied strategies

The environment should offer spatially and temporally different opportunities:
light, mineral availability, flow, attachment surfaces, shelters and accessible
biomass. Tradeoffs can arise between structural investment, acquisition surface,
mobility, sensing, defense and reproduction. Resource recycling and the ability
to modify a niche can create dependencies among strategies.

We will first establish actual material transfer, growth and viable coexistence
of supplied founders. Then vary inherited programs and environmental conditions,
measuring which strategies persist and which dependencies develop. Reproduction
plus random mutation alone is not evidence of open-ended evolution.

The fly-informed mobile residents remain one lineage in this larger world.
Different organisms need not each instantiate an entire adult fly graph: they
can express different subsets or cognitive organs while sharing developmental
and ecological mechanisms. Learning useful sensorimotor behavior remains part
of the project, alongside the emergence of the world those behaviors inhabit.


## Executed integration, 5 September 2026

Three supplied related founders built over 1,600 physical parts in 180 model
seconds using the native reactor, growth kernel and MuJoCo topology updates.
The recording used no neural controller. Tissue turnover retained dead physical
scaffold. An experimental leaf removal transferred remaining tissue and
detritus into another compartment, where the common chemistry digested it.
Elemental and energy residuals were below 1e-12 in the model’s abstract units.
A restored checkpoint reproduced four subsequent physical/chemical/developmental
steps exactly in the same runtime.

This establishes coupled mechanisms, not autonomous grazing, ecological
selection, reproduction or self-sustaining coexistence. See
[BIOSPHERE.md](BIOSPHERE.md) for the command, evidence scope and integration
limits. The public [recording](https://emberian.github.io/chreatures/reef.html)
contains only selected geometry from this experiment.
