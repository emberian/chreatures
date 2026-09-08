# Chreatures ecology core

`chreatures-ecology-core` is the current reusable native ecology kernel for new
fly worlds. It does not load or reinterpret frozen Biosphere lives.

The kernel owns finite regional, organism, packet and constructed-structure
material; diffusion and signed advection over a connected region graph;
budgeted reaction expression; private acclimation and inherited parameter
variation; material-funded construction and reproduction; structure decay;
and ATP-equivalent maintenance and measured physical-work costs. It contains no
Python and can be called once per tick by native or Wasm hosts.

All coordinates are SI meters and all durations are seconds. A millimetre fly
therefore has dimensions around `0.001`, never a hidden unit conversion. The
material unit is explicitly synthetic. The reaction validator enforces every
declared elemental axis and rejects undeclared energy creation.

## Host boundary

`TickInput` contains physical facts only: route clearance and flow, measured
contacts, finite directional transfers gated by contact plus actuators, photon
exposure, measured-work ATP demand, and physically available construction or
birth sites. A directional transfer is still limited by the donor's finite
inventory and the receiver's capacity. Mouth intake and saliva therefore use
the same operation, in opposite directions, and cannot create food or water.

`prepare_step` computes a complete successor without advancing the committed
world. Its `WorldDelta` contains exact material records plus physical create and
remove proposals. The host applies topology atomically and returns the exact
proposal IDs to `commit_step`; `abort_step` leaves time, matter, physiology and
RNG unchanged. Snapshots include an outstanding prepared successor and delta,
so a restored owner can finish or abort the known transaction without replaying
an unknown mutation. `EcologyBatch` provides the same boundary for many worlds
and validates every receipt before committing any of them.

Chemical samples and `fly_interoception12` are inputs to the body afferent
transducer. They must enter the anatomical CNS recurrence before cognition or
motor selection. They are not a parallel controller observation API.

## Joined example

`examples/closed_ecology.rs` declares the shared eight-pool garden chemistry:
water, soluble carbon, amino nutrient, mineral, oxygen, oxidized carbon,
biomass and volatile carbon, with conserved C, N, water, mineral and O axes. Its
two inherited capacity sets have no strategy labels. The scenario joins
regional diffusion/advection, a physically anchored colony, a mobile fly,
contact-gated intake and finite saliva, measured-work ATP cost, construction
that blocks a host-measured route, structure decay that reopens it, one inherited
offspring, packet retirement, exact pending-state restoration and a two-world
batch continuation.

Run the joined receipt:

```sh
cargo run --release --example closed_ecology
```

Export the same validated recipe without advancing a world:

```sh
cargo run --release --example closed_ecology -- \
  --export-config fixtures/finite-garden-v2.json
```

The checked-in export also records the fixed odor permeability vector and the
host respiratory-surface conductance recipe. `ContactEvent` conductance is a
volume rate and is multiplied by inherited membrane permeability. A
`DirectedExchange` is a finite per-tick upper bound already gated by measured
contact and actuation; `MetabolicWorkDemand` is likewise a per-tick ATP amount.

The executable asserts all transactions and reports the maximum elemental
residual. It establishes implemented coupling and deterministic restoration;
it does not claim learned feeding, ecological optimality or biological rate
calibration.


## Autonomous local colony development

`propose_growth(&GrowthInput)` creates candidate sites from inherited development,
anchored colony poses, current committed structures, local light, measured surface
patches and clearance rays. Apical extension follows the last funded apical tip;
lateral branching visits committed stems with a private cursor. The inherited
`branch_angle_rad`, `lateral_probability`, `phototropism`, `contact_avoidance` and
`directional_persistence` determine orientation. The parent's developmental
parameters mutate with its existing bounded inheritance rule. Reproduction also
inherits `dispersal_distance_m`.

The host supplies `GrowthInput { dt_s, colonies }`. Each colony entry contains
`organism_id`, `position_m`, `orientation_xyzw`, `surface_normal`,
`light_direction`, bounded `light_intensity`, `nearby_surfaces`,
`clearance_samples`, `host_template_id` and optional `child_template_id`.
A measured surface has `surface_id`, `region_id`, `point_m`, a free-space
`normal`, and `attachable`. A clearance ray has `origin_m`, `direction` and the
measured `free_distance_m`. Every position, radius and distance uses meters.
Empty surface/ray arrays mean no such measurements are available, never that
an unqueried world is collision-free.

The returned `GrowthProposals` contains a token, the committed step index,
construction/birth sites and `clearance_queries`. Each query names a site and
has `from_m`, `to_m`, `radius_m`, `kind` and optional `attachment_binding`.
The candidate capsule's local +z axis runs from `from_m` to `to_m`; its pose is
centered between them. The attachment may touch the proximal connection. The
host must check the complete proposed capsule against actual current geometry.
It passes only exact accepted sites into `TickInput`, with
`growth_token: Some(proposal.token)`. Geometry cannot be edited under that token.
Resource allocation and physical create/commit remain the ordinary transaction.
The nearest intersected material route is reported only as a hint; the next
host-measured route openness controls actual diffusion/advection.

Repeated identical proposal input is idempotent. Different input requires
`discard_growth(token)` or completion of the current tick. Pending proposals,
tentative RNG and prepared ecology transactions survive snapshots. Only an
actually funded and committed creation installs developmental RNG/cursor changes;
rejected geometry and material/ATP shortfall do not. Aborting a step restores the
exact pre-prepare state and retains the known proposal. A rejected capsule's
measured obstruction can be supplied in the next tick's clearance rays to steer
subsequent growth, without adding hidden controller observations.

Birth proposals require both an anchored parent and an explicit attachable nearby
surface within inherited dispersal range. `child_template_id: None` disables them.
The child is an anchored ecological colony with finite parental endowment, not a
synthetic fly offspring. Mobile organisms never generate proposals through this
API. Raw geometry and material-region identities stay entirely in the host/ecology
boundary and cannot enter the fly controller.

The joined development scenario runs one complete native growth/transaction loop:

```sh
cargo run --release --example developmental_growth
```

It couples phototropism, contact avoidance, apical/lateral growth, inherited colony
birth, finite funding, changed route transport and exact pending restore/abort.
Its clearance host is explicitly analytical capsule geometry. The production
MuJoCo integration is a separate host measurement, not claimed by this example.
