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
  --export-config fixtures/finite-garden-v1.json
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
