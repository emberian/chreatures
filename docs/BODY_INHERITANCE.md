# Inherited articulated morphology

Each habitat resident may carry an `articulated_traits` mapping. The traits
resolve the shared six-leg body definition into resident-specific trunk, limb,
antenna, density, friction, torque, sweep, and cadence parameters before MuJoCo
compiles the world. Residents therefore have different collision geometry,
mass and inertia while retaining the same eight high-level actions and twelve
ordered joint-position and joint-velocity channels.

The bounds and coupled torque/frequency scaling are engineered developmental
constraints for stable simulation. They are not measured biological allometry.
Long limbs receive a conservative cadence cap, while body size and density
affect mass faster than the bounded torque scale, creating locomotor tradeoffs.
The resolved source and per-resident hashes and trait receipts appear in world
snapshots and views. Private dynamic and learning state remain resident-local.

This mechanism keeps six legs and the existing tripod reflex. It does not claim
variable limb counts, climbing specialization, flight, or inherited behavior.
