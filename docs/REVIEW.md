# Runtime review

This review exercises the September 5, 2026 first-life runtime as an integrated
system. It tests the measured graph controller against the physical world, not
only the individual serialization schemas.

## What the validation establishes

- A habitat advanced for 60 ticks can be saved, loaded, and advanced for another
  60 ticks with the same world state, neural state, actions, RNG continuation,
  events, histories, and outcomes.
- Checkpoints reject a changed checksum, mismatched world and brain resident IDs,
  malformed neural state, and an anatomical artifact whose SHA-256 differs from
  the one recorded by each brain.
- Recurrent transmission has a causal effect. From identical neural and RNG
  state under an identical 30-step sensory sequence, recurrent silencing changes
  the measured downstream rates, decoded sensory coordinates, and forward
  action. This is an ablation of the recurrent term as a whole; the current
  runtime does not yet expose a cell-type or edge-selective intervention.
- Controller sensory encoding ignores injected coordinate, heading, target, and
  object-ID keys. The runtime supplies only energy, gut, and fatigue as the
  separate physiology mapping.
- Read-only HTTP state remains available regardless of Origin. State-changing
  HTTP commands require a matching Origin when one is present, require JSON,
  enforce a 16 KiB body limit, and return client errors for malformed or
  physically invalid commands without changing world state.
- A short regression trajectory confirms that all three bodies move and that
  body physiology, action outputs, neural rates, and support remain finite.

The automated cases are in `tests/test_runtime.py`. They intentionally compare
full evolving snapshots rather than only checking that a checkpoint file can be
parsed.

## Headless behavioral probe

A longer non-test probe ran seed 7 for 3,600 ticks, or 180 simulated seconds.
All residents remained finite, moved far from their hatch positions, contacted
food, and received nutrition. They consumed 1.807 of the initial 3.7 food units:
Mica received 0.816 nutrition, Fern 0.499, and Pip 0.491. Their ending energies
were 1.000, 0.999, and 0.998 after digestion, and no simulation exception
occurred.

Three separate seeds were also run for 1,200 ticks, or 60 simulated seconds
each. Eight of the nine resident histories received nutrition in that first
minute; seed 7's Mica approached to 1.43 habitat units beyond the food edges but
did not cross the one-unit ingestion margin until later. Total food consumed per
nursery was 0.990, 1.206, and 1.531 units for seeds 7, 19, and 37. These fixed-seed
results show that local odor steering can now cause contact and feeding without
coordinate or target input. They establish repeatable behavior for these seeds,
not a general foraging success rate.

## Limits of the claim

The circuit edges are measured FlyWire connections. Sensory assignment, rate
dynamics, recurrent gain, output calibration, plasticity, and action rules are
engineered. The silencing result establishes that measured recurrent paths affect
the implemented behavior; it does not establish that the behavior reproduces a
fly or that a particular biological cell class causes it.

The current `silenced` control is an internal brain flag. A research fork can be
created in Python and changed explicitly, but selective intervention metadata
and a server command are not part of this runtime slice. Likewise, the
checkpoint SHA-256 is an integrity checksum rather than an authenticity
signature.

Connectome loading uses a process-local cache for the parsed graph, while restore
also hashes the current artifact bytes. The runtime test replaces an artifact at
the same path after it has entered the cache and verifies that restore rejects
it; a distinct path or fresh process alone would not exercise that boundary.
