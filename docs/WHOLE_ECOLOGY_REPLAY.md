# Whole ecology replay probe

`scripts/probe_whole_ecology.py` creates a fresh research world and compares a
continued trajectory with a checkpoint-restored trajectory. It can select a
habitat, biosphere birth configuration, inherited motor genome, personal memory
and plasticity, and predictive model. It never connects to a live service for
the recorded recycling-reef receipt.

The comparison covers physics, transported fields, legacy resources, the full
biosphere and exchange ledger, acoustics, visitor schedule, adaptive organs,
living motors and personal memory, foresight state, neural responses, feature
statistics, outcomes, senses, journal, history, pending tick state, and runtime
selectors. Remote native neural snapshots compare by exact checksum. Wall-clock
performance and `saved_at` are intentionally outside the causal-state contract.

`report.json` contains per-owner expected and restored hashes and the first
structural difference on failure. `receipt.json` is the compact provenance
record with source artifact hashes, checkpoint identity, state evidence, owner
hashes, and neural checksum.

The [recycling receipt](../data/ecology/whole-recycling-replay-v1.receipt.json)
records 70 ticks before saving and 12 ticks after. That fresh research world
developed 12 physical parts, returned nonzero material, and accumulated 17
predictive observations per resident. All 26 recorded sections and the native
neural checksum matched exactly. Acoustics and the older resource mechanism
were disabled in this run; their null sections do not validate those dynamics.
This is a finite same-runtime continuation check, not cross-device equivalence
or a claim about indefinitely long trajectories.

The subsequent [exudation integration receipt](../data/ecology/whole-exudation-replay-v1.receipt.json)
uses the current exchange-v2 birth with 24 mobile return slots and 12 colony
emission slots, the optimized growth implementation, and request-receipt
transport. It advances 80 ticks before saving and 12 afterward. The world
contains 12 constructed parts, six active exudate objects, nonzero material
return and private predictive/motor state. Every recorded owner and the native
brain snapshot match exactly. Its 0.072 emitted mass comes from supplied founder
stores; the separate zero-founder-reserve light/dark assay establishes the
source-production claim. Acoustics, the older resource engine and external
perception were disabled in this integration.
