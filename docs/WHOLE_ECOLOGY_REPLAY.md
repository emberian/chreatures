# Whole ecology replay probe

`scripts/probe_whole_ecology.py` creates a fresh research world and compares a
continued trajectory with a checkpoint-restored trajectory. It can select a
habitat, biosphere birth configuration, inherited motor genome, personal memory
and plasticity, and predictive model. It never connects to a live service for
the recorded recycling-reef receipt.

With `--visitor-materials`, the probe saves a complete birth checkpoint before
any resident advances. `--offer-material` and `--offer-position X Y Z` then
place a finite declared choice through `Habitat3D.command`; supply inventory and
offering-slot state participate in the same exact continuation comparison.

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

The [visitor-offering integration receipt](../data/ecology/whole-visitor-offering-replay-v1.receipt.json)
records a fresh three-resident world using the native motor and biosphere
kernels, current MaleCNS artifact, inherited chemical-encounter motor,
recycling/exudation chemistry, personal learning, and the incumbent predictive
model. A birth checkpoint was written before the first advance. The visitor
then placed one `reserve-fruit` through `Habitat3D.command` at
`[1.0, 1.42, 0.16]`, transferring 0.22 reserve into a physical 0.06715-mass
object. After 50 warmup and 12 continuation ticks, all 27 mutable-owner hashes
and the full native neural snapshot matched after restore. The receipt records
the exact native binary and retained input identities; acoustics and the legacy
resource owner were disabled for this assay.

A fresh private neural service for this assay can be started from outside the
repository so the selected out-of-tree world-kernel module stays first on the
import path:

```sh
cd /tmp
PYTHONPATH=/tmp/chreatures-composed-native-v1:/Users/ember/dev/chreatures \
  /Users/ember/dev/chreatures/.venv/bin/python \
  /Users/ember/dev/chreatures/scripts/serve_metal.py \
  --artifact /Users/ember/dev/chreatures/data/metal-brain/metal-csr-v2.bin \
  --port-bundle /Users/ember/dev/chreatures/data/ports/retinal-v1-maps.npz \
  --snapshot-dir /Users/ember/dev/chreatures/runs/metal-visitor-offering-replay/brain \
  --pid-file /Users/ember/dev/chreatures/runs/metal-visitor-offering-replay/service.pid \
  --bind 127.0.0.1 --port 18774 --kernel simd
```

Port 18774 is a research-only example and must be checked for availability.
The recorded run did not connect to or mutate the live service on 18773.

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
