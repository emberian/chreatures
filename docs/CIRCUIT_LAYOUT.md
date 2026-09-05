# Cache-aware MaleCNS row scheduling

`chreatures.circuit_layout` is an optional experimental execution layout for the
complete MaleCNS v1.0 graph: 165,122 neurons, 25,563,197 directed edges, and
124,025,046 measured synapses. It does not replace the canonical graph or alter
an existing resident.

The current fused kernel launches one GPU program per postsynaptic row and uses
wave lanes for residents. Its rate tensor for batch 48 is about 31.7 MB, but a
worst-case edge-gather byte count does not establish DRAM saturation. Observed
training utilization has included host gaps and low memory-busy time. This
experiment instead changes the issue order of independent rows so nearby
programs are more likely to read the same source-rate pages while they remain in
the RX 6750 XT's cache hierarchy.

## Invariants

`CircuitRowLayout` stores a checked `int32[N]` permutation and inverse. The
canonical MaleCNS dataset hash and permutation SHA-256 are embedded in the NPZ.
Construction uses exact neuron annotations plus three source-page quartile
samples per CSR row. It requires O(N) scratch and never forms a dense N-by-N
array.

All rate, adaptation, support, input, and readout tensors retain canonical neuron
ordering. Each scheduled program maps its launch row back to a canonical target
row. The canonical CSR pointers, source indices, and weights are unchanged, and
edges within every row are accumulated in their original order. Therefore this
layout introduces no float32 edge-reduction reordering. Snapshots use the same
complete canonical state schema as `TritonFusedCircuit`.

## Reproduction

Build and inspect without using the GPU:

```bash
python scripts/benchmark_circuit_layout.py \
  --graph /tank/chreatures/data/malecns/derived \
  --layout /tank/chreatures/layout/row-layout-page2048.npz --build-only
```

On the isolated hbox ROCm 6.3 environment, after coordinating exclusive GPU use:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  /tank/chreatures/envs/rocm-dev/bin/python scripts/benchmark_circuit_layout.py \
  --graph /tank/chreatures/data/malecns/derived \
  --ports /tank/chreatures/data/ports/retinal-v1-maps.npz \
  --layout /tank/chreatures/layout/row-layout-page2048.npz \
  --batch-size 48 --output /tank/chreatures/layout/benchmark-page2048.json
```

The report contains complete rate/adaptation/support/input/readout parity from a
nonzero randomized canonical state. It separately times the two recurrent/update
kernels and an ABBA-interleaved complete device step, which also includes input
projection, both recurrent substeps, readout, and physiology summaries.

## Full-graph build result

The source-page size of 2,048 neurons is 384 KiB for a batch-48 float32 rate
slice. On the pinned graph, the fraction of adjacent scheduled rows sharing at
least one sampled source page increased from 0.236548 in canonical order to
0.982855. This is a scheduling-locality proxy, not a bandwidth claim. GPU timing
rejected it as a useful optimization. With batch 48, the anatomy/source schedule
changed the median two-kernel time from 26.804 ms to 30.689 ms (0.873x) and the
ABBA complete path from 33.882 ms to 36.436 ms (0.930x). A degree-descending
schedule also lost: 28.170 ms to 29.151 ms (0.966x) for the two kernels and
34.276 ms to 36.140 ms (0.948x) for the complete path. Each candidate used four
repeat groups after three warmups.

Both schedules produced zero maximum absolute delta for rates, adaptation,
support, times, every readout feature, and physiology from a randomized nonzero
complete state. The mapping and kernel are correct, but neither improves
performance. Dispatch order is therefore not a reliable way to control useful
cache residence here; physical state/CSR tiling or a persistent work scheduler
is the next justified experiment.

The layout remains experimental and should not be integrated into training or
the persistent service. Existing residents and checkpoints continue to use the
established backend. Raw reports and generated NPZ mappings remain in hbox bulk
scratch at `/tank/chreatures/layout-src-v1/artifacts`.
