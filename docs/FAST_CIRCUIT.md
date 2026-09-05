# Full-graph MaleCNS throughput

`chreatures.fast_circuit` evaluates the complete 165,122-neuron,
25,563,197-edge MaleCNS graph in float32 with the existing two-substep rate,
adaptation, and support equations. It does not prune edges, change the time
step, reduce update frequency, or substitute a smaller anatomy.

## Interfaces

`TritonFusedCircuit` is the measured throughput choice on the hbox RX 6750 XT.
`MicrobatchedResidentCircuit` remains the portable Torch CSR fallback:

```python
import numpy as np

from chreatures.fast_circuit import TritonFusedCircuit
from chreatures.malecns import MaleCNSGraph
from chreatures.neural_ports import NeuralPortBundle

graph = MaleCNSGraph.load("/tank/chreatures/data/malecns/derived")
ports = NeuralPortBundle.load(
    "/tank/chreatures/data/ports/retinal-v1-maps.npz", graph
)
circuit = TritonFusedCircuit(
    graph,
    batch_size=48,
    device="cuda",
    input_map=(ports.input_names, ports.input_map),
    readout_map=(ports.readout_names, ports.readout_map),
)

# One column per resident, in the bundle's declared channel order.
channels = np.ascontiguousarray(encoded_senses.T, dtype=np.float32)  # [351, 48]
result = circuit.step_numpy(channels, dt=0.05)
features = result.features       # [48, 384]
physiology = result.physiology   # activity mean, peak, support mean
```

Input is one contiguous CPU float32 array shaped `[channels, residents]`.
Features and all three physiology values return in one device-to-host copy.
The fixed cohort avoids resident dictionaries, slot gathers/scatters, and
device synchronization for slot IDs.

The Triton backend requires a HIP wave32 target and Triton 3.5.1 in the
measured environment. The project run used
`HSA_OVERRIDE_GFX_VERSION=10.3.0` and a persistent `TRITON_CACHE_DIR`; it did
not modify the shared ROCm installation or GPU driver.

`export_state()` and `import_state()` use `rates`, `adaptation`, and `support`
arrays shaped `[residents, neurons]`, plus a `[residents]` float64 clock. This
is the existing external snapshot orientation even when an internal experiment
uses neuron-major state.

`RemoteBrain(..., microbatch_size=3)` and `serve_brain.py --microbatch-size 3`
provide a compatibility path. Slot-ordered prefix cohorts use state views and
bypass `index_select`/`index_copy`; other resident selections retain the safe
gather/scatter path. Omitting the option keeps the original unsplit execution.

## Layout result

The first hypothesis was that persistent neuron-major `[N, B]` state would
avoid a hidden copy of the current resident-major `[B, N].T` operand. ROCm's
CSR implementation did not reward that layout consistently:

| Batch | Resident-major end-to-end | Neuron-major end-to-end |
|---:|---:|---:|
| 3 | 9.304 ms | 9.340 ms |
| 48 | 191.475 ms | 188.508 ms |
| 96 | 375.013 ms | 380.720 ms |
| 192 | 772.190 ms | 758.336 ms |

The full neural state and all 384 readouts were identical across four common
input steps. A separately reduced physiology mean differed by at most
`1.19e-7`, one float32-scale reduction-order difference. Snapshot/restore
replay was exact. Because neither layout won consistently, neuron-major is
available for investigation but is not the runtime recommendation.

## Microbatch result

Resident dynamics are independent columns under shared immutable weights.
Splitting a 48-resident request into smaller CSR dense operands therefore
preserves the equations and state while changing only kernel scheduling. The
paired warmed sweep found:

| Microbatch | B48 end-to-end | Resident steps/s |
|---:|---:|---:|
| 1 | 205.963 ms | 233.1 |
| 2 | 156.536 ms | 306.6 |
| **3** | **146.614 ms** | **327.4** |
| 4 | 147.933 ms | 324.5 |
| 6 | 153.491 ms | 312.7 |
| 8 | 161.255 ms | 297.7 |
| 12 | 152.619 ms | 314.5 |
| 16 | 157.823 ms | 304.1 |
| 24 | 157.895 ms | 304.0 |
| 48 | 158.839 ms | 302.2 |

A deeper paired run with three warmups, six iterations, and five repeats
confirmed microbatch 3 at `146.664 ms` (`327.28` resident steps/s) versus the
unsplit chunk at `159.026 ms` (`301.84` resident steps/s), an `8.43%` gain.
Microbatch 3 used 398,568,448 peak Torch bytes versus 493,821,440 for the
unsplit path.

The microbatched engine was also compared against the unsplit reference for
three full 48-resident steps. Rates, adaptation, support, clocks, and all 384
features matched exactly. Physiology reductions differed by at most `4.17e-7`;
snapshot/restore replay had maximum absolute delta zero.

These paired measurements are the defensible optimization result. The earlier
cold reference measured `191.475 ms`, and the active learner under contention
measured about `208.7 ms` and 230 resident steps/s. Those runs show operational
headroom but are not used to claim the paired speedup.

## Bottleneck and next kernel

In the confirmed warmed unsplit run, one isolated full B48 recurrent CSR SpMM
took `74.72 ms`; two such operations account for roughly 94% of the `159.03 ms`
device step. The direct NumPy path added only about `0.21 ms` around the
microbatch-3 device step, so local array transfer is not the main cost. HTTP,
JSON, world simulation, and GPU queue contention are separate costs in service
runs.

The hbox project environment has Triton 3.5.1 and recognizes HIP `gfx1030`,
although the host has no `hipcc` and PyTorch reports no `ROCM_HOME`.
`TritonFusedCircuit` is a separate experimental backend compiled through
Triton's HIP target. Each program assigns one postsynaptic row to a wave32
resident tile. It walks the row's CSR edges, broadcasts each edge weight and
source index, reads neuron-major resident values coalescently, and writes the
new rate directly. Two launches ping-pong rate buffers, preserving a global
Jacobi boundary between substeps; adaptation and support update in the second
launch.

This mapping follows AMD's observation that sparse multiplication is generally
limited by memory bandwidth, while adapting the usual CSR row assignment to
the circuit's dense resident dimension. See AMD GPUOpen's [Sparse matrix vector
multiplication notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-spmv-docs-spmv_part1/)
and the [rocSPARSE user guide](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/how-to/using-rocsparse.html).

The first bounded check used 4,096 real MaleCNS target rows, 2,847,165 edges,
and 48 residents. One fused recurrence/rate substep differed from Torch by at
most `3.73e-9` (mean `2.11e-10`) and all outputs were finite. Its median was
`5.079 ms` versus `12.301 ms` for Torch CSR plus the same rate update, a `2.42x`
slice speedup. This is promising slice evidence, not a full-graph throughput
claim.

A subsequent correctness-only run covered all 165,122 neurons and 25,563,197
edges at batch 48 for four common two-substep inputs. Compared with the Torch
CSR reference, maximum absolute differences were `1.19e-7` in rates,
`3.73e-9` in adaptation, `5.96e-8` in support, `5.96e-8` in the 384 readouts,
and `2.38e-7` in physiology; clocks were exact. Snapshot/restore replay was
bit-exact for state and every output.

The coordinated full-graph batch-48 timing used two ABBA cycles with ten
complete steps per block. Each step included the 351-channel host input copy,
two recurrent substeps over every edge, 384 readouts, physiology reductions,
and one host output copy. The Torch CSR median was `159.442 ms`; the Triton
median was `27.286 ms`, a `5.843x` speedup and `1,759.15` resident steps/s
instead of `301.05`. Individual Torch blocks ranged from `159.217` to
`160.377 ms`; Triton blocks ranged from `27.270` to `27.325 ms`.

This backend changes sparse reduction order. Transitioning a saved Torch state
to Triton therefore preserves its float32 state arrays at the boundary but does
not promise the old backend's future trajectory bit for bit. The measured
cross-backend bound is `1e-6`; restore and replay within Triton are bit-exact.

Short post-training ABBA runs established the batch-size boundary. These were
run while the independent live services and worlds remained operational, so
the batch-48 exclusive result above is the primary adoption measurement:

| Batch | Torch CSR | Triton fused | Speedup | Triton resident steps/s |
|---:|---:|---:|---:|---:|
| 3 | 11.529 ms | 14.390 ms | 0.80x | 208.5 |
| 48 | 159.442 ms | 27.286 ms | 5.84x | 1,759.2 |
| 96 | 382.507 ms | 51.832 ms | 7.38x | 1,852.1 |
| 192 | 788.586 ms | 105.388 ms | 7.48x | 1,821.8 |

The native kernel is therefore selected explicitly for large fixed cohorts.
The Torch path remains preferable for a three-resident cohort and remains the
default for the small live services.

## ROCm 7 / gfx1150 portability

The same source and canonical graph were staged on persvati without changing
its shared Python or ROCm installations. The run uses Torch 2.10.0 with ROCm
7.0 from `/home/ember/kaxsim/.venv7`, Triton 3.6.0, and the existing isolated
dependency target `/home/ember/chreatures-compute/envs/python-packages`.
Persvati reports an AMD Radeon 890M, `gfx1150`, and wave32. All six derived
graph arrays, the graph manifest, the retinal bundle, and its JSON spec match
the hbox files byte for byte.

Full graph parity passed at batches 3 and 48. At batch 48 the largest state
difference from Torch CSR was `1.19e-7`, the largest readout difference was
`5.96e-8`, physiology differed by at most `3.58e-7`, and same-backend replay
was exact. Paired ABBA throughput was:

| Batch | Torch CSR | Triton fused | Speedup | Triton resident steps/s |
|---:|---:|---:|---:|---:|
| 3 | 29.381 ms | 36.678 ms | 0.80x | 81.8 |
| 48 | 1,884.757 ms | 215.699 ms | 8.74x | 222.5 |

The integrated GPU is much slower in absolute terms than hbox's discrete GPU,
but the full graph kernel is functional and provides useful independent
compute. A fresh 20,000-step canonical control is running there alongside the
matched-rewire hbox run. Both use the exact same imported shared model arrays,
seed, curriculum, and source hashes. The ROCm, Torch, and GPU differences can
change reductions and stochastic trajectories, so analysis must compare
seeded evaluation distributions with uncertainty rather than claim stepwise
trajectory identity.

The existing 48-resident developmental run then restored its step-1,230
checkpoint into the Triton circuit and completed at step 8,000. This process
executed 6,770 steps, or 324,960 resident steps, in `416.408 s`: `780.39`
resident steps/s for the whole learner. The process-local circuit timer was
reset on restore and accumulated `234.866 s`, corresponding to `1,383.60`
resident steps/s; physics took `106.943 s` and sense encoding `54.730 s`. The
circuit therefore moved from roughly 89% of the old learner wall time to 56.4%
of this continuation. The summary's `resident_steps: 384000` is the absolute
step-8,000 exposure count and must not be divided by this continuation's
elapsed time. The summary also records `404,154,880` allocated device bytes and
a completed final neural checkpoint. These are measured training figures,
while policy quality is evaluated separately from kernel throughput.

## Reproduce

```shell
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
python scripts/benchmark_fast_circuit.py \
  --graph /tank/chreatures/data/malecns/derived \
  --port-bundle /tank/chreatures/data/ports/retinal-v1-maps.npz \
  --batches 3 48 96 192 \
  --output /tank/chreatures/runs/benchmarks/fast-circuit.json
```

Executed reports are stored at:

- `/tank/chreatures/runs/benchmarks/fast-circuit-initial.json`
- `/tank/chreatures/runs/benchmarks/fast-circuit-microbatch48.json`
- `/tank/chreatures/runs/benchmarks/fast-circuit-microbatch48-confirm.json`
- `/tank/chreatures/runs/benchmarks/fast-circuit-triton-full-contention-parity.json`
- `/tank/chreatures/runs/benchmarks/fast-circuit-triton-full-b48-abba.json`
- `/tank/chreatures/runs/benchmarks/fast-circuit-triton-full-b3-abba-operational.json`
- `/tank/chreatures/runs/benchmarks/fast-circuit-triton-full-b96-abba-operational.json`
- `/tank/chreatures/runs/benchmarks/fast-circuit-triton-full-b192-abba-operational.json`
- `/tank/chreatures/runs/learning/affordance-16x3-v3-continuation/summary.json`

Persvati receipts are stored at:

- `/home/ember/chreatures-compute/runs/benchmarks/fast-circuit-gfx1150-b3-parity.json`
- `/home/ember/chreatures-compute/runs/benchmarks/fast-circuit-gfx1150-b48-parity.json`
- `/home/ember/chreatures-compute/runs/benchmarks/fast-circuit-gfx1150-b3-abba.json`
- `/home/ember/chreatures-compute/runs/benchmarks/fast-circuit-gfx1150-b48-abba.json`
- `/home/ember/chreatures-compute/runs/learning/affordance-canonical-fresh-16x3-v1/cross-hardware-control.json`
