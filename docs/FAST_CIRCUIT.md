# Full-graph MaleCNS throughput

`chreatures.fast_circuit` evaluates the complete 165,122-neuron,
25,563,197-edge MaleCNS graph in float32 with the existing two-substep rate,
adaptation, and support equations. It does not prune edges, change the time
step, reduce update frequency, or substitute a smaller anatomy.

## Interfaces

`MicrobatchedResidentCircuit` is the measured throughput choice on the hbox RX
6750 XT:

```python
import numpy as np

from chreatures.fast_circuit import MicrobatchedResidentCircuit
from chreatures.malecns import MaleCNSGraph
from chreatures.neural_ports import NeuralPortBundle

graph = MaleCNSGraph.load("/tank/chreatures/data/malecns/derived")
ports = NeuralPortBundle.load(
    "/tank/chreatures/data/ports/retinal-v1-maps.npz", graph
)
circuit = MicrobatchedResidentCircuit(
    graph,
    batch_size=48,
    microbatch_size=3,
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

The hbox project environment has Triton 3.5.1 and recognizes HIP `gfx1030`, but
the host has no `hipcc` and PyTorch reports no `ROCM_HOME`. No native extension
was compiled or installed. A future kernel should map wave32 lanes to resident
columns for each postsynaptic row, use neuron-major source loads, and ping-pong
rate buffers so each substep remains a global Jacobi update. It must beat the
measured microbatch path and pass full-state/replay comparison before runtime
integration.

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
