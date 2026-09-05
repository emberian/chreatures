# Edge-tiled MaleCNS recurrence

`EdgeTiledTritonCircuit` is an optional experimental backend for the complete
MaleCNS v1.0 graph (165,122 neurons, 25,563,197 directed edges, 124,025,046
synapses). It preserves all graph edges, canonical external neuron ordering,
input/readout ports, two Jacobi substeps, equations, state tensors, and float32
precision.

The established fused kernel walks each postsynaptic row one edge at a time. One
source index and weight are loaded, then a B32 resident-rate vector is gathered,
forming a long serial dependency chain for high-degree rows. The tiled kernel
loads 4, 8, or 16 consecutive canonical edges, gathers an `[edge, resident]`
block, reduces its edge axis, and serially accumulates tile totals. The final tile
uses an explicit edge mask. This creates independent gathers that the compiler
and GPU can overlap while keeping intermediate state on device.

This changes the float32 reduction tree. Edges stay in canonical consecutive
groups, but values within each group use a tree reduction. The backend metadata
pins `edge_tile`, `num_warps`, the reduction description, graph hash, precision,
and dynamics. Complete nonzero state, readouts, physiology, and snapshot-shaped
exports are compared against the canonical E1 kernel over multiple steps; the
benchmark reports maximum absolute deltas rather than claiming bit identity.

## Benchmark

After coordinating exclusive access to the hbox RX 6750 XT:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  /tank/chreatures/envs/rocm-dev/bin/python scripts/benchmark_tiled_circuit.py \
  --graph /tank/chreatures/data/malecns/derived \
  --ports /tank/chreatures/data/ports/retinal-v1-maps.npz \
  --batch-size 48 --output /tank/chreatures/tiled/artifacts/e4-e16.json
```

The bounded comparison covers E4, E8, and E16 with one wave per row/resident
tile. It separately measures the two recurrence/update launches and interleaves
complete-path timings, including input projection, recurrence, readout, and
physiology. It is a latency-hiding experiment; no DRAM-saturation inference is
made from a worst-case gather-byte estimate.

## RX 6750 XT result

The benchmark used the pinned full graph, 351 retinal-v1 inputs, 384 readouts,
batch 48, float32 state, and a HIP wave32 target. One warp is intentional: its
32 lanes cover the B32 resident tile while edge parallelism is expressed in the
compiler-visible tensor. Larger edge tiles increase the live `[E, B32]` value
block, so E4, E8, and E16 form the bounded register-pressure comparison.

| backend | recurrent pair | speedup | complete path | speedup |
| --- | ---: | ---: | ---: | ---: |
| canonical E1/W1 | 28.425 ms | 1.000x | 34.059 ms | 1.000x |
| E4/W1 | 17.254 ms | 1.647x | 22.582 ms | 1.508x |
| E8/W1 | 16.086 ms | 1.767x | **20.522 ms** | **1.660x** |
| E16/W1 | **15.404 ms** | **1.845x** | 20.586 ms | 1.655x |

Each pure-kernel median contains four repeat groups of eight two-substep pairs
after three warmups. Complete paths were interleaved in forward and reverse
order; each median contains eight samples. E8 is the integration candidate
because it had the best complete-path median, while E16's larger live tile only
won the isolated recurrence measurement.

Five steps began from the same randomized, nonzero complete state. For E8, the
maximum absolute deltas against the canonical serial reduction were
`2.5332e-7` for rates, `1.1176e-8` for adaptation, `5.9605e-8` for support and
combined outputs, and zero for resident times. E4 and E16 had the same output,
adaptation, support, and time bounds; their rate bounds were `2.8312e-7` and
`2.3842e-7`, respectively.

Snapshot replay was also exercised at B48 after the five evolving steps:
snapshot, one further step, restore, and repeat produced exactly zero maximum
delta for output, rates, adaptation, support, and resident times. The pinned
runtime-candidate import is:

```python
from chreatures.tiled_circuit import MaleCNSEdgeTiledCircuit
```

Its constructor and `step_numpy`, `export_state`, `import_state`, and `metadata`
contract match the existing fixed circuit. It fixes E8/W1 and identifies itself
as `fixed-cohort-triton-edge8-csr-v1`.

The measured E8 backend is eligible for explicit integration, but this module
does not switch production residents or training automatically. Raw results are
stored on hbox at
`/tank/chreatures/tiled-src-v1/artifacts/benchmark-e4-e8-e16.json`.
