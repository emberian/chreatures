# Experimental local Metal MaleCNS inference

This experimental backend executes the complete MaleCNS v1.0 graph locally on
Apple Silicon. It remains opt-in and does not alter an existing runtime. The
measured graph is the curated traced graph with 165,122 neurons, 25,563,197
directed edges, and 124,025,046 source synapses (dataset SHA-256
`48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`).

## Kernel and state

`native/metal-brain` contains a Rust benchmark and a persistent newline-JSON
worker using the `metal` crate and MSL compute kernels. CSR rows are
postsynaptic targets. One GPU thread owns one
target row, walks its incoming edges, and accumulates all three residents in a
`float4`; the fourth lane is padding. Rates use ping-pong buffers, while
adaptation and support are private persistent buffers. Each model step submits
two Jacobi substeps, exactly matching `fast_circuit.py`:

```
activation = 0.005 + drive + 0.92 * (W @ rate) - 0.10 * adaptation
target = max(tanh(activation), 0)
rate += min(1, dt / (2 * 0.16)) * (target * support - rate)
```

After the second substep, adaptation and support update once. All arrays remain
float32 and the graph remains sparse. The dispatch uses the pipeline's reported
`threadExecutionWidth` as its threadgroup width, following Apple's guidance to
use a multiple of that width. Shared buffers fit Apple Silicon's unified-memory
model and make snapshot verification direct. Relevant primary references are
[threadExecutionWidth](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth)
and [storageModeShared](https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared).

The v2 compact local artifact contains uint32 CSR pointers/indices and float32
weights for recurrence, the actual 351-channel retinal-v1 input projection,
and the actual 384-row readout. It is 207,261,844 bytes. The preparation script
derives it from the canonical arrays and compact public port bundle; it does
not change either. The root `.gitignore` excludes `data/metal-brain/` bulk artifacts.

`data/metal-brain/metal-csr-v2.manifest.json` is the checked-in canonical
receipt for the ignored binary. It pins artifact SHA-256 and byte size, graph
SHA-256, retinal-v1 semantic spec SHA-256, compact port-bundle SHA-256, array
counts, format, and construction recipe. `MetalCircuit` hashes the full binary
and compact port bundle and validates every identity before spawning native
code. A same-shaped altered binary cannot inherit the canonical graph label.

The persistent worker retains rate, adaptation, support, drive, and scratch
buffers. A Python request transfers 351 x 3 input scalars and returns 387 x 3
scalars; it never transfers the full neural state during a step. All input
projection, recurrence, readout, and physiology work executes in one Metal
command buffer. A two-stage GPU reduction computes activity mean, activity
peak, and support mean without serializing over 165,122 neurons.

`chreatures.metal_circuit.MetalCircuit` implements resident allocation,
arbitrary request order, inactive-slot preservation, times, metadata, step,
remove, full-cohort snapshot, and restore. Snapshots atomically store all
private GPU state plus resident slot order and times. `scripts/serve_metal.py`
provides the same sequenced localhost HTTP endpoints as `serve_brain.py`
without importing Torch.

The stable default remains `kernel="row"`. An explicitly selected
`kernel="simd"` assigns one 32-lane SIMD group to each CSR row; lanes traverse
the row at stride 32 and reduce the three resident accumulators with
`simd_sum`. Select it only for a new world, either through the Python argument
or `serve_metal.py --kernel simd`. Snapshot version 3 pins the artifact, graph,
canonical port spec, and kernel identity, so the two float32 reduction orders
cannot silently cross-restore. Version 2 snapshots without artifact identity
have one labeled migration path: they are accepted only when the loaded
artifact independently authenticates as the pinned canonical v2 digest and
the graph and port hashes are canonical. Kernel identity must still match.

## Reproduce

With the canonical graph available locally:

```bash
.venv/bin/python scripts/prepare_metal_brain.py \
  --graph /path/to/malecns/derived \
  --port-spec data/ports/retinal-v1.json
.venv/bin/python scripts/benchmark_metal_brain.py --iterations 20
```

The preparation command verifies the canonical graph hashes and the compact
port bundle against the semantic spec before writing
`data/metal-brain/metal-csr-v2.bin` and its sidecar through temporary files and
atomic renames. The checked 413 KB port bundle is included
in the repository. If it is absent, also pass `--annotation` with the pinned
MaleCNS annotation Feather named by the spec; the command will rebuild and
verify the bundle first. No hbox path is assumed. Add `--complete` to the
benchmark for the real input/readout and full-sequence SciPy comparison; that
reference mode also needs `--graph /path/to/malecns/derived`.

An isolated service can be started with:

```bash
.venv/bin/python scripts/serve_metal.py \
  --snapshot-dir runs/metal-local/snapshots \
  --pid-file runs/metal-local/server.pid
```

## Result on this machine

Measured 2026-09-05 on the 38-core Apple M2 Max, macOS 26.6, Metal 4, using 20
timed iterations after parity and replay checks:

| quantity | result |
|---|---:|
| complete two-substep B3 update, median | 8.853 ms |
| complete update, minimum | 8.558 ms |
| maximum absolute delta vs scalar float32 CPU | 2.3e-8 |
| same-backend snapshot/restore replay delta | 0.0 |

The complete retinal-v1 backend was then measured over 20 B3 requests after
three warmups:

| quantity | result |
|---|---:|
| complete Python request, median | 15.429 ms |
| complete Python request, minimum | 13.496 ms |
| Metal command buffer, median | 14.256 ms |
| 384 readouts vs full SciPy reference, maximum delta | 3.73e-8 |
| physiology vs full SciPy reference, maximum delta | 9.30e-6 |
| complete-state snapshot replay delta | 0.0 |

The selectable SIMD-row kernel materially improves the same complete workload:

| quantity | row | SIMD-row |
|---|---:|---:|
| complete Python request, median | 15.429 ms | 9.546 ms |
| Metal command buffer, median | 14.256 ms | 8.415 ms |
| complete Python request, minimum | 13.496 ms | 7.383 ms |

Both modes retain every edge, two substeps, float32 state, and the actual port
maps. Over 20 identical sequential steps their maximum readout difference was
`2.98e-8`; this is a faithful float32 reduction-order difference rather than
bit identity. SIMD mode's four-step SciPy comparison retained the row mode's
`3.73e-8` maximum readout delta and `9.30e-6` physiology delta.

## Cost and roofline estimates

Actual graph row degrees explain why SIMD parallelism helps: mean degree is
154.8, median 100, p95 455, p99 861, and maximum 11,526. There are 1,029 rows
over degree 1,024, containing 1,705,009 edges. Only 1.46% of edges belong to
rows shorter than 32, so idle SIMD lanes on short rows cover little graph work.
The original one-thread-per-row kernel made each long row a serial tail.

Each recurrent edge nominally requests 4 bytes of source index, 4 bytes of
weight, and a 16-byte `float4` rate gather. Across two substeps that is 1.227 GB
of requested algorithmic edge access, before small row/state traffic. This is
not an estimate of bytes reaching DRAM. Dividing this requested nominal volume
by complete command-buffer time gives 86 GB/s for row mode and
146 GB/s for SIMD-row mode. These are effective access-rate estimates, not
hardware-counter DRAM measurements: source-rate cache reuse can lower physical
traffic, while random gathers and transaction granularity can raise it. Apple's
[M2 Max specification](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/)
states 400 GB/s unified-memory bandwidth, making the estimates about 22% and
36% of advertised peak respectively. The comparison does not demonstrate peak
saturation. The measured speedup plus the degree distribution instead supports
row imbalance and gather latency as major costs. Splitting the 1,029 longest
rows across multiple SIMD groups is a plausible next exact-float32 experiment,
but requires a second partial-reduction pass and has not been implemented or
timed.

The earlier 8.853 ms measurement covers only the two recurrent CSR passes. The
15.429 ms measurement includes Python validation, JSON pipe transport, actual
input projection, both recurrent passes, actual readout, physiology reduction,
and the small result transfer. It excludes HTTP transport. The physiology delta
comes from parallel float32 reduction order. Fixed B3 is intentional; larger
and dynamic resident cohorts have not been measured.

The benchmark performs one untimed GPU warm-up step before its checks and timed
loop. The parity check compares that first step against an independent scalar
Rust implementation over the same full CSR and deterministic initial state.

## Explicit execution migration

Normal restore rejects a different numerical kernel. The operator tool
`scripts/migrate_metal_world.py` instead makes a new world manifest and native
receipt from an authenticated paused checkpoint. It preserves all three native
state arrays byte for byte, records both parent hashes and the payload hash,
and declares the changed future float32 reduction order. It leaves the source
files intact. The optional physical execution change is recorded separately.

The continuous terrarium migrated at tick 11,825 from `row`/`reference` to
`simd`/`vectorized`. Before resumption, physics, chemical fields, ecology,
acoustics, motors, outcomes, neural summaries, feature statistics, resident row
order, visitor state and the entire native state payload matched the paused
source exactly. Its current manifest is `runs/terrarium-simd.json`; the older
`runs/terrarium-garden.json` is a preserved historical checkpoint. This is exact
state preservation at the boundary, not a claim of bit-identical future
trajectories across different neural reduction orders.
