# Local Metal MaleCNS service

The current Apple Silicon backend executes the complete canonical MaleCNS v1.0
graph: 165,122 traced neurons, 25,563,197 directed edges, and 124,025,046
measured source synapses. Its dataset SHA-256 is
`48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625`.
The backend uses the retinal-v2 physical sensor semantics and their actual
351-channel input and 384-row readout maps.

## Native execution

`native/metal-brain` is one persistent Rust process using Metal compute
kernels. CSR rows are postsynaptic targets. Residents are stored in `float4`
tiles, with `ceil(capacity / 4)` tiles for each neuron. Every input projection,
recurrent, correction, readout, gather, and physiology kernel uses that same
runtime stride. Capacity is explicit, limited to 6 through 32 residents, and
authenticated by the native startup response and snapshot metadata. A 32-bit
active mask preserves unstepped residents inside the shared cohort state.

The SIMD kernel assigns a 32-lane SIMD group to each `(neuron, resident tile)`
pair. Each lane walks an incoming row at stride 32 and `simd_sum` reduces all
four resident lanes. Two Jacobi substeps implement:

```text
activation = 0.005 + drive + 0.92 * (W @ rate) - 0.10 * adaptation
target = max(tanh(activation), 0)
rate += min(1, dt / (2 * 0.16)) * (target * support - rate)
```

Adaptation and support update after the second substep. Rates, weights, and
private neural state remain float32. The graph remains sparse. One HTTP step
transfers `351 * capacity` inputs and returns `387 * capacity` values; full
neural state stays in unified memory.

Snapshots use `neuron-major-float4-tiles-v1`. Restore requires the exact
capacity, kernel, graph, binary artifact, artifact manifest, retinal-v2 port
spec, input/readout names, and optional mushroom state identity. Older neural
state layouts are rejected.

## Artifact identity and generation

The ignored binary cache is
`data/metal-brain/metal-csr-retinal-v2.bin` (207,261,844 bytes, SHA-256
`4a2df4b62208cb4021c6abe1e33c02f008f13d8964c90eebe8255a68a9b88df0`).
The compact checked receipt is
`data/metal-brain/metal-csr-retinal-v2.manifest.json` (SHA-256
`0a2ece24ff71b9ccecb7f8351594bc9e79ef16430d432cb003fb5eacf548aa0b`).
It pins the canonical graph, artifact size and checksum, retinal-v2 bundle
SHA-256 `933b871fdd11dafa8c43afceb9862101984bf0950592af84631a4d7aa9bebe53`,
and semantic spec SHA-256
`a3182cc5c546fac164774e56cfcf3d4f185c2feab5a994fe3d2a37cc8604302e`.

Build directly from the canonical graph cache:

```bash
.venv/bin/python scripts/prepare_metal_brain.py \
  --graph /path/to/malecns/derived \
  --port-spec data/ports/retinal-v2.json \
  --port-bundle data/ports/retinal-v2-maps.npz \
  --output data/metal-brain/metal-csr-retinal-v2.bin
```

The checked retinal-v1 and retinal-v2 bundles have different physical sensor
specifications but equal dtype, shape, names, and element values for every
sparse map array. A machine that already has the canonical binary can export
the current cache without loading the bulk graph. This route verifies the old
binary and bundle against their receipt, proves all map arrays equal, copies
the unchanged canonical weights/maps, and records the source identities:

```bash
.venv/bin/python scripts/prepare_metal_brain.py \
  --source-artifact data/metal-brain/metal-csr-v2.bin \
  --source-manifest data/metal-brain/metal-csr-v2.manifest.json \
  --source-port-bundle data/ports/retinal-v1-maps.npz \
  --port-spec data/ports/retinal-v2.json \
  --port-bundle data/ports/retinal-v2-maps.npz \
  --output data/metal-brain/metal-csr-retinal-v2.bin
```

The source and exported binary hashes are equal because the graph weights and
sparse maps are equal. The manifest changes because retinal-v2 defines the
current physical preprocessing semantics.

## Service launch

Build the release worker, then launch an isolated six-resident service:

```bash
cargo build --release --manifest-path native/metal-brain/Cargo.toml

.venv/bin/python scripts/serve_metal.py \
  --artifact data/metal-brain/metal-csr-retinal-v2.bin \
  --port-bundle data/ports/retinal-v2-maps.npz \
  --capacity 6 \
  --kernel simd \
  --snapshot-dir runs/server/malecns-retinal-v2-metal-b6/snapshots \
  --pid-file runs/server/malecns-retinal-v2-metal-b6/brain.pid \
  --bind 127.0.0.1 \
  --port 18776
```

Startup hashes the 198 MiB binary, the compact bundle, and manifest before
spawning native code. The native worker compiles the Metal shader and reports
the device, configured capacity, tile count, neuron count, input count,
readout count, and kernel. The Python host rejects any mismatch. The service
advertises these identities at `/v1/metadata` and starts with no residents.

On the 96 GB Apple M2 Max used for the joined launch, an empty B6 service used
about 61 MB RSS in Python and 449 MB RSS in the native worker. The machine had
116 GiB disk free after writing the 198 MiB binary cache. One actual full B6
step produced six distinct finite readout vectors. The first and next Metal
command buffers took 34.3 ms and 22.5 ms. An exact snapshot/restore continuation
was bit-identical; its neural state file was 15,879,714 bytes. These are a
single launch validation, not a throughput distribution.

The localhost HTTP layer uses sequenced mutation requests and
`chreatures-request-receipt-v1`. Its last 64 outcomes can be queried by exact
service incarnation, sequence, and canonical request SHA-256 without executing
the operation again. An uncertain or failed mutation pauses its caller; the
host does not assume that a missing reply means no state changed.
