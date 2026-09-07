# Portable resident browser host

This small wasm-bindgen host uses the existing `cognitive-core` Rust engine
without its Python or Accelerate features. It contains no second controller,
world implementation, network loader, or privileged sensory channel. The host
can run in a worker or in Node without a DOM.

Build with the installed wasm-bindgen CLI **0.2.127** and the
`wasm32-unknown-unknown` Rust target:

```sh
python3 native/resident-runtime/build_wasm.py
# Outputs site/live/pkg/resident_runtime.js and resident_runtime_bg.wasm.
# For a Node fixture run:
python3 native/resident-runtime/build_wasm.py --target nodejs --out-dir /tmp/chreatures-resident-node
```

The helper enables `simd128`; `--out-dir` chooses generated artifacts and
`--target-dir` moves Rust build storage. `WASM_BINDGEN` may select a matching
project-local CLI. Generated JS has its TypeScript API next to it.

```js
import init, { ResidentRuntime } from './pkg/resident_runtime.js';
await init();
const resident = new ResidentRuntime(JSON.stringify({
  batch: 3, action_mode: 'sample', action_seed: 314, suffix_seed: 271,
  core_sha256, predictor_sha256,
  sequence_control_version: 1, sequence_control_sha256,
  research_training: false,
}), corePacked, predictorPacked, sequencePacked);

// z: Float32Array[B*512], previous: Float32Array[B*12], resident-major.
// The graph/readout supplies z; the physics receipt supplies own prior command.
const ticks = new BigUint64Array(3).fill(0n);
const result = resident.stepFlat(z, previous, ticks, new Uint8Array([1, 1, 1]));
const commands = result.proposedCommand;             // stable copied Float32Array
const observer = JSON.parse(result.diagnosticsJson); // copied observer data
result.free();
// After physics executes the commands, acknowledge exactly what it delivered.
resident.acknowledge(ticks, commands);
const checkpoint = resident.saveBytes();             // copied Uint8Array
resident.loadBytes(checkpoint);                      // validates then commits
const expanded = resident.expanded(1, 19n, 23n);      // explicit new cohort
expanded.free();
resident.free();
```

The caller authenticates downloaded packed-array bytes against the asset
manifest before construction. The immutable artifact SHA identities bind
resident snapshots. Inputs `corePacked`, `predictorPacked`, and `sequencePacked`
are float32 arrays in the current training artifact packing order. Reset is a
byte vector containing only 0 or 1; tick arrays retain full u64 precision.

`ResidentStep` and its getter results do not alias mutable engine memory.
Diagnostics include neural state, achieved goals and their provenance, personal
memory counts, sampled control decision/value/log probability, recalled suffix
flags, execution phase, cancellation counters, and receipt status. Observer
data must never be routed back to the policy as a parallel sensory input.

The checkpoint envelope preserves the existing v10 schema and its private RNG,
memory and pending receipt boundaries. It does not contain browser world or
neural graph state: the orchestrator must checkpoint all three together.

Reproduce the joined native/Node Wasm verification without browser automation:

```sh
cargo run --manifest-path native/cognitive-core/Cargo.toml --no-default-features --release --example portable_resident_probe -- /tmp/chreatures-resident-parity
python3 native/resident-runtime/build_wasm.py --target nodejs --out-dir /tmp/chreatures-resident-node
node native/resident-runtime/parity_probe.cjs /tmp/chreatures-resident-node /tmp/chreatures-resident-parity
```

The fixture uses deterministic nonzero weights, two residents and fourteen
sampled ticks. It tests mechanism parity and persistence, not learned behavior.
