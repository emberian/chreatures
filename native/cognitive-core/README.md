# Chreatures cognitive core

One Rust engine serves the Python extension and the browser Wasm host. The
current resident is `chreatures-cns-resident-native-v10`. Its sensory input is
only a learned CNS-derived 512-value latent; the remaining inputs are its own
previous 12-value motor command, per-resident tick, and reset. Raw physiology,
vision, world coordinates, and object labels cannot enter this controller.

The engine owns the GRU256 state, 128-dimensional achieved goal keys, private
128-slot goal reservoir, eight-tick goal selection, three recurrent predictors,
32-slot acquired neural-context suffix memory, and learned sequence termination and
selection. Candidate rollouts read the resident's own CNS-derived state. A step
returns proposed commands; the host acknowledges delivered commands before the
next step can proceed. Learning uses the next actual CNS outcome, not imagined
outcomes or raw physiology. `sample` and `map` are explicit construction modes.

`developmental.rs` implements `from_packed`, `step_flat`, `acknowledge_flat`,
`save_bytes`, `load_bytes`, and `expanded_portable`. All arrays are flat and
resident-major. The Python host in `developmental/python.rs` only adapts NumPy
arrays and the existing dictionary protocol. The separate `resident-runtime`
crate adapts the same methods to wasm-bindgen; it has no numeric controller of
its own. Packed weights retain their canonical training-artifact order and
model identities. The byte snapshot is the existing v10 six-field JSON
envelope encoded in UTF-8. It preserves every resident's recurrent state,
private memory, sampled RNG, goal provenance, pending commands and receipts.
Restore validates before mutation; cohort expansion retains old rows.

`cns_adapter.rs` is the portable reference for the current V2 anatomical
adapter and full-graph CPU equation: raw observations enter its afferent
encoder, recurrence evolves signed deviations around the learned tonic
baseline, and only masked baseline-relative graph rates enter the factorized
rank-64 readout. Direct afferent projection columns are forcibly zero and the
authenticated neutral drive must be zero away from afferent rows. This is a
separate sensory-side component, not a resident policy information port. The
Python adapter boundary is in `cns_adapter/python.rs`.

Default features are `python` and `accelerate`. `python` enables only the PyO3
and NumPy host dependencies. `accelerate` selects Apple's native GEMM on macOS;
all other native platforms use `matrixmultiply`. Building without default
features requires neither Python nor Accelerate. Wasm with `simd128` uses a
bounded four-lane dot-product GEMM, with the same GRU equations and layer order.
Changing GEMM backend may produce normal float32 rounding differences; model
and snapshot protocol identities do not change.

```sh
# Check the portable engine without Python, or build the intended Python host.
cargo check --manifest-path native/cognitive-core/Cargo.toml --no-default-features
python native/cognitive-core/build_extension.py --output-dir .
# Browser build instructions and ABI live in native/resident-runtime/README.md.
```

A joined deterministic nonzero-weight probe executed two residents for fourteen
sampled ticks through native `matrixmultiply` and Node Wasm SIMD. Maximum command
and GRU-state differences were `2.98e-8` and `1.79e-7`; selected candidates, goal
slots and private RNG matched. It also exercised native-to-Wasm checkpoint
continuation, exact Wasm replay, pending receipt rejection, atomic invalid
restore, stable output copies and cohort expansion. This establishes portable
mechanism execution, not trained motor competence or full-brain browser speed.
