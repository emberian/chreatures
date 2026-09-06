# Native world kernels

This PyO3 extension holds whole-world numerical kernels that are costly in the
Python simulation loop. It is required by the 3D world runtime; import or ABI
errors stop world construction rather than silently selecting another engine.

Build it with the same interpreter that will import it:

```bash
.venv/bin/python native/world-kernels/build_extension.py --output-dir .
```

The build script discovers the MuJoCo headers and shared library bundled with
that interpreter's `mujoco` wheel, pins `PYO3_PYTHON`, and installs an extension
with the interpreter's exact ABI suffix. On macOS it also replaces the wheel's
framework-style install name with the actual bundled library path. At runtime,
`ContactBatch` compares `mjVERSION_HEADER` with `mj_version()` before accepting
MuJoCo object addresses.

The builder honors Cargo's configured target directory, including
`CARGO_TARGET_DIR`, and stages the completed extension before an atomic rename.
It never rewrites a library inode already mapped by a running resident. Use
`--output-dir /path/to/isolated-build` for research or a pending migration.

`ContactBatch.evaluate` crosses the FFI boundary once for all contacts in a
MuJoCo substep, whether or not an acoustic engine is attached. A small C shim
performs the MuJoCo ABI calls and fixed-vector contact arithmetic; Rust owns
reusable capacity and returns each result as one contiguous NumPy allocation.
Model and data pointers are borrowed only for the duration of the call, so model
rebuilds cannot leave cached native pointers.

`AcousticEngine` owns finite oscillator energy, cooldowns, transduction,
radiation and their ledgers. Hinge loads and contact work enter through batches;
native pose transforms and MuJoCo visibility rays turn oscillator state into
body-local three-tone input. Bindings are rebuilt when physical topology changes.
The C bridge validates every index before writing forces. Hinge extraction stays
at the physical substep, while cooldown and radiative decay remain at the
completed-world interval. The seven-resonator Living Reef comparison preserved
physical trajectories and oscillator state exactly, with sound differences below
`2.8e-17`. Complete-path timing varied from 0.94× to 1.08× on the busy laptop;
no stable speedup is established by this port.

`TransportSolver` owns reusable flux and change arrays for a complete
multichannel chemical grid. Each call processes conservative diffusion and
upwind advection across all x/y/z faces, including solids, heterogeneous
permeability and moving membrane factors. Explicit shape validation protects
the bounds-free numerical loop. Arithmetic order follows the archived NumPy
face passes without fast-math reassociation. The reference equation remains
only in `scripts/probe_native_transport.py`.

Executed checks on a 48×32×14 three-channel field produced bit-exact reference
concentrations through transport, source/sink/decay composition and restored
continuation, with and without membranes. On the busy M2 Max, median complete
field advance improved from 16.74 to 11.88 ms without membranes and 17.55 to
12.89 ms with them. The hbox contact-only substitution improved a complete
rich-world measurement from 23.74 to 29.14 steps/s. These are separate measured
comparisons, not a claimed combined speedup.

`BiosphereTissue` keeps a dense structural matrix across ticks, validates colony
totals, applies structural turnover, and publishes the resource dictionaries
used by checkpoints. Growth, removal and restore explicitly rebind that matrix.
No Python turnover implementation remains in production. The 2,789-part
[measurement](../../data/performance/biosphere-tissue-native-v1.receipt.json)
preserved full world and Biosphere continuation exactly.

`MotorRuntime` owns packed immutable inherited parameters, accepting batches of
1–256 residents for policy/value inference, predictions, projection, variance
and recurrent context updates. Apple builds use Accelerate SGEMM; other targets
currently use explicit float32 accumulation. Python owns boundary validation,
normalization, private RNG and action timing. Arithmetic differs slightly from
the previous NumPy path, so v2 motor snapshots carry an execution identity and
reject an unannounced arithmetic transition. See
[motor deployment](../../docs/MOTOR_INHERITANCE.md) for parity and performance.
