# Native fly world host

This crate is the native MuJoCo 3.12 host for the current fly world. It uses
the same `chreatures-browser-world::WorldCore` as the browser host; the C shim
only owns opaque MuJoCo objects and bulk ABI operations. Physics, contact
sampling, retinal rays, physical illumination, ecology, acoustics, actuation,
growth, and CNS afferents therefore retain one set of Rust equations.

Build with the pinned MuJoCo installation:

```sh
MUJOCO_INCLUDE_DIR=/path/to/mujoco/include \
MUJOCO_LIB_DIR=/path/to/mujoco/lib \
cargo build --release --manifest-path native/fly-world/Cargo.toml
```

Run the newline-framed research bridge:

```sh
native/fly-world/target/release/chreatures-fly-world \
  --scene native/fly-body/scenes/training-4/world.json --seed 23
```

The bridge accepts the existing `sample`, `advance`, and `close` messages used
by `research/fly_learning/world_bridge.mjs`. It also accepts `stimulus`,
`visitor_force`, `routes`, `snapshot`, and `restore`. Numeric arrays use the
same `{dtype,length,base64}` little-endian envelope. Failed physical mutations
pause the host until a coherent checkpoint is restored.

Policy consumers receive only `optic` and `body`. The remaining `sample`
fields are research observer data for training targets and receipts.
