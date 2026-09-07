# In-tab articulated physical world

Implemented 2026-09-07 in `native/browser-world/`. The current browser epoch is
`mujoco-3.12.0-wasm-browser-epoch-1`. This is a newly authored life, not a restore
or reinterpretation of a frozen native resident.

## Decision and author implementation

Use the canonical Google DeepMind **@mujoco/mujoco 3.12.0** Wasm engine for contact
physics, and a separate Rust/Wasm module for recurring body mechanics, sensory
transduction, physiology and physical fields. JavaScript transports packed state
between those two compiled modules and the worker. No Python is in the recurring
browser loop. The Python fixture exporter only authors a fresh model offline.

Consequential primary sources:

- [Canonical MuJoCo Wasm bindings](https://github.com/google-deepmind/mujoco/tree/3.12.0/wasm)
  provide precompiled single-thread Wasm and the maintained engine API. This removes
  the need to replace the existing MuJoCo body with Rapier joints and different
  contact dynamics. The single-thread build avoids a SharedArrayBuffer requirement.
- [MuJoCo API header, pinned 3.12.0](https://github.com/google-deepmind/mujoco/blob/3.12.0/include/mujoco/mujoco.h)
  supplies `mj_multiRay`, body-local object velocity, and full integration state
  access. The retina uses the actual physics geometry, not renderer visibility or
  world labels supplied to a controller.
- [Official Wasm memory guidance](https://github.com/google-deepmind/mujoco/blob/3.12.0/wasm/README.md)
  requires explicit disposal of Embind handles. All resident model/data, ray
  buffers, contact vectors and contact handles have explicit lifetimes here.

The installed 3.12.0 type definitions are authoritative for the tested ABI:
`new DoubleBuffer(elementCount)` and `GetView()`. Some prose examples in the
upstream README use a different constructor/accessor spelling. The `eq_active`
array getter also fails in the shipped bindings with an unregistered bool memory
view; topology migration copies that state through `mj_getState`/`mj_setState`
instead. These observations informed the implementation rather than substituting
a speculative untested wrapper.

## Delivered physical and sensory mechanism

`fixtures/garden.xml` is exported through the current `ArticulatedWorld` model
builder, not a new flat avatar. Three founders each have a free six-DOF trunk,
six articulated legs, twelve driven hinges, tarsus contact friction, head,
abdomen and antenna geometry. The body is the existing **engineered enlarged
hexapod**, not a claim of anatomically reconstructed fly musculature.

The initial garden has 112 collision geometries: ramps, high walk, arch, nook,
hinged mechanisms, suspended and movable objects, food carriers, stackable
construction boxes and a physical video screen. Source MJCF and optic atlas
SHA256 values are explicit. `export_fixture.py` creates fresh initial state and
never advances/restores a resident or connects to the live native world.

Rust receives the existing 12 motor axes in order:

```
thrust yaw gaze_pitch posture grip
signal_low signal_mid signal_high eat release secrete allocate
```

Thrust/yaw drive the existing tripod stance/return equations and bounded joint
PD torques; locomotion comes from MuJoCo feet/terrain contacts. No trunk position
or resource-seeking controller is supplied. A bounded posture reflex, optical
gaze axis, local-reach grip spring with equal/opposite body forces, ingestion,
three-band emission, secretion and developmental allocation are explicit body
mechanisms. The grip spring and simplified physiological coefficients are part of
this new browser epoch and are not asserted to exactly reproduce native lives.

The policy boundary is only `sample()`:

| Array | Resident-major layout | Meaning |
| --- | --- | --- |
| `optic` | `Float32Array[B*1771*3]` | Anatomical-order sampled RGB |
| `body` | `Float32Array[B*43]` | Body-local nonvisual afferents |

The imported atlas contains 879 left and 892 right sites. Actual atlas membership
and supported-site masks are preserved. Viewing angles remain the explicitly
engineered affine axial-hex calibration from `physical_batch.py`. Two eye
origins move with the physical head and bilateral rays undergo its full 3D
rotation plus the engineered gaze axis. MuJoCo returns nearest intersections;
Rust samples geometry RGB or the physical screen's texture at that hit. Blockers,
head/body movement, range and screen sidedness therefore change retinal activity.
No screen frame is passed to the CNS directly. Unsupported anatomical sites stay
zero. Range is 3.2 m.

The 43-channel order matches the current nonvisual transduction contract:
odor6; signed linear-velocity opponents6; angular-velocity opponents6; contact
normal opponents6; contact count1; coarse touch2; sound3; overhead shade1;
interoception12. Velocities and contact normals are body-local. The two coarse
touch channels currently share whole-body contact presence; they are not claimed
to be separate measured antennal touch transducers. Shade uses a real overhead
physics ray. Interoception order is unchanged, including signed speed and turn.

Rust owns each resident's physiology, grip, gaze and pending work; resource
quantities and finite growth reservoirs; bounded delayed acoustic wave emissions;
spatial odor marks with spreading/decay; world clock; private RNG; and an opaque
CNS-derived memory checkpoint. Sound travels at an explicit engineered 30 m/s in
this enlarged world. Current odor fields are radial kernels and **do not yet use
barrier-aware diffusion**. Resource collision volumes shrink/regrow from Rust
quantity; inertial carrier mass is fixed. These are coupled browser-epoch rules,
not the complete native regional-material, inherited-acclimation or lifecycle
engine. No evidence here establishes browser mating, birth or learned motor
competence.

## Physical outside interaction and construction

`queueVisitorForce(entityId, force3)` applies a bounded world-space force during
the next normal 0.05-second tick; it never advances physics on its own. These
forces, like ordinary contacts, reach neural control only through the sensory
path. `visitorSound(position3, amplitude3)` creates a physical sound emitter with
finite propagation. `setScreenFrame(rgb, width, height)` changes an emitting
surface, not policy information.

`await insertObject({position, size, shape, rgba, food, odor})` appends a bounded
physical material object, with optional food and odor components. Maximum 96
entities and 0.5 m half-size. It compiles a private candidate MJCF, checks that
prior model addresses are unchanged, transfers integration state, rejects real
penetration, extends Rust ecology, and swaps once. An asynchronous transaction
whose world advanced during digest computation rejects as stale. The worker
should serialize this operation with ticks. Blocks can then be pushed, gripped,
stacked or used as optical occluders through actual contact geometry. There is no
privileged construction-success flag or object-name channel into cognition.

Observer geometry uses MuJoCo enum types: sphere=2, capsule=3, ellipsoid=4,
cylinder=5, box=6. Sizes are radii/half-lengths/half-axes. Capsule and cylinder
length axes are local Z. Position arrays are world XYZ; rotations are row-major
3x3 matrices. The screen is a box thin in local X, emitting toward local -X;
local -Y is screen right and local +Z is up. Its UV mapping is
`u=.5-localY/(2*halfY), v=.5-localZ/(2*halfZ)`.

## Checkpoint and integration ABI

```javascript
const world = await createBrowserWorld({fixture, xml, seed: 7});
world.setScreenFrame(rgbFloat32, width, height);
const {optic, body} = world.sample(); // send only these to CNS afferent injection
world.advance(cnsDerivedMotorCommands, 0.05);
const display = world.observe();    // observer only
const saved = world.snapshot();
```

A snapshot includes complete `mjSTATE_INTEGRATION`, mutable geometry, Rust state,
physical screen, queued visitor forces and updated XML/fixture. `restore` rejects
wrong engine/model/atlas or incoherent clocks. For a checkpoint containing appended
geometry, create a world using `saved.fixture` and `saved.xml`, then restore it.
Memory in the Rust checkpoint is opaque text supplied by the cognitive owner;
the physics runtime does not derive personal memory from physical observations.
The resident runtime remains responsible for its own learned neural memories and
private RNG/state. A physical tick failure pauses mutation and prevents sampling
or replacing the last coherent checkpoint; it is not silently retried.

## Build, staging and executed verification

From the repository root:

```sh
native/browser-world/build.sh
```

This uses pinned npm dependencies, Rust's `wasm32-unknown-unknown` target and
`wasm-bindgen 0.2.127`, runs the joined headless test, and stages assets locally.
`pkg/` is generated. `stage_site.py` copies the canonical runtime into
`site/live/world-runtime.mjs`, stages the Rust package, official MuJoCo JS/Wasm,
MuJoCo's Apache license and fixture, and emits `physics-assets.json` with exact
versions, sizes and SHA256 hashes. It does not publish or start a server.
The staged runtime resolves MuJoCo by relative local import. Source remains in
`native/browser-world/`; do not edit the generated copy.

Executed `node native/browser-world/test.mjs` exercises real official MuJoCo Wasm
and Rust Wasm without a browser or server. The joined scenario passes:

- Three physical residents, 112 initial geometries, 15939 optical and 129 body
  scalars; finite articulated motion through actual contacts.
- A physical black/white screen changes 240 retinal scalars. An inserted blocker
  changes that screen contrast to zero for Mica's retina.
- Penetrating insertion is rejected while the entire prior snapshot is unchanged.
- Successful topology insertion yields 113 geometries and preserves private
  CNS memory. Pending physical sound/visitor force and the new topology restore
  into another independently owned model/core and replay exactly.
- Full integration state, physiology, RNG and delayed field snapshots replay
  exactly. `serde_json`'s `float_roundtrip` feature was required to retain exact
  emission coordinates through JSON restoration.

The joined test took approximately 0.24 seconds on the laptop in Node for its
1.1 simulated-second scenario and additional sampling/checkpoint instances.
This is a headless module check, **not measured browser frame rate**, and does not
include the full MaleCNS WebGPU or learned resident runtime. Root owns that joined
worker integration. No browser automation, live-world advancement, publication
or driver change was used for this work.
