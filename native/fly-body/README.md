# Anatomical fly body import

This directory converts one pinned author model into a portable MuJoCo fixture:
FlyGym/NeuroMechFly v2.1.0 at commit
`ca65a510c2afe6ac61c51df4f274c8d190c2f95f`. The body uses the bundled
`simplified_max2000faces` micro-CT-derived meshes and the author viewer's
YAW-PITCH-ROLL joint ordering. Python is used only to compose and export assets.
Native and browser simulation load the resulting MJCF directly.

The fixture contains 69 named anatomical segments, 126 hinge coordinates, 84
position servos, six adhesion actuators, six per-leg ground-contact sensors,
two calibrated eye-camera anchors, and the author neutral pose. It contains no
gait controller, behavior selector, reward, or destination logic.

The active joint bank is an explicit Chreatures selection over the author axes:
walking42 through each first tarsomere, head3, pedicels6, rostrum/haustellum6,
abdomen15, wings6, and halteres6. Eye6, funiculus/arista12, and distal-tarsus24
axes remain passive. This selection preserves head, sensing-organ, feeding,
abdominal, and flight-apparatus capability without pretending that position
servos are muscles.

## Rebuild the fixture

Use an isolated environment with the pinned FlyGym checkout and its declared
MuJoCo 3.9 dependency:

```sh
python tools/export_neuromechfly.py \
  --flygym-source /path/to/flygym-ca65a510 \
  --output assets/neuromechfly-2.1.0-ca65a510-ypr
```

The exporter refuses another Git revision. It copies the flattened MJCF, all
referenced meshes, author configuration and license, and writes `schema.json`
with exact order, addresses, units, ranges, hashes, and model limitations.

## Compose a fly-scale ecology scene

`tools/compose_ecology_scene.py` is a build-time compiler over the flattened
fixture. It can replicate one to sixteen independently prefixed flies, retains
the exact 126-axis and 90-actuator order for every resident, and adds leaf and
curved-bark platforms, a segmented stem, climbable ramps, moist patches, and
movable grains in model millimetres. Contact filtering allows fly-terrain,
fly-grain, grain-grain, and inter-fly contact while suppressing self-collision.
The actual haustellum mesh is contact-enabled for the native mouth-tip law.
Six zero-mass collision proxies derived from the pinned pedicel, funiculus, and
arista meshes make antennal contact reciprocal while preserving author segment
mass and inertia. Same-resident antenna contact is excluded; material and
conspecific contact enters only the existing BODY807 joint/load/contact path.
Compiled movable entities carry their exact free-joint qpos/dof addresses for
observer diagnostics; those IDs and coordinates are not controller inputs.

```sh
python tools/compose_ecology_scene.py \
  --output /path/to/scene.xml \
  --manifest /path/to/physics.json \
  --residents 4 --habitat-plan /path/to/habitat-plan.json --startup-steps 100
```

The current composer requires a deterministic `chreatures.fly-habitat-plan.v1`
from `../fly-habitat`; the former fixed hand-placed environment is no longer a
production path. The manifest is generated from the compiled MuJoCo model. It records numeric
body, joint, DOF, actuator, geom, mesh, site, sensor, camera, and keyframe IDs;
neutral controls; a piecewise normalized servo map; VFS mesh hashes; and the
observer-only information boundary. The checked current output is
`../browser-world/fixtures/fly-ecology/`. The 1,771 retinal directions preserve
the optic-atlas site order and use the explicitly engineered equidistant
projection frozen for CNS V4 from the author camera's 157-degree vertical FOV.
They are not measured ommatidial angles.

Two deterministic compiled outputs are retained. The public two-resident scene
is `../browser-world/fixtures/fly-ecology/`. The four-resident collection scene
is `scenes/training-4/`; its XML SHA-256 is
`c880facc75d6a2ac3bcbfa18ae30320f5a6110a77176268157963ba3889fbb6a`.
Both use 100 physics substeps per 0.01-second CNS control interval.

## Sealed author walking reference

`tools/export_author_step_bank.py` evaluates the pinned author's
`FlyBodyPreprogrammedSteps` at 2,048 evenly spaced periodic phases. The retained
`assets/author-step-bank-v1/trajectory-bank.npz` contains exact walking42 target
radians, the same targets normalized through the current fixture neutral and
control ranges, and six binary adhesion channels. Its SHA-256 is
`97a5615c5e59baf86f29293fc06c2d92eee2bd8b84ea1abe42a378fc394a929b`;
periodic linear interpolation differs from the author spline by at most
`3.611e-5` radians over 8,192 deterministic probes.

This bank is an offline teacher. The source was fitted for FlyBody from a short
NeuroMechFly v1 ball-walking clip, so applying it to the imported NeuroMechFly
body is a cross-morphology transfer through shared joint identifiers. It is not
a runtime controller or evidence of learned CNS control. A two-second B4 replay
and its raw M90/contact trace are retained under
`receipts/author-teacher-b4/`. The forward command moved 3.58 mm in its initial
body-forward direction and the stop command drifted 0.000137 mm. The two simple
asymmetric-amplitude probes both yawed in the same direction, so this replay
does not establish reliable directional turning.

`SupportedAuthorContinuation` in `tools/replay_author_teacher.py` provides a
narrower offline use of the same sealed bank after a learner command. It accepts
only a physically upright state with at least three contacting feet. At
takeover it fits one shared tripod phase and a bounded amplitude against both
the actual walking-joint angles and the last physical servo targets, in radians
scaled by each joint's control span. Candidate phases that would immediately
release a contacting foot are penalized. It then slews all 84 active servo
targets toward the fitted reference at no more than 0.04 radians per 100 Hz
tick; six unitless adhesion commands use a separate 0.25-per-tick bridge.
The same helper supplies continuation, stop, left/right asymmetric amplitude,
and neutral-leg antenna modes. A requested author amplitude is reached from the
fitted entry amplitude over 64 actually delivered supported ticks. Phase is not
reset between modes and continues through supported stop/antenna intervals.
Counterfactual teacher targets use `advance_state=False`, so a learner-controlled
tick cannot advance hidden teacher phase or ramp state. `neutral_command` shares
the physical slew and normalization path and remains available outside the
author support domain for cold support and failed-life correction.

`tools/demonstrate_supported_author_continuation.py` records paired physical
research branches. The first retains an actual CNS action block and its failed
outcome. The sibling restores the exact last support-gated MuJoCo snapshot and
starts the fitted continuation. The restored snapshot contains no private CNS
state; a future collector must recompute private CNS history from its checkpoint
and preserve the matching recurrent-state cadence (currently seven private CNS
state tensors) before presenting the sibling as training evidence. States
outside the support gate remain failed evidence and are never described as
self-righting.

## Standalone startup

The native validator links to the MuJoCo library supplied by the project Python
environment. It loads the real model, resets its neutral keyframe, advances it,
and checks its dimensions and finite state:

```sh
MUJOCO_INCLUDE_DIR=/path/to/mujoco/include \
MUJOCO_LIB_DIR=/path/to/mujoco/lib \
cargo run --release -- \
  assets/neuromechfly-2.1.0-ca65a510-ypr/model/model.xml 100
```

The WebAssembly check uses the already-pinned MuJoCo 3.12 package from
`native/browser-world` and loads the same XML and STL bytes into its virtual
filesystem:

```sh
node wasm_startup.mjs \
  assets/neuromechfly-2.1.0-ca65a510-ypr/model/model.xml 100
```

## Scientific boundary

NeuroMechFly geometry comes from a micro-CT scan of an adult female fly, with
some author adjustments including the antennae. The current connectome is male,
so pairing them is an explicit inter-animal and sex transfer. The model's joint
axes are anatomical assumptions, while the servos, passive parameters, force
bounds, and adhesion are engineered simulation mechanisms. NeuroMechFly does
not identify its position servos with individual muscles, and it supplies no anatomical
joint limits. The source declares millimetres, seconds, and radians, but does
not explicitly name its mass base unit; the rigging entries sum to 0.0009997844
model mass units, numerically 0.9997844 mg only if those units are grams. SI
world conversion must keep that uncertainty explicit. Any MaleCNS motor-neuron mapping must retain a separate evidence
grade and address exact joint IDs rather than infer muscle identity from side or
segment alone.

Third-party material under `assets/**/author-source` and generated model assets
remain Apache-2.0, copyright 2023-2026 The NeuroMechFly v2 Authors. Chreatures
pipeline and validation code is AGPL-3.0-or-later.
