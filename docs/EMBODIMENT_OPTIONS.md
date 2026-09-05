# Embodiment options for a three-dimensional habitat

## Recommendation

Use MuJoCo as the authoritative physical world and keep the organism's interface
at continuous motor and sensor channels. For the always-on multi-resident habitat,
start with a small engineered articulated body in the same MuJoCo scene. Keep the
full NeuroMechFly body as a compatible laboratory mode for one or two residents,
kinematic validation, and later optimization.

This preserves the parts that matter for discovery: gravity, momentum, friction,
occlusion, pushing, climbing, collisions, body configuration, and consequences of
other residents' motion. The brain should emit joint/adhesion activity or a generic
low-dimensional temporal motor basis. It should never choose among named commands
such as `forage`, `court`, `fight`, or `go to food`. Those labels can be used after
the fact to analyze behavior, not as the policy's action vocabulary.

The full model works and is much richer than a sprite, but the current two-fly CPU
cost is too high for the default interactive loop without further optimization.
This conclusion comes from an actual pinned headless run, rather than package
claims alone.

## What was tested

[`scripts/probe_flygym.py`](../scripts/probe_flygym.py) pins and checks FlyGym
`2.1.0`, tag/commit `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`, with MuJoCo
`3.9.0` and Python `3.12.7`. FlyGym declares Apache-2.0 in its
[`pyproject.toml`](https://github.com/NeLy-EPFL/flygym/blob/v2.1.0/pyproject.toml)
and repository license. The isolated installation is at
`hbox:/tank/chreatures/flygym-probe`; it did not modify a global or GPU environment.

The probe builds one `FlatGroundWorld` containing two independently named
`NeuroMechFly` instances and a free-jointed sphere. Each fly has all biological
joint axes, the neutral pose, 42 low-level leg position actuators, six independent
adhesion controls, and six per-leg ground-contact sensors. The probe applies a
small deterministic joint perturbation with no goal or behavior selector. The ball
receives an initial physical velocity so object collision can be observed.

The compiled no-vision scene has 140 bodies, 140 geoms, 273 position coordinates,
270 velocity coordinates, 96 actuators, and 12 contact sensors. Each fly exposes:

| Channel | Observed shape/count |
| --- | ---: |
| Body positions and quaternions | 69 body segments |
| Joint angles | 126 |
| Joint velocities | 126 |
| Position targets | 42 |
| Adhesion inputs | 6 |
| Ground contact found | `(6,)` |
| Contact force, torque, position, normal, tangent | five `(6, 3)` arrays |
| Optional compound-eye output | `(2, 721, 2)` float32 per fly |

The sphere moved from `(0, 0, 0.52)` mm to approximately
`(0.430, 0.006, 0.388)` mm in the 0.1 s vision run and contacted a fly on 875 of
1,000 physics steps. Cross-fly body collision was compiled, although the neutral
open-loop probe did not make the separated flies contact one another. The same
contact machinery supports crowding, bumping, object carrying/pushing, and body
contact if resident motion brings geoms together.

### Measured hbox cost

These are single-process observations, useful for sizing rather than a general
benchmark:

| Test | Result |
| --- | ---: |
| Scene construction | 0.34-0.36 s |
| Bare two-fly physics | 6,003-6,258 steps/s |
| Bare real-time factor at 0.1 ms timestep | 0.60-0.63x |
| Low-level control + contact read each step | 3,016-3,239 steps/s |
| Control/read real-time factor | 0.30-0.32x |
| Resident memory, no rendered eyes | 337-417 MiB observed |
| Resident memory with rendered eyes | about 535 MiB |
| First two-fly EGL ommatidia query | 2.68 s |
| Warm two-fly ommatidia query | 8.2 ms |

Replacing all mesh collision geoms with FlyGym's fitted capsules did not improve
throughput in the single 5,000-step comparison (0.57x bare, 0.28x with reads;
362 MiB observed peak memory). This one run is not enough to treat the timing or
memory differences as stable.

The FlyGym package itself occupied 3.1 MiB in site-packages. Bundled assets were
2.3 MiB, including 2.0 MiB of default simplified NeuroMechFly meshes. The complete
dedicated environment was 418 MiB because it includes MuJoCo, NumPy, SciPy,
Matplotlib, Numba, and other dependencies. FlyBody and FlyMimic's larger meshes
are fetched lazily in 2.1.0; the probe did not fetch them. The default NeuroMechFly
is already the appropriate body for this test. FlyMimic is experimental and only
its left front leg is muscle-driven, while FlyBody adds detail and cost we do not
currently need.

## Current API and practical gaps

FlyGym 2.1 replaced its previous backend with MuJoCo's native `MjSpec`. Its
official [`BaseWorld` API](https://neuromechfly.org/api_reference/flygym/compose/world/base_world/)
says a world can contain multiple flies, and `add_fly` attaches each named fly at
an independent 3D position and quaternion. The
[`Simulation` API](https://neuromechfly.org/api_reference/flygym/simulation/)
accepts actuator arrays by fly name and returns joint, body, force, contact, and
vision data lazily. The official
[`advanced composition tutorial`](https://neuromechfly.org/tutorials/1b_advanced_model_composition/)
recommends procedural `MjSpec` edits for custom objects instead of hand-editing
XML. The probe uses that path for the movable ball and collision pairs.

Several details need local adapter code:

- In v2.1.0, adding ground-contact sensors for a second fly fails because the
  built-in helper reuses names such as `ground_contact_lf_leg` and resets its
  per-fly lookup. The probe's `MultiFlyFlatGroundWorld` prefixes those names and
  preserves both lookups. Multi-fly composition itself works after this small fix.
- NeuroMechFly geoms use `contype=0` and `conaffinity=0`; ground contact is created
  through explicit geom pairs. Movable-object and social collisions must likewise
  be declared. The probe adds 138 ball/fly pairs and 441 coarse cross-fly pairs.
  A habitat adapter should generate these systematically and avoid a quadratic
  all-segment/all-segment expansion when many residents are present.
- The stock complex-terrain world has multiple ground geoms and explicitly does
  not create the six per-leg contact sensors. It recommends raw body-segment
  contact-force queries instead. A custom terrarium should expose raw external
  contacts, then aggregate them into stable left/right/leg/body channels.
- The 2.1 source has rendered compound-eye support but no current olfactory world
  or odor sensor implementation; only a legacy v1 configuration mentions
  olfaction. Odor, sound, humidity, temperature, taste, internal physiology, and
  chemical trails therefore remain habitat-side fields sampled at antenna/mouth
  coordinates. They should enter through explicit sensor channels rather than
  goal logic.
- Adding arbitrary new bodies changes `MjModel` topology and normally requires a
  recompile. For caregiver placement during a live run, preallocate a bounded pool
  of free-jointed spheres, boxes, capsules, and static structures. Activation can
  move an unused body into the scene and enable its collision/visibility. Truly
  novel topology can be a deliberate paused rebuild with state transfer.

[FlyGym's current overview](https://neuromechfly.org/) reports roughly 2x real-time
for CPU and 60x for GPU batches, but those figures do not imply two interacting
flies with Python sensor extraction will meet real time on hbox. The measured
probe is the relevant baseline. GPU MuJoCo Warp is most attractive for large,
fixed-topology developmental batches; interactive worlds with changing objects
and direct browser state still need a CPU path or a carefully preallocated model.

## State and browser rendering

MuJoCo provides `mj_getState` and `mj_setState`; its
[`mjtState` documentation](https://mujoco.readthedocs.io/en/3.9.0/APIreference/APItypes.html#mjtstate)
defines `mjSTATE_INTEGRATION` as full physics, user-controlled quantities, and
warm-start state. The probe captured 2,020 float64 values (16,160 bytes) without
eye cameras, restored them halfway through the run, and replayed the remaining
steps with zero bit difference. With eye cameras the vector was 2,038 values.

That is necessary but not a complete application checkpoint. An exact habitat
snapshot also needs:

- the pinned MJCF/spec or a strong model hash and package versions;
- mutable model fields such as geom collision masks, friction, colors, and active
  object slots, because these live in `MjModel`, not the integration-state vector;
- brain/controller state, all RNGs, physiology, odor/sound fields, delayed events,
  and the authoritative simulation step number;
- renderer sampling state only if frame timing itself is part of continuation.

Exact replay was demonstrated within one compiled model on one host. It should
not be presented as bit-portable across MuJoCo versions, CPU architectures, or
changed model topology.

For the direct 3D browser, Python/MuJoCo remains authoritative. Stream named body
translations and quaternions plus active object transforms at 20-30 Hz. Convert
the small bundled STL meshes to web assets once, or use a designed skin driven by
the same body transforms. The browser interpolates between snapshots and sends
caregiver commands; it does not advance physics. This avoids server-side RGB
rendering for the habitat view. Rendered eye cameras remain private sensory input
and can run less often than the 10 kHz physics solver.

## Full NeuroMechFly versus an engineered body

| Concern | NeuroMechFly 2.1 | Small engineered MuJoCo body |
| --- | --- | --- |
| Anatomy | 69 micro-CT-derived visual/body segments and 126 joint axes | Chosen 8-20 bodies and roughly 12-30 joints |
| Control | 42 leg position targets + six adhesion channels in this probe | Direct torques/targets or generic oscillator coordinates |
| Sensors | Rich joint/body/contact APIs; optional 721-ommatidium eyes | Add only contact, proprioception, rays/cameras actually used |
| Two-resident hbox speed | 0.30-0.63x real time in this probe | Expected much faster; must be measured after design |
| Object/social collisions | Explicit pair-generation adapter required | Collision masks can be designed for ordinary dynamic contact |
| Scientific grounding | Recognizable morphology and published model assumptions | Engineered embodiment with no anatomical-fidelity claim |
| Behavior discovery | High-dimensional and difficult, but unrestricted | Easier search; risk of over-compressing meaningful body dynamics |

NeuroMechFly is based on a real fly scan, but it is still a simulation with fitted
joint, actuator, contact, mass, and adhesion assumptions. The documentation itself
describes adhesion switching as an abstraction because leg release is not fully
understood. Neither body option should be called physiologically faithful.

The engineered body should retain six independently articulated legs, head/antenna
pose, abdomen, contact-rich feet, free 3D orientation, and object-scale mass. A
generic phase/amplitude motor basis can lower control dimension while leaving
coordination, direction, approach, avoidance, signaling, and social patterns to
emerge. Keep raw joint targets available for experiments. Do not bake destination,
reward, gait name, or interaction intent into the primitive.

## Reproducing the probe

```bash
ssh hbox
cd /tank/chreatures/flygym-probe
python3 -m venv venv
venv/bin/pip install \
  'flygym @ git+https://github.com/NeLy-EPFL/flygym.git@v2.1.0'

# Two flies, contacts, movable object, state replay, and timing
venv/bin/python probe_flygym.py --steps 5000 --output result.json

# Also exercise two rendered compound eyes through EGL
MUJOCO_GL=egl venv/bin/python probe_flygym.py \
  --steps 1000 --vision --output result-vision.json
```

The probe is a feasibility instrument, not a controller and not a proposed runtime
implementation. It owns no goals, rewards, navigation rules, or social behaviors.

Primary references: [NeuroMechFly v2 paper](https://doi.org/10.1038/s41592-024-02497-y),
[original NeuroMechFly paper](https://doi.org/10.1038/s41592-022-01466-7),
[FlyGym 2.1 release](https://github.com/NeLy-EPFL/flygym/releases/tag/v2.1.0), and
[MuJoCo model editing](https://mujoco.readthedocs.io/en/3.9.0/python.html#model-editing).
