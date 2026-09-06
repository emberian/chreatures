# Physical execution throughput

`chreatures.physical_batch.FastArticulatedSensoriumWorld` is an opt-in execution
backend for persistent process-owned training worlds. It uses the same MuJoCo
model, `MjData`, timestep, substeps, collision/contact operations, controls,
physiology, RNG, delayed events, sensor equations and snapshot representation as
`ArticulatedSensoriumWorld`.

The fast path changes three measured Python costs:

- It binds immutable joint/body addresses and morphology coefficients into one
  native articulated cohort. Dynamic action and physiology values are packed
  once per tick; one native gait call and one native grip call update all
  residents per substep before each unchanged `mj_step`.
- It computes body illumination once during one `sense` call. Retina, shade and
  the explicit illumination channel previously repeated the same static scene
  query three times at the same `MjData.time`.
- It skips the second identical public pose/object reconstruction in `advance`
  when food components did not change. The first reconstruction still follows
  `mj_forward`; a nutrition event retains the second reconstruction so public
  object amounts update at exactly the same point as the reference backend.

No state is shared across worlds. In particular, each process retains its own
MuJoCo data, neural-independent physics state, RNG, signals, grips, physiology
and future optional acoustic engine.

## Measurement

On 2026-09-05, `scripts/benchmark_physical_batch.py --worlds 4 --steps 150`
ran in a dedicated directory on hbox with the project ROCm environment and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. The active 16-world learner and GPU
service were left untouched.

| Backend | Wall time | Aggregate sense worker time | Aggregate physics worker time | Resident steps/s |
| --- | ---: | ---: | ---: | ---: |
| Reference | 1.575 s | 1.248 s | 4.033 s | 1142.8 |
| Fast | 1.229 s | 1.039 s | 2.870 s | 1464.4 |

This is a 1.281x end-to-end CPU-world speedup for the four-process cohort.
The audit preceding that run applied 30 varied action steps to paired reference
and fast worlds. Maximum absolute qpos/qvel error, 351-channel encoded sensor
error and restored snapshot state error were all exactly `0.0`.

The current learner can adopt the backend between stages by changing only the
worker-local import from `chreatures.sensorium.ArticulatedSensoriumWorld` to
`chreatures.physical_batch.FastArticulatedSensoriumWorld` and constructing and
restoring that class at the existing lines. The pipe protocol and checkpoint
schema do not change. Do not edit a running learner's source tree; switch after
it checkpoints and exits, then exact-resume that checkpoint.

The benchmark includes process spawn and model construction in wall time, so it
is conservative for long-lived training workers. It uses four workers to avoid
starving the active 16-worker run; a stage-boundary 16-worker receipt remains the
right final integration measurement.

## Retired v1 retinal transduction receipt

The first batched sensorium removed a Python loop over 80 results by passing one
`mj_multiRay` result to `transduce_retina`. The receipt below describes that
historical 80-ray implementation; it is not the current retinal contract.

[`sensorium-native-v1.receipt.json`](../data/performance/sensorium-native-v1.receipt.json)
records a focused 2026-09-06 laptop run of 1,200 three-resident batches. Both
the legacy and body-relative retinal frames, including an intervening material
colour mutation, matched the retired Python calculation with zero retinal and
encoded-channel error. The complete `PhysicsWorld.sense` plus 351-channel encode
path increased from 632 to 936 resident samples/s (1.482x). This is a focused
sensor measurement; it excludes physics advance, field transport, neural work,
and runtime communication.

The current [`rich-body-v1` sensorium](SENSORIUM.md) replaces that path with a
persistent Rust cohort, 1,024 physical rays per articulated resident, direct
`float32[B,4096]` output, and a declared native 5x16 area pool for the
`retinal-v2` 351-channel interface. It has no legacy-frame selector or separate
80-ray collision pass.

## Native articulated actuation

[`actuation-cohort-native-v1.receipt.json`](../data/performance/actuation-cohort-native-v1.receipt.json)
records one 24-advance, three-resident heterogeneous-world comparison with
shared chemistry, mobile physiology, recycling, and colony exudation. Neural
execution was excluded. The native cohort matched the research-only scalar
reference exactly for MuJoCo position and velocity, chemical pools, world time,
and developed-part count. Whole scenario time fell from 286.6 ms to 96.9 ms on
the local host, a 2.956x speedup. The kernel preserves gait, external hand,
grip, acoustics, then MuJoCo ordering and computes positive active mechanical
work directly from the forces it applies, avoiding full global force-array
copies for every resident and substep.
The timing scenario did not establish a grip attachment. A joined follow-up
used Mica's active grip on the free `chemical-packet-0`: object pose, all qpos
and qvel values, chemistry, and positive mechanical work matched the scalar
reference exactly (`0.2609207854437227` joules reported by both paths).
