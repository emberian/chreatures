# Physical execution throughput

`chreatures.physical_batch.FastArticulatedSensoriumWorld` is an opt-in execution
backend for persistent process-owned training worlds. It uses the same MuJoCo
model, `MjData`, timestep, substeps, collision/contact operations, controls,
physiology, RNG, delayed events, sensor equations and snapshot representation as
`ArticulatedSensoriumWorld`.

The fast path changes three measured Python costs:

- It evaluates the twelve hip/knee servo equations in NumPy arrays using cached
  MuJoCo qpos/dof addresses. Forces are still written to `qfrc_applied` before
  each unchanged `mj_step`.
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
