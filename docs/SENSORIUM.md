# Batched native sensorium

`chreatures.sensorium` accelerates one measured hot path: the 3-D retinal collision queries. It provides `SensoriumWorld` and `ArticulatedSensoriumWorld`, API-compatible alternatives to the corresponding existing worlds. No default runtime is switched by this module.

The retinal coordinate system is explicitly versioned in the habitat spec. An absent selector retains the historical behavior:

```json
{"sensorium": {"frame": "body-v1"}}
```

`legacy-world-v0` uses public heading for horizontal azimuth, world-z for elevation, an origin at root z plus 0.035 m, and excludes geoms attached directly to the root body. This is also the behavior when `sensorium` is absent, preserving old lives and snapshots exactly.

`body-v1` places the lens just beyond the physical head geom, or just beyond the crawler shell when no head exists. It constructs gaze and all retinal directions in local body coordinates and rotates them by the complete MuJoCo trunk/head transform. Roll, pitch, inversion, and yaw therefore rotate vision with the organism. The origin and center gaze basis match `RetinalRenderer.camera_pose`; the sparse retina retains its established 150-degree horizontal fan and five elevation samples, while the image renderer retains its perspective projection and configured vertical field of view.

The body-v1 lens is outside the head and every retinal ray points into the forward hemisphere. It deliberately passes `bodyexclude=-1`: antennae and articulated limbs can occlude rays, matching the visible body geometry, while the trunk behind the lens does not hide the scene. `view()["sensorium"]` publishes the selected frame, self-occlusion rule, and each resident's lens origin/forward/up/right basis.

## Measurement and choice

The benchmark uses the actual hollow-garden model, all three residents, all public `sense()` fields, and the normal 5×16 retina plus 16-element central visual fan. On local MuJoCo 3.12.0, five runs of 400 complete sensory passes produced these medians:

| Workload | Existing scalar rays | Batched rays | Speedup | Saved per pass |
| --- | ---: | ---: | ---: | ---: |
| Three crawler residents | 1.195 ms | 0.595 ms | 2.01× | 0.600 ms |
| Three articulated residents | 1.463 ms | 0.812 ms | 1.80× | 0.651 ms |

All legacy-mode `vision` and `retina3d` values matched the scalar implementation exactly in the comparison (`max_output_error = 0.0`). The same audit measured odor plus sound for all residents at 0.141 ms, an idle three-resident physics step at 2.052 ms, and a saturated 384-record/48-feature personal-memory recall at 0.612 ms.

Run the measurement with:

```bash
.venv/bin/python scripts/benchmark_sensorium.py --iterations 400 --repeats 5
```

The memory result deserves a later storage-layout change in the cognition owner: today each recall rebuilds dense arrays from a list of dictionaries. Moving only its distance arithmetic to an extension would leave that conversion cost in place. The full neural service already batches residents into one channel array, performs sparse matrix operations on its Torch device, and returns a small feature vector per resident. A local PyO3 rewrite of that boundary would compete with the existing GPU kernels without evidence that Python arithmetic dominates it.

## Native operation

The original retina calls `mj_ray` once for each direction in Python. The new path constructs the same 80 directions as one contiguous array and invokes MuJoCo's `mj_multiRay` once. MuJoCo evaluates the batch in compiled C against its live model and collision structures. The central 16-ray `vision` result reuses the center band from the same batch, avoiding duplicate intersections during `sense()`.

The postprocessing preserves the existing contract:

- same 150-degree horizontal fan and five elevation offsets;
- same legacy body exclusion and inclusion of static geometry;
- same 3.2 m range cutoff;
- same material color and illumination adjustment;
- same proximity encoding and nested Python-list output.

MuJoCo describes `mj_multiRay` as intersecting multiple rays from a common point, with semantics corresponding to `mj_ray`; its Python binding checks a `3*nray` direction buffer and calls the native function directly. See the official [C API declaration](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mujoco.h) and [Python binding implementation](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/functions.cc).

## Why this is not a Rust crate

Rust/PyO3 is useful when Rust owns a coherent computation and its data can cross the boundary in a small number of contiguous buffers. Here, MuJoCo already owns the geometry, transforms, broad phase, and exact collision routines in native memory. A Rust wrapper would still call MuJoCo, require access to its live pointers, and add another extension build and foreign-function boundary. It would not replace the work that dominates the measurement.

The selected implementation therefore uses the engine's existing compiled batch API. A later Rust sensorium is justified if it owns a substantial independent operation, such as a persistent spatial field index with batched source updates and receptor queries. That should be measured after entity/source counts grow; the present odor and sound workload is too small to justify such a subsystem.

## Use

```python
from chreatures.sensorium import ArticulatedSensoriumWorld, SensoriumWorld

crawler = SensoriumWorld(seed=7)
hexapod = ArticulatedSensoriumWorld(seed=7)
```

Both classes retain the inherited `sense`, `advance`, `view`, `snapshot`, and `restore` boundaries. `native_retina(world, body)` is also exposed for a world owner who wants to integrate the batch directly later.

The selector lives inside `world.spec`, which is already covered by the compiled model signature and serialized in the world snapshot. A body-v1 snapshot restores body-v1 exactly. Invalid selector names fail construction rather than falling back, and old snapshots without a selector restore as legacy-world-v0.
