# Native body-bound sensorium

The current sensorium is `rich-body-v1`. It is a persistent Rust/PyO3 cohort bound to one articulated MuJoCo model. At bind time it resolves and copies every resident's thorax and head geom IDs, builds the local angular templates, and allocates reusable direction, distance, geom-ID, and output scratch. A sensory sample crosses Python once for the cohort, releases the GIL, reads current head poses and materials from MuJoCo, transforms the complete ray set in native code, and calls `mj_multiRay` once per resident.

There is one current profile. A habitat that omits `sensorium` receives this identity before model construction; an explicit selector must provide all three fields:

```json
{
  "frame": "body-v1",
  "profile": "rich-body-v1",
  "profile_sha256": "c71380718ba5535dbaebdeaf8aa2e88cc45cf218312a03e13507877f02a5554e"
}
```

That exact identity participates in the model signature, training-profile hash, and world snapshot. Restore reconstructs the same profile and rejects a changed profile or hash. Earlier running processes retain their loaded code and frozen files; the current module has no old-frame or 80-ray selector.

## Physical camera and arrays

The lens sits 0.004 m beyond the front x radius of the current physical head geom. Directions use the head's full MuJoCo rotation, so yaw, pitch, roll, inversion, and parent-body motion affect the observation. Gaze composes as

```text
pitch = clamp(0.62 * body.gaze_pitch + ray_pitch_offset, -1.15, 1.15)
```

No resident body is excluded from collision. The front-mounted lens prevents the head behind it from hiding the forward view, while antennae, legs, other organisms, objects, and scene geometry can occlude it.

Each resident has 1,024 measured rays:

- `peripheral[8,32,4]`: elevation centers from -50° through +50° and azimuth centers from -100° through +100°;
- `foveal[24,32,4]`: elevation centers from -22° through +22° and azimuth centers from -30° through +30°;
- component order `[red, green, blue, proximity]`;
- raster order elevation, azimuth, component; peripheral precedes foveal in the packed row.

`world.rich_retina_batch()` returns a read-only, resident-major, C-contiguous `float32[B,4096]` view. `RICH_CHANNEL_NAMES` provides all 4,096 names in that order; their newline-delimited SHA-256 is `b4c6b328116d820143e16ee922ccffd7b950dbe008efc580ad93056e01349bfa`. `world.sense(id)["rich_retina"]` exposes native NumPy views for the two declared raster shapes. `serialize_rich_retina()` is the explicit conversion to nested lists for a JSON or display boundary.

A miss is four zeros. A hit uses the live material or geom RGB multiplied by `0.45 + 0.55 * illumination`, capped at one. Proximity is `max(0, 1 - distance / 3.2 m)`. The grid, lens, transduction, and channel order are engineered sensors over real MuJoCo geometry; they are not biological ommatidial measurements.

## Canonical 351-channel projection

The current `retinal-v2` MaleCNS port still contains 320 ordered 5×16 retinal channels and 31 other body-local channels. It performs no additional collision queries. Native code area-pools the measured peripheral raster:

- adjacent azimuth pairs `[2j, 2j+1]` produce the 16 output columns;
- peripheral elevation groups `[0:2]`, `[2:3]`, `[3:5]`, `[5:6]`, and `[6:8]` produce the five rows;
- each output component is the unweighted mean of its two or four source rays.

The resulting `retina3d[5,16,4]` drives the unchanged 351-column sparse anatomical map. Its physical source semantics and spec identity changed, so the current files are [retinal-v2.json](../data/ports/retinal-v2.json) and `retinal-v2-maps.npz`. The larger visual learner consumes `[B,4096]` directly as a separate declared input rather than pretending the full MaleCNS mapping accepts those channels.

## Joined physical evidence

The final isolated hbox build sampled the actual three-resident hollow garden after twelve articulated physical steps with gaze pitches `-0.72`, `0`, and `+0.72`. It returned C-contiguous `float32[3,4096]`, with 2,724–3,912 nonzero channels per resident. An independent NumPy formulation of the declared coarse pool matched the native `retina3d` exactly (`max_abs = 0`). Mutating live MuJoCo material RGB changed 2,429 red channels on the next refreshed sample without rebinding. Distinct gaze pitches changed 3,118–4,096 packed channels, and a matched world snapshot/restore reproduced the packed array exactly.

The single joined capture and receipt are stored on hbox at:

```text
/tank/chreatures/runs/sensorium/rich-body-v1/receipt.json
/tank/chreatures/runs/sensorium/rich-body-v1/three-resident-world.png
/tank/chreatures/runs/sensorium/rich-body-v1/three-resident-rich-retina.png
```

The retinal image lays out peripheral RGB, peripheral proximity, foveal RGB, and foveal proximity for each resident. The retinal PNG SHA-256 is `b4c1bea3754fbc1ff460d86239db5dd2c746263b6f2bec9787da5f0b161fece2`; the headless MuJoCo world image SHA-256 is `89ed277f53f871bdfaa4267aba3168cf8cb1e603febfb374b508579738d06238`.

## Use

```python
from chreatures.physical_batch import FastArticulatedSensoriumWorld
from chreatures.sensorium import encode_rich_physical_senses

world = FastArticulatedSensoriumWorld(seed=7)
packed_batch = world.rich_retina_batch(refresh=True)  # [resident, 4096]

sense = world.sense(world.bodies[0].id)
names, packed_row = encode_rich_physical_senses(sense)
```

The canonical encoder remains independently available through `chreatures.neural_ports.encode_physical_senses`; under `retinal-v2` its 5×16 retina is the declared native pool above.

MuJoCo's authoritative batch-ray declaration is in its [C API header](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mujoco.h). The implementation here calls that API through `native/world-kernels/src/sensorium_shim.c`; it does not reconstruct collision geometry in Rust or Python.
