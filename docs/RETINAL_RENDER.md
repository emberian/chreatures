# Body-local retinal rendering

`chreatures/retinal_render.py` renders the physical hollow garden from a
resident's body pose and saved gaze. It restores the exact MuJoCo snapshot,
dispatching to `PhysicsWorld` or `ArticulatedWorld` from the embedded body
specification, then uses MuJoCo's native offscreen renderer. It does not rebuild
the scene from the public browser view.

The camera origin sits just beyond the body's front surface. For the engineered
hexapod it uses the actual head geom position and orientation; for the ciliary
body it uses the shell's compiled root transform and dimensions. Forward and up
are rotated by the resident's saved gaze pitch. The default 82-degree vertical
field of view is perspective rendered at 512×384.

MuJoCo documents that the two low-level
[`mjvGLCamera`](https://mujoco.readthedocs.io/en/latest/programming/visualization.html#cameras)
records in `mjvScene` directly specify eye position, forward direction, up
direction, and frustum. The renderer writes those fields for both eyes after
`Renderer.update_scene`, with stereo disabled. Labels, coordinate frames, and
overlays are explicitly disabled. The resulting PNG contains only RGB pixels
from the physical scene.

## Restore and render

The helper authenticates a `chreatures-3d-checkpoint-v1` envelope before
restoring its world. It never skips the saved model signature or MuJoCo state
contract. An embedded `articulated_body_spec` selects `ArticulatedWorld`;
otherwise it selects `PhysicsWorld`.

Render without semantic inference:

```sh
.venv/bin/python scripts/observe_fov.py runs/hollow-garden.json \
  --resident pip --output-dir runs/perception/fov
```

The output report marks `experienced: true` because the pixels come from the
resident's saved physical perspective. It marks `observed: false` and
`perception.status: not_requested` until a native perception endpoint actually
processes those pixels.

## Send only the field of view

With a perception endpoint available:

```sh
.venv/bin/python scripts/observe_fov.py runs/hollow-garden.json \
  --perception-url http://127.0.0.1:18775 \
  --dense-count 64 --output-dir runs/perception/resident-fov
```

The outbound request contains:

- an opaque sensor digest;
- checkpoint tick, model time, and capture timestamp;
- `provenance: resident_fov`;
- one base64 PNG and the dense-feature request flag.

It contains no resident name or ID, world coordinates, object IDs, object
kinds, component labels, or world-state archive. The local evidence report may
retain the resident-to-frame association and camera pose for audit, but that
record is not the perception payload and is not injected into the controller.

The report retains uncertain semantic hypotheses exactly as the perception
service returns them. When requested, it retains 64 evenly spaced values from
the 960-dimensional native image vector, along with their source indices and a
SHA-256 of the complete vector. This keeps a reproducible dense trace without
quietly presenting a handcrafted feature as model output.

## Persvati service

The pinned SmolVLM2 snapshot is copied from hbox to
`/home/ember/chreatures/models/SmolVLM2-500M-Video-Instruct` on persvati. Torch
2.10.0+ROCm 7.0 remains in the pre-existing immutable runtime; service-specific
Transformers 4.57.1, torchvision 0.25.0+ROCm 7.0, Pillow 11.3.0, and related
dependencies live separately in
`/home/ember/chreatures/envs/perception-packages`.

Start the endpoint only during a scheduled perception window:

```sh
PYTHONPATH=/home/ember/chreatures/envs/perception-packages:/home/ember/chreatures/perception-src \
PYTORCH_KERNEL_CACHE_PATH=/home/ember/chreatures/cache/pytorch \
HF_HOME=/home/ember/chreatures/cache/huggingface \
/home/ember/kaxsim/.venv7/bin/python \
  /home/ember/chreatures/perception-src/scripts/serve_perception.py \
  --backend smolvlm2 \
  --model-path /home/ember/chreatures/models/SmolVLM2-500M-Video-Instruct \
  --device cuda --dtype float16 --max-new-tokens 128 \
  --max-workers 1 --max-pending 2 --bind 127.0.0.1 --port 8775 \
  --pid-file /home/ember/chreatures/runs/perception/perception.pid
```

Use an SSH tunnel from the physics host:

```sh
ssh -N -L 18775:127.0.0.1:8775 persvati
```

## Measured resident observation

The first native run restored the actively saved `runs/hollow-garden.json`
without changing it or advancing the world. The authenticated source state was
tick 17,399 at model time 869.95 seconds, checkpoint SHA-256
`f500a81d58be982faae8fc5b414b4b0de4402b8b466121d491292790967372d0`.
The concrete restore class was `PhysicsWorld`, with model signature
`48a6732fec0ee59df864a29a539c9f14a623e6e92cfb6ae04a05bd2e3c0fa9ec`.

Pip's actual saved body pose and gaze produced
`runs/perception/resident-fov/pip-17399.png`, a 512×384, 6,226-byte overlay-free
PNG with SHA-256
`9bed638b067de7db504189e9ad93efbae6d60604fade0ccfae0ff899e357e5e4`.
Only this image and the opaque timing envelope went to persvati.

The pinned native VLM completed in 7.280 seconds and returned:

```json
{
  "status": "ok",
  "scene_summary": "Visible hypotheses: tree",
  "objects": [
    {"id": "o1", "label": "tree", "confidence": 0.5, "uncertainty": 0.5}
  ],
  "affordances": [
    {"object_id": "o1", "action": "stand", "confidence": 0.5,
     "uncertainty": 0.5}
  ],
  "uncertainty_kind": "complement of model-self-reported confidence; uncalibrated"
}
```

The dark frame shows tall garden structures, so `tree` is a weak and likely
misleading semantic hypothesis. It remains tagged as model output rather than
world truth. The dense path returned 960 native dimensions; the report retains
64 values and indices, and records complete-vector SHA-256
`e7ebb411995c11cbf3cc91bba6b0f1cd6baeeac5d8b27a1cc4ffc61157b6b992`.

The complete evidence is
`runs/perception/resident-fov/observations.json` (4,517 bytes, file SHA-256
`8a36a3823a160e2e3ec56e01719a72cd5d1ceafd2f22c3d4d3d595620c887914`).
Its authenticated state digest is
`fdb139b112f4d6d363d23c92d81b90416da6a24f1a5974417a8657753d902dfd`.
It records `experienced: true`, `observed: true`, and
`controller_injection: false`.

While loaded, the integrated GPU reported 301,068,288 bytes VRAM and
3,672,506,368 bytes GTT in use. After shutdown those values were 298,856,448
and 26,955,776 bytes, respectively; the perception process therefore added
about 3.40 GiB of shared GTT. The service and SSH tunnel were stopped after the
measurement.
