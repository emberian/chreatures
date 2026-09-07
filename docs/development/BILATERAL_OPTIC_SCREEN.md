# Native bilateral optic screen sampler

Implemented: `native/world-kernels/src/optic_retina.rs`, registered as
`_world_kernels.OpticRetina`. The core transforms bilateral head-local rays into
world space, intersects a finite emitting screen and bilinearly samples its
current RGB frame. Image interpolation and scene/screen visibility composition
run in Rust with the GIL released. No circuit or controller is part of this
class. The MuJoCo scene-ray join is owned by the physical-world integration lane.

## Fixed construction ABI

```
OpticRetina(
    site_side_hex: int16[1771,3],
    supported_site_mask: bool[1771],
    eye_origins: float64[2,3],
    eye_calibration: float64[2,8],
    max_range: float,
    background_rgb: float32[3],
    atlas_sha256: str,
)
```

All arrays are C contiguous. Site order is exactly the atlas: ascending side
(1=L,2=R), then assigned hex1, then hex2; 879L and892R sites. An absent
photoreceptor mapping does **not** erase an optical sample. Supply support mask
from the unique `photoreceptor_site_indices` (1,486 true sites), preserving the
285 unsupported sites for explicit downstream afferent masking. These are
anatomical coverage flags, not screen-hit flags.

The local head frame is +X forward, +Y left, +Z up. Eye origins are separate
offsets in that frame and use the same spatial units as the physical world.
Each calibration row is `[q0,r0,az0,el0,daz_dq,daz_dr,del_dq,del_dr]` in radians:

```
q = hex1-q0; r = hex2-r0
azimuth = az0+daz_dq*q+daz_dr*r
elevation = el0+del_dq*q+del_dr*r
direction = [cos(elevation)*cos(azimuth),
             cos(elevation)*sin(azimuth), sin(elevation)]
```

These optical parameters are explicitly engineered; assigned hex alone does
not specify an angular receptive field. There is no hidden left/right remapping
or arbitrary empty-site fill. Persist all constructor parameters in the sensory
configuration artifact. `metadata()` supplies format/identity/atlas hash, exact
site order/support mask, `ray_directions_head[1771,3]`, `eye_origins[2,3]`, maximum
range and background. Identity hashes the full constructor contents, including
calibration and support mask.

## Per-boundary sample ABI

```
sample(
    head_positions: float64[B,3],
    head_rotations: float64[B,3,3],
    screen_position: float64[3],
    screen_rotation: float64[3,3],
    screen_size: float64[2],
    frame_rgb: float32[H,W,3],
    occlusion_distance: optional float64[B,1771],
    scene_rgb: optional float32[B,1771,3],
    occlusion_geom: optional int32[B,1771],
    screen_geom: optional int,
) -> dict
```

Rotations are proper orthonormal local→world transforms, verified at the boundary.
Screen local X is image-right, local Y up, and local +Z its emitting front face.
Its width/height define finite bounds. Frame row0 is the top; values must be
finite RGB in[0,1]. Back-face, behind-eye, parallel, out-of-bounds and beyond-range
rays do not see the screen. The texture is not rescaled directly into neural
inputs: head translation, rotation and eye separation change screen intersections.

For physical scene composition, cast the exact metadata rays from the same two
eye origins against MuJoCo, **including the screen and its supporting body**.
Pass nearest geom IDs, distances and material RGB, plus the configured screen
geom ID. The two geom arguments are paired; IDs must be≥−1 (−1 means miss),
screen ID≥0, and distances are required. Paint only when the nearest actual geom
is the screen and its distance agrees with the analytic front-face distance to
within1e−5 world units. Thus the supplied screen position must be the emitting
front face, not a box's center. Other geometry, misses or disagreement block the
texture. The returned `screen_geometry_verified` identifies this physical mode.

For isolated assay composition without geom IDs, screen distance must be strictly
closer than the supplied occluder distance. This mode requires nearest distances
excluding the screen, and does not prove actual physical screen geometry was hit.
+infinity means no opaque hit; NaN or negative distances are invalid. `scene_rgb`
requires `occlusion_distance`; visible screen hits overwrite scene RGB in Rust.
Without scene RGB, misses receive the configured background. Without occlusion
distances, the assay contains only the bounded screen/background and reports that
limitation explicitly.

Output `optic_rgb[B,1771,3]` is the 5,313-scalar optical boundary to the CNS sensory
adapter. Other outputs—`screen_hit[B,1771]`, `screen_distance[B,1771]`,
`screen_uv[B,1771,2]`, `scene_occlusion_supplied`, and identity—are observer or
receipt metadata. They must not reach the cognitive controller. Miss distance
is maximum range and miss UV is[0,0]; the hit mask is authoritative.

The sampler is stateless and does not choose video time. The orchestrator supplies
the decoded frame corresponding to the physical tick. Archive source clip hash,
decode color/range/resolution/timebase, frame index/physical time, optics identity,
atlas identity, CNS adapter/dynamics versions and resident reset boundaries. Raw
optical frames can be training targets; downstream action/memory features must
come through the CNS-derived adapters specified by root.

## Executed validation and limits

The same module exposes `nonvisual_afferent_batch(raw_senses:float32[B,18],
contact_normals:float32[B,8,3], contact_counts:uint8[B],
physiology:float32[B,12]) -> float32[B,43]`. Raw18 order is odor6, body-local
linear velocity3, angular velocity3, touch2, sound3, shade1. The host only packs
the supplied sensed dictionaries and pads unused contacts; this honors runtime
odor overrides and does not construct a retinal or old351-channel observation.

Rust produces odor6 clipped after division by4; linear6 positive/negative
opponents divided by4; angular6 opponents divided by8; contact6 per-axis
positive/negative maxima across actual contacts; contact count/8; clipped touch2;
sound3 clipped after division by2; clipped shade1. These are the exact prior
nonretinal31 semantics. It appends the supplied finite physiology12 unchanged.
More than8 contacts, nonfinite values or contact normals of magnitude>1.0001
are rejected. The resulting43 afferents belong exclusively at the CNS boundary.

`cargo check` passed. An isolated release extension was built at
`/tmp/chreatures-optic-screen-v1`; no running module inode was replaced. Its B4
startup scenario used the actual 1,771-site atlas and a synthetic two-tone finite
screen. Normal, head-yawπ, translated and occluded poses produced screen hit
counts `[1771,0,1420,0]`. Frame inversion changed the actual ray samples, opaque
occlusion preserved supplied scene RGB, and invalid rotations were rejected.
Executed receipt: `data/ports/optic-screen-native-v1.receipt.json`.

The later physical-geometry gate also passed `cargo check` and an isolated native
startup check (`/tmp/chreatures-optic-screen-v2`). Matching screen geom/distance,
different nearest geom, ray miss and distance mismatch produced hit counts
`[1771,0,0,0]`; invalid geom/depth argument pairings were rejected. These were
supplied scene receipts, not MuJoCo-generated ones. Executed receipt:
`data/ports/optic-screen-native-v2.receipt.json`.

After adding body transduction, an isolated release at
`/tmp/chreatures-optic-body-v1` passed a combined B4 screen/body startup scenario.
The31 native channels matched the previous encoder's nonretinal slice exactly
(maximum error0), opposing contacts were preserved, changed odor inputs changed
native afferents, physiology copied exactly and excess contacts were rejected.
The old encoder was used only as an isolated numerical reference. Production
native transduction never reads a retinal tensor or invokes Python encoding.
Receipt: `data/ports/optic-body-native-v1.receipt.json`.

One call took approximately0.154ms in this scenario; that measures this sampler
only, not MuJoCo ray casting, video decoding, CNS simulation or the complete path.
The actual Bad Apple clip and world/circuit/creature recording belong to the
joined integration; this component check does not establish their success.
