"""Thin Python adapter for the native body-bound cohort retina.

The extension owns ray templates, model bindings, scratch, body-frame
transforms, collision queries, transduction, and the coarse projection. Python
selects the versioned profile once, passes per-resident gaze and illumination,
and exposes native contiguous arrays to training code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from .articulated import ArticulatedWorld
from .native_world import load_world_kernels
from .physics import PhysicsBody, PhysicsWorld


ROOT = Path(__file__).resolve().parent.parent
RICH_PROFILE_PATH = ROOT / "data" / "sensorium" / "rich-body-v1.json"
BODY_FRAME = "body-v1"
RICH_PROFILE_NAME = "rich-body-v1"
SENSORIUM_FRAMES = frozenset((BODY_FRAME,))
RICH_COMPONENTS = ("red", "green", "blue", "proximity")
PERIPHERAL_SHAPE = (8, 32, 4)
FOVEAL_SHAPE = (24, 32, 4)
RICH_RAYS = 1024
RICH_CHANNELS = RICH_RAYS * 4


def _load_profile(path: Path = RICH_PROFILE_PATH) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    profile = json.loads(raw)
    if (
        not isinstance(profile, dict)
        or profile.get("format") != "chreatures-sensorium-profile"
        or profile.get("version") != 1
        or profile.get("name") != RICH_PROFILE_NAME
        or profile.get("frame") != "articulated-head-local"
        or profile.get("maximum_range_m") != 3.2
        or profile.get("transduction", {}).get("components") != list(RICH_COMPONENTS)
        or profile.get("transduction", {}).get("dtype") != "float32"
    ):
        raise ValueError("rich body sensorium profile header is invalid")
    lens = profile.get("lens", {})
    gaze = profile.get("gaze", {})
    transduction = profile.get("transduction", {})
    if lens != {
        "attachment": "resident:{id}:geom:head",
        "origin": "head center plus local-forward (head x radius + 0.004 m)",
        "self_occlusion": "full",
        "body_exclusion_id": -1,
    } or gaze != {
        "source": "PhysicsBody.gaze_pitch",
        "scale": 0.62,
        "clamp_radians": [-1.15, 1.15],
        "composition": "clamp(scale * gaze_pitch + ray_pitch_offset)",
    } or transduction != {
        "components": list(RICH_COMPONENTS),
        "rgb": "min(1, material_or_geom_rgb * (0.45 + 0.55 * illumination))",
        "proximity": "max(0, 1 - hit_distance / maximum_range_m)",
        "miss": [0.0, 0.0, 0.0, 0.0],
        "dtype": "float32",
    }:
        raise ValueError("rich body lens, gaze, or transduction contract is invalid")
    rasters = profile.get("rasters")
    if not isinstance(rasters, list) or len(rasters) != 2:
        raise ValueError("rich body sensorium requires peripheral and foveal rasters")
    expected = (
        ("peripheral", PERIPHERAL_SHAPE, [-50.0, 50.0], [-100.0, 100.0]),
        ("foveal", FOVEAL_SHAPE, [-22.0, 22.0], [-30.0, 30.0]),
    )
    for raster, (name, shape, pitch, yaw) in zip(rasters, expected, strict=True):
        if (
            raster.get("name") != name
            or tuple(raster.get("shape", ())) != shape
            or raster.get("pitch_degrees") != pitch
            or raster.get("yaw_degrees") != yaw
            or raster.get("sampling") != "inclusive-linear-centers-elevation-major"
        ):
            raise ValueError(f"invalid {name} retinal raster declaration")
    packed = profile.get("packed", {})
    if packed != {
        "shape": [RICH_RAYS, 4],
        "flat_channels": RICH_CHANNELS,
        "layout": "resident-major, peripheral then foveal, elevation, azimuth, component",
    }:
        raise ValueError("rich body packed channel count is invalid")
    projection = profile.get("coarse_projection", {})
    if (
        projection.get("method") != "unweighted area pooling of measured rays"
        or projection.get("elevation_offsets") != [0, 2, 3, 5, 6, 8]
        or projection.get("shape") != [5, 16, 4]
        or projection.get("extra_collision_rays") != 0
    ):
        raise ValueError("rich body coarse projection declaration is invalid")
    return profile, hashlib.sha256(raw).hexdigest()


RICH_PROFILE, RICH_PROFILE_SHA256 = _load_profile()


def _templates(profile: Mapping[str, Any] = RICH_PROFILE) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    for raster in profile["rasters"]:
        elevation_count, azimuth_count, components = raster["shape"]
        if components != 4:
            raise ValueError("retinal rasters require four transduced components")
        pitch = np.linspace(*map(math.radians, raster["pitch_degrees"]), elevation_count)
        yaw = np.linspace(*map(math.radians, raster["yaw_degrees"]), azimuth_count)
        rows.extend(
            (float(pitch_offset), math.cos(float(yaw_offset)), math.sin(float(yaw_offset)))
            for pitch_offset in pitch
            for yaw_offset in yaw
        )
    result = np.ascontiguousarray(rows, dtype=np.float64)
    if result.shape != (RICH_RAYS, 3):
        raise ValueError("rich body profile did not produce exactly 1024 rays")
    return result


RICH_RAY_TEMPLATES = _templates()
COARSE_ELEVATION_OFFSETS = np.asarray([0, 2, 3, 5, 6, 8], dtype=np.int32)


def _channel_names() -> tuple[str, ...]:
    names: list[str] = []
    for raster in RICH_PROFILE["rasters"]:
        elevations, azimuths, _ = raster["shape"]
        for elevation in range(elevations):
            for azimuth in range(azimuths):
                names.extend(
                    f"rich/{raster['name']}/e{elevation:02d}/a{azimuth:02d}/{component}"
                    for component in RICH_COMPONENTS
                )
    if len(names) != RICH_CHANNELS or len(set(names)) != RICH_CHANNELS:
        raise RuntimeError("rich retinal channel order is malformed")
    return tuple(names)


RICH_CHANNEL_NAMES = _channel_names()
RICH_CHANNEL_NAMES_SHA256 = hashlib.sha256(
    ("\n".join(RICH_CHANNEL_NAMES) + "\n").encode()
).hexdigest()


def profile_identity() -> dict[str, Any]:
    """Return the immutable identity carried by world specs and snapshots."""
    return {
        "frame": BODY_FRAME,
        "profile": RICH_PROFILE_NAME,
        "profile_sha256": RICH_PROFILE_SHA256,
    }


def _profiled_spec(spec: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    habitat = PhysicsWorld._load_spec(spec)
    configured = habitat.get("sensorium")
    expected = profile_identity()
    # The current generation has one sensorium. An omitted selector receives
    # that current profile; any explicit selector must provide its full hash.
    if configured is not None and configured != expected:
        raise ValueError(
            f"sensorium must select the current {RICH_PROFILE_NAME} profile"
        )
    habitat["sensorium"] = expected
    return habitat


def sensorium_frame(world: PhysicsWorld) -> str:
    if world.spec.get("sensorium") != profile_identity():
        raise ValueError("world sensorium profile identity differs")
    return BODY_FRAME


def body_retina_pose(world: PhysicsWorld, body: PhysicsBody) -> dict[str, np.ndarray]:
    """Return the native lens pose for display at the public view boundary."""
    index = world._sensorium_index[body.id]
    head_id = int(world._sensorium_head_geoms[index])
    rotation = np.asarray(world.data.geom_xmat[head_id], dtype=np.float64).reshape(3, 3)
    local_forward = rotation[:, 0]
    local_up = rotation[:, 2]
    head_position = np.asarray(world.data.geom_xpos[head_id], dtype=np.float64)
    origin = head_position + local_forward * (
        float(world.model.geom_size[head_id, 0]) + 0.004
    )
    gaze = float(np.clip(body.gaze_pitch * 0.62, -1.15, 1.15))
    forward = local_forward * math.cos(gaze) + local_up * math.sin(gaze)
    up = -local_forward * math.sin(gaze) + local_up * math.cos(gaze)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    up -= forward * float(np.dot(forward, up))
    up /= max(float(np.linalg.norm(up)), 1e-12)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    return {"origin": origin, "forward": forward, "up": up, "right": right}


def encode_rich_physical_senses(
    senses: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Pack one rich retinal observation as a contiguous float32 row.

    The function performs only a boundary copy. Native cohort consumers should
    prefer :meth:`BatchedRaySensoriumMixin.rich_retina_batch`, which already
    owns a resident-major contiguous `[B,4096]` array.
    """
    if profile is not None and dict(profile) != profile_identity():
        raise ValueError("rich retinal encoder profile identity differs")
    rich = senses.get("rich_retina")
    if not isinstance(rich, Mapping) or rich.get("profile") != profile_identity():
        raise ValueError("senses do not carry the current rich retinal profile")
    peripheral = np.asarray(rich.get("peripheral"), dtype=np.float32)
    foveal = np.asarray(rich.get("foveal"), dtype=np.float32)
    if peripheral.shape != PERIPHERAL_SHAPE or foveal.shape != FOVEAL_SHAPE:
        raise ValueError("rich retinal arrays have invalid dimensions")
    result = np.concatenate((peripheral.reshape(-1), foveal.reshape(-1)))
    result = np.ascontiguousarray(result, dtype=np.float32)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("rich retinal transduction is outside [0,1]")
    return RICH_CHANNEL_NAMES, result


def serialize_rich_retina(senses: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize native senses only at the explicit JSON/view boundary.

    The pooled retina is a native array too. Internal control and collection
    retain arrays; both retinal resolutions must cross a public/checkpoint
    boundary as JSON values.
    """
    def value(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Mapping):
            return {key: value(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [value(child) for child in item]
        return item

    return value(senses)


class BatchedRaySensoriumMixin:
    """Bind and sample one native retina cohort for a MuJoCo world."""

    def __init__(self, *args: Any, **kwargs: Any):
        forwarded = list(args)
        if len(forwarded) >= 2:
            forwarded[1] = _profiled_spec(forwarded[1])
        elif "spec" in kwargs:
            kwargs["spec"] = _profiled_spec(kwargs["spec"])
        else:
            kwargs["spec"] = _profiled_spec(None)
        super().__init__(*forwarded, **kwargs)
        self._prepare_native_sensorium()

    def _prepare_native_sensorium(self) -> None:
        self.sensorium_frame = sensorium_frame(self)
        roots = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"resident:{body.id}:geom:thorax",
                )
                for body in self.bodies
            ],
            dtype=np.int32,
        )
        heads = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"resident:{body.id}:geom:head",
                )
                for body in self.bodies
            ],
            dtype=np.int32,
        )
        if np.any(roots < 0) or np.any(heads < 0):
            raise ValueError("rich body retina requires physical thorax and head geoms")
        native = load_world_kernels()
        cohort_type = getattr(native, "RetinaCohort", None)
        if cohort_type is None:
            raise RuntimeError(
                "installed _world_kernels predates native RetinaCohort; rebuild it"
            )
        self._native_retina = cohort_type(
            int(self.model._address),
            roots,
            heads,
            RICH_RAY_TEMPLATES,
            PERIPHERAL_SHAPE[0] * PERIPHERAL_SHAPE[1],
            COARSE_ELEVATION_OFFSETS,
            float(RICH_PROFILE["maximum_range_m"]),
            RICH_PROFILE_SHA256,
        )
        self._sensorium_index = {body.id: index for index, body in enumerate(self.bodies)}
        self._sensorium_head_geoms = heads
        self._retina_time: float | None = None
        self._retina_consumed: set[str] = set()
        self._coarse_retina = np.empty((len(self.bodies), 5, 16, 4), dtype=np.float32)
        self._rich_retina = np.empty((len(self.bodies), RICH_CHANNELS), dtype=np.float32)
        self._active_retina: tuple[str, np.ndarray, np.ndarray] | None = None

    def _rebuild_preserving(self) -> None:
        super()._rebuild_preserving()
        self._prepare_native_sensorium()

    def _adopt_topology_candidate(
        self, candidate: PhysicsWorld, replaced_entities: set[str],
    ) -> None:
        """Rebind model-address state after an atomic topology transaction."""
        super()._adopt_topology_candidate(candidate, replaced_entities)
        self._prepare_native_sensorium()

    def _sample_retina_cohort(self) -> None:
        gaze = np.ascontiguousarray(
            [body.gaze_pitch for body in self.bodies], dtype=np.float64
        )
        illumination = np.ascontiguousarray(
            [self._illumination(body) for body in self.bodies], dtype=np.float64
        )
        coarse, rich = self._native_retina.sample(
            int(self.model._address), int(self.data._address), gaze, illumination
        )
        self._coarse_retina = np.asarray(coarse, dtype=np.float32)
        self._rich_retina = np.asarray(rich, dtype=np.float32)
        if self._coarse_retina.shape != (len(self.bodies), 5, 16, 4):
            raise RuntimeError("native coarse retinal output has invalid dimensions")
        if self._rich_retina.shape != (len(self.bodies), RICH_CHANNELS):
            raise RuntimeError("native rich retinal output has invalid dimensions")
        if not self._coarse_retina.flags.c_contiguous or not self._rich_retina.flags.c_contiguous:
            raise RuntimeError("native retinal outputs are not resident-major contiguous")
        self._retina_time = float(self.data.time)
        self._retina_consumed.clear()

    def _retina_row(self, body_id: str) -> tuple[np.ndarray, np.ndarray]:
        now = float(self.data.time)
        if self._retina_time != now or body_id in self._retina_consumed:
            self._sample_retina_cohort()
        self._retina_consumed.add(body_id)
        index = self._sensorium_index[body_id]
        return self._coarse_retina[index], self._rich_retina[index]

    def rich_retina_batch(self, *, refresh: bool = False) -> np.ndarray:
        """Return the current resident-major contiguous `[B,4096]` native array."""
        if refresh or self._retina_time != float(self.data.time):
            self._sample_retina_cohort()
        view = self._rich_retina.view()
        view.setflags(write=False)
        return view

    def sense(self, body_id: str) -> dict[str, Any]:
        coarse, packed = self._retina_row(body_id)
        self._active_retina = (body_id, coarse, packed)
        try:
            result = super().sense(body_id)
        finally:
            self._active_retina = None
        peripheral_count = int(np.prod(PERIPHERAL_SHAPE))
        result["rich_retina"] = {
            "peripheral": packed[:peripheral_count].reshape(PERIPHERAL_SHAPE),
            "foveal": packed[peripheral_count:].reshape(FOVEAL_SHAPE),
            "profile": profile_identity(),
        }
        return result

    def _vision(self, body: PhysicsBody, pitch_offset: float = 0.0) -> Any:
        if abs(float(pitch_offset)) > 1e-12:
            raise ValueError("rich body sensorium exposes one declared retinal profile")
        active = self._active_retina
        if active is None or active[0] != body.id:
            coarse, _ = self._retina_row(body.id)
        else:
            coarse = active[1]
        return coarse[2]

    def _retina3d(self, body: PhysicsBody) -> np.ndarray:
        active = self._active_retina
        if active is None or active[0] != body.id:
            coarse, _ = self._retina_row(body.id)
        else:
            coarse = active[1]
        return coarse

    def view(self) -> dict[str, Any]:
        value = super().view()
        value["sensorium"] = {
            **profile_identity(),
            "native_identity": str(self._native_retina.identity()),
            "rays_per_resident": RICH_RAYS,
            "packed_channels": RICH_CHANNELS,
            "ordered_channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
            "layout": RICH_PROFILE["packed"]["layout"],
            "self_occlusion": "full-body-visible",
            "coarse_projection": copy.deepcopy(RICH_PROFILE["coarse_projection"]),
            "retina_pose": {
                body.id: {
                    key: vector.astype(float).tolist()
                    for key, vector in body_retina_pose(self, body).items()
                }
                for body in self.bodies
            },
        }
        return value


class SensoriumWorld(BatchedRaySensoriumMixin, PhysicsWorld):
    """The rich profile requires an articulated head and rejects crawler use."""


class ArticulatedSensoriumWorld(BatchedRaySensoriumMixin, ArticulatedWorld):
    """Articulated world with one native body-bound retina cohort."""


__all__ = [
    "BODY_FRAME",
    "SENSORIUM_FRAMES",
    "RICH_PROFILE_NAME",
    "RICH_PROFILE_SHA256",
    "RICH_CHANNEL_NAMES",
    "RICH_CHANNEL_NAMES_SHA256",
    "RICH_CHANNELS",
    "PERIPHERAL_SHAPE",
    "FOVEAL_SHAPE",
    "profile_identity",
    "sensorium_frame",
    "body_retina_pose",
    "encode_rich_physical_senses",
    "serialize_rich_retina",
    "BatchedRaySensoriumMixin",
    "SensoriumWorld",
    "ArticulatedSensoriumWorld",
]
