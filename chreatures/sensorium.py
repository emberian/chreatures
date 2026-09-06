"""Batched native retinal raycasting for MuJoCo-backed worlds.

MuJoCo already owns the live collision acceleration structures.  This module
uses its compiled ``mj_multiRay`` operation rather than mirroring model state in
another extension module merely to remove a Python loop.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from .articulated import ArticulatedWorld
from .native_world import load_world_kernels
from .physics import PhysicsBody, PhysicsWorld


RETINA_PITCH_OFFSETS = np.asarray((-0.42, -0.21, 0.0, 0.21, 0.42), dtype=np.float64)
RETINA_YAW_OFFSETS = np.linspace(-math.radians(75.0), math.radians(75.0), 16, dtype=np.float64)
RETINA_MAX_RANGE = 3.2
LEGACY_FRAME = "legacy-world-v0"
BODY_FRAME = "body-v1"
SENSORIUM_FRAMES = frozenset((LEGACY_FRAME, BODY_FRAME))


def sensorium_frame(world: PhysicsWorld) -> str:
    """Return and validate the snapshot-carried retinal frame selector."""
    config = world.spec.get("sensorium")
    if config is None:
        return LEGACY_FRAME
    if not isinstance(config, dict) or set(config) != {"frame"}:
        raise ValueError("sensorium must contain exactly one frame selector")
    frame = config.get("frame")
    if frame not in SENSORIUM_FRAMES:
        raise ValueError(f"unsupported sensorium frame: {frame!r}")
    return str(frame)


def body_retina_pose(world: PhysicsWorld, body: PhysicsBody) -> dict[str, np.ndarray]:
    """Compute the body-v1 lens origin and gaze basis in world coordinates.

    This mirrors :meth:`RetinalRenderer.camera_pose`: articulated residents use
    the front of the physical head geom, while crawler residents use the front
    of the prototype shell.  Gaze pitch rotates both forward and up with the
    full physical trunk/head frame.
    """
    root_id = world._body_mj[body.id]
    rotation = np.asarray(world.data.xmat[root_id], dtype=np.float64).reshape(3, 3)
    root_position = np.asarray(world.data.xpos[root_id], dtype=np.float64)
    local_forward = rotation[:, 0]
    local_up = rotation[:, 2]
    head_id = mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_GEOM, f"resident:{body.id}:geom:head"
    )
    if head_id >= 0:
        rotation = np.asarray(world.data.geom_xmat[head_id], dtype=np.float64).reshape(3, 3)
        local_forward = rotation[:, 0]
        local_up = rotation[:, 2]
        head_position = np.asarray(world.data.geom_xpos[head_id], dtype=np.float64)
        origin = head_position + local_forward * (float(world.model.geom_size[head_id, 0]) + 0.004)
    else:
        shape = world.spec["body_prototype"]["shape"]
        size = np.asarray(shape["size"], dtype=np.float64)
        forward_extent = float(size[0] if size.size == 3 else body.radius)
        origin = (
            root_position
            + local_forward * (forward_extent + 0.006)
            + local_up * min(0.02, body.radius * 0.18)
        )
    gaze = float(np.clip(body.gaze_pitch * 0.62, -1.15, 1.15))
    forward = local_forward * math.cos(gaze) + local_up * math.sin(gaze)
    up = -local_forward * math.sin(gaze) + local_up * math.cos(gaze)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    up -= forward * float(np.dot(forward, up))
    up /= max(float(np.linalg.norm(up)), 1e-12)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    return {"origin": origin, "forward": forward, "up": up, "right": right, "rotation": rotation}


def _ray_geometry(world: PhysicsWorld, body: PhysicsBody) -> tuple[np.ndarray, np.ndarray, int]:
    pitch = np.clip(body.gaze_pitch * 0.62 + RETINA_PITCH_OFFSETS, -1.15, 1.15)
    if sensorium_frame(world) == LEGACY_FRAME:
        yaw = body.heading + RETINA_YAW_OFFSETS
        cos_pitch = np.cos(pitch)[:, None]
        directions = np.empty((len(pitch), len(yaw), 3), dtype=np.float64)
        directions[:, :, 0] = cos_pitch * np.cos(yaw)[None, :]
        directions[:, :, 1] = cos_pitch * np.sin(yaw)[None, :]
        directions[:, :, 2] = np.sin(pitch)[:, None]
        origin = np.asarray([body.x, body.y, body.z + 0.035], dtype=np.float64)
        return origin, directions, world._body_mj[body.id]

    pose = body_retina_pose(world, body)
    # Build the same spherical raster in head-local coordinates, then rotate
    # every ray by the complete physical frame. Negative azimuth remains the
    # first (left-labelled) half of the established retinal channel order.
    yaw = RETINA_YAW_OFFSETS
    local = np.empty((len(pitch), len(yaw), 3), dtype=np.float64)
    local[:, :, 0] = np.cos(pitch)[:, None] * np.cos(yaw)[None, :]
    local[:, :, 1] = np.cos(pitch)[:, None] * np.sin(yaw)[None, :]
    local[:, :, 2] = np.sin(pitch)[:, None]
    directions = local @ pose["rotation"].T
    # The lens is outside the head and all rays have positive local-forward
    # components. Body-v1 therefore includes the resident's complete geometry:
    # legs or antennae may intentionally occlude the view as in the renderer.
    return pose["origin"], directions, -1


def native_retina(world: PhysicsWorld, body: PhysicsBody) -> list[list[list[float]]]:
    """Return the selected 5x16x4 retina using one compiled collision batch.

    With no selector, geometry, exclusion, range, pitch and illumination match
    the historical :meth:`PhysicsWorld._retina3d`. ``body-v1`` instead grounds
    origin and rays in the full physical body frame. The returned values remain
    ordinary lists so callers see the existing serialization boundary.
    """
    origin, directions, excluded_body = _ray_geometry(world, body)
    flat = np.ascontiguousarray(directions.reshape(-1))
    ray_count = len(RETINA_PITCH_OFFSETS) * len(RETINA_YAW_OFFSETS)
    geom_ids = np.full(ray_count, -1, dtype=np.int32)
    distances = np.full(ray_count, -1.0, dtype=np.float64)
    mujoco.mj_multiRay(
        world.model,
        world.data,
        origin,
        flat,
        None,
        True,
        excluded_body,
        geom_ids,
        distances,
        None,
        ray_count,
        RETINA_MAX_RANGE,
    )

    native = load_world_kernels()
    transducer = getattr(native, "transduce_retina", None)
    if transducer is None:
        raise RuntimeError(
            "installed _world_kernels predates native retinal transduction; "
            "rebuild native/world-kernels"
        )
    result = transducer(
        distances,
        geom_ids,
        world.model.geom_matid,
        world.model.mat_rgba,
        world.model.geom_rgba,
        world._illumination(body),
        RETINA_MAX_RANGE,
    )
    return result.tolist()


class BatchedRaySensoriumMixin:
    """Replace scalar retinal rays while preserving the world API."""

    _pending_retina: tuple[str, float, list[list[list[float]]]] | None = None

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Validate once at construction; the value remains in the habitat spec,
        # model signature and snapshot rather than hidden mutable Python state.
        self.sensorium_frame = sensorium_frame(self)

    def _vision(self, body: PhysicsBody, pitch_offset: float = 0.0) -> list[list[float]]:
        if abs(float(pitch_offset)) > 1e-12:
            # The public sense path requests only the central fan here. Keep
            # uncommon offset calls behaviorally exact through the base method.
            return super()._vision(body, pitch_offset)
        retina = native_retina(self, body)
        self._pending_retina = (body.id, float(self.data.time), retina)
        return retina[2]

    def _retina3d(self, body: PhysicsBody) -> list[list[list[float]]]:
        pending = self._pending_retina
        self._pending_retina = None
        if pending is not None and pending[0] == body.id and pending[1] == float(self.data.time):
            return pending[2]
        return native_retina(self, body)

    def view(self) -> dict[str, Any]:
        value = super().view()
        residents = {}
        if self.sensorium_frame == BODY_FRAME:
            for body in self.bodies:
                pose = body_retina_pose(self, body)
                residents[body.id] = {
                    key: pose[key].astype(float).tolist()
                    for key in ("origin", "forward", "up", "right")
                }
        value["sensorium"] = {
            "frame": self.sensorium_frame,
            "self_occlusion": "full-body-visible" if self.sensorium_frame == BODY_FRAME else "root-body-excluded",
            "retina_pose": residents,
        }
        return value


class SensoriumWorld(BatchedRaySensoriumMixin, PhysicsWorld):
    """Default crawler world with batched native retinal collision queries."""


class ArticulatedSensoriumWorld(BatchedRaySensoriumMixin, ArticulatedWorld):
    """Articulated world with batched native retinal collision queries."""


__all__ = [
    "LEGACY_FRAME",
    "BODY_FRAME",
    "SENSORIUM_FRAMES",
    "RETINA_MAX_RANGE",
    "RETINA_PITCH_OFFSETS",
    "RETINA_YAW_OFFSETS",
    "native_retina",
    "sensorium_frame",
    "body_retina_pose",
    "BatchedRaySensoriumMixin",
    "SensoriumWorld",
    "ArticulatedSensoriumWorld",
]
