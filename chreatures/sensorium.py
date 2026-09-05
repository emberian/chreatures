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
from .physics import PhysicsBody, PhysicsWorld


RETINA_PITCH_OFFSETS = np.asarray((-0.42, -0.21, 0.0, 0.21, 0.42), dtype=np.float64)
RETINA_YAW_OFFSETS = np.linspace(-math.radians(75.0), math.radians(75.0), 16, dtype=np.float64)
RETINA_MAX_RANGE = 3.2


def native_retina(world: PhysicsWorld, body: PhysicsBody) -> list[list[list[float]]]:
    """Return the existing 5x16x4 retina using one compiled collision batch.

    The geometry, exclusion, range, pitch and illumination semantics match
    :meth:`PhysicsWorld._retina3d`.  The returned values remain ordinary lists
    so callers see the existing serialization boundary.
    """
    pitch = np.clip(body.gaze_pitch * 0.62 + RETINA_PITCH_OFFSETS, -1.15, 1.15)
    yaw = body.heading + RETINA_YAW_OFFSETS
    cos_pitch = np.cos(pitch)[:, None]
    directions = np.empty((len(pitch), len(yaw), 3), dtype=np.float64)
    directions[:, :, 0] = cos_pitch * np.cos(yaw)[None, :]
    directions[:, :, 1] = cos_pitch * np.sin(yaw)[None, :]
    directions[:, :, 2] = np.sin(pitch)[:, None]
    flat = np.ascontiguousarray(directions.reshape(-1))
    ray_count = len(pitch) * len(yaw)
    geom_ids = np.full(ray_count, -1, dtype=np.int32)
    distances = np.full(ray_count, -1.0, dtype=np.float64)
    origin = np.asarray([body.x, body.y, body.z + 0.035], dtype=np.float64)
    mujoco.mj_multiRay(
        world.model,
        world.data,
        origin,
        flat,
        None,
        True,
        world._body_mj[body.id],
        geom_ids,
        distances,
        None,
        ray_count,
        RETINA_MAX_RANGE,
    )

    illumination = world._illumination(body)
    result: list[list[list[float]]] = []
    for band in range(len(pitch)):
        rows = []
        for column in range(len(yaw)):
            index = band * len(yaw) + column
            distance, geom_id = float(distances[index]), int(geom_ids[index])
            # mj_multiRay's cutoff is a broad-phase optimization and can still
            # report a farther exact hit. Preserve the public 3.2 m threshold.
            if distance < 0.0 or distance > RETINA_MAX_RANGE or geom_id < 0:
                rows.append([0.0, 0.0, 0.0, 0.0])
            else:
                rgb = world._geom_rgb(geom_id)
                rows.append([
                    min(1.0, channel * (0.45 + 0.55 * illumination)) for channel in rgb
                ] + [max(0.0, 1.0 - distance / RETINA_MAX_RANGE)])
        result.append(rows)
    return result


class BatchedRaySensoriumMixin:
    """Replace scalar retinal rays while preserving the world API."""

    _pending_retina: tuple[str, float, list[list[list[float]]]] | None = None

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


class SensoriumWorld(BatchedRaySensoriumMixin, PhysicsWorld):
    """Default crawler world with batched native retinal collision queries."""


class ArticulatedSensoriumWorld(BatchedRaySensoriumMixin, ArticulatedWorld):
    """Articulated world with batched native retinal collision queries."""


__all__ = [
    "RETINA_MAX_RANGE",
    "RETINA_PITCH_OFFSETS",
    "RETINA_YAW_OFFSETS",
    "native_retina",
    "BatchedRaySensoriumMixin",
    "SensoriumWorld",
    "ArticulatedSensoriumWorld",
]
