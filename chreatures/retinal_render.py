"""Offscreen MuJoCo images from a resident body's physical pose and gaze."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from typing import Any
import zlib

import mujoco
import numpy as np

from .physics import PhysicsWorld


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]
    vertical_fov_degrees: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetinalFrame:
    resident_id: str
    model_time: float
    captured_at: float
    width: int
    height: int
    camera: CameraPose
    rgb: np.ndarray

    def png(self) -> bytes:
        return encode_png(self.rgb)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def extract_world_snapshot(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract and authenticate a world snapshot from a world or checkpoint value."""

    if not isinstance(value, dict):
        raise ValueError("snapshot must be a JSON object")
    source: dict[str, Any] = {}
    if value.get("format") == "chreatures-3d-checkpoint-v1":
        state = value.get("state")
        if not isinstance(state, dict):
            raise ValueError("3D checkpoint state is missing")
        digest = hashlib.sha256(_canonical(state)).hexdigest()
        if digest != value.get("sha256"):
            raise ValueError("3D checkpoint checksum mismatch")
        world_snapshot = state.get("world")
        source = {
            "checkpoint_sha256": digest,
            "tick": state.get("tick"),
            "habitat_id": state.get("id"),
        }
    else:
        world_snapshot = value
    if not isinstance(world_snapshot, dict):
        raise ValueError("checkpoint has no 3D world snapshot")
    return world_snapshot, source


def restore_snapshot_world(value: Any) -> tuple[PhysicsWorld, dict[str, Any]]:
    """Restore the concrete world class recorded by the embedded body spec."""

    snapshot, source = extract_world_snapshot(value)
    spec = snapshot.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("world snapshot has no embedded model specification")
    if "articulated_body_spec" in spec:
        from .articulated import ArticulatedWorld

        world_class: type[PhysicsWorld] = ArticulatedWorld
    else:
        world_class = PhysicsWorld
    world = world_class.restore(snapshot)
    source["world_class"] = world_class.__name__
    source["model_signature"] = snapshot.get("model_signature")
    return world, source


def load_snapshot_world(path: str | Path) -> tuple[PhysicsWorld, dict[str, Any]]:
    checkpoint = Path(path)
    value = json.loads(checkpoint.read_text())
    world, source = restore_snapshot_world(value)
    source["checkpoint_path"] = str(checkpoint.resolve())
    return world, source


class RetinalRenderer:
    """One-thread offscreen renderer with a body-attached perspective camera."""

    def __init__(
        self,
        world: PhysicsWorld,
        *,
        width: int = 512,
        height: int = 384,
        vertical_fov_degrees: float = 82.0,
    ):
        if not 64 <= width <= 2048 or not 64 <= height <= 2048:
            raise ValueError("render dimensions must be in 64..2048")
        if not 20.0 <= vertical_fov_degrees <= 140.0:
            raise ValueError("vertical field of view must be in 20..140 degrees")
        self.world = world
        self.width = int(width)
        self.height = int(height)
        self.vertical_fov_degrees = float(vertical_fov_degrees)
        self.renderer = mujoco.Renderer(
            world.model, height=self.height, width=self.width
        )
        self.option = mujoco.MjvOption()
        self.option.label = mujoco.mjtLabel.mjLABEL_NONE
        self.option.frame = mujoco.mjtFrame.mjFRAME_NONE
        self._closed = False

    def camera_pose(self, resident_id: str) -> CameraPose:
        try:
            body = next(body for body in self.world.bodies if body.id == resident_id)
            root_id = self.world._body_mj[resident_id]
        except (KeyError, StopIteration) as error:
            raise KeyError(f"unknown resident: {resident_id}") from error

        rotation = np.asarray(self.world.data.xmat[root_id], dtype=float).reshape(3, 3)
        root_position = np.asarray(self.world.data.xpos[root_id], dtype=float)
        local_forward = rotation[:, 0]
        local_up = rotation[:, 2]

        head_id = mujoco.mj_name2id(
            self.world.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"resident:{resident_id}:geom:head",
        )
        if head_id >= 0:
            head_rotation = np.asarray(
                self.world.data.geom_xmat[head_id], dtype=float
            ).reshape(3, 3)
            local_forward = head_rotation[:, 0]
            local_up = head_rotation[:, 2]
            head_position = np.asarray(self.world.data.geom_xpos[head_id], dtype=float)
            forward_extent = float(self.world.model.geom_size[head_id, 0])
            eye_position = head_position + local_forward * (forward_extent + 0.004)
        else:
            shape = self.world.spec["body_prototype"]["shape"]
            size = np.asarray(shape["size"], dtype=float)
            forward_extent = float(size[0] if size.size == 3 else body.radius)
            eye_position = (
                root_position
                + local_forward * (forward_extent + 0.006)
                + local_up * min(0.02, body.radius * 0.18)
            )

        pitch = float(np.clip(body.gaze_pitch * 0.62, -1.15, 1.15))
        forward = local_forward * math.cos(pitch) + local_up * math.sin(pitch)
        up = -local_forward * math.sin(pitch) + local_up * math.cos(pitch)
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        up -= forward * float(np.dot(forward, up))
        up /= max(float(np.linalg.norm(up)), 1e-12)
        return CameraPose(
            tuple(float(value) for value in eye_position),
            tuple(float(value) for value in forward),
            tuple(float(value) for value in up),
            self.vertical_fov_degrees,
        )

    def render(self, resident_id: str, *, captured_at: float | None = None) -> RetinalFrame:
        if self._closed:
            raise RuntimeError("retinal renderer is closed")
        pose = self.camera_pose(resident_id)
        self.renderer.update_scene(self.world.data, scene_option=self.option)
        self.renderer.scene.stereo = mujoco.mjtStereo.mjSTEREO_NONE
        half_angle = math.radians(pose.vertical_fov_degrees / 2.0)
        for camera in self.renderer.scene.camera:
            camera.pos[:] = pose.position
            camera.forward[:] = pose.forward
            camera.up[:] = pose.up
            half_height = float(camera.frustum_near) * math.tan(half_angle)
            camera.frustum_bottom = -half_height
            camera.frustum_top = half_height
            camera.frustum_center = 0.0
            camera.orthographic = 0
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
        if rgb.shape != (self.height, self.width, 3):
            raise RuntimeError(f"unexpected MuJoCo render shape: {rgb.shape}")
        return RetinalFrame(
            resident_id=resident_id,
            model_time=float(self.world.time),
            captured_at=time.time() if captured_at is None else float(captured_at),
            width=self.width,
            height=self.height,
            camera=pose,
            rgb=rgb,
        )

    def close(self) -> None:
        if not self._closed:
            self.renderer.close()
            self._closed = True

    def __enter__(self) -> "RetinalRenderer":
        return self

    def __exit__(self, *_errors: object) -> None:
        self.close()


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an RGB8 image without adding labels or metadata chunks."""

    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("PNG input must be an HxWx3 uint8 array")
    height, width, _ = array.shape
    if height < 1 or width < 1:
        raise ValueError("PNG input cannot be empty")
    scanlines = b"".join(b"\x00" + row.tobytes() for row in array)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + chunk(b"IEND", b"")
    )


__all__ = [
    "CameraPose",
    "RetinalFrame",
    "RetinalRenderer",
    "encode_png",
    "extract_world_snapshot",
    "load_snapshot_world",
    "restore_snapshot_world",
]
