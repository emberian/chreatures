"""Face rasterization for opt-in moving chemical membranes.

The membrane approximation changes finite-volume face transmission only. It
does not occupy cells, delete concentration, displace fluid, or model pressure
and hydrodynamic coupling.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable

import numpy as np


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _vector(value: Any, length: int, name: str, low: float, high: float) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return np.asarray([_number(item, name, low, high) for item in value], dtype=np.float64)


def _quaternion(value: Any) -> np.ndarray:
    result = _vector(value, 4, "barrier quaternion", -1.0, 1.0)
    norm = float(np.linalg.norm(result))
    if norm < 1e-12:
        raise ValueError("barrier quaternion cannot be zero")
    result /= norm
    if result[0] < 0.0:
        result *= -1.0
    return result


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class DynamicFaceBarriers:
    """Cached transmission factors for moving oriented box membranes."""

    VERSION = 1

    def __init__(self, size: Iterable[float], shape_xyz: Iterable[int]):
        self.size = _vector(list(size), 3, "barrier world size", 1e-9, 1e6)
        shape = tuple(int(value) for value in shape_xyz)
        if len(shape) != 3 or any(value < 2 for value in shape):
            raise ValueError("barrier grid shape must contain three dimensions >= 2")
        self.nx, self.ny, self.nz = shape
        self.spacing = self.size / np.asarray(shape, dtype=np.float64)
        self.faces = [
            np.ones((self.nz, self.ny, self.nx - 1), dtype=np.float64),
            np.ones((self.nz, self.ny - 1, self.nx), dtype=np.float64),
            np.ones((self.nz - 1, self.ny, self.nx), dtype=np.float64),
        ]
        self.topology: list[dict[str, Any]] | None = None
        self.records: list[dict[str, Any]] = []
        self.update_count = 0
        self._face_centers: list[np.ndarray] | None = None

    @staticmethod
    def normalize(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list) or not 1 <= len(raw) <= 32:
            raise ValueError("dynamic barriers must be a list with 1..32 entries")
        result = []
        ids: set[str] = set()
        for item in raw:
            if not isinstance(item, dict) or set(item) != {
                "id", "permeability", "translation_epsilon", "rotation_epsilon", "shapes",
            }:
                raise ValueError("invalid dynamic barrier record")
            barrier_id = item["id"]
            if not isinstance(barrier_id, str) or not barrier_id or len(barrier_id) > 160 or barrier_id in ids:
                raise ValueError("dynamic barrier ids must be unique short strings")
            shapes = item["shapes"]
            if not isinstance(shapes, list) or not 1 <= len(shapes) <= 16:
                raise ValueError("a dynamic barrier requires 1..16 box shapes")
            normalized_shapes = []
            for shape in shapes:
                if not isinstance(shape, dict) or set(shape) != {"type", "size", "position", "quaternion"} or shape["type"] != "box":
                    raise ValueError("dynamic barrier shapes must be oriented boxes")
                size = _vector(shape["size"], 3, "barrier size", 0.002, 20.0)
                position = _vector(shape["position"], 3, "barrier position", -1e6, 1e6)
                normalized_shapes.append({
                    "type": "box", "size": size.tolist(), "position": position.tolist(),
                    "quaternion": _quaternion(shape["quaternion"]).tolist(),
                })
            result.append({
                "id": barrier_id,
                "permeability": _number(item["permeability"], "barrier permeability", 0.0, 1.0),
                "translation_epsilon": _number(item["translation_epsilon"], "barrier translation epsilon", 1e-7, 0.25),
                "rotation_epsilon": _number(item["rotation_epsilon"], "barrier rotation epsilon", 1e-7, 0.5),
                "shapes": normalized_shapes,
            })
            ids.add(barrier_id)
        return sorted(result, key=lambda value: value["id"])

    @staticmethod
    def _topology(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{
            "id": item["id"], "permeability": item["permeability"],
            "translation_epsilon": item["translation_epsilon"],
            "rotation_epsilon": item["rotation_epsilon"],
            "shape_types": [shape["type"] for shape in item["shapes"]],
            "shape_sizes": [shape["size"] for shape in item["shapes"]],
        } for item in records]

    def _moved(self, current: list[dict[str, Any]]) -> bool:
        if not self.records:
            return True
        for old, new in zip(self.records, current, strict=True):
            for old_shape, new_shape in zip(old["shapes"], new["shapes"], strict=True):
                translation = np.linalg.norm(
                    np.asarray(new_shape["position"]) - np.asarray(old_shape["position"])
                )
                old_q = np.asarray(old_shape["quaternion"])
                new_q = np.asarray(new_shape["quaternion"])
                angle = 2.0 * math.acos(float(np.clip(abs(np.dot(old_q, new_q)), 0.0, 1.0)))
                if translation >= new["translation_epsilon"] or angle >= new["rotation_epsilon"]:
                    return True
        return False

    def sync(self, raw: Any) -> bool:
        records = self.normalize(raw)
        topology = self._topology(records)
        if self.topology is None:
            self.topology = topology
        elif topology != self.topology:
            raise ValueError("dynamic barrier topology changed after field construction")
        if not self._moved(records):
            return False
        self._rasterize(records)
        self.records = copy.deepcopy(records)
        self.update_count += 1
        return True

    def _centers(self) -> list[np.ndarray]:
        if self._face_centers is not None:
            return self._face_centers
        dx, dy, dz = self.spacing
        cell_x = (np.arange(self.nx) + 0.5) * dx
        cell_y = (np.arange(self.ny) + 0.5) * dy
        cell_z = (np.arange(self.nz) + 0.5) * dz
        grids = [
            np.meshgrid(cell_z, cell_y, np.arange(1, self.nx) * dx, indexing="ij"),
            np.meshgrid(cell_z, np.arange(1, self.ny) * dy, cell_x, indexing="ij"),
            np.meshgrid(np.arange(1, self.nz) * dz, cell_y, cell_x, indexing="ij"),
        ]
        self._face_centers = [
            np.stack((grid[2], grid[1], grid[0]), axis=-1).reshape(-1, 3)
            for grid in grids
        ]
        return self._face_centers

    @staticmethod
    def _segment_intersects_box(
        centers: np.ndarray, world_half_segment: np.ndarray,
        position: np.ndarray, rotation: np.ndarray, half_size: np.ndarray,
    ) -> np.ndarray:
        local_center = (centers - position) @ rotation
        direction = world_half_segment @ rotation
        lower = np.full(len(centers), -1.0)
        upper = np.full(len(centers), 1.0)
        active = np.ones(len(centers), dtype=bool)
        for axis in range(3):
            if abs(direction[axis]) < 1e-14:
                active &= np.abs(local_center[:, axis]) <= half_size[axis]
                continue
            first = (-half_size[axis] - local_center[:, axis]) / direction[axis]
            second = (half_size[axis] - local_center[:, axis]) / direction[axis]
            lower = np.maximum(lower, np.minimum(first, second))
            upper = np.minimum(upper, np.maximum(first, second))
        return active & (lower <= upper)

    def _rasterize(self, records: list[dict[str, Any]]) -> None:
        for face in self.faces:
            face.fill(1.0)
        axes = np.eye(3, dtype=np.float64)
        for record in records:
            for shape in record["shapes"]:
                position = np.asarray(shape["position"], dtype=np.float64)
                rotation = _rotation(np.asarray(shape["quaternion"], dtype=np.float64))
                half_size = np.asarray(shape["size"], dtype=np.float64)
                for axis, (centers, face) in enumerate(zip(self._centers(), self.faces, strict=True)):
                    half_segment = axes[axis] * (0.5 * self.spacing[axis])
                    intersects = self._segment_intersects_box(
                        centers, half_segment, position, rotation, half_size,
                    )
                    flat = face.reshape(-1)
                    flat[intersects] = np.minimum(flat[intersects], record["permeability"])

    def snapshot(self) -> dict[str, Any]:
        if self.topology is None:
            raise RuntimeError("dynamic barriers have not been initialized")
        return {
            "version": self.VERSION, "topology": copy.deepcopy(self.topology),
            "records": copy.deepcopy(self.records),
            "faces": [face.tolist() for face in self.faces],
            "update_count": self.update_count,
        }

    @classmethod
    def restore(
        cls, size: Iterable[float], shape_xyz: Iterable[int], snapshot: dict[str, Any],
    ) -> "DynamicFaceBarriers":
        if not isinstance(snapshot, dict) or snapshot.get("version") != cls.VERSION:
            raise ValueError("unsupported dynamic barrier snapshot")
        instance = cls(size, shape_xyz)
        records = instance.normalize(snapshot.get("records"))
        topology = instance._topology(records)
        if snapshot.get("topology") != topology:
            raise ValueError("dynamic barrier snapshot topology differs")
        faces = snapshot.get("faces")
        if not isinstance(faces, list) or len(faces) != 3:
            raise ValueError("dynamic barrier snapshot requires three face arrays")
        restored_faces = [np.asarray(value, dtype=np.float64) for value in faces]
        if any(
            value.shape != baseline.shape or not np.isfinite(value).all()
            or np.any((value < 0.0) | (value > 1.0))
            for value, baseline in zip(restored_faces, instance.faces, strict=True)
        ):
            raise ValueError("invalid dynamic barrier face factors")
        update_count = snapshot.get("update_count")
        if isinstance(update_count, bool) or not isinstance(update_count, int) or update_count < 1:
            raise ValueError("invalid dynamic barrier update count")
        # Re-rasterize to reject a self-consistent-looking face mask that does
        # not actually follow the saved physical membrane poses.
        instance._rasterize(records)
        if any(not np.array_equal(expected, value) for expected, value in zip(instance.faces, restored_faces, strict=True)):
            raise ValueError("dynamic barrier faces do not match saved poses")
        instance.faces = [value.copy() for value in restored_faces]
        instance.topology = topology
        instance.records = copy.deepcopy(records)
        instance.update_count = update_count
        return instance


__all__ = ["DynamicFaceBarriers"]
