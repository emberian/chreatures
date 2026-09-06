"""Conservative three-dimensional chemical fields for the physical habitat.

The solver uses cell-centered finite volumes on a regular z/y/x grid.  Every
interior advective or diffusive face flux is added to one cell and subtracted
from its neighbor, while outer and solid faces have zero flux.  Source and sink
kernels are normalized in mass space.  Consequently total mass changes only by
reported injection, explicit decay, or uptake, apart from floating-point error.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .field_barriers import DynamicFaceBarriers


DEFAULT_CHANNELS = [
    {"name": "odor0", "diffusion": 0.035, "decay": 0.006, "uptake": 0.0},
    {"name": "odor1", "diffusion": 0.030, "decay": 0.008, "uptake": 0.0},
    {"name": "odor2", "diffusion": 0.022, "decay": 0.004, "uptake": 0.0},
]


def _number(value: Any, name: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _vector(value: Any, length: int, name: str, bound: float = 1e6) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return np.asarray([_number(item, name, -bound, bound) for item in value], dtype=np.float64)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quaternion_matrix(quaternion: Iterable[float]) -> np.ndarray:
    w, x, y, z = _vector(list(quaternion), 4, "quaternion")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("quaternion cannot be zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class FieldEnvironment:
    """Multi-channel finite-volume advection/diffusion environment.

    ``shape`` and config grid dimensions are expressed x/y/z.  Field arrays are
    stored channel/z/y/x, matching NumPy's efficient contiguous x rows.
    Concentration units are mass per cubic meter.
    """

    VERSION = 5
    TRANSPORT = "rust-face-source-v2"
    MAX_STATIC_RASTER_SHAPES = 65_536

    def __init__(
        self,
        size: Iterable[float] = (12.0, 8.0, 3.5),
        config: dict[str, Any] | None = None,
        *,
        solid_mask: np.ndarray | None = None,
        permeability: np.ndarray | float | None = None,
    ):
        self.size = _vector(list(size), 3, "world size")
        if np.any(self.size <= 0.0):
            raise ValueError("world size must be positive")
        self.config = self._normalized_config(config)
        self.shape_xyz = tuple(int(value) for value in self.config["grid"])
        self.nx, self.ny, self.nz = self.shape_xyz
        self.grid_shape = (self.nz, self.ny, self.nx)
        self.spacing = self.size / np.asarray(self.shape_xyz, dtype=np.float64)
        self.dx, self.dy, self.dz = map(float, self.spacing)
        self.cell_volume = float(np.prod(self.spacing))
        self.channels = [entry["name"] for entry in self.config["channels"]]
        self.channel_index = {name: index for index, name in enumerate(self.channels)}
        self.diffusion = np.asarray([entry["diffusion"] for entry in self.config["channels"]], dtype=np.float64)
        self.decay = np.asarray([entry["decay"] for entry in self.config["channels"]], dtype=np.float64)
        self.passive_uptake = np.asarray([entry["uptake"] for entry in self.config["channels"]], dtype=np.float64)
        self.concentration = np.zeros((len(self.channels), *self.grid_shape), dtype=np.float64)
        if solid_mask is None:
            self.solid = np.zeros(self.grid_shape, dtype=bool)
        else:
            mask = np.asarray(solid_mask, dtype=bool)
            if mask.shape != self.grid_shape:
                raise ValueError(f"solid_mask must have shape {self.grid_shape}")
            self.solid = mask.copy()
        if permeability is None:
            self.permeability = np.ones(self.grid_shape, dtype=np.float64)
        elif np.isscalar(permeability):
            self.permeability = np.full(self.grid_shape, _number(permeability, "permeability", 0.0, 1.0))
        else:
            values = np.asarray(permeability, dtype=np.float64)
            if values.shape != self.grid_shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f"permeability must be finite in [0,1] with shape {self.grid_shape}")
            self.permeability = values.copy()
        self.base_flow = self._flow_array(self.config.get("flow", [0.0, 0.0, 0.0]))
        self.time = 0.0
        self.rng = np.random.default_rng(int(self.config["seed"]))
        self._source_positions: dict[str, list[float]] = {}
        # Worlds without moving membranes allocate no barrier raster.
        self._dynamic_barriers: DynamicFaceBarriers | None = None
        self.last_sources: list[dict[str, Any]] = []
        self.last_sinks: list[dict[str, Any]] = []
        self.diagnostics = {
            "initial_mass": [0.0] * len(self.channels),
            "injected_mass": [0.0] * len(self.channels),
            "outside_domain_emission": [0.0] * len(self.channels),
            "decayed_mass": [0.0] * len(self.channels),
            "uptake_mass": [0.0] * len(self.channels),
            "numerical_residual": [0.0] * len(self.channels),
            "substeps": 0,
            "last_cfl": 0.0,
        }
        self._base_permeability = self.permeability.copy()
        self._apply_permeability_zones(
            self.config.get("permeability_zones", []), self._base_permeability,
        )
        self.permeability[:] = self._base_permeability
        self.permeability[self.solid] = 0.0
        self._static_revision: int | None = None
        self._static_topology = {
            "rasterizations": 0,
            "displaced_mass": [0.0] * len(self.channels),
            "mass_distance": [0.0] * len(self.channels),
            "max_distance": 0.0,
            "last": None,
        }
        try:
            from _world_kernels import TransportSolver
        except ImportError as exc:
            raise RuntimeError(
                "3D fields require the native world kernels: run "
                "python native/world-kernels/build_extension.py with this interpreter"
            ) from exc
        self._native_transport = TransportSolver(
            self.shape_xyz, len(self.channels), self.spacing.tolist(), self.diffusion.tolist()
        )

    @classmethod
    def from_world(cls, world: Any, config: dict[str, Any] | None = None) -> "FieldEnvironment":
        """Build a field grid and voxelize the world's static physical geoms."""
        view = world.view()
        if view.get("dimension") != 3:
            raise ValueError("FieldEnvironment.from_world requires a 3D physical world")
        size = [view["width"], view["height"], view.get("depth", 3.5)]
        environment = cls(size=size, config=config)
        revision = getattr(world, "model_revision", None)
        environment._sync_static_view(view, revision)
        provider = getattr(world, "diffusion_barriers", None)
        if callable(provider):
            barriers = provider()
            if barriers:
                environment.sync_dynamic_barriers(barriers)
        return environment

    def sync_static_geometry(self, world: Any) -> dict[str, Any] | None:
        """Rerasterize static geometry once after a world topology commit.

        The revision check is the fast path. Candidate rasterization and mass
        relocation complete before any field array is mutated.
        """
        revision = getattr(world, "model_revision", None)
        if isinstance(revision, bool) or not isinstance(revision, (int, np.integer)) or revision < 0:
            raise ValueError("world model_revision must be a nonnegative integer")
        revision = int(revision)
        if self._static_revision == revision:
            return None
        view = world.view()
        return self._sync_static_view(view, revision)

    def _sync_static_view(self, view: Any, revision: Any) -> dict[str, Any]:
        if not isinstance(view, dict) or view.get("dimension") != 3:
            raise ValueError("static field synchronization requires a 3D world view")
        world_size = np.asarray(
            [view.get("width"), view.get("height"), view.get("depth")], dtype=np.float64,
        )
        if world_size.shape != (3,) or not np.isfinite(world_size).all() or not np.array_equal(world_size, self.size):
            raise ValueError("world and field dimensions differ")
        if isinstance(revision, bool) or not isinstance(revision, (int, np.integer)) or revision < 0:
            raise ValueError("world model_revision must be a nonnegative integer")
        entities = view.get("entities")
        if not isinstance(entities, list):
            raise ValueError("world view entities must be a list")
        static_shapes: list[dict[str, Any]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("world view entities must contain objects")
            if entity.get("mobility") != "static":
                continue
            shapes = entity.get("shapes")
            if not isinstance(shapes, list):
                raise ValueError("static entity shapes must be a list")
            if not all(isinstance(shape, dict) for shape in shapes):
                raise ValueError("static entity shapes must contain objects")
            static_shapes.extend(shapes)
        if len(static_shapes) > self.MAX_STATIC_RASTER_SHAPES:
            raise ValueError(
                f"static rasterization supports at most {self.MAX_STATIC_RASTER_SHAPES} shapes"
            )
        if math.prod(self.shape_xyz) > 2_500_000:
            raise ValueError("static rasterization supports at most 2.5M field cells")
        candidate_solid = np.zeros(self.grid_shape, dtype=bool)
        for shape in static_shapes:
            self._voxelize_shape(shape, candidate_solid)

        newly_solid = candidate_solid & ~self.solid
        reopened = self.solid & ~candidate_solid
        candidate_concentration = self.concentration.copy()
        before = self.total_mass.copy()
        displaced_mass = np.zeros(len(self.channels), dtype=np.float64)
        mass_distance = np.zeros(len(self.channels), dtype=np.float64)
        max_distance = 0.0
        occupied = newly_solid & np.any(candidate_concentration > 0.0, axis=0)
        if np.any(occupied):
            fluid = ~candidate_solid
            if not np.any(fluid):
                raise ValueError("static topology would make a mass-bearing field all solid")
            from scipy.ndimage import distance_transform_edt

            distances, nearest = distance_transform_edt(
                candidate_solid,
                sampling=(self.dz, self.dy, self.dx),
                return_indices=True,
            )
            sources = np.nonzero(occupied)
            destinations = tuple(nearest[axis][sources] for axis in range(3))
            source_distance = distances[sources]
            max_distance = float(np.max(source_distance, initial=0.0))
            for channel in range(len(self.channels)):
                moved = candidate_concentration[(np.full(len(sources[0]), channel), *sources)]
                displaced_mass[channel] = float(moved.sum()) * self.cell_volume
                mass_distance[channel] = float(np.dot(moved, source_distance)) * self.cell_volume
                np.add.at(candidate_concentration[channel], destinations, moved)
            candidate_concentration[:, occupied] = 0.0
        candidate_concentration[:, candidate_solid] = 0.0
        after = candidate_concentration.sum(axis=(1, 2, 3)) * self.cell_volume
        residual = after - before
        tolerance = np.maximum(1e-14, np.abs(before) * 2e-13)
        if np.any(np.abs(residual) > tolerance):
            raise FloatingPointError("static topology mass redistribution exceeded roundoff tolerance")
        mean_distance = np.divide(
            mass_distance, displaced_mass,
            out=np.zeros_like(mass_distance), where=displaced_mass > 0.0,
        )
        report = {
            "revision": int(revision),
            "static_shapes": len(static_shapes),
            "new_solid_cells": int(np.count_nonzero(newly_solid)),
            "reopened_cells": int(np.count_nonzero(reopened)),
            "displaced_mass": displaced_mass.tolist(),
            "mean_displacement": mean_distance.tolist(),
            "max_displacement": max_distance,
            "mass_before": before.tolist(),
            "mass_after": after.tolist(),
            "mass_residual": residual.tolist(),
        }
        # Adopt only after the complete candidate and ledger validate.
        self.solid[:] = candidate_solid
        self.concentration[:] = candidate_concentration
        self.permeability[:] = self._base_permeability
        self.permeability[self.solid] = 0.0
        self._static_revision = int(revision)
        self._static_topology["rasterizations"] += 1
        self._static_topology["displaced_mass"] = (
            np.asarray(self._static_topology["displaced_mass"]) + displaced_mass
        ).tolist()
        self._static_topology["mass_distance"] = (
            np.asarray(self._static_topology["mass_distance"]) + mass_distance
        ).tolist()
        self._static_topology["max_distance"] = max(
            float(self._static_topology["max_distance"]), max_distance,
        )
        self._static_topology["last"] = copy.deepcopy(report)
        return report

    def sync_dynamic_barriers(self, barriers: Any) -> bool:
        """Synchronize declared membrane poses at a physical step boundary.

        Empty input without membranes is a constant-time no-op. Once a field
        has barrier topology, that topology cannot change in place.
        """
        if isinstance(barriers, tuple):
            if barriers:
                raise ValueError("dynamic barriers must be a list")
        elif not isinstance(barriers, list):
            raise ValueError("dynamic barriers must be a list")
        if not barriers and self._dynamic_barriers is None:
            return False
        if self._dynamic_barriers is None:
            engine = DynamicFaceBarriers(self.size, self.shape_xyz)
            changed = engine.sync(barriers)
            self._dynamic_barriers = engine
            return changed
        return self._dynamic_barriers.sync(barriers)

    @staticmethod
    def _normalized_config(config: dict[str, Any] | None) -> dict[str, Any]:
        raw = copy.deepcopy(config or {})
        grid = raw.get("grid", [48, 32, 14])
        if not isinstance(grid, (list, tuple)) or len(grid) != 3:
            raise ValueError("grid must be [nx, ny, nz]")
        grid = [int(value) if not isinstance(value, bool) else 0 for value in grid]
        if any(value < 4 or value > 256 for value in grid) or math.prod(grid) > 2_500_000:
            raise ValueError("grid dimensions must be 4..256 with at most 2.5M cells")
        channels = raw.get("channels", DEFAULT_CHANNELS)
        if not isinstance(channels, list) or not 1 <= len(channels) <= 32:
            raise ValueError("channels must contain 1..32 definitions")
        normalized_channels = []
        names: set[str] = set()
        for index, channel in enumerate(channels):
            if isinstance(channel, str):
                channel = {"name": channel}
            if not isinstance(channel, dict) or not isinstance(channel.get("name"), str) or not channel["name"]:
                raise ValueError("each field channel requires a name")
            if channel["name"] in names or len(channel["name"]) > 80:
                raise ValueError("field channel names must be unique and short")
            names.add(channel["name"])
            normalized_channels.append({
                "name": channel["name"],
                "diffusion": _number(channel.get("diffusion", 0.03), f"channel {index} diffusion", 0.0, 2.0),
                "decay": _number(channel.get("decay", 0.0), f"channel {index} decay", 0.0, 10.0),
                "uptake": _number(channel.get("uptake", 0.0), f"channel {index} uptake", 0.0, 10.0),
            })
        fixed_dt = _number(raw.get("integration_dt", 0.01), "integration_dt", 1e-5, 0.2)
        max_cfl = _number(raw.get("max_cfl", 0.82), "max_cfl", 0.05, 0.95)
        max_substeps = raw.get("max_substeps", 512)
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or not 1 <= max_substeps <= 10_000:
            raise ValueError("max_substeps must be an integer in [1,10000]")
        seed = raw.get("seed", 7)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("field seed must be an integer")
        result = {
            "grid": grid,
            "channels": normalized_channels,
            "integration_dt": fixed_dt,
            "max_cfl": max_cfl,
            "max_substeps": max_substeps,
            "seed": seed,
            "flow": copy.deepcopy(raw.get("flow", [0.0, 0.0, 0.0])),
            "permeability_zones": copy.deepcopy(raw.get("permeability_zones", [])),
        }
        return result

    @property
    def total_mass(self) -> np.ndarray:
        return self.concentration.sum(axis=(1, 2, 3)) * self.cell_volume

    @property
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx,
            (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy,
            (np.arange(self.nz, dtype=np.float64) + 0.5) * self.dz,
        )

    def _apply_permeability_zones(self, zones: Any, target: np.ndarray) -> None:
        if not isinstance(zones, list):
            raise ValueError("permeability_zones must be a list")
        x, y, z = self.cell_centers
        for zone in zones:
            if not isinstance(zone, dict):
                raise ValueError("permeability zone must be an object")
            lower = _vector(zone.get("min"), 3, "permeability zone min")
            upper = _vector(zone.get("max"), 3, "permeability zone max")
            value = _number(zone.get("value"), "permeability zone value", 0.0, 1.0)
            if np.any(upper <= lower):
                raise ValueError("permeability zone max must exceed min")
            mask = (
                (z[:, None, None] >= lower[2]) & (z[:, None, None] <= upper[2])
                & (y[None, :, None] >= lower[1]) & (y[None, :, None] <= upper[1])
                & (x[None, None, :] >= lower[0]) & (x[None, None, :] <= upper[0])
            )
            target[mask] = value

    def _voxelize_shape(self, shape: dict[str, Any], target: np.ndarray | None = None) -> None:
        if target is None:
            target = self.solid
        kind = shape.get("type")
        if kind not in {"box", "sphere", "capsule", "cylinder", "ellipsoid"}:
            return
        position = _vector(shape.get("position"), 3, "shape position")
        quaternion = shape.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        rotation = _quaternion_matrix(quaternion)
        size = np.asarray(shape.get("size"), dtype=np.float64)
        x, y, z = self.cell_centers
        # Restrict work to a conservative world-axis bounding cube.
        extent = float(np.linalg.norm(size))
        if kind == "sphere":
            extent = float(size[0])
        elif kind in ("capsule", "cylinder"):
            extent = float(size[0] + size[1])
        ix = np.flatnonzero((x >= position[0] - extent) & (x <= position[0] + extent))
        iy = np.flatnonzero((y >= position[1] - extent) & (y <= position[1] + extent))
        iz = np.flatnonzero((z >= position[2] - extent) & (z <= position[2] + extent))
        if not len(ix) or not len(iy) or not len(iz):
            return
        zz, yy, xx = np.meshgrid(z[iz], y[iy], x[ix], indexing="ij")
        points = np.stack((xx - position[0], yy - position[1], zz - position[2]), axis=-1)
        local = points @ rotation
        if kind == "box":
            inside = np.all(np.abs(local) <= size[:3], axis=-1)
        elif kind == "sphere":
            inside = np.sum(local * local, axis=-1) <= size[0] ** 2
        elif kind == "ellipsoid":
            inside = np.sum((local / size[:3]) ** 2, axis=-1) <= 1.0
        elif kind == "cylinder":
            inside = np.sum(local[..., :2] ** 2, axis=-1) <= size[0] ** 2
            inside &= np.abs(local[..., 2]) <= size[1]
        else:  # capsule: radius and cylindrical half-length along local z
            axial = np.maximum(np.abs(local[..., 2]) - size[1], 0.0)
            inside = np.sum(local[..., :2] ** 2, axis=-1) + axial * axial <= size[0] ** 2
        target[np.ix_(iz, iy, ix)] |= inside

    def _flow_array(self, flow: Any) -> np.ndarray:
        if flow is None:
            return np.zeros((3, *self.grid_shape), dtype=np.float64)
        if isinstance(flow, (list, tuple)) and len(flow) == 3 and all(np.isscalar(value) for value in flow):
            vector = _vector(flow, 3, "flow", 50.0)
            result = np.empty((3, *self.grid_shape), dtype=np.float64)
            result[:] = vector[:, None, None, None]
            return result
        value = np.asarray(flow, dtype=np.float64)
        if value.shape == (*self.grid_shape, 3):
            value = np.moveaxis(value, -1, 0)
        if value.shape != (3, *self.grid_shape) or not np.isfinite(value).all():
            raise ValueError(f"flow must be a 3-vector or array shaped (3,{self.nz},{self.ny},{self.nx})")
        if np.max(np.abs(value), initial=0.0) > 50.0:
            raise ValueError("flow exceeds 50 m/s bound")
        return value.copy()

    def _channel(self, value: Any) -> int:
        if isinstance(value, str):
            if value not in self.channel_index:
                raise ValueError(f"unknown field channel: {value}")
            return self.channel_index[value]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < len(self.channels):
            raise ValueError("field channel index is out of range")
        return int(value)

    def _normalize_sources(self, sources: Any) -> list[dict[str, Any]]:
        if sources is None:
            return []
        if not isinstance(sources, list) or len(sources) > 4096:
            raise ValueError("sources must be a list with at most 4096 entries")
        result = []
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise ValueError("each source must be an object")
            position = _vector(source.get("position"), 3, f"source {index} position")
            key = source.get("key")
            if key is not None and (not isinstance(key, str) or not key or len(key) > 160):
                raise ValueError("source key must be a short nonempty string")
            previous = source.get("previous_position")
            if previous is not None:
                previous = _vector(previous, 3, "previous source position")
            elif key is not None and key in self._source_positions:
                previous = np.asarray(self._source_positions[key], dtype=np.float64)
            else:
                previous = position.copy()
            result.append({
                "key": key,
                "position": position,
                "previous_position": previous,
                "channel": self._channel(source.get("channel", 0)),
                "rate": _number(source.get("rate", 0.0), "source rate", 0.0, 1e6),
                "spread": _number(source.get("spread", 0.0), "source spread", 0.0, float(np.max(self.size))),
            })
        return result

    def _normalize_sinks(self, sinks: Any) -> list[dict[str, Any]]:
        if sinks is None:
            return []
        if not isinstance(sinks, list) or len(sinks) > 4096:
            raise ValueError("sinks must be a list with at most 4096 entries")
        result = []
        for index, sink in enumerate(sinks):
            if not isinstance(sink, dict):
                raise ValueError("each sink must be an object")
            position = _vector(sink.get("position"), 3, f"sink {index} position")
            if np.any(position < 0.0) or np.any(position > self.size):
                raise ValueError("sink is outside field bounds")
            channel_value = sink.get("channel")
            channels = list(range(len(self.channels))) if channel_value is None else [self._channel(channel_value)]
            result.append({
                "position": position,
                "channels": channels,
                "rate": _number(sink.get("rate", 0.0), "sink rate", 0.0, 1e6),
                "spread": _number(sink.get("spread", 0.15), "sink spread", 0.0, float(np.max(self.size))),
            })
        return result

    def _cfl(self, flow: np.ndarray, dt: float) -> float:
        advection = (
            float(np.max(np.abs(flow[0]), initial=0.0)) / self.dx
            + float(np.max(np.abs(flow[1]), initial=0.0)) / self.dy
            + float(np.max(np.abs(flow[2]), initial=0.0)) / self.dz
        )
        diffusion = 2.0 * float(np.max(self.diffusion, initial=0.0)) * (
            1.0 / self.dx**2 + 1.0 / self.dy**2 + 1.0 / self.dz**2
        )
        return dt * (advection + diffusion)

    def advance(
        self,
        dt: float,
        sources: list[dict[str, Any]] | None = None,
        sinks: list[dict[str, Any]] | None = None,
        flow: Any = None,
    ) -> dict[str, Any]:
        """Advance transport, returning a mass ledger for the interval."""
        duration = _number(dt, "field dt", 1e-6, 10.0)
        source_values = self._normalize_sources(sources)
        sink_values = self._normalize_sinks(sinks)
        flow_values = self.base_flow if flow is None else self._flow_array(flow)
        fixed_dt = float(self.config["integration_dt"])
        fixed_cfl = self._cfl(flow_values, fixed_dt)
        if fixed_cfl > float(self.config["max_cfl"]) + 1e-12:
            raise ValueError(
                f"configured integration_dt violates CFL: {fixed_cfl:.4g} > {self.config['max_cfl']:.4g}"
            )
        substeps = max(1, int(math.ceil(duration / fixed_dt)))
        if substeps > int(self.config["max_substeps"]):
            raise ValueError("field advance exceeds configured substep capacity")
        step = duration / substeps
        actual_cfl = self._cfl(flow_values, step)
        if actual_cfl > float(self.config["max_cfl"]) + 1e-12:
            raise ValueError("field step violates CFL")

        before = self.total_mass.copy()
        injected = np.zeros(len(self.channels), dtype=np.float64)
        decayed = np.zeros(len(self.channels), dtype=np.float64)
        uptake = np.zeros(len(self.channels), dtype=np.float64)
        outside = np.zeros(len(self.channels), dtype=np.float64)
        source_positions = np.asarray(
            [source["position"] for source in source_values], dtype=np.float64
        ).reshape((-1, 3))
        source_previous = np.asarray(
            [source["previous_position"] for source in source_values], dtype=np.float64
        ).reshape((-1, 3))
        source_rates = np.asarray(
            [source["rate"] for source in source_values], dtype=np.float64
        )
        source_spreads = np.asarray(
            [source["spread"] for source in source_values], dtype=np.float64
        )
        source_channels = np.asarray(
            [source["channel"] for source in source_values], dtype=np.int32
        )
        for substep in range(substeps):
            fraction = (substep + 0.5) / substeps
            step_injected, step_outside = self._native_transport.deposit_sources(
                step, fraction, self.concentration, self.solid,
                source_positions, source_previous, source_rates,
                source_spreads, source_channels,
            )
            injected += step_injected
            outside += step_outside
            self._transport(step, flow_values)
            old_mass = self.total_mass.copy()
            if np.any(self.decay > 0.0):
                self.concentration *= np.exp(-self.decay * step)[:, None, None, None]
                decayed += old_mass - self.total_mass
            old_mass = self.total_mass.copy()
            if np.any(self.passive_uptake > 0.0):
                self.concentration *= np.exp(-self.passive_uptake * step)[:, None, None, None]
                uptake += old_mass - self.total_mass
            for sink in sink_values:
                for channel in sink["channels"]:
                    uptake[channel] += self._remove_mass(
                        sink["position"], channel, sink["rate"] * step, sink["spread"]
                    )
            self.concentration[:, self.solid] = 0.0

        # Forget absent emitters so a source that later reappears does not draw
        # a fictitious deposition segment across the interval when it was off.
        self._source_positions = {
            source["key"]: source["position"].tolist()
            for source in source_values if source["key"] is not None
        }
        self.time += duration
        self.last_sources = [
            {"key": item["key"], "position": item["position"].tolist(),
             "channel": item["channel"], "rate": item["rate"], "spread": item["spread"]}
            for item in source_values
        ]
        self.last_sinks = [
            {"position": item["position"].tolist(), "channels": list(item["channels"]),
             "rate": item["rate"], "spread": item["spread"]}
            for item in sink_values
        ]
        after = self.total_mass.copy()
        residual = after - (before + injected - decayed - uptake)
        for name, value in (("injected_mass", injected), ("decayed_mass", decayed), ("uptake_mass", uptake)):
            self.diagnostics[name] = (np.asarray(self.diagnostics[name]) + value).tolist()
        self.diagnostics["outside_domain_emission"] = (
            np.asarray(self.diagnostics["outside_domain_emission"]) + outside
        ).tolist()
        self.diagnostics["numerical_residual"] = (
            np.asarray(self.diagnostics["numerical_residual"]) + residual
        ).tolist()
        self.diagnostics["substeps"] = int(self.diagnostics["substeps"]) + substeps
        self.diagnostics["last_cfl"] = actual_cfl
        return {
            "time": self.time,
            "mass_before": before.tolist(),
            "mass_after": after.tolist(),
            "injected": injected.tolist(),
            "outside_domain_emission": outside.tolist(),
            "decayed": decayed.tolist(),
            "uptake": uptake.tolist(),
            "numerical_residual": residual.tolist(),
            "substeps": substeps,
            "cfl": actual_cfl,
        }

    def _transport(self, dt: float, flow: np.ndarray) -> None:
        self._native_transport.step(
            dt, self.concentration, flow, self.permeability, self.solid,
            None if self._dynamic_barriers is None else tuple(self._dynamic_barriers.faces),
        )

    def _kernel(self, position: np.ndarray, spread: float) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        if spread <= 0.0:
            scaled = position / self.spacing - 0.5
            lower = np.floor(scaled).astype(int)
            fraction = scaled - lower
            cells: list[tuple[int, int, int, float]] = []
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        ix, iy, iz = lower + np.array([dx, dy, dz])
                        if not (0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz) or self.solid[iz, iy, ix]:
                            continue
                        weight = (
                            (fraction[0] if dx else 1 - fraction[0])
                            * (fraction[1] if dy else 1 - fraction[1])
                            * (fraction[2] if dz else 1 - fraction[2])
                        )
                        if weight > 0:
                            cells.append((iz, iy, ix, float(weight)))
            if not cells:
                return self._nearest_fluid_kernel(position)
            indices = tuple(np.asarray([cell[axis] for cell in cells], dtype=int) for axis in range(3))
            weights = np.asarray([cell[3] for cell in cells], dtype=np.float64)
            weights /= weights.sum()
            return indices, weights
        x, y, z = self.cell_centers
        radius = 3.5 * spread
        ix = np.flatnonzero(np.abs(x - position[0]) <= radius)
        iy = np.flatnonzero(np.abs(y - position[1]) <= radius)
        iz = np.flatnonzero(np.abs(z - position[2]) <= radius)
        if not len(ix) or not len(iy) or not len(iz):
            return self._nearest_fluid_kernel(position)
        zz, yy, xx = np.meshgrid(z[iz], y[iy], x[ix], indexing="ij")
        weights = np.exp(-((xx - position[0]) ** 2 + (yy - position[1]) ** 2 + (zz - position[2]) ** 2) / (2 * spread * spread))
        weights[self.solid[np.ix_(iz, iy, ix)]] = 0.0
        if float(weights.sum()) <= 0.0:
            return self._nearest_fluid_kernel(position)
        weights /= weights.sum()
        local_indices = np.nonzero(weights > 0.0)
        indices = (iz[local_indices[0]], iy[local_indices[1]], ix[local_indices[2]])
        return indices, weights[local_indices]

    def _nearest_fluid_kernel(self, position: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        fluid = np.argwhere(~self.solid)
        if not len(fluid):
            raise ValueError("field has no fluid cells")
        centers = np.column_stack((
            (fluid[:, 2] + 0.5) * self.dx,
            (fluid[:, 1] + 0.5) * self.dy,
            (fluid[:, 0] + 0.5) * self.dz,
        ))
        index = int(np.argmin(np.sum((centers - position) ** 2, axis=1)))
        iz, iy, ix = fluid[index]
        return (np.asarray([iz]), np.asarray([iy]), np.asarray([ix])), np.asarray([1.0])

    def _deposit_mass(self, position: np.ndarray, channel: int, mass: float, spread: float) -> None:
        indices, weights = self._kernel(position, spread)
        self.concentration[(np.full(len(weights), channel, dtype=int), *indices)] += mass * weights / self.cell_volume

    def _remove_mass(self, position: np.ndarray, channel: int, requested: float, spread: float) -> float:
        if requested <= 0.0:
            return 0.0
        indices, weights = self._kernel(position, spread)
        concentration = self.concentration[(np.full(len(weights), channel, dtype=int), *indices)]
        weighted_available = concentration * weights * self.cell_volume
        available = float(weighted_available.sum())
        removed = min(requested, available)
        if removed <= 0.0:
            return 0.0
        shares = weighted_available / available
        cell_removal = np.minimum(concentration * self.cell_volume, removed * shares)
        # Redistribute a tiny capped remainder if one weighted cell emptied.
        remainder = removed - float(cell_removal.sum())
        if remainder > 1e-15:
            capacity = np.maximum(concentration * self.cell_volume - cell_removal, 0.0)
            if float(capacity.sum()) > 0.0:
                cell_removal += remainder * capacity / capacity.sum()
        self.concentration[(np.full(len(weights), channel, dtype=int), *indices)] -= cell_removal / self.cell_volume
        return float(cell_removal.sum())

    def sample(self, points: Any) -> list[list[float]]:
        """Trilinearly sample channel concentrations at local sensor points."""
        values = np.asarray(points, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("points must have shape (n,3) with finite coordinates")
        result = np.zeros((len(values), len(self.channels)), dtype=np.float64)
        for point_index, point in enumerate(values):
            if np.any(point < 0.0) or np.any(point > self.size):
                continue
            scaled = point / self.spacing - 0.5
            lower = np.floor(scaled).astype(int)
            fraction = scaled - lower
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        ix, iy, iz = lower + np.array([dx, dy, dz])
                        if not (0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz):
                            continue
                        weight = (
                            (fraction[0] if dx else 1 - fraction[0])
                            * (fraction[1] if dy else 1 - fraction[1])
                            * (fraction[2] if dz else 1 - fraction[2])
                        )
                        result[point_index] += weight * self.concentration[:, iz, iy, ix]
        return result.tolist()

    def deposit(self, position: Iterable[float], channel: str | int, mass: float, spread: float = 0.0) -> None:
        """Deposit a known mass immediately, useful for secretion and trails."""
        point = _vector(list(position), 3, "deposit position")
        if np.any(point < 0.0) or np.any(point > self.size):
            raise ValueError("deposit is outside field bounds")
        amount = _number(mass, "deposit mass", 0.0, 1e9)
        width = _number(spread, "deposit spread", 0.0, float(np.max(self.size)))
        channel_index = self._channel(channel)
        self._deposit_mass(point, channel_index, amount, width)
        self.diagnostics["injected_mass"][channel_index] += amount

    def sources_from_world(self, world: Any, emission_scale: float = 0.018) -> list[dict[str, Any]]:
        """Translate moving scented components into internal source parameters.

        Keys track source motion for segment deposition, but ``sample`` never
        returns those keys or any entity identity to an organism.
        """
        scale = _number(emission_scale, "emission_scale", 0.0, 1e6)
        result = []
        for obj in getattr(world, "objects", []):
            components = getattr(obj, "components", [])
            scent = next((item for item in components if item.get("type") == "scent"), None)
            if scent is None:
                continue
            food = next((item for item in components if item.get("type") == "food"), None)
            availability = max(0.0, float(food.get("amount", 0.0))) if food else 1.0
            if availability <= 0.0:
                continue
            odor = int(scent["odor"])
            if odor >= len(self.channels):
                continue
            result.append({
                "key": f"physical-source:{obj.id}:{odor}",
                "position": [float(obj.x), float(obj.y), float(obj.z)],
                "channel": odor,
                "rate": scale * float(scent.get("strength", 1.0)) * availability,
                "spread": max(0.0, min(float(obj.radius) * 0.45, 0.25)),
            })
        return result

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "version": self.VERSION,
            "transport": self.TRANSPORT,
            "size": self.size.tolist(),
            "config": copy.deepcopy(self.config),
            "time": self.time,
            "concentration": self.concentration.tolist(),
            "solid": self.solid.tolist(),
            "permeability": self.permeability.tolist(),
            "base_permeability": self._base_permeability.tolist(),
            "static_revision": self._static_revision,
            "static_topology": copy.deepcopy(self._static_topology),
            "base_flow": self.base_flow.tolist(),
            "source_positions": copy.deepcopy(self._source_positions),
            "last_sources": copy.deepcopy(self.last_sources),
            "last_sinks": copy.deepcopy(self.last_sinks),
            "diagnostics": copy.deepcopy(self.diagnostics),
            "rng_state": _json_value(copy.deepcopy(self.rng.bit_generator.state)),
        }
        if self._dynamic_barriers is not None:
            snapshot["dynamic_barriers"] = self._dynamic_barriers.snapshot()
        return snapshot

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> "FieldEnvironment":
        if not isinstance(snapshot, dict) or snapshot.get("version") != cls.VERSION:
            raise ValueError("unsupported field snapshot")
        imported_version = snapshot["version"]
        if snapshot.get("transport") != cls.TRANSPORT:
            raise ValueError("field snapshot transport implementation differs")
        raw_solid = np.asarray(snapshot["solid"])
        if raw_solid.dtype.kind != "b":
            raise ValueError("invalid solid field snapshot")
        saved_solid = raw_solid.astype(bool, copy=False)
        saved_permeability = np.asarray(snapshot["permeability"], dtype=np.float64)
        supplied_base = snapshot.get("base_permeability")
        if imported_version == cls.VERSION or supplied_base is not None:
            base_permeability = np.asarray(snapshot.get("base_permeability"), dtype=np.float64)
        else:
            # Legacy snapshots did not retain permeability hidden underneath a
            # static solid. Recover configured defaults there while preserving
            # every saved fluid-cell value.
            base_permeability = None
        environment = cls(
            size=snapshot["size"],
            config=snapshot["config"],
            solid_mask=saved_solid,
            permeability=1.0,
        )
        if (
            saved_permeability.shape != environment.grid_shape
            or not np.isfinite(saved_permeability).all()
            or np.any((saved_permeability < 0.0) | (saved_permeability > 1.0))
        ):
            raise ValueError("invalid permeability field snapshot")
        if base_permeability is None:
            base_permeability = environment._base_permeability.copy()
            base_permeability[~saved_solid] = saved_permeability[~saved_solid]
        if (
            base_permeability.shape != environment.grid_shape
            or not np.isfinite(base_permeability).all()
            or np.any((base_permeability < 0.0) | (base_permeability > 1.0))
        ):
            raise ValueError("invalid base permeability field snapshot")
        expected_permeability = base_permeability.copy()
        expected_permeability[saved_solid] = 0.0
        if not np.array_equal(saved_permeability, expected_permeability):
            raise ValueError("saved permeability differs from base permeability and solids")
        environment._base_permeability[:] = base_permeability
        environment.permeability[:] = saved_permeability
        environment.permeability[environment.solid] = 0.0
        concentration = np.asarray(snapshot.get("concentration"), dtype=np.float64)
        flow = np.asarray(snapshot.get("base_flow"), dtype=np.float64)
        if concentration.shape != environment.concentration.shape or not np.isfinite(concentration).all() or np.any(concentration < 0.0):
            raise ValueError("invalid concentration field snapshot")
        if np.any(concentration[:, environment.solid] != 0.0):
            raise ValueError("solid cells cannot contain field mass")
        if flow.shape != environment.base_flow.shape or not np.isfinite(flow).all():
            raise ValueError("invalid flow field snapshot")
        environment.concentration[:] = concentration
        environment.base_flow[:] = flow
        environment.time = _number(snapshot.get("time"), "field time", 0.0, 1e15)
        source_positions = snapshot.get("source_positions")
        if not isinstance(source_positions, dict):
            raise ValueError("invalid remembered source positions")
        environment._source_positions = {
            str(key): _vector(value, 3, "source position").tolist() for key, value in source_positions.items()
        }
        environment.last_sources = copy.deepcopy(snapshot.get("last_sources", []))
        environment.last_sinks = copy.deepcopy(snapshot.get("last_sinks", []))
        environment.diagnostics = copy.deepcopy(snapshot["diagnostics"])
        if imported_version == cls.VERSION:
            revision = snapshot.get("static_revision")
            if revision is not None and (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision < 0
            ):
                raise ValueError("invalid static topology revision")
            topology = snapshot.get("static_topology")
            if not isinstance(topology, dict) or set(topology) != {
                "rasterizations", "displaced_mass", "mass_distance", "max_distance", "last",
            }:
                raise ValueError("invalid static topology ledger")
            rasterizations = topology["rasterizations"]
            if isinstance(rasterizations, bool) or not isinstance(rasterizations, int) or rasterizations < 0:
                raise ValueError("invalid static topology rasterization count")
            for key in ("displaced_mass", "mass_distance"):
                values = np.asarray(topology[key], dtype=np.float64)
                if values.shape != (len(environment.channels),) or not np.isfinite(values).all() or np.any(values < 0.0):
                    raise ValueError(f"invalid static topology {key}")
            _number(topology["max_distance"], "static topology max distance", 0.0, 1e9)
            last = topology["last"]
            if (rasterizations == 0) != (last is None) or (last is not None and revision is None):
                raise ValueError("static topology ledger and revision differ")
            if last is not None:
                if not isinstance(last, dict) or set(last) != {
                    "revision", "static_shapes", "new_solid_cells", "reopened_cells",
                    "displaced_mass", "mean_displacement", "max_displacement",
                    "mass_before", "mass_after", "mass_residual",
                }:
                    raise ValueError("invalid last static topology report")
                for key in ("revision", "static_shapes", "new_solid_cells", "reopened_cells"):
                    if isinstance(last[key], bool) or not isinstance(last[key], int) or last[key] < 0:
                        raise ValueError(f"invalid last static topology {key}")
                for key in ("displaced_mass", "mean_displacement", "mass_before", "mass_after"):
                    values = np.asarray(last[key], dtype=np.float64)
                    if values.shape != (len(environment.channels),) or not np.isfinite(values).all() or np.any(values < 0.0):
                        raise ValueError(f"invalid last static topology {key}")
                residual = np.asarray(last["mass_residual"], dtype=np.float64)
                if residual.shape != (len(environment.channels),) or not np.isfinite(residual).all():
                    raise ValueError("invalid last static topology mass residual")
                _number(last["max_displacement"], "last static topology max displacement", 0.0, 1e9)
                if last["revision"] != revision:
                    raise ValueError("last static topology revision differs")
            environment._static_revision = revision
            environment._static_topology = copy.deepcopy(topology)
        try:
            environment.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid field RNG state") from exc
        if "dynamic_barriers" in snapshot:
            environment._dynamic_barriers = DynamicFaceBarriers.restore(
                environment.size, environment.shape_xyz, snapshot["dynamic_barriers"],
            )
        return environment


def _demo() -> None:
    """Run a small mass-ledger and no-flux wall demonstration."""
    shape = (40, 20, 10)
    solid = np.zeros((shape[2], shape[1], shape[0]), dtype=bool)
    solid[:, :, shape[0] // 2] = True
    field = FieldEnvironment(
        size=(4.0, 2.0, 1.0),
        config={"grid": list(shape), "channels": [{"name": "trace", "diffusion": 0.02}], "integration_dt": 0.005},
        solid_mask=solid,
    )
    ledger = field.advance(1.0, sources=[{"position": [1.7, 1.0, 0.5], "channel": "trace", "rate": 1.0}])
    left, right = field.sample([[1.8, 1.0, 0.5], [2.2, 1.0, 0.5]])
    print(json.dumps({
        "mass": ledger,
        "left_of_wall": left[0],
        "right_of_wall": right[0],
    }, indent=2))


if __name__ == "__main__":
    _demo()


__all__ = ["DEFAULT_CHANNELS", "FieldEnvironment"]
