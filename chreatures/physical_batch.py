"""Throughput-oriented articulated world with unchanged MuJoCo dynamics.

This module is deliberately an opt-in execution backend.  It keeps one model,
``MjData``, RNG and all delayed state per world, while removing Python scalar
work from the twelve-joint reflex and redundant derived-state work at the
existing world-worker boundary.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .native_world import load_world_kernels
from .physics import MODEL_DT, PhysicsBody, PhysicsWorld
from .sensorium import ArticulatedSensoriumWorld


ROOT = Path(__file__).resolve().parent.parent
OPTIC_ATLAS_PATH = ROOT / "data" / "ports" / "optic-anatomy-audit-v1.npz"
OPTIC_SITES = 1771
OPTIC_RGB_SHAPE = (OPTIC_SITES, 3)
NONVISUAL_CHANNELS = 43
NONVISUAL_RAW_CHANNELS = 18
MAX_CONTACT_NORMALS = 8

# The atlas supplies measured side/hex membership, not ray angles.  These
# engineered compound-eye optics map axial hex coordinates to a broad,
# overlapping bilateral field in the articulated head frame.  Left/right
# calibration is mirrored around the sagittal plane while elevation is shared.
_HEX_CENTER = (18.85, 19.92)
_AZIMUTH_CENTER = math.radians(55.0)
_AZIMUTH_Q = math.radians(2.7)
_AZIMUTH_R = math.radians(1.35)
_ELEVATION_Q = math.radians(-1.35)
_ELEVATION_R = math.radians(2.7)
OPTIC_EYE_CALIBRATION = np.asarray(
    [
        [*_HEX_CENTER, _AZIMUTH_CENTER, 0.0,
         _AZIMUTH_Q, _AZIMUTH_R, _ELEVATION_Q, _ELEVATION_R],
        [*_HEX_CENTER, -_AZIMUTH_CENTER, 0.0,
         -_AZIMUTH_Q, -_AZIMUTH_R, _ELEVATION_Q, _ELEVATION_R],
    ],
    dtype=np.float64,
)
OPTIC_BACKGROUND_RGB = np.asarray([0.015, 0.020, 0.025], dtype=np.float32)
OPTIC_MAXIMUM_RANGE = 3.2


def _optic_atlas() -> tuple[np.ndarray, np.ndarray, str]:
    raw = OPTIC_ATLAS_PATH.read_bytes()
    with np.load(OPTIC_ATLAS_PATH, allow_pickle=False) as atlas:
        sites = np.ascontiguousarray(atlas["site_side_hex"], dtype=np.int16)
        supported = np.zeros(OPTIC_SITES, dtype=np.bool_)
        evidence = np.asarray(atlas["photoreceptor_site_indices"], dtype=np.int64)
    if sites.shape != (OPTIC_SITES, 3) or np.any((evidence < 0) | (evidence >= OPTIC_SITES)):
        raise ValueError("optic anatomy atlas has invalid site membership")
    supported[np.unique(evidence)] = True
    return sites, supported, hashlib.sha256(raw).hexdigest()


OPTIC_SITE_SIDE_HEX, OPTIC_SUPPORTED_SITE_MASK, OPTIC_ATLAS_SHA256 = _optic_atlas()


class NativeActuationCohort:
    """Persistent native articulated layout with one packed update per tick."""

    def __init__(self, roots, qpos, dofs, sides, phases, controller):
        self._native = load_world_kernels().ActuationCohort(
            roots, qpos, dofs, sides, phases, controller
        )

    def begin_tick(self, dynamic: np.ndarray, grips: np.ndarray) -> None:
        self._native.begin_tick(dynamic, grips)

    def apply_gait(self, model, data, world_time: float, timestep: float) -> None:
        self._native.apply_gait(
            int(model._address), int(data._address), world_time, timestep
        )

    def apply_grip(self, model, data, world_time: float, timestep: float) -> None:
        self._native.apply_grip(
            int(model._address), int(data._address), world_time, timestep
        )

    def finish_tick(self) -> np.ndarray:
        return np.asarray(self._native.finish_tick(), dtype=np.float64)

    @staticmethod
    def identity() -> str:
        return str(load_world_kernels().ActuationCohort.identity())


class FastArticulatedSensoriumWorld(ArticulatedSensoriumWorld):
    """Exact-backend fast path for persistent process-owned training worlds."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prepare_fast_articulation()

    def _prepare_fast_articulation(self) -> None:
        """Cache immutable model addresses and controller scalars as arrays."""
        layout = self.articulation_spec["legs"]["layout"]
        self._fast_leg_names = tuple(str(leg["name"]) for leg in layout)
        self._fast_leg_sides = np.asarray(
            [int(leg["side"]) for leg in layout], dtype=np.float64
        )
        self._fast_leg_phases = np.asarray(
            [float(leg["phase"]) for leg in layout], dtype=np.float64
        )
        self._fast_joint_qpos: dict[str, np.ndarray] = {}
        self._fast_joint_dof: dict[str, np.ndarray] = {}
        for body in self.bodies:
            joint_ids = [
                self._leg_joints[body.id][name][kind]
                for name in self._fast_leg_names
                for kind in ("hip", "knee")
            ]
            self._fast_joint_qpos[body.id] = np.asarray(
                [self.model.jnt_qposadr[joint_id] for joint_id in joint_ids],
                dtype=np.int32,
            )
            self._fast_joint_dof[body.id] = np.asarray(
                [self.model.jnt_dofadr[joint_id] for joint_id in joint_ids],
                dtype=np.int32,
            )
        self._fast_joint_kp = {}
        self._fast_joint_kd = {}
        for body in self.bodies:
            controller = self._resident_articulation[body.id]["controller"]
            self._fast_joint_kp[body.id] = np.tile(
                np.asarray(
                    [float(controller["hip_kp"]), float(controller["knee_kp"])],
                    dtype=np.float64,
                ),
                len(layout),
            )
            self._fast_joint_kd[body.id] = np.tile(
                np.asarray(
                    [float(controller["hip_kd"]), float(controller["knee_kd"])],
                    dtype=np.float64,
                ),
                len(layout),
            )
        self._fast_sense_illumination: dict[str, float] | None = None
        self._fast_advance_syncs = -1
        self._fast_food_amounts: tuple[float, ...] = ()
        self._fast_dynamic = np.empty((len(self.bodies), 5), dtype=np.float64)
        self._fast_grips = np.empty(len(self.bodies), dtype=np.int32)
        self._prepare_optic_retina()

    def _prepare_optic_retina(self) -> None:
        """Bind anatomical rays to current head geoms and optional screen geom."""
        previous_frame = getattr(self, "_optic_screen_frame", None)
        native = load_world_kernels()
        optic_type = getattr(native, "OpticRetina", None)
        ray_type = getattr(native, "SceneRayBatch", None)
        if optic_type is None or ray_type is None:
            raise RuntimeError(
                "installed _world_kernels predates bilateral optic sampling; rebuild it"
            )
        self._native_nonvisual = getattr(native, "nonvisual_afferent_batch", None)
        if self._native_nonvisual is None:
            raise RuntimeError(
                "installed _world_kernels predates native nonvisual afferents; rebuild it"
            )
        heads = np.ascontiguousarray(
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
        if np.any(heads < 0):
            raise ValueError("bilateral optics require physical head geoms")
        # One cohort has one immutable optics identity.  Place both eyes beyond
        # the largest current head so inherited body scaling cannot bury a lens.
        head_sizes = np.asarray(self.model.geom_size[heads], dtype=np.float64)
        forward = float(np.max(head_sizes[:, 0])) + 0.004
        lateral = float(np.max(head_sizes[:, 1])) * 0.72
        eye_origins = np.ascontiguousarray(
            [[forward, lateral, 0.0], [forward, -lateral, 0.0]], dtype=np.float64
        )
        self._optic_retina = optic_type(
            OPTIC_SITE_SIDE_HEX,
            OPTIC_SUPPORTED_SITE_MASK,
            eye_origins,
            OPTIC_EYE_CALIBRATION,
            OPTIC_MAXIMUM_RANGE,
            OPTIC_BACKGROUND_RGB,
            OPTIC_ATLAS_SHA256,
        )
        metadata = self._optic_retina.metadata()
        directions = np.ascontiguousarray(
            metadata["ray_directions_head"], dtype=np.float64
        )
        declared_origins = np.ascontiguousarray(metadata["eye_origins"], dtype=np.float64)
        if directions.shape != (OPTIC_SITES, 3) or not np.array_equal(
            declared_origins, eye_origins
        ):
            raise RuntimeError("native optic metadata differs from its physical binding")
        self._optic_screen_geom = self._resolve_optic_screen_geom()
        self._optic_texture_id: int | None = None
        self._optic_texture_shape: tuple[int, int] | None = None
        self._optic_screen_frame: np.ndarray | None = None
        if self._optic_screen_geom is not None:
            self._optic_texture_id = int(mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "optic-screen:texture"
            ))
            if self._optic_texture_id < 0:
                raise RuntimeError("compiled optic screen texture is missing")
            texture_id = self._optic_texture_id
            if int(self.model.tex_nchannel[texture_id]) != 3:
                raise RuntimeError("compiled optic screen texture must have three channels")
            self._optic_texture_shape = (
                int(self.model.tex_height[texture_id]),
                int(self.model.tex_width[texture_id]),
            )
            if previous_frame is not None and previous_frame.shape == (
                *self._optic_texture_shape, 3
            ):
                self._stage_optic_screen_frame(previous_frame)
        self._optic_scene_rays = ray_type(
            int(self.model._address),
            heads,
            np.ascontiguousarray(OPTIC_SITE_SIDE_HEX[:, 0], dtype=np.int16),
            declared_origins,
            directions,
            OPTIC_MAXIMUM_RANGE,
            -1,
            OPTIC_BACKGROUND_RGB,
        )
        self._optic_head_geoms = heads
        self._optic_rgb = np.empty((len(self.bodies), *OPTIC_RGB_SHAPE), dtype=np.float32)
        self._optic_time: float | None = None

    def _resolve_optic_screen_geom(self) -> int | None:
        configured = self.spec.get("optic_screen")
        if configured is None:
            return None
        if (
            not isinstance(configured, dict)
            or set(configured) - {"entity", "shape_index", "front", "texture_shape"}
            or not {"entity", "shape_index", "front"}.issubset(configured)
        ):
            raise ValueError(
                "optic_screen requires exactly entity, shape_index, and front"
            )
        entity = configured["entity"]
        shape_index = configured["shape_index"]
        if (
            not isinstance(entity, str)
            or entity not in self._entity_mj
            or isinstance(shape_index, bool)
            or not isinstance(shape_index, int)
            or configured["front"] not in {"+x", "-x"}
        ):
            raise ValueError("optic_screen binding is invalid")
        geom = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"entity:{entity}:geom:{shape_index}",
        )
        if geom < 0 or int(self.model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("optic_screen must bind an existing box geometry")
        size = np.asarray(self.model.geom_size[geom], dtype=np.float64)
        if size[0] >= min(size[1], size[2]):
            raise ValueError("optic_screen front +/-x must be the box's thin axis")
        return int(geom)

    def _optic_screen_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        geom = self._optic_screen_geom
        if geom is None:
            raise RuntimeError("this physical world has no optic_screen geometry")
        sign = 1.0 if self.spec["optic_screen"]["front"] == "+x" else -1.0
        geom_rotation = np.asarray(self.data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        size = np.asarray(self.model.geom_size[geom], dtype=np.float64)
        normal = sign * geom_rotation[:, 0]
        right = sign * geom_rotation[:, 1]
        up = geom_rotation[:, 2]
        rotation = np.ascontiguousarray(np.column_stack((right, up, normal)))
        position = np.ascontiguousarray(
            np.asarray(self.data.geom_xpos[geom], dtype=np.float64) + normal * size[0]
        )
        dimensions = np.ascontiguousarray([2.0 * size[1], 2.0 * size[2]])
        return position, rotation, dimensions

    def _stage_optic_screen_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Write one RGB frame into MuJoCo's model texture in one bulk copy."""
        texture_id = self._optic_texture_id
        texture_shape = self._optic_texture_shape
        if texture_id is None or texture_shape is None:
            raise RuntimeError("this physical world has no compiled optic screen texture")
        frame = np.asarray(frame_rgb, dtype=np.float32)
        expected = (*texture_shape, 3)
        if (
            frame.shape != expected
            or not np.isfinite(frame).all()
            or np.any((frame < 0.0) | (frame > 1.0))
        ):
            raise ValueError(f"optic screen frame must be finite {expected} RGB in [0,1]")
        pixels = np.rint(frame * 255.0).astype(np.uint8)
        address = int(self.model.tex_adr[texture_id])
        channels = int(self.model.tex_nchannel[texture_id])
        count = texture_shape[0] * texture_shape[1] * channels
        self.model.tex_data[address : address + count] = pixels.reshape(-1)
        # The retinal sampler consumes precisely the values visible in the
        # 8-bit MuJoCo texture, preventing observer/afferent frame drift.
        self._optic_screen_frame = np.ascontiguousarray(
            pixels.astype(np.float32) / 255.0
        )
        return self._optic_screen_frame

    def prepare_observer_renderer(self, renderer: Any) -> None:
        """Upload the staged screen texture to an existing MuJoCo renderer."""
        if self._optic_texture_id is None:
            return
        if getattr(renderer, "model", None) is not self.model:
            raise ValueError("observer renderer is bound to a different MuJoCo model")
        context = getattr(renderer, "_mjr_context", None)
        if context is None:
            raise ValueError("observer renderer exposes no live MuJoCo render context")
        mujoco.mjr_uploadTexture(self.model, context, self._optic_texture_id)

    def optic_retina_batch(
        self, frame_rgb: np.ndarray | None = None, *, refresh: bool = False
    ) -> np.ndarray:
        """Return native bilateral optical input as contiguous ``[B,1771,3]``.

        If the world declares ``optic_screen``, ``frame_rgb`` is required for
        every sample. The frame reaches a site only when that ray's nearest
        physical MuJoCo hit is the configured screen geometry.
        """
        now = float(self.data.time)
        if frame_rgb is None and self._optic_screen_geom is not None:
            raise ValueError("optic_screen sampling requires the current RGB frame")
        if frame_rgb is not None and self._optic_screen_geom is None:
            raise ValueError("RGB frame supplied to a world without optic_screen")
        if frame_rgb is not None or refresh or self._optic_time != now:
            illumination = np.ascontiguousarray(
                [self._illumination(body) for body in self.bodies], dtype=np.float64
            )
            distance, scene_rgb, hit_geom = self._optic_scene_rays.sample(
                int(self.model._address), int(self.data._address), illumination
            )
            scene_rgb = np.asarray(scene_rgb, dtype=np.float32)
            if self._optic_screen_geom is None:
                sampled = scene_rgb
            else:
                visible_frame = self._stage_optic_screen_frame(frame_rgb)
                position, rotation, dimensions = self._optic_screen_pose()
                heads = self._optic_head_geoms
                result = self._optic_retina.sample(
                    np.ascontiguousarray(self.data.geom_xpos[heads], dtype=np.float64),
                    np.ascontiguousarray(
                        self.data.geom_xmat[heads], dtype=np.float64
                    ).reshape(len(heads), 3, 3),
                    position,
                    rotation,
                    dimensions,
                    visible_frame,
                    np.asarray(distance, dtype=np.float64),
                    scene_rgb,
                    np.asarray(hit_geom, dtype=np.int32),
                    int(self._optic_screen_geom),
                )
                sampled = np.asarray(result["optic_rgb"], dtype=np.float32)
            if (
                sampled.shape != (len(self.bodies), *OPTIC_RGB_SHAPE)
                or not sampled.flags.c_contiguous
                or not np.isfinite(sampled).all()
                or np.any((sampled < 0.0) | (sampled > 1.0))
            ):
                raise RuntimeError("native bilateral optic output is malformed")
            self._optic_rgb = sampled
            self._optic_time = now
        view = self._optic_rgb.view()
        view.setflags(write=False)
        return view

    def nonvisual_body_batch(
        self, sensed: Mapping[str, Mapping[str, Any]], physiology: np.ndarray
    ) -> np.ndarray:
        """Transduce sensed nonvisual afferents and interoception as ``[B,43]``.

        ``sensed`` may carry field-transported odor overrides. Python performs
        only the ragged-dictionary packing; scaling, opponent channels, contact
        reduction and concatenation execute in the native kernel.
        """
        interoception = np.asarray(physiology, dtype=np.float32)
        residents = len(self.bodies)
        if (
            not isinstance(sensed, Mapping)
            or set(sensed) != {body.id for body in self.bodies}
            or interoception.shape != (residents, 12)
            or not np.isfinite(interoception).all()
        ):
            raise ValueError("nonvisual body input requires all residents and finite [B,12]")
        raw = np.empty((residents, NONVISUAL_RAW_CHANNELS), dtype=np.float32)
        contact_normals = np.zeros(
            (residents, MAX_CONTACT_NORMALS, 3), dtype=np.float32
        )
        contact_counts = np.zeros(residents, dtype=np.uint8)
        for row, body in enumerate(self.bodies):
            values = sensed[body.id]
            if not isinstance(values, Mapping):
                raise ValueError("each resident sense row must be a mapping")
            odor = np.asarray(values.get("odor"), dtype=np.float32)
            linear = np.asarray(values.get("linear_velocity"), dtype=np.float32)
            angular = np.asarray(values.get("angular_velocity3d"), dtype=np.float32)
            touch = np.asarray(values.get("touch"), dtype=np.float32)
            sound = np.asarray(values.get("sound"), dtype=np.float32)
            shade = np.asarray(values.get("shade"), dtype=np.float32)
            if (
                odor.shape != (2, 3)
                or linear.shape != (3,)
                or angular.shape != (3,)
                or touch.shape != (2,)
                or sound.shape != (3,)
                or shade.shape != ()
            ):
                raise ValueError("resident nonvisual senses have invalid dimensions")
            raw[row] = np.concatenate(
                (odor.reshape(6), linear, angular, touch, sound, shade.reshape(1))
            )
            normals = np.asarray(values.get("contact_normals", ()), dtype=np.float32)
            if normals.size == 0:
                normals = normals.reshape(0, 3)
            if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) > MAX_CONTACT_NORMALS:
                raise ValueError("resident contact normals must have shape [0..8,3]")
            contact_normals[row, : len(normals)] = normals
            contact_counts[row] = len(normals)
        if not np.isfinite(raw).all() or not np.isfinite(contact_normals).all():
            raise ValueError("resident nonvisual senses must be finite")
        result = np.asarray(
            self._native_nonvisual(
                raw, contact_normals, contact_counts,
                np.ascontiguousarray(interoception),
            ),
            dtype=np.float32,
        )
        if (
            result.shape != (residents, NONVISUAL_CHANNELS)
            or not result.flags.c_contiguous
            or not np.isfinite(result).all()
            or np.any((result[:, :34] < 0.0) | (result[:, :34] > 1.0))
            or np.any(np.abs(result[:, 34:36]) > 1.0)
            or np.any((result[:, 36:] < 0.0) | (result[:, 36:] > 1.0))
        ):
            raise RuntimeError("native nonvisual body input is malformed")
        return result

    def optic_retina_metadata(self) -> dict[str, Any]:
        """Return optics identity and static anatomy for host/checkpoint metadata."""
        value = dict(self._optic_retina.metadata())
        value["scene_ray_identity"] = str(self._optic_scene_rays.identity())
        value["screen_geom_bound"] = self._optic_screen_geom is not None
        value["screen_texture_shape"] = self._optic_texture_shape
        return value

    def _begin_resident_actuation(self, clean: dict[str, dict[str, Any]]) -> bool:
        """Fill persistent cohort buffers from the already packed v4 actions."""
        self._fast_dynamic[:, :2] = self._action_cohort[:, :2]
        for index, body in enumerate(self.bodies):
            self._fast_dynamic[index, 2:] = (
                body.energy, body.fatigue, self._active_effort_scale[body.id],
            )
            entity = self._grips.get(body.id)
            self._fast_grips[index] = (
                -1 if not entity or entity not in self._entity_mj else self._entity_mj[entity]
            )
        self._native_actuation.begin_tick(
            self._fast_dynamic.reshape(-1), self._fast_grips,
        )
        return True

    def _rebuild_preserving(self) -> None:
        """Rebind model-address caches after a successful topology rebuild.

        Dynamic entities may insert free joints ahead of resident joints in the
        compiled model. The inherited rebuild preserves state by joint name;
        only after that transaction completes is it safe to resolve the new
        integer qpos/dof addresses used by the vectorized controller.
        """
        super()._rebuild_preserving()
        self._prepare_fast_articulation()

    def _adopt_topology_candidate(
        self, candidate: PhysicsWorld, replaced_entities: set[str],
    ) -> None:
        super()._adopt_topology_candidate(candidate, replaced_entities)
        self._prepare_fast_articulation()

    def _illumination(self, body: PhysicsBody) -> float:
        cache = self._fast_sense_illumination
        if cache is not None:
            value = cache.get(body.id)
            if value is None:
                value = super()._illumination(body)
                cache[body.id] = value
            return value
        return super()._illumination(body)

    def sense(self, body_id: str) -> dict[str, Any]:
        # A base sense computes illumination for retina, shade and its explicit
        # channel. Geometry and light state cannot change during this call.
        self._fast_sense_illumination = {}
        try:
            return super().sense(body_id)
        finally:
            self._fast_sense_illumination = None

    def advance(
        self, actions: dict[str, dict[str, Any]], dt: float = MODEL_DT
    ) -> dict[str, dict[str, float]]:
        # PhysicsWorld advances MuJoCo, calls mj_forward, then synchronizes all
        # public poses. Its final sync repeats the same derived pose/object copy
        # after changing physiology only. Keep the first and elide the duplicate.
        self._fast_advance_syncs = 0
        self._fast_food_amounts = tuple(
            float(component["amount"])
            for entity in self._entities
            for component in self._components[entity["id"]]
            if component.get("type") == "food"
        )
        try:
            return super().advance(actions, dt)
        finally:
            self._fast_advance_syncs = -1
            self._fast_food_amounts = ()

    def _sync_public_state(self) -> None:
        if getattr(self, "_fast_advance_syncs", -1) >= 0:
            self._fast_advance_syncs += 1
            if self._fast_advance_syncs > 1:
                current_food = tuple(
                    float(component["amount"])
                    for entity in self._entities
                    for component in self._components[entity["id"]]
                    if component.get("type") == "food"
                )
                if current_food == self._fast_food_amounts:
                    return
        super()._sync_public_state()

__all__ = [
    "FastArticulatedSensoriumWorld",
    "NONVISUAL_CHANNELS",
    "OPTIC_ATLAS_SHA256",
    "OPTIC_RGB_SHAPE",
    "OPTIC_SITES",
]
