"""Throughput-oriented articulated world with unchanged MuJoCo dynamics.

This module is deliberately an opt-in execution backend.  It keeps one model,
``MjData``, RNG and all delayed state per world, while removing Python scalar
work from the twelve-joint reflex and redundant derived-state work at the
existing world-worker boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .native_world import load_world_kernels
from .physics import MODEL_DT, PhysicsBody
from .sensorium import ArticulatedSensoriumWorld


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

__all__ = ["FastArticulatedSensoriumWorld"]
