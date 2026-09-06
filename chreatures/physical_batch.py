"""Throughput-oriented articulated world with unchanged MuJoCo dynamics.

This module is deliberately an opt-in execution backend.  It keeps one model,
``MjData``, RNG and all delayed state per world, while removing Python scalar
work from the twelve-joint reflex and redundant derived-state work at the
existing world-worker boundary.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .physics import MODEL_DT, PhysicsBody
from .sensorium import ArticulatedSensoriumWorld


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

    def _apply_crawler_forces(
        self, body: PhysicsBody, action: dict[str, Any], noise: np.ndarray
    ) -> None:
        """Vectorized form of the articulated tripod reflex equations."""
        del noise  # The articulated reference controller also ignores motor noise.
        controller = self._resident_articulation[body.id]["controller"]
        forward = float(action.get("forward", action.get("thrust", 0.0)))
        turn = float(action.get("turn", action.get("yaw", 0.0)))
        activity = max(abs(forward), abs(turn))
        strength = (1.0 - 0.72 * body.fatigue) * (0.18 + 0.82 * body.energy)
        frequency = float(controller["frequency_hz"]) * (0.32 + 0.68 * activity)
        stance_fraction = float(controller["stance_fraction"])
        hip_amplitude = math.radians(float(controller["hip_sweep_degrees"]))
        knee_stance = math.radians(float(controller["knee_stance_degrees"]))
        knee_swing = math.radians(float(controller["knee_swing_degrees"]))
        idle_knee = math.radians(float(controller["idle_knee_degrees"]))
        torque_limit = float(controller["max_joint_torque"]) * strength
        active_scale = self._active_effort_scale[body.id]

        side_drive = np.clip(
            forward + self._fast_leg_sides * float(controller["turn_gain"]) * turn,
            -1.0,
            1.0,
        )
        inactive = (activity < 1e-4) | (np.abs(side_drive) < 1e-4)
        cycle = np.remainder(self.time * frequency + self._fast_leg_phases, 1.0)
        stance = cycle < stance_fraction
        progress = np.where(
            stance,
            cycle / stance_fraction,
            (cycle - stance_fraction) / (1.0 - stance_fraction),
        )
        sweep = np.where(stance, 1.0 - 2.0 * progress, -1.0 + 2.0 * progress)
        stride = 0.30 + 0.70 * np.abs(side_drive)
        hip = (
            -self._fast_leg_sides * np.sign(side_drive) * hip_amplitude * stride * sweep
        )
        knee = self._fast_leg_sides * np.where(stance, knee_stance, knee_swing)
        hip[inactive] = 0.0
        knee[inactive] = self._fast_leg_sides[inactive] * idle_knee
        targets = np.column_stack((hip, knee)).reshape(-1)

        qpos = self._fast_joint_qpos[body.id]
        dof = self._fast_joint_dof[body.id]
        torque = (
            self._fast_joint_kp[body.id] * (targets - self.data.qpos[qpos])
            - self._fast_joint_kd[body.id] * self.data.qvel[dof]
        )
        self.data.qfrc_applied[dof] = np.clip(torque, -torque_limit, torque_limit) * active_scale

        root = self._body_mj[body.id]
        rotation = self.data.xmat[root].reshape(3, 3)
        _, angular = self._velocity(root)
        # cross(rotation[:, 2], world-up), expanded to avoid allocating and
        # normalizing temporary axis arrays in every body/substep.
        correction = np.asarray(
            [rotation[1, 2], -rotation[0, 2], 0.0], dtype=np.float64
        ) * float(controller["posture_kp"])
        correction -= angular * float(controller["posture_kd"])
        correction[2] = 0.0
        limit = float(controller["max_posture_torque"])
        norm = float(np.linalg.norm(correction))
        if norm > limit:
            correction *= limit / norm
        self.data.xfrc_applied[root, 3:6] += correction * active_scale


__all__ = ["FastArticulatedSensoriumWorld"]
