"""Offline author/diagnostic targets for the articulated fly curriculum.

This module is collection-only.  Its observations include privileged physical
state, so importing it from a lifetime controller is a contract violation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .curriculum import Bout
from .data import MOTOR


class AuthorSteps(Protocol):
    def get_joint_angles_by_dof_order(
        self, phases: np.ndarray, magnitudes: np.ndarray
    ) -> np.ndarray: ...

    def get_adhesion_onoff_by_phase(self, phases: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class TeacherObservation:
    """Observer values allowed only on the offline teacher/evaluator side."""

    active_joint_position: np.ndarray  # [84], radians in actuator order
    thorax_up: float
    thorax_height: float
    thorax_linear_velocity: np.ndarray  # [3], local
    thorax_angular_velocity: np.ndarray  # [3], local
    foot_contact: np.ndarray  # [6]
    foot_slip_speed: np.ndarray  # [6]
    antenna_target_local: np.ndarray  # [3]
    mouth_target_local: np.ndarray  # [3]
    mouth_contact: bool


@dataclass(frozen=True)
class TeacherCommand:
    normalized_motor: np.ndarray  # [92]
    physical_control: np.ndarray  # [90]
    valid: bool
    target_kind: str


class FlyCurriculumTeacher:
    """Counterbalanced physical references without a deployable gait policy."""

    def __init__(self, schema_path: Path, author_steps: AuthorSteps | None) -> None:
        schema = json.loads(schema_path.read_text())
        actuators = schema.get("actuators", [])
        if len(actuators) != 90:
            raise ValueError("teacher requires the frozen 84-servo plus six-adhesion schema")
        self.schema_path = schema_path.resolve()
        self.author_steps = author_steps
        self.low = np.asarray([row["control_range"][0] for row in actuators[:84]], np.float32)
        self.high = np.asarray([row["control_range"][1] for row in actuators[:84]], np.float32)
        self.neutral = np.asarray([row["neutral_control"] for row in actuators[:84]], np.float32)
        self.groups: dict[str, np.ndarray] = {}
        for group in {row["control_group"] for row in actuators[:84]}:
            self.groups[group] = np.asarray(
                [i for i, row in enumerate(actuators[:84]) if row["control_group"] == group],
                np.int64,
            )
        if self.groups.get("walking", np.empty(0)).size != 42:
            raise ValueError("author walking bank must contain exactly 42 ordered servos")

    def _normalized(self, position: np.ndarray) -> np.ndarray:
        negative = np.maximum(self.neutral - self.low, 1e-6)
        positive = np.maximum(self.high - self.neutral, 1e-6)
        result = np.where(
            position < self.neutral,
            (position - self.neutral) / negative,
            (position - self.neutral) / positive,
        )
        return np.clip(result, -1, 1).astype(np.float32)

    def _author_walk(self, phase: float, turn: float, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
        if self.author_steps is None:
            raise RuntimeError("author FlyGym preprogrammed steps are required for locomotor teacher bouts")
        # Tripod offsets retain the author's leg ordering.  Turning changes
        # ipsilateral amplitudes, not joint targets by an invented inverse map.
        offsets = np.asarray([0, 0.5, 0, 0.5, 0, 0.5], np.float32)
        phases = (phase + offsets) % 1.0
        magnitudes = np.ones(6, np.float32) * amplitude
        magnitudes[:3] *= np.clip(1.0 - turn, 0.15, 1.5)
        magnitudes[3:] *= np.clip(1.0 + turn, 0.15, 1.5)
        angle = np.asarray(
            self.author_steps.get_joint_angles_by_dof_order(phases, magnitudes), np.float32
        ).reshape(-1)
        adhesion = np.asarray(
            self.author_steps.get_adhesion_onoff_by_phase(phases), np.float32
        ).reshape(-1)
        if angle.shape != (42,) or adhesion.shape != (6,):
            raise ValueError("author preprogrammed-step order differs from walking42/feet6")
        return angle, np.clip(adhesion, 0, 1)

    def command(
        self,
        bout: Bout,
        tick_in_bout: int,
        observation: TeacherObservation,
        rng: np.random.Generator,
    ) -> TeacherCommand:
        if observation.active_joint_position.shape != (84,):
            raise ValueError("teacher observer active joint position must be [84]")
        x = self.neutral.copy()
        adhesion = np.zeros(6, np.float32)
        oral = np.zeros(2, np.float32)
        u = tick_in_bout / max(1, bout.stop - bout.start)
        cycle = (tick_in_bout * 0.01 * (7.0 + 2.0 * bout.difficulty)) % 1.0
        phase = bout.phase
        target_kind = "engineered-diagnostic-reference"

        if phase in {
            "posture-support", "forward-locomotion", "turning", "terrain-transition",
            "slip-contact-recovery", "chemical-gradient-forage", "movable-material-push",
        }:
            turn = 0.0
            if phase == "turning":
                turn = (-1.0 if (tick_in_bout // 16) % 2 else 1.0) * (0.25 + 0.45 * bout.difficulty)
            if phase == "slip-contact-recovery":
                turn = float(np.clip(np.mean(observation.foot_slip_speed[:3]) - np.mean(observation.foot_slip_speed[3:]), -0.6, 0.6))
            walking, adhesion = self._author_walk(cycle, turn, 0.45 + 0.5 * bout.difficulty)
            x[self.groups["walking"]] = walking
            target_kind = "flygym-author-preprogrammed-step"
        elif phase == "self-right-recovery":
            walking, adhesion = self._author_walk(cycle, 0.0, 1.0)
            x[self.groups["walking"]] = walking
            # Observer feedback is allowed for this offline recovery target.
            x[self.groups["abdomen"]] += np.clip(1.0 - observation.thorax_up, 0, 1) * 0.35 * np.sin(2 * np.pi * u)
        elif phase == "stopping":
            blend = min(1.0, tick_in_bout / 20.0)
            x = (1 - blend) * observation.active_joint_position + blend * self.neutral
            adhesion[:] = (np.asarray(observation.foot_contact) > 0).astype(np.float32)
        elif phase == "safe-babble":
            # Smooth, low-energy excitation spans every physical group and
            # yields failures as well as viable transitions without white-noise impacts.
            harmonics = np.arange(84, dtype=np.float32) % 7 + 1
            x += (0.05 + 0.12 * bout.difficulty) * np.sin(2 * np.pi * u * harmonics + harmonics)
        elif phase in {"antenna-orient-contact", "conspecific-antenna-contact"}:
            target = np.asarray(observation.antenna_target_local, np.float32)
            yaw = float(np.clip(np.arctan2(target[1], max(abs(target[0]), 1e-4)), -0.7, 0.7))
            pitch = float(np.clip(-np.arctan2(target[2], max(np.linalg.norm(target[:2]), 1e-4)), -0.6, 0.6))
            x[self.groups["head"][:2]] = (yaw * 0.35, pitch * 0.35)
            pedicels = self.groups["pedicels"]
            x[pedicels] = np.asarray([yaw, pitch, 0, yaw, pitch, 0], np.float32)
        elif phase in {"mouth-reach-touch-withdraw", "gustatory-intake-pump-salivary"}:
            target = np.asarray(observation.mouth_target_local, np.float32)
            extend = float(np.clip(np.linalg.norm(target) / 2.0, 0.15, 1.0))
            direction = -1.0 if (phase == "mouth-reach-touch-withdraw" and u > 0.65) else 1.0
            proboscis = self.groups["proboscis"]
            x[proboscis] = direction * extend * np.asarray([0, 0.35, 0, 0, 0.55, 0], np.float32)
            if phase == "gustatory-intake-pump-salivary" and observation.mouth_contact:
                oral[:] = (0.85, 0.55)
        elif phase == "light-acoustic-orient":
            head = self.groups["head"]
            x[head] = np.asarray([0.35 * np.sin(2 * np.pi * u), 0.2 * np.cos(2 * np.pi * u), 0], np.float32)
        elif phase == "free-consequence":
            # This target is deliberately invalid: only the CNS intervention is
            # delivered, preserving uncurated consequences in the corpus.
            return TeacherCommand(np.zeros(MOTOR, np.float32), np.r_[self.neutral, adhesion], False, "none")

        x = np.clip(x, self.low, self.high)
        normalized = np.concatenate((self._normalized(x), adhesion, oral)).astype(np.float32)
        physical = np.concatenate((x, adhesion)).astype(np.float32)
        if normalized.shape != (MOTOR,) or physical.shape != (90,):
            raise AssertionError("teacher output contract differs")
        return TeacherCommand(normalized, physical, True, target_kind)
