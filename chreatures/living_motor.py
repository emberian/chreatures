"""Private inherited motor organ joined to experienced relational refinement.

This wrapper is a strict five-physics-tick state machine.  Its relational key
is the inherited artifact's frozen 64-dimensional projection of neural input;
it has no simulator geometry, identity, named-object, or goal interface.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .context_memory import ContextMemoryConfig, RelationalContextMemory
from .contextual_motor import (
    MEMORY_OUTCOMES,
    MOTOR_ACTIONS,
    ContextualMotorConfig,
    ContextualMotorRefiner,
)
from .motor_inheritance import MotorArtifact, MotorOrgan


FORMAT = "chreatures-living-motor-organ-v1"
PHYSICS_DT = 0.05
MACRO_STEPS = 5
LEGACY_PROFILE = "projection-v1"
RESEARCH_PROFILE = "projection-v2"
CUSTOM_PROFILE = "custom"
RELATIONAL_PROFILES = frozenset((LEGACY_PROFILE, RESEARCH_PROFILE, CUSTOM_PROFILE))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _profile_memory_config(profile: str) -> ContextMemoryConfig:
    common = {"feature_dim": 64, "action_dim": len(MOTOR_ACTIONS),
              "outcome_dim": len(MEMORY_OUTCOMES)}
    if profile == LEGACY_PROFILE:
        return ContextMemoryConfig(**common)
    if profile == RESEARCH_PROFILE:
        # Live orchard measurements found useful projected trajectory changes
        # around 0.32--0.55 RMS, while the v1 allocation threshold was 0.95.
        # V2 retains the frozen projection and resolves those states instead of
        # amplifying or replacing the inherited representation.
        return ContextMemoryConfig(
            **common,
            observation_bandwidth=0.28,
            action_bandwidth=0.20,
            new_context_distance=0.34,
        )
    raise ValueError(f"unsupported built-in relational profile: {profile!r}")


class LivingMotorOrgan:
    """An inherited motor policy with private action-conditioned experience."""

    def __init__(
        self,
        artifact: MotorArtifact | str | Path,
        *,
        seed: int | None = None,
        deterministic: bool = False,
        contextual_enabled: bool = True,
        frozen: bool = False,
        relational_profile: str | None = None,
        memory_config: ContextMemoryConfig | None = None,
        refiner_config: ContextualMotorConfig | None = None,
    ) -> None:
        shared = artifact if isinstance(artifact, MotorArtifact) else MotorArtifact.load(artifact)
        projection_dim = int(shared.config["projection_dim"])
        if projection_dim != 64:
            raise ValueError("living motor requires the inherited 64-dimensional frozen projection")
        if int(shared.config["macro_steps"]) != MACRO_STEPS:
            raise ValueError("living motor requires a five-tick inherited motor macro")
        private_seed = int(shared.config["seed"]) + 311 if seed is None else int(seed)
        self.motor = MotorOrgan(shared, seed=private_seed, deterministic=deterministic)
        if relational_profile is not None and relational_profile not in RELATIONAL_PROFILES:
            raise ValueError(f"unsupported relational profile: {relational_profile!r}")
        if memory_config is None:
            # Projection-v2 remains an explicit research setting until its
            # increased state resolution has longitudinal evidence.
            self.relational_profile = relational_profile or LEGACY_PROFILE
            if self.relational_profile == CUSTOM_PROFILE:
                raise ValueError("custom relational profile requires memory_config")
            memory_config = _profile_memory_config(self.relational_profile)
        else:
            self.relational_profile = relational_profile or CUSTOM_PROFILE
            if self.relational_profile != CUSTOM_PROFILE:
                expected_profile = _profile_memory_config(self.relational_profile)
                if memory_config != expected_profile:
                    raise ValueError("memory configuration differs from its relational profile")
        if (
            memory_config.feature_dim != projection_dim
            or memory_config.action_dim != len(MOTOR_ACTIONS)
            or memory_config.outcome_dim != len(MEMORY_OUTCOMES)
        ):
            raise ValueError("memory configuration differs from living motor dimensions")
        self.memory = RelationalContextMemory(memory_config)
        self.refiner = ContextualMotorRefiner(
            projection_dim,
            seed=private_seed + 1709,
            config=refiner_config,
            enabled=contextual_enabled,
        )
        self.refiner.freeze(frozen)
        self.pending: dict[str, Any] | None = None
        self.last_record: dict[str, Any] | None = None
        self.last_actual_correction: float | None = None
        self.metrics: dict[str, int | float] = {
            "ticks": 0,
            "macro_decisions": 0,
            "recorded_transitions": 0,
            "context_changed_choices": 0,
            "experienced_nutrition": 0.0,
            "experienced_effort": 0.0,
            "contacted_transitions": 0,
            "episodes": 0,
        }

    @property
    def artifact(self) -> MotorArtifact:
        return self.motor.artifact

    @property
    def last_prediction_error(self) -> float | None:
        return self.motor.last_prediction_error

    @staticmethod
    def _local_physiology(value: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError("local physiology must be a mapping")
        # MotorOrgan performs the strict no-extra-fields check and validates all
        # six local channels. The relational consequence model retains only
        # homeostatic state, not kinematic or environment fields.
        MotorOrgan.physiology_vector(value)
        return ContextualMotorRefiner.physiology(value)

    @staticmethod
    def _physics_outcome(value: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError("previous physics outcome must be a mapping")
        nutrition = _finite(value.get("nutrition", 0.0), "nutrition")
        effort = _finite(value.get("effort", 0.0), "effort")
        contact = _finite(value.get("contact", 0.0), "contact")
        if nutrition < 0 or effort < 0 or not 0 <= contact <= 1:
            raise ValueError("nutrition/effort must be nonnegative and contact must be in [0, 1]")
        return {"nutrition": nutrition, "effort": effort, "contact": contact}

    @staticmethod
    def _validate_dt(dt: Any) -> float:
        step = _finite(dt, "dt")
        if not math.isclose(step, PHYSICS_DT, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"living motor requires fixed dt={PHYSICS_DT}")
        return step

    def _feature(self, normalized: np.ndarray) -> np.ndarray:
        projected = self.motor.projected(normalized)
        return _vector(projected, 64, "projected relational feature")

    def _new_pending(
        self,
        feature: np.ndarray,
        physiology: Mapping[str, float],
        decision: Mapping[str, Any],
    ) -> None:
        action = _vector(decision["action_vector"], len(MOTOR_ACTIONS), "committed action")
        selected = int(decision["selected_index"])
        candidate = decision["candidates"][selected]
        self.pending = {
            "feature": feature.copy(),
            "before_physiology": dict(physiology),
            "action": action.copy(),
            "outcome": {"nutrition": 0.0, "effort": 0.0, "contact": 0.0},
            "outcomes_seen": 0,
        }
        self.metrics["macro_decisions"] += 1
        self.metrics["context_changed_choices"] += int(decision["context_changed_choice"])
        self.last_actual_correction = float(candidate["contextual_correction"])

    def _accumulate(self, value: Mapping[str, Any], dt: float) -> None:
        if self.pending is None:
            raise RuntimeError("physics outcome arrived without a pending motor transition")
        outcome = self._physics_outcome(value)
        total = self.pending["outcome"]
        total["nutrition"] += outcome["nutrition"]
        total["effort"] += outcome["effort"] * dt
        total["contact"] = max(total["contact"], outcome["contact"])
        self.pending["outcomes_seen"] += 1

    def tick(
        self,
        raw_neural_features: Any,
        local_physiology: Mapping[str, Any],
        previous_physics_outcome: Mapping[str, Any] | None,
        dt: float = PHYSICS_DT,
        *,
        candidate_evidence: Callable[
            [tuple[tuple[float, ...], ...]], Mapping[str, Any]
        ] | None = None,
    ) -> dict[str, float]:
        """Consume the preceding outcome and return this tick's physical action.

        The first call of an episode requires ``previous_physics_outcome=None``.
        Every later call consumes exactly one body's outcome from the physics
        step driven by the action returned from the preceding call.
        """
        step = self._validate_dt(dt)
        normalized = self.motor.normalize(raw_neural_features)
        projected = self._feature(normalized)
        physiology = self._local_physiology(local_physiology)

        if self.motor.held_ticks == 0:
            if self.pending is not None:
                raise RuntimeError("pending transition exists at an open motor boundary")
            if previous_physics_outcome is not None:
                raise ValueError("initial episode tick cannot consume a prior physics outcome")
            decision = self.refiner.refine_and_commit(
                self.motor, self.memory, projected, raw_neural_features,
                local_physiology, step, candidate_evidence=candidate_evidence,
            )
            self._new_pending(projected, physiology, decision)
            self.metrics["ticks"] += 1
            return dict(decision["action"])

        if self.pending is None:
            raise RuntimeError("held motor action has no pending transition")
        if previous_physics_outcome is None:
            raise ValueError("each held tick requires the preceding physics outcome")
        expected_seen = self.motor.held_ticks - 1
        if int(self.pending["outcomes_seen"]) != expected_seen:
            raise RuntimeError("pending outcome count differs from inherited hold state")
        self._accumulate(previous_physics_outcome, step)

        if self.motor.held_ticks < MACRO_STEPS:
            action = self.motor.continue_macro_action(step)
            self.metrics["ticks"] += 1
            return action

        if int(self.pending["outcomes_seen"]) != MACRO_STEPS:
            raise RuntimeError("completed macro does not contain five physics outcomes")
        completed = self.pending
        record = self.refiner.record(
            self.memory,
            completed["feature"],
            completed["action"],
            projected,
            completed["before_physiology"],
            physiology,
            completed["outcome"],
        )
        self.last_record = copy.deepcopy(record)
        self.metrics["recorded_transitions"] += 1
        self.metrics["experienced_nutrition"] += float(completed["outcome"]["nutrition"])
        self.metrics["experienced_effort"] += float(completed["outcome"]["effort"])
        self.metrics["contacted_transitions"] += int(completed["outcome"]["contact"] > 0)
        self.pending = None

        # The relational graph sees the completed transition before inherited
        # context advances and the next candidate set is evaluated.
        self.motor.open_macro_boundary(normalized)
        decision = self.refiner.refine_and_commit(
            self.motor, self.memory, projected, raw_neural_features,
            local_physiology, step, candidate_evidence=candidate_evidence,
        )
        self._new_pending(projected, physiology, decision)
        self.metrics["ticks"] += 1
        return dict(decision["action"])

    def freeze(self, frozen: bool = True) -> None:
        """Freeze learning while retaining already learned refinement."""
        self.refiner.freeze(frozen)

    def set_refinement_enabled(self, enabled: bool) -> None:
        """Enable contextual choice changes without changing learning state."""
        self.refiner.enabled = bool(enabled)

    def clear_memory(self) -> None:
        """Explicitly clear learned context while preserving inherited state."""
        self.refiner.clear(self.memory)

    def reset_episode(self) -> None:
        """Discard an incomplete transition and reset inference, not learning."""
        self.motor.reset_episode()
        self.memory.reset()
        self.pending = None
        self.metrics["episodes"] += 1

    def view(self) -> dict[str, Any]:
        state = self.memory.state()
        return {
            "controller": "living-contextual-motor",
            "artifact_sha256": self.artifact.sha256,
            "motor": self.motor.view(),
            "memory": {
                "profile": self.relational_profile,
                "contexts": state["contexts"],
                "transitions": state["transitions"],
                "active_context": state["active_context"],
                "revision": self.refiner.memory_revision,
            },
            "learning": {
                "frozen": not self.refiner.learning,
                "refinement_enabled": self.refiner.enabled,
            },
            "last_contextual_correction": self.last_actual_correction,
            "last_record": copy.deepcopy(self.last_record),
            "metrics": copy.deepcopy(self.metrics),
            "pending": None if self.pending is None else {
                "outcomes_seen": int(self.pending["outcomes_seen"]),
                "nutrition": float(self.pending["outcome"]["nutrition"]),
                "effort": float(self.pending["outcome"]["effort"]),
                "contact": float(self.pending["outcome"]["contact"]),
            },
        }

    def snapshot(self) -> dict[str, Any]:
        pending = None
        if self.pending is not None:
            pending = {
                "feature": self.pending["feature"].tolist(),
                "before_physiology": copy.deepcopy(self.pending["before_physiology"]),
                "action": self.pending["action"].tolist(),
                "outcome": copy.deepcopy(self.pending["outcome"]),
                "outcomes_seen": int(self.pending["outcomes_seen"]),
            }
        return {
            "format": FORMAT,
            "version": 1,
            "artifact_sha256": self.artifact.sha256,
            "physics_dt": PHYSICS_DT,
            "macro_steps": MACRO_STEPS,
            "relational_profile": self.relational_profile,
            "motor": self.motor.snapshot_value(include_artifact=False),
            "contextual": self.refiner.snapshot(self.memory),
            "pending": pending,
            "last_actual_correction": self.last_actual_correction,
            "last_record": copy.deepcopy(self.last_record),
            "metrics": copy.deepcopy(self.metrics),
        }

    def snapshot_value(self, *, include_artifact: bool = False) -> dict[str, Any]:
        """Return private JSON state with a shared artifact reference by default."""
        value = self.snapshot()
        if include_artifact:
            value["artifact"] = self.artifact.to_value()
        return value

    @classmethod
    def restore(
        cls,
        value: Any,
        artifact: MotorArtifact | str | Path,
    ) -> "LivingMotorOrgan":
        if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("version") != 1:
            raise ValueError("unsupported living motor snapshot")
        if value.get("physics_dt") != PHYSICS_DT or value.get("macro_steps") != MACRO_STEPS:
            raise ValueError("living motor timing contract differs")
        shared = artifact if isinstance(artifact, MotorArtifact) else MotorArtifact.load(artifact)
        if value.get("artifact_sha256") != shared.sha256:
            raise ValueError("living motor artifact identity differs")
        profile = value.get("relational_profile", LEGACY_PROFILE)
        if profile not in RELATIONAL_PROFILES:
            raise ValueError("unsupported saved relational profile")
        try:
            saved_config = ContextMemoryConfig(**value["contextual"]["memory"]["config"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid saved relational memory configuration") from error
        instance = cls(
            shared,
            relational_profile=profile,
            memory_config=saved_config if profile == CUSTOM_PROFILE else None,
        )
        instance.motor = MotorOrgan.restore_value(value["motor"], shared)
        instance.refiner, instance.memory = ContextualMotorRefiner.restore(value["contextual"])
        if instance.refiner.feature_dim != 64:
            raise ValueError("living motor contextual feature dimension differs")
        if profile != CUSTOM_PROFILE and instance.memory.config != _profile_memory_config(profile):
            raise ValueError("saved memory configuration differs from its relational profile")
        instance.relational_profile = profile
        instance.pending = instance._restore_pending(value.get("pending"))
        correction = value.get("last_actual_correction")
        instance.last_actual_correction = None if correction is None else _finite(
            correction, "last actual correction"
        )
        last_record = value.get("last_record")
        if last_record is not None and not isinstance(last_record, dict):
            raise ValueError("invalid last living motor record")
        instance.last_record = copy.deepcopy(last_record)
        instance.metrics = instance._restore_metrics(value.get("metrics"))
        instance._validate_pending_state()
        if instance.last_actual_correction is not None and abs(instance.last_actual_correction) > (
            instance.refiner.config.max_logit_correction + 1e-7
        ):
            raise ValueError("last contextual correction exceeds configured bound")
        if not (
            instance.motor.decision_count == instance.refiner.decision_count
            == int(instance.metrics["macro_decisions"])
        ):
            raise ValueError("living motor decision counters differ")
        if int(instance.metrics["recorded_transitions"]) > int(instance.metrics["macro_decisions"]):
            raise ValueError("living motor transition count exceeds decisions")
        return instance

    @classmethod
    def restore_value(
        cls,
        value: Any,
        artifact: MotorArtifact | str | Path | None = None,
    ) -> "LivingMotorOrgan":
        """Restore private state against a shared or embedded immutable artifact."""
        if artifact is None:
            if not isinstance(value, dict) or "artifact" not in value:
                raise ValueError("living motor restore requires its shared artifact")
            artifact = MotorArtifact.from_value(value["artifact"])
        return cls.restore(value, artifact)

    @staticmethod
    def _restore_pending(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {
            "feature", "before_physiology", "action", "outcome", "outcomes_seen",
        }:
            raise ValueError("invalid pending living motor transition")
        physiology = ContextualMotorRefiner.physiology(value["before_physiology"])
        action = _vector(value["action"], len(MOTOR_ACTIONS), "pending action")
        if np.any(np.abs(action) > 1) or np.any(action[[3, 4, 5, 6]] < 0):
            raise ValueError("invalid pending physical action")
        outcome = LivingMotorOrgan._physics_outcome(value["outcome"])
        seen = int(value["outcomes_seen"])
        if isinstance(value["outcomes_seen"], bool) or not 0 <= seen <= MACRO_STEPS:
            raise ValueError("invalid pending outcome count")
        return {
            "feature": _vector(value["feature"], 64, "pending feature"),
            "before_physiology": physiology,
            "action": action,
            "outcome": outcome,
            "outcomes_seen": seen,
        }

    @staticmethod
    def _restore_metrics(value: Any) -> dict[str, int | float]:
        expected = {
            "ticks", "macro_decisions", "recorded_transitions",
            "context_changed_choices", "experienced_nutrition",
            "experienced_effort", "contacted_transitions", "episodes",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("invalid living motor metrics")
        result: dict[str, int | float] = {}
        float_fields = {"experienced_nutrition", "experienced_effort"}
        for name in expected:
            number = _finite(value[name], f"metric {name}")
            if number < 0:
                raise ValueError("living motor metrics cannot be negative")
            if name in float_fields:
                result[name] = number
            else:
                integer = int(number)
                if isinstance(value[name], bool) or number != integer:
                    raise ValueError(f"metric {name} must be an integer")
                result[name] = integer
        return result

    def _validate_pending_state(self) -> None:
        held = self.motor.held_ticks
        if held == 0:
            if self.pending is not None:
                raise ValueError("open motor boundary cannot have a pending transition")
            return
        if self.pending is None:
            raise ValueError("held motor action requires a pending transition")
        if int(self.pending["outcomes_seen"]) != held - 1:
            raise ValueError("pending outcome count differs from held motor ticks")
        if not np.array_equal(self.pending["action"], self.motor.held_action):
            raise ValueError("pending action differs from inherited held action")


__all__ = [
    "FORMAT", "PHYSICS_DT", "MACRO_STEPS", "LEGACY_PROFILE",
    "RESEARCH_PROFILE", "CUSTOM_PROFILE", "RELATIONAL_PROFILES",
    "LivingMotorOrgan",
]
