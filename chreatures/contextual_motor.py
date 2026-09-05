"""Conservative relational-memory refinement of inherited motor candidates.

The refiner does not generate goals or unrestricted actions.  At a macro
boundary it selects among a small set drawn from an inherited continuous policy,
using only private experienced sensory/action/outcome relations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Any, Mapping

import numpy as np

from .context_memory import RelationalContextMemory
from .motor_inheritance import ACTIONS as MOTOR_ACTIONS, MotorOrgan


MEMORY_OUTCOMES = (
    "energy_delta", "gut_delta", "fatigue_delta",
    "nutrition", "effort", "contact",
)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class ContextualMotorConfig:
    candidate_count: int = 5
    candidate_std_scale: float = 0.42
    max_logit_correction: float = 0.32
    correction_gain: float = 1.6
    minimum_support: float = 0.5
    full_support: float = 5.0
    uncertainty_ceiling: float = 0.58
    feature_value_min_count: int = 8
    feature_value_rate: float = 0.04
    feature_value_gain: float = 0.18
    energy_target: float = 0.85
    gut_target: float = 0.18
    energy_drive_weight: float = 12.0
    gut_drive_weight: float = 0.35
    fatigue_drive_weight: float = 0.50
    nutrition_weight: float = 3.0
    effort_weight: float = 0.0002

    def validate(self) -> None:
        if not 1 <= self.candidate_count <= 9:
            raise ValueError("candidate_count must be in 1..9")
        if not 0 < self.candidate_std_scale <= 1.0:
            raise ValueError("candidate_std_scale must be in (0, 1]")
        if not 0 < self.max_logit_correction <= 0.5 or self.correction_gain <= 0:
            raise ValueError("invalid contextual correction bound or gain")
        if not 0 <= self.minimum_support < self.full_support:
            raise ValueError("support gates are inconsistent")
        if not 0 < self.uncertainty_ceiling < 1:
            raise ValueError("uncertainty_ceiling must be in (0, 1)")
        if self.feature_value_min_count < 1 or not 0 < self.feature_value_rate <= 0.2:
            raise ValueError("invalid feature-value learning parameters")
        numeric = asdict(self)
        if any(isinstance(value, float) and not math.isfinite(value) for value in numeric.values()):
            raise ValueError("contextual motor configuration must be finite")


class ContextualMotorRefiner:
    """Private uncertainty-gated reranker for inherited action candidates."""

    VERSION = 1

    def __init__(
        self,
        feature_dim: int,
        *,
        seed: int = 0,
        config: ContextualMotorConfig | None = None,
        enabled: bool = True,
    ):
        if not 2 <= int(feature_dim) <= 4096:
            raise ValueError("feature_dim must be in 2..4096")
        self.feature_dim = int(feature_dim)
        self.config = config or ContextualMotorConfig()
        self.config.validate()
        self.rng = np.random.default_rng(int(seed))
        self.enabled = bool(enabled)
        self.learning = True
        self.feature_value = np.zeros(self.feature_dim + 1, dtype=np.float32)
        self.feature_value_count = 0
        self.memory_revision = 0
        self.decision_count = 0
        self.last_decision: dict[str, Any] | None = None

    @staticmethod
    def physiology(value: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError("physiology must be a mapping")
        required = {"energy", "gut", "fatigue"}
        if not required <= set(value):
            raise ValueError("physiology requires energy, gut and fatigue")
        result = {name: float(value[name]) for name in required}
        if any(not math.isfinite(number) or not 0 <= number <= 1 for number in result.values()):
            raise ValueError("physiology values must be finite in [0, 1]")
        return result

    def _drive(self, physiology: Mapping[str, float]) -> float:
        c = self.config
        return (
            c.energy_drive_weight * (c.energy_target - physiology["energy"]) ** 2
            + c.gut_drive_weight * (c.gut_target - physiology["gut"]) ** 2
            + c.fatigue_drive_weight * physiology["fatigue"] ** 2
        )

    def _experienced_utility(
        self,
        before: Mapping[str, float],
        after: Mapping[str, float],
        physical_outcome: Mapping[str, Any],
    ) -> float:
        nutrition = max(0.0, float(physical_outcome.get("nutrition", 0.0)))
        effort = max(0.0, float(physical_outcome.get("effort", 0.0)))
        if not math.isfinite(nutrition) or not math.isfinite(effort):
            raise ValueError("physical outcome must be finite")
        return float(
            self._drive(before) - self._drive(after)
            + self.config.nutrition_weight * nutrition * max(0.0, 1.0 - after["energy"])
            - self.config.effort_weight * effort
        )

    def memory_outcome(
        self,
        before_physiology: Mapping[str, Any],
        after_physiology: Mapping[str, Any],
        physical_outcome: Mapping[str, Any],
    ) -> np.ndarray:
        if not isinstance(physical_outcome, Mapping):
            raise ValueError("physical outcome must be a mapping")
        before, after = self.physiology(before_physiology), self.physiology(after_physiology)
        nutrition = float(physical_outcome.get("nutrition", 0.0))
        effort = float(physical_outcome.get("effort", 0.0))
        contact = float(physical_outcome.get("contact", 0.0))
        if (
            not math.isfinite(nutrition) or nutrition < 0
            or not math.isfinite(effort) or effort < 0
            or not math.isfinite(contact) or not 0 <= contact <= 1
        ):
            raise ValueError("nutrition/effort must be nonnegative and contact must be in [0, 1]")
        values = {
            "energy_delta": after["energy"] - before["energy"],
            "gut_delta": after["gut"] - before["gut"],
            "fatigue_delta": after["fatigue"] - before["fatigue"],
            "nutrition": nutrition,
            "effort": effort,
            "contact": contact,
        }
        return _vector([values[name] for name in MEMORY_OUTCOMES], len(MEMORY_OUTCOMES), "memory outcome")

    @staticmethod
    def _check_memory(memory: RelationalContextMemory, feature_dim: int) -> None:
        dimensions = memory.config
        if (
            dimensions.feature_dim != feature_dim
            or dimensions.action_dim != len(MOTOR_ACTIONS)
            or dimensions.outcome_dim != len(MEMORY_OUTCOMES)
        ):
            raise ValueError("relational memory dimensions differ from contextual motor contract")

    def candidates(
        self,
        policy_mean: Any,
        policy_log_std: Any,
        *,
        baseline_latent: Any | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw an ancestral policy action and bounded alternatives around it."""
        mean = _vector(policy_mean, len(MOTOR_ACTIONS), "policy mean")
        log_std = np.clip(
            _vector(policy_log_std, len(MOTOR_ACTIONS), "policy log_std"), -3.5, 0.3
        )
        std = np.exp(log_std).astype(np.float32)
        if baseline_latent is None:
            baseline = mean.copy() if deterministic else (
                mean + std * self.rng.standard_normal(len(MOTOR_ACTIONS), dtype=np.float32)
            )
        else:
            baseline = _vector(baseline_latent, len(MOTOR_ACTIONS), "baseline latent")
            if deterministic and not np.array_equal(baseline, mean):
                raise ValueError("a deterministic baseline must equal the policy mean")
        count = self.config.candidate_count
        latent = np.empty((count, len(MOTOR_ACTIONS)), dtype=np.float32)
        latent[0] = baseline
        if count > 1:
            noise = self.rng.standard_normal((count - 1, len(MOTOR_ACTIONS)), dtype=np.float32)
            latent[1:] = baseline + (
                std * np.float32(self.config.candidate_std_scale)
            ) * noise
        action = np.tanh(latent).astype(np.float32)
        # Match MotorOrgan.tick's projection from Gaussian action to the
        # physical schema: grip and emission strengths cannot be negative.
        action[:, [3, 4, 5, 6]] = np.maximum(action[:, [3, 4, 5, 6]], 0)
        # Mean log-density per coordinate keeps inherited score differences on
        # the same scale as the bounded contextual correction.
        z = (latent - baseline) / std
        inherited = (-0.5 * np.mean(z * z, axis=1)).astype(np.float32)
        return action, inherited, latent

    def _predicted_utility(
        self,
        prediction: Mapping[str, Any],
        current_physiology: Mapping[str, float],
    ) -> tuple[float, float, dict[str, float]]:
        outcome = _vector(prediction["outcome"], len(MEMORY_OUTCOMES), "predicted outcome")
        named = {name: float(value) for name, value in zip(MEMORY_OUTCOMES, outcome, strict=True)}
        after = {
            "energy": float(np.clip(current_physiology["energy"] + named["energy_delta"], 0, 1)),
            "gut": float(np.clip(current_physiology["gut"] + named["gut_delta"], 0, 1)),
            "fatigue": float(np.clip(current_physiology["fatigue"] + named["fatigue_delta"], 0, 1)),
        }
        utility = (
            self._drive(current_physiology) - self._drive(after)
            + self.config.nutrition_weight * max(0.0, named["nutrition"]) * max(0.0, 1.0 - after["energy"])
            - self.config.effort_weight * max(0.0, named["effort"])
        )
        feature_term = 0.0
        if self.feature_value_count >= self.config.feature_value_min_count:
            next_feature = _vector(
                prediction["next_observation"], self.feature_dim, "predicted next feature"
            )
            joined = np.concatenate((next_feature, [1.0])).astype(np.float32)
            feature_term = float(np.clip(self.feature_value @ joined, -1.0, 1.0))
            maturity = self.feature_value_count / (self.feature_value_count + 24.0)
            utility += self.config.feature_value_gain * maturity * feature_term
        return float(utility), feature_term, named

    def refine(
        self,
        memory: RelationalContextMemory,
        feature: Any,
        physiology: Mapping[str, Any],
        policy_mean: Any,
        policy_log_std: Any,
        *,
        policy_artifact_sha256: str | None = None,
        candidate_actions: Any | None = None,
        inherited_scores: Any | None = None,
        baseline_latent: Any | None = None,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Choose one inherited candidate and return complete decision provenance."""
        self._check_memory(memory, self.feature_dim)
        feature_value = _vector(feature, self.feature_dim, "feature")
        local = self.physiology(physiology)
        mean_value = _vector(policy_mean, len(MOTOR_ACTIONS), "policy mean")
        log_std_value = np.clip(
            _vector(policy_log_std, len(MOTOR_ACTIONS), "policy log_std"), -3.5, 0.3
        )
        if policy_artifact_sha256 is not None and (
            not isinstance(policy_artifact_sha256, str)
            or len(policy_artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in policy_artifact_sha256)
        ):
            raise ValueError("policy_artifact_sha256 must be a lowercase SHA-256 hex digest")
        if memory.current_observation is None:
            memory.begin(feature_value, learn=False)
        elif not np.allclose(memory.current_observation, feature_value, rtol=0.0, atol=1e-6):
            raise ValueError(
                "feature differs from relational memory's current observation; "
                "call memory.begin(feature) at the macro boundary"
            )
        if candidate_actions is None:
            actions, inherited, latent = self.candidates(
                mean_value, log_std_value,
                baseline_latent=baseline_latent, deterministic=deterministic,
            )
            candidate_source = (
                "deterministic inherited mean plus bounded local alternatives"
                if deterministic else
                "stochastic ancestral Gaussian baseline plus bounded local alternatives"
            )
            latent_basis = "exact generated pre-tanh candidates"
            baseline_mode = "deterministic mean" if deterministic else "ancestral Gaussian draw"
        else:
            if baseline_latent is not None:
                raise ValueError("baseline_latent cannot accompany caller-supplied candidates")
            actions = np.asarray(candidate_actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != len(MOTOR_ACTIONS) or not 1 <= len(actions) <= 9:
                raise ValueError("candidate_actions must have shape 1..9 by 8")
            if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                raise ValueError("candidate actions must be finite in [-1, 1]")
            if np.any(actions[:, [3, 4, 5, 6]] < 0):
                raise ValueError("grip and signal candidate values must be nonnegative")
            latent = np.arctanh(np.clip(actions, -0.999999, 0.999999)).astype(np.float32)
            if inherited_scores is None:
                baseline = latent[0]
                inherited = (-0.5 * np.mean(
                    ((latent - baseline) / np.exp(log_std_value)) ** 2, axis=1
                )).astype(np.float32)
            else:
                inherited = _vector(inherited_scores, len(actions), "inherited scores")
                if float(inherited[0]) < float(np.max(inherited)):
                    raise ValueError("candidate zero must be the inherited baseline")
            candidate_source = "caller-supplied inherited candidates"
            latent_basis = "inverse-tanh proxy for supplied physical candidates"
            baseline_mode = "caller-supplied physical baseline"

        audits = []
        total = inherited.astype(np.float64).copy()
        for index, action in enumerate(actions):
            prediction = memory.predict(action)
            utility, feature_term, predicted_outcome = self._predicted_utility(prediction, local)
            support = float(prediction["support"])
            uncertainty = float(prediction["uncertainty"])
            support_gate = float(np.clip(
                (support - self.config.minimum_support)
                / (self.config.full_support - self.config.minimum_support), 0.0, 1.0
            ))
            uncertainty_gate = float(np.clip(
                (self.config.uncertainty_ceiling - uncertainty)
                / self.config.uncertainty_ceiling, 0.0, 1.0
            ))
            gate = support_gate * uncertainty_gate if self.enabled else 0.0
            correction = float(np.clip(
                self.config.correction_gain * utility * gate,
                -self.config.max_logit_correction,
                self.config.max_logit_correction,
            ))
            total[index] += correction
            audits.append({
                "index": index,
                "action": action.astype(float).tolist(),
                "pre_tanh_candidate": latent[index].astype(float).tolist(),
                "inherited_score": float(inherited[index]),
                "predicted_utility": utility,
                "predicted_feature_value": feature_term,
                "predicted_outcome": predicted_outcome,
                "support": support,
                "uncertainty": uncertainty,
                "coverage_gate": gate,
                "contextual_correction": correction,
                "combined_score": float(total[index]),
            })
        inherited_choice = int(np.argmax(inherited))
        selected = int(np.argmax(total))
        self.decision_count += 1
        decision = {
            "action": {
                name: float(value) for name, value in zip(MOTOR_ACTIONS, actions[selected], strict=True)
            },
            "action_vector": actions[selected].astype(float).tolist(),
            "selected_index": selected,
            "inherited_selected_index": inherited_choice,
            "context_changed_choice": selected != inherited_choice,
            "candidates": audits,
            "provenance": {
                "kind": "uncertainty-gated relational reranking",
                "candidate_source": candidate_source,
                "candidate_latent_basis": latent_basis,
                "baseline_mode": baseline_mode,
                "inherited_score_basis": "penalty relative to candidate-zero baseline",
                "policy_artifact_sha256": policy_artifact_sha256,
                "pre_tanh_mean": mean_value.astype(float).tolist(),
                "log_std": log_std_value.astype(float).tolist(),
                "memory_revision": self.memory_revision,
                "decision_count": self.decision_count,
                "max_logit_correction": self.config.max_logit_correction,
                "uncertainty_status": "heuristic; weak empirical calibration, not a probability",
            },
        }
        self.last_decision = copy.deepcopy(decision)
        return decision

    def record(
        self,
        memory: RelationalContextMemory,
        feature: Any,
        executed_action: Any,
        next_feature: Any,
        before_physiology: Mapping[str, Any],
        after_physiology: Mapping[str, Any],
        physical_outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind an actual macro transition and update private feature value."""
        self._check_memory(memory, self.feature_dim)
        current = _vector(feature, self.feature_dim, "feature")
        action = _vector(executed_action, len(MOTOR_ACTIONS), "executed action")
        if np.any(np.abs(action) > 1):
            raise ValueError("executed action must be in [-1, 1]")
        if np.any(action[[3, 4, 5, 6]] < 0):
            raise ValueError("executed grip and signal values must be nonnegative")
        following = _vector(next_feature, self.feature_dim, "next feature")
        before, after = self.physiology(before_physiology), self.physiology(after_physiology)
        outcome = self.memory_outcome(before, after, physical_outcome)
        if memory.current_observation is None:
            memory.begin(current, learn=self.learning)
        elif not np.allclose(memory.current_observation, current, rtol=0.0, atol=1e-6):
            raise ValueError(
                "feature differs from relational memory's current observation; "
                "record transitions in execution order or call memory.begin(feature)"
            )
        transition = memory.step(action, following, outcome, learn=self.learning)
        utility = self._experienced_utility(before, after, physical_outcome)
        if self.learning:
            joined = np.concatenate((following, [1.0])).astype(np.float32)
            error = utility - float(self.feature_value @ joined)
            self.feature_value += np.float32(
                self.config.feature_value_rate * error / max(1.0, float(joined @ joined))
            ) * joined
            np.clip(self.feature_value, -2.0, 2.0, out=self.feature_value)
            self.feature_value_count += 1
            self.memory_revision += 1
        return {
            "memory_revision": self.memory_revision,
            "experienced_utility": utility,
            "context": transition.get("context"),
            "contexts": memory.context_count,
            "transitions": memory.transition_count,
        }

    def refine_and_commit(
        self,
        motor: MotorOrgan,
        memory: RelationalContextMemory,
        feature: Any,
        raw_motor_senses: Any,
        local_physiology: Mapping[str, Any],
        dt: float,
    ) -> dict[str, Any]:
        """Refine and install the first tick of a MotorOrgan macro action."""
        if not isinstance(motor, MotorOrgan):
            raise ValueError("motor must be a MotorOrgan")
        if motor.held_ticks != 0:
            raise ValueError("contextual selection requires an open motor macro boundary")
        if not math.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        normalized = motor.normalize(raw_motor_senses)
        motor_physiology = motor.physiology_vector(local_physiology)
        mean, inherited_value, hidden = motor.forward(normalized, motor_physiology)
        log_std = np.clip(motor.artifact.arrays["log_std"], -3.5, 0.3).astype(np.float32)
        baseline_latent = mean if motor.deterministic else (
            mean + np.exp(log_std).astype(np.float32)
            * motor.rng.standard_normal(len(MOTOR_ACTIONS), dtype=np.float32)
        )
        decision = self.refine(
            memory,
            feature,
            local_physiology,
            mean,
            log_std,
            policy_artifact_sha256=motor.artifact.sha256,
            baseline_latent=baseline_latent,
            deterministic=motor.deterministic,
        )
        decision["action"] = motor.commit_macro_action(
            normalized, hidden, decision["action_vector"], float(dt)
        )
        decision["provenance"].update({
            "inherited_value": float(inherited_value),
            "motor_macro_steps": int(motor.artifact.config["macro_steps"]),
            "motor_commit": "MotorOrgan.commit_macro_action",
        })
        self.last_decision = copy.deepcopy(decision)
        return decision

    def freeze(self, frozen: bool = True) -> None:
        self.learning = not bool(frozen)

    def clear(self, memory: RelationalContextMemory) -> None:
        """Clear this resident's optional learned refinement state explicitly."""
        self._check_memory(memory, self.feature_dim)
        memory.__init__(memory.config)
        self.feature_value.fill(0.0)
        self.feature_value_count = 0
        self.memory_revision += 1
        self.last_decision = None

    def snapshot(self, memory: RelationalContextMemory) -> dict[str, Any]:
        self._check_memory(memory, self.feature_dim)
        return {
            "version": self.VERSION,
            "feature_dim": self.feature_dim,
            "config": asdict(self.config),
            "enabled": self.enabled,
            "learning": self.learning,
            "feature_value": self.feature_value.tolist(),
            "feature_value_count": self.feature_value_count,
            "memory_revision": self.memory_revision,
            "decision_count": self.decision_count,
            "last_decision": copy.deepcopy(self.last_decision),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "memory": memory.snapshot(),
        }

    @classmethod
    def restore(
        cls, value: Any
    ) -> tuple["ContextualMotorRefiner", RelationalContextMemory]:
        if not isinstance(value, dict) or value.get("version") != cls.VERSION:
            raise ValueError("unsupported contextual motor snapshot")
        if not isinstance(value.get("enabled"), bool) or not isinstance(value.get("learning"), bool):
            raise ValueError("invalid contextual motor flags")
        instance = cls(
            int(value["feature_dim"]),
            config=ContextualMotorConfig(**value["config"]),
            enabled=bool(value["enabled"]),
        )
        instance.learning = bool(value["learning"])
        instance.feature_value = _vector(
            value["feature_value"], instance.feature_dim + 1, "feature value"
        )
        instance.feature_value_count = int(value["feature_value_count"])
        instance.memory_revision = int(value["memory_revision"])
        instance.decision_count = int(value["decision_count"])
        if min(instance.feature_value_count, instance.memory_revision, instance.decision_count) < 0:
            raise ValueError("invalid contextual motor counters")
        if value.get("last_decision") is not None and not isinstance(value.get("last_decision"), dict):
            raise ValueError("invalid last contextual decision")
        instance.last_decision = copy.deepcopy(value["last_decision"])
        try:
            instance.rng.bit_generator.state = copy.deepcopy(value["rng_state"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid contextual motor RNG state") from error
        memory = RelationalContextMemory.restore(value["memory"])
        instance._check_memory(memory, instance.feature_dim)
        return instance, memory


__all__ = [
    "MOTOR_ACTIONS", "MEMORY_OUTCOMES", "ContextualMotorConfig",
    "ContextualMotorRefiner",
]
