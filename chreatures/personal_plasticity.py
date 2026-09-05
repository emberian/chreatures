"""Private lifetime adaptation around an immutable inherited motor policy.

This module owns no world policy.  It receives an anonymous frozen feature
vector, an inherited Gaussian motor distribution, the action actually sampled
from that distribution, and the later measured physiological transition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from .homeostasis import FiniteEnergyConfig, FiniteEnergyObjective, FORMAT as ENERGY_FORMAT
from .motor_inheritance import ACTIONS


FORMAT = "chreatures-personal-motor-plasticity-v1"
DECISION_FORMAT = "chreatures-personal-motor-decision-v1"
CONSERVATIVE_CREDIT = "executed-match-v1"
LATENT_PROPOSAL_CREDIT = "latent-proposal-v2"
CREDIT_ASSIGNMENTS = frozenset((CONSERVATIVE_CREDIT, LATENT_PROPOSAL_CREDIT))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_value(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "data": array.tolist()}


def _array_from_value(value: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(item) for item in value["shape"])
        array = np.asarray(value["data"], dtype=dtype)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid encoded array {name}") from exc
    if array.shape != shape or not np.issubdtype(dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"invalid encoded array {name}")
    return array


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def _physiology(value: Mapping[str, Any], name: str) -> dict[str, float]:
    required = {"energy", "gut", "fatigue"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{name} physiology must contain only energy, gut and fatigue")
    result = {key: float(value[key]) for key in required}
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result.values()):
        raise ValueError(f"{name} physiology values must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class PersonalPlasticityConfig:
    """Bounded online actor-critic configuration.

    The actor has only ``action_dim * (feature_dim + 1)`` learned parameters.
    Its output is capped in inherited pre-tanh action units.
    """

    version: int = 1
    feature_dim: int = 64
    action_dim: int = len(ACTIONS)
    max_mean_offset: float = 0.32
    actor_learning_rate: float = 0.018
    critic_learning_rate: float = 0.055
    discount_seconds: float = 8.0
    trace_decay: float = 0.72
    weight_decay: float = 0.001
    gradient_clip: float = 2.5
    advantage_clip: float = 1.5
    weight_clip: float = 2.0
    credit_assignment: str = LATENT_PROPOSAL_CREDIT
    seed: int = 20260907

    def __post_init__(self) -> None:
        if self.version != 1 or self.feature_dim < 1 or self.action_dim != len(ACTIONS):
            raise ValueError("unsupported personal plasticity dimensions/version")
        values = (
            self.max_mean_offset, self.actor_learning_rate, self.critic_learning_rate,
            self.discount_seconds, self.gradient_clip, self.advantage_clip, self.weight_clip,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("personal plasticity scales must be finite and positive")
        if not math.isfinite(self.trace_decay) or not 0.0 <= self.trace_decay <= 1.0:
            raise ValueError("trace_decay must be in [0, 1]")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if self.credit_assignment not in CREDIT_ASSIGNMENTS:
            raise ValueError("unsupported personal plasticity credit assignment")


@dataclass(frozen=True)
class MotorDecision:
    """One auditable sample from the adapted inherited Gaussian."""

    sequence: int
    feature: np.ndarray
    inherited_mean: np.ndarray
    adapted_mean: np.ndarray
    log_std: np.ndarray
    standard_noise: np.ndarray
    latent_action: np.ndarray
    proposal_action: np.ndarray
    physical_action: np.ndarray
    baseline_value: float
    proposal_log_probability: float
    actor_eligible: bool
    provenance: str
    credit_assignment: str
    selection_pipeline: str
    proposal_independent_selection: bool

    def to_value(self) -> dict[str, Any]:
        return {
            "format": DECISION_FORMAT,
            "sequence": int(self.sequence),
            "baseline_value": float(self.baseline_value),
            "proposal_log_probability": float(self.proposal_log_probability),
            "actor_eligible": bool(self.actor_eligible),
            "provenance": self.provenance,
            "credit_assignment": self.credit_assignment,
            "selection_pipeline": self.selection_pipeline,
            "proposal_independent_selection": self.proposal_independent_selection,
            "arrays": {
                name: _array_value(getattr(self, name))
                for name in (
                    "feature", "inherited_mean", "adapted_mean", "log_std",
                    "standard_noise", "latent_action", "proposal_action", "physical_action",
                )
            },
        }

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "MotorDecision":
        try:
            if value.get("format") != DECISION_FORMAT:
                raise ValueError("unsupported personal motor decision")
            arrays = value["arrays"]
            return cls(
                sequence=int(value["sequence"]),
                baseline_value=float(value["baseline_value"]),
                proposal_log_probability=float(value["proposal_log_probability"]),
                actor_eligible=bool(value["actor_eligible"]),
                provenance=str(value["provenance"]),
                credit_assignment=str(value["credit_assignment"]),
                selection_pipeline=str(value["selection_pipeline"]),
                proposal_independent_selection=bool(value["proposal_independent_selection"]),
                **{name: _array_from_value(arrays[name], name).astype(np.float32, copy=False)
                   for name in (
                       "feature", "inherited_mean", "adapted_mean", "log_std",
                       "standard_noise", "latent_action", "proposal_action", "physical_action",
                   )},
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid personal motor decision") from exc


class PersonalMotorPlasticity:
    """A small private score-function adapter around inherited motor means.

    The inherited parameters are inputs and are never retained or modified.
    Exactly one committed macro action may be pending.  ``observe`` consumes it
    after the world's measured consequence becomes available.
    """

    def __init__(
        self,
        config: PersonalPlasticityConfig | None = None,
        *,
        objective_config: FiniteEnergyConfig | None = None,
        enabled: bool = True,
        learning: bool = True,
    ) -> None:
        self.config = config or PersonalPlasticityConfig()
        self.objective = FiniteEnergyObjective(objective_config)
        self.enabled = bool(enabled)
        self.learning = bool(learning)
        self.rng = np.random.default_rng(int(self.config.seed))
        width = self.config.feature_dim + 1
        shape = (self.config.action_dim, width)
        self.actor = np.zeros(shape, dtype=np.float32)
        self.actor_trace = np.zeros(shape, dtype=np.float32)
        self.critic = np.zeros(width, dtype=np.float32)
        self.critic_trace = np.zeros(width, dtype=np.float32)
        self.pending: MotorDecision | None = None
        self.decision_count = 0
        self.update_count = 0
        self.actor_update_count = 0
        self.actor_skip_count = 0
        self.reward_sum = 0.0
        self.last_reward: float | None = None
        self.last_td_error: float | None = None
        self.last_offset = np.zeros(self.config.action_dim, dtype=np.float32)
        self.last_objective: dict[str, float] | None = None

    def _basis(self, feature: Any) -> np.ndarray:
        feature = _vector(feature, self.config.feature_dim, "feature")
        if np.any(np.abs(feature) > 1.000001):
            raise ValueError("feature must be a bounded frozen projection in [-1, 1]")
        # Fixed scaling bounds a whole-vector update while retaining magnitude.
        return np.concatenate((feature, np.ones(1, dtype=np.float32))) / math.sqrt(
            self.config.feature_dim + 1
        )

    def mean_offset(self, feature: Any) -> np.ndarray:
        """Return a bounded private offset in inherited pre-tanh units."""
        if not self.enabled:
            return np.zeros(self.config.action_dim, dtype=np.float32)
        basis = self._basis(feature)
        raw = self.actor @ basis
        return (self.config.max_mean_offset * np.tanh(raw)).astype(np.float32)

    def adapt_mean(self, feature: Any, inherited_mean: Any) -> np.ndarray:
        """Adapt a mean; disabled mode returns an unmodified copy bit-for-bit."""
        inherited = _vector(inherited_mean, self.config.action_dim, "inherited_mean")
        if not self.enabled:
            return inherited.copy()
        return (inherited + self.mean_offset(feature)).astype(np.float32)

    @staticmethod
    def physical_action(latent_action: Any) -> np.ndarray:
        latent = _vector(latent_action, len(ACTIONS), "latent_action")
        action = np.tanh(latent).astype(np.float32)
        action[3:7] = np.maximum(action[3:7], 0.0)
        return action

    def propose(
        self,
        feature: Any,
        inherited_mean: Any,
        log_std: Any,
        *,
        inherited_noise: Any | None = None,
        deterministic: bool = False,
    ) -> MotorDecision:
        """Sample a proposal without committing it.

        A caller that already drew from the inherited RNG can pass that exact
        standard-normal ``inherited_noise``.  With adaptation disabled, the
        resulting latent and action then exactly match the inherited sample.
        """
        feature = _vector(feature, self.config.feature_dim, "feature")
        inherited = _vector(inherited_mean, self.config.action_dim, "inherited_mean")
        log_std = _vector(log_std, self.config.action_dim, "log_std")
        if np.any(log_std < -8.0) or np.any(log_std > 2.0):
            raise ValueError("log_std is outside the supported range [-8, 2]")
        mean = self.adapt_mean(feature, inherited)
        if deterministic:
            latent = mean.copy()
            noise = np.zeros(self.config.action_dim, dtype=np.float32)
        else:
            noise = (
                self.rng.standard_normal(self.config.action_dim, dtype=np.float32)
                if inherited_noise is None
                else _vector(inherited_noise, self.config.action_dim, "inherited_noise")
            )
            std = np.exp(np.clip(log_std, -3.5, 0.3)).astype(np.float32)
            latent = (mean + std * noise).astype(np.float32)
        basis = self._basis(feature)
        value = float(self.critic @ basis) if self.enabled else 0.0
        effective_log_std = np.clip(log_std.astype(np.float64), -3.5, 0.3)
        proposal_log_probability = float(-0.5 * np.sum(
            noise.astype(np.float64) ** 2 + 2.0 * effective_log_std
            + math.log(2.0 * math.pi)
        ))
        physical = self.physical_action(latent)
        return MotorDecision(
            sequence=self.decision_count,
            feature=feature.copy(), inherited_mean=inherited.copy(), adapted_mean=mean,
            log_std=log_std.copy(), standard_noise=noise.copy(), latent_action=latent,
            proposal_action=physical.copy(), physical_action=physical,
            baseline_value=value, proposal_log_probability=proposal_log_probability,
            actor_eligible=bool(self.enabled and not deterministic),
            provenance="personal_gaussian" if self.enabled else "inherited_unchanged",
            credit_assignment=self.config.credit_assignment,
            selection_pipeline="direct-proposal-v1",
            proposal_independent_selection=True,
        )

    def commit(
        self, decision: MotorDecision, *, executed_action: Any | None = None,
        provenance: str | None = None,
        selection_pipeline: str = "direct-proposal-v1",
        proposal_independent: bool | None = None,
    ) -> np.ndarray:
        """Record the action and declare how it was selected from the proposal.

        ``latent-proposal-v2`` permits score credit through a downstream
        selector only when the caller declares that selector conditionally
        independent of private actor parameters given the sampled baseline.
        Unknown selection or caller-supplied actions retain the v1 skip.
        """
        if self.pending is not None:
            raise RuntimeError("the previous personal motor decision has no outcome")
        self._validate_decision(decision)
        if decision.sequence != self.decision_count:
            raise ValueError("personal motor decision sequence differs")
        changed = False
        if executed_action is not None:
            executed = _vector(executed_action, self.config.action_dim, "executed_action")
            if np.any(executed < -1.0) or np.any(executed > 1.0) or np.any(executed[3:7] < 0.0):
                raise ValueError("executed_action is outside its physical bounds")
            changed = not np.array_equal(executed, decision.proposal_action)
            decision = replace(decision, physical_action=executed.copy())
        if not isinstance(selection_pipeline, str) or not selection_pipeline:
            raise ValueError("selection_pipeline must be a nonempty string")
        independent = (
            selection_pipeline == "direct-proposal-v1" and not changed
            if proposal_independent is None else bool(proposal_independent)
        )
        eligible = bool(
            decision.actor_eligible
            and (
                (
                    decision.credit_assignment == LATENT_PROPOSAL_CREDIT
                    and independent
                )
                or (
                    decision.credit_assignment == CONSERVATIVE_CREDIT
                    and selection_pipeline == "direct-proposal-v1"
                    and not changed
                )
            )
        )
        decision = replace(
            decision,
            actor_eligible=eligible,
            selection_pipeline=selection_pipeline,
            proposal_independent_selection=independent,
        )
        if provenance is not None:
            decision = replace(decision, provenance=str(provenance))
        self.pending = MotorDecision.from_value(decision.to_value())
        self.decision_count += 1
        self.last_offset = (decision.adapted_mean - decision.inherited_mean).astype(np.float32)
        return decision.physical_action.copy()

    def begin(
        self, feature: Any, inherited_mean: Any, log_std: Any, *,
        inherited_noise: Any | None = None, deterministic: bool = False,
    ) -> np.ndarray:
        """Propose and commit one macro action."""
        return self.commit(self.propose(
            feature, inherited_mean, log_std,
            inherited_noise=inherited_noise, deterministic=deterministic,
        ))

    def discard_pending(self) -> None:
        """Clear an aborted world transition without assigning an outcome."""
        self.pending = None

    def observe(
        self,
        next_feature: Any,
        *,
        transitions: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    ) -> dict[str, Any]:
        """Consume one completed macro transition and optionally learn from it.

        Every transition is an actual physical tick containing ``before``,
        ``after``, ``nutrition``, ``effort`` and ``dt``.  No physiological
        intermediate is synthesized for the five-tick motor macro.
        """
        if self.pending is None:
            raise RuntimeError("no committed personal motor decision is pending")
        next_basis = self._basis(next_feature)
        reward, objective, duration = self._experienced_reward(transitions)

        decision = self.pending
        self.pending = None
        basis = self._basis(decision.feature)
        gamma = math.exp(-duration / self.config.discount_seconds)
        next_value = float(self.critic @ next_basis) if self.enabled else 0.0
        td_error = float(reward + gamma * next_value - decision.baseline_value)
        clipped_advantage = float(np.clip(
            td_error, -self.config.advantage_clip, self.config.advantage_clip
        ))

        actor_changed = False
        critic_changed = False
        if self.enabled and self.learning:
            trace_factor = gamma * self.config.trace_decay
            self.critic_trace = (trace_factor * self.critic_trace + basis).astype(np.float32)
            critic_trace = self._clipped(self.critic_trace, self.config.gradient_clip)
            self.critic *= np.float32(1.0 - self.config.critic_learning_rate * self.config.weight_decay)
            self.critic += np.float32(self.config.critic_learning_rate * clipped_advantage) * critic_trace
            np.clip(self.critic, -self.config.weight_clip, self.config.weight_clip, out=self.critic)
            critic_changed = True
            if decision.actor_eligible:
                std = np.exp(np.clip(decision.log_std, -3.5, 0.3)).astype(np.float32)
                score_mean = (decision.latent_action - decision.adapted_mean) / np.maximum(std * std, 1e-8)
                raw = self.actor @ basis
                derivative = self.config.max_mean_offset * (1.0 - np.tanh(raw) ** 2)
                actor_gradient = np.outer(score_mean * derivative, basis).astype(np.float32)
                self.actor_trace = (trace_factor * self.actor_trace + actor_gradient).astype(np.float32)
                actor_trace = self._clipped(self.actor_trace, self.config.gradient_clip)
                self.actor *= np.float32(1.0 - self.config.actor_learning_rate * self.config.weight_decay)
                self.actor += np.float32(self.config.actor_learning_rate * clipped_advantage) * actor_trace
                np.clip(self.actor, -self.config.weight_clip, self.config.weight_clip, out=self.actor)
                actor_changed = True
                self.actor_update_count += 1
            else:
                # An off-policy intervention breaks the causal eligibility
                # chain; a later on-policy reward must not revive old credit
                # through an action selected by an unknown distribution.
                self.actor_trace.fill(0.0)
                self.actor_skip_count += 1
            self.update_count += 1

        self.reward_sum += reward
        self.last_reward = reward
        self.last_td_error = td_error
        self.last_objective = objective
        return {
            "reward": reward,
            "td_error": td_error,
            "updated": critic_changed,
            "actor_updated": actor_changed,
            "actor_update_skipped": bool(self.enabled and self.learning and not decision.actor_eligible),
            "action_provenance": decision.provenance,
            "credit_assignment": decision.credit_assignment,
            "selection_pipeline": decision.selection_pipeline,
            "proposal_independent_selection": decision.proposal_independent_selection,
            "proposal_log_probability": decision.proposal_log_probability,
            "objective": dict(objective),
            "offset": self.mean_offset(next_feature).astype(float).tolist(),
        }

    @staticmethod
    def _clipped(value: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        return value if norm <= limit else value * np.float32(limit / max(norm, 1e-12))

    def _experienced_reward(
        self, transitions: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    ) -> tuple[float, dict[str, float], float]:
        """Sum objective-v1 over actual recorded physical ticks."""
        if not isinstance(transitions, (list, tuple)) or not transitions:
            raise ValueError("transitions must contain actual physical tick records")
        reward_total = 0.0
        potential_delta = 0.0
        effort_cost = 0.0
        nutrition_total = 0.0
        effort_integral = 0.0
        duration = 0.0
        previous_after: dict[str, float] | None = None
        for index, record in enumerate(transitions):
            if not isinstance(record, Mapping) or set(record) != {"before", "after", "nutrition", "effort", "dt"}:
                raise ValueError("each physical tick needs before/after/nutrition/effort/dt")
            before = _physiology(record["before"], f"transition {index} before")
            after = _physiology(record["after"], f"transition {index} after")
            if previous_after is not None and any(
                abs(before[key] - previous_after[key]) > 2e-6 for key in before
            ):
                raise ValueError("physical tick physiology is not contiguous")
            nutrition = float(record["nutrition"])
            effort = float(record["effort"])
            dt = float(record["dt"])
            reward, terms = self.objective.transition(
                before, after, nutrition=nutrition, effort=effort, dt=dt,
            )
            reward_total += float(reward)
            potential_delta += float(terms["potential_delta_energy"])
            effort_cost += float(terms["effort_cost_energy"])
            nutrition_total += nutrition
            effort_integral += effort * dt
            duration += dt
            previous_after = after
        return reward_total, {
            "reward": reward_total,
            "potential_delta_energy": potential_delta,
            "effort_cost_energy": effort_cost,
            "nutrition_observed": nutrition_total,
            "mean_effort": effort_integral / duration,
            "duration": duration,
            "physical_ticks": float(len(transitions)),
        }, duration

    def set_learning(self, value: bool) -> None:
        self.learning = bool(value)

    def set_enabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def view(self) -> dict[str, Any]:
        return {
            "kind": "personal_motor_plasticity",
            "format": FORMAT,
            "enabled": self.enabled,
            "learning": self.learning,
            "decision_count": self.decision_count,
            "update_count": self.update_count,
            "actor_update_count": self.actor_update_count,
            "actor_skip_count": self.actor_skip_count,
            "pending": self.pending is not None,
            "reward_sum": self.reward_sum,
            "last_reward": self.last_reward,
            "last_td_error": self.last_td_error,
            "last_offset": self.last_offset.astype(float).tolist(),
            "max_abs_offset": float(np.max(np.abs(self.last_offset))),
            "actor_norm": float(np.linalg.norm(self.actor)),
            "critic_norm": float(np.linalg.norm(self.critic)),
            "objective_format": ENERGY_FORMAT,
            "objective_sha256": self.objective.config.to_value()["sha256"],
            "credit_assignment": self.config.credit_assignment,
            "last_objective": None if self.last_objective is None else dict(self.last_objective),
        }

    def snapshot_value(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "format": FORMAT,
            "version": 1,
            "config": asdict(self.config),
            "objective_config": self.objective.config.to_value(),
            "enabled": self.enabled,
            "learning": self.learning,
            "rng": self.rng.bit_generator.state,
            "decision_count": self.decision_count,
            "update_count": self.update_count,
            "actor_update_count": self.actor_update_count,
            "actor_skip_count": self.actor_skip_count,
            "reward_sum": self.reward_sum,
            "last_reward": self.last_reward,
            "last_td_error": self.last_td_error,
            "last_objective": self.last_objective,
            "has_pending": self.pending is not None,
            "pending": None if self.pending is None else self.pending.to_value(),
            "arrays": {
                name: _array_value(getattr(self, name))
                for name in ("actor", "actor_trace", "critic", "critic_trace", "last_offset")
            },
        }
        value["sha256"] = hashlib.sha256(_json(value).encode()).hexdigest()
        return value

    @classmethod
    def restore_value(cls, value: Mapping[str, Any]) -> "PersonalMotorPlasticity":
        try:
            if value.get("format") != FORMAT or value.get("version") != 1:
                raise ValueError("unsupported personal plasticity snapshot")
            clean = dict(value)
            claimed = clean.pop("sha256")
            if claimed != hashlib.sha256(_json(clean).encode()).hexdigest():
                raise ValueError("personal plasticity snapshot checksum differs")
            config = PersonalPlasticityConfig(**value["config"])
            objective_config = FiniteEnergyConfig.from_value(value["objective_config"])
            instance = cls(
                config, objective_config=objective_config,
                enabled=bool(value["enabled"]), learning=bool(value["learning"]),
            )
            arrays = value["arrays"]
            for name in ("actor", "actor_trace", "critic", "critic_trace", "last_offset"):
                setattr(instance, name, _array_from_value(arrays[name], name).astype(np.float32, copy=False))
            instance.rng.bit_generator.state = value["rng"]
            instance.decision_count = int(value["decision_count"])
            instance.update_count = int(value["update_count"])
            instance.actor_update_count = int(value["actor_update_count"])
            instance.actor_skip_count = int(value["actor_skip_count"])
            instance.reward_sum = float(value["reward_sum"])
            instance.last_reward = None if value["last_reward"] is None else float(value["last_reward"])
            instance.last_td_error = None if value["last_td_error"] is None else float(value["last_td_error"])
            instance.last_objective = value["last_objective"]
            instance.pending = MotorDecision.from_value(value["pending"]) if value["has_pending"] else None
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("invalid personal plasticity snapshot") from exc
        instance._validate_state()
        return instance

    def _validate_decision(self, decision: MotorDecision) -> None:
        if not isinstance(decision, MotorDecision):
            raise TypeError("decision must be a MotorDecision")
        if decision.sequence < 0 or not math.isfinite(decision.baseline_value):
            raise ValueError("invalid personal motor decision metadata")
        _vector(decision.feature, self.config.feature_dim, "decision feature")
        for name in (
            "inherited_mean", "adapted_mean", "log_std", "latent_action",
            "standard_noise", "proposal_action", "physical_action",
        ):
            _vector(getattr(decision, name), self.config.action_dim, f"decision {name}")
        if not math.isfinite(decision.proposal_log_probability) or not decision.provenance:
            raise ValueError("invalid decision probability/provenance")
        if (
            decision.credit_assignment != self.config.credit_assignment
            or decision.credit_assignment not in CREDIT_ASSIGNMENTS
            or not decision.selection_pipeline
        ):
            raise ValueError("decision credit-assignment contract differs")
        expected = self.physical_action(decision.latent_action)
        if not np.array_equal(expected, decision.proposal_action):
            raise ValueError("proposal action does not match its sampled latent action")
        std = np.exp(np.clip(decision.log_std, -3.5, 0.3)).astype(np.float32)
        sampled = (decision.adapted_mean + std * decision.standard_noise).astype(np.float32)
        if not np.array_equal(sampled, decision.latent_action):
            raise ValueError("proposal latent action does not match its preserved raw noise")
        if decision.actor_eligible and not (
            (
                decision.credit_assignment == LATENT_PROPOSAL_CREDIT
                and decision.proposal_independent_selection
            )
            or (
                decision.credit_assignment == CONSERVATIVE_CREDIT
                and decision.selection_pipeline == "direct-proposal-v1"
                and np.array_equal(decision.proposal_action, decision.physical_action)
            )
        ):
            raise ValueError("decision is not eligible under its proposal-credit contract")

    def _validate_state(self) -> None:
        width = self.config.feature_dim + 1
        expected = {
            "actor": (self.config.action_dim, width),
            "actor_trace": (self.config.action_dim, width),
            "critic": (width,), "critic_trace": (width,),
            "last_offset": (self.config.action_dim,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"personal plasticity array {name} differs")
        if np.any(np.abs(self.actor) > self.config.weight_clip + 1e-6) or np.any(
            np.abs(self.critic) > self.config.weight_clip + 1e-6
        ):
            raise ValueError("personal plasticity weights exceed their declared bound")
        if np.any(np.abs(self.last_offset) > self.config.max_mean_offset + 1e-6):
            raise ValueError("personal plasticity offset exceeds its declared bound")
        if (
            self.decision_count < 0 or not 0 <= self.update_count <= self.decision_count
            or not 0 <= self.actor_update_count <= self.update_count
            or not 0 <= self.actor_skip_count <= self.update_count
            or self.actor_update_count + self.actor_skip_count > self.update_count
        ):
            raise ValueError("personal plasticity counters are invalid")
        if not math.isfinite(self.reward_sum):
            raise ValueError("personal plasticity reward sum is invalid")
        if self.pending is not None:
            self._validate_decision(self.pending)
            if self.pending.sequence != self.decision_count - 1:
                raise ValueError("pending personal motor decision sequence differs")


__all__ = [
    "FORMAT", "DECISION_FORMAT", "CONSERVATIVE_CREDIT", "LATENT_PROPOSAL_CREDIT",
    "CREDIT_ASSIGNMENTS", "PersonalPlasticityConfig", "MotorDecision",
    "PersonalMotorPlasticity",
]
