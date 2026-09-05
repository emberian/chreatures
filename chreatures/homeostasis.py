"""Versioned, opt-in finite-energy reward accounting for future training stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


FORMAT = "chreatures-finite-energy-homeostasis-v1"


@dataclass(frozen=True)
class FiniteEnergyConfig:
    """Coefficients expressed in body-energy fractions and physical seconds."""

    version: int = 1
    assimilation_efficiency: float = 0.84
    reserve_target: float = 0.85
    reserve_temperature: float = 0.08
    fatigue_energy_weight: float = 0.08
    gut_comfort: float = 0.55
    gut_overload_energy_weight: float = 0.08
    effort_energy_rate: float = 0.0042
    effort_extra_weight: float = 0.25
    reward_per_energy: float = 12.0
    max_interval_seconds: float = 2.0

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all() or self.version != 1:
            raise ValueError("unsupported or nonfinite homeostasis configuration")
        if not 0 < self.assimilation_efficiency <= 1:
            raise ValueError("assimilation efficiency must be in (0, 1]")
        if not 0 < self.reserve_target <= 1 or self.reserve_temperature <= 0:
            raise ValueError("reserve target/temperature are invalid")
        if not 0 <= self.gut_comfort <= 1:
            raise ValueError("gut comfort must be in [0, 1]")
        if min(
            self.fatigue_energy_weight, self.gut_overload_energy_weight,
            self.effort_energy_rate, self.effort_extra_weight, self.reward_per_energy,
        ) < 0:
            raise ValueError("homeostasis weights must be nonnegative")
        if self.max_interval_seconds <= 0:
            raise ValueError("maximum accounting interval must be positive")

    def to_value(self) -> dict[str, Any]:
        value = {"format": FORMAT, "config": asdict(self)}
        value["sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "FiniteEnergyConfig":
        if value.get("format") != FORMAT:
            raise ValueError("incompatible homeostasis configuration")
        clean = {"format": value["format"], "config": value["config"]}
        digest = hashlib.sha256(
            json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if value.get("sha256") != digest:
            raise ValueError("homeostasis configuration checksum differs")
        return cls(**value["config"])


def _physiology(value: Mapping[str, Any] | np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(value, Mapping):
        unknown = set(value) - {"energy", "gut", "fatigue"}
        if unknown or not {"energy", "gut", "fatigue"}.issubset(value):
            raise ValueError(f"{name} physiology must contain only energy, gut and fatigue")
        energy, gut, fatigue = (
            np.asarray(value[key], dtype=np.float64) for key in ("energy", "gut", "fatigue")
        )
        try:
            energy, gut, fatigue = np.broadcast_arrays(energy, gut, fatigue)
        except ValueError as exc:
            raise ValueError(f"{name} physiology fields do not broadcast") from exc
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != 3:
            raise ValueError(f"{name} physiology must end in [energy, gut, fatigue]")
        energy, gut, fatigue = (array[..., index] for index in range(3))
    if not all(np.isfinite(item).all() for item in (energy, gut, fatigue)):
        raise ValueError(f"{name} physiology is nonfinite")
    if any(np.any((item < 0) | (item > 1)) for item in (energy, gut, fatigue)):
        raise ValueError(f"{name} physiology must be in [0, 1]")
    return energy, gut, fatigue


class FiniteEnergyObjective:
    """Auditable physical reward; prediction progress remains a separate term."""

    def __init__(self, config: FiniteEnergyConfig | None = None) -> None:
        self.config = config or FiniteEnergyConfig()

    def potential(
        self, physiology: Mapping[str, Any] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return state utility components in body-energy-fraction units."""
        energy, gut, fatigue = _physiology(physiology, "state")
        c = self.config
        reserve = energy + c.assimilation_efficiency * gut
        # A smooth one-sided shortfall is monotonic: burning energy can never
        # improve this term, including when reserve is above the target.
        shortfall = c.reserve_temperature * np.logaddexp(
            0.0, (c.reserve_target - reserve) / c.reserve_temperature
        )
        fatigue_cost = c.fatigue_energy_weight * fatigue * fatigue
        gut_excess = np.maximum(gut - c.gut_comfort, 0.0)
        gut_overload_cost = c.gut_overload_energy_weight * gut_excess * gut_excess
        value = -shortfall - fatigue_cost - gut_overload_cost
        return {
            "reserve_energy": reserve,
            "reserve_shortfall_energy": shortfall,
            "fatigue_cost_energy": fatigue_cost,
            "gut_overload_cost_energy": gut_overload_cost,
            "potential_energy": value,
        }

    def transition(
        self,
        before: Mapping[str, Any] | np.ndarray,
        after: Mapping[str, Any] | np.ndarray,
        *,
        nutrition: Any,
        effort: Any,
        dt: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Score one causal physical transition, returning reward and named terms.

        ``nutrition`` is observed ingestion in the world's nutrition units and
        is logged, not paid again: its stored value already enters through gut.
        ``effort`` is the world's dimensionless [0,1] actuator effort.
        """
        if not np.isfinite(dt) or not 0 < dt <= self.config.max_interval_seconds:
            raise ValueError(
                f"dt must be finite and in (0, {self.config.max_interval_seconds}] seconds"
            )
        before_terms = self.potential(before)
        after_terms = self.potential(after)
        nutrition = np.asarray(nutrition, dtype=np.float64)
        effort = np.asarray(effort, dtype=np.float64)
        shape = before_terms["potential_energy"].shape
        try:
            nutrition = np.broadcast_to(nutrition, shape)
            effort = np.broadcast_to(effort, shape)
            after_potential = np.broadcast_to(after_terms["potential_energy"], shape)
        except ValueError as exc:
            raise ValueError("transition outcome shapes differ") from exc
        if not np.isfinite(nutrition).all() or np.any(nutrition < 0):
            raise ValueError("nutrition must be finite and nonnegative")
        if not np.isfinite(effort).all() or np.any((effort < 0) | (effort > 1)):
            raise ValueError("effort must be finite and in [0, 1]")
        c = self.config
        potential_delta = after_potential - before_terms["potential_energy"]
        effort_cost = c.effort_extra_weight * c.effort_energy_rate * effort * float(dt)
        reward = c.reward_per_energy * (potential_delta - effort_cost)
        reserve_before = before_terms["reserve_energy"]
        hunger_gate = 1.0 / (1.0 + np.exp(
            np.clip((reserve_before - c.reserve_target) / c.reserve_temperature, -60, 60)
        ))
        components = {
            "potential_delta_energy": potential_delta,
            "effort_cost_energy": effort_cost,
            "nutrition_observed": nutrition,
            "hunger_gate": hunger_gate,
            "reward": reward,
            **{f"before_{key}": value for key, value in before_terms.items()},
            **{f"after_{key}": value for key, value in after_terms.items()},
        }
        return reward.astype(np.float32), {
            key: np.asarray(value, dtype=np.float32) for key, value in components.items()
        }


def transition_reward(
    before: Mapping[str, Any] | np.ndarray,
    after: Mapping[str, Any] | np.ndarray,
    *, nutrition: Any, effort: Any, dt: float,
    config: FiniteEnergyConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stateless convenience entry point for the version-1 objective."""
    return FiniteEnergyObjective(config).transition(
        before, after, nutrition=nutrition, effort=effort, dt=dt
    )
