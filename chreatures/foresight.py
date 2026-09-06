"""Private counterfactual action-plan evidence from an inherited predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .homeostasis import FiniteEnergyConfig, FiniteEnergyObjective
from .motor_inheritance import ACTIONS
from .predictive_native import NativePredictiveCohort


FORMAT = "chreatures-private-foresight-v1"
QUERY_CONTRACT = "candidate-and-frozen-state-only-v1"
NONNEGATIVE_ACTIONS = tuple(ACTIONS.index(name) for name in (
    "grip", "signal_low", "signal_mid", "signal_high",
))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_value(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "data": array.tolist()}


def _array_from_value(value: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(item) for item in value["shape"])
        array = np.asarray(value["data"], dtype=dtype)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid encoded foresight array: {name}") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid encoded foresight array: {name}")
    return array


@dataclass(frozen=True)
class ForesightConfig:
    horizon: int = 8
    branches: int = 4
    action_noise_sigma: float = 0.18
    action_noise_ar: float = 0.72
    discount: float = 0.94
    correction_gain: float = 0.08
    max_correction: float = 0.12
    macro_dt_seconds: float = 0.25
    seed: int = 20260905

    def __post_init__(self) -> None:
        if self.horizon != 8 or self.branches != 4:
            raise ValueError("foresight v1 requires eight steps and four branches")
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("foresight config must be finite")
        if self.action_noise_sigma != 0.18 or not 0 <= self.action_noise_ar < 1:
            raise ValueError("unsupported foresight action-plan noise")
        if not 0 < self.discount <= 1 or min(self.correction_gain, self.max_correction) < 0:
            raise ValueError("invalid foresight scoring configuration")
        if self.macro_dt_seconds <= 0:
            raise ValueError("macro interval must be positive")

    def to_value(self) -> dict[str, Any]:
        value = {"format": FORMAT, "config": asdict(self)}
        value["sha256"] = hashlib.sha256(_canonical(value).encode()).hexdigest()
        return value

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "ForesightConfig":
        clean = {"format": value.get("format"), "config": value.get("config")}
        if clean["format"] != FORMAT or value.get("sha256") != hashlib.sha256(
            _canonical(clean).encode()
        ).hexdigest():
            raise ValueError("foresight config identity differs")
        return cls(**clean["config"])


class ResidentForesight:
    """One resident's experienced state and read-only imagined query cohort."""

    def __init__(
        self,
        export: str | Path,
        *,
        config: ForesightConfig | None = None,
        homeostasis: FiniteEnergyConfig | None = None,
    ) -> None:
        self.export = Path(export)
        self.config = config or ForesightConfig()
        self.homeostasis = homeostasis or FiniteEnergyConfig()
        self.objective = FiniteEnergyObjective(self.homeostasis)
        self.experienced = NativePredictiveCohort(self.export, batch_size=1)
        self.query_cohort = NativePredictiveCohort(
            self.export, batch_size=9 * self.config.branches,
        )
        if (
            self.experienced.model_identity != self.query_cohort.model_identity
            or self.experienced.input_identity != self.query_cohort.input_identity
        ):
            raise ValueError("experienced and query predictors differ")
        if tuple(self.experienced.metadata.get("actions", ())) != ACTIONS:
            raise ValueError("predictor action order differs from inherited motor actions")
        physiology = tuple(self.experienced.metadata.get("physiology", ()))
        if physiology[:3] != ("energy", "gut", "fatigue"):
            raise ValueError("predictor physiology must begin energy, gut, fatigue")
        temporal = self.experienced.metadata.get("temporal_contract")
        if not isinstance(temporal, Mapping) or (
            temporal.get("macro_steps") != 5
            or temporal.get("physics_dt_seconds") != 0.05
            or temporal.get("observation_interval_seconds")
            != self.config.macro_dt_seconds
        ):
            raise ValueError("predictor temporal contract differs from motor macros")
        self.rng = np.random.default_rng(self.config.seed)
        self.observation_count = 0
        self.last_observed_features: np.ndarray | None = None
        self.last_observed_physiology: np.ndarray | None = None
        self.last_executed_action: np.ndarray | None = None
        self.intention_tail: np.ndarray | None = None
        self._query_cache: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        declared = self.experienced.metadata.get("forecast_status")
        if isinstance(declared, str) and declared:
            return declared
        record = self.experienced.metadata.get("training_input_identity", {})
        return str(record.get("status", "unknown; research forecasts are not calibrated"))

    def observe(
        self,
        features: Any,
        physiology: Any,
        previous_action: Any,
        *,
        reset: bool = False,
    ) -> np.ndarray:
        """Advance experienced recurrent state at an actual macro boundary."""
        raw_feature = self._vector(features, self.experienced.feature_dim, "raw features")
        feature = np.asarray(
            self.experienced.normalize_source_features(raw_feature[None, :]),
            dtype=np.float32,
        )
        if feature.shape != (1, self.experienced.feature_dim) or not np.isfinite(feature).all():
            raise ValueError("predictor source normalizer returned invalid features")
        feature = feature[0]
        physical = self._vector(physiology, self.experienced.physiology_dim, "physiology")
        action = self._action(previous_action, "previous action")
        if np.any((physical[:3] < 0) | (physical[:3] > 1)):
            raise ValueError("observed energy, gut and fatigue must be in [0, 1]")
        latent = self.experienced.observe(
            feature[None, :], physical[None, :], action[None, :],
            np.asarray([bool(reset)], dtype=np.bool_),
        )
        self.last_observed_features = feature.copy()
        self.last_observed_physiology = physical.copy()
        self.last_executed_action = action.copy()
        self.observation_count += 1
        self._query_cache = None
        if reset:
            self.intention_tail = None
        return np.asarray(latent)[0].copy()

    def candidate_evidence(
        self, candidates: tuple[tuple[float, ...], ...],
    ) -> dict[str, Any]:
        """Return frozen candidate corrections for ContextualMotorRefiner."""
        actions = np.asarray(candidates, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(ACTIONS) or not 1 <= len(actions) <= 9:
            raise ValueError("foresight candidates must have shape 1..9 by 8")
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError("foresight candidates must be finite in [-1, 1]")
        if np.any(actions[:, NONNEGATIVE_ACTIONS] < 0):
            raise ValueError("grip and signal candidates must be nonnegative")
        if self.last_observed_physiology is None:
            raise RuntimeError("foresight requires an actual observation before query")
        key = hashlib.sha256(actions.tobytes(order="C")).hexdigest()
        if self._query_cache is not None and self._query_cache["key"] == key:
            return copy.deepcopy(self._query_cache["report"])

        plans = self._plans(actions)
        count = len(actions)
        flat = plans.transpose(0, 1, 2, 3).reshape(
            self.config.horizon, count * self.config.branches, len(ACTIONS),
        )
        frozen = self.experienced.snapshot()
        prediction = self.query_cohort.query_from_snapshot(frozen, flat)
        scores, diagnostics, best = self._score(prediction, count)
        relative = scores - scores[0] if np.isfinite(scores[0]) else np.zeros_like(scores)
        corrections = np.clip(
            self.config.correction_gain * relative,
            -self.config.max_correction, self.config.max_correction,
        ).astype(np.float32)
        corrections[~np.isfinite(scores)] = 0.0
        report = {
            "source": "private-native-predictive-foresight-v1",
            "corrections": corrections.astype(float).tolist(),
            "diagnostics": diagnostics,
            "proposal_credit_contract": QUERY_CONTRACT,
        }
        best_tails = np.stack([
            plans[1:, index, best[index], :] if best[index] >= 0
            else np.zeros((self.config.horizon - 1, len(ACTIONS)), dtype=np.float32)
            for index in range(count)
        ])
        self._query_cache = {
            "key": key, "candidates": actions.copy(), "plans": plans.copy(),
            "best_branches": best.copy(), "best_tails": best_tails,
            "report": copy.deepcopy(report),
        }
        return report

    def commit_executed(self, action: Any) -> None:
        """Retain a chosen candidate's best future tail, or clear intention."""
        selected = self._action(action, "executed action")
        cache = self._query_cache
        self.intention_tail = None
        if cache is not None:
            matches = np.flatnonzero(np.all(cache["candidates"] == selected[None, :], axis=1))
            if len(matches) == 1 and cache["best_branches"][int(matches[0])] >= 0:
                self.intention_tail = cache["best_tails"][int(matches[0])].copy()
        self._query_cache = None

    def snapshot(self) -> dict[str, Any]:
        cache = None
        if self._query_cache is not None:
            cache = {
                name: (_array_value(value) if isinstance(value, np.ndarray) else copy.deepcopy(value))
                for name, value in self._query_cache.items()
            }
        native = self.experienced.snapshot()
        return {
            "format": FORMAT, "version": 1,
            "config": self.config.to_value(), "homeostasis": self.homeostasis.to_value(),
            "model_identity": copy.deepcopy(self.experienced.model_identity),
            "input_identity": copy.deepcopy(self.experienced.input_identity),
            "experienced": {
                name: _array_value(value) if isinstance(value, np.ndarray) else copy.deepcopy(value)
                for name, value in native.items()
            },
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "observation_count": self.observation_count,
            "last_observed_features": self._optional_array(self.last_observed_features),
            "last_observed_physiology": self._optional_array(self.last_observed_physiology),
            "last_executed_action": self._optional_array(self.last_executed_action),
            "intention_tail": self._optional_array(self.intention_tail),
            "query_cache": cache,
        }

    @classmethod
    def restore(cls, export: str | Path, value: Mapping[str, Any]) -> "ResidentForesight":
        expected = {
            "format", "version", "config", "homeostasis", "model_identity",
            "input_identity", "experienced", "rng", "observation_count",
            "last_observed_features", "last_observed_physiology",
            "last_executed_action", "intention_tail", "query_cache",
        }
        if set(value) != expected:
            raise ValueError("foresight snapshot fields differ")
        if value.get("format") != FORMAT or value.get("version") != 1:
            raise ValueError("unsupported foresight snapshot")
        result = cls(
            export, config=ForesightConfig.from_value(value["config"]),
            homeostasis=FiniteEnergyConfig.from_value(value["homeostasis"]),
        )
        if value["model_identity"] != result.experienced.model_identity:
            raise ValueError("foresight model identity differs")
        if value["input_identity"] != result.experienced.input_identity:
            raise ValueError("foresight input identity differs")
        native = {
            name: _array_from_value(item, f"experienced.{name}")
            if isinstance(item, Mapping) and set(item) == {"dtype", "shape", "data"} else copy.deepcopy(item)
            for name, item in value["experienced"].items()
        }
        result.experienced.restore(native)
        result.rng.bit_generator.state = copy.deepcopy(value["rng"])
        result.observation_count = int(value["observation_count"])
        if result.observation_count < 0:
            raise ValueError("foresight observation count is invalid")
        for name in ("last_observed_features", "last_observed_physiology", "last_executed_action", "intention_tail"):
            setattr(result, name, result._decode_optional(value[name], name))
        expected_shapes = {
            "last_observed_features": (result.experienced.feature_dim,),
            "last_observed_physiology": (result.experienced.physiology_dim,),
            "last_executed_action": (len(ACTIONS),),
            "intention_tail": (result.config.horizon - 1, len(ACTIONS)),
        }
        for name, shape in expected_shapes.items():
            item = getattr(result, name)
            if item is not None and item.shape != shape:
                raise ValueError(f"restored {name} shape differs")
        if value["query_cache"] is not None:
            result._query_cache = {
                name: _array_from_value(item, f"query_cache.{name}")
                if isinstance(item, Mapping) and set(item) == {"dtype", "shape", "data"} else copy.deepcopy(item)
                for name, item in value["query_cache"].items()
            }
        return result

    def _plans(self, candidates: np.ndarray) -> np.ndarray:
        count = len(candidates)
        plans = np.empty(
            (self.config.horizon, count, self.config.branches, len(ACTIONS)), dtype=np.float32,
        )
        plans[0] = candidates[:, None, :]
        for candidate in range(count):
            for branch in range(self.config.branches):
                if branch == 0 and self.intention_tail is not None:
                    plans[1:, candidate, branch] = self.intention_tail
                    continue
                latent = np.arctanh(np.clip(candidates[candidate], -0.999, 0.999)).astype(np.float64)
                noise = np.zeros(len(ACTIONS), dtype=np.float64)
                innovation = math.sqrt(1.0 - self.config.action_noise_ar**2)
                for step in range(1, self.config.horizon):
                    noise = self.config.action_noise_ar * noise + innovation * self.rng.normal(
                        0.0, self.config.action_noise_sigma, len(ACTIONS),
                    )
                    latent = latent + noise
                    action = np.tanh(latent).astype(np.float32)
                    action[list(NONNEGATIVE_ACTIONS)] = np.maximum(
                        action[list(NONNEGATIVE_ACTIONS)], 0.0,
                    )
                    plans[step, candidate, branch] = action
        return plans

    def _score(
        self, prediction: Mapping[str, Any], candidate_count: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
        required = {
            "feature_mean", "feature_residual_scale", "physiology_mean",
            "physiology_residual_scale", "valid", "horizon_support",
        }
        if set(prediction) != required:
            raise ValueError("native foresight prediction fields differ")
        shape = (self.config.horizon, candidate_count * self.config.branches)
        physiology = np.asarray(prediction["physiology_mean"], dtype=np.float32)
        scale = np.asarray(prediction["physiology_residual_scale"], dtype=np.float32)
        valid = np.asarray(prediction["valid"], dtype=np.bool_)
        support = np.asarray(prediction["horizon_support"], dtype=np.float32)
        if physiology.shape != shape + (self.experienced.physiology_dim,) or scale.shape != physiology.shape:
            raise ValueError("native foresight physiology shape differs")
        if valid.shape != shape or support.shape != shape:
            raise ValueError("native foresight validity shape differs")
        physical = physiology[..., :3]
        branch_valid = valid.all(axis=0) & np.isfinite(physical).all(axis=(0, 2))
        branch_valid &= np.all((physical >= 0) & (physical <= 1), axis=(0, 2))
        branch_valid &= np.isfinite(scale[..., :3]).all(axis=(0, 2)) & np.all(scale[..., :3] >= 0, axis=(0, 2))
        branch_scores = np.full(shape[1], np.nan, dtype=np.float64)
        initial = self.objective.potential(self.last_observed_physiology[:3])["potential_energy"]
        potential = self.objective.potential(physical)["potential_energy"]
        for branch in np.flatnonzero(branch_valid):
            previous = float(initial)
            total = 0.0
            for step in range(self.config.horizon):
                current = float(potential[step, branch])
                total += self.config.discount**step * (current - previous)
                previous = current
            branch_scores[branch] = self.homeostasis.reward_per_energy * total
        matrix = branch_scores.reshape(candidate_count, self.config.branches)
        scores = np.asarray([
            np.mean(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in matrix
        ])
        best = np.asarray([
            int(np.nanargmax(row)) if np.isfinite(row).any() else -1 for row in matrix
        ], dtype=np.int64)
        diagnostics = []
        phys_scale = scale[..., :3].reshape(
            self.config.horizon, candidate_count, self.config.branches, 3,
        )
        support_view = support.reshape(self.config.horizon, candidate_count, self.config.branches)
        for index in range(candidate_count):
            valid_count = int(np.isfinite(matrix[index]).sum())
            diagnostics.append({
                "forecast_score": None if not np.isfinite(scores[index]) else float(scores[index]),
                "valid_branches": valid_count,
                "branches": self.config.branches,
                "best_branch": None if best[index] < 0 else int(best[index]),
                "physiology_residual_scale_mean": phys_scale[:, index].mean(axis=(0, 1)).astype(float).tolist(),
                "horizon_support_mean": float(support_view[:, index].mean()),
                "residual_scale_status": "learned conditional residual scale; heuristic, not calibrated confidence",
                "forecast_status": self.status,
                "model_artifact_sha256": self.experienced.model_identity["artifact_sha256"],
                "input_identity_sha256": self.experienced.input_identity["sha256"],
            })
        return scores, diagnostics, best

    @staticmethod
    def _vector(value: Any, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {size} finite values")
        return array

    @classmethod
    def _action(cls, value: Any, name: str) -> np.ndarray:
        action = cls._vector(value, len(ACTIONS), name)
        if np.any(np.abs(action) > 1) or np.any(action[list(NONNEGATIVE_ACTIONS)] < 0):
            raise ValueError(f"{name} is outside the physical action schema")
        return action

    @staticmethod
    def _optional_array(value: np.ndarray | None) -> dict[str, Any] | None:
        return None if value is None else _array_value(value)

    @staticmethod
    def _decode_optional(value: Any, name: str) -> np.ndarray | None:
        return None if value is None else _array_from_value(value, name)


__all__ = ["ForesightConfig", "ResidentForesight"]
