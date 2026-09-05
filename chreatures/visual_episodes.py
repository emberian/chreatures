"""Slow native-vision episodes for private, delayed sensory experience.

This NumPy organ binds pairs of completed 960-value body-view features around
an exact short motor interval. It accepts no world coordinates, identities,
object labels, scene graph, or archive handle. A received feature is always a
delayed capture-time observation, never a claim about the current visual field.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .motor_inheritance import ACTIONS as MOTOR_ACTIONS8


FEATURE_DIMENSION = 960
DEFAULT_OUTCOMES = ("nutrition", "contact", "distance", "effort")
DEFAULT_PHYSIOLOGY = (
    "energy", "gut", "fatigue", "speed", "angular_velocity", "support"
)
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tick(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _named_vector(value: Any, names: tuple[str, ...], name: str) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{name} fields must exactly match {list(names)}")
    result = np.asarray([value[item] for item in names], dtype=np.float64)
    if result.shape != (len(names),) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {len(names)} finite values")
    return result


def feature_sha256(feature: Any) -> str:
    value = _vector(feature, FEATURE_DIMENSION, "native feature").astype(
        "<f4", copy=False
    )
    return hashlib.sha256(value.tobytes()).hexdigest()


@dataclass(frozen=True)
class VisualEpisodeConfig:
    capacity: int = 64
    interval_ticks: int = 5
    interval_seconds: float = 0.25
    interval_tolerance_seconds: float = 1e-6
    stable_action_span: float = 1e-5
    maximum_action_mae: float = 0.22
    maximum_action_error: float = 0.55
    minimum_visual_cosine: float = 0.60
    visual_temperature: float = 0.14
    action_temperature: float = 0.12
    age_half_life_ticks: int = 200
    maximum_controller_influence: float = 0.08
    neighbors: int = 5

    def validate(self) -> None:
        if not 1 <= self.capacity <= 4096:
            raise ValueError("capacity must be in 1..4096")
        if not 1 <= self.interval_ticks <= 256:
            raise ValueError("interval_ticks must be in 1..256")
        if self.interval_seconds <= 0 or self.interval_tolerance_seconds < 0:
            raise ValueError("interval seconds must be positive with nonnegative tolerance")
        if min(
            self.stable_action_span,
            self.maximum_action_mae,
            self.maximum_action_error,
            self.visual_temperature,
            self.action_temperature,
        ) <= 0:
            raise ValueError("distance and action thresholds must be positive")
        if not -1 <= self.minimum_visual_cosine <= 1:
            raise ValueError("minimum_visual_cosine must be in [-1, 1]")
        if self.age_half_life_ticks < 1:
            raise ValueError("age_half_life_ticks must be positive")
        if not 0 <= self.maximum_controller_influence <= 0.25:
            raise ValueError("maximum_controller_influence must be in [0, .25]")
        if not 1 <= self.neighbors <= 32:
            raise ValueError("neighbors must be in 1..32")


class VisualEpisodeMemory:
    """Bounded personal episodes over exact five-command motor intervals."""

    FORMAT = "chreatures-visual-episode-memory-v2"
    LEGACY_FORMAT = "chreatures-visual-episode-memory-v1"
    VERSION = 2
    OUTCOME_AGGREGATION = "physics-outcome-v2-effort-rate-integrated"

    def __init__(
        self,
        *,
        config: VisualEpisodeConfig | None = None,
        outcome_names: Iterable[str] = DEFAULT_OUTCOMES,
        physiology_names: Iterable[str] = DEFAULT_PHYSIOLOGY,
    ):
        self.config = config or VisualEpisodeConfig()
        self.config.validate()
        self.outcome_names = self._names(outcome_names, "outcome_names")
        self.physiology_names = self._names(physiology_names, "physiology_names")
        if not {"nutrition", "contact", "effort"} <= set(self.outcome_names):
            raise ValueError("outcomes require nutrition, contact and effort")
        if not {"energy", "gut", "fatigue"} <= set(self.physiology_names):
            raise ValueError("physiology requires energy, gut and fatigue")
        self._records: list[dict[str, Any]] = []
        self.clock = 0
        self.model_revision: str | None = None
        self.pooling_version: str | None = None
        self.history_migration: dict[str, Any] | None = None

    @property
    def has_native_history(self) -> bool:
        """Whether at least one completed native interval has been bound."""

        return bool(self._records)

    @property
    def record_count(self) -> int:
        return len(self._records)

    @staticmethod
    def _names(value: Iterable[str], name: str) -> tuple[str, ...]:
        result = tuple(value)
        if not result or len(set(result)) != len(result):
            raise ValueError(f"{name} must be nonempty and unique")
        if any(not isinstance(item, str) or not item.isidentifier() for item in result):
            raise ValueError(f"invalid {name}")
        return result

    def _capture(self, value: Any, name: str) -> dict[str, Any]:
        required = {
            "feature", "feature_sha256", "frame_sha256", "response_sha256",
            "capture_tick", "delivery_tick", "model_time", "model_revision",
            "pooling_version",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"{name} fields differ from completed-capture contract")
        feature = _vector(value["feature"], FEATURE_DIMENSION, f"{name}.feature")
        expected = feature_sha256(feature)
        if _digest(value["feature_sha256"], f"{name}.feature_sha256") != expected:
            raise ValueError(f"{name} feature hash differs")
        capture_tick = _tick(value["capture_tick"], f"{name}.capture_tick")
        delivery_tick = _tick(value["delivery_tick"], f"{name}.delivery_tick")
        if delivery_tick <= capture_tick:
            raise ValueError(f"{name} must be delivered after its capture tick")
        revision = value["model_revision"]
        pooling = value["pooling_version"]
        if not isinstance(revision, str) or not revision or len(revision) > 128:
            raise ValueError(f"invalid {name} model revision")
        if not isinstance(pooling, str) or not pooling or len(pooling) > 128:
            raise ValueError(f"invalid {name} pooling version")
        return {
            "feature": feature.tolist(),
            "feature_sha256": expected,
            "frame_sha256": _digest(value["frame_sha256"], f"{name}.frame_sha256"),
            "response_sha256": _digest(
                value["response_sha256"], f"{name}.response_sha256"
            ),
            "capture_tick": capture_tick,
            "delivery_tick": delivery_tick,
            "model_time": _finite(value["model_time"], f"{name}.model_time"),
            "model_revision": revision,
            "pooling_version": pooling,
        }

    def _steps(
        self, value: Any, start: dict[str, Any], end: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) != self.config.interval_ticks:
            raise ValueError(
                f"steps must retain exactly {self.config.interval_ticks} transitions"
            )
        result = []
        previous_after = None
        expected_from = start["capture_tick"]
        total_dt = 0.0
        required = {
            "from_tick", "to_tick", "dt", "action", "outcome",
            "physiology_before", "physiology_after",
        }
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping) or set(raw) != required:
                raise ValueError(f"step {index} fields differ from interval contract")
            from_tick = _tick(raw["from_tick"], f"step {index}.from_tick")
            to_tick = _tick(raw["to_tick"], f"step {index}.to_tick")
            if from_tick != expected_from or to_tick != from_tick + 1:
                raise ValueError("steps must be contiguous one-tick transitions")
            action = _named_vector(raw["action"], MOTOR_ACTIONS8, f"step {index}.action")
            if np.any(action < -1) or np.any(action > 1):
                raise ValueError("motor actions must be in [-1, 1]")
            for channel in ("grip", "signal_low", "signal_mid", "signal_high"):
                if action[MOTOR_ACTIONS8.index(channel)] < 0:
                    raise ValueError(f"{channel} must be nonnegative")
            outcome = _named_vector(
                raw["outcome"], self.outcome_names, f"step {index}.outcome"
            )
            before = _named_vector(
                raw["physiology_before"], self.physiology_names,
                f"step {index}.physiology_before",
            )
            after = _named_vector(
                raw["physiology_after"], self.physiology_names,
                f"step {index}.physiology_after",
            )
            for bounded in ("energy", "gut", "fatigue", "support"):
                if bounded in self.physiology_names:
                    position = self.physiology_names.index(bounded)
                    if not (0 <= before[position] <= 1 and 0 <= after[position] <= 1):
                        raise ValueError(f"step {index} {bounded} must be in [0, 1]")
            for nonnegative in ("nutrition", "distance", "effort"):
                if nonnegative in self.outcome_names:
                    position = self.outcome_names.index(nonnegative)
                    if outcome[position] < 0:
                        raise ValueError(f"step {index} {nonnegative} must be nonnegative")
            if "contact" in self.outcome_names:
                position = self.outcome_names.index("contact")
                if not 0 <= outcome[position] <= 1:
                    raise ValueError(f"step {index} contact must be in [0, 1]")
            if previous_after is not None and not np.allclose(
                before, previous_after, rtol=0.0, atol=1e-6
            ):
                raise ValueError("physiology is discontinuous between retained steps")
            dt = _finite(raw["dt"], f"step {index}.dt")
            if dt <= 0:
                raise ValueError("step dt must be positive")
            result.append(
                {
                    "from_tick": from_tick,
                    "to_tick": to_tick,
                    "dt": dt,
                    "action": {
                        field: float(item)
                        for field, item in zip(MOTOR_ACTIONS8, action)
                    },
                    "outcome": {
                        field: float(item)
                        for field, item in zip(self.outcome_names, outcome)
                    },
                    "physiology_before": {
                        field: float(item)
                        for field, item in zip(self.physiology_names, before)
                    },
                    "physiology_after": {
                        field: float(item)
                        for field, item in zip(self.physiology_names, after)
                    },
                }
            )
            previous_after = after
            expected_from = to_tick
            total_dt += dt
        if expected_from != end["capture_tick"]:
            raise ValueError("retained steps do not end at the second capture")
        model_duration = end["model_time"] - start["model_time"]
        tolerance = self.config.interval_tolerance_seconds
        if (
            not math.isclose(total_dt, model_duration, rel_tol=0.0, abs_tol=tolerance)
            or not math.isclose(
                model_duration, self.config.interval_seconds,
                rel_tol=0.0, abs_tol=tolerance,
            )
        ):
            raise ValueError("capture and step durations differ from configured horizon")
        return result

    def _make_record(
        self, start_capture: Any, end_capture: Any, steps: Any
    ) -> dict[str, Any]:
        start = self._capture(start_capture, "start_capture")
        end = self._capture(end_capture, "end_capture")
        if end["capture_tick"] - start["capture_tick"] != self.config.interval_ticks:
            raise ValueError("capture tick horizon differs from configured interval")
        if end["delivery_tick"] != start["delivery_tick"]:
            raise ValueError("paired features must come from one fixed delivery cohort")
        if end["response_sha256"] != start["response_sha256"]:
            raise ValueError("paired features must share one completed response")
        if (
            end["model_revision"] != start["model_revision"]
            or end["pooling_version"] != start["pooling_version"]
        ):
            raise ValueError("paired features use incompatible native encoders")
        retained = self._steps(steps, start, end)
        actions = np.asarray(
            [[step["action"][name] for name in MOTOR_ACTIONS8] for step in retained],
            dtype=np.float64,
        )
        outcomes = np.asarray(
            [[step["outcome"][name] for name in self.outcome_names] for step in retained],
            dtype=np.float64,
        )
        step_dt = np.asarray([step["dt"] for step in retained], dtype=np.float64)
        physiology_start = np.asarray(
            [retained[0]["physiology_before"][name] for name in self.physiology_names],
            dtype=np.float64,
        )
        physiology_end = np.asarray(
            [retained[-1]["physiology_after"][name] for name in self.physiology_names],
            dtype=np.float64,
        )
        action_span = np.ptp(actions, axis=0)
        summary = {
            "duration_ticks": self.config.interval_ticks,
            "duration_model_seconds": float(sum(step["dt"] for step in retained)),
            "action_mean": actions.mean(axis=0).tolist(),
            "action_std": actions.std(axis=0).tolist(),
            "action_minimum": actions.min(axis=0).tolist(),
            "action_maximum": actions.max(axis=0).tolist(),
            "action_mean_absolute": np.abs(actions).mean(axis=0).tolist(),
            "action_total_variation": np.abs(np.diff(actions, axis=0)).sum(axis=0).tolist(),
            "maximum_action_span": float(action_span.max()),
            "stable_action": bool(action_span.max() <= self.config.stable_action_span),
            "outcome_sum": outcomes.sum(axis=0).tolist(),
            "outcome_mean": outcomes.mean(axis=0).tolist(),
            "outcome_integral": np.sum(
                outcomes.astype(np.float64) * step_dt[:, None], axis=0
            ).tolist(),
            "outcome_maximum": outcomes.max(axis=0).tolist(),
            "physiology_delta": (physiology_end - physiology_start).tolist(),
        }
        identity = {
            "start_feature_sha256": start["feature_sha256"],
            "end_feature_sha256": end["feature_sha256"],
            "start_tick": start["capture_tick"],
            "end_tick": end["capture_tick"],
            "steps": retained,
        }
        return {
            "schema_version": self.VERSION,
            "outcome_aggregation": self.OUTCOME_AGGREGATION,
            "interval_sha256": hashlib.sha256(_canonical(identity)).hexdigest(),
            "start_capture": start,
            "end_capture": end,
            "steps": retained,
            "summary": summary,
        }

    def bind_interval(
        self, start_capture: Any, end_capture: Any, steps: Any
    ) -> dict[str, Any]:
        """Bind one completed pair and its exact intervening five transitions."""

        record = self._make_record(start_capture, end_capture, steps)
        if any(
            existing["interval_sha256"] == record["interval_sha256"]
            for existing in self._records
        ):
            raise ValueError("visual interval is already bound")
        revision = record["start_capture"]["model_revision"]
        pooling = record["start_capture"]["pooling_version"]
        if self.model_revision not in (None, revision) or self.pooling_version not in (
            None, pooling
        ):
            raise ValueError("native encoder differs from this personal memory")
        self.model_revision = revision
        self.pooling_version = pooling
        self.clock += 1
        record["sequence"] = self.clock
        self._records.append(record)
        if len(self._records) > self.config.capacity:
            del self._records[0]
        return {
            "sequence": record["sequence"],
            "interval_sha256": record["interval_sha256"],
            "start_feature_sha256": record["start_capture"]["feature_sha256"],
            "end_feature_sha256": record["end_capture"]["feature_sha256"],
            "delivery_tick": record["end_capture"]["delivery_tick"],
            "duration_ticks": record["summary"]["duration_ticks"],
            "duration_model_seconds": record["summary"]["duration_model_seconds"],
            "stable_action": record["summary"]["stable_action"],
            "record_count": len(self._records),
        }

    @staticmethod
    def _action(value: Any) -> np.ndarray:
        action = _named_vector(value, MOTOR_ACTIONS8, "candidate action")
        if np.any(action < -1) or np.any(action > 1):
            raise ValueError("candidate action must be in [-1, 1]")
        for channel in ("grip", "signal_low", "signal_mid", "signal_high"):
            if action[MOTOR_ACTIONS8.index(channel)] < 0:
                raise ValueError(f"candidate {channel} must be nonnegative")
        return action

    def recall(
        self,
        delayed_capture: Any,
        candidate_actions: Iterable[Mapping[str, Any]],
        *,
        current_tick: int,
    ) -> dict[str, Any]:
        """Retrieve empirical interval consequences for continuous actions."""

        query = self._capture(delayed_capture, "delayed_capture")
        now = _tick(current_tick, "current_tick")
        if now < query["delivery_tick"]:
            raise ValueError("native capture has not reached its fixed delivery tick")
        if self.model_revision not in (None, query["model_revision"]) or self.pooling_version not in (
            None, query["pooling_version"]
        ):
            raise ValueError("query native encoder differs from personal memory")
        actions = [self._action(value) for value in candidate_actions]
        if not 1 <= len(actions) <= 32:
            raise ValueError("candidate_actions must contain 1..32 actions")
        query_feature = np.asarray(query["feature"], dtype=np.float32)
        query_norm = max(float(np.linalg.norm(query_feature)), 1e-8)
        query_age = now - query["capture_tick"]
        query_age_factor = 2.0 ** (-query_age / self.config.age_half_life_ticks)
        candidates = []
        for candidate_index, action in enumerate(actions):
            evidence = []
            for record in self._records:
                start = np.asarray(record["start_capture"]["feature"], dtype=np.float32)
                cosine = float(
                    np.dot(query_feature, start)
                    / (query_norm * max(float(np.linalg.norm(start)), 1e-8))
                )
                native_rms = float(np.sqrt(np.mean((query_feature - start) ** 2)))
                recorded_action = np.asarray(
                    record["summary"]["action_mean"], dtype=np.float32
                )
                absolute = np.abs(action - recorded_action)
                action_mae = float(absolute.mean())
                action_max = float(absolute.max())
                evidence_age = max(0, now - record["end_capture"]["capture_tick"])
                rejection = []
                if now < record["end_capture"]["delivery_tick"]:
                    rejection.append("record has not reached its fixed delivery tick")
                if not record["summary"]["stable_action"]:
                    rejection.append("recorded action changed within interval")
                if cosine < self.config.minimum_visual_cosine:
                    rejection.append("native view is outside visual support")
                if action_mae > self.config.maximum_action_mae:
                    rejection.append("mean absolute action error is too large")
                if action_max > self.config.maximum_action_error:
                    rejection.append("one or more action channels are too far")
                visual_strength = math.exp(
                    -(1.0 - float(np.clip(cosine, -1.0, 1.0)))
                    / self.config.visual_temperature
                )
                action_strength = math.exp(
                    -action_mae / self.config.action_temperature
                )
                age_factor = 2.0 ** (
                    -evidence_age / self.config.age_half_life_ticks
                )
                support = visual_strength * action_strength * age_factor
                evidence.append(
                    {
                        "sequence": record["sequence"],
                        "interval_sha256": record["interval_sha256"],
                        "cosine_similarity": cosine,
                        "cosine_distance": 1.0 - cosine,
                        "native_rms_distance": native_rms,
                        "action_mean_absolute_error": action_mae,
                        "action_maximum_absolute_error": action_max,
                        "absolute_action_error": {
                            name: float(value)
                            for name, value in zip(MOTOR_ACTIONS8, absolute)
                        },
                        "recorded_action_mean": {
                            name: float(value)
                            for name, value in zip(MOTOR_ACTIONS8, recorded_action)
                        },
                        "recorded_action_std": record["summary"]["action_std"],
                        "recorded_action_span_max": record["summary"][
                            "maximum_action_span"
                        ],
                        "capture_age_ticks": evidence_age,
                        "age_factor": age_factor,
                        "support": support,
                        "comparable": not rejection,
                        "rejection_reasons": rejection,
                        "record": record,
                    }
                )
            evidence.sort(key=lambda item: (-item["support"], item["sequence"]))
            eligible = [item for item in evidence if item["comparable"]][
                : self.config.neighbors
            ]
            if eligible:
                strengths = np.asarray(
                    [item["support"] for item in eligible], dtype=np.float64
                )
                weights = strengths / max(float(strengths.sum()), 1e-12)
                outcome = sum(
                    weight * np.asarray(
                        item["record"]["summary"]["outcome_sum"], dtype=np.float64
                    )
                    for weight, item in zip(weights, eligible)
                )
                outcome_integral = sum(
                    weight * np.asarray(
                        item["record"]["summary"]["outcome_integral"],
                        dtype=np.float64,
                    )
                    for weight, item in zip(weights, eligible)
                )
                outcome_maximum = sum(
                    weight * np.asarray(
                        item["record"]["summary"]["outcome_maximum"],
                        dtype=np.float64,
                    )
                    for weight, item in zip(weights, eligible)
                )
                physiology = sum(
                    weight * np.asarray(
                        item["record"]["summary"]["physiology_delta"],
                        dtype=np.float64,
                    )
                    for weight, item in zip(weights, eligible)
                )
                outcome_rows = np.asarray(
                    [item["record"]["summary"]["outcome_sum"] for item in eligible],
                    dtype=np.float64,
                )
                physiology_rows = np.asarray(
                    [item["record"]["summary"]["physiology_delta"] for item in eligible],
                    dtype=np.float64,
                )
                outcome_dispersion = np.sqrt(
                    np.sum(weights[:, None] * (outcome_rows - outcome) ** 2, axis=0)
                )
                physiology_dispersion = np.sqrt(
                    np.sum(
                        weights[:, None] * (physiology_rows - physiology) ** 2,
                        axis=0,
                    )
                )
                effective = float(1.0 / np.sum(weights * weights))
                support_fraction = min(1.0, float(strengths.sum()))
                sample_fraction = min(1.0, len(eligible) / 3.0)
                influence = (
                    self.config.maximum_controller_influence
                    * query_age_factor
                    * support_fraction
                    * sample_fraction
                )
                prediction = {
                    "outcome_sum": {
                        name: float(value)
                        for name, value in zip(self.outcome_names, outcome)
                    },
                    "outcome_integral": {
                        name: float(value)
                        for name, value in zip(self.outcome_names, outcome_integral)
                    },
                    "outcome_maximum": {
                        name: float(value)
                        for name, value in zip(self.outcome_names, outcome_maximum)
                    },
                    "contextual_physical_outcome": {
                        "nutrition": float(
                            outcome[self.outcome_names.index("nutrition")]
                        ),
                        "effort": float(
                            outcome_integral[self.outcome_names.index("effort")]
                        ),
                        "contact": float(
                            outcome_maximum[self.outcome_names.index("contact")]
                        ),
                    },
                    "physiology_delta": {
                        name: float(value)
                        for name, value in zip(self.physiology_names, physiology)
                    },
                    "empirical_outcome_dispersion": {
                        name: float(value)
                        for name, value in zip(self.outcome_names, outcome_dispersion)
                    },
                    "empirical_physiology_dispersion": {
                        name: float(value)
                        for name, value in zip(
                            self.physiology_names, physiology_dispersion
                        )
                    },
                    "support_count": len(eligible),
                    "effective_support": effective,
                    "basis": "empirical comparable 0.25-second native-view intervals",
                    "uncertainty_note": (
                        "dispersion is empirical neighbor spread; one episode does not "
                        "establish low predictive uncertainty"
                    ),
                }
                reason = None
            else:
                prediction = None
                influence = 0.0
                reason = "no interval has defensible visual, horizon, and action support"
            public_evidence = []
            for item in evidence[: self.config.neighbors]:
                public_evidence.append(
                    {key: copy.deepcopy(value) for key, value in item.items() if key != "record"}
                )
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "action": {
                        name: float(value)
                        for name, value in zip(MOTOR_ACTIONS8, action)
                    },
                    "prediction": prediction,
                    "no_prediction_reason": reason,
                    "controller_influence_bound": float(influence),
                    "evidence": public_evidence,
                }
            )
        return {
            "sensory_status": "delayed_native_observation",
            "usable_as_current_perception": False,
            "capture_tick": query["capture_tick"],
            "delivery_tick": query["delivery_tick"],
            "current_tick": now,
            "capture_age_ticks": query_age,
            "feature_sha256": query["feature_sha256"],
            "frame_sha256": query["frame_sha256"],
            "response_sha256": query["response_sha256"],
            "model_revision": query["model_revision"],
            "pooling_version": query["pooling_version"],
            "age_influence_ceiling": float(
                self.config.maximum_controller_influence * query_age_factor
            ),
            "action_names": list(MOTOR_ACTIONS8),
            "interval_horizon": {
                "ticks": self.config.interval_ticks,
                "model_seconds": self.config.interval_seconds,
            },
            "candidates": candidates,
        }

    def contextual_candidate_evidence(
        self,
        delayed_capture: Any,
        *,
        current_tick: int,
        current_physiology: Mapping[str, Any],
        utility_config: Any,
    ) -> Callable[[tuple[tuple[float, ...], ...]], dict[str, Any]]:
        """Build the callback accepted by ``ContextualMotorRefiner.refine``.

        Utility uses that refiner's exact energy/gut/fatigue drive and
        nutrition/effort weights.  The returned correction remains bounded by
        the visual episode's age, visual support, action support, and sample
        support.  Creating a callback without native history is rejected so an
        absent organ cannot silently become neutral evidence.
        """

        if not self._records:
            raise ValueError("candidate evidence requires bound native visual history")
        frozen_capture = self._capture(delayed_capture, "delayed_capture")
        frozen_tick = _tick(current_tick, "current_tick")
        if frozen_tick < frozen_capture["delivery_tick"]:
            raise ValueError("native capture has not reached its fixed delivery tick")
        if not isinstance(current_physiology, Mapping):
            raise ValueError("current_physiology must be a mapping")
        current = {
            name: _finite(current_physiology.get(name), f"current_physiology.{name}")
            for name in ("energy", "gut", "fatigue")
        }
        for name in ("energy", "gut", "fatigue"):
            if name not in current or not 0 <= current[name] <= 1:
                raise ValueError("current physiology requires energy/gut/fatigue in [0, 1]")

        def parameter(name: str) -> float:
            raw = (
                utility_config.get(name)
                if isinstance(utility_config, Mapping)
                else getattr(utility_config, name, None)
            )
            return _finite(raw, f"utility_config.{name}")

        weights = {
            name: parameter(name)
            for name in (
                "energy_target", "gut_target", "energy_drive_weight",
                "gut_drive_weight", "fatigue_drive_weight", "nutrition_weight",
                "effort_weight", "correction_gain",
            )
        }
        if (
            not 0 <= weights["energy_target"] <= 1
            or not 0 <= weights["gut_target"] <= 1
            or min(
                weights["energy_drive_weight"], weights["gut_drive_weight"],
                weights["fatigue_drive_weight"], weights["nutrition_weight"],
                weights["effort_weight"], weights["correction_gain"],
            ) < 0
        ):
            raise ValueError("utility configuration has invalid targets or weights")

        def drive(body: Mapping[str, float]) -> float:
            return float(
                weights["energy_drive_weight"]
                * (weights["energy_target"] - body["energy"]) ** 2
                + weights["gut_drive_weight"]
                * (weights["gut_target"] - body["gut"]) ** 2
                + weights["fatigue_drive_weight"] * body["fatigue"] ** 2
            )

        def callback(actions: tuple[tuple[float, ...], ...]) -> dict[str, Any]:
            if not isinstance(actions, tuple):
                raise ValueError("candidate actions must be an immutable tuple")
            named_actions = [
                {name: value for name, value in zip(MOTOR_ACTIONS8, action)}
                for action in actions
            ]
            recalled = self.recall(
                frozen_capture, named_actions, current_tick=frozen_tick
            )
            corrections, diagnostics = [], []
            for candidate in recalled["candidates"]:
                prediction = candidate["prediction"]
                if prediction is None:
                    corrections.append(0.0)
                    diagnostics.append(
                        {
                            "status": "no_comparable_interval",
                            "capture_age_ticks": recalled["capture_age_ticks"],
                            "controller_influence_bound": 0.0,
                            "evidence": candidate["evidence"],
                        }
                    )
                    continue
                delta = prediction["physiology_delta"]
                after = {
                    name: float(np.clip(current[name] + delta[name], 0.0, 1.0))
                    for name in ("energy", "gut", "fatigue")
                }
                outcomes = prediction["contextual_physical_outcome"]
                utility = float(
                    drive(current) - drive(after)
                    + weights["nutrition_weight"]
                    * max(0.0, outcomes.get("nutrition", 0.0))
                    * max(0.0, 1.0 - after["energy"])
                    - weights["effort_weight"]
                    * max(0.0, outcomes.get("effort", 0.0))
                )
                bound = float(candidate["controller_influence_bound"])
                correction = bound * math.tanh(weights["correction_gain"] * utility)
                corrections.append(correction)
                diagnostics.append(
                    {
                        "status": "comparable_empirical_interval",
                        "capture_age_ticks": recalled["capture_age_ticks"],
                        "predicted_after_physiology": after,
                        "predicted_physiology_delta": copy.deepcopy(delta),
                        "predicted_physical_outcome": copy.deepcopy(outcomes),
                        "predicted_utility": utility,
                        "controller_influence_bound": bound,
                        "support_count": prediction["support_count"],
                        "effective_support": prediction["effective_support"],
                        "evidence": candidate["evidence"],
                    }
                )
            return {
                "source": "native-visual-episodes-v2",
                "corrections": corrections,
                "diagnostics": diagnostics,
            }

        return callback

    def snapshot(self) -> dict[str, Any]:
        state = {
            "version": self.VERSION,
            "config": asdict(self.config),
            "outcome_names": list(self.outcome_names),
            "physiology_names": list(self.physiology_names),
            "clock": self.clock,
            "model_revision": self.model_revision,
            "pooling_version": self.pooling_version,
            "outcome_aggregation": self.OUTCOME_AGGREGATION,
            "history_migration": copy.deepcopy(self.history_migration),
            "records": copy.deepcopy(self._records),
        }
        return {
            "format": self.FORMAT,
            "sha256": hashlib.sha256(_canonical(state)).hexdigest(),
            "state": state,
        }

    @classmethod
    def restore(cls, value: Any) -> "VisualEpisodeMemory":
        if not isinstance(value, Mapping) or value.get("format") not in {
            cls.FORMAT, cls.LEGACY_FORMAT
        }:
            raise ValueError("unsupported visual episode snapshot")
        source_format = value.get("format")
        state = value.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("visual episode snapshot has no state")
        if value.get("sha256") != hashlib.sha256(_canonical(state)).hexdigest():
            raise ValueError("visual episode snapshot hash differs")
        expected_version = 1 if source_format == cls.LEGACY_FORMAT else cls.VERSION
        if state.get("version") != expected_version:
            raise ValueError("unsupported visual episode state version")
        if source_format == cls.FORMAT and state.get(
            "outcome_aggregation"
        ) != cls.OUTCOME_AGGREGATION:
            raise ValueError("visual episode outcome aggregation differs")
        instance = cls(
            config=VisualEpisodeConfig(**state["config"]),
            outcome_names=state["outcome_names"],
            physiology_names=state["physiology_names"],
        )
        records = state.get("records")
        if not isinstance(records, list) or len(records) > instance.config.capacity:
            raise ValueError("invalid visual episode record collection")
        sequences = []
        for saved in records:
            if not isinstance(saved, Mapping):
                raise ValueError("invalid visual episode record")
            rebuilt = instance._make_record(
                saved.get("start_capture"), saved.get("end_capture"), saved.get("steps")
            )
            if rebuilt["interval_sha256"] != saved.get("interval_sha256"):
                raise ValueError("visual episode identity differs")
            if source_format == cls.FORMAT and (
                rebuilt["summary"] != saved.get("summary")
                or saved.get("schema_version") != cls.VERSION
                or saved.get("outcome_aggregation") != cls.OUTCOME_AGGREGATION
            ):
                raise ValueError("visual episode derived data differs")
            sequence = _tick(saved.get("sequence"), "record sequence")
            rebuilt["sequence"] = sequence
            sequences.append(sequence)
            instance._records.append(rebuilt)
        if len(sequences) != len(set(sequences)) or sequences != sorted(sequences):
            raise ValueError("visual episode sequences are not unique and ordered")
        instance.clock = _tick(state.get("clock"), "visual episode clock")
        if sequences and instance.clock < sequences[-1]:
            raise ValueError("visual episode clock precedes records")
        instance.model_revision = state.get("model_revision")
        instance.pooling_version = state.get("pooling_version")
        if source_format == cls.LEGACY_FORMAT:
            instance.history_migration = {
                "from_format": cls.LEGACY_FORMAT,
                "source_snapshot_sha256": value.get("sha256"),
                "method": (
                    "revalidated raw captures and five retained steps; recomputed "
                    "all v2 summaries with effort integrated as sum(rate*dt)"
                ),
            }
        else:
            migration = state.get("history_migration")
            if migration is not None and not isinstance(migration, Mapping):
                raise ValueError("invalid visual episode migration record")
            instance.history_migration = copy.deepcopy(migration)
        if instance.model_revision is not None and (
            not isinstance(instance.model_revision, str)
            or not instance.model_revision
            or len(instance.model_revision) > 128
        ):
            raise ValueError("invalid visual episode model revision")
        if instance.pooling_version is not None and (
            not isinstance(instance.pooling_version, str)
            or not instance.pooling_version
            or len(instance.pooling_version) > 128
        ):
            raise ValueError("invalid visual episode pooling version")
        for record in instance._records:
            if (
                record["start_capture"]["model_revision"] != instance.model_revision
                or record["start_capture"]["pooling_version"]
                != instance.pooling_version
            ):
                raise ValueError("visual episode encoder metadata differs")
        return instance


__all__ = [
    "DEFAULT_OUTCOMES", "DEFAULT_PHYSIOLOGY", "FEATURE_DIMENSION",
    "MOTOR_ACTIONS8", "VisualEpisodeConfig", "VisualEpisodeMemory",
    "feature_sha256",
]
