"""Private visual episodes and action-conditioned relational prediction.

The organ accepts native visual features plus the creature's own action and
outcome stream. It has no world-coordinate, object-ID, or scene-graph input.
Every episode retains its raw native features so a changed learned projection
can re-encode old experience instead of comparing incompatible cached keys.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


SOURCES = {"experienced", "told", "inferred", "imagined"}
SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.:@/+\-]{0,127}\Z")


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result


def _names(value: Iterable[str], name: str) -> tuple[str, ...]:
    result = tuple(value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    if any(not isinstance(item, str) or not SAFE_NAME.fullmatch(item) for item in result):
        raise ValueError(f"invalid {name}")
    return result


def _version(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True)
class VisualWeights:
    input_mean: np.ndarray
    input_scale: np.ndarray
    projection: np.ndarray
    dynamics_weight: np.ndarray
    dynamics_bias: np.ndarray
    outcome_weight: np.ndarray
    outcome_bias: np.ndarray


def _validated_weights(
    weights: VisualWeights, action_count: int, outcome_count: int
) -> tuple[VisualWeights, int, int]:
    input_dim = int(np.asarray(weights.input_mean).size)
    projection = np.asarray(weights.projection)
    if projection.ndim != 2:
        raise ValueError("projection must be a matrix")
    latent_dim = int(projection.shape[0])
    result = VisualWeights(
        _vector(weights.input_mean, input_dim, "input mean"),
        _vector(weights.input_scale, input_dim, "input scale"),
        _matrix(weights.projection, (latent_dim, input_dim), "projection"),
        _matrix(
            weights.dynamics_weight,
            (latent_dim, latent_dim + action_count),
            "dynamics weight",
        ),
        _vector(weights.dynamics_bias, latent_dim, "dynamics bias"),
        _matrix(
            weights.outcome_weight,
            (outcome_count, latent_dim + action_count),
            "outcome weight",
        ),
        _vector(weights.outcome_bias, outcome_count, "outcome bias"),
    )
    if np.any(result.input_scale <= 0):
        raise ValueError("input scale must be positive")
    return result, input_dim, latent_dim


def _path_matrices(
    context_dim: int, action_count: int, outcome_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stable fixed recurrence without any spatial coordinates."""

    rng = np.random.default_rng(seed)
    recurrent = rng.normal(0.0, 1.0, (context_dim, context_dim)).astype(np.float32)
    norm = max(float(np.linalg.svd(recurrent, compute_uv=False)[0]), 1e-6)
    recurrent *= np.float32(0.72 / norm)
    drive = rng.normal(
        0.0, 0.35, (context_dim, action_count + outcome_count)
    ).astype(np.float32)
    return recurrent, drive


class VisualMemory:
    """Fast-bound raw transitions with contextual recall and prediction."""

    VERSION = 2

    def __init__(
        self,
        weights: VisualWeights,
        *,
        action_names: Iterable[str],
        outcome_names: Iterable[str],
        native_encoder_version: str = "native-encoder-unknown",
        projection_version: str = "projection-initial",
        capacity: int = 512,
        context_dim: int = 24,
        context_seed: int = 7301,
        _path_recurrent: Any | None = None,
        _path_drive: Any | None = None,
    ):
        self.action_names = _names(action_names, "action names")
        self.outcome_names = _names(outcome_names, "outcome names")
        self.native_encoder_version = _version(
            native_encoder_version, "native encoder version"
        )
        self.projection_version = _version(projection_version, "projection version")
        if not 1 <= capacity <= 8192:
            raise ValueError("capacity must be in 1..8192")
        if not 4 <= context_dim <= 256:
            raise ValueError("context_dim must be in 4..256")
        self.capacity = int(capacity)
        self.context_dim = int(context_dim)
        self.context_seed = int(context_seed)
        self.weights, self.input_dim, self.latent_dim = _validated_weights(
            weights, len(self.action_names), len(self.outcome_names)
        )
        if _path_recurrent is None or _path_drive is None:
            recurrent, drive = _path_matrices(
                self.context_dim,
                len(self.action_names),
                len(self.outcome_names),
                self.context_seed,
            )
        else:
            recurrent = _matrix(
                _path_recurrent,
                (self.context_dim, self.context_dim),
                "path recurrent matrix",
            )
            drive = _matrix(
                _path_drive,
                (self.context_dim, len(self.action_names) + len(self.outcome_names)),
                "path drive matrix",
            )
        self.path_recurrent = recurrent
        self.path_drive = drive
        self.path_context = np.zeros(self.context_dim, dtype=np.float32)
        self.records: list[dict[str, Any]] = []
        self.clock = 0

    @classmethod
    def from_artifact(cls, path: str | Path, *, capacity: int = 512) -> "VisualMemory":
        value = json.loads(Path(path).read_text())
        if value.get("format") != "chreatures-visual-weights-v1":
            raise ValueError("unsupported visual weight artifact")
        weights = value.get("weights", {})
        return cls(
            VisualWeights(
                *(np.asarray(weights.get(name), dtype=np.float32)
                  for name in VisualWeights.__dataclass_fields__)
            ),
            action_names=value["action_names"],
            outcome_names=value["outcome_names"],
            native_encoder_version=value.get(
                "native_encoder_version", "native-encoder-unknown"
            ),
            projection_version=value.get("projection_version", "projection-initial"),
            capacity=capacity,
        )

    def replace_projection(
        self,
        weights: VisualWeights,
        *,
        projection_version: str,
        native_encoder_version: str | None = None,
    ) -> None:
        """Install a map while retaining raw episodes for current reprojection."""

        native = self.native_encoder_version if native_encoder_version is None else _version(
            native_encoder_version, "native encoder version"
        )
        if native != self.native_encoder_version and self.records:
            raise ValueError(
                "cannot reinterpret bound raw features with another native encoder"
            )
        checked, input_dim, latent_dim = _validated_weights(
            weights, len(self.action_names), len(self.outcome_names)
        )
        if input_dim != self.input_dim:
            raise ValueError("replacement input dimension differs")
        self.weights = checked
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.native_encoder_version = native
        self.projection_version = _version(projection_version, "projection version")

    def encode(self, feature: Any) -> np.ndarray:
        value = _vector(feature, self.input_dim, "visual feature")
        normalized = np.clip(
            (value - self.weights.input_mean) / self.weights.input_scale, -8.0, 8.0
        )
        latent = self.weights.projection @ normalized
        norm = float(np.linalg.norm(latent))
        return (latent / max(norm, 1e-8)).astype(np.float32)

    def action_vector(self, action: Any) -> np.ndarray:
        if isinstance(action, dict):
            unknown = set(action) - set(self.action_names)
            if unknown:
                raise ValueError(f"unknown action fields: {sorted(unknown)}")
            value = [action.get(name, 0.0) for name in self.action_names]
        else:
            value = action
        return _vector(value, len(self.action_names), "action")

    def outcome_vector(self, outcome: Any) -> np.ndarray:
        if not isinstance(outcome, dict):
            raise ValueError("outcome must be a mapping")
        unknown = set(outcome) - set(self.outcome_names)
        if unknown:
            raise ValueError(f"unknown outcome fields: {sorted(unknown)}")
        return _vector(
            [outcome.get(name, 0.0) for name in self.outcome_names],
            len(self.outcome_names),
            "outcome",
        )

    def reset_path(self) -> None:
        """Start a bodily trajectory without deleting episodic memory."""

        self.path_context.fill(0.0)

    def _advance_path(self, action: np.ndarray, outcome: np.ndarray) -> None:
        drive = np.concatenate((action, outcome)).astype(np.float32)
        self.path_context = np.tanh(
            self.path_recurrent @ self.path_context + self.path_drive @ drive
        ).astype(np.float32)

    def predict(self, feature: Any, action: Any) -> dict[str, Any]:
        latent = self.encode(feature)
        action_value = self.action_vector(action)
        joined = np.concatenate((latent, action_value)).astype(np.float32)
        predicted = self.weights.dynamics_weight @ joined + self.weights.dynamics_bias
        norm = float(np.linalg.norm(predicted))
        predicted = (predicted / max(norm, 1e-8)).astype(np.float32)
        outcome = self.weights.outcome_weight @ joined + self.weights.outcome_bias
        return {
            "next_representation": predicted.tolist(),
            "representation_version": self.projection_version,
            "affordance": {
                name: float(value) for name, value in zip(self.outcome_names, outcome)
            },
            "basis": "learned experienced transitions",
        }

    def bind(
        self,
        feature: Any,
        action: Any,
        outcome: dict[str, Any],
        next_feature: Any,
        *,
        model_time: float,
        source: str = "experienced",
    ) -> dict[str, Any]:
        """Bind one raw visual relation immediately; no repetition is required."""

        if source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}")
        if isinstance(model_time, bool) or not math.isfinite(float(model_time)):
            raise ValueError("model_time must be finite")
        raw = _vector(feature, self.input_dim, "visual feature")
        raw_next = _vector(next_feature, self.input_dim, "next visual feature")
        next_visual = self.encode(raw_next)
        action_value = self.action_vector(action)
        outcome_value = self.outcome_vector(outcome)
        prediction = self.predict(raw, action_value)
        predicted = np.asarray(prediction["next_representation"], dtype=np.float32)
        error = float(1.0 - np.clip(np.dot(predicted, next_visual), -1.0, 1.0))
        self.clock += 1
        record = {
            "sequence": self.clock,
            "model_time": float(model_time),
            "source": source,
            "native_encoder_version": self.native_encoder_version,
            "bound_projection_version": self.projection_version,
            "raw_visual": raw.tolist(),
            "path_context": self.path_context.tolist(),
            "action": action_value.tolist(),
            "outcome": outcome_value.tolist(),
            "raw_next_visual": raw_next.tolist(),
            "prediction_error_at_binding": error,
        }
        self.records.append(record)
        self._advance_path(action_value, outcome_value)
        if len(self.records) > self.capacity:
            errors = np.asarray(
                [item["prediction_error_at_binding"] for item in self.records]
            )
            consequences = np.asarray(
                [np.linalg.norm(item["outcome"]) for item in self.records]
            )
            recency = np.asarray([item["sequence"] / self.clock for item in self.records])
            remove = int(np.argmin(errors + consequences + 0.2 * recency))
            del self.records[remove]
        return copy.deepcopy(record)

    def recall(
        self,
        feature: Any,
        action: Any,
        *,
        limit: int = 5,
        sources: Iterable[str] | None = None,
        use_context: bool = True,
        similarity: str = "cosine",
    ) -> dict[str, Any]:
        if not 1 <= limit <= 32:
            raise ValueError("limit must be in 1..32")
        if similarity not in {"cosine", "euclidean", "manhattan"}:
            raise ValueError("similarity must be cosine, euclidean, or manhattan")
        allowed = set(SOURCES if sources is None else sources)
        if not allowed <= SOURCES:
            raise ValueError(f"unknown information sources: {sorted(allowed - SOURCES)}")
        candidates = [record for record in self.records if record["source"] in allowed]
        learned = self.predict(feature, action)
        if not candidates:
            return {
                **learned,
                "neighbors": [],
                "uncertainty": 1.0,
                "retrieval_basis": "learned model only; no bound episodes",
                "similarity": similarity,
            }
        visual = self.encode(feature)
        action_value = self.action_vector(action)
        keys = np.asarray(
            [self.encode(record["raw_visual"]) for record in candidates],
            dtype=np.float32,
        )
        actions = np.asarray([record["action"] for record in candidates], dtype=np.float32)
        delta = keys - visual
        if similarity == "cosine":
            distance = 1.0 - np.clip(keys @ visual, -1.0, 1.0)
        elif similarity == "euclidean":
            distance = 0.5 * np.linalg.norm(delta, axis=1)
        elif similarity == "manhattan":
            distance = 0.5 * np.mean(np.abs(delta), axis=1) * math.sqrt(
                self.latent_dim
            )
        distance += 0.10 * np.mean((actions - action_value) ** 2, axis=1)
        if use_context:
            contexts = np.asarray(
                [record["path_context"] for record in candidates], dtype=np.float32
            )
            distance += 0.35 * np.mean((contexts - self.path_context) ** 2, axis=1)
        indices = np.argsort(distance)[: min(limit, len(candidates))]
        strengths = np.exp(-distance[indices] * 5.0)
        strengths /= max(float(strengths.sum()), 1e-9)
        episodic_next = sum(
            strength * self.encode(candidates[index]["raw_next_visual"])
            for strength, index in zip(strengths, indices)
        )
        episodic_outcome = sum(
            strength * np.asarray(candidates[index]["outcome"], dtype=np.float32)
            for strength, index in zip(strengths, indices)
        )
        learned_next = np.asarray(learned["next_representation"], dtype=np.float32)
        combined = 0.65 * episodic_next + 0.35 * learned_next
        combined /= max(float(np.linalg.norm(combined)), 1e-8)
        learned_outcome = np.asarray(
            [learned["affordance"][name] for name in self.outcome_names], dtype=np.float32
        )
        combined_outcome = 0.65 * episodic_outcome + 0.35 * learned_outcome
        return {
            "next_representation": combined.tolist(),
            "representation_version": self.projection_version,
            "native_encoder_version": self.native_encoder_version,
            "affordance": {
                name: float(value)
                for name, value in zip(self.outcome_names, combined_outcome)
            },
            "uncertainty": float(np.clip(distance[indices[0]] / 2.0, 0.0, 1.0)),
            "retrieval_basis": "reprojected private episodes plus learned transition",
            "similarity": similarity,
            "neighbors": [
                {
                    "sequence": candidates[index]["sequence"],
                    "model_time": candidates[index]["model_time"],
                    "source": candidates[index]["source"],
                    "distance": float(distance[index]),
                    "bound_projection_version": candidates[index][
                        "bound_projection_version"
                    ],
                    "current_projection_version": self.projection_version,
                }
                for index in indices
            ],
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "native_encoder_version": self.native_encoder_version,
            "projection_version": self.projection_version,
            "action_names": list(self.action_names),
            "outcome_names": list(self.outcome_names),
            "capacity": self.capacity,
            "clock": self.clock,
            "context_dim": self.context_dim,
            "context_seed": self.context_seed,
            "path_context": self.path_context.tolist(),
            "path_recurrent": self.path_recurrent.tolist(),
            "path_drive": self.path_drive.tolist(),
            "weights": {
                name: getattr(self.weights, name).tolist()
                for name in VisualWeights.__dataclass_fields__
            },
            "records": copy.deepcopy(self.records),
        }

    @classmethod
    def restore(cls, value: Any) -> "VisualMemory":
        if not isinstance(value, dict) or value.get("version") != cls.VERSION:
            raise ValueError("unsupported visual memory checkpoint")
        weights = value.get("weights", {})
        instance = cls(
            VisualWeights(
                *(np.asarray(weights.get(name), dtype=np.float32)
                  for name in VisualWeights.__dataclass_fields__)
            ),
            action_names=value["action_names"],
            outcome_names=value["outcome_names"],
            native_encoder_version=value["native_encoder_version"],
            projection_version=value["projection_version"],
            capacity=int(value["capacity"]),
            context_dim=int(value["context_dim"]),
            context_seed=int(value["context_seed"]),
            _path_recurrent=value["path_recurrent"],
            _path_drive=value["path_drive"],
        )
        if value.get("input_dim") != instance.input_dim or value.get("latent_dim") != instance.latent_dim:
            raise ValueError("visual memory dimensions differ")
        records = value.get("records")
        if not isinstance(records, list) or len(records) > instance.capacity:
            raise ValueError("invalid visual episode collection")
        instance.clock = int(value.get("clock"))
        if instance.clock < len(records):
            raise ValueError("visual memory clock precedes its records")
        instance.path_context = _vector(
            value.get("path_context"), instance.context_dim, "path context"
        )
        instance.records = copy.deepcopy(records)
        for record in instance.records:
            try:
                if record.get("source") not in SOURCES:
                    raise ValueError("unknown episode information source")
                if record.get("native_encoder_version") != instance.native_encoder_version:
                    raise ValueError("episode native encoder version differs")
                _version(record["bound_projection_version"], "bound projection version")
                _vector(record["raw_visual"], instance.input_dim, "episode raw visual")
                _vector(record["path_context"], instance.context_dim, "episode path context")
                _vector(record["action"], len(instance.action_names), "episode action")
                _vector(record["outcome"], len(instance.outcome_names), "episode outcome")
                _vector(
                    record["raw_next_visual"], instance.input_dim,
                    "episode next raw visual",
                )
                if (
                    int(record["sequence"]) < 1
                    or int(record["sequence"]) > instance.clock
                    or not math.isfinite(float(record["model_time"]))
                    or not math.isfinite(float(record["prediction_error_at_binding"]))
                ):
                    raise ValueError("invalid episode sequence or time")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid visual episode") from error
        sequences = [int(record["sequence"]) for record in instance.records]
        if len(sequences) != len(set(sequences)):
            raise ValueError("visual episode sequences are not unique")
        return instance


__all__ = ["SOURCES", "VisualMemory", "VisualWeights"]
