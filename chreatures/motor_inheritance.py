"""Torch-free deployment of an inherited predictive-PPO motor organ.

The NPZ artifact contains immutable population-trained parameters and a frozen
feature normalizer.  Each :class:`MotorOrgan` owns only private working state.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


ACTIONS = (
    "thrust", "yaw", "gaze_pitch", "grip",
    "signal_low", "signal_mid", "signal_high", "posture",
)
PHYSIOLOGY = ("energy", "gut", "fatigue", "speed", "angular_velocity", "support")
ARTIFACT_FORMAT = "chreatures-predictive-ppo-motor-organ-v1"
SNAPSHOT_FORMAT = "chreatures-motor-organ-snapshot-v1"


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


def artifact_identity(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    """Stable content identity independent of ZIP compression and timestamps."""
    clean = dict(metadata)
    clean.pop("artifact_sha256", None)
    digest = hashlib.sha256(_json(clean).encode())
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(_json(list(value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class MotorArtifact:
    """Validated immutable arrays used by any number of private residents."""

    REQUIRED = (
        "log_std", "projection", "context_feature", "context_action", "context_recur",
        "feature_encoder.0.weight", "feature_encoder.0.bias",
        "trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
        "policy_mean.weight", "policy_mean.bias", "value.weight", "value.bias",
        "predictor.0.weight", "predictor.0.bias", "predictor.2.weight", "predictor.2.bias",
        "normalizer_mean", "normalizer_m2", "normalizer_count",
    )

    def __init__(self, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
        self.metadata = dict(metadata)
        if self.metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError("incompatible motor artifact format")
        if tuple(self.metadata.get("actions", ())) != ACTIONS:
            raise ValueError("motor artifact action schema differs")
        missing = set(self.REQUIRED) - set(arrays)
        if missing:
            raise ValueError(f"motor artifact is missing arrays: {sorted(missing)}")
        validated = {}
        for name in self.REQUIRED:
            value = np.asarray(arrays[name])
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"invalid motor artifact array {name}")
            value = np.asarray(value, dtype=np.float64 if name.startswith("normalizer_") else np.float32)
            value.setflags(write=False)
            validated[name] = value
        self.arrays = MappingProxyType(validated)
        expected = str(self.metadata.get("artifact_sha256", ""))
        actual = artifact_identity(self.metadata, self.arrays)
        if len(expected) != 64 or expected != actual:
            raise ValueError("motor artifact content identity differs")
        self.sha256 = actual
        self.config = dict(self.metadata["config"])
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        c = self.config
        f, p, x, h, q = (int(c[k]) for k in (
            "feature_dim", "physiology_dim", "context_dim", "hidden_dim", "projection_dim"
        ))
        if p != len(PHYSIOLOGY) or int(c["macro_steps"]) <= 0:
            raise ValueError("unsupported motor artifact dimensions")
        shapes = {
            "log_std": (len(ACTIONS),), "projection": (q, f),
            "context_feature": (x, q), "context_action": (x, len(ACTIONS)),
            "context_recur": (x, x), "feature_encoder.0.weight": (h, f),
            "feature_encoder.0.bias": (h,), "trunk.0.weight": (h, h + p + x),
            "trunk.0.bias": (h,), "trunk.2.weight": (h, h), "trunk.2.bias": (h,),
            "policy_mean.weight": (len(ACTIONS), h), "policy_mean.bias": (len(ACTIONS),),
            "value.weight": (1, h), "value.bias": (1,),
            "predictor.0.weight": (h, h + len(ACTIONS)), "predictor.0.bias": (h,),
            "predictor.2.weight": (q, h), "predictor.2.bias": (q,),
            "normalizer_mean": (f,), "normalizer_m2": (f,), "normalizer_count": (),
        }
        for name, shape in shapes.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"motor artifact array {name} has shape {self.arrays[name].shape}, expected {shape}")
        if float(self.arrays["normalizer_count"]) <= 0 or np.any(self.arrays["normalizer_m2"] < 0):
            raise ValueError("normalizer count/moments are invalid")
        provenance = self.metadata.get("training_provenance", {})
        for key in ("checkpoint_sha256", "graph_sha256", "port_spec_sha256"):
            value = str(provenance.get(key, ""))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                raise ValueError(f"motor artifact lacks a valid {key}")

    @classmethod
    def load(cls, path: str | Path) -> "MotorArtifact":
        with np.load(Path(path), allow_pickle=False) as value:
            metadata = json.loads(str(value["metadata"]))
            arrays = {name: np.asarray(value[name]) for name in value.files if name != "metadata"}
        return cls(metadata, arrays)

    def to_value(self) -> dict[str, Any]:
        """Return a JSON-compatible, dtype-preserving artifact value."""
        return {
            "metadata": self.metadata,
            "arrays": {name: _array_value(value) for name, value in self.arrays.items()},
        }

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "MotorArtifact":
        try:
            metadata = value["metadata"]
            encoded = value["arrays"]
            arrays = {name: _array_from_value(item, name) for name, item in encoded.items()}
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValueError("invalid encoded motor artifact") from exc
        return cls(metadata, arrays)


class MotorOrgan:
    """One resident's stateful inherited policy, using NumPy only."""

    def __init__(
        self, artifact: MotorArtifact | str | Path, *, seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        self.artifact = artifact if isinstance(artifact, MotorArtifact) else MotorArtifact.load(artifact)
        c = self.artifact.config
        self.context = np.zeros(int(c["context_dim"]), dtype=np.float32)
        self.rng = np.random.default_rng(int(c["seed"]) + 311 if seed is None else int(seed))
        self.deterministic = bool(deterministic)
        self.held_action = np.zeros(len(ACTIONS), dtype=np.float32)
        self.held_ticks = 0
        self.macro_time = 0.0
        self.previous_features: np.ndarray | None = None
        self.previous_prediction: np.ndarray | None = None
        self.last_prediction_error: float | None = None
        self.decision_count = 0

    def normalize(self, raw_features: Any) -> np.ndarray:
        raw = np.asarray(raw_features, dtype=np.float32)
        expected = (int(self.artifact.config["feature_dim"]),)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(f"features must be finite with shape {expected}")
        a = self.artifact.arrays
        variance = a["normalizer_m2"] / max(float(a["normalizer_count"]), 1.0)
        normalized = (raw.astype(np.float64) - a["normalizer_mean"]) / np.sqrt(np.maximum(variance, 1e-5))
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    @staticmethod
    def physiology_vector(local: Mapping[str, Any] | np.ndarray) -> np.ndarray:
        if isinstance(local, Mapping):
            unknown = set(local) - set(PHYSIOLOGY)
            if unknown:
                raise ValueError(f"physiology contains nonlocal/unknown fields: {sorted(unknown)}")
            values = np.asarray([
                local.get("energy", .7), local.get("gut", 0), local.get("fatigue", .1),
                math.tanh(float(local.get("speed", 0)) / 2),
                math.tanh(float(local.get("angular_velocity", 0)) / 4),
                local.get("support", 1),
            ], dtype=np.float32)
        else:
            values = np.asarray(local, dtype=np.float32)
        if values.shape != (len(PHYSIOLOGY),) or not np.isfinite(values).all():
            raise ValueError("physiology must contain six finite local values")
        return values

    def _linear(self, name: str, value: np.ndarray) -> np.ndarray:
        a = self.artifact.arrays
        return value @ a[name + ".weight"].T + a[name + ".bias"]

    def forward(self, normalized: np.ndarray, physiology: np.ndarray) -> tuple[np.ndarray, np.float32, np.ndarray]:
        encoded = np.tanh(self._linear("feature_encoder.0", normalized)).astype(np.float32)
        joined = np.concatenate((encoded, physiology, self.context)).astype(np.float32)
        hidden = np.tanh(self._linear("trunk.0", joined)).astype(np.float32)
        hidden = np.tanh(self._linear("trunk.2", hidden)).astype(np.float32)
        mean = self._linear("policy_mean", hidden).astype(np.float32)
        value = np.float32(self._linear("value", hidden)[0])
        return mean, value, hidden

    def projected(self, normalized: np.ndarray) -> np.ndarray:
        return np.tanh(normalized @ self.artifact.arrays["projection"].T).astype(np.float32)

    def predictor(self, hidden: np.ndarray, action: np.ndarray) -> np.ndarray:
        joined = np.concatenate((hidden, action)).astype(np.float32)
        hidden_prediction = np.tanh(self._linear("predictor.0", joined)).astype(np.float32)
        return self._linear("predictor.2", hidden_prediction).astype(np.float32)

    def _update_context(self, next_features: np.ndarray) -> None:
        a = self.artifact.arrays
        projected = self.projected(next_features)
        self.context = np.tanh(
            projected @ a["context_feature"].T
            + self.held_action @ a["context_action"].T
            + self.context @ a["context_recur"].T
        ).astype(np.float32)
        if self.previous_features is not None and self.previous_prediction is not None:
            target = projected - self.projected(self.previous_features)
            self.last_prediction_error = float(np.mean((target - self.previous_prediction) ** 2))

    @staticmethod
    def _physical_action(action: np.ndarray) -> dict[str, float]:
        physical = np.asarray(action, dtype=np.float32).copy()
        physical[[3, 4, 5, 6]] = np.maximum(physical[[3, 4, 5, 6]], 0)
        return {name: float(value) for name, value in zip(ACTIONS, physical, strict=True)}

    def _commit_macro_action(
        self, normalized_features: np.ndarray, hidden: np.ndarray,
        selected_action: np.ndarray, dt: float,
    ) -> dict[str, float]:
        self.held_action = selected_action.copy()
        self.previous_features = normalized_features.copy()
        self.previous_prediction = self.predictor(hidden, self.held_action)
        self.held_ticks = 1
        self.macro_time = float(dt)
        self.decision_count += 1
        return self._physical_action(self.held_action)

    def commit_macro_action(
        self, normalized_features: Any, hidden: Any,
        selected_physical_action: Any, dt: float,
    ) -> dict[str, float]:
        """Commit a contextual selector's action as the macro's first held tick."""
        if self.held_ticks != 0:
            raise ValueError("a macro action can be committed only at an empty boundary")
        if not np.isfinite(dt) or not 0 < dt <= .2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        features = np.asarray(normalized_features, dtype=np.float32)
        hidden = np.asarray(hidden, dtype=np.float32)
        action = np.asarray(selected_physical_action, dtype=np.float32)
        expected_features = (int(self.artifact.config["feature_dim"]),)
        expected_hidden = (int(self.artifact.config["hidden_dim"]),)
        if features.shape != expected_features or not np.isfinite(features).all():
            raise ValueError(f"normalized features must be finite with shape {expected_features}")
        if hidden.shape != expected_hidden or not np.isfinite(hidden).all():
            raise ValueError(f"hidden state must be finite with shape {expected_hidden}")
        if action.shape != (len(ACTIONS),) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("selected physical action must be eight finite values in [-1, 1]")
        if np.any(action[[3, 4, 5, 6]] < 0):
            raise ValueError("physical grip and signal actions must be nonnegative")
        return self._commit_macro_action(features, hidden, action, float(dt))

    def continue_macro_action(self, dt: float) -> dict[str, float]:
        """Account for and return physical ticks two through five of a committed action."""
        macro_steps = int(self.artifact.config["macro_steps"])
        if not 0 < self.held_ticks < macro_steps:
            raise ValueError("there is no committed macro action with a remaining held tick")
        if not np.isfinite(dt) or not 0 < dt <= .2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        self.held_ticks += 1
        self.macro_time += float(dt)
        return self._physical_action(self.held_action)

    def open_macro_boundary(self, next_normalized_features: Any) -> np.ndarray:
        """Finish a five-tick action and expose the updated private context."""
        macro_steps = int(self.artifact.config["macro_steps"])
        if self.held_ticks != macro_steps:
            raise ValueError("macro boundary opens only after every held physical tick")
        features = np.asarray(next_normalized_features, dtype=np.float32)
        expected = (int(self.artifact.config["feature_dim"]),)
        if features.shape != expected or not np.isfinite(features).all():
            raise ValueError(f"normalized features must be finite with shape {expected}")
        self._update_context(features)
        self.held_ticks = 0
        self.macro_time = 0.0
        return self.context.copy()

    def tick(self, features: Any, local_physiology: Mapping[str, Any] | np.ndarray, dt: float) -> dict[str, float]:
        """Return one physical-tick action, updating context only at macro boundaries."""
        if not np.isfinite(dt) or not 0 < dt <= .2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        normalized = self.normalize(features)
        physiology = self.physiology_vector(local_physiology)
        macro_steps = int(self.artifact.config["macro_steps"])
        if self.held_ticks >= macro_steps:
            self.open_macro_boundary(normalized)
        if self.held_ticks == 0:
            mean, _value, hidden = self.forward(normalized, physiology)
            if self.deterministic:
                latent = mean
            else:
                std = np.exp(np.clip(self.artifact.arrays["log_std"], -3.5, .3)).astype(np.float32)
                latent = mean + std * self.rng.standard_normal(len(ACTIONS), dtype=np.float32)
            return self._commit_macro_action(
                normalized, hidden, np.tanh(latent).astype(np.float32), float(dt)
            )
        self.held_ticks += 1
        self.macro_time += float(dt)
        return self._physical_action(self.held_action)

    def reset_episode(self) -> None:
        self.context.fill(0)
        self.held_action.fill(0)
        self.held_ticks = 0
        self.macro_time = 0.0
        self.previous_features = None
        self.previous_prediction = None
        self.last_prediction_error = None

    def view(self) -> dict[str, Any]:
        """Compact inherited-policy working state for inspection."""
        return {
            "artifact_sha256": self.artifact.sha256,
            "context": self.context.astype(float).tolist(),
            "decision_count": self.decision_count,
            "prediction_error": self.last_prediction_error,
            "action": self._physical_action(self.held_action),
            "held_ticks": self.held_ticks,
        }

    def snapshot_value(self, *, include_artifact: bool = False) -> dict[str, Any]:
        """Return JSON-compatible private state, optionally embedding its artifact."""
        arrays = {"context": _array_value(self.context), "held_action": _array_value(self.held_action)}
        if self.previous_features is not None:
            arrays["previous_features"] = _array_value(self.previous_features)
            arrays["previous_prediction"] = _array_value(self.previous_prediction)
        value = {
            "format": SNAPSHOT_FORMAT, "version": 1,
            "artifact_sha256": self.artifact.sha256,
            "deterministic": self.deterministic, "held_ticks": self.held_ticks,
            "macro_time": self.macro_time, "decision_count": self.decision_count,
            "rng": self.rng.bit_generator.state,
            "has_previous": self.previous_features is not None,
            "last_prediction_error": self.last_prediction_error,
            "arrays": arrays,
        }
        if include_artifact:
            value["artifact"] = self.artifact.to_value()
        return value

    @classmethod
    def restore_value(
        cls, value: Mapping[str, Any], artifact: MotorArtifact | str | Path | None = None,
    ) -> "MotorOrgan":
        """Restore private state against a shared or embedded immutable artifact."""
        try:
            if value.get("format") != SNAPSHOT_FORMAT or value.get("version") != 1:
                raise ValueError("unsupported motor snapshot")
            if artifact is None:
                artifact = MotorArtifact.from_value(value["artifact"])
            elif not isinstance(artifact, MotorArtifact):
                artifact = MotorArtifact.load(artifact)
            if artifact.sha256 != value.get("artifact_sha256"):
                raise ValueError("motor snapshot artifact identity differs")
            arrays = value["arrays"]
            instance = cls(artifact, deterministic=bool(value["deterministic"]))
            instance.context = _array_from_value(arrays["context"], "context").astype(np.float32, copy=False)
            instance.held_action = _array_from_value(arrays["held_action"], "held_action").astype(np.float32, copy=False)
            if value["has_previous"]:
                instance.previous_features = _array_from_value(arrays["previous_features"], "previous_features").astype(np.float32, copy=False)
                instance.previous_prediction = _array_from_value(arrays["previous_prediction"], "previous_prediction").astype(np.float32, copy=False)
            instance.held_ticks = int(value["held_ticks"])
            instance.macro_time = float(value["macro_time"])
            instance.decision_count = int(value["decision_count"])
            instance.last_prediction_error = value["last_prediction_error"]
            instance.rng.bit_generator.state = value["rng"]
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("invalid encoded motor snapshot") from exc
        instance._validate_private_state()
        return instance

    def _validate_private_state(self) -> None:
        artifact = self.artifact
        if not 0 <= self.held_ticks <= int(artifact.config["macro_steps"]):
            raise ValueError("invalid held-action position in motor snapshot")
        if self.context.shape != (int(artifact.config["context_dim"]),):
            raise ValueError("motor snapshot context shape differs")
        if self.held_action.shape != (len(ACTIONS),):
            raise ValueError("motor snapshot held-action shape differs")
        if (self.previous_features is None) != (self.previous_prediction is None):
            raise ValueError("motor snapshot macro history is incomplete")
        if self.previous_features is not None and (
            self.previous_features.shape != (int(artifact.config["feature_dim"]),)
            or self.previous_prediction.shape != (int(artifact.config["projection_dim"]),)
        ):
            raise ValueError("motor snapshot macro history shape differs")
        private = [self.context, self.held_action]
        if self.previous_features is not None:
            private.extend((self.previous_features, self.previous_prediction))
        if not all(np.isfinite(item).all() for item in private):
            raise ValueError("motor snapshot contains nonfinite private state")
        if not np.isfinite(self.macro_time) or self.macro_time < 0 or self.decision_count < 0:
            raise ValueError("motor snapshot counters are invalid")

    def snapshot(self, path: str | Path) -> dict[str, Any]:
        """Write a self-contained exact resident continuation without pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": SNAPSHOT_FORMAT, "version": 1,
            "artifact_sha256": self.artifact.sha256,
            "artifact_metadata": self.artifact.metadata,
            "deterministic": self.deterministic, "held_ticks": self.held_ticks,
            "macro_time": self.macro_time, "decision_count": self.decision_count,
            "rng": self.rng.bit_generator.state,
            "has_previous": self.previous_features is not None,
            "last_prediction_error": self.last_prediction_error,
        }
        arrays = {"context": self.context, "held_action": self.held_action}
        if self.previous_features is not None:
            arrays["previous_features"] = self.previous_features
            arrays["previous_prediction"] = self.previous_prediction
        arrays.update({f"artifact::{name}": value for name, value in self.artifact.arrays.items()})
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, metadata=np.asarray(_json(metadata)), **arrays)
        temporary.replace(path)
        return {"path": str(path), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "artifact_sha256": self.artifact.sha256}

    @classmethod
    def restore(cls, path: str | Path, *, expected_sha256: str | None = None) -> "MotorOrgan":
        path = Path(path)
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and file_hash != expected_sha256:
            raise ValueError("motor snapshot file checksum differs")
        with np.load(path, allow_pickle=False) as value:
            metadata = json.loads(str(value["metadata"]))
            if metadata.get("format") != SNAPSHOT_FORMAT or metadata.get("version") != 1:
                raise ValueError("unsupported motor snapshot")
            artifact_arrays = {
                name.removeprefix("artifact::"): np.asarray(value[name])
                for name in value.files if name.startswith("artifact::")
            }
            artifact = MotorArtifact(metadata["artifact_metadata"], artifact_arrays)
            if artifact.sha256 != metadata.get("artifact_sha256"):
                raise ValueError("embedded motor artifact identity differs")
            private = {
                "format": SNAPSHOT_FORMAT, "version": 1,
                "artifact_sha256": artifact.sha256,
                **{name: metadata[name] for name in (
                    "deterministic", "held_ticks", "macro_time", "decision_count", "rng",
                    "has_previous", "last_prediction_error",
                )},
                "arrays": {
                    name: _array_value(np.asarray(value[name]))
                    for name in ("context", "held_action", "previous_features", "previous_prediction")
                    if name in value.files
                },
            }
        return cls.restore_value(private, artifact)
