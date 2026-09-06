"""Checked one-copy adapter for the native recurrent predictive state organ."""

from __future__ import annotations
import hashlib, importlib, json
from pathlib import Path
from typing import Any
import numpy as np

FORMAT = "chreatures-predictive-state-v1"
PACK_ORDER = (
    "normalizer.input_mean",
    "normalizer.input_scale",
    "normalizer.physiology_delta_mean",
    "normalizer.physiology_delta_scale",
    "observation_encoder.0.weight",
    "observation_encoder.0.bias",
    "action_encoder.0.weight",
    "action_encoder.0.bias",
    "observe_rnn.weight_ih_l0",
    "observe_rnn.weight_hh_l0",
    "observe_rnn.bias_ih_l0",
    "observe_rnn.bias_hh_l0",
    "transition_cell.weight_ih",
    "transition_cell.weight_hh",
    "transition_cell.bias_ih",
    "transition_cell.bias_hh",
    "feature_mean.weight",
    "feature_mean.bias",
    "feature_log_std.weight",
    "feature_log_std.bias",
    "physiology_delta_mean.weight",
    "physiology_delta_mean.bias",
    "physiology_delta_log_std.weight",
    "physiology_delta_log_std.bias",
)
SOURCE_NORMALIZER_ORDER = (
    "source_normalizer.count",
    "source_normalizer.mean",
    "source_normalizer.m2",
)


def _extension():
    try:
        return importlib.import_module("_cognitive_core")
    except ImportError as exc:
        raise RuntimeError(
            "native predictive state requested but _cognitive_core is unavailable"
        ) from exc


class NativePredictiveCohort:
    def __init__(self, export: str | Path, batch_size: int):
        export = Path(export)
        export_sha256 = hashlib.sha256(export.read_bytes()).hexdigest()
        with np.load(export, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            config = metadata.get("config", {})
            if metadata.get("format") != FORMAT or metadata.get("version") not in (
                1,
                2,
                3,
            ):
                raise ValueError("incompatible predictive-state export")
            if metadata["version"] == 3:
                temporal = metadata.get("temporal_contract")
                source_identity = metadata.get("training_input_identity", {})
                if (
                    not isinstance(temporal, dict)
                    or temporal.get("source_dataset_manifest_sha256")
                    != source_identity.get("dataset_manifest_sha256")
                    or temporal.get("physics_dt_seconds") != 0.05
                    or temporal.get("macro_steps") != 5
                    or temporal.get("observation_interval_seconds") != 0.25
                ):
                    raise ValueError("predictive temporal contract differs")
            manifest = metadata.get("tensors", {})
            arrays = []
            expected_names = set(PACK_ORDER)
            if metadata["version"] >= 2:
                expected_names.update(SOURCE_NORMALIZER_ORDER)
            if set(manifest) != expected_names:
                raise ValueError("native tensor manifest differs")
            for name in expected_names:
                value = np.asarray(archive[name])
                record = manifest[name]
                if (
                    str(value.dtype) != record["dtype"]
                    or list(value.shape) != record["shape"]
                    or not value.flags.c_contiguous
                ):
                    raise ValueError(f"invalid native tensor {name}")
                if (
                    hashlib.sha256(value.tobytes(order="C")).hexdigest()
                    != record["sha256"]
                ):
                    raise ValueError(f"native tensor checksum differs: {name}")
            for name in PACK_ORDER:
                value = np.asarray(archive[name])
                if value.dtype != np.float32:
                    raise ValueError(f"native model tensor is not float32: {name}")
                arrays.append(value.reshape(-1))
            if metadata["version"] >= 2:
                self._source_count = float(
                    np.asarray(archive["source_normalizer.count"]).reshape(-1)[0]
                )
                self._source_mean = np.asarray(
                    archive["source_normalizer.mean"], dtype=np.float64
                ).copy()
                self._source_m2 = np.asarray(
                    archive["source_normalizer.m2"], dtype=np.float64
                ).copy()
            else:
                self._source_count = None
                self._source_mean = None
                self._source_m2 = None
            packed = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
        self.feature_dim = int(config["feature_dim"])
        self.physiology_dim = int(config["physiology_dim"])
        self.action_dim = int(config["action_dim"])
        self.latent_dim = int(config["latent_dim"])
        self.batch_size = int(batch_size)
        self._native = _extension().PredictiveCohort(
            self.batch_size,
            self.feature_dim,
            self.physiology_dim,
            self.action_dim,
            self.latent_dim,
            int(config["encoder_dim"]),
            max(config["horizons"]),
            packed,
        )
        self.metadata = metadata
        self.model_identity = {
            "format": FORMAT,
            "artifact_sha256": export_sha256,
            "tensor_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        input_record = {
            "training_input_identity": metadata.get(
                "training_input_identity", {"status": "unknown"}
            ),
            "feature_layout": metadata["feature_layout"],
            "actions": metadata["actions"],
            "physiology": metadata["physiology"],
        }
        self.input_identity = {
            "record": input_record,
            "sha256": hashlib.sha256(
                json.dumps(input_record, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    def normalize_source_features(self, raw: np.ndarray) -> np.ndarray:
        """Apply the artifact-bound PPO transform to raw MaleCNS readouts."""
        if self._source_mean is None or self._source_m2 is None:
            raise ValueError("predictive artifact does not embed source normalization")
        raw = np.asarray(raw)
        if (
            raw.ndim != 2
            or raw.shape[1] != self.feature_dim
            or not np.issubdtype(raw.dtype, np.floating)
            or not np.isfinite(raw).all()
        ):
            raise ValueError("raw source features must be finite floating [B,F]")
        variance = self._source_m2 / max(self._source_count, 1.0)
        normalized = (raw.astype(np.float64) - self._source_mean) / np.sqrt(
            np.maximum(variance, 1e-5)
        )
        return np.ascontiguousarray(np.clip(normalized, -5.0, 5.0), dtype=np.float32)

    def observe(
        self,
        features: np.ndarray,
        physiology: np.ndarray,
        previous_action: np.ndarray,
        reset: np.ndarray,
    ) -> np.ndarray:
        values = [
            np.ascontiguousarray(x)
            for x in (features, physiology, previous_action, reset)
        ]
        return np.asarray(self._native.observe(*values))

    def imagine(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        (
            feature_mean,
            feature_scale,
            physiology_mean,
            physiology_scale,
            valid,
            support,
        ) = self._native.imagine(np.ascontiguousarray(actions, dtype=np.float32))
        return {
            "feature_mean": np.asarray(feature_mean),
            "physiology_mean": np.asarray(physiology_mean),
            "feature_residual_scale": np.asarray(feature_scale),
            "physiology_residual_scale": np.asarray(physiology_scale),
            "valid": np.asarray(valid),
            "horizon_support": np.asarray(support),
        }

    def snapshot(self) -> dict[str, Any]:
        state, action, physiology = self._native.snapshot()
        return {
            "latent": np.asarray(state).copy(),
            "previous_action": np.asarray(action).copy(),
            "physiology_anchor": np.asarray(physiology).copy(),
            "model_identity": dict(self.model_identity),
            "input_identity": dict(self.input_identity),
        }

    def restore(self, value: dict[str, Any]) -> None:
        if set(value) != {
            "latent",
            "previous_action",
            "physiology_anchor",
            "model_identity",
            "input_identity",
        }:
            raise ValueError("native predictive snapshot fields differ")
        if value["model_identity"] != self.model_identity:
            raise ValueError("native predictive snapshot model identity differs")
        if value["input_identity"] != self.input_identity:
            raise ValueError("native predictive snapshot input identity differs")
        self._native.restore(
            np.ascontiguousarray(value["latent"], dtype=np.float32),
            np.ascontiguousarray(value["previous_action"], dtype=np.float32),
            np.ascontiguousarray(value["physiology_anchor"], dtype=np.float32),
        )

    def query_from_snapshot(
        self, snapshot: dict[str, Any], actions: np.ndarray
    ) -> dict[str, np.ndarray]:
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        if (
            actions.ndim != 3
            or actions.shape[2] != self.action_dim
            or not 1 <= actions.shape[1] <= self.batch_size
        ):
            raise ValueError(
                "query actions must be [T,Bq,action_dim] with 1<=Bq<=capacity"
            )
        if not 1 <= len(actions) <= 8:
            raise ValueError("query horizon must be in 1..8")
        query = actions.shape[1]
        padded: dict[str, Any] = {
            "model_identity": snapshot["model_identity"],
            "input_identity": snapshot["input_identity"],
        }
        for name, width in (
            ("latent", self.latent_dim),
            ("previous_action", self.action_dim),
            ("physiology_anchor", self.physiology_dim),
        ):
            source = np.asarray(snapshot[name], dtype=np.float32)
            if source.shape[0] == 1:
                source = np.repeat(source, query, axis=0)
            if source.shape != (query, width):
                raise ValueError(f"snapshot {name} cannot tile to query batch")
            padded[name] = np.zeros((self.batch_size, width), np.float32)
            padded[name][:query] = source
        self.restore(padded)
        action_pad = np.zeros(
            (len(actions), self.batch_size, self.action_dim), np.float32
        )
        action_pad[:, :query] = actions
        return {
            name: value[:, :query].copy()
            for name, value in self.imagine(action_pad).items()
        }
