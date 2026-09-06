"""Validated Torch-free boundary for the native sensorimotor worker."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DEVELOPMENTAL_FORMAT = "chreatures-native-developmental-resident-rich-v3"
PERSONAL_GOAL_CONTRACT = {
    "format": "chreatures-private-goal-associations-v1",
    "objective": {
        "format": "chreatures-finite-energy-homeostasis-v1",
        "sha256": "01ae937a153a056c8cc5fa5be4d55cdfb38dbfcede4dbceb16ec33e19c5f4d00",
        "config": {
            "version": 1,
            "assimilation_efficiency": 0.84,
            "reserve_target": 0.85,
            "reserve_temperature": 0.08,
            "fatigue_energy_weight": 0.08,
            "gut_comfort": 0.55,
            "gut_overload_energy_weight": 0.08,
            "effort_energy_rate": 0.0042,
            "effort_extra_weight": 0.25,
            "reward_per_energy": 12.0,
            "max_interval_seconds": 2.0,
        },
    },
    "features": [
        "bias",
        "two_energy_minus_one",
        "two_gut_minus_one",
        "two_fatigue_minus_one",
    ],
    "slots": 128,
    "horizon_ticks": 10,
    "return_scale": 0.01,
    "learning_rate": 0.05,
    "weight_norm_limit": 4.0,
    "logit_gain": 0.35,
    "learning_default_enabled": True,
}
PREDICTOR_ORDER = (
    "input.mean",
    "input.scale",
    "target.mean",
    "target.scale",
    "residual.scale",
) + tuple(
    f"member.{member}.{part}"
    for member in range(3)
    for part in (
        "layer0.weight",
        "layer0.bias",
        "layer1.weight",
        "layer1.bias",
        "output.weight",
        "output.bias",
    )
)
PREDICTOR_ENCODER_NAMES = (
    "visual.peripheral.first.weight",
    "visual.peripheral.first.bias",
    "visual.peripheral.second.weight",
    "visual.peripheral.second.bias",
    "visual.foveal.first.weight",
    "visual.foveal.first.bias",
    "visual.foveal.second.weight",
    "visual.foveal.second.bias",
    "visual.peripheral_projection.0.weight",
    "visual.peripheral_projection.0.bias",
    "visual.foveal_projection.0.weight",
    "visual.foveal_projection.0.bias",
    "body.0.weight",
    "body.0.bias",
    "goal_encoder.0.weight",
    "goal_encoder.0.bias",
    "goal_encoder.2.weight",
    "goal_encoder.2.bias",
)
PREDICTOR_SHAPES = {
    "input.mean": (1426,),
    "input.scale": (1426,),
    "target.mean": (262,),
    "target.scale": (262,),
    "residual.scale": (262,),
}
for _member in range(3):
    PREDICTOR_SHAPES.update(
        {
            f"member.{_member}.layer0.weight": (256, 1426),
            f"member.{_member}.layer0.bias": (256,),
            f"member.{_member}.layer1.weight": (256, 256),
            f"member.{_member}.layer1.bias": (256,),
            f"member.{_member}.output.weight": (262, 256),
            f"member.{_member}.output.bias": (262,),
        }
    )
MANAGER_ORDER = (
    "manager.query.0.weight",
    "manager.query.0.bias",
    "manager.query.2.weight",
    "manager.query.2.bias",
    "manager.query_gain",
)
RICH_DEVELOPMENTAL_ORDER = (
    "normalizer.mean",
    "normalizer.scale",
    "model.visual.peripheral.first.weight",
    "model.visual.peripheral.first.bias",
    "model.visual.peripheral.second.weight",
    "model.visual.peripheral.second.bias",
    "model.visual.foveal.first.weight",
    "model.visual.foveal.first.bias",
    "model.visual.foveal.second.weight",
    "model.visual.foveal.second.bias",
    "model.visual.peripheral_projection.0.weight",
    "model.visual.peripheral_projection.0.bias",
    "model.visual.foveal_projection.0.weight",
    "model.visual.foveal_projection.0.bias",
    "model.body.0.weight",
    "model.body.0.bias",
    "model.goal_encoder.0.weight",
    "model.goal_encoder.0.bias",
    "model.goal_encoder.2.weight",
    "model.goal_encoder.2.bias",
    "model.observation_projection.0.weight",
    "model.observation_projection.0.bias",
    "model.history.weight_ih_l0",
    "model.history.weight_hh_l0",
    "model.history.bias_ih_l0",
    "model.history.bias_hh_l0",
    "model.policy_trunk.0.weight",
    "model.policy_trunk.0.bias",
    "model.signed_head.weight",
    "model.signed_head.bias",
    "model.active_head.weight",
    "model.active_head.bias",
    "model.positive_head.weight",
    "model.positive_head.bias",
) + MANAGER_ORDER
RICH_DEVELOPMENTAL_SHAPES = {
    "normalizer.mean": (4453,),
    "normalizer.scale": (4453,),
    "model.visual.peripheral.first.weight": (16, 4, 3, 3),
    "model.visual.peripheral.first.bias": (16,),
    "model.visual.peripheral.second.weight": (24, 16, 3, 3),
    "model.visual.peripheral.second.bias": (24,),
    "model.visual.foveal.first.weight": (16, 4, 3, 3),
    "model.visual.foveal.first.bias": (16,),
    "model.visual.foveal.second.weight": (24, 16, 3, 3),
    "model.visual.foveal.second.bias": (24,),
    "model.visual.peripheral_projection.0.weight": (64, 768),
    "model.visual.peripheral_projection.0.bias": (64,),
    "model.visual.foveal_projection.0.weight": (64, 2304),
    "model.visual.foveal_projection.0.bias": (64,),
    "model.body.0.weight": (128, 357),
    "model.body.0.bias": (128,),
    "model.goal_encoder.0.weight": (256, 1024),
    "model.goal_encoder.0.bias": (256,),
    "model.goal_encoder.2.weight": (64, 256),
    "model.goal_encoder.2.bias": (64,),
    "model.observation_projection.0.weight": (128, 265),
    "model.observation_projection.0.bias": (128,),
    "model.history.weight_ih_l0": (384, 128),
    "model.history.weight_hh_l0": (384, 128),
    "model.history.bias_ih_l0": (384,),
    "model.history.bias_hh_l0": (384,),
    "model.policy_trunk.0.weight": (256, 201),
    "model.policy_trunk.0.bias": (256,),
    "model.signed_head.weight": (260, 256),
    "model.signed_head.bias": (260,),
    "model.active_head.weight": (4, 256),
    "model.active_head.bias": (4,),
    "model.positive_head.weight": (128, 256),
    "model.positive_head.bias": (128,),
    "manager.query.0.weight": (128, 518),
    "manager.query.0.bias": (128,),
    "manager.query.2.weight": (64, 128),
    "manager.query.2.bias": (64,),
    "manager.query_gain": (1,),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _artifact_identity(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    clean = copy.deepcopy(metadata)
    clean.pop("artifact_sha256", None)
    digest = hashlib.sha256(_canonical(clean))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(_canonical(list(value.shape)))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _predictor_identity(metadata, arrays):
    clean = copy.deepcopy(metadata)
    clean.pop("artifact_identity", None)
    receipts = {
        name: {
            "dtype": np.ascontiguousarray(value).dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }
    return hashlib.sha256(
        _canonical({"metadata": clean, "arrays": receipts})
    ).hexdigest()


def _extension():
    try:
        return importlib.import_module("_cognitive_core")
    except ImportError as exc:
        raise RuntimeError("native cognitive core is unavailable") from exc


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "base64": base64.b64encode(array.tobytes()).decode(),
        }
    if isinstance(value, dict):
        return {name: _encode(item) for name, item in value.items()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"dtype", "shape", "base64"}:
        dtype = np.dtype(value["dtype"])
        shape = tuple(value["shape"])
        array = np.frombuffer(
            base64.b64decode(value["base64"], validate=True), dtype=dtype
        ).copy()
        if array.size != int(np.prod(shape, dtype=np.int64)):
            raise ValueError("encoded developmental array shape differs")
        return array.reshape(shape)
    if isinstance(value, dict):
        return {name: _decode(item) for name, item in value.items()}
    return value


class DevelopmentalResidentCohort:
    """Complete recurring resident controller with native private state."""

    def __init__(
        self,
        artifact: str | Path,
        batch_size: int,
        *,
        action_mode: str,
        goal_seed: int,
        action_seed: int,
    ) -> None:
        path = Path(artifact).expanduser().resolve()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            order = RICH_DEVELOPMENTAL_ORDER + tuple(
                "predictor." + x for x in PREDICTOR_ORDER
            )
            if (
                metadata.get("format") != DEVELOPMENTAL_FORMAT
                or metadata.get("version") != 3
                or metadata.get("execution")
                != "developmental-resident-native-rich-predictive-personal-goals-v3"
                or metadata.get("pack_order") != list(order)
                or set(archive.files) != set(order) | {"metadata"}
            ):
                raise ValueError("developmental resident artifact metadata differs")
            arrays = {}
            for name in order:
                value = np.asarray(archive[name])
                receipt = metadata.get("tensors", {}).get(name, {})
                if (
                    value.dtype != np.float32
                    or (
                        name in RICH_DEVELOPMENTAL_SHAPES
                        and value.shape != RICH_DEVELOPMENTAL_SHAPES[name]
                    )
                    or (
                        name.startswith("predictor.")
                        and value.shape != PREDICTOR_SHAPES[name[10:]]
                    )
                    or list(value.shape) != receipt.get("shape")
                    or receipt.get("dtype") != "float32"
                    or hashlib.sha256(value.tobytes()).hexdigest()
                    != receipt.get("sha256")
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"invalid developmental resident tensor: {name}")
                arrays[name] = value
        if _artifact_identity(metadata, arrays) != metadata.get("artifact_sha256"):
            raise ValueError("developmental resident artifact identity differs")
        if metadata.get("personal_goal_associations") != PERSONAL_GOAL_CONTRACT:
            raise ValueError("personal goal association contract differs")
        temporal = metadata.get("temporal_contract", {})
        if (
            temporal.get("observation_interval_seconds") != 0.05
            or temporal.get("manager_commit_ticks") != 10
        ):
            raise ValueError("developmental resident timing differs")
        refinement = metadata.get("consequence_refinement", {})
        law_bank = refinement.get("law_bank")
        if (
            not isinstance(law_bank, dict)
            or refinement.get("law_content_sha256")
            != hashlib.sha256(_canonical(law_bank)).hexdigest()
            or [
                refinement.get(x)
                for x in (
                    "candidates",
                    "tilt",
                    "learning_rate",
                    "error_decay",
                    "innovation_limit",
                )
            ]
            != [4, 0.5, 0.05, 0.99, 4.0]
        ):
            raise ValueError("developmental consequence refinement differs")
        predictor = metadata.get("predictor", {})
        predictor_order = tuple(predictor.get("pack_order", ()))
        predictor_metadata = predictor.get("metadata", {})
        predictor_identity = predictor.get("artifact_identity")
        goal_rms = predictor.get("goal_forecast_rms")
        predictor_arrays = {
            name: arrays["predictor." + name] for name in PREDICTOR_ORDER
        }
        predictor_arrays.update(
            {
                "encoder." + name: arrays["model." + name]
                for name in PREDICTOR_ENCODER_NAMES
            }
        )
        predictor_arrays.update(
            {
                "observation_normalizer.mean": arrays["normalizer.mean"],
                "observation_normalizer.scale": arrays["normalizer.scale"],
            }
        )
        if (
            predictor_order != tuple("predictor." + x for x in PREDICTOR_ORDER)
            or not isinstance(predictor_identity, str)
            or len(predictor_identity) != 64
            or _predictor_identity(predictor_metadata, predictor_arrays)
            != predictor_identity
            or not isinstance(goal_rms, (int, float))
            or not np.isfinite(goal_rms)
            or goal_rms < 1e-4
            or not isinstance(
                predictor_metadata.get("source", {}).get("frame_encoder_sha256"), str
            )
            or len(predictor_metadata["source"]["frame_encoder_sha256"]) != 64
        ):
            raise ValueError("predictive consequence identity differs")
        if action_mode not in {"sample", "map"}:
            raise ValueError("action_mode must be sample or map")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 256
        ):
            raise ValueError("batch_size must be an integer in 1..256")
        for name, seed in (("goal_seed", goal_seed), ("action_seed", action_seed)):
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 0 <= seed < 2**64
            ):
                raise ValueError(f"{name} must be an unsigned 64-bit integer")
        packed = np.ascontiguousarray(
            np.concatenate(
                [arrays[name].reshape(-1) for name in RICH_DEVELOPMENTAL_ORDER]
            ),
            dtype=np.float32,
        )
        predictor_packed = np.ascontiguousarray(
            np.concatenate([arrays[name].reshape(-1) for name in predictor_order]),
            dtype=np.float32,
        )
        self.artifact_path = path
        self.batch_size = batch_size
        self.action_mode = action_mode
        self.model_identity = {
            "format": DEVELOPMENTAL_FORMAT,
            "artifact_sha256": metadata["artifact_sha256"],
            "file_sha256": file_hash,
            "mode": "rich-achieved-goal",
            "execution": metadata["execution"],
            "personal_goal_contract_sha256": hashlib.sha256(
                _canonical(PERSONAL_GOAL_CONTRACT)
            ).hexdigest(),
            "finite_energy_objective_sha256": PERSONAL_GOAL_CONTRACT["objective"][
                "sha256"
            ],
        }
        trained = metadata.get("training_identity", {})
        self.neural_contract = {
            name: trained.get(name)
            for name in ("graph_sha256", "port_spec_sha256", "port_bundle_sha256")
        }
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in self.neural_contract.values()
        ):
            raise ValueError(
                "developmental artifact lacks its trained neural substrate"
            )
        self.observation_contract = copy.deepcopy(metadata.get("observation_contract"))
        if self.observation_contract != {
            "format": "chreatures-rich-sensorimotor-observation-v1",
            "observation_dim": 4453,
            "rich_body_dim": 4096,
            "rich_profile_sha256": "c71380718ba5535dbaebdeaf8aa2e88cc45cf218312a03e13507877f02a5554e",
            "rich_channel_names_sha256": "b4c6b328116d820143e16ee922ccffd7b950dbe008efc580ad93056e01349bfa",
            "observation_order": [
                "rich_body_v1_4096",
                "canonical_channels_351",
                "physiology_6",
            ],
            "source_sense_dim": 351,
            "physiology_dim": 6,
            "neural_readout_dim": 384,
            "previous_action_plus_oral_dim": 9,
        }:
            raise ValueError("developmental resident observation contract differs")
        self._native = _extension().DevelopmentalResidentCohort(
            batch_size,
            "rich-achieved-goal",
            action_mode,
            goal_seed,
            action_seed,
            packed,
            _canonical(law_bank).decode(),
            refinement["law_file_sha256"],
            refinement["learning_rate"],
            refinement["error_decay"],
            refinement["innovation_limit"],
            predictor_packed,
            predictor["goal_forecast_rms"],
        )

    def step(
        self,
        observations,
        neural,
        physiology,
        actual_previous_plus_oral,
        ticks,
        times,
        reset,
    ):
        result = self._native.step(
            np.ascontiguousarray(observations, dtype=np.float32),
            np.ascontiguousarray(neural, dtype=np.float32),
            np.ascontiguousarray(physiology, dtype=np.float32),
            np.ascontiguousarray(actual_previous_plus_oral, dtype=np.float32),
            np.ascontiguousarray(ticks, dtype=np.uint64),
            np.ascontiguousarray(times, dtype=np.float64),
            np.ascontiguousarray(reset, dtype=np.bool_),
        )
        result = {name: np.asarray(value) for name, value in result.items()}
        expected = {
            "proposed_action": (self.batch_size, 8),
            "oral_command": (self.batch_size,),
            "candidate_scores": (self.batch_size, 4),
            "candidate_out_of_domain": (self.batch_size, 4),
            "selected_candidate": (self.batch_size,),
            "selected_consequence_correction": (self.batch_size, 3),
            "personal_consequence_updates": (self.batch_size,),
            "forecast_progress": (self.batch_size, 4),
            "forecast_disagreement": (self.batch_size, 4),
            "forecast_input_clipped": (self.batch_size, 4),
            "forecast_tilt": (self.batch_size, 4),
            "forecast_goal_rms": (),
            "actual_previous_action": (self.batch_size, 8),
            "hidden": (self.batch_size, 128),
            "physiology": (self.batch_size, 6),
            "memory_inserted_slot": (self.batch_size,),
            "memory_count": (self.batch_size,),
            "goal_slot": (self.batch_size,),
            "goal_valid": (self.batch_size,),
            "goal_changed": (self.batch_size,),
            "goal_recorded_tick": (self.batch_size,),
            "goal_recorded_time": (self.batch_size,),
            "goal_generation": (self.batch_size,),
            "goal_remaining_ticks": (self.batch_size,),
            "goal_key": (self.batch_size, 64),
            "goal_window": (self.batch_size, 4, 4453),
            "personal_goal_selected_bias": (self.batch_size,),
            "personal_goal_prediction": (self.batch_size,),
            "personal_goal_last_reward": (self.batch_size,),
            "personal_goal_last_return": (self.batch_size,),
            "personal_goal_completed": (self.batch_size,),
            "personal_goal_attributed": (self.batch_size,),
            "personal_goal_learned": (self.batch_size,),
            "personal_goal_completed_total": (self.batch_size,),
            "personal_goal_learned_total": (self.batch_size,),
            "personal_goal_frozen_total": (self.batch_size,),
            "personal_goal_skipped_total": (self.batch_size,),
            "personal_goal_cancelled_total": (self.batch_size,),
            "personal_goal_learning_enabled": (),
        }
        if set(result) != set(expected):
            raise RuntimeError("native developmental result fields differ")
        for name, shape in expected.items():
            if result[name].shape != shape:
                raise RuntimeError(f"native developmental result shape differs: {name}")
            if (
                np.issubdtype(result[name].dtype, np.floating)
                and not np.isfinite(result[name]).all()
            ):
                raise RuntimeError(f"native developmental result is nonfinite: {name}")
        return result

    def observe_consequences(
        self,
        ticks,
        before_physiology,
        after_physiology,
        executed_actions_plus_oral,
        effort,
        *,
        dt: float,
    ):
        result = self._native.observe_consequences(
            np.ascontiguousarray(ticks, dtype=np.uint64),
            np.ascontiguousarray(before_physiology, dtype=np.float32),
            np.ascontiguousarray(after_physiology, dtype=np.float32),
            np.ascontiguousarray(executed_actions_plus_oral, dtype=np.float32),
            np.ascontiguousarray(effort, dtype=np.float32),
            float(dt),
        )
        result = {name: np.asarray(value) for name, value in result.items()}
        expected = {
            "reward": (self.batch_size,),
            "completed": (self.batch_size,),
            "summed_return": (self.batch_size,),
            "attributed": (self.batch_size,),
            "learned": (self.batch_size,),
            "completed_total": (self.batch_size,),
            "learned_total": (self.batch_size,),
            "frozen_total": (self.batch_size,),
            "skipped_total": (self.batch_size,),
            "cancelled_total": (self.batch_size,),
        }
        if set(result) != set(expected):
            raise RuntimeError("native personal goal receipt fields differ")
        for name, shape in expected.items():
            if result[name].shape != shape:
                raise RuntimeError(
                    f"native personal goal receipt shape differs: {name}"
                )
            if (
                np.issubdtype(result[name].dtype, np.floating)
                and not np.isfinite(result[name]).all()
            ):
                raise RuntimeError(f"native personal goal receipt is nonfinite: {name}")
        return result

    def set_personal_goal_learning(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean")
        self._native.set_personal_goal_learning(enabled)

    def snapshot_value(self) -> dict[str, Any]:
        return {
            "format": "chreatures-developmental-resident-rich-snapshot-v3",
            "version": 3,
            "model_identity": copy.deepcopy(self.model_identity),
            "batch_size": self.batch_size,
            "observation_contract": copy.deepcopy(self.observation_contract),
            "action_mode": self.action_mode,
            "native": _encode(dict(self._native.snapshot())),
        }

    @classmethod
    def restore_value(
        cls, value: dict[str, Any], artifact: str | Path
    ) -> DevelopmentalResidentCohort:
        if (
            not isinstance(value, dict)
            or value.get("format")
            != "chreatures-developmental-resident-rich-snapshot-v3"
            or value.get("version") != 3
        ):
            raise ValueError("unsupported developmental resident snapshot")
        instance = cls(
            artifact,
            int(value["batch_size"]),
            action_mode=value["action_mode"],
            goal_seed=0,
            action_seed=0,
        )
        if (
            value.get("model_identity") != instance.model_identity
            or value.get("observation_contract") != instance.observation_contract
        ):
            raise ValueError("developmental resident snapshot model identity differs")
        native = _decode(value["native"])
        instance._native.restore(native)
        return instance


__all__ = ["DevelopmentalResidentCohort"]
