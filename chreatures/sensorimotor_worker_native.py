"""Strict Torch-free host boundary for the native CNS-only resident."""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .organism_interface import ACTION_DIM, RECTIFIED_AXES, SIGNED_AXES
from .resident_contract import (
    CONTROLLER_INPUT_FORMAT,
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
    NATIVE_SNAPSHOT_FORMAT,
    NATIVE_SNAPSHOT_VERSION,
)
from .sequence_control import (
    CNS_LATENT_DIM,
    CONTRACT_SHA256 as CONTROL_CONTRACT_SHA256,
    CORE_ORDER,
    EMBEDDED_ORDER,
    ORDER as CONTROL_ORDER,
    PREDICTOR_ORDER,
    RESIDENT_ORDER,
    RESIDENT_SHAPES,
    ControlArtifact,
    canonical,
    inherited_dependencies,
    load_control_artifact,
    packed_sha256,
    validate_control_artifact,
    validate_dependencies,
)

DEVELOPMENTAL_FORMAT = NATIVE_POPULATION_FORMAT
NATIVE_RESULT_FORMAT = "chreatures-cns-resident-native-v10"


def _extension():
    try:
        return importlib.import_module("_cognitive_core")
    except ImportError as exc:
        raise RuntimeError("native cognitive core is unavailable") from exc


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact_identity(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    clean = copy.deepcopy(dict(metadata))
    clean.pop("artifact_sha256", None)
    digest = hashlib.sha256(canonical(clean))
    for name in RESIDENT_ORDER:
        value = arrays[name]
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(canonical(list(value.shape)))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _packed(arrays: Mapping[str, np.ndarray], order: tuple[str, ...]) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([arrays[name].reshape(-1) for name in order]),
        dtype=np.float32,
    )


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "base64": base64.b64encode(array.tobytes()).decode(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {name: _encode(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"dtype", "shape", "base64"}:
        dtype = np.dtype(value["dtype"])
        shape = tuple(value["shape"])
        array = np.frombuffer(
            base64.b64decode(value["base64"], validate=True), dtype=dtype
        ).copy()
        if array.size != int(np.prod(shape, dtype=np.int64)):
            raise ValueError("encoded CNS resident array shape differs")
        return array.reshape(shape)
    if isinstance(value, dict):
        return {name: _decode(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _load_resident(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], ControlArtifact]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", *RESIDENT_ORDER}:
            raise ValueError("CNS resident artifact tensor set differs")
        metadata = json.loads(str(archive["metadata"].item()))
        expected_metadata = {
            "format", "version", "execution", "controller_input",
            "source_revision", "organism_interface", "organism_interface_sha256",
            "cns_service", "cns_service_provenance", "controller_components",
            "initialization", "pack_order", "tensors", "artifact_sha256",
        }
        if (
            set(metadata) != expected_metadata
            or
            metadata.get("format") != NATIVE_POPULATION_FORMAT
            or metadata.get("version") != NATIVE_POPULATION_VERSION
            or metadata.get("execution") != NATIVE_EXECUTION
            or metadata.get("controller_input") != CONTROLLER_INPUT_FORMAT
            or metadata.get("pack_order") != list(RESIDENT_ORDER)
        ):
            raise ValueError("CNS resident artifact metadata differs")
        arrays: dict[str, np.ndarray] = {}
        for name in RESIDENT_ORDER:
            value = np.asarray(archive[name])
            receipt = metadata.get("tensors", {}).get(name, {})
            if (
                value.dtype != np.float32
                or value.shape != RESIDENT_SHAPES[name]
                or not value.flags.c_contiguous
                or not np.isfinite(value).all()
                or receipt
                != {
                    "dtype": "float32",
                    "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
            ):
                raise ValueError(f"CNS resident tensor differs: {name}")
            arrays[name] = np.ascontiguousarray(value).copy()
    if _artifact_identity(metadata, arrays) != metadata.get("artifact_sha256"):
        raise ValueError("CNS resident artifact identity differs")
    initialization = metadata.get("initialization", {})
    service_provenance = metadata.get("cns_service_provenance", {})
    if (
        initialization.get("training_status") != "initialized-untrained"
        or initialization.get("competence_claim") is not None
        or service_provenance.get("training_status") not in {
            "initialized-untrained", "trained"
        }
    ):
        raise ValueError("CNS resident initialization provenance differs")
    cns = validate_dependencies(metadata.get("cns_service", {}), control=False)
    interface = metadata.get("organism_interface")
    if (
        not isinstance(interface, dict)
        or hashlib.sha256(canonical(interface)).hexdigest()
        != metadata.get("organism_interface_sha256")
    ):
        raise ValueError("CNS resident organism interface identity differs")
    components = metadata.get("controller_components", {})
    if (
        components.get("core_pack_order") != list(CORE_ORDER)
        or components.get("predictor_pack_order") != list(PREDICTOR_ORDER)
        or components.get("core_packed_sha256") != packed_sha256(arrays, CORE_ORDER)
        or components.get("predictor_packed_sha256")
        != packed_sha256(arrays, PREDICTOR_ORDER)
    ):
        raise ValueError("CNS resident component identity differs")
    embedded = validate_control_artifact(
        components.get("sequence_control", {}),
        {
            name: arrays[embedded_name]
            for name, embedded_name in zip(CONTROL_ORDER, EMBEDDED_ORDER, strict=True)
        },
    )
    if embedded.metadata["dependencies"] != inherited_dependencies(metadata):
        raise ValueError("embedded sequence-control dependencies differ")
    return metadata, arrays, embedded


class DevelopmentalResidentCohort:
    """One current recurrent controller whose only sensory input is CNS Z512."""

    def __init__(
        self,
        artifact: str | Path,
        batch_size: int,
        *,
        action_mode: str,
        action_seed: int,
        suffix_seed: int,
        sequence_control: ControlArtifact | None = None,
        research_training: bool = False,
    ) -> None:
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("CNS resident batch size must be in 1..4096")
        if action_mode not in {"sample", "map"}:
            raise ValueError("action_mode must be sample or map")
        for name, seed in (("action_seed", action_seed), ("suffix_seed", suffix_seed)):
            if type(seed) is not int or not 0 <= seed < 2**64:
                raise ValueError(f"{name} must be an unsigned 64-bit integer")
        if type(research_training) is not bool:
            raise TypeError("research_training must be boolean")
        path = Path(artifact).expanduser().resolve()
        metadata, arrays, embedded = _load_resident(path)
        if sequence_control is None:
            control = embedded
        elif isinstance(sequence_control, ControlArtifact):
            control = validate_control_artifact(
                sequence_control.metadata, sequence_control.arrays
            )
        else:
            raise TypeError("sequence_control must be an authenticated ControlArtifact")
        if control.metadata["dependencies"] != inherited_dependencies(metadata):
            raise ValueError("sequence-control replacement dependencies differ")

        self.artifact_path = str(path)
        self.batch_size = batch_size
        self.action_mode = action_mode
        self._research_training = research_training
        self._control = control
        self._arrays = arrays
        self.model_identity = {
            "format": metadata["format"],
            "version": metadata["version"],
            "execution": metadata["execution"],
            "artifact_sha256": metadata["artifact_sha256"],
            "file_sha256": _file_sha256(path),
            "source_revision": metadata["source_revision"],
            "controller_input": metadata["controller_input"],
            "training_status": metadata["initialization"]["training_status"],
            "cns_service": copy.deepcopy(metadata["cns_service"]),
            "controller_components": {
                "core_packed_sha256": metadata["controller_components"]["core_packed_sha256"],
                "predictor_packed_sha256": metadata["controller_components"]["predictor_packed_sha256"],
            },
            "sequence_control": self._control_identity(control),
        }
        self.neural_contract = {
            **copy.deepcopy(metadata["cns_service"]),
        }
        self._native = _extension().DevelopmentalResidentCohort(
            batch_size,
            action_mode,
            action_seed,
            suffix_seed,
            _packed(arrays, CORE_ORDER),
            metadata["controller_components"]["core_packed_sha256"],
            _packed(arrays, PREDICTOR_ORDER),
            metadata["controller_components"]["predictor_packed_sha256"],
            control.packed(),
            control.version,
            control.sha256,
            research_training,
        )

    @staticmethod
    def _control_identity(control: ControlArtifact) -> dict[str, Any]:
        return {
            "format": control.metadata["format"],
            "contract_sha256": CONTROL_CONTRACT_SHA256,
            "version": control.version,
            "artifact_sha256": control.sha256,
            "dependencies": copy.deepcopy(control.metadata["dependencies"]),
        }

    def replace_sequence_control(
        self,
        path: str | Path,
        *,
        expected_version: int,
        expected_sha256: str,
    ) -> dict[str, Any]:
        if not self._research_training:
            raise RuntimeError("ordinary resident cohorts reject controller replacement")
        if (expected_version, expected_sha256) != (
            self._control.version,
            self._control.sha256,
        ):
            raise ValueError("sequence-control expected identity is stale")
        replacement = load_control_artifact(path)
        if (
            replacement.version != expected_version + 1
            or replacement.metadata["parent_sha256"] != expected_sha256
            or replacement.metadata["dependencies"]
            != self._control.metadata["dependencies"]
        ):
            raise ValueError("sequence-control update lineage or dependencies differ")
        receipt = {
            "old_version": expected_version,
            "old_sha256": expected_sha256,
            "new_version": replacement.version,
            "new_sha256": replacement.sha256,
            "file_sha256": replacement.file_sha256,
            "preserves_private_state": True,
        }
        native_receipt = dict(self._native.replace_sequence_control(
            replacement.packed(),
            replacement.version,
            replacement.sha256,
            expected_version,
            expected_sha256,
        ))
        if native_receipt != {
            "old_version": expected_version,
            "old_sha256": expected_sha256,
            "new_version": replacement.version,
            "new_sha256": replacement.sha256,
        }:
            raise RuntimeError("native sequence-control commit receipt differs")
        self._control = replacement
        self.model_identity["sequence_control"] = self._control_identity(replacement)
        return receipt

    def expanded(
        self, additions: int, *, action_seed: int, suffix_seed: int
    ) -> "DevelopmentalResidentCohort":
        """Append cold residents while preserving every existing private prefix."""
        if type(additions) is not int or not 1 <= additions <= 4096 - self.batch_size:
            raise ValueError("resident additions are outside the cohort limit")
        for name, seed in (("action_seed", action_seed), ("suffix_seed", suffix_seed)):
            if type(seed) is not int or not 0 <= seed < 2**64:
                raise ValueError(f"{name} must be an unsigned 64-bit integer")
        native = self._native.expanded(additions, action_seed, suffix_seed)
        result = object.__new__(type(self))
        result.artifact_path = self.artifact_path
        result.batch_size = self.batch_size + additions
        result.action_mode = self.action_mode
        result._research_training = self._research_training
        result._control = self._control
        result._arrays = self._arrays
        result.model_identity = copy.deepcopy(self.model_identity)
        result.neural_contract = copy.deepcopy(self.neural_contract)
        result._native = native
        return result

    def step(self, cns_latent, previous_command, ticks, reset) -> dict[str, Any]:
        result = self._native.step(
            self._input(cns_latent, (self.batch_size, CNS_LATENT_DIM), np.float32, "cns_latent"),
            self._input(previous_command, (self.batch_size, ACTION_DIM), np.float32, "previous_command"),
            self._input(ticks, (self.batch_size,), np.uint64, "ticks"),
            self._input(reset, (self.batch_size,), np.bool_, "reset"),
        )
        return self._validate_decision(result)

    def preview_sequence_control(
        self, cns_latent, previous_command, ticks, reset
    ) -> dict[str, Any]:
        result = self._native.preview_sequence_control(
            self._input(cns_latent, (self.batch_size, CNS_LATENT_DIM), np.float32, "cns_latent"),
            self._input(previous_command, (self.batch_size, ACTION_DIM), np.float32, "previous_command"),
            self._input(ticks, (self.batch_size,), np.uint64, "ticks"),
            self._input(reset, (self.batch_size,), np.bool_, "reset"),
        )
        return self._validate_decision(result)

    def acknowledge(self, ticks, delivered_command) -> dict[str, Any]:
        result = self._native.acknowledge(
            self._input(ticks, (self.batch_size,), np.uint64, "ticks"),
            self._input(
                delivered_command,
                (self.batch_size, ACTION_DIM),
                np.float32,
                "delivered_command",
            ),
        )
        return self._validate_acknowledgement(result)

    def sequence_control_likelihood(
        self,
        state,
        proposal,
        active,
        proposal_mask,
        active_mask,
        hazard_decision,
        selected_candidate,
    ) -> dict[str, Any]:
        """Replay archived decisions without consuming resident or policy RNG."""
        raw = self._native.sequence_control_likelihood(
            self._input(state, (self.batch_size, 909), np.float32, "state"),
            self._input(
                proposal, (self.batch_size, 8, 234), np.float32, "proposal"
            ),
            self._input(active, (self.batch_size, 234), np.float32, "active"),
            self._input(
                proposal_mask,
                (self.batch_size, 8),
                np.bool_,
                "proposal_mask",
            ),
            self._input(
                active_mask, (self.batch_size,), np.bool_, "active_mask"
            ),
            self._input(
                hazard_decision,
                (self.batch_size,),
                np.bool_,
                "hazard_decision",
            ),
            self._input(
                selected_candidate,
                (self.batch_size,),
                np.int32,
                "selected_candidate",
            ),
        )
        scalar_names = {
            "format", "sequence_control_policy_version",
            "sequence_control_policy_sha256",
        }
        fields = {
            "sequence_control_hazard_logit": ((self.batch_size,), np.float32),
            "sequence_control_selector_logits": ((self.batch_size, 8), np.float32),
            "sequence_control_value": ((self.batch_size,), np.float32),
            "sequence_control_hazard_decision": ((self.batch_size,), np.bool_),
            "selected_candidate": ((self.batch_size,), np.int32),
            "sequence_control_hazard_mask": ((self.batch_size,), np.bool_),
            "sequence_control_selector_mask": ((self.batch_size,), np.bool_),
            "sequence_control_behavior_hazard_logp": ((self.batch_size,), np.float32),
            "sequence_control_behavior_selector_logp": ((self.batch_size,), np.float32),
            "sequence_control_behavior_logp": ((self.batch_size,), np.float32),
        }
        if not isinstance(raw, Mapping) or set(raw) != scalar_names | set(fields):
            raise RuntimeError("native sequence-control replay fields differ")
        result = {
            "format": raw["format"],
            "sequence_control_policy_version": raw["sequence_control_policy_version"],
            "sequence_control_policy_sha256": raw["sequence_control_policy_sha256"],
        }
        if (
            result["format"] != NATIVE_RESULT_FORMAT
            or int(result["sequence_control_policy_version"]) != self._control.version
            or result["sequence_control_policy_sha256"] != self._control.sha256
        ):
            raise RuntimeError("native sequence-control replay identity differs")
        for name, (shape, dtype) in fields.items():
            value = np.asarray(raw[name])
            if (
                value.shape != shape
                or value.dtype != dtype
                or (value.dtype.kind == "f" and not np.isfinite(value).all())
            ):
                raise RuntimeError(f"native sequence-control replay differs: {name}")
            result[name] = value
        return result

    @staticmethod
    def _input(value, shape, dtype, name) -> np.ndarray:
        array = np.ascontiguousarray(value, dtype=dtype)
        if array.shape != shape or (
            array.dtype.kind == "f" and not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} has invalid shape or values")
        return array

    def _validate_decision(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise RuntimeError("native CNS resident result is not a mapping")
        scalars = {
            "format": str,
            "sequence_control_policy_version": (int, np.integer),
            "sequence_control_policy_sha256": str,
        }
        arrays = {
            "proposed_command": ((self.batch_size, 12), np.float32),
            "cns_recurrent_state": ((self.batch_size, 256), np.float32),
            "latent_goal": ((self.batch_size, 128), np.float32),
            "current_cns_key": ((self.batch_size, 128), np.float32),
            "sequence_control_state": ((self.batch_size, 909), np.float32),
            "sequence_control_proposal": ((self.batch_size, 8, 234), np.float32),
            "sequence_control_active": ((self.batch_size, 234), np.float32),
            "sequence_control_proposal_mask": ((self.batch_size, 8), np.bool_),
            "sequence_control_active_mask": ((self.batch_size,), np.bool_),
            "sequence_control_hazard_logit": ((self.batch_size,), np.float32),
            "sequence_control_selector_logits": ((self.batch_size, 8), np.float32),
            "sequence_control_value": ((self.batch_size,), np.float32),
            "sequence_control_hazard_decision": ((self.batch_size,), np.bool_),
            "selected_candidate": ((self.batch_size,), np.int32),
            "sequence_control_hazard_mask": ((self.batch_size,), np.bool_),
            "sequence_control_selector_mask": ((self.batch_size,), np.bool_),
            "sequence_control_behavior_hazard_logp": ((self.batch_size,), np.float32),
            "sequence_control_behavior_selector_logp": ((self.batch_size,), np.float32),
            "sequence_control_behavior_logp": ((self.batch_size,), np.float32),
            "candidate_is_recalled_suffix": ((self.batch_size, 8), np.bool_),
            "candidate_suffix_slot": ((self.batch_size, 8), np.int32),
            "candidate_suffix_generation": ((self.batch_size, 8), np.uint64),
            "candidate_suffix_length": ((self.batch_size, 8), np.uint8),
            "active_source_slot": ((self.batch_size,), np.int32),
            "active_source_generation": ((self.batch_size,), np.uint64),
            "active_phase": ((self.batch_size,), np.uint8),
            "active_remaining": ((self.batch_size,), np.uint8),
            "motor_suffix_cancellation_totals": ((self.batch_size, 5), np.uint64),
            "motor_suffix_cancellation_reason": ((self.batch_size,), np.str_),
            "command_pending": ((self.batch_size,), np.bool_),
            "goal_origin_slot": ((self.batch_size,), np.int32),
            "goal_origin_generation": ((self.batch_size,), np.uint64),
            "goal_origin_tick": ((self.batch_size,), np.uint64),
            "goal_selected_tick": ((self.batch_size,), np.uint64),
            "memory_inserted_slot": ((self.batch_size,), np.int32),
            "memory_count": ((self.batch_size,), np.uint16),
        }
        if set(raw) != set(scalars) | set(arrays):
            raise RuntimeError("native CNS resident result fields differ")
        result: dict[str, Any] = {}
        for name, expected_type in scalars.items():
            value = raw[name]
            if not isinstance(value, expected_type):
                raise RuntimeError(f"native CNS resident scalar differs: {name}")
            result[name] = value
        for name, (shape, dtype) in arrays.items():
            value = np.asarray(raw[name])
            if value.shape != shape:
                raise RuntimeError(f"native CNS resident shape differs: {name}")
            if dtype is np.str_:
                if value.dtype.kind not in "US":
                    raise RuntimeError(f"native CNS resident dtype differs: {name}")
            elif value.dtype != dtype:
                raise RuntimeError(f"native CNS resident dtype differs: {name}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise RuntimeError(f"native CNS resident output is nonfinite: {name}")
            result[name] = value
        if result["format"] != NATIVE_RESULT_FORMAT:
            raise RuntimeError("native CNS resident result format differs")
        if (
            int(result["sequence_control_policy_version"]) != self._control.version
            or result["sequence_control_policy_sha256"] != self._control.sha256
        ):
            raise RuntimeError("native sequence-control identity differs")
        proposed = result["proposed_command"]
        if (
            np.any(np.abs(proposed[:, SIGNED_AXES]) > 1.000001)
            or np.any(proposed[:, RECTIFIED_AXES] < -1e-7)
            or np.any(proposed[:, RECTIFIED_AXES] > 1.000001)
        ):
            raise RuntimeError("native proposed command violates canonical bounds")
        return result

    def _validate_acknowledgement(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        scalar_names = {
            "format", "sequence_control_policy_version",
            "sequence_control_policy_sha256",
        }
        expected = {
            "acknowledged": ((self.batch_size,), np.bool_),
            "command_exact": ((self.batch_size,), np.bool_),
            "command_pending": ((self.batch_size,), np.bool_),
            "cns_outcome_pending": ((self.batch_size,), np.bool_),
            "motor_suffix_cancellation_totals": ((self.batch_size, 5), np.uint64),
            "motor_suffix_cancellation_reason": ((self.batch_size,), np.str_),
        }
        if not isinstance(raw, Mapping) or set(raw) != set(expected) | scalar_names:
            raise RuntimeError("native command acknowledgement fields differ")
        result = {
            "format": raw["format"],
            "sequence_control_policy_version": raw["sequence_control_policy_version"],
            "sequence_control_policy_sha256": raw["sequence_control_policy_sha256"],
        }
        if (
            not isinstance(result["format"], str)
            or not isinstance(result["sequence_control_policy_version"], (int, np.integer))
            or not isinstance(result["sequence_control_policy_sha256"], str)
        ):
            raise RuntimeError("native command acknowledgement scalars differ")
        for name, (shape, dtype) in expected.items():
            value = np.asarray(raw[name])
            if value.shape != shape or (
                dtype is np.str_ and value.dtype.kind not in "US"
            ) or (dtype is not np.str_ and value.dtype != dtype):
                raise RuntimeError(f"native command acknowledgement differs: {name}")
            result[name] = value
        if (
            result["format"] != NATIVE_RESULT_FORMAT
            or int(result["sequence_control_policy_version"]) != self._control.version
            or result["sequence_control_policy_sha256"] != self._control.sha256
            or not result["acknowledged"].all()
            or result["command_pending"].any()
            or not result["cns_outcome_pending"].all()
        ):
            raise RuntimeError("native command acknowledgement was incomplete")
        return result

    def snapshot_value(self) -> dict[str, Any]:
        return {
            "format": NATIVE_SNAPSHOT_FORMAT,
            "version": NATIVE_SNAPSHOT_VERSION,
            "model_identity": copy.deepcopy(self.model_identity),
            "batch_size": self.batch_size,
            "action_mode": self.action_mode,
            "research_training": self._research_training,
            "sequence_control": {
                "metadata": copy.deepcopy(self._control.metadata),
                "arrays": _encode(self._control.arrays),
            },
            "native": _encode(dict(self._native.snapshot())),
        }

    @classmethod
    def restore_value(
        cls, value: Mapping[str, Any], artifact: str | Path
    ) -> "DevelopmentalResidentCohort":
        if (
            not isinstance(value, Mapping)
            or value.get("format") != NATIVE_SNAPSHOT_FORMAT
            or value.get("version") != NATIVE_SNAPSHOT_VERSION
        ):
            raise ValueError("unsupported CNS resident snapshot")
        control_state = value.get("sequence_control", {})
        control = validate_control_artifact(
            control_state.get("metadata", {}),
            _decode(control_state.get("arrays", {})),
        )
        instance = cls(
            artifact,
            int(value["batch_size"]),
            action_mode=value["action_mode"],
            action_seed=0,
            suffix_seed=0,
            sequence_control=control,
            research_training=value["research_training"],
        )
        if value.get("model_identity") != instance.model_identity:
            raise ValueError("CNS resident snapshot model identity differs")
        instance._native.restore(_decode(value["native"]))
        return instance


__all__ = ["DevelopmentalResidentCohort", "NATIVE_RESULT_FORMAT"]
