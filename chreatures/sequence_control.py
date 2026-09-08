"""Immutable tensors for the native CNS-only resident controller.

This module is a serialization boundary only.  The neural service creates the
authenticated CNS latent, Rust owns recurrent inference and private state, and
Torch may optimize archived tensors.  No raw sense or physiology vector is
accepted by this contract.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from .cns_adapter_contract import (
    CONTROLLER_FORMAT,
    FORMAT as CNS_SERVICE_FORMAT,
    LATENT_DIM as SERVICE_LATENT_DIM,
    SENSORY_DIM,
    MOTOR_DIM as SERVICE_MOTOR_DIM,
    CONTEXT_DIM as SERVICE_CONTEXT_DIM,
)

FORMAT = "chreatures-cns-context-sequence-control-v1"
CONTRACT_VERSION = 1
CNS_LATENT_DIM = 512
ACTION_DIM = 12  # Abstract signed CNS context; never physical actuator commands.
CONTEXT_POLICY_VERSION = "signed-context12-v1"
CONTEXT_NAMES = tuple(f"cns.context.{i}" for i in range(ACTION_DIM))
CORE_HIDDEN_DIM = 256
GOAL_DIM = 128
CONTEXT_DIM = CNS_LATENT_DIM + CORE_HIDDEN_DIM + GOAL_DIM + ACTION_DIM
STATE_DIM = CONTEXT_DIM + 1
CANDIDATES = 8
CANDIDATE_DIM = 234

CORE_SHAPES = {
    "core.weight_ih": (768, 524),
    "core.weight_hh": (768, 256),
    "core.bias_ih": (768,),
    "core.bias_hh": (768,),
    "goal_encoder.weight": (128, 768),
    "goal_encoder.bias": (128,),
    "proposal_hidden.weight": (256, 396),
    "proposal_hidden.bias": (256,),
    "proposal_out.weight": (48, 256),
    "proposal_out.bias": (48,),
}
CORE_ORDER = tuple(CORE_SHAPES)

PREDICTOR_SHAPES: dict[str, tuple[int, ...]] = {}
for _member in range(3):
    PREDICTOR_SHAPES.update(
        {
            f"predictor.{_member}.context.weight": (256, 908),
            f"predictor.{_member}.context.bias": (256,),
            f"predictor.{_member}.transition.weight_ih": (768, 12),
            f"predictor.{_member}.transition.weight_hh": (768, 256),
            f"predictor.{_member}.transition.bias_ih": (768,),
            f"predictor.{_member}.transition.bias_hh": (768,),
            f"predictor.{_member}.delta.weight": (512, 256),
            f"predictor.{_member}.delta.bias": (512,),
        }
    )
PREDICTOR_ORDER = tuple(PREDICTOR_SHAPES)

_HEAD_LAYERS = (
    ("state_encoder", 256, 909),
    ("candidate_encoder", 128, 234),
    ("selector_hidden", 128, 384),
    ("selector_out", 1, 128),
    ("hazard_hidden", 128, 512),
    ("hazard_out", 1, 128),
    ("value_hidden", 128, 512),
    ("value_out", 1, 128),
)
SHAPES = {
    f"{name}.{part}": (out, inputs) if part == "weight" else (out,)
    for name, out, inputs in _HEAD_LAYERS
    for part in ("weight", "bias")
}
ORDER = tuple(SHAPES)
EMBEDDED_ORDER = tuple("sequence_control." + name for name in ORDER)
RESIDENT_SHAPES = {**CORE_SHAPES, **PREDICTOR_SHAPES,
                   **dict(zip(EMBEDDED_ORDER, SHAPES.values(), strict=True))}
RESIDENT_ORDER = CORE_ORDER + PREDICTOR_ORDER + EMBEDDED_ORDER

CNS_DEPENDENCY_KEYS = (
    "format",
    "adapter_sha256",
    "service_artifact_sha256",
    "graph_sha256",
    "atlas_sha256",
    "anatomy_sha256",
    "readout_mask_sha256",
    "sensory_dim",
    "latent_dim",
    "context_dim",
    "motor_dim",
)
CONTROL_DEPENDENCY_KEYS = CNS_DEPENDENCY_KEYS + (
    "controller_input",
    "organism_interface_sha256",
    "source_revision",
    "core_packed_sha256",
    "predictor_packed_sha256",
)

CONTRACT = {
    "format": "chreatures-cns-context-sequence-control-contract-v1",
    "version": CONTRACT_VERSION,
    "controller_input": CONTROLLER_FORMAT,
    "ingress": {
        "cns_latent": [CNS_LATENT_DIM, "float32"],
        "previous_delivered_context": [ACTION_DIM, "float32"],
        "ticks": [1, "uint64"],
        "reset": [1, "bool"],
    },
    "private_core": {
        "gru_input": CNS_LATENT_DIM + ACTION_DIM,
        "gru_hidden": CORE_HIDDEN_DIM,
        "goal": GOAL_DIM,
        "goal_formula": "l2_normalize(tanh(linear(concat(cns_latent,hidden))),eps=1e-8)",
        "context": ["cns_latent512", "hidden256", "goal128", "previous_delivered_context12"],
        "context_dim": CONTEXT_DIM,
    },
    "proposal": {
        "local_candidates": 4,
        "local_duration_ticks": 1,
        "maximum_acquired_ticks": 8,
        "context_dim": ACTION_DIM,
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "signed_output": "tanh all12",
        "physical_actuation": False,
    },
    "predictor": {"members": 3, "target": "cns_latent_delta512"},
    "sequence_control": {
        "state_dim": STATE_DIM,
        "candidates": CANDIDATES,
        "candidate_dim": CANDIDATE_DIM,
        "layers": [[name, out, inputs] for name, out, inputs in _HEAD_LAYERS],
        "likelihood": "masked_bernoulli_termination_plus_conditional_categorical",
    },
    "weight_orientation": "torch-out-in",
    "hidden_activation": "tanh",
    "receipt": "explicit acknowledged context delivered into CNS recurrence and tick only",
    "forbidden_ingress": [
        "raw_visual", "raw_sensory", "raw_physiology", "neural_readouts384",
        "world_position", "object_kind", "reward", "outcome",
    ],
    "service_format": CNS_SERVICE_FORMAT,
}


def controller_interface() -> dict[str, Any]:
    return {
        "format": "chreatures-cns-context-organism-interface-v1",
        "controller_input": CONTROLLER_FORMAT,
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "cns_latent": {"dtype": "float32", "dimension": CNS_LATENT_DIM},
        "previous_delivered_context": {
            "dtype": "float32", "dimension": ACTION_DIM, "order": list(CONTEXT_NAMES), "bounds": [-1, 1],
        },
        "ticks": "uint64", "reset": "bool",
        "output_context": {
            "dtype": "float32", "dimension": ACTION_DIM, "order": list(CONTEXT_NAMES), "bounds": [-1, 1],
        },
        "receipt": ["ticks", "delivered_context"],
        "delivery_boundary": "context entered CNS recurrence",
        "physical_actuation": False,
    }


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


CONTRACT_SHA256 = hashlib.sha256(canonical(CONTRACT)).hexdigest()


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def valid_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_dependencies(
    value: Mapping[str, Any], *, control: bool = True
) -> dict[str, str]:
    expected = CONTROL_DEPENDENCY_KEYS if control else CNS_DEPENDENCY_KEYS
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("CNS-only controller dependency fields differ")
    result = dict(value)
    for name in expected:
        if name == "format":
            if result[name] != CNS_SERVICE_FORMAT:
                raise ValueError("CNS service format differs")
        elif name == "controller_input":
            if result[name] != CONTROLLER_FORMAT:
                raise ValueError("CNS controller input format differs")
        elif name == "source_revision":
            if not valid_revision(result[name]):
                raise ValueError("controller source revision must be a full Git identity")
        elif name == "sensory_dim":
            if result[name] != SENSORY_DIM:
                raise ValueError("CNS service sensory dimension differs")
        elif name == "latent_dim":
            if result[name] != SERVICE_LATENT_DIM:
                raise ValueError("CNS service latent dimension differs")
        elif name == "context_dim":
            if result[name] != SERVICE_CONTEXT_DIM:
                raise ValueError("CNS service context dimension differs")
        elif name == "motor_dim":
            if result[name] != SERVICE_MOTOR_DIM:
                raise ValueError("CNS service motor dimension differs")
        elif not valid_sha256(result[name]):
            raise ValueError(f"controller dependency requires SHA-256: {name}")
    return result


def packed_sha256(arrays: Mapping[str, np.ndarray], order: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in order:
        value = np.ascontiguousarray(arrays[name], dtype="<f4")
        digest.update(value.tobytes())
    return digest.hexdigest()


def inherited_dependencies(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact immutable dependencies for a swappable head bundle."""
    cns = validate_dependencies(metadata.get("cns_service", {}), control=False)
    arrays = metadata.get("controller_components", {})
    result = {
        **cns,
        "controller_input": metadata.get("controller_input"),
        "organism_interface_sha256": metadata.get("organism_interface_sha256"),
        "source_revision": metadata.get("source_revision"),
        "core_packed_sha256": arrays.get("core_packed_sha256"),
        "predictor_packed_sha256": arrays.get("predictor_packed_sha256"),
    }
    return validate_dependencies(result)


def tensor_receipts(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "dtype": "float32",
            "shape": list(arrays[name].shape),
            "sha256": hashlib.sha256(arrays[name].tobytes()).hexdigest(),
        }
        for name in ORDER
    }


def artifact_identity(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    clean = copy.deepcopy(dict(metadata))
    clean.pop("artifact_sha256", None)
    digest = hashlib.sha256(canonical(clean))
    for name in ORDER:
        value = arrays[name]
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(canonical(list(value.shape)))
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ControlArtifact:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    path: Path | None = None
    file_sha256: str | None = None

    @property
    def sha256(self) -> str:
        return self.metadata["artifact_sha256"]

    @property
    def version(self) -> int:
        return self.metadata["version"]

    def packed(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.concatenate([self.arrays[name].reshape(-1) for name in ORDER]),
            dtype=np.float32,
        )


def validate_control_artifact(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> ControlArtifact:
    value = copy.deepcopy(dict(metadata))
    expected_metadata = {
        "format", "contract_sha256", "version", "parent_sha256", "pack_order",
        "dependencies", "provenance", "tensors", "artifact_sha256",
    }
    if (
        set(value) != expected_metadata
        or value.get("format") != FORMAT
        or value.get("contract_sha256") != CONTRACT_SHA256
    ):
        raise ValueError("sequence-control artifact contract differs")
    version = value.get("version")
    parent = value.get("parent_sha256")
    if type(version) is not int or not 0 <= version < 2**64:
        raise ValueError("sequence-control version must be an unsigned integer")
    if (version == 0 and parent is not None) or (version > 0 and not valid_sha256(parent)):
        raise ValueError("sequence-control parent/version relationship differs")
    if value.get("pack_order") != list(ORDER) or set(arrays) != set(ORDER):
        raise ValueError("sequence-control tensor set/order differs")
    value["dependencies"] = validate_dependencies(value.get("dependencies", {}))
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("training_status") not in {
        "initialized-untrained", "trained"
    }:
        raise ValueError("sequence-control training status is required")
    checked: dict[str, np.ndarray] = {}
    for name in ORDER:
        array = np.asarray(arrays[name])
        if array.dtype != np.float32 or array.shape != SHAPES[name] or not np.isfinite(array).all():
            raise ValueError(f"sequence-control tensor differs: {name}")
        checked[name] = np.ascontiguousarray(array).copy()
    if value.get("tensors") != tensor_receipts(checked):
        raise ValueError("sequence-control tensor receipt differs")
    if artifact_identity(value, checked) != value.get("artifact_sha256"):
        raise ValueError("sequence-control artifact identity differs")
    for array in checked.values():
        array.setflags(write=False)
    return ControlArtifact(value, checked)


def load_control_artifact(path: str | Path) -> ControlArtifact:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", *ORDER}:
            raise ValueError("sequence-control archive tensor set differs")
        result = validate_control_artifact(
            json.loads(str(archive["metadata"].item())),
            {name: archive[name] for name in ORDER},
        )
    with source.open("rb") as stream:
        file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return ControlArtifact(result.metadata, result.arrays, source, file_hash)


def write_control_artifact(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    version: int,
    parent_sha256: str | None,
    dependencies: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> ControlArtifact:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    converted = {
        name: np.ascontiguousarray(arrays[name], dtype=np.float32) for name in ORDER
    } if set(arrays) == set(ORDER) else {}
    if not converted:
        raise ValueError("sequence-control write tensor set differs")
    metadata = {
        "format": FORMAT,
        "contract_sha256": CONTRACT_SHA256,
        "version": version,
        "parent_sha256": parent_sha256,
        "pack_order": list(ORDER),
        "dependencies": copy.deepcopy(dict(dependencies)),
        "provenance": copy.deepcopy(dict(provenance)),
        "tensors": tensor_receipts(converted),
    }
    metadata["artifact_sha256"] = artifact_identity(metadata, converted)
    checked = validate_control_artifact(metadata, converted)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(canonical(checked.metadata).decode()),
                **checked.arrays,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        os.chmod(target, 0o444)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return load_control_artifact(target)


def _xavier(generator: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    limit = math.sqrt(6.0 / (shape[0] + shape[1]))
    return generator.uniform(-limit, limit, size=shape).astype(np.float32)


def initialize_controller_arrays(
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create an explicit archived untrained controller; the seed is provenance."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("controller seed must be an unsigned 64-bit integer")
    generator = np.random.Generator(np.random.PCG64(seed))
    all_shapes = {**CORE_SHAPES, **PREDICTOR_SHAPES, **SHAPES}
    arrays: dict[str, np.ndarray] = {}
    zero_weight = {
        *(f"predictor.{member}.delta.weight" for member in range(3)),
        "selector_out.weight",
        "hazard_out.weight",
        "value_out.weight",
    }
    for name, shape in all_shapes.items():
        if name.endswith(".bias") or "bias_i" in name or "bias_h" in name:
            arrays[name] = np.zeros(shape, dtype=np.float32)
        elif name in zero_weight:
            arrays[name] = np.zeros(shape, dtype=np.float32)
        elif name == "proposal_out.weight":
            arrays[name] = np.float32(0.3) * _xavier(generator, shape)
        else:
            arrays[name] = _xavier(generator, shape)
    arrays["hazard_out.bias"][0] = np.float32(-math.log(7.0))
    return (
        {name: arrays[name] for name in CORE_ORDER},
        {name: arrays[name] for name in PREDICTOR_ORDER},
        {name: arrays[name] for name in ORDER},
    )


__all__ = [
    "ACTION_DIM", "CANDIDATES", "CANDIDATE_DIM", "CNS_DEPENDENCY_KEYS",
    "CNS_LATENT_DIM", "CONTEXT_DIM", "CONTRACT", "CONTRACT_SHA256",
    "CONTROL_DEPENDENCY_KEYS", "CORE_HIDDEN_DIM", "CORE_ORDER", "CORE_SHAPES",
    "ControlArtifact", "EMBEDDED_ORDER", "FORMAT", "GOAL_DIM", "ORDER",
    "PREDICTOR_ORDER", "PREDICTOR_SHAPES", "RESIDENT_ORDER", "RESIDENT_SHAPES",
    "SHAPES", "STATE_DIM", "artifact_identity", "canonical",
    "inherited_dependencies", "initialize_controller_arrays",
    "load_control_artifact", "packed_sha256", "valid_sha256",
    "validate_control_artifact", "validate_dependencies", "write_control_artifact",
]
