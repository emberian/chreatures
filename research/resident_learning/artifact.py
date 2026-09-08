"""Load and publish immutable trained resident artifacts in canonical order."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from chreatures.sequence_control import (
    CONTEXT_POLICY_VERSION, CORE_ORDER, EMBEDDED_ORDER, ORDER as CONTROL_ORDER, PREDICTOR_ORDER,
    RESIDENT_ORDER, RESIDENT_SHAPES, canonical, packed_sha256, write_control_artifact,
    controller_interface,
)

RESIDENT_FORMAT = "chreatures-native-cns-context-resident-population-v1"


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_parent(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", *RESIDENT_ORDER}:
            raise ValueError("parent resident tensor set differs")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: np.ascontiguousarray(archive[name], dtype=np.float32) for name in RESIDENT_ORDER}
    if (metadata.get("format") != RESIDENT_FORMAT
            or metadata.get("context_policy_version") != CONTEXT_POLICY_VERSION
            or metadata.get("controller_input") != controller_interface()):
        raise ValueError("parent is not a fresh signed context12 resident artifact")
    for name, value in arrays.items():
        if value.shape != RESIDENT_SHAPES[name] or not np.isfinite(value).all():
            raise ValueError(f"parent resident tensor differs: {name}")
    return metadata, arrays


def _receipts(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: {"dtype": "float32", "shape": list(arrays[name].shape), "sha256": hashlib.sha256(arrays[name].tobytes()).hexdigest()} for name in RESIDENT_ORDER}


def _identity(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    clean = copy.deepcopy(dict(metadata)); clean.pop("artifact_sha256", None)
    digest = hashlib.sha256(canonical(clean))
    for name in RESIDENT_ORDER:
        value = arrays[name]; digest.update(name.encode()); digest.update(value.dtype.str.encode()); digest.update(canonical(list(value.shape))); digest.update(value.tobytes())
    return digest.hexdigest()


def publish_trained(
    resident_output: Path,
    control_output: Path,
    parent_metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    episode_identities: list[dict[str, str]],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    resident_output, control_output = resident_output.resolve(), control_output.resolve()
    if resident_output.exists() or control_output.exists():
        raise FileExistsError("trained resident outputs must be new paths")
    values = {name: np.ascontiguousarray(arrays[name], dtype=np.float32) for name in RESIDENT_ORDER}
    for name, value in values.items():
        if value.shape != RESIDENT_SHAPES[name] or not np.isfinite(value).all():
            raise ValueError(f"trained resident tensor differs: {name}")
    core_hash, predictor_hash = packed_sha256(values, CORE_ORDER), packed_sha256(values, PREDICTOR_ORDER)
    old_control = parent_metadata["controller_components"]["sequence_control"]
    dependencies = {
        **parent_metadata["cns_service"],
        "controller_input": parent_metadata["controller_input"],
        "organism_interface_sha256": parent_metadata["organism_interface_sha256"],
        "source_revision": parent_metadata["source_revision"],
        "core_packed_sha256": core_hash,
        "predictor_packed_sha256": predictor_hash,
        "context_policy_version": CONTEXT_POLICY_VERSION,
    }
    head_arrays = {name: values[embedded] for name, embedded in zip(CONTROL_ORDER, EMBEDDED_ORDER, strict=True)}
    control = write_control_artifact(
        control_output, head_arrays, version=int(old_control["version"]) + 1,
        parent_sha256=old_control["artifact_sha256"], dependencies=dependencies,
        provenance={"training_status": "trained", "operation": "joint-cns-resident-closed-loop-fit", "episodes": episode_identities, **dict(training)},
    )
    metadata = copy.deepcopy(dict(parent_metadata))
    metadata["format"] = RESIDENT_FORMAT
    metadata["context_policy_version"] = CONTEXT_POLICY_VERSION
    metadata["controller_input"] = controller_interface()
    metadata["controller_components"] = {
        "core_pack_order": list(CORE_ORDER), "core_packed_sha256": core_hash,
        "predictor_pack_order": list(PREDICTOR_ORDER), "predictor_packed_sha256": predictor_hash,
        "sequence_control": copy.deepcopy(control.metadata),
    }
    metadata["initialization"] = {
        "training_status": "trained", "competence_claim": None,
        "source_policy": None,
        "parent_artifact_sha256": parent_metadata["artifact_sha256"],
        "episodes": episode_identities, "training": dict(training),
    }
    metadata["tensors"] = _receipts(values)
    metadata["artifact_sha256"] = _identity(metadata, values)
    resident_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resident_output.with_name(f".{resident_output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, metadata=np.asarray(canonical(metadata).decode()), **values)
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, resident_output)
        os.chmod(resident_output, 0o444)
        directory = os.open(resident_output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "resident": {"path": str(resident_output), "file_sha256": file_sha256(resident_output), "artifact_sha256": metadata["artifact_sha256"]},
        "sequence_control": {"path": str(control.path), "file_sha256": control.file_sha256, "artifact_sha256": control.sha256, "version": control.version},
    }
