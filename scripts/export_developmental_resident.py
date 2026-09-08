#!/usr/bin/env python3
"""Export a fresh, explicitly untrained CNS-only native resident artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.cns_adapter_contract import (
    CONTROLLER_FORMAT,
    load_service_artifact,
    service_identity,
)
from chreatures.resident_contract import (
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
)
from chreatures.sequence_control import (
    CNS_DEPENDENCY_KEYS,
    CORE_ORDER,
    controller_interface,
    EMBEDDED_ORDER,
    ORDER as CONTROL_ORDER,
    PREDICTOR_ORDER,
    RESIDENT_ORDER,
    canonical,
    initialize_controller_arrays,
    packed_sha256,
    valid_sha256,
    valid_revision,
    write_control_artifact,
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def current_revision() -> str:
    override = os.environ.get("CHREATURES_SOURCE_REVISION")
    if override is not None:
        if not valid_revision(override):
            raise ValueError(
                "CHREATURES_SOURCE_REVISION must be a full lowercase Git identity"
            )
        return override
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not valid_revision(revision):
        raise ValueError("source revision is not a full lowercase Git identity")
    return revision


def read_cns_service_identity(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    arrays, metadata = load_service_artifact(source)
    del arrays
    if metadata["training_status"] not in {"initialized-untrained", "trained"}:
        raise ValueError("CNS service training status differs")
    return {
        "identity": service_identity(metadata, file_sha256(source)),
        "training_status": metadata["training_status"],
        "path": str(source),
    }


def receipts(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "dtype": "float32",
            "shape": list(arrays[name].shape),
            "sha256": hashlib.sha256(arrays[name].tobytes()).hexdigest(),
        }
        for name in RESIDENT_ORDER
    }


def artifact_identity(
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cns-service", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-control-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    output = args.output.expanduser().resolve()
    head_output = args.sequence_control_output.expanduser().resolve()
    if output.exists() or head_output.exists():
        raise FileExistsError("resident and sequence-control outputs must be new paths")
    if output == head_output:
        raise ValueError("resident and sequence-control outputs must differ")

    revision = current_revision()
    service = read_cns_service_identity(args.cns_service)
    interface = controller_interface()
    interface_sha256 = hashlib.sha256(canonical(interface)).hexdigest()
    core, predictor, heads = initialize_controller_arrays(args.seed)
    core_hash = packed_sha256(core, CORE_ORDER)
    predictor_hash = packed_sha256(predictor, PREDICTOR_ORDER)
    # The V5 service identity also carries embodiment and plasticity fields. The
    # resident/control ABI deliberately stores the exact dependency subset
    # accepted by sequence_control; service_artifact_sha256 and adapter_sha256
    # transitively bind the full V5 metadata, including those source fields.
    cns_identity = {
        name: service["identity"][name] for name in CNS_DEPENDENCY_KEYS
    }
    control_dependencies = {
        **cns_identity,
        "controller_input": CONTROLLER_FORMAT,
        "organism_interface_sha256": interface_sha256,
        "source_revision": revision,
    }
    control = write_control_artifact(
        head_output,
        heads,
        version=0,
        parent_sha256=None,
        dependencies={
            **control_dependencies,
            "core_packed_sha256": core_hash,
            "predictor_packed_sha256": predictor_hash,
        },
        provenance={
            "training_status": "initialized-untrained",
            "competence_claim": None,
            "operation": "seeded-numpy-pcg64-xavier-initialization",
            "seed": args.seed,
            "source_revision": revision,
            "exporter_file_sha256": file_sha256(Path(__file__).resolve()),
        },
    )
    arrays = {
        **core,
        **predictor,
        **{
            embedded: control.arrays[name]
            for embedded, name in zip(EMBEDDED_ORDER, CONTROL_ORDER, strict=True)
        },
    }
    metadata = {
        "format": NATIVE_POPULATION_FORMAT,
        "version": NATIVE_POPULATION_VERSION,
        "execution": NATIVE_EXECUTION,
        "controller_input": CONTROLLER_FORMAT,
        "source_revision": revision,
        "organism_interface": interface,
        "organism_interface_sha256": interface_sha256,
        "cns_service": cns_identity,
        "cns_service_provenance": {
            "path": service["path"],
            "training_status": service["training_status"],
        },
        "controller_components": {
            "core_pack_order": list(CORE_ORDER),
            "core_packed_sha256": core_hash,
            "predictor_pack_order": list(PREDICTOR_ORDER),
            "predictor_packed_sha256": predictor_hash,
            "sequence_control": copy.deepcopy(control.metadata),
        },
        "initialization": {
            "training_status": "initialized-untrained",
            "competence_claim": None,
            "context_policy_version": "signed-context12-v1",
            "private_learning_version": "context-consequence-v1",
            "native_runtime_format": "chreatures-cns-context-resident-native-v12",
            "context_suffix_memory_format": "chreatures-private-cns-context-suffix-v2",
            "tick_seconds": 0.01,
            "goal_horizon_seconds": 0.4,
            "source_policy": None,
            "seed": args.seed,
            "algorithm": (
                "fresh signed-context12 numpy-pcg64-xavier; proposal output gain0.3 with "
                "zero bias; zero predictor/control output projections except hazard bias -ln(7)"
            ),
        },
        "pack_order": list(RESIDENT_ORDER),
        "tensors": receipts(arrays),
    }
    metadata["artifact_sha256"] = artifact_identity(metadata, arrays)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(canonical(metadata).decode()),
                **arrays,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        os.chmod(output, 0o444)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "resident": {
                    "path": str(output),
                    "file_sha256": file_sha256(output),
                    "artifact_sha256": metadata["artifact_sha256"],
                    "training_status": "initialized-untrained",
                },
                "sequence_control": {
                    "path": str(control.path),
                    "file_sha256": control.file_sha256,
                    "artifact_sha256": control.sha256,
                    "version": control.version,
                },
                "cns_service": cns_identity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
