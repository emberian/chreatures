#!/usr/bin/env python3
"""Offline, one-way upgrade of one pinned CHCNS4 service to CHCNS5.

The frozen reader below exists only in this migration program. Runtime code accepts
CHCNS5 exclusively, and the migration creates no neural or plastic experience.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from chreatures.cns_adapter_contract import (
    ARRAY_SPECS,
    DIMENSIONS,
    GRAPH_QUANTIZATION,
    PLASTICITY_CONTRACT,
    PLASTICITY_NAMES,
    canonical,
    write_service_artifact,
)

V4_MAGIC = b"CHCNS4\0\0"
V4_FORMAT = "chreatures-cns-service-v4"
V4_DIMENSIONS = {
    "neurons": 165122, "edges": 25563197, "sites": 1771,
    "receptors": 4107, "receptor_types": 10, "site_edges": 4669,
    "body_targets": 11798, "neuron_types": 11752, "body_inputs": 807,
    "context_inputs": 12, "context_targets": 1314, "motor_outputs": 92,
    "motor_targets": 815, "latent": 512, "readout_rank": 64,
    "modulator_families": 3,
}
# Frozen CHCNS4 wire order. Do not replace this with the current service reader.
V4_ARRAY_SPECS = (
    ("graph.crow", "<u4", (165123,)),
    ("graph.col", "<u4", (25563197,)),
    ("graph.weight_bits", "<u2", (25563197,)),
    ("graph.channel", "<u4", (165122,)),
    ("atlas.receptor_rows", "<u4", (4107,)),
    ("atlas.receptor_type", "<u4", (4107,)),
    ("atlas.receptor_ptr", "<u4", (4108,)),
    ("atlas.site_indices", "<u4", (4669,)),
    ("atlas.site_weight", "<f4", (4669,)),
    ("atlas.body_rows", "<u4", (11798,)),
    ("atlas.body_mask", "<f4", (11798, 807)),
    ("atlas.context_rows", "<u4", (1314,)),
    ("atlas.motor_rows", "<u4", (815,)),
    ("atlas.motor_mask", "<f4", (92, 815)),
    ("atlas.neuron_type", "<u4", (165122,)),
    ("optic.spectral_logits", "<f4", (10, 3)),
    ("optic.gain_raw", "<f4", (10,)),
    ("optic.bias", "<f4", (10,)),
    ("body.mean", "<f4", (807,)),
    ("body.scale", "<f4", (807,)),
    ("body.weight", "<f4", (11798, 807)),
    ("body.bias", "<f4", (11798,)),
    ("context.weight", "<f4", (1314, 12)),
    ("context.bias", "<f4", (1314,)),
    ("dynamics.baseline_raw", "<f4", (11752,)),
    ("dynamics.recurrent_gain_raw", "<f4", (11752,)),
    ("dynamics.tau_raw", "<f4", (11752,)),
    ("dynamics.adaptation_gain_raw", "<f4", (11752,)),
    ("dynamics.adaptation_tau_raw", "<f4", (11752,)),
    ("dynamics.release_tau_raw", "<f4", (11752,)),
    ("dynamics.release_use_raw", "<f4", (11752,)),
    ("dynamics.mod_gain_raw", "<f4", (11752, 3)),
    ("dynamics.mod_adaptation_raw", "<f4", (11752, 3)),
    ("dynamics.modulation_tau_raw", "<f4", (3,)),
    ("afferent.neutral_drive", "<f4", (165122,)),
    ("readout.projection.weight", "<f4", (64, 165122)),
    ("readout.output.weight", "<f4", (512, 64)),
    ("readout.output.bias", "<f4", (512,)),
    ("motor.reference_rate", "<f4", (815,)),
    ("motor.rate_scale", "<f4", (815,)),
    ("motor.weight", "<f4", (92, 815)),
    ("motor.intercept", "<f4", (92,)),
)
V4_IDENTITY_KEYS = (
    "format", "graph_sha256", "atlas_sha256", "anatomy_sha256",
    "morphology_sha256", "sensory_schema_sha256", "actuator_schema_sha256",
    "motor_calibration_sha256", "graph_source_weight_sha256",
    "graph_quantization", "readout_mask_sha256", "dimensions",
    "parameter_order", "array_sha256",
)
ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: str, label: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def load_frozen_v4(path: Path, expected_sha256: str):
    require_sha(expected_sha256, "expected V4 identity")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("pinned CHCNS4 file SHA-256 differs")
    with path.open("rb") as stream:
        if stream.read(8) != V4_MAGIC:
            raise ValueError("migration source must be CHCNS4")
        raw_length = stream.read(4)
        if len(raw_length) != 4:
            raise ValueError("truncated CHCNS4 metadata length")
        metadata_length = struct.unpack("<I", raw_length)[0]
        if not 0 < metadata_length < 8 << 20:
            raise ValueError("invalid CHCNS4 metadata length")
        metadata = json.loads(stream.read(metadata_length))
    if (
        metadata.get("format") != V4_FORMAT
        or metadata.get("dimensions") != V4_DIMENSIONS
        or metadata.get("graph_quantization") != GRAPH_QUANTIZATION
    ):
        raise ValueError("frozen CHCNS4 metadata contract differs")
    identity = {key: metadata[key] for key in V4_IDENTITY_KEYS}
    if hashlib.sha256(canonical(identity)).hexdigest() != metadata.get("adapter_sha256"):
        raise ValueError("CHCNS4 adapter identity differs")
    arrays = {}
    offset = 12 + metadata_length
    for name, dtype, shape in V4_ARRAY_SPECS:
        value = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=shape)
        expected = metadata.get("array_sha256", {}).get(name)
        if expected is None or hashlib.sha256(value).hexdigest() != expected:
            raise ValueError(f"CHCNS4 tensor identity differs: {name}")
        arrays[name] = value
        offset += value.nbytes
    if path.stat().st_size != offset:
        raise ValueError("CHCNS4 artifact has trailing or missing bytes")
    return arrays, metadata, actual_sha256


def load_selector(path: Path, expected_sha256: str):
    require_sha(expected_sha256, "expected selector identity")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("pinned plasticity selector file SHA-256 differs")
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("format")
        != "chreatures-cns-v5-lifetime-plasticity-selector-v1-receipt"
        or receipt.get("artifact_sha256") != actual_sha256
        or receipt.get("artifact") != path.name
    ):
        raise ValueError("plasticity selector receipt differs")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(PLASTICITY_NAMES):
            raise ValueError("plasticity selector arrays differ")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for name, dtype, shape in ARRAY_SPECS[-5:]:
        value = arrays[name]
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"plasticity selector {name} differs")
        observed = hashlib.sha256(value).hexdigest()
        if receipt.get("array_sha256", {}).get(name) != observed:
            raise ValueError(f"plasticity selector tensor identity differs: {name}")
    return arrays, receipt, actual_sha256, file_sha256(receipt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-service", type=Path, required=True)
    parser.add_argument("--expected-v4-sha256", required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--expected-selector-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".receipt.json")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("V5 migration refuses an existing artifact or receipt")

    old, old_metadata, old_file_sha = load_frozen_v4(
        args.v4_service, args.expected_v4_sha256
    )
    selected, selector_receipt, selector_file_sha, selector_receipt_sha = load_selector(
        args.selector, args.expected_selector_sha256
    )
    selector_source = selector_receipt.get("source", {})
    if selector_source.get("graph_sha256") != old_metadata["graph_sha256"]:
        raise ValueError("selector graph identity differs from CHCNS4 parent")
    for selector_key, array_name in (
        ("graph_crow_sha256", "graph.crow"),
        ("graph_col_sha256", "graph.col"),
        ("graph_weight_bits_sha256", "graph.weight_bits"),
        ("graph_channel_sha256", "graph.channel"),
    ):
        if selector_source.get(selector_key) != old_metadata["array_sha256"][array_name]:
            raise ValueError(f"selector canonical graph tensor differs: {array_name}")

    arrays = dict(old)
    arrays.update(selected)
    receipt = write_service_artifact(
        args.output,
        arrays,
        graph_sha256=old_metadata["graph_sha256"],
        atlas_sha256=old_metadata["atlas_sha256"],
        anatomy_sha256=old_metadata["anatomy_sha256"],
        morphology_sha256=old_metadata["morphology_sha256"],
        sensory_schema_sha256=old_metadata["sensory_schema_sha256"],
        actuator_schema_sha256=old_metadata["actuator_schema_sha256"],
        motor_calibration_sha256=old_metadata["motor_calibration_sha256"],
        graph_source_weight_sha256=old_metadata["graph_source_weight_sha256"],
        training_status=old_metadata["training_status"],
        provenance={
            "migration": "one-way frozen CHCNS4 to current CHCNS5; no living snapshot migrated",
            "source_v4_file_sha256": old_file_sha,
            "source_v4_adapter_sha256": old_metadata["adapter_sha256"],
            "source_v4_training_status": old_metadata["training_status"],
            "preserved_v4_arrays": [name for name, _, _ in V4_ARRAY_SPECS],
            "preserved_v4_array_sha256": old_metadata["array_sha256"],
            "plasticity_selector_file_sha256": selector_file_sha,
            "plasticity_selector_receipt_sha256": selector_receipt_sha,
            "plasticity_selector_sha256": selector_receipt["selector_sha256"],
            "selector_compiler_parent_v4_file_sha256": selector_source[
                "parent_service_sha256"
            ],
            "plasticity_contract": PLASTICITY_CONTRACT,
            "initial_private_state": "efficacy deviation and eligibility are zero at birth",
            "competence_claim": "none; inherited parameters retain their prior training status",
            "migration_source_sha256": {
                "research/anatomical_cns/upgrade_v4_to_v5.py": file_sha256(
                    ROOT / "research/anatomical_cns/upgrade_v4_to_v5.py"
                ),
                "chreatures/cns_adapter_contract.py": file_sha256(
                    ROOT / "chreatures/cns_adapter_contract.py"
                ),
            },
        },
    )
    with sidecar.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "path": receipt["path"], "bytes": receipt["bytes"],
        "file_sha256": receipt["file_sha256"],
        "adapter_sha256": receipt["metadata"]["adapter_sha256"],
        "plasticity_sha256": receipt["metadata"]["plasticity_sha256"],
        "source_v4_file_sha256": old_file_sha, "receipt": str(sidecar.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
