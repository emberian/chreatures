#!/usr/bin/env python3
"""One-way V3 research seed plus fly atlas to a fresh current CHCNS4 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from chreatures.cns_adapter_contract import (
    canonical,
    neutral_afferent_drive,
    write_service_artifact,
)
from research.anatomical_cns.model import initialized_arrays

V3_MAGIC = b"CHCNS3\0\0"
V3_SPECS = (
    ("graph.crow", "<u4", (165123,)),
    ("graph.col", "<u4", (25563197,)),
    ("graph.weight", "<f4", (25563197,)),
    ("graph.channel", "<u4", (165122,)),
    ("atlas.receptor_rows", "<u4", (4107,)),
    ("atlas.receptor_type", "<u4", (4107,)),
    ("atlas.receptor_ptr", "<u4", (4108,)),
    ("atlas.site_indices", "<u4", (4669,)),
    ("atlas.site_weight", "<f4", (4669,)),
    ("atlas.body_rows", "<u4", (11233,)),
    ("atlas.body_mask", "<f4", (11233, 110)),
    ("atlas.context_rows", "<u4", (1314,)),
    ("atlas.motor_rows", "<u4", (815,)),
    ("atlas.motor_mask", "<f4", (34, 815)),
    ("atlas.neuron_type", "<u4", (165122,)),
    ("optic.spectral_logits", "<f4", (10, 3)),
    ("optic.gain_raw", "<f4", (10,)),
    ("optic.bias", "<f4", (10,)),
    ("body.mean", "<f4", (110,)),
    ("body.scale", "<f4", (110,)),
    ("body.weight", "<f4", (11233, 110)),
    ("body.bias", "<f4", (11233,)),
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
    ("motor.weight_raw", "<f4", (34, 815)),
    ("motor.bias", "<f4", (34,)),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_v3(path: Path):
    with path.open("rb") as stream:
        if stream.read(8) != V3_MAGIC:
            raise ValueError("seed must be a sealed CHCNS3 research artifact")
        metadata_length = struct.unpack("<I", stream.read(4))[0]
        metadata = json.loads(stream.read(metadata_length))
    if metadata.get("format") != "chreatures-cns-service-v3":
        raise ValueError("seed metadata is not CHCNS3")
    offset = 12 + metadata_length
    arrays = {}
    for name, dtype, shape in V3_SPECS:
        value = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=shape)
        expected = metadata.get("array_sha256", {}).get(name)
        if expected is None or hashlib.sha256(value).hexdigest() != expected:
            raise ValueError(f"V3 seed tensor receipt differs: {name}")
        arrays[name] = value
        offset += value.nbytes
    if path.stat().st_size != offset:
        raise ValueError("V3 seed has trailing or missing bytes")
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-service", type=Path, required=True)
    parser.add_argument("--anatomy-v3", type=Path, required=True)
    parser.add_argument("--fly-atlas", type=Path, required=True)
    parser.add_argument("--morphology-schema", type=Path, required=True)
    parser.add_argument("--sensory-schema", type=Path, required=True)
    parser.add_argument("--actuator-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    arguments = parser.parse_args()

    old, old_metadata = load_v3(arguments.v3_service)
    with np.load(arguments.anatomy_v3, allow_pickle=False) as archive:
        measured_graph_weight = np.asarray(archive["graph.weight"])
        measured_graph_channel = np.asarray(archive["graph.channel"])
        measured_graph_identity = str(archive["graph_sha256"])
    if (
        measured_graph_weight.shape != (25563197,)
        or measured_graph_weight.dtype != np.dtype("<f4")
        or measured_graph_channel.shape != (165122,)
        or measured_graph_channel.dtype != np.dtype("<u4")
        or measured_graph_identity != old_metadata["graph_sha256"]
    ):
        raise ValueError("measured V3 anatomy graph differs from the V4 seed graph")
    with np.load(arguments.fly_atlas, allow_pickle=False) as archive:
        atlas = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "atlas.body_rows": (11798,),
        "atlas.body_mask": (11798, 807),
        "atlas.motor_rows": (815,),
        "atlas.motor_mask": (92, 815),
    }
    for name, shape in required.items():
        if name not in atlas or atlas[name].shape != shape:
            raise ValueError(f"fly atlas {name} must have shape {shape}")
    if str(atlas.get("atlas.schema", "")) != "chreatures.fly-body-neural-atlas.v1":
        raise ValueError("fly atlas schema differs")
    if str(atlas.get("atlas.body_schema_sha256", "")) != digest(
        arguments.morphology_schema
    ) or str(atlas.get("atlas.body_channel_schema_sha256", "")) != digest(
        arguments.sensory_schema
    ):
        raise ValueError("fly atlas body or sensory schema identity differs")

    actuator_schema = json.loads(arguments.actuator_schema.read_text())
    if (
        actuator_schema.get("schema") != "chreatures.motor-output-channels.v1"
        or actuator_schema.get("identity") != "MOTOR92"
        or actuator_schema.get("length") != 92
        or actuator_schema.get("body_schema_sha256")
        != digest(arguments.morphology_schema)
    ):
        raise ValueError("actuator schema is not the canonical MOTOR92 interface")
    channels = actuator_schema.get("channels")
    channel_indices = (
        [entry.get("index") for entry in channels]
        if isinstance(channels, list)
        else None
    )
    if channel_indices != list(range(92)):
        raise ValueError("actuator schema channels must cover indices 0 through 91")

    graph_source = np.ascontiguousarray(measured_graph_weight, dtype="<f4")
    quantized = graph_source.astype("<f2").view("<u2")
    static = {
        name: old[name]
        for name in (
            "graph.crow",
            "graph.col",
            "atlas.receptor_rows",
            "atlas.receptor_type",
            "atlas.receptor_ptr",
            "atlas.site_indices",
            "atlas.site_weight",
            "atlas.context_rows",
            "atlas.neuron_type",
        )
    }
    static.update(
        {
            "graph.weight_bits": np.ascontiguousarray(quantized),
            "graph.channel": np.ascontiguousarray(measured_graph_channel, dtype="<u4"),
            "atlas.body_rows": np.ascontiguousarray(
                atlas["atlas.body_rows"], dtype="<u4"
            ),
            "atlas.body_mask": np.ascontiguousarray(
                atlas["atlas.body_mask"], dtype="<f4"
            ),
            "atlas.motor_rows": np.ascontiguousarray(
                atlas["atlas.motor_rows"], dtype="<u4"
            ),
            "atlas.motor_mask": np.ascontiguousarray(
                atlas["atlas.motor_mask"], dtype="<f4"
            ),
        }
    )
    arrays = initialized_arrays(static, arguments.seed)
    reused = (
        "optic.spectral_logits",
        "optic.gain_raw",
        "optic.bias",
        "context.weight",
        "context.bias",
        "dynamics.baseline_raw",
        "dynamics.recurrent_gain_raw",
        "dynamics.tau_raw",
        "dynamics.adaptation_gain_raw",
        "dynamics.adaptation_tau_raw",
        "dynamics.release_tau_raw",
        "dynamics.release_use_raw",
        "dynamics.mod_gain_raw",
        "dynamics.mod_adaptation_raw",
        "dynamics.modulation_tau_raw",
        "readout.projection.weight",
        "readout.output.weight",
        "readout.output.bias",
    )
    for name in reused:
        arrays[name] = np.ascontiguousarray(old[name], dtype="<f4")
    motor_types = arrays["atlas.neuron_type"][arrays["atlas.motor_rows"]]
    arrays["motor.reference_rate"] = np.ascontiguousarray(
        0.05 + 0.4 / (1.0 + np.exp(-arrays["dynamics.baseline_raw"][motor_types])),
        dtype="<f4",
    )
    arrays["afferent.neutral_drive"] = neutral_afferent_drive(arrays)
    calibration = {
        "status": "initialized-untrained",
        "seed": arguments.seed,
        "reference": "type baseline at birth",
        "rate_scale": 0.05,
        "decoder": "small zero-mean signed weights within anatomical mask",
    }
    calibration_sha = hashlib.sha256(canonical(calibration)).hexdigest()
    receipt = write_service_artifact(
        arguments.output,
        arrays,
        graph_sha256=old_metadata["graph_sha256"],
        atlas_sha256=old_metadata["atlas_sha256"],
        anatomy_sha256=digest(arguments.fly_atlas),
        morphology_sha256=digest(arguments.morphology_schema),
        sensory_schema_sha256=digest(arguments.sensory_schema),
        actuator_schema_sha256=digest(arguments.actuator_schema),
        motor_calibration_sha256=calibration_sha,
        graph_source_weight_sha256=hashlib.sha256(graph_source).hexdigest(),
        training_status="initialized-untrained",
        provenance={
            "source_v3_file_sha256": digest(arguments.v3_service),
            "source_v3_adapter_sha256": old_metadata["adapter_sha256"],
            "measured_graph_anatomy_sha256": digest(arguments.anatomy_v3),
            "migration": "one-way seed; no V3 runtime loader or private state migration",
            "reused": list(reused),
            "motor_calibration": calibration,
            "new_untrained_interfaces": [
                "BODY807 afferents",
                "centered signed MOTOR92 decoder",
            ],
            "graph_weight_conversion": "source float32 rounded once to IEEE binary16 bits",
        },
    )
    receipt_path = arguments.output.with_suffix(
        arguments.output.suffix + ".receipt.json"
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "path": receipt["path"],
                "sha256": receipt["file_sha256"],
                "adapter_sha256": receipt["metadata"]["adapter_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
