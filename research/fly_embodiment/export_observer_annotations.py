#!/usr/bin/env python3
"""Export compact, row-aligned MaleCNS annotations for observers only."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "chreatures-male-cns-observer-annotations-v1"
GRAPH_SHA256 = "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625"
NEURONS_SHA256 = "0e6706229b93cbdcab48905504bd3b8f6075534488c65ad2c8f844668036ea9a"
ATLAS_SHA256 = "066136815c53fd7855e12246a60ce05702f44f4ea01ee547f01734418b0bc873"
ROWS = 165_122

FIELDS = (
    "superclasses", "classes", "subclasses", "types", "manc_types", "sides",
    "entry_nerves", "exit_nerves", "predicted_nt", "effective_nt", "nt_basis",
)
FLAG_BITS = {
    "sensory_annotation": 0,
    "sensory_atlas_row": 1,
    "descending_context_row": 2,
    "motor_annotation": 3,
    "motor92_routed": 4,
    "body807_routed": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoded(array: np.ndarray, dtype: str) -> str:
    value = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
    return base64.b64encode(value.tobytes()).decode("ascii")


def dictionary_column(values: np.ndarray) -> dict:
    dictionary, codes = np.unique(values.astype(str), return_inverse=True)
    dtype = "u2" if len(dictionary) <= np.iinfo(np.uint16).max else "u4"
    return {
        "dictionary": dictionary.tolist(),
        "codes": encoded(codes, dtype),
        "codes_dtype": dtype,
        "missing_code": int(np.searchsorted(dictionary, "")) if "" in dictionary else None,
    }


def routing_csr(global_rows: np.ndarray, mask_output_by_row: np.ndarray, total_rows: int) -> tuple[np.ndarray, np.ndarray]:
    local_row, channel = np.nonzero(mask_output_by_row.T)
    routed_rows = global_rows[local_row]
    counts = np.bincount(routed_rows, minlength=total_rows)
    ptr = np.empty(total_rows + 1, np.uint32)
    ptr[0] = 0
    np.cumsum(counts, out=ptr[1:])
    return ptr, channel.astype(np.uint16)


def build(neurons_path: Path, atlas_path: Path) -> tuple[dict, dict]:
    observed_neurons_sha = sha256(neurons_path)
    observed_atlas_sha = sha256(atlas_path)
    if observed_neurons_sha != NEURONS_SHA256:
        raise ValueError(f"neurons.npz differs: {observed_neurons_sha}")
    if observed_atlas_sha != ATLAS_SHA256:
        raise ValueError(f"fly atlas differs: {observed_atlas_sha}")

    with np.load(neurons_path, allow_pickle=False) as neurons, np.load(atlas_path, allow_pickle=False) as atlas:
        if len(neurons["body_ids"]) != ROWS:
            raise ValueError("canonical neuron row count differs")
        if str(atlas["atlas.neurons_sha256"]) != NEURONS_SHA256:
            raise ValueError("atlas is bound to another neurons.npz")
        for role in ("afferent", "motor"):
            rows = np.asarray(atlas[f"atlas.{role}_rows"], np.int64)
            if not np.array_equal(neurons["ids"][rows], atlas[f"atlas.{role}_ids"]):
                raise ValueError(f"{role} exact IDs do not align with canonical rows")
            if not np.array_equal(neurons["body_ids"][rows], atlas[f"atlas.{role}_body_ids"]):
                raise ValueError(f"{role} bodyIds do not align with canonical rows")

        columns = {field: dictionary_column(neurons[field]) for field in FIELDS}
        superclasses = neurons["superclasses"]
        flags = np.zeros(ROWS, np.uint8)
        sensory = np.fromiter(
            (("_sensory" in value) or value.startswith("sensory_") for value in superclasses),
            bool, count=ROWS,
        )
        motor = np.isin(superclasses, ["vnc_motor", "cb_motor"])
        flags[sensory] |= 1 << FLAG_BITS["sensory_annotation"]
        flags[atlas["atlas.afferent_rows"]] |= 1 << FLAG_BITS["sensory_atlas_row"]
        flags[superclasses == "descending_neuron"] |= 1 << FLAG_BITS["descending_context_row"]
        flags[motor] |= 1 << FLAG_BITS["motor_annotation"]
        motor_rows = atlas["atlas.motor_rows"]
        flags[motor_rows[np.asarray(atlas["atlas.motor_row_supported"], bool)]] |= 1 << FLAG_BITS["motor92_routed"]
        afferent_rows = atlas["atlas.afferent_rows"]
        body_supported = np.any(atlas["atlas.body_mask"] != 0, axis=1)
        flags[afferent_rows[body_supported]] |= 1 << FLAG_BITS["body807_routed"]

        sensory_ptr, sensory_indices = routing_csr(
            afferent_rows, np.asarray(atlas["atlas.afferent_port_mask"], bool), ROWS
        )
        motor_ptr, motor_indices = routing_csr(
            motor_rows, np.asarray(atlas["atlas.motor_mask"] != 0, bool), ROWS
        )

        payload = {
            "format": FORMAT,
            "version": 1,
            "graph_sha256": GRAPH_SHA256,
            "row_count": ROWS,
            "row_identity": "canonical local row; neurons sorted by ascending exact bodyId",
            "body_ids": {"data": encoded(neurons["body_ids"], "i8"), "dtype": "i8", "encoding": "base64 little-endian; decode as BigInt"},
            "columns": columns,
            "nt_confidence": {"data": encoded(neurons["nt_confidence"], "f4"), "dtype": "f4"},
            "route_flags": {"data": encoded(flags, "u1"), "dtype": "u1", "bits": FLAG_BITS},
            "sensory_ports": {
                "names": atlas["atlas.afferent_port_names"].tolist(),
                "modalities": atlas["atlas.afferent_port_modalities"].tolist(),
                "evidence_grades": atlas["atlas.afferent_port_evidence_grades"].tolist(),
                "row_ptr": encoded(sensory_ptr, "u4"),
                "indices": encoded(sensory_indices, "u2"),
            },
            "motor_outputs": {
                "names": atlas["atlas.actuator_ids"].tolist(),
                "targets": atlas["atlas.actuator_targets"].tolist(),
                "groups": atlas["atlas.actuator_groups"].tolist(),
                "evidence_grades": atlas["atlas.actuator_evidence_grades"].tolist(),
                "row_ptr": encoded(motor_ptr, "u4"),
                "indices": encoded(motor_indices, "u2"),
            },
            "scope": "observer-only annotations; forbidden as controller, trainer, predictor, memory, or action-scoring input",
        }
        report = {
            "format": f"{FORMAT}-manifest",
            "version": 1,
            "graph_sha256": GRAPH_SHA256,
            "rows": ROWS,
            "sources": {
                "neurons_npz": {"sha256": observed_neurons_sha, "row_order": "ascending exact bodyId"},
                "fly_body_neural_atlas": {"sha256": observed_atlas_sha, "schema": str(atlas["atlas.schema"])},
            },
            "counts": {
                "sensory_annotation": int(sensory.sum()),
                "sensory_atlas_rows": len(afferent_rows),
                "descending_context_rows": int((superclasses == "descending_neuron").sum()),
                "motor_annotation": int(motor.sum()),
                "motor92_routed_rows": int(np.asarray(atlas["atlas.motor_row_supported"], bool).sum()),
                "motor92_unsupported_rows": int((~np.asarray(atlas["atlas.motor_row_supported"], bool)).sum()),
                "body807_routed_rows": int(body_supported.sum()),
                "body807_unsupported_sensory_rows": int((~body_supported).sum()),
                "sensory_port_memberships": len(sensory_indices),
                "motor_output_memberships": len(motor_indices),
            },
            "semantics": {
                "annotations": "exact dataset strings; empty remains missing",
                "predicted_nt": "dataset prediction, not receptor effect or physiological sign",
                "routing": "membership copied from the hash-bound actual-fly atlas; engineered mappings retain its evidence labels",
                "coordinates": "absent; this sidecar makes no coordinate claim",
                "information_boundary": payload["scope"],
            },
        }
    return payload, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    payload, report = build(args.neurons, args.atlas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "male-cns-observer-annotations-v1.json.gz"
    manifest = args.output_dir / "male-cns-observer-annotations-v1.manifest.json"
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite versioned observer sidecar")
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with output.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=9) as compressed:
            compressed.write(raw)
    report["artifact"] = {
        "filename": output.name,
        "encoding": "gzip JSON; deterministic mtime=0",
        "uncompressed_bytes": len(raw),
        "transport_bytes": output.stat().st_size,
        "transport_sha256": sha256(output),
    }
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
