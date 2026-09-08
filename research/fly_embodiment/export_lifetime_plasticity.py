#!/usr/bin/env python3
"""Build the compact CNS V5 gamma1pedc plasticity selector.

This is an offline compiler.  It authenticates a frozen CHCNS4 parent and the
two prior anatomical audits, then copies only canonical CSR edge positions and
the fixed V5 rule.  It never derives weights from the historical float32
full-graph bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np

FORMAT = "chreatures-cns-v5-lifetime-plasticity-selector-v1"
GRAPH_SHA256 = "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625"
TARGET_ROWS = np.asarray([655, 1306], dtype="<u4")
TARGET_BODY_IDS = np.asarray([10704, 11402], dtype="<i8")
DAN_ROWS = np.asarray([1774, 1235], dtype="<u4")
DAN_BODY_IDS = np.asarray([11900, 11327], dtype="<i8")
TARGET_PTR = np.asarray([0, 2048, 4184], dtype="<u4")
RULE = np.asarray([[0.8, 0.25, 0.8], [0.8, 0.25, 0.8]], dtype="<f4")
V4_MAGIC = b"CHCNS4\0\0"
V4_GRAPH_ARRAYS = (
    ("graph.crow", "<u4", (165123,)),
    ("graph.col", "<u4", (25563197,)),
    ("graph.weight_bits", "<u2", (25563197,)),
    ("graph.channel", "<u4", (165122,)),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def selector_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(FORMAT.encode())
    for name, value in arrays.items():
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def load_frozen_v4_graph(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read only the graph prefix using the frozen CHCNS4 wire contract."""
    with path.open("rb") as stream:
        if stream.read(8) != V4_MAGIC:
            raise ValueError("parent must be a frozen CHCNS4 service")
        raw_length = stream.read(4)
        if len(raw_length) != 4:
            raise ValueError("truncated CHCNS4 metadata length")
        metadata = json.loads(stream.read(struct.unpack("<I", raw_length)[0]))
        offset = stream.tell()
    if metadata.get("format") != "chreatures-cns-service-v4":
        raise ValueError("parent metadata is not the frozen V4 contract")
    arrays: dict[str, np.ndarray] = {}
    for name, dtype, shape in V4_GRAPH_ARRAYS:
        value = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=shape)
        expected = metadata.get("array_sha256", {}).get(name)
        if expected is None or array_sha256(value) != expected:
            raise ValueError(f"CHCNS4 graph tensor receipt differs: {name}")
        arrays[name] = value
        offset += value.nbytes
    return arrays, metadata


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ whose bytes do not depend on wall-clock ZIP timestamps."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", type=Path, help="authenticated frozen CHCNS4 parent")
    parser.add_argument("candidate", type=Path, help="lifetime-plasticity candidate NPZ")
    parser.add_argument("old_selector", type=Path, help="audited gamma1pedc selector NPZ")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt_path = args.output.with_suffix(".json")
    if args.output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite versioned V5 selector artifact")

    graph, service_meta = load_frozen_v4_graph(args.service)
    if service_meta["graph_sha256"] != GRAPH_SHA256:
        raise ValueError("parent service graph identity differs")
    with np.load(args.candidate, allow_pickle=False) as candidate, np.load(
        args.old_selector, allow_pickle=False
    ) as old:
        mask = np.asarray(candidate["candidate.recommended_gamma1pedc"], dtype=bool)
        positions = np.asarray(candidate["candidate.edge_positions"][mask], dtype="<u4")
        pre = np.asarray(candidate["candidate.pre_rows"][mask], dtype="<u4")
        post = np.asarray(candidate["candidate.post_rows"][mask], dtype="<u4")
        synapses = np.asarray(candidate["candidate.synapse_counts"][mask], dtype="<u4")
        if str(candidate["graph_sha256"]) != GRAPH_SHA256:
            raise ValueError("candidate graph identity differs")
        old_meta = json.loads(str(old["metadata"]))
        if old_meta["graph_sha256"] != GRAPH_SHA256:
            raise ValueError("old selector graph identity differs")
        checks = (
            (old["mbon_neuron_indices"], TARGET_ROWS, "MBON rows"),
            (old["mbon_body_ids"], TARGET_BODY_IDS, "MBON body IDs"),
            (old["dan_neuron_indices"], DAN_ROWS, "PPL101 rows"),
            (old["dan_body_ids"], DAN_BODY_IDS, "PPL101 body IDs"),
            (old["edge_source_neuron_indices"], pre, "KC source rows"),
            (old["edge_target_neuron_indices"], post, "MBON edge targets"),
            (old["synapse_counts"], synapses, "edge synapse counts"),
        )
        for observed, expected, label in checks:
            if not np.array_equal(observed, expected):
                raise ValueError(f"{label} differ from the audited selector")
        if not np.array_equal(old["edge_target_positions"], np.repeat([0, 1], np.diff(TARGET_PTR))):
            raise ValueError("old selector target grouping differs")

    if positions.shape != (4184,) or np.any(positions[1:] <= positions[:-1]):
        raise ValueError("selected canonical edge positions must be sorted unique [4184]")
    if not np.array_equal(post, np.repeat(TARGET_ROWS, np.diff(TARGET_PTR))):
        raise ValueError("selected edges do not have the required target grouping")
    csr_targets = np.searchsorted(graph["graph.crow"], positions, side="right") - 1
    sources = np.asarray(graph["graph.col"][positions], dtype="<u4")
    weight_bits = np.asarray(graph["graph.weight_bits"][positions], dtype="<u2")
    if not np.array_equal(csr_targets.astype("<u4"), post):
        raise ValueError("candidate positions do not resolve to their canonical CSR targets")
    if not np.array_equal(sources, pre):
        raise ValueError("candidate sources differ from canonical graph.col")
    if np.any(graph["graph.channel"][sources] != 1):
        raise ValueError("every selected KC source must use fast channel 1")
    decoded = weight_bits.view("<f2").astype("<f4")
    if not np.isfinite(decoded).all() or np.any(decoded <= 0):
        raise ValueError("selected canonical half coefficients must be finite and positive")
    if np.any(~np.isfinite(RULE)) or np.any((RULE[:, 0] < .05) | (RULE[:, 0] > 5)) \
            or np.any((RULE[:, 1] < 0) | (RULE[:, 1] > 1)) \
            or np.any((RULE[:, 2] < 0) | (RULE[:, 2] > .95)):
        raise ValueError("plasticity rule is outside the V5 contract")

    arrays = {
        "plasticity.edge_positions": positions,
        "plasticity.target_ptr": TARGET_PTR,
        "plasticity.target_rows": TARGET_ROWS,
        "plasticity.dan_rows": DAN_ROWS,
        "plasticity.rule": RULE,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(args.output, arrays)
    receipt = {
        "format": f"{FORMAT}-receipt",
        "version": 1,
        "artifact": args.output.name,
        "artifact_bytes": args.output.stat().st_size,
        "artifact_sha256": file_sha256(args.output),
        "selector_sha256": selector_sha256(arrays),
        "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
        "source": {
            "parent_service_sha256": file_sha256(args.service),
            "parent_service_format": service_meta["format"],
            "graph_sha256": GRAPH_SHA256,
            "graph_col_sha256": service_meta["array_sha256"]["graph.col"],
            "graph_crow_sha256": service_meta["array_sha256"]["graph.crow"],
            "graph_channel_sha256": service_meta["array_sha256"]["graph.channel"],
            "graph_weight_bits_sha256": service_meta["array_sha256"]["graph.weight_bits"],
            "candidate_artifact_sha256": file_sha256(args.candidate),
            "old_anatomical_selector_sha256": file_sha256(args.old_selector),
        },
        "validated": {
            "edge_count": 4184,
            "synapse_count": int(synapses.sum()),
            "connected_kc_rows": int(np.unique(sources).size),
            "target_ptr": TARGET_PTR.astype(int).tolist(),
            "target_rows": TARGET_ROWS.astype(int).tolist(),
            "target_body_ids": TARGET_BODY_IDS.astype(int).tolist(),
            "dan_rows": DAN_ROWS.astype(int).tolist(),
            "dan_body_ids": DAN_BODY_IDS.astype(int).tolist(),
            "all_sources_fast_channel": True,
            "edge_positions_match_candidate": True,
            "endpoints_match_old_selector": True,
            "csr_targets_match": True,
            "canonical_selected_weight_bits_sha256": array_sha256(weight_bits),
            "canonical_selected_decoded_f32_sha256": array_sha256(decoded),
            "canonical_half_coefficient_min": float(decoded.min()),
            "canonical_half_coefficient_max": float(decoded.max()),
        },
        "provenance": {
            "measured": "CSR positions/endpoints, synapse counts, cell annotations, and canonical deployed half coefficients",
            "engineered": "bilateral gamma1pedc subset, side-paired PPL101 gate, and fixed rule rows",
            "numeric_rule": "The artifact contains no copied weight array; runtime derives source rows and exact baseline coefficients from graph.col/graph.weight_bits at validated positions.",
            "excluded": "Historical gamma1pedc-fullgraph-bridge-v1 float32 coefficients are not read.",
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
