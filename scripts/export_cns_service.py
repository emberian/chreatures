#!/usr/bin/env python3
"""Bind learned CNS parameters to the actual graph and optic anatomy artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chreatures.cns_adapter_contract import (
    ARRAY_SPECS, PARAMETER_ORDER, service_identity, write_service_artifact,
)
from chreatures.malecns import MaleCNSGraph


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assemble(graph, atlas_path, parameter_path):
    with np.load(atlas_path, allow_pickle=False) as bundle:
        atlas = {name: np.asarray(bundle[name]) for name in bundle.files}
    if not np.array_equal(atlas["body_ids"], graph.body_ids):
        raise ValueError("atlas body IDs differ from graph order")
    if not np.array_equal(atlas["type_names"][atlas["type_index"]], graph.types):
        raise ValueError("atlas type order differs from graph annotations")
    rows = atlas["photoreceptor_graph_rows"].astype("<u4")
    receptor_labels = atlas["type_names"][atlas["type_index"][rows]]
    receptor_types, type_index = np.unique(receptor_labels, return_inverse=True)
    if len(receptor_types) != 10 or len(atlas["type_names"]) != 11752:
        raise ValueError("anatomical cell type counts differ")
    body_rows = np.flatnonzero(np.isin(graph.superclasses, ["cb_sensory", "vnc_sensory"]))
    ptr = atlas["photoreceptor_site_indptr"].astype("<u4")
    weights = atlas["photoreceptor_site_synapse_counts"].astype("<f4")
    for start, stop in zip(ptr[:-1], ptr[1:], strict=True):
        if stop > start:
            weights[start:stop] /= weights[start:stop].sum(dtype=np.float32)
    matrix = graph.matrix(normalized=True, signed=True).tocsr()
    arrays = {
        "graph.crow": np.asarray(matrix.indptr, dtype="<u4"),
        "graph.col": np.asarray(matrix.indices, dtype="<u4"),
        "graph.weight": np.asarray(matrix.data, dtype="<f4"),
        "atlas.receptor_rows": rows,
        "atlas.receptor_type": np.asarray(type_index, dtype="<u4"),
        "atlas.receptor_ptr": ptr,
        "atlas.site_indices": atlas["photoreceptor_site_indices"].astype("<u4"),
        "atlas.site_weight": weights,
        "atlas.body_rows": np.asarray(body_rows, dtype="<u4"),
        "atlas.neuron_type": atlas["type_index"].astype("<u4"),
    }
    with np.load(parameter_path, allow_pickle=False) as parameters:
        if set(parameters.files) - {"metadata"} != set(PARAMETER_ORDER):
            raise ValueError("parameter archive keys differ from CNS adapter contract")
        for name in PARAMETER_ORDER:
            arrays[name] = np.asarray(parameters[name])
        parameter_metadata = json.loads(str(parameters["metadata"])) if "metadata" in parameters.files else {}
    if (
        parameter_metadata.get("format") != "chreatures-cns-adapter-parameters-v1"
        or parameter_metadata.get("service_format") != "chreatures-cns-service-v1"
        or parameter_metadata.get("parameter_order") != list(PARAMETER_ORDER)
        or parameter_metadata.get("static_identity", {}).get("graph_dataset_sha256") != graph.hash
        or parameter_metadata.get("static_identity", {}).get("atlas_file_sha256") != sha(atlas_path)
    ):
        raise ValueError("parameter provenance differs from the actual graph/atlas/interface")
    for name, dtype, shape in ARRAY_SPECS:
        value = arrays[name]
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"{name}: expected {dtype}{shape}; received {value.dtype}{value.shape}")
    clean = dict(parameter_metadata)
    expected = clean.pop("artifact_sha256", None)
    parameter_receipts = {
        name: dict(shape=list(arrays[name].shape), dtype=arrays[name].dtype.str,
                   sha256=hashlib.sha256(arrays[name].tobytes()).hexdigest())
        for name in PARAMETER_ORDER
    }
    observed = hashlib.sha256(json.dumps(dict(metadata=clean, arrays=parameter_receipts),
                                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if observed != expected:
        raise ValueError("parameter artifact internal identity differs")
    return arrays, dict(parameter_metadata=parameter_metadata,
                        receptor_type_names=receptor_types.tolist(),
                        body_target_rule="superclass in [cb_sensory,vnc_sensory], ascending graph row",
                        unsupported_receptor_current="exactly zero, including tonic",
                        sensory_order="optic site_side_hex RGB row-major; canonical320:351; physiology12")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--training-status", choices=["initialized-untrained", "trained"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".receipt.json")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("CNS export refuses an existing artifact or receipt")
    graph = MaleCNSGraph.load(args.graph, mmap=True, verify=True)
    arrays, provenance = assemble(graph, args.atlas, args.parameters)
    if provenance["parameter_metadata"]["training_status"] != args.training_status:
        raise ValueError("export cannot relabel the parameter artifact's training status")
    provenance["parameter_file_sha256"] = sha(args.parameters)
    provenance["export_source_sha256"] = {
        name: sha(ROOT / name) for name in (
            "scripts/export_cns_service.py", "chreatures/cns_adapter_contract.py", "chreatures/malecns.py"
        )
    }
    receipt = write_service_artifact(args.output, arrays, graph_sha256=graph.hash,
                                     atlas_sha256=sha(args.atlas), training_status=args.training_status,
                                     provenance=provenance)
    receipt["cns_adapter"] = service_identity(receipt["metadata"], receipt["file_sha256"])
    with sidecar.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"path": receipt["path"], "bytes": receipt["bytes"],
                      "file_sha256": receipt["file_sha256"], "cns_adapter": receipt["cns_adapter"],
                      "receipt": str(sidecar)}))


if __name__ == "__main__":
    main()
