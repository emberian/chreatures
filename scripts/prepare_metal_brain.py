#!/usr/bin/env python3
"""Build the compact Metal v2 graph and port artifact reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chreatures.malecns import MaleCNSGraph
from chreatures.neural_ports import NeuralPortBundle, build_neural_port, load_port_spec


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


PORT_ARRAYS = (
    "input_data",
    "input_indices",
    "input_indptr",
    "input_names",
    "input_shape",
    "readout_data",
    "readout_indices",
    "readout_indptr",
    "readout_names",
    "readout_shape",
)


def port_metadata(path):
    with np.load(path, allow_pickle=False) as bundle:
        return json.loads(str(bundle["metadata"])), json.loads(str(bundle["spec"]))


def assert_equal_port_arrays(source, current):
    with (
        np.load(source, allow_pickle=False) as before,
        np.load(current, allow_pickle=False) as after,
    ):
        for name in PORT_ARRAYS:
            if (
                before[name].dtype != after[name].dtype
                or before[name].shape != after[name].shape
                or not np.array_equal(before[name], after[name])
            ):
                raise ValueError(f"retinal port array differs: {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--graph", type=Path, help="canonical MaleCNS derived directory"
    )
    source.add_argument(
        "--source-artifact",
        type=Path,
        help="validated prior binary whose sparse port arrays equal the current bundle",
    )
    p.add_argument(
        "--port-spec", type=Path, default=ROOT / "data/ports/retinal-v2.json"
    )
    p.add_argument(
        "--port-bundle", type=Path, default=ROOT / "data/ports/retinal-v2-maps.npz"
    )
    p.add_argument(
        "--annotation",
        type=Path,
        help="pinned MaleCNS annotation Feather; builds a missing port bundle",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/metal-brain/metal-csr-retinal-v2.bin",
    )
    p.add_argument("--source-manifest", type=Path)
    p.add_argument("--source-port-bundle", type=Path)
    p.add_argument(
        "--manifest",
        type=Path,
        help="sidecar path (default: OUTPUT with .manifest.json suffix)",
    )
    a = p.parse_args()
    graph = (
        MaleCNSGraph.load(a.graph, mmap=True, verify=True)
        if a.graph is not None
        else None
    )
    spec = load_port_spec(a.port_spec)
    if not a.port_bundle.exists():
        if graph is None or a.annotation is None:
            p.error("a missing --port-bundle requires both --graph and --annotation")
        receipt = build_neural_port(
            graph, a.port_spec, annotation_path=a.annotation
        ).save(a.port_bundle)
    else:
        receipt = {
            "path": str(a.port_bundle),
            "bytes": a.port_bundle.stat().st_size,
            "sha256": sha(a.port_bundle),
        }
    document = json.loads(a.port_spec.read_text())
    built = document.get("built_artifact", {})
    semantic_hash = hashlib.sha256(
        json.dumps(
            spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    metadata, bundle_spec = port_metadata(a.port_bundle)
    if bundle_spec != spec or metadata.get("spec_hash") != semantic_hash:
        raise ValueError("port bundle semantic spec differs from --port-spec")
    if built and (
        receipt["sha256"] != built.get("sha256")
        or receipt["bytes"] != built.get("bytes")
    ):
        raise ValueError("port bundle checksum/size differs from port spec receipt")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".tmp")
    provenance = {}
    if graph is not None:
        bundle = NeuralPortBundle.load(a.port_bundle, graph)
        recurrent = graph.matrix(normalized=True, signed=True)
        inputs = bundle.input_map.tocsr()
        readouts = bundle.readout_map.tocsr()
        if inputs.shape != (graph.n, 351) or readouts.shape != (384, graph.n):
            raise ValueError("Metal requires 351 input and 384 readout maps")
        with tmp.open("wb") as f:
            f.write(
                struct.pack(
                    "<IIIIII",
                    0x4D424732,
                    graph.n,
                    graph.edge_count,
                    inputs.nnz,
                    readouts.nnz,
                    0,
                )
            )
            for array, dtype in (
                (recurrent.indptr, "<u4"),
                (recurrent.indices, "<u4"),
                (recurrent.data, "<f4"),
                (inputs.indptr, "<u4"),
                (inputs.indices, "<u4"),
                (inputs.data, "<f4"),
                (readouts.indptr, "<u4"),
                (readouts.indices, "<u4"),
                (readouts.data, "<f4"),
            ):
                f.write(np.asarray(array, dtype=dtype).tobytes())
        graph_hash = graph.hash
        neurons, edges = graph.n, graph.edge_count
    else:
        if a.source_manifest is None or a.source_port_bundle is None:
            p.error(
                "--source-artifact requires --source-manifest and --source-port-bundle"
            )
        source_receipt = json.loads(a.source_manifest.read_text(encoding="utf-8"))
        if (
            sha(a.source_artifact) != source_receipt.get("artifact_sha256")
            or a.source_artifact.stat().st_size != source_receipt.get("artifact_bytes")
            or sha(a.source_port_bundle) != source_receipt.get("port_bundle_sha256")
            or metadata.get("graph_hash") != source_receipt.get("graph_sha256")
        ):
            raise ValueError(
                "source artifact, graph, or port identity differs from its manifest"
            )
        assert_equal_port_arrays(a.source_port_bundle, a.port_bundle)
        with a.source_artifact.open("rb") as stream:
            header = struct.unpack("<IIIIII", stream.read(24))
        if (
            header[0] != 0x4D424732
            or header[1] != source_receipt.get("neurons")
            or header[2] != source_receipt.get("edges")
        ):
            raise ValueError("source artifact header differs from its manifest")
        shutil.copyfile(a.source_artifact, tmp)
        graph_hash = source_receipt["graph_sha256"]
        neurons, edges = header[1], header[2]
        provenance = {
            "source_artifact_sha256": source_receipt["artifact_sha256"],
            "source_port_bundle_sha256": source_receipt["port_bundle_sha256"],
            "port_map_equality": "equal dtype, shape, and element values for every sparse map array and channel name",
        }
    tmp.replace(a.output)
    artifact_hash = sha(a.output)
    manifest_path = a.manifest or a.output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 2,
        "format": "metal-csr-v2",
        "recipe": "normalized-signed-float32-recurrence+retinal-v2-csr",
        "artifact_sha256": artifact_hash,
        "artifact_bytes": a.output.stat().st_size,
        "graph_sha256": graph_hash,
        "port_spec_sha256": semantic_hash,
        "port_bundle_sha256": receipt["sha256"],
        "neurons": neurons,
        "edges": edges,
        "inputs": 351,
        "readouts": 384,
        **provenance,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_suffix(".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_tmp.replace(manifest_path)
    print(
        json.dumps(
            {
                "artifact": {
                    "path": str(a.output),
                    "bytes": a.output.stat().st_size,
                    "sha256": artifact_hash,
                    "format": "metal-csr-v2",
                    "neurons": neurons,
                    "edges": edges,
                    "inputs": 351,
                    "readouts": 384,
                },
                "graph_sha256": graph_hash,
                "port_spec_sha256": semantic_hash,
                "port_bundle": receipt,
                "manifest": {"path": str(manifest_path), **manifest},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
