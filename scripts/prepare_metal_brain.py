#!/usr/bin/env python3
"""Build the compact Metal v2 graph and port artifact reproducibly."""

from __future__ import annotations
import argparse, hashlib, json, struct, sys
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph", type=Path, required=True, help="canonical MaleCNS derived directory"
    )
    p.add_argument(
        "--port-spec", type=Path, default=ROOT / "data/ports/retinal-v1.json"
    )
    p.add_argument(
        "--port-bundle", type=Path, default=ROOT / "data/ports/retinal-v1-maps.npz"
    )
    p.add_argument(
        "--annotation",
        type=Path,
        help="pinned MaleCNS annotation Feather; builds a missing port bundle",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "data/metal-brain/metal-csr-v2.bin"
    )
    p.add_argument(
        "--manifest",
        type=Path,
        help="sidecar path (default: OUTPUT with .manifest.json suffix)",
    )
    a = p.parse_args()
    graph = MaleCNSGraph.load(a.graph, mmap=True, verify=True)
    spec = load_port_spec(a.port_spec)
    if not a.port_bundle.exists():
        if a.annotation is None:
            p.error("--annotation is required when --port-bundle does not exist")
        receipt = build_neural_port(
            graph, a.port_spec, annotation_path=a.annotation
        ).save(a.port_bundle)
    else:
        receipt = {
            "path": str(a.port_bundle),
            "bytes": a.port_bundle.stat().st_size,
            "sha256": sha(a.port_bundle),
        }
    bundle = NeuralPortBundle.load(a.port_bundle, graph)
    document = json.loads(a.port_spec.read_text())
    built = document.get("built_artifact", {})
    semantic_hash = hashlib.sha256(
        json.dumps(
            spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    if bundle.spec != spec or bundle.spec_hash != semantic_hash:
        raise ValueError("port bundle semantic spec differs from --port-spec")
    if built and (
        receipt["sha256"] != built.get("sha256")
        or receipt["bytes"] != built.get("bytes")
    ):
        raise ValueError("port bundle checksum/size differs from port spec receipt")
    recurrent = graph.matrix(normalized=True, signed=True)
    inputs = bundle.input_map.tocsr()
    readouts = bundle.readout_map.tocsr()
    if inputs.shape != (graph.n, 351) or readouts.shape != (384, graph.n):
        raise ValueError("Metal v2 requires retinal-v1 351 input and 384 readout maps")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".tmp")
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
    tmp.replace(a.output)
    artifact_hash = sha(a.output)
    manifest_path = a.manifest or a.output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "format": "metal-csr-v2",
        "recipe": "normalized-signed-float32-recurrence+retinal-v1-csr",
        "artifact_sha256": artifact_hash,
        "artifact_bytes": a.output.stat().st_size,
        "graph_sha256": graph.hash,
        "port_spec_sha256": bundle.spec_hash,
        "port_bundle_sha256": receipt["sha256"],
        "neurons": graph.n,
        "edges": graph.edge_count,
        "inputs": 351,
        "readouts": 384,
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
                    "neurons": graph.n,
                    "edges": graph.edge_count,
                    "inputs": 351,
                    "readouts": 384,
                },
                "graph_sha256": graph.hash,
                "port_spec_sha256": bundle.spec_hash,
                "port_bundle": receipt,
                "manifest": {"path": str(manifest_path), **manifest},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
