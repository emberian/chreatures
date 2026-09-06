import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from chreatures.circuit_blueprint import (
    CircuitBlueprint,
    DerivedCircuitGraph,
    compile_blueprint,
)
from chreatures.malecns import MaleCNSGraph
from chreatures.neural_ports import NeuralPortBundle

FIELDS = (
    "body_ids",
    "ids",
    "labels",
    "instances",
    "types",
    "sides",
    "superclasses",
    "classes",
    "subclasses",
    "soma_neuromeres",
    "entry_nerves",
    "exit_nerves",
    "statuses",
    "status_labels",
    "predicted_nt",
    "ground_truth_nt",
    "consensus_nt",
    "effective_nt",
    "nt_basis",
    "nt_confidence",
    "sign",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_graph(root: Path) -> MaleCNSGraph:
    root.mkdir()
    # target rows: 0=[], 1=[0:2,2:5], 2=[1:3], 3=[0:11,2:7]
    arrays = {
        "indptr.npy": np.asarray([0, 0, 2, 3, 5], np.int64),
        "indices.npy": np.asarray([0, 2, 1, 0, 2], np.int32),
        "counts.npy": np.asarray([2, 5, 3, 11, 7], np.uint32),
        "row_synapses.npy": np.asarray([0, 7, 3, 18], np.uint64),
    }
    for name, array in arrays.items():
        np.save(root / name, array, allow_pickle=False)
    metadata = {}
    for field in FIELDS:
        if field == "body_ids":
            metadata[field] = np.asarray([10, 20, 30, 40], np.int64)
        elif field == "ids":
            metadata[field] = np.asarray(["10", "20", "30", "40"])
        elif field == "classes":
            metadata[field] = np.asarray(["input", "KC", "MBON", "output"])
        elif field == "types":
            metadata[field] = np.asarray(["sensory", "KCg", "MBON11", "motor"])
        elif field == "superclasses":
            metadata[field] = np.asarray(
                ["cb_sensory", "cb_intrinsic", "cb_intrinsic", "cb_motor"]
            )
        elif field == "sign" or field == "nt_confidence":
            metadata[field] = np.ones(4, np.float32)
        else:
            metadata[field] = np.full(4, "")
    np.savez_compressed(root / "neurons.npz", **metadata)
    artifacts = {
        name: {"bytes": (root / name).stat().st_size, "sha256": _hash(root / name)}
        for name in (*arrays, "neurons.npz")
    }
    digest = hashlib.sha256()
    for name in artifacts:
        digest.update(name.encode())
        digest.update(artifacts[name]["sha256"].encode())
    manifest = {
        "schema_version": 1,
        "dataset_hash": digest.hexdigest(),
        "dataset": {"name": "toy", "license": "CC BY 4.0"},
        "artifacts": artifacts,
        "sources": {},
        "counts": {"neurons": 4, "edges": 5, "synapses": 28, "classes": {}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return MaleCNSGraph.load(root, mmap=True)


def test_compile_typed_duplicate_and_sparse_edits(tmp_path: Path) -> None:
    graph = _toy_graph(tmp_path / "parent")
    selector = tmp_path / "selector.npz"
    np.savez_compressed(selector, cells=np.asarray([20, 30], np.int64))
    spec = {"name": "toy-ports", "graph": {"dataset_hash": graph.hash}}
    ports = NeuralPortBundle(
        spec=spec,
        graph_hash=graph.hash,
        input_names=["sense"],
        input_map=sparse.csr_matrix(([1.0], ([0], [0])), shape=(4, 1)),
        readout_names=["act"],
        readout_map=sparse.csr_matrix(([1.0], ([0], [2])), shape=(1, 4)),
        input_ports=[],
        readout_ports=[],
    )
    selector_hash = _hash(selector)
    blueprint = CircuitBlueprint.from_value(
        {
            "schema_version": 1,
            "name": "toy-derived",
            "parent": {"graph_sha256": graph.hash, "port_spec_sha256": ports.spec_hash},
            "modules": [
                {
                    "name": "memory",
                    "copies": 1,
                    "boundary": "bidirectional",
                    "ports": "inherit",
                    "selector": {
                        "npz": "selector.npz",
                        "sha256": selector_hash,
                        "fields": ["cells"],
                    },
                }
            ],
            "edits": {
                "add": [
                    {
                        "source": {
                            "kind": "copy",
                            "module": "memory",
                            "copy_index": 1,
                            "body_id": 20,
                        },
                        "target": {"kind": "ancestor", "body_id": 40},
                        "count": 4,
                    }
                ],
                "remove": [
                    {
                        "source": {
                            "kind": "copy",
                            "module": "memory",
                            "copy_index": 1,
                            "body_id": 30,
                        },
                        "target": {
                            "kind": "copy",
                            "module": "memory",
                            "copy_index": 1,
                            "body_id": 20,
                        },
                    }
                ],
                "reweight": [
                    {
                        "source": {
                            "kind": "copy",
                            "module": "memory",
                            "copy_index": 1,
                            "body_id": 20,
                        },
                        "target": {
                            "kind": "copy",
                            "module": "memory",
                            "copy_index": 1,
                            "body_id": 30,
                        },
                        "count": 6,
                    }
                ],
            },
        }
    )
    source_hashes = {
        name: _hash(graph.path / name) for name in graph.manifest["artifacts"]
    }
    receipt = compile_blueprint(
        graph, ports, blueprint, tmp_path / "derived", selector_root=tmp_path
    )
    assert source_hashes == {
        name: _hash(graph.path / name) for name in graph.manifest["artifacts"]
    }
    derived = DerivedCircuitGraph(receipt["path"], mmap=False)
    derived.verify_artifacts()
    assert derived.n == 6
    assert derived.body_ids[:4].tolist() == [10, 20, 30, 40]
    assert derived.ancestral_body_ids.tolist() == [10, 20, 30, 40, 20, 30]
    measured = {
        (int(s), int(t)): int(c)
        for s, t, c in zip(*derived.edge_arrays())
        if s < 4 and t < 4
    }
    assert measured == {(0, 1): 2, (2, 1): 5, (1, 2): 3, (0, 3): 11, (2, 3): 7}
    edges = {(int(s), int(t)): int(c) for s, t, c in zip(*derived.edge_arrays())}
    assert edges[(0, 4)] == 2  # measured incoming boundary cloned
    assert edges[(4, 5)] == 6  # cloned internal edge, explicitly reweighted
    assert (5, 4) not in edges  # cloned internal reverse edge, explicitly removed
    assert edges[(5, 3)] == 7  # measured outgoing boundary cloned
    assert edges[(4, 3)] == 4  # explicit synthetic addition
    with np.load(
        Path(receipt["path"]) / "derived-edges.npz", allow_pickle=False
    ) as provenance:
        pairs = list(
            zip(
                provenance["source_indices"].tolist(),
                provenance["target_indices"].tolist(),
                provenance["basis_codes"].tolist(),
                strict=True,
            )
        )
        assert (4, 3, 2) in pairs
        assert (4, 5, 3) in pairs
        assert (5, 3, 1) in pairs
    inherited = NeuralPortBundle.load(Path(receipt["path"]) / "ports.npz", derived)
    assert inherited.input_map.shape == (6, 1)
    assert inherited.readout_map.shape == (1, 6)
    assert np.allclose(inherited.readout_map.toarray(), [[0, 0, 0.5, 0, 0, 0.5]])
    # One actual sparse rate update reaches both duplicate cells.
    rate = np.zeros(6, np.float32)
    rate[0] = 0.8
    recurrent = derived.matrix(normalized=True, signed=True)
    for _ in range(4):
        rate += np.float32(0.15625) * (
            np.maximum(np.tanh(0.005 + 0.92 * (recurrent @ rate)), 0) - rate
        )
    assert rate[4] > 0 and rate[5] > 0
