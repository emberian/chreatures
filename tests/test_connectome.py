import hashlib
import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).parents[1] / "data" / "connectome"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_connectome_artifact_is_internally_consistent():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    with np.load(DATA_DIR / "circuit.npz", allow_pickle=False) as circuit:
        assert set(circuit.files) == {
            "ids", "pre", "post", "count", "sign", "labels", "type",
            "side", "group", "predicted_nt", "effective_nt", "nt_basis",
            "nt_confidence",
        }
        neuron_count = len(circuit["ids"])
        edge_count = len(circuit["pre"])
        assert neuron_count == manifest["counts"]["neurons"] == 6_789
        assert edge_count == manifest["counts"]["edges"] == 564_810
        assert int(circuit["count"].astype(np.int64).sum()) == 1_367_086

        assert circuit["ids"].dtype.kind == "U"
        assert np.all(np.char.isnumeric(circuit["ids"]))
        assert len(np.unique(circuit["ids"])) == neuron_count
        # FlyWire IDs exceed JavaScript's exact-integer range, so strings matter.
        assert min(map(int, circuit["ids"])) > 2**53

        assert circuit["pre"].dtype == np.int32
        assert circuit["post"].dtype == np.int32
        assert circuit["count"].dtype == np.float32
        assert circuit["sign"].dtype == np.float32
        assert len(circuit["post"]) == len(circuit["count"]) == edge_count
        assert circuit["pre"].min() >= 0 and circuit["post"].min() >= 0
        assert circuit["pre"].max() < neuron_count
        assert circuit["post"].max() < neuron_count
        assert np.all(circuit["count"] > 0)
        assert np.all(circuit["count"] == np.floor(circuit["count"]))
        assert set(np.unique(circuit["sign"])) <= {-1.0, 0.0, 1.0}
        for key in (
            "sign", "labels", "type", "side", "group", "predicted_nt",
            "effective_nt", "nt_basis", "nt_confidence",
        ):
            assert len(circuit[key]) == neuron_count

        # Curated literature annotations restore the known cholinergic KC output
        # despite the source top_nt predictor calling almost every KC dopamine.
        kc = circuit["group"] == "KC"
        assert np.sum(circuit["predicted_nt"][kc] == "dopamine") == 5_172
        assert np.all(circuit["effective_nt"][kc] == "acetylcholine")
        assert np.all(circuit["nt_basis"][kc] == "known_nt")
        assert np.all(circuit["sign"][kc] == 1.0)

        observed_groups = dict(
            zip(*np.unique(circuit["group"], return_counts=True), strict=True)
        )
        assert observed_groups == manifest["counts"]["groups"]


def test_neuron_metadata_and_provenance_match_binary_artifact():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    neurons = json.loads((DATA_DIR / "neurons.json").read_text())
    with np.load(DATA_DIR / "circuit.npz", allow_pickle=False) as circuit:
        assert len(neurons) == len(circuit["ids"])
        assert [row["index"] for row in neurons] == list(range(len(neurons)))
        assert [row["root_id"] for row in neurons] == circuit["ids"].tolist()
        assert [row["group"] for row in neurons] == circuit["group"].tolist()
        assert [row["side"] for row in neurons] == circuit["side"].tolist()
        assert [row["effective_nt"] for row in neurons] == circuit["effective_nt"].tolist()
        assert [row["nt_basis"] for row in neurons] == circuit["nt_basis"].tolist()
        np.testing.assert_array_equal(
            np.asarray([row["model_sign"] for row in neurons], dtype=np.float32),
            circuit["sign"],
        )

    assert manifest["dataset"]["materialization_version"] == 783
    assert manifest["sources"]["connectivity"]["revision"] == (
        "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
    )
    assert manifest["sources"]["annotations"]["tag"] == "v3.1.0"
    for name, metadata in manifest["artifacts"].items():
        path = DATA_DIR / name
        assert path.stat().st_size == metadata["bytes"]
        assert _sha256(path) == metadata["sha256"]
