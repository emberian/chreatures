"""A lost save reply must retain local private state and the prior checkpoint."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from chreatures.checkpoint import canonical
from chreatures.neural_client import NeuralServiceError
from chreatures.runtime3d import Habitat3D


def test_interrupted_save_preserves_complete_local_state(tmp_path):
    habitat = Habitat3D.__new__(Habitat3D)
    for name in (
        "field", "resources", "resource_state", "biosphere", "acoustics",
        "acoustic_state", "vision", "motor_artifact", "pending_step",
    ):
        setattr(habitat, name, None)
    for name in (
        "last_senses", "organs", "foresights", "outcomes", "feature_mean",
        "feature_variance", "history",
    ):
        setattr(habitat, name, {})
    for name, value in {
        "id": "test-world", "tick": 42, "branch": "research", "paused": False,
        "speed": 1, "body_mode": "articulated", "physics_backend": "vectorized",
        "execution_migrations": [], "sensed_at": 2.1, "personal_memory": True,
        "personal_plasticity": True, "remote_ids": {"a": "test-world:a"},
        "neural_state": {"a": {"time": 2.1}}, "journal": [], "saved_at": 123,
    }.items():
        setattr(habitat, name, value)
    habitat.world = SimpleNamespace(snapshot=lambda: {"time": 2.1, "rng": [3, 4]})
    habitat.visitor = SimpleNamespace(snapshot=lambda: {"pending": ["cue-at-3"]})
    habitat.motors = {"a": SimpleNamespace(
        snapshot_value=lambda: {"private_weights": [0.2, 0.7], "rng": [9, 8]},
    )}

    def lost_snapshot(name, residents):
        assert name == "world-test-world-42"
        assert residents == ["test-world:a"]
        raise NeuralServiceError("response lost")

    habitat.neural = SimpleNamespace(
        snapshot=lost_snapshot, next_seq=51, service_incarnation="a" * 32,
        input_names=["sensory"], output_names=["population"],
        url="http://127.0.0.1:1", graph={"sha256": "b" * 64},
    )
    path = tmp_path / "world.json"
    path.write_bytes(b"previous complete checkpoint")
    with pytest.raises(NeuralServiceError, match="response lost"):
        habitat.save(path)
    assert path.read_bytes() == b"previous complete checkpoint"
    assert habitat.saved_at == 123
    value = json.loads((tmp_path / "world.interrupted-42.json").read_text())
    assert value["format"] == "chreatures-3d-interrupted-save-v1"
    assert value["sha256"] == hashlib.sha256(canonical(value["state"])).hexdigest()
    state = value["state"]
    assert state["world"]["motors"]["a"]["rng"] == [9, 8]
    assert state["world"]["world"]["rng"] == [3, 4]
    assert state["world"]["visitor"]["pending"] == ["cue-at-3"]
    assert state["neural_request"]["seq"] == 51
    assert "neural_snapshot" not in state["world"]
