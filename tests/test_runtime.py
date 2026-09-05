import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chreatures.brain import Brain, ROOT
from chreatures.runtime import Habitat
from chreatures.server import create_app


CONNECTOME = ROOT / "data/connectome/circuit.npz"


def test_habitat_checkpoint_continues_exactly_after_reload(tmp_path):
    checkpoint = tmp_path / "resident.json"
    original = Habitat(seed=37)
    original.step(60)
    original.command({"op": "signal", "x": 250.0, "y": 330.0, "tone": 2})
    original.save(checkpoint)
    reloaded = Habitat.load(checkpoint)

    assert reloaded.snapshot() == original.snapshot()
    original.step(60)
    reloaded.step(60)
    assert reloaded.snapshot() == original.snapshot()


def test_restore_rejects_world_brain_identity_mismatch():
    state = Habitat(seed=8).snapshot()
    state["brains"]["intruder"] = state["brains"].pop("mica")
    with pytest.raises(ValueError, match="identities differ"):
        Habitat.restore(state)


def test_checkpoint_checksum_detects_tampering(tmp_path):
    checkpoint = tmp_path / "resident.json"
    habitat = Habitat(seed=8)
    habitat.step(3)
    habitat.save(checkpoint)
    envelope = json.loads(checkpoint.read_text())
    envelope["state"]["tick"] += 1
    checkpoint.write_text(json.dumps(envelope))

    with pytest.raises(ValueError, match="checksum"):
        Habitat.load(checkpoint)


@pytest.mark.parametrize(
    ("field", "value"),
    [("time", "not-a-number"), ("exploration", "not-a-number"), ("episodes", "not-a-list")],
)
def test_restore_rejects_malformed_brain_scalar_state(field, value):
    state = Habitat(seed=8).snapshot()
    state["brains"]["mica"]["state"][field] = value
    with pytest.raises(ValueError):
        Habitat.restore(state)


def test_restore_rejects_checkpoint_from_different_connectome(tmp_path):
    # A trailing byte leaves the NPZ readable but gives it a distinct identity,
    # modeling a stale or silently replaced anatomical artifact.
    stale = tmp_path / "stale-circuit.npz"
    stale.write_bytes(CONNECTOME.read_bytes() + b"\n")
    state = Habitat(seed=10).snapshot()

    with pytest.raises(ValueError, match="different anatomical scaffold"):
        Habitat.restore(state, connectome_path=stale)


def test_same_path_connectome_replacement_is_not_hidden_by_cache(tmp_path):
    artifact = tmp_path / "circuit.npz"
    artifact.write_bytes(CONNECTOME.read_bytes())
    brain = Brain(artifact, seed=12)
    state = brain.snapshot()
    artifact.write_bytes(artifact.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="different anatomical scaffold"):
        Brain.restore(state, artifact)


def _stimulus():
    return {
        "odor": [[0.9, 0.1, 0.0], [0.05, 0.8, 0.2]],
        "vision": [[0.7, 0.2, 0.1, 0.1]] * 8 + [[0.2, 0.6, 0.1, 0.75]] * 8,
        "touch": [0.0, 0.0],
        "sound": [0.2, 0.7, 0.0],
        "shade": 0.1,
        "speed": 0.0,
        "angular_velocity": 0.0,
    }


def test_recurrent_silencing_changes_downstream_activity_and_action():
    source = Brain(CONNECTOME, seed=91)
    senses = _stimulus()
    physiology = {"energy": 0.45, "gut": 0.1, "fatigue": 0.1}
    for _ in range(20):
        source.step(senses, physiology, 0.05)
    intact = Brain.restore(source.snapshot(), CONNECTOME)
    silenced = Brain.restore(source.snapshot(), CONNECTOME)
    silenced.silenced = True

    for _ in range(30):
        intact_action = intact.step(senses, physiology, 0.05)
        silenced_action = silenced.step(senses, physiology, 0.05)

    downstream = intact.graph.output_cells
    assert np.linalg.norm(intact.rates[downstream] - silenced.rates[downstream]) > 1.0
    assert np.linalg.norm(intact.last_decoded - silenced.last_decoded) > 0.1
    assert abs(intact_action["forward"] - silenced_action["forward"]) > 0.05


def test_coordinates_do_not_enter_sensory_encoding_or_physiology():
    senses = _stimulus()
    encoded = Brain.encode(senses)
    decorated = {**senses, "x": 999.0, "y": -200.0, "heading": 1.7, "target_id": "food-1"}
    assert np.array_equal(Brain.encode(decorated), encoded)

    habitat = Habitat(seed=14)
    captured = []
    brain = habitat.brains["mica"]
    original_step = brain.step

    def inspect(sensory, physiology, dt, reward=0.0):
        captured.append((set(sensory), set(physiology)))
        return original_step(sensory, physiology, dt, reward)

    brain.step = inspect
    habitat.step()
    sensory_keys, physiology_keys = captured[0]
    assert not sensory_keys & {"x", "y", "heading", "target_id", "object_id"}
    assert physiology_keys == {"energy", "gut", "fatigue"}


def test_headless_trajectory_moves_and_remains_finite():
    habitat = Habitat(seed=7)
    starts = {body.id: (body.x, body.y) for body in habitat.world.bodies}
    habitat.step(240)

    for body in habitat.world.bodies:
        values = [body.x, body.y, body.heading, body.speed, body.angular_velocity,
                  body.energy, body.gut, body.fatigue, body.age]
        assert all(math.isfinite(value) for value in values)
        assert math.dist(starts[body.id], (body.x, body.y)) > 20.0
        brain = habitat.brains[body.id]
        assert np.isfinite(brain.rates).all()
        assert np.isfinite(brain.support).all()
        assert all(math.isfinite(value) for value in brain.last_action.values())


def test_http_state_is_readable_but_commands_are_same_origin(tmp_path):
    checkpoint = tmp_path / "api-resident.json"
    with TestClient(create_app(checkpoint=checkpoint, seed=5, autostep=False)) as client:
        state = client.get("/api/state", headers={"origin": "https://untrusted.example"})
        assert state.status_code == 200
        assert state.json()["tick"] == 0

        denied = client.post(
            "/api/command",
            json={"op": "signal", "x": 100, "y": 100, "tone": 1},
            headers={"origin": "https://untrusted.example"},
        )
        assert denied.status_code == 403

        accepted = client.post(
            "/api/command",
            json={"op": "signal", "x": 100, "y": 100, "tone": 1},
            headers={"origin": "http://testserver"},
        )
        assert accepted.status_code == 200
        assert accepted.json()["ok"] is True
        assert len(client.get("/api/state").json()["signals"]) == 1


@pytest.mark.parametrize(
    ("content", "headers", "status"),
    [
        (b"[]", {"content-type": "application/json"}, 400),
        (b"{", {"content-type": "application/json"}, 400),
        (b'{"op":"signal"}', {"content-type": "text/plain"}, 415),
        (b"x" * 16_385, {"content-type": "application/json"}, 413),
        (b'{"op":"signal","x":"bad","y":20}', {"content-type": "application/json"}, 400),
    ],
)
def test_http_rejects_malformed_commands(tmp_path, content, headers, status):
    checkpoint = tmp_path / "api-invalid.json"
    with TestClient(create_app(checkpoint=checkpoint, seed=6, autostep=False)) as client:
        before = client.get("/api/state").json()
        response = client.post("/api/command", content=content, headers=headers)
        assert response.status_code == status
        after = client.get("/api/state").json()
        assert after["tick"] == before["tick"]
        assert after["objects"] == before["objects"]
        assert after["signals"] == before["signals"]
