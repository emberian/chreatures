import numpy as np

from chreatures.physical_batch import FastArticulatedSensoriumWorld
from chreatures.sensorium import ArticulatedSensoriumWorld

HABITAT = "data/habitats/reef-garden.json"


def test_heterogeneous_morphology_is_physical_exact_and_restorable():
    reference = ArticulatedSensoriumWorld(seed=12, spec=HABITAT)
    fast = FastArticulatedSensoriumWorld(seed=12, spec=HABITAT)
    thoraces = {
        body.id: tuple(
            reference.model.geom_size[
                reference.model.geom(f"resident:{body.id}:geom:thorax").id
            ]
        )
        for body in reference.bodies
    }
    assert len(set(thoraces.values())) == 3
    assert (
        len(
            {
                reference._resident_articulation[body.id]["controller"]["frequency_hz"]
                for body in reference.bodies
            }
        )
        == 3
    )

    for index in range(5):
        actions = {
            body.id: {
                "forward": float(np.sin(index * 0.3 + body_index)),
                "turn": float(np.cos(index * 0.2 - body_index) * 0.4),
            }
            for body_index, body in enumerate(reference.bodies)
        }
        assert reference.advance(actions, 0.05) == fast.advance(actions, 0.05)
        np.testing.assert_array_equal(reference.data.qpos, fast.data.qpos)
        assert {body.id: reference.sense(body.id) for body in reference.bodies} == {
            body.id: fast.sense(body.id) for body in fast.bodies
        }

    snapshot = fast.snapshot()
    restored = FastArticulatedSensoriumWorld.restore(snapshot)
    assert restored.snapshot() == snapshot
    assert set(snapshot["articulated_morphology"]["resident_sha256"]) == {
        "mica",
        "fern",
        "pip",
    }
    assert all(
        restored._fast_joint_kp[body.id].shape == (12,) for body in restored.bodies
    )
