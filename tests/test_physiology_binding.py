import copy

import numpy as np

from chreatures.physical_batch import FastArticulatedSensoriumWorld


class Owner:
    def __init__(self, scales):
        self.scales = scales
        self.finished = []

    def begin_step(self, actions, dt):
        self.begun = (actions, dt)
        return self.scales.copy()

    def finish_step(self, actions, outcomes, contact_samples, dt):
        self.finished.append((copy.deepcopy(actions), copy.deepcopy(outcomes), contact_samples, dt))
        for outcome in outcomes.values():
            outcome["nutrition"] = 0.125


def test_bound_physiology_scales_active_forces_and_rebinds_after_restore():
    habitat = "data/habitats/reef-garden.json"
    active = FastArticulatedSensoriumWorld(seed=51, spec=habitat)
    passive = FastArticulatedSensoriumWorld(seed=51, spec=habitat)
    ids = {body.id for body in active.bodies}
    active_owner = Owner({body_id: 0.0 for body_id in ids})
    passive_owner = Owner({body_id: 0.0 for body_id in ids})
    active.bind_physiology(active_owner)
    passive.bind_physiology(passive_owner)
    energetic = {
        body_id: {"forward": 1.0, "turn": -0.8, "signal_high": 1.0}
        for body_id in ids
    }
    quiet = {body_id: {"forward": 0.0, "turn": 0.0} for body_id in ids}
    active_outcome = active.advance(energetic, 0.05)
    passive.advance(quiet, 0.05)
    np.testing.assert_array_equal(active.data.qpos, passive.data.qpos)
    np.testing.assert_array_equal(active.data.qvel, passive.data.qvel)
    assert all(value["mechanical_work"] == 0.0 for value in active_outcome.values())
    assert all(value["nutrition"] == 0.125 for value in active_outcome.values())
    assert active.signals == []
    samples = active_owner.finished[0][2]
    assert samples
    assert all(set(sample) == {
        "resident_id", "participant_resident_ids", "geom_ids", "geom_names",
        "entity_ids", "entity_shape_indices", "point",
    } for sample in samples)

    snapshot = active.snapshot()
    restored = FastArticulatedSensoriumWorld.restore(snapshot)
    assert restored._physiology is None
    restored.bind_physiology(Owner({body_id: 1.0 for body_id in ids}))
    outcome = restored.advance(energetic, 0.05)
    assert any(value["mechanical_work"] > 0.0 for value in outcome.values())
    assert len(restored.signals) == len(ids)
    assert all(signal.strength == 1.0 for signal in restored.signals)

    half_funded = FastArticulatedSensoriumWorld(seed=52, spec=habitat)
    half_funded.bind_physiology(Owner({body_id: 0.5 for body_id in ids}))
    half_funded.advance({body_id: {"signal_mid": 0.8} for body_id in ids}, 0.05)
    assert all(signal.strength == 0.4 for signal in half_funded.signals)
