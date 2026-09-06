#!/usr/bin/env python3
"""Operate, consume, remove, and exactly restore one finite physical packet."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from chreatures.material_objects import MaterialObjects
from chreatures.metabolism import Chemistry, MetabolicWeb
from chreatures.physics import PhysicsWorld


ROOT = Path(__file__).resolve().parents[1]


class ContactRecorder:
    """Enable and retain PhysicsWorld's physiology-grade contact samples."""

    def __init__(self, world: PhysicsWorld):
        self.world = world
        self.samples: list[dict] = []

    def begin_step(self, actions: dict, dt: float) -> dict[str, float]:
        return {body.id: 1.0 for body in self.world.bodies}

    def finish_step(
        self, actions: dict, outcomes: dict, samples: list[dict], dt: float
    ) -> None:
        self.samples = copy.deepcopy(samples)


def physical_world() -> PhysicsWorld:
    spec = json.loads((ROOT / "data/habitats/hollow-garden.json").read_text())
    spec["name"] = "finite-material-packet-probe"
    spec["size"] = [12.0, 8.0, 3.5]
    spec["bodies"] = [{
        "id": "mica", "name": "Mica", "position": [0.9, 1.0, 0.11],
        "heading": 0.0, "material": "mica", "energy": 0.8, "gut": 0.1,
        "fatigue": 0.04,
    }]
    spec["entities"] = [
        {
            "id": "ground", "mobility": "static", "material": "soil",
            "physical_material": "earth", "position": [1.5, 1.0, -0.08],
            "shapes": [{"type": "box", "size": [1.5, 1.0, 0.08]}],
            "components": [],
        },
        {
            "id": "material-packet", "mobility": "free", "material": "berry",
            "physical_material": "light", "position": [1.035, 1.0, 0.09],
            "shapes": [{"type": "sphere", "size": [0.08]}],
            "components": [],
        },
    ]
    spec.pop("assemblies", None)
    return PhysicsWorld(seed=271, spec=spec)


def chemical_web() -> MetabolicWeb:
    chemistry = Chemistry.load(ROOT / "data/metabolism/common-chemistry.json")
    return MetabolicWeb(
        chemistry,
        [{}, {}, {}],
        [{"soft_tissue": 0.8, "detritus": 0.2}, {}, {}],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )


def contacts(world: PhysicsWorld) -> list[dict]:
    recorder = ContactRecorder(world)
    world.bind_physiology(recorder)
    for _ in range(16):
        world.advance({"mica": {}}, 0.01)
        samples = recorder.samples
        if any("material-packet" in sample["entity_ids"] for sample in samples):
            return copy.deepcopy(samples)
    raise RuntimeError("resident never made physical contact with packet")


def transfer_remaining(
    world: PhysicsWorld, web: MetabolicWeb, materials: MaterialObjects
) -> dict:
    assert materials.contact_entities("mica", contacts(world)) == ("material-packet",)
    remaining = {
        name: float(web.pools[0, index])
        for index, name in enumerate(web.chemistry.pools)
        if web.pools[0, index] > 0.0
    }
    return materials.withdraw_batch([
        {"entity": "material-packet", "receiver_row": 1, "resources": remaining},
        {"entity": "material-packet", "receiver_row": 2, "resources": remaining},
    ])


def main() -> None:
    world = physical_world()
    web = chemical_web()
    materials = MaterialObjects(
        world, web, ROOT / "data/materials/finite-packet-v1.json"
    )
    initial_totals = web.totals()
    initial_mass = materials._physical_mass("material-packet")
    initial_cue = materials.surface_cues()[0]

    proposals = materials.acquisition_proposals(
        "mica", 1, contacts(world), {"soft_tissue": 0.3, "detritus": 0.05}
    )
    assert len(proposals) == 1 and proposals[0]["contact_resident"] == "mica"
    partial = materials.commit(proposals[0])
    partial_mass = materials._physical_mass("material-packet")
    assert partial["boundary_after"] == 1
    np.testing.assert_allclose(partial_mass / initial_mass, 0.78 ** 3, rtol=1e-13)
    assert web.pools[0].tolist() == [0.0, 0.0, 0.0, 0.5, 0.0, 0.15000000000000002]
    assert web.pools[1].tolist() == [0.0, 0.0, 0.0, 0.3, 0.0, 0.05]
    partial_cue = materials.surface_cues()[0]
    assert partial_cue != initial_cue

    atomic_before = (world.snapshot(), web.snapshot(), materials.snapshot())
    try:
        materials.withdraw_batch([{
            "entity": "material-packet", "receiver_row": 1,
            "resources": {"soft_tissue": float("nan")},
        }])
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite material request was accepted")
    assert (world.snapshot(), web.snapshot(), materials.snapshot()) == atomic_before
    materials_before_noop = materials.snapshot()
    assert materials.withdraw_batch([])["moved_resources"] == []
    assert materials.snapshot() == materials_before_noop

    # Joined world, web, and derivation state restore before the next contact.
    joined = json.loads(json.dumps({
        "world": world.snapshot(), "web": web.snapshot(),
        "materials": materials.snapshot(),
    }))
    restored_world = PhysicsWorld.restore(joined["world"])
    restored_web = MetabolicWeb.restore(joined["web"])
    restored_materials = MaterialObjects.restore(
        restored_world, restored_web, joined["materials"]
    )
    assert restored_materials.snapshot() == joined["materials"]

    exhausted = transfer_remaining(world, web, materials)
    restored_exhausted = transfer_remaining(
        restored_world, restored_web, restored_materials
    )
    assert exhausted == restored_exhausted
    assert "material-packet" not in {entity["id"] for entity in world._entities}
    assert np.all(web.pools[0] == 0.0)
    np.testing.assert_array_equal(web.pools[1], web.pools[2] + np.asarray(
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.05]
    ))
    assert np.array_equal(web.pools, restored_web.pools)
    assert world.snapshot() == restored_world.snapshot()
    assert materials.snapshot() == restored_materials.snapshot()
    assert web.snapshot() == restored_web.snapshot()

    final_totals = web.totals()
    elemental_residual = {
        name: final_totals["elements"][name] - amount
        for name, amount in initial_totals["elements"].items()
    }
    assert max(map(abs, elemental_residual.values())) < 1e-12
    assert abs(final_totals["stored_energy"] - initial_totals["stored_energy"]) < 1e-12
    print(json.dumps({
        "physical_contact_proved": True,
        "partial_receipt": partial,
        "initial_physical_mass": initial_mass,
        "partial_physical_mass": partial_mass,
        "mass_ratio": partial_mass / initial_mass,
        "initial_surface_cue": initial_cue,
        "partial_surface_cue": partial_cue,
        "exhausted_receipt": exhausted,
        "packet_removed": True,
        "receiver_pools": [
            dict(zip(web.chemistry.pools, row.tolist())) for row in web.pools[1:]
        ],
        "elemental_residual": elemental_residual,
        "stored_energy_residual": final_totals["stored_energy"] - initial_totals["stored_energy"],
        "invalid_batch_atomic": True,
        "same_runtime_continuation_exact": True,
    }, indent=2))


if __name__ == "__main__":
    main()
