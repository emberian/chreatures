#!/usr/bin/env python3
"""Spawn, deplete, recycle, and exactly replay one dormant material slot."""

from __future__ import annotations

import copy
import json

import numpy as np

from chreatures.material_objects import MaterialObjects
from chreatures.metabolism import Chemistry, MetabolicWeb
from chreatures.physics import PhysicsWorld
from probe_material_objects import ROOT, physical_world


def dormant_system() -> tuple[PhysicsWorld, MetabolicWeb, MaterialObjects]:
    world = physical_world()
    template = copy.deepcopy(world._entity("material-packet"))
    world.prepare_topology_batch([{"op": "remove", "id": "material-packet"}]).commit()
    chemistry = Chemistry.load(ROOT / "data/metabolism/common-chemistry.json")
    web = MetabolicWeb(
        chemistry,
        [{}, {}, {}, {}],
        [
            {},
            {"soft_tissue": 0.6, "detritus": 0.2},
            {"soft_tissue": 0.6, "detritus": 0.2},
            {},
        ],
        [0.0] * 4,
        [0.0] * 4,
    )
    capacity = {name: 0.0 for name in chemistry.pools}
    capacity.update({"soft_tissue": 0.6, "detritus": 0.2})
    config = {
        "format": "chreatures-material-objects-v1",
        "chemistry_sha256": chemistry.sha256,
        "max_transfer": 1.0,
        "objects": [{
            "entity": "material-packet",
            "row": 0,
            "capacities": capacity,
            "content_weights": {"soft_tissue": 1.0, "detritus": 1.0},
            "remove_when_empty": True,
            "boundaries": [
                {"minimum_content": 0.7, "scale": 1.0},
                {"minimum_content": 0.3, "scale": 0.78},
                {"minimum_content": 0.0, "scale": 0.55},
            ],
            "surface": {
                "rgb_bias": [0.06, 0.03, 0.02],
                "rgb_coefficients": {
                    "soft_tissue": [0.9, 0.18, 0.12],
                    "detritus": [0.22, 0.16, 0.08],
                },
                "odor_coefficients": {
                    "soft_tissue": [1.3, 0.08, 0.0],
                    "detritus": [0.0, 0.55, 0.12],
                },
            },
            "dormant_template": template,
        }],
    }
    return world, web, MaterialObjects(world, web, config)


def exact_resources(web: MetabolicWeb, row: int) -> dict[str, float]:
    return {
        name: float(web.pools[row, index])
        for index, name in enumerate(web.chemistry.pools)
        if web.pools[row, index] > 0.0
    }


def continuation(
    world: PhysicsWorld, web: MetabolicWeb, materials: MaterialObjects
) -> list[dict]:
    receipts = []
    partial = materials.prepare_withdraw(
        "material-packet", 3, {"soft_tissue": 0.2, "detritus": 0.05}
    )
    receipts.append(materials.commit(partial))
    assert receipts[-1]["boundary_after"] == 1

    receipts.append(materials.withdraw_batch([{
        "entity": "material-packet",
        "receiver_row": 3,
        "resources": exact_resources(web, 0),
    }]))
    assert "material-packet" not in materials._existing_ids()
    assert np.all(web.pools[0] == 0.0)

    second_position = [2.4, 1.55, 0.18]
    receipts.append(materials.deposit_batch([{
        "entity": "material-packet",
        "donor_row": 3,
        "resources": {"soft_tissue": 0.3, "detritus": 0.1},
        "position": second_position,
    }]))
    assert receipts[-1]["changes"][0]["spawn_position"] == second_position
    np.testing.assert_array_equal(world._pose("material-packet")[0], second_position)
    assert materials._base_entities["material-packet"]["position"] == second_position
    return receipts


def main() -> None:
    world, web, materials = dormant_system()
    initial_totals = web.totals()
    assert "material-packet" not in materials._existing_ids()
    dormant_atomic = (world.snapshot(), web.snapshot(), materials.snapshot())
    try:
        materials.deposit_batch([{
            "entity": "material-packet", "donor_row": 1,
            "resources": {"soft_tissue": 0.1},
            "position": [float("nan"), 1.2, 0.18],
        }])
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite dormant spawn position was accepted")
    assert (world.snapshot(), web.snapshot(), materials.snapshot()) == dormant_atomic
    first_position = [1.55, 1.2, 0.18]
    request = {"soft_tissue": 0.6, "detritus": 0.2}
    spawned = materials.deposit_batch([
        {
            "entity": "material-packet", "donor_row": 1,
            "resources": request, "position": first_position,
        },
        {
            "entity": "material-packet", "donor_row": 2,
            "resources": request, "position": first_position,
        },
    ])
    moved = np.asarray(spawned["moved_resources"])
    blocked = np.asarray(spawned["capacity_blocked_resources"])
    np.testing.assert_array_equal(moved[0], moved[1])
    np.testing.assert_array_equal(blocked[0], blocked[1])
    assert np.any(blocked > 0.0)
    np.testing.assert_allclose(
        web.pools[0], [0.0, 0.0, 0.0, 0.6, 0.0, 0.2], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        web.pools[1], [0.0, 0.0, 0.0, 0.3, 0.0, 0.1], rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(web.pools[1], web.pools[2])
    np.testing.assert_array_equal(world._pose("material-packet")[0], first_position)

    atomic = (world.snapshot(), web.snapshot(), materials.snapshot())
    try:
        materials.deposit_batch([{
            "entity": "material-packet", "donor_row": 1,
            "resources": {"soft_tissue": 0.01}, "position": [3.0, 1.0, 0.2],
        }])
    except ValueError:
        pass
    else:
        raise AssertionError("active material object accepted a teleporting deposit")
    assert (world.snapshot(), web.snapshot(), materials.snapshot()) == atomic

    joined = json.loads(json.dumps({
        "world": world.snapshot(),
        "web": web.snapshot(),
        "materials": materials.snapshot(),
    }))
    restored_world = PhysicsWorld.restore(joined["world"])
    restored_web = MetabolicWeb.restore(joined["web"])
    restored_materials = MaterialObjects.restore(
        restored_world, restored_web, joined["materials"]
    )
    receipts = continuation(world, web, materials)
    restored_receipts = continuation(restored_world, restored_web, restored_materials)
    assert receipts == restored_receipts
    assert world.snapshot() == restored_world.snapshot()
    assert web.snapshot() == restored_web.snapshot()
    assert materials.snapshot() == restored_materials.snapshot()

    totals = web.totals()
    elemental_residual = {
        name: totals["elements"][name] - amount
        for name, amount in initial_totals["elements"].items()
    }
    energy_residual = totals["stored_energy"] - initial_totals["stored_energy"]
    assert max(map(abs, elemental_residual.values())) < 1e-12
    assert abs(energy_residual) < 1e-12
    print(json.dumps({
        "initially_dormant": True,
        "first_spawn_position": first_position,
        "capacity_fair_moved_resources": spawned["moved_resources"],
        "capacity_blocked_resources": spawned["capacity_blocked_resources"],
        "nonfinite_spawn_rejected_atomically": True,
        "active_teleport_rejected_atomically": True,
        "partial_boundary": receipts[0]["boundary_after"],
        "empty_removed": receipts[1]["changes"][0]["boundary_after"] is None,
        "respawn_position": receipts[2]["changes"][0]["spawn_position"],
        "elemental_residual": elemental_residual,
        "stored_energy_residual": energy_residual,
        "replay_exact": True,
    }, indent=2))


if __name__ == "__main__":
    main()
