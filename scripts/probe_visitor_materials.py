#!/usr/bin/env python3
"""Offer finite chemistry at a real mouth contact and replay it exactly."""

from __future__ import annotations

import json

import numpy as np

from chreatures.biosphere import Biosphere
from chreatures.checkpoint import canonical
from chreatures.physics import PhysicsWorld
from chreatures.visitor_materials import VisitorMaterialSupply
from scripts.build_chemical_habitat import (
    configure,
    configure_visitor_material_supply,
)


def advance(world, biosphere, body_id, steps):
    observations = []
    for _ in range(steps):
        outcomes = world.advance({body_id: {"eat": 1.0}}, 0.05)
        report = biosphere.advance(0.05)
        observations.append(
            {
                "outcomes": outcomes,
                "biosphere": report,
                "mobile": dict(biosphere.mobility.last[body_id]),
            }
        )
    return observations


def main() -> None:
    habitat, birth = configure()
    supply_spec = configure_visitor_material_supply(habitat, birth)
    world = PhysicsWorld(seed=409, spec=habitat)
    biosphere = Biosphere.from_config(world, birth)
    supply = VisitorMaterialSupply(biosphere, supply_spec)
    body_id = "mica"
    position = biosphere.mobility._mouth(body_id).tolist()

    atomic = {
        "world": world.snapshot(),
        "biosphere": biosphere.snapshot(),
        "supply": supply.snapshot(),
    }
    try:
        supply.command(
            {
                "op": "offer_material",
                "material": "reserve-fruit",
                "x": float("nan"),
                "y": position[1],
                "z": position[2],
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite visitor placement was accepted")
    assert atomic == {
        "world": world.snapshot(),
        "biosphere": biosphere.snapshot(),
        "supply": supply.snapshot(),
    }

    totals_before = biosphere.web.totals()
    receipt = supply.command(
        {
            "op": "offer_material",
            "material": "reserve-fruit",
            "x": position[0],
            "y": position[1],
            "z": position[2],
        }
    )
    np.testing.assert_allclose(
        world._pose(receipt["slot"])[0], np.asarray(position), rtol=0, atol=1e-9
    )
    totals_after_offer = biosphere.web.totals()
    np.testing.assert_allclose(
        list(totals_before["elements"].values()),
        list(totals_after_offer["elements"].values()),
        rtol=0,
        atol=1e-14,
    )
    assert abs(
        totals_before["stored_energy"] - totals_after_offer["stored_energy"]
    ) < 1e-14

    material_row = biosphere.materials.donor_rows[receipt["slot"]]
    gut_row = biosphere.mobility.residents[body_id]["gut_row"]
    packet_before = biosphere.web.pools[material_row].copy()
    gut_before = biosphere.web.pools[gut_row].copy()
    contact = advance(world, biosphere, body_id, 1)[0]
    packet_after = biosphere.web.pools[material_row].copy()
    gut_after = biosphere.web.pools[gut_row].copy()
    assert contact["mobile"]["mouth_material_contacts"] > 0
    assert contact["mobile"]["ingested_mass"] > 0
    assert np.any(packet_after < packet_before)
    assert np.any(gut_after > gut_before)

    encoded = canonical(
        {
            "world": world.snapshot(),
            "biosphere": biosphere.snapshot(),
            "supply": supply.snapshot(),
        }
    )
    checkpoint = json.loads(encoded)
    restored_world = PhysicsWorld.restore(checkpoint["world"])
    restored_biosphere = Biosphere.restore(
        restored_world, checkpoint["biosphere"]
    )
    restored_supply = VisitorMaterialSupply.restore(
        restored_biosphere, checkpoint["supply"]
    )
    assert restored_world.snapshot() == world.snapshot()
    assert restored_biosphere.snapshot() == biosphere.snapshot()
    assert restored_supply.snapshot() == supply.snapshot()

    continued = advance(world, biosphere, body_id, 3)
    restored_continued = advance(
        restored_world, restored_biosphere, body_id, 3
    )
    assert restored_continued == continued
    assert restored_world.snapshot() == world.snapshot()
    assert restored_biosphere.snapshot() == biosphere.snapshot()
    assert restored_supply.snapshot() == supply.snapshot()

    accounting = supply.accounting()["reserve-fruit"]
    print(
        json.dumps(
            {
                "format": "chreatures-visitor-material-probe-v1",
                "choice": receipt["choice"],
                "physical_slot": receipt["slot"],
                "spawn_position": receipt["position"],
                "outside_boundary": receipt["outside_boundary"],
                "remaining_source_resources": accounting["remaining_resources"],
                "web_transfer_elemental_residual": receipt["material_receipt"][
                    "elemental_residual"
                ],
                "web_transfer_energy_residual": receipt["material_receipt"][
                    "stored_energy_residual"
                ],
                "actual_mouth_contacts": contact["mobile"][
                    "mouth_material_contacts"
                ],
                "contact_ingested_mass": contact["mobile"]["ingested_mass"],
                "packet_inventory_decreased": True,
                "gut_inventory_increased": True,
                "nonfinite_command_atomic": True,
                "canonical_restore_exact": True,
                "continuation_steps": 3,
                "continuation_exact": True,
                "scope": (
                    "Fresh non-neural chemical world. The outside reserve, "
                    "physical packet, and resident gut share one MetabolicWeb."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
