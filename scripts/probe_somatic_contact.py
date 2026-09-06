#!/usr/bin/env python3
"""Physical mouth-contact controls for the shared chemical physiology."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chreatures.biosphere import Biosphere
from chreatures.physics import PhysicsWorld
from scripts.build_chemical_habitat import configure


def scenario(location, eat):
    habitat, birth = configure()
    soma = copy.deepcopy(birth["compartments"][birth["mobiles"][0]["body_row"]])
    gut = copy.deepcopy(birth["compartments"][birth["mobiles"][0]["gut_row"]])
    gut["pools"] = {}
    mobile = copy.deepcopy(birth["mobiles"][0])
    mobile.update(body_row=0, gut_row=1)
    packet = copy.deepcopy(birth["material_objects"]["objects"][0])
    packet.update(
        row=2, capacities={"soft_tissue": 0.08}, content_weights={"soft_tissue": 5.0}
    )
    packet["boundaries"] = [
        {"minimum_content": 0.3, "scale": 1.0},
        {"minimum_content": 0.0, "scale": 0.7},
    ]
    packet["surface"] = {
        "rgb_bias": [0.1, 0.05, 0.02],
        "rgb_coefficients": {"soft_tissue": [2.0, 1.0, 0.2]},
        "odor_coefficients": {"soft_tissue": [1.0, 0.1, 0.0]},
    }
    birth.update(
        colonies=[],
        mobiles=[mobile],
        compartments=[
            soma,
            gut,
            {
                "enzymes": {},
                "pools": {"soft_tissue": 0.08},
                "atp": 0.0,
                "atp_capacity": 0.0,
            },
        ],
    )
    birth["material_objects"]["objects"] = [packet]
    habitat["bodies"] = habitat["bodies"][:1]
    habitat["bodies"][0].update(position=[0.5, 1.4, 0.15], heading=0.0)
    habitat["gravity"] = [0, 0, 0]
    habitat["entities"] = [
        e for e in habitat["entities"] if e["id"] in {"ground", "chemical-packet-0"}
    ]
    item = next(e for e in habitat["entities"] if e["id"] == "chemical-packet-0")
    item["position"] = [0.665, 1.4, 0.15] if location == "mouth" else [0.5, 1.53, 0.15]
    world = PhysicsWorld(seed=413, spec=habitat)
    sphere = Biosphere.from_config(world, birth)
    action = {"mica": {"eat": eat}}
    contacts = 0
    for _ in range(80):
        world.advance(action, 0.05)
        contacts += len(world._step_contact_samples)
        sphere.advance(0.05)
    snapshot = {"world": world.snapshot(), "biosphere": sphere.snapshot()}
    restored_world = PhysicsWorld.restore(snapshot["world"])
    restored_sphere = Biosphere.restore(restored_world, snapshot["biosphere"])
    for _ in range(4):
        world.advance(action, 0.05)
        sphere.advance(0.05)
        restored_world.advance(action, 0.05)
        restored_sphere.advance(0.05)
    exact = (
        world.snapshot() == restored_world.snapshot()
        and sphere.snapshot() == restored_sphere.snapshot()
    )
    assert exact
    return {
        "location": location,
        "eat": eat,
        "contact_samples": contacts,
        "physiology": sphere.mobility.view(),
        "remaining_packet": sphere.web.pools[2].tolist(),
        "accounting": sphere.accounting(),
        "same_runtime_continuation_exact": exact,
        "scope": "supplied quiet actions and founding placement; no neural policy or learned foraging",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "runs/somatic-contact-v1")
    a = p.parse_args()
    started = time.perf_counter()
    trials = [scenario("mouth", 1.0), scenario("side", 1.0), scenario("mouth", 0.0)]
    amounts = [
        t["physiology"]["residents"]["mica"]["totals"]["ingested_mass"] for t in trials
    ]
    absorbed = [
        t["physiology"]["residents"]["mica"]["totals"]["absorbed"] for t in trials
    ]
    assert all(t["contact_samples"] > 0 for t in trials)
    assert amounts[0] > 0 and amounts[1:] == [0, 0]
    assert absorbed[0] > 0 and absorbed[1:] == [0, 0]
    assert (
        max(
            abs(v)
            for t in trials
            for v in t["accounting"]["elemental_residual"].values()
        )
        < 1e-10
    )
    assert max(abs(t["accounting"]["energy_residual"]) for t in trials) < 1e-10
    a.output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "chreatures-somatic-contact-assay-v1",
        "wall_seconds": time.perf_counter() - started,
        "trials": trials,
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ingested_mass": amounts,
                "absorbed_reserve": absorbed,
                "wall_seconds": report["wall_seconds"],
                "receipt": str(a.output / "report.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
