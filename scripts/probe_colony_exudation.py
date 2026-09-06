#!/usr/bin/env python3
"""Assay light-funded colony exudation, replay, and physical mouth uptake."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.biosphere import Biosphere
from chreatures.physics import PhysicsWorld
from scripts.build_chemical_habitat import configure


def _source_world(*, dark=False, secretion=True):
    habitat, birth = configure(recycling=True)
    # Causal source assay: no founder reserve can fund a packet. All reserve
    # subsequently present was produced by the native reaction network.
    for colony in birth["colonies"]:
        birth["compartments"][colony["body_row"]]["pools"]["reserve"] = 0.0
        if dark:
            # The schema requires positive flux; this declared near-dark value
            # is twelve orders below the matched daylight source.
            colony["photon_flux"] = 1e-12
    for emitter in birth["exchange"]["emitters"]:
        emitter["reserve_floors"]["reserve"] = 0.0
    if not secretion:
        birth["exchange"]["emitters"] = []
    world = PhysicsWorld(seed=906, spec=habitat)
    sphere = Biosphere.from_config(world, birth)
    return world, sphere, birth


def _advance(world, sphere, steps, actions=None):
    outcomes = None
    for _ in range(steps):
        outcomes = world.advance(actions or {}, 0.05)
        sphere.advance(0.05)
    return outcomes


def _summary(world, sphere, birth):
    names = list(sphere.web.chemistry.pools)
    reserve = names.index("reserve")
    emitter_slots = [
        slot for emitter in birth["exchange"]["emitters"]
        for slot in emitter["deposit_slots"]
    ]
    return {
        "captured_photons": sphere.accounting()["captured_photons"],
        "emitted": copy.deepcopy(sphere.exchange.emitted),
        "emitted_reserve": float(sum(row[reserve] for row in sphere.exchange.emitted.values())),
        "active_emitter_slots": [slot for slot in emitter_slots if slot in world._entity_mj],
        "available_slot_chemistry": {
            slot: sphere.web.pools[sphere.materials.donor_rows[slot]].tolist()
            for slot in emitter_slots if slot in world._entity_mj
        },
        "colony_body_reserve": {
            colony["id"]: float(sphere.web.pools[colony["body_row"], reserve])
            for colony in birth["colonies"]
        },
        "accounting": sphere.accounting(),
    }


def _move_to_mouth(world, sphere, slot, resident):
    position = sphere.mobility._mouth(resident)
    body = world._entity_mj[slot]
    joint = world.model.body_jntadr[body]
    qpos = world.model.jnt_qposadr[joint]
    dof = world.model.jnt_dofadr[joint]
    world.data.qpos[qpos : qpos + 3] = position
    world.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(world.model, world.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/colony-exudation-v1",
    )
    args = parser.parse_args()
    trials = {}
    light_world = light_sphere = None
    for name, options in {
        "daylight_secretion": {},
        "dark_secretion": {"dark": True},
        "daylight_no_secretion": {"secretion": False},
    }.items():
        world, sphere, birth = _source_world(**options)
        initially_active = [
            slot for emitter in birth["exchange"]["emitters"]
            for slot in emitter["deposit_slots"] if slot in world._entity_mj
        ]
        assert not initially_active
        _advance(world, sphere, 400)
        trials[name] = _summary(world, sphere, birth)
        if name == "daylight_secretion":
            light_world, light_sphere = world, sphere

    assert trials["daylight_secretion"]["captured_photons"] > 0
    assert trials["daylight_secretion"]["emitted_reserve"] > 0
    assert trials["dark_secretion"]["captured_photons"] < 1e-10
    assert trials["dark_secretion"]["emitted_reserve"] == 0
    assert trials["daylight_no_secretion"]["emitted_reserve"] == 0
    assert sum(trials["daylight_no_secretion"]["colony_body_reserve"].values()) > 0

    saved = {"world": light_world.snapshot(), "biosphere": light_sphere.snapshot()}
    restored_world = PhysicsWorld.restore(saved["world"])
    restored_sphere = Biosphere.restore(restored_world, saved["biosphere"])
    _advance(light_world, light_sphere, 5)
    _advance(restored_world, restored_sphere, 5)
    continuation_exact = (
        light_world.snapshot() == restored_world.snapshot()
        and light_sphere.snapshot() == restored_sphere.snapshot()
    )
    assert continuation_exact

    slot = trials["daylight_secretion"]["active_emitter_slots"][0]
    resident = "mica"
    donor = light_sphere.materials.donor_rows[slot]
    reserve_index = light_sphere.web.chemistry.pools.index("reserve")
    packet_before = float(light_sphere.web.pools[donor, reserve_index])
    gut = light_sphere.mobility.residents[resident]["gut_row"]
    gut_before = float(light_sphere.web.pools[gut, reserve_index])
    _move_to_mouth(light_world, light_sphere, slot, resident)
    contacts = 0
    ingested = 0.0
    for _ in range(2):
        outcomes = light_world.advance({resident: {"eat": 1.0}}, 0.05)
        contacts += int(outcomes[resident]["mouth_material_contacts"])
        ingested += float(light_sphere.mobility.last[resident]["ingested_mass"])
        light_sphere.advance(0.05)
    gut_after = float(light_sphere.web.pools[gut, reserve_index])
    packet_after = float(light_sphere.web.pools[donor, reserve_index])
    assert contacts > 0 and ingested > 0 and gut_after > gut_before
    assert packet_after < packet_before

    for trial in trials.values():
        assert max(abs(value) for value in trial["accounting"]["elemental_residual"].values()) < 1e-10
        assert abs(trial["accounting"]["energy_residual"]) < 1e-10
    report = {
        "format": "chreatures-colony-exudation-assay-v1",
        "scope": (
            "Supplied secretion from native chemistry; zero founder reserve in the "
            "source assay and engineered mouth contact. No learned foraging claim."
        ),
        "trials": trials,
        "continuation_exact": continuation_exact,
        "mouth_assay": {
            "resident": resident, "emitted_slot": slot,
            "mouth_contacts": contacts, "ingested_mass": ingested,
            "packet_reserve_before": packet_before,
            "packet_reserve_after": packet_after,
            "gut_reserve_before": gut_before, "gut_reserve_after": gut_after,
        },
        "sources": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "chreatures/ecological_exchange.py",
                "chreatures/material_objects.py",
                "chreatures/biosphere.py",
                "scripts/build_chemical_habitat.py",
                "scripts/probe_colony_exudation.py",
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
