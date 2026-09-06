#!/usr/bin/env python3
"""Real physical egestion, contact-mediated root acquisition and replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.biosphere import Biosphere
from chreatures.physics import PhysicsWorld
from scripts.build_chemical_habitat import configure


def move_deposit(world, position):
    # An explicit positional intervention in a disposable research instance.
    body = world._entity_mj["deposit-0"]
    joint = world.model.body_jntadr[body]
    qpos, dof = world.model.jnt_qposadr[joint], world.model.jnt_dofadr[joint]
    world.data.qpos[qpos : qpos + 3] = position
    world.data.qvel[dof : dof + 6] = 0
    mujoco.mj_forward(world.model, world.data)


def advance(world, biosphere, count):
    for _ in range(count):
        world.advance({}, 0.05)
        biosphere.advance(0.05)


def trial(snapshot, location):
    world = PhysicsWorld.restore(snapshot["world"])
    sphere = Biosphere.restore(world, snapshot["biosphere"])
    part = next(
        part
        for part in sphere.parts.values()
        if part["kind"] == "root" and part["colony"] == "colony-c"
    )
    geom = mujoco.mj_name2id(
        world.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"entity:{part['entity']}:geom:{part['shape_index']}",
    )
    position = world.data.geom_xpos[geom].copy() + [0.0, 0.0, 0.018]
    if location == "separated":
        position[1] += 0.8
    move_deposit(world, position)
    contacts = len(sphere.exchange._root_contacts())
    advance(world, sphere, 1)
    acquired = copy.deepcopy(sphere.exchange.acquired)
    saved = {"world": world.snapshot(), "biosphere": sphere.snapshot()}
    second_world = PhysicsWorld.restore(saved["world"])
    second_sphere = Biosphere.restore(second_world, saved["biosphere"])
    advance(world, sphere, 4)
    advance(second_world, second_sphere, 4)
    exact = (
        world.snapshot() == second_world.snapshot()
        and sphere.snapshot() == second_sphere.snapshot()
    )
    assert exact
    return {
        "location": location,
        "initial_root_material_contacts": contacts,
        "acquired_after_one_tick": acquired,
        "same_runtime_continuation_exact": exact,
        "accounting": sphere.accounting(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/ecological-exchange-v1"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    habitat, birth = configure(recycling=True)
    world = PhysicsWorld(seed=318, spec=habitat)
    sphere = Biosphere.from_config(world, birth)
    assert all(
        slot not in world._entity_mj for slot in birth["exchange"]["deposit_slots"]
    )
    advance(world, sphere, 60)
    snapshot = {"world": world.snapshot(), "biosphere": sphere.snapshot()}
    egested = copy.deepcopy(sphere.exchange.egested)
    assert all(sum(vector) > 0 for vector in egested.values())
    assert all(f"deposit-{index}" in world._entity_mj for index in range(3))
    trials = [trial(snapshot, location) for location in ("contact", "separated")]
    assert trials[0]["initial_root_material_contacts"] > 0
    assert trials[1]["initial_root_material_contacts"] == 0
    assert sum(trials[0]["acquired_after_one_tick"]["colony-c"]) > 0
    assert sum(trials[1]["acquired_after_one_tick"]["colony-c"]) == 0
    for result in trials:
        assert (
            max(abs(v) for v in result["accounting"]["elemental_residual"].values())
            < 1e-10
        )
        assert abs(result["accounting"]["energy_residual"]) < 1e-10
    report = {
        "format": "chreatures-ecological-exchange-assay-v1",
        "scope": "Supplied quiet actions; endogenous deposits then a recorded positional intervention. No learned foraging or cooperation claim.",
        "wall_seconds": time.perf_counter() - started,
        "egested_before_intervention": egested,
        "pools": list(sphere.web.chemistry.pools),
        "trials": trials,
        "sources": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "chreatures/ecological_exchange.py",
                "chreatures/material_objects.py",
                "chreatures/biosphere.py",
                "chreatures/somatic.py",
                "scripts/build_chemical_habitat.py",
                "scripts/probe_ecological_exchange.py",
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
