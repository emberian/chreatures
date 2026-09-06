"""Reproduce the September 2026 native material-packet performance receipt."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from chreatures.biosphere import Biosphere
from chreatures.habitat_family import RegionalHabitatFamily
from chreatures.metabolism import canonical
from chreatures.physical_batch import FastArticulatedSensoriumWorld

ROOT = Path(__file__).resolve().parents[2]
ACTION_NAMES = (
    "thrust",
    "yaw",
    "gaze_pitch",
    "posture",
    "grip",
    "signal_low",
    "signal_mid",
    "signal_high",
    "eat",
    "release",
    "secrete",
    "allocate",
)


def build() -> tuple[FastArticulatedSensoriumWorld, Biosphere]:
    family = RegionalHabitatFamily(
        ROOT / "data/habitat-families/regional-v1.json",
        ROOT / "data/habitats/living-reef.json",
        ROOT / "data/biosphere/living-reef.json",
    )
    residents = json.loads(
        (ROOT / "data/habitat-families/regional-residents-v1.json").read_text()
    )
    genome = family.initial_genome(
        seed=2026090602,
        archetype="terraced-delta",
        resident_count=8,
        profile_sha256="1" * 64,
        epoch=0,
    )
    generated = family.generate(genome, residents)
    world = FastArticulatedSensoriumWorld(seed=37, spec=dict(generated.habitat))
    return world, Biosphere.from_config(world, dict(generated.biosphere))


def actions(world: FastArticulatedSensoriumWorld, *, full: bool) -> dict:
    result = {body.id: dict.fromkeys(ACTION_NAMES, 0.0) for body in world.bodies}
    for row in result.values():
        row["release"] = 0.1
        if full:
            row["secrete"] = 0.2
            row["allocate"] = 0.3
    return result


def disable_preflight(biosphere: Biosphere) -> None:
    # Research-only control: force the existing authoritative compiled check.
    biosphere.materials._guaranteed_spawn_overlaps = lambda *_: set()


def advance(world, biosphere, action, steps: int) -> list[float]:
    elapsed = []
    for _ in range(steps):
        world.advance(action, 0.05)
        started = time.perf_counter()
        biosphere.advance(0.05)
        elapsed.append(time.perf_counter() - started)
    return elapsed


def timed(force_compiled: bool) -> dict[str, float]:
    world, biosphere = build()
    if force_compiled:
        disable_preflight(biosphere)
    action = actions(world, full=True)
    advance(world, biosphere, action, 3)
    elapsed = np.asarray(advance(world, biosphere, action, 20)) * 1_000.0
    return {
        "mean_ms": float(elapsed.mean()),
        "p95_ms": float(np.percentile(elapsed, 95)),
    }


def state_comparison() -> dict[str, object]:
    states = []
    for force_compiled in (False, True):
        world, biosphere = build()
        if force_compiled:
            disable_preflight(biosphere)
        advance(world, biosphere, actions(world, full=False), 90)
        states.append(
            (
                hashlib.sha256(canonical(biosphere.snapshot())).hexdigest(),
                hashlib.sha256(canonical(world.snapshot())).hexdigest(),
            )
        )
    return {
        "identical": states[0] == states[1],
        "biosphere_sha256": states[0][0],
        "world_sha256": states[0][1],
    }


def outlet_position(world, biosphere) -> list[float]:
    key = "mica"
    body = world._body(key)
    body_id = world._body_mj[key]
    local = np.asarray(biosphere.exchange.mobiles[key]["offset_radii"]) * body.radius
    position = world.data.xpos[body_id] + world.data.xmat[body_id].reshape(3, 3) @ local
    return position.tolist()


def deposit_comparison(*, insufficient: bool) -> dict[str, object]:
    states = []
    for force_compiled in (False, True):
        world, biosphere = build()
        if force_compiled:
            disable_preflight(biosphere)
        private = biosphere.mobility.residents["mica"]
        names = biosphere.web.chemistry.pools
        requests = []
        if insufficient:
            donor = private["gut_row"]
            pool = next(
                name
                for name, value in zip(names, biosphere.web.pools[donor], strict=True)
                if value == 0.0
            )
            donors = [(donor, pool, 0.0005)]
        else:
            donors = []
            for donor in (private["gut_row"], private["body_row"]):
                stock = biosphere.web.pools[donor]
                index = next(index for index, value in enumerate(stock) if value > 1e-5)
                donors.append((donor, names[index], 1e-5))
        for donor, pool, amount in donors:
            requests.append(
                {
                    "entity": "living-deposit-00",
                    "donor_row": donor,
                    "resources": {pool: amount},
                    "position": outlet_position(world, biosphere),
                }
            )
        receipt = biosphere.materials.deposit_batch(requests)
        states.append(
            (
                hashlib.sha256(canonical(receipt)).hexdigest(),
                hashlib.sha256(canonical(biosphere.snapshot())).hexdigest(),
                receipt["moved_resources"],
                receipt["receiver_limiter"],
            )
        )
    return {"identical": states[0] == states[1], "optimized": states[0]}


def main() -> None:
    result = {
        "platform": "Apple M2; macOS 26.6.1; Python 3.12.14; MuJoCo 3.12.0",
        "forced_compiled": timed(True),
        "native_preflight": timed(False),
        "state_comparison": state_comparison(),
        "insufficient_donor": deposit_comparison(insufficient=True),
        "same_entity_edges": deposit_comparison(insufficient=False),
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
