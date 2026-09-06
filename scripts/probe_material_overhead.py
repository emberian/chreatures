#!/usr/bin/env python3
"""Profile recurring finite-material overhead in an isolated recycling world.

The research-reference branch emulates the scans used before the transient
MaterialObjects caches. It is local to this probe and is never selectable by a
runtime or production world. Whole-world wall time is reported as context, not
as evidence of an end-to-end speedup.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import hashlib
import json
import platform
import pstats
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.biosphere import Biosphere
from chreatures.physical_batch import FastArticulatedSensoriumWorld
from scripts.build_chemical_habitat import configure


PROFILE_NAMES = {"sync_geometry", "surface_cues", "_check_physical_state"}


def _install_research_reference(materials) -> None:
    """Bind prior scan behavior to one disposable instance only."""

    def item(self, entity_id):
        try:
            return next(
                value
                for value in self.config["objects"]
                if value["entity"] == entity_id
            )
        except StopIteration as exc:
            raise ValueError("unknown material entity") from exc

    def expected_entity(self, entity_id, boundary_index):
        value = self._item(entity_id)
        return self._scaled_entity(
            self._base_entities[entity_id], value["boundaries"][boundary_index]
        )

    def check_physical_state(self):
        existing = {entity["id"] for entity in self.world._entities}
        for entity_id, state in self._state.items():
            if state["active"]:
                if (
                    entity_id not in existing
                    or self._world_entity(entity_id)
                    != self._expected_entity(entity_id, state["boundary"])
                ):
                    raise ValueError("material state and physical geometry differ")
            elif entity_id in existing:
                raise ValueError("exhausted material entity remains physical")

    def sync_geometry(self):
        web = self._web()
        operations = []
        changes = []
        for value in self.config["objects"]:
            entity_id = value["entity"]
            state = self._state[entity_id]
            capacity = np.asarray(
                [
                    value["capacities"].get(name, 0.0)
                    for name in web.chemistry.pools
                ]
            )
            if np.any(web.pools[value["row"]] > capacity):
                raise ValueError("material inventory exceeds declared capacity")
            boundary = self._boundary(value, web.pools[value["row"]])
            if (
                state["boundary"] is None
                and boundary is not None
                and value.get("dormant_template") is not None
            ):
                raise ValueError(
                    "dormant material activation requires positioned deposit_batch"
                )
            operation = self._topology_operation(
                entity_id, state["boundary"], boundary
            )
            if operation is not None:
                operations.append(operation)
                changes.append(
                    {
                        "entity": entity_id,
                        "boundary_before": state["boundary"],
                        "boundary_after": boundary,
                    }
                )
        if not operations:
            self._check_physical_state()
            return []
        transaction = self.world.prepare_topology_batch(operations)
        transaction.commit()
        for change in changes:
            self._state[change["entity"]] = {
                "active": change["boundary_after"] is not None,
                "boundary": change["boundary_after"],
            }
        self.geometry_syncs += 1
        self.last_geometry_sync = copy.deepcopy(changes)
        self._check_physical_state()
        return changes

    def surface_cues(self):
        web = self._web()
        names = web.chemistry.pools
        result = []
        for value in self.config["objects"]:
            if not self._state[value["entity"]]["active"]:
                continue
            pools = web.pools[value["row"]]
            surface = value["surface"]
            rgb = np.asarray(surface["rgb_bias"], dtype=np.float64)
            odor = np.zeros(3, dtype=np.float64)
            for name, coefficient in surface["rgb_coefficients"].items():
                rgb += pools[names.index(name)] * np.asarray(coefficient)
            for name, coefficient in surface["odor_coefficients"].items():
                odor += pools[names.index(name)] * np.asarray(coefficient)
            result.append(
                {
                    "entity": value["entity"],
                    "rgb": np.clip(rgb, 0.0, 1.0).tolist(),
                    "odor": np.clip(odor, 0.0, 4.0).tolist(),
                }
            )
        return result

    bindings = {
        "_item": item,
        "_expected_entity": expected_entity,
        "_check_physical_state": check_physical_state,
        "sync_geometry": sync_geometry,
        "surface_cues": surface_cues,
    }
    for name, function in bindings.items():
        setattr(materials, name, types.MethodType(function, materials))


def _advance(world, biosphere, steps: int, dt: float) -> None:
    for _ in range(steps):
        world.advance({}, dt)
        biosphere.advance(dt)


def _profile_branch(base, *, reference: bool, steps: int, dt: float):
    world = FastArticulatedSensoriumWorld.restore(base["world"])
    biosphere = Biosphere.restore(world, base["biosphere"])
    if reference:
        _install_research_reference(biosphere.materials)
    profile = cProfile.Profile()
    profile.enable()
    started = time.perf_counter()
    _advance(world, biosphere, steps, dt)
    elapsed = time.perf_counter() - started
    profile.disable()
    stats = pstats.Stats(profile)
    aliases = {
        "sync_geometry": "sync_geometry",
        "surface_cues": "surface_cues",
        "_check_physical_state": "_check_physical_state",
        "check_physical_state": "_check_physical_state",
        "sync_geometry_reference": "sync_geometry",
        "surface_cues_reference": "surface_cues",
        "check_physical_state_reference": "_check_physical_state",
    }
    cumulative = dict.fromkeys(PROFILE_NAMES, 0.0)
    for (_, _, function_name), values in stats.stats.items():
        key = aliases.get(function_name)
        if key is not None:
            cumulative[key] += float(values[3])
    return {
        "wall_seconds": elapsed,
        "function_calls": stats.total_calls,
        "material_cumulative_seconds": cumulative,
    }, {
        "world": world.snapshot(),
        "biosphere": biosphere.snapshot(),
    }


def _material_loop(base, *, reference: bool, iterations: int):
    world = FastArticulatedSensoriumWorld.restore(base["world"])
    biosphere = Biosphere.restore(world, base["biosphere"])
    if reference:
        _install_research_reference(biosphere.materials)
    started = time.perf_counter()
    cues = None
    for _ in range(iterations):
        if biosphere.materials.sync_geometry():
            raise RuntimeError("unchanged isolated material loop changed geometry")
        cues = biosphere.materials.surface_cues()
    return time.perf_counter() - started, {
        "cues": cues,
        "materials": biosphere.materials.snapshot(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=381)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--material-iterations", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.steps <= 0
        or args.material_iterations <= 0
        or not np.isfinite(args.dt)
        or not 0.0 < args.dt <= 1.0
    ):
        raise ValueError("profile parameters are outside their bounds")

    habitat, birth = configure(recycling=True)
    world = FastArticulatedSensoriumWorld(seed=args.seed, spec=habitat)
    biosphere = Biosphere.from_config(world, birth)
    _advance(world, biosphere, args.warmup, args.dt)
    base = {"world": world.snapshot(), "biosphere": biosphere.snapshot()}

    reference_profile, reference_state = _profile_branch(
        base, reference=True, steps=args.steps, dt=args.dt
    )
    optimized_profile, optimized_state = _profile_branch(
        base, reference=False, steps=args.steps, dt=args.dt
    )
    reference_loop, reference_observable = _material_loop(
        base, reference=True, iterations=args.material_iterations
    )
    optimized_loop, optimized_observable = _material_loop(
        base, reference=False, iterations=args.material_iterations
    )

    source_paths = [
        "chreatures/material_objects.py",
        "chreatures/biosphere.py",
        "chreatures/ecological_exchange.py",
        "chreatures/physical_batch.py",
        "scripts/build_chemical_habitat.py",
        "scripts/probe_material_overhead.py",
    ]
    report = {
        "format": "chreatures-material-overhead-probe-v1",
        "scope": (
            "Fresh isolated recycling world; no neural controller and no live "
            "runtime or service."
        ),
        "research_reference": (
            "One disposable instance emulates the prior unconditional scans. "
            "This branch is not production-selectable."
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "parameters": {
            "seed": args.seed,
            "dt_seconds": args.dt,
            "warmup_steps": args.warmup,
            "profile_steps": args.steps,
            "isolated_material_iterations": args.material_iterations,
            "developed_parts_after_warmup": len(biosphere.parts),
            "configured_deposit_slots": len(biosphere.exchange.config["deposit_slots"]),
            "active_deposit_slots_after_warmup": sum(
                slot in world._entity_mj
                for slot in biosphere.exchange.config["deposit_slots"]
            ),
        },
        "paired_profile": {
            "research_reference": reference_profile,
            "optimized": optimized_profile,
            "full_world_and_biosphere_snapshots_exact": (
                reference_state == optimized_state
            ),
        },
        "isolated_recurring_calls": {
            "research_reference_seconds": reference_loop,
            "optimized_seconds": optimized_loop,
            "speedup": reference_loop / optimized_loop,
            "returned_cues_and_material_snapshot_exact": (
                reference_observable == optimized_observable
            ),
        },
        "interpretation": {
            "supported_claim": (
                "The exact fast paths reduce recurring MaterialObjects overhead "
                "for this scene."
            ),
            "whole_world_speedup_claim": False,
            "reason": (
                "Shared-host whole-world wall time is noisy and physics plus real "
                "topology compilation dominate."
            ),
        },
        "sources": {path: _sha256(ROOT / path) for path in source_paths},
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
