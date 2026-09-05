#!/usr/bin/env python3
"""Measure and audit the opt-in articulated physical execution fast path."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.neural_ports import encode_physical_senses, load_port_spec
from chreatures.physical_batch import FastArticulatedSensoriumWorld
from chreatures.sensorium import ArticulatedSensoriumWorld


HABITAT = ROOT / "data/habitats/hollow-garden.json"


def actions(world: ArticulatedSensoriumWorld, step: int) -> dict[str, dict[str, float]]:
    """Deterministic varied controls shared by reference and fast worlds."""
    return {
        body.id: {
            "forward": float(np.sin(step * 0.17 + index)),
            "turn": float(np.cos(step * 0.11 - index) * 0.7),
            "gaze_pitch": float(np.sin(step * 0.07 + index) * 0.8),
            "grip": float((step + index) % 5 == 0),
            "eat": float((step + index) % 3 == 0),
        }
        for index, body in enumerate(world.bodies)
    }


def run_world(kind: str, steps: int, seed: int) -> dict[str, float]:
    cls = ArticulatedSensoriumWorld if kind == "reference" else FastArticulatedSensoriumWorld
    spec = json.loads(HABITAT.read_text(encoding="utf-8"))
    world = cls(seed=seed, spec=spec)
    port_spec = load_port_spec()
    sense_seconds = physics_seconds = 0.0
    for step in range(steps + 3):
        started = time.perf_counter()
        for body in world.bodies:
            encode_physical_senses(world.sense(body.id), port_spec)
        sensed = time.perf_counter()
        world.advance(actions(world, step), 0.05)
        advanced = time.perf_counter()
        if step >= 3:
            sense_seconds += sensed - started
            physics_seconds += advanced - sensed
    return {"sense_seconds": sense_seconds, "physics_seconds": physics_seconds}


def worker(connection: Any, kind: str, steps: int, seed: int) -> None:
    try:
        connection.send(run_world(kind, steps, seed))
    finally:
        connection.close()


def cohort(kind: str, worlds: int, steps: int) -> dict[str, float]:
    context = mp.get_context("spawn")
    connections, processes = [], []
    started = time.perf_counter()
    for index in range(worlds):
        parent, child = context.Pipe(False)
        process = context.Process(target=worker, args=(child, kind, steps, 1000 + index))
        process.start()
        child.close()
        connections.append(parent)
        processes.append(process)
    values = [connection.recv() for connection in connections]
    wall = time.perf_counter() - started
    for connection in connections:
        connection.close()
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"physical benchmark worker exited {process.exitcode}")
    return {
        "wall_seconds": wall,
        "sense_worker_seconds": sum(value["sense_seconds"] for value in values),
        "physics_worker_seconds": sum(value["physics_seconds"] for value in values),
        "resident_steps_per_second": worlds * 3 * steps / wall,
    }


def equivalence(steps: int = 30) -> dict[str, Any]:
    spec = json.loads(HABITAT.read_text(encoding="utf-8"))
    reference = ArticulatedSensoriumWorld(seed=91, spec=spec)
    fast = FastArticulatedSensoriumWorld(seed=91, spec=spec)
    port_spec = load_port_spec()
    maximum_state_error = maximum_sensor_error = 0.0
    for step in range(steps):
        for left, right in zip(reference.bodies, fast.bodies, strict=True):
            left_channels = encode_physical_senses(reference.sense(left.id), port_spec)[1]
            right_channels = encode_physical_senses(fast.sense(right.id), port_spec)[1]
            maximum_sensor_error = max(
                maximum_sensor_error, float(np.max(np.abs(left_channels - right_channels)))
            )
        control = actions(reference, step)
        left_outcome = reference.advance(control, 0.05)
        right_outcome = fast.advance(control, 0.05)
        if left_outcome != right_outcome:
            raise RuntimeError(f"outcomes diverged at step {step}")
        if (
            [body.to_dict() for body in reference.bodies]
            != [body.to_dict() for body in fast.bodies]
            or [obj.to_dict() for obj in reference.objects]
            != [obj.to_dict() for obj in fast.objects]
        ):
            raise RuntimeError(f"public physical state diverged at step {step}")
        maximum_state_error = max(
            maximum_state_error,
            float(np.max(np.abs(reference.data.qpos - fast.data.qpos))),
            float(np.max(np.abs(reference.data.qvel - fast.data.qvel))),
        )
    snapshot = fast.snapshot()
    restored = FastArticulatedSensoriumWorld.restore(snapshot)
    restore_error = max(
        float(np.max(np.abs(restored.data.qpos - fast.data.qpos))),
        float(np.max(np.abs(restored.data.qvel - fast.data.qvel))),
    )
    if maximum_state_error or maximum_sensor_error or restore_error:
        raise RuntimeError("fast physical path differs from the reference backend")
    return {
        "steps": steps,
        "maximum_state_abs_error": maximum_state_error,
        "maximum_sensor_abs_error": maximum_sensor_error,
        "restore_state_abs_error": restore_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 1:
        raise SystemExit("worlds and steps must be positive")
    audit = equivalence(min(args.steps, 30))
    reference = cohort("reference", args.worlds, args.steps)
    fast = cohort("fast", args.worlds, args.steps)
    print(json.dumps({
        "worlds": args.worlds,
        "steps_per_world": args.steps,
        "equivalence": audit,
        "reference": reference,
        "fast": fast,
        "wall_speedup": reference["wall_seconds"] / fast["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
