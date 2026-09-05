#!/usr/bin/env python3
"""Measure the real 3-D sensory paths and batched retinal implementation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chreatures.articulated import ArticulatedWorld
from chreatures.cognition import AdaptiveOrgan
from chreatures.physics import PhysicsWorld
from chreatures.sensorium import ArticulatedSensoriumWorld, SensoriumWorld


def measure(callable_, iterations: int, repeats: int) -> dict[str, float]:
    for _ in range(10):
        callable_()
    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                callable_()
            samples.append((time.perf_counter_ns() - started) / iterations / 1e6)
    finally:
        if was_enabled:
            gc.enable()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def settled(world_class):
    world = world_class(seed=23)
    for _ in range(20):
        world.advance({}, 0.05)
    return world


def sensory_case(base_class, fast_class, iterations: int, repeats: int) -> dict:
    base, fast = settled(base_class), settled(fast_class)
    base_call = lambda: [base.sense(body.id) for body in base.bodies]
    fast_call = lambda: [fast.sense(body.id) for body in fast.bodies]
    base_values, fast_values = base_call(), fast_call()
    error = 0.0
    for original, batched in zip(base_values, fast_values, strict=True):
        for field in ("vision", "retina3d"):
            error = max(error, float(np.max(np.abs(
                np.asarray(original[field], dtype=float) - np.asarray(batched[field], dtype=float)
            ))))
    baseline = measure(base_call, iterations, repeats)
    batched = measure(fast_call, iterations, repeats)
    return {
        "residents": len(base.bodies),
        "rays_per_resident": 96,
        "baseline": baseline,
        "batched": batched,
        "speedup": baseline["median_ms"] / batched["median_ms"],
        "median_saved_ms": baseline["median_ms"] - batched["median_ms"],
        "max_output_error": error,
    }


def audit_other_paths(iterations: int, repeats: int) -> dict:
    world = settled(PhysicsWorld)
    fields = lambda: [(world._odor(body), world._sound(body)) for body in world.bodies]
    advance_world = settled(PhysicsWorld)

    organ = AdaptiveOrgan(feature_dim=48, seed=31)
    features = np.zeros(48, dtype=np.float32)
    context = np.zeros(24, dtype=np.float32)
    action = np.zeros(8, dtype=np.float32)
    for index in range(organ.memory.capacity):
        sample = np.sin(np.arange(48, dtype=np.float32) * 0.1 + index * 0.03)
        organ.memory.remember(sample, context, action, sample + 0.01, float(index % 5), index * 0.05)

    return {
        "odor_and_sound_three_residents": measure(fields, iterations, repeats),
        "physics_advance_idle_three_residents": measure(
            lambda: advance_world.advance({}, 0.05), max(20, iterations // 5), repeats
        ),
        "personal_memory_recall_384x48": measure(
            lambda: organ.memory.recall(features, context), iterations, repeats
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 20 or args.repeats < 1:
        parser.error("iterations must be >=20 and repeats >=1")
    report = {
        "mujoco_version": mujoco.__version__,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "crawler_sense_all": sensory_case(PhysicsWorld, SensoriumWorld, args.iterations, args.repeats),
        "articulated_sense_all": sensory_case(
            ArticulatedWorld, ArticulatedSensoriumWorld, args.iterations, args.repeats
        ),
        "audit": audit_other_paths(args.iterations, args.repeats),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["crawler_sense_all"]["max_output_error"] != 0.0:
        raise SystemExit("batched crawler retina differs from scalar reference")
    if report["articulated_sense_all"]["max_output_error"] != 0.0:
        raise SystemExit("batched articulated retina differs from scalar reference")


if __name__ == "__main__":
    main()
