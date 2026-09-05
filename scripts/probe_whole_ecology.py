#!/usr/bin/env python3
"""Probe exact continuation across the complete 3-D ecological runtime.

This is an integration artifact rather than a synthetic unit test: it uses a
real remote MaleCNS service, articulated MuJoCo residents, transported fields,
renewable resources, and finite-energy physical acoustics together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from chreatures.runtime import canonical
from chreatures.runtime3d import Habitat3D


ROOT = Path(__file__).resolve().parents[1]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _continuation_state(habitat: Habitat3D) -> dict[str, Any]:
    """Collect every mutable subsystem required by this probe's contract."""
    return {
        "physics": habitat.world.snapshot(),
        "field": None if habitat.field is None else habitat.field.snapshot(),
        "resources": None if habitat.resources is None else habitat.resources.snapshot(),
        "resource_state": habitat.resource_state,
        "acoustics": None if habitat.acoustics is None else habitat.acoustics.snapshot(),
        "acoustic_state": habitat.acoustic_state,
        "organs": {key: value.snapshot() for key, value in habitat.organs.items()},
        "neural_response": habitat.neural_state,
        "feature_mean": {key: value.tolist() for key, value in habitat.feature_mean.items()},
        "feature_variance": {key: value.tolist() for key, value in habitat.feature_variance.items()},
        "outcomes": habitat.outcomes,
        "last_senses": habitat.last_senses,
        "sensed_at": habitat.sensed_at,
        "tick": habitat.tick,
        "branch": habitat.branch,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"

    started = time.perf_counter()
    habitat = Habitat3D(
        seed=args.seed,
        brain_url=args.brain_url,
        resources=args.resources,
        acoustics=args.acoustics,
    )
    habitat.branch = "research"
    habitat.command({"op": "impulse", "id": "violet-ball", "impulse": [0.0, 0.0, 0.75]})
    habitat.step(args.warmup_steps)
    checkpoint_sha256 = habitat.save(checkpoint)

    habitat.step(args.continuation_steps)
    expected = _continuation_state(habitat)
    remote_name = f"whole-ecology-final-{habitat.id}"
    expected_neural = habitat.neural.snapshot(remote_name, list(habitat.remote_ids.values()))

    restored = Habitat3D.load(checkpoint, brain_url=args.brain_url)
    restored.step(args.continuation_steps)
    actual = _continuation_state(restored)
    # Canonical checkpoint JSON sorts mapping keys. Preserve the original
    # cohort row order explicitly so the remote NPZ identity compares state,
    # rather than incidental dictionary insertion order after JSON loading.
    actual_neural = restored.neural.snapshot(remote_name, expected_neural["residents"])

    section_hashes = {
        key: {"expected": _digest(expected[key]), "restored": _digest(actual[key])}
        for key in expected
    }
    exact = {key: value["expected"] == value["restored"] for key, value in section_hashes.items()}
    remote_exact = expected_neural["sha256"] == actual_neural["sha256"]
    passed = all(exact.values()) and remote_exact
    report = {
        "format": "chreatures-whole-ecology-probe-v1",
        "passed": passed,
        "seed": args.seed,
        "brain_url": args.brain_url,
        "branch": habitat.branch,
        "steps": {"before_checkpoint": args.warmup_steps, "continuation": args.continuation_steps},
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        "neural": {
            "graph_sha256": habitat.neural.graph["sha256"],
            "neurons": habitat.neural.graph["neurons"],
            "connections": habitat.neural.graph["edges"],
            "inputs": len(habitat.neural.input_names),
            "readouts": len(habitat.neural.output_names),
            "snapshot_expected": expected_neural,
            "snapshot_restored": actual_neural,
            "exact": remote_exact,
        },
        "sections": section_hashes,
        "exact": exact,
        "elapsed_seconds": time.perf_counter() - started,
        "final": {
            "world_time": restored.world.time,
            "field_time": None if restored.field is None else restored.field.time,
            "resource_time": None if restored.resources is None else restored.resources.time,
            "acoustic_time": None if restored.acoustics is None else restored.acoustics.time,
            "acoustic_energy_residual": None if restored.acoustics is None else restored.acoustics.view()["energy_residual"],
            "acoustic_mechanical_residual": None if restored.acoustics is None else restored.acoustics.view()["mechanical_residual"],
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        failed = [key for key, value in exact.items() if not value]
        raise RuntimeError(f"whole-ecology continuation diverged: {failed}; remote_neural={remote_exact}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-url", default="http://127.0.0.1:18769")
    parser.add_argument("--resources", type=Path, default=ROOT / "data/ecology/portable-orchard.json")
    parser.add_argument("--acoustics", type=Path, default=ROOT / "data/components/acoustic-play.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/whole-ecology-probe")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--continuation-steps", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
