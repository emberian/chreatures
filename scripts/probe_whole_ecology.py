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
import time
from pathlib import Path
from typing import Any

from chreatures.checkpoint import canonical
from chreatures.runtime3d import Habitat3D

ROOT = Path(__file__).resolve().parents[1]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "expected": repr(left)[:240], "restored": repr(right)[:240]}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {"path": path, "expected_keys": sorted(left), "restored_keys": sorted(right)}
        for key in sorted(left):
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {"path": path, "expected_length": len(left), "restored_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if left != right:
        return {"path": path, "expected": repr(left)[:240], "restored": repr(right)[:240]}
    return None


def _continuation_state(habitat: Habitat3D) -> dict[str, Any]:
    """Collect every mutable subsystem required by this probe's contract."""
    return {
        "physics": habitat.world.snapshot(),
        "field": None if habitat.field is None else habitat.field.snapshot(),
        "resources": None if habitat.resources is None else habitat.resources.snapshot(),
        "resource_state": habitat.resource_state,
        "acoustics": None if habitat.acoustics is None else habitat.acoustics.snapshot(),
        "acoustic_state": habitat.acoustic_state,
        "biosphere": None if habitat.biosphere is None else habitat.biosphere.snapshot(),
        "visitor": habitat.visitor.snapshot(),
        "organs": {key: value.snapshot() for key, value in habitat.organs.items()},
        "motors": {key: value.snapshot_value() for key, value in habitat.motors.items()},
        "foresights": {key: value.snapshot() for key, value in habitat.foresights.items()},
        "foresight_deployment": habitat.foresight_deployment,
        "neural_response": habitat.neural_state,
        "feature_mean": {key: value.tolist() for key, value in habitat.feature_mean.items()},
        "feature_variance": {key: value.tolist() for key, value in habitat.feature_variance.items()},
        "outcomes": habitat.outcomes,
        "last_senses": habitat.last_senses,
        "sensed_at": habitat.sensed_at,
        "tick": habitat.tick,
        "branch": habitat.branch,
        "paused": habitat.paused,
        "speed": habitat.speed,
        "pending_step": habitat.pending_step,
        "journal": list(habitat.journal),
        "history": {key: list(value) for key, value in habitat.history.items()},
        "execution_migrations": habitat.execution_migrations,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"

    started = time.perf_counter()
    habitat = Habitat3D(
        seed=args.seed,
        brain_url=args.brain_url,
        spec=json.loads(args.habitat.read_text()) if args.habitat is not None else None,
        resources=None if args.biosphere is not None else args.resources,
        biosphere=args.biosphere,
        acoustics=None if args.no_acoustics else args.acoustics,
        motor_genome=args.motor_genome,
        personal_memory=args.personal_memory,
        personal_plasticity=args.personal_plasticity,
        predictive_model=args.predictive_model,
    )
    habitat.branch = "research"
    if args.impulse_entity is not None:
        habitat.command({"op": "impulse", "id": args.impulse_entity, "impulse": [0.0, 0.0, 0.75]})
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
    biosphere_state = expected["biosphere"]
    state_evidence = {
        "developed_parts": 0 if biosphere_state is None else len(biosphere_state["parts"]),
        "growth_buds": {} if biosphere_state is None else {
            key: len(value["state"]["buds"]) for key, value in biosphere_state["growth"].items()
        },
        "egested_mass": 0.0 if biosphere_state is None or biosphere_state["exchange"] is None else sum(
            sum(values) for values in biosphere_state["exchange"]["egested"].values()
        ),
        "foresight_observations": {
            key: value["observation_count"] for key, value in expected["foresights"].items()
        },
        "motor_macro_steps": {
            key: value["macro_steps"] for key, value in expected["motors"].items()
        },
    }
    report = {
        "format": "chreatures-whole-ecology-probe-v2",
        "passed": passed,
        "seed": args.seed,
        "brain_url": args.brain_url,
        "sources": {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in {
                "habitat": args.habitat, "biosphere": args.biosphere,
                "motor_genome": args.motor_genome, "predictive_model": args.predictive_model,
                "brain_artifact": args.brain_artifact, "port_bundle": args.port_bundle,
            }.items() if path is not None
        },
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
        "first_differences": {
            key: difference for key in expected
            if not exact[key] and (difference := _first_difference(expected[key], actual[key])) is not None
        },
        "elapsed_seconds": time.perf_counter() - started,
        "state_evidence": state_evidence,
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
    receipt = {
        "format": "chreatures-whole-ecology-replay-receipt-v1",
        "passed": passed, "seed": args.seed, "steps": report["steps"],
        "checkpoint": report["checkpoint"], "sources": report["sources"],
        "state_evidence": state_evidence,
        "owner_sha256": {key: value["expected"] for key, value in section_hashes.items()},
        "neural_snapshot_sha256": expected_neural["sha256"],
        "neural_exact": remote_exact,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        failed = [key for key, value in exact.items() if not value]
        raise RuntimeError(f"whole-ecology continuation diverged: {failed}; remote_neural={remote_exact}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-url", default="http://127.0.0.1:18769")
    parser.add_argument("--habitat", type=Path)
    parser.add_argument("--biosphere", type=Path)
    parser.add_argument("--motor-genome", type=Path)
    parser.add_argument("--personal-memory", action="store_true")
    parser.add_argument("--personal-plasticity", action="store_true")
    parser.add_argument("--predictive-model", type=Path)
    parser.add_argument("--brain-artifact", type=Path, default=ROOT / "data/metal-brain/metal-csr-v2.bin")
    parser.add_argument("--port-bundle", type=Path, default=ROOT / "data/ports/retinal-v1-maps.npz")
    parser.add_argument("--resources", type=Path, default=ROOT / "data/ecology/portable-orchard.json")
    parser.add_argument("--acoustics", type=Path, default=ROOT / "data/components/acoustic-play.json")
    parser.add_argument("--no-acoustics", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/whole-ecology-probe")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--continuation-steps", type=int, default=6)
    parser.add_argument("--impulse-entity", default="violet-ball")
    parser.add_argument("--no-initial-impulse", action="store_const", const=None, dest="impulse_entity")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
