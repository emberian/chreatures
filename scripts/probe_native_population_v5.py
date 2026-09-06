#!/usr/bin/env python3
"""Replay the compact population-v5 Torch/native mechanics reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/training/developmental-resident-population-v5.native-reference.npz"
RECEIPT = ROOT / "data/training/developmental-resident-population-v5.native-receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    return parser.parse_args()


def adapter(row: int, index: int, population: dict, organism_sha256: str) -> dict:
    from chreatures.population import canonical_bytes

    value = {
        "candidate_sha256": f"{row + 1:064x}",
        "loci_sha256": "",
        "policy_adapter_index": index,
        "population_adapter_bank_sha256": population["identity"],
        "policy_adapter_count": population["count"],
        "policy_adapter_rank": population["rank"],
        "organism_interface_sha256": organism_sha256,
        "recurrent_gain": 1.0,
        "learning_rate_gain": 1.0,
        "action_gain": [1.0] * 12,
        "action_logit_temperature_offset": [0.0] * 12,
    }
    excluded = {"candidate_sha256", "loci_sha256"}
    body = {key: item for key, item in value.items() if key not in excluded}
    value["loci_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    return value


def main() -> int:
    args = arguments()
    receipt = json.loads(RECEIPT.read_text())
    if sha256(args.artifact) != receipt["external_artifact"]["sha256"]:
        raise ValueError("parity artifact bytes differ")
    if sha256(args.reference) != receipt["reference"]["sha256"]:
        raise ValueError("parity reference bytes differ")
    sys.path.insert(0, str(ROOT))
    if args.native_dir is not None:
        sys.path.insert(0, str(args.native_dir.resolve()))

    from chreatures.organism_interface import identity
    from chreatures.sensorimotor_worker_native import DevelopmentalResidentCohort
    from chreatures.population import canonical_bytes

    extension_path = Path(importlib.import_module("_cognitive_core").__file__).resolve()
    if sha256(extension_path) != receipt["native_binary"]["sha256"]:
        raise ValueError("native binary bytes differ")
    with np.load(args.artifact, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    organism_sha = hashlib.sha256(canonical_bytes(identity())).hexdigest()
    adapters = [
        adapter(row, row, metadata["population_adapters"], organism_sha)
        for row in range(3)
    ]
    cohort = DevelopmentalResidentCohort(
        args.artifact,
        3,
        action_mode="map",
        goal_seed=3,
        action_seed=4,
        candidate_adapters=adapters,
    )
    reference = np.load(args.reference, allow_pickle=False)
    maxima = []
    last = None
    for tick in range(2):
        observation = reference["raw_observation"][tick]
        physiology = observation[:, -12:]
        ticks = np.full(3, tick, dtype=np.uint64)
        last = cohort.step(
            observation,
            np.zeros((3, 384), dtype=np.float32),
            physiology,
            reference["previous"][tick],
            ticks,
            np.full(3, tick * 0.05, dtype=np.float64),
            reference["reset"][tick],
        )
        maxima.append({
            "hidden": float(np.max(np.abs(last["hidden"] - reference["states"][tick]))),
            "visible_actuators": float(np.max(np.abs(
                last["proposed_action"][:, 8:12] - reference["map_action"][tick, :, 8:12]
            ))),
        })
        cohort.observe_consequences(
            ticks,
            physiology,
            physiology.copy(),
            last["proposed_action"],
            np.zeros(3, dtype=np.float32),
            dt=0.05,
        )
    if max(item["hidden"] for item in maxima) > receipt["tolerance"]["hidden_atol"]:
        raise AssertionError("native recurrent state differs from Torch")
    if max(item["visible_actuators"] for item in maxima) != 0.0:
        raise AssertionError("native visible actuator MAP differs from Torch")

    snapshot = cohort.snapshot_value()
    observation = reference["raw_observation"][1]
    physiology = observation[:, -12:]
    step_args = (
        observation,
        np.zeros((3, 384), dtype=np.float32),
        physiology,
        last["proposed_action"],
        np.full(3, 2, dtype=np.uint64),
        np.full(3, 0.1, dtype=np.float64),
        np.zeros(3, dtype=np.bool_),
    )
    expected = cohort.step(*step_args)
    restored = DevelopmentalResidentCohort.restore_value(snapshot, args.artifact)
    actual = restored.step(*step_args)
    if any(not np.array_equal(expected[name], actual[name]) for name in expected):
        raise AssertionError("native snapshot continuation differs")
    print(json.dumps({
        "artifact_sha256": metadata["artifact_sha256"],
        "maxima": maxima,
        "snapshot_continuation": "bit-exact",
        "visible_actions": last["proposed_action"][:, 8:12].tolist(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
