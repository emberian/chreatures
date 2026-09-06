#!/usr/bin/env python3
"""Train one fixed three-member action-skilled predictive ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.predictive_skills import (  # noqa: E402
    PredictiveSkillsConfig,
    PredictiveSkillsTrainer,
    SkillsNormalizer,
)
from chreatures.predictive_state import PredictiveSequence  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk(
    sequence: PredictiveSequence, start: int, rows: int, burn_in: int
) -> tuple[PredictiveSequence, int]:
    begin = max(0, start - burn_in)
    stop = min(len(sequence.features), start + rows + 16)
    reset = sequence.reset[begin:stop].copy()
    reset[0] = True
    return (
        PredictiveSequence(
            sequence.features[begin:stop],
            sequence.physiology[begin:stop],
            sequence.actions[begin:stop],
            reset,
            sequence.valid[begin:stop],
        ).validated(),
        start - begin,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=160)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    manifest_path = arguments.dataset / "manifest.json"
    layouts_path = arguments.dataset / "layouts.json"
    manifest = json.loads(manifest_path.read_text())
    episodes = sorted(arguments.dataset.glob("episode-*"))
    train = [PredictiveSequence.from_episode(path, "train") for path in episodes]
    holdout = [PredictiveSequence.from_episode(path, "holdout") for path in episodes]
    identity = {
        "dataset_manifest_sha256": sha256(manifest_path),
        "layouts_sha256": sha256(layouts_path),
        "graph_sha256": manifest["graph_sha256"],
        "port_spec_sha256": manifest["port_spec_sha256"],
        "port_bundle_sha256": manifest["port_bundle_sha256"],
        "source_normalizer": manifest["normalizer"],
        "split": manifest["splits"],
        "episodes": [path.name for path in episodes],
        "scope": "earlier scalar physiology; no common-chemistry state or target",
    }
    normalizer = SkillsNormalizer.fit(train, identity)
    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reports = []
    member_paths = []
    for member, seed in enumerate((20260906, 20260907, 20260908)):
        config = PredictiveSkillsConfig(seed=seed)
        trainer = PredictiveSkillsTrainer(config, normalizer, device=arguments.device)
        final_train = None
        for _ in range(arguments.epochs):
            for sequence in train:
                for start in range(0, len(sequence.features) - 1, arguments.chunk_rows):
                    section, loss_start = chunk(
                        sequence, start, arguments.chunk_rows, arguments.burn_in
                    )
                    final_train = trainer.update(section, loss_start=loss_start)
        trainer.model.eval()
        heldout = []
        with torch.no_grad():
            for sequence in holdout:
                for start in range(0, len(sequence.features) - 1, arguments.chunk_rows):
                    section, loss_start = chunk(
                        sequence, start, arguments.chunk_rows, arguments.burn_in
                    )
                    _, metrics = trainer.loss(section, loss_start=loss_start)
                    heldout.append(metrics)
        path = arguments.output / f"member-{member}.pt"
        value = {
            "format": "chreatures-predictive-skills-research-v1",
            "architecture": "signed-plus-effort-action-horizon-h1-h16",
            "config": asdict(config),
            "normalizer": {
                "input_mean": normalizer.input_mean,
                "input_scale": normalizer.input_scale,
                "physiology_delta_mean": normalizer.physiology_delta_mean,
                "physiology_delta_scale": normalizer.physiology_delta_scale,
                "identity": normalizer.identity,
            },
            "model": trainer.model.state_dict(),
            "updates": trainer.updates,
        }
        torch.save(value, path)
        keys = sorted(heldout[0])
        report = {
            "member": member,
            "seed": seed,
            "updates": trainer.updates,
            "last_training": final_train,
            "heldout": {
                key: float(np.mean([row[key] for row in heldout])) for key in keys
            },
            "checkpoint_sha256": sha256(path),
        }
        reports.append(report)
        member_paths.append(str(path))
    elapsed = time.perf_counter() - started
    receipt = {
        "format": "chreatures-predictive-skills-training-v1",
        "research_only": True,
        "architecture": {
            "action_basis": "all 8 executed actions plus abs(thrust,yaw,gaze_pitch)",
            "physiology_decoder": "latent + horizon embedding + current effective action basis",
            "horizons": list(range(1, 17)),
            "ensemble_members": 3,
            "uncertainty": "member mean disagreement is epistemic proxy; per-member Gaussian scale is residual/misfit",
        },
        "data_identity": identity,
        "data_limitation": "scalar E/G/F predates common chemistry; results do not establish chemical reserve regulation",
        "fixed_budget": {
            "epochs": arguments.epochs,
            "chunk_rows": arguments.chunk_rows,
            "burn_in": arguments.burn_in,
            "updates_per_member": reports[0]["updates"],
        },
        "device": str(arguments.device),
        "elapsed_seconds": elapsed,
        "members": reports,
        "member_paths": member_paths,
    }
    (arguments.output / "training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
