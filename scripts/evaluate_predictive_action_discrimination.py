#!/usr/bin/env python3
"""Measure frozen predictive-organ sensitivity to matched versus altered actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.predictive_state import (  # noqa: E402
    PredictiveSequence,
    PredictiveStateTrainer,
)

PHYSIOLOGY = ("energy", "gut", "fatigue", "speed_local", "angular_local", "support")
VARIANTS = ("matched", "resident_shifted", "zero")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def accumulators() -> dict[str, dict[str, np.ndarray | float | int]]:
    return {
        name: {
            "feature_squared_error": 0.0,
            "physiology_squared_error": np.zeros(6, np.float64),
            "feature_effect_squared": 0.0,
            "physiology_effect_squared": np.zeros(6, np.float64),
            "resident_targets": 0,
        }
        for name in VARIANTS
    }


@torch.no_grad()
def evaluate_episode(
    trainer: PredictiveStateTrainer, sequence: PredictiveSequence
) -> dict[int, dict[str, dict[str, np.ndarray | float | int]]]:
    device = trainer.device
    config = trainer.config
    states = torch.as_tensor(
        trainer.encode(
            sequence.features,
            sequence.physiology,
            sequence.actions,
            sequence.reset,
        ),
        device=device,
    )
    actions = torch.as_tensor(sequence.actions, device=device)
    shifted = torch.roll(actions, shifts=1, dims=1)
    valid = torch.as_tensor(sequence.valid, device=device)
    resets = torch.as_tensor(sequence.reset, device=device)
    feature = torch.as_tensor(sequence.features, device=device)
    physiology = torch.as_tensor(sequence.physiology, device=device)
    time, residents, _ = actions.shape
    starts = time - 1
    imagined = {name: states[:-1].reshape(-1, config.latent_dim) for name in VARIANTS}
    alive = valid[:-1].clone()
    result: dict[int, dict[str, dict[str, np.ndarray | float | int]]] = {}

    for distance in config.horizons:
        usable = time - distance
        action_by_variant = {
            "matched": actions[distance - 1 : time - 1],
            "resident_shifted": shifted[distance - 1 : time - 1],
            "zero": torch.zeros_like(actions[distance - 1 : time - 1]),
        }
        predictions: dict[str, dict[str, torch.Tensor]] = {}
        for name, future_action in action_by_variant.items():
            padded = torch.zeros((starts, residents, config.action_dim), device=device)
            padded[:usable] = future_action
            imagined[name] = trainer.model.transition(
                imagined[name], padded.reshape(-1, config.action_dim)
            )
            feature_mean, _, phys_delta, _ = trainer.model.prediction(
                imagined[name].reshape(starts, residents, -1)[:usable]
            )
            mean = torch.as_tensor(trainer.normalizer.input_mean, device=device)
            scale = torch.as_tensor(trainer.normalizer.input_scale, device=device)
            delta_mean = torch.as_tensor(
                trainer.normalizer.physiology_delta_mean[distance - 1], device=device
            )
            delta_scale = torch.as_tensor(
                trainer.normalizer.physiology_delta_scale[distance - 1], device=device
            )
            predictions[name] = {
                "feature": mean[: config.feature_dim]
                + scale[: config.feature_dim] * feature_mean,
                "physiology": physiology[:-distance]
                + delta_mean
                + delta_scale * phys_delta,
            }

        alive[:usable] &= valid[distance:] & ~resets[distance:]
        alive[usable:] = False
        mask = alive[:usable]
        target_feature = feature[distance:]
        target_physiology = physiology[distance:]
        matched_feature = predictions["matched"]["feature"]
        matched_physiology = predictions["matched"]["physiology"]
        horizon = accumulators()
        count = int(mask.sum().item())
        for name in VARIANTS:
            pred_feature = predictions[name]["feature"]
            pred_physiology = predictions[name]["physiology"]
            horizon[name]["feature_squared_error"] = float(
                (pred_feature[mask] - target_feature[mask]).square().sum().item()
            )
            horizon[name]["physiology_squared_error"] = (
                (pred_physiology[mask] - target_physiology[mask])
                .square()
                .sum(0)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            horizon[name]["feature_effect_squared"] = float(
                (pred_feature[mask] - matched_feature[mask]).square().sum().item()
            )
            horizon[name]["physiology_effect_squared"] = (
                (pred_physiology[mask] - matched_physiology[mask])
                .square()
                .sum(0)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            horizon[name]["resident_targets"] = count
        result[distance] = horizon
    return result


def merge(
    destination: dict[int, dict[str, dict[str, np.ndarray | float | int]]],
    source: dict[int, dict[str, dict[str, np.ndarray | float | int]]],
) -> None:
    for horizon, variants in source.items():
        if horizon not in destination:
            destination[horizon] = accumulators()
        for name, values in variants.items():
            for key, value in values.items():
                destination[horizon][name][key] += value


def summarize(
    aggregate: dict[int, dict[str, dict[str, np.ndarray | float | int]]],
    feature_dim: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for horizon, variants in sorted(aggregate.items()):
        count = int(variants["matched"]["resident_targets"])
        row: dict[str, object] = {"resident_targets": count, "variants": {}}
        matched_feature_mse = float(variants["matched"]["feature_squared_error"]) / (
            count * feature_dim
        )
        matched_phys_mse = (
            np.asarray(variants["matched"]["physiology_squared_error"]) / count
        )
        for name, values in variants.items():
            feature_mse = float(values["feature_squared_error"]) / (count * feature_dim)
            physiology_mse = np.asarray(values["physiology_squared_error"]) / count
            feature_effect = float(values["feature_effect_squared"]) / (
                count * feature_dim
            )
            physiology_effect = np.asarray(values["physiology_effect_squared"]) / count
            row["variants"][name] = {
                "feature_rmse": feature_mse**0.5,
                "feature_mse_increase_over_matched": feature_mse - matched_feature_mse,
                "feature_prediction_effect_rms": feature_effect**0.5,
                "physiology_rmse": {
                    key: float(value**0.5)
                    for key, value in zip(PHYSIOLOGY, physiology_mse, strict=True)
                },
                "physiology_mse_increase_over_matched": {
                    key: float(value)
                    for key, value in zip(
                        PHYSIOLOGY, physiology_mse - matched_phys_mse, strict=True
                    )
                },
                "physiology_prediction_effect_rms": {
                    key: float(value**0.5)
                    for key, value in zip(PHYSIOLOGY, physiology_effect, strict=True)
                },
            }
        output[f"h{horizon}"] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    trainer = PredictiveStateTrainer.restore(
        arguments.checkpoint, device=arguments.device
    )
    trainer.model.eval()
    aggregate: dict[int, dict[str, dict[str, np.ndarray | float | int]]] = {}
    episodes = sorted(arguments.dataset.glob("episode-*"))
    for episode in episodes:
        sequence = PredictiveSequence.from_episode(episode, "holdout")
        merge(aggregate, evaluate_episode(trainer, sequence))

    manifest = arguments.dataset / "manifest.json"
    receipt = {
        "format": "chreatures-predictive-action-discrimination-v1",
        "method": {
            "contexts": "frozen posterior states from whole-world heldout residents",
            "matched": "recorded future executed actions",
            "resident_shifted": "same-time actions circularly shifted by one heldout resident",
            "zero": "all future actions zero",
            "interpretation": "positive MSE increase means matched actions better predict the observed future; prediction effect measures sensitivity and is not a causal counterfactual estimate",
            "weights_or_normalizer_updated": False,
        },
        "dataset_manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(arguments.checkpoint),
        "export_sha256": sha256(arguments.export),
        "device": str(trainer.device),
        "episodes": [episode.name for episode in episodes],
        "results": summarize(aggregate, trainer.config.feature_dim),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
