#!/usr/bin/env python3
"""Evaluate frozen action-skilled ensemble discrimination and disagreement."""

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

from research.predictive_skills import (  # noqa: E402
    PredictiveSkillsConfig,
    PredictiveSkillsTrainer,
    SkillsNormalizer,
)
from chreatures.predictive_state import PredictiveSequence  # noqa: E402

PHYSIOLOGY = ("energy", "gut", "fatigue", "speed_local", "angular_local", "support")
HORIZONS = (1, 4, 8, 16)
VARIANTS = ("matched", "resident_shifted", "zero")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_member(path: Path, device: str) -> PredictiveSkillsTrainer:
    value = torch.load(path, map_location=device, weights_only=False)
    item = value["normalizer"]
    normalizer = SkillsNormalizer(
        np.asarray(item["input_mean"]),
        np.asarray(item["input_scale"]),
        np.asarray(item["physiology_delta_mean"]),
        np.asarray(item["physiology_delta_scale"]),
        dict(item["identity"]),
    )
    trainer = PredictiveSkillsTrainer(
        PredictiveSkillsConfig(**value["config"]), normalizer, device=device
    )
    trainer.model.load_state_dict(value["model"])
    trainer.model.eval()
    return trainer


@torch.no_grad()
def predict(
    trainer: PredictiveSkillsTrainer, sequence: PredictiveSequence
) -> dict[int, dict[str, np.ndarray]]:
    posterior, _, actions, resets, valid = trainer._posterior(sequence)
    time, residents = valid.shape
    variants = {
        "matched": actions,
        "resident_shifted": torch.roll(actions, 1, 1),
        "zero": torch.zeros_like(actions),
    }
    states = {
        name: posterior[:-1].reshape(-1, trainer.config.latent_dim) for name in VARIANTS
    }
    alive = valid[:-1].clone()
    physiology = torch.as_tensor(sequence.physiology, device=trainer.device)
    fmean = torch.as_tensor(trainer.normalizer.input_mean[:384], device=trainer.device)
    fscale = torch.as_tensor(
        trainer.normalizer.input_scale[:384], device=trainer.device
    )
    result: dict[int, dict[str, np.ndarray]] = {}
    for horizon in range(1, 17):
        usable = time - horizon
        current_actions = {}
        for name in VARIANTS:
            padded = torch.zeros_like(actions[:-1])
            padded[:usable] = variants[name][horizon - 1 : time - 1]
            current_actions[name] = padded
            states[name] = trainer.model.transition(states[name], padded.reshape(-1, 8))
        alive[:usable] &= valid[horizon:] & ~resets[horizon:]
        alive[usable:] = False
        if horizon not in HORIZONS:
            continue
        mask = alive[:usable].cpu().numpy()
        row: dict[str, np.ndarray] = {
            "mask": mask,
            "target_feature": sequence.features[horizon:],
            "target_physiology": sequence.physiology[horizon:],
        }
        delta_mean = torch.as_tensor(
            trainer.normalizer.physiology_delta_mean[horizon - 1], device=trainer.device
        )
        delta_scale = torch.as_tensor(
            trainer.normalizer.physiology_delta_scale[horizon - 1],
            device=trainer.device,
        )
        for name in VARIANTS:
            state = states[name].reshape(time - 1, residents, -1)[:usable]
            feature, _, phys, _ = trainer.model.prediction(
                state, current_actions[name][:usable], horizon
            )
            row[f"{name}_feature"] = (fmean + fscale * feature).cpu().numpy()
            row[f"{name}_physiology"] = (
                (physiology[:-horizon] + delta_mean + delta_scale * phys).cpu().numpy()
            )
        result[horizon] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--members", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    trainers = [load_member(path, arguments.device) for path in arguments.members]
    aggregate = {
        h: {
            v: {"fse": 0.0, "pse": np.zeros(6), "feffect": 0.0, "peffect": np.zeros(6)}
            for v in VARIANTS
        }
        | {
            "count": 0,
            "fdis": 0.0,
            "pdis": np.zeros(6),
            "persistence_fse": 0.0,
            "persistence_pse": np.zeros(6),
            "feature_calibration": np.zeros(6),
            "physiology_calibration": np.zeros((6, 6)),
        }
        for h in HORIZONS
    }
    for episode in sorted(arguments.dataset.glob("episode-*")):
        sequence = PredictiveSequence.from_episode(episode, "holdout")
        member_predictions = [predict(trainer, sequence) for trainer in trainers]
        for horizon in HORIZONS:
            mask = member_predictions[0][horizon]["mask"]
            target_f = member_predictions[0][horizon]["target_feature"][mask]
            target_p = member_predictions[0][horizon]["target_physiology"][mask]
            current_f = sequence.features[:-horizon][mask]
            current_p = sequence.physiology[:-horizon][mask]
            count = int(mask.sum())
            aggregate[horizon]["count"] += count
            aggregate[horizon]["persistence_fse"] += float(
                np.square(current_f - target_f).sum()
            )
            aggregate[horizon]["persistence_pse"] += np.square(
                current_p - target_p
            ).sum(0)
            ensemble = {}
            for variant in VARIANTS:
                features = np.stack(
                    [
                        row[horizon][f"{variant}_feature"][mask]
                        for row in member_predictions
                    ]
                )
                physiology = np.stack(
                    [
                        row[horizon][f"{variant}_physiology"][mask]
                        for row in member_predictions
                    ]
                )
                ensemble[variant] = (features.mean(0), physiology.mean(0))
                aggregate[horizon][variant]["fse"] += float(
                    np.square(features.mean(0) - target_f).sum()
                )
                aggregate[horizon][variant]["pse"] += np.square(
                    physiology.mean(0) - target_p
                ).sum(0)
                if variant == "matched":
                    feature_variance = features.var(0)
                    physiology_variance = physiology.var(0)
                    feature_error = np.square(features.mean(0) - target_f)
                    physiology_error = np.square(physiology.mean(0) - target_p)
                    aggregate[horizon]["fdis"] += float(feature_variance.sum())
                    aggregate[horizon]["pdis"] += physiology_variance.sum(0)
                    x = feature_variance.reshape(-1).astype(np.float64)
                    y = feature_error.reshape(-1).astype(np.float64)
                    aggregate[horizon]["feature_calibration"] += np.asarray(
                        [
                            len(x),
                            x.sum(),
                            y.sum(),
                            np.square(x).sum(),
                            np.square(y).sum(),
                            (x * y).sum(),
                        ]
                    )
                    for channel in range(6):
                        x = physiology_variance[:, channel].astype(np.float64)
                        y = physiology_error[:, channel].astype(np.float64)
                        aggregate[horizon]["physiology_calibration"][channel] += (
                            np.asarray(
                                [
                                    len(x),
                                    x.sum(),
                                    y.sum(),
                                    np.square(x).sum(),
                                    np.square(y).sum(),
                                    (x * y).sum(),
                                ]
                            )
                        )
            matched_f, matched_p = ensemble["matched"]
            for variant in ("resident_shifted", "zero"):
                feature, physiology = ensemble[variant]
                aggregate[horizon][variant]["feffect"] += float(
                    np.square(feature - matched_f).sum()
                )
                aggregate[horizon][variant]["peffect"] += np.square(
                    physiology - matched_p
                ).sum(0)
    results = {}
    for horizon in HORIZONS:
        values = aggregate[horizon]
        count = values["count"]
        matched_fmse = values["matched"]["fse"] / (count * 384)
        matched_pmse = values["matched"]["pse"] / count

        def correlation(stats: np.ndarray) -> float:
            n, sx, sy, sxx, syy, sxy = stats
            covariance = sxy / n - (sx / n) * (sy / n)
            variance_x = max(sxx / n - (sx / n) ** 2, 0.0)
            variance_y = max(syy / n - (sy / n) ** 2, 0.0)
            denominator = np.sqrt(variance_x * variance_y)
            return float(covariance / denominator) if denominator > 0 else 0.0

        row = {
            "resident_targets": count,
            "persistence_rmse": {
                "feature": float(np.sqrt(values["persistence_fse"] / (count * 384))),
                "physiology": {
                    name: float(x)
                    for name, x in zip(
                        PHYSIOLOGY,
                        np.sqrt(values["persistence_pse"] / count),
                        strict=True,
                    )
                },
            },
            "ensemble_epistemic_rms": {
                "feature": float(np.sqrt(values["fdis"] / (count * 384))),
                "physiology": {
                    name: float(x)
                    for name, x in zip(
                        PHYSIOLOGY, np.sqrt(values["pdis"] / count), strict=True
                    )
                },
            },
            "disagreement_squared_error_pearson": {
                "feature": correlation(values["feature_calibration"]),
                "physiology": {
                    name: correlation(stats)
                    for name, stats in zip(
                        PHYSIOLOGY, values["physiology_calibration"], strict=True
                    )
                },
            },
            "variants": {},
        }
        for variant in VARIANTS:
            fmse = values[variant]["fse"] / (count * 384)
            pmse = values[variant]["pse"] / count
            row["variants"][variant] = {
                "feature_rmse": float(np.sqrt(fmse)),
                "feature_mse_increase_over_matched": float(fmse - matched_fmse),
                "feature_effect_rms": float(
                    np.sqrt(values[variant]["feffect"] / (count * 384))
                ),
                "physiology_rmse": {
                    name: float(x)
                    for name, x in zip(PHYSIOLOGY, np.sqrt(pmse), strict=True)
                },
                "physiology_mse_increase_over_matched": {
                    name: float(x)
                    for name, x in zip(PHYSIOLOGY, pmse - matched_pmse, strict=True)
                },
                "physiology_effect_rms": {
                    name: float(x)
                    for name, x in zip(
                        PHYSIOLOGY,
                        np.sqrt(values[variant]["peffect"] / count),
                        strict=True,
                    )
                },
            }
        results[f"h{horizon}"] = row
    receipt = {
        "format": "chreatures-predictive-skills-evaluation-v1",
        "research_only": True,
        "dataset_manifest_sha256": sha256(arguments.dataset / "manifest.json"),
        "member_sha256": [sha256(path) for path in arguments.members],
        "controls": "same heldout posterior context; recorded actions, same-time next-resident actions, or zero actions",
        "uncertainty_scope": "ensemble standard deviation is a model-disagreement proxy, not calibrated OOD probability",
        "data_limitation": "scalar E/G/F predates common chemistry; no chemical reserve claim",
        "results": results,
    }
    arguments.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
