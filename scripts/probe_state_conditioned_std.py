#!/usr/bin/env python3
"""Validate global-v1 compatibility and learnable state-conditioned PPO variance."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.learning import (
    ACTIONS,
    MacroRollout,
    PredictivePPOConfig,
    PredictivePPOTrainer,
)  # noqa: E402


def make_rollout(
    trainer: PredictivePPOTrainer, rng: np.random.Generator, steps: int = 5
) -> MacroRollout:
    rollout = MacroRollout()
    count, cfg = len(trainer.resident_ids), trainer.config
    for step in range(steps):
        features = rng.normal(size=(count, cfg.feature_dim)).astype(np.float32)
        physiology = rng.normal(size=(count, cfg.physiology_dim)).astype(np.float32)
        previous = trainer.act(features, physiology)
        target = rng.normal(size=(count, cfg.projection_dim)).astype(np.float32) * 0.1
        reward = np.linspace(-0.4, 0.7, count, dtype=np.float32) + step * 0.03
        rollout.append(
            features=previous["features"],
            physiology=previous["physiology"],
            context=previous["context"],
            latent=previous["latent"],
            action=previous["action"],
            log_prob=previous["log_prob"],
            value=previous["value"],
            reward=reward,
            done=np.zeros(count, np.float32),
            prediction_target=target,
        )
    return rollout


def tensor_delta(left: dict, right: dict) -> float:
    values = []
    for key in left.keys() & right.keys():
        if torch.is_tensor(left[key]):
            values.append(float((left[key].cpu() - right[key].cpu()).abs().max()))
    return max(values, default=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rng = np.random.default_rng(7781)
    cfg = PredictivePPOConfig(
        feature_dim=12,
        physiology_dim=6,
        context_dim=5,
        projection_dim=7,
        hidden_dim=16,
        epochs=2,
        minibatch_size=8,
        seed=7781,
    )
    legacy = PredictivePPOTrainer(["a", "b", "c"], cfg)
    legacy.update(make_rollout(legacy, rng), np.zeros(3, np.float32), 0.25)
    legacy.context[:] = rng.normal(size=legacy.context.shape).astype(np.float32)
    legacy.error_fast[:] = (0.1, 0.2, 0.3)
    legacy.error_slow[:] = (0.2, 0.3, 0.4)
    with tempfile.TemporaryDirectory() as directory:
        old_path = Path(directory) / "global.pt"
        legacy.snapshot(old_path, extra={"marker": "private-preserved"})
        restored, _ = PredictivePPOTrainer.restore(old_path)
        upgraded, extra = PredictivePPOTrainer.upgrade_state_conditioned(old_path)

        features = rng.normal(size=(3, cfg.feature_dim)).astype(np.float32)
        physiology = rng.normal(size=(3, cfg.physiology_dim)).astype(np.float32)
        sample_rng = torch.get_rng_state()
        old_action = restored.act(features, physiology)
        torch.set_rng_state(sample_rng)
        new_action = upgraded.act(features, physiology)
        paired = {
            key: float(np.max(np.abs(old_action[key] - new_action[key])))
            for key in ("latent", "action", "log_prob", "value", "prediction")
        }

        old_opt, new_opt = (
            restored.optimizer.state_dict(),
            upgraded.optimizer.state_dict(),
        )
        legacy_ids = old_opt["param_groups"][0]["params"]
        optimizer_delta = max(
            (
                float(
                    (
                        old_opt["state"][old_id][key].cpu()
                        - new_opt["state"][new_id][key].cpu()
                    )
                    .abs()
                    .max()
                )
                for old_id, new_id in zip(
                    legacy_ids, new_opt["param_groups"][0]["params"], strict=False
                )
                for key in old_opt["state"].get(old_id, {})
            ),
            default=0.0,
        )
        new_ids = new_opt["param_groups"][0]["params"][len(legacy_ids) :]
        new_optimizer_state_max = max(
            (
                float(value.abs().max())
                for pid in new_ids
                for value in new_opt["state"][pid].values()
            ),
            default=0.0,
        )
        private_delta = max(
            float(np.max(np.abs(restored.context - upgraded.context))),
            float(np.max(np.abs(restored.error_fast - upgraded.error_fast))),
            float(np.max(np.abs(restored.error_slow - upgraded.error_slow))),
        )
        head_before = upgraded.model.std_offset.weight.detach().clone()
        probe_features = torch.as_tensor(np.stack((features[0], features[1])))
        probe_body = torch.as_tensor(np.stack((physiology[0], physiology[1])))
        probe_context = torch.as_tensor(
            np.stack((upgraded.context[0], upgraded.context[1]))
        )
        with torch.no_grad():
            mean, _, hidden = upgraded.model(probe_features, probe_body, probe_context)
            std_before = upgraded.model.distribution(mean, hidden).scale.clone()
        metrics = upgraded.update(
            make_rollout(upgraded, rng), np.zeros(3, np.float32), 0.25
        )
        with torch.no_grad():
            mean, _, hidden = upgraded.model(probe_features, probe_body, probe_context)
            std_after = upgraded.model.distribution(mean, hidden).scale.clone()
        head_change = float(
            (upgraded.model.std_offset.weight.detach() - head_before).abs().max()
        )
        conditioned_span = float((std_after[0] - std_after[1]).abs().max())

        v2_path = Path(directory) / "conditioned.pt"
        upgraded.snapshot(v2_path)
        expected = upgraded.act(features, physiology)
        replay, _ = PredictivePPOTrainer.restore(v2_path)
        actual = replay.act(features, physiology)
        replay_delta = max(
            float(np.max(np.abs(expected[key] - actual[key])))
            for key in ("latent", "action", "log_prob", "value", "prediction")
        )
    report = {
        "actions": list(ACTIONS),
        "upgrade_profile": upgraded.config.std_profile,
        "zero_initialized_head": float(head_before.abs().max()) == 0.0,
        "paired_initial_sampling_max_abs_delta": paired,
        "legacy_optimizer_moment_max_abs_delta": optimizer_delta,
        "new_optimizer_state_initial_max_abs": new_optimizer_state_max,
        "private_state_max_abs_delta": private_delta,
        "upgrade_extra": extra,
        "ppo_update": metrics,
        "head_weight_max_change": head_change,
        "conditioned_std_between_states_max_abs_delta": conditioned_span,
        "std_after_vs_before_max_abs_delta": float(
            (std_after - std_before).abs().max()
        ),
        "v2_checkpoint_replay_max_abs_delta": replay_delta,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
