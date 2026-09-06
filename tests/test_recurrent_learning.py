from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chreatures.learning import (  # noqa: E402
    MacroRollout,
    PredictiveActorCritic,
    PredictivePPOConfig,
    PredictivePPOTrainer,
)


def _config(**changes) -> PredictivePPOConfig:
    values = {
        "feature_dim": 7,
        "physiology_dim": 3,
        "context_dim": 5,
        "projection_dim": 4,
        "hidden_dim": 11,
        "epochs": 2,
        "minibatch_size": 8,
        "sequence_length": 4,
        "seed": 1701,
        "context_profile": "gated-v1",
    }
    values.update(changes)
    return PredictivePPOConfig(**values)


def _sequence_inputs(config: PredictivePPOConfig, steps: int = 7, residents: int = 3):
    generator = torch.Generator().manual_seed(82)
    features = torch.randn(steps, residents, config.feature_dim, generator=generator)
    physiology = torch.randn(
        steps, residents, config.physiology_dim, generator=generator
    )
    context = torch.randn(residents, config.context_dim, generator=generator)
    actions = torch.tanh(torch.randn(
        steps, residents, 8, generator=generator
    ))
    done = torch.zeros(steps, residents, dtype=torch.bool)
    done[2, 0] = True
    done[4, 2] = True
    return features, physiology, context, actions, done


def _online_sequence(model, features, physiology, context, actions, done):
    current = context.detach()
    outputs = ([], [], [])
    for index in range(len(features)):
        for values, value in zip(
            outputs, model(features[index], physiology[index], current), strict=True
        ):
            values.append(value)
        if index + 1 < len(features):
            current = model.next_context(current, features[index + 1], actions[index])
            current = current * (~done[index]).unsqueeze(-1)
    return tuple(torch.stack(values) for values in outputs)


def test_sequence_matches_online_calls_and_is_causal() -> None:
    config = _config()
    model = PredictiveActorCritic(config)
    inputs = _sequence_inputs(config)
    replay = model.sequence(*inputs)
    online = _online_sequence(model, *inputs)
    for replay_value, online_value in zip(replay, online, strict=True):
        torch.testing.assert_close(replay_value, online_value, rtol=0, atol=0)

    changed = list(inputs)
    changed[0] = changed[0].clone()
    changed[1] = changed[1].clone()
    changed[0][4:] += 100
    changed[1][4:] -= 100
    future_changed = model.sequence(*changed)
    for original, modified in zip(replay, future_changed, strict=True):
        torch.testing.assert_close(original[:4], modified[:4], rtol=0, atol=0)


def test_done_resets_only_that_resident_and_blocks_prior_episode_gradient() -> None:
    config = _config()
    model = PredictiveActorCritic(config)
    features, physiology, context, actions, done = _sequence_inputs(config)
    baseline = model.sequence(features, physiology, context, actions, done)

    modified_features = features.clone()
    modified_actions = actions.clone()
    modified_features[:3, 0] += 31
    modified_actions[:3, 0] *= -1
    modified = model.sequence(
        modified_features, physiology, context, modified_actions, done
    )
    for original, changed in zip(baseline, modified, strict=True):
        torch.testing.assert_close(original[:, 1], changed[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(original[3:, 0], changed[3:, 0], rtol=0, atol=0)

    gradient_features = features.clone().requires_grad_(True)
    gradient_actions = actions.clone().requires_grad_(True)
    mean, value, _ = model.sequence(
        gradient_features, physiology, context, gradient_actions, done
    )
    (mean[4, 0].square().sum() + value[4, 0].square()).backward()
    assert torch.count_nonzero(gradient_features.grad[:3, 0]) == 0
    assert torch.count_nonzero(gradient_actions.grad[:3, 0]) == 0
    assert torch.count_nonzero(gradient_features.grad[3:5, 0]) > 0


def test_delayed_loss_reaches_candidate_and_gate_parameters() -> None:
    config = _config()
    model = PredictiveActorCritic(config)
    features, physiology, context, actions, done = _sequence_inputs(config)
    done.zero_()
    mean, value, _ = model.sequence(features, physiology, context, actions, done)
    (mean[-1].square().mean() + value[-1].square().mean()).backward()

    candidate = (
        model.context_feature, model.context_action, model.context_recur,
    )
    gates = (
        model.context_gate_feature, model.context_gate_action,
        model.context_gate_recur, model.context_gate_bias,
    )
    assert all(parameter.grad is not None for parameter in (*candidate, *gates))
    assert sum(float(parameter.grad.abs().sum()) for parameter in candidate) > 0
    assert sum(float(parameter.grad.abs().sum()) for parameter in gates) > 0


def test_padded_sequence_chunks_include_each_real_sample_once() -> None:
    config = _config(sequence_length=4, minibatch_size=8)
    trainer = PredictivePPOTrainer(["a", "b", "c"], config)
    steps, residents = 6, 3
    data = {
        name: np.zeros((steps, residents), dtype=np.float32)
        for name in ("latent", "action", "prediction_target")
    }
    data["latent"] = np.zeros((steps, residents, 8), dtype=np.float32)
    data["action"] = np.zeros((steps, residents, 8), dtype=np.float32)
    data["prediction_target"] = np.zeros(
        (steps, residents, config.projection_dim), dtype=np.float32
    )
    data.update({
        "features": np.zeros((steps, residents, config.feature_dim), dtype=np.float32),
        "physiology": np.zeros((steps, residents, config.physiology_dim), dtype=np.float32),
        "context": np.zeros((steps, residents, config.context_dim), dtype=np.float32),
        "log_prob": np.zeros((steps, residents), dtype=np.float32),
        "value": np.zeros((steps, residents), dtype=np.float32),
        "reward": np.arange(steps * residents, dtype=np.float32).reshape(steps, residents),
        "done": np.zeros((steps, residents), dtype=bool),
    })
    observed = []
    for tensors, _outputs, _targets in trainer._optimization_batches(
        data, np.zeros((steps, residents), dtype=np.float32),
        np.zeros((steps, residents), dtype=np.float32),
    ):
        observed.extend(tensors["reward"].cpu().numpy().tolist())
    assert len(observed) == steps * residents
    assert sorted(observed) == list(np.arange(steps * residents, dtype=np.float32))


def _rollout_from_arrays(arrays: dict[str, np.ndarray]) -> MacroRollout:
    rollout = MacroRollout()
    for index in range(len(arrays["reward"])):
        rollout.append(**{name: values[index] for name, values in arrays.items()})
    return rollout


def _collect_rollout(
    trainer: PredictivePPOTrainer, *, steps: int, seed: int
) -> tuple[MacroRollout, np.ndarray]:
    rng = np.random.default_rng(seed)
    rollout = MacroRollout()
    features = rng.normal(size=(len(trainer.resident_ids), trainer.config.feature_dim)).astype(
        np.float32
    )
    physiology = rng.normal(
        size=(len(trainer.resident_ids), trainer.config.physiology_dim)
    ).astype(np.float32)
    for index in range(steps):
        previous = trainer.act(features, physiology)
        next_features = rng.normal(size=features.shape).astype(np.float32)
        reward = rng.normal(size=len(trainer.resident_ids)).astype(np.float32)
        done = np.zeros(len(trainer.resident_ids), dtype=bool)
        if index == steps - 2:
            done[0] = True
        transition = trainer.finish_transition(
            previous, next_features, reward, done, 0.25
        )
        rollout.append(
            features=previous["features"], physiology=previous["physiology"],
            context=previous["context"], latent=previous["latent"],
            action=previous["action"], log_prob=previous["log_prob"],
            value=previous["value"], reward=transition["reward"], done=done,
            prediction_target=transition["prediction_target"],
        )
        features = next_features
    return rollout, trainer.bootstrap_value(features, physiology)


def test_checkpoint_restore_reproduces_next_update_with_pending_rollout(
    tmp_path: Path,
) -> None:
    config = _config(epochs=2, minibatch_size=8)
    trainer = PredictivePPOTrainer(["a", "b", "c"], config)
    first, bootstrap = _collect_rollout(trainer, steps=6, seed=91)
    trainer.update(first, bootstrap, 0.25)
    pending, bootstrap = _collect_rollout(trainer, steps=7, seed=92)
    pending_arrays = pending.arrays()
    pending_path = tmp_path / "rollout.npz"
    np.savez_compressed(pending_path, **pending_arrays)
    receipt = trainer.snapshot(tmp_path / "learner.pt", extra={"rollout": pending_path.name})
    saved_torch_rng = torch.get_rng_state().clone()

    restored, extra = PredictivePPOTrainer.restore(
        receipt["path"], expected_sha256=receipt["sha256"]
    )
    assert extra == {"rollout": pending_path.name}
    torch.testing.assert_close(torch.get_rng_state(), saved_torch_rng, rtol=0, atol=0)
    assert json.dumps(restored.rng.bit_generator.state, sort_keys=True) == json.dumps(
        trainer.rng.bit_generator.state, sort_keys=True
    )
    np.testing.assert_array_equal(restored.context, trainer.context)

    with np.load(pending_path, allow_pickle=False) as value:
        restored_rollout = _rollout_from_arrays({
            name: np.asarray(value[name]) for name in MacroRollout.FIELDS
        })
    original_metrics = trainer.update(
        _rollout_from_arrays(copy.deepcopy(pending_arrays)), bootstrap.copy(), 0.25
    )
    restored_metrics = restored.update(restored_rollout, bootstrap.copy(), 0.25)
    assert original_metrics == restored_metrics
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_version_must_match_context_architecture(tmp_path: Path) -> None:
    trainer = PredictivePPOTrainer(["a", "b"], _config())
    path = tmp_path / "gated.pt"
    trainer.snapshot(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    assert value["version"] == 3
    value["version"] = 1
    tampered = tmp_path / "mislabeled.pt"
    torch.save(value, tampered)
    with pytest.raises(ValueError, match="version differs from context architecture"):
        PredictivePPOTrainer.restore(tampered)


def test_reservoir_context_update_matches_legacy_equation() -> None:
    config = _config(context_profile="reservoir-v1")
    model = PredictiveActorCritic(config)
    generator = torch.Generator().manual_seed(22)
    context = torch.randn(3, config.context_dim, generator=generator)
    features = torch.randn(3, config.feature_dim, generator=generator)
    actions = torch.randn(3, 8, generator=generator)
    projected = torch.tanh(features @ model.projection.T)
    expected = torch.tanh(
        projected @ model.context_feature.T
        + actions @ model.context_action.T
        + context @ model.context_recur.T
    )
    actual = model.next_context(context, features, actions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameter_names = dict(model.named_parameters())
    assert not any(name.startswith("context_") for name in parameter_names)


def test_gated_descendant_inherits_shared_model_with_fresh_optimizer_and_context() -> None:
    parent_config = _config(context_profile="reservoir-v1")
    parent = PredictivePPOTrainer(["a", "b", "c"], parent_config)
    parent.context[:] = np.arange(parent.context.size, dtype=np.float32).reshape(
        parent.context.shape
    )
    parent.moments.update(np.ones((3, parent_config.feature_dim), dtype=np.float32))
    parent.update_count = 9
    parent.decision_count = 41

    child_config = _config(context_profile="gated-v1")
    child = PredictivePPOTrainer(["a", "b", "c"], child_config)
    child.inherit_model(parent)
    parent_state = parent.model.state_dict()
    child_state = child.model.state_dict()
    for name, value in parent_state.items():
        torch.testing.assert_close(child_state[name], value, rtol=0, atol=0)
    assert set(child_state) - set(parent_state) == {
        "context_gate_feature", "context_gate_action",
        "context_gate_recur", "context_gate_bias",
    }
    assert child.optimizer.state_dict()["state"] == {}
    np.testing.assert_array_equal(child.context, np.zeros_like(child.context))
    assert child.moments.snapshot() == parent.moments.snapshot()
    assert (child.update_count, child.decision_count) == (9, 41)

    features = np.zeros((3, parent_config.feature_dim), dtype=np.float32)
    physiology = np.zeros((3, parent_config.physiology_dim), dtype=np.float32)
    parent.context.fill(0)
    parent_action = parent.act(features, physiology, deterministic=True)["action"]
    child_action = child.act(features, physiology, deterministic=True)["action"]
    np.testing.assert_array_equal(child_action, parent_action)
