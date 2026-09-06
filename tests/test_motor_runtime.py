import copy
from pathlib import Path

import numpy as np
import pytest

from chreatures.motor_inheritance import ACTIONS, MotorArtifact, MotorOrgan

ROOT = Path(__file__).parents[1]


def _linear(arrays, name, value):
    return value @ arrays[name + ".weight"].T + arrays[name + ".bias"]


def test_native_motor_runtime_real_artifact_batch_matches_numpy_and_replays():
    artifact = MotorArtifact.load(ROOT / "data/genomes/chemical-encounters-v1-step20000.npz")
    arrays = artifact.arrays
    config = artifact.config
    rng = np.random.default_rng(771)
    batch = 3
    features = rng.normal(size=(batch, config["feature_dim"])).astype(np.float32)
    physiology = rng.normal(size=(batch, config["physiology_dim"])).astype(np.float32)
    context = rng.normal(size=(batch, config["context_dim"])).astype(np.float32)
    action = rng.normal(size=(batch, len(ACTIONS))).astype(np.float32)
    previous = rng.normal(size=features.shape).astype(np.float32)
    prediction = rng.normal(size=(batch, config["projection_dim"])).astype(np.float32)

    encoded = np.tanh(_linear(arrays, "feature_encoder.0", features)).astype(np.float32)
    joined = np.concatenate((encoded, physiology, context), axis=1).astype(np.float32)
    hidden = np.tanh(_linear(arrays, "trunk.0", joined)).astype(np.float32)
    hidden = np.tanh(_linear(arrays, "trunk.2", hidden)).astype(np.float32)
    mean = _linear(arrays, "policy_mean", hidden).astype(np.float32)
    value = _linear(arrays, "value", hidden)[:, 0].astype(np.float32)
    native_mean, native_value, native_hidden = artifact.runtime.forward(
        features, physiology, context
    )
    np.testing.assert_allclose(native_mean, mean, rtol=0, atol=2e-6)
    np.testing.assert_allclose(native_value, value, rtol=0, atol=2e-6)
    np.testing.assert_allclose(native_hidden, hidden, rtol=0, atol=2e-6)

    projected = np.tanh(features @ arrays["projection"].T).astype(np.float32)
    old_projected = np.tanh(previous @ arrays["projection"].T).astype(np.float32)
    expected_context = np.tanh(
        projected @ arrays["context_feature"].T
        + action @ arrays["context_action"].T
        + context @ arrays["context_recur"].T
    ).astype(np.float32)
    expected_error = np.mean(
        (projected - old_projected - prediction) ** 2, axis=1, dtype=np.float32
    )
    native_context, native_projected, native_error = artifact.runtime.update_context(
        features, action, context, previous, prediction, np.ones(batch, dtype=np.bool_)
    )
    np.testing.assert_allclose(native_context, expected_context, rtol=0, atol=2e-6)
    np.testing.assert_allclose(native_projected, projected, rtol=0, atol=2e-6)
    np.testing.assert_allclose(native_error, expected_error, rtol=0, atol=2e-6)

    original = MotorOrgan(artifact, seed=991)
    for row in features:
        original.tick(row, physiology[0], 0.05)
    restored = MotorOrgan.restore_value(original.snapshot_value(), artifact)
    for row in np.tile(features, (4, 1)):
        assert original.tick(row, physiology[0], 0.05) == restored.tick(
            row, physiology[0], 0.05
        )
        np.testing.assert_array_equal(original.context, restored.context)

    incompatible = copy.deepcopy(original.snapshot_value())
    incompatible["runtime_identity"] = "chreatures-motor-runtime-v1:other-arithmetic"
    with pytest.raises(ValueError, match="arithmetic identity"):
        MotorOrgan.restore_value(incompatible, artifact)

    legacy = copy.deepcopy(original.snapshot_value())
    legacy.update(format="chreatures-motor-organ-snapshot-v1", version=1)
    legacy.pop("runtime_identity")
    with pytest.raises(ValueError, match="explicit trajectory migration"):
        MotorOrgan.restore_value(legacy, artifact)


def test_native_motor_runtime_rejects_bad_batch_before_computation():
    artifact = MotorArtifact.load(ROOT / "data/genomes/nursery-8000.npz")
    config = artifact.config
    features = np.zeros((1, config["feature_dim"]), dtype=np.float32)
    physiology = np.zeros((1, config["physiology_dim"]), dtype=np.float32)
    context = np.zeros((1, config["context_dim"]), dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        artifact.runtime.forward(features + np.float32(np.nan), physiology, context)
    with pytest.raises(ValueError, match="shape"):
        artifact.runtime.forward(features[:, :-1], physiology, context)
