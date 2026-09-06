import copy
from pathlib import Path

import numpy as np
import pytest

from chreatures.motor_inheritance import (
    ACTIONS,
    ARTIFACT_FORMAT_V3,
    MotorArtifact,
    MotorOrgan,
    artifact_identity,
)


ROOT = Path(__file__).parents[1]


def gated_artifact() -> MotorArtifact:
    rng = np.random.default_rng(73)
    f, p, x, h, q = 7, 6, 4, 9, 5
    shapes = {
        "log_std": (len(ACTIONS),),
        "projection": (q, f),
        "context_feature": (x, q),
        "context_action": (x, len(ACTIONS)),
        "context_recur": (x, x),
        "context_gate_feature": (x, q),
        "context_gate_action": (x, len(ACTIONS)),
        "context_gate_recur": (x, x),
        "context_gate_bias": (x,),
        "feature_encoder.0.weight": (h, f),
        "feature_encoder.0.bias": (h,),
        "trunk.0.weight": (h, h + p + x),
        "trunk.0.bias": (h,),
        "trunk.2.weight": (h, h),
        "trunk.2.bias": (h,),
        "policy_mean.weight": (len(ACTIONS), h),
        "policy_mean.bias": (len(ACTIONS),),
        "value.weight": (1, h),
        "value.bias": (1,),
        "predictor.0.weight": (h, h + len(ACTIONS)),
        "predictor.0.bias": (h,),
        "predictor.2.weight": (q, h),
        "predictor.2.bias": (q,),
        "std_offset.weight": (len(ACTIONS), h),
        "std_offset.bias": (len(ACTIONS),),
    }
    arrays = {
        name: rng.normal(0, 0.12, shape).astype(np.float32)
        for name, shape in shapes.items()
    }
    arrays.update({
        "normalizer_count": np.asarray(40.0, dtype=np.float64),
        "normalizer_mean": np.linspace(-0.2, 0.2, f, dtype=np.float64),
        "normalizer_m2": np.linspace(30.0, 50.0, f, dtype=np.float64),
    })
    metadata = {
        "format": ARTIFACT_FORMAT_V3,
        "version": 3,
        "architecture": "gated-v1",
        "actions": list(ACTIONS),
        "config": {
            "feature_dim": f,
            "physiology_dim": p,
            "context_dim": x,
            "hidden_dim": h,
            "projection_dim": q,
            "macro_steps": 5,
            "seed": 19,
            "std_profile": "state-conditioned-v2",
            "context_profile": "gated-v1",
        },
        "training_provenance": {
            "checkpoint_sha256": "1" * 64,
            "graph_sha256": "2" * 64,
            "port_spec_sha256": "3" * 64,
        },
    }
    metadata["artifact_sha256"] = artifact_identity(metadata, arrays)
    return MotorArtifact(metadata, arrays)


def test_gated_v3_equation_shape_validation_and_private_continuation():
    artifact = gated_artifact()
    organ = MotorOrgan(artifact, seed=101)
    organ.context[:] = np.asarray([0.4, -0.3, 0.2, -0.1], dtype=np.float32)
    organ.held_action[:] = np.linspace(-0.6, 0.6, len(ACTIONS), dtype=np.float32)
    next_features = organ.normalize(np.linspace(-1, 1, 7, dtype=np.float32))
    before = organ.context.copy()
    projected = organ.projected(next_features)
    arrays = artifact.arrays
    candidate = np.tanh(
        projected @ arrays["context_feature"].T
        + organ.held_action @ arrays["context_action"].T
        + before @ arrays["context_recur"].T
    ).astype(np.float32)
    logits = (
        projected @ arrays["context_gate_feature"].T
        + organ.held_action @ arrays["context_gate_action"].T
        + before @ arrays["context_gate_recur"].T
        + arrays["context_gate_bias"]
    ).astype(np.float32)
    gate = (np.float32(1) / (np.float32(1) + np.exp(-logits))).astype(np.float32)
    expected = (before + gate * (candidate - before)).astype(np.float32)
    organ._update_context(next_features)
    np.testing.assert_allclose(organ.context, expected, rtol=0, atol=2e-6)

    malformed = {name: value.copy() for name, value in artifact.arrays.items()}
    malformed["context_gate_recur"] = malformed["context_gate_recur"][:-1]
    metadata = copy.deepcopy(artifact.metadata)
    metadata["artifact_sha256"] = artifact_identity(metadata, malformed)
    with pytest.raises(ValueError, match="context_gate_recur.*expected"):
        MotorArtifact(metadata, malformed)

    wrong_architecture = copy.deepcopy(artifact.metadata)
    wrong_architecture["architecture"] = "reservoir-v1"
    wrong_architecture["artifact_sha256"] = artifact_identity(
        wrong_architecture, artifact.arrays
    )
    with pytest.raises(ValueError, match="architecture metadata differs"):
        MotorArtifact(wrong_architecture, artifact.arrays)

    original = MotorOrgan(artifact, seed=211)
    physiology = np.asarray([0.7, 0.1, 0.2, 0.05, -0.1, 1.0], dtype=np.float32)
    inputs = [
        np.sin(np.arange(7, dtype=np.float32) + index / 3)
        for index in range(18)
    ]
    for features in inputs[:3]:
        original.tick(features, physiology, 0.05)
    restored = MotorOrgan.restore_value(
        original.snapshot_value(include_artifact=True)
    )
    for features in inputs[3:]:
        assert original.tick(features, physiology, 0.05) == restored.tick(
            features, physiology, 0.05
        )
        np.testing.assert_array_equal(original.context, restored.context)
        np.testing.assert_array_equal(original.held_action, restored.held_action)
        assert original.held_ticks == restored.held_ticks
        assert original.decision_count == restored.decision_count

    for path in sorted((ROOT / "data/genomes").glob("*.npz")):
        legacy = MotorArtifact.load(path)
        assert legacy.sha256 == legacy.metadata["artifact_sha256"]
        assert legacy.config.get("context_profile", "reservoir-v1") == "reservoir-v1"
