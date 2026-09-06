from __future__ import annotations

import gzip
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest


def _load_runner():
    """Load storage helpers without importing the optional Torch runtime."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "learn_affordances.py"
    spec = importlib.util.spec_from_file_location("evaluation_resume_runner", path)
    module = importlib.util.module_from_spec(spec)
    process = mp.current_process()
    original_name = process.name
    process.name = "EvaluationReceiptTest"
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        process.name = original_name
    return module


def _write(path: Path, content: bytes) -> dict[str, object]:
    path.write_bytes(content)
    runner = _load_runner()
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": runner.sha256(path),
    }


def test_final_step_resume_reuses_condition_and_rejects_changed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    output = tmp_path / "run"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    learner = _write(checkpoint_dir / "learner-step-0000020.pt", b"learner")
    neural_file = checkpoint_dir / "neural-step-0000020.npz"
    neural_file.write_bytes(b"neural")
    neural = {
        "name": "neural-step-0000020", "bytes": neural_file.stat().st_size,
        "sha256": runner.sha256(neural_file),
    }
    rollout = _write(checkpoint_dir / "rollout-step-0000020.npz", b"rollout")
    rollout["path"] = Path(str(rollout["path"])).name
    rollout["length"] = 0
    state = {
        "step": 20,
        "learner": {**learner, "path": "/different/host/learner-step-0000020.pt"},
        "neural": neural,
        "rollout": rollout,
    }
    checkpoint = checkpoint_dir / "cohort-step-0000020.json.gz"
    with gzip.open(checkpoint, "wt", encoding="utf-8") as handle:
        json.dump(state, handle)

    recovered = runner.existing_checkpoint_receipt(checkpoint)
    stable = runner.stable_checkpoint_identity(recovered)
    moved_receipt = json.loads(json.dumps(recovered))
    moved_receipt["cohort"] = "/another/host/cohort-step-0000020.json.gz"
    moved_receipt["learner"]["path"] = "/another/host/learner-step-0000020.pt"
    assert runner.stable_checkpoint_identity(moved_receipt) == stable

    (output / "updates.jsonl").write_text(json.dumps({
        "step": 20, "elapsed_seconds": 12.5,
        "timing_cumulative_seconds": {"brain": 7.0, "physics": 4.0},
    }) + "\n")
    assert runner.completed_training_receipt(output, 20) == {
        "elapsed_seconds": 12.5,
        "timing_cumulative_seconds": {"brain": 7.0, "physics": 4.0},
    }

    identity = {
        "training_checkpoint": stable,
        "learner_architecture": {"std_profile": "state-conditioned-v2"},
        "neural": {"requested_backend": "tiled", "dynamics": {"substeps": 2}},
        "runtime": {
            "native_world_kernels": {"sha256": "a" * 64},
            "libraries": {"torch": "2.9.1+rocm6.3"},
        },
    }
    calls = []
    monkeypatch.setattr(
        runner, "evaluate", lambda *args, **kwargs: calls.append(kwargs) or {"score": 1.0},
    )
    genome = tmp_path / "genome.npz"
    np.savez(genome, placeholder=np.asarray(1))
    result, first_receipt = runner.persisted_evaluation(
        output, "learned", identity, object(), object(), genome, object(), marker="first",
    )
    assert result == {"score": 1.0}
    assert first_receipt["reused"] is False
    receipt_path = Path(first_receipt["path"])
    original_sha = runner.sha256(receipt_path)

    monkeypatch.setattr(
        runner, "evaluate",
        lambda *args, **kwargs: pytest.fail("completed condition was evaluated again"),
    )
    resumed_identity = json.loads(json.dumps(identity))
    resumed_identity["training_checkpoint"] = runner.stable_checkpoint_identity(
        moved_receipt
    )
    resumed_result, resumed_receipt = runner.persisted_evaluation(
        output, "learned", resumed_identity, object(), object(), genome, object(),
    )
    assert resumed_result == result
    assert resumed_receipt["reused"] is True
    assert runner.sha256(receipt_path) == original_sha
    assert len(calls) == 1

    changed_identity = json.loads(json.dumps(resumed_identity))
    changed_identity["learner_architecture"]["std_profile"] = "global-v1"
    with pytest.raises(ValueError, match="persisted evaluation identity differs"):
        runner.persisted_evaluation(
            output, "learned", changed_identity,
            object(), object(), genome, object(),
        )
