#!/usr/bin/env python3
"""Compare native cohort recurrence and imagination with the Torch organ."""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chreatures.predictive_native import NativePredictiveCohort
from chreatures.predictive_state import PredictiveSequence, PredictiveStateTrainer


def delta(a, b):
    return float(
        np.max(
            np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)),
            initial=0,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    trainer = PredictiveStateTrainer.restore(a.checkpoint)
    seq = (
        PredictiveSequence.from_episode(a.rollout, "train")
        if a.rollout.is_dir()
        else PredictiveSequence.from_rollout(a.rollout)
    )
    batch = seq.features.shape[1]
    native = NativePredictiveCohort(a.export, batch)
    torch_states = trainer.encode(seq.features, seq.physiology, seq.actions, seq.reset)
    observed = []
    zero = np.zeros((batch, seq.actions.shape[2]), np.float32)
    for t in range(len(seq.features)):
        previous = zero if t == 0 else seq.actions[t - 1]
        observed.append(
            native.observe(seq.features[t], seq.physiology[t], previous, seq.reset[t])
        )
    observe_delta = delta(np.stack(observed), torch_states)
    rng = np.random.default_rng(4401)
    candidates = rng.uniform(-1, 1, size=(6, batch, seq.actions.shape[2])).astype(
        np.float32
    )
    with torch.no_grad():
        expected = trainer.imagine_physical(
            torch.as_tensor(torch_states[-1]),
            torch.as_tensor(seq.physiology[-1]),
            torch.as_tensor(candidates),
        )
    actual = native.imagine(candidates)
    imagine = {
        "feature_mean": delta(actual["feature_mean"], expected["feature_mean"].numpy()),
        "physiology_mean": delta(
            actual["physiology_mean"], expected["physiology_mean"].numpy()
        ),
        "feature_residual_scale": delta(
            actual["feature_residual_scale"], expected["feature_residual_scale"].numpy()
        ),
        "physiology_residual_scale": delta(
            actual["physiology_residual_scale"],
            expected["physiology_residual_scale"].numpy(),
        ),
        "valid": delta(actual["valid"], expected["valid"].numpy()),
        "support": delta(
            actual["horizon_support"], expected["horizon_support"].numpy()
        ),
    }
    snap = native.snapshot()
    f = seq.features[-1]
    pbody = seq.physiology[-1]
    pa = seq.actions[-1]
    rst = np.zeros(batch, bool)
    first = native.observe(f, pbody, pa, rst)
    first_dream = native.imagine(candidates)["feature_mean"]
    native.restore(snap)
    second = native.observe(f, pbody, pa, rst)
    second_dream = native.imagine(candidates)["feature_mean"]
    identity_rejected = False
    with tempfile.TemporaryDirectory() as directory:
        other = PredictiveStateTrainer(trainer.config, trainer.normalizer)
        with torch.no_grad():
            next(iter(other.model.parameters())).add_(0.01)
        other_path = Path(directory) / "other.npz"
        other.export(other_path)
        other_native = NativePredictiveCohort(other_path, batch)
        try:
            other_native.restore(snap)
        except ValueError:
            identity_rejected = True

    def measure(fn, repeats=8):
        values = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            values.append((time.perf_counter() - start) * 1000)
        return statistics.median(values)

    def native_observe_sequence():
        for t in range(len(seq.features)):
            native.observe(
                seq.features[t],
                seq.physiology[t],
                zero if t == 0 else seq.actions[t - 1],
                seq.reset[t],
            )

    torch_observe_ms = measure(
        lambda: trainer.encode(seq.features, seq.physiology, seq.actions, seq.reset)
    )
    native_observe_ms = measure(native_observe_sequence)
    with torch.no_grad():
        torch_imagine_ms = measure(
            lambda: trainer.imagine_physical(
                torch.as_tensor(torch_states[-1]),
                torch.as_tensor(seq.physiology[-1]),
                torch.as_tensor(candidates),
            )
        )
    native_imagine_ms = measure(lambda: native.imagine(candidates))
    report = {
        "observe_state_max_abs_delta": observe_delta,
        "imagine_max_abs_delta": imagine,
        "snapshot_continuation_state_max_abs_delta": delta(first, second),
        "snapshot_continuation_imagine_max_abs_delta": delta(first_dream, second_dream),
        "same_shape_other_model_snapshot_rejected": identity_rejected,
        "rows": len(seq.features),
        "residents": batch,
        "cpu_median_ms": {
            "torch_observe_sequence": torch_observe_ms,
            "native_observe_sequence": native_observe_ms,
            "torch_imagine_6": torch_imagine_ms,
            "native_imagine_6": native_imagine_ms,
        },
    }
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
