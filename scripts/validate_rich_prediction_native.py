#!/usr/bin/env python3
"""Compare the recurrent rich predictor's Torch and native inference paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import _cognitive_core
from research.sensorimotor_skills.rich_prediction import (
    RichRecurrentConsequenceEnsemble,
    bounded_physiology_deltas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol <= 0:
        parser.error("--atol must be finite and positive")
    return args


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    with np.load(args.predictor, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in metadata["pack_order"]
        }

    model = RichRecurrentConsequenceEnsemble().eval()
    state = {}
    names = (
        "context.weight",
        "context.bias",
        "transition.weight_ih",
        "transition.weight_hh",
        "transition.bias_ih",
        "transition.bias_hh",
        "output.weight",
        "output.bias",
    )
    for member in range(3):
        for name in names:
            state[f"members.{member}.{name}"] = torch.from_numpy(
                arrays[f"member.{member}.{name}"]
            )
    model.load_state_dict(state, strict=True)

    rng = np.random.default_rng(args.seed)
    batch, candidates, horizon = 3, 4, 4
    context_mean, context_scale = arrays["context.mean"], arrays["context.scale"]
    action_mean, action_scale = arrays["action.mean"], arrays["action.scale"]
    context = np.ascontiguousarray(
        context_mean
        + rng.normal(0, 0.35, (batch, 1560)).astype(np.float32) * context_scale
    )
    actions = np.ascontiguousarray(
        action_mean
        + rng.normal(0, 0.35, (batch, candidates, horizon, 12)).astype(np.float32)
        * action_scale
    )
    lower = np.array([0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0], np.float32)
    physiology = np.ascontiguousarray(
        rng.uniform(lower + 0.05, np.ones(12, np.float32) - 0.05, (batch, 12))
        .astype(np.float32)
    )

    normalized_context = torch.from_numpy((context - context_mean) / context_scale)
    normalized_actions = torch.from_numpy((actions - action_mean) / action_scale)
    with torch.inference_mode():
        raw = model(
            normalized_context[:, None]
            .expand(batch, candidates, 1560)
            .reshape(batch * candidates, 1560),
            normalized_actions.reshape(batch * candidates, horizon, 12),
        )
        raw = raw * torch.from_numpy(arrays["target.scale"]) + torch.from_numpy(
            arrays["target.mean"]
        )
        raw = raw.reshape(batch, candidates, 3, horizon, 268)
        anchors = (
            torch.from_numpy(physiology)[:, None, None]
            .expand(batch, candidates, 3, 12)
            .reshape(batch * candidates * 3, 12)
        )
        decoded = bounded_physiology_deltas(
            raw[..., 256:].reshape(batch * candidates * 3, horizon, 12), anchors
        ).reshape(batch, candidates, 3, horizon, 12)
        members = raw.clone()
        members[..., 256:] = decoded
        absolute_code = (
            torch.from_numpy(context[:, 768:1024])[:, None, None, None]
            + members[..., :256].cumsum(3)
        )
        absolute_physiology = (
            torch.from_numpy(physiology)[:, None, None, None] + decoded.cumsum(3)
        )
        mean = members.mean(2)
        disagreement = ((members - mean[:, :, None]) ** 2).mean(2).sqrt()

    packed = np.concatenate(
        [arrays[name].reshape(-1) for name in metadata["pack_order"]]
    ).astype(np.float32)
    native = _cognitive_core.PredictiveSensoryEnsemble(packed)
    result = native.forecast_sequences(context, actions, physiology)
    expected = {
        "member_delta": members.numpy(),
        "mean_delta": mean.numpy(),
        "disagreement": disagreement.numpy(),
        "absolute_code": absolute_code.numpy(),
        "absolute_physiology": absolute_physiology.numpy(),
    }
    receipt = {
        "format": "chreatures-recurrent-predictor-native-parity-v1",
        "seed": args.seed,
        "shape": {"B": batch, "K": candidates, "H": horizon},
        "predictor_file_sha256": sha256(args.predictor),
        "predictor_artifact_identity": metadata["artifact_identity"],
        "native_extension_sha256": sha256(Path(_cognitive_core.__file__)),
        "max_abs": {
            name: float(np.max(np.abs(np.asarray(result[name]) - value)))
            for name, value in expected.items()
        },
        "valid_all": bool(np.asarray(result["valid"]).all()),
    }
    receipt["absolute_tolerance"] = args.atol
    receipt["passed"] = receipt["valid_all"] and all(
        np.isfinite(error) and error <= args.atol
        for error in receipt["max_abs"].values()
    )
    receipt["max_abs"] = {
        name: error if np.isfinite(error) else None
        for name, error in receipt["max_abs"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit("Torch/native prediction parity exceeded its tolerance")


if __name__ == "__main__":
    main()
