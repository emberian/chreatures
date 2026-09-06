#!/usr/bin/env python3
"""Export a trusted Torch sensorimotor worker into the native NPZ contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEVELOPMENTAL_FORMAT = "chreatures-native-developmental-resident-rich-v2"
RICH_DEVELOPMENT_CHECKPOINT_FORMAT = (
    "chreatures-rich-online-sensorimotor-development-v1"
)
RICH_PROFILE_SHA256 = "c71380718ba5535dbaebdeaf8aa2e88cc45cf218312a03e13507877f02a5554e"
RICH_CHANNEL_NAMES_SHA256 = (
    "b4c6b328116d820143e16ee922ccffd7b950dbe008efc580ad93056e01349bfa"
)
MANAGER_ORDER = (
    "manager.query.0.weight",
    "manager.query.0.bias",
    "manager.query.2.weight",
    "manager.query.2.bias",
    "manager.query_gain",
)
RICH_MODEL_NAMES = (
    "visual.peripheral.first.weight",
    "visual.peripheral.first.bias",
    "visual.peripheral.second.weight",
    "visual.peripheral.second.bias",
    "visual.foveal.first.weight",
    "visual.foveal.first.bias",
    "visual.foveal.second.weight",
    "visual.foveal.second.bias",
    "visual.peripheral_projection.0.weight",
    "visual.peripheral_projection.0.bias",
    "visual.foveal_projection.0.weight",
    "visual.foveal_projection.0.bias",
    "body.0.weight",
    "body.0.bias",
    "goal_encoder.0.weight",
    "goal_encoder.0.bias",
    "goal_encoder.2.weight",
    "goal_encoder.2.bias",
    "observation_projection.0.weight",
    "observation_projection.0.bias",
    "history.weight_ih_l0",
    "history.weight_hh_l0",
    "history.bias_ih_l0",
    "history.bias_hh_l0",
    "policy_trunk.0.weight",
    "policy_trunk.0.bias",
    "signed_head.weight",
    "signed_head.bias",
    "active_head.weight",
    "active_head.bias",
    "positive_head.weight",
    "positive_head.bias",
)
PREDICTOR_ENCODER_NAMES = (
    "visual.peripheral.first.weight",
    "visual.peripheral.first.bias",
    "visual.peripheral.second.weight",
    "visual.peripheral.second.bias",
    "visual.foveal.first.weight",
    "visual.foveal.first.bias",
    "visual.foveal.second.weight",
    "visual.foveal.second.bias",
    "visual.peripheral_projection.0.weight",
    "visual.peripheral_projection.0.bias",
    "visual.foveal_projection.0.weight",
    "visual.foveal_projection.0.bias",
    "body.0.weight",
    "body.0.bias",
    "goal_encoder.0.weight",
    "goal_encoder.0.bias",
    "goal_encoder.2.weight",
    "goal_encoder.2.bias",
)
RICH_ORDER = (
    ("normalizer.mean", "normalizer.scale")
    + tuple("model." + name for name in RICH_MODEL_NAMES)
    + MANAGER_ORDER
)
RICH_SHAPES = {
    "normalizer.mean": (4453,),
    "normalizer.scale": (4453,),
    "model.visual.peripheral.first.weight": (16, 4, 3, 3),
    "model.visual.peripheral.first.bias": (16,),
    "model.visual.peripheral.second.weight": (24, 16, 3, 3),
    "model.visual.peripheral.second.bias": (24,),
    "model.visual.foveal.first.weight": (16, 4, 3, 3),
    "model.visual.foveal.first.bias": (16,),
    "model.visual.foveal.second.weight": (24, 16, 3, 3),
    "model.visual.foveal.second.bias": (24,),
    "model.visual.peripheral_projection.0.weight": (64, 768),
    "model.visual.peripheral_projection.0.bias": (64,),
    "model.visual.foveal_projection.0.weight": (64, 2304),
    "model.visual.foveal_projection.0.bias": (64,),
    "model.body.0.weight": (128, 357),
    "model.body.0.bias": (128,),
    "model.goal_encoder.0.weight": (256, 1024),
    "model.goal_encoder.0.bias": (256,),
    "model.goal_encoder.2.weight": (64, 256),
    "model.goal_encoder.2.bias": (64,),
    "model.observation_projection.0.weight": (128, 265),
    "model.observation_projection.0.bias": (128,),
    "model.history.weight_ih_l0": (384, 128),
    "model.history.weight_hh_l0": (384, 128),
    "model.history.bias_ih_l0": (384,),
    "model.history.bias_hh_l0": (384,),
    "model.policy_trunk.0.weight": (256, 201),
    "model.policy_trunk.0.bias": (256,),
    "model.signed_head.weight": (260, 256),
    "model.signed_head.bias": (260,),
    "model.active_head.weight": (4, 256),
    "model.active_head.bias": (4,),
    "model.positive_head.weight": (128, 256),
    "model.positive_head.bias": (128,),
    "manager.query.0.weight": (128, 518),
    "manager.query.0.bias": (128,),
    "manager.query.2.weight": (64, 128),
    "manager.query.2.bias": (64,),
    "manager.query_gain": (1,),
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    clean = copy.deepcopy(metadata)
    clean.pop("artifact_sha256", None)
    digest = hashlib.sha256(canonical(clean))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(canonical(list(value.shape)))
        digest.update(value.tobytes())
    return digest.hexdigest()


def predictor_identity(metadata, arrays):
    clean = dict(metadata)
    clean.pop("artifact_identity", None)
    receipts = {
        name: {
            "dtype": np.ascontiguousarray(value).dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }
    return hashlib.sha256(
        canonical({"metadata": clean, "arrays": receipts})
    ).hexdigest()


def export_rich(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    output: Path,
    laws_path: Path,
    predictor_path: Path,
) -> None:
    identity = checkpoint.get("identity", {})
    if identity.get("controller_mode") != "rich-achieved-goal":
        raise ValueError("rich checkpoint controller mode differs")
    normalizer = identity.get("normalizer", {})
    normalizer_body = dict(normalizer)
    normalizer_hash = normalizer_body.pop("sha256", None)
    if (
        normalizer_body.get("format") != "chreatures-rich-observation-normalizer-v1"
        or normalizer_body.get("version") != 1
        or normalizer_hash != hashlib.sha256(canonical(normalizer_body)).hexdigest()
        or identity.get("rich_profile_sha256") != RICH_PROFILE_SHA256
        or identity.get("rich_channel_names_sha256") != RICH_CHANNEL_NAMES_SHA256
    ):
        raise ValueError("rich checkpoint observation identity differs")
    model = checkpoint.get("model", {})
    manager = checkpoint.get("goal_manager", {})
    source = {
        "normalizer.mean": normalizer["mean"],
        "normalizer.scale": normalizer["scale"],
        **{"model." + name: model[name] for name in RICH_MODEL_NAMES},
        "manager.query.0.weight": manager["query.0.weight"],
        "manager.query.0.bias": manager["query.0.bias"],
        "manager.query.2.weight": manager["query.2.weight"],
        "manager.query.2.bias": manager["query.2.bias"],
        "manager.query_gain": manager["query_gain"].reshape(1),
    }
    arrays = {
        name: np.ascontiguousarray(
            value.detach().cpu().numpy() if hasattr(value, "detach") else value,
            dtype=np.float32,
        )
        for name, value in source.items()
    }
    predictor_names = [
        "input.mean",
        "input.scale",
        "target.mean",
        "target.scale",
        "residual.scale",
    ] + [
        f"member.{m}.{part}"
        for m in range(3)
        for part in (
            "layer0.weight",
            "layer0.bias",
            "layer1.weight",
            "layer1.bias",
            "output.weight",
            "output.bias",
        )
    ]
    with np.load(predictor_path, allow_pickle=False) as archive:
        predictor_metadata = json.loads(str(archive["metadata"]))
        all_predictor_arrays = {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
        predictor_arrays = {
            "predictor." + name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in predictor_names
        }
        if set(archive.files) != {"metadata", *predictor_metadata.get("tensors", {})}:
            raise ValueError("predictor tensor set differs")
        for name in predictor_names:
            raw = np.asarray(archive[name])
            receipt = predictor_metadata.get("tensors", {}).get(name, {})
            if (
                raw.dtype != np.float32
                or list(raw.shape) != receipt.get("shape")
                or hashlib.sha256(raw.tobytes()).hexdigest() != receipt.get("sha256")
            ):
                raise ValueError(f"predictor tensor receipt differs: {name}")
    if (
        predictor_metadata.get("format") != "chreatures-rich-consequence-ensemble-v1"
        or predictor_identity(predictor_metadata, all_predictor_arrays)
        != predictor_metadata.get("artifact_identity")
        or not isinstance(
            predictor_metadata.get("source", {}).get("frame_encoder_sha256"), str
        )
        or len(predictor_metadata["source"]["frame_encoder_sha256"]) != 64
    ):
        raise ValueError("predictor identity differs")
    with np.load(predictor_path, allow_pickle=False) as archive:
        for name in PREDICTOR_ENCODER_NAMES:
            if not np.array_equal(
                np.asarray(archive["encoder." + name]), arrays["model." + name]
            ):
                raise ValueError(f"predictor/controller frozen encoder differs: {name}")
        if not np.array_equal(
            archive["observation_normalizer.mean"], arrays["normalizer.mean"]
        ) or not np.array_equal(
            archive["observation_normalizer.scale"], arrays["normalizer.scale"]
        ):
            raise ValueError("predictor/controller observation normalizer differs")
    arrays.update(predictor_arrays)
    full_order = RICH_ORDER + tuple("predictor." + x for x in predictor_names)
    if (
        tuple(arrays) != full_order
        or any(arrays[name].shape != RICH_SHAPES[name] for name in RICH_ORDER)
        or any(
            not value.flags.c_contiguous or not np.isfinite(value).all()
            for value in arrays.values()
        )
    ):
        raise ValueError("rich export tensor order, shape, or values differ")
    tensors = {
        name: {
            "dtype": "float32",
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in arrays.items()
    }
    law_bytes = laws_path.read_bytes()
    law_bank = json.loads(law_bytes)
    if law_bank.get("schema") != "chreatures-gam-consequence-law-bank-v1":
        raise ValueError("GAM law schema differs")
    refinement = {
        "law_bank": law_bank,
        "law_file_sha256": hashlib.sha256(law_bytes).hexdigest(),
        "law_content_sha256": hashlib.sha256(canonical(law_bank)).hexdigest(),
        "candidates": 4,
        "tilt": 0.5,
        "learning_rate": 0.05,
        "error_decay": 0.99,
        "innovation_limit": 4.0,
    }
    metadata = {
        "format": DEVELOPMENTAL_FORMAT,
        "version": 2,
        "execution": "developmental-resident-native-rich-predictive-v2",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "format": RICH_DEVELOPMENT_CHECKPOINT_FORMAT,
            "updates": int(checkpoint.get("updates", 0)),
            "physical_steps": int(checkpoint.get("physical_steps", 0)),
        },
        "observation_contract": {
            "format": "chreatures-rich-sensorimotor-observation-v1",
            "observation_dim": 4453,
            "rich_body_dim": 4096,
            "rich_profile_sha256": RICH_PROFILE_SHA256,
            "rich_channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
            "observation_order": [
                "rich_body_v1_4096",
                "canonical_channels_351",
                "physiology_6",
            ],
            "source_sense_dim": 351,
            "physiology_dim": 6,
            "neural_readout_dim": 384,
            "previous_action_plus_oral_dim": 9,
        },
        "normalizer_sha256": normalizer_hash,
        "consequence_refinement": refinement,
        "predictor": {
            "file_sha256": sha256(predictor_path),
            "artifact_identity": predictor_metadata["artifact_identity"],
            "metadata": predictor_metadata,
            "goal_forecast_rms": max(
                float(
                    predictor_metadata["validation"][
                        "runtime_empirical_goal_error_scale"
                    ]
                ),
                1e-4,
            ),
            "pack_order": ["predictor." + x for x in predictor_names],
        },
        "training_identity": json.loads(canonical(identity)),
        "tensors": tensors,
        "pack_order": list(full_order),
        "temporal_contract": {
            "observation_interval_seconds": 0.05,
            "reset": "reset-before-row",
            "previous": "actual-executed-action-8-plus-oral-1",
            "manager_commit_ticks": 10,
        },
    }
    metadata["artifact_sha256"] = artifact_identity(metadata, arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, metadata=np.asarray(canonical(metadata).decode()), **arrays
        )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "file_sha256": sha256(output),
                "artifact_sha256": metadata["artifact_sha256"],
                "mode": "rich-achieved-goal",
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--consequence-laws", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    args = parser.parse_args()
    if not args.trusted_checkpoint:
        raise SystemExit("Torch deserialization requires --trusted-checkpoint")
    if args.output.exists():
        raise SystemExit("output already exists")

    import torch

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != RICH_DEVELOPMENT_CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not the current rich joined controller")
    export_rich(
        checkpoint,
        checkpoint_path,
        args.output,
        args.consequence_laws.expanduser().resolve(),
        args.predictor.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
