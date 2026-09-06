#!/usr/bin/env python3
"""Export one trusted v4 Torch population controller to immutable native NPZ."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.organism_interface import identity as organism_identity
from chreatures.sensorimotor_worker_native import PERSONAL_GOAL_CONTRACT

FORMAT = "chreatures-native-developmental-resident-population-v4"
EXECUTION = "developmental-resident-native-population-v4"
CHECKPOINT_FORMAT = "chreatures-rich-online-sensorimotor-development-v4"
V3_CHECKPOINT_FORMAT = "chreatures-rich-online-sensorimotor-development-v1"
MODEL_NAMES = (
    "visual.peripheral.first.weight", "visual.peripheral.first.bias",
    "visual.peripheral.second.weight", "visual.peripheral.second.bias",
    "visual.foveal.first.weight", "visual.foveal.first.bias",
    "visual.foveal.second.weight", "visual.foveal.second.bias",
    "visual.peripheral_projection.0.weight", "visual.peripheral_projection.0.bias",
    "visual.foveal_projection.0.weight", "visual.foveal_projection.0.bias",
    "body.0.weight", "body.0.bias", "goal_encoder.0.weight", "goal_encoder.0.bias",
    "goal_encoder.2.weight", "goal_encoder.2.bias",
    "observation_projection.0.weight", "observation_projection.0.bias",
    "history.weight_ih_l0", "history.weight_hh_l0",
    "history.bias_ih_l0", "history.bias_hh_l0",
    "policy_trunk.0.weight", "policy_trunk.0.bias",
    "signed_head.weight", "signed_head.bias", "active_head.weight", "active_head.bias",
    "positive_head.weight", "positive_head.bias",
)
MANAGER_NAMES = (
    "query.0.weight", "query.0.bias", "query.2.weight", "query.2.bias", "query_gain",
)
PREDICTOR_ENCODER_NAMES = MODEL_NAMES[:18]
PREDICTOR_MLP_NAMES = (
    "input.mean", "input.scale", "target.mean", "target.scale", "residual.scale",
) + tuple(
    f"member.{member}.{part}"
    for member in range(3)
    for part in (
        "layer0.weight", "layer0.bias", "layer1.weight", "layer1.bias",
        "output.weight", "output.bias",
    )
)
PREDICTOR_NAMES = (
    tuple("encoder." + name for name in PREDICTOR_ENCODER_NAMES)
    + ("observation_normalizer.mean", "observation_normalizer.scale")
    + PREDICTOR_MLP_NAMES
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    clean = copy.deepcopy(metadata)
    clean.pop("artifact_sha256", None)
    digest = hashlib.sha256(canonical(clean))
    for name in sorted(arrays):
        value = arrays[name]
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(canonical(list(value.shape)))
        digest.update(value.tobytes())
    return digest.hexdigest()


def array(value) -> np.ndarray:
    return np.ascontiguousarray(
        value.detach().cpu().numpy() if hasattr(value, "detach") else value,
        dtype=np.float32,
    )


def load_predictor(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if set(archive.files) != {"metadata", *PREDICTOR_NAMES}:
            raise ValueError("predictor tensor set differs")
        arrays = {
            "predictor." + name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in PREDICTOR_NAMES
        }
    receipts = metadata.get("tensors", {})
    for exported, value in arrays.items():
        name = exported.removeprefix("predictor.")
        receipt = receipts.get(name, {})
        if (
            receipt.get("dtype") != value.dtype.str
            or receipt.get("shape") != list(value.shape)
            or receipt.get("sha256") != hashlib.sha256(value.tobytes()).hexdigest()
        ):
            raise ValueError(f"predictor tensor receipt differs: {name}")
    clean = copy.deepcopy(metadata)
    expected_identity = clean.pop("artifact_identity", None)
    array_receipts = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in sorted(
            (name.removeprefix("predictor."), value)
            for name, value in arrays.items()
        )
    }
    actual_identity = hashlib.sha256(
        canonical({"metadata": clean, "arrays": array_receipts})
    ).hexdigest()
    if metadata.get("format") != "chreatures-rich-consequence-ensemble-v1":
        raise ValueError("predictor format differs")
    if expected_identity != actual_identity:
        raise ValueError("predictor artifact identity differs")
    return metadata, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--consequence-laws", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--cold-inherit-v3", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--candidate-adapter-rank", type=int, default=8)
    parser.add_argument("--variation-seed", type=int, default=0)
    parser.add_argument("--variation-scale", type=float, default=0.01)
    args = parser.parse_args()
    if not args.trusted_checkpoint:
        raise SystemExit("Torch deserialization requires --trusted-checkpoint")
    if args.output.exists():
        raise SystemExit("output already exists")
    import torch

    from research.sensorimotor_skills.rich_data import RichNormalizer
    from research.sensorimotor_skills.rich_model import (
        PopulationAdapterBank,
        RichSensorimotorModel,
        cold_inherit_v3_model,
    )
    from research.sensorimotor_skills.rich_online import (
        SlowGoalManager,
        cold_inherit_v3_manager,
    )

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_format = checkpoint.get("format")
    if args.cold_inherit_v3 != (source_format == V3_CHECKPOINT_FORMAT):
        raise ValueError("checkpoint format and cold-inheritance selection differ")
    if source_format not in {CHECKPOINT_FORMAT, V3_CHECKPOINT_FORMAT}:
        raise ValueError("development checkpoint format differs")
    identity = copy.deepcopy(checkpoint["identity"])
    if args.cold_inherit_v3:
        model = cold_inherit_v3_model(checkpoint["model"])
        manager = cold_inherit_v3_manager(checkpoint["goal_manager"])
        normalizer = RichNormalizer.cold_inherit_v3(identity["normalizer"])
        adapters = PopulationAdapterBank(
            args.candidate_count, args.candidate_adapter_rank
        )
        if args.candidate_count > 1:
            adapters.vary(
                torch.arange(1, args.candidate_count),
                seed=args.variation_seed,
                scale=args.variation_scale,
            )
        adapter_state = adapters.state_dict()
        conversion = {
            "format": "chreatures-v3-to-v4-cold-inheritance-v1",
            "source_checkpoint_sha256": sha256(checkpoint_path),
            "source_format": source_format,
            "new_physiology_normalization": "mean-zero-scale-one",
            "new_active_bias": -8.0,
            "old_to_new_action_columns": [0, 1, 2, 4, 5, 6, 7, 3, 8],
            "optimizer": "not inherited",
            "private_state": "not inherited",
        }
    else:
        model = checkpoint["model"]
        manager = checkpoint["goal_manager"]
        normalizer = RichNormalizer.from_value(identity["normalizer"])
        adapter_state = checkpoint["candidate_adapters"]
        conversion = None
    expected_model = RichSensorimotorModel().state_dict()
    for name in MODEL_NAMES:
        if name not in model or model[name].shape != expected_model[name].shape:
            raise ValueError(f"development model tensor differs: {name}")
    expected_manager = SlowGoalManager().state_dict()
    for name in MANAGER_NAMES:
        if name not in manager or manager[name].shape != expected_manager[name].shape:
            raise ValueError(f"development manager tensor differs: {name}")
    down, up, bias = (adapter_state[name] for name in ("down", "up", "bias"))
    if (
        down.ndim != 3
        or down.shape[2] != 256
        or up.shape != (down.shape[0], 256, down.shape[1])
        or bias.shape != (down.shape[0], 256)
        or not 1 <= down.shape[0] <= 4096
        or not 1 <= down.shape[1] <= 32
    ):
        raise ValueError("population adapter tensor shapes differ")
    arrays = {
        "normalizer.mean": array(normalizer.mean),
        "normalizer.scale": array(normalizer.scale),
        **{"model." + name: array(model[name]) for name in MODEL_NAMES},
        **{
            "manager." + name: array(manager[name]).reshape(1)
            if name == "query_gain"
            else array(manager[name])
            for name in MANAGER_NAMES
        },
        "population_adapter.down": array(adapter_state["down"]),
        "population_adapter.up": array(adapter_state["up"]),
        "population_adapter.bias": array(adapter_state["bias"]),
    }
    predictor_path = args.predictor.resolve()
    predictor_metadata, predictor_arrays = load_predictor(predictor_path)
    embedded_predictor_metadata = copy.deepcopy(predictor_metadata)
    embedded_predictor_metadata["source_pack_order"] = copy.deepcopy(
        embedded_predictor_metadata.get("pack_order")
    )
    embedded_predictor_metadata["pack_order"] = list(predictor_arrays)
    for name in PREDICTOR_ENCODER_NAMES:
        inherited = predictor_arrays["predictor.encoder." + name]
        shared = arrays["model." + name]
        if name == "body.0.weight":
            shared = shared[:, :357]
        if not np.array_equal(inherited, shared):
            raise ValueError(f"frozen H1/shared inherited encoder differs: {name}")
    if not np.array_equal(
        predictor_arrays["predictor.observation_normalizer.mean"],
        arrays["normalizer.mean"][:4453],
    ) or not np.array_equal(
        predictor_arrays["predictor.observation_normalizer.scale"],
        arrays["normalizer.scale"][:4453],
    ):
        raise ValueError("frozen H1/shared inherited normalizer differs")
    arrays.update(predictor_arrays)
    for name, value in arrays.items():
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValueError(f"export tensor is not finite contiguous float32: {name}")
    laws_path = args.consequence_laws.resolve()
    law_bytes = laws_path.read_bytes()
    laws = json.loads(law_bytes)
    if laws.get("schema") != "chreatures-gam-consequence-law-bank-v1":
        raise ValueError("consequence law bank format differs")
    tensors = {
        name: {
            "dtype": "float32",
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in arrays.items()
    }
    adapter_identity = hashlib.sha256(
        canonical(
            {
                name: tensors[name]
                for name in arrays
                if name.startswith("population_")
            }
        )
    ).hexdigest()
    metadata = {
        "format": FORMAT,
        "version": 4,
        "execution": EXECUTION,
        "organism_interface": organism_identity(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "format": source_format,
            "updates": int(checkpoint.get("updates", 0)),
            "physical_steps": int(checkpoint.get("physical_steps", 0)),
        },
        "cold_inheritance": conversion,
        "training_identity": identity,
        "population_adapters": {
            "count": int(arrays["population_adapter.down"].shape[0]),
            "rank": int(arrays["population_adapter.down"].shape[1]),
            "identity": adapter_identity,
        },
        "runtime_contract": {
            "temporal": {
                "observation_interval_seconds": 0.05,
                "manager_commit_ticks": 10,
            },
            "consequence_refinement": {
                "candidates": 4,
                "tilt": 0.5,
                "learning_rate": 0.05,
                "error_decay": 0.99,
                "innovation_limit": 4.0,
            },
            "personal_goal_associations": copy.deepcopy(PERSONAL_GOAL_CONTRACT),
        },
        "inherited_h1_predictor": {
            "file_sha256": sha256(predictor_path),
            "artifact_identity": predictor_metadata["artifact_identity"],
            "goal_forecast_rms": max(
                float(
                    predictor_metadata["validation"][
                        "runtime_empirical_goal_error_scale"
                    ]
                ),
                1e-4,
            ),
            "pack_order": list(predictor_arrays),
            "metadata": embedded_predictor_metadata,
            "projection": {
                "observation": "rich4096+canonical351+physiology[0:6]",
                "previous_v4_indices_to_v3": [0, 1, 2, 4, 5, 6, 7, 3, 8],
                "unsupported_actions": ["release", "secrete", "allocate"],
                "unsupported_physiology": list(organism_identity()["physiology"][6:]),
            },
        },
        "consequence_laws": {
            "value": laws,
            "file_sha256": hashlib.sha256(law_bytes).hexdigest(),
            "content_sha256": hashlib.sha256(canonical(laws)).hexdigest(),
        },
        "pack_order": list(arrays),
        "tensors": tensors,
    }
    metadata["artifact_sha256"] = artifact_identity(metadata, arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, metadata=np.asarray(canonical(metadata).decode()), **arrays
        )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "bytes": args.output.stat().st_size,
                "file_sha256": sha256(args.output),
                "artifact_sha256": metadata["artifact_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
