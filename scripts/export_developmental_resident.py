#!/usr/bin/env python3
"""Export one trusted current Torch population controller to immutable native NPZ."""

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
from chreatures.resident_contract import (
    DEVELOPMENT_FORMAT,
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
)

FORMAT = NATIVE_POPULATION_FORMAT
EXECUTION = NATIVE_EXECUTION
CHECKPOINT_FORMAT = DEVELOPMENT_FORMAT
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
    "physiology_adapter.weight",
    "history.weight_ih_l0", "history.weight_hh_l0",
    "history.bias_ih_l0", "history.bias_hh_l0",
    "policy_trunk.0.weight", "policy_trunk.0.bias",
    "signed_head.weight", "signed_head.bias", "active_head.weight", "active_head.bias",
    "positive_head.weight", "positive_head.bias",
    "new_actuator_active.weight", "new_actuator_active.bias",
    "new_actuator_positive.weight", "new_actuator_positive.bias",
)
PHYSIOLOGY_ADAPTER_NAMES = ("physiology_adapter.weight",)
NEW_ACTUATOR_NAMES = (
    "new_actuator_active.weight", "new_actuator_active.bias",
    "new_actuator_positive.weight", "new_actuator_positive.bias",
)
ESTABLISHED_MODEL_NAMES = tuple(
    name
    for name in MODEL_NAMES
    if name not in {*PHYSIOLOGY_ADAPTER_NAMES, *NEW_ACTUATOR_NAMES}
)
MANAGER_NAMES = (
    "query.0.weight", "query.0.bias", "query.2.weight", "query.2.bias", "query_gain",
)
PREDICTOR_NAMES = (
    "context.mean",
    "context.scale",
    "action.mean",
    "action.scale",
    "target.mean",
    "target.scale",
) + tuple(
    f"member.{member}.{part}"
    for member in range(3)
    for part in (
        "context.weight",
        "context.bias",
        "transition.weight_ih",
        "transition.weight_hh",
        "transition.bias_ih",
        "transition.bias_hh",
        "output.weight",
        "output.bias",
    )
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
    if metadata.get("format") != "chreatures-rich-recurrent-consequence-ensemble-v3":
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
    args = parser.parse_args()
    if not args.trusted_checkpoint:
        raise SystemExit("Torch deserialization requires --trusted-checkpoint")
    if args.output.exists():
        raise SystemExit("output already exists")
    import torch

    from research.sensorimotor_skills.rich_data import RichNormalizer
    from research.sensorimotor_skills.rich_model import (
        RichSensorimotorModel,
    )
    from research.sensorimotor_skills.rich_online import (
        SlowGoalManager,
    )

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_format = checkpoint.get("format")
    if source_format != CHECKPOINT_FORMAT:
        raise ValueError("development checkpoint format differs")
    identity = copy.deepcopy(checkpoint["identity"])
    model = checkpoint["model"]
    manager = checkpoint["goal_manager"]
    normalizer = RichNormalizer.from_value(identity["normalizer"])
    adapter_state = checkpoint["candidate_adapters"]
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
        **{
            "model." + name: array(model[name])
            for name in ESTABLISHED_MODEL_NAMES
        },
        **{
            "manager." + name: array(manager[name]).reshape(1)
            if name == "query_gain"
            else array(manager[name])
            for name in MANAGER_NAMES
        },
        **{
            "model." + name: array(model[name])
            for name in PHYSIOLOGY_ADAPTER_NAMES
        },
        "population_adapter.down": array(adapter_state["down"]),
        "population_adapter.up": array(adapter_state["up"]),
        "population_adapter.bias": array(adapter_state["bias"]),
        **{
            "model." + name: array(model[name]) for name in NEW_ACTUATOR_NAMES
        },
    }
    predictor_path = args.predictor.resolve()
    predictor_metadata, predictor_arrays = load_predictor(predictor_path)
    embedded_predictor_metadata = copy.deepcopy(predictor_metadata)
    embedded_predictor_metadata["source_pack_order"] = copy.deepcopy(
        embedded_predictor_metadata.get("pack_order")
    )
    embedded_predictor_metadata["pack_order"] = list(predictor_arrays)
    representation = predictor_metadata.get("representation", {})
    if representation.get("file_sha256") != sha256(checkpoint_path):
        raise ValueError("predictor/resident recurrent representation differs")
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
        "version": NATIVE_POPULATION_VERSION,
        "execution": EXECUTION,
        "organism_interface": organism_identity(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "format": source_format,
            "updates": int(checkpoint.get("updates", 0)),
            "physical_steps": int(checkpoint.get("physical_steps", 0)),
        },
        "training_identity": identity,
        "population_adapters": {
            "count": int(arrays["population_adapter.down"].shape[0]),
            "rank": int(arrays["population_adapter.down"].shape[1]),
            "identity": adapter_identity,
        },
        "shared_trainable_organs": {
            "scope": "inherited base artifact; not private lifetime state",
            "physiology_adapter": {
                "input": "normalized observation physiology12",
                "output": "worker pre-GRU projection128",
            },
            "new_actuator_projection": {
                "input": "candidate-adapted policy hidden256",
                "output": "canonical action axes8:12 active and positive logits",
                "action_names": ["eat", "release", "secrete", "allocate"],
            },
            "achieved_goal_geometry": "frozen",
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
        "recurrent_predictor": {
            "file_sha256": sha256(predictor_path),
            "artifact_identity": predictor_metadata["artifact_identity"],
            "pack_order": list(predictor_arrays),
            "metadata": embedded_predictor_metadata,
            "context": "codes4x256+private_effective_worker_context128+neural384+raw_physiology12+previous_delivered_action12",
            "candidate_actions": "[B,K,H,12], H=1..8, delivered canonical actions",
            "outputs": "per-tick frame-code delta256 + raw physiology delta12",
            "runtime_scoring": {
                "horizon_ticks": 4,
                "horizon_seconds": 0.2,
                "proposal_suffix": "hold the proposed delivered action constant for four ticks",
                "goal_error_rms": predictor_metadata["validation"][
                    "goal_calibration"
                ]["empirical_goal_rms_by_horizon"][3],
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
