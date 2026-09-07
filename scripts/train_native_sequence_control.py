#!/usr/bin/env python3
"""Initialize or update learned native sequence-control heads from native rollouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.sequence_control import (
    CONTRACT_SHA256,
    inherited_dependencies,
    load_control_artifact,
    valid_sha256,
    write_control_artifact,
)
from research.sensorimotor_skills.sequence_control import (
    ACTIVE_ROLLOUT_CONTRACT,
    CANCELLATION_REASONS,
    CONTRACT_VERSION,
    PARAMETER_ORDER,
    SequenceControlHeads,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    generalized_advantage_estimate,
    hierarchical_terms,
    initial_parameter_arrays,
    load_parameter_arrays,
    load_rollout,
    parameter_arrays,
    ppo_loss,
    replay_error,
    tensor_rollout,
)


OPTIMIZER_FORMAT = "chreatures-sequence-control-adamw-state-v1"
RESULT_FORMAT = "chreatures-sequence-control-training-result-v1"
SWAP_FORMAT = "chreatures-sequence-control-swap-request-v1"


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def current_revision() -> str:
    override = os.environ.get("CHREATURES_SOURCE_REVISION")
    if override is not None:
        if len(override) != 40 or any(char not in "0123456789abcdef" for char in override):
            raise ValueError("CHREATURES_SOURCE_REVISION must be a full lowercase Git revision")
        return override
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha(value: Any, label: str) -> str:
    if not valid_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def initialize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resident-artifact", type=Path, required=True)
    parser.add_argument("--expected-resident-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)


def update_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent-policy", type=Path, required=True)
    parser.add_argument("--expected-parent-file-sha256", required=True)
    parser.add_argument("--expected-parent-artifact-sha256", required=True)
    parser.add_argument("--parent-optimizer", type=Path)
    parser.add_argument("--expected-parent-optimizer-file-sha256")
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--expected-rollout-content-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--hazard-entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--selector-entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--likelihood-tolerance", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--device", default="cpu")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize_args(commands.add_parser("initialize"))
    update_args(commands.add_parser("update"))
    return parser.parse_args()


def load_resident_metadata(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.resolve()
    if file_sha256(path) != sha(expected_sha256, "resident file identity"):
        raise ValueError("resident artifact file SHA-256 differs")
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("resident artifact metadata is missing")
        metadata = json.loads(str(archive["metadata"].item()))
    return metadata


def initialize(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit("output policy already exists")
    metadata = load_resident_metadata(
        args.resident_artifact, args.expected_resident_file_sha256
    )
    dependencies = inherited_dependencies(metadata)
    arrays = initial_parameter_arrays(args.seed)
    artifact = write_control_artifact(
        args.output,
        arrays,
        version=0,
        parent_sha256=None,
        dependencies=dependencies,
        provenance={
            "training_status": "initialized-untrained",
            "operation": "seeded-xavier-initialization",
            "seed": args.seed,
            "output_heads": {
                "selector": "zero-weight uniform over available proposals",
                "hazard": "zero-weight bias -ln(7)",
                "value": "zero",
            },
            "source_revision": current_revision(),
            "script_sha256": file_sha256(Path(__file__).resolve()),
        },
    )
    print(
        json.dumps(
            {
                "path": str(artifact.path),
                "file_sha256": artifact.file_sha256,
                "artifact_sha256": artifact.sha256,
                "version": artifact.version,
                "contract_sha256": CONTRACT_SHA256,
            },
            sort_keys=True,
        )
    )


def optimizer_values(
    optimizer: torch.optim.Optimizer, model: SequenceControlHeads
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not state:
            raise RuntimeError(f"optimizer state missing after update: {name}")
        result[f"exp_avg.{name}"] = np.ascontiguousarray(
            state["exp_avg"].detach().cpu().numpy(), dtype=np.float32
        )
        result[f"exp_avg_sq.{name}"] = np.ascontiguousarray(
            state["exp_avg_sq"].detach().cpu().numpy(), dtype=np.float32
        )
        step = state["step"]
        result[f"step.{name}"] = np.asarray(
            [int(step.item() if torch.is_tensor(step) else step)], dtype=np.int64
        )
    return result


def optimizer_identity(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    clean = copy.deepcopy(dict(metadata))
    clean.pop("artifact_sha256", None)
    receipts = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }
    return canonical_sha256({"metadata": clean, "arrays": receipts})


def save_optimizer(
    path: Path,
    optimizer: torch.optim.Optimizer,
    model: SequenceControlHeads,
    *,
    policy_sha256: str,
    policy_version: int,
    parent_policy_sha256: str,
    parent_optimizer_file_sha256: str | None,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    arrays = optimizer_values(optimizer, model)
    metadata = {
        "format": OPTIMIZER_FORMAT,
        "policy_sha256": policy_sha256,
        "policy_version": policy_version,
        "parent_policy_sha256": parent_policy_sha256,
        "parent_optimizer_file_sha256": parent_optimizer_file_sha256,
        "parameter_order": list(PARAMETER_ORDER),
        "configuration": copy.deepcopy(dict(configuration)),
    }
    metadata["artifact_sha256"] = optimizer_identity(metadata, arrays)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(canonical_bytes(metadata).decode()),
            **arrays,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "artifact_sha256": metadata["artifact_sha256"],
    }


def restore_optimizer(
    path: Path,
    expected_file_sha256: str,
    expected_policy_sha256: str,
    optimizer: torch.optim.Optimizer,
    model: SequenceControlHeads,
) -> dict[str, Any]:
    path = path.resolve()
    observed_file = file_sha256(path)
    if observed_file != sha(expected_file_sha256, "optimizer file identity"):
        raise ValueError("parent optimizer file SHA-256 differs")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    if (
        metadata.get("format") != OPTIMIZER_FORMAT
        or metadata.get("policy_sha256") != expected_policy_sha256
        or tuple(metadata.get("parameter_order", ())) != PARAMETER_ORDER
        or metadata.get("artifact_sha256") != optimizer_identity(metadata, arrays)
    ):
        raise ValueError("parent optimizer identity differs")
    expected_names = {
        f"{field}.{name}"
        for name in PARAMETER_ORDER
        for field in ("exp_avg", "exp_avg_sq", "step")
    }
    if set(arrays) != expected_names:
        raise ValueError("parent optimizer tensor set differs")
    for name, parameter in model.named_parameters():
        exp_avg = arrays[f"exp_avg.{name}"]
        exp_avg_sq = arrays[f"exp_avg_sq.{name}"]
        step = arrays[f"step.{name}"]
        if (
            exp_avg.dtype != np.float32
            or exp_avg.shape != tuple(parameter.shape)
            or exp_avg_sq.dtype != np.float32
            or exp_avg_sq.shape != tuple(parameter.shape)
            or step.dtype != np.int64
            or step.shape != (1,)
            or int(step[0]) < 1
            or not np.all(np.isfinite(exp_avg))
            or not np.all(np.isfinite(exp_avg_sq))
        ):
            raise ValueError(f"parent optimizer tensor differs: {name}")
        optimizer.state[parameter] = {
            "step": torch.tensor(float(step[0]), device=parameter.device),
            "exp_avg": torch.as_tensor(exp_avg, device=parameter.device).clone(),
            "exp_avg_sq": torch.as_tensor(exp_avg_sq, device=parameter.device).clone(),
        }
    return {**metadata, "file_sha256": observed_file}


def next_values(
    model: SequenceControlHeads,
    tensors: Mapping[str, torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    shape = tensors["next_value"].shape
    flat = {name: value.flatten(0, 1) for name, value in tensors.items()}
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, flat["next_state"].shape[0], batch_size):
            rows = slice(start, start + batch_size)
            outputs = model(
                flat["next_state"][rows],
                flat["next_proposal"][rows],
                flat["next_active"][rows],
                flat["next_proposal_mask"][rows],
                flat["next_active_mask"][rows],
            )
            values.append(outputs["value"])
    return torch.cat(values).reshape(shape)


def train_update(
    model: SequenceControlHeads,
    optimizer: torch.optim.Optimizer,
    tensors: Mapping[str, torch.Tensor],
    advantage: torch.Tensor,
    returns: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    names = (
        "state",
        "proposal",
        "active",
        "proposal_mask",
        "active_mask",
        "hazard_decision",
        "selected_candidate",
        "hazard_mask",
        "selector_mask",
        "behavior_logp",
        "actor_valid",
    )
    flat = {name: tensors[name].flatten(0, 1) for name in names}
    flat["advantage"] = advantage.flatten()
    flat["returns"] = returns.flatten()
    count = flat["state"].shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    traces = []
    model.train()
    for epoch in range(args.epochs):
        totals: dict[str, float] = {}
        batches = 0
        for indices_cpu in torch.randperm(count, generator=generator).split(args.batch_size):
            indices = indices_cpu.to(flat["state"].device)
            outputs = model(
                flat["state"][indices],
                flat["proposal"][indices],
                flat["active"][indices],
                flat["proposal_mask"][indices],
                flat["active_mask"][indices],
            )
            terms = hierarchical_terms(
                outputs,
                flat["proposal_mask"][indices],
                flat["active_mask"][indices],
                flat["hazard_decision"][indices],
                flat["selected_candidate"][indices],
                flat["hazard_mask"][indices],
                flat["selector_mask"][indices],
            )
            loss, metrics = ppo_loss(
                outputs,
                terms,
                flat["behavior_logp"][indices],
                flat["advantage"][indices],
                flat["returns"][indices],
                flat["actor_valid"][indices],
                flat["hazard_mask"][indices],
                flat["selector_mask"][indices],
                clip_ratio=args.clip_ratio,
                value_coefficient=args.value_coefficient,
                hazard_entropy_coefficient=args.hazard_entropy_coefficient,
                selector_entropy_coefficient=args.selector_entropy_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            values = {
                name: float(value.detach().cpu()) for name, value in metrics.items()
            }
            values["gradient_norm"] = float(gradient_norm.detach().cpu())
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1
        traces.append(
            {"epoch": epoch + 1}
            | {name: value / batches for name, value in totals.items()}
        )
    return traces


def validate_schedule(args: argparse.Namespace) -> None:
    if (
        not 1 <= args.epochs <= 32
        or not 32 <= args.batch_size <= 65536
        or not 0.0 < args.learning_rate <= 0.01
        or not 0.0 <= args.weight_decay <= 0.1
        or not 0.0 <= args.clip_ratio <= 1.0
        or not 0.0 <= args.value_coefficient <= 10.0
        or not 0.0 <= args.hazard_entropy_coefficient <= 1.0
        or not 0.0 <= args.selector_entropy_coefficient <= 1.0
        or not 0.0 < args.max_grad_norm <= 100.0
        or not 0.0 < args.likelihood_tolerance <= 1e-3
    ):
        raise SystemExit("invalid PPO optimization schedule")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    if (args.parent_optimizer is None) != (
        args.expected_parent_optimizer_file_sha256 is None
    ):
        raise SystemExit("parent optimizer and its expected file SHA are paired")


def update(args: argparse.Namespace) -> None:
    validate_schedule(args)
    expected_parent_file = sha(
        args.expected_parent_file_sha256, "parent policy file identity"
    )
    expected_parent_artifact = sha(
        args.expected_parent_artifact_sha256, "parent policy artifact identity"
    )
    parent = load_control_artifact(args.parent_policy)
    if (
        parent.file_sha256 != expected_parent_file
        or parent.sha256 != expected_parent_artifact
    ):
        raise ValueError("parent policy identity differs")
    if parent.version > 0 and args.parent_optimizer is None:
        raise ValueError("noninitial policy requires its archived optimizer state")
    rollout = load_rollout(args.rollout)
    if rollout.manifest.get("content_sha256") != sha(
        args.expected_rollout_content_sha256, "rollout content identity"
    ):
        raise ValueError("rollout content identity differs")
    rollout_policy = rollout.manifest.get("policy", {})
    if (
        rollout_policy.get("artifact_sha256") != parent.sha256
        or rollout_policy.get("file_sha256") != parent.file_sha256
        or rollout_policy.get("version") != parent.version
        or rollout.manifest.get("dependencies") != parent.metadata["dependencies"]
    ):
        raise ValueError("rollout policy or inherited dependencies differ")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA/ROCm device is unavailable")
    torch.manual_seed(args.seed)
    model = SequenceControlHeads().to(device)
    load_parameter_arrays(model, parent.arrays)
    tensors = tensor_rollout(rollout, device)
    replay = replay_error(model, tensors, batch_size=args.batch_size)
    if max(replay.values()) > args.likelihood_tolerance:
        raise ValueError(
            f"native/Torch decision replay differs: {replay}"
        )
    if not torch.any(tensors["actor_valid"]):
        raise ValueError("rollout has no actor-valid acknowledged decisions")
    old_next_value = next_values(model, tensors, args.batch_size)
    reward_contract = rollout.manifest["reward"]
    advantage, returns = generalized_advantage_estimate(
        tensors["reward"],
        tensors["value"],
        old_next_value,
        tensors["terminal"],
        tensors["truncated"],
        float(reward_contract["discount"]),
        float(reward_contract["gae_lambda"]),
    )
    actor_advantage = advantage[tensors["actor_valid"]]
    advantage = (
        advantage - actor_advantage.mean()
    ) / actor_advantage.std(unbiased=False).clamp_min(1e-6)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    parent_optimizer = None
    if args.parent_optimizer is not None:
        parent_optimizer = restore_optimizer(
            args.parent_optimizer,
            args.expected_parent_optimizer_file_sha256,
            parent.sha256,
            optimizer,
            model,
        )
        prior_config = parent_optimizer.get("configuration")
        if prior_config != {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        }:
            raise ValueError("optimizer configuration changed across policy versions")

    started = time.monotonic()
    traces = train_update(model, optimizer, tensors, advantage, returns, args)
    elapsed = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    policy_path = args.output / "sequence-control.npz"
    optimizer_path = args.output / "optimizer-state.npz"
    configuration = {
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    child = write_control_artifact(
        policy_path,
        parameter_arrays(model),
        version=parent.version + 1,
        parent_sha256=parent.sha256,
        dependencies=parent.metadata["dependencies"],
        provenance={
            "training_status": "trained",
            "operation": "native-on-policy-clipped-ppo",
            "source_revision": current_revision(),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "parent_policy_file_sha256": parent.file_sha256,
            "parent_optimizer_file_sha256": (
                None if parent_optimizer is None else parent_optimizer["file_sha256"]
            ),
            "rollout_manifest_file_sha256": rollout.manifest_file_sha256,
            "rollout_content_sha256": rollout.manifest["content_sha256"],
            "before_checkpoint_sha256": rollout.manifest["boundary"][
                "before_checkpoint_sha256"
            ],
            "after_checkpoint_sha256": rollout.manifest["boundary"][
                "after_checkpoint_sha256"
            ],
            "reward": reward_contract,
            "ppo": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "clip_ratio": args.clip_ratio,
                "value_coefficient": args.value_coefficient,
                "hazard_entropy_coefficient": args.hazard_entropy_coefficient,
                "selector_entropy_coefficient": args.selector_entropy_coefficient,
                "max_grad_norm": args.max_grad_norm,
                "seed": args.seed,
            },
        },
    )
    optimizer_receipt = save_optimizer(
        optimizer_path,
        optimizer,
        model,
        policy_sha256=child.sha256,
        policy_version=child.version,
        parent_policy_sha256=parent.sha256,
        parent_optimizer_file_sha256=(
            None if parent_optimizer is None else parent_optimizer["file_sha256"]
        ),
        configuration=configuration,
    )
    counts = {
        "transitions": int(tensors["reward"].numel()),
        "actor_valid": int(tensors["actor_valid"].sum().cpu()),
        "hazard": int((tensors["actor_valid"] & tensors["hazard_mask"]).sum().cpu()),
        "selector": int((tensors["actor_valid"] & tensors["selector_mask"]).sum().cpu()),
        "terminal": int(tensors["terminal"].sum().cpu()),
        "truncated": int(tensors["truncated"].sum().cpu()),
        "cancellation_reasons": {
            name: int((tensors["cancellation_reason"] == index).sum().cpu())
            for index, name in enumerate(CANCELLATION_REASONS)
        },
    }
    swap = {
        "format": SWAP_FORMAT,
        "contract_sha256": CONTRACT_SHA256,
        "old_policy": {
            "path": str(Path(args.parent_policy).resolve()),
            "file_sha256": parent.file_sha256,
            "artifact_sha256": parent.sha256,
            "version": parent.version,
        },
        "new_policy": {
            "path": str(child.path),
            "file_sha256": child.file_sha256,
            "artifact_sha256": child.sha256,
            "version": child.version,
        },
        "optimizer": optimizer_receipt,
        "coherent_rollout_checkpoint_sha256": rollout.manifest["boundary"][
            "after_checkpoint_sha256"
        ],
        "required_transaction": "stage every paused research-training participant, commit one version, acknowledge exact version/hash, checkpoint, then resume",
        "next_rollout_index": int(rollout.manifest.get("rollout_index", 0)) + 1,
    }
    swap["content_sha256"] = canonical_sha256(swap)
    swap_path = args.output / "swap-request.json"
    atomic_json(swap_path, swap)
    result = {
        "format": RESULT_FORMAT,
        "policy": swap["new_policy"],
        "optimizer": optimizer_receipt,
        "rollout": {
            "path": str(Path(args.rollout).resolve()),
            "manifest_file_sha256": rollout.manifest_file_sha256,
            "content_sha256": rollout.manifest["content_sha256"],
        },
        "native_torch_replay_max_abs": replay,
        "counts": counts,
        "advantage": {
            "mean": float(actor_advantage.mean().cpu()),
            "std": float(actor_advantage.std(unbiased=False).cpu()),
            "return_mean": float(returns.mean().cpu()),
        },
        "training_seconds": elapsed,
        "epochs": traces,
        "swap_request": {
            "path": str(swap_path.resolve()),
            "content_sha256": swap["content_sha256"],
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    args = arguments()
    if ACTIVE_ROLLOUT_CONTRACT is None:
        raise SystemExit(
            "sequence-control training is disabled pending the CNS-derived input contract"
        )
    if args.command == "initialize":
        initialize(args)
    else:
        update(args)


if __name__ == "__main__":
    main()
