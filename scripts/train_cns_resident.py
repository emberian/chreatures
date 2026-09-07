#!/usr/bin/env python3
"""Collect, train, export, and compare the current CNS-only native resident."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.resident_learning.artifact import file_sha256, load_parent, publish_trained
from research.resident_learning.data import load_episode, write_episode

RESULT_FORMAT = "chreatures-cns-resident-training-result-v1"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def convert_recording(args: argparse.Namespace) -> None:
    source, output = args.recording.resolve(), args.output.resolve()
    intent = json.loads((source / "intent.json").read_text())
    latent = np.asarray(np.load(source / "cns_latents.npy", mmap_mode="r"), dtype=np.float32)
    commands = np.asarray(np.load(source / "commands.npy", mmap_mode="r"), dtype=np.float32)
    positions = np.asarray(np.load(source / "positions.npy", mmap_mode="r"), dtype=np.float64)
    times = np.asarray(np.load(source / "time_seconds.npy", mmap_mode="r"), dtype=np.float64)
    if latent.shape[:2] != commands.shape[:2] or positions.shape != (*commands.shape[:2], 3) or times.shape != (commands.shape[0],):
        raise ValueError("joined recording arrays differ")
    if latent.shape[-1] != 512 or commands.shape[-1] != 12 or latent.shape[0] < 9:
        raise ValueError("joined recording is not a CNS Z512 action12 history")
    # A transition action at frame t leads to the physical/CNS observation at t+1.
    delivered = commands[:-1]
    delta = positions[1:, :, :2] - positions[:-1, :, :2]
    dt = np.maximum(times[1:] - times[:-1], 1e-6)[:, None]
    speed = np.linalg.norm(delta, axis=-1) / dt
    effort = np.mean(np.square(delivered), axis=-1)
    reward = np.asarray(speed - args.effort_cost * effort, dtype=np.float32)
    previous = np.empty_like(commands)
    previous[0] = 0.0
    previous[1:] = delivered
    reset = np.zeros(commands.shape[:2], dtype=np.bool_); reset[0] = True
    terminal = np.zeros(delivered.shape[:2], dtype=np.bool_); terminal[-1] = True
    episode = write_episode(
        output,
        {
            "cns_latent": latent,
            "previous_delivered_command": previous,
            "reset": reset,
            "delivered_command": delivered,
            "physical_reward": reward,
            "terminal": terminal,
        },
        cns_service={"file_sha256": intent["service_sha256"], "latent": 512},
        resident_artifact={"file_sha256": intent["controller_sha256"]},
        provenance={
            "source_format": intent["format"], "source_directory": str(source),
            "source_intent_sha256": _sha(source / "intent.json"),
            "condition": intent.get("condition"),
            "reward": {"formula": "horizontal_speed-effort_cost*mean(command^2)", "effort_cost": args.effort_cost},
            "geometry_retained": False,
        },
    )
    print(json.dumps({"path": str(episode.path), "file_sha256": episode.file_sha256, "episode_sha256": episode.metadata["episode_sha256"], "transitions": episode.transitions, "residents": episode.residents}, sort_keys=True))


def inspect(args: argparse.Namespace) -> None:
    episode = load_episode(args.episode)
    reward = episode.arrays["physical_reward"]
    command = episode.arrays["delivered_command"]
    print(json.dumps({
        "path": str(episode.path), "file_sha256": episode.file_sha256,
        "episode_sha256": episode.metadata["episode_sha256"],
        "transitions": episode.transitions, "residents": episode.residents,
        "controller_input_fields": episode.metadata["controller_input_fields"],
        "teacher_only_fields": episode.metadata["teacher_only_fields"],
        "reward_mean": float(reward.mean()), "reward_positive_fraction": float((reward > 0).mean()),
        "command_rms": float(np.sqrt(np.mean(np.square(command)))),
    }, sort_keys=True))


def pack_joined(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    parent_metadata, _ = load_parent(args.parent)
    parent_hash = file_sha256(args.parent.resolve())
    with np.load(source, allow_pickle=False) as archive:
        required = {"metadata", "cns_latent", "delivered_command", "physical_reward", "reset", "terminal"}
        if set(archive.files) != required:
            raise ValueError("joined learning source tensor set differs")
        metadata = json.loads(str(archive["metadata"].item()))
        latent = np.ascontiguousarray(archive["cns_latent"], dtype=np.float32)
        delivered = np.ascontiguousarray(archive["delivered_command"], dtype=np.float32)
        reward = np.ascontiguousarray(archive["physical_reward"], dtype=np.float32)
        reset = np.ascontiguousarray(archive["reset"], dtype=np.bool_)
        terminal = np.ascontiguousarray(archive["terminal"], dtype=np.bool_)
    if metadata.get("action_source") not in {"privileged-physical-teacher", "resident-policy"}:
        raise ValueError("joined history must identify teacher or resident action source")
    if metadata.get("cns_service_artifact_sha256") != parent_metadata["cns_service"]["service_artifact_sha256"]:
        raise ValueError("joined history CNS service differs from the resident binding")
    if latent.ndim != 3 or delivered.ndim != 3 or latent.shape[0] != delivered.shape[0] + 1:
        raise ValueError("joined history must align T+1 CNS observations with T deliveries")
    previous = np.zeros((latent.shape[0], latent.shape[1], 12), np.float32)
    previous[1:] = delivered
    episode = write_episode(
        args.output,
        {"cns_latent": latent, "previous_delivered_command": previous, "reset": reset,
         "delivered_command": delivered, "physical_reward": reward, "terminal": terminal},
        cns_service=parent_metadata["cns_service"],
        resident_artifact={"file_sha256": parent_hash, "artifact_sha256": parent_metadata["artifact_sha256"]},
        provenance={"source_joined_file_sha256": file_sha256(source), **metadata},
    )
    print(json.dumps({"path": str(episode.path), "file_sha256": episode.file_sha256, "episode_sha256": episode.metadata["episode_sha256"], "action_source": metadata["action_source"], "transitions": episode.transitions, "residents": episode.residents}, sort_keys=True))


def _tensor_window(episode, start: int, length: int, residents: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    import torch
    arrays = episode.arrays
    result = {
        "cns_latent": arrays["cns_latent"][start : start + length + 1, residents],
        "previous_delivered_command": arrays["previous_delivered_command"][start : start + length + 1, residents],
        "reset": arrays["reset"][start : start + length + 1, residents],
        "delivered_command": arrays["delivered_command"][start : start + length, residents],
        "physical_reward": arrays["physical_reward"][start : start + length, residents],
        "terminal": arrays["terminal"][start : start + length, residents],
    }
    # A sampled window is a fresh truncated recurrent training sequence.
    result["reset"] = result["reset"].copy(); result["reset"][0] = True
    return {name: torch.as_tensor(value, device=device) for name, value in result.items()}


def train(args: argparse.Namespace) -> None:
    import torch
    from research.resident_learning.model import CnsResidentModel, training_loss
    began = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested ROCm/CUDA device is unavailable")
    parent_metadata, parent_arrays = load_parent(args.parent)
    parent_file = file_sha256(args.parent.resolve())
    episodes = [load_episode(path) for path in args.episode]
    validation = [load_episode(path) for path in args.validation_episode]
    for episode in [*episodes, *validation]:
        declared = episode.metadata["resident_artifact"].get("file_sha256")
        service = episode.metadata["cns_service"]
        declared_service = service.get("service_artifact_sha256", service.get("file_sha256"))
        if declared != parent_file:
            raise ValueError("episode was not collected by the exact parent resident")
        if declared_service != parent_metadata["cns_service"]["service_artifact_sha256"]:
            raise ValueError("episode CNS service differs from the parent resident binding")
        declared_adapter = service.get("adapter_sha256")
        if declared_adapter is not None and declared_adapter != parent_metadata["cns_service"]["adapter_sha256"]:
            raise ValueError("episode CNS adapter differs from the parent resident binding")
    model = CnsResidentModel.from_arrays(parent_arrays, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = np.random.default_rng(args.seed)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, float | int]] = []

    def evaluate() -> dict[str, float]:
        if not validation:
            return {}
        rows = []
        model.eval()
        with torch.no_grad():
            for episode in validation:
                length = min(args.sequence_length, episode.transitions)
                resident_rows = np.arange(min(args.residents_per_batch, episode.residents))
                terms = training_loss(model, _tensor_window(episode, 0, length, resident_rows, device), args.discount)
                rows.append({name: float(getattr(terms, name)) for name in ("total", "action", "prediction", "memory", "selector", "hazard", "value")})
        model.train()
        return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}

    validation_before = evaluate()
    model.train()
    for update in range(1, args.updates + 1):
        episode = episodes[int(generator.integers(len(episodes)))]
        length = min(args.sequence_length, episode.transitions)
        start = int(generator.integers(episode.transitions - length + 1))
        count = min(args.residents_per_batch, episode.residents)
        residents = generator.choice(episode.residents, size=count, replace=False)
        batch = _tensor_window(episode, start, length, residents, device)
        optimizer.zero_grad(set_to_none=True)
        terms = training_loss(model, batch, args.discount)
        terms.total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        row = {"update": update, "loss": float(terms.total.detach()), "action": float(terms.action.detach()), "prediction": float(terms.prediction.detach()), "memory": float(terms.memory.detach()), "selector": float(terms.selector.detach()), "hazard": float(terms.hazard.detach()), "value": float(terms.value.detach()), "grad_norm": float(grad), "seconds": time.monotonic() - began}
        history.append(row)
        if update == 1 or update % args.report_every == 0 or update == args.updates:
            print(json.dumps(row, sort_keys=True), flush=True)
            atomic_json(output / "progress.json", {"format": RESULT_FORMAT, "complete": False, "latest": row, "history": history})
        if update % args.checkpoint_every == 0 or update == args.updates:
            checkpoint = output / f"checkpoint-{update:06d}.pt"
            torch.save({"format": RESULT_FORMAT, "update": update, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "episodes": [e.metadata["episode_sha256"] for e in episodes]}, checkpoint)
            atomic_json(output / "last-checkpoint.json", {"path": str(checkpoint), "sha256": _sha(checkpoint), "update": update})
    arrays = model.arrays()
    publication = publish_trained(
        output / args.resident_name, output / args.control_name,
        parent_metadata, arrays,
        episode_identities=[{"episode_sha256": e.metadata["episode_sha256"], "file_sha256": e.file_sha256} for e in episodes],
        training={"updates": args.updates, "seed": args.seed, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "sequence_length": args.sequence_length, "discount": args.discount, "device": str(device), "final_loss": history[-1]},
    )
    validation_after = evaluate()
    result = {"format": RESULT_FORMAT, "complete": True, "parent": {"path": str(args.parent.resolve()), "file_sha256": parent_file, "artifact_sha256": parent_metadata["artifact_sha256"]}, "episodes": [{"path": str(e.path), "file_sha256": e.file_sha256, "episode_sha256": e.metadata["episode_sha256"]} for e in episodes], "validation_episodes": [{"path": str(e.path), "file_sha256": e.file_sha256, "episode_sha256": e.metadata["episode_sha256"]} for e in validation], "validation": {"before": validation_before, "after": validation_after}, "publication": publication, "history": history, "seconds": time.monotonic() - began}
    atomic_json(output / "result.json", result)
    atomic_json(output / "progress.json", {"format": RESULT_FORMAT, "complete": True, "result_sha256": _sha(output / "result.json"), "latest": history[-1]})
    print(json.dumps({"result": str(output / "result.json"), "result_sha256": _sha(output / "result.json"), **publication}, sort_keys=True))


def compare_native(args: argparse.Namespace) -> None:
    import torch
    from chreatures.sensorimotor_worker_native import DevelopmentalResidentCohort
    from research.resident_learning.model import CnsResidentModel
    episode = load_episode(args.episode)
    _, arrays = load_parent(args.resident)
    device = torch.device("cpu")
    model = CnsResidentModel.from_arrays(arrays, device).eval()
    count = min(args.residents, episode.residents)
    latent = torch.as_tensor(episode.arrays["cns_latent"][:1, :count])
    previous = torch.as_tensor(episode.arrays["previous_delivered_command"][:1, :count])
    reset = torch.ones((1, count), dtype=torch.bool)
    with torch.no_grad():
        expected = model.unroll(latent, previous, reset)
    cohort = DevelopmentalResidentCohort(args.resident, count, action_mode="map", action_seed=1, suffix_seed=2)
    native = cohort.step(latent[0].numpy(), previous[0].numpy(), np.zeros(count, np.uint64), np.ones(count, np.bool_))
    native_local = native["sequence_control_proposal"][:, :4, :12]
    errors = {
        "recurrent_state_max_abs": float(np.max(np.abs(native["cns_recurrent_state"] - expected["state"][0].numpy()))),
        "current_key_max_abs": float(np.max(np.abs(native["current_cns_key"] - expected["key"][0].numpy()))),
        "local_proposal_max_abs": float(np.max(np.abs(native_local - expected["local"][0].numpy()))),
    }
    tolerance = args.tolerance
    result = {"format": "chreatures-cns-resident-native-comparison-v1", "resident_file_sha256": file_sha256(args.resident.resolve()), "episode_sha256": episode.metadata["episode_sha256"], "residents": count, "tolerance": tolerance, "errors": errors, "passed": max(errors.values()) <= tolerance}
    if args.output:
        atomic_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-recording")
    convert.add_argument("--recording", type=Path, required=True); convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--effort-cost", type=float, default=0.02); convert.set_defaults(function=convert_recording)
    show = commands.add_parser("inspect"); show.add_argument("--episode", type=Path, required=True); show.set_defaults(function=inspect)
    joined = commands.add_parser("pack-joined")
    joined.add_argument("--source", type=Path, required=True); joined.add_argument("--parent", type=Path, required=True); joined.add_argument("--output", type=Path, required=True); joined.set_defaults(function=pack_joined)
    fit = commands.add_parser("train")
    fit.add_argument("--parent", type=Path, required=True); fit.add_argument("--episode", type=Path, action="append", required=True)
    fit.add_argument("--validation-episode", type=Path, action="append", default=[])
    fit.add_argument("--output", type=Path, required=True); fit.add_argument("--resident-name", default="cns-resident-trained.npz"); fit.add_argument("--control-name", default="sequence-control-trained.npz")
    fit.add_argument("--updates", type=int, default=256); fit.add_argument("--sequence-length", type=int, default=32); fit.add_argument("--residents-per-batch", type=int, default=8)
    fit.add_argument("--learning-rate", type=float, default=2e-4); fit.add_argument("--weight-decay", type=float, default=1e-5); fit.add_argument("--discount", type=float, default=0.97); fit.add_argument("--max-grad-norm", type=float, default=1.0)
    fit.add_argument("--checkpoint-every", type=int, default=32); fit.add_argument("--report-every", type=int, default=8); fit.add_argument("--seed", type=int, default=20260907); fit.add_argument("--device", default="cpu"); fit.set_defaults(function=train)
    compare = commands.add_parser("compare-native")
    compare.add_argument("--resident", type=Path, required=True); compare.add_argument("--episode", type=Path, required=True); compare.add_argument("--residents", type=int, default=4); compare.add_argument("--tolerance", type=float, default=3e-5); compare.add_argument("--output", type=Path); compare.set_defaults(function=compare_native)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = arguments(); parsed.function(parsed)
