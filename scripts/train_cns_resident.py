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
from research.resident_learning.data import (
    CORPUS_FORMAT,
    SKILLS,
    load_episode,
    validate_corpus,
    validate_curriculum,
    write_episode,
)

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


def _cpu_tree(value: Any) -> Any:
    """Move optimizer/model tensors into a backend-portable checkpoint."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def atomic_torch(path: Path, value: Any) -> None:
    import torch
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        torch.save(_cpu_tree(value), handle)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def _pack_joined(source: Path, parent: Path, output: Path):
    source = source.resolve(); parent = parent.resolve(); output = output.resolve()
    parent_metadata, _ = load_parent(parent)
    parent_hash = file_sha256(parent)
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
    if metadata.get("adapter_sha256") != parent_metadata["cns_service"]["adapter_sha256"]:
        raise ValueError("joined history CNS adapter differs from the resident binding")
    if metadata.get("resident_file_sha256") != parent_hash or metadata.get("resident_artifact_sha256") != parent_metadata["artifact_sha256"]:
        raise ValueError("joined history parent resident differs")
    if latent.ndim != 3 or delivered.ndim != 3 or latent.shape[0] != delivered.shape[0] + 1:
        raise ValueError("joined history must align T+1 CNS observations with T deliveries")
    validate_curriculum(metadata, delivered.shape[0])
    if metadata.get("raw_geometry_retained") is not False or metadata.get("raw_senses_retained") is not False:
        raise ValueError("joined history must explicitly discard raw geometry and senses")
    forbidden_metadata = {"targets", "target_positions", "poses", "world_positions", "raw_visual", "raw_sensory"}
    if forbidden_metadata.intersection(metadata):
        raise ValueError("joined history retains privileged/raw controller-bypass metadata")
    previous = np.zeros((latent.shape[0], latent.shape[1], 12), np.float32)
    previous[1:] = delivered
    episode = write_episode(
        output,
        {"cns_latent": latent, "previous_delivered_command": previous, "reset": reset,
         "delivered_command": delivered, "physical_reward": reward, "terminal": terminal},
        cns_service=parent_metadata["cns_service"],
        resident_artifact={"file_sha256": parent_hash, "artifact_sha256": parent_metadata["artifact_sha256"]},
        provenance={"source_joined_file_sha256": file_sha256(source), **metadata},
    )
    return episode, metadata


def pack_joined(args: argparse.Namespace) -> None:
    episode, metadata = _pack_joined(args.source, args.parent, args.output)
    print(json.dumps({"path": str(episode.path), "file_sha256": episode.file_sha256, "episode_sha256": episode.metadata["episode_sha256"], "action_source": metadata["action_source"], "transitions": episode.transitions, "residents": episode.residents}, sort_keys=True))


def pack_corpus(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve(); output = args.output.resolve()
    collection_receipt_path = args.collection_receipt.resolve()
    collection_receipt_bytes = collection_receipt_path.read_bytes()
    collection_receipt = json.loads(collection_receipt_bytes)
    if collection_receipt.get("format") != "chreatures-browser-teacher-collection-combined-receipt-v2" or collection_receipt.get("complete") is not True:
        raise ValueError("teacher collection receipt is incomplete or differs")
    contract = collection_receipt.get("curriculum_contract_sha256")
    if not isinstance(contract, str) or len(contract) != 64 or any(value not in "0123456789abcdef" for value in contract):
        raise ValueError("teacher curriculum contract identity differs")
    receipt_episodes = collection_receipt.get("episodes")
    if not isinstance(receipt_episodes, list) or len(receipt_episodes) != 8:
        raise ValueError("teacher collection receipt episode count differs")
    if not isinstance(collection_receipt.get("source_amendments"), list):
        raise ValueError("teacher collection source amendments are missing")
    receipt_by_index = {}
    for receipt in receipt_episodes:
        required = {"file", "sha256", "episode_index", "split", "collector_source_sha256"}
        if not isinstance(receipt, dict) or not required.issubset(receipt):
            raise ValueError("teacher collection episode receipt differs")
        index = receipt["episode_index"]
        source_sha = receipt["collector_source_sha256"]
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(source_sha, str) or len(source_sha) != 64 or any(value not in "0123456789abcdef" for value in source_sha):
            raise ValueError("teacher collection episode source identity differs")
        expected_split = "train" if index < 6 else "heldout-worlds"
        if receipt["split"] != expected_split:
            raise ValueError("teacher collection episode split differs")
        if index in receipt_by_index:
            raise ValueError("teacher collection repeats an episode receipt")
        receipt_by_index[index] = receipt
    if set(receipt_by_index) != set(range(8)):
        raise ValueError("teacher collection episode indices differ")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    episodes = []
    entries = []
    for index in range(8):
        split = "train" if index < 6 else "heldout-world"
        source = source_dir / f"episode-{index:02d}-{split}.npz"
        source_receipt = receipt_by_index[index]
        if source_receipt["file"] != source.name or source_receipt["sha256"] != file_sha256(source):
            raise ValueError("teacher source file differs from its combined receipt")
        destination = output / f"resident-episode-{index:02d}-{split}.npz"
        episode, metadata = _pack_joined(source, args.parent, destination)
        if metadata.get("curriculum_contract_sha256", contract) != contract:
            raise ValueError("teacher episode curriculum contract differs")
        episodes.append(episode)
        entries.append({
            "episode_index": index,
            "split": metadata["split"],
            "source": str(source),
            "source_file_sha256": file_sha256(source),
            "collector_source_sha256": source_receipt["collector_source_sha256"],
            "file": destination.name,
            "file_sha256": episode.file_sha256,
            "episode_sha256": episode.metadata["episode_sha256"],
            "world_seed": metadata["world_seed"],
            "variation_seed": metadata["variation_seed"],
            "layout_identity": metadata["layout_identity"],
        })
    train_episodes, validation_episodes = validate_corpus(episodes)
    parent = args.parent.resolve(); parent_metadata, _ = load_parent(parent)
    manifest = {
        "format": CORPUS_FORMAT,
        "collection_receipt": {"file": "collection-receipt.json", "file_sha256": hashlib.sha256(collection_receipt_bytes).hexdigest()},
        "curriculum_contract_sha256": contract,
        "parent": {"file_sha256": file_sha256(parent), "artifact_sha256": parent_metadata["artifact_sha256"]},
        "controller_input_fields": ["cns_latent", "previous_delivered_command", "reset"],
        "teacher_only_fields": ["delivered_command", "physical_reward", "terminal", "teacher_bouts"],
        "episodes": entries,
        "train": [episode.path.name for episode in train_episodes],
        "heldout_worlds": [episode.path.name for episode in validation_episodes],
    }
    with (output / "collection-receipt.json").open("xb") as handle:
        handle.write(collection_receipt_bytes); handle.flush(); os.fsync(handle.fileno())
    os.chmod(output / "collection-receipt.json", 0o444)
    atomic_json(output / "corpus.json", manifest)
    os.chmod(output / "corpus.json", 0o444)
    print(json.dumps({"manifest": str(output / "corpus.json"), "sha256": _sha(output / "corpus.json"), "train_episodes": 6, "heldout_worlds": 2}, sort_keys=True))


def _load_corpus(path: Path, parent_file: str) -> tuple[list[Any], list[Any], str]:
    manifest_path = path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != {"format", "collection_receipt", "curriculum_contract_sha256", "parent", "controller_input_fields", "teacher_only_fields", "episodes", "train", "heldout_worlds"} or manifest["format"] != CORPUS_FORMAT:
        raise ValueError("resident corpus manifest differs")
    if manifest["parent"].get("file_sha256") != parent_file:
        raise ValueError("resident corpus parent file differs")
    if manifest["controller_input_fields"] != ["cns_latent", "previous_delivered_command", "reset"]:
        raise ValueError("resident corpus controller boundary differs")
    base = manifest_path.parent
    receipt_path = base / manifest["collection_receipt"]["file"]
    if file_sha256(receipt_path) != manifest["collection_receipt"]["file_sha256"]:
        raise ValueError("resident corpus collection receipt differs")
    collection_receipt = json.loads(receipt_path.read_text())
    if collection_receipt.get("complete") is not True or collection_receipt.get("curriculum_contract_sha256") != manifest["curriculum_contract_sha256"]:
        raise ValueError("resident corpus curriculum contract differs")
    episodes = []
    for entry in manifest["episodes"]:
        entry_fields = {"episode_index", "split", "source", "source_file_sha256", "collector_source_sha256", "file", "file_sha256", "episode_sha256", "world_seed", "variation_seed", "layout_identity"}
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise ValueError("resident corpus episode manifest fields differ")
        episode = load_episode(base / entry["file"])
        if episode.file_sha256 != entry["file_sha256"] or episode.metadata["episode_sha256"] != entry["episode_sha256"]:
            raise ValueError("resident corpus episode receipt differs")
        curriculum = episode.curriculum
        for name in ("episode_index", "split", "world_seed", "variation_seed", "layout_identity"):
            if entry[name] != curriculum[name]:
                raise ValueError("resident corpus episode identity metadata differs")
        episodes.append(episode)
    train, validation = validate_corpus(episodes)
    if manifest["train"] != [episode.path.name for episode in train] or manifest["heldout_worlds"] != [episode.path.name for episode in validation]:
        raise ValueError("resident corpus split listing differs")
    return train, validation, _sha(manifest_path)


def _tensor_window(episode, start: int, length: int, residents: np.ndarray, device: torch.device, burn_in: int = 0) -> dict[str, Any]:
    import torch
    arrays = episode.arrays
    context_start = max(0, start - burn_in)
    context_length = start - context_start
    end = start + length
    result = {
        "cns_latent": arrays["cns_latent"][context_start : end + 1, residents],
        "previous_delivered_command": arrays["previous_delivered_command"][context_start : end + 1, residents],
        "reset": arrays["reset"][context_start : end + 1, residents],
        "delivered_command": arrays["delivered_command"][context_start:end, residents],
        "physical_reward": arrays["physical_reward"][context_start:end, residents],
        "terminal": arrays["terminal"][context_start:end, residents],
    }
    # A truncated chronology starts from zero, consumes up to burn_in actual
    # CNS/action ticks without gradients, then optimizes the requested window.
    result["reset"] = result["reset"].copy(); result["reset"][0] = True
    converted = {name: torch.as_tensor(value, device=device) for name, value in result.items()}
    converted["burn_in"] = context_length
    return converted


def _bout_windows(episodes, length: int) -> dict[str, list[tuple[Any, int, int]]]:
    windows = {skill: [] for skill in SKILLS}
    for episode in episodes:
        for bout in episode.curriculum["teacher_bouts"]:
            first, end = bout["start_tick"], bout["end_tick"]
            if end - first >= length:
                windows[bout["skill"]].append((episode, first, end - length))
    missing = [skill for skill, rows in windows.items() if not rows]
    if missing:
        raise ValueError(f"corpus has no full training window for skills {missing}")
    return windows


def train(args: argparse.Namespace) -> None:
    import torch
    from research.resident_learning.model import CnsResidentModel, training_loss
    began = time.monotonic()
    if args.updates < 1 or args.sequence_length < 8 or args.burn_in < 0 or args.residents_per_batch < 1:
        raise ValueError("training sizes must be positive and sequence length must cover the eight-step skill horizon")
    if args.checkpoint_every < 1 or args.report_every < 1 or not 0 < args.discount <= 1:
        raise ValueError("training intervals or discount differ")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested ROCm/CUDA device is unavailable")
    parent_metadata, parent_arrays = load_parent(args.parent)
    parent_file = file_sha256(args.parent.resolve())
    episodes, validation, corpus_sha = _load_corpus(args.corpus, parent_file)
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
    training_windows = _bout_windows(episodes, args.sequence_length)
    validation_windows = _bout_windows(validation, args.sequence_length)
    output = args.output.resolve()
    restored_history = None
    restored_validation_before = None
    if args.resume is None:
        output.mkdir(parents=True, exist_ok=False)
        first_update = 1
    else:
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        expected = {"format", "update", "model", "optimizer", "parent_file_sha256", "corpus_sha256", "numpy_rng_state", "training", "history", "validation_before"}
        if set(checkpoint) != expected or checkpoint["format"] != RESULT_FORMAT:
            raise ValueError("resident optimizer checkpoint differs")
        if checkpoint["parent_file_sha256"] != parent_file or checkpoint["corpus_sha256"] != corpus_sha:
            raise ValueError("resident optimizer checkpoint lineage differs")
        current_training = {"seed": args.seed, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "sequence_length": args.sequence_length, "burn_in": args.burn_in, "residents_per_batch": args.residents_per_batch, "discount": args.discount, "max_grad_norm": args.max_grad_norm}
        if checkpoint["training"] != current_training:
            raise ValueError("resident optimizer checkpoint hyperparameters differ")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.bit_generator.state = checkpoint["numpy_rng_state"]
        restored_history = checkpoint["history"]
        restored_validation_before = checkpoint["validation_before"]
        first_update = int(checkpoint["update"]) + 1
        if first_update > args.updates:
            raise ValueError("resume checkpoint is already beyond requested updates")
    history: list[dict[str, Any]] = [] if restored_history is None else restored_history

    def evaluate() -> dict[str, float]:
        if not validation:
            return {}
        rows = []
        by_skill = {skill: [] for skill in SKILLS}
        model.eval()
        with torch.no_grad():
            for skill in SKILLS:
                for episode, low, high in validation_windows[skill]:
                    start = (low + high) // 2
                    resident_rows = np.arange(min(args.residents_per_batch, episode.residents))
                    terms = training_loss(model, _tensor_window(episode, start, args.sequence_length, resident_rows, device, args.burn_in), args.discount)
                    row = {name: float(getattr(terms, name)) for name in ("total", "action", "prediction", "memory", "selector", "hazard", "value")}
                    rows.append(row); by_skill[skill].append(row)
        model.train()
        overall = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
        return {"overall": overall, "by_skill": {skill: {name: float(np.mean([row[name] for row in skill_rows])) for name in skill_rows[0]} for skill, skill_rows in by_skill.items()}}

    validation_before = evaluate() if restored_validation_before is None else restored_validation_before
    model.train()
    for update in range(first_update, args.updates + 1):
        skill = SKILLS[int(generator.integers(len(SKILLS)))]
        candidates = training_windows[skill]
        episode, low, high = candidates[int(generator.integers(len(candidates)))]
        start = int(generator.integers(low, high + 1))
        count = min(args.residents_per_batch, episode.residents)
        residents = generator.choice(episode.residents, size=count, replace=False)
        batch = _tensor_window(episode, start, args.sequence_length, residents, device, args.burn_in)
        optimizer.zero_grad(set_to_none=True)
        terms = training_loss(model, batch, args.discount)
        terms.total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        row = {"update": update, "skill": skill, "episode_index": episode.curriculum["episode_index"], "start_tick": start, "burn_in": int(batch["burn_in"]), "loss": float(terms.total.detach()), "action": float(terms.action.detach()), "prediction": float(terms.prediction.detach()), "memory": float(terms.memory.detach()), "selector": float(terms.selector.detach()), "hazard": float(terms.hazard.detach()), "value": float(terms.value.detach()), "grad_norm": float(grad), "seconds": time.monotonic() - began}
        history.append(row)
        if update == 1 or update % args.report_every == 0 or update == args.updates:
            print(json.dumps(row, sort_keys=True), flush=True)
            atomic_json(output / "progress.json", {"format": RESULT_FORMAT, "complete": False, "latest": row, "history": history})
        if update % args.checkpoint_every == 0 or update == args.updates:
            checkpoint = output / f"checkpoint-{update:06d}.pt"
            atomic_torch(checkpoint, {"format": RESULT_FORMAT, "update": update, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "parent_file_sha256": parent_file, "corpus_sha256": corpus_sha, "numpy_rng_state": generator.bit_generator.state, "training": {"seed": args.seed, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "sequence_length": args.sequence_length, "burn_in": args.burn_in, "residents_per_batch": args.residents_per_batch, "discount": args.discount, "max_grad_norm": args.max_grad_norm}, "history": history, "validation_before": validation_before})
            atomic_json(output / "last-checkpoint.json", {"path": str(checkpoint), "sha256": _sha(checkpoint), "update": update})
    arrays = model.arrays()
    publication = publish_trained(
        output / args.resident_name, output / args.control_name,
        parent_metadata, arrays,
        episode_identities=[{"episode_sha256": e.metadata["episode_sha256"], "file_sha256": e.file_sha256} for e in episodes],
        training={"updates": args.updates, "seed": args.seed, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "sequence_length": args.sequence_length, "burn_in": args.burn_in, "residents_per_batch": args.residents_per_batch, "discount": args.discount, "max_grad_norm": args.max_grad_norm, "device": str(device), "corpus_sha256": corpus_sha, "balanced_skills": list(SKILLS), "final_loss": history[-1]},
    )
    validation_after = evaluate()
    result = {"format": RESULT_FORMAT, "complete": True, "source": {"trainer_sha256": _sha(Path(__file__)), "model_sha256": _sha(ROOT / "research/resident_learning/model.py"), "data_sha256": _sha(ROOT / "research/resident_learning/data.py")}, "parent": {"path": str(args.parent.resolve()), "file_sha256": parent_file, "artifact_sha256": parent_metadata["artifact_sha256"]}, "corpus": {"path": str(args.corpus.resolve()), "file_sha256": corpus_sha}, "episodes": [{"path": str(e.path), "file_sha256": e.file_sha256, "episode_sha256": e.metadata["episode_sha256"]} for e in episodes], "validation_episodes": [{"path": str(e.path), "file_sha256": e.file_sha256, "episode_sha256": e.metadata["episode_sha256"]} for e in validation], "validation": {"before": validation_before, "after": validation_after}, "publication": publication, "history": history, "seconds": time.monotonic() - began}
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
    corpus = commands.add_parser("pack-corpus")
    corpus.add_argument("--source-dir", type=Path, required=True); corpus.add_argument("--collection-receipt", type=Path, required=True); corpus.add_argument("--parent", type=Path, required=True); corpus.add_argument("--output", type=Path, required=True); corpus.set_defaults(function=pack_corpus)
    fit = commands.add_parser("train")
    fit.add_argument("--parent", type=Path, required=True); fit.add_argument("--corpus", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True); fit.add_argument("--resident-name", default="cns-resident-trained.npz"); fit.add_argument("--control-name", default="sequence-control-trained.npz")
    fit.add_argument("--updates", type=int, default=256); fit.add_argument("--sequence-length", type=int, default=32); fit.add_argument("--burn-in", type=int, default=32); fit.add_argument("--residents-per-batch", type=int, default=8)
    fit.add_argument("--learning-rate", type=float, default=2e-4); fit.add_argument("--weight-decay", type=float, default=1e-5); fit.add_argument("--discount", type=float, default=0.97); fit.add_argument("--max-grad-norm", type=float, default=1.0)
    fit.add_argument("--checkpoint-every", type=int, default=32); fit.add_argument("--report-every", type=int, default=8); fit.add_argument("--seed", type=int, default=20260907); fit.add_argument("--device", default="cpu"); fit.add_argument("--resume", type=Path); fit.set_defaults(function=train)
    compare = commands.add_parser("compare-native")
    compare.add_argument("--resident", type=Path, required=True); compare.add_argument("--episode", type=Path, required=True); compare.add_argument("--residents", type=int, default=4); compare.add_argument("--tolerance", type=float, default=3e-5); compare.add_argument("--output", type=Path); compare.set_defaults(function=compare_native)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = arguments(); parsed.function(parsed)
