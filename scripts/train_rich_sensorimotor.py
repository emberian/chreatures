#!/usr/bin/env python3
"""Fit the breaking rich visual/goal front and bootstrap worker on real play."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.sensorimotor_skills.rich_data import RichNormalizer, RichPlayDataset
from chreatures.resident_contract import BOOTSTRAP_FORMAT
from research.sensorimotor_skills.rich_model import RichSensorimotorModel

FORMAT = BOOTSTRAP_FORMAT
BUCKETS = ((1, 2), (3, 5), (6, 10), (11, 20), (21, 40))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(path: Path, value, tensor: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if tensor:
        torch.save(value, temporary)
    else:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-worlds", type=int, nargs="+", required=True)
    parser.add_argument("--goal-updates", type=int, default=512)
    parser.add_argument("--worker-epochs", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def goal_batch(dataset, columns, normalizer, rng, count, device):
    episode = dataset.episodes[int(rng.integers(len(dataset.episodes)))]
    ends = rng.integers(3, len(episode.observation), count)
    residents = rng.choice(columns, count)
    offsets = np.arange(-3, 1)
    raw = episode.observation[ends[:, None] + offsets, residents[:, None]]
    return torch.as_tensor(normalizer.normalize(raw), device=device)


def episode_tensors(episode, columns, normalizer, device):
    observation = torch.as_tensor(
        normalizer.normalize(episode.observation[:, columns]), device=device
    )
    previous = torch.as_tensor(episode.previous[:, columns], device=device)
    action = torch.as_tensor(episode.actions[:, columns], device=device)
    reset = torch.as_tensor(episode.reset[:, columns], device=device)
    return observation, previous, action, reset


@torch.no_grad()
def encode_achieved_goals(model, observation):
    codes = observation.new_zeros((*observation.shape[:2], 64))
    for start in range(3, len(observation), 16):
        ends = torch.arange(
            start, min(start + 16, len(observation)), device=observation.device
        )
        offsets = torch.arange(-3, 1, device=observation.device)
        windows = observation[ends[:, None] + offsets].transpose(1, 2)
        codes[ends] = model.encode_goal(windows)
    return codes


def sample_goals(codes, start, stop, rng):
    shape = (stop - start, codes.shape[1])
    bucket = rng.integers(0, len(BUCKETS), shape)
    offsets = np.empty(shape, dtype=np.int64)
    for index, (low, high) in enumerate(BUCKETS):
        mask = bucket == index
        offsets[mask] = rng.integers(low, high + 1, int(mask.sum()))
    offsets = torch.as_tensor(offsets, device=codes.device)
    times = torch.arange(start, stop, device=codes.device)[:, None] + offsets
    residents = torch.arange(codes.shape[1], device=codes.device)[None]
    return codes[times, residents], offsets.float().log1p()[..., None] / math.log(41)


def main() -> int:
    args = arguments()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    if min(args.goal_updates, args.worker_epochs, args.chunk) < 1:
        raise SystemExit("training schedule must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    dataset = RichPlayDataset(args.dataset)
    columns = dataset.columns(args.train_worlds)
    normalizer = RichNormalizer.fit(dataset, args.train_worlds)
    model = RichSensorimotorModel().to(device)
    identity = {
        "format": FORMAT,
        "dataset_files": dataset.file_sha256s,
        "dataset_manifest_content_sha256": dataset.manifest["content_sha256"],
        "normalizer": normalizer.to_value(),
        "config": asdict(model.config),
        "train_worlds": args.train_worlds,
        "seed": args.seed,
        "goal_updates": args.goal_updates,
        "worker_epochs": args.worker_epochs,
        "chunk": args.chunk,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "research/sensorimotor_skills/rich_model.py",
                ROOT / "research/sensorimotor_skills/rich_data.py",
                ROOT / "research/sensorimotor_skills/trajectory-schema-rich-v3.json",
            )
        },
    }
    atomic_save(args.output / "identity.json", identity)
    started = time.perf_counter()
    representation_parameters = [
        *model.visual.parameters(),
        *model.body.parameters(),
        *model.goal_encoder.parameters(),
        *model.goal_decoder.parameters(),
    ]
    optimizer = torch.optim.AdamW(representation_parameters, lr=3e-4, weight_decay=1e-5)
    goal_trace = []
    for update in range(args.goal_updates):
        batch = goal_batch(dataset, columns, normalizer, rng, 64, device)
        loss, terms = model.goal_reconstruction_loss(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite rich goal loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(representation_parameters, 1.0)
        optimizer.step()
        if (update + 1) % 32 == 0:
            record = {"update": update + 1, "loss": float(loss.detach())} | {
                key: float(value) for key, value in terms.items()
            }
            goal_trace.append(record)
            print(json.dumps({"goal": record}), flush=True)
    # Achieved keys must stay stable across online experience. The trained
    # visual/body front is consequently frozen with the goal encoder.
    for module in (model.visual, model.body, model.goal_encoder, model.goal_decoder):
        module.requires_grad_(False)
        module.eval()
    worker_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(worker_parameters, lr=3e-4, weight_decay=1e-5)
    sequences = []
    for episode in dataset.episodes:
        observation, previous, action, reset = episode_tensors(
            episode, columns, normalizer, device
        )
        sequences.append(
            (
                observation,
                previous,
                action,
                reset,
                encode_achieved_goals(model, observation),
            )
        )
    worker_trace = []
    for epoch in range(args.worker_epochs):
        losses = []
        for order in rng.permutation(len(sequences)):
            observation, previous, action, reset, codes = sequences[int(order)]
            hidden = None
            limit = len(action) - 40
            for start in range(0, limit, args.chunk):
                stop = min(start + args.chunk, limit)
                states, hidden = model.encode_sequence(
                    observation[start:stop],
                    previous[start:stop],
                    hidden,
                    reset[start:stop],
                )
                hidden = hidden.detach()
                goal, horizon = sample_goals(codes, start, stop, rng)
                logits = model.policy(states, goal, horizon, previous[start:stop])
                loss = model.action_nll(logits, action[start:stop]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite rich worker loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(worker_parameters, 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        record = {"epoch": epoch + 1, "mean_nll": float(np.mean(losses))}
        worker_trace.append(record)
        print(json.dumps({"worker": record}), flush=True)
    artifact = {
        "format": FORMAT,
        "identity": identity,
        "model": model.state_dict(),
        "frozen_modules": ["visual", "body", "goal_encoder", "goal_decoder"],
        "goal_trace": goal_trace,
        "worker_trace": worker_trace,
    }
    atomic_save(args.output / "rich-worker.pt", artifact, tensor=True)
    atomic_save(
        args.output / "result.json",
        {
            "format": FORMAT,
            "status": "research bootstrap; no behavior claim",
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_sha256": sha256(args.output / "rich-worker.pt"),
            "last_goal": goal_trace[-1],
            "last_worker": worker_trace[-1],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
