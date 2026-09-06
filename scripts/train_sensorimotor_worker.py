#!/usr/bin/env python3
"""Train a research worker from actual achieved sensory histories.

This writes an offline candidate, never a live resident or a promoted policy.
World slots are partitioned before normalization and representation learning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.sensorimotor_skills.data import PlayDataset, Normalizer
from research.sensorimotor_skills.model import GoalEncoder, SensorimotorWorker

BUCKETS = ((1, 2), (3, 5), (6, 10), (11, 20), (21, 40))
ARTIFACT_MODES = ("goal-conditioned", "goal-free")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(path, value, *, tensor=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        if tensor:
            torch.save(value, handle)
        else:
            handle.write(
                (
                    json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ).encode()
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def windows(observations, ends):
    # Each goal consists solely of observations ending at its achieved time.
    # No action suffix or later-than-goal observation enters the representation.
    offsets = torch.arange(-3, 1, device=observations.device)
    return observations[ends[:, None] + offsets]


def sample_goal_windows(episodes, columns, normalizer, rng, count, device):
    episode = episodes[int(rng.integers(len(episodes)))]
    ends = rng.integers(3, len(episode.observations), size=count)
    residents = rng.choice(columns, size=count)
    value = episode.observations[ends[:, None] + np.arange(-3, 1), residents[:, None]]
    return torch.as_tensor(normalizer.normalize(value), device=device)


@torch.no_grad()
def representation_metrics(encoder, episodes, columns, normalizer, seed, device):
    value = sample_goal_windows(
        episodes, columns, normalizer, np.random.default_rng(seed), 1024, device
    )
    latent, reconstruction = encoder(value)
    singular = torch.linalg.svdvals(latent - latent.mean(0))
    probability = singular.square() / singular.square().sum().clamp_min(1e-12)
    rank = torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
    return {
        "huber": float(F.huber_loss(reconstruction, value)),
        "training_mean_baseline_huber": float(
            F.huber_loss(torch.zeros_like(value), value)
        ),
        "latent_effective_rank": float(rank),
        "windows": len(value),
    }


@torch.no_grad()
def prepare(episode, columns, normalizer, encoder, device):
    observation = torch.as_tensor(
        normalizer.normalize(episode.observations[:, columns]), device=device
    )
    previous = torch.as_tensor(episode.previous[:, columns], device=device)
    action = torch.as_tensor(episode.actions[:, columns], device=device)
    reset = torch.as_tensor(episode.reset[:, columns], device=device)
    codes = observation.new_zeros((*observation.shape[:2], encoder.config.goal_dim))
    # Encode once after freezing; the worker cannot collapse or rewrite its goals.
    for start in range(3, len(observation), 32):
        ends = torch.arange(start, min(start + 32, len(observation)), device=device)
        window = windows(observation, ends).transpose(1, 2)
        codes[ends] = encoder.encode(window)
    return {
        "observation": observation,
        "previous": previous,
        "action": action,
        "reset": reset,
        "codes": codes,
        "episode": episode.episode,
    }


def quantized(action):
    result = torch.round((action + 1) * 32).long()
    result[..., 3:7] = torch.ceil(action[..., 3:7] * 32).long()
    return result


def goals_for(sequence, start, stop, rng):
    shape = (stop - start, sequence["action"].shape[1])
    bucket = rng.integers(0, len(BUCKETS), size=shape)
    offsets = np.empty(shape, dtype=np.int64)
    for index, (low, high) in enumerate(BUCKETS):
        mask = bucket == index
        offsets[mask] = rng.integers(low, high + 1, size=int(mask.sum()))
    device = sequence["action"].device
    offsets = torch.as_tensor(offsets, device=device)
    future = torch.arange(start, stop, device=device)[:, None] + offsets
    residents = torch.arange(shape[1], device=device)[None, :]
    goal = sequence["codes"][future, residents]
    alternative_offsets = offsets.clone()
    for index, (low, high) in enumerate(BUCKETS):
        selected = torch.as_tensor(bucket == index, device=device)
        width = high - low + 1
        alternative_offsets[selected] = (
            (alternative_offsets[selected] - low + 1) % width
        ) + low
    alternative_future = (
        torch.arange(start, stop, device=device)[:, None] + alternative_offsets
    )
    alternative = sequence["codes"][alternative_future, residents]
    horizon = offsets.float().log1p() / math.log(41)
    return goal, alternative, horizon, bucket


def chunks(worker, sequence, chunk_size):
    hidden = None
    # Reserve the same complete horizon range for every scored observation.
    limit = len(sequence["action"]) - 40
    for start in range(0, limit, chunk_size):
        stop = min(start + chunk_size, limit)
        state, hidden = worker.encode_sequence(
            sequence["observation"][start:stop],
            sequence["previous"][start:stop],
            hidden,
            sequence["reset"][start:stop],
        )
        hidden = hidden.detach()
        yield start, stop, state


def effective_goal(goal, artifact_mode):
    return torch.zeros_like(goal) if artifact_mode == "goal-free" else goal


@torch.no_grad()
def evaluate(worker, sequences, *, seed, chunk_size, artifact_mode, controls=False):
    worker.eval()
    rng = np.random.default_rng(seed)
    results = []
    for sequence in sequences:
        count = 0
        changed_count = 0
        total = np.zeros(8)
        correct = np.zeros(8)
        change_total = np.zeros(8)
        worker_squared_error = np.zeros(8)
        repeat_squared_error = np.zeros(8)
        worker_change_squared_error = np.zeros(8)
        repeat_change_squared_error = np.zeros(8)
        stopping_count = np.zeros(8)
        worker_stopping_squared_error = np.zeros(8)
        repeat_stopping_squared_error = np.zeros(8)
        repeat_correct = np.zeros(8)
        control_total = {
            name: np.zeros(8)
            for name in (
                "permuted_goal",
                "zero_goal",
                "zero_explicit_previous",
            )
        }
        bucket_sum = np.zeros((5, 2))
        per_resident = np.zeros((sequence["action"].shape[1], 3))
        for start, stop, state in chunks(worker, sequence, chunk_size):
            goal, alternative, horizon, bucket = goals_for(sequence, start, stop, rng)
            goal = effective_goal(goal, artifact_mode)
            previous = sequence["previous"][start:stop, :, :8]
            action = sequence["action"][start:stop]
            valid = (
                torch.arange(start, stop, device=action.device)[:, None].expand(
                    action.shape[:2]
                )
                >= 3
            )
            logits = worker.policy(state, goal, horizon, previous)
            loss = worker.action_nll(logits, action)
            decoded = worker.decode(logits)
            target_bins = quantized(action)
            previous_bins = quantized(previous)
            changed_axes = target_bins != previous_bins
            changed = changed_axes.any(-1) & valid
            zero_bins = torch.as_tensor(
                [32, 32, 32, 0, 0, 0, 0, 32], device=action.device
            )
            stopping = (previous_bins != zero_bins) & (target_bins == zero_bins)
            stopping &= valid[..., None]
            total += loss[valid].sum(0).cpu().numpy()
            correct += (quantized(decoded) == target_bins)[valid].sum(0).cpu().numpy()
            repeat_correct += (previous_bins == target_bins)[valid].sum(0).cpu().numpy()
            change_total += loss[changed].sum(0).cpu().numpy()
            squared_worker = (decoded - action).square()
            squared_repeat = (previous - action).square()
            worker_squared_error += squared_worker[valid].sum(0).cpu().numpy()
            repeat_squared_error += squared_repeat[valid].sum(0).cpu().numpy()
            worker_change_squared_error += squared_worker[changed].sum(0).cpu().numpy()
            repeat_change_squared_error += squared_repeat[changed].sum(0).cpu().numpy()
            stopping_count += stopping.sum((0, 1)).cpu().numpy()
            worker_stopping_squared_error += (
                (squared_worker * stopping)[valid].sum(0).cpu().numpy()
            )
            repeat_stopping_squared_error += (
                (squared_repeat * stopping)[valid].sum(0).cpu().numpy()
            )
            count += int(valid.sum())
            changed_count += int(changed.sum())
            for name, target, old_action in (
                (
                    ("permuted_goal", alternative, previous),
                    ("zero_goal", torch.zeros_like(goal), previous),
                    (
                        "zero_explicit_previous",
                        goal,
                        torch.zeros_like(previous),
                    ),
                )
                if controls
                else ()
            ):
                target = effective_goal(target, artifact_mode)
                altered = worker.action_nll(
                    worker.policy(state, target, horizon, old_action), action
                )
                control_total[name] += altered[valid].sum(0).cpu().numpy()
                if name == "permuted_goal":
                    difference = (altered - loss).mean(-1)
                    for b in range(5):
                        selected = (
                            torch.as_tensor(bucket == b, device=action.device) & valid
                        )
                        bucket_sum[b] += (
                            float(difference[selected].sum()),
                            int(selected.sum()),
                        )
                    per_resident[:, 0] += (loss.mean(-1) * valid).sum(0).cpu().numpy()
                    per_resident[:, 1] += (
                        (altered.mean(-1) * valid).sum(0).cpu().numpy()
                    )
                    per_resident[:, 2] += valid.sum(0).cpu().numpy()
        result = {
            "episode": sequence["episode"],
            "samples": count,
            "natural_nll_by_axis": (total / count).tolist(),
            "quantized_accuracy_by_axis": (correct / count).tolist(),
            "repeat_last_quantized_accuracy_by_axis": (repeat_correct / count).tolist(),
            "action_persistence_by_axis": (repeat_correct / count).tolist(),
            "worker_map_rmse_by_axis": np.sqrt(worker_squared_error / count).tolist(),
            "repeat_last_rmse_by_axis": np.sqrt(repeat_squared_error / count).tolist(),
            "action_change_samples": changed_count,
            "action_change_nll_by_axis": (
                change_total / max(changed_count, 1)
            ).tolist(),
            "worker_map_action_change_rmse_by_axis": np.sqrt(
                worker_change_squared_error / max(changed_count, 1)
            ).tolist(),
            "repeat_last_action_change_rmse_by_axis": np.sqrt(
                repeat_change_squared_error / max(changed_count, 1)
            ).tolist(),
            "stopping_samples_by_axis": stopping_count.astype(int).tolist(),
            "worker_map_stopping_rmse_by_axis": np.sqrt(
                worker_stopping_squared_error / np.maximum(stopping_count, 1)
            ).tolist(),
            "repeat_last_stopping_rmse_by_axis": np.sqrt(
                repeat_stopping_squared_error / np.maximum(stopping_count, 1)
            ).tolist(),
        }
        if controls:
            result.update(
                {
                    "controls_nll_by_axis": {
                        k: (v / count).tolist() for k, v in control_total.items()
                    },
                    "permuted_goal_delta_nll_by_horizon_bucket": (
                        bucket_sum[:, 0] / np.maximum(bucket_sum[:, 1], 1)
                    ).tolist(),
                    "per_resident_nll_and_permuted_nll": (
                        per_resident[:, :2] / np.maximum(per_resident[:, 2:], 1)
                    ).tolist(),
                }
            )
        results.append(result)
    return {
        "episodes": results,
        "mean_nll": float(np.mean([r["natural_nll_by_axis"] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact-mode", choices=ARTIFACT_MODES, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--goal-updates", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=64)
    args = parser.parse_args()
    if args.goal_updates < 1 or args.epochs < 1 or args.chunk < 2:
        raise SystemExit("positive updates/epochs and chunk >=2 required")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    dataset = PlayDataset(args.dataset)
    if any(len(episode.actions) <= 43 for episode in dataset.episodes):
        raise SystemExit(
            "episodes need more than 43 transitions for complete goal windows"
        )
    normalizer = Normalizer.fit(dataset)
    encoder = GoalEncoder().to(args.device)
    goal_optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-5)
    identity = {
        "dataset": dict(dataset.identity.file_sha256s),
        "normalizer": normalizer.to_value(),
        "config": asdict(encoder.config),
        "seed": args.seed,
        "goal_updates": args.goal_updates,
        "worker_epochs": args.epochs,
        "chunk": args.chunk,
        "offset_buckets": BUCKETS,
        "artifact_mode": args.artifact_mode,
        "alternate_goal_control": (
            "another achieved time for the same resident and episode, with a "
            "different offset in the same log-horizon bucket; not asserted reachable"
        ),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(args.device),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "research/sensorimotor_skills/model.py",
                ROOT / "research/sensorimotor_skills/data.py",
            )
        },
    }
    atomic_save(args.output / "identity.json", identity)
    for update in range(args.goal_updates):
        value = sample_goal_windows(
            dataset.episodes,
            dataset.indices("train"),
            normalizer,
            rng,
            256,
            args.device,
        )
        loss, _ = encoder.reconstruction_loss(value)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite goal representation loss")
        goal_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        goal_optimizer.step()
        if (update + 1) % 64 == 0:
            print(
                json.dumps({"goal_update": update + 1, "loss": float(loss.detach())}),
                flush=True,
            )
    encoder.freeze()
    representation_validation = representation_metrics(
        encoder,
        dataset.episodes,
        dataset.indices("validation"),
        normalizer,
        args.seed + 1,
        args.device,
    )
    sequences = {
        split: [
            prepare(ep, dataset.indices(split), normalizer, encoder, args.device)
            for ep in dataset.episodes
        ]
        for split in ("train", "validation")
    }
    worker = SensorimotorWorker().to(args.device)
    optimizer = torch.optim.AdamW(worker.parameters(), lr=3e-4, weight_decay=1e-5)
    best = math.inf
    trace = []
    for epoch in range(args.epochs):
        worker.train()
        losses = []
        for index in rng.permutation(len(sequences["train"])):
            sequence = sequences["train"][index]
            for start, stop, state in chunks(worker, sequence, args.chunk):
                goal, _, horizon, _ = goals_for(sequence, start, stop, rng)
                goal = effective_goal(goal, args.artifact_mode)
                previous = sequence["previous"][start:stop, :, :8]
                action = sequence["action"][start:stop]
                valid = (
                    torch.arange(start, stop, device=action.device)[:, None].expand(
                        action.shape[:2]
                    )
                    >= 3
                )
                changed = (quantized(action) != quantized(previous)).any(-1) & valid
                weights = valid.float() / valid.sum().clamp_min(1)
                if changed.any():
                    weights = 0.5 * weights + 0.5 * changed / changed.sum()
                logits = worker.policy(
                    state, goal, horizon, previous, mask_previous_probability=0.3
                )
                loss = (worker.action_nll(logits, action).mean(-1) * weights).sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite worker loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(worker.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        validation = evaluate(
            worker,
            sequences["validation"],
            seed=args.seed + 2,
            chunk_size=args.chunk,
            artifact_mode=args.artifact_mode,
        )
        record = {
            "epoch": epoch + 1,
            "training_weighted_nll": float(np.mean(losses)),
            "validation_nll": validation["mean_nll"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record), flush=True)
        if validation["mean_nll"] < best:
            best = validation["mean_nll"]
            atomic_save(
                args.output / "worker.pt",
                {
                    "format": "chreatures-sensorimotor-worker-research-v2",
                    "artifact_mode": args.artifact_mode,
                    "identity": identity,
                    "epoch": epoch + 1,
                    "goal_encoder": encoder.state_dict(),
                    "worker": worker.state_dict(),
                },
                tensor=True,
            )
    selected = torch.load(
        args.output / "worker.pt", map_location=args.device, weights_only=False
    )
    if (
        selected.get("format") != "chreatures-sensorimotor-worker-research-v2"
        or selected.get("artifact_mode") != args.artifact_mode
        or selected.get("identity", {}).get("artifact_mode") != args.artifact_mode
    ):
        raise RuntimeError("selected worker artifact mode or format differs")
    worker.load_state_dict(selected["worker"])
    # Heldout goals, representations and action outcomes are evaluated only now.
    holdout = [
        prepare(ep, dataset.indices("holdout"), normalizer, encoder, args.device)
        for ep in dataset.episodes
    ]
    result = {
        "format": "chreatures-sensorimotor-worker-offline-result-v2",
        "artifact_mode": args.artifact_mode,
        "status": "research candidate; no behavioral promotion",
        "selected_epoch": selected["epoch"],
        "trace": trace,
        "representation_validation": representation_validation,
        "representation_holdout": representation_metrics(
            encoder,
            dataset.episodes,
            dataset.indices("holdout"),
            normalizer,
            args.seed + 3,
            args.device,
        ),
        "heldout": evaluate(
            worker,
            holdout,
            seed=args.seed + 4,
            chunk_size=args.chunk,
            artifact_mode=args.artifact_mode,
            controls=True,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_sha256": sha256(args.output / "worker.pt"),
        "limits": [
            "Factorized action heads do not represent all coordinated multimodal plans.",
            "TBPTT detaches carried state at chunk boundaries; parameters change during an episode.",
            "Goal permutation measures reliance on experienced goals, not closed-loop goal attainment.",
            "Oral action remains a supplied physiology law, not a learned worker output.",
            "Two heldout world slots across episodes do not establish broad environment transfer.",
        ],
    }
    atomic_save(args.output / "result.json", result)
    print(
        json.dumps(
            {
                "result": str(args.output / "result.json"),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
