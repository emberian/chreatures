#!/usr/bin/env python3
"""Train the current recurrent rich consequence ensemble from executed trajectories."""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.organism_interface import ACTION_NAMES, PHYSIOLOGY_NAMES
from chreatures.resident_contract import BOOTSTRAP_FORMAT, DEVELOPMENT_FORMAT
from research.sensorimotor_skills.rich_data import (
    RichNormalizer,
    RichPlayDataset,
    canonical_sha256,
)
from research.sensorimotor_skills.rich_model import RichSensorimotorModel
from research.sensorimotor_skills.rich_prediction import (
    ACTION_DIM,
    ACTION_SCALE_FLOOR,
    CODE_DELTA_SCALE_FLOOR,
    CONTEXT_DIM,
    CONTEXT_SEGMENTS,
    CONTEXT_SCALE_FLOOR,
    FORMAT,
    FRAME_CODE_DIM,
    MAX_HORIZON,
    MEMBERS,
    OBSERVATION_INTERVAL_SECONDS,
    OUTPUT_DIM,
    PHYSIOLOGY_DELTA_SCALE_FLOOR,
    PHYSIOLOGY_DIM,
    PHYSIOLOGY_LINK_EPSILON,
    PHYSIOLOGY_LOWER,
    PHYSIOLOGY_UPPER,
    RichRecurrentConsequenceEnsemble,
    artifact_identity,
    bounded_physiology_deltas,
    denormalize_deltas,
    normalize_actions,
    normalize_context,
    tensor_bundle_sha256,
)

def sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name("." + path.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )
    os.replace(tmp, path)


def atomic_npz(path, arrays):
    tmp = path.with_name("." + path.name + f".tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--representation-checkpoint", type=Path, required=True)
    p.add_argument("--expected-representation-sha256", required=True)
    p.add_argument("--expected-collection-source-revision", required=True)
    p.add_argument("--expected-resident-artifact-sha256", required=True)
    p.add_argument("--expected-resident-artifact-identity", required=True)
    p.add_argument("--initial-predictor", type=Path, required=True)
    p.add_argument("--expected-initial-predictor-sha256", required=True)
    p.add_argument("--expected-initial-predictor-identity", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--row-cache",
        type=Path,
        required=True,
        help="bulk directory for authenticated disk-backed encoded predictor rows",
    )
    p.add_argument("--trusted-checkpoint", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--device", default="cuda")
    p.add_argument("--encoding-batch-size", type=int, default=2048)
    p.add_argument(
        "--validation-worlds",
        type=int,
        nargs="+",
        required=True,
        help="whole world slots held out from fitting and evaluated once",
    )
    p.add_argument(
        "--heldout-worlds",
        type=int,
        nargs="+",
        required=True,
        help="whole world slots evaluated once after fitting",
    )
    return p.parse_args()


def load_representation(path, device):
    if not path.is_file():
        raise ValueError("representation checkpoint is missing")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("format") not in (BOOTSTRAP_FORMAT, DEVELOPMENT_FORMAT):
        raise ValueError("representation must be a current v5 checkpoint")
    identity = copy.deepcopy(value["identity"])
    model = RichSensorimotorModel()
    model.load_state_dict(value["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    normalizer = RichNormalizer.from_value(identity["normalizer"])
    state = {
        name: np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4")
        for name, t in model.state_dict().items()
    }
    deployed_state = {
        name: value
        for name, value in state.items()
        if not name.startswith("goal_decoder.")
        and name not in {"signed_centers", "positive_centers"}
    }
    groups = {
        "frame": {k: v for k, v in state.items() if k.startswith(("visual.", "body."))},
        "goal": {k: v for k, v in state.items() if k.startswith("goal_encoder.")},
        "worker_recurrent": {
            k: v
            for k, v in state.items()
            if k.startswith(("observation_projection.", "physiology_adapter.", "history."))
        },
    }
    return (
        model,
        normalizer,
        identity,
        {
            "file_sha256": sha256(path),
            "format": value["format"],
            "model_tensor_sha256": tensor_bundle_sha256(deployed_state),
            "frame_encoder_sha256": tensor_bundle_sha256(groups["frame"]),
            "goal_encoder_sha256": tensor_bundle_sha256(groups["goal"]),
            "worker_recurrent_sha256": tensor_bundle_sha256(
                groups["worker_recurrent"]
            ),
            "updates": int(value.get("updates", 0)),
        },
    )


def load_initial_predictor(path, expected_sha256, expected_identity):
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError("initial predictor file SHA-256 differs")
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("initial predictor metadata is missing")
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32).copy()
            for name in archive.files
            if name != "metadata"
        }
    if (
        metadata.get("format") != FORMAT
        or metadata.get("version") != 3
        or metadata.get("artifact_identity") != expected_identity
        or artifact_identity(metadata, arrays) != expected_identity
        or metadata.get("config")
        != {
            "context_dim": CONTEXT_DIM,
            "action_dim": ACTION_DIM,
            "latent_dim": 256,
            "output_dim": OUTPUT_DIM,
            "members": MEMBERS,
            "max_horizon": MAX_HORIZON,
            "interval_seconds": OBSERVATION_INTERVAL_SECONDS,
        }
    ):
        raise ValueError("initial predictor identity or contract differs")
    expected_arrays = {
        "context.mean",
        "context.scale",
        "action.mean",
        "action.scale",
        "target.mean",
        "target.scale",
    }
    for member_index in range(MEMBERS):
        expected_arrays.update(
            f"member.{member_index}.{name}"
            for name in RichRecurrentConsequenceEnsemble()
            .members[member_index]
            .state_dict()
        )
    if set(arrays) != expected_arrays:
        raise ValueError("initial predictor tensor set differs")
    return metadata, arrays


def inheritance_probe(cache_path, chunks, arrays, new_norm):
    old_context_mean = arrays["context.mean"]
    old_context_scale = arrays["context.scale"]
    old_action_mean = arrays["action.mean"]
    old_action_scale = arrays["action.scale"]
    new_context_mean, new_context_scale, new_action_mean, new_action_scale = new_norm[:4]
    for chunk_index, chunk in enumerate(chunks):
        context, actions, _ = chunk_arrays(cache_path, chunk)
        for start in range(0, len(context), 4096):
            stop = min(start + 4096, len(context))
            context_rows = np.asarray(context[start:stop])
            action_rows = np.asarray(actions[start:stop])
            valid = np.max(
                np.abs((context_rows - old_context_mean) / old_context_scale), axis=1
            ) < 7.5
            valid &= np.max(
                np.abs((context_rows - new_context_mean) / new_context_scale), axis=1
            ) < 7.5
            valid &= np.max(
                np.abs((action_rows - old_action_mean) / old_action_scale),
                axis=(1, 2),
            ) < 7.5
            valid &= np.max(
                np.abs((action_rows - new_action_mean) / new_action_scale),
                axis=(1, 2),
            ) < 7.5
            selected = np.flatnonzero(valid)[:8]
            if len(selected):
                return (
                    np.ascontiguousarray(context_rows[selected]),
                    np.ascontiguousarray(action_rows[selected]),
                    {
                        "cache_chunk": chunk_index,
                        "cache_row_start": start,
                        "selected_rows": selected.astype(int).tolist(),
                    },
                )
    raise ValueError("no unclipped inherited-normalization comparison rows")


@torch.no_grad()
def inherit_predictor(
    ensemble,
    arrays,
    new_norm,
    device,
    probe_context,
    probe_actions,
    probe_receipt,
):
    for member_index, member in enumerate(ensemble.members):
        state = {
            name: torch.as_tensor(
                arrays[f"member.{member_index}.{name}"], device=device
            )
            for name in member.state_dict()
        }
        member.load_state_dict(state, strict=True)
    (
        old_context_mean,
        old_context_scale,
        old_action_mean,
        old_action_scale,
        old_target_mean,
        old_target_scale,
    ) = [
        torch.as_tensor(arrays[name], device=device)
        for name in (
            "context.mean",
            "context.scale",
            "action.mean",
            "action.scale",
            "target.mean",
            "target.scale",
        )
    ]
    (
        new_context_mean,
        new_context_scale,
        new_action_mean,
        new_action_scale,
        new_target_mean,
        new_target_scale,
    ) = [torch.as_tensor(value, device=device) for value in new_norm]
    context_ratio = new_context_scale / old_context_scale
    context_offset = (new_context_mean - old_context_mean) / old_context_scale
    action_ratio = new_action_scale / old_action_scale
    action_offset = (new_action_mean - old_action_mean) / old_action_scale
    target_ratio = old_target_scale / new_target_scale
    target_offset = (old_target_mean - new_target_mean) / new_target_scale
    raw_context = torch.as_tensor(probe_context, device=device)
    raw_actions = torch.as_tensor(probe_actions, device=device)
    old_context, old_context_clipped = normalize_context(
        raw_context, old_context_mean, old_context_scale
    )
    old_actions, old_action_clipped = normalize_actions(
        raw_actions, old_action_mean, old_action_scale
    )
    if old_context_clipped.any() or old_action_clipped.any():
        raise ValueError("inherited predictor comparison unexpectedly clipped")
    old_raw_prediction = denormalize_deltas(
        ensemble(old_context, old_actions), old_target_mean, old_target_scale
    )
    for member in ensemble.members:
        old_weight = member.context.weight.detach().clone()
        member.context.weight.copy_(old_weight * context_ratio[None])
        member.context.bias.add_(old_weight @ context_offset)
        old_weight = member.transition.weight_ih.detach().clone()
        member.transition.weight_ih.copy_(old_weight * action_ratio[None])
        member.transition.bias_ih.add_(old_weight @ action_offset)
        old_weight = member.output.weight.detach().clone()
        old_bias = member.output.bias.detach().clone()
        member.output.weight.copy_(old_weight * target_ratio[:, None])
        member.output.bias.copy_(old_bias * target_ratio + target_offset)
    new_context, new_context_clipped = normalize_context(
        raw_context, new_context_mean, new_context_scale
    )
    new_actions, new_action_clipped = normalize_actions(
        raw_actions, new_action_mean, new_action_scale
    )
    if new_context_clipped.any() or new_action_clipped.any():
        raise ValueError("rebased predictor comparison unexpectedly clipped")
    new_raw_prediction = denormalize_deltas(
        ensemble(new_context, new_actions), new_target_mean, new_target_scale
    )
    maximum_absolute_error = float(
        (new_raw_prediction - old_raw_prediction).abs().max()
    )
    if maximum_absolute_error > 2e-5:
        raise ValueError("inherited predictor normalization rebase differs")
    return {
        "normalization_rebase": (
            "analytic affine rebase preserves the inherited raw predictor away "
            "from old/new +/-8 clipping boundaries"
        ),
        "raw_forecast_comparison": {
            **probe_receipt,
            "rows": len(probe_context),
            "maximum_absolute_error": maximum_absolute_error,
            "tolerance": 2e-5,
        },
    }


ROW_CACHE_FORMAT = "chreatures-rich-predictor-row-cache-v1"


@dataclass(frozen=True)
class CausalTail:
    start_tick: int
    codes: np.ndarray
    physiology: np.ndarray
    neural: np.ndarray
    reset: np.ndarray
    actions: np.ndarray
    worker: np.ndarray


class RunningMoments:
    def __init__(self, width):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, value):
        rows = np.asarray(value).reshape(-1, self.mean.size)
        for start in range(0, len(rows), 4096):
            batch = rows[start : start + 4096]
            batch_mean = batch.mean(0, dtype=np.float64)
            centered = batch.astype(np.float64) - batch_mean
            batch_m2 = np.sum(centered * centered, axis=0)
            total = self.count + len(batch)
            delta = batch_mean - self.mean
            self.mean += delta * len(batch) / total
            self.m2 += (
                batch_m2
                + delta * delta * self.count * len(batch) / total
            )
            self.count = total

    def finish(self, floor):
        if not self.count:
            raise ValueError("cannot normalize an empty training split")
        mean = self.mean.astype(np.float32)
        scale = np.sqrt(self.m2 / self.count).astype(np.float32)
        return mean, np.maximum(scale, np.asarray(floor, dtype=np.float32))


def atomic_npy(path, value):
    tmp = path.with_name("." + path.name + f".tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "shape": list(value.shape),
        "dtype": "<f4",
    }


@torch.inference_mode()
def encode_observations(observation, model, normalizer, device, batch_size):
    shape = observation.shape[:2]
    flat = observation.reshape(-1, observation.shape[-1])
    codes = np.empty((len(flat), FRAME_CODE_DIM), dtype=np.float32)
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        batch = torch.as_tensor(
            normalizer.normalize(flat[start:stop]), device=device
        )
        codes[start:stop] = model.encode_frames(batch).cpu().numpy()
    return codes.reshape(*shape, FRAME_CODE_DIM)


def predictor_rows(
    codes,
    physiology,
    actions,
    reset,
    neural,
    worker,
    columns,
    local_start,
    local_stop,
):
    times = np.arange(local_start, local_stop, dtype=np.int64)
    valid = np.ones((len(times), len(columns)), dtype=bool)
    # A reset on the first history frame begins a valid causal window. Any reset
    # after that frame would mix distinct lives and invalidates the row.
    for off in range(-2, MAX_HORIZON + 1):
        valid &= ~reset[times[:, None] + off, columns[None, :]]
    tr, cr = np.nonzero(valid)
    ti = times[tr]
    ci = columns[cr]
    frames = np.concatenate([codes[ti + off, ci] for off in range(-3, 1)], 1)
    context = np.concatenate(
        (
            frames,
            worker[ti, ci],
            neural[ti, ci],
            physiology[ti, ci],
            actions[ti - 1, ci],
        ),
        1,
    ).astype(np.float32, copy=False)
    action_rows = np.stack(
        [actions[ti + horizon, ci] for horizon in range(MAX_HORIZON)], 1
    ).astype(np.float32, copy=False)
    target = np.stack(
        [
            np.concatenate(
                (
                    codes[ti + horizon + 1, ci] - codes[ti + horizon, ci],
                    physiology[ti + horizon + 1, ci]
                    - physiology[ti + horizon, ci],
                ),
                1,
            )
            for horizon in range(MAX_HORIZON)
        ],
        1,
    ).astype(np.float32, copy=False)
    if (
        context.shape[1:] != (CONTEXT_DIM,)
        or action_rows.shape[1:] != (MAX_HORIZON, ACTION_DIM)
        or target.shape[1:] != (MAX_HORIZON, OUTPUT_DIM)
        or not np.isfinite(context).all()
        or not np.isfinite(action_rows).all()
        or not np.isfinite(target).all()
    ):
        raise RuntimeError("constructed recurrent consequence rows differ")
    return context, action_rows, target


def cache_contract(dataset, representation, normalizer):
    return {
        "format": ROW_CACHE_FORMAT,
        "version": 1,
        "dataset_manifest_file_sha256": dataset.manifest_file_sha256,
        "packet_sha256s": [packet.sha256 for packet in dataset.packets],
        "representation_file_sha256": representation["file_sha256"],
        "representation_normalizer_sha256": normalizer.to_value()["sha256"],
        "context_dim": CONTEXT_DIM,
        "action_dim": ACTION_DIM,
        "output_dim": OUTPUT_DIM,
        "horizon": MAX_HORIZON,
        "source_sha256": {
            "trainer": sha256(Path(__file__).resolve()),
            "data_boundary": sha256(
                ROOT / "research/sensorimotor_skills/rich_data.py"
            ),
        },
    }


def verify_cache(cache_path, contract):
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    body = copy.deepcopy(manifest)
    content_sha256 = body.pop("content_sha256", None)
    if content_sha256 != canonical_sha256(body) or any(
        manifest.get(key) != value for key, value in contract.items()
    ):
        raise ValueError("predictor row cache identity differs")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, dict) or set(chunks) != {
        "train",
        "validation",
        "heldout",
    }:
        raise ValueError("predictor row cache splits differ")
    for values in chunks.values():
        if not isinstance(values, list) or not values:
            raise ValueError("predictor row cache split is empty")
        for chunk in values:
            for key, expected_shape in (
                ("context", (int(chunk["rows"]), CONTEXT_DIM)),
                ("actions", (int(chunk["rows"]), MAX_HORIZON, ACTION_DIM)),
                ("target", (int(chunk["rows"]), MAX_HORIZON, OUTPUT_DIM)),
            ):
                receipt = chunk[key]
                path = (cache_path / receipt["path"]).resolve()
                if (
                    not path.is_relative_to(cache_path.resolve())
                    or not path.is_file()
                    or path.stat().st_size != int(receipt["bytes"])
                    or sha256(path) != receipt["sha256"]
                    or tuple(receipt["shape"]) != expected_shape
                    or receipt["dtype"] != "<f4"
                ):
                    raise ValueError("predictor row cache receipt differs")
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if value.shape != expected_shape or value.dtype != np.dtype("<f4"):
                    raise ValueError("predictor row cache array differs")
    return manifest


def prepare_cache(
    cache_path,
    dataset,
    representation_model,
    normalizer,
    representation,
    split,
    device,
    encoding_batch_size,
):
    contract = cache_contract(dataset, representation, normalizer)
    cached = verify_cache(cache_path, contract) if cache_path.exists() else None
    if cached is not None:
        if cached.get("splits") != {
            name: list(worlds) for name, worlds in split.items()
        }:
            raise ValueError("predictor row cache partition differs")
        return cached
    if cache_path.exists() and any(cache_path.iterdir()):
        raise ValueError("incomplete predictor row cache must be empty")
    cache_path.mkdir(parents=True, exist_ok=True)
    columns = {name: dataset.columns(worlds) for name, worlds in split.items()}
    chunks = {name: [] for name in split}
    tail = None
    active_episode = None
    next_center = None
    sequence = 0
    for episode in dataset.iter_contiguous_shards():
        if episode.neural is None:
            raise ValueError("current predictor requires neural384 rows")
        if episode.episode != active_episode:
            active_episode = episode.episode
            tail = None
            next_center = 3
        codes = encode_observations(
            episode.observation,
            representation_model,
            normalizer,
            device,
            encoding_batch_size,
        )
        current_physiology = episode.observation[..., -PHYSIOLOGY_DIM:]
        if tail is None:
            base_tick = episode.start_tick
            all_codes = codes
            physiology = current_physiology
            neural = episode.neural
            reset = episode.reset
            actions = episode.actions
            worker = episode.worker_recurrent_context
        else:
            base_tick = tail.start_tick
            all_codes = np.concatenate((tail.codes, codes[1:]), axis=0)
            physiology = np.concatenate(
                (tail.physiology, current_physiology[1:]), axis=0
            )
            neural = np.concatenate((tail.neural, episode.neural[1:]), axis=0)
            reset = np.concatenate((tail.reset, episode.reset[1:]), axis=0)
            actions = np.concatenate((tail.actions, episode.actions), axis=0)
            worker = np.concatenate(
                (tail.worker, episode.worker_recurrent_context), axis=0
            )
        stop_center = episode.stop_tick - MAX_HORIZON + 1
        if next_center < stop_center:
            local_start = next_center - base_tick
            local_stop = stop_center - base_tick
            for name in split:
                context, action_rows, target = predictor_rows(
                    all_codes,
                    physiology,
                    actions,
                    reset,
                    neural,
                    worker,
                    columns[name],
                    local_start,
                    local_stop,
                )
                prefix = f"chunk-{sequence:03d}-{name}"
                chunk = {
                    "episode": episode.episode,
                    "through_shard": episode.shard,
                    "center_start_tick": next_center,
                    "center_stop_tick": stop_center,
                    "rows": len(context),
                    "context": atomic_npy(cache_path / f"{prefix}-context.npy", context),
                    "actions": atomic_npy(
                        cache_path / f"{prefix}-actions.npy", action_rows
                    ),
                    "target": atomic_npy(cache_path / f"{prefix}-target.npy", target),
                }
                expected_rows = (stop_center - next_center) * len(columns[name])
                if len(context) != expected_rows:
                    raise ValueError("predictor causal row count differs")
                chunks[name].append(chunk)
                del context, action_rows, target
            sequence += 1
            next_center = stop_center
        keep_tick = max(episode.start_tick, episode.stop_tick - MAX_HORIZON - 2)
        keep_observation = keep_tick - base_tick
        keep_action = keep_tick - base_tick
        tail = CausalTail(
            keep_tick,
            all_codes[keep_observation:].copy(),
            physiology[keep_observation:].copy(),
            neural[keep_observation:].copy(),
            reset[keep_observation:].copy(),
            actions[keep_action:].copy(),
            worker[keep_action:].copy(),
        )
        del all_codes, physiology, neural, reset, actions, worker, codes
        del current_physiology, episode
    totals = {
        name: sum(int(chunk["rows"]) for chunk in values)
        for name, values in chunks.items()
    }
    expected_centers = int(dataset.manifest_scope["episodes"]) * (
        dataset.steps_per_episode - MAX_HORIZON - 2
    )
    for name, worlds in split.items():
        expected = expected_centers * len(worlds) * dataset.residents_per_world
        if totals[name] != expected:
            raise ValueError("predictor complete episode row count differs")
    body = contract | {
        "splits": {name: list(worlds) for name, worlds in split.items()},
        "rows": totals,
        "shuffle": "random shard order and random contiguous minibatch-block order",
        "causal_boundary": "four actual frame codes ending at t; actual delivered actions t..t+7; exact shard seams checked",
        "chunks": chunks,
    }
    manifest = body | {"content_sha256": canonical_sha256(body)}
    atomic_json(cache_path / "manifest.json", manifest)
    return manifest


def chunk_arrays(cache_path, chunk):
    return tuple(
        np.load(cache_path / chunk[key]["path"], mmap_mode="r", allow_pickle=False)
        for key in ("context", "actions", "target")
    )


def cache_moments(cache_path, chunks):
    context = RunningMoments(CONTEXT_DIM)
    actions = RunningMoments(ACTION_DIM)
    target = RunningMoments(OUTPUT_DIM)
    for chunk in chunks:
        x, a, y = chunk_arrays(cache_path, chunk)
        context.update(x)
        actions.update(a)
        target.update(y)
    context_mean, context_scale = context.finish(CONTEXT_SCALE_FLOOR)
    action_mean, action_scale = actions.finish(ACTION_SCALE_FLOOR)
    floor = np.r_[
        np.full(FRAME_CODE_DIM, CODE_DELTA_SCALE_FLOOR),
        np.full(PHYSIOLOGY_DIM, PHYSIOLOGY_DELTA_SCALE_FLOOR),
    ]
    target_mean, target_scale = target.finish(floor)
    return (
        context_mean,
        context_scale,
        action_mean,
        action_scale,
        target_mean,
        target_scale,
    )


def train_ensemble(ensemble, cache_path, chunks, norm, optimizer, cfg, device):
    rng = np.random.default_rng(cfg.seed)
    cm, cs, am, action_scale, ym, ys = [
        torch.as_tensor(value, device=device) for value in norm
    ]
    traces = [[] for _ in range(MEMBERS)]
    for epoch in range(cfg.epochs):
        totals = np.zeros(MEMBERS, dtype=np.float64)
        count = 0
        for chunk_index in rng.permutation(len(chunks)):
            x, a, y = chunk_arrays(cache_path, chunks[int(chunk_index)])
            starts = np.arange(0, len(x), cfg.batch_size)
            for start in starts[rng.permutation(len(starts))]:
                stop = min(int(start) + cfg.batch_size, len(x))
                raw_context = torch.tensor(x[start:stop], device=device)
                raw_actions = torch.tensor(a[start:stop], device=device)
                raw_target = torch.tensor(y[start:stop], device=device)
                context, _ = normalize_context(raw_context, cm, cs)
                action, _ = normalize_actions(raw_actions, am, action_scale)
                target = ((raw_target - ym) / ys)[:, None].expand(
                    -1, MEMBERS, -1, -1
                )
                prediction = ensemble(context, action)
                visual = torch.nn.functional.smooth_l1_loss(
                    prediction[..., :128], target[..., :128], reduction="none"
                ).mean(dim=(0, 2, 3))
                body = torch.nn.functional.smooth_l1_loss(
                    prediction[..., 128:256], target[..., 128:256], reduction="none"
                ).mean(dim=(0, 2, 3))
                proposed_physiology = prediction[..., 256:] * ys[256:] + ym[256:]
                decoded_physiology = bounded_physiology_deltas(
                    proposed_physiology,
                    raw_context[:, None, 1536:1548],
                )
                physiology_error = (
                    decoded_physiology
                    - raw_target[:, None, :, 256:].expand(-1, MEMBERS, -1, -1)
                ) / ys[256:]
                physiology = torch.nn.functional.smooth_l1_loss(
                    physiology_error,
                    torch.zeros_like(physiology_error),
                    reduction="none",
                ).mean(dim=(0, 2, 3))
                cumulative = torch.nn.functional.smooth_l1_loss(
                    physiology_error.cumsum(2),
                    torch.zeros_like(physiology_error),
                    reduction="none",
                ).mean(dim=(0, 2, 3))
                member_loss = (visual + body + physiology) / 3 + 0.05 * cumulative
                loss = member_loss.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite recurrent consequence loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for member in ensemble.members:
                    torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                optimizer.step()
                batch_rows = stop - int(start)
                totals += member_loss.detach().cpu().numpy() * batch_rows
                count += batch_rows
        for member_index in range(MEMBERS):
            record = {"epoch": epoch + 1, "loss": float(totals[member_index] / count)}
            traces[member_index].append(record)
        print(
            json.dumps(
                {
                    "training": {
                        "epoch": epoch + 1,
                        "member_loss": [trace[-1]["loss"] for trace in traces],
                    }
                }
            ),
            flush=True,
        )
    return traces


class MetricAccumulator:
    GROUPS = (("visual", 0, 128), ("body", 128, 256), ("physiology", 256, 268))

    def __init__(self, goal=False):
        self.squared = np.zeros((MAX_HORIZON, len(self.GROUPS)), dtype=np.float64)
        self.absolute = np.zeros_like(self.squared)
        self.elements = np.zeros_like(self.squared)
        self.clipped = 0
        self.clip_elements = 0
        self.rows = 0
        self.goal_squared = np.zeros(MAX_HORIZON, dtype=np.float64) if goal else None
        self.persistence_squared = (
            np.zeros(MAX_HORIZON, dtype=np.float64) if goal else None
        )

    def metrics(self):
        result = {}
        for horizon in range(MAX_HORIZON):
            result[str(horizon + 1)] = {
                name: {
                    "rmse": float(
                        np.sqrt(
                            self.squared[horizon, index]
                            / self.elements[horizon, index]
                        )
                    ),
                    "mae": float(
                        self.absolute[horizon, index]
                        / self.elements[horizon, index]
                    ),
                }
                for index, (name, _, _) in enumerate(self.GROUPS)
            }
        return result

    def goal_calibration(self):
        denominator = self.rows * 64
        return {
            "empirical_goal_rms_by_horizon": np.sqrt(
                self.goal_squared / denominator
            ).tolist(),
            "persistence_goal_rms_by_horizon": np.sqrt(
                self.persistence_squared / denominator
            ).tolist(),
            "scope": "descriptive reserved-world error; not calibrated probability or causal evidence",
        }


@torch.inference_mode()
def evaluate_stream(
    ensemble,
    representation,
    cache_path,
    chunks,
    norm,
    batch_size,
    device,
    *,
    goals,
):
    accumulator = MetricAccumulator(goal=goals)
    cm, cs, am, action_scale, ym, ys = [
        torch.as_tensor(value, device=device) for value in norm
    ]
    for chunk in chunks:
        x, a, y = chunk_arrays(cache_path, chunk)
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            raw_context = torch.tensor(x[start:stop], device=device)
            raw_actions = torch.tensor(a[start:stop], device=device)
            target = torch.tensor(y[start:stop], device=device)
            context, context_clipped = normalize_context(raw_context, cm, cs)
            action, action_clipped = normalize_actions(
                raw_actions, am, action_scale
            )
            members = denormalize_deltas(ensemble(context, action), ym, ys)
            members[..., 256:] = bounded_physiology_deltas(
                members[..., 256:], raw_context[:, None, 1536:1548]
            )
            mean = members.mean(1)
            error = mean - target
            for group_index, (_, group_start, group_stop) in enumerate(
                accumulator.GROUPS
            ):
                group = error[..., group_start:group_stop]
                accumulator.squared[:, group_index] += (
                    group.square().sum(dim=(0, 2)).double().cpu().numpy()
                )
                accumulator.absolute[:, group_index] += (
                    group.abs().sum(dim=(0, 2)).double().cpu().numpy()
                )
                accumulator.elements[:, group_index] += (
                    len(group) * (group_stop - group_start)
                )
            clipped = context_clipped[:, None] | action_clipped
            accumulator.clipped += int(clipped.sum())
            accumulator.clip_elements += clipped.numel()
            accumulator.rows += len(raw_context)
            if goals:
                base = raw_context[:, : 4 * FRAME_CODE_DIM].reshape(
                    -1, 4, FRAME_CODE_DIM
                )
                anchor = base[:, -1]
                predicted_future = anchor[:, None] + mean[..., :FRAME_CODE_DIM].cumsum(1)
                actual_future = anchor[:, None] + target[..., :FRAME_CODE_DIM].cumsum(1)
                for horizon in range(MAX_HORIZON):
                    predicted_window = torch.cat(
                        (base, predicted_future[:, : horizon + 1]), dim=1
                    )[:, -4:]
                    actual_window = torch.cat(
                        (base, actual_future[:, : horizon + 1]), dim=1
                    )[:, -4:]
                    persistence_window = torch.cat(
                        (base, base[:, -1:].expand(-1, horizon + 1, -1)), dim=1
                    )[:, -4:]
                    actual_goal = representation.goal_encoder(actual_window.flatten(1))
                    predicted_goal = representation.goal_encoder(
                        predicted_window.flatten(1)
                    )
                    persistence_goal = representation.goal_encoder(
                        persistence_window.flatten(1)
                    )
                    accumulator.goal_squared[horizon] += float(
                        (predicted_goal - actual_goal).square().sum()
                    )
                    accumulator.persistence_squared[horizon] += float(
                        (persistence_goal - actual_goal).square().sum()
                    )
    return accumulator


def main():
    cfg = args()
    if not cfg.trusted_checkpoint:
        raise SystemExit(
            "--trusted-checkpoint is required for Torch representation state"
        )
    if cfg.output.exists() and any(cfg.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    if (
        not 1 <= cfg.epochs <= 100
        or not 32 <= cfg.batch_size <= 8192
        or not 64 <= cfg.encoding_batch_size <= 8192
    ):
        raise SystemExit("invalid schedule")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA/ROCm device is unavailable")
    dataset = RichPlayDataset(cfg.dataset)
    collection_identity = dataset.manifest["collection_identity"]
    collected_resident = collection_identity["resident_artifact"]
    if (
        collection_identity["source"]["revision"]
        != cfg.expected_collection_source_revision
        or collected_resident["file_sha256"]
        != cfg.expected_resident_artifact_sha256
        or collected_resident["artifact_sha256"]
        != cfg.expected_resident_artifact_identity
    ):
        raise ValueError("collection source or resident identity differs")
    validation_worlds = tuple(sorted(set(cfg.validation_worlds)))
    heldout_worlds = tuple(sorted(set(cfg.heldout_worlds)))
    train_worlds = tuple(
        world
        for world in range(dataset.world_count)
        if world not in {*validation_worlds, *heldout_worlds}
    )
    if (
        not train_worlds
        or not validation_worlds
        or not heldout_worlds
        or set(validation_worlds) & set(heldout_worlds)
        or min(validation_worlds) < 0
        or max(validation_worlds) >= dataset.world_count
        or min(heldout_worlds) < 0
        or max(heldout_worlds) >= dataset.world_count
    ):
        raise ValueError("explicit whole-world train/validation split is invalid")
    declared = dataset.manifest["scope"]
    if (
        tuple(declared.get("train_world_slots", ())) != train_worlds
        or tuple(declared.get("validation_world_slots", ())) != validation_worlds
        or tuple(declared.get("heldout_world_slots", ())) != heldout_worlds
    ):
        raise ValueError("trainer split differs from immutable collection split")
    model, obsnorm, rep_identity, rep = load_representation(
        cfg.representation_checkpoint, device
    )
    if rep["file_sha256"] != cfg.expected_representation_sha256:
        raise ValueError("representation checkpoint SHA-256 differs")
    if dataset.dt_seconds != OBSERVATION_INTERVAL_SECONDS:
        raise ValueError("current predictor interval differs")
    split = {
        "train": train_worlds,
        "validation": validation_worlds,
        "heldout": heldout_worlds,
    }
    preparation_started = time.monotonic()
    cache = prepare_cache(
        cfg.row_cache.resolve(),
        dataset,
        model,
        obsnorm,
        rep,
        split,
        device,
        cfg.encoding_batch_size,
    )
    preparation_seconds = time.monotonic() - preparation_started
    norm = cache_moments(cfg.row_cache.resolve(), cache["chunks"]["train"])
    (
        context_mean,
        context_scale,
        action_mean,
        action_scale,
        target_mean,
        target_scale,
    ) = norm
    initial_metadata, initial_arrays = load_initial_predictor(
        cfg.initial_predictor,
        cfg.expected_initial_predictor_sha256,
        cfg.expected_initial_predictor_identity,
    )
    probe_context, probe_actions, probe_receipt = inheritance_probe(
        cfg.row_cache.resolve(),
        cache["chunks"]["train"],
        initial_arrays,
        norm,
    )
    ensemble = RichRecurrentConsequenceEnsemble().to(device)
    inheritance = inherit_predictor(
        ensemble,
        initial_arrays,
        norm,
        device,
        probe_context,
        probe_actions,
        probe_receipt,
    )
    del initial_arrays
    optimizer = torch.optim.AdamW(
        ensemble.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    traces = train_ensemble(
        ensemble,
        cfg.row_cache.resolve(),
        cache["chunks"]["train"],
        norm,
        optimizer,
        cfg,
        device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.monotonic() - started
    train_monitor = evaluate_stream(
        ensemble,
        model,
        cfg.row_cache.resolve(),
        cache["chunks"]["train"],
        norm,
        cfg.batch_size,
        device,
        goals=False,
    )
    validation = evaluate_stream(
        ensemble,
        model,
        cfg.row_cache.resolve(),
        cache["chunks"]["validation"],
        norm,
        cfg.batch_size,
        device,
        goals=True,
    )
    heldout = evaluate_stream(
        ensemble,
        model,
        cfg.row_cache.resolve(),
        cache["chunks"]["heldout"],
        norm,
        cfg.batch_size,
        device,
        goals=True,
    )
    arrays = {
        "context.mean": context_mean,
        "context.scale": context_scale,
        "action.mean": action_mean,
        "action.scale": action_scale,
        "target.mean": target_mean,
        "target.scale": target_scale,
    }
    for mi, member in enumerate(ensemble.members):
        for name, tensor in member.state_dict().items():
            arrays[f"member.{mi}.{name}"] = np.ascontiguousarray(
                tensor.detach().cpu().numpy(), dtype=np.float32
            )
    metadata = {
        "format": FORMAT,
        "version": 3,
        "config": asdict(ensemble.config),
        "temporal_contract": {
            "observation_interval_seconds": OBSERVATION_INTERVAL_SECONDS,
            "horizons_ticks": list(range(1, MAX_HORIZON + 1)),
            "alignment": "delivered_action[t+j] predicts code and physiology deltas t+j to t+j+1",
        },
        "input_contract": {
            "context_dim": CONTEXT_DIM,
            "context_segments": CONTEXT_SEGMENTS,
            "worker_recurrent_context": "native state plus recurrent_adapter after current observation and policy adapter update",
            "actions": ["B", "K", "H", ACTION_DIM],
            "action_names": list(ACTION_NAMES),
            "physiology_names": list(PHYSIOLOGY_NAMES),
            "normalization": "train-worlds-only; clamp +/-8",
        },
        "output_contract": {
            "member_delta": ["B", "K", MEMBERS, "H", OUTPUT_DIM],
            "absolute_physiology": ["B", "K", MEMBERS, "H", PHYSIOLOGY_DIM],
            "ensemble_spread": "population RMS; uncalibrated",
            "physiology_link": {
                "proposal": "target-denormalized output head",
                "formula": "up*tanh(qplus/max(up,1e-4))-down*tanh(qminus/max(down,1e-4))",
                "signed_split": "stable qplus/qminus from sqrt(q*q+1e-8), epsilon=1e-4",
                "anchor": "actual physiology at t, then member-private predicted state",
                "epsilon": PHYSIOLOGY_LINK_EPSILON,
                "lower": list(PHYSIOLOGY_LOWER),
                "upper": list(PHYSIOLOGY_UPPER),
                "clipping": False,
            },
        },
        "representation": rep | {"identity": rep_identity},
        "inherited_predictor": {
            "file_sha256": cfg.expected_initial_predictor_sha256,
            "artifact_identity": cfg.expected_initial_predictor_identity,
            "source_training_data": copy.deepcopy(
                initial_metadata.get("training_data")
            ),
            **inheritance,
        },
        "training_data": {
            "manifest_file_sha256": dataset.manifest_file_sha256,
            "collection_identity_sha256": dataset.manifest[
                "collection_identity_sha256"
            ],
            "resident_artifact": copy.deepcopy(
                dataset.manifest["collection_identity"]["resident_artifact"]
            ),
            "packet_sha256s": [packet.sha256 for packet in dataset.packets],
            "train_worlds": list(train_worlds),
            "validation_worlds": list(validation_worlds),
            "heldout_worlds": list(heldout_worlds),
            "train_rows": cache["rows"]["train"],
            "validation_rows": cache["rows"]["validation"],
            "heldout_rows": cache["rows"]["heldout"],
            "row_cache_content_sha256": cache["content_sha256"],
            "causal_boundary": cache["causal_boundary"],
        },
        "training": {
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "encoding_batch_size": cfg.encoding_batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "objective": "equal visual/body/physiology Huber plus 0.05 cumulative physiology Huber; three independent members updated together",
            "shuffle": cache["shuffle"],
            "row_cache_preparation_seconds": preparation_seconds,
            "seconds": training_seconds,
            "traces": traces,
        },
        "validation": {
            "metrics": validation.metrics(),
            "goal_calibration": validation.goal_calibration(),
            "input_clipped_fraction": validation.clipped / validation.clip_elements,
        },
        "heldout_once": {
            "metrics": heldout.metrics(),
            "goal_calibration": heldout.goal_calibration(),
            "input_clipped_fraction": heldout.clipped / heldout.clip_elements,
        },
        "train_monitor": train_monitor.metrics(),
        "pack_order": list(arrays),
        "tensors": {
            n: {
                "shape": list(v.shape),
                "dtype": v.dtype.str,
                "sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            }
            for n, v in arrays.items()
        },
    }
    metadata["artifact_identity"] = artifact_identity(metadata, arrays)
    cfg.output.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        cfg.output / "rich-recurrent-consequence-v3.npz",
        {"metadata": np.asarray(json.dumps(metadata, sort_keys=True)), **arrays},
    )
    result = {
        "format": "chreatures-rich-recurrent-consequence-fit-v3",
        "artifact_identity": metadata["artifact_identity"],
        "artifact_file_sha256": sha256(
            cfg.output / "rich-recurrent-consequence-v3.npz"
        ),
        "validation": metadata["validation"],
        "training_seconds": metadata["training"]["seconds"],
    }
    atomic_json(cfg.output / "result.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
