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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.organism_interface import ACTION_NAMES, PHYSIOLOGY_NAMES
from chreatures.resident_contract import BOOTSTRAP_FORMAT, DEVELOPMENT_FORMAT
from research.sensorimotor_skills.rich_data import RichNormalizer, RichPlayDataset
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trusted-checkpoint", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--device", default="cuda")
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
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get("format") not in (BOOTSTRAP_FORMAT, DEVELOPMENT_FORMAT):
        raise ValueError("representation must be a current v5 checkpoint")
    identity = copy.deepcopy(value["identity"])
    model = RichSensorimotorModel().to(device)
    model.load_state_dict(value["model"], strict=True)
    model.eval().requires_grad_(False)
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


@torch.inference_mode()
def encode_episode(ep, model, normalizer, device):
    t, n = ep.observation.shape[:2]
    codes = np.empty((t, n, FRAME_CODE_DIM), np.float32)
    for start in range(0, t, 64):
        stop = min(start + 64, t)
        obs = torch.as_tensor(
            normalizer.normalize(ep.observation[start:stop]), device=device
        )
        codes[start:stop] = model.encode_frames(obs).cpu().numpy()
    return codes


def rows(ep, codes, columns):
    times = np.arange(3, len(ep.actions) - MAX_HORIZON + 1, dtype=np.int64)
    valid = np.ones((len(times), len(columns)), bool)
    for off in range(-3, MAX_HORIZON + 1):
        valid &= ~ep.reset[times[:, None] + off, columns[None, :]]
    tr, cr = np.nonzero(valid)
    ti = times[tr]
    ci = columns[cr]
    frames = np.concatenate([codes[ti + off, ci] for off in range(-3, 1)], 1)
    phys = ep.observation[..., -PHYSIOLOGY_DIM:]
    context = np.concatenate(
        (
            frames,
            ep.worker_recurrent_context[ti, ci],
            ep.neural[ti, ci],
            phys[ti, ci],
            ep.previous[ti, ci],
        ),
        1,
    ).astype(np.float32)
    actions = np.stack([ep.actions[ti + h, ci] for h in range(MAX_HORIZON)], 1).astype(
        np.float32
    )
    target = np.stack(
        [
            np.concatenate(
                (
                    codes[ti + h + 1, ci] - codes[ti + h, ci],
                    phys[ti + h + 1, ci] - phys[ti + h, ci],
                ),
                1,
            )
            for h in range(MAX_HORIZON)
        ],
        1,
    ).astype(np.float32)
    if (
        context.shape[1:] != (CONTEXT_DIM,)
        or actions.shape[1:] != (MAX_HORIZON, ACTION_DIM)
        or target.shape[1:] != (MAX_HORIZON, OUTPUT_DIM)
    ):
        raise RuntimeError("constructed recurrent consequence rows differ")
    return context, actions, target


def moments(value, floor, axes):
    mean = value.mean(axis=axes, dtype=np.float64).astype(np.float32)
    scale = value.std(axis=axes, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, np.asarray(floor, np.float32))


def train_member(member, x, a, y, norm, opt, args, device, seed):
    rng = np.random.default_rng(seed)
    trace = []
    means = [torch.as_tensor(v, device=device) for v in norm]
    cm, cs, am, ass, ym, ys = means
    for epoch in range(args.epochs):
        order = rng.permutation(len(x))
        total = 0.0
        count = 0
        for start in range(0, len(order), args.batch_size):
            ix = order[start : start + args.batch_size]
            cx, _ = normalize_context(torch.as_tensor(x[ix], device=device), cm, cs)
            ax, _ = normalize_actions(torch.as_tensor(a[ix], device=device), am, ass)
            target = (torch.as_tensor(y[ix], device=device) - ym) / ys
            pred = member(cx, ax)
            visual = torch.nn.functional.smooth_l1_loss(
                pred[..., :128], target[..., :128]
            )
            body = torch.nn.functional.smooth_l1_loss(
                pred[..., 128:256], target[..., 128:256]
            )
            physiology_anchor = torch.as_tensor(
                x[ix, 1536:1548], device=device
            )
            proposed_physiology = pred[..., 256:] * ys[256:] + ym[256:]
            decoded_physiology = bounded_physiology_deltas(
                proposed_physiology, physiology_anchor
            )
            actual_physiology = torch.as_tensor(y[ix, :, 256:], device=device)
            physiology_error = (decoded_physiology - actual_physiology) / ys[256:]
            phys = torch.nn.functional.smooth_l1_loss(
                physiology_error, torch.zeros_like(physiology_error)
            )
            cumulative = torch.nn.functional.smooth_l1_loss(
                physiology_error.cumsum(1),
                torch.zeros_like(physiology_error),
            )
            loss = (visual + body + phys) / 3 + 0.05 * cumulative
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(ix)
            count += len(ix)
        record = {"epoch": epoch + 1, "loss": total / count}
        print(json.dumps({"training": record}), flush=True)
        trace.append(record)
    return trace


@torch.inference_mode()
def predict(model, x, a, norm, batch, device):
    outputs = np.empty((len(x), MEMBERS, MAX_HORIZON, OUTPUT_DIM), np.float32)
    clipped = np.empty((len(x), MAX_HORIZON), bool)
    cm, cs, am, ass, ym, ys = [torch.as_tensor(v, device=device) for v in norm]
    for start in range(0, len(x), batch):
        stop = min(start + batch, len(x))
        cx, cclip = normalize_context(
            torch.as_tensor(x[start:stop], device=device), cm, cs
        )
        ax, aclip = normalize_actions(
            torch.as_tensor(a[start:stop], device=device), am, ass
        )
        proposed = denormalize_deltas(model(cx, ax), ym, ys)
        proposed[..., 256:] = bounded_physiology_deltas(
            proposed[..., 256:],
            torch.as_tensor(x[start:stop, 1536:1548], device=device)[:, None, :],
        )
        outputs[start:stop] = proposed.cpu().numpy()
        clipped[start:stop] = np.logical_or(
            cclip.cpu().numpy()[:, None], aclip.cpu().numpy()
        )
    return outputs, clipped


def metrics(members, target):
    mean = members.mean(1)
    result = {}
    for h in range(MAX_HORIZON):
        groups = {}
        for name, s, e in [
            ("visual", 0, 128),
            ("body", 128, 256),
            ("physiology", 256, 268),
        ]:
            groups[name] = {
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (mean[:, h, s:e] - target[:, h, s:e]).astype(np.float64)
                            ** 2
                        )
                    )
                ),
                "mae": float(np.mean(np.abs(mean[:, h, s:e] - target[:, h, s:e]))),
            }
        result[str(h + 1)] = groups
    return result


@torch.inference_mode()
def goal_calibration(representation, context, members, target, batch_size, device):
    predicted_delta = members.mean(1)[..., :FRAME_CODE_DIM]
    actual_delta = target[..., :FRAME_CODE_DIM]
    predicted_future = context[:, None, 3 * FRAME_CODE_DIM : 4 * FRAME_CODE_DIM] + np.cumsum(
        predicted_delta, axis=1
    )
    actual_future = context[:, None, 3 * FRAME_CODE_DIM : 4 * FRAME_CODE_DIM] + np.cumsum(
        actual_delta, axis=1
    )
    history = context[:, : 4 * FRAME_CODE_DIM].reshape(-1, 4, FRAME_CODE_DIM)
    sums = np.zeros(MAX_HORIZON, dtype=np.float64)
    persistence = np.zeros(MAX_HORIZON, dtype=np.float64)
    for start in range(0, len(context), batch_size):
        stop = min(start + batch_size, len(context))
        base = torch.as_tensor(history[start:stop], device=device)
        for horizon in range(MAX_HORIZON):
            predicted_sequence = torch.as_tensor(
                predicted_future[start:stop, : horizon + 1], device=device
            )
            actual_sequence = torch.as_tensor(
                actual_future[start:stop, : horizon + 1], device=device
            )
            combined_predicted = torch.cat((base, predicted_sequence), dim=1)[:, -4:]
            combined_actual = torch.cat((base, actual_sequence), dim=1)[:, -4:]
            actual_goal = representation.goal_encoder(combined_actual.flatten(1))
            predicted_goal = representation.goal_encoder(combined_predicted.flatten(1))
            persistence_goal = representation.goal_encoder(
                torch.cat(
                    (
                        base,
                        base[:, -1:].expand(-1, horizon + 1, -1),
                    ),
                    dim=1,
                )[:, -4:].flatten(1)
            )
            sums[horizon] += float((predicted_goal - actual_goal).square().sum())
            persistence[horizon] += float((persistence_goal - actual_goal).square().sum())
    denominator = len(context) * 64
    return {
        "empirical_goal_rms_by_horizon": (np.sqrt(sums / denominator)).tolist(),
        "persistence_goal_rms_by_horizon": (np.sqrt(persistence / denominator)).tolist(),
        "scope": "descriptive reserved-world error; not calibrated probability or causal evidence",
    }


def main():
    cfg = args()
    if not cfg.trusted_checkpoint:
        raise SystemExit(
            "--trusted-checkpoint is required for Torch representation state"
        )
    if cfg.output.exists() and any(cfg.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    if not 1 <= cfg.epochs <= 100 or not 32 <= cfg.batch_size <= 8192:
        raise SystemExit("invalid schedule")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    dataset = RichPlayDataset(cfg.dataset)
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
    if dataset.dt_seconds != OBSERVATION_INTERVAL_SECONDS or any(
        ep.neural is None for ep in dataset.episodes
    ):
        raise ValueError("current predictor requires neural384 and 0.05 second rows")
    split = {
        "train": train_worlds,
        "validation": validation_worlds,
        "heldout": heldout_worlds,
    }
    chunks = {name: [] for name in split}
    columns = {name: dataset.columns(worlds) for name, worlds in split.items()}
    for episode in dataset.episodes:
        codes = encode_episode(episode, model, obsnorm, device)
        for name in split:
            chunks[name].append(rows(episode, codes, columns[name]))
    assembled = {
        name: tuple(
            np.concatenate([part[index] for part in values], 0)
            for index in range(3)
        )
        for name, values in chunks.items()
    }
    train_x, train_a, train_y = assembled["train"]
    val_x, val_a, val_y = assembled["validation"]
    heldout_x, heldout_a, heldout_y = assembled["heldout"]
    context_mean, context_scale = moments(train_x, CONTEXT_SCALE_FLOOR, (0,))
    action_mean, action_scale = moments(train_a, ACTION_SCALE_FLOOR, (0, 1))
    floor = np.r_[
        np.full(256, CODE_DELTA_SCALE_FLOOR),
        np.full(PHYSIOLOGY_DIM, PHYSIOLOGY_DELTA_SCALE_FLOOR),
    ]
    target_mean, target_scale = moments(train_y, floor, (0, 1))
    norm = (
        context_mean,
        context_scale,
        action_mean,
        action_scale,
        target_mean,
        target_scale,
    )
    ensemble = RichRecurrentConsequenceEnsemble().to(device)
    traces = []
    started = time.monotonic()
    for i, member in enumerate(ensemble.members):
        traces.append(
            train_member(
                member,
                train_x,
                train_a,
                train_y,
                norm,
                torch.optim.AdamW(
                    member.parameters(),
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                ),
                cfg,
                device,
                cfg.seed + 1009 * i,
            )
        )
    train_pred, _ = predict(ensemble, train_x, train_a, norm, cfg.batch_size, device)
    val_pred, val_clipped = predict(
        ensemble, val_x, val_a, norm, cfg.batch_size, device
    )
    heldout_pred, heldout_clipped = predict(
        ensemble, heldout_x, heldout_a, norm, cfg.batch_size, device
    )
    validation_goal = goal_calibration(
        model, val_x, val_pred, val_y, cfg.batch_size, device
    )
    heldout_goal = goal_calibration(
        model, heldout_x, heldout_pred, heldout_y, cfg.batch_size, device
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
        "training_data": {
            "manifest_file_sha256": dataset.manifest_file_sha256,
            "packet_sha256s": [ep.packet_sha256 for ep in dataset.episodes],
            "train_worlds": list(train_worlds),
            "validation_worlds": list(validation_worlds),
            "heldout_worlds": list(heldout_worlds),
            "train_rows": len(train_x),
            "validation_rows": len(val_x),
            "heldout_rows": len(heldout_x),
        },
        "training": {
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "seconds": time.monotonic() - started,
            "traces": traces,
        },
        "validation": {
            "metrics": metrics(val_pred, val_y),
            "goal_calibration": validation_goal,
            "input_clipped_fraction": float(val_clipped.mean()),
        },
        "heldout_once": {
            "metrics": metrics(heldout_pred, heldout_y),
            "goal_calibration": heldout_goal,
            "input_clipped_fraction": float(heldout_clipped.mean()),
        },
        "train_monitor": metrics(train_pred, train_y),
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
