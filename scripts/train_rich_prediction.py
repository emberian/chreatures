#!/usr/bin/env python3
"""Fit an ensemble that predicts rich sensory/body consequences of candidate actions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.sensorimotor_skills.rich_data import RichNormalizer, RichPlayDataset
from research.sensorimotor_skills.rich_model import RichSensorimotorModel
from research.sensorimotor_skills.rich_prediction import (
    ACTION_DIM,
    ACTION_SUFFIX_FORMAT,
    ACTION_SUFFIX_HORIZONS,
    ACTION_SUFFIX_OUTPUT_DIM,
    ACTION_SUFFIX_OUTPUT_SEGMENTS,
    CODE_DELTA_SCALE_FLOOR,
    FORMAT,
    FRAME_CODE_SEGMENTS,
    INPUT_DIM,
    INPUT_SCALE_FLOOR,
    INPUT_SEGMENTS,
    MEMBERS,
    NORMALIZED_INPUT_CLIP,
    OUTPUT_DIM,
    OUTPUT_SEGMENTS,
    PHYSIOLOGY_DELTA_SCALE_FLOOR,
    PHYSIOLOGY_DIM,
    ActionSuffixConsequenceEnsemble,
    RichConsequenceEnsemble,
    artifact_identity,
    action_suffix_input_dim,
    action_suffix_input_segments,
    array_sha256,
    denormalize_output,
    denormalize_suffix_output,
    ensemble_summary,
    normalized_input,
    normalized_suffix_input,
    suffix_ensemble_summary,
    tensor_bundle_sha256,
)
from chreatures.organism_interface import ACTION_NAMES, PHYSIOLOGY_NAMES

BOOTSTRAP_FORMAT = "chreatures-rich-sensorimotor-bootstrap-v4"
TRAIN_WORLDS = (0, 1, 2)
VALIDATION_WORLDS = (3,)
PHYSIOLOGY_BOUNDS = (
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--suffix-horizons",
        type=int,
        nargs="+",
        choices=ACTION_SUFFIX_HORIZONS,
        help="fit experimental action-suffix endpoint models instead of the frozen H1 contract",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    if not 1 <= args.epochs <= 100 or not 32 <= args.batch_size <= 8192:
        raise SystemExit("invalid epoch or batch schedule")
    if not 0 < args.learning_rate <= 0.01 or not 0 <= args.weight_decay <= 0.1:
        raise SystemExit("invalid optimizer schedule")
    if args.suffix_horizons and len(set(args.suffix_horizons)) != len(args.suffix_horizons):
        raise SystemExit("suffix horizons must be unique")


def load_bootstrap(path: Path, device: torch.device):
    expected = sha256(path)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        value = torch.load(path, map_location=device, weights_only=True)
    if value.get("format") != BOOTSTRAP_FORMAT:
        raise ValueError("rich bootstrap format differs")
    model = RichSensorimotorModel().to(device)
    identity = value.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("format") != BOOTSTRAP_FORMAT
        or identity.get("config") != asdict(model.config)
    ):
        raise ValueError("rich bootstrap identity or current model dimensions differ")
    model.load_state_dict(value["model"], strict=True)
    model.eval().requires_grad_(False)
    normalizer = RichNormalizer.from_value(identity["normalizer"])
    return model, normalizer, identity, expected


@torch.inference_mode()
def encode_episode(episode, model, normalizer, device, time_chunk=64) -> np.ndarray:
    encoded = np.empty((*episode.observation.shape[:2], 256), dtype=np.float32)
    for start in range(0, len(episode.observation), time_chunk):
        stop = min(start + time_chunk, len(episode.observation))
        normalized = normalizer.normalize(episode.observation[start:stop])
        value = torch.as_tensor(normalized, device=device)
        encoded[start:stop] = model.encode_frames(value).cpu().numpy()
    return encoded


def transition_rows(episode, encoded, columns) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(3, len(episode.actions), dtype=np.int64)
    valid = np.ones((len(times), len(columns)), dtype=np.bool_)
    # reset means reset-before-row; any reset in the full t-3..t+1 interval
    # makes the four-frame history and next-frame target non-contiguous.
    for offset in range(-3, 2):
        valid &= ~episode.reset[times[:, None] + offset, columns[None, :]]
    time_rows, column_rows = np.nonzero(valid)
    selected_time = times[time_rows]
    selected_column = columns[column_rows]
    frames = np.concatenate(
        [encoded[selected_time + offset, selected_column] for offset in range(-3, 1)],
        axis=1,
    )
    previous = episode.previous[selected_time, selected_column]
    candidate = episode.actions[selected_time, selected_column]
    x = np.ascontiguousarray(
        np.concatenate(
            (frames, episode.neural[selected_time, selected_column], previous, candidate),
            axis=1,
        ),
        dtype=np.float32,
    )
    physiology = episode.observation[..., -PHYSIOLOGY_DIM:]
    y = np.ascontiguousarray(
        np.concatenate(
            (
                encoded[selected_time + 1, selected_column]
                - encoded[selected_time, selected_column],
                physiology[selected_time + 1, selected_column]
                - physiology[selected_time, selected_column],
            ),
            axis=1,
        ),
        dtype=np.float32,
    )
    if x.shape[1] != INPUT_DIM or y.shape[1] != OUTPUT_DIM:
        raise RuntimeError("constructed consequence rows have the wrong shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("constructed consequence rows are nonfinite")
    return x, y


def moments(x: np.ndarray, floor: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.asarray(floor, dtype=np.float32))
    return mean, scale


def clip_report(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    clipped = np.abs((x - mean) / scale) > NORMALIZED_INPUT_CLIP
    return {
        "coordinate_fraction": float(clipped.mean()),
        "row_fraction": float(clipped.any(axis=1).mean()),
        "rows": int(len(x)),
        "clipped_rows": int(clipped.any(axis=1).sum()),
    }


def train(
    ensemble,
    x,
    y,
    input_mean,
    input_scale,
    target_mean,
    target_scale,
    args,
    device,
) -> list[dict[str, Any]]:
    input_mean_t = torch.as_tensor(input_mean, device=device)
    input_scale_t = torch.as_tensor(input_scale, device=device)
    target_mean_t = torch.as_tensor(target_mean, device=device)
    target_scale_t = torch.as_tensor(target_scale, device=device)
    rng = np.random.default_rng(args.seed)
    traces = []
    for member_index, member in enumerate(ensemble.members):
        optimizer = torch.optim.AdamW(
            member.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for epoch in range(args.epochs):
            squared_error = 0.0
            count = 0
            for start in range(0, len(x), args.batch_size):
                if start == 0:
                    order = rng.permutation(len(x))
                rows = order[start : start + args.batch_size]
                xb = torch.as_tensor(x[rows], device=device)
                yb = torch.as_tensor(y[rows], device=device)
                xb, _ = normalized_input(xb, input_mean_t, input_scale_t)
                yb = (yb - target_mean_t) / target_scale_t
                prediction = member(xb)
                loss = torch.mean((prediction - yb) ** 2)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite rich consequence loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                optimizer.step()
                squared_error += float(loss.detach()) * len(rows)
                count += len(rows)
            record = {
                "member": member_index,
                "epoch": epoch + 1,
                "normalized_mse": squared_error / count,
            }
            traces.append(record)
            print(json.dumps({"training": record}), flush=True)
    return traces


@torch.inference_mode()
def predict(
    ensemble,
    x,
    input_mean,
    input_scale,
    target_mean,
    target_scale,
    batch_size,
    device,
    candidate_override=None,
):
    means = np.empty((len(x), OUTPUT_DIM), dtype=np.float32)
    disagreements = np.empty_like(means)
    clipped = np.empty(len(x), dtype=np.bool_)
    input_mean_t = torch.as_tensor(input_mean, device=device)
    input_scale_t = torch.as_tensor(input_scale, device=device)
    target_mean_t = torch.as_tensor(target_mean, device=device)
    target_scale_t = torch.as_tensor(target_scale, device=device)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        batch = np.array(x[start:stop], copy=True)
        if candidate_override is not None:
            batch[:, INPUT_SEGMENTS["candidate_action"][0] :] = candidate_override[
                start:stop
            ]
        value, was_clipped = normalized_input(
            torch.as_tensor(batch, device=device), input_mean_t, input_scale_t
        )
        raw_members = denormalize_output(
            ensemble(value), target_mean_t, target_scale_t
        )
        mean, disagreement = ensemble_summary(raw_members)
        means[start:stop] = mean.cpu().numpy()
        disagreements[start:stop] = disagreement.cpu().numpy()
        clipped[start:stop] = was_clipped.cpu().numpy()
    return means, disagreements, clipped


def errors(prediction, observed, target_scale) -> dict[str, Any]:
    groups = {
        "visual_code_delta": (0, 128),
        "body_code_delta": (128, 256),
        "physiology_delta": (256, 256 + PHYSIOLOGY_DIM),
    }
    result = {}
    for name, (start, stop) in groups.items():
        residual = prediction[:, start:stop] - observed[:, start:stop]
        scaled = residual / target_scale[start:stop]
        result[name] = {
            "raw_rmse": float(np.sqrt(np.mean(residual.astype(np.float64) ** 2))),
            "raw_mae": float(np.mean(np.abs(residual.astype(np.float64)))),
            "train_scale_rmse": float(np.sqrt(np.mean(scaled.astype(np.float64) ** 2))),
        }
    residual = prediction - observed
    result["all_outputs_train_scale_rmse"] = float(
        np.sqrt(np.mean((residual / target_scale).astype(np.float64) ** 2))
    )
    return result


@torch.inference_mode()
def goal_calibration(model, x, y, predicted, batch_size, device):
    squared = np.zeros(64, dtype=np.float64)
    zero_squared = np.zeros(64, dtype=np.float64)
    count = 0
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        rows = torch.as_tensor(x[start:stop, :1024], device=device).reshape(-1, 4, 256)
        current = rows[:, -1]
        actual_next = current + torch.as_tensor(y[start:stop, :256], device=device)
        predicted_next = current + torch.as_tensor(
            predicted[start:stop, :256], device=device
        )
        actual_goal = model.goal_encoder(
            torch.cat((rows[:, 1:], actual_next[:, None]), dim=1).flatten(1)
        )
        predicted_goal = model.goal_encoder(
            torch.cat((rows[:, 1:], predicted_next[:, None]), dim=1).flatten(1)
        )
        zero_goal = model.goal_encoder(
            torch.cat((rows[:, 1:], current[:, None]), dim=1).flatten(1)
        )
        squared += ((predicted_goal - actual_goal) ** 2).sum(0).cpu().numpy()
        zero_squared += ((zero_goal - actual_goal) ** 2).sum(0).cpu().numpy()
        count += stop - start
    per_coordinate = np.sqrt(squared / count)
    zero_per_coordinate = np.sqrt(zero_squared / count)
    return {
        "rows": count,
        "per_goal_coordinate_rmse": per_coordinate.tolist(),
        "overall_rms": float(np.sqrt(squared.sum() / (count * 64))),
        "zero_delta_per_goal_coordinate_rmse": zero_per_coordinate.tolist(),
        "zero_delta_overall_rms": float(
            np.sqrt(zero_squared.sum() / (count * 64))
        ),
        "runtime_empirical_error_scale": float(
            max(np.sqrt(squared.sum() / (count * 64)), 1e-4)
        ),
        "interpretation": "empirical goal-space RMS on representation-exposed reserved world; not a probability interval",
    }


def copied_encoders(model) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    source = model.state_dict()
    result = {}
    groups = {"frame": {}, "goal": {}}
    for name, tensor in source.items():
        if name.startswith(("visual.", "body.")):
            group = "frame"
        elif name.startswith("goal_encoder."):
            group = "goal"
        else:
            continue
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        export_name = f"encoder.{name}"
        result[export_name] = array
        groups[group][export_name] = array
    return result, {
        "frame_encoder_sha256": tensor_bundle_sha256(groups["frame"]),
        "goal_encoder_sha256": tensor_bundle_sha256(groups["goal"]),
        "copied_encoder_sha256": tensor_bundle_sha256(result),
    }


def suffix_transition_rows(episode, encoded, columns, horizon):
    """Build reset-safe H-step rows and endpoint four-frame targets."""
    times = np.arange(3, len(episode.actions) - horizon + 1, dtype=np.int64)
    valid = np.ones((len(times), len(columns)), dtype=np.bool_)
    for offset in range(-3, horizon + 1):
        valid &= ~episode.reset[times[:, None] + offset, columns[None, :]]
    time_rows, column_rows = np.nonzero(valid)
    selected_time = times[time_rows]
    selected_column = columns[column_rows]
    frames = np.concatenate(
        [encoded[selected_time + offset, selected_column] for offset in range(-3, 1)],
        axis=1,
    )
    suffix = np.concatenate(
        [episode.actions[selected_time + offset, selected_column] for offset in range(horizon)],
        axis=1,
    )
    x = np.ascontiguousarray(
        np.concatenate(
            (
                frames,
                episode.neural[selected_time, selected_column],
                episode.previous[selected_time, selected_column],
                suffix,
            ),
            axis=1,
        ),
        dtype=np.float32,
    )
    current = encoded[selected_time, selected_column]
    future_window_deltas = np.concatenate(
        [
            encoded[selected_time + horizon - 3 + offset, selected_column] - current
            for offset in range(4)
        ],
        axis=1,
    )
    physiology = episode.observation[..., -PHYSIOLOGY_DIM:]
    y = np.ascontiguousarray(
        np.concatenate(
            (
                future_window_deltas,
                physiology[selected_time + horizon, selected_column]
                - physiology[selected_time, selected_column],
            ),
            axis=1,
        ),
        dtype=np.float32,
    )
    if x.shape[1] != action_suffix_input_dim(horizon):
        raise RuntimeError(f"constructed H{horizon} input rows have the wrong shape")
    if y.shape[1] != ACTION_SUFFIX_OUTPUT_DIM:
        raise RuntimeError(f"constructed H{horizon} target rows have the wrong shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError(f"constructed H{horizon} rows are nonfinite")
    return x, y


def zero_current_window_prediction(x):
    """Persist the already observed four-frame window and current physiology."""
    frames = x[:, :1024].reshape(-1, 4, 256)
    current = frames[:, -1:, :]
    deltas = (frames - current).reshape(-1, 1024)
    return np.ascontiguousarray(
        np.concatenate(
            (deltas, np.zeros((len(x), PHYSIOLOGY_DIM), dtype=np.float32)), axis=1
        ),
        dtype=np.float32,
    )


def train_suffix(
    ensemble,
    x,
    y,
    input_mean,
    input_scale,
    target_mean,
    target_scale,
    args,
    device,
    horizon,
    seed,
):
    input_mean_t = torch.as_tensor(input_mean, device=device)
    input_scale_t = torch.as_tensor(input_scale, device=device)
    target_mean_t = torch.as_tensor(target_mean, device=device)
    target_scale_t = torch.as_tensor(target_scale, device=device)
    rng = np.random.default_rng(seed)
    traces = []
    failures = []
    for member_index, member in enumerate(ensemble.members):
        optimizer = torch.optim.AdamW(
            member.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        try:
            for epoch in range(args.epochs):
                order = rng.permutation(len(x))
                squared_error = 0.0
                count = 0
                for start in range(0, len(x), args.batch_size):
                    rows = order[start : start + args.batch_size]
                    xb = torch.as_tensor(x[rows], device=device)
                    yb = torch.as_tensor(y[rows], device=device)
                    xb, _ = normalized_suffix_input(
                        xb, input_mean_t, input_scale_t, horizon
                    )
                    yb = (yb - target_mean_t) / target_scale_t
                    prediction = member(xb)
                    loss = torch.mean((prediction - yb) ** 2)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite suffix consequence loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                    optimizer.step()
                    squared_error += float(loss.detach()) * len(rows)
                    count += len(rows)
                item = {
                    "horizon": horizon,
                    "member": member_index,
                    "epoch": epoch + 1,
                    "normalized_mse": squared_error / count,
                }
                traces.append(item)
                print(json.dumps({"training": item}), flush=True)
        except Exception as error:
            failures.append({"member": member_index, "error": repr(error)})
            break
    if failures:
        raise RuntimeError(f"H{horizon} member training failed: {failures}")
    return traces, failures


@torch.inference_mode()
def predict_suffix(
    ensemble,
    x,
    input_mean,
    input_scale,
    target_mean,
    target_scale,
    batch_size,
    device,
    horizon,
    suffix_override=None,
    retain_members=False,
):
    means = np.empty((len(x), ACTION_SUFFIX_OUTPUT_DIM), dtype=np.float32)
    disagreements = np.empty_like(means)
    members = (
        np.empty((len(x), MEMBERS, ACTION_SUFFIX_OUTPUT_DIM), dtype=np.float32)
        if retain_members
        else None
    )
    clipped = np.empty(len(x), dtype=np.bool_)
    input_mean_t = torch.as_tensor(input_mean, device=device)
    input_scale_t = torch.as_tensor(input_scale, device=device)
    target_mean_t = torch.as_tensor(target_mean, device=device)
    target_scale_t = torch.as_tensor(target_scale, device=device)
    suffix_start = action_suffix_input_segments(horizon)[
        "executed_action_suffix_t_through_t_plus_h_minus_1"
    ][0]
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        batch = np.array(x[start:stop], copy=True)
        if suffix_override is not None:
            batch[:, suffix_start:] = suffix_override[start:stop]
        value, was_clipped = normalized_suffix_input(
            torch.as_tensor(batch, device=device), input_mean_t, input_scale_t, horizon
        )
        raw_members = denormalize_suffix_output(
            ensemble(value), target_mean_t, target_scale_t
        )
        mean, disagreement = suffix_ensemble_summary(raw_members)
        means[start:stop] = mean.cpu().numpy()
        disagreements[start:stop] = disagreement.cpu().numpy()
        clipped[start:stop] = was_clipped.cpu().numpy()
        if members is not None:
            members[start:stop] = raw_members.cpu().numpy()
    return means, disagreements, clipped, members


def suffix_errors(prediction, observed, target_scale):
    residual = prediction - observed
    scaled = residual / target_scale
    visual_indices = np.concatenate(
        [np.arange(offset, offset + 128) for offset in range(0, 1024, 256)]
    )
    body_indices = np.concatenate(
        [np.arange(offset + 128, offset + 256) for offset in range(0, 1024, 256)]
    )
    groups = {
        "future_window_code_delta": np.arange(1024),
        "future_window_visual_code_delta": visual_indices,
        "future_window_body_code_delta": body_indices,
        "future_physiology_delta": np.arange(1024, 1024 + PHYSIOLOGY_DIM),
    }
    result = {}
    for name, indices in groups.items():
        raw = residual[:, indices].astype(np.float64)
        normalized = scaled[:, indices].astype(np.float64)
        result[name] = {
            "raw_rmse": float(np.sqrt(np.mean(raw**2))),
            "raw_mae": float(np.mean(np.abs(raw))),
            "train_scale_rmse": float(np.sqrt(np.mean(normalized**2))),
        }
    result["all_outputs_train_scale_rmse"] = float(
        np.sqrt(np.mean(scaled.astype(np.float64) ** 2))
    )
    return result


@torch.inference_mode()
def suffix_goal_calibration(
    model, x, observed, predicted, permuted, zero, batch_size, device
):
    sums = {
        "model": np.zeros(64, dtype=np.float64),
        "zero_current_window": np.zeros(64, dtype=np.float64),
        "action_suffix_permuted": np.zeros(64, dtype=np.float64),
    }
    count = 0
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        frames = torch.as_tensor(
            x[start:stop, :1024], device=device
        ).reshape(-1, 4, 256)
        current = frames[:, -1:, :]

        def goal(values):
            deltas = torch.as_tensor(
                values[start:stop, :1024], device=device
            ).reshape(-1, 4, 256)
            return model.goal_encoder((current + deltas).flatten(1))

        actual_goal = goal(observed)
        for name, values in (
            ("model", predicted),
            ("zero_current_window", zero),
            ("action_suffix_permuted", permuted),
        ):
            sums[name] += ((goal(values) - actual_goal) ** 2).sum(0).cpu().numpy()
        count += stop - start
    result = {"rows": count}
    for name, squared in sums.items():
        result[name] = {
            "per_goal_coordinate_rmse": np.sqrt(squared / count).tolist(),
            "overall_rms": float(np.sqrt(squared.sum() / (count * 64))),
        }
    result["empirical_model_error_scale"] = max(result["model"]["overall_rms"], 1e-4)
    result["interpretation"] = (
        "empirical frozen-goal-space RMS on a representation-exposed reserved world; "
        "not a probability interval"
    )
    return result


def suffix_plan_coverage(x, input_mean, input_scale, horizon):
    start = action_suffix_input_segments(horizon)[
        "executed_action_suffix_t_through_t_plus_h_minus_1"
    ][0]
    suffix = x[:, start:].reshape(-1, horizon, ACTION_DIM)
    exact_constant = np.max(np.abs(suffix - suffix[:, :1]), axis=(1, 2)) <= 1e-7
    action_stop = np.max(np.abs(suffix), axis=(1, 2)) <= 0.05

    def domain(values):
        candidate = np.array(x, copy=True)
        candidate[:, start:] = values.reshape(len(x), -1)
        standardized = (candidate - input_mean) / input_scale
        suffix_standardized = standardized[:, start:]
        return {
            "row_8_sigma_clip_fraction": float(
                np.any(np.abs(standardized) > NORMALIZED_INPUT_CLIP, axis=1).mean()
            ),
            "suffix_standardized_rms_mean": float(
                np.sqrt(np.mean(suffix_standardized.astype(np.float64) ** 2, axis=1)).mean()
            ),
            "suffix_standardized_abs_max": float(np.max(np.abs(suffix_standardized))),
        }

    zero_plan = np.zeros_like(suffix)
    repeated_first = np.repeat(suffix[:, :1], horizon, axis=1)
    return {
        "actual_suffix": {
            **domain(suffix),
            "exact_constant_fraction": float(exact_constant.mean()),
            "action_stop_le_0.05_fraction": float(action_stop.mean()),
            "action_stop_rows": int(action_stop.sum()),
        },
        "candidate_constant_zero_action": domain(zero_plan),
        "candidate_repeat_first_observed_action": domain(repeated_first),
        "interpretation": (
            "input-domain diagnostics only; a future constant-plan controller would execute "
            "the first action and replan, but no such controller is implemented"
        ),
    }


def suffix_strata(x, y, prediction, zero, target_scale, horizon):
    start = action_suffix_input_segments(horizon)[
        "executed_action_suffix_t_through_t_plus_h_minus_1"
    ][0]
    suffix = x[:, start:].reshape(-1, horizon, ACTION_DIM)
    exact_constant = np.max(np.abs(suffix - suffix[:, :1]), axis=(1, 2)) <= 1e-7
    action_stop = np.max(np.abs(suffix), axis=(1, 2)) <= 0.05
    result = {}
    for name, mask in (
        ("actual_exact_constant_suffix", exact_constant),
        ("actual_action_stop_suffix_le_0.05", action_stop),
        ("actual_variable_suffix", ~exact_constant),
    ):
        result[name] = {
            "rows": int(mask.sum()),
            "model": suffix_errors(prediction[mask], y[mask], target_scale)
            if mask.any()
            else None,
            "zero_current_window": suffix_errors(zero[mask], y[mask], target_scale)
            if mask.any()
            else None,
        }
    return result


def run_suffix_horizon(
    args,
    dataset,
    model,
    observation_normalizer,
    bootstrap_identity,
    bootstrap_sha,
    encoded,
    horizon,
    device,
):
    started = time.perf_counter()
    train_columns = dataset.columns(TRAIN_WORLDS)
    validation_columns = dataset.columns(VALIDATION_WORLDS)
    train_parts = [
        suffix_transition_rows(episode, codes, train_columns, horizon)
        for episode, codes in zip(dataset.episodes, encoded, strict=True)
    ]
    validation_parts = [
        suffix_transition_rows(episode, codes, validation_columns, horizon)
        for episode, codes in zip(dataset.episodes, encoded, strict=True)
    ]
    train_x = np.concatenate([part[0] for part in train_parts])
    train_y = np.concatenate([part[1] for part in train_parts])
    validation_x = np.concatenate([part[0] for part in validation_parts])
    validation_y = np.concatenate([part[1] for part in validation_parts])
    del train_parts, validation_parts
    input_mean, input_scale = moments(train_x, INPUT_SCALE_FLOOR)
    target_floor = np.concatenate(
        (
            np.full(1024, CODE_DELTA_SCALE_FLOOR, dtype=np.float32),
            np.full(PHYSIOLOGY_DIM, PHYSIOLOGY_DELTA_SCALE_FLOOR, dtype=np.float32),
        )
    )
    target_mean, target_scale = moments(train_y, target_floor)
    training_seed = args.seed + horizon * 1009
    torch.manual_seed(training_seed)
    ensemble = ActionSuffixConsequenceEnsemble(horizon).to(device)
    trace, failures = train_suffix(
        ensemble,
        train_x,
        train_y,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
        args,
        device,
        horizon,
        training_seed,
    )
    ensemble.eval()
    train_prediction, _, _, _ = predict_suffix(
        ensemble,
        train_x,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
        args.batch_size,
        device,
        horizon,
    )
    validation_prediction, validation_disagreement, validation_clipped, _ = predict_suffix(
        ensemble,
        validation_x,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
        args.batch_size,
        device,
        horizon,
    )
    zero_prediction = zero_current_window_prediction(validation_x)
    suffix_start = action_suffix_input_segments(horizon)[
        "executed_action_suffix_t_through_t_plus_h_minus_1"
    ][0]
    rng = np.random.default_rng(training_seed + 991)
    permuted_suffix = validation_x[rng.permutation(len(validation_x)), suffix_start:]
    permuted_prediction, _, permuted_clipped, _ = predict_suffix(
        ensemble,
        validation_x,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
        args.batch_size,
        device,
        horizon,
        suffix_override=permuted_suffix,
    )
    residual_scale = np.maximum(
        np.sqrt(np.mean((train_prediction - train_y).astype(np.float64) ** 2, axis=0)),
        1e-8,
    ).astype(np.float32)
    calibration = suffix_goal_calibration(
        model,
        validation_x,
        validation_y,
        validation_prediction,
        permuted_prediction,
        zero_prediction,
        args.batch_size,
        device,
    )
    report = {
        "format": "chreatures-rich-action-suffix-fit-report-v2",
        "status": (
            "experimental descriptive action-suffix forecast; no runtime, causal, "
            "welfare, or calibrated-uncertainty claim"
        ),
        "horizon": {
            "ticks": horizon,
            "seconds": horizon * dataset.dt_seconds,
            "target_window_ticks": list(range(horizon - 3, horizon + 1)),
            "valid_t": f"3 through T-{horizon}",
            "reset_exclusion": f"any reset from t-3 through t+{horizon}",
        },
        "split": {
            "train_worlds": list(TRAIN_WORLDS),
            "validation_worlds": list(VALIDATION_WORLDS),
            "train_rows": len(train_x),
            "validation_rows": len(validation_x),
            "disclosure": (
                "the frozen representation bootstrap saw all four worlds; world 3 is "
                "predictor validation only, not end-to-end held-out generalization"
            ),
        },
        "metrics": {
            "train": suffix_errors(train_prediction, train_y, target_scale),
            "validation": suffix_errors(validation_prediction, validation_y, target_scale),
            "validation_zero_current_window": suffix_errors(
                zero_prediction, validation_y, target_scale
            ),
            "validation_action_suffix_permuted": suffix_errors(
                permuted_prediction, validation_y, target_scale
            ),
            "validation_actual_suffix_strata": suffix_strata(
                validation_x,
                validation_y,
                validation_prediction,
                zero_prediction,
                target_scale,
                horizon,
            ),
        },
        "goal_space_calibration": calibration,
        "input_clipping_at_8_sigma": {
            "train": clip_report(train_x, input_mean, input_scale),
            "validation": clip_report(validation_x, input_mean, input_scale),
            "validation_action_suffix_permuted_row_fraction": float(
                permuted_clipped.mean()
            ),
            "runtime_rule_if_integrated": (
                "clamp every normalized coordinate to [-8,8]; any clipped plan receives "
                "zero predictor selection tilt"
            ),
        },
        "constant_plan_coverage": suffix_plan_coverage(
            validation_x, input_mean, input_scale, horizon
        ),
        "ensemble": {
            "members": MEMBERS,
            "failed_models": failures,
            "validation_mean_disagreement": {
                "future_window_code_delta": float(
                    validation_disagreement[:, :1024].mean()
                ),
                "future_physiology_delta": float(
                    validation_disagreement[:, 1024:].mean()
                ),
            },
            "interpretation": (
                "population RMS spread of three member predictions in raw target units; "
                "not calibrated uncertainty"
            ),
        },
        "training_trace": trace,
        "exposure": {
            "epochs_per_member": args.epochs,
            "members": MEMBERS,
            "rows_per_epoch": len(train_x),
            "row_presentations_total": len(train_x) * args.epochs * MEMBERS,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    encoder_arrays, encoder_hashes = copied_encoders(model)
    arrays = {
        **encoder_arrays,
        "observation_normalizer.mean": np.ascontiguousarray(
            observation_normalizer.mean, dtype="<f4"
        ),
        "observation_normalizer.scale": np.ascontiguousarray(
            observation_normalizer.scale, dtype="<f4"
        ),
        "input.mean": np.ascontiguousarray(input_mean, dtype="<f4"),
        "input.scale": np.ascontiguousarray(input_scale, dtype="<f4"),
        "target.mean": np.ascontiguousarray(target_mean, dtype="<f4"),
        "target.scale": np.ascontiguousarray(target_scale, dtype="<f4"),
        "residual.scale": np.ascontiguousarray(residual_scale, dtype="<f4"),
    }
    for member_index, member in enumerate(ensemble.members):
        for name, tensor in member.state_dict().items():
            arrays[f"member.{member_index}.{name}"] = np.ascontiguousarray(
                tensor.detach().cpu().numpy(), dtype="<f4"
            )
    normalizer_value = observation_normalizer.to_value()
    metadata = {
        "format": ACTION_SUFFIX_FORMAT,
        "version": 2,
        "experimental": True,
        "horizon_ticks": horizon,
        "horizon_seconds": horizon * dataset.dt_seconds,
        "architecture": {
            "members": MEMBERS,
            "layers": [
                [action_suffix_input_dim(horizon), 256, "tanh"],
                [256, 256, "tanh"],
                [256, ACTION_SUFFIX_OUTPUT_DIM, "linear"],
            ],
            "ensemble_aggregation": "arithmetic mean in raw target units",
            "disagreement": (
                "per-coordinate population RMS member deviation in raw target units; "
                "uncalibrated"
            ),
        },
        "input": {
            "dimension": action_suffix_input_dim(horizon),
            "segments": action_suffix_input_segments(horizon),
            "action_order": list(ACTION_NAMES),
            "suffix_semantics": (
                "actual 12-axis executed action for every tick t through t+H-1"
            ),
            "normalization": (
                "train-world mean and population standard deviation, floor 0.02, clamp [-8,8]"
            ),
            "clipped_candidate_selection_tilt_if_integrated": 0.0,
        },
        "output": {
            "dimension": ACTION_SUFFIX_OUTPUT_DIM,
            "segments": ACTION_SUFFIX_OUTPUT_SEGMENTS,
            "frame_delta_anchor": "each of four endpoint-window codes minus e[t]",
            "endpoint_window": "e[t+H-3] through e[t+H] reconstructed independently",
            "normalization": (
                "train-world delta mean and population standard deviation; code floor "
                "1e-3, physiology floor 1e-4"
            ),
            "physiology_forecast": (
                "raw physiology[t] plus predicted raw delta at t+H; do not clamp"
            ),
        },
        "source": {
            "dataset_manifest_content_sha256": dataset.manifest["content_sha256"],
            "dataset_files": dataset.file_sha256s,
            "bootstrap_file_sha256": bootstrap_sha,
            "bootstrap_format": bootstrap_identity["format"],
            "bootstrap_identity_sha256": hashlib.sha256(
                json.dumps(
                    bootstrap_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
            "bootstrap_source_sha256": bootstrap_identity["source_sha256"],
            "bootstrap_observation_normalizer_sha256": normalizer_value["sha256"],
            **encoder_hashes,
            "source_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    ROOT / "research/sensorimotor_skills/rich_prediction.py",
                    ROOT / "research/sensorimotor_skills/rich_model.py",
                    ROOT / "research/sensorimotor_skills/rich_data.py",
                    ROOT / "research/sensorimotor_skills/trajectory-schema-rich-v3.json",
                )
            },
        },
        "training": {
            "train_worlds": list(TRAIN_WORLDS),
            "validation_worlds": list(VALIDATION_WORLDS),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": training_seed,
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "validation": {
            "goal_space_model_rms": calibration["model"]["overall_rms"],
            "goal_space_zero_current_window_rms": calibration[
                "zero_current_window"
            ]["overall_rms"],
            "goal_space_action_suffix_permuted_rms": calibration[
                "action_suffix_permuted"
            ]["overall_rms"],
            "empirical_model_error_scale": calibration["empirical_model_error_scale"],
            "raw_input_8_sigma_clip_row_fraction": report[
                "input_clipping_at_8_sigma"
            ]["validation"]["row_fraction"],
        },
        "prospective_use": (
            "compare constant H-tick candidate plans, execute only the first action, and "
            "replan; not implemented by this artifact"
        ),
        "limitations": [
            "The frozen representation front saw all four worlds before this predictor fit.",
            "Action suffixes are observational and do not establish intervention effects or causality.",
            "Ensemble disagreement is an uncalibrated diagnostic, not a probability or welfare estimate.",
            "No runtime planner or resident integration is implemented by this experiment.",
        ],
    }
    metadata["tensors"] = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    metadata["artifact_identity"] = artifact_identity(metadata, arrays)
    output = args.output / f"horizon-{horizon:03d}"
    output.mkdir(parents=True, exist_ok=False)
    arrays_with_metadata = {
        "metadata": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
    } | arrays
    artifact_path = output / f"rich-action-suffix-h{horizon:03d}-ensemble.npz"
    atomic_npz(artifact_path, arrays_with_metadata)
    report["artifact"] = {
        "file": artifact_path.name,
        "bytes": artifact_path.stat().st_size,
        "sha256": sha256(artifact_path),
        "artifact_identity": metadata["artifact_identity"],
    }
    reference_rows = np.asarray([0, len(validation_x) // 2, len(validation_x) - 1])
    reference_prediction, reference_disagreement, reference_clipped, reference_members = (
        predict_suffix(
            ensemble,
            validation_x[reference_rows],
            input_mean,
            input_scale,
            target_mean,
            target_scale,
            len(reference_rows),
            device,
            horizon,
            retain_members=True,
        )
    )
    reference = {
        "input": validation_x[reference_rows],
        "target": validation_y[reference_rows],
        "member_raw_output": reference_members,
        "ensemble_mean": reference_prediction,
        "ensemble_disagreement": reference_disagreement,
        "input_clipped": reference_clipped,
    }
    reference_path = output / f"rich-action-suffix-h{horizon:03d}-reference.npz"
    atomic_npz(reference_path, reference)
    report["reference"] = {
        "file": reference_path.name,
        "bytes": reference_path.stat().st_size,
        "sha256": sha256(reference_path),
        "rows": reference_rows.tolist(),
        "source": "actual reserved-world validation rows",
    }
    atomic_json(output / "fit-report.json", report)
    atomic_json(output / "identity.json", metadata)
    result = {
        "format": ACTION_SUFFIX_FORMAT,
        "status": report["status"],
        "horizon_ticks": horizon,
        "artifact": report["artifact"],
        "reference": report["reference"],
        "fit_report_sha256": sha256(output / "fit-report.json"),
        "identity_sha256": sha256(output / "identity.json"),
        "validation": metadata["validation"],
        "failed_models": failures,
        "elapsed_seconds": report["elapsed_seconds"],
    }
    atomic_json(output / "result.json", result)
    del train_x, train_y, validation_x, validation_y, train_prediction
    return result


def run_suffix_experiments(
    args,
    dataset,
    model,
    observation_normalizer,
    bootstrap_identity,
    bootstrap_sha,
    encoded,
    device,
):
    results = []
    for horizon in args.suffix_horizons:
        result = run_suffix_horizon(
            args,
            dataset,
            model,
            observation_normalizer,
            bootstrap_identity,
            bootstrap_sha,
            encoded,
            horizon,
            device,
        )
        results.append(result)
        print(json.dumps({"completed_horizon": result}, sort_keys=True), flush=True)
    suite = {
        "format": "chreatures-rich-action-suffix-experiment-suite-v2",
        "status": "completed experimental Torch fits; no runtime integration",
        "horizons": list(args.suffix_horizons),
        "results": results,
        "failed_horizons": [],
    }
    atomic_json(args.output / "result.json", suite)
    print(json.dumps(suite, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = arguments()
    validate_args(args)
    args.dataset = args.dataset.expanduser().resolve()
    args.bootstrap = args.bootstrap.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dataset = RichPlayDataset(args.dataset)
    if dataset.world_count != 4 or dataset.residents_per_world != 6:
        raise ValueError("rich prediction requires the recorded four-world, six-resident split")
    if any(episode.neural is None for episode in dataset.episodes):
        raise ValueError("rich prediction requires recorded neural readouts")
    model, observation_normalizer, bootstrap_identity, bootstrap_sha = load_bootstrap(
        args.bootstrap, device
    )
    if bootstrap_identity["dataset_manifest_content_sha256"] != dataset.manifest["content_sha256"]:
        raise ValueError("bootstrap and consequence dataset identities differ")

    started = time.perf_counter()
    encoded = []
    for episode in dataset.episodes:
        values = encode_episode(episode, model, observation_normalizer, device)
        encoded.append(values)
        print(json.dumps({"encoded_episode": episode.episode, "shape": list(values.shape)}), flush=True)
    if args.suffix_horizons:
        return run_suffix_experiments(
            args,
            dataset,
            model,
            observation_normalizer,
            bootstrap_identity,
            bootstrap_sha,
            encoded,
            device,
        )
    train_columns = dataset.columns(TRAIN_WORLDS)
    validation_columns = dataset.columns(VALIDATION_WORLDS)
    train_parts = [
        transition_rows(episode, codes, train_columns)
        for episode, codes in zip(dataset.episodes, encoded, strict=True)
    ]
    validation_parts = [
        transition_rows(episode, codes, validation_columns)
        for episode, codes in zip(dataset.episodes, encoded, strict=True)
    ]
    train_x = np.concatenate([part[0] for part in train_parts])
    train_y = np.concatenate([part[1] for part in train_parts])
    validation_x = np.concatenate([part[0] for part in validation_parts])
    validation_y = np.concatenate([part[1] for part in validation_parts])
    del train_parts, validation_parts, encoded

    input_mean, input_scale = moments(train_x, INPUT_SCALE_FLOOR)
    output_floor = np.concatenate(
        (
            np.full(256, CODE_DELTA_SCALE_FLOOR, dtype=np.float32),
            np.full(PHYSIOLOGY_DIM, PHYSIOLOGY_DELTA_SCALE_FLOOR, dtype=np.float32),
        )
    )
    target_mean, target_scale = moments(train_y, output_floor)
    ensemble = RichConsequenceEnsemble().to(device)
    trace = train(
        ensemble,
        train_x,
        train_y,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
        args,
        device,
    )
    ensemble.eval()

    train_prediction, _, _ = predict(
        ensemble, train_x, input_mean, input_scale, target_mean, target_scale,
        args.batch_size, device,
    )
    validation_prediction, validation_disagreement, validation_clipped = predict(
        ensemble, validation_x, input_mean, input_scale, target_mean, target_scale,
        args.batch_size, device,
    )
    residual_scale = np.maximum(
        np.sqrt(np.mean((train_prediction - train_y).astype(np.float64) ** 2, axis=0)),
        1e-8,
    ).astype(np.float32)
    rng = np.random.default_rng(args.seed + 991)
    candidate_start = INPUT_SEGMENTS["candidate_action"][0]
    permuted_candidate = validation_x[
        rng.permutation(len(validation_x)), candidate_start:
    ]
    permuted_prediction, _, permuted_clipped = predict(
        ensemble, validation_x, input_mean, input_scale, target_mean, target_scale,
        args.batch_size, device, candidate_override=permuted_candidate,
    )

    previous_start, previous_stop = INPUT_SEGMENTS["previous_executed_action"]
    action_change = np.mean(
        np.abs(
            validation_x[:, candidate_start:]
            - validation_x[:, previous_start:previous_stop]
        ),
        axis=1,
    )
    strata = {}
    for name, mask in (
        ("near_unchanged_le_0.05", action_change <= 0.05),
        ("changed_0.05_to_0.25", (action_change > 0.05) & (action_change <= 0.25)),
        ("changed_gt_0.25", action_change > 0.25),
    ):
        strata[name] = {
            "rows": int(mask.sum()),
            "action_change_mean_abs_12": float(action_change[mask].mean())
            if mask.any()
            else None,
            "model": errors(validation_prediction[mask], validation_y[mask], target_scale)
            if mask.any()
            else None,
            "zero_delta": errors(np.zeros_like(validation_y[mask]), validation_y[mask], target_scale)
            if mask.any()
            else None,
        }

    calibration = goal_calibration(
        model, validation_x, validation_y, validation_prediction, args.batch_size, device
    )
    report = {
        "format": "chreatures-rich-consequence-fit-report-v2",
        "status": "descriptive reserved-world prediction; no causal, welfare, or calibrated-uncertainty claim",
        "split": {
            "train_worlds": list(TRAIN_WORLDS),
            "validation_worlds": list(VALIDATION_WORLDS),
            "train_rows": len(train_x),
            "validation_rows": len(validation_x),
            "disclosure": "the frozen representation bootstrap saw all four worlds; world 3 is predictor validation only, not end-to-end held-out generalization",
        },
        "metrics": {
            "train": errors(train_prediction, train_y, target_scale),
            "validation": errors(validation_prediction, validation_y, target_scale),
            "validation_zero_delta": errors(np.zeros_like(validation_y), validation_y, target_scale),
            "validation_action_permuted": errors(permuted_prediction, validation_y, target_scale),
            "validation_action_change_strata": strata,
        },
        "input_clipping_at_8_sigma": {
            "train": clip_report(train_x, input_mean, input_scale),
            "validation": clip_report(validation_x, input_mean, input_scale),
            "validation_action_permuted_row_fraction": float(permuted_clipped.mean()),
            "runtime_rule": "clamp every normalized coordinate to [-8,8]; any clipped candidate receives zero predictor selection tilt",
        },
        "ensemble": {
            "members": MEMBERS,
            "validation_mean_disagreement_by_group": {
                "visual_code_delta": float(validation_disagreement[:, :128].mean()),
                "body_code_delta": float(validation_disagreement[:, 128:256].mean()),
                "physiology_delta": float(validation_disagreement[:, 256:].mean()),
            },
            "interpretation": "population RMS spread of three member predictions in raw target units; not calibrated uncertainty",
        },
        "goal_space_calibration": calibration,
        "training_trace": trace,
        "elapsed_seconds": time.perf_counter() - started,
    }

    encoder_arrays, encoder_hashes = copied_encoders(model)
    arrays = {
        **encoder_arrays,
        "observation_normalizer.mean": np.ascontiguousarray(observation_normalizer.mean, dtype="<f4"),
        "observation_normalizer.scale": np.ascontiguousarray(observation_normalizer.scale, dtype="<f4"),
        "input.mean": np.ascontiguousarray(input_mean, dtype="<f4"),
        "input.scale": np.ascontiguousarray(input_scale, dtype="<f4"),
        "target.mean": np.ascontiguousarray(target_mean, dtype="<f4"),
        "target.scale": np.ascontiguousarray(target_scale, dtype="<f4"),
        "residual.scale": np.ascontiguousarray(residual_scale, dtype="<f4"),
    }
    for member_index, member in enumerate(ensemble.members):
        for name, tensor in member.state_dict().items():
            arrays[f"member.{member_index}.{name}"] = np.ascontiguousarray(
                tensor.detach().cpu().numpy(), dtype="<f4"
            )
    normalizer_value = observation_normalizer.to_value()
    metadata = {
        "format": FORMAT,
        "version": 2,
        "architecture": {
            "members": MEMBERS,
            "layers": [[INPUT_DIM, 256, "tanh"], [256, 256, "tanh"], [256, OUTPUT_DIM, "linear"]],
            "ensemble_aggregation": "arithmetic mean in raw target units",
            "disagreement": "per-coordinate population RMS member deviation in raw target units; uncalibrated",
        },
        "input": {
            "dimension": INPUT_DIM,
            "segments": INPUT_SEGMENTS,
            "action_order": list(ACTION_NAMES),
            "timing": {
                "dt_seconds": dataset.dt_seconds,
                "frame_history_offsets": [-3, -2, -1, 0],
                "previous_action_tick": "t-1",
                "candidate_action_tick": "t",
                "target_observation_tick": "t+1",
            },
            "normalization": "train-world mean and population standard deviation, floor 0.02, clamp [-8,8]",
            "clipped_candidate_selection_tilt": 0.0,
        },
        "output": {
            "dimension": OUTPUT_DIM,
            "segments": OUTPUT_SEGMENTS,
            "frame_code_segments": FRAME_CODE_SEGMENTS,
            "normalization": "train-world delta mean and population standard deviation; code floor 1e-3, physiology floor 1e-4",
            "physiology_forecast": "raw physiology[t] + predicted raw delta; do not clamp, return validity separately",
            "physiology_order": list(PHYSIOLOGY_NAMES),
            "physiology_bounds": [list(bounds) for bounds in PHYSIOLOGY_BOUNDS],
            "validity": "input row was not clipped and every raw predicted next-physiology coordinate is finite and within physiology_bounds",
        },
        "source": {
            "dataset_manifest_content_sha256": dataset.manifest["content_sha256"],
            "dataset_files": dataset.file_sha256s,
            "bootstrap_file_sha256": bootstrap_sha,
            "bootstrap_format": bootstrap_identity["format"],
            "bootstrap_identity_sha256": hashlib.sha256(
                json.dumps(bootstrap_identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "bootstrap_source_sha256": bootstrap_identity["source_sha256"],
            "bootstrap_observation_normalizer_sha256": normalizer_value["sha256"],
            **encoder_hashes,
            "source_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    ROOT / "research/sensorimotor_skills/rich_prediction.py",
                    ROOT / "research/sensorimotor_skills/rich_model.py",
                    ROOT / "research/sensorimotor_skills/rich_data.py",
                    ROOT / "research/sensorimotor_skills/trajectory-schema-rich-v3.json",
                )
            },
        },
        "training": {
            "train_worlds": list(TRAIN_WORLDS),
            "validation_worlds": list(VALIDATION_WORLDS),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "validation": {
            "goal_space_overall_rms": calibration["overall_rms"],
            "goal_space_zero_delta_overall_rms": calibration["zero_delta_overall_rms"],
            "runtime_empirical_goal_error_scale": calibration["runtime_empirical_error_scale"],
            "raw_input_8_sigma_clip_row_fraction": report["input_clipping_at_8_sigma"]["validation"]["row_fraction"],
        },
        "limitations": [
            "The frozen representation front saw all four worlds before this predictor fit.",
            "Action-conditioned prediction is observational and does not establish intervention effects or causality.",
            "Ensemble disagreement is an uncalibrated diagnostic, not a probability or welfare estimate.",
        ],
    }
    metadata["tensors"] = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    metadata["artifact_identity"] = artifact_identity(metadata, arrays)
    arrays_with_metadata = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))} | arrays
    artifact_path = args.output / "rich-consequence-ensemble.npz"
    atomic_npz(artifact_path, arrays_with_metadata)
    report["artifact"] = {
        "file": artifact_path.name,
        "bytes": artifact_path.stat().st_size,
        "sha256": sha256(artifact_path),
        "artifact_identity": metadata["artifact_identity"],
    }
    atomic_json(args.output / "fit-report.json", report)
    atomic_json(args.output / "identity.json", metadata)
    atomic_json(
        args.output / "result.json",
        {
            "format": FORMAT,
            "status": report["status"],
            "artifact": report["artifact"],
            "fit_report_sha256": sha256(args.output / "fit-report.json"),
            "identity_sha256": sha256(args.output / "identity.json"),
            "validation": metadata["validation"],
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    print(json.dumps(read_json(args.output / "result.json"), indent=2, sort_keys=True))
    return 0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
