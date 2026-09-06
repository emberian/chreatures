#!/usr/bin/env python3
"""Fit an ensemble that predicts rich sensory/body consequences of candidate actions."""

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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.sensorimotor_skills.rich_data import RichNormalizer, RichPlayDataset
from research.sensorimotor_skills.rich_model import RichSensorimotorModel
from research.sensorimotor_skills.rich_prediction import (
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
    RichConsequenceEnsemble,
    artifact_identity,
    array_sha256,
    denormalize_output,
    ensemble_summary,
    normalized_input,
    tensor_bundle_sha256,
)

BOOTSTRAP_FORMAT = "chreatures-rich-sensorimotor-bootstrap-v1"
TRAIN_WORLDS = (0, 1, 2)
VALIDATION_WORLDS = (3,)


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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    if not 1 <= args.epochs <= 100 or not 32 <= args.batch_size <= 8192:
        raise SystemExit("invalid epoch or batch schedule")
    if not 0 < args.learning_rate <= 0.01 or not 0 <= args.weight_decay <= 0.1:
        raise SystemExit("invalid optimizer schedule")


def load_bootstrap(path: Path, device: torch.device):
    expected = sha256(path)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        value = torch.load(path, map_location=device, weights_only=True)
    if value.get("format") != BOOTSTRAP_FORMAT:
        raise ValueError("rich bootstrap format differs")
    model = RichSensorimotorModel().to(device)
    model.load_state_dict(value["model"], strict=True)
    model.eval().requires_grad_(False)
    normalizer = RichNormalizer.from_value(value["identity"]["normalizer"])
    return model, normalizer, value["identity"], expected


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
    candidate = np.concatenate(
        (
            episode.actions[selected_time, selected_column],
            episode.oral[selected_time, selected_column, None],
        ),
        axis=1,
    )
    x = np.ascontiguousarray(
        np.concatenate(
            (frames, episode.neural[selected_time, selected_column], previous, candidate),
            axis=1,
        ),
        dtype=np.float32,
    )
    physiology = episode.observation[..., 4447:]
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
            batch[:, INPUT_SEGMENTS["candidate_action_plus_oral"][0] :] = candidate_override[
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
        "physiology_delta": (256, 262),
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
            np.full(6, PHYSIOLOGY_DELTA_SCALE_FLOOR, dtype=np.float32),
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
    candidate_start = INPUT_SEGMENTS["candidate_action_plus_oral"][0]
    permuted_candidate = validation_x[
        rng.permutation(len(validation_x)), candidate_start:
    ]
    permuted_prediction, _, permuted_clipped = predict(
        ensemble, validation_x, input_mean, input_scale, target_mean, target_scale,
        args.batch_size, device, candidate_override=permuted_candidate,
    )

    action_change = np.mean(
        np.abs(validation_x[:, candidate_start:] - validation_x[:, 1408:1417]), axis=1
    )
    strata = {}
    for name, mask in (
        ("near_unchanged_le_0.05", action_change <= 0.05),
        ("changed_0.05_to_0.25", (action_change > 0.05) & (action_change <= 0.25)),
        ("changed_gt_0.25", action_change > 0.25),
    ):
        strata[name] = {
            "rows": int(mask.sum()),
            "action_change_mean_abs_9": float(action_change[mask].mean()) if mask.any() else None,
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
        "format": "chreatures-rich-consequence-fit-report-v1",
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
        "version": 1,
        "architecture": {
            "members": MEMBERS,
            "layers": [[INPUT_DIM, 256, "tanh"], [256, 256, "tanh"], [256, OUTPUT_DIM, "linear"]],
            "ensemble_aggregation": "arithmetic mean in raw target units",
            "disagreement": "per-coordinate population RMS member deviation in raw target units; uncalibrated",
        },
        "input": {
            "dimension": INPUT_DIM,
            "segments": INPUT_SEGMENTS,
            "normalization": "train-world mean and population standard deviation, floor 0.02, clamp [-8,8]",
            "clipped_candidate_selection_tilt": 0.0,
        },
        "output": {
            "dimension": OUTPUT_DIM,
            "segments": OUTPUT_SEGMENTS,
            "frame_code_segments": FRAME_CODE_SEGMENTS,
            "normalization": "train-world delta mean and population standard deviation; code floor 1e-3, physiology floor 1e-4",
            "physiology_forecast": "raw physiology[t] + predicted raw delta; do not clamp, return validity separately",
            "physiology_bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
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
                    ROOT / "research/sensorimotor_skills/trajectory-schema-rich-v2.json",
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
