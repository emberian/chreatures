#!/usr/bin/env python3
"""Collect body-view experience and train/evaluate a compact visual memory organ."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ACTION_NAMES = ("forward", "turn", "gaze_pitch")
OUTCOME_NAMES = ("nutrition", "contact", "distance", "effort")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


SCENARIOS = (
    {"condition": "same_ball_west", "target": [4.7, 3.65, 0.19], "approach": [-0.95, -0.12], "occluded": False, "moved": False},
    {"condition": "same_ball_east", "target": [8.85, 6.45, 0.19], "approach": [-0.85, 0.28], "occluded": False, "moved": False},
    {"condition": "same_ball_moved", "target": [6.65, 2.25, 0.19], "approach": [-0.90, -0.22], "occluded": False, "moved": True},
    {"condition": "same_ball_occluded", "target": [2.25, 5.35, 0.19], "approach": [-1.00, 0.18], "occluded": True, "moved": False},
)


def research_spec(episode: int, repeats: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from chreatures.physics import DEFAULT_SPEC

    scenario = SCENARIOS[episode % len(SCENARIOS)]
    repeat = episode // len(SCENARIOS)
    rng = np.random.default_rng(seed + episode * 104729)
    spec = json.loads(DEFAULT_SPEC.read_text())
    by_id = {entity["id"]: entity for entity in spec["entities"]}
    target_id = "violet-ball" if episode % 2 == 0 else "cyan-ball"
    other_id = "cyan-ball" if target_id == "violet-ball" else "violet-ball"
    # These are deliberately indistinguishable local objects at different
    # locations. Their IDs are written only to evaluation metadata.
    by_id["violet-ball"]["material"] = "violet"
    by_id["cyan-ball"]["material"] = "violet"
    jitter = rng.normal(0.0, 0.08, 2)
    target = np.asarray(scenario["target"], dtype=float)
    target[:2] += jitter
    by_id[target_id]["position"] = target.tolist()
    by_id[other_id]["position"] = [10.9, 0.8 + 0.18 * (repeat % 3), 0.19]

    approach = np.asarray(scenario["approach"], dtype=float)
    start = target[:2] + approach
    heading = math.atan2(-approach[1], -approach[0]) + float(rng.normal(0, 0.07))
    bodies = {body["id"]: body for body in spec["bodies"]}
    bodies["mica"]["position"] = [float(start[0]), float(start[1]), 0.10]
    bodies["mica"]["heading"] = heading
    bodies["fern"]["position"] = [0.55, 0.55, 0.10]
    bodies["pip"]["position"] = [11.45, 7.45, 0.10]
    if scenario["occluded"]:
        midpoint = (target[:2] + start) / 2
        by_id["stack-box-a"]["position"] = [float(midpoint[0]), float(midpoint[1]), 0.28]
        by_id["stack-box-a"]["material"] = "wood"
    else:
        by_id["stack-box-a"]["position"] = [0.8, 7.2, 0.16]
    spec["name"] = f"visual-memory-research-{scenario['condition']}-{repeat}"
    metadata = {
        "condition": scenario["condition"],
        "repeat": repeat,
        "heldout_path": repeat == repeats - 1,
        "target_entity_evaluation_only": target_id,
        "target_position_evaluation_only": target.tolist(),
        "occluded_evaluation_only": bool(scenario["occluded"]),
        "moved_evaluation_only": bool(scenario["moved"]),
    }
    return spec, metadata


def compact_pixels(rgb: np.ndarray) -> np.ndarray:
    height, width, channels = rgb.shape
    if channels != 3 or height % 12 or width % 16:
        raise ValueError("RGB dimensions must be divisible into a 16x12 compact grid")
    pooled = rgb.reshape(12, height // 12, 16, width // 16, 3).mean(axis=(1, 3))
    return (pooled.reshape(-1) / 255.0).astype(np.float32)


def collect(args: argparse.Namespace) -> None:
    from chreatures.physics import PhysicsWorld
    from chreatures.retinal_render import RetinalRenderer

    if args.episodes < 8 or args.episodes % len(SCENARIOS):
        raise SystemExit(f"--episodes must be >=8 and divisible by {len(SCENARIOS)}")
    if not 8 <= args.frames_per_episode <= 64:
        raise SystemExit("--frames-per-episode must be in 8..64")
    if args.width % 16 or args.height % 12:
        raise SystemExit("render width/height must be divisible by 16/12")
    repeats = args.episodes // len(SCENARIOS)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    compact, retina, actions, outcomes = [], [], [], []
    episode_ids, steps, splits, image_paths, model_times = [], [], [], [], []
    conditions, occluded, moved, hashes = [], [], [], []
    episode_metadata = []
    started = time.perf_counter()

    for episode in range(args.episodes):
        spec, evaluation = research_spec(episode, repeats, args.seed)
        world = PhysicsWorld(seed=args.seed + episode, spec=spec)
        target = evaluation["target_position_evaluation_only"]
        world.command({
            "op": "light", "x": target[0], "y": target[1], "z": 1.1,
            "intensity": 0.72, "duration": 30.0, "color": [1.0, 0.92, 0.78],
        })
        for _ in range(4):
            world.advance({}, args.dt)
        episode_metadata.append({"episode": episode, **evaluation})
        with RetinalRenderer(
            world, width=args.width, height=args.height,
            vertical_fov_degrees=args.vertical_fov,
        ) as renderer:
            for step in range(args.frames_per_episode):
                frame = renderer.render("mica")
                png = frame.png()
                relative = Path("frames") / f"episode-{episode:03d}-{step:03d}.png"
                path = args.output / relative
                path.write_bytes(png)
                sense = world.sense("mica")
                phase = 2 * math.pi * step / max(1, args.frames_per_episode - 1)
                action = {
                    "forward": float(0.34 + 0.10 * math.sin(phase + episode * 0.3)),
                    "turn": float(0.28 * math.sin(phase * 1.4 + episode * 0.7)),
                    "gaze_pitch": float(0.34 * math.sin(phase * 0.8 - 0.4)),
                }
                if evaluation["moved_evaluation_only"] and step == args.frames_per_episode // 2:
                    world.command({
                        "op": "impulse",
                        "id": evaluation["target_entity_evaluation_only"],
                        "impulse": [0.0, 0.035, 0.012],
                    })
                outcome = world.advance({"mica": action}, args.dt)["mica"]
                compact.append(compact_pixels(frame.rgb))
                retina.append(np.asarray(sense["retina3d"], dtype=np.float32).reshape(-1))
                actions.append([action[name] for name in ACTION_NAMES])
                outcomes.append([outcome[name] for name in OUTCOME_NAMES])
                episode_ids.append(episode)
                steps.append(step)
                splits.append(1 if evaluation["heldout_path"] else 0)
                image_paths.append(str(relative))
                model_times.append(frame.model_time)
                conditions.append(evaluation["condition"])
                occluded.append(evaluation["occluded_evaluation_only"])
                moved.append(evaluation["moved_evaluation_only"])
                hashes.append(hashlib.sha256(png).hexdigest())

    episode_ids_array = np.asarray(episode_ids, dtype=np.int32)
    steps_array = np.asarray(steps, dtype=np.int16)
    next_index = np.full(len(episode_ids), -1, dtype=np.int32)
    same_episode = episode_ids_array[1:] == episode_ids_array[:-1]
    next_index[:-1][same_episode] = np.arange(1, len(episode_ids), dtype=np.int32)[same_episode]
    dataset_path = args.output / "experience.npz"
    np.savez_compressed(
        dataset_path,
        compact=np.asarray(compact, dtype=np.float32),
        retina=np.asarray(retina, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        outcomes=np.asarray(outcomes, dtype=np.float32),
        episode=episode_ids_array,
        step=steps_array,
        next_index=next_index,
        split=np.asarray(splits, dtype=np.uint8),
        image_path=np.asarray(image_paths),
        model_time=np.asarray(model_times, dtype=np.float64),
        eval_condition=np.asarray(conditions),
        eval_occluded=np.asarray(occluded, dtype=np.bool_),
        eval_moved=np.asarray(moved, dtype=np.bool_),
        frame_sha256=np.asarray(hashes),
    )
    manifest = {
        "format": "chreatures-visual-experience-v1",
        "created_unix": time.time(),
        "seed": args.seed,
        "episodes": args.episodes,
        "frames_per_episode": args.frames_per_episode,
        "views": len(episode_ids),
        "render": {"width": args.width, "height": args.height, "vertical_fov": args.vertical_fov},
        "actions": list(ACTION_NAMES),
        "outcomes": list(OUTCOME_NAMES),
        "collection_policy": "fixed exploratory forward/turn/gaze waves; research data only",
        "information_boundary": "training arrays contain pixels, ray retina, action, outcome and temporal order; geometry fields below are evaluation only",
        "evaluation_only": episode_metadata,
        "dataset": {"path": dataset_path.name, "bytes": dataset_path.stat().st_size, "sha256": sha256(dataset_path)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def load_dataset(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "chreatures-visual-experience-v1":
        raise ValueError("unsupported visual experience dataset")
    path = directory / manifest["dataset"]["path"]
    if sha256(path) != manifest["dataset"]["sha256"]:
        raise ValueError("visual experience dataset checksum mismatch")
    archive = np.load(path, allow_pickle=False)
    return {name: archive[name] for name in archive.files}, manifest


def native_vlm_features(
    dataset: dict[str, np.ndarray], directory: Path, model_path: Path,
    device: str, dtype_name: str, batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, dtype_name)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, dtype=dtype
    ).to(device)
    model.eval()
    rows = []
    native_rows_per_view: int | None = None
    started = time.perf_counter()
    peak_before = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    try:
        for start in range(0, len(dataset["image_path"]), batch_size):
            paths = dataset["image_path"][start : start + batch_size]
            images = [Image.open(directory / str(path)).convert("RGB") for path in paths]
            inputs = processor.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
            mask = inputs.get("pixel_attention_mask")
            if mask is not None:
                mask = mask.to(device)
            with torch.inference_mode():
                values = model.get_image_features(pixel_values, mask)
                values = values.float().mean(dim=tuple(range(1, values.ndim - 1)))
                if values.shape[0] % len(images):
                    raise RuntimeError(
                        "native image-feature rows do not group by input image"
                    )
                # SmolVLM may flatten the global image and adaptive tiles into
                # the leading dimension. Preserve one native representation per
                # actual view by pooling exactly that image's returned rows.
                rows_per_view = values.shape[0] // len(images)
                if native_rows_per_view not in (None, rows_per_view):
                    raise RuntimeError("native tiling changed within one feature run")
                native_rows_per_view = rows_per_view
                values = values.reshape(len(images), rows_per_view, values.shape[-1]).mean(dim=1)
            rows.append(values.cpu().numpy())
            for image in images:
                image.close()
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    features = np.concatenate(rows).astype(np.float32)
    metadata = {
        "model_path": str(model_path.resolve()),
        "device": device,
        "dtype": dtype_name,
        "torch": torch.__version__,
        "views": int(features.shape[0]),
        "dimension": int(features.shape[1]),
        "native_rows_per_view": native_rows_per_view,
        "batch_size": batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(max(peak_before, torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0),
    }
    return features, metadata


def train_projector(
    features: np.ndarray, dataset: dict[str, np.ndarray], *, device: str,
    latent_dim: int, epochs: int, seed: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    transition = np.flatnonzero((dataset["next_index"] >= 0) & (dataset["split"] == 0))
    next_indices = dataset["next_index"][transition]
    mean = features[dataset["split"] == 0].mean(axis=0)
    scale = features[dataset["split"] == 0].std(axis=0)
    scale = np.maximum(scale, 1e-4)
    x_all = torch.as_tensor((features - mean) / scale, dtype=torch.float32, device=device)
    current = torch.as_tensor(transition, dtype=torch.long, device=device)
    following = torch.as_tensor(next_indices, dtype=torch.long, device=device)
    action = torch.as_tensor(dataset["actions"][transition], dtype=torch.float32, device=device)
    outcome_np = dataset["outcomes"][transition]
    outcome_mean = outcome_np.mean(axis=0)
    outcome_scale = np.maximum(outcome_np.std(axis=0), 1e-4)
    outcome = torch.as_tensor((outcome_np - outcome_mean) / outcome_scale, dtype=torch.float32, device=device)

    projection = torch.nn.Linear(features.shape[1], latent_dim, bias=False, device=device)
    dynamics = torch.nn.Linear(latent_dim + action.shape[1], latent_dim, device=device)
    outcome_head = torch.nn.Linear(latent_dim + action.shape[1], outcome.shape[1], device=device)
    optimizer = torch.optim.Adam(
        [*projection.parameters(), *dynamics.parameters(), *outcome_head.parameters()], lr=0.008
    )
    started = time.perf_counter()
    final = {}
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        z = functional.normalize(projection(x_all[current]), dim=1)
        z_next = functional.normalize(projection(x_all[following]), dim=1)
        logits = z @ z_next.T / 0.12
        labels = torch.arange(len(transition), device=device)
        contrastive = 0.5 * (
            functional.cross_entropy(logits, labels)
            + functional.cross_entropy(logits.T, labels)
        )
        joined = torch.cat((z, action), dim=1)
        predicted_next = functional.normalize(dynamics(joined), dim=1)
        dynamics_loss = (1.0 - (predicted_next * z_next.detach()).sum(dim=1)).mean()
        outcome_loss = functional.mse_loss(outcome_head(joined), outcome)
        loss = contrastive + 0.8 * dynamics_loss + 0.20 * outcome_loss
        loss.backward()
        optimizer.step()
        final = {
            "loss": float(loss.detach()),
            "contrastive": float(contrastive.detach()),
            "dynamics": float(dynamics_loss.detach()),
            "outcome": float(outcome_loss.detach()),
        }
    with torch.inference_mode():
        learned = functional.normalize(projection(x_all), dim=1).cpu().numpy().astype(np.float32)
    raw_outcome_weight = outcome_head.weight.detach().cpu().numpy() * outcome_scale[:, None]
    raw_outcome_bias = outcome_head.bias.detach().cpu().numpy() * outcome_scale + outcome_mean
    weights = {
        "input_mean": mean.tolist(),
        "input_scale": scale.tolist(),
        "projection": projection.weight.detach().cpu().numpy().tolist(),
        "dynamics_weight": dynamics.weight.detach().cpu().numpy().tolist(),
        "dynamics_bias": dynamics.bias.detach().cpu().numpy().tolist(),
        "outcome_weight": raw_outcome_weight.tolist(),
        "outcome_bias": raw_outcome_bias.tolist(),
    }
    metadata = {
        "epochs": epochs,
        "latent_dim": latent_dim,
        "train_transitions": len(transition),
        "elapsed_seconds": time.perf_counter() - started,
        "device": device,
        "final_loss": final,
    }
    return weights, metadata, learned


def normalized_representation(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mean = values[train_mask].mean(axis=0)
    scale = np.maximum(values[train_mask].std(axis=0), 1e-5)
    result = np.clip((values - mean) / scale, -8, 8)
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    return (result / np.maximum(norm, 1e-8)).astype(np.float32)


def learned_prediction_metrics(
    learned: np.ndarray, weights: dict[str, Any], dataset: dict[str, np.ndarray]
) -> dict[str, Any]:
    indices = np.flatnonzero((dataset["split"] == 1) & (dataset["next_index"] >= 0))
    joined = np.concatenate((learned[indices], dataset["actions"][indices]), axis=1)
    dynamics_weight = np.asarray(weights["dynamics_weight"], dtype=np.float32)
    dynamics_bias = np.asarray(weights["dynamics_bias"], dtype=np.float32)
    predicted = joined @ dynamics_weight.T + dynamics_bias
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-8)
    actual = learned[dataset["next_index"][indices]]
    outcome_weight = np.asarray(weights["outcome_weight"], dtype=np.float32)
    outcome_bias = np.asarray(weights["outcome_bias"], dtype=np.float32)
    predicted_outcome = joined @ outcome_weight.T + outcome_bias
    return {
        "heldout_transitions": len(indices),
        "next_representation_cosine_mean": float(
            np.mean(np.sum(predicted * actual, axis=1))
        ),
        "experienced_outcome_mae": float(
            np.mean(np.abs(predicted_outcome - dataset["outcomes"][indices]))
        ),
    }


def path_context(
    actions: np.ndarray, outcomes: np.ndarray, episodes: np.ndarray,
    *, context_dim: int = 24, seed: int = 7301,
) -> np.ndarray:
    """Encode only recurrent motor/outcome history; never evaluation positions."""

    from chreatures.visual_memory import _path_matrices

    recurrent, drive = _path_matrices(
        context_dim, actions.shape[1], outcomes.shape[1], seed
    )
    result = np.zeros((len(actions), context_dim), dtype=np.float32)
    context = np.zeros(context_dim, dtype=np.float32)
    previous = -1
    for index, episode in enumerate(episodes):
        if int(episode) != previous:
            context[:] = 0
            previous = int(episode)
        result[index] = context
        sensory_consequence = np.concatenate((actions[index], outcomes[index]))
        context = np.tanh(recurrent @ context + drive @ sensory_consequence)
    return result


def retrieval_metrics(values: np.ndarray, dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    episode = dataset["episode"]
    step = dataset["step"]
    heldout = dataset["split"] == 1
    transition = dataset["next_index"] >= 0
    bind_cutoff = max(3, int(step.max() + 1) // 3)
    memory = np.flatnonzero((~heldout & transition) | (heldout & transition & (step < bind_cutoff)))
    queries = np.flatnonzero(heldout & transition & (step >= bind_cutoff))
    contexts = path_context(dataset["actions"], dataset["outcomes"], episode)

    def run(use_context: bool, similarity: str) -> dict[str, Any]:
        next_cosine, outcome_error, condition_match, episode_match = [], [], [], []
        occluded_match, moved_match = [], []
        for query in queries:
            delta = values[memory] - values[query]
            if similarity == "cosine":
                distance = 1.0 - np.clip(values[memory] @ values[query], -1, 1)
            elif similarity == "euclidean":
                distance = np.linalg.norm(delta, axis=1)
            elif similarity == "manhattan":
                distance = np.abs(delta).sum(axis=1)
            else:
                raise ValueError(f"unknown similarity {similarity}")
            # Put each visual metric on a comparable per-query scale before
            # adding bodily relation terms. This does not change visual-only
            # ranks and uses no labels or privileged state.
            distance = distance / max(float(np.median(distance)), 1e-6)
            distance += 0.10 * np.mean(
                (dataset["actions"][memory] - dataset["actions"][query]) ** 2, axis=1
            )
            if use_context:
                distance += 0.35 * np.mean(
                    (contexts[memory] - contexts[query]) ** 2, axis=1
                )
            neighbor = memory[int(np.argmin(distance))]
            query_next = int(dataset["next_index"][query])
            neighbor_next = int(dataset["next_index"][neighbor])
            next_cosine.append(float(values[query_next] @ values[neighbor_next]))
            outcome_error.append(float(np.mean(np.abs(dataset["outcomes"][query] - dataset["outcomes"][neighbor]))))
            matched = bool(dataset["eval_condition"][query] == dataset["eval_condition"][neighbor])
            condition_match.append(matched)
            episode_match.append(bool(episode[query] == episode[neighbor]))
            if dataset["eval_occluded"][query]:
                occluded_match.append(matched)
            if dataset["eval_moved"][query]:
                moved_match.append(matched)
        return {
            "queries": len(queries),
            "next_representation_cosine_mean": float(np.mean(next_cosine)),
            "experienced_outcome_mae": float(np.mean(outcome_error)),
            "evaluation_condition_top1": float(np.mean(condition_match)),
            "evaluation_episode_top1": float(np.mean(episode_match)),
            "evaluation_occluded_condition_top1": float(np.mean(occluded_match)) if occluded_match else None,
            "evaluation_moved_condition_top1": float(np.mean(moved_match)) if moved_match else None,
        }

    return {
        similarity: {
            "visual_only": run(False, similarity),
            "with_path_context": run(True, similarity),
        }
        for similarity in ("cosine", "euclidean", "manhattan")
    }


def extract_train(args: argparse.Namespace) -> None:
    dataset, manifest = load_dataset(args.dataset)
    feature_path = args.output / "smolvlm-features.npy"
    args.output.mkdir(parents=True, exist_ok=True)
    previous_report_path = args.output / "report.json"
    if args.reuse_features:
        if not feature_path.exists():
            raise SystemExit(f"--reuse-features requested but missing {feature_path}")
        features = np.load(feature_path, allow_pickle=False)
        if features.shape != (len(dataset["image_path"]), 960):
            raise SystemExit(f"unexpected saved native feature shape {features.shape}")
        previous = (
            json.loads(previous_report_path.read_text())
            if previous_report_path.exists() else {}
        )
        extraction = dict(previous.get("extraction", {}))
        extraction["reused_existing_artifact"] = True
    else:
        features, extraction = native_vlm_features(
            dataset, args.dataset, args.model_path, args.device, args.dtype,
            args.batch_size,
        )
        np.save(feature_path, features, allow_pickle=False)
    weights, training, learned = train_projector(
        features, dataset, device=args.device, latent_dim=args.latent_dim,
        epochs=args.epochs, seed=args.seed,
    )
    train_mask = dataset["split"] == 0
    fixed_indices = np.linspace(0, features.shape[1] - 1, 64).round().astype(int)
    representations = {
        "compact_pixels_576": normalized_representation(dataset["compact"], train_mask),
        "ray_retina_320": normalized_representation(dataset["retina"], train_mask),
        "smolvlm_full_960": normalized_representation(features, train_mask),
        "smolvlm_fixed_indices_64": normalized_representation(features[:, fixed_indices], train_mask),
        f"smolvlm_learned_{args.latent_dim}": learned,
    }
    metrics = {name: retrieval_metrics(values, dataset) for name, values in representations.items()}
    prediction_metrics = learned_prediction_metrics(learned, weights, dataset)
    native_encoder_version = args.model_version
    projection_version = "projection-" + hashlib.sha256(canonical(weights)).hexdigest()[:16]
    artifact = {
        "format": "chreatures-visual-weights-v1",
        "created_unix": time.time(),
        "source_dataset_sha256": manifest["dataset"]["sha256"],
        "feature_artifact": {"path": feature_path.name, "bytes": feature_path.stat().st_size, "sha256": sha256(feature_path)},
        "action_names": list(ACTION_NAMES),
        "outcome_names": list(OUTCOME_NAMES),
        "native_encoder_version": native_encoder_version,
        "projection_version": projection_version,
        "weights": weights,
        "training": training,
        "extraction": extraction,
        "information_boundary": "weights use views, actions, outcomes and temporal adjacency only; evaluation geometry labels are excluded",
        "episode_contract": "persist full raw native features with native encoder and projection versions; reproject on recall",
    }
    weights_path = args.output / "visual-weights.json"
    write_json(weights_path, artifact)
    from chreatures.visual_memory import VisualMemory

    transition_indices = np.flatnonzero(dataset["next_index"] >= 0)
    memory = VisualMemory.from_artifact(
        weights_path, capacity=max(512, len(transition_indices))
    )
    previous_episode = -1
    for index in transition_indices:
        episode = int(dataset["episode"][index])
        if episode != previous_episode:
            memory.reset_path()
            previous_episode = episode
        memory.bind(
            features[index],
            {
                name: float(value)
                for name, value in zip(ACTION_NAMES, dataset["actions"][index])
            },
            {
                name: float(value)
                for name, value in zip(OUTCOME_NAMES, dataset["outcomes"][index])
            },
            features[int(dataset["next_index"][index])],
            model_time=float(dataset["model_time"][index]),
            source="experienced",
        )
    memory_value = memory.snapshot()
    memory_value["research_provenance"] = {
        "kind": "research_world_experience",
        "source_dataset_sha256": manifest["dataset"]["sha256"],
        "resident_identity": None,
        "evaluation_geometry_in_memory": False,
    }
    memory_path = args.output / "research-memory.json"
    write_json(memory_path, memory_value)
    report = {
        "format": "chreatures-visual-development-v1",
        "source_manifest": str((args.dataset / "manifest.json").resolve()),
        "source_dataset_sha256": manifest["dataset"]["sha256"],
        "extraction": extraction,
        "training": training,
        "metrics": metrics,
        "learned_prediction": prediction_metrics,
        "fixed_feature_indices": fixed_indices.tolist(),
        "weights": {"path": weights_path.name, "bytes": weights_path.stat().st_size, "sha256": sha256(weights_path)},
        "feature_artifact": artifact["feature_artifact"],
        "private_memory": {
            "path": memory_path.name,
            "bytes": memory_path.stat().st_size,
            "sha256": sha256(memory_path),
            "bound_transitions": len(transition_indices),
            "sources": ["experienced"],
        },
        "native_encoder_version": native_encoder_version,
        "projection_version": projection_version,
        "evaluation_note": "condition, moved and occluded labels are used only in these metrics, never projector or memory training",
        "context_note": "temporal context is a fixed recurrent encoding of action and experienced outcome history; it receives no world coordinates or object IDs",
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect", help="render actual research-world experience")
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--episodes", type=int, default=20)
    collection.add_argument("--frames-per-episode", type=int, default=16)
    collection.add_argument("--width", type=int, default=192)
    collection.add_argument("--height", type=int, default=144)
    collection.add_argument("--vertical-fov", type=float, default=82.0)
    collection.add_argument("--dt", type=float, default=0.08)
    collection.add_argument("--seed", type=int, default=20260905)
    collection.set_defaults(run=collect)

    training = commands.add_parser("extract-train", help="batch native VLM features and train the relational model")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--model-path", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--device", default="cuda")
    training.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--latent-dim", type=int, default=32)
    training.add_argument("--epochs", type=int, default=300)
    training.add_argument("--seed", type=int, default=7301)
    training.add_argument(
        "--model-version",
        default="smolvlm2-500m@7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    )
    training.add_argument(
        "--reuse-features", action="store_true",
        help="reuse the checksummed native feature array already in --output",
    )
    training.set_defaults(run=extract_train)
    return result


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
