#!/usr/bin/env python3
"""ROCm training for the actual-physics anatomical MaleCNS V3 bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from chreatures.cns_adapter_contract import (
    ARRAY_SPECS,
    FORMAT as SERVICE_FORMAT,
    MAGIC,
    adapter_identity_payload,
    canonical,
    validate_arrays,
    write_service_artifact,
)
from research.anatomical_cns.data import (
    BODY,
    CONTEXT,
    MOTOR,
    OPTIC,
    RESIDENTS,
    SKILLS,
    ContractError,
    Episode,
    load_corpus,
    seal_corpus,
    sha256_file,
    skill_windows,
)
from research.anatomical_cns.model import AnatomicalCNS, export_arrays


TRAINING_FORMAT = "chreatures-anatomical-cns-physical-bootstrap-v1"
CHECKPOINT_FORMAT = "chreatures-anatomical-cns-physical-optimizer-v1"


def load_service(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("current CHCNS3 service artifact required")
        size = struct.unpack("<I", stream.read(4))[0]
        if not 0 < size < 8 * 1024**2:
            raise ValueError("invalid CNS metadata length")
        metadata = json.loads(stream.read(size))
    if metadata.get("format") != SERVICE_FORMAT:
        raise ValueError("CNS service format differs")
    if hashlib.sha256(canonical(adapter_identity_payload(metadata))).hexdigest() != metadata.get("adapter_sha256"):
        raise ValueError("CNS adapter identity differs")
    offset, arrays = 12 + size, {}
    for name, dtype, shape in ARRAY_SPECS:
        value = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=shape)
        if hashlib.sha256(value.tobytes()).hexdigest() != metadata["array_sha256"][name]:
            raise ValueError(f"CNS tensor checksum differs: {name}")
        arrays[name] = value
        offset += value.nbytes
    if path.stat().st_size != offset:
        raise ValueError("CNS artifact has trailing or missing bytes")
    validate_arrays(arrays)
    return metadata, arrays


class TrainingHeads(nn.Module):
    """Training-only probes that couple current CNS state to actual next senses."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(512 + MOTOR, 256), nn.SiLU(), nn.Linear(256, 192), nn.SiLU())
        self.future_sensory = nn.Linear(192, BODY + 36)
        self.future_pose = nn.Linear(192, 12)
        self.joint_target = nn.Sequential(nn.Linear(512, 96), nn.SiLU(), nn.Linear(96, 12))

    def forward(self, latent: torch.Tensor, motor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(torch.cat((latent, motor), dim=-1))
        return self.future_sensory(hidden), self.future_pose(hidden), self.joint_target(latent)


def optic_summary(optic: torch.Tensor) -> torch.Tensor:
    """Fixed 12-region RGB means, retaining only actual retinal measurements."""
    # [B,1771,3] -> [B,12,3]. The final region receives the remainder.
    boundaries = torch.linspace(0, 1771, 13, device=optic.device).round().long()
    return torch.cat([optic[:, boundaries[i] : boundaries[i + 1]].mean(1) for i in range(12)], dim=-1)


def normalize_pose_delta(current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    delta = future - current
    return torch.cat((20 * delta[:, :3], 2 * delta[:, 3:]), dim=-1)


def target_sensory_delta(
    body: torch.Tensor, next_body: torch.Tensor, optic: torch.Tensor, next_optic: torch.Tensor
) -> torch.Tensor:
    body_delta = next_body - body
    # Small physical changes remain visible to the objective without allowing
    # any future value into the recurrent model input.
    return torch.cat((5 * body_delta, 2 * (optic_summary(next_optic) - optic_summary(optic))), dim=-1)


def motor_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Antagonist channels carry the physical bootstrap; ancillary channels are
    # retained but cannot dominate the loss through their mostly-zero targets.
    weights = target.new_tensor([1.0] * 24 + [0.35] * 2 + [0.2] * 8)
    return (F.smooth_l1_loss(prediction, target, reduction="none", beta=.08) * weights).mean()


def set_trainable(model: AnatomicalCNS) -> list[str]:
    names = {
        "body_weight",
        "body_bias",
        "motor_weight_raw",
        "motor_bias",
        "dynamics_recurrent_gain_raw",
        "dynamics_release_tau_raw",
        "dynamics_release_use_raw",
        "dynamics_mod_gain_raw",
        "dynamics_mod_adaptation_raw",
        "dynamics_modulation_tau_raw",
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    missing = names - dict(model.named_parameters()).keys()
    if missing:
        raise RuntimeError(f"V3 model trainable interface differs: {sorted(missing)}")
    return sorted(names)


def training_body_statistics(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    count = np.float64(0)
    total = np.zeros(BODY, np.float64)
    square = np.zeros(BODY, np.float64)
    for episode in episodes:
        values = episode.body.astype(np.float64, copy=False).reshape(-1, BODY)
        total += values.sum(0); square += np.square(values).sum(0); count += len(values)
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-6)
    return mean.astype("<f4"), np.sqrt(variance).clip(.01, None).astype("<f4")


@dataclass(frozen=True)
class Recipe:
    seed: int
    updates: int
    sequence: int
    burn_in: int
    motor_learning_rate: float
    body_learning_rate: float
    dynamics_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    max_grad_norm: float
    checkpoint_every: int
    motor_weight: float
    sensory_weight: float
    pose_weight: float
    joint_target_weight: float


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(value), device=device)


def run_window(
    model: AnatomicalCNS,
    heads: TrainingHeads,
    episode: Episode,
    resident: int,
    start: int,
    sequence: int,
    burn_in: int,
    device: torch.device,
    diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    first = max(0, start - burn_in)
    state = None
    with torch.no_grad():
        for tick in range(first, start):
            optic = _tensor(episode.optic_rgb[tick, resident : resident + 1].reshape(1, 1771, 3), device)
            body = _tensor(episode.body[tick, resident : resident + 1], device)
            context = _tensor(episode.delivered_context[tick, resident : resident + 1], device)
            _, _, state = model(optic, body, context, state)
    if state is not None:
        state = state.detach()
    values: dict[str, list[torch.Tensor]] = {name: [] for name in ("motor", "sensory", "pose", "joint", "persistence_sensory", "persistence_pose")}
    predicted_motors: list[torch.Tensor] = []
    for tick in range(start, start + sequence):
        optic = _tensor(episode.optic_rgb[tick, resident : resident + 1].reshape(1, 1771, 3), device)
        next_optic = _tensor(episode.optic_rgb[tick + 1, resident : resident + 1].reshape(1, 1771, 3), device)
        body = _tensor(episode.body[tick, resident : resident + 1], device)
        next_body = _tensor(episode.body[tick + 1, resident : resident + 1], device)
        context = _tensor(episode.delivered_context[tick, resident : resident + 1], device)
        target_motor = _tensor(episode.delivered_motor[tick, resident : resident + 1], device)
        target_joint = _tensor(episode.teacher_joint_target[tick, resident : resident + 1], device)
        pose = _tensor(episode.root_pose[tick, resident : resident + 1], device)
        next_pose = _tensor(episode.root_pose[tick + 1, resident : resident + 1], device)
        latent, predicted_motor, state = model(optic, body, context, state)
        # The recorded next state was caused by the delivered teacher motor.
        # Condition the temporal model on that experienced intervention; using
        # the currently predicted motor here would pair the wrong action with
        # the physical outcome and corrupt both objectives.
        predicted_sensory, predicted_pose, predicted_joint = heads(latent, target_motor)
        sensory_target = target_sensory_delta(body, next_body, optic, next_optic)
        pose_target = normalize_pose_delta(pose, next_pose)
        values["motor"].append(motor_loss(predicted_motor, target_motor))
        values["sensory"].append(F.smooth_l1_loss(predicted_sensory, sensory_target, beta=.05))
        values["pose"].append(F.smooth_l1_loss(predicted_pose, pose_target, beta=.05))
        values["joint"].append(F.smooth_l1_loss(predicted_joint, target_joint, beta=.08))
        values["persistence_sensory"].append(F.smooth_l1_loss(torch.zeros_like(sensory_target), sensory_target, beta=.05))
        values["persistence_pose"].append(F.smooth_l1_loss(torch.zeros_like(pose_target), pose_target, beta=.05))
        if diagnostics:
            predicted_motors.append(predicted_motor.detach())
    result = {name: torch.stack(items).mean() for name, items in values.items()}
    if diagnostics:
        result["_motor_values"] = torch.cat(predicted_motors, dim=0)
    return result


def combined_loss(losses: dict[str, torch.Tensor], recipe: Recipe) -> torch.Tensor:
    return (
        recipe.motor_weight * losses["motor"]
        + recipe.sensory_weight * losses["sensory"]
        + recipe.pose_weight * losses["pose"]
        + recipe.joint_target_weight * losses["joint"]
    )


def fixed_evaluation_windows(episodes: list[Episode], sequence: int) -> list[tuple[Episode, int, int]]:
    result = []
    for episode in episodes:
        for skill in range(len(SKILLS)):
            windows = skill_windows([episode], skill, sequence)
            if not windows:
                raise ContractError(f"heldout episode lacks {SKILLS[skill]} window")
            selected = windows[len(windows) // 2]
            result.append((selected[0], skill % RESIDENTS, selected[1]))
    return result


@torch.no_grad()
def evaluate(
    model: AnatomicalCNS,
    heads: TrainingHeads,
    episodes: list[Episode],
    recipe: Recipe,
    device: torch.device,
) -> dict[str, Any]:
    model.eval(); heads.eval()
    rows = []
    for episode, resident, start in fixed_evaluation_windows(episodes, recipe.sequence):
        losses = run_window(model, heads, episode, resident, start, recipe.sequence, recipe.burn_in, device, diagnostics=True)
        motors = losses.pop("_motor_values")
        rows.append({name: float(value) for name, value in losses.items()} | {
            "motor_mean": motors.mean(0).cpu().tolist(),
            "motor_min": float(motors.min()), "motor_max": float(motors.max()),
            "motor_low_fraction": float((motors < .01).float().mean()),
            "motor_high_fraction": float((motors > .99).float().mean()),
        })
    loss_names = ("motor", "sensory", "pose", "joint", "persistence_sensory", "persistence_pose")
    mean = {name: float(np.mean([row[name] for row in rows])) for name in loss_names}
    mean["total"] = float(
        recipe.motor_weight * mean["motor"] + recipe.sensory_weight * mean["sensory"]
        + recipe.pose_weight * mean["pose"] + recipe.joint_target_weight * mean["joint"]
    )
    vectors = np.asarray([row["motor_mean"] for row in rows], np.float64).reshape(len(episodes), len(SKILLS), MOTOR)
    skill_means = vectors.mean(0)
    pairwise = [float(np.sqrt(np.mean(np.square(skill_means[a] - skill_means[b]))))
                for a in range(4) for b in range(a + 1, 4)]
    responsiveness = {
        "motor_mean_by_skill": {skill: skill_means[index].tolist() for index, skill in enumerate(SKILLS)},
        "four_tone_skill_pairwise_rms_mean": float(np.mean(pairwise)),
        "four_tone_skill_pairwise_rms_min": float(np.min(pairwise)),
        "window_motor_min": float(min(row["motor_min"] for row in rows)),
        "window_motor_max": float(max(row["motor_max"] for row in rows)),
        "low_saturation_fraction": float(np.mean([row["motor_low_fraction"] for row in rows])),
        "high_saturation_fraction": float(np.mean([row["motor_high_fraction"] for row in rows])),
        "warning": "fixed collection bout order confounds tone and physical state; counterbalanced physical assay required for tone mapping",
    }
    model.train(); heads.train()
    return {"overall": mean, "responsiveness": responsiveness, "windows": rows, "whole_world_split": "heldout-worlds"}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(
    path: Path,
    model: AnatomicalCNS,
    heads: TrainingHeads,
    optimizer: torch.optim.Optimizer,
    update: int,
    recipe: Recipe,
    identity: dict[str, Any],
    history: list[dict[str, float]],
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "update": update,
        "recipe": asdict(recipe),
        "identity": identity,
        "model": cpu_tree(model.state_dict()),
        "training_heads": cpu_tree(heads.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "history": history,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def command_train(arguments: argparse.Namespace) -> None:
    run = arguments.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    train_episodes, heldout_episodes = load_corpus(arguments.corpus)
    parent_meta, parent_arrays = load_service(arguments.service.resolve())
    parent_file_sha = sha256_file(arguments.service.resolve())
    random.seed(arguments.seed); np.random.seed(arguments.seed); torch.manual_seed(arguments.seed)
    # Normalization is fit using training worlds only and becomes part of the
    # exported body interface. Heldout bodies never select this operating point.
    body_mean, body_scale = training_body_statistics(train_episodes)
    arrays = dict(parent_arrays); arrays["body.mean"] = body_mean; arrays["body.scale"] = body_scale
    device = torch.device(arguments.device)
    model = AnatomicalCNS(arrays, device=device)
    trainable = set_trainable(model)
    heads = TrainingHeads().to(device)
    recipe = Recipe(
        seed=arguments.seed, updates=arguments.updates, sequence=arguments.sequence,
        burn_in=arguments.burn_in, motor_learning_rate=arguments.motor_learning_rate,
        body_learning_rate=arguments.body_learning_rate, dynamics_learning_rate=arguments.dynamics_learning_rate,
        head_learning_rate=arguments.head_learning_rate,
        weight_decay=arguments.weight_decay, max_grad_norm=arguments.max_grad_norm,
        checkpoint_every=arguments.checkpoint_every, motor_weight=arguments.motor_weight,
        sensory_weight=arguments.sensory_weight, pose_weight=arguments.pose_weight,
        joint_target_weight=arguments.joint_target_weight,
    )
    named = dict(model.named_parameters())
    motor_parameters = [named[name] for name in ("motor_weight_raw", "motor_bias")]
    body_parameters = [named[name] for name in ("body_weight", "body_bias")]
    dynamics_parameters = [named[name] for name in trainable if name.startswith("dynamics_")]
    head_parameters = list(heads.parameters())
    parameters = motor_parameters + body_parameters + dynamics_parameters + head_parameters
    optimizer = torch.optim.AdamW([
        {"params": motor_parameters, "lr": recipe.motor_learning_rate, "name": "motor"},
        {"params": body_parameters, "lr": recipe.body_learning_rate, "name": "body"},
        {"params": dynamics_parameters, "lr": recipe.dynamics_learning_rate, "name": "dynamics"},
        {"params": head_parameters, "lr": recipe.head_learning_rate, "name": "training-heads"},
    ], weight_decay=recipe.weight_decay)
    corpus_manifest = arguments.corpus.resolve() / "corpus.json"
    identity = {
        "training_format": TRAINING_FORMAT,
        "parent_service_sha256": parent_file_sha,
        "parent_adapter_sha256": parent_meta["adapter_sha256"],
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "data_sha256": sha256_file(Path(__file__).with_name("data.py")),
        "model_sha256": sha256_file(Path(__file__).with_name("model.py")),
        "trainable_parameters": trainable,
        "training_only_heads": [name for name, _ in heads.named_parameters()],
    }
    before = evaluate(model, heads, heldout_episodes, recipe, device)
    candidates = {skill: skill_windows(train_episodes, skill, recipe.sequence) for skill in range(len(SKILLS))}
    history: list[dict[str, float]] = []
    progress = {"format": TRAINING_FORMAT, "status": "running", "update": 0, "recipe": asdict(recipe),
                "identity": identity, "heldout_before": before}
    atomic_json(run / "progress.json", progress)
    began = time.monotonic()
    for update in range(1, recipe.updates + 1):
        skill = (update - 1) % len(SKILLS)
        episode, start, _ = random.choice(candidates[skill]); resident = random.randrange(RESIDENTS)
        optimizer.zero_grad(set_to_none=True)
        losses = run_window(model, heads, episode, resident, start, recipe.sequence, recipe.burn_in, device)
        total = combined_loss(losses, recipe)
        if not torch.isfinite(total):
            raise RuntimeError("non-finite physical bootstrap loss")
        total.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, recipe.max_grad_norm))
        if not math.isfinite(gradient):
            raise RuntimeError("non-finite physical bootstrap gradient")
        optimizer.step()
        row = {name: float(value.detach()) for name, value in losses.items()}
        row.update(update=update, total=float(total.detach()), gradient_norm=gradient, skill=skill)
        history.append(row)
        progress.update(update=update, elapsed_seconds=time.monotonic() - began, last=row)
        atomic_json(run / "progress.json", progress)
        if update % recipe.checkpoint_every == 0 or update == recipe.updates:
            checkpoint = run / f"checkpoint-{update:06d}.pt"
            save_checkpoint(checkpoint, model, heads, optimizer, update, recipe, identity, history)
            atomic_json(run / "last-checkpoint.json", {"update": update, "path": checkpoint.name,
                        "sha256": sha256_file(checkpoint)})
        if update <= 4 or update % 16 == 0:
            print(json.dumps({"update": update, "loss": row["total"], "motor": row["motor"],
                              "sensory": row["sensory"], "pose": row["pose"], "gradient": gradient}), flush=True)
    after = evaluate(model, heads, heldout_episodes, recipe, device)
    exported = export_arrays(model, arrays)
    artifact = run / "anatomical-cns-v3-physical-bootstrap.bin"
    artifact_receipt = write_service_artifact(
        artifact, exported, graph_sha256=parent_meta["graph_sha256"], atlas_sha256=parent_meta["atlas_sha256"],
        anatomy_sha256=parent_meta["anatomy_sha256"], training_status="trained",
        provenance={
            "training_format": TRAINING_FORMAT, "parent_service_sha256": parent_file_sha,
            "parent_adapter_sha256": parent_meta["adapter_sha256"], "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
            "world_split": "six train, two heldout-worlds", "context_source": "zero-initial-motor-bootstrap",
            "model_inputs": ["actual optic_rgb", "actual BODY110", "delivered context12"],
            "target_only": ["teacher motor34", "teacher joint target", "root pose", "future actual sensory"],
            "training_only_heads_exported": False, "heldout_used_for_selection": False,
        },
    )
    result = {
        "format": TRAINING_FORMAT, "completed": True, "recipe": asdict(recipe), "identity": identity,
        "elapsed_seconds": time.monotonic() - began, "heldout_before": before, "heldout_after": after,
        "artifact": artifact_receipt, "final_checkpoint": json.loads((run / "last-checkpoint.json").read_text()),
        "claims": "offline actual-physics motor bootstrap; learned closed-loop competence requires the separate headless assay",
    }
    atomic_json(run / "result.json", result)
    progress.update(status="complete", result_sha256=sha256_file(run / "result.json"), artifact=artifact_receipt)
    atomic_json(run / "progress.json", progress)
    print(json.dumps({"result": str(run / "result.json"), "artifact": artifact_receipt["path"],
                      "artifact_sha256": artifact_receipt["file_sha256"], "before": before["overall"],
                      "after": after["overall"]}), flush=True)


@torch.inference_mode()
def motor_activity_diagnostic(
    service: Path, episodes: list[Episode], device: torch.device
) -> dict[str, Any]:
    metadata, arrays = load_service(service)
    model = AnatomicalCNS(arrays, device=device).eval()
    weight = (F.softplus(model.motor_weight_raw) * model.motor_mask).detach()
    r0 = model.effective()[0]
    baseline_drive = (weight @ r0[model.motor_rows.long()] + model.motor_bias[:, None]).squeeze(1)
    deviations, dynamic_drives, motors = [], [], []
    for episode in episodes:
        state = None
        for tick in range(episode.delivered_motor.shape[0]):
            optic = _tensor(episode.optic_rgb[tick].reshape(RESIDENTS, 1771, 3), device)
            body = _tensor(episode.body[tick], device)
            context = _tensor(episode.delivered_context[tick], device)
            _, motor, state = model(optic, body, context, state)
            deviation = state.rates[model.motor_rows.long()] - r0[model.motor_rows.long()]
            deviations.append(deviation.T.cpu().numpy())
            dynamic_drives.append((weight @ deviation).T.cpu().numpy())
            motors.append(motor.cpu().numpy())
    deviation = np.concatenate(deviations, axis=0).astype(np.float64)
    dynamic = np.concatenate(dynamic_drives, axis=0).astype(np.float64)
    motor = np.concatenate(motors, axis=0).astype(np.float64)
    neuron_std = deviation.std(0)
    channel_std = dynamic.std(0)
    motor_std = motor.std(0)
    baseline = baseline_drive.cpu().numpy().astype(np.float64)
    bias = model.motor_bias.detach().cpu().numpy().astype(np.float64)
    result = {
        "service_sha256": sha256_file(service), "adapter_sha256": metadata["adapter_sha256"],
        "training_status": metadata["training_status"], "worlds": len(episodes),
        "resident_ticks": int(deviation.shape[0]),
        "motor_neuron_rate_deviation": {
            "rms": float(np.sqrt(np.mean(np.square(deviation)))),
            "temporal_std_median": float(np.median(neuron_std)),
            "temporal_std_p95": float(np.quantile(neuron_std, .95)),
            "temporal_std_max": float(neuron_std.max()),
            "count_std_above_1e-5": int(np.count_nonzero(neuron_std > 1e-5)),
            "count_std_above_1e-4": int(np.count_nonzero(neuron_std > 1e-4)),
            "count_std_above_1e-3": int(np.count_nonzero(neuron_std > 1e-3)),
            "neurons": int(neuron_std.size),
        },
        "decoder": {
            "bias_min": float(bias.min()), "bias_max": float(bias.max()),
            "baseline_plus_bias_min": float(baseline.min()), "baseline_plus_bias_max": float(baseline.max()),
            "dynamic_preactivation_rms": float(np.sqrt(np.mean(np.square(dynamic)))),
            "dynamic_preactivation_std_mean": float(channel_std.mean()),
            "dynamic_preactivation_std_max": float(channel_std.max()),
            "dynamic_preactivation_abs_max": float(np.abs(dynamic).max()),
            "dynamic_to_baseline_rms_ratio": float(
                np.sqrt(np.mean(np.square(dynamic))) / max(1e-12, np.sqrt(np.mean(np.square(baseline))))
            ),
            "effective_weight_mean": float(weight.mean().cpu()),
            "effective_weight_max": float(weight.max().cpu()),
        },
        "motor_output": {
            "min": float(motor.min()), "max": float(motor.max()),
            "channel_std_mean": float(motor_std.mean()), "channel_std_max": float(motor_std.max()),
        },
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def command_diagnose(arguments: argparse.Namespace) -> None:
    train, heldout = load_corpus(arguments.corpus)
    episodes = train if arguments.split == "train" else heldout
    device = torch.device(arguments.device)
    result = {
        "format": "chreatures-anatomical-cns-motor-activity-diagnostic-v1",
        "corpus_manifest_sha256": sha256_file(arguments.corpus.resolve() / "corpus.json"),
        "split": arguments.split,
        "services": [motor_activity_diagnostic(path.resolve(), episodes, device) for path in arguments.service],
        "interpretation_limit": "Activity and decoder attribution only; no motor competence or tone association claim.",
    }
    atomic_json(arguments.output.resolve(), result)
    print(json.dumps({"output": str(arguments.output.resolve()), "sha256": sha256_file(arguments.output.resolve()),
                      "services": result["services"]}, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("seal-corpus", help="authenticate and seal eight actual physical episodes")
    pack.add_argument("source", type=Path); pack.add_argument("output", type=Path)
    inspect = sub.add_parser("inspect", help="authenticate a sealed corpus")
    inspect.add_argument("corpus", type=Path)
    diagnose = sub.add_parser("diagnose-motor", help="attribute motor constancy to CNS activity and decoder drive")
    diagnose.add_argument("--corpus", type=Path, required=True)
    diagnose.add_argument("--service", type=Path, action="append", required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--device", default="cuda")
    diagnose.add_argument("--split", choices=("train", "heldout"), default="heldout")
    train = sub.add_parser("train", help="fit V3 physical interfaces on ROCm")
    train.add_argument("--corpus", type=Path, required=True); train.add_argument("--service", type=Path, required=True)
    train.add_argument("--run", type=Path, required=True); train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=20260907); train.add_argument("--updates", type=int, default=320)
    train.add_argument("--sequence", type=int, default=4); train.add_argument("--burn-in", type=int, default=8)
    train.add_argument("--motor-learning-rate", type=float, default=3e-3)
    train.add_argument("--body-learning-rate", type=float, default=1e-3)
    train.add_argument("--dynamics-learning-rate", type=float, default=3e-4)
    train.add_argument("--head-learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--max-grad-norm", type=float, default=1.0); train.add_argument("--checkpoint-every", type=int, default=40)
    train.add_argument("--motor-weight", type=float, default=1.0); train.add_argument("--sensory-weight", type=float, default=.20)
    train.add_argument("--pose-weight", type=float, default=.12); train.add_argument("--joint-target-weight", type=float, default=.18)
    return value


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "seal-corpus":
        print(json.dumps(seal_corpus(arguments.source, arguments.output), indent=2))
    elif arguments.command == "inspect":
        train, heldout = load_corpus(arguments.corpus)
        print(json.dumps({"train": [episode.sha256 for episode in train],
                          "heldout": [episode.sha256 for episode in heldout]}, indent=2))
    elif arguments.command == "diagnose-motor":
        command_diagnose(arguments)
    else:
        command_train(arguments)


if __name__ == "__main__":
    main()
