#!/usr/bin/env python3
"""Substantial two-stage full-CNS and private-context training for the fly body."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from chreatures.cns_adapter_contract import (
    load_service_artifact,
    service_identity,
    write_service_artifact,
)
from chreatures.sequence_control import CNS_DEPENDENCY_KEYS
from research.anatomical_cns.model import AnatomicalCNS, CNSState, export_arrays
from research.resident_learning.artifact import load_parent, publish_trained
from research.resident_learning.model import CnsResidentModel

from .curriculum import CONTEXT, CONTROL_SOURCES, PHASES, RESIDENTS, TICKS
from .data import (
    BODY_AFFERENTS,
    LATENT,
    MOTOR,
    OUTCOMES,
    Corpus,
    Episode,
    combine_corpora,
    load_corpus,
    load_nursery_corpus,
    seal_corpus,
    sha256_file,
)
from .resident_objective import training_loss as resident_training_loss
from .sampling import BalancedWindowSampler, Window, slice_window


TRAINING_FORMAT = "chreatures-actual-fly-cns-development-fit-v1"
CHECKPOINT_FORMAT = "chreatures-actual-fly-cns-development-optimizer-v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(_cpu_tree(value), temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(value), device=device)


def _cns_control_step(
    model: AnatomicalCNS, optic: torch.Tensor, body: torch.Tensor,
    context: torch.Tensor, state: CNSState | None,
) -> tuple[torch.Tensor, torch.Tensor, CNSState]:
    """Advance one control tick with two internal dt/2 rate integrations."""
    return model(optic, body, context, state, dt=0.01)


def body_statistics(episodes: Iterable[Episode]) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(BODY_AFFERENTS, np.float64)
    square = np.zeros(BODY_AFFERENTS, np.float64)
    count = 0
    for episode in episodes:
        value = episode.body_afferents.reshape(-1, BODY_AFFERENTS).astype(np.float64, copy=False)
        total += value.sum(0)
        square += np.square(value).sum(0)
        count += value.shape[0]
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-10)
    # The floor is explicit in the artifact provenance and prevents unsupported
    # or constant channels from becoming arbitrarily amplified.
    scale = np.sqrt(variance).clip(1e-4, None)
    return mean.astype("<f4"), scale.astype("<f4")


@torch.inference_mode()
def replay_episode(
    model: AnatomicalCNS,
    episode: Episode,
    device: torch.device,
    *,
    collect_motor_rates: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    state: CNSState | None = None
    latents = np.empty((TICKS + 1, RESIDENTS, LATENT), np.float32)
    rates = (
        np.empty((TICKS + 1, RESIDENTS, model.motor_rows.numel()), np.float32)
        if collect_motor_rates else None
    )
    for tick in range(TICKS + 1):
        context_tick = min(tick, TICKS - 1)
        optic = _tensor(episode.optic_rgb[tick], device)
        body = _tensor(episode.body_afferents[tick], device)
        context = _tensor(episode.delivered_context[context_tick], device)
        z, _, state = _cns_control_step(model, optic, body, context, state)
        latents[tick] = z.cpu().numpy()
        if rates is not None:
            rates[tick] = model.selected_activity(state)["motor"].cpu().numpy()
    return latents, rates


def install_train_world_normalization(
    model: AnatomicalCNS,
    episodes: tuple[Episode, ...],
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    rate_sum = np.zeros(815, np.float64)
    rate_square = np.zeros(815, np.float64)
    count = 0
    for episode in episodes:
        _, rates = replay_episode(model, episode, device, collect_motor_rates=True)
        assert rates is not None
        flat = rates.reshape(-1, 815).astype(np.float64, copy=False)
        rate_sum += flat.sum(0)
        rate_square += np.square(flat).sum(0)
        count += flat.shape[0]
    reference = rate_sum / count
    variance = np.maximum(rate_square / count - np.square(reference), 0)
    raw_scale = np.sqrt(variance)
    rate_scale = raw_scale.clip(1e-4, None)
    model.set_motor_normalization(
        torch.as_tensor(reference.astype(np.float32), device=device),
        torch.as_tensor(rate_scale.astype(np.float32), device=device),
    )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            reference_rate=reference.astype("<f4"),
            rate_scale=rate_scale.astype("<f4"),
            raw_rate_std=raw_scale.astype("<f4"),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return {
        "world_split": "train worlds 0..7 only",
        "resident_ticks": count,
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "raw_std_median": float(np.median(raw_scale)),
        "raw_std_p95": float(np.quantile(raw_scale, 0.95)),
        "raw_std_max": float(raw_scale.max()),
        "scale_floor": 1e-4,
        "scale_floor_count": int(np.count_nonzero(raw_scale < 1e-4)),
        "arrays_sha256": sha256_file(output),
    }


@torch.inference_mode()
def validate_collected_parent_cache(
    model: AnatomicalCNS, episodes: tuple[Episode, ...], device: torch.device,
    ticks: int = 16,
) -> dict[str, Any]:
    """Confirm collection timing/service before changing trainable interfaces."""
    maxima: list[float] = []
    for episode in episodes:
        state: CNSState | None = None
        error = 0.0
        for tick in range(min(ticks, TICKS + 1)):
            context_tick = min(tick, TICKS - 1)
            z, _, state = _cns_control_step(
                model,
                _tensor(episode.optic_rgb[tick], device),
                _tensor(episode.body_afferents[tick], device),
                _tensor(episode.delivered_context[context_tick], device),
                state,
            )
            error = max(
                error,
                float(np.max(np.abs(z.cpu().numpy() - episode.collected_latent[tick]))),
            )
        maxima.append(error)
    maximum = max(maxima, default=0.0)
    if maximum > 5e-5:
        raise RuntimeError(
            f"collected parent latent/timing replay differs (max {maximum:.9g})"
        )
    return {
        "ticks_per_world": ticks,
        "world_max_abs_error": maxima,
        "max_abs_error": maximum,
        "tolerance": 5e-5,
        "cadence": "one dt=.01 forward with two internal dt/2 rate integrations",
    }


class PhysicalPredictionHeads(nn.Module):
    """Training-only action-conditioned consequences from CNS latent state."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(LATENT + MOTOR + CONTEXT, 512),
            nn.SiLU(),
            nn.Linear(512, 384),
            nn.SiLU(),
        )
        self.body_delta = nn.Linear(384, BODY_AFFERENTS)
        self.latent_delta = nn.Linear(384, LATENT)
        self.outcome = nn.Linear(384, OUTCOMES)
        self.reward = nn.Linear(384, 1)

    def forward(
        self, latent: torch.Tensor, delivered_motor: torch.Tensor, delivered_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(torch.cat((latent, delivered_motor, delivered_context), -1))
        return (
            self.body_delta(hidden),
            self.latent_delta(hidden),
            self.outcome(hidden),
            self.reward(hidden).squeeze(-1),
        )


@dataclass(frozen=True)
class Recipe:
    seed: int
    cns_updates: int
    resident_updates: int
    batch_size: int
    burn_in: int
    sequence: int
    cns_motor_lr: float
    cns_afferent_lr: float
    cns_dynamics_lr: float
    cns_readout_lr: float
    head_lr: float
    resident_lr: float
    weight_decay: float
    max_grad_norm: float
    checkpoint_every: int
    gradient_checkpoint_steps: bool


def configure_cns(model: AnatomicalCNS) -> dict[str, list[nn.Parameter]]:
    groups = {
        "motor": [model.motor_weight, model.motor_intercept],
        "afferent": [
            model.body_weight, model.body_bias, model.context_weight, model.context_bias,
            model.optic_spectral_logits, model.optic_gain_raw, model.optic_bias,
        ],
        "dynamics": [
            model.dynamics_recurrent_gain_raw,
            model.dynamics_release_tau_raw,
            model.dynamics_release_use_raw,
            model.dynamics_mod_gain_raw,
            model.dynamics_mod_adaptation_raw,
            model.dynamics_modulation_tau_raw,
        ],
        "readout": [
            model.readout_projection,
            model.readout_output.weight,
            model.readout_output.bias,
        ],
    }
    selected = {id(parameter) for parameters in groups.values() for parameter in parameters}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    return groups


def _stack_windows(
    episodes: tuple[Episode, ...], windows: tuple[Window, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    rows = [slice_window(episodes[window.episode_index], window) for window in windows]
    result: dict[str, torch.Tensor] = {}
    for key in (
        "optic_rgb", "body_afferents", "delivered_context", "delivered_motor", "teacher_motor",
        "teacher_valid", "target_outcome", "target_reward",
    ):
        result[key] = _tensor(np.stack([row[key] for row in rows], axis=1), device)
    return result


def _motor_loss(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = target.new_ones(MOTOR)
    weights[84:90] = 1.5
    weights[90:] = 3.0
    per = (F.smooth_l1_loss(predicted, target, beta=0.05, reduction="none") * weights).mean(-1)
    valid_f = valid.to(per.dtype)
    return (per * valid_f).sum() / valid_f.sum().clamp_min(1)


def cns_window_loss(
    model: AnatomicalCNS,
    heads: PhysicalPredictionHeads,
    arrays: dict[str, torch.Tensor],
    burn_in: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    state: CNSState | None = None
    with torch.no_grad():
        for tick in range(burn_in):
            _, _, state = _cns_control_step(model,
                arrays["optic_rgb"][tick], arrays["body_afferents"][tick],
                arrays["delivered_context"][tick], state,
            )
    state = None if state is None else state.detach()
    sequence = arrays["delivered_context"].shape[0] - burn_in
    latent, motor = [], []
    for tick in range(burn_in, burn_in + sequence + 1):
        context_tick = min(tick, burn_in + sequence - 1)
        optic_tick = arrays["optic_rgb"][tick]
        body_tick = arrays["body_afferents"][tick]
        context_value = arrays["delivered_context"][context_tick]
        if torch.is_grad_enabled() and state is not None:
            def recurrent_step(optic, body, context, *fields):
                next_z, next_motor, next_state = _cns_control_step(model,
                    optic, body, context, CNSState(*fields)
                )
                return (next_z, next_motor, *next_state.fields())

            output = checkpoint(
                recurrent_step, optic_tick, body_tick, context_value,
                *state.fields(), use_reentrant=False,
            )
            z, m, state = output[0], output[1], CNSState(*output[2:])
        else:
            z, m, state = _cns_control_step(model, optic_tick, body_tick, context_value, state)
        latent.append(z)
        motor.append(m)
    z = torch.stack(latent)
    predicted_motor = torch.stack(motor[:-1])
    delivered_motor = arrays["teacher_motor"][burn_in:]
    delivered_context = arrays["delivered_context"][burn_in:]
    physical_motor = arrays["delivered_motor"][burn_in:]
    body_delta, latent_delta, outcome, reward = heads(
        z[:-1].reshape(-1, LATENT),
        physical_motor.reshape(-1, MOTOR),
        delivered_context.reshape(-1, CONTEXT),
    )
    batch = z.shape[1]
    body_now = arrays["body_afferents"][burn_in:-1]
    body_next = arrays["body_afferents"][burn_in + 1:]
    body_target = ((body_next - body_now) / model.body_scale[None, None]).clamp(-8, 8)
    latent_target = (z[1:] - z[:-1]).detach()
    outcome_target = arrays["target_outcome"][burn_in:]
    reward_target = arrays["target_reward"][burn_in:]
    terms = {
        "motor": _motor_loss(
            predicted_motor, delivered_motor, arrays["teacher_valid"][burn_in:]
        ),
        "body_prediction": F.smooth_l1_loss(
            body_delta, body_target.reshape(-1, BODY_AFFERENTS), beta=0.05
        ),
        "latent_prediction": F.smooth_l1_loss(
            latent_delta, latent_target.reshape(-1, LATENT), beta=0.02
        ),
        "outcome_prediction": F.smooth_l1_loss(
            outcome, outcome_target.reshape(-1, OUTCOMES), beta=0.05
        ),
        "reward_prediction": F.smooth_l1_loss(
            reward, reward_target.reshape(-1), beta=0.05
        ),
    }
    terms["motor_temporal_std"] = predicted_motor.std((0, 1)).mean()
    total = (
        2.0 * terms["motor"]
        + terms["body_prediction"]
        + 0.5 * terms["latent_prediction"]
        + 0.25 * terms["outcome_prediction"]
        + 0.25 * terms["reward_prediction"]
    )
    return total, terms


@torch.inference_mode()
def evaluate_cns(
    model: AnatomicalCNS,
    heads: PhysicalPredictionHeads,
    corpus: Corpus,
    split: str,
    recipe: Recipe,
    device: torch.device,
) -> dict[str, float]:
    sampler = BalancedWindowSampler(corpus, burn_in=recipe.burn_in, optimize=recipe.sequence, seed=17, split=split)
    rows: list[dict[str, float]] = []
    model.eval(); heads.eval()
    for windows in (sampler.sample(recipe.batch_size) for _ in range(8)):
        arrays = _stack_windows(sampler.episodes, windows, device)
        total, terms = cns_window_loss(model, heads, arrays, recipe.burn_in)
        rows.append({"total": float(total), **{key: float(value) for key, value in terms.items()}})
    model.train(); heads.train()
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def _service_kwargs(parent: dict[str, Any], calibration_sha: str, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "graph_sha256": parent["graph_sha256"],
        "atlas_sha256": parent["atlas_sha256"],
        "anatomy_sha256": parent["anatomy_sha256"],
        "morphology_sha256": parent["morphology_sha256"],
        "sensory_schema_sha256": parent["sensory_schema_sha256"],
        "actuator_schema_sha256": parent["actuator_schema_sha256"],
        "motor_calibration_sha256": calibration_sha,
        "graph_source_weight_sha256": parent["graph_source_weight_sha256"],
        "training_status": "trained",
        "provenance": provenance,
    }


def _save_stage_checkpoint(
    path: Path, stage: str, update: int, recipe: Recipe, identity: dict[str, Any],
    model: nn.Module, optimizer: torch.optim.Optimizer, heads: nn.Module | None = None,
) -> None:
    atomic_torch(path, {
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "update": update,
        "recipe": asdict(recipe),
        "identity": identity,
        "model": model.state_dict(),
        "heads": None if heads is None else heads.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    })


def _cache_latents(
    model: AnatomicalCNS, corpus: Corpus, device: torch.device, root: Path
) -> dict[str, np.ndarray]:
    root.mkdir(parents=True, exist_ok=False)
    result: dict[str, np.ndarray] = {}
    episodes = (*corpus.train, *corpus.validation, *corpus.heldout)
    world_counts: dict[int, int] = {}
    for episode in episodes:
        world = int(episode.metadata["world_index"])
        world_counts[world] = world_counts.get(world, 0) + 1
    for episode in episodes:
        latent, _ = replay_episode(model, episode, device)
        world = int(episode.metadata["world_index"])
        suffix = "" if world_counts[world] == 1 else f"-{episode.sha256[:12]}"
        path = root / f"world-{world:02d}{suffix}.npy"
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, latent, allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        result[episode.sha256] = np.load(path, mmap_mode="r")
    return result


@dataclass(frozen=True)
class Consequence:
    count: int
    utility: float
    uncertainty: float


def consequence_table(episodes: tuple[Episode, ...]) -> dict[tuple[int, int], Consequence]:
    table: dict[tuple[int, int], Consequence] = {}
    for phase in range(len(PHASES)):
        for source in range(len(CONTROL_SOURCES)):
            values = np.concatenate([
                episode.reward[(episode.curriculum_phase == phase) & (episode.control_source == source)]
                for episode in episodes
            ]).astype(np.float64, copy=False)
            if values.size == 0:
                table[(phase, source)] = Consequence(1, 0.0, 1.0)
                continue
            variance = float(values.var(ddof=1)) if values.size > 1 else 0.0
            stderr = math.sqrt(max(variance, 0.0) / values.size)
            table[(phase, source)] = Consequence(
                int(values.size), float(values.mean() - 0.5 * stderr), stderr
            )
    return table


def _resident_batch(
    sampler: BalancedWindowSampler,
    windows: tuple[Window, ...],
    cache: dict[str, np.ndarray],
    consequence: dict[tuple[int, int], Consequence],
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    latent_rows, context_rows, reward_rows, reset_rows = [], [], [], []
    count_rows, utility_rows, uncertainty_rows = [], [], []
    for window in windows:
        episode = sampler.episodes[window.episode_index]
        start, stop, resident = window.history_start, window.stop, window.resident
        latent_rows.append(np.asarray(cache[episode.sha256][start : stop + 1, resident]))
        context = episode.delivered_context[start:stop, resident]
        context_rows.append(context)
        reward_rows.append(episode.reward[start:stop, resident])
        reset_rows.append(episode.reset[start : stop + 1, resident])
        c, u, s = [], [], []
        for tick in range(start, stop):
            item = consequence[(int(episode.curriculum_phase[tick, resident]), int(episode.control_source[tick, resident]))]
            c.append(item.count); u.append(item.utility); s.append(item.uncertainty)
        count_rows.append(c); utility_rows.append(u); uncertainty_rows.append(s)
    latent = np.stack(latent_rows, axis=1)
    delivered = np.stack(context_rows, axis=1)
    previous = np.zeros((delivered.shape[0] + 1, delivered.shape[1], CONTEXT), np.float32)
    previous[1:] = delivered
    return {
        "cns_latent": _tensor(latent, device),
        "previous_delivered_context": _tensor(previous, device),
        "reset": _tensor(np.stack(reset_rows, axis=1), device),
        "delivered_context": _tensor(delivered, device),
        "physical_reward": _tensor(np.stack(reward_rows, axis=1), device),
        "consequence_count": _tensor(np.asarray(count_rows, np.float32).T, device),
        "consequence_utility": _tensor(np.asarray(utility_rows, np.float32).T, device),
        "consequence_uncertainty": _tensor(np.asarray(uncertainty_rows, np.float32).T, device),
        "burn_in": sampler.burn_in,
    }


@torch.inference_mode()
def evaluate_resident(
    model: CnsResidentModel,
    sampler: BalancedWindowSampler,
    cache: dict[str, np.ndarray],
    consequence: dict[tuple[int, int], Consequence],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    rows = []
    for windows in (sampler.sample(batch_size) for _ in range(8)):
        loss = resident_training_loss(model, _resident_batch(sampler, windows, cache, consequence, device))
        rows.append({name: float(getattr(loss, name)) for name in loss.__dataclass_fields__})
    model.train()
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def train(arguments: argparse.Namespace) -> None:
    run = arguments.run.expanduser().resolve()
    run.mkdir(parents=True, exist_ok=False)
    primary = load_corpus(arguments.corpus)
    source_receipts = [{
        "role": "shared-geometry-body-bootstrap",
        "format": primary.manifest["format"],
        "path": str((primary.root / "corpus.json").resolve()),
        "sha256": sha256_file(primary.root / "corpus.json"),
    }]
    corpus = primary
    if arguments.nursery_corpus is not None:
        nursery_path = arguments.nursery_corpus.expanduser().resolve()
        nursery = load_nursery_corpus(nursery_path)
        manifest_path = nursery_path / "nursery-corpus.json" if nursery_path.is_dir() else nursery_path
        source_receipts.append({
            "role": "diverse-layout-physical-stimulus-nursery",
            "format": nursery.manifest["format"],
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        })
        corpus = combine_corpora(primary, nursery)
    source_identities = [
        {name: row[name] for name in ("role", "format", "sha256")}
        for row in source_receipts
    ]
    if len(source_identities) == 1:
        corpus_identity_sha = source_receipts[0]["sha256"]
    else:
        corpus_identity_sha = hashlib.sha256(
            json.dumps(source_identities, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    atomic_json(run / "corpus-sources.json", {
        "format": "chreatures-actual-fly-training-corpus-set-v1",
        "identity_sha256": corpus_identity_sha,
        "sources": source_receipts,
    })
    service_path = arguments.service.expanduser().resolve()
    parent_arrays, parent_metadata = load_service_artifact(service_path)
    parent_service_sha = sha256_file(service_path)
    all_episodes = (*corpus.train, *corpus.validation, *corpus.heldout)
    if any(
        episode.metadata["cns_service_sha256"] != parent_service_sha
        or episode.metadata["cns_adapter_sha256"] != parent_metadata["adapter_sha256"]
        for episode in all_episodes
    ):
        raise RuntimeError("corpus was not collected through the supplied parent CNS service")
    device = torch.device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual full-CNS training requires the isolated ROCm CUDA interface")
    random.seed(arguments.seed); np.random.seed(arguments.seed); torch.manual_seed(arguments.seed)
    recipe = Recipe(
        seed=arguments.seed,
        cns_updates=arguments.cns_updates,
        resident_updates=arguments.resident_updates,
        batch_size=arguments.batch_size,
        burn_in=arguments.burn_in,
        sequence=arguments.sequence,
        cns_motor_lr=arguments.cns_motor_lr,
        cns_afferent_lr=arguments.cns_afferent_lr,
        cns_dynamics_lr=arguments.cns_dynamics_lr,
        cns_readout_lr=arguments.cns_readout_lr,
        head_lr=arguments.head_lr,
        resident_lr=arguments.resident_lr,
        weight_decay=arguments.weight_decay,
        max_grad_norm=arguments.max_grad_norm,
        checkpoint_every=arguments.checkpoint_every,
        gradient_checkpoint_steps=True,
    )
    identity = {
        "format": TRAINING_FORMAT,
        "source_revision": arguments.source_revision,
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "data_sha256": sha256_file(Path(__file__).with_name("data.py")),
        "resident_objective_sha256": sha256_file(Path(__file__).with_name("resident_objective.py")),
        "corpus_manifest_sha256": corpus_identity_sha,
        "corpus_sources": source_identities,
        "parent_service_sha256": parent_service_sha,
        "parent_adapter_sha256": parent_metadata["adapter_sha256"],
        "parent_resident_sha256": None if arguments.parent_resident is None else sha256_file(arguments.parent_resident),
    }
    progress: dict[str, Any] = {
        "format": TRAINING_FORMAT, "status": "normalization", "stage": "cns",
        "update": 0, "recipe": asdict(recipe), "identity": identity,
    }
    atomic_json(run / "progress.json", progress)

    parent_model = AnatomicalCNS(parent_arrays, device=device).eval()
    parent_cache_check = validate_collected_parent_cache(
        parent_model, (*corpus.train, *corpus.validation, *corpus.heldout), device
    )
    del parent_model
    torch.cuda.empty_cache()
    atomic_json(run / "parent-cache-check.json", parent_cache_check)

    mutable_arrays = {name: np.asarray(value) for name, value in parent_arrays.items()}
    mean, scale = body_statistics(corpus.train)
    mutable_arrays["body.mean"] = mean
    mutable_arrays["body.scale"] = scale
    model = AnatomicalCNS(mutable_arrays, device=device)
    model.set_body_normalization(
        torch.as_tensor(mean, device=device), torch.as_tensor(scale, device=device)
    )
    normalization = install_train_world_normalization(
        model, corpus.train, device, run / "train-world-normalization.npz"
    )
    atomic_json(run / "normalization.json", normalization)
    groups = configure_cns(model)
    heads = PhysicalPredictionHeads().to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["motor"], "lr": recipe.cns_motor_lr, "name": "centered-motor"},
            {"params": groups["afferent"], "lr": recipe.cns_afferent_lr, "name": "afferents"},
            {"params": groups["dynamics"], "lr": recipe.cns_dynamics_lr, "name": "selected-dynamics"},
            {"params": groups["readout"], "lr": recipe.cns_readout_lr, "name": "readout"},
            {"params": heads.parameters(), "lr": recipe.head_lr, "name": "training-only-predictors"},
        ],
        weight_decay=recipe.weight_decay,
    )
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    train_sampler = BalancedWindowSampler(
        corpus, burn_in=recipe.burn_in, optimize=recipe.sequence,
        seed=recipe.seed, split="train",
    )
    validation_before = evaluate_cns(
        model, heads, corpus, "validation-worlds", recipe, device
    )
    progress.update(
        status="running", normalization=normalization,
        parent_cache_check=parent_cache_check, validation_before=validation_before,
    )
    atomic_json(run / "progress.json", progress)
    began = time.monotonic()
    cns_history: list[dict[str, float]] = []
    for update in range(1, recipe.cns_updates + 1):
        windows = train_sampler.sample(recipe.batch_size)
        # The shared stack includes the actually delivered physical cause for
        # action-conditioned prediction in both training and validation.
        batch = _stack_windows(train_sampler.episodes, windows, device)
        optimizer.zero_grad(set_to_none=True)
        total, terms = cns_window_loss(model, heads, batch, recipe.burn_in)
        if not torch.isfinite(total):
            raise RuntimeError("non-finite full-CNS physical loss")
        total.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, recipe.max_grad_norm))
        if not math.isfinite(gradient):
            raise RuntimeError("non-finite full-CNS gradient")
        optimizer.step()
        row = {"update": update, "total": float(total.detach()), "gradient_norm": gradient}
        row.update({name: float(value.detach()) for name, value in terms.items()})
        cns_history.append(row)
        append_jsonl(run / "cns-history.jsonl", row)
        progress.update(update=update, elapsed_seconds=time.monotonic() - began, last=row)
        atomic_json(run / "progress.json", progress)
        if update % recipe.checkpoint_every == 0 or update == recipe.cns_updates:
            checkpoint = run / f"cns-checkpoint-{update:06d}.pt"
            _save_stage_checkpoint(checkpoint, "cns", update, recipe, identity, model, optimizer, heads)
            atomic_json(run / "last-cns-checkpoint.json", {
                "update": update, "path": checkpoint.name, "sha256": sha256_file(checkpoint)
            })
        if update <= 4 or update % 16 == 0:
            print(json.dumps({"stage": "cns", **row}), flush=True)
    validation_after = evaluate_cns(model, heads, corpus, "validation-worlds", recipe, device)

    exported = export_arrays(model, mutable_arrays)
    calibration_payload = {
        "kind": "train-world-centered-motor-neuron-normalization-v1",
        **normalization,
        "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
    }
    calibration_sha = hashlib.sha256(
        json.dumps(calibration_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    service_output = run / "actual-fly-cns-development.bin"
    service_receipt = write_service_artifact(
        service_output,
        exported,
        **_service_kwargs(parent_metadata, calibration_sha, {
            "training_format": TRAINING_FORMAT,
            "parent_service_sha256": parent_service_sha,
            "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
            "world_split": "per corpus: 0..7 train, 8..9 validation, 10..11 untouched heldout",
            "model_ingress": ["optic1771 RGB", "BODY807", "delivered context12"],
            "observer_values": "targets and sampling only",
            "training_only_heads_exported": False,
            "normalization": calibration_payload,
        }),
    )
    atomic_json(run / "cns-service-receipt.json", service_receipt)

    # Recompute every latent from the final service before private-context
    # fitting.  This avoids training the resident against a stale parent cache.
    model.eval()
    latent_cache = _cache_latents(model, corpus, device, run / "latent-cache")
    resident_receipt = None
    resident_history: list[dict[str, float]] = []
    resident_validation_before = resident_validation_after = resident_heldout = None
    if arguments.parent_resident is not None:
        parent_resident_metadata, parent_resident_arrays = load_parent(arguments.parent_resident)
        resident = CnsResidentModel.from_arrays(parent_resident_arrays, device)
        resident_optimizer = torch.optim.AdamW(
            resident.parameters(), lr=recipe.resident_lr, weight_decay=recipe.weight_decay
        )
        resident_train_sampler = BalancedWindowSampler(
            corpus, burn_in=recipe.burn_in, optimize=max(recipe.sequence, 64),
            seed=recipe.seed + 1, split="train",
        )
        resident_validation_sampler = BalancedWindowSampler(
            corpus, burn_in=recipe.burn_in, optimize=max(recipe.sequence, 64),
            seed=23, split="validation-worlds",
        )
        consequences = consequence_table(corpus.train)
        resident_validation_before = evaluate_resident(
            resident, resident_validation_sampler, latent_cache, consequences,
            device, recipe.batch_size,
        )
        progress.update(stage="resident", update=0, resident_validation_before=resident_validation_before)
        atomic_json(run / "progress.json", progress)
        for update in range(1, recipe.resident_updates + 1):
            windows = resident_train_sampler.sample(recipe.batch_size)
            batch = _resident_batch(
                resident_train_sampler, windows, latent_cache, consequences, device
            )
            resident_optimizer.zero_grad(set_to_none=True)
            losses = resident_training_loss(resident, batch)
            if not torch.isfinite(losses.total):
                raise RuntimeError("non-finite private-context objective")
            losses.total.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(resident.parameters(), recipe.max_grad_norm))
            if not math.isfinite(gradient):
                raise RuntimeError("non-finite private-context gradient")
            resident_optimizer.step()
            row = {name: float(getattr(losses, name).detach()) for name in losses.__dataclass_fields__}
            row.update(update=update, gradient_norm=gradient)
            resident_history.append(row)
            append_jsonl(run / "resident-history.jsonl", row)
            progress.update(update=update, elapsed_seconds=time.monotonic() - began, last=row)
            atomic_json(run / "progress.json", progress)
            if update % recipe.checkpoint_every == 0 or update == recipe.resident_updates:
                checkpoint = run / f"resident-checkpoint-{update:06d}.pt"
                _save_stage_checkpoint(
                    checkpoint, "resident", update, recipe, identity,
                    resident, resident_optimizer,
                )
                atomic_json(run / "last-resident-checkpoint.json", {
                    "update": update, "path": checkpoint.name, "sha256": sha256_file(checkpoint)
                })
            if update <= 4 or update % 16 == 0:
                print(json.dumps({"stage": "resident", **row}), flush=True)
        resident_validation_after = evaluate_resident(
            resident, resident_validation_sampler, latent_cache, consequences,
            device, recipe.batch_size,
        )
        # Heldout uses deterministic windows once, after every parameter is frozen.
        heldout_proxy = Corpus(
            corpus.root, corpus.manifest, corpus.heldout, corpus.heldout, tuple()
        )
        heldout_sampler = BalancedWindowSampler(
            heldout_proxy, burn_in=recipe.burn_in, optimize=max(recipe.sequence, 64),
            seed=29, split="validation-worlds",
        )
        resident_heldout = evaluate_resident(
            resident, heldout_sampler, latent_cache, consequences, device, recipe.batch_size
        )
        rebound_parent = dict(parent_resident_metadata)
        trained_service_identity = service_identity(
            service_receipt["metadata"], service_receipt["file_sha256"]
        )
        rebound_parent["cns_service"] = {
            key: trained_service_identity[key] for key in CNS_DEPENDENCY_KEYS
        }
        episode_identities = [
            {"sha256": episode.sha256, "split": episode.metadata["split"]}
            for episode in corpus.train
        ]
        resident_receipt = publish_trained(
            run / "fly-context-resident.npz",
            run / "fly-sequence-control.npz",
            rebound_parent,
            resident.arrays(),
            episode_identities=episode_identities,
            training={
                "format": TRAINING_FORMAT,
                "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
                "cns_service_sha256": service_receipt["file_sha256"],
                "goal_horizon_seconds": 0.4,
                "goal_horizon_ticks": 40,
                "maximum_suffix_phases": 8,
                "control_dt_s": 0.01,
                "model_ingress": ["CNS latent512", "previous acknowledged context12", "tick/reset"],
                "privileged_values_are_targets_only": True,
            },
        )
        atomic_json(run / "resident-receipt.json", resident_receipt)

    heldout_corpus = Corpus(corpus.root, corpus.manifest, corpus.heldout, corpus.heldout, tuple())
    heldout_cns = evaluate_cns(
        model, heads, heldout_corpus, "validation-worlds", recipe, device
    )
    result = {
        "format": TRAINING_FORMAT,
        "completed": True,
        "recipe": asdict(recipe),
        "identity": identity,
        "elapsed_seconds": time.monotonic() - began,
        "normalization": normalization,
        "parent_cache_check": parent_cache_check,
        "cns": {
            "validation_before": validation_before,
            "validation_after": validation_after,
            "heldout_offline": heldout_cns,
            "service": service_receipt,
            "updates": len(cns_history),
        },
        "resident": {
            "trained": resident_receipt is not None,
            "validation_before": resident_validation_before,
            "validation_after": resident_validation_after,
            "heldout_offline": resident_heldout,
            "artifacts": resident_receipt,
            "updates": len(resident_history),
        },
        "evaluation_arms": {
            "goal_free": "trained CNS service, exact zero context12 at 100Hz",
            "learned_private_context": "same trained CNS service plus bound native resident",
            "teacher": "offline collection/evaluator reference only; never a deployable arm",
        },
        "claim": "offline developmental fit; physical competence requires a fresh matched whole-world assay",
    }
    atomic_json(run / "result.json", result)
    progress.update(
        status="complete", stage="complete", result_sha256=sha256_file(run / "result.json"),
        cns_service_sha256=service_receipt["file_sha256"], resident=resident_receipt,
    )
    atomic_json(run / "progress.json", progress)
    print(json.dumps({
        "result": str(run / "result.json"),
        "cns_service_sha256": service_receipt["file_sha256"],
        "resident": resident_receipt,
    }), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--source", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--corpus", type=Path, required=True)
    fit.add_argument("--nursery-corpus", type=Path)
    fit.add_argument("--service", type=Path, required=True)
    fit.add_argument("--parent-resident", type=Path)
    fit.add_argument("--run", type=Path, required=True)
    fit.add_argument("--source-revision", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--seed", type=int, default=20260908)
    fit.add_argument("--cns-updates", type=int, default=512)
    fit.add_argument("--resident-updates", type=int, default=512)
    fit.add_argument("--batch-size", type=int, default=2)
    fit.add_argument("--burn-in", type=int, default=40)
    fit.add_argument("--sequence", type=int, default=16)
    fit.add_argument("--cns-motor-lr", type=float, default=3e-3)
    fit.add_argument("--cns-afferent-lr", type=float, default=3e-4)
    fit.add_argument("--cns-dynamics-lr", type=float, default=2e-4)
    fit.add_argument("--cns-readout-lr", type=float, default=5e-4)
    fit.add_argument("--head-lr", type=float, default=1e-3)
    fit.add_argument("--resident-lr", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-5)
    fit.add_argument("--max-grad-norm", type=float, default=1.0)
    fit.add_argument("--checkpoint-every", type=int, default=64)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "seal":
        print(json.dumps(seal_corpus(arguments.source, arguments.output), sort_keys=True))
    else:
        if (
            arguments.source_revision is None
            or len(arguments.source_revision) != 40
            or any(character not in "0123456789abcdef" for character in arguments.source_revision)
        ):
            raise ValueError("training requires an exact full Git source revision")
        train(arguments)


if __name__ == "__main__":
    main()
