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
    ANATOMICAL_BODY_SCHEMA_SHA256,
    BODY_AFFERENTS,
    CNS_BODY807_SCHEMA_SHA256,
    LATENT,
    MOTOR,
    MORPHOLOGY_ASSET_SET_SHA256,
    OUTCOMES,
    Corpus,
    Episode,
    combine_corpora,
    load_corpus,
    load_nursery_corpus,
    load_recovery_corpus,
    load_support_acquisition_corpus,
    seal_corpus,
    sha256_file,
)
from .resident_objective import training_loss as resident_training_loss
from .sampling import BalancedWindowSampler, MixedCausalSampler, Window, slice_window


TRAINING_FORMAT = "chreatures-actual-fly-cns-development-fit-v1"
CHECKPOINT_FORMAT = "chreatures-actual-fly-cns-development-optimizer-v1"
PREDICTION_HEAD_FORMAT = "chreatures-actual-fly-physical-prediction-heads-v1"


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


def extract_prediction_heads(
    checkpoint_path: Path, result_path: Path, output: Path,
) -> dict[str, Any]:
    """Extract the trained action-conditioned surrogate with exact child binding."""
    checkpoint_path, result_path, output = (
        checkpoint_path.expanduser().resolve(), result_path.expanduser().resolve(),
        output.expanduser().resolve(),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = json.loads(result_path.read_text())
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("stage") != "cns"
        or int(checkpoint.get("update", -1)) != int(checkpoint.get("recipe", {}).get("cns_updates", -2))
        or result.get("completed") is not True
        or int(result.get("cns", {}).get("updates", -1)) != int(checkpoint["update"])
        or not isinstance(checkpoint.get("heads"), dict)
    ):
        raise RuntimeError("prediction heads require a completed final CNS checkpoint/result")
    service = result["cns"]["service"]
    service_path = result_path.parent / Path(str(service["path"])).name
    if sha256_file(service_path) != service["file_sha256"]:
        raise RuntimeError("result child service identity differs")
    payload = {
        "format": PREDICTION_HEAD_FORMAT,
        "child_service_sha256": service["file_sha256"],
        "child_adapter_sha256": service["metadata"]["adapter_sha256"],
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_result_sha256": sha256_file(result_path),
        "source_revision": checkpoint["identity"]["source_revision"],
        "state": checkpoint["heads"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch(output, payload)
    return {
        key: value for key, value in payload.items() if key != "state"
    } | {"path": str(output), "file_sha256": sha256_file(output)}


def load_prediction_heads(
    path: Path, child_service_sha256: str, device: torch.device,
) -> tuple[PhysicalPredictionHeads, dict[str, Any]]:
    path = path.expanduser().resolve()
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("format") != PREDICTION_HEAD_FORMAT
        or payload.get("child_service_sha256") != child_service_sha256
        or not isinstance(payload.get("state"), dict)
    ):
        raise RuntimeError("physical prediction heads are not bound to the train-start CNS")
    model = PhysicalPredictionHeads().to(device)
    model.load_state_dict(payload["state"], strict=True)
    identity = {key: value for key, value in payload.items() if key != "state"}
    identity["file_sha256"] = sha256_file(path)
    return model, identity


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


def preserve_parent_normalization(
    model: AnatomicalCNS,
    output: Path,
    parent_service_sha256: str,
) -> dict[str, Any]:
    """Snapshot an inherited normalization without changing decoder semantics."""
    arrays = {
        "body_mean": model.body_mean.detach().cpu().numpy().astype("<f4"),
        "body_scale": model.body_scale.detach().cpu().numpy().astype("<f4"),
        "reference_rate": model.motor_reference_rate.detach().cpu().numpy().astype("<f4"),
        "rate_scale": model.motor_rate_scale.detach().cpu().numpy().astype("<f4"),
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return {
        "mode": "inherited-from-train-start-service",
        "parent_service_sha256": parent_service_sha256,
        "reason": "preserve the centered motor decoder and BODY807 coordinate system during continuation",
        "body_scale_min": float(arrays["body_scale"].min()),
        "body_scale_max": float(arrays["body_scale"].max()),
        "reference_min": float(arrays["reference_rate"].min()),
        "reference_max": float(arrays["reference_rate"].max()),
        "rate_scale_min": float(arrays["rate_scale"].min()),
        "rate_scale_max": float(arrays["rate_scale"].max()),
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
    reset_prefix_fraction: float
    reset_sequence: int
    viability_weighted_motor: bool
    motor_slew_weight: float
    counterfactual_stability_weight: float
    motor_decoder_norm_weight: float
    cold_neutral_weight: float


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
) -> dict[str, torch.Tensor | int | bool]:
    burn_ins = {window.burn_in for window in windows}
    if len(burn_ins) != 1:
        raise ValueError("a recurrent batch must use one causal burn-in length")
    rows = [slice_window(episodes[window.episode_index], window) for window in windows]
    result: dict[str, torch.Tensor | int | bool] = {
        "burn_in": windows[0].burn_in,
        "reset_prefix": all(window.start == 0 and window.burn_in == 0 for window in windows),
    }
    for key in (
        "optic_rgb", "body_afferents", "delivered_context", "delivered_motor", "teacher_motor",
        "teacher_valid", "target_outcome", "target_reward", "target_success", "target_failure",
    ):
        result[key] = _tensor(np.stack([row[key] for row in rows], axis=1), device)
    return result


def _motor_loss(
    predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
    transition_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = target.new_ones(MOTOR)
    weights[84:90] = 1.5
    weights[90:] = 3.0
    per = (F.smooth_l1_loss(predicted, target, beta=0.05, reduction="none") * weights).mean(-1)
    valid_f = valid.to(per.dtype)
    if transition_weight is not None:
        valid_f = valid_f * transition_weight
    return (per * valid_f).sum() / valid_f.sum().clamp_min(1)


def _motor_slew_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    transition_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if predicted.shape[0] < 2:
        return predicted.new_zeros(())
    weights = target.new_ones(MOTOR)
    weights[84:90] = 1.5
    weights[90:] = 3.0
    per = (
        F.smooth_l1_loss(
            predicted[1:] - predicted[:-1], target[1:] - target[:-1],
            beta=0.025, reduction="none",
        ) * weights
    ).mean(-1)
    # A slew target is defined only when both adjacent teacher commands are
    # valid. In particular, the deliberately uncurated free-consequence bout
    # must not turn its placeholder zero command into a training target.
    pair_valid = (valid[1:] & valid[:-1]).to(per.dtype)
    if transition_weight is not None:
        pair_valid = pair_valid * torch.minimum(
            transition_weight[1:], transition_weight[:-1]
        )
    return (per * pair_valid).sum() / pair_valid.sum().clamp_min(1)


def _frozen_outcome_prediction(
    heads: PhysicalPredictionHeads,
    latent: torch.Tensor,
    motor: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Differentiate through action, while holding the learned surrogate fixed."""
    value = torch.cat((latent, motor, context), -1)
    first, _, second, _ = heads.shared
    value = F.silu(F.linear(value, first.weight.detach(), first.bias.detach()))
    value = F.silu(F.linear(value, second.weight.detach(), second.bias.detach()))
    return F.linear(value, heads.outcome.weight.detach(), heads.outcome.bias.detach())


def _decoder_only_motor(model: AnatomicalCNS, state: CNSState) -> torch.Tensor:
    """Decode detached MN activity so an auxiliary cannot reshape CNS state."""
    rates = state.rates[model.motor_rows.long()].detach()
    centered = (
        rates - model.motor_reference_rate[:, None]
    ) / model.motor_rate_scale[:, None]
    pre = (model.motor_weight * model.motor_mask) @ centered + model.motor_intercept[:, None]
    return torch.cat((torch.tanh(pre[:84]), torch.sigmoid(pre[84:])), 0).T


def cns_window_loss(
    model: AnatomicalCNS,
    heads: PhysicalPredictionHeads,
    arrays: dict[str, torch.Tensor | int | bool],
    burn_in: int,
    *,
    viability_weighted_motor: bool = False,
    motor_slew_weight: float = 0.0,
    counterfactual_stability_weight: float = 0.0,
    motor_decoder_norm_weight: float = 0.0,
    cold_neutral_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tensor_arrays = {key: value for key, value in arrays.items() if isinstance(value, torch.Tensor)}
    state: CNSState | None = None
    with torch.no_grad():
        for tick in range(burn_in):
            _, _, state = _cns_control_step(model,
                tensor_arrays["optic_rgb"][tick], tensor_arrays["body_afferents"][tick],
                tensor_arrays["delivered_context"][tick], state,
            )
    state = None if state is None else state.detach()
    sequence = tensor_arrays["delivered_context"].shape[0] - burn_in
    latent, motor, decoder_only_motor = [], [], []
    for tick in range(burn_in, burn_in + sequence + 1):
        context_tick = min(tick, burn_in + sequence - 1)
        optic_tick = tensor_arrays["optic_rgb"][tick]
        body_tick = tensor_arrays["body_afferents"][tick]
        context_value = tensor_arrays["delivered_context"][context_tick]
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
        decoder_only_motor.append(_decoder_only_motor(model, state))
    z = torch.stack(latent)
    predicted_motor = torch.stack(motor[:-1])
    decoder_candidate = torch.stack(decoder_only_motor[:-1])
    delivered_motor = tensor_arrays["teacher_motor"][burn_in:]
    delivered_context = tensor_arrays["delivered_context"][burn_in:]
    physical_motor = tensor_arrays["delivered_motor"][burn_in:]
    body_delta, latent_delta, outcome, reward = heads(
        z[:-1].reshape(-1, LATENT),
        physical_motor.reshape(-1, MOTOR),
        delivered_context.reshape(-1, CONTEXT),
    )
    batch = z.shape[1]
    body_now = tensor_arrays["body_afferents"][burn_in:-1]
    body_next = tensor_arrays["body_afferents"][burn_in + 1:]
    body_target = ((body_next - body_now) / model.body_scale[None, None]).clamp(-8, 8)
    latent_target = (z[1:] - z[:-1]).detach()
    outcome_target = tensor_arrays["target_outcome"][burn_in:]
    reward_target = tensor_arrays["target_reward"][burn_in:]
    transition_weight = None
    if viability_weighted_motor:
        # Continuous, measured viability keeps failed transitions in the loss
        # while giving stable/supporting demonstrations more influence.  The
        # values are target-only and never enter model inference.
        upright = ((outcome_target[..., 0] + 1.0) * 0.5).clamp(0, 1)
        stability = outcome_target[..., 13].clamp(0, 1)
        support = outcome_target[..., 6].clamp(0, 1)
        progress = torch.sigmoid(reward_target.clamp(-8, 8))
        success = tensor_arrays["target_success"][burn_in:].to(reward_target.dtype)
        transition_weight = (
            0.25 + 0.35 * upright + 0.35 * stability
            + 0.2 * support + 0.2 * progress + 0.25 * success
        ).detach()
        transition_weight = transition_weight / transition_weight.mean().clamp_min(1e-6)
    terms = {
        "motor": _motor_loss(
            predicted_motor, delivered_motor, tensor_arrays["teacher_valid"][burn_in:],
            transition_weight,
        ),
        "motor_slew": _motor_slew_loss(
            predicted_motor,
            delivered_motor,
            tensor_arrays["teacher_valid"][burn_in:],
            transition_weight,
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
    if bool(arrays.get("reset_prefix", False)) and counterfactual_stability_weight > 0:
        counterfactual = _frozen_outcome_prediction(
            heads,
            z[:-1].detach().reshape(-1, LATENT),
            decoder_candidate.reshape(-1, MOTOR),
            delivered_context.detach().reshape(-1, CONTEXT),
        )
        terms["counterfactual_stability"] = (
            F.softplus(0.35 - counterfactual[:, 0])
            + 0.5 * F.softplus(-counterfactual[:, 1])
            + 0.5 * F.softplus(0.45 - counterfactual[:, 6])
            + 0.1 * F.softplus(counterfactual[:, 7])
            + 0.03 * F.softplus(counterfactual[:, 14])
        ).mean()
    else:
        terms["counterfactual_stability"] = predicted_motor.new_zeros(())
    terms["motor_temporal_std"] = predicted_motor.std((0, 1)).mean()
    supported_weight = model.motor_weight * model.motor_mask
    supported_count = model.motor_mask.sum().clamp_min(1)
    terms["motor_decoder_rms"] = (
        supported_weight.square().sum() / supported_count
    ).sqrt()
    terms["motor_decoder_penalty"] = (
        (supported_weight / 0.02).square().sum() / supported_count
    )
    # The normalized position contract defines zero as the authored neutral
    # pose. Anchor the explicit centered decoder at the real CNS cold state so
    # ordinary imitation cannot hide a large unsafe servo offset in its
    # intercept. Dynamic weights remain trainable and are still supervised by
    # every ordinary/history-conditioned transition.
    cold_motor = _decoder_only_motor(model, model.initial_state(batch))
    terms["cold_neutral"] = F.smooth_l1_loss(
        cold_motor[:, :84], torch.zeros_like(cold_motor[:, :84]), beta=0.02
    )
    terms["cold_reference_servo_abs_mean"] = cold_motor[:, :84].abs().mean()
    terms["cold_motor_abs_mean"] = (
        predicted_motor[0, :, :84].abs().mean()
        if bool(arrays.get("reset_prefix", False)) else predicted_motor.new_zeros(())
    )
    total = (
        2.0 * terms["motor"]
        + terms["body_prediction"]
        + 0.5 * terms["latent_prediction"]
        + 0.25 * terms["outcome_prediction"]
        + 0.25 * terms["reward_prediction"]
        + motor_slew_weight * terms["motor_slew"]
        + counterfactual_stability_weight * terms["counterfactual_stability"]
        + motor_decoder_norm_weight * terms["motor_decoder_penalty"]
        + cold_neutral_weight * terms["cold_neutral"]
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
    sampler = _cns_sampler(corpus, recipe, seed=17, split=split)
    rows: list[dict[str, float]] = []
    model.eval(); heads.eval()
    for windows in (sampler.sample(recipe.batch_size) for _ in range(8)):
        arrays = _stack_windows(sampler.episodes, windows, device)
        total, terms = cns_window_loss(
            model, heads, arrays, int(arrays["burn_in"]),
            viability_weighted_motor=recipe.viability_weighted_motor,
            motor_slew_weight=recipe.motor_slew_weight,
            counterfactual_stability_weight=recipe.counterfactual_stability_weight,
            motor_decoder_norm_weight=recipe.motor_decoder_norm_weight,
            cold_neutral_weight=recipe.cold_neutral_weight,
        )
        rows.append({"total": float(total), **{key: float(value) for key, value in terms.items()}})
    model.train(); heads.train()
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def _cns_sampler(
    corpus: Corpus, recipe: Recipe, *, seed: int, split: str,
) -> BalancedWindowSampler | MixedCausalSampler:
    if recipe.reset_prefix_fraction <= 0:
        return BalancedWindowSampler(
            corpus, burn_in=recipe.burn_in, optimize=recipe.sequence,
            seed=seed, split=split,
        )
    return MixedCausalSampler(
        corpus, burn_in=recipe.burn_in, optimize=recipe.sequence,
        reset_optimize=recipe.reset_sequence,
        reset_fraction=recipe.reset_prefix_fraction,
        seed=seed, split=split,
    )


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
    sampler: BalancedWindowSampler | MixedCausalSampler,
    windows: tuple[Window, ...],
    cache: dict[str, np.ndarray],
    consequence: dict[tuple[int, int], Consequence],
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    burn_ins = {window.burn_in for window in windows}
    if len(burn_ins) != 1:
        raise ValueError("a resident batch must use one causal burn-in length")
    batch_burn_in = windows[0].burn_in
    latent_rows, context_rows, previous_rows, reward_rows, reset_rows = [], [], [], [], []
    count_rows, utility_rows, uncertainty_rows = [], [], []
    for window in windows:
        episode = sampler.episodes[window.episode_index]
        start, stop, resident = window.history_start, window.stop, window.resident
        latent_rows.append(np.asarray(cache[episode.sha256][start : stop + 1, resident]))
        context = episode.delivered_context[start:stop, resident]
        context_rows.append(context)
        previous = np.zeros((context.shape[0] + 1, CONTEXT), np.float32)
        if start:
            previous[0] = episode.delivered_context[start - 1, resident]
        previous[1:] = context
        previous_rows.append(previous)
        reward_rows.append(episode.reward[start:stop, resident])
        reset_rows.append(episode.reset[start : stop + 1, resident])
        c, u, s = [], [], []
        for tick in range(start, stop):
            item = consequence[(int(episode.curriculum_phase[tick, resident]), int(episode.control_source[tick, resident]))]
            c.append(item.count); u.append(item.utility); s.append(item.uncertainty)
        count_rows.append(c); utility_rows.append(u); uncertainty_rows.append(s)
    latent = np.stack(latent_rows, axis=1)
    delivered = np.stack(context_rows, axis=1)
    return {
        "cns_latent": _tensor(latent, device),
        "previous_delivered_context": _tensor(np.stack(previous_rows, axis=1), device),
        "reset": _tensor(np.stack(reset_rows, axis=1), device),
        "delivered_context": _tensor(delivered, device),
        "physical_reward": _tensor(np.stack(reward_rows, axis=1), device),
        "consequence_count": _tensor(np.asarray(count_rows, np.float32).T, device),
        "consequence_utility": _tensor(np.asarray(utility_rows, np.float32).T, device),
        "consequence_uncertainty": _tensor(np.asarray(uncertainty_rows, np.float32).T, device),
        "burn_in": batch_burn_in,
    }


@torch.inference_mode()
def evaluate_resident(
    model: CnsResidentModel,
    sampler: BalancedWindowSampler | MixedCausalSampler,
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
    additional_corpora: list[Corpus] = []
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
        additional_corpora.append(nursery)
    if arguments.recovery_corpus is not None:
        recovery_path = arguments.recovery_corpus.expanduser().resolve()
        recovery = load_recovery_corpus(recovery_path)
        manifest_path = (
            recovery_path / "recovery-corpus.json"
            if recovery_path.is_dir() else recovery_path
        )
        source_receipts.append({
            "role": "on-policy-failure-teacher-correction-release",
            "format": recovery.manifest["format"],
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        })
        additional_corpora.append(recovery)
    if arguments.support_corpus is not None:
        support_path = arguments.support_corpus.expanduser().resolve()
        support = load_support_acquisition_corpus(support_path)
        manifest_path = (
            support_path / "support-acquisition-corpus.json"
            if support_path.is_dir() else support_path
        )
        source_receipts.append({
            "role": "fresh-life-neutral-support-short-probe-acquisition",
            "format": support.manifest["format"],
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        })
        additional_corpora.append(support)
    if additional_corpora:
        corpus = combine_corpora(primary, *additional_corpora)
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
    collection_service_paths = (
        [service_path] if arguments.collection_service is None
        else [path.expanduser().resolve() for path in arguments.collection_service]
    )
    collection_services: dict[str, tuple[dict[str, np.ndarray], dict[str, Any], Path]] = {}
    for path in collection_service_paths:
        sha = sha256_file(path)
        if sha in collection_services:
            raise RuntimeError("collection CNS service was supplied more than once")
        if sha == parent_service_sha:
            arrays, metadata = parent_arrays, parent_metadata
        else:
            arrays, metadata = load_service_artifact(path)
        collection_services[sha] = (arrays, metadata, path)
    continuing = set(collection_services) != {parent_service_sha}
    if continuing and not arguments.preserve_parent_normalization:
        raise RuntimeError(
            "continuation from a service other than the collection service requires "
            "--preserve-parent-normalization"
        )
    all_episodes = (*corpus.train, *corpus.validation, *corpus.heldout)
    for episode in all_episodes:
        service_sha = episode.metadata["cns_service_sha256"]
        record = collection_services.get(service_sha)
        if record is None or episode.metadata["cns_adapter_sha256"] != record[1]["adapter_sha256"]:
            raise RuntimeError("corpus was not collected through a supplied collection CNS service")
    used_collection_services = {
        episode.metadata["cns_service_sha256"] for episode in all_episodes
    }
    if used_collection_services != set(collection_services):
        raise RuntimeError("supplied collection CNS services must exactly cover the corpora")
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
        reset_prefix_fraction=arguments.reset_prefix_fraction,
        reset_sequence=arguments.reset_sequence,
        viability_weighted_motor=arguments.viability_weighted_motor,
        motor_slew_weight=arguments.motor_slew_weight,
        counterfactual_stability_weight=arguments.counterfactual_stability_weight,
        motor_decoder_norm_weight=arguments.motor_decoder_norm_weight,
        cold_neutral_weight=arguments.cold_neutral_weight,
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
        "collection_services": [
            {"file_sha256": sha, "adapter_sha256": metadata["adapter_sha256"]}
            for sha, (_, metadata, _) in sorted(collection_services.items())
        ],
        "continuation_from_trained_parent": continuing,
        "normalization_mode": (
            "inherited-from-train-start-service"
            if arguments.preserve_parent_normalization else "estimated-from-current-train-worlds"
        ),
        "parent_resident_sha256": None if arguments.parent_resident is None else sha256_file(arguments.parent_resident),
        "parent_prediction_heads_sha256": (
            None if arguments.prediction_heads is None else sha256_file(arguments.prediction_heads)
        ),
    }
    progress: dict[str, Any] = {
        "format": TRAINING_FORMAT, "status": "normalization", "stage": "cns",
        "update": 0, "recipe": asdict(recipe), "identity": identity,
    }
    atomic_json(run / "progress.json", progress)

    collection_cache_check = {"services": []}
    for service_sha, (collection_arrays, collection_metadata, _) in sorted(collection_services.items()):
        episodes = tuple(
            episode for episode in all_episodes
            if episode.metadata["cns_service_sha256"] == service_sha
        )
        collection_model = AnatomicalCNS(collection_arrays, device=device).eval()
        check = validate_collected_parent_cache(collection_model, episodes, device)
        collection_cache_check["services"].append({
            "file_sha256": service_sha,
            "adapter_sha256": collection_metadata["adapter_sha256"],
            "worlds": len(episodes),
            **check,
        })
        del collection_model
        torch.cuda.empty_cache()
    atomic_json(run / "collection-cache-check.json", collection_cache_check)

    mutable_arrays = {name: np.asarray(value) for name, value in parent_arrays.items()}
    if arguments.preserve_parent_normalization:
        model = AnatomicalCNS(mutable_arrays, device=device)
        normalization = preserve_parent_normalization(
            model, run / "train-world-normalization.npz", parent_service_sha
        )
    else:
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
    if arguments.prediction_heads is None:
        heads = PhysicalPredictionHeads().to(device)
        prediction_heads_parent = None
    else:
        heads, prediction_heads_parent = load_prediction_heads(
            arguments.prediction_heads, parent_service_sha, device
        )
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
    train_sampler = _cns_sampler(corpus, recipe, seed=recipe.seed, split="train")
    validation_before = evaluate_cns(
        model, heads, corpus, "validation-worlds", recipe, device
    )
    progress.update(
        status="running", normalization=normalization,
        collection_cache_check=collection_cache_check, validation_before=validation_before,
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
        total, terms = cns_window_loss(
            model, heads, batch, int(batch["burn_in"]),
            viability_weighted_motor=recipe.viability_weighted_motor,
            motor_slew_weight=recipe.motor_slew_weight,
            counterfactual_stability_weight=recipe.counterfactual_stability_weight,
            motor_decoder_norm_weight=recipe.motor_decoder_norm_weight,
            cold_neutral_weight=recipe.cold_neutral_weight,
        )
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
            "collection_services": identity["collection_services"],
            "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
            "world_split": "per corpus: 0..7 train, 8..9 validation, 10..11 untouched heldout",
            "model_ingress": ["optic1771 RGB", "BODY807", "delivered context12"],
            "physical_identity_semantics": {
                "anatomical_body_schema_sha256": ANATOMICAL_BODY_SCHEMA_SHA256,
                "cns_body807_schema_sha256": CNS_BODY807_SCHEMA_SHA256,
                "morphology_asset_set_sha256": MORPHOLOGY_ASSET_SET_SHA256,
                "bootstrap_recorded_body_schema_field": "BODY807 sensory schema; authenticated metadata amendment",
            },
            "observer_values": "targets and sampling only",
            "training_only_heads_exported": False,
            "normalization": calibration_payload,
        }),
    )
    atomic_json(run / "cns-service-receipt.json", service_receipt)
    prediction_heads_output = run / "physical-prediction-heads.pt"
    prediction_heads_payload = {
        "format": PREDICTION_HEAD_FORMAT,
        "child_service_sha256": service_receipt["file_sha256"],
        "child_adapter_sha256": service_receipt["metadata"]["adapter_sha256"],
        "source_checkpoint_sha256": sha256_file(run / f"cns-checkpoint-{recipe.cns_updates:06d}.pt"),
        "source_revision": identity["source_revision"],
        "parent_prediction_heads": prediction_heads_parent,
        "state": heads.state_dict(),
    }
    atomic_torch(prediction_heads_output, prediction_heads_payload)
    prediction_heads_receipt = {
        key: value for key, value in prediction_heads_payload.items() if key != "state"
    } | {
        "path": str(prediction_heads_output),
        "file_sha256": sha256_file(prediction_heads_output),
    }

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
        if recipe.reset_prefix_fraction > 0:
            resident_train_sampler = MixedCausalSampler(
                corpus, burn_in=recipe.burn_in, optimize=max(recipe.sequence, 64),
                reset_optimize=max(recipe.reset_sequence, 64),
                reset_fraction=recipe.reset_prefix_fraction,
                seed=recipe.seed + 1, split="train",
            )
            resident_validation_sampler = MixedCausalSampler(
                corpus, burn_in=recipe.burn_in, optimize=max(recipe.sequence, 64),
                reset_optimize=max(recipe.reset_sequence, 64),
                reset_fraction=recipe.reset_prefix_fraction,
                seed=23, split="validation-worlds",
            )
        else:
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
        "collection_cache_check": collection_cache_check,
        "cns": {
            "validation_before": validation_before,
            "validation_after": validation_after,
            "heldout_offline": heldout_cns,
            "service": service_receipt,
            "updates": len(cns_history),
            "physical_prediction_heads": prediction_heads_receipt,
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
    extract = sub.add_parser("extract-heads")
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--result", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--corpus", type=Path, required=True)
    fit.add_argument("--nursery-corpus", type=Path)
    fit.add_argument("--recovery-corpus", type=Path)
    fit.add_argument("--support-corpus", type=Path)
    fit.add_argument("--service", type=Path, required=True)
    fit.add_argument(
        "--collection-service", type=Path, action="append",
        help="service used to generate collected_latent; repeat for mixed lineages",
    )
    fit.add_argument(
        "--preserve-parent-normalization", action="store_true",
        help="retain BODY807 and centered-MN normalization from --service",
    )
    fit.add_argument("--parent-resident", type=Path)
    fit.add_argument("--prediction-heads", type=Path)
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
    fit.add_argument("--reset-prefix-fraction", type=float, default=0.0)
    fit.add_argument("--reset-sequence", type=int, default=40)
    fit.add_argument("--viability-weighted-motor", action="store_true")
    fit.add_argument("--motor-slew-weight", type=float, default=0.0)
    fit.add_argument("--counterfactual-stability-weight", type=float, default=0.0)
    fit.add_argument("--motor-decoder-norm-weight", type=float, default=0.0)
    fit.add_argument("--cold-neutral-weight", type=float, default=0.0)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "seal":
        print(json.dumps(seal_corpus(arguments.source, arguments.output), sort_keys=True))
    elif arguments.command == "extract-heads":
        print(json.dumps(extract_prediction_heads(
            arguments.checkpoint, arguments.result, arguments.output
        ), sort_keys=True))
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
