#!/usr/bin/env python3
"""Pretrain the CNS-only sensory adapter through the full fixed MaleCNS graph."""

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

from research.sensorimotor_skills.cns_adapter import (
    BODY_CHANNELS,
    CHECKPOINT_TICKS,
    CNSStaticArrays,
    OPTIC_CHANNELS,
    OPTIC_SITES,
    SensoryPredictionHeads,
    TrainableCNSAdapter,
    file_sha256,
    sensory_prediction_loss,
    write_parameter_artifact,
)


RUN_FORMAT = "chreatures-cns-sensory-pretraining-v2"
GENERATOR_FORMAT = "chreatures-procedural-sensory-sequences-v2"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=32)
    parser.add_argument("--checkpoint-ticks", type=int, default=CHECKPOINT_TICKS)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--heldout-seed", type=int, default=20270920)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def source_receipts() -> dict[str, str]:
    names = (
        "scripts/train_cns_adapter.py",
        "research/sensorimotor_skills/cns_adapter.py",
        "chreatures/cns_adapter_contract.py",
    )
    return {name: file_sha256(ROOT / name) for name in names}


def optic_coordinates(atlas: Path, device: torch.device) -> torch.Tensor:
    with np.load(atlas, allow_pickle=False) as bundle:
        locations = np.asarray(bundle["site_side_hex"], dtype=np.float32)
    if locations.shape != (OPTIC_SITES, 3):
        raise ValueError("optic site order differs")
    side, q, r = locations.T
    x = q + 0.5 * r
    y = np.float32(np.sqrt(0.75)) * r
    for values in (x, y):
        values -= values.mean()
        values /= max(float(values.std()), 1e-6)
    # Mirror the two eyes into one body-centric visual coordinate system while
    # retaining their exact atlas row order.
    x = np.where(side == 1, -np.abs(x), np.abs(x))
    coordinates = np.stack((x, y), axis=-1).astype(np.float32)
    return torch.from_numpy(coordinates).to(device=device)


class ProceduralOpticSequences:
    """Batch-native independent motion fields with no tick exposed as an input."""

    def __init__(self, coordinates: torch.Tensor, *, ticks: int, batch_size: int):
        self.coordinates = coordinates
        self.ticks = ticks
        self.batch_size = batch_size
        self.device = coordinates.device

    def make(self, seed: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed)
        b, t = self.batch_size, self.ticks

        def parameter(low: float, high: float, width: int | None = None) -> torch.Tensor:
            shape = (b,) if width is None else (b, width)
            value = rng.uniform(low, high, size=shape).astype(np.float32)
            return torch.from_numpy(value).to(self.device)

        theta = parameter(-np.pi, np.pi)
        spatial = parameter(1.3, 4.8)
        temporal = parameter(-0.42, 0.42)
        phase = parameter(-np.pi, np.pi)
        width = parameter(0.14, 0.55)
        color_a = parameter(0.05, 0.95, OPTIC_CHANNELS)
        color_b = parameter(0.05, 0.95, OPTIC_CHANNELS)
        family = torch.from_numpy(rng.integers(0, 3, size=(b,))).to(self.device)

        action_phase = parameter(-np.pi, np.pi, 12)
        action_frequency = parameter(0.035, 0.16, 12)
        tick = torch.arange(t, device=self.device, dtype=torch.float32)[:, None, None]
        delivered_action = 0.45 * torch.sin(
            tick * action_frequency[None] + action_phase[None]
        )
        displacement = torch.cat(
            (
                torch.zeros((1, b), device=self.device),
                torch.cumsum(delivered_action[:-1, :, 0], dim=0),
            ),
            dim=0,
        ) * 0.055

        xy = self.coordinates[None, None]
        projection = torch.cos(theta)[None, :, None] * xy[..., 0] + torch.sin(
            theta
        )[None, :, None] * xy[..., 1]
        orthogonal = -torch.sin(theta)[None, :, None] * xy[..., 0] + torch.cos(
            theta
        )[None, :, None] * xy[..., 1]
        motion = (
            phase[None, :, None]
            + temporal[None, :, None] * tick
            + displacement[:, :, None]
        )
        argument = spatial[None, :, None] * projection + motion
        bars = 0.5 + 0.5 * torch.tanh(3.0 * torch.sin(argument))
        checker = 0.5 + 0.5 * torch.tanh(
            3.0
            * torch.sin(argument)
            * torch.sin(spatial[None, :, None] * orthogonal - motion)
        )
        center = 1.6 * torch.sin(0.7 * motion)
        occluder = 1.0 - torch.exp(
            -0.5 * ((projection - center) / width[None, :, None]).square()
        )
        intensity = torch.where(
            (family == 0)[None, :, None],
            bars,
            torch.where((family == 1)[None, :, None], checker, occluder),
        )
        clean_optic = (
            color_a[None, :, None]
            + intensity[..., None] * (color_b - color_a)[None, :, None]
        ).clamp(0.0, 1.0)
        noise_rng = torch.Generator(device=self.device)
        noise_rng.manual_seed(seed ^ 0x5A17)
        input_optic = (
            clean_optic
            + 0.018 * torch.randn(clean_optic.shape, generator=noise_rng, device=self.device)
        ).clamp(0.0, 1.0)
        # Independent procedural body-local channels train the coupled afferent path.
        # These are calibration signals, not physically collected physiology.
        body_phase = parameter(-np.pi, np.pi, BODY_CHANNELS)
        body_frequency = parameter(0.025, 0.18, BODY_CHANNELS)
        body_amplitude = parameter(0.1, 0.7, BODY_CHANNELS)
        body = body_amplitude[None] * torch.sin(
            tick * body_frequency[None] + body_phase[None]
        )
        reset = torch.zeros((t, b), dtype=torch.bool, device=self.device)
        reset[0] = True
        next_valid = torch.ones((t, b), dtype=torch.bool, device=self.device)
        next_valid[-1] = False
        return {
            "input_optic": input_optic,
            "clean_optic": clean_optic,
            "body": body,
            "delivered_action": delivered_action,
            "reset": reset,
            "next_valid": next_valid,
        }


def compute_loss(
    model: TrainableCNSAdapter,
    heads: SensoryPredictionHeads,
    packet: dict[str, torch.Tensor],
    checkpoint_ticks: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    latent, _ = model.forward_sequence(
        packet["input_optic"],
        packet["body"],
        packet["reset"],
        checkpoint_ticks=checkpoint_ticks,
    )
    loss, metrics = sensory_prediction_loss(
        latent,
        heads,
        packet["clean_optic"],
        packet["body"],
        packet["delivered_action"],
        packet["reset"],
        packet["next_valid"],
        model.body_mean,
        model.body_scale,
        body_coefficient=0.1,
        variance_coefficient=0.0,
        covariance_coefficient=0.0,
        forecast_horizon=4,
    )
    return loss, metrics, latent


def evaluate(
    model: TrainableCNSAdapter,
    heads: SensoryPredictionHeads,
    packet: dict[str, torch.Tensor],
    checkpoint_ticks: int,
) -> dict[str, float]:
    model.eval()
    heads.eval()
    with torch.no_grad():
        loss, metrics, latent = compute_loss(model, heads, packet, checkpoint_ticks)
    result = {name: float(value) for name, value in metrics.items()}
    result["latent_abs_mean"] = float(latent.abs().mean())
    result["latent_temporal_delta_mean"] = float(
        (latent[1:] - latent[:-1]).abs().mean()
    )
    result["loss"] = float(loss)
    return result


def packet_identity(packet: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in packet.items()
    }


def main() -> None:
    args = arguments()
    if args.updates < 1 or args.batch_size < 1 or args.ticks < 5:
        raise ValueError("updates/batch-size must be positive and ticks at least five")
    if not 1 <= args.checkpoint_ticks <= args.ticks:
        raise ValueError("checkpoint tick group differs")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    log_path = args.output / "metrics.jsonl"
    status_path = args.output / "status.json"
    initial_path = args.output / "cns-adapter-initialized-untrained.npz"
    fitted_path = args.output / "cns-adapter-sensory-pretrained.npz"
    checkpoint_path = args.output / f"training-checkpoint-update{args.updates}.pt"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested ROCm/CUDA device is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    static_started = time.perf_counter()
    static = CNSStaticArrays.load(args.graph, args.atlas)
    static_load_seconds = time.perf_counter() - static_started
    model = TrainableCNSAdapter(static, device=device)
    heads = SensoryPredictionHeads(device=device)
    coordinates = optic_coordinates(args.atlas, device)
    generator = ProceduralOpticSequences(
        coordinates, ticks=args.ticks, batch_size=args.batch_size
    )
    heldout = generator.make(args.heldout_seed)
    sources = source_receipts()
    configuration = {
        "format": RUN_FORMAT,
        "generator_format": GENERATOR_FORMAT,
        "arguments": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "ticks": args.ticks,
            "checkpoint_ticks": args.checkpoint_ticks,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "heldout_seed": args.heldout_seed,
            "device": str(device),
        },
        "static_identity": static.identity,
        "source_sha256": sources,
        "objective": {
            "description": "current and action-conditioned four-tick clean sensory prediction through the recurrent full MaleCNS V2 latent",
            "training_stimuli": "independent procedural moving color bars, checker fields, occluders and body-local oscillations",
            "raw_target_use": "training-only discarded decoders",
            "body": "independent finite 43-channel procedural calibration; not physically collected physiology or competence",
            "bad_apple_probe_in_training": False,
            "controller_inputs": "CNS latent only; no raw target or generator coordinate",
        },
    }
    atomic_json(args.output / "configuration.json", configuration)
    initial_receipt = write_parameter_artifact(
        initial_path,
        model,
        training_status="initialized-untrained",
        provenance={
            **configuration["objective"],
            "seed": args.seed,
            "source_sha256": sources,
        },
    )
    initialized_heldout = evaluate(model, heads, heldout, args.checkpoint_ticks)

    optimizer = torch.optim.AdamW(
        [*model.parameters(), *heads.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    started = time.perf_counter()
    last_gradient: dict[str, float] = {}
    model.train()
    heads.train()
    with log_path.open("x", buffering=1) as log:
        for update in range(1, args.updates + 1):
            packet = generator.make(args.seed + update)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, latent = compute_loss(
                model, heads, packet, args.checkpoint_ticks
            )
            loss.backward()
            last_gradient = {
                "optic_spectral_logits": float(model.optic_spectral_logits.grad.norm()),
                "optic_gain_raw": float(model.optic_gain_raw.grad.norm()),
                "optic_bias": float(model.optic_bias.grad.norm()),
                "dynamics_baseline_raw": float(model.dynamics_baseline_raw.grad.norm()),
                "dynamics_tau_raw": float(model.dynamics_tau_raw.grad.norm()),
                "dynamics_recurrent_gain_raw": float(model.dynamics_recurrent_gain_raw.grad.norm()),
                "dynamics_adaptation_gain_raw": float(model.dynamics_adaptation_gain_raw.grad.norm()),
                "dynamics_adaptation_tau_raw": float(model.dynamics_adaptation_tau_raw.grad.norm()),
                "readout_projection_weight": float(model.readout_projection.weight.grad.norm()),
                "readout_output_weight": float(model.readout_output.weight.grad.norm()),
            }
            optimizer.step()
            with torch.no_grad():
                model.readout_projection.weight.mul_(model.readout_mask.unsqueeze(0))
            elapsed = time.perf_counter() - started
            record = {
                "update": update,
                "resident_ticks": update * args.ticks * args.batch_size,
                "elapsed_seconds": elapsed,
                "resident_ticks_per_second": update * args.ticks * args.batch_size / elapsed,
                **{name: float(value) for name, value in metrics.items()},
                "latent_abs_mean": float(latent.detach().abs().mean()),
                "gradient_norm": last_gradient,
            }
            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            os.fsync(log.fileno())
            atomic_json(
                status_path,
                {
                    "format": RUN_FORMAT,
                    "state": "training",
                    "pid": os.getpid(),
                    "update": update,
                    "updates": args.updates,
                    "latest": record,
                },
            )

    fitted_heldout = evaluate(model, heads, heldout, args.checkpoint_ticks)
    training_elapsed = time.perf_counter() - started
    fitted_receipt = write_parameter_artifact(
        fitted_path,
        model,
        training_status="trained",
        provenance={
            **configuration["objective"],
            "qualification": "procedural optic/body sensory pretraining only; initialized controller; no motor, feeding or physiological competence claim",
            "seed": args.seed,
            "heldout_seed": args.heldout_seed,
            "updates": args.updates,
            "resident_ticks": args.updates * args.ticks * args.batch_size,
            "initial_parameter_artifact_sha256": initial_receipt["artifact_sha256"],
            "source_sha256": sources,
        },
    )
    checkpoint = {
        "format": RUN_FORMAT,
        "update": args.updates,
        "configuration": configuration,
        "adapter": model.state_dict(),
        "training_heads": heads.state_dict(),
        "optimizer": optimizer.state_dict(),
        "initialized_parameter_receipt": initial_receipt,
        "fitted_parameter_receipt": fitted_receipt,
        "heldout_packet_sha256": packet_identity(heldout),
        "initialized_heldout": initialized_heldout,
        "fitted_heldout": fitted_heldout,
        "final_gradient_norm": last_gradient,
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    finished = {
        "format": RUN_FORMAT,
        "state": "complete",
        "pid": os.getpid(),
        "updates": args.updates,
        "resident_ticks": args.updates * args.ticks * args.batch_size,
        "static_load_seconds": static_load_seconds,
        "training_seconds": training_elapsed,
        "initialized_parameter": initial_receipt,
        "fitted_parameter": fitted_receipt,
        "training_checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "file_sha256": file_sha256(checkpoint_path),
        },
        "heldout_packet_sha256": checkpoint["heldout_packet_sha256"],
        "initialized_heldout": initialized_heldout,
        "fitted_heldout": fitted_heldout,
        "final_gradient_norm": last_gradient,
    }
    atomic_json(args.output / "receipt.json", finished)
    atomic_json(status_path, finished)
    print(json.dumps(finished, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
