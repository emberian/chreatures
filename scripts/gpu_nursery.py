#!/usr/bin/env python3
"""Run reproducible batched sensory-history experiments on the fly circuit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chreatures.brain import Brain, CHANNELS, load_connectome
from chreatures.torch_circuit import TorchCircuit

BLOCK_STEPS = 128
CYCLE_STEPS = BLOCK_STEPS * 3
REWARD_PHASE = 104


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def senses_at(step: int, *, forced_odor: int | None = None) -> dict[str, Any]:
    """A deterministic local stimulus with bilateral odor and other modalities."""
    block = step // BLOCK_STEPS
    phase = step % BLOCK_STEPS
    odor_index = block % 3 if forced_odor is None else forced_odor
    strength = 0.72 if forced_odor is None else 0.65
    left = np.zeros(3, dtype=np.float32)
    right = np.zeros(3, dtype=np.float32)
    left[odor_index] = strength * (0.92 if block % 2 else 1.0)
    right[odor_index] = strength * (1.0 if block % 2 else 0.92)

    vision = np.zeros((16, 4), dtype=np.float32)
    vision[:, odor_index] = 0.16 + 0.03 * np.cos(phase / BLOCK_STEPS * 2 * np.pi)
    if phase < 24:
        vision[:8, 3] = 0.18
    elif 64 <= phase < 88:
        vision[8:, 3] = 0.18
    sound = np.zeros(3, dtype=np.float32)
    sound[odor_index] = 0.35 if 32 <= phase < 80 else 0.08
    return {
        "odor": np.stack((left, right)).tolist(),
        "vision": vision.tolist(),
        "sound": sound.tolist(),
        "shade": 0.22 if phase >= 96 else 0.05,
        "touch": [0.25 if phase == 16 else 0.0, 0.25 if phase == 80 else 0.0],
    }


def encoded_cycle() -> np.ndarray:
    return np.stack([Brain.encode(senses_at(step)) for step in range(CYCLE_STEPS)])


def reward_at(step: int, histories: np.ndarray) -> np.ndarray:
    odor_index = (step // BLOCK_STEPS) % 3
    pulse = step % BLOCK_STEPS == REWARD_PHASE
    return ((histories == odor_index) & pulse).astype(np.float32) * 0.004


def run_condition(
    circuit: TorchCircuit,
    cycle: torch.Tensor,
    histories: np.ndarray,
    *,
    steps: int,
    dt: float,
    learning: bool,
    silenced: bool,
    sample_every: int,
) -> tuple[dict[str, np.ndarray], float]:
    circuit.reset()
    histories_device = torch.as_tensor(histories, device=circuit.device)
    samples: dict[str, list[np.ndarray]] = {
        "step": [],
        "activity_mean": [],
        "support_mean": [],
        "decoded": [],
        "values": [],
    }
    synchronize(circuit.device)
    started = time.perf_counter()
    for step in range(steps):
        encoded = cycle[step % len(cycle)].expand(circuit.batch_size, -1)
        odor_index = (step // BLOCK_STEPS) % 3
        pulse = step % BLOCK_STEPS == REWARD_PHASE
        if pulse:
            reward = (histories_device == odor_index).to(circuit.dtype) * 0.004
        else:
            reward = 0.0
        result = circuit.step(
            encoded,
            dt,
            reward,
            learning=learning,
            silenced=silenced,
            validate=False,
        )
        if step % sample_every == 0 or step == steps - 1:
            samples["step"].append(np.asarray(step, dtype=np.int32))
            samples["activity_mean"].append(
                circuit.rates.mean(dim=1).float().cpu().numpy()
            )
            samples["support_mean"].append(
                circuit.support.mean(dim=1).float().cpu().numpy()
            )
            samples["decoded"].append(result.decoded.float().cpu().numpy())
            samples["values"].append(circuit.values.float().cpu().numpy())
    synchronize(circuit.device)
    elapsed = time.perf_counter() - started

    learned = circuit.learned_state()
    probe_prediction = []
    probe_decoded = []
    for odor in range(3):
        circuit.reset_dynamics()
        circuit.load_learned_state(learned)
        probe_encoded = torch.as_tensor(
            Brain.encode(senses_at(0, forced_odor=odor)),
            device=circuit.device,
            dtype=circuit.dtype,
        ).expand(circuit.batch_size, -1)
        for _ in range(64):
            result = circuit.step(
                probe_encoded,
                dt,
                learning=False,
                silenced=silenced,
                validate=False,
            )
        probe_prediction.append(result.prediction.float().cpu().numpy())
        probe_decoded.append(result.decoded.float().cpu().numpy())

    arrays = {name: np.stack(values) for name, values in samples.items()}
    arrays.update(
        {
            "histories": histories.astype(np.int16),
            "final_values": learned["values"].float().cpu().numpy(),
            "final_sound_memory": learned["sound_memory"].float().cpu().numpy(),
            "total_nutrition": learned["total_nutrition"].float().cpu().numpy(),
            "probe_prediction": np.stack(probe_prediction, axis=1),
            "probe_decoded": np.stack(probe_decoded, axis=1),
        }
    )
    return arrays, elapsed


def parity_check(
    graph_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    dt: float,
    tolerance: float,
) -> dict[str, Any]:
    graph = load_connectome(str(graph_path))
    batch = 2
    histories = np.arange(batch) % 3
    circuit = TorchCircuit(graph, batch, device=device, dtype=dtype)
    brains = [Brain(graph_path, seed=900 + index) for index in range(batch)]
    fields = (
        "rates",
        "adaptation",
        "support",
        "context",
        "eligibility",
        "values",
        "sound_memory",
        "sound_trace",
        "last_decoded",
        "last_prediction",
    )
    maximum = {field: 0.0 for field in fields}
    checkpoints = {0, 1, 15, 63, 127, 255, steps - 1}
    started = time.perf_counter()
    for step in range(steps):
        senses = [senses_at(step + index * 7) for index in range(batch)]
        encoded = np.stack([Brain.encode(value) for value in senses])
        reward = reward_at(step, histories)
        circuit.step(encoded, dt, reward, validate=False)
        for index, brain in enumerate(brains):
            brain.step(senses[index], {"energy": 0.7}, dt, float(reward[index]))
        if step in checkpoints:
            for field in fields:
                actual = getattr(circuit, field)
                if field == "last_prediction":
                    expected = np.asarray([brain.last_prediction for brain in brains])
                else:
                    expected = np.stack([getattr(brain, field) for brain in brains])
                error = float(
                    np.max(np.abs(actual.float().cpu().numpy() - expected), initial=0)
                )
                maximum[field] = max(maximum[field], error)
    synchronize(device)
    elapsed = time.perf_counter() - started
    overall = max(maximum.values())
    return {
        "steps": steps,
        "batch_size": batch,
        "simulated_seconds": steps * dt,
        "elapsed_seconds": elapsed,
        "tolerance": tolerance,
        "max_abs_error": overall,
        "within_tolerance": overall <= tolerance,
        "max_abs_error_by_field": maximum,
    }


def summary_rows(
    condition: str, arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    histories = arrays["histories"]
    for history in range(3):
        selected = histories == history
        assigned = arrays["probe_prediction"][selected, history]
        others = np.delete(arrays["probe_prediction"][selected], history, axis=1).mean(axis=1)
        for probe in range(3):
            predictions = arrays["probe_prediction"][selected, probe]
            rows.append(
                {
                    "condition": condition,
                    "reinforced_history": history,
                    "probe_odor": probe,
                    "n": int(selected.sum()),
                    "prediction_mean": float(predictions.mean()),
                    "prediction_std": float(predictions.std()),
                    "learned_value_mean": float(arrays["final_values"][selected, probe].mean()),
                    "assigned_minus_other_prediction": float((assigned - others).mean()),
                    "activity_mean": float(arrays["activity_mean"][-1, selected].mean()),
                    "support_mean": float(arrays["support_mean"][-1, selected].mean()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", type=Path, default=Path("data/connectome/circuit.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--sample-every", type=int, default=16)
    parser.add_argument("--parity-steps", type=int, default=256)
    parser.add_argument("--parity-tolerance", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.batch_size < 3 or args.steps <= 0 or args.sample_every <= 0:
        parser.error("batch-size >= 3 and positive steps/sample-every are required")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    graph_path = args.connectome.resolve()
    graph = load_connectome(str(graph_path))
    histories = np.arange(args.batch_size, dtype=np.int16) % 3
    cycle = torch.as_tensor(encoded_cycle(), device=device, dtype=dtype)
    circuit = TorchCircuit(graph, args.batch_size, device=device, dtype=dtype)
    conditions = {
        "learned_recurrent": (True, False),
        "plasticity_disabled": (False, False),
        "recurrent_silenced": (True, True),
    }
    results: dict[str, dict[str, np.ndarray]] = {}
    runtimes: dict[str, float] = {}
    all_rows: list[dict[str, Any]] = []
    for name, (learning, silenced) in conditions.items():
        print(f"running {name}: {args.batch_size} brains x {args.steps} steps", file=sys.stderr)
        arrays, elapsed = run_condition(
            circuit,
            cycle,
            histories,
            steps=args.steps,
            dt=args.dt,
            learning=learning,
            silenced=silenced,
            sample_every=args.sample_every,
        )
        results[name] = arrays
        runtimes[name] = elapsed
        all_rows.extend(summary_rows(name, arrays))
        print(f"finished {name} in {elapsed:.3f}s", file=sys.stderr)

    print("running NumPy/PyTorch real-connectome parity", file=sys.stderr)
    parity = parity_check(
        graph_path,
        device=device,
        dtype=dtype,
        steps=args.parity_steps,
        dt=args.dt,
        tolerance=args.parity_tolerance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "activity.npz"
    np.savez_compressed(
        arrays_path,
        **{
            f"{condition}__{name}": value
            for condition, arrays in results.items()
            for name, value in arrays.items()
        },
    )
    csv_path = args.output_dir / "probe_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_info = {
            "name": properties.name,
            "gcn_arch_name": getattr(properties, "gcnArchName", None),
            "total_memory_bytes": properties.total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    else:
        device_info = {"name": "cpu"}
    metadata = {
        "experiment": "batched circuit sensory-history experiment",
        "limitations": "Circuit-only sensory histories; no physical world was simulated.",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "command": sys.argv,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "dt": args.dt,
        "simulated_seconds": args.steps * args.dt,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device": device_info,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "HSA_OVERRIDE_GFX_VERSION",
                "PYTORCH_KERNEL_CACHE_PATH",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
            if os.environ.get(name) is not None
        },
        "connectome": {
            "path": str(graph_path),
            "sha256": graph.hash,
            "neurons": graph.n,
            "edges": len(graph.pre),
            "calibration_mae": graph.calibration_error,
        },
        "code_sha256": {
            "gpu_nursery.py": sha256(Path(__file__)),
            "torch_circuit.py": sha256(Path(__file__).parents[1] / "chreatures" / "torch_circuit.py"),
            "brain.py": sha256(Path(__file__).parents[1] / "chreatures" / "brain.py"),
        },
        "conditions": {
            name: {"learning": learning, "recurrent_edges": not silenced}
            for name, (learning, silenced) in conditions.items()
        },
        "condition_runtime_seconds": runtimes,
        "parity": parity,
        "artifacts": {
            arrays_path.name: {"sha256": sha256(arrays_path), "bytes": arrays_path.stat().st_size},
            csv_path.name: {"sha256": sha256(csv_path), "bytes": csv_path.stat().st_size},
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if not parity["within_tolerance"]:
        raise SystemExit("NumPy/PyTorch parity exceeded tolerance")


if __name__ == "__main__":
    main()
