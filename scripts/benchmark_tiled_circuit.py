#!/usr/bin/env python3
"""Compare E4/E8/E16 tiled recurrence against canonical full-MaleCNS Triton."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.fast_circuit import TritonFusedCircuit, _csr_rate_substep, triton  # noqa: E402
from chreatures.malecns import MaleCNSGraph  # noqa: E402
from chreatures.neural_ports import NeuralPortBundle  # noqa: E402
from chreatures.tiled_circuit import EdgeTiledTritonCircuit  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    operation, device: torch.device, *, warmup: int, iterations: int, repeats: int
) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    sync(device)
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter()
        for _ in range(iterations):
            operation()
        sync(device)
        samples.append((time.perf_counter() - begin) * 1000 / iterations)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


def launch_baseline(
    circuit: TritonFusedCircuit, drive: torch.Tensor, dt: float
) -> None:
    alpha = min(1.0, dt / 2 / circuit.tau)
    grid = (circuit.n, triton.cdiv(circuit.batch_size, circuit.resident_tile))
    head = (circuit._row_pointer, circuit._columns, circuit._weights)
    tail = (circuit.batch_size, alpha, circuit.gain, dt, circuit.support_recovery)
    options = {"BLOCK_B": circuit.resident_tile, "num_warps": 1}
    _csr_rate_substep[grid](
        *head,
        circuit.rates,
        circuit.rate_buffer,
        circuit.adaptation,
        circuit.support,
        drive,
        *tail,
        FINAL=False,
        **options,
    )
    _csr_rate_substep[grid](
        *head,
        circuit.rate_buffer,
        circuit.rates,
        circuit.adaptation,
        circuit.support,
        drive,
        *tail,
        FINAL=True,
        **options,
    )


def delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right), initial=0.0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph", type=Path, default=Path("/tank/chreatures/data/malecns/derived")
    )
    p.add_argument(
        "--ports",
        type=Path,
        default=Path("/tank/chreatures/data/ports/retinal-v1-maps.npz"),
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--parity-steps", type=int, default=5)
    args = p.parse_args()

    device = torch.device(args.device)
    graph = MaleCNSGraph.load(args.graph, mmap=True)
    ports = NeuralPortBundle.load(args.ports, graph)
    recurrent = graph.matrix(normalized=True, signed=True)
    common = dict(
        device=device,
        recurrent_matrix=recurrent,
        input_map=(ports.input_names, ports.input_map),
        readout_map=(ports.readout_names, ports.readout_map),
    )
    specs = ((4, 1), (8, 1), (16, 1))
    circuits = {"canonical_e1_w1": TritonFusedCircuit(graph, args.batch_size, **common)}
    circuits.update(
        {
            f"tiled_e{edge}_w{warps}": EdgeTiledTritonCircuit(
                graph, args.batch_size, edge_tile=edge, num_warps=warps, **common
            )
            for edge, warps in specs
        }
    )
    rng = np.random.default_rng(9217)
    initial = {
        "rates": rng.random((args.batch_size, graph.n), dtype=np.float32) * 0.1,
        "adaptation": rng.random((args.batch_size, graph.n), dtype=np.float32) * 0.03,
        "support": 0.8 + rng.random((args.batch_size, graph.n), dtype=np.float32) * 0.2,
        "times": np.zeros(args.batch_size, np.float64),
    }
    streams = [
        np.ascontiguousarray(
            rng.random((len(ports.input_names), args.batch_size), dtype=np.float32)
        )
        for _ in range(args.parity_steps)
    ]
    baseline = circuits["canonical_e1_w1"]
    baseline.import_state(initial)
    expected_outputs = [
        baseline.step_numpy(channels, 0.05).combined.copy() for channels in streams
    ]
    expected_state = baseline.export_state()
    parity = {}
    for name, circuit in circuits.items():
        if circuit is baseline:
            continue
        circuit.import_state(initial)
        output_max = 0.0
        for channels, expected in zip(streams, expected_outputs, strict=True):
            output_max = max(
                output_max, delta(circuit.step_numpy(channels, 0.05).combined, expected)
            )
        state = circuit.export_state()
        parity[name] = {
            "steps": args.parity_steps,
            "output_max_abs_delta": output_max,
            "state_max_abs_delta": {
                key: delta(state[key], expected_state[key]) for key in expected_state
            },
        }
        snapshot = circuit.export_state()
        first = circuit.step_numpy(streams[0], 0.05)
        first_state = circuit.export_state()
        circuit.import_state(snapshot)
        replay = circuit.step_numpy(streams[0], 0.05)
        replay_state = circuit.export_state()
        parity[name]["snapshot_replay_max_abs_delta"] = {
            "output": delta(first.combined, replay.combined),
            "state": {
                key: delta(first_state[key], replay_state[key]) for key in first_state
            },
        }

    channels = streams[-1]
    dev_channels = torch.from_numpy(channels).to(device)
    drive = torch.sparse.mm(baseline.input_matrix, dev_channels).contiguous()
    pure, complete = {}, {}
    for name, circuit in circuits.items():
        pure_op = (
            (lambda c=circuit: c._launch_pair(drive, 0.05))
            if isinstance(circuit, EdgeTiledTritonCircuit)
            else lambda c=circuit: launch_baseline(c, drive, 0.05)
        )
        pure[name] = measure(
            pure_op,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    # Interleave complete paths by repeat to reduce thermal and clock bias.
    complete_samples = {name: [] for name in circuits}
    order = list(circuits) + list(reversed(circuits))
    for _ in range(args.repeats):
        for name in order:
            circuit = circuits[name]
            timing = measure(
                lambda c=circuit: c.step_device(dev_channels, 0.05, validate=False),
                device,
                warmup=1,
                iterations=args.iterations,
                repeats=1,
            )
            complete_samples[name].append(float(timing["median_ms"]))
    for name, samples in complete_samples.items():
        complete[name] = {
            "median_ms": statistics.median(samples),
            "minimum_ms": min(samples),
            "maximum_ms": max(samples),
            "samples_ms": samples,
        }
    base_pure = float(pure["canonical_e1_w1"]["median_ms"])
    base_complete = float(complete["canonical_e1_w1"]["median_ms"])
    report = {
        "graph": graph.summary(),
        "batch_size": args.batch_size,
        "candidate_metadata": {
            name: circuit.metadata() for name, circuit in circuits.items()
        },
        "parity": parity,
        "two_recurrent_update_kernels": pure,
        "complete_path_interleaved": complete,
        "speedup": {
            name: {
                "pure": base_pure / float(pure[name]["median_ms"]),
                "complete": base_complete / float(complete[name]["median_ms"]),
            }
            for name in circuits
            if name != "canonical_e1_w1"
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
