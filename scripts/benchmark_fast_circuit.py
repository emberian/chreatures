#!/usr/bin/env python3
"""Benchmark neuron-major and resident-major full MaleCNS GPU dynamics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.fast_circuit import (  # noqa: E402
    MicrobatchedResidentCircuit,
    NeuronMajorCircuit,
    ResidentMajorReferenceCircuit,
)
from chreatures.malecns import MaleCNSGraph  # noqa: E402
from chreatures.neural_ports import NeuralPortBundle  # noqa: E402


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    operation: Callable[[], Any],
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        synchronize(device)
        samples.append((time.perf_counter() - started) * 1000 / iterations)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def maximum_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right), initial=0))


def correctness_check(
    graph: MaleCNSGraph,
    recurrent_matrix: Any,
    input_map: Any,
    readout_map: Any,
    device: torch.device,
) -> dict[str, Any]:
    batch_size = 3
    reference = ResidentMajorReferenceCircuit(
        graph,
        batch_size,
        device=device,
        recurrent_matrix=recurrent_matrix,
        input_map=input_map,
        readout_map=readout_map,
    )
    fast = NeuronMajorCircuit(
        graph,
        batch_size,
        device=device,
        recurrent_matrix=recurrent_matrix,
        input_map=input_map,
        readout_map=readout_map,
    )
    rng = np.random.default_rng(7301)
    streams = []
    for _ in range(5):
        channels = rng.random((fast.input_count, batch_size), dtype=np.float32)
        channels[rng.random(channels.shape) < 0.35] = 0
        streams.append(np.ascontiguousarray(channels))
    state_delta = {name: 0.0 for name in ("rates", "adaptation", "support", "times")}
    output_delta = {"features": 0.0, "physiology": 0.0, "times": 0.0}
    for channels in streams[:4]:
        expected = reference.step_numpy(channels, 0.05)
        actual = fast.step_numpy(channels, 0.05)
        output_delta["features"] = max(
            output_delta["features"], maximum_delta(expected.features, actual.features)
        )
        output_delta["physiology"] = max(
            output_delta["physiology"],
            maximum_delta(expected.physiology, actual.physiology),
        )
        output_delta["times"] = max(
            output_delta["times"], maximum_delta(expected.times, actual.times)
        )
        expected_state = reference.export_state()
        actual_state = fast.export_state()
        for name in state_delta:
            state_delta[name] = max(
                state_delta[name], maximum_delta(expected_state[name], actual_state[name])
            )

    snapshot = fast.export_state()
    first = fast.step_numpy(streams[4], 0.05)
    first_state = fast.export_state()
    fast.import_state(snapshot)
    replay = fast.step_numpy(streams[4], 0.05)
    replay_state = fast.export_state()
    replay_delta = {
        "features": maximum_delta(first.features, replay.features),
        "physiology": maximum_delta(first.physiology, replay.physiology),
        "times": maximum_delta(first.times, replay.times),
        "state": max(
            maximum_delta(first_state[name], replay_state[name])
            for name in ("rates", "adaptation", "support", "times")
        ),
    }
    result = {
        "batch_size": batch_size,
        "steps": 4,
        "state_max_abs_delta": state_delta,
        "output_max_abs_delta": output_delta,
        "snapshot_replay_max_abs_delta": replay_delta,
        "neural_state_and_features_exact": not any(state_delta.values())
        and output_delta["features"] == 0
        and output_delta["times"] == 0,
        "within_float32_reduction_tolerance": output_delta["physiology"] <= 5e-7,
        "snapshot_replay_exact": not any(replay_delta.values()),
    }
    del reference, fast
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def benchmark_layout(
    circuit_type: type,
    graph: MaleCNSGraph,
    recurrent_matrix: Any,
    input_map: Any,
    readout_map: Any,
    batch_size: int,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    circuit_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    circuit = circuit_type(
        graph,
        batch_size,
        device=device,
        recurrent_matrix=recurrent_matrix,
        input_map=input_map,
        readout_map=readout_map,
        **(circuit_kwargs or {}),
    )
    rng = np.random.default_rng(1000 + batch_size)
    host_channels = rng.random(
        (circuit.input_count, batch_size), dtype=np.float32
    )
    host_channels[rng.random(host_channels.shape) < 0.35] = 0
    host_channels = np.ascontiguousarray(host_channels)
    device_channels = torch.from_numpy(host_channels).to(device)
    full_device = measure(
        lambda: circuit.step_device(device_channels, 0.05, validate=False),
        device,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    end_to_end = measure(
        lambda: circuit.step_numpy(host_channels, 0.05),
        device,
        warmup=1,
        iterations=max(1, iterations // 2),
        repeats=repeats,
    )
    dense_operand = (
        circuit.rates
        if circuit.layout == "neuron_major"
        else circuit.rates.T
    )
    spmm = measure(
        lambda: torch.sparse.mm(circuit.matrix, dense_operand),
        device,
        warmup=warmup,
        iterations=iterations * 2,
        repeats=repeats,
    )
    result = {
        **circuit.metadata(),
        "dense_operand": {
            "shape": list(dense_operand.shape),
            "stride": list(dense_operand.stride()),
            "contiguous": dense_operand.is_contiguous(),
        },
        "full_device_step": full_device,
        "end_to_end_numpy_step": end_to_end,
        "isolated_recurrent_spmm": spmm,
        "resident_steps_per_second_device": batch_size
        / (full_device["median_ms"] / 1000),
        "resident_steps_per_second_end_to_end": batch_size
        / (end_to_end["median_ms"] / 1000),
    }
    if device.type == "cuda":
        result["peak_torch_bytes"] = int(torch.cuda.max_memory_allocated(device))
    del circuit, dense_operand, device_channels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=Path("/tank/chreatures/data/malecns/derived")
    )
    parser.add_argument(
        "--port-bundle",
        type=Path,
        default=Path("/tank/chreatures/data/ports/retinal-v1-maps.npz"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=[3, 48, 96, 192])
    parser.add_argument(
        "--microbatches", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24]
    )
    parser.add_argument("--microbatch-total", type=int, default=48)
    parser.add_argument("--only-microbatches", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(batch < 1 or batch > 4096 for batch in args.batches):
        parser.error("batch sizes must be in 1..4096")
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations, and repeats must be positive")

    device = torch.device(args.device)
    graph = MaleCNSGraph.load(args.graph, mmap=True)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    recurrent_matrix = graph.matrix(normalized=True, signed=True)
    input_map = ports.input_names, ports.input_map
    readout_map = ports.readout_names, ports.readout_map
    report: dict[str, Any] = {
        "schema_version": 1,
        "graph_sha256": graph.hash,
        "neurons": graph.n,
        "edges": graph.edge_count,
        "ports": {
            "name": ports.spec["name"],
            "spec_sha256": ports.spec_hash,
            "inputs": len(ports.input_names),
            "readouts": len(ports.readout_names),
        },
        "device": {
            "type": device.type,
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "benchmark": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "dtype": "float32",
            "substeps": 2,
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report["device"].update(
            {"name": properties.name, "gcn_arch_name": properties.gcnArchName}
        )
    if not args.only_microbatches:
        print("running full-state and snapshot replay comparison", flush=True)
        report["correctness"] = correctness_check(
            graph, recurrent_matrix, input_map, readout_map, device
        )
        print(
            json.dumps({"correctness": report["correctness"]}, sort_keys=True),
            flush=True,
        )
    rows = []
    if not args.only_microbatches:
        for batch_size in args.batches:
            for circuit_type in (ResidentMajorReferenceCircuit, NeuronMajorCircuit):
                print(
                    f"benchmarking layout={circuit_type.layout} batch={batch_size}",
                    flush=True,
                )
                row = benchmark_layout(
                    circuit_type,
                    graph,
                    recurrent_matrix,
                    input_map,
                    readout_map,
                    batch_size,
                    device,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    for microbatch_size in args.microbatches:
        if microbatch_size > args.microbatch_total:
            continue
        circuit_type = MicrobatchedResidentCircuit
        batch_size = args.microbatch_total
        print(
            f"benchmarking layout={circuit_type.layout} batch={batch_size} "
            f"microbatch={microbatch_size}",
            flush=True,
        )
        row = benchmark_layout(
            circuit_type,
            graph,
            recurrent_matrix,
            input_map,
            readout_map,
            batch_size,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            circuit_kwargs={"microbatch_size": microbatch_size},
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    report["results"] = rows
    report["finished_unix"] = time.time()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
