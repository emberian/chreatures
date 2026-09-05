#!/usr/bin/env python3
"""Build, validate, and benchmark a cache-aware full-MaleCNS row schedule."""

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

from chreatures.circuit_layout import (  # noqa: E402
    CircuitRowLayout, ScheduledTritonCircuit, _scheduled_csr_rate_substep,
)
from chreatures.fast_circuit import TritonFusedCircuit, _csr_rate_substep, triton  # noqa: E402
from chreatures.malecns import MaleCNSGraph  # noqa: E402
from chreatures.neural_ports import NeuralPortBundle  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def locality(graph: MaleCNSGraph, order: np.ndarray, page: int) -> dict[str, float]:
    """Report adjacent-row source-page overlap from bounded quartile samples."""
    ptr, col = np.asarray(graph.indptr), np.asarray(graph.indices)
    degree = np.diff(ptr)
    signatures = np.full((graph.n, 3), -1, np.int32)
    rows = np.flatnonzero(degree)
    for slot, q in enumerate((1, 2, 3)):
        signatures[rows, slot] = col[ptr[rows] + (degree[rows] - 1) * q // 4] // page
    scheduled = signatures[order]
    overlap = (scheduled[1:, :, None] == scheduled[:-1, None, :]).any(axis=(1, 2))
    return {"adjacent_quartile_page_overlap_fraction": float(overlap.mean()),
            "sampled_rows": int(graph.n), "source_page_neurons": int(page)}


def timed(operation, device: torch.device, warmup: int, iterations: int, repeats: int):
    for _ in range(warmup):
        operation()
    sync(device)
    values = []
    for _ in range(repeats):
        begin = time.perf_counter()
        for _ in range(iterations):
            operation()
        sync(device)
        values.append((time.perf_counter() - begin) * 1000 / iterations)
    return {"median_ms": statistics.median(values), "minimum_ms": min(values),
            "maximum_ms": max(values), "samples_ms": values}


def max_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b), initial=0.0))


def launch_recurrent_pair(circuit, drive: torch.Tensor, dt: float) -> None:
    """Time only the two recurrent/update kernels, excluding ports and summaries."""
    alpha = min(1.0, dt / 2 / circuit.tau)
    grid = (circuit.n, triton.cdiv(circuit.batch_size, circuit.resident_tile))
    tail = (circuit.batch_size, alpha, circuit.gain, dt, circuit.support_recovery)
    options = {"BLOCK_B": circuit.resident_tile, "num_warps": 1}
    if isinstance(circuit, ScheduledTritonCircuit):
        head = (circuit._schedule, circuit._row_pointer, circuit._columns, circuit._weights)
        kernel = _scheduled_csr_rate_substep
    else:
        head = (circuit._row_pointer, circuit._columns, circuit._weights)
        kernel = _csr_rate_substep
    kernel[grid](*head, circuit.rates, circuit.rate_buffer, circuit.adaptation,
                 circuit.support, drive, *tail, FINAL=False, **options)
    kernel[grid](*head, circuit.rate_buffer, circuit.rates, circuit.adaptation,
                 circuit.support, drive, *tail, FINAL=True, **options)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=Path("/tank/chreatures/data/malecns/derived"))
    p.add_argument("--ports", type=Path, default=Path("/tank/chreatures/data/ports/retinal-v1-maps.npz"))
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--page-neurons", type=int, default=2048)
    p.add_argument("--method", choices=("anatomy-source-page", "degree-descending"),
                   default="anatomy-source-page")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args()

    graph = MaleCNSGraph.load(args.graph, mmap=True)
    if args.layout.exists():
        layout = CircuitRowLayout.load(args.layout, graph)
    else:
        layout = CircuitRowLayout.build(graph, source_page_neurons=args.page_neurons,
                                        method=args.method)
        layout.save(args.layout)
    canonical = np.arange(graph.n, dtype=np.int32)
    report = {"graph": graph.summary(), "layout": layout.metadata(),
              "canonical_locality": locality(graph, canonical, layout.source_page_neurons),
              "scheduled_locality": locality(graph, layout.row_order, layout.source_page_neurons)}
    if args.build_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    device = torch.device(args.device)
    ports = NeuralPortBundle.load(args.ports, graph)
    recurrent = graph.matrix(normalized=True, signed=True)
    input_map = (list(ports.input_names), ports.input_map)
    readout_map = (list(ports.readout_names), ports.readout_map)
    base = TritonFusedCircuit(graph, args.batch_size, device=device,
                              recurrent_matrix=recurrent, input_map=input_map,
                              readout_map=readout_map)
    candidate = ScheduledTritonCircuit(graph, args.batch_size, device=device,
                                       recurrent_matrix=recurrent, input_map=input_map,
                                       readout_map=readout_map, row_layout=layout)
    rng = np.random.default_rng(9917)
    initial = {"rates": rng.random((args.batch_size, graph.n), dtype=np.float32) * .1,
               "adaptation": rng.random((args.batch_size, graph.n), dtype=np.float32) * .03,
               "support": .8 + rng.random((args.batch_size, graph.n), dtype=np.float32) * .2,
               "times": np.zeros(args.batch_size, np.float64)}
    channels = np.ascontiguousarray(rng.random((base.input_count, args.batch_size), dtype=np.float32))
    base.import_state(initial); candidate.import_state(initial)
    expected = base.step_numpy(channels, .05); actual = candidate.step_numpy(channels, .05)
    left, right = base.export_state(), candidate.export_state()
    report["parity"] = {
        "features_max_abs_delta": max_delta(expected.features, actual.features),
        "physiology_max_abs_delta": max_delta(expected.physiology, actual.physiology),
        "times_max_abs_delta": max_delta(expected.times, actual.times),
        "state_max_abs_delta": {key: max_delta(left[key], right[key]) for key in left},
    }
    dev_channels = torch.from_numpy(channels).to(device)
    sequence = []
    timings = {"canonical": [], "scheduled": []}
    # ABBA ordering limits clock/thermal drift; each observation is a complete
    # two-substep path including input projection, readout, and physiology.
    for _ in range(args.repeats):
        for name, circuit in (("canonical", base), ("scheduled", candidate),
                              ("scheduled", candidate), ("canonical", base)):
            value = timed(lambda c=circuit: c.step_device(dev_channels, .05, validate=False),
                          device, args.warmup, args.iterations, 1)["median_ms"]
            timings[name].append(value); sequence.append({"engine": name, "ms": value})
    medians = {key: statistics.median(value) for key, value in timings.items()}
    report["complete_path_abba"] = {"sequence": sequence, "median_ms": medians,
        "scheduled_speedup": medians["canonical"] / medians["scheduled"],
        "resident_steps_per_second": {key: args.batch_size * 1000 / value
                                       for key, value in medians.items()}}
    drive = torch.sparse.mm(base.input_matrix, dev_channels).contiguous()
    pure = {}
    for name, circuit in (("canonical", base), ("scheduled", candidate)):
        pure[name] = timed(lambda c=circuit: launch_recurrent_pair(c, drive, .05), device,
                           args.warmup, args.iterations, args.repeats)
    report["two_recurrent_update_kernels"] = pure
    report["two_recurrent_update_kernel_speedup"] = (
        pure["canonical"]["median_ms"] / pure["scheduled"]["median_ms"]
    )
    report["device"] = str(device)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
