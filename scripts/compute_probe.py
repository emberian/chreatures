#!/usr/bin/env python3
"""Small, reproducible PyTorch CPU/ROCm compute probe.

The probe checks dense matrix multiplication, indexed accumulation, and sparse
matrix multiplication.  It prints JSON so results can be archived and compared.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from typing import Any, Callable

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    operation: Callable[[], Any], device: torch.device, warmup: int, repetitions: int
) -> float:
    for _ in range(warmup):
        operation()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repetitions):
        operation()
    synchronize(device)
    return (time.perf_counter() - started) / repetitions


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    difference = (actual - expected).abs()
    denominator = expected.abs().clamp_min(1e-7)
    return {
        "max_abs_error": difference.max().item(),
        "max_rel_error": (difference / denominator).max().item(),
        "mean_abs_error": difference.mean().item(),
    }


def dense_probe(
    device: torch.device, size: int, warmup: int, repetitions: int, dtype: torch.dtype
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(1)
    lhs = torch.randn((size, size), device=device, dtype=dtype, generator=generator)
    rhs = torch.randn((size, size), device=device, dtype=dtype, generator=generator)
    output = torch.empty_like(lhs)

    def operation() -> None:
        torch.mm(lhs, rhs, out=output)

    seconds = measure(operation, device, warmup, repetitions)

    check_size = min(size, 256)
    check_lhs = lhs[:check_size, :check_size]
    check_rhs = rhs[:check_size, :check_size]
    actual = check_lhs @ check_rhs
    expected = check_lhs.float().cpu() @ check_rhs.float().cpu()
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "size": size,
        "seconds": seconds,
        "tflops": (2.0 * size**3) / seconds / 1e12,
        "correctness_size": check_size,
        **error_metrics(actual, expected),
    }


def index_add_probe(
    device: torch.device,
    rows: int,
    updates: int,
    width: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(2)
    indices = torch.randint(rows, (updates,), device=device, generator=generator)
    source = torch.randn((updates, width), device=device, generator=generator)
    output = torch.zeros((rows, width), device=device)

    def operation() -> None:
        output.zero_()
        output.index_add_(0, indices, source)

    seconds = measure(operation, device, warmup, repetitions)

    check_updates = min(updates, 4096)
    check_rows = min(rows, 1024)
    check_indices = indices[:check_updates] % check_rows
    check_source = source[:check_updates]
    actual = torch.zeros((check_rows, width), device=device)
    actual.index_add_(0, check_indices, check_source)
    expected = torch.zeros((check_rows, width)).index_add_(
        0, check_indices.cpu(), check_source.cpu()
    )
    return {
        "rows": rows,
        "updates": updates,
        "width": width,
        "seconds": seconds,
        "million_updates_per_second": updates / seconds / 1e6,
        "million_values_per_second": updates * width / seconds / 1e6,
        "correctness_updates": check_updates,
        **error_metrics(actual, expected),
    }


def sparse_mm_probe(
    device: torch.device,
    size: int,
    nonzeros: int,
    width: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(3)
    indices = torch.randint(
        size, (2, nonzeros), device=device, generator=generator
    )
    values = torch.randn((nonzeros,), device=device, generator=generator)
    matrix = torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()
    dense = torch.randn((size, width), device=device, generator=generator)
    output: torch.Tensor | None = None

    def operation() -> None:
        nonlocal output
        output = torch.sparse.mm(matrix, dense)

    seconds = measure(operation, device, warmup, repetitions)
    effective_nonzeros = matrix._nnz()

    check_size = min(size, 256)
    check_nonzeros = min(nonzeros, 4096)
    check_indices = indices[:, :check_nonzeros] % check_size
    check_values = values[:check_nonzeros]
    check_matrix = torch.sparse_coo_tensor(
        check_indices, check_values, (check_size, check_size)
    ).coalesce()
    check_dense = dense[:check_size]
    actual = torch.sparse.mm(check_matrix, check_dense)
    expected = check_matrix.cpu().to_dense() @ check_dense.cpu()
    return {
        "size": size,
        "requested_nonzeros": nonzeros,
        "effective_nonzeros": effective_nonzeros,
        "width": width,
        "seconds": seconds,
        "gflops": (2.0 * effective_nonzeros * width) / seconds / 1e9,
        "correctness_size": check_size,
        **error_metrics(actual, expected),
    }


def capture(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"status": "ok", **function()}
    except Exception as error:  # Keep independent kernels diagnosable.
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--matmul-size", type=int, default=2048)
    parser.add_argument("--index-rows", type=int, default=65536)
    parser.add_argument("--index-updates", type=int, default=262144)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--sparse-size", type=int, default=4096)
    parser.add_argument("--sparse-nnz", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()

    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested == "auto":
        requested = "cpu"
    device = torch.device(requested)

    report: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "HSA_OVERRIDE_GFX_VERSION",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "PYTORCH_KERNEL_CACHE_PATH",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
            if os.environ.get(name) is not None
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "warp_size": properties.warp_size,
            "gcn_arch_name": getattr(properties, "gcnArchName", None),
        }
        report["memory_before_bytes"] = {
            "free": torch.cuda.mem_get_info(device)[0],
            "total": torch.cuda.mem_get_info(device)[1],
        }

    report["dense_fp32"] = capture(
        "dense_fp32",
        lambda: dense_probe(
            device, args.matmul_size, args.warmup, args.repetitions, torch.float32
        ),
    )
    report["dense_fp16"] = capture(
        "dense_fp16",
        lambda: dense_probe(
            device, args.matmul_size, args.warmup, args.repetitions, torch.float16
        ),
    )
    report["index_add_fp32"] = capture(
        "index_add_fp32",
        lambda: index_add_probe(
            device,
            args.index_rows,
            args.index_updates,
            args.width,
            args.warmup,
            args.repetitions,
        ),
    )
    report["sparse_mm_fp32"] = capture(
        "sparse_mm_fp32",
        lambda: sparse_mm_probe(
            device,
            args.sparse_size,
            args.sparse_nnz,
            args.width,
            args.warmup,
            args.repetitions,
        ),
    )
    if device.type == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
