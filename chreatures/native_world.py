"""Batch adapter for the required native physical world kernels."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def load_world_kernels() -> Any:
    """Load the current engine; there is no alternate production contact loop."""
    try:
        return importlib.import_module("_world_kernels")
    except ImportError as exc:
        raise RuntimeError(
            "native world backend requested but _world_kernels is unavailable; "
            "build native/world-kernels for this Python interpreter"
        ) from exc


class NativeContactBatch:
    """One-call wrapper around reusable native MuJoCo contact scratch."""

    def __init__(self, capacity: int = 256) -> None:
        self._native = load_world_kernels().ContactBatch(capacity)

    def evaluate(
        self, model: Any, data: Any, timestep: float,
        impulse_limit: float, work_limit: float,
    ) -> tuple[np.ndarray, ...]:
        model_address = int(getattr(model, "_address", 0))
        data_address = int(getattr(data, "_address", 0))
        if not model_address or not data_address:
            raise RuntimeError("MuJoCo Python objects do not expose native addresses")
        values = self._native.evaluate(
            model_address, data_address, int(data.ncon), float(timestep),
            float(impulse_limit), float(work_limit),
        )
        arrays = tuple(np.asarray(value) for value in values)
        expected = (
            (data.ncon,), (data.ncon,), (data.ncon, 3), (data.ncon, 3),
            (data.ncon,), (data.ncon,), (data.ncon,), (data.ncon,),
        )
        if tuple(value.shape for value in arrays) != expected:
            raise RuntimeError("native contact kernel returned malformed arrays")
        return arrays


__all__ = ["NativeContactBatch", "load_world_kernels"]
