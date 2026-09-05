"""Lossless cache-oriented row schedules for the full MaleCNS CSR circuit.

The graph and all externally visible tensors stay in canonical neuron order.  A
layout only changes the order in which independent postsynaptic rows are issued
to the GPU.  Edges within a row are never reordered, so recurrent float32 sums
retain the canonical acquisition order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .fast_circuit import TritonFusedCircuit, _csr_rate_substep, triton


LAYOUT_VERSION = "malecns-row-schedule-v1"


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


@dataclass(frozen=True)
class CircuitRowLayout:
    """A bijective execution schedule mapping launch rows to canonical rows."""

    row_order: np.ndarray
    inverse_order: np.ndarray
    graph_hash: str
    source_page_neurons: int
    method: str = "anatomy-source-page"

    @classmethod
    def build(cls, graph: Any, *, source_page_neurons: int = 2048,
              method: str = "anatomy-source-page") -> "CircuitRowLayout":
        if source_page_neurons < 32:
            raise ValueError("source_page_neurons must be at least 32")
        n = int(graph.n)
        pointers = np.asarray(graph.indptr)
        columns = np.asarray(graph.indices)
        degree = np.diff(pointers)

        # Edges are strictly source-sorted within each canonical row.  Quartile
        # samples cheaply describe its source working set without E-sized scratch.
        samples = np.full((n, 3), n // source_page_neurons + 1, dtype=np.int32)
        nonempty = np.flatnonzero(degree)
        starts = pointers[nonempty]
        lengths = degree[nonempty]
        for slot, numerator in enumerate((1, 2, 3)):
            positions = starts + ((lengths - 1) * numerator // 4)
            samples[nonempty, slot] = columns[positions] // source_page_neurons

        if method not in {"anatomy-source-page", "degree-descending"}:
            raise ValueError("unknown row scheduling method")
        # Anatomical keys make the schedule stable and keep related cells close;
        # source-page keys then cluster rows likely to reuse the same rate pages.
        fields = ("superclasses", "classes", "soma_neuromeres", "sides")
        anatomy = []
        for name in fields:
            values = np.asarray(getattr(graph, name)).astype(str)
            _, encoded = np.unique(values, return_inverse=True)
            anatomy.append(encoded.astype(np.int32, copy=False))
        canonical = np.arange(n, dtype=np.int32)
        if method == "degree-descending":
            # Drain expensive rows first so a small set of high-degree cells does
            # not form a serialized tail at the end of each synchronized substep.
            row_order = np.lexsort((canonical, -degree)).astype(np.int32, copy=False)
        else:
            row_order = np.lexsort(
                (canonical, degree, samples[:, 2], samples[:, 0], samples[:, 1], *anatomy[::-1])
            ).astype(np.int32, copy=False)
        inverse = np.empty(n, dtype=np.int32)
        inverse[row_order] = canonical
        return cls(row_order, inverse, str(graph.hash), int(source_page_neurons), method)

    def validate(self, graph: Any) -> None:
        n = int(graph.n)
        if self.graph_hash != str(graph.hash):
            raise ValueError("layout graph hash does not match the loaded graph")
        if self.row_order.dtype != np.int32 or self.row_order.shape != (n,):
            raise ValueError("row_order must be int32[N]")
        if self.inverse_order.dtype != np.int32 or self.inverse_order.shape != (n,):
            raise ValueError("inverse_order must be int32[N]")
        canonical = np.arange(n, dtype=np.int32)
        if not np.array_equal(np.sort(self.row_order), canonical):
            raise ValueError("row_order is not a permutation")
        if not np.array_equal(self.inverse_order[self.row_order], canonical):
            raise ValueError("inverse_order does not invert row_order")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(self.metadata(), sort_keys=True)
        np.savez(path, row_order=self.row_order, inverse_order=self.inverse_order,
                 metadata=np.asarray(metadata))

    @classmethod
    def load(cls, path: str | Path, graph: Any) -> "CircuitRowLayout":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("version") != LAYOUT_VERSION:
                raise ValueError("unsupported circuit row layout version")
            layout = cls(
                np.asarray(archive["row_order"], dtype=np.int32),
                np.asarray(archive["inverse_order"], dtype=np.int32),
                str(metadata["graph_sha256"]), int(metadata["source_page_neurons"]),
                str(metadata["method"]),
            )
        layout.validate(graph)
        if metadata["row_order_sha256"] != _array_hash(layout.row_order):
            raise ValueError("layout row-order checksum mismatch")
        return layout

    def metadata(self) -> dict[str, Any]:
        return {"version": LAYOUT_VERSION, "method": self.method,
                "graph_sha256": self.graph_hash,
                "source_page_neurons": self.source_page_neurons,
                "row_order_sha256": _array_hash(self.row_order),
                "canonical_external_order": True, "within_row_edge_order": "unchanged"}


if triton is not None:
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _scheduled_csr_rate_substep(
        schedule, row_pointer, columns, weights, rate_in, rate_out, adaptation,
        support, drive, batch_size, alpha, gain, dt, support_recovery,
        BLOCK_B: tl.constexpr, FINAL: tl.constexpr,
    ):
        launch_row = tl.program_id(0)
        row = tl.load(schedule + launch_row)
        resident = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
        mask = resident < batch_size
        edge = tl.load(row_pointer + row)
        stop = tl.load(row_pointer + row + 1)
        recurrent = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for position in tl.range(edge, stop, num_stages=1):
            source = tl.load(columns + position)
            weight = tl.load(weights + position)
            recurrent += weight * tl.load(rate_in + source * batch_size + resident,
                                          mask=mask, other=0.0)
        offset = row * batch_size + resident
        old_rate = tl.load(rate_in + offset, mask=mask, other=0.0)
        old_adaptation = tl.load(adaptation + offset, mask=mask, other=0.0)
        old_support = tl.load(support + offset, mask=mask, other=1.0)
        sensory = tl.load(drive + offset, mask=mask, other=0.0)
        target = tl.maximum(libdevice.tanh(0.005 + sensory + gain * recurrent
                                           - 0.10 * old_adaptation), 0.0)
        new_rate = old_rate + alpha * (target * old_support - old_rate)
        tl.store(rate_out + offset, new_rate, mask=mask)
        if FINAL:
            tl.store(adaptation + offset, old_adaptation + dt / 5.0 *
                     (new_rate - old_adaptation), mask=mask)
            new_support = old_support + dt * (support_recovery * (1.0 - old_support)
                                               - 0.003 * new_rate)
            tl.store(support + offset, tl.maximum(0.65, tl.minimum(1.0, new_support)),
                     mask=mask)


class ScheduledTritonCircuit(TritonFusedCircuit):
    """Triton circuit whose independent CSR rows use a cache-aware schedule."""

    layout = "canonical_state_scheduled_rows"

    def __init__(self, graph: Any, batch_size: int, *, row_layout: CircuitRowLayout,
                 **kwargs: Any) -> None:
        row_layout.validate(graph)
        super().__init__(graph, batch_size, **kwargs)
        self.row_layout = row_layout
        self._schedule = torch.as_tensor(row_layout.row_order, device=self.device)

    @torch.no_grad()
    def step_device(self, channels: torch.Tensor, dt: float, *, validate: bool = True) -> torch.Tensor:
        if validate:
            self._validate_device_channels(channels)
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        drive = torch.sparse.mm(self.input_matrix, channels).contiguous()
        alpha = min(1.0, dt / 2 / self.tau)
        grid = (self.n, triton.cdiv(self.batch_size, self.resident_tile))
        common = (self._schedule, self._row_pointer, self._columns, self._weights)
        _scheduled_csr_rate_substep[grid](*common, self.rates, self.rate_buffer,
            self.adaptation, self.support, drive, self.batch_size, alpha, self.gain,
            dt, self.support_recovery, BLOCK_B=self.resident_tile, FINAL=False, num_warps=1)
        _scheduled_csr_rate_substep[grid](*common, self.rate_buffer, self.rates,
            self.adaptation, self.support, drive, self.batch_size, alpha, self.gain,
            dt, self.support_recovery, BLOCK_B=self.resident_tile, FINAL=True, num_warps=1)
        readout = torch.sparse.mm(self.readout_matrix, self.rates)
        physiology = torch.stack((self.rates.mean(0), self.rates.amax(0), self.support.mean(0)))
        return torch.cat((readout, physiology), dim=0).T.contiguous()

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update({"engine": "fixed-cohort-triton-scheduled-csr-v1-experimental",
                      "row_schedule": self.row_layout.metadata()})
        return value
