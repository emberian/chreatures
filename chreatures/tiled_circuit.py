"""Experimental edge-tiled Triton dynamics for the full MaleCNS circuit."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .fast_circuit import PHYSIOLOGY_NAMES, TritonFusedCircuit, triton

if triton is not None:
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _edge_tiled_rate_substep(
        row_pointer, columns, weights, rate_in, rate_out, adaptation, support,
        drive, batch_size, alpha, gain, dt, support_recovery,
        BLOCK_B: tl.constexpr, EDGE_TILE: tl.constexpr, FINAL: tl.constexpr,
    ):
        """One CSR Jacobi update with parallel gathers across a small edge tile."""
        row = tl.program_id(0)
        resident = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
        resident_mask = resident < batch_size
        start = tl.load(row_pointer + row)
        stop = tl.load(row_pointer + row + 1)
        recurrent = tl.zeros((BLOCK_B,), dtype=tl.float32)
        # EDGE_TILE is deliberately small: [E, B32] rate values stay bounded.
        for base in tl.range(start, stop, EDGE_TILE, num_stages=1):
            lane = tl.arange(0, EDGE_TILE)
            position = base + lane
            edge_mask = position < stop
            source = tl.load(columns + position, mask=edge_mask, other=0)
            weight = tl.load(weights + position, mask=edge_mask, other=0.0)
            offsets = source[:, None] * batch_size + resident[None, :]
            values = tl.load(rate_in + offsets,
                             mask=edge_mask[:, None] & resident_mask[None, :],
                             other=0.0)
            recurrent += tl.sum(weight[:, None] * values, axis=0)

        offset = row * batch_size + resident
        old_rate = tl.load(rate_in + offset, mask=resident_mask, other=0.0)
        old_adaptation = tl.load(adaptation + offset, mask=resident_mask, other=0.0)
        old_support = tl.load(support + offset, mask=resident_mask, other=1.0)
        sensory = tl.load(drive + offset, mask=resident_mask, other=0.0)
        target = tl.maximum(libdevice.tanh(
            0.005 + sensory + gain * recurrent - 0.10 * old_adaptation), 0.0)
        new_rate = old_rate + alpha * (target * old_support - old_rate)
        tl.store(rate_out + offset, new_rate, mask=resident_mask)
        if FINAL:
            tl.store(adaptation + offset,
                     old_adaptation + dt / 5.0 * (new_rate - old_adaptation),
                     mask=resident_mask)
            new_support = old_support + dt * (
                support_recovery * (1.0 - old_support) - 0.003 * new_rate)
            tl.store(support + offset,
                     tl.maximum(0.65, tl.minimum(1.0, new_support)),
                     mask=resident_mask)


class EdgeTiledTritonCircuit(TritonFusedCircuit):
    """Canonical-state MaleCNS backend with an edge-parallel reduction tree."""

    layout = "neuron_major_triton_edge_tiled"

    def __init__(self, graph: Any, batch_size: int, *, edge_tile: int = 8,
                 num_warps: int = 1, **kwargs: Any) -> None:
        if edge_tile not in (4, 8, 16):
            raise ValueError("edge_tile must be one of 4, 8, 16")
        if num_warps not in (1, 2, 4):
            raise ValueError("num_warps must be one of 1, 2, 4")
        super().__init__(graph, batch_size, **kwargs)
        self.edge_tile = int(edge_tile)
        self.num_warps = int(num_warps)

    def _launch_pair(self, drive: torch.Tensor, dt: float) -> None:
        alpha = min(1.0, dt / 2 / self.tau)
        grid = (self.n, triton.cdiv(self.batch_size, self.resident_tile))
        head = (self._row_pointer, self._columns, self._weights)
        tail = (self.batch_size, alpha, self.gain, dt, self.support_recovery)
        options = {"BLOCK_B": self.resident_tile, "EDGE_TILE": self.edge_tile,
                   "num_warps": self.num_warps}
        _edge_tiled_rate_substep[grid](*head, self.rates, self.rate_buffer,
            self.adaptation, self.support, drive, *tail, FINAL=False, **options)
        _edge_tiled_rate_substep[grid](*head, self.rate_buffer, self.rates,
            self.adaptation, self.support, drive, *tail, FINAL=True, **options)

    @torch.no_grad()
    def step_device(self, channels: torch.Tensor, dt: float,
                    *, validate: bool = True) -> torch.Tensor:
        if validate:
            self._validate_device_channels(channels)
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        drive = torch.sparse.mm(self.input_matrix, channels).contiguous()
        self._launch_pair(drive, dt)
        readout = torch.sparse.mm(self.readout_matrix, self.rates)
        physiology = torch.stack((self.rates.mean(0), self.rates.amax(0),
                                  self.support.mean(0)))
        return torch.cat((readout, physiology), dim=0).T.contiguous()

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update({
            "engine": "fixed-cohort-triton-edge-tiled-csr-v1-experimental",
            "edge_tile": self.edge_tile,
            "num_warps": self.num_warps,
            "edge_reduction": "float32 tile tree then serial tile accumulation",
            "within_row_edge_order": "grouped in canonical consecutive tiles",
            "canonical_external_order": True,
            "physiology_names": list(PHYSIOLOGY_NAMES),
        })
        return value


class MaleCNSEdgeTiledCircuit(EdgeTiledTritonCircuit):
    """Pinned E8/W1 candidate with the standard fixed-circuit constructor API."""

    def __init__(self, graph: Any, batch_size: int, **kwargs: Any) -> None:
        super().__init__(graph, batch_size, edge_tile=8, num_warps=1, **kwargs)

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value["engine"] = "fixed-cohort-triton-edge8-csr-v1"
        value["selection_basis"] = "best complete-path median on RX6750XT B48"
        return value
