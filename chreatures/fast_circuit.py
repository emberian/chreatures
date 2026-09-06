"""High-throughput full MaleCNS dynamics with a fixed resident cohort.

The fast layout stores state as contiguous ``[neurons, residents]`` tensors,
which is the dense operand shape consumed by CSR sparse matrix multiplication.
The reference layout stores ``[residents, neurons]`` and supplies a strided
transpose, matching :class:`chreatures.remote_brain.RemoteBrain` numerics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np
import torch
from scipy import sparse

try:  # Optional: only required by the experimental fused HIP backend.
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - exercised on non-Triton installations
    triton = None
    tl = None
    libdevice = None


PHYSIOLOGY_NAMES = ("activity_mean", "activity_peak", "support_mean")
NEURAL_VARIANT_ARRAYS = (
    "input_gain",
    "readout_gain",
    "excitability_gain",
    "recurrent_source_gain",
    "recurrent_target_gain",
    "learning_rate_gain",
    "modulator_gain",
)


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha256_f32_ones(shape: tuple[int, int]) -> str:
    digest = hashlib.sha256()
    remaining = int(np.prod(shape, dtype=np.int64))
    block = np.ones(min(remaining, 1 << 20), dtype=np.float32).tobytes()
    block_values = len(block) // 4
    while remaining:
        take = min(remaining, block_values)
        digest.update(block[: take * 4])
        remaining -= take
    return digest.hexdigest()


def _identity_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


if triton is not None:

    @triton.jit
    def _csr_rate_substep(
        row_pointer,
        columns,
        weights,
        rate_in,
        rate_out,
        recurrent_source_gain,
        recurrent_target_gain,
        excitability_gain,
        adaptation,
        support,
        drive,
        batch_size,
        alpha,
        gain,
        dt,
        support_recovery,
        BLOCK_B: tl.constexpr,
        FINAL: tl.constexpr,
    ):
        """One CSR Jacobi substep; wave lanes cover resident columns."""
        row = tl.program_id(0)
        resident = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
        resident_mask = resident < batch_size
        edge = tl.load(row_pointer + row)
        edge_stop = tl.load(row_pointer + row + 1)
        recurrent = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for position in tl.range(edge, edge_stop, num_stages=1):
            source = tl.load(columns + position)
            weight = tl.load(weights + position)
            source_rate = tl.load(
                rate_in + source * batch_size + resident,
                mask=resident_mask,
                other=0.0,
            )
            source_gain = tl.load(
                recurrent_source_gain + source * batch_size + resident,
                mask=resident_mask,
                other=1.0,
            )
            recurrent += weight * source_gain * source_rate
        offset = row * batch_size + resident
        old_rate = tl.load(rate_in + offset, mask=resident_mask, other=0.0)
        old_adaptation = tl.load(
            adaptation + offset, mask=resident_mask, other=0.0
        )
        old_support = tl.load(support + offset, mask=resident_mask, other=1.0)
        sensory_drive = tl.load(drive + offset, mask=resident_mask, other=0.0)
        target_gain = tl.load(
            recurrent_target_gain + offset, mask=resident_mask, other=1.0
        )
        excitability = tl.load(
            excitability_gain + offset, mask=resident_mask, other=1.0
        )
        activation = 0.005 + excitability * (
            sensory_drive + gain * target_gain * recurrent
        ) - 0.10 * old_adaptation
        target = tl.maximum(libdevice.tanh(activation), 0.0)
        new_rate = old_rate + alpha * (target * old_support - old_rate)
        tl.store(rate_out + offset, new_rate, mask=resident_mask)
        if FINAL:
            new_adaptation = old_adaptation + dt / 5.0 * (
                new_rate - old_adaptation
            )
            new_support = old_support + dt * (
                support_recovery * (1.0 - old_support) - 0.003 * new_rate
            )
            new_support = tl.maximum(0.65, tl.minimum(1.0, new_support))
            tl.store(adaptation + offset, new_adaptation, mask=resident_mask)
            tl.store(support + offset, new_support, mask=resident_mask)


def _torch_csr(matrix: sparse.spmatrix, device: torch.device) -> torch.Tensor:
    csr = matrix.tocsr().astype(np.float32, copy=False)
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
        csr.sort_indices()
    index_dtype = np.int32 if csr.nnz <= np.iinfo(np.int32).max else np.int64
    crow = torch.as_tensor(csr.indptr.astype(index_dtype, copy=False), device=device)
    columns = torch.as_tensor(csr.indices.astype(index_dtype, copy=False), device=device)
    values = torch.as_tensor(csr.data, dtype=torch.float32, device=device)
    return torch.sparse_csr_tensor(
        crow,
        columns,
        values,
        size=csr.shape,
        dtype=torch.float32,
        device=device,
    )


def _mapping(
    graph: Any,
    supplied: tuple[Sequence[str], sparse.spmatrix] | None,
    default_name: str,
) -> tuple[list[str], sparse.csr_matrix]:
    if supplied is None:
        supplied = getattr(graph, default_name)
        supplied = supplied() if callable(supplied) else supplied
    names = [str(name) for name in supplied[0]]
    matrix = supplied[1]
    if not names or len(names) != len(set(names)) or not sparse.issparse(matrix):
        raise ValueError(f"{default_name} has invalid names or is not sparse")
    return names, matrix.tocsr().astype(np.float32, copy=False)


@dataclass(frozen=True)
class FastStepResult:
    """One host transfer containing readouts and physiology for a cohort."""

    combined: np.ndarray
    feature_count: int
    times: np.ndarray

    @property
    def features(self) -> np.ndarray:
        return self.combined[:, : self.feature_count]

    @property
    def physiology(self) -> np.ndarray:
        return self.combined[:, self.feature_count :]

    @property
    def physiology_names(self) -> tuple[str, str, str]:
        return PHYSIOLOGY_NAMES


class _FixedCircuit:
    layout: str

    def __init__(
        self,
        graph: Any,
        batch_size: int,
        *,
        device: str | torch.device = "cuda",
        input_map: tuple[Sequence[str], sparse.spmatrix] | None = None,
        readout_map: tuple[Sequence[str], sparse.spmatrix] | None = None,
        recurrent_matrix: sparse.spmatrix | None = None,
        tau: float = 0.16,
        gain: float = 0.92,
        support_recovery: float = 0.024,
    ) -> None:
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be in 1..4096")
        if not 0 < tau or not 0 <= gain < 1:
            raise ValueError("tau must be positive and gain must be in [0, 1)")
        self.graph = graph
        self.graph_hash = str(graph.hash)
        self.n = int(graph.n)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.tau = float(tau)
        self.gain = float(gain)
        self.support_recovery = float(support_recovery)
        matrix = (
            graph.matrix(normalized=True, signed=True)
            if recurrent_matrix is None
            else recurrent_matrix
        )
        if not sparse.issparse(matrix) or matrix.shape != (self.n, self.n):
            raise ValueError("recurrent_matrix must be sparse N x N")
        self.matrix = _torch_csr(matrix, self.device)
        self.input_names, inputs = _mapping(graph, input_map, "default_input_map")
        self.readout_names, readouts = _mapping(
            graph, readout_map, "default_readout_map"
        )
        if inputs.shape != (self.n, len(self.input_names)):
            raise ValueError("input map must have shape N x channels")
        if readouts.shape != (len(self.readout_names), self.n):
            raise ValueError("readout map must have shape outputs x N")
        self.input_matrix = _torch_csr(inputs, self.device)
        self.readout_matrix = _torch_csr(readouts, self.device)
        self.times = np.zeros(self.batch_size, dtype=np.float64)
        neuron_shape = (self.n, self.batch_size)
        self.input_gain = torch.ones(
            (self.input_count, self.batch_size), dtype=self.dtype, device=self.device
        )
        self.readout_gain = torch.ones(
            (self.feature_count, self.batch_size), dtype=self.dtype, device=self.device
        )
        self.excitability_gain = torch.ones(neuron_shape, dtype=self.dtype, device=self.device)
        self.recurrent_source_gain = torch.ones_like(self.excitability_gain)
        self.recurrent_target_gain = torch.ones_like(self.excitability_gain)
        neutral_hashes = {
            "input_gain": _sha256_f32_ones((self.input_count, self.batch_size)),
            "readout_gain": _sha256_f32_ones((self.feature_count, self.batch_size)),
            **{
                name: _sha256_f32_ones(neuron_shape)
                for name in NEURAL_VARIANT_ARRAYS[2:]
            },
        }
        self._neural_variant = {
            "mode": "neutral",
            "compatibility_group": _identity_hash(
                {
                    "graph_sha256": self.graph_hash,
                    "inputs": self.input_count,
                    "readouts": self.feature_count,
                    "mode": "neutral",
                }
            ),
            "phenotype_sha256": ["neutral"] * self.batch_size,
            "array_sha256": neutral_hashes,
            "active_parameters": list(NEURAL_VARIANT_ARRAYS[:5]),
            "inactive_parameters": {
                "learning_rate_gain": "no generic local plasticity update exists in this circuit",
                "modulator_gain": "no generic modulatory current path exists in this circuit",
            },
        }
        self._neural_variant["state_identity"] = _identity_hash(self._neural_variant)

    @property
    def input_count(self) -> int:
        return len(self.input_names)

    @property
    def feature_count(self) -> int:
        return len(self.readout_names)

    def bind_neural_phenotypes(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        phenotype_sha256: Sequence[str],
        compatibility_group: str,
    ) -> str:
        """Bind one immutable phenotype column per resident.

        Array layout is channel/output/neuron-major with residents in the last
        dimension. Binding performs the sole host-to-device copies; step calls
        reuse these tensors without Python loops or reconstructed gains.
        """
        if set(arrays) != set(NEURAL_VARIANT_ARRAYS):
            raise ValueError(f"neural phenotype arrays must contain {NEURAL_VARIANT_ARRAYS}")
        if (
            not isinstance(compatibility_group, str)
            or len(compatibility_group) != 64
            or any(character not in "0123456789abcdef" for character in compatibility_group)
        ):
            raise ValueError("compatibility_group must be a lowercase SHA-256")
        phenotype_ids = [str(value) for value in phenotype_sha256]
        if len(phenotype_ids) != self.batch_size or any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in phenotype_ids
        ):
            raise ValueError("one phenotype SHA-256 is required per resident")
        expected = {
            "input_gain": (self.input_count, self.batch_size),
            "readout_gain": (self.feature_count, self.batch_size),
            **{
                name: (self.n, self.batch_size)
                for name in NEURAL_VARIANT_ARRAYS[2:]
            },
        }
        host: dict[str, np.ndarray] = {}
        for name in NEURAL_VARIANT_ARRAYS:
            value = arrays[name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise ValueError(f"{name} must be a float32 NumPy array")
            if value.shape != expected[name] or not value.flags.c_contiguous:
                raise ValueError(f"{name} must be contiguous with shape {expected[name]}")
            if not np.isfinite(value).all() or np.any((value < 0.05) | (value > 4.0)):
                raise ValueError(f"{name} must be finite in [0.05,4]")
            host[name] = value
        array_hashes = {name: _sha256_array(value) for name, value in host.items()}
        identity = {
            "mode": "candidate_phenotypes",
            "graph_sha256": self.graph_hash,
            "compatibility_group": compatibility_group,
            "phenotype_sha256": phenotype_ids,
            "array_sha256": array_hashes,
            "active_parameters": list(NEURAL_VARIANT_ARRAYS[:5]),
            "inactive_parameters": {
                "learning_rate_gain": "no generic local plasticity update exists in this circuit",
                "modulator_gain": "no generic modulatory current path exists in this circuit",
            },
        }
        identity["state_identity"] = _identity_hash(identity)
        for name in NEURAL_VARIANT_ARRAYS[:5]:
            getattr(self, name).copy_(torch.from_numpy(host[name]).to(self.device))
        self._neural_variant = identity
        return identity["state_identity"]

    @property
    def neural_variant_state_identity(self) -> str:
        return str(self._neural_variant["state_identity"])

    def _state_identity_array(self) -> np.ndarray:
        return np.asarray(self.neural_variant_state_identity)

    def _require_state_identity(self, state: Mapping[str, Any]) -> None:
        supplied = np.asarray(state.get("neural_variant_state_identity"))
        if supplied.shape != () or str(supplied) != self.neural_variant_state_identity:
            raise ValueError("snapshot neural phenotype identity differs")

    def _validate_device_channels(self, channels: torch.Tensor) -> None:
        if channels.shape != (self.input_count, self.batch_size):
            raise ValueError(
                f"channels must have shape {(self.input_count, self.batch_size)}"
            )
        if channels.dtype != self.dtype or channels.device != self.device:
            raise ValueError("channels must be float32 on the circuit device")
        if not channels.is_contiguous():
            raise ValueError("channels must be contiguous [channels, residents]")

    def _host_channels(self, channels: np.ndarray) -> torch.Tensor:
        if not isinstance(channels, np.ndarray):
            raise TypeError("channels must be a NumPy array")
        if channels.dtype != np.float32:
            raise ValueError("channels must use float32")
        if channels.shape != (self.input_count, self.batch_size):
            raise ValueError(
                f"channels must have shape {(self.input_count, self.batch_size)}"
            )
        if not channels.flags.c_contiguous:
            raise ValueError("channels must be contiguous [channels, residents]")
        if not np.isfinite(channels).all() or np.any((channels < 0) | (channels > 1)):
            raise ValueError("channels must be finite values in [0, 1]")
        return torch.from_numpy(channels).to(self.device)

    @torch.no_grad()
    def step_numpy(self, channels: np.ndarray, dt: float) -> FastStepResult:
        device_channels = self._host_channels(channels)
        combined = self.step_device(device_channels, dt, validate=False)
        # Readouts and all three physiology summaries cross the device boundary
        # in one contiguous copy. Feature/physiology results remain views of it.
        host = combined.detach().cpu().numpy()
        self.times += dt
        return FastStepResult(host, self.feature_count, self.times.copy())

    def reset(self) -> None:
        self._reset_tensors()
        self.times.fill(0)

    def metadata(self) -> dict[str, Any]:
        return {
            "engine": "fixed-cohort-csr-v1",
            "layout": self.layout,
            "graph_sha256": self.graph_hash,
            "neurons": self.n,
            "edges": int(self.matrix._nnz()),
            "batch_size": self.batch_size,
            "dtype": str(self.dtype),
            "inputs": self.input_count,
            "readouts": self.feature_count,
            "state_shape": list(self.state_shape),
            "dynamics": {
                "tau": self.tau,
                "gain": self.gain,
                "support_recovery": self.support_recovery,
                "substeps": 2,
            },
            "neural_variant": dict(self._neural_variant),
        }

    def step_device(
        self, channels: torch.Tensor, dt: float, *, validate: bool = True
    ) -> torch.Tensor:
        raise NotImplementedError

    def export_state(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def import_state(self, state: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def _reset_tensors(self) -> None:
        raise NotImplementedError


class NeuronMajorCircuit(_FixedCircuit):
    """Fixed cohort with contiguous state shaped ``[neurons, residents]``."""

    layout = "neuron_major"

    def __init__(self, graph: Any, batch_size: int, **kwargs: Any) -> None:
        super().__init__(graph, batch_size, **kwargs)
        shape = (self.n, self.batch_size)
        self.rates = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.adaptation = torch.zeros_like(self.rates)
        self.support = torch.ones_like(self.rates)
        self.state_shape = shape

    @torch.no_grad()
    def step_device(
        self, channels: torch.Tensor, dt: float, *, validate: bool = True
    ) -> torch.Tensor:
        if validate:
            self._validate_device_channels(channels)
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        drive = torch.sparse.mm(self.input_matrix, channels * self.input_gain)
        alpha = min(1.0, dt / 2 / self.tau)
        for _ in range(2):
            recurrent = self.recurrent_target_gain * torch.sparse.mm(
                self.matrix, self.rates * self.recurrent_source_gain
            )
            target = torch.relu(
                torch.tanh(
                    0.005
                    + self.excitability_gain * (drive + self.gain * recurrent)
                    - 0.10 * self.adaptation
                )
            )
            self.rates.add_(alpha * (target * self.support - self.rates))
        self.adaptation.add_(dt / 5 * (self.rates - self.adaptation))
        self.support.add_(
            dt
            * (
                self.support_recovery * (1 - self.support)
                - 0.003 * self.rates
            )
        ).clamp_(0.65, 1)
        readout = self.readout_gain * torch.sparse.mm(self.readout_matrix, self.rates)
        physiology = torch.stack(
            (
                self.rates.mean(dim=0),
                self.rates.amax(dim=0),
                self.support.mean(dim=0),
            )
        )
        return torch.cat((readout, physiology), dim=0).T.contiguous()

    def export_state(self) -> dict[str, np.ndarray]:
        return {
            "rates": self.rates.T.contiguous().cpu().numpy(),
            "adaptation": self.adaptation.T.contiguous().cpu().numpy(),
            "support": self.support.T.contiguous().cpu().numpy(),
            "times": self.times.copy(),
            "neural_variant_state_identity": self._state_identity_array(),
        }

    def import_state(self, state: dict[str, np.ndarray]) -> None:
        self._require_state_identity(state)
        expected = (self.batch_size, self.n)
        for name in ("rates", "adaptation", "support"):
            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"snapshot {name} must have shape {expected}")
            getattr(self, name).copy_(
                torch.from_numpy(np.ascontiguousarray(value.T)).to(self.device)
            )
        times = np.asarray(state["times"], dtype=np.float64)
        if times.shape != (self.batch_size,) or not np.isfinite(times).all():
            raise ValueError("snapshot times are malformed")
        self.times[:] = times

    def _reset_tensors(self) -> None:
        self.rates.zero_()
        self.adaptation.zero_()
        self.support.fill_(1)


class TritonFusedCircuit(NeuronMajorCircuit):
    """Experimental HIP/Triton CSR recurrence fused with each rate update."""

    layout = "neuron_major_triton_fused"

    def __init__(
        self,
        graph: Any,
        batch_size: int,
        *,
        resident_tile: int = 32,
        **kwargs: Any,
    ) -> None:
        if triton is None:
            raise RuntimeError("Triton is not installed")
        if resident_tile != 32:
            raise ValueError("the gfx1030 experimental kernel requires wave32 tiles")
        super().__init__(graph, batch_size, **kwargs)
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or target.warp_size != 32:
            raise RuntimeError(
                f"experimental kernel needs a HIP wave32 target, received {target}"
            )
        self.resident_tile = resident_tile
        self.rate_buffer = torch.empty_like(self.rates)
        self._row_pointer = self.matrix.crow_indices()
        self._columns = self.matrix.col_indices()
        self._weights = self.matrix.values()

    @torch.no_grad()
    def step_device(
        self, channels: torch.Tensor, dt: float, *, validate: bool = True
    ) -> torch.Tensor:
        if validate:
            self._validate_device_channels(channels)
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        drive = torch.sparse.mm(
            self.input_matrix, channels * self.input_gain
        ).contiguous()
        alpha = min(1.0, dt / 2 / self.tau)
        grid = (self.n, triton.cdiv(self.batch_size, self.resident_tile))
        arguments = (
            self._row_pointer,
            self._columns,
            self._weights,
        )
        _csr_rate_substep[grid](
            *arguments,
            self.rates,
            self.rate_buffer,
            self.recurrent_source_gain,
            self.recurrent_target_gain,
            self.excitability_gain,
            self.adaptation,
            self.support,
            drive,
            self.batch_size,
            alpha,
            self.gain,
            dt,
            self.support_recovery,
            BLOCK_B=self.resident_tile,
            FINAL=False,
            num_warps=1,
        )
        _csr_rate_substep[grid](
            *arguments,
            self.rate_buffer,
            self.rates,
            self.recurrent_source_gain,
            self.recurrent_target_gain,
            self.excitability_gain,
            self.adaptation,
            self.support,
            drive,
            self.batch_size,
            alpha,
            self.gain,
            dt,
            self.support_recovery,
            BLOCK_B=self.resident_tile,
            FINAL=True,
            num_warps=1,
        )
        readout = self.readout_gain * torch.sparse.mm(self.readout_matrix, self.rates)
        physiology = torch.stack(
            (
                self.rates.mean(dim=0),
                self.rates.amax(dim=0),
                self.support.mean(dim=0),
            )
        )
        return torch.cat((readout, physiology), dim=0).T.contiguous()

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update(
            {
                "engine": "fixed-cohort-triton-csr-v1-experimental",
                "resident_tile": self.resident_tile,
                "backend": str(triton.runtime.driver.active.get_current_target()),
            }
        )
        return value


class ResidentMajorReferenceCircuit(_FixedCircuit):
    """Resident-major bit-reference without RemoteBrain JSON/dictionary work."""

    layout = "resident_major_transpose"

    def __init__(self, graph: Any, batch_size: int, **kwargs: Any) -> None:
        super().__init__(graph, batch_size, **kwargs)
        shape = (self.batch_size, self.n)
        self.rates = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.adaptation = torch.zeros_like(self.rates)
        self.support = torch.ones_like(self.rates)
        self.state_shape = shape

    @torch.no_grad()
    def step_device(
        self, channels: torch.Tensor, dt: float, *, validate: bool = True
    ) -> torch.Tensor:
        if validate:
            self._validate_device_channels(channels)
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        drive = torch.sparse.mm(self.input_matrix, channels * self.input_gain).T
        alpha = min(1.0, dt / 2 / self.tau)
        for _ in range(2):
            recurrent = (
                self.recurrent_target_gain
                * torch.sparse.mm(
                    self.matrix, self.rates.T * self.recurrent_source_gain
                )
            ).T
            target = torch.relu(
                torch.tanh(
                    0.005
                    + self.excitability_gain.T * (drive + self.gain * recurrent)
                    - 0.10 * self.adaptation
                )
            )
            self.rates.add_(alpha * (target * self.support - self.rates))
        self.adaptation.add_(dt / 5 * (self.rates - self.adaptation))
        self.support.add_(
            dt
            * (
                self.support_recovery * (1 - self.support)
                - 0.003 * self.rates
            )
        ).clamp_(0.65, 1)
        readout = (
            self.readout_gain * torch.sparse.mm(self.readout_matrix, self.rates.T)
        ).T
        physiology = torch.stack(
            (
                self.rates.mean(dim=1),
                self.rates.amax(dim=1),
                self.support.mean(dim=1),
            ),
            dim=1,
        )
        return torch.cat((readout, physiology), dim=1).contiguous()

    def export_state(self) -> dict[str, np.ndarray]:
        return {
            "rates": self.rates.contiguous().cpu().numpy(),
            "adaptation": self.adaptation.contiguous().cpu().numpy(),
            "support": self.support.contiguous().cpu().numpy(),
            "times": self.times.copy(),
            "neural_variant_state_identity": self._state_identity_array(),
        }

    def import_state(self, state: dict[str, np.ndarray]) -> None:
        self._require_state_identity(state)
        expected = (self.batch_size, self.n)
        for name in ("rates", "adaptation", "support"):
            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"snapshot {name} must have shape {expected}")
            getattr(self, name).copy_(torch.from_numpy(value).to(self.device))
        times = np.asarray(state["times"], dtype=np.float64)
        if times.shape != (self.batch_size,) or not np.isfinite(times).all():
            raise ValueError("snapshot times are malformed")
        self.times[:] = times

    def _reset_tensors(self) -> None:
        self.rates.zero_()
        self.adaptation.zero_()
        self.support.fill_(1)


class MicrobatchedResidentCircuit(ResidentMajorReferenceCircuit):
    """Resident-major state evaluated in exact independent resident chunks."""

    layout = "resident_major_microbatched"

    def __init__(
        self, graph: Any, batch_size: int, *, microbatch_size: int = 3, **kwargs: Any
    ) -> None:
        if microbatch_size < 1 or microbatch_size > batch_size:
            raise ValueError("microbatch_size must be in 1..batch_size")
        self.microbatch_size = int(microbatch_size)
        super().__init__(graph, batch_size, **kwargs)

    @torch.no_grad()
    def step_device(
        self, channels: torch.Tensor, dt: float, *, validate: bool = True
    ) -> torch.Tensor:
        if validate:
            self._validate_device_channels(channels)
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        alpha = min(1.0, dt / 2 / self.tau)
        combined = torch.empty(
            (self.batch_size, self.feature_count + len(PHYSIOLOGY_NAMES)),
            dtype=self.dtype,
            device=self.device,
        )
        for start in range(0, self.batch_size, self.microbatch_size):
            stop = min(start + self.microbatch_size, self.batch_size)
            rates = self.rates[start:stop]
            adaptation = self.adaptation[start:stop]
            support = self.support[start:stop]
            chunk_channels = (
                channels[:, start:stop] * self.input_gain[:, start:stop]
            ).contiguous()
            drive = torch.sparse.mm(self.input_matrix, chunk_channels).T
            for _ in range(2):
                recurrent = (
                    self.recurrent_target_gain[:, start:stop]
                    * torch.sparse.mm(
                        self.matrix,
                        rates.T * self.recurrent_source_gain[:, start:stop],
                    )
                ).T
                target = torch.relu(
                    torch.tanh(
                        0.005
                        + self.excitability_gain[:, start:stop].T
                        * (drive + self.gain * recurrent)
                        - 0.10 * adaptation
                    )
                )
                rates.add_(alpha * (target * support - rates))
            adaptation.add_(dt / 5 * (rates - adaptation))
            support.add_(
                dt
                * (
                    self.support_recovery * (1 - support)
                    - 0.003 * rates
                )
            ).clamp_(0.65, 1)
            readout = (
                self.readout_gain[:, start:stop]
                * torch.sparse.mm(self.readout_matrix, rates.T)
            ).T
            combined[start:stop, : self.feature_count] = readout
            combined[start:stop, self.feature_count :] = torch.stack(
                (
                    rates.mean(dim=1),
                    rates.amax(dim=1),
                    support.mean(dim=1),
                ),
                dim=1,
            )
        return combined

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value["microbatch_size"] = self.microbatch_size
        return value
