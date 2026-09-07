"""Trainable sensory afferents and differentiable full-MaleCNS recurrence.

Raw optic and body-local sensory values enter only through the afferent modules.
Every exported latent is read from the recurrent 165,122-neuron graph after
masking all directly injected neurons.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from chreatures.cns_adapter_contract import (
    ARRAY_SPECS as SERVICE_ARRAY_SPECS,
    FORMAT as SERVICE_FORMAT,
    PARAMETER_ORDER,
)

CNS_NEURONS = 165_122
CNS_EDGES = 25_563_197
OPTIC_SITES = 1_771
OPTIC_CHANNELS = 3
PHOTORECEPTORS = 4_107
PHOTORECEPTOR_TYPES = 10
BODY_CHANNELS = 43
BODY_AFFERENTS = 11_233
CNS_TYPES = 11_752
LATENT_DIM = 512
DT_SECONDS = 0.05
RECURRENT_GAIN = 0.92
ADAPTATION_RATE = 1.0 / 5.0
ADAPTATION_COEFFICIENT = 0.10
SUPPORT_RECOVERY = 0.024
SUPPORT_COST = 0.003
SUPPORT_MINIMUM = 0.65
NORMALIZED_BODY_CLIP = 8.0
CHECKPOINT_TICKS = 4
ARTIFACT_FORMAT = "chreatures-trainable-cns-adapter-v1"
PARAMETER_ARTIFACT_FORMAT = "chreatures-cns-adapter-parameters-v1"
RAW_SENSORY_DIM = OPTIC_SITES * OPTIC_CHANNELS + BODY_CHANNELS


TRAINABLE_SHAPES = {
    "optic.spectral_logits": (PHOTORECEPTOR_TYPES, OPTIC_CHANNELS),
    "optic.gain_raw": (PHOTORECEPTOR_TYPES,),
    "optic.bias": (PHOTORECEPTOR_TYPES,),
    "body.input.weight": (128, BODY_CHANNELS),
    "body.input.bias": (128,),
    "body.output.weight": (BODY_AFFERENTS, 128),
    "body.output.bias": (BODY_AFFERENTS,),
    "dynamics.bias_raw": (CNS_TYPES,),
    "dynamics.tau_raw": (CNS_TYPES,),
    "dynamics.source_raw": (CNS_TYPES,),
    "dynamics.target_raw": (CNS_TYPES,),
    "dynamics.excitability_raw": (CNS_TYPES,),
    "readout.weight": (LATENT_DIM, CNS_NEURONS),
    "readout.bias": (LATENT_DIM,),
}
BUFFER_SHAPES = {
    "body.mean": (BODY_CHANNELS,),
    "body.scale": (BODY_CHANNELS,),
}
EXPORT_ORDER = PARAMETER_ORDER
if set(EXPORT_ORDER) != {*TRAINABLE_SHAPES, *BUFFER_SHAPES}:
    raise RuntimeError("Torch and native CNS adapter parameter contracts differ")
if {
    name: shape for name, _, shape in SERVICE_ARRAY_SPECS if name in EXPORT_ORDER
} != {**TRAINABLE_SHAPES, **BUFFER_SHAPES}:
    raise RuntimeError("Torch and native CNS adapter tensor shapes differ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


@dataclass(frozen=True)
class CNSState:
    rates: torch.Tensor
    adaptation: torch.Tensor
    support: torch.Tensor

    def detach(self) -> "CNSState":
        return CNSState(
            self.rates.detach(), self.adaptation.detach(), self.support.detach()
        )


@dataclass(frozen=True)
class CNSStaticArrays:
    recurrent_crow: np.ndarray
    recurrent_columns: np.ndarray
    recurrent_values: np.ndarray
    recurrent_transpose_crow: np.ndarray
    recurrent_transpose_columns: np.ndarray
    recurrent_transpose_values: np.ndarray
    optic_crow: np.ndarray
    optic_columns: np.ndarray
    optic_values: np.ndarray
    optic_graph_rows: np.ndarray
    optic_type_index: np.ndarray
    optic_supported: np.ndarray
    body_graph_rows: np.ndarray
    neuron_type_index: np.ndarray
    readout_mask: np.ndarray
    identity: dict[str, Any]

    @classmethod
    def load(cls, graph: Path, atlas: Path) -> "CNSStaticArrays":
        """Authenticate and prepare fixed sparse arrays without dense NxN data."""

        from scipy import sparse

        graph = graph.expanduser().resolve()
        atlas = atlas.expanduser().resolve()
        manifest_path = graph / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        counts_meta = manifest.get("counts", {})
        if (
            int(counts_meta.get("neurons", -1)) != CNS_NEURONS
            or int(counts_meta.get("edges", -1)) != CNS_EDGES
        ):
            raise ValueError("MaleCNS graph scope differs")
        for name, receipt in manifest.get("artifacts", {}).items():
            path = graph / name
            if (
                not path.is_file()
                or path.stat().st_size != int(receipt.get("bytes", -1))
                or file_sha256(path) != receipt.get("sha256")
            ):
                raise ValueError(f"MaleCNS graph receipt differs: {name}")

        indptr = np.load(graph / "indptr.npy", mmap_mode="r")
        columns = np.load(graph / "indices.npy", mmap_mode="r")
        counts = np.load(graph / "counts.npy", mmap_mode="r")
        row_synapses = np.load(graph / "row_synapses.npy", mmap_mode="r")
        if (
            indptr.dtype != np.int64
            or indptr.shape != (CNS_NEURONS + 1,)
            or columns.dtype != np.int32
            or columns.shape != (CNS_EDGES,)
            or counts.dtype != np.uint32
            or counts.shape != (CNS_EDGES,)
            or row_synapses.dtype != np.uint64
            or row_synapses.shape != (CNS_NEURONS,)
        ):
            raise ValueError("MaleCNS derived array layout differs")
        with np.load(graph / "neurons.npz", allow_pickle=False) as neurons:
            sign = np.ascontiguousarray(neurons["sign"], dtype=np.float32)
            superclasses = np.asarray(neurons["superclasses"])
            graph_body_ids = np.ascontiguousarray(neurons["body_ids"])
        values = np.empty(CNS_EDGES, dtype=np.float32)
        for start in range(0, CNS_NEURONS, 4096):
            stop = min(start + 4096, CNS_NEURONS)
            edge_start = int(indptr[start])
            edge_stop = int(indptr[stop])
            denominators = np.maximum(row_synapses[start:stop], 1).astype(np.float32)
            values[edge_start:edge_stop] = (
                counts[edge_start:edge_stop].astype(np.float32)
                / np.repeat(denominators, np.diff(indptr[start : stop + 1]))
            )
        values *= sign[np.asarray(columns)]
        recurrent = sparse.csr_matrix(
            (values, np.asarray(columns), np.asarray(indptr)),
            shape=(CNS_NEURONS, CNS_NEURONS),
        )
        recurrent.has_sorted_indices = True
        recurrent.has_canonical_format = True
        recurrent_t = recurrent.transpose().tocsr()

        with np.load(atlas, allow_pickle=False) as value:
            required = {
                "body_ids",
                "type_names",
                "type_index",
                "site_side_hex",
                "photoreceptor_graph_rows",
                "photoreceptor_site_indptr",
                "photoreceptor_site_indices",
                "photoreceptor_site_synapse_counts",
            }
            if not required.issubset(value.files):
                raise ValueError("optic atlas arrays are incomplete")
            atlas_body_ids = np.ascontiguousarray(value["body_ids"])
            type_names = np.asarray(value["type_names"])
            neuron_type = np.ascontiguousarray(value["type_index"], dtype=np.int64)
            site_side_hex = np.ascontiguousarray(value["site_side_hex"])
            optic_rows = np.ascontiguousarray(
                value["photoreceptor_graph_rows"], dtype=np.int64
            )
            optic_crow = np.ascontiguousarray(
                value["photoreceptor_site_indptr"], dtype=np.int64
            )
            optic_columns = np.ascontiguousarray(
                value["photoreceptor_site_indices"], dtype=np.int64
            )
            optic_counts = np.ascontiguousarray(
                value["photoreceptor_site_synapse_counts"], dtype=np.float32
            )
        if (
            not np.array_equal(graph_body_ids, atlas_body_ids)
            or type_names.shape != (CNS_TYPES,)
            or neuron_type.shape != (CNS_NEURONS,)
            or site_side_hex.shape != (OPTIC_SITES, 3)
            or optic_rows.shape != (PHOTORECEPTORS,)
            or optic_crow.shape != (PHOTORECEPTORS + 1,)
            or len(np.unique(optic_rows)) != PHOTORECEPTORS
        ):
            raise ValueError("optic atlas differs from the full graph")
        global_optic_types = np.unique(neuron_type[optic_rows])
        if global_optic_types.shape != (PHOTORECEPTOR_TYPES,):
            raise ValueError("photoreceptor type count differs")
        compact_type = np.searchsorted(global_optic_types, neuron_type[optic_rows])
        optic_supported = np.diff(optic_crow) > 0
        optic_values = np.empty_like(optic_counts)
        for row in range(PHOTORECEPTORS):
            start, stop = int(optic_crow[row]), int(optic_crow[row + 1])
            if stop > start:
                optic_values[start:stop] = optic_counts[start:stop] / optic_counts[
                    start:stop
                ].sum()

        body_rows = np.flatnonzero(
            np.isin(superclasses, ("cb_sensory", "vnc_sensory"))
        ).astype(np.int64)
        if body_rows.shape != (BODY_AFFERENTS,) or np.intersect1d(
            body_rows, optic_rows
        ).size:
            raise ValueError("body afferent atlas differs or overlaps photoreceptors")
        mask = np.ones(CNS_NEURONS, dtype=np.float32)
        mask[optic_rows] = 0.0
        mask[body_rows] = 0.0
        identity = {
            "format": "chreatures-cns-static-training-arrays-v1",
            "graph_dataset_sha256": manifest.get("dataset_hash"),
            "graph_manifest_sha256": file_sha256(manifest_path),
            "atlas_file_sha256": file_sha256(atlas),
            "neurons": CNS_NEURONS,
            "edges": CNS_EDGES,
            "optic_sites": OPTIC_SITES,
            "photoreceptors": PHOTORECEPTORS,
            "body_afferents": BODY_AFFERENTS,
            "readout_unmasked": int(mask.sum()),
            "photoreceptor_type_names": [
                str(type_names[index]) for index in global_optic_types
            ],
            "body_superclasses": ["cb_sensory", "vnc_sensory"],
        }
        identity["content_sha256"] = canonical_sha256(identity)
        return cls(
            np.ascontiguousarray(recurrent.indptr, dtype=np.int32),
            np.ascontiguousarray(recurrent.indices, dtype=np.int32),
            np.ascontiguousarray(recurrent.data, dtype=np.float32),
            np.ascontiguousarray(recurrent_t.indptr, dtype=np.int32),
            np.ascontiguousarray(recurrent_t.indices, dtype=np.int32),
            np.ascontiguousarray(recurrent_t.data, dtype=np.float32),
            optic_crow.astype(np.int32),
            optic_columns.astype(np.int32),
            optic_values,
            optic_rows,
            compact_type.astype(np.int64),
            optic_supported.astype(np.float32),
            body_rows,
            neuron_type,
            mask,
            identity,
        )


def _csr(crow, columns, values, shape, device) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.as_tensor(np.array(crow, copy=True), device=device),
        torch.as_tensor(np.array(columns, copy=True), device=device),
        torch.as_tensor(np.array(values, copy=True), device=device),
        size=shape,
        dtype=torch.float32,
        device=device,
    )


class _FixedSparseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, transpose, dense):
        ctx.transpose = transpose
        return torch.sparse.mm(matrix, dense)

    @staticmethod
    def backward(ctx, gradient):
        return None, None, torch.sparse.mm(ctx.transpose, gradient)


def fixed_sparse_mm(
    matrix: torch.Tensor, transpose: torch.Tensor, dense: torch.Tensor
) -> torch.Tensor:
    return _FixedSparseMM.apply(matrix, transpose, dense)


class TrainableCNSAdapter(nn.Module):
    """Afferents, type dynamics, full fixed graph, and masked latent readout."""

    def __init__(self, static: CNSStaticArrays, *, device: torch.device) -> None:
        super().__init__()
        self.static_identity = static.identity
        self.optic_spectral_logits = nn.Parameter(
            torch.zeros(PHOTORECEPTOR_TYPES, OPTIC_CHANNELS, device=device)
        )
        self.optic_gain_raw = nn.Parameter(
            torch.full(
                (PHOTORECEPTOR_TYPES,), inverse_softplus(1.0), device=device
            )
        )
        self.optic_bias = nn.Parameter(torch.zeros(PHOTORECEPTOR_TYPES, device=device))
        self.body_input = nn.Linear(BODY_CHANNELS, 128, device=device)
        self.body_output = nn.Linear(128, BODY_AFFERENTS, device=device)
        self.register_buffer(
            "body_mean", torch.zeros(BODY_CHANNELS, device=device), persistent=True
        )
        self.register_buffer(
            "body_scale", torch.ones(BODY_CHANNELS, device=device), persistent=True
        )

        initial_bias = math.atanh(0.005 / 0.5)
        initial_tau = inverse_sigmoid((0.16 - 0.025) / 0.475)
        self.dynamics_bias_raw = nn.Parameter(
            torch.full((CNS_TYPES,), initial_bias, device=device)
        )
        self.dynamics_tau_raw = nn.Parameter(
            torch.full((CNS_TYPES,), initial_tau, device=device)
        )
        self.dynamics_source_raw = nn.Parameter(torch.zeros(CNS_TYPES, device=device))
        self.dynamics_target_raw = nn.Parameter(torch.zeros(CNS_TYPES, device=device))
        self.dynamics_excitability_raw = nn.Parameter(
            torch.zeros(CNS_TYPES, device=device)
        )
        self.readout = nn.Linear(CNS_NEURONS, LATENT_DIM, device=device)

        self.register_buffer(
            "recurrent",
            _csr(
                static.recurrent_crow,
                static.recurrent_columns,
                static.recurrent_values,
                (CNS_NEURONS, CNS_NEURONS),
                device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "recurrent_transpose",
            _csr(
                static.recurrent_transpose_crow,
                static.recurrent_transpose_columns,
                static.recurrent_transpose_values,
                (CNS_NEURONS, CNS_NEURONS),
                device,
            ),
            persistent=False,
        )
        optic_matrix = _csr(
            static.optic_crow,
            static.optic_columns,
            static.optic_values,
            (PHOTORECEPTORS, OPTIC_SITES),
            device,
        )
        self.register_buffer("optic_matrix", optic_matrix, persistent=False)
        self.register_buffer(
            "optic_graph_rows",
            torch.as_tensor(static.optic_graph_rows, device=device),
            persistent=False,
        )
        self.register_buffer(
            "optic_type_index",
            torch.as_tensor(static.optic_type_index, device=device),
            persistent=False,
        )
        self.register_buffer(
            "optic_supported",
            torch.as_tensor(static.optic_supported, device=device),
            persistent=False,
        )
        self.register_buffer(
            "body_graph_rows",
            torch.as_tensor(static.body_graph_rows, device=device),
            persistent=False,
        )
        self.register_buffer(
            "neuron_type_index",
            torch.as_tensor(static.neuron_type_index, device=device),
            persistent=False,
        )
        self.register_buffer(
            "readout_mask",
            torch.as_tensor(static.readout_mask, device=device),
            persistent=False,
        )
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.optic_bias.device

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.xavier_uniform_(self.body_input.weight)
            self.body_input.bias.zero_()
            nn.init.xavier_uniform_(self.body_output.weight)
            self.body_output.bias.fill_(math.log(0.05 / 0.95))
            nn.init.xavier_uniform_(self.readout.weight)
            self.readout.weight.mul_(self.readout_mask.unsqueeze(0))
            self.readout.bias.zero_()

    def set_body_moments(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if (
            mean.shape != (BODY_CHANNELS,)
            or scale.shape != (BODY_CHANNELS,)
            or not torch.all(torch.isfinite(mean))
            or not torch.all(torch.isfinite(scale))
            or torch.any(scale <= 0)
        ):
            raise ValueError("body moments must be finite 43-vectors with positive scale")
        with torch.no_grad():
            self.body_mean.copy_(mean)
            self.body_scale.copy_(scale)

    def initial_state(self, batch_size: int) -> CNSState:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        shape = (CNS_NEURONS, batch_size)
        return CNSState(
            torch.zeros(shape, device=self.device),
            torch.zeros(shape, device=self.device),
            torch.ones(shape, device=self.device),
        )

    def afferent_drive(self, optic_rgb: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        if (
            optic_rgb.ndim != 3
            or optic_rgb.shape[1:] != (OPTIC_SITES, OPTIC_CHANNELS)
            or body.shape != (optic_rgb.shape[0], BODY_CHANNELS)
            or optic_rgb.dtype != torch.float32
            or body.dtype != torch.float32
            or optic_rgb.device != self.device
            or body.device != self.device
            or not torch.all(torch.isfinite(optic_rgb))
            or not torch.all(torch.isfinite(body))
            or torch.any((optic_rgb < 0) | (optic_rgb > 1))
        ):
            raise ValueError("CNS afferents require optic[B,1771,3] and body[B,43]")
        batch = optic_rgb.shape[0]
        site_rgb = optic_rgb.permute(1, 0, 2).reshape(OPTIC_SITES, batch * 3)
        receptor_rgb = torch.sparse.mm(self.optic_matrix, site_rgb).reshape(
            PHOTORECEPTORS, batch, 3
        ).permute(1, 0, 2)
        spectral = torch.softmax(self.optic_spectral_logits, dim=-1)[
            self.optic_type_index
        ]
        mixture = (receptor_rgb * spectral.unsqueeze(0)).sum(dim=-1)
        gain = F.softplus(self.optic_gain_raw)[self.optic_type_index]
        bias = self.optic_bias[self.optic_type_index]
        optic_current = torch.sigmoid(
            bias.unsqueeze(0) + gain.unsqueeze(0) * (2.0 * mixture - 1.0)
        ) * self.optic_supported.unsqueeze(0)

        standardized = ((body - self.body_mean) / self.body_scale).clamp(
            -NORMALIZED_BODY_CLIP, NORMALIZED_BODY_CLIP
        )
        body_current = torch.sigmoid(
            self.body_output(torch.tanh(self.body_input(standardized)))
        )
        drive = optic_rgb.new_zeros((batch, CNS_NEURONS))
        drive = torch.index_copy(drive, 1, self.optic_graph_rows, optic_current)
        drive = torch.index_copy(drive, 1, self.body_graph_rows, body_current)
        return drive.T.contiguous()

    def effective_dynamics(self) -> tuple[torch.Tensor, ...]:
        type_index = self.neuron_type_index
        bias = (0.5 * torch.tanh(self.dynamics_bias_raw))[type_index]
        tau = (0.025 + 0.475 * torch.sigmoid(self.dynamics_tau_raw))[type_index]
        source = (0.5 + torch.sigmoid(self.dynamics_source_raw))[type_index]
        target = (0.5 + torch.sigmoid(self.dynamics_target_raw))[type_index]
        excitability = (0.5 + torch.sigmoid(self.dynamics_excitability_raw))[type_index]
        return bias[:, None], tau[:, None], source[:, None], target[:, None], excitability[:, None]

    def step(
        self, optic_rgb: torch.Tensor, body: torch.Tensor, state: CNSState
    ) -> tuple[torch.Tensor, CNSState]:
        batch = optic_rgb.shape[0]
        expected = (CNS_NEURONS, batch)
        if any(
            value.shape != expected or value.dtype != torch.float32 or value.device != self.device
            for value in (state.rates, state.adaptation, state.support)
        ):
            raise ValueError("CNS recurrent state shape, dtype, or device differs")
        drive = self.afferent_drive(optic_rgb, body)
        bias, tau, source, target_gain, excitability = self.effective_dynamics()
        alpha = (DT_SECONDS / (2.0 * tau)).clamp_max(1.0)
        rates = state.rates
        for _ in range(2):
            recurrent = target_gain * fixed_sparse_mm(
                self.recurrent, self.recurrent_transpose, source * rates
            )
            target_rate = torch.relu(
                torch.tanh(
                    bias
                    + excitability * (drive + RECURRENT_GAIN * recurrent)
                    - ADAPTATION_COEFFICIENT * state.adaptation
                )
            )
            rates = rates + alpha * (target_rate * state.support - rates)
        adaptation = state.adaptation + DT_SECONDS * ADAPTATION_RATE * (
            rates - state.adaptation
        )
        support = torch.clamp(
            state.support
            + DT_SECONDS
            * (
                SUPPORT_RECOVERY * (1.0 - state.support)
                - SUPPORT_COST * rates
            ),
            SUPPORT_MINIMUM,
            1.0,
        )
        latent = torch.tanh(self.readout((rates.T * self.readout_mask)))
        return latent, CNSState(rates, adaptation, support)

    def _group(
        self,
        optic: torch.Tensor,
        body: torch.Tensor,
        reset: torch.Tensor,
        rates: torch.Tensor,
        adaptation: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = []
        state = CNSState(rates, adaptation, support)
        for index in range(optic.shape[0]):
            keep = (~reset[index]).to(rates.dtype).unsqueeze(0)
            state = CNSState(
                state.rates * keep,
                state.adaptation * keep,
                state.support * keep + (1.0 - keep),
            )
            latent, state = self.step(optic[index], body[index], state)
            values.append(latent)
        return torch.stack(values), state.rates, state.adaptation, state.support

    def forward_sequence(
        self,
        optic: torch.Tensor,
        body: torch.Tensor,
        reset: torch.Tensor,
        state: CNSState | None = None,
        *,
        checkpoint_ticks: int = CHECKPOINT_TICKS,
    ) -> tuple[torch.Tensor, CNSState]:
        if (
            optic.ndim != 4
            or optic.shape[2:] != (OPTIC_SITES, OPTIC_CHANNELS)
            or body.shape != (*optic.shape[:2], BODY_CHANNELS)
            or reset.shape != optic.shape[:2]
            or reset.dtype != torch.bool
            or not 1 <= checkpoint_ticks <= optic.shape[0]
        ):
            raise ValueError("CNS sequence must be optic[T,B,1771,3], body[T,B,43]")
        state = self.initial_state(optic.shape[1]) if state is None else state
        values = []
        for start in range(0, optic.shape[0], checkpoint_ticks):
            stop = min(start + checkpoint_ticks, optic.shape[0])
            arguments = (
                optic[start:stop],
                body[start:stop],
                reset[start:stop],
                state.rates,
                state.adaptation,
                state.support,
            )
            if self.training and torch.is_grad_enabled():
                result = checkpoint(self._group, *arguments, use_reentrant=False)
            else:
                result = self._group(*arguments)
            latent, rates, adaptation, support = result
            values.append(latent)
            state = CNSState(rates, adaptation, support)
        return torch.cat(values, dim=0), state


def export_arrays(model: TrainableCNSAdapter) -> dict[str, np.ndarray]:
    mapping = {
        "optic.spectral_logits": model.optic_spectral_logits,
        "optic.gain_raw": model.optic_gain_raw,
        "optic.bias": model.optic_bias,
        "body.input.weight": model.body_input.weight,
        "body.input.bias": model.body_input.bias,
        "body.output.weight": model.body_output.weight,
        "body.output.bias": model.body_output.bias,
        "dynamics.bias_raw": model.dynamics_bias_raw,
        "dynamics.tau_raw": model.dynamics_tau_raw,
        "dynamics.source_raw": model.dynamics_source_raw,
        "dynamics.target_raw": model.dynamics_target_raw,
        "dynamics.excitability_raw": model.dynamics_excitability_raw,
        "readout.weight": model.readout.weight,
        "readout.bias": model.readout.bias,
        "body.mean": model.body_mean,
        "body.scale": model.body_scale,
    }
    result = {
        name: np.ascontiguousarray(mapping[name].detach().cpu().numpy(), dtype=np.float32)
        for name in EXPORT_ORDER
    }
    for name, shape in TRAINABLE_SHAPES.items():
        if result[name].shape != shape:
            raise RuntimeError(f"CNS adapter tensor shape differs: {name}")
    for name, shape in BUFFER_SHAPES.items():
        if result[name].shape != shape:
            raise RuntimeError(f"CNS adapter buffer shape differs: {name}")
    direct = model.readout_mask.detach().cpu().numpy() == 0
    if np.any(result["readout.weight"][:, direct] != 0):
        raise RuntimeError("CNS readout contains nonzero direct-afferent columns")
    return result


def load_export_arrays(
    model: TrainableCNSAdapter, arrays: Mapping[str, np.ndarray]
) -> None:
    if set(arrays) != set(EXPORT_ORDER):
        raise ValueError("CNS adapter parameter set differs")
    tensor = {}
    for name, shape in {**TRAINABLE_SHAPES, **BUFFER_SHAPES}.items():
        value = np.asarray(arrays[name])
        if (
            value.dtype != np.float32
            or value.shape != shape
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"CNS adapter parameter differs: {name}")
        tensor[name] = torch.as_tensor(value.copy(), device=model.device)
    with torch.no_grad():
        model.optic_spectral_logits.copy_(tensor["optic.spectral_logits"])
        model.optic_gain_raw.copy_(tensor["optic.gain_raw"])
        model.optic_bias.copy_(tensor["optic.bias"])
        model.body_mean.copy_(tensor["body.mean"])
        model.body_scale.copy_(tensor["body.scale"])
        model.body_input.weight.copy_(tensor["body.input.weight"])
        model.body_input.bias.copy_(tensor["body.input.bias"])
        model.body_output.weight.copy_(tensor["body.output.weight"])
        model.body_output.bias.copy_(tensor["body.output.bias"])
        model.dynamics_bias_raw.copy_(tensor["dynamics.bias_raw"])
        model.dynamics_tau_raw.copy_(tensor["dynamics.tau_raw"])
        model.dynamics_source_raw.copy_(tensor["dynamics.source_raw"])
        model.dynamics_target_raw.copy_(tensor["dynamics.target_raw"])
        model.dynamics_excitability_raw.copy_(tensor["dynamics.excitability_raw"])
        model.readout.weight.copy_(tensor["readout.weight"])
        model.readout.bias.copy_(tensor["readout.bias"])


def parameter_artifact_identity(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    clean = copy.deepcopy(dict(metadata))
    clean.pop("artifact_sha256", None)
    receipts = {
        name: {
            "shape": list(arrays[name].shape),
            "dtype": arrays[name].dtype.str,
            "sha256": hashlib.sha256(arrays[name].tobytes()).hexdigest(),
        }
        for name in EXPORT_ORDER
    }
    return canonical_sha256({"metadata": clean, "arrays": receipts})


def write_parameter_artifact(
    path: Path,
    model: TrainableCNSAdapter,
    *,
    training_status: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if training_status not in {"initialized-untrained", "trained"}:
        raise ValueError("CNS adapter training status differs")
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(path)
    arrays = export_arrays(model)
    metadata = {
        "format": PARAMETER_ARTIFACT_FORMAT,
        "service_format": SERVICE_FORMAT,
        "training_status": training_status,
        "parameter_order": list(EXPORT_ORDER),
        "static_identity": copy.deepcopy(model.static_identity),
        "dynamics": {
            "dt_seconds": DT_SECONDS,
            "substeps": 2,
            "recurrent_gain": RECURRENT_GAIN,
            "adaptation_rate": ADAPTATION_RATE,
            "adaptation_coefficient": ADAPTATION_COEFFICIENT,
            "support_recovery": SUPPORT_RECOVERY,
            "support_cost": SUPPORT_COST,
            "support_minimum": SUPPORT_MINIMUM,
            "type_parameterization": {
                "bias": "0.5*tanh(raw)",
                "tau": "0.025+0.475*sigmoid(raw)",
                "source_target_excitability": "0.5+sigmoid(raw)",
            },
        },
        "afferents": {
            "raw_input_shape": [OPTIC_SITES, OPTIC_CHANNELS],
            "body_input_count": BODY_CHANNELS,
            "normalized_body_clip": NORMALIZED_BODY_CLIP,
            "optic": "normalized measured site mixture then learned type spectral/gain/tonic",
            "unsupported_receptors": "zero drive including tonic",
        },
        "readout": {
            "shape": [LATENT_DIM, CNS_NEURONS],
            "activation": "tanh",
            "direct_afferents_masked": PHOTORECEPTORS + BODY_AFFERENTS,
        },
        "provenance": copy.deepcopy(dict(provenance)),
    }
    metadata["artifact_sha256"] = parameter_artifact_identity(metadata, arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            np.savez_compressed(
                handle,
                metadata=np.asarray(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "artifact_sha256": metadata["artifact_sha256"],
        "training_status": training_status,
    }


class SensoryPredictionHeads(nn.Module):
    """Training-only decoders; none of these tensors enter the CNS service."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.current = nn.Linear(LATENT_DIM, RAW_SENSORY_DIM, device=device)
        self.transition = nn.GRUCell(LATENT_DIM + 12, LATENT_DIM, device=device)
        self.future = nn.Linear(LATENT_DIM, RAW_SENSORY_DIM, device=device)

    @staticmethod
    def decode(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        optic, body = value.split((OPTIC_SITES * OPTIC_CHANNELS, BODY_CHANNELS), dim=-1)
        return torch.sigmoid(optic), body

    def forward(
        self,
        latent: torch.Tensor,
        delivered_action: torch.Tensor,
        reset: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        if (
            latent.ndim != 3
            or latent.shape[-1] != LATENT_DIM
            or delivered_action.shape != (*latent.shape[:2], 12)
            or reset.shape != latent.shape[:2]
            or reset.dtype != torch.bool
        ):
            raise ValueError("sensory prediction sequence shapes differ")
        current = self.decode(self.current(latent))
        hidden = torch.zeros_like(latent[0])
        predictions = []
        for index in range(latent.shape[0]):
            hidden = hidden * (~reset[index]).to(hidden.dtype).unsqueeze(-1)
            hidden = self.transition(
                torch.cat((latent[index], delivered_action[index]), dim=-1), hidden
            )
            predictions.append(self.future(hidden))
        future = self.decode(torch.stack(predictions))
        return current, future


def sensory_prediction_loss(
    latent: torch.Tensor,
    heads: SensoryPredictionHeads,
    clean_optic: torch.Tensor,
    clean_body: torch.Tensor,
    delivered_action: torch.Tensor,
    reset: torch.Tensor,
    next_valid: torch.Tensor,
    body_mean: torch.Tensor,
    body_scale: torch.Tensor,
    *,
    current_coefficient: float = 0.5,
    future_coefficient: float = 1.0,
    body_coefficient: float = 0.1,
    variance_coefficient: float = 0.02,
    covariance_coefficient: float = 0.002,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Predict only clean raw senses; raw tensors never bypass the CNS into control."""

    if (
        clean_optic.shape != (*latent.shape[:2], OPTIC_SITES, OPTIC_CHANNELS)
        or clean_body.shape != (*latent.shape[:2], BODY_CHANNELS)
        or next_valid.shape != latent.shape[:2]
        or next_valid.dtype != torch.bool
    ):
        raise ValueError("clean sensory targets differ from latent sequence")
    current, future = heads(latent, delivered_action, reset)
    optic_target = clean_optic.flatten(2)
    body_target = ((clean_body - body_mean) / body_scale).clamp(
        -NORMALIZED_BODY_CLIP, NORMALIZED_BODY_CLIP
    )
    current_optic = F.mse_loss(current[0], optic_target)
    current_body = F.smooth_l1_loss(current[1], body_target)
    if not torch.any(next_valid):
        raise ValueError("sequence contains no valid next-sensory target")
    # ``future[t]`` predicts the acknowledged raw senses at t+1.  Explicit
    # indexing prevents reset or clip seams from becoming training targets.
    predict_mask = next_valid[:-1]
    if not torch.any(predict_mask):
        raise ValueError("sequence contains no within-clip future target")
    future_optic = F.mse_loss(future[0][:-1][predict_mask], optic_target[1:][predict_mask])
    future_body = F.smooth_l1_loss(
        future[1][:-1][predict_mask], body_target[1:][predict_mask]
    )

    samples = latent.reshape(-1, LATENT_DIM)
    centered = samples - samples.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.square().mean(dim=0) + 1e-4)
    variance = torch.relu(1.0 - standard_deviation).mean()
    covariance = centered.T @ centered / max(samples.shape[0] - 1, 1)
    covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = covariance.square().sum() / LATENT_DIM
    reconstruction = current_optic + body_coefficient * current_body
    prediction = future_optic + body_coefficient * future_body
    total = (
        current_coefficient * reconstruction
        + future_coefficient * prediction
        + variance_coefficient * variance
        + covariance_coefficient * covariance_loss
    )
    return total, {
        "loss": total.detach(),
        "current_optic_mse": current_optic.detach(),
        "current_body_huber": current_body.detach(),
        "future_optic_mse": future_optic.detach(),
        "future_body_huber": future_body.detach(),
        "latent_variance_penalty": variance.detach(),
        "latent_covariance_penalty": covariance_loss.detach(),
        "latent_std_mean": standard_deviation.mean().detach(),
    }
