"""Persistent batched sparse neural state for the full curated MaleCNS graph."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy import sparse


SNAPSHOT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


def _torch_csr(matrix: sparse.spmatrix, device: torch.device) -> torch.Tensor:
    csr = matrix.tocsr().astype(np.float32, copy=False)
    if csr.nnz <= np.iinfo(np.int32).max:
        crow = torch.as_tensor(csr.indptr.astype(np.int32, copy=False), device=device)
        columns = torch.as_tensor(csr.indices.astype(np.int32, copy=False), device=device)
    else:
        crow = torch.as_tensor(csr.indptr.astype(np.int64, copy=False), device=device)
        columns = torch.as_tensor(csr.indices.astype(np.int64, copy=False), device=device)
    values = torch.as_tensor(csr.data, dtype=torch.float32, device=device)
    return torch.sparse_csr_tensor(
        crow, columns, values, size=csr.shape, dtype=torch.float32, device=device
    )


def _mapping_pair(graph: Any, name: str) -> tuple[list[str], sparse.csr_matrix]:
    value = getattr(graph, name)
    value = value() if callable(value) else value
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must return (names, scipy CSR matrix)")
    names, matrix = value
    names = [str(item) for item in names]
    if len(set(names)) != len(names) or any(not item for item in names):
        raise ValueError(f"{name} contains invalid or duplicate names")
    if not sparse.issparse(matrix):
        raise TypeError(f"{name} must remain sparse")
    return names, matrix.tocsr().astype(np.float32, copy=False)


class RemoteBrain:
    """Full sparse graph and fixed-capacity private resident state on one device."""

    def __init__(
        self,
        graph: Any,
        *,
        capacity: int = 16,
        device: str | torch.device = "cuda",
        tau: float = 0.16,
        gain: float = 0.92,
        support_recovery: float = 0.024,
        input_map: tuple[Sequence[str], sparse.spmatrix] | None = None,
        readout_map: tuple[Sequence[str], sparse.spmatrix] | None = None,
        port_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if capacity <= 0 or capacity > 4096:
            raise ValueError("capacity must be between 1 and 4096")
        if not 0 < tau or not 0 <= gain < 1:
            raise ValueError("tau must be positive and recurrent gain must be in [0, 1)")
        self.graph = graph
        self.capacity = capacity
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.tau = float(tau)
        self.gain = float(gain)
        self.support_recovery = float(support_recovery)
        self.n = int(graph.n)
        self.graph_hash = str(graph.hash)
        self.edge_count = int(graph.edge_count)
        try:
            self.port_metadata = json.loads(json.dumps(dict(port_metadata or {
                "mode": "default",
                "name": "malecns-default-16x48",
            }), sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("port metadata must be finite JSON data") from exc
        if not self.port_metadata.get("mode") or not self.port_metadata.get("name"):
            raise ValueError("port metadata needs nonempty mode and name")

        matrix = graph.matrix(normalized=True, signed=True)
        if matrix.shape != (self.n, self.n) or not sparse.issparse(matrix):
            raise ValueError("MaleCNS matrix must be sparse N x N")
        self.matrix = _torch_csr(matrix, self.device)

        if input_map is None:
            self.input_names, inputs = _mapping_pair(graph, "default_input_map")
        else:
            self.input_names = [str(name) for name in input_map[0]]
            inputs = input_map[1].tocsr().astype(np.float32, copy=False)
        if inputs.shape != (self.n, len(self.input_names)):
            raise ValueError("input map must have shape N x channels")
        self.input_matrix = _torch_csr(inputs, self.device)

        if readout_map is None:
            self.readout_names, readouts = _mapping_pair(graph, "default_readout_map")
        else:
            self.readout_names = [str(name) for name in readout_map[0]]
            readouts = readout_map[1].tocsr().astype(np.float32, copy=False)
        if readouts.shape != (len(self.readout_names), self.n):
            raise ValueError("readout map must have shape outputs x N")
        self.readout_matrix = _torch_csr(readouts, self.device)

        self._input_position = {name: index for index, name in enumerate(self.input_names)}
        self._slots: dict[str, int] = {}
        self._resident_for_slot: list[str | None] = [None] * capacity
        state_shape = (capacity, self.n)
        self.rates = torch.zeros(state_shape, dtype=self.dtype, device=self.device)
        self.adaptation = torch.zeros_like(self.rates)
        self.support = torch.ones_like(self.rates)
        self.times = torch.zeros(capacity, dtype=torch.float64, device="cpu")

    @classmethod
    def from_malecns(
        cls, path: str | Path | None = None, **kwargs: Any
    ) -> "RemoteBrain":
        from .malecns import MaleCNSGraph

        return cls(MaleCNSGraph.load(path, mmap=True), **kwargs)

    @property
    def resident_ids(self) -> list[str]:
        return [resident for resident in self._resident_for_slot if resident is not None]

    def add_residents(self, resident_ids: Sequence[str]) -> dict[str, int]:
        clean = [str(resident) for resident in resident_ids]
        if not clean or len(set(clean)) != len(clean):
            raise ValueError("resident IDs must be a nonempty unique list")
        if any(not resident or len(resident) > 128 for resident in clean):
            raise ValueError("resident IDs must contain 1-128 characters")
        if any(resident in self._slots for resident in clean):
            raise ValueError("resident already exists")
        free = [i for i, resident in enumerate(self._resident_for_slot) if resident is None]
        if len(free) < len(clean):
            raise ValueError("resident capacity exceeded")
        assigned = dict(zip(clean, free[: len(clean)], strict=True))
        for resident, slot in assigned.items():
            self._slots[resident] = slot
            self._resident_for_slot[slot] = resident
            self.rates[slot].zero_()
            self.adaptation[slot].zero_()
            self.support[slot].fill_(1)
            self.times[slot] = 0
        return assigned

    def remove_residents(self, resident_ids: Sequence[str]) -> None:
        clean = [str(resident) for resident in resident_ids]
        if not clean or len(set(clean)) != len(clean):
            raise ValueError("resident IDs must be a nonempty unique list")
        if any(resident not in self._slots for resident in clean):
            raise KeyError("resident does not exist")
        for resident in clean:
            slot = self._slots.pop(resident)
            self._resident_for_slot[slot] = None
            self.rates[slot].zero_()
            self.adaptation[slot].zero_()
            self.support[slot].fill_(1)
            self.times[slot] = 0

    @torch.no_grad()
    def step(self, residents: Sequence[Mapping[str, Any]], dt: float) -> list[dict[str, Any]]:
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        if not residents or len(residents) > self.capacity:
            raise ValueError("step must contain 1..capacity residents")
        if any(not isinstance(item, Mapping) for item in residents):
            raise ValueError("each resident step must be an object")
        ids = [str(item.get("id", "")) for item in residents]
        if len(set(ids)) != len(ids) or any(resident not in self._slots for resident in ids):
            raise ValueError("step resident IDs must be unique and already allocated")
        slots = torch.tensor([self._slots[resident] for resident in ids], device=self.device)
        channels = np.zeros((len(residents), len(self.input_names)), dtype=np.float32)
        for row, item in enumerate(residents):
            senses = item.get("senses")
            if not isinstance(senses, Mapping):
                raise ValueError("each resident needs a senses object")
            unknown = set(senses) - set(self._input_position)
            if unknown:
                raise ValueError(f"unknown sensory channels: {sorted(unknown)}")
            for name, value in senses.items():
                scalar = float(value)
                if not np.isfinite(scalar) or not 0 <= scalar <= 1:
                    raise ValueError("sensory channel values must be finite in [0, 1]")
                channels[row, self._input_position[name]] = scalar

        selected_rates = self.rates.index_select(0, slots)
        selected_adaptation = self.adaptation.index_select(0, slots)
        selected_support = self.support.index_select(0, slots)
        channel_tensor = torch.as_tensor(channels, device=self.device)
        drive = torch.sparse.mm(self.input_matrix, channel_tensor.T).T
        alpha = min(1.0, dt / 2 / self.tau)
        for _ in range(2):
            recurrent = torch.sparse.mm(self.matrix, selected_rates.T).T
            target = torch.relu(
                torch.tanh(
                    0.005
                    + drive
                    + self.gain * recurrent
                    - 0.10 * selected_adaptation
                )
            )
            selected_rates.add_(
                alpha * (target * selected_support - selected_rates)
            )
        selected_adaptation.add_(
            dt / 5 * (selected_rates - selected_adaptation)
        )
        selected_support.add_(
            dt
            * (
                self.support_recovery * (1 - selected_support)
                - 0.003 * selected_rates
            )
        ).clamp_(0.65, 1)
        self.rates.index_copy_(0, slots, selected_rates)
        self.adaptation.index_copy_(0, slots, selected_adaptation)
        self.support.index_copy_(0, slots, selected_support)
        readout = torch.sparse.mm(self.readout_matrix, selected_rates.T).T
        activity_mean = selected_rates.mean(dim=1)
        activity_peak = selected_rates.amax(dim=1)
        support_mean = selected_support.mean(dim=1)
        readout_cpu = readout.float().cpu().numpy()
        means = activity_mean.float().cpu().numpy()
        peaks = activity_peak.float().cpu().numpy()
        supports = support_mean.float().cpu().numpy()
        output = []
        for row, (resident, slot) in enumerate(zip(ids, slots.cpu().tolist(), strict=True)):
            self.times[slot] += dt
            vector = readout_cpu[row].tolist()
            output.append(
                {
                    "id": resident,
                    "time": float(self.times[slot]),
                    "features": vector,
                    "readout_vector": vector,
                    "readouts": dict(zip(self.readout_names, vector, strict=True)),
                    "activity": float(means[row]),
                    "activity_mean": float(means[row]),
                    "activity_peak": float(peaks[row]),
                    "support": float(supports[row]),
                    "support_mean": float(supports[row]),
                }
            )
        return output

    def metadata(self) -> dict[str, Any]:
        device: dict[str, Any] = {"type": self.device.type}
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            device.update(
                {
                    "name": properties.name,
                    "gcn_arch_name": getattr(properties, "gcnArchName", None),
                    "memory_free_bytes": free,
                    "memory_total_bytes": total,
                    "memory_allocated_bytes": torch.cuda.memory_allocated(self.device),
                }
            )
        return {
            "graph": {
                "sha256": self.graph_hash,
                "neurons": self.n,
                "edges": self.edge_count,
                "source": self.graph.manifest,
            },
            "device": device,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "capacity": self.capacity,
            "residents": self.resident_ids,
            "dynamics": {
                "tau": self.tau,
                "gain": self.gain,
                "support_recovery": self.support_recovery,
                "substeps": 2,
            },
            "inputs": self.input_names,
            "readouts": self.readout_names,
            "ports": {
                **self.port_metadata,
                "input_count": len(self.input_names),
                "readout_count": len(self.readout_names),
            },
        }

    def snapshot(
        self,
        directory: str | Path,
        name: str,
        resident_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        path = self._snapshot_path(directory, name)
        if resident_ids is None:
            residents = self.resident_ids
            scope = "all"
        else:
            residents = [str(resident) for resident in resident_ids]
            if not residents or len(set(residents)) != len(residents):
                raise ValueError("snapshot resident IDs must be a nonempty unique list")
            missing = [resident for resident in residents if resident not in self._slots]
            if missing:
                raise KeyError(f"snapshot residents are not allocated: {missing}")
            scope = "cohort"
        active = [(resident, self._slots[resident]) for resident in residents]
        slots = [slot for _, slot in active]
        resident_ids = np.asarray([resident for resident, _ in active], dtype=np.str_)
        metadata = {
            "version": 2,
            "scope": scope,
            "graph_sha256": self.graph_hash,
            "neurons": self.n,
            "input_names": self.input_names,
            "readout_names": self.readout_names,
            "tau": self.tau,
            "gain": self.gain,
            "support_recovery": self.support_recovery,
            "ports": self.port_metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                resident_ids=resident_ids,
                rates=self.rates[slots].float().cpu().numpy(),
                adaptation=self.adaptation[slots].float().cpu().numpy(),
                support=self.support[slots].float().cpu().numpy(),
                times=self.times[slots].numpy(),
            )
        os.replace(temporary, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "name": name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "scope": scope,
            "residents": residents,
        }

    def restore(
        self,
        directory: str | Path,
        name: str,
        expected_sha256: str | None = None,
        resident_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        path = self._snapshot_path(directory, name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("snapshot checksum does not match")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            version = metadata.get("version")
            if (
                version not in {1, 2}
                or metadata.get("graph_sha256") != self.graph_hash
                or metadata.get("neurons") != self.n
                or metadata.get("input_names") != self.input_names
                or metadata.get("readout_names") != self.readout_names
                or metadata.get("tau") != self.tau
                or metadata.get("gain") != self.gain
                or metadata.get("support_recovery") != self.support_recovery
            ):
                raise ValueError("snapshot is incompatible with this graph or mapping")
            if version == 2 and metadata.get("ports") != self.port_metadata:
                raise ValueError("snapshot belongs to a different neural port interface")
            residents = data["resident_ids"].astype(str).tolist()
            if len(residents) > self.capacity or len(set(residents)) != len(residents):
                raise ValueError("snapshot resident set is invalid")
            shape = (len(residents), self.n)
            arrays = {
                name: np.asarray(data[name], dtype=np.float32)
                for name in ("rates", "adaptation", "support")
            }
            if any(array.shape != shape or not np.isfinite(array).all() for array in arrays.values()):
                raise ValueError("snapshot neural state is invalid")
            times = np.asarray(data["times"], dtype=np.float64)
            if times.shape != (len(residents),) or not np.isfinite(times).all():
                raise ValueError("snapshot times are invalid")
        if resident_ids is not None:
            expected_residents = [str(resident) for resident in resident_ids]
            if not expected_residents or len(set(expected_residents)) != len(expected_residents):
                raise ValueError("restore resident IDs must be a nonempty unique list")
            if expected_residents != residents:
                raise ValueError("restore resident IDs differ from the snapshot receipt")
        scope = metadata.get("scope", "all") if version == 2 else "all"
        if scope not in {"all", "cohort"}:
            raise ValueError("snapshot scope is invalid")
        if scope == "all":
            self._slots.clear()
            self._resident_for_slot = [None] * self.capacity
            self.rates.zero_()
            self.adaptation.zero_()
            self.support.fill_(1)
            self.times.zero_()
            if residents:
                self.add_residents(residents)
        else:
            missing = [resident for resident in residents if resident not in self._slots]
            available = sum(resident is None for resident in self._resident_for_slot)
            if len(missing) > available:
                raise ValueError("cohort restore exceeds remaining resident capacity")
            if missing:
                self.add_residents(missing)
        slot_values = [self._slots[resident] for resident in residents]
        slots = torch.as_tensor(slot_values, device=self.device, dtype=torch.long)
        for field, array in arrays.items():
            getattr(self, field).index_copy_(
                0, slots, torch.as_tensor(array, device=self.device)
            )
        self.times[slot_values] = torch.from_numpy(times.copy())
        return {
            "name": name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "scope": scope,
            "residents": residents,
        }

    @staticmethod
    def _snapshot_path(directory: str | Path, name: str) -> Path:
        if not SNAPSHOT_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError("snapshot name must be 1-80 safe filename characters")
        return Path(directory).resolve() / f"{name}.npz"


class SequencedBrain:
    """Serialize mutations and reject missing, duplicated, or reordered requests."""

    def __init__(self, brain: RemoteBrain, snapshot_directory: str | Path):
        self.brain = brain
        self.snapshot_directory = Path(snapshot_directory)
        self.next_sequence = 0
        self.lock = threading.RLock()

    def mutate(self, sequence: Any, operation: Callable[[], Any]) -> tuple[int, Any]:
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ValueError("seq must be an integer")
        with self.lock:
            if sequence != self.next_sequence:
                raise ValueError(f"expected seq {self.next_sequence}, received {sequence}")
            result = operation()
            self.next_sequence += 1
            return sequence, result
