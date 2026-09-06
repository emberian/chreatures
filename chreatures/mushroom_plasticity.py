"""Bounded associative plasticity on a measured MaleCNS mushroom-body scaffold.

The anatomical substrate is the bilateral KC -> MBON11 (gamma1pedc) edge set.
Raw synapse counts and their normalized baseline are immutable.  Eligibility,
modulator state, and learned efficacy deviations belong to one private model
instance and never modify the source graph or bundle.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BUNDLE_FORMAT = "chreatures-mushroom-substrate-v1"
SNAPSHOT_FORMAT = "chreatures-mushroom-plasticity-snapshot-v1"
BRIDGE_FORMAT = "chreatures-mushroom-fullgraph-bridge-v1"
BRIDGE_SNAPSHOT_FORMAT = "chreatures-mushroom-fullgraph-snapshot-v1"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(graph_hash: str, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(BUNDLE_FORMAT.encode())
    digest.update(graph_hash.encode())
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _bridge_content_hash(
    graph_hash: str, substrate_hash: str, arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(BRIDGE_FORMAT.encode())
    digest.update(graph_hash.encode())
    digest.update(substrate_hash.encode())
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.flags.writeable = False
    return result


class MushroomBodySubstrate:
    """Exact compact KC -> MBON edge bundle plus compartment references."""

    ARRAY_DTYPES = {
        "kc_neuron_indices": np.int32,
        "kc_body_ids": np.int64,
        "mbon_neuron_indices": np.int32,
        "mbon_body_ids": np.int64,
        "dan_neuron_indices": np.int32,
        "dan_body_ids": np.int64,
        "edge_source_positions": np.int32,
        "edge_source_neuron_indices": np.int32,
        "edge_source_body_ids": np.int64,
        "edge_target_positions": np.int32,
        "edge_target_neuron_indices": np.int32,
        "edge_target_body_ids": np.int64,
        "synapse_counts": np.uint32,
        "baseline_weights": np.float64,
    }

    def __init__(self, metadata: Mapping[str, Any], **arrays: np.ndarray) -> None:
        missing = set(self.ARRAY_DTYPES).difference(arrays)
        extra = set(arrays).difference(self.ARRAY_DTYPES)
        if missing or extra:
            raise ValueError(
                f"substrate arrays differ; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        self.metadata = json.loads(json.dumps(dict(metadata), sort_keys=True))
        for name, dtype in self.ARRAY_DTYPES.items():
            setattr(self, name, _readonly(np.asarray(arrays[name]), dtype))
        self._validate()

    @property
    def graph_hash(self) -> str:
        return str(self.metadata["graph_sha256"])

    @property
    def substrate_hash(self) -> str:
        return str(self.metadata["substrate_sha256"])

    @property
    def kc_count(self) -> int:
        return len(self.kc_body_ids)

    @property
    def target_count(self) -> int:
        return len(self.mbon_body_ids)

    @property
    def edge_count(self) -> int:
        return len(self.synapse_counts)

    def _arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.ARRAY_DTYPES}

    def _validate(self) -> None:
        if self.metadata.get("format") != BUNDLE_FORMAT:
            raise ValueError("unsupported mushroom substrate format")
        if self.kc_count < 1 or self.target_count != 2 or len(self.dan_body_ids) != 2:
            raise ValueError("gamma1pedc substrate population sizes are malformed")
        edge_arrays = (
            self.edge_source_positions,
            self.edge_source_neuron_indices,
            self.edge_source_body_ids,
            self.edge_target_positions,
            self.edge_target_neuron_indices,
            self.edge_target_body_ids,
            self.synapse_counts,
            self.baseline_weights,
        )
        if not self.edge_count or any(len(value) != self.edge_count for value in edge_arrays):
            raise ValueError("substrate edge arrays are not aligned")
        if (
            self.edge_source_positions.min() < 0
            or self.edge_source_positions.max() >= self.kc_count
            or self.edge_target_positions.min() < 0
            or self.edge_target_positions.max() >= self.target_count
        ):
            raise ValueError("substrate edge positions are out of range")
        if np.any(self.synapse_counts < 1) or not np.isfinite(self.baseline_weights).all():
            raise ValueError("substrate weights are malformed")
        if np.any(self.baseline_weights <= 0):
            raise ValueError("baseline weights must be positive")
        if not np.array_equal(
            self.kc_neuron_indices[self.edge_source_positions],
            self.edge_source_neuron_indices,
        ) or not np.array_equal(
            self.kc_body_ids[self.edge_source_positions], self.edge_source_body_ids
        ):
            raise ValueError("substrate source identity arrays disagree")
        if not np.array_equal(
            self.mbon_neuron_indices[self.edge_target_positions],
            self.edge_target_neuron_indices,
        ) or not np.array_equal(
            self.mbon_body_ids[self.edge_target_positions], self.edge_target_body_ids
        ):
            raise ValueError("substrate target identity arrays disagree")
        for target in range(self.target_count):
            observed = self.baseline_weights[self.edge_target_positions == target].sum()
            if not np.isclose(observed, 1.0, rtol=0, atol=1e-12):
                raise ValueError("baseline weights must sum to one for each MBON")
        observed_hash = _content_hash(self.graph_hash, self._arrays())
        if observed_hash != self.substrate_hash:
            raise ValueError("substrate content hash differs")

    @classmethod
    def from_graph(cls, graph: Any) -> "MushroomBodySubstrate":
        kc_all = np.flatnonzero(graph.classes == "Kenyon_Cell").astype(np.int32)
        mbon = np.flatnonzero(graph.types == "MBON11").astype(np.int32)
        dan = np.flatnonzero(graph.types == "PPL101").astype(np.int32)
        if len(kc_all) != 4064 or len(mbon) != 2 or len(dan) != 2:
            raise ValueError("MaleCNS gamma1pedc annotations differ from the pinned selection")
        mbon = np.asarray(
            sorted(mbon, key=lambda index: (str(graph.sides[index]), int(graph.body_ids[index]))),
            dtype=np.int32,
        )
        dan = np.asarray(
            sorted(dan, key=lambda index: (str(graph.sides[index]), int(graph.body_ids[index]))),
            dtype=np.int32,
        )
        kc_mask = np.zeros(graph.n, dtype=bool)
        kc_mask[kc_all] = True
        edge_sources: list[np.ndarray] = []
        edge_targets: list[np.ndarray] = []
        edge_counts: list[np.ndarray] = []
        for target_position, target in enumerate(mbon):
            start, stop = int(graph.indptr[target]), int(graph.indptr[target + 1])
            selected = kc_mask[graph.indices[start:stop]]
            sources = np.asarray(graph.indices[start:stop][selected], dtype=np.int32)
            counts = np.asarray(graph.counts[start:stop][selected], dtype=np.uint32)
            edge_sources.append(sources)
            edge_targets.append(
                np.full(len(sources), target_position, dtype=np.int32)
            )
            edge_counts.append(counts)
        source_neuron_indices = np.concatenate(edge_sources)
        target_positions = np.concatenate(edge_targets)
        counts = np.concatenate(edge_counts)
        kc = np.unique(source_neuron_indices).astype(np.int32)
        source_positions = np.searchsorted(kc, source_neuron_indices).astype(np.int32)
        target_neuron_indices = mbon[target_positions]
        baseline = counts.astype(np.float64)
        target_synapses = []
        for target in range(len(mbon)):
            mask = target_positions == target
            total = int(counts[mask].sum(dtype=np.uint64))
            target_synapses.append(total)
            baseline[mask] /= total

        def connectivity_context(
            sources: np.ndarray, targets: np.ndarray
        ) -> dict[str, int]:
            source_mask = np.zeros(graph.n, dtype=bool)
            source_mask[sources] = True
            connections = 0
            synapses = 0
            for target in targets:
                start, stop = int(graph.indptr[target]), int(graph.indptr[target + 1])
                selected = source_mask[graph.indices[start:stop]]
                connections += int(np.count_nonzero(selected))
                synapses += int(
                    graph.counts[start:stop][selected].sum(dtype=np.uint64)
                )
            return {"connections": connections, "synapses": synapses}

        arrays = {
            "kc_neuron_indices": kc,
            "kc_body_ids": np.asarray(graph.body_ids[kc], dtype=np.int64),
            "mbon_neuron_indices": mbon,
            "mbon_body_ids": np.asarray(graph.body_ids[mbon], dtype=np.int64),
            "dan_neuron_indices": dan,
            "dan_body_ids": np.asarray(graph.body_ids[dan], dtype=np.int64),
            "edge_source_positions": source_positions,
            "edge_source_neuron_indices": source_neuron_indices,
            "edge_source_body_ids": np.asarray(
                graph.body_ids[source_neuron_indices], dtype=np.int64
            ),
            "edge_target_positions": target_positions,
            "edge_target_neuron_indices": target_neuron_indices,
            "edge_target_body_ids": np.asarray(
                graph.body_ids[target_neuron_indices], dtype=np.int64
            ),
            "synapse_counts": counts,
            "baseline_weights": baseline,
        }
        substrate_hash = _content_hash(str(graph.hash), arrays)
        metadata = {
            "format": BUNDLE_FORMAT,
            "version": 1,
            "name": "MaleCNS-v1.0-gamma1pedc-KC-MBON11",
            "graph_sha256": str(graph.hash),
            "substrate_sha256": substrate_hash,
            "selection": {
                "source": {"class": "Kenyon_Cell"},
                "target": {"type": "MBON11"},
                "dan_reference": {"type": "PPL101"},
                "compartment_basis": "MaleCNS instance annotations y1pedc/y1ped",
            },
            "counts": {
                "annotated_kcs": int(len(kc_all)),
                "connected_kcs": int(len(kc)),
                "mbons": int(len(mbon)),
                "dan_references": int(len(dan)),
                "edges": int(len(counts)),
                "synapses": int(counts.sum(dtype=np.uint64)),
            },
            "targets": [
                {
                    "body_id": int(graph.body_ids[index]),
                    "instance": str(graph.instances[index]),
                    "side": str(graph.sides[index]),
                    "effective_nt_annotation": str(graph.effective_nt[index]),
                    "nt_basis": str(graph.nt_basis[index]),
                    "edges": int(np.count_nonzero(target_positions == position)),
                    "synapses": target_synapses[position],
                }
                for position, index in enumerate(mbon)
            ],
            "dan_references": [
                {
                    "body_id": int(graph.body_ids[index]),
                    "instance": str(graph.instances[index]),
                    "side": str(graph.sides[index]),
                    "status": str(graph.statuses[index]),
                    "status_label": str(graph.status_labels[index]),
                    "effective_nt_annotation": str(graph.effective_nt[index]),
                    "nt_basis": str(graph.nt_basis[index]),
                }
                for index in dan
            ],
            "dan_connectivity_context": {
                "PPL101_to_Kenyon_Cell": connectivity_context(dan, kc_all),
                "Kenyon_Cell_to_PPL101": connectivity_context(kc_all, dan),
                "PPL101_to_MBON11": connectivity_context(dan, mbon),
                "MBON11_to_PPL101": connectivity_context(mbon, dan),
                "role": "measured context only; these edges are outside the plastic edge bundle",
            },
            "baseline": {
                "raw": "MaleCNS minconf-0.5 directed synapse counts",
                "model_weight": "count divided by all selected KC counts into each MBON",
                "immutable": True,
            },
            "dataset": dict(graph.manifest["dataset"]),
            "sources": dict(graph.manifest["sources"]),
            "model_boundary": {
                "measured": "neuron identities, annotations, directed edges, synapse counts",
                "engineered": "count normalization, cue activity, eligibility equation, efficacy deviations, synthetic modulator",
                "transmitter_warning": "annotations are provenance and do not establish receptor physiology or a signed plasticity rule",
            },
        }
        return cls(metadata, **arrays)

    def save(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
                **self._arrays(),
            )
        os.replace(temporary, path)
        receipt = {
            "format": BUNDLE_FORMAT,
            "artifact": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "substrate_sha256": self.substrate_hash,
            "graph_sha256": self.graph_hash,
            "license": self.metadata["dataset"]["license"],
            "license_url": self.metadata["dataset"]["license_url"],
            "attribution": self.metadata["dataset"]["citation"],
            "source_sha256": {
                name: source["sha256"]
                for name, source in self.metadata["sources"].items()
            },
        }
        path.with_suffix(".json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return receipt

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str | None = None
    ) -> "MushroomBodySubstrate":
        path = Path(path)
        if expected_sha256 is not None and _sha256(path) != expected_sha256:
            raise ValueError("mushroom substrate artifact checksum differs")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            arrays = {name: np.asarray(data[name]) for name in cls.ARRAY_DTYPES}
        return cls(metadata, **arrays)


class MushroomFullGraphBridgeSpec:
    """Exact mapping from the compact substrate into canonical recurrence."""

    ARRAY_DTYPES = {
        "selected_neuron_indices": np.int32,
        "selected_body_ids": np.int64,
        "target_neuron_indices": np.int32,
        "target_body_ids": np.int64,
        "edge_source_positions": np.int32,
        "edge_source_neuron_indices": np.int32,
        "edge_source_body_ids": np.int64,
        "edge_target_positions": np.int32,
        "synapse_counts": np.uint32,
        "source_model_signs": np.float32,
        "full_row_weights": np.float32,
        "target_row_synapses": np.uint64,
    }

    def __init__(self, metadata: Mapping[str, Any], **arrays: np.ndarray) -> None:
        missing = set(self.ARRAY_DTYPES).difference(arrays)
        extra = set(arrays).difference(self.ARRAY_DTYPES)
        if missing or extra:
            raise ValueError(
                f"bridge arrays differ; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        self.metadata = json.loads(json.dumps(dict(metadata), sort_keys=True))
        for name, dtype in self.ARRAY_DTYPES.items():
            setattr(self, name, _readonly(np.asarray(arrays[name]), dtype))
        self._validate()

    @property
    def graph_hash(self) -> str:
        return str(self.metadata["graph_sha256"])

    @property
    def substrate_hash(self) -> str:
        return str(self.metadata["substrate_sha256"])

    @property
    def bridge_hash(self) -> str:
        return str(self.metadata["bridge_sha256"])

    @property
    def selected_count(self) -> int:
        return len(self.selected_neuron_indices)

    @property
    def edge_count(self) -> int:
        return len(self.full_row_weights)

    def _arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.ARRAY_DTYPES}

    def _validate(self) -> None:
        if self.metadata.get("format") != BRIDGE_FORMAT:
            raise ValueError("unsupported mushroom full-graph bridge format")
        counts = self.metadata["counts"]
        kc_count = int(counts["connected_kcs"])
        if (
            self.selected_count != kc_count + 4
            or len(self.target_neuron_indices) != 2
            or len(self.target_body_ids) != 2
            or len(self.target_row_synapses) != 2
            or self.edge_count != int(counts["edges"])
        ):
            raise ValueError("mushroom bridge population sizes are malformed")
        edge_arrays = (
            self.edge_source_positions,
            self.edge_source_neuron_indices,
            self.edge_source_body_ids,
            self.edge_target_positions,
            self.synapse_counts,
            self.source_model_signs,
            self.full_row_weights,
        )
        if any(len(value) != self.edge_count for value in edge_arrays):
            raise ValueError("mushroom bridge edge arrays are not aligned")
        if (
            self.edge_source_positions.min() < 0
            or self.edge_source_positions.max() >= kc_count
            or self.edge_target_positions.min() < 0
            or self.edge_target_positions.max() >= 2
            or np.any(self.target_row_synapses < 1)
        ):
            raise ValueError("mushroom bridge positions are out of range")
        if not np.array_equal(
            self.selected_neuron_indices[self.edge_source_positions],
            self.edge_source_neuron_indices,
        ) or not np.array_equal(
            self.selected_body_ids[self.edge_source_positions], self.edge_source_body_ids
        ):
            raise ValueError("mushroom bridge source identities disagree")
        if not np.array_equal(self.selected_neuron_indices[-2:], self.target_neuron_indices):
            raise ValueError("mushroom bridge targets must be the final selected neurons")
        if not np.array_equal(self.selected_body_ids[-2:], self.target_body_ids):
            raise ValueError("mushroom bridge target body IDs disagree")
        expected = self.synapse_counts.astype(np.float32)
        expected /= self.target_row_synapses[self.edge_target_positions].astype(np.float32)
        expected *= self.source_model_signs
        if not np.array_equal(expected, self.full_row_weights):
            raise ValueError("mushroom bridge full-row normalization differs")
        if not np.isfinite(self.full_row_weights).all():
            raise ValueError("mushroom bridge weights are not finite")
        observed_hash = _bridge_content_hash(
            self.graph_hash, self.substrate_hash, self._arrays()
        )
        if observed_hash != self.bridge_hash:
            raise ValueError("mushroom full-graph bridge content hash differs")

    @classmethod
    def from_graph(
        cls, graph: Any, substrate: MushroomBodySubstrate
    ) -> "MushroomFullGraphBridgeSpec":
        if str(graph.hash) != substrate.graph_hash:
            raise ValueError("substrate and full graph hashes differ")
        if not np.array_equal(
            graph.body_ids[substrate.kc_neuron_indices], substrate.kc_body_ids
        ) or not np.array_equal(
            graph.body_ids[substrate.mbon_neuron_indices], substrate.mbon_body_ids
        ) or not np.array_equal(
            graph.body_ids[substrate.dan_neuron_indices], substrate.dan_body_ids
        ):
            raise ValueError("substrate local indices do not map to its body IDs")
        selected_indices = np.concatenate(
            (
                substrate.kc_neuron_indices,
                substrate.dan_neuron_indices,
                substrate.mbon_neuron_indices,
            )
        ).astype(np.int32, copy=False)
        selected_body_ids = np.concatenate(
            (substrate.kc_body_ids, substrate.dan_body_ids, substrate.mbon_body_ids)
        ).astype(np.int64, copy=False)
        target_rows = np.asarray(
            graph.row_synapses[substrate.mbon_neuron_indices], dtype=np.uint64
        )
        source_signs = np.asarray(
            graph.sign[substrate.edge_source_neuron_indices], dtype=np.float32
        )
        full_weights = substrate.synapse_counts.astype(np.float32)
        full_weights /= target_rows[substrate.edge_target_positions].astype(np.float32)
        full_weights *= source_signs
        arrays = {
            "selected_neuron_indices": selected_indices,
            "selected_body_ids": selected_body_ids,
            "target_neuron_indices": substrate.mbon_neuron_indices,
            "target_body_ids": substrate.mbon_body_ids,
            "edge_source_positions": substrate.edge_source_positions,
            "edge_source_neuron_indices": substrate.edge_source_neuron_indices,
            "edge_source_body_ids": substrate.edge_source_body_ids,
            "edge_target_positions": substrate.edge_target_positions,
            "synapse_counts": substrate.synapse_counts,
            "source_model_signs": source_signs,
            "full_row_weights": full_weights,
            "target_row_synapses": target_rows,
        }
        bridge_hash = _bridge_content_hash(graph.hash, substrate.substrate_hash, arrays)
        metadata = {
            "format": BRIDGE_FORMAT,
            "version": 1,
            "graph_sha256": str(graph.hash),
            "substrate_sha256": substrate.substrate_hash,
            "bridge_sha256": bridge_hash,
            "counts": {
                "connected_kcs": substrate.kc_count,
                "dan_references": 2,
                "targets": 2,
                "selected_neurons": len(selected_indices),
                "edges": substrate.edge_count,
            },
            "selected_order": "connected KCs, PPL101 left/right, MBON11 left/right",
            "normalization": {
                "equation": "float32(synapse_count) / float32(all incoming row synapses) * float32(source model sign)",
                "denominator_scope": "all retained canonical incoming edges for each MBON11 row",
                "runtime_match": "MaleCNSGraph.matrix(normalized=True, signed=True, dtype=float32)",
                "caveat": "source model sign is an engineered runtime transform, not receptor physiology",
            },
            "integration": {
                "correction": "sum(full_row_weight * private_efficacy_deviation * lagged_actual_KC_rate) per MBON11",
                "placement": "inside the recurrent term before the runtime recurrent gain",
                "lag_steps": 1,
                "baseline_ownership": "the native full graph already applies every immutable baseline edge",
            },
            "dataset": dict(graph.manifest["dataset"]),
            "sources": dict(graph.manifest["sources"]),
        }
        return cls(metadata, **arrays)

    def save(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
                **self._arrays(),
            )
        os.replace(temporary, path)
        receipt = {
            "format": BRIDGE_FORMAT,
            "artifact": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "bridge_sha256": self.bridge_hash,
            "substrate_sha256": self.substrate_hash,
            "graph_sha256": self.graph_hash,
            "source_sha256": {
                name: source["sha256"]
                for name, source in self.metadata["sources"].items()
            },
            "license": self.metadata["dataset"]["license"],
            "license_url": self.metadata["dataset"]["license_url"],
            "attribution": self.metadata["dataset"]["citation"],
        }
        path.with_suffix(".json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return receipt

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str | None = None
    ) -> "MushroomFullGraphBridgeSpec":
        path = Path(path)
        if expected_sha256 is not None and _sha256(path) != expected_sha256:
            raise ValueError("mushroom bridge artifact checksum differs")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            arrays = {name: np.asarray(data[name]) for name in cls.ARRAY_DTYPES}
        return cls(metadata, **arrays)


@dataclass(frozen=True)
class PlasticityConfig:
    """Engineered research rule; values are not fitted physiological constants."""

    eligibility_tau: float = 0.8
    modulation_tau: float = 0.12
    depression_rate: float = 0.25
    maximum_depression: float = 0.8

    def validate(self) -> None:
        if (
            self.eligibility_tau <= 0
            or self.modulation_tau <= 0
            or self.depression_rate < 0
            or not 0 <= self.maximum_depression < 1
        ):
            raise ValueError("plasticity configuration values are out of range")


@dataclass(frozen=True)
class MushroomStep:
    response: np.ndarray
    baseline_response: np.ndarray
    eligibility_mean: float
    modulation_state: np.ndarray
    efficacy_deviation_mean: float
    time: float


class MushroomPlasticity:
    """Private eligibility and efficacy state over an immutable substrate."""

    def __init__(
        self,
        substrate: MushroomBodySubstrate,
        *,
        config: PlasticityConfig | None = None,
        plasticity_enabled: bool = True,
    ) -> None:
        self.substrate = substrate
        self.config = config or PlasticityConfig()
        self.config.validate()
        self.plasticity_enabled = bool(plasticity_enabled)
        self.eligibility = np.zeros(substrate.edge_count, dtype=np.float64)
        self.efficacy_deviation = np.zeros(substrate.edge_count, dtype=np.float64)
        self.modulation_state = np.zeros(substrate.target_count, dtype=np.float64)
        self.time = 0.0
        self.update_count = 0

    def _activity(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.substrate.kc_count,):
            raise ValueError(
                f"input_KC_activity must have shape {(self.substrate.kc_count,)}"
            )
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError("input_KC_activity must be finite values in [0, 1]")
        return result

    def _modulator(self, value: float | Sequence[float] | np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 0:
            result = np.full(self.substrate.target_count, float(result))
        if result.shape != (self.substrate.target_count,):
            raise ValueError(
                f"modulator must be scalar or shape {(self.substrate.target_count,)}"
            )
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError("modulator must be finite values in [0, 1]")
        return result

    @property
    def effective_weights(self) -> np.ndarray:
        return self.substrate.baseline_weights * (1.0 + self.efficacy_deviation)

    def response(self, input_KC_activity: Sequence[float] | np.ndarray) -> np.ndarray:
        activity = self._activity(input_KC_activity)
        edge_activity = activity[self.substrate.edge_source_positions]
        return np.bincount(
            self.substrate.edge_target_positions,
            weights=self.effective_weights * edge_activity,
            minlength=self.substrate.target_count,
        ).astype(np.float64, copy=False)

    def baseline_response(
        self, input_KC_activity: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        activity = self._activity(input_KC_activity)
        return np.bincount(
            self.substrate.edge_target_positions,
            weights=(
                self.substrate.baseline_weights
                * activity[self.substrate.edge_source_positions]
            ),
            minlength=self.substrate.target_count,
        ).astype(np.float64, copy=False)

    def step(
        self,
        input_KC_activity: Sequence[float] | np.ndarray,
        modulator: float | Sequence[float] | np.ndarray,
        *,
        dt: float = 0.1,
    ) -> MushroomStep:
        """Advance a cue-before-modulator eligibility rule.

        The modulator is an explicit synthetic experimental input ordered by
        the bundle's MBON/PPL101 sides.  It has no built-in reward, punishment,
        nutrition, or motor meaning.  Current KC activity becomes eligible
        only after the current modulator update, enforcing forward ordering.
        """
        if not np.isfinite(dt) or not 0 < dt <= 10:
            raise ValueError("dt must be finite and in (0, 10]")
        activity = self._activity(input_KC_activity)
        pulse = self._modulator(modulator)
        self.eligibility *= np.exp(-dt / self.config.eligibility_tau)
        self.modulation_state *= np.exp(-dt / self.config.modulation_tau)
        np.maximum(self.modulation_state, pulse, out=self.modulation_state)
        if self.plasticity_enabled and np.any(self.modulation_state):
            decrement = (
                self.config.depression_rate
                * dt
                * self.eligibility
                * self.modulation_state[self.substrate.edge_target_positions]
            )
            self.efficacy_deviation -= decrement
            np.clip(
                self.efficacy_deviation,
                -self.config.maximum_depression,
                0.0,
                out=self.efficacy_deviation,
            )
            self.update_count += 1
        edge_activity = activity[self.substrate.edge_source_positions]
        np.maximum(self.eligibility, edge_activity, out=self.eligibility)
        self.time += dt
        return MushroomStep(
            response=self.response(activity),
            baseline_response=self.baseline_response(activity),
            eligibility_mean=float(self.eligibility.mean()),
            modulation_state=self.modulation_state.copy(),
            efficacy_deviation_mean=float(self.efficacy_deviation.mean()),
            time=self.time,
        )

    def washout(self, dt: float = 1.0) -> None:
        """Decay private traces without changing learned efficacy."""
        if not np.isfinite(dt) or not 0 < dt <= 100:
            raise ValueError("dt must be finite and in (0, 100]")
        self.eligibility *= np.exp(-dt / self.config.eligibility_tau)
        self.modulation_state *= np.exp(-dt / self.config.modulation_tau)
        self.time += dt

    def export_state(self) -> dict[str, np.ndarray]:
        return {
            "eligibility": self.eligibility.copy(),
            "efficacy_deviation": self.efficacy_deviation.copy(),
            "modulation_state": self.modulation_state.copy(),
            "time": np.asarray(self.time, dtype=np.float64),
            "update_count": np.asarray(self.update_count, dtype=np.int64),
        }

    def import_state(self, state: Mapping[str, np.ndarray]) -> None:
        expected = {
            "eligibility": (self.substrate.edge_count,),
            "efficacy_deviation": (self.substrate.edge_count,),
            "modulation_state": (self.substrate.target_count,),
        }
        values: dict[str, np.ndarray] = {}
        for name, shape in expected.items():
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"snapshot {name} must have shape {shape}")
            values[name] = value
        if np.any(values["eligibility"] < 0):
            raise ValueError("snapshot eligibility cannot be negative")
        if np.any(
            (values["efficacy_deviation"] < -self.config.maximum_depression)
            | (values["efficacy_deviation"] > 0)
        ):
            raise ValueError("snapshot efficacy deviations are out of range")
        if np.any((values["modulation_state"] < 0) | (values["modulation_state"] > 1)):
            raise ValueError("snapshot modulation state is out of range")
        time = float(np.asarray(state["time"], dtype=np.float64))
        updates = int(np.asarray(state["update_count"], dtype=np.int64))
        if not np.isfinite(time) or time < 0 or updates < 0:
            raise ValueError("snapshot counters are malformed")
        self.eligibility[:] = values["eligibility"]
        self.efficacy_deviation[:] = values["efficacy_deviation"]
        self.modulation_state[:] = values["modulation_state"]
        self.time = time
        self.update_count = updates

    def save_snapshot(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": SNAPSHOT_FORMAT,
            "version": 1,
            "substrate_sha256": self.substrate.substrate_hash,
            "config": asdict(self.config),
            "plasticity_enabled": self.plasticity_enabled,
        }
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                **self.export_state(),
            )
        os.replace(temporary, path)
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "substrate_sha256": self.substrate.substrate_hash,
        }

    @classmethod
    def load_snapshot(
        cls,
        path: str | Path,
        substrate: MushroomBodySubstrate,
        *,
        expected_sha256: str | None = None,
    ) -> "MushroomPlasticity":
        path = Path(path)
        if expected_sha256 is not None and _sha256(path) != expected_sha256:
            raise ValueError("mushroom plasticity snapshot checksum differs")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if (
                metadata.get("format") != SNAPSHOT_FORMAT
                or metadata.get("substrate_sha256") != substrate.substrate_hash
            ):
                raise ValueError("snapshot belongs to a different substrate")
            instance = cls(
                substrate,
                config=PlasticityConfig(**metadata["config"]),
                plasticity_enabled=bool(metadata["plasticity_enabled"]),
            )
            instance.import_state(
                {
                    name: np.asarray(data[name])
                    for name in (
                        "eligibility",
                        "efficacy_deviation",
                        "modulation_state",
                        "time",
                        "update_count",
                    )
                }
            )
        return instance


@dataclass(frozen=True)
class MushroomBridgeStep:
    """Post-step observations and recurrent correction for the next step."""

    target_recurrent_correction: np.ndarray
    actual_dan_rates: np.ndarray
    actual_mbon_rates: np.ndarray
    plasticity: MushroomStep


class WholeBrainMushroomBridge:
    """One-resident causal bridge between actual graph rates and private plasticity.

    A native step consumes ``target_recurrent_correction`` from the previous
    call.  After that step, ``advance`` receives actual selected-neuron rates and
    computes the correction for the next native step.  This is a deterministic
    one-step lag and avoids algebraic feedback through the same recurrent step.
    """

    MODULATOR_MODES = {"synthetic", "actual_ppl101_rate"}

    def __init__(
        self,
        substrate: MushroomBodySubstrate,
        bridge: MushroomFullGraphBridgeSpec,
        *,
        config: PlasticityConfig | None = None,
        plasticity_enabled: bool = True,
        modulator_mode: str = "synthetic",
    ) -> None:
        if bridge.substrate_hash != substrate.substrate_hash:
            raise ValueError("mushroom bridge and substrate hashes differ")
        if bridge.graph_hash != substrate.graph_hash:
            raise ValueError("mushroom bridge and substrate graph hashes differ")
        if modulator_mode not in self.MODULATOR_MODES:
            raise ValueError(
                f"modulator_mode must be one of {sorted(self.MODULATOR_MODES)}"
            )
        self.substrate = substrate
        self.bridge = bridge
        self.model = MushroomPlasticity(
            substrate, config=config, plasticity_enabled=plasticity_enabled
        )
        self.modulator_mode = modulator_mode
        self.target_recurrent_correction = np.zeros(2, dtype=np.float32)

    @property
    def bridge_hash(self) -> str:
        return self.bridge.bridge_hash

    @property
    def selected_neuron_indices(self) -> np.ndarray:
        return self.bridge.selected_neuron_indices

    @property
    def selected_body_ids(self) -> np.ndarray:
        return self.bridge.selected_body_ids

    @property
    def target_neuron_indices(self) -> np.ndarray:
        return self.bridge.target_neuron_indices

    @property
    def target_body_ids(self) -> np.ndarray:
        return self.bridge.target_body_ids

    def correction_for_current_step(self) -> np.ndarray:
        return self.target_recurrent_correction.copy()

    def _selected_rates(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        rates = np.asarray(values, dtype=np.float32)
        if rates.shape != (self.bridge.selected_count,):
            raise ValueError(
                f"selected_rates must have shape {(self.bridge.selected_count,)}"
            )
        if not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1)):
            raise ValueError("selected_rates must be finite values in [0, 1]")
        return rates

    def _correction(self, kc_rates: np.ndarray) -> np.ndarray:
        edge_rates = kc_rates[self.bridge.edge_source_positions]
        deviations = self.model.efficacy_deviation.astype(np.float32, copy=False)
        terms = self.bridge.full_row_weights * deviations * edge_rates
        result = np.zeros(2, dtype=np.float32)
        np.add.at(result, self.bridge.edge_target_positions, terms)
        return result

    def advance(
        self,
        selected_rates: Sequence[float] | np.ndarray,
        modulator: float | Sequence[float] | np.ndarray | None,
        *,
        dt: float = 0.1,
    ) -> MushroomBridgeStep:
        """Consume actual post-step rates and prepare the next-step correction.

        In ``synthetic`` mode a caller must supply the explicit scalar or
        bilateral pulse.  In ``actual_ppl101_rate`` mode the pulse is an
        engineered identity mapping of the two actual PPL101 rates and the
        argument must be ``None``.  Neither mode assigns behavioral meaning.
        """
        rates = self._selected_rates(selected_rates)
        kc_count = self.substrate.kc_count
        dan_rates = rates[kc_count : kc_count + 2]
        mbon_rates = rates[kc_count + 2 : kc_count + 4]
        if self.modulator_mode == "synthetic":
            if modulator is None:
                raise ValueError("synthetic modulator mode requires an explicit pulse")
            pulse = modulator
        else:
            if modulator is not None:
                raise ValueError(
                    "actual_ppl101_rate mode derives modulation from selected rates"
                )
            pulse = dan_rates
        plasticity = self.model.step(rates[:kc_count], pulse, dt=dt)
        self.target_recurrent_correction[:] = self._correction(rates[:kc_count])
        return MushroomBridgeStep(
            target_recurrent_correction=self.correction_for_current_step(),
            actual_dan_rates=dan_rates.copy(),
            actual_mbon_rates=mbon_rates.copy(),
            plasticity=plasticity,
        )

    def reset(self) -> None:
        self.model = MushroomPlasticity(
            self.substrate,
            config=self.model.config,
            plasticity_enabled=self.model.plasticity_enabled,
        )
        self.target_recurrent_correction.fill(0)

    def export_state(self) -> dict[str, np.ndarray]:
        state = self.model.export_state()
        state["target_recurrent_correction"] = (
            self.target_recurrent_correction.copy()
        )
        return state

    def import_state(self, state: Mapping[str, np.ndarray]) -> None:
        correction = np.asarray(state["target_recurrent_correction"], dtype=np.float32)
        if correction.shape != (2,) or not np.isfinite(correction).all():
            raise ValueError("snapshot target recurrent correction must have shape (2,)")
        self.model.import_state(state)
        self.target_recurrent_correction[:] = correction

    def metadata(self) -> dict[str, Any]:
        return {
            "format": BRIDGE_SNAPSHOT_FORMAT,
            "version": 1,
            "graph_sha256": self.bridge.graph_hash,
            "substrate_sha256": self.substrate.substrate_hash,
            "bridge_sha256": self.bridge.bridge_hash,
            "selected_neuron_indices": self.selected_neuron_indices.tolist(),
            "selected_body_ids": self.selected_body_ids.tolist(),
            "target_neuron_indices": self.target_neuron_indices.tolist(),
            "target_body_ids": self.target_body_ids.tolist(),
            "modulator_mode": self.modulator_mode,
            "lag_steps": 1,
            "config": asdict(self.model.config),
            "plasticity_enabled": self.model.plasticity_enabled,
        }

    def save_snapshot(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(self.metadata(), sort_keys=True)),
                **self.export_state(),
            )
        os.replace(temporary, path)
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "bridge_sha256": self.bridge.bridge_hash,
        }

    @classmethod
    def load_snapshot(
        cls,
        path: str | Path,
        substrate: MushroomBodySubstrate,
        bridge: MushroomFullGraphBridgeSpec,
        *,
        expected_sha256: str | None = None,
    ) -> "WholeBrainMushroomBridge":
        path = Path(path)
        if expected_sha256 is not None and _sha256(path) != expected_sha256:
            raise ValueError("whole-brain mushroom snapshot checksum differs")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            expected_identity = {
                "format": BRIDGE_SNAPSHOT_FORMAT,
                "version": 1,
                "graph_sha256": bridge.graph_hash,
                "substrate_sha256": substrate.substrate_hash,
                "bridge_sha256": bridge.bridge_hash,
                "selected_neuron_indices": bridge.selected_neuron_indices.tolist(),
                "selected_body_ids": bridge.selected_body_ids.tolist(),
                "target_neuron_indices": bridge.target_neuron_indices.tolist(),
                "target_body_ids": bridge.target_body_ids.tolist(),
                "lag_steps": 1,
            }
            if any(metadata.get(name) != value for name, value in expected_identity.items()):
                raise ValueError("whole-brain mushroom snapshot identity differs")
            instance = cls(
                substrate,
                bridge,
                config=PlasticityConfig(**metadata["config"]),
                plasticity_enabled=bool(metadata["plasticity_enabled"]),
                modulator_mode=str(metadata["modulator_mode"]),
            )
            instance.import_state(
                {
                    name: np.asarray(data[name])
                    for name in (
                        "eligibility",
                        "efficacy_deviation",
                        "modulation_state",
                        "time",
                        "update_count",
                        "target_recurrent_correction",
                    )
                }
            )
        return instance


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    original = np.asarray(value)
    array = np.ascontiguousarray(original)
    return {
        "dtype": array.dtype.str,
        "shape": list(original.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _decode_array(value: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(str(value["dtype"]))
    shape = tuple(int(size) for size in value["shape"])
    raw = base64.b64decode(str(value["data"]), validate=True)
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError("encoded mushroom state array byte length differs")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


class WholeBrainMushroomCohort:
    """Fixed-slot bridge adapter for an isolated native research circuit."""

    def __init__(
        self,
        substrate: MushroomBodySubstrate,
        bridge: MushroomFullGraphBridgeSpec,
        *,
        capacity: int = 3,
        config: PlasticityConfig | None = None,
        plasticity_enabled: bool = True,
        modulator_mode: str = "synthetic",
    ) -> None:
        if capacity < 1 or capacity > 64:
            raise ValueError("mushroom bridge capacity must be in 1..64")
        self.substrate = substrate
        self.bridge = bridge
        self.capacity = int(capacity)
        self._slots = [
            WholeBrainMushroomBridge(
                substrate,
                bridge,
                config=config,
                plasticity_enabled=plasticity_enabled,
                modulator_mode=modulator_mode,
            )
            for _ in range(self.capacity)
        ]

    @property
    def bridge_hash(self) -> str:
        return self.bridge.bridge_hash

    @property
    def config(self) -> PlasticityConfig:
        return self._slots[0].model.config

    @property
    def plasticity_enabled(self) -> bool:
        return self._slots[0].model.plasticity_enabled

    @property
    def modulator_mode(self) -> str:
        return self._slots[0].modulator_mode

    @property
    def selected_neuron_indices(self) -> np.ndarray:
        return self.bridge.selected_neuron_indices

    @property
    def selected_body_ids(self) -> np.ndarray:
        return self.bridge.selected_body_ids

    @property
    def target_neuron_indices(self) -> np.ndarray:
        return self.bridge.target_neuron_indices

    @property
    def target_body_ids(self) -> np.ndarray:
        return self.bridge.target_body_ids

    @property
    def pending_correction(self) -> np.ndarray:
        return np.stack(
            [slot.target_recurrent_correction for slot in self._slots], axis=1
        ).astype(np.float32, copy=False)

    def advance(
        self,
        selected_rates: np.ndarray,
        modulator: np.ndarray | None,
        active_mask: int,
        *,
        dt: float = 0.1,
    ) -> list[MushroomBridgeStep | None]:
        rates = np.asarray(selected_rates, dtype=np.float32)
        expected = (self.bridge.selected_count, self.capacity)
        if rates.shape != expected:
            raise ValueError(f"selected_rates must have shape {expected}")
        if active_mask < 0 or active_mask >= (1 << self.capacity):
            raise ValueError("mushroom active mask is out of range")
        if self.modulator_mode == "synthetic":
            if modulator is None:
                raise ValueError("synthetic mode requires bilateral modulation")
            pulses = np.asarray(modulator, dtype=np.float32)
            if pulses.shape != (2, self.capacity):
                raise ValueError(
                    f"modulator must have shape {(2, self.capacity)}"
                )
            if not np.isfinite(pulses).all() or np.any((pulses < 0) | (pulses > 1)):
                raise ValueError("modulator values must be finite values in [0, 1]")
        else:
            if modulator is not None:
                raise ValueError("actual PPL101 mode does not accept synthetic modulation")
            pulses = None
        results: list[MushroomBridgeStep | None] = []
        for slot, state in enumerate(self._slots):
            if not active_mask & (1 << slot):
                results.append(None)
                continue
            pulse = None if pulses is None else pulses[:, slot]
            results.append(state.advance(rates[:, slot], pulse, dt=dt))
        return results

    def reset_slots(self, active_mask: int) -> None:
        if active_mask < 0 or active_mask >= (1 << self.capacity):
            raise ValueError("mushroom active mask is out of range")
        for slot, state in enumerate(self._slots):
            if active_mask & (1 << slot):
                state.reset()

    def snapshot(self) -> dict[str, Any]:
        return {
            "format": "chreatures-mushroom-fullgraph-cohort-state-v1",
            "bridge_sha256": self.bridge.bridge_hash,
            "substrate_sha256": self.substrate.substrate_hash,
            "graph_sha256": self.bridge.graph_hash,
            "capacity": self.capacity,
            "config": asdict(self.config),
            "plasticity_enabled": self.plasticity_enabled,
            "modulator_mode": self.modulator_mode,
            "lag_steps": 1,
            "slots": [
                {name: _encode_array(value) for name, value in slot.export_state().items()}
                for slot in self._slots
            ],
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        expected = {
            "format": "chreatures-mushroom-fullgraph-cohort-state-v1",
            "bridge_sha256": self.bridge.bridge_hash,
            "substrate_sha256": self.substrate.substrate_hash,
            "graph_sha256": self.bridge.graph_hash,
            "capacity": self.capacity,
            "config": asdict(self.config),
            "plasticity_enabled": self.plasticity_enabled,
            "modulator_mode": self.modulator_mode,
            "lag_steps": 1,
        }
        if any(snapshot.get(name) != value for name, value in expected.items()):
            raise ValueError("mushroom cohort snapshot identity or configuration differs")
        slots = snapshot.get("slots")
        if not isinstance(slots, list) or len(slots) != self.capacity:
            raise ValueError("mushroom cohort snapshot slot count differs")
        decoded = [
            {name: _decode_array(value) for name, value in state.items()}
            for state in slots
        ]
        for state, values in zip(self._slots, decoded, strict=True):
            state.import_state(values)
