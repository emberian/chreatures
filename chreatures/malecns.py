"""Memory-mapped access to the full curated MaleCNS v1.0 graph.

The graph is anatomical connectivity.  Default interface populations come
from annotations, while channel injection, population readout, normalization,
and neuron-level transmitter signs are explicit runtime modeling choices.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Collection, Mapping
from functools import cached_property
from pathlib import Path
from typing import TypeAlias

import numpy as np
from scipy import sparse


DEFAULT_ARTIFACT_DIR = Path("/tank/chreatures/data/malecns/derived")
SelectorValue: TypeAlias = str | int | Collection[str] | Collection[int] | None
Selector: TypeAlias = Mapping[str, SelectorValue] | np.ndarray | Collection[int]
GRAPH_KINDS = ("canonical", "derived")

_FIELD_ALIASES = {
    "id": "ids",
    "body_id": "body_ids",
    "label": "labels",
    "instance": "instances",
    "type": "types",
    "side": "sides",
    "group": "superclasses",
    "groups": "superclasses",
    "superclass": "superclasses",
    "class": "classes",
    "class_": "classes",
    "subclass": "subclasses",
    "soma_neuromere": "soma_neuromeres",
    "entry_nerve": "entry_nerves",
    "exit_nerve": "exit_nerves",
    "status": "statuses",
    "status_label": "status_labels",
    "predicted_transmitter": "predicted_nt",
    "effective_transmitter": "effective_nt",
}

DEFAULT_INPUT_CHANNELS = [
    "odor L0", "odor L1", "odor L2", "odor R0", "odor R1", "odor R2",
    "obstacle left", "obstacle right", "red", "green", "blue",
    "tone low", "tone middle", "tone high", "shade", "contact",
]


def _resolve_artifact_dir(path: str | Path | None) -> Path:
    if path is None:
        path = os.environ.get("CHREATURES_MALECNS_DIR", DEFAULT_ARTIFACT_DIR)
    resolved = Path(path).expanduser()
    return resolved.parent if resolved.is_file() else resolved


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph_artifact(
    path: str | Path | None = None, *, kind: str = "canonical",
    expected_sha256: str | None = None, mmap: bool = True, verify: bool = True,
) -> MaleCNSGraph:
    """Load one explicit canonical or derived graph through the shared interface."""
    if kind not in GRAPH_KINDS:
        raise ValueError(f"graph kind must be one of {GRAPH_KINDS}")
    artifact = _resolve_artifact_dir(path)
    manifest_path = artifact / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MaleCNS manifest not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph_format = manifest.get("format")
    if graph_format not in {None, "chreatures-derived-circuit-v1"}:
        raise ValueError(f"unsupported neural graph artifact format {graph_format!r}")
    derived = graph_format == "chreatures-derived-circuit-v1"
    if kind == "canonical" and derived:
        raise ValueError("derived circuit requires explicit graph kind 'derived'")
    if kind == "derived" and not derived:
        raise ValueError("requested derived graph is not a compiled circuit blueprint")
    if kind == "derived" and expected_sha256 is None:
        raise ValueError("derived circuit requires an expected graph SHA-256")
    if expected_sha256 is not None and (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected graph SHA-256 must be lowercase hexadecimal")
    if derived:
        from .circuit_blueprint import DerivedCircuitGraph
        graph = DerivedCircuitGraph.load(artifact, mmap=mmap, verify=verify)
    else:
        graph = MaleCNSGraph.load(artifact, mmap=mmap, verify=verify)
    if expected_sha256 is not None and graph.hash != expected_sha256:
        raise ValueError(
            f"graph SHA-256 differs: expected {expected_sha256}, observed {graph.hash}"
        )
    return graph


class MaleCNSGraph:
    """The complete induced graph of traced MaleCNS v1.0 neurons.

    CSR rows are postsynaptic targets and ``indices`` are presynaptic sources.
    Large edge arrays are memory mapped by default.
    """

    _required_arrays = {
        "indptr.npy": np.dtype(np.int64),
        "indices.npy": np.dtype(np.int32),
        "counts.npy": np.dtype(np.uint32),
        "row_synapses.npy": np.dtype(np.uint64),
    }
    _required_metadata = {
        "body_ids", "ids", "labels", "instances", "types", "sides",
        "superclasses", "classes", "subclasses", "soma_neuromeres",
        "entry_nerves", "exit_nerves", "statuses", "status_labels",
        "predicted_nt", "ground_truth_nt", "consensus_nt", "effective_nt",
        "nt_basis", "nt_confidence", "sign",
    }

    def __init__(self, artifact_dir: str | Path | None = None, *, mmap: bool = True):
        self.path = _resolve_artifact_dir(artifact_dir)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"MaleCNS manifest not found at {manifest_path}; "
                "set CHREATURES_MALECNS_DIR or pass artifact_dir"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.hash = str(self.manifest.get("dataset_hash", ""))
        mode = "r" if mmap else None
        for filename, dtype in self._required_arrays.items():
            array = np.load(self.path / filename, mmap_mode=mode, allow_pickle=False)
            if array.dtype != dtype or array.ndim != 1:
                raise ValueError(f"Malformed {filename}: expected one-dimensional {dtype}")
            setattr(self, filename.removesuffix(".npy"), array)

        with np.load(self.path / "neurons.npz", allow_pickle=False) as metadata:
            missing = self._required_metadata.difference(metadata.files)
            if missing:
                raise ValueError(f"neurons.npz is missing fields: {sorted(missing)}")
            self.metadata_fields = tuple(metadata.files)
            for name in metadata.files:
                setattr(self, name, metadata[name])

        self.n = len(self.body_ids)
        self.edge_count = len(self.indices)
        self.shape = (self.n, self.n)
        self.groups = self.superclasses
        self._validate_structure()

    @classmethod
    def load(
        cls, artifact_dir: str | Path | None = None, *, mmap: bool = True,
        verify: bool = False,
    ) -> "MaleCNSGraph":
        graph = cls(artifact_dir, mmap=mmap)
        if verify:
            graph.verify_artifacts()
        return graph

    def _validate_structure(self) -> None:
        expected_n = int(self.manifest["counts"]["neurons"])
        expected_e = int(self.manifest["counts"]["edges"])
        if self.n != expected_n or self.edge_count != expected_e:
            raise ValueError("MaleCNS arrays disagree with manifest counts")
        if len(self.indptr) != self.n + 1 or len(self.row_synapses) != self.n:
            raise ValueError("Malformed MaleCNS CSR row arrays")
        if self.indptr[0] != 0 or self.indptr[-1] != self.edge_count:
            raise ValueError("Malformed MaleCNS CSR pointers")
        if np.any(np.diff(self.indptr) < 0):
            raise ValueError("MaleCNS CSR pointers are not monotonic")
        for field in self.metadata_fields:
            if len(getattr(self, field)) != self.n:
                raise ValueError(f"MaleCNS metadata field {field!r} is not row-aligned")
        if self.edge_count and (self.indices.min() < 0 or self.indices.max() >= self.n):
            raise ValueError("MaleCNS source index out of bounds")

    def verify_artifacts(self) -> None:
        """Hash all derived files against the pinned manifest."""
        for filename, expected in self.manifest["artifacts"].items():
            path = self.path / filename
            if path.stat().st_size != expected["bytes"]:
                raise ValueError(f"size mismatch for {filename}")
            observed = _sha256(path)
            if observed != expected["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {filename}: {observed}")

    def summary(self) -> dict:
        return {
            "neurons": self.n,
            "connections": self.edge_count,
            "synapses": int(self.manifest["counts"]["synapses"]),
            "afferent_neurons": len(self.population_indices("afferent")),
            "efferent_neurons": len(self.population_indices("efferent")),
            "sha256": self.hash,
            "source": self.manifest["dataset"],
        }

    def field(self, name: str) -> np.ndarray:
        canonical = _FIELD_ALIASES.get(name, name)
        if canonical not in self.metadata_fields:
            raise KeyError(f"unknown MaleCNS metadata field {name!r}")
        return getattr(self, canonical)

    def select(self, **criteria: SelectorValue) -> np.ndarray:
        """Return local neuron indices matching exact metadata criteria.

        A scalar means equality, a collection means membership, and ``None``
        selects missing/empty values.  Different fields are combined with AND.
        """
        mask = np.ones(self.n, dtype=bool)
        for name, requested in criteria.items():
            values = self.field(name)
            if requested is None:
                mask &= values == ""
            elif isinstance(requested, (str, int, np.integer)):
                mask &= values == requested
            else:
                mask &= np.isin(values, list(requested))
        return np.flatnonzero(mask).astype(np.int32)

    def population_indices(self, role: str) -> np.ndarray:
        """Return annotation-derived afferent or efferent neuron indices."""
        values = self.superclasses
        if role == "afferent":
            mask = np.asarray([
                ("_sensory" in value) or value.startswith("sensory_")
                for value in values
            ])
        elif role == "efferent":
            mask = np.asarray([
                ("_motor" in value)
                or ("_efferent" in value)
                or value.startswith("efferent_")
                for value in values
            ])
        else:
            raise ValueError("role must be 'afferent' or 'efferent'")
        return np.flatnonzero(mask).astype(np.int32)

    def _selector_indices(self, selector: Selector) -> np.ndarray:
        if isinstance(selector, Mapping):
            return self.select(**selector)
        array = np.asarray(selector)
        if array.dtype == np.bool_:
            if array.shape != (self.n,):
                raise ValueError(f"boolean selector must have shape {(self.n,)}")
            return np.flatnonzero(array).astype(np.int32)
        indices = np.asarray(array, dtype=np.int64)
        if indices.ndim != 1 or (len(indices) and (indices.min() < 0 or indices.max() >= self.n)):
            raise ValueError("selector indices are malformed or out of bounds")
        return np.unique(indices).astype(np.int32)

    @staticmethod
    def _gain(name: str, gains: float | Mapping[str, float]) -> float:
        return float(gains[name] if isinstance(gains, Mapping) else gains)

    def build_input_map(
        self,
        assignments: Mapping[str, Selector],
        *,
        gains: float | Mapping[str, float] = 1.0,
        normalize: bool = False,
    ) -> tuple[list[str], sparse.csr_matrix]:
        """Build a sparse neuron-by-channel injection map."""
        names = list(assignments)
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for column, (name, selector) in enumerate(assignments.items()):
            indices = self._selector_indices(selector)
            gain = self._gain(name, gains)
            value = gain / len(indices) if normalize and len(indices) else gain
            rows.append(indices)
            columns.append(np.full(len(indices), column, dtype=np.int32))
            data.append(np.full(len(indices), value, dtype=np.float32))
        if rows:
            row = np.concatenate(rows)
            column = np.concatenate(columns)
            values = np.concatenate(data)
        else:
            row = column = np.empty(0, dtype=np.int32)
            values = np.empty(0, dtype=np.float32)
        matrix = sparse.csr_matrix((values, (row, column)), shape=(self.n, len(names)))
        return names, matrix

    def build_readout_map(
        self,
        assignments: Mapping[str, Selector],
        *,
        gains: float | Mapping[str, float] = 1.0,
        normalize: bool = True,
    ) -> tuple[list[str], sparse.csr_matrix]:
        """Build a sparse readout-by-neuron population map."""
        names = list(assignments)
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for row, (name, selector) in enumerate(assignments.items()):
            indices = self._selector_indices(selector)
            gain = self._gain(name, gains)
            value = gain / len(indices) if normalize and len(indices) else gain
            rows.append(np.full(len(indices), row, dtype=np.int32))
            columns.append(indices)
            data.append(np.full(len(indices), value, dtype=np.float32))
        if rows:
            row_indices = np.concatenate(rows)
            column_indices = np.concatenate(columns)
            values = np.concatenate(data)
        else:
            row_indices = column_indices = np.empty(0, dtype=np.int32)
            values = np.empty(0, dtype=np.float32)
        matrix = sparse.csr_matrix(
            (values, (row_indices, column_indices)), shape=(len(names), self.n)
        )
        return names, matrix

    @cached_property
    def default_input_map(self) -> tuple[list[str], sparse.csr_matrix]:
        afferent = self.population_indices("afferent")
        afferent_mask = np.zeros(self.n, dtype=bool)
        afferent_mask[afferent] = True
        unassigned_side = ~np.isin(self.sides, ["L", "R"])
        left = (self.sides == "L") | (unassigned_side & ((self.body_ids % 2) == 0))
        right = (self.sides == "R") | (unassigned_side & ((self.body_ids % 2) == 1))
        olfactory = afferent_mask & (self.classes == "olfactory")
        visual = afferent_mask & (self.classes == "visual")
        auditory = afferent_mask & (self.subclasses == "auditory")
        shade = afferent_mask & np.isin(
            self.classes, ["hygrosensory", "thermosensory"]
        )
        contact = afferent_mask & np.isin(
            self.classes,
            ["mechanosensory", "mechanosensory_tactile", "mechanosensory_tbc"],
        )
        assignments: dict[str, np.ndarray] = {}
        for side_name, side_mask in (("L", left), ("R", right)):
            for odor in range(3):
                assignments[f"odor {side_name}{odor}"] = np.flatnonzero(
                    olfactory & side_mask & ((self.body_ids % 3) == odor)
                ).astype(np.int32)
        assignments["obstacle left"] = np.flatnonzero(visual & left).astype(np.int32)
        assignments["obstacle right"] = np.flatnonzero(visual & right).astype(np.int32)
        for bucket, name in enumerate(("red", "green", "blue")):
            assignments[name] = np.flatnonzero(
                visual & ((self.body_ids % 3) == bucket)
            ).astype(np.int32)
        for bucket, name in enumerate(("tone low", "tone middle", "tone high")):
            assignments[name] = np.flatnonzero(
                auditory & ((self.body_ids % 3) == bucket)
            ).astype(np.int32)
        assignments["shade"] = np.flatnonzero(shade).astype(np.int32)
        assignments["contact"] = np.flatnonzero(contact).astype(np.int32)
        if list(assignments) != DEFAULT_INPUT_CHANNELS:
            raise RuntimeError("default MaleCNS input channel order changed")
        return self.build_input_map(assignments)

    @cached_property
    def default_readout_map(self) -> tuple[list[str], sparse.csr_matrix]:
        efferent = self.population_indices("efferent")
        signatures: dict[str, list[int]] = {}
        for index in efferent:
            region = (
                self.exit_nerves[index]
                or self.soma_neuromeres[index]
                or self.classes[index]
                or "unassigned"
            )
            side = self.sides[index] or "unknown"
            name = f"{self.superclasses[index]}|{region}|{side}"
            signatures.setdefault(name, []).append(int(index))
        if len(signatures) < 48:
            raise ValueError("MaleCNS annotations do not provide 48 efferent populations")
        ranked = sorted(signatures, key=lambda name: (-len(signatures[name]), name))
        assignments = {
            name: np.asarray(signatures[name], dtype=np.int32)
            for name in ranked[:47]
        }
        assignments["other_efferent"] = np.asarray(
            sorted(index for name in ranked[47:] for index in signatures[name]),
            dtype=np.int32,
        )
        return self.build_readout_map(assignments)

    def matrix(
        self,
        *,
        normalized: bool = True,
        signed: bool = True,
        dtype: np.dtype | type = np.float32,
    ) -> sparse.csr_matrix:
        """Return postsynaptic-row CSR connectivity without a dense NxN array."""
        dtype = np.dtype(dtype)
        if dtype.kind != "f":
            raise ValueError("matrix dtype must be floating point")
        working_dtype = np.dtype(np.float32) if dtype.itemsize < 4 else dtype
        values = self.counts.astype(working_dtype, copy=True)
        if normalized:
            for start in range(0, self.n, 4096):
                stop = min(start + 4096, self.n)
                edge_start = int(self.indptr[start])
                edge_stop = int(self.indptr[stop])
                if edge_start == edge_stop:
                    continue
                denominators = np.maximum(self.row_synapses[start:stop], 1).astype(working_dtype)
                values[edge_start:edge_stop] /= np.repeat(
                    denominators, np.diff(self.indptr[start:stop + 1])
                )
        if signed:
            values *= self.sign[self.indices].astype(working_dtype, copy=False)
        if working_dtype != dtype:
            values = values.astype(dtype)
        matrix = sparse.csr_matrix(
            (values, np.asarray(self.indices), np.asarray(self.indptr)),
            shape=self.shape,
            copy=False,
        )
        # The acquisition script guarantees strictly increasing source indices
        # in every row.  Declaring that property prevents SciPy from attempting
        # an in-place sort of read-only memory maps.
        matrix.has_sorted_indices = True
        matrix.has_canonical_format = True
        return matrix

    def edge_arrays(
        self,
        *,
        effective: bool = False,
        normalized: bool = True,
        signed: bool = True,
        include_targets: bool = True,
    ) -> tuple[np.ndarray, ...]:
        """Return source, optional target, and count/effective-weight arrays.

        Materializing targets costs an additional int32[E] allocation, so code
        that understands CSR should use ``indptr`` and ``indices`` directly.
        """
        values = (
            self.matrix(normalized=normalized, signed=signed).data
            if effective else self.counts
        )
        if not include_targets:
            return self.indices, values
        targets = np.repeat(
            np.arange(self.n, dtype=np.int32), np.diff(self.indptr)
        )
        return self.indices, targets, values


def load_malecns(
    artifact_dir: str | Path | None = None, *, mmap: bool = True,
    verify: bool = False,
) -> MaleCNSGraph:
    return MaleCNSGraph.load(artifact_dir, mmap=mmap, verify=verify)
