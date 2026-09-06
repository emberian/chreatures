"""Compile sparse, inherited circuit variants without mutating their ancestor.

The compiler deliberately separates measured MaleCNS records from synthetic
developmental operations.  Its output uses the ordinary graph and neural-port
contracts, while a compact ledger identifies every cloned or edited edge.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .malecns import MaleCNSGraph
from .neural_ports import NeuralPortBundle

SCHEMA_VERSION = 1
GRAPH_FORMAT = "chreatures-derived-circuit-v1"
NATIVE_FORMAT = "metal-csr-v2"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_hash(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _file_hash(path)}


def _clean_name(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if (
        not value
        or len(value) > 96
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for c in value
        )
    ):
        raise ValueError(f"{field} must contain 1-96 safe characters")
    return value


@dataclass(frozen=True)
class CircuitBlueprint:
    """Validated, canonical description of inherited graph changes."""

    document: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> CircuitBlueprint:
        return cls.from_value(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> CircuitBlueprint:
        try:
            document = json.loads(_canonical_json(dict(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError("blueprint must be finite JSON data") from exc
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported circuit blueprint schema")
        _clean_name(document.get("name"), "name")
        parent = document.get("parent")
        if (
            not isinstance(parent, dict)
            or len(str(parent.get("graph_sha256", ""))) != 64
        ):
            raise ValueError("blueprint parent.graph_sha256 must pin an ancestor")
        if len(str(parent.get("port_spec_sha256", ""))) != 64:
            raise ValueError(
                "blueprint parent.port_spec_sha256 must pin port semantics"
            )
        modules = document.get("modules")
        if not isinstance(modules, list) or not modules:
            raise ValueError("blueprint needs at least one module")
        seen: set[str] = set()
        for module in modules:
            if not isinstance(module, dict):
                raise TypeError("module entries must be objects")
            name = _clean_name(module.get("name"), "module.name")
            if name in seen:
                raise ValueError(f"duplicate module name {name!r}")
            seen.add(name)
            copies = module.get("copies")
            if (
                isinstance(copies, bool)
                or not isinstance(copies, int)
                or not 1 <= copies <= 32
            ):
                raise ValueError("module copies must be an integer in 1..32")
            selector = module.get("selector")
            if not isinstance(selector, dict) or set(selector) != {
                "npz",
                "sha256",
                "fields",
            }:
                raise ValueError(
                    "module selector needs exactly npz, sha256, and fields"
                )
            if len(str(selector["sha256"])) != 64:
                raise ValueError("module selector sha256 is malformed")
            if not isinstance(selector["fields"], list) or not selector["fields"]:
                raise ValueError("module selector fields must be a nonempty list")
            boundary = module.get("boundary")
            if boundary not in ("internal", "incoming", "bidirectional"):
                raise ValueError(
                    "module boundary must be internal, incoming, or bidirectional"
                )
            ports = module.get("ports")
            if ports not in ("none", "inherit"):
                raise ValueError("module ports must be none or inherit")
        edits = document.get("edits", {})
        if not isinstance(edits, dict) or set(edits).difference(
            {"add", "remove", "reweight"}
        ):
            raise ValueError("edits may contain only add, remove, and reweight")
        for kind in ("add", "remove", "reweight"):
            if not isinstance(edits.get(kind, []), list):
                raise TypeError(f"edits.{kind} must be a list")
        return cls(document)

    @property
    def sha256(self) -> str:
        return _json_hash(self.document)


class DerivedCircuitGraph(MaleCNSGraph):
    """MaleCNSGraph-compatible loader requiring a derivation receipt."""

    def __init__(self, artifact_dir: str | Path, *, mmap: bool = True):
        super().__init__(artifact_dir, mmap=mmap)
        derived = self.manifest.get("derivation")
        if self.manifest.get("format") != GRAPH_FORMAT or not isinstance(derived, dict):
            raise ValueError("artifact is not a compiled circuit blueprint")
        if derived.get("blueprint_sha256") != _json_hash(derived.get("blueprint")):
            raise ValueError("derived graph blueprint receipt is inconsistent")

    def verify_artifacts(self) -> None:
        super().verify_artifacts()
        for filename, expected in self.manifest.get("derived_artifacts", {}).items():
            path = self.path / filename
            if path.stat().st_size != expected["bytes"]:
                raise ValueError(f"size mismatch for {filename}")
            observed = _file_hash(path)
            if observed != expected["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {filename}: {observed}")


def _module_ids(module: Mapping[str, Any], root: Path) -> np.ndarray:
    selector = module["selector"]
    path = (root / selector["npz"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("module selector path must remain below selector_root")
    if not path.is_file() or _file_hash(path) != selector["sha256"]:
        raise ValueError(f"module selector artifact differs from its receipt: {path}")
    values: list[np.ndarray] = []
    with np.load(path, allow_pickle=False) as data:
        for field in selector["fields"]:
            if field not in data.files:
                raise ValueError(f"module selector artifact lacks {field!r}")
            array = np.asarray(data[field])
            if array.ndim != 1 or array.dtype.kind not in "iu":
                raise ValueError(f"selector field {field!r} must be integer vector")
            values.append(array.astype(np.int64, copy=False))
    return np.unique(np.concatenate(values))


def _resolve_modules(
    graph: Any, blueprint: CircuitBlueprint, root: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], int]]:
    body_ids = np.asarray(graph.body_ids, dtype=np.int64)
    if len(body_ids) and (np.any(np.diff(body_ids) <= 0)):
        raise ValueError("ancestor body_ids must be strictly increasing")
    next_body = int(body_ids[-1]) + 1
    if next_body >= np.iinfo(np.int64).max:
        raise ValueError("no positive int64 body IDs remain for derived neurons")
    resolved: list[dict[str, Any]] = []
    references: dict[tuple[str, int, int], int] = {}
    next_index = int(graph.n)
    for module in blueprint.document["modules"]:
        selected_ids = _module_ids(module, root)
        local = np.searchsorted(body_ids, selected_ids)
        valid = (local < graph.n) & (
            body_ids[np.minimum(local, graph.n - 1)] == selected_ids
        )
        if not valid.all():
            missing = selected_ids[~valid][:8].tolist()
            raise ValueError(
                f"module {module['name']!r} references absent body IDs: {missing}"
            )
        local = local.astype(np.int32)
        copies: list[np.ndarray] = []
        synthetic_ids: list[np.ndarray] = []
        for copy_index in range(1, int(module["copies"]) + 1):
            duplicate = np.arange(next_index, next_index + len(local), dtype=np.int32)
            if next_body + len(local) > np.iinfo(np.int64).max:
                raise ValueError("derived body ID range overflows int64")
            new_ids = np.arange(next_body, next_body + len(local), dtype=np.int64)
            next_index += len(local)
            next_body += len(local)
            copies.append(duplicate)
            synthetic_ids.append(new_ids)
            for ancestral_id, index in zip(selected_ids, duplicate, strict=True):
                key = (str(module["name"]), copy_index, int(ancestral_id))
                references[key] = int(index)
        resolved.append(
            {
                "spec": module,
                "ancestor_indices": local,
                "ancestor_body_ids": selected_ids,
                "copy_indices": copies,
                "copy_body_ids": synthetic_ids,
            }
        )
    return resolved, references


def _reservoir(
    values: list[tuple[int, int, int]],
    item: tuple[int, int, int],
    seen: int,
    capacity: int,
    rng: np.random.Generator,
) -> None:
    if capacity <= 0:
        return
    if len(values) < capacity:
        values.append(item)
        return
    replacement = int(rng.integers(0, seen))
    if replacement < capacity:
        values[replacement] = item


def _cloned_edge_indices(graph: Any, resolved: dict[str, Any]) -> Any:
    """Yield source, target, count for the exact edges one module will clone."""
    spec = resolved["spec"]
    ancestors = resolved["ancestor_indices"]
    member = np.zeros(graph.n, dtype=bool)
    member[ancestors] = True
    for duplicate in resolved["copy_indices"]:
        copy_of = np.full(graph.n, -1, dtype=np.int32)
        copy_of[ancestors] = duplicate
        for ancestor_target, duplicate_target in zip(ancestors, duplicate, strict=True):
            start = int(graph.indptr[ancestor_target])
            stop = int(graph.indptr[ancestor_target + 1])
            sources = np.asarray(graph.indices[start:stop])
            counts = np.asarray(graph.counts[start:stop])
            keep = (
                member[sources]
                if spec["boundary"] == "internal"
                else np.ones(len(sources), dtype=bool)
            )
            for source, count in zip(sources[keep], counts[keep], strict=True):
                mapped_source = copy_of[source] if member[source] else source
                yield int(mapped_source), int(duplicate_target), int(count)
        if spec["boundary"] == "bidirectional":
            for target in range(graph.n):
                if member[target]:
                    continue
                start = int(graph.indptr[target])
                stop = int(graph.indptr[target + 1])
                sources = np.asarray(graph.indices[start:stop])
                counts = np.asarray(graph.counts[start:stop])
                for source, count in zip(
                    sources[member[sources]], counts[member[sources]], strict=True
                ):
                    yield int(copy_of[source]), target, int(count)


def _index_ref(
    index: int,
    graph: Any,
    reverse_copies: Mapping[int, tuple[str, int, int]],
) -> dict[str, Any]:
    if index < graph.n:
        return {"kind": "ancestor", "body_id": int(graph.body_ids[index])}
    module, copy_index, ancestral_body_id = reverse_copies[index]
    return {
        "kind": "copy",
        "module": module,
        "copy_index": copy_index,
        "body_id": ancestral_body_id,
    }


def materialize_module_variation(
    parent: Any,
    ports: NeuralPortBundle,
    *,
    name: str,
    template: Mapping[str, Any],
    bounds: Mapping[str, Any],
    seed: int,
    mutation_scale: float,
    selector_root: str | Path = ".",
    parent_port_sha256: str | None = None,
) -> CircuitBlueprint:
    """Materialize bounded variation as exact edits of cloned module edges.

    Selection is deterministic for the parent graph, template, seed, and scale.
    Only edges created by this duplication are eligible; ancestral-to-ancestral
    measurements are never edited by this operator.
    """
    _clean_name(name, "blueprint name")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("variation seed must be an unsigned 64-bit integer")
    scale = float(mutation_scale)
    if not np.isfinite(scale) or not 0 < scale <= 0.5:
        raise ValueError("variation mutation_scale must be in (0, 0.5]")
    required_bounds = {
        "maximum_removals",
        "maximum_reweights",
        "reweight_factor_min",
        "reweight_factor_max",
    }
    if set(bounds) != required_bounds:
        raise ValueError(f"variation bounds must contain {sorted(required_bounds)}")
    max_removals = int(bounds["maximum_removals"])
    max_reweights = int(bounds["maximum_reweights"])
    factor_min = float(bounds["reweight_factor_min"])
    factor_max = float(bounds["reweight_factor_max"])
    if (
        isinstance(bounds["maximum_removals"], bool)
        or isinstance(bounds["maximum_reweights"], bool)
        or not 0 <= max_removals <= 64
        or not 0 <= max_reweights <= 64
        or not 0.5 <= factor_min < 1 < factor_max <= 2
    ):
        raise ValueError("variation bounds are outside compiler limits")
    module = {
        "name": _clean_name(template.get("name"), "template.name"),
        "copies": 1,
        "boundary": template.get("boundary"),
        "ports": template.get("ports"),
        "selector": template.get("selector"),
    }
    parent_record = {
        "graph_sha256": parent.hash,
        "port_spec_sha256": ports.spec_hash,
    }
    if parent_port_sha256 is not None:
        parent_record["port_bundle_sha256"] = parent_port_sha256
    base = CircuitBlueprint.from_value(
        {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "parent": parent_record,
            "modules": [module],
            "edits": {"add": [], "remove": [], "reweight": []},
        }
    )
    resolved, references = _resolve_modules(parent, base, Path(selector_root).resolve())
    reverse_copies = {index: key for key, index in references.items()}
    rng = np.random.Generator(np.random.PCG64(seed))
    removals = int(rng.binomial(max_removals, scale))
    reweights = int(rng.binomial(max_reweights, scale))
    removal_sample: list[tuple[int, int, int]] = []
    reweight_sample: list[tuple[int, int, int]] = []
    seen_removal = 0
    seen_reweight = 0
    for edge in _cloned_edge_indices(parent, resolved[0]):
        seen_removal += 1
        _reservoir(removal_sample, edge, seen_removal, removals, rng)
        count = edge[2]
        low = max(1, int(np.ceil(count * factor_min)))
        high = min(np.iinfo(np.uint32).max, int(np.floor(count * factor_max)))
        if low < count or high > count:
            seen_reweight += 1
            _reservoir(
                reweight_sample,
                edge,
                seen_reweight,
                reweights + removals,
                rng,
            )
    removed_pairs = {(source, target) for source, target, _ in removal_sample}
    reweight_sample = [
        edge for edge in reweight_sample if (edge[0], edge[1]) not in removed_pairs
    ][:reweights]

    def refs(edge: tuple[int, int, int]) -> dict[str, Any]:
        source, target, _ = edge
        return {
            "source": _index_ref(source, parent, reverse_copies),
            "target": _index_ref(target, parent, reverse_copies),
        }

    remove_edits = [refs(edge) for edge in removal_sample]
    reweight_edits = []
    for edge in reweight_sample:
        count = edge[2]
        low = max(1, int(np.ceil(count * factor_min)))
        high = min(np.iinfo(np.uint32).max, int(np.floor(count * factor_max)))
        factor = float(np.clip(np.exp(rng.normal(0, scale)), factor_min, factor_max))
        changed = int(np.clip(round(count * factor), low, high))
        if changed == count:
            changed = low if low < count else high
        reweight_edits.append({**refs(edge), "count": changed})
    document = base.document
    document["description"] = (
        "Inherited typed module duplication with deterministic bounded variation "
        "restricted to the newly cloned edges."
    )
    document["model_boundary"] = {
        "measured": "Ancestral annotations and copied edge directions/counts.",
        "synthetic": "Duplication and every removed or reweighted cloned edge.",
    }
    document["edits"] = {
        "add": [],
        "remove": remove_edits,
        "reweight": reweight_edits,
    }
    document["variation"] = {
        "operator": "bounded-cloned-edge-variation-v1",
        "seed": seed,
        "mutation_scale": scale,
        "bounds": dict(bounds),
        "eligible_cloned_edges": seen_removal,
        "eligible_reweight_edges": seen_reweight,
        "removed_edges": len(remove_edits),
        "reweighted_edges": len(reweight_edits),
    }
    return CircuitBlueprint.from_value(document)


def _clone_edges(
    graph: Any, modules: list[dict[str, Any]], n: int
) -> tuple[sparse.csr_matrix, list[dict[str, Any]], set[int]]:
    base_indptr = np.empty(n + 1, dtype=np.int64)
    base_indptr[: graph.n + 1] = graph.indptr
    base_indptr[graph.n + 1 :] = graph.edge_count
    base = sparse.csr_matrix(
        (np.asarray(graph.counts), np.asarray(graph.indices), base_indptr),
        shape=(n, n),
        copy=False,
    )
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    ledger: list[dict[str, Any]] = []
    normalized_rows: set[int] = set()
    for resolved in modules:
        spec = resolved["spec"]
        ancestors = resolved["ancestor_indices"]
        member = np.zeros(graph.n, dtype=bool)
        member[ancestors] = True
        for copy_number, duplicate in enumerate(resolved["copy_indices"], 1):
            copy_of = np.full(graph.n, -1, dtype=np.int32)
            copy_of[ancestors] = duplicate
            copied_by_kind = Counter()
            # Duplicate target rows. Sources inside the module map to their
            # homologous duplicate; external sources remain shared inputs.
            for ancestor_target, duplicate_target in zip(
                ancestors, duplicate, strict=True
            ):
                start, stop = (
                    int(graph.indptr[ancestor_target]),
                    int(graph.indptr[ancestor_target + 1]),
                )
                sources = np.asarray(graph.indices[start:stop])
                if spec["boundary"] == "internal":
                    keep = member[sources]
                else:
                    keep = np.ones(len(sources), dtype=bool)
                sources = sources[keep]
                counts = np.asarray(graph.counts[start:stop])[keep]
                mapped = np.where(member[sources], copy_of[sources], sources).astype(
                    np.int32
                )
                source_parts.append(mapped)
                target_parts.append(
                    np.full(len(mapped), duplicate_target, dtype=np.int32)
                )
                count_parts.append(counts)
                copied_by_kind["internal"] += int(member[sources].sum())
                copied_by_kind["incoming"] += int((~member[sources]).sum())
            if spec["boundary"] == "bidirectional":
                # Copy only outgoing edges whose targets are outside the module;
                # internal homologous edges were already emitted above.
                for target in range(graph.n):
                    if member[target]:
                        continue
                    start, stop = (
                        int(graph.indptr[target]),
                        int(graph.indptr[target + 1]),
                    )
                    sources = np.asarray(graph.indices[start:stop])
                    keep = member[sources]
                    if not keep.any():
                        continue
                    mapped = copy_of[sources[keep]]
                    source_parts.append(mapped)
                    target_parts.append(np.full(len(mapped), target, dtype=np.int32))
                    count_parts.append(np.asarray(graph.counts[start:stop])[keep])
                    copied_by_kind["outgoing"] += int(keep.sum())
                    normalized_rows.add(target)
            ledger.append(
                {
                    "operation": "duplicate_module",
                    "module": spec["name"],
                    "copy_index": copy_number,
                    "boundary": spec["boundary"],
                    "neurons": len(ancestors),
                    "edges": dict(sorted(copied_by_kind.items())),
                    "basis": "copied measured direction and integer synapse count",
                    "evidence": "synthetic developmental hypothesis",
                }
            )
    if source_parts:
        source = np.concatenate(source_parts)
        target = np.concatenate(target_parts)
        counts = np.concatenate(count_parts).astype(np.uint32, copy=False)
        extra = sparse.coo_matrix((counts, (target, source)), shape=(n, n)).tocsr()
        if extra.nnz != len(counts):
            raise ValueError("module duplication produced colliding cloned edges")
        result = base + extra
    else:
        result = base.copy()
    result.sum_duplicates()
    result.sort_indices()
    return result.astype(np.uint32, copy=False), ledger, normalized_rows


def _reference_index(
    ref: Mapping[str, Any], graph: Any, references: Mapping[tuple[str, int, int], int]
) -> int:
    kind = ref.get("kind")
    body_id = int(ref.get("body_id"))
    if kind == "ancestor":
        where = np.searchsorted(graph.body_ids, body_id)
        if where >= graph.n or int(graph.body_ids[where]) != body_id:
            raise ValueError(f"unknown ancestral body ID {body_id}")
        return int(where)
    if kind == "copy":
        key = (str(ref.get("module")), int(ref.get("copy_index")), body_id)
        if key not in references:
            raise ValueError(f"unknown duplicate neuron reference {key}")
        return references[key]
    raise ValueError("edge reference kind must be ancestor or copy")


def _edge_position(matrix: sparse.csr_matrix, source: int, target: int) -> int | None:
    start, stop = int(matrix.indptr[target]), int(matrix.indptr[target + 1])
    position = int(np.searchsorted(matrix.indices[start:stop], source)) + start
    return (
        position
        if position < stop and int(matrix.indices[position]) == source
        else None
    )


def _verify_ancestral_induced(graph: Any, matrix: sparse.csr_matrix) -> None:
    """Prove module cloning did not alter any measured ancestral edge."""
    for target in range(graph.n):
        parent_start, parent_stop = (
            int(graph.indptr[target]),
            int(graph.indptr[target + 1]),
        )
        start, stop = int(matrix.indptr[target]), int(matrix.indptr[target + 1])
        keep = matrix.indices[start:stop] < graph.n
        if not np.array_equal(
            matrix.indices[start:stop][keep], graph.indices[parent_start:parent_stop]
        ):
            raise RuntimeError(f"ancestral edge sources changed at target row {target}")
        if not np.array_equal(
            matrix.data[start:stop][keep], graph.counts[parent_start:parent_stop]
        ):
            raise RuntimeError(f"ancestral edge counts changed at target row {target}")


def _apply_edits(
    matrix: sparse.csr_matrix,
    graph: Any,
    references: Mapping[tuple[str, int, int], int],
    edits: Mapping[str, Any],
    ledger: list[dict[str, Any]],
    normalized_rows: set[int],
) -> sparse.csr_matrix:
    additions: list[tuple[int, int, int]] = []
    touched: set[tuple[int, int]] = set()
    for item in edits.get("add", []):
        source = _reference_index(item["source"], graph, references)
        target = _reference_index(item["target"], graph, references)
        count = item.get("count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= np.iinfo(np.uint32).max
        ):
            raise ValueError("added edge count must be a positive uint32 integer")
        if (source, target) in touched:
            raise ValueError("an edge may appear in only one explicit edit")
        touched.add((source, target))
        if _edge_position(matrix, source, target) is not None:
            raise ValueError("add operation names an edge that already exists")
        additions.append((source, target, count))
        normalized_rows.add(target)
        ledger.append(
            {
                "operation": "add",
                "source_index": source,
                "target_index": target,
                "previous_count": None,
                "new_count": count,
                "evidence": "synthetic developmental hypothesis",
            }
        )
    if additions:
        source, target, count = map(np.asarray, zip(*additions, strict=True))
        matrix = (
            matrix
            + sparse.coo_matrix(
                (count.astype(np.uint32), (target, source)), shape=matrix.shape
            ).tocsr()
        )
        matrix.sort_indices()
    remove_positions: list[int] = []
    for item in edits.get("remove", []):
        source = _reference_index(item["source"], graph, references)
        target = _reference_index(item["target"], graph, references)
        position = _edge_position(matrix, source, target)
        if position is None:
            raise ValueError("remove operation names an absent edge")
        if (source, target) in touched:
            raise ValueError("an edge may appear in only one explicit edit")
        touched.add((source, target))
        old = int(matrix.data[position])
        matrix.data[position] = 0
        remove_positions.append(position)
        normalized_rows.add(target)
        ledger.append(
            {
                "operation": "remove",
                "source_index": source,
                "target_index": target,
                "previous_count": old,
                "new_count": None,
                "evidence": "synthetic developmental hypothesis",
            }
        )
    if remove_positions:
        matrix.eliminate_zeros()
        matrix.sort_indices()
    for item in edits.get("reweight", []):
        source = _reference_index(item["source"], graph, references)
        target = _reference_index(item["target"], graph, references)
        position = _edge_position(matrix, source, target)
        if position is None:
            raise ValueError("reweight operation names an absent edge")
        if (source, target) in touched:
            raise ValueError("an edge may appear in only one explicit edit")
        touched.add((source, target))
        count = item.get("count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= np.iinfo(np.uint32).max
        ):
            raise ValueError("reweighted edge count must be a positive uint32 integer")
        old = int(matrix.data[position])
        matrix.data[position] = count
        normalized_rows.add(target)
        ledger.append(
            {
                "operation": "reweight",
                "source_index": source,
                "target_index": target,
                "previous_count": old,
                "new_count": count,
                "evidence": "synthetic developmental hypothesis",
            }
        )
    return matrix


def _metadata(graph: Any, modules: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    duplicate_sources = np.concatenate(
        [
            module["ancestor_indices"]
            for module in modules
            for _ in module["copy_indices"]
        ]
    ).astype(np.int32)
    duplicate_body_ids = np.concatenate(
        [ids for module in modules for ids in module["copy_body_ids"]]
    ).astype(np.int64)
    module_names = np.concatenate(
        [
            np.full(len(module["ancestor_indices"]), module["spec"]["name"])
            for module in modules
            for _ in module["copy_indices"]
        ]
    )
    copy_numbers = np.concatenate(
        [
            np.full(len(module["ancestor_indices"]), number, dtype=np.int16)
            for module in modules
            for number in range(1, len(module["copy_indices"]) + 1)
        ]
    )
    metadata: dict[str, np.ndarray] = {}
    for field in graph.metadata_fields:
        parent = np.asarray(getattr(graph, field))
        if field == "body_ids":
            duplicate = duplicate_body_ids
        elif field == "ids":
            duplicate = np.asarray(
                [
                    f"derived:{module}:{copy}:{ancestor}"
                    for module, copy, ancestor in zip(
                        module_names,
                        copy_numbers,
                        graph.body_ids[duplicate_sources],
                        strict=True,
                    )
                ]
            )
        else:
            duplicate = parent[duplicate_sources]
        metadata[field] = np.concatenate((parent, duplicate))

    def inherited(field: str, default: np.ndarray) -> np.ndarray:
        if field in graph.metadata_fields:
            return np.asarray(getattr(graph, field))
        return default

    root_indices = inherited(
        "ancestral_indices", np.arange(graph.n, dtype=np.int32)
    ).astype(np.int32, copy=False)
    root_body_ids = inherited(
        "ancestral_body_ids", np.asarray(graph.body_ids, dtype=np.int64)
    ).astype(np.int64, copy=False)
    birth_parent_indices = inherited(
        "birth_parent_indices", np.arange(graph.n, dtype=np.int32)
    ).astype(np.int32, copy=False)
    birth_parent_body_ids = inherited(
        "birth_parent_body_ids", np.asarray(graph.body_ids, dtype=np.int64)
    ).astype(np.int64, copy=False)
    parent_origin = inherited("origin", np.full(graph.n, "measured_ancestor")).astype(
        str, copy=False
    )
    parent_module = inherited("module", np.full(graph.n, "")).astype(str, copy=False)
    parent_copy = inherited("copy_index", np.zeros(graph.n, dtype=np.int16)).astype(
        np.int16, copy=False
    )
    parent_depth = inherited("lineage_depth", np.zeros(graph.n, dtype=np.int16)).astype(
        np.int16, copy=False
    )
    metadata["ancestral_body_ids"] = np.concatenate(
        (root_body_ids, root_body_ids[duplicate_sources])
    ).astype(np.int64)
    metadata["ancestral_indices"] = np.concatenate(
        (root_indices, root_indices[duplicate_sources])
    )
    metadata["birth_parent_body_ids"] = np.concatenate(
        (birth_parent_body_ids, np.asarray(graph.body_ids)[duplicate_sources])
    ).astype(np.int64)
    metadata["birth_parent_indices"] = np.concatenate(
        (birth_parent_indices, duplicate_sources)
    ).astype(np.int32)
    metadata["origin"] = np.concatenate(
        (
            parent_origin,
            np.full(len(duplicate_sources), "synthetic_duplicate"),
        )
    )
    metadata["module"] = np.concatenate((parent_module, module_names))
    metadata["copy_index"] = np.concatenate((parent_copy, copy_numbers))
    metadata["lineage_depth"] = np.concatenate(
        (parent_depth, parent_depth[duplicate_sources] + 1)
    )
    return metadata


def _inherit_ports(
    parent: NeuralPortBundle,
    graph_hash: str,
    parent_n: int,
    modules: list[dict[str, Any]],
    n: int,
    blueprint_hash: str,
) -> NeuralPortBundle:
    parent_inputs = parent.input_map.tocsr()
    readouts = parent.readout_map.tocsc(copy=True)
    inherited_sources: list[int] = []
    readout_lineages: Counter[int] = Counter()
    for module in modules:
        if module["spec"]["ports"] != "inherit":
            inherited_sources.extend([-1] * sum(map(len, module["copy_indices"])))
            continue
        ancestors = module["ancestor_indices"]
        for _ in module["copy_indices"]:
            inherited_sources.extend(ancestors.tolist())
            readout_lineages.update(map(int, ancestors))
    duplicate_n = n - parent_n
    inherited_array = np.asarray(inherited_sources, dtype=np.int32)
    inheriting = np.flatnonzero(inherited_array >= 0).astype(np.int32)
    inheritance = sparse.csr_matrix(
        (
            np.ones(len(inheriting), dtype=np.float32),
            (inheriting, inherited_array[inheriting]),
        ),
        shape=(duplicate_n, parent_n),
    )
    input_extra = inheritance @ parent_inputs
    inputs = sparse.vstack((parent_inputs, input_extra), format="csr", dtype=np.float32)
    # Preserve each readout's aggregate coefficient by splitting a parent's
    # coefficient equally across itself and every port-inheriting descendant.
    columns: list[sparse.csc_matrix] = []
    for ancestor in range(parent_n):
        column = readouts[:, ancestor]
        divisor = 1 + readout_lineages.get(ancestor, 0)
        columns.append(column / divisor)
    for source in inherited_sources:
        if source < 0:
            columns.append(sparse.csc_matrix((readouts.shape[0], 1), dtype=np.float32))
        else:
            columns.append(readouts[:, source] / (1 + readout_lineages[source]))
    derived_readouts = sparse.hstack(columns, format="csr", dtype=np.float32)
    spec = json.loads(_canonical_json(parent.spec))
    spec["graph"] = dict(spec.get("graph", {}))
    spec["graph"]["dataset_hash"] = graph_hash
    spec["graph"]["derived_from"] = parent.graph_hash
    spec["graph"]["blueprint_sha256"] = blueprint_hash
    input_ports = json.loads(_canonical_json(parent.input_ports))
    readout_ports = json.loads(_canonical_json(parent.readout_ports))
    return NeuralPortBundle(
        spec=spec,
        graph_hash=graph_hash,
        input_names=list(parent.input_names),
        input_map=inputs,
        readout_names=list(parent.readout_names),
        readout_map=derived_readouts,
        input_ports=input_ports,
        readout_ports=readout_ports,
    )


def pack_native_csr(
    graph: Any, ports: NeuralPortBundle, output: str | Path
) -> dict[str, Any]:
    """Write the existing dimension-generic native CSR wire format."""
    recurrent = graph.matrix(normalized=True, signed=True).tocsr()
    inputs = ports.input_map.tocsr().astype(np.float32, copy=False)
    readouts = ports.readout_map.tocsr().astype(np.float32, copy=False)
    if recurrent.shape != (graph.n, graph.n):
        raise ValueError("recurrent matrix shape differs from graph")
    if inputs.shape[0] != graph.n or readouts.shape[1] != graph.n:
        raise ValueError("port maps differ from graph size")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as stream:
        stream.write(
            struct.pack(
                "<IIIIII",
                0x4D424732,
                graph.n,
                graph.edge_count,
                inputs.nnz,
                readouts.nnz,
                0,
            )
        )
        for array, dtype in (
            (recurrent.indptr, "<u4"),
            (recurrent.indices, "<u4"),
            (recurrent.data, "<f4"),
            (inputs.indptr, "<u4"),
            (inputs.indices, "<u4"),
            (inputs.data, "<f4"),
            (readouts.indptr, "<u4"),
            (readouts.indices, "<u4"),
            (readouts.data, "<f4"),
        ):
            stream.write(np.asarray(array, dtype=dtype).tobytes())
    os.replace(temporary, output)
    return {
        "format": NATIVE_FORMAT,
        "path": str(output),
        **_receipt(output),
        "graph_sha256": graph.hash,
        "port_spec_sha256": ports.spec_hash,
        "neurons": graph.n,
        "edges": graph.edge_count,
        "inputs": len(ports.input_names),
        "readouts": len(ports.readout_names),
    }


def compile_blueprint(
    parent: MaleCNSGraph,
    ports: NeuralPortBundle,
    blueprint: CircuitBlueprint,
    output_directory: str | Path,
    *,
    selector_root: str | Path = ".",
    parent_port_sha256: str | None = None,
) -> dict[str, Any]:
    """Compile one immutable parent graph into a separate derived artifact."""
    document = blueprint.document
    if parent.hash != document["parent"]["graph_sha256"]:
        raise ValueError("blueprint belongs to a different ancestral graph")
    if (
        ports.graph_hash != parent.hash
        or ports.spec_hash != document["parent"]["port_spec_sha256"]
    ):
        raise ValueError("blueprint belongs to a different ancestral port interface")
    expected_port = document["parent"].get("port_bundle_sha256")
    if expected_port and parent_port_sha256 != expected_port:
        raise ValueError("ancestral port bundle checksum differs from blueprint")
    # Verify the immutable parent at both ends of the build. This reads the
    # canonical files but never opens them writable.
    parent.verify_artifacts()
    output = Path(output_directory).expanduser().resolve()
    if output == Path(parent.path).resolve():
        raise ValueError("derived output must differ from immutable ancestor")
    if output.exists():
        raise FileExistsError(f"derived output already exists: {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")
    partial.mkdir(parents=True)
    try:
        modules, references = _resolve_modules(
            parent, blueprint, Path(selector_root).resolve()
        )
        n = parent.n + sum(
            len(values) for module in modules for values in module["copy_indices"]
        )
        matrix, ledger, normalized_rows = _clone_edges(parent, modules, n)
        _verify_ancestral_induced(parent, matrix)
        matrix = _apply_edits(
            matrix,
            parent,
            references,
            document.get("edits", {}),
            ledger,
            normalized_rows,
        )
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.data.dtype != np.uint32 or np.any(matrix.data == 0):
            raise RuntimeError("compiled counts are not positive uint32 integers")
        row_synapses = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.uint64)
        np.save(
            partial / "indptr.npy",
            matrix.indptr.astype(np.int64, copy=False),
            allow_pickle=False,
        )
        np.save(
            partial / "indices.npy",
            matrix.indices.astype(np.int32, copy=False),
            allow_pickle=False,
        )
        np.save(partial / "counts.npy", matrix.data, allow_pickle=False)
        np.save(partial / "row_synapses.npy", row_synapses, allow_pickle=False)
        np.savez_compressed(partial / "neurons.npz", **_metadata(parent, modules))
        (partial / "blueprint.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (partial / "edge-provenance.json").write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Every surviving synthetic edge receives a compact row in this ledger.
        # Code 1 is a cloned measured edge, 2 an explicit addition, and 3 an
        # explicit reweight. Removed edges remain in the JSON operation ledger.
        metadata = _metadata(parent, modules)
        targets = np.repeat(np.arange(n, dtype=np.int32), np.diff(matrix.indptr))
        sources = np.asarray(matrix.indices, dtype=np.int32)
        provenance_mask = (sources >= parent.n) | (targets >= parent.n)
        explicit_pairs: dict[tuple[int, int], int] = {}
        for operation in ledger:
            if operation["operation"] in {"add", "reweight"}:
                pair = (int(operation["source_index"]), int(operation["target_index"]))
                explicit_pairs[pair] = 2 if operation["operation"] == "add" else 3
        if explicit_pairs:
            for source, target in explicit_pairs:
                position = _edge_position(matrix, source, target)
                if position is not None:
                    provenance_mask[position] = True
        positions = np.flatnonzero(provenance_mask)
        provenance_codes = np.ones(len(positions), dtype=np.uint8)
        pair_positions = {
            (int(sources[position]), int(targets[position])): offset
            for offset, position in enumerate(positions)
        }
        for pair, code in explicit_pairs.items():
            if pair in pair_positions:
                provenance_codes[pair_positions[pair]] = code
        np.savez_compressed(
            partial / "derived-edges.npz",
            source_indices=sources[positions],
            target_indices=targets[positions],
            ancestral_source_indices=metadata["ancestral_indices"][sources[positions]],
            ancestral_target_indices=metadata["ancestral_indices"][targets[positions]],
            counts=np.asarray(matrix.data)[positions],
            basis_codes=provenance_codes,
            basis_names=np.asarray(
                [
                    "unused",
                    "cloned_measured_edge",
                    "explicit_synthetic_addition",
                    "explicit_synthetic_reweight",
                ]
            ),
        )
        artifact_names = (
            "indptr.npy",
            "indices.npy",
            "counts.npy",
            "row_synapses.npy",
            "neurons.npz",
            "blueprint.json",
            "edge-provenance.json",
            "derived-edges.npz",
        )
        artifacts = {name: _receipt(partial / name) for name in artifact_names}
        digest = hashlib.sha256()
        for name in artifact_names:
            digest.update(name.encode())
            digest.update(artifacts[name]["sha256"].encode())
        graph_hash = digest.hexdigest()
        ancestor_mask = matrix.indices < parent.n
        ancestral_edge_mask = ancestor_mask & (targets < parent.n)
        affected_ancestral_rows = np.asarray(
            sorted(normalized_rows & set(range(parent.n))), dtype=np.int32
        )

        def categorical_counts(field: str) -> dict[str, int]:
            return dict(
                sorted(
                    Counter(
                        str(value) or "unavailable" for value in metadata[field]
                    ).items()
                )
            )

        superclasses = np.asarray(metadata["superclasses"])
        afferent = np.asarray(
            [
                ("_sensory" in value) or value.startswith("sensory_")
                for value in superclasses
            ]
        )
        efferent = np.asarray(
            [
                ("_motor" in value)
                or ("_efferent" in value)
                or value.startswith("efferent_")
                for value in superclasses
            ]
        )
        manifest = json.loads(_canonical_json(parent.manifest))
        graph_ancestry = list(manifest.get("graph_ancestry", []))
        if not graph_ancestry:
            graph_ancestry.append(
                {
                    "graph_sha256": parent.hash,
                    "kind": "measured_root",
                    "blueprint_sha256": None,
                }
            )
        graph_ancestry.append(
            {
                "graph_sha256": graph_hash,
                "parent_graph_sha256": parent.hash,
                "kind": "circuit_blueprint",
                "blueprint_sha256": blueprint.sha256,
            }
        )
        manifest.update(
            {
                "schema_version": max(3, int(manifest.get("schema_version", 1))),
                "format": GRAPH_FORMAT,
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset_hash": graph_hash,
                "artifacts": artifacts,
                "counts": {
                    **manifest["counts"],
                    "neurons": n,
                    "edges": int(matrix.nnz),
                    "synapses": int(matrix.data.sum(dtype=np.uint64)),
                    "afferent_neurons": int(afferent.sum()),
                    "efferent_neurons": int(efferent.sum()),
                    "classes": categorical_counts("classes"),
                    "superclasses": categorical_counts("superclasses"),
                    "effective_transmitters": categorical_counts("effective_nt"),
                    "nt_basis": categorical_counts("nt_basis"),
                },
                "source_graph": {
                    "dataset_hash": parent.hash,
                    "artifacts": parent.manifest["artifacts"],
                    "manifest_sha256": _file_hash(Path(parent.path) / "manifest.json"),
                    "immutable": True,
                },
                "graph_ancestry": graph_ancestry,
                "derivation": {
                    "kind": "circuit_blueprint",
                    "blueprint": document,
                    "blueprint_sha256": blueprint.sha256,
                    "model_boundary": (
                        "Measured ancestral IDs and annotations are retained; ancestral edge directions "
                        "and counts are retained except where the explicit edit ledger says otherwise. "
                        "Duplicated neurons and cloned/edited connections are synthetic developmental hypotheses."
                    ),
                    "ancestral_neurons": parent.n,
                    "derived_neurons": n - parent.n,
                    "ancestral_to_ancestral_edges": int(ancestral_edge_mask.sum()),
                    "affected_ancestral_normalization_rows": len(
                        affected_ancestral_rows
                    ),
                    "affected_ancestral_normalization_indices_sha256": hashlib.sha256(
                        affected_ancestral_rows.tobytes()
                    ).hexdigest(),
                    "operations": ledger,
                },
            }
        )
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        graph = DerivedCircuitGraph(partial, mmap=True)
        derived_ports = _inherit_ports(
            ports, graph_hash, parent.n, modules, n, blueprint.sha256
        )
        port_receipt = derived_ports.save(partial / "ports.npz")
        native_receipt = pack_native_csr(
            graph, derived_ports, partial / "native-csr-v2.bin"
        )
        # Receipts describe the durable path after the atomic directory rename,
        # never the temporary construction path.
        port_receipt["path"] = str(output / "ports.npz")
        native_receipt["path"] = str(output / "native-csr-v2.bin")
        native_manifest = {
            "schema_version": 1,
            **native_receipt,
            "port_bundle_sha256": port_receipt["sha256"],
        }
        (partial / "native-csr-v2.manifest.json").write_text(
            json.dumps(native_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["derived_artifacts"] = {
            "ports.npz": _receipt(partial / "ports.npz"),
            "native-csr-v2.bin": _receipt(partial / "native-csr-v2.bin"),
            "native-csr-v2.manifest.json": _receipt(
                partial / "native-csr-v2.manifest.json"
            ),
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        parent.verify_artifacts()
        os.replace(partial, output)
        return {
            "path": str(output),
            "graph_sha256": graph_hash,
            "blueprint_sha256": blueprint.sha256,
            "parent_graph_sha256": parent.hash,
            "neurons": n,
            "derived_neurons": n - parent.n,
            "edges": int(matrix.nnz),
            "synapses": int(matrix.data.sum(dtype=np.uint64)),
            "affected_ancestral_normalization_rows": len(affected_ancestral_rows),
            "ports": {**port_receipt, "spec_sha256": derived_ports.spec_hash},
            "native": native_receipt,
        }
    except Exception as exc:
        # Keep the partial build and a small failure receipt for diagnosis;
        # never alter the source directory.
        (partial / "FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        raise
