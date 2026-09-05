"""Sparse, degree-matched controls for curated connectome graphs.

The MaleCNS CSR convention is postsynaptic rows with presynaptic indices.
Swapping the two presynaptic endpoints of equal-weight edges preserves every
neuron's directed degree and weighted strength.  Restricting swaps to equal
source-transmitter, source-category, and target-category strata additionally
preserves those annotation mixing statistics.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


Progress = Callable[[str], None]


@dataclass(frozen=True)
class MatchedRewireSpec:
    """Versioned definition of a directed matched-rewiring control."""

    name: str = "matched-rewire-v1"
    seed: int = 20260905
    category_field: str = "superclasses"
    transmitter_field: str = "effective_nt"
    equal_synapse_count: bool = True
    passes: int = 1
    duplicate_policy: str = "reject"
    self_loop_policy: str = "source"
    schema_version: int = 1

    def validate(self) -> None:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        if not self.name or any(char not in allowed for char in self.name):
            raise ValueError(
                "control name must contain only letters, digits, dot, underscore, or dash"
            )
        if self.seed < 0 or self.seed >= 2**64:
            raise ValueError("seed must be in [0, 2**64)")
        if self.passes != 1:
            raise ValueError("matched-rewire-v1 currently requires exactly one pass")
        if not self.equal_synapse_count:
            raise ValueError("matched-rewire-v1 requires equal synapse counts")
        if self.duplicate_policy != "reject":
            raise ValueError("the curated simple-edge graph requires duplicate rejection")
        if self.self_loop_policy != "source":
            raise ValueError("self_loop_policy must be 'source'")

    def document(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "algorithm": "directed presynaptic-endpoint double swaps",
            "strata": [
                "exact synapse count",
                f"source {self.transmitter_field}",
                f"source {self.category_field}",
                f"target {self.category_field}",
            ],
            "semantics": (
                "For u->v and x->y in one stratum, propose x->v and u->y. "
                "Each edge participates in at most one proposal."
            ),
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.document())


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _categorical_codes(values: np.ndarray) -> tuple[list[str], np.ndarray]:
    names, inverse = np.unique(values.astype(str), return_inverse=True)
    return names.tolist(), inverse.astype(np.int32, copy=False)


def _targets(indptr: np.ndarray, path: Path | None = None) -> np.ndarray:
    edge_count = int(indptr[-1])
    if path is None:
        result = np.empty(edge_count, dtype=np.int32)
    else:
        result = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.int32, shape=(edge_count,)
        )
    for row_start in range(0, len(indptr) - 1, 4096):
        row_stop = min(row_start + 4096, len(indptr) - 1)
        edge_start = int(indptr[row_start])
        edge_stop = int(indptr[row_stop])
        result[edge_start:edge_stop] = np.repeat(
            np.arange(row_start, row_stop, dtype=np.int32),
            np.diff(indptr[row_start : row_stop + 1]),
        )
    if isinstance(result, np.memmap):
        result.flush()
    return result


def _work_array(path: Path | None, dtype: np.dtype, length: int) -> np.ndarray:
    if path is None:
        return np.empty(length, dtype=dtype)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(length,))


def _stratum_keys(
    sources: np.ndarray,
    targets: np.ndarray,
    counts: np.ndarray,
    category_codes: np.ndarray,
    transmitter_codes: np.ndarray,
    category_count: int,
    transmitter_count: int,
    output: np.ndarray,
) -> None:
    for start in range(0, len(sources), 2_000_000):
        stop = min(start + 2_000_000, len(sources))
        pre = np.asarray(sources[start:stop])
        post = np.asarray(targets[start:stop])
        key = np.asarray(counts[start:stop], dtype=np.uint64)
        key = key * np.uint64(transmitter_count) + transmitter_codes[pre]
        key = key * np.uint64(category_count) + category_codes[pre]
        key = key * np.uint64(category_count) + category_codes[post]
        output[start:stop] = key
    if isinstance(output, np.memmap):
        output.flush()


def _reject_duplicate_pairs(
    original: np.ndarray,
    output: np.ndarray,
    partner: np.ndarray,
    indptr: np.ndarray,
) -> tuple[int, int]:
    """Revert proposed pairs until each CSR row is a unique source set."""
    rejected_pairs = 0
    passes = 0
    while True:
        passes += 1
        rejected_this_pass = 0
        for row in range(len(indptr) - 1):
            start, stop = int(indptr[row]), int(indptr[row + 1])
            if stop - start < 2:
                continue
            values = np.asarray(output[start:stop])
            ordered = np.sort(values)
            repeated = ordered[1:][ordered[1:] == ordered[:-1]]
            if not len(repeated):
                continue
            repeated = np.unique(repeated)
            active = np.flatnonzero(
                (np.asarray(partner[start:stop]) >= 0)
                & np.isin(values, repeated, assume_unique=False)
            )
            for local in active:
                position = start + int(local)
                other = int(partner[position])
                if other < 0:
                    continue
                output[position] = original[position]
                output[other] = original[other]
                partner[position] = -1
                partner[other] = -1
                rejected_pairs += 1
                rejected_this_pass += 1
        if rejected_this_pass == 0:
            return rejected_pairs, passes


def _rewire_sources(
    original: np.ndarray,
    counts: np.ndarray,
    indptr: np.ndarray,
    targets: np.ndarray,
    category_codes: np.ndarray,
    transmitter_codes: np.ndarray,
    *,
    seed: int,
    reject_new_self_loops: bool,
    output: np.ndarray,
    scratch: Path | None = None,
    progress: Progress = lambda _message: None,
) -> dict[str, Any]:
    """Apply one disjoint matched double-swap pass to edge source indices."""
    edge_count = len(original)
    if not (len(counts) == len(targets) == len(output) == edge_count):
        raise ValueError("edge arrays have inconsistent lengths")
    np.copyto(output, original)
    category_count = int(category_codes.max(initial=-1)) + 1
    transmitter_count = int(transmitter_codes.max(initial=-1)) + 1
    key_path = None if scratch is None else scratch / "strata.npy"
    partner_path = None if scratch is None else scratch / "partners.npy"
    keys = _work_array(key_path, np.dtype(np.uint64), edge_count)
    partner = _work_array(partner_path, np.dtype(np.int32), edge_count)
    partner.fill(-1)
    _stratum_keys(
        original, targets, counts, category_codes, transmitter_codes,
        category_count, transmitter_count, keys,
    )
    progress("sorting edges into exact weight/transmitter/category strata")
    order = np.argsort(keys, kind="stable")
    sorted_keys = np.asarray(keys[order])
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, edge_count]
    occupied_strata = len(starts)
    rng = np.random.default_rng(seed)
    attempted_pairs = 0
    singleton_edges = 0
    same_source_pairs = 0
    same_target_pairs = 0
    self_loop_rejections = 0
    progress(f"proposing swaps inside {occupied_strata:,} occupied strata")
    for start, stop in zip(starts, stops, strict=True):
        segment = order[int(start) : int(stop)]
        rng.shuffle(segment)
        if len(segment) % 2:
            singleton_edges += 1
            segment = segment[:-1]
        if not len(segment):
            continue
        left = segment[0::2]
        right = segment[1::2]
        attempted_pairs += len(left)
        same_source = np.asarray(original[left]) == np.asarray(original[right])
        same_target = np.asarray(targets[left]) == np.asarray(targets[right])
        valid = ~(same_source | same_target)
        same_source_pairs += int(same_source.sum())
        same_target_pairs += int((same_target & ~same_source).sum())
        if reject_new_self_loops:
            creates_self = (
                (np.asarray(original[right]) == np.asarray(targets[left]))
                | (np.asarray(original[left]) == np.asarray(targets[right]))
            )
            self_loop_rejections += int((valid & creates_self).sum())
            valid &= ~creates_self
        left = left[valid]
        right = right[valid]
        output[left] = original[right]
        output[right] = original[left]
        partner[left] = right.astype(np.int32, copy=False)
        partner[right] = left.astype(np.int32, copy=False)
    if isinstance(output, np.memmap):
        output.flush()
    if isinstance(partner, np.memmap):
        partner.flush()
    del order, sorted_keys, boundaries, starts, stops, keys

    progress("rejecting proposals that would create duplicate directed edges")
    duplicate_rejections, cleanup_passes = _reject_duplicate_pairs(
        original, output, partner, indptr
    )
    accepted_pairs = int(np.count_nonzero(partner >= 0) // 2)
    if isinstance(output, np.memmap):
        output.flush()
    return {
        "attempted_pairs": int(attempted_pairs),
        "accepted_pairs": accepted_pairs,
        "singleton_stratum_edges": int(singleton_edges),
        "same_source_rejected_pairs": int(same_source_pairs),
        "same_target_rejected_pairs": int(same_target_pairs),
        "self_loop_rejected_pairs": int(self_loop_rejections),
        "duplicate_rejected_pairs": int(duplicate_rejections),
        "duplicate_cleanup_passes": int(cleanup_passes),
        "occupied_strata": int(occupied_strata),
    }


def _sort_canonical(
    indices: np.ndarray, counts: np.ndarray, indptr: np.ndarray
) -> None:
    for row in range(len(indptr) - 1):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if stop - start < 2:
            continue
        sources = np.asarray(indices[start:stop]).copy()
        weights = np.asarray(counts[start:stop]).copy()
        order = np.argsort(sources, kind="stable")
        sources = sources[order]
        if np.any(sources[1:] == sources[:-1]):
            raise RuntimeError(f"duplicate directed edge remains in row {row}")
        indices[start:stop] = sources
        counts[start:stop] = weights[order]
    if isinstance(indices, np.memmap):
        indices.flush()
    if isinstance(counts, np.memmap):
        counts.flush()


def _self_loop_count(indices: np.ndarray, indptr: np.ndarray) -> int:
    total = 0
    for start in range(0, len(indptr) - 1, 4096):
        stop = min(start + 4096, len(indptr) - 1)
        edge_start, edge_stop = int(indptr[start]), int(indptr[stop])
        targets = np.repeat(
            np.arange(start, stop, dtype=np.int32), np.diff(indptr[start : stop + 1])
        )
        total += int(np.count_nonzero(indices[edge_start:edge_stop] == targets))
    return total


def _row_synapse_sums(counts: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    sums = np.zeros(len(indptr) - 1, dtype=np.uint64)
    starts = np.asarray(indptr[:-1])
    nonempty = starts < np.asarray(indptr[1:])
    if nonempty.any():
        sums[nonempty] = np.add.reduceat(
            np.asarray(counts, dtype=np.uint64), starts[nonempty], dtype=np.uint64
        )
    return sums


def _edge_overlap(original: np.ndarray, control: np.ndarray, indptr: np.ndarray) -> int:
    overlap = 0
    for row in range(len(indptr) - 1):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if start == stop:
            continue
        before = np.asarray(original[start:stop])
        after = np.asarray(control[start:stop])
        positions = np.searchsorted(before, after)
        bounded = np.minimum(positions, len(before) - 1)
        overlap += int(np.count_nonzero((positions < len(before)) & (before[bounded] == after)))
    return overlap


def _degree_and_mixing_invariants(
    original: np.ndarray,
    control: np.ndarray,
    original_counts: np.ndarray,
    control_counts: np.ndarray,
    targets: np.ndarray,
    category_codes: np.ndarray,
    transmitter_codes: np.ndarray,
) -> dict[str, bool]:
    neuron_count = len(category_codes)
    before_degree = np.bincount(original, minlength=neuron_count)
    after_degree = np.bincount(control, minlength=neuron_count)
    before_strength = np.bincount(
        original, weights=original_counts, minlength=neuron_count
    )
    after_strength = np.bincount(
        control, weights=control_counts, minlength=neuron_count
    )
    category_count = int(category_codes.max(initial=-1)) + 1
    relation_count = (int(transmitter_codes.max(initial=-1)) + 1) * category_count**2
    edge_before = np.zeros(relation_count, dtype=np.int64)
    edge_after = np.zeros(relation_count, dtype=np.int64)
    weight_before = np.zeros(relation_count, dtype=np.float64)
    weight_after = np.zeros(relation_count, dtype=np.float64)
    for start in range(0, len(original), 2_000_000):
        stop = min(start + 2_000_000, len(original))
        post_category = category_codes[targets[start:stop]]
        old_source = np.asarray(original[start:stop])
        new_source = np.asarray(control[start:stop])
        old_relation = (
            (transmitter_codes[old_source] * category_count + category_codes[old_source])
            * category_count
            + post_category
        )
        new_relation = (
            (transmitter_codes[new_source] * category_count + category_codes[new_source])
            * category_count
            + post_category
        )
        edge_before += np.bincount(old_relation, minlength=relation_count)
        edge_after += np.bincount(new_relation, minlength=relation_count)
        old_weight = np.asarray(original_counts[start:stop], dtype=np.float64)
        new_weight = np.asarray(control_counts[start:stop], dtype=np.float64)
        weight_before += np.bincount(
            old_relation, weights=old_weight, minlength=relation_count
        )
        weight_after += np.bincount(
            new_relation, weights=new_weight, minlength=relation_count
        )
    return {
        "per_neuron_out_degree_exact": bool(np.array_equal(before_degree, after_degree)),
        "per_neuron_out_synapse_strength_exact": bool(
            np.array_equal(before_strength, after_strength)
        ),
        "transmitter_category_edge_mixing_exact": bool(
            np.array_equal(edge_before, edge_after)
        ),
        "transmitter_category_synapse_mixing_exact": bool(
            np.array_equal(weight_before, weight_after)
        ),
    }


def quick_invariant_check(
    graph: Any, *, rows: int = 4096, seed: int = 0
) -> dict[str, Any]:
    """Exercise the real graph prefix before a full bulk build."""
    rows = min(max(int(rows), 1), int(graph.n))
    edge_stop = int(graph.indptr[rows])
    indptr = np.asarray(graph.indptr[: rows + 1]).copy()
    original = np.asarray(graph.indices[:edge_stop]).copy()
    counts = np.asarray(graph.counts[:edge_stop]).copy()
    targets = _targets(indptr)
    _, categories = _categorical_codes(graph.superclasses)
    _, transmitters = _categorical_codes(graph.effective_nt)
    output = np.empty_like(original)
    original_self_loops = int(np.count_nonzero(original == targets))
    proposal = _rewire_sources(
        original, counts, indptr, targets, categories, transmitters,
        seed=seed, reject_new_self_loops=original_self_loops == 0, output=output,
    )
    control_counts = counts.copy()
    _sort_canonical(output, control_counts, indptr)
    invariants = _degree_and_mixing_invariants(
        original, output, counts, control_counts, targets, categories, transmitters
    )
    invariants.update({
        "per_target_in_degree_exact": True,
        "per_target_in_synapse_strength_exact": bool(
            np.array_equal(
                _row_synapse_sums(counts, indptr),
                _row_synapse_sums(control_counts, indptr),
            )
        ),
        "canonical_unique_rows": True,
    })
    if not all(invariants.values()):
        raise RuntimeError(f"quick control invariant failed: {invariants}")
    return {
        "target_rows": rows,
        "edges": edge_stop,
        "accepted_pairs": proposal["accepted_pairs"],
        "invariants": invariants,
    }


def load_connectome_control(
    artifact_directory: str | Path,
    *,
    source_dataset_hash: str | None = None,
    verify: bool = True,
) -> Any:
    """Load a control through the normal MaleCNS API and validate its receipt."""
    from .malecns import MaleCNSGraph

    graph = MaleCNSGraph.load(artifact_directory, mmap=True, verify=verify)
    control = graph.manifest.get("control")
    source = graph.manifest.get("source_graph")
    if not isinstance(control, dict) or control.get("kind") != "matched_rewire":
        raise ValueError("artifact is not a matched-rewiring connectome control")
    if not isinstance(source, dict) or not source.get("dataset_hash"):
        raise ValueError("control manifest has no source graph receipt")
    if source_dataset_hash is not None and source["dataset_hash"] != source_dataset_hash:
        raise ValueError("control was derived from a different source graph")
    if control.get("spec_sha256") != _json_sha256(control.get("spec")):
        raise ValueError("control spec hash does not match its manifest")
    return graph


def build_matched_control(
    graph: Any,
    output_directory: str | Path,
    spec: MatchedRewireSpec,
    *,
    scratch_directory: str | Path | None = None,
    quick_check_rows: int = 4096,
    progress: Progress = print,
) -> dict[str, Any]:
    """Build a full MaleCNSGraph-compatible matched-rewiring artifact."""
    spec.validate()
    output_directory = Path(output_directory).expanduser().resolve()
    source_directory = Path(graph.path).resolve()
    if output_directory == source_directory:
        raise ValueError("control output must differ from the source graph directory")
    if output_directory.exists():
        raise FileExistsError(f"control output already exists: {output_directory}")
    temporary = output_directory.with_name(output_directory.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"partial control output already exists: {temporary}")
    scratch = (
        Path(scratch_directory).expanduser().resolve()
        if scratch_directory is not None
        else temporary / "scratch"
    )
    temporary.mkdir(parents=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        quick = quick_invariant_check(graph, rows=quick_check_rows, seed=spec.seed)
        progress(f"real-graph prefix invariant check passed: {json.dumps(quick, sort_keys=True)}")
        for filename in ("indptr.npy", "counts.npy", "row_synapses.npy", "neurons.npz"):
            shutil.copy2(source_directory / filename, temporary / filename)
        output_indices = np.lib.format.open_memmap(
            temporary / "indices.npy", mode="w+", dtype=np.int32,
            shape=(int(graph.edge_count),),
        )
        output_counts = np.load(temporary / "counts.npy", mmap_mode="r+", allow_pickle=False)
        target_array = _targets(graph.indptr, scratch / "targets.npy")
        category_names, category_codes = _categorical_codes(graph.field(spec.category_field))
        transmitter_names, transmitter_codes = _categorical_codes(
            graph.field(spec.transmitter_field)
        )
        original_self_loops = _self_loop_count(graph.indices, graph.indptr)
        proposal = _rewire_sources(
            graph.indices,
            graph.counts,
            graph.indptr,
            target_array,
            category_codes,
            transmitter_codes,
            seed=spec.seed,
            reject_new_self_loops=original_self_loops == 0,
            output=output_indices,
            scratch=scratch,
            progress=progress,
        )
        progress("canonicalizing control CSR rows")
        _sort_canonical(output_indices, output_counts, graph.indptr)
        progress("measuring full control invariants and edge overlap")
        invariants = _degree_and_mixing_invariants(
            graph.indices,
            output_indices,
            graph.counts,
            output_counts,
            target_array,
            category_codes,
            transmitter_codes,
        )
        invariants.update({
            "per_target_in_degree_exact": bool(
                np.array_equal(
                    np.diff(graph.indptr),
                    np.diff(np.load(temporary / "indptr.npy", mmap_mode="r")),
                )
            ),
            "per_target_in_synapse_strength_exact": bool(
                np.array_equal(
                    graph.row_synapses,
                    _row_synapse_sums(output_counts, graph.indptr),
                )
            ),
            "edge_synapse_count_histogram_exact": bool(
                np.array_equal(
                    np.bincount(graph.counts), np.bincount(output_counts)
                )
            ),
            "canonical_unique_rows": True,
        })
        if not all(invariants.values()):
            raise RuntimeError(f"full control invariant failed: {invariants}")
        overlap = _edge_overlap(graph.indices, output_indices, graph.indptr)
        control_self_loops = _self_loop_count(output_indices, graph.indptr)
        changed_edges = int(graph.edge_count - overlap)
        statistics = {
            **proposal,
            "edges": int(graph.edge_count),
            "synapses": int(graph.counts.sum(dtype=np.uint64)),
            "edge_set_overlap": int(overlap),
            "rewired_edges": changed_edges,
            "rewired_fraction": changed_edges / int(graph.edge_count),
            "untouched_edges": int(overlap),
            "untouched_fraction": overlap / int(graph.edge_count),
            "source_self_loops": original_self_loops,
            "control_self_loops": control_self_loops,
            "self_loops_rejected": original_self_loops == 0,
            "categories": category_names,
            "transmitters": transmitter_names,
            "invariants": invariants,
            "quick_check": quick,
        }
        spec_document = spec.document()
        (temporary / "control-spec.json").write_text(
            json.dumps(spec_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_names = (
            "indptr.npy", "indices.npy", "counts.npy", "row_synapses.npy",
            "neurons.npz", "control-spec.json",
        )
        artifacts = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": _file_sha256(temporary / name),
            }
            for name in artifact_names
        }
        digest = hashlib.sha256()
        for name in artifact_names:
            digest.update(name.encode("utf-8"))
            digest.update(artifacts[name]["sha256"].encode("ascii"))
        dataset_hash = digest.hexdigest()
        manifest = json.loads(json.dumps(graph.manifest))
        manifest["schema_version"] = max(2, int(manifest.get("schema_version", 1)))
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["dataset"] = {
            **manifest["dataset"],
            "name": f"{manifest['dataset']['name']} matched-rewiring control",
        }
        manifest["source_graph"] = {
            "path_at_build": str(source_directory),
            "dataset_hash": str(graph.hash),
            "artifacts": graph.manifest["artifacts"],
        }
        manifest["control"] = {
            "kind": "matched_rewire",
            "spec": spec_document,
            "spec_sha256": spec.sha256,
            "statistics": statistics,
            "interpretation": (
                "A topology control for comparative learning runs. Preserved graph size "
                "and matched statistics do not make it a biological connectome."
            ),
        }
        manifest["artifacts"] = artifacts
        manifest["dataset_hash"] = dataset_hash
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for work_file in scratch.glob("*.npy"):
            work_file.unlink()
        if scratch == temporary / "scratch":
            scratch.rmdir()
        os.replace(temporary, output_directory)
        return {
            "path": str(output_directory),
            "dataset_hash": dataset_hash,
            "source_dataset_hash": str(graph.hash),
            "spec_sha256": spec.sha256,
            "artifacts": artifacts,
            "statistics": statistics,
        }
    except Exception:
        # Keep a partial directory for diagnosis; never mutate or replace source data.
        raise
