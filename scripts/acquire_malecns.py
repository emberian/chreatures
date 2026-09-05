#!/usr/bin/env python3
"""Acquire MaleCNS v1.0 and build the full curated-neuron CSR graph.

Bulk source and derived arrays belong on large storage (normally
``/tank/chreatures/data/malecns``).  The extractor makes two streaming passes
over the 152-million-row Arrow connectivity table and never materializes it as
a pandas DataFrame or a global COO edge list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc
except ImportError as exc:  # pragma: no cover - dependency error for operators
    raise SystemExit("MaleCNS extraction needs pyarrow (pip install pyarrow).") from exc


VERSION = "v1.0"
BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome"
)
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
PROJECT_URL = "https://male-cns.janelia.org/"
DOWNLOAD_URL = "https://male-cns.janelia.org/download/"

SOURCES = {
    "annotations": {
        "filename": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    },
    "neurotransmitters": {
        "filename": "body-neurotransmitters-male-cns-v1.0.feather",
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    },
    "connectivity": {
        "filename": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    },
}

ANNOTATION_COLUMNS = {
    "bodyId": "body_ids",
    "instance": "instances",
    "type": "types",
    "superclass": "superclasses",
    "class": "classes",
    "subclass": "subclasses",
    "somaSide": "soma_sides",
    "rootSide": "root_sides",
    "somaNeuromere": "soma_neuromeres",
    "entryNerve": "entry_nerves",
    "exitNerve": "exit_nerves",
    "status": "statuses",
    "statusLabel": "status_labels",
    "fruDsx": "fru_dsx",
    "receptorType": "receptor_types",
    "dimorphism": "dimorphism",
    "hemibrainType": "hemibrain_types",
    "flywireType": "flywire_types",
    "mancType": "manc_types",
}
NT_COLUMNS = (
    "body",
    "ground_truth",
    "predicted_nt",
    "predicted_nt_confidence",
    "consensus_nt",
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unicode_array(values: Iterable[object]) -> np.ndarray:
    cleaned = ["" if value is None else str(value) for value in values]
    width = max((len(value) for value in cleaned), default=1)
    return np.asarray(cleaned, dtype=f"<U{max(width, 1)}")


def open_feather(path: Path):
    source = pa.memory_map(str(path), "r")
    return source, ipc.open_file(source)


def download(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "chreatures-malecns/1"})
    print(f"downloading {url} -> {path}", flush=True)
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def read_columns(path: Path, names: Iterable[str]) -> dict[str, pa.ChunkedArray]:
    names = tuple(names)
    source, reader = open_feather(path)
    try:
        missing = set(names).difference(reader.schema.names)
        if missing:
            raise SystemExit(f"{path.name} is missing columns: {sorted(missing)}")
        chunks: dict[str, list[pa.Array]] = {name: [] for name in names}
        positions = {name: reader.schema.get_field_index(name) for name in names}
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            for name in names:
                array = batch.column(positions[name])
                if pa.types.is_dictionary(array.type):
                    array = pc.cast(array, pa.string())
                chunks[name].append(array)
        return {name: pa.chunked_array(parts) for name, parts in chunks.items()}
    finally:
        source.close()


def arrow_strings(array: pa.ChunkedArray) -> np.ndarray:
    return unicode_array(array.to_pylist())


def source_rows(path: Path) -> int:
    source, reader = open_feather(path)
    try:
        return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
    finally:
        source.close()


def selected_positions(values: np.ndarray, ordered_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(ordered_ids, values)
    bounded = np.minimum(positions, len(ordered_ids) - 1)
    mask = (positions < len(ordered_ids)) & (ordered_ids[bounded] == values)
    return positions, mask


def connectivity_batches(path: Path):
    source, reader = open_feather(path)
    required = {"body_pre", "body_post", "weight"}
    if not required.issubset(reader.schema.names):
        source.close()
        raise SystemExit(f"{path.name} is missing {sorted(required - set(reader.schema.names))}")
    positions = [reader.schema.get_field_index(name) for name in ("body_pre", "body_post", "weight")]
    try:
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            yield tuple(batch.column(pos).to_numpy(zero_copy_only=False) for pos in positions)
    finally:
        source.close()


def count_rows(connectivity: Path, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """First pass: count CSR rows and source/selected totals."""
    row_counts = np.zeros(len(ids), dtype=np.int64)
    row_synapses = np.zeros(len(ids), dtype=np.uint64)
    edge_count = 0
    synapse_count = 0
    source_count = 0
    for pre, post, weight in connectivity_batches(connectivity):
        pre_pos, pre_selected = selected_positions(pre, ids)
        post_pos, post_selected = selected_positions(post, ids)
        mask = pre_selected & post_selected
        selected_post = post_pos[mask]
        selected_weight = weight[mask]
        edge_count += int(mask.sum())
        synapse_count += int(selected_weight.sum(dtype=np.uint64))
        source_count += len(weight)
        row_counts += np.bincount(selected_post, minlength=len(ids)).astype(np.int64)
        np.add.at(row_synapses, selected_post, selected_weight.astype(np.uint64))
    return row_counts, row_synapses, edge_count, synapse_count, source_count


def fill_csr(
    connectivity: Path,
    ids: np.ndarray,
    indptr: np.ndarray,
    indices: np.memmap,
    counts: np.memmap,
) -> None:
    """Second pass: append each batch into its final CSR row."""
    cursor = indptr[:-1].copy()
    for pre, post, weight in connectivity_batches(connectivity):
        pre_pos, pre_selected = selected_positions(pre, ids)
        post_pos, post_selected = selected_positions(post, ids)
        mask = pre_selected & post_selected
        if not mask.any():
            continue
        selected_pre = pre_pos[mask].astype(np.int32, copy=False)
        selected_post = post_pos[mask].astype(np.int32, copy=False)
        selected_weight = weight[mask].astype(np.uint32, copy=False)

        order = np.argsort(selected_post, kind="stable")
        ordered_post = selected_post[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_post)) + 1]
        unique_post = ordered_post[starts]
        sizes = np.diff(np.r_[starts, len(order)])
        repeated_starts = np.repeat(starts, sizes)
        destinations = (
            np.repeat(cursor[unique_post], sizes)
            + np.arange(len(order), dtype=np.int64)
            - repeated_starts
        )
        indices[destinations] = selected_pre[order]
        counts[destinations] = selected_weight[order]
        cursor[unique_post] += sizes

    if not np.array_equal(cursor, indptr[1:]):
        raise RuntimeError("CSR fill count differs from first pass")
    indices.flush()
    counts.flush()


def sort_csr_rows(
    indptr: np.ndarray, indices: np.memmap, counts: np.memmap
) -> None:
    """Sort sources within each target row and reject duplicate pairs."""
    for row in range(len(indptr) - 1):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if stop - start < 2:
            continue
        order = np.argsort(indices[start:stop], kind="stable")
        sorted_indices = indices[start:stop][order]
        if np.any(np.diff(sorted_indices) == 0):
            raise RuntimeError(f"duplicate directed pair in CSR row {row}")
        indices[start:stop] = sorted_indices
        counts[start:stop] = counts[start:stop][order]
    indices.flush()
    counts.flush()


def resolve_nt(ground_truth: str, consensus: str, predicted: str) -> tuple[str, str]:
    for value, basis in (
        (ground_truth, "ground_truth"),
        (consensus, "consensus_nt"),
        (predicted, "predicted_nt"),
    ):
        value = value.strip().lower()
        if value and value != "unclear":
            return value, basis
    return "unavailable", "unavailable"


def model_sign(transmitter: str) -> float:
    # A declared rate-model simplification, not measured synaptic physiology.
    if transmitter == "acetylcholine":
        return 1.0
    if transmitter in {"gaba", "glutamate", "histamine"}:
        return -1.0
    return 0.0


def choose_labels(metadata: dict[str, np.ndarray], body_ids: np.ndarray) -> np.ndarray:
    labels = []
    for i, body_id in enumerate(body_ids):
        labels.append(next((metadata[field][i] for field in ("types", "instances", "hemibrain_types", "manc_types") if metadata[field][i]), str(body_id)))
    return unicode_array(labels)


def counter(values: np.ndarray, *, missing: str = "unavailable") -> dict[str, int]:
    return dict(sorted(Counter(value if value else missing for value in values.tolist()).items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tank/chreatures/data/malecns"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-source-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir or args.root / "source"
    output_dir = args.output_dir or args.root / "derived"
    paths = {name: source_dir / item["filename"] for name, item in SOURCES.items()}
    for name, path in paths.items():
        source = SOURCES[name]
        if not path.is_file() and args.download:
            download(path, f"{BASE_URL}/{source['filename']}")
        if not path.is_file():
            raise SystemExit(f"missing {name} source: {path}")
        if not args.skip_source_hashes:
            observed = sha256(path)
            if observed != source["sha256"]:
                raise SystemExit(f"hash mismatch for {path}: {observed}")

    annotation = read_columns(paths["annotations"], ANNOTATION_COLUMNS)
    traced = pc.fill_null(
        pc.equal(annotation["status"], "Traced"), False
    ).to_numpy(zero_copy_only=False)
    body_ids = annotation["bodyId"].filter(pa.array(traced)).to_numpy(zero_copy_only=False)
    order = np.argsort(body_ids, kind="stable")
    body_ids = body_ids[order].astype(np.int64, copy=False)
    if len(body_ids) == 0 or np.any(np.diff(body_ids) <= 0):
        raise SystemExit("traced annotation body IDs are empty or non-unique")

    metadata: dict[str, np.ndarray] = {}
    for source_name, output_name in ANNOTATION_COLUMNS.items():
        if source_name == "bodyId":
            continue
        values = arrow_strings(annotation[source_name].filter(pa.array(traced)))[order]
        metadata[output_name] = values
    metadata["labels"] = choose_labels(metadata, body_ids)
    metadata["sides"] = unicode_array(
        soma if soma else root
        for soma, root in zip(metadata["soma_sides"], metadata["root_sides"], strict=True)
    )

    nt = read_columns(paths["neurotransmitters"], NT_COLUMNS)
    nt_ids = nt["body"].to_numpy(zero_copy_only=False)
    nt_order = np.argsort(nt_ids, kind="stable")
    nt_ids = nt_ids[nt_order]
    if np.any(np.diff(nt_ids) <= 0):
        raise SystemExit("neurotransmitter body IDs are not unique")
    nt_positions = np.searchsorted(nt_ids, body_ids)
    nt_found = (nt_positions < len(nt_ids)) & (nt_ids[np.minimum(nt_positions, len(nt_ids) - 1)] == body_ids)

    def joined_nt(name: str) -> np.ndarray:
        source_values = arrow_strings(nt[name])[nt_order]
        values = np.full(len(body_ids), "", dtype=source_values.dtype)
        values[nt_found] = source_values[nt_positions[nt_found]]
        return values

    metadata["ground_truth_nt"] = joined_nt("ground_truth")
    metadata["predicted_nt"] = joined_nt("predicted_nt")
    metadata["consensus_nt"] = joined_nt("consensus_nt")
    confidence_source = nt["predicted_nt_confidence"].to_numpy(zero_copy_only=False)[nt_order]
    confidence = np.zeros(len(body_ids), dtype=np.float32)
    confidence[nt_found] = np.nan_to_num(confidence_source[nt_positions[nt_found]], nan=0.0).astype(np.float32)
    metadata["nt_confidence"] = confidence
    resolved = [
        resolve_nt(ground, consensus, predicted)
        for ground, consensus, predicted in zip(
            metadata["ground_truth_nt"], metadata["consensus_nt"], metadata["predicted_nt"], strict=True
        )
    ]
    metadata["effective_nt"] = unicode_array(value for value, _ in resolved)
    metadata["nt_basis"] = unicode_array(basis for _, basis in resolved)
    metadata["sign"] = np.asarray([model_sign(value) for value in metadata["effective_nt"]], dtype=np.float32)
    metadata["ids"] = unicode_array(body_ids)
    metadata["body_ids"] = body_ids

    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts, row_synapses, edge_count, synapse_count, connectivity_rows = count_rows(paths["connectivity"], body_ids)
    indptr = np.empty(len(body_ids) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    if int(indptr[-1]) != edge_count:
        raise RuntimeError("CSR row counts do not sum to selected edge count")

    temporary_indices = output_dir / "indices.partial.npy"
    temporary_counts = output_dir / "counts.partial.npy"
    indices = np.lib.format.open_memmap(temporary_indices, mode="w+", dtype=np.int32, shape=(edge_count,))
    counts = np.lib.format.open_memmap(temporary_counts, mode="w+", dtype=np.uint32, shape=(edge_count,))
    fill_csr(paths["connectivity"], body_ids, indptr, indices, counts)
    sort_csr_rows(indptr, indices, counts)
    del indices, counts
    os.replace(temporary_indices, output_dir / "indices.npy")
    os.replace(temporary_counts, output_dir / "counts.npy")
    np.save(output_dir / "indptr.npy", indptr, allow_pickle=False)
    np.save(output_dir / "row_synapses.npy", row_synapses, allow_pickle=False)
    np.savez_compressed(output_dir / "neurons.npz", **metadata)

    afferent = np.asarray([
        ("_sensory" in value) or value.startswith("sensory_")
        for value in metadata["superclasses"]
    ])
    efferent = np.asarray([
        ("_motor" in value) or ("_efferent" in value) or value.startswith("efferent_")
        for value in metadata["superclasses"]
    ])
    excluded_status_counts = counter(arrow_strings(annotation["status"])[~traced])
    source_details = {}
    for name, source in SOURCES.items():
        path = paths[name]
        source_details[name] = {
            "url": f"{BASE_URL}/{source['filename']}",
            "gs_url": f"gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/{source['filename']}",
            "filename": source["filename"],
            "sha256": source["sha256"],
            "bytes": path.stat().st_size,
            "rows": source_rows(path),
        }

    artifact_names = ("indptr.npy", "indices.npy", "counts.npy", "row_synapses.npy", "neurons.npz")
    artifacts = {
        name: {"sha256": sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
        for name in artifact_names
    }
    digest = hashlib.sha256()
    for name in artifact_names:
        digest.update(name.encode())
        digest.update(artifacts[name]["sha256"].encode())

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "MaleCNS connectome",
            "version": VERSION,
            "scope": "adult male Drosophila brain and ventral nerve cord",
            "project_url": PROJECT_URL,
            "download_url": DOWNLOAD_URL,
            "license": "CC BY 4.0",
            "license_url": LICENSE_URL,
            "citation": "Sexual dimorphism in the complete Drosophila male central nervous system connectome, Cell (2026), doi:10.1016/j.cell.2026.08.015",
            "doi": "10.1016/j.cell.2026.08.015",
        },
        "selection": {
            "neuron_rule": "body annotation status equals 'Traced'",
            "edge_rule": "full directed weight row iff both body_pre and body_post satisfy neuron_rule",
            "threshold": "none beyond the official source file's minconf-0.5 inclusion",
            "included_annotation_rows": int(len(body_ids)),
            "excluded_annotation_rows": int(len(annotation["bodyId"]) - len(body_ids)),
            "excluded_statuses": excluded_status_counts,
            "reason": "Traced annotations are curated neurons; orphan, glia, unimportant, assign, anchor, and missing-status segments are excluded as non-curated graph fragments/classes.",
        },
        "sources": source_details,
        "counts": {
            "neurons": int(len(body_ids)),
            "edges": int(edge_count),
            "synapses": int(synapse_count),
            "source_connectivity_rows": int(connectivity_rows),
            "afferent_neurons": int(afferent.sum()),
            "efferent_neurons": int(efferent.sum()),
            "superclasses": counter(metadata["superclasses"]),
            "classes": counter(metadata["classes"]),
            "effective_transmitters": counter(metadata["effective_nt"]),
            "nt_basis": counter(metadata["nt_basis"]),
        },
        "graph": {
            "format": "memory-mappable NumPy CSR",
            "orientation": "rows are postsynaptic targets; indices are presynaptic sources",
            "shape": [int(len(body_ids)), int(len(body_ids))],
            "indptr": "int64[N+1]",
            "indices": "int32[E], presynaptic local indices",
            "counts": "uint32[E], measured synapse counts",
            "row_synapses": "uint64[N], total measured incoming synapses retained per target",
            "ordering": "neurons by ascending exact bodyId; edges grouped by postsynaptic row",
            "canonical": "presynaptic indices strictly increase within every row; source has one row per directed pair",
        },
        "metadata": {
            "file": "neurons.npz",
            "ordering": "row-aligned with graph local indices",
            "fields": {name: str(value.dtype) for name, value in metadata.items()},
        },
        "populations": {
            "afferent": {
                "rule": "superclass contains '_sensory' or starts with 'sensory_'",
                "count": int(afferent.sum()),
                "default_channels": "16 engineered runtime channels derived from afferent class, subclass, side, and deterministic body-ID partitions; see chreatures.malecns.DEFAULT_INPUT_CHANNELS",
            },
            "efferent": {
                "rule": "superclass contains '_motor'/'_efferent' or starts with 'efferent_'",
                "count": int(efferent.sum()),
                "default_readouts": "47 largest exact superclass/region/side signatures plus other_efferent; 48 disjoint row-normalized populations",
            },
            "note": "These annotation-derived populations are interface selectors, not physiological response or control models.",
        },
        "neurotransmitters": {
            "resolution": "ground_truth, then non-unclear consensus_nt, then non-unclear predicted_nt",
            "model_sign": {
                "acetylcholine": 1.0,
                "gaba": -1.0,
                "glutamate": -1.0,
                "histamine": -1.0,
                "other_or_unavailable": 0.0,
            },
            "caveat": "Transmitter labels and neuron-level model signs do not determine exact synaptic physiology, receptor-specific effects, cotransmission, or modulatory dynamics.",
        },
        "normalization": {
            "runtime_default": "count * model_sign[source] / max(total retained incoming count[target], 1)",
            "caveat": "This is an explicit stable rate-model transform, not a measured MaleCNS property.",
        },
        "artifacts": artifacts,
        "dataset_hash": digest.hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": manifest["counts"], "dataset_hash": manifest["dataset_hash"]}, indent=2))


if __name__ == "__main__":
    main()
