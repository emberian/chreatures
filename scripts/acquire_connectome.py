#!/usr/bin/env python3
"""Extract a compact, typed circuit from pinned FlyWire v783 tables.

The two large source tables are intentionally kept outside this repository.
This script is deterministic for fixed inputs and emits only the compact NPZ,
per-neuron JSON metadata, and a provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by users of data extra
    raise SystemExit(
        "Extraction needs pandas and pyarrow (pip install pandas pyarrow)."
    ) from exc


MODEL_REVISION = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOTATION_TAG = "v3.1.0"
ANNOTATION_REVISION = "8587524c1748ce5ef2080822a2fc890fc03bf597"
MODEL_URL = "https://github.com/philshiu/Drosophila_brain_model"
ANNOTATION_URL = "https://github.com/flyconnectome/flywire_annotations"
FLYWIRE_LICENSE_URL = "https://flywire.ai/guidelines"

SOURCES = {
    "connectivity": {
        "relative_path": "source_model/Connectivity_783.parquet",
        "url": f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{MODEL_REVISION}/Connectivity_783.parquet",
        "sha256": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    },
    "completeness": {
        "relative_path": "source_model/Completeness_783.csv",
        "url": f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{MODEL_REVISION}/Completeness_783.csv",
        "sha256": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
    },
    "annotations": {
        "relative_path": f"Supplemental_file1_neuron_annotations.{ANNOTATION_TAG}.tsv",
        "url": f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATION_TAG}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "sha256": "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
    },
}

CORE_CLASSES = {
    "ALPN": "PN",
    "Kenyon_Cell": "KC",
    "MBON": "MBON",
    "DAN": "DAN",
}

REQUIRED_ANNOTATION_COLUMNS = {
    "root_id",
    "flow",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "hemibrain_type",
    "top_nt",
    "top_nt_conf",
    "known_nt",
    "known_nt_source",
    "side",
}

EDGE_COLUMNS = ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unicode_array(values: Iterable[object]) -> np.ndarray:
    values = ["" if value is None else str(value) for value in values]
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{max(width, 1)}")


def isin_sorted(values: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
    if len(sorted_values) == 0:
        return np.zeros(len(values), dtype=bool)
    positions = np.searchsorted(sorted_values, values)
    bounded = np.minimum(positions, len(sorted_values) - 1)
    return (positions < len(sorted_values)) & (sorted_values[bounded] == values)


def score_candidates(
    parquet: pq.ParquetFile,
    core_ids: np.ndarray,
    candidate_ids: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Sum measured synapses on edges in either direction between candidates/core."""
    scores = np.zeros(len(candidate_ids), dtype=np.int64)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=EDGE_COLUMNS):
        pre = batch.column(0).to_numpy(zero_copy_only=False)
        post = batch.column(1).to_numpy(zero_copy_only=False)
        count = batch.column(2).to_numpy(zero_copy_only=False)

        positions = np.searchsorted(candidate_ids, pre)
        candidate_pre = (positions < len(candidate_ids))
        if len(candidate_ids):
            candidate_pre &= candidate_ids[np.minimum(positions, len(candidate_ids) - 1)] == pre
        mask = candidate_pre & isin_sorted(post, core_ids)
        np.add.at(scores, positions[mask], count[mask])

        positions = np.searchsorted(candidate_ids, post)
        candidate_post = (positions < len(candidate_ids))
        if len(candidate_ids):
            candidate_post &= candidate_ids[np.minimum(positions, len(candidate_ids) - 1)] == post
        mask = candidate_post & isin_sorted(pre, core_ids)
        np.add.at(scores, positions[mask], count[mask])
    return scores


def strongest(candidate_ids: np.ndarray, scores: np.ndarray, limit: int) -> np.ndarray:
    connected = scores > 0
    ids = candidate_ids[connected]
    values = scores[connected]
    # Primary key is descending synapse score, secondary key ascending root ID.
    order = np.lexsort((ids, -values))
    return ids[order[:limit]]


def induced_edges(
    parquet: pq.ParquetFile,
    ordered_ids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_order = np.argsort(ordered_ids)
    sorted_ids = ordered_ids[sorted_order]
    all_pre: list[np.ndarray] = []
    all_post: list[np.ndarray] = []
    all_count: list[np.ndarray] = []

    for batch in parquet.iter_batches(batch_size=batch_size, columns=EDGE_COLUMNS):
        pre_id = batch.column(0).to_numpy(zero_copy_only=False)
        post_id = batch.column(1).to_numpy(zero_copy_only=False)
        count = batch.column(2).to_numpy(zero_copy_only=False)
        pre_pos = np.searchsorted(sorted_ids, pre_id)
        post_pos = np.searchsorted(sorted_ids, post_id)
        mask = (pre_pos < len(sorted_ids)) & (post_pos < len(sorted_ids))
        bounded_pre = np.minimum(pre_pos, len(sorted_ids) - 1)
        bounded_post = np.minimum(post_pos, len(sorted_ids) - 1)
        mask &= sorted_ids[bounded_pre] == pre_id
        mask &= sorted_ids[bounded_post] == post_id
        if mask.any():
            all_pre.append(sorted_order[pre_pos[mask]].astype(np.int32))
            all_post.append(sorted_order[post_pos[mask]].astype(np.int32))
            all_count.append(count[mask].astype(np.int64))

    pre = np.concatenate(all_pre)
    post = np.concatenate(all_post)
    count = np.concatenate(all_count)

    # The source currently has one row per directed pair. Aggregate defensively
    # if a future byte-identical schema contains repeated regional rows.
    pair_key = pre.astype(np.int64) * len(ordered_ids) + post.astype(np.int64)
    order = np.argsort(pair_key, kind="stable")
    pair_key = pair_key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pair_key)) + 1]
    return (
        pre[order][starts].astype(np.int32),
        post[order][starts].astype(np.int32),
        np.add.reduceat(count[order], starts).astype(np.float32),
    )


def choose_label(row: pd.Series) -> str:
    for column in ("cell_type", "hemibrain_type", "cell_sub_class", "cell_class"):
        if row[column]:
            return row[column]
    return row["root_id"]


SMALL_MOLECULE_TRANSMITTERS = (
    "acetylcholine", "gaba", "glutamate", "dopamine", "serotonin",
    "octopamine", "tyramine", "nitric oxide",
)


def resolve_transmitter(known: str, predicted: str) -> tuple[str, str]:
    """Prefer positive literature annotations, otherwise retain top_nt prediction."""
    tokens = [token.strip().lower() for token in known.replace(";", ",").split(",")]
    positives = {
        token for token in tokens
        if token in SMALL_MOLECULE_TRANSMITTERS and not token.endswith("-negative")
    }
    if positives:
        ordered = [value for value in SMALL_MOLECULE_TRANSMITTERS if value in positives]
        return "+".join(ordered), "known_nt"
    if predicted:
        return predicted.lower(), "top_nt"
    return "unavailable", "unavailable"


def sign_for_transmitter(value: str) -> float:
    # Declared rate-model assumption; these are not signs measured by EM.
    parts = set(value.split("+"))
    has_excitation = "acetylcholine" in parts
    has_inhibition = bool(parts.intersection({"gaba", "glutamate"}))
    if has_excitation and has_inhibition:
        return 0.0
    if has_excitation:
        return 1.0
    if has_inhibition:
        return -1.0
    return 0.0


def download_source(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    print(f"downloading {url} -> {path}")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        help="Bulk source root; fills any source path not given explicitly",
    )
    parser.add_argument("--connectivity", type=Path)
    parser.add_argument("--completeness", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--download", action="store_true",
        help="Download missing pinned inputs (requires --source-dir)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--navigation-limit", type=int, default=250)
    parser.add_argument("--descending-limit", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in SOURCES:
        path = getattr(args, name)
        if path is None and args.source_dir is not None:
            path = args.source_dir / SOURCES[name]["relative_path"]
            setattr(args, name, path)
        if path is None:
            raise SystemExit(f"provide --{name.replace('_', '-')} or --source-dir")
        if not path.is_file() and args.download:
            if args.source_dir is None:
                raise SystemExit("--download requires --source-dir")
            download_source(path, SOURCES[name]["url"])
        if not path.is_file():
            raise SystemExit(f"missing source file: {path}")
        observed_hash = sha256(path)
        if observed_hash != SOURCES[name]["sha256"]:
            raise SystemExit(
                f"source hash mismatch for {path}: {observed_hash}; "
                f"expected {SOURCES[name]['sha256']}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(
        args.annotations, sep="\t", dtype=str, keep_default_na=False
    )
    missing = REQUIRED_ANNOTATION_COLUMNS.difference(annotations.columns)
    if missing:
        raise SystemExit(f"annotation table missing columns: {sorted(missing)}")
    completeness = pd.read_csv(args.completeness, dtype=str, keep_default_na=False)
    complete_ids = set(completeness.iloc[:, 0])
    annotations = annotations[annotations["root_id"].isin(complete_ids)].copy()
    if annotations["root_id"].duplicated().any():
        raise SystemExit("annotation root_id values are not unique")

    core_mask = annotations["cell_class"].isin(CORE_CLASSES)
    core_ids = np.sort(annotations.loc[core_mask, "root_id"].astype(np.int64).to_numpy())
    cx_ids = np.sort(
        annotations.loc[annotations["cell_class"].eq("CX"), "root_id"]
        .astype(np.int64)
        .to_numpy()
    )
    descending_ids = np.sort(
        annotations.loc[annotations["super_class"].eq("descending"), "root_id"]
        .astype(np.int64)
        .to_numpy()
    )

    parquet = pq.ParquetFile(args.connectivity)
    if not set(EDGE_COLUMNS).issubset(parquet.schema.names):
        raise SystemExit(f"connectivity table missing one of {EDGE_COLUMNS}")
    cx_scores = score_candidates(parquet, core_ids, cx_ids, args.batch_size)
    descending_scores = score_candidates(parquet, core_ids, descending_ids, args.batch_size)
    selected_cx = strongest(cx_ids, cx_scores, args.navigation_limit)
    selected_descending = strongest(
        descending_ids, descending_scores, args.descending_limit
    )

    group_by_id: dict[int, str] = {}
    for cell_class, group in CORE_CLASSES.items():
        ids = annotations.loc[
            annotations["cell_class"].eq(cell_class), "root_id"
        ].astype(np.int64)
        group_by_id.update({int(root_id): group for root_id in ids})
    group_by_id.update({int(root_id): "CX" for root_id in selected_cx})
    group_by_id.update({int(root_id): "descending" for root_id in selected_descending})
    group_order = {name: i for i, name in enumerate(["PN", "KC", "MBON", "DAN", "CX", "descending"])}
    ordered_ids = np.asarray(
        sorted(group_by_id, key=lambda root_id: (group_order[group_by_id[root_id]], root_id)),
        dtype=np.int64,
    )

    by_id = annotations.set_index("root_id", drop=False)
    rows = by_id.loc[[str(root_id) for root_id in ordered_ids]].copy()
    rows["group"] = [group_by_id[int(root_id)] for root_id in ordered_ids]
    labels = [choose_label(row) for _, row in rows.iterrows()]
    predicted_nt = rows["top_nt"].tolist()
    resolved = [
        resolve_transmitter(known, predicted)
        for known, predicted in zip(rows["known_nt"], predicted_nt)
    ]
    effective_nt = [value for value, _ in resolved]
    nt_basis = [basis for _, basis in resolved]
    signs = np.asarray(
        [sign_for_transmitter(value) for value in effective_nt], dtype=np.float32
    )
    confidence = pd.to_numeric(rows["top_nt_conf"], errors="coerce").fillna(0.0).to_numpy(np.float32)

    pre, post, count = induced_edges(parquet, ordered_ids, args.batch_size)
    if len(pre) == 0 or np.any(count <= 0):
        raise SystemExit("extracted graph has no edges or a non-positive edge count")

    npz_path = args.output_dir / "circuit.npz"
    np.savez_compressed(
        npz_path,
        ids=unicode_array(str(root_id) for root_id in ordered_ids),
        pre=pre,
        post=post,
        count=count,
        sign=signs,
        labels=unicode_array(labels),
        type=unicode_array(rows["cell_type"]),
        side=unicode_array(rows["side"]),
        group=unicode_array(rows["group"]),
        predicted_nt=unicode_array(predicted_nt),
        effective_nt=unicode_array(effective_nt),
        nt_basis=unicode_array(nt_basis),
        nt_confidence=confidence,
    )

    neuron_fields = [
        "root_id", "group", "flow", "super_class", "cell_class",
        "cell_sub_class", "cell_type", "hemibrain_type", "side", "top_nt",
        "top_nt_conf", "known_nt", "known_nt_source", "vfb_id", "fbbt_id",
    ]
    neuron_records = []
    for index, ((_, row), label) in enumerate(zip(rows.iterrows(), labels)):
        record = {field: row.get(field, "") for field in neuron_fields}
        record.update(
            index=index,
            label=label,
            effective_nt=effective_nt[index],
            nt_basis=nt_basis[index],
            model_sign=float(signs[index]),
        )
        neuron_records.append(record)
    neurons_path = args.output_dir / "neurons.json"
    neurons_path.write_text(
        json.dumps(neuron_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    group_counts = rows["group"].value_counts().sort_index().to_dict()
    nt_counts = rows["top_nt"].replace("", "unavailable").value_counts().sort_index().to_dict()
    effective_nt_counts = dict(sorted(pd.Series(effective_nt).value_counts().to_dict().items()))
    nt_basis_counts = dict(sorted(pd.Series(nt_basis).value_counts().to_dict().items()))
    sign_counts = {
        str(float(value)): int((signs == value).sum())
        for value in (-1.0, 0.0, 1.0)
    }
    side_counts = rows["side"].replace("", "unavailable").value_counts().sort_index().to_dict()
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "FlyWire FAFB public connectome",
            "materialization_version": 783,
            "license": "CC BY-NC 4.0",
            "license_url": FLYWIRE_LICENSE_URL,
            "attribution": [
                "Dorkenwald et al., Nature 634, 124-138 (2024), doi:10.1038/s41586-024-07558-y",
                "Schlegel et al., Nature 634, 139-152 (2024), doi:10.1038/s41586-024-07686-5",
            ],
            "license_note": (
                "FlyWire's public-release guidelines specify CC BY-NC 4.0. "
                "The Zenodo connectivity record metadata currently reports CC BY 4.0; "
                "this derivative conservatively follows the FlyWire guideline."
            ),
        },
        "sources": {
            "connectivity": {
                "repository": MODEL_URL,
                "revision": MODEL_REVISION,
                "path": "Connectivity_783.parquet",
                "sha256": sha256(args.connectivity),
                "rows": parquet.metadata.num_rows,
                "repository_license": "MIT",
                "underlying_data_record": "https://doi.org/10.5281/zenodo.10676866",
            },
            "completeness": {
                "repository": MODEL_URL,
                "revision": MODEL_REVISION,
                "path": "Completeness_783.csv",
                "sha256": sha256(args.completeness),
                "rows": len(completeness),
                "repository_license": "MIT",
            },
            "annotations": {
                "repository": ANNOTATION_URL,
                "tag": ANNOTATION_TAG,
                "revision": ANNOTATION_REVISION,
                "path": "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
                "sha256": sha256(args.annotations),
                "rows": len(pd.read_csv(args.annotations, sep="\t", usecols=["root_id"])),
                "materialization_version": 783,
            },
        },
        "selection": {
            "core_cell_classes": CORE_CLASSES,
            "navigation": f"top {args.navigation_limit} CX cells by measured synapse count incident to core",
            "descending": f"top {args.descending_limit} descending-superclass cells by measured synapse count incident to core",
            "edge_rule": "all measured directed source rows with both endpoints selected; duplicate pairs summed",
            "threshold": "none; one-synapse directed connections are retained",
        },
        "model_assumptions": {
            "sign": {
                "acetylcholine": 1.0,
                "gaba": -1.0,
                "glutamate": -1.0,
                "other_or_unavailable": 0.0,
                "mixed_excitation_and_inhibition": 0.0,
            },
            "transmitter_resolution": (
                "Positive small-molecule known_nt annotations take precedence; "
                "top_nt prediction is the fallback. Negative evidence and peptide-only "
                "known_nt fields do not override top_nt. Multiple positive known "
                "transmitters are retained with '+'."
            ),
            "note": "Sign is a declared rate-model assumption applied to effective_nt, not an EM measurement.",
            "synthetic_mappings": "No sensory injection or action readout mapping is stored in this artifact.",
        },
        "counts": {
            "neurons": int(len(ordered_ids)),
            "edges": int(len(pre)),
            "synapses": int(count.astype(np.int64).sum()),
            "groups": group_counts,
            "predicted_transmitters": nt_counts,
            "effective_transmitters": effective_nt_counts,
            "nt_basis": nt_basis_counts,
            "signs": sign_counts,
            "sides": side_counts,
        },
        "npz_schema": {
            "ids": "Unicode[N], exact FlyWire root IDs",
            "pre": "int32[E], presynaptic local indices",
            "post": "int32[E], postsynaptic local indices",
            "count": "float32[E], positive measured synapse counts",
            "sign": "float32[N], model sign by presynaptic neuron",
            "labels": "Unicode[N]",
            "type": "Unicode[N], cell_type annotation",
            "side": "Unicode[N]",
            "group": "Unicode[N]",
            "predicted_nt": "Unicode[N], top_nt annotation",
            "effective_nt": "Unicode[N], known_nt-first model transmitter",
            "nt_basis": "Unicode[N], known_nt/top_nt/unavailable provenance",
            "nt_confidence": "float32[N], top_nt_conf annotation; 0 if unavailable",
        },
        "artifacts": {
            "circuit.npz": {"sha256": sha256(npz_path), "bytes": npz_path.stat().st_size},
            "neurons.json": {"sha256": sha256(neurons_path), "bytes": neurons_path.stat().st_size},
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
