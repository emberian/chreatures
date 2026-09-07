#!/usr/bin/env python3
"""Read pinned MaleCNS anatomy and export an optic audit; never step a circuit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.graph / "manifest.json").read_text())
    annotation_hash = sha256(args.annotations)
    assert annotation_hash == manifest["sources"]["annotations"]["sha256"]
    neurons = np.load(args.graph / "neurons.npz", allow_pickle=False)
    ids, types, sides = (neurons[k] for k in ("body_ids", "types", "sides"))
    rows = feather.read_table(args.annotations).to_pylist()
    rows = sorted((row for row in rows if row["status"] == "Traced"), key=lambda r: r["bodyId"])
    assert np.array_equal(ids, np.asarray([row["bodyId"] for row in rows]))
    hexes = np.full((len(ids), 2), -1, dtype=np.int16)
    locations = {}
    for field in ("somaLocation", "tosomaLocation"):
        xyz = np.full((len(ids), 3), -1, dtype=np.int32)
        for i, row in enumerate(rows):
            if row[field] is not None:
                assert len(row[field]) == 3
                xyz[i] = row[field]
        locations[field] = xyz
    for i, row in enumerate(rows):
        if row["assignedOlHex1"] is not None and row["assignedOlHex2"] is not None:
            hexes[i] = (row["assignedOlHex1"], row["assignedOlHex2"])
    anchors = np.flatnonzero((hexes >= 0).all(axis=1))
    side_code = np.asarray([1 if s == "L" else 2 if s == "R" else 0 for s in sides], dtype=np.uint8)
    sites = sorted({(int(side_code[i]), *map(int, hexes[i])) for i in anchors})
    assert all(site[0] in (1, 2) for site in sites)
    site_position = {site: i for i, site in enumerate(sites)}
    site_index = np.full(len(ids), -1, dtype=np.int32)
    for i in anchors:
        site_index[i] = site_position[(int(side_code[i]), *map(int, hexes[i]))]
    phot = np.asarray([str(t).startswith(("R1", "R7", "R8")) for t in types])
    phot_rows = np.flatnonzero(phot)
    indptr, indices, counts = (np.load(args.graph / name, mmap_mode="r") for name in
                               ("indptr.npy", "indices.npy", "counts.npy"))
    projection = defaultdict(Counter)
    cross_side = 0
    for target in anchors:
        span = slice(indptr[target], indptr[target + 1])
        pres, weights = indices[span], counts[span]
        for pre, weight in zip(pres[phot[pres]], weights[phot[pres]], strict=True):
            if side_code[pre] != side_code[target]:
                cross_side += int(weight)
            else:
                projection[int(pre)][int(site_index[target])] += int(weight)
    photo_ptr, photo_sites, photo_counts = [0], [], []
    for pre in phot_rows:
        for site, count in sorted(projection[int(pre)].items()):
            photo_sites.append(site)
            photo_counts.append(count)
        photo_ptr.append(len(photo_sites))
    type_names, type_code = np.unique(types, return_inverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, body_ids=ids, side=side_code, type_names=type_names,
        type_index=type_code.astype(np.int16), hex_coordinates=hexes,
        soma_xyz_source_units=locations["somaLocation"],
        tosoma_xyz_source_units=locations["tosomaLocation"],
        site_side_hex=np.asarray(sites, dtype=np.int16), neuron_site=site_index,
        photoreceptor_graph_rows=phot_rows.astype(np.int32),
        photoreceptor_site_indptr=np.asarray(photo_ptr, dtype=np.int32),
        photoreceptor_site_indices=np.asarray(photo_sites, dtype=np.int32),
        photoreceptor_site_synapse_counts=np.asarray(photo_counts, dtype=np.uint32),
    )
    ports = np.load(args.ports, allow_pickle=False)
    meta = json.loads(ports["metadata"].item())
    port_rows = np.repeat(np.arange(len(ids)), np.diff(ports["input_indptr"]))
    retina = ports["input_indices"] < 320
    by_type = {}
    for typ in sorted(set(types[phot])):
        cells = np.flatnonzero(phot & (types == typ))
        mapped = [int(i) for i in cells if projection[int(i)]]
        fractions = [max(projection[i].values()) / sum(projection[i].values()) for i in mapped]
        by_type[str(typ)] = {
            "cells": len(cells), "mapped": len(mapped),
            "sides": dict(Counter(map(str, sides[cells]))),
            "top_column_fraction_quantiles_0_25_50_75_100": np.quantile(fractions, [0, .25, .5, .75, 1]).tolist() if fractions else [],
        }
    receipt = {
        "format": "chreatures-optic-anatomy-audit-v1", "status": "executed read-only anatomy extraction; no circuit stepped",
        "graph_hash": manifest["dataset_hash"], "annotation_sha256": annotation_hash,
        "port_sha256": sha256(args.ports), "atlas_sha256": sha256(args.output),
        "atlas_bytes": args.output.stat().st_size, "graph_counts": {k: manifest["counts"][k] for k in ("neurons", "edges", "synapses")},
        "assigned_hex_cells": len(anchors), "side_hex_sites": len(sites),
        "side_hex_site_counts": {s: sum(t[0] == code for t in sites) for s, code in (("L", 1), ("R", 2))},
        "hex_type_counts": dict(sorted(Counter(map(str, types[anchors])).items())),
        "soma_location_cells": int((locations["somaLocation"] >= 0).all(axis=1).sum()),
        "soma_location_hex_cells": int((locations["somaLocation"][anchors] >= 0).all(axis=1).sum()),
        "tosoma_location_cells": int((locations["tosomaLocation"] >= 0).all(axis=1).sum()),
        "tosoma_location_hex_cells": int((locations["tosomaLocation"][anchors] >= 0).all(axis=1).sum()),
        "coordinate_units": "source integer coordinates retained; inferred 8nm EM voxels, supported by official SWC same-cell comparison; annotation-specific unit metadata absent",
        "retinal_v2": {
            "scalar_channels": 320, "spatial_bins": 80,
            "unique_targets": int(len(np.unique(port_rows[retina]))), "nonzeros": int(retina.sum()),
            "fallback_ports": sum(bool(p["nearest_hex_fallback"]) for p in meta["inputs"]["ports"] if p.get("family") == "retina"),
            "target_sides": dict(Counter(map(str, sides[np.unique(port_rows[retina])]))),
        },
        "photoreceptors": {
            "cells": len(phot_rows), "direct_assigned_hex_cells": int((hexes[phot] >= 0).all(axis=1).sum()),
            "mapped_via_same_side_outgoing_synapses_to_hex_anchors": sum(bool(projection[int(i)]) for i in phot_rows),
            "mapping_synapses": sum(photo_counts), "cross_side_synapses_excluded": cross_side,
            "evidenced_site_count": len(set(photo_sites)),
            "evidenced_site_counts_by_side": {s: sum(sites[i][0] == code for i in set(photo_sites)) for s, code in (("L", 1), ("R", 2))},
            "types": by_type,
        },
        "mapping_limit": "Measured connectivity-derived column distribution; not measured optical angle, spectral sensitivity or physiological receptive field. Unmapped receptors stay unmapped.",
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"atlas": str(args.output), "atlas_bytes": receipt["atlas_bytes"], "mapped_photoreceptors": receipt["photoreceptors"]["mapped_via_same_side_outgoing_synapses_to_hex_anchors"], "side_hex_sites": len(sites)}))


if __name__ == "__main__":
    main()
