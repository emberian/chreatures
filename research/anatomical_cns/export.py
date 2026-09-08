#!/usr/bin/env python3
"""One-way sealed V2 plus annotation-derived graph seed to a fresh CHCNS3."""
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from chreatures.cns_adapter_contract import write_service_artifact
from research.anatomical_cns.model import initialized_arrays

V2_MAGIC = b"CHCNS2\0\0"
V2_SPECS = (
    ("graph.crow", "<u4", (165123,)),
    ("graph.col", "<u4", (25563197,)),
    ("graph.weight", "<f4", (25563197,)),
    ("atlas.receptor_rows", "<u4", (4107,)),
    ("atlas.receptor_type", "<u4", (4107,)),
    ("atlas.receptor_ptr", "<u4", (4108,)),
    ("atlas.site_indices", "<u4", (4669,)),
    ("atlas.site_weight", "<f4", (4669,)),
    ("atlas.body_rows", "<u4", (11233,)),
    ("atlas.neuron_type", "<u4", (165122,)),
    ("optic.spectral_logits", "<f4", (10, 3)),
    ("optic.gain_raw", "<f4", (10,)),
    ("optic.bias", "<f4", (10,)),
    ("body.mean", "<f4", (43,)),
    ("body.scale", "<f4", (43,)),
    ("body.input.weight", "<f4", (128, 43)),
    ("body.input.bias", "<f4", (128,)),
    ("body.output.weight", "<f4", (11233, 128)),
    ("body.output.bias", "<f4", (11233,)),
    ("dynamics.baseline_raw", "<f4", (11752,)),
    ("dynamics.recurrent_gain_raw", "<f4", (11752,)),
    ("dynamics.tau_raw", "<f4", (11752,)),
    ("dynamics.adaptation_gain_raw", "<f4", (11752,)),
    ("dynamics.adaptation_tau_raw", "<f4", (11752,)),
    ("afferent.neutral_drive", "<f4", (165122,)),
    ("readout.projection.weight", "<f4", (64, 165122)),
    ("readout.output.weight", "<f4", (512, 64)),
    ("readout.output.bias", "<f4", (512,)),
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_v2(path):
    with path.open("rb") as f:
        if f.read(8) != V2_MAGIC:
            raise ValueError("seed must be sealed CHCNS2")
        n = struct.unpack("<I", f.read(4))[0]
        meta = json.loads(f.read(n))
        offset = 12 + n
    if meta.get("format") != "chreatures-cns-service-v2":
        raise ValueError("seed metadata is not CHCNS2")
    arrays = {}
    for name, dtype, shape in V2_SPECS:
        arrays[name] = np.memmap(
            path, mode="r", offset=offset, dtype=dtype, shape=shape
        )
        offset += arrays[name].nbytes
        expected = meta.get("array_sha256", {}).get(name)
        if expected is None or hashlib.sha256(arrays[name]).hexdigest() != expected:
            raise ValueError(f"V2 tensor receipt differs: {name}")
    if path.stat().st_size != offset:
        raise ValueError("V2 trailing bytes")
    return arrays, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2-service", type=Path, required=True)
    p.add_argument("--anatomy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260907)
    args = p.parse_args()
    old, meta = load_v2(args.v2_service)
    with np.load(args.anatomy, allow_pickle=False) as z:
        anatomy = {k: np.asarray(z[k]) for k in z.files}
    anatomy_graph = str(anatomy.get("graph_sha256", ""))
    if anatomy_graph != meta.get("graph_sha256"):
        raise ValueError("V2 and anatomy graph identities differ")
    static = {
        k: old[k]
        for k in (
            "graph.crow",
            "graph.col",
            "atlas.receptor_rows",
            "atlas.receptor_type",
            "atlas.receptor_ptr",
            "atlas.site_indices",
            "atlas.site_weight",
            "atlas.neuron_type",
        )
    }
    for k in (
        "graph.weight",
        "graph.channel",
        "atlas.body_rows",
        "atlas.body_mask",
        "atlas.context_rows",
        "atlas.motor_rows",
        "atlas.motor_mask",
    ):
        static[k] = anatomy[k]
    arrays = initialized_arrays(static, args.seed)
    reused = (
        "optic.spectral_logits",
        "optic.gain_raw",
        "optic.bias",
        "dynamics.baseline_raw",
        "dynamics.recurrent_gain_raw",
        "dynamics.tau_raw",
        "dynamics.adaptation_gain_raw",
        "dynamics.adaptation_tau_raw",
        "readout.projection.weight",
        "readout.output.weight",
        "readout.output.bias",
    )
    for k in reused:
        arrays[k] = np.ascontiguousarray(old[k], dtype="<f4")
    from chreatures.cns_adapter_contract import neutral_afferent_drive

    arrays["afferent.neutral_drive"] = neutral_afferent_drive(arrays)
    receipt = write_service_artifact(
        args.output,
        arrays,
        graph_sha256=meta["graph_sha256"],
        atlas_sha256=meta["atlas_sha256"],
        anatomy_sha256=digest(args.anatomy),
        training_status="initialized-untrained",
        provenance={
            "source_v2_file_sha256": digest(args.v2_service),
            "source_v2_adapter_sha256": meta["adapter_sha256"],
            "migration": "one-way seed; no V2 runtime loader or private state migration",
            "reused": list(reused),
            "new_untrained_interfaces": [
                "body110 masked tuning",
                "context12 descending injection",
                "motor34 positive masked readout",
                "release resource",
                "three-family modulation",
            ],
        },
    )
    args.output.with_suffix(args.output.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "path": receipt["path"],
                "sha256": receipt["file_sha256"],
                "adapter_sha256": receipt["metadata"]["adapter_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
