#!/usr/bin/env python3
"""Export an exact Torch full-CNS recurrence fixture for native parity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.sensorimotor_skills.cns_adapter import (
    BODY_CHANNELS,
    CNSState,
    CNSStaticArrays,
    OPTIC_CHANNELS,
    OPTIC_SITES,
    PARAMETER_ARTIFACT_FORMAT,
    TrainableCNSAdapter,
    file_sha256,
    load_export_arrays,
    parameter_artifact_identity,
)


FORMAT = "chreatures-cns-torch-parity-fixture-v2"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sensory_packet(atlas: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(atlas, allow_pickle=False) as bundle:
        locations = np.asarray(bundle["site_side_hex"], dtype=np.float32)
    if locations.shape != (OPTIC_SITES, 3):
        raise ValueError("optic atlas site order differs")
    _, q, r = locations.T
    x = q + 0.5 * r
    x = (x - x.mean()) / max(float(x.std()), 1e-6)
    x = torch.from_numpy(x.astype(np.float32)).to(device)

    optic = torch.zeros((4, 2, OPTIC_SITES, OPTIC_CHANNELS), device=device)
    optic[1] = 1.0
    centers = (-0.65, 0.55)
    colors = torch.tensor(((1.0, 0.25, 0.05), (0.10, 0.45, 1.0)), device=device)
    for batch, center in enumerate(centers):
        stripe = torch.exp(-0.5 * ((x - center) / 0.28).square())
        optic[2, batch] = stripe[:, None] * colors[batch]
        shifted = torch.exp(-0.5 * ((x - center - 0.35) / 0.28).square())
        optic[3, batch] = shifted[:, None] * colors[batch]

    channel = torch.linspace(-0.35, 0.35, BODY_CHANNELS, device=device)
    body = torch.stack(
        (
            torch.stack((channel * 0.00, channel * 0.15)),
            torch.stack((channel * 0.20, channel * -0.10)),
            torch.stack((channel * -0.25, channel * 0.30)),
            torch.stack((channel * 0.35, channel * -0.20)),
        )
    )
    return optic.to(torch.float32), body.to(torch.float32)


def main() -> None:
    args = arguments()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    static = CNSStaticArrays.load(args.graph, args.atlas)
    model = TrainableCNSAdapter(static, device=device)
    with np.load(args.parameters, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("parameter metadata is missing")
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name != "metadata"
        }
    if (
        metadata.get("format") != PARAMETER_ARTIFACT_FORMAT
        or metadata.get("artifact_sha256")
        != parameter_artifact_identity(metadata, arrays)
        or metadata.get("static_identity") != static.identity
    ):
        raise ValueError("parameter artifact identity or static substrate differs")
    load_export_arrays(model, arrays)
    model.eval()
    optic, body = sensory_packet(args.atlas, device)
    initial = model.initial_state(2)
    with torch.no_grad():
        first_drive = model.afferent_drive(optic[0], body[0])
        latent, final = model.forward_sequence(
            optic,
            body,
            torch.zeros((4, 2), dtype=torch.bool, device=device),
            initial,
            checkpoint_ticks=4,
        )
    sensory = torch.cat((optic.flatten(2), body), dim=-1)
    fixture_metadata = {
        "format": FORMAT,
        "parameter_path": str(args.parameters.resolve()),
        "parameter_file_sha256": file_sha256(args.parameters),
        "parameter_artifact_sha256": metadata["artifact_sha256"],
        "training_status": metadata["training_status"],
        "static_identity": static.identity,
        "sensory_order": "optic[1771,RGB] row-major then body43",
        "sequence": "black, white, colored stripe, shifted colored stripe",
        "state_orientation": "neurons,batch",
        "source_sha256": file_sha256(Path(__file__)),
    }
    values = {
        "metadata": np.asarray(
            json.dumps(fixture_metadata, sort_keys=True, separators=(",", ":"))
        ),
        "sensory": sensory.cpu().numpy().astype("<f4"),
        "initial_rates": initial.rates.cpu().numpy().astype("<f4"),
        "initial_adaptation": initial.adaptation.cpu().numpy().astype("<f4"),
        "initial_support": initial.support.cpu().numpy().astype("<f4"),
        "first_drive": first_drive.cpu().numpy().astype("<f4"),
        "latent": latent.cpu().numpy().astype("<f4"),
        "final_rates": final.rates.cpu().numpy().astype("<f4"),
        "final_adaptation": final.adaptation.cpu().numpy().astype("<f4"),
        "final_support": final.support.cpu().numpy().astype("<f4"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "file_sha256": file_sha256(output),
                "parameter_artifact_sha256": metadata["artifact_sha256"],
                "training_status": metadata["training_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
