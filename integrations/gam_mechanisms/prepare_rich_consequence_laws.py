#!/usr/bin/env python3
"""Extract the exact native 12-feature law contract from immutable rich play."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    manifest_path = args.collection / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    packets = sorted(args.collection.glob("episode-*.npz"))
    if len(packets) < 2:
        raise RuntimeError("rich successor fit requires at least two completed episodes")
    feature_rows, target_rows, units, slots, hashes = [], [], [], [], []
    residents_per_world = int(manifest["scope"]["residents_per_world"])
    worlds = int(manifest["scope"]["worlds"])
    for episode_index, packet in enumerate(packets):
        data = np.load(packet)
        observation = data["observation"]
        action = data["executed_actions"]
        oral = data["oral_command"]
        neural = data["neural_readouts"]
        physiology = observation[..., 4447:4453]
        if action.shape[1] != worlds * residents_per_world:
            raise RuntimeError(f"{packet.name} resident layout differs from manifest")
        state = physiology[:-1].astype(np.float64)
        action64, oral64 = action.astype(np.float64), oral.astype(np.float64)
        motor = np.abs(action64[..., :4]).mean(axis=-1)
        features = np.stack((state[..., 0], state[..., 2], state[..., 3], state[..., 5],
            neural[:-1].mean(axis=-1), action64[..., 0], action64[..., 1],
            action64[..., 3], oral64, motor, action64[..., 0] * state[..., 2],
            action64[..., 1] * state[..., 3]), axis=-1)
        targets = np.stack((physiology[1:, :, 3] - state[..., 3],
            state[..., 0] - physiology[1:, :, 0],
            state[..., 2] - physiology[1:, :, 2]), axis=-1)
        slot = np.arange(action.shape[1], dtype=np.int8) // residents_per_world
        feature_rows.append(features.reshape(-1, 12).astype(np.float32))
        target_rows.append(targets.reshape(-1, 3).astype(np.float32))
        slots.append(np.broadcast_to(slot, action.shape[:2]).reshape(-1))
        units.append(np.broadcast_to(episode_index * worlds + slot, action.shape[:2]).reshape(-1))
        hashes.append(sha256(packet))
    contract = {
        "status": "successor candidate for a new birth; does not alter running residents",
        "source_revision": args.source_revision,
        "collection_format": manifest.get("format"),
        "manifest_sha256": sha256(manifest_path),
        "manifest_content_sha256": manifest.get("content_sha256"),
        "graph_sha256": manifest.get("graph_sha256"),
        "port_spec_sha256": manifest.get("port_spec_sha256"),
        "world_profile_sha256": manifest.get("profile", {}).get("sha256"),
        "rich_profile_sha256": manifest.get("rich_sensorium", {}).get("profile_sha256"),
        "rich_channel_names_sha256": manifest.get("rich_sensorium", {}).get("channel_names_sha256"),
        "birth_receipts": manifest.get("birth_checkpoints"),
        "dt_seconds": float(manifest["scope"]["dt_seconds"]),
        "physiology_contract": "observation[...,4447:4453] = energy,gut,fatigue,tanh(speed/2),tanh(angular_velocity/4),support",
        "feature_contract": "native LawBank::fitted_features v1 exact 12-feature order",
        "target_contract": ["next[3]-now[3]", "now[0]-next[0]", "now[2]-next[2]"],
        "world_unit": f"{worlds} independently seeded physical worlds per episode; {residents_per_world} residents share each world",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.concatenate(feature_rows),
        outcomes=np.concatenate(target_rows), world_unit=np.concatenate(units),
        world_slot=np.concatenate(slots), heldout_world_slot=np.asarray(3, dtype=np.int8),
        source_sha256=np.asarray(hashes), source_contract=np.asarray(json.dumps(contract, sort_keys=True)))
    print(json.dumps({"rows": sum(len(x) for x in feature_rows), "episodes": len(packets),
        "worlds": worlds, "residents_per_world": residents_per_world,
        "heldout_world_slot": 3, "output_sha256": sha256(args.output),
        "source": contract}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
