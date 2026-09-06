#!/usr/bin/env python3
"""Explicitly change a paused world's execution kernel without changing state.

Normal restore rejects a kernel mismatch. This separate, audited operation
copies every native state byte, changes only the declared future reduction
kernel, and writes a new world manifest. Original artifacts remain intact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chreatures.metal_circuit import SAFE_NAME
from chreatures.runtime import canonical


def migrate(source, output, snapshots, target_kernel, physics_backend=None):
    source, output, snapshots = Path(source), Path(output), Path(snapshots)
    if output.exists():
        raise ValueError(
            "Output must be a new manifest; original lives are never overwritten"
        )
    envelope = json.loads(source.read_text())
    if envelope.get("format") != "chreatures-3d-checkpoint-v1":
        raise ValueError("Expected a whole 3D world checkpoint")
    state = envelope["state"]
    source_hash = hashlib.sha256(canonical(state)).hexdigest()
    if source_hash != envelope["sha256"] or state.get("paused") is not True:
        raise ValueError("Source must be an authenticated, paused world")
    receipt = state["neural_snapshot"]
    if not SAFE_NAME.fullmatch(receipt["name"]):
        raise ValueError("Unsafe source snapshot name")
    native = (snapshots / (receipt["name"] + ".npz")).read_bytes()
    if (
        hashlib.sha256(native).hexdigest() != receipt["sha256"]
        or native[:8] != b"MBST1\0\0\0"
    ):
        raise ValueError("Native state artifact does not match the world's receipt")
    length = struct.unpack("<Q", native[8:16])[0]
    if length > 2_000_000:
        raise ValueError("Native metadata is too large")
    metadata = json.loads(native[16 : 16 + length])
    payload = native[16 + length :]
    if (
        metadata.get("version") not in (2, 3)
        or metadata.get("kernel") not in ("row", "simd")
        or metadata.get("graph_sha256") != state["graph_sha256"]
        or target_kernel not in ("row", "simd")
    ):
        raise ValueError("Unsupported source/target execution contract")
    if (
        len(payload) != 165122 * 4 * 4 * 3
        or not np.isfinite(np.frombuffer(payload, dtype="<f4")).all()
    ):
        raise ValueError(
            "Native rate/adaptation/support payload has invalid dimensions or values"
        )
    if [x for x in metadata["resident_ids"] if x is not None] != receipt["residents"]:
        raise ValueError("Resident slot order differs from its world receipt")
    migration = {
        "kind": "execution-backend-migration-v1",
        "world_id": state["id"],
        "tick": state["tick"],
        "source_world_sha256": source_hash,
        "source_neural_sha256": receipt["sha256"],
        "native_state_sha256": hashlib.sha256(payload).hexdigest(),
        "from_kernel": metadata["kernel"],
        "to_kernel": target_kernel,
        "from_physics": state.get("physics_backend", "reference"),
        "to_physics": physics_backend or state.get("physics_backend", "reference"),
        "state_changes": "none; every rate, adaptation, support and organism/world state byte is retained",
        "future_numerics": "Metal reduction order may differ in float32; vectorized physical equations are unchanged",
    }
    metadata = copy.deepcopy(metadata)
    metadata["kernel"] = target_kernel
    metadata["execution_migration"] = migration
    encoded = canonical(metadata)
    migrated_native = native[:8] + struct.pack("<Q", len(encoded)) + encoded + payload
    target_name = f"migration-{source_hash[:20]}-{target_kernel}"
    target_path = snapshots / (target_name + ".npz")
    if target_path.exists() and target_path.read_bytes() != migrated_native:
        raise ValueError("Conflicting migration artifact already exists")
    if not target_path.exists():
        with target_path.open("xb") as stream:
            stream.write(migrated_native)
            stream.flush()
            import os

            os.fsync(stream.fileno())
    changed = copy.deepcopy(state)
    changed["neural_snapshot"] = {
        **receipt,
        "name": target_name,
        "sha256": hashlib.sha256(migrated_native).hexdigest(),
        "bytes": len(migrated_native),
    }
    if physics_backend is not None:
        if (
            physics_backend not in ("reference", "vectorized")
            or state.get("body_mode") != "articulated"
        ):
            raise ValueError("Physical backend migration requires an articulated body")
        changed["physics_backend"] = physics_backend
    changed.setdefault("execution_migrations", []).append(migration)
    changed["journal"].append(
        {
            "id": f"{state['id']}:{state['tick']}:execution-migration",
            "time": float(state["world"]["mj_state"][0]),
            "kind": "implementation",
            "text": "Execution backend changed with all model state retained.",
            "migration": migration,
        }
    )
    changed["journal"] = changed["journal"][-256:]
    digest = hashlib.sha256(canonical(changed)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(
            canonical(
                {"format": envelope["format"], "sha256": digest, "state": changed}
            )
        )
        stream.flush()
        import os

        os.fsync(stream.fileno())
    return {
        "output": str(output),
        "sha256": digest,
        "migration": migration,
        "neural_snapshot": changed["neural_snapshot"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--kernel", choices=("row", "simd"), required=True)
    parser.add_argument("--physics-backend", choices=("reference", "vectorized"))
    args = parser.parse_args()
    print(
        json.dumps(
            migrate(
                args.source,
                args.output,
                args.snapshot_dir,
                args.kernel,
                args.physics_backend,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
