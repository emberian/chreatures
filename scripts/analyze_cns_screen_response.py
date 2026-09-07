#!/usr/bin/env python3
"""Fail-closed paired analysis for completed CNS screen-response recordings.

This is descriptive analysis of one matched seed.  It neither advances a world
nor supplies any observation to a controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "chreatures-cns-screen-comparison-v1"
THRESHOLD = 1e-6
ARRAYS = ("neural_rates", "cns_latents", "commands", "positions", "time_seconds")
EXPECTED_SHAPES = {
    "neural_rates": (600, 165122),
    "cns_latents": (600, 8, 512),
    "commands": (600, 8, 12),
    "positions": (600, 8, 3),
    "time_seconds": (600,),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_completed_bundle(directory: Path, expected_condition: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    receipt_path = directory / "response.receipt.json"
    require(receipt_path.is_file(), f"{expected_condition}: missing completion receipt")
    receipt = load_json(receipt_path)
    intent = load_json(directory / "intent.json")
    require(receipt.get("format") == "chreatures-cns-screen-response-v1", f"{expected_condition}: unexpected receipt format")
    require(receipt.get("status") == "completed", f"{expected_condition}: receipt is not completed")
    require(receipt.get("condition") == expected_condition, f"{expected_condition}: receipt condition mismatch")
    require(intent.get("condition") == expected_condition, f"{expected_condition}: intent condition mismatch")
    require(intent.get("format") == "chreatures-cns-screen-assay-v1", f"{expected_condition}: unexpected intent format")
    require(receipt.get("frames") == 600 and intent.get("frames") == 600, f"{expected_condition}: expected 600 frames")
    require(receipt.get("world_source_revision") == intent.get("source_revision"), f"{expected_condition}: source revision mismatch")
    require(receipt.get("controller_sha256") == intent.get("controller_sha256"), f"{expected_condition}: controller hash mismatch")
    restore = receipt.get("whole_loop_restore")
    require(isinstance(restore, dict) and bool(restore), f"{expected_condition}: missing whole-loop restore checks")
    require(all(value is True for value in restore.values()), f"{expected_condition}: whole-loop restore is not exact")

    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {"response_receipt": sha256(receipt_path), "intent": sha256(directory / "intent.json")}
    for name in ARRAYS:
        path = directory / f"{name}.npy"
        require(path.is_file(), f"{expected_condition}: missing {path.name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        require(values.shape == EXPECTED_SHAPES[name], f"{expected_condition}: {name} shape is {values.shape}, expected {EXPECTED_SHAPES[name]}")
        require(np.issubdtype(values.dtype, np.number), f"{expected_condition}: {name} is non-numeric")
        arrays[name] = values
        hashes[name] = sha256(path)
    for name, receipt_key in (("neural_rates", "rate_capture_sha256"), ("commands", "commands_sha256"), ("positions", "positions_sha256")):
        require(receipt.get(receipt_key) == hashes[name], f"{expected_condition}: {name} hash does not match receipt")
    return receipt, intent, arrays, hashes


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def rms(a: np.ndarray, b: np.ndarray) -> float:
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def first_different_frame(a: np.ndarray, b: np.ndarray) -> int | None:
    difference = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    changed = np.any(difference.reshape((difference.shape[0], -1)) > THRESHOLD, axis=1)
    indices = np.flatnonzero(changed)
    return None if len(indices) == 0 else int(indices[0])


def public_identity(receipt: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    cns_identity = receipt.get("cns_identity")
    execution = receipt.get("neural_execution")
    require(isinstance(cns_identity, dict), "receipt missing CNS model identity")
    require(isinstance(execution, dict), "receipt missing neural execution identity")
    return {
        "source_revision": receipt["world_source_revision"],
        "controller_sha256": receipt["controller_sha256"],
        "service_sha256": intent["service_sha256"],
        "cns_model_identity_sha256": canonical_sha256(cns_identity),
        "neural_execution_identity_sha256": canonical_sha256(execution),
        "cns_graph_sha256": cns_identity.get("graph_sha256"),
        "seed": intent["seed"],
        "warmup_ticks": intent["warmup_ticks"],
        "dt_seconds": intent["dt"],
        "frame_count": intent["frames"],
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--blank", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, default=Path("data/ports/optic-anatomy-audit-v1.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clip_receipt, clip_intent, clip, clip_hashes = load_completed_bundle(args.clip, "clip")
    blank_receipt, blank_intent, blank, blank_hashes = load_completed_bundle(args.blank, "blank")
    identity = public_identity(clip_receipt, clip_intent)
    require(identity == public_identity(blank_receipt, blank_intent), "paired bundles do not share source/model/controller identity")
    require(clip_intent.get("stimulus_sha256") == blank_intent.get("stimulus_sha256"), "paired bundles do not share stimulus source")
    require(np.array_equal(clip["time_seconds"], blank["time_seconds"]), "paired bundles have different time samples")
    require(np.isfinite(clip["time_seconds"]).all(), "clip has non-finite time samples")
    require(np.all(np.diff(clip["time_seconds"]) > 0), "time samples are not strictly increasing")

    with np.load(args.atlas, allow_pickle=False) as atlas:
        photo = np.asarray(atlas["photoreceptor_graph_rows"], dtype=np.int64)
    require(photo.ndim == 1 and len(photo) > 0 and np.all((0 <= photo) & (photo < 165122)), "invalid photoreceptor graph rows")
    require(len(np.unique(photo)) == len(photo), "photoreceptor graph rows are not unique")
    photo_mask = np.zeros(165122, dtype=bool)
    photo_mask[photo] = True

    first_frame = {
        name: {"exact": bool(np.array_equal(clip[name][0], blank[name][0])), "max_abs_difference": max_abs(clip[name][0], blank[name][0]),
               "equivalent_at_threshold": max_abs(clip[name][0], blank[name][0]) <= THRESHOLD}
        for name in ("neural_rates", "cns_latents", "commands", "positions")
    }
    neural_difference = np.abs(np.asarray(clip["neural_rates"], dtype=np.float64) - np.asarray(blank["neural_rates"], dtype=np.float64))
    ever_different = np.any(neural_difference > THRESHOLD, axis=0)
    resident_metrics = []
    for resident in range(8):
        trajectory_delta = np.asarray(clip["positions"][:, resident], dtype=np.float64) - np.asarray(blank["positions"][:, resident], dtype=np.float64)
        resident_metrics.append({
            "resident": resident,
            "latent_rms_difference": rms(clip["cns_latents"][:, resident], blank["cns_latents"][:, resident]),
            "action_rms_difference": rms(clip["commands"][:, resident], blank["commands"][:, resident]),
            "final_position_distance": float(np.linalg.norm(trajectory_delta[-1])),
            "max_trajectory_separation": float(np.max(np.linalg.norm(trajectory_delta, axis=1))),
            "onset_first_different_frame": {
                "latent": first_different_frame(clip["cns_latents"][:, resident], blank["cns_latents"][:, resident]),
                "actions": first_different_frame(clip["commands"][:, resident], blank["commands"][:, resident]),
                "positions": first_different_frame(clip["positions"][:, resident], blank["positions"][:, resident]),
            },
        })
    output = {
        "format": FORMAT,
        "analysis_scope": "one paired seed; descriptive response comparison only",
        "interpretation_limits": [
            "This is one paired seed, not independent trials.",
            "The controller was initialized and untrained; these measurements do not establish learning or behavioral benefit.",
            "No position or physiological input was supplied to any controller by this analysis.",
        ],
        "thresholds": {"difference_absolute": THRESHOLD, "onset_rule": "first frame with any absolute elementwise difference greater than threshold"},
        "shared_identity": identity,
        "whole_loop_restore_exact": clip_receipt["whole_loop_restore"],
        "input_file_sha256": {"clip": clip_hashes, "blank": blank_hashes, "optic_anatomy_audit": sha256(args.atlas)},
        "first_frame_equivalence": first_frame,
        "onset_first_different_frame": {"neural_rates": first_different_frame(clip["neural_rates"], blank["neural_rates"])},
        "neurons_ever_different": {
            "threshold": THRESHOLD,
            "total": int(ever_different.sum()),
            "photoreceptor_rows": int(ever_different[photo_mask].sum()),
            "other_rows": int(ever_different[~photo_mask].sum()),
            "photoreceptor_row_count": int(photo_mask.sum()),
            "other_row_count": int((~photo_mask).sum()),
        },
        "per_resident": resident_metrics,
    }
    atomic_json(args.output, output)
    print(json.dumps({"output": args.output.name, "sha256": sha256(args.output), "residents": len(resident_metrics)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
