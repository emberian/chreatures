#!/usr/bin/env python3
"""Reusable acknowledged-step machinery for a future CNS-only rollout.

There is deliberately no active PPO packet schema in this wave.  The first
CNS adapter work is supervised/self-supervised, and an external teaching signal
has not yet been approved for policy optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chreatures.resident_contract import CONTROLLER_INPUT_FORMAT
from chreatures.sensorimotor_worker_native import DevelopmentalResidentCohort
from chreatures.sequence_control import ACTION_DIM, CNS_LATENT_DIM, CONTRACT_SHA256

# Must remain None until a CNS-derived teaching signal and packet schema are
# accepted together.  The old state668/raw-physiology packet is not supported.
ACTIVE_ROLLOUT_CONTRACT: str | None = None


class UncertainPhysicalMutation(RuntimeError):
    """The physical boundary may have committed and must never be retried."""


@dataclass(frozen=True)
class AcknowledgedStep:
    decision: dict[str, Any]
    acknowledgement: dict[str, Any]
    delivered_command: np.ndarray
    transition: Any


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                **{name: np.ascontiguousarray(value) for name, value in arrays.items()},
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def source_identity() -> dict[str, Any]:
    paths = (
        Path("research/sensorimotor_skills/native_sequence_rollout.py"),
        Path("chreatures/sensorimotor_worker_native.py"),
        Path("chreatures/sequence_control.py"),
        Path("chreatures/resident_contract.py"),
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "revision": revision,
        "files": {
            str(path): {
                "bytes": (ROOT / path).stat().st_size,
                "sha256": sha256(ROOT / path),
            }
            for path in paths
        },
    }


def control_identity(residents: DevelopmentalResidentCohort) -> dict[str, Any]:
    value = residents.model_identity.get("sequence_control")
    if (
        not isinstance(value, dict)
        or value.get("contract_sha256") != CONTRACT_SHA256
        or type(value.get("version")) is not int
        or not isinstance(value.get("artifact_sha256"), str)
    ):
        raise RuntimeError("resident sequence-control identity differs")
    return dict(value)


def run_acknowledged_step(
    residents: DevelopmentalResidentCohort,
    *,
    cns_latent: np.ndarray,
    previous_command: np.ndarray,
    ticks: np.ndarray,
    reset: np.ndarray,
    advance: Callable[[np.ndarray], tuple[np.ndarray, Any]],
) -> AcknowledgedStep:
    """Execute exactly one decision and explicit physical receipt.

    ``advance`` is the existing physical barrier, not a simulation callback. It
    returns the command the world actually accepted and its external transition
    payload.  Any exception after entry to that barrier is treated as an
    uncertain mutation and must be resolved from a whole-system checkpoint.
    """
    z = np.ascontiguousarray(cns_latent, dtype=np.float32)
    previous = np.ascontiguousarray(previous_command, dtype=np.float32)
    tick_array = np.ascontiguousarray(ticks, dtype=np.uint64)
    reset_array = np.ascontiguousarray(reset, dtype=np.bool_)
    if z.ndim != 2 or z.shape[1:] != (CNS_LATENT_DIM,):
        raise ValueError("cns_latent must be [B,512]")
    if previous.shape != (len(z), ACTION_DIM) or tick_array.shape != (len(z),):
        raise ValueError("previous command or tick batch differs")
    if reset_array.shape != (len(z),):
        raise ValueError("reset batch differs")
    decision = residents.step(z, previous, tick_array, reset_array)
    proposed = np.ascontiguousarray(decision["proposed_command"], dtype=np.float32)
    try:
        delivered, transition = advance(proposed.copy())
        delivered = np.ascontiguousarray(delivered, dtype=np.float32)
        if delivered.shape != proposed.shape or not np.isfinite(delivered).all():
            raise RuntimeError("physical barrier returned an invalid delivered command")
        acknowledgement = residents.acknowledge(tick_array, delivered)
    except BaseException as exc:
        raise UncertainPhysicalMutation(
            "physical advance or command acknowledgement is uncertain; do not retry"
        ) from exc
    return AcknowledgedStep(
        decision=decision,
        acknowledgement=acknowledgement,
        delivered_command=delivered,
        transition=transition,
    )


def preview_next_boundary(
    residents: DevelopmentalResidentCohort,
    *,
    cns_latent: np.ndarray,
    previous_command: np.ndarray,
    ticks: np.ndarray,
    reset: np.ndarray,
) -> dict[str, Any]:
    """Read the exact next head inputs from a private native clone."""
    return residents.preview_sequence_control(
        cns_latent, previous_command, ticks, reset
    )


def main() -> int:
    raise SystemExit(
        "No active CNS-only PPO rollout contract: train the CNS service adapter "
        "with its approved supervised/self-supervised pipeline. The withdrawn "
        "state668/raw-physiology packet cannot be collected."
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_ROLLOUT_CONTRACT",
    "AcknowledgedStep",
    "CONTROLLER_INPUT_FORMAT",
    "UncertainPhysicalMutation",
    "atomic_json",
    "atomic_npz",
    "canonical_hash",
    "control_identity",
    "preview_next_boundary",
    "run_acknowledged_step",
    "source_identity",
]
