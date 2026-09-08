#!/usr/bin/env python3
"""Seal FlyGym's FlyBodyPreprogrammedSteps as an offline teacher bank.

The generated arrays are reference trajectories only.  They are not loaded by
the recurring simulation and make no claim about measured muscle activation or
learned CNS control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

PIN = "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
BANK_ID = "flygym-2.1.0-ca65a510-flybody-steps-v1"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def periodic_linear(samples: np.ndarray, phase_cycles: np.ndarray) -> np.ndarray:
    count = len(samples)
    scaled = np.mod(phase_cycles, 1.0) * count
    lower = np.floor(scaled).astype(np.int64)
    fraction = scaled - lower
    upper = (lower + 1) % count
    return samples[lower] * (1.0 - fraction[:, None]) + samples[upper] * fraction[:, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flygym-source", type=Path, required=True)
    parser.add_argument("--body-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    args = parser.parse_args()
    if args.samples < 256:
        raise SystemExit("at least 256 phase samples are required")
    source = args.flygym_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        raise SystemExit(f"expected FlyGym {PIN}, found {revision}")
    sys.path.insert(0, str(source / "src"))

    from flygym_demo.complex_terrain.common import get_default_locomotion_dof_order
    from flygym_demo.complex_terrain.preprogrammed import FlyBodyPreprogrammedSteps

    schema_path = args.body_schema.resolve()
    body_schema = json.loads(schema_path.read_text())
    walking = [entry for entry in body_schema["actuators"] if entry["control_group"] == "walking"]
    if len(walking) != 42 or any(entry["index"] != index for index, entry in enumerate(walking)):
        raise RuntimeError("body schema does not begin with the frozen walking42 bank")
    author_dofs = {dof.name: dof for dof in get_default_locomotion_dof_order()}
    ordered_dofs = [author_dofs[entry["target"]] for entry in walking]

    teacher = FlyBodyPreprogrammedSteps()
    phase_cycles = np.arange(args.samples, dtype=np.float64) / args.samples
    phase_radians = phase_cycles * (2.0 * np.pi)
    joint_targets = np.empty((args.samples, 42), dtype=np.float64)
    adhesion = np.empty((args.samples, 6), dtype=np.uint8)
    for index, phase in enumerate(phase_radians):
        leg_phases = np.full(6, phase, dtype=np.float64)
        joint_targets[index] = teacher.get_joint_angles_by_dof_order(
            leg_phases, output_dof_order=ordered_dofs
        )
        adhesion[index] = teacher.get_adhesion_onoff_by_phase(leg_phases)

    neutral = np.asarray([entry["neutral_control"] for entry in walking], dtype=np.float64)
    teacher_neutral = np.asarray(
        teacher.get_joint_angles_by_dof_order(
            np.full(6, np.pi), output_dof_order=ordered_dofs
        ),
        dtype=np.float64,
    )
    ranges = np.asarray([entry["control_range"] for entry in walking], dtype=np.float64)
    delta = joint_targets - neutral
    scales = np.where(delta < 0, neutral - ranges[:, 0], ranges[:, 1] - neutral)
    normalized = delta / scales
    if np.any(joint_targets < ranges[:, 0]) or np.any(joint_targets > ranges[:, 1]):
        raise RuntimeError("author teacher exceeds physical servo control range")
    if np.max(np.abs(normalized)) > 1.0 + 1e-12:
        raise RuntimeError("author teacher exceeds normalized physical servo range")

    # Quantify the error introduced when a non-Python runtime linearly
    # interpolates the sealed bank instead of evaluating SciPy's cubic spline.
    rng = np.random.default_rng(20260908)
    probes = rng.random(8192)
    interpolated = periodic_linear(joint_targets, probes)
    exact = np.empty_like(interpolated)
    for index, phase in enumerate(probes * (2.0 * np.pi)):
        exact[index] = teacher.get_joint_angles_by_dof_order(
            np.full(6, phase), output_dof_order=ordered_dofs
        )
    interpolation_error = np.abs(exact - interpolated)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / "trajectory-bank.npz"
    np.savez_compressed(
        bank_path,
        phase_cycles=phase_cycles,
        joint_targets_rad=joint_targets,
        joint_targets_normalized=normalized,
        adhesion=adhesion,
        fixture_neutral_rad=neutral,
        teacher_neutral_rad=teacher_neutral,
        control_ranges_rad=ranges,
    )

    authored = source / "src/flygym_demo/complex_terrain"
    source_files = {
        "preprogrammed.py": authored / "preprogrammed.py",
        "common.py": authored / "common.py",
        "single_steps_flybody.npz": authored / "assets/single_steps_flybody.npz",
        "single_steps_flybody.meta.json": authored / "assets/single_steps_flybody.meta.json",
        "ball_flybody_clip.npz": source / "src/flygym_demo/ball_flybody_data/assets/ball_flybody_clip.npz",
        "FlyGym-LICENSE": source / "LICENSE",
    }
    source_dir = output / "author-source"
    source_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for name, path in source_files.items():
        source_hashes[name] = sha256(path)
        shutil.copy2(path, source_dir / name)

    manifest = {
        "format": "chreatures.author-step-bank.v1",
        "identity": BANK_ID,
        "purpose": "offline physical teacher/reference only; never a production gait controller or evidence of CNS learning",
        "source": {
            "project": "FlyGym / NeuroMechFly v2",
            "version": "2.1.0",
            "git_revision": revision,
            "class": "flygym_demo.complex_terrain.preprogrammed.FlyBodyPreprogrammedSteps",
            "license": "Apache-2.0",
            "files_sha256": source_hashes,
        },
        "body_schema_sha256": sha256(schema_path),
        "bank": {
            "file": bank_path.name,
            "sha256": sha256(bank_path),
            "sample_count": args.samples,
            "phase_cycles": {"shape": [args.samples], "dtype": "float64", "range": [0.0, 1.0], "endpoint": False},
            "joint_targets_rad": {"shape": [args.samples, 42], "dtype": "float64"},
            "joint_targets_normalized": {
                "shape": [args.samples, 42],
                "dtype": "float64",
                "range_observed": [float(normalized.min()), float(normalized.max())],
                "mapping": "(target-neutral)/(neutral-low) for target<neutral; (target-neutral)/(high-neutral) otherwise",
            },
            "adhesion": {"shape": [args.samples, 6], "dtype": "uint8", "values": [0, 1]},
            "fixture_neutral_rad": {"shape": [42], "dtype": "float64"},
            "teacher_neutral_rad": {"shape": [42], "dtype": "float64", "definition": "FlyBodyPreprogrammedSteps at phase pi and magnitude zero reference"},
            "interpolation": "periodic linear between adjacent samples",
            "maximum_interpolation_error_rad_8192_probes": float(interpolation_error.max()),
            "mean_interpolation_error_rad_8192_probes": float(interpolation_error.mean()),
        },
        "phase": {
            "zero": "posterior extreme position and start of swing",
            "author_nominal_cycle_s": float(teacher.duration),
            "author_nominal_frequency_hz": float(teacher.step_cycle_frequency_hz),
            "swing_period_cycles": {
                leg: [float(x / (2.0 * np.pi)) for x in teacher.swing_period[leg]] for leg in LEGS
            },
        },
        "orders": {
            "legs": list(LEGS),
            "walking_joint_ids": [entry["target"] for entry in walking],
            "walking_actuator_ids": [entry["id"] for entry in walking],
            "per_leg_slices": {leg: [index * 7, (index + 1) * 7] for index, leg in enumerate(LEGS)},
            "adhesion_actuator_ids": [entry["id"] for entry in body_schema["actuators"][84:90]],
        },
        "physical_mapping": {
            "walking_servo_indices": [0, 42],
            "remaining_servo_indices": [42, 84],
            "remaining_servo_command": "fixture neutral",
            "adhesion_indices": [84, 90],
            "control_rate_hz": 100,
            "physics_timestep_s": 0.0001,
            "substeps_per_control": 100,
        },
        "limits": [
            "The trajectories derive from a short NeuroMechFly v1 ball-walking clip and a hand-picked inverse-kinematic extraction, not measured motor-neuron or muscle activation.",
            "FlyBodyPreprogrammedSteps was fitted for the author's FlyBody model. Replaying it on the imported NeuroMechFly body is an engineered cross-morphology transfer through shared anatomical joint IDs.",
            "Left and right reuse the same F/M/H trajectory under FlyBody's symmetric joint-axis convention.",
            "Successful replay is a physical teacher diagnostic and does not demonstrate autonomous CNS control or faithful live-fly locomotion.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"bank": str(bank_path), "manifest": str(manifest_path), "bank_sha256": manifest["bank"]["sha256"], "max_interpolation_error_rad": manifest["bank"]["maximum_interpolation_error_rad_8192_probes"]}, indent=2))


if __name__ == "__main__":
    main()
