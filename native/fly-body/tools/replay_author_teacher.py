#!/usr/bin/env python3
"""Replay the sealed author gait bank in the actual B4 MuJoCo scene.

This is a one-off physical feasibility diagnostic.  It is not a production
controller and does not pass world state around the CNS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
MODES = ("forward", "turn_left", "stop", "turn_right")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def periodic_leg_sample(samples: np.ndarray, phases: np.ndarray) -> np.ndarray:
    count = len(samples)
    result = np.empty(42, dtype=np.float64)
    for leg, phase in enumerate(phases):
        scaled = (phase % 1.0) * count
        lower = int(math.floor(scaled))
        upper = (lower + 1) % count
        fraction = scaled - lower
        section = slice(leg * 7, (leg + 1) * 7)
        result[section] = samples[lower, section] * (1.0 - fraction) + samples[upper, section] * fraction
    return result


def periodic_adhesion(samples: np.ndarray, phases: np.ndarray) -> np.ndarray:
    indices = np.floor(np.mod(phases, 1.0) * len(samples)).astype(np.int64)
    return np.asarray([samples[index, leg] for leg, index in enumerate(indices)], dtype=np.float64)


def yaw(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation).reshape(3, 3)
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=here / "scenes/training-4/scene.xml")
    parser.add_argument("--physics", type=Path, default=here / "scenes/training-4/physics.json")
    parser.add_argument("--bank", type=Path, default=here / "assets/author-step-bank-v1/trajectory-bank.npz")
    parser.add_argument("--bank-manifest", type=Path, default=here / "assets/author-step-bank-v1/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--settle", type=float, default=0.1)
    parser.add_argument("--frequency", type=float, default=12.0)
    parser.add_argument("--adhesion-gain", type=float, default=40.0)
    args = parser.parse_args()

    scene = args.scene.resolve()
    physics = json.loads(args.physics.read_text())
    bank_manifest = json.loads(args.bank_manifest.read_text())
    if sha256(scene) != physics["source_mjcf_sha256"]:
        raise RuntimeError("scene hash differs from compiled physics manifest")
    if sha256(args.bank) != bank_manifest["bank"]["sha256"]:
        raise RuntimeError("teacher trajectory bank hash differs")
    bank = np.load(args.bank, allow_pickle=False)
    targets = np.asarray(bank["joint_targets_rad"], dtype=np.float64)
    teacher_neutral = np.asarray(bank["teacher_neutral_rad"], dtype=np.float64)

    model = mj.MjModel.from_xml_path(str(scene))
    if len(physics["bodies"]) != 4 or model.nu != 360:
        raise RuntimeError("the feasibility demonstration requires the pinned B4 scene")
    if abs(float(model.opt.timestep) - 0.0001) > 1e-15:
        raise RuntimeError("unexpected physics timestep")
    substeps = 100
    control_dt = substeps * float(model.opt.timestep)
    ticks = round(args.duration / control_dt)
    settle_ticks = round(args.settle / control_dt)
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)

    bodies = physics["bodies"]
    detailed = physics["residents"]
    for body in bodies:
        adhesion_ids = body["actuators"][84:90]
        model.actuator_gainprm[adhesion_ids, 0] = args.adhesion_gain
        data.ctrl[body["actuators"][:84]] = body["neutral"]
        data.ctrl[adhesion_ids] = 1.0
    for _ in range(settle_ticks * substeps):
        mj.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        raise RuntimeError("non-finite state during neutral settling")

    initial_position = np.asarray([data.xpos[body["root"]].copy() for body in bodies])
    initial_rotation = np.asarray([data.xmat[body["root"]].copy().reshape(3, 3) for body in bodies])
    initial_yaw = np.asarray([yaw(matrix) for matrix in initial_rotation])
    control_trace = np.empty((ticks, 4, 90), dtype=np.float64)
    walking_trace = np.empty((ticks, 4, 42), dtype=np.float64)
    contact_trace = np.empty((ticks, 4, 6, 16), dtype=np.float64)
    root_pose = np.empty((ticks + 1, 4, 7), dtype=np.float64)
    joint_qpos = np.empty((ticks + 1, 4, 126), dtype=np.float64)

    def capture(index: int) -> None:
        for resident, body in enumerate(bodies):
            root_pose[index, resident, :3] = data.xpos[body["root"]]
            root_pose[index, resident, 3:] = data.xquat[body["root"]]
            joint_qpos[index, resident] = data.qpos[body["qpos"]]

    capture(0)
    tripod = np.asarray([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    magnitudes = (
        np.ones(6),
        np.asarray([0.35, 0.35, 0.35, 1.0, 1.0, 1.0]),
        np.zeros(6),
        np.asarray([1.0, 1.0, 1.0, 0.35, 0.35, 0.35]),
    )
    for tick in range(ticks):
        phases = tripod + args.frequency * tick * control_dt
        blend = min(1.0, (tick + 1) * control_dt / 0.25)
        sampled = periodic_leg_sample(targets, phases)
        sampled_adhesion = periodic_adhesion(bank["adhesion"], phases)
        for resident, (body, mode, magnitude) in enumerate(zip(bodies, MODES, magnitudes)):
            controls = np.zeros(90, dtype=np.float64)
            controls[:84] = body["neutral"]
            if mode == "stop":
                walking = np.asarray(body["neutral"][:42], dtype=np.float64)
                adhesion = np.ones(6, dtype=np.float64)
            else:
                expanded = np.repeat(magnitude, 7)
                desired = teacher_neutral + expanded * (sampled - teacher_neutral)
                walking = np.asarray(body["neutral"][:42]) + blend * (desired - np.asarray(body["neutral"][:42]))
                adhesion = sampled_adhesion if blend >= 1.0 else np.ones(6, dtype=np.float64)
            controls[:42] = walking
            controls[84:90] = adhesion
            actuator_ids = body["actuators"]
            ranges = np.asarray([model.actuator_ctrlrange[index] for index in actuator_ids])
            if np.any(controls < ranges[:, 0]) or np.any(controls > ranges[:, 1]):
                raise RuntimeError(f"{mode} target outside physical control range")
            data.ctrl[actuator_ids] = controls
            control_trace[tick, resident] = controls
            walking_trace[tick, resident] = walking
        for _ in range(substeps):
            mj.mj_step(model, data)
        if not all(np.all(np.isfinite(array)) for array in (data.qpos, data.qvel, data.sensordata)):
            raise RuntimeError(f"non-finite state after control tick {tick}")
        for resident, sensors in enumerate(detailed):
            for leg, sensor in enumerate(sensors["contact_sensors6x16"]):
                address = sensor["data_address"]
                contact_trace[tick, resident, leg] = data.sensordata[address:address + 16]
        capture(tick + 1)

    final_position = root_pose[-1, :, :3]
    final_yaw = np.asarray([yaw(data.xmat[body["root"]]) for body in bodies])
    outcomes = []
    for resident, mode in enumerate(MODES):
        displacement = final_position[resident] - initial_position[resident]
        forward_axis = initial_rotation[resident, :, 0]
        left_axis = initial_rotation[resident, :, 1]
        outcomes.append(
            {
                "resident": resident,
                "mode": mode,
                "displacement_world_mm": [float(x) for x in displacement],
                "forward_displacement_mm": float(displacement @ forward_axis),
                "lateral_displacement_mm": float(displacement @ left_axis),
                "yaw_change_rad": float(math.atan2(math.sin(final_yaw[resident] - initial_yaw[resident]), math.cos(final_yaw[resident] - initial_yaw[resident]))),
                "root_height_range_mm": [float(root_pose[:, resident, 2].min()), float(root_pose[:, resident, 2].max())],
                "leg_contact_found_fraction": [float(x) for x in (contact_trace[:, resident, :, 0] > 0).mean(axis=0)],
                "mean_contact_force_model_units": float(np.linalg.norm(contact_trace[:, resident, :, 1:4], axis=-1).mean()),
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "teacher-replay-trace.npz"
    np.savez_compressed(
        trace_path,
        applied_controls=control_trace,
        walking_targets_rad=walking_trace,
        raw_contact_sensors=contact_trace,
        root_pose_world=root_pose,
        joint_qpos_rad=joint_qpos,
    )
    receipt = {
        "format": "chreatures.author-teacher-physical-feasibility.v1",
        "status": "executed",
        "engine": "mujoco-3.12.0-native",
        "scene": str(args.scene),
        "scene_sha256": physics["source_mjcf_sha256"],
        "teacher_bank_sha256": bank_manifest["bank"]["sha256"],
        "source_revision": bank_manifest["source"]["git_revision"],
        "body_schema_sha256": physics["body_schema_sha256"],
        "residents": 4,
        "modes": list(MODES),
        "frequency_hz": args.frequency,
        "duration_s": args.duration,
        "settle_s": args.settle,
        "physics_dt_s": float(model.opt.timestep),
        "control_dt_s": control_dt,
        "substeps_per_control": substeps,
        "control_ticks": ticks,
        "fixture_servo_kp": 50.0,
        "adhesion_gain_in_memory": args.adhesion_gain,
        "nonwalking_servo_command": "all 42 remaining servos held at per-resident fixture neutral",
        "trace": {
            "file": trace_path.name,
            "sha256": sha256(trace_path),
            "applied_controls_shape": list(control_trace.shape),
            "walking_targets_shape": list(walking_trace.shape),
            "raw_contact_shape": list(contact_trace.shape),
            "joint_qpos_shape": list(joint_qpos.shape),
        },
        "outcomes": outcomes,
        "observed_checks": {
            "forward_displacement_positive": outcomes[0]["forward_displacement_mm"] > 0.0,
            "stop_translation_mm": float(np.linalg.norm(final_position[2] - initial_position[2])),
            "asymmetric_turn_commands_produced_opposite_yaw_signs": outcomes[1]["yaw_change_rad"] * outcomes[3]["yaw_change_rad"] < 0.0,
        },
        "finite": True,
        "interpretation": "Research-only author teacher replay. Displacement and contact show behavior of this engineered model/controller pairing; they do not establish faithful fly locomotion or learned CNS control.",
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
