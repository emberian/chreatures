#!/usr/bin/env python3
"""Build-time export of the pinned author NeuroMechFly body.

Python and FlyGym are deliberately confined to this asset conversion step. The
exported MJCF is loaded directly by MuJoCo in native and WebAssembly runtimes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

PIN = "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
IDENTITY = "neuromechfly-2.1.0-ca65a510-ypr"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def short_name(value: str) -> str:
    return value.split("/", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flygym-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.flygym_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        raise SystemExit(f"expected FlyGym {PIN}, found {revision}")
    sys.path.insert(0, str(source / "src"))

    import mujoco as mj
    import numpy as np
    from flygym import assets_dir
    from flygym.anatomy import (
        ActuatedDOFPreset,
        AxisOrder,
        ContactBodiesPreset,
        JointPreset,
        Skeleton,
    )
    from flygym.compose import (
        ActuatorType,
        FlatGroundWorld,
        KinematicPosePreset,
        NeuroMechFly,
    )
    from flygym.utils.math import Rotation3D

    if importlib.metadata.version("flygym") != "2.1.0":
        raise SystemExit("the build environment must resolve FlyGym 2.1.0")

    fly = NeuroMechFly(name="fly")
    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.YAW_PITCH_ROLL,
    )
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    walking = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    all_dofs = list(skeleton.iter_jointdofs())
    active_groups = {
        "walking": walking,
        "head": [d for d in all_dofs if d.child.name == "c_head"],
        "pedicels": [d for d in all_dofs if d.child.name in {"l_pedicel", "r_pedicel"}],
        "proboscis": [d for d in all_dofs if d.child.name in {"c_rostrum", "c_haustellum"}],
        "abdomen": [d for d in all_dofs if d.child.name.startswith("c_abdomen")],
        "wings": [d for d in all_dofs if d.child.name in {"l_wing", "r_wing"}],
        "halteres": [d for d in all_dofs if d.child.name in {"l_haltere", "r_haltere"}],
    }
    expected_group_sizes = {
        "walking": 42,
        "head": 3,
        "pedicels": 6,
        "proboscis": 6,
        "abdomen": 15,
        "wings": 6,
        "halteres": 6,
    }
    actual_group_sizes = {name: len(dofs) for name, dofs in active_groups.items()}
    if actual_group_sizes != expected_group_sizes:
        raise RuntimeError(f"active anatomical group mismatch: {actual_group_sizes}")
    actuated = [dof for group in active_groups.values() for dof in group]
    if len(actuated) != 84 or len(set(actuated)) != 84:
        raise RuntimeError("active anatomy must contain 84 distinct joint DOFs")
    control_group_by_dof = {
        dof: group for group, dofs in active_groups.items() for dof in dofs
    }
    fly.add_actuators(
        actuated,
        actuator_type=ActuatorType.POSITION,
        neutral_input=KinematicPosePreset.NEUTRAL,
        kp=50.0,
        ctrlrange=(-3.14, 3.14),
    )
    fly.add_leg_adhesion(gain=1.0)
    fly.add_joint_sites(JointPreset.ALL_BIOLOGICAL.to_joint_list())
    fly.add_vision()
    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")

    world = FlatGroundWorld(name="chreatures_fly_body", half_size=20)
    world.add_fly(
        fly,
        spawn_position=(0, 0, 0.8),
        spawn_rotation=Rotation3D("quat", (1, 0, 0, 0)),
        bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
        add_ground_contact_sensors=True,
    )

    output = args.output.resolve()
    model_dir = output / "model"
    source_dir = output / "author-source"
    model_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    world.save_xml_with_assets(model_dir, "model.xml")
    shutil.copy2(source / "LICENSE", source_dir / "FlyGym-LICENSE")
    authored = assets_dir / "model" / "neuromechfly"
    for relative in (
        "rigging.yaml",
        "mujoco_globals.yaml",
        "vision.yaml",
        "visuals.yaml",
        "pose/neutral/yaw_pitch_roll.yaml",
        "meshes/simplified_max2000faces/simplification_metadata.csv",
        "compound_eye.npz",
    ):
        source_path = authored / relative
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)

    model = mj.MjModel.from_xml_path(str(model_dir / "model.xml"))
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)

    joints = []
    for index, dof in enumerate(all_dofs):
        compiled_name = f"fly/{dof.name}"
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, compiled_name)
        if joint_id < 0:
            raise RuntimeError(f"compiled joint missing: {compiled_name}")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        joints.append(
            {
                "index": index,
                "id": dof.name,
                "compiled_name": compiled_name,
                "parent": dof.parent.name,
                "child": dof.child.name,
                "axis_semantic": dof.axis.value,
                "axis_model": [float(x) for x in model.jnt_axis[joint_id]],
                "qpos_address": qpos_adr,
                "dof_address": dof_adr,
                "neutral_rad": float(data.qpos[qpos_adr]),
                "spring_reference_rad": float(model.qpos_spring[qpos_adr]),
                "limited": bool(model.jnt_limited[joint_id]),
                "range_rad": [float(x) for x in model.jnt_range[joint_id]],
                "stiffness": float(model.jnt_stiffness[joint_id]),
                "damping": float(model.dof_damping[dof_adr]),
                "armature": float(model.dof_armature[dof_adr]),
                "actuated": dof in actuated,
                "control_group": control_group_by_dof.get(dof),
            }
        )

    actuators = []
    for actuator_id in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        target_id = int(model.actuator_trnid[actuator_id, 0])
        is_adhesion = name.endswith("-adhesion")
        target_name = mj.mj_id2name(
            model,
            mj.mjtObj.mjOBJ_BODY if is_adhesion else mj.mjtObj.mjOBJ_JOINT,
            target_id,
        )
        actuators.append(
            {
                "index": actuator_id,
                "id": short_name(name),
                "compiled_name": name,
                "kind": "adhesion" if is_adhesion else "position_target",
                "target": short_name(target_name),
                "control_range": [float(x) for x in model.actuator_ctrlrange[actuator_id]],
                "force_range": [float(x) for x in model.actuator_forcerange[actuator_id]],
                "neutral_control": float(data.ctrl[actuator_id]),
                "evidence_grade": "engineered",
                "control_group": (
                    "adhesion"
                    if is_adhesion
                    else control_group_by_dof[
                        next(d for d in actuated if d.name == short_name(target_name))
                    ]
                ),
            }
        )

    sensors = []
    for sensor_id in range(model.nsensor):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, sensor_id)
        sensors.append(
            {
                "index": sensor_id,
                "id": short_name(name),
                "compiled_name": name,
                "data_address": int(model.sensor_adr[sensor_id]),
                "dimension": int(model.sensor_dim[sensor_id]),
                "data_type": int(model.sensor_datatype[sensor_id]),
                "evidence_grade": "measured_simulated_contact",
            }
        )

    body_segments = []
    for index, segment in enumerate(fly.get_bodysegs_order()):
        compiled_name = f"fly/{segment.name}"
        body_segments.append(
            {
                "index": index,
                "id": segment.name,
                "compiled_name": compiled_name,
                "compiled_body_id": int(
                    mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, compiled_name)
                ),
            }
        )

    asset_files = sorted(path for path in output.rglob("*") if path.is_file())
    schema = {
        "schema": "chreatures.neuromechfly-body.v1",
        "identity": IDENTITY,
        "source": {
            "project": "FlyGym / NeuroMechFly v2",
            "repository": "https://github.com/NeLy-EPFL/flygym",
            "version": "2.1.0",
            "revision": PIN,
            "license": "Apache-2.0",
            "morphology_basis": "micro-CT scan of one adult female Drosophila melanogaster; author-adjusted segments include antennae",
            "mesh_variant": "simplified_max2000faces with author left-to-right mirroring",
        },
        "fixture": {
            "axis_order": "yaw_pitch_roll",
            "length_unit": "millimeter",
            "time_unit": "second",
            "angle_unit": "radian",
            "mesh_source_length_unit": "meter",
            "mesh_scale_to_model": 1000,
            "gravity_mm_s2": [float(x) for x in model.opt.gravity],
            "physics_timestep_s": float(model.opt.timestep),
            "proposed_cns_control_period_s": 0.01,
            "spawn_position_mm": [0.0, 0.0, 0.8],
            "spawn_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "neutral_keyframe": 0,
            "total_mass_model_units": float(mj.mj_getTotalmass(model)),
            "source_rigging_mass_sum_model_units": 0.0009997844,
            "mass_unit_status": "FlyGym source does not explicitly name the mass base unit; the numerical total is consistent with 0.9997844 mg only if interpreted as grams",
        },
        "compiled_counts": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "ngeom": int(model.ngeom),
            "nsite": int(model.nsite),
            "nsensor": int(model.nsensor),
            "nsensordata": int(model.nsensordata),
            "nkey": int(model.nkey),
        },
        "body_segments": body_segments,
        "joint_dofs": joints,
        "actuators": actuators,
        "contact_sensors": sensors,
        "sensory_anchors": [
            {"id": "left_eye", "body": "l_eye", "camera": "l_eye_cam_camera", "ommatidia": 721},
            {"id": "right_eye", "body": "r_eye", "camera": "r_eye_cam_camera", "ommatidia": 721},
            {"id": "left_antenna", "bodies": ["l_pedicel", "l_funiculus", "l_arista"]},
            {"id": "right_antenna", "bodies": ["r_pedicel", "r_funiculus", "r_arista"]},
            {"id": "mouth", "bodies": ["c_rostrum", "c_haustellum"]},
        ],
        "interfaces": {
            "cns_eligible_body_measurements": {
                "joint_position_rad": {"shape": [126], "dtype": "float64", "order": "joint_dofs"},
                "joint_velocity_rad_s": {"shape": [126], "dtype": "float64", "order": "joint_dofs"},
                "ground_contact_raw": {
                    "shape": [6, 16],
                    "dtype": "float64",
                    "order": list(LEGS),
                    "channels": ["found", "force_contact_frame_xyz", "torque_contact_frame_xyz", "position_world_xyz", "normal_world_xyz", "tangent_world_xyz"],
                    "reduction": "MuJoCo netforce over each complete leg subtree against the ground plane",
                },
            },
            "observer_only_kinematics": {
                "body_position_mm": {"shape": [69, 3], "dtype": "float64", "order": "body_segments"},
                "body_quaternion_wxyz": {"shape": [69, 4], "dtype": "float64", "order": "body_segments"},
                "policy_access": "forbidden; labels, rewards, diagnostics and rendering only",
            },
            "physical_action": {
                "position_target_rad": {"shape": [84], "dtype": "float64", "range": [-3.14, 3.14], "order": "actuators[0:84]", "groups": expected_group_sizes},
                "adhesion": {"shape": [6], "dtype": "float64", "range": [0.0, 1.0], "order": list(LEGS)},
            },
            "training_transport": "cast physical state and actions to float32; world/body truth may be targets and diagnostics but only CNS-derived state enters policy/memory",
        },
        "model_limits": [
            "The body geometry is grounded in one adult female micro-CT scan, while the current CNS substrate is male; this is an inter-animal and sex transfer.",
            "All 126 joints are author-model anatomical axes but have no supplied anatomical limits; range [0,0] with limited=false means unlimited.",
            "The active84 selection is a Chreatures engineering choice over author anatomical axes: walking42, head3, pedicels6, proboscis6, abdomen15, wings6 and halteres6. Its servos, gains, force bounds and six adhesion actuators are not identified muscles or measured neuromuscular junctions.",
            "Eyes6, funiculus/arista12 and distal tarsus24 remain passive, for 42 passive joint axes total.",
            "Eye cameras support the author 721-ommatidium reduction; the body contains no author olfactory, auditory, humidity, taste or physiological sensor implementation.",
            "Fly geoms use explicit contact pairs; arbitrary objects and social body contact require deliberate pair generation when composing a world.",
        ],
        "files": {
            str(path.relative_to(output)): sha256(path) for path in asset_files
        },
    }
    schema_path = output / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    print(json.dumps({"identity": IDENTITY, "output": str(output), **schema["compiled_counts"]}, indent=2))


if __name__ == "__main__":
    main()
