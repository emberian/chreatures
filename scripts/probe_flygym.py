#!/usr/bin/env python3
"""Headless FlyGym 2.1 probe: two articulated flies and a movable object.

Run this inside an isolated environment containing flygym==2.1.0. The probe
applies only low-level joint targets and adhesion states; it contains no goal,
navigation policy, or hard-coded behavior selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import platform
import resource
import time

import mujoco as mj
import numpy as np

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
    GeomFittingOption,
    KinematicPosePreset,
    NeuroMechFly,
)
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D


PINNED_FLYGYM_VERSION = "2.1.0"


class MultiFlyFlatGroundWorld(FlatGroundWorld):
    """Work around FlyGym 2.1.0's duplicate multi-fly contact sensor names."""

    def _add_ground_contact_sensors(self, fly, bodysegs_with_ground_contact) -> None:
        if len(self.ground_geoms) != 1:
            self.legpos_to_groundcontactsensors_by_fly = None
            return
        if self.legpos_to_groundcontactsensors_by_fly is None:
            self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)
        contact_geoms_by_leg = defaultdict(list)
        for body_segment in bodysegs_with_ground_contact:
            if body_segment.is_leg():
                contact_geoms_by_leg[body_segment.pos].append(body_segment)
        for leg, contact_geoms in contact_geoms_by_leg.items():
            subtree_root = min(
                contact_geoms, key=lambda segment: fly.LEG_LINKS.index(segment.link)
            )
            subtree_body = fly.bodyseg_to_mjcfbody[subtree_root]
            sensor = self.mjcf_root.add_sensor(
                name=f"{fly.name}/ground_contact_{leg}_leg",
                type=mj.mjtSensor.mjSENS_CONTACT,
                objtype=mj.mjtObj.mjOBJ_XBODY,
                objname=subtree_body.name,
                reftype=mj.mjtObj.mjOBJ_GEOM,
                refname=self.ground_geoms[0].name,
                intprm=[119, 3, 1],
            )
            self.legpos_to_groundcontactsensors_by_fly[fly.name][leg] = sensor


def make_fly(
    name: str, vision: bool, capsules: bool
) -> tuple[NeuroMechFly, np.ndarray]:
    geom_option = (
        GeomFittingOption.ALL_TO_CAPSULES
        if capsules
        else GeomFittingOption.UNMODIFIED
    )
    fly = NeuroMechFly(name=name, geom_fitting_option=geom_option)
    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.ROLL_PITCH_YAW,
    )
    pose = KinematicPosePreset.NEUTRAL
    fly.add_joints(skeleton, neutral_pose=pose)
    joint_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        joint_dofs,
        actuator_type=ActuatorType.POSITION,
        neutral_input=pose,
        kp=50,
    )
    fly.add_leg_adhesion(gain=1.0)
    if vision:
        fly.add_vision()
    ordered_dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    neutral = np.asarray(
        [
            fly.jointdof_to_neutralaction_by_type[ActuatorType.POSITION][dof]
            for dof in ordered_dofs
        ],
        dtype=np.float64,
    )
    return fly, neutral


def add_movable_ball(world: FlatGroundWorld) -> tuple[str, str]:
    body = world.mjcf_root.worldbody.add_body(name="movable_ball", pos=[0, 0, 0.52])
    joint = body.add_freejoint(name="movable_ball_freejoint")
    geom = body.add_geom(
        name="movable_ball_geom",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.5],
        density=0.05,
        friction=[0.8, 0.02, 0.001],
        rgba=[0.9, 0.25, 0.1, 1.0],
    )
    # FlatGroundWorld disables ordinary plane collisions and defines explicit
    # pairs for fly segments. Give the ball its own explicit ground pair.
    world.mjcf_root.add_pair(
        name="movable_ball-ground",
        geomname1=geom.name,
        geomname2=world.ground_geom.name,
        friction=[0.8, 0.02, 0.001, 0.0001, 0.0001],
    )
    # Ensure the world's generated neutral keyframe includes this free joint.
    world.world_dof_neutral_states.add(joint.name)
    return body.name, joint.name


def add_external_collision_pairs(world, flies) -> dict[str, int]:
    """Enable ball/fly and coarse body/body contact for contype=0 fly geoms."""
    ball_geom = world.mjcf_root.geom("movable_ball_geom")
    object_pairs = 0
    for fly in flies:
        for geoms in fly.bodyseg_to_mjcfgeom.values():
            for geom in geoms:
                world.mjcf_root.add_pair(
                    name=f"ball-{geom.name}",
                    geomname1=ball_geom.name,
                    geomname2=geom.name,
                )
                object_pairs += 1

    # Cross-fly contact is explicit because NeuroMechFly geoms have contype=0.
    # Non-leg body geoms are enough for bumping, crowding, antenna and body contact
    # without compiling every possible leg/leg pair.
    social_geoms = []
    for fly in flies:
        social_geoms.append(
            [
                geom
                for segment, geoms in fly.bodyseg_to_mjcfgeom.items()
                if not segment.is_leg()
                for geom in geoms
            ]
        )
    social_pairs = 0
    for left_geom in social_geoms[0]:
        for right_geom in social_geoms[1]:
            world.mjcf_root.add_pair(
                name=f"social-{left_geom.name}-{right_geom.name}",
                geomname1=left_geom.name,
                geomname2=right_geom.name,
            )
            social_pairs += 1
    return {"ball_fly_pairs": object_pairs, "cross_fly_pairs": social_pairs}


def build_simulation(vision: bool = False, capsules: bool = False) -> tuple[
    Simulation, dict[str, np.ndarray], str, str, dict[str, int]
]:
    world = MultiFlyFlatGroundWorld(name="chreatures_probe", half_size=20)
    ball_body_name, ball_joint_name = add_movable_ball(world)
    neutral_by_fly: dict[str, np.ndarray] = {}
    flies = []
    for name, position, quaternion in (
        ("fly_left", [-1.4, 0.0, 0.72], [1, 0, 0, 0]),
        ("fly_right", [1.4, 0.0, 0.72], [0, 0, 0, 1]),
    ):
        fly, neutral = make_fly(name, vision, capsules)
        world.add_fly(
            fly,
            spawn_position=position,
            spawn_rotation=Rotation3D("quat", quaternion),
            bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
            add_ground_contact_sensors=True,
        )
        flies.append(fly)
        neutral_by_fly[name] = neutral
    collision_pair_counts = add_external_collision_pairs(world, flies)
    simulation = Simulation(world)
    simulation.reset()
    return (
        simulation,
        neutral_by_fly,
        ball_body_name,
        ball_joint_name,
        collision_pair_counts,
    )


def apply_open_loop_probe(
    simulation: Simulation,
    neutral_by_fly: dict[str, np.ndarray],
    step_index: int,
) -> None:
    """Exercise independent motor channels without choosing a task or goal."""
    dt = simulation.mj_model.opt.timestep
    phase = 2 * np.pi * 7.0 * step_index * dt
    for fly_index, (name, neutral) in enumerate(neutral_by_fly.items()):
        offsets = np.arange(len(neutral), dtype=np.float64) * 0.37 + fly_index * 0.8
        targets = neutral + 0.04 * np.sin(phase + offsets)
        simulation.set_actuator_inputs(name, ActuatorType.POSITION, targets)
        simulation.set_leg_adhesion_states(name, np.ones(6, dtype=bool))


def contact_categories(simulation: Simulation, ball_geom_id: int) -> set[str]:
    categories: set[str] = set()
    for contact in simulation.mj_data.contact[: simulation.mj_data.ncon]:
        names = {
            mj.mj_id2name(simulation.mj_model, mj.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            mj.mj_id2name(simulation.mj_model, mj.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        }
        joined = " ".join(name or "" for name in names)
        if ball_geom_id in (int(contact.geom1), int(contact.geom2)):
            if "fly_left/" in joined or "fly_right/" in joined:
                categories.add("ball-fly")
            if "ground_plane" in names:
                categories.add("ball-ground")
        if "fly_left/" in joined and "fly_right/" in joined:
            categories.add("fly-fly")
    return categories


def run(steps: int, vision: bool = False, capsules: bool = False) -> dict:
    version = importlib.metadata.version("flygym")
    if version != PINNED_FLYGYM_VERSION:
        raise RuntimeError(
            f"expected flygym {PINNED_FLYGYM_VERSION}, found {version}"
        )
    build_start = time.perf_counter()
    (
        sim,
        neutral_by_fly,
        ball_body_name,
        ball_joint_name,
        collision_pair_counts,
    ) = build_simulation(vision, capsules)
    build_seconds = time.perf_counter() - build_start
    model, data = sim.mj_model, sim.mj_data
    ball_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ball_body_name)
    ball_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "movable_ball_geom")
    ball_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, ball_joint_name)
    ball_dof_adr = int(model.jnt_dofadr[ball_joint_id])
    mj.mj_forward(model, data)
    initial_ball_position = data.xpos[ball_body_id].copy()
    data.qvel[ball_dof_adr] = 10.0  # mm/s: launch toward the right fly
    mj.mj_forward(model, data)

    contact_steps = {"ball-ground": 0, "ball-fly": 0, "fly-fly": 0}
    ground_contact_steps = {name: 0 for name in neutral_by_fly}
    snapshot_step = steps // 2
    state_spec = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(model, state_spec), dtype=np.float64)
    tail_final_state = None

    step_start = time.perf_counter()
    for index in range(steps):
        apply_open_loop_probe(sim, neutral_by_fly, index)
        sim.step()
        for category in contact_categories(sim, ball_geom_id):
            contact_steps[category] += 1
        for name in neutral_by_fly:
            found, *_ = sim.get_ground_contact_info(name)
            ground_contact_steps[name] += int(np.any(found))
        if index == snapshot_step:
            mj.mj_getState(model, data, state, state_spec)
    step_seconds = time.perf_counter() - step_start
    tail_final_state = np.empty_like(state)
    mj.mj_getState(model, data, tail_final_state, state_spec)

    # Verify MuJoCo's complete integration state can replay the deterministic tail.
    mj.mj_setState(model, data, state, state_spec)
    mj.mj_forward(model, data)
    replay_start = time.perf_counter()
    for index in range(snapshot_step + 1, steps):
        apply_open_loop_probe(sim, neutral_by_fly, index)
        sim.step()
    replay_seconds = time.perf_counter() - replay_start
    replay_state = np.empty_like(state)
    mj.mj_getState(model, data, replay_state, state_spec)
    final_ball_position = data.xpos[ball_body_id].copy()

    # A separate bare-step timing shows physics cost without Python sensor scans.
    mj.mj_setState(model, data, state, state_spec)
    mj.mj_forward(model, data)
    bare_start = time.perf_counter()
    for _ in range(steps):
        sim.step()
    bare_seconds = time.perf_counter() - bare_start

    fly_observations = {}
    for name, neutral in neutral_by_fly.items():
        found, forces, torques, positions, normals, tangents = sim.get_ground_contact_info(name)
        fly_observations[name] = {
            "position_actuators": len(neutral),
            "body_segments": int(sim.get_body_positions(name).shape[0]),
            "joint_angles": int(sim.get_joint_angles(name).shape[0]),
            "joint_velocities": int(sim.get_joint_velocities(name).shape[0]),
            "ground_contact_found_shape": list(found.shape),
            "ground_contact_force_shape": list(forces.shape),
            "ground_contact_torque_shape": list(torques.shape),
            "ground_contact_position_shape": list(positions.shape),
            "ground_contact_normal_shape": list(normals.shape),
            "ground_contact_tangent_shape": list(tangents.shape),
            "ground_contact_steps": ground_contact_steps[name],
        }

    vision_result = {"requested": vision}
    if vision:
        vision_start = time.perf_counter()
        try:
            for name in neutral_by_fly:
                readout = sim.get_ommatidia_readouts(name)
                fly_observations[name]["ommatidia_readout_shape"] = list(readout.shape)
                fly_observations[name]["ommatidia_dtype"] = str(readout.dtype)
            cold_seconds = time.perf_counter() - vision_start
            warm_start = time.perf_counter()
            for name in neutral_by_fly:
                sim.get_ommatidia_readouts(name)
            vision_result.update(
                success=True,
                cold_two_fly_query_seconds=cold_seconds,
                warm_two_fly_query_seconds=time.perf_counter() - warm_start,
            )
        except Exception as exc:
            vision_result.update(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                cold_two_fly_query_seconds=time.perf_counter() - vision_start,
            )
        finally:
            if sim.eye_renderer is not None:
                sim.eye_renderer.close()
                sim.eye_renderer = None

    simulated_seconds = steps * model.opt.timestep
    return {
        "versions": {
            "flygym": version,
            "mujoco": mj.__version__,
            "python": platform.python_version(),
        },
        "model": {
            "flies": list(neutral_by_fly),
            "fly_collision_geometry": "all_to_capsules" if capsules else "unmodified",
            "nbody": model.nbody,
            "ngeom": model.ngeom,
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nsensor": model.nsensor,
            "explicit_collision_pairs": collision_pair_counts,
            "timestep_seconds": model.opt.timestep,
            "movable_ball_body_id": ball_body_id,
            "movable_ball_initial_position_mm": initial_ball_position.tolist(),
            "movable_ball_final_position_mm": final_ball_position.tolist(),
        },
        "observations": fly_observations,
        "vision": vision_result,
        "contacts": contact_steps,
        "performance": {
            "build_seconds": build_seconds,
            "steps": steps,
            "simulated_seconds": simulated_seconds,
            "step_wall_seconds": step_seconds,
            "control_sensor_steps_per_wall_second": steps / step_seconds,
            "control_sensor_realtime_factor": simulated_seconds / step_seconds,
            "bare_physics_wall_seconds": bare_seconds,
            "bare_physics_steps_per_wall_second": steps / bare_seconds,
            "bare_physics_realtime_factor": simulated_seconds / bare_seconds,
            "snapshot_tail_replay_wall_seconds": replay_seconds,
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "snapshot": {
            "state_spec": "mjSTATE_INTEGRATION",
            "float64_values": len(state),
            "bytes": state.nbytes,
            "tail_replay_exact": bool(np.array_equal(tail_final_state, replay_state)),
            "tail_replay_max_abs_error": float(np.max(np.abs(tail_final_state - replay_state))),
        },
        "claims": {
            "controller": "deterministic low-amplitude actuator probe only; no goal or behavior policy",
            "fidelity": "articulated physics feasibility test; no physiological fidelity claim",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--output", help="optional JSON result path")
    parser.add_argument(
        "--vision", action="store_true",
        help="also exercise both flies' rendered ommatidia readouts",
    )
    parser.add_argument(
        "--capsules", action="store_true",
        help="replace mesh collision geometry with fitted capsules",
    )
    args = parser.parse_args()
    result = run(args.steps, vision=args.vision, capsules=args.capsules)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
