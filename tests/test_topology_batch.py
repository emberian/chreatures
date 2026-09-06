import copy

import mujoco
import numpy as np
import pytest

from chreatures.physics import PhysicsWorld


def _named_joint_state(world):
    result = {}
    for joint_id in range(world.model.njnt):
        name = mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(world.model.jnt_type[joint_id])
        qn = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 1
        vn = 6 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 1
        qa, da = int(world.model.jnt_qposadr[joint_id]), int(world.model.jnt_dofadr[joint_id])
        result[name] = (world.data.qpos[qa:qa + qn].copy(), world.data.qvel[da:da + vn].copy())
    return result


def test_topology_batch_compiles_then_atomically_preserves_world_state_and_restore():
    world = PhysicsWorld(seed=441)
    world.advance({body.id: {"forward": 0.4, "turn": -0.2} for body in world.bodies}, 0.05)
    world.command({"op": "hand", "id": "violet-ball", "x": 4.0, "y": 4.0, "z": 0.8})
    preserved_geom = mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_GEOM, "entity:high-walk:geom:0"
    )
    world.model.geom_friction[preserved_geom, 0] = 1.234
    before_joints = _named_joint_state(world)
    before_rng = copy.deepcopy(world.rng.bit_generator.state)
    before_private = ([body.to_dict() for body in world.bodies], copy.deepcopy(world._touch),
                      copy.deepcopy(world._contact_normals), world._grips.copy(), copy.deepcopy(world._hand))
    transaction = world.prepare_topology_batch([
        {"op": "add", "entity": {
            "id": "growth-root", "mobility": "static", "material": "wood",
            "physical_material": "timber", "position": [3.0, 3.0, 0.0],
            "shapes": [{"type": "capsule", "size": [0.025], "fromto": [0, 0, 0, 0, 0, 0.5]}],
            "components": [],
        }},
        {"op": "add", "entity": {
            "id": "growth-branch", "mobility": "static", "material": "leaf",
            "physical_material": "light", "position": [3.0, 3.0, 0.5],
            "shapes": [{"type": "ellipsoid", "size": [0.12, 0.04, 0.01], "position": [0.1, 0, 0.1]}],
            "components": [],
        }},
        {"op": "append_shapes", "id": "high-walk", "shapes": [
            {"type": "capsule", "size": [0.02], "fromto": [0, 0, 0, 0.2, 0.1, 0.3]},
        ]},
        {"op": "remove", "id": "berry-a"},
    ])
    assert "growth-root" not in {entity["id"] for entity in world._entities}
    result = transaction.commit()
    assert result["model_revision"] == 1
    assert {"growth-root", "growth-branch"} <= {entity["id"] for entity in world._entities}
    assert "berry-a" not in {entity["id"] for entity in world._entities}
    preserved_geom = mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_GEOM, "entity:high-walk:geom:0"
    )
    assert world.model.geom_friction[preserved_geom, 0] == 1.234
    for name, (qpos, qvel) in before_joints.items():
        if name == "entity:berry-a:free":
            continue
        now_qpos, now_qvel = _named_joint_state(world)[name]
        np.testing.assert_array_equal(qpos, now_qpos)
        np.testing.assert_array_equal(qvel, now_qvel)
    assert world.rng.bit_generator.state == before_rng
    assert ([body.to_dict() for body in world.bodies], world._touch, world._contact_normals,
            world._grips, world._hand) == before_private
    assert PhysicsWorld.restore(world.snapshot()).snapshot() == world.snapshot()


def test_topology_batch_failure_and_stale_commit_leave_world_unchanged():
    world = PhysicsWorld(seed=442)
    before = world.snapshot()
    with pytest.raises(ValueError):
        world.prepare_topology_batch([{"op": "append_shapes", "id": "high-walk", "shapes": [
            {"type": "sphere", "size": [-1.0]},
        ]}])
    assert world.snapshot() == before

    first = world.prepare_topology_batch([{"op": "remove", "id": "berry-a"}])
    stale = world.prepare_topology_batch([{"op": "remove", "id": "berry-b"}])
    first.commit()
    committed = world.snapshot()
    with pytest.raises(RuntimeError, match="stale"):
        stale.commit()
    assert world.snapshot() == committed


def test_topology_commit_preserves_dynamics_that_advance_after_prepare():
    world = PhysicsWorld(seed=443)
    transaction = world.prepare_topology_batch([{"op": "remove", "id": "berry-a"}])
    for _ in range(3):
        world.advance({body.id: {"forward": 0.65, "turn": 0.25} for body in world.bodies}, 0.05)
    live_time = float(world.data.time)
    live_joints = _named_joint_state(world)
    live_rng = copy.deepcopy(world.rng.bit_generator.state)

    transaction.commit()

    assert world.data.time == live_time
    assert world.rng.bit_generator.state == live_rng
    for name, (qpos, qvel) in live_joints.items():
        if name == "entity:berry-a:free":
            continue
        now_qpos, now_qvel = _named_joint_state(world)[name]
        np.testing.assert_array_equal(qpos, now_qpos)
        np.testing.assert_array_equal(qvel, now_qvel)


def test_topology_replace_uses_candidate_geometry_and_components():
    world = PhysicsWorld(seed=444)
    replacement = copy.deepcopy(world._entity("violet-ball"))
    replacement["shapes"][0]["size"] = [0.30]
    replacement["components"] = [{"type": "scent", "odor": 2, "strength": 0.123}]
    old_geom = mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_GEOM, "entity:violet-ball:geom:0"
    )
    world.model.geom_size[old_geom, 0] = 0.19

    world.prepare_topology_batch([{
        "op": "replace", "id": "violet-ball", "entity": replacement,
    }]).commit()

    geom = mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_GEOM, "entity:violet-ball:geom:0"
    )
    assert world.model.geom_size[geom, 0] == 0.30
    assert world._components["violet-ball"] == replacement["components"]
    center = world.data.geom_xpos[geom].copy()
    hit_geom = np.empty(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        world.model, world.data, center - [1.0, 0.0, 0.0],
        np.asarray([1.0, 0.0, 0.0]), None, False, -1, hit_geom,
    )
    assert int(hit_geom[0]) == geom
    assert distance == pytest.approx(0.70, abs=1e-10)
