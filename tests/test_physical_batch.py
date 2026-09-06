import json
from pathlib import Path
from types import MethodType

import mujoco
import numpy as np

from chreatures.acoustics import Acoustics
from chreatures.ecology import Ecology
from chreatures.fields import FieldEnvironment
from chreatures.physical_batch import (
    FastArticulatedSensoriumWorld,
)
from chreatures.physics import PhysicsWorld
from chreatures.sensorium import ArticulatedSensoriumWorld
from chreatures.training_environment import (
    EmbodiedTrainingProfile, EmbodiedTrainingWorld, embodied_training_spec,
)
from tests._reference_contacts import collect_contacts as reference_collect_contacts


ROOT = Path(__file__).parents[1]
HABITAT = ROOT / "data/habitats/terrarium-garden.json"
RESOURCES = ROOT / "data/ecology/terrarium-orchard.json"
ACOUSTICS = ROOT / "data/components/terrarium-play.json"


def _stack(world_type):
    spec = json.loads(HABITAT.read_text())
    spec["sensorium"] = {"frame": "body-v1"}
    world = world_type(seed=815, spec=spec)
    return (
        world,
        FieldEnvironment.from_world(world),
        Ecology(world, RESOURCES, seed=815),
        Acoustics(world, ACOUSTICS),
    )


def _step(stack, index):
    world, field, ecology, acoustics = stack
    actions = {
        body.id: {
            "forward": float(np.sin(index * 0.19 + body_index)),
            "turn": float(np.cos(index * 0.13 - body_index) * 0.7),
            "gaze_pitch": float(np.sin(index * 0.07 + body_index) * 0.6),
            "grip": float((index + body_index) % 7 == 0),
        }
        for body_index, body in enumerate(world.bodies)
    }
    outcome = world.advance(actions, 0.05)
    acoustic = acoustics.advance(0.05)
    resource = ecology.advance(0.05)
    field_report = field.advance(0.05, sources=field.sources_from_world(world))
    return outcome, acoustic, resource, field_report


def _senses(stack):
    world, field, _, _ = stack
    result = {}
    for body in world.bodies:
        value = world.sense(body.id)
        value["odor"] = (-np.expm1(-np.asarray(field.sample(value["antenna_position"]))[:, :3] / 0.1)).tolist()
        result[body.id] = value
    return result


def _assert_parity(reference, fast):
    left, right = reference[0], fast[0]
    np.testing.assert_array_equal(left.data.qpos, right.data.qpos)
    np.testing.assert_array_equal(left.data.qvel, right.data.qvel)
    assert [body.to_dict() for body in left.bodies] == [body.to_dict() for body in right.bodies]
    assert [obj.to_dict() for obj in left.objects] == [obj.to_dict() for obj in right.objects]
    assert _senses(reference) == _senses(fast)
    np.testing.assert_array_equal(reference[1].concentration, fast[1].concentration)
    assert reference[2].snapshot() == fast[2].snapshot()
    assert reference[3].snapshot() == fast[3].snapshot()


def test_fast_world_rebinds_after_dynamic_entity_and_restores_full_ecosystem():
    reference = _stack(ArticulatedSensoriumWorld)
    fast = _stack(FastArticulatedSensoriumWorld)
    for index in range(9):
        assert _step(reference, index) == _step(fast, index)
    _assert_parity(reference, fast)

    command = {"op": "add", "preset": "play-ball", "id": "visitor-ball", "x": 6.0, "y": 3.2, "z": 0.22}
    assert reference[0].command(command) == fast[0].command(command)
    for body in fast[0].bodies:
        expected_qpos, expected_dof = [], []
        for leg in fast[0]._fast_leg_names:
            for kind in ("hip", "knee"):
                joint = mujoco.mj_name2id(
                    fast[0].model, mujoco.mjtObj.mjOBJ_JOINT,
                    f"resident:{body.id}:joint:{leg}:{kind}",
                )
                expected_qpos.append(fast[0].model.jnt_qposadr[joint])
                expected_dof.append(fast[0].model.jnt_dofadr[joint])
        np.testing.assert_array_equal(fast[0]._fast_joint_qpos[body.id], expected_qpos)
        np.testing.assert_array_equal(fast[0]._fast_joint_dof[body.id], expected_dof)

    hand = {"op": "hand", "id": "visitor-ball", "x": 6.65, "y": 3.45, "z": 0.34,
            "stiffness": 24.0, "damping": 3.0}
    assert reference[0].command(hand) == fast[0].command(hand)
    for index in range(9, 17):
        assert _step(reference, index) == _step(fast, index)
    _assert_parity(reference, fast)

    reference_snapshots = tuple(item.snapshot() for item in reference)
    fast_snapshots = tuple(item.snapshot() for item in fast)
    assert reference_snapshots == fast_snapshots
    restored_reference_world = ArticulatedSensoriumWorld.restore(reference_snapshots[0])
    restored_fast_world = FastArticulatedSensoriumWorld.restore(fast_snapshots[0])
    restored_reference = (
        restored_reference_world,
        FieldEnvironment.restore(reference_snapshots[1]),
        Ecology.restore(restored_reference_world, reference_snapshots[2]),
        Acoustics.restore(restored_reference_world, reference_snapshots[3]),
    )
    restored_fast = (
        restored_fast_world,
        FieldEnvironment.restore(fast_snapshots[1]),
        Ecology.restore(restored_fast_world, fast_snapshots[2]),
        Acoustics.restore(restored_fast_world, fast_snapshots[3]),
    )
    assert _step(restored_reference, 17) == _step(restored_fast, 17)
    _assert_parity(restored_reference, restored_fast)


def test_embodied_training_world_fast_backend_is_exact_across_restore():
    profile = EmbodiedTrainingProfile.current()
    spec = embodied_training_spec(918, profile=profile)
    reference = EmbodiedTrainingWorld(918, spec, profile, physical_backend="reference")
    fast = EmbodiedTrainingWorld(918, spec, profile, physical_backend="fast")

    for index in range(8):
        actions = {
            body.id: {
                "forward": float(np.sin(index * 0.23 + body_index)),
                "turn": float(np.cos(index * 0.17 - body_index) * 0.8),
                "gaze_pitch": float(np.sin(index * 0.11 + body_index) * 0.7),
                "grip": float((index + body_index) % 5 == 0),
            }
            for body_index, body in enumerate(reference.bodies)
        }
        assert {body.id: reference.sense(body.id) for body in reference.bodies} == {
            body.id: fast.sense(body.id) for body in fast.bodies
        }
        assert reference.advance(actions, 0.05) == fast.advance(actions, 0.05)
        assert reference.last_telemetry == fast.last_telemetry

    add = {"op": "add", "preset": "play-ball", "id": "training-ball",
           "x": 6.1, "y": 3.1, "z": 0.22}
    assert reference.world.command(add) == fast.world.command(add)
    assert reference.snapshot() == fast.snapshot()

    reference_restored = EmbodiedTrainingWorld.restore(
        reference.snapshot(), expected_profile=profile, physical_backend="reference"
    )
    fast_restored = EmbodiedTrainingWorld.restore(
        fast.snapshot(), expected_profile=profile, physical_backend="fast"
    )
    actions = {body.id: {"forward": 0.4, "turn": -0.2} for body in reference_restored.bodies}
    assert reference_restored.advance(actions, 0.05) == fast_restored.advance(actions, 0.05)
    assert {body.id: reference_restored.sense(body.id) for body in reference_restored.bodies} == {
        body.id: fast_restored.sense(body.id) for body in fast_restored.bodies
    }


def test_native_contact_batch_preserves_full_acoustic_world_state():
    reference = _stack(FastArticulatedSensoriumWorld)
    native = _stack(FastArticulatedSensoriumWorld)
    reference[0]._collect_contacts = MethodType(reference_collect_contacts, reference[0])

    for index in range(12):
        assert _step(reference, index) == _step(native, index)
        _assert_parity(reference, native)

    assert native[0].data.ncon > 0


def test_native_contact_batch_preserves_generic_nonacoustic_world_state():
    reference = PhysicsWorld(seed=271)
    native = PhysicsWorld(seed=271)
    reference._collect_contacts = MethodType(reference_collect_contacts, reference)
    for index in range(8):
        actions = {
            body.id: {"forward": float(np.sin(index * 0.2)), "turn": -0.35}
            for body in reference.bodies
        }
        assert reference.advance(actions, 0.05) == native.advance(actions, 0.05)
        np.testing.assert_array_equal(reference.data.qpos, native.data.qpos)
        np.testing.assert_array_equal(reference.data.qvel, native.data.qvel)
        assert {body.id: reference.sense(body.id) for body in reference.bodies} == {
            body.id: native.sense(body.id) for body in native.bodies
        }


def test_native_contact_batch_empty_shapes_are_stable():
    spec = json.loads((ROOT / "data/habitats/hollow-garden.json").read_text())
    spec["gravity"] = [0.0, 0.0, 0.0]
    spec["entities"] = [entity for entity in spec["entities"] if entity["id"] == "ground"]
    for index, body in enumerate(spec["bodies"]):
        body["position"] = [1.0 + index * 2.0, 1.0, 2.0]
    world = PhysicsWorld(seed=272, spec=spec)
    assert world.data.ncon == 0
    arrays = world._native_contacts.evaluate(
        world.model, world.data, world.model.opt.timestep, 10.0, 5.0
    )
    assert tuple(value.shape for value in arrays) == (
        (0,), (0,), (0, 3), (0, 3), (0,), (0,), (0,), (0,),
    )
