import json
import math

import pytest

from chreatures.world import MAX_OBJECTS, Body, Object, World


def bare_world(seed: int = 11) -> World:
    world = World(seed)
    world.bodies = [Body("mica", "Mica", 100.0, 100.0, 0.0, energy=0.9)]
    world.objects = []
    world.signals = []
    world._touch = {"mica": [0.0, 0.0]}
    return world


def test_initial_world_and_observation_contract_are_local_json_values():
    world = World(7)
    assert (world.width, world.height) == (1200.0, 800.0)
    assert [body.name for body in world.bodies] == ["Mica", "Fern", "Pip"]
    assert len({obj.odor for obj in world.objects if obj.kind == "food"}) >= 2
    assert {"stone", "shelter", "toy", "flower", "beacon"} <= {obj.kind for obj in world.objects}

    senses = world.sense("mica")
    assert len(senses["odor"]) == 2 and all(len(row) == 3 for row in senses["odor"])
    assert len(senses["vision"]) == 16 and all(len(row) == 4 for row in senses["vision"])
    assert len(senses["touch"]) == 2
    assert len(senses["sound"]) == 3
    assert not ({"x", "y", "heading"} & senses.keys())
    json.dumps(senses)


def test_gaussian_scent_is_local_and_bilateral():
    world = bare_world()
    # Facing right: antenna index 0 is anatomically left (smaller y).
    world.objects = [Object("source", "food", 125.0, 55.0, 7.0, "#ff0000", odor=1)]
    odor = world.sense("mica")["odor"]
    assert odor[0][1] > odor[1][1] > 0.0
    assert odor[0][0] == odor[0][2] == 0.0

    world.objects[0].x = 600.0
    assert world.sense("mica")["odor"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_retinal_ray_uses_nearest_object_as_occluder():
    world = bare_world()
    stone = Object("stone", "stone", 155.0, 100.0, 24.0, "#204060")
    food = Object("food", "food", 215.0, 100.0, 30.0, "#ff0000", odor=0)
    world.objects = [stone, food]

    blocked = world.sense("mica")["vision"]
    expected_stone = [0x20 / 255, 0x40 / 255, 0x60 / 255]
    for ray in (7, 8):
        assert blocked[ray][:3] == pytest.approx(expected_stone)

    world.objects.remove(stone)
    clear = world.sense("mica")["vision"]
    for ray in (7, 8):
        assert clear[ray][:3] == pytest.approx([1.0, 0.0, 0.0])
        assert blocked[ray][3] > clear[ray][3]


def test_food_requires_contact_and_an_eat_action():
    world = bare_world()
    food = Object("meal", "food", 118.0, 100.0, 9.0, "#ff5566", odor=0, amount=0.4)
    world.objects = [food]

    initial_gut = world.bodies[0].gut
    world.advance({"mica": {"eat": 0.0}}, 0.1)
    assert food.amount == pytest.approx(0.4)
    assert world.bodies[0].gut < initial_gut  # ordinary digestion still occurs

    before_bite = food.amount
    outcome = world.advance({"mica": {"eat": 1.0}}, 0.1)["mica"]
    assert outcome["nutrition"] > 0.0
    assert food.amount < before_bite
    assert outcome["contact"] > 0.0

    food.x = 300.0
    before_far = food.amount
    assert world.advance({"mica": {"eat": 1.0}}, 0.1)["mica"]["nutrition"] == 0.0
    assert food.amount == before_far


def test_forward_motion_pushes_a_movable_scented_toy():
    world = bare_world()
    toy = Object("toy", "toy", 132.0, 100.0, 12.0, "#aa55dd", odor=2, movable=True)
    world.objects = [toy]
    start = toy.x

    contacts = []
    for _ in range(20):
        contacts.append(world.advance({"mica": {"forward": 1.0}}, 0.05)["mica"]["contact"])

    assert toy.x > start + 5.0
    assert world.bodies[0].x < toy.x
    assert max(contacts) > 0.0


def test_snapshot_restores_rng_events_and_exact_continuation():
    world = World(231)
    world.command({"op": "signal", "x": 260.0, "y": 340.0, "tone": 2})
    warmup = {body.id: {"forward": 0.65, "turn": 0.3, "signal": 0.2} for body in world.bodies}
    world.advance(warmup, 0.05)
    encoded = json.dumps(world.snapshot())
    restored = World.restore(json.loads(encoded))

    actions = {
        "mica": {"forward": 0.8, "turn": -0.4, "eat": 1.0},
        "fern": {"forward": 0.4, "turn": 0.7},
        "pip": {"forward": 0.9, "turn": 0.1, "signal": 0.6},
    }
    for _ in range(12):
        assert restored.advance(actions, 0.05) == world.advance(actions, 0.05)
        assert restored.snapshot() == world.snapshot()


@pytest.mark.parametrize(
    "bad_command",
    [
        {"op": "add", "kind": "food", "x": math.nan, "y": 20},
        {"op": "signal", "x": 20, "y": 20, "tone": 9},
        {"op": "move", "id": "missing", "x": 20, "y": 20},
        {"op": "remove", "id": "missing"},
        {"op": "add", "kind": "toy", "x": 20, "y": 20, "radius": 10000},
    ],
)
def test_invalid_commands_are_atomic(bad_command):
    world = World(19)
    before = world.snapshot()
    with pytest.raises(ValueError):
        world.command(bad_command)
    assert world.snapshot() == before


def test_invalid_actions_are_atomic_including_rng_state():
    world = World(19)
    before = world.snapshot()
    with pytest.raises(ValueError):
        world.advance({"mica": {"forward": float("inf")}}, 0.05)
    assert world.snapshot() == before
    with pytest.raises(ValueError):
        world.advance({"mica": {"forward": 1.01}}, 0.05)
    assert world.snapshot() == before


def test_sound_has_spatial_falloff_and_transient_signals_expire():
    world = bare_world()
    world.command({"op": "signal", "x": 105.0, "y": 100.0, "tone": 2})
    near = world.sense("mica")["sound"][2]
    world.bodies[0].x = 800.0
    far = world.sense("mica")["sound"][2]
    assert near > far > 0.0
    for _ in range(26):
        world.advance({}, 0.05)
    assert world.sense("mica")["sound"][2] == 0.0


def test_continuous_signal_action_is_rate_limited_and_restorable():
    world = bare_world()
    for _ in range(100):
        world.advance({"mica": {"signal": 1.0}}, 0.01)
    # One pulse immediately, then no more than two pulses per elapsed second.
    assert world._next_signal_id - 1 <= 2
    restored = World.restore(json.loads(json.dumps(world.snapshot())))
    assert restored._signal_cooldown == world._signal_cooldown
    for _ in range(60):
        restored.advance({"mica": {"signal": 1.0}}, 0.01)
        world.advance({"mica": {"signal": 1.0}}, 0.01)
    assert restored.snapshot() == world.snapshot()


def test_world_enforces_object_capacity_without_partial_add():
    world = bare_world()
    world.objects = [
        Object(f"stone-{i}", "stone", 200.0, 200.0, 2.0, "#777777")
        for i in range(MAX_OBJECTS)
    ]
    before = world.snapshot()
    with pytest.raises(ValueError, match="more than"):
        world.command({"op": "add", "kind": "food", "x": 20, "y": 20, "odor": 0})
    assert world.snapshot() == before


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda state: state["bodies"][0].update(x=float("nan")),
        lambda state: state["bodies"][1].update(id=state["bodies"][0]["id"]),
        lambda state: state["objects"][0].update(x=-1.0),
        lambda state: state["signals"].append(
            {"id": "bad", "x": 20, "y": 20, "tone": 8, "strength": 1, "remaining": 1}
        ),
        lambda state: state.update(signal_cooldown={"not-a-body": 0.2}),
    ],
)
def test_restore_rejects_malformed_snapshots(corrupt):
    state = World(4).snapshot()
    corrupt(state)
    with pytest.raises(ValueError):
        World.restore(state)


def test_boundary_touch_reports_wall_on_anatomical_side():
    world = bare_world()
    body = world.bodies[0]
    body.y = body.radius - 0.5  # top wall is to the left while facing right
    world.advance({}, 0.05)
    assert world.sense("mica")["touch"] == [1.0, 0.0]
