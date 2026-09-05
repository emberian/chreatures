#!/usr/bin/env python3
"""Probe anonymous physical coupling between residents and full MaleCNS ports."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.acoustics import Acoustics
from chreatures.neural_ports import NeuralPortBundle, encode_physical_senses, load_port_spec
from chreatures.sensorium import ArticulatedSensoriumWorld, SensoriumWorld


HABITAT = ROOT / "data/habitats/hollow-garden.json"


def probe_spec(*, occluded: bool, shared_object: bool = False) -> dict[str, Any]:
    """Make a small physical scene without putting scenario facts in senses."""
    spec = json.loads(HABITAT.read_text(encoding="utf-8"))
    entities = {entity["id"]: entity for entity in spec["entities"]}
    spec["name"] = "anonymous-social-coupling-probe"
    spec["bodies"] = [
        {
            "id": "listener", "name": "listener", "position": [4.5, 4.0, 0.18],
            "heading": math.pi, "material": "mica", "energy": 0.8,
            "gut": 0.1, "fatigue": 0.03,
        },
        {
            "id": "source", "name": "source", "position": [3.0, 4.0, 0.18],
            "heading": 0.0, "material": "fern", "energy": 0.8,
            "gut": 0.1, "fatigue": 0.03,
        },
    ]
    keep = ["ground", "west-wall", "east-wall", "north-wall", "south-wall"]
    spec["entities"] = [copy.deepcopy(entities[name]) for name in keep]
    barrier = {
        "id": "barrier", "mobility": "static", "material": "slate",
        "physical_material": "rock",
        "position": [3.75, 4.0 if occluded else 6.3, 0.45],
        "shapes": [{"type": "box", "size": [0.12, 0.72, 0.45]}],
        "components": [],
    }
    spec["entities"].append(barrier)
    if shared_object:
        box = copy.deepcopy(entities["stack-box-a"])
        box["position"] = [3.30, 4.0, 0.16]
        spec["entities"].append(box)
    return spec


def channels(world: Any) -> tuple[dict[str, float], dict[str, Any]]:
    raw = world.sense("listener")
    names, values = encode_physical_senses(raw, load_port_spec())
    return dict(zip(names, values.astype(float).tolist(), strict=True)), raw


def channel_contrast(
    silent: dict[str, float], active: dict[str, float]
) -> dict[str, Any]:
    changed = [name for name in silent if active[name] != silent[name]]
    ranked = sorted(changed, key=lambda name: abs(active[name] - silent[name]), reverse=True)
    return {
        "changed_channel_count": len(changed),
        "changed_channels": [
            {"name": name, "silent": silent[name], "active": active[name],
             "delta": active[name] - silent[name]}
            for name in ranked[:16]
        ],
        "l1_delta": float(sum(abs(active[name] - silent[name]) for name in silent)),
        "maximum_abs_delta": float(max((abs(active[name] - silent[name]) for name in silent), default=0.0)),
    }


def sound_pair(*, occluded: bool, physical_visibility: bool) -> dict[str, Any]:
    seed_world = ArticulatedSensoriumWorld(seed=710, spec=probe_spec(occluded=occluded))
    snapshot = seed_world.snapshot()
    silent_world = ArticulatedSensoriumWorld.restore(snapshot)
    active_world = ArticulatedSensoriumWorld.restore(snapshot)
    engines = []
    if physical_visibility:
        # Attaching the physical acoustic layer makes resident signal propagation
        # use its collision-ray visibility. It adds no precharged transducer.
        engines = [Acoustics(silent_world, {"version": 1, "include_authored": False}),
                   Acoustics(active_world, {"version": 1, "include_authored": False})]
    common = {"listener": {}, "source": {}}
    silent_world.advance(common, 0.05)
    active_world.advance({"listener": {}, "source": {"signal_mid": 1.0}}, 0.05)
    silent, silent_raw = channels(silent_world)
    active, active_raw = channels(active_world)
    for engine in engines:
        engine.close()
    return {
        "silent": silent, "active": active,
        "listener_sound_silent": silent_raw["sound"],
        "listener_sound_active": active_raw["sound"],
        "contrast": channel_contrast(silent, active),
    }


def shared_motion_pair(steps: int = 60) -> dict[str, Any]:
    """Let one resident move a common box while the listener remains passive."""
    seed_world = SensoriumWorld(seed=711, spec=probe_spec(occluded=False, shared_object=True))
    snapshot = seed_world.snapshot()
    silent_world = SensoriumWorld.restore(snapshot)
    active_world = SensoriumWorld.restore(snapshot)
    for _ in range(steps):
        silent_world.advance({"listener": {}, "source": {}}, 0.05)
        active_world.advance({"listener": {}, "source": {"forward": 1.0}}, 0.05)
    silent, _ = channels(silent_world)
    active, _ = channels(active_world)
    silent_box = next(obj for obj in silent_world.objects if obj.id == "stack-box-a")
    active_box = next(obj for obj in active_world.objects if obj.id == "stack-box-a")
    box_delta = np.asarray(
        [active_box.x - silent_box.x, active_box.y - silent_box.y, active_box.z - silent_box.z]
    )
    return {
        "silent": silent, "active": active, "steps": steps,
        "shared_object_displacement": box_delta.astype(float).tolist(),
        "shared_object_distance": float(np.linalg.norm(box_delta)),
        "contrast": channel_contrast(silent, active),
    }


def neural_contrasts(
    pairs: dict[str, dict[str, Any]], graph_path: Path, port_path: Path
) -> dict[str, Any]:
    """Apply each pair from identical zero state to the canonical full graph."""
    from chreatures.malecns import MaleCNSGraph
    from chreatures.remote_brain import RemoteBrain

    graph = MaleCNSGraph.load(graph_path, mmap=True)
    ports = NeuralPortBundle.load(port_path, graph)
    resident_ids = [f"probe-{index}" for index in range(len(pairs) * 2)]
    brain = RemoteBrain(
        graph, capacity=len(resident_ids), device="cpu", microbatch_size=2,
        **ports.remote_brain_kwargs(),
    )
    brain.add_residents(resident_ids)
    entries = []
    labels = []
    for index, (name, pair) in enumerate(pairs.items()):
        for branch, channels_value in (("silent", pair["silent"]), ("active", pair["active"])):
            resident_id = resident_ids[index * 2 + (branch == "active")]
            entries.append({"id": resident_id, "senses": channels_value})
            labels.append((name, branch))
    outputs = brain.step(entries, 0.05)
    by_label = {label: output for label, output in zip(labels, outputs, strict=True)}
    result = {}
    for name in pairs:
        silent = np.asarray(by_label[(name, "silent")]["features"], dtype=np.float32)
        active = np.asarray(by_label[(name, "active")]["features"], dtype=np.float32)
        delta = active - silent
        order = np.argsort(np.abs(delta))[::-1]
        result[name] = {
            "changed_readout_count": int(np.count_nonzero(delta)),
            "l1_readout_delta": float(np.abs(delta).sum()),
            "maximum_abs_readout_delta": float(np.abs(delta).max(initial=0.0)),
            "top_readouts": [
                {"name": ports.readout_names[int(i)], "delta": float(delta[i]),
                 "silent": float(silent[i]), "active": float(active[i])}
                for i in order[:10] if delta[i] != 0
            ],
        }
    return {
        "backend": "private-cpu-RemoteBrain",
        "graph_sha256": graph.hash,
        "neurons": graph.n,
        "edges": graph.edge_count,
        "inputs": len(ports.input_names),
        "readouts": len(ports.readout_names),
        "contrasts": result,
    }


def causal_checks(receipt: dict[str, Any]) -> dict[str, bool]:
    physical = receipt["physical"]
    legacy_clear = physical["legacy_signal_unoccluded"]
    legacy_blocked = physical["legacy_signal_occluded"]
    physical_clear = physical["physical_signal_unoccluded"]
    physical_blocked = physical["physical_signal_occluded"]
    shared = physical["shared_object_motion"]
    checks = {
        "tone_reaches_listener_only_through_sound_1": all(
            item["contrast"]["changed_channel_count"] == 1
            and item["contrast"]["changed_channels"][0]["name"] == "sound/1"
            for item in (legacy_clear, legacy_blocked, physical_clear, physical_blocked)
        ),
        "legacy_direct_signal_is_not_occluded": (
            legacy_clear["listener_sound_active"] == legacy_blocked["listener_sound_active"]
        ),
        "acoustic_visibility_attenuates_barrier_path": (
            0.0 < physical_blocked["listener_sound_active"][1]
            < physical_clear["listener_sound_active"][1]
        ),
        "source_motion_moves_shared_object": shared["shared_object_distance"] > 1e-4,
        "shared_motion_reaches_retina": any(
            item["name"].startswith("retina/") for item in shared["contrast"]["changed_channels"]
        ),
        "shared_motion_reaches_sound": any(
            item["name"].startswith("sound/") for item in shared["contrast"]["changed_channels"]
        ),
    }
    if "neural" in receipt:
        neural = receipt["neural"]["contrasts"]
        checks["all_physical_contrasts_reach_full_graph_readouts"] = all(
            neural[name]["changed_readout_count"] > 0 for name in physical
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"social coupling checks failed: {failed}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--port-bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-neural", action="store_true")
    args = parser.parse_args()
    if not args.skip_neural and (args.graph is None or args.port_bundle is None):
        raise SystemExit("--graph and --port-bundle are required unless --skip-neural is used")

    pairs = {
        "legacy_signal_unoccluded": sound_pair(occluded=False, physical_visibility=False),
        "legacy_signal_occluded": sound_pair(occluded=True, physical_visibility=False),
        "physical_signal_unoccluded": sound_pair(occluded=False, physical_visibility=True),
        "physical_signal_occluded": sound_pair(occluded=True, physical_visibility=True),
        "shared_object_motion": shared_motion_pair(),
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "method": {
            "paired_world_snapshots": True,
            "listener": "passive",
            "identity_or_world_truth_in_senses": False,
            "signal_tone": 1,
            "dt": 0.05,
        },
        "physical": {
            name: {key: value for key, value in pair.items() if key not in {"silent", "active"}}
            for name, pair in pairs.items()
        },
    }
    if not args.skip_neural:
        receipt["neural"] = neural_contrasts(pairs, args.graph, args.port_bundle)
    receipt["checks"] = causal_checks(receipt)
    text = json.dumps(receipt, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(text)


if __name__ == "__main__":
    main()
