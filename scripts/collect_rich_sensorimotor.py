#!/usr/bin/env python3
"""Collect current family-v5 trajectories from native developmental residents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.sensorimotor_worker_native import DevelopmentalResidentCohort
from chreatures.training_cohort import (
    OUTCOME_FIELDS,
    RICH_CHANNEL_NAMES_SHA256,
    RICH_OBSERVATION_CHANNELS,
    RICH_SENSORIUM_PROFILE_SHA256,
    TrainingCohortBrain,
    WorldTrainingPool,
    load_training_graph,
)

FORMAT = "chreatures-sensorimotor-play-rich-v2"
SCHEMA = ROOT / "research/sensorimotor_skills/trajectory-schema-rich-v2.json"
ACTION_NAMES = (
    "thrust",
    "yaw",
    "gaze_pitch",
    "grip",
    "signal_low",
    "signal_mid",
    "signal_high",
    "posture",
)
OBSERVATION_ORDER = (
    "rich_body_v1_4096",
    "canonical_channels_351",
    "physiology_6",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_identity() -> dict[str, Any]:
    paths = (
        Path("scripts/collect_rich_sensorimotor.py"),
        Path("chreatures/training_cohort.py"),
        Path("chreatures/sensorimotor_worker_native.py"),
        Path("chreatures/training_environment.py"),
        Path("research/sensorimotor_skills/trajectory-schema-rich-v2.json"),
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_rows = subprocess.run(
        ["git", "status", "--porcelain", "--", *(str(path) for path in paths)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "revision": revision,
        "dirty": bool(dirty_rows),
        "dirty_paths": sorted(row[3:] for row in dirty_rows),
        "files": {
            str(path): {"bytes": (ROOT / path).stat().st_size, "sha256": sha256(ROOT / path)}
            for path in paths
        },
    }


def resident_model_identity(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    return {
        "path": str(path),
        "file_sha256": sha256(path),
        "artifact_sha256": metadata.get("artifact_sha256"),
        "format": metadata.get("format"),
        "version": metadata.get("version"),
        "execution": metadata.get("execution"),
    }


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resident-artifact", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--chemical-habitat", type=Path, required=True)
    parser.add_argument("--chemical-biosphere", type=Path, required=True)
    parser.add_argument("--nursery-family-config", type=Path, required=True)
    parser.add_argument("--nursery-family-schedule", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--heldout-worlds", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--brain-backend", choices=("tiled", "triton"), default="tiled")
    parser.add_argument("--physical-backend", choices=("fast",), default="fast")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not 2 <= args.worlds <= 16 or not 1 <= args.heldout_worlds < args.worlds:
        raise SystemExit("world split requires 2..16 worlds and a nonempty holdout")
    if not 1 <= args.episodes <= 64 or not 44 <= args.steps <= 100_000:
        raise SystemExit("episodes or steps outside bounded training-data range")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    for path in (
        args.resident_artifact,
        args.port_bundle,
        args.chemical_habitat,
        args.chemical_biosphere,
        args.nursery_family_config,
        args.nursery_family_schedule,
    ):
        if not path.is_file():
            raise SystemExit(f"required artifact does not exist: {path}")


def physiology(bodies, circuit: np.ndarray) -> np.ndarray:
    rows = []
    index = 0
    for world in bodies:
        for body in world:
            rows.append(
                (
                    body["energy"],
                    body["gut"],
                    body["fatigue"],
                    math.tanh(float(body["speed"]) / 2),
                    math.tanh(float(body["angular_velocity"]) / 4),
                    circuit[index, 2],
                )
            )
            index += 1
    return np.asarray(rows, dtype=np.float32)


def observe(pool, brain, dt: float):
    rich, canonical, bodies = pool.observe_arrays()
    neural, circuit, _ = brain.step_channels(canonical, dt)
    physical = physiology(bodies, circuit)
    observation = np.ascontiguousarray(
        np.concatenate((rich, canonical, physical), axis=1), dtype=np.float32
    )
    return observation, canonical, physical, neural, bodies


def action_payload(actions: np.ndarray, oral: np.ndarray, bodies):
    payloads = []
    row = 0
    for world in bodies:
        mapped = {}
        for body in world:
            mapped[str(body["id"])] = {
                **dict(zip(ACTION_NAMES, actions[row].astype(float), strict=True)),
                "eat": float(oral[row]),
            }
            row += 1
        payloads.append({"actions": mapped, "dt": 0.05})
    return payloads


def outcome_rows(outcomes, bodies) -> np.ndarray:
    rows = []
    for world_outcomes, world_bodies in zip(outcomes, bodies, strict=True):
        for body in world_bodies:
            value = world_outcomes[str(body["id"])]
            rows.append([float(value.get(name, 0.0)) for name in OUTCOME_FIELDS])
    return np.asarray(rows, dtype=np.float32)


def main() -> int:
    args = arguments()
    validate(args)
    args.output.mkdir(parents=True, exist_ok=True)
    from chreatures.neural_ports import NeuralPortBundle
    from chreatures.training_environment import EmbodiedTrainingProfile

    profile = EmbodiedTrainingProfile.nursery_family(
        args.chemical_habitat,
        args.chemical_biosphere,
        args.nursery_family_config,
        args.nursery_family_schedule,
    )
    graph = load_training_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    transport = profile.component("family")["transport"]
    required = {
        "residents": 6,
        "rich": RICH_OBSERVATION_CHANNELS,
        "physical": len(ports.input_names),
        "physiology": 6,
        "controller": RICH_OBSERVATION_CHANNELS + len(ports.input_names) + 6,
        "readouts": len(ports.readout_names),
    }
    if int(profile.component("version")) != 5 or transport != required or required != {
        "residents": 6,
        "rich": 4096,
        "physical": 351,
        "physiology": 6,
        "controller": 4453,
        "readouts": 384,
    }:
        raise SystemExit("family profile transport differs from collector interfaces")
    count = args.worlds * 6
    brain = TrainingCohortBrain(
        graph, ports, count, device=args.device, backend=args.brain_backend
    )
    pool = WorldTrainingPool(
        args.worlds,
        dict(ports.spec),
        profile.to_value(),
        args.physical_backend,
        residents_per_world=6,
    )
    artifact_path = args.resident_artifact.resolve()
    identity = {
        "format": f"{FORMAT}-identity",
        "source": source_identity(),
        "resident_artifact": resident_model_identity(artifact_path),
        "profile": profile.to_value(),
        "graph_sha256": str(graph.hash),
        "port_spec_sha256": ports.spec_hash,
        "port_bundle_sha256": sha256(args.port_bundle.resolve()),
        "rich_profile_sha256": RICH_SENSORIUM_PROFILE_SHA256,
        "rich_channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
        "observation_order": list(OBSERVATION_ORDER),
        "transition_outcome_order": list(OUTCOME_FIELDS),
        "split": {
            "train_world_slots": list(range(args.worlds - args.heldout_worlds)),
            "heldout_world_slots": list(
                range(args.worlds - args.heldout_worlds, args.worlds)
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    identity["sha256"] = canonical_hash(identity)
    identity_receipt = atomic_json(args.output / "identity.json", identity)
    packets = []
    end_checkpoints = []
    transport_timing = None
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            heldout_start = args.worlds - args.heldout_worlds
            bodies = pool.reset(
                [
                    {
                        "seed": args.seed + episode * 1009 + world,
                        "held_out": world >= heldout_start,
                    }
                    for world in range(args.worlds)
                ]
            )
            resident_ids = [
                f"episode-{episode:03d}/world-{world:03d}/resident-{resident:02d}"
                for world in range(args.worlds)
                for resident in range(6)
            ]
            brain.reset_residents(resident_ids)
            residents = DevelopmentalResidentCohort(
                artifact_path,
                count,
                action_mode="sample",
                goal_seed=args.seed ^ (episode << 24) ^ 0x60A1,
                action_seed=args.seed ^ (episode << 24) ^ 0xAC71,
            )
            expected_neural = {
                "graph_sha256": str(graph.hash),
                "port_spec_sha256": ports.spec_hash,
                "port_bundle_sha256": sha256(args.port_bundle.resolve()),
            }
            if residents.neural_contract != expected_neural:
                raise RuntimeError(
                    "developmental resident neural substrate differs from collection"
                )
            previous = np.zeros((count, 9), dtype=np.float32)
            reset = np.ones(count, dtype=np.bool_)
            observation, canonical, physical, neural, bodies = observe(pool, brain, 0.05)
            observations = [observation]
            canonicals = [canonical]
            neurals = [neural]
            resets = [reset.copy()]
            actions = []
            oral_commands = []
            outcomes_sequence = []
            for tick in range(args.steps):
                result = residents.step(
                    observation,
                    neural,
                    physical,
                    previous,
                    np.full(count, tick, dtype=np.uint64),
                    np.full(count, tick * 0.05, dtype=np.float64),
                    reset,
                )
                action = np.ascontiguousarray(result["proposed_action"], dtype=np.float32)
                oral = np.ascontiguousarray(result["oral_command"], dtype=np.float32)
                before = physical.copy()
                advanced = pool.advance(action_payload(action, oral, bodies))
                world_outcomes = [value[0] for value in advanced]
                observation, canonical, physical, neural, bodies = observe(pool, brain, 0.05)
                executed = np.ascontiguousarray(
                    np.concatenate((action, oral[:, None]), axis=1), dtype=np.float32
                )
                effort = np.asarray(
                    [
                        world_outcomes[world][str(body["id"])]["effort"]
                        for world, world_bodies in enumerate(bodies)
                        for body in world_bodies
                    ],
                    dtype=np.float32,
                )
                residents.observe_consequences(
                    np.full(count, tick, dtype=np.uint64),
                    before,
                    physical,
                    executed,
                    effort,
                    dt=0.05,
                )
                previous = executed
                reset = np.zeros(count, dtype=np.bool_)
                observations.append(observation)
                canonicals.append(canonical)
                neurals.append(neural)
                resets.append(reset.copy())
                actions.append(action)
                oral_commands.append(oral)
                outcomes_sequence.append(outcome_rows(world_outcomes, bodies))
            arrays = {
                "observation": np.stack(observations).astype(np.float32),
                "canonical_channels": np.stack(canonicals).astype(np.float32),
                "neural_readouts": np.stack(neurals).astype(np.float32),
                "executed_actions": np.stack(actions).astype(np.float32),
                "oral_command": np.stack(oral_commands).astype(np.float32),
                "transition_outcomes": np.stack(outcomes_sequence).astype(np.float32),
                "reset": np.stack(resets).astype(np.bool_),
                "dt_seconds": np.asarray(0.05, dtype=np.float64),
            }
            packet = atomic_npz(args.output / f"episode-{episode:03d}.npz", arrays)
            packets.append(
                {
                    "episode": episode,
                    "stage": 0,
                    **packet,
                    "model_array_shapes": {
                        key: list(value.shape) for key, value in arrays.items()
                    },
                    "resident_partitions": resident_ids,
                }
            )
            checkpoint_dir = args.output / f"episode-{episode:03d}-end"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            neural_receipt = brain.snapshot(checkpoint_dir, "neural")
            world_receipt = atomic_json(checkpoint_dir / "worlds.json", pool.snapshot())
            private_receipt = atomic_json(
                checkpoint_dir / "developmental-private.json",
                residents.snapshot_value(),
            )
            end_checkpoints.append(
                {
                    "episode": episode,
                    "worlds": world_receipt,
                    "neural": neural_receipt,
                    "private": private_receipt,
                }
            )
        transport_timing = pool.timing_snapshot()
    finally:
        pool.close()
    manifest = {
        "format": FORMAT,
        "version": 2,
        "completed": True,
        "collection_identity": identity,
        "identity_receipt": identity_receipt,
        "collection_identity_sha256": identity["sha256"],
        "scope": {
            "worlds": args.worlds,
            "residents_per_world": 6,
            "episodes": args.episodes,
            "steps_per_episode": args.steps,
            "dt_seconds": 0.05,
            "train_world_slots": identity["split"]["train_world_slots"],
            "heldout_world_slots": identity["split"]["heldout_world_slots"],
        },
        "schema": {
            "path": str(SCHEMA.relative_to(ROOT)),
            "bytes": SCHEMA.stat().st_size,
            "sha256": sha256(SCHEMA),
        },
        "profile": profile.to_value(),
        "rich_sensorium": {
            "profile_sha256": RICH_SENSORIUM_PROFILE_SHA256,
            "channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
            "observation_order": list(OBSERVATION_ORDER),
        },
        "transition_outcome_order": list(OUTCOME_FIELDS),
        "packets": packets,
        "end_checkpoints": end_checkpoints,
        "transitions": args.worlds * 6 * args.episodes * args.steps,
        "elapsed_seconds": time.perf_counter() - started,
        "world_transport_timing": transport_timing,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
