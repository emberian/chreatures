#!/usr/bin/env python3
"""Collect current regional population trajectories from native developmental residents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.organism_interface import (
    ACTION_DIM,
    ACTION_NAMES,
    OBSERVATION_DIM,
    OBSERVATION_ORDER,
    PHYSIOLOGY_DIM,
    PREVIOUS_DIM,
)
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

FORMAT = "chreatures-sensorimotor-play-rich-v4"
VERSION = 4
SHARD_STEPS = 512
CHECKPOINT_STEPS = 1024
SCHEMA = ROOT / "research/sensorimotor_skills/trajectory-schema-rich-v4.json"


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
        Path("research/sensorimotor_skills/trajectory-schema-rich-v4.json"),
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
    parser.add_argument("--candidate-genomes", type=Path, required=True)
    parser.add_argument("--neural-recipe", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--chemical-habitat", type=Path, required=True)
    parser.add_argument("--chemical-biosphere", type=Path, required=True)
    parser.add_argument("--nursery-family-config", type=Path, required=True)
    parser.add_argument("--nursery-family-schedule", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--validation-worlds", type=int, default=1)
    parser.add_argument("--heldout-worlds", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--brain-backend", choices=("tiled", "triton"), default="tiled")
    parser.add_argument("--physical-backend", choices=("fast",), default="fast")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if (
        not 3 <= args.worlds <= 16
        or not 1 <= args.validation_worlds
        or not 1 <= args.heldout_worlds
        or args.validation_worlds + args.heldout_worlds >= args.worlds
    ):
        raise SystemExit("world split requires train, validation, and final holdout worlds")
    if not 1 <= args.episodes <= 64 or not 44 <= args.steps <= 100_000:
        raise SystemExit("episodes or steps outside bounded training-data range")
    if args.steps % SHARD_STEPS:
        raise SystemExit("steps must be divisible by the fixed 512-tick shard size")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    for path in (
        args.resident_artifact,
        args.candidate_genomes,
        args.neural_recipe,
        args.port_bundle,
        args.chemical_habitat,
        args.chemical_biosphere,
        args.nursery_family_config,
        args.nursery_family_schedule,
    ):
        if not path.is_file():
            raise SystemExit(f"required artifact does not exist: {path}")


def observe(pool, brain, dt: float):
    rich, canonical, bodies = pool.observe_arrays()
    neural, circuit, _ = brain.step_channels(canonical, dt)
    physical = pool.physiology_array(circuit[:, 2])
    observation = np.ascontiguousarray(
        np.concatenate((rich, canonical, physical), axis=1), dtype=np.float32
    )
    return observation, canonical, physical, neural, bodies


def action_payload(actions: np.ndarray, bodies):
    payloads = []
    row = 0
    for world in bodies:
        mapped = {}
        for body in world:
            mapped[str(body["id"])] = {
                **dict(zip(ACTION_NAMES, actions[row].astype(float), strict=True)),
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
    from chreatures.training_environment import EmbodiedTrainingProfile, PROFILE_VERSION

    profile = EmbodiedTrainingProfile.nursery_family(
        args.chemical_habitat,
        args.chemical_biosphere,
        args.nursery_family_config,
        args.nursery_family_schedule,
    )
    graph = load_training_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    transport = profile.component("family")["transport"]
    residents_per_world = transport["residents"]
    required = {
        "residents": residents_per_world,
        "rich": RICH_OBSERVATION_CHANNELS,
        "physical": len(ports.input_names),
        "physiology": PHYSIOLOGY_DIM,
        "controller": OBSERVATION_DIM,
        "readouts": len(ports.readout_names),
        "actions": ACTION_DIM,
    }
    if (
        int(profile.component("version")) != PROFILE_VERSION
        or any(transport.get(name) != value for name, value in required.items())
        or type(transport.get("max_residents")) is not int
        or transport["max_residents"] < residents_per_world
    ):
        raise SystemExit("regional profile transport differs from collector interfaces")
    count = args.worlds * residents_per_world
    from chreatures.population import CandidateGenome
    from chreatures.neural_genotype import NeuralVariantRecipe, compile_population_phenotypes

    candidate_values = json.loads(args.candidate_genomes.read_text())
    if not isinstance(candidate_values, list) or len(candidate_values) != count:
        raise ValueError("collection requires one inherited candidate genome per cohort row")
    candidates = [CandidateGenome(value) for value in candidate_values]
    phenotypes = compile_population_phenotypes(
        candidates, NeuralVariantRecipe.load(args.neural_recipe), graph, ports,
        sha256(args.port_bundle), sha256(args.resident_artifact),
    )
    brain = TrainingCohortBrain(
        graph, ports, count, device=args.device, backend=args.brain_backend
    )
    brain.bind_phenotypes(phenotypes)
    pool = WorldTrainingPool(
        args.worlds,
        dict(ports.spec),
        profile.to_value(),
        args.physical_backend,
        residents_per_world=residents_per_world,
    )
    artifact_path = args.resident_artifact.resolve()
    identity = {
        "format": f"{FORMAT}-identity",
        "source": source_identity(),
        "resident_artifact": resident_model_identity(artifact_path),
        "candidate_genomes_sha256": sha256(args.candidate_genomes),
        "candidate_order": [candidate.sha256 for candidate in candidates],
        "neural_phenotypes": [phenotype.sha256 for phenotype in phenotypes],
        "profile": profile.to_value(),
        "graph_sha256": str(graph.hash),
        "port_spec_sha256": ports.spec_hash,
        "port_bundle_sha256": sha256(args.port_bundle.resolve()),
        "rich_profile_sha256": RICH_SENSORIUM_PROFILE_SHA256,
        "rich_channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
        "observation_order": list(OBSERVATION_ORDER),
        "transition_outcome_order": list(OUTCOME_FIELDS),
        "organ_flow_order": ["release", "secretion", "allocation"],
        "split": {
            "train_world_slots": list(
                range(args.worlds - args.validation_worlds - args.heldout_worlds)
            ),
            "validation_world_slots": list(
                range(
                    args.worlds - args.validation_worlds - args.heldout_worlds,
                    args.worlds - args.heldout_worlds,
                )
            ),
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
    progress_sequence = 0

    def publish_progress() -> None:
        nonlocal progress_sequence
        progress_sequence += 1
        progress = {
            "format": f"{FORMAT}-progress",
            "version": VERSION,
            "completed": False,
            "sequence": progress_sequence,
            "collection_identity_sha256": identity["sha256"],
            "identity_receipt": identity_receipt,
            "packets": packets,
            "checkpoints": end_checkpoints,
        }
        progress["content_sha256"] = canonical_hash(progress)
        atomic_json(args.output / "progress.json", progress)

    transport_timing = None
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            heldout_start = args.worlds - args.validation_worlds - args.heldout_worlds
            bodies = pool.reset(
                [
                    {
                        "seed": args.seed + episode * 1009 + world,
                        "held_out": world >= heldout_start,
                        "candidates": candidate_values[
                            world * residents_per_world:(world + 1) * residents_per_world
                        ],
                    }
                    for world in range(args.worlds)
                ]
            )
            resident_ids = [
                f"episode-{episode:03d}/world-{world:03d}/resident-{resident:02d}"
                for world in range(args.worlds)
                for resident in range(residents_per_world)
            ]
            brain.reset_residents(resident_ids)
            residents = DevelopmentalResidentCohort(
                artifact_path,
                count,
                action_mode="sample",
                goal_seed=args.seed ^ (episode << 24) ^ 0x60A1,
                action_seed=args.seed ^ (episode << 24) ^ 0xAC71,
                candidate_adapters=[candidate.controller_adapter() for candidate in candidates],
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
            previous = np.zeros((count, PREVIOUS_DIM), dtype=np.float32)
            reset = np.ones(count, dtype=np.bool_)
            observation, canonical, physical, neural, bodies = observe(pool, brain, 0.05)
            for shard, start_tick in enumerate(range(0, args.steps, SHARD_STEPS)):
                stop_tick = start_tick + SHARD_STEPS
                buffer_dir = args.output / (
                    f".episode-{episode:03d}-shard-{shard:03d}.buffers"
                )
                buffer_dir.mkdir()

                def shard_buffer(name, shape, dtype):
                    return np.lib.format.open_memmap(
                        buffer_dir / f"{name}.npy",
                        mode="w+",
                        dtype=dtype,
                        shape=shape,
                    )

                observations = shard_buffer(
                    "observation",
                    (SHARD_STEPS + 1, count, OBSERVATION_DIM),
                    np.float32,
                )
                canonicals = shard_buffer(
                    "canonical_channels",
                    (SHARD_STEPS + 1, count, len(ports.input_names)),
                    np.float32,
                )
                neurals = shard_buffer(
                    "neural_readouts",
                    (SHARD_STEPS + 1, count, len(ports.readout_names)),
                    np.float32,
                )
                resets = shard_buffer(
                    "reset", (SHARD_STEPS + 1, count), np.bool_
                )
                resets[:] = False
                actions = shard_buffer(
                    "executed_actions", (SHARD_STEPS, count, ACTION_DIM), np.float32
                )
                worker_contexts = shard_buffer(
                    "worker_recurrent_context", (SHARD_STEPS, count, 128), np.float32
                )
                outcomes_sequence = shard_buffer(
                    "transition_outcomes",
                    (SHARD_STEPS, count, len(OUTCOME_FIELDS)),
                    np.float32,
                )
                organ_flows = shard_buffer(
                    "organ_flows", (SHARD_STEPS, count, 3), np.float32
                )
                observations[0] = observation
                canonicals[0] = canonical
                neurals[0] = neural
                resets[0] = reset
                for local_tick, tick in enumerate(range(start_tick, stop_tick)):
                    result = residents.step(
                        observation,
                        neural,
                        physical,
                        previous,
                        np.full(count, tick, dtype=np.uint64),
                        np.full(count, tick * 0.05, dtype=np.float64),
                        reset,
                    )
                    action = np.ascontiguousarray(
                        result["proposed_action"], dtype=np.float32
                    )
                    worker_context = np.ascontiguousarray(
                        result["worker_recurrent_context"], dtype=np.float32
                    ).copy()
                    if worker_context.shape != (count, 128) or not np.isfinite(
                        worker_context
                    ).all():
                        raise RuntimeError("native effective worker context differs")
                    before = physical.copy()
                    advanced = pool.advance(action_payload(action, bodies))
                    world_outcomes = [value[0] for value in advanced]
                    observation, canonical, physical, neural, bodies = observe(
                        pool, brain, 0.05
                    )
                    executed = np.ascontiguousarray(action, dtype=np.float32)
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
                    observations[local_tick + 1] = observation
                    canonicals[local_tick + 1] = canonical
                    neurals[local_tick + 1] = neural
                    resets[local_tick + 1] = reset
                    actions[local_tick] = executed
                    worker_contexts[local_tick] = worker_context
                    outcomes_sequence[local_tick] = outcome_rows(
                        world_outcomes, bodies
                    )
                    organ_flows[local_tick] = pool.organ_flows_array()
                arrays = {
                    "observation": observations,
                    "canonical_channels": canonicals,
                    "neural_readouts": neurals,
                    "executed_actions": actions,
                    "worker_recurrent_context": worker_contexts,
                    "transition_outcomes": outcomes_sequence,
                    "organ_flows": organ_flows,
                    "reset": resets,
                    "dt_seconds": np.asarray(0.05, dtype=np.float64),
                }
                for value in arrays.values():
                    if isinstance(value, np.memmap):
                        value.flush()
                array_shapes = {
                    key: list(value.shape) for key, value in arrays.items()
                }
                packet_path = args.output / (
                    f"episode-{episode:03d}-shard-{shard:03d}.npz"
                )
                packet = atomic_npz(packet_path, arrays)
                for value in arrays.values():
                    if isinstance(value, np.memmap):
                        value._mmap.close()
                for path in buffer_dir.iterdir():
                    path.unlink()
                buffer_dir.rmdir()
                packets.append(
                    {
                        "episode": episode,
                        "shard": shard,
                        "start_tick": start_tick,
                        "stop_tick": stop_tick,
                        "stage": 0,
                        **packet,
                        "model_array_shapes": array_shapes,
                        "resident_partitions": resident_ids,
                    }
                )
                publish_progress()
                if stop_tick % CHECKPOINT_STEPS == 0 or stop_tick == args.steps:
                    checkpoint_dir = args.output / (
                        f"episode-{episode:03d}-tick-{stop_tick:08d}"
                    )
                    checkpoint_dir.mkdir(parents=True, exist_ok=False)
                    neural_receipt = brain.snapshot(checkpoint_dir, "neural")
                    neural_receipt["path"] = "neural.npz"
                    neural_receipt.pop("name", None)
                    files = {
                        "boundary": atomic_npz(
                            checkpoint_dir / "boundary.npz",
                            {
                                "observation": observation,
                                "canonical_channels": canonical,
                                "physiology": physical,
                                "neural_readouts": neural,
                                "previous_action": previous,
                                "reset": reset,
                                "tick": np.asarray(stop_tick, dtype=np.uint64),
                                "time_seconds": np.asarray(
                                    stop_tick * 0.05, dtype=np.float64
                                ),
                            },
                        ),
                        "worlds": atomic_json(
                            checkpoint_dir / "worlds.json", pool.snapshot()
                        ),
                        "neural": neural_receipt,
                        "private": atomic_json(
                            checkpoint_dir / "developmental-private.json",
                            residents.snapshot_value(),
                        ),
                    }
                    end_checkpoints.append(
                        {
                            "episode": episode,
                            "tick": stop_tick,
                            "path": checkpoint_dir.name,
                            "files": files,
                            "tree_sha256": canonical_hash(files),
                        }
                    )
                    publish_progress()
        transport_timing = pool.timing_snapshot()
    finally:
        pool.close()
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "completed": True,
        "collection_identity": identity,
        "identity_receipt": identity_receipt,
        "collection_identity_sha256": identity["sha256"],
        "scope": {
            "worlds": args.worlds,
            "residents_per_world": residents_per_world,
            "episodes": args.episodes,
            "steps_per_episode": args.steps,
            "shard_steps": SHARD_STEPS,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "dt_seconds": 0.05,
            "train_world_slots": identity["split"]["train_world_slots"],
            "validation_world_slots": identity["split"]["validation_world_slots"],
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
        "organ_flow_order": ["release", "secretion", "allocation"],
        "packets": packets,
        "checkpoints": end_checkpoints,
        "transitions": count * args.episodes * args.steps,
        "elapsed_seconds": time.perf_counter() - started,
        "world_transport_timing": transport_timing,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
