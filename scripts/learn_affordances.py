#!/usr/bin/env python3
"""Train and evaluate predictive PPO through rich MaleCNS ports and 3-D bodies."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shlex
import signal
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if mp.current_process().name == "MainProcess":
    import torch
    from chreatures.learning import (
        ACTIONS, MacroRollout, PredictivePPOConfig, PredictivePPOTrainer,
        RunningMoments,
    )
    from chreatures.malecns import MaleCNSGraph
    from chreatures.neural_ports import NeuralPortBundle
    from chreatures.remote_brain import RemoteBrain
    from chreatures.fast_circuit import MicrobatchedResidentCircuit


HABITAT = ROOT / "data/habitats/hollow-garden.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--episode-steps", type=int, default=400)
    parser.add_argument("--macro-steps", type=int, default=5)
    parser.add_argument("--rollout-decisions", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=4_000)
    parser.add_argument(
        "--first-checkpoint", type=int, default=0,
        help="write one early full checkpoint, then use --checkpoint-every",
    )
    parser.add_argument("--eval-worlds", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=800)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--brain-backend", choices=("microbatch", "reference"), default="microbatch")
    parser.add_argument("--microbatch-size", type=int, default=3)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--restore-audit-only", action="store_true",
        help="load an exact full checkpoint, write restore-audit.json, and exit",
    )
    parser.add_argument(
        "--warm-start-learner", type=Path,
        help="reuse shared model/optimizer/moments, resetting all resident/world/neural state",
    )
    return parser.parse_args()


def _safe(value: float, high: float, margin: float = 0.35) -> float:
    return float(np.clip(value, margin, high - margin))


def affordance_spec(seed: int, episode: int, *, held_out: bool = False) -> dict[str, Any]:
    """Create adequate-food layouts; positions never enter learner observations."""
    spec = json.loads(HABITAT.read_text())
    rng = np.random.default_rng(seed + episode * 104729 + (80_000_003 if held_out else 0))
    width, height = map(float, spec["size"][:2])
    resources = ("berry-a", "nectar-a", "berry-b")
    by_id = {entity["id"]: entity for entity in spec["entities"]}
    for index, (body, resource_id) in enumerate(zip(spec["bodies"], resources, strict=True)):
        # Keep training cohorts separated enough to avoid social crowding while
        # varying egocentric resource angle and local geometry.
        body["position"][0] = _safe(body["position"][0] + rng.uniform(-0.35, 0.35), width)
        body["position"][1] = _safe(body["position"][1] + rng.uniform(-0.35, 0.35), height)
        body["position"][2] = 0.18
        body["heading"] = float(rng.uniform(-math.pi, math.pi))
        body["energy"] = float(rng.uniform(0.76, 0.84))
        body["gut"] = float(rng.uniform(0.08, 0.16))
        body["fatigue"] = float(rng.uniform(0.02, 0.06))
        bearing_span = math.pi if held_out or episode >= 4 else 0.75
        angle = body["heading"] + float(rng.uniform(-bearing_span, bearing_span))
        distance = float(rng.uniform(0.24, 0.46))
        resource = by_id[resource_id]
        resource["position"] = [
            _safe(body["position"][0] + math.cos(angle) * distance, width),
            _safe(body["position"][1] + math.sin(angle) * distance, height),
            0.14,
        ]
        # Some episodes put a movable object in the nearby perceptual/contact
        # field. It carries no bonus or object label in the learning stream.
        ball = by_id[("violet-ball", "cyan-ball", "stack-box-a")[index]]
        ball_angle = body["heading"] + float(rng.uniform(-1.0, 1.0))
        ball_distance = float(rng.uniform(0.34, 0.62))
        ball["position"][0] = _safe(body["position"][0] + math.cos(ball_angle) * ball_distance, width)
        ball["position"][1] = _safe(body["position"][1] + math.sin(ball_angle) * ball_distance, height)
    dx, dy = rng.uniform(-0.18, 0.18, size=2)
    for entity_id in ("high-walk", "west-ramp", "east-ramp"):
        entity = by_id[entity_id]
        entity["position"][0] = _safe(entity["position"][0] + dx, width)
        entity["position"][1] = _safe(entity["position"][1] + dy, height)
    entity = by_id["hollow-arch"]
    entity["position"][0] = _safe(entity["position"][0] + rng.uniform(-0.18, 0.18), width)
    entity["position"][1] = _safe(entity["position"][1] + rng.uniform(-0.18, 0.18), height)
    spec["name"] = "articulated-rich-affordance-heldout" if held_out else "articulated-rich-affordance-training"
    return spec


def _world_worker(connection, port_spec: dict[str, Any]) -> None:
    """Own one MuJoCo instance so native and Python work spans CPU cores."""
    from chreatures.neural_ports import encode_physical_senses
    from chreatures.sensorium import ArticulatedSensoriumWorld
    world = None
    try:
        while True:
            operation, payload = connection.recv()
            if operation == "close":
                connection.send((True, None))
                return
            if operation == "reset":
                world = ArticulatedSensoriumWorld(seed=payload["seed"], spec=payload["spec"])
                result = [body.to_dict() for body in world.bodies]
            elif operation == "restore":
                world = ArticulatedSensoriumWorld.restore(payload)
                result = [body.to_dict() for body in world.bodies]
            elif operation == "observe":
                vectors = [
                    encode_physical_senses(world.sense(body.id), port_spec)[1]
                    for body in world.bodies
                ]
                result = (np.stack(vectors), [body.to_dict() for body in world.bodies])
            elif operation == "advance":
                outcome = world.advance(payload["actions"], payload["dt"])
                result = (outcome, [body.to_dict() for body in world.bodies])
            elif operation == "snapshot":
                result = world.snapshot()
            else:
                raise ValueError(f"unknown world worker operation {operation}")
            connection.send((True, result))
    except BaseException as exc:
        connection.send((False, {"error": repr(exc), "traceback": traceback.format_exc()}))
    finally:
        connection.close()


class ProcessWorldPool:
    def __init__(self, count: int, port_spec: dict[str, Any]) -> None:
        context = mp.get_context("spawn")
        self.connections = []
        self.processes = []
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_world_worker, args=(child, port_spec), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def call_all(self, operation: str, payloads: list[Any] | None = None) -> list[Any]:
        payloads = payloads if payloads is not None else [None] * len(self.connections)
        if len(payloads) != len(self.connections):
            raise ValueError("world worker payload count differs")
        for connection, payload in zip(self.connections, payloads, strict=True):
            connection.send((operation, payload))
        results = []
        for connection in self.connections:
            ok, value = connection.recv()
            if not ok:
                raise RuntimeError(f"world worker failed: {value['error']}\n{value['traceback']}")
            results.append(value)
        return results

    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for connection in self.connections:
            try:
                connection.recv()
            except (BrokenPipeError, EOFError):
                pass
            connection.close()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


class FixedCohortBrain:
    """Resident-ID and checkpoint shell around the measured fixed GPU circuit."""

    def __init__(
        self, graph: Any, ports: Any, batch_size: int, *, device: str, microbatch_size: int
    ) -> None:
        self.graph = graph
        self.ports = ports
        self.graph_hash = str(graph.hash)
        self.device = torch.device(device)
        self.capacity = batch_size
        self.resident_ids: list[str] = []
        self.circuit = MicrobatchedResidentCircuit(
            graph, batch_size, device=device, microbatch_size=microbatch_size,
            input_map=(ports.input_names, ports.input_map),
            readout_map=(ports.readout_names, ports.readout_map),
        )

    def add_residents(self, resident_ids: list[str]) -> None:
        clean = [str(value) for value in resident_ids]
        if self.resident_ids or not clean or len(clean) > self.capacity or len(set(clean)) != len(clean):
            raise ValueError("fixed circuit requires one unique prefix cohort within capacity")
        self.resident_ids = clean
        self.circuit.reset()

    def remove_residents(self, resident_ids: list[str]) -> None:
        if list(resident_ids) != self.resident_ids:
            raise ValueError("fixed circuit can reset only its complete ordered cohort")
        self.resident_ids = []
        self.circuit.reset()

    def step_channels(
        self, channels: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
        values = np.asarray(channels, dtype=np.float32)
        active = len(self.resident_ids)
        if values.shape != (active, self.circuit.input_count):
            raise ValueError("fixed circuit channel batch has the wrong shape")
        if active < self.capacity:
            padded = np.zeros((self.circuit.input_count, self.capacity), dtype=np.float32)
            padded[:, :active] = values.T
            device_input = padded
        else:
            device_input = np.ascontiguousarray(values.T)
        result = self.circuit.step_numpy(device_input, dt)
        physiology = result.physiology[:active]
        neural = [
            {
                "activity": float(row[0]), "activity_peak": float(row[1]),
                "support": float(row[2]),
            }
            for row in physiology
        ]
        return result.features[:active].copy(), physiology.copy(), neural

    def snapshot(self, directory: Path, name: str) -> dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.npz"
        temporary = path.with_name(path.name + ".tmp")
        state = self.circuit.export_state()
        metadata = {
            "version": 1, "engine": "fixed-microbatch-learning-v1",
            "graph_sha256": self.graph_hash, "resident_ids": self.resident_ids,
            "circuit": self.circuit.metadata(),
        }
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata)), **state)
        os.replace(temporary, path)
        return {"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)}

    def restore(self, directory: Path, name: str, expected_sha256: str | None = None) -> dict[str, Any]:
        path = Path(directory) / f"{name}.npz"
        digest = sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("fixed circuit snapshot checksum differs")
        with np.load(path, allow_pickle=False) as value:
            metadata = json.loads(str(value["metadata"]))
            if metadata.get("graph_sha256") != self.graph_hash:
                raise ValueError("fixed circuit snapshot graph differs")
            residents = [str(item) for item in metadata["resident_ids"]]
            state = {
                key: np.asarray(value[key]) for key in ("rates", "adaptation", "support", "times")
            }
        if not residents or len(residents) > self.capacity:
            raise ValueError("fixed circuit snapshot cohort size differs")
        self.resident_ids = residents
        self.circuit.import_state(state)
        return {"name": name, "bytes": path.stat().st_size, "sha256": digest, "residents": residents}

    def metadata(self) -> dict[str, Any]:
        value = self.circuit.metadata()
        value["device"] = {
            "type": self.device.type,
            "name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "memory_allocated_bytes": (
                torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
            ),
        }
        value["residents"] = self.resident_ids
        return value


class AffordanceCohort:
    """Independent articulated worlds sharing one full rich-port GPU circuit."""

    def __init__(self, brain: RemoteBrain, ports: NeuralPortBundle, worlds: int, workers: int, seed: int) -> None:
        self.brain = brain
        self.ports = ports
        self.world_count = worlds
        self.seed = seed
        self.episode = 0
        self.world_pool = ProcessWorldPool(worlds, ports.spec)
        self.timings = {name: 0.0 for name in ("world_build", "sense_encode", "brain", "physics")}
        self.body_states: list[list[dict[str, Any]]] = []
        self.resident_ids: list[str] = []
        self.reset(0)

    def reset(self, episode: int, *, held_out: bool = False) -> None:
        started = time.perf_counter()
        old_ids = self.brain.resident_ids
        if old_ids:
            self.brain.remove_residents(old_ids)
        self.episode = episode
        payloads = [
            {
                "seed": self.seed + episode * 1009 + index,
                "spec": affordance_spec(self.seed + index * 17, episode, held_out=held_out),
            }
            for index in range(self.world_count)
        ]
        self.body_states = self.world_pool.call_all("reset", payloads)
        prefix = "eval" if held_out else "train"
        self.resident_ids = [
            f"{prefix}-w{world_index:02d}:{body['id']}"
            for world_index, bodies in enumerate(self.body_states)
            for body in bodies
        ]
        self.brain.add_residents(self.resident_ids)
        self.timings["world_build"] += time.perf_counter() - started

    def observe(self, dt: float) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        started = time.perf_counter()
        observed = self.world_pool.call_all("observe")
        vectors = [value[0] for value in observed]
        self.body_states = [value[1] for value in observed]
        channel_rows = []
        entries = []
        physiology = []
        for world_index, bodies in enumerate(self.body_states):
            for body_index, body in enumerate(bodies):
                channel_rows.append(vectors[world_index][body_index])
                entries.append({
                    "id": self.resident_ids[world_index * 3 + body_index],
                    "senses": dict(zip(
                        self.ports.input_names,
                        vectors[world_index][body_index].astype(float).tolist(),
                        strict=True,
                    )),
                })
                physiology.append([
                    body["energy"], body["gut"], body["fatigue"],
                    math.tanh(body["speed"] / 2), math.tanh(body["angular_velocity"] / 4), 1.0,
                ])
        self.timings["sense_encode"] += time.perf_counter() - started
        started = time.perf_counter()
        if hasattr(self.brain, "step_channels"):
            features, circuit_physiology, neural = self.brain.step_channels(
                np.asarray(channel_rows, dtype=np.float32), dt
            )
        else:
            neural = self.brain.step(entries, dt)
            features = np.asarray([value["features"] for value in neural], dtype=np.float32)
            circuit_physiology = np.asarray([
                [value["activity"], value["activity_peak"], value["support"]]
                for value in neural
            ], dtype=np.float32)
        self.timings["brain"] += time.perf_counter() - started
        physiology_array = np.asarray(physiology, dtype=np.float32)
        physiology_array[:, 5] = circuit_physiology[:, 2]
        return features, physiology_array, neural

    def advance(self, action_values: np.ndarray, dt: float) -> tuple[np.ndarray, dict[str, float]]:
        before_energy = np.asarray(
            [body["energy"] for bodies in self.body_states for body in bodies], dtype=np.float32
        )
        actions_by_world = []
        for world_index, bodies in enumerate(self.body_states):
            actions = {}
            for body_index, body in enumerate(bodies):
                vector = action_values[world_index * 3 + body_index]
                action = dict(zip(ACTIONS, vector.astype(float).tolist(), strict=True))
                for name in ("grip", "signal_low", "signal_mid", "signal_high"):
                    action[name] = max(0.0, action[name])
                action["eat"] = float(np.clip((1 - body["gut"]) * (1.1 - body["energy"]), 0, 1))
                actions[body["id"]] = action
            actions_by_world.append(actions)
        started = time.perf_counter()
        advanced = self.world_pool.call_all("advance", [
            {"actions": actions, "dt": dt} for actions in actions_by_world
        ])
        outcomes = [value[0] for value in advanced]
        self.body_states = [value[1] for value in advanced]
        self.timings["physics"] += time.perf_counter() - started
        after_energy = np.asarray(
            [body["energy"] for bodies in self.body_states for body in bodies], dtype=np.float32
        )
        nutrition, contact, distance, effort = [], [], [], []
        for bodies, result in zip(self.body_states, outcomes, strict=True):
            for body in bodies:
                value = result[body["id"]]
                nutrition.append(value["nutrition"])
                contact.append(value["contact"])
                distance.append(value["distance"])
                effort.append(value["effort"])
        nutrition = np.asarray(nutrition, dtype=np.float32)
        effort = np.asarray(effort, dtype=np.float32)
        old_drive = (0.85 - before_energy) ** 2
        new_drive = (0.85 - after_energy) ** 2
        reward = (
            (old_drive - new_drive) * 12
            + nutrition * np.maximum(0, 1 - after_energy) * 3
            - effort * np.float32(0.0002 * dt)
        ).astype(np.float32)
        return reward, {
            "nutrition": float(nutrition.sum()),
            "nutrition_events": float(np.count_nonzero(nutrition > 0)),
            "contacts": float(np.count_nonzero(np.asarray(contact) > 0)),
            "distance": float(np.sum(distance)),
            "effort": float(effort.mean()),
            "energy": float(after_energy.mean()),
        }

    def close(self) -> None:
        self.world_pool.close()


def save_checkpoint(
    output: Path, cohort: AffordanceCohort, trainer: PredictivePPOTrainer,
    step: int, episode_step: int, features: np.ndarray, physiology: np.ndarray,
) -> dict[str, Any]:
    directory = output / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    tag = f"step-{step:07d}"
    neural = cohort.brain.snapshot(directory, f"neural-{tag}")
    learner = trainer.snapshot(directory / f"learner-{tag}.pt")
    state = {
        "version": 1, "step": step, "episode": cohort.episode,
        "episode_step": episode_step, "resident_ids": cohort.resident_ids,
        "graph_sha256": cohort.brain.graph_hash, "port_spec_sha256": cohort.ports.spec_hash,
        "neural": neural, "learner": learner,
        "worlds": cohort.world_pool.call_all("snapshot"),
        "features": features.tolist(), "physiology": physiology.tolist(),
        "cohort": {"worlds": cohort.world_count, "seed": cohort.seed},
    }
    path = directory / f"cohort-{tag}.json.gz"
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(state, handle, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, path)
    return {
        "step": step, "cohort": path.name, "cohort_bytes": path.stat().st_size,
        "cohort_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "neural": neural, "learner": learner,
    }


def restore_checkpoint(
    path: Path, brain: RemoteBrain, ports: NeuralPortBundle, workers: int,
) -> tuple[AffordanceCohort, PredictivePPOTrainer, int, int, np.ndarray, np.ndarray]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        state = json.load(handle)
    if (
        state.get("version") != 1
        or state.get("graph_sha256") != brain.graph_hash
        or state.get("port_spec_sha256") != ports.spec_hash
    ):
        raise ValueError("training checkpoint graph or rich ports differ")
    neural = state["neural"]
    brain.restore(path.parent, neural["name"], neural["sha256"])
    cohort = AffordanceCohort.__new__(AffordanceCohort)
    cohort.brain = brain
    cohort.ports = ports
    cohort.world_count = int(state["cohort"]["worlds"])
    cohort.seed = int(state["cohort"]["seed"])
    cohort.episode = int(state["episode"])
    cohort.world_pool = ProcessWorldPool(cohort.world_count, ports.spec)
    cohort.timings = {name: 0.0 for name in ("world_build", "sense_encode", "brain", "physics")}
    cohort.body_states = cohort.world_pool.call_all("restore", state["worlds"])
    cohort.resident_ids = [str(value) for value in state["resident_ids"]]
    if cohort.resident_ids != brain.resident_ids:
        raise ValueError("checkpoint neural and physical resident order differs")
    learner_path = path.parent / Path(state["learner"]["path"]).name
    trainer, _ = PredictivePPOTrainer.restore(
        learner_path, device=brain.device,
        expected_sha256=state["learner"]["sha256"],
    )
    if trainer.resident_ids != cohort.resident_ids:
        raise ValueError("checkpoint learner residents differ")
    features = np.asarray(state["features"], dtype=np.float32)
    physiology = np.asarray(state["physiology"], dtype=np.float32)
    return (
        cohort, trainer, int(state["step"]), int(state["episode_step"]),
        features, physiology,
    )


def evaluate(
    brain: RemoteBrain, ports: NeuralPortBundle, genome: Path,
    moments: RunningMoments, *, worlds: int, steps: int, macro_steps: int,
    workers: int, seed: int, silence_features: bool,
) -> dict[str, float]:
    cohort = AffordanceCohort(brain, ports, worlds, workers, seed)
    cohort.reset(0, held_out=True)
    config = PredictivePPOConfig(feature_dim=len(ports.readout_names), macro_steps=macro_steps, seed=seed)
    trainer = PredictivePPOTrainer(cohort.resident_ids, config, device=brain.device)
    trainer.import_genome(genome)
    trainer.moments = RunningMoments.restore(moments.snapshot())
    raw, physiology, _ = cohort.observe(0.05)
    normalized = trainer.normalize(raw, update=False)
    totals = {name: 0.0 for name in ("nutrition", "nutrition_events", "contacts", "distance")}
    efforts, energies, rewards = [], [], []
    try:
        for _ in range(0, steps, macro_steps):
            previous = trainer.act(
                normalized, physiology, deterministic=True,
                silence_features=silence_features,
            )
            accumulated = np.zeros(len(cohort.resident_ids), dtype=np.float32)
            for _substep in range(macro_steps):
                reward, metrics = cohort.advance(previous["action"], 0.05)
                accumulated += reward
                raw, physiology, _ = cohort.observe(0.05)
                for name in totals:
                    totals[name] += metrics[name]
                efforts.append(metrics["effort"])
                energies.append(metrics["energy"])
            normalized = trainer.normalize(raw, update=False)
            finish_features = np.zeros_like(normalized) if silence_features else normalized
            trainer.finish_transition(
                previous, finish_features, accumulated,
                np.zeros(len(cohort.resident_ids), dtype=bool), macro_steps * 0.05,
            )
            rewards.extend(accumulated.tolist())
    finally:
        cohort.close()
    return {
        **totals,
        "effort_mean": float(np.mean(efforts)),
        "energy_final": float(energies[-1]),
        "reward_total": float(np.sum(rewards)),
        "reward_mean_per_macro_resident": float(np.mean(rewards)),
    }


def main() -> int:
    args = arguments()
    if not 1 <= args.worlds <= 16 or not 1 <= args.eval_worlds <= args.worlds:
        raise SystemExit("world counts must satisfy 1 <= eval <= train <= 16")
    if args.steps % args.macro_steps or args.episode_steps % args.macro_steps or args.eval_steps % args.macro_steps:
        raise SystemExit("step counts must be divisible by macro steps")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume and args.warm_start_learner:
        raise SystemExit("use either exact resume or warm-start, not both")
    if args.restore_audit_only and not args.resume:
        raise SystemExit("--restore-audit-only requires --resume")
    if args.first_checkpoint and (
        not args.checkpoint_every or args.first_checkpoint >= args.checkpoint_every
    ):
        raise SystemExit("--first-checkpoint must be smaller than --checkpoint-every")
    graph = MaleCNSGraph.load(args.graph, mmap=True)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    config = PredictivePPOConfig(
        feature_dim=len(ports.readout_names), macro_steps=args.macro_steps, seed=args.seed
    )
    if args.brain_backend == "microbatch":
        brain = FixedCohortBrain(
            graph, ports, args.worlds * 3, device=args.device,
            microbatch_size=args.microbatch_size,
        )
    else:
        brain = RemoteBrain(
            graph, capacity=args.worlds * 3, device=args.device,
            **ports.remote_brain_kwargs(),
        )
    source_paths = [
        ROOT / "chreatures" / name for name in (
            "learning.py", "fast_circuit.py", "remote_brain.py", "malecns.py", "neural_ports.py",
            "physics.py", "articulated.py", "sensorium.py",
        )
    ] + [Path(__file__).resolve(), HABITAT, ROOT / "data/bodies/hexapod.json",
         ROOT / "data/ports/retinal-v1.json"]
    run_record = {
        "format": "chreatures-affordance-run-v1", "started_unix": time.time(),
        "pid": os.getpid(), "argv": [sys.executable, *sys.argv],
        "command": shlex.join([sys.executable, *sys.argv]),
        "graph_sha256": graph.hash, "port_spec_sha256": ports.spec_hash,
        "port_bundle_sha256": sha256(args.port_bundle),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "device": brain.metadata()["device"],
        "warm_start": (
            {"path": str(args.warm_start_learner.resolve()),
             "sha256": sha256(args.warm_start_learner.resolve()),
             "semantics": "shared model, optimizer and normalization only; new private cohort"}
            if args.warm_start_learner else None
        ),
        "resume": (
            {"path": str(args.resume.resolve()), "sha256": sha256(args.resume.resolve()),
             "semantics": "exact neural, physical, learner, optimizer and private-state restore"}
            if args.resume else None
        ),
    }
    (args.output / "run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n")
    initial_genome = args.output / "initial-genome.npz"
    rollout = MacroRollout()
    if args.resume:
        cohort, trainer, step, episode_step, raw, physiology = restore_checkpoint(
            args.resume.resolve(), brain, ports, args.workers
        )
        if cohort.world_count != args.worlds or trainer.config != config:
            raise SystemExit("resume world or learner configuration differs")
        if not initial_genome.exists():
            raise SystemExit("resume run is missing its fixed initial genome")
        normalized = trainer.normalize(raw, update=False)
        neural = []
        if args.restore_audit_only:
            receipt = {
                "format": "chreatures-affordance-restore-audit-v1",
                "checkpoint": str(args.resume.resolve()), "step": step,
                "episode": cohort.episode, "episode_step": episode_step,
                "worlds": cohort.world_count, "residents": len(cohort.resident_ids),
                "learner_updates": trainer.update_count,
                "neural_times_min": float(brain.circuit.times.min()) if hasattr(brain, "circuit") else None,
                "neural_times_max": float(brain.circuit.times.max()) if hasattr(brain, "circuit") else None,
                "brain": brain.metadata(),
            }
            (args.output / "restore-audit.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            cohort.close()
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
    else:
        cohort = AffordanceCohort(brain, ports, args.worlds, args.workers, args.seed)
        trainer = PredictivePPOTrainer(cohort.resident_ids, config, device=args.device)
        trainer.export_genome(initial_genome)
        if args.warm_start_learner:
            inherited, _ = PredictivePPOTrainer.restore(
                args.warm_start_learner.resolve(), device=args.device
            )
            if inherited.config != config or inherited.resident_ids != cohort.resident_ids:
                raise SystemExit("warm-start learner configuration or cohort identities differ")
            trainer = inherited
            trainer.reset_private_state()
        raw, physiology, neural = cohort.observe(0.05)
        normalized = trainer.normalize(raw, update=True)
        step = 0
        episode_step = 0
    started = time.perf_counter()
    stop = False
    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    updates, macro_rows, checkpoints = [], [], []
    algorithm_seconds = 0.0
    ppo_seconds = 0.0
    regular_checkpoint = (
        (step // args.checkpoint_every + 1) * args.checkpoint_every
        if args.checkpoint_every else args.steps + 1
    )
    next_checkpoint = (
        args.first_checkpoint
        if args.first_checkpoint and step < args.first_checkpoint
        else regular_checkpoint
    )
    try:
        while step < args.steps and not stop:
            algorithm_started = time.perf_counter()
            previous = trainer.act(normalized, physiology)
            algorithm_seconds += time.perf_counter() - algorithm_started
            accumulated = np.zeros(len(cohort.resident_ids), dtype=np.float32)
            totals = {name: 0.0 for name in ("nutrition", "nutrition_events", "contacts", "distance")}
            efforts, energies = [], []
            for _ in range(args.macro_steps):
                reward, metrics = cohort.advance(previous["action"], 0.05)
                accumulated += reward
                raw, physiology, neural = cohort.observe(0.05)
                step += 1
                episode_step += 1
                for name in totals:
                    totals[name] += metrics[name]
                efforts.append(metrics["effort"])
                energies.append(metrics["energy"])
            algorithm_started = time.perf_counter()
            normalized_next = trainer.normalize(raw, update=True)
            done = np.full(len(cohort.resident_ids), episode_step >= args.episode_steps)
            learning = trainer.finish_transition(
                previous, normalized_next, accumulated, done, args.macro_steps * 0.05
            )
            algorithm_seconds += time.perf_counter() - algorithm_started
            rollout.append(
                features=previous["features"], physiology=previous["physiology"],
                context=previous["context"], latent=previous["latent"],
                action=previous["action"], log_prob=previous["log_prob"],
                value=previous["value"], reward=learning["reward"], done=done,
                prediction_target=learning["prediction_target"],
            )
            row = {
                "step": step, "episode": cohort.episode, "reward": float(learning["reward"].mean()),
                "prediction_error": float(learning["prediction_error"].mean()),
                "learning_progress": float(learning["learning_progress"].mean()),
                **totals, "effort": float(np.mean(efforts)), "energy": float(energies[-1]),
                "activity": float(np.mean([item["activity"] for item in neural])),
            }
            macro_rows.append(row)
            with (args.output / "macros.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

            if bool(done[0]):
                cohort.reset(cohort.episode + 1)
                trainer.reset_private_state()
                raw, physiology, neural = cohort.observe(0.05)
                normalized_next = trainer.normalize(raw, update=True)
                episode_step = 0
            normalized = normalized_next
            if len(rollout) >= args.rollout_decisions or step >= args.steps:
                bootstrap = trainer.bootstrap_value(normalized, physiology)
                if bool(done[0]):
                    bootstrap.fill(0)
                ppo_started = time.perf_counter()
                update = trainer.update(rollout, bootstrap, args.macro_steps * 0.05)
                ppo_seconds += time.perf_counter() - ppo_started
                update.update({"step": step, "elapsed_seconds": time.perf_counter() - started})
                update["timing_cumulative_seconds"] = {
                    **cohort.timings,
                    "algorithm": algorithm_seconds,
                    "ppo": ppo_seconds,
                }
                updates.append(update)
                with (args.output / "updates.jsonl").open("a") as handle:
                    handle.write(json.dumps(update, sort_keys=True) + "\n")
                print(
                    f"step={step}/{args.steps} update={trainer.update_count} "
                    f"reward={update['reward_mean']:.6g} policy={update['policy_loss']:.5g} "
                    f"predict={update['prediction_loss']:.5g}", flush=True,
                )
                if args.checkpoint_every and step >= next_checkpoint:
                    checkpoints.append(save_checkpoint(
                        args.output, cohort, trainer, step, episode_step,
                        raw, physiology,
                    ))
                    next_checkpoint = (
                        regular_checkpoint
                        if next_checkpoint == args.first_checkpoint
                        else next_checkpoint + args.checkpoint_every
                    )
    except BaseException:
        cohort.close()
        raise

    if stop:
        checkpoints.append(save_checkpoint(
            args.output, cohort, trainer, step, episode_step, raw, physiology
        ))
        cohort.close()
        return 130
    learned_genome = args.output / "learned-genome.npz"
    learned_receipt = trainer.export_genome(learned_genome)
    if not checkpoints or checkpoints[-1]["step"] != step:
        checkpoints.append(save_checkpoint(
            args.output, cohort, trainer, step, episode_step, raw, physiology
        ))

    training_elapsed = time.perf_counter() - started
    training_timings = cohort.timings.copy()
    training_timings.update({"algorithm": algorithm_seconds, "ppo": ppo_seconds})
    cohort.close()
    evaluations = {
        "fixed_initial": evaluate(
            brain, ports, initial_genome, trainer.moments, worlds=args.eval_worlds,
            steps=args.eval_steps, macro_steps=args.macro_steps, workers=args.workers,
            seed=args.seed + 900_000, silence_features=False,
        ),
        "learned": evaluate(
            brain, ports, learned_genome, trainer.moments, worlds=args.eval_worlds,
            steps=args.eval_steps, macro_steps=args.macro_steps, workers=args.workers,
            seed=args.seed + 900_000, silence_features=False,
        ),
        "learned_neural_silenced": evaluate(
            brain, ports, learned_genome, trainer.moments, worlds=args.eval_worlds,
            steps=args.eval_steps, macro_steps=args.macro_steps, workers=args.workers,
            seed=args.seed + 900_000, silence_features=True,
        ),
    }
    summary = {
        "format": "chreatures-affordance-learning-v1",
        "completed": True, "steps": step, "resident_steps": step * args.worlds * 3,
        "elapsed_training_seconds": training_elapsed,
        "training_timing_seconds": training_timings,
        "config": vars(args) | {"graph": str(args.graph), "port_bundle": str(args.port_bundle), "output": str(args.output), "resume": str(args.resume) if args.resume else None},
        "learner": asdict(config), "learner_update_count": trainer.update_count,
        "updates_this_process": updates, "evaluations": evaluations,
        "checkpoints": checkpoints, "initial_genome_sha256": hashlib.sha256(initial_genome.read_bytes()).hexdigest(),
        "learned_genome": learned_receipt, "brain": brain.metadata(),
        "command": shlex.join([sys.executable, *sys.argv]), "pid": os.getpid(),
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "environment": {name: os.environ[name] for name in ("HSA_OVERRIDE_GFX_VERSION", "PYTORCH_KERNEL_CACHE_PATH") if name in os.environ},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"evaluations": evaluations, "steps": step, "resident_steps": summary["resident_steps"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
