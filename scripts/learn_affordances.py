#!/usr/bin/env python3
"""Train and evaluate predictive PPO through rich MaleCNS ports and 3-D bodies."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import shlex
import signal
import sys
import sysconfig
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
    from chreatures.homeostasis import FiniteEnergyConfig, FiniteEnergyObjective
    from chreatures.training_environment import EmbodiedTrainingProfile
    from chreatures.circuit_blueprint import DerivedCircuitGraph, GRAPH_FORMAT
    from chreatures.malecns import MaleCNSGraph
    from chreatures.neural_ports import NeuralPortBundle
    from chreatures.remote_brain import RemoteBrain
    from chreatures.fast_circuit import MicrobatchedResidentCircuit, TritonFusedCircuit
    from chreatures.tiled_circuit import MaleCNSEdgeTiledCircuit


HABITAT = ROOT / "data/habitats/hollow-garden.json"

RESIDENTS_PER_WORLD = 3
ACTION_FIELDS = (
    "thrust", "yaw", "gaze_pitch", "grip",
    "signal_low", "signal_mid", "signal_high", "posture", "eat",
)
BODY_SCALARS = (
    "x", "y", "z", "heading", "radius", "energy", "gut", "fatigue",
    "speed", "angular_velocity", "age", "gaze_pitch",
)
BODY_VECTORS = (("quaternion", 4), ("linear_velocity", 3), ("angular_velocity3d", 3))
OUTCOME_FIELDS = (
    "nutrition", "contact", "distance", "effort", "mechanical_work",
    "ingested_mass", "mouth_material_contacts", "homeostatic_reward",
)
HOMEOSTASIS_FIELDS = (
    "potential_delta_energy", "effort_cost_energy", "nutrition_observed",
    "hunger_gate", "reward",
    "before_reserve_energy", "before_reserve_shortfall_energy",
    "before_fatigue_cost_energy", "before_gut_overload_cost_energy",
    "before_potential_energy", "after_reserve_energy",
    "after_reserve_shortfall_energy", "after_fatigue_cost_energy",
    "after_gut_overload_cost_energy", "after_potential_energy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def package_version(name: str) -> str | None:
    """Return an installed distribution version without making it mandatory."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Durably publish one JSON artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
    }


def stable_checkpoint_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Content identity for a checkpoint, independent of host paths."""
    return {
        "step": int(receipt["step"]),
        "cohort_sha256": str(receipt["cohort_sha256"]),
        "learner_sha256": str(receipt["learner"]["sha256"]),
        "neural_sha256": str(receipt["neural"]["sha256"]),
        "rollout_sha256": str(receipt["rollout"]["sha256"]),
        "rollout_length": int(receipt["rollout"]["length"]),
    }


def stable_native_identity(receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep native implementation identity while excluding machine-local paths."""
    if receipt is None:
        return None
    return {
        key: copy.deepcopy(receipt[key])
        for key in ("bytes", "sha256", "python_soabi", "cache_tag")
        if key in receipt
    }


def stable_brain_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Remove allocation and resident state from neural semantic metadata."""
    value = copy.deepcopy(dict(metadata))
    value.pop("residents", None)
    device = value.pop("device", {})
    stable_device = {
        key: copy.deepcopy(device[key])
        for key in ("type", "name", "gcn_arch_name")
        if key in device
    }
    return {"metadata": value, "device": stable_device}


def existing_checkpoint_receipt(path: Path) -> dict[str, Any]:
    """Verify and describe an existing cohort checkpoint without rewriting it."""
    path = path.resolve()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        state = json.load(handle)
    parent = path.parent
    learner = copy.deepcopy(state["learner"])
    recorded_learner_path = Path(learner["path"])
    local_learner_path = parent / recorded_learner_path.name
    learner_path = (
        local_learner_path if local_learner_path.is_file() else recorded_learner_path
    )
    neural = copy.deepcopy(state["neural"])
    neural_path = parent / f"{neural['name']}.npz"
    rollout = copy.deepcopy(state["rollout"])
    rollout_path = parent / Path(rollout["path"]).name
    for artifact, artifact_path, label in (
        (learner, learner_path, "learner"),
        (neural, neural_path, "neural"),
        (rollout, rollout_path, "rollout"),
    ):
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"existing checkpoint {label} checksum differs")
        if int(artifact["bytes"]) != artifact_path.stat().st_size:
            raise ValueError(f"existing checkpoint {label} size differs")
    return {
        "step": int(state["step"]), "cohort": path.name,
        "cohort_bytes": path.stat().st_size, "cohort_sha256": sha256(path),
        "neural": neural, "learner": learner, "rollout": rollout,
    }


def completed_training_receipt(output: Path, step: int) -> dict[str, Any]:
    """Recover cumulative training timing for an evaluation-only final resume."""
    path = output / "updates.jsonl"
    if not path.is_file():
        raise ValueError("final-step resume is missing updates.jsonl")
    last: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if last is None or int(last.get("step", -1)) != step:
        raise ValueError("final-step resume has no completed update at its checkpoint step")
    timings = last.get("timing_cumulative_seconds")
    if not isinstance(timings, dict):
        raise ValueError("final-step update is missing cumulative timing")
    return {
        "elapsed_seconds": float(last["elapsed_seconds"]),
        "timing_cumulative_seconds": {
            str(name): float(seconds) for name, seconds in timings.items()
        },
    }


def existing_genome_receipt(path: Path, trainer: Any) -> dict[str, Any]:
    """Verify a previously exported final genome against restored model state."""
    path = path.resolve()
    with np.load(path, allow_pickle=False) as value:
        metadata = json.loads(str(value["metadata"]))
        for name, tensor in trainer.model.state_dict().items():
            if name not in value.files or not np.array_equal(
                np.asarray(value[name]), tensor.detach().cpu().numpy()
            ):
                raise ValueError(f"existing final genome differs: {name}")
    if (
        metadata.get("config") != asdict(trainer.config)
        or int(metadata.get("updates", -1)) != trainer.update_count
        or int(metadata.get("decisions", -1)) != trainer.decision_count
    ):
        raise ValueError("existing final genome metadata differs")
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256(path), "metadata": metadata,
    }


def native_extension_receipt() -> dict[str, Any]:
    """Identify the exact native binary and Python ABI used by a run."""
    spec = importlib.util.find_spec("_world_kernels")
    if spec is None or spec.origin is None:
        raise RuntimeError("native world-kernel extension is unavailable")
    path = Path(spec.origin).resolve()
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "cache_tag": sys.implementation.cache_tag,
    }


def load_graph(path: Path):
    """Load measured or explicitly derived anatomy with the proper verifier."""
    manifest = json.loads((path.resolve() / "manifest.json").read_text())
    if manifest.get("format") == GRAPH_FORMAT:
        return DerivedCircuitGraph.load(path.resolve(), mmap=True, verify=True)
    return MaleCNSGraph.load(path.resolve(), mmap=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument(
        "--port-graph", type=Path,
        help="canonical graph used to validate port maps when training a neuron-aligned control",
    )
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
    parser.add_argument(
        "--std-profile", choices=("global-v1", "state-conditioned-v2"),
        default="global-v1",
    )
    parser.add_argument(
        "--context-profile", choices=("reservoir-v1", "gated-v1"),
        default="reservoir-v1",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument(
        "--reward-objective", choices=("legacy", "finite-energy-v1"), default="legacy"
    )
    parser.add_argument(
        "--training-profile",
        choices=(
            "legacy", "current-life-v1", "current-life-v2",
            "chemical-nursery-v3", "chemical-encounters-v4",
        ),
        default="legacy",
    )
    parser.add_argument("--chemical-habitat", type=Path)
    parser.add_argument("--chemical-biosphere", type=Path)
    parser.add_argument("--chemical-conditions", type=Path)
    parser.add_argument("--curriculum-start-stage", type=int, default=0)
    parser.add_argument(
        "--physical-backend", choices=("reference", "fast"), default="fast",
        help="implementation-equivalent physical engine used by current-life-v1 workers",
    )
    parser.add_argument(
        "--allow-physical-backend-transition", action="store_true",
        help="permit an exact checkpoint state to move between validated physical engines",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--brain-backend", choices=("tiled", "triton", "microbatch", "reference"),
        default="tiled",
    )
    parser.add_argument("--microbatch-size", type=int, default=3)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--restore-audit-only", action="store_true",
        help="load an exact full checkpoint, write restore-audit.json, and exit",
    )
    parser.add_argument(
        "--resume-drops-pending-rollout", action="store_true",
        help="record that a legacy checkpoint omitted its pending PPO rollout",
    )
    parser.add_argument(
        "--warm-start-learner", type=Path,
        help="reuse shared model/optimizer/moments, resetting all resident/world/neural state",
    )
    parser.add_argument(
        "--comparison-genome", type=Path,
        help="explicit immutable policy used for evaluation comparisons",
    )
    return parser.parse_args()


def _safe(value: float, high: float, margin: float = 0.35) -> float:
    return float(np.clip(value, margin, high - margin))


def affordance_spec(
    seed: int, episode: int, *, held_out: bool = False,
    depletion_recovery: bool = False,
) -> dict[str, Any]:
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
        if depletion_recovery:
            # A benign recovery problem: reserves begin below the comfort
            # target, but every body is mobile and has nearby adequate food.
            body["energy"] = float(rng.uniform(0.68, 0.76))
            body["gut"] = float(rng.uniform(0.04, 0.10))
        else:
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


def _shared_array_layout(worlds: int, channels: int) -> tuple[dict[str, dict[str, Any]], int]:
    """Lay out one cache-line-aligned fixed cohort block."""
    definitions = (
        ("observations", np.dtype("<f4"), (worlds, RESIDENTS_PER_WORLD, channels)),
        ("bodies", np.dtype("<f8"), (
            worlds, RESIDENTS_PER_WORLD,
            len(BODY_SCALARS) + sum(size for _name, size in BODY_VECTORS),
        )),
        ("actions", np.dtype("<f4"), (
            worlds, RESIDENTS_PER_WORLD, len(ACTION_FIELDS),
        )),
        ("outcomes", np.dtype("<f8"), (
            worlds, RESIDENTS_PER_WORLD, len(OUTCOME_FIELDS),
        )),
        ("homeostasis", np.dtype("<f8"), (
            worlds, RESIDENTS_PER_WORLD, len(HOMEOSTASIS_FIELDS),
        )),
        ("intervals", np.dtype("<f8"), (worlds,)),
        ("completed", np.dtype("<i8"), (worlds,)),
        ("worker_seconds", np.dtype("<f8"), (worlds, 2)),
    )
    layout: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, dtype, shape in definitions:
        offset = (offset + 63) // 64 * 64
        layout[name] = {"dtype": dtype.str, "shape": shape, "offset": offset}
        offset += int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    return layout, (offset + 63) // 64 * 64


def _shared_array_views(
    memory: shared_memory.SharedMemory, layout: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        name: np.ndarray(
            tuple(specification["shape"]), dtype=np.dtype(specification["dtype"]),
            buffer=memory.buf, offset=int(specification["offset"]),
        )
        for name, specification in layout.items()
    }


def _body_values(bodies: list[Any]) -> np.ndarray:
    if len(bodies) != RESIDENTS_PER_WORLD:
        raise ValueError("shared world transport requires exactly three residents")
    rows = np.empty(
        (RESIDENTS_PER_WORLD, len(BODY_SCALARS) + sum(
            size for _name, size in BODY_VECTORS
        )), dtype=np.float64,
    )
    for row, body in enumerate(bodies):
        body = body.to_dict() if hasattr(body, "to_dict") else body
        values = [float(body[name]) for name in BODY_SCALARS]
        for name, size in BODY_VECTORS:
            vector = np.asarray(body[name], dtype=np.float64)
            if vector.shape != (size,):
                raise ValueError(f"body field {name} has the wrong fixed shape")
            values.extend(vector.astype(float).tolist())
        rows[row] = values
    if not np.isfinite(rows).all():
        raise ValueError("body shared state contains nonfinite values")
    return rows


def _write_outcomes(
    target: np.ndarray, homeostasis_target: np.ndarray,
    outcomes: Mapping[str, Mapping[str, Any]], bodies: list[Any],
) -> None:
    target.fill(0)
    homeostasis_target.fill(0)
    for row, body in enumerate(bodies):
        body_id = body.id if hasattr(body, "id") else str(body["id"])
        outcome = outcomes[body_id]
        homeostasis = outcome.get("homeostasis", {})
        unknown = set(outcome) - set(OUTCOME_FIELDS) - {"homeostasis"}
        unknown_homeostasis = set(homeostasis) - set(HOMEOSTASIS_FIELDS)
        if unknown or unknown_homeostasis:
            raise ValueError(
                "shared outcome schema differs: "
                f"fields={sorted(unknown)} homeostasis={sorted(unknown_homeostasis)}"
            )
        target[row] = [float(outcome.get(name, 0.0)) for name in OUTCOME_FIELDS]
        homeostasis_target[row] = [
            float(homeostasis.get(name, 0.0)) for name in HOMEOSTASIS_FIELDS
        ]
    if not np.isfinite(target).all() or not np.isfinite(homeostasis_target).all():
        raise ValueError("outcome shared state contains nonfinite values")


class SharedWorldCohort:
    """Parent-owned fixed buffers with disjoint three-resident world chunks."""

    def __init__(self, worlds: int, channels: int) -> None:
        self.layout, size = _shared_array_layout(worlds, channels)
        self.memory = shared_memory.SharedMemory(create=True, size=size)
        self.arrays = _shared_array_views(self.memory, self.layout)
        for array in self.arrays.values():
            array.fill(0)

    def descriptor(self) -> dict[str, Any]:
        return {"name": self.memory.name, "layout": self.layout}

    def close(self) -> None:
        self.arrays.clear()
        self.memory.close()
        self.memory.unlink()


def _world_worker(
    connection, port_spec: dict[str, Any], profile_value: dict[str, Any] | None,
    physical_backend: str, shared_descriptor: Mapping[str, Any], world_index: int,
) -> None:
    """Own one MuJoCo instance so native and Python work spans CPU cores."""
    from chreatures.neural_ports import encode_physical_senses
    if profile_value is None:
        from chreatures.physical_batch import FastArticulatedSensoriumWorld as ArticulatedSensoriumWorld
        profile = None
    else:
        from chreatures.training_environment import (
            EmbodiedTrainingProfile, EmbodiedTrainingWorld, embodied_training_spec,
        )
        profile = EmbodiedTrainingProfile.from_value(profile_value)
    memory = shared_memory.SharedMemory(name=str(shared_descriptor["name"]))
    shared = _shared_array_views(memory, shared_descriptor["layout"])
    world = None
    try:
        while True:
            try:
                operation, payload = connection.recv()
            except EOFError:
                if world is not None and hasattr(world, "close"):
                    world.close()
                return
            if operation == "close":
                if world is not None and hasattr(world, "close"):
                    world.close()
                connection.send((True, None))
                return
            if operation == "observe_shared":
                started = time.perf_counter()
                sequence = int(payload)
                if world is None:
                    raise RuntimeError("world must be reset before observation")
                vectors = [
                    encode_physical_senses(world.sense(body.id), port_spec)[1]
                    for body in world.bodies
                ]
                observations = np.asarray(vectors, dtype=np.float32)
                expected = shared["observations"][world_index].shape
                if observations.shape != expected or not np.isfinite(observations).all():
                    raise ValueError("encoded shared observations have the wrong shape")
                shared["observations"][world_index] = observations
                shared["bodies"][world_index] = _body_values(world.bodies)
                shared["worker_seconds"][world_index, 0] += time.perf_counter() - started
                shared["completed"][world_index] = sequence
                connection.send((True, sequence))
                continue
            if operation == "advance_shared":
                started = time.perf_counter()
                sequence = int(payload)
                if world is None:
                    raise RuntimeError("world must be reset before advance")
                action_rows = shared["actions"][world_index]
                if not np.isfinite(action_rows).all():
                    raise ValueError("shared actions contain nonfinite values")
                actions = {
                    body.id: {
                        name: float(action_rows[row, column])
                        for column, name in enumerate(ACTION_FIELDS)
                    }
                    for row, body in enumerate(world.bodies)
                }
                outcome = world.advance(actions, float(shared["intervals"][world_index]))
                shared["bodies"][world_index] = _body_values(world.bodies)
                _write_outcomes(
                    shared["outcomes"][world_index],
                    shared["homeostasis"][world_index], outcome, world.bodies,
                )
                shared["worker_seconds"][world_index, 1] += time.perf_counter() - started
                shared["completed"][world_index] = sequence
                connection.send((True, sequence))
                continue
            if operation == "reset":
                if world is not None and hasattr(world, "close"):
                    world.close()
                if profile is None:
                    world = ArticulatedSensoriumWorld(seed=payload["seed"], spec=payload["spec"])
                else:
                    spec = embodied_training_spec(
                        payload["seed"], held_out=payload.get("held_out", False),
                        stage=payload.get("stage", 0),
                        profile=profile,
                    )
                    world = EmbodiedTrainingWorld(
                        payload["seed"], spec, profile,
                        physical_backend=physical_backend,
                    )
                result = [body.to_dict() for body in world.bodies]
            elif operation == "restore":
                if world is not None and hasattr(world, "close"):
                    world.close()
                world = (
                    ArticulatedSensoriumWorld.restore(payload) if profile is None else
                    EmbodiedTrainingWorld.restore(
                        payload, expected_profile=profile.sha256,
                        physical_backend=physical_backend,
                    )
                )
                result = [body.to_dict() for body in world.bodies]
            elif operation == "observe":
                raise RuntimeError("numeric observation must use shared cohort transport")
            elif operation == "advance":
                raise RuntimeError("numeric advance must use shared cohort transport")
            elif operation == "telemetry":
                result = copy.deepcopy(world.last_telemetry) if profile is not None else {}
            elif operation == "snapshot":
                result = world.snapshot()
            elif operation == "terminal":
                if profile is not None:
                    result = world.terminal_outcomes()
                else:
                    result = {
                        "format": "chreatures-legacy-terminal-outcomes-v1",
                        "time": float(world.time),
                        "residents": {
                            body.id: {
                                "energy": float(body.energy),
                                "gut": float(body.gut),
                                "fatigue": float(body.fatigue),
                                "speed": float(body.speed),
                            }
                            for body in world.bodies
                        },
                    }
            else:
                raise ValueError(f"unknown world worker operation {operation}")
            connection.send((True, result))
    except BaseException as exc:
        try:
            connection.send((False, {"error": repr(exc), "traceback": traceback.format_exc()}))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        memory.close()
        connection.close()


class ProcessWorldPool:
    """Fixed-shape numeric transport and rare structured world commands.

    Each worker owns one disjoint world row.  Numeric calls complete through a
    cohort-wide sequence barrier before the parent reads any shared buffer.
    """

    def __init__(
        self, count: int, port_spec: dict[str, Any],
        profile_value: dict[str, Any] | None = None,
        physical_backend: str = "fast",
    ) -> None:
        context = mp.get_context("spawn")
        channels = int(port_spec["physical_inputs"]["count"])
        ordered_names = port_spec["physical_inputs"]["ordered_names"]
        if channels != len(ordered_names) or count <= 0:
            raise ValueError("invalid fixed world cohort dimensions")
        self.shared = SharedWorldCohort(count, channels)
        self._sequence = 0
        self._closed = False
        self._hot_calls = {"observe": 0, "advance": 0}
        self._hot_wall_seconds = {"observe": 0.0, "advance": 0.0}
        self._body_templates: list[list[dict[str, Any]]] | None = None
        self._has_homeostasis = profile_value is not None
        self.connections = []
        self.processes = []
        try:
            descriptor = self.shared.descriptor()
            for world_index in range(count):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_world_worker,
                    args=(
                        child, port_spec, profile_value, physical_backend,
                        descriptor, world_index,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
        except BaseException:
            self._abort()
            raise

    def _abort(self) -> None:
        """Close this pool after any broken barrier; its buffers are unsafe."""
        if self._closed:
            return
        self._closed = True
        for connection in self.connections:
            try:
                connection.close()
            except OSError:
                pass
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        self.shared.close()

    def _barrier(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError("world worker pool is closed")
        self._sequence += 1
        sequence = self._sequence
        try:
            for connection in self.connections:
                connection.send((operation, sequence))
            for connection in self.connections:
                if not connection.poll(300):
                    raise TimeoutError(f"world worker timed out during {operation}")
                ok, value = connection.recv()
                if not ok:
                    raise RuntimeError(
                        f"world worker failed: {value['error']}\n{value['traceback']}"
                    )
                if int(value) != sequence:
                    raise RuntimeError("world worker acknowledged the wrong sequence")
            if not np.all(self.shared.arrays["completed"] == sequence):
                raise RuntimeError("world cohort barrier completed with stale rows")
        except BaseException:
            self._abort()
            raise

    def _bodies(self) -> list[list[dict[str, Any]]]:
        if self._body_templates is None:
            raise RuntimeError("worlds must be reset before reading shared body state")
        result = copy.deepcopy(self._body_templates)
        values = self.shared.arrays["bodies"]
        scalar_count = len(BODY_SCALARS)
        for world_index, bodies in enumerate(result):
            for resident_index, body in enumerate(bodies):
                row = values[world_index, resident_index]
                for column, name in enumerate(BODY_SCALARS):
                    body[name] = float(row[column])
                column = scalar_count
                for name, size in BODY_VECTORS:
                    body[name] = row[column:column + size].astype(float).tolist()
                    column += size
        return result

    def _outcomes(self, bodies: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        values = self.shared.arrays["outcomes"]
        homeostasis = self.shared.arrays["homeostasis"]
        for world_index, world_bodies in enumerate(bodies):
            by_resident = {}
            for resident_index, body in enumerate(world_bodies):
                outcome = {
                    name: (
                        int(values[world_index, resident_index, column])
                        if name == "mouth_material_contacts" else
                        float(values[world_index, resident_index, column])
                    )
                    for column, name in enumerate(OUTCOME_FIELDS)
                }
                if self._has_homeostasis:
                    outcome["homeostasis"] = {
                        name: float(homeostasis[world_index, resident_index, column])
                        for column, name in enumerate(HOMEOSTASIS_FIELDS)
                    }
                by_resident[str(body["id"])] = outcome
            result.append(by_resident)
        return result

    def _structured_call(self, operation: str, payloads: list[Any]) -> list[Any]:
        if self._closed:
            raise RuntimeError("world worker pool is closed")
        try:
            for connection, payload in zip(self.connections, payloads, strict=True):
                connection.send((operation, payload))
            results = []
            for connection in self.connections:
                if not connection.poll(300):
                    raise TimeoutError(f"world worker timed out during {operation}")
                ok, value = connection.recv()
                if not ok:
                    raise RuntimeError(
                        f"world worker failed: {value['error']}\n{value['traceback']}"
                    )
                results.append(value)
            return results
        except BaseException:
            self._abort()
            raise

    def timing_snapshot(self) -> dict[str, Any]:
        """Return bounded transport costs without synchronizing a GPU."""
        worker = self.shared.arrays["worker_seconds"].copy()
        return {
            "format": "chreatures-shared-world-transport-timing-v1",
            "buffer_bytes": int(self.shared.memory.size),
            "worlds": len(self.connections),
            "residents_per_world": RESIDENTS_PER_WORLD,
            "observation_channels": int(
                self.shared.arrays["observations"].shape[-1]
            ),
            "numeric_layout": {
                name: {
                    "shape": list(array.shape), "dtype": array.dtype.str,
                }
                for name, array in self.shared.arrays.items()
                if name not in {"completed", "worker_seconds"}
            },
            "hot_calls": copy.deepcopy(self._hot_calls),
            "parent_wall_seconds": copy.deepcopy(self._hot_wall_seconds),
            "worker_cpu_seconds": {
                "observe_sum": float(worker[:, 0].sum()),
                "observe_max_world": float(worker[:, 0].max(initial=0.0)),
                "advance_sum": float(worker[:, 1].sum()),
                "advance_max_world": float(worker[:, 1].max(initial=0.0)),
            },
        }

    def call_all(self, operation: str, payloads: list[Any] | None = None) -> list[Any]:
        payloads = payloads if payloads is not None else [None] * len(self.connections)
        if len(payloads) != len(self.connections):
            raise ValueError("world worker payload count differs")
        if operation == "observe":
            if any(payload is not None for payload in payloads):
                raise ValueError("observe does not accept payloads")
            started = time.perf_counter()
            self._barrier("observe_shared")
            observations = self.shared.arrays["observations"].copy()
            bodies = self._bodies()
            self._hot_calls["observe"] += 1
            self._hot_wall_seconds["observe"] += time.perf_counter() - started
            return [(observations[index], bodies[index]) for index in range(len(bodies))]
        if operation == "advance":
            if self._body_templates is None:
                raise RuntimeError("worlds must be reset before advance")
            started = time.perf_counter()
            action_buffer = self.shared.arrays["actions"]
            intervals = self.shared.arrays["intervals"]
            for world_index, (payload, bodies) in enumerate(zip(
                payloads, self._body_templates, strict=True,
            )):
                if not isinstance(payload, Mapping) or set(payload) != {"actions", "dt"}:
                    raise ValueError("advance payload must contain only actions and dt")
                interval = float(payload["dt"])
                if not math.isfinite(interval) or interval <= 0:
                    raise ValueError("advance dt must be finite and positive")
                intervals[world_index] = interval
                actions = payload["actions"]
                if set(actions) != {str(body["id"]) for body in bodies}:
                    raise ValueError("advance action resident IDs differ from world")
                for resident_index, body in enumerate(bodies):
                    action = actions[str(body["id"])]
                    unknown = set(action) - set(ACTION_FIELDS)
                    if unknown:
                        raise ValueError(f"unknown action fields: {sorted(unknown)}")
                    action_buffer[world_index, resident_index] = [
                        float(action.get(name, 0.0)) for name in ACTION_FIELDS
                    ]
            if not np.isfinite(action_buffer).all():
                raise ValueError("shared action cohort contains nonfinite values")
            self._barrier("advance_shared")
            bodies = self._bodies()
            outcomes = self._outcomes(bodies)
            self._hot_calls["advance"] += 1
            self._hot_wall_seconds["advance"] += time.perf_counter() - started
            return [
                (outcomes[index], bodies[index], {}) for index in range(len(bodies))
            ]
        results = self._structured_call(operation, payloads)
        if operation in {"reset", "restore"}:
            self._body_templates = copy.deepcopy(results)
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for connection in self.connections:
            try:
                if connection.poll(5):
                    connection.recv()
            except (BrokenPipeError, EOFError):
                pass
            connection.close()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.shared.close()


class FixedCohortBrain:
    """Resident-ID and checkpoint shell around the measured fixed GPU circuit."""

    def __init__(
        self, graph: Any, ports: Any, batch_size: int, *, device: str,
        backend: str, microbatch_size: int,
    ) -> None:
        self.graph = graph
        self.ports = ports
        self.graph_hash = str(graph.hash)
        self.device = torch.device(device)
        self.capacity = batch_size
        self.resident_ids: list[str] = []
        kwargs = {
            "device": device,
            "input_map": (ports.input_names, ports.input_map),
            "readout_map": (ports.readout_names, ports.readout_map),
        }
        self.circuit = (
            MaleCNSEdgeTiledCircuit(graph, batch_size, **kwargs)
            if backend == "tiled" else
            TritonFusedCircuit(graph, batch_size, **kwargs)
            if backend == "triton" else
            MicrobatchedResidentCircuit(
                graph, batch_size, microbatch_size=microbatch_size, **kwargs
            )
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

    def __init__(
        self, brain: RemoteBrain, ports: NeuralPortBundle, worlds: int,
        workers: int, seed: int, reward_objective: Any | None = None,
        training_profile: Any | None = None, physical_backend: str = "fast",
        curriculum_start_stage: int = 0,
        ingestion_enabled: bool = True,
    ) -> None:
        self.brain = brain
        self.ports = ports
        self.world_count = worlds
        self.seed = seed
        self.reward_objective = reward_objective
        self.training_profile = training_profile
        self.physical_backend = physical_backend
        self.curriculum_start_stage = int(curriculum_start_stage)
        self.ingestion_enabled = bool(ingestion_enabled)
        self.episode = 0
        self.world_pool = ProcessWorldPool(
            worlds, ports.spec,
            training_profile.to_value() if training_profile is not None else None,
            physical_backend,
        )
        self.timings = {name: 0.0 for name in ("world_build", "sense_encode", "brain", "physics")}
        self.body_states: list[list[dict[str, Any]]] = []
        self.last_world_telemetry: list[dict[str, Any]] = []
        self.terminal_outcomes: list[dict[str, Any]] = []
        self.episode_steps_advanced = 0
        self.last_source_channels = np.empty((0, len(ports.input_names)), dtype=np.float32)
        self.resident_ids: list[str] = []
        self.reset(0)

    def reset(self, episode: int, *, held_out: bool = False) -> None:
        started = time.perf_counter()
        if self.episode_steps_advanced:
            if len(self.terminal_outcomes) >= 1024:
                raise RuntimeError("training terminal outcome history exceeds its bound")
            self.terminal_outcomes.append({
                "episode": self.episode,
                "physical_steps": self.episode_steps_advanced,
                "worlds": self.world_pool.call_all("terminal"),
            })
        old_ids = self.brain.resident_ids
        if old_ids:
            self.brain.remove_residents(old_ids)
        self.episode = episode
        payloads = []
        for index in range(self.world_count):
            world_seed = self.seed + episode * 1009 + index
            payload = {"seed": world_seed, "held_out": held_out}
            if self.training_profile is not None:
                payload["stage"] = (
                    0 if int(self.training_profile.component("version")) == 3 else
                    2 if held_out else min(2, self.curriculum_start_stage + episode)
                )
            if self.training_profile is None:
                payload["spec"] = affordance_spec(
                    self.seed + index * 17, episode, held_out=held_out,
                    depletion_recovery=self.reward_objective is not None,
                )
            payloads.append(payload)
        self.body_states = self.world_pool.call_all("reset", payloads)
        prefix = "eval" if held_out else "train"
        self.resident_ids = [
            f"{prefix}-w{world_index:02d}:{body['id']}"
            for world_index, bodies in enumerate(self.body_states)
            for body in bodies
        ]
        self.brain.add_residents(self.resident_ids)
        self.episode_steps_advanced = 0
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
        self.last_source_channels = np.asarray(channel_rows, dtype=np.float32)
        if hasattr(self.brain, "step_channels"):
            features, circuit_physiology, neural = self.brain.step_channels(
                self.last_source_channels, dt
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
        before_physiology = np.asarray([
            [body["energy"], body["gut"], body["fatigue"]]
            for bodies in self.body_states for body in bodies
        ], dtype=np.float32)
        actions_by_world = []
        eat_requests = []
        for world_index, bodies in enumerate(self.body_states):
            actions = {}
            for body_index, body in enumerate(bodies):
                vector = action_values[world_index * 3 + body_index]
                action = dict(zip(ACTIONS, vector.astype(float).tolist(), strict=True))
                for name in ("grip", "signal_low", "signal_mid", "signal_high"):
                    action[name] = max(0.0, action[name])
                action["eat"] = (
                    float(np.clip((1 - body["gut"]) * (1.1 - body["energy"]), 0, 1))
                    if self.ingestion_enabled else 0.0
                )
                eat_requests.append(action["eat"])
                actions[body["id"]] = action
            actions_by_world.append(actions)
        started = time.perf_counter()
        advanced = self.world_pool.call_all("advance", [
            {"actions": actions, "dt": dt} for actions in actions_by_world
        ])
        self.episode_steps_advanced += 1
        outcomes = [value[0] for value in advanced]
        self.body_states = [value[1] for value in advanced]
        self.last_world_telemetry = [value[2] for value in advanced]
        self.timings["physics"] += time.perf_counter() - started
        after_energy = np.asarray(
            [body["energy"] for bodies in self.body_states for body in bodies], dtype=np.float32
        )
        after_physiology = np.asarray([
            [body["energy"], body["gut"], body["fatigue"]]
            for bodies in self.body_states for body in bodies
        ], dtype=np.float32)
        nutrition, ingested, mouth_contacts, contact, distance, effort = [], [], [], [], [], []
        for bodies, result in zip(self.body_states, outcomes, strict=True):
            for body in bodies:
                value = result[body["id"]]
                nutrition.append(value["nutrition"])
                ingested.append(value.get("ingested_mass", 0.0))
                mouth_contacts.append(value.get("mouth_material_contacts", 0))
                contact.append(value["contact"])
                distance.append(value["distance"])
                effort.append(value["effort"])
        nutrition = np.asarray(nutrition, dtype=np.float32)
        ingested = np.asarray(ingested, dtype=np.float32)
        mouth_contacts = np.asarray(mouth_contacts, dtype=np.int64)
        eat_requests = np.asarray(eat_requests, dtype=np.float32)
        contact = np.asarray(contact, dtype=np.float32)
        effort = np.asarray(effort, dtype=np.float32)
        old_drive = (0.85 - before_energy) ** 2
        new_drive = (0.85 - after_energy) ** 2
        if self.reward_objective is None:
            reward = (
                (old_drive - new_drive) * 12
                + nutrition * np.maximum(0, 1 - after_energy) * 3
                - effort * np.float32(0.0002 * dt)
            ).astype(np.float32)
            homeostasis = {}
        else:
            reward, components = self.reward_objective.transition(
                before_physiology, after_physiology,
                nutrition=nutrition, effort=effort, dt=dt,
            )
            homeostasis = {
                key: float(np.mean(value)) for key, value in components.items()
            }
        reserve = after_physiology[:, 0] + np.float32(0.84) * after_physiology[:, 1]
        return reward, {
            "nutrition": float(nutrition.sum()),
            "nutrition_events": float(np.count_nonzero(nutrition > 0)),
            "absorbed": float(nutrition.sum()),
            "ingested_mass": float(ingested.sum()),
            "ingestion_events": float(np.count_nonzero(ingested > 0)),
            "mouth_material_contacts": float(mouth_contacts.sum()),
            "eat_request_steps": float(np.count_nonzero(eat_requests > 0)),
            "eat_request_mean": float(eat_requests.mean()),
            "contact_while_eating": float(np.count_nonzero(
                (contact > 0) & (eat_requests > 0)
            )),
            "contacts": float(np.count_nonzero(contact > 0)),
            "distance": float(np.sum(distance)),
            "effort": float(effort.mean()),
            "energy": float(after_energy.mean()),
            "gut": float(after_physiology[:, 1].mean()),
            "fatigue": float(after_physiology[:, 2].mean()),
            "reserve_energy": float(reserve.mean()),
            "stationary_fraction": float(np.mean([
                abs(float(body["speed"])) < 1e-3
                for bodies in self.body_states for body in bodies
            ])),
            "homeostasis": homeostasis,
        }

    def physiology_summary(self) -> dict[str, float]:
        bodies = [body for values in self.body_states for body in values]
        energy = np.asarray([body["energy"] for body in bodies], dtype=np.float64)
        gut = np.asarray([body["gut"] for body in bodies], dtype=np.float64)
        fatigue = np.asarray([body["fatigue"] for body in bodies], dtype=np.float64)
        speed = np.asarray([abs(body["speed"]) for body in bodies], dtype=np.float64)
        reserve = energy + 0.84 * gut
        return {
            "energy_mean": float(energy.mean()), "energy_min": float(energy.min()),
            "gut_mean": float(gut.mean()), "fatigue_mean": float(fatigue.mean()),
            "fatigue_max": float(fatigue.max()), "reserve_mean": float(reserve.mean()),
            "depleted_fraction": float(np.mean(energy < 0.05)),
            "exhausted_fraction": float(np.mean(fatigue > 0.95)),
            "stationary_fraction": float(np.mean(speed < 1e-3)),
        }

    def close(self) -> None:
        self.world_pool.close()


def save_checkpoint(
    output: Path, cohort: AffordanceCohort, trainer: PredictivePPOTrainer,
    rollout: MacroRollout, step: int, episode_step: int,
    features: np.ndarray, physiology: np.ndarray,
) -> dict[str, Any]:
    directory = output / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    tag = f"step-{step:07d}"
    neural = cohort.brain.snapshot(directory, f"neural-{tag}")
    learner = trainer.snapshot(directory / f"learner-{tag}.pt")
    rollout_path = directory / f"rollout-{tag}.npz"
    rollout_temporary = rollout_path.with_name(rollout_path.name + ".tmp")
    rollout_arrays = rollout.arrays() if len(rollout) else {}
    with rollout_temporary.open("wb") as handle:
        np.savez_compressed(handle, length=np.asarray(len(rollout)), **rollout_arrays)
    os.replace(rollout_temporary, rollout_path)
    rollout_receipt = {
        "path": rollout_path.name, "length": len(rollout),
        "bytes": rollout_path.stat().st_size, "sha256": sha256(rollout_path),
    }
    state = {
        "version": 6, "step": step, "episode": cohort.episode,
        "episode_step": episode_step, "resident_ids": cohort.resident_ids,
        "graph_sha256": cohort.brain.graph_hash, "port_spec_sha256": cohort.ports.spec_hash,
        "neural": neural, "learner": learner, "rollout": rollout_receipt,
        "worlds": cohort.world_pool.call_all("snapshot"),
        "features": features.tolist(), "physiology": physiology.tolist(),
        "cohort": {
            "worlds": cohort.world_count, "seed": cohort.seed,
            "curriculum_start_stage": cohort.curriculum_start_stage,
        },
        "reward_objective": (
            cohort.reward_objective.config.to_value()
            if cohort.reward_objective is not None else None
        ),
        "training_profile": (
            cohort.training_profile.to_value()
            if cohort.training_profile is not None else None
        ),
        "physical_backend": cohort.physical_backend,
        "ingestion_enabled": cohort.ingestion_enabled,
        "terminal_outcomes": copy.deepcopy(cohort.terminal_outcomes),
        "episode_steps_advanced": cohort.episode_steps_advanced,
    }
    path = directory / f"cohort-{tag}.json.gz"
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(state, handle, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, path)
    return {
        "step": step, "cohort": path.name, "cohort_bytes": path.stat().st_size,
        "cohort_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "neural": neural, "learner": learner, "rollout": rollout_receipt,
    }


def restore_checkpoint(
    path: Path, brain: RemoteBrain, ports: NeuralPortBundle, workers: int,
    reward_objective: Any | None, training_profile: Any | None = None,
    physical_backend: str = "fast", allow_physical_backend_transition: bool = False,
    target_std_profile: str = "global-v1",
    target_context_profile: str = "reservoir-v1",
    curriculum_start_stage: int = 0,
) -> tuple[AffordanceCohort, PredictivePPOTrainer, MacroRollout, int, int, np.ndarray, np.ndarray]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        state = json.load(handle)
    if (
        state.get("version") not in (1, 2, 3, 4, 5, 6)
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
    cohort.reward_objective = reward_objective
    cohort.training_profile = training_profile
    cohort.physical_backend = physical_backend
    cohort.curriculum_start_stage = int(curriculum_start_stage)
    cohort.ingestion_enabled = bool(state.get("ingestion_enabled", True))
    saved_curriculum_start = int(state["cohort"].get("curriculum_start_stage", 0))
    if saved_curriculum_start != cohort.curriculum_start_stage:
        raise ValueError("checkpoint curriculum start stage differs")
    expected_reward = (
        reward_objective.config.to_value() if reward_objective is not None else None
    )
    if state.get("version") < 3 and reward_objective is not None:
        raise ValueError("legacy checkpoint cannot exact-resume with a new reward objective")
    if state.get("version") >= 3 and state.get("reward_objective") != expected_reward:
        raise ValueError("checkpoint reward objective differs")
    expected_profile = training_profile.to_value() if training_profile is not None else None
    if state.get("version") < 4 and training_profile is not None:
        raise ValueError("legacy checkpoint cannot exact-resume with an embodied profile")
    if state.get("version") >= 4 and state.get("training_profile") != expected_profile:
        raise ValueError("checkpoint embodied training profile differs")
    saved_physical_backend = state.get(
        "physical_backend", "reference" if expected_profile is not None else "fast"
    )
    if saved_physical_backend != physical_backend and not allow_physical_backend_transition:
        raise ValueError(
            "checkpoint physical backend differs; exact validated transitions require "
            "--allow-physical-backend-transition"
        )
    cohort.episode = int(state["episode"])
    cohort.world_pool = ProcessWorldPool(
        cohort.world_count, ports.spec, expected_profile, physical_backend,
    )
    cohort.timings = {name: 0.0 for name in ("world_build", "sense_encode", "brain", "physics")}
    cohort.body_states = cohort.world_pool.call_all("restore", state["worlds"])
    cohort.last_world_telemetry = []
    cohort.terminal_outcomes = copy.deepcopy(state.get("terminal_outcomes", []))
    cohort.episode_steps_advanced = int(
        state.get("episode_steps_advanced", state.get("episode_step", 0))
    )
    cohort.last_source_channels = np.empty(
        (0, len(ports.input_names)), dtype=np.float32
    )
    cohort.resident_ids = [str(value) for value in state["resident_ids"]]
    if cohort.resident_ids != brain.resident_ids:
        raise ValueError("checkpoint neural and physical resident order differs")
    learner_path = path.parent / Path(state["learner"]["path"]).name
    trainer, _ = PredictivePPOTrainer.restore(
        learner_path, device=brain.device,
        expected_sha256=state["learner"]["sha256"],
    )
    if trainer.config.std_profile != target_std_profile:
        raise ValueError("exact resume cannot change learner variance architecture")
    if trainer.config.context_profile != target_context_profile:
        raise ValueError("exact resume cannot change working-context architecture")
    if trainer.resident_ids != cohort.resident_ids:
        raise ValueError("checkpoint learner residents differ")
    rollout = MacroRollout()
    if state.get("version") >= 2:
        receipt = state["rollout"]
        rollout_path = path.parent / receipt["path"]
        if sha256(rollout_path) != receipt["sha256"]:
            raise ValueError("checkpoint rollout checksum differs")
        with np.load(rollout_path, allow_pickle=False) as value:
            length = int(value["length"])
            if length != int(receipt["length"]):
                raise ValueError("checkpoint rollout length differs")
            if length:
                arrays = {name: np.asarray(value[name]) for name in MacroRollout.FIELDS}
                if any(len(array) != length for array in arrays.values()):
                    raise ValueError("checkpoint rollout arrays differ")
                for index in range(length):
                    rollout.append(**{name: array[index] for name, array in arrays.items()})
    features = np.asarray(state["features"], dtype=np.float32)
    physiology = np.asarray(state["physiology"], dtype=np.float32)
    return (
        cohort, trainer, rollout, int(state["step"]), int(state["episode_step"]),
        features, physiology,
    )


def evaluate(
    brain: RemoteBrain, ports: NeuralPortBundle, genome: Path,
    moments: RunningMoments, *, worlds: int, steps: int, macro_steps: int,
    workers: int, seed: int, silence_features: bool,
    reward_objective: Any | None = None, training_profile: Any | None = None,
    telemetry_every: int = 0, physical_backend: str = "fast",
    curriculum_start_stage: int = 0,
    ingestion_enabled: bool = True,
    std_profile: str = "global-v1",
) -> dict[str, Any]:
    cohort = AffordanceCohort(
        brain, ports, worlds, workers, seed, reward_objective=reward_objective,
        training_profile=training_profile,
        physical_backend=physical_backend,
        curriculum_start_stage=curriculum_start_stage,
        ingestion_enabled=ingestion_enabled,
    )
    cohort.reset(0, held_out=True)
    config = PredictivePPOConfig(
        feature_dim=len(ports.readout_names), macro_steps=macro_steps,
        seed=seed, std_profile=std_profile,
    )
    trainer = PredictivePPOTrainer(cohort.resident_ids, config, device=brain.device)
    trainer.import_genome(genome)
    trainer.moments = RunningMoments.restore(moments.snapshot())
    raw, physiology, _ = cohort.observe(0.05)
    normalized = trainer.normalize(raw, update=False)
    totals = {name: 0.0 for name in (
        "nutrition", "nutrition_events", "absorbed", "ingested_mass",
        "ingestion_events", "mouth_material_contacts", "eat_request_steps",
        "contact_while_eating", "contacts", "distance",
    )}
    efforts, energies, guts, fatigues, reserves, stationary, rewards = [], [], [], [], [], [], []
    homeostasis: dict[str, list[float]] = {}
    trajectory: list[dict[str, Any]] = []
    physical_step = 0
    terminal_outcomes = None
    final_physiology = None
    transport_timing = None
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
                physical_step += 1
                raw, physiology, _ = cohort.observe(0.05)
                for name in totals:
                    totals[name] += metrics[name]
                efforts.append(metrics["effort"])
                energies.append(metrics["energy"])
                guts.append(metrics["gut"])
                fatigues.append(metrics["fatigue"])
                reserves.append(metrics["reserve_energy"])
                stationary.append(metrics["stationary_fraction"])
                for name, value in metrics["homeostasis"].items():
                    homeostasis.setdefault(name, []).append(value)
                if telemetry_every and physical_step % telemetry_every == 0:
                    cohort.last_world_telemetry = cohort.world_pool.call_all("telemetry")
                    trajectory.append({
                        "step": physical_step, "model_seconds": physical_step * 0.05,
                        "cumulative": totals.copy(), "physiology": cohort.physiology_summary(),
                        "world_components": copy.deepcopy(cohort.last_world_telemetry),
                    })
            normalized = trainer.normalize(raw, update=False)
            finish_features = np.zeros_like(normalized) if silence_features else normalized
            trainer.finish_transition(
                previous, finish_features, accumulated,
                np.zeros(len(cohort.resident_ids), dtype=bool), macro_steps * 0.05,
            )
            rewards.extend(accumulated.tolist())
    finally:
        try:
            if physical_step:
                terminal_outcomes = cohort.world_pool.call_all("terminal")
                final_physiology = cohort.physiology_summary()
            transport_timing = cohort.world_pool.timing_snapshot()
        finally:
            cohort.close()
    return {
        **totals,
        "effort_mean": float(np.mean(efforts)),
        "energy_final": float(energies[-1]),
        "gut_final": float(guts[-1]),
        "fatigue_final": float(fatigues[-1]),
        "reserve_final": float(reserves[-1]),
        "stationary_fraction_mean": float(np.mean(stationary)),
        "final_physiology": final_physiology,
        "reward_total": float(np.sum(rewards)),
        "reward_mean_per_macro_resident": float(np.mean(rewards)),
        "homeostasis_mean_per_step": {
            name: float(np.mean(values)) for name, values in homeostasis.items()
        },
        "terminal_outcomes": terminal_outcomes,
        "ingestion_enabled": ingestion_enabled,
        "trajectory": trajectory,
        "world_transport_timing": transport_timing,
    }


def persisted_evaluation(
    output: Path, name: str, identity: Mapping[str, Any],
    brain: RemoteBrain, ports: NeuralPortBundle, genome: Path,
    moments: RunningMoments, **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run once and atomically retain a completed held-out condition."""
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
        raise ValueError("evaluation condition name is unsafe")
    expected = copy.deepcopy(dict(identity))
    path = output / "evaluations" / f"{name}.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        if (
            receipt.get("format") != "chreatures-affordance-evaluation-v1"
            or receipt.get("version") != 1
            or receipt.get("name") != name
            or receipt.get("identity") != expected
            or not isinstance(receipt.get("result"), dict)
        ):
            raise ValueError(f"persisted evaluation identity differs: {name}")
        return copy.deepcopy(receipt["result"]), {
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path), "reused": True,
        }
    result = evaluate(brain, ports, genome, moments, **kwargs)
    artifact = {
        "format": "chreatures-affordance-evaluation-v1", "version": 1,
        "name": name, "completed_unix": time.time(),
        "identity": expected, "result": result,
    }
    receipt = atomic_json(path, artifact)
    receipt["reused"] = False
    return result, receipt


def main() -> int:
    args = arguments()
    if not 1 <= args.worlds <= 16 or not 1 <= args.eval_worlds <= args.worlds:
        raise SystemExit("world counts must satisfy 1 <= eval <= train <= 16")
    if args.steps % args.macro_steps or args.episode_steps % args.macro_steps or args.eval_steps % args.macro_steps:
        raise SystemExit("step counts must be divisible by macro steps")
    if args.sequence_length < 2:
        raise SystemExit("--sequence-length must be at least 2")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume and args.warm_start_learner:
        raise SystemExit("use either exact resume or warm-start, not both")
    if args.restore_audit_only and not args.resume:
        raise SystemExit("--restore-audit-only requires --resume")
    if args.resume_drops_pending_rollout and not args.resume:
        raise SystemExit("--resume-drops-pending-rollout requires --resume")
    if args.first_checkpoint and (
        not args.checkpoint_every or args.first_checkpoint >= args.checkpoint_every
    ):
        raise SystemExit("--first-checkpoint must be smaller than --checkpoint-every")
    training_profile = None
    if args.training_profile not in ("chemical-nursery-v3", "chemical-encounters-v4") and (
        args.chemical_habitat is not None or args.chemical_biosphere is not None
        or args.chemical_conditions is not None
    ):
        raise SystemExit("chemical config paths require a chemical training profile")
    if args.training_profile != "legacy":
        if args.resume:
            with gzip.open(args.resume.resolve(), "rt", encoding="utf-8") as handle:
                carried_profile = json.load(handle).get("training_profile")
            training_profile = (
                EmbodiedTrainingProfile.from_value(carried_profile)
                if carried_profile is not None else EmbodiedTrainingProfile.current()
            )
        else:
            if args.training_profile in ("chemical-nursery-v3", "chemical-encounters-v4"):
                if args.chemical_habitat is None or args.chemical_biosphere is None:
                    raise SystemExit(
                        "fresh chemical profile requires --chemical-habitat and --chemical-biosphere"
                    )
                if args.training_profile == "chemical-encounters-v4":
                    if args.chemical_conditions is None:
                        raise SystemExit(
                            "fresh chemical encounters require --chemical-conditions"
                        )
                    training_profile = EmbodiedTrainingProfile.chemical_encounters(
                        args.chemical_habitat, args.chemical_biosphere,
                        args.chemical_conditions,
                    )
                else:
                    if args.chemical_conditions is not None:
                        raise SystemExit("chemical nursery v3 does not accept encounter conditions")
                    training_profile = EmbodiedTrainingProfile.chemical_nursery(
                        args.chemical_habitat, args.chemical_biosphere
                    )
            else:
                training_profile = (
                    EmbodiedTrainingProfile.current_v2()
                    if args.training_profile == "current-life-v2"
                    else EmbodiedTrainingProfile.current()
                )
    if training_profile is not None:
        profile_version = int(training_profile.component("version"))
        requested_version = {
            "current-life-v1": 1,
            "current-life-v2": 2,
            "chemical-nursery-v3": 3,
            "chemical-encounters-v4": 4,
        }[args.training_profile]
        if profile_version != requested_version:
            raise SystemExit(
                f"checkpoint carries training profile v{profile_version}, "
                f"but --training-profile requests v{requested_version}"
            )
        if profile_version == 1 and args.curriculum_start_stage != 0:
            raise SystemExit("current-life-v1 requires --curriculum-start-stage 0")
        if profile_version == 2 and args.curriculum_start_stage not in range(3):
            raise SystemExit("current-life-v2 curriculum start stage must be 0, 1, or 2")
        if profile_version == 3 and args.curriculum_start_stage != 0:
            raise SystemExit("chemical-nursery-v3 requires --curriculum-start-stage 0")
        if profile_version == 4 and args.curriculum_start_stage not in range(3):
            raise SystemExit("chemical-encounters-v4 start stage must be 0, 1, or 2")
        horizons = training_profile.component("horizons")
        expected = (
            int(horizons["training_episode_steps"]),
            int(horizons["heldout_steps"]),
            int(horizons["checkpoint_every_steps"]),
        )
        actual = (args.episode_steps, args.eval_steps, args.checkpoint_every)
        if actual != expected:
            raise SystemExit(
                "selected training profile requires episode/eval/checkpoint steps "
                f"{expected}, received {actual}"
            )
        if args.steps < 2 * args.episode_steps:
            raise SystemExit("selected training profile requires at least two full training episodes")
        if args.reward_objective != "finite-energy-v1":
            raise SystemExit("selected training profile requires --reward-objective finite-energy-v1")
    graph = load_graph(args.graph)
    port_graph = load_graph(args.port_graph) if args.port_graph else graph
    if (
        port_graph is not graph
        and (port_graph.n != graph.n or not np.array_equal(port_graph.ids, graph.ids))
    ):
        raise SystemExit("port graph and recurrence graph neuron ordering differs")
    ports = NeuralPortBundle.load(args.port_bundle, port_graph)
    config = PredictivePPOConfig(
        feature_dim=len(ports.readout_names), macro_steps=args.macro_steps,
        seed=args.seed, std_profile=args.std_profile,
        context_profile=args.context_profile, sequence_length=args.sequence_length,
    )
    reward_objective = None
    if args.reward_objective == "finite-energy-v1":
        objective_config = (
            FiniteEnergyConfig.from_value(training_profile.component("homeostasis"))
            if training_profile is not None else FiniteEnergyConfig()
        )
        reward_objective = FiniteEnergyObjective(objective_config)
    if args.brain_backend in ("tiled", "triton", "microbatch"):
        brain = FixedCohortBrain(
            graph, ports, args.worlds * 3, device=args.device,
            backend=args.brain_backend, microbatch_size=args.microbatch_size,
        )
    else:
        brain = RemoteBrain(
            graph, capacity=args.worlds * 3, device=args.device,
            **ports.remote_brain_kwargs(),
        )
    source_paths = [
        ROOT / "chreatures" / name for name in (
            "learning.py", "fast_circuit.py", "tiled_circuit.py", "remote_brain.py",
            "malecns.py", "neural_ports.py", "physics.py", "articulated.py",
            "sensorium.py", "physical_batch.py", "training_environment.py",
            "homeostasis.py", "fields.py", "ecology.py", "acoustics.py",
            "biosphere.py", "somatic.py", "metabolism.py", "growth.py",
            "material_objects.py",
        )
    ] + [Path(__file__).resolve(), HABITAT, ROOT / "data/bodies/hexapod.json",
         ROOT / "data/ports/retinal-v1.json"]
    warm_source_std_profile = None
    warm_source_context_profile = None
    if args.warm_start_learner:
        warm_value = torch.load(
            args.warm_start_learner.resolve(), map_location="cpu", weights_only=False
        )
        warm_source_std_profile = warm_value.get("config", {}).get("std_profile", "global-v1")
        warm_source_context_profile = warm_value.get("config", {}).get(
            "context_profile", "reservoir-v1"
        )
        del warm_value
    resume_source_std_profile = None
    resume_source_context_profile = None
    if args.resume:
        with gzip.open(args.resume.resolve(), "rt", encoding="utf-8") as handle:
            resume_state = json.load(handle)
        learner_path = Path(resume_state["learner"]["path"])
        if not learner_path.is_absolute():
            learner_path = args.resume.resolve().parent / learner_path
        resume_value = torch.load(learner_path, map_location="cpu", weights_only=False)
        resume_source_std_profile = resume_value.get("config", {}).get(
            "std_profile", "global-v1"
        )
        resume_source_context_profile = resume_value.get("config", {}).get(
            "context_profile", "reservoir-v1"
        )
        del resume_value
    run_record = {
        "format": "chreatures-affordance-run-v1", "started_unix": time.time(),
        "pid": os.getpid(), "argv": [sys.executable, *sys.argv],
        "command": shlex.join([sys.executable, *sys.argv]),
        "graph_sha256": graph.hash, "port_spec_sha256": ports.spec_hash,
        "graph_loader": type(graph).__name__,
        "port_graph_sha256": port_graph.hash,
        "reward_objective": (
            reward_objective.config.to_value() if reward_objective is not None
            else {"format": "legacy-inline-v1"}
        ),
        "training_profile": training_profile.to_value() if training_profile is not None else None,
        "physical_backend": {
            "engine": args.physical_backend,
            "transition_authorized": bool(args.allow_physical_backend_transition),
            "semantics": "implementation-equivalent physical state and dynamics",
        },
        "port_bundle_sha256": sha256(args.port_bundle),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "libraries": {
            "python": sys.version.split()[0],
            "python_soabi": sysconfig.get_config_var("SOABI"),
            "numpy": np.__version__,
            "torch": torch.__version__, "torch_hip": torch.version.hip,
            "triton": package_version("triton"),
        },
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "device": brain.metadata()["device"],
        "native_world_kernels": (
            native_extension_receipt()
            if training_profile is not None
            and int(training_profile.component("version")) in (3, 4) else None
        ),
        "warm_start": (
            {"path": str(args.warm_start_learner.resolve()),
             "sha256": sha256(args.warm_start_learner.resolve()),
             "semantics": "shared model and normalization only; new private cohort",
             "source_std_profile": warm_source_std_profile,
             "target_std_profile": args.std_profile,
             "source_context_profile": warm_source_context_profile,
             "target_context_profile": args.context_profile,
             "optimizer": "fresh for every target-architecture parameter",
             "private_context": "reset",
             "variance_upgrade": (
                 "global-v1 to zero-offset state-conditioned-v2"
                 if warm_source_std_profile == "global-v1"
                 and args.std_profile == "state-conditioned-v2" else None
             )}
            if args.warm_start_learner else None
        ),
        "resume": (
            {"path": str(args.resume.resolve()), "sha256": sha256(args.resume.resolve()),
             "source_std_profile": resume_source_std_profile,
             "target_std_profile": args.std_profile,
             "source_context_profile": resume_source_context_profile,
             "target_context_profile": args.context_profile,
             "semantics": (
                 "neural, physical, learner, optimizer and private-state restore; "
                 "legacy pending rollout explicitly discarded"
                 if args.resume_drops_pending_rollout else
                 "exact neural, physical, learner, optimizer, private-state and rollout restore"
             ),
             "training_discontinuity": bool(args.resume_drops_pending_rollout)}
            if args.resume else None
        ),
        "architecture_transition": (
            {
                "source_std_profile": warm_source_std_profile,
                "target_std_profile": args.std_profile,
                "source_context_profile": warm_source_context_profile,
                "target_context_profile": args.context_profile,
                "semantics": "fresh descendant model inheritance; no private state",
                "optimizer": "fresh",
            }
            if args.warm_start_learner and (
                warm_source_std_profile != args.std_profile
                or warm_source_context_profile != args.context_profile
            ) else None
        ),
    }
    initial_genome = args.output / "initial-genome.npz"
    if args.resume:
        cohort, trainer, rollout, step, episode_step, raw, physiology = restore_checkpoint(
            args.resume.resolve(), brain, ports, args.workers, reward_objective,
            training_profile, args.physical_backend,
            args.allow_physical_backend_transition,
            args.std_profile,
            args.context_profile,
            args.curriculum_start_stage,
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
                "pending_rollout_decisions": len(rollout),
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
        rollout = MacroRollout()
        cohort = AffordanceCohort(
            brain, ports, args.worlds, args.workers, args.seed,
            reward_objective=reward_objective, training_profile=training_profile,
            physical_backend=args.physical_backend,
            curriculum_start_stage=args.curriculum_start_stage,
        )
        trainer = PredictivePPOTrainer(cohort.resident_ids, config, device=args.device)
        trainer.export_genome(initial_genome)
        comparison_genome = initial_genome
        comparison_scope = "fixed random initialization"
        if args.warm_start_learner:
            inherited, _ = PredictivePPOTrainer.restore(
                args.warm_start_learner.resolve(), device=args.device
            )
            if (
                inherited.config.std_profile == "global-v1"
                and config.std_profile == "state-conditioned-v2"
            ):
                inherited, _ = PredictivePPOTrainer.upgrade_state_conditioned(
                    args.warm_start_learner.resolve(), device=args.device,
                    expected_sha256=sha256(args.warm_start_learner.resolve()),
                )
            trainer.inherit_model(inherited)
            trainer.reset_private_state()
            comparison_genome = args.output / "inherited-comparison-genome.npz"
            trainer.export_genome(comparison_genome)
            comparison_scope = (
                "initial gated descendant after declared shared-model inheritance"
                if config.context_profile == "gated-v1" else
                "initial descendant after declared shared-model inheritance"
            )
        raw, physiology, neural = cohort.observe(0.05)
        normalized = trainer.normalize(raw, update=True)
        step = 0
        episode_step = 0
    if args.resume:
        architecture_comparison = args.output / "architecture-comparison-genome.npz"
        comparison_genome = (
            architecture_comparison if architecture_comparison.exists() else
            args.output / "inherited-comparison-genome.npz"
            if (args.output / "inherited-comparison-genome.npz").exists()
            else initial_genome
        )
        comparison_scope = (
            "zero-offset state-conditioned upgrade at architecture branch"
            if comparison_genome == architecture_comparison else
            "inherited policy before this reward-version stage"
            if comparison_genome != initial_genome else "fixed random initialization"
        )
    if args.comparison_genome is not None:
        comparison_genome = args.comparison_genome.resolve()
        if not comparison_genome.is_file():
            raise SystemExit("explicit comparison genome does not exist")
        comparison_scope = "explicit immutable transferred-policy comparison"
    process_start_step = step
    if step > args.steps:
        cohort.close()
        raise SystemExit("resume checkpoint is beyond requested --steps")
    evaluation_only_resume = bool(args.resume and step == args.steps)
    run_record_path = args.output / (
        "evaluation-resume.json" if evaluation_only_resume else "run.json"
    )
    atomic_json(run_record_path, run_record)
    started = time.perf_counter()
    stop = False
    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    updates, macro_rows = [], []
    checkpoints = (
        [existing_checkpoint_receipt(args.resume)] if evaluation_only_resume else []
    )
    training_totals = {
        name: 0.0 for name in (
            "nutrition", "nutrition_events", "absorbed", "ingested_mass",
            "ingestion_events", "mouth_material_contacts", "eat_request_steps",
            "contact_while_eating", "contacts", "distance",
        )
    }
    telemetry_every = (
        int(training_profile.component("horizons")["telemetry_every_steps"])
        if training_profile is not None else 0
    )
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
            totals = {name: 0.0 for name in training_totals}
            efforts, energies = [], []
            homeostasis_values: dict[str, list[float]] = {}
            for _ in range(args.macro_steps):
                reward, metrics = cohort.advance(previous["action"], 0.05)
                accumulated += reward
                raw, physiology, neural = cohort.observe(0.05)
                step += 1
                episode_step += 1
                for name in totals:
                    totals[name] += metrics[name]
                    training_totals[name] += metrics[name]
                efforts.append(metrics["effort"])
                energies.append(metrics["energy"])
                for name, value in metrics["homeostasis"].items():
                    homeostasis_values.setdefault(name, []).append(value)
                if telemetry_every and step % telemetry_every == 0:
                    cohort.last_world_telemetry = cohort.world_pool.call_all("telemetry")
                    telemetry = {
                        "format": "chreatures-embodied-training-telemetry-v1",
                        "step": step, "episode": cohort.episode,
                        "episode_step": episode_step,
                        "model_seconds_total": step * 0.05,
                        "model_seconds_episode": episode_step * 0.05,
                        "process_start_step": process_start_step,
                        "process_cumulative": training_totals.copy(),
                        "physiology": cohort.physiology_summary(),
                        "world_components": copy.deepcopy(cohort.last_world_telemetry),
                    }
                    with (args.output / "telemetry.jsonl").open("a") as handle:
                        handle.write(json.dumps(
                            telemetry, sort_keys=True,
                            default=lambda value: (
                                value.tolist() if isinstance(value, np.ndarray) else float(value)
                            ),
                        ) + "\n")
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
                "gut": metrics["gut"], "fatigue": metrics["fatigue"],
                "reserve_energy": metrics["reserve_energy"],
                "stationary_fraction": metrics["stationary_fraction"],
                "activity": float(np.mean([item["activity"] for item in neural])),
                "executed_action": {
                    name: {
                        "mean": float(previous["action"][:, index].mean()),
                        "abs_mean": float(np.abs(previous["action"][:, index]).mean()),
                        "std": float(previous["action"][:, index].std()),
                    }
                    for index, name in enumerate(ACTIONS)
                },
                "eat_request_mean": float(metrics["eat_request_mean"]),
                "homeostasis": {
                    name: float(np.mean(values))
                    for name, values in homeostasis_values.items()
                },
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
                        args.output, cohort, trainer, rollout, step, episode_step,
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
            args.output, cohort, trainer, rollout, step, episode_step, raw, physiology
        ))
        cohort.close()
        return 130
    learned_genome = args.output / "learned-genome.npz"
    learned_receipt = (
        existing_genome_receipt(learned_genome, trainer)
        if evaluation_only_resume and learned_genome.is_file()
        else trainer.export_genome(learned_genome)
    )
    if not checkpoints or checkpoints[-1]["step"] != step:
        checkpoints.append(save_checkpoint(
            args.output, cohort, trainer, rollout, step, episode_step, raw, physiology
        ))

    if evaluation_only_resume:
        completed_training = completed_training_receipt(args.output, step)
        training_elapsed = completed_training["elapsed_seconds"]
        training_timings = completed_training["timing_cumulative_seconds"]
    else:
        training_elapsed = time.perf_counter() - started
        training_timings = cohort.timings.copy()
        training_timings.update({"algorithm": algorithm_seconds, "ppo": ppo_seconds})
    training_terminal_outcomes = copy.deepcopy(cohort.terminal_outcomes)
    if cohort.episode_steps_advanced:
        training_terminal_outcomes.append({
            "episode": cohort.episode,
            "physical_steps": cohort.episode_steps_advanced,
            "partial_at_training_end": cohort.episode_steps_advanced < args.episode_steps,
            "worlds": cohort.world_pool.call_all("terminal"),
        })
    training_transport_timing = cohort.world_pool.timing_snapshot()
    cohort.close()
    evaluation_common = {
        "worlds": args.eval_worlds, "steps": args.eval_steps,
        "macro_steps": args.macro_steps, "workers": args.workers,
        "seed": args.seed + 900_000, "reward_objective": reward_objective,
        "training_profile": training_profile, "telemetry_every": telemetry_every,
        "physical_backend": args.physical_backend,
        "curriculum_start_stage": 2,
        "std_profile": args.std_profile,
    }
    brain_identity = stable_brain_identity(brain.metadata())
    evaluation_base_identity = {
        "graph_sha256": graph.hash,
        "port_spec_sha256": ports.spec_hash,
        "port_bundle_sha256": sha256(args.port_bundle),
        "training_profile_sha256": (
            training_profile.sha256 if training_profile is not None else None
        ),
        "reward_objective_sha256": canonical_sha256(run_record["reward_objective"]),
        "normalizer_sha256": canonical_sha256(trainer.moments.snapshot()),
        "source_sha256": copy.deepcopy(run_record["source_sha256"]),
        "training_checkpoint": stable_checkpoint_identity(checkpoints[-1]),
        "training_step": step,
        "learner_architecture": {
            "std_profile": trainer.config.std_profile,
            "config": asdict(trainer.config),
        },
        "neural": {
            "requested_backend": args.brain_backend,
            "microbatch_size": (
                args.microbatch_size if args.brain_backend == "microbatch" else None
            ),
            **brain_identity,
        },
        "runtime": {
            "libraries": copy.deepcopy(run_record["libraries"]),
            "native_world_kernels": stable_native_identity(
                run_record["native_world_kernels"]
            ),
            "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        },
        "evaluation": {
            "worlds": args.eval_worlds, "physical_steps": args.eval_steps,
            "macro_steps": args.macro_steps, "seed": args.seed + 900_000,
            "curriculum_stage": 2, "held_out": True,
            "deterministic_policy": True,
            "physical_backend": args.physical_backend,
        },
    }
    evaluations: dict[str, Any] = {}
    evaluation_receipts: dict[str, Any] = {}

    def run_evaluation_condition(
        name: str, genome: Path, *, silence_features: bool,
        ingestion_enabled: bool = True,
    ) -> None:
        identity = copy.deepcopy(evaluation_base_identity)
        identity.update({
            "condition": {
                "name": name, "silence_neural_features": silence_features,
                "ingestion_enabled": ingestion_enabled,
            },
            "policy_genome_sha256": sha256(genome),
        })
        result, receipt = persisted_evaluation(
            args.output, name, identity, brain, ports, genome, trainer.moments,
            silence_features=silence_features,
            ingestion_enabled=ingestion_enabled,
            **evaluation_common,
        )
        evaluations[name] = result
        evaluation_receipts[name] = receipt

    run_evaluation_condition(
        "fixed_comparison", comparison_genome, silence_features=False,
    )
    run_evaluation_condition("learned", learned_genome, silence_features=False)
    run_evaluation_condition(
        "learned_neural_silenced", learned_genome, silence_features=True,
    )
    if training_profile is not None and int(training_profile.component("version")) in (3, 4):
        run_evaluation_condition(
            "fixed_comparison_ingestion_disabled", comparison_genome,
            silence_features=False, ingestion_enabled=False,
        )
    summary = {
        "format": "chreatures-affordance-learning-v1",
        "completed": True, "steps": step,
        "evaluation_only_resume": evaluation_only_resume,
        "process_start_step": process_start_step,
        "steps_advanced": step - process_start_step,
        "resident_steps_advanced": (step - process_start_step) * args.worlds * 3,
        "cumulative_resident_steps": step * args.worlds * 3,
        "policy_exposure_resident_steps": (
            trainer.decision_count * args.macro_steps * args.worlds * 3
        ),
        "elapsed_training_seconds": training_elapsed,
        "training_timing_seconds": training_timings,
        "world_transport_timing": training_transport_timing,
        "config": vars(args) | {"graph": str(args.graph), "port_bundle": str(args.port_bundle), "output": str(args.output), "resume": str(args.resume) if args.resume else None},
        "learner": asdict(config), "learner_update_count": trainer.update_count,
        "updates_this_process": updates, "evaluations": evaluations,
        "evaluation_receipts": evaluation_receipts,
        "comparison_policy": {"scope": comparison_scope, "path": str(comparison_genome),
                              "sha256": sha256(comparison_genome)},
        "checkpoints": checkpoints, "initial_genome_sha256": hashlib.sha256(initial_genome.read_bytes()).hexdigest(),
        "learned_genome": learned_receipt, "brain": brain.metadata(),
        "command": shlex.join([sys.executable, *sys.argv]), "pid": os.getpid(),
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "training_profile": training_profile.to_value() if training_profile is not None else None,
        "training_terminal_outcomes": training_terminal_outcomes,
        "environment": {name: os.environ[name] for name in ("HSA_OVERRIDE_GFX_VERSION", "PYTORCH_KERNEL_CACHE_PATH") if name in os.environ},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        "evaluations": evaluations, "steps": step,
        "resident_steps_advanced": summary["resident_steps_advanced"],
        "cumulative_resident_steps": summary["cumulative_resident_steps"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
