"""Batched developmental runs coupling full MaleCNS state to MuJoCo worlds."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cognition import AdaptiveOrgan
from .neural_client import sensory_channels
from .physics import PhysicsWorld
from .remote_brain import RemoteBrain


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HABITAT = ROOT / "data/habitats/hollow-garden.json"


@dataclass
class DevelopmentConfig:
    worlds: int = 8
    simple_steps: int = 768
    rich_steps: int = 1280
    dt: float = 0.05
    checkpoint_every: int = 512
    record_every: int = 1
    workers: int = 8
    seed: int = 20260905
    inheritance_seed: int = 7301

    @property
    def residents(self) -> int:
        return self.worlds * 3

    @property
    def total_steps(self) -> int:
        return self.simple_steps + self.rich_steps


def _safe_position(value: float, high: float, margin: float = 0.25) -> float:
    return float(np.clip(value, margin, high - margin))


def nursery_spec(seed: int, phase: str) -> dict[str, Any]:
    """Vary benign layouts without exposing layout facts to a policy."""
    if phase not in {"simple", "rich"}:
        raise ValueError("phase must be simple or rich")
    spec = json.loads(DEFAULT_HABITAT.read_text())
    rng = np.random.default_rng(seed)
    width, height = map(float, spec["size"][:2])
    for body in spec["bodies"]:
        body["position"][0] = _safe_position(body["position"][0] + rng.uniform(-0.18, 0.18), width)
        body["position"][1] = _safe_position(body["position"][1] + rng.uniform(-0.18, 0.18), height)
        body["heading"] = float(rng.uniform(-np.pi, np.pi))
        body["energy"] = float(rng.uniform(0.74, 0.84))
        body["gut"] = float(rng.uniform(0.08, 0.18))
        body["fatigue"] = float(rng.uniform(0.03, 0.08))

    by_id = {entity["id"]: entity for entity in spec["entities"]}
    food_ids = ("berry-a", "berry-b", "nectar-a", "nectar-b")
    for body, entity_id in zip(spec["bodies"], food_ids[:3], strict=True):
        angle = float(rng.uniform(-np.pi, np.pi))
        distance = float(rng.uniform(0.24, 0.42) if phase == "simple" else rng.uniform(0.42, 0.85))
        entity = by_id[entity_id]
        entity["position"][0] = _safe_position(body["position"][0] + np.cos(angle) * distance, width)
        entity["position"][1] = _safe_position(body["position"][1] + np.sin(angle) * distance, height)
        entity["position"][2] = 0.14

    for entity_id in ("violet-ball", "cyan-ball", "stack-box-a", "stack-box-b"):
        entity = by_id[entity_id]
        entity["position"][0] = _safe_position(entity["position"][0] + rng.uniform(-0.55, 0.55), width)
        entity["position"][1] = _safe_position(entity["position"][1] + rng.uniform(-0.55, 0.55), height)

    if phase == "simple":
        keep = {
            "ground", "west-wall", "east-wall", "north-wall", "south-wall",
            *food_ids, "violet-ball", "cyan-ball",
        }
        spec["entities"] = [entity for entity in spec["entities"] if entity["id"] in keep]
        spec["name"] = "hollow-garden-simple-affordances"
    else:
        for group in (
            ("high-walk", "west-ramp", "east-ramp"),
            ("seesaw", "seesaw-fulcrum"),
            ("pendulum", "pendulum-frame"),
            ("hollow-arch",),
        ):
            dx, dy = rng.uniform(-0.22, 0.22, size=2)
            for entity_id in group:
                entity = by_id[entity_id]
                entity["position"][0] = _safe_position(entity["position"][0] + dx, width)
                entity["position"][1] = _safe_position(entity["position"][1] + dy, height)
        spec["name"] = "hollow-garden-rich-development"
    return spec


class DevelopmentNursery:
    """Independent physical worlds sharing one full sparse GPU anatomy."""

    def __init__(
        self,
        brain: RemoteBrain,
        output_directory: str | Path,
        config: DevelopmentConfig,
    ) -> None:
        if config.residents > brain.capacity:
            raise ValueError("brain capacity is smaller than the nursery population")
        self.brain = brain
        self.output_directory = Path(output_directory)
        self.config = config
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (self.output_directory / "checkpoints").mkdir(exist_ok=True)
        self.phase = "simple"
        self.phase_step = 0
        self.step_index = 0
        self.worlds = self._make_worlds("simple")
        self.resident_ids = [
            f"world-{world_index:02d}:{body.id}"
            for world_index, world in enumerate(self.worlds)
            for body in world.bodies
        ]
        self.brain.add_residents(self.resident_ids)
        self.organs: dict[str, AdaptiveOrgan] = {}
        for index, resident in enumerate(self.resident_ids):
            organ = AdaptiveOrgan(feature_dim=48, seed=config.inheritance_seed)
            organ.rng = np.random.default_rng(config.seed * 1009 + index)
            self.organs[resident] = organ
        self.initial_parameters = {
            name: getattr(next(iter(self.organs.values())), name).copy()
            for name in ("actor", "critic", "model")
        }
        self.outcomes = {resident: {} for resident in self.resident_ids}
        self.feature_mean = {
            resident: np.zeros(48, dtype=np.float32) for resident in self.resident_ids
        }
        self.feature_variance = {
            resident: np.ones(48, dtype=np.float32) * 0.01 for resident in self.resident_ids
        }
        self.trajectory: dict[str, list[np.ndarray]] = {
            "step": [], "phase": [], "features": [], "normalized_features": [],
            "actions": [], "physiology": [], "outcomes": [], "activity": [],
            "support": [], "advantage": [], "prediction_error": [],
            "learning_progress": [],
        }
        self.log_path = self.output_directory / "development.jsonl"
        self.started = time.perf_counter()
        self._executor = ThreadPoolExecutor(max_workers=min(config.workers, config.worlds))

    @classmethod
    def restore(
        cls,
        brain: RemoteBrain,
        checkpoint_path: str | Path,
        *,
        output_directory: str | Path | None = None,
    ) -> "DevelopmentNursery":
        """Resume an exact physical, neural, cognitive and random state."""
        checkpoint_path = Path(checkpoint_path).resolve()
        with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("version") != 1 or state.get("graph_sha256") != brain.graph_hash:
            raise ValueError("development checkpoint is incompatible with this graph")
        config = DevelopmentConfig(**state["config"])
        if config.residents > brain.capacity:
            raise ValueError("brain capacity is smaller than the checkpoint population")
        directory = (
            Path(output_directory).resolve()
            if output_directory is not None
            else checkpoint_path.parent.parent
        )
        neural = state["neural"]
        brain.restore(
            checkpoint_path.parent,
            neural["name"],
            expected_sha256=neural["sha256"],
        )
        instance = cls.__new__(cls)
        instance.brain = brain
        instance.output_directory = directory
        instance.config = config
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "checkpoints").mkdir(exist_ok=True)
        instance.phase = str(state["phase"])
        instance.phase_step = int(state["phase_step"])
        instance.step_index = int(state["step"])
        instance.worlds = [PhysicsWorld.restore(value) for value in state["worlds"]]
        instance.resident_ids = [str(value) for value in state["resident_ids"]]
        if brain.resident_ids != instance.resident_ids:
            raise ValueError("checkpoint resident ordering differs from neural snapshot")
        expected_ids = [
            f"world-{world_index:02d}:{body.id}"
            for world_index, world in enumerate(instance.worlds)
            for body in world.bodies
        ]
        if expected_ids != instance.resident_ids:
            raise ValueError("checkpoint physical residents differ from neural residents")
        instance.organs = {
            resident: AdaptiveOrgan.restore(state["organs"][resident])
            for resident in instance.resident_ids
        }
        instance.initial_parameters = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in state["initial_parameters"].items()
        }
        instance.outcomes = {
            resident: dict(state["outcomes"][resident])
            for resident in instance.resident_ids
        }
        instance.feature_mean = {
            resident: np.asarray(state["feature_mean"][resident], dtype=np.float32)
            for resident in instance.resident_ids
        }
        instance.feature_variance = {
            resident: np.asarray(state["feature_variance"][resident], dtype=np.float32)
            for resident in instance.resident_ids
        }
        instance.trajectory = {
            "step": [], "phase": [], "features": [], "normalized_features": [],
            "actions": [], "physiology": [], "outcomes": [], "activity": [],
            "support": [], "advantage": [], "prediction_error": [],
            "learning_progress": [],
        }
        instance.log_path = directory / "development.jsonl"
        instance.started = time.perf_counter()
        instance._executor = ThreadPoolExecutor(
            max_workers=min(config.workers, config.worlds)
        )
        return instance

    def _make_worlds(self, phase: str) -> list[PhysicsWorld]:
        offset = 0 if phase == "simple" else 100_000
        return [
            PhysicsWorld(
                seed=self.config.seed + offset + index,
                spec=nursery_spec(self.config.seed + offset + index, phase),
            )
            for index in range(self.config.worlds)
        ]

    def _sense_world(self, item: tuple[int, PhysicsWorld]):
        index, world = item
        return index, {body.id: world.sense(body.id) for body in world.bodies}

    def _advance_world(self, item: tuple[PhysicsWorld, dict[str, dict[str, float]]]):
        world, actions = item
        return world.advance(actions, self.config.dt)

    def _begin_rich_phase(self) -> None:
        self.phase = "rich"
        self.phase_step = 0
        self.worlds = self._make_worlds("rich")
        self.outcomes = {resident: {} for resident in self.resident_ids}
        for organ in self.organs.values():
            # A habitat reset is not a consequence of the preceding action.
            organ.last_energy = None

    def step(self) -> dict[str, float]:
        sensed_by_world = dict(
            self._executor.map(self._sense_world, enumerate(self.worlds))
        )
        entries = []
        for world_index, world in enumerate(self.worlds):
            for body in world.bodies:
                resident = f"world-{world_index:02d}:{body.id}"
                entries.append(
                    {"id": resident, "senses": sensory_channels(sensed_by_world[world_index][body.id])}
                )
        neural = self.brain.step(entries, self.config.dt)
        response_by_id = {response["id"]: response for response in neural}
        actions_by_world: list[dict[str, dict[str, float]]] = []
        recorded_features = []
        recorded_normalized = []
        recorded_actions = []
        recorded_physiology = []
        recorded_activity = []
        recorded_support = []
        recorded_advantage = []
        recorded_error = []
        recorded_progress = []
        for world_index, world in enumerate(self.worlds):
            world_actions = {}
            for body in world.bodies:
                resident = f"world-{world_index:02d}:{body.id}"
                response = response_by_id[resident]
                features = np.asarray(response["features"], dtype=np.float32)
                mean = self.feature_mean[resident]
                variance = self.feature_variance[resident]
                delta = features - mean
                mean += np.float32(self.config.dt / 20) * delta
                variance += np.float32(self.config.dt / 20) * (delta * delta - variance)
                normalized = np.clip(
                    delta / np.sqrt(np.maximum(variance, 1e-6)), -2, 2
                ).astype(np.float32)
                physiology = {
                    key: float(getattr(body, key))
                    for key in ("energy", "gut", "fatigue", "speed", "angular_velocity")
                }
                physiology["support"] = float(response["support"])
                action = self.organs[resident].step(
                    normalized,
                    physiology,
                    self.outcomes[resident],
                    self.config.dt,
                )
                for channel in ("grip", "signal_low", "signal_mid", "signal_high"):
                    action[channel] = max(0.0, action[channel])
                action["eat"] = float(
                    np.clip((1 - body.gut) * (1.1 - body.energy), 0, 1)
                )
                world_actions[body.id] = action
                metrics = self.organs[resident].last_metrics
                recorded_features.append(features)
                recorded_normalized.append(normalized)
                recorded_actions.append(
                    [action[name] for name in AdaptiveOrgan.ACTIONS]
                )
                recorded_physiology.append(
                    [physiology[name] for name in ("energy", "gut", "fatigue", "speed", "angular_velocity", "support")]
                )
                recorded_activity.append(response["activity"])
                recorded_support.append(response["support"])
                recorded_advantage.append(metrics["advantage"])
                recorded_error.append(metrics["prediction_error"])
                recorded_progress.append(metrics["learning_progress"])
            actions_by_world.append(world_actions)
        advanced = list(
            self._executor.map(self._advance_world, zip(self.worlds, actions_by_world))
        )
        recorded_outcomes = []
        for world_index, outcome in enumerate(advanced):
            for body in self.worlds[world_index].bodies:
                resident = f"world-{world_index:02d}:{body.id}"
                self.outcomes[resident] = outcome[body.id]
                recorded_outcomes.append(
                    [outcome[body.id][name] for name in ("nutrition", "contact", "distance", "effort")]
                )

        if self.step_index % self.config.record_every == 0:
            self.trajectory["step"].append(np.asarray(self.step_index, dtype=np.int32))
            self.trajectory["phase"].append(np.asarray(0 if self.phase == "simple" else 1, dtype=np.int8))
            for name, values in (
                ("features", recorded_features),
                ("normalized_features", recorded_normalized),
                ("actions", recorded_actions),
                ("physiology", recorded_physiology),
                ("outcomes", recorded_outcomes),
                ("activity", recorded_activity),
                ("support", recorded_support),
                ("advantage", recorded_advantage),
                ("prediction_error", recorded_error),
                ("learning_progress", recorded_progress),
            ):
                self.trajectory[name].append(np.asarray(values, dtype=np.float32))

        self.step_index += 1
        self.phase_step += 1
        metrics = {
            "step": self.step_index,
            "phase": self.phase,
            "nutrition": float(np.sum(np.asarray(recorded_outcomes)[:, 0])),
            "contacts": float(np.sum(np.asarray(recorded_outcomes)[:, 1] > 0)),
            "distance": float(np.sum(np.asarray(recorded_outcomes)[:, 2])),
            "effort": float(np.mean(np.asarray(recorded_outcomes)[:, 3])),
            "energy": float(np.mean(np.asarray(recorded_physiology)[:, 0])),
            "activity": float(np.mean(recorded_activity)),
            "advantage": float(np.mean(recorded_advantage)),
            "prediction_error": float(np.mean(recorded_error)),
            "elapsed_seconds": time.perf_counter() - self.started,
        }
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        if self.step_index == self.config.simple_steps:
            self._begin_rich_phase()
        return metrics

    def save_checkpoint(self) -> dict[str, Any]:
        name = f"neural-step-{self.step_index:06d}"
        neural = self.brain.snapshot(self.output_directory / "checkpoints", name)
        state = {
            "version": 1,
            "step": self.step_index,
            "phase": self.phase,
            "phase_step": self.phase_step,
            "config": asdict(self.config),
            "graph_sha256": self.brain.graph_hash,
            "neural": neural,
            "resident_ids": self.resident_ids,
            "initial_parameters": {
                name: value.tolist() for name, value in self.initial_parameters.items()
            },
            "worlds": [world.snapshot() for world in self.worlds],
            "organs": {resident: organ.snapshot() for resident, organ in self.organs.items()},
            "outcomes": self.outcomes,
            "feature_mean": {key: value.tolist() for key, value in self.feature_mean.items()},
            "feature_variance": {key: value.tolist() for key, value in self.feature_variance.items()},
        }
        encoded = json.dumps(state, separators=(",", ":"), allow_nan=False).encode()
        path = self.output_directory / "checkpoints" / f"development-step-{self.step_index:06d}.json.gz"
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wb", compresslevel=6) as handle:
            handle.write(encoded)
        os.replace(temporary, path)
        return {
            "step": self.step_index,
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "neural": neural,
        }

    def export(self) -> dict[str, Any]:
        for organ in self.organs.values():
            organ.consolidate(64)
        arrays = {}
        inherited_fields = (
            "context_in", "context_recur", "encoder", "actor", "critic", "model"
        )
        for name in inherited_fields:
            arrays[name] = np.mean(
                np.stack([getattr(organ, name) for organ in self.organs.values()]),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
        arrays["feature_mean"] = np.mean(
            np.stack(list(self.feature_mean.values())), axis=0
        ).astype(np.float32)
        arrays["feature_variance"] = np.mean(
            np.stack(list(self.feature_variance.values())), axis=0
        ).astype(np.float32)
        egg_path = self.output_directory / "egg.npz"
        np.savez_compressed(egg_path, **arrays)
        episodes = {
            resident: {
                "memory": organ.memory.snapshot(),
                "view": organ.view(),
            }
            for resident, organ in self.organs.items()
        }
        episodes_path = self.output_directory / "experienced_episodes.json.gz"
        with gzip.open(episodes_path, "wt", encoding="utf-8") as handle:
            json.dump(episodes, handle, separators=(",", ":"), allow_nan=False)
        deltas = {
            name: float(
                np.linalg.norm(arrays[name] - self.initial_parameters[name])
            )
            for name in ("actor", "critic", "model")
        }
        manifest = {
            "format": "chreatures-developmental-egg-v1",
            "scope": "inherited sensorimotor interface; no personal neural or autobiographical state",
            "graph_sha256": self.brain.graph_hash,
            "config": asdict(self.config),
            "residents": len(self.resident_ids),
            "steps": self.step_index,
            "parameter_l2_change": deltas,
            "artifacts": {},
        }
        for path in (egg_path, episodes_path, self.log_path):
            manifest["artifacts"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest_path = self.output_directory / "egg-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest

    def save_trajectory(self) -> Path:
        path = self.output_directory / "trajectory.npz"
        np.savez_compressed(
            path,
            resident_ids=np.asarray(self.resident_ids),
            **{name: np.stack(values) for name, values in self.trajectory.items()},
        )
        return path

    def close(self) -> None:
        self._executor.shutdown(wait=True)


def apply_egg(organ: AdaptiveOrgan, egg_path: str | Path) -> None:
    """Load only inherited arrays, leaving personal state and memory untouched."""
    with np.load(egg_path, allow_pickle=False) as data:
        expected = {"context_in", "context_recur", "encoder", "actor", "critic", "model"}
        if not expected.issubset(data.files):
            raise ValueError("egg is missing inherited arrays")
        for name in expected:
            value = np.asarray(data[name], dtype=np.float32)
            target = getattr(organ, name)
            if value.shape != target.shape or not np.isfinite(value).all():
                raise ValueError(f"invalid inherited array: {name}")
            target[...] = value
