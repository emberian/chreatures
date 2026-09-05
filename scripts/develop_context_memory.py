#!/usr/bin/env python3
"""Develop relational context memory on actual articulated 3-D trajectories."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chreatures.context_memory import ContextMemoryConfig, RelationalContextMemory
from chreatures.physics import PhysicsWorld
from chreatures.sensorium import ArticulatedSensoriumWorld, BODY_FRAME


def sensory_vector(values: dict[str, Any]) -> np.ndarray:
    """Flatten only body-accessible channels; never include IDs or positions."""
    pieces = [
        np.asarray(values["retina3d"], dtype=np.float32).reshape(-1),
        np.asarray(values["odor"], dtype=np.float32).reshape(-1),
        np.asarray(values["sound"], dtype=np.float32),
        np.asarray(values["touch"], dtype=np.float32),
        np.asarray([values["shade"], values["illumination"]], dtype=np.float32),
        np.asarray(values["linear_velocity"], dtype=np.float32),
        np.asarray(values["angular_velocity3d"], dtype=np.float32),
        np.asarray(values["tarsal_contact"], dtype=np.float32),
        np.asarray(values["joint_position"], dtype=np.float32),
        np.asarray(values["joint_velocity"], dtype=np.float32),
    ]
    result = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("physical sensor produced nonfinite values")
    return result


def action_vector(action: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [action.get("forward", 0.0), action.get("turn", 0.0), action.get("gaze_pitch", 0.0)],
        dtype=np.float32,
    )


def outcome_vector(outcome: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [outcome.get("distance", 0.0), outcome.get("contact", 0.0),
         outcome.get("effort", 0.0), outcome.get("nutrition", 0.0)],
        dtype=np.float32,
    )


def trajectory(seed: int, steps: int, physics_stride: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec = PhysicsWorld._load_spec(None)
    spec["sensorium"] = {"frame": BODY_FRAME}
    position = [float(rng.uniform(0.9, spec["size"][0] - 0.9)),
                float(rng.uniform(0.9, spec["size"][1] - 0.9)), 0.11]
    spec["bodies"] = [{
        "id": "resident", "name": "Resident", "position": position,
        "heading": float(rng.uniform(-math.pi, math.pi)), "material": "mica",
        "energy": 0.9, "gut": 0.1, "fatigue": 0.03,
    }]
    world = ArticulatedSensoriumWorld(seed=seed, spec=spec)
    for _ in range(20):
        world.advance({}, 0.05)
    observations = [sensory_vector(world.sense("resident"))]
    actions, outcomes = [], []
    action: dict[str, float] = {}
    for step in range(steps):
        if step % 8 == 0:
            action = {
                "forward": float(rng.choice((-0.35, 0.25, 0.55, 0.82))),
                "turn": float(rng.choice((-0.62, -0.28, 0.0, 0.28, 0.62))),
                "gaze_pitch": float(rng.choice((-0.45, 0.0, 0.45))),
            }
        aggregate = {"distance": 0.0, "contact": 0.0, "effort": 0.0, "nutrition": 0.0}
        for _ in range(physics_stride):
            result = world.advance({"resident": action}, 0.05)["resident"]
            aggregate["distance"] += result["distance"]
            aggregate["contact"] = max(aggregate["contact"], result["contact"])
            aggregate["effort"] += result["effort"] / physics_stride
            aggregate["nutrition"] += result["nutrition"]
        actions.append(action_vector(action))
        outcomes.append(outcome_vector(aggregate))
        observations.append(sensory_vector(world.sense("resident")))
    return {
        "observation": np.asarray(observations, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "outcome": np.asarray(outcomes, dtype=np.float32),
    }


class FrozenProjection:
    def __init__(self, training: list[dict[str, np.ndarray]], output_dim: int, seed: int):
        raw = np.concatenate([item["observation"] for item in training], axis=0)
        self.mean = raw.mean(axis=0)
        self.scale = np.maximum(raw.std(axis=0), 0.04)
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0.0, 1.0, (output_dim, raw.shape[1])).astype(np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        self.matrix = matrix

    def __call__(self, value: np.ndarray) -> np.ndarray:
        normalized = np.clip((value - self.mean) / self.scale, -5.0, 5.0)
        return np.tanh(normalized @ self.matrix.T).astype(np.float32)


class ShortHistoryKNN:
    """Bounded two-step action/outcome history nearest-neighbor baseline."""

    def __init__(self, feature_dim: int, action_dim: int, outcome_dim: int, capacity: int, history: int = 2):
        self.feature_dim, self.action_dim, self.outcome_dim = feature_dim, action_dim, outcome_dim
        self.capacity, self.history = capacity, history
        self.records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.path: deque[np.ndarray] = deque(maxlen=history)

    def reset(self) -> None:
        self.path.clear()

    def _history(self) -> np.ndarray:
        width = self.action_dim + self.outcome_dim
        missing = self.history - len(self.path)
        return np.concatenate([np.zeros(missing * width, dtype=np.float32), *self.path])

    def bind(self, observation, action, outcome, next_observation) -> None:
        key = np.concatenate((observation, self._history())).astype(np.float32)
        self.records.append((key, action.copy(), next_observation - observation, outcome.copy()))
        if len(self.records) > self.capacity:
            del self.records[0]
        self.path.append(np.concatenate((action, outcome)).astype(np.float32))

    def predict(self, observation, action) -> tuple[np.ndarray, np.ndarray]:
        key = np.concatenate((observation, self._history())).astype(np.float32)
        keys = np.asarray([record[0] for record in self.records])
        actions = np.asarray([record[1] for record in self.records])
        distance = np.mean((keys - key) ** 2, axis=1) + 0.20 * np.mean((actions - action) ** 2, axis=1)
        indices = np.argsort(distance)[:5]
        weight = np.exp(-distance[indices] * 4.0); weight /= max(float(weight.sum()), 1e-9)
        delta = sum(w * self.records[i][2] for w, i in zip(weight, indices, strict=True))
        outcome = sum(w * self.records[i][3] for w, i in zip(weight, indices, strict=True))
        return observation + delta, outcome

    def advance(self, action, outcome) -> None:
        self.path.append(np.concatenate((action, outcome)).astype(np.float32))


def projected(items: list[dict[str, np.ndarray]], projection: FrozenProjection) -> list[dict[str, np.ndarray]]:
    return [{**item, "feature": projection(item["observation"])} for item in items]


def train_models(train, feature_dim: int, capacity: int):
    memory = RelationalContextMemory(ContextMemoryConfig(
        feature_dim=feature_dim, action_dim=3, outcome_dim=4,
        context_capacity=max(32, capacity // 4), transition_capacity=capacity,
        observation_bandwidth=0.48, action_bandwidth=0.24,
        new_context_distance=0.58, clone_min_visits=3,
    ))
    baseline = ShortHistoryKNN(feature_dim, 3, 4, capacity)
    for item in train:
        memory.reset(); memory.begin(item["feature"][0], learn=True); baseline.reset()
        for index, action in enumerate(item["action"]):
            memory.step(action, item["feature"][index + 1], item["outcome"][index], learn=True)
            baseline.bind(item["feature"][index], action, item["outcome"][index], item["feature"][index + 1])
    return memory, baseline


def ambiguity_scores(train, test) -> np.ndarray:
    observation = np.concatenate([item["feature"][:-1] for item in train])
    action = np.concatenate([item["action"] for item in train])
    delta = np.concatenate([item["feature"][1:] - item["feature"][:-1] for item in train])
    result = []
    for item in test:
        for index, current_action in enumerate(item["action"]):
            d = np.mean((observation - item["feature"][index]) ** 2, axis=1)
            d += 0.15 * np.mean((action - current_action) ** 2, axis=1)
            nearest = np.argsort(d)[:8]
            result.append(float(np.mean(np.var(delta[nearest], axis=0))) / max(float(d[nearest[0]]), 1e-5))
    return np.asarray(result)


def evaluate(memory, baseline, test, ambiguity: np.ndarray) -> dict[str, Any]:
    relational_error, baseline_error = [], []
    relational_outcome, baseline_outcome, uncertainty = [], [], []
    cursor = 0
    for item in test:
        memory.reset(); memory.begin(item["feature"][0], learn=False); baseline.reset()
        for index, action in enumerate(item["action"]):
            relational = memory.predict(action)
            predicted = np.asarray(relational["next_observation"], dtype=np.float32)
            base_predicted, base_outcome = baseline.predict(item["feature"][index], action)
            target = item["feature"][index + 1]; outcome = item["outcome"][index]
            relational_error.append(float(np.mean((predicted - target) ** 2)))
            baseline_error.append(float(np.mean((base_predicted - target) ** 2)))
            relational_outcome.append(float(np.mean((np.asarray(relational["outcome"]) - outcome) ** 2)))
            baseline_outcome.append(float(np.mean((base_outcome - outcome) ** 2)))
            uncertainty.append(float(relational["uncertainty"]))
            memory.step(action, target, outcome, learn=False)
            baseline.advance(action, outcome)
            cursor += 1
    relational_error = np.asarray(relational_error); baseline_error = np.asarray(baseline_error)
    cutoff = float(np.quantile(ambiguity, 0.75)); alias = ambiguity >= cutoff
    correlation = float(np.corrcoef(uncertainty, relational_error)[0, 1])
    return {
        "transitions": cursor,
        "overall_next_mse": {"relational": float(relational_error.mean()), "short_history": float(baseline_error.mean())},
        "overall_outcome_mse": {"relational": float(np.mean(relational_outcome)), "short_history": float(np.mean(baseline_outcome))},
        "aliased_subset": {
            "count": int(alias.sum()), "selection": "top quartile local successor disagreement / observation-action distance",
            "relational_next_mse": float(relational_error[alias].mean()),
            "short_history_next_mse": float(baseline_error[alias].mean()),
        },
        "uncertainty_error_correlation": correlation,
    }


def rapid_binding_check(train, feature_dim: int) -> dict[str, float]:
    """Bind one high-change physical transition, then query it once."""
    candidates = []
    for item in train:
        delta = item["feature"][1:] - item["feature"][:-1]
        for index, magnitude in enumerate(np.mean(delta * delta, axis=1)):
            candidates.append((float(magnitude), item, index))
    _, item, index = max(candidates, key=lambda value: value[0])
    memory = RelationalContextMemory(ContextMemoryConfig(
        feature_dim, 3, 4, context_capacity=8, transition_capacity=16,
        observation_bandwidth=0.48, action_bandwidth=0.24,
        new_context_distance=0.58,
    ))
    current, target = item["feature"][index], item["feature"][index + 1]
    action, outcome = item["action"][index], item["outcome"][index]
    memory.begin(current, learn=True)
    before = np.asarray(memory.predict(action)["next_observation"])
    memory.step(action, target, outcome, learn=True)
    memory.reset(); memory.begin(current, learn=False)
    after = np.asarray(memory.predict(action)["next_observation"])
    return {
        "before_next_mse": float(np.mean((before - target) ** 2)),
        "after_one_experience_next_mse": float(np.mean((after - target) ** 2)),
        "contexts": memory.context_count,
        "transitions": memory.transition_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trajectories", type=int, default=8)
    parser.add_argument("--test-trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--physics-stride", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if min(args.train_trajectories, args.test_trajectories) < 1 or args.steps < 16:
        parser.error("trajectory counts must be positive and steps >=16")
    started = time.perf_counter()
    train_raw = [trajectory(args.seed + index * 101, args.steps, args.physics_stride) for index in range(args.train_trajectories)]
    test_raw = [trajectory(args.seed + 1_000_003 + index * 103, args.steps, args.physics_stride) for index in range(args.test_trajectories)]
    projection = FrozenProjection(train_raw, args.feature_dim, args.seed + 17)
    train, test = projected(train_raw, projection), projected(test_raw, projection)
    memory, baseline = train_models(train, args.feature_dim, args.capacity)
    ambiguity = ambiguity_scores(train, test)
    report = {
        "format": "chreatures-context-memory-development-v1",
        "source": "actual ArticulatedSensoriumWorld body-v1 physical trajectories",
        "privacy_boundary": "retina/local chemical/audio/contact/proprioception + executed action/outcome; no coordinates or IDs",
        "train_trajectories": args.train_trajectories,
        "test_trajectories": args.test_trajectories,
        "steps_per_trajectory": args.steps,
        "physics_seconds_per_transition": args.physics_stride * 0.05,
        "raw_sensor_dim": int(train_raw[0]["observation"].shape[1]),
        "feature_dim": args.feature_dim,
        "contexts": memory.context_count,
        "stored_transitions": memory.transition_count,
        "transition_capacity": args.capacity,
        "rapid_binding": rapid_binding_check(train, args.feature_dim),
        "evaluation": evaluate(memory, baseline, test, ambiguity),
        "elapsed_seconds": time.perf_counter() - started,
    }
    restored = RelationalContextMemory.restore(memory.snapshot())
    report["snapshot_exact"] = restored.snapshot() == memory.snapshot()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
