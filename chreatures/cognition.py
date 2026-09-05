"""Personal sensorimotor memory and an adaptive continuous motor interface.

This module deliberately has no fruit-color, target-coordinate, creature-ID or
object-kind logic. Its inputs are circuit features, sensed proprioception and
bodily outcomes. A small online actor/critic and an action-conditioned forward
model learn around the immutable anatomical scaffold. It is a synthetic
learning mechanism, not a claim about recovered Drosophila cognition.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass
class LearningGenome:
    actor_rate: float = 0.006
    critic_rate: float = 0.025
    model_rate: float = 0.012
    discount_seconds: float = 8.0
    eligibility_seconds: float = 2.0
    exploration: float = 0.32
    curiosity_gain: float = 0.08
    memory_capacity: int = 384
    memory_threshold: float = 0.22


class PersonalMemory:
    """Finite, selective, privately acquired encounters with contextual retrieval.

    Similar sensory views are disambiguated by a recurrent history key. The
    archive remains elsewhere: absence from this store means the organism may
    no longer retrieve an encounter, even if a human can inspect its record.
    """
    def __init__(self, feature_dim: int, context_dim: int, capacity=384):
        self.feature_dim, self.context_dim, self.capacity = feature_dim, context_dim, capacity
        self.records: list[dict[str, Any]] = []
        self.clock = 0

    def recall(self, features, context, limit=4):
        if not self.records:
            return np.zeros(self.feature_dim, dtype=np.float32), 0.0, 1.0
        keys = np.asarray([r["features"] for r in self.records], dtype=np.float32)
        contexts = np.asarray([r["context"] for r in self.records], dtype=np.float32)
        distance = np.mean((keys - features) ** 2, axis=1) + 0.35 * np.mean((contexts - context) ** 2, axis=1)
        indices = np.argsort(distance)[:limit]
        weights = np.exp(-distance[indices] * 18)
        weights /= max(float(weights.sum()), 1e-8)
        prediction = sum(w * np.asarray(self.records[i]["next"], dtype=np.float32) for w, i in zip(weights, indices))
        value = float(sum(w * self.records[i]["outcome"] for w, i in zip(weights, indices)))
        novelty = float(np.clip(np.sqrt(distance[indices[0]]) * 3, 0, 1))
        return prediction.astype(np.float32), value, novelty

    def remember(self, features, context, action, next_features, outcome, model_time):
        self.clock += 1
        self.records.append({"features": np.asarray(features).tolist(), "context": np.asarray(context).tolist(),
                             "action": np.asarray(action).tolist(), "next": np.asarray(next_features).tolist(),
                             "outcome": float(outcome), "time": float(model_time), "origin": "experienced",
                             "sequence": self.clock})
        if len(self.records) > self.capacity:
            # Retain rare/salient and recent experiences instead of perfect playback.
            scores = [abs(r["outcome"]) * 8 + (r["sequence"] / max(1, self.clock)) for r in self.records]
            del self.records[int(np.argmin(scores))]

    def snapshot(self):
        return {"feature_dim": self.feature_dim, "context_dim": self.context_dim, "capacity": self.capacity,
                "clock": self.clock, "records": copy.deepcopy(self.records)}

    @classmethod
    def restore(cls, value):
        instance = cls(value["feature_dim"], value["context_dim"], value["capacity"])
        if len(value["records"]) > instance.capacity:
            raise ValueError("Memory exceeds inherited capacity")
        instance.clock = int(value["clock"])
        instance.records = copy.deepcopy(value["records"])
        for record in instance.records:
            if record.get("origin") != "experienced":
                raise ValueError("Unrecognized autobiographical origin")
        return instance


class AdaptiveOrgan:
    """A learned action interface and predictive memory over neural features.

    Actions are eight continuous actuator coordinates; the body adapter defines
    their units. Defaults: thrust, yaw, gaze pitch, grip, three signal amplitudes,
    and vertical/postural effort. This layer never assumes a scene layout.
    """
    VERSION = 1
    ACTIONS = ("thrust", "yaw", "gaze_pitch", "grip", "signal_low", "signal_mid", "signal_high", "posture")

    def __init__(self, feature_dim=48, seed=0, genome: dict | None = None):
        self.feature_dim = int(feature_dim)
        self.context_dim = 24
        self.genome = LearningGenome(**(genome or {}))
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.learning = True
        self.context = np.zeros(self.context_dim, dtype=np.float32)
        self.context_in = self.rng.normal(0, 0.16, (self.context_dim, feature_dim + len(self.ACTIONS))).astype(np.float32)
        self.context_recur = self.rng.normal(0, 0.11, (self.context_dim, self.context_dim)).astype(np.float32)
        norm = np.linalg.norm(self.context_recur, ord=2)
        self.context_recur *= 0.70 / max(norm, 1e-6)
        # Current state, remembered expectation, bounded history, body variables.
        self.input_dim = feature_dim * 2 + self.context_dim + 6
        self.hidden_dim = 48
        self.encoder = self.rng.normal(0, 1 / math.sqrt(self.input_dim), (self.hidden_dim, self.input_dim)).astype(np.float32)
        self.actor = self.rng.normal(0, 0.025, (len(self.ACTIONS), self.hidden_dim + 1)).astype(np.float32)
        self.critic = np.zeros(self.hidden_dim + 1, dtype=np.float32)
        self.elig_actor = np.zeros_like(self.actor)
        self.elig_critic = np.zeros_like(self.critic)
        self.model = np.zeros((feature_dim, feature_dim + len(self.ACTIONS) + 1), dtype=np.float32)
        self.model_input = np.zeros(feature_dim + len(self.ACTIONS) + 1, dtype=np.float32)
        self.predicted = np.zeros(feature_dim, dtype=np.float32)
        self.error_fast = 0.0
        self.error_slow = 0.0
        self.last_features = np.zeros(feature_dim, dtype=np.float32)
        self.last_hidden = np.zeros(self.hidden_dim + 1, dtype=np.float32)
        self.last_action = np.zeros(len(self.ACTIONS), dtype=np.float32)
        self.last_mean = np.zeros(len(self.ACTIONS), dtype=np.float32)
        self.action_noise = np.zeros(len(self.ACTIONS), dtype=np.float32)
        self.last_context = self.context.copy()
        self.last_value = 0.0
        self.last_energy = None
        self.last_memory_time = -10.0
        self.memory = PersonalMemory(feature_dim, self.context_dim, self.genome.memory_capacity)
        self.last_metrics = {"prediction_error": 0.0, "learning_progress": 0.0, "novelty": 1.0,
                             "advantage": 0.0, "homeostatic_outcome": 0.0, "recalled_value": 0.0}

    def step(self, features, physiology: dict, outcome: dict | None = None, dt=0.05):
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (self.feature_dim,) or not np.isfinite(features).all() or not 0 < dt <= 0.2:
            raise ValueError("Invalid cognitive input")
        features = np.clip(features, -2, 2)
        if set(physiology) - {"energy", "gut", "fatigue", "speed", "angular_velocity", "support"}:
            raise ValueError("Cognitive physiology contains nonlocal state")
        outcome = outcome or {}
        energy = float(physiology.get("energy", 0.7))
        fatigue = float(physiology.get("fatigue", 0.1))
        support = float(physiology.get("support", 1.0))
        body = np.array([energy, physiology.get("gut", 0), fatigue,
                         physiology.get("speed", 0), physiology.get("angular_velocity", 0), support], dtype=np.float32)
        if not np.isfinite(body).all():
            raise ValueError("Nonfinite physiology")
        self.time += dt
        self.context += np.float32(min(1, dt / 0.45)) * (np.tanh(self.context_in @ np.concatenate((features, self.last_action)) + self.context_recur @ self.context) - self.context)
        recalled, recalled_value, novelty = self.memory.recall(features, self.context)
        hidden = np.concatenate((np.tanh(self.encoder @ np.concatenate((features, recalled, self.context, body))), [1])).astype(np.float32)
        current_value = float(self.critic @ hidden)
        progress = 0.0
        homeostatic = 0.0
        advantage = 0.0
        if self.last_energy is not None:
            delta = features - self.last_features
            model_error = delta - self.predicted
            error = float(np.mean(model_error ** 2))
            self.error_fast += min(1, dt / 1.0) * (error - self.error_fast)
            self.error_slow += min(1, dt / 12.0) * (error - self.error_slow)
            # A stationary random stimulus stays unpredictable but stops earning
            # intrinsic return: reward improvement, not indefinitely raw surprise.
            progress = float(np.clip(self.error_slow - self.error_fast, 0, 0.20))
            old_drive = (0.85 - self.last_energy) ** 2
            new_drive = (0.85 - energy) ** 2
            homeostatic = (old_drive - new_drive) * 12
            # Ingestion is available as interoceptive feedback before digestion.
            nutrition = max(0.0, float(outcome.get("nutrition", 0)))
            homeostatic += nutrition * max(0, 1 - energy) * 3
            outcome_value = homeostatic + self.genome.curiosity_gain * progress * dt - float(outcome.get("effort", 0)) * 0.0002 * dt
            discount = math.exp(-dt / self.genome.discount_seconds)
            advantage = float(np.clip(outcome_value + discount * current_value - self.last_value, -0.3, 0.3))
            decay = math.exp(-dt / self.genome.eligibility_seconds)
            self.elig_critic *= decay
            self.elig_critic += self.last_hidden * np.float32(dt)
            sigma = max(0.04, self.genome.exploration)
            # Score of the unconstrained Gaussian command with a tanh actuator.
            score = (self.last_action - self.last_mean) / (sigma * sigma)
            self.elig_actor *= decay
            self.elig_actor += np.outer(score, self.last_hidden).astype(np.float32) * np.float32(dt)
            if self.learning:
                self.critic += np.float32(self.genome.critic_rate * advantage) * self.elig_critic
                self.actor += np.float32(self.genome.actor_rate * advantage) * self.elig_actor
                self.model += np.float32(self.genome.model_rate) * np.outer(model_error, self.model_input) / max(1, float(self.model_input @ self.model_input))
                np.clip(self.actor, -2, 2, out=self.actor)
                np.clip(self.critic, -3, 3, out=self.critic)
                np.clip(self.model, -2, 2, out=self.model)
            if self.time - self.last_memory_time > 1.0 and (novelty > self.genome.memory_threshold or nutrition > 0):
                self.memory.remember(self.last_features, self.last_context, self.last_action, features, outcome_value, self.time)
                self.last_memory_time = self.time
        mean = self.actor @ hidden
        # Temporally correlated motor exploration can discover sustained actions.
        self.action_noise += np.float32(dt * -1.5) * self.action_noise + np.float32(math.sqrt(dt) * self.genome.exploration) * self.rng.normal(size=len(self.ACTIONS)).astype(np.float32)
        pre_action = mean + self.action_noise
        action = np.tanh(pre_action).astype(np.float32)
        self.model_input = np.concatenate((features, action, [1])).astype(np.float32)
        self.predicted = self.model @ self.model_input
        self.last_features = features.copy()
        self.last_hidden = hidden
        self.last_action = pre_action.astype(np.float32)
        self.last_mean = mean.astype(np.float32)
        self.last_context = self.context.copy()
        self.last_value = current_value
        self.last_energy = energy
        self.last_metrics = {"prediction_error": self.error_fast, "learning_progress": progress,
                             "novelty": novelty, "advantage": advantage, "homeostatic_outcome": homeostatic,
                             "recalled_value": recalled_value}
        return {key: float(value) for key, value in zip(self.ACTIONS, action)}

    def consolidate(self, updates=32):
        """Replay experienced transitions into the personal forward model only.

        Actor updates require online outcomes; this does not fabricate personal
        episodes or train the policy off-policy without correction.
        """
        if not self.learning or not self.memory.records:
            return {"updates": 0}
        for _ in range(min(int(updates), 512)):
            record = self.memory.records[int(self.rng.integers(len(self.memory.records)))]
            f = np.asarray(record["features"], dtype=np.float32)
            a = np.tanh(np.asarray(record["action"], dtype=np.float32))
            x = np.concatenate((f, a, [1])).astype(np.float32)
            target = np.asarray(record["next"], dtype=np.float32) - f
            error = target - self.model @ x
            self.model += np.float32(self.genome.model_rate) * np.outer(error, x) / max(1, float(x @ x))
        return {"updates": min(int(updates), 512), "source": "experienced transitions"}

    def view(self):
        return {"time": self.time, "memory_count": len(self.memory.records), "learning": self.learning,
                "metrics": self.last_metrics.copy(), "action_mean": self.last_mean.tolist(),
                "context": self.context.tolist()}

    def snapshot(self):
        arrays = {key: value.tolist() for key, value in vars(self).items() if isinstance(value, np.ndarray)}
        scalars = {key: copy.deepcopy(value) for key, value in vars(self).items()
                   if key not in arrays and key not in ("genome", "rng", "memory")}
        return {"version": self.VERSION, "genome": asdict(self.genome), "rng": copy.deepcopy(self.rng.bit_generator.state),
                "arrays": arrays, "state": scalars, "memory": self.memory.snapshot()}

    @classmethod
    def restore(cls, value):
        if value.get("version") != cls.VERSION:
            raise ValueError("Unsupported cognitive organ checkpoint")
        instance = cls(value["state"]["feature_dim"], genome=value["genome"])
        for key, array in value["arrays"].items():
            restored = np.asarray(array, dtype=np.float32)
            if not hasattr(instance, key) or not isinstance(getattr(instance, key), np.ndarray) or restored.shape != getattr(instance, key).shape or not np.isfinite(restored).all():
                raise ValueError(f"Invalid personal array {key}")
            setattr(instance, key, restored)
        for key, scalar in value["state"].items():
            if key not in vars(instance) or isinstance(getattr(instance, key), np.ndarray):
                raise ValueError(f"Unexpected cognitive scalar {key}")
            setattr(instance, key, copy.deepcopy(scalar))
        instance.memory = PersonalMemory.restore(value["memory"])
        instance.rng.bit_generator.state = copy.deepcopy(value["rng"])
        return instance
