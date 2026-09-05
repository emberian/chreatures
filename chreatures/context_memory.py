"""Bounded action-conditioned relational memory over experienced vectors.

This optional organ sees only an anonymous sensory feature vector, the action
that was executed, and its outcome.  It learns latent context clones and a
transition graph; it has no simulator-coordinate, object-identity, or goal API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Any

import numpy as np


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class ContextMemoryConfig:
    feature_dim: int
    action_dim: int
    outcome_dim: int
    context_capacity: int = 160
    transition_capacity: int = 640
    observation_bandwidth: float = 0.70
    action_bandwidth: float = 0.28
    new_context_distance: float = 0.95
    clone_min_visits: int = 3
    posterior_floor: float = 0.025

    def validate(self) -> None:
        if not 2 <= self.feature_dim <= 4096:
            raise ValueError("feature_dim must be in 2..4096")
        if not 1 <= self.action_dim <= 64 or not 1 <= self.outcome_dim <= 64:
            raise ValueError("action/outcome dimensions are outside bounds")
        if not 2 <= self.context_capacity <= 4096:
            raise ValueError("context_capacity must be in 2..4096")
        if not self.context_capacity <= self.transition_capacity <= 32768:
            raise ValueError("transition_capacity is outside bounds")
        for value, name in (
            (self.observation_bandwidth, "observation_bandwidth"),
            (self.action_bandwidth, "action_bandwidth"),
            (self.new_context_distance, "new_context_distance"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.clone_min_visits < 1 or not 0 <= self.posterior_floor <= 1:
            raise ValueError("invalid clone_min_visits or posterior_floor")


class RelationalContextMemory:
    """Online latent transition map with rapid binding and clone states.

    ``begin`` starts inference for one trajectory. ``predict`` queries an
    action from the current posterior. ``step`` observes the resulting sensory
    and outcome vectors, optionally updating the bounded map.
    """

    VERSION = 1

    def __init__(self, config: ContextMemoryConfig):
        config.validate()
        self.config = config
        c, e = config.context_capacity, config.transition_capacity
        f, a, o = config.feature_dim, config.action_dim, config.outcome_dim
        self.context_count = 0
        self.transition_count = 0
        self.clock = 0
        self.observation_mean = np.zeros((c, f), dtype=np.float32)
        self.observation_m2 = np.zeros((c, f), dtype=np.float32)
        self.context_visits = np.zeros(c, dtype=np.int32)
        self.context_last_seen = np.zeros(c, dtype=np.int64)
        self.edge_source = np.full(e, -1, dtype=np.int32)
        self.edge_destination = np.full(e, -1, dtype=np.int32)
        self.edge_action_mean = np.zeros((e, a), dtype=np.float32)
        self.edge_action_m2 = np.zeros((e, a), dtype=np.float32)
        self.edge_delta_mean = np.zeros((e, f), dtype=np.float32)
        self.edge_delta_m2 = np.zeros((e, f), dtype=np.float32)
        self.edge_outcome_mean = np.zeros((e, o), dtype=np.float32)
        self.edge_outcome_m2 = np.zeros((e, o), dtype=np.float32)
        self.edge_count = np.zeros(e, dtype=np.int32)
        self.edge_last_seen = np.zeros(e, dtype=np.int64)
        self.posterior = np.zeros(c, dtype=np.float32)
        self.current_observation: np.ndarray | None = None

    def reset(self) -> None:
        self.posterior.fill(0.0)
        self.current_observation = None

    def _observation_distance(self, observation: np.ndarray) -> np.ndarray:
        if not self.context_count:
            return np.empty(0, dtype=np.float32)
        difference = self.observation_mean[: self.context_count] - observation
        return np.sqrt(np.mean(difference * difference, axis=1)).astype(np.float32)

    def _emission(self, observation: np.ndarray) -> np.ndarray:
        distance = self._observation_distance(observation)
        return np.exp(
            -0.5 * (distance / self.config.observation_bandwidth) ** 2
        ).astype(np.float32)

    def _allocate_context(self, observation: np.ndarray) -> int:
        if self.context_count < self.config.context_capacity:
            index = self.context_count
            self.context_count += 1
        else:
            active = int(np.argmax(self.posterior)) if self.posterior.any() else -1
            score = self.context_visits.astype(np.float64) + 0.002 / np.maximum(
                1.0, self.clock - self.context_last_seen
            )
            if active >= 0:
                score[active] = np.inf
            index = int(np.argmin(score))
            invalid = (self.edge_source[: self.transition_count] == index) | (
                self.edge_destination[: self.transition_count] == index
            )
            self._remove_edges(np.flatnonzero(invalid))
        self.observation_mean[index] = observation
        self.observation_m2[index].fill(0.0)
        self.context_visits[index] = 1
        self.context_last_seen[index] = self.clock
        return index

    def _remove_edges(self, indices: np.ndarray) -> None:
        for index in sorted((int(value) for value in indices), reverse=True):
            last = self.transition_count - 1
            if index != last:
                for array in self._edge_arrays():
                    array[index] = array[last]
            self.transition_count -= 1

    def _edge_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.edge_source, self.edge_destination, self.edge_action_mean,
            self.edge_action_m2, self.edge_delta_mean, self.edge_delta_m2,
            self.edge_outcome_mean, self.edge_outcome_m2, self.edge_count,
            self.edge_last_seen,
        )

    def begin(self, observation: Any, *, learn: bool = True) -> dict[str, Any]:
        value = _vector(observation, self.config.feature_dim, "observation")
        self.posterior.fill(0.0)
        if not self.context_count:
            if not learn:
                self.current_observation = value.copy()
                return self.state()
            index = self._allocate_context(value)
            self.posterior[index] = 1.0
        else:
            emission = self._emission(value)
            distance = self._observation_distance(value)
            if learn and float(distance.min()) > self.config.new_context_distance:
                index = self._allocate_context(value)
                self.posterior[index] = 1.0
            else:
                weights = emission * np.maximum(
                    self.context_visits[: self.context_count].astype(np.float32), 1.0
                ) ** 0.25
                self.posterior[: self.context_count] = weights / max(float(weights.sum()), 1e-9)
        self.current_observation = value.copy()
        return self.state()

    def _action_distance(self, action: np.ndarray) -> np.ndarray:
        if not self.transition_count:
            return np.empty(0, dtype=np.float32)
        difference = self.edge_action_mean[: self.transition_count] - action
        return np.sqrt(np.mean(difference * difference, axis=1)).astype(np.float32)

    def _edge_weights(self, action: np.ndarray) -> np.ndarray:
        action_distance = self._action_distance(action)
        if not len(action_distance):
            return action_distance
        source = self.edge_source[: self.transition_count]
        source_weight = self.posterior[source]
        if self.current_observation is not None and self.context_count:
            # The posterior supplies path identity. A small emission route lets
            # the graph reuse a learned transition when the exact clone has not
            # yet seen this action, without erasing contextual preference.
            emission = self._emission(self.current_observation)
            emission /= max(float(emission.sum()), 1e-9)
            source_weight = 0.82 * source_weight + 0.18 * emission[source]
        action_weight = np.exp(
            -0.5 * (action_distance / self.config.action_bandwidth) ** 2
        )
        reliability = 1.0 - 1.0 / (self.edge_count[: self.transition_count] + 1.0)
        return (source_weight * action_weight * reliability).astype(np.float32)

    def predict(self, action: Any) -> dict[str, Any]:
        action_value = _vector(action, self.config.action_dim, "action")
        if self.current_observation is None:
            raise RuntimeError("begin(observation) must precede prediction")
        weights = self._edge_weights(action_value)
        support = float(weights.sum())
        if support <= 1e-10:
            next_observation = self.current_observation.copy()
            outcome = np.zeros(self.config.outcome_dim, dtype=np.float32)
            transition_variance = 1.0
            effective = 0.0
        else:
            normalized = weights / support
            used = np.flatnonzero(normalized > 1e-8)
            edge_weight = normalized[used]
            delta = self.edge_delta_mean[used]
            outcome_values = self.edge_outcome_mean[used]
            predicted_delta = edge_weight @ delta
            destination_mean = self.observation_mean[
                self.edge_destination[used]
            ]
            mapped_next = edge_weight @ destination_mean
            # Delta preserves fine local continuity; the destination prototype
            # supplies the reusable relational map when two paths share a view.
            next_observation = 0.55 * (
                self.current_observation + predicted_delta
            ) + 0.45 * mapped_next
            outcome = edge_weight @ outcome_values
            residual = delta - predicted_delta
            between = float(np.sum(edge_weight[:, None] * residual * residual) / self.config.feature_dim)
            within = 0.0
            for weight, edge in zip(edge_weight, used, strict=True):
                count = max(1, int(self.edge_count[edge]) - 1)
                within += float(weight) * float(np.mean(self.edge_delta_m2[edge] / count))
            transition_variance = between + within
            effective = float(np.sum(edge_weight * self.edge_count[used]))
        active = self.posterior[: self.context_count]
        nonzero = active[active > 1e-12]
        entropy = 0.0 if not len(nonzero) else float(
            -np.sum(nonzero * np.log(nonzero)) / max(math.log(max(2, self.context_count)), 1e-9)
        )
        action_novelty = 1.0
        if self.transition_count:
            candidate = self._action_distance(action_value)
            source_mask = self.posterior[self.edge_source[: self.transition_count]] > 1e-4
            if source_mask.any():
                action_novelty = float(np.clip(candidate[source_mask].min() / self.config.action_bandwidth, 0, 1))
        observation_novelty = 1.0
        if self.context_count:
            observation_novelty = float(np.clip(
                self._observation_distance(self.current_observation).min()
                / self.config.observation_bandwidth, 0.0, 1.0
            ))
        uncertainty = float(np.clip(
            0.26 / (1.0 + effective) + 0.22 * action_novelty
            + 0.22 * min(1.0, math.sqrt(max(0.0, transition_variance)))
            + 0.14 * entropy + 0.16 * observation_novelty,
            0.0, 1.0,
        ))
        return {
            "next_observation": next_observation.astype(np.float32).tolist(),
            "outcome": outcome.astype(np.float32).tolist(),
            "uncertainty": uncertainty,
            "confidence": 1.0 - uncertainty,
            "support": effective,
            "context_entropy": entropy,
            "basis": "experienced action-conditioned latent transitions",
        }

    def _matching_outgoing(self, source: int, action: np.ndarray) -> np.ndarray:
        if not self.transition_count:
            return np.empty(0, dtype=np.int64)
        distance = self._action_distance(action)
        return np.flatnonzero(
            (self.edge_source[: self.transition_count] == source)
            & (distance <= self.config.action_bandwidth * 1.8)
        )

    def _choose_destination(
        self, source: int, action: np.ndarray, observation: np.ndarray, learn: bool
    ) -> int:
        distances = self._observation_distance(observation)
        outgoing = self._matching_outgoing(source, action)
        if len(outgoing):
            action_distance = self._action_distance(action)[outgoing]
            destination = self.edge_destination[outgoing]
            score = distances[destination] + 0.35 * action_distance
            best = int(outgoing[int(np.argmin(score))])
            if float(distances[self.edge_destination[best]]) <= self.config.new_context_distance:
                return int(self.edge_destination[best])
        if not self.context_count or float(distances.min()) > self.config.new_context_distance:
            return self._allocate_context(observation) if learn else int(np.argmin(distances))
        candidate = int(np.argmin(distances))
        if not learn or candidate == source or self.context_visits[candidate] < self.config.clone_min_visits:
            return candidate
        incoming = np.flatnonzero(self.edge_destination[: self.transition_count] == candidate)
        established_route = any(
            int(self.edge_source[edge]) == source
            and float(np.sqrt(np.mean((self.edge_action_mean[edge] - action) ** 2)))
                <= self.config.action_bandwidth * 1.8
            for edge in incoming
        )
        # A familiar-looking emission reached by a novel route gets a clone.
        # Subsequent transitions can then give the clones different successors.
        if len(incoming) and not established_route:
            return self._allocate_context(observation)
        return candidate

    @staticmethod
    def _update_moments(mean: np.ndarray, m2: np.ndarray, count: int, value: np.ndarray) -> None:
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)

    def _allocate_edge(self, source: int, destination: int, action: np.ndarray) -> int:
        if self.transition_count < self.config.transition_capacity:
            index = self.transition_count
            self.transition_count += 1
        else:
            score = self.edge_count[: self.transition_count].astype(np.float64) + 0.002 / np.maximum(
                1.0, self.clock - self.edge_last_seen[: self.transition_count]
            )
            index = int(np.argmin(score))
        self.edge_source[index] = source
        self.edge_destination[index] = destination
        self.edge_action_mean[index] = action
        self.edge_action_m2[index].fill(0.0)
        self.edge_delta_mean[index].fill(0.0)
        self.edge_delta_m2[index].fill(0.0)
        self.edge_outcome_mean[index].fill(0.0)
        self.edge_outcome_m2[index].fill(0.0)
        self.edge_count[index] = 0
        self.edge_last_seen[index] = self.clock
        return index

    def _find_edge(self, source: int, destination: int, action: np.ndarray) -> int:
        if not self.transition_count:
            return -1
        distance = self._action_distance(action)
        candidates = np.flatnonzero(
            (self.edge_source[: self.transition_count] == source)
            & (self.edge_destination[: self.transition_count] == destination)
            & (distance <= self.config.action_bandwidth * 1.8)
        )
        return -1 if not len(candidates) else int(candidates[np.argmin(distance[candidates])])

    def step(
        self,
        action: Any,
        next_observation: Any,
        outcome: Any,
        *,
        learn: bool = True,
    ) -> dict[str, Any]:
        action_value = _vector(action, self.config.action_dim, "action")
        next_value = _vector(next_observation, self.config.feature_dim, "next observation")
        outcome_value = _vector(outcome, self.config.outcome_dim, "outcome")
        if self.current_observation is None:
            raise RuntimeError("begin(observation) must precede step")
        prediction = self.predict(action_value)
        self.clock += 1
        if not self.context_count:
            if not learn:
                self.current_observation = next_value.copy()
                return prediction
            source = self._allocate_context(self.current_observation)
            self.posterior[source] = 1.0
        source = int(np.argmax(self.posterior[: self.context_count]))
        destination = self._choose_destination(source, action_value, next_value, learn)

        if learn:
            self.context_visits[source] += 1
            source_count = int(self.context_visits[source])
            self._update_moments(
                self.observation_mean[source], self.observation_m2[source],
                source_count, self.current_observation,
            )
            self.context_last_seen[source] = self.clock
            edge = self._find_edge(source, destination, action_value)
            if edge < 0:
                edge = self._allocate_edge(source, destination, action_value)
            self.edge_count[edge] += 1
            count = int(self.edge_count[edge])
            self._update_moments(
                self.edge_action_mean[edge], self.edge_action_m2[edge], count, action_value
            )
            self._update_moments(
                self.edge_delta_mean[edge], self.edge_delta_m2[edge], count,
                next_value - self.current_observation,
            )
            self._update_moments(
                self.edge_outcome_mean[edge], self.edge_outcome_m2[edge], count, outcome_value
            )
            self.edge_last_seen[edge] = self.clock
            self.context_visits[destination] += 1
            destination_count = int(self.context_visits[destination])
            self._update_moments(
                self.observation_mean[destination], self.observation_m2[destination],
                destination_count, next_value,
            )
            self.context_last_seen[destination] = self.clock
            self.posterior.fill(0.0)
            self.posterior[destination] = 1.0
        else:
            emission = self._emission(next_value)
            prior = np.zeros(self.context_count, dtype=np.float32)
            weights = self._edge_weights(action_value)
            for edge, weight in enumerate(weights):
                prior[self.edge_destination[edge]] += weight
            prior /= max(float(prior.sum()), 1e-9)
            combined = emission * (prior + self.config.posterior_floor)
            self.posterior.fill(0.0)
            self.posterior[: self.context_count] = combined / max(float(combined.sum()), 1e-9)
        self.current_observation = next_value.copy()
        prediction.update({
            "context": destination,
            "contexts": self.context_count,
            "transitions": self.transition_count,
        })
        return prediction

    def state(self) -> dict[str, Any]:
        active = self.posterior[: self.context_count]
        return {
            "contexts": self.context_count,
            "transitions": self.transition_count,
            "active_context": None if not len(active) or not active.any() else int(np.argmax(active)),
            "posterior": active.tolist(),
            "clock": self.clock,
        }

    def snapshot(self) -> dict[str, Any]:
        arrays = {
            name: getattr(self, name).tolist()
            for name in (
                "observation_mean", "observation_m2", "context_visits", "context_last_seen",
                "edge_source", "edge_destination", "edge_action_mean", "edge_action_m2",
                "edge_delta_mean", "edge_delta_m2", "edge_outcome_mean", "edge_outcome_m2",
                "edge_count", "edge_last_seen", "posterior",
            )
        }
        return {
            "version": self.VERSION,
            "config": asdict(self.config),
            "context_count": self.context_count,
            "transition_count": self.transition_count,
            "clock": self.clock,
            "current_observation": None if self.current_observation is None else self.current_observation.tolist(),
            "arrays": arrays,
        }

    @classmethod
    def restore(cls, value: Any) -> "RelationalContextMemory":
        if not isinstance(value, dict) or value.get("version") != cls.VERSION:
            raise ValueError("unsupported relational context checkpoint")
        instance = cls(ContextMemoryConfig(**value["config"]))
        instance.context_count = int(value["context_count"])
        instance.transition_count = int(value["transition_count"])
        instance.clock = int(value["clock"])
        if not 0 <= instance.context_count <= instance.config.context_capacity:
            raise ValueError("invalid context count")
        if not 0 <= instance.transition_count <= instance.config.transition_capacity:
            raise ValueError("invalid transition count")
        arrays = value.get("arrays")
        expected = {
            "observation_mean", "observation_m2", "context_visits", "context_last_seen",
            "edge_source", "edge_destination", "edge_action_mean", "edge_action_m2",
            "edge_delta_mean", "edge_delta_m2", "edge_outcome_mean", "edge_outcome_m2",
            "edge_count", "edge_last_seen", "posterior",
        }
        if not isinstance(arrays, dict) or set(arrays) != expected:
            raise ValueError("relational context arrays differ")
        for name, raw in arrays.items():
            target = getattr(instance, name)
            restored = np.asarray(raw, dtype=target.dtype)
            if restored.shape != target.shape or not np.isfinite(restored).all():
                raise ValueError(f"invalid context array: {name}")
            target[:] = restored
        current = value.get("current_observation")
        instance.current_observation = None if current is None else _vector(
            current, instance.config.feature_dim, "current observation"
        )
        if instance.clock < 0 or np.any(instance.context_visits < 0) or np.any(instance.edge_count < 0):
            raise ValueError("invalid relational counters")
        if instance.transition_count:
            source = instance.edge_source[: instance.transition_count]
            destination = instance.edge_destination[: instance.transition_count]
            if np.any(source < 0) or np.any(source >= instance.context_count) or np.any(destination < 0) or np.any(destination >= instance.context_count):
                raise ValueError("transition refers to an absent context")
        total = float(instance.posterior[: instance.context_count].sum())
        if np.any(instance.posterior < 0) or (total > 0 and not math.isclose(total, 1.0, abs_tol=2e-5)):
            raise ValueError("invalid context posterior")
        return instance


__all__ = ["ContextMemoryConfig", "RelationalContextMemory"]
