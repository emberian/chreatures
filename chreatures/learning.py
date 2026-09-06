"""Versioned predictive PPO for embodied, full-circuit developmental learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


ACTIONS = (
    "thrust", "yaw", "gaze_pitch", "grip",
    "signal_low", "signal_mid", "signal_high", "posture",
)


@dataclass(frozen=True)
class PredictivePPOConfig:
    feature_dim: int = 384
    physiology_dim: int = 6
    context_dim: int = 32
    projection_dim: int = 64
    hidden_dim: int = 128
    macro_steps: int = 5
    gamma_seconds: float = 8.0
    gae_seconds: float = 2.0
    clip_ratio: float = 0.18
    learning_rate: float = 3e-4
    predictor_rate: float = 0.35
    value_rate: float = 0.55
    entropy_rate: float = 0.002
    curiosity_rate: float = 0.08
    epochs: int = 4
    minibatch_size: int = 512
    max_grad_norm: float = 0.8
    seed: int = 20260905
    std_profile: str = "global-v1"
    context_profile: str = "reservoir-v1"
    sequence_length: int = 32


class RunningMoments:
    """Mergeable population moments used only to standardize neural readouts."""

    def __init__(self, dimension: int) -> None:
        self.count = 1e-4
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.ones(dimension, dtype=np.float64) * 1e-4

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != self.mean.shape or not np.isfinite(values).all():
            raise ValueError("running-moment observations have the wrong shape")
        count = len(values)
        if not count:
            return
        mean = values.mean(axis=0)
        centered = values - mean
        m2 = np.sum(centered * centered, axis=0)
        delta = mean - self.mean
        total = self.count + count
        self.mean += delta * count / total
        self.m2 += m2 + delta * delta * self.count * count / total
        self.count = total

    def normalize(self, values: np.ndarray) -> np.ndarray:
        variance = self.m2 / max(self.count, 1.0)
        return np.clip(
            (np.asarray(values, dtype=np.float64) - self.mean)
            / np.sqrt(np.maximum(variance, 1e-5)),
            -5.0,
            5.0,
        ).astype(np.float32)

    def snapshot(self) -> dict[str, Any]:
        return {"count": self.count, "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    @classmethod
    def restore(cls, value: dict[str, Any]) -> "RunningMoments":
        mean = np.asarray(value["mean"], dtype=np.float64)
        instance = cls(len(mean))
        instance.count = float(value["count"])
        instance.mean = mean
        instance.m2 = np.asarray(value["m2"], dtype=np.float64)
        if instance.count <= 0 or instance.m2.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(instance.m2).all():
            raise ValueError("invalid running moments")
        return instance


class PredictiveActorCritic(nn.Module):
    """Compact shared genome with private external recurrent context."""

    def __init__(self, config: PredictivePPOConfig) -> None:
        super().__init__()
        self.config = config
        if config.std_profile not in {"global-v1", "state-conditioned-v2"}:
            raise ValueError("std_profile must be global-v1 or state-conditioned-v2")
        if config.context_profile not in {"reservoir-v1", "gated-v1"}:
            raise ValueError("unknown working-context architecture")
        if config.sequence_length < 2:
            raise ValueError("sequence_length must be at least two decisions")
        torch.manual_seed(config.seed)
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(config.hidden_dim + config.physiology_dim + config.context_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(config.hidden_dim, len(ACTIONS))
        self.value = nn.Linear(config.hidden_dim, 1)
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + len(ACTIONS), config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.projection_dim),
        )
        self.log_std = nn.Parameter(torch.tensor(
            [math.log(v) for v in (0.62, 0.54, 0.22, 0.28, 0.16, 0.16, 0.16, 0.28)],
            dtype=torch.float32,
        ))
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 91)
        projection = torch.randn(
            config.projection_dim, config.feature_dim, generator=generator
        ) / math.sqrt(config.feature_dim)
        context_feature = torch.randn(
            config.context_dim, config.projection_dim, generator=generator
        ) / math.sqrt(config.projection_dim)
        context_action = torch.randn(
            config.context_dim, len(ACTIONS), generator=generator
        ) / math.sqrt(len(ACTIONS))
        context_recur = torch.randn(
            config.context_dim, config.context_dim, generator=generator
        ) / math.sqrt(config.context_dim)
        radius = torch.linalg.matrix_norm(context_recur, ord=2)
        context_recur *= 0.72 / radius.clamp_min(1e-6)
        self.register_buffer("projection", projection)
        for name, value in (
            ("context_feature", context_feature), ("context_action", context_action),
            ("context_recur", context_recur),
        ):
            if config.context_profile == "gated-v1":
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)
        if config.context_profile == "gated-v1":
            self.context_gate_feature = nn.Parameter(torch.zeros_like(context_feature))
            self.context_gate_action = nn.Parameter(torch.zeros_like(context_action))
            self.context_gate_recur = nn.Parameter(torch.zeros_like(context_recur))
            # Initial retention spans several physical timescales; experience
            # learns what to write and retain. No state or behavior labels.
            timescales = torch.logspace(math.log10(0.5), math.log10(32.0), config.context_dim)
            fraction = -torch.expm1(-float(config.macro_steps * 0.05) / timescales)
            self.context_gate_bias = nn.Parameter(torch.logit(fraction))
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)
        self.std_offset: nn.Linear | None = None
        if config.std_profile == "state-conditioned-v2":
            rng_state = torch.get_rng_state()
            self.std_offset = nn.Linear(config.hidden_dim, len(ACTIONS))
            nn.init.zeros_(self.std_offset.weight)
            nn.init.zeros_(self.std_offset.bias)
            torch.set_rng_state(rng_state)

    def forward(
        self, features: torch.Tensor, physiology: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.feature_encoder(features)
        hidden = self.trunk(torch.cat((encoded, physiology, context), dim=-1))
        return self.policy_mean(hidden), self.value(hidden).squeeze(-1), hidden

    def distribution(self, mean: torch.Tensor, hidden: torch.Tensor | None = None) -> Normal:
        effective = self.log_std
        if self.std_offset is not None:
            if hidden is None or hidden.shape[:-1] != mean.shape[:-1]:
                raise ValueError("state-conditioned distribution requires matching hidden state")
            effective = effective + 2.0 * torch.tanh(self.std_offset(hidden) / 2.0)
        std = effective.clamp(-3.5, 0.3).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def squashed_log_prob(distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(latent)
        return (
            distribution.log_prob(latent)
            - torch.log(torch.clamp(1 - action.square(), min=1e-6))
        ).sum(dim=-1)

    def projected(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(features @ self.projection.T)

    def next_context(
        self, context: torch.Tensor, next_features: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        projected = self.projected(next_features)
        candidate = torch.tanh(
            projected @ self.context_feature.T
            + action @ self.context_action.T
            + context @ self.context_recur.T
        )
        if self.config.context_profile == "reservoir-v1":
            return candidate
        gate = torch.sigmoid(
            projected @ self.context_gate_feature.T
            + action @ self.context_gate_action.T
            + context @ self.context_gate_recur.T
            + self.context_gate_bias
        )
        return context + gate * (candidate - context)

    def sequence(
        self, features: torch.Tensor, physiology: torch.Tensor,
        initial_context: torch.Tensor, actions: torch.Tensor, done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay causal histories with gradients through working memory.

        The recorded context at a chunk boundary is detached. Within a chunk,
        the next observation enters memory only before the next decision.
        Done resets each resident independently; there is no cross-life credit.
        """
        context = initial_context.detach()
        means, values, hidden_states = [], [], []
        for index in range(len(features)):
            mean, value, hidden = self(features[index], physiology[index], context)
            means.append(mean)
            values.append(value)
            hidden_states.append(hidden)
            if index + 1 < len(features):
                context = self.next_context(context, features[index + 1], actions[index])
                context = context * (~done[index].bool()).unsqueeze(-1)
        return torch.stack(means), torch.stack(values), torch.stack(hidden_states)


class MacroRollout:
    """Resident-major macro transitions retained until one PPO update."""

    FIELDS = (
        "features", "physiology", "context", "latent", "action", "log_prob",
        "value", "reward", "done", "prediction_target",
    )

    def __init__(self) -> None:
        self.rows: dict[str, list[np.ndarray]] = {name: [] for name in self.FIELDS}

    def append(self, **values: np.ndarray) -> None:
        if set(values) != set(self.FIELDS):
            raise ValueError("macro rollout transition fields differ")
        for name in self.FIELDS:
            value = np.asarray(values[name])
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite rollout field {name}")
            self.rows[name].append(value.copy())

    def __len__(self) -> int:
        return len(self.rows["reward"])

    def arrays(self) -> dict[str, np.ndarray]:
        if not len(self):
            raise ValueError("rollout is empty")
        return {name: np.stack(values) for name, values in self.rows.items()}

    def clear(self) -> None:
        for values in self.rows.values():
            values.clear()


class PredictivePPOTrainer:
    """Owns shared learned parameters and private per-resident context state."""

    VERSION = 3

    def __init__(
        self,
        resident_ids: Sequence[str],
        config: PredictivePPOConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        ids = [str(value) for value in resident_ids]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("resident ids must be nonempty and unique")
        self.resident_ids = ids
        self.config = config or PredictivePPOConfig()
        self.device = torch.device(device)
        self.model = PredictiveActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        count = len(ids)
        self.context = np.zeros((count, self.config.context_dim), dtype=np.float32)
        self.moments = RunningMoments(self.config.feature_dim)
        self.error_fast = np.zeros(count, dtype=np.float32)
        self.error_slow = np.zeros(count, dtype=np.float32)
        self.update_count = 0
        self.decision_count = 0
        self.rng = np.random.default_rng(self.config.seed + 311)

    def normalize(self, raw_features: np.ndarray, *, update: bool) -> np.ndarray:
        raw = np.asarray(raw_features, dtype=np.float32)
        expected = (len(self.resident_ids), self.config.feature_dim)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(f"features must have shape {expected}")
        if update:
            self.moments.update(raw)
        return self.moments.normalize(raw)

    @torch.no_grad()
    def act(
        self,
        normalized_features: np.ndarray,
        physiology: np.ndarray,
        *,
        deterministic: bool = False,
        silence_features: bool = False,
    ) -> dict[str, np.ndarray]:
        features = np.asarray(normalized_features, dtype=np.float32)
        physiology = np.asarray(physiology, dtype=np.float32)
        if silence_features:
            features = np.zeros_like(features)
        feature_tensor = torch.as_tensor(features, device=self.device)
        physiology_tensor = torch.as_tensor(physiology, device=self.device)
        context_tensor = torch.as_tensor(self.context, device=self.device)
        mean, value, hidden = self.model(feature_tensor, physiology_tensor, context_tensor)
        distribution = self.model.distribution(mean, hidden)
        latent = mean if deterministic else distribution.sample()
        action = torch.tanh(latent)
        log_prob = self.model.squashed_log_prob(distribution, latent)
        prediction = self.model.predictor(torch.cat((hidden, action), dim=-1))
        self.decision_count += 1
        return {
            "features": features.copy(),
            "physiology": physiology.copy(),
            "context": self.context.copy(),
            "latent": latent.cpu().numpy(),
            "action": action.cpu().numpy(),
            "log_prob": log_prob.cpu().numpy(),
            "value": value.cpu().numpy(),
            "prediction": prediction.cpu().numpy(),
        }

    @torch.no_grad()
    def finish_transition(
        self,
        previous: dict[str, np.ndarray],
        next_normalized_features: np.ndarray,
        accumulated_reward: np.ndarray,
        done: np.ndarray,
        macro_dt: float,
    ) -> dict[str, np.ndarray]:
        next_features = np.asarray(next_normalized_features, dtype=np.float32)
        next_tensor = torch.as_tensor(next_features, device=self.device)
        previous_tensor = torch.as_tensor(previous["features"], device=self.device)
        target = (
            self.model.projected(next_tensor) - self.model.projected(previous_tensor)
        ).cpu().numpy()
        prediction_error = np.mean((target - previous["prediction"]) ** 2, axis=1)
        fast_alpha = min(1.0, macro_dt / 1.0)
        slow_alpha = min(1.0, macro_dt / 12.0)
        self.error_fast += fast_alpha * (prediction_error - self.error_fast)
        self.error_slow += slow_alpha * (prediction_error - self.error_slow)
        learning_progress = np.clip(self.error_slow - self.error_fast, 0, 0.2)
        reward = np.asarray(accumulated_reward, dtype=np.float32)
        reward = reward + np.float32(
            self.config.curiosity_rate * macro_dt
        ) * learning_progress
        context_tensor = torch.as_tensor(self.context, device=self.device)
        action_tensor = torch.as_tensor(previous["action"], device=self.device)
        self.context = self.model.next_context(
            context_tensor, next_tensor, action_tensor
        ).cpu().numpy()
        self.context[np.asarray(done, dtype=bool)] = 0
        return {
            "reward": reward,
            "prediction_target": target.astype(np.float32),
            "prediction_error": prediction_error.astype(np.float32),
            "learning_progress": learning_progress.astype(np.float32),
        }

    @torch.no_grad()
    def bootstrap_value(
        self, normalized_features: np.ndarray, physiology: np.ndarray
    ) -> np.ndarray:
        features = torch.as_tensor(normalized_features, device=self.device)
        body = torch.as_tensor(physiology, device=self.device)
        context = torch.as_tensor(self.context, device=self.device)
        return self.model(features, body, context)[1].cpu().numpy()

    def _optimization_batches(
        self, data: dict[str, np.ndarray], advantages: np.ndarray, returns: np.ndarray,
    ):
        """Keep time and resident identity intact while learning memory."""
        steps, residents = data["reward"].shape
        if self.config.context_profile == "reservoir-v1":
            flat = {name: value.reshape((-1, *value.shape[2:])) for name, value in data.items()}
            order = self.rng.permutation(steps * residents)
            for start in range(0, len(order), self.config.minibatch_size):
                indices = order[start:start + self.config.minibatch_size]
                tensors = {name: torch.as_tensor(value[indices], device=self.device)
                           for name, value in flat.items()}
                outputs = self.model(tensors["features"], tensors["physiology"], tensors["context"])
                yield tensors, outputs, (
                    torch.as_tensor(advantages.reshape(-1)[indices], device=self.device),
                    torch.as_tensor(returns.reshape(-1)[indices], device=self.device),
                )
            return
        length = min(self.config.sequence_length, steps)
        chunks = [(resident, start) for resident in range(residents)
                  for start in range(0, steps, length)]
        order = self.rng.permutation(len(chunks))
        batch_chunks = max(1, self.config.minibatch_size // length)
        for offset in range(0, len(chunks), batch_chunks):
            selected = [chunks[index] for index in order[offset:offset + batch_chunks]]
            resident_indices = np.asarray([item[0] for item in selected])
            starts = np.asarray([item[1] for item in selected])
            time_indices = starts[None, :] + np.arange(length)[:, None]
            valid = time_indices < steps
            clipped = np.minimum(time_indices, steps - 1)
            tensors = {name: torch.as_tensor(value[clipped, resident_indices], device=self.device)
                       for name, value in data.items()}
            initial = torch.as_tensor(data["context"][starts, resident_indices], device=self.device)
            outputs = self.model.sequence(
                tensors["features"], tensors["physiology"], initial,
                tensors["action"], tensors["done"],
            )
            mask = torch.as_tensor(valid, device=self.device)
            yield {name: value[mask] for name, value in tensors.items()}, tuple(
                value[mask] for value in outputs
            ), (
                torch.as_tensor(advantages[clipped, resident_indices], device=self.device)[mask],
                torch.as_tensor(returns[clipped, resident_indices], device=self.device)[mask],
            )

    def update(self, rollout: MacroRollout, bootstrap_value: np.ndarray, macro_dt: float) -> dict[str, float]:
        data = rollout.arrays()
        rewards = data["reward"].astype(np.float32)
        values = data["value"].astype(np.float32)
        dones = data["done"].astype(np.float32)
        last = np.asarray(bootstrap_value, dtype=np.float32)
        if rewards.shape != values.shape or last.shape != rewards.shape[1:]:
            raise ValueError("rollout value shapes differ")
        gamma = math.exp(-macro_dt / self.config.gamma_seconds)
        lam = math.exp(-macro_dt / self.config.gae_seconds)
        advantages = np.zeros_like(rewards)
        gae = np.zeros_like(last)
        next_value = last
        for index in range(len(rewards) - 1, -1, -1):
            continuation = 1.0 - dones[index]
            delta = rewards[index] + gamma * next_value * continuation - values[index]
            gae = delta + gamma * lam * continuation * gae
            advantages[index] = gae
            next_value = values[index]
        returns = advantages + values
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / max(float(flat_advantages.std()), 1e-6)
        count = len(flat_advantages)
        normalized_advantages = flat_advantages.reshape(advantages.shape)
        metrics = {name: [] for name in (
            "policy_loss", "value_loss", "prediction_loss", "entropy",
            "approx_kl", "clip_fraction",
        )}
        for _ in range(self.config.epochs):
            for tensors, (mean, value, hidden), (advantage, return_value) in self._optimization_batches(
                data, normalized_advantages, returns
            ):
                latent, action = tensors["latent"], tensors["action"]
                old_log_prob, target = tensors["log_prob"], tensors["prediction_target"]
                distribution = self.model.distribution(mean, hidden)
                log_prob = self.model.squashed_log_prob(distribution, latent)
                ratio = torch.exp(log_prob - old_log_prob)
                clipped = torch.clamp(
                    ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                )
                policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                value_loss = 0.5 * (value - return_value).square().mean()
                predicted = self.model.predictor(torch.cat((hidden, action), dim=-1))
                prediction_loss = 0.5 * (predicted - target).square().mean()
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = (
                    policy_loss
                    + self.config.value_rate * value_loss
                    + self.config.predictor_rate * prediction_loss
                    - self.config.entropy_rate * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    metrics["policy_loss"].append(float(policy_loss))
                    metrics["value_loss"].append(float(value_loss))
                    metrics["prediction_loss"].append(float(prediction_loss))
                    metrics["entropy"].append(float(entropy))
                    metrics["approx_kl"].append(float((old_log_prob - log_prob).mean()))
                    metrics["clip_fraction"].append(float((torch.abs(ratio - 1) > self.config.clip_ratio).float().mean()))
        self.update_count += 1
        rollout.clear()
        output = {name: float(np.mean(values)) for name, values in metrics.items()}
        output.update({
            "reward_mean": float(rewards.mean()),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "explained_variance": float(
                1 - np.var(flat_returns - values.reshape(-1))
                / max(np.var(flat_returns), 1e-8)
            ),
            "samples": float(count),
            "update": float(self.update_count),
        })
        return output

    def reset_private_state(self) -> None:
        self.context.fill(0)
        self.error_fast.fill(0)
        self.error_slow.fill(0)

    def snapshot(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "version": self._genome_version(),
            "architecture": self.config.std_profile,
            "config": asdict(self.config),
            "resident_ids": self.resident_ids,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "context": self.context,
            "moments": self.moments.snapshot(),
            "error_fast": self.error_fast,
            "error_slow": self.error_slow,
            "update_count": self.update_count,
            "decision_count": self.decision_count,
            "numpy_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "device_rng": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda" else None
            ),
            "extra": extra or {},
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(value, temporary)
        temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}

    @classmethod
    def restore(
        cls, path: str | Path, *, device: str | torch.device = "cpu",
        expected_sha256: str | None = None,
    ) -> tuple["PredictivePPOTrainer", dict[str, Any]]:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("learner checkpoint checksum differs")
        value = torch.load(path, map_location=device, weights_only=False)
        if value.get("version") not in {1, 2, cls.VERSION}:
            raise ValueError("unsupported learner checkpoint")
        restored_config = PredictivePPOConfig(**value["config"])
        expected_version = (
            3 if restored_config.context_profile == "gated-v1" else
            2 if restored_config.std_profile == "state-conditioned-v2" else 1
        )
        if value["version"] != expected_version:
            raise ValueError("learner checkpoint version differs from context architecture")
        architecture = value.get("architecture", restored_config.std_profile)
        if architecture != restored_config.std_profile:
            raise ValueError("learner checkpoint architecture differs from config")
        instance = cls(value["resident_ids"], restored_config, device=device)
        instance.model.load_state_dict(value["model"], strict=True)
        instance.optimizer.load_state_dict(value["optimizer"])
        instance.context = np.asarray(value["context"], dtype=np.float32)
        instance.moments = RunningMoments.restore(value["moments"])
        instance.error_fast = np.asarray(value["error_fast"], dtype=np.float32)
        instance.error_slow = np.asarray(value["error_slow"], dtype=np.float32)
        instance.update_count = int(value["update_count"])
        instance.decision_count = int(value["decision_count"])
        instance.rng.bit_generator.state = value["numpy_rng"]
        torch.set_rng_state(value["torch_rng"].cpu())
        if instance.device.type == "cuda" and value.get("device_rng") is not None:
            torch.cuda.set_rng_state(value["device_rng"].cpu(), instance.device)
        return instance, dict(value.get("extra", {}))

    @classmethod
    def upgrade_state_conditioned(
        cls, path: str | Path, *, device: str | torch.device = "cpu",
        expected_sha256: str | None = None,
    ) -> tuple["PredictivePPOTrainer", dict[str, Any]]:
        """Upgrade a global-v1 checkpoint with an exactly zero v2 offset head."""
        legacy, extra = cls.restore(path, device=device, expected_sha256=expected_sha256)
        if legacy.config.std_profile != "global-v1":
            raise ValueError("upgrade source must use global-v1 variance")
        torch_rng = torch.get_rng_state()
        device_rng = (torch.cuda.get_rng_state(legacy.device)
                      if legacy.device.type == "cuda" else None)
        upgraded = cls(legacy.resident_ids,
                       replace(legacy.config, std_profile="state-conditioned-v2"),
                       device=legacy.device)
        missing, unexpected = upgraded.model.load_state_dict(
            legacy.model.state_dict(), strict=False)
        if set(missing) != {"std_offset.weight", "std_offset.bias"} or unexpected:
            raise RuntimeError("legacy model differs outside the new variance head")
        old_optimizer = copy.deepcopy(legacy.optimizer.state_dict())
        new_optimizer = upgraded.optimizer.state_dict()
        for old_group, new_group in zip(old_optimizer["param_groups"],
                                        new_optimizer["param_groups"], strict=True):
            old_count = len(old_group["params"])
            old_group["params"] = list(new_group["params"])
            parameters = list(upgraded.model.parameters())
            for parameter_id, parameter in zip(new_group["params"][old_count:],
                                                parameters[old_count:], strict=True):
                old_optimizer["state"][parameter_id] = {
                    "step": torch.tensor(0.0),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
        upgraded.optimizer.load_state_dict(old_optimizer)
        upgraded.context = legacy.context.copy()
        upgraded.moments = RunningMoments.restore(legacy.moments.snapshot())
        upgraded.error_fast = legacy.error_fast.copy()
        upgraded.error_slow = legacy.error_slow.copy()
        upgraded.update_count = legacy.update_count
        upgraded.decision_count = legacy.decision_count
        upgraded.rng.bit_generator.state = copy.deepcopy(legacy.rng.bit_generator.state)
        torch.set_rng_state(torch_rng)
        if upgraded.device.type == "cuda" and device_rng is not None:
            torch.cuda.set_rng_state(device_rng.cpu(), upgraded.device)
        return upgraded, extra

    def _genome_version(self) -> int:
        if self.config.context_profile == "gated-v1":
            return 3
        return 1 if self.config.std_profile == "global-v1" else 2

    def inherit_model(self, inherited: "PredictivePPOTrainer") -> None:
        """Transfer shared capacities into a fresh developmental cohort.

        New context dynamics are an architectural change. Their optimizer
        starts fresh and the parent's private working state is not inherited.
        """
        source, target = inherited.config, self.config
        for key in ("feature_dim", "physiology_dim", "context_dim", "projection_dim", "hidden_dim", "std_profile"):
            if getattr(source, key) != getattr(target, key):
                raise ValueError(f"inherited motor dimensions/variance differ: {key}")
        allowed = {("reservoir-v1", "gated-v1"), (target.context_profile, target.context_profile)}
        if (source.context_profile, target.context_profile) not in allowed:
            raise ValueError("unsupported developmental context transfer")
        missing, unexpected = self.model.load_state_dict(inherited.model.state_dict(), strict=False)
        expected_missing = {
            "context_gate_feature", "context_gate_action", "context_gate_recur", "context_gate_bias"
        } if source.context_profile != target.context_profile else set()
        if set(missing) != expected_missing or unexpected:
            raise ValueError("inherited motor state differs outside the declared gate")
        self.moments = RunningMoments.restore(inherited.moments.snapshot())
        self.update_count = inherited.update_count
        self.decision_count = inherited.decision_count

    def export_genome(self, path: str | Path) -> dict[str, Any]:
        """Write shared learned arrays only; exclude resident context and memory."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: value.detach().cpu().numpy()
            for name, value in self.model.state_dict().items()
        }
        genome_version = self._genome_version()
        metadata = {
            "format": f"chreatures-predictive-ppo-genome-v{genome_version}",
            "version": genome_version,
            "architecture": self.config.std_profile,
            "config": asdict(self.config),
            "actions": list(ACTIONS),
            "updates": self.update_count,
            "decisions": self.decision_count,
            "scope": "shared policy, value, predictor and context mechanism; no personal state",
        }
        arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(path, **arrays)
        return {
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": metadata,
        }

    def import_genome(self, path: str | Path) -> None:
        """Load shared arrays while retaining private context and normalizers."""
        with np.load(path, allow_pickle=False) as value:
            metadata = json.loads(str(value["metadata"]))
            expected_format = f"chreatures-predictive-ppo-genome-v{self._genome_version()}"
            if metadata.get("format") != expected_format or metadata.get("actions") != list(ACTIONS):
                raise ValueError("incompatible predictive PPO genome")
            if metadata.get("architecture", self.config.std_profile) != self.config.std_profile:
                raise ValueError("predictive PPO genome architecture differs")
            if metadata.get("config", {}).get("context_profile", "reservoir-v1") != self.config.context_profile:
                raise ValueError("predictive PPO genome working context differs")
            state = self.model.state_dict()
            restored = {}
            for name, target in state.items():
                array = np.asarray(value[name])
                if tuple(array.shape) != tuple(target.shape) or not np.isfinite(array).all():
                    raise ValueError(f"invalid genome array {name}")
                restored[name] = torch.as_tensor(array, device=self.device, dtype=target.dtype)
            self.model.load_state_dict(restored, strict=True)
