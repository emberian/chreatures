"""Research action-skilled recurrent predictor with ensemble uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from chreatures.predictive_state import PredictiveSequence


@dataclass(frozen=True)
class PredictiveSkillsConfig:
    feature_dim: int = 384
    physiology_dim: int = 6
    action_dim: int = 8
    encoder_dim: int = 128
    latent_dim: int = 96
    horizon_dim: int = 16
    max_horizon: int = 16
    learning_rate: float = 3e-4
    consistency_rate: float = 0.08
    max_grad_norm: float = 1.0
    seed: int = 20260906

    def __post_init__(self) -> None:
        if (
            min(
                self.feature_dim,
                self.physiology_dim,
                self.action_dim,
                self.encoder_dim,
                self.latent_dim,
                self.horizon_dim,
            )
            <= 0
        ):
            raise ValueError("predictive skill dimensions must be positive")
        if self.max_horizon != 16:
            raise ValueError("research predictive skills require H1..H16")


@dataclass(frozen=True)
class SkillsNormalizer:
    input_mean: np.ndarray
    input_scale: np.ndarray
    physiology_delta_mean: np.ndarray
    physiology_delta_scale: np.ndarray
    identity: dict[str, Any]

    @classmethod
    def fit(
        cls,
        sequences: Sequence[PredictiveSequence],
        identity: dict[str, Any],
        max_horizon: int = 16,
    ) -> "SkillsNormalizer":
        observations = np.concatenate(
            [
                np.concatenate((s.features, s.physiology), axis=-1).reshape(-1, 390)
                for s in sequences
            ]
        ).astype(np.float64)
        mean = observations.mean(0)
        scale = np.maximum(observations.std(0), 1e-4)
        delta_mean, delta_scale = [], []
        for horizon in range(1, max_horizon + 1):
            values = []
            for sequence in sequences:
                mask = sequence.valid[:-horizon] & sequence.valid[horizon:]
                mask &= ~sequence.reset[horizon:]
                values.append(
                    (sequence.physiology[horizon:] - sequence.physiology[:-horizon])[
                        mask
                    ]
                )
            delta = np.concatenate(values).astype(np.float64)
            delta_mean.append(delta.mean(0))
            delta_scale.append(np.maximum(delta.std(0), 1e-6))
        return cls(
            mean.astype(np.float32),
            scale.astype(np.float32),
            np.asarray(delta_mean, np.float32),
            np.asarray(delta_scale, np.float32),
            dict(identity),
        )


class PredictiveSkillsModel(nn.Module):
    """GRU dynamics with explicit effort-symmetric action features."""

    def __init__(self, config: PredictiveSkillsConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = nn.Sequential(
            nn.Linear(config.feature_dim + config.physiology_dim, config.encoder_dim),
            nn.Tanh(),
        )
        self.action_basis_dim = config.action_dim + 3
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_basis_dim, 64), nn.Tanh()
        )
        self.observe_rnn = nn.GRU(
            config.encoder_dim + self.action_basis_dim, config.latent_dim
        )
        self.transition_cell = nn.GRUCell(64, config.latent_dim)
        self.horizon_embedding = nn.Embedding(config.max_horizon, config.horizon_dim)
        feature_input = config.latent_dim + config.horizon_dim
        physiology_input = feature_input + self.action_basis_dim
        self.feature_mean = nn.Linear(feature_input, config.feature_dim)
        self.feature_log_std = nn.Linear(feature_input, config.feature_dim)
        self.physiology_delta_mean = nn.Sequential(
            nn.Linear(physiology_input, 96),
            nn.Tanh(),
            nn.Linear(96, config.physiology_dim),
        )
        self.physiology_delta_log_std = nn.Sequential(
            nn.Linear(physiology_input, 96),
            nn.Tanh(),
            nn.Linear(96, config.physiology_dim),
        )
        nn.init.constant_(self.feature_log_std.bias, -0.5)
        nn.init.constant_(self.physiology_delta_log_std[-1].bias, -0.5)

    @staticmethod
    def action_basis(action: torch.Tensor) -> torch.Tensor:
        # Thrust, yaw, and gaze/vertical are signed controls whose physical
        # effort uses magnitude. Grip and signals are already clipped >= 0.
        return torch.cat((action, action[..., :3].abs()), dim=-1)

    def transition(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition_cell(
            self.action_encoder(self.action_basis(action)), state
        )

    def prediction(
        self, state: torch.Tensor, action: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index = torch.full(
            state.shape[:-1], horizon - 1, dtype=torch.long, device=state.device
        )
        horizon_value = self.horizon_embedding(index)
        feature_input = torch.cat((state, horizon_value), dim=-1)
        physiology_input = torch.cat((feature_input, self.action_basis(action)), dim=-1)
        return (
            self.feature_mean(feature_input),
            torch.clamp(self.feature_log_std(feature_input), -3.0, 1.0),
            self.physiology_delta_mean(physiology_input),
            torch.clamp(self.physiology_delta_log_std(physiology_input), -3.0, 1.0),
        )


class PredictiveSkillsTrainer:
    def __init__(
        self,
        config: PredictiveSkillsConfig,
        normalizer: SkillsNormalizer,
        *,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.normalizer = normalizer
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        self.model = PredictiveSkillsModel(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.updates = 0

    def _posterior(self, sequence: PredictiveSequence) -> tuple[torch.Tensor, ...]:
        observation = np.concatenate((sequence.features, sequence.physiology), axis=-1)
        observation = (
            observation - self.normalizer.input_mean
        ) / self.normalizer.input_scale
        obs = torch.as_tensor(observation, device=self.device)
        actions = torch.as_tensor(sequence.actions, device=self.device)
        reset = torch.as_tensor(sequence.reset, device=self.device)
        valid = torch.as_tensor(sequence.valid, device=self.device)
        encoded = self.model.observation_encoder(obs)
        zero = torch.zeros_like(actions[:1])
        previous = torch.cat((zero, actions[:-1]))
        recurrent = torch.cat((encoded, self.model.action_basis(previous)), dim=-1)
        state = torch.zeros(
            (1, obs.shape[1], self.config.latent_dim), device=self.device
        )
        chunks = []
        boundaries = sorted(
            {
                0,
                len(obs),
                *torch.nonzero((reset | ~valid).any(1), as_tuple=False)
                .flatten()
                .cpu()
                .tolist(),
            }
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            mask = reset[start] | ~valid[start]
            state = torch.where(mask[None, :, None], torch.zeros_like(state), state)
            output, state = self.model.observe_rnn(recurrent[start:stop], state)
            chunks.append(output)
        return torch.cat(chunks), obs, actions, reset, valid

    def loss(
        self, sequence: PredictiveSequence, *, loss_start: int = 0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        posterior, obs, actions, resets, valid = self._posterior(sequence)
        time, residents = valid.shape
        imagined = posterior[:-1].reshape(-1, self.config.latent_dim)
        alive = valid[:-1].clone()
        alive[:loss_start] = False
        raw_physiology = torch.as_tensor(sequence.physiology, device=self.device)
        delta_mean = torch.as_tensor(
            self.normalizer.physiology_delta_mean, device=self.device
        )
        delta_scale = torch.as_tensor(
            self.normalizer.physiology_delta_scale, device=self.device
        )
        terms, consistency_terms = [], []
        metrics: dict[str, float] = {}
        for horizon in range(1, self.config.max_horizon + 1):
            usable = time - horizon
            shifted = torch.zeros_like(actions[:-1])
            shifted[:usable] = actions[horizon - 1 : time - 1]
            imagined = self.model.transition(
                imagined, shifted.reshape(-1, self.config.action_dim)
            )
            alive[:usable] &= valid[horizon:] & ~resets[horizon:]
            alive[usable:] = False
            selected = alive[:usable].clone()
            state = imagined.reshape(time - 1, residents, -1)[:usable]
            feature_mean, feature_log_std, phys_mean, phys_log_std = (
                self.model.prediction(state, shifted[:usable], horizon)
            )
            feature_error = (
                obs[horizon:, :, : self.config.feature_dim] - feature_mean
            ) * torch.exp(-feature_log_std)
            phys_target = (
                raw_physiology[horizon:]
                - raw_physiology[:-horizon]
                - delta_mean[horizon - 1]
            ) / delta_scale[horizon - 1]
            phys_error = (phys_target - phys_mean) * torch.exp(-phys_log_std)
            feature_nll = (0.5 * feature_error.square() + feature_log_std).mean(-1)[
                selected
            ]
            phys_nll = (0.5 * phys_error.square() + phys_log_std).mean(-1)[selected]
            terms.append(0.5 * feature_nll + 0.5 * phys_nll)
            consistency_terms.append(
                (state - posterior[horizon:].detach()).square().mean(-1)[selected]
            )
            metrics[f"feature_nll_h{horizon}"] = float(feature_nll.mean().detach())
            metrics[f"physiology_nll_h{horizon}"] = float(phys_nll.mean().detach())
        nll = torch.cat(terms).mean()
        consistency = torch.cat(consistency_terms).mean()
        return nll + self.config.consistency_rate * consistency, metrics

    def update(
        self, sequence: PredictiveSequence, *, loss_start: int = 0
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(sequence, loss_start=loss_start)
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.updates += 1
        return {
            **metrics,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient),
            "update": self.updates,
        }

    def state_value(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "updates": self.updates,
        }
