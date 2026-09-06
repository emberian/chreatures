"""Research-only achieved-goal sensorimotor worker models.

This module contains portable Torch modules and losses only. It is deliberately
outside the production ``chreatures`` package; data partitioning, normalization,
training schedules, checkpoint manifests, and evaluation belong to the owning
research scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

SIGNED_AXES = (0, 1, 2, 7)
RECTIFIED_AXES = (3, 4, 5, 6)


@dataclass(frozen=True)
class SensorimotorSkillConfig:
    """Fixed tensor dimensions for the first 20 Hz hindsight worker."""

    observation_dim: int = 357
    recurrent_previous_action_dim: int = 9
    explicit_previous_action_dim: int = 8
    goal_window: int = 4
    goal_hidden_dim: int = 256
    goal_dim: int = 64
    observation_hidden_dim: int = 128
    recurrent_dim: int = 128
    policy_hidden_dim: int = 256
    signed_bins: int = 65
    positive_bins: int = 32

    def __post_init__(self) -> None:
        expected = {
            "observation_dim": 357,
            "recurrent_previous_action_dim": 9,
            "explicit_previous_action_dim": 8,
            "goal_window": 4,
            "goal_hidden_dim": 256,
            "goal_dim": 64,
            "observation_hidden_dim": 128,
            "recurrent_dim": 128,
            "policy_hidden_dim": 256,
            "signed_bins": 65,
            "positive_bins": 32,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"sensorimotor worker v1 requires {name}={value}")

    @property
    def flattened_goal_observation_dim(self) -> int:
        return self.goal_window * self.observation_dim


class GoalEncoder(nn.Module):
    """Encode four observations ending at an achieved future time.

    ``encode(window)`` accepts ``[...,4,357]`` and returns ``[...,64]``.
    ``decode(z)`` reconstructs ``[...,4,357]``. The trainer fits this module on
    training worlds only and calls :meth:`freeze` before worker optimization.
    """

    def __init__(self, config: SensorimotorSkillConfig | None = None) -> None:
        super().__init__()
        self.config = config or SensorimotorSkillConfig()
        flattened = self.config.flattened_goal_observation_dim
        self.encoder = nn.Sequential(
            nn.Linear(flattened, self.config.goal_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.goal_hidden_dim, self.config.goal_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.config.goal_dim, self.config.goal_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.goal_hidden_dim, flattened),
        )

    def encode(self, window: torch.Tensor) -> torch.Tensor:
        if window.shape[-2:] != (
            self.config.goal_window,
            self.config.observation_dim,
        ):
            raise ValueError("goal window must end with [4,357]")
        return self.encoder(window.flatten(start_dim=-2))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.config.goal_dim:
            raise ValueError("goal latent must end with 64")
        shape = (
            *latent.shape[:-1],
            self.config.goal_window,
            self.config.observation_dim,
        )
        return self.decoder(latent).reshape(shape)

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(window)
        return latent, self.decode(latent)

    def reconstruction_loss(
        self, window: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Huber reconstruction plus declared anti-collapse penalties."""
        latent, reconstruction = self(window)
        reconstruction_term = F.huber_loss(reconstruction, window)
        samples = latent.reshape(-1, self.config.goal_dim)
        standard_deviation = samples.std(dim=0, unbiased=False)
        variance_floor = F.relu(0.3 - standard_deviation).square().mean()
        centered = samples - samples.mean(dim=0, keepdim=True)
        denominator = max(len(samples) - 1, 1)
        covariance = centered.transpose(0, 1) @ centered / denominator
        off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
        covariance_penalty = off_diagonal.square().mean()
        total = reconstruction_term + 0.01 * variance_floor + 0.001 * covariance_penalty
        return total, {
            "reconstruction_huber": reconstruction_term.detach(),
            "variance_floor": variance_floor.detach(),
            "off_diagonal_covariance": covariance_penalty.detach(),
        }

    def freeze(self) -> "GoalEncoder":
        self.requires_grad_(False)
        self.eval()
        return self


class SensorimotorWorker(nn.Module):
    """Causal history encoder and factorized goal-conditioned action policy.

    ``encode_sequence`` consumes normalized observations ``[T,B,357]``, exact
    prior executed controls plus oral command ``[T,B,9]``, optional hidden state
    ``[1,B,128]``, and reset-before-row mask ``[T,B]``.

    ``policy`` accepts broadcast-compatible states ``[...,128]``, frozen goals
    ``[...,64]``, normalized log horizon ``[...,1]``, and explicit previous motor
    actions ``[...,8]``. It returns logits named ``signed [...,4,65]``, ``active
    [...,4]``, and ``positive [...,4,32]``.
    """

    def __init__(self, config: SensorimotorSkillConfig | None = None) -> None:
        super().__init__()
        self.config = config or SensorimotorSkillConfig()
        self.observation_projection = nn.Sequential(
            nn.Linear(
                self.config.observation_dim + self.config.recurrent_previous_action_dim,
                self.config.observation_hidden_dim,
            ),
            nn.Tanh(),
        )
        self.history = nn.GRU(
            self.config.observation_hidden_dim, self.config.recurrent_dim
        )
        policy_input = (
            self.config.recurrent_dim
            + self.config.goal_dim
            + 1
            + self.config.explicit_previous_action_dim
        )
        self.policy_trunk = nn.Sequential(
            nn.Linear(policy_input, self.config.policy_hidden_dim), nn.Tanh()
        )
        self.signed_head = nn.Linear(
            self.config.policy_hidden_dim, len(SIGNED_AXES) * self.config.signed_bins
        )
        self.active_head = nn.Linear(self.config.policy_hidden_dim, len(RECTIFIED_AXES))
        self.positive_head = nn.Linear(
            self.config.policy_hidden_dim,
            len(RECTIFIED_AXES) * self.config.positive_bins,
        )
        self.register_buffer(
            "signed_centers", torch.linspace(-1.0, 1.0, self.config.signed_bins)
        )
        self.register_buffer(
            "positive_centers",
            torch.arange(1, self.config.positive_bins + 1, dtype=torch.float32)
            / self.config.positive_bins,
        )

    def encode_sequence(
        self,
        observation: torch.Tensor,
        previous_action: torch.Tensor,
        hidden: torch.Tensor | None,
        reset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a TBPTT chunk while applying every real reset causally."""
        if (
            observation.ndim != 3
            or observation.shape[-1] != self.config.observation_dim
        ):
            raise ValueError("observation must be [T,B,357]")
        expected = (*observation.shape[:2], self.config.recurrent_previous_action_dim)
        if previous_action.shape != expected:
            raise ValueError("previous action must be [T,B,9]")
        if reset.shape != observation.shape[:2] or reset.dtype != torch.bool:
            raise ValueError("reset must be bool [T,B]")
        time, batch = reset.shape
        if hidden is None:
            hidden = observation.new_zeros((1, batch, self.config.recurrent_dim))
        if hidden.shape != (1, batch, self.config.recurrent_dim):
            raise ValueError("hidden must be [1,B,128]")
        encoded = self.observation_projection(
            torch.cat((observation, previous_action), dim=-1)
        )
        hidden = torch.where(reset[0][None, :, None], torch.zeros_like(hidden), hidden)
        if time == 1 or not bool(reset[1:].any().item()):
            return self.history(encoded, hidden)
        states = []
        for index in range(time):
            if index:
                hidden = torch.where(
                    reset[index][None, :, None], torch.zeros_like(hidden), hidden
                )
            state, hidden = self.history(encoded[index : index + 1], hidden)
            states.append(state)
        return torch.cat(states), hidden

    def policy(
        self,
        states: torch.Tensor,
        goal: torch.Tensor,
        normalized_log_horizon: torch.Tensor,
        explicit_previous_action: torch.Tensor,
        *,
        mask_previous_probability: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return action logits; optionally mask only the explicit skip input."""
        leading = states.shape[:-1]
        if states.shape[-1] != self.config.recurrent_dim:
            raise ValueError("states must end with 128")
        if goal.shape != (*leading, self.config.goal_dim):
            raise ValueError("goal must match states and end with 64")
        if normalized_log_horizon.shape == leading:
            normalized_log_horizon = normalized_log_horizon.unsqueeze(-1)
        if normalized_log_horizon.shape != (*leading, 1):
            raise ValueError("normalized log horizon must match states and end with 1")
        if explicit_previous_action.shape != (
            *leading,
            self.config.explicit_previous_action_dim,
        ):
            raise ValueError(
                "explicit previous action must match states and end with 8"
            )
        if not 0.0 <= mask_previous_probability <= 1.0:
            raise ValueError("previous-action mask probability must be in [0,1]")
        previous = explicit_previous_action
        if self.training and mask_previous_probability:
            keep = (
                torch.rand(
                    (*leading, 1),
                    device=states.device,
                    generator=generator,
                )
                >= mask_previous_probability
            )
            previous = previous * keep
        value = self.policy_trunk(
            torch.cat((states, goal, normalized_log_horizon, previous), dim=-1)
        )
        return {
            "signed": self.signed_head(value).reshape(
                *leading, len(SIGNED_AXES), self.config.signed_bins
            ),
            "active": self.active_head(value),
            "positive": self.positive_head(value).reshape(
                *leading, len(RECTIFIED_AXES), self.config.positive_bins
            ),
        }

    def action_nll(
        self, logits: dict[str, torch.Tensor], action: torch.Tensor
    ) -> torch.Tensor:
        """Return equal-status per-axis NLL in original action order ``[...,8]``."""
        if action.shape[-1] != 8 or not torch.isfinite(action).all():
            raise ValueError("action must be finite and end with 8")
        if torch.any(action < -1) or torch.any(action > 1):
            raise ValueError("action lies outside physical bounds")
        leading = action.shape[:-1]
        if logits.get("signed", torch.empty(0)).shape != (*leading, 4, 65):
            raise ValueError("signed logits must be [...,4,65]")
        if logits.get("active", torch.empty(0)).shape != (*leading, 4):
            raise ValueError("active logits must be [...,4]")
        if logits.get("positive", torch.empty(0)).shape != (*leading, 4, 32):
            raise ValueError("positive logits must be [...,4,32]")
        result = action.new_empty((*leading, 8))
        signed_target = torch.round((action[..., SIGNED_AXES] + 1.0) * 32.0).long()
        signed_target = signed_target.clamp(0, 64)
        signed_nll = F.cross_entropy(
            logits["signed"].reshape(-1, 65),
            signed_target.reshape(-1),
            reduction="none",
        ).reshape(*leading, 4)
        result[..., SIGNED_AXES] = signed_nll
        rectified = action[..., RECTIFIED_AXES]
        if torch.any(rectified < 0):
            raise ValueError("grip and signal actions must be nonnegative")
        active_target = rectified > 0
        active_nll = F.binary_cross_entropy_with_logits(
            logits["active"], active_target.to(action.dtype), reduction="none"
        )
        positive_target = torch.ceil(rectified * 32.0).long().clamp(1, 32) - 1
        positive_nll = F.cross_entropy(
            logits["positive"].reshape(-1, 32),
            positive_target.reshape(-1),
            reduction="none",
        ).reshape(*leading, 4)
        result[..., RECTIFIED_AXES] = active_nll + active_target * positive_nll
        return result

    def decode(
        self,
        logits: dict[str, torch.Tensor],
        mode: Literal["mode", "sample"] = "mode",
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode bounded actions in canonical eight-axis order."""
        signed_logits = logits["signed"]
        active_logits = logits["active"]
        positive_logits = logits["positive"]
        leading = signed_logits.shape[:-2]
        if signed_logits.shape[-2:] != (4, 65):
            raise ValueError("signed logits must end with [4,65]")
        if active_logits.shape != (*leading, 4) or positive_logits.shape != (
            *leading,
            4,
            32,
        ):
            raise ValueError("rectified logits differ from signed leading shape")
        if mode == "mode":
            signed_index = signed_logits.argmax(-1)
            positive_index = positive_logits.argmax(-1)
            # The hurdle distribution has one inactive outcome and 32 joint
            # active-positive outcomes. Its MAP decision must compare the best
            # complete active outcome with the inactive mass; thresholding the
            # Bernoulli alone can select an active state whose probability is
            # divided across many positive bins.
            inactive_log_probability = F.logsigmoid(-active_logits)
            best_active_log_probability = F.logsigmoid(active_logits) + F.log_softmax(
                positive_logits, dim=-1
            ).amax(-1)
            active = best_active_log_probability > inactive_log_probability
        elif mode == "sample":
            signed_index = torch.multinomial(
                signed_logits.softmax(-1).reshape(-1, 65),
                1,
                generator=generator,
            ).reshape(*leading, 4)
            active = (
                torch.rand(
                    active_logits.shape,
                    device=active_logits.device,
                    generator=generator,
                )
                < active_logits.sigmoid()
            )
            positive_index = torch.multinomial(
                positive_logits.softmax(-1).reshape(-1, 32),
                1,
                generator=generator,
            ).reshape(*leading, 4)
        else:
            raise ValueError("decode mode must be 'mode' or 'sample'")
        output = signed_logits.new_zeros((*leading, 8))
        output[..., SIGNED_AXES] = self.signed_centers[signed_index]
        positive = self.positive_centers[positive_index]
        output[..., RECTIFIED_AXES] = torch.where(active, positive, 0.0)
        return output
