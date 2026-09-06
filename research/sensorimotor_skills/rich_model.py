"""Breaking rich-vision sensorimotor worker for the next research dataset.

The model consumes direct ``rich-body-v1`` rays plus the coarse sensory/body
vector. It has no 357-input compatibility path. Normalization moments and the
sensorium profile identity belong to the future dataset/checkpoint contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import RECTIFIED_AXES, SIGNED_AXES

RICH_PROFILE_SHA256 = "c71380718ba5535dbaebdeaf8aa2e88cc45cf218312a03e13507877f02a5554e"
RICH_CHANNEL_NAMES_SHA256 = (
    "b4c6b328116d820143e16ee922ccffd7b950dbe008efc580ad93056e01349bfa"
)
OBSERVATION_ORDER = ("rich_body_v1_4096", "canonical_channels_351", "physiology_6")


@dataclass(frozen=True)
class RichSensorimotorConfig:
    rich_dim: int = 4096
    body_dim: int = 357
    observation_dim: int = 4453
    previous_dim: int = 9
    vision_dim: int = 128
    body_embedding_dim: int = 128
    recurrent_dim: int = 128
    goal_window: int = 4
    goal_dim: int = 64
    policy_hidden_dim: int = 256
    rich_profile_sha256: str = RICH_PROFILE_SHA256
    rich_channel_names_sha256: str = RICH_CHANNEL_NAMES_SHA256
    observation_order: tuple[str, str, str] = OBSERVATION_ORDER

    def __post_init__(self) -> None:
        expected = (4096, 357, 4453, 9, 128, 128, 128, 4, 64, 256)
        dimensions = (
            self.rich_dim,
            self.body_dim,
            self.observation_dim,
            self.previous_dim,
            self.vision_dim,
            self.body_embedding_dim,
            self.recurrent_dim,
            self.goal_window,
            self.goal_dim,
            self.policy_hidden_dim,
        )
        if dimensions != expected:
            raise ValueError("rich sensorimotor v1 dimensions are fixed")
        if (
            self.rich_profile_sha256 != RICH_PROFILE_SHA256
            or self.rich_channel_names_sha256 != RICH_CHANNEL_NAMES_SHA256
            or self.observation_order != OBSERVATION_ORDER
        ):
            raise ValueError("rich sensorimotor sensor identity differs")


class AngularConv(nn.Module):
    """Two finite-FOV convolutions with replicated angular boundary values."""

    def __init__(self, height: int) -> None:
        super().__init__()
        self.height = height
        self.first = nn.Conv2d(4, 16, kernel_size=3, stride=(1, 2))
        self.second = nn.Conv2d(16, 24, kernel_size=3, stride=(2, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-3:] != (4, self.height, 32):
            raise ValueError(f"angular tensor must end with [4,{self.height},32]")
        # Both axes cover finite fields of view. Replication does not invent a
        # periodic seam between the left and right rays.
        value = torch.tanh(self.first(F.pad(value, (1, 1, 1, 1), mode="replicate")))
        return torch.tanh(self.second(F.pad(value, (1, 1, 1, 1), mode="replicate")))


class RichVisualFront(nn.Module):
    """Encode peripheral ``8x32`` and foveal ``24x32`` RGB-proximity rays."""

    def __init__(self) -> None:
        super().__init__()
        self.peripheral = AngularConv(8)
        self.foveal = AngularConv(24)
        self.peripheral_projection = nn.Sequential(nn.Linear(24 * 4 * 8, 64), nn.Tanh())
        self.foveal_projection = nn.Sequential(nn.Linear(24 * 12 * 8, 64), nn.Tanh())

    def forward(self, rich: torch.Tensor) -> torch.Tensor:
        if rich.shape[-1] != 4096:
            raise ValueError("rich vision must end with 4096")
        leading = rich.shape[:-1]
        rays = rich.reshape(-1, 1024, 4)
        peripheral = rays[:, :256].reshape(-1, 8, 32, 4).permute(0, 3, 1, 2)
        foveal = rays[:, 256:].reshape(-1, 24, 32, 4).permute(0, 3, 1, 2)
        peripheral = self.peripheral(peripheral).flatten(1)
        foveal = self.foveal(foveal).flatten(1)
        encoded = torch.cat(
            (self.peripheral_projection(peripheral), self.foveal_projection(foveal)),
            dim=-1,
        )
        return encoded.reshape(*leading, 128)


class RichSensorimotorModel(nn.Module):
    """Shared sensory front, causal goal encoder, GRU worker, and action heads."""

    def __init__(self, config: RichSensorimotorConfig | None = None) -> None:
        super().__init__()
        self.config = config or RichSensorimotorConfig()
        self.visual = RichVisualFront()
        self.body = nn.Sequential(nn.Linear(357, 128), nn.Tanh())
        self.goal_encoder = nn.Sequential(
            nn.Linear(4 * 256, 256), nn.Tanh(), nn.Linear(256, 64)
        )
        # Training-only decoder directly reconstructs all rich and body inputs.
        self.goal_decoder = nn.Sequential(
            nn.Linear(64, 256), nn.Tanh(), nn.Linear(256, 4 * 4453)
        )
        self.observation_projection = nn.Sequential(
            nn.Linear(128 + 128 + 9, 128), nn.Tanh()
        )
        self.history = nn.GRU(128, 128)
        self.policy_trunk = nn.Sequential(nn.Linear(128 + 64 + 1 + 8, 256), nn.Tanh())
        self.signed_head = nn.Linear(256, 4 * 65)
        self.active_head = nn.Linear(256, 4)
        self.positive_head = nn.Linear(256, 4 * 32)
        self.register_buffer("signed_centers", torch.linspace(-1, 1, 65))
        self.register_buffer("positive_centers", torch.arange(1, 33) / 32)

    def encode_frames(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode normalized ``[rich4096, canonical351, physiology6]`` rows."""
        if observation.shape[-1] != 4453:
            raise ValueError("rich observation must end with 4453")
        return torch.cat(
            (self.visual(observation[..., :4096]), self.body(observation[..., 4096:])),
            dim=-1,
        )

    def encode_goal(self, window: torch.Tensor) -> torch.Tensor:
        """Encode four causal frames ending at an achieved tick."""
        if window.shape[-2:] != (4, 4453):
            raise ValueError("rich goal window must end with [4,4453]")
        return self.goal_encoder(self.encode_frames(window).flatten(start_dim=-2))

    def goal_reconstruction_loss(
        self, window: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Balanced direct rich/body Huber objective with anti-collapse terms."""
        latent = self.encode_goal(window)
        reconstruction = self.goal_decoder(latent).reshape(*window.shape)
        rich = F.huber_loss(reconstruction[..., :4096], window[..., :4096])
        body = F.huber_loss(reconstruction[..., 4096:], window[..., 4096:])
        samples = latent.reshape(-1, 64)
        deviation = samples.std(0, unbiased=False)
        variance_floor = F.relu(0.3 - deviation).square().mean()
        centered = samples - samples.mean(0, keepdim=True)
        covariance = centered.T @ centered / max(len(samples) - 1, 1)
        off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
        covariance_penalty = off_diagonal.square().mean()
        loss = 0.5 * (rich + body) + 0.01 * variance_floor + 0.001 * covariance_penalty
        return loss, {
            "rich_reconstruction_huber": rich.detach(),
            "body_reconstruction_huber": body.detach(),
            "variance_floor": variance_floor.detach(),
            "off_diagonal_covariance": covariance_penalty.detach(),
        }

    def encode_sequence(self, observation, previous, hidden, reset):
        """Encode ``[T,B,4453]`` causally, resetting private hidden before rows."""
        if observation.ndim != 3 or observation.shape[-1] != 4453:
            raise ValueError("observation must be [T,B,4453]")
        if previous.shape != (*observation.shape[:2], 9):
            raise ValueError("previous executed action must be [T,B,9]")
        if reset.shape != observation.shape[:2] or reset.dtype != torch.bool:
            raise ValueError("reset must be bool [T,B]")
        batch = observation.shape[1]
        if hidden is None:
            hidden = observation.new_zeros((1, batch, 128))
        encoded = self.observation_projection(
            torch.cat((self.encode_frames(observation), previous), dim=-1)
        )
        hidden = torch.where(reset[0][None, :, None], 0, hidden)
        if len(observation) == 1 or not bool(reset[1:].any().item()):
            return self.history(encoded, hidden)
        states = []
        for index in range(len(observation)):
            if index:
                hidden = torch.where(reset[index][None, :, None], 0, hidden)
            state, hidden = self.history(encoded[index : index + 1], hidden)
            states.append(state)
        return torch.cat(states), hidden

    def policy(self, states, goal, horizon, previous_action):
        leading = states.shape[:-1]
        if states.shape[-1] != 128 or goal.shape != (*leading, 64):
            raise ValueError("policy state/goal shape differs")
        if horizon.shape == leading:
            horizon = horizon[..., None]
        if horizon.shape != (*leading, 1) or previous_action.shape != (*leading, 8):
            raise ValueError("policy horizon/previous action shape differs")
        hidden = self.policy_trunk(
            torch.cat((states, goal, horizon, previous_action), -1)
        )
        return {
            "signed": self.signed_head(hidden).reshape(*leading, 4, 65),
            "active": self.active_head(hidden),
            "positive": self.positive_head(hidden).reshape(*leading, 4, 32),
        }

    def action_nll(self, logits, action):
        """Return factorized NLL in canonical eight-axis order."""
        leading = action.shape[:-1]
        result = action.new_empty((*leading, 8))
        signed_target = (
            torch.round((action[..., SIGNED_AXES] + 1) * 32).long().clamp(0, 64)
        )
        result[..., SIGNED_AXES] = F.cross_entropy(
            logits["signed"].reshape(-1, 65),
            signed_target.reshape(-1),
            reduction="none",
        ).reshape(*leading, 4)
        rectified = action[..., RECTIFIED_AXES]
        active = rectified > 0
        hurdle = F.binary_cross_entropy_with_logits(
            logits["active"], active.to(action.dtype), reduction="none"
        )
        positive_target = torch.ceil(rectified * 32).long().clamp(1, 32) - 1
        positive = F.cross_entropy(
            logits["positive"].reshape(-1, 32),
            positive_target.reshape(-1),
            reduction="none",
        ).reshape(*leading, 4)
        result[..., RECTIFIED_AXES] = hurdle + active * positive
        return result
