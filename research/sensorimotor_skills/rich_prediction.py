"""Recurrent action-conditioned consequences for the current rich-v3 body."""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn

from chreatures.organism_interface import ACTION_DIM, PHYSIOLOGY_DIM

FORMAT = "chreatures-rich-recurrent-consequence-ensemble-v3"
FRAME_CODE_DIM = 256
FRAME_WINDOW = 4
NEURAL_DIM = 384
WORKER_HIDDEN_DIM = 128
CONTEXT_DIM = (
    FRAME_WINDOW * FRAME_CODE_DIM
    + WORKER_HIDDEN_DIM
    + NEURAL_DIM
    + PHYSIOLOGY_DIM
    + ACTION_DIM
)
OUTPUT_DIM = FRAME_CODE_DIM + PHYSIOLOGY_DIM
LATENT_DIM = 256
MEMBERS = 3
MAX_HORIZON = 8
OBSERVATION_INTERVAL_SECONDS = 0.05
NORMALIZED_INPUT_CLIP = 8.0
CONTEXT_SCALE_FLOOR = 0.02
ACTION_SCALE_FLOOR = 0.02
CODE_DELTA_SCALE_FLOOR = 1e-3
PHYSIOLOGY_DELTA_SCALE_FLOOR = 1e-4
PHYSIOLOGY_LINK_EPSILON = 1e-4
PHYSIOLOGY_LOWER = (0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PHYSIOLOGY_UPPER = (1.0,) * PHYSIOLOGY_DIM
CONTEXT_SEGMENTS = {
    "frame_codes_t_minus_3_through_t": [0, 1024],
    "private_effective_worker_context_t": [1024, 1152],
    "neural_readouts_t": [1152, 1536],
    "raw_physiology_t": [1536, 1548],
    "previous_delivered_action": [1548, 1560],
}
OUTPUT_SEGMENTS = {
    "next_frame_code_delta": [0, 256],
    "next_raw_physiology_delta": [256, 268],
}
FRAME_CODE_SEGMENTS = {"visual": [0, 128], "body": [128, 256]}


@dataclass(frozen=True)
class RichPredictionConfig:
    context_dim: int = CONTEXT_DIM
    action_dim: int = ACTION_DIM
    latent_dim: int = LATENT_DIM
    output_dim: int = OUTPUT_DIM
    members: int = MEMBERS
    max_horizon: int = MAX_HORIZON
    interval_seconds: float = OBSERVATION_INTERVAL_SECONDS

    def __post_init__(self):
        if tuple(asdict(self).values()) != (
            CONTEXT_DIM,
            ACTION_DIM,
            LATENT_DIM,
            OUTPUT_DIM,
            MEMBERS,
            MAX_HORIZON,
            OBSERVATION_INTERVAL_SECONDS,
        ):
            raise ValueError("rich recurrent consequence dimensions are fixed")


class RichRecurrentConsequenceMember(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Linear(CONTEXT_DIM, LATENT_DIM)
        self.transition = nn.GRUCell(ACTION_DIM, LATENT_DIM)
        self.output = nn.Linear(LATENT_DIM, OUTPUT_DIM)

    def forward(self, context, actions):
        if context.ndim != 2 or context.shape[-1] != CONTEXT_DIM:
            raise ValueError("context must be [N,1560]")
        if (
            actions.ndim != 3
            or actions.shape[0] != context.shape[0]
            or actions.shape[-1] != ACTION_DIM
            or not 1 <= actions.shape[1] <= MAX_HORIZON
        ):
            raise ValueError("actions must be [N,H,12], 1<=H<=8")
        state = torch.tanh(self.context(context))
        values = []
        for step in range(actions.shape[1]):
            state = self.transition(actions[:, step], state)
            values.append(self.output(state))
        return torch.stack(values, dim=1)


class RichRecurrentConsequenceEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = RichPredictionConfig()
        self.members = nn.ModuleList(
            RichRecurrentConsequenceMember() for _ in range(MEMBERS)
        )

    def forward(self, context, actions):
        return torch.stack([m(context, actions) for m in self.members], dim=1)


def normalize_context(value, mean, scale):
    z = (value - mean) / scale
    return z.clamp(-NORMALIZED_INPUT_CLIP, NORMALIZED_INPUT_CLIP), torch.any(
        torch.abs(z) > NORMALIZED_INPUT_CLIP, dim=-1
    )


def normalize_actions(value, mean, scale):
    z = (value - mean) / scale
    return z.clamp(-NORMALIZED_INPUT_CLIP, NORMALIZED_INPUT_CLIP), torch.any(
        torch.abs(z) > NORMALIZED_INPUT_CLIP, dim=-1
    )


def denormalize_deltas(value, mean, scale):
    return value * scale + mean


def bounded_physiology_deltas(proposals, physiology_anchor):
    """Decode proposals to feasible deltas without clipping predicted state."""
    if proposals.shape[-1] != PHYSIOLOGY_DIM:
        raise ValueError("physiology proposals must end with 12")
    lower = proposals.new_tensor(PHYSIOLOGY_LOWER)
    upper = proposals.new_tensor(PHYSIOLOGY_UPPER)
    state = physiology_anchor
    values = []
    for horizon in range(proposals.shape[-2]):
        proposal = proposals[..., horizon, :]
        absolute = proposal.abs()
        root = torch.sqrt(proposal.square() + PHYSIOLOGY_LINK_EPSILON**2)
        small = PHYSIOLOGY_LINK_EPSILON**2 / (2 * (root + absolute))
        positive = torch.where(proposal >= 0, absolute + small, small)
        negative = torch.where(proposal < 0, absolute + small, small)
        upward = upper - state
        downward = state - lower
        delta = upward * torch.tanh(
            positive / upward.clamp_min(PHYSIOLOGY_LINK_EPSILON)
        ) - downward * torch.tanh(
            negative / downward.clamp_min(PHYSIOLOGY_LINK_EPSILON)
        )
        values.append(delta)
        state = state + delta
    return torch.stack(values, dim=-2)


def cumulative_forecast(member_deltas, code_anchor, physiology_anchor):
    if member_deltas.ndim != 5:
        raise ValueError("member deltas must be [B,K,3,H,268]")
    return (
        code_anchor[:, None, None, None, :] + member_deltas[..., :256].cumsum(3),
        physiology_anchor[:, None, None, None, :]
        + member_deltas[..., 256:].cumsum(3),
    )


def ensemble_summary(member_values):
    if member_values.ndim != 5:
        raise ValueError("member values must be [B,K,3,H,D]")
    mean = member_values.mean(dim=2)
    return mean, torch.sqrt(
        torch.mean((member_values - mean[:, :, None]) ** 2, dim=2)
    )


def tensor_bundle_sha256(values: Mapping[str, np.ndarray]) -> str:
    d = hashlib.sha256()
    for name in sorted(values):
        v = np.ascontiguousarray(values[name], dtype="<f4")
        d.update(name.encode())
        d.update(b"\0")
        d.update(json.dumps(list(v.shape), separators=(",", ":")).encode())
        d.update(b"\0<f4\0")
        d.update(v.tobytes())
    return d.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def artifact_identity(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    clean = dict(metadata)
    clean.pop("artifact_identity", None)
    receipts = {
        n: {
            "dtype": np.ascontiguousarray(v).dtype.str,
            "shape": list(v.shape),
            "sha256": array_sha256(v),
        }
        for n, v in sorted(arrays.items())
    }
    return hashlib.sha256(
        json.dumps(
            {"metadata": clean, "arrays": receipts},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
