"""Action-conditioned prediction over the current rich-v3 trajectory contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from chreatures.organism_interface import ACTION_DIM, PHYSIOLOGY_DIM

FORMAT = "chreatures-rich-consequence-ensemble-v2"
ACTION_SUFFIX_FORMAT = "chreatures-rich-action-suffix-consequence-ensemble-v2"
ACTION_SUFFIX_HORIZONS = (5, 20)
FRAME_CODE_DIM = 256
FRAME_WINDOW = 4
NEURAL_DIM = 384
INPUT_DIM = FRAME_WINDOW * FRAME_CODE_DIM + NEURAL_DIM + 2 * ACTION_DIM
OUTPUT_DIM = FRAME_CODE_DIM + PHYSIOLOGY_DIM
ACTION_SUFFIX_OUTPUT_DIM = FRAME_WINDOW * FRAME_CODE_DIM + PHYSIOLOGY_DIM
MEMBERS = 3
INPUT_SCALE_FLOOR = 0.02
CODE_DELTA_SCALE_FLOOR = 1e-3
PHYSIOLOGY_DELTA_SCALE_FLOOR = 1e-4
NORMALIZED_INPUT_CLIP = 8.0

INPUT_SEGMENTS = {
    "frame_codes_t_minus_3_through_t": [0, 1024],
    "neural_readouts_t": [1024, 1408],
    "previous_executed_action": [1408, 1420],
    "candidate_action": [1420, 1432],
}
OUTPUT_SEGMENTS = {
    "next_frame_code_delta": [0, 256],
    "next_raw_physiology_delta": [256, 268],
}
FRAME_CODE_SEGMENTS = {"visual": [0, 128], "body": [128, 256]}


def action_suffix_input_dim(horizon: int) -> int:
    if horizon not in ACTION_SUFFIX_HORIZONS:
        raise ValueError(f"action suffix horizon must be one of {ACTION_SUFFIX_HORIZONS}")
    return FRAME_WINDOW * FRAME_CODE_DIM + NEURAL_DIM + ACTION_DIM * (1 + horizon)


def action_suffix_input_segments(horizon: int) -> dict[str, list[int]]:
    dimension = action_suffix_input_dim(horizon)
    return {
        "frame_codes_t_minus_3_through_t": [0, 1024],
        "neural_readouts_t": [1024, 1408],
        "previous_executed_action": [1408, 1420],
        "executed_action_suffix_t_through_t_plus_h_minus_1": [1420, dimension],
    }


ACTION_SUFFIX_OUTPUT_SEGMENTS = {
    "future_four_frame_code_deltas_from_current": [0, 1024],
    "future_raw_physiology_delta": [1024, 1036],
}


@dataclass(frozen=True)
class RichPredictionConfig:
    input_dim: int = INPUT_DIM
    hidden_dim: int = 256
    output_dim: int = OUTPUT_DIM
    members: int = MEMBERS
    frame_code_dim: int = FRAME_CODE_DIM
    frame_window: int = FRAME_WINDOW
    neural_dim: int = NEURAL_DIM
    action_dim: int = ACTION_DIM
    physiology_dim: int = PHYSIOLOGY_DIM

    def __post_init__(self) -> None:
        if tuple(asdict(self).values()) != (
            INPUT_DIM,
            256,
            OUTPUT_DIM,
            MEMBERS,
            FRAME_CODE_DIM,
            FRAME_WINDOW,
            NEURAL_DIM,
            ACTION_DIM,
            PHYSIOLOGY_DIM,
        ):
            raise ValueError("rich consequence ensemble dimensions are fixed")


class RichConsequenceMember(nn.Module):
    """A single independent deterministic consequence predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.layer0 = nn.Linear(INPUT_DIM, 256)
        self.layer1 = nn.Linear(256, 256)
        self.output = nn.Linear(256, OUTPUT_DIM)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != INPUT_DIM:
            raise ValueError(f"rich predictor input must end with {INPUT_DIM}")
        value = torch.tanh(self.layer0(value))
        value = torch.tanh(self.layer1(value))
        return self.output(value)


class RichConsequenceEnsemble(nn.Module):
    """Three independently initialized members with no shared parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.config = RichPredictionConfig()
        self.members = nn.ModuleList(RichConsequenceMember() for _ in range(MEMBERS))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return member predictions as ``[...,3,268]`` normalized targets."""
        return torch.stack([member(value) for member in self.members], dim=-2)


@dataclass(frozen=True)
class ActionSuffixPredictionConfig:
    horizon: int
    hidden_dim: int = 256
    members: int = MEMBERS
    frame_code_dim: int = FRAME_CODE_DIM
    frame_window: int = FRAME_WINDOW
    neural_dim: int = NEURAL_DIM
    action_dim: int = ACTION_DIM
    physiology_dim: int = PHYSIOLOGY_DIM

    def __post_init__(self) -> None:
        action_suffix_input_dim(self.horizon)
        if tuple(asdict(self).values())[1:] != (
            256,
            MEMBERS,
            FRAME_CODE_DIM,
            FRAME_WINDOW,
            NEURAL_DIM,
            ACTION_DIM,
            PHYSIOLOGY_DIM,
        ):
            raise ValueError("action suffix consequence dimensions are fixed")

    @property
    def input_dim(self) -> int:
        return action_suffix_input_dim(self.horizon)

    @property
    def output_dim(self) -> int:
        return ACTION_SUFFIX_OUTPUT_DIM


class ActionSuffixConsequenceMember(nn.Module):
    """One independent fixed-horizon endpoint-window predictor."""

    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.config = ActionSuffixPredictionConfig(horizon)
        self.layer0 = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.layer1 = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.output = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"H{self.config.horizon} suffix input must end with {self.config.input_dim}"
            )
        value = torch.tanh(self.layer0(value))
        value = torch.tanh(self.layer1(value))
        return self.output(value)


class ActionSuffixConsequenceEnsemble(nn.Module):
    """Three independent members for one declared action-suffix horizon."""

    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.config = ActionSuffixPredictionConfig(horizon)
        self.members = nn.ModuleList(
            ActionSuffixConsequenceMember(horizon) for _ in range(MEMBERS)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(value) for member in self.members], dim=-2)


def normalized_suffix_input(
    value: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a declared suffix input and report rows crossing the clamp."""
    dimension = action_suffix_input_dim(horizon)
    if value.shape[-1] != dimension or mean.shape != (dimension,) or scale.shape != (
        dimension,
    ):
        raise ValueError("action suffix input normalization shapes differ")
    standardized = (value - mean) / scale
    clipped = torch.any(torch.abs(standardized) > NORMALIZED_INPUT_CLIP, dim=-1)
    return torch.clamp(standardized, -NORMALIZED_INPUT_CLIP, NORMALIZED_INPUT_CLIP), clipped


def denormalize_suffix_output(
    normalized: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if (
        normalized.shape[-1] != ACTION_SUFFIX_OUTPUT_DIM
        or mean.shape != (ACTION_SUFFIX_OUTPUT_DIM,)
        or scale.shape != (ACTION_SUFFIX_OUTPUT_DIM,)
    ):
        raise ValueError("action suffix output normalization shapes differ")
    return normalized * scale + mean


def suffix_ensemble_summary(
    raw_members: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_members.shape[-2:] != (MEMBERS, ACTION_SUFFIX_OUTPUT_DIM):
        raise ValueError("action suffix ensemble output shape differs")
    mean = raw_members.mean(dim=-2)
    disagreement = torch.sqrt(
        torch.mean((raw_members - mean[..., None, :]) ** 2, dim=-2)
    )
    return mean, disagreement


def normalized_input(
    value: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and clamp, returning a row mask for any clipped coordinate."""
    if value.shape[-1] != INPUT_DIM or mean.shape != (INPUT_DIM,) or scale.shape != (
        INPUT_DIM,
    ):
        raise ValueError("rich prediction input normalization shapes differ")
    standardized = (value - mean) / scale
    clipped = torch.any(torch.abs(standardized) > NORMALIZED_INPUT_CLIP, dim=-1)
    return torch.clamp(standardized, -NORMALIZED_INPUT_CLIP, NORMALIZED_INPUT_CLIP), clipped


def denormalize_output(
    normalized: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if normalized.shape[-1] != OUTPUT_DIM or mean.shape != (OUTPUT_DIM,) or scale.shape != (
        OUTPUT_DIM,
    ):
        raise ValueError("rich prediction output normalization shapes differ")
    return normalized * scale + mean


def ensemble_summary(raw_members: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw-unit ensemble mean and uncalibrated population RMS spread."""
    if raw_members.shape[-2:] != (MEMBERS, OUTPUT_DIM):
        raise ValueError("rich prediction ensemble output shape differs")
    mean = raw_members.mean(dim=-2)
    disagreement = torch.sqrt(torch.mean((raw_members - mean[..., None, :]) ** 2, dim=-2))
    return mean, disagreement


def tensor_bundle_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Hash sorted tensor names, exact little-endian float32 shapes, and bytes."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name], dtype="<f4")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0<f4\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def artifact_identity(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    """Content identity independent of NPZ container compression and timestamps."""
    clean = dict(metadata)
    clean.pop("artifact_identity", None)
    array_receipts = {
        name: {
            "dtype": np.ascontiguousarray(value).dtype.str,
            "shape": list(value.shape),
            "sha256": array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    encoded = json.dumps(
        {"metadata": clean, "arrays": array_receipts},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ACTION_DIM",
    "ACTION_SUFFIX_FORMAT",
    "ACTION_SUFFIX_HORIZONS",
    "ACTION_SUFFIX_OUTPUT_DIM",
    "ACTION_SUFFIX_OUTPUT_SEGMENTS",
    "CODE_DELTA_SCALE_FLOOR",
    "FORMAT",
    "FRAME_CODE_DIM",
    "FRAME_CODE_SEGMENTS",
    "FRAME_WINDOW",
    "INPUT_DIM",
    "INPUT_SCALE_FLOOR",
    "INPUT_SEGMENTS",
    "MEMBERS",
    "NEURAL_DIM",
    "NORMALIZED_INPUT_CLIP",
    "OUTPUT_DIM",
    "OUTPUT_SEGMENTS",
    "PHYSIOLOGY_DELTA_SCALE_FLOOR",
    "PHYSIOLOGY_DIM",
    "ActionSuffixConsequenceEnsemble",
    "ActionSuffixPredictionConfig",
    "RichConsequenceEnsemble",
    "RichPredictionConfig",
    "artifact_identity",
    "action_suffix_input_dim",
    "action_suffix_input_segments",
    "array_sha256",
    "denormalize_output",
    "ensemble_summary",
    "denormalize_suffix_output",
    "normalized_input",
    "normalized_suffix_input",
    "suffix_ensemble_summary",
    "tensor_bundle_sha256",
]
