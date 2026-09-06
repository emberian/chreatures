"""Research-only slow goal selector for joined sensorimotor development."""

from __future__ import annotations

import math

import torch
from torch import nn


class SlowGoalManager(nn.Module):
    """Select one private achieved-history key at 2 Hz.

    The current private worker state, canonical neural readouts, and local
    physiology form a query. ``policy`` scores only valid keys from the same
    resident's causal achieved-history reservoir. Zero query gain makes the
    initial selector uniform while leaving the query network trainable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(128 + 384 + 6, 128), nn.Tanh(), nn.Linear(128, 64)
        )
        self.value = nn.Sequential(
            nn.Linear(128 + 384 + 6, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        nn.init.zeros_(self.query[-1].weight)
        nn.init.zeros_(self.query[-1].bias)
        self.query_gain = nn.Parameter(torch.tensor(0.05))

    def policy(
        self,
        worker_hidden: torch.Tensor,
        neural_readouts: torch.Tensor,
        physiology: torch.Tensor,
        keys: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked goal logits ``[B,R]`` and manager values ``[B]``."""
        batch = worker_hidden.shape[0]
        if worker_hidden.shape != (batch, 128):
            raise ValueError("worker hidden must be [B,128]")
        if neural_readouts.shape != (batch, 384) or physiology.shape != (batch, 6):
            raise ValueError("manager inputs must be neural [B,384], physiology [B,6]")
        if keys.ndim != 3 or keys.shape[0] != batch or keys.shape[2] != 64:
            raise ValueError("private achieved keys must be [B,R,64]")
        if valid.shape != keys.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("goal mask must be bool [B,R]")
        if not torch.all(valid.any(1)):
            raise ValueError("every resident must have an achieved goal")
        context = torch.cat((worker_hidden, neural_readouts, physiology), dim=-1)
        query = self.query(context)
        logits = torch.einsum("bd,brd->br", query, keys) / math.sqrt(64)
        logits = logits * self.query_gain
        logits = logits.masked_fill(~valid, -torch.inf)
        return logits, self.value(context).squeeze(-1)
