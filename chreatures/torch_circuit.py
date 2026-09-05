"""Batched PyTorch execution of the measured recurrent circuit.

PyTorch calls ROCm devices ``cuda``.  This module intentionally mirrors the
neural, adaptation, support, decoder, eligibility, and value updates in
``Brain.step`` while leaving world interaction and action selection to the
ordinary runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .brain import CHANNELS, Connectome, Genome

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CircuitStep:
    decoded: torch.Tensor
    prediction: torch.Tensor
    odor: torch.Tensor


class TorchCircuit:
    """Independent brains sharing one immutable sparse connectome."""

    def __init__(
        self,
        graph: Connectome,
        batch_size: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        genome: Genome | dict | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError("dtype must be float16, float32, or float64")
        self.graph = graph
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.genome = genome if isinstance(genome, Genome) else Genome(**(genome or {}))

        coo = graph.matrix.tocoo()
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
        ).to(self.device)
        values = torch.from_numpy(coo.data).to(device=self.device, dtype=dtype)
        self.matrix = torch.sparse_coo_tensor(
            indices, values, coo.shape, device=self.device, dtype=dtype
        ).coalesce()
        self.input_map = torch.as_tensor(
            graph.input_map, device=self.device, dtype=dtype
        )
        self.output_cells = torch.as_tensor(
            graph.output_cells, device=self.device, dtype=torch.long
        )
        self.baseline = torch.as_tensor(
            graph.baseline, device=self.device, dtype=dtype
        )
        self.output_baseline = self.baseline.index_select(0, self.output_cells)
        self.decoder = torch.as_tensor(
            graph.decoder, device=self.device, dtype=dtype
        )
        self.reset()

    def reset(self) -> None:
        """Reset all private state while keeping the shared graph tensors."""
        shape = (self.batch_size, self.graph.n)
        self.rates = self.baseline.expand(self.batch_size, -1).clone()
        self.adaptation = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.support = torch.ones(shape, device=self.device, dtype=self.dtype)
        self.context = torch.zeros(
            (self.batch_size, len(CHANNELS)), device=self.device, dtype=self.dtype
        )
        self.eligibility = torch.zeros(
            (self.batch_size, 3), device=self.device, dtype=self.dtype
        )
        self.values = torch.tensor(
            [0.22, 0.22, 0.12], device=self.device, dtype=self.dtype
        ).expand(self.batch_size, -1).clone()
        self.sound_memory = torch.zeros(
            (self.batch_size, 3, 3), device=self.device, dtype=self.dtype
        )
        self.sound_trace = torch.zeros(
            (self.batch_size, 3), device=self.device, dtype=self.dtype
        )
        self.modulator = torch.zeros(
            self.batch_size, device=self.device, dtype=self.dtype
        )
        self.total_nutrition = torch.zeros(
            self.batch_size, device=self.device, dtype=self.dtype
        )
        self.last_decoded = torch.zeros(
            (self.batch_size, len(CHANNELS)), device=self.device, dtype=self.dtype
        )
        self.last_prediction = torch.zeros(
            self.batch_size, device=self.device, dtype=self.dtype
        )
        self.time = 0.0

    def reset_dynamics(self) -> None:
        """Reset transient circuit state while preserving learned associations."""
        self.rates.copy_(self.baseline.expand(self.batch_size, -1))
        self.adaptation.zero_()
        self.support.fill_(1)
        self.context.zero_()
        self.eligibility.zero_()
        self.sound_trace.zero_()
        self.modulator.zero_()
        self.last_decoded.zero_()
        self.last_prediction.zero_()

    @torch.no_grad()
    def step(
        self,
        encoded: torch.Tensor | NDArray[np.floating],
        dt: float,
        reward: torch.Tensor | NDArray[np.floating] | float = 0.0,
        *,
        learning: bool = True,
        silenced: bool = False,
        validate: bool = True,
    ) -> CircuitStep:
        """Advance the same circuit and associative updates as ``Brain.step``."""
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("Invalid neural timestep")
        encoded_tensor = torch.as_tensor(encoded, device=self.device, dtype=self.dtype)
        if encoded_tensor.shape != (self.batch_size, len(CHANNELS)):
            raise ValueError(
                f"encoded must have shape {(self.batch_size, len(CHANNELS))}"
            )
        if validate and not bool(torch.isfinite(encoded_tensor).all()):
            raise ValueError("Non-finite sensation")
        encoded_tensor = encoded_tensor.clamp(0, 1)
        reward_tensor = torch.as_tensor(reward, device=self.device, dtype=self.dtype)
        if reward_tensor.ndim == 0:
            reward_tensor = reward_tensor.expand(self.batch_size)
        if reward_tensor.shape != (self.batch_size,):
            raise ValueError(f"reward must be finite with shape {(self.batch_size,)}")
        if validate and not bool(torch.isfinite(reward_tensor).all()):
            raise ValueError("reward must be finite")

        self.time += dt
        drive = encoded_tensor @ self.input_map.T
        alpha = min(1.0, dt / 2 / self.genome.neural_tau)
        for _ in range(2):
            if silenced:
                recurrent = torch.zeros_like(self.rates)
            else:
                recurrent = torch.sparse.mm(self.matrix, self.rates.T).T
            target = torch.relu(
                torch.tanh(
                    0.005
                    + drive
                    + self.genome.neural_gain * recurrent
                    - 0.10 * self.adaptation
                )
            )
            self.rates.add_(alpha * (target * self.support - self.rates))

        self.adaptation.add_(dt / 5 * (self.rates - self.adaptation))
        self.support.add_(
            dt
            * (
                self.genome.support_recovery * (1 - self.support)
                - 0.003 * self.rates
            )
        ).clamp_(0.65, 1)
        output_rates = self.rates.index_select(1, self.output_cells)
        decoded = ((output_rates - self.output_baseline) @ self.decoder).clamp(0, 1)
        self.last_decoded.copy_(decoded)
        self.context.add_(dt / 3.0 * (decoded - self.context))
        odor = decoded[:, :6].reshape(self.batch_size, 2, 3)
        odor_mean = odor.mean(dim=1)
        self.eligibility.mul_(float(np.exp(-dt / 4))).add_(odor_mean * (dt / 4))
        self.sound_trace.mul_(float(np.exp(-dt / 3))).add_(
            decoded[:, 11:14] * (dt / 3)
        )
        prediction = (self.values * odor_mean).sum(dim=1)
        self.last_prediction.copy_(prediction)

        positive = reward_tensor.clamp_min(0)
        target_modulator = (positive / dt * 30).clamp_max(1)
        self.modulator.add_(dt / 0.8 * (target_modulator - self.modulator))
        self.total_nutrition.add_(positive)
        if learning:
            positive_mask = positive > 0
            increment = (
                self.genome.learning_rate
                * positive[:, None]
                * 120
                * self.eligibility
                * (1 - self.values)
            )
            self.values.add_(increment * positive_mask[:, None])
            association = self.sound_trace[:, :, None] * self.eligibility[:, None, :]
            self.sound_memory.add_(
                positive[:, None, None]
                * 25
                * association
                * positive_mask[:, None, None]
            )
            no_reward = ~positive_mask
            extinction = (
                dt
                * 0.0008
                * self.eligibility
                * torch.clamp_min(self.values - 0.12, 0)
            )
            self.values.sub_(extinction * no_reward[:, None])
            self.values.clamp_(0.05, 1)
            self.sound_memory.clamp_(0, 1)

        return CircuitStep(decoded=decoded, prediction=prediction, odor=odor)

    def learned_state(self) -> dict[str, torch.Tensor]:
        return {
            "values": self.values.clone(),
            "sound_memory": self.sound_memory.clone(),
            "total_nutrition": self.total_nutrition.clone(),
        }

    def load_learned_state(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("values", "sound_memory", "total_nutrition"):
            target = getattr(self, name)
            source = torch.as_tensor(state[name], device=self.device, dtype=self.dtype)
            if source.shape != target.shape or not bool(torch.isfinite(source).all()):
                raise ValueError(f"Invalid learned state: {name}")
            target.copy_(source)
