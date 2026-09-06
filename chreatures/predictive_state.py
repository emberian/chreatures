"""Trainable action-conditioned recurrent predictive state from local experience."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


ACTIONS = ("thrust", "yaw", "gaze_pitch", "grip", "signal_low",
           "signal_mid", "signal_high", "posture")
PHYSIOLOGY = ("energy", "gut", "fatigue", "speed_local", "angular_local", "support")
FORMAT = "chreatures-predictive-state-v1"


@dataclass(frozen=True)
class PredictiveStateConfig:
    feature_dim: int = 384
    physiology_dim: int = 6
    action_dim: int = 8
    latent_dim: int = 96
    encoder_dim: int = 128
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    learning_rate: float = 3e-4
    consistency_rate: float = 0.08
    max_grad_norm: float = 1.0
    seed: int = 20260905

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(self.horizons))
        if min(self.feature_dim, self.physiology_dim, self.action_dim,
               self.latent_dim, self.encoder_dim) < 1:
            raise ValueError("predictive-state dimensions must be positive")
        if tuple(sorted(set(self.horizons))) != self.horizons or self.horizons[0] < 1:
            raise ValueError("horizons must be unique increasing positive integers")


@dataclass(frozen=True)
class PredictiveSequence:
    """Time-major anonymous experience; action[t] precedes observation[t+1]."""
    features: np.ndarray
    physiology: np.ndarray
    actions: np.ndarray
    reset: np.ndarray
    valid: np.ndarray

    @classmethod
    def from_rollout(cls, path: str | Path) -> "PredictiveSequence":
        with np.load(path, allow_pickle=False) as value:
            features = np.asarray(value["features"], np.float32)
            physiology = np.asarray(value["physiology"], np.float32)
            actions = np.asarray(value["action"], np.float32)
            done = np.asarray(value["done"], bool)
        reset = np.zeros(done.shape, bool)
        reset[0] = True
        reset[1:] = done[:-1]
        return cls(features, physiology, actions, reset, np.ones(done.shape, bool)).validated()

    @classmethod
    def load(cls, path: str | Path) -> "PredictiveSequence":
        with np.load(path, allow_pickle=False) as value:
            if str(value["format"]) != FORMAT:
                raise ValueError("incompatible predictive sequence format")
            if value["action_names"].astype(str).tolist() != list(ACTIONS):
                raise ValueError("predictive sequence action order differs")
            if value["physiology_names"].astype(str).tolist() != list(PHYSIOLOGY):
                raise ValueError("predictive sequence physiology order differs")
            result = cls(*(np.asarray(value[name]) for name in
                           ("features", "physiology", "actions", "reset", "valid")))
        return result.validated()

    def validated(self) -> "PredictiveSequence":
        if self.features.ndim != 3 or self.physiology.ndim != 3 or self.actions.ndim != 3:
            raise ValueError("sequence arrays must be [time,resident,channel]")
        leading = self.features.shape[:2]
        if self.physiology.shape[:2] != leading or self.actions.shape[:2] != leading:
            raise ValueError("sequence time/resident axes differ")
        if self.reset.shape != leading or self.valid.shape != leading:
            raise ValueError("reset and valid must be [time,resident]")
        for value in (self.features, self.physiology, self.actions):
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("sequence channels must be finite float32")
        if not np.all(self.reset[0] | ~self.valid[0]):
            raise ValueError("every valid stream must reset at its first row")
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, format=np.asarray(FORMAT), features=self.features,
            physiology=self.physiology, actions=self.actions, reset=self.reset,
            valid=self.valid, action_names=np.asarray(ACTIONS),
            physiology_names=np.asarray(PHYSIOLOGY))


class PredictiveStateModel(nn.Module):
    """Observed posterior state plus action-only recurrent future dynamics."""
    def __init__(self, config: PredictiveStateConfig) -> None:
        super().__init__(); self.config = config; torch.manual_seed(config.seed)
        observation_dim = config.feature_dim + config.physiology_dim
        self.observation_encoder = nn.Sequential(nn.Linear(observation_dim, config.encoder_dim), nn.Tanh())
        self.action_encoder = nn.Sequential(nn.Linear(config.action_dim, config.encoder_dim // 2), nn.Tanh())
        self.observe_cell = nn.GRUCell(config.encoder_dim + config.action_dim, config.latent_dim)
        self.transition_cell = nn.GRUCell(config.encoder_dim // 2, config.latent_dim)
        self.decoder_mean = nn.Linear(config.latent_dim, observation_dim)
        self.decoder_log_std = nn.Linear(config.latent_dim, observation_dim)
        nn.init.zeros_(self.decoder_log_std.weight)
        nn.init.zeros_(self.decoder_log_std.bias)

    def observe(self, state: torch.Tensor, observation: torch.Tensor,
                previous_action: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
        state = torch.where(reset[:, None], torch.zeros_like(state), state)
        return self.observe_cell(torch.cat((self.observation_encoder(observation), previous_action), -1), state)

    def transition(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition_cell(self.action_encoder(action), state)

    def prediction(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.decoder_mean(state)
        raw = self.decoder_log_std(state)
        log_std = -0.5 + torch.tanh(raw)
        return mean, log_std.clamp(-1.5, 0.5)

    @torch.no_grad()
    def imagine(self, state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Roll action-conditioned futures; outputs are never training evidence."""
        means, residual_scales, support = [], [], []
        maximum = float(self.config.horizons[-1])
        for index, action in enumerate(actions):
            state = self.transition(state, action)
            mean, log_std = self.prediction(state)
            means.append(mean); residual_scales.append(log_std.exp())
            support.append(torch.full(mean.shape[:-1], np.exp(-(index + 1) / maximum),
                                      dtype=mean.dtype, device=mean.device))
        return {"mean": torch.stack(means), "residual_scale": torch.stack(residual_scales),
                "horizon_support": torch.stack(support), "final_state": state}


class PredictiveStateTrainer:
    VERSION = 1
    def __init__(self, config: PredictiveStateConfig | None = None,
                 *, device: str | torch.device = "cpu") -> None:
        self.config = config or PredictiveStateConfig(); self.device = torch.device(device)
        self.model = PredictiveStateModel(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.update_count = 0; self.rng = np.random.default_rng(self.config.seed + 1)

    def _tensors(self, sequence: PredictiveSequence):
        sequence.validated(); c = self.config
        if sequence.features.shape[2] != c.feature_dim or sequence.physiology.shape[2] != c.physiology_dim or sequence.actions.shape[2] != c.action_dim:
            raise ValueError("sequence channel dimensions differ from model config")
        observation = np.concatenate((sequence.features, sequence.physiology), axis=-1)
        return tuple(torch.as_tensor(x, device=self.device) for x in
                     (observation, sequence.actions, sequence.reset, sequence.valid))

    def loss(self, sequence: PredictiveSequence) -> tuple[torch.Tensor, dict[str, float]]:
        obs, actions, resets, valid = self._tensors(sequence)
        time, residents, _ = obs.shape
        state = torch.zeros((residents, self.config.latent_dim), device=self.device)
        posterior = []
        zero_action = torch.zeros_like(actions[0])
        for index in range(time):
            previous = zero_action if index == 0 else actions[index - 1]
            state = self.model.observe(state, obs[index], previous, resets[index] | ~valid[index])
            posterior.append(state)
        posterior_tensor = torch.stack(posterior)
        nll_terms, consistency_terms = [], []
        by_horizon: dict[int, list[torch.Tensor]] = {h: [] for h in self.config.horizons}
        maximum = self.config.horizons[-1]
        for start in range(time - 1):
            imagined = posterior_tensor[start]
            alive = valid[start].clone()
            for distance in range(1, min(maximum, time - start - 1) + 1):
                imagined = self.model.transition(imagined, actions[start + distance - 1])
                target_index = start + distance
                alive = alive & valid[target_index] & ~resets[target_index]
                if distance not in by_horizon or not bool(alive.any()): continue
                mean, log_std = self.model.prediction(imagined)
                error = (obs[target_index] - mean) * torch.exp(-log_std)
                per_row = (0.5 * error.square() + log_std).mean(-1)
                nll_terms.append(per_row[alive]); by_horizon[distance].append(per_row[alive].detach())
                consistency_terms.append((imagined - posterior_tensor[target_index].detach()).square().mean(-1)[alive])
        if not nll_terms:
            raise ValueError("sequence has no valid future targets")
        nll = torch.cat(nll_terms).mean()
        consistency = torch.cat(consistency_terms).mean()
        loss = nll + self.config.consistency_rate * consistency
        metrics = {f"nll_h{h}": float(torch.cat(values).mean()) for h, values in by_horizon.items() if values}
        metrics.update({"nll": float(nll.detach()), "consistency": float(consistency.detach()),
                        "targets": float(sum(x.numel() for x in nll_terms))})
        return loss, metrics

    def update(self, sequence: PredictiveSequence) -> dict[str, float]:
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(sequence); loss.backward()
        gradient = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step(); self.update_count += 1
        metrics.update({"loss": float(loss.detach()), "gradient_norm": float(gradient),
                        "update": float(self.update_count)})
        return metrics

    @torch.no_grad()
    def encode(self, features: np.ndarray, physiology: np.ndarray, actions: np.ndarray,
               reset: np.ndarray) -> np.ndarray:
        sequence = PredictiveSequence(np.asarray(features, np.float32), np.asarray(physiology, np.float32),
            np.asarray(actions, np.float32), np.asarray(reset, bool), np.ones(reset.shape, bool))
        obs, act, rst, valid = self._tensors(sequence); state = torch.zeros((obs.shape[1], self.config.latent_dim), device=self.device)
        output=[]; zero=torch.zeros_like(act[0])
        for t in range(obs.shape[0]):
            state=self.model.observe(state,obs[t],zero if t==0 else act[t-1],rst[t]|~valid[t]); output.append(state)
        return torch.stack(output).cpu().numpy()

    def checkpoint(self, path: str | Path) -> dict[str, Any]:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        value={"format":FORMAT,"version":self.VERSION,"config":asdict(self.config),"model":self.model.state_dict(),
               "optimizer":self.optimizer.state_dict(),"update_count":self.update_count,
               "numpy_rng":self.rng.bit_generator.state,"torch_rng":torch.get_rng_state(),
               "device_rng":torch.cuda.get_rng_state(self.device) if self.device.type=="cuda" else None}
        temporary=path.with_name(path.name+".tmp"); torch.save(value,temporary); temporary.replace(path)
        return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}

    @classmethod
    def restore(cls,path: str|Path,*,device: str|torch.device="cpu") -> "PredictiveStateTrainer":
        value=torch.load(path,map_location=device,weights_only=False)
        if value.get("format")!=FORMAT or value.get("version")!=cls.VERSION: raise ValueError("incompatible predictive-state checkpoint")
        instance=cls(PredictiveStateConfig(**value["config"]),device=device); instance.model.load_state_dict(value["model"])
        instance.optimizer.load_state_dict(value["optimizer"]); instance.update_count=int(value["update_count"])
        instance.rng.bit_generator.state=value["numpy_rng"]; torch.set_rng_state(value["torch_rng"].cpu())
        if instance.device.type=="cuda" and value.get("device_rng") is not None:
            torch.cuda.set_rng_state(value["device_rng"].cpu(),instance.device)
        return instance

    def export(self,path: str|Path) -> dict[str,Any]:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        arrays={name:value.detach().cpu().contiguous().numpy().astype(np.float32,copy=False)
                for name,value in self.model.state_dict().items()}
        tensor_manifest={name:{"shape":list(value.shape),"dtype":"float32",
            "sha256":hashlib.sha256(value.tobytes(order="C")).hexdigest()}
            for name,value in arrays.items()}
        metadata={"format":FORMAT,"version":self.VERSION,"config":asdict(self.config),"actions":list(ACTIONS),
                  "physiology":list(PHYSIOLOGY),"tensor_layout":"row-major-f32","updates":self.update_count,
                  "feature_layout":"neural_feature_000..feature_dim-1 then named physiology",
                  "gru_gate_order":"reset,update,new (PyTorch GRUCell r,z,n)",
                  "tensors":tensor_manifest,
                  "scope":"immutable shared observation encoder, recurrent cells, transition and uncertainty decoder"}
        arrays["metadata"]=np.asarray(json.dumps(metadata,sort_keys=True)); np.savez(path,**arrays)
        return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"metadata":metadata}
