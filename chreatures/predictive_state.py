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


ACTIONS = (
    "thrust",
    "yaw",
    "gaze_pitch",
    "grip",
    "signal_low",
    "signal_mid",
    "signal_high",
    "posture",
)
PHYSIOLOGY = ("energy", "gut", "fatigue", "speed_local", "angular_local", "support")
FORMAT = "chreatures-predictive-state-v1"


@dataclass(frozen=True)
class PredictiveStateConfig:
    feature_dim: int = 384
    physiology_dim: int = 6
    action_dim: int = 8
    latent_dim: int = 96
    encoder_dim: int = 128
    horizons: tuple[int, ...] = tuple(range(1, 9))
    learning_rate: float = 3e-4
    consistency_rate: float = 0.08
    max_grad_norm: float = 1.0
    seed: int = 20260905

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(self.horizons))
        if (
            min(
                self.feature_dim,
                self.physiology_dim,
                self.action_dim,
                self.latent_dim,
                self.encoder_dim,
            )
            < 1
        ):
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
    def from_episode(cls, directory: str | Path, split: str) -> "PredictiveSequence":
        if split not in {"train", "holdout"}:
            raise ValueError("split must be train or holdout")
        path = Path(directory) / split
        features = np.load(path / "features.npy", mmap_mode="r")
        physiology = np.load(path / "physiology.npy", mmap_mode="r")
        actions = np.load(path / "action.npy", mmap_mode="r")
        reset = np.load(path / "reset.npy", mmap_mode="r")
        valid = np.load(path / "valid.npy", mmap_mode="r")
        terminal_features = np.load(path / "terminal_features.npy", mmap_mode="r")
        terminal_physiology = np.load(path / "terminal_physiology.npy", mmap_mode="r")
        features = np.concatenate((features, terminal_features[None]), axis=0).astype(
            np.float32
        )
        physiology = np.concatenate(
            (physiology, terminal_physiology[None]), axis=0
        ).astype(np.float32)
        actions = np.concatenate((actions, np.zeros_like(actions[:1])), axis=0).astype(
            np.float32
        )
        reset = np.concatenate((reset, np.zeros_like(reset[:1])), axis=0).astype(bool)
        valid = np.concatenate((valid, np.ones_like(valid[:1])), axis=0).astype(bool)
        return cls(features, physiology, actions, reset, valid).validated()

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
        return cls(
            features, physiology, actions, reset, np.ones(done.shape, bool)
        ).validated()

    @classmethod
    def load(cls, path: str | Path) -> "PredictiveSequence":
        with np.load(path, allow_pickle=False) as value:
            if str(value["format"]) != FORMAT:
                raise ValueError("incompatible predictive sequence format")
            if value["action_names"].astype(str).tolist() != list(ACTIONS):
                raise ValueError("predictive sequence action order differs")
            if value["physiology_names"].astype(str).tolist() != list(PHYSIOLOGY):
                raise ValueError("predictive sequence physiology order differs")
            result = cls(
                *(
                    np.asarray(value[name])
                    for name in ("features", "physiology", "actions", "reset", "valid")
                )
            )
        return result.validated()

    def validated(self) -> "PredictiveSequence":
        if (
            self.features.ndim != 3
            or self.physiology.ndim != 3
            or self.actions.ndim != 3
        ):
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format=np.asarray(FORMAT),
            features=self.features,
            physiology=self.physiology,
            actions=self.actions,
            reset=self.reset,
            valid=self.valid,
            action_names=np.asarray(ACTIONS),
            physiology_names=np.asarray(PHYSIOLOGY),
        )


@dataclass(frozen=True)
class PredictiveNormalizer:
    input_mean: np.ndarray
    input_scale: np.ndarray
    physiology_delta_mean: np.ndarray
    physiology_delta_scale: np.ndarray
    physiology_delta_count: np.ndarray
    identity: dict[str, Any]

    @classmethod
    def fit(
        cls, sequences: Sequence[PredictiveSequence], identity: dict[str, Any]
    ) -> "PredictiveNormalizer":
        observations = []
        deltas = [[] for _ in range(8)]
        for sequence in sequences:
            sequence.validated()
            observation = np.concatenate(
                (sequence.features, sequence.physiology), axis=-1
            )
            observations.append(observation[sequence.valid])
            for h in range(1, 9):
                mask = sequence.valid[:-h] & sequence.valid[h:]
                for offset in range(1, h + 1):
                    mask &= ~sequence.reset[offset : len(sequence.reset) - h + offset]
                deltas[h - 1].append(
                    (sequence.physiology[h:] - sequence.physiology[:-h])[mask]
                )
        rows = np.concatenate(observations).astype(np.float64)
        mean = rows.mean(0)
        scale = np.maximum(rows.std(0), 1e-6)
        delta_mean, delta_scale, count = [], [], []
        for values in deltas:
            joined = np.concatenate(values).astype(np.float64)
            delta_mean.append(joined.mean(0))
            delta_scale.append(np.maximum(joined.std(0), 1e-6))
            count.append(len(joined))
        return cls(
            mean.astype(np.float32),
            scale.astype(np.float32),
            np.asarray(delta_mean, np.float32),
            np.asarray(delta_scale, np.float32),
            np.asarray(count, np.int64),
            identity,
        )

    def metadata(self) -> dict[str, Any]:
        arrays = (
            self.input_mean,
            self.input_scale,
            self.physiology_delta_mean,
            self.physiology_delta_scale,
            self.physiology_delta_count,
        )
        digest = hashlib.sha256(
            b"".join(np.ascontiguousarray(x).view(np.uint8) for x in arrays)
        ).hexdigest()
        return {
            "format": "chreatures-predictive-normalizer-v1",
            "sha256": digest,
            "fit_scope": self.identity,
            "horizons": list(range(1, 9)),
            "physiology_delta_counts": self.physiology_delta_count.tolist(),
        }


class PredictiveStateModel(nn.Module):
    """Observed posterior state plus action-only recurrent future dynamics."""

    def __init__(self, config: PredictiveStateConfig) -> None:
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)
        observation_dim = config.feature_dim + config.physiology_dim
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim, config.encoder_dim), nn.Tanh()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.encoder_dim // 2), nn.Tanh()
        )
        self.observe_rnn = nn.GRU(
            config.encoder_dim + config.action_dim, config.latent_dim
        )
        self.transition_cell = nn.GRUCell(config.encoder_dim // 2, config.latent_dim)
        self.feature_mean = nn.Linear(config.latent_dim, config.feature_dim)
        self.feature_log_std = nn.Linear(config.latent_dim, config.feature_dim)
        self.physiology_delta_mean = nn.Linear(config.latent_dim, config.physiology_dim)
        self.physiology_delta_log_std = nn.Linear(
            config.latent_dim, config.physiology_dim
        )
        for head in (self.feature_log_std, self.physiology_delta_log_std):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def observe(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        previous_action: torch.Tensor,
        reset: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.where(reset[:, None], torch.zeros_like(state), state)
        output, _ = self.observe_rnn(
            torch.cat(
                (self.observation_encoder(observation), previous_action), -1
            ).unsqueeze(0),
            state.unsqueeze(0),
        )
        return output[0]

    def transition(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition_cell(self.action_encoder(action), state)

    def prediction(self, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feature_log_std = (-0.5 + torch.tanh(self.feature_log_std(state))).clamp(
            -1.5, 0.5
        )
        physiology_log_std = (
            -0.5 + torch.tanh(self.physiology_delta_log_std(state))
        ).clamp(-1.5, 0.5)
        return (
            self.feature_mean(state),
            feature_log_std,
            self.physiology_delta_mean(state),
            physiology_log_std,
        )

    @torch.no_grad()
    def imagine(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Roll action-conditioned futures; outputs are never training evidence."""
        feature_means, feature_scales, physiology_means, physiology_scales, support = (
            [],
            [],
            [],
            [],
            [],
        )
        maximum = float(self.config.horizons[-1])
        for index, action in enumerate(actions):
            state = self.transition(state, action)
            feature_mean, feature_log_std, physiology_mean, physiology_log_std = (
                self.prediction(state)
            )
            feature_means.append(feature_mean)
            feature_scales.append(feature_log_std.exp())
            physiology_means.append(physiology_mean)
            physiology_scales.append(physiology_log_std.exp())
            support.append(
                torch.full(
                    feature_mean.shape[:-1],
                    np.exp(-(index + 1) / maximum),
                    dtype=feature_mean.dtype,
                    device=feature_mean.device,
                )
            )
        return {
            "feature_mean": torch.stack(feature_means),
            "feature_residual_scale": torch.stack(feature_scales),
            "physiology_delta_mean": torch.stack(physiology_means),
            "physiology_delta_residual_scale": torch.stack(physiology_scales),
            "horizon_support": torch.stack(support),
            "final_state": state,
        }


class PredictiveStateTrainer:
    VERSION = 1

    def __init__(
        self,
        config: PredictiveStateConfig,
        normalizer: PredictiveNormalizer,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.model = PredictiveStateModel(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.update_count = 0
        self.rng = np.random.default_rng(self.config.seed + 1)

    @torch.no_grad()
    def imagine_physical(
        self,
        state: torch.Tensor,
        physiology_anchor: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if len(actions) > 8:
            raise ValueError("imagination exceeds trained horizon 8")
        predicted = self.model.imagine(state, actions)
        mean = torch.as_tensor(self.normalizer.input_mean, device=state.device)
        scale = torch.as_tensor(self.normalizer.input_scale, device=state.device)
        delta_mean = torch.as_tensor(
            self.normalizer.physiology_delta_mean, device=state.device
        )
        delta_scale = torch.as_tensor(
            self.normalizer.physiology_delta_scale, device=state.device
        )
        time = len(actions)
        feature_dim = self.config.feature_dim
        feature_mean = (
            mean[:feature_dim] + scale[:feature_dim] * predicted["feature_mean"]
        )
        feature_scale = scale[:feature_dim] * predicted["feature_residual_scale"]
        physiology_mean = (
            physiology_anchor[None]
            + delta_mean[:time, None]
            + delta_scale[:time, None] * predicted["physiology_delta_mean"]
        )
        physiology_scale = (
            delta_scale[:time, None] * predicted["physiology_delta_residual_scale"]
        )
        valid = (
            torch.isfinite(feature_mean).all(-1)
            & torch.isfinite(feature_scale).all(-1)
            & torch.isfinite(physiology_mean).all(-1)
            & torch.isfinite(physiology_scale).all(-1)
        )
        return {
            "feature_mean": feature_mean,
            "feature_residual_scale": feature_scale,
            "physiology_mean": physiology_mean,
            "physiology_residual_scale": physiology_scale,
            "valid": valid,
            "horizon_support": predicted["horizon_support"],
            "final_state": predicted["final_state"],
        }

    def _tensors(self, sequence: PredictiveSequence):
        sequence.validated()
        c = self.config
        if (
            sequence.features.shape[2] != c.feature_dim
            or sequence.physiology.shape[2] != c.physiology_dim
            or sequence.actions.shape[2] != c.action_dim
        ):
            raise ValueError("sequence channel dimensions differ from model config")
        observation = np.concatenate((sequence.features, sequence.physiology), axis=-1)
        observation = (
            observation - self.normalizer.input_mean
        ) / self.normalizer.input_scale
        return tuple(
            torch.as_tensor(x, device=self.device)
            for x in (observation, sequence.actions, sequence.reset, sequence.valid)
        )

    def loss(
        self, sequence: PredictiveSequence, *, loss_start: int = 0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        obs, actions, resets, valid = self._tensors(sequence)
        time, residents, _ = obs.shape
        state = torch.zeros((1, residents, self.config.latent_dim), device=self.device)
        zero_action = torch.zeros_like(actions[0])
        previous = torch.cat((zero_action[None], actions[:-1]), dim=0)
        encoded = self.model.observation_encoder(obs)
        recurrent_input = torch.cat((encoded, previous), dim=-1)
        boundaries = sorted(
            set(
                [0, time]
                + torch.nonzero((resets | ~valid).any(1), as_tuple=False)
                .flatten()
                .cpu()
                .tolist()
            )
        )
        chunks = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            mask = resets[start] | ~valid[start]
            state = torch.where(mask[None, :, None], torch.zeros_like(state), state)
            output, state = self.model.observe_rnn(recurrent_input[start:stop], state)
            chunks.append(output)
        posterior_tensor = torch.cat(chunks)
        imagined = posterior_tensor[:-1].reshape(-1, self.config.latent_dim)
        alive = valid[:-1].clone()
        alive[:loss_start] = False
        nll_terms, consistency_terms = [], []
        feature_by_horizon, physiology_by_horizon = {}, {}
        raw_physiology = torch.as_tensor(sequence.physiology, device=self.device)
        delta_mean = torch.as_tensor(
            self.normalizer.physiology_delta_mean, device=self.device
        )
        delta_scale = torch.as_tensor(
            self.normalizer.physiology_delta_scale, device=self.device
        )
        for distance in self.config.horizons:
            usable = time - distance
            shifted_actions = torch.zeros_like(actions[:-1])
            shifted_actions[:usable] = actions[distance - 1 : time - 1]
            imagined = self.model.transition(
                imagined, shifted_actions.reshape(-1, self.config.action_dim)
            )
            alive[:usable] &= valid[distance:] & ~resets[distance:]
            alive[usable:] = False
            selected = alive[:usable].clone()
            feature_mean, feature_log_std, physiology_mean, physiology_log_std = (
                self.model.prediction(
                    imagined.reshape(time - 1, residents, -1)[:usable]
                )
            )
            feature_error = (
                obs[distance:, :, : self.config.feature_dim] - feature_mean
            ) * torch.exp(-feature_log_std)
            physiology_target = (
                raw_physiology[distance:]
                - raw_physiology[:-distance]
                - delta_mean[distance - 1]
            ) / delta_scale[distance - 1]
            physiology_error = (physiology_target - physiology_mean) * torch.exp(
                -physiology_log_std
            )
            feature_nll = (0.5 * feature_error.square() + feature_log_std).mean(-1)[
                selected
            ]
            physiology_nll = (
                0.5 * physiology_error.square() + physiology_log_std
            ).mean(-1)[selected]
            balanced = 0.5 * feature_nll + 0.5 * physiology_nll
            nll_terms.append(balanced)
            feature_by_horizon[distance] = feature_nll.detach()
            physiology_by_horizon[distance] = physiology_nll.detach()
            consistency_terms.append(
                (
                    imagined.reshape(time - 1, residents, -1)[:usable]
                    - posterior_tensor[distance:].detach()
                )
                .square()
                .mean(-1)[selected]
            )
        if not nll_terms:
            raise ValueError("sequence has no valid future targets")
        nll = torch.cat(nll_terms).mean()
        consistency = torch.cat(consistency_terms).mean()
        loss = nll + self.config.consistency_rate * consistency
        metrics = {
            f"feature_nll_h{h}": float(v.mean()) for h, v in feature_by_horizon.items()
        }
        metrics.update(
            {
                f"physiology_delta_nll_h{h}": float(v.mean())
                for h, v in physiology_by_horizon.items()
            }
        )
        metrics.update(
            {
                "nll": float(nll.detach()),
                "consistency": float(consistency.detach()),
                "targets": float(sum(x.numel() for x in nll_terms)),
            }
        )
        return loss, metrics

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
        self.update_count += 1
        metrics.update(
            {
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient),
                "update": float(self.update_count),
            }
        )
        return metrics

    @torch.no_grad()
    def encode(
        self,
        features: np.ndarray,
        physiology: np.ndarray,
        actions: np.ndarray,
        reset: np.ndarray,
    ) -> np.ndarray:
        sequence = PredictiveSequence(
            np.asarray(features, np.float32),
            np.asarray(physiology, np.float32),
            np.asarray(actions, np.float32),
            np.asarray(reset, bool),
            np.ones(reset.shape, bool),
        )
        obs, act, rst, valid = self._tensors(sequence)
        state = torch.zeros((obs.shape[1], self.config.latent_dim), device=self.device)
        output = []
        zero = torch.zeros_like(act[0])
        for t in range(obs.shape[0]):
            state = self.model.observe(
                state, obs[t], zero if t == 0 else act[t - 1], rst[t] | ~valid[t]
            )
            output.append(state)
        return torch.stack(output).cpu().numpy()

    def checkpoint(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "format": FORMAT,
            "version": self.VERSION,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "numpy_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "device_rng": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
            "normalizer": {
                "input_mean": self.normalizer.input_mean,
                "input_scale": self.normalizer.input_scale,
                "physiology_delta_mean": self.normalizer.physiology_delta_mean,
                "physiology_delta_scale": self.normalizer.physiology_delta_scale,
                "physiology_delta_count": self.normalizer.physiology_delta_count,
                "identity": self.normalizer.identity,
            },
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(value, temporary)
        temporary.replace(path)
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    @classmethod
    def restore(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> "PredictiveStateTrainer":
        value = torch.load(path, map_location=device, weights_only=False)
        if value.get("format") != FORMAT or value.get("version") != cls.VERSION:
            raise ValueError("incompatible predictive-state checkpoint")
        item = value["normalizer"]
        normalizer = PredictiveNormalizer(
            *(
                np.asarray(item[name])
                for name in (
                    "input_mean",
                    "input_scale",
                    "physiology_delta_mean",
                    "physiology_delta_scale",
                    "physiology_delta_count",
                )
            ),
            dict(item["identity"]),
        )
        instance = cls(
            PredictiveStateConfig(**value["config"]), normalizer, device=device
        )
        instance.model.load_state_dict(value["model"])
        instance.optimizer.load_state_dict(value["optimizer"])
        instance.update_count = int(value["update_count"])
        instance.rng.bit_generator.state = value["numpy_rng"]
        torch.set_rng_state(value["torch_rng"].cpu())
        if instance.device.type == "cuda" and value.get("device_rng") is not None:
            torch.cuda.set_rng_state(value["device_rng"].cpu(), instance.device)
        return instance

    def export(
        self,
        path: str | Path,
        *,
        training_input_identity: dict[str, Any] | None = None,
        source_normalizer_path: str | Path | None = None,
        source_dataset_manifest_path: str | Path | None = None,
    ) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: value.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.float32, copy=False)
            for name, value in self.model.state_dict().items()
        }
        arrays.update(
            {
                f"normalizer.{name}": np.ascontiguousarray(
                    getattr(self.normalizer, name)
                )
                for name in (
                    "input_mean",
                    "input_scale",
                    "physiology_delta_mean",
                    "physiology_delta_scale",
                )
            }
        )
        identity = training_input_identity or {
            "graph": {"status": "unknown"},
            "ports": {"status": "unknown"},
            "normalizer": {"status": "unknown"},
            "scope": "source rollout did not serialize these identities; research smoke only",
        }
        export_version = self.VERSION
        source_normalizer = None
        temporal_contract = None
        if source_normalizer_path is not None:
            source_path = Path(source_normalizer_path)
            expected = identity.get("source_normalizer", {})
            artifact_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if artifact_sha256 != expected.get("artifact_sha256"):
                raise ValueError("source normalizer artifact identity differs")
            with np.load(source_path, allow_pickle=False) as archive:
                count = np.asarray(archive["count"], dtype=np.float64)
                mean = np.asarray(archive["mean"], dtype=np.float64)
                m2 = np.asarray(archive["m2"], dtype=np.float64)
            if (
                count.shape != ()
                or mean.shape != (self.config.feature_dim,)
                or m2.shape != mean.shape
                or float(count) <= 0
                or not np.isfinite(mean).all()
                or not np.isfinite(m2).all()
            ):
                raise ValueError("invalid source normalizer moments")
            moment_value = {
                "count": float(count),
                "mean": mean.tolist(),
                "m2": m2.tolist(),
            }
            moment_sha256 = hashlib.sha256(
                json.dumps(moment_value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if moment_sha256 != expected.get("sha256"):
                raise ValueError("source normalizer moment identity differs")
            arrays.update(
                {
                    "source_normalizer.count": np.asarray(count),
                    "source_normalizer.mean": np.ascontiguousarray(mean),
                    "source_normalizer.m2": np.ascontiguousarray(m2),
                }
            )
            source_normalizer = {
                "format": "chreatures-running-moments-v1",
                "count": float(count),
                "moment_sha256": moment_sha256,
                "artifact_sha256": artifact_sha256,
                "variance": "m2 / max(count, 1.0)",
                "minimum_variance": 1e-5,
                "output_clip": [-5.0, 5.0],
            }
            export_version = 2
        if source_dataset_manifest_path is not None:
            if source_normalizer is None:
                raise ValueError(
                    "temporal contract requires embedded complete source normalizer"
                )
            manifest_path = Path(source_dataset_manifest_path)
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_sha256 != identity.get("dataset_manifest_sha256"):
                raise ValueError("source dataset manifest identity differs")
            source_manifest = json.loads(manifest_path.read_text())
            physics_dt = float(source_manifest["physical_dt"])
            macro_steps = int(source_manifest["macro_steps"])
            observation_interval = float(source_manifest["macro_dt"])
            if (
                physics_dt <= 0
                or macro_steps <= 0
                or observation_interval <= 0
                or not np.isclose(
                    physics_dt * macro_steps,
                    observation_interval,
                    rtol=0,
                    atol=1e-12,
                )
            ):
                raise ValueError("invalid source dataset temporal contract")
            temporal_contract = {
                "physics_dt_seconds": physics_dt,
                "macro_steps": macro_steps,
                "observation_interval_seconds": observation_interval,
                "source_dataset_manifest_sha256": manifest_sha256,
            }
            export_version = 3
        tensor_manifest = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
            for name, value in arrays.items()
        }
        metadata = {
            "format": FORMAT,
            "version": export_version,
            "config": asdict(self.config),
            "actions": list(ACTIONS),
            "physiology": list(PHYSIOLOGY),
            "tensor_layout": "model tensors row-major-f32; source moments float64",
            "updates": self.update_count,
            "feature_layout": "neural_feature_000..feature_dim-1 then named physiology",
            "gru_gate_order": "reset,update,new (PyTorch GRUCell r,z,n)",
            "tensors": tensor_manifest,
            "training_input_identity": identity,
            "normalizer": self.normalizer.metadata(),
            "source_normalizer": source_normalizer,
            "temporal_contract": temporal_contract,
            "forecast_status": "trained real anonymous sequences; residual scales are not epistemic/OOD calibrated"
            if self.update_count
            else "untrained",
            "scope": "immutable shared observation encoder, recurrent cells, transition and uncertainty decoder",
        }
        arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez(path, **arrays)
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": metadata,
        }
