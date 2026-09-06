"""Authenticated data boundary for breaking rich-body sensorimotor records."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chreatures.organism_interface import (
    ACTION_DIM,
    OBSERVATION_DIM,
    OBSERVATION_ORDER,
    PHYSIOLOGY_DIM,
    PREVIOUS_DIM,
    RECTIFIED_AXES,
)

DATASET_FORMAT = "chreatures-sensorimotor-play-rich-v3"
SCHEMA_ID = "chreatures-sensorimotor-play-rich-trajectory-v3"
SCHEMA_PATH = Path(__file__).with_name("trajectory-schema-rich-v3.json")
PROFILE_SHA256 = "c71380718ba5535dbaebdeaf8aa2e88cc45cf218312a03e13507877f02a5554e"
CHANNEL_NAMES_SHA256 = (
    "b4c6b328116d820143e16ee922ccffd7b950dbe008efc580ad93056e01349bfa"
)
NORMALIZER_FORMAT = "chreatures-rich-observation-normalizer-v4"
OUTCOME_ORDER = (
    "nutrition",
    "contact",
    "distance",
    "effort",
    "mechanical_work",
    "ingested_mass",
    "mouth_material_contacts",
    "homeostatic_reward",
)
_PARTITION = re.compile(
    r"episode-(?P<episode>[0-9]{3})/"
    r"world-(?P<world>[0-9]{3})/resident-(?P<resident>[0-9]{2})\Z"
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class RichEpisode:
    episode: int
    stage: int
    packet_path: Path
    packet_sha256: str
    world_slots: np.ndarray
    observation: np.ndarray
    actions: np.ndarray
    reset: np.ndarray
    outcomes: np.ndarray
    canonical: np.ndarray
    neural: np.ndarray | None
    worker_recurrent_context: np.ndarray

    @property
    def previous(self) -> np.ndarray:
        value = np.zeros(
            (*self.observation.shape[:2], PREVIOUS_DIM), dtype=np.float32
        )
        value[1:] = self.actions
        return value


class RichPlayDataset:
    """Verify every receipt before exposing direct 4459-column observations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / "manifest.json"
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        body = copy.deepcopy(manifest)
        expected_content = body.pop("content_sha256", None)
        if (
            manifest.get("format") != DATASET_FORMAT
            or manifest.get("version") != 3
            or expected_content != canonical_sha256(body)
        ):
            raise ValueError("rich dataset manifest identity differs")
        schema = json.loads(SCHEMA_PATH.read_text())
        schema_hash = file_sha256(SCHEMA_PATH)
        receipt = manifest.get("schema", {})
        if (
            schema.get("$id") != SCHEMA_ID
            or receipt.get("sha256") != schema_hash
            or int(receipt.get("bytes", -1)) != SCHEMA_PATH.stat().st_size
        ):
            raise ValueError("rich trajectory schema differs")
        identity = manifest.get("collection_identity", {})
        if (
            identity.get("rich_profile_sha256") != PROFILE_SHA256
            or identity.get("rich_channel_names_sha256") != CHANNEL_NAMES_SHA256
            or tuple(identity.get("observation_order", ())) != OBSERVATION_ORDER
        ):
            raise ValueError("rich sensor profile identity differs")
        if tuple(manifest.get("transition_outcome_order", ())) != OUTCOME_ORDER:
            raise ValueError("transition outcome order differs")
        scope = manifest.get("scope", {})
        self.world_count = int(scope["worlds"])
        self.residents_per_world = int(scope["residents_per_world"])
        self.steps_per_episode = int(scope["steps_per_episode"])
        self.dt_seconds = float(scope["dt_seconds"])
        if min(self.world_count, self.residents_per_world, self.steps_per_episode) < 1:
            raise ValueError("rich dataset scope is empty")
        if self.dt_seconds != 0.05:
            raise ValueError("rich dataset interval differs")
        packets = manifest.get("packets")
        if not isinstance(packets, list) or not packets:
            raise ValueError("rich packet receipts are missing")
        episodes = []
        hashes = {
            "manifest.json": hashlib.sha256(raw).hexdigest(),
            str(SCHEMA_PATH): schema_hash,
        }
        for number, packet_receipt in enumerate(packets):
            if int(packet_receipt.get("episode", -1)) != number:
                raise ValueError("rich episode ordering differs")
            packet = (self.path / str(packet_receipt.get("path"))).resolve()
            if not packet.is_relative_to(self.path) or not packet.is_file():
                raise ValueError("rich packet path is invalid")
            digest = file_sha256(packet)
            if digest != packet_receipt.get("sha256") or packet.stat().st_size != int(
                packet_receipt.get("bytes", -1)
            ):
                raise ValueError("rich packet receipt differs")
            episodes.append(self._load_episode(number, packet, digest, packet_receipt))
            hashes[packet.name] = digest
        self.manifest = manifest
        self.manifest_file_sha256 = hashlib.sha256(raw).hexdigest()
        self.file_sha256s = hashes
        self.episodes = episodes

    def _load_episode(self, number, path, digest, receipt) -> RichEpisode:
        with np.load(path, allow_pickle=False) as value:
            required = {
                "observation",
                "executed_actions",
                "reset",
                "dt_seconds",
                "transition_outcomes",
                "canonical_channels",
                "worker_recurrent_context",
            }
            if set(value.files) not in (required, required | {"neural_readouts"}):
                raise ValueError("rich packet arrays differ")
            arrays = {name: np.asarray(value[name]) for name in value.files}
        t, n = self.steps_per_episode, self.world_count * self.residents_per_world
        shapes = {
            "observation": (t + 1, n, OBSERVATION_DIM),
            "executed_actions": (t, n, ACTION_DIM),
            "reset": (t + 1, n),
            "dt_seconds": (),
            "transition_outcomes": (t, n, 8),
            "canonical_channels": (t + 1, n, 351),
            "worker_recurrent_context": (t, n, 128),
        }
        if "neural_readouts" in arrays:
            shapes["neural_readouts"] = (t + 1, n, 384)
        for name, shape in shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(f"rich {name} shape differs")
        for name in required - {"reset", "dt_seconds"}:
            if (
                arrays[name].dtype != np.dtype("<f4")
                or not np.isfinite(arrays[name]).all()
            ):
                raise ValueError(f"rich {name} dtype or finiteness differs")
        if "neural_readouts" in arrays and (
            arrays["neural_readouts"].dtype != np.dtype("<f4")
            or not np.isfinite(arrays["neural_readouts"]).all()
        ):
            raise ValueError("rich neural readout dtype or finiteness differs")
        if (
            arrays["reset"].dtype != np.dtype("|b1")
            or not arrays["reset"][0].all()
            or arrays["reset"][1:].any()
        ):
            raise ValueError("rich reset boundary differs")
        if (
            arrays["dt_seconds"].dtype != np.dtype("<f8")
            or float(arrays["dt_seconds"]) != 0.05
        ):
            raise ValueError("rich packet interval differs")
        observation = arrays["observation"]
        if not observation.flags.c_contiguous:
            raise ValueError("rich observation is not C-contiguous")
        if not np.array_equal(
            observation[..., 4096:4447], arrays["canonical_channels"]
        ):
            raise ValueError("embedded canonical channels differ")
        if np.any((observation[..., :4447] < 0) | (observation[..., :4447] > 1)):
            raise ValueError("rich or canonical channels exceed [0,1]")
        physiology = observation[..., -PHYSIOLOGY_DIM:]
        if (
            np.any((physiology[..., :3] < 0) | (physiology[..., :3] > 1))
            or np.any((physiology[..., 3:5] < -1) | (physiology[..., 3:5] > 1))
            or np.any((physiology[..., 5:] < 0) | (physiology[..., 5:] > 1))
        ):
            raise ValueError("rich physiology bounds differ")
        actions = arrays["executed_actions"]
        if np.any((actions < -1) | (actions > 1)) or np.any(
            actions[..., RECTIFIED_AXES] < 0
        ):
            raise ValueError("rich action bounds differ")
        partitions = receipt.get("resident_partitions")
        if not isinstance(partitions, list) or len(partitions) != n:
            raise ValueError("rich resident partitions differ")
        slots = []
        residents = []
        for value in partitions:
            match = _PARTITION.fullmatch(str(value))
            if match is None or int(match.group("episode")) != number:
                raise ValueError("rich partition key differs")
            slots.append(int(match.group("world")))
            residents.append(int(match.group("resident")))
        if slots != np.repeat(
            np.arange(self.world_count), self.residents_per_world
        ).tolist() or residents != np.tile(
            np.arange(self.residents_per_world), self.world_count
        ).tolist():
            raise ValueError("rich resident partition order differs")
        return RichEpisode(
            number,
            int(receipt["stage"]),
            path,
            digest,
            readonly(np.asarray(slots, dtype=np.int64)),
            readonly(observation),
            readonly(actions),
            readonly(arrays["reset"]),
            readonly(arrays["transition_outcomes"]),
            readonly(arrays["canonical_channels"]),
            readonly(arrays["neural_readouts"])
            if "neural_readouts" in arrays
            else None,
            readonly(arrays["worker_recurrent_context"]),
        )

    def columns(self, world_slots: Sequence[int]) -> np.ndarray:
        wanted = np.asarray(world_slots, dtype=np.int64)
        if (
            wanted.ndim != 1
            or not len(wanted)
            or np.any((wanted < 0) | (wanted >= self.world_count))
        ):
            raise ValueError("world slots are invalid")
        slots = self.episodes[0].world_slots
        return readonly(np.flatnonzero(np.isin(slots, wanted)).astype(np.int64))


class RichNormalizer:
    """Immutable per-column moments fit only on explicitly declared worlds."""

    def __init__(self, mean, scale, training_statistics: Mapping[str, Any]):
        self.mean = readonly(np.ascontiguousarray(mean, dtype=np.float32))
        self.scale = readonly(np.ascontiguousarray(scale, dtype=np.float32))
        if (
            self.mean.shape != (OBSERVATION_DIM,)
            or self.scale.shape != (OBSERVATION_DIM,)
            or np.any(self.scale < 0.02)
        ):
            raise ValueError("rich normalizer arrays differ")
        self.training_statistics = copy.deepcopy(dict(training_statistics))

    @classmethod
    def cold_inherit_v3(cls, value: Mapping[str, Any]) -> "RichNormalizer":
        """Extend the six-physiology v3 normalizer at an explicit cold birth."""
        source = copy.deepcopy(dict(value))
        expected = source.pop("sha256", None)
        if (
            source.get("format") != "chreatures-rich-observation-normalizer-v1"
            or source.get("version") != 1
            or expected != canonical_sha256(source)
        ):
            raise ValueError("v3 normalizer identity differs")
        mean = np.asarray(source["mean"], dtype=np.float32)
        scale = np.asarray(source["scale"], dtype=np.float32)
        if mean.shape != (4453,) or scale.shape != (4453,):
            raise ValueError("v3 normalizer dimensions differ")
        return cls(
            np.concatenate((mean, np.zeros(6, dtype=np.float32))),
            np.concatenate((scale, np.ones(6, dtype=np.float32))),
            {
                "scope": "explicit v3-to-v4 cold inheritance",
                "source": copy.deepcopy(dict(source.get("training_statistics", {}))),
                "new_physiology": "identity normalization pending v4 train-only fit",
            },
        )

    @classmethod
    def fit(cls, dataset: RichPlayDataset, training_world_slots: Sequence[int]):
        columns = dataset.columns(training_world_slots)
        count = 0
        mean = np.zeros(OBSERVATION_DIM, dtype=np.float64)
        m2 = np.zeros(OBSERVATION_DIM, dtype=np.float64)
        for episode in dataset.episodes:
            for start in range(0, len(episode.observation), 64):
                rows = episode.observation[start : start + 64, columns].reshape(
                    -1, OBSERVATION_DIM
                )
                batch_mean = rows.mean(0, dtype=np.float64)
                centered = rows.astype(np.float64) - batch_mean
                batch_m2 = np.sum(centered * centered, axis=0)
                total = count + len(rows)
                delta = batch_mean - mean
                mean += delta * len(rows) / total
                m2 += batch_m2 + delta * delta * count * len(rows) / total
                count = total
        scale = np.maximum(np.sqrt(m2 / count), 0.02)
        statistics = {
            "split": "explicit-train-worlds",
            "world_slots": list(training_world_slots),
            "resident_columns": columns.tolist(),
            "samples": count,
            "floor_std": 0.02,
            "clip": 8.0,
            "manifest_file_sha256": dataset.manifest_file_sha256,
            "packet_sha256s": [episode.packet_sha256 for episode in dataset.episodes],
        }
        return cls(mean, scale, statistics)

    def normalize(self, observation) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape[-1] != OBSERVATION_DIM or not np.isfinite(value).all():
            raise ValueError("rich observations must be finite [...,4459]")
        return np.clip((value - self.mean) / self.scale, -8, 8).astype(np.float32)

    def to_value(self) -> dict[str, Any]:
        body = {
            "format": NORMALIZER_FORMAT,
            "version": 4,
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "training_statistics": self.training_statistics,
        }
        return body | {"sha256": canonical_sha256(body)}

    @classmethod
    def from_value(cls, value: Mapping[str, Any]):
        body = dict(value)
        expected = body.pop("sha256", None)
        if (
            body.get("format") != NORMALIZER_FORMAT
            or body.get("version") != 4
            or expected != canonical_sha256(body)
        ):
            raise ValueError("rich normalizer identity differs")
        return cls(body["mean"], body["scale"], body["training_statistics"])
