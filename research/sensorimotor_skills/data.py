"""Verified NumPy data boundary for body-local sensorimotor play records."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "research/sensorimotor_skills/trajectory-schema-v1.json"
DATASET_FORMAT = "chreatures-sensorimotor-play-dataset-v1"
IDENTITY_FORMAT = "chreatures-sensorimotor-play-collection-identity-v1"
NORMALIZER_FORMAT = "chreatures-sensorimotor-play-normalizer-v1"
SCHEMA_ID = "chreatures-sensorimotor-play-trajectory-v1"
OBSERVATION_DIM = 357
ACTION_DIM = 8
PREVIOUS_DIM = 9
PHYSIOLOGY_DIM = 6
SOURCE_SENSE_DIM = 351
DT_SECONDS = 0.05
_PARTITION = re.compile(
    r"episode-(?P<episode>[0-9]{3})/world-(?P<world>[0-9]{3})/resident-(?P<resident>[0-9]{2})\Z"
)
_PACKET = re.compile(r"episode-(?P<episode>[0-9]{3})\.npz\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(value: Mapping[str, Any], field: str) -> str:
    body = copy.deepcopy(dict(value))
    body.pop(field, None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _safe_file(directory: Path, name: Any, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"{label} path is invalid")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"{label} path escapes the dataset")
    return path


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify_receipt(
    path: Path, receipt: Mapping[str, Any], label: str
) -> tuple[str, tuple[int, int, int, int, int]]:
    if not path.is_file():
        raise ValueError(f"{label} file is missing")
    try:
        expected_bytes = int(receipt["bytes"])
        expected_hash = str(receipt["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} receipt is invalid") from exc
    if expected_bytes < 0 or len(expected_hash) != 64:
        raise ValueError(f"{label} receipt is invalid")
    before = _file_identity(path)
    actual_hash = _sha256(path)
    after = _file_identity(path)
    if before != after or after[2] != expected_bytes or actual_hash != expected_hash:
        raise ValueError(f"{label} receipt differs")
    return actual_hash, after


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class DatasetIdentity:
    """Verified collection identity plus hashes of every model-data input."""

    value: Mapping[str, Any]
    sha256: str
    file_sha256s: Mapping[str, str]


@dataclass(frozen=True)
class Episode:
    """One verified episode. Partition strings are deliberately not model fields."""

    episode: int
    stage: int
    packet_path: Path
    packet_sha256: str
    world_slots: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    oral: np.ndarray
    reset: np.ndarray
    previous: np.ndarray


class PlayDataset:
    """Load checksum-bound play packets without reading birth checkpoints."""

    def __init__(self, path: str | Path) -> None:
        supplied = Path(path).expanduser().resolve()
        manifest_path = (
            supplied if supplied.name == "manifest.json" else supplied / "manifest.json"
        )
        self.path = manifest_path.parent
        if not manifest_path.is_file():
            raise ValueError("sensorimotor dataset manifest is missing")
        try:
            manifest_raw = manifest_path.read_bytes()
            manifest_bytes_hash = hashlib.sha256(manifest_raw).hexdigest()
            manifest = json.loads(manifest_raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("sensorimotor dataset manifest is invalid") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != DATASET_FORMAT
            or manifest.get("version") != 1
            or manifest.get("content_sha256")
            != _content_sha256(manifest, "content_sha256")
        ):
            raise ValueError("sensorimotor dataset manifest identity differs")
        self.manifest = manifest

        scope = manifest.get("scope")
        if not isinstance(scope, dict):
            raise ValueError("sensorimotor dataset scope is invalid")
        try:
            worlds = int(scope["worlds"])
            residents_per_world = int(scope["residents_per_world"])
            episode_count = int(scope["episodes"])
            steps = int(scope["steps_per_episode"])
            dt = float(scope["dt_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sensorimotor dataset dimensions are invalid") from exc
        if (
            worlds < 6
            or residents_per_world <= 0
            or episode_count <= 0
            or steps <= 0
            or dt != DT_SECONDS
        ):
            raise ValueError("sensorimotor dataset dimensions or interval differ")
        self.world_count = worlds
        self.residents_per_world = residents_per_world
        self.steps_per_episode = steps
        self.dt_seconds = dt

        schema_receipt = manifest.get("schema")
        if (
            not isinstance(schema_receipt, dict)
            or schema_receipt.get("path")
            != "research/sensorimotor_skills/trajectory-schema-v1.json"
        ):
            raise ValueError("sensorimotor schema receipt is invalid")
        schema_raw = SCHEMA_PATH.read_bytes()
        schema_hash = hashlib.sha256(schema_raw).hexdigest()
        if schema_receipt.get("sha256") != schema_hash:
            raise ValueError("pinned sensorimotor schema differs")
        schema = json.loads(schema_raw)
        if (
            schema.get("$id") != SCHEMA_ID
            or schema.get("additionalProperties") is not False
            or set(schema.get("required", ()))
            != {
                "source_senses",
                "physiology",
                "executed_actions",
                "oral_command",
                "reset",
                "dt_seconds",
            }
        ):
            raise ValueError("pinned sensorimotor schema contract differs")
        sources = manifest.get("sources")
        embedded_identity = manifest.get("collection_identity")
        if not isinstance(sources, dict) or not isinstance(embedded_identity, dict):
            raise ValueError("dataset source or identity records are invalid")
        identity_sources = embedded_identity.get("sources")
        if not isinstance(identity_sources, dict):
            raise ValueError("collection identity source records are invalid")
        source_schema = sources.get("schema", {})
        identity_source_schema = identity_sources.get("schema", {})
        if not all(
            isinstance(value, dict) for value in (source_schema, identity_source_schema)
        ):
            raise ValueError("schema source receipts are invalid")
        if any(
            receipt.get("sha256") != schema_hash
            or int(receipt.get("bytes", -1)) != SCHEMA_PATH.stat().st_size
            for receipt in (source_schema, identity_source_schema)
        ):
            raise ValueError("schema source receipts differ")

        identity_receipt = manifest.get("collection_identity_receipt")
        if (
            not isinstance(identity_receipt, dict)
            or identity_receipt.get("path") != "identity.json"
        ):
            raise ValueError("collection identity receipt is invalid")
        identity_path = _safe_file(self.path, identity_receipt["path"], "identity")
        identity_file_hash, identity_file_identity = _verify_receipt(
            identity_path, identity_receipt, "collection identity"
        )
        try:
            identity_value = json.loads(identity_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("collection identity file is invalid") from exc
        if _file_identity(identity_path) != identity_file_identity:
            raise ValueError("collection identity changed while loading")
        if (
            not isinstance(identity_value, dict)
            or identity_value.get("format") != IDENTITY_FORMAT
            or identity_value.get("sha256") != _content_sha256(identity_value, "sha256")
            or identity_value != manifest.get("collection_identity")
        ):
            raise ValueError("collection identity content differs")

        packet_receipts = manifest.get("packets")
        if (
            not isinstance(packet_receipts, list)
            or len(packet_receipts) != episode_count
        ):
            raise ValueError("episode packet manifest differs")
        receipts_by_episode: dict[
            int, tuple[Mapping[str, Any], Path, str, tuple[int, int, int, int, int]]
        ] = {}
        file_hashes = {
            "manifest.json": manifest_bytes_hash,
            "identity.json": identity_file_hash,
            "research/sensorimotor_skills/trajectory-schema-v1.json": schema_hash,
        }
        # Authenticate every packet before np.load is called on any packet.
        for receipt in packet_receipts:
            if not isinstance(receipt, dict):
                raise ValueError("episode packet receipt is invalid")
            try:
                episode_number = int(receipt["episode"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("episode packet number is invalid") from exc
            name = receipt.get("path")
            match = _PACKET.fullmatch(str(name))
            if (
                match is None
                or int(match.group("episode")) != episode_number
                or episode_number in receipts_by_episode
            ):
                raise ValueError("episode packet path or ordering identity differs")
            packet_path = _safe_file(self.path, name, "episode packet")
            packet_hash, packet_identity = _verify_receipt(
                packet_path, receipt, f"episode {episode_number} packet"
            )
            receipts_by_episode[episode_number] = (
                receipt,
                packet_path,
                packet_hash,
                packet_identity,
            )
            file_hashes[name] = packet_hash
        if set(receipts_by_episode) != set(range(episode_count)):
            raise ValueError("episode packet numbers are not contiguous")

        episodes = []
        expected_slots: np.ndarray | None = None
        for episode_number in range(episode_count):
            receipt, packet_path, packet_hash, packet_identity = receipts_by_episode[
                episode_number
            ]
            episode = self._load_episode(
                episode_number, receipt, packet_path, packet_hash, packet_identity
            )
            if expected_slots is None:
                expected_slots = episode.world_slots
            elif not np.array_equal(episode.world_slots, expected_slots):
                raise ValueError("resident world-slot columns differ across episodes")
            episodes.append(episode)
        self.episodes = episodes
        self._world_slots = expected_slots
        self.identity = DatasetIdentity(
            value=MappingProxyType(copy.deepcopy(identity_value)),
            sha256=str(identity_value["sha256"]),
            file_sha256s=MappingProxyType(file_hashes),
        )

    def _load_episode(
        self,
        episode_number: int,
        receipt: Mapping[str, Any],
        packet_path: Path,
        packet_hash: str,
        packet_identity: tuple[int, int, int, int, int],
    ) -> Episode:
        partitions = receipt.get("resident_partitions")
        count = self.world_count * self.residents_per_world
        if not isinstance(partitions, list) or len(partitions) != count:
            raise ValueError("resident partition metadata differs")
        slots = []
        residents_by_world: dict[int, set[int]] = {}
        for value in partitions:
            match = _PARTITION.fullmatch(str(value))
            if match is None or int(match.group("episode")) != episode_number:
                raise ValueError("resident partition key is invalid")
            world = int(match.group("world"))
            resident = int(match.group("resident"))
            if not 0 <= world < self.world_count:
                raise ValueError("resident partition world slot is invalid")
            residents_by_world.setdefault(world, set()).add(resident)
            slots.append(world)
        expected_residents = set(range(self.residents_per_world))
        if set(residents_by_world) != set(range(self.world_count)) or any(
            value != expected_residents for value in residents_by_world.values()
        ):
            raise ValueError("resident partition coverage differs")
        world_slots = np.asarray(slots, dtype=np.int64)

        if _file_identity(packet_path) != packet_identity:
            raise ValueError("episode packet changed after verification")
        with np.load(packet_path, allow_pickle=False) as packet:
            required = {
                "source_senses",
                "physiology",
                "executed_actions",
                "oral_command",
                "reset",
                "dt_seconds",
            }
            names = set(packet.files)
            if names not in (required, required | {"neural_readouts"}):
                raise ValueError("episode packet arrays differ from the pinned schema")
            source = np.asarray(packet["source_senses"])
            physiology = np.asarray(packet["physiology"])
            actions = np.asarray(packet["executed_actions"])
            oral = np.asarray(packet["oral_command"])
            reset = np.asarray(packet["reset"])
            dt = np.asarray(packet["dt_seconds"])
            neural = (
                np.asarray(packet["neural_readouts"])
                if "neural_readouts" in names
                else None
            )
        if _file_identity(packet_path) != packet_identity:
            raise ValueError("episode packet changed while loading")

        t, n = self.steps_per_episode, count
        expected = {
            "source_senses": (t + 1, n, SOURCE_SENSE_DIM),
            "physiology": (t + 1, n, PHYSIOLOGY_DIM),
            "executed_actions": (t, n, ACTION_DIM),
            "oral_command": (t, n),
            "reset": (t + 1, n),
            "dt_seconds": (),
        }
        arrays = {
            "source_senses": source,
            "physiology": physiology,
            "executed_actions": actions,
            "oral_command": oral,
            "reset": reset,
            "dt_seconds": dt,
        }
        shapes_receipt = receipt.get("model_array_shapes")
        if not isinstance(shapes_receipt, dict):
            raise ValueError("episode array-shape receipt is invalid")
        for name, shape in expected.items():
            if arrays[name].shape != shape or shapes_receipt.get(name) != list(shape):
                raise ValueError(f"episode array {name} shape differs")
        for name in ("source_senses", "physiology", "executed_actions", "oral_command"):
            if (
                arrays[name].dtype != np.dtype("<f4")
                or not np.isfinite(arrays[name]).all()
            ):
                raise ValueError(f"episode array {name} dtype or finiteness differs")
        if reset.dtype != np.dtype("|b1"):
            raise ValueError("episode reset dtype differs")
        if (
            dt.dtype != np.dtype("<f8")
            or not np.isfinite(dt)
            or float(dt) != DT_SECONDS
        ):
            raise ValueError("episode physical interval differs")
        if int(receipt.get("steps", -1)) != t:
            raise ValueError("episode step receipt differs")
        if int(receipt.get("stage", -1)) < 0:
            raise ValueError("episode stage is invalid")

        if np.any((source < 0) | (source > 1)):
            raise ValueError("source senses exceed encoded [0,1] bounds")
        if (
            np.any((physiology[..., :3] < 0) | (physiology[..., :3] > 1))
            or np.any((physiology[..., 3:5] < -1) | (physiology[..., 3:5] > 1))
            or np.any((physiology[..., 5] < 0) | (physiology[..., 5] > 1))
        ):
            raise ValueError("physiology exceeds declared physical bounds")
        if np.any((actions < -1) | (actions > 1)) or np.any(actions[..., 3:7] < 0):
            raise ValueError("executed actions exceed physical bounds")
        if np.any((oral < 0) | (oral > 1)):
            raise ValueError("oral command exceeds [0,1] bounds")
        if not reset[0].all() or reset[1:].any():
            raise ValueError("episode reset boundary differs")

        if neural is not None:
            neural_shape = (t + 1, n, 384)
            if (
                neural.shape != neural_shape
                or shapes_receipt.get("neural_readouts") != list(neural_shape)
                or neural.dtype != np.dtype("<f4")
                or not np.isfinite(neural).all()
            ):
                raise ValueError("optional neural readouts differ")
        elif set(shapes_receipt) != set(expected):
            raise ValueError("episode array-shape receipt contains unknown arrays")
        if neural is not None and set(shapes_receipt) != set(expected) | {
            "neural_readouts"
        }:
            raise ValueError("episode array-shape receipt contains unknown arrays")

        observations = np.ascontiguousarray(
            np.concatenate((source, physiology), axis=-1), dtype=np.float32
        )
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        oral = np.ascontiguousarray(oral, dtype=np.float32)
        reset = np.ascontiguousarray(reset, dtype=np.bool_)
        previous = np.zeros((t + 1, n, PREVIOUS_DIM), dtype=np.float32)
        previous[1:, :, :ACTION_DIM] = actions
        previous[1:, :, ACTION_DIM] = oral
        return Episode(
            episode=episode_number,
            stage=int(receipt["stage"]),
            packet_path=packet_path,
            packet_sha256=packet_hash,
            world_slots=_readonly(world_slots),
            observations=_readonly(observations),
            actions=_readonly(actions),
            oral=_readonly(oral),
            reset=_readonly(reset),
            previous=_readonly(previous),
        )

    def indices(self, split: str) -> np.ndarray:
        """Return resident columns for fixed whole-world train/evaluation splits."""
        boundaries = {
            "train": (0, self.world_count - 4),
            "validation": (self.world_count - 4, self.world_count - 2),
            "holdout": (self.world_count - 2, self.world_count),
        }
        if split not in boundaries:
            raise ValueError("split must be train, validation, or holdout")
        start, stop = boundaries[split]
        result = np.flatnonzero(
            (self._world_slots >= start) & (self._world_slots < stop)
        ).astype(np.int64)
        return _readonly(result)


class Normalizer:
    """Frozen observation normalizer fit only from training-world columns."""

    def __init__(
        self,
        mean: Any,
        scale: Any,
        training_statistics: Mapping[str, Any],
    ) -> None:
        self.mean = _readonly(np.ascontiguousarray(mean, dtype=np.float32))
        self.scale = _readonly(np.ascontiguousarray(scale, dtype=np.float32))
        if (
            self.mean.shape != (OBSERVATION_DIM,)
            or self.scale.shape != (OBSERVATION_DIM,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or np.any(self.scale < np.float32(0.02))
        ):
            raise ValueError("normalizer arrays are invalid")
        self.training_statistics = copy.deepcopy(dict(training_statistics))
        if self.training_statistics.get("split") != "train":
            raise ValueError("normalizer statistics must identify the training split")

    @classmethod
    def fit(cls, dataset: PlayDataset) -> "Normalizer":
        columns = dataset.indices("train")
        count = 0
        mean = np.zeros(OBSERVATION_DIM, dtype=np.float64)
        m2 = np.zeros(OBSERVATION_DIM, dtype=np.float64)
        for episode in dataset.episodes:
            for start in range(0, len(episode.observations), 256):
                # Advanced indexing materializes only training resident columns.
                rows = episode.observations[start : start + 256, columns, :].reshape(
                    -1, OBSERVATION_DIM
                )
                batch_count = len(rows)
                batch_mean = rows.mean(axis=0, dtype=np.float64)
                centered = rows.astype(np.float64) - batch_mean
                batch_m2 = np.sum(centered * centered, axis=0)
                total = count + batch_count
                delta = batch_mean - mean
                mean += delta * (batch_count / total)
                m2 += batch_m2 + delta * delta * count * batch_count / total
                count = total
        if count <= 0:
            raise ValueError("training split contains no observations")
        scale = np.maximum(np.sqrt(m2 / count), 0.02)
        train_slots = list(range(dataset.world_count - 4))
        statistics = {
            "split": "train",
            "world_slots": train_slots,
            "resident_columns": columns.astype(int).tolist(),
            "samples": count,
            "floor_std": 0.02,
            "source_manifest_content_sha256": dataset.manifest["content_sha256"],
            "source_identity_sha256": dataset.identity.sha256,
            "source_packet_sha256s": [
                episode.packet_sha256 for episode in dataset.episodes
            ],
        }
        return cls(mean.astype(np.float32), scale.astype(np.float32), statistics)

    def normalize(self, observations: Any) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if values.shape[-1:] != (OBSERVATION_DIM,) or not np.isfinite(values).all():
            raise ValueError("observations must be finite and end with dimension 357")
        return np.clip((values - self.mean) / self.scale, -8, 8).astype(np.float32)

    def to_value(self) -> dict[str, Any]:
        body = {
            "format": NORMALIZER_FORMAT,
            "version": 1,
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "training_statistics": copy.deepcopy(self.training_statistics),
        }
        return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "Normalizer":
        if not isinstance(value, Mapping):
            raise ValueError("normalizer value must be an object")
        body = dict(value)
        expected = body.pop("sha256", None)
        if (
            body.get("format") != NORMALIZER_FORMAT
            or body.get("version") != 1
            or expected != hashlib.sha256(_canonical(body)).hexdigest()
        ):
            raise ValueError("normalizer identity differs")
        return cls(
            body.get("mean"), body.get("scale"), body.get("training_statistics", {})
        )
