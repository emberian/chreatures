"""Authenticated episode format for the current CNS-only resident.

Only the masked full-CNS latent and the resident's own previously delivered
neural context current are controller inputs. Delivered context currents and
scalar physical rewards are training targets. Geometry used to calculate a
reward is deliberately not written into this artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

import numpy as np

FORMAT = "chreatures-cns-context-resident-episode-v2"
CONTEXT_POLICY_VERSION = "signed-context12-v1"
CONTROLLER_FIELDS = ("cns_latent", "previous_delivered_context", "reset")
TEACHER_FIELDS = ("delivered_context", "physical_reward", "terminal")
ARRAY_FIELDS = CONTROLLER_FIELDS + TEACHER_FIELDS
CORPUS_FORMAT = "chreatures-cns-context-resident-corpus-v2"
SKILLS = (
    "approach",
    "heading-correction",
    "stop",
    "withdraw",
    "contact-recovery",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def _identity(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    clean = dict(metadata)
    clean.pop("episode_sha256", None)
    digest = hashlib.sha256(canonical(clean))
    for name in ARRAY_FIELDS:
        digest.update(name.encode())
        digest.update(arrays[name].tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidentEpisode:
    path: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    file_sha256: str

    @property
    def transitions(self) -> int:
        return self.arrays["delivered_context"].shape[0]

    @property
    def residents(self) -> int:
        return self.arrays["delivered_context"].shape[1]

    @property
    def curriculum(self) -> dict[str, Any]:
        """Privileged collection labels used only to stratify training windows."""
        return validate_curriculum(self.metadata["provenance"], self.transitions)


def validate_curriculum(provenance: Mapping[str, Any], transitions: int) -> dict[str, Any]:
    """Validate the independent-world split without exposing labels to the model."""
    required = {
        "episode_index", "split", "world_seed", "variation_seed",
        "layout_identity", "teacher_bouts",
    }
    missing = required - set(provenance)
    if missing:
        raise ValueError(f"episode curriculum metadata is missing {sorted(missing)}")
    episode_index = provenance["episode_index"]
    split = provenance["split"]
    if not isinstance(episode_index, int) or isinstance(episode_index, bool) or not 0 <= episode_index < 8:
        raise ValueError("episode_index must be in [0,8)")
    if split not in {"train", "heldout-worlds"}:
        raise ValueError("episode split differs")
    expected_split = "train" if episode_index < 6 else "heldout-worlds"
    if split != expected_split:
        raise ValueError("episode index and world split differ")
    for name in ("world_seed", "variation_seed"):
        if not isinstance(provenance[name], int) or isinstance(provenance[name], bool):
            raise ValueError(f"{name} must be an integer")
    layout = provenance["layout_identity"]
    if not isinstance(layout, str) or _SHA256.fullmatch(layout) is None:
        raise ValueError("layout_identity must be a lowercase SHA-256")
    bouts = provenance["teacher_bouts"]
    if not isinstance(bouts, list) or not bouts:
        raise ValueError("teacher_bouts must be a nonempty list")
    cursor = 0
    clean_bouts = []
    for bout in bouts:
        if not isinstance(bout, dict) or set(bout) != {"start_tick", "end_tick", "skill"}:
            raise ValueError("teacher bout fields differ")
        start, end, skill = bout["start_tick"], bout["end_tick"], bout["skill"]
        if not isinstance(start, int) or not isinstance(end, int) or start != cursor or end <= start:
            raise ValueError("teacher bouts must be ordered, gapless, and nonempty")
        if skill not in SKILLS:
            raise ValueError("teacher bout skill differs")
        clean_bouts.append({"start_tick": start, "end_tick": end, "skill": skill})
        cursor = end
    if cursor != transitions:
        raise ValueError("teacher bouts must cover the complete episode")
    return {
        "episode_index": episode_index,
        "split": split,
        "world_seed": provenance["world_seed"],
        "variation_seed": provenance["variation_seed"],
        "layout_identity": layout,
        "teacher_bouts": clean_bouts,
    }


def validate_corpus(episodes: list[ResidentEpisode]) -> tuple[list[ResidentEpisode], list[ResidentEpisode]]:
    """Require six train worlds and two identity-disjoint held-out worlds."""
    if len(episodes) != 8:
        raise ValueError("resident corpus must contain exactly eight independent episodes")
    by_index: dict[int, ResidentEpisode] = {}
    identities: set[tuple[int, int, str]] = set()
    for episode in episodes:
        if episode.transitions != 512 or episode.residents != 3:
            raise ValueError("resident corpus episodes must be exactly 512 transitions by three residents")
        curriculum = episode.curriculum
        index = curriculum["episode_index"]
        if index in by_index:
            raise ValueError("resident corpus repeats an episode index")
        identity = (
            curriculum["world_seed"], curriculum["variation_seed"],
            curriculum["layout_identity"],
        )
        if identity in identities:
            raise ValueError("resident corpus repeats a world/variation/layout identity")
        identities.add(identity)
        by_index[index] = episode
    if set(by_index) != set(range(8)):
        raise ValueError("resident corpus episode indices differ")
    ordered = [by_index[index] for index in range(8)]
    return ordered[:6], ordered[6:]


def _validated_arrays(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if set(values) != set(ARRAY_FIELDS):
        raise ValueError("CNS resident episode tensor set differs")
    arrays = {
        "cns_latent": np.ascontiguousarray(values["cns_latent"], dtype=np.float32),
        "previous_delivered_context": np.ascontiguousarray(
            values["previous_delivered_context"], dtype=np.float32
        ),
        "reset": np.ascontiguousarray(values["reset"], dtype=np.bool_),
        "delivered_context": np.ascontiguousarray(values["delivered_context"], dtype=np.float32),
        "physical_reward": np.ascontiguousarray(values["physical_reward"], dtype=np.float32),
        "terminal": np.ascontiguousarray(values["terminal"], dtype=np.bool_),
    }
    t, b, action = arrays["delivered_context"].shape
    if t < 2 or b < 1 or action != 12:
        raise ValueError("episode must contain at least two [resident,action12] transitions")
    expected = {
        "cns_latent": (t + 1, b, 512),
        "previous_delivered_context": (t + 1, b, 12),
        "reset": (t + 1, b),
        "physical_reward": (t, b),
        "terminal": (t, b),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"episode tensor shape differs: {name}")
    for name, value in arrays.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"episode tensor is nonfinite: {name}")
    for name in ("previous_delivered_context", "delivered_context"):
        value = arrays[name]
        if np.any(np.abs(value) > 1.000001):
            raise ValueError(f"episode signed context bounds differ: {name}")
    if not arrays["reset"][0].all():
        raise ValueError("every episode resident must start with reset=true")
    if np.any(arrays["terminal"][:-1]):
        raise ValueError("terminal is only valid at an episode's final transition")
    if not np.allclose(arrays["previous_delivered_context"][1:], arrays["delivered_context"], atol=0, rtol=0):
        raise ValueError("previous delivered context is not the exact CNS-injection receipt")
    return arrays


def write_episode(
    path: str | Path,
    arrays: Mapping[str, Any],
    *,
    cns_service: Mapping[str, Any],
    resident_artifact: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> ResidentEpisode:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    values = _validated_arrays(arrays)
    metadata: dict[str, Any] = {
        "format": FORMAT,
        "controller_input_fields": list(CONTROLLER_FIELDS),
        "teacher_only_fields": list(TEACHER_FIELDS),
        "forbidden_controller_fields": [
            "raw_visual", "raw_sensory", "raw_physiology", "world_position",
            "world_geometry", "object_kind", "reward", "outcome",
        ],
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "causality": "z_t,previous_delivered_context_t -> proposed_context_t -> CNS_t+1 -> z_t+1",
        "cns_service": dict(cns_service),
        "resident_artifact": dict(resident_artifact),
        "provenance": dict(provenance),
        "arrays": {name: _array_receipt(values[name]) for name in ARRAY_FIELDS},
    }
    metadata["episode_sha256"] = _identity(metadata, values)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, metadata=np.asarray(canonical(metadata).decode()), **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return load_episode(destination)


def load_episode(path: str | Path) -> ResidentEpisode:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", *ARRAY_FIELDS}:
            raise ValueError("CNS resident episode archive fields differ")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = _validated_arrays({name: archive[name] for name in ARRAY_FIELDS})
    required = {
        "format", "controller_input_fields", "teacher_only_fields",
        "forbidden_controller_fields", "context_policy_version", "causality", "cns_service",
        "resident_artifact", "provenance", "arrays", "episode_sha256",
    }
    if (set(metadata) != required or metadata["format"] != FORMAT
            or metadata["context_policy_version"] != CONTEXT_POLICY_VERSION):
        raise ValueError("CNS resident episode metadata differs")
    if tuple(metadata["controller_input_fields"]) != CONTROLLER_FIELDS or tuple(metadata["teacher_only_fields"]) != TEACHER_FIELDS:
        raise ValueError("CNS resident episode information boundary differs")
    if metadata["arrays"] != {name: _array_receipt(arrays[name]) for name in ARRAY_FIELDS}:
        raise ValueError("CNS resident episode tensor receipt differs")
    if metadata["episode_sha256"] != _identity(metadata, arrays):
        raise ValueError("CNS resident episode identity differs")
    return ResidentEpisode(source, metadata, arrays, file_sha256(source))


class ClosedLoopCollector:
    """Alternating observation/delivery collector for a fresh embodied run.

    The interface has no slot for a camera image, physiology vector, position,
    object label, or other possible controller bypass. Callers calculate the
    scalar teacher reward on their side and discard its privileged ingredients.
    """

    def __init__(
        self,
        initial_cns_latent: Any,
        *,
        cns_service: Mapping[str, Any],
        resident_artifact: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> None:
        initial = np.ascontiguousarray(initial_cns_latent, dtype=np.float32)
        if initial.ndim != 2 or initial.shape[1] != 512 or not np.isfinite(initial).all():
            raise ValueError("initial closed-loop observation must be finite [resident,512]")
        self._batch = initial.shape[0]
        self._latent = [initial]
        self._previous = [np.zeros((self._batch, 12), np.float32)]
        self._reset = [np.ones(self._batch, np.bool_)]
        self._delivered: list[np.ndarray] = []
        self._reward: list[np.ndarray] = []
        self._terminal: list[np.ndarray] = []
        self._closed = False
        self.cns_service = dict(cns_service)
        self.resident_artifact = dict(resident_artifact)
        self.provenance = dict(provenance)

    def append(
        self,
        delivered_context: Any,
        physical_reward: Any,
        terminal: Any,
        next_cns_latent: Any,
        *,
        next_reset: Any | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("closed-loop collector is finalized")
        context = np.ascontiguousarray(delivered_context, dtype=np.float32)
        reward = np.ascontiguousarray(physical_reward, dtype=np.float32)
        done = np.ascontiguousarray(terminal, dtype=np.bool_)
        latent = np.ascontiguousarray(next_cns_latent, dtype=np.float32)
        reset = np.zeros(self._batch, np.bool_) if next_reset is None else np.ascontiguousarray(next_reset, dtype=np.bool_)
        if context.shape != (self._batch, 12) or reward.shape != (self._batch,) or done.shape != (self._batch,) or latent.shape != (self._batch, 512) or reset.shape != (self._batch,):
            raise ValueError("closed-loop transition shapes differ")
        if not np.isfinite(context).all() or np.any(np.abs(context) > 1.000001) or not np.isfinite(reward).all() or not np.isfinite(latent).all():
            raise ValueError("closed-loop transition is nonfinite")
        if self._terminal and self._terminal[-1].any():
            raise RuntimeError("cannot append after a terminal transition")
        self._delivered.append(context); self._reward.append(reward); self._terminal.append(done)
        self._latent.append(latent); self._previous.append(context.copy()); self._reset.append(reset)

    def finalize(self, path: str | Path) -> ResidentEpisode:
        if self._closed or len(self._delivered) < 2:
            raise RuntimeError("closed-loop collector is finalized or too short")
        self._closed = True
        return write_episode(
            path,
            {
                "cns_latent": np.stack(self._latent),
                "previous_delivered_context": np.stack(self._previous),
                "reset": np.stack(self._reset),
                "delivered_context": np.stack(self._delivered),
                "physical_reward": np.stack(self._reward),
                "terminal": np.stack(self._terminal),
            },
            cns_service=self.cns_service,
            resident_artifact=self.resident_artifact,
            provenance=self.provenance,
        )
