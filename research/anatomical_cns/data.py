"""Strict actual-physics curriculum data for the anatomical CNS V3 trainer.

The privileged teacher pose and bout labels in this format are targets and
sampling metadata.  They are never inputs to the CNS model.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORMAT = "chreatures-anatomical-cns-physical-episode-v1"
CORPUS_FORMAT = "chreatures-anatomical-cns-physical-corpus-v1"
TICKS = 512
RESIDENTS = 3
OPTIC = 5313
BODY = 110
MOTOR = 34
CONTEXT = 12
POSE = 12  # root xyz followed by the row-major 3x3 root rotation
SKILLS = ("pose-hold", "turn", "stop", "reach", "babble")
SPLITS = ("train", "heldout-worlds")


class ContractError(ValueError):
    """The physical curriculum does not satisfy the one current contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(value: np.ndarray, path: Path) -> dict[str, Any]:
    if value.shape != () or value.dtype.kind != "U":
        raise ContractError(f"{path}: metadata must be a scalar Unicode JSON value")
    try:
        result = json.loads(str(value.item()))
    except (TypeError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: invalid metadata JSON") from error
    if not isinstance(result, dict):
        raise ContractError(f"{path}: metadata must decode to an object")
    return result


def _array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    path: Path,
) -> np.ndarray:
    if name not in archive.files:
        raise ContractError(f"{path}: missing {name}")
    value = archive[name]
    if value.shape != shape or value.dtype != dtype:
        raise ContractError(
            f"{path}: {name} is {value.dtype}{value.shape}, expected {dtype}{shape}"
        )
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ContractError(f"{path}: {name} contains non-finite values")
    return value


@dataclass(frozen=True)
class Episode:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    optic_rgb: np.ndarray
    body: np.ndarray
    delivered_context: np.ndarray
    delivered_motor: np.ndarray
    teacher_joint_target: np.ndarray
    root_pose: np.ndarray
    skill_id: np.ndarray
    reset: np.ndarray
    terminal: np.ndarray

    @property
    def split(self) -> str:
        return str(self.metadata["split"])


def load_episode(path: str | Path) -> Episode:
    path = Path(path).resolve()
    expected_members = {
        "metadata",
        "optic_rgb",
        "body",
        "delivered_context",
        "delivered_motor",
        "teacher_joint_target",
        "root_pose",
        "skill_id",
        "reset",
        "terminal",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_members:
            raise ContractError(
                f"{path}: members differ: {sorted(set(archive.files) ^ expected_members)}"
            )
        metadata = _metadata(archive["metadata"], path)
        episode = Episode(
            path=path,
            sha256=sha256_file(path),
            metadata=metadata,
            optic_rgb=_array(archive, "optic_rgb", (TICKS + 1, RESIDENTS, OPTIC), np.dtype("<f4"), path).copy(),
            body=_array(archive, "body", (TICKS + 1, RESIDENTS, BODY), np.dtype("<f4"), path).copy(),
            delivered_context=_array(archive, "delivered_context", (TICKS, RESIDENTS, CONTEXT), np.dtype("<f4"), path).copy(),
            delivered_motor=_array(archive, "delivered_motor", (TICKS, RESIDENTS, MOTOR), np.dtype("<f4"), path).copy(),
            teacher_joint_target=_array(archive, "teacher_joint_target", (TICKS, RESIDENTS, 12), np.dtype("<f4"), path).copy(),
            root_pose=_array(archive, "root_pose", (TICKS + 1, RESIDENTS, POSE), np.dtype("<f4"), path).copy(),
            skill_id=_array(archive, "skill_id", (TICKS, RESIDENTS), np.dtype("|u1"), path).copy(),
            reset=_array(archive, "reset", (TICKS + 1, RESIDENTS), np.dtype("|b1"), path).copy(),
            terminal=_array(archive, "terminal", (TICKS, RESIDENTS), np.dtype("|b1"), path).copy(),
        )
    _validate_episode(episode)
    return episode


def _validate_episode(episode: Episode) -> None:
    meta, path = episode.metadata, episode.path
    if meta.get("format") != FORMAT:
        raise ContractError(f"{path}: episode format differs")
    index = meta.get("episode_index")
    split = meta.get("split")
    if not isinstance(index, int) or not 0 <= index < 8:
        raise ContractError(f"{path}: episode_index must be in [0,8)")
    if split != ("train" if index < 6 else "heldout-worlds"):
        raise ContractError(f"{path}: whole-world split differs")
    for key in (
        "collector_sha256",
        "curriculum_contract_sha256",
        "source_mjcf_sha256",
        "atlas_sha256",
    ):
        value = meta.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ContractError(f"{path}: {key} must be a lowercase SHA-256")
    if not isinstance(meta.get("world_seed"), int) or not isinstance(meta.get("variation_seed"), int):
        raise ContractError(f"{path}: world/variation seeds are required")
    if meta.get("dt_seconds") != 0.05 or meta.get("transitions") != TICKS or meta.get("residents") != RESIDENTS:
        raise ContractError(f"{path}: timing or dimensions differ")
    if meta.get("context_source") not in ("zero-initial-motor-bootstrap", "controlled-signed-perturbation"):
        raise ContractError(f"{path}: undeclared context source")
    if meta.get("teacher_privileged_fields_retained") is not True:
        raise ContractError(f"{path}: privileged root pose must be declared target-only")
    if meta.get("teacher_privileged_fields_are_model_inputs") is not False:
        raise ContractError(f"{path}: privileged teacher fields cannot be model inputs")
    if meta.get("old_abstract_action_relabelled_as_context") is not False:
        raise ContractError(f"{path}: old action records cannot be context")
    if meta.get("optic_body_are_actual_pre_action_samples") is not True:
        raise ContractError(f"{path}: actual sensory chronology must be declared")
    if not np.array_equal(episode.reset[0], np.ones(RESIDENTS, dtype=np.bool_)) or episode.reset[1:].any():
        raise ContractError(f"{path}: reset must mark only the initial state")
    if not np.array_equal(episode.terminal[-1], np.ones(RESIDENTS, dtype=np.bool_)) or episode.terminal[:-1].any():
        raise ContractError(f"{path}: terminal must mark only the final transition")
    if np.any(np.abs(episode.delivered_context) > 1):
        raise ContractError(f"{path}: context outside [-1,1]")
    if meta["context_source"] == "zero-initial-motor-bootstrap" and np.any(episode.delivered_context != 0):
        raise ContractError(f"{path}: bootstrap context must be exactly zero")
    motor = episode.delivered_motor
    if np.any(motor[..., :24] < 0) or np.any(motor[..., :24] > 1):
        raise ContractError(f"{path}: antagonist motor outside [0,1]")
    if np.any(motor[..., 24:26] < -1) or np.any(motor[..., 24:26] > 1):
        raise ContractError(f"{path}: signed motor outside [-1,1]")
    if np.any(motor[..., 26:] < 0) or np.any(motor[..., 26:] > 1):
        raise ContractError(f"{path}: ancillary motor outside [0,1]")
    if np.any(episode.skill_id >= len(SKILLS)):
        raise ContractError(f"{path}: skill_id outside the frozen curriculum")
    bouts = meta.get("teacher_bouts")
    if not isinstance(bouts, list) or not bouts:
        raise ContractError(f"{path}: teacher bouts missing")
    cursor = 0
    for bout in bouts:
        if not isinstance(bout, dict) or bout.get("start_tick") != cursor:
            raise ContractError(f"{path}: teacher bouts must be gapless")
        end, skill = bout.get("end_tick"), bout.get("skill")
        if not isinstance(end, int) or end <= cursor or skill not in SKILLS:
            raise ContractError(f"{path}: invalid teacher bout")
        skill_index = SKILLS.index(skill)
        if not np.all(episode.skill_id[cursor:end] == skill_index):
            raise ContractError(f"{path}: skill_id differs from teacher bouts")
        cursor = end
    if cursor != TICKS:
        raise ContractError(f"{path}: teacher bouts must cover all transitions")


def seal_corpus(source: str | Path, output: str | Path) -> dict[str, Any]:
    source, output = Path(source).resolve(), Path(output).resolve()
    paths = sorted(source.glob("episode-??-*.npz"))
    if len(paths) != 8:
        raise ContractError(f"expected exactly eight source episodes, found {len(paths)}")
    episodes = [load_episode(path) for path in paths]
    if [episode.metadata["episode_index"] for episode in episodes] != list(range(8)):
        raise ContractError("episodes must be ordered exactly 0..7")
    unique_fields = ("world_seed", "variation_seed", "layout_identity")
    for field in unique_fields:
        values = [episode.metadata.get(field) for episode in episodes]
        if len(set(values)) != 8:
            raise ContractError(f"{field} must be distinct across worlds")
    contracts = {episode.metadata["curriculum_contract_sha256"] for episode in episodes}
    if len(contracts) != 1:
        raise ContractError("curriculum contract differs across episodes")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for episode in episodes:
        target = output / episode.path.name
        try:
            os.link(episode.path, target)
        except OSError:
            target.write_bytes(episode.path.read_bytes())
        records.append(
            {
                "episode_index": episode.metadata["episode_index"],
                "split": episode.split,
                "file": target.name,
                "sha256": episode.sha256,
                "world_seed": episode.metadata["world_seed"],
                "variation_seed": episode.metadata["variation_seed"],
                "layout_identity": episode.metadata["layout_identity"],
                "collector_sha256": episode.metadata["collector_sha256"],
            }
        )
    manifest = {
        "format": CORPUS_FORMAT,
        "completed": True,
        "episodes": records,
        "curriculum_contract_sha256": contracts.pop(),
        "split_policy": "episodes 0..5 train; 6..7 heldout-worlds; heldout worlds never select checkpoints",
        "model_inputs": ["optic_rgb", "body", "delivered_context"],
        "target_only": ["delivered_motor", "teacher_joint_target", "root_pose", "skill_id"],
        "raw_world_geometry_retained": False,
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    temporary = output / ".corpus.json.tmp"
    temporary.write_text(payload)
    os.replace(temporary, output / "corpus.json")
    return manifest


def load_corpus(path: str | Path) -> tuple[list[Episode], list[Episode]]:
    path = Path(path).resolve()
    manifest_path = path / "corpus.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != CORPUS_FORMAT or manifest.get("completed") is not True:
        raise ContractError("incomplete or incompatible corpus")
    records = manifest.get("episodes")
    if not isinstance(records, list) or len(records) != 8:
        raise ContractError("corpus must name exactly eight episodes")
    episodes = []
    for record in records:
        episode_path = path / record["file"]
        if sha256_file(episode_path) != record.get("sha256"):
            raise ContractError(f"{episode_path}: corpus hash differs")
        episode = load_episode(episode_path)
        for key in ("episode_index", "split", "world_seed", "variation_seed", "layout_identity", "collector_sha256"):
            if episode.metadata.get(key) != record.get(key):
                raise ContractError(f"{episode_path}: manifest {key} differs")
        episodes.append(episode)
    train = [episode for episode in episodes if episode.split == "train"]
    heldout = [episode for episode in episodes if episode.split == "heldout-worlds"]
    if len(train) != 6 or len(heldout) != 2:
        raise ContractError("whole-world split must be six train and two heldout")
    return train, heldout


def skill_windows(
    episodes: Iterable[Episode], skill: int, length: int
) -> list[tuple[Episode, int, int]]:
    if not 0 <= skill < len(SKILLS) or length < 1:
        raise ValueError("invalid skill or window length")
    windows: list[tuple[Episode, int, int]] = []
    for episode in episodes:
        matches = episode.skill_id[:, 0] == skill
        start = 0
        while start < TICKS:
            while start < TICKS and not matches[start]:
                start += 1
            end = start
            while end < TICKS and matches[end]:
                end += 1
            for at in range(start, end - length + 1):
                windows.append((episode, at, at + length))
            start = end + 1
    return windows
