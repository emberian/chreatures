"""Fail-closed corpus contract for actual articulated-fly CNS learning."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Final

import numpy as np

from .curriculum import CONTEXT, CONTROL_SOURCES, PHASES, RESIDENTS, TICKS, split_for_world


FORMAT: Final = "chreatures-actual-fly-cns-development-corpus-v1"
EPISODE_FORMAT: Final = "chreatures-actual-fly-cns-development-episode-v1"
OPTIC_SITES: Final = 1771
LATENT: Final = 512
BODY_AFFERENTS: Final = 807
SENSORY: Final = 6120
BODY_ROWS: Final = 11798
MOTOR_ROWS: Final = 815
JOINTS: Final = 126
SEGMENTS: Final = 69
FEET: Final = 6
SITES: Final = 5  # four olfactory anchors plus the mouth anchor
MOTOR: Final = 92  # 84 position, six adhesion, pharyngeal pump, salivary drive
BODY_CONTROL: Final = 90
OUTCOME_NAMES: Final = (
    "upright_score",
    "thorax_height_progress",
    "forward_progress",
    "yaw_progress",
    "translation_speed",
    "angular_speed",
    "support_contact_fraction",
    "surface_slip_speed",
    "antenna_proximity_score",
    "mouth_contact_fraction",
    "chemical_gradient_progress",
    "material_displacement",
    "actual_intake_flow",
    "post_transition_stability",
    "actuator_effort",
    "regulatory_reserve_progress",
)
OUTCOMES: Final = len(OUTCOME_NAMES)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class FlyLearningContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in archive.files:
        raise FlyLearningContractError("episode lacks scalar metadata")
    value = archive["metadata"]
    if value.shape != () or value.dtype.kind != "U":
        raise FlyLearningContractError("metadata must be a scalar Unicode JSON value")
    result = json.loads(str(value))
    if result.get("format") != EPISODE_FORMAT:
        raise FlyLearningContractError("episode format differs")
    return result


def _array(archive: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    if name not in archive.files:
        raise FlyLearningContractError(f"episode lacks {name}")
    value = np.asarray(archive[name])
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise FlyLearningContractError(f"{name} expected {shape}/{np.dtype(dtype)}, got {value.shape}/{value.dtype}")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise FlyLearningContractError(f"{name} contains non-finite values")
    return value


@dataclass(frozen=True)
class Episode:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    optic_rgb: np.ndarray
    body_afferents: np.ndarray
    delivered_context: np.ndarray
    collected_latent: np.ndarray
    delivered_motor: np.ndarray
    cns_motor: np.ndarray
    teacher_motor: np.ndarray
    applied_body_control: np.ndarray
    teacher_valid: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    segment_pose: np.ndarray
    ground_contact_raw: np.ndarray
    sensory_site_position: np.ndarray
    mouth_contact_raw: np.ndarray
    outcome: np.ndarray
    reward: np.ndarray
    success: np.ndarray
    failure: np.ndarray
    reset: np.ndarray
    active: np.ndarray
    terminal: np.ndarray
    control_source: np.ndarray
    curriculum_phase: np.ndarray

    @property
    def model_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The only lifetime/model ingress exposed by the loader."""
        return self.optic_rgb, self.body_afferents, self.delivered_context, self.reset


@dataclass(frozen=True)
class Corpus:
    root: Path
    manifest: dict[str, Any]
    train: tuple[Episode, ...]
    validation: tuple[Episode, ...]
    heldout: tuple[Episode, ...]


def load_episode(path: Path, expected_sha256: str | None = None) -> Episode:
    file_sha = sha256_file(path)
    if expected_sha256 is not None and file_sha != expected_sha256:
        raise FlyLearningContractError(f"episode checksum differs: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        meta = _metadata(archive)
        ticks, residents = int(meta["ticks"]), int(meta["residents"])
        body_dim, motor_dim, outcome_dim = (
            int(meta[name]) for name in ("body_afferent_dim", "motor_dim", "outcome_dim")
        )
        if (ticks, residents) != (TICKS, RESIDENTS):
            raise FlyLearningContractError("episode must be 1024 ticks by four actual physical residents")
        if body_dim != BODY_AFFERENTS or motor_dim != MOTOR or outcome_dim != OUTCOMES:
            raise FlyLearningContractError("frozen BODY807/MOTOR92/outcome dimensions differ")
        if meta.get("outcome_names") != list(OUTCOME_NAMES):
            raise FlyLearningContractError("physical outcome order differs")
        if (int(meta["sensory_dim"]), int(meta["body_afferent_rows"]), int(meta["motor_rows"])) != (
            SENSORY, BODY_ROWS, MOTOR_ROWS
        ):
            raise FlyLearningContractError("frozen sensory6120/body-row11798/motor-row815 dimensions differ")
        if meta.get("cns_format") != "chreatures-cns-service-v4":
            raise FlyLearningContractError("current CHCNS4 corpus required")
        physics_dt = float(meta["physics_dt_s"])
        control_dt = float(meta["control_dt_s"])
        control_substeps = int(meta["control_substeps"])
        cns_dt = float(meta["cns_dt_s"])
        cns_substeps = int(meta["cns_substeps"])
        if abs(physics_dt - 1e-4) > 1e-12 or abs(control_dt - 0.01) > 1e-12 or control_substeps != 100:
            raise FlyLearningContractError("physical/control timing contract differs")
        if abs(control_dt - physics_dt * control_substeps) > 1e-9:
            raise FlyLearningContractError("control interval does not equal its physical substeps")
        if abs(cns_dt - 0.005) > 1e-12 or cns_substeps != 2 or abs(control_dt - cns_dt * cns_substeps) > 1e-12:
            raise FlyLearningContractError("CNS recurrence must take two 0.005-second substeps")
        if meta.get("terminal_context_semantics") != "repeat-last-delivered":
            raise FlyLearningContractError("terminal CNS cache context semantics differ")
        expected_members = {
            "metadata", "optic_rgb", "body_afferents", "delivered_context", "collected_latent",
            "delivered_motor", "cns_motor", "teacher_motor", "applied_body_control", "teacher_valid",
            "joint_position", "joint_velocity", "segment_pose", "ground_contact_raw",
            "sensory_site_position", "mouth_contact_raw", "outcome", "reward", "success", "failure",
            "reset", "active", "terminal", "control_source", "curriculum_phase",
        }
        if set(archive.files) != expected_members:
            raise FlyLearningContractError(f"episode members differ: {sorted(set(archive.files) ^ expected_members)}")
        values = {
            "optic_rgb": _array(archive, "optic_rgb", (ticks + 1, residents, OPTIC_SITES, 3), "<f4"),
            "body_afferents": _array(archive, "body_afferents", (ticks + 1, residents, body_dim), "<f4"),
            "delivered_context": _array(archive, "delivered_context", (ticks, residents, CONTEXT), "<f4"),
            "collected_latent": _array(archive, "collected_latent", (ticks + 1, residents, LATENT), "<f4"),
            "delivered_motor": _array(archive, "delivered_motor", (ticks, residents, MOTOR), "<f4"),
            "cns_motor": _array(archive, "cns_motor", (ticks, residents, MOTOR), "<f4"),
            "teacher_motor": _array(archive, "teacher_motor", (ticks, residents, MOTOR), "<f4"),
            "applied_body_control": _array(
                archive, "applied_body_control", (ticks, residents, BODY_CONTROL), "<f4"
            ),
            "teacher_valid": _array(archive, "teacher_valid", (ticks, residents), "|b1"),
            "joint_position": _array(archive, "joint_position", (ticks + 1, residents, JOINTS), "<f4"),
            "joint_velocity": _array(archive, "joint_velocity", (ticks + 1, residents, JOINTS), "<f4"),
            "segment_pose": _array(archive, "segment_pose", (ticks + 1, residents, SEGMENTS, 7), "<f4"),
            "ground_contact_raw": _array(
                archive, "ground_contact_raw", (ticks + 1, residents, FEET, 16), "<f4"
            ),
            "sensory_site_position": _array(archive, "sensory_site_position", (ticks + 1, residents, SITES, 3), "<f4"),
            "mouth_contact_raw": _array(archive, "mouth_contact_raw", (ticks + 1, residents, 4), "<f4"),
            "outcome": _array(archive, "outcome", (ticks, residents, outcome_dim), "<f4"),
            "reward": _array(archive, "reward", (ticks, residents), "<f4"),
            "success": _array(archive, "success", (ticks, residents), "|b1"),
            "failure": _array(archive, "failure", (ticks, residents), "|b1"),
            "reset": _array(archive, "reset", (ticks + 1, residents), "|b1"),
            "active": _array(archive, "active", (ticks, residents), "|b1"),
            "terminal": _array(archive, "terminal", (ticks, residents), "|b1"),
            "control_source": _array(archive, "control_source", (ticks, residents), "|u1"),
            "curriculum_phase": _array(archive, "curriculum_phase", (ticks, residents), "|u1"),
        }
    if not np.all(values["reset"][0]) or np.any(values["reset"][1:]):
        raise FlyLearningContractError("each episode must have one exact reset at t=0")
    if not np.all(values["terminal"][-1]) or np.any(values["terminal"][:-1]):
        raise FlyLearningContractError("each chronology must terminate exactly after tick1023")
    for name in ("optic_rgb",):
        if values[name].min(initial=0) < -1e-6 or values[name].max(initial=0) > 1.000001:
            raise FlyLearningContractError(f"{name} leaves normalized RGB bounds")
    if np.any(values["success"] & values["failure"]):
        raise FlyLearningContractError("a transition cannot be both successful and failed")
    if values["control_source"].max(initial=0) >= len(CONTROL_SOURCES):
        raise FlyLearningContractError("unknown control-source index")
    if values["curriculum_phase"].max(initial=0) >= len(PHASES):
        raise FlyLearningContractError("unknown curriculum phase index")
    if np.max(np.abs(values["delivered_context"]), initial=0) > 1.000001:
        raise FlyLearningContractError("delivered context leaves signed bounds")
    for name in ("delivered_motor", "cns_motor", "teacher_motor"):
        value = values[name]
        if np.max(np.abs(value[..., :84]), initial=0) > 1.000001:
            raise FlyLearningContractError(f"{name} servo targets leave signed bounds")
        if value[..., 84:].min(initial=0) < -1e-6 or value[..., 84:].max(initial=0) > 1.000001:
            raise FlyLearningContractError(f"{name} adhesion/oral values leave [0,1]")
    applied = values["applied_body_control"]
    if np.max(np.abs(applied[..., :84]), initial=0) > 3.140001:
        raise FlyLearningContractError("applied author position target leaves radian bounds")
    if applied[..., 84:].min(initial=0) < -1e-6 or applied[..., 84:].max(initial=0) > 1.000001:
        raise FlyLearningContractError("applied author adhesion leaves [0,1]")
    if not HEX40.fullmatch(str(meta.get("source_revision", ""))):
        raise FlyLearningContractError("episode source revision must be a full Git identity")
    required_hashes = (
        "collector_sha256", "body_schema_sha256", "morphology_sha256",
        "motor_atlas_sha256", "cns_service_sha256", "cns_adapter_sha256", "motor_calibration_sha256",
        "retina_mapping_sha256", "scene_manifest_sha256", "world_instance_identity",
        "scene_layout_identity", "native_runtime_sha256",
        "core_wasm_sha256",
        "initial_snapshot_sha256", "curriculum_plan_sha256",
    )
    if any(not HEX64.fullmatch(str(meta.get(name, ""))) for name in required_hashes):
        raise FlyLearningContractError("episode lacks a required 64-hex identity")
    if int(meta.get("world_seed", -1)) < 0 or int(meta.get("variation_seed", -1)) < 0:
        raise FlyLearningContractError("episode lacks deterministic world seeds")
    index = int(meta["world_index"])
    if meta.get("split") != split_for_world(index):
        raise FlyLearningContractError("episode split differs from frozen whole-world split")
    if meta.get("raw_geometry_controller_access") is not False:
        raise FlyLearningContractError("raw observer geometry must be excluded from controller access")
    return Episode(path.resolve(), file_sha, meta, **values)


def load_corpus(root: Path) -> Corpus:
    root = root.resolve()
    manifest_path = root / "corpus.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("completed") is not True:
        raise FlyLearningContractError("sealed completed corpus required")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) != 12:
        raise FlyLearningContractError("corpus must contain twelve whole-world episodes")
    episodes: list[Episode] = []
    for expected_index, row in enumerate(rows):
        if int(row.get("world_index", -1)) != expected_index or not HEX64.fullmatch(str(row.get("sha256", ""))):
            raise FlyLearningContractError("manifest episode order/identity differs")
        episode = load_episode(root / str(row["file"]), str(row["sha256"]))
        if int(episode.metadata["world_index"]) != expected_index:
            raise FlyLearningContractError("episode world index differs from manifest")
        if row.get("core_wasm_sha256") != episode.metadata["core_wasm_sha256"]:
            raise FlyLearningContractError("manifest core Wasm identity differs")
        episodes.append(episode)
    identity_keys = ("world_instance_identity", "initial_snapshot_sha256")
    for key in identity_keys:
        values = [episode.metadata[key] for episode in episodes]
        if len(set(values)) != len(values):
            raise FlyLearningContractError(f"whole-world {key} values overlap")
    return Corpus(root, manifest, tuple(episodes[:8]), tuple(episodes[8:10]), tuple(episodes[10:]))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal_corpus(source: Path, output: Path) -> dict[str, Any]:
    """Authenticate twelve collector episodes and copy them without rewriting."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FlyLearningContractError("refusing to overwrite a corpus directory")
    staged = output.with_name(f".{output.name}.{os.getpid()}.staging")
    if staged.exists():
        raise FlyLearningContractError("staging directory already exists")
    staged.mkdir(parents=True)
    rows, episodes = [], []
    try:
        for world_index in range(12):
            name = f"episode-{world_index:02d}.npz"
            source_path = source / name
            episode = load_episode(source_path)
            if int(episode.metadata["world_index"]) != world_index:
                raise FlyLearningContractError(f"{name} has the wrong world index")
            destination = staged / name
            try:
                os.link(source_path, destination)
            except OSError:
                shutil.copy2(source_path, destination)
            if sha256_file(destination) != episode.sha256:
                raise FlyLearningContractError(f"{name} changed while sealing")
            rows.append({
                "world_index": world_index, "file": name, "sha256": episode.sha256,
                "split": split_for_world(world_index),
                "world_instance_identity": episode.metadata["world_instance_identity"],
                "scene_layout_identity": episode.metadata["scene_layout_identity"],
                "core_wasm_sha256": episode.metadata["core_wasm_sha256"],
                "initial_snapshot_sha256": episode.metadata["initial_snapshot_sha256"],
            })
            episodes.append(episode)
        shared_keys = (
            "source_revision", "collector_sha256", "body_schema_sha256", "morphology_sha256",
            "motor_atlas_sha256", "cns_service_sha256", "cns_adapter_sha256",
            "motor_calibration_sha256", "retina_mapping_sha256", "scene_manifest_sha256",
            "scene_layout_identity", "native_runtime_sha256",
            "body_afferent_dim", "motor_dim", "outcome_dim", "sensory_dim",
            "body_afferent_rows", "motor_rows", "cns_format",
            "physics_dt_s", "control_dt_s", "control_substeps", "cns_dt_s", "cns_substeps",
            "terminal_context_semantics", "outcome_names",
        )
        shared = {key: episodes[0].metadata[key] for key in shared_keys}
        for key, expected in shared.items():
            if any(episode.metadata[key] != expected for episode in episodes[1:]):
                raise FlyLearningContractError(f"mixed {key} across corpus")
        core_hashes = [episode.metadata["core_wasm_sha256"] for episode in episodes]
        unique_cores = list(dict.fromkeys(core_hashes))
        amendment = None
        if len(unique_cores) > 1:
            amendment_path = source / "source-amendment.json"
            amendment = json.loads(amendment_path.read_text())
            if (
                amendment.get("format") != "chreatures-fly-corpus-source-amendment-v1"
                or amendment.get("compatible_semantic_contract") is not True
                or amendment.get("before_core_wasm_sha256") != unique_cores[0]
                or amendment.get("after_core_wasm_sha256") != unique_cores[1]
                or amendment.get("first_after_world_index") != core_hashes.index(unique_cores[1])
                or len(unique_cores) != 2
            ):
                raise FlyLearningContractError("mixed core Wasm lineage lacks exact bounded amendment")
            amendment = {
                "path": str(amendment_path.resolve()),
                "sha256": sha256_file(amendment_path),
                "value": amendment,
            }
        manifest = {
            "format": FORMAT, "completed": True, "worlds": 12, "ticks": TICKS,
            "residents": RESIDENTS, "split": {"train": [0, 7], "validation-worlds": [8, 9],
                                                  "heldout-worlds": [10, 11]},
            "private_goal_horizon_seconds": 0.4,
            **shared, "episodes": rows,
            "core_wasm_sha256_by_episode": core_hashes,
            "source_amendment": amendment,
            "model_ingress": ["optic_rgb", "body_afferents", "delivered_context", "reset"],
            "artifact_bound_cache": ["collected_latent", "cns_motor"],
            "target_or_evaluator_only": [
                "teacher_motor", "applied_body_control", "joint_position", "joint_velocity", "segment_pose",
                "ground_contact_raw", "sensory_site_position", "mouth_contact_raw", "outcome",
                "reward", "success", "failure", "curriculum_phase", "control_source",
            ],
            "raw_geometry_controller_access": False,
        }
        _atomic_json(staged / "corpus.json", manifest)
        os.replace(staged, output)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    # Re-open through the public contract before reporting completion.
    corpus = load_corpus(output)
    return {
        "path": str(output), "manifest_sha256": sha256_file(output / "corpus.json"),
        "episode_sha256": [episode.sha256 for episode in (*corpus.train, *corpus.validation, *corpus.heldout)],
    }
