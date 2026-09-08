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

from .curriculum import (
    CONTEXT, CONTROL_INDEX, CONTROL_SOURCES, PHASE_INDEX, PHASES,
    RESIDENTS, TICKS, split_for_world,
)


FORMAT: Final = "chreatures-actual-fly-cns-development-corpus-v1"
EPISODE_FORMAT: Final = "chreatures-actual-fly-cns-development-episode-v1"
NURSERY_FORMAT: Final = "chreatures-embodied-nursery-corpus-v1"
RECOVERY_FORMAT: Final = "chreatures-fly-on-policy-recovery-corpus-v1"
SUPPORT_ACQUISITION_FORMAT: Final = "chreatures-fly-support-acquisition-corpus-v1"
SUPPORTED_CONTINUATION_FORMAT: Final = "chreatures-fly-supported-continuation-corpus-v1"
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
ANATOMICAL_BODY_SCHEMA_SHA256: Final = "d8c3ff3d22b7f68ec8fb752ba210689ce6531820df57edc96c3bd305766a2d5a"
CNS_BODY807_SCHEMA_SHA256: Final = "97b48925c5580c883e1e06bac2d14ad458d84c8dec8ed8b25b143975dc0b1786"
MORPHOLOGY_ASSET_SET_SHA256: Final = "2da4b8004d89d2d89f211bd51079d524376a197abcdbbdd9512be1a86ecf6c94"


class FlyLearningContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_numerical_sha256(episode: Episode) -> str:
    """Hash all numerical chronology arrays in the nursery sealer's order."""
    members = (
        "optic_rgb", "body_afferents", "delivered_context", "collected_latent",
        "delivered_motor", "cns_motor", "teacher_motor", "applied_body_control",
        "teacher_valid", "joint_position", "joint_velocity", "segment_pose",
        "ground_contact_raw", "sensory_site_position", "mouth_contact_raw",
        "outcome", "reward", "success", "failure", "reset", "active", "terminal",
        "control_source", "curriculum_phase",
    )
    digest = hashlib.sha256()
    for name in members:
        value = np.ascontiguousarray(getattr(episode, name))
        header = json.dumps(
            {"name": name, "dtype": value.dtype.str, "shape": value.shape},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(4, "little"))
        digest.update(header)
        digest.update(value.tobytes(order="C"))
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
        "initial_snapshot_sha256", "curriculum_plan_sha256",
    )
    if any(not HEX64.fullmatch(str(meta.get(name, ""))) for name in required_hashes):
        raise FlyLearningContractError("episode lacks a required 64-hex identity")
    if meta.get("execution_backend") == "native-fly-world":
        native_hashes = (
            "native_host_binary_sha256", "native_host_source_manifest_sha256",
            "mujoco_library_sha256",
        )
        if any(not HEX64.fullmatch(str(meta.get(name, ""))) for name in native_hashes):
            raise FlyLearningContractError("native episode lacks authenticated deployment identities")
    elif not HEX64.fullmatch(str(meta.get("core_wasm_sha256", ""))):
        raise FlyLearningContractError("Wasm episode lacks its core identity")
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
    _validate_core_lineage(
        root, manifest, [episode.metadata["core_wasm_sha256"] for episode in episodes]
    )
    _validate_metadata_meaning(root, manifest, episodes)
    return Corpus(root, manifest, tuple(episodes[:8]), tuple(episodes[8:10]), tuple(episodes[10:]))


def load_nursery_corpus(path: Path) -> Corpus:
    """Load the separately sealed diverse-layout, physically stimulated corpus."""
    path = path.resolve()
    if path.is_dir():
        path = path / "nursery-corpus.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("format") != NURSERY_FORMAT
        or manifest.get("completed") is not True
        or int(manifest.get("worlds", -1)) != 12
        or int(manifest.get("ticks", -1)) != TICKS
        or int(manifest.get("residents", -1)) != RESIDENTS
    ):
        raise FlyLearningContractError("sealed embodied nursery corpus required")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) != 12:
        raise FlyLearningContractError("nursery corpus must contain twelve whole worlds")
    episodes: list[Episode] = []
    for expected_index, row in enumerate(rows):
        if (
            int(row.get("world_index", -1)) != expected_index
            or row.get("split") != split_for_world(expected_index)
            or not HEX64.fullmatch(str(row.get("sha256", "")))
        ):
            raise FlyLearningContractError("nursery episode order or split differs")
        episode = load_episode(path.parent / str(row["file"]), str(row["sha256"]))
        metadata = episode.metadata
        if (
            int(metadata["world_index"]) != expected_index
            or metadata.get("nursery_format") != NURSERY_FORMAT
            or metadata.get("nursery_raw_stimulus_controller_access") is not False
            or metadata.get("body_schema_sha256") != ANATOMICAL_BODY_SCHEMA_SHA256
            or metadata.get("cns_body807_schema_sha256") != CNS_BODY807_SCHEMA_SHA256
            or metadata.get("morphology_sha256") != MORPHOLOGY_ASSET_SET_SHA256
        ):
            raise FlyLearningContractError("nursery episode contract differs")
        for name in (
            "scene_manifest_sha256", "scene_layout_identity", "nursery_layout_sha256",
            "initial_snapshot_sha256", "nursery_stimulus_schedule_sha256",
        ):
            row_name = "stimulus_schedule_sha256" if name == "nursery_stimulus_schedule_sha256" else name
            if (
                row.get(row_name) != metadata.get(name)
                or not HEX64.fullmatch(str(metadata.get(name, "")))
            ):
                raise FlyLearningContractError(f"nursery manifest {name} differs")
        schedule = metadata.get("nursery_stimulus_schedule")
        deliveries = metadata.get("nursery_stimulus_deliveries")
        if (
            not isinstance(schedule, list) or len(schedule) != 16
            or not isinstance(deliveries, list) or len(deliveries) != 16
            or [int(item.get("tick", -1)) for item in deliveries] != list(range(0, TICKS, 64))
        ):
            raise FlyLearningContractError("nursery episode lacks sixteen physical stimulus receipts")
        schedule_sha = hashlib.sha256(
            json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if schedule_sha != metadata["nursery_stimulus_schedule_sha256"]:
            raise FlyLearningContractError("nursery stimulus schedule checksum differs")
        for planned, delivered in zip(schedule, deliveries, strict=True):
            if delivered.get("planned") != planned:
                raise FlyLearningContractError("nursery planned and delivered stimulus differ")
            sound = delivered.get("bridge_sound")
            if (
                delivered.get("bridge_screen_sha256") != planned.get("screen_frame_sha256")
                or not isinstance(sound, dict)
                or sound.get("position_mm") != planned.get("position_mm")
                or sound.get("frequency_hz") != planned.get("frequency_hz")
                or sound.get("envelope") != planned.get("envelope")
                or sound.get("duration_s") != planned.get("duration_s")
                or abs(float(delivered.get("acknowledged_world_time_s", -1)) - int(delivered["tick"]) * 0.01) > 1e-8
            ):
                raise FlyLearningContractError("nursery bridge acknowledgement differs from plan")
        episodes.append(episode)
    for name in (
        "scene_manifest_sha256", "scene_layout_identity", "nursery_layout_sha256",
        "initial_snapshot_sha256",
    ):
        values = [episode.metadata[name] for episode in episodes]
        if len(set(values)) != 12:
            raise FlyLearningContractError(f"nursery {name} values must be disjoint")
    for name in ("native_runtime_sha256", "core_wasm_sha256", "collector_sha256"):
        if len({episode.metadata[name] for episode in episodes}) != 1:
            raise FlyLearningContractError(f"nursery mixes {name} without an amendment")
    supplemental_rows = manifest.get("supplemental_train")
    if not isinstance(supplemental_rows, list) or len(supplemental_rows) != 1:
        raise FlyLearningContractError("nursery corpus lacks its supplemental receipt row")
    supplemental_row = supplemental_rows[0]
    supplemental_path = path.parent / str(supplemental_row.get("file", ""))
    amendment_path = path.parent / str(supplemental_row.get("amendment_file", ""))
    if (
        supplemental_row.get("duplicate_layout_of_world") != 0
        or sha256_file(supplemental_path) != supplemental_row.get("sha256")
        or sha256_file(amendment_path) != supplemental_row.get("amendment_sha256")
    ):
        raise FlyLearningContractError("nursery supplemental file identity differs")
    supplemental = load_episode(supplemental_path, str(supplemental_row["sha256"]))
    amendment = json.loads(amendment_path.read_text())
    scopes = amendment.get("correct_scopes")
    if (
        amendment.get("format") != "chreatures-nursery-metadata-scope-amendment-v1"
        or amendment.get("sha256") != supplemental.sha256
        or amendment.get("corpus_role") != "supplemental-train-history-with-duplicate-layout"
        or amendment.get("executed_physics_and_cns_valid") is not True
        or amendment.get("metadata_change_only") is not True
        or amendment.get("numerical_dynamics_action_reward_contract_unchanged") is not True
        or amendment.get("usable_for_training") is not True
        or amendment.get("split_constraint") != (
            "train only; shares world/layout with canonical episode-00 and is not a distinct-layout slot"
        )
        or not isinstance(scopes, dict)
        or scopes.get("body_schema_sha256") != ANATOMICAL_BODY_SCHEMA_SHA256
        or scopes.get("cns_body807_schema_sha256") != CNS_BODY807_SCHEMA_SHA256
        or supplemental.metadata.get("body_schema_sha256") != CNS_BODY807_SCHEMA_SHA256
        or supplemental.metadata.get("morphology_sha256") != MORPHOLOGY_ASSET_SET_SHA256
        or supplemental.metadata.get("world_index") != 0
        or supplemental.metadata.get("split") != "train"
        or supplemental.metadata.get("scene_layout_identity") != episodes[0].metadata["scene_layout_identity"]
    ):
        raise FlyLearningContractError("nursery supplemental amendment scope differs")
    supplemental_schedule = supplemental.metadata.get("nursery_stimulus_schedule")
    supplemental_deliveries = supplemental.metadata.get("nursery_stimulus_deliveries")
    if (
        supplemental.metadata.get("nursery_format") != NURSERY_FORMAT
        or supplemental.metadata.get("nursery_raw_stimulus_controller_access") is not False
        or not isinstance(supplemental_schedule, list) or len(supplemental_schedule) != 16
        or not isinstance(supplemental_deliveries, list) or len(supplemental_deliveries) != 16
        or [int(item.get("tick", -1)) for item in supplemental_deliveries] != list(range(0, TICKS, 64))
    ):
        raise FlyLearningContractError("nursery supplemental stimulus contract differs")
    for planned, delivered in zip(supplemental_schedule, supplemental_deliveries, strict=True):
        if (
            delivered.get("planned") != planned
            or delivered.get("bridge_screen_sha256") != planned.get("screen_frame_sha256")
            or delivered.get("bridge_sound", {}).get("frequency_hz") != planned.get("frequency_hz")
        ):
            raise FlyLearningContractError("nursery supplemental stimulus receipt differs")
    supplemental_numerical = episode_numerical_sha256(supplemental)
    canonical_numerical = episode_numerical_sha256(episodes[0])
    distinct = supplemental_numerical != canonical_numerical
    if (
        supplemental_row.get("supplemental_numerical_trajectory_sha256") != supplemental_numerical
        or supplemental_row.get("canonical_world00_numerical_trajectory_sha256") != canonical_numerical
        or supplemental_row.get("numerically_distinct") is not distinct
        or supplemental_row.get("usable_training_row") is not distinct
        or supplemental_row.get("replicate_label") != (
            "same-seed-layout-replicate" if distinct else "numerically-identical-receipt-only"
        )
    ):
        raise FlyLearningContractError("nursery supplemental numerical deduplication differs")
    train = tuple(episodes[:8]) + ((supplemental,) if distinct else tuple())
    return Corpus(
        path.parent, manifest,
        train, tuple(episodes[8:10]), tuple(episodes[10:]),
    )


def load_recovery_corpus(path: Path) -> Corpus:
    """Load native on-policy probe/correction/release chronologies."""
    path = path.resolve()
    if path.is_dir():
        path = path / "recovery-corpus.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("format") != RECOVERY_FORMAT
        or manifest.get("completed") is not True
        or int(manifest.get("worlds", -1)) != 12
        or int(manifest.get("ticks", -1)) != TICKS
        or int(manifest.get("residents", -1)) != RESIDENTS
    ):
        raise FlyLearningContractError("sealed native recovery corpus required")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) != 12:
        raise FlyLearningContractError("recovery corpus must contain twelve whole worlds")
    episodes = []
    for index, row in enumerate(rows):
        if (
            int(row.get("world_index", -1)) != index
            or row.get("split") != split_for_world(index)
            or not HEX64.fullmatch(str(row.get("sha256", "")))
            or not all(HEX64.fullmatch(str(row.get(key, ""))) for key in (
                "scene_layout_identity", "initial_snapshot_sha256"
            ))
        ):
            raise FlyLearningContractError("recovery manifest order or identity differs")
        episode = load_episode(path.parent / str(row["file"]), str(row["sha256"]))
        if (
            episode.metadata.get("recovery_format") != RECOVERY_FORMAT
            or episode.metadata.get("execution_backend") != "native-fly-world"
            or episode.metadata.get("raw_geometry_controller_access") is not False
            or int(episode.metadata.get("world_index", -1)) != index
            or row.get("scene_layout_identity") != episode.metadata["scene_layout_identity"]
            or row.get("initial_snapshot_sha256") != episode.metadata["initial_snapshot_sha256"]
        ):
            raise FlyLearningContractError("native recovery episode contract differs")
        curriculum = episode.metadata.get("curriculum", {})
        if (
            curriculum.get("recovery_format") != RECOVERY_FORMAT
            or curriculum.get("same_state_teacher_takeover") is not True
            or curriculum.get("reset_between_probe_correction_release") is not False
            or curriculum.get("observer_teacher_only") is not True
        ):
            raise FlyLearningContractError("recovery intervention chronology differs")
        episodes.append(episode)
    for key in ("scene_layout_identity", "initial_snapshot_sha256", "world_instance_identity"):
        if len({episode.metadata[key] for episode in episodes}) != 12:
            raise FlyLearningContractError(f"recovery {key} values overlap")
    for key in (
        "native_host_binary_sha256", "native_host_source_manifest_sha256",
        "mujoco_library_sha256", "recovery_source_sha256",
    ):
        if len({episode.metadata[key] for episode in episodes}) != 1:
            raise FlyLearningContractError(f"recovery mixes {key}")
    return Corpus(
        path.parent, manifest, tuple(episodes[:8]),
        tuple(episodes[8:10]), tuple(episodes[10:]),
    )


def load_support_acquisition_corpus(path: Path) -> Corpus:
    """Load fresh-life support, short-probe, and acquisition chronologies."""
    path = path.resolve()
    if path.is_dir():
        path = path / "support-acquisition-corpus.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("format") != SUPPORT_ACQUISITION_FORMAT
        or manifest.get("completed") is not True
        or int(manifest.get("worlds", -1)) != 12
        or int(manifest.get("ticks", -1)) != TICKS
        or int(manifest.get("residents", -1)) != RESIDENTS
        or int(manifest.get("cold_support_ticks", -1)) != 40
        or int(manifest.get("probe_ticks", -1)) != 8
        or manifest.get("reset_between_interventions") is not False
        or manifest.get("observer_teacher_only") is not True
    ):
        raise FlyLearningContractError("sealed support-acquisition corpus required")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) != 12:
        raise FlyLearningContractError("support-acquisition corpus must contain twelve whole worlds")
    episodes = []
    for index, row in enumerate(rows):
        if (
            int(row.get("world_index", -1)) != index
            or row.get("split") != split_for_world(index)
            or not HEX64.fullmatch(str(row.get("sha256", "")))
            or not all(HEX64.fullmatch(str(row.get(key, ""))) for key in (
                "scene_layout_identity", "initial_snapshot_sha256", "collection_life_identity"
            ))
        ):
            raise FlyLearningContractError("support-acquisition manifest order or identity differs")
        episode = load_episode(path.parent / str(row["file"]), str(row["sha256"]))
        meta, curriculum = episode.metadata, episode.metadata.get("curriculum", {})
        if (
            meta.get("support_acquisition_format") != SUPPORT_ACQUISITION_FORMAT
            or meta.get("execution_backend") != "native-fly-world"
            or meta.get("raw_geometry_controller_access") is not False
            or int(meta.get("world_index", -1)) != index
            or row.get("scene_layout_identity") != meta["scene_layout_identity"]
            or row.get("initial_snapshot_sha256") != meta["initial_snapshot_sha256"]
            or row.get("collection_life_identity") != meta["collection_life_identity"]
            or curriculum.get("support_acquisition_format") != SUPPORT_ACQUISITION_FORMAT
            or int(curriculum.get("cold_support_ticks", -1)) != 40
            or int(curriculum.get("probe_ticks", -1)) != 8
            or int(curriculum.get("correction_ticks", -1)) != 16
            or int(curriculum.get("acquisition_ticks", -1)) != 16
            or int(curriculum.get("cycles", -1)) != 24
            or curriculum.get("reset_between_interventions") is not False
            or curriculum.get("observer_teacher_only") is not True
            or float(curriculum.get("teacher_servo_slew_rad_per_tick", -1)) != 0.04
            or float(curriculum.get("teacher_author_hz", -1)) != 1.5
            or float(curriculum.get("teacher_author_magnitude", -1)) != 0.25
        ):
            raise FlyLearningContractError("support-acquisition episode contract differs")
        bouts = curriculum.get("bouts")
        if not isinstance(bouts, list) or len(bouts) != RESIDENTS:
            raise FlyLearningContractError("support-acquisition bout rows differ")
        expected_schedule = [(0, 40, "cold-support")]
        for cycle in range(24):
            start = 40 + 40 * cycle
            expected_schedule.extend((
                (start, start + 8, "short-probe"),
                (start + 8, start + 24, "neutral-correction"),
                (start + 24, start + 40, "contact-acquisition"),
            ))
        expected_schedule.append((1000, TICKS, "tail-support"))
        for resident, resident_bouts in enumerate(bouts):
            if (
                not isinstance(resident_bouts, list)
                or [
                    (int(bout.get("start", -1)), int(bout.get("stop", -1)),
                     bout.get("intended_outcome"))
                    for bout in resident_bouts
                ] != expected_schedule
            ):
                raise FlyLearningContractError("support-acquisition role schedule differs")
            cursor = 0
            for bout in resident_bouts:
                start, stop = int(bout.get("start", -1)), int(bout.get("stop", -1))
                source, phase = bout.get("control_source"), bout.get("phase")
                if (
                    start != cursor or stop <= start
                    or source not in CONTROL_INDEX or phase not in PHASE_INDEX
                ):
                    raise FlyLearningContractError("support-acquisition bouts are not gapless")
                region = np.s_[start:stop, resident]
                expected_motor = (
                    episode.teacher_motor[region]
                    if source == "offline-author-teacher" else episode.cns_motor[region]
                )
                if (
                    not np.array_equal(episode.delivered_motor[region], expected_motor)
                    or not np.all(episode.control_source[region] == CONTROL_INDEX[source])
                    or not np.all(episode.curriculum_phase[region] == PHASE_INDEX[phase])
                ):
                    raise FlyLearningContractError("support-acquisition delivered chronology differs")
                cursor = stop
            if cursor != TICKS:
                raise FlyLearningContractError("support-acquisition bouts do not cover the episode")
        if (
            np.any(episode.delivered_context[:40])
            or np.any(episode.delivered_motor[:40, :, :84])
        ):
            raise FlyLearningContractError("support-acquisition cold support differs")
        teacher_rows = (
            episode.control_source[1:] == CONTROL_INDEX["offline-author-teacher"]
        )
        physical_slew = np.abs(np.diff(episode.applied_body_control[:, :, :84], axis=0))
        if np.any(physical_slew[teacher_rows] > 0.04001):
            raise FlyLearningContractError("support-acquisition teacher slew differs")
        episodes.append(episode)
    for key in (
        "scene_layout_identity", "initial_snapshot_sha256",
        "world_instance_identity", "collection_life_identity",
    ):
        if len({episode.metadata[key] for episode in episodes}) != 12:
            raise FlyLearningContractError(f"support-acquisition {key} values overlap")
    for key in (
        "native_host_binary_sha256", "native_host_source_manifest_sha256",
        "mujoco_library_sha256", "support_acquisition_source_sha256",
    ):
        if len({episode.metadata[key] for episode in episodes}) != 1:
            raise FlyLearningContractError(f"support-acquisition mixes {key}")
    return Corpus(
        path.parent, manifest, tuple(episodes[:8]),
        tuple(episodes[8:10]), tuple(episodes[10:]),
    )


def load_supported_continuation_corpus(path: Path) -> Corpus:
    """Load long supplied-author continuations and rare terminal CNS probes."""
    path = path.resolve()
    if path.is_dir():
        path = path / "supported-continuation-corpus.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("format") != SUPPORTED_CONTINUATION_FORMAT
        or manifest.get("completed") is not True
        or int(manifest.get("worlds", -1)) != 12
        or int(manifest.get("ticks", -1)) != TICKS
        or int(manifest.get("residents", -1)) != RESIDENTS
        or int(manifest.get("cold_support_ticks", -1)) != 40
        or int(manifest.get("author_bout_ticks", -1)) != 160
        or int(manifest.get("author_bouts", -1)) != 6
        or int(manifest.get("probe_start", -1)) != 1000
        or int(manifest.get("probe_ticks", -1)) != 8
        or int(manifest.get("probe_residents_per_world", -1)) != 2
        or manifest.get("context") != "exact zero12 throughout"
        or manifest.get("reset_between_interventions") is not False
        or manifest.get("observer_teacher_only") is not True
    ):
        raise FlyLearningContractError("sealed supported-continuation corpus required")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) != 12:
        raise FlyLearningContractError("supported-continuation corpus must contain twelve whole worlds")
    from .supported_continuation import SHARED_KEYS, validate_episode

    episodes = []
    for index, row in enumerate(rows):
        if (
            int(row.get("world_index", -1)) != index
            or row.get("split") != split_for_world(index)
            or not HEX64.fullmatch(str(row.get("sha256", "")))
            or not all(HEX64.fullmatch(str(row.get(key, ""))) for key in (
                "scene_layout_identity", "initial_snapshot_sha256", "collection_life_identity"
            ))
        ):
            raise FlyLearningContractError("supported-continuation manifest order or identity differs")
        episode = load_episode(path.parent / str(row["file"]), str(row["sha256"]))
        meta = episode.metadata
        if (
            meta.get("supported_continuation_format") != SUPPORTED_CONTINUATION_FORMAT
            or meta.get("execution_backend") != "native-fly-world"
            or meta.get("raw_geometry_controller_access") is not False
            or int(meta.get("world_index", -1)) != index
            or row.get("scene_layout_identity") != meta["scene_layout_identity"]
            or row.get("initial_snapshot_sha256") != meta["initial_snapshot_sha256"]
            or row.get("collection_life_identity") != meta["collection_life_identity"]
        ):
            raise FlyLearningContractError("supported-continuation episode identity differs")
        try:
            validate_episode(episode, index)
        except (KeyError, TypeError, ValueError) as error:
            raise FlyLearningContractError(
                f"supported-continuation episode {index} contract differs"
            ) from error
        episodes.append(episode)
    shared = manifest.get("shared_identity")
    if not isinstance(shared, dict) or set(shared) != set(SHARED_KEYS):
        raise FlyLearningContractError("supported-continuation shared identity differs")
    for key in SHARED_KEYS:
        if shared[key] != episodes[0].metadata[key] or any(
            episode.metadata[key] != shared[key] for episode in episodes[1:]
        ):
            raise FlyLearningContractError(f"supported-continuation mixes {key}")
    if manifest.get("source_service_sha256") != shared["cns_service_sha256"]:
        raise FlyLearningContractError("supported-continuation source service differs")
    for key in (
        "scene_layout_identity", "initial_snapshot_sha256",
        "world_instance_identity", "collection_life_identity",
    ):
        if len({episode.metadata[key] for episode in episodes}) != 12:
            raise FlyLearningContractError(f"supported-continuation {key} values overlap")
    return Corpus(
        path.parent, manifest, tuple(episodes[:8]),
        tuple(episodes[8:10]), tuple(episodes[10:]),
    )


def combine_corpora(primary: Corpus, *additional: Corpus) -> Corpus:
    """Combine compatible physical contracts while retaining whole-world splits."""
    corpora = (primary, *additional)
    episodes = tuple(
        episode
        for corpus in corpora
        for episode in (*corpus.train, *corpus.validation, *corpus.heldout)
    )
    contract_keys = (
        "morphology_sha256", "motor_atlas_sha256",
        "retina_mapping_sha256", "body_afferent_dim", "motor_dim", "outcome_dim",
        "sensory_dim", "body_afferent_rows", "motor_rows", "control_dt_s",
    )
    for name in contract_keys:
        expected = episodes[0].metadata[name]
        if any(episode.metadata[name] != expected for episode in episodes[1:]):
            raise FlyLearningContractError(f"bootstrap/nursery {name} differs")
    if primary.manifest.get("anatomical_body_schema_sha256") != ANATOMICAL_BODY_SCHEMA_SHA256:
        raise FlyLearningContractError("primary corpus anatomical body schema differs")
    for corpus in additional:
        for episode in (*corpus.train, *corpus.validation, *corpus.heldout):
            if (
                episode.metadata["body_schema_sha256"] != ANATOMICAL_BODY_SCHEMA_SHA256
                or episode.metadata["cns_body807_schema_sha256"] != CNS_BODY807_SCHEMA_SHA256
            ):
                raise FlyLearningContractError("combined corpus body schema semantics differ")
    return Corpus(
        primary.root,
        {
            "format": "chreatures-actual-fly-cns-combined-training-view-v1",
            "completed": True,
            "sources": [corpus.manifest["format"] for corpus in corpora],
        },
        tuple(episode for corpus in corpora for episode in corpus.train),
        tuple(episode for corpus in corpora for episode in corpus.validation),
        tuple(episode for corpus in corpora for episode in corpus.heldout),
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_core_lineage(
    root: Path, manifest: dict[str, Any], core_hashes: list[str]
) -> dict[str, Any] | None:
    """Require a portable, byte-authenticated receipt for a bounded core repair."""
    unique = list(dict.fromkeys(core_hashes))
    amendment = manifest.get("source_amendment")
    if len(unique) == 1:
        if amendment is not None:
            raise FlyLearningContractError("single-core corpus must not claim a source amendment")
        return None
    if len(unique) != 2 or not isinstance(amendment, dict):
        raise FlyLearningContractError("mixed core Wasm lineage lacks an exact bounded amendment")
    if amendment.get("file") != "source-amendment.json" or not HEX64.fullmatch(
        str(amendment.get("sha256", ""))
    ):
        raise FlyLearningContractError("source amendment file identity differs")
    path = root / "source-amendment.json"
    if sha256_file(path) != amendment["sha256"]:
        raise FlyLearningContractError("source amendment checksum differs")
    value = json.loads(path.read_text())
    if value != amendment.get("value"):
        raise FlyLearningContractError("source amendment embedded value differs")
    first_after = int(value.get("first_after_world_index", -1))
    if (
        value.get("format") != "chreatures-fly-corpus-source-amendment-v1"
        or value.get("compatible_semantic_contract") is not True
        or value.get("before_core_wasm_sha256") != unique[0]
        or value.get("after_core_wasm_sha256") != unique[1]
        or core_hashes != [unique[0]] * first_after + [unique[1]] * (len(core_hashes) - first_after)
    ):
        raise FlyLearningContractError("source amendment does not exactly cover core transition")
    return amendment


def _validate_metadata_meaning(
    root: Path, manifest: dict[str, Any], episodes: list[Episode]
) -> dict[str, Any]:
    """Authenticate the bootstrap's mislabeled BODY807 receipt field."""
    amendment = manifest.get("metadata_meaning_amendment")
    if not isinstance(amendment, dict) or amendment.get("file") != "metadata-meaning-amendment.json":
        raise FlyLearningContractError("bootstrap corpus lacks metadata meaning amendment")
    if not HEX64.fullmatch(str(amendment.get("sha256", ""))):
        raise FlyLearningContractError("metadata meaning amendment identity differs")
    path = root / "metadata-meaning-amendment.json"
    if sha256_file(path) != amendment["sha256"]:
        raise FlyLearningContractError("metadata meaning amendment checksum differs")
    value = json.loads(path.read_text())
    if value != amendment.get("value"):
        raise FlyLearningContractError("metadata meaning amendment embedded value differs")
    expected_fixture = episodes[0].metadata["scene_manifest_sha256"]
    if (
        value.get("format") != "chreatures-fly-corpus-metadata-meaning-amendment-v1"
        or value.get("compatible_semantic_contract") is not True
        or value.get("applies_to_world_indices") != [0, 11]
        or value.get("collector_source_revision") != episodes[0].metadata["source_revision"]
        or value.get("frozen_world_fixture_sha256") != expected_fixture
        or value.get("recorded_body_schema_field_meaning") != "cns_BODY807_sensory_schema_sha256"
        or value.get("recorded_body_schema_sha256") != CNS_BODY807_SCHEMA_SHA256
        or value.get("actual_anatomical_body_schema_sha256") != ANATOMICAL_BODY_SCHEMA_SHA256
        or value.get("actual_morphology_asset_set_sha256") != MORPHOLOGY_ASSET_SET_SHA256
    ):
        raise FlyLearningContractError("metadata meaning amendment does not match frozen body contract")
    if any(
        episode.metadata["body_schema_sha256"] != CNS_BODY807_SCHEMA_SHA256
        or episode.metadata["morphology_sha256"] != MORPHOLOGY_ASSET_SET_SHA256
        or episode.metadata["scene_manifest_sha256"] != expected_fixture
        for episode in episodes
    ):
        raise FlyLearningContractError("episode identities differ from metadata meaning amendment")
    return amendment


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
        meaning_path = source / "metadata-meaning-amendment.json"
        meaning_value = json.loads(meaning_path.read_text())
        if (
            meaning_value.get("format") != "chreatures-fly-corpus-metadata-meaning-amendment-v1"
            or meaning_value.get("compatible_semantic_contract") is not True
            or meaning_value.get("applies_to_world_indices") != [0, 11]
            or meaning_value.get("collector_source_revision") != shared["source_revision"]
            or meaning_value.get("frozen_world_fixture_sha256") != shared["scene_manifest_sha256"]
            or meaning_value.get("recorded_body_schema_field_meaning") != "cns_BODY807_sensory_schema_sha256"
            or meaning_value.get("recorded_body_schema_sha256") != CNS_BODY807_SCHEMA_SHA256
            or meaning_value.get("actual_anatomical_body_schema_sha256") != ANATOMICAL_BODY_SCHEMA_SHA256
            or meaning_value.get("actual_morphology_asset_set_sha256") != MORPHOLOGY_ASSET_SET_SHA256
            or shared["body_schema_sha256"] != CNS_BODY807_SCHEMA_SHA256
            or shared["morphology_sha256"] != MORPHOLOGY_ASSET_SET_SHA256
        ):
            raise FlyLearningContractError("metadata meaning amendment does not match frozen body contract")
        meaning_copy = staged / "metadata-meaning-amendment.json"
        shutil.copy2(meaning_path, meaning_copy)
        meaning_amendment = {
            "file": meaning_copy.name,
            "sha256": sha256_file(meaning_copy),
            "value": meaning_value,
        }
        core_hashes = [episode.metadata["core_wasm_sha256"] for episode in episodes]
        unique_cores = list(dict.fromkeys(core_hashes))
        amendment = None
        if len(unique_cores) > 1:
            amendment_path = source / "source-amendment.json"
            amendment_value = json.loads(amendment_path.read_text())
            if (
                amendment_value.get("format") != "chreatures-fly-corpus-source-amendment-v1"
                or amendment_value.get("compatible_semantic_contract") is not True
                or amendment_value.get("before_core_wasm_sha256") != unique_cores[0]
                or amendment_value.get("after_core_wasm_sha256") != unique_cores[1]
                or amendment_value.get("first_after_world_index") != core_hashes.index(unique_cores[1])
                or len(unique_cores) != 2
            ):
                raise FlyLearningContractError("mixed core Wasm lineage lacks exact bounded amendment")
            amendment_copy = staged / "source-amendment.json"
            shutil.copy2(amendment_path, amendment_copy)
            amendment_sha = sha256_file(amendment_copy)
            amendment = {
                "file": amendment_copy.name,
                "sha256": amendment_sha,
                "value": amendment_value,
            }
        manifest = {
            "format": FORMAT, "completed": True, "worlds": 12, "ticks": TICKS,
            "residents": RESIDENTS, "split": {"train": [0, 7], "validation-worlds": [8, 9],
                                                  "heldout-worlds": [10, 11]},
            "private_goal_horizon_seconds": 0.4,
            **shared, "episodes": rows,
            "core_wasm_sha256_by_episode": core_hashes,
            "source_amendment": amendment,
            "metadata_meaning_amendment": meaning_amendment,
            "anatomical_body_schema_sha256": ANATOMICAL_BODY_SCHEMA_SHA256,
            "cns_body807_schema_sha256": CNS_BODY807_SCHEMA_SHA256,
            "morphology_asset_set_sha256": MORPHOLOGY_ASSET_SET_SHA256,
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
