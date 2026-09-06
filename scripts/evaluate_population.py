#!/usr/bin/env python3
"""Evaluate one immutable heterogeneous population batch in current worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.neural_genotype import (
    NeuralVariantRecipe,
    compile_population_phenotypes,
)
from chreatures.organism_interface import (
    ACTION_DIM,
    ACTION_NAMES,
    NEURAL_DIM,
    OBSERVATION_DIM,
    PHYSIOLOGY_DIM,
    PHYSIOLOGY_NAMES,
    PREVIOUS_DIM,
)
from chreatures.population import CandidateGenome, canonical_bytes, content_sha256
from chreatures.sensorimotor_worker_native import DevelopmentalResidentCohort
from chreatures.training_cohort import (
    OUTCOME_FIELDS,
    TrainingCohortBrain,
    WorldTrainingPool,
    load_training_graph,
)


FORMAT = "chreatures-population-episode-evaluation-v1"
CHECKPOINT_FORMAT = "chreatures-population-coupled-checkpoint-v1"
ASSIGNMENT_FORMAT = "chreatures-population-evaluation-assignments-v1"
DT = 0.05
ORGAN_FLOW_ORDER = ("release_mass", "secretion_mass", "allocation_mass")
TRAJECTORY_RESIDENT_FIELDS = frozenset({
    "valid_ticks", "has_valid_observation", "valid_time_seconds",
    "visited_spatial_cells", "mouth_contact_ticks", "mouth_contact_bouts",
    "contact_ticks", "contact_bouts", "quiet_ticks", "outside_world_ticks",
    "outside_deviation_sum", "outside_deviation_max", "physiology_mean",
    "physiology_min", "physiology_max", "executed_action_mean",
    "executed_action_abs_mean", "outcome_sum", "organ_flow_sum", "contact_sum",
    "distance_sum", "effort_sum", "mechanical_work_sum", "ingested_mass_sum",
    "signal_activity_sum", "release_mass_sum", "secretion_mass_sum",
    "allocation_mass_sum", "energy_change", "mean_actual_speed", "height_mean",
    "height_range",
})
CONTROLLER_OUTCOME_FIELDS = (
    "reward", "completed", "summed_return", "attributed", "learned",
    "completed_total", "learned_total", "frozen_total", "skipped_total",
    "cancelled_total",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_bytes(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, json.dumps(
        jsonable(value), indent=2, sort_keys=True, allow_nan=False
    ).encode() + b"\n")


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--resident-artifact", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--neural-recipe", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=600)
    parser.add_argument("--telemetry-every", type=int, default=120)
    parser.add_argument("--gam-trace-chunk-steps", type=int, default=0,
                        help="write causal row-level GAM trace chunks; zero disables")
    parser.add_argument("--spatial-bin-width", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--brain-backend", choices=("tiled", "triton"), default="tiled")
    parser.add_argument("--physical-backend", choices=("fast",), default="fast")
    parser.add_argument("--action-mode", choices=("sample", "map"), default="sample")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not 1 <= args.steps <= 10_000_000:
        raise SystemExit("steps must be in 1..10000000")
    if not 1 <= args.checkpoint_every <= args.steps:
        raise SystemExit("checkpoint-every must be in 1..steps")
    if not 1 <= args.telemetry_every <= args.steps:
        raise SystemExit("telemetry-every must be in 1..steps")
    if not 0 <= args.gam_trace_chunk_steps <= args.steps:
        raise SystemExit("gam-trace-chunk-steps must be zero or in 1..steps")
    if not np.isfinite(args.spatial_bin_width) or args.spatial_bin_width <= 0:
        raise SystemExit("spatial-bin-width must be finite and positive")
    for path in (
        args.profile,
        args.resident_artifact,
        args.port_bundle,
        args.neural_recipe,
        args.assignments,
    ):
        if not path.is_file():
            raise SystemExit(f"required input is absent: {path}")
    if not args.graph.is_dir():
        raise SystemExit(f"graph directory is absent: {args.graph}")
    if args.resume:
        if not (args.output / "identity.json").is_file():
            raise SystemExit("resume output has no sealed evaluation identity")
    elif args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("new evaluation output must be absent or empty")


def source_identity() -> dict[str, Any]:
    paths = (
        Path("scripts/evaluate_population.py"),
        Path("chreatures/training_cohort.py"),
        Path("chreatures/training_environment.py"),
        Path("chreatures/sensorimotor_worker_native.py"),
        Path("chreatures/neural_genotype.py"),
        Path("chreatures/population.py"),
    )
    revision = None
    dirty: list[str] = []
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *(str(path) for path in paths)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "revision": revision,
        "scoped_dirty": bool(dirty),
        "scoped_dirty_paths": sorted(row[3:] for row in dirty),
        "files": {
            str(path): {"bytes": (ROOT / path).stat().st_size, "sha256": file_sha256(ROOT / path)}
            for path in paths
        },
    }


def load_assignments(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("worlds"), list):
        raise ValueError("assignment document must contain a worlds list")
    if value.get("format") not in (None, ASSIGNMENT_FORMAT):
        raise ValueError("assignment format differs")
    if "version" in value and value["version"] != 1:
        raise ValueError("assignment version differs")
    if "sha256" in value:
        body = dict(value)
        expected = _sha(body.pop("sha256"), "assignment identity")
        if canonical_sha256(body) != expected:
            raise ValueError("assignment identity differs")
    worlds = value["worlds"]
    if not worlds:
        raise ValueError("assignment batch has no worlds")
    resident_count = None
    normalized = []
    seen_lives: set[str] = set()
    for world_index, item in enumerate(worlds):
        if not isinstance(item, dict) or set(item) - {"environment", "seed", "candidates", "world_id"}:
            raise ValueError(f"assignment world {world_index} fields differ")
        environment = item.get("environment")
        if (
            not isinstance(environment, dict)
            or set(environment) != {"split", "index"}
            or environment["split"] not in {"training", "heldout"}
            or isinstance(environment["index"], bool)
            or not isinstance(environment["index"], int)
            or environment["index"] < 0
        ):
            raise ValueError(f"assignment world {world_index} environment differs")
        seed = item.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError(f"assignment world {world_index} seed must be uint64")
        world_id = item.get("world_id", f"world-{world_index:04d}")
        if not isinstance(world_id, str) or not world_id or len(world_id) > 128:
            raise ValueError(f"assignment world {world_index} ID differs")
        raw_candidates = item.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"assignment world {world_index} has no candidates")
        candidates = []
        for resident_index, raw in enumerate(raw_candidates):
            candidate = CandidateGenome(raw)
            if content_sha256(candidate.to_value()) != candidate.sha256:
                raise ValueError("candidate genome identity differs")
            life_id = (
                f"world-{world_index:04d}/resident-{resident_index:03d}/"
                f"{candidate.sha256[:16]}"
            )
            if life_id in seen_lives:
                raise ValueError("assignment life identity is duplicated")
            seen_lives.add(life_id)
            candidates.append(candidate)
        if resident_count is None:
            resident_count = len(candidates)
        elif len(candidates) != resident_count:
            raise ValueError("every assignment world must have the same resident count")
        normalized.append({
            "world_id": world_id,
            "environment": environment,
            "seed": seed,
            "candidates": candidates,
        })
    return value, normalized, int(resident_count)


def resident_artifact_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    identity = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "format": metadata.get("format"),
        "version": metadata.get("version"),
        "execution": metadata.get("execution"),
        "artifact_sha256": metadata.get("artifact_sha256"),
    }
    if identity["version"] != 4 or identity["execution"] != "developmental-resident-native-population-v4":
        raise ValueError("base resident artifact is not the current population v4 runtime")
    _sha(identity["artifact_sha256"], "resident artifact identity")
    return identity, metadata


def runtime_identity(device_name: str) -> dict[str, Any]:
    import torch

    from chreatures.native_world import load_world_kernels

    extension = Path(load_world_kernels().__file__).resolve()
    executable = Path(sys.executable).resolve()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA/ROCm device is unavailable")
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "executable": str(executable),
            "executable_sha256": file_sha256(executable),
            "version": sys.version,
        },
        "numpy": np.__version__,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": {
            "request": device_name,
            "type": device.type,
            "name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
        },
        "native_world": {
            "path": str(extension),
            "bytes": extension.stat().st_size,
            "sha256": file_sha256(extension),
        },
    }


def validate_candidate_sources(
    worlds: Sequence[Mapping[str, Any]], graph: Any, ports: Any,
    port_bundle_sha256: str, resident_metadata: Mapping[str, Any],
    resident_file_sha256: str, developmental_base_sha256: str,
) -> str:
    candidates = [candidate for world in worlds for candidate in world["candidates"]]
    controller_hashes = {candidate.to_value().get("base_controller_sha256") for candidate in candidates}
    if len(controller_hashes) != 1:
        raise ValueError("batch candidates do not share one base controller")
    controller_hash = _sha(controller_hashes.pop(), "candidate base controller")
    if controller_hash != resident_file_sha256:
        raise ValueError("candidate base controller differs from resident artifact bytes")
    developmental_hashes = {
        candidate.to_value().get("developmental_base_sha256") for candidate in candidates
    }
    if len(developmental_hashes) != 1:
        raise ValueError("batch candidates do not share one developmental base")
    developmental_hash = _sha(developmental_hashes.pop(), "candidate developmental base")
    if developmental_hash != developmental_base_sha256:
        raise ValueError("candidate developmental base differs from biological profile")
    population_bank = resident_metadata.get("population_adapters", {})
    for candidate in candidates:
        value = candidate.to_value()
        if value.get("graph_sha256") != str(graph.hash):
            raise ValueError("candidate graph differs from evaluation graph")
        if value.get("port_spec_sha256") != ports.spec_hash:
            raise ValueError("candidate port spec differs from evaluation ports")
        if (
            value.get("population_adapter_bank_sha256") != population_bank.get("identity")
            or value.get("policy_adapter_count") != population_bank.get("count")
            or value.get("policy_adapter_rank") != population_bank.get("rank")
        ):
            raise ValueError("candidate controller bank differs from resident artifact")
    trained = resident_metadata.get("training_identity", {})
    if trained.get("graph_sha256") not in (None, str(graph.hash)):
        raise ValueError("resident artifact graph differs from evaluation graph")
    if trained.get("port_spec_sha256") not in (None, ports.spec_hash):
        raise ValueError("resident artifact port spec differs")
    if trained.get("port_bundle_sha256") not in (None, port_bundle_sha256):
        raise ValueError("resident artifact port bundle differs")
    return controller_hash


def compile_phenotypes(
    worlds: Sequence[Mapping[str, Any]], recipe: NeuralVariantRecipe,
    graph: Any, ports: Any, port_bundle_sha256: str, controller_hash: str,
) -> list[Any]:
    candidates = [
        candidate for world in worlds for candidate in world["candidates"]
    ]
    result = compile_population_phenotypes(
        candidates,
        recipe,
        graph,
        ports,
        port_bundle_sha256,
        controller_hash,
    )
    if len({item.compatibility_group for item in result}) != 1:
        raise ValueError("assignment batch contains multiple neural compatibility groups")
    return result


def bind_environment_records(
    worlds: Sequence[dict[str, Any]], profile: Any,
) -> None:
    variants = profile.component("family")["variants"]
    for world in worlds:
        selected = next((
            item for item in variants
            if item["split"] == world["environment"]["split"]
            and item["index"] == world["environment"]["index"]
        ), None)
        if selected is None:
            raise ValueError("assignment environment is outside the pinned profile")
        record = selected.get("environment_record")
        environment_sha256 = _sha(
            selected.get("environment_sha256"), "profile environment identity"
        )
        if not isinstance(record, dict) or record.get("sha256") != environment_sha256:
            raise ValueError("profile variant environment record differs")
        world["environment_sha256"] = environment_sha256
        world["environment_record"] = record


def life_records(
    worlds: Sequence[Mapping[str, Any]], assignment_file_sha256: str,
) -> list[dict[str, Any]]:
    rows = []
    for world_index, world in enumerate(worlds):
        for resident_index, candidate in enumerate(world["candidates"]):
            human_label = (
                f"world-{world_index:04d}/resident-{resident_index:03d}/"
                f"{candidate.sha256[:16]}"
            )
            life_identity = {
                "assignment_file_sha256": assignment_file_sha256,
                "world_id": world["world_id"],
                "world_slot": world_index,
                "resident_slot": resident_index,
                "candidate_sha256": candidate.sha256,
                "environment_seed": world["seed"],
                "environment_split": world["environment"]["split"],
                "environment_index": world["environment"]["index"],
                "environment_sha256": world["environment_sha256"],
            }
            rows.append({
                "life_id": canonical_sha256(life_identity),
                "human_label": human_label,
                "assignment_file_sha256": assignment_file_sha256,
                "world_slot": world_index,
                "resident_slot": resident_index,
                "world_id": world["world_id"],
                "environment": world["environment"],
                "environment_seed": world["seed"],
                "evaluation_seed": world["seed"],
                "environment_sha256": world["environment_sha256"],
                "candidate_sha256": candidate.sha256,
            })
    return rows


def action_payload(actions: np.ndarray, bodies: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    payloads = []
    offset = 0
    for world_bodies in bodies:
        mapped = {}
        for body in world_bodies:
            mapped[str(body["id"])] = dict(zip(
                ACTION_NAMES, actions[offset].astype(float), strict=True
            ))
            offset += 1
        payloads.append({"actions": mapped, "dt": DT})
    if offset != len(actions):
        raise ValueError("physical body rows differ from delivered action rows")
    return payloads


def outcome_array(
    advanced: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> np.ndarray:
    rows = []
    for outcomes, bodies in advanced:
        for body in bodies:
            value = outcomes[str(body["id"])]
            rows.append([float(value.get(name, 0.0)) for name in OUTCOME_FIELDS])
    result = np.ascontiguousarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("population outcomes are nonfinite")
    return result


def body_positions(bodies: Sequence[Sequence[Mapping[str, Any]]]) -> np.ndarray:
    result = np.asarray(
        [[float(body[name]) for name in ("x", "y", "z")] for world in bodies for body in world],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise RuntimeError("population body positions are nonfinite")
    return result


def observe(pool: WorldTrainingPool, brain: TrainingCohortBrain) -> tuple[Any, ...]:
    rich, canonical, bodies = pool.observe_arrays()
    neural, circuit, _ = brain.step_channels(canonical, DT)
    physiology = pool.physiology_array(circuit[:, 2])
    observation = np.ascontiguousarray(
        np.concatenate((rich, canonical, physiology), axis=1), dtype=np.float32
    )
    if observation.shape != (len(physiology), OBSERVATION_DIM):
        raise RuntimeError("population observation contract differs")
    return observation, neural, circuit, physiology, bodies


def world_sizes(snapshots: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    result = []
    for snapshot in snapshots:
        size = snapshot.get("world", {}).get("spec", {}).get("size")
        value = np.asarray(size, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("world snapshot omits its positive physical size")
        result.append(value.astype(float).tolist())
    return result


def trajectory_objects(world_sizes_value: Sequence[Sequence[float]], residents: int, width: float):
    from chreatures.native_world import load_world_kernels

    trajectory_type = load_world_kernels().PopulationTrajectory
    result = [
        trajectory_type(residents, list(size), float(width), DT)
        for size in world_sizes_value
    ]
    for trajectory in result:
        summary = jsonable(trajectory.summary())
        if (
            summary.get("executed_action_order") != list(ACTION_NAMES)
            or summary.get("physiology_order") != list(PHYSIOLOGY_NAMES)
            or summary.get("outcome_order") != list(OUTCOME_FIELDS)
            or summary.get("organ_flow_order") != list(ORGAN_FLOW_ORDER)
            or set(summary.get("resident_axis_keys", ()))
            != TRAJECTORY_RESIDENT_FIELDS
        ):
            raise RuntimeError("native population trajectory interface differs")
    return result


def advance_trajectories(
    trajectories: Sequence[Any], positions: np.ndarray, physiology: np.ndarray,
    actions: np.ndarray, outcomes: np.ndarray, organ_flows: np.ndarray,
    residents: int,
) -> None:
    total = len(trajectories) * residents
    expected = (
        (positions, (total, 3)),
        (physiology, (total, PHYSIOLOGY_DIM)),
        (actions, (total, ACTION_DIM)),
        (outcomes, (total, len(OUTCOME_FIELDS))),
        (organ_flows, (total, len(ORGAN_FLOW_ORDER))),
    )
    if any(value.shape != shape or not np.isfinite(value).all() for value, shape in expected):
        raise ValueError("population trajectory input batch differs")
    if np.any(outcomes[:, :7] < 0) or np.any(organ_flows < 0):
        raise ValueError("physical trajectory measures must be nonnegative")
    valid = np.ones(residents, dtype=np.bool_)
    for world_index, trajectory in enumerate(trajectories):
        rows = slice(world_index * residents, (world_index + 1) * residents)
        trajectory.advance(
            np.ascontiguousarray(positions[rows], dtype=np.float64),
            np.ascontiguousarray(physiology[rows], dtype=np.float64),
            np.ascontiguousarray(actions[rows], dtype=np.float64),
            np.ascontiguousarray(outcomes[rows], dtype=np.float64),
            np.ascontiguousarray(organ_flows[rows], dtype=np.float64),
            valid,
        )


def boundary_arrays(
    observation: np.ndarray, neural: np.ndarray, physiology: np.ndarray,
    previous: np.ndarray, reset: np.ndarray, completed_steps: int,
    controller_outcomes: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {
        "observation": np.ascontiguousarray(observation, dtype=np.float32),
        "neural": np.ascontiguousarray(neural, dtype=np.float32),
        "physiology": np.ascontiguousarray(physiology, dtype=np.float32),
        "actual_previous_actions": np.ascontiguousarray(previous, dtype=np.float32),
        "reset": np.ascontiguousarray(reset, dtype=np.bool_),
        "completed_steps": np.asarray(completed_steps, dtype=np.uint64),
        "time_seconds": np.asarray(completed_steps * DT, dtype=np.float64),
    }
    count = len(observation)
    for name in CONTROLLER_OUTCOME_FIELDS:
        value = np.asarray(
            controller_outcomes.get(name, np.zeros(count, dtype=np.float64))
        )
        if value.shape != (count,) or (
            np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
        ):
            raise ValueError(f"controller boundary differs: {name}")
        result[f"controller_outcome.{name}"] = value.copy()
    return result


def write_checkpoint(
    output: Path, identity_sha256: str, completed_steps: int,
    pool: WorldTrainingPool, brain: TrainingCohortBrain,
    residents: DevelopmentalResidentCohort, trajectories: Sequence[Any],
    boundary: Mapping[str, np.ndarray], lives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final = output / "checkpoints" / f"step-{completed_steps:010d}"
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    temporary = final.with_name(f".{final.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        files: dict[str, Any] = {}
        files["worlds"] = atomic_json(temporary / "worlds.json", pool.snapshot())
        files["controller"] = atomic_json(
            temporary / "controller.json", residents.snapshot_value()
        )
        files["boundary"] = atomic_npz(temporary / "boundary.npz", boundary)
        neural_receipt = brain.snapshot(temporary, "neural")
        files["neural"] = {
            "path": "neural.npz",
            "bytes": neural_receipt["bytes"],
            "sha256": neural_receipt["sha256"],
        }
        trajectory_files = []
        for world_index, trajectory in enumerate(trajectories):
            trajectory_files.append(atomic_bytes(
                temporary / f"trajectory-world-{world_index:04d}.bin",
                bytes(trajectory.snapshot()),
            ))
        files["trajectories"] = trajectory_files
        receipt = {
            "format": CHECKPOINT_FORMAT,
            "version": 1,
            "evaluation_identity_sha256": identity_sha256,
            "completed_steps": completed_steps,
            "completed_resident_transitions": completed_steps * len(lives),
            "life_ids": [row["life_id"] for row in lives],
            "files": files,
            "created_utc": utc_now(),
        }
        atomic_json(temporary / "checkpoint.json", receipt)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)
        checkpoint_hash = file_sha256(final / "checkpoint.json")
        latest = {
            "format": "chreatures-population-checkpoint-pointer-v1",
            "evaluation_identity_sha256": identity_sha256,
            "completed_steps": completed_steps,
            "checkpoint": str(final.relative_to(output)),
            "checkpoint_receipt_sha256": checkpoint_hash,
        }
        atomic_json(output / "latest.json", latest)
        return {**latest, "checkpoint_receipt_bytes": (final / "checkpoint.json").stat().st_size}
    except BaseException:
        # A temp directory is deliberately retained for postmortem evidence.
        raise


def verify_checkpoint(output: Path, identity_sha256: str) -> tuple[Path, dict[str, Any]]:
    latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    if latest.get("evaluation_identity_sha256") != identity_sha256:
        raise ValueError("checkpoint pointer belongs to another evaluation")
    root = (output / latest["checkpoint"]).resolve()
    root.relative_to(output.resolve())
    receipt_path = root / "checkpoint.json"
    if file_sha256(receipt_path) != latest["checkpoint_receipt_sha256"]:
        raise ValueError("checkpoint receipt hash differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format") != CHECKPOINT_FORMAT
        or receipt.get("version") != 1
        or receipt.get("evaluation_identity_sha256") != identity_sha256
        or receipt.get("completed_steps") != latest.get("completed_steps")
    ):
        raise ValueError("coupled checkpoint identity differs")
    records = receipt["files"]
    flat = [records["worlds"], records["controller"], records["boundary"], records["neural"]]
    flat.extend(records["trajectories"])
    for item in flat:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"coupled checkpoint file differs: {item['path']}")
    return root, receipt


def load_boundary(path: Path, count: int) -> dict[str, np.ndarray]:
    expected = {
        "observation": ((count, OBSERVATION_DIM), np.float32),
        "neural": ((count, NEURAL_DIM), np.float32),
        "physiology": ((count, PHYSIOLOGY_DIM), np.float32),
        "actual_previous_actions": ((count, PREVIOUS_DIM), np.float32),
        "reset": ((count,), np.bool_),
        "completed_steps": ((), np.uint64),
        "time_seconds": ((), np.float64),
    }
    controller_names = {f"controller_outcome.{name}" for name in CONTROLLER_OUTCOME_FIELDS}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected) | controller_names:
            raise ValueError("checkpoint boundary fields differ")
        value = {name: np.asarray(archive[name]).copy() for name in archive.files}
    for name, (shape, dtype) in expected.items():
        if value[name].shape != shape or value[name].dtype != dtype:
            raise ValueError(f"checkpoint boundary differs: {name}")
    for name in controller_names:
        if value[name].shape != (count,):
            raise ValueError(f"checkpoint controller boundary differs: {name}")
    if any(
        np.issubdtype(item.dtype, np.floating) and not np.isfinite(item).all()
        for item in value.values()
    ):
        raise ValueError("checkpoint boundary contains nonfinite values")
    return value


def resident_summary_row(summary: Any, row: int, residents: int) -> Any:
    if not isinstance(summary, Mapping):
        raise TypeError("native trajectory summary must be an object")
    resident_fields = set(summary.get("resident_axis_keys", ()))
    if resident_fields != TRAJECTORY_RESIDENT_FIELDS:
        raise ValueError("native trajectory resident-axis contract differs")
    result = {}
    for key, value in summary.items():
        if key in resident_fields:
            value = np.asarray(value)
            if value.ndim < 1 or value.shape[0] != residents:
                raise ValueError(f"trajectory resident axis differs: {key}")
            result[str(key)] = jsonable(value[row])
        else:
            result[str(key)] = jsonable(value)
    return result


def telemetry_receipt(
    output: Path, completed_steps: int, physiology: np.ndarray,
    actions: np.ndarray, outcomes: np.ndarray, trajectories: Sequence[Any],
    timing: Mapping[str, Any], controller_outcomes: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    value = {
        "format": "chreatures-population-bounded-telemetry-v1",
        "completed_steps": completed_steps,
        "physiology": {
            "mean": physiology.mean(axis=0).astype(float).tolist(),
            "minimum": physiology.min(axis=0).astype(float).tolist(),
            "maximum": physiology.max(axis=0).astype(float).tolist(),
        },
        "executed_action_abs_mean": np.abs(actions).mean(axis=0).astype(float).tolist(),
        "transition_outcome_sum": outcomes.sum(axis=0).astype(float).tolist(),
        "trajectory": [jsonable(trajectory.summary()) for trajectory in trajectories],
        "controller_outcomes": {
            name: jsonable(value) for name, value in controller_outcomes.items()
        },
        "transport_timing": timing,
    }
    return atomic_json(output / "telemetry" / f"step-{completed_steps:010d}.json", value)


def write_gam_trace_chunk(output: Path, identity_sha256: str, start_tick: int,
                          pre_physiology: list[np.ndarray], actions: list[np.ndarray],
                          post_physiology: list[np.ndarray], outcomes: list[np.ndarray],
                          organ_flows: list[np.ndarray]) -> dict[str, Any]:
    if not pre_physiology:
        raise ValueError("empty GAM trace chunk")
    steps = len(pre_physiology)
    arrays = {
        "format": np.asarray("chreatures-population-gam-trace-v1"),
        "evaluation_identity_sha256": np.asarray(identity_sha256),
        "start_tick": np.asarray(start_tick, dtype=np.uint64),
        "tick": np.arange(start_tick, start_tick + steps, dtype=np.uint64),
        "pre_physiology12": np.stack(pre_physiology).astype(np.float32, copy=False),
        "executed_action12": np.stack(actions).astype(np.float32, copy=False),
        "post_physiology12": np.stack(post_physiology).astype(np.float32, copy=False),
        "outcomes8": np.stack(outcomes).astype(np.float32, copy=False),
        "organ_flows3": np.stack(organ_flows).astype(np.float32, copy=False),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()
           if np.issubdtype(value.dtype, np.floating)):
        raise ValueError("GAM trace chunk contains nonfinite values")
    return atomic_npz(output / "gam_trace" / f"tick-{start_tick:010d}.npz", arrays)


def main() -> int:
    args = arguments()
    validate_arguments(args)
    args.output.mkdir(parents=True, exist_ok=True)

    from chreatures.neural_ports import NeuralPortBundle
    from chreatures.training_environment import EmbodiedTrainingProfile

    profile_value = json.loads(args.profile.read_text(encoding="utf-8"))
    profile = EmbodiedTrainingProfile.from_value(profile_value)
    assignment_value, worlds, residents_per_world = load_assignments(args.assignments)
    assignment_file_sha256 = file_sha256(args.assignments)
    bind_environment_records(worlds, profile)
    declared = profile.component("family")["transport"]["residents"]
    if residents_per_world != declared:
        raise ValueError("assignment resident count differs from pinned profile")
    count = len(worlds) * residents_per_world
    graph = load_training_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    port_bundle_sha256 = file_sha256(args.port_bundle)
    artifact_identity, resident_metadata = resident_artifact_identity(args.resident_artifact)
    developmental_base_sha256 = _sha(
        profile.component("sources")["biosphere_birth"]["sha256"],
        "profile biological developmental base",
    )
    controller_hash = validate_candidate_sources(
        worlds, graph, ports, port_bundle_sha256, resident_metadata,
        artifact_identity["file_sha256"], developmental_base_sha256,
    )
    recipe = NeuralVariantRecipe.load(args.neural_recipe)
    phenotypes = compile_phenotypes(
        worlds, recipe, graph, ports, port_bundle_sha256, controller_hash
    )
    lives = life_records(worlds, assignment_file_sha256)
    adapters = [
        candidate.controller_adapter()
        for world in worlds for candidate in world["candidates"]
    ]
    identity = {
        "format": f"{FORMAT}-identity",
        "version": 1,
        "source": source_identity(),
        "runtime": runtime_identity(args.device),
        "profile_file": {"path": str(args.profile.resolve()), "sha256": file_sha256(args.profile)},
        "profile_sha256": profile.sha256,
        "resident_artifact": artifact_identity,
        "developmental_base_sha256": developmental_base_sha256,
        "graph": {"path": str(args.graph.resolve()), "sha256": str(graph.hash)},
        "ports": {
            "path": str(args.port_bundle.resolve()),
            "file_sha256": port_bundle_sha256,
            "spec_sha256": ports.spec_hash,
        },
        "neural_recipe": {
            "path": str(args.neural_recipe.resolve()),
            "file_sha256": file_sha256(args.neural_recipe),
            "sha256": recipe.sha256,
        },
        "assignments": {
            "path": str(args.assignments.resolve()),
            "file_sha256": assignment_file_sha256,
            "content_sha256": canonical_sha256(assignment_value),
        },
        "phenotype_order": [item.sha256 for item in phenotypes],
        "phenotypes": {
            candidate.sha256: {
                "phenotype_sha256": phenotype.sha256,
                "receipt": phenotype.receipt,
            }
            for candidate, phenotype in zip(
                [candidate for world in worlds for candidate in world["candidates"]],
                phenotypes,
                strict=True,
            )
        },
        "neural_compatibility_group": phenotypes[0].compatibility_group,
        "life_records": lives,
        "execution": {
            "steps": args.steps,
            "checkpoint_every": args.checkpoint_every,
            "telemetry_every": args.telemetry_every,
            "gam_trace_chunk_steps": args.gam_trace_chunk_steps,
            "spatial_bin_width": args.spatial_bin_width,
            "dt_seconds": DT,
            "device": args.device,
            "brain_backend": args.brain_backend,
            "physical_backend": args.physical_backend,
            "action_mode": args.action_mode,
        },
    }
    identity["sha256"] = canonical_sha256(identity)
    identity_path = args.output / "identity.json"
    if identity_path.exists():
        stored_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored_identity != identity:
            raise ValueError("evaluation resume identity differs")
    else:
        if args.resume:
            raise ValueError("resume requested without an evaluation identity")
        atomic_json(identity_path, identity)
    completed_result = args.output / "result.json"
    if args.resume and completed_result.is_file():
        value = json.loads(completed_result.read_text(encoding="utf-8"))
        claimed = _sha(value.pop("content_sha256", None), "completed result identity")
        if (
            value.get("status") != "completed"
            or value.get("evaluation_identity_sha256") != identity["sha256"]
            or canonical_sha256(value) != claimed
        ):
            raise ValueError("completed evaluation result identity differs")
        print(json.dumps({
            "status": "completed",
            "result": str(completed_result),
            "content_sha256": claimed,
        }, sort_keys=True))
        return 0

    brain: TrainingCohortBrain | None = None
    pool: WorldTrainingPool | None = None
    controller: DevelopmentalResidentCohort | None = None
    trajectories: list[Any] = []
    latest_checkpoint: dict[str, Any] | None = None
    completed_steps = 0
    started = time.perf_counter()
    last_actions = np.zeros((count, ACTION_DIM), dtype=np.float32)
    last_outcomes = np.zeros((count, len(OUTCOME_FIELDS)), dtype=np.float32)
    last_controller_outcomes: dict[str, np.ndarray] = {}
    gam_pre: list[np.ndarray] = []
    gam_actions: list[np.ndarray] = []
    gam_post: list[np.ndarray] = []
    gam_outcomes: list[np.ndarray] = []
    gam_flows: list[np.ndarray] = []
    gam_start_tick = 0
    try:
        brain = TrainingCohortBrain(
            graph, ports, count, device=args.device, backend=args.brain_backend
        )
        pool = WorldTrainingPool(
            len(worlds), dict(ports.spec), profile.to_value(), args.physical_backend,
            residents_per_world=residents_per_world,
        )
        brain.reset_residents([row["life_id"] for row in lives])
        brain.bind_phenotypes(phenotypes)
        if args.resume:
            checkpoint_root, checkpoint_receipt = verify_checkpoint(
                args.output, identity["sha256"]
            )
            world_snapshots = json.loads(
                (checkpoint_root / "worlds.json").read_text(encoding="utf-8")
            )
            bodies = pool.restore(world_snapshots)
            brain.restore(
                checkpoint_root, "neural",
                expected_sha256=checkpoint_receipt["files"]["neural"]["sha256"],
            )
            if brain.resident_ids != [row["life_id"] for row in lives]:
                raise RuntimeError("restored neural resident order differs")
            controller = DevelopmentalResidentCohort.restore_value(
                json.loads((checkpoint_root / "controller.json").read_text(encoding="utf-8")),
                args.resident_artifact,
            )
            if controller.candidate_adapters != adapters:
                raise RuntimeError("restored controller candidate order differs")
            boundary = load_boundary(checkpoint_root / "boundary.npz", count)
            completed_steps = int(boundary["completed_steps"])
            observation = boundary["observation"]
            neural = boundary["neural"]
            physiology = boundary["physiology"]
            previous = boundary["actual_previous_actions"]
            reset = boundary["reset"]
            last_controller_outcomes = {
                name: boundary[f"controller_outcome.{name}"]
                for name in CONTROLLER_OUTCOME_FIELDS
            }
            trajectories = trajectory_objects(
                world_sizes(world_snapshots), residents_per_world, args.spatial_bin_width
            )
            for trajectory, receipt in zip(
                trajectories, checkpoint_receipt["files"]["trajectories"], strict=True
            ):
                trajectory.restore((checkpoint_root / receipt["path"]).read_bytes())
            latest_checkpoint = json.loads((args.output / "latest.json").read_text())
        else:
            bodies = pool.reset([
                {
                    "seed": world["seed"],
                    "held_out": world["environment"]["split"] == "heldout",
                    "environment": world["environment"],
                    "candidates": [candidate.to_value() for candidate in world["candidates"]],
                }
                for world in worlds
            ])
            controller = DevelopmentalResidentCohort(
                args.resident_artifact,
                count,
                action_mode=args.action_mode,
                goal_seed=int(canonical_sha256({"identity": identity["sha256"], "stream": "goal"})[:16], 16),
                action_seed=int(canonical_sha256({"identity": identity["sha256"], "stream": "action"})[:16], 16),
                candidate_adapters=adapters,
            )
            observation, neural, _circuit, physiology, bodies = observe(pool, brain)
            previous = np.zeros((count, PREVIOUS_DIM), dtype=np.float32)
            reset = np.ones(count, dtype=np.bool_)
            completed_steps = 0
            initial_worlds = pool.snapshot()
            trajectories = trajectory_objects(
                world_sizes(initial_worlds), residents_per_world, args.spatial_bin_width
            )
            latest_checkpoint = write_checkpoint(
                args.output, identity["sha256"], 0, pool, brain, controller,
                trajectories,
                boundary_arrays(
                    observation, neural, physiology, previous, reset, 0,
                    last_controller_outcomes,
                ),
                lives,
            )
        if controller is None:
            raise AssertionError("population controller was not initialized")
        expected_contract = {
            "graph_sha256": str(graph.hash),
            "port_spec_sha256": ports.spec_hash,
            "port_bundle_sha256": port_bundle_sha256,
        }
        if controller.neural_contract != expected_contract:
            raise RuntimeError("resident artifact neural contract differs")

        while completed_steps < args.steps:
            tick_values = np.full(count, completed_steps, dtype=np.uint64)
            time_values = np.full(count, completed_steps * DT, dtype=np.float64)
            result = controller.step(
                observation, neural, physiology, previous,
                tick_values, time_values, reset,
            )
            delivered = np.ascontiguousarray(result["proposed_action"], dtype=np.float32)
            if delivered.shape != (count, ACTION_DIM):
                raise RuntimeError("controller action batch differs from organism interface")
            before = physiology.copy()
            advanced = pool.advance(action_payload(delivered, bodies))
            outcomes = outcome_array(advanced)
            bodies = [world_bodies for _outcomes, world_bodies in advanced]
            observation, neural, _circuit, physiology, bodies = observe(pool, brain)
            effort_column = OUTCOME_FIELDS.index("effort")
            last_controller_outcomes = controller.observe_consequences(
                tick_values, before, physiology, delivered,
                outcomes[:, effort_column], dt=DT,
            )
            if not hasattr(pool, "organ_flows_array"):
                raise RuntimeError("current world transport does not expose actual organ flows")
            organ_flows = np.ascontiguousarray(pool.organ_flows_array(), dtype=np.float32)
            if organ_flows.shape != (count, len(ORGAN_FLOW_ORDER)):
                raise RuntimeError("actual organ flow cohort has the wrong shape")
            if args.gam_trace_chunk_steps:
                if not gam_pre:
                    gam_start_tick = completed_steps
                gam_pre.append(np.ascontiguousarray(before, dtype=np.float32))
                gam_actions.append(delivered.copy())
                gam_post.append(np.ascontiguousarray(physiology, dtype=np.float32))
                gam_outcomes.append(outcomes.copy())
                gam_flows.append(organ_flows.copy())
            advance_trajectories(
                trajectories, body_positions(bodies), physiology, delivered,
                outcomes, organ_flows, residents_per_world,
            )
            completed_steps += 1
            if args.gam_trace_chunk_steps and (
                len(gam_pre) == args.gam_trace_chunk_steps
                or completed_steps % args.checkpoint_every == 0
                or completed_steps == args.steps
            ):
                write_gam_trace_chunk(
                    args.output, identity["sha256"], gam_start_tick, gam_pre, gam_actions,
                    gam_post, gam_outcomes, gam_flows,
                )
                gam_pre.clear(); gam_actions.clear(); gam_post.clear()
                gam_outcomes.clear(); gam_flows.clear()
            previous = delivered
            reset = np.zeros(count, dtype=np.bool_)
            last_actions = delivered
            last_outcomes = outcomes
            if completed_steps % args.telemetry_every == 0 or completed_steps == args.steps:
                telemetry_receipt(
                    args.output, completed_steps, physiology, delivered, outcomes,
                    trajectories, pool.timing_snapshot(), last_controller_outcomes,
                )
            if completed_steps % args.checkpoint_every == 0 or completed_steps == args.steps:
                latest_checkpoint = write_checkpoint(
                    args.output, identity["sha256"], completed_steps,
                    pool, brain, controller, trajectories,
                    boundary_arrays(
                        observation, neural, physiology, previous, reset,
                        completed_steps, last_controller_outcomes,
                    ),
                    lives,
                )

        if not hasattr(pool, "terminal_outcomes"):
            raise RuntimeError("current world transport does not expose terminal outcomes")
        terminal = pool.terminal_outcomes()
        trajectory_summaries = [jsonable(value.summary()) for value in trajectories]
        rows = []
        for life in lives:
            world_index = int(life["world_slot"])
            resident_index = int(life["resident_slot"])
            body = bodies[world_index][resident_index]
            body_id = str(body["id"])
            resident_metrics = resident_summary_row(
                trajectory_summaries[world_index], resident_index, residents_per_world
            )
            cohort_trajectory_snapshot_sha256 = hashlib.sha256(
                bytes(trajectories[world_index].snapshot())
            ).hexdigest()
            trajectory_hash = canonical_sha256({
                "life": life,
                "cohort_snapshot_sha256": cohort_trajectory_snapshot_sha256,
                "resident_metrics": resident_metrics,
            })
            rows.append({
                **life,
                "status": "completed",
                "physical_body_id": body_id,
                "committed_ticks": completed_steps,
                "trajectory_sha256": trajectory_hash,
                "cohort_trajectory_snapshot_sha256": cohort_trajectory_snapshot_sha256,
                "trajectory_metrics": resident_metrics,
                "terminal_outcome": terminal[world_index]["residents"][body_id],
                "controller_outcome": {
                    name: jsonable(value[world_index * residents_per_world + resident_index])
                    for name, value in last_controller_outcomes.items()
                },
            })
        result_value = {
            "format": FORMAT,
            "version": 1,
            "status": "completed",
            "evaluation_identity_sha256": identity["sha256"],
            "completed_steps": completed_steps,
            "completed_resident_transitions": completed_steps * count,
            "worlds": len(worlds),
            "residents_per_world": residents_per_world,
            "lives": rows,
            "environments": {
                world["environment_sha256"]: world["environment_record"]
                for world in worlds
            },
            "world_terminal_outcomes": terminal,
            "trajectory_cohorts": [
                {
                    "world_slot": index,
                    "snapshot_sha256": hashlib.sha256(bytes(value.snapshot())).hexdigest(),
                    "summary": trajectory_summaries[index],
                }
                for index, value in enumerate(trajectories)
            ],
            "final_checkpoint": latest_checkpoint,
            "brain": brain.metadata(),
            "controller": controller.model_identity,
            "transport_timing": pool.timing_snapshot(),
            "elapsed_seconds": time.perf_counter() - started,
            "last_step_audit": {
                "executed_action_mean": last_actions.mean(axis=0).astype(float).tolist(),
                "outcome_sum": last_outcomes.sum(axis=0).astype(float).tolist(),
            },
            "completed_utc": utc_now(),
        }
        result_value["content_sha256"] = canonical_sha256(result_value)
        atomic_json(args.output / "result.json", result_value)
        return 0
    except BaseException as error:
        trace = traceback.format_exc()[-32_768:]
        trace_sha256 = hashlib.sha256(trace.encode()).hexdigest()
        checkpoint_hash = (
            latest_checkpoint.get("checkpoint_receipt_sha256")
            if latest_checkpoint is not None else None
        )
        candidate_failures = []
        for row in lives:
            world_index = int(row["world_slot"])
            trajectory_snapshot_sha256 = (
                hashlib.sha256(bytes(trajectories[world_index].snapshot())).hexdigest()
                if world_index < len(trajectories) else None
            )
            partial_trajectory_sha256 = canonical_sha256({
                "life": row,
                "completed_steps": completed_steps,
                "checkpoint_receipt_sha256": checkpoint_hash,
                "cohort_trajectory_snapshot_sha256": trajectory_snapshot_sha256,
                "failure_trace_sha256": trace_sha256,
            })
            candidate_failures.append({
                **row,
                "status": "failed",
                "committed_ticks": completed_steps,
                "trajectory_sha256": partial_trajectory_sha256,
                "cohort_trajectory_snapshot_sha256": trajectory_snapshot_sha256,
                "checkpoint_receipt_sha256": checkpoint_hash,
                "failure": "cohort-step-failure",
                "failure_trace_sha256": trace_sha256,
            })
        diagnostic = {
            "format": "chreatures-population-evaluation-failure-v1",
            "evaluation_identity_sha256": identity["sha256"],
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": trace,
            "traceback_sha256": trace_sha256,
            "last_completed_checkpoint": latest_checkpoint,
            "candidate_failures": candidate_failures,
            "failed_utc": utc_now(),
        }
        diagnostic["content_sha256"] = canonical_sha256(diagnostic)
        immutable_failure = (
            args.output / "failures" / f"{diagnostic['content_sha256']}.json"
        )
        if immutable_failure.is_file():
            existing = json.loads(immutable_failure.read_text(encoding="utf-8"))
            if existing != diagnostic:
                raise RuntimeError("immutable evaluation failure receipt differs") from error
        else:
            atomic_json(immutable_failure, diagnostic)
        atomic_json(args.output / "failure.json", diagnostic)
        raise
    finally:
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
