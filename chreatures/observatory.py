"""Immutable, read-only evidence archive for 3D life and development runs.

The observatory verifies source receipts, summarizes physical/ecological/cognitive
state without copying autobiographical records, fits native GAMs on whole-world
holdouts, and imports the resulting evidence graph through Universal Weave.
It is an external research archive; none of its records are creature memory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import FileResponse
except ModuleNotFoundError:  # The isolated analysis environment only needs NumPy/gamfit.
    APIRouter = HTTPException = FileResponse = None  # type: ignore[assignment,misc]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "integrations" / "artifacts" / "observatory3d"
WEAVE_MANIFEST = ROOT / "integrations" / "weave" / "Cargo.toml"
GAMFIT_VERSION = "0.1.259"
GAMFIT_SOURCE_COMMIT = "7c7eca8ac4826de95c8e743a20294bee132a9bcc"
WEAVE_SOURCE_COMMIT = "7a5a0dabb94885e44ad8a6c4355c015d7f38020f"
SCHEMA_VERSION = 2

TIMESERIES_FIELDS = (
    "step",
    "phase",
    "world_id",
    "energy",
    "gut",
    "fatigue",
    "speed",
    "support",
    "nutrition",
    "contacts",
    "distance",
    "effort",
    "activity",
    "prediction_error",
    "learning_progress",
    "distance_next",
    "energy_next",
    "prediction_error_next",
)

MODEL_SPECS = {
    "movement_outcome_dynamics": {
        "target": "distance_next",
        "predictors": (
            "speed",
            "support",
            "activity",
            "effort",
            "distance",
            "phase",
        ),
        "formula": (
            "distance_next ~ s(speed, k=10) + s(support, k=8) + "
            "s(activity, k=8) + s(effort, k=8) + s(distance, k=10) + phase"
        ),
        "persistence": "distance",
        "description": "one-step locomotor outcome under the learned controller",
    },
    "energy_dynamics": {
        "target": "energy_next",
        "predictors": (
            "energy",
            "nutrition",
            "effort",
            "contact",
            "prediction_error",
            "phase",
        ),
        "formula": (
            "energy_next ~ s(energy, k=10) + s(nutrition, k=8) + "
            "s(effort, k=8) + s(contact, k=6) + s(prediction_error, k=8) + phase"
        ),
        "persistence": "energy",
        "description": "one-step mean bodily-energy dynamics after physical outcomes",
    },
    "prediction_error_dynamics": {
        "target": "prediction_error_next",
        "predictors": (
            "prediction_error",
            "activity",
            "learning_progress",
            "step_fraction",
            "phase",
        ),
        "formula": (
            "prediction_error_next ~ s(prediction_error, k=10) + s(activity, k=8) + "
            "s(learning_progress, k=8) + s(step_fraction, k=12) + phase"
        ),
        "persistence": "prediction_error",
        "description": "one-step action-conditioned forward-model error dynamics",
    },
}


class ObservatoryError(RuntimeError):
    """An input or generated artifact failed an integrity requirement."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_uri(digest: str) -> str:
    return f"urn:sha256:{digest}"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ObservatoryError(f"{path.name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ObservatoryError(f"{path.name} must contain a JSON object")
    return value, payload


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ObservatoryError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservatoryError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ObservatoryError(f"{field} must be finite")
    return number


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(data)),
        "mean": float(np.mean(data)),
        "max": float(np.max(data)),
    }


def _component_counts(components: object) -> dict[str, int]:
    if not isinstance(components, dict):
        return {}
    counts: Counter[str] = Counter()
    for attached in components.values():
        if isinstance(attached, list):
            for component in attached:
                if isinstance(component, dict) and isinstance(component.get("type"), str):
                    counts[component["type"]] += 1
    return dict(sorted(counts.items()))


def _cognitive_summary(organs: object) -> dict[str, Any]:
    if not isinstance(organs, dict):
        return {"resident_count": 0, "residents": []}
    residents = []
    for resident_id, organ in sorted(organs.items()):
        if not isinstance(organ, dict):
            continue
        memory = organ.get("memory") if isinstance(organ.get("memory"), dict) else {}
        state = organ.get("state") if isinstance(organ.get("state"), dict) else {}
        records = memory.get("records") if isinstance(memory.get("records"), list) else []
        origins = Counter(
            str(record.get("origin", "unknown"))
            for record in records
            if isinstance(record, dict)
        )
        metrics = state.get("last_metrics") if isinstance(state.get("last_metrics"), dict) else {}
        residents.append(
            {
                "resident_id": str(resident_id),
                "learning": bool(state.get("learning", False)),
                "model_time": state.get("time"),
                "memory": {
                    "count": len(records),
                    "capacity": memory.get("capacity"),
                    "origin_counts": dict(sorted(origins.items())),
                },
                "metrics": {
                    key: metrics.get(key)
                    for key in (
                        "prediction_error",
                        "learning_progress",
                        "novelty",
                        "advantage",
                        "homeostatic_outcome",
                        "recalled_value",
                    )
                },
            }
        )
    return {
        "resident_count": len(residents),
        "memory_records": sum(item["memory"]["count"] for item in residents),
        "residents": residents,
        "scope": "summary only; personal records remain in the source checkpoint",
    }


def load_world_checkpoint(path: Path) -> dict[str, Any]:
    envelope, raw = _read_object(path)
    if envelope.get("format") != "chreatures-3d-checkpoint-v1":
        raise ObservatoryError("world input is not a chreatures-3d-checkpoint-v1 envelope")
    state = envelope.get("state")
    if not isinstance(state, dict):
        raise ObservatoryError("3D checkpoint state must be an object")
    state_digest = sha256_bytes(canonical(state))
    if envelope.get("sha256") != state_digest:
        raise ObservatoryError("3D checkpoint state checksum does not match")

    world = state.get("world")
    if not isinstance(world, dict):
        raise ObservatoryError("3D checkpoint world must be an object")
    if world.get("dimension") != 3:
        raise ObservatoryError("whole-world checkpoint is not three-dimensional")
    spec = world.get("spec") if isinstance(world.get("spec"), dict) else {}
    bodies = world.get("bodies") if isinstance(world.get("bodies"), list) else []
    entities = spec.get("entities") if isinstance(spec.get("entities"), list) else []
    mobility = Counter(
        str(entity.get("mobility", "preset"))
        for entity in entities
        if isinstance(entity, dict)
    )
    jointed = [entity for entity in entities if isinstance(entity, dict) and "joint" in entity]
    body_mode = state.get("body_mode")
    body_mode_recorded = isinstance(body_mode, str)
    if not body_mode_recorded:
        body_mode = "legacy crawler (checkpoint predates explicit body_mode)"
    pose3d_bodies = sum(
        isinstance(body, dict)
        and all(key in body for key in ("x", "y", "z", "quaternion"))
        and isinstance(body.get("quaternion"), list)
        and len(body["quaternion"]) == 4
        for body in bodies
    )
    velocity3d_bodies = sum(
        isinstance(body, dict)
        and isinstance(body.get("linear_velocity"), list)
        and len(body["linear_velocity"]) == 3
        and isinstance(body.get("angular_velocity3d"), list)
        and len(body["angular_velocity3d"]) == 3
        for body in bodies
    )
    if not bodies or pose3d_bodies != len(bodies) or velocity3d_bodies != len(bodies):
        raise ObservatoryError("whole-world residents do not contain complete 3D state")
    field = state.get("field")
    ecology_kind = "diffusion" if isinstance(field, dict) else "analytic/legacy"

    histories = state.get("history") if isinstance(state.get("history"), dict) else {}
    history_rows: list[dict[str, Any]] = []
    history_summary = []
    for resident_id, rows in sorted(histories.items()):
        if not isinstance(rows, list):
            continue
        clean = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean_row = {"resident_id": resident_id}
            try:
                for field_name in ("time", "x", "y", "z", "energy", "activity", "memory"):
                    clean_row[field_name] = _finite_number(
                        row.get(field_name), f"world history {resident_id}.{field_name}"
                    )
            except ObservatoryError:
                continue
            clean.append(clean_row)
        history_rows.extend(clean)
        history_summary.append(
            {
                "resident_id": resident_id,
                "rows": len(clean),
                "time_range": [clean[0]["time"], clean[-1]["time"]] if clean else None,
                "energy": _summary([row["energy"] for row in clean]),
                "activity": _summary([row["activity"] for row in clean]),
                "memory_range": (
                    [int(min(row["memory"] for row in clean)), int(max(row["memory"] for row in clean))]
                    if clean
                    else None
                ),
            }
        )

    resource_remaining: dict[str, float] = {}
    components = world.get("components")
    if isinstance(components, dict):
        for entity_id, attached in components.items():
            if not isinstance(attached, list):
                continue
            for component in attached:
                if isinstance(component, dict) and component.get("type") == "food":
                    resource_remaining[str(entity_id)] = _finite_number(
                        component.get("amount", 0), f"resource {entity_id}.amount"
                    )

    journal = state.get("journal") if isinstance(state.get("journal"), list) else []
    safe_journal = [
        event
        for event in journal
        if isinstance(event, dict)
        and isinstance(event.get("id"), str)
        and isinstance(event.get("text"), str)
    ]
    model_times = [
        _finite_number(body.get("age"), "body.age")
        for body in bodies
        if isinstance(body, dict) and body.get("age") is not None
    ]
    return {
        "source": {
            "path": display_path(path),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "artifact_uri": artifact_uri(sha256_bytes(raw)),
            "format": envelope["format"],
            "state_sha256": state_digest,
            "checksum_verified": True,
        },
        "summary": {
            "habitat_id": state.get("id"),
            "tick": state.get("tick"),
            "model_time": max(model_times) if model_times else None,
            "branch": state.get("branch"),
            "anatomy_sha256": state.get("graph_sha256"),
            "physics": {
                "dimension": world.get("dimension"),
                "engine": world.get("engine"),
                "model_signature": world.get("model_signature"),
                "habitat": spec.get("name"),
                "size": spec.get("size"),
                "units": spec.get("units"),
                "entity_count": len(entities),
                "resident_count": len(bodies),
            },
            "articulation": {
                "body_mode": body_mode,
                "body_mode_recorded": body_mode_recorded,
                "resident_pose_3d_count": pose3d_bodies,
                "resident_velocity_3d_count": velocity3d_bodies,
                "jointed_environment_entities": len(jointed),
                "mobility_counts": dict(sorted(mobility.items())),
            },
            "ecology": {
                "kind": ecology_kind,
                "explicit_field_state": isinstance(field, dict),
                "component_counts": _component_counts(components),
                "resource_remaining": resource_remaining,
            },
            "cognition": _cognitive_summary(state.get("organs")),
            "history": history_summary,
            "journal_events": len(safe_journal),
        },
        "history_rows": history_rows,
        "journal": safe_journal,
    }


def _verify_receipt(path: Path, receipt: object, label: str) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ObservatoryError(f"missing receipt for {label}")
    expected_hash = receipt.get("sha256")
    expected_bytes = receipt.get("bytes")
    if not isinstance(expected_hash, str) or not isinstance(expected_bytes, int):
        raise ObservatoryError(f"invalid receipt for {label}")
    if not path.is_file():
        raise ObservatoryError(f"missing development artifact {label}")
    actual_hash = sha256_file(path)
    actual_bytes = path.stat().st_size
    if actual_hash != expected_hash or actual_bytes != expected_bytes:
        raise ObservatoryError(f"development artifact receipt mismatch for {label}")
    return {
        "path": display_path(path),
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "artifact_uri": artifact_uri(actual_hash),
        "verified": True,
    }


def _world_number(resident_id: str) -> int:
    prefix = resident_id.split(":", 1)[0]
    if not prefix.startswith("world-"):
        raise ObservatoryError(f"invalid development resident id {resident_id!r}")
    try:
        return int(prefix.removeprefix("world-"))
    except ValueError as exc:
        raise ObservatoryError(f"invalid development resident id {resident_id!r}") from exc


def load_development_run(directory: Path) -> dict[str, Any]:
    summary, summary_raw = _read_object(directory / "summary.json")
    run, run_raw = _read_object(directory / "run.json")
    egg, egg_raw = _read_object(directory / "egg-manifest.json")
    if run.get("format") != "chreatures-development-run-v1":
        raise ObservatoryError("development run.json has an unsupported format")
    if egg.get("format") != "chreatures-developmental-egg-v1":
        raise ObservatoryError("development egg manifest has an unsupported format")
    if summary.get("egg") != egg:
        raise ObservatoryError("summary egg receipt differs from egg-manifest.json")
    if summary.get("completed") is not True or summary.get("stopped_by_signal") is not False:
        raise ObservatoryError("development run is not a completed uninterrupted result")

    receipts = {
        "summary.json": {
            "path": display_path(directory / "summary.json"),
            "bytes": len(summary_raw),
            "sha256": sha256_bytes(summary_raw),
            "artifact_uri": artifact_uri(sha256_bytes(summary_raw)),
            "verified": True,
        },
        "run.json": {
            "path": display_path(directory / "run.json"),
            "bytes": len(run_raw),
            "sha256": sha256_bytes(run_raw),
            "artifact_uri": artifact_uri(sha256_bytes(run_raw)),
            "verified": True,
        },
        "egg-manifest.json": {
            "path": display_path(directory / "egg-manifest.json"),
            "bytes": len(egg_raw),
            "sha256": sha256_bytes(egg_raw),
            "artifact_uri": artifact_uri(sha256_bytes(egg_raw)),
            "verified": True,
        },
    }
    for name, receipt in egg.get("artifacts", {}).items():
        receipts[name] = _verify_receipt(directory / name, receipt, name)
    receipts["trajectory.npz"] = _verify_receipt(
        directory / "trajectory.npz", summary.get("trajectory"), "trajectory.npz"
    )

    with np.load(directory / "trajectory.npz", allow_pickle=False) as arrays:
        required = {
            "resident_ids",
            "step",
            "phase",
            "physiology",
            "outcomes",
            "activity",
            "prediction_error",
            "learning_progress",
        }
        missing = required - set(arrays.files)
        if missing:
            raise ObservatoryError(f"trajectory is missing arrays: {sorted(missing)}")
        resident_ids = [str(value) for value in arrays["resident_ids"].tolist()]
        step = np.asarray(arrays["step"], dtype=np.int64)
        phase = np.asarray(arrays["phase"], dtype=np.int8)
        physiology = np.asarray(arrays["physiology"], dtype=np.float64)
        outcomes = np.asarray(arrays["outcomes"], dtype=np.float64)
        activity = np.asarray(arrays["activity"], dtype=np.float64)
        prediction_error = np.asarray(arrays["prediction_error"], dtype=np.float64)
        learning_progress = np.asarray(arrays["learning_progress"], dtype=np.float64)

    length = len(step)
    residents = len(resident_ids)
    expected_matrix = (length, residents)
    if (
        phase.shape != (length,)
        or physiology.shape[:2] != expected_matrix
        or physiology.shape[2] < 6
        or outcomes.shape[:2] != expected_matrix
        or outcomes.shape[2] < 4
        or activity.shape != expected_matrix
        or prediction_error.shape != expected_matrix
        or learning_progress.shape != expected_matrix
    ):
        raise ObservatoryError("development trajectory arrays are not resident-aligned")
    for name, array in (
        ("physiology", physiology),
        ("outcomes", outcomes),
        ("activity", activity),
        ("prediction_error", prediction_error),
        ("learning_progress", learning_progress),
    ):
        if not np.isfinite(array).all():
            raise ObservatoryError(f"development {name} contains non-finite values")

    world_numbers = np.asarray([_world_number(value) for value in resident_ids])
    worlds = sorted(set(world_numbers.tolist()))
    if not worlds or any(int(np.sum(world_numbers == world)) < 1 for world in worlds):
        raise ObservatoryError("development trajectory has no complete worlds")
    rows: list[dict[str, float | int | str]] = []
    for time_index in range(length - 1):
        for world in worlds:
            mask = world_numbers == world
            current_physiology = physiology[time_index, mask]
            current_outcomes = outcomes[time_index, mask]
            rows.append(
                {
                    "step": int(step[time_index]),
                    "phase": int(phase[time_index]),
                    "world_id": f"world-{world:02d}",
                    "energy": float(np.mean(current_physiology[:, 0])),
                    "gut": float(np.mean(current_physiology[:, 1])),
                    "fatigue": float(np.mean(current_physiology[:, 2])),
                    "speed": float(np.mean(current_physiology[:, 3])),
                    "support": float(np.mean(current_physiology[:, 5])),
                    "nutrition": float(np.sum(current_outcomes[:, 0])),
                    "contacts": int(np.sum(current_outcomes[:, 1] > 0)),
                    "distance": float(np.sum(current_outcomes[:, 2])),
                    "effort": float(np.mean(current_outcomes[:, 3])),
                    "activity": float(np.mean(activity[time_index, mask])),
                    "prediction_error": float(np.mean(prediction_error[time_index, mask])),
                    "learning_progress": float(np.mean(learning_progress[time_index, mask])),
                    "distance_next": float(np.sum(outcomes[time_index + 1, mask, 2])),
                    "energy_next": float(np.mean(physiology[time_index + 1, mask, 0])),
                    "prediction_error_next": float(
                        np.mean(prediction_error[time_index + 1, mask])
                    ),
                }
            )

    memory_counts: dict[str, int] = {}
    episodes_path = directory / "experienced_episodes.json.gz"
    with gzip.open(episodes_path, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)
    if not isinstance(episodes, dict):
        raise ObservatoryError("experienced episode archive must be an object")
    for resident_id, payload in episodes.items():
        memory = payload.get("memory") if isinstance(payload, dict) else None
        records = memory.get("records") if isinstance(memory, dict) else None
        memory_counts[str(resident_id)] = len(records) if isinstance(records, list) else 0

    held_out_worlds = [f"world-{world:02d}" for world in worlds if world % 4 == 3]
    if not held_out_worlds or len(held_out_worlds) == len(worlds):
        raise ObservatoryError("whole-world holdout rule produced an invalid split")
    config = egg.get("config") if isinstance(egg.get("config"), dict) else {}
    return {
        "source": {
            "path": display_path(directory),
            "format": run["format"],
            "receipts": receipts,
            "receipt_set_sha256": sha256_bytes(canonical(receipts)),
            "all_receipts_verified": True,
        },
        "summary": {
            "completed": True,
            "worlds": len(worlds),
            "residents": residents,
            "steps": length,
            "finished_unix": summary.get("finished_unix"),
            "phases": {
                "simple_steps": int(np.sum(phase == 0)),
                "rich_steps": int(np.sum(phase == 1)),
            },
            "physics": {
                "dimension": 3,
                "world_family": "MuJoCo synthetic crawler development worlds",
                "articulation_recorded": False,
            },
            "ecology": {
                "curriculum": ["simple resource garden", "rich hollow garden"],
                "world_seed_count": len(worlds),
            },
            "cognition": {
                "scope": egg.get("scope"),
                "parameter_l2_change": egg.get("parameter_l2_change"),
                "memory_records": {
                    "total": sum(memory_counts.values()),
                    "min": min(memory_counts.values()) if memory_counts else None,
                    "mean": float(np.mean(list(memory_counts.values()))) if memory_counts else None,
                    "max": max(memory_counts.values()) if memory_counts else None,
                },
                "prediction_error": _summary(prediction_error.ravel().tolist()),
                "learning_progress": _summary(learning_progress.ravel().tolist()),
            },
            "outcomes": {
                "nutrition_total": float(np.sum(outcomes[:, :, 0])),
                "contact_positive_resident_steps": int(np.sum(outcomes[:, :, 1] > 0)),
                "distance_total": float(np.sum(outcomes[:, :, 2])),
                "effort_mean": float(np.mean(outcomes[:, :, 3])),
            },
            "graph_sha256": egg.get("graph_sha256"),
            "config": {
                key: config.get(key)
                for key in (
                    "dt",
                    "worlds",
                    "simple_steps",
                    "rich_steps",
                    "seed",
                    "inheritance_seed",
                )
            },
        },
        "rows": rows,
        "worlds": [f"world-{world:02d}" for world in worlds],
        "held_out_worlds": held_out_worlds,
    }


def freeze_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    payload = stream.getvalue().encode("utf-8")
    write_bytes_atomic(path, payload)
    return {
        "file": path.name,
        "rows": len(rows),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "artifact_uri": artifact_uri(sha256_bytes(payload)),
    }


def _capture_native_stderr(callback: Callable[[], Any]) -> tuple[Any, list[str]]:
    with tempfile.TemporaryFile() as capture:
        saved_stderr = os.dup(2)
        try:
            os.dup2(capture.fileno(), 2)
            result = callback()
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        capture.seek(0)
        messages = capture.read().decode(errors="replace").strip().splitlines()
    return result, messages


def _metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def fit_development_models(
    rows: list[dict[str, Any]], held_out_worlds: list[str], output_dir: Path
) -> dict[str, Any]:
    try:
        import gamfit
    except ImportError:
        return {
            "status": "unavailable",
            "reason": f"pinned native gamfit {GAMFIT_VERSION} is not installed",
            "models": {},
        }
    if gamfit.__version__ != GAMFIT_VERSION:
        return {
            "status": "unavailable",
            "reason": f"expected gamfit {GAMFIT_VERSION}, loaded {gamfit.__version__}",
            "models": {},
        }
    build = gamfit.build_info()
    if not build.get("available"):
        return {
            "status": "unavailable",
            "reason": "gamfit native extension is unavailable",
            "models": {},
        }

    held_out = set(held_out_worlds)
    max_step = max(int(row["step"]) for row in rows)
    prepared = []
    for row in rows:
        prepared.append(
            {
                **row,
                "contact": float(row["contacts"] > 0),
                "step_fraction": float(row["step"]) / max(max_step, 1),
            }
        )
    train = [row for row in prepared if row["world_id"] not in held_out]
    test = [row for row in prepared if row["world_id"] in held_out]
    if len(train) < 100 or len(test) < 100:
        raise ObservatoryError("whole-world model split is too small")

    result: dict[str, Any] = {
        "status": "complete",
        "interpretation": "descriptive held-out prediction only; no causal claim",
        "library": {
            "name": "gamfit",
            "version": gamfit.__version__,
            "source_commit": GAMFIT_SOURCE_COMMIT,
            "native_extension_available": True,
            "native_crate": build.get("crate"),
            "native_engine_crate": build.get("engine_crate"),
        },
        "split": {
            "unit": "complete physical world",
            "rule": "world index modulo 4 equals 3",
            "training_worlds": sorted({str(row["world_id"]) for row in train}),
            "held_out_worlds": sorted(held_out),
            "training_rows": len(train),
            "held_out_rows": len(test),
            "adjacent_rows_cross_boundary": False,
        },
        "models": {},
    }
    failures = 0
    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        formula = spec["formula"]
        predictor_names = spec["predictors"]
        model_path = output_dir / f"{model_name}.gam.gz"
        native_path = output_dir / f".{model_name}.gam.tmp"
        (output_dir / f"{model_name}.gam").unlink(missing_ok=True)
        model_path.unlink(missing_ok=True)
        native_path.unlink(missing_ok=True)
        try:
            fit_rows = [
                {name: row[name] for name in (target, *predictor_names)}
                for row in train
            ]
            gamfit.validate_formula(fit_rows, formula)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model, native_messages = _capture_native_stderr(
                    lambda: gamfit.fit(fit_rows, formula)
                )
            predictors = [
                {name: row[name] for name in predictor_names}
                for row in test
            ]
            observed = np.asarray([row[target] for row in test], dtype=np.float64)
            predicted = np.asarray(model.predict(predictors), dtype=np.float64)
            persistence = np.asarray(
                [row[spec["persistence"]] for row in test], dtype=np.float64
            )
            train_mean = float(np.mean([row[target] for row in train]))
            mean_baseline = np.full(observed.shape, train_mean)
            model.save(native_path)
            reloaded = gamfit.load(native_path)
            reload_prediction = np.asarray(reloaded.predict(predictors), dtype=np.float64)
            reload_delta = float(np.max(np.abs(predicted - reload_prediction)))
            if not np.isfinite(predicted).all() or reload_delta != 0.0:
                raise ObservatoryError("native model did not round-trip exactly")
            native_payload = native_path.read_bytes()
            write_bytes_atomic(
                model_path,
                gzip.compress(native_payload, compresslevel=9, mtime=0),
            )
            model_hash = sha256_file(model_path)
            fit_metrics = _metrics(predicted, observed)
            persistence_metrics = _metrics(persistence, observed)
            mean_metrics = _metrics(mean_baseline, observed)
            result["models"][model_name] = {
                "status": "complete",
                "description": spec["description"],
                "formula": formula,
                "target": target,
                "held_out": fit_metrics,
                "baselines": {
                    "persistence": persistence_metrics,
                    "training_mean": mean_metrics,
                },
                "rmse_vs_persistence": (
                    fit_metrics["rmse"] / persistence_metrics["rmse"]
                    if persistence_metrics["rmse"]
                    else None
                ),
                "native_messages": native_messages,
                "warnings": [str(item.message) for item in caught],
                "artifact": {
                    "file": model_path.name,
                    "bytes": model_path.stat().st_size,
                    "sha256": model_hash,
                    "artifact_uri": artifact_uri(model_hash),
                    "compression": "gzip",
                    "native_payload_sha256": sha256_bytes(native_payload),
                    "reload_max_abs_prediction_delta": reload_delta,
                },
            }
        except Exception as exc:
            failures += 1
            result["models"][model_name] = {
                "status": "failed",
                "description": spec["description"],
                "formula": formula,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        finally:
            native_path.unlink(missing_ok=True)
    if failures:
        result["status"] = "partial" if failures < len(MODEL_SPECS) else "failed"
    return result


def _artifact_entry(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "artifact_uri": artifact_uri(digest),
    }


def build_weave_request(
    world: dict[str, Any], development: dict[str, Any], models: dict[str, Any]
) -> dict[str, Any]:
    world_id = "source:adult-3d-checkpoint"
    development_id = "source:development-run"
    journal: list[dict[str, Any]] = [
        {
            "id": world_id,
            "time": world["summary"]["model_time"],
            "kind": "verified_source",
            "text": "Verified 3D whole-world checkpoint.",
            "artifact_uri": world["source"]["artifact_uri"],
        },
        {
            "id": development_id,
            "time": development["summary"]["steps"],
            "kind": "verified_source",
            "text": "Verified eight-world developmental run.",
            "artifact_uri": development["source"]["receipts"]["summary.json"]["artifact_uri"],
        },
    ]
    for event in world["journal"]:
        journal.append(
            {
                **event,
                "id": f"adult-event:{event['id']}",
                "source_event_id": event["id"],
            }
        )

    evidence: list[dict[str, Any]] = [
        {
            "id": "summary:adult-3d",
            "time": world["summary"]["model_time"],
            "text": (
                f"Three-resident 3D state at tick {world['summary']['tick']}; "
                f"{world['summary']['cognition']['memory_records']} externalized memory-count summary."
            ),
            "artifact_uri": world["source"]["artifact_uri"],
            "parent_ids": [world_id],
        },
        {
            "id": "summary:development",
            "time": development["summary"]["steps"],
            "text": (
                f"Development across {development['summary']['worlds']} worlds and "
                f"{development['summary']['residents']} residents."
            ),
            "artifact_uri": development["source"]["receipts"]["trajectory.npz"]["artifact_uri"],
            "parent_ids": [development_id],
        },
    ]
    model_ids = []
    for model_name, model in models.get("models", {}).items():
        if model.get("status") != "complete":
            continue
        model_id = f"model:{model_name}"
        model_ids.append(model_id)
        ratio = model.get("rmse_vs_persistence")
        ratio_text = f"{ratio:.6g}" if isinstance(ratio, (int, float)) else "undefined"
        evidence.append(
            {
                "id": model_id,
                "time": development["summary"]["steps"],
                "text": (
                    f"Native GAM {model_name} on complete held-out worlds; "
                    f"RMSE/persistence={ratio_text}. No causal claim."
                ),
                "artifact_uri": model["artifact"]["artifact_uri"],
                "parent_ids": ["summary:development"],
            }
        )
    comparison_parents = ["summary:adult-3d", "summary:development", *model_ids]
    evidence.append(
        {
            "id": "comparison:adult-and-development",
            "time": development["summary"]["steps"],
            "text": (
                "Cross-source comparison joins the adult whole-world state, developmental "
                "population summary, and held-out dynamics. It is archival evidence, not creature memory."
            ),
            "artifact_uri": None,
            "parent_ids": comparison_parents,
        }
    )
    return {
        "habitat_id": world["summary"]["habitat_id"],
        "journal": journal,
        "evidence": evidence,
    }


def run_native_weave(request_path: Path, output_path: Path) -> dict[str, Any]:
    if not WEAVE_MANIFEST.is_file():
        return {
            "status": "unavailable",
            "reason": "Universal Weave Cargo manifest is absent",
            "source_commit": WEAVE_SOURCE_COMMIT,
        }
    command = [
        "cargo",
        "run",
        "--quiet",
        "--locked",
        "--manifest-path",
        str(WEAVE_MANIFEST),
        "--",
        "--input",
        str(request_path),
        "--output",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=240, check=True)
        receipt = json.loads(completed.stdout)
        if not isinstance(receipt, dict):
            raise ObservatoryError("native Weave receipt is not an object")
        receipt["artifact"] = output_path.name
        return {"status": "complete", **receipt}
    except FileNotFoundError:
        return {
            "status": "unavailable",
            "reason": "cargo is unavailable",
            "source_commit": WEAVE_SOURCE_COMMIT,
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        return {
            "status": "failed",
            "reason": str(detail).strip()[-2000:],
            "source_commit": WEAVE_SOURCE_COMMIT,
        }


def format_dag(request: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    records = [*request["journal"], *request["evidence"]]
    by_id = {str(record["id"]): record for record in records}
    depths: dict[str, int] = {}
    nodes = []
    edges = []
    for record in records:
        source_id = str(record["id"])
        parents = [str(value) for value in record.get("parent_ids", [])]
        depth = max((depths.get(parent, 0) + 1 for parent in parents), default=0)
        depths[source_id] = depth
        kind = str(record.get("kind", "evidence" if parents else "episode"))
        nodes.append(
            {
                "id": source_id,
                "kind": kind,
                "title": str(record.get("text", source_id)),
                "time": record.get("time"),
                "artifact_uri": record.get("artifact_uri"),
                "depth": depth,
                "lane": "comparison" if len(parents) > 1 else ("analysis" if parents else "source"),
                "parent_count": len(parents),
            }
        )
        for parent in parents:
            if parent not in by_id:
                raise ObservatoryError(f"formatted DAG parent {parent!r} is absent")
            edges.append({"source": parent, "target": source_id})
    return {
        "schema_version": 1,
        "nodes": nodes,
        "edges": edges,
        "levels": max(depths.values(), default=0) + 1,
        "multi_parent_nodes": sum(node["parent_count"] > 1 for node in nodes),
        "native_roundtrip": {
            "status": receipt.get("status"),
            "reload_equal": receipt.get("reload_equal"),
            "validated_after_reload": receipt.get("validated_after_reload"),
            "node_count": receipt.get("node_count"),
        },
    }


def build_observatory(
    world_path: Path,
    development_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    world = load_world_checkpoint(world_path)
    development = load_development_run(development_dir)
    if world["summary"]["anatomy_sha256"] != development["summary"]["graph_sha256"]:
        raise ObservatoryError("adult and developmental runs use different anatomy hashes")
    output_dir.mkdir(parents=True, exist_ok=True)

    world_timeseries = freeze_csv(
        output_dir / "adult_world_timeseries.csv",
        ("resident_id", "time", "x", "y", "z", "energy", "activity", "memory"),
        world["history_rows"],
    )
    development_timeseries = freeze_csv(
        output_dir / "development_world_timeseries.csv",
        TIMESERIES_FIELDS,
        development["rows"],
    )
    models = fit_development_models(
        development["rows"], development["held_out_worlds"], output_dir
    )
    request = build_weave_request(world, development, models)
    request_path = output_dir / "observatory.weave-request.json"
    write_json_atomic(request_path, request)
    weave_path = output_dir / "observatory.weave.json"
    weave_path.unlink(missing_ok=True)
    weave = run_native_weave(request_path, weave_path)
    dag = format_dag(request, weave)
    dag_path = output_dir / "dag-view.json"
    write_json_atomic(dag_path, dag)

    report = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "chreatures-3d-observatory",
        "source_snapshot_unix": development["summary"].get("finished_unix"),
        "archive_scope": (
            "external immutable evidence archive; summaries and references are not resident memory"
        ),
        "source_set_sha256": sha256_bytes(
            canonical({"world": world["source"], "development": development["source"]})
        ),
        "world": {"source": world["source"], "summary": world["summary"]},
        "development": {
            "source": development["source"],
            "summary": development["summary"],
            "whole_world_holdout": development["held_out_worlds"],
        },
        "timeseries": {
            "adult_world": world_timeseries,
            "development_world": development_timeseries,
        },
        "gamfit": models,
        "weave": weave,
        "dag": {
            "file": dag_path.name,
            "nodes": len(dag["nodes"]),
            "edges": len(dag["edges"]),
            "multi_parent_nodes": dag["multi_parent_nodes"],
        },
        "limitations": [
            "The adult checkpoint predates explicit body_mode and diffusion-field fields.",
            "The development run used synthetic crawler bodies; it is not articulated-body evidence.",
            "GAM results are descriptive held-out predictions and do not establish causality or skill.",
            "The observatory is an external archive and does not alter resident memory.",
        ],
    }
    report_path = output_dir / "observatory.json"
    write_json_atomic(report_path, report)

    artifact_paths = [
        report_path,
        request_path,
        weave_path,
        dag_path,
        output_dir / world_timeseries["file"],
        output_dir / development_timeseries["file"],
    ]
    for model in models.get("models", {}).values():
        if model.get("status") == "complete":
            artifact_paths.append(output_dir / model["artifact"]["file"])
    artifacts = {
        path.name: _artifact_entry(path)
        for path in artifact_paths
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "format": "chreatures-observatory-manifest-v1",
        "report": report_path.name,
        "dag_view": dag_path.name,
        "artifacts": artifacts,
        "artifact_set_sha256": sha256_bytes(canonical(artifacts)),
        "all_sources_verified": True,
        "native_weave_roundtrip": weave.get("status") == "complete"
        and weave.get("reload_equal") is True
        and weave.get("validated_after_reload") is True,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return {"manifest": manifest, "report": report}


def create_observatory_router(artifact_root: Path = DEFAULT_OUTPUT) -> APIRouter:
    """Return a read-only router; callers explicitly include it in their app."""

    if APIRouter is None or HTTPException is None or FileResponse is None:
        raise RuntimeError("FastAPI is required to create the observatory router")
    root = Path(artifact_root)
    router = APIRouter(prefix="/api/observatory", tags=["observatory"])

    def load(name: str) -> dict[str, Any]:
        path = root / name
        if not path.is_file():
            raise HTTPException(503, "observatory artifacts have not been built")
        value, _ = _read_object(path)
        return value

    def load_verified_json(name: str, manifest: dict[str, Any]) -> dict[str, Any]:
        artifacts = manifest.get("artifacts")
        record = artifacts.get(name) if isinstance(artifacts, dict) else None
        path = root / name
        if (
            not isinstance(record, dict)
            or record.get("file") != name
            or not path.is_file()
            or sha256_file(path) != record.get("sha256")
        ):
            raise HTTPException(503, "observatory artifact failed its immutable receipt")
        return load(name)

    @router.get("")
    def overview() -> dict[str, Any]:
        manifest = load("manifest.json")
        report = load_verified_json(
            str(manifest.get("report", "observatory.json")), manifest
        )
        return {
            "manifest": manifest,
            "world": report.get("world"),
            "development": report.get("development"),
            "gamfit": report.get("gamfit"),
            "limitations": report.get("limitations"),
        }

    @router.get("/graph")
    def graph() -> dict[str, Any]:
        manifest = load("manifest.json")
        return load_verified_json(str(manifest.get("dag_view", "dag-view.json")), manifest)

    @router.get("/artifacts/{artifact_name}")
    def artifact(artifact_name: str) -> FileResponse:
        manifest = load("manifest.json")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or artifact_name not in artifacts:
            raise HTTPException(404, "unknown observatory artifact")
        record = artifacts[artifact_name]
        if not isinstance(record, dict) or record.get("file") != artifact_name:
            raise HTTPException(500, "invalid observatory manifest")
        path = root / artifact_name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise HTTPException(503, "observatory artifact failed its immutable receipt")
        return FileResponse(path, filename=artifact_name)

    return router


router = create_observatory_router() if APIRouter is not None else None
