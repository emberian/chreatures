#!/usr/bin/env python3
"""Finalize a completed population evaluation from sealed, non-simulated evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from chreatures.population import canonical_bytes


FORMAT = "chreatures-population-episode-evaluation-v1"
REPORT_FORMAT = "chreatures-population-evidence-finalizer-v1"
CONTROLLER_RESIDENT_FIELDS = frozenset({
    "actual_attained", "attributed", "cancelled_total", "completed",
    "completed_total", "frozen_total", "learned", "learned_total",
    "measurement_latest_rms", "measurement_min_rms", "measurement_samples",
    "measurement_start_rms", "measurement_window_ending_last_observed_tick",
    "observed_normalized_progress", "population_response_in_domain",
    "population_response_in_domain_total", "population_response_out_of_domain_total",
    "reward", "skipped_total", "summed_return",
})
CONTROLLER_GLOBAL_FIELDS = frozenset({
    "population_feature_contract_identity", "population_response_identity",
})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


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


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def verify_canonical(value: Mapping[str, Any], field: str, description: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"{description} lacks {field}")
    body = {key: item for key, item in value.items() if key != field}
    if canonical_sha256(body) != claimed:
        raise ValueError(f"{description} canonical identity differs")
    return claimed


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path leaves evaluation root: {relative}")
    return path


def verify_receipt_file(root: Path, receipt: Mapping[str, Any]) -> Path:
    path = contained(root, str(receipt["path"]))
    if not path.is_file() or path.stat().st_size != int(receipt["bytes"]):
        raise ValueError(f"checkpoint artifact size differs: {path}")
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError(f"checkpoint artifact hash differs: {path}")
    return path


def resident_summary(summary: Mapping[str, Any], resident: int, count: int) -> dict[str, Any]:
    resident_fields = summary.get("resident_axis_keys")
    if not isinstance(resident_fields, list) or not resident_fields:
        raise ValueError("trajectory summary omits resident-axis contract")
    fields = set(resident_fields)
    result: dict[str, Any] = {}
    for key, value in summary.items():
        if key in fields:
            array = np.asarray(value)
            if array.ndim < 1 or array.shape[0] != count:
                raise ValueError(f"trajectory resident axis differs: {key}")
            result[key] = array[resident].tolist()
        else:
            result[key] = value
    return result


def supervisor_receipt(path: Path) -> dict[str, Any]:
    value = load_json(path / "exit-status.json")
    start, end = int(value["start_epoch"]), int(value["end_epoch"])
    if end < start:
        raise ValueError("supervision wall span is negative")
    return {"artifact": artifact(path / "exit-status.json"), **value, "wall_span_seconds": end - start}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--original-supervision", type=Path, required=True)
    parser.add_argument("--resume-supervision", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-finalized", type=Path)
    parser.add_argument("--supplemental-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    evaluation = args.evaluation.resolve()
    if (args.output is None) == (args.verify_finalized is None):
        raise ValueError("choose one of --output or --verify-finalized")
    if args.verify_finalized is not None and args.supplemental_output is None:
        raise ValueError("--verify-finalized requires --supplemental-output")
    if args.output is not None and args.supplemental_output is not None:
        raise ValueError("--supplemental-output is only for --verify-finalized")
    output = (
        args.output.resolve() if args.output is not None
        else args.supplemental_output.resolve()
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite finalized report: {output}")
    if output.parent != evaluation:
        raise ValueError("finalized report must remain beside its sealed evaluation identity")

    identity_path = evaluation / "identity.json"
    identity = load_json(identity_path)
    identity_sha256 = verify_canonical(identity, "sha256", "evaluation identity")
    lives = identity.get("life_records")
    if not isinstance(lives, list) or not lives:
        raise ValueError("evaluation identity omits life records")
    count = len(lives)
    worlds = max(int(item["world_slot"]) for item in lives) + 1
    residents = count // worlds
    if worlds * residents != count:
        raise ValueError("life records do not form a fixed world cohort")

    latest_path = evaluation / "latest.json"
    latest = load_json(latest_path)
    if latest.get("evaluation_identity_sha256") != identity_sha256:
        raise ValueError("latest checkpoint points at another evaluation")
    completed_steps = int(latest["completed_steps"])
    expected_steps = int(identity["execution"]["steps"])
    if completed_steps != expected_steps:
        raise ValueError("evaluation did not reach its declared final boundary")
    checkpoint_root = contained(evaluation, str(latest["checkpoint"]))
    checkpoint_path = checkpoint_root / "checkpoint.json"
    if file_sha256(checkpoint_path) != latest["checkpoint_receipt_sha256"]:
        raise ValueError("latest checkpoint receipt hash differs")
    checkpoint = load_json(checkpoint_path)
    if (
        checkpoint.get("evaluation_identity_sha256") != identity_sha256
        or int(checkpoint.get("completed_steps", -1)) != completed_steps
        or checkpoint.get("life_ids") != [item["life_id"] for item in lives]
    ):
        raise ValueError("final checkpoint boundary differs from evaluation identity")
    files = checkpoint["files"]
    boundary_path = verify_receipt_file(checkpoint_root, files["boundary"])
    verify_receipt_file(checkpoint_root, files["worlds"])
    verify_receipt_file(checkpoint_root, files["controller"])
    verify_receipt_file(checkpoint_root, files["neural"])
    trajectory_paths = [verify_receipt_file(checkpoint_root, item) for item in files["trajectories"]]
    if len(trajectory_paths) != worlds:
        raise ValueError("checkpoint trajectory world count differs")

    telemetry_path = evaluation / "telemetry" / f"step-{completed_steps:010d}.json"
    telemetry = load_json(telemetry_path)
    if int(telemetry.get("completed_steps", -1)) != completed_steps:
        raise ValueError("final telemetry boundary differs")
    summaries = telemetry.get("trajectory")
    if not isinstance(summaries, list) or len(summaries) != worlds:
        raise ValueError("final trajectory telemetry world count differs")
    from _world_kernels import PopulationTrajectory

    for world_index, (summary, trajectory_path) in enumerate(
        zip(summaries, trajectory_paths, strict=True)
    ):
        restored = PopulationTrajectory(
            residents,
            summary["world_size"],
            float(summary["spatial_bin_width"]),
            float(summary["sampling_dt_seconds"]),
        )
        restored.restore(trajectory_path.read_bytes())
        if jsonable(restored.summary()) != summary:
            raise ValueError(
                f"telemetry trajectory summary differs from checkpoint state: {world_index}"
            )

    trace_paths = sorted((evaluation / "gam_trace").glob("tick-*.npz"))
    if not trace_paths:
        raise ValueError("evaluation has no causal GAM trace")
    trace_path = trace_paths[-1]
    with np.load(trace_path, allow_pickle=False) as trace:
        if str(trace["evaluation_identity_sha256"].item()) != identity_sha256:
            raise ValueError("final trace identity differs")
        ticks = np.asarray(trace["tick"], dtype=np.uint64)
        if ticks.ndim != 1 or len(ticks) == 0 or int(ticks[-1]) + 1 != completed_steps:
            raise ValueError("final trace does not reach completed boundary")
        last_pre = np.asarray(trace["pre_physiology12"][-1], dtype=np.float32)
        last_actions = np.asarray(trace["executed_action12"][-1], dtype=np.float32)
        last_post = np.asarray(trace["post_physiology12"][-1], dtype=np.float32)
        last_outcomes = np.asarray(trace["outcomes8"][-1], dtype=np.float32)
        last_flows = np.asarray(trace["organ_flows3"][-1], dtype=np.float32)
    for name, array, width in (
        ("pre physiology", last_pre, 12), ("executed action", last_actions, 12),
        ("post physiology", last_post, 12), ("outcome", last_outcomes, 8),
        ("organ flow", last_flows, 3),
    ):
        if array.shape != (count, width) or not np.isfinite(array).all():
            raise ValueError(f"final {name} array differs")

    with np.load(boundary_path, allow_pickle=False) as boundary:
        if int(boundary["completed_steps"].item()) != completed_steps:
            raise ValueError("boundary completed step differs")
        if not np.array_equal(boundary["actual_previous_actions"], last_actions):
            raise ValueError("boundary action differs from final causal trace")
        if not np.array_equal(boundary["physiology"], last_post):
            raise ValueError("boundary physiology differs from final causal trace")
        boundary_controller = {
            key.removeprefix("controller_outcome."): np.asarray(boundary[key])
            for key in boundary.files if key.startswith("controller_outcome.")
        }

    controller_rows: dict[str, np.ndarray] = {}
    controller_globals: dict[str, Any] = {}
    for name, value in telemetry["controller_outcomes"].items():
        array = np.asarray(value)
        if array.shape == (count,):
            controller_rows[name] = array
        elif array.shape == () and isinstance(value, str) and len(value) == 64:
            controller_globals[name] = value
        else:
            raise ValueError(f"controller outcome shape differs: {name} {array.shape}")
    if set(controller_rows) != CONTROLLER_RESIDENT_FIELDS:
        raise ValueError("controller resident outcome field set differs")
    if set(controller_globals) != CONTROLLER_GLOBAL_FIELDS:
        raise ValueError("controller global outcome field set differs")
    for name, value in boundary_controller.items():
        if name not in controller_rows or not np.array_equal(value, controller_rows[name]):
            raise ValueError(f"boundary controller outcome differs from telemetry: {name}")

    profile_path = Path(identity["profile_file"]["path"])
    if file_sha256(profile_path) != identity["profile_file"]["sha256"]:
        raise ValueError("profile file differs from sealed identity")
    profile = load_json(profile_path)
    variants = profile["value"]["family"]["variants"]
    environments = {item["environment_sha256"]: item["environment_record"] for item in variants}

    rows = []
    cohort_rows = []
    for world_index, (summary, trajectory_path) in enumerate(zip(summaries, trajectory_paths, strict=True)):
        trajectory_sha = file_sha256(trajectory_path)
        if trajectory_sha != files["trajectories"][world_index]["sha256"]:
            raise ValueError("trajectory checkpoint hash differs")
        cohort_rows.append({"world_slot": world_index, "snapshot_sha256": trajectory_sha, "summary": summary})
    for life in lives:
        world_index, resident_index = int(life["world_slot"]), int(life["resident_slot"])
        flat = world_index * residents + resident_index
        metrics = resident_summary(summaries[world_index], resident_index, residents)
        cohort_sha = files["trajectories"][world_index]["sha256"]
        trajectory_sha = canonical_sha256({
            "life": life,
            "cohort_snapshot_sha256": cohort_sha,
            "resident_metrics": metrics,
        })
        rows.append({
            **life,
            "status": "completed",
            "committed_ticks": completed_steps,
            "trajectory_sha256": trajectory_sha,
            "cohort_trajectory_snapshot_sha256": cohort_sha,
            "trajectory_metrics": metrics,
            "controller_outcome": {name: value[flat].item() for name, value in controller_rows.items()},
            "last_transition": {
                "tick": completed_steps - 1,
                "pre_physiology12": last_pre[flat].astype(float).tolist(),
                "executed_action12": last_actions[flat].astype(float).tolist(),
                "post_physiology12": last_post[flat].astype(float).tolist(),
                "outcomes8": last_outcomes[flat].astype(float).tolist(),
                "organ_flows3": last_flows[flat].astype(float).tolist(),
            },
        })

    original_supervision = supervisor_receipt(args.original_supervision.resolve())
    resume_supervision = [supervisor_receipt(path.resolve()) for path in args.resume_supervision]
    failure_receipts = []
    for path in sorted((evaluation / "failures").glob("*.json")):
        failure = load_json(path)
        verify_canonical(failure, "content_sha256", f"failure receipt {path.name}")
        failure_receipts.append({
            "artifact": artifact(path),
            "content_sha256": failure["content_sha256"],
            "traceback_sha256": failure["traceback_sha256"],
            "error_type": failure["error_type"],
            "committed_ticks": sorted({int(row["committed_ticks"]) for row in failure["candidate_failures"]}),
        })

    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "completed",
        "evaluation_identity_sha256": identity_sha256,
        "completed_steps": completed_steps,
        "completed_resident_transitions": completed_steps * count,
        "worlds": worlds,
        "residents_per_world": residents,
        "lives": rows,
        "environments": environments,
        "trajectory_cohorts": cohort_rows,
        "final_checkpoint": latest,
        "controller": {
            "resident_artifact": identity["resident_artifact"],
            "population_response": identity["population_response"],
            "final_global_outcomes": controller_globals,
        },
        "brain": {
            "graph": identity["graph"], "ports": identity["ports"],
            "neural_compatibility_group": identity["neural_compatibility_group"],
        },
        "transport_timing": telemetry["transport_timing"],
        "performance": {
            "original_supervision_wall_span_seconds": original_supervision["wall_span_seconds"],
            "resident_transitions_per_original_wall_second": completed_steps * count / original_supervision["wall_span_seconds"],
            "scope": "includes original startup, all checkpoints and the failed first result assembly",
        },
        "last_step_audit": {
            "tick": completed_steps - 1,
            "executed_action_mean": last_actions.mean(axis=0).astype(float).tolist(),
            "outcome_sum": last_outcomes.sum(axis=0).astype(float).tolist(),
            "organ_flow_sum": last_flows.sum(axis=0).astype(float).tolist(),
        },
        "evidence_finalizer": {
            "format": REPORT_FORMAT,
            "tool": artifact(Path(__file__).resolve()),
            "evaluation_identity": artifact(identity_path),
            "latest": artifact(latest_path),
            "checkpoint_receipt": artifact(checkpoint_path),
            "final_telemetry": {
                **artifact(telemetry_path),
                "authentication": "hash measured by finalizer; no upstream telemetry manifest",
            },
            "final_trace": {
                **artifact(trace_path),
                "authentication": "hash measured by finalizer; no upstream trace manifest",
            },
            "original_supervision": original_supervision,
            "resume_supervision": resume_supervision,
            "preserved_failures": failure_receipts,
            "method": "authenticated final checkpoint, final telemetry and causal trace; no world restore, reset, sampling or advance",
        },
        "completed_utc": checkpoint["created_utc"],
        "finalized_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if args.verify_finalized is not None:
        finalized_path = args.verify_finalized.resolve()
        finalized = load_json(finalized_path)
        finalized_content = verify_canonical(
            finalized, "content_sha256", "finalized population result"
        )
        if (
            finalized.get("format") != FORMAT
            or finalized.get("evaluation_identity_sha256") != identity_sha256
            or int(finalized.get("completed_steps", -1)) != completed_steps
        ):
            raise ValueError("finalized population result protocol differs")
        supplement: dict[str, Any] = {
            "format": "chreatures-population-evidence-finalizer-supplement-v1",
            "evaluation_identity_sha256": identity_sha256,
            "finalized_result": {
                **artifact(finalized_path), "content_sha256": finalized_content,
            },
            "tool": artifact(Path(__file__).resolve()),
            "final_checkpoint_receipt": artifact(checkpoint_path),
            "checkpoint_trajectory_sha256": [
                item["sha256"] for item in files["trajectories"]
            ],
            "trajectory_summary_restore_exact": True,
            "controller_resident_fields": sorted(CONTROLLER_RESIDENT_FIELDS),
            "controller_global_fields": sorted(CONTROLLER_GLOBAL_FIELDS),
            "final_telemetry": {
                **artifact(telemetry_path),
                "authentication": "hash measured by finalizer; no upstream telemetry manifest",
            },
            "final_trace": {
                **artifact(trace_path),
                "authentication": "hash measured by finalizer; no upstream trace manifest",
            },
            "scope": (
                "read-only native trajectory decoding and controller field validation; "
                "no world, neural or controller restore, sampling, reset or advance"
            ),
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        supplement["content_sha256"] = canonical_sha256(supplement)
        data = json.dumps(supplement, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        print(json.dumps({
            "output": str(output), "content_sha256": supplement["content_sha256"],
            "trajectory_summary_restore_exact": True,
        }, sort_keys=True))
        return 0
    result["content_sha256"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"output": str(output), "content_sha256": result["content_sha256"], "lives": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
