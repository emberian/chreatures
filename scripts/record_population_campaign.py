#!/usr/bin/env python3
"""Convert authenticated population runs into an idempotent native Weave ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.population import (
    PopulationSearch,
    canonical_bytes as population_canonical_bytes,
)
from chreatures.population_evidence import (
    BATCH_FORMAT,
    LEDGER_FORMAT,
    PopulationEvidenceError,
    atomic_write_json,
    blob_ref,
    canonical_bytes,
    empty_ledger,
    environment_record_from_native,
    evaluation_records_from_native,
    evidence_record,
    genome_record_from_native,
    read_json,
    sha256_file,
    validate_records,
)


EVALUATION_FORMAT = "chreatures-population-episode-evaluation-v1"
FAILURE_FORMAT = "chreatures-population-evaluation-failure-v1"
CHECKPOINT_FORMAT = "chreatures-population-coupled-checkpoint-v1"
PANEL_FORMAT = "chreatures-population-probe-panel-v1"
CAMPAIGN_FORMAT = "chreatures-population-campaign-v1"
LIFE_KEYS = (
    "life_id",
    "human_label",
    "assignment_file_sha256",
    "world_slot",
    "resident_slot",
    "world_id",
    "environment",
    "environment_seed",
    "evaluation_seed",
    "environment_sha256",
    "candidate_sha256",
)
SPATIAL_CELL_SCALE = 256.0


def _metric_row(
    life: Mapping[str, Any], environment: Mapping[str, Any]
) -> dict[str, float]:
    row = life.get("trajectory_metrics")
    dimensions = environment.get("dimensions_m")
    if not isinstance(row, Mapping) or not isinstance(dimensions, list) or len(dimensions) != 3:
        raise PopulationEvidenceError("campaign metric inputs differ")
    ticks = int(life["committed_ticks"])
    valid = int(row["valid_ticks"])
    dt = float(row["sampling_dt_seconds"])
    action = row["executed_action_mean"]
    physiology = row["physiology_mean"]
    if ticks <= 0 or valid <= 0 or valid > ticks or dt != 0.05:
        raise PopulationEvidenceError("completed life sampling differs")
    metrics = {
        "mean_action_thrust": float(action[0]),
        "spatial_coverage": float(row["visited_spatial_cells"]) / SPATIAL_CELL_SCALE,
        "elevation_fraction": float(row["height_range"]) / float(dimensions[2]),
        "signal_activity_rate": float(row["signal_activity_sum"]) / valid,
        "allocated_mass_rate": float(row["allocation_mass_sum"]) / (valid * dt),
        "mean_energy": float(physiology[0]),
        "energy_delta": float(row["energy_change"]),
        "mean_effort": float(row["effort_sum"]) / valid,
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise PopulationEvidenceError("derived campaign metric is nonfinite")
    return metrics


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise PopulationEvidenceError(f"{name} must be a lowercase SHA-256")
    return value


def _verified_identity(document: Mapping[str, Any], field: str, name: str) -> str:
    expected = _hash(document.get(field), f"{name}.{field}")
    body = dict(document)
    body.pop(field)
    if hashlib.sha256(population_canonical_bytes(body)).hexdigest() != expected:
        raise PopulationEvidenceError(f"{name} content identity differs")
    return expected


def _write_blob(
    directory: Path,
    value: Any,
    *,
    role: str,
    media_type: str = "application/json",
    population_canonical: bool = False,
) -> tuple[Path, dict[str, Any]]:
    payload = (
        population_canonical_bytes(value)
        if population_canonical
        else canonical_bytes(value)
    )
    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise PopulationEvidenceError(f"content-addressed blob collision at {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    return path, blob_ref(
        role=role,
        sha256=digest,
        bytes=len(payload),
        media_type=media_type,
        verification="verified_local_sha256",
    )


def _source_blob(path: Path, role: str) -> dict[str, Any]:
    return blob_ref(
        role=role,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        media_type="application/json",
        verification="verified_local_sha256",
    )


def _campaign(
    path: Path, expected_config_sha256: str, expected_panel_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    source = read_json(path)
    if (
        source.get("format") != CAMPAIGN_FORMAT
        or source.get("version") != 1
        or source.get("search_config_sha256") != expected_config_sha256
    ):
        raise PopulationEvidenceError("campaign manifest differs from native search")
    value = source.get("probe_panel")
    if not isinstance(value, Mapping) or value.get("format") != PANEL_FORMAT:
        raise PopulationEvidenceError("probe panel format differs")
    logical = hashlib.sha256(population_canonical_bytes(value)).hexdigest()
    if logical != expected_panel_sha256:
        raise PopulationEvidenceError("probe panel differs from native search config")
    policy = _hash(value.get("controller_file_sha256"), "probe policy")
    if value.get("action_mode") not in {"sample", "argmax"}:
        raise PopulationEvidenceError("probe panel action mode differs")
    if not isinstance(value.get("fine_tuning"), bool):
        raise PopulationEvidenceError("probe panel fine_tuning is not boolean")
    environments = source.get("environment_index")
    if not isinstance(environments, Mapping):
        raise PopulationEvidenceError("campaign manifest has no environment index")
    return dict(source), dict(value), [policy]


def _verify_checkpoint_directory(
    root: Path, identity_sha256: str, expected_life_ids: set[str]
) -> tuple[dict[str, Any], Path]:
    receipt_path = root / "checkpoint.json"
    receipt = read_json(receipt_path)
    if (
        receipt.get("format") != CHECKPOINT_FORMAT
        or receipt.get("version") != 1
        or receipt.get("evaluation_identity_sha256") != identity_sha256
    ):
        raise PopulationEvidenceError(f"checkpoint identity differs at {root}")
    life_ids = receipt.get("life_ids")
    if (
        not isinstance(life_ids, list)
        or len(life_ids) != len(set(life_ids))
        or set(life_ids) != expected_life_ids
    ):
        raise PopulationEvidenceError("checkpoint life allocation set differs")
    completed_steps = receipt.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
        or receipt.get("completed_resident_transitions")
        != completed_steps * len(expected_life_ids)
    ):
        raise PopulationEvidenceError("checkpoint transition count differs")
    items: list[Mapping[str, Any]] = []
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise PopulationEvidenceError("checkpoint has no file receipts")
    for key in ("worlds", "controller", "boundary", "neural"):
        if not isinstance(files.get(key), Mapping):
            raise PopulationEvidenceError(f"checkpoint lacks {key} receipt")
        items.append(files[key])
    trajectories = files.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise PopulationEvidenceError("checkpoint has no trajectory receipts")
    items.extend(trajectories)
    for item in items:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise PopulationEvidenceError("checkpoint file receipt escapes its directory")
        artifact = (root / relative).resolve()
        artifact.relative_to(root.resolve())
        if (
            not artifact.is_file()
            or artifact.stat().st_size != item.get("bytes")
            or sha256_file(artifact) != item.get("sha256")
        ):
            raise PopulationEvidenceError(f"checkpoint component differs: {artifact}")
    return receipt, receipt_path


def _checkpoints(
    run: Path, identity_sha256: str, expected_life_ids: set[str]
) -> list[tuple[dict[str, Any], Path]]:
    root = run / "checkpoints"
    if not root.exists():
        return []
    values = [
        _verify_checkpoint_directory(path, identity_sha256, expected_life_ids)
        for path in sorted(root.glob("step-*"))
        if path.is_dir()
    ]
    steps = [value[0].get("completed_steps") for value in values]
    if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps):
        raise PopulationEvidenceError("checkpoint step is invalid")
    if steps != sorted(set(steps)):
        raise PopulationEvidenceError("checkpoint steps are duplicated or unordered")
    return values


def _latest_checkpoint(
    run: Path,
    identity_sha256: str,
    checkpoints: Sequence[tuple[Mapping[str, Any], Path]],
) -> dict[str, Any] | None:
    path = run / "latest.json"
    if not checkpoints:
        if path.exists():
            raise PopulationEvidenceError("checkpoint pointer exists without a checkpoint")
        return None
    pointer = read_json(path)
    if (
        pointer.get("format") != "chreatures-population-checkpoint-pointer-v1"
        or pointer.get("evaluation_identity_sha256") != identity_sha256
    ):
        raise PopulationEvidenceError("latest checkpoint pointer identity differs")
    receipt, receipt_path = checkpoints[-1]
    expected_relative = str(receipt_path.parent.relative_to(run))
    if (
        pointer.get("checkpoint") != expected_relative
        or pointer.get("completed_steps") != receipt.get("completed_steps")
        or pointer.get("checkpoint_receipt_sha256") != sha256_file(receipt_path)
    ):
        raise PopulationEvidenceError("latest checkpoint pointer differs from checkpoint")
    return pointer


def _load_run(run: Path) -> dict[str, Any]:
    identity_path = run / "identity.json"
    identity = read_json(identity_path)
    identity_sha256 = _verified_identity(identity, "sha256", "evaluation identity")
    result_path = run / "result.json"
    failure_path = run / "failure.json"
    if result_path.is_file() == failure_path.is_file():
        raise PopulationEvidenceError(f"{run} must have exactly one result or failure")
    source_path = result_path if result_path.is_file() else failure_path
    source = read_json(source_path)
    content_sha256 = _verified_identity(source, "content_sha256", source_path.name)
    if source_path == failure_path:
        immutable_path = run / "failures" / f"{content_sha256}.json"
        if not immutable_path.is_file() or read_json(immutable_path) != source:
            raise PopulationEvidenceError("immutable evaluation failure receipt differs")
        source_path = immutable_path
    if source.get("evaluation_identity_sha256") != identity_sha256:
        raise PopulationEvidenceError("evaluation output belongs to another identity")
    if source_path == result_path:
        if source.get("format") != EVALUATION_FORMAT or source.get("status") != "completed":
            raise PopulationEvidenceError("completed evaluation output format differs")
        lives = source.get("lives")
        status = "success"
    else:
        if source.get("format") != FAILURE_FORMAT or source.get("status") != "failed":
            raise PopulationEvidenceError("failed evaluation output format differs")
        trace = source.get("traceback")
        if (
            not isinstance(trace, str)
            or hashlib.sha256(trace.encode()).hexdigest() != source.get("traceback_sha256")
        ):
            raise PopulationEvidenceError("failed evaluation traceback receipt differs")
        lives = source.get("candidate_failures")
        status = "failure"
    if not isinstance(lives, list) or not lives:
        raise PopulationEvidenceError("evaluation output has no whole-life records")
    identity_lives = identity.get("life_records")
    if not isinstance(identity_lives, list) or any(
        not isinstance(item, Mapping) for item in identity_lives
    ):
        raise PopulationEvidenceError("evaluation identity has no planned lives")
    for item in identity_lives:
        for field in (
            "life_id",
            "assignment_file_sha256",
            "candidate_sha256",
            "environment_sha256",
        ):
            _hash(item.get(field), f"planned life {field}")
    planned = {item.get("life_id"): item for item in identity_lives}
    if len(planned) != len(identity_lives):
        raise PopulationEvidenceError("evaluation identity repeats a life")
    for life in lives:
        if not isinstance(life, Mapping) or life.get("life_id") not in planned:
            raise PopulationEvidenceError("evaluation output contains an unplanned life")
        for key in LIFE_KEYS:
            if life.get(key) != planned[life["life_id"]].get(key):
                raise PopulationEvidenceError(
                    f"evaluation life {life['life_id']} changes planned field {key}"
                )
        if life.get("world_id") != life.get("environment_sha256"):
            raise PopulationEvidenceError("world_id is not the native environment identity")
        if status == "success":
            if not isinstance(life.get("trajectory_metrics"), Mapping):
                raise PopulationEvidenceError("completed life lacks trajectory metrics")
        else:
            required_failure = {
                "checkpoint_receipt_sha256",
                "cohort_trajectory_snapshot_sha256",
                "failure_trace_sha256",
            }
            if not required_failure.issubset(life):
                raise PopulationEvidenceError(
                    "failed life lacks reconstructible trajectory receipts"
                )
            for name in required_failure:
                if life[name] is not None:
                    _hash(life[name], f"failed life {name}")
    if {life["life_id"] for life in lives} != set(planned):
        raise PopulationEvidenceError(
            "evaluation output does not terminate every planned life"
        )
    committed_ticks = {life.get("committed_ticks") for life in lives}
    if len(committed_ticks) != 1 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in committed_ticks
    ):
        raise PopulationEvidenceError("cohort life committed ticks differ")
    if status == "success":
        completed_steps = next(iter(committed_ticks))
        if (
            source.get("completed_steps") != completed_steps
            or source.get("completed_resident_transitions")
            != completed_steps * len(lives)
        ):
            raise PopulationEvidenceError("completed cohort transition count differs")
    checkpoints = _checkpoints(run, identity_sha256, set(planned))
    latest = _latest_checkpoint(run, identity_sha256, checkpoints)
    reported_latest = source.get(
        "final_checkpoint" if status == "success" else "last_completed_checkpoint"
    )
    if latest is None:
        if reported_latest is not None:
            raise PopulationEvidenceError("evaluation reports an absent checkpoint")
    elif not isinstance(reported_latest, Mapping) or any(
        reported_latest.get(key) != latest.get(key)
        for key in (
            "format",
            "evaluation_identity_sha256",
            "completed_steps",
            "checkpoint",
            "checkpoint_receipt_sha256",
        )
    ):
        raise PopulationEvidenceError("evaluation terminal checkpoint differs from latest.json")
    return {
        "path": run,
        "identity": identity,
        "identity_path": identity_path,
        "identity_sha256": identity_sha256,
        "source": source,
        "source_path": source_path,
        "source_status": status,
        "lives": lives,
        "checkpoints": checkpoints,
        "latest": latest,
    }


def _life_trace(run: Mapping[str, Any], life: Mapping[str, Any]) -> dict[str, Any]:
    base = {key: life[key] for key in LIFE_KEYS}
    if run["source_status"] == "success":
        cohorts = run["source"].get("trajectory_cohorts")
        if not isinstance(cohorts, list):
            raise PopulationEvidenceError("completed run has no trajectory cohorts")
        cohort = next(
            (item for item in cohorts if item.get("world_slot") == life["world_slot"]),
            None,
        )
        if cohort is None:
            raise PopulationEvidenceError("completed life has no trajectory cohort")
        final_checkpoint = run["checkpoints"][-1][0]
        trajectories = final_checkpoint["files"]["trajectories"]
        world_slot = life["world_slot"]
        if (
            world_slot >= len(trajectories)
            or trajectories[world_slot].get("sha256") != cohort.get("snapshot_sha256")
        ):
            raise PopulationEvidenceError(
                "completed life trajectory differs from its final checkpoint"
            )
        value = {
            "life": base,
            "cohort_snapshot_sha256": cohort.get("snapshot_sha256"),
            "resident_metrics": life.get("trajectory_metrics"),
        }
    else:
        if life.get("failure_trace_sha256") != run["source"].get("traceback_sha256"):
            raise PopulationEvidenceError("failed life traceback identity differs")
        expected_checkpoint = (
            run["latest"].get("checkpoint_receipt_sha256")
            if run["latest"] is not None
            else None
        )
        if life.get("checkpoint_receipt_sha256") != expected_checkpoint:
            raise PopulationEvidenceError("failed life checkpoint receipt differs")
        value = {
            "life": base,
            "completed_steps": life.get("committed_ticks"),
            "checkpoint_receipt_sha256": life.get("checkpoint_receipt_sha256"),
            "cohort_trajectory_snapshot_sha256": life.get(
                "cohort_trajectory_snapshot_sha256"
            ),
            "failure_trace_sha256": life.get("failure_trace_sha256"),
        }
    if hashlib.sha256(population_canonical_bytes(value)).hexdigest() != life.get(
        "trajectory_sha256"
    ):
        raise PopulationEvidenceError(f"life {life['life_id']} trajectory identity differs")
    return value


def _foundation_records(
    state: Mapping[str, Any], campaign_path: Path, blob_dir: Path, campaign_id: str
) -> tuple[list[dict[str, Any]], str, str, str, dict[str, Any]]:
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise PopulationEvidenceError("native search state has no config")
    config_sha256 = _hash(state.get("config_sha256"), "search config")
    if hashlib.sha256(population_canonical_bytes(config)).hexdigest() != config_sha256:
        raise PopulationEvidenceError("native search config content differs")
    _config_path, config_blob = _write_blob(blob_dir, config, role="search_config")
    run_id = f"population-run:{campaign_id}"
    records = [
        evidence_record(
            id=run_id,
            time={"domain": "identity", "value": 0},
            record_type="population_run",
            text="Authenticated native population-search campaign.",
            blobs=[config_blob],
            fields={
                "campaign_id": campaign_id,
                "search_config_sha256": config_sha256,
                "search_state_format": state.get("format"),
            },
        )
    ]
    recipe = {
        "descriptor_version": state.get("descriptor_version"),
        "descriptor_axes": config.get("descriptor_axes"),
        "source_search_config_sha256": config_sha256,
    }
    _recipe_path, recipe_blob = _write_blob(blob_dir, recipe, role="descriptor_recipe")
    epoch_id = f"descriptor-epoch:{recipe_blob['sha256']}"
    epoch_record_id = epoch_id
    records.append(
        evidence_record(
            id=epoch_record_id,
            time={"domain": "identity", "value": 0},
            record_type="descriptor_epoch",
            text="Frozen physical descriptor recipe from native population search.",
            parents={run_id: "campaign"},
            blobs=[recipe_blob],
            fields={
                "descriptor_epoch_id": epoch_id,
                "descriptor_epoch_index": 0,
                "environment_epoch": int(config.get("environment_epoch", 0)),
                "descriptor_recipe_sha256": recipe_blob["sha256"],
                "descriptor_dimension": len(config.get("descriptor_axes", [])),
            },
        )
    )
    campaign_value, panel, policy_hashes = _campaign(
        campaign_path,
        config_sha256,
        _hash(config.get("environment_probe_panel_sha256"), "search probe panel"),
    )
    descriptor = campaign_value.get("descriptor")
    quality = campaign_value.get("quality")
    controller = campaign_value.get("controller")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("version") != "physical-population-descriptor-v1"
        or descriptor.get("spatial_cell_scale") != SPATIAL_CELL_SCALE
        or descriptor.get("axes") != config.get("descriptor_axes")
        or not isinstance(quality, Mapping)
        or quality.get("version") != "initial-regulation-quality-v1"
        or quality.get("terms") != config.get("quality_terms")
        or not isinstance(controller, Mapping)
        or controller.get("file_sha256") != panel.get("controller_file_sha256")
    ):
        raise PopulationEvidenceError("campaign metric or probe recipe differs")
    panel_sha256 = hashlib.sha256(population_canonical_bytes(panel)).hexdigest()
    _panel_blob_path, panel_blob = _write_blob(
        blob_dir, panel, role="probe_policy_panel"
    )
    panel_record_id = f"probe-panel:{panel_sha256}"
    records.append(
        evidence_record(
            id=panel_record_id,
            time={"domain": "identity", "value": 0},
            record_type="environment_probe_panel",
            text="Frozen environment probe-policy panel.",
            parents={run_id: "campaign", epoch_record_id: "descriptor_epoch"},
            blobs=[panel_blob],
            fields={
                "probe_panel_id": panel_record_id,
                "probe_panel_sha256": panel_sha256,
                "descriptor_epoch_id": epoch_id,
                "policy_artifact_sha256s": policy_hashes,
            },
        )
    )
    genomes = state.get("genomes")
    environments = state.get("environments")
    if not isinstance(genomes, Mapping) or not isinstance(environments, Mapping):
        raise PopulationEvidenceError("native search state lacks genomes or environments")
    for digest, genome in sorted(genomes.items()):
        if not isinstance(genome, Mapping) or genome.get("sha256") != digest:
            raise PopulationEvidenceError("native genome map identity differs")
        _path, artifact = _write_blob(blob_dir, genome, role="genome_artifact")
        records.append(
            genome_record_from_native(
                genome,
                campaign_record_id=run_id,
                time={"domain": "identity", "value": 0},
                artifact=artifact,
            )
        )
    for digest, environment in sorted(environments.items()):
        if not isinstance(environment, Mapping) or environment.get("sha256") != digest:
            raise PopulationEvidenceError("native environment map identity differs")
        if environment.get("epoch") != config.get("environment_epoch"):
            raise PopulationEvidenceError("native environment belongs to another epoch")
        _path, artifact = _write_blob(blob_dir, environment, role="environment_artifact")
        records.append(
            environment_record_from_native(
                environment,
                campaign_record_id=run_id,
                probe_panel_record_id=panel_record_id,
                probe_panel_sha256=panel_sha256,
                time={"domain": "identity", "value": 0},
                artifact=artifact,
            )
        )
    return records, run_id, epoch_record_id, panel_record_id, campaign_value


def record_population_campaign(
    *,
    search_state_path: Path,
    campaign_path: Path,
    evaluation_runs: Sequence[Path],
    ledger_path: Path,
    campaign_id: str,
    description: str,
    blob_dir: Path,
    batch_dir: Path,
) -> tuple[Path | None, dict[str, Any]]:
    """Build the next content-derived batch; return None when already recorded."""
    PopulationSearch(search_state_path).validate()
    state = read_json(search_state_path)
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        if (
            ledger.get("format") != LEDGER_FORMAT
            or ledger.get("campaign_id") != campaign_id
            or ledger.get("description") != description
        ):
            raise PopulationEvidenceError("existing population ledger belongs elsewhere")
    else:
        ledger = empty_ledger(campaign_id, description)
    loaded_runs = [_load_run(path) for path in sorted(set(evaluation_runs))]
    records, run_id, epoch_id, panel_id, campaign = _foundation_records(
        state, campaign_path, blob_dir, campaign_id
    )
    epoch_name = records[1]["fields"]["descriptor_epoch_id"]
    panel_sha256 = records[2]["fields"]["probe_panel_sha256"]
    evaluations = state.get("evaluations")
    if not isinstance(evaluations, list):
        raise PopulationEvidenceError("native search state lacks evaluations")
    if any(not isinstance(item, Mapping) for item in evaluations):
        raise PopulationEvidenceError("native search state has a non-object evaluation")
    evaluation_by_life = {item.get("life_id"): item for item in evaluations}
    if len(evaluation_by_life) != len(evaluations):
        raise PopulationEvidenceError("native search state repeats a life evaluation")
    existing_evaluations = {
        record["fields"]["evaluation_id"]
        for record in ledger["records"]
        if record.get("record_type") in {"evaluation_completed", "evaluation_failed"}
    }
    supplied_evaluations: set[str] = set()
    supplied_lives: set[str] = set()

    for run in loaded_runs:
        source_blob = _source_blob(run["source_path"], "physical_evaluation_output")
        checkpoints = run["checkpoints"]
        for life in run["lives"]:
            if life["life_id"] in supplied_lives:
                raise PopulationEvidenceError(
                    f"life {life['life_id']} occurs in more than one evaluation output"
                )
            supplied_lives.add(life["life_id"])
            native = evaluation_by_life.get(life["life_id"])
            if native is None:
                raise PopulationEvidenceError(
                    f"life {life['life_id']} has not been ingested by native search"
                )
            expected_native_status = (
                life.get("status")
                if run["source_status"] == "success"
                else "infrastructure-failure"
            )
            if expected_native_status not in {
                "completed",
                "organism-terminal",
                "infrastructure-failure",
            }:
                raise PopulationEvidenceError("evaluator life has invalid terminal status")
            if (
                native.get("candidate_sha256") != life.get("candidate_sha256")
                or native.get("environment_sha256") != life.get("environment_sha256")
                or native.get("status") != expected_native_status
                or native.get("committed_ticks") != life.get("committed_ticks")
                or native.get("trajectory_sha256") != life.get("trajectory_sha256")
            ):
                raise PopulationEvidenceError(
                    f"native evaluation for life {life['life_id']} differs from evaluator output"
                )
            if native.get("evaluation_seed") != life.get("evaluation_seed"):
                raise PopulationEvidenceError(
                    f"native evaluation seed for life {life['life_id']} differs"
                )
            supplied_evaluations.add(native["evaluation_sha256"])
            if run["latest"] is not None:
                latest_step = run["latest"]["completed_steps"]
                if latest_step > native["committed_ticks"] or (
                    native["status"] in {"completed", "organism-terminal"}
                    and latest_step != native["committed_ticks"]
                ):
                    raise PopulationEvidenceError(
                        f"terminal checkpoint for life {life['life_id']} is inconsistent"
                    )
            if life["environment_sha256"] not in state["environments"]:
                raise PopulationEvidenceError("evaluator environment is absent from search state")
            if run["source_status"] == "success":
                source_environment = run["source"].get("environments", {}).get(
                    life["environment_sha256"]
                )
                if source_environment != state["environments"][life["environment_sha256"]]:
                    raise PopulationEvidenceError("evaluator environment artifact differs")
                environment_metadata = campaign["environment_index"].get(
                    life["environment_sha256"]
                )
                if not isinstance(environment_metadata, Mapping):
                    raise PopulationEvidenceError(
                        "evaluator environment is absent from campaign manifest"
                    )
                if native.get("metrics") != _metric_row(life, environment_metadata):
                    raise PopulationEvidenceError(
                        f"native metrics for life {life['life_id']} differ from evaluator"
                    )
            elif native.get("metrics") != {}:
                raise PopulationEvidenceError("failed native evaluation has metrics")

            trace = _life_trace(run, life)
            _trace_path, trace_blob = _write_blob(
                blob_dir,
                trace,
                role="evaluation_trace",
                population_canonical=True,
            )
            if trace_blob["sha256"] != life["trajectory_sha256"]:
                raise PopulationEvidenceError("materialized life trace hash differs")
            _native_path, native_blob = _write_blob(
                blob_dir, native, role="evaluation_result"
            )
            life_checkpoints = [
                (receipt, path)
                for receipt, path in checkpoints
                if life["life_id"] in receipt.get("life_ids", [])
                and receipt["completed_steps"] <= native["committed_ticks"]
            ]
            allocated = any(receipt["completed_steps"] == 0 for receipt, _ in life_checkpoints)
            if native["status"] in {"completed", "organism-terminal"} and not allocated:
                raise PopulationEvidenceError("completed life has no step-zero allocation proof")
            if native["committed_ticks"] > 0 and not allocated:
                raise PopulationEvidenceError("advanced life has no step-zero allocation proof")
            continuation: str | None = None
            if allocated:
                birth_id = f"birth:{life['life_id']}"
                records.append(
                    evidence_record(
                        id=birth_id,
                        time={"domain": "evaluation_tick", "value": 0},
                        record_type="birth",
                        text="Fresh experimental evaluation life allocated in a physical world.",
                        parents={
                            f"genome:{life['candidate_sha256']}": "candidate_genome",
                            f"environment:{life['environment_sha256']}": "environment",
                        },
                        fields={
                            "life_id": life["life_id"],
                            "birth_mode": "experimental_initialization",
                            "genome_sha256": life["candidate_sha256"],
                            "environment_sha256": life["environment_sha256"],
                            "evaluation_identity_sha256": run["identity_sha256"],
                        },
                    )
                )
                continuation = birth_id
                for checkpoint, checkpoint_path in life_checkpoints:
                    checkpoint_sha256 = sha256_file(checkpoint_path)
                    checkpoint_id = (
                        f"life-checkpoint:{life['life_id']}:"
                        f"{checkpoint['completed_steps']}:{checkpoint_sha256}"
                    )
                    records.append(
                        evidence_record(
                            id=checkpoint_id,
                            time={
                                "domain": "evaluation_tick",
                                "value": checkpoint["completed_steps"],
                            },
                            record_type="life_checkpoint",
                            text="Authenticated coupled physical/neural life checkpoint.",
                            parents={continuation: "life_continuation"},
                            blobs=[_source_blob(checkpoint_path, "life_checkpoint")],
                            fields={
                                "life_id": life["life_id"],
                                "checkpoint_sha256": checkpoint_sha256,
                                "tick": checkpoint["completed_steps"],
                                "evaluation_identity_sha256": run["identity_sha256"],
                            },
                        )
                    )
                    continuation = checkpoint_id
            terminal, decision = evaluation_records_from_native(
                native,
                life_id=life["life_id"],
                continuation_record_id=continuation,
                campaign_record_id=None if allocated else run_id,
                allocated=allocated,
                descriptor_epoch_record_id=epoch_id,
                descriptor_epoch_id=epoch_name,
                probe_panel_record_id=panel_id,
                probe_panel_sha256=panel_sha256,
                time={
                    "domain": "evaluation_tick",
                    "value": native["committed_ticks"],
                },
                result_artifact=native_blob,
                trace_artifact=trace_blob,
            )
            terminal["blob_refs"].append(source_blob)
            terminal["fields"].update(
                evaluation_identity_sha256=run["identity_sha256"],
                physical_output_sha256=source_blob["sha256"],
            )
            records.extend((terminal, decision))

    state_evaluations = {item["evaluation_sha256"] for item in evaluations}
    missing_evaluations = state_evaluations - existing_evaluations - supplied_evaluations
    if missing_evaluations:
        raise PopulationEvidenceError(
            "native terminal evaluations lack authenticated evaluator outputs: "
            + ", ".join(sorted(missing_evaluations))
        )

    state_blob = _source_blob(search_state_path, "population_search_state")
    state_id = f"population-snapshot:{state_blob['sha256']}"
    decision_ids = [f"archive-decision:{item['evaluation_sha256']}" for item in evaluations]
    records.append(
        evidence_record(
            id=state_id,
            time={"domain": "search_ask_count", "value": int(state.get("ask_count", 0))},
            record_type="population_snapshot",
            text="Authenticated native population search snapshot.",
            parents={run_id: "campaign"}
            | {record_id: "archive_decision" for record_id in decision_ids},
            blobs=[state_blob],
            fields={
                "ask_count": int(state.get("ask_count", 0)),
                "state_sha256": state_blob["sha256"],
                "evaluation_count": len(evaluations),
                "infrastructure_failure_count": sum(
                    item.get("status") == "infrastructure-failure"
                    for item in evaluations
                ),
                "organism_terminal_count": sum(
                    item.get("status") == "organism-terminal" for item in evaluations
                ),
            },
        )
    )

    existing = {record["id"]: record for record in ledger["records"]}
    generated: dict[str, dict[str, Any]] = {}
    for record in records:
        previous = generated.get(record["id"])
        if previous is not None and canonical_bytes(previous) != canonical_bytes(record):
            raise PopulationEvidenceError(
                f"conversion generated conflicting record {record['id']}"
            )
        generated[record["id"]] = record
    new_records = []
    for record in generated.values():
        previous = existing.get(record["id"])
        if previous is None:
            new_records.append(record)
        elif canonical_bytes(previous) != canonical_bytes(record):
            raise PopulationEvidenceError(
                f"stable population record {record['id']} changed across conversion"
            )
    combined = [*ledger["records"], *new_records]
    validate_records(combined, campaign_id=campaign_id)
    if not new_records:
        return None, {"status": "unchanged", "record_count": len(combined)}
    batch_hash = hashlib.sha256(canonical_bytes(new_records)).hexdigest()
    batch = {
        "format": BATCH_FORMAT,
        "schema_version": 1,
        "campaign_id": campaign_id,
        "batch_id": f"population-campaign:{batch_hash}",
        "records": new_records,
        "sources": {
            "search_state_sha256": sha256_file(search_state_path),
            "campaign_manifest_file_sha256": sha256_file(campaign_path),
            "probe_panel_content_sha256": panel_sha256,
            "evaluation_output_sha256s": sorted(
                sha256_file(run["source_path"]) for run in loaded_runs
            ),
        },
    }
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / f"{batch_hash}.json"
    if batch_path.exists():
        if read_json(batch_path) != batch:
            raise PopulationEvidenceError("content-addressed campaign batch differs")
    else:
        atomic_write_json(batch_path, batch)
    return batch_path, {
        "status": "new_batch",
        "batch_id": batch["batch_id"],
        "new_records": len(new_records),
        "combined_records": len(combined),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-state", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--evaluation-run", type=Path, action="append", default=[])
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--blob-dir", type=Path)
    parser.add_argument("--batch-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    blob_dir = args.blob_dir or args.ledger.parent / "evidence-blobs"
    batch_dir = args.batch_dir or args.ledger.parent / "evidence-batches"
    batch_path, receipt = record_population_campaign(
        search_state_path=args.search_state,
        campaign_path=args.campaign,
        evaluation_runs=args.evaluation_run,
        ledger_path=args.ledger,
        campaign_id=args.campaign_id,
        description=args.description,
        blob_dir=blob_dir,
        batch_dir=batch_dir,
    )
    command = [
        sys.executable,
        "-m",
        "scripts.build_population_weave",
        "--ledger",
        str(args.ledger),
        "--campaign-id",
        args.campaign_id,
        "--description",
        args.description,
        "--population-state",
        str(args.search_state),
    ]
    if batch_path is not None:
        command.extend(("--batch", str(batch_path)))
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    built = json.loads(completed.stdout)
    print(
        json.dumps(
            {
                "format": "chreatures-population-campaign-record-receipt-v1",
                **receipt,
                "batch": str(batch_path) if batch_path is not None else None,
                "native_weave": {
                    "path": built["artifacts"]["weave"]["path"],
                    "sha256": built["artifacts"]["weave"]["sha256"],
                    "nodes": built["native"]["node_count"],
                    "edges": built["native"]["edge_count"],
                    "reload_equal": built["native"]["reload_equal"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
