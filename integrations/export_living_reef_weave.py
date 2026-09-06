#!/usr/bin/env python3
"""Export one authenticated living-reef biography through Universal Weave.

The checkpoint remains private.  The exported graph contains its content hash,
bounded public summaries, and sparse intervals derived from its journal tail.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEAVE_DIR = ROOT / "integrations" / "weave"
DEFAULT_OUTPUT = ROOT / "integrations" / "artifacts" / "living-reef-biography"
CHECKPOINT_FORMAT = "chreatures-developmental-habitat-checkpoint-v1"
RESIDENT_FORMAT = "chreatures-native-developmental-resident-rich-v1"
PREDICTOR_FORMAT = "chreatures-rich-consequence-ensemble-v1"
PREDICTOR_RECEIPT_FORMAT = "chreatures-rich-consequence-fit-report-v1"
LAW_FORMAT = "chreatures-gam-consequence-law-bank-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def npz_metadata(path: Path, expected_format: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if "metadata" not in payload:
            raise ValueError(f"{path} has no metadata")
        value = json.loads(str(payload["metadata"].item()))
    if value.get("format") != expected_format:
        raise ValueError(f"{path} has format {value.get('format')!r}, expected {expected_format}")
    return value


def blob(
    digest: str,
    role: str,
    media_type: str,
    *,
    size: int | None = None,
    verification: str,
) -> dict[str, Any]:
    if not HEX64.fullmatch(digest):
        raise ValueError(f"{role} has invalid SHA-256")
    result: dict[str, Any] = {
        "role": role,
        "uri": f"urn:sha256:{digest}",
        "sha256": digest,
        "media_type": media_type,
        "verification": verification,
    }
    if size is not None:
        result["bytes"] = size
    return result


def local_blob(path: Path, role: str, media_type: str) -> dict[str, Any]:
    return blob(
        sha256(path),
        role,
        media_type,
        size=path.stat().st_size,
        verification="verified_local_sha256",
    )


def record(
    record_id: str,
    time: Any,
    record_type: str,
    text: str,
    parents: list[str],
    *,
    blobs: list[dict[str, Any]] | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "time": time,
        "record_type": record_type,
        "text": text,
        "parent_ids": parents,
        "blob_refs": blobs or [],
        "fields": fields or {},
    }


def capture_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Read once: the live process replaces this file periodically.  This binds
    # every summary and the file hash below to one coherent byte sequence.
    data = path.read_bytes()
    envelope = json.loads(data)
    if envelope.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint has the wrong format")
    state = envelope.get("state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no state")
    state_sha = digest_bytes(canonical(state))
    if state_sha != envelope.get("sha256"):
        raise ValueError("checkpoint canonical state hash differs from its envelope")
    checkpoint_blob = blob(
        digest_bytes(data),
        "private_living_reef_checkpoint",
        "application/json",
        size=len(data),
        verification="verified_from_single_coherent_read_not_published",
    )
    return envelope, state, checkpoint_blob


def journal_tick(item: dict[str, Any], habitat_id: str) -> int:
    match = re.fullmatch(re.escape(habitat_id) + r":([0-9]+):([0-9]+)", str(item.get("id")))
    if match is None:
        raise ValueError(f"journal event has an unrecognized host id: {item.get('id')!r}")
    return int(match.group(1))


def interval_records(
    state: dict[str, Any], parent: str
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    habitat_id = str(state["id"])
    raw = state.get("journal", [])
    if not isinstance(raw, list) or not raw:
        return [], parent, {"events": 0, "intervals": 0}
    events: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("journal contains a non-object")
        events.append((journal_tick(item, habitat_id), index, item))
    events.sort(key=lambda value: (value[0], value[1]))
    ticks = sorted({tick for tick, _, _ in events})
    groups = min(4, len(ticks))
    boundaries = [len(ticks) * index // groups for index in range(groups + 1)]
    records: list[dict[str, Any]] = []
    previous = parent
    for group in range(groups):
        selected_ticks = set(ticks[boundaries[group] : boundaries[group + 1]])
        selected = [item for tick, _, item in events if tick in selected_ticks]
        selected_bytes = canonical(selected)
        selected_sha = digest_bytes(selected_bytes)
        event_ticks = [journal_tick(item, habitat_id) for item in selected]
        resident_counts = Counter(str(item.get("resident", "unspecified")) for item in selected)
        kind_counts = Counter(str(item.get("kind", "episode")) for item in selected)
        unique_ids = len({str(item["id"]) for item in selected})
        interval_id = (
            f"experienced-interval:{habitat_id}:{min(event_ticks)}-{max(event_ticks)}:"
            f"{selected_sha}"
        )
        records.append(
            record(
                interval_id,
                {
                    "domain": "model_tick_interval",
                    "start": min(event_ticks),
                    "end": max(event_ticks),
                },
                "experienced_interval",
                (
                    f"Authenticated checkpoint journal tail records {len(selected)} actual "
                    f"resident event observations in this interval."
                ),
                [previous],
                fields={
                    "evidence_scope": "actual_experience",
                    "source": "bounded_checkpoint_journal_tail",
                    "source_event_sha256": selected_sha,
                    "source_event_count": len(selected),
                    "source_host_id_unique_count": unique_ids,
                    "source_host_id_reuse_count": len(selected) - unique_ids,
                    "event_kind_counts": dict(sorted(kind_counts.items())),
                    "resident_event_counts": dict(sorted(resident_counts.items())),
                    "start_time_seconds": min(float(item["time"]) for item in selected),
                    "end_time_seconds": max(float(item["time"]) for item in selected),
                    "raw_events_published": False,
                },
            )
        )
        previous = interval_id
    return records, previous, {
        "events": len(events),
        "intervals": len(records),
        "unique_host_ids": len({str(item["id"]) for _, _, item in events}),
        "host_id_reuses": len(events) - len({str(item["id"]) for _, _, item in events}),
    }


def gam_records(
    law_path: Path,
    fit_path: Path,
    native_evaluation_path: Path,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    laws = read_json(law_path)
    fit = read_json(fit_path)
    native = read_json(native_evaluation_path)
    law_blob = local_blob(law_path, "native_gam_law_bank", "application/json")
    fit_blob = local_blob(fit_path, "gam_fit_report", "application/json")
    native_blob = local_blob(native_evaluation_path, "gam_native_evaluation", "application/json")
    if laws.get("schema") != LAW_FORMAT:
        raise ValueError("GAM law bank has the wrong schema")
    if fit.get("artifact", {}).get("sha256") != law_blob["sha256"]:
        raise ValueError("GAM fit report does not authenticate the law bank")
    if native.get("artifact_sha256") != law_blob["sha256"]:
        raise ValueError("GAM native evaluation used another law bank")
    if "executed native Rust" not in str(native.get("status")):
        raise ValueError("GAM native evaluation is not an executed native receipt")
    telemetry = list(laws.get("source", {}).get("telemetry_sha256", []))
    source_digest = digest_bytes(canonical(telemetry))
    source_id = f"gam-source:{source_digest}"
    records = [
        record(
            source_id,
            {"domain": "research_collection", "value": "sensorimotor-play-v1"},
            "verified_source",
            "Two hash-pinned exploratory telemetry packets supplied the descriptive GAM fits.",
            [],
            fields={
                "evidence_scope": "research_fit_source",
                "telemetry_sha256": telemetry,
                "model_library": laws["source"]["model_library"],
                "model_version": laws["source"]["model_version"],
                "model_source_commit": laws["source"]["model_source_commit"],
            },
        )
    ]
    fit_ids: list[str] = []
    metrics = fit.get("metrics", {})
    for law in laws.get("laws", []):
        name = str(law["name"])
        fit_id = f"gam-law-fit:{name}:{law_blob['sha256']}"
        fit_ids.append(fit_id)
        records.append(
            record(
                fit_id,
                {"domain": "research_fit", "value": name},
                "gam_law_fit",
                f"Native GAM fit for {name}; conditional prediction with no causal claim.",
                [source_id],
                fields={
                    "evidence_scope": "research_fit",
                    "metrics": metrics.get(name),
                    "conservative_residual_bound": law.get("conservative_residual_bound"),
                    "term_count": len(law.get("terms", [])),
                },
            )
        )
    bank_id = f"gam-law-bank:{law_blob['sha256']}"
    records.append(
        record(
            bank_id,
            {"domain": "research_fit", "value": "law-bank"},
            "gam_law_bank",
            "Three descriptive GAM laws were exported and executed by the native Rust evaluator.",
            fit_ids,
            blobs=[law_blob, fit_blob, native_blob],
            fields={
                "evidence_scope": "inherited_research_law",
                "law_names": [law["name"] for law in laws["laws"]],
                "fit_status": fit.get("status"),
                "native_evaluation_status": native.get("status"),
            },
        )
    )
    return records, bank_id, {"law_bank": law_blob, "fit_report": fit_blob, "native": native_blob}


def predictor_records(
    artifact_path: Path,
    receipt_path: Path,
    reference_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = npz_metadata(artifact_path, PREDICTOR_FORMAT)
    receipt = read_json(receipt_path)
    if receipt.get("format") != PREDICTOR_RECEIPT_FORMAT:
        raise ValueError("predictor receipt has the wrong format")
    artifact_blob = local_blob(artifact_path, "rich_consequence_predictor", "application/x-npz")
    receipt_blob = local_blob(receipt_path, "rich_consequence_fit_receipt", "application/json")
    reference_blob = local_blob(reference_path, "predictor_float32_reference", "application/x-npz")
    receipt_artifact = receipt.get("artifact", {})
    if receipt_artifact.get("sha256") != artifact_blob["sha256"]:
        raise ValueError("predictor receipt does not authenticate the artifact file")
    if receipt_artifact.get("artifact_identity") != metadata.get("artifact_identity"):
        raise ValueError("predictor receipt and metadata identities differ")
    source = metadata["source"]
    dataset_identity = source["dataset_manifest_content_sha256"]
    dataset_id = f"rich-prediction-data:{dataset_identity}"
    front_id = f"frozen-rich-front:{source['copied_encoder_sha256']}"
    fit_id = f"forecast-fit:{metadata['artifact_identity']}"
    proposal_id = f"proposed-organ:{metadata['artifact_identity']}"
    records = [
        record(
            dataset_id,
            {"domain": "research_collection", "value": "rich-v2-seed20260912"},
            "verified_source",
            "Hash-verified rich sensorimotor play supplied the predictor fit rows.",
            [],
            fields={
                "evidence_scope": "forecast_fit_source",
                "dataset_manifest_content_sha256": dataset_identity,
                "dataset_file_sha256": {
                    (
                        "research/sensorimotor_skills/trajectory-schema-rich-v2.json"
                        if Path(name).name == "trajectory-schema-rich-v2.json"
                        else Path(name).name
                    ): digest
                    for name, digest in source["dataset_files"].items()
                },
            },
        ),
        record(
            front_id,
            {"domain": "research_artifact", "value": "frozen-rich-front"},
            "frozen_representation",
            "The predictor copied and froze the bootstrap visual, body, and goal encoders.",
            [dataset_id],
            blobs=[
                blob(
                    source["bootstrap_file_sha256"],
                    "rich_sensorimotor_bootstrap",
                    "application/x-pytorch",
                    verification="reported_by_verified_predictor_artifact",
                )
            ],
            fields={
                "evidence_scope": "forecast_fit_source",
                "bootstrap_identity_sha256": source["bootstrap_identity_sha256"],
                "copied_encoder_sha256": source["copied_encoder_sha256"],
                "frame_encoder_sha256": source["frame_encoder_sha256"],
                "goal_encoder_sha256": source["goal_encoder_sha256"],
                "representation_saw_all_four_worlds": True,
            },
        ),
        record(
            fit_id,
            {"domain": "research_fit", "value": "rich-consequence-v1"},
            "forecast_fit",
            "Three action-conditioned consequence members were fit on worlds 0–2 and described on world 3.",
            [dataset_id, front_id],
            blobs=[artifact_blob, receipt_blob, reference_blob],
            fields={
                "evidence_scope": "forecast_fit",
                "artifact_identity": metadata["artifact_identity"],
                "training": metadata["training"],
                "validation_scaled_rmse": receipt["metrics"]["validation"]["all_outputs_train_scale_rmse"],
                "zero_delta_scaled_rmse": receipt["metrics"]["validation_zero_delta"]["all_outputs_train_scale_rmse"],
                "action_permuted_scaled_rmse": receipt["metrics"]["validation_action_permuted"]["all_outputs_train_scale_rmse"],
                "goal_space_rms": receipt["goal_space_calibration"]["overall_rms"],
                "goal_space_zero_delta_rms": receipt["goal_space_calibration"]["zero_delta_overall_rms"],
                "fit_seconds": receipt["elapsed_seconds"],
                "status": receipt["status"],
            },
        ),
        record(
            proposal_id,
            {"domain": "research_design", "value": "next-native-resident"},
            "proposed_organ",
            "The fitted consequence predictor is a proposed organ for a later resident generation.",
            [fit_id],
            fields={
                "evidence_scope": "proposed_future_organ",
                "installed_in_living_reef_checkpoint": False,
                "current_life_modified": False,
                "causal_claim": False,
                "uncertainty_claim": False,
            },
        ),
    ]
    return records, {"artifact": artifact_blob, "receipt": receipt_blob, "reference": reference_blob}


def build_request(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope, state, checkpoint_blob = capture_checkpoint(args.checkpoint)
    resident = npz_metadata(args.resident_artifact, RESIDENT_FORMAT)
    resident_blob = local_blob(args.resident_artifact, "native_update20_resident", "application/x-npz")
    deployment = read_json(args.deployment_manifest)
    deployment_blob = local_blob(args.deployment_manifest, "frozen_deployment_manifest", "application/json")
    if args.checkpoint.parent.parent != args.deployment_manifest.parent:
        raise ValueError("checkpoint and deployment manifest are not in the same frozen deployment")
    model_identity = state.get("resident_controller", {}).get("model_identity", {})
    if model_identity.get("file_sha256") != resident_blob["sha256"]:
        raise ValueError("checkpoint resident file hash differs from update-20 artifact")
    if model_identity.get("artifact_sha256") != resident.get("artifact_sha256"):
        raise ValueError("checkpoint resident internal identity differs from update-20 artifact")
    if deployment.get("files", {}).get("models/resident.npz") != resident_blob["sha256"]:
        raise ValueError("deployment manifest resident hash differs from update-20 artifact")
    if deployment.get("source_revision") != "215647bde626994af3b18fd6adc97a5d0c72574b":
        raise ValueError("unexpected frozen deployment revision")

    gam, law_bank_id, gam_summary = gam_records(
        args.gam_law_bank, args.gam_fit_report, args.gam_native_evaluation
    )
    consequence = resident.get("consequence_refinement", {})
    if consequence.get("law_file_sha256") != gam_summary["law_bank"]["sha256"]:
        raise ValueError("update-20 resident does not embed this GAM law bank")
    if consequence.get("law_content_sha256") != digest_bytes(
        canonical(read_json(args.gam_law_bank))
    ):
        raise ValueError("update-20 resident GAM law content identity differs")

    checkpoint = resident.get("checkpoint", {})
    training_sha = str(checkpoint.get("sha256", ""))
    training_id = f"online-update20:{training_sha}"
    resident_id = f"native-resident:{resident['artifact_sha256']}"
    deployment_id = f"deployment:{deployment['source_revision']}:{deployment_blob['sha256']}"
    life_id = f"research-birth:{state['id']}:{deployment_blob['sha256']}"
    evidence: list[dict[str, Any]] = gam + [
        record(
            training_id,
            {"domain": "online_update", "value": checkpoint["updates"]},
            "developmental_milestone",
            "Rich online development reached the update-20 checkpoint used for this native export.",
            [],
            blobs=[
                blob(
                    training_sha,
                    "rich_online_update20_checkpoint",
                    "application/x-pytorch",
                    verification="reported_by_verified_native_export",
                )
            ],
            fields={
                "evidence_scope": "model_provenance",
                "format": checkpoint["format"],
                "updates": checkpoint["updates"],
                "physical_steps": checkpoint["physical_steps"],
                "source_locator": (
                    f"{Path(checkpoint['path']).parent.name}/{Path(checkpoint['path']).name} "
                    "(private training archive)"
                ),
            },
        ),
        record(
            resident_id,
            {"domain": "online_update", "value": checkpoint["updates"]},
            "model_export",
            "Update-20 worker, achieved-goal manager, and inherited GAM laws were exported for native execution.",
            [training_id, law_bank_id],
            blobs=[resident_blob],
            fields={
                "evidence_scope": "installed_model_provenance",
                "artifact_identity": resident["artifact_sha256"],
                "format": resident["format"],
                "execution": resident["execution"],
                "embedded_gam_law_file_sha256": consequence["law_file_sha256"],
                "private_gam_learning_supported": True,
            },
        ),
        record(
            deployment_id,
            {"domain": "unix_time", "value": deployment["created_unix"]},
            "model_promotion",
            "The frozen 215647b deployment pinned the update-20 resident and native engines for a new life.",
            [resident_id],
            blobs=[deployment_blob],
            fields={
                "evidence_scope": "installed_release",
                "source_revision": deployment["source_revision"],
                "source_archive_sha256": deployment["source_archive_sha256"],
                "native_file_sha256": {
                    key: value for key, value in deployment["files"].items() if key.startswith("_")
                },
                "deployment_semantics": deployment["semantics"],
            },
        ),
        record(
            life_id,
            {"domain": "model_tick", "value": 0},
            "research_birth",
            "Six residents began a new research life from the frozen deployment.",
            [deployment_id],
            fields={
                "evidence_scope": "actual_life",
                "habitat_id": state["id"],
                "resident_names": sorted(state.get("cognition_state", {})),
                "birth_attestation": "frozen deployment semantics plus authenticated later checkpoint",
                "raw_birth_checkpoint_available": False,
            },
        ),
    ]

    intervals, last_experience, interval_summary = interval_records(state, life_id)
    evidence.extend(intervals)
    goals: list[dict[str, Any]] = []
    goal_ids: list[str] = []
    for resident_name, cognition in sorted(state.get("cognition_state", {}).items()):
        goal = cognition.get("goal", {}) if isinstance(cognition, dict) else {}
        if not goal.get("valid"):
            continue
        goal_id = (
            f"achieved-goal-current:{state['id']}:{resident_name}:"
            f"{state['tick']}:{goal['slot']}:{goal['recorded_tick']}"
        )
        goal_ids.append(goal_id)
        goals.append(
            record(
                goal_id,
                {"domain": "model_tick", "value": state["tick"]},
                "achieved_goal_selection",
                f"{resident_name} held a private achieved-history goal at the captured checkpoint.",
                [last_experience, resident_id],
                fields={
                    "evidence_scope": "actual_private_state_summary",
                    "resident": resident_name,
                    "selected_slot": goal["slot"],
                    "achieved_history_recorded_tick": goal["recorded_tick"],
                    "achieved_history_recorded_time_seconds": goal["recorded_time"],
                    "remaining_commitment_ticks": goal["remaining_ticks"],
                    "selection_observed_at_checkpoint_tick": state["tick"],
                    "private_key_window_or_rng_published": False,
                },
            )
        )
    evidence.extend(goals)

    bodies = state.get("world", {}).get("bodies", [])
    body_summary = [
        {
            "resident": body["id"],
            "age_seconds": body["age"],
            "energy": body["energy"],
            "fatigue": body["fatigue"],
            "gut": body["gut"],
            "history_samples": len(state.get("history", {}).get(body["id"], [])),
            "achieved_memory_count": state.get("cognition_state", {}).get(body["id"], {}).get("memory_count"),
        }
        for body in bodies
    ]
    personal = state.get("resident_controller", {}).get("native", {}).get("personal_consequences")
    personal = json.loads(personal) if isinstance(personal, str) else personal
    personal_summary = []
    if isinstance(personal, dict):
        names = [body["id"] for body in bodies]
        individuals = personal.get("individuals", [])
        if len(individuals) != len(names):
            raise ValueError("private GAM individual count differs from resident batch")
        personal_summary = [
            {
                "resident": name,
                "updates": individual.get("updates"),
                "out_of_domain_observations": individual.get("out_of_domain"),
                "last_completed_tick": individual.get("last_completed_tick"),
            }
            for name, individual in zip(names, individuals, strict=True)
        ]
    neural = state.get("neural_snapshot", {})
    neural_blob = blob(
        neural["sha256"],
        "private_neural_state",
        "application/octet-stream",
        size=int(neural["bytes"]),
        verification="reported_by_authenticated_checkpoint",
    )
    snapshot_id = f"snapshot:{state['id']}:{state['tick']}:{envelope['sha256']}"
    snapshot_parents = list(dict.fromkeys([last_experience, *goal_ids]))
    evidence.append(
        record(
            snapshot_id,
            {"domain": "model_tick", "value": state["tick"]},
            "snapshot",
            f"Authenticated living-reef checkpoint captured at model tick {state['tick']}.",
            snapshot_parents,
            blobs=[checkpoint_blob, neural_blob],
            fields={
                "evidence_scope": "actual_life_snapshot",
                "habitat_id": state["id"],
                "checkpoint_state_sha256": envelope["sha256"],
                "paused": state["paused"],
                "branch": state["branch"],
                "resident_count": len(body_summary),
                "resident_state_summary": body_summary,
                "achieved_goal_selection_count": len(goal_ids),
                "goal_memory_capacity": state["resident_controller"]["native"]["goal_memory"]["capacity"],
                "goal_observation_dimension": state["resident_controller"]["native"]["goal_memory"]["observation_dim"],
                "private_goal_keys_windows_rng_published": False,
                "private_gam_adaptation_summary": personal_summary,
                "gam_private_update_claim": "none observed in this checkpoint; all recorded candidates were out of inherited fit domain",
                "biosphere": {
                    "format": state.get("biosphere", {}).get("format"),
                    "config_sha256": state.get("biosphere", {}).get("config_sha256"),
                    "part_count": len(state.get("biosphere", {}).get("parts", {})),
                },
                "engine_identity_sha256": state.get("engine_identity", {}).get("sha256"),
                "checkpoint_file_published": False,
            },
        )
    )

    proposed, predictor_summary = predictor_records(
        args.predictor_artifact, args.predictor_receipt, args.predictor_reference
    )
    evidence.extend(proposed)
    archive_id = f"living-reef-biography:{state['id']}:{envelope['sha256']}"
    request = {
        "archive_id": archive_id,
        "habitat_id": state["id"],
        "description": (
            "External public biography of an authenticated living-reef snapshot plus "
            "separate research-fit provenance; never organism memory or controller input."
        ),
        "journal": [],
        "evidence": evidence,
    }
    summary = {
        "archive_id": archive_id,
        "source_commit": args.source_commit,
        "actual_life": {
            "deployment": deployment_blob,
            "resident_artifact": resident_blob,
            "checkpoint": checkpoint_blob,
            "checkpoint_state_sha256": envelope["sha256"],
            "checkpoint_tick": state["tick"],
            "journal": interval_summary,
            "resident_count": len(body_summary),
            "goal_selection_count": len(goal_ids),
            "private_checkpoint_published": False,
        },
        "gam": gam_summary,
        "proposed_predictor": {**predictor_summary, "installed_in_checkpoint": False},
    }
    return request, summary


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--resident-artifact", type=Path, default=ROOT / "data/genomes/developmental-resident-rich-update20.npz")
    parser.add_argument("--gam-law-bank", type=Path, default=ROOT / "integrations/gam_mechanisms/artifacts/body_consequence_laws.json")
    parser.add_argument("--gam-fit-report", type=Path, default=ROOT / "integrations/gam_mechanisms/artifacts/fit_report.json")
    parser.add_argument("--gam-native-evaluation", type=Path, default=ROOT / "integrations/gam_mechanisms/artifacts/native_evaluation.json")
    parser.add_argument("--predictor-artifact", type=Path, default=ROOT / "data/genomes/rich-consequence-ensemble-v1.npz")
    parser.add_argument("--predictor-receipt", type=Path, default=ROOT / "data/training/rich-consequence-v1.receipt.json")
    parser.add_argument("--predictor-reference", type=Path, default=ROOT / "data/training/rich-consequence-v1.native-reference.npz")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be a full lowercase Git commit")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request, sources = build_request(args)
    request_path = args.output_dir / "living-reef-biography.request.json"
    weave_path = args.output_dir / "living-reef-biography.weave.json"
    portable_path = args.output_dir / "living-reef-biography.json"
    manifest_path = args.output_dir / "manifest.json"
    write_json(request_path, request)
    completed = subprocess.run(
        [
            "cargo", "run", "--locked", "--quiet", "--",
            "--input", os.path.relpath(request_path, WEAVE_DIR),
            "--output", os.path.relpath(weave_path, WEAVE_DIR),
        ],
        cwd=WEAVE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout)
    if not receipt.get("reload_equal") or not receipt.get("validated_after_reload"):
        raise RuntimeError("native Universal Weave round-trip did not validate")
    receipt["artifact"] = weave_path.name
    write_json(portable_path, receipt)
    artifacts = {
        path.name: local_blob(path, path.stem, "application/json")
        for path in (request_path, weave_path, portable_path)
    }
    manifest = {
        "format": "chreatures-living-reef-biography-manifest-v1",
        "archive_scope": "external public evidence graph; never resident memory or controller input",
        "actual_experience_and_forecast_fit_are_separate_branches": True,
        "native_weave_roundtrip": True,
        "node_count": receipt["node_count"],
        "edge_count": receipt["edge_count"],
        "source": sources,
        "artifacts": artifacts,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
