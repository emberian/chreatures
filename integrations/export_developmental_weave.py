#!/usr/bin/env python3
"""Export sparse developmental provenance through the native Universal Weave adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEAVE_DIR = ROOT / "integrations" / "weave"
DEFAULT_OUTPUT = ROOT / "integrations" / "artifacts" / "developmental-biography"
CHECKPOINT_FORMAT = "chreatures-developmental-habitat-checkpoint-v1"
NATIVE_FORMATS = {
    "chreatures-native-developmental-resident-v1",
    "chreatures-native-developmental-resident-rich-v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def read_json(path: Path) -> Any:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def local_blob(path: Path, role: str, media_type: str) -> dict[str, Any]:
    digest = sha256(path)
    return {
        "role": role,
        "uri": f"urn:sha256:{digest}",
        "sha256": digest,
        "bytes": path.stat().st_size,
        "media_type": media_type,
        "verification": "verified_local_sha256",
    }


def reported_blob(
    digest: str, role: str, media_type: str, size: int | None
) -> dict[str, Any]:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{role} does not have a lowercase SHA-256")
    result: dict[str, Any] = {
        "role": role,
        "uri": f"urn:sha256:{digest}",
        "sha256": digest,
        "media_type": media_type,
        "verification": "reported_by_hash_verified_source_receipt",
    }
    if size is not None:
        result["bytes"] = size
    return result


def native_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if "metadata" not in payload:
            raise ValueError("native artifact has no metadata")
        metadata = json.loads(str(payload["metadata"].item()))
    if metadata.get("format") not in NATIVE_FORMATS:
        raise ValueError("native artifact format is not a developmental resident")
    artifact_identity = metadata.get("artifact_sha256")
    if not isinstance(artifact_identity, str) or len(artifact_identity) != 64:
        raise ValueError("native artifact has no valid internal identity")
    return metadata


def verified_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = read_json(path)
    if envelope.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a current developmental checkpoint")
    state = envelope.get("state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state is absent")
    if hashlib.sha256(canonical(state)).hexdigest() != envelope.get("sha256"):
        raise ValueError("checkpoint envelope state hash differs")
    return envelope, state


def event(
    record_id: str,
    time: Any,
    record_type: str,
    text: str,
    parents: list[str],
    *,
    blobs: list[dict[str, Any]] | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not record_id or not record_type or not text:
        raise ValueError("evidence nodes require id, record_type, and text")
    return {
        "id": record_id,
        "time": time,
        "record_type": record_type,
        "text": text,
        "parent_ids": parents,
        "blob_refs": blobs or [],
        "fields": fields or {},
    }


def checkpoint_events(
    path: Path,
    native_id: str,
    native_identity: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    envelope, state = verified_checkpoint(path)
    habitat_id = str(state.get("id", ""))
    tick = int(state.get("tick", -1))
    if not habitat_id or tick < 0:
        raise ValueError("checkpoint lacks habitat id or tick")
    snapshot_model = state.get("resident_controller", {}).get("model_identity", {})
    if snapshot_model.get("artifact_sha256") != native_identity:
        raise ValueError("checkpoint resident artifact differs from native artifact")

    journal: list[dict[str, Any]] = []
    previous = native_id
    for raw in state.get("journal", []):
        if not isinstance(raw, dict):
            raise ValueError("checkpoint journal contains a non-object")
        item = {
            "id": str(raw["id"]),
            "time": raw["time"],
            "record_type": str(raw.get("kind", "episode")),
            "text": str(raw["text"]),
            "parent_ids": [previous],
            "fields": {
                key: value
                for key, value in raw.items()
                if key not in {"id", "time", "kind", "text"}
            },
        }
        journal.append(item)
        previous = item["id"]

    hatched = next(
        (item["id"] for item in journal if item["record_type"] == "hatched"), native_id
    )
    evidence: list[dict[str, Any]] = []
    goal_ids: list[str] = []
    for resident, cognition in sorted(state.get("cognition_state", {}).items()):
        goal = cognition.get("goal", {}) if isinstance(cognition, dict) else {}
        if not goal.get("valid"):
            continue
        recorded_tick = int(goal["recorded_tick"])
        slot = int(goal["slot"])
        goal_id = f"goal-selection:{habitat_id}:{resident}:{recorded_tick}:{slot}"
        goal_ids.append(goal_id)
        evidence.append(
            event(
                goal_id,
                {"domain": "model_tick", "value": recorded_tick},
                "achieved_goal_selection",
                f"{resident} selected a private achieved-history goal for a ten-tick attempt.",
                [hatched, native_id],
                fields={
                    "resident": resident,
                    "slot": slot,
                    "recorded_tick": recorded_tick,
                    "recorded_time": goal.get("recorded_time"),
                    "remaining_ticks_at_snapshot": goal.get("remaining_ticks"),
                    "snapshot_tick": tick,
                    "private_goal_key_or_window_exported": False,
                },
            )
        )

    checkpoint_blob = local_blob(path, "developmental_habitat_checkpoint", "application/json")
    snapshot_id = f"snapshot:{habitat_id}:{tick}:{envelope['sha256']}"
    parents = ([journal[-1]["id"]] if journal else [native_id]) + goal_ids
    evidence.append(
        event(
            snapshot_id,
            {"domain": "model_tick", "value": tick},
            "snapshot",
            f"Authenticated developmental habitat snapshot at model tick {tick}.",
            list(dict.fromkeys(parents)),
            blobs=[checkpoint_blob],
            fields={
                "habitat_id": habitat_id,
                "state_sha256": envelope["sha256"],
                "resident_count": len(state.get("cognition_state", {})),
                "goal_selection_count": len(goal_ids),
                "journal_event_count": len(journal),
            },
        )
    )
    return journal + evidence, {
        "habitat_id": habitat_id,
        "tick": tick,
        "blob": checkpoint_blob,
        "state_sha256": envelope["sha256"],
        "goal_selection_count": len(goal_ids),
    }


def gam_events(report_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    if report.get("gamfit", {}).get("status") != "complete":
        raise ValueError("GAM observatory report is not complete")
    artifacts = manifest.get("artifacts", {})
    report_blob = local_blob(report_path, "gam_observatory_report", "application/json")
    manifest_blob = local_blob(manifest_path, "gam_observatory_manifest", "application/json")
    source_id = f"gam-source:{report['source_set_sha256']}"
    records = [
        event(
            source_id,
            {"domain": "development_step", "value": report["development"]["summary"]["steps"]},
            "verified_source",
            "Verified multi-world developmental telemetry used by the native GAM fits.",
            [],
            blobs=[report_blob, manifest_blob],
            fields={
                "source_set_sha256": report["source_set_sha256"],
                "held_out_worlds": report["gamfit"]["split"]["held_out_worlds"],
                "interpretation": report["gamfit"]["interpretation"],
            },
        )
    ]
    for name, model in sorted(report["gamfit"]["models"].items()):
        artifact = model["artifact"]
        listed = artifacts.get(artifact["file"])
        if not isinstance(listed, dict) or listed.get("sha256") != artifact.get("sha256"):
            raise ValueError(f"manifest does not authenticate GAM model {name}")
        records.append(
            event(
                f"gam-law-fit:{name}:{artifact['sha256']}",
                {"domain": "development_step", "value": report["development"]["summary"]["steps"]},
                "gam_law_fit",
                f"Native GAM fit for {name}; descriptive held-out prediction, with no causal claim.",
                [source_id],
                blobs=[
                    reported_blob(
                        artifact["sha256"],
                        f"native_gam_{name}",
                        "application/gzip",
                        int(artifact["bytes"]),
                    )
                ],
                fields={
                    "formula": model["formula"],
                    "held_out": model["held_out"],
                    "baselines": model["baselines"],
                    "reload_max_abs_prediction_delta": artifact[
                        "reload_max_abs_prediction_delta"
                    ],
                    "gamfit_source_commit": report["gamfit"]["library"][
                        "source_commit"
                    ],
                },
            )
        )
    return records, {"report": report_blob, "manifest": manifest_blob, "models": len(records) - 1}


def build_request(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    result = read_json(args.online_result)
    identity = read_json(args.online_identity)
    if result.get("format") != "chreatures-online-sensorimotor-development-v1":
        raise ValueError("online result has the wrong format")
    if identity.get("format") != result["format"]:
        raise ValueError("online identity and result formats differ")
    if int(result.get("updates", -1)) != 160 or int(result.get("physical_steps", -1)) <= 0:
        raise ValueError("online result is not the completed 160-update run")
    if result.get("status") != "research joined-development run; no behavior claim":
        raise ValueError("online result status differs from the completed research run")

    native = native_metadata(args.native_artifact)
    checkpoint_identity = native.get("checkpoint", {})
    if checkpoint_identity.get("sha256") != result.get("artifact_sha256"):
        raise ValueError("native artifact does not derive from the online result artifact")
    source_commit = args.source_commit.lower()
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("--source-commit must be a full lowercase Git commit")

    result_blob = local_blob(args.online_result, "online_result", "application/json")
    identity_blob = local_blob(args.online_identity, "online_run_identity", "application/json")
    native_blob = local_blob(
        args.native_artifact, "native_developmental_resident", "application/x-npz"
    )
    training_blob = reported_blob(
        result["artifact_sha256"],
        "online_training_checkpoint",
        "application/x-pytorch",
        args.training_artifact_bytes,
    )
    seed = int(identity["seed"])
    run_id = f"online-development:seed-{seed}:{result_blob['sha256']}"
    final_update_id = f"developmental-milestone:{run_id}:update-{result['updates']}"
    training_artifact_id = f"model-artifact:online:{result['artifact_sha256']}"
    native_id = f"model-export:native:{native['artifact_sha256']}"
    evidence = [
        event(
            run_id,
            {"domain": "online_update", "value": 0},
            "development_run",
            "Joined online sensorimotor worker and achieved-goal manager research run began.",
            [],
            blobs=[identity_blob, result_blob],
            fields={
                "source_commit": source_commit,
                "seed": seed,
                "worlds": identity["arguments"]["worlds"],
                "residents_per_world": result["world_transport_timing"]["residents_per_world"],
                "dt_seconds": identity["dt_seconds"],
                "graph_sha256": identity["graph_sha256"],
                "port_bundle_sha256": identity["port_bundle_sha256"],
                "behavior_claim": False,
            },
        ),
        event(
            final_update_id,
            {"domain": "online_update", "value": result["updates"]},
            "developmental_milestone",
            f"Completed {result['updates']} online updates and {result['physical_steps']} physical steps.",
            [run_id],
            fields={
                "last_update": result["last_update"],
                "elapsed_seconds": result["elapsed_seconds"],
                "status": result["status"],
            },
        ),
        event(
            training_artifact_id,
            {"domain": "online_update", "value": result["updates"]},
            "model_artifact",
            "Final joined-development PyTorch artifact recorded by the completed run.",
            [final_update_id],
            blobs=[training_blob],
            fields={"format": result["format"], "verification": training_blob["verification"]},
        ),
        event(
            native_id,
            {"domain": "online_update", "value": result["updates"]},
            "model_export",
            "Joined worker, goal encoder, and goal manager exported for native resident execution.",
            [training_artifact_id],
            blobs=[native_blob],
            fields={
                "format": native["format"],
                "artifact_identity": native["artifact_sha256"],
                "execution": native["execution"],
                "observation_contract": native["observation_contract"],
                "temporal_contract": native["temporal_contract"],
            },
        ),
    ]
    inputs: dict[str, Any] = {
        "online_result": result_blob,
        "online_identity": identity_blob,
        "online_training_checkpoint": training_blob,
        "native_artifact": {**native_blob, "artifact_identity": native["artifact_sha256"]},
    }
    habitat_id = None
    if args.checkpoint is not None:
        checkpoint_records, checkpoint_summary = checkpoint_events(
            args.checkpoint, native_id, native["artifact_sha256"]
        )
        evidence.extend(checkpoint_records)
        habitat_id = checkpoint_summary["habitat_id"]
        inputs["developmental_checkpoint"] = checkpoint_summary
        snapshot_id = next(
            record["id"]
            for record in reversed(checkpoint_records)
            if record["record_type"] == "snapshot"
        )
        evidence.append(
            event(
                f"model-promotion:{habitat_id}:{native['artifact_sha256']}",
                {"domain": "model_tick", "value": checkpoint_summary["tick"]},
                "model_promotion",
                "Native developmental artifact instantiated in an authenticated resident habitat.",
                [native_id, snapshot_id],
                fields={
                    "habitat_id": habitat_id,
                    "artifact_identity": native["artifact_sha256"],
                    "criterion": "present in authenticated checkpoint",
                    "behavior_claim": False,
                },
            )
        )

    if (args.gam_report is None) != (args.gam_manifest is None):
        raise ValueError("--gam-report and --gam-manifest must be supplied together")
    if args.gam_report is not None:
        records, gam_summary = gam_events(args.gam_report, args.gam_manifest)
        evidence.extend(records)
        inputs["gam"] = gam_summary

    archive_id = f"developmental-biography:{result_blob['sha256']}:{native['artifact_sha256']}"
    return {
        "archive_id": archive_id,
        "habitat_id": habitat_id,
        "description": (
            "External sparse developmental biography and research-law provenance; "
            "this graph is not organism memory."
        ),
        "journal": [],
        "evidence": evidence,
    }, {"archive_id": archive_id, "source_commit": source_commit, "inputs": inputs}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-result", type=Path, required=True)
    parser.add_argument("--online-identity", type=Path, required=True)
    parser.add_argument("--native-artifact", type=Path, required=True)
    parser.add_argument("--training-artifact-bytes", type=int)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gam-report", type=Path)
    parser.add_argument("--gam-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.training_artifact_bytes is not None and args.training_artifact_bytes < 1:
        parser.error("--training-artifact-bytes must be positive")
    for name in ("online_result", "online_identity", "native_artifact", "checkpoint", "gam_report", "gam_manifest"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request, source_manifest = build_request(args)
    request_path = args.output_dir / "developmental-biography.request.json"
    weave_path = args.output_dir / "developmental-biography.weave.json"
    portable_path = args.output_dir / "developmental-biography.json"
    manifest_path = args.output_dir / "manifest.json"
    write_json(request_path, request)

    relative_input = Path(os.path.relpath(request_path, WEAVE_DIR))
    relative_output = Path(os.path.relpath(weave_path, WEAVE_DIR))
    completed = subprocess.run(
        ["cargo", "run", "--locked", "--quiet", "--", "--input", str(relative_input), "--output", str(relative_output)],
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

    artifacts = {}
    for path in (request_path, weave_path, portable_path):
        artifacts[path.name] = local_blob(path, path.stem, "application/json")
    manifest = {
        "format": "chreatures-developmental-biography-manifest-v1",
        "archive_scope": "external evidence graph; never resident memory or controller input",
        "native_weave_roundtrip": True,
        "node_count": receipt["node_count"],
        "edge_count": receipt["edge_count"],
        "source": source_manifest,
        "artifacts": artifacts,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
