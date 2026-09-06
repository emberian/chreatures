#!/usr/bin/env python3
"""Export the completed grandchild batch and separate courtyard pause to Weave."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from export_developmental_weave import (
    event,
    local_blob,
    reported_blob,
    sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
WEAVE_DIR = ROOT / "integrations" / "weave"
DEFAULT_OUTPUT = ROOT / "integrations" / "artifacts" / "grandchild-evidence"
DEFAULT_COURTYARD = (
    Path.home() / "paperbin/chreatures/deployments/predictive-courtyard-a5bdee3"
)
WEAVE_COMMIT = "7a5a0dabb94885e44ad8a6c4355c015d7f38020f"
ATLAS_COMMIT = "c6ec91f40801c09a518bc2b3f16f5c3a4cea5481"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def verified_external_blob(
    path: Path, expected_sha: str, role: str, media_type: str
) -> dict[str, Any]:
    expected_sha = require_sha(expected_sha, role)
    actual = sha256(path)
    if actual != expected_sha:
        raise ValueError(f"{role} hash differs: expected {expected_sha}, got {actual}")
    return {
        "role": role,
        "uri": f"urn:sha256:{actual}",
        "sha256": actual,
        "bytes": path.stat().st_size,
        "media_type": media_type,
        "verification": "verified_external_local_sha256",
    }


def grandchild_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = read_object(args.native_receipt)
    audit = read_object(args.behavior_audit)
    atlas = read_object(args.gam_atlas)
    provenance = read_object(args.gam_provenance)

    if receipt.get("format") != "chreatures-native-developmental-resident-export-receipt-v1":
        raise ValueError("native export receipt has the wrong format")
    native = receipt.get("artifact", {})
    checkpoint = receipt.get("checkpoint", {})
    native_sha = require_sha(native.get("file_sha256"), "native artifact")
    if sha256(args.native_artifact) != native_sha:
        raise ValueError("native V3 artifact differs from its receipt")
    if native.get("format") != "chreatures-native-developmental-resident-rich-v3":
        raise ValueError("native artifact is not the completed rich V3 export")
    if native_sha != "c843f94937832df786435feffc18239822a795ea63ee2c609d18235677f1b6a5":
        raise ValueError("native artifact differs from the frozen c843f949 export")

    if audit.get("format") != "chreatures-rich-development-behavior-analysis-v1":
        raise ValueError("behavior audit has the wrong format")
    scope = audit.get("scope", {})
    rows = audit.get("per_episode_world_resident", [])
    if len(rows) != 96 or int(scope.get("transitions", -1)) != 491_520:
        raise ValueError("behavior audit is not the complete 96-trajectory batch")
    contact_trajectories = sum(int(row["mouth_contact_steps"]) > 0 for row in rows)
    energy_negative = sum(float(row["energy_end"]) < float(row["energy_start"]) for row in rows)
    reward_negative = sum(float(row["reward_sum"]) < 0.0 for row in rows)
    if (contact_trajectories, energy_negative, reward_negative) != (90, 96, 96):
        raise ValueError("behavior audit headline counts differ from the frozen result")
    if audit.get("source", {}).get("final_checkpoint_sha256") != checkpoint.get("sha256"):
        raise ValueError("audit and native receipt identify different final checkpoints")

    if atlas.get("schema") != "chreatures-developmental-gam-atlas-v1":
        raise ValueError("GAM atlas has the wrong schema")
    if provenance.get("format") != "chreatures-grandchild-developmental-atlas-provenance-v1":
        raise ValueError("GAM provenance has the wrong format")
    for key in ("final_checkpoint_sha256", "result_sha256", "identity_sha256"):
        if provenance.get(key) != audit.get("source", {}).get(key):
            raise ValueError(f"GAM provenance and audit disagree on {key}")
    models = atlas.get("models", {})
    fitted = {name: model for name, model in models.items() if model.get("status") == "fitted native GAM"}
    failed = {name: model for name, model in models.items() if model.get("status") == "native fit failed; no model minted"}
    if set(fitted) != {"body_law_residual", "effort"} or set(failed) != {"goal_progress"}:
        raise ValueError("GAM atlas fit/failure set differs from the frozen result")

    native_blob = local_blob(args.native_artifact, "native_v3_export", "application/x-npz")
    receipt_blob = local_blob(args.native_receipt, "native_v3_export_receipt", "application/json")
    audit_blob = local_blob(args.behavior_audit, "full_transition_audit", "application/json")
    atlas_blob = local_blob(args.gam_atlas, "native_gam_atlas", "application/json")
    provenance_blob = local_blob(args.gam_provenance, "native_gam_atlas_provenance", "application/json")
    checkpoint_blob = reported_blob(
        require_sha(checkpoint.get("sha256"), "grandchild checkpoint"),
        "grandchild_training_checkpoint",
        "application/x-pytorch",
        int(checkpoint["bytes"]),
    )

    run_id = f"grandchild-run:{audit['source']['identity_sha256']}"
    complete_id = f"grandchild-complete:{checkpoint['sha256']}"
    export_id = f"native-v3-export:{native_sha}"
    audit_id = f"transition-audit:{audit_blob['sha256']}"
    atlas_id = f"gam-atlas:{atlas_blob['sha256']}"
    records = [
        event(
            run_id,
            {"domain": "training_update", "value": 0},
            "development_run",
            "Grandchild developmental run began from inherited controller state with fresh world, neural, private-history, and RNG state.",
            [],
            fields={
                "source_commit": checkpoint["source_commit"],
                "parent_checkpoint_sha256": receipt["lineage"]["parent_checkpoint_sha256"],
                "one_lineage_one_seed": True,
                "matched_control": False,
            },
        ),
        event(
            complete_id,
            {"domain": "training_update", "value": 160},
            "completed_checkpoint",
            "The grandchild run completed 160 updates and 491,520 resident transitions.",
            [run_id],
            blobs=[checkpoint_blob],
            fields={
                "physical_steps": int(checkpoint["physical_steps"]),
                "resident_transitions": int(checkpoint["resident_transitions"]),
                "pending_rollout_length": int(checkpoint["pending_rollout_length"]),
                "checkpoint_stored_outside_git": True,
            },
        ),
        event(
            export_id,
            {"domain": "training_update", "value": 160},
            "native_model_export",
            "The completed checkpoint was exported as the authenticated native rich V3 artifact for a future birth.",
            [complete_id],
            blobs=[native_blob, receipt_blob],
            fields={
                "format": native["format"],
                "artifact_identity": native["artifact_sha256"],
                "file_sha256": native_sha,
                "status": receipt["status"],
                "instantiated_in_this_record": False,
            },
        ),
        event(
            audit_id,
            {"domain": "resident_transition", "value": int(scope["transitions"])},
            "full_transition_audit",
            "All recorded grandchild transitions were audited: 90 of 96 trajectories made mouth-material contact, while all 96 lost energy.",
            [complete_id],
            blobs=[audit_blob],
            fields={
                "trajectories": len(rows),
                "mouth_contact_trajectories": contact_trajectories,
                "mouth_contact_steps": sum(int(row["mouth_contact_steps"]) for row in rows),
                "ingested_mass_total": sum(float(row["ingested_mass_sum"]) for row in rows),
                "energy_negative_trajectories": energy_negative,
                "physical_reward_negative_trajectories": reward_negative,
                "energy_delta_total": audit["physiology"]["energy_delta_total"],
                "matched_control": False,
            },
        ),
        event(
            atlas_id,
            {"domain": "resident_transition", "value": int(scope["transitions"])},
            "native_gam_atlas",
            "Native gamfit analyzed the completed grandchild telemetry; the atlas is analyst-only and was not installed in a resident.",
            [complete_id, audit_id],
            blobs=[atlas_blob, provenance_blob],
            fields={
                "source_commit": ATLAS_COMMIT,
                "gamfit_version": atlas["source"]["gamfit_version"],
                "gamfit_source_commit": atlas["source"]["gam_source_commit"],
                "fit_seconds": atlas["fit_seconds"],
                "split": atlas["split"],
                "resident_promotion": False,
            },
        ),
    ]

    for name, model in sorted(fitted.items()):
        records.append(
            event(
                f"gam-fit:{name}:{model['model_sha256']}",
                {"domain": "resident_transition", "value": int(scope["transitions"])},
                "native_gam_fit",
                f"Native GAM fit for {name} completed with descriptive validation and held-out episode metrics.",
                [atlas_id],
                blobs=[
                    reported_blob(
                        require_sha(model["model_sha256"], f"{name} GAM"),
                        f"native_gam_{name}",
                        "application/octet-stream",
                        None,
                    )
                ],
                fields={
                    "formula": model["formula"],
                    "unit": model["unit"],
                    "metrics": model["metrics"],
                    "causal_claim": False,
                    "installed_in_resident": False,
                },
            )
        )

    failure = failed["goal_progress"]
    records.append(
        event(
            f"gam-fit-failure:goal-progress:{atlas_blob['sha256']}",
            {"domain": "resident_transition", "value": int(scope["transitions"])},
            "native_gam_fit_failure",
            "The goal-progress GAM failed REML certification after 200 outer iterations; no model or surface was minted.",
            [atlas_id],
            fields={
                "formula": failure["formula"],
                "unit": failure["unit"],
                "train_rows": failure["train_rows"],
                "projected_gradient_norm": 0.6186,
                "stationarity_bound": 0.00412,
                "hessian_positive_semidefinite": False,
                "model_minted": False,
                "full_error_preserved_in_atlas": True,
            },
        )
    )

    proposal_id = f"proposal:paired-goal-coefficient:{checkpoint['sha256']}"
    records.append(
        event(
            proposal_id,
            {"domain": "proposal", "value": 1},
            "future_training_proposal",
            "A paired continuation from the frozen update-160 checkpoint is proposed to compare goal-progress coefficients 0.01 and 0.001.",
            [audit_id],
            fields={
                "status": "proposed_only",
                "shared_start_checkpoint_sha256": checkpoint["sha256"],
                "paired_single_variable": "goal_progress_coefficient",
                "selection_metrics": [
                    "full_episode_energy_change",
                    "conserved_ingested_mass",
                    "effort",
                    "mouth_contact_bouts",
                ],
                "fork_started": False,
                "result_claim": False,
            },
        )
    )
    for coefficient, role in ((0.01, "control"), (0.001, "reduced_goal_shaping")):
        records.append(
            event(
                f"proposal-arm:goal-coefficient-{coefficient:g}:{checkpoint['sha256']}",
                {"domain": "proposal_arm", "value": coefficient},
                "proposed_training_arm",
                f"Proposed {role} continuation with goal_progress_coefficient={coefficient:g}; it has not run.",
                [proposal_id],
                fields={
                    "coefficient": coefficient,
                    "role": role,
                    "status": "proposed_not_executed",
                    "checkpoint_created": False,
                    "resident_born": False,
                    "autostart": False,
                },
            )
        )

    return records, {
        "checkpoint": checkpoint_blob,
        "native_artifact": native_blob,
        "native_receipt": receipt_blob,
        "behavior_audit": audit_blob,
        "gam_atlas": atlas_blob,
        "gam_provenance": provenance_blob,
    }


def courtyard_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = read_object(args.courtyard_receipt)
    pause = read_object(args.courtyard_pause)
    if receipt.get("format") != "chreatures-predictive-courtyard-birth-v2":
        raise ValueError("courtyard receipt has the wrong format")
    if receipt.get("world_id") != "0102b60c-dfed-483d-a636-12a0d0f33f19":
        raise ValueError("courtyard receipt identifies an unexpected life")
    if "automatically paused after incomplete tick" not in str(pause.get("reason")):
        raise ValueError("courtyard pause record does not describe the physical failure")
    checkpoint_blob = verified_external_blob(
        args.courtyard_checkpoint,
        pause["checkpoint_sha256"],
        "courtyard_last_complete_checkpoint",
        "application/json",
    )
    pause_blob = local_blob(args.courtyard_pause, "courtyard_incomplete_tick_pause", "application/json")
    receipt_blob = local_blob(args.courtyard_receipt, "courtyard_life_receipt", "application/json")

    with args.courtyard_checkpoint.open() as stream:
        checkpoint = json.load(stream)
    state = checkpoint.get("state", {})
    if checkpoint.get("format") != "chreatures-developmental-habitat-checkpoint-v2":
        raise ValueError("courtyard checkpoint has the wrong format")
    if state.get("id") != receipt["world_id"]:
        raise ValueError("courtyard checkpoint and receipt identify different lives")
    if state.get("resident_controller", {}).get("model_identity", {}).get("artifact_sha256") != receipt["controller"]["artifact_sha256"]:
        raise ValueError("courtyard checkpoint and receipt identify different controllers")

    life_id = f"courtyard-v2-life:{receipt['world_id']}"
    checkpoint_id = f"courtyard-checkpoint:{checkpoint_blob['sha256']}"
    failure_id = f"courtyard-physical-failure:{pause_blob['sha256']}"
    records = [
        event(
            life_id,
            {"domain": "model_tick", "value": 0},
            "separate_research_life",
            "Predictive Courtyard V2 began as a separate research life with its own world and private state.",
            [],
            blobs=[receipt_blob],
            fields={
                "world_id": receipt["world_id"],
                "source_revision": receipt["deployment"]["source_revision"],
                "controller_format": receipt["controller"]["format"],
                "controller_artifact_sha256": receipt["controller"]["artifact_sha256"],
                "same_life_as_grandchild": False,
            },
        ),
        event(
            checkpoint_id,
            {"domain": "model_tick", "value": int(state["tick"])},
            "last_complete_checkpoint",
            "The last complete Courtyard V2 checkpoint was preserved outside Git before the later incomplete tick.",
            [life_id],
            blobs=[checkpoint_blob],
            fields={
                "world_id": state["id"],
                "tick": int(state["tick"]),
                "model_seconds": float(state["tick"]) * 0.05,
                "state_sha256": require_sha(checkpoint["sha256"], "courtyard checkpoint state"),
                "checkpoint_stored_outside_git": True,
            },
        ),
        event(
            failure_id,
            {"domain": "model_seconds_read_only", "value": pause["read_only_state_time"]},
            "physical_source_failure_pause",
            "Courtyard V2 paused after a physical source position left the field range; the incomplete tick was not saved.",
            [checkpoint_id],
            blobs=[pause_blob],
            fields={
                "reason": pause["reason"],
                "source_index": 68,
                "paused": True,
                "incomplete_tick_committed": False,
                "continuation_started": False,
                "autostart": False,
            },
        ),
    ]
    return records, {
        "life_receipt": receipt_blob,
        "last_complete_checkpoint": checkpoint_blob,
        "pause_record": pause_blob,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-artifact", type=Path, default=ROOT / "data/genomes/developmental-resident-rich-grandchild-update160-v3.npz")
    parser.add_argument("--native-receipt", type=Path, default=ROOT / "data/genomes/developmental-resident-rich-grandchild-update160-v3.receipt.json")
    parser.add_argument("--behavior-audit", type=Path, default=ROOT / "data/analysis/rich-grandchild160-behavior.json")
    parser.add_argument("--gam-atlas", type=Path, default=ROOT / "integrations/gam_mechanisms/artifacts/grandchild_developmental_atlas_v1/developmental_atlas.json")
    parser.add_argument("--gam-provenance", type=Path, default=ROOT / "integrations/gam_mechanisms/artifacts/grandchild_developmental_atlas_v1/provenance.json")
    parser.add_argument("--courtyard-receipt", type=Path, default=ROOT / "data/development/predictive-courtyard-v2.receipt.json")
    parser.add_argument("--courtyard-checkpoint", type=Path, default=DEFAULT_COURTYARD / "runs/courtyard.json")
    parser.add_argument("--courtyard-pause", type=Path, default=DEFAULT_COURTYARD / "runs/incomplete-tick-pause.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())

    grandchild, grandchild_sources = grandchild_records(args)
    courtyard, courtyard_sources = courtyard_records(args)
    evidence = grandchild + courtyard
    archive_id = "grandchild-batch-and-courtyard-pause:" + hashlib.sha256(
        json.dumps([item["id"] for item in evidence], separators=(",", ":")).encode()
    ).hexdigest()
    request = {
        "archive_id": archive_id,
        "habitat_id": None,
        "description": (
            "External evidence record for the completed grandchild batch and a separate "
            "paused Courtyard V2 life; never organism memory or controller input."
        ),
        "journal": [],
        "evidence": evidence,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / "grandchild-evidence.request.json"
    weave_path = args.output_dir / "grandchild-evidence.weave.json"
    portable_path = args.output_dir / "grandchild-evidence.json"
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
    portable = json.loads(completed.stdout)
    if not portable.get("reload_equal") or not portable.get("validated_after_reload"):
        raise RuntimeError("native Universal Weave round-trip did not validate")
    portable["artifact"] = weave_path.name
    write_json(portable_path, portable)

    artifacts = {
        path.name: local_blob(path, path.stem, "application/json")
        for path in (request_path, weave_path, portable_path)
    }
    manifest = {
        "format": "chreatures-grandchild-evidence-weave-manifest-v1",
        "archive_scope": "external provenance only; no runtime, controller, organism-memory, or autostart role",
        "archive_id": archive_id,
        "native_weave": {
            "library": "universal-weave",
            "version": "0.5.0",
            "source_commit": WEAVE_COMMIT,
            "roundtrip_equal": True,
            "validated_after_reload": True,
        },
        "node_count": portable["node_count"],
        "edge_count": portable["edge_count"],
        "branches": {
            "grandchild_completed": len(grandchild),
            "courtyard_separate_paused_life": len(courtyard),
        },
        "source": {
            "grandchild": grandchild_sources,
            "courtyard": courtyard_sources,
            "atlas_source_commit": ATLAS_COMMIT,
        },
        "artifacts": artifacts,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
