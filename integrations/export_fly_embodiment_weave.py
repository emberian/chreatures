#!/usr/bin/env python3
"""Export the actual-fly V4 embodiment evidence through native Universal Weave.

Large service/parity artifacts remain in paperbin.  The public projection keeps
their verified identities and compact causal records.  Work without a sealed
receipt is represented by a proposed node and can be replaced by an executed
node by rerunning this exporter with the corresponding receipt option.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from integrations.export_live_cns_weave import (
    blob,
    canonical,
    node,
    read_json,
    require,
    require_hash,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
WEAVE = ROOT / "integrations" / "weave"
DEFAULT_PAPERBIN = Path.home() / "paperbin" / "chreatures" / "integration" / "fly-v4"
DEFAULT_OUTPUT = ROOT / "integrations" / "artifacts" / "fly-embodiment-wave-v4"
DEFAULT_PUBLIC = ROOT / "site" / "assets" / "fly-embodiment-weave-v4.json"
DEFAULT_NATIVE_PUBLIC = ROOT / "site" / "assets" / "fly-embodiment-evidence.weave.json"


def npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        key = "metadata_json" if "metadata_json" in archive.files else "metadata"
        require(key in archive.files, f"{path} has no scalar metadata")
        value = json.loads(str(archive[key].item()))
    require(isinstance(value, dict), f"{path} metadata differs")
    return value


def evidence_counts(channels: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(channel.get("evidence_grade") for channel in channels)
    require(set(counts) <= {"measured", "inter_animal_inferred", "engineered", "unsupported"},
            "BODY807 evidence grade differs")
    return {name: counts.get(name, 0) for name in
            ("measured", "inter_animal_inferred", "engineered", "unsupported")}


def completed(value: dict[str, Any]) -> bool:
    return value.get("completed") is True or value.get("passed") is True or value.get("status") in {
        "completed", "executed", "passed", "trained",
    }


def optional_result(
    path: Path | None,
    *,
    role: str,
    record_type: str,
    stage: int,
    text: str,
    parents: dict[str, str],
) -> dict[str, Any] | None:
    if path is None:
        return None
    value = read_json(path)
    require(completed(value), f"{role} does not declare a completed execution")
    source = blob(path, role, "application/json")
    return node(
        f"{record_type}:{source['sha256']}", stage, record_type, text,
        parents=parents, blobs=[source],
        fields={
            "status": "executed receipt",
            "format": value.get("format"),
            "receipt_summary": value,
            "causal_role": role,
        },
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    body_schema_path = ROOT / "native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/schema.json"
    body807_path = ROOT / "research/fly_embodiment/body807-channel-schema.json"
    motor92_path = ROOT / "research/fly_embodiment/motor92-channel-schema.json"
    atlas_path = ROOT / "research/fly_embodiment/fly-body-neural-atlas-v1.npz"
    atlas_manifest_path = ROOT / "research/fly_embodiment/fly-body-neural-atlas-v1.manifest.json"
    source_ledger_path = ROOT / "research/fly_embodiment/source_ledger.json"
    teacher_dir = ROOT / "native/fly-body/assets/author-step-bank-v1"
    teacher_manifest_path = teacher_dir / "manifest.json"
    teacher_bank_path = teacher_dir / "trajectory-bank.npz"
    teacher_receipt_path = ROOT / "native/fly-body/receipts/author-teacher-b4/receipt.json"
    teacher_trace_path = teacher_receipt_path.parent / "teacher-replay-trace.npz"
    physical_start_path = ROOT / "native/fly-body/receipts/training-4-startup-mujoco-3.12.json"

    body_schema = read_json(body_schema_path)
    body807 = read_json(body807_path)
    motor92 = read_json(motor92_path)
    atlas_manifest = read_json(atlas_manifest_path)
    ledger = read_json(source_ledger_path)
    teacher_manifest = read_json(teacher_manifest_path)
    teacher_receipt = read_json(teacher_receipt_path)
    physical_start = read_json(physical_start_path)

    body_blob = blob(body_schema_path, "female_flygym_body_schema", "application/json")
    body807_blob = blob(body807_path, "body807_cns_sensory_schema", "application/json")
    motor92_blob = blob(motor92_path, "motor92_cns_actuator_schema", "application/json")
    atlas_blob = blob(atlas_path, "male_cns_fly_body_atlas", "application/x-npz",
                      atlas_manifest["artifact_sha256"])
    atlas_manifest_blob = blob(atlas_manifest_path, "anatomical_atlas_manifest", "application/json")
    ledger_blob = blob(source_ledger_path, "anatomical_source_ledger", "application/json")
    require(body_schema.get("schema") == "chreatures.neuromechfly-body.v1", "body schema differs")
    require(body807.get("schema") == "chreatures.body-afferent-channels.v1" and body807.get("length") == 807,
            "BODY807 schema differs")
    require(motor92.get("schema") == "chreatures.motor-output-channels.v1" and motor92.get("length") == 92,
            "MOTOR92 schema differs")
    require(atlas_manifest.get("schema") == "chreatures.fly-body-neural-atlas.v1", "atlas manifest differs")
    require(atlas_manifest["inputs"]["body_schema"]["sha256"] == body_blob["sha256"], "atlas body identity differs")
    require(atlas_manifest["inputs"]["body807_channel_schema"]["sha256"] == body807_blob["sha256"],
            "atlas BODY807 identity differs")
    require(atlas_manifest["inputs"]["motor92_channel_schema"]["sha256"] == motor92_blob["sha256"],
            "atlas MOTOR92 identity differs")
    counts = atlas_manifest["counts"]

    body_id = f"female-body-morphology:{body_blob['sha256']}"
    sensory_id = f"body807-interface:{body807_blob['sha256']}"
    motor_id = f"motor92-interface:{motor92_blob['sha256']}"
    cns_source_id = f"male-cns-source-ledger:{ledger_blob['sha256']}"
    atlas_id = f"cross-animal-anatomical-atlas:{atlas_blob['sha256']}"

    records = [
        node(
            body_id, 0, "body_morphology_import",
            "FlyGym/NeuroMechFly supplies the articulated body geometry, joint order and effective servo boundary used by this wave.",
            blobs=[body_blob],
            fields={
                "status": "author-derived artifact imported",
                "causal_role": "defines physical geometry, 126 joint axes and the M90 MuJoCo control order",
                "biological_subject": "micro-CT-derived morphology of one adult female Drosophila; not the MaleCNS specimen",
                "evidence": {"observed": "author morphology and rigging", "inferred": "joint/servo mechanics are model choices", "engineered": "effective position servos and adhesion"},
                "segments": len(body_schema["body_segments"]), "joint_dofs": len(body_schema["joint_dofs"]),
                "physical_actuators": len(body_schema["actuators"]), "axis_order": body_schema["fixture"]["axis_order"],
                "morphology_asset_set_sha256": atlas_manifest["cross_schema_identity"]["morphology_asset_set_sha256"]["sha256"],
                "source": body_schema["source"], "mass_unit_status": body_schema["fixture"]["mass_unit_status"],
            },
        ),
        node(
            sensory_id, 1, "body_to_cns_interface",
            "BODY807 names the physical and engineered channels that may reach annotated MaleCNS afferents; zero-mask channels remain explicit.",
            parents={body_id: "physical_measurement_source"}, blobs=[body807_blob],
            fields={
                "status": "implemented structural interface",
                "causal_role": "normalizes body-local state before anatomy-masked injection into full CNS recurrence",
                "channels": 807, "blocks": body807["blocks"], "evidence_grade_counts": evidence_counts(body807["channels"]),
                "unsupported_channels": counts["unsupported_body_channels"],
                "supported_afferent_rows": counts["nonvisual_afferent_rows_with_body807_support"],
                "unsupported_afferent_rows": counts["unsupported_nonvisual_afferent_rows"],
                "controller_bypass": False,
            },
        ),
        node(
            motor_id, 1, "cns_to_body_interface",
            "MOTOR92 defines signed effective body targets plus adhesion, pharyngeal pump and salivary drive; it does not claim individual muscle recruitment.",
            parents={body_id: "physical_actuator_boundary"}, blobs=[motor92_blob],
            fields={
                "status": "implemented semantic interface", "causal_role": "decodes post-recurrence motor-neuron state to physical targets",
                "channels": 92, "blocks": motor92["blocks"], "physical_mjcf_outputs": 90, "native_physiology_outputs": 2,
                "warning": "signed servo weights are learned within anatomical masks; direction is not inferred from side or muscle name",
            },
        ),
        node(
            cns_source_id, 0, "male_cns_anatomical_sources",
            "MaleCNS neuron identities are joined to MANC/FANC and sensory literature with every cross-animal inference labeled.",
            blobs=[ledger_blob],
            fields={
                "status": "source ledger verified locally", "causal_role": "defines CNS rows and the evidence basis for body-part masks",
                "biological_subject": "male CNS reconstruction; MANC/FANC homolog evidence includes different animals and a female VNC reference",
                "source_count": len(ledger["sources"]),
                "evidence_terms": atlas_manifest["evidence_terms"],
            },
        ),
        node(
            atlas_id, 2, "anatomy_mask_join",
            "The anatomical atlas joins the female body model to the male CNS through explicit measured, inter-animal inferred, engineered and unsupported mappings.",
            parents={body_id: "morphology_order", sensory_id: "sensory_channels", motor_id: "actuator_channels", cns_source_id: "neuron_annotations"},
            blobs=[atlas_blob, atlas_manifest_blob],
            fields={
                "status": "machine-readable atlas generated", "causal_role": "structural masks constrain trainable BODY807 and MOTOR92 adapters",
                "animals": {"body": "female FlyGym specimen", "brain_and_cord": "MaleCNS male specimen", "fine_motor_homology": "MANC/FANC cross-animal inference"},
                "motor_rows": counts["motor_rows"], "motor_rows_supported": counts["motor_rows_with_m92_support"],
                "motor_rows_unsupported": counts["unsupported_motor_rows"], "fine_cohort_rows": counts["fine_cohort_motor_rows"],
                "fine_unresolved_motor_rows": counts["motor_rows"] - counts["fine_cohort_motor_rows"],
                "afferent_rows": counts["nonvisual_afferent_rows"], "afferent_rows_supported": counts["nonvisual_afferent_rows_with_body807_support"],
                "unsupported_motor": atlas_manifest["unsupported_motor"],
                "learned_adapter_scope": "signed weights may distribute only within nonzero anatomical cohorts",
            },
        ),
    ]

    teacher_manifest_blob = blob(teacher_manifest_path, "author_teacher_bank_manifest", "application/json")
    teacher_bank_blob = blob(teacher_bank_path, "author_teacher_trajectory_bank", "application/x-npz",
                             teacher_manifest["bank"]["sha256"])
    require(teacher_manifest.get("format") == "chreatures.author-step-bank.v1", "teacher bank differs")
    require(teacher_manifest["body_schema_sha256"] == body_blob["sha256"], "teacher body identity differs")
    teacher_bank_id = f"author-teacher-bank:{teacher_bank_blob['sha256']}"
    records.append(node(
        teacher_bank_id, 3, "offline_author_teacher",
        "An author walking trajectory bank supplies loss-only targets through shared joint IDs and is absent from the deployed controller.",
        parents={body_id: "shared_joint_order", motor_id: "walking42_and_adhesion6_targets"},
        blobs=[teacher_manifest_blob, teacher_bank_blob],
        fields={"status": "derived author artifact", "causal_role": "offline supervision only", "sample_count": teacher_manifest["bank"]["sample_count"],
                "source": teacher_manifest["source"], "limits": teacher_manifest["limits"], "production_policy_ingress": False},
    ))
    teacher_receipt_blob = blob(teacher_receipt_path, "author_teacher_physical_replay", "application/json")
    teacher_trace_blob = blob(teacher_trace_path, "author_teacher_replay_trace", "application/x-npz",
                              teacher_receipt["trace"]["sha256"])
    start_blob = blob(physical_start_path, "b4_native_physical_startup", "application/json")
    require(teacher_receipt.get("status") == "executed" and teacher_receipt.get("finite") is True,
            "teacher replay is not a completed finite execution")
    require(teacher_receipt["teacher_bank_sha256"] == teacher_bank_blob["sha256"] and
            teacher_receipt["body_schema_sha256"] == body_blob["sha256"], "teacher replay identity differs")
    teacher_replay_id = f"author-teacher-replay:{teacher_receipt_blob['sha256']}"
    records.append(node(
        teacher_replay_id, 4, "executed_teacher_replay",
        "Four author-derived target modes ran for two seconds in the actual B4 MuJoCo body; this tests physical feasibility, not learned control.",
        parents={teacher_bank_id: "target_trajectory", body_id: "physical_body"},
        blobs=[teacher_receipt_blob, teacher_trace_blob, start_blob],
        fields={"status": "executed physical receipt", "causal_role": "checks the offline teacher/body pairing",
                "residents": 4, "control_ticks": teacher_receipt["control_ticks"], "physics_dt_s": teacher_receipt["physics_dt_s"],
                "modes": teacher_receipt["modes"], "observed_checks": teacher_receipt["observed_checks"],
                "outcomes": teacher_receipt["outcomes"], "interpretation": teacher_receipt["interpretation"]},
    ))

    initialized = args.paperbin / "initialized"
    equivalence_path = initialized / "identity-correction-equivalence.json"
    old_fixture_path = initialized / "torch-parity-cns-v4.npz"
    corrected_fixture_path = initialized / "torch-parity-cns-v4-corrected.npz"
    webgpu_path = initialized / "webgpu-parity-v4.json"
    corrected_service_path = initialized / "corrected-initialized-cns-v4.bin"
    equivalence = read_json(equivalence_path)
    webgpu = read_json(webgpu_path)
    old_fixture = npz_metadata(old_fixture_path)
    corrected_fixture = npz_metadata(corrected_fixture_path)
    equivalence_blob = blob(equivalence_path, "v4_identity_correction_equivalence", "application/json")
    webgpu_blob = blob(webgpu_path, "original_v4_full_cns_webgpu_parity", "application/json",
                       equivalence["linked_numeric_evidence"]["webgpu"]["report_sha256"])
    old_fixture_blob = blob(old_fixture_path, "original_v4_torch_parity_fixture", "application/x-npz",
                            equivalence["old_fixture"]["file_sha256"])
    corrected_fixture_blob = blob(corrected_fixture_path, "corrected_v4_torch_parity_fixture", "application/x-npz",
                                  equivalence["corrected_fixture"]["file_sha256"])
    corrected_service_blob = blob(corrected_service_path, "corrected_initialized_chcns4_service", "application/octet-stream",
                                  equivalence["corrected_service"]["file_sha256"])
    require(equivalence.get("format") == "chreatures-cns-v4-metadata-correction-v1", "V4 equivalence differs")
    require(equivalence.get("service_array_count") == 42 and equivalence.get("service_array_sha256_identical") is True
            and equivalence.get("fixture_numeric_array_count") == 20
            and equivalence.get("fixture_numeric_array_sha256_identical") is True,
            "V4 identity correction did not preserve every numerical array")
    require(webgpu.get("format") == "chreatures-cns-webgpu-v4-dawn-probe-v1" and webgpu.get("passed") is True,
            "original V4 WebGPU parity differs")
    require(old_fixture.get("format") == corrected_fixture.get("format") == "chreatures-cns-v4-parity-v1",
            "V4 Torch fixture format differs")
    require(old_fixture["service_file_sha256"] == equivalence["old_service"]["file_sha256"], "old service identity differs")
    require(corrected_fixture["service_file_sha256"] == corrected_service_blob["sha256"], "corrected service identity differs")
    original_id = f"original-v4-service:{equivalence['old_service']['file_sha256']}"
    original_parity_id = f"original-v4-full-cns-parity:{webgpu_blob['sha256']}"
    corrected_id = f"corrected-v4-service:{corrected_service_blob['sha256']}"
    equivalence_id = f"corrected-v4-numeric-equivalence:{equivalence_blob['sha256']}"
    records.extend([
        node(original_id, 3, "original_initialized_cns_identity",
             "The first initialized CHCNS4 file carried the morphology hash in its actuator-schema field; its numerical tensors remain a distinct historical identity.",
             parents={atlas_id: "anatomical_arrays"},
             fields={"status": "superseded identity; numerical execution retained", "service_artifact_sha256": equivalence["old_service"]["file_sha256"],
                     "adapter_sha256": equivalence["old_service"]["adapter_sha256"], "incorrect_actuator_schema_sha256": equivalence["old_service"]["actuator_schema_sha256"],
                     "causal_role": "historical input to the executed parity run"}),
        node(original_parity_id, 4, "executed_original_full_cns_parity",
             "Dawn/WebGPU executed the original initialized full 165,122-neuron V4 recurrence against its Torch fixture and passed the declared numerical gate.",
             parents={original_id: "executed_service", atlas_id: "full_graph_interfaces"}, blobs=[webgpu_blob, old_fixture_blob],
             fields={"status": "executed numerical parity", "causal_role": "checks WebGPU recurrence, Z512 and MOTOR92 against Torch",
                     "service_artifact_sha256": webgpu["serviceArtifactSha256"], "neurons": 165122, "ticks": webgpu["ticks"],
                     "limits": webgpu["limits"], "final_max_abs": webgpu["finalMaxAbs"],
                     "latent_max_abs_by_tick": webgpu["latentMaxAbsByTick"], "motor_max_abs_by_tick": webgpu["motorMaxAbsByTick"],
                     "byte_exact_restore": webgpu["byteExactRestore"],
                     "scope": "original service identity only; no corrected file execution is claimed"}),
        node(corrected_id, 4, "corrected_initialized_cns_identity",
             "The corrected initialized CHCNS4 binds the canonical MOTOR92 schema and has a new service and adapter identity.",
             parents={atlas_id: "canonical_motor92_binding", original_id: "metadata_correction_predecessor"},
             blobs=[corrected_service_blob, corrected_fixture_blob],
             fields={"status": "initialized-untrained artifact", "causal_role": "current full-CNS substrate for collection",
                     "service_artifact_sha256": equivalence["corrected_service"]["file_sha256"],
                     "adapter_sha256": equivalence["corrected_service"]["adapter_sha256"],
                     "actuator_schema_sha256": equivalence["corrected_service"]["actuator_schema_sha256"],
                     "training_status": "initialized-untrained", "competence_claim": None}),
        node(equivalence_id, 5, "executed_identity_equivalence",
             "A separate receipt proves the original and corrected artifacts have identical 42 service arrays and 20 parity arrays; it does not relabel the original backend run.",
             parents={original_parity_id: "original_executed_numbers", corrected_id: "corrected_identity"}, blobs=[equivalence_blob],
             fields={"status": "executed byte-identity comparison", "causal_role": "transfers numerical values while preserving distinct artifact identities",
                     "reason": equivalence["reason"], "service_array_count": equivalence["service_array_count"],
                     "fixture_numeric_array_count": equivalence["fixture_numeric_array_count"],
                     "service_arrays_identical": equivalence["service_array_sha256_identical"],
                     "fixture_arrays_identical": equivalence["fixture_numeric_array_sha256_identical"],
                     "original_backend_execution_relabelled": False,
                     "linked_numeric_evidence": equivalence["linked_numeric_evidence"]}),
    ])

    resident_receipt_path = args.resident_receipt
    if resident_receipt_path is not None:
        resident_receipt = read_json(resident_receipt_path)
        require(resident_receipt.get("format") == "chreatures-initialized-cns-resident-v4-receipt-v1",
                "resident initialization receipt differs")
        require(resident_receipt["cns_service"]["service_artifact_sha256"] == corrected_service_blob["sha256"],
                "resident CNS service differs")
        resident_receipt_blob = blob(resident_receipt_path, "initialized_private_resident_receipt", "application/json")
        records.append(node(
            f"initialized-private-resident:{resident_receipt['resident']['artifact_sha256']}", 6, "initialized_private_resident",
            "A fresh private context resident is bound to the corrected CNS and carries no trained-status or competence claim.",
            parents={corrected_id: "cns_service_dependency", equivalence_id: "numerical_identity_boundary"}, blobs=[resident_receipt_blob],
            fields={"status": "initialized-untrained", "causal_role": "future parent for private context/consequence learning",
                    "artifact_sha256": resident_receipt["resident"]["artifact_sha256"],
                    "sequence_control_artifact_sha256": resident_receipt["sequence_control"]["artifact_sha256"],
                    "ingress": resident_receipt["ingress"], "initialization": resident_receipt["initialization"],
                    "native_validation": resident_receipt["validation"]},
        ))

    ecology_path = args.paperbin / "ecology-growth-joined-v2.json"
    host_path = args.host_growth_receipt
    ecology = read_json(ecology_path)
    host = read_json(host_path)
    ecology_blob = blob(ecology_path, "native_ecology_growth_receipt", "application/json")
    host_blob = blob(host_path, "photon_coupled_mujoco_growth_receipt", "application/json")
    require(ecology.get("format") == "chreatures-ecology-growth-joined-v2" and ecology.get("passed") is True,
            "native ecology growth receipt differs")
    require(
        host.get("format") == "chreatures-fly-ecology-host-joined-receipt-v1"
        and host.get("checks", {}).get("autonomous_physical_construction") is True
        and host.get("checks", {}).get("finite_body_optic") is True
        and host.get("checks", {}).get("physical_light_funds_native_photon_capture") is True
        and host.get("checks", {}).get("full_snapshot_restore_replay") == "exact"
        and host.get("scenario", {}).get("control_ticks_to_construction") == 200
        and host.get("growth", {}).get("proposed_at_construction") == 2
        and host.get("growth", {}).get("accepted") == 1
        and host.get("growth", {}).get("blocked") == 1
        and host.get("illumination", {}).get("captured_photon_energy", 0) > 0,
        "joined physical growth receipt differs",
    )
    ecology_id = f"native-ecology-growth:{ecology_blob['sha256']}"
    host_id = f"photon-physical-growth:{host_blob['sha256']}"
    records.extend([
        node(ecology_id, 3, "executed_native_ecology_growth",
             "Native ecology executed inherited apical and lateral construction, anchored colony birth, transaction replay and geometry-dependent route transport.",
             blobs=[ecology_blob],
             fields={"status": "executed native receipt", "causal_role": "owns finite material, developmental RNG and growth transactions",
                     **ecology, "scope_boundary": ecology["scope"]}),
        node(host_id, 5, "executed_photon_physical_growth_join",
             "The MuJoCo/Wasm host supplied physical light and geometry to native ecology: one branch was accepted, one blocked, and the coupled state replayed exactly.",
             parents={ecology_id: "native_growth_transaction", body_id: "physical_geometry"}, blobs=[host_blob],
             fields={"status": "executed joined host receipt", "causal_role": "couples physical photons and geomDistance clearance to native growth",
                     "scenario": host["scenario"], "growth": host["growth"], "illumination": host["illumination"],
                     "retina": host["retina"], "checks": host["checks"], "performance": host["performance"],
                     "limitations": host["limitations"]}),
    ])

    collection_parents = {teacher_replay_id: "offline_teacher_targets", corrected_id: "current_full_cns", host_id: "actual_body_and_ecology_host"}
    corpus_record = optional_result(args.corpus_receipt, role="sealed_corpus_receipt", record_type="completed_fly_corpus",
                                    stage=7, text="The full actual-body CNS collection corpus completed and was sealed.", parents=collection_parents)
    if corpus_record is None:
        corpus_id = "proposed-fly-corpus:pending"
        records.append(node(corpus_id, 7, "proposed_collection_result",
                            "The 12-world B4 full-CNS corpus is still collecting; no completed corpus result is represented here.",
                            parents=collection_parents,
                            fields={"status": "proposed; receipt absent", "causal_role": "future training corpus",
                                    "expected_worlds": 12, "model_ingress": ["optic_rgb", "BODY807", "delivered_context12"],
                                    "teacher_scope": "loss-only", "completion_claim": False}))
    else:
        records.append(corpus_record); corpus_id = corpus_record["id"]

    trained_record = optional_result(args.training_receipt, role="trained_resident_receipt", record_type="completed_resident_training",
                                     stage=8, text="A fresh resident and CNS service completed training on the sealed actual-body corpus.",
                                     parents={corpus_id: "sealed_training_data"})
    if trained_record is None:
        trained_id = "proposed-resident-training:pending"
        records.append(node(trained_id, 8, "proposed_training_result",
                            "Resident/CNS training has no completed receipt yet; initialized artifacts are not presented as trained.",
                            parents={corpus_id: "future_training_data"},
                            fields={"status": "proposed; receipt absent", "causal_role": "future learned CNS-to-context and motor adaptation", "training_status": "absent", "completion_claim": False}))
    else:
        records.append(trained_record); trained_id = trained_record["id"]

    gam_record = optional_result(args.gam_receipt, role="gam_mechanism_receipt", record_type="completed_gam_analysis",
                                 stage=9, text="Native GAM analysis completed on the joined fly-body outcomes.",
                                 parents={corpus_id: "observed_outcomes", trained_id: "candidate_controller"})
    if gam_record is None:
        gam_id = "proposed-gam-analysis:pending"
        records.append(node(gam_id, 9, "proposed_gam_result", "No completed V4 GAM receipt exists yet.",
                            parents={corpus_id: "future_observed_outcomes", trained_id: "future_controller"},
                            fields={"status": "proposed; receipt absent", "causal_role": "future mechanism analysis", "completion_claim": False}))
    else:
        records.append(gam_record); gam_id = gam_record["id"]

    joined_record = optional_result(args.joined_receipt, role="trained_joined_assay_receipt", record_type="completed_trained_joined_assay",
                                    stage=10, text="The trained resident completed a joined full-CNS physical/ecology assay.",
                                    parents={trained_id: "trained_controller", host_id: "physical_ecology", gam_id: "mechanism_analysis"})
    if joined_record is None:
        records.append(node("proposed-trained-joined-assay:pending", 10, "proposed_joined_result",
                            "No trained joined CNS/body/ecology assay receipt exists yet.",
                            parents={trained_id: "future_trained_controller", host_id: "validated_physical_ecology", gam_id: "future_analysis"},
                            fields={"status": "proposed; receipt absent", "causal_role": "future behavioral evidence", "completion_claim": False}))
    else:
        records.append(joined_record)

    request = {
        "archive_id": "fly-embodiment-wave-v4-20260908",
        "description": "A fly in a growing world: actual body morphology, cross-animal anatomical masks, full MaleCNS numerical evidence, offline teacher feasibility and native physical ecology. Pending learning remains proposed.",
        "evidence": records,
    }
    for record in records:
        require(len(canonical(record["fields"])) <= 40000, f"record fields exceed public bound: {record['id']}")
        require(set(record["parent_ids"]) == set(record["fields"]["parent_roles"]), f"parent roles differ: {record['id']}")

    output = args.output_dir.resolve()
    public = args.public.resolve()
    native_public = args.native_public.resolve()
    request_path = output / "fly-embodiment-evidence.request.json"
    weave_path = output / "fly-embodiment-evidence.weave.json"
    receipt_path = output / "fly-embodiment-evidence.receipt.json"
    targets = [request_path, weave_path, receipt_path, public, native_public]
    if not args.replace:
        require(not any(path.exists() for path in targets), "refusing to replace an existing fly embodiment projection")
    output.mkdir(parents=True, exist_ok=True)
    public.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n")
    completed_weave = subprocess.run(
        ["cargo", "run", "--locked", "--release", "--manifest-path", str(WEAVE / "Cargo.toml"), "--",
         "--input", str(request_path), "--output", str(weave_path)],
        cwd=WEAVE, check=True, capture_output=True, text=True,
    )
    portable = json.loads(completed_weave.stdout)
    require(portable.get("reload_equal") is True and portable.get("validated_after_reload") is True,
            "native Weave serialize/reload validation failed")
    require(portable.get("node_count") == len(records), "native Weave node count differs")
    portable["artifact"] = native_public.name
    portable["artifact_sha256"] = sha256(weave_path)
    portable["public_summary"] = {
        "title": "A fly in a growing world",
        "architecture": "fly-embodiment-v4",
        "body_subject": "female FlyGym/NeuroMechFly morphology",
        "cns_subject": "MaleCNS male brain and cord",
        "neurons": 165122, "body_channels": 807, "motor_channels": 92,
        "materialized_nodes": sum(not record["record_type"].startswith("proposed_") for record in records),
        "executed_receipt_nodes": sum(record["record_type"].startswith("executed_") for record in records),
        "proposed_nodes": sum(record["record_type"].startswith("proposed_") for record in records),
        "training_status": "completed receipt imported" if args.training_receipt else "pending",
        "original_corrected_identity_separation": True,
        "native_layout": "Universal Weave stable node IDs and topological order",
    }
    encoded = json.dumps(portable, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    require(len(encoded) <= 1024 * 1024 and weave_path.stat().st_size <= 1024 * 1024,
            "fly embodiment public Weave exceeds 1 MiB")
    public.write_bytes(encoded)
    shutil.copyfile(weave_path, native_public)
    receipt = {
        "format": "chreatures-fly-embodiment-weave-export-v1",
        "exporter_sha256": sha256(Path(__file__).resolve()),
        "request_sha256": sha256(request_path), "weave_sha256": sha256(weave_path),
        "portable_sha256": sha256(public), "native_public_sha256": sha256(native_public),
        "node_count": portable["node_count"], "edge_count": portable["edge_count"],
        "multi_parent_nodes": portable["multi_parent_nodes"], "reload_equal": True,
        "validated_after_reload": True, "universal_weave": portable["library"],
        "pending": {"corpus": corpus_record is None, "training": trained_record is None,
                    "gam": gam_record is None, "joined_assay": joined_record is None},
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paperbin", type=Path, default=DEFAULT_PAPERBIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--native-public", type=Path, default=DEFAULT_NATIVE_PUBLIC)
    parser.add_argument("--resident-receipt", type=Path,
                        default=Path.home() / "paperbin/chreatures/fly-learning-v4/initialized-resident-v4/receipt.json")
    parser.add_argument("--host-growth-receipt", type=Path,
                        default=Path.home() / "paperbin/chreatures/integration/fly-ecology-v4-host/host-joined-receipt.json")
    parser.add_argument("--corpus-receipt", type=Path)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--gam-receipt", type=Path)
    parser.add_argument("--joined-receipt", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
