#!/usr/bin/env python3
"""Authenticate a research birth/checkpoint and prepare recording evidence links."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.checkpoint import canonical
from chreatures.population_evidence import (
    BATCH_FORMAT,
    LEDGER_FORMAT,
    PopulationEvidenceError,
    atomic_write_json,
    canonical_bytes,
    environment_record_from_native,
    evidence_record,
    genome_record_from_native,
    local_blob,
    read_json,
    sha256_file,
    validate_records,
)
from chreatures.resident_birth import validate_manifest


BINDING_FORMAT = "chreatures-living-recording-private-binding-v1"
BIRTH_EXPORT_FORMAT = "chreatures-population-birth-export-v1"
CHECKPOINT_FORMAT = "chreatures-developmental-habitat-checkpoint-v4"
LINK_FORMAT = "chreatures-living-recording-evidence-link-v2"
MIGRATION_FORMAT = "chreatures-research-continuation-migration-v1"
RECORDING_FORMAT = "chreatures-living-reef-public-recording-v2"


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PopulationEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _authenticated(value: Mapping[str, Any], label: str) -> str:
    declared = _hash(value.get("sha256"), f"{label} identity")
    body = dict(value)
    body.pop("sha256")
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != declared:
        raise PopulationEvidenceError(f"{label} content differs from its identity")
    return declared


def _checkpoint(path: Path) -> tuple[dict[str, Any], str, str]:
    envelope = read_json(path)
    if envelope.get("format") != CHECKPOINT_FORMAT or set(envelope) != {
        "format", "sha256", "state",
    }:
        raise PopulationEvidenceError("research checkpoint is not current habitat v4")
    state = envelope["state"]
    if not isinstance(state, dict):
        raise PopulationEvidenceError("research checkpoint lacks state")
    state_sha = _hash(envelope.get("sha256"), "checkpoint state")
    if hashlib.sha256(canonical(state)).hexdigest() != state_sha:
        raise PopulationEvidenceError("research checkpoint state hash differs")
    return state, state_sha, sha256_file(path)


def _birth_export(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    receipt_path = path / "receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("format") != BIRTH_EXPORT_FORMAT:
        raise PopulationEvidenceError("unsupported cold-birth export")
    receipt_sha = _authenticated(receipt, "cold-birth receipt")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PopulationEvidenceError("cold-birth receipt lacks output hashes")
    for name in ("habitat.json", "biosphere.json", "resident-birth.json"):
        target = path / name
        if not target.is_file() or sha256_file(target) != outputs.get(name):
            raise PopulationEvidenceError(f"cold-birth output differs: {name}")
    manifest = validate_manifest(read_json(path / "resident-birth.json"))
    return receipt, manifest, receipt_sha


def _binding(path: Path, recording: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    binding = read_json(path)
    expected_fields = {
        "format", "source_world_id", "world_source_revision",
        "world_source_content_sha256", "capture_tool_revision",
        "capture_tool_file_sha256", "physical_profile_sha256", "graph_sha256",
        "resident_artifact_sha256", "engine_identity_sha256", "bodies",
        "content_sha256",
    }
    if binding.get("format") != BINDING_FORMAT or set(binding) != expected_fields:
        raise PopulationEvidenceError("unsupported private body binding")
    binding_sha = _hash(binding.get("content_sha256"), "private body binding identity")
    body = dict(binding)
    body.pop("content_sha256")
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != binding_sha:
        raise PopulationEvidenceError("private body binding content differs")
    receipt = recording.get("private_binding_receipt")
    bodies = binding.get("bodies")
    if not isinstance(bodies, list):
        raise PopulationEvidenceError("private body binding lacks bodies")
    if not isinstance(receipt, Mapping) or any(
        receipt.get(key) != expected
        for key, expected in (
            ("format", BINDING_FORMAT),
            ("content_sha256", binding_sha),
            ("file_sha256", sha256_file(path)),
            ("bytes", path.stat().st_size),
            ("body_count", len(bodies)),
            ("scope", "private source-ID binding; excluded from public site assets"),
        )
    ):
        raise PopulationEvidenceError("public recording does not authenticate its private binding")
    return binding, binding_sha


def _recording(path: Path) -> dict[str, Any]:
    recording = read_json(path)
    if recording.get("format") != RECORDING_FORMAT:
        raise PopulationEvidenceError("unsupported public recording")
    declared = _hash(recording.get("content_sha256"), "recording content")
    body = dict(recording)
    body.pop("content_sha256")
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != declared:
        raise PopulationEvidenceError("public recording content hash differs")
    return recording


def _migration(
    path: Path,
    *,
    source_checkpoint: tuple[Mapping[str, Any], str, str],
    target_initial_checkpoint: tuple[Mapping[str, Any], str, str],
) -> tuple[dict[str, Any], str, str]:
    receipt = read_json(path)
    expected_fields = {
        "format", "from_world_id", "to_world_id", "tick", "from_revision",
        "to_revision", "from_engine_sha256", "to_engine_sha256",
        "source_checkpoint_file_sha256", "source_checkpoint_state_sha256",
        "source_neural_file_sha256", "source_neural_payload_sha256",
        "source_event_snapshot_sha256", "source_event_head_sha256",
        "target_neural_file_sha256", "target_neural_payload_sha256",
        "reason", "body_identity_mapping", "no_model_advance_during_migration",
        "state_changes", "future_numerics", "output_checkpoint_file_sha256",
        "output_checkpoint_state_sha256", "output_neural_bytes",
        "retained_component_sha256", "source_execution_migration_count", "sha256",
    }
    if receipt.get("format") != MIGRATION_FORMAT or set(receipt) != expected_fields:
        raise PopulationEvidenceError("unsupported research-continuation migration receipt")
    receipt_sha = _authenticated(receipt, "research-continuation migration receipt")
    receipt_file_sha = sha256_file(path)
    source_state, source_state_sha, source_file_sha = source_checkpoint
    target_state, target_state_sha, target_file_sha = target_initial_checkpoint
    tick = receipt.get("tick")
    mapping = receipt.get("body_identity_mapping")
    if (
        isinstance(tick, bool)
        or not isinstance(tick, int)
        or tick < 0
        or source_state.get("tick") != tick
        or target_state.get("tick") != tick
    ):
        raise PopulationEvidenceError("migration tick differs from its checkpoints")
    if any(
        receipt.get(key) != expected
        for key, expected in (
            ("from_world_id", source_state.get("id")),
            ("to_world_id", target_state.get("id")),
            ("source_checkpoint_file_sha256", source_file_sha),
            ("source_checkpoint_state_sha256", source_state_sha),
            ("output_checkpoint_file_sha256", target_file_sha),
            ("output_checkpoint_state_sha256", target_state_sha),
            ("from_engine_sha256", source_state.get("engine_identity", {}).get("sha256")),
            ("to_engine_sha256", target_state.get("engine_identity", {}).get("sha256")),
        )
    ):
        raise PopulationEvidenceError("migration receipt differs from checkpoint identity")
    for key in (
        "from_engine_sha256", "to_engine_sha256", "source_neural_file_sha256",
        "source_neural_payload_sha256", "source_event_snapshot_sha256",
        "source_event_head_sha256", "target_neural_file_sha256",
        "target_neural_payload_sha256",
    ):
        _hash(receipt.get(key), f"migration {key}")
    if (
        receipt.get("no_model_advance_during_migration") is not True
        or receipt.get("source_neural_payload_sha256")
        != receipt.get("target_neural_payload_sha256")
        or receipt.get("source_execution_migration_count") != 0
    ):
        raise PopulationEvidenceError("migration does not preserve frozen neural state")
    retained = receipt.get("retained_component_sha256")
    if not isinstance(retained, Mapping) or not retained:
        raise PopulationEvidenceError("migration lacks retained component identities")
    for name, digest in retained.items():
        _hash(digest, f"migration retained component {name}")
    if (
        not isinstance(mapping, Mapping)
        or not mapping
        or len(set(mapping.values())) != len(mapping)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items())
    ):
        raise PopulationEvidenceError("migration body identity mapping is invalid")
    if not isinstance(receipt.get("state_changes"), list) or not receipt["state_changes"]:
        raise PopulationEvidenceError("migration receipt lacks explicit state changes")
    return receipt, receipt_sha, receipt_file_sha


def _hatch_parent_indices(recording: Mapping[str, Any]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for event in recording.get("events", []):
        if event.get("kind") != "hatching":
            continue
        roles = event.get("actors", {}).get("body_roles", {})
        offspring = [int(key) for key, role in roles.items() if role == "offspring"]
        parents = [int(key) for key, role in roles.items() if role == "parent"]
        if len(offspring) != 1 or len(parents) != 1:
            raise PopulationEvidenceError("recorded hatching lacks exact parent/offspring roles")
        if offspring[0] in result:
            raise PopulationEvidenceError("one recorded body has multiple hatching events")
        result[offspring[0]] = (parents[0], int(event["tick"]))
    return result


def _law_ids(args: argparse.Namespace) -> tuple[list[str], dict[str, list[str]]]:
    event_laws: dict[str, list[str]] = {}
    for raw in args.event_law:
        if "=" not in raw:
            raise PopulationEvidenceError("--event-law must be KIND=RECORD_ID")
        kind, record_id = raw.split("=", 1)
        if not kind or not record_id:
            raise PopulationEvidenceError("--event-law must be KIND=RECORD_ID")
        event_laws.setdefault(kind, []).append(record_id)
    laws = list(args.associated_law_fit)
    if len(laws) != len(set(laws)) or any(
        len(values) != len(set(values)) for values in event_laws.values()
    ):
        raise PopulationEvidenceError("law-fit links must be unique")
    return laws, event_laws


def build(args: argparse.Namespace) -> dict[str, Any]:
    ledger = read_json(args.ledger)
    if ledger.get("format") != LEDGER_FORMAT:
        raise PopulationEvidenceError("unsupported population evidence ledger")
    validate_records(ledger["records"], campaign_id=ledger["campaign_id"])
    prior_batch = read_json(args.output_batch) if args.output_batch.exists() else None
    prior_records = {
        record["id"]: record
        for record in (prior_batch or {}).get("records", [])
        if isinstance(record, Mapping) and isinstance(record.get("id"), str)
    }
    recording = _recording(args.recording)
    binding, binding_sha = _binding(args.binding, recording)
    state, checkpoint_state_sha, checkpoint_file_sha = _checkpoint(args.checkpoint)
    source_checkpoint = _checkpoint(args.source_checkpoint)
    target_initial_checkpoint = _checkpoint(args.target_initial_checkpoint)
    source_state, source_checkpoint_state_sha, source_checkpoint_file_sha = source_checkpoint
    initial_state, target_initial_state_sha, target_initial_file_sha = (
        target_initial_checkpoint
    )
    migration, migration_sha, migration_file_sha = _migration(
        args.migration_receipt,
        source_checkpoint=source_checkpoint,
        target_initial_checkpoint=target_initial_checkpoint,
    )
    receipt, founder_manifest, birth_receipt_sha = _birth_export(args.birth_export)

    provenance = recording.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise PopulationEvidenceError("public recording lacks provenance")
    capture_tool = provenance.get("capture_tool")
    if not isinstance(capture_tool, Mapping):
        raise PopulationEvidenceError("public recording lacks capture-tool provenance")
    _hash(binding.get("world_source_content_sha256"), "world source content")
    _hash(binding.get("capture_tool_file_sha256"), "capture tool file")
    if any(
        binding.get(key) != provenance.get(provenance_key)
        for key, provenance_key in (
            ("world_source_revision", "world_source_revision"),
            ("world_source_content_sha256", "world_source_content_sha256"),
            ("physical_profile_sha256", "physical_profile_sha256"),
            ("graph_sha256", "graph_sha256"),
            ("resident_artifact_sha256", "resident_artifact_sha256"),
        )
    ) or binding.get("engine_identity_sha256") != provenance.get(
        "engine_identity", {}
    ).get("sha256") or any(
        binding.get(key) != capture_tool.get(provenance_key)
        for key, provenance_key in (
            ("capture_tool_revision", "revision"),
            ("capture_tool_file_sha256", "file_sha256"),
        )
    ):
        raise PopulationEvidenceError("private binding and public provenance differ")
    world_id = str(binding.get("source_world_id", ""))
    if not world_id or state.get("id") != world_id or recording.get(
        "event_stream", {}
    ).get("world_id") != world_id:
        raise PopulationEvidenceError("recording, binding and checkpoint world identities differ")
    if isinstance(state.get("tick"), bool) or not isinstance(state.get("tick"), int) or state["tick"] < 0:
        raise PopulationEvidenceError("life binding checkpoint tick is invalid")
    if state.get("engine_identity", {}).get("sha256") != binding["engine_identity_sha256"]:
        raise PopulationEvidenceError("checkpoint engine differs from recording")
    if state.get("id") != initial_state.get("id") or state.get("tick", -1) <= migration["tick"]:
        raise PopulationEvidenceError("recorded checkpoint is not the migrated research branch")
    neural_identity = state.get("neural_identity", {})
    checkpoint_graph = neural_identity.get("graph_sha256") or neural_identity.get(
        "graph", {}
    ).get("sha256")
    if checkpoint_graph != binding["graph_sha256"]:
        raise PopulationEvidenceError("checkpoint neural graph differs from recording")
    controller = state.get("resident_controller", {}).get("model_identity", {})
    controller_artifact = controller.get("artifact_sha256") or state.get(
        "resident_controller", {}
    ).get("artifact_sha256")
    if controller_artifact != binding["resident_artifact_sha256"]:
        raise PopulationEvidenceError("checkpoint resident artifact differs from recording")

    source_manifest = validate_manifest(source_state.get("birth_manifest", {}))
    initial_manifest = validate_manifest(initial_state.get("birth_manifest", {}))
    checkpoint_manifest = validate_manifest(state.get("birth_manifest", {}))
    founders = founder_manifest["residents"]
    source_residents = source_manifest["residents"]
    residents = checkpoint_manifest["residents"]
    if (
        source_residents != founders
        or initial_manifest["residents"] != source_residents
        or residents[: len(source_residents)] != source_residents
    ):
        raise PopulationEvidenceError("migration founders differ from cold-birth export")
    source_bodies = source_state.get("world", {}).get("bodies")
    if not isinstance(source_bodies, list) or len(source_bodies) != len(source_residents):
        raise PopulationEvidenceError("source checkpoint body order differs from birth manifest")
    migrated_bodies = initial_state.get("world", {}).get("bodies")
    if not isinstance(migrated_bodies, list) or len(migrated_bodies) != len(source_bodies):
        raise PopulationEvidenceError("migration changed the initial resident count")
    identity_mapping = migration["body_identity_mapping"]
    for source_body, target_body in zip(source_bodies, migrated_bodies, strict=True):
        source_identity = f"{migration['from_world_id']}:{source_body.get('id')}"
        target_identity = f"{migration['to_world_id']}:{target_body.get('id')}"
        if identity_mapping.get(source_identity) != target_identity:
            raise PopulationEvidenceError("migration body identity mapping differs from checkpoints")
    target_bodies = state.get("world", {}).get("bodies")
    if not isinstance(target_bodies, list) or len(target_bodies) != len(residents):
        raise PopulationEvidenceError("checkpoint body order differs from its birth manifest")
    body_rows = binding.get("bodies")
    if not isinstance(body_rows, list) or len(body_rows) != len(target_bodies):
        raise PopulationEvidenceError("private binding does not cover the checkpoint cohort")
    for index, (body, bound) in enumerate(zip(target_bodies, body_rows, strict=True)):
        if not isinstance(bound, Mapping) or bound.get("public_body") != index or bound.get(
            "source_body_id"
        ) != body.get("id"):
            raise PopulationEvidenceError("public body order differs from checkpoint body order")

    source = receipt.get("source", {})
    world = receipt.get("world", {})
    environment_receipt = world.get("environment_receipt", {})
    environment = environment_receipt.get("environment_record", {})
    environment_sha = _hash(environment.get("sha256"), "birth environment")
    if world.get("assignment_world_id") != environment_sha or environment_receipt.get(
        "environment_sha256"
    ) != environment_sha:
        raise PopulationEvidenceError("cold-birth environment identities differ")
    if source.get("profile", {}).get("sha256") != binding["physical_profile_sha256"]:
        raise PopulationEvidenceError("cold-birth profile differs from recording")
    if source.get("graph", {}).get("sha256") != binding["graph_sha256"]:
        raise PopulationEvidenceError("cold-birth graph differs from recording")
    if source.get("resident_artifact", {}).get("artifact_sha256") != binding[
        "resident_artifact_sha256"
    ]:
        raise PopulationEvidenceError("cold-birth controller differs from recording")

    existing = {record["id"]: record for record in ledger["records"]}
    campaigns = [
        record for record in ledger["records"] if record["record_type"] == "population_run"
    ]
    panels = [
        record for record in ledger["records"]
        if record["record_type"] == "environment_probe_panel"
        and record["fields"].get("descriptor_epoch_id")
    ]
    if len(campaigns) != 1 or len(panels) != 1:
        raise PopulationEvidenceError("living research binding requires one campaign and probe panel")
    campaign_id = campaigns[0]["id"]
    panel_id = panels[0]["id"]
    panel_sha = panels[0]["fields"]["probe_panel_sha256"]
    birth_blob = local_blob(
        args.birth_export / "resident-birth.json",
        role="genome_artifact",
        media_type="application/json",
    )
    receipt_blob = local_blob(
        args.birth_export / "receipt.json",
        role="environment_artifact",
        media_type="application/json",
    )
    checkpoint_blob = local_blob(
        args.checkpoint, role="life_checkpoint", media_type="application/json"
    )
    source_checkpoint_blob = local_blob(
        args.source_checkpoint, role="life_checkpoint", media_type="application/json"
    )
    migration_blob = local_blob(
        args.migration_receipt, role="migration_receipt", media_type="application/json"
    )
    target_initial_blob = local_blob(
        args.target_initial_checkpoint,
        role="target_initial_checkpoint",
        media_type="application/json",
    )
    records: list[dict[str, Any]] = []
    genomes: dict[str, Mapping[str, Any]] = {}
    for row in residents:
        genome = row["candidate"]
        genomes[str(genome["sha256"])] = genome
    for genome_sha, genome in sorted(genomes.items()):
        record_id = f"genome:{genome_sha}"
        candidate_record = genome_record_from_native(
            genome,
            campaign_record_id=campaign_id,
            time={"domain": "research_birth", "value": 0},
            artifact=birth_blob,
        )
        if record_id in prior_records and canonical_bytes(
            prior_records[record_id]
        ) != canonical_bytes(candidate_record):
            raise PopulationEvidenceError(
                f"existing birth batch candidate differs: {record_id}"
            )
        if record_id not in existing or record_id in prior_records:
            records.append(
                prior_records.get(record_id, candidate_record)
            )
    environment_id = f"environment:{environment_sha}"
    environment_record = environment_record_from_native(
        environment,
        campaign_record_id=campaign_id,
        probe_panel_record_id=panel_id,
        probe_panel_sha256=panel_sha,
        time={"domain": "research_birth", "value": 0},
        artifact=receipt_blob,
    )
    if environment_id in prior_records and canonical_bytes(
        prior_records[environment_id]
    ) != canonical_bytes(environment_record):
        raise PopulationEvidenceError(
            "existing birth batch environment differs from authenticated export"
        )
    if environment_id not in existing or environment_id in prior_records:
        records.append(
            prior_records.get(environment_id, environment_record)
        )

    hatch_parents = _hatch_parent_indices(recording)
    source_life_ids: list[str] = []
    life_ids: list[str] = []
    life_root_ids: list[str] = []
    observed_life_record_ids: list[str] = []
    for index, (source_body, resident) in enumerate(
        zip(source_bodies, source_residents, strict=True)
    ):
        source_body_id = str(source_body["id"])
        genome_sha = str(resident["candidate"]["sha256"])
        source_life_id = hashlib.sha256(
            canonical_bytes(
                {
                    "format": "chreatures-independent-research-life-v1",
                    "world_id": migration["from_world_id"],
                    "source_body_id": source_body_id,
                    "genome_sha256": genome_sha,
                }
            )
        ).hexdigest()
        source_life_ids.append(source_life_id)
        birth_id = f"birth:{source_life_id}"
        records.append(
            evidence_record(
                id=birth_id,
                time={"domain": "model_tick", "value": 0},
                record_type="birth",
                text="Original research resident instantiated by the authenticated canonical birth export.",
                parents={
                    f"genome:{genome_sha}": "candidate_genome",
                    environment_id: "environment",
                },
                fields={
                    "life_id": source_life_id,
                    "birth_mode": "experimental_initialization",
                    "genome_sha256": genome_sha,
                    "environment_sha256": environment_sha,
                    "birth_export_receipt_sha256": birth_receipt_sha,
                    "birth_proof_checkpoint_state_sha256": source_checkpoint_state_sha,
                    "world_instance_sha256": hashlib.sha256(
                        migration["from_world_id"].encode()
                    ).hexdigest(),
                    "source_body_id_sha256": hashlib.sha256(
                        source_body_id.encode()
                    ).hexdigest(),
                    "public_body": index,
                },
            )
        )
        source_checkpoint_id = (
            f"life-checkpoint:{source_life_id}:{migration['tick']}:"
            f"{source_checkpoint_file_sha}"
        )
        records.append(
            evidence_record(
                id=source_checkpoint_id,
                time={"domain": "model_tick", "value": migration["tick"]},
                record_type="life_checkpoint",
                text="Last coherent checkpoint of the original paused research life.",
                parents={birth_id: "life_continuation"},
                blobs=[source_checkpoint_blob],
                fields={
                    "life_id": source_life_id,
                    "checkpoint_sha256": source_checkpoint_file_sha,
                    "checkpoint_state_sha256": source_checkpoint_state_sha,
                    "tick": migration["tick"],
                },
            )
        )
        target_body_id = str(migrated_bodies[index]["id"])
        branch_life_id = hashlib.sha256(
            canonical_bytes(
                {
                    "format": "chreatures-research-branch-life-v1",
                    "migration_receipt_sha256": migration_sha,
                    "source_life_id": source_life_id,
                    "target_world_id": migration["to_world_id"],
                    "target_body_id": target_body_id,
                    "genome_sha256": genome_sha,
                }
            )
        ).hexdigest()
        life_ids.append(branch_life_id)
        branch_id = f"research-branch:{migration_sha}:{index}"
        life_root_ids.append(branch_id)
        records.append(
            evidence_record(
                id=branch_id,
                time={"domain": "model_tick", "value": migration["tick"]},
                record_type="research_branch",
                text="Authenticated research copy branched from a coherent paused-life checkpoint.",
                parents={
                    source_checkpoint_id: "source_checkpoint",
                    f"genome:{genome_sha}": "candidate_genome",
                    environment_id: "environment",
                },
                blobs=[migration_blob, target_initial_blob],
                fields={
                    "life_id": branch_life_id,
                    "source_life_id": source_life_id,
                    "source_tick": migration["tick"],
                    "branch_mode": "authenticated_research_copy",
                    "genome_sha256": genome_sha,
                    "environment_sha256": environment_sha,
                    "migration_receipt_sha256": migration_sha,
                    "migration_receipt_file_sha256": migration_file_sha,
                    "source_checkpoint_sha256": source_checkpoint_file_sha,
                    "source_checkpoint_state_sha256": source_checkpoint_state_sha,
                    "source_neural_snapshot_sha256": migration["source_neural_file_sha256"],
                    "source_neural_payload_sha256": migration["source_neural_payload_sha256"],
                    "source_event_snapshot_sha256": migration[
                        "source_event_snapshot_sha256"
                    ],
                    "source_event_head_sha256": migration["source_event_head_sha256"],
                    "target_initial_checkpoint_sha256": target_initial_file_sha,
                    "target_initial_checkpoint_state_sha256": target_initial_state_sha,
                    "target_neural_snapshot_sha256": migration["target_neural_file_sha256"],
                    "target_neural_payload_sha256": migration["target_neural_payload_sha256"],
                    "from_source_revision": migration["from_revision"],
                    "to_source_revision": migration["to_revision"],
                    "from_engine_identity_sha256": migration["from_engine_sha256"],
                    "to_engine_identity_sha256": migration["to_engine_sha256"],
                    "no_model_advance_during_migration": True,
                    "world_instance_sha256": hashlib.sha256(
                        migration["to_world_id"].encode()
                    ).hexdigest(),
                    "source_body_id_sha256": hashlib.sha256(
                        target_body_id.encode()
                    ).hexdigest(),
                    "public_body": index,
                },
            )
        )

    for index, (bound, resident) in enumerate(zip(body_rows, residents, strict=True)):
        source_body_id = str(bound["source_body_id"])
        genome_sha = str(resident["candidate"]["sha256"])
        if index < len(source_residents):
            life_id = life_ids[index]
            root_id = life_root_ids[index]
        else:
            life_id = hashlib.sha256(
                canonical_bytes(
                    {
                        "format": "chreatures-independent-research-life-v1",
                        "world_id": world_id,
                        "source_body_id": source_body_id,
                        "genome_sha256": genome_sha,
                    }
                )
            ).hexdigest()
            life_ids.append(life_id)
            hatch = hatch_parents.get(index)
            if hatch is None:
                raise PopulationEvidenceError(
                    "a post-migration resident lacks a captured hatching parent receipt"
                )
            parent_index, birth_tick = hatch
            if parent_index >= index:
                raise PopulationEvidenceError("hatching parent must precede its offspring")
            root_id = f"birth:{life_id}"
            life_root_ids.append(root_id)
            records.append(
                evidence_record(
                    id=root_id,
                    time={"domain": "model_tick", "value": birth_tick},
                    record_type="birth",
                    text="Funded offspring committed by the recorded physical world.",
                    parents={
                        f"genome:{genome_sha}": "candidate_genome",
                        environment_id: "environment",
                        life_root_ids[parent_index]: "physical_parent_birth",
                    },
                    fields={
                        "life_id": life_id,
                        "birth_mode": "embodied_reproduction",
                        "genome_sha256": genome_sha,
                        "environment_sha256": environment_sha,
                        "birth_export_receipt_sha256": birth_receipt_sha,
                        "birth_proof_checkpoint_state_sha256": checkpoint_state_sha,
                        "world_instance_sha256": hashlib.sha256(world_id.encode()).hexdigest(),
                        "source_body_id_sha256": hashlib.sha256(
                            source_body_id.encode()
                        ).hexdigest(),
                        "public_body": index,
                    },
                )
            )
        checkpoint_id = (
            f"life-checkpoint:{life_id}:{state['tick']}:"
            f"{checkpoint_file_sha}"
        )
        records.append(
            evidence_record(
                id=checkpoint_id,
                time={"domain": "model_tick", "value": int(state["tick"])},
                record_type="life_checkpoint",
                text="Authenticated whole-world checkpoint containing this research branch.",
                parents={root_id: "life_continuation"},
                blobs=[checkpoint_blob],
                fields={
                    "life_id": life_id,
                    "checkpoint_sha256": checkpoint_file_sha,
                    "checkpoint_state_sha256": checkpoint_state_sha,
                    "tick": int(state["tick"]),
                },
            )
        )

        observed_life_record_ids.append(checkpoint_id)

    candidate_records = {record["id"]: record for record in records}
    for record in records:
        for parent_id in record["parent_ids"]:
            if parent_id not in existing and parent_id not in candidate_records:
                raise PopulationEvidenceError(
                    f"living research evidence parent is absent: {parent_id}"
                )
    batch_digest = hashlib.sha256(canonical_bytes(records)).hexdigest()
    batch = {
        "format": BATCH_FORMAT,
        "schema_version": 1,
        "campaign_id": ledger["campaign_id"],
        "batch_id": f"population-campaign:{batch_digest}",
        "records": records,
        "sources": {
            "birth_export_receipt_sha256": birth_receipt_sha,
            "binding_content_sha256": binding_sha,
            "checkpoint_file_sha256": checkpoint_file_sha,
            "checkpoint_state_sha256": checkpoint_state_sha,
            "migration_receipt_sha256": migration_sha,
            "migration_receipt_file_sha256": migration_file_sha,
            "source_checkpoint_file_sha256": source_checkpoint_file_sha,
            "source_checkpoint_state_sha256": source_checkpoint_state_sha,
            "target_initial_checkpoint_file_sha256": target_initial_file_sha,
            "target_initial_checkpoint_state_sha256": target_initial_state_sha,
            "recording_content_sha256": recording["content_sha256"],
        },
    }
    duplicate = set(existing).intersection(candidate_records)
    if duplicate:
        if batch["batch_id"] not in ledger.get("applied_batches", []) or any(
            canonical_bytes(existing[record_id])
            != canonical_bytes(candidate_records[record_id])
            for record_id in duplicate
        ):
            raise PopulationEvidenceError(
                "living research life identity was already recorded outside this batch"
            )
    validate_records(
        [
            *ledger["records"],
            *(record for record in records if record["id"] not in existing),
        ],
        campaign_id=ledger["campaign_id"],
    )
    laws, event_laws = _law_ids(args)
    for record_id in {value for values in [laws, *event_laws.values()] for value in values}:
        law = existing.get(record_id)
        if law is None or law.get("record_type") != "gam_fit_attempt" or law.get(
            "fields", {}
        ).get("status") != "completed":
            raise PopulationEvidenceError(
                f"living recording law link is not a completed fit: {record_id}"
            )
    link = {
        "format": LINK_FORMAT,
        "campaign_id": ledger["campaign_id"],
        "campaign_record_id": campaign_id,
        "environment_record_id": environment_id,
        "body_life_record_ids": {
            str(index): record_id
            for index, record_id in enumerate(observed_life_record_ids)
        },
        "associated_law_fit_record_ids": laws,
        "event_law_fit_record_ids": event_laws,
        "recording_content_sha256": recording["content_sha256"],
        "birth_batch_id": batch["batch_id"],
        "sources": dict(batch["sources"]),
    }
    for target, value in ((args.output_batch, batch), (args.output_link, link)):
        if target.exists():
            if read_json(target) != value:
                raise PopulationEvidenceError(f"existing output differs: {target}")
        else:
            atomic_write_json(target, value)
    return {
        "birth_batch": str(args.output_batch),
        "birth_batch_id": batch["batch_id"],
        "link": str(args.output_link),
        "founders": len(founders),
        "offspring": len(residents) - len(founders),
        "lives": len(life_ids),
        "research_branches": len(source_life_ids),
        "source_checkpoint_tick": migration["tick"],
        "new_genomes": sum(record["record_type"] == "genome_candidate" for record in records),
        "new_environments": sum(record["record_type"] == "environment_candidate" for record in records),
        "checkpoint_tick": int(state["tick"]),
        "recording_content_sha256": recording["content_sha256"],
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--target-initial-checkpoint", required=True, type=Path)
    parser.add_argument("--migration-receipt", required=True, type=Path)
    parser.add_argument("--birth-export", required=True, type=Path)
    parser.add_argument("--associated-law-fit", action="append", default=[])
    parser.add_argument("--event-law", action="append", default=[])
    parser.add_argument("--output-batch", required=True, type=Path)
    parser.add_argument("--output-link", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
