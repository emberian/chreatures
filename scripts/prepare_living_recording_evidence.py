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

    checkpoint_manifest = validate_manifest(state.get("birth_manifest", {}))
    founders = founder_manifest["residents"]
    residents = checkpoint_manifest["residents"]
    if residents[: len(founders)] != founders:
        raise PopulationEvidenceError("checkpoint founders differ from cold-birth export")
    source_bodies = state.get("world", {}).get("bodies")
    if not isinstance(source_bodies, list) or len(source_bodies) != len(residents):
        raise PopulationEvidenceError("checkpoint body order differs from its birth manifest")
    body_rows = binding.get("bodies")
    if not isinstance(body_rows, list) or len(body_rows) != len(source_bodies):
        raise PopulationEvidenceError("private binding does not cover the checkpoint cohort")
    for index, (body, bound) in enumerate(zip(source_bodies, body_rows, strict=True)):
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
    life_ids = []
    for index, (bound, resident) in enumerate(zip(body_rows, residents, strict=True)):
        source_body_id = str(bound["source_body_id"])
        genome_sha = str(resident["candidate"]["sha256"])
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
        parents = {
            f"genome:{genome_sha}": "candidate_genome",
            environment_id: "environment",
        }
        birth_tick = 0
        birth_mode = "experimental_initialization"
        if index >= len(founders):
            hatch = hatch_parents.get(index)
            if hatch is None:
                raise PopulationEvidenceError(
                    "a non-founder resident lacks a captured hatching parent receipt"
                )
            parent_index, birth_tick = hatch
            if parent_index >= index:
                raise PopulationEvidenceError("hatching parent must precede its offspring")
            parents[f"birth:{life_ids[parent_index]}"] = "physical_parent_birth"
            birth_mode = "embodied_reproduction"
        birth_id = f"birth:{life_id}"
        records.append(
            evidence_record(
                id=birth_id,
                time={"domain": "model_tick", "value": birth_tick},
                record_type="birth",
                text=(
                    "Fresh research resident instantiated from an authenticated cold birth."
                    if birth_mode == "experimental_initialization"
                    else "Funded offspring committed by the recorded physical world."
                ),
                parents=parents,
                fields={
                    "life_id": life_id,
                    "birth_mode": birth_mode,
                    "genome_sha256": genome_sha,
                    "environment_sha256": environment_sha,
                    "birth_export_receipt_sha256": birth_receipt_sha,
                    "birth_proof_checkpoint_state_sha256": checkpoint_state_sha,
                    "world_instance_sha256": hashlib.sha256(world_id.encode()).hexdigest(),
                    "source_body_id_sha256": hashlib.sha256(source_body_id.encode()).hexdigest(),
                    "public_body": index,
                },
            )
        )
        records.append(
            evidence_record(
                id=(
                    f"life-checkpoint:{life_id}:{state['tick']}:"
                    f"{checkpoint_file_sha}"
                ),
                time={"domain": "model_tick", "value": int(state["tick"])},
                record_type="life_checkpoint",
                text="Authenticated whole-world checkpoint containing this research life.",
                parents={birth_id: "life_continuation"},
                blobs=[checkpoint_blob],
                fields={
                    "life_id": life_id,
                    "checkpoint_sha256": checkpoint_file_sha,
                    "checkpoint_state_sha256": checkpoint_state_sha,
                    "tick": int(state["tick"]),
                },
            )
        )

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
            str(index): f"birth:{life_id}" for index, life_id in enumerate(life_ids)
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
