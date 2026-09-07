#!/usr/bin/env python3
"""Bind a fresh canonical v8 life and actual v4 recording to its own campaign."""

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
    empty_ledger,
)
from chreatures.resident_birth import validate_manifest
from chreatures.population import CandidateGenome, PopulationSearch
from chreatures.population_launch import value_sha256


BINDING_FORMAT = "chreatures-living-recording-private-binding-v1"
BIRTH_EXPORT_FORMAT = "chreatures-population-birth-export-v1"
CHECKPOINT_FORMAT = "chreatures-developmental-habitat-checkpoint-v4"
LINK_FORMAT = "chreatures-living-recording-evidence-link-v2"
RECORDING_FORMAT = "chreatures-living-reef-public-recording-v4"


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PopulationEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


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
    receipt_sha = _hash(receipt.get("sha256"), "cold-birth receipt")
    body = dict(receipt)
    body.pop("sha256")
    if value_sha256(body) != receipt_sha:
        raise PopulationEvidenceError("cold-birth receipt content differs")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PopulationEvidenceError("cold-birth receipt lacks output hashes")
    for name in ("habitat.json", "biosphere.json", "resident-birth.json"):
        target = path / name
        if not target.is_file() or sha256_file(target) != outputs.get(name):
            raise PopulationEvidenceError(f"cold-birth output differs: {name}")
    manifest = validate_manifest(read_json(path / "resident-birth.json"))
    phenotypes = receipt.get("phenotypes")
    if not isinstance(phenotypes, list) or len(phenotypes) != len(manifest["residents"]):
        raise PopulationEvidenceError("birth phenotype receipts differ from founders")
    for index, (resident, phenotype) in enumerate(zip(manifest["residents"], phenotypes, strict=True)):
        target = (path / str(phenotype.get("local_path"))).resolve()
        if (
            phenotype.get("index") != index
            or phenotype.get("candidate_sha256") != resident["candidate"]["sha256"]
            or any(phenotype.get(key) != value for key, value in resident["neural_phenotype"].items())
            or not target.is_relative_to(path.resolve())
            or not target.is_file()
            or sha256_file(target) != phenotype.get("artifact_sha256")
        ):
            raise PopulationEvidenceError("birth neural phenotype artifact differs")
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


def _preparation(path: Path, birth: Mapping[str, Any], founders: Mapping[str, Any]):
    """Authenticate the fresh bank/assignment/search chain without claiming trials."""
    receipt = read_json(path / "receipt.json")
    if (
        receipt.get("format") != "chreatures-rich-collection-launch-v1"
        or receipt.get("version") != 1
        or value_sha256(dict(receipt, sha256="")) != receipt.get("sha256")
    ):
        raise PopulationEvidenceError("fresh collection preparation identity differs")
    files = {}
    for key, name in (
        ("profile", "profile.json"), ("founding_bank", "founding-bank.json"),
        ("founder_assignments", "founder-assignments.json"), ("search", "search.json"),
    ):
        target = path / name
        reference = receipt.get(key, {})
        if reference.get("path") != name or sha256_file(target) != reference.get("file_sha256"):
            raise PopulationEvidenceError(f"fresh preparation file differs: {name}")
        files[key] = read_json(target)
    bank, assignments, search = (
        files["founding_bank"], files["founder_assignments"], files["search"]
    )
    if (
        bank.get("format") != "chreatures-rich-collection-founding-bank-v1"
        or value_sha256(dict(bank, sha256="")) != bank.get("sha256")
        or bank.get("sha256") != receipt["founding_bank"].get("semantic_sha256")
        or assignments.get("format") != "chreatures-population-evaluation-assignments-v1"
        or value_sha256({k: v for k, v in assignments.items() if k != "sha256"})
        != assignments.get("sha256")
        or assignments.get("sha256") != receipt["founder_assignments"].get("semantic_sha256")
        or assignments.get("founding_bank_sha256") != bank.get("sha256")
        or bank.get("search_state_sha256") != receipt["search"].get("file_sha256")
        or value_sha256(search.get("config")) != search.get("config_sha256")
        or search.get("config_sha256") != bank.get("search_config_sha256")
        or search.get("config_sha256") != receipt["search"].get("config_sha256")
    ):
        raise PopulationEvidenceError("founding bank, assignments and search provenance differ")
    # Use the current native validator for registry content and parent identities.
    PopulationSearch(path / "search.json").validate()
    candidates = bank.get("candidates", [])
    if len(candidates) != bank.get("candidate_count") or any(
        search.get("genomes", {}).get(candidate.get("sha256")) != candidate
        for candidate in candidates
    ):
        raise PopulationEvidenceError("founding candidates differ from native search")
    for candidate in candidates:
        CandidateGenome(candidate)
    source = birth["source"]
    index = source.get("assignments", {}).get("world_index")
    worlds = assignments.get("worlds", [])
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(worlds):
        raise PopulationEvidenceError("birth assignment world index is invalid")
    selected = worlds[index]
    if (
        source["assignments"].get("file_sha256") != receipt["founder_assignments"]["file_sha256"]
        or source["assignments"].get("content_sha256") != assignments["sha256"]
        or selected.get("candidates") != [row["candidate"] for row in founders["residents"]]
        or selected.get("world_id") != birth["world"].get("assignment_world_id")
        or selected.get("environment") != birth["world"].get("environment")
        or selected.get("seed") != birth["world"].get("seed")
        or source["profile"].get("file_sha256") != receipt["profile"]["file_sha256"]
        or source["profile"].get("sha256") != bank.get("profile_sha256")
        or assignments.get("profile_sha256") != bank.get("profile_sha256")
        or assignments.get("resident_artifact_sha256") != bank["controller"].get("file_sha256")
        or source["resident_artifact"].get("file_sha256") != bank["controller"].get("file_sha256")
        or source["resident_artifact"].get("artifact_sha256") != bank["controller"].get("artifact_sha256")
    ):
        raise PopulationEvidenceError("canonical birth differs from fresh founding assignments")
    panel = receipt["search"].get("probe_panel")
    if (
        not isinstance(panel, Mapping)
        or value_sha256(panel) != search["config"].get("environment_probe_panel_sha256")
        or panel.get("controller_file_sha256") != bank["controller"].get("file_sha256")
    ):
        raise PopulationEvidenceError("fresh probe-policy provenance differs")
    return receipt, search, panel


def _blob_value(directory: Path, value: Any, role: str):
    digest = hashlib.sha256(canonical_bytes(value)).hexdigest()
    target = directory / f"{digest}.json"
    if target.exists():
        if read_json(target) != value:
            raise PopulationEvidenceError("existing provenance blob differs")
    else:
        atomic_write_json(target, value)
    return local_blob(target, role=role, media_type="application/json")


def _foundation(preparation, search, panel, campaign_id, blob_dir):
    config = search["config"]
    campaign_record_id = f"population-run:{campaign_id}"
    records = [evidence_record(
        id=campaign_record_id, time={"domain": "identity", "value": 0},
        record_type="population_run",
        text="Fresh embodied campaign from authenticated founding assignments; search pending rows are generation provenance, not executed evaluations.",
        blobs=[_blob_value(blob_dir, config, "search_config")],
        fields={"campaign_id": campaign_id, "search_config_sha256": search["config_sha256"],
                "search_state_format": search["format"],
                "preparation_receipt_sha256": preparation["sha256"]},
    )]
    recipe = {"descriptor_version": search["descriptor_version"],
              "descriptor_axes": config["descriptor_axes"],
              "source_search_config_sha256": search["config_sha256"]}
    recipe_blob = _blob_value(blob_dir, recipe, "descriptor_recipe")
    epoch_id = f"descriptor-epoch:{recipe_blob['sha256']}"
    records.append(evidence_record(
        id=epoch_id, time={"domain": "identity", "value": 0},
        record_type="descriptor_epoch", text="Declared fresh campaign physical descriptor recipe; no outcomes inferred.",
        parents={campaign_record_id: "campaign"}, blobs=[recipe_blob],
        fields={"descriptor_epoch_id": epoch_id, "descriptor_epoch_index": 0,
                "environment_epoch": config["environment_epoch"],
                "descriptor_recipe_sha256": recipe_blob["sha256"],
                "descriptor_dimension": len(config["descriptor_axes"])},
    ))
    panel_sha = value_sha256(panel)
    panel_id = f"probe-panel:{panel_sha}"
    records.append(evidence_record(
        id=panel_id, time={"domain": "identity", "value": 0},
        record_type="environment_probe_panel", text="Declared probe-policy panel from fresh preparation; no probe execution claimed.",
        parents={campaign_record_id: "campaign", epoch_id: "descriptor_epoch"},
        blobs=[_blob_value(blob_dir, panel, "probe_policy_panel")],
        fields={"probe_panel_id": panel_id, "probe_panel_sha256": panel_sha,
                "descriptor_epoch_id": epoch_id,
                "policy_artifact_sha256s": [panel["controller_file_sha256"]]},
    ))
    for genome in sorted(search["genomes"].values(), key=lambda item: item["sha256"]):
        records.append(genome_record_from_native(
            genome, campaign_record_id=campaign_record_id,
            time={"domain": "identity", "value": 0},
            artifact=_blob_value(blob_dir, genome, "genome_artifact"),
        ))
    for environment in sorted(search["environments"].values(), key=lambda item: item["sha256"]):
        records.append(environment_record_from_native(
            environment, campaign_record_id=campaign_record_id,
            probe_panel_record_id=panel_id, probe_panel_sha256=panel_sha,
            time={"domain": "identity", "value": 0},
            artifact=_blob_value(blob_dir, environment, "environment_artifact"),
        ))
    return records, campaign_record_id, panel_id, panel_sha


def build(args: argparse.Namespace) -> dict[str, Any]:
    recording = _recording(args.recording)
    binding, binding_sha = _binding(args.binding, recording)
    state, checkpoint_state_sha, checkpoint_file_sha = _checkpoint(args.checkpoint)
    receipt, founder_manifest, birth_receipt_sha = _birth_export(args.birth_export)
    preparation, search, panel = _preparation(args.preparation, receipt, founder_manifest)
    provenance = recording.get("provenance", {})
    capture_tool = provenance.get("capture_tool", {})
    world_id = binding.get("source_world_id")
    if not isinstance(world_id, str) or not world_id or any(
        binding.get(key) != provenance.get(key)
        for key in ("world_source_revision", "world_source_content_sha256", "physical_profile_sha256",
                    "graph_sha256", "resident_artifact_sha256")
    ) or any(binding.get(key) != capture_tool.get(other) for key, other in (
        ("capture_tool_revision", "revision"), ("capture_tool_file_sha256", "file_sha256")
    )) or binding.get("engine_identity_sha256") != provenance.get("engine_identity", {}).get("sha256"):
        raise PopulationEvidenceError("private binding and recording provenance differ")
    if (
        state.get("id") != world_id
        or recording.get("event_stream", {}).get("world_id") != world_id
        or state.get("execution_migrations") != []
        or state.get("engine_identity") != provenance.get("engine_identity")
        or state.get("engine_identity", {}).get("sha256") != binding["engine_identity_sha256"]
        or state.get("resident_controller", {}).get("format")
        != "chreatures-developmental-resident-population-snapshot-v8"
    ):
        raise PopulationEvidenceError("recording checkpoint is not this fresh current-engine life")
    tick = state.get("tick")
    observed_ticks = recording.get("sampling", {}).get("observed_ticks", [])
    if (isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
        or not observed_ticks or tick < max(observed_ticks)):
        raise PopulationEvidenceError("checkpoint precedes the recorded life")
    neural = state.get("neural_identity", {})
    graph = neural.get("graph_sha256") or neural.get("graph", {}).get("sha256")
    controller = state["resident_controller"].get("model_identity", {})
    if graph != binding["graph_sha256"] or controller.get("artifact_sha256") != binding["resident_artifact_sha256"]:
        raise PopulationEvidenceError("checkpoint neural/controller identity differs")
    source = receipt["source"]
    if (
        source["profile"].get("sha256") != binding["physical_profile_sha256"]
        or source["graph"].get("sha256") != binding["graph_sha256"]
        or source["resident_artifact"].get("artifact_sha256") != binding["resident_artifact_sha256"]
    ):
        raise PopulationEvidenceError("canonical birth and recorded mechanisms differ")
    residents = validate_manifest(state.get("birth_manifest", {}))["residents"]
    founders = founder_manifest["residents"]
    bodies = state.get("world", {}).get("bodies", [])
    birth_bodies = read_json(args.birth_export / "habitat.json").get("bodies", [])
    if (residents[:len(founders)] != founders or len(bodies) != len(residents)
        or len(birth_bodies) != len(founders) or len(binding["bodies"]) != len(bodies)
        or [body.get("id") for body in bodies[:len(founders)]] != [body.get("id") for body in birth_bodies]):
        raise PopulationEvidenceError("actual founder/body order differs from canonical birth")
    for index, (body, bound) in enumerate(zip(bodies, binding["bodies"], strict=True)):
        if bound.get("public_body") != index or bound.get("source_body_id") != body.get("id"):
            raise PopulationEvidenceError("public body binding differs from checkpoint")
    environment_receipt = receipt["world"].get("environment_receipt", {})
    environment = environment_receipt.get("environment_record", {})
    environment_sha = _hash(environment.get("sha256"), "birth environment")
    if (
        receipt["world"].get("assignment_world_id") != environment_sha
        or environment_receipt.get("environment_sha256") != environment_sha
        or search["environments"].get(environment_sha) != environment
    ):
        raise PopulationEvidenceError("birth environment differs from fresh search provenance")
    campaign_identity = {"preparation_sha256": preparation["sha256"],
                         "birth_export_sha256": birth_receipt_sha,
                         "world_id": world_id}
    campaign_id = "fresh-ecology-" + hashlib.sha256(canonical_bytes(campaign_identity)).hexdigest()
    description = "Fresh canonical embodied ecology; authenticated observed lives and events only."
    ledger = read_json(args.ledger) if args.ledger.exists() else empty_ledger(campaign_id, description)
    if ledger.get("format") != LEDGER_FORMAT or ledger.get("campaign_id") != campaign_id:
        raise PopulationEvidenceError("ledger is not the fresh birth's independent campaign")
    if ledger["records"]:
        validate_records(ledger["records"], campaign_id=campaign_id)
    existing = {record["id"]: record for record in ledger["records"]}
    foundation, run_id, _, _ = _foundation(
        preparation, search, panel, campaign_id, args.output_batch.parent / "blobs"
    )
    prior = read_json(args.output_batch) if args.output_batch.exists() else None
    prior_records = {record["id"]: record for record in (prior or {}).get("records", [])}
    records = []
    def add(record):
        previous = existing.get(record["id"])
        if previous is not None and canonical_bytes(previous) != canonical_bytes(record):
            raise PopulationEvidenceError(f"stable life/provenance record differs: {record['id']}")
        if previous is None or record["id"] in prior_records:
            records.append(record)
    for record in foundation:
        add(record)
    environment_id = f"environment:{environment_sha}"
    actual_genome_blob = local_blob(args.checkpoint, role="genome_artifact", media_type="application/json")
    checkpoint_blob = local_blob(args.checkpoint, role="life_checkpoint", media_type="application/json")
    hatch_parents = _hatch_parent_indices(recording)
    roots = []
    checkpoints = []
    known_genomes = {record["id"] for record in foundation}
    for index, (body, resident) in enumerate(zip(bodies, residents, strict=True)):
        genome = resident["candidate"]
        genome_id = f"genome:{genome['sha256']}"
        if genome_id not in known_genomes:
            add(genome_record_from_native(genome, campaign_record_id=run_id,
                time={"domain": "model_tick", "value": tick}, artifact=actual_genome_blob))
            known_genomes.add(genome_id)
        life_id = hashlib.sha256(canonical_bytes({
            "format": "chreatures-independent-research-life-v1", "world_id": world_id,
            "source_body_id": body["id"], "genome_sha256": genome["sha256"],
        })).hexdigest()
        birth_id = f"birth:{life_id}"
        roots.append(birth_id)
        parents = {genome_id: "candidate_genome", environment_id: "environment"}
        birth_tick = 0
        mode = "experimental_initialization"
        if index >= len(founders):
            mode = "embodied_reproduction"
            hatch = hatch_parents.get(index)
            if hatch is None:
                previous = existing.get(birth_id)
                if previous is None:
                    raise PopulationEvidenceError("offspring lacks an authenticated captured hatching event")
                birth_tick = previous["time"]["value"]
                parents = dict(previous["fields"]["parent_roles"])
            else:
                parent_index, birth_tick = hatch
                if not 0 <= parent_index < index:
                    raise PopulationEvidenceError("hatching parent must precede its offspring")
                parents[roots[parent_index]] = "physical_parent_birth"
        add(evidence_record(
            id=birth_id, time={"domain": "model_tick", "value": birth_tick}, record_type="birth",
            text=("Founder instantiated from authenticated fresh canonical birth." if index < len(founders)
                  else "Funded offspring committed by the recorded physical world."),
            parents=parents,
            fields={"life_id": life_id, "birth_mode": mode, "genome_sha256": genome["sha256"],
                    "environment_sha256": environment_sha, "birth_export_receipt_sha256": birth_receipt_sha,
                    "world_instance_sha256": hashlib.sha256(world_id.encode()).hexdigest(),
                    "source_body_id_sha256": hashlib.sha256(body["id"].encode()).hexdigest(), "public_body": index},
        ))
        checkpoint_id = f"life-checkpoint:{life_id}:{tick}:{checkpoint_file_sha}"
        if checkpoint_id in existing:
            checkpoint_record = existing[checkpoint_id]
        else:
            prior_checkpoints = [record for record in ledger["records"]
                if record["record_type"] == "life_checkpoint" and record["fields"].get("life_id") == life_id]
            predecessor = max(prior_checkpoints, key=lambda record: record["fields"]["tick"], default=None)
            if predecessor is not None and predecessor["fields"]["tick"] >= tick:
                raise PopulationEvidenceError("new life checkpoint does not advance its continuation")
            checkpoint_record = evidence_record(
                id=checkpoint_id, time={"domain": "model_tick", "value": tick}, record_type="life_checkpoint",
                text="Authenticated whole-world checkpoint containing this independently born life.",
                parents={(predecessor["id"] if predecessor else birth_id): "life_continuation"},
                blobs=[checkpoint_blob], fields={"life_id": life_id, "checkpoint_sha256": checkpoint_file_sha,
                    "checkpoint_state_sha256": checkpoint_state_sha, "tick": tick},
            )
        add(checkpoint_record)
        checkpoints.append(checkpoint_id)
    validate_records([*ledger["records"], *(record for record in records if record["id"] not in existing)], campaign_id=campaign_id)
    sources = {"preparation_receipt_sha256": preparation["sha256"], "birth_export_receipt_sha256": birth_receipt_sha,
               "binding_content_sha256": binding_sha, "checkpoint_file_sha256": checkpoint_file_sha,
               "checkpoint_state_sha256": checkpoint_state_sha, "recording_content_sha256": recording["content_sha256"]}
    batch_id = f"population-campaign:{hashlib.sha256(canonical_bytes(records)).hexdigest()}"
    batch = {"format": BATCH_FORMAT, "schema_version": 1, "campaign_id": campaign_id,
             "batch_id": batch_id, "records": records, "sources": sources}
    if prior is not None and prior != batch:
        raise PopulationEvidenceError("existing prepared birth batch differs")
    laws, event_laws = _law_ids(args)
    for record_id in {value for values in [laws, *event_laws.values()] for value in values}:
        law = existing.get(record_id)
        if law is None or law["record_type"] != "gam_fit_attempt" or law["fields"].get("status") != "completed":
            raise PopulationEvidenceError(f"law link is not a completed fit in this campaign: {record_id}")
    link = {"format": LINK_FORMAT, "campaign_id": campaign_id, "campaign_record_id": run_id,
            "environment_record_id": environment_id,
            "body_life_record_ids": {str(index): record_id for index, record_id in enumerate(checkpoints)},
            "associated_law_fit_record_ids": laws, "event_law_fit_record_ids": event_laws,
            "recording_content_sha256": recording["content_sha256"], "birth_batch_id": batch_id, "sources": sources}
    for target, value in ((args.output_batch, batch), (args.output_link, link)):
        if target.exists():
            if read_json(target) != value:
                raise PopulationEvidenceError(f"existing output differs: {target}")
        else:
            atomic_write_json(target, value)
    return {"campaign_id": campaign_id, "description": description, "birth_batch": str(args.output_batch),
            "birth_batch_id": batch_id, "link": str(args.output_link), "founders": len(founders),
            "offspring": len(residents) - len(founders), "checkpoint_tick": tick,
            "new_records": len(records), "recording_content_sha256": recording["content_sha256"],
            "next_step": "Apply the batch with build_population_weave.py using this campaign_id and description; then link the actual v4 recording."}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", required=True, type=Path)
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
