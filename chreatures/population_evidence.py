"""Typed, appendable evidence records for population-search campaigns.

This module builds the external evidence ledger consumed by the native Universal
Weave adapter.  It does not own search, mutate a live world, or expose the ledger
to an organism.  Large state remains in content-addressed blobs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "chreatures-population-evidence-v1"
LEDGER_FORMAT = "chreatures-population-evidence-ledger-v1"
BATCH_FORMAT = "chreatures-population-evidence-batch-v1"

RECORD_TYPES = frozenset(
    {
        "population_run",
        "descriptor_epoch",
        "environment_probe_panel",
        "genome_candidate",
        "environment_candidate",
        "birth",
        "life_checkpoint",
        "evaluation_completed",
        "evaluation_failed",
        "archive_decision",
        "transfer_trial",
        "population_snapshot",
        "gam_fit_attempt",
    }
)

_ROLE_RULES: dict[str, dict[str, tuple[frozenset[str], int, int | None]]] = {
    "population_run": {},
    "descriptor_epoch": {
        "campaign": (frozenset({"population_run"}), 1, 1),
        "previous_descriptor_epoch": (frozenset({"descriptor_epoch"}), 0, 1),
    },
    "environment_probe_panel": {
        "campaign": (frozenset({"population_run"}), 1, 1),
        "descriptor_epoch": (frozenset({"descriptor_epoch"}), 1, 1),
    },
    "genome_candidate": {
        "campaign": (frozenset({"population_run"}), 1, 1),
        "genome_parent": (frozenset({"genome_candidate"}), 0, 2),
        "inherited_law_fit": (frozenset({"gam_fit_attempt"}), 0, None),
    },
    "environment_candidate": {
        "campaign": (frozenset({"population_run"}), 1, 1),
        "probe_panel": (frozenset({"environment_probe_panel"}), 1, 1),
        "environment_parent": (frozenset({"environment_candidate"}), 0, 2),
    },
    "birth": {
        "candidate_genome": (frozenset({"genome_candidate"}), 1, 1),
        "environment": (frozenset({"environment_candidate"}), 1, 1),
        "physical_parent_birth": (frozenset({"birth"}), 0, 2),
    },
    "life_checkpoint": {
        "life_continuation": (frozenset({"birth", "life_checkpoint"}), 1, 1),
    },
    "evaluation_completed": {
        "life_continuation": (frozenset({"birth", "life_checkpoint"}), 1, 1),
        "candidate_genome": (frozenset({"genome_candidate"}), 1, 1),
        "environment": (frozenset({"environment_candidate"}), 1, 1),
        "descriptor_epoch": (frozenset({"descriptor_epoch"}), 1, 1),
        "probe_panel": (frozenset({"environment_probe_panel"}), 1, 1),
    },
    "evaluation_failed": {
        "life_continuation": (frozenset({"birth", "life_checkpoint"}), 0, 1),
        "planned_campaign": (frozenset({"population_run"}), 0, 1),
        "candidate_genome": (frozenset({"genome_candidate"}), 1, 1),
        "environment": (frozenset({"environment_candidate"}), 1, 1),
        "descriptor_epoch": (frozenset({"descriptor_epoch"}), 1, 1),
        "probe_panel": (frozenset({"environment_probe_panel"}), 1, 1),
    },
    "archive_decision": {
        "evaluated_candidate": (
            frozenset({"evaluation_completed", "evaluation_failed"}),
            1,
            1,
        ),
        "descriptor_epoch": (frozenset({"descriptor_epoch"}), 1, 1),
    },
    "transfer_trial": {
        "source_evaluation": (
            frozenset({"evaluation_completed", "evaluation_failed"}),
            1,
            1,
        ),
        "target_evaluation": (
            frozenset({"evaluation_completed", "evaluation_failed"}),
            1,
            1,
        ),
        "candidate_genome": (frozenset({"genome_candidate"}), 1, 1),
        "target_environment": (frozenset({"environment_candidate"}), 1, 1),
        "probe_panel": (frozenset({"environment_probe_panel"}), 1, 1),
    },
    "population_snapshot": {
        "campaign": (frozenset({"population_run"}), 1, 1),
        "archive_decision": (frozenset({"archive_decision"}), 0, None),
    },
    "gam_fit_attempt": {
        "source_evaluation": (
            frozenset({"evaluation_completed", "evaluation_failed"}),
            1,
            None,
        ),
    },
}

_REQUIRED_BLOBS = {
    "population_run": frozenset({"search_config"}),
    "descriptor_epoch": frozenset({"descriptor_recipe"}),
    "environment_probe_panel": frozenset({"probe_policy_panel"}),
    "genome_candidate": frozenset({"genome_artifact"}),
    "environment_candidate": frozenset({"environment_artifact"}),
    "life_checkpoint": frozenset({"life_checkpoint"}),
    "evaluation_completed": frozenset({"evaluation_result", "evaluation_trace"}),
    "evaluation_failed": frozenset({"evaluation_result", "evaluation_trace"}),
    "population_snapshot": frozenset({"population_search_state"}),
    "gam_fit_attempt": frozenset({"gam_fit_report"}),
}

_HASH_FIELDS = {
    "population_run": ("search_config_sha256",),
    "descriptor_epoch": ("descriptor_recipe_sha256",),
    "environment_probe_panel": ("probe_panel_sha256",),
    "genome_candidate": (
        "genome_sha256",
        "graph_sha256",
        "port_spec_sha256",
        "base_controller_sha256",
        "developmental_base_sha256",
        "population_adapter_bank_sha256",
        "organism_interface_sha256",
        "variation_recipe_sha256",
    ),
    "environment_candidate": (
        "environment_sha256",
        "topology_sha256",
        "resource_sha256",
        "profile_sha256",
        "variation_recipe_sha256",
    ),
    "life_checkpoint": ("checkpoint_sha256",),
    "evaluation_completed": ("trajectory_sha256",),
    "evaluation_failed": ("trajectory_sha256",),
}

_PRIVATE_GENOME_TOKENS = frozenset(
    {"state", "memory", "optimizer", "rng", "history", "checkpoint", "rates"}
)


class PopulationEvidenceError(ValueError):
    """A population evidence batch violates the typed ledger contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blob_ref(
    *,
    role: str,
    sha256: str,
    bytes: int | None = None,
    media_type: str | None = None,
    verification: str = "reported_by_authenticated_source",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "uri": f"urn:sha256:{sha256}",
        "sha256": sha256,
        "verification": verification,
    }
    if bytes is not None:
        result["bytes"] = bytes
    if media_type is not None:
        result["media_type"] = media_type
    _validate_blob(result, "constructed blob")
    return result


def local_blob(path: Path, *, role: str, media_type: str) -> dict[str, Any]:
    return blob_ref(
        role=role,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        media_type=media_type,
        verification="verified_local_sha256",
    )


def evidence_record(
    *,
    id: str,
    time: Mapping[str, Any],
    record_type: str,
    text: str,
    parents: Mapping[str, str] | None = None,
    fields: Mapping[str, Any] | None = None,
    blobs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    parent_roles = dict(parents or {})
    body = dict(fields or {})
    if "parent_roles" in body:
        raise PopulationEvidenceError("parent_roles is supplied through parents")
    body["parent_roles"] = parent_roles
    return {
        "id": id,
        "time": dict(time),
        "record_type": record_type,
        "text": text,
        "parent_ids": list(parent_roles),
        "blob_refs": [dict(blob) for blob in blobs],
        "fields": body,
    }


def empty_ledger(campaign_id: str, description: str) -> dict[str, Any]:
    if not campaign_id.strip() or not description.strip():
        raise PopulationEvidenceError("campaign_id and description must be nonempty")
    return {
        "format": LEDGER_FORMAT,
        "schema_version": 1,
        "campaign_id": campaign_id,
        "description": description,
        "records": [],
        "applied_batches": [],
    }


def genome_record_from_native(
    genome: Mapping[str, Any],
    *,
    campaign_record_id: str,
    time: Mapping[str, Any],
    artifact: Mapping[str, Any],
    inherited_law_fit_record_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Normalize one authenticated population-core Genome without private state."""
    digest = genome.get("sha256")
    _valid_hash(digest, "native genome sha256")
    parents = genome.get("parents")
    variation = genome.get("variation")
    values = genome.get("values")
    if not isinstance(parents, list) or len(parents) > 2:
        raise PopulationEvidenceError("native genome parents must contain zero to two hashes")
    for parent in parents:
        _valid_hash(parent, "native parent genome")
    if not isinstance(variation, Mapping) or not isinstance(values, Mapping):
        raise PopulationEvidenceError("native genome lacks variation or values")
    for name in [*values, *variation.get("mutated", [])]:
        tokens = str(name).lower().replace("-", "_").split("_")
        if any(token in _PRIVATE_GENOME_TOKENS for token in tokens):
            raise PopulationEvidenceError(
                f"native genome parameter {name!r} crosses the private-state boundary"
            )
    parent_roles = {campaign_record_id: "campaign"}
    parent_roles.update({f"genome:{parent}": "genome_parent" for parent in parents})
    parent_roles.update({record_id: "inherited_law_fit" for record_id in inherited_law_fit_record_ids})
    return evidence_record(
        id=f"genome:{digest}",
        time=time,
        record_type="genome_candidate",
        text="Immutable population-search genome candidate.",
        parents=parent_roles,
        blobs=[artifact],
        fields={
            "genome_sha256": digest,
            "parent_genome_sha256s": list(parents),
            "graph_sha256": genome.get("graph_sha256"),
            "port_spec_sha256": genome.get("port_spec_sha256"),
            "base_controller_sha256": genome.get("base_controller_sha256"),
            "developmental_base_sha256": genome.get("developmental_base_sha256"),
            "population_adapter_bank_sha256": genome.get(
                "population_adapter_bank_sha256"
            ),
            "organism_interface_sha256": genome.get("organism_interface_sha256"),
            "policy_adapter_count": genome.get("policy_adapter_count"),
            "policy_adapter_rank": genome.get("policy_adapter_rank"),
            "variation_operator": variation.get("operator"),
            "variation_seed": variation.get("seed"),
            "variation_recipe_sha256": variation.get("recipe_sha256"),
            "inherited_law_fit_ids": list(inherited_law_fit_record_ids),
            "mutated_parameters": list(variation.get("mutated", [])),
            "parameter_values": dict(values),
        },
    )


def environment_record_from_native(
    environment: Mapping[str, Any],
    *,
    campaign_record_id: str,
    probe_panel_record_id: str,
    probe_panel_sha256: str,
    time: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one authenticated population-core EnvironmentGenome."""
    digest = environment.get("sha256")
    _valid_hash(digest, "native environment sha256")
    parents = environment.get("parents")
    variation = environment.get("variation")
    if not isinstance(parents, list) or len(parents) > 2:
        raise PopulationEvidenceError("native environment parents must contain zero to two hashes")
    for parent in parents:
        _valid_hash(parent, "native parent environment")
    if not isinstance(variation, Mapping):
        raise PopulationEvidenceError("native environment lacks variation")
    parent_roles = {campaign_record_id: "campaign", probe_panel_record_id: "probe_panel"}
    parent_roles.update(
        {f"environment:{parent}": "environment_parent" for parent in parents}
    )
    return evidence_record(
        id=f"environment:{digest}",
        time=time,
        record_type="environment_candidate",
        text="Immutable physical-environment candidate.",
        parents=parent_roles,
        blobs=[artifact],
        fields={
            "environment_sha256": digest,
            "parent_environment_sha256s": list(parents),
            "topology_sha256": environment.get("topology_sha256"),
            "resource_sha256": environment.get("resource_sha256"),
            "profile_sha256": environment.get("profile_sha256"),
            "environment_epoch": environment.get("epoch"),
            "variation_operator": variation.get("operator"),
            "variation_seed": variation.get("seed"),
            "variation_recipe_sha256": variation.get("recipe_sha256"),
            "probe_panel_sha256": probe_panel_sha256,
        },
    )


def evaluation_records_from_native(
    evaluation: Mapping[str, Any],
    *,
    life_id: str,
    continuation_record_id: str | None,
    campaign_record_id: str | None,
    allocated: bool,
    descriptor_epoch_record_id: str,
    descriptor_epoch_id: str,
    probe_panel_record_id: str,
    probe_panel_sha256: str,
    time: Mapping[str, Any],
    result_artifact: Mapping[str, Any],
    trace_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize a terminal population-core Evaluation and its archive decision.

    The coordinator supplies the life and provenance links because population-core
    intentionally has no wall clock, birth state, or checkpoint ownership.
    """
    evaluation_id = evaluation.get("evaluation_sha256")
    genome_sha = evaluation.get("candidate_sha256")
    environment_sha = evaluation.get("environment_sha256")
    for name, digest in (
        ("evaluation", evaluation_id),
        ("candidate", genome_sha),
        ("environment", environment_sha),
    ):
        _valid_hash(digest, f"native {name} sha256")
    native_status = evaluation.get("status")
    if native_status not in {"success", "failure"}:
        raise PopulationEvidenceError("native evaluation status must be success or failure")
    if native_status == "success" and not allocated:
        raise PopulationEvidenceError("a completed evaluation must have allocated a life")
    if allocated != (continuation_record_id is not None) or allocated == (
        campaign_record_id is not None
    ):
        raise PopulationEvidenceError("evaluation allocation proof edges are inconsistent")
    if evaluation.get("life_id") != life_id:
        raise PopulationEvidenceError("native evaluation life_id differs from campaign life")
    trajectory_sha256 = evaluation.get("trajectory_sha256")
    _valid_hash(trajectory_sha256, "native trajectory sha256")
    if trace_artifact.get("sha256") != trajectory_sha256:
        raise PopulationEvidenceError("trace blob differs from native evaluation")
    record_type = "evaluation_completed" if native_status == "success" else "evaluation_failed"
    fields: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "life_id": life_id,
        "genome_sha256": genome_sha,
        "environment_sha256": environment_sha,
        "status": "completed" if native_status == "success" else "failed",
        "allocation_status": "allocated" if allocated else "not_allocated",
        "failure": evaluation.get("failure", ""),
        "metrics": dict(evaluation.get("metrics", {})),
        "evaluation_seed": evaluation.get("evaluation_seed"),
        "committed_ticks": evaluation.get("committed_ticks"),
        "trajectory_sha256": trajectory_sha256,
        "descriptor_epoch_id": descriptor_epoch_id,
        "probe_panel_sha256": probe_panel_sha256,
    }
    if native_status == "success":
        fields.update(
            descriptor=list(evaluation.get("descriptor", [])),
            cell=list(evaluation.get("cell", [])),
            quality=evaluation.get("quality"),
        )
    terminal_id = f"evaluation:{evaluation_id}:{fields['status']}"
    parents = {
        f"genome:{genome_sha}": "candidate_genome",
        f"environment:{environment_sha}": "environment",
        descriptor_epoch_record_id: "descriptor_epoch",
        probe_panel_record_id: "probe_panel",
    }
    parents[
        continuation_record_id if allocated else campaign_record_id
    ] = "life_continuation" if allocated else "planned_campaign"
    terminal = evidence_record(
        id=terminal_id,
        time=time,
        record_type=record_type,
        text=(
            "Complete physical population evaluation."
            if native_status == "success"
            else "Retained failed physical population evaluation."
        ),
        parents=parents,
        blobs=[result_artifact, trace_artifact],
        fields=fields,
    )
    retained = bool(evaluation.get("archive_retained", False))
    decision = evidence_record(
        id=f"archive-decision:{evaluation_id}",
        time=time,
        record_type="archive_decision",
        text="Native quality-diversity archive admission result.",
        parents={
            terminal_id: "evaluated_candidate",
            descriptor_epoch_record_id: "descriptor_epoch",
        },
        fields={
            "evaluation_id": evaluation_id,
            "decision": "retained" if retained else "rejected",
            "descriptor_epoch_id": descriptor_epoch_id,
            "cell": list(evaluation.get("cell") or []),
            "quality": evaluation.get("quality") if native_status == "success" else None,
            "failure_retained": native_status == "failure",
        },
    )
    return terminal, decision


def append_batches(
    ledger: Mapping[str, Any], batches: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    result = deepcopy(dict(ledger))
    _validate_ledger_envelope(result)
    applied = set(result["applied_batches"])
    existing = {record["id"]: record for record in result["records"]}
    for batch in batches:
        _validate_batch_envelope(batch, result["campaign_id"])
        batch_id = batch["batch_id"]
        if batch_id in applied:
            for record in batch["records"]:
                previous = existing.get(record["id"])
                if previous is None or canonical_bytes(previous) != canonical_bytes(record):
                    raise PopulationEvidenceError(
                        f"applied batch {batch_id!r} differs from ledger record "
                        f"{record['id']!r}"
                    )
            continue
        copied = deepcopy(batch["records"])
        result["records"].extend(copied)
        result["applied_batches"].append(batch_id)
        applied.add(batch_id)
        existing.update({record["id"]: record for record in copied})
    validate_records(result["records"], campaign_id=result["campaign_id"])
    result["records_sha256"] = hashlib.sha256(
        canonical_bytes(result["records"])
    ).hexdigest()
    return result


def reconcile_population_state(
    records: Sequence[Mapping[str, Any]], state: Mapping[str, Any]
) -> dict[str, int]:
    """Require every terminal result in a native search snapshot to be retained."""
    if state.get("format") != "chreatures-population-search-v1":
        raise PopulationEvidenceError("unsupported native population search state")
    evaluations = state.get("evaluations")
    if not isinstance(evaluations, list):
        raise PopulationEvidenceError("native population state lacks evaluations")
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise PopulationEvidenceError("native population state lacks search config")
    config_sha256 = state.get("config_sha256")
    probe_panel_sha256 = config.get("environment_probe_panel_sha256")
    environment_epoch = config.get("environment_epoch")
    _valid_hash(config_sha256, "native search config identity")
    if hashlib.sha256(canonical_bytes(config)).hexdigest() != config_sha256:
        raise PopulationEvidenceError("native search config content hash differs")
    _valid_hash(probe_panel_sha256, "native environment probe panel identity")
    if isinstance(environment_epoch, bool) or not isinstance(environment_epoch, int):
        raise PopulationEvidenceError("native environment epoch is invalid")
    run = next(record for record in records if record["record_type"] == "population_run")
    if run["fields"].get("search_config_sha256") != config_sha256:
        raise PopulationEvidenceError("population run differs from native search config")
    terminals: dict[str, Mapping[str, Any]] = {}
    decisions: dict[str, Mapping[str, Any]] = {}
    environments: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("record_type") in {"evaluation_completed", "evaluation_failed"}:
            terminals[record.get("fields", {}).get("evaluation_id")] = record
        elif record.get("record_type") == "archive_decision":
            decisions[record.get("fields", {}).get("evaluation_id")] = record
        elif record.get("record_type") == "environment_candidate":
            environments[record.get("fields", {}).get("environment_sha256")] = record
    failures = 0
    seen_evaluations: set[str] = set()
    for raw in evaluations:
        if not isinstance(raw, Mapping):
            raise PopulationEvidenceError("native population evaluation is not an object")
        evaluation_id = raw.get("evaluation_sha256")
        _valid_hash(evaluation_id, "native evaluation identity")
        if evaluation_id in seen_evaluations:
            raise PopulationEvidenceError(
                f"native population state repeats evaluation {evaluation_id}"
            )
        seen_evaluations.add(evaluation_id)
        status = raw.get("status")
        if status not in {"success", "failure"}:
            raise PopulationEvidenceError(
                f"native evaluation {evaluation_id} has invalid status"
            )
        terminal = terminals.get(evaluation_id)
        expected_type = "evaluation_completed" if status == "success" else "evaluation_failed"
        if terminal is None or terminal.get("record_type") != expected_type:
            raise PopulationEvidenceError(
                f"native evaluation {evaluation_id} is absent from terminal evidence"
            )
        fields = terminal["fields"]
        if (
            fields.get("life_id") != raw.get("life_id")
            or fields.get("genome_sha256") != raw.get("candidate_sha256")
            or fields.get("environment_sha256") != raw.get("environment_sha256")
            or fields.get("trajectory_sha256") != raw.get("trajectory_sha256")
            or fields.get("probe_panel_sha256") != probe_panel_sha256
        ):
            raise PopulationEvidenceError(
                f"native evaluation {evaluation_id} identity differs from terminal evidence"
            )
        environment = environments.get(raw.get("environment_sha256"))
        if environment is None or environment["fields"].get("environment_epoch") != environment_epoch:
            raise PopulationEvidenceError(
                f"native evaluation {evaluation_id} environment epoch is absent or differs"
            )
        retained = raw.get("archive_retained")
        if not isinstance(retained, bool):
            raise PopulationEvidenceError(
                f"native evaluation {evaluation_id} lacks boolean archive_retained"
            )
        decision = decisions.get(evaluation_id)
        expected_decision = "retained" if retained else "rejected"
        if decision is None or decision.get("fields", {}).get("decision") != expected_decision:
            raise PopulationEvidenceError(
                f"native evaluation {evaluation_id} archive decision is absent or differs"
            )
        failures += int(status == "failure")
    return {"evaluations": len(evaluations), "failed_evaluations": failures}


def weave_request(ledger: Mapping[str, Any]) -> dict[str, Any]:
    _validate_ledger_envelope(ledger)
    validate_records(ledger["records"], campaign_id=ledger["campaign_id"])
    return {
        "archive_id": ledger["campaign_id"],
        "description": ledger["description"],
        "evidence_schema": SCHEMA,
        "evidence": deepcopy(ledger["records"]),
    }


def validate_records(records: Sequence[Mapping[str, Any]], *, campaign_id: str) -> None:
    if not records:
        raise PopulationEvidenceError("population ledger has no records")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise PopulationEvidenceError("every record requires a nonempty string id")
        if record_id in by_id:
            raise PopulationEvidenceError(f"duplicate record id {record_id}")
        by_id[record_id] = record

    campaign_nodes = [r for r in records if r.get("record_type") == "population_run"]
    if len(campaign_nodes) != 1:
        raise PopulationEvidenceError("ledger requires exactly one population_run")
    if campaign_nodes[0].get("fields", {}).get("campaign_id") != campaign_id:
        raise PopulationEvidenceError("population_run campaign_id differs from ledger")

    continuations: Counter[str] = Counter()
    terminal_evaluations: dict[str, str] = {}
    archive_decisions: dict[str, str] = {}
    epoch_indices: dict[int, Mapping[str, Any]] = {}
    epoch_ids: set[str] = set()
    panel_ids: set[str] = set()
    life_ids: set[str] = set()

    for record in records:
        _validate_record(record, by_id)
        fields = record["fields"]
        record_type = record["record_type"]
        roles = fields["parent_roles"]
        if "life_continuation" in roles.values():
            continuation = next(k for k, v in roles.items() if v == "life_continuation")
            continuations[continuation] += 1
            if fields["life_id"] != by_id[continuation]["fields"]["life_id"]:
                raise PopulationEvidenceError(
                    f"{record['id']} changes life_id across a continuation edge"
                )
        if record_type in {"evaluation_completed", "evaluation_failed"}:
            evaluation_id = _string(fields, "evaluation_id", record["id"])
            if evaluation_id in terminal_evaluations:
                raise PopulationEvidenceError(
                    f"evaluation {evaluation_id} has more than one terminal record"
                )
            terminal_evaluations[evaluation_id] = record["id"]
        elif record_type == "archive_decision":
            evaluation_id = _string(fields, "evaluation_id", record["id"])
            if evaluation_id in archive_decisions:
                raise PopulationEvidenceError(
                    f"evaluation {evaluation_id} has more than one archive decision"
                )
            archive_decisions[evaluation_id] = record["id"]
        elif record_type == "descriptor_epoch":
            index = _integer(fields, "descriptor_epoch_index", record["id"])
            epoch_id = _string(fields, "descriptor_epoch_id", record["id"])
            if index in epoch_indices or epoch_id in epoch_ids:
                raise PopulationEvidenceError("descriptor epoch identity is duplicated")
            epoch_indices[index] = record
            epoch_ids.add(epoch_id)
        elif record_type == "environment_probe_panel":
            panel_id = _string(fields, "probe_panel_id", record["id"])
            if panel_id in panel_ids:
                raise PopulationEvidenceError("probe panel identity is duplicated")
            panel_ids.add(panel_id)
        elif record_type == "birth":
            life_id = _string(fields, "life_id", record["id"])
            if life_id in life_ids:
                raise PopulationEvidenceError(f"life_id {life_id} has more than one birth")
            life_ids.add(life_id)

    branched = [source for source, count in continuations.items() if count > 1]
    if branched:
        raise PopulationEvidenceError(
            f"life continuation branches at {', '.join(sorted(branched))}"
        )
    if set(archive_decisions) != set(terminal_evaluations):
        missing = sorted(set(terminal_evaluations) - set(archive_decisions))
        unknown = sorted(set(archive_decisions) - set(terminal_evaluations))
        raise PopulationEvidenceError(
            f"terminal/archive decision mismatch; missing={missing}, unknown={unknown}"
        )
    for index, epoch in epoch_indices.items():
        previous = _parents_with_role(epoch, "previous_descriptor_epoch")
        if index == 0 and previous:
            raise PopulationEvidenceError("descriptor epoch zero cannot have a predecessor")
        if index > 0:
            if len(previous) != 1:
                raise PopulationEvidenceError(
                    f"descriptor epoch {index} requires its immediate predecessor"
                )
            if by_id[previous[0]]["fields"].get("descriptor_epoch_index") != index - 1:
                raise PopulationEvidenceError(
                    f"descriptor epoch {index} predecessor is not epoch {index - 1}"
                )


def _validate_record(record: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]) -> None:
    record_id = str(record["id"])
    record_type = record.get("record_type")
    if record_type not in RECORD_TYPES:
        raise PopulationEvidenceError(f"{record_id} has unsupported record_type {record_type!r}")
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        raise PopulationEvidenceError(f"{record_id} requires nonempty text")
    time = record.get("time")
    if not isinstance(time, Mapping) or not isinstance(time.get("domain"), str):
        raise PopulationEvidenceError(f"{record_id} requires a typed time domain")
    value = time.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PopulationEvidenceError(f"{record_id} has invalid time value")
    parents = record.get("parent_ids")
    fields = record.get("fields")
    blobs = record.get("blob_refs")
    if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
        raise PopulationEvidenceError(f"{record_id} parent_ids must be strings")
    if len(set(parents)) != len(parents):
        raise PopulationEvidenceError(f"{record_id} repeats a parent")
    if not isinstance(fields, Mapping) or not isinstance(fields.get("parent_roles"), Mapping):
        raise PopulationEvidenceError(f"{record_id} requires fields.parent_roles")
    roles = fields["parent_roles"]
    if set(roles) != set(parents) or any(not isinstance(v, str) for v in roles.values()):
        raise PopulationEvidenceError(
            f"{record_id} parent_roles keys must exactly match parent_ids"
        )
    if not isinstance(blobs, list):
        raise PopulationEvidenceError(f"{record_id} blob_refs must be an array")
    for blob in blobs:
        _validate_blob(blob, record_id)
    blob_roles = [blob["role"] for blob in blobs]
    if len(set(blob_roles)) != len(blob_roles):
        raise PopulationEvidenceError(f"{record_id} repeats a blob role")
    missing_blob_roles = _REQUIRED_BLOBS.get(record_type, frozenset()) - set(blob_roles)
    if missing_blob_roles:
        raise PopulationEvidenceError(
            f"{record_id} lacks blob roles {sorted(missing_blob_roles)}"
        )

    allowed = _ROLE_RULES[record_type]
    counts = Counter(roles.values())
    unknown_roles = set(counts) - set(allowed)
    if unknown_roles:
        raise PopulationEvidenceError(f"{record_id} has invalid parent roles {sorted(unknown_roles)}")
    for role, (parent_types, minimum, maximum) in allowed.items():
        count = counts[role]
        if count < minimum or (maximum is not None and count > maximum):
            high = "unbounded" if maximum is None else str(maximum)
            raise PopulationEvidenceError(
                f"{record_id} requires {minimum}..{high} parent(s) for role {role}"
            )
        for parent_id in _parents_with_role(record, role):
            parent = by_id.get(parent_id)
            if parent is None:
                raise PopulationEvidenceError(f"{record_id} parent {parent_id} is absent")
            if parent.get("record_type") not in parent_types:
                raise PopulationEvidenceError(
                    f"{record_id} role {role} cannot target {parent.get('record_type')}"
                )

    for field in _HASH_FIELDS.get(record_type, ()):
        _hash(fields, field, record_id)
    _validate_type_fields(record, by_id)


def _validate_type_fields(
    record: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> None:
    record_id = str(record["id"])
    record_type = str(record["record_type"])
    fields = record["fields"]
    if record_type == "descriptor_epoch":
        index = _integer(fields, "descriptor_epoch_index", record_id)
        if index < 0:
            raise PopulationEvidenceError(f"{record_id} descriptor epoch is negative")
        _string(fields, "descriptor_epoch_id", record_id)
        dimension = _integer(fields, "descriptor_dimension", record_id)
        if dimension < 1 or dimension > 4096:
            raise PopulationEvidenceError(f"{record_id} descriptor dimension is invalid")
    elif record_type == "environment_probe_panel":
        _string(fields, "probe_panel_id", record_id)
        policies = fields.get("policy_artifact_sha256s")
        if not isinstance(policies, list) or not policies or len(set(policies)) != len(policies):
            raise PopulationEvidenceError(f"{record_id} needs unique probe policy hashes")
        for index, digest in enumerate(policies):
            _valid_hash(digest, f"{record_id}.policy_artifact_sha256s[{index}]")
        epoch = by_id[_single_parent(record, "descriptor_epoch")]
        if fields.get("descriptor_epoch_id") != epoch["fields"].get("descriptor_epoch_id"):
            raise PopulationEvidenceError(f"{record_id} descriptor epoch identity differs")
    elif record_type == "genome_candidate":
        if record_id != f"genome:{fields.get('genome_sha256')}":
            raise PopulationEvidenceError(f"{record_id} is not keyed by genome_sha256")
        private_path = _private_genome_field_path(fields)
        if private_path is not None:
            raise PopulationEvidenceError(
                f"{record_id} genome field {private_path!r} crosses the private-state boundary"
            )
        _string(fields, "variation_operator", record_id)
        if _integer(fields, "variation_seed", record_id) < 0:
            raise PopulationEvidenceError(f"{record_id} variation_seed is negative")
        if _integer(fields, "policy_adapter_count", record_id) < 1:
            raise PopulationEvidenceError(f"{record_id} policy_adapter_count is invalid")
        if _integer(fields, "policy_adapter_rank", record_id) < 1:
            raise PopulationEvidenceError(f"{record_id} policy_adapter_rank is invalid")
        expected = [
            by_id[parent]["fields"]["genome_sha256"]
            for parent in _parents_with_role(record, "genome_parent")
        ]
        if fields.get("parent_genome_sha256s") != expected:
            raise PopulationEvidenceError(f"{record_id} parent genome hashes differ from edges")
        inherited = _parents_with_role(record, "inherited_law_fit")
        if fields.get("inherited_law_fit_ids") != inherited:
            raise PopulationEvidenceError(f"{record_id} inherited law fits differ from edges")
        if any(by_id[parent]["fields"].get("status") != "completed" for parent in inherited):
            raise PopulationEvidenceError(f"{record_id} inherits an unsuccessful law fit")
    elif record_type == "environment_candidate":
        if record_id != f"environment:{fields.get('environment_sha256')}":
            raise PopulationEvidenceError(f"{record_id} is not keyed by environment_sha256")
        _string(fields, "variation_operator", record_id)
        if _integer(fields, "variation_seed", record_id) < 0:
            raise PopulationEvidenceError(f"{record_id} variation_seed is negative")
        if _integer(fields, "environment_epoch", record_id) < 0:
            raise PopulationEvidenceError(f"{record_id} environment_epoch is negative")
        expected = [
            by_id[parent]["fields"]["environment_sha256"]
            for parent in _parents_with_role(record, "environment_parent")
        ]
        if fields.get("parent_environment_sha256s") != expected:
            raise PopulationEvidenceError(f"{record_id} parent environment hashes differ from edges")
        panel = by_id[_single_parent(record, "probe_panel")]
        if fields.get("probe_panel_sha256") != panel["fields"].get("probe_panel_sha256"):
            raise PopulationEvidenceError(f"{record_id} probe panel identity differs")
    elif record_type == "birth":
        life_id = _string(fields, "life_id", record_id)
        mode = fields.get("birth_mode")
        physical = _parents_with_role(record, "physical_parent_birth")
        if mode == "experimental_initialization" and physical:
            raise PopulationEvidenceError(f"{record_id} experimental birth has physical parents")
        if mode == "embodied_reproduction" and not physical:
            raise PopulationEvidenceError(f"{record_id} embodied birth lacks physical parents")
        if mode not in {"experimental_initialization", "embodied_reproduction"}:
            raise PopulationEvidenceError(f"{record_id} has invalid birth_mode")
        if any(by_id[parent]["fields"].get("life_id") == life_id for parent in physical):
            raise PopulationEvidenceError(f"{record_id} reuses a physical parent's life_id")
        _validate_linked_identity(record, by_id)
    elif record_type == "life_checkpoint":
        _string(fields, "life_id", record_id)
        if _integer(fields, "tick", record_id) < 0:
            raise PopulationEvidenceError(f"{record_id} checkpoint tick is negative")
        parent = by_id[_single_parent(record, "life_continuation")]
        if parent["record_type"] == "life_checkpoint" and fields["tick"] <= parent["fields"]["tick"]:
            raise PopulationEvidenceError(f"{record_id} checkpoint tick does not advance")
    elif record_type in {"evaluation_completed", "evaluation_failed"}:
        _string(fields, "life_id", record_id)
        _string(fields, "evaluation_id", record_id)
        if _integer(fields, "evaluation_seed", record_id) < 0:
            raise PopulationEvidenceError(f"{record_id} evaluation_seed is negative")
        committed_ticks = _integer(fields, "committed_ticks", record_id)
        if committed_ticks < 0 or (
            record_type == "evaluation_completed" and committed_ticks == 0
        ):
            raise PopulationEvidenceError(f"{record_id} committed_ticks is invalid")
        allocation_status = fields.get("allocation_status")
        continuation = _parents_with_role(record, "life_continuation")
        planned = _parents_with_role(record, "planned_campaign")
        if record_type == "evaluation_completed":
            valid_allocation = allocation_status == "allocated" and len(continuation) == 1 and not planned
        elif allocation_status == "allocated":
            valid_allocation = len(continuation) == 1 and not planned
        elif allocation_status == "not_allocated":
            valid_allocation = not continuation and len(planned) == 1 and committed_ticks == 0
        else:
            valid_allocation = False
        if not valid_allocation:
            raise PopulationEvidenceError(f"{record_id} allocation proof is inconsistent")
        trace_blob = next(
            blob for blob in record["blob_refs"] if blob["role"] == "evaluation_trace"
        )
        if trace_blob["sha256"] != fields["trajectory_sha256"]:
            raise PopulationEvidenceError(f"{record_id} trace blob identity differs")
        if fields.get("status") != ("completed" if record_type.endswith("completed") else "failed"):
            raise PopulationEvidenceError(f"{record_id} terminal status differs from its type")
        metrics = fields.get("metrics")
        if not isinstance(metrics, Mapping) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in metrics.values()
        ):
            raise PopulationEvidenceError(f"{record_id} metrics must be finite numbers")
        if record_type == "evaluation_completed":
            descriptor = fields.get("descriptor")
            if not isinstance(descriptor, list) or not descriptor or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in descriptor
            ):
                raise PopulationEvidenceError(f"{record_id} descriptor must be finite")
            quality = fields.get("quality")
            if isinstance(quality, bool) or not isinstance(quality, (int, float)) or not math.isfinite(quality):
                raise PopulationEvidenceError(f"{record_id} quality must be finite")
        elif not isinstance(fields.get("failure"), str) or not fields["failure"].strip():
            raise PopulationEvidenceError(f"{record_id} failed evaluation lacks failure evidence")
        _validate_linked_identity(record, by_id)
        epoch = by_id[_single_parent(record, "descriptor_epoch")]
        panel = by_id[_single_parent(record, "probe_panel")]
        if fields.get("descriptor_epoch_id") != epoch["fields"].get("descriptor_epoch_id"):
            raise PopulationEvidenceError(f"{record_id} descriptor epoch identity differs")
        if fields.get("probe_panel_sha256") != panel["fields"].get("probe_panel_sha256"):
            raise PopulationEvidenceError(f"{record_id} probe panel identity differs")
        if panel["fields"].get("descriptor_epoch_id") != epoch["fields"].get("descriptor_epoch_id"):
            raise PopulationEvidenceError(f"{record_id} probe panel belongs to another epoch")
        if record_type == "evaluation_completed" and len(fields["descriptor"]) != epoch["fields"].get("descriptor_dimension"):
            raise PopulationEvidenceError(f"{record_id} descriptor dimension differs from its epoch")
    elif record_type == "archive_decision":
        decision = fields.get("decision")
        if decision not in {"retained", "rejected", "replaced"}:
            raise PopulationEvidenceError(f"{record_id} has invalid archive decision")
        evaluation = by_id[_single_parent(record, "evaluated_candidate")]
        if fields.get("evaluation_id") != evaluation["fields"].get("evaluation_id"):
            raise PopulationEvidenceError(f"{record_id} evaluation identity differs")
        if evaluation["record_type"] == "evaluation_failed" and decision != "rejected":
            raise PopulationEvidenceError(f"{record_id} cannot retain a failed evaluation")
        epoch = by_id[_single_parent(record, "descriptor_epoch")]
        if fields.get("descriptor_epoch_id") != epoch["fields"].get("descriptor_epoch_id"):
            raise PopulationEvidenceError(f"{record_id} descriptor epoch identity differs")
        if evaluation["fields"].get("descriptor_epoch_id") != fields.get("descriptor_epoch_id"):
            raise PopulationEvidenceError(f"{record_id} decision crosses descriptor epochs")
    elif record_type == "transfer_trial":
        if fields.get("direct_before_fine_tuning") is not True:
            raise PopulationEvidenceError(f"{record_id} must record direct transfer before tuning")
        if fields.get("status") not in {"completed", "failed"}:
            raise PopulationEvidenceError(f"{record_id} has invalid transfer status")
        source = by_id[_single_parent(record, "source_evaluation")]
        target = by_id[_single_parent(record, "target_evaluation")]
        if source["fields"].get("evaluation_id") == target["fields"].get("evaluation_id"):
            raise PopulationEvidenceError(f"{record_id} source and target evaluations are identical")
        candidate = by_id[_single_parent(record, "candidate_genome")]
        environment = by_id[_single_parent(record, "target_environment")]
        panel = by_id[_single_parent(record, "probe_panel")]
        candidate_sha = candidate["fields"].get("genome_sha256")
        if source["fields"].get("genome_sha256") != candidate_sha or target["fields"].get("genome_sha256") != candidate_sha:
            raise PopulationEvidenceError(f"{record_id} transfer changes candidate genome")
        if target["fields"].get("environment_sha256") != environment["fields"].get("environment_sha256"):
            raise PopulationEvidenceError(f"{record_id} target environment identity differs")
        if target["fields"].get("probe_panel_sha256") != panel["fields"].get("probe_panel_sha256"):
            raise PopulationEvidenceError(f"{record_id} probe panel identity differs")
    elif record_type == "population_snapshot":
        if _integer(fields, "ask_count", record_id) < 0:
            raise PopulationEvidenceError(f"{record_id} ask_count is negative")
    elif record_type == "gam_fit_attempt":
        status = fields.get("status")
        if status not in {"completed", "failed"}:
            raise PopulationEvidenceError(f"{record_id} has invalid fit status")
        if fields.get("unit_of_analysis") not in {"whole_life", "world"}:
            raise PopulationEvidenceError(f"{record_id} has invalid fit unit")
        law_blobs = [b for b in record["blob_refs"] if b["role"] == "gam_law"]
        if status == "completed" and len(law_blobs) != 1:
            raise PopulationEvidenceError(f"{record_id} completed fit needs one gam_law blob")
        if status == "failed" and law_blobs:
            raise PopulationEvidenceError(f"{record_id} failed fit cannot mint a law")
        if status == "failed" and not str(fields.get("failure", "")).strip():
            raise PopulationEvidenceError(f"{record_id} failed fit lacks failure evidence")


def _validate_linked_identity(
    record: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> None:
    fields = record["fields"]
    candidate = by_id[_single_parent(record, "candidate_genome")]
    environment = by_id[_single_parent(record, "environment")]
    if fields.get("genome_sha256") != candidate["fields"].get("genome_sha256"):
        raise PopulationEvidenceError(f"{record['id']} genome identity differs from its edge")
    if fields.get("environment_sha256") != environment["fields"].get("environment_sha256"):
        raise PopulationEvidenceError(f"{record['id']} environment identity differs from its edge")


def _validate_blob(blob: Mapping[str, Any], owner: str) -> None:
    if not isinstance(blob, Mapping) or not isinstance(blob.get("role"), str) or not blob["role"]:
        raise PopulationEvidenceError(f"{owner} has invalid blob role")
    digest = blob.get("sha256")
    _valid_hash(digest, f"{owner} blob {blob['role']}")
    if blob.get("uri") != f"urn:sha256:{digest}":
        raise PopulationEvidenceError(f"{owner} blob URI differs from its hash")
    size = blob.get("bytes")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
        raise PopulationEvidenceError(f"{owner} blob size is invalid")


def _valid_hash(value: Any, owner: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise PopulationEvidenceError(f"{owner} must be a lowercase SHA-256")


def _private_genome_field_path(value: Any, path: str = "fields") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            tokens = str(key).lower().replace("-", "_").split("_")
            child_path = f"{path}.{key}"
            if any(token in _PRIVATE_GENOME_TOKENS for token in tokens):
                return child_path
            found = _private_genome_field_path(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _private_genome_field_path(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _hash(fields: Mapping[str, Any], key: str, owner: str) -> str:
    value = fields.get(key)
    _valid_hash(value, f"{owner}.{key}")
    return value


def _string(fields: Mapping[str, Any], key: str, owner: str) -> str:
    value = fields.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PopulationEvidenceError(f"{owner}.{key} must be a nonempty string")
    return value


def _integer(fields: Mapping[str, Any], key: str, owner: str) -> int:
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PopulationEvidenceError(f"{owner}.{key} must be an integer")
    return value


def _parents_with_role(record: Mapping[str, Any], role: str) -> list[str]:
    roles = record["fields"]["parent_roles"]
    return [parent for parent in record["parent_ids"] if roles[parent] == role]


def _single_parent(record: Mapping[str, Any], role: str) -> str:
    parents = _parents_with_role(record, role)
    if len(parents) != 1:
        raise PopulationEvidenceError(f"{record['id']} does not have exactly one {role}")
    return parents[0]


def _validate_ledger_envelope(ledger: Mapping[str, Any]) -> None:
    if ledger.get("format") != LEDGER_FORMAT or ledger.get("schema_version") != 1:
        raise PopulationEvidenceError("unsupported population ledger format")
    if not isinstance(ledger.get("campaign_id"), str) or not ledger["campaign_id"].strip():
        raise PopulationEvidenceError("ledger campaign_id must be nonempty")
    if not isinstance(ledger.get("description"), str) or not ledger["description"].strip():
        raise PopulationEvidenceError("ledger description must be nonempty")
    if not isinstance(ledger.get("records"), list) or not isinstance(
        ledger.get("applied_batches"), list
    ):
        raise PopulationEvidenceError("ledger records and applied_batches must be arrays")
    if len(set(ledger["applied_batches"])) != len(ledger["applied_batches"]):
        raise PopulationEvidenceError("ledger repeats an applied batch")


def _validate_batch_envelope(batch: Mapping[str, Any], campaign_id: str) -> None:
    if batch.get("format") != BATCH_FORMAT or batch.get("schema_version") != 1:
        raise PopulationEvidenceError("unsupported population evidence batch")
    if batch.get("campaign_id") != campaign_id:
        raise PopulationEvidenceError("batch belongs to another campaign")
    if not isinstance(batch.get("batch_id"), str) or not batch["batch_id"].strip():
        raise PopulationEvidenceError("batch_id must be nonempty")
    if not isinstance(batch.get("records"), list) or not batch["records"]:
        raise PopulationEvidenceError("batch records must be a nonempty array")
    record_ids = [record.get("id") for record in batch["records"]]
    if any(not isinstance(record_id, str) for record_id in record_ids) or len(
        set(record_ids)
    ) != len(record_ids):
        raise PopulationEvidenceError("batch record ids must be unique strings")
    records_sha256 = hashlib.sha256(canonical_bytes(batch["records"])).hexdigest()
    expected_id = f"population-campaign:{records_sha256}"
    if batch["batch_id"] != expected_id:
        raise PopulationEvidenceError("batch_id does not authenticate batch records")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PopulationEvidenceError(f"{path} must contain a JSON object")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)
