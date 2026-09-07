#!/usr/bin/env python3
"""Project native population search state and public evidence into a Pages record."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from chreatures.population_evidence import validate_records


FORMAT = "chreatures-public-population-campaign-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def sha(value: str, label: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def finite_mapping(value: Mapping[str, Any], label: str) -> dict[str, float]:
    result = {}
    for key, raw in value.items():
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{label}.{key} is nonfinite")
        result[str(key)] = number
    return result


def physical_summary(value: Mapping[str, Any]) -> dict[str, float]:
    """Flatten only the documented numeric resident summary into named measures."""
    result: dict[str, float] = {}
    scalar_fields = (
        "valid_ticks", "valid_time_seconds", "visited_spatial_cells",
        "mouth_contact_ticks", "mouth_contact_bouts", "contact_ticks",
        "contact_bouts", "quiet_ticks", "outside_world_ticks",
        "outside_deviation_sum", "outside_deviation_max", "contact_sum",
        "distance_sum", "effort_sum", "mechanical_work_sum",
        "ingested_mass_sum", "signal_activity_sum", "release_mass_sum",
        "secretion_mass_sum", "allocation_mass_sum", "energy_change",
        "mean_actual_speed", "height_mean", "height_range",
    )
    for name in scalar_fields:
        raw = value.get(name)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"life trajectory metrics.{name} is nonfinite")
            result[name] = number
    arrays = (
        ("physiology_mean", "physiology_order"),
        ("physiology_min", "physiology_order"),
        ("physiology_max", "physiology_order"),
        ("executed_action_mean", "executed_action_order"),
        ("executed_action_abs_mean", "executed_action_order"),
        ("outcome_sum", "outcome_order"),
        ("organ_flow_sum", "organ_flow_order"),
    )
    for field, order_field in arrays:
        values, order = value.get(field), value.get(order_field)
        if values is None:
            continue
        if not isinstance(values, list) or not isinstance(order, list) or len(values) != len(order):
            raise ValueError(f"life trajectory metrics.{field} differs from {order_field}")
        for name, raw in zip(order, values, strict=True):
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"life trajectory metrics.{field}.{name} is nonfinite")
            result[f"{field}.{name}"] = number
    return result


def evidence_records(value: dict[str, Any]) -> list[dict[str, Any]]:
    records = value.get("records", value.get("evidence_records", []))
    if not isinstance(records, list):
        raise ValueError("public evidence records must be a list")
    for record in records:
        parents = record.get("parent_ids")
        roles = record.get("fields", {}).get("parent_roles")
        if not isinstance(parents, list) or not isinstance(roles, dict):
            raise ValueError("each evidence record requires parent_ids and fields.parent_roles")
        if set(parents) != set(roles):
            raise ValueError("evidence parent roles must exactly cover parent_ids")
    validate_records(records, campaign_id=str(value.get("campaign_id", "")))
    return records


def cohort_failure_events(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("record_type") != "evaluation_failed":
            continue
        fields = record["fields"]
        key = str(fields.get("physical_output_sha256", fields.get("evaluation_identity_sha256", "")))
        if len(key) != 64:
            raise ValueError("failed evaluation lacks its shared physical output identity")
        event = grouped.setdefault(key, {
            "physical_output_sha256": key,
            "evaluation_identity_sha256": fields.get("evaluation_identity_sha256"),
            "failure": str(fields.get("failure", "evaluation failure")),
            "committed_ticks": int(fields.get("committed_ticks", 0)),
            "affected_life_ids": [],
            "last_coherent_checkpoint_tick": 0,
        })
        if event["failure"] != str(fields.get("failure", "evaluation failure")) or event["committed_ticks"] != int(fields.get("committed_ticks", 0)):
            raise ValueError("one physical evaluation output has inconsistent terminal facts")
        event["affected_life_ids"].append(str(fields["life_id"]))
        continuation = [parent for parent in record["parent_ids"] if fields["parent_roles"].get(parent) == "life_continuation"]
        if continuation and continuation[0].startswith("life-checkpoint:"):
            try:
                event["last_coherent_checkpoint_tick"] = max(event["last_coherent_checkpoint_tick"], int(continuation[0].split(":")[-2]))
            except ValueError as exc:
                raise ValueError("failed evaluation checkpoint identity has an invalid tick") from exc
    return sorted(grouped.values(), key=lambda item: item["physical_output_sha256"])


def nullable_numbers(value: Any, label: str, *, integers: bool = False) -> list[float] | list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list or null")
    if integers:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError(f"{label} must contain integers")
        return list(value)
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a nonfinite value")
    return result


def evaluator_lives(values: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Project public terminal facts from authenticated physical evaluator outputs."""
    lives: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for value in values:
        status = value.get("status")
        rows = value.get("lives") if status == "completed" else value.get("candidate_failures")
        if status not in {"completed", "failed"} or not isinstance(rows, list):
            raise ValueError("evaluator output must be a completed result or failed receipt")
        for row in rows:
            life_id = str(row["life_id"])
            if life_id in by_id:
                raise ValueError(f"duplicate evaluator life {life_id}")
            trajectory = sha(str(row["trajectory_sha256"]), "evaluator trajectory")
            public = {
                "life_id": life_id,
                "candidate_sha256": sha(str(row["candidate_sha256"]), "life candidate"),
                "environment_sha256": sha(str(row["environment_sha256"]), "life environment"),
                "birth_kind": str(row.get("birth_kind", row.get("birth_mode", "experimental_initialization"))),
                "terminal_status": "completed" if status == "completed" else "failed",
                "committed_ticks": int(row.get("committed_ticks", 0)),
                "trajectory_sha256": trajectory,
                "physical_metrics": physical_summary(row["trajectory_metrics"])
                    if isinstance(row.get("trajectory_metrics"), Mapping) else {},
            }
            lives.append(public)
            by_id[life_id] = public
    return lives, by_id


def regional_records(values: Iterable[dict[str, Any]], environments: set[str]) -> list[dict[str, Any]]:
    records = []
    for value in values:
        if value.get("runtime_visible") is not False:
            raise ValueError("regional analyst record must state runtime_visible=false")
        record = value.get("environment_record")
        if not isinstance(record, Mapping):
            raise ValueError("regional analyst record lacks environment_record")
        identity = str(record.get("environment_sha256", record.get("sha256", "")))
        if identity not in environments:
            raise ValueError("regional analyst geometry belongs to an unknown environment")
        public = dict(record)
        public["sha256"] = identity
        # Graph coordinates are analyst annotations; they are never controller input.
        graph = value.get("graph")
        if isinstance(graph, Mapping):
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            if not isinstance(nodes, list) or not isinstance(edges, list):
                raise ValueError("regional analyst graph nodes and edges must be lists")
            regions = []
            node_ids = []
            for index, node in enumerate(nodes):
                position, half = node.get("position_m"), node.get("half_size_m")
                if not (isinstance(position, list) and len(position) == 3 and
                        isinstance(half, list) and len(half) == 2):
                    raise ValueError("regional analyst node geometry differs")
                x, y, z = (float(item) for item in position)
                hx, hy = (float(item) for item in half)
                if not all(math.isfinite(item) for item in (x, y, z, hx, hy)) or hx <= 0 or hy <= 0:
                    raise ValueError("regional analyst node geometry is invalid")
                node_id = str(node.get("id", f"region-{index}"))
                node_ids.append(node_id)
                regions.append({
                    "id": node_id, "position_m": [x, y, z], "half_size_m": [hx, hy],
                    "polygon": [[x-hx, y-hy], [x+hx, y-hy], [x+hx, y+hy], [x-hx, y+hy]],
                    "elevation": z, "lane": node.get("lane"),
                    "underpass": bool(node.get("underpass", False)),
                    "sheltered": bool(node.get("sheltered", False)),
                })
            public["regions"] = regions
            public["region_edges"] = []
            for edge in edges:
                endpoints = edge.get("nodes")
                if not (isinstance(endpoints, list) and len(endpoints) == 2 and
                        all(isinstance(item, int) and 0 <= item < len(node_ids) for item in endpoints)):
                    raise ValueError("regional analyst edge endpoints differ")
                public["region_edges"].append({
                    "id": str(edge.get("id", "regional-edge")),
                    "from": node_ids[endpoints[0]], "to": node_ids[endpoints[1]],
                    "rise_m": float(edge.get("rise_m", 0.0)),
                    "run_m": float(edge.get("run_m", 0.0)),
                    "width_m": float(edge.get("width_m", 0.0)),
                })
            public["region_graph_connected"] = bool(graph.get("connected", False))
        records.append(public)
    return records


def trajectory_curves(values: Iterable[dict[str, Any]], known: set[str]) -> list[dict[str, Any]]:
    curves = []
    for value in values:
        rows = value.get("trajectories", value.get("trajectory_records"))
        if not isinstance(rows, list):
            raise ValueError("trajectory curve input requires trajectories")
        for row in rows:
            digest = sha(str(row["trajectory_sha256"]), "trajectory curve")
            if digest not in known:
                raise ValueError("trajectory curve does not match an evaluator trace identity")
            series = row.get("series")
            if not isinstance(series, Mapping) or not series:
                raise ValueError("trajectory curve requires nonempty recorded series")
            clean_series = {str(name): nullable_numbers(points, f"trajectory {name}") for name, points in series.items()}
            lengths = {len(points) for points in clean_series.values() if points is not None}
            if len(lengths) != 1:
                raise ValueError("trajectory series lengths differ")
            curves.append({**row, "trajectory_sha256": digest, "series": clean_series})
    return curves


def project(search: dict[str, Any], status: str, evidence: dict[str, Any] | None,
            supplements: dict[str, Any] | None, evaluator_outputs: Iterable[dict[str, Any]] = (),
            analysts: Iterable[dict[str, Any]] = (), curve_inputs: Iterable[dict[str, Any]] = (),
            weave_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    if search.get("format") != "chreatures-population-search-v1":
        raise ValueError("native search format differs")
    genomes = search.get("genomes")
    environments = search.get("environments")
    archive = search.get("archive")
    evaluations = search.get("evaluations")
    if not all(isinstance(value, expected) for value, expected in (
        (genomes, dict), (environments, dict), (archive, dict), (evaluations, list)
    )):
        raise ValueError("native search state is incomplete")
    public_environments = []
    for key, environment in sorted(environments.items()):
        sha(str(key), "environment key")
        if environment.get("sha256") != key:
            raise ValueError("environment map key differs from its identity")
        public_environments.append({
            "sha256": key,
            "parents": [sha(str(item), "environment parent") for item in environment.get("parents", [])],
            "variation_recipe_sha256": sha(str(environment["variation"]["recipe_sha256"]), "environment recipe"),
            "epoch": int(environment["epoch"]),
            "topology_sha256": sha(str(environment["topology_sha256"]), "topology"),
            "resource_sha256": sha(str(environment["resource_sha256"]), "resources"),
            "profile_sha256": sha(str(environment["profile_sha256"]), "environment profile"),
            "region_geometry_ref": f"urn:sha256:{environment['topology_sha256']}",
        })
    public_cells = []
    for key, members in sorted(archive.items()):
        coordinates = [int(item) for item in key.split(":")]
        public_cells.append({"coordinates": coordinates, "members": [{
            "candidate_sha256": sha(str(member["candidate_sha256"]), "archive candidate"),
            "evaluation_sha256": sha(str(member["evaluation_sha256"]), "archive evaluation"),
            "quality": float(member["quality"]),
            "descriptor": [float(item) for item in member["descriptor"]],
        } for member in members]})
    public_evaluations = []
    for evaluation in evaluations:
        public_evaluations.append({
            "evaluation_sha256": sha(str(evaluation["evaluation_sha256"]), "evaluation"),
            "candidate_sha256": sha(str(evaluation["candidate_sha256"]), "evaluation candidate"),
            "environment_sha256": sha(str(evaluation["environment_sha256"]), "evaluation environment"),
            "status": str(evaluation["status"]),
            "failure": {"summary": str(evaluation.get("failure", ""))} if evaluation.get("failure") else None,
            "metrics": finite_mapping(evaluation.get("metrics", {}), "evaluation metrics"),
            "descriptor": nullable_numbers(evaluation.get("descriptor"), "evaluation descriptor"),
            "cell": nullable_numbers(evaluation.get("cell"), "evaluation cell", integers=True),
            "quality": None if evaluation.get("quality") is None else float(evaluation["quality"]),
            "archive_retained": bool(evaluation.get("archive_retained", False)),
            "life_id": str(evaluation.get("life_id", "")),
            "evaluation_seed": evaluation.get("evaluation_seed"),
            "committed_ticks": int(evaluation.get("committed_ticks", 0)),
            "trajectory_sha256": sha(str(evaluation["trajectory_sha256"]), "evaluation trajectory"),
        })
    candidates = []
    for key, genome in sorted(genomes.items()):
        sha(str(key), "genome key")
        candidates.append({
            "sha256": key,
            "parents": [sha(str(item), "genome parent") for item in genome.get("parents", [])],
            "variation_recipe_sha256": sha(str(genome["variation"]["recipe_sha256"]), "genome recipe"),
            "variation_operator": str(genome["variation"]["operator"]),
            "mutated_parameters": [str(item) for item in genome["variation"].get("mutated", [])],
            "values": finite_mapping(genome.get("values", {}), "genome values"),
        })
    public_records = evidence_records(evidence) if evidence else []
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": status,
        "descriptor_version": str(search["descriptor_version"]),
        "quality_version": str(search["quality_version"]),
        "search_identity_sha256": hashlib.sha256(canonical(search)).hexdigest(),
        "candidates": candidates,
        "environments": public_environments,
        "cells": public_cells,
        "evaluations": public_evaluations,
        "recording_references": [],
        "evidence_records": public_records,
        "failure_events": cohort_failure_events(public_records),
    }
    lives, evaluator_by_life = evaluator_lives(evaluator_outputs)
    native_by_life = {item["life_id"]: item for item in public_evaluations}
    for life_id, life in evaluator_by_life.items():
        native = native_by_life.get(life_id)
        if native is None or any(native[key] != life[key] for key in (
            "candidate_sha256", "environment_sha256", "committed_ticks", "trajectory_sha256"
        )):
            raise ValueError(f"evaluator life {life_id} differs from native archive")
    if lives:
        result["lives"] = lives
    result["environment_records"] = regional_records(analysts, set(environments))
    result["trajectories"] = trajectory_curves(
        curve_inputs, {item["trajectory_sha256"] for item in public_evaluations}
    )
    if weave_receipt:
        native = weave_receipt.get("native")
        artifact = weave_receipt.get("artifacts", {}).get("weave")
        if not isinstance(native, Mapping) or not isinstance(artifact, Mapping):
            raise ValueError("Weave receipt lacks native validation or artifact identity")
        result["weave_receipt"] = {
            "integration": str(native.get("integration")),
            "node_count": int(native["node_count"]),
            "edge_count": int(native["edge_count"]),
            "multi_parent_nodes": int(native["multi_parent_nodes"]),
            "validated_after_reload": bool(native["validated_after_reload"]),
            "reload_equal": bool(native["reload_equal"]),
            "artifact_sha256": sha(str(artifact["sha256"]), "Weave artifact"),
        }
    if supplements:
        allowed = {
            "name", "lives", "environment_records", "trajectories", "gam_surfaces",
            "recording_references", "campaign_summary",
        }
        unknown = set(supplements) - allowed
        if unknown:
            raise ValueError(f"unknown public supplement fields: {sorted(unknown)}")
        result.update(supplements)
    result["content_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def project_recording_summary(ledger_path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Publish an authenticated recording summary without private ledger records."""
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    records = evidence_records(ledger)
    native = receipt["native"]
    if (
        receipt["campaign_id"] != ledger["campaign_id"]
        or receipt["artifacts"]["ledger"]["sha256"] != hashlib.sha256(ledger_bytes).hexdigest()
        or receipt["record_count"] != len(records)
        or native["node_count"] != len(records)
        or native["edge_count"] != sum(len(row["parent_ids"]) for row in records)
        or native["validated_after_reload"] is not True
        or native["reload_equal"] is not True
    ):
        raise ValueError("recording ledger differs from its verified native Weave receipt")
    recordings = [row for row in records if row["record_type"] == "embodied_recording"]
    if len(recordings) != 1:
        raise ValueError("recording summary requires exactly one authenticated recording")
    fields = recordings[0]["fields"]
    if fields["recording_format"] != "chreatures-living-reef-public-recording-v4":
        raise ValueError("recording summary requires current v4 evidence")
    # Explicit allowlists: never serialize private body bindings, life IDs,
    # source records, filesystem artifact locators, or checkpoint contents.
    public_recording = {key: fields[key] for key in (
        "recording_format", "recording_content_sha256", "recording_sha256",
        "recording_transport_sha256", "recording_transport_decoded_sha256",
        "recording_transport_encoding", "first_tick", "last_tick", "frame_count",
        "resident_count", "event_count",
    )}
    result = {
        "format": FORMAT,
        "status": "completed",
        "name": "Historical v8 ecological specialization recording evidence",
        "scope": "Completed historical recording and its external evidence graph; not the newer CNS-only controller architecture or a competence evaluation.",
        "campaign_id": ledger["campaign_id"],
        "campaign_summary": {
            "architecture": "ecological-specialization-v8",
            "historical": True,
            "record_types": dict(sorted(Counter(row["record_type"] for row in records).items())),
            "event_kinds": dict(sorted(Counter(row["fields"]["kind"] for row in records if row["record_type"] in {
                "organism_transfer", "development_event", "environment_event", "interaction_event"
            }).items())),
            "private_records_published": False,
        },
        "recording_references": [public_recording],
        "evidence_records": [],
        "weave_receipt": {
            **{key: native[key] for key in (
                "integration", "library", "node_count", "edge_count", "multi_parent_nodes",
                "validated_after_reload", "reload_equal",
            )},
            "artifact_sha256": sha(receipt["artifacts"]["weave"]["sha256"], "Weave artifact"),
            "ledger_sha256": sha(receipt["artifacts"]["ledger"]["sha256"], "ledger artifact"),
        },
    }
    result["content_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path)
    parser.add_argument("--status", choices=("campaign-in-progress", "completed"))
    parser.add_argument("--historical-v8-recording-summary", action="store_true",
                        help="Project only public recording facts and native graph counts from --evidence and --weave-receipt")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--supplements", type=Path)
    parser.add_argument("--evaluation-output", type=Path, action="append", default=[])
    parser.add_argument("--regional-analyst", type=Path, action="append", default=[])
    parser.add_argument("--trajectory-curves", type=Path, action="append", default=[])
    parser.add_argument("--weave-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists")
    if args.historical_v8_recording_summary:
        if not args.evidence or not args.weave_receipt or args.search or args.status:
            parser.error("--historical-v8-recording-summary requires --evidence and --weave-receipt, without --search or --status")
        result = project_recording_summary(args.evidence, load_object(args.weave_receipt, "Weave receipt"))
    else:
        if not args.search or not args.status:
            parser.error("population projection requires --search and --status")
        result = project(
            load_object(args.search, "search"), args.status,
            load_object(args.evidence, "evidence") if args.evidence else None,
            load_object(args.supplements, "supplements") if args.supplements else None,
            [load_object(path, "evaluation output") for path in args.evaluation_output],
            [load_object(path, "regional analyst") for path in args.regional_analyst],
            [load_object(path, "trajectory curves") for path in args.trajectory_curves],
            load_object(args.weave_receipt, "Weave receipt") if args.weave_receipt else None,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical(result) + b"\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "content_sha256": result["content_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
