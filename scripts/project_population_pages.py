#!/usr/bin/env python3
"""Project native population search state and public evidence into a Pages record."""

from __future__ import annotations

import argparse
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
                "physical_metrics": finite_mapping(row.get("trajectory_metrics", {}), "life trajectory metrics")
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
        if isinstance(value.get("graph"), Mapping):
            public["region_graph"] = value["graph"]
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
            analysts: Iterable[dict[str, Any]] = (), curve_inputs: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
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
        "evidence_records": evidence_records(evidence) if evidence else [],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--status", choices=("campaign-in-progress", "completed"), required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--supplements", type=Path)
    parser.add_argument("--evaluation-output", type=Path, action="append", default=[])
    parser.add_argument("--regional-analyst", type=Path, action="append", default=[])
    parser.add_argument("--trajectory-curves", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists")
    result = project(
        load_object(args.search, "search"), args.status,
        load_object(args.evidence, "evidence") if args.evidence else None,
        load_object(args.supplements, "supplements") if args.supplements else None,
        [load_object(path, "evaluation output") for path in args.evaluation_output],
        [load_object(path, "regional analyst") for path in args.regional_analyst],
        [load_object(path, "trajectory curves") for path in args.trajectory_curves],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical(result) + b"\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "content_sha256": result["content_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
