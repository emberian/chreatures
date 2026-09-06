#!/usr/bin/env python3
"""Durable coordinator for native population ask/plan/tell campaigns."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chreatures.organism_interface import identity as organism_identity
from chreatures.population import (
    PopulationSearch,
    canonical_bytes,
    current_parameter_recipe,
)
from chreatures.training_environment import EmbodiedTrainingProfile
from chreatures.resident_contract import (
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
)

FORMAT = "chreatures-population-campaign-v2"
PLAN_FORMAT = "chreatures-population-campaign-plan-v2"
ASSIGNMENT_FORMAT = "chreatures-population-evaluation-assignments-v1"
DESCRIPTOR_RECIPE = "physical-population-descriptor-v2"
QUALITY_RECIPE = "finite-life-quality-v2"
SPATIAL_CELL_SCALE = 256.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def valid_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
            + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def controller_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    bank = metadata.get("population_adapters")
    if (
        metadata.get("format") != NATIVE_POPULATION_FORMAT
        or metadata.get("version") != NATIVE_POPULATION_VERSION
        or metadata.get("execution") != NATIVE_EXECUTION
        or not isinstance(bank, dict)
        or not isinstance(bank.get("count"), int)
        or not isinstance(bank.get("rank"), int)
    ):
        raise ValueError("controller is not the current native population artifact")
    valid_sha(bank.get("identity"), "population adapter bank")
    artifact = {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "artifact_sha256": valid_sha(
            metadata.get("artifact_sha256"), "controller artifact"
        ),
        "population_adapter_bank_sha256": bank["identity"],
        "population_adapter_count": bank["count"],
        "population_adapter_rank": bank["rank"],
    }
    return artifact, metadata


def init_command(args: argparse.Namespace) -> None:
    destination = args.output.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit("campaign output must be absent or empty")
    profile_raw = load_json(args.profile)
    profile = EmbodiedTrainingProfile.from_value(profile_raw)
    controller, metadata = controller_identity(args.controller)
    sources = profile.component("sources")
    developmental_base = valid_sha(
        sources["biosphere_birth"]["sha256"], "biosphere birth"
    )
    graph_sha = valid_sha(
        metadata["training_identity"]["graph_sha256"], "training graph"
    )
    port_spec_sha = valid_sha(
        metadata["training_identity"]["port_spec_sha256"], "training ports"
    )
    interface_sha = value_sha256(organism_identity())
    if metadata.get("organism_interface") != organism_identity():
        raise ValueError("controller organism interface differs")
    variants = profile.component("family")["variants"]
    epochs = {int(item["environment_record"]["epoch"]) for item in variants}
    if len(epochs) != 1:
        raise ValueError("campaign environments span archive epochs")
    specs, founder = current_parameter_recipe(
        policy_adapter_count=controller["population_adapter_count"],
        heritable_policy_adapter_rows=False,
    )
    variation_receipt = {
        "operator": "bounded-genome-variation-v1",
        "parameters": specs,
        "policy_adapter_selection": "fixed-row-zero-v1",
    }
    probe_panel = {
        "format": "chreatures-population-probe-panel-v1",
        "controller_file_sha256": controller["file_sha256"],
        "action_mode": "sample",
        "fine_tuning": False,
    }
    search_config = {
        "graph_sha256": graph_sha,
        "port_spec_sha256": port_spec_sha,
        "base_controller_sha256": controller["file_sha256"],
        "developmental_base_sha256": developmental_base,
        "population_adapter_bank_sha256": controller["population_adapter_bank_sha256"],
        "organism_interface_sha256": interface_sha,
        "policy_adapter_count": controller["population_adapter_count"],
        "policy_adapter_rank": controller["population_adapter_rank"],
        "parameter_specs": specs,
        "founder_values": founder,
        "descriptor_axes": [
            {"component": "mean_action_thrust", "low": -1.0, "high": 1.0, "bins": 8},
            {"component": "spatial_coverage", "low": 0.0, "high": 1.0, "bins": 8},
            {"component": "elevation_fraction", "low": 0.0, "high": 1.0, "bins": 6},
            {"component": "signal_activity_rate", "low": 0.0, "high": 2.0, "bins": 6},
            {"component": "allocated_mass_rate", "low": 0.0, "high": 0.02, "bins": 6},
        ],
        "quality_terms": [
            {"component": "mean_energy", "scale": 1.0, "weight": 0.5, "direction": 1.0},
            {
                "component": "energy_delta",
                "scale": 1.0,
                "weight": 0.3,
                "direction": 1.0,
            },
            {
                "component": "mean_effort",
                "scale": 1.0,
                "weight": 0.2,
                "direction": -1.0,
            },
        ],
        "archive_members_per_cell": 4,
        "variation_recipe_sha256": value_sha256(variation_receipt),
        "environment_probe_panel_sha256": value_sha256(probe_panel),
        "environment_epoch": epochs.pop(),
        "environment_novelty_weight": 0.25,
        "environment_cost_weight": 0.10,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = destination.with_name(f".{destination.name}.init-{os.getpid()}")
    root.mkdir()
    campaign_profile_path = root / "profile.json"
    atomic_json(campaign_profile_path, profile.to_value())
    search = PopulationSearch.initialize(
        root / "search.json", search_config, seed=args.seed
    )
    environment_index = {}
    for item in variants:
        record = item["environment_record"]
        if record.get("sha256") != item.get("environment_sha256"):
            raise ValueError("profile environment record differs")
        search.register_environment(record)
        environment_index[record["sha256"]] = {
            "split": item["split"],
            "index": item["index"],
            "dimensions_m": item["dimensions_m"],
            "resident_count": item["resident_count"],
        }
    campaign = {
        "format": FORMAT,
        "version": 2,
        "status": "initialized",
        "seed": args.seed,
        "profile": {
            "path": str(campaign_profile_path.relative_to(root)),
            "source_path": str(args.profile.resolve()),
            "source_file_sha256": file_sha256(args.profile),
            "file_sha256": file_sha256(campaign_profile_path),
            "sha256": profile.sha256,
        },
        "controller": controller,
        "organism_interface_sha256": interface_sha,
        "search_state": "search.json",
        "search_config_sha256": value_sha256(search_config),
        "descriptor": {
            "version": DESCRIPTOR_RECIPE,
            "spatial_cell_scale": SPATIAL_CELL_SCALE,
            "axes": search_config["descriptor_axes"],
        },
        "quality": {
            "version": QUALITY_RECIPE,
            "terms": search_config["quality_terms"],
            "meaning": "initial bounded regulation/search score; not ecological fitness or feeding competence",
        },
        "probe_panel": probe_panel,
        "sources": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "chreatures/population.py",
                ROOT / "native/population-core/Cargo.toml",
                ROOT / "native/population-core/Cargo.lock",
                ROOT / "native/population-core/src/lib.rs",
                ROOT / "native/population-core/src/main.rs",
            )
        },
        "environment_index": environment_index,
        "plans": [],
        "active_transaction": None,
        "ingested_sources": [],
    }
    atomic_json(root / "campaign.json", campaign)
    if destination.exists():
        destination.rmdir()
    os.replace(root, destination)
    print(
        json.dumps(
            {
                "campaign": str(destination),
                "environments": len(environment_index),
                "search_config_sha256": campaign["search_config_sha256"],
            },
            sort_keys=True,
        )
    )


def campaign(root: Path) -> tuple[dict[str, Any], PopulationSearch]:
    value = load_json(root / "campaign.json")
    if value.get("format") != FORMAT or value.get("version") != 2:
        raise ValueError("campaign format differs")
    return value, PopulationSearch(root / value["search_state"])


def assignment_document(
    worlds: list[dict[str, Any]], plan_sha: str, batch: int
) -> dict[str, Any]:
    body = {
        "format": ASSIGNMENT_FORMAT,
        "version": 1,
        "campaign_plan_sha256": plan_sha,
        "batch_index": batch,
        "worlds": worlds,
    }
    return body | {"sha256": value_sha256(body)}


def plan_command(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    value, search = campaign(root)
    search.validate()
    if not 1 <= args.worlds_per_batch <= 4 or not 1 <= args.candidate_waves <= 1024:
        raise SystemExit("worlds-per-batch must be 1..4 and candidate-waves 1..1024")
    if value["plans"] and any(item["status"] != "ingested" for item in value["plans"]):
        raise SystemExit("prior campaign plan is not fully ingested")
    state = search.snapshot_value()
    claimed = {
        pair for plan in value["plans"] for pair in plan.get("assignment_pairs", [])
    }
    unplanned = set(state["pending_assignments"]) - claimed
    transaction = value.get("active_transaction")
    if transaction is None and unplanned:
        raise RuntimeError("native state has unplanned pending assignments")
    environments = sorted(value["environment_index"])
    resident_counts = {
        value["environment_index"][key]["resident_count"] for key in environments
    }
    if len(resident_counts) != 1:
        raise ValueError("environment resident counts differ")
    residents = resident_counts.pop()
    requested = residents * len(environments) * args.candidate_waves
    if requested > 4096:
        raise SystemExit(
            "one durable native ask is limited to 4096 candidates; reduce candidate-waves"
        )
    if transaction is None:
        transaction = {
            "operation": "plan-ask",
            "index": len(value["plans"]),
            "requested": requested,
            "preexisting_pending": sorted(state["pending_assignments"]),
            "preexisting_genomes": sorted(state["genomes"]),
            "worlds_per_batch": args.worlds_per_batch,
            "candidate_waves": args.candidate_waves,
            "selection": args.selection,
        }
        value["active_transaction"] = transaction
        atomic_json(root / "campaign.json", value)
        if args.selection == "challenge":
            assignments = search.ask_challenges(requested)
        else:
            assignments = search.ask_transfers(requested)
            if len(assignments) < requested:
                assignments.extend(search.ask(requested - len(assignments)))
    else:
        if (
            transaction["requested"] != requested
            or transaction["worlds_per_batch"] != args.worlds_per_batch
            or transaction["candidate_waves"] != args.candidate_waves
            or transaction["selection"] != args.selection
        ):
            raise SystemExit("resume arguments differ from active plan transaction")
        state = search.snapshot_value()
        prior = set(transaction["preexisting_pending"])
        pairs = sorted(set(state["pending_assignments"]) - prior)
        if len(pairs) < requested:
            if args.selection == "challenge":
                if pairs:
                    raise RuntimeError("interrupted challenge ask is not atomic")
                search.ask_challenges(requested)
            else:
                # Native operations publish their whole state atomically. A
                # short set means the transfer pass completed and mutation did
                # not; completing that exact remainder is safe.
                search.ask(requested - len(pairs))
            state = search.snapshot_value()
            pairs = sorted(set(state["pending_assignments"]) - prior)
        if len(pairs) != requested:
            raise RuntimeError("interrupted native ask has an invalid pending count")
        assignments = []
        for pair in pairs:
            candidate_sha, environment_sha = pair.split(":", 1)
            assignments.append(
                {
                    "candidate": state["genomes"][candidate_sha],
                    "environment_sha256": environment_sha,
                    "phase": (
                        "history-challenge-repeat"
                        if args.selection == "challenge" and pair in state["pair_histories"]
                        else "history-challenge-transfer"
                        if args.selection == "challenge"
                        else "direct-transfer"
                    ),
                    "selection": state["pending_evidence"][pair],
                }
            )
    normalized = [
        {
            "candidate": item["candidate"].to_value()
            if hasattr(item["candidate"], "to_value")
            else item["candidate"],
            "environment_sha256": item["environment_sha256"],
            "phase": item["phase"],
            "selection": item["selection"],
        }
        for item in assignments
    ]
    preexisting_genomes = set(transaction["preexisting_genomes"])
    transfer_count = sum(
        item["candidate"]["sha256"] in preexisting_genomes for item in normalized
    )
    grouped = {key: [] for key in environments}
    for item in normalized:
        grouped[item["environment_sha256"]].append(item["candidate"])
    if any(len(rows) != residents * args.candidate_waves for rows in grouped.values()):
        raise RuntimeError("native ask did not form complete environment populations")
    plan_identity = {
        "format": PLAN_FORMAT,
        "version": 2,
        "index": transaction["index"],
        "worlds_per_batch": args.worlds_per_batch,
        "candidate_waves": args.candidate_waves,
        "resident_count": residents,
        "environment_count": len(environments),
        "requested_candidates": requested,
        "retained_transfer_assignments": transfer_count,
        "new_variant_assignments": requested - transfer_count,
        "selection": args.selection,
        "selection_phases": {
            phase: sum(item["phase"] == phase for item in normalized)
            for phase in sorted({item["phase"] for item in normalized})
        },
        "search_config_sha256": value["search_config_sha256"],
        "profile_sha256": value["profile"]["sha256"],
        "controller_file_sha256": value["controller"]["file_sha256"],
        "environment_sha256": environments,
    }
    plan_sha = value_sha256(plan_identity)
    plan_dir = root / "plans" / f"plan-{transaction['index']:04d}"
    batches = []
    assignment_pairs = []
    for wave in range(args.candidate_waves):
        worlds = []
        for environment_sha in environments:
            meta = value["environment_index"][environment_sha]
            candidates = grouped[environment_sha][
                wave * residents : (wave + 1) * residents
            ]
            seed = int.from_bytes(
                hashlib.sha256(
                    canonical_bytes(
                        [value["seed"], transaction["index"], wave, environment_sha]
                    )
                ).digest()[:8],
                "little",
            )
            worlds.append(
                {
                    "world_id": environment_sha,
                    "seed": seed,
                    "environment": {"split": meta["split"], "index": meta["index"]},
                    "candidates": candidates,
                }
            )
            assignment_pairs.extend(
                f"{candidate['sha256']}:{environment_sha}" for candidate in candidates
            )
        for offset in range(0, len(worlds), args.worlds_per_batch):
            batch_index = len(batches)
            document = assignment_document(
                worlds[offset : offset + args.worlds_per_batch], plan_sha, batch_index
            )
            path = plan_dir / f"batch-{batch_index:04d}.json"
            atomic_json(path, document)
            batches.append(
                {
                    "index": batch_index,
                    "status": "planned",
                    "assignment": str(path.relative_to(root)),
                    "assignment_file_sha256": file_sha256(path),
                    "assignment_content_sha256": document["sha256"],
                    "source_results": [],
                }
            )
    plan = {
        "format": PLAN_FORMAT,
        "version": 2,
        "identity": plan_identity,
        "identity_sha256": plan_sha,
        "status": "planned",
        "batches": batches,
        "assignment_pairs": sorted(assignment_pairs),
        "selection_records": [
            {
                "candidate_sha256": item["candidate"]["sha256"],
                "environment_sha256": item["environment_sha256"],
                "phase": item["phase"],
                "selection": item["selection"],
            }
            for item in normalized
        ],
    }
    atomic_json(plan_dir / "plan.json", plan)
    value["plans"].append(
        {
            "index": plan_identity["index"],
            "path": str((plan_dir / "plan.json").relative_to(root)),
            "sha256": plan_sha,
            "status": "planned",
            "assignment_pairs": plan["assignment_pairs"],
        }
    )
    value["active_transaction"] = None
    value["status"] = "planned"
    atomic_json(root / "campaign.json", value)
    print(
        json.dumps(
            {
                "plan": str(plan_dir / "plan.json"),
                "worlds": len(environments) * args.candidate_waves,
                "batches": len(batches),
                "candidates": requested,
                "sha256": plan_sha,
            },
            sort_keys=True,
        )
    )


def metric_row(
    life: Mapping[str, Any], environment: Mapping[str, Any]
) -> dict[str, float]:
    row = life["trajectory_metrics"]
    ticks = int(life["committed_ticks"])
    valid = int(row["valid_ticks"])
    dt = float(row["sampling_dt_seconds"])
    if ticks <= 0 or valid <= 0 or valid > ticks or dt != 0.05:
        raise ValueError("completed life sampling differs")
    action = row["executed_action_mean"]
    phys = row["physiology_mean"]
    dimensions = environment["dimensions_m"]
    metrics = {
        "mean_action_thrust": float(action[0]),
        "spatial_coverage": float(row["visited_spatial_cells"]) / SPATIAL_CELL_SCALE,
        "elevation_fraction": float(row["height_range"]) / float(dimensions[2]),
        "signal_activity_rate": float(row["signal_activity_sum"]) / valid,
        "allocated_mass_rate": float(row["allocation_mass_sum"]) / (valid * dt),
        "mean_energy": float(phys[0]),
        "energy_delta": float(row["energy_change"]),
        "mean_effort": float(row["effort_sum"]) / valid,
    }
    if any(not math.isfinite(item) for item in metrics.values()):
        raise ValueError("derived campaign metric is nonfinite")
    return metrics


def locate_batch(
    root: Path, value: Mapping[str, Any], assignment_sha: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    for summary in value["plans"]:
        path = root / summary["path"]
        plan = load_json(path)
        for batch in plan["batches"]:
            if batch["assignment_file_sha256"] == assignment_sha:
                return path, plan, batch
    raise ValueError("evaluation assignment is outside campaign")


def ingest_command(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    value, search = campaign(root)
    source = args.result.resolve()
    source_sha = file_sha256(source)
    if source_sha in value["ingested_sources"]:
        print(json.dumps({"status": "already-ingested", "source_sha256": source_sha}))
        return
    document = load_json(source)
    status = document.get("status")
    if (
        document.get("format") == "chreatures-population-episode-evaluation-v1"
        and status == "completed"
    ):
        rows = document["lives"]
        failure = False
    elif (
        document.get("format") == "chreatures-population-evaluation-failure-v1"
        and status == "failed"
    ):
        rows = document["candidate_failures"]
        failure = True
    else:
        raise ValueError("input is not a terminal population evaluation")
    if document.get("content_sha256") != value_sha256(
        {k: v for k, v in document.items() if k != "content_sha256"}
    ):
        raise ValueError("evaluation result content identity differs")
    assignment_shas = {row["assignment_file_sha256"] for row in rows}
    if len(assignment_shas) != 1:
        raise ValueError("evaluation rows span assignment artifacts")
    plan_path, plan, batch = locate_batch(root, value, assignment_shas.pop())
    existing = {
        item["evaluation_sha256"] for item in search.snapshot_value()["evaluations"]
    }
    assignment = load_json(root / batch["assignment"])
    expected_lives = {
        (candidate["sha256"], world["world_id"], world["seed"])
        for world in assignment["worlds"]
        for candidate in world["candidates"]
    }
    actual_lives = {
        (row["candidate_sha256"], row["world_id"], row["evaluation_seed"])
        for row in rows
    }
    if actual_lives != expected_lives or len(rows) != len(expected_lives):
        raise ValueError("terminal evaluation does not cover its exact assigned lives")
    told = []
    for row in rows:
        environment_sha = valid_sha(row["environment_sha256"], "life environment")
        meta = value["environment_index"][environment_sha]
        result = {
            "life_id": valid_sha(row["life_id"], "life"),
            "evaluation_seed": int(row["evaluation_seed"]),
            "committed_ticks": int(row["committed_ticks"]),
            "trajectory_sha256": valid_sha(row["trajectory_sha256"], "trajectory"),
            "candidate_sha256": valid_sha(row["candidate_sha256"], "candidate"),
            "environment_sha256": environment_sha,
            "status": "infrastructure-failure" if failure else "completed",
            "failure": str(
                row.get("failure")
                or row.get("failure_trace_sha256")
                or document.get("traceback_sha256", "")
            ),
            "metrics": {} if failure else metric_row(row, meta),
        }
        result["evaluation_sha256"] = ""
        evaluation_sha = value_sha256(result)
        result["evaluation_sha256"] = evaluation_sha
        if evaluation_sha not in existing:
            told.append(search.tell(result))
            existing.add(evaluation_sha)
    batch["status"] = "failed" if failure else "completed"
    batch["source_results"].append(
        {
            "path": str(source),
            "file_sha256": source_sha,
            "content_sha256": document["content_sha256"],
            "status": batch["status"],
            "evaluations": len(rows),
        }
    )
    if all(item["status"] in {"completed", "failed"} for item in plan["batches"]):
        plan["status"] = "ingested"
    atomic_json(plan_path, plan)
    for summary in value["plans"]:
        if summary["index"] == plan["identity"]["index"]:
            summary["status"] = plan["status"]
    value["ingested_sources"].append(source_sha)
    value["status"] = "ready" if plan["status"] == "ingested" else "evaluating"
    atomic_json(root / "campaign.json", value)
    print(
        json.dumps(
            {
                "source_sha256": source_sha,
                "new_evaluations": len(told),
                "batch_status": batch["status"],
                "plan_status": plan["status"],
            },
            sort_keys=True,
        )
    )


def scores_command(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    _, search = campaign(root)
    scores = load_json(args.scores.resolve())
    search.register_proposal_scores(scores)
    print(
        json.dumps(
            {"artifact": scores["artifact_sha256"], "pairs": len(scores["scores"])}
        )
    )


def frontier_command(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    _, search = campaign(root)
    print(json.dumps(search.environment_frontier(), sort_keys=True))


def retry_command(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    _, search = campaign(root)
    search.authorize_infrastructure_retry(args.candidate, args.environment)
    print(json.dumps({"candidate": args.candidate, "environment": args.environment}))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--profile", type=Path, required=True)
    init.add_argument("--controller", type=Path, required=True)
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--worlds-per-batch", type=int, required=True)
    plan.add_argument("--candidate-waves", type=int, required=True)
    plan.add_argument("--selection", choices=("evolve", "challenge"), required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--result", type=Path, required=True)
    scores = sub.add_parser("register-challenge-scores")
    scores.add_argument("--output", type=Path, required=True)
    scores.add_argument("--scores", type=Path, required=True)
    frontier = sub.add_parser("environment-frontier")
    frontier.add_argument("--output", type=Path, required=True)
    retry = sub.add_parser("authorize-infrastructure-retry")
    retry.add_argument("--output", type=Path, required=True)
    retry.add_argument("--candidate", required=True)
    retry.add_argument("--environment", required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.command == "init":
        init_command(args)
    elif args.command == "plan":
        plan_command(args)
    elif args.command == "ingest":
        ingest_command(args)
    elif args.command == "register-challenge-scores":
        scores_command(args)
    elif args.command == "environment-frontier":
        frontier_command(args)
    else:
        retry_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
