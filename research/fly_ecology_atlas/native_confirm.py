#!/usr/bin/env python3
"""Prepare or evaluate untouched-layout checks for a GAM diversity proposal set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.fly_ecology_atlas.native_fit import TARGETS, metrics, sha, write
from research.fly_ecology_atlas.prepare_native import FORMAT


def prepare(parent_path: Path, fit_dir: Path, output: Path) -> None:
    parent = json.loads(parent_path.read_text())
    report = json.loads((fit_dir / "report.json").read_text())
    if report["plan_sha256"] != sha(parent_path):
        raise ValueError("fit report parent differs")
    layout = next(item for item in parent["layouts"] if item["split"] == "confirmation")
    proposals = report["confirmation_proposals"]
    if not proposals:
        raise ValueError("fit produced no diversity proposals")
    units = []
    for index, genotype in enumerate(proposals):
        units.append(
            {
                "unit_id": f"{genotype['genotype_id']}--{layout['layout_id']}",
                "genotype": genotype,
                "layout": layout,
                "replicate": 0,
                "ecology_seed": parent["seed"] + 900_000 + 2 * index,
                "physics_seed": parent["seed"] + 900_001 + 2 * index,
            }
        )
    plan = {
        **parent,
        "status": "planned-not-executed-root-pin-required",
        "genotypes": proposals,
        "layouts": [layout],
        "units": units,
        "parent_plan_sha256": sha(parent_path),
        "fit_report_sha256": sha(fit_dir / "report.json"),
        "confirmation_role": "untouched physical layout; outcomes were absent from fit and proposal selection",
    }
    write(output, plan)


def evaluate(plan_path: Path, fit_dir: Path, results: Path, output: Path) -> None:
    plan = json.loads(plan_path.read_text())
    fit = json.loads((fit_dir / "report.json").read_text())
    if plan.get("format") != FORMAT or plan["fit_report_sha256"] != sha(
        fit_dir / "report.json"
    ):
        raise ValueError("confirmation identity differs")
    records = []
    failures = []
    for unit in plan["units"]:
        path = results / f"{unit['unit_id']}.json"
        run = json.loads(path.read_text())
        if run.get("unit") != unit or run["provenance"]["plan_sha256"] != sha(plan_path):
            raise ValueError("confirmation run identity differs")
        if not run.get("completed"):
            failures.append(
                {"unit_id": unit["unit_id"], "run_sha256": sha(path), "error": run["error"]}
            )
            continue
        observed = metrics(run)
        predicted = unit["genotype"]["predicted_fit_layout_mean"]
        errors = {
            target: abs(float(predicted[target]) - observed[target]) for target in TARGETS
        }
        limits = {}
        checks = {}
        for target in TARGETS:
            label = fit["selected_models"].get(target)
            if label is None:
                continue
            diagnostic = fit["diagnostics"][target]["models"][label]
            limit = 2 * max(
                diagnostic["leave_genotype_out"]["rmse"],
                diagnostic["leave_layout_out"]["rmse"],
                1e-9,
            )
            limits[target] = limit
            checks[target] = errors[target] <= limit
        records.append(
            {
                "proposal": unit["genotype"],
                "run_sha256": sha(path),
                "observed": observed,
                "absolute_error_from_fit_layout_mean": errors,
                "diagnostic_limits": limits,
                "checks": checks,
                "all_modeled_checks_hold": bool(checks) and all(checks.values()),
            }
        )
    response = np.asarray(
        [[record["observed"][target] for target in TARGETS] for record in records],
        np.float64,
    )
    diversity = None
    if len(response) > 1:
        normalized = (response - response.mean(0)) / np.maximum(response.std(0), 1e-12)
        distance = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
        diversity = {
            "minimum_pairwise_standardized_response_distance": float(
                distance[np.triu_indices(len(response), 1)].min()
            ),
            "maximum_pairwise_standardized_response_distance": float(distance.max()),
        }
    report = {
        "format": "chreatures-native-fly-ecology-confirmation-v3",
        "status": "diversity-confirmation-executed-no-promoted-winner",
        "plan_sha256": sha(plan_path),
        "fit_report_sha256": sha(fit_dir / "report.json"),
        "records": records,
        "failed_units": failures,
        "realized_response_diversity": diversity,
        "promoted": [],
        "claim_limit": plan["claim_limit"],
    }
    write(output, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.plan, args.fit, args.output)
    else:
        evaluate(args.plan, args.fit, args.results, args.output)
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
