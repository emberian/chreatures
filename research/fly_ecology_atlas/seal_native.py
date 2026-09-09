#!/usr/bin/env python3
"""Seal compact evidence for the committed-response native GAM campaign."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from research.fly_ecology_atlas.native_fit import TARGETS, sha, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    fit = json.loads((args.fit / "report.json").read_text())
    fit_receipt = json.loads((args.fit / "receipt.json").read_text())
    confirmation = json.loads(args.confirmation.read_text())
    if (
        fit["plan_sha256"] != sha(args.plan)
        or fit_receipt["report_sha256"] != sha(args.fit / "report.json")
        or confirmation["fit_report_sha256"] != sha(args.fit / "report.json")
    ):
        raise ValueError("campaign receipt chain differs")
    model_artifacts = []
    for target in TARGETS:
        label = fit["selected_models"].get(target)
        if label is None:
            continue
        path = args.fit / f"{target}-{label}.gam"
        model_artifacts.append(
            {"target": target, "model": label, "sha256": sha(path)}
        )
    native_module = importlib.import_module("gamfit._rust")
    response_ranges = {}
    for target in TARGETS:
        values = [item["metrics"][target] for item in fit["records"]]
        if values:
            response_ranges[target] = [min(values), max(values)]
    receipt = {
        "format": "chreatures-native-fly-ecology-atlas-receipt-v3",
        "status": confirmation["status"],
        "campaign": "committed-construction-resource-aperture-native-host",
        "plan_sha256": sha(args.plan),
        "native_binary_sha256": plan["native_binary_sha256"],
        "runner_sha256": plan["runner_sha256"],
        "fit_report_sha256": sha(args.fit / "report.json"),
        "fit_receipt_sha256": sha(args.fit / "receipt.json"),
        "confirmation_sha256": sha(args.confirmation),
        "completed_fit_worlds": len(fit["records"]),
        "failed_fit_worlds": len(fit["failed_units"]),
        "completed_confirmation_worlds": len(confirmation["records"]),
        "failed_confirmation_worlds": len(confirmation["failed_units"]),
        "experimental_unit": plan["experimental_unit"],
        "selected_native_gam_models": model_artifacts,
        "native_gam": fit["native_gam"],
        "native_gam_module_sha256": sha(native_module.__file__),
        "response_ranges": response_ranges,
        "diversity_proposals": fit["confirmation_proposals"],
        "realized_response_diversity": confirmation["realized_response_diversity"],
        "next_environment_priorities": fit["next_environment_priorities"],
        "historical_aperture_rows_reused": False,
        "historical_evidence_status": "preserved separately; not pooled because committed-growth and immediate-route-invalidation contracts differ",
        "counter_semantics": {
            "construction_clearance_approved": "host clearance only; never a committed branch",
            "committed_constructions": "post-core and post-physical commit with owner_id/physics_binding verification",
            "construction_material_allocated": "cumulative postcommit transfer quantities by pool",
        },
        "promoted": [],
        "claim_limit": plan["claim_limit"],
    }
    write(args.output, receipt)
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
