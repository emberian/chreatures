#!/usr/bin/env python3
"""Gate selected native-GAM candidates on the untouched fourth physical layout."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.dynamics_v2.gam_fit import _require_native_gamfit
from research.fly_ecology_atlas.native_fit import ENV, TARGETS, metrics


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    fit = json.loads((a.fit / "report.json").read_text())
    gam = _require_native_gamfit()
    layout = next(x for x in plan["layouts"] if x["split"] == "confirmation")
    records = []
    for proposal in plan["genotypes"]:
        path = a.results / f"{proposal['genotype_id']}--{layout['layout_id']}.json"
        run = json.loads(path.read_text())
        if run["genotype"]["genes"] != proposal["genes"] or run["layout"] != layout:
            raise ValueError("final holdout run identity differs")
        observed = metrics(run)
        query = {
            **proposal["genes"],
            **{name: observed[name] for name in ENV},
            "response": 0.0,
        }
        predicted = {}
        limits = {}
        checks = {}
        for target in TARGETS:
            label = fit["selected_models"][target]
            model = gam.load(a.fit / f"{target}-{label}.gam")
            predicted[target] = float(np.asarray(model.predict([query])).reshape(-1)[0])
            diagnostic = fit["diagnostics"][target]["models"][label]
            limits[target] = 2 * max(
                diagnostic["validation_genotype_holdout"]["rmse"],
                diagnostic["fit_layout_holdout"]["rmse"],
                1e-9,
            )
            checks[target] = abs(predicted[target] - observed[target]) <= limits[target]
        response_gate_holds = (
            all(checks.values()) and 0 <= observed["mean_route_aperture"] <= 1
        )
        # The frozen runner omitted owner_id structure records. Prepared-site counters
        # cannot substitute for committed branching, so promotion remains impossible.
        holds = False
        records.append(
            {
                "proposal": proposal,
                "run_sha256": sha(path),
                "observed": observed,
                "predicted_from_final_holdout_initial_geometry": predicted,
                "absolute_error": {k: abs(predicted[k] - observed[k]) for k in TARGETS},
                "limits": limits,
                "checks": checks,
                "response_gate_holds": response_gate_holds,
                "committed_branching_observed": False,
                "holds": holds,
            }
        )
    report = {
        "format": "chreatures-native-fly-ecology-confirmation-v2",
        "status": "not-promoted-insufficient-committed-structure-observation",
        "parent_plan_sha256": plan["parent_plan_sha256"],
        "fit_report_sha256": sha(a.fit / "report.json"),
        "layout": layout,
        "records": records,
        "promoted": [],
        "claim_limit": "Final fourth-layout validation of aperture/resource laws for synthetic colonies only. Frozen traces omitted owner_id structure records; clearance-approved site counters are not committed branches and cannot authorize promotion or a fly-skill claim.",
    }
    a.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(a.output),
                "sha256": sha(a.output),
                "status": report["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
