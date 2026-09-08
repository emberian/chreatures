#!/usr/bin/env python3
"""Seal compact evidence for the measured-aperture native-host campaign."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hash(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.campaign / "plan.json").read_text())
    fit_dir = a.campaign / "fit-native-gam-amended"
    fit = json.loads((fit_dir / "report.json").read_text())
    confirmation = json.loads((a.campaign / "confirmation-amended.json").read_text())
    runs = list((a.campaign / "results").glob("*.json"))
    confirm_runs = list(
        (a.campaign / "confirmation-results-parent-proposals").glob("*.json")
    )
    if len(runs) != 72 or len(confirm_runs) != 2:
        raise ValueError("campaign is incomplete")
    ranges = {
        target: [
            float(min(r["metrics"][target] for r in fit["records"])),
            float(max(r["metrics"][target] for r in fit["records"])),
        ]
        for target in fit["targets"]
    }
    models = []
    for target, label in fit["selected_models"].items():
        path = fit_dir / f"{target}-{label}.gam"
        models.append({"target": target, "model": label, "sha256": sha(path)})
    diagnostics = {
        target: {
            "sd": data["sd"],
            "selected": fit["selected_models"].get(target),
            "fit_leave_genotype_out": data["models"][fit["selected_models"][target]][
                "fit_leave_genotype_out"
            ],
            "validation_genotype_holdout": data["models"][
                fit["selected_models"][target]
            ]["validation_genotype_holdout"],
            "fit_layout_holdout": data["models"][fit["selected_models"][target]][
                "fit_layout_holdout"
            ],
        }
        for target, data in fit["diagnostics"].items()
        if target in fit["selected_models"]
    }
    receipt = {
        "format": "chreatures-native-fly-ecology-atlas-receipt-v2",
        "status": confirmation["status"],
        "campaign": "measured-route-aperture-native-host",
        "historical_unsampled_campaign_preserved": True,
        "plan_sha256": sha(a.campaign / "plan.json"),
        "native_binary_sha256": plan["native_binary_sha256"],
        "runner_sha256": plan["runner_sha256"],
        "fit_report_sha256": sha(fit_dir / "report.json"),
        "confirmation_sha256": sha(a.campaign / "confirmation-amended.json"),
        "fit_result_set_sha256": tree_hash(runs),
        "confirmation_result_set_sha256": tree_hash(confirm_runs),
        "completed_fit_and_holdout_runs": len(runs),
        "completed_confirmation_runs": len(confirm_runs),
        "failed_runs": 0,
        "duration_seconds_each": plan["seconds"],
        "genotypes": 24,
        "fit_layouts": 3,
        "genotype_holdout_count": 6,
        "response_ranges": ranges,
        "selected_native_gam_models": models,
        "holdout_diagnostics": diagnostics,
        "confirmation_status": confirmation["status"],
        "promoted": confirmation["promoted"],
        "confirmation_records": confirmation["records"],
        "invalidated_metrics": {
            "growth.accepted": "clearance-approved construction and birth sites before ecology commit",
            "growth.blocked": "clearance events; not rejected committed branches and not bounded by proposal count",
            "branch_count": "frozen runner filtered structure owner instead of owner_id and therefore recorded zero",
        },
        "invalidated_analysis_sha256": sha(
            a.campaign / "fit-native-gam-invalidated-precommit-counters/report.json"
        ),
        "frozen_runner_copy_sha256": sha(a.campaign / "frozen-native-run.py"),
        "interpretation": "Measured aperture and colony resource responses varied, and both preserved parent proposals passed those three numerical fourth-layout checks. Neither is promoted because the frozen trace cannot reconstruct committed structure counts. Prepared-site counters are excluded from the amended GAM evidence and make no construction or fly-skill claim.",
        "claim_limit": plan["claim_limit"],
    }
    a.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(a.output),
                "sha256": sha(a.output),
                "status": receipt["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
