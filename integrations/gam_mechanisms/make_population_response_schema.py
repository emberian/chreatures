#!/usr/bin/env python3
"""Seal a fit schema for measured population targets and explicit holdouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TARGETS = (
    ("energy_state_delta", "expected_energy_state_delta", "dimensionless state delta", "signed_tanh"),
    ("fatigue_state_delta", "expected_fatigue_state_delta", "dimensionless state delta", "signed_tanh"),
    ("effort", "expected_effort", "training-cohort effort per 50 ms", "positive_softplus"),
    ("ingested_mass", "expected_ingested_mass", "elemental-equivalent mass per 50 ms", "positive_softplus"),
    ("contact", "expected_contact", "training-cohort contact measure per 50 ms", "positive_softplus"),
    ("release_mass", "expected_release_mass", "elemental-equivalent mass per 50 ms", "positive_softplus"),
    ("secretion_mass", "expected_secretion_mass", "elemental-equivalent mass per 50 ms", "positive_softplus"),
    ("allocation_mass", "expected_allocation_mass", "elemental-equivalent mass per 50 ms", "positive_softplus"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--heldout-lineage", action="append", required=True)
    parser.add_argument("--heldout-environment", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.feature_contract.read_text())
    with np.load(args.rows, allow_pickle=False) as rows:
        target = np.asarray(rows["targets"], dtype=float)
        if target.ndim != 2 or target.shape[1] != len(TARGETS) or not np.isfinite(target).all():
            raise ValueError("prepared population targets differ")
        lineage = np.asarray(rows["lineage_unit"]).astype(str)
        environment = np.asarray(rows["environment_unit"]).astype(str)
        fit_rows = ~(np.isin(lineage, args.heldout_lineage)
                     | np.isin(environment, args.heldout_environment))
        if not fit_rows.any() or fit_rows.all():
            raise ValueError("held-out lineage/environment split is empty")
        responses = []
        score_scales = {}
        for column, (law, mechanism, unit, kind) in enumerate(TARGETS):
            extent = max(float(np.max(np.abs(target[fit_rows, column]))) * 1.05, 1e-6)
            transform = ({"kind": kind, "magnitude": extent} if kind == "signed_tanh"
                         else {"kind": kind, "ceiling": extent})
            responses.append({"law": law, "mechanism": mechanism, "unit": unit,
                              "transform": transform})
            score_scales[mechanism] = max(float(np.std(target[fit_rows, column])), 1e-6)
    schema = {
        "format": "chreatures-population-response-fit-v1",
        "feature_contract_sha256": digest(args.feature_contract),
        "features": contract["features"],
        "responses": responses,
        "budgets": [],
        "candidate_score": {"maximum_tilt": 0.25, "terms": [
            {"mechanism":"expected_energy_state_delta", "weight":1.0, "scale":score_scales["expected_energy_state_delta"]},
            {"mechanism":"expected_fatigue_state_delta", "weight":-1.0, "scale":score_scales["expected_fatigue_state_delta"]},
            {"mechanism":"expected_effort", "weight":-0.5, "scale":score_scales["expected_effort"]},
            {"mechanism":"expected_ingested_mass", "weight":0.5, "scale":score_scales["expected_ingested_mass"]},
        ]},
        "basis_size": 9,
        "split": {"heldout_lineages": args.heldout_lineage,
                  "heldout_environments": args.heldout_environment,
                  "validation_world_mod": 5},
        "source_contract": {
            "format": "chreatures-population-gam-trace-v1",
            "dt_seconds": 0.05,
            "history_window_ticks": 64,
            "target_semantics": "observed one-step associations under executed actions; not causal effects",
            "rows_sha256": digest(args.rows),
        },
    }
    args.output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"schema": str(args.output), "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
