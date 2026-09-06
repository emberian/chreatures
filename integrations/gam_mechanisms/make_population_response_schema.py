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
SMOOTH_FEATURES = {
    "energy_state_delta": ["energy", "gut", "fatigue", "history_energy_mean", "thrust", "eat", "allocate"],
    "fatigue_state_delta": ["fatigue", "history_fatigue_mean", "speed", "thrust", "yaw", "posture", "grip"],
    "effort": ["fatigue", "history_fatigue_mean", "speed", "thrust", "yaw", "posture", "grip"],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--heldout-lineage", action="append", default=[])
    parser.add_argument("--heldout-candidate", action="append", default=[])
    parser.add_argument("--heldout-environment", action="append", default=[])
    parser.add_argument("--validation-candidate", action="append", default=[])
    parser.add_argument("--target", action="append", choices=[x[0] for x in TARGETS],
                        help="explicit supported-law subset; default attempts all measured targets")
    parser.add_argument("--training-tick-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.training_tick_stride <= 64:
        raise ValueError("training tick stride must be in 1..64")
    contract = json.loads(args.feature_contract.read_text())
    receipt_path = args.rows.with_suffix(".receipt.json")
    rows_receipt = json.loads(receipt_path.read_text())
    with np.load(args.rows, allow_pickle=False) as rows:
        target = np.asarray(rows["targets"], dtype=float)
        if target.ndim != 2 or target.shape[1] != len(TARGETS) or not np.isfinite(target).all():
            raise ValueError("prepared population targets differ")
        lineage = np.asarray(rows["lineage_unit"]).astype(str)
        environment = np.asarray(rows["environment_unit"]).astype(str)
        candidate = np.asarray(rows["candidate_unit"]).astype(str)
        fit_rows = ~(np.isin(lineage, args.heldout_lineage)
                     | np.isin(candidate, args.heldout_candidate)
                     | np.isin(environment, args.heldout_environment))
        fit_rows &= ~np.isin(candidate, args.validation_candidate)
        fit_rows &= (np.asarray(rows["tick_unit"], dtype=np.uint64)
                     % args.training_tick_stride == 0)
        if not args.validation_candidate:
            fit_rows &= np.asarray(rows["world_unit"], dtype=np.int64) % 5 != 0
        if not fit_rows.any() or fit_rows.all():
            raise ValueError("held-out lineage/environment split is empty")
        selected = set(args.target or [x[0] for x in TARGETS])
        responses = []
        score_scales = {}
        for column, (law, mechanism, unit, kind) in enumerate(TARGETS):
            if law not in selected:
                continue
            extent = max(float(np.max(np.abs(target[fit_rows, column]))) * 1.05, 1e-6)
            transform = ({"kind": kind, "magnitude": extent} if kind == "signed_tanh"
                         else {"kind": kind, "ceiling": extent})
            responses.append({"law": law, "mechanism": mechanism, "unit": unit,
                              "target_column": column, "transform": transform,
                              "smooth_features": SMOOTH_FEATURES.get(law,
                                  [item["name"] for item in contract["features"]])})
            score_scales[mechanism] = max(float(np.std(target[fit_rows, column])), 1e-6)
    schema = {
        "format": "chreatures-population-response-fit-v1",
        "feature_contract_sha256": digest(args.feature_contract),
        "features": contract["features"],
        "responses": responses,
        "budgets": [],
        "candidate_score": {"maximum_tilt": 0.25, "terms": [
            {"mechanism":"expected_energy_state_delta", "weight":1.0, "scale":score_scales["expected_energy_state_delta"]},
            *([{"mechanism":"expected_fatigue_state_delta", "weight":-0.75,
                "scale":score_scales["expected_fatigue_state_delta"]}]
              if "fatigue_state_delta" in selected else []),
            {"mechanism":"expected_effort", "weight":-0.5, "scale":score_scales["expected_effort"]},
        ]} if {"energy_state_delta", "effort"}.issubset(selected) else None,
        "basis_size": 9,
        "training_tick_stride": args.training_tick_stride,
        "split": {"heldout_lineages": args.heldout_lineage,
                  "heldout_candidates": args.heldout_candidate,
                  "heldout_environments": args.heldout_environment,
                  "validation_candidates": args.validation_candidate,
                  "validation_world_mod": 5},
        "source_contract": {
            "format": "chreatures-population-gam-trace-v1",
            "dt_seconds": 0.05,
            "history_window_ticks": 64,
            "target_semantics": "observed one-step associations under executed actions; not causal effects",
            "training_sampling":"target-blind tick modulo stride equals zero within every fit life",
            "rows_sha256": digest(args.rows),
            "rows_receipt_sha256": digest(receipt_path),
            "source_status": rows_receipt["source_status"],
            "trace_feature_contract_sha256": rows_receipt["trace_feature_contract_sha256"],
            "fit_feature_contract_sha256": rows_receipt["fit_feature_contract_sha256"],
            "feature_identity_mapping": rows_receipt["feature_identity_mapping"],
            "censored_after_complete_trace_tick": rows_receipt["censored_after_complete_trace_tick"],
            "planned_physical_ticks": rows_receipt["planned_physical_ticks"],
            "terminal_content_sha256": rows_receipt["terminal_content_sha256"],
        },
    }
    args.output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"schema": str(args.output), "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
