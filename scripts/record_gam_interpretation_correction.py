#!/usr/bin/env python3
"""Build one typed evidence batch for an authenticated GAM interpretation correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.population_evidence import (
    BATCH_FORMAT,
    LEDGER_FORMAT,
    PopulationEvidenceError,
    atomic_write_json,
    canonical_bytes,
    evidence_record,
    local_blob,
    read_json,
    sha256_file,
    validate_records,
)


CORRECTION_FORMAT = "chreatures-population-gam-score-semantics-correction-v1"


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PopulationEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    ledger = read_json(args.ledger)
    if ledger.get("format") != LEDGER_FORMAT:
        raise PopulationEvidenceError("unsupported population evidence ledger")
    validate_records(ledger["records"], campaign_id=ledger["campaign_id"])
    correction = read_json(args.correction)
    if correction.get("format") != CORRECTION_FORMAT:
        raise PopulationEvidenceError("unsupported GAM interpretation correction")
    correction_sha = sha256_file(args.correction)
    bank_sha = _hash(correction.get("bank_sha256"), "corrected GAM bank")
    support_sha = _hash(
        correction.get("original_support_report_sha256"), "corrected support report"
    )
    surface_sha = _hash(correction.get("surface_sha256"), "corrected surface")
    parent_id = f"gam-fit:{bank_sha}"
    by_id = {record["id"]: record for record in ledger["records"]}
    parent = by_id.get(parent_id)
    if parent is None or parent.get("record_type") != "gam_fit_attempt" or parent.get(
        "fields", {}
    ).get("status") != "completed":
        raise PopulationEvidenceError("correction does not name a completed GAM fit")
    if any(
        parent["fields"].get(key) != expected
        for key, expected in (
            ("bank_sha256", bank_sha),
            ("support_report_sha256", support_sha),
            ("surface_sha256", surface_sha),
        )
    ):
        raise PopulationEvidenceError("correction identities differ from the fitted-law node")
    if correction.get("status") != (
        "supplementary interpretation correction; fitted bank and source evidence unchanged"
    ):
        raise PopulationEvidenceError("correction status overstates its scope")
    record = evidence_record(
        id=f"interpretation-correction:{correction_sha}",
        time={"domain": "analysis_sequence", "value": args.analysis_sequence},
        record_type="interpretation_correction",
        text="Supplementary correction to fitted-score interpretation; model unchanged.",
        parents={parent_id: "corrected_record"},
        blobs=[local_blob(
            args.correction,
            role="correction_receipt",
            media_type="application/json",
        )],
        fields={
            "correction_sha256": correction_sha,
            "bank_sha256": bank_sha,
            "original_support_report_sha256": support_sha,
            "surface_sha256": surface_sha,
            "status": "supplementary-interpretation-correction",
            "claims": str(correction.get("claims", "")),
        },
    )
    prior = by_id.get(record["id"])
    if prior is not None and canonical_bytes(prior) != canonical_bytes(record):
        raise PopulationEvidenceError("stable interpretation correction record changed")
    records = [record]
    validate_records(
        ledger["records"] if prior is not None else [*ledger["records"], record],
        campaign_id=ledger["campaign_id"],
    )
    digest = hashlib.sha256(canonical_bytes(records)).hexdigest()
    batch = {
        "format": BATCH_FORMAT,
        "schema_version": 1,
        "campaign_id": ledger["campaign_id"],
        "batch_id": f"population-campaign:{digest}",
        "records": records,
        "sources": {
            "correction_file_sha256": correction_sha,
            "corrected_record_id": parent_id,
        },
    }
    if args.output.exists():
        if read_json(args.output) != batch:
            raise PopulationEvidenceError("existing correction batch differs")
    else:
        atomic_write_json(args.output, batch)
    applied = batch["batch_id"] in ledger.get("applied_batches", [])
    if (prior is not None) != applied:
        raise PopulationEvidenceError("correction record and applied batch disagree")
    return {
        "status": "unchanged" if prior is not None else "recorded",
        "output": str(args.output),
        "batch_id": batch["batch_id"],
        "record_id": record["id"],
        "correction_sha256": correction_sha,
        "file_sha256": sha256_file(args.output),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--correction", required=True, type=Path)
    parser.add_argument("--analysis-sequence", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.analysis_sequence < 0:
        parser.error("--analysis-sequence must be nonnegative")
    return args


def main() -> None:
    print(json.dumps(build(arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
