#!/usr/bin/env python3
"""Build an idempotent population-evidence batch from one public reef recording."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from chreatures.population_evidence import (
    BATCH_FORMAT,
    LEDGER_FORMAT,
    PopulationEvidenceError,
    atomic_write_json,
    canonical_bytes,
    local_blob,
    read_json,
    recording_evidence_records,
    sha256_file,
    validate_records,
)


LINK_FORMAT = "chreatures-living-recording-evidence-link-v1"


def _bindings(value: Mapping[str, Any]) -> dict[int, str]:
    if value.get("format") != LINK_FORMAT:
        raise PopulationEvidenceError("unsupported recording evidence link format")
    raw = value.get("body_life_record_ids")
    if not isinstance(raw, Mapping):
        raise PopulationEvidenceError("recording evidence link lacks body-life bindings")
    result = {}
    for key, record_id in raw.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise PopulationEvidenceError("recording evidence link has invalid body index")
        if not isinstance(record_id, str) or not record_id:
            raise PopulationEvidenceError("recording evidence link has invalid life record id")
        result[int(key)] = record_id
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    ledger = read_json(args.ledger)
    if ledger.get("format") != LEDGER_FORMAT:
        raise PopulationEvidenceError("ledger format differs")
    link = read_json(args.link)
    if link.get("campaign_id") != ledger.get("campaign_id"):
        raise PopulationEvidenceError("recording link and ledger campaign differ")
    recording = read_json(args.recording)
    laws = link.get("associated_law_fit_record_ids", [])
    event_laws = link.get("event_law_fit_record_ids", {})
    if not isinstance(laws, list) or not isinstance(event_laws, Mapping):
        raise PopulationEvidenceError("recording link law-fit references are invalid")
    records = recording_evidence_records(
        recording,
        recording_artifact=local_blob(
            args.recording,
            role="public_recording",
            media_type="application/json",
        ),
        campaign_record_id=str(link.get("campaign_record_id", "")),
        environment_record_id=str(link.get("environment_record_id", "")),
        body_life_record_ids=_bindings(link),
        associated_law_fit_record_ids=laws,
        event_law_fit_record_ids=event_laws,
    )
    existing = {record["id"]: record for record in ledger.get("records", [])}
    new_records = []
    for record in records:
        previous = existing.get(record["id"])
        if previous is None:
            new_records.append(record)
        elif canonical_bytes(previous) != canonical_bytes(record):
            raise PopulationEvidenceError(
                f"stable recording evidence {record['id']} changed"
            )
    if len(new_records) not in {0, len(records)}:
        raise PopulationEvidenceError("recording evidence was only partially applied")
    validate_records(
        [*ledger["records"], *new_records], campaign_id=ledger["campaign_id"]
    )
    digest = hashlib.sha256(canonical_bytes(records)).hexdigest()
    batch_id = f"population-campaign:{digest}"
    applied = set(ledger.get("applied_batches", []))
    if not new_records and batch_id not in applied:
        raise PopulationEvidenceError(
            "recording records exist without their authenticated applied batch"
        )
    if new_records and batch_id in applied:
        raise PopulationEvidenceError("recording batch is applied but its records are absent")
    batch = {
        "format": BATCH_FORMAT,
        "schema_version": 1,
        "campaign_id": ledger["campaign_id"],
        "batch_id": batch_id,
        "records": records,
        "sources": {
            "recording_file_sha256": sha256_file(args.recording),
            "recording_content_sha256": recording["content_sha256"],
            "link_file_sha256": sha256_file(args.link),
        },
    }
    if args.output.exists():
        if read_json(args.output) != batch:
            raise PopulationEvidenceError("existing recording batch differs")
    else:
        atomic_write_json(args.output, batch)
    return {
        "status": "recorded" if new_records else "unchanged",
        "output": str(args.output),
        "batch_id": batch["batch_id"],
        "new_records": len(new_records),
        "record_types": sorted({record["record_type"] for record in new_records}),
        "recording_content_sha256": recording["content_sha256"],
        "file_sha256": sha256_file(args.output),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--link", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
