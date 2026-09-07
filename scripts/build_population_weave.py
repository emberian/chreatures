#!/usr/bin/env python3
"""Append typed population batches and build a native Universal Weave ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from chreatures.population_evidence import (
    SCHEMA,
    append_batches,
    atomic_write_json,
    empty_ledger,
    read_json,
    reconcile_population_state,
    sha256_file,
    validate_records,
    weave_request,
)


ROOT = Path(__file__).resolve().parents[1]
WEAVE_CRATE = ROOT / "integrations" / "weave"


def _default_output(ledger: Path, suffix: str) -> Path:
    name = ledger.name[:-5] if ledger.name.endswith(".json") else ledger.name
    return ledger.with_name(f"{name}{suffix}")


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with source.open("rb") as incoming, temporary.open("wb") as outgoing:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
            outgoing.write(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    temporary.replace(target)


def _native_build(request: dict[str, Any], weave_output: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="chreatures-population-weave-") as raw:
        temporary_dir = Path(raw)
        request_path = temporary_dir / "request.json"
        weave_path = temporary_dir / "population.weave.json"
        atomic_write_json(request_path, request)
        command = [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--",
            "--input",
            str(request_path),
            "--output",
            str(weave_path),
            "--compact-receipt",
        ]
        completed = subprocess.run(
            command,
            cwd=WEAVE_CRATE,
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(completed.stdout)
        if (
            receipt.get("validated_after_reload") is not True
            or receipt.get("reload_equal") is not True
            or receipt.get("archive", {}).get("evidence_schema") != SCHEMA
        ):
            raise RuntimeError("native Universal Weave round trip was not verified")
        _atomic_copy(weave_path, weave_output)
    receipt["artifact"] = str(weave_output)
    receipt["bytes"] = weave_output.stat().st_size
    return receipt


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.ledger.exists():
        ledger = read_json(args.ledger)
        if args.campaign_id is not None and ledger.get("campaign_id") != args.campaign_id:
            raise ValueError("--campaign-id differs from the existing ledger")
    else:
        if args.campaign_id is None or args.description is None:
            raise ValueError("a new ledger requires --campaign-id and --description")
        ledger = empty_ledger(args.campaign_id, args.description)

    batches = [read_json(path) for path in args.batch]
    if batches:
        ledger = append_batches(ledger, batches)
    else:
        validate_records(ledger["records"], campaign_id=ledger["campaign_id"])
    request = weave_request(ledger)
    reconciled_states = []
    for path in args.population_state:
        reconciliation = reconcile_population_state(ledger["records"], read_json(path))
        reconciled_states.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                **reconciliation,
            }
        )

    request_output = args.request_output or _default_output(args.ledger, ".weave-request.json")
    weave_output = args.weave_output or _default_output(args.ledger, ".weave.json")
    receipt_output = args.receipt_output or _default_output(args.ledger, ".weave-receipt.json")
    native = _native_build(request, weave_output)

    types = Counter(record["record_type"] for record in ledger["records"])
    terminal_count = types["evaluation_completed"] + types["evaluation_failed"]
    statuses = Counter(
        record.get("fields", {}).get("status")
        for record in ledger["records"]
        if record["record_type"] in {"evaluation_completed", "evaluation_failed"}
    )
    receipt = {
        "format": "chreatures-population-weave-build-receipt-v2",
        "schema": SCHEMA,
        "campaign_id": ledger["campaign_id"],
        "applied_batches": list(ledger["applied_batches"]),
        "record_count": len(ledger["records"]),
        "record_types": dict(sorted(types.items())),
        "terminal_evaluations": terminal_count,
        "infrastructure_failures_retained": statuses["infrastructure-failure"],
        "organism_terminal_evaluations": statuses["organism-terminal"],
        "reconciled_population_states": reconciled_states,
        "artifacts": {
            "ledger": {
                "path": str(args.ledger),
                "sha256": hashlib.sha256(
                    json.dumps(
                        ledger, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                    + b"\n"
                ).hexdigest(),
            },
            "request": {"path": str(request_output)},
            "weave": {
                "path": str(weave_output),
                "sha256": sha256_file(weave_output),
                "bytes": weave_output.stat().st_size,
            },
        },
        "native": {
            "integration": native["integration"],
            "library": native["library"],
            "node_id_contract": native["node_id_contract"],
            "node_count": native["node_count"],
            "edge_count": native["edge_count"],
            "multi_parent_nodes": native["multi_parent_nodes"],
            "reload_equal": native["reload_equal"],
            "validated_after_reload": native["validated_after_reload"],
        },
        "scope": "external population evidence; never organism memory or runtime state",
    }

    # Publish the coherent set only after typed Python validation and the native
    # serialize/reload check both succeed. The source batch files are immutable.
    atomic_write_json(args.ledger, ledger)
    atomic_write_json(request_output, request)
    receipt["artifacts"]["ledger"]["sha256"] = sha256_file(args.ledger)
    receipt["artifacts"]["request"].update(
        sha256=sha256_file(request_output), bytes=request_output.stat().st_size
    )
    atomic_write_json(receipt_output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ledger", type=Path, required=True)
    result.add_argument("--batch", type=Path, action="append", default=[])
    result.add_argument(
        "--population-state",
        type=Path,
        action="append",
        default=[],
        help="native search snapshot whose complete evaluation list must be present",
    )
    result.add_argument("--campaign-id")
    result.add_argument("--description")
    result.add_argument("--request-output", type=Path)
    result.add_argument("--weave-output", type=Path)
    result.add_argument("--receipt-output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    receipt = build(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
