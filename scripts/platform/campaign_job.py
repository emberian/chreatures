#!/usr/bin/env python3
"""Seal, validate, launch, resume, and inspect a campaign job manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chreatures.campaign_jobs import (  # noqa: E402
    CampaignJobError,
    atomic_json,
    job_status,
    launch_job,
    load_manifest,
    seal_manifest,
    supervise_job,
    validate_runtime,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("input", type=Path)
    seal.add_argument("output", type=Path)
    for name in ("validate", "launch", "resume", "status"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
    internal = commands.add_parser("_supervise", help=argparse.SUPPRESS)
    internal.add_argument("manifest", type=Path)
    internal.add_argument("attempt", type=int)
    internal.add_argument("mode", choices=("launch", "resume"))
    return parser.parse_args()


def emit(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main() -> int:
    args = arguments()
    try:
        if args.operation == "seal":
            raw = json.loads(args.input.read_text())
            value = seal_manifest(raw)
            atomic_json(args.output, value)
            emit({"manifest": str(args.output.resolve()), "identity_sha256": value["identity_sha256"]})
            return 0
        if args.operation == "validate":
            manifest = load_manifest(args.manifest)
            emit({
                "identity_sha256": manifest["identity_sha256"],
                "runtime": validate_runtime(manifest),
                "status": "valid",
            })
            return 0
        if args.operation in {"launch", "resume"}:
            emit(launch_job(
                args.manifest,
                launcher_path=Path(__file__),
                resume=args.operation == "resume",
            ))
            return 0
        if args.operation == "status":
            emit(job_status(args.manifest))
            return 0
        if args.operation == "_supervise":
            return supervise_job(args.manifest, args.attempt, args.mode)
    except (CampaignJobError, OSError, ValueError) as error:
        print(f"campaign job error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(args.operation)


if __name__ == "__main__":
    raise SystemExit(main())
