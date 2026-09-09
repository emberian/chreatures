#!/usr/bin/env python3
"""Run sealed independent atlas worlds with bounded CPU concurrency and budget."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--budget-minutes", type=float, default=60.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.budget_minutes <= 0:
        raise ValueError("campaign worker/budget bounds differ")
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "planned-not-executed-root-pin-required":
        raise ValueError("campaign plan status differs")
    args.results.mkdir(parents=True, exist_ok=True)
    pending = list(plan["units"])
    deadline = time.monotonic() + args.budget_minutes * 60

    def run(unit: dict) -> dict:
        output = args.results / f"{unit['unit_id']}.json"
        if output.exists():
            record = json.loads(output.read_text())
            return {
                "unit": unit["unit_id"],
                "status": "preserved-complete" if record.get("completed") else "preserved-failed",
                "output": str(output),
            }
        result = subprocess.run(
            [
                sys.executable,
                plan["runner"],
                "--plan",
                str(args.plan),
                "--unit",
                unit["unit_id"],
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if not output.exists():
            raise RuntimeError(
                f"runner did not seal {unit['unit_id']}: {result.stderr[-1000:]}"
            )
        record = json.loads(output.read_text())
        return {
            "unit": unit["unit_id"],
            "status": "completed" if record.get("completed") else "failed-preserved",
            "output": str(output),
            "runner_exit": result.returncode,
        }

    active = set()
    launched = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or active:
            while pending and len(active) < args.workers and time.monotonic() < deadline:
                active.add(pool.submit(run, pending.pop(0)))
                launched += 1
            if not active:
                break
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                print(json.dumps(future.result(), separators=(",", ":")), flush=True)
    print(
        json.dumps(
            {
                "status": "budget-stopped" if pending else "all-units-accounted",
                "launched_this_invocation": launched,
                "not_launched": [unit["unit_id"] for unit in pending],
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
