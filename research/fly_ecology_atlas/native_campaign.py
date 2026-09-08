#!/usr/bin/env python3
"""Run the 24-by-3 native ecology atlas at bounded M2 concurrency."""
from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.workers not in (1, 2):
        raise ValueError("M2 campaign concurrency must be one or two")
    plan = json.loads(a.plan.read_text())
    a.results.mkdir(parents=True, exist_ok=True)
    layouts = [x for x in plan["layouts"] if x["split"] == "fit-layout"]
    jobs = [(g, l) for g in plan["genotypes"] for l in layouts]

    def run(job):
        g, l = job
        out = a.results / f"{g['genotype_id']}--{l['layout_id']}.json"
        if out.exists():
            return str(out), "preserved"
        subprocess.run(
            [
                sys.executable,
                plan["runner"],
                "--plan",
                str(a.plan),
                "--genotype",
                g["genotype_id"],
                "--layout",
                l["layout_id"],
                "--output",
                str(out),
            ],
            check=True,
        )
        return str(out), "completed"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for done in as_completed([pool.submit(run, j) for j in jobs]):
            print(
                json.dumps(dict(zip(("output", "status"), done.result()))), flush=True
            )


if __name__ == "__main__":
    main()
