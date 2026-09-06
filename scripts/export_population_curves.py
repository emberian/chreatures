#!/usr/bin/env python3
"""Project per-life cumulative curves from committed evaluator telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = {
    "energy_change_since_birth": "energy_change",
    "cumulative_distance_m": "distance_sum",
    "cumulative_effort": "effort_sum",
    "cumulative_mechanical_work": "mechanical_work_sum",
    "mouth_contact_bouts": "mouth_contact_bouts",
    "cumulative_ingested_mass": "ingested_mass_sum",
    "cumulative_allocated_mass": "allocation_mass_sum",
    "cumulative_released_mass": "release_mass_sum",
    "cumulative_secreted_mass": "secretion_mass_sum",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(evaluation: Path) -> dict:
    result_path = evaluation / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "completed":
        raise ValueError("this exporter requires a completed evaluator result")
    worlds, residents = result["worlds"], result["residents_per_world"]
    files = sorted((evaluation / "telemetry").glob("step-*.json"))
    if not files:
        raise ValueError("no committed telemetry")
    rows = [json.loads(path.read_text()) for path in files]
    ticks = [row["completed_steps"] for row in rows]
    if ticks != sorted(set(ticks)) or ticks[-1] != result["completed_steps"]:
        raise ValueError("telemetry must reach the completed result monotonically")
    for row in rows:
        if len(row["trajectory"]) != worlds:
            raise ValueError("telemetry world axis differs")
        for cohort in row["trajectory"]:
            if (cohort["completed_batch_ticks"] != row["completed_steps"]
                    or cohort["sampling_dt_seconds"] != 0.05
                    or len(cohort["valid_ticks"]) != residents):
                raise ValueError("trajectory time/resident contract differs")
    curves = []
    occupied = set()
    for life in result["lives"]:
        world, resident = life["world_slot"], life["resident_slot"]
        if ((world, resident) in occupied or not 0 <= world < worlds
                or not 0 <= resident < residents):
            raise ValueError("life slot mapping differs")
        occupied.add((world, resident))
        series = {}
        for public_name, source_name in METRICS.items():
            values = [row["trajectory"][world][source_name][resident] for row in rows]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("nonfinite trajectory")
            if values[-1] != life["trajectory_metrics"][source_name]:
                raise ValueError("final cumulative trajectory differs from result")
            series[public_name] = values
        curves.append({
            "life_id": life["life_id"],
            "candidate_sha256": life["candidate_sha256"],
            "environment_sha256": life["environment_sha256"],
            "trajectory_sha256": life["trajectory_sha256"],
            "ticks": ticks,
            "model_seconds": [tick * 0.05 for tick in ticks],
            "sampling": "cumulative per-resident measurements at committed telemetry boundaries",
            "series": series,
        })
    if len(occupied) != worlds * residents:
        raise ValueError("completed result omits an allocated life")
    return {
        "format": "chreatures-population-cumulative-curves-v1",
        "source_result_sha256": digest(result_path),
        "source_telemetry": [{"name": path.name, "sha256": digest(path)} for path in files],
        "scope": "individual cumulative measurements; adjacent points are not independent trials",
        "trajectories": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = export(args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"curves": len(value["trajectories"]), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
