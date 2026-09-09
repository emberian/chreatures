#!/usr/bin/env python3
"""Freeze an independent-unit committed-growth atlas without executing worlds."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


FEATURES = (
    "branch_angle_rad",
    "lateral_probability",
    "phototropism",
    "contact_avoidance",
)
RANGES = ((0.25, 1.15), (0.08, 0.75), (0.1, 1.6), (0.2, 2.0))
FORMAT = "chreatures-native-fly-ecology-atlas-v3"


def sha(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def design(seed: int, count: int = 16) -> list[dict]:
    rng = np.random.default_rng(seed)
    best = None
    score = -1.0
    for _ in range(512):
        unit = np.column_stack(
            [(rng.permutation(count) + rng.random(count)) / count for _ in FEATURES]
        )
        distance = ((unit[:, None] - unit[None, :]) ** 2).sum(-1)
        np.fill_diagonal(distance, np.inf)
        if distance.min() > score:
            best, score = unit, float(distance.min())
    return [
        {
            "genotype_id": f"genotype-{index:02d}",
            "split": "fit" if index < 12 else "genotype-holdout",
            "genes": {
                name: float(lo + row[j] * (hi - lo))
                for j, (name, (lo, hi)) in enumerate(zip(FEATURES, RANGES))
            },
            "coordinates": {
                name: float(2 * row[j] - 1) for j, name in enumerate(FEATURES)
            },
        }
        for index, row in enumerate(best)
    ]


def load_layouts(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "chreatures-fly-ecology-layout-set-v1":
        raise ValueError("layout manifest format differs")
    layouts = manifest.get("layouts")
    if not isinstance(layouts, list) or len(layouts) != 4:
        raise ValueError("exactly four physical layouts are required")
    if sum(item.get("split") == "fit-layout" for item in layouts) != 3:
        raise ValueError("exactly three fitting layouts are required")
    if sum(item.get("split") == "confirmation" for item in layouts) != 1:
        raise ValueError("exactly one untouched confirmation layout is required")
    if len({item.get("layout_id") for item in layouts}) != len(layouts):
        raise ValueError("layout identities differ")
    result = []
    for item in layouts:
        fixture_path = Path(item["fixture"])
        if not fixture_path.is_absolute():
            fixture_path = path.parent / fixture_path
        fixture_path = fixture_path.resolve()
        fixture = json.loads(fixture_path.read_text())
        scene = (fixture_path.parent / fixture["scene_xml"]).resolve()
        if sha(scene) != fixture["scene_xml_sha256"]:
            raise ValueError(f"layout scene changed: {item['layout_id']}")
        routes = fixture["ecology"]["routes"]
        regions = fixture["ecology"]["regions"]
        result.append(
            {
                "layout_id": item["layout_id"],
                "split": item["split"],
                "fixture": str(fixture_path),
                "fixture_sha256": sha(fixture_path),
                "scene_xml_sha256": sha(scene),
                "route_geometry_sha256": canonical_sha(
                    {
                        "regions": regions,
                        "routes": routes,
                        "collision_geometry_identity": fixture["source_mjcf_sha256"],
                    }
                ),
                "description": item.get("description", "measured physical layout"),
                "engine": fixture["engine"],
            }
        )
    if len({item["engine"] for item in result}) != 1:
        raise ValueError("layout engine identities differ")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--native-source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--seconds", type=float, default=32.0)
    parser.add_argument("--sample-every-ticks", type=int, default=100)
    args = parser.parse_args()
    if args.seconds <= 0 or args.sample_every_ticks <= 0:
        raise ValueError("campaign duration/cadence outside bounds")
    if len(args.native_source_revision) != 40:
        raise ValueError("native source revision must be a full Git identity")
    subprocess.run(
        ["git", "cat-file", "-e", f"{args.native_source_revision}^{{commit}}"], check=True
    )
    args.output.mkdir(parents=True, exist_ok=False)
    layouts = load_layouts(args.layout_manifest)
    genotypes = design(args.seed)
    fit_layouts = [item for item in layouts if item["split"] == "fit-layout"]
    units = []
    for genotype_index, genotype in enumerate(genotypes):
        for layout_index, layout in enumerate(fit_layouts):
            unit_index = genotype_index * len(fit_layouts) + layout_index
            units.append(
                {
                    "unit_id": f"{genotype['genotype_id']}--{layout['layout_id']}",
                    "genotype": genotype,
                    "layout": layout,
                    "replicate": 0,
                    "ecology_seed": args.seed + 100_000 + 2 * unit_index,
                    "physics_seed": args.seed + 100_001 + 2 * unit_index,
                }
            )
    runner = Path(__file__).with_name("native_run.py")
    ticks = round(args.seconds / 0.01)
    if abs(ticks * 0.01 - args.seconds) > 1e-12:
        raise ValueError("duration is not an exact 100 Hz tick count")
    plan = {
        "format": FORMAT,
        "status": "planned-not-executed-root-pin-required",
        "seed": args.seed,
        "control_dt": 0.01,
        "ticks": ticks,
        "seconds": args.seconds,
        "sample_every_ticks": args.sample_every_ticks,
        "experimental_unit": "one fresh native process/world seed for one genotype-layout cell; two colonies within a world are coupled subsamples and never treated as replicates",
        "features": list(FEATURES),
        "ranges": dict(zip(FEATURES, RANGES)),
        "genotypes": genotypes,
        "layouts": layouts,
        "units": units,
        "native_binary": str(args.native_binary.resolve()),
        "native_binary_sha256": sha(args.native_binary),
        "runner": str(runner.resolve()),
        "runner_sha256": sha(runner),
        "layout_manifest_sha256": sha(args.layout_manifest),
        "native_source_revision": args.native_source_revision,
        "analysis_source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "engine": layouts[0]["engine"],
        "historical_aperture_rows_reused": False,
        "historical_exclusion": "Earlier aperture rows predate committed-growth counters and corrected immediate physical route invalidation. They remain evidence but are not pooled into this joint response fit.",
        "claim_limit": "Actual native MuJoCo colony construction/resource/aperture response under neutral diagnostic MOTOR92. No CNS or fly-skill evaluation; clearance approval is never a committed branch.",
    }
    write(args.output / "plan.json", plan)
    print(
        json.dumps(
            {
                "plan": str(args.output / "plan.json"),
                "sha256": sha(args.output / "plan.json"),
                "fit_units": len(units),
                "ticks_per_unit": ticks,
                "confirmation_layouts_executed": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
