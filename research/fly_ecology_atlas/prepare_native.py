#!/usr/bin/env python3
"""Freeze the current native-host ecology atlas design and four physical layouts."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

FEATURES = (
    "branch_angle_rad",
    "lateral_probability",
    "phototropism",
    "contact_avoidance",
)
RANGES = ((0.25, 1.15), (0.08, 0.75), (0.1, 1.6), (0.2, 2.0))
FORMAT = "chreatures-native-fly-ecology-atlas-v2"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def design(seed):
    rng = np.random.default_rng(seed)
    best = None
    score = -1.0
    for _ in range(256):
        unit = np.column_stack(
            [(rng.permutation(24) + rng.random(24)) / 24 for _ in FEATURES]
        )
        distance = ((unit[:, None] - unit[None, :]) ** 2).sum(-1)
        np.fill_diagonal(distance, np.inf)
        if distance.min() > score:
            best, score = unit, distance.min()
    return [
        dict(
            genotype_id=f"genotype-{i:02d}",
            split="fit" if i < 18 else "genotype-holdout",
            genes={
                name: float(lo + row[j] * (hi - lo))
                for j, (name, (lo, hi)) in enumerate(zip(FEATURES, RANGES))
            },
            coordinates={
                name: float(2 * row[j] - 1) for j, name in enumerate(FEATURES)
            },
        )
        for i, row in enumerate(best)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--seconds", type=float, default=32.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.fixture.read_text())
    source_dir = args.fixture.parent
    source_xml = source_dir / base["scene_xml"]
    if (
        sha(source_xml) != base["scene_xml_sha256"]
        or base["engine"] != "mujoco-3.12.0-neuromechfly-cns-v4"
    ):
        raise ValueError("current native fixture identity differs")
    layouts = [
        ("layout-0", None),
        ("layout-1", ("ecology/curved-bark-03", [8, -6, 1.2], [0.8, 2.4, 0.7])),
        ("layout-2", ("ecology/curved-bark-03", [-7, 7, 1.5], [0.7, 1.8, 0.9])),
        (
            "confirmation-layout",
            ("ecology/curved-bark-03", [2, 0, 1.1], [1.2, 2.8, 0.8]),
        ),
    ]
    frozen = []
    for index, (layout_id, obstacle) in enumerate(layouts):
        folder = args.output / "layouts" / layout_id
        folder.mkdir(parents=True)
        tree = ET.parse(source_xml)
        if obstacle:
            name, pos, size = obstacle
            geom = next(node for node in tree.iter("geom") if node.get("name") == name)
            geom.set("pos", " ".join(map(str, pos)))
            geom.set("size", " ".join(map(str, size)))
        xml_path = folder / "scene.xml"
        tree.write(xml_path, encoding="unicode")
        fixture = copy.deepcopy(base)
        fixture["fixture_id"] = f"{base['fixture_id']}-native-atlas-{layout_id}"
        fixture["scene_xml"] = "scene.xml"
        fixture["source_mjcf_sha256"] = fixture["scene_xml_sha256"] = sha(xml_path)
        fixture["ecology"]["seed"] = args.seed + 1000 + index
        for asset in fixture["mesh_assets"]:
            source = (source_dir / asset["path"]).resolve()
            destination = folder / asset["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(source, destination)
        fixture_path = folder / "world.json"
        write(fixture_path, fixture)
        frozen.append(
            dict(
                layout_id=layout_id,
                split="confirmation" if index == 3 else "fit-layout",
                fixture=str(fixture_path.resolve()),
                fixture_sha256=sha(fixture_path),
                scene_xml_sha256=sha(xml_path),
                world_seed=args.seed + 1000 + index,
            )
        )
    runner = Path(__file__).with_name("native_run.py")
    plan = dict(
        format=FORMAT,
        status="planned-not-executed",
        seed=args.seed,
        control_dt=0.01,
        ticks=round(args.seconds / 0.01),
        seconds=args.seconds,
        features=list(FEATURES),
        ranges=dict(zip(FEATURES, RANGES)),
        genotypes=design(args.seed),
        layouts=frozen,
        native_binary=str(args.native_binary.resolve()),
        native_binary_sha256=sha(args.native_binary),
        runner=str(runner.resolve()),
        runner_sha256=sha(runner),
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        claim_limit="Synthetic colony response and selection evidence under actual native physical geometry; neutral diagnostic flies, no fly skill or CNS competence claim.",
    )
    write(args.output / "plan.json", plan)
    print(
        json.dumps(
            {
                "plan": str(args.output / "plan.json"),
                "sha256": sha(args.output / "plan.json"),
                "runs": 24 * 3,
                "ticks_per_run": plan["ticks"],
            }
        )
    )


if __name__ == "__main__":
    main()
