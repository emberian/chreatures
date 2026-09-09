#!/usr/bin/env python3
"""Build four authenticated synthetic habitats for the committed-response atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "chreatures-fly-ecology-layout-set-v1"
LAYOUTS = (
    ("fit-open-overhead", "fit-layout", 0, [0.0, 0.0, 1.0], 0.80, 0.20),
    ("fit-east-oblique", "fit-layout", 1, [0.40, 0.10, 0.91], 0.65, 0.35),
    ("fit-northwest-oblique", "fit-layout", 2, [-0.30, 0.45, 0.84], 0.90, 0.10),
    ("holdout-southeast-mixed", "confirmation", 3, [0.55, -0.45, 0.70], 0.55, 0.45),
)
BODY_IDENTITIES = (
    "fixture_id",
    "source_body_mjcf_sha256",
    "body_schema_sha256",
    "morphology_sha256",
    "morphology_asset_set_sha256",
    "sensory_schema_sha256",
    "physical_actuator_schema_sha256",
    "actuator_schema_sha256",
    "wing_aerodynamics",
)


def sha(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def normalized(direction: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in direction))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("illumination direction differs")
    return [value / norm for value in direction]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--residents", type=int, default=2)
    args = parser.parse_args()
    if args.residents != 2:
        raise ValueError("committed-response atlas is frozen to B2")
    args.output.mkdir(parents=True, exist_ok=False)
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    planner_manifest = ROOT / "native/fly-habitat/Cargo.toml"
    composer = ROOT / "native/fly-body/tools/compose_ecology_scene.py"
    exporter = ROOT / "native/browser-world/export_fixture.py"
    layout_records = []
    body_identity = None
    for layout_id, split, offset, sky, sky_intensity, screen_intensity in LAYOUTS:
        folder = args.output / layout_id
        folder.mkdir()
        plan_path = folder / "habitat-plan.json"
        scene_path = folder / "scene.xml"
        physics_path = folder / "physics.json"
        world_path = folder / "world.json"
        layout_seed = args.seed + offset
        run(
            [
                "cargo",
                "run",
                "--release",
                "--manifest-path",
                str(planner_manifest),
                "--",
                "--output",
                str(plan_path),
                "--seed",
                str(layout_seed),
                "--residents",
                "2",
                "--width-mm",
                "80",
                "--depth-mm",
                "60",
                "--height-mm",
                "30",
            ]
        )
        run(
            [
                sys.executable,
                str(composer),
                "--output",
                str(scene_path),
                "--manifest",
                str(physics_path),
                "--residents",
                "2",
                "--habitat-plan",
                str(plan_path),
                "--startup-steps",
                "100",
            ]
        )
        run(
            [
                sys.executable,
                str(exporter),
                "--physics",
                str(physics_path),
                "--habitat-plan",
                str(plan_path),
                "--output",
                str(world_path),
            ]
        )
        world = json.loads(world_path.read_text())
        world["illumination"] = {
            "sky_direction_world": normalized(sky),
            "sky_intensity": sky_intensity,
            "screen_intensity": screen_intensity,
            "photon_energy_per_second": 0.3,
        }
        world["experiment_layout"] = {
            "format": "chreatures-fly-ecology-synthetic-layout-v1",
            "layout_id": layout_id,
            "split": split,
            "layout_seed": layout_seed,
            "authorship": "synthetic deterministic habitat arrangement; not observed natural diversity",
            "illumination_role": "host physics/ecology configuration and observer provenance; never a controller input",
        }
        world_path.unlink()
        write(world_path, world)
        current_identity = {name: world[name] for name in BODY_IDENTITIES}
        if body_identity is None:
            body_identity = current_identity
        elif current_identity != body_identity:
            raise ValueError("layout changed the imported fly body identity")
        if world["scene_xml_sha256"] != sha(scene_path):
            raise ValueError("composed scene identity differs")
        plan = json.loads(plan_path.read_text())
        if plan["seed"] != layout_seed or plan["parameters"]["residents"] != 2:
            raise ValueError("habitat planner identity differs")
        manifest = {
            "format": "chreatures-fly-ecology-layout-artifact-v1",
            "layout_id": layout_id,
            "split": split,
            "source_revision": source_revision,
            "builders": {
                "fly_habitat_manifest_sha256": sha(planner_manifest),
                "composer_sha256": sha(composer),
                "exporter_sha256": sha(exporter),
            },
            "files": {
                "habitat_plan": {"path": "habitat-plan.json", "sha256": sha(plan_path)},
                "scene": {"path": "scene.xml", "sha256": sha(scene_path)},
                "physics": {"path": "physics.json", "sha256": sha(physics_path)},
                "world": {"path": "world.json", "sha256": sha(world_path)},
            },
            "illumination": world["illumination"],
            "compiled_counts": world["compiled_counts"],
            "body_identity": current_identity,
            "layout_physical_sensory_schema_sha256": world[
                "physical_sensory_schema_sha256"
            ],
            "habitat_validation": plan["validation"],
        }
        manifest_path = folder / "artifact-manifest.json"
        write(manifest_path, manifest)
        layout_records.append(
            {
                "layout_id": layout_id,
                "split": split,
                "fixture": str(world_path.resolve()),
                "fixture_sha256": sha(world_path),
                "artifact_manifest": str(manifest_path.resolve()),
                "artifact_manifest_sha256": sha(manifest_path),
                "description": f"new synthetic habitat seed {layout_seed} with pinned {layout_id} illumination",
            }
        )
    layout_manifest = {
        "format": FORMAT,
        "status": "built-authenticated-not-executed",
        "source_revision": source_revision,
        "generator_sha256": sha(__file__),
        "same_imported_fly_body_all_layouts": True,
        "body_identity": body_identity,
        "layouts": layout_records,
        "claim_limit": "Newly authored deterministic synthetic arrangements for intervention, not measurements of natural habitat diversity.",
    }
    path = args.output / "layouts.json"
    write(path, layout_manifest)
    print(json.dumps({"manifest": str(path), "sha256": sha(path), "layouts": 4}))


if __name__ == "__main__":
    main()
