#!/usr/bin/env python3
"""Stage one authenticated current fly-physics release without publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / "site" / "live"
MOTOR_SCHEMA = ROOT / "research" / "fly_embodiment" / "motor92-channel-schema.json"
FLY_LICENSE = (
    ROOT
    / "native"
    / "fly-body"
    / "assets"
    / "neuromechfly-2.1.0-ca65a510-ypr"
    / "author-source"
    / "FlyGym-LICENSE"
)
SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sha256(value: str, label: str) -> str:
    value = value.lower()
    if not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_revision(value: str) -> str:
    value = value.lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source revision must be a full Git commit SHA")
    return value


def copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"missing regular release input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def hardlink_mesh(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"missing regular mesh asset: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        raise OSError(
            f"mesh staging requires a hardlink on one filesystem: {source} -> {destination}"
        ) from error


def stage(
    *,
    fixture_directory: Path,
    runtime: Path,
    output: Path,
    source_revision: str,
    expected_fixture_sha256: str,
    expected_scene_sha256: str,
    expected_runtime_sha256: str,
    expected_core_wasm_sha256: str,
) -> dict[str, object]:
    source_revision = require_revision(source_revision)
    expected_fixture_sha256 = require_sha256(expected_fixture_sha256, "fixture SHA-256")
    expected_scene_sha256 = require_sha256(expected_scene_sha256, "scene SHA-256")
    expected_runtime_sha256 = require_sha256(expected_runtime_sha256, "runtime SHA-256")
    expected_core_wasm_sha256 = require_sha256(
        expected_core_wasm_sha256, "browser core Wasm SHA-256"
    )
    fixture_directory = fixture_directory.resolve()
    runtime = runtime.resolve()
    output = output.resolve()
    fixture_path = fixture_directory / "world.json"
    scene_path = fixture_directory / "scene.xml"
    if digest(fixture_path) != expected_fixture_sha256:
        raise ValueError("Selected fly fixture SHA-256 differs")
    if digest(scene_path) != expected_scene_sha256:
        raise ValueError("Selected MuJoCo scene SHA-256 differs")
    if digest(runtime) != expected_runtime_sha256:
        raise ValueError("Selected browser-world runtime SHA-256 differs")
    fixture = json.loads(fixture_path.read_text())
    if fixture.get("source_mjcf_sha256") != expected_scene_sha256:
        raise ValueError("Fixture does not bind the selected MuJoCo scene")
    if fixture.get("engine") != "mujoco-3.12.0-neuromechfly-cns-v4":
        raise ValueError("Selected fly physical engine differs")
    for field in (
        "atlas_sha256",
        "body_schema_sha256",
        "sensory_schema_sha256",
        "physical_sensory_schema_sha256",
        "actuator_schema_sha256",
    ):
        require_sha256(str(fixture.get(field)), f"fixture {field}")
    require_sha256(
        str(fixture.get("wing_aerodynamics", {}).get("aerodynamic_schema_sha256")),
        "fixture aerodynamic schema",
    )
    antenna_proxies = fixture.get("antenna_contact_proxies", {})
    if antenna_proxies.get("format") != "chreatures-antenna-contact-proxies-v1":
        raise ValueError("Selected fly fixture lacks the current antenna contact contract")
    antenna_proxy_sha256 = require_sha256(
        str(antenna_proxies.get("artifact_sha256")),
        "fixture antenna contact proxy artifact",
    )
    compiled_antenna_proxies = antenna_proxies.get("compiled")
    if not isinstance(compiled_antenna_proxies, list) or len(compiled_antenna_proxies) != 12:
        raise ValueError("Current B2 fly release requires 12 compiled antenna contact proxies")
    if digest(MOTOR_SCHEMA) != fixture.get("actuator_schema_sha256"):
        raise ValueError("Motor92 schema does not match the selected fly fixture")
    meshes = fixture.get("mesh_assets")
    if not isinstance(meshes, list) or len(meshes) != 39:
        raise ValueError("Current fly release requires exactly 39 authenticated meshes")
    mesh_names = [entry.get("path") for entry in meshes]
    if len(set(mesh_names)) != len(mesh_names) or any(
        not isinstance(name, str) or Path(name).name != name for name in mesh_names
    ):
        raise ValueError("Mesh VFS paths must be unique flat filenames")

    core_wasm = HERE / "pkg" / "chreatures_browser_world_bg.wasm"
    if digest(core_wasm) != expected_core_wasm_sha256:
        raise ValueError("Selected browser core Wasm SHA-256 differs")
    runtime_source = runtime.read_text()
    import_source = 'import("@mujoco/mujoco")'
    if runtime_source.count(import_source) != 1:
        raise ValueError("Browser-world MuJoCo import boundary differs")
    staged_runtime = (
        "// GENERATED from native/browser-world/runtime.mjs.\n"
        + runtime_source.replace(import_source, 'import("./vendor/mujoco/mujoco.js")')
    ).encode()

    # Replace only the physical runtime namespace. Other live-site data remains intact.
    for stale in (
        output / "world-runtime.mjs",
        output / "physics-assets.json",
        output / "fixtures" / "garden.json",
        output / "fixtures" / "garden.xml",
        output / "pkg" / "chreatures_browser_world.js",
        output / "pkg" / "chreatures_browser_world_bg.wasm",
        output / "vendor" / "mujoco" / "mujoco.js",
        output / "vendor" / "mujoco" / "mujoco.wasm",
        output / "vendor" / "mujoco" / "LICENSE",
    ):
        stale.unlink(missing_ok=True)
    shutil.rmtree(output / "fixtures" / "fly-ecology", ignore_errors=True)

    items = {
        "fixtures/fly-ecology/world.json": fixture_path,
        "fixtures/fly-ecology/scene.xml": scene_path,
        "fixtures/fly-ecology/motor92.json": MOTOR_SCHEMA,
        "pkg/chreatures_browser_world.js": HERE / "pkg" / "chreatures_browser_world.js",
        "pkg/chreatures_browser_world_bg.wasm": core_wasm,
        "vendor/mujoco/mujoco.js": HERE / "node_modules" / "@mujoco" / "mujoco" / "mujoco.js",
        "vendor/mujoco/mujoco.wasm": HERE / "node_modules" / "@mujoco" / "mujoco" / "mujoco.wasm",
        "vendor/mujoco/LICENSE": HERE / "licenses" / "MuJoCo-LICENSE",
        "fixtures/fly-ecology/FlyGym-LICENSE": FLY_LICENSE,
    }
    runtime_target = output / "world-runtime.mjs"
    runtime_target.parent.mkdir(parents=True, exist_ok=True)
    runtime_target.write_bytes(staged_runtime)
    assets: dict[str, dict[str, object]] = {
        "world-runtime.mjs": {
            "bytes": len(staged_runtime),
            "sha256": hashlib.sha256(staged_runtime).hexdigest(),
        }
    }
    for target, source in items.items():
        destination = output / target
        copy_regular(source, destination)
        raw = destination.read_bytes()
        assets[target] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    for entry in meshes:
        name = entry["path"]
        source = fixture_directory / name
        if digest(source) != entry.get("sha256"):
            raise ValueError(f"Body mesh hash differs: {name}")
        target = f"fixtures/fly-ecology/{name}"
        destination = output / target
        hardlink_mesh(source, destination)
        assets[target] = {
            "bytes": destination.stat().st_size,
            "sha256": entry["sha256"],
        }

    selection = {
        "sourceRevision": source_revision,
        "engine": fixture.get("engine"),
        "fixtureSha256": expected_fixture_sha256,
        "sceneXmlSha256": expected_scene_sha256,
        "runtimeSourceSha256": expected_runtime_sha256,
        "runtimeStagedSha256": assets["world-runtime.mjs"]["sha256"],
        "coreWasmSha256": expected_core_wasm_sha256,
        "bodySchemaSha256": fixture.get("body_schema_sha256"),
        "cnsSensorySchemaSha256": fixture.get("sensory_schema_sha256"),
        "physicalSensorySchemaSha256": fixture.get("physical_sensory_schema_sha256"),
        "actuatorSchemaSha256": fixture.get("actuator_schema_sha256"),
        "atlasSha256": fixture.get("atlas_sha256"),
        "aerodynamicSchemaSha256": fixture.get("wing_aerodynamics", {}).get(
            "aerodynamic_schema_sha256"
        ),
        "antennaContactProxyArtifactSha256": antenna_proxy_sha256,
        "antennaContactProxyCount": len(compiled_antenna_proxies),
        "compiledCounts": fixture.get("compiled_counts"),
        "residentCount": len(fixture.get("bodies", [])),
        "meshCount": len(meshes),
    }
    manifest = {
        "format": "chreatures-browser-physics-assets-v5",
        "selection": selection,
        "modelBoundary": "separate-authenticated-browser-model-release",
        "upstream": {
            "name": "@mujoco/mujoco",
            "version": "3.12.0",
            "source": "https://github.com/google-deepmind/mujoco/tree/3.12.0/wasm",
            "license": "Apache-2.0",
            "license_path": "vendor/mujoco/LICENSE",
        },
        "bodySource": {
            "name": "NeuroMechFly/FlyGym",
            "revision": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
            "license": "Apache-2.0",
            "license_path": "fixtures/fly-ecology/FlyGym-LICENSE",
        },
        "originalCodeLicense": "AGPL-3.0-or-later",
        "fixtureNotice": (
            "Micro-CT-derived female fly body paired explicitly with MaleCNS; 126 axes, "
            "84 effective position servos, six adhesion and two native oral actuators. "
            "Servo, physiological and aerodynamic dynamics are engineered assumptions, "
            "not identified muscles or demonstrated behavioral competence."
        ),
        "assets": dict(sorted(assets.items())),
    }
    manifest_path = output / "physics-assets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "staged": str(output),
        "files": len(assets),
        "bytes": sum(int(value["bytes"]) for value in assets.values()),
        "manifestSha256": digest(manifest_path),
        "selection": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-directory", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-scene-sha256", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--expected-core-wasm-sha256", required=True)
    arguments = parser.parse_args()
    result = stage(
        fixture_directory=arguments.fixture_directory,
        runtime=arguments.runtime,
        output=arguments.output,
        source_revision=arguments.source_revision,
        expected_fixture_sha256=arguments.expected_fixture_sha256,
        expected_scene_sha256=arguments.expected_scene_sha256,
        expected_runtime_sha256=arguments.expected_runtime_sha256,
        expected_core_wasm_sha256=arguments.expected_core_wasm_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
