#!/usr/bin/env python3
"""Stage one authenticated unified native fly-world release without publishing it."""
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
DEFAULT_OUT = ROOT / "site/live"
MOTOR_SCHEMA = ROOT / "research/fly_embodiment/motor92-channel-schema.json"
FLY_LICENSE = ROOT / "native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/author-source/FlyGym-LICENSE"
MUJOCO_LICENSE = HERE / "licenses/MuJoCo-LICENSE"
SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def require_sha256(value: str, label: str) -> str:
    value = value.lower()
    if not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_revision(value: str) -> str:
    value = value.lower()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
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
    except OSError:
        shutil.copy2(source, destination)


def stage(*, fixture_directory: Path, runtime: Path, module_directory: Path,
          output: Path, source_revision: str, expected_fixture_sha256: str,
          expected_scene_sha256: str, expected_runtime_sha256: str,
          expected_module_mjs_sha256: str, expected_module_wasm_sha256: str) -> dict[str, object]:
    source_revision = require_revision(source_revision)
    expected_fixture_sha256 = require_sha256(expected_fixture_sha256, "fixture SHA-256")
    expected_scene_sha256 = require_sha256(expected_scene_sha256, "scene SHA-256")
    expected_runtime_sha256 = require_sha256(expected_runtime_sha256, "runtime SHA-256")
    expected_module_mjs_sha256 = require_sha256(expected_module_mjs_sha256, "module JS SHA-256")
    expected_module_wasm_sha256 = require_sha256(expected_module_wasm_sha256, "module Wasm SHA-256")
    fixture_directory, runtime = fixture_directory.resolve(), runtime.resolve()
    module_directory, output = module_directory.resolve(), output.resolve()
    fixture_path, scene_path = fixture_directory / "world.json", fixture_directory / "scene.xml"
    module_mjs = module_directory / "chreatures-fly-world.mjs"
    module_wasm = module_directory / "chreatures-fly-world.wasm"
    module_receipt = module_directory / "build-receipt.json"
    checks = ((fixture_path, expected_fixture_sha256, "fixture"),
              (scene_path, expected_scene_sha256, "scene"),
              (runtime, expected_runtime_sha256, "unified host"),
              (module_mjs, expected_module_mjs_sha256, "unified module JS"),
              (module_wasm, expected_module_wasm_sha256, "unified module Wasm"))
    for path, expected, label in checks:
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Selected {label} differs")
    if module_receipt.is_symlink() or not module_receipt.is_file():
        raise FileNotFoundError("missing regular unified module build receipt")
    build = json.loads(module_receipt.read_text())
    if (build.get("format") != "chreatures-unified-fly-world-wasm-build-v1"
            or build.get("mjs_sha256") != expected_module_mjs_sha256
            or build.get("wasm_sha256") != expected_module_wasm_sha256):
        raise ValueError("Unified module build receipt differs")
    fixture = json.loads(fixture_path.read_text())
    if fixture.get("source_mjcf_sha256") != expected_scene_sha256:
        raise ValueError("Fixture does not bind the selected MuJoCo scene")
    for field in ("atlas_sha256", "body_schema_sha256", "sensory_schema_sha256",
                  "physical_sensory_schema_sha256", "actuator_schema_sha256"):
        require_sha256(str(fixture.get(field)), f"fixture {field}")
    aerodynamic_sha256 = require_sha256(
        str(fixture.get("wing_aerodynamics", {}).get("aerodynamic_schema_sha256")),
        "fixture aerodynamic schema",
    )
    antenna_proxies = fixture.get("antenna_contact_proxies", {})
    if antenna_proxies.get("format") != "chreatures-antenna-contact-proxies-v1":
        raise ValueError("Selected fixture lacks the current antenna contact contract")
    antenna_proxy_sha256 = require_sha256(
        str(antenna_proxies.get("artifact_sha256")),
        "fixture antenna contact proxy artifact",
    )
    compiled_antenna_proxies = antenna_proxies.get("compiled")
    expected_proxy_count = 6 * len(fixture.get("bodies", ()))
    if (not isinstance(compiled_antenna_proxies, list)
            or len(compiled_antenna_proxies) != expected_proxy_count):
        raise ValueError("Compiled antenna contact proxy count differs")
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

    # Replace the complete physical namespace. The unified module embeds MuJoCo
    # and WorldCore; old browser-world and vendor runtimes must not survive.
    for stale in ("world-runtime.mjs", "physics-assets.json",
                  "pkg/chreatures_browser_world.js", "pkg/chreatures_browser_world_bg.wasm",
                  "vendor/mujoco/mujoco.js", "vendor/mujoco/mujoco.wasm", "vendor/mujoco/LICENSE"):
        (output / stale).unlink(missing_ok=True)
    shutil.rmtree(output / "fixtures/fly-ecology", ignore_errors=True)
    shutil.rmtree(output / "vendor/mujoco", ignore_errors=True)

    items = {
        "world-runtime.mjs": runtime,
        "pkg/chreatures-fly-world.mjs": module_mjs,
        "pkg/chreatures-fly-world.wasm": module_wasm,
        "pkg/chreatures-fly-world-build.json": module_receipt,
        "fixtures/fly-ecology/world.json": fixture_path,
        "fixtures/fly-ecology/scene.xml": scene_path,
        "fixtures/fly-ecology/motor92.json": MOTOR_SCHEMA,
        "fixtures/fly-ecology/FlyGym-LICENSE": FLY_LICENSE,
        "licenses/MuJoCo-LICENSE": MUJOCO_LICENSE,
    }
    assets: dict[str, dict[str, object]] = {}
    for target, source in items.items():
        destination = output / target
        copy_regular(source, destination)
        assets[target] = {"bytes": destination.stat().st_size, "sha256": digest(destination)}
    for entry in meshes:
        name = entry["path"]
        source = fixture_directory / name
        if digest(source) != entry.get("sha256"):
            raise ValueError(f"Body mesh hash differs: {name}")
        target, destination = f"fixtures/fly-ecology/{name}", output / "fixtures/fly-ecology" / name
        hardlink_mesh(source, destination)
        assets[target] = {"bytes": destination.stat().st_size, "sha256": entry["sha256"]}

    selection = {
        "sourceRevision": source_revision,
        "engine": "unified-native-fly-world-mujoco-3.12.0-v1",
        "fixtureEngine": fixture.get("engine"),
        "fixtureSha256": expected_fixture_sha256,
        "sceneXmlSha256": expected_scene_sha256,
        "runtimeSourceSha256": expected_runtime_sha256,
        "unifiedModuleMjsSha256": expected_module_mjs_sha256,
        "unifiedModuleWasmSha256": expected_module_wasm_sha256,
        "bodySchemaSha256": fixture.get("body_schema_sha256"),
        "cnsSensorySchemaSha256": fixture.get("sensory_schema_sha256"),
        "physicalSensorySchemaSha256": fixture.get("physical_sensory_schema_sha256"),
        "actuatorSchemaSha256": fixture.get("actuator_schema_sha256"),
        "atlasSha256": fixture.get("atlas_sha256"),
        "aerodynamicSchemaSha256": aerodynamic_sha256,
        "antennaContactProxyArtifactSha256": antenna_proxy_sha256,
        "antennaContactProxyCount": len(compiled_antenna_proxies),
        "compiledCounts": fixture.get("compiled_counts"),
        "residentCount": len(fixture.get("bodies", [])),
        "meshCount": len(meshes),
    }
    manifest = {
        "format": "chreatures-browser-physics-assets-v6",
        "selection": selection,
        "modelBoundary": "separate-authenticated-browser-model-release",
        "runtimeArchitecture": "single-unified-rust-mujoco-wasm",
        "upstream": {"name": "MuJoCo", "version": "3.12.0", "license": "Apache-2.0",
                     "license_path": "licenses/MuJoCo-LICENSE"},
        "bodySource": {"name": "NeuroMechFly/FlyGym",
                       "revision": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
                       "license": "Apache-2.0",
                       "license_path": "fixtures/fly-ecology/FlyGym-LICENSE"},
        "originalCodeLicense": "AGPL-3.0-or-later",
        "assets": dict(sorted(assets.items())),
    }
    manifest_path = output / "physics-assets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"staged": str(output), "files": len(assets),
            "bytes": sum(int(v["bytes"]) for v in assets.values()),
            "manifestSha256": digest(manifest_path), "selection": selection}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-directory", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=ROOT / "native/fly-world/runtime.mjs")
    parser.add_argument("--module-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-scene-sha256", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--expected-module-mjs-sha256", required=True)
    parser.add_argument("--expected-module-wasm-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(stage(fixture_directory=args.fixture_directory, runtime=args.runtime,
        module_directory=args.module_directory, output=args.output,
        source_revision=args.source_revision, expected_fixture_sha256=args.expected_fixture_sha256,
        expected_scene_sha256=args.expected_scene_sha256, expected_runtime_sha256=args.expected_runtime_sha256,
        expected_module_mjs_sha256=args.expected_module_mjs_sha256,
        expected_module_wasm_sha256=args.expected_module_wasm_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
