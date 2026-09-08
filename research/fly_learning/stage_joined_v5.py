#!/usr/bin/env python3
"""Stage an isolated, authenticated V5 LiveEngine tree with unified fly physics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_physical_stager():
    path = ROOT / "native/browser-world/stage_site.py"
    spec = importlib.util.spec_from_file_location("chreatures_physical_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load unified physical stager")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch", type=int, choices=(2, 4), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--unified-module-directory", type=Path, required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-scene-sha256", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--expected-module-mjs-sha256", required=True)
    parser.add_argument("--expected-module-wasm-sha256", required=True)
    parser.add_argument("--cargo-target-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    if len(arguments.revision) != 40 or any(c not in "0123456789abcdef" for c in arguments.revision):
        raise ValueError("revision must be a lowercase full Git identity")

    output, live = arguments.output.resolve(), arguments.output.resolve() / "live"
    shutil.copytree(ROOT / "site/live", live, ignore=shutil.ignore_patterns("model", "__pycache__"))

    model = arguments.model.resolve()
    shutil.copytree(model, live / "model", copy_function=shutil.copy2)
    resident = json.loads((live / "model/resident-manifest.json").read_text())
    cns = json.loads((live / "model/cns-manifest.json").read_text())
    observer = json.loads((live / "model/observer-manifest.json").read_text())
    release = json.loads((live / "model/release.json").read_text())
    if cns.get("format") != "chreatures-cns-webgpu-v5" or cns.get("version") != 5:
        raise ValueError("joined stage requires the current V5 CNS pack")
    if (resident.get("format") != "chreatures-browser-resident-v1"
            or resident.get("cnsServiceArtifactSha256") != cns.get("serviceArtifactSha256")
            or observer.get("graph") != cns.get("identity", {}).get("graph")):
        raise ValueError("joined model identities differ")
    if resident.get("config", {}).get("batch") != arguments.batch:
        raise ValueError("resident pack batch differs from the physical cohort")
    if not resident.get("config", {}).get("private_learning_version"):
        raise ValueError("resident pack lacks a private learning identity")
    if (release.get("format") != "chreatures-browser-release-v1"
            or release.get("serviceArtifactSha256") != cns["serviceArtifactSha256"]
            or release.get("residentArtifactSha256") != resident["artifactSha256"]):
        raise ValueError("model release identities differ")
    model_revision = release.get("sourceRevision", "")
    if len(model_revision) != 40 or any(c not in "0123456789abcdef" for c in model_revision):
        raise ValueError("model release source revision differs")
    for name, identity in release.get("files", {}).items():
        path = live / "model" / name
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"model release file differs: {name}")

    fixture_path = arguments.fixture.resolve()
    fixture = json.loads(fixture_path.read_text())
    if len(fixture.get("bodies", ())) != arguments.batch:
        raise ValueError("fixture body count differs from resident batch")
    stage_physics = load_physical_stager()
    physical = stage_physics(
        fixture_directory=fixture_path.parent,
        runtime=ROOT / "native/fly-world/runtime.mjs",
        module_directory=arguments.unified_module_directory,
        output=live,
        source_revision=arguments.revision,
        expected_fixture_sha256=arguments.expected_fixture_sha256,
        expected_scene_sha256=arguments.expected_scene_sha256,
        expected_runtime_sha256=arguments.expected_runtime_sha256,
        expected_module_mjs_sha256=arguments.expected_module_mjs_sha256,
        expected_module_wasm_sha256=arguments.expected_module_wasm_sha256,
    )
    subprocess.run(["python3", str(ROOT / "native/resident-runtime/build_wasm.py"),
                    "--out-dir", str(live / "pkg"), "--target-dir",
                    str(arguments.cargo_target_dir.resolve())], cwd=ROOT, check=True)

    physics_manifest = json.loads((live / "physics-assets.json").read_text())
    runtime_names = {
        "engine.js", "worker.js", "assets.js", "cns-webgpu.js", "physical-world.js",
        "world-runtime.mjs", "pkg/resident_runtime.js", "pkg/resident_runtime_bg.wasm",
        "shaders/afferent.wgsl", "shaders/dynamics.wgsl", "shaders/readout.wgsl",
        "shaders/motor.wgsl", "shaders/observe.wgsl", "shaders/plasticity.wgsl",
        *physics_manifest["assets"],
    }
    runtime_files = {}
    for name in sorted(runtime_names):
        path = live / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing V5 runtime file: {name}")
        runtime_files[name] = {"url": name, "byteLength": path.stat().st_size, "sha256": sha256(path)}
    runtime_manifest = {"format": "chreatures-live-runtime-v1",
                        "sourceRevision": arguments.revision, "files": runtime_files}
    (live / "runtime-manifest.json").write_text(json.dumps(runtime_manifest, sort_keys=True, separators=(",", ":")) + "\n")
    files = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(output.rglob("*")) if path.is_file()}
    receipt = {
        "format": "chreatures-v5-joined-stage-v1",
        "source_revision": arguments.revision,
        "model_source_revision": model_revision,
        "batch": arguments.batch,
        "service_sha256": cns["serviceArtifactSha256"],
        "adapter_sha256": cns["identity"]["artifact"],
        "resident_artifact_sha256": resident["artifactSha256"],
        "physics_manifest_sha256": sha256(live / "physics-assets.json"),
        "runtime_manifest_sha256": sha256(live / "runtime-manifest.json"),
        "unified_module_wasm_sha256": arguments.expected_module_wasm_sha256,
        "files": files,
    }
    (output / "stage-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
