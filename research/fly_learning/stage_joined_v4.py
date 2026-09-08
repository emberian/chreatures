#!/usr/bin/env python3
"""Stage an isolated, authenticated V4 headless LiveEngine tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"staging input must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--cargo-target-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    if len(arguments.revision) != 40 or any(
        character not in "0123456789abcdef" for character in arguments.revision
    ):
        raise ValueError("revision must be a lowercase full Git identity")
    if arguments.batch != 2:
        raise ValueError("the current joined V4 evidence contract is exactly B2")

    output = arguments.output.resolve()
    live = output / "live"
    shutil.copytree(
        ROOT / "site/live",
        live,
        ignore=shutil.ignore_patterns("model", "__pycache__"),
    )

    model = arguments.model.resolve()
    shutil.copytree(model, live / "model")
    resident_manifest_path = live / "model/resident-manifest.json"
    resident = json.loads(resident_manifest_path.read_text())
    cns = json.loads((live / "model/cns-manifest.json").read_text())
    observer = json.loads((live / "model/observer-manifest.json").read_text())
    if (
        resident.get("format") != "chreatures-browser-resident-v1"
        or resident.get("cnsServiceArtifactSha256")
        != cns.get("serviceArtifactSha256")
        or observer.get("graph") != cns.get("identity", {}).get("graph")
    ):
        raise ValueError("joined model identities differ")
    resident["config"]["batch"] = arguments.batch
    resident["config"]["private_learning_version"] = "context-consequence-v1"
    resident_manifest_path.write_text(
        json.dumps(resident, sort_keys=True, separators=(",", ":")) + "\n"
    )
    model_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted((live / "model").iterdir())
        if path.is_file() and path.name != "release.json"
    }
    release = {
        "format": "chreatures-browser-release-v1",
        "sourceRevision": arguments.revision,
        "serviceArtifactSha256": cns["serviceArtifactSha256"],
        "residentArtifactSha256": resident["artifactSha256"],
        "files": model_files,
        "totalBytes": sum(item["bytes"] for item in model_files.values()),
    }
    (live / "model/release.json").write_text(
        json.dumps(release, sort_keys=True, separators=(",", ":")) + "\n"
    )

    browser = ROOT / "native/browser-world"
    fixture_source = browser / "fixtures/fly-ecology"
    fixture = json.loads((fixture_source / "world.json").read_text())
    physical = {
        "fixtures/fly-ecology/world.json": fixture_source / "world.json",
        "fixtures/fly-ecology/scene.xml": fixture_source / "scene.xml",
        "fixtures/fly-ecology/motor92.json": ROOT
        / "research/fly_embodiment/motor92-channel-schema.json",
        "pkg/chreatures_browser_world.js": browser
        / "pkg/chreatures_browser_world.js",
        "pkg/chreatures_browser_world_bg.wasm": browser
        / "pkg/chreatures_browser_world_bg.wasm",
        "vendor/mujoco/mujoco.js": browser
        / "node_modules/@mujoco/mujoco/mujoco.js",
        "vendor/mujoco/mujoco.wasm": browser
        / "node_modules/@mujoco/mujoco/mujoco.wasm",
        "vendor/mujoco/LICENSE": browser / "licenses/MuJoCo-LICENSE",
        "fixtures/fly-ecology/FlyGym-LICENSE": ROOT
        / "native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/author-source/FlyGym-LICENSE",
    }
    for mesh in fixture["mesh_assets"]:
        name = str(mesh["path"])
        if Path(name).name != name:
            raise ValueError("mesh path must be a flat authenticated filename")
        source = fixture_source / name
        if sha256(source) != mesh["sha256"]:
            raise ValueError(f"mesh identity differs: {name}")
        physical[f"fixtures/fly-ecology/{name}"] = source
    for name, source in physical.items():
        copy(source, live / name)

    runtime_source = (browser / "runtime.mjs").read_text()
    (live / "world-runtime.mjs").write_text(
        "// GENERATED from native/browser-world/runtime.mjs.\n"
        + runtime_source.replace(
            'import("@mujoco/mujoco")', 'import("./vendor/mujoco/mujoco.js")'
        )
    )
    subprocess.run(
        [
            "python3",
            str(ROOT / "native/resident-runtime/build_wasm.py"),
            "--out-dir",
            str(live / "pkg"),
            "--target-dir",
            str(arguments.cargo_target_dir.resolve()),
        ],
        cwd=ROOT,
        check=True,
    )

    physical_assets = {
        name: {"bytes": (live / name).stat().st_size, "sha256": sha256(live / name)}
        for name in sorted((*physical, "world-runtime.mjs"))
    }
    (live / "physics-assets.json").write_text(
        json.dumps(
            {
                "format": "chreatures-browser-physics-assets-v4",
                "engine": fixture["engine"],
                "assets": physical_assets,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    runtime_names = {
        "engine.js",
        "worker.js",
        "assets.js",
        "cns-webgpu.js",
        "world-runtime.mjs",
        "pkg/resident_runtime.js",
        "pkg/resident_runtime_bg.wasm",
        "pkg/chreatures_browser_world.js",
        "pkg/chreatures_browser_world_bg.wasm",
        "vendor/mujoco/mujoco.js",
        "vendor/mujoco/mujoco.wasm",
        "shaders/afferent.wgsl",
        "shaders/dynamics.wgsl",
        "shaders/readout.wgsl",
        "shaders/motor.wgsl",
        "shaders/observe.wgsl",
        *physical,
    }
    runtime_files = {
        name: {
            "url": name,
            "byteLength": (live / name).stat().st_size,
            "sha256": sha256(live / name),
        }
        for name in sorted(runtime_names)
    }
    runtime_manifest = {
        "format": "chreatures-live-runtime-v1",
        "sourceRevision": arguments.revision,
        "files": runtime_files,
    }
    (live / "runtime-manifest.json").write_text(
        json.dumps(runtime_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    files = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    receipt = {
        "format": "chreatures-v4-joined-stage-v1",
        "source_revision": arguments.revision,
        "batch": arguments.batch,
        "service_sha256": cns["serviceArtifactSha256"],
        "adapter_sha256": cns["identity"]["artifact"],
        "resident_artifact_sha256": resident["artifactSha256"],
        "runtime_manifest_sha256": sha256(live / "runtime-manifest.json"),
        "files": files,
    }
    (output / "stage-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: receipt[key] for key in receipt if key != "files"}))


if __name__ == "__main__":
    main()
