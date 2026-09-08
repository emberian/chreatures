#!/usr/bin/env python3
"""Create or fetch the single authenticated input bundle for a Pages build."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.request


FORMAT = "chreatures-pages-release-selection-v1"
RUNTIME_FORMAT = "chreatures-joined-resident-runtime-v1"
AUTHORIZED_RELEASE = "https://github.com/emberian/chreatures/releases/download/"
RUNTIME_FILES = ("pkg/resident_runtime.js", "pkg/resident_runtime_bg.wasm")
SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def require_sha256(value: str, label: str) -> str:
    if not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def checked_manifest(path: Path, expected_sha256: str, format_name: str) -> dict:
    if path.is_symlink() or not path.is_file() or digest(path) != expected_sha256:
        raise ValueError(f"Authenticated manifest differs: {path}")
    value = json.loads(path.read_text())
    if value.get("format") != format_name:
        raise ValueError(f"Manifest format differs: {path}")
    return value


def checked_file(path: Path, expected: dict, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Missing regular {label}: {path}")
    if path.stat().st_size != expected.get("bytes") or digest(path) != expected.get("sha256"):
        raise ValueError(f"Authenticated {label} differs: {path.name}")


def tar_info(name: str, size: int = 0, directory: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name + ("/" if directory and not name.endswith("/") else ""))
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = 0 if directory else size
    info.mode = 0o755 if directory else 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def write_archive(output: Path, root_name: str, files: dict[str, Path | bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + f".tmp-{os.getpid()}")
    partial.unlink(missing_ok=True)
    directories = {root_name}
    for name in files:
        path = PurePosixPath(root_name) / name
        directories.update(str(parent) for parent in path.parents if str(parent) != ".")
    try:
        with partial.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
                        archive.addfile(tar_info(directory, directory=True))
                    for name, source in sorted(files.items()):
                        target = f"{root_name}/{name}"
                        if isinstance(source, bytes):
                            import io

                            archive.addfile(tar_info(target, len(source)), io.BytesIO(source))
                        else:
                            if source.is_symlink() or not source.is_file():
                                raise FileNotFoundError(f"Missing regular release input: {source}")
                            with source.open("rb") as stream:
                                archive.addfile(tar_info(target, source.stat().st_size), stream)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def create(arguments: argparse.Namespace) -> None:
    source_revision = arguments.source_revision.lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a full lowercase Git commit SHA")
    model_sha = require_sha256(arguments.expected_model_release_sha256, "model release")
    source_runtime_sha = require_sha256(
        arguments.expected_source_runtime_manifest_sha256, "source runtime manifest"
    )
    physics_sha = require_sha256(
        arguments.expected_physics_manifest_sha256, "physical release manifest"
    )
    if arguments.training_status != "initialized-untrained":
        raise ValueError("The current public selection must remain initialized-untrained")
    root_name = arguments.bundle_root
    if PurePosixPath(root_name).name != root_name or not re.fullmatch(r"[a-z0-9-]+", root_name):
        raise ValueError("bundle root must be one lowercase flat directory name")
    if Path(arguments.archive_name).name != arguments.archive_name:
        raise ValueError("archive name must be flat")

    model = arguments.model_directory.resolve()
    release = checked_manifest(model / "release.json", model_sha, "chreatures-browser-release-v1")
    model_files = release.get("files")
    if not isinstance(model_files, dict) or not model_files:
        raise ValueError("Model release has no authenticated files")
    for name, expected in model_files.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Invalid model file name: {name}")
        checked_file(model / name, expected, "model file")
    cns = json.loads((model / "cns-manifest.json").read_text())
    resident = json.loads((model / "resident-manifest.json").read_text())
    if (
        cns.get("trainingStatus") != arguments.training_status
        or resident.get("trainingStatus") != arguments.training_status
    ):
        raise ValueError("Model training status differs")

    runtime = arguments.runtime_directory.resolve()
    source_runtime = checked_manifest(
        runtime / "runtime-manifest.json", source_runtime_sha, "chreatures-live-runtime-v1"
    )
    runtime_files: dict[str, dict[str, object]] = {}
    for name in RUNTIME_FILES:
        expected = source_runtime.get("files", {}).get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"Joined runtime lacks {name}")
        normalized = {"bytes": expected.get("byteLength"), "sha256": expected.get("sha256")}
        checked_file(runtime / name, normalized, "joined resident runtime file")
        if expected.get("url") != name:
            raise ValueError(f"Joined runtime URL differs: {name}")
        runtime_files[name] = {
            "byteLength": normalized["bytes"],
            "sha256": normalized["sha256"],
            "url": name,
        }
    runtime_manifest = {
        "format": RUNTIME_FORMAT,
        "sourceManifestSha256": source_runtime_sha,
        "sourceRevision": source_runtime.get("sourceRevision"),
        "files": runtime_files,
    }
    runtime_manifest_bytes = canonical_json(runtime_manifest)
    runtime_sha = hashlib.sha256(runtime_manifest_bytes).hexdigest()

    physics = arguments.physics_directory.resolve()
    physical = checked_manifest(
        physics / "physics-assets.json", physics_sha, "chreatures-browser-physics-assets-v5"
    )
    physical_files = physical.get("assets")
    if not isinstance(physical_files, dict) or not physical_files:
        raise ValueError("Physical release has no authenticated files")
    for name, expected in physical_files.items():
        checked_file(physics / name, expected, "physical release file")

    files: dict[str, Path | bytes] = {
        "model/release.json": model / "release.json",
        "runtime/runtime-manifest.json": runtime_manifest_bytes,
        "physical/physics-assets.json": physics / "physics-assets.json",
    }
    files.update({f"model/{name}": model / name for name in model_files})
    files.update({f"runtime/{name}": runtime / name for name in RUNTIME_FILES})
    files.update({f"physical/{name}": physics / name for name in physical_files})
    write_archive(arguments.archive_output, root_name, files)
    archive_sha = digest(arguments.archive_output)
    archive_bytes = arguments.archive_output.stat().st_size
    release_url = f"{AUTHORIZED_RELEASE}{arguments.release_tag}/{arguments.archive_name}"
    selection = {
        "format": FORMAT,
        "sourceRevision": source_revision,
        "trainingStatus": arguments.training_status,
        "competenceStatus": "initialized-untrained-no-competence-claim",
        "archive": {
            "url": release_url,
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "root": root_name,
        },
        "layout": {"model": "model", "runtime": "runtime", "physical": "physical"},
        "components": {
            "modelReleaseSha256": model_sha,
            "modelReleaseSourceRevision": release.get("sourceRevision"),
            "modelComponentSourceRevision": cns.get("sourceRevision"),
            "serviceArtifactSha256": release.get("serviceArtifactSha256"),
            "residentArtifactSha256": release.get("residentArtifactSha256"),
            "sourceRuntimeManifestSha256": source_runtime_sha,
            "runtimeManifestSha256": runtime_sha,
            "residentRuntimeWasmSha256": runtime_files["pkg/resident_runtime_bg.wasm"]["sha256"],
            "physicsManifestSha256": physics_sha,
            "fixtureSha256": physical.get("selection", {}).get("fixtureSha256"),
            "sceneXmlSha256": physical.get("selection", {}).get("sceneXmlSha256"),
            "runtimeSourceSha256": physical.get("selection", {}).get("runtimeSourceSha256"),
            "browserCoreWasmSha256": physical.get("selection", {}).get("coreWasmSha256"),
            "antennaContactProxyArtifactSha256": physical.get("selection", {}).get(
                "antennaContactProxyArtifactSha256"
            ),
            "antennaContactProxyCount": physical.get("selection", {}).get(
                "antennaContactProxyCount"
            ),
        },
    }
    arguments.selection_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.selection_output.write_bytes(canonical_json(selection))
    print(
        json.dumps(
            {
                "archive": str(arguments.archive_output),
                "archiveBytes": archive_bytes,
                "archiveSha256": archive_sha,
                "runtimeManifestSha256": runtime_sha,
                "selection": str(arguments.selection_output),
            },
            sort_keys=True,
        )
    )


def safe_extract(archive_path: Path, destination: Path, root_name: str) -> None:
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.parts[0] != root_name
                    or not (member.isdir() or member.isfile())
                ):
                    raise ValueError(f"Unsafe release archive entry: {member.name}")
            archive.extractall(temporary)
        extracted = temporary / root_name
        if not extracted.is_dir():
            raise ValueError("Release archive root differs")
        shutil.rmtree(destination, ignore_errors=True)
        extracted.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def fetch(arguments: argparse.Namespace) -> None:
    selection = checked_manifest(arguments.selection, digest(arguments.selection), FORMAT)
    archive = selection.get("archive", {})
    url = archive.get("url")
    if not isinstance(url, str) or not url.startswith(AUTHORIZED_RELEASE):
        raise ValueError("Release bundle URL is not the authorized project")
    expected_sha = require_sha256(str(archive.get("sha256")), "release archive")
    expected_bytes = archive.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError("Release archive byte count differs")
    if arguments.archive is None:
        download = arguments.output.with_name(Path(url).name + f".tmp-{os.getpid()}")
        download.parent.mkdir(parents=True, exist_ok=True)
        download.unlink(missing_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "chreatures-pages/1"})
        try:
            with urllib.request.urlopen(request, timeout=120) as source, download.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            archive_path = download
            if archive_path.stat().st_size != expected_bytes or digest(archive_path) != expected_sha:
                raise ValueError("Downloaded release archive identity differs")
            safe_extract(archive_path, arguments.output, archive["root"])
        finally:
            download.unlink(missing_ok=True)
    else:
        archive_path = arguments.archive.resolve()
        if archive_path.stat().st_size != expected_bytes or digest(archive_path) != expected_sha:
            raise ValueError("Local release archive identity differs")
        safe_extract(archive_path, arguments.output, archive["root"])

    components = selection["components"]
    layout = selection["layout"]
    model = arguments.output / layout["model"]
    runtime = arguments.output / layout["runtime"]
    physical = arguments.output / layout["physical"]
    checks = (
        (model / "release.json", components["modelReleaseSha256"]),
        (runtime / "runtime-manifest.json", components["runtimeManifestSha256"]),
        (physical / "physics-assets.json", components["physicsManifestSha256"]),
    )
    if any(digest(path) != expected for path, expected in checks):
        raise ValueError("Extracted release component identity differs")
    outputs = {
        "model_directory": str(model),
        "runtime_directory": str(runtime),
        "physics_directory": str(physical),
        "model_release_sha256": components["modelReleaseSha256"],
        "runtime_manifest_sha256": components["runtimeManifestSha256"],
        "physics_manifest_sha256": components["physicsManifestSha256"],
        "training_status": selection["trainingStatus"],
    }
    if arguments.github_output is not None:
        with arguments.github_output.open("a") as output:
            for name, value in outputs.items():
                output.write(f"{name}={value}\n")
    print(json.dumps(outputs, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--model-directory", type=Path, required=True)
    create_parser.add_argument("--runtime-directory", type=Path, required=True)
    create_parser.add_argument("--physics-directory", type=Path, required=True)
    create_parser.add_argument("--source-revision", required=True)
    create_parser.add_argument("--training-status", required=True)
    create_parser.add_argument("--expected-model-release-sha256", required=True)
    create_parser.add_argument("--expected-source-runtime-manifest-sha256", required=True)
    create_parser.add_argument("--expected-physics-manifest-sha256", required=True)
    create_parser.add_argument("--release-tag", required=True)
    create_parser.add_argument("--archive-name", required=True)
    create_parser.add_argument("--bundle-root", required=True)
    create_parser.add_argument("--archive-output", type=Path, required=True)
    create_parser.add_argument("--selection-output", type=Path, required=True)
    create_parser.set_defaults(function=create)
    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("--selection", type=Path, required=True)
    fetch_parser.add_argument("--output", type=Path, required=True)
    fetch_parser.add_argument("--archive", type=Path)
    fetch_parser.add_argument("--github-output", type=Path)
    fetch_parser.set_defaults(function=fetch)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
