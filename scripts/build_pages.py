#!/usr/bin/env python3
"""Build the deliberately public, static GitHub Pages artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUTPUT = ROOT / "dist" / "site"

PUBLIC_DOC_ASSETS = (
    "articulated-garden.png",
    "hollow-garden.png",
    "learning-garden.png",
    "terrarium-garden.png",
)

PUBLIC_THREE_ASSETS = (
    "three.module.min.js",
    "three.core.min.js",
    "OrbitControls.js",
    "LICENSE",
    "VERSION",
)


class _HtmlReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src", "poster"} and value is not None:
                self.references.append(value)


def _copy_tree(source: Path, destination: Path) -> None:
    """Copy regular files while refusing links out of the publication root."""
    for path in sorted(source.rglob("*")):
        if any(part in {"node_modules", "__pycache__", ".git"} for part in path.relative_to(source).parts):
            continue
        if path.is_symlink():
            raise ValueError(f"public site must not contain symlinks: {path.relative_to(ROOT)}")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        else:
            raise ValueError(f"unsupported public site entry: {path.relative_to(ROOT)}")


def _copy_allowlist(source: Path, names: tuple[str, ...], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing regular public asset: {path.relative_to(ROOT)}")
        shutil.copy2(path, destination / name)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_assets(root: Path, assets: dict[str, dict[str, object]], label: str) -> None:
    if not isinstance(assets, dict) or not assets:
        raise ValueError(f"{label} has no authenticated files")
    for name, expected in assets.items():
        if not isinstance(name, str) or not isinstance(expected, dict):
            raise ValueError(f"{label} asset entry differs")
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid {label} asset path: {name}")
        raw = path.read_bytes()
        if len(raw) != expected.get("bytes") or hashlib.sha256(raw).hexdigest() != expected.get("sha256"):
            raise ValueError(f"{label} asset differs: {name}")


def _overlay_physics(
    physics_directory: Path, expected_manifest_sha256: str
) -> dict[str, object]:
    """Install one already authenticated physical release into the clean build."""
    source = physics_directory.resolve()
    manifest_path = source / "physics-assets.json"
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
        or _sha256(manifest_path) != expected_manifest_sha256
    ):
        raise ValueError("Selected physical release manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "chreatures-browser-physics-assets-v5":
        raise ValueError("Stage the selected V4 anatomical fly release before building Pages")
    assets = manifest.get("assets")
    _validated_assets(source, assets, "physical release")
    live = OUTPUT / "live"
    # Never retain a second, generic-body production mechanism in the artifact.
    for stale in (live / "fixtures" / "garden.json", live / "fixtures" / "garden.xml"):
        stale.unlink(missing_ok=True)
    shutil.rmtree(live / "fixtures" / "fly-ecology", ignore_errors=True)
    for name in assets:
        target = live / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(manifest_path, live / "physics-assets.json")
    return {
        "manifestSha256": expected_manifest_sha256,
        **manifest["selection"],
    }


JOINED_RUNTIME_FILES = (
    "pkg/resident_runtime.js",
    "pkg/resident_runtime_bg.wasm",
)


def _overlay_joined_runtime(
    runtime_directory: Path, expected_manifest_sha256: str
) -> dict[str, object]:
    """Install the resident Wasm package exercised by the selected joined run.

    Current browser/UI sources come from the selected source revision. The
    generated resident package is selected separately because it is rebuilt by
    the joined release process rather than kept current under site/live/pkg.
    """
    source = runtime_directory.resolve()
    manifest_path = source / "runtime-manifest.json"
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
        or _sha256(manifest_path) != expected_manifest_sha256
    ):
        raise ValueError("Selected joined runtime manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "chreatures-joined-resident-runtime-v1":
        raise ValueError("Selected joined runtime contract differs")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Selected joined runtime has no authenticated files")
    selected: dict[str, dict[str, object]] = {}
    live = OUTPUT / "live"
    for name in JOINED_RUNTIME_FILES:
        expected = files.get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"Selected joined runtime lacks {name}")
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing regular joined runtime asset: {name}")
        raw = path.read_bytes()
        if (
            len(raw) != expected.get("byteLength")
            or hashlib.sha256(raw).hexdigest() != expected.get("sha256")
            or expected.get("url") != name
        ):
            raise ValueError(f"Selected joined runtime asset differs: {name}")
        target = live / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        shutil.copy2(path, target)
        selected[name] = {
            "bytes": len(raw),
            "sha256": expected["sha256"],
        }
    return {
        "manifestSha256": expected_manifest_sha256,
        "sourceRevision": manifest.get("sourceRevision"),
        "fileSetSha256": hashlib.sha256(
            json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": selected,
    }


def _revision(argument: str | None) -> str:
    revision = argument or os.environ.get("GITHUB_SHA")
    if revision is None:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if len(revision) != 40 or any(char not in "0123456789abcdefABCDEF" for char in revision):
        raise ValueError("revision must be a full 40-character Git commit SHA")
    return revision.lower()


def _built_at(argument: str | None) -> str:
    if argument is not None:
        try:
            value = datetime.fromisoformat(argument.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("built-at must be an ISO-8601 timestamp") from exc
        if value.tzinfo is None:
            raise ValueError("built-at must include a timezone")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_reference(owner: Path, reference: str) -> None:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return
    if parsed.path.startswith("/"):
        raise ValueError(
            f"root-relative public URL breaks the /chreatures/ project path: "
            f"{owner.relative_to(OUTPUT)} -> {reference}"
        )
    target = (owner.parent / unquote(parsed.path)).resolve()
    try:
        target.relative_to(OUTPUT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"public URL escapes the Pages artifact: {owner.relative_to(OUTPUT)} -> {reference}"
        ) from exc
    if target.is_dir():
        target /= "index.html"
    if not target.is_file():
        raise FileNotFoundError(
            f"broken public URL: {owner.relative_to(OUTPUT)} -> {reference}"
        )


def _validate_public_urls() -> None:
    css_url = re.compile(r"url\(\s*['\"]?([^'\")]+)")
    js_fetch = re.compile(r"\bfetch\s*\(\s*['\"]([^'\"]+)")
    js_import = re.compile(r"\bfrom\s*['\"]([^'\"]+)")
    for path in sorted(OUTPUT.rglob("*")):
        if path.suffix == ".html":
            parser = _HtmlReferences()
            parser.feed(path.read_text(encoding="utf-8"))
            for reference in parser.references:
                _validate_reference(path, reference)
        elif path.suffix == ".css":
            for reference in css_url.findall(path.read_text(encoding="utf-8")):
                _validate_reference(path, reference)
        elif path.suffix == ".js":
            source = path.read_text(encoding="utf-8")
            for reference in js_fetch.findall(source):
                _validate_reference(path, reference)
            for reference in js_import.findall(source):
                # A bare name is resolved by an import map, not as a file URL.
                if reference.startswith((".", "/")):
                    _validate_reference(path, reference)


def _runtime_identity(
    revision: str,
    joined_runtime_identity: dict[str, object],
    physical_identity: dict[str, object],
    model_identity: dict[str, object],
) -> dict[str, object]:
    """Bind saved lives to executable bytes as well as immutable model tensors."""
    live = OUTPUT / "live"
    if not (live / "engine.js").exists():
        raise FileNotFoundError("live engine entry is required")
    names = ["engine.js", "worker.js", "assets.js", "cns-webgpu.js", "world-runtime.mjs",
             "pkg/resident_runtime.js", "pkg/resident_runtime_bg.wasm",
             "pkg/chreatures_browser_world.js", "pkg/chreatures_browser_world_bg.wasm",
             "vendor/mujoco/mujoco.js", "vendor/mujoco/mujoco.wasm",
             "shaders/afferent.wgsl", "shaders/dynamics.wgsl", "shaders/readout.wgsl", "shaders/motor.wgsl", "shaders/observe.wgsl"]
    physics = json.loads((live / "physics-assets.json").read_text())
    if physics["format"] != "chreatures-browser-physics-assets-v5":
        raise ValueError("Selected physical release contract differs")
    _validated_assets(live, physics["assets"], "staged physical release")
    names.extend(physics["assets"])
    names.append("model-selection.json")
    if (live / "release-selection.json").is_file():
        names.append("release-selection.json")
    manifest = {
        # The runtime loader accepts this current envelope and ignores added
        # authenticated selection fields when booting the exact file table.
        "format": "chreatures-live-runtime-v1",
        "sourceRevision": revision,
        "joinedResidentRuntime": joined_runtime_identity,
        "physical": physical_identity,
        "model": model_identity,
        "files": {},
    }
    for name in sorted(set(names)):
        content = (live / name).read_bytes()
        manifest["files"][name] = {"url": name, "byteLength": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()}
    (live / "runtime-manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def _copy_model(
    model_directory: Path,
    expected_training_status: str,
    expected_release_sha256: str,
    physical_identity: dict[str, object],
) -> dict[str, object]:
    """Materialize exactly one selected model release and expose its boundary."""
    if expected_training_status not in {"initialized-untrained", "trained"}:
        raise ValueError("Unknown expected model training status")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_release_sha256):
        raise ValueError("Expected model release SHA-256 must be lowercase hexadecimal")
    source = model_directory.resolve()
    release_path = source / "release.json"
    receipt = json.loads(release_path.read_text())
    if receipt.get("format") != "chreatures-browser-release-v1":
        raise ValueError("Unknown local browser model release format")
    release_sha256 = _sha256(release_path)
    files = receipt["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Browser model release has no files")
    destination = OUTPUT / "live" / "model"
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
            raise ValueError("Invalid public model asset name")
        target = destination / name
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            local_asset = source / name
            if local_asset.is_symlink() or not local_asset.is_file():
                raise FileNotFoundError(f"Missing regular model release asset: {name}")
            with local_asset.open("rb") as stream, partial.open("wb") as output:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > expected["bytes"]:
                        raise ValueError(f"Oversized model asset: {name}")
                    digest.update(chunk)
                    output.write(chunk)
            if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Model asset identity differs: {name}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    for required in ("cns-manifest.json", "resident-manifest.json", "observer-manifest.json"):
        if required not in files:
            raise ValueError(f"Browser model release lacks {required}")
    cns = json.loads((destination / "cns-manifest.json").read_text())
    resident = json.loads((destination / "resident-manifest.json").read_text())
    release_source_revision = receipt.get("sourceRevision")
    component_source_revision = cns.get("sourceRevision")
    training_status = cns.get("trainingStatus")
    if (
        cns.get("format") != "chreatures-cns-webgpu-v4"
        or resident.get("format") != "chreatures-browser-resident-v1"
        or not isinstance(release_source_revision, str)
        or not isinstance(component_source_revision, str)
        or resident.get("sourceRevision") != component_source_revision
        or resident.get("trainingStatus") != training_status
        or training_status != expected_training_status
        or cns.get("serviceArtifactSha256") != receipt.get("serviceArtifactSha256")
        or resident.get("artifactSha256") != receipt.get("residentArtifactSha256")
        or resident.get("cnsServiceArtifactSha256") != receipt.get("serviceArtifactSha256")
    ):
        raise ValueError("Selected browser model identity or training status differs")
    cns_identity = cns.get("identity", {})
    counts = cns.get("counts", {})
    if (
        cns_identity.get("morphology") != physical_identity.get("bodySchemaSha256")
        or cns_identity.get("sensorySchema") != physical_identity.get("cnsSensorySchemaSha256")
        or cns_identity.get("actuatorSchema") != physical_identity.get("actuatorSchemaSha256")
        or cns_identity.get("atlas") != physical_identity.get("atlasSha256")
        or resident.get("config", {}).get("batch") != physical_identity.get("residentCount")
        or counts.get("opticValues") != 5313
        or counts.get("bodyChannels") != 807
        or counts.get("motor") != 92
        or cns.get("controlDt") != 0.01
    ):
        raise ValueError("Selected physical release and CNS/resident model boundary differ")
    if not re.fullmatch(r"[0-9a-f]{64}", str(release_sha256)):
        raise ValueError("Browser model release SHA-256 is required")
    if release_sha256 != expected_release_sha256:
        raise ValueError("Selected browser model release SHA-256 differs")
    file_set_sha256 = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    identity = {
        "releaseSha256": release_sha256,
        "releaseSourceRevision": release_source_revision,
        "componentSourceRevision": component_source_revision,
        "trainingStatus": training_status,
        "competenceStatus": (
            "initialized-untrained-no-competence-claim"
            if training_status == "initialized-untrained"
            else "trained-status-only-no-competence-claim"
        ),
        "serviceArtifactSha256": receipt.get("serviceArtifactSha256"),
        "residentArtifactSha256": receipt.get("residentArtifactSha256"),
        "fileSetSha256": file_set_sha256,
        "fileCount": len(files),
    }
    selection = {
        "format": "chreatures-pages-model-selection-v1",
        **identity,
        "files": files,
    }
    (OUTPUT / "live" / "model-selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    # The source tree's former lock must not describe a different public model.
    (OUTPUT / "live" / "model.lock.json").unlink(missing_ok=True)
    return identity



def _version_live_publication() -> dict[str, object]:
    """Publish one complete live bundle under a content-derived URL.

    The stable /live tree also supports headless tooling. Public entry points use
    the versioned copy so a refresh cannot mix cached JS from one engine with
    another engine's Wasm, shaders, fixture or model. No prior bundles are kept.
    """
    live = OUTPUT / "live"
    if not (live / "runtime-manifest.json").exists():
        raise FileNotFoundError("authenticated live runtime manifest is required")
    runtime = json.loads((live / "runtime-manifest.json").read_text())
    inputs = {"runtime": runtime["files"], "files": {}}
    for relative in ("live.js", "live.css", "live/view.js", "live/model/cns-manifest.json",
                     "live/model/resident-manifest.json", "live/model/observer-manifest.json"):
        path = OUTPUT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing live publication input: {relative}")
        inputs["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prefix = "live-" + identity[:24]
    shutil.copytree(live, OUTPUT / prefix)
    entry = (OUTPUT / "live.js").read_text()
    for old, new in (("'./live/view.js'", f"'./{prefix}/view.js'"),
                     ("'./live/worker.js'", f"'./{prefix}/worker.js'")):
        if entry.count(old) != 1:
            raise ValueError(f"live entry reference changed: {old}")
        entry = entry.replace(old, new)
    (OUTPUT / (prefix + ".js")).write_text(entry)
    shutil.copy2(OUTPUT / "live.css", OUTPUT / (prefix + ".css"))
    page = OUTPUT / "live.html"
    html = page.read_text()
    for old, new in (("src=\"live.js\"", f"src=\"{prefix}.js\""),
                     ("href=\"live.css\"", f"href=\"{prefix}.css\"")):
        if html.count(old) != 1:
            raise ValueError(f"live page entry changed: {old}")
        html = html.replace(old, new)
    page.write_text(html)
    publication = {
        "format": "chreatures-live-publication-v2", "identity": identity,
        "basePath": prefix + "/", "entry": prefix + ".js", "style": prefix + ".css",
        "sourceRevision": runtime["sourceRevision"],
        "joinedResidentRuntime": runtime["joinedResidentRuntime"],
        "physical": runtime["physical"], "model": runtime["model"],
    }
    (OUTPUT / "live-publication.json").write_text(
        json.dumps(publication, sort_keys=True) + "\n"
    )
    return publication


def build(
    revision: str | None,
    built_at: str | None,
    model_directory: Path,
    runtime_directory: Path,
    expected_runtime_manifest_sha256: str,
    physics_directory: Path,
    expected_physics_manifest_sha256: str,
    model_training_status: str,
    expected_model_release_sha256: str,
) -> Path:
    if not (SITE / "index.html").is_file():
        raise FileNotFoundError("site/index.html is required")
    if SITE.is_symlink():
        raise ValueError("site must be a repository directory, not a symlink")

    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True)
    _copy_tree(SITE, OUTPUT)
    _copy_allowlist(
        ROOT / "docs" / "assets",
        PUBLIC_DOC_ASSETS,
        OUTPUT / "assets" / "recorded",
    )
    _copy_allowlist(
        ROOT / "web" / "vendor" / "three",
        PUBLIC_THREE_ASSETS,
        OUTPUT / "vendor" / "three",
    )
    selected_revision = _revision(revision)
    joined_runtime_identity = _overlay_joined_runtime(
        runtime_directory, expected_runtime_manifest_sha256
    )
    physical_identity = _overlay_physics(
        physics_directory, expected_physics_manifest_sha256
    )
    model_identity = _copy_model(
        model_directory,
        model_training_status,
        expected_model_release_sha256,
        physical_identity,
    )
    _runtime_identity(
        selected_revision,
        joined_runtime_identity,
        physical_identity,
        model_identity,
    )
    publication = _version_live_publication()
    info = {
        "format": "chreatures-pages-build-v2",
        "revision": selected_revision,
        "built_at": _built_at(built_at),
        "base_path": "/chreatures/",
        "livePublicationIdentity": publication["identity"],
        "joinedResidentRuntime": joined_runtime_identity,
        "physical": physical_identity,
        "model": model_identity,
    }
    (OUTPUT / "build-info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _validate_public_urls()
    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="full source Git commit SHA")
    parser.add_argument("--built-at", help="ISO-8601 build timestamp")
    parser.add_argument("--model-directory", type=Path, required=True, help="Selected local browser model release")
    parser.add_argument("--runtime-directory", type=Path, required=True, help="Live directory containing the resident Wasm used by the selected successful joined run")
    parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    parser.add_argument("--physics-directory", type=Path, required=True, help="Authenticated output from native/browser-world/stage_site.py")
    parser.add_argument("--expected-physics-manifest-sha256", required=True)
    parser.add_argument("--model-training-status", choices=("initialized-untrained", "trained"), default="initialized-untrained")
    parser.add_argument("--expected-model-release-sha256", required=True)
    args = parser.parse_args()
    output = build(
        args.revision,
        args.built_at,
        args.model_directory,
        args.runtime_directory,
        args.expected_runtime_manifest_sha256,
        args.physics_directory,
        args.expected_physics_manifest_sha256,
        args.model_training_status,
        args.expected_model_release_sha256,
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
