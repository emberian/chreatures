#!/usr/bin/env python3
"""Build the deliberately public, static GitHub Pages artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import hashlib
import http.client
import urllib.request
import urllib.error
import time
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


def _runtime_identity(revision: str) -> None:
    """Bind saved lives to executable bytes as well as immutable model tensors."""
    live = OUTPUT / "live"
    if not (live / "engine.js").exists():
        return
    names = ["engine.js", "worker.js", "assets.js", "cns-webgpu.js", "world-runtime.mjs",
             "pkg/resident_runtime.js", "pkg/resident_runtime_bg.wasm",
             "pkg/chreatures_browser_world.js", "pkg/chreatures_browser_world_bg.wasm",
             "vendor/mujoco/mujoco.js", "vendor/mujoco/mujoco.wasm",
             "shaders/afferent.wgsl", "shaders/dynamics.wgsl", "shaders/readout.wgsl", "shaders/motor.wgsl", "shaders/observe.wgsl"]
    physics = json.loads((live / "physics-assets.json").read_text())
    if physics["format"] != "chreatures-browser-physics-assets-v4":
        raise ValueError("Stage the current anatomical fly runtime before building Pages")
    for name, expected in physics["assets"].items():
        path = (live / name).resolve()
        if not path.is_relative_to(live.resolve()):
            raise ValueError("Invalid physical asset path")
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Staged physical asset differs: {name}")
        names.append(name)
    manifest = {"format": "chreatures-live-runtime-v1", "sourceRevision": revision, "files": {}}
    for name in sorted(set(names)):
        content = (live / name).read_bytes()
        manifest["files"][name] = {"url": name, "byteLength": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()}
    (live / "runtime-manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


def _copy_model(model_directory: Path | None) -> None:
    """Materialize only the pinned public model release, outside source Git."""
    lock_path = SITE / "live" / "model.lock.json"
    if model_directory is not None:
        source = model_directory.resolve()
        receipt = json.loads((source / "release.json").read_text())
        files = receipt["files"]
    elif lock_path.exists():
        receipt = json.loads(lock_path.read_text())
        if receipt.get("format") != "chreatures-pages-model-lock-v1":
            raise ValueError("Unknown browser model lock format")
        base = receipt["baseURL"]
        if not base.startswith("https://github.com/emberian/chreatures/releases/download/") or not base.endswith("/"):
            raise ValueError("Model release must belong to the authorized project")
        source = None
        files = receipt["files"]
    else:
        return
    destination = OUTPUT / "live" / "model"
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
            raise ValueError("Invalid public model asset name")
        target = destination / name
        partial = target.with_suffix(target.suffix + ".part")
        attempts = 5 if source is None else 1
        for attempt in range(attempts):
            digest = hashlib.sha256()
            size = 0
            try:
                if source is None:
                    request = urllib.request.Request(base + name, headers={"User-Agent": "chreatures-pages/1"})
                    stream = urllib.request.urlopen(request, timeout=90)
                else:
                    stream = (source / name).open("rb")
                with stream, partial.open("wb") as output:
                    while chunk := stream.read(1024 * 1024):
                        size += len(chunk)
                        if size > expected["bytes"]:
                            raise ValueError(f"Oversized model asset: {name}")
                        digest.update(chunk)
                        output.write(chunk)
                if source is None and size < expected["bytes"]:
                    raise http.client.IncompleteRead(b"", expected["bytes"] - size)
                if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                    raise ValueError(f"Model asset identity differs: {name}")
                partial.replace(target)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead) as error:
                if (source is not None or attempt + 1 == attempts or
                    isinstance(error, urllib.error.HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}):
                    raise
                print(f"Retrying public model asset {name} ({attempt + 2}/{attempts}) after transport failure")
                time.sleep(min(2 ** attempt, 8))
            finally:
                partial.unlink(missing_ok=True)



def _version_live_publication() -> None:
    """Publish one complete live bundle under a content-derived URL.

    The stable /live tree also supports headless tooling. Public entry points use
    the versioned copy so a refresh cannot mix cached JS from one engine with
    another engine's Wasm, shaders, fixture or model. No prior bundles are kept.
    """
    live = OUTPUT / "live"
    if not (live / "runtime-manifest.json").exists():
        return
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
    (OUTPUT / "live-publication.json").write_text(json.dumps({
        "format": "chreatures-live-publication-v1", "identity": identity,
        "basePath": prefix + "/", "entry": prefix + ".js", "style": prefix + ".css",
        "sourceRevision": runtime["sourceRevision"],
    }, sort_keys=True) + "\n")


def build(revision: str | None = None, built_at: str | None = None, model_directory: Path | None = None) -> Path:
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
    _copy_model(model_directory)
    _runtime_identity(_revision(revision))
    _version_live_publication()
    info = {
        "format": "chreatures-pages-build-v1",
        "revision": _revision(revision),
        "built_at": _built_at(built_at),
        "base_path": "/chreatures/",
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
    parser.add_argument("--model-directory", type=Path, help="Verified local export for headless integration; deployed builds use the pinned release lock")
    args = parser.parse_args()
    output = build(args.revision, args.built_at, args.model_directory)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
