#!/usr/bin/env python3
"""Build the deliberately public, static GitHub Pages artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
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


def build(revision: str | None = None, built_at: str | None = None) -> Path:
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
    args = parser.parse_args()
    output = build(args.revision, args.built_at)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
