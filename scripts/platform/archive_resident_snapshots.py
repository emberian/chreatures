#!/usr/bin/env python3
"""Archive old complete brain snapshots for the current population-v5 deployment.

This helper is deliberately scoped to one deployment and one hbox archive.  It
publishes a remote object and a durable local index before removing a local
snapshot.  Fetch restores the exact indexed bytes without deleting the archive.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import struct


DEPLOYMENT_NAME = "population-v5-trained-ee18250"
DEPLOYMENT = Path(
    "/Users/ember/paperbin/chreatures/deployments/population-v5-trained-ee18250"
)
SNAPSHOTS = DEPLOYMENT / "run/brain/snapshots"
WORLD_CHECKPOINT = DEPLOYMENT / "run/world.json"
INDEX = DEPLOYMENT / "run/brain/archive-index.json"
LOCK = DEPLOYMENT / "run/brain/archive.lock"
REMOTE_HOST = "hbox"
REMOTE_TARGET = "hbox@192.168.50.39"
REMOTE_ROOT = Path(
    "/tank/chreatures/resident-archives/population-v5-trained-ee18250"
)
KEEP_RECENT = 8
MIN_AGE_SECONDS = 180.0
MAX_FILES_PER_RUN = 12
FORMAT = "chreatures-resident-snapshot-archive-v1"
EXPECTED_GRAPH_SHA256 = (
    "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625"
)
EXPECTED_NEURONS = 165_122
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,239}\.npz\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def empty_index() -> dict[str, object]:
    value = {
        "format": FORMAT,
        "version": 1,
        "local_snapshot_directory": str(SNAPSHOTS),
        "remote_host": REMOTE_HOST,
        "remote_root": str(REMOTE_ROOT),
        "entries": {},
    }
    value["deployment"] = DEPLOYMENT_NAME
    identity = checkpoint_identity()
    if identity is not None:
        value["world_id"] = identity["world_id"]
    return value


def load_index() -> dict[str, object]:
    if not INDEX.exists():
        return empty_index()
    value = json.loads(INDEX.read_text(encoding="utf-8"))
    expected = empty_index()
    fields = [
        "format",
        "version",
        "local_snapshot_directory",
        "remote_host",
        "remote_root",
    ]
    fields.extend(("deployment", "world_id"))
    for field in fields:
        if value.get(field) != expected.get(field):
            raise RuntimeError(f"archive index {field} differs")
    if not isinstance(value.get("entries"), dict):
        raise RuntimeError("archive index entries differ")
    return value


def checkpoint_identity() -> dict[str, str] | None:
    try:
        value = json.loads(WORLD_CHECKPOINT.read_text(encoding="utf-8"))
        state = value["state"]
        name = state["neural_snapshot"]["name"]
        world_id = state["id"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(name, str) or not isinstance(world_id, str) or not world_id:
        return None
    filename = name if name.endswith(".npz") else f"{name}.npz"
    if not _NAME.fullmatch(filename) or not filename.startswith(f"world-{world_id}-"):
        return None
    return {"filename": filename, "world_id": world_id}


def checkpoint_reference() -> str | None:
    identity = checkpoint_identity()
    return None if identity is None else identity["filename"]


def complete_npz(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 16:
            return False
        with path.open("rb") as stream:
            if stream.read(8) != b"MBST1\0\0\0":
                return False
            length = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < length <= 2_000_000:
                return False
            metadata = json.loads(stream.read(length))
        capacity = metadata.get("capacity")
        if (
            metadata.get("version") != 6
            or metadata.get("state_layout") != "neuron-major-float4-tiles-v1"
            or metadata.get("graph_sha256") != EXPECTED_GRAPH_SHA256
            or isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or not 1 <= capacity <= 32
        ):
            return False
        payload = path.stat().st_size - 16 - length
        bytes_per_neuron = 3 * ((capacity + 3) // 4) * 16
        return payload == EXPECTED_NEURONS * bytes_per_neuron
    except (OSError, ValueError, TypeError, json.JSONDecodeError, struct.error):
        return False


def eligible_snapshots(now: float) -> list[Path]:
    if not SNAPSHOTS.is_dir():
        return []
    paths = [
        path
        for path in SNAPSHOTS.iterdir()
        if path.is_file() and _NAME.fullmatch(path.name)
    ]
    newest = {
        path.name
        for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns)[-KEEP_RECENT:]
    }
    referenced = checkpoint_reference()
    eligible = []
    for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns):
        if (
            path.name in newest
            or path.name == referenced
            or now - path.stat().st_mtime < MIN_AGE_SECONDS
            or not complete_npz(path)
        ):
            continue
        eligible.append(path)
    return eligible


def ssh(arguments: str, *, stdin=None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            REMOTE_TARGET,
            arguments,
        ],
        stdin=stdin,
        text=stdin is None,
        capture_output=stdin is None,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() if isinstance(result.stderr, str) else ""
        raise RuntimeError(f"ssh command failed ({result.returncode}): {detail}")
    return result


def publish_remote(path: Path, digest: str, size: int) -> str:
    remote_directory = REMOTE_ROOT / "snapshots"
    destination = remote_directory / path.name
    temporary = remote_directory / f".{path.name}.tmp-{os.getpid()}"
    mkdir = f"mkdir -p -- {shlex.quote(str(remote_directory))}"
    ssh(mkdir)
    command = f"cat > {shlex.quote(str(temporary))}"
    with path.open("rb") as stream:
        subprocess.run(
            [
                "/usr/bin/ssh",
                "-o",
                "BatchMode=yes",
                REMOTE_TARGET,
                command,
            ],
            stdin=stream,
            check=True,
        )
    verify = "\n".join(
        (
            "set -eu",
            f"tmp={shlex.quote(str(temporary))}",
            f"dst={shlex.quote(str(destination))}",
            f"expected_sha={shlex.quote(digest)}",
            f"expected_size={size}",
            'test "$(stat -c %s "$tmp")" = "$expected_size"',
            'test "$(sha256sum "$tmp" | cut -d\" \" -f1)" = "$expected_sha"',
            'if test -e "$dst"; then',
            '  test "$(stat -c %s "$dst")" = "$expected_size"',
            '  test "$(sha256sum "$dst" | cut -d\" \" -f1)" = "$expected_sha"',
            '  rm -f -- "$tmp"',
            "else",
            '  mv -- "$tmp" "$dst"',
            "fi",
        )
    )
    ssh(verify)
    fsync_script = (
        "import os,sys; "
        "f=os.open(sys.argv[1],os.O_RDONLY); os.fsync(f); os.close(f); "
        "d=os.open(sys.argv[2],os.O_RDONLY); os.fsync(d); os.close(d)"
    )
    ssh(
        f"python3 -c {shlex.quote(fsync_script)} "
        f"{shlex.quote(str(destination))} {shlex.quote(str(remote_directory))}"
    )
    return str(destination)


def archive_once() -> dict[str, object]:
    started = time.time()
    index = load_index()
    entries = index["entries"]
    assert isinstance(entries, dict)
    moved = []
    skipped = []
    candidates = eligible_snapshots(started)
    for path in candidates[:MAX_FILES_PER_RUN]:
        before = path.stat()
        digest = sha256(path)
        remote = publish_remote(path, digest, before.st_size)
        current = path.stat()
        if (
            current.st_dev != before.st_dev
            or current.st_ino != before.st_ino
            or current.st_size != before.st_size
            or current.st_mtime_ns != before.st_mtime_ns
            or sha256(path) != digest
        ):
            skipped.append({"name": path.name, "reason": "local file changed"})
            continue
        entry = {
            "name": path.name,
            "bytes": before.st_size,
            "sha256": digest,
            "mtime_ns": before.st_mtime_ns,
            "remote_path": remote,
            "remote_durability": "file-and-directory-fsync-v1",
            "archived_at_unix": time.time(),
        }
        existing = entries.get(path.name)
        if existing is not None and existing != entry:
            stable = dict(entry)
            stable.pop("archived_at_unix")
            old = dict(existing)
            old.pop("archived_at_unix", None)
            if old != stable:
                skipped.append({"name": path.name, "reason": "index identity differs"})
                continue
            entry = existing
        entries[path.name] = entry
        index["updated_at_unix"] = time.time()
        atomic_json(INDEX, index)
        current_reference = checkpoint_reference()
        if current_reference is None:
            skipped.append(
                {"name": path.name, "reason": "world checkpoint is unreadable"}
            )
            continue
        if path.name == current_reference:
            skipped.append(
                {"name": path.name, "reason": "world checkpoint now references snapshot"}
            )
            continue
        path.unlink()
        directory = os.open(SNAPSHOTS, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        moved.append(entry)
    return {
        "format": "chreatures-resident-snapshot-archive-run-v1",
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "moved": moved,
        "skipped": skipped,
        "deferred": [path.name for path in candidates[MAX_FILES_PER_RUN:]],
        "bytes_moved": sum(int(item["bytes"]) for item in moved),
        "remaining_local": len(list(SNAPSHOTS.glob("*.npz"))),
        "checkpoint_reference": checkpoint_reference(),
    }


def fetch(name: str, expected_sha256: str | None) -> dict[str, object]:
    if not _NAME.fullmatch(name):
        raise RuntimeError("invalid snapshot name")
    index = load_index()
    entry = index["entries"].get(name)
    if not isinstance(entry, dict):
        raise RuntimeError("snapshot is absent from the authenticated archive index")
    if expected_sha256 is not None and entry.get("sha256") != expected_sha256:
        raise RuntimeError("requested snapshot hash differs from the archive index")
    destination = SNAPSHOTS / name
    if destination.exists():
        if destination.stat().st_size != entry["bytes"] or sha256(destination) != entry["sha256"]:
            raise RuntimeError("existing local snapshot differs")
        return {"status": "already-present", "path": str(destination), **entry}
    temporary = destination.with_name(f".{name}.fetch-{os.getpid()}")
    command = f"cat -- {shlex.quote(str(entry['remote_path']))}"
    with temporary.open("wb") as stream:
        subprocess.run(
            [
                "/usr/bin/ssh",
                "-o",
                "BatchMode=yes",
                REMOTE_TARGET,
                command,
            ],
            stdout=stream,
            check=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    if temporary.stat().st_size != entry["bytes"] or sha256(temporary) != entry["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("fetched snapshot differs from the archive index")
    os.replace(temporary, destination)
    directory = os.open(SNAPSHOTS, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"status": "restored", "path": str(destination), **entry}


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("archive")
    commands.add_parser("status")
    restore = commands.add_parser("fetch")
    restore.add_argument("name")
    restore.add_argument("--expected-sha256")
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "already-running"}, sort_keys=True))
            return 0
        if args.command == "archive":
            result = archive_once()
        elif args.command == "fetch":
            result = fetch(args.name, args.expected_sha256)
        else:
            result = {
                "format": FORMAT,
                "eligible": [path.name for path in eligible_snapshots(time.time())],
                "checkpoint_reference": checkpoint_reference(),
                "index": load_index(),
            }
        print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
