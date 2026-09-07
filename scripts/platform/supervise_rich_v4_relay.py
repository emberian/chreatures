#!/usr/bin/env python3
"""Supervise one existing rich-v4 collector and relay its completed corpus once.

The supervisor never starts, resumes, or terminates a collector.  It records a
small status document once per interval and fails closed before any relay if the
collector identity, manifest, or destination prefix is not exactly expected.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
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


def process_state(pid: int) -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,etimes=,stat=,rss=,comm="],
        text=True,
        capture_output=True,
        check=False,
    )
    fields = result.stdout.split()
    if result.returncode or len(fields) < 5:
        return {"pid": pid, "present": False}
    return {
        "pid": int(fields[0]), "present": True, "elapsed_seconds": int(fields[1]),
        "state": fields[2], "rss_kib": int(fields[3]), "command": " ".join(fields[4:]),
    }


def load_relay(path: Path):
    spec = importlib.util.spec_from_file_location("rich_v4_relay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen relay module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-pid", required=True, type=int)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--relay-python", required=True, type=Path)
    parser.add_argument("--relay-script", required=True, type=Path)
    parser.add_argument("--relay-script-sha256", required=True)
    parser.add_argument("--destination-host", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval_seconds < 10:
        raise SystemExit("interval must be at least 10 seconds")
    root = args.collection_root.resolve()
    state_root = args.state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path = state_root / "supervisor.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another collection supervisor already holds the lock")
        if sha256(args.relay_script) != args.relay_script_sha256:
            raise SystemExit("frozen relay script hash differs")
        intent = {
            "format": "chreatures-rich-v4-relay-supervisor-v1",
            "collector_pid": args.collector_pid,
            "collection_root": str(root),
            "expected_collection_identity": args.expected_identity,
            "relay_script": str(args.relay_script.resolve()),
            "relay_script_sha256": args.relay_script_sha256,
            "destination_host": args.destination_host,
            "destination_root": args.destination_root,
            "receipt": str(args.receipt.resolve()),
        }
        intent_path = state_root / "intent.json"
        if intent_path.exists():
            if json.loads(intent_path.read_text()) != intent:
                raise SystemExit("existing supervisor intent differs")
        else:
            atomic_json(intent_path, intent)
        relay = load_relay(args.relay_script)
        while True:
            manifest_path = root / "manifest.json"
            status: dict[str, Any] = {
                "observed_at_unix": time.time(), "collector": process_state(args.collector_pid),
                "manifest_present": manifest_path.is_file(),
            }
            if (root / "progress.json").is_file():
                try:
                    progress = json.loads((root / "progress.json").read_text())
                    status["progress"] = {
                        "sequence": progress.get("sequence"), "completed": progress.get("completed"),
                        "packets": len(progress.get("packets", [])), "checkpoints": len(progress.get("checkpoints", [])),
                    }
                    if progress.get("collection_identity_sha256") != args.expected_identity:
                        raise RuntimeError("progress collection identity differs")
                except Exception as error:
                    status["error"] = str(error)
                    atomic_json(state_root / "status.json", status)
                    raise
            atomic_json(state_root / "status.json", status)
            if not manifest_path.is_file():
                if not status["collector"]["present"]:
                    raise SystemExit("collector exited before manifest publication")
                time.sleep(args.interval_seconds)
                continue
            manifest, files = relay.validate_manifest(manifest_path)
            relay.validate_source(root, files)
            relay.validate_identity_file(root, manifest)
            if manifest.get("collection_identity_sha256") != args.expected_identity:
                raise SystemExit("completed manifest collection identity differs")
            relay_intent = state_root / "relay-start.json"
            if relay_intent.exists() or args.receipt.exists():
                raise SystemExit("relay was previously attempted; refusing duplicate")
            existing_relay = subprocess.run(
                ["pgrep", "-f", str(args.relay_script.resolve())],
                text=True, capture_output=True, check=False,
            )
            if existing_relay.returncode == 0 and existing_relay.stdout.strip():
                raise SystemExit("an existing relay process is present; refusing duplicate")
            remote_check = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", args.destination_host,
                 f"test ! -e {subprocess.list2cmdline([args.destination_root])}"],
                check=False,
            )
            if remote_check.returncode != 0:
                raise SystemExit("destination prefix exists or is not inspectable")
            argv = [str(args.relay_python), str(args.relay_script), str(root),
                    "--destination-host", args.destination_host, "--destination-root", args.destination_root,
                    "--receipt", str(args.receipt)]
            atomic_json(relay_intent, {"argv": argv, "started_at_unix": time.time(), "manifest_sha256": sha256(manifest_path)})
            result = subprocess.run(argv, text=True, capture_output=True, check=False)
            atomic_json(state_root / "relay-result.json", {
                "finished_at_unix": time.time(), "returncode": result.returncode,
                "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:],
            })
            if result.returncode:
                raise SystemExit(f"relay failed with {result.returncode}")
            atomic_json(state_root / "status.json", {"observed_at_unix": time.time(), "relay": "completed", "manifest_sha256": sha256(manifest_path)})
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
