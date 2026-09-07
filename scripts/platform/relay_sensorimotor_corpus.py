#!/usr/bin/env python3
"""Durably relay one sealed rich-v4 sensorimotor corpus over SSH.

The collector publishes ``manifest.json`` only after every shard and coupled
checkpoint is complete.  This helper authenticates that manifest and its
declared files, copies files without transforming them, and publishes the
destination manifest last.  It never opens trajectory arrays or selects data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
from typing import Any, Mapping


FORMAT = "chreatures-sensorimotor-play-rich-v4"
VERSION = 4
SHARD_STEPS = 512
CHECKPOINT_STEPS = 1024
TRANSFER_FORMAT = "chreatures-sensorimotor-corpus-relay-v1"
PROGRESS_FORMAT = f"{FORMAT}-progress"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CHECKPOINT_FILES = frozenset(("boundary", "worlds", "neural", "private"))
PROGRESS_FIELDS = frozenset(
    (
        "format",
        "version",
        "completed",
        "sequence",
        "collection_identity_sha256",
        "identity_receipt",
        "packets",
        "checkpoints",
        "content_sha256",
    )
)
MANIFEST_FIELDS = frozenset(
    (
        "format",
        "version",
        "completed",
        "content_sha256",
        "collection_identity",
        "identity_receipt",
        "collection_identity_sha256",
        "scope",
        "schema",
        "profile",
        "rich_sensorium",
        "transition_outcome_order",
        "organ_flow_order",
        "packets",
        "checkpoints",
        "transitions",
        "elapsed_seconds",
        "world_transport_timing",
    )
)


class RelayError(RuntimeError):
    """Raised when a source or destination fails closed authentication."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RelayError(f"{name} must be an integer >= {minimum}")
    return value


def safe_relative(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RelayError(f"{name} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise RelayError(f"{name} is not a safe relative path")
    if str(path) != value:
        raise RelayError(f"{name} is not canonical POSIX syntax")
    return path


def file_receipt(value: Any, name: str) -> tuple[PurePosixPath, int, str]:
    if not isinstance(value, Mapping):
        raise RelayError(f"{name} must be an object")
    path = safe_relative(value.get("path"), f"{name}.path")
    size = exact_int(value.get("bytes"), f"{name}.bytes", minimum=1)
    digest = value.get("sha256")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise RelayError(f"{name}.sha256 must be lowercase SHA-256")
    return path, size, digest


def add_file(
    files: dict[PurePosixPath, tuple[int, str]],
    path: PurePosixPath,
    size: int,
    digest: str,
) -> None:
    identity = (size, digest)
    existing = files.get(path)
    if existing is not None and existing != identity:
        raise RelayError(f"conflicting receipts for {path}")
    files[path] = identity


def progress_file_set(
    value: Mapping[str, Any],
) -> dict[PurePosixPath, tuple[int, str]]:
    files: dict[PurePosixPath, tuple[int, str]] = {}
    identity_path, identity_size, identity_sha = file_receipt(
        value.get("identity_receipt"), "identity_receipt"
    )
    if identity_path != PurePosixPath("identity.json"):
        raise RelayError("identity receipt path must be identity.json")
    add_file(files, identity_path, identity_size, identity_sha)
    packets = value.get("packets")
    checkpoints = value.get("checkpoints")
    if not isinstance(packets, list) or not isinstance(checkpoints, list):
        raise RelayError("progress packets/checkpoints must be arrays")
    previous: tuple[int, int] | None = None
    sealed_stop: dict[int, int] = {}
    for index, packet in enumerate(packets):
        if not isinstance(packet, Mapping):
            raise RelayError(f"packets[{index}] must be an object")
        episode = exact_int(packet.get("episode"), f"packets[{index}].episode")
        shard = exact_int(packet.get("shard"), f"packets[{index}].shard")
        if previous is None:
            expected = (0, 0)
        elif previous[0] == episode:
            expected = (previous[0], previous[1] + 1)
        else:
            expected = (previous[0] + 1, 0)
        if (episode, shard) != expected:
            raise RelayError("progress packets are not a contiguous prefix")
        previous = (episode, shard)
        start = exact_int(packet.get("start_tick"), f"packets[{index}].start_tick")
        stop = exact_int(packet.get("stop_tick"), f"packets[{index}].stop_tick")
        if start != shard * SHARD_STEPS or stop != start + SHARD_STEPS:
            raise RelayError(f"packets[{index}] tick interval differs")
        packet_path, size, digest = file_receipt(packet, f"packets[{index}]")
        if packet_path != PurePosixPath(
            f"episode-{episode:03d}-shard-{shard:03d}.npz"
        ):
            raise RelayError(f"packets[{index}] path differs")
        add_file(files, packet_path, size, digest)
        sealed_stop[episode] = stop
    previous_checkpoint: tuple[int, int] | None = None
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise RelayError(f"checkpoints[{index}] must be an object")
        episode = exact_int(
            checkpoint.get("episode"), f"checkpoints[{index}].episode"
        )
        tick = exact_int(checkpoint.get("tick"), f"checkpoints[{index}].tick", minimum=1)
        if tick > sealed_stop.get(episode, -1) or (
            tick % CHECKPOINT_STEPS and tick != sealed_stop.get(episode)
        ):
            raise RelayError(f"checkpoints[{index}] is not at a sealed boundary")
        if previous_checkpoint is not None and (episode, tick) <= previous_checkpoint:
            raise RelayError("progress checkpoints are not strictly ordered")
        previous_checkpoint = (episode, tick)
        directory = safe_relative(
            checkpoint.get("path"), f"checkpoints[{index}].path"
        )
        if directory != PurePosixPath(f"episode-{episode:03d}-tick-{tick:08d}"):
            raise RelayError(f"checkpoints[{index}] directory differs")
        members = checkpoint.get("files")
        if not isinstance(members, Mapping) or set(members) != CHECKPOINT_FILES:
            raise RelayError(f"checkpoints[{index}] file set differs")
        tree = checkpoint.get("tree_sha256")
        if not isinstance(tree, str) or canonical_hash(members) != tree:
            raise RelayError(f"checkpoints[{index}] tree_sha256 differs")
        for member_name in sorted(members):
            member_path, size, digest = file_receipt(
                members[member_name], f"checkpoints[{index}].files.{member_name}"
            )
            if len(member_path.parts) != 1:
                raise RelayError("checkpoint member path must be a basename")
            add_file(files, directory / member_path, size, digest)
    return files


def validate_progress_bytes(
    data: bytes,
) -> tuple[dict[str, Any], dict[PurePosixPath, tuple[int, str]]]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelayError(f"cannot decode progress.json: {error}") from error
    if not isinstance(value, dict) or set(value) != PROGRESS_FIELDS:
        raise RelayError("progress fields differ from the rich-v4 relay contract")
    if (
        value.get("format") != PROGRESS_FORMAT
        or value.get("version") != VERSION
        or value.get("completed") is not False
    ):
        raise RelayError("progress identity differs")
    sequence = exact_int(value.get("sequence"), "progress.sequence", minimum=1)
    identity_sha = value.get("collection_identity_sha256")
    if not isinstance(identity_sha, str) or not SHA256.fullmatch(identity_sha):
        raise RelayError("progress collection identity SHA-256 is invalid")
    claimed = value.get("content_sha256")
    unhashed = dict(value)
    unhashed.pop("content_sha256")
    if not isinstance(claimed, str) or canonical_hash(unhashed) != claimed:
        raise RelayError("progress content_sha256 differs")
    files = progress_file_set(value)
    if sequence != len(value["packets"]) + len(value["checkpoints"]):
        raise RelayError("progress sequence differs from sealed object count")
    return value, files


def extends_progress(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    return (
        new["sequence"] >= old["sequence"]
        and new["collection_identity_sha256"]
        == old["collection_identity_sha256"]
        and new["identity_receipt"] == old["identity_receipt"]
        and new["packets"][: len(old["packets"])] == old["packets"]
        and new["checkpoints"][: len(old["checkpoints"])] == old["checkpoints"]
    )


def validate_manifest(path: Path) -> tuple[dict[str, Any], dict[PurePosixPath, tuple[int, str]]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelayError(f"cannot read completed manifest: {error}") from error
    if not isinstance(value, dict):
        raise RelayError("manifest must be an object")
    if set(value) != MANIFEST_FIELDS:
        raise RelayError("manifest fields differ from the rich-v4 relay contract")
    if (
        value.get("format") != FORMAT
        or value.get("version") != VERSION
        or value.get("completed") is not True
    ):
        raise RelayError("manifest is not a completed rich-v4 corpus")
    claimed = value.get("content_sha256")
    if not isinstance(claimed, str) or not SHA256.fullmatch(claimed):
        raise RelayError("manifest content_sha256 is invalid")
    unhashed = dict(value)
    unhashed.pop("content_sha256")
    if canonical_hash(unhashed) != claimed:
        raise RelayError("manifest content_sha256 differs")
    collection_identity = value.get("collection_identity")
    if not isinstance(collection_identity, Mapping):
        raise RelayError("collection_identity must be an object")
    collection_identity_sha = value.get("collection_identity_sha256")
    if (
        not isinstance(collection_identity_sha, str)
        or not SHA256.fullmatch(collection_identity_sha)
        or collection_identity.get("sha256") != collection_identity_sha
    ):
        raise RelayError("collection identity SHA-256 fields differ")
    identity_unhashed = dict(collection_identity)
    identity_unhashed.pop("sha256", None)
    if canonical_hash(identity_unhashed) != collection_identity_sha:
        raise RelayError("collection identity content differs")

    scope = value.get("scope")
    if not isinstance(scope, Mapping):
        raise RelayError("manifest scope must be an object")
    if (
        scope.get("shard_steps") != SHARD_STEPS
        or scope.get("checkpoint_steps") != CHECKPOINT_STEPS
    ):
        raise RelayError("manifest shard/checkpoint intervals differ")
    worlds = exact_int(scope.get("worlds"), "scope.worlds", minimum=1)
    residents = exact_int(
        scope.get("residents_per_world"), "scope.residents_per_world", minimum=1
    )
    episodes = exact_int(scope.get("episodes"), "scope.episodes", minimum=1)
    episode_steps = exact_int(
        scope.get("steps_per_episode"), "scope.steps_per_episode", minimum=1
    )
    if episode_steps % SHARD_STEPS:
        raise RelayError("steps_per_episode is not divisible by shard_steps")
    split_names = (
        "train_world_slots",
        "validation_world_slots",
        "heldout_world_slots",
    )
    split_rows = [scope.get(name) for name in split_names]
    if not all(
        isinstance(rows, list)
        and all(isinstance(row, int) and not isinstance(row, bool) for row in rows)
        for rows in split_rows
    ):
        raise RelayError("world split rows must be integer arrays")
    if [row for rows in split_rows for row in rows] != list(range(worlds)):
        raise RelayError("world splits must partition ordered world slots")
    if not split_rows[0] or not split_rows[1] or not split_rows[2]:
        raise RelayError("train, validation, and heldout splits must be nonempty")
    expected_transitions = worlds * residents * episodes * episode_steps
    if value.get("transitions") != expected_transitions:
        raise RelayError("manifest transition count differs")

    files: dict[PurePosixPath, tuple[int, str]] = {}
    identity_path, identity_size, identity_sha = file_receipt(
        value.get("identity_receipt"), "identity_receipt"
    )
    if identity_path != PurePosixPath("identity.json"):
        raise RelayError("identity receipt path must be identity.json")
    add_file(files, identity_path, identity_size, identity_sha)

    packets = value.get("packets")
    if not isinstance(packets, list):
        raise RelayError("manifest packets must be an array")
    expected_packet_count = episodes * (episode_steps // SHARD_STEPS)
    if len(packets) != expected_packet_count:
        raise RelayError("manifest packet count differs")
    for index, packet in enumerate(packets):
        if not isinstance(packet, Mapping):
            raise RelayError(f"packets[{index}] must be an object")
        episode = exact_int(packet.get("episode"), f"packets[{index}].episode")
        shard = exact_int(packet.get("shard"), f"packets[{index}].shard")
        expected_episode = index // (episode_steps // SHARD_STEPS)
        expected_shard = index % (episode_steps // SHARD_STEPS)
        if (episode, shard) != (expected_episode, expected_shard):
            raise RelayError("packets are not ordered by contiguous episode/shard")
        start = exact_int(packet.get("start_tick"), f"packets[{index}].start_tick")
        stop = exact_int(packet.get("stop_tick"), f"packets[{index}].stop_tick")
        if start != shard * SHARD_STEPS or stop != start + SHARD_STEPS:
            raise RelayError(f"packets[{index}] tick interval differs")
        packet_path, size, digest = file_receipt(packet, f"packets[{index}]")
        expected_name = f"episode-{episode:03d}-shard-{shard:03d}.npz"
        if packet_path != PurePosixPath(expected_name):
            raise RelayError(f"packets[{index}] path differs")
        add_file(files, packet_path, size, digest)

    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise RelayError("manifest checkpoints must be an array")
    checkpoints_per_episode = (
        episode_steps + CHECKPOINT_STEPS - 1
    ) // CHECKPOINT_STEPS
    if len(checkpoints) != episodes * checkpoints_per_episode:
        raise RelayError("manifest checkpoint count differs")
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise RelayError(f"checkpoints[{index}] must be an object")
        episode = exact_int(
            checkpoint.get("episode"), f"checkpoints[{index}].episode"
        )
        tick = exact_int(checkpoint.get("tick"), f"checkpoints[{index}].tick", minimum=1)
        expected_episode = index // checkpoints_per_episode
        expected_tick = min(
            (index % checkpoints_per_episode + 1) * CHECKPOINT_STEPS,
            episode_steps,
        )
        if (episode, tick) != (expected_episode, expected_tick):
            raise RelayError("checkpoints are not ordered by contiguous episode/tick")
        directory = safe_relative(
            checkpoint.get("path"), f"checkpoints[{index}].path"
        )
        if directory != PurePosixPath(f"episode-{episode:03d}-tick-{tick:08d}"):
            raise RelayError(f"checkpoints[{index}] directory differs")
        members = checkpoint.get("files")
        if not isinstance(members, Mapping) or set(members) != CHECKPOINT_FILES:
            raise RelayError(f"checkpoints[{index}] file set differs")
        tree = checkpoint.get("tree_sha256")
        if not isinstance(tree, str) or not SHA256.fullmatch(tree):
            raise RelayError(f"checkpoints[{index}].tree_sha256 is invalid")
        if canonical_hash(members) != tree:
            raise RelayError(f"checkpoints[{index}] tree_sha256 differs")
        for member_name in sorted(members):
            member_path, size, digest = file_receipt(
                members[member_name], f"checkpoints[{index}].files.{member_name}"
            )
            if len(member_path.parts) != 1:
                raise RelayError("checkpoint member path must be a basename")
            add_file(files, directory / member_path, size, digest)
    return value, files


def validate_source(
    root: Path, files: Mapping[PurePosixPath, tuple[int, str]], *, hash_files: bool
) -> None:
    for relative, (size, digest) in files.items():
        path = root.joinpath(*relative.parts)
        try:
            actual_size = path.stat().st_size
        except OSError as error:
            raise RelayError(f"cannot stat source {relative}: {error}") from error
        if not path.is_file() or actual_size != size:
            raise RelayError(f"source size differs for {relative}")
        if hash_files and sha256(path) != digest:
            raise RelayError(f"source SHA-256 differs for {relative}")


def validate_identity_file(root: Path, manifest: Mapping[str, Any]) -> None:
    path = root / "identity.json"
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelayError(f"cannot read identity.json: {error}") from error
    if identity != manifest["collection_identity"]:
        raise RelayError("identity.json differs from manifest collection_identity")


def validate_progress_identity(root: Path, progress: Mapping[str, Any]) -> None:
    path = root / "identity.json"
    receipt_path, size, digest = file_receipt(
        progress["identity_receipt"], "identity_receipt"
    )
    if receipt_path != PurePosixPath("identity.json"):
        raise RelayError("progress identity receipt path differs")
    try:
        data = path.read_bytes()
        identity = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelayError(f"cannot read progress identity.json: {error}") from error
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise RelayError("progress identity receipt differs from identity.json")
    if not isinstance(identity, dict):
        raise RelayError("progress identity.json must be an object")
    claimed = identity.get("sha256")
    unhashed = dict(identity)
    unhashed.pop("sha256", None)
    if (
        claimed != progress["collection_identity_sha256"]
        or canonical_hash(unhashed) != claimed
    ):
        raise RelayError("progress identity content differs")


def ssh_command(target: str, remote_argv: list[str]) -> list[str]:
    return [
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
        target,
        shlex.join(remote_argv),
    ]


REMOTE_STATUS = """
import hashlib,json,os,sys
path=sys.argv[1]; expected_size=int(sys.argv[2]); expected_sha=sys.argv[3]
if not os.path.exists(path):
    print(json.dumps({"status":"missing"})); raise SystemExit(0)
if not os.path.isfile(path) or os.path.getsize(path)!=expected_size:
    print(json.dumps({"status":"different"})); raise SystemExit(0)
h=hashlib.sha256()
with open(path,"rb") as f:
    for block in iter(lambda:f.read(8<<20),b""): h.update(block)
print(json.dumps({"status":"exact" if h.hexdigest()==expected_sha else "different"}))
""".strip()


REMOTE_RECEIVE = """
import hashlib,json,os,sys
dst,tmp,expected_sha=sys.argv[1:4]; expected_size=int(sys.argv[4])
os.makedirs(os.path.dirname(dst),exist_ok=True)
h=hashlib.sha256(); size=0
try:
    with open(tmp,"xb") as f:
        while True:
            block=sys.stdin.buffer.read(8<<20)
            if not block: break
            f.write(block); h.update(block); size+=len(block)
        f.flush(); os.fsync(f.fileno())
    if size!=expected_size or h.hexdigest()!=expected_sha:
        raise RuntimeError("received bytes differ")
    if os.path.exists(dst):
        raise RuntimeError("destination appeared during transfer")
    os.replace(tmp,dst)
    d=os.open(os.path.dirname(dst),os.O_RDONLY)
    try: os.fsync(d)
    finally: os.close(d)
    print(json.dumps({"status":"published","bytes":size,"sha256":expected_sha}))
except BaseException:
    try: os.unlink(tmp)
    except FileNotFoundError: pass
    raise
""".strip()


REMOTE_READ_JSON = """
import hashlib,json,os,sys
path=sys.argv[1]
if not os.path.exists(path):
    print(json.dumps({"status":"missing"})); raise SystemExit(0)
if not os.path.isfile(path): raise RuntimeError("marker is not a file")
data=open(path,"rb").read()
value={"status":"present","sha256":hashlib.sha256(data).hexdigest(),"value":json.loads(data)}
print(json.dumps(value))
""".strip()


REMOTE_REPLACE = """
import hashlib,json,os,sys
dst,tmp,new_sha,previous_sha=sys.argv[1:5]; expected_size=int(sys.argv[5])
os.makedirs(os.path.dirname(dst),exist_ok=True)
data=sys.stdin.buffer.read(); actual=hashlib.sha256(data).hexdigest()
if len(data)!=expected_size or actual!=new_sha: raise RuntimeError("received marker differs")
if os.path.exists(dst):
    old=open(dst,"rb").read(); old_sha=hashlib.sha256(old).hexdigest()
    if old_sha==new_sha:
        print(json.dumps({"status":"already-present","sha256":new_sha})); raise SystemExit(0)
    if not previous_sha or old_sha!=previous_sha: raise RuntimeError("destination marker changed")
with open(tmp,"xb") as f:
    f.write(data); f.flush(); os.fsync(f.fileno())
try:
    os.replace(tmp,dst)
finally:
    try: os.unlink(tmp)
    except FileNotFoundError: pass
f=os.open(dst,os.O_RDONLY); os.fsync(f); os.close(f)
d=os.open(os.path.dirname(dst),os.O_RDONLY); os.fsync(d); os.close(d)
print(json.dumps({"status":"published","sha256":new_sha}))
""".strip()


def remote_status(target: str, destination: str, size: int, digest: str) -> str:
    result = subprocess.run(
        ssh_command(
            target,
            ["python3", "-c", REMOTE_STATUS, destination, str(size), digest],
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RelayError(f"remote status failed: {result.stderr.strip()}")
    try:
        status = json.loads(result.stdout)["status"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RelayError("remote status returned invalid JSON") from error
    if status not in {"missing", "exact", "different"}:
        raise RelayError("remote status returned an unknown state")
    return status


def remote_json(target: str, destination: str) -> tuple[dict[str, Any] | None, str | None]:
    result = subprocess.run(
        ssh_command(target, ["python3", "-c", REMOTE_READ_JSON, destination]),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RelayError(f"remote marker read failed: {result.stderr.strip()}")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RelayError("remote marker read returned invalid JSON") from error
    if response.get("status") == "missing":
        return None, None
    if (
        response.get("status") != "present"
        or not isinstance(response.get("value"), dict)
        or not isinstance(response.get("sha256"), str)
    ):
        raise RelayError("remote marker read returned an unknown state")
    return response["value"], response["sha256"]


def replace_remote_marker(
    data: bytes,
    target: str,
    destination: str,
    previous_sha: str | None,
    token: str,
) -> str:
    digest = hashlib.sha256(data).hexdigest()
    temporary = f"{destination}.tmp-{token}"
    result = subprocess.run(
        ssh_command(
            target,
            [
                "python3",
                "-c",
                REMOTE_REPLACE,
                destination,
                temporary,
                digest,
                previous_sha or "",
                str(len(data)),
            ],
        ),
        input=data,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RelayError(
            f"remote marker publish failed: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RelayError("remote marker publish returned invalid JSON") from error
    if response.get("sha256") != digest or response.get("status") not in {
        "published",
        "already-present",
    }:
        raise RelayError("remote marker publication was not authenticated")
    return digest


def publish_file(
    source: Path,
    target: str,
    destination: str,
    size: int,
    digest: str,
    token: str,
) -> str:
    status = remote_status(target, destination, size, digest)
    if status == "exact":
        if source.stat().st_size != size or sha256(source) != digest:
            raise RelayError(f"source changed for {source}")
        return "already-present"
    if status == "different":
        raise RelayError(f"destination differs: {destination}")
    before = source.stat()
    temporary = f"{destination}.tmp-{token}"
    command = ssh_command(
        target,
        [
            "python3",
            "-c",
            REMOTE_RECEIVE,
            destination,
            temporary,
            digest,
            str(size),
        ],
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    local_digest = hashlib.sha256()
    try:
        assert process.stdin is not None
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                local_digest.update(block)
                process.stdin.write(block)
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if process.returncode != 0:
        raise RelayError(
            f"remote publish failed for {source.name}: "
            f"{stderr.decode(errors='replace').strip()}"
        )
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RelayError("remote publish returned invalid JSON") from error
    after = source.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_size != size
        or local_digest.hexdigest() != digest
    ):
        raise RelayError(f"source changed or differs during transfer: {source}")
    if response.get("status") != "published":
        raise RelayError("remote publish did not confirm publication")
    return "published"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="completed local corpus directory")
    parser.add_argument("--destination-host")
    parser.add_argument("--destination-root")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="local receipt path (default: sibling of source directory)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="authenticate the local manifest and files without connecting",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="relay authenticated progress prefixes until the final manifest appears",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    source = args.source.resolve()
    manifest_path = source / "manifest.json"
    progress_path = source / "progress.json"
    if args.poll_seconds < 1.0 or args.poll_seconds > 300.0:
        raise RelayError("poll-seconds must be between 1 and 300")
    if args.validate_only:
        if manifest_path.is_file():
            manifest, files = validate_manifest(manifest_path)
            validate_identity_file(source, manifest)
            identity = manifest["content_sha256"]
            status = "valid-complete"
        elif progress_path.is_file():
            data = progress_path.read_bytes()
            progress, files = validate_progress_bytes(data)
            validate_progress_identity(source, progress)
            identity = progress["content_sha256"]
            status = "valid-progress"
        else:
            raise RelayError("neither manifest.json nor progress.json exists")
        validate_source(source, files, hash_files=True)
        print(
            json.dumps(
                {
                    "status": status,
                    "receipt_content_sha256": identity,
                    "files": len(files),
                    "bytes": sum(size for size, _digest in files.values()),
                },
                sort_keys=True,
            )
        )
        return 0

    if args.destination_root is None or args.destination_host is None:
        raise RelayError("destination host and root are required for relay")
    destination_root = PurePosixPath(args.destination_root)
    if (
        not destination_root.is_absolute()
        or ".." in destination_root.parts
        or "." in destination_root.parts
        or str(destination_root) != args.destination_root
    ):
        raise RelayError("destination root must be absolute")
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", args.destination_host):
        raise RelayError("destination host is invalid")
    receipt_path = args.receipt or source.with_name(f"{source.name}.relay.json")
    started = time.time()
    token = f"{os.getpid()}-{int(started)}"
    remote_progress_path = str(destination_root / "progress.json")
    previous_progress, previous_progress_file_sha = remote_json(
        args.destination_host, remote_progress_path
    )
    if previous_progress is not None:
        previous_progress, _ = validate_progress_bytes(
            canonical_bytes(previous_progress)
        )
    while not manifest_path.is_file():
        if not args.watch:
            raise RelayError("completed manifest does not exist (use --watch for progress)")
        try:
            progress_data = progress_path.read_bytes()
        except FileNotFoundError:
            time.sleep(args.poll_seconds)
            continue
        progress, progress_files = validate_progress_bytes(progress_data)
        validate_progress_identity(source, progress)
        if previous_progress is not None and not extends_progress(
            previous_progress, progress
        ):
            raise RelayError("source progress does not extend destination progress")
        if (
            previous_progress is None
            or progress["sequence"] > previous_progress["sequence"]
        ):
            for relative in sorted(progress_files, key=str):
                size, digest = progress_files[relative]
                publish_file(
                    source.joinpath(*relative.parts),
                    args.destination_host,
                    str(destination_root / relative),
                    size,
                    digest,
                    token,
                )
            previous_progress_file_sha = replace_remote_marker(
                progress_data,
                args.destination_host,
                remote_progress_path,
                previous_progress_file_sha,
                token,
            )
            previous_progress = progress
            partial = {
                "format": TRANSFER_FORMAT,
                "version": 1,
                "status": "in-progress",
                "collection_identity_sha256": progress[
                    "collection_identity_sha256"
                ],
                "progress_sequence": progress["sequence"],
                "progress_content_sha256": progress["content_sha256"],
                "destination": {
                    "host": args.destination_host,
                    "root": args.destination_root,
                },
                "file_count": len(progress_files),
                "payload_bytes": sum(size for size, _ in progress_files.values()),
                "started_at_unix": started,
                "updated_at_unix": time.time(),
            }
            partial["content_sha256"] = canonical_hash(partial)
            atomic_json(receipt_path, partial)
        time.sleep(args.poll_seconds)

    manifest, files = validate_manifest(manifest_path)
    validate_identity_file(source, manifest)
    if previous_progress is not None:
        if (
            manifest["collection_identity_sha256"]
            != previous_progress["collection_identity_sha256"]
            or manifest["identity_receipt"] != previous_progress["identity_receipt"]
            or manifest["packets"][: len(previous_progress["packets"])]
            != previous_progress["packets"]
            or manifest["checkpoints"][: len(previous_progress["checkpoints"])]
            != previous_progress["checkpoints"]
        ):
            raise RelayError("final manifest does not extend published progress")
    rows = []
    for relative in sorted(files, key=str):
        size, digest = files[relative]
        local_path = source.joinpath(*relative.parts)
        destination = str(PurePosixPath(args.destination_root) / relative)
        publish_file(
            local_path,
            args.destination_host,
            destination,
            size,
            digest,
            token,
        )
        rows.append(
            {
                "path": str(relative),
                "bytes": size,
                "sha256": digest,
            }
        )

    manifest_size = manifest_path.stat().st_size
    manifest_sha = sha256(manifest_path)
    transfer = {
        "format": TRANSFER_FORMAT,
        "version": 1,
        "source_manifest": {
            "file_sha256": manifest_sha,
            "content_sha256": manifest["content_sha256"],
        },
        "destination": {
            "host": args.destination_host,
            "root": args.destination_root,
            "publication_marker": "manifest.json",
        },
        "files": rows,
        "file_count": len(rows),
        "payload_bytes": sum(row["bytes"] for row in rows),
    }
    transfer["content_sha256"] = canonical_hash(transfer)
    atomic_json(receipt_path, transfer)
    receipt_sha = sha256(receipt_path)
    receipt_destination = str(
        PurePosixPath(args.destination_root) / "transfer-receipt.json"
    )
    publish_file(
        receipt_path,
        args.destination_host,
        receipt_destination,
        receipt_path.stat().st_size,
        receipt_sha,
        token,
    )
    # The completed source manifest is the destination publication marker and
    # must remain the final rename in a successful relay.
    publish_file(
        manifest_path,
        args.destination_host,
        str(PurePosixPath(args.destination_root) / "manifest.json"),
        manifest_size,
        manifest_sha,
        token,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_file_sha256": manifest_sha,
                "manifest_content_sha256": manifest["content_sha256"],
                "receipt": str(receipt_path),
                "receipt_file_sha256": receipt_sha,
                "files": len(rows),
                "bytes": transfer["payload_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RelayError, OSError, ValueError) as error:
        print(f"sensorimotor corpus relay error: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error
