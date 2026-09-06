"""Identity-bound launch and supervision for external campaign jobs.

This module deliberately knows nothing about population search.  It verifies a
sealed job description, prepares private candidate storage, and supervises one
ordinary argv command without invoking a shell.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


FORMAT = "chreatures-campaign-job-v1"
STATE_FORMAT = "chreatures-campaign-job-state-v1"
EXIT_FORMAT = "chreatures-campaign-job-exit-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL = {
    "format",
    "version",
    "job_id",
    "campaign_root",
    "host",
    "source",
    "environment",
    "compatibility_group",
    "artifacts",
    "candidates",
    "resources",
    "command",
    "paths",
    "metadata",
    "identity_sha256",
}


class CampaignJobError(ValueError):
    """A campaign job is malformed or conflicts with durable state."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignJobError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignJobError(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignJobError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: Any, name: str) -> str:
    value = _string(value, name)
    if not _SHA256.fullmatch(value):
        raise CampaignJobError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, name: str) -> str:
    value = _string(value, name)
    if not _IDENTIFIER.fullmatch(value):
        raise CampaignJobError(f"{name} is not a bounded identifier")
    return value


def _absolute(value: Any, name: str) -> Path:
    path = Path(_string(value, name)).expanduser()
    if not path.is_absolute():
        raise CampaignJobError(f"{name} must be absolute")
    return path.resolve()


def _relative_under(root: Path, value: Any, name: str) -> Path:
    raw = Path(_string(value, name))
    if raw.is_absolute():
        raise CampaignJobError(f"{name} must be relative to campaign_root")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CampaignJobError(f"{name} escapes campaign_root") from error
    return path


def unsealed(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    value.pop("identity_sha256", None)
    return value


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = unsealed(manifest)
    validate_manifest_shape(value, sealed=False)
    value["identity_sha256"] = canonical_sha256(value)
    validate_manifest_shape(value, sealed=True)
    return value


def load_manifest(path: str | Path, *, sealed: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text())
    validate_manifest_shape(value, sealed=sealed)
    if sealed and value["identity_sha256"] != canonical_sha256(unsealed(value)):
        raise CampaignJobError("campaign job manifest seal differs")
    return value


def validate_manifest_shape(value: Any, *, sealed: bool) -> None:
    manifest = _mapping(value, "manifest")
    unknown = set(manifest) - _TOP_LEVEL
    if unknown:
        raise CampaignJobError(f"unknown manifest fields: {sorted(unknown)}")
    required = _TOP_LEVEL - {"metadata", "identity_sha256"}
    missing = required - set(manifest)
    if missing:
        raise CampaignJobError(f"missing manifest fields: {sorted(missing)}")
    if manifest.get("format") != FORMAT or manifest.get("version") != 1:
        raise CampaignJobError("campaign job format or version differs")
    _identifier(manifest.get("job_id"), "job_id")
    _absolute(manifest.get("campaign_root"), "campaign_root")

    host = _mapping(manifest.get("host"), "host")
    _identifier(host.get("role"), "host.role")
    names = host.get("names")
    if not isinstance(names, list) or not names:
        raise CampaignJobError("host.names must be a nonempty list")
    for index, name in enumerate(names):
        _string(name, f"host.names[{index}]")

    source = _mapping(manifest.get("source"), "source")
    if source.get("kind") not in {"git", "archive"}:
        raise CampaignJobError("source.kind must be git or archive")
    _absolute(source.get("root"), "source.root")
    _string(source.get("revision"), "source.revision")
    if source["kind"] == "git":
        clean = source.get("clean_paths")
        if not isinstance(clean, list) or not clean:
            raise CampaignJobError("git source requires clean_paths")
        for index, item in enumerate(clean):
            if Path(_string(item, f"source.clean_paths[{index}]")).is_absolute():
                raise CampaignJobError("source.clean_paths must be relative")
    else:
        receipt = _mapping(source.get("receipt"), "source.receipt")
        _absolute(receipt.get("path"), "source.receipt.path")
        _digest(receipt.get("sha256"), "source.receipt.sha256")
        scoped_files = source.get("scoped_files")
        if not isinstance(scoped_files, list) or not scoped_files:
            raise CampaignJobError("archive source requires scoped_files")
        for index, item in enumerate(scoped_files):
            item = _mapping(item, f"source.scoped_files[{index}]")
            relative = Path(_string(
                item.get("path"), f"source.scoped_files[{index}].path"
            ))
            if relative.is_absolute() or ".." in relative.parts:
                raise CampaignJobError("archive scoped source path escapes root")
            _digest(
                item.get("sha256"), f"source.scoped_files[{index}].sha256"
            )

    environment = _mapping(manifest.get("environment"), "environment")
    _absolute(environment.get("executable"), "environment.executable")
    _digest(environment.get("executable_sha256"), "environment.executable_sha256")
    variables = _mapping(environment.get("variables", {}), "environment.variables")
    for key, item in variables.items():
        _string(key, "environment variable name")
        if not isinstance(item, str):
            raise CampaignJobError("environment variable values must be strings")
    unset = environment.get("unset", [])
    if not isinstance(unset, list) or not all(isinstance(item, str) for item in unset):
        raise CampaignJobError("environment.unset must be a string list")

    group = _mapping(manifest.get("compatibility_group"), "compatibility_group")
    _identifier(group.get("id"), "compatibility_group.id")
    for key in ("graph_sha256", "ports_sha256", "controller_interface_sha256"):
        _digest(group.get(key), f"compatibility_group.{key}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CampaignJobError("artifacts must be a nonempty list")
    artifact_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        artifact = _mapping(artifact, f"artifacts[{index}]")
        artifact_id = _identifier(artifact.get("id"), f"artifacts[{index}].id")
        if artifact_id in artifact_ids:
            raise CampaignJobError("artifact IDs must be unique")
        artifact_ids.add(artifact_id)
        _absolute(artifact.get("path"), f"artifacts[{index}].path")
        if artifact.get("kind") == "file":
            _digest(artifact.get("sha256"), f"artifacts[{index}].sha256")
        elif artifact.get("kind") == "directory":
            identity_file = Path(_string(
                artifact.get("identity_file"), f"artifacts[{index}].identity_file"
            ))
            if identity_file.is_absolute() or ".." in identity_file.parts:
                raise CampaignJobError("artifact identity_file must stay in its directory")
            _digest(
                artifact.get("identity_sha256"),
                f"artifacts[{index}].identity_sha256",
            )
            _digest(
                artifact.get("logical_sha256"),
                f"artifacts[{index}].logical_sha256",
            )
        else:
            raise CampaignJobError("artifact.kind must be file or directory")
    environment_receipts = environment.get("receipt_artifact_ids")
    if not isinstance(environment_receipts, list) or not environment_receipts:
        raise CampaignJobError("environment requires receipt_artifact_ids")
    if not all(
        isinstance(item, str) and item in artifact_ids
        for item in environment_receipts
    ):
        raise CampaignJobError("environment references an unknown receipt artifact")

    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CampaignJobError("candidates must be a nonempty list")
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        candidate = _mapping(candidate, f"candidates[{index}]")
        candidate_id = _identifier(candidate.get("id"), f"candidates[{index}].id")
        if candidate_id in candidate_ids:
            raise CampaignJobError("candidate IDs must be unique")
        candidate_ids.add(candidate_id)
        references = candidate.get("artifact_ids")
        if not isinstance(references, list) or not references:
            raise CampaignJobError("candidate artifact_ids must be a nonempty list")
        if not all(isinstance(item, str) and item in artifact_ids for item in references):
            raise CampaignJobError("candidate references an unknown artifact")

    resources = _mapping(manifest.get("resources"), "resources")
    for key in (
        "min_available_memory_bytes",
        "min_free_disk_bytes",
        "requested_device_memory_bytes",
    ):
        _integer(resources.get(key), f"resources.{key}")
    _integer(resources.get("cpu_threads"), "resources.cpu_threads", minimum=1)
    _absolute(resources.get("disk_path"), "resources.disk_path")
    _string(resources.get("device"), "resources.device")

    command = _mapping(manifest.get("command"), "command")
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise CampaignJobError("command.argv must be a nonempty string list")
    _absolute(argv[0], "command.argv[0]")
    _absolute(command.get("cwd"), "command.cwd")
    resume_argv = command.get("resume_argv")
    resume_mode = command.get("resume_mode")
    if (resume_argv is None) != (resume_mode is None):
        raise CampaignJobError("resume_argv and resume_mode must be declared together")
    if resume_argv is not None:
        if not isinstance(resume_argv, list) or not resume_argv or not all(
            isinstance(x, str) and x for x in resume_argv
        ):
            raise CampaignJobError("command.resume_argv must be a nonempty string list")
        _absolute(resume_argv[0], "command.resume_argv[0]")
        resume_mode = _mapping(resume_mode, "command.resume_mode")
        _identifier(resume_mode.get("id"), "command.resume_mode.id")
        _string(resume_mode.get("semantics"), "command.resume_mode.semantics")

    paths = _mapping(manifest.get("paths"), "paths")
    root = _absolute(manifest.get("campaign_root"), "campaign_root")
    resolved = [
        _relative_under(root, paths.get(key), f"paths.{key}")
        for key in ("run", "supervision", "private_candidates", "shared_cache")
    ]
    if len(set(resolved)) != len(resolved):
        raise CampaignJobError("campaign output paths must be distinct")
    if sealed:
        _digest(manifest.get("identity_sha256"), "identity_sha256")


def resolved_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    root = _absolute(manifest["campaign_root"], "campaign_root")
    values = {"root": root}
    values.update({
        key: _relative_under(root, manifest["paths"][key], f"paths.{key}")
        for key in ("run", "supervision", "private_candidates", "shared_cache")
    })
    return values


def _available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for row in meminfo.read_text().splitlines():
            if row.startswith("MemAvailable:"):
                return int(row.split()[1]) * 1024
    if sys.platform == "darwin":
        text = subprocess.run(
            ["vm_stat"], check=True, capture_output=True, text=True
        ).stdout
        page_size = int(re.search(r"page size of (\d+) bytes", text).group(1))
        rows = {
            key: int(value.replace(".", ""))
            for key, value in re.findall(r"^([^:]+):\s+([0-9.]+)$", text, re.MULTILINE)
        }
        pages = sum(rows.get(key, 0) for key in (
            "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"
        ))
        return pages * page_size
    pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(pages) * int(os.sysconf("SC_PAGE_SIZE"))


def _available_amd_vram_bytes() -> int | None:
    for device in sorted(Path("/sys/class/drm").glob("card*/device")):
        try:
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
            total = int((device / "mem_info_vram_total").read_text())
            used = int((device / "mem_info_vram_used").read_text())
            return max(0, total - used)
        except (FileNotFoundError, ValueError):
            continue
    return None


def resource_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    resources = manifest["resources"]
    disk = shutil.disk_usage(_absolute(resources["disk_path"], "resources.disk_path"))
    return {
        "available_memory_bytes": _available_memory_bytes(),
        "free_disk_bytes": disk.free,
        "available_amd_vram_bytes": _available_amd_vram_bytes(),
        "requested": dict(resources),
    }


def verify_host_and_resources(manifest: Mapping[str, Any]) -> dict[str, Any]:
    names = set(manifest["host"]["names"])
    actual = {socket.gethostname(), socket.getfqdn(), platform.node()}
    if not names.intersection(actual):
        raise CampaignJobError(
            f"job host differs: expected one of {sorted(names)}, got {sorted(actual)}"
        )
    snapshot = resource_snapshot(manifest)
    resources = manifest["resources"]
    if snapshot["available_memory_bytes"] < resources["min_available_memory_bytes"]:
        raise CampaignJobError("available host memory is below the job request")
    if snapshot["free_disk_bytes"] < resources["min_free_disk_bytes"]:
        raise CampaignJobError("free bulk disk is below the job request")
    if resources["cpu_threads"] > (os.cpu_count() or 1):
        raise CampaignJobError("host CPU thread count is below the job request")
    available_vram = snapshot["available_amd_vram_bytes"]
    requested_vram = resources["requested_device_memory_bytes"]
    if available_vram is not None and available_vram < requested_vram:
        raise CampaignJobError("available AMD VRAM is below the job request")
    return snapshot


def verify_source(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = manifest["source"]
    root = _absolute(source["root"], "source.root")
    if not root.is_dir():
        raise CampaignJobError("source root does not exist")
    if source["kind"] == "git":
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != source["revision"]:
            raise CampaignJobError("source Git revision differs")
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", *source["clean_paths"]],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if dirty:
            raise CampaignJobError("source scoped paths are dirty")
        return {"kind": "git", "revision": revision, "dirty": False}
    receipt = source["receipt"]
    receipt_path = _absolute(receipt["path"], "source.receipt.path")
    if not receipt_path.is_file() or file_sha256(receipt_path) != receipt["sha256"]:
        raise CampaignJobError("authenticated source archive receipt differs")
    for item in source["scoped_files"]:
        path = (root / item["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise CampaignJobError("archive scoped source path escapes root") from error
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise CampaignJobError(f"archive scoped source differs: {item['path']}")
    return {
        "kind": "archive",
        "revision": source["revision"],
        "receipt_sha256": receipt["sha256"],
    }


def verify_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = {}
    for artifact in manifest["artifacts"]:
        path = _absolute(artifact["path"], f"artifact {artifact['id']} path")
        if artifact["kind"] == "file":
            if not path.is_file():
                raise CampaignJobError(f"artifact {artifact['id']} is missing")
            digest = file_sha256(path)
            if digest != artifact["sha256"]:
                raise CampaignJobError(f"artifact {artifact['id']} hash differs")
            verified[artifact["id"]] = {"kind": "file", "sha256": digest}
        else:
            if not path.is_dir():
                raise CampaignJobError(f"artifact {artifact['id']} directory is missing")
            identity_path = (path / artifact["identity_file"]).resolve()
            try:
                identity_path.relative_to(path)
            except ValueError as error:
                raise CampaignJobError("artifact identity path escapes directory") from error
            if not identity_path.is_file():
                raise CampaignJobError(f"artifact {artifact['id']} identity is missing")
            digest = file_sha256(identity_path)
            if digest != artifact["identity_sha256"]:
                raise CampaignJobError(f"artifact {artifact['id']} identity differs")
            verified[artifact["id"]] = {
                "kind": "directory",
                "identity_sha256": digest,
                "logical_sha256": artifact["logical_sha256"],
            }
    executable = _absolute(
        manifest["environment"]["executable"], "environment.executable"
    )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise CampaignJobError("environment executable is missing or not executable")
    if file_sha256(executable) != manifest["environment"]["executable_sha256"]:
        raise CampaignJobError("environment executable hash differs")
    return verified


def validate_runtime(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate resource budget before touching graph or weight artifacts."""
    resources = verify_host_and_resources(manifest)
    source = verify_source(manifest)
    artifacts = verify_artifacts(manifest)
    command = manifest["command"]
    cwd = _absolute(command["cwd"], "command.cwd")
    if not cwd.is_dir():
        raise CampaignJobError("command cwd does not exist")
    executable = _absolute(
        manifest["environment"]["executable"], "environment.executable"
    )
    if _absolute(command["argv"][0], "command.argv[0]") != executable:
        raise CampaignJobError("command argv[0] differs from pinned executable")
    if "resume_argv" in command and _absolute(
        command["resume_argv"][0], "command.resume_argv[0]"
    ) != executable:
        raise CampaignJobError("command resume_argv[0] differs from pinned executable")
    return {"resources": resources, "source": source, "artifacts": artifacts}


@contextmanager
def _launch_lock(supervision: Path) -> Iterator[None]:
    supervision.mkdir(parents=True, exist_ok=True)
    with (supervision / ".launch.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _read_state(supervision: Path) -> dict[str, Any] | None:
    path = supervision / "state.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _state(
    manifest: Mapping[str, Any], status: str, attempt: int, **extra: Any
) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "version": 1,
        "job_id": manifest["job_id"],
        "job_identity_sha256": manifest["identity_sha256"],
        "status": status,
        "attempt": attempt,
        "updated_utc": utc_now(),
        **extra,
    }


def prepare_candidate_storage(manifest: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    base = paths["private_candidates"]
    base.mkdir(parents=True, exist_ok=True)
    for candidate in manifest["candidates"]:
        target = base / candidate["id"]
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "candidate.json"
        value = {
            "format": "chreatures-private-candidate-state-v1",
            "job_identity_sha256": manifest["identity_sha256"],
            "compatibility_group_id": manifest["compatibility_group"]["id"],
            "candidate": candidate,
            "candidate_identity_sha256": canonical_sha256(candidate),
        }
        if marker.is_file():
            current = json.loads(marker.read_text())
            if current != value:
                raise CampaignJobError(
                    f"private candidate identity differs for {candidate['id']}"
                )
        else:
            atomic_json(marker, value)


def launch_job(
    manifest_path: str | Path,
    *,
    launcher_path: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    runtime = validate_runtime(manifest)
    paths = resolved_paths(manifest)
    supervision = paths["supervision"]
    with _launch_lock(supervision):
        stored_manifest = supervision / "job.json"
        if stored_manifest.is_file():
            stored = load_manifest(stored_manifest)
            if stored["identity_sha256"] != manifest["identity_sha256"]:
                raise CampaignJobError("supervision directory belongs to another job identity")
        else:
            atomic_json(stored_manifest, manifest)
        state = _read_state(supervision)
        if state is not None:
            if state.get("job_identity_sha256") != manifest["identity_sha256"]:
                raise CampaignJobError("durable job state identity differs")
            if state["status"] in {"pending", "running", "completed"}:
                return state
            if state["status"] != "failed":
                raise CampaignJobError("durable job state status is invalid")
            if not resume:
                return state
        elif resume:
            raise CampaignJobError("cannot resume a job with no failed attempt")
        if resume and "resume_argv" not in manifest["command"]:
            raise CampaignJobError("job does not declare an explicit resume mode")
        if not resume and paths["run"].exists():
            if not paths["run"].is_dir() or any(paths["run"].iterdir()):
                raise CampaignJobError("new job run directory is not empty")

        paths["shared_cache"].mkdir(parents=True, exist_ok=True)
        prepare_candidate_storage(manifest, paths)
        attempt = 1 if state is None else int(state["attempt"]) + 1
        attempt_dir = supervision / "attempts" / f"{attempt:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        pending = _state(
            manifest,
            "pending",
            attempt,
            mode="resume" if resume else "launch",
            runtime_validation=runtime,
        )
        atomic_json(supervision / "state.json", pending)
        log = (attempt_dir / "run.log").open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(launcher_path).expanduser().resolve()),
                    "_supervise",
                    str(stored_manifest),
                    str(attempt),
                    "resume" if resume else "launch",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            log.close()
        atomic_text(attempt_dir / "supervisor-launch.pid", f"{process.pid}\n")
    for _ in range(50):
        time.sleep(0.1)
        current = _read_state(supervision)
        if current is not None and current["status"] != "pending":
            return current
        if process.poll() is not None:
            break
    return _read_state(supervision) or pending


def supervise_job(manifest_path: str | Path, attempt: int, mode: str) -> int:
    manifest = load_manifest(manifest_path)
    paths = resolved_paths(manifest)
    supervision = paths["supervision"]
    attempt_dir = supervision / "attempts" / f"{attempt:04d}"
    child: subprocess.Popen[bytes] | None = None
    runtime: dict[str, Any] | None = None
    started = utc_now()
    try:
        with _launch_lock(supervision):
            state = _read_state(supervision)
            if (
                state is None
                or state.get("job_identity_sha256") != manifest["identity_sha256"]
                or state.get("status") != "pending"
                or int(state.get("attempt", -1)) != attempt
                or state.get("mode") != mode
            ):
                raise CampaignJobError("pending supervisor state differs")
            runtime = validate_runtime(manifest)
            atomic_text(attempt_dir / "supervisor.pid", f"{os.getpid()}\n")
            atomic_text(
                attempt_dir / "launcher.sha256",
                f"{file_sha256(Path(sys.argv[0]).resolve())}\n",
            )
            command = manifest["command"]
            argv = command["resume_argv"] if mode == "resume" else command["argv"]
            environment = os.environ.copy()
            for name in manifest["environment"].get("unset", []):
                environment.pop(name, None)
            environment.update(manifest["environment"].get("variables", {}))
            child = subprocess.Popen(
                argv,
                cwd=command["cwd"],
                env=environment,
                close_fds=True,
            )
            atomic_text(attempt_dir / "child.pid", f"{child.pid}\n")
            running = _state(
                manifest,
                "running",
                attempt,
                mode=mode,
                started_utc=started,
                supervisor_pid=os.getpid(),
                child_pid=child.pid,
                runtime_validation=runtime,
            )
            atomic_json(supervision / "state.json", running)

        def forward(signum: int, _frame: Any) -> None:
            if child is not None and child.poll() is None:
                child.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        code = child.wait()
        ended = utc_now()
        status = "completed" if code == 0 else "failed"
        exit_value = {
            "format": EXIT_FORMAT,
            "version": 1,
            "job_id": manifest["job_id"],
            "job_identity_sha256": manifest["identity_sha256"],
            "attempt": attempt,
            "mode": mode,
            "started_utc": started,
            "ended_utc": ended,
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid,
            "exit_code": code,
            "status": status,
            "runtime_validation": runtime,
        }
        atomic_json(attempt_dir / "exit-status.json", exit_value)
        with _launch_lock(supervision):
            atomic_json(
                supervision / "state.json",
                _state(
                    manifest,
                    status,
                    attempt,
                    mode=mode,
                    started_utc=started,
                    ended_utc=ended,
                    supervisor_pid=os.getpid(),
                    child_pid=child.pid,
                    exit_code=code,
                    exit_receipt=str(attempt_dir / "exit-status.json"),
                    runtime_validation=runtime,
                ),
            )
        return code
    except BaseException as error:
        ended = utc_now()
        exit_value = {
            "format": EXIT_FORMAT,
            "version": 1,
            "job_id": manifest["job_id"],
            "job_identity_sha256": manifest["identity_sha256"],
            "attempt": attempt,
            "mode": mode,
            "started_utc": started,
            "ended_utc": ended,
            "supervisor_pid": os.getpid(),
            "child_pid": child.pid if child is not None else None,
            "exit_code": 125,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "runtime_validation": runtime,
        }
        atomic_json(attempt_dir / "exit-status.json", exit_value)
        with _launch_lock(supervision):
            atomic_json(
                supervision / "state.json",
                _state(
                    manifest,
                    "failed",
                    attempt,
                    mode=mode,
                    started_utc=started,
                    ended_utc=ended,
                    supervisor_pid=os.getpid(),
                    child_pid=child.pid if child is not None else None,
                    exit_code=125,
                    exit_receipt=str(attempt_dir / "exit-status.json"),
                    runtime_validation=runtime,
                ),
            )
        print(f"campaign supervisor failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 125


def job_status(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    supervision = resolved_paths(manifest)["supervision"]
    if not supervision.exists():
        return _state(manifest, "pending", 0, reason="not launched")
    stored = supervision / "job.json"
    if stored.is_file() and load_manifest(stored)["identity_sha256"] != manifest["identity_sha256"]:
        raise CampaignJobError("supervision directory belongs to another job identity")
    return _read_state(supervision) or _state(
        manifest, "pending", 0, reason="not launched"
    )
