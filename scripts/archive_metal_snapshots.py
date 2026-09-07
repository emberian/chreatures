#!/usr/bin/env python3
"""Plan, archive, or restore old automatic Metal brain snapshots."""

from __future__ import annotations
import argparse, hashlib, json, os, re, stat, subprocess, tempfile, time
from contextlib import contextmanager
from pathlib import Path

AUTO = re.compile(r"world-([0-9a-f-]{36})-([0-9]+)\.npz\Z")
HEX64 = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
WORLD_NAME = re.compile(rb"world-[0-9a-f-]{36}-[0-9]+")
ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path, stat):
    return (
        str(path.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def cached_digest(path, stat, cache):
    """Reuse a digest only while the file's complete identity is unchanged."""
    if cache is None:
        return digest(path)
    key = file_identity(path, stat)
    value = cache.get(key)
    if value is None:
        value = digest(path)
        cache[key] = value
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


@contextmanager
def ssh_multiplex(host):
    """Reuse only this process's private SSH connection and close it on exit."""
    with tempfile.TemporaryDirectory(prefix="cma-", dir="/tmp") as directory:
        os.chmod(directory, 0o700)
        control = str(Path(directory) / "control.sock")
        options = [
            "-C",  # Compress transport; remote snapshots keep their exact raw bytes.
            "-o",
            f"ControlPath={control}",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=30",
        ]
        try:
            yield options
        finally:
            subprocess.run(
                ["ssh", *options, "-O", "exit", host],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def references(roots, ignore=(), cache=None):
    ignored = {Path(x).resolve() for x in ignore}
    hashes = set()
    names = set()
    sources = []
    for root in roots:
        try:
            root.stat()
        except FileNotFoundError:
            continue
        for p in root.rglob("*"):
            if (
                p.resolve() in ignored
                or not (
                    p.suffix.lower() in {".json", ".backup", ".bak"}
                    or "manifest" in p.name.lower()
                )
            ):
                continue
            try:
                entry_stat = p.stat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                continue
            while True:
                try:
                    before = p.stat()
                    key = file_identity(p, before)
                    tokens = None if cache is None else cache.get(key)
                    if tokens is None:
                        raw = p.read_bytes()
                        after = p.stat()
                        if file_identity(p, after) != key:
                            continue
                        tokens = (
                            frozenset(x.decode() for x in HEX64.findall(raw)),
                            frozenset(x.decode() for x in WORLD_NAME.findall(raw)),
                        )
                        if cache is not None:
                            cache[key] = tokens
                except FileNotFoundError:
                    tokens = None
                break
            if tokens is None:
                continue
            hashes.update(tokens[0])
            names.update(tokens[1])
            sources.append(str(p))
    return hashes, names, sources


def scan(
    snapshot_dirs,
    roots,
    min_age,
    ignore=(),
    digest_cache=None,
    reference_cache=None,
):
    hashes, names, sources = references(roots, ignore, cache=reference_cache)
    now = time.time()
    groups = {}
    rows = []
    for directory in snapshot_dirs:
        for p in directory.glob("*.npz"):
            m = AUTO.fullmatch(p.name)
            if m:
                groups.setdefault((str(directory.resolve()), m.group(1)), []).append(p)
    newest = {
        max(
            v,
            key=lambda p: (p.stat().st_mtime_ns, int(AUTO.fullmatch(p.name).group(2))),
        )
        for v in groups.values()
    }
    for files in groups.values():
        for p in files:
            st = p.stat()
            sha = cached_digest(p, st, digest_cache)
            reason = None
            if p in newest:
                reason = "newest_for_world"
            elif p.stem in names or sha in hashes:
                reason = "referenced"
            elif now - st.st_mtime < min_age:
                reason = "too_new"
            rows.append(
                {
                    "path": str(p.resolve()),
                    "name": p.name,
                    "sha256": sha,
                    "bytes": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "safe": reason is None,
                    "reason": reason,
                }
            )
    return rows, {
        "reference_files": len(sources),
        "referenced_hashes": len(hashes),
        "referenced_names": len(names),
    }


def remote_object(base, sha):
    return f"{base}/objects/{sha[:2]}/{sha}.npz"


def archive_one(row, host, base, ssh_options=()):
    path = Path(row["path"])
    st = path.stat()
    if (st.st_size, st.st_mtime_ns, digest(path)) != (
        row["bytes"],
        row["mtime_ns"],
        row["sha256"],
    ):
        raise RuntimeError(f"changed during archive scan: {path}")
    target = remote_object(base, row["sha256"])
    temporary = f"{target}.tmp-{os.getpid()}"
    parent = target.rsplit("/", 1)[0]
    subprocess.run(["ssh", *ssh_options, host, "mkdir", "-p", parent], check=True)
    subprocess.run(
        ["scp", *ssh_options, str(path), f"{host}:{temporary}"], check=True
    )
    check = subprocess.check_output(
        ["ssh", *ssh_options, host, "sha256sum", temporary], text=True
    ).split()[0]
    size = int(
        subprocess.check_output(
            ["ssh", *ssh_options, host, "stat", "-c", "%s", temporary],
            text=True,
        )
    )
    if check != row["sha256"] or size != row["bytes"]:
        raise RuntimeError(f"remote verification failed: {path}")
    subprocess.run(["ssh", *ssh_options, host, "mv", temporary, target], check=True)
    return target


def still_safe(
    row,
    snapshot_dirs,
    roots,
    min_age,
    catalog,
    digest_cache=None,
    reference_cache=None,
):
    current, _ = scan(
        snapshot_dirs,
        roots,
        min_age,
        [catalog],
        digest_cache=digest_cache,
        reference_cache=reference_cache,
    )
    return any(
        x["path"] == row["path"]
        and x["safe"]
        and x["sha256"] == row["sha256"]
        and x["mtime_ns"] == row["mtime_ns"]
        for x in current
    )


def restore(args, catalog, ssh_options=()):
    entry = catalog.get("objects", {}).get(args.restore)
    if not entry:
        raise SystemExit("hash is absent from local archive catalog")
    target = (
        Path(args.restore_to) if args.restore_to else Path(entry["original_paths"][0])
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) == args.restore:
            print(
                json.dumps(
                    {
                        "status": "already_present",
                        "path": str(target),
                        "sha256": args.restore,
                    }
                )
            )
            return
        raise SystemExit(f"refusing to overwrite existing file: {target}")
    tmp = target.with_suffix(".archive-fetch.tmp")
    subprocess.run(
        ["scp", *ssh_options, f"{args.host}:{entry['remote_path']}", str(tmp)],
        check=True,
    )
    if tmp.stat().st_size != entry["bytes"] or digest(tmp) != args.restore:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("restored bytes failed catalog verification")
    os.replace(tmp, target)
    print(
        json.dumps(
            {
                "status": "restored",
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": args.restore,
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot-dir", type=Path, action="append", default=[])
    p.add_argument("--reference-root", type=Path, action="append", default=[])
    p.add_argument(
        "--catalog", type=Path, default=ROOT / "runs/metal-archive/catalog.json"
    )
    p.add_argument("--host", default="hbox")
    p.add_argument("--remote-root", default="/tank/chreatures/archives/metal-snapshots")
    p.add_argument("--minimum-age-hours", type=float, default=1.0)
    p.add_argument("--apply", action="store_true")
    p.add_argument(
        "--delete-local",
        action="store_true",
        help="after verified archive, re-scan references and remove safe originals",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--restore", metavar="SHA256")
    p.add_argument("--restore-to", type=Path)
    args = p.parse_args()
    if args.delete_local and not args.apply:
        p.error("--delete-local requires --apply")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.host):
        p.error("--host contains unsafe characters")
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", args.remote_root):
        p.error("--remote-root must be an absolute safe path")
    if args.restore and (not re.fullmatch(r"[0-9a-f]{64}", args.restore)):
        p.error("--restore requires a lowercase SHA-256")
    catalog = (
        json.loads(args.catalog.read_text())
        if args.catalog.exists()
        else {"schema_version": 1, "remote_root": args.remote_root, "objects": {}}
    )
    if args.restore:
        with ssh_multiplex(args.host) as ssh_options:
            restore(args, catalog, ssh_options)
        return
    dirs = args.snapshot_dir or [ROOT / "runs/metal-terrarium/brain"]
    roots = args.reference_root or [ROOT / "runs"]
    digest_cache = {}
    reference_cache = {}
    rows, summary = scan(
        dirs,
        roots,
        args.minimum_age_hours * 3600,
        [args.catalog],
        digest_cache=digest_cache,
        reference_cache=reference_cache,
    )
    safe = sorted((x for x in rows if x["safe"]), key=lambda x: x["mtime_ns"])
    eligible_count = len(safe)
    eligible_bytes = sum(x["bytes"] for x in safe)
    safe = safe[: args.limit] if args.limit else safe
    report = {
        "mode": "apply" if args.apply else "plan",
        "snapshot_count": len(rows),
        "eligible_count": eligible_count,
        "eligible_bytes": eligible_bytes,
        "selected_count": len(safe),
        "selected_bytes": sum(x["bytes"] for x in safe),
        "protected_count": len(rows) - eligible_count,
        **summary,
        "candidates": safe,
    }
    if not args.apply:
        print(json.dumps(report, indent=2))
        return
    archived = []
    with ssh_multiplex(args.host) as ssh_options:
        for row in safe:
            remote = archive_one(
                row, args.host, args.remote_root, ssh_options=ssh_options
            )
            entry = catalog["objects"].setdefault(
                row["sha256"],
                {
                    "bytes": row["bytes"],
                    "remote_path": remote,
                    "original_paths": [],
                    "archived_at": time.time(),
                },
            )
            if row["path"] not in entry["original_paths"]:
                entry["original_paths"].append(row["path"])
            entry["verified_sha256"] = row["sha256"]
            entry["verified_bytes"] = row["bytes"]
            atomic_json(args.catalog, catalog)
            deleted = False
            if args.delete_local:
                if not still_safe(
                    row,
                    dirs,
                    roots,
                    args.minimum_age_hours * 3600,
                    args.catalog,
                    digest_cache=digest_cache,
                    reference_cache=reference_cache,
                ):
                    raise RuntimeError(
                        f"snapshot became referenced/newest before deletion: {row['path']}"
                    )
                path = Path(row["path"])
                if digest(path) != row["sha256"]:
                    raise RuntimeError(f"snapshot changed before deletion: {path}")
                path.unlink()
                deleted = True
            archived.append(
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "remote_path": remote,
                    "deleted": deleted,
                }
            )
    report["archived"] = archived
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
