import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_metal_snapshots", ROOT / "scripts/archive_metal_snapshots.py"
)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def test_digest_cache_invalidates_and_new_reference_protects(tmp_path, monkeypatch):
    snapshots = tmp_path / "snapshots"
    references = tmp_path / "references"
    snapshots.mkdir()
    references.mkdir()
    world = "01234567-89ab-cdef-0123-456789abcdef"
    older = snapshots / f"world-{world}-1.npz"
    newest = snapshots / f"world-{world}-2.npz"
    older.write_bytes(b"old bytes")
    newest.write_bytes(b"newest")
    reference = references / "checkpoint.json"
    reference.write_text("{}")
    os.utime(older, (1, 1))
    os.utime(newest, (2, 2))
    monkeypatch.setattr(archive.time, "time", lambda: 10_000)

    calls = []
    real_digest = archive.digest

    def counted(path):
        calls.append(Path(path))
        return real_digest(path)

    monkeypatch.setattr(archive, "digest", counted)
    cache = {}
    reference_cache = {}
    rows, _ = archive.scan(
        [snapshots],
        [references],
        60,
        digest_cache=cache,
        reference_cache=reference_cache,
    )
    first_hash = next(row["sha256"] for row in rows if row["path"] == str(older.resolve()))
    assert len(calls) == 2

    archive.scan(
        [snapshots],
        [references],
        60,
        digest_cache=cache,
        reference_cache=reference_cache,
    )
    assert len(calls) == 2

    older.write_bytes(b"changed bytes")
    os.utime(older, (1, 1))
    rows, _ = archive.scan(
        [snapshots],
        [references],
        60,
        digest_cache=cache,
        reference_cache=reference_cache,
    )
    changed = next(row for row in rows if row["path"] == str(older.resolve()))
    assert changed["sha256"] != first_hash
    assert len(calls) == 3

    reference.write_text('{"sha256":"' + changed["sha256"] + '"}')
    rows, _ = archive.scan(
        [snapshots],
        [references],
        60,
        digest_cache=cache,
        reference_cache=reference_cache,
    )
    protected = next(row for row in rows if row["path"] == str(older.resolve()))
    assert protected["safe"] is False
    assert protected["reason"] == "referenced"
    assert len(calls) == 3

    reference.unlink()
    reference.write_text("{}")
    rows, _ = archive.scan(
        [snapshots],
        [references],
        60,
        digest_cache=cache,
        reference_cache=reference_cache,
    )
    replaced = next(row for row in rows if row["path"] == str(older.resolve()))
    assert replaced["safe"] is True
    assert replaced["reason"] is None

    real_read_bytes = Path.read_bytes

    def unreadable(path):
        if path == reference:
            raise PermissionError("checkpoint is unreadable")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(PermissionError, match="checkpoint is unreadable"):
        archive.scan(
            [snapshots],
            [references],
            60,
            digest_cache={},
            reference_cache={},
        )
