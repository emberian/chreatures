"""Shared deterministic encoding for authenticated project state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    """Encode finite JSON state with stable key and separator ordering."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def write_envelope(path: str | Path, state: Any, *, format: str) -> str:
    """Atomically replace a local artifact after flushing its checksummed data."""
    digest = hashlib.sha256(canonical(state)).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical({"format": format, "sha256": digest, "state": state}))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return digest
