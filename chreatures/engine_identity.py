"""Pin a resident's local implementation before its first distributed mutation."""

from __future__ import annotations

import hashlib
import importlib
import platform
from pathlib import Path
from typing import Any

from .checkpoint import canonical


def current_engine_identity() -> dict[str, Any]:
    """Describe exact source and loaded native artifacts without recording paths.

    This is a cold startup/restore operation. Live ticks and checkpoint writes
    retain the birth identity even while the development checkout changes.
    Deployment should use an immutable source directory and extension files.
    """
    package = Path(__file__).resolve().parent
    sources = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    native = {}
    for name in ("_world_kernels", "_cognitive_core"):
        module = importlib.import_module(name)
        native[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    dependencies = {
        name: str(importlib.import_module(name).__version__)
        for name in ("numpy", "mujoco")
    }
    value = {
        "format": "chreatures-local-engine-identity-v1",
        "sources": sources,
        "native": native,
        "dependencies": dependencies,
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    value["sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return value
