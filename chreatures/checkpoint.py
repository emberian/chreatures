"""Shared deterministic encoding for authenticated project state."""

from __future__ import annotations

import json
from typing import Any


def canonical(value: Any) -> bytes:
    """Encode finite JSON state with stable key and separator ordering."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
