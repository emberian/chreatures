#!/usr/bin/env python3
"""Schedule one gentle, body-agnostic Living Reef visitor performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen


FORMAT = "chreatures-living-reef-visitor-performance-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material", default="reserve-fruit")
    parser.add_argument(
        "--offer-position", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="Fixed caregiver position derived from static habitat geometry",
    )
    parser.add_argument("--start-in", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-content-sha256", required=True)
    parser.add_argument(
        "--execute", action="store_true",
        help="Post the material and schedule; without this flag only print the plan",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise SystemExit("--url must be HTTP(S) without embedded credentials")
    if args.output.exists():
        raise SystemExit("output must not already exist")
    if not 0 <= args.start_in <= 30:
        raise SystemExit("start delay must be in 0..30 model seconds")
    if not 1 <= args.timeout_seconds <= 600 or not 0.01 <= args.poll_seconds <= 1:
        raise SystemExit("invalid wait configuration")
    if len(args.source_content_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.source_content_sha256
    ):
        raise SystemExit("source content SHA-256 is invalid")


def request_json(base: str, route: str, payload: Any | None = None) -> dict[str, Any]:
    body = None if payload is None else canonical(payload)
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        base.rstrip("/") + route,
        data=body,
        headers=headers,
        method="GET" if body is None else "POST",
    )
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"{route} returned HTTP {response.status}")
        raw = response.read(4 << 20)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{route} returned a non-object")
    return value


def number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is nonfinite")
    return result


def plan(
    state: Mapping[str, Any], material: str, start_in: float,
    offer_position: list[float] | None = None,
) -> dict[str, Any]:
    width = number(state["width"], "width")
    height = number(state["height"], "height")
    depth = number(state["depth"], "depth")
    if min(width, height, depth) <= 0:
        raise ValueError("habitat bounds are invalid")
    # Every point is a declared fraction of habitat bounds. Resident state,
    # names, positions, goals, and sensory values are never consulted.
    point = lambda x, y, z: {
        "x": round(width * x, 8),
        "y": round(height * y, 8),
        "z": round(min(max(depth * z, 0.16), depth), 8),
    }
    if offer_position is None:
        offer_point = point(0.58, 0.44, 0.08)
        coordinate_rule = "fixed fractions of public habitat bounds"
    else:
        x, y, z = (number(value, "offer position") for value in offer_position)
        if not (0 <= x <= width and 0 <= y <= height and 0 <= z <= depth):
            raise ValueError("fixed offer position lies outside habitat bounds")
        offer_point = {"x": x, "y": y, "z": z}
        coordinate_rule = "caller-declared fixed static habitat geometry"
    offer = {"op": "offer_material", "material": material, **offer_point}
    events = [
        {
            "at": 0.50,
            "command": {
                "op": "signal", **point(0.28, 0.32, 0.12),
                "tone": 0, "strength": 0.16,
            },
        },
        {
            "at": 1.75,
            "command": {
                "op": "light", **point(0.70, 0.62, 0.30),
                "intensity": 0.24, "duration": 2.25,
                "color": [0.72, 0.86, 1.0],
            },
        },
        {
            "at": 4.50,
            "command": {
                "op": "signal", **point(0.72, 0.68, 0.12),
                "tone": 2, "strength": 0.13,
            },
        },
        {
            "at": 6.00,
            "command": {
                "op": "light", **point(0.36, 0.72, 0.24),
                "intensity": 0.18, "duration": 1.50,
                "color": [1.0, 0.78, 0.58],
            },
        },
    ]
    schedule = {
        "name": "Living Reef: light across water",
        "duration": 8.0,
        "events": events,
        "start_in": float(start_in),
    }
    return {
        "coordinate_rule": coordinate_rule,
        "resident_state_consulted": False,
        "offer": offer,
        "schedule": schedule,
    }


def public_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "id", "motif_id", "name", "duration", "event_count", "start_time",
            "status", "failure", "delivered", "events",
        )
    }


def public_offer_response(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("ok") is not True or not isinstance(value.get("result"), Mapping):
        raise RuntimeError("material offering response is malformed")
    result = value["result"]
    return {
        key: result.get(key)
        for key in (
            "format", "choice", "position", "fraction", "pools",
            "moved_resources", "remaining_source_resources", "outside_boundary",
            "offer_count",
        )
    }


def main() -> int:
    args = arguments()
    validate(args)
    initial = request_json(args.url, "/api/state")
    if initial.get("error") or initial.get("paused"):
        raise SystemExit("habitat must be healthy and advancing")
    performance = plan(initial, args.material, args.start_in, args.offer_position)
    if not args.execute:
        print(json.dumps(performance, indent=2, sort_keys=True))
        return 0
    supplies = initial.get("visitor_materials")
    choice = supplies.get("choices", {}).get(args.material) if isinstance(supplies, dict) else None
    if not isinstance(choice, dict) or not choice.get("available"):
        raise SystemExit("requested finite material offering is unavailable")

    offered_at = {
        "tick": int(initial["tick"]), "model_time": number(initial["time"], "model time"),
    }
    source_identity = {
        "revision": args.source_revision,
        "content_sha256": args.source_content_sha256,
        "graph_sha256": initial.get("anatomy", {}).get("sha256"),
        "resident_artifact_sha256": initial.get(
            "resident_controller", {},
        ).get("artifact_sha256"),
        "biosphere_config_sha256": (
            initial.get("biosphere") or {}
        ).get("config_sha256"),
    }
    offer_result = public_offer_response(request_json(
        args.url, "/api/command", performance["offer"],
    ))
    atomic_json(args.output, {
        "format": FORMAT,
        "status": "material-offered; schedule response pending",
        "source": source_identity,
        "performance": performance,
        "execution": {"offered_at": offered_at, "offer_result": offer_result},
    })
    schedule_result = request_json(
        args.url, "/api/visitor/schedules", performance["schedule"],
    )
    schedule_id = schedule_result.get("id")
    if not isinstance(schedule_id, str):
        raise RuntimeError("visitor schedule response lacks an identifier")
    atomic_json(args.output, {
        "format": FORMAT,
        "status": "scheduled; completion pending",
        "source": source_identity,
        "performance": performance,
        "execution": {
            "offered_at": offered_at, "offer_result": offer_result,
            "schedule": public_schedule(schedule_result),
        },
    })

    deadline = time.monotonic() + args.timeout_seconds
    final_schedule = None
    while time.monotonic() < deadline:
        visitor = request_json(args.url, "/api/visitor")
        final_schedule = next(
            (row for row in visitor.get("queue", []) if row.get("id") == schedule_id),
            None,
        )
        if final_schedule and final_schedule.get("status") in {
            "completed", "failed", "cancelled",
        }:
            break
        time.sleep(args.poll_seconds)
    if not final_schedule or final_schedule.get("status") != "completed":
        raise RuntimeError(f"visitor performance did not complete: {final_schedule}")
    final_state = request_json(args.url, "/api/state")

    result = {
        "format": FORMAT,
        "status": "executed ordinary external physical events; no behavioral interpretation",
        "source": source_identity,
        "performance": performance,
        "execution": {
            "offered_at": offered_at,
            "offer_result": offer_result,
            "schedule": public_schedule(schedule_result),
            "completion": public_schedule(final_schedule),
            "completed_tick": int(final_state["tick"]),
            "completed_model_time": number(final_state["time"], "final model time"),
        },
        "causal_boundary": {
            "resident_ids_in_commands": False,
            "resident_positions_consulted": False,
            "object_ids_in_commands": False,
            "operations": ["offer_material", "signal", "light"],
            "material_accounting": "finite visitor supply transferred through shared chemistry",
            "dispatch": "authoritative model-time visitor queue",
        },
        "limitations": [
            "The performance exposes ordinary physical light, acoustics, and material; it does not test or imply learning.",
            "Subsequent creature encounters are contingent physical events, not targeted or scripted outcomes.",
        ],
    }
    result["content_sha256"] = sha256_bytes(canonical(result))
    atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output), "content_sha256": result["content_sha256"],
        "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
