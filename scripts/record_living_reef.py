#!/usr/bin/env python3
"""Record a compact, public, headless Living Reef episode from `/api/state`."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "chreatures-living-reef-public-recording-v2"
V4_ACTION_NAMES = (
    "thrust", "yaw", "gaze_pitch", "posture", "grip",
    "signal_low", "signal_mid", "signal_high", "eat", "release", "secrete",
    "allocate",
)
V4_PHYSIOLOGY_NAMES = (
    "energy", "gut", "fatigue", "speed", "turn", "neural_support",
    "structural_integrity", "development_fraction", "gland_fill", "brood_fill",
    "reproductive_maturity", "exchange_load",
)
BODY_STATE_FIELDS = {
    "energy": ("body", "energy", "native reserve"),
    "gut": ("body", "gut", "native fill"),
    "fatigue": ("body", "fatigue", "native load"),
    "speed": ("body", "speed", "m/s unsigned world speed"),
    "angular_velocity_z": ("body", "angular_velocity", "rad/s world z"),
    "neural_support": ("neural", "support", "native support"),
    "structural_integrity": ("body", "structural_integrity", "fraction"),
    "development_fraction": ("body", "development_fraction", "fraction"),
    "gland_fill": ("body", "gland_fill", "native fill"),
    "brood_fill": ("body", "brood_fill", "native fill"),
    "reproductive_maturity": ("body", "reproductive_maturity", "fraction"),
    "exchange_load": ("body", "exchange_load", "native load"),
}
PERIPHERAL_SHAPE = (8, 32, 4)
FOVEAL_SHAPE = (24, 32, 4)
EVENT_KINDS = (
    "root-material-acquisition",
    "mobile-material-release",
    "colony-material-emission",
    "developmental-growth-committed",
    "developmental-attachment-invalidated",
    "developmental-parts-removed",
    "hatching",
    "goal_episode_completed",
    "signal_emission",
    "contact_begin",
    "contact_end",
    "visitor_material",
    "visitor_stimulus",
)
UNAVAILABLE_EVENT_CAPABILITIES = (
    "material_ingestion",
    "material_secretion",
    "material_allocation",
    "birth",
    "lineage",
    "construction",
    "topology",
    "growth",
    "contact",
)
ZERO_HASH = "0" * 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode())
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Read-only Habitat3D base URL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binding-output", type=Path, required=True,
        help="Private body-index binding receipt outside every public asset tree",
    )
    parser.add_argument(
        "--raw-output", type=Path,
        help="Private gzip JSONL of exact accepted host views",
    )
    parser.add_argument("--resident-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--stride-ticks", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--strict-ticks", action="store_true",
        help="Fail instead of accepting a later real tick when polling skips a target",
    )
    parser.add_argument("--world-source-revision", required=True)
    parser.add_argument("--world-source-content-sha256", required=True)
    parser.add_argument(
        "--capture-tool-revision", required=True,
        help="Revision containing this separately frozen recorder source",
    )
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--expected-graph-sha256")
    parser.add_argument("--expected-resident-artifact-sha256")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise SystemExit("--url must be an HTTP(S) base URL without credentials")
    if args.output.exists():
        raise SystemExit("output must not already exist")
    if args.binding_output.exists():
        raise SystemExit("binding output must not already exist")
    binding_path = args.binding_output.resolve()
    for public_root in (ROOT / "site", ROOT / "docs" / "assets"):
        try:
            binding_path.relative_to(public_root.resolve())
        except ValueError:
            continue
        raise SystemExit("binding output must be outside public asset trees")
    if args.raw_output is not None and args.raw_output.exists():
        raise SystemExit("raw output must not already exist")
    outputs = [args.output.resolve(), args.binding_output.resolve()]
    if args.raw_output is not None:
        outputs.append(args.raw_output.resolve())
    if len(outputs) != len(set(outputs)):
        raise SystemExit("recording, binding and raw outputs must be distinct")
    if not 0 <= args.resident_index < 64:
        raise SystemExit("resident index must be in 0..63")
    if not 2 <= args.frames <= 10_000 or not 1 <= args.stride_ticks <= 1_000:
        raise SystemExit("invalid frame count or tick stride")
    if not 0.001 <= args.poll_seconds <= 1.0 or not 1 <= args.timeout_seconds <= 86_400:
        raise SystemExit("invalid polling interval or timeout")
    if not args.world_source_revision.strip() or not args.capture_tool_revision.strip():
        raise SystemExit("world and capture-tool revisions must be nonempty")
    for label, value in (
        ("world source content", args.world_source_content_sha256),
        ("profile", args.profile_sha256),
    ):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise SystemExit(f"{label} SHA-256 is invalid")


def fetch_state(base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/state"
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"state endpoint returned HTTP {response.status}")
        if "application/json" not in response.headers.get("content-type", ""):
            raise RuntimeError("state endpoint did not return JSON")
        payload = response.read(32 << 20)
        if len(payload) >= 32 << 20:
            raise RuntimeError("state response exceeds 32 MiB")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("state endpoint returned a non-object")
    return value


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is nonfinite")
    return result


def vector(value: Any, size: int, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {size}-vector")
    return array.astype(float).tolist()


def retina_u8(value: Any, shape: tuple[int, ...], label: str) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} has invalid shape or values")
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{label} lies outside [0,1]")
    packed = np.rint(array * 255).astype(np.uint8).tobytes()
    return {
        "encoding": "base64-u8-linear-0-1", "shape": list(shape),
        "data": base64.b64encode(packed).decode(),
    }


def float32_blob(value: Any, size: int, label: str) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {size}-vector")
    packed = np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes()
    return {
        "encoding": "base64-little-endian-float32", "shape": [size],
        "data": base64.b64encode(packed).decode(),
    }


def shape_geometry(shape: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": str(shape["type"]),
        "size": vector(shape["size"], len(shape["size"]), "shape size"),
        "position": vector(shape["position"], 3, "shape position"),
        "quaternion": vector(shape["quaternion"], 4, "shape quaternion"),
        "color": str(shape["color"]),
    }


def body_geometry(body: Mapping[str, Any], public_index: int) -> dict[str, Any]:
    result = {
        "body": public_index,
        "position": [finite(body[key], f"body {key}") for key in ("x", "y", "z")],
        "quaternion": vector(body["quaternion"], 4, "body quaternion"),
        "speed": finite(body["speed"], "body speed"),
        "angular_velocity": finite(body["angular_velocity"], "body angular velocity"),
    }
    return result


def object_geometry(entity: Mapping[str, Any], public_index: int) -> dict[str, Any]:
    return quantize_geometry({
        "entity": public_index,
        "shapes": [shape_geometry(shape) for shape in entity.get("shapes", [])],
        "joint": entity.get("joint"),
    })


def quantize_geometry(value: Any) -> Any:
    """Bound display geometry precision while the private raw capture stays exact."""
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, list):
        return [quantize_geometry(item) for item in value]
    if isinstance(value, Mapping):
        return {key: quantize_geometry(item) for key, item in value.items()}
    return value


def hash_fields(value: Any, prefix: str = "") -> dict[str, str]:
    result = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).endswith("sha256") and isinstance(child, str):
                result[path] = child
            else:
                result.update(hash_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(hash_fields(child, f"{prefix}[{index}]"))
    return result


def organism_contract(state: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the published controller boundary without guessing from array sizes."""
    controller = state.get("resident_controller", {})
    contract = controller.get("observation_contract")
    if isinstance(contract, Mapping) and contract.get("format") == "chreatures-organism-interface-v4":
        actions = tuple(contract.get("actions", ()))
        physiology = tuple(contract.get("physiology", ()))
        if actions != V4_ACTION_NAMES or physiology != V4_PHYSIOLOGY_NAMES:
            raise ValueError("host v4 organism interface differs from the exact public contract")
        return {"format": contract["format"], "actions": actions, "physiology": physiology}
    raise ValueError("host does not publish the current v4 organism interface")


def body_state_values(
    body: Mapping[str, Any], neural: Mapping[str, Any], label: str,
) -> dict[str, float]:
    result = {}
    for name, (owner, key, _unit) in BODY_STATE_FIELDS.items():
        source = neural if owner == "neural" else body
        result[name] = finite(source[key], f"{label} body state {name}")
    return result


def optional_summary(value: Any, label: str) -> dict[str, Any]:
    """Copy a bounded public diagnostic or mark the stream unavailable."""
    if value is None:
        return {"status": "unavailable", "reason": f"host did not publish {label}"}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object when published")
    encoded = canonical_bytes(value)
    if len(encoded) > 64 << 10:
        raise ValueError(f"{label} exceeds the 64 KiB public summary bound")
    return {"status": "recorded", "value": json.loads(encoded)}


def event_capabilities(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    stream = state.get("evidence_event_stream")
    if stream is not None and not isinstance(stream, Mapping):
        raise ValueError("evidence_event_stream must be an object")
    declared = stream.get("capabilities") if stream is not None else None
    if declared is not None and not isinstance(declared, Mapping):
        raise ValueError("evidence event capabilities must be an object")
    stream_present = isinstance(state.get("evidence_events"), list)
    result = {}
    for kind in (*EVENT_KINDS, *UNAVAILABLE_EVENT_CAPABILITIES):
        supported = bool(declared.get(kind, False)) if declared is not None else False
        if stream_present and supported:
            result[kind] = {
                "status": "recorded",
                "source_path": f"api/state.evidence_events[kind={kind}]",
            }
        else:
            result[kind] = {
                "status": "unavailable",
                "source_path": f"api/state.evidence_events[kind={kind}]",
                "reason": (
                    "host did not declare an authenticated receipt stream for this event kind"
                ),
            }
    return result


def public_blob_ref(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    allowed = {"role", "uri", "sha256", "bytes", "media_type", "verification"}
    if not set(value).issubset(allowed):
        raise ValueError(f"{label} has unsupported fields")
    role = str(value.get("role", ""))
    digest = str(value.get("sha256", ""))
    if not role or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} has invalid role or SHA-256")
    if value.get("uri") != f"urn:sha256:{digest}":
        raise ValueError(f"{label} URI differs from its SHA-256")
    result = {key: value[key] for key in allowed if key in value}
    if "bytes" in result and (
        isinstance(result["bytes"], bool)
        or not isinstance(result["bytes"], int)
        or result["bytes"] < 0
    ):
        raise ValueError(f"{label} bytes is invalid")
    return result


def public_event_value(
    value: Any,
    body_indices: Mapping[str, int],
    entity_indices: Mapping[str, int],
) -> Any:
    """Replace actor identities inside metadata while preserving receipt meaning."""
    if isinstance(value, str):
        if value in body_indices:
            return {"public_body": body_indices[value]}
        if value in entity_indices:
            return {"public_entity": entity_indices[value]}
        return value
    if isinstance(value, list):
        return [public_event_value(item, body_indices, entity_indices) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            public_key = (
                f"body:{body_indices[str(key)]}"
                if str(key) in body_indices
                else (
                    f"entity:{entity_indices[str(key)]}"
                    if str(key) in entity_indices
                    else str(key)
                )
            )
            result[public_key] = public_event_value(
                child, body_indices, entity_indices
            )
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError("evidence event metadata contains a non-JSON value")


def public_event_source(
    value: Mapping[str, Any],
    body_indices: Mapping[str, int],
    entity_indices: Mapping[str, int],
) -> dict[str, Any]:
    """Sanitize mechanism provenance without publishing opaque runtime IDs."""
    source = dict(value)
    for key in ("signal_id", "offer_id"):
        opaque = source.pop(key, None)
        if opaque is not None:
            if not isinstance(opaque, str) or not opaque:
                raise ValueError(f"evidence event source {key} is invalid")
            source[f"{key}_sha256"] = sha256_bytes(opaque.encode())
    result = public_event_value(source, body_indices, entity_indices)
    if not isinstance(result, dict):
        raise ValueError("evidence event source must remain an object")
    return result


def evidence_events(
    state: Mapping[str, Any],
    body_indices: Mapping[str, int],
    entity_indices: dict[str, int],
    *,
    after_sequence: int,
) -> list[dict[str, Any]]:
    raw = state.get("evidence_events")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("evidence_events must be an array")
    result = []
    for index, source in enumerate(raw):
        if not isinstance(source, Mapping):
            raise ValueError(f"evidence event {index} must be an object")
        event_id = str(source.get("event_id", ""))
        kind = str(source.get("kind", ""))
        sequence = source.get("sequence")
        tick = source.get("tick")
        if not event_id or kind not in EVENT_KINDS:
            raise ValueError(f"evidence event {index} has invalid identity or kind")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or isinstance(tick, bool)
            or not isinstance(tick, int)
            or tick < 0
        ):
            raise ValueError(f"evidence event {event_id} has invalid sequence or tick")
        if sequence <= after_sequence:
            continue
        actors = source.get("actors", {})
        if not isinstance(actors, Mapping) or set(actors) != {"bodies", "entities"}:
            raise ValueError(f"evidence event {event_id} has invalid actors")
        body_ids = actors["bodies"]
        entity_ids = actors["entities"]
        if not isinstance(body_ids, list) or not isinstance(entity_ids, list):
            raise ValueError(f"evidence event {event_id} actors must be arrays")
        if any(str(value) not in body_indices for value in body_ids):
            raise ValueError(f"evidence event {event_id} names an unknown body")
        for value in entity_ids:
            entity_id = str(value)
            if entity_id not in entity_indices:
                # A short-lived constructed or released entity may be committed
                # between sampled scene frames. Preserve its event identity even
                # when no geometry frame was observed.
                entity_indices[entity_id] = len(entity_indices)
        details = source.get("details", {})
        if not isinstance(details, Mapping):
            raise ValueError(f"evidence event {event_id} details must be an object")
        body_roles = details.get("body_roles", {})
        entity_roles = details.get("entity_roles", {})
        if not isinstance(body_roles, Mapping) or not isinstance(entity_roles, Mapping):
            raise ValueError(f"evidence event {event_id} actor roles must be objects")
        if not set(map(str, body_roles)).issubset(set(map(str, body_ids))) or not set(
            map(str, entity_roles)
        ).issubset(set(map(str, entity_ids))):
            raise ValueError(f"evidence event {event_id} roles name absent actors")
        if any(not str(role).strip() for role in [*body_roles.values(), *entity_roles.values()]):
            raise ValueError(f"evidence event {event_id} has an empty actor role")
        raw_details = {
            key: value
            for key, value in details.items()
            if key not in {"body_roles", "entity_roles"}
        }
        if kind == "goal_episode_completed" and isinstance(
            raw_details.get("goal"), Mapping
        ):
            goal = raw_details["goal"]
            raw_details["goal"] = {
                key: goal[key]
                for key in (
                    "valid", "changed", "recorded_tick", "recorded_time",
                    "remaining_ticks",
                )
                if key in goal
            }
        if kind == "signal_emission" and isinstance(
            raw_details.get("signal"), Mapping
        ):
            signal = raw_details["signal"]
            raw_details["signal"] = {
                key: signal[key]
                for key in ("x", "y", "z", "tone", "strength", "remaining")
                if key in signal
            }
        mechanism_details = public_event_value(
            raw_details, body_indices, entity_indices
        )
        encoded_details = canonical_bytes(mechanism_details)
        if len(encoded_details) > 24 << 10:
            raise ValueError(f"evidence event {event_id} details exceed 24 KiB")
        quantities = source.get("quantities", [])
        if not isinstance(quantities, list):
            raise ValueError(f"evidence event {event_id} quantities must be an array")
        public_quantities = []
        for quantity_index, quantity in enumerate(quantities):
            if not isinstance(quantity, Mapping) or set(quantity) != {"name", "value", "unit"}:
                raise ValueError(
                    f"evidence event {event_id} quantity {quantity_index} has invalid fields"
                )
            name = str(quantity["name"])
            unit = str(quantity["unit"])
            if not name or not unit:
                raise ValueError(f"evidence event {event_id} has an unnamed quantity")
            public_quantities.append(
                {"name": name, "value": finite(quantity["value"], name), "unit": unit}
            )
        blobs = source.get("blob_refs", [])
        if not isinstance(blobs, list):
            raise ValueError(f"evidence event {event_id} blob_refs must be an array")
        source_receipt = source.get("source", {})
        if not isinstance(source_receipt, Mapping):
            raise ValueError(f"evidence event {event_id} source must be an object")
        public_source = public_event_source(
            source_receipt, body_indices, entity_indices
        )
        previous_sha256 = str(source.get("previous_sha256", ""))
        event_sha256 = str(source.get("sha256", ""))
        for label, digest in (
            ("previous_sha256", previous_sha256),
            ("sha256", event_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"evidence event {event_id} has invalid {label}")
        authenticated = dict(source)
        authenticated.pop("sha256", None)
        if sha256_bytes(canonical_bytes(authenticated)) != event_sha256:
            raise ValueError(f"evidence event {event_id} content hash differs")
        result.append(
            {
                "event_id": f"event:{event_sha256}",
                "sequence": sequence,
                "tick": tick,
                "model_time": finite(
                    source.get("model_time"), f"evidence event {event_id} time"
                ),
                "kind": kind,
                "actors": {
                    "bodies": [body_indices[str(value)] for value in body_ids],
                    "entities": [entity_indices[str(value)] for value in entity_ids],
                    "body_roles": {
                        str(body_indices[str(key)]): str(role)
                        for key, role in body_roles.items()
                    },
                    "entity_roles": {
                        str(entity_indices[str(key)]): str(role)
                        for key, role in entity_roles.items()
                    },
                },
                "quantities": public_quantities,
                "details": json.loads(encoded_details),
                "source": public_source,
                "blob_refs": [
                    public_blob_ref(value, f"evidence event {event_id} blob")
                    for value in blobs
                ],
                "source_receipt": {
                    "event_id_sha256": sha256_bytes(event_id.encode()),
                    "previous_sha256": previous_sha256,
                    "sha256": event_sha256,
                },
            }
        )
    result.sort(key=lambda value: value["sequence"])
    return result


def latest_event_sequence(state: Mapping[str, Any]) -> int:
    stream = state.get("evidence_event_stream")
    if stream is not None and not isinstance(stream, Mapping):
        raise ValueError("evidence_event_stream must be an object")
    declared = stream.get("last_sequence") if stream is not None else None
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
            raise ValueError("evidence_event_sequence must be a nonnegative integer")
        return declared
    raw = state.get("evidence_events")
    if not isinstance(raw, list):
        return 0
    sequences = [value.get("sequence", 0) for value in raw if isinstance(value, Mapping)]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in sequences):
        raise ValueError("evidence event sequence is invalid")
    return max(sequences, default=0)


def event_stream_head(state: Mapping[str, Any]) -> str:
    stream = state.get("evidence_event_stream")
    if stream is None:
        return ZERO_HASH
    if not isinstance(stream, Mapping):
        raise ValueError("evidence_event_stream must be an object")
    if stream.get("format") != "chreatures-committed-evidence-events-v1":
        raise ValueError("unsupported evidence event stream")
    digest = str(stream.get("head_sha256", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("evidence event stream has invalid head")
    for key in ("first_sequence", "last_sequence", "retired_events", "capacity"):
        value = stream.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"evidence event stream has invalid {key}")
    return digest


def event_stream_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    stream = state.get("evidence_event_stream")
    if stream is None:
        return {
            "status": "unavailable",
            "reason": "host did not publish a committed evidence-event stream",
        }
    event_stream_head(state)
    world_id = stream.get("world_id")
    if not isinstance(world_id, str) or not world_id:
        raise ValueError("evidence event stream lacks a world identity")
    return {
        "status": "recorded",
        "format": stream["format"],
        "world_id": world_id,
        "first_sequence": int(stream["first_sequence"]),
        "last_sequence": int(stream["last_sequence"]),
        "retired_events": int(stream["retired_events"]),
        "head_sha256": str(stream["head_sha256"]),
        "capacity": int(stream["capacity"]),
    }


def identity(state: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    anatomy = state.get("anatomy", {})
    controller = state.get("resident_controller", {})
    graph = str(anatomy.get("sha256", ""))
    artifact = str(controller.get("artifact_sha256", ""))
    engine = state.get("engine_identity")
    if not isinstance(engine, Mapping) or len(str(engine.get("sha256", ""))) != 64:
        raise ValueError("host view lacks its pinned engine identity")
    compact_engine = {
        key: engine.get(key)
        for key in (
            "format", "sha256", "native", "dependencies", "python", "system", "machine",
        )
    }
    if len(graph) != 64:
        raise ValueError("host view lacks a graph SHA-256")
    if args.expected_graph_sha256 and graph != args.expected_graph_sha256:
        raise ValueError("host graph differs from expected identity")
    if args.expected_resident_artifact_sha256 and artifact != args.expected_resident_artifact_sha256:
        raise ValueError("resident artifact differs from expected identity")
    sensorium = dict(state.get("sensorium", {}))
    sensorium.pop("retina_pose", None)
    controller_public = {
        key: controller.get(key)
        for key in ("format", "artifact_sha256", "file_sha256", "mode", "execution")
        if key in controller
    }
    controller_public["observation_contract"] = controller.get("observation_contract")
    return {
        "world_source_revision": args.world_source_revision,
        "world_source_content_sha256": args.world_source_content_sha256,
        "world_source_content_semantics": (
            "caller-pinned immutable world source archive or its declared content identity"
        ),
        "capture_tool": {
            "name": "record_living_reef.py",
            "revision": args.capture_tool_revision,
            "file_sha256": sha256(Path(__file__).resolve()),
        },
        "engine_identity": compact_engine,
        "physical_profile_sha256": args.profile_sha256,
        "graph_sha256": graph,
        "resident_artifact_sha256": artifact or None,
        "host_hash_fields": hash_fields({
            "anatomy": anatomy, "resident_controller": controller_public,
            "biosphere": state.get("biosphere"), "sensorium": sensorium,
        }),
        "anatomy": {
            key: anatomy.get(key)
            for key in ("dataset", "scope", "neurons", "connections", "inputs", "readouts")
        },
        "sensorium": sensorium,
        "resident_controller": controller_public,
    }


def resident_detail(
    *,
    body_id: str,
    public_index: int,
    body: Mapping[str, Any],
    senses: Mapping[str, Any],
    neural: Mapping[str, Any],
    cognition: Mapping[str, Any],
    retina_pose: Mapping[str, Any],
    outcome: Mapping[str, Any],
    action_names: tuple[str, ...],
) -> dict[str, Any]:
    action = cognition.get("executed_action", {})
    proposal = cognition.get("sampled_proposal", {})
    if set(action) != set(action_names) or set(proposal) != set(action_names):
        raise ValueError(f"resident {public_index} action contract differs")
    goal = cognition.get("goal")
    if not isinstance(goal, Mapping):
        raise ValueError(f"resident {public_index} goal summary is unavailable")
    consequence = cognition.get("consequence_refinement")
    forecast = cognition.get("sensory_forecast")
    if consequence is None and forecast is None:
        refinement_summary = optional_summary(None, "consequence refinement")
        forecast_summary = optional_summary(None, "sensory forecast")
    elif not isinstance(consequence, Mapping) or not isinstance(forecast, Mapping):
        raise ValueError(f"resident {public_index} has a partial forecast report")
    else:
        candidate_count = len(consequence.get("candidate_scores", []))
        if candidate_count < 1 or any(
            len(report.get(key, [])) != candidate_count
            for report, keys in (
                (consequence, ("candidate_scores", "candidate_out_of_domain")),
                (
                    forecast,
                    (
                        "candidate_progress",
                        "candidate_disagreement",
                        "candidate_forecast_invalid",
                        "candidate_logit_tilt",
                    ),
                ),
            )
            for key in keys
        ):
            raise ValueError(f"resident {public_index} candidate reports differ in length")
        physiology_predictions = forecast.get("candidate_physiology", [])
        if len(physiology_predictions) != candidate_count or any(
            np.asarray(row).shape != (12,) for row in physiology_predictions
        ):
            raise ValueError(
                f"resident {public_index} forecast physiology differs in length"
            )
        refinement_summary = {
            "status": "recorded",
            "candidate_scores": [
                finite(value, "candidate consequence score")
                for value in consequence["candidate_scores"]
            ],
            "candidate_out_of_domain": [
                bool(value) for value in consequence["candidate_out_of_domain"]
            ],
            "selected_candidate": int(consequence["selected_candidate"]),
            "selected_private_correction": vector(
                consequence["selected_private_correction"],
                3,
                "selected private correction",
            ),
            "completed_private_updates_before_action": int(
                consequence["completed_private_updates_before_action"]
            ),
        }
        forecast_summary = {
            "status": "recorded",
            **{
                key: [
                    finite(value, f"sensory forecast {key}")
                    for value in forecast[key]
                ]
                for key in (
                    "candidate_progress",
                    "candidate_disagreement",
                    "candidate_logit_tilt",
                )
            },
            "candidate_forecast_invalid": [
                bool(value) for value in forecast["candidate_forecast_invalid"]
            ],
            "empirical_goal_error_scale": finite(
                forecast["empirical_goal_error_scale"], "empirical goal error scale"
            ),
            "horizon_ticks": int(forecast["horizon_ticks"]),
            "horizon_seconds": finite(
                forecast["horizon_seconds"], "sensory forecast horizon"
            ),
            "candidate_physiology": [
                vector(row, 12, "sensory forecast candidate physiology")
                for row in physiology_predictions
            ],
            "proposal_suffix": str(forecast["proposal_suffix"]),
            "meaning": str(forecast["meaning"]),
        }
    pose = {
        key: vector(retina_pose[key], 3, f"resident {public_index} retina pose {key}")
        for key in ("origin", "forward", "up", "right")
    }
    learning = cognition.get("personal_goal_learning")
    public_learning = optional_summary(learning, "personal goal learning summary")
    sequence_summary = cognition.get("sequence_memory")
    contextual_summary = cognition.get("contextual_memory")
    return {
        "body": public_index,
        "retina": {
            "peripheral": retina_u8(
                senses["peripheral"], PERIPHERAL_SHAPE, "peripheral retina"
            ),
            "foveal": retina_u8(senses["foveal"], FOVEAL_SHAPE, "foveal retina"),
            "profile": senses["profile"],
        },
        "retina_pose": pose,
        "neural_readouts": float32_blob(
            neural["features"], 384, f"resident {public_index} neural readouts"
        ),
        "neural_summary": {
            key: finite(neural[key], f"resident {public_index} neural {key}")
            for key in ("activity", "support")
        },
        "goal": {
            "valid": bool(goal["valid"]),
            "changed": bool(goal["changed"]),
            "recorded_tick": int(goal["recorded_tick"]),
            "recorded_time": finite(goal["recorded_time"], "goal recorded time"),
            "remaining_ticks": int(goal["remaining_ticks"]),
        },
        "sampled_proposal": {
            key: finite(proposal[key], f"resident {public_index} proposal {key}")
            for key in action_names
        },
        "refinement": refinement_summary,
        "forecast": forecast_summary,
        "committed_action": {
            key: finite(action[key], f"resident {public_index} committed action {key}")
            for key in action_names
        },
        "recorded_body_state": body_state_values(
            body, neural, f"resident {public_index}"
        ),
        "outcome": {
            key: finite(outcome.get(key, 0.0), f"resident {public_index} outcome {key}")
            for key in (
                "nutrition",
                "ingested_mass",
                "effort",
                "contact",
                "mouth_material_contacts",
            )
        },
        "path_sample": {
            "position": [finite(body[key], f"resident {public_index} {key}") for key in ("x", "y", "z")],
            "quaternion": vector(body["quaternion"], 4, "body quaternion"),
        },
        "gaze_sample": pose,
        "memory_summary": {
            "record_count": int(cognition.get("memory_count", 0)),
            "personal_goal_learning": public_learning,
            "sequence": optional_summary(sequence_summary, "sequence memory summary"),
            "contextual": optional_summary(
                contextual_summary, "contextual memory summary"
            ),
        },
        "population_response": optional_summary(
            cognition.get("population_response"), "population response summary"
        ),
    }


def extract_frame(
    state: Mapping[str, Any], selected_id: str, body_indices: dict[str, int],
    entity_indices: dict[str, int], previous: Mapping[str, Any] | None,
    contract: Mapping[str, Any], event_cursor: int,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    bodies = state["bodies"]
    entities = state["entities"]
    by_body = {str(body["id"]): body for body in bodies}
    by_entity = {str(entity["id"]): entity for entity in entities}
    if selected_id not in by_body:
        raise ValueError("selected resident disappeared during recording")
    for body_id in by_body:
        if body_id not in body_indices:
            body_indices[body_id] = len(body_indices)
    for entity_id in by_entity:
        if entity_id not in entity_indices:
            entity_indices[entity_id] = len(entity_indices)
    action_names = tuple(contract["actions"])
    outcomes = state.get("outcomes", {})
    retina_poses = state.get("sensorium", {}).get("retina_pose", {})
    if not isinstance(retina_poses, Mapping):
        raise ValueError("host view lacks resident retina physical poses")
    details = []
    for body_id, body in by_body.items():
        pose = retina_poses.get(body_id)
        if not isinstance(pose, Mapping):
            raise ValueError(f"host view lacks retina pose for body {body_indices[body_id]}")
        details.append(
            resident_detail(
                body_id=body_id,
                public_index=body_indices[body_id],
                body=body,
                senses=state["senses"][body_id]["rich_retina"],
                neural=state["neural"][body_id],
                cognition=state["cognition"][body_id],
                retina_pose=pose,
                outcome=outcomes.get(body_id, {}),
                action_names=action_names,
            )
        )
    details.sort(key=lambda value: value["body"])
    selected_index = body_indices[selected_id]
    selected_detail = next(value for value in details if value["body"] == selected_index)
    frame = {
        "tick": int(state["tick"]),
        "model_time": finite(state["time"], "model time"),
        "sampled_sensory_time": finite(state["sensed_at"], "sampled sensory time"),
        "bodies": [
            body_geometry(body, body_indices[body_id])
            for body_id, body in by_body.items()
        ],
        "resident_details": details,
        "entities": [
            object_geometry(entity, entity_indices[entity_id])
            for entity_id, entity in by_entity.items()
        ],
        "articulations": [
            {
                "body": body_indices[str(item["id"])],
                # Replay uses rendered shapes and transforms. The exact raw
                # capture retains link, joint, and site diagnostics.
                "geoms": quantize_geometry([
                    {
                        key: geom[key]
                        for key in ("type", "size", "position", "quaternion", "color")
                        if key in geom
                    }
                    for geom in item.get("geoms", [])
                ]),
            }
            for item in state.get("articulations", [])
        ],
        "assemblies": [
            {
                **{
                    key: value for key, value in assembly.items()
                    if key not in {"id", "joint_a", "joint_b", "coordinates"}
                },
                "entities": [
                    entity_indices[str(assembly["joint_a"])],
                    entity_indices[str(assembly["joint_b"])],
                ],
                "coordinates": [
                    {
                        "position": finite(
                            assembly["coordinates"][key]["position"],
                            "assembly position",
                        ),
                        "velocity": finite(
                            assembly["coordinates"][key]["velocity"],
                            "assembly velocity",
                        ),
                    }
                    for key in ("joint_a", "joint_b")
                ],
            }
            for assembly in state.get("assemblies", [])
        ],
        "signals": [
            {
                "position": [finite(signal[key], f"signal {key}") for key in ("x", "y", "z")],
                "tone": int(signal["tone"]),
                "strength": finite(signal["strength"], "signal strength"),
                "remaining": finite(signal["remaining"], "signal remaining"),
            }
            for signal in state.get("signals", [])
        ],
        "lights": [
            {
                **({"entity": entity_indices[str(light["entity"])]} if str(
                    light.get("entity", "")
                ) in entity_indices else {}),
                "position": vector(light["position"], 3, "light position"),
                "direction": vector(light["direction"], 3, "light direction"),
                "color": vector(light["color"], 3, "light color"),
                "intensity": finite(light["intensity"], "light intensity"),
                "radius": finite(light["radius"], "light radius"),
                "directional": bool(light.get("directional", False)),
                "ambient_intensity": finite(
                    light.get("ambient_intensity", 0.0),
                    "light ambient intensity",
                ),
            }
            for light in state.get("lights", [])
        ],
        "selected_body": selected_index,
    }
    reasons = []
    if selected_detail["goal"]["changed"]:
        reasons.append("private-goal-commit")
    outcome = selected_detail["outcome"]
    # Ignore native roundoff-scale transport residue in the public index.
    if outcome["ingested_mass"] > 1e-9 or outcome["nutrition"] > 1e-9:
        reasons.append("physical-ingestion")
    if outcome["mouth_material_contacts"] > 0:
        reasons.append("mouth-material-contact")
    if previous is not None:
        old = next(
            value
            for value in previous["resident_details"]
            if value["body"] == selected_index
        )
        if len(frame["signals"]) != len(previous["signals"]):
            reasons.append("signal-field-change")
        if abs(
            selected_detail["recorded_body_state"]["energy"]
            - old["recorded_body_state"]["energy"]
        ) >= 0.01:
            reasons.append("energy-change")
        selected_body = next(
            value for value in frame["bodies"] if value["body"] == selected_index
        )
        if selected_body["speed"] >= 0.1:
            reasons.append("rapid-motion")
    return frame, reasons, evidence_events(
        state, body_indices, entity_indices, after_sequence=event_cursor
    )


def encode_entity_deltas(frames: list[dict[str, Any]]) -> None:
    """Encode full entity keyframe followed by full-object replacements."""
    previous: dict[int, Any] = {}
    for frame_index, frame in enumerate(frames):
        entities = frame["entities"]
        current = {int(entity["entity"]): entity for entity in entities}
        if len(current) != len(entities):
            raise ValueError("duplicate public entity index")
        if frame_index:
            frame["entities"] = {
                "changed": [
                    current[key] for key in sorted(current)
                    if previous.get(key) != current[key]
                ],
                "removed": sorted(set(previous) - set(current)),
            }
        previous = current


def main() -> int:
    args = arguments()
    validate_arguments(args)
    initial = fetch_state(args.url)
    if initial.get("error") or initial.get("paused"):
        raise SystemExit("host must be healthy and advancing for read-only recording")
    bodies = initial.get("bodies", [])
    entities = initial.get("entities", [])
    if args.resident_index >= len(bodies):
        raise SystemExit("selected resident is absent")
    body_ids = [str(body["id"]) for body in bodies]
    body_indices = {body_id: index for index, body_id in enumerate(body_ids)}
    entity_indices = {
        str(entity["id"]): index for index, entity in enumerate(entities)
    }
    selected_id = body_ids[args.resident_index]
    provenance = identity(initial, args)
    contract = organism_contract(initial)
    event_stream_capabilities = event_capabilities(initial)
    event_stream_start = event_stream_metadata(initial)
    event_cursor = latest_event_sequence(initial)
    source_event_head = event_stream_head(initial)
    public_event_head = ZERO_HASH
    event_start_tick = int(initial["tick"])
    start_tick = int(initial["tick"]) + args.stride_ticks
    target_tick = start_tick
    deadline = time.monotonic() + args.timeout_seconds
    frames = []
    events: list[dict[str, Any]] = []
    event_by_id: dict[str, dict[str, Any]] = {}
    geometry_entity_indexes: set[int] = set(entity_indices.values())
    moments = []
    seen_phenomena: set[str] = set()
    previous = None
    target_index = 0
    raw_temporary = None
    raw_stream = None
    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_temporary = args.raw_output.with_name(
            f".{args.raw_output.name}.tmp-{os.getpid()}"
        )
        raw_stream = gzip.open(raw_temporary, "wt", encoding="utf-8")
    try:
        while target_index < args.frames:
            if time.monotonic() >= deadline:
                raise TimeoutError("recording timed out before all requested frames arrived")
            state = fetch_state(args.url)
            if state.get("error"):
                raise RuntimeError(f"host reported an error: {state['error']}")
            tick = int(state["tick"])
            if tick < target_tick:
                time.sleep(args.poll_seconds)
                continue
            if args.strict_ticks and tick != target_tick:
                raise RuntimeError(
                    f"read-only sampler missed requested tick {target_tick}; observed {tick}"
                )
            if identity(state, args) != provenance:
                raise RuntimeError("host identity changed during recording")
            if raw_stream is not None:
                raw_stream.write(json.dumps(
                    state, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ) + "\n")
            if event_capabilities(state) != event_stream_capabilities:
                raise RuntimeError("host event capabilities changed during recording")
            frame, reasons, frame_events = extract_frame(
                state,
                selected_id,
                body_indices,
                entity_indices,
                previous,
                contract,
                event_cursor,
            )
            for event in frame_events:
                if event["tick"] > tick:
                    raise RuntimeError("host published an event from a future tick")
                if event["sequence"] != event_cursor + 1:
                    raise RuntimeError(
                        "authenticated event sequence was truncated between recorded frames"
                    )
                if event["source_receipt"]["previous_sha256"] != source_event_head:
                    raise RuntimeError("authenticated event hash chain changed or has a gap")
                public_event = dict(event)
                public_event["previous_sha256"] = public_event_head
                public_event["sha256"] = sha256_bytes(canonical_bytes(public_event))
                event = public_event
                previous_event = event_by_id.get(event["event_id"])
                if previous_event is not None and previous_event != event:
                    raise RuntimeError("stable event id changed during recording")
                if previous_event is None:
                    events.append(event)
                    event_by_id[event["event_id"]] = event
                event_cursor = event["sequence"]
                source_event_head = event["source_receipt"]["sha256"]
                public_event_head = event["sha256"]
            if latest_event_sequence(state) != event_cursor:
                raise RuntimeError("host event cursor advanced beyond retained event receipts")
            if event_stream_head(state) != source_event_head:
                raise RuntimeError("host event head differs from captured receipt chain")
            frame["event_ids"] = [event["event_id"] for event in frame_events]
            geometry_entity_indexes.update(
                int(entity["entity"]) for entity in frame["entities"]
            )
            frames.append(frame)
            novel = [reason for reason in reasons if reason not in seen_phenomena]
            if novel:
                moments.append({
                    "frame": target_index, "tick": tick,
                    "model_time": frame["model_time"], "phenomena": novel,
                })
                seen_phenomena.update(novel)
            previous = frame
            target_index += 1
            target_tick = tick + args.stride_ticks
    finally:
        if raw_stream is not None:
            raw_stream.close()
    raw_receipt = None
    if raw_temporary is not None:
        os.replace(raw_temporary, args.raw_output)
        raw_receipt = {
            "format": "gzip-jsonl-exact-api-state-v1",
            "frames": len(frames),
            "sha256": sha256(args.raw_output),
            "bytes": args.raw_output.stat().st_size,
        }

    model_dt = finite(initial.get("performance", {}).get("dt"), "model dt")
    ticks = np.asarray([frame["tick"] for frame in frames], dtype=np.int64)
    tick_intervals = np.diff(ticks)
    if np.any(tick_intervals < args.stride_ticks):
        raise RuntimeError("observed frames violate the minimum tick stride")
    intervals = np.diff([frame["model_time"] for frame in frames])
    if not np.allclose(intervals, tick_intervals * model_dt, rtol=0.0, atol=2e-8):
        raise RuntimeError("observed model times differ from their actual tick intervals")
    sensory_lags = np.asarray([
        frame["model_time"] - frame["sampled_sensory_time"] for frame in frames
    ])
    if not np.isfinite(sensory_lags).all() or np.any(sensory_lags < -2e-8):
        raise RuntimeError("sampled sensory time follows its published physical frame")

    def selected_detail(frame: Mapping[str, Any]) -> Mapping[str, Any]:
        return next(
            value
            for value in frame["resident_details"]
            if value["body"] == args.resident_index
        )

    def selected_body(frame: Mapping[str, Any]) -> Mapping[str, Any]:
        return next(
            value for value in frame["bodies"] if value["body"] == args.resident_index
        )

    extrema = {
        "peak-speed": max(
            range(len(frames)),
            key=lambda index: selected_body(frames[index])["speed"],
        ),
        "peak-effort": max(
            range(len(frames)),
            key=lambda index: selected_detail(frames[index])["outcome"]["effort"],
        ),
        "minimum-energy": min(
            range(len(frames)),
            key=lambda index: selected_detail(frames[index])["recorded_body_state"][
                "energy"
            ],
        ),
        "peak-neural-activity": max(
            range(len(frames)),
            key=lambda index: selected_detail(frames[index])["neural_summary"][
                "activity"
            ],
        ),
    }
    by_frame = {item["frame"]: item for item in moments}
    for phenomenon, frame_index in extrema.items():
        if frame_index not in by_frame:
            frame = frames[frame_index]
            item = {
                "frame": frame_index, "tick": frame["tick"],
                "model_time": frame["model_time"], "phenomena": [],
            }
            moments.append(item)
            by_frame[frame_index] = item
        by_frame[frame_index]["phenomena"].append(phenomenon)
    moments.sort(key=lambda item: item["frame"])
    source_body_model = initial.get("body_model") or {}
    body_model = {
        key: source_body_model.get(key)
        for key in ("name", "kind", "joint_count_per_resident", "source_sha256", "actuation")
        if key in source_body_model
    }
    resident_hashes = source_body_model.get("resident_sha256", {})
    if resident_hashes:
        body_model["resident_sha256"] = [resident_hashes[body_id] for body_id in body_ids]

    capabilities = {
        "multi_resident_retina": {
            "status": "recorded",
            "source_path": "api/state.senses[*].rich_retina",
        },
        "multi_resident_neural_readouts": {
            "status": "recorded",
            "source_path": "api/state.neural[*].features",
        },
        "multi_resident_cognition": {
            "status": "recorded",
            "source_path": "api/state.cognition[*]",
        },
        "sequence_memory_summary": (
            {
                "status": "recorded",
                "source_path": "api/state.cognition[*].sequence_memory",
            }
            if any(
                detail["memory_summary"]["sequence"]["status"] == "recorded"
                for frame in frames
                for detail in frame["resident_details"]
            )
            else {
                "status": "unavailable",
                "source_path": "api/state.cognition[*].sequence_memory",
                "reason": "host did not publish bounded sequence-memory summaries",
            }
        ),
        "contextual_memory_summary": (
            {
                "status": "recorded",
                "source_path": "api/state.cognition[*].contextual_memory",
            }
            if any(
                detail["memory_summary"]["contextual"]["status"] == "recorded"
                for frame in frames
                for detail in frame["resident_details"]
            )
            else {
                "status": "unavailable",
                "source_path": "api/state.cognition[*].contextual_memory",
                "reason": "host did not publish bounded contextual-memory summaries",
            }
        ),
        "population_response_summary": (
            {
                "status": "recorded",
                "source_path": "api/state.cognition[*].population_response",
            }
            if any(
                detail["population_response"]["status"] == "recorded"
                for frame in frames
                for detail in frame["resident_details"]
            )
            else {
                "status": "unavailable",
                "source_path": "api/state.cognition[*].population_response",
                "reason": "host did not publish a fitted population-response summary",
            }
        ),
        "events": event_stream_capabilities,
    }
    event_counts: dict[str, int] = {kind: 0 for kind in EVENT_KINDS}
    for event in events:
        event_counts[event["kind"]] += 1
    for kind, count in event_counts.items():
        if count:
            capabilities["events"][kind] = {
                "status": "recorded",
                "source_path": f"api/state.evidence_events[kind={kind}]",
                "records": count,
            }

    event_stream = dict(event_stream_start)
    if event_stream["status"] == "recorded":
        event_stream.update(
            first_captured_sequence=(events[0]["sequence"] if events else None),
            last_captured_sequence=(events[-1]["sequence"] if events else None),
            captured_event_count=len(events),
            last_sequence_at_end=event_cursor,
            source_head_sha256_at_end=source_event_head,
            public_head_sha256=public_event_head,
        )

    first_observed_tick = {
        body: min(
            frame["tick"]
            for frame in frames
            if any(detail["body"] == body for detail in frame["resident_details"])
        )
        for body in body_indices.values()
    }
    private_binding = {
        "format": "chreatures-living-recording-private-binding-v1",
        "source_world_id": str(initial["id"]),
        "world_source_revision": args.world_source_revision,
        "world_source_content_sha256": args.world_source_content_sha256,
        "capture_tool_revision": provenance["capture_tool"]["revision"],
        "capture_tool_file_sha256": provenance["capture_tool"]["file_sha256"],
        "physical_profile_sha256": args.profile_sha256,
        "graph_sha256": provenance["graph_sha256"],
        "resident_artifact_sha256": provenance["resident_artifact_sha256"],
        "engine_identity_sha256": provenance["engine_identity"]["sha256"],
        "bodies": [
            {
                "public_body": public_index,
                "source_body_id": source_id,
                "first_observed_tick": first_observed_tick[public_index],
            }
            for source_id, public_index in sorted(
                body_indices.items(), key=lambda item: item[1]
            )
        ],
    }
    private_binding["content_sha256"] = sha256_bytes(
        canonical_bytes(private_binding)
    )
    atomic_json(args.binding_output, private_binding)
    binding_receipt = {
        "format": private_binding["format"],
        "content_sha256": private_binding["content_sha256"],
        "file_sha256": sha256(args.binding_output),
        "bytes": args.binding_output.stat().st_size,
        "body_count": len(body_indices),
        "scope": "private source-ID binding; excluded from public site assets",
    }

    encode_entity_deltas(frames)
    result = {
        "format": FORMAT,
        "geometry_encoding": "entity-replacement-delta-v1",
        "status": (
            "observed physical recording; behavior and growth are not evidence of learned competence"
        ),
        "sampling": {
            "first_requested_tick": start_tick,
            "event_start_tick": event_start_tick,
            "minimum_stride_ticks": args.stride_ticks,
            "strict_requested_ticks": args.strict_ticks,
            "model_dt_seconds": model_dt,
            "model_interval_seconds": model_dt * args.stride_ticks,
            "model_interval_semantics": (
                "minimum requested interval; observed ticks and times are authoritative"
            ),
            "frames": args.frames,
            "observed_ticks": ticks.astype(int).tolist(),
            "observed_model_times": [frame["model_time"] for frame in frames],
            "observed_sensory_times": [
                frame["sampled_sensory_time"] for frame in frames
            ],
            "sensory_to_postphysics_lag_seconds": sensory_lags.astype(float).tolist(),
            "transport": "read-only HTTP GET /api/state; no browser and no world commands",
        },
        "privacy": {
            "selected_resident": f"resident-{args.resident_index}",
            "source_body_ids_exported": False,
            "source_entity_ids_exported": False,
            "private_state": (
                "bounded memory counts, learning diagnostics, goal timing, and optional sequence "
                "summaries are exported; memory content, slots, account identity, raw hidden "
                "state, and native RNG are excluded"
            ),
        },
        "capabilities": capabilities,
        "event_stream": event_stream,
        "provenance": provenance,
        "organism_interface": {
            "format": contract["format"],
            "action_order": list(contract["actions"]),
            "host_physiology_order": list(contract["physiology"]),
        },
        "recorded_body_state": {
            "timing": "post-physics host view at frame model_time; not the normalized pre-action controller input",
            "fields": [
                {"name": name, "unit": definition[2]}
                for name, definition in BODY_STATE_FIELDS.items()
            ],
        },
        "private_raw_receipt": raw_receipt,
        "private_binding_receipt": binding_receipt,
        "geometry": {
            "dimension": int(initial["dimension"]),
            "bounds": [
                finite(initial[key], key) for key in ("width", "height", "depth")
            ],
            "body_count": len(body_indices), "entity_count": len(entity_indices),
            "event_only_entity_count": len(entity_indices) - len(geometry_entity_indexes),
            "engine": initial["engine"],
            "body_model": body_model,
        },
        "frames": frames,
        "events": events,
        "phenomena_moments": moments,
        "limitations": [
            "Retinal values are public-display quantized to uint8; neural readouts retain exact float32 bytes.",
            "Entity geometry uses one full keyframe followed by full-object replacement deltas; exact host states remain content-addressed in the private raw recording.",
            "Rendered entity and articulation geometry is rounded to five decimal places for the public replay.",
            "Phenomena labels index observed physical changes and do not interpret intention or learning.",
            "Every present resident's direct retina and neural readouts are sampled; unavailable diagnostic streams are explicit in capabilities.",
            "Events are copied only from stable host receipts; absent event capabilities are not reconstructed from scene changes.",
        ],
    }
    result["content_sha256"] = sha256_bytes(canonical_bytes(result))
    atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output), "frames": len(frames),
        "moments": len(moments), "content_sha256": result["content_sha256"],
        "file_sha256": sha256(args.output),
        "private_binding_output": str(args.binding_output),
        "private_binding_content_sha256": private_binding["content_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
