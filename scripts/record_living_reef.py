#!/usr/bin/env python3
"""Record a compact, public, headless Living Reef episode from `/api/state`."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


FORMAT = "chreatures-living-reef-public-recording-v1"
ACTION_NAMES = (
    "thrust", "yaw", "gaze_pitch", "grip",
    "signal_low", "signal_mid", "signal_high", "posture",
)
PERIPHERAL_SHAPE = (8, 32, 4)
FOVEAL_SHAPE = (24, 32, 4)


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
    parser.add_argument("--resident-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--stride-ticks", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--strict-ticks", action="store_true",
        help="Fail instead of accepting a later real tick when polling skips a target",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-content-sha256", required=True)
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
    if not 0 <= args.resident_index < 64:
        raise SystemExit("resident index must be in 0..63")
    if not 2 <= args.frames <= 10_000 or not 1 <= args.stride_ticks <= 1_000:
        raise SystemExit("invalid frame count or tick stride")
    if not 0.001 <= args.poll_seconds <= 1.0 or not 1 <= args.timeout_seconds <= 86_400:
        raise SystemExit("invalid polling interval or timeout")
    for label, value in (
        ("source content", args.source_content_sha256),
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
    return {
        "entity": public_index,
        "shapes": [shape_geometry(shape) for shape in entity.get("shapes", [])],
        "joint": entity.get("joint"),
    }


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
        "source_revision": args.source_revision,
        "source_content_sha256": args.source_content_sha256,
        "source_content_semantics": (
            "caller-pinned source archive or the identical engine_identity.sha256"
        ),
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


def extract_frame(
    state: Mapping[str, Any], selected_id: str, body_ids: list[str],
    entity_indices: dict[str, int], previous: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    bodies = state["bodies"]
    entities = state["entities"]
    by_body = {str(body["id"]): body for body in bodies}
    by_entity = {str(entity["id"]): entity for entity in entities}
    if list(by_body) != body_ids:
        raise ValueError("resident topology changed during recording")
    for entity_id in by_entity:
        if entity_id not in entity_indices:
            entity_indices[entity_id] = len(entity_indices)
    senses = state["senses"][selected_id]["rich_retina"]
    neural = state["neural"][selected_id]
    cognition = state["cognition"][selected_id]
    retina_pose = state.get("sensorium", {}).get("retina_pose", {}).get(selected_id)
    if not isinstance(retina_pose, Mapping):
        raise ValueError("host view lacks the selected retina physical pose")
    selected = by_body[selected_id]
    action = cognition.get("executed_action", {})
    if set(action) != set(ACTION_NAMES) | {"oral"}:
        raise ValueError("host has not published the committed action-plus-oral schema")
    goal = cognition["goal"]
    outcomes = state.get("outcomes", {})
    frame = {
        "tick": int(state["tick"]),
        "model_time": finite(state["time"], "model time"),
        "sampled_sensory_time": finite(state["sensed_at"], "sampled sensory time"),
        "bodies": [body_geometry(by_body[key], index) for index, key in enumerate(body_ids)],
        "resident_traces": [
            {
                "body": index,
                "committed_action": {
                    key: finite(
                        state["cognition"][body_id]["executed_action"][key],
                        f"resident action {key}",
                    )
                    for key in (*ACTION_NAMES, "oral")
                },
                "metabolism": {
                    key: finite(by_body[body_id][key], f"resident metabolism {key}")
                    for key in ("energy", "gut", "fatigue")
                },
                "outcome": {
                    key: finite(
                        outcomes.get(body_id, {}).get(key, 0.0),
                        f"resident outcome {key}",
                    )
                    for key in (
                        "nutrition", "ingested_mass", "effort", "contact",
                        "mouth_material_contacts",
                    )
                },
            }
            for index, body_id in enumerate(body_ids)
        ],
        "entities": [
            object_geometry(entity, entity_indices[entity_id])
            for entity_id, entity in by_entity.items()
        ],
        "articulations": [
            {
                "body": body_ids.index(str(item["id"])),
                "links": item.get("links", []), "geoms": item.get("geoms", []),
                "joints": item.get("joints", []), "sites": item.get("sites", []),
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
        "selected": {
            "body": body_ids.index(selected_id),
            "retina": {
                "peripheral": retina_u8(
                    senses["peripheral"], PERIPHERAL_SHAPE, "peripheral retina",
                ),
                "foveal": retina_u8(senses["foveal"], FOVEAL_SHAPE, "foveal retina"),
                "profile": senses["profile"],
            },
            "retina_pose": {
                key: vector(retina_pose[key], 3, f"retina pose {key}")
                for key in ("origin", "forward", "up", "right")
            },
            "neural_readouts": float32_blob(neural["features"], 384, "neural readouts"),
            "neural_summary": {
                key: finite(neural[key], f"neural {key}")
                for key in ("activity", "support")
            },
            "goal": {
                "valid": bool(goal["valid"]),
                "changed": bool(goal["changed"]),
                "recorded_tick": int(goal["recorded_tick"]),
                "recorded_time": finite(goal["recorded_time"], "goal recorded time"),
                "commit_remaining_ticks": int(goal["remaining_ticks"]),
            },
            "committed_action": {
                key: finite(action[key], f"action {key}")
                for key in (*ACTION_NAMES, "oral")
            },
            "metabolism": {
                key: finite(selected[key], f"metabolism {key}")
                for key in ("energy", "gut", "fatigue")
            },
            "outcome": {
                key: finite(outcomes.get(selected_id, {}).get(key, 0.0), f"outcome {key}")
                for key in (
                    "nutrition", "ingested_mass", "effort", "contact",
                    "mouth_material_contacts",
                )
            },
        },
    }
    reasons = []
    if goal["changed"]:
        reasons.append("private-goal-commit")
    outcome = frame["selected"]["outcome"]
    if outcome["ingested_mass"] > 0 or outcome["nutrition"] > 0:
        reasons.append("physical-ingestion")
    if outcome["mouth_material_contacts"] > 0:
        reasons.append("mouth-material-contact")
    if previous is not None:
        old = previous["selected"]
        if len(frame["signals"]) != len(previous["signals"]):
            reasons.append("signal-field-change")
        if abs(frame["selected"]["metabolism"]["energy"] - old["metabolism"]["energy"]) >= 0.01:
            reasons.append("energy-change")
        if frame["bodies"][body_ids.index(selected_id)]["speed"] >= 0.1:
            reasons.append("rapid-motion")
    return frame, reasons


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
    entity_indices = {
        str(entity["id"]): index for index, entity in enumerate(entities)
    }
    selected_id = body_ids[args.resident_index]
    provenance = identity(initial, args)
    start_tick = int(initial["tick"]) + args.stride_ticks
    target_tick = start_tick
    deadline = time.monotonic() + args.timeout_seconds
    frames = []
    moments = []
    seen_phenomena: set[str] = set()
    previous = None
    target_index = 0
    while target_index < args.frames:
        if time.monotonic() >= deadline:
            raise TimeoutError("recording timed out before all fixed ticks arrived")
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
        frame, reasons = extract_frame(
            state, selected_id, body_ids, entity_indices, previous,
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

    extrema = {
        "peak-speed": max(
            range(len(frames)),
            key=lambda index: frames[index]["bodies"][args.resident_index]["speed"],
        ),
        "peak-effort": max(
            range(len(frames)),
            key=lambda index: frames[index]["selected"]["outcome"]["effort"],
        ),
        "minimum-energy": min(
            range(len(frames)),
            key=lambda index: frames[index]["selected"]["metabolism"]["energy"],
        ),
        "peak-neural-activity": max(
            range(len(frames)),
            key=lambda index: frames[index]["selected"]["neural_summary"]["activity"],
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

    result = {
        "format": FORMAT,
        "status": (
            "observed physical recording; behavior and growth are not evidence of learned competence"
        ),
        "sampling": {
            "first_requested_tick": start_tick,
            "minimum_stride_ticks": args.stride_ticks,
            "strict_requested_ticks": args.strict_ticks,
            "model_dt_seconds": model_dt,
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
            "resident": f"resident-{args.resident_index}",
            "source_body_ids_exported": False,
            "source_entity_ids_exported": False,
            "private_state": (
                "only goal recorded time and remaining commitment are exported; memory content, "
                "slots, account identity, and native controller state are excluded"
            ),
        },
        "provenance": provenance,
        "geometry": {
            "dimension": int(initial["dimension"]),
            "bounds": [
                finite(initial[key], key) for key in ("width", "height", "depth")
            ],
            "body_count": len(body_ids), "entity_count": len(entity_indices),
            "engine": initial["engine"],
            "body_model": body_model,
        },
        "frames": frames,
        "phenomena_moments": moments,
        "limitations": [
            "Retinal values are public-display quantized to uint8; neural readouts retain exact float32 bytes.",
            "Phenomena labels index observed physical changes and do not interpret intention or learning.",
            "The recording samples one resident's direct retina and neural readouts while retaining all resident motion.",
        ],
    }
    result["content_sha256"] = sha256_bytes(canonical_bytes(result))
    atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output), "frames": len(frames),
        "moments": len(moments), "content_sha256": result["content_sha256"],
        "file_sha256": sha256(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
