#!/usr/bin/env python3
"""Render saved resident fields of view and optionally queue native VLM observations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request

from chreatures.retinal_render import RetinalRenderer, load_snapshot_world


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def post_perception(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/perceive",
        data=canonical(payload),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = {"message": str(error)}
        return {"status": "transport_error", "http_status": error.code, "detail": detail}
    except (OSError, TimeoutError) as error:
        return {"status": "transport_error", "reason": f"{type(error).__name__}: {error}"}
    if not isinstance(value, dict):
        return {"status": "transport_error", "reason": "perception response was not an object"}
    return value


def select_dense_feature(response: dict[str, Any], count: int) -> dict[str, Any]:
    result = dict(response)
    dense = result.get("dense_feature")
    if not isinstance(dense, dict) or not isinstance(dense.get("values"), list):
        return result
    values = dense["values"]
    dimension = len(values)
    vector_sha256 = hashlib.sha256(canonical(values)).hexdigest()
    if count <= 0 or dimension == 0:
        result.pop("dense_feature", None)
        return result
    count = min(count, dimension)
    if count == 1:
        indices = [0]
    else:
        indices = sorted(
            {round(index * (dimension - 1) / (count - 1)) for index in range(count)}
        )
    result["dense_feature"] = {
        "kind": dense.get("kind"),
        "source_dimension": dense.get("dimension", dimension),
        "source_vector_sha256": vector_sha256,
        "selection": "evenly spaced indices across native vector",
        "indices": indices,
        "values": [values[index] for index in indices],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--resident", action="append", dest="residents")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/perception/fov"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--vertical-fov", type=float, default=82.0)
    parser.add_argument("--perception-url")
    parser.add_argument("--dense-count", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if not 0 <= args.dense_count <= 960:
        parser.error("--dense-count must be in 0..960")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    world, source = load_snapshot_world(args.checkpoint)
    resident_ids = [body.id for body in world.bodies]
    requested = args.residents or resident_ids
    if len(set(requested)) != len(requested):
        parser.error("resident selections must be unique")
    unknown = set(requested) - set(resident_ids)
    if unknown:
        parser.error(f"unknown resident selection: {sorted(unknown)}")
    tick = source.get("tick")
    world_sequence = (
        tick
        if isinstance(tick, int) and not isinstance(tick, bool) and tick >= 0
        else max(0, round(world.time / 0.05))
    )
    report_path = args.report or args.output_dir / "observations.json"
    observations = []

    with RetinalRenderer(
        world,
        width=args.width,
        height=args.height,
        vertical_fov_degrees=args.vertical_fov,
    ) as renderer:
        for resident_id in requested:
            captured_at = time.time()
            frame = renderer.render(resident_id, captured_at=captured_at)
            png = frame.png()
            png_sha256 = hashlib.sha256(png).hexdigest()
            image_path = args.output_dir / f"{resident_id}-{world_sequence}.png"
            write_atomic(image_path, png)
            sensor_digest = hashlib.sha256(
                f"{source.get('checkpoint_sha256', '')}:{resident_id}".encode()
            ).hexdigest()[:20]
            sensor_id = f"retina.{sensor_digest}"
            request_id = f"fov.{world_sequence}.{sensor_digest}"
            payload = {
                "request_id": request_id,
                "source": {
                    "sensor_id": sensor_id,
                    "world_sequence": world_sequence,
                    "model_time": frame.model_time,
                    "captured_at": frame.captured_at,
                    "provenance": "resident_fov",
                },
                "frames": [
                    {
                        "mime_type": "image/png",
                        "data_base64": base64.b64encode(png).decode(),
                    }
                ],
                "include_dense_feature": args.dense_count > 0,
            }
            if args.perception_url:
                perception = select_dense_feature(
                    post_perception(args.perception_url, payload, args.timeout),
                    args.dense_count,
                )
            else:
                perception = {
                    "status": "not_requested",
                    "reason": "no --perception-url was supplied",
                }
            observations.append(
                {
                    "resident_id": resident_id,
                    "experienced": True,
                    "observed": perception.get("status") == "ok",
                    "provenance": "resident_fov",
                    "captured_at": frame.captured_at,
                    "model_time": frame.model_time,
                    "world_sequence": world_sequence,
                    "semantics_origin": "model_hypothesis",
                    "frame": {
                        "path": str(image_path.resolve()),
                        "mime_type": "image/png",
                        "width": frame.width,
                        "height": frame.height,
                        "bytes": len(png),
                        "sha256": png_sha256,
                        "camera": frame.camera.to_dict(),
                        "overlays": False,
                    },
                    "perception_request": {
                        "request_id": request_id,
                        "sensor_id": sensor_id,
                        "payload_content": "rendered RGB pixels and source timing only",
                        "scene_labels_or_ids": False,
                    },
                    "perception": perception,
                }
            )

    body = {
        "version": 1,
        "source": source,
        "renderer": {
            "engine": "MuJoCo",
            "version": world.snapshot()["engine"]["version"],
            "width": args.width,
            "height": args.height,
            "vertical_fov_degrees": args.vertical_fov,
        },
        "perception_url_used": bool(args.perception_url),
        "controller_injection": False,
        "observations": observations,
    }
    envelope = {
        "format": "chreatures-retinal-observations-v1",
        "sha256": hashlib.sha256(canonical(body)).hexdigest(),
        "state": body,
    }
    write_atomic(report_path, canonical(envelope) + b"\n")
    print(json.dumps(envelope, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
