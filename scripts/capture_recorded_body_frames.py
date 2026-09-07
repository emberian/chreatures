#!/usr/bin/env python3
"""Capture bounded offscreen MuJoCo body views from actual world checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from chreatures.retinal_render import load_snapshot_world, restore_snapshot_world


FORMAT = "chreatures-recorded-body-frames-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_world(path: Path) -> tuple[Any, dict[str, Any]]:
    """Accept both physics checkpoints and authenticated Habitat3D envelopes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    state = value.get("state") if isinstance(value, dict) else None
    if isinstance(state, dict) and isinstance(state.get("world"), dict):
        encoded = json.dumps(
            state, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != value.get("sha256"):
            raise ValueError("Habitat3D checkpoint content hash differs")
        world, source = restore_snapshot_world(state["world"])
        source.update(
            checkpoint_format=value.get("format"),
            checkpoint_sha256=value["sha256"], tick=state.get("tick"),
        )
        return world, source
    return load_snapshot_world(path)


class MuJoCoBodyFrameCapture:
    """Bounded writer called at committed physical states of one actual world."""

    def __init__(
        self, output_dir: str | Path, *, frame_count: int, resident_index: int = 0,
        width: int = 640, height: int = 480, azimuth: float = 135.0,
        elevation: float = -24.0, distance: float = 2.5,
        headlight_ambient: float = 0.34, headlight_diffuse: float = 0.78,
        headlight_specular: float = 0.18,
    ):
        if not 1 <= frame_count <= 3600:
            raise ValueError("frame_count must lie in 1..3600")
        if not 0 <= resident_index < 64:
            raise ValueError("resident_index must lie in 0..63")
        if not 64 <= width <= 2048 or not 64 <= height <= 2048:
            raise ValueError("capture dimensions must lie in 64..2048")
        if not 0.05 <= distance <= 100:
            raise ValueError("camera distance must lie in [0.05,100]")
        if any(
            not np.isfinite(value) or not 0 <= value <= 1
            for value in (headlight_ambient, headlight_diffuse, headlight_specular)
        ):
            raise ValueError("observer headlight values must lie in [0,1]")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = frame_count
        self.resident_index = resident_index
        self.width, self.height = width, height
        self.azimuth, self.elevation, self.distance = azimuth, elevation, distance
        self.headlight_ambient = float(headlight_ambient)
        self.headlight_diffuse = float(headlight_diffuse)
        self.headlight_specular = float(headlight_specular)
        self.frame_path = self.output_dir / "body_frames.npy"
        self.time_path = self.output_dir / "body_time_seconds.npy"
        self.receipt_path = self.output_dir / "body-frames.receipt.json"
        for path in (self.frame_path, self.time_path, self.receipt_path):
            if path.exists():
                raise FileExistsError(f"body capture refuses existing output: {path}")
        self._frame_tmp = self.output_dir / f".body_frames.tmp-{os.getpid()}.npy"
        self._time_tmp = self.output_dir / f".body_time_seconds.tmp-{os.getpid()}.npy"
        self.frames = np.lib.format.open_memmap(
            self._frame_tmp, mode="w+", dtype=np.uint8,
            shape=(frame_count, height, width, 3),
        )
        self.times = np.lib.format.open_memmap(
            self._time_tmp, mode="w+", dtype=np.float64, shape=(frame_count,),
        )
        self.index = 0
        self._renderer = None
        self._model_address = None
        self._active_model_signature = None
        self._model_epochs: list[dict[str, Any]] = []
        self._resident_id: str | None = None

    def _ensure_renderer(self, world: Any, timestamp: float) -> None:
        import mujoco
        address = int(world.model._address)
        signature = getattr(world, "_model_signature", None)
        if not isinstance(signature, str) or not signature:
            raise ValueError("world does not publish its physical model signature")
        signature_changed = signature != self._active_model_signature
        if signature_changed:
            self._model_epochs.append({
                "first_frame": self.index, "time_seconds": timestamp,
                "model_signature": signature,
            })
            self._active_model_signature = signature
        if (
            self._renderer is not None
            and address == self._model_address
            and not signature_changed
        ):
            return
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(
            world.model, height=self.height, width=self.width,
        )
        self._model_address = address

    def append(self, world: Any, *, model_time: float | None = None) -> None:
        import mujoco
        if self.index >= self.frame_count:
            raise RuntimeError("body capture exceeds its declared frame bound")
        if self.resident_index >= len(world.bodies):
            raise ValueError("selected resident is absent from the world")
        timestamp = float(world.time if model_time is None else model_time)
        if not np.isfinite(timestamp) or (self.index and timestamp <= self.times[self.index - 1]):
            raise ValueError("body frame times must be finite and strictly increasing")
        self._ensure_renderer(world, timestamp)
        body = world.bodies[self.resident_index]
        resident_id = str(body.id)
        if self._resident_id is None:
            self._resident_id = resident_id
        elif resident_id != self._resident_id:
            raise ValueError("selected resident index now refers to a different body")
        root = world._body_mj[body.id]
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = np.asarray(world.data.xpos[root], dtype=np.float64)
        camera.distance = self.distance
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation
        option = mujoco.MjvOption()
        option.label = mujoco.mjtLabel.mjLABEL_NONE
        option.frame = mujoco.mjtFrame.mjFRAME_NONE
        headlight = world.model.vis.headlight
        saved_headlight = (
            np.asarray(headlight.ambient).copy(), np.asarray(headlight.diffuse).copy(),
            np.asarray(headlight.specular).copy(),
        )
        try:
            headlight.ambient[:] = self.headlight_ambient
            headlight.diffuse[:] = self.headlight_diffuse
            headlight.specular[:] = self.headlight_specular
            prepare_observer = getattr(world, "prepare_observer_renderer", None)
            if prepare_observer is not None:
                prepare_observer(self._renderer)
            self._renderer.update_scene(world.data, camera=camera, scene_option=option)
            frame = np.asarray(self._renderer.render(), dtype=np.uint8)
        finally:
            headlight.ambient[:] = saved_headlight[0]
            headlight.diffuse[:] = saved_headlight[1]
            headlight.specular[:] = saved_headlight[2]
        if frame.shape != (self.height, self.width, 3):
            raise RuntimeError("MuJoCo returned an unexpected body-frame shape")
        self.frames[self.index] = frame
        self.times[self.index] = timestamp
        self.index += 1

    def seal(self, *, checkpoint_hashes: list[str], source_world_revision: str) -> dict[str, Any]:
        if self.index != self.frame_count:
            raise RuntimeError(f"body capture has {self.index} of {self.frame_count} frames")
        if not source_world_revision:
            raise ValueError("source world revision is required")
        for path in (self.frame_path, self.time_path, self.receipt_path):
            if path.exists():
                raise FileExistsError(f"body capture refuses existing output at seal: {path}")
        self.frames.flush(); self.times.flush()
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
        os.replace(self._frame_tmp, self.frame_path)
        os.replace(self._time_tmp, self.time_path)
        receipt = {
            "format": FORMAT,
            "status": "offscreen MuJoCo frames rendered from committed physical world states",
            "frame_count": self.frame_count,
            "resident_index": self.resident_index,
            "resident_id": self._resident_id,
            "shape": [self.frame_count, self.height, self.width, 3],
            "time_seconds": [float(self.times[0]), float(self.times[-1])],
            "camera": {
                "kind": "free camera tracking the selected physical root body",
                "azimuth_degrees": self.azimuth,
                "elevation_degrees": self.elevation,
                "distance_meters": self.distance,
            },
            "observer_lighting": {
                "kind": "temporary MuJoCo observer headlight, restored after each render",
                "ambient_rgb": [self.headlight_ambient] * 3,
                "diffuse_rgb": [self.headlight_diffuse] * 3,
                "specular_rgb": [self.headlight_specular] * 3,
            },
            "model_signature_epochs": self._model_epochs,
            "source_world_revision": source_world_revision,
            "checkpoint_sha256": checkpoint_hashes,
            "frames": {"file": self.frame_path.name, "sha256": sha256(self.frame_path), "bytes": self.frame_path.stat().st_size},
            "times": {"file": self.time_path.name, "sha256": sha256(self.time_path), "bytes": self.time_path.stat().st_size},
        }
        self.receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt

    def abort(self) -> None:
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
        self._frame_tmp.unlink(missing_ok=True); self._time_tmp.unlink(missing_ok=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True, help="Committed checkpoint; repeat in playback order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-world-revision", required=True)
    parser.add_argument("--resident-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-24.0)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--headlight-ambient", type=float, default=0.34)
    parser.add_argument("--headlight-diffuse", type=float, default=0.78)
    parser.add_argument("--headlight-specular", type=float, default=0.18)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    capture = MuJoCoBodyFrameCapture(
        args.output_dir, frame_count=len(args.checkpoint),
        resident_index=args.resident_index, width=args.width, height=args.height,
        azimuth=args.azimuth, elevation=args.elevation, distance=args.distance,
        headlight_ambient=args.headlight_ambient,
        headlight_diffuse=args.headlight_diffuse,
        headlight_specular=args.headlight_specular,
    )
    hashes = []
    try:
        for path in args.checkpoint:
            hashes.append(sha256(path))
            world, _source = load_checkpoint_world(path)
            capture.append(world)
        receipt = capture.seal(
            checkpoint_hashes=hashes,
            source_world_revision=args.source_world_revision,
        )
    except BaseException:
        capture.abort()
        raise
    print(json.dumps({
        "output_dir": str(args.output_dir), "frames": receipt["frame_count"],
        "frames_sha256": receipt["frames"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
