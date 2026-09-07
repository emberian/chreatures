"""Immutable video-file transport for a physical screen; no policy interface."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

FORMAT = "chreatures-physical-screen-stimulus-v1"


class ScreenStimulus:
    """Select a recorded source frame by physical tick, without advancing a life."""

    def __init__(self, specification):
        required = {"format", "frames_path", "frames_sha256", "fps", "start_tick", "end"}
        if not isinstance(specification, dict) or set(specification) != required or specification["format"] != FORMAT:
            raise ValueError("physical screen stimulus specification differs")
        if (type(specification["start_tick"]) is not int or specification["start_tick"] < 0
                or specification["end"] not in {"black", "hold"}
                or type(specification["fps"]) not in {int, float}
                or not math.isfinite(specification["fps"]) or not 0 < specification["fps"] <= 120):
            raise ValueError("physical screen timing differs")
        path = Path(specification["frames_path"]).expanduser().resolve()
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != specification["frames_sha256"]:
            raise ValueError("physical screen file identity differs")
        self.frames = np.load(path, mmap_mode="r", allow_pickle=False)
        if (self.frames.dtype != np.uint8 or self.frames.ndim != 4 or self.frames.shape[-1] != 3
                or not 1 <= len(self.frames) <= 36000 or min(self.frames.shape[1:3]) < 2
                or max(self.frames.shape[1:3]) > 4096):
            raise ValueError("screen frames must be bounded uint8[T,H,W,3]")
        self.specification = copy.deepcopy(specification)
        self.specification["frames_path"] = str(path)
        self._black = np.zeros(self.frames.shape[1:], dtype=np.float32)

    def frame(self, tick: int, dt: float) -> np.ndarray:
        if type(tick) is not int or tick < 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("invalid physical stimulus clock")
        elapsed = tick - self.specification["start_tick"]
        if elapsed < 0:
            return self._black
        index = math.floor(elapsed * dt * self.specification["fps"] + 1e-9)
        if index >= len(self.frames):
            if self.specification["end"] == "black":
                return self._black
            index = len(self.frames) - 1
        # Bulk dtype transport only. Projection, occlusion and transduction run
        # in the native physical sampler; no frame reaches a policy directly.
        return self.frames[index].astype(np.float32) / np.float32(255)

    def snapshot(self):
        return copy.deepcopy(self.specification)
