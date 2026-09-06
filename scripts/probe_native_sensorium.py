#!/usr/bin/env python3
"""Exactness and focused throughput probe for native retinal transduction.

The Python receptor loop in this file is a research reference for the retired
production calculation.  It is never selected by a habitat or runtime.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-module-dir",
        type=Path,
        help="directory containing an isolated _world_kernels build",
    )
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be nonnegative")
    if args.native_module_dir is not None:
        sys.path.insert(0, str(args.native_module_dir.resolve()))
    sys.path.insert(1 if args.native_module_dir is not None else 0, str(ROOT))

    import mujoco
    import numpy as np

    from chreatures.neural_ports import encode_physical_senses, load_port_spec
    from chreatures.physical_batch import FastArticulatedSensoriumWorld
    import chreatures.sensorium as sensorium

    native_retina = sensorium.native_retina
    habitat = json.loads(
        (ROOT / "data/habitats/hollow-garden.json").read_text(encoding="utf-8")
    )
    ports = load_port_spec()

    def reference_retina(world, body):
        """Retired scalar transduction loop, retained only as this oracle."""
        origin, directions, excluded_body = sensorium._ray_geometry(world, body)
        ray_count = len(sensorium.RETINA_PITCH_OFFSETS) * len(
            sensorium.RETINA_YAW_OFFSETS
        )
        geom_ids = np.full(ray_count, -1, dtype=np.int32)
        distances = np.full(ray_count, -1.0, dtype=np.float64)
        mujoco.mj_multiRay(
            world.model,
            world.data,
            origin,
            np.ascontiguousarray(directions.reshape(-1)),
            None,
            True,
            excluded_body,
            geom_ids,
            distances,
            None,
            ray_count,
            sensorium.RETINA_MAX_RANGE,
        )
        illumination = world._illumination(body)
        result = []
        for band in range(len(sensorium.RETINA_PITCH_OFFSETS)):
            rows = []
            for column in range(len(sensorium.RETINA_YAW_OFFSETS)):
                index = band * len(sensorium.RETINA_YAW_OFFSETS) + column
                distance, geom_id = float(distances[index]), int(geom_ids[index])
                if (
                    distance < 0.0
                    or distance > sensorium.RETINA_MAX_RANGE
                    or geom_id < 0
                ):
                    rows.append([0.0, 0.0, 0.0, 0.0])
                else:
                    rgb = world._geom_rgb(geom_id)
                    rows.append(
                        [
                            min(1.0, channel * (0.45 + 0.55 * illumination))
                            for channel in rgb
                        ]
                        + [max(0.0, 1.0 - distance / sensorium.RETINA_MAX_RANGE)]
                    )
            result.append(rows)
        return result

    audit = {}
    maximum_retina_error = 0.0
    maximum_channel_error = 0.0
    for frame in sorted(sensorium.SENSORIUM_FRAMES):
        spec = json.loads(json.dumps(habitat))
        spec["sensorium"] = {"frame": frame}
        world = FastArticulatedSensoriumWorld(seed=731, spec=spec)
        frame_retina_error = frame_channel_error = 0.0
        for body in world.bodies:
            sensorium.native_retina = reference_retina
            reference_senses = world.sense(body.id)
            sensorium.native_retina = native_retina
            native_senses = world.sense(body.id)
            retina_error = float(
                np.max(
                    np.abs(
                        np.asarray(reference_senses["retina3d"], dtype=np.float64)
                        - np.asarray(native_senses["retina3d"], dtype=np.float64)
                    )
                )
            )
            reference_channels = encode_physical_senses(reference_senses, ports)[1]
            native_channels = encode_physical_senses(native_senses, ports)[1]
            channel_error = float(
                np.max(np.abs(reference_channels - native_channels))
            )
            frame_retina_error = max(frame_retina_error, retina_error)
            frame_channel_error = max(frame_channel_error, channel_error)

        # Material colour is mutable without a topology revision.  Both paths
        # must observe the current model array on the very next sense.
        world.model.mat_rgba[:, 0] = np.linspace(
            0.01, 0.99, world.model.nmat, dtype=np.float32
        )
        body = world.bodies[0]
        reference = np.asarray(reference_retina(world, body), dtype=np.float64)
        current = np.asarray(native_retina(world, body), dtype=np.float64)
        mutable_error = float(np.max(np.abs(reference - current)))
        frame_retina_error = max(frame_retina_error, mutable_error)
        audit[frame] = {
            "residents": len(world.bodies),
            "retina_max_abs_error": frame_retina_error,
            "encoded_351_max_abs_error": frame_channel_error,
            "mutable_material_max_abs_error": mutable_error,
        }
        maximum_retina_error = max(maximum_retina_error, frame_retina_error)
        maximum_channel_error = max(maximum_channel_error, frame_channel_error)

    timing_spec = json.loads(json.dumps(habitat))
    timing_spec["sensorium"] = {"frame": sensorium.BODY_FRAME}
    timing_world = FastArticulatedSensoriumWorld(seed=733, spec=timing_spec)

    def sample_all() -> None:
        for body in timing_world.bodies:
            encode_physical_senses(timing_world.sense(body.id), ports)

    def measure(implementation, iterations: int) -> float:
        sensorium.native_retina = implementation
        for _ in range(args.warmup):
            sample_all()
        enabled = gc.isenabled()
        gc.disable()
        try:
            started = time.perf_counter()
            for _ in range(iterations):
                sample_all()
            return time.perf_counter() - started
        finally:
            if enabled:
                gc.enable()

    try:
        reference_seconds = measure(reference_retina, args.iterations)
        native_seconds = measure(native_retina, args.iterations)
    finally:
        sensorium.native_retina = native_retina
    resident_samples = args.iterations * len(timing_world.bodies)
    module = sys.modules["_world_kernels"]
    module_path = Path(module.__file__).resolve()
    report = {
        "format": "chreatures-native-sensorium-v1",
        "scope": "complete PhysicsWorld.sense plus 351-channel encode; no physics advance or whole-runtime claim",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
        },
        "source_sha256": {
            "chreatures/sensorium.py": _sha256(ROOT / "chreatures/sensorium.py"),
            "chreatures/neural_ports.py": _sha256(ROOT / "chreatures/neural_ports.py"),
            "native/world-kernels/src/sensorium.rs": _sha256(
                ROOT / "native/world-kernels/src/sensorium.rs"
            ),
            "native_module": _sha256(module_path),
        },
        "native_module": str(module_path),
        "audit": audit,
        "maximum_retina_abs_error": maximum_retina_error,
        "maximum_encoded_351_abs_error": maximum_channel_error,
        "timing": {
            "warmup_batches_per_path": args.warmup,
            "measured_batches_per_path": args.iterations,
            "residents_per_batch": len(timing_world.bodies),
            "reference_seconds": reference_seconds,
            "native_seconds": native_seconds,
            "reference_resident_senses_per_second": resident_samples
            / reference_seconds,
            "native_resident_senses_per_second": resident_samples / native_seconds,
            "complete_sense_encode_speedup": reference_seconds / native_seconds,
        },
    }
    if maximum_retina_error != 0.0 or maximum_channel_error != 0.0:
        raise RuntimeError("native retinal transduction changed sensed values")
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
