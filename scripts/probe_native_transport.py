#!/usr/bin/env python3
"""Measure the native field kernel against the complete reference field solver."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chreatures.fields import FieldEnvironment


class ReferenceField(FieldEnvironment):
    """Archived NumPy equation used only as an independent numerical probe."""

    def _transport(self, dt: float, flow: np.ndarray) -> None:
        current = self.concentration
        change = np.zeros_like(current)
        # concentration axes are channel,z,y,x. Flow components are x,y,z.
        for flow_component, axis, spacing in ((0, 3, self.dx), (1, 2, self.dy), (2, 1, self.dz)):
            left_slice = [slice(None)] * 4
            right_slice = [slice(None)] * 4
            left_slice[axis] = slice(None, -1)
            right_slice[axis] = slice(1, None)
            left_slice = tuple(left_slice)
            right_slice = tuple(right_slice)
            scalar_axis = axis - 1
            p_left = [slice(None)] * 3
            p_right = [slice(None)] * 3
            p_left[scalar_axis] = slice(None, -1)
            p_right[scalar_axis] = slice(1, None)
            p_left = tuple(p_left)
            p_right = tuple(p_right)
            face_permeability = np.minimum(self.permeability[p_left], self.permeability[p_right])
            face_permeability *= (~self.solid[p_left]) & (~self.solid[p_right])
            if self._dynamic_barriers is not None:
                face_permeability *= self._dynamic_barriers.faces[flow_component]
            left = current[left_slice]
            right = current[right_slice]
            diffusive_rate = (
                self.diffusion[:, None, None, None]
                * face_permeability[None, ...]
                * (left - right)
                / (spacing * spacing)
            )
            face_velocity = 0.5 * (flow[flow_component][p_left] + flow[flow_component][p_right])
            upwind = np.where(face_velocity[None, ...] >= 0.0, left, right)
            advective_rate = face_permeability[None, ...] * face_velocity[None, ...] * upwind / spacing
            net_to_right = diffusive_rate + advective_rate
            change[left_slice] -= net_to_right
            change[right_slice] += net_to_right
        self.concentration += dt * change
        minimum = float(np.min(self.concentration, initial=0.0))
        if minimum < -1e-10:
            raise FloatingPointError(f"negative concentration {minimum:g}; CFL guard was insufficient")
        np.maximum(self.concentration, 0.0, out=self.concentration)



def field_pair(seed: int, barrier: bool) -> tuple[FieldEnvironment, FieldEnvironment]:
    rng = np.random.default_rng(seed)
    config = {"grid": [48, 32, 14], "flow": [.035, -.022, .007]}
    shape = (14, 32, 48)
    solid = rng.random(shape) < .04
    permeability = rng.uniform(.15, 1, shape)
    reference = ReferenceField(config=config, solid_mask=solid, permeability=permeability)
    native = FieldEnvironment(
        config=config,
        solid_mask=solid, permeability=permeability,
    )
    reference.concentration[:] = rng.exponential(.1, reference.concentration.shape)
    reference.concentration[:, solid] = 0
    native.concentration[:] = reference.concentration
    if barrier:
        record = [{
            "id": "gate", "permeability": .02,
            "translation_epsilon": .001, "rotation_epsilon": .001,
            "shapes": [{"type": "box", "size": [.03, 1.5, 1.0],
                        "position": [6., 4., 1.], "quaternion": [1., 0., 0., 0.]}],
        }]
        reference.sync_dynamic_barriers(record)
        native.sync_dynamic_barriers(record)
    return reference, native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/native-transport.json"))
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not 10 <= args.steps <= 10000:
        parser.error("steps must be in 10..10000")
    receipt = {"kernel": "rust-face-v1", "grid": [48, 32, 14], "channels": 3, "cases": []}
    for barrier in (False, True):
        ref, native = field_pair(371, barrier)
        initial_mass = ref.total_mass.copy()
        for _ in range(args.steps):
            ref._transport(.01, ref.base_flow)
            native._transport(.01, native.base_flow)
        maximum_error = float(np.max(np.abs(ref.concentration - native.concentration)))
        mass_error = float(np.max(np.abs(native.total_mass - initial_mass)))
        assert maximum_error < 1e-12, maximum_error
        assert mass_error < 1e-11, mass_error

        # The full solver includes moving sources, explicit decay, and sinks.
        ref, native = field_pair(817, barrier)
        native_snapshot = json.loads(json.dumps(native.snapshot()))
        restored = FieldEnvironment.restore(native_snapshot)
        legacy = copy.deepcopy(native_snapshot)
        legacy["version"] = 2 if barrier else 1
        del legacy["transport"]
        assert FieldEnvironment.restore(legacy).snapshot() == native_snapshot
        for tick in range(24):
            sources = [{"key": "moving", "position": [2 + .004 * tick, 3., .8],
                        "channel": 0, "rate": .02, "spread": .12}]
            sinks = [{"position": [3., 3., .8], "channel": 1, "rate": .003}]
            a = ref.advance(.05, sources, sinks)
            b = native.advance(.05, sources, sinks)
            c = restored.advance(.05, sources, sinks)
            assert b == c
            assert np.array_equal(native.concentration, restored.concentration)
            np.testing.assert_allclose(a["mass_after"], b["mass_after"], rtol=0, atol=1e-11)
        full_error = float(np.max(np.abs(ref.concentration - native.concentration)))
        assert full_error < 1e-12
        assert native.snapshot() == restored.snapshot()
        damaged = copy.deepcopy(native_snapshot)
        damaged["transport"] = "unknown-implementation"
        try:
            FieldEnvironment.restore(damaged)
        except ValueError:
            pass
        else:
            raise AssertionError("backend/snapshot mismatch accepted")

        # Alternate timing order to reduce warmup/order bias; the denominator
        # includes FFI, validation, and buffer handling, not only inner Rust.
        durations = {"reference": [], "native": []}
        full_durations = {"reference": [], "native": []}
        for repeat in range(6):
            for name in (("reference", "native") if repeat % 2 == 0 else ("native", "reference")):
                world = field_pair(199, barrier)[name == "native"]
                start = time.perf_counter()
                for _ in range(args.steps):
                    world._transport(.01, world.base_flow)
                durations[name].append((time.perf_counter() - start) / args.steps)
                start = time.perf_counter()
                for _ in range(20):
                    world.advance(.05, [{"position": [2., 3., .8], "rate": .02, "spread": .12}])
                full_durations[name].append((time.perf_counter() - start) / 20)
        transport_ms = {key: statistics.median(values) * 1000 for key, values in durations.items()}
        advance_ms = {key: statistics.median(values) * 1000 for key, values in full_durations.items()}
        receipt["cases"].append({
            "barrier": barrier, "transport_max_abs_error": maximum_error,
            "conserved_mass_max_abs_error": mass_error, "full_solver_max_abs_error": full_error,
            "native_restore_exact": True, "transport_median_ms": transport_ms,
            "advance_median_ms": advance_ms,
            "transport_speedup": transport_ms["reference"] / transport_ms["native"],
            "advance_speedup": advance_ms["reference"] / advance_ms["native"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
