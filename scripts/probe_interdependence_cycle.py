#!/usr/bin/env python3
"""Exercise one finite decomposer-to-fermentate-consumer material cycle."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco  # noqa: F401  # Preload the wheel's native MuJoCo library.
import numpy as np

from chreatures.metabolism import Chemistry, MetabolicWeb
from chreatures.native_world import load_world_kernels

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    chemistry = Chemistry.load(
        ROOT / "data/metabolism/interdependence-chemistry-v1.json"
    )
    web = MetabolicWeb(
        chemistry,
        [{"detritus_hydrolysis": 0.05}, {}, {"fermentate_respiration": 0.05}],
        [{"detritus": 1.0}, {}, {}],
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
    )
    elements_before = web.totals()["elements"]
    web.step(1.0, np.zeros(3), np.zeros(3))
    fermentate = chemistry.pools.index("fermentate")
    rates = np.zeros((1, 2, len(chemistry.pools)))
    rates[0, 0, fermentate] = 1.0
    vectors, _, _, mass = load_world_kernels().mobile_release_candidates(
        1.0,
        np.zeros(1), np.zeros(1), np.ones(1),
        np.ascontiguousarray(web.pools[[[0, 1]]]), rates,
        np.ones(1), np.array([1e-9]), np.ones(1),
        np.ascontiguousarray(chemistry._arrays[1].sum(axis=1)),
    )
    released = np.asarray(vectors)[0, 0]
    web.transfer(0, 1, dict(zip(chemistry.pools, released.tolist(), strict=True)))
    web.transfer(1, 2, dict(zip(chemistry.pools, web.pools[1].tolist(), strict=True)))
    atp_before = float(web.atp[2])
    web.step(1.0, np.zeros(3), np.zeros(3))
    atp_after = float(web.atp[2])
    elements_after = web.totals()["elements"]
    if released[fermentate] <= 0.0 or atp_after <= atp_before:
        raise AssertionError("finite fermentate cross-feeding did not complete")
    if any(
        abs(elements_after[name] - value) > 1e-12
        for name, value in elements_before.items()
    ):
        raise AssertionError("cross-feeding changed conserved elements")
    print(json.dumps({
        "released_fermentate": float(released[fermentate]),
        "native_release_mass": float(np.asarray(mass)[0]),
        "consumer_atp_gain": atp_after - atp_before,
        "elements_before": elements_before,
        "elements_after": elements_after,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
