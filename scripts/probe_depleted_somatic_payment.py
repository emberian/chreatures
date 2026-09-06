#!/usr/bin/env python3
"""Reproduce the depleted-ATP somatic partition and atomic cohort debit."""

from __future__ import annotations

import json

import _world_kernels as native
import numpy as np


def metabolic_web(atp: np.ndarray) -> object:
    return native.MetabolicCohort(
        np.array([[-1.0, 1.0]]),
        np.array([[1.0], [1.0]]),
        np.zeros(2),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.array([[1.0, 0.0]]),
        np.zeros(1),
        np.zeros((2, 1)),
        np.zeros((2, 2)),
        atp,
        np.ones(2),
        np.zeros(2),
        0.0,
    )


def main() -> None:
    available = np.array([5.224452869761614e-14], dtype=np.float64)
    dt = 0.05
    traits = np.zeros((1, 18), dtype=np.float64)
    traits[0, 0] = 1.9013879491758178e-14 / dt
    traits[0, 1] = 1.0
    traits[0, 7:11] = 1.0
    traits[0, 11] = 1.0
    traits[0, 14] = 1.0
    somatic = native.SomaticCohort("0" * 64, traits, np.zeros(1))
    payments, scales = somatic.begin(
        np.zeros((1, 12), dtype=np.float32), available, dt,
    )
    payments = np.asarray(payments)
    debit = payments[:, 0] + payments[:, 1]
    if debit[0] > available[0]:
        raise AssertionError("somatic payment exceeds available ATP")

    web = metabolic_web(np.array([available[0], 0.25]))
    before = np.asarray(web.atp).copy()
    try:
        web.pay_work_batch(
            np.array([0, 1], dtype=np.int64),
            np.array([debit[0], 0.5]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid cohort debit was accepted")
    if not np.array_equal(np.asarray(web.atp), before):
        raise AssertionError("failed cohort debit mutated ATP")

    web.pay_work_batch(np.array([0], dtype=np.int64), debit)
    remaining = float(np.asarray(web.atp)[0])
    if not np.isfinite(remaining) or remaining < 0.0:
        raise AssertionError("valid depleted debit produced invalid ATP")
    print(json.dumps({
        "available_atp": float(available[0]),
        "maintenance": float(payments[0, 0]),
        "activation": float(payments[0, 1]),
        "debit": float(debit[0]),
        "remaining_atp": remaining,
        "funded_scale": float(np.asarray(scales)[0]),
        "invalid_batch_atomic": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
