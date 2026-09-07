"""One native enzyme-acclimation conservation and persistence check."""

from __future__ import annotations

import json

import numpy as np

from chreatures.metabolism import Chemistry, MetabolicWeb


def main() -> None:
    chemistry = Chemistry(
        {
            "format": "chreatures-common-chemistry-v1",
            "elements": ["C"],
            "enzyme_budget": {"maximum_expression": 0.06, "row_sum": 0.10},
            "pools": [
                {"name": "food", "composition": [1.0], "chemical_energy": 1.0},
                {"name": "waste", "composition": [1.0], "chemical_energy": 0.4},
            ],
            "reactions": [
                {
                    "name": "digest",
                    "consume": {"food": 1.0},
                    "produce": {"waste": 1.0},
                    "half_saturation": 0.1,
                    "atp_cost": 0.0,
                    "atp_yield": 0.4,
                    "photon_cost": 0.0,
                }
            ],
        }
    )
    rule = {
        "baseline": {"digest": 0.02},
        "substrate_response": {"digest": 0.04},
        "atp_response": {"digest": -0.02},
        "time_constant_seconds": 2.0,
        "change_cost_atp_per_expression": 0.5,
    }
    web = MetabolicWeb(
        chemistry,
        [{"digest": 0.02}],
        [{"food": 2.0}],
        [0.5],
        [2.0],
        regulation=[rule],
    )
    elements_before = web.totals()["elements"]
    report = web.step(0.25, [0.0], [0.0])
    enzyme = float(web.enzyme_activity[0, 0])
    regulation_cost = float(report["regulation_atp_cost"][0])
    assert 0.02 < enzyme <= 0.06
    assert regulation_cost > 0.0
    assert abs(float(report["energy_residual"][0])) < 1e-12
    assert elements_before == web.totals()["elements"]

    restored = MetabolicWeb.restore(web.snapshot())
    expected = web.step(0.25, [0.0], [0.0])
    actual = restored.step(0.25, [0.0], [0.0])
    assert all(np.array_equal(expected[key], actual[key]) for key in expected)
    assert np.array_equal(web.enzyme_activity, restored.enzyme_activity)

    expanded = web.expanded(
        [{"digest": 0.01}],
        [1.0],
        [{**rule, "baseline": {"digest": 0.01}}],
    )
    assert np.array_equal(expanded.enzyme_activity[:-1], web.enzyme_activity)
    assert expanded.enzyme_activity[-1, 0] == 0.01
    print(
        json.dumps(
            {
                "enzyme_after_first_step": enzyme,
                "regulation_atp_cost": regulation_cost,
                "energy_residual": float(report["energy_residual"][0]),
                "element_conservation": True,
                "snapshot_continuation_exact": True,
                "expanded_parent_state_exact": True,
                "newborn_initialized_from_baseline": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
