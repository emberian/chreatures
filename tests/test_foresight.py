import json
from types import SimpleNamespace

import numpy as np

from chreatures.foresight import ForesightConfig, ResidentForesight
from chreatures.homeostasis import FiniteEnergyConfig, FiniteEnergyObjective


def _organ(prediction):
    organ = ResidentForesight.__new__(ResidentForesight)
    organ.config = ForesightConfig(seed=17)
    organ.homeostasis = FiniteEnergyConfig()
    organ.objective = FiniteEnergyObjective(organ.homeostasis)
    organ.rng = np.random.default_rng(17)
    organ.observation_count = 1
    organ.last_observed_features = np.zeros(2, dtype=np.float32)
    organ.last_observed_physiology = np.array(
        [0.72, 0.1, 0.08, 0.0, 0.0, 1.0], dtype=np.float32,
    )
    organ.last_executed_action = np.zeros(8, dtype=np.float32)
    organ.intention_tail = None
    organ._query_cache = None
    organ.experienced = SimpleNamespace(
        feature_dim=2,
        physiology_dim=6,
        metadata={"forecast_status": "regression fixture"},
        model_identity={"artifact_sha256": "a" * 64},
        input_identity={"sha256": "b" * 64},
        snapshot=lambda: {"frozen": True},
    )
    organ.query_cohort = SimpleNamespace(
        query_from_snapshot=lambda snapshot, actions: prediction,
    )
    return organ


def _prediction(candidate_count=2):
    horizon = 8
    branches = candidate_count * 4
    physiology = np.tile(
        np.array([0.73, 0.11, 0.07, 0.0, 0.0, 1.0], dtype=np.float32),
        (horizon, branches, 1),
    )
    return {
        "feature_mean": np.zeros((horizon, branches, 2), dtype=np.float32),
        "feature_residual_scale": np.ones((horizon, branches, 2), dtype=np.float32),
        "physiology_mean": physiology,
        "physiology_residual_scale": np.full_like(physiology, 0.02),
        "valid": np.ones((horizon, branches), dtype=np.bool_),
        "horizon_support": np.full((horizon, branches), 0.8, dtype=np.float32),
    }


def test_mixed_invalid_forecasts_are_rejected_before_potential_and_serialize():
    prediction = _prediction()
    # Candidate zero: one valid, then negative, >1, and nonfinite physical branches.
    prediction["physiology_mean"][:, 1, 0] = -0.01
    prediction["physiology_mean"][:, 2, 1] = 1.01
    prediction["physiology_mean"][:, 3, 2] = np.nan
    # Candidate one: retain one valid branch; reject bad residual/support metadata too.
    prediction["physiology_residual_scale"][:, 5, 0] = np.inf
    prediction["horizon_support"][:, 6] = np.nan
    prediction["valid"][:, 7] = False
    report = _organ(prediction).candidate_evidence(
        (tuple(np.zeros(8)), tuple(np.full(8, 0.1))),
    )
    assert [item["valid_branches"] for item in report["diagnostics"]] == [1, 1]
    assert np.isfinite(report["corrections"]).all()
    json.dumps(report, allow_nan=False)


def test_all_invalid_candidate_has_zero_correction_and_null_diagnostics():
    prediction = _prediction()
    prediction["physiology_mean"][:, :, 0] = np.nan
    report = _organ(prediction).candidate_evidence(
        (tuple(np.zeros(8)), tuple(np.full(8, 0.1))),
    )
    assert report["corrections"] == [0.0, 0.0]
    for diagnostic in report["diagnostics"]:
        assert diagnostic["forecast_score"] is None
        assert diagnostic["best_branch"] is None
        assert diagnostic["physiology_residual_scale_mean"] == [None, None, None]
        assert diagnostic["horizon_support_mean"] is None
    json.dumps(report, allow_nan=False)
