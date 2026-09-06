#!/usr/bin/env python3
"""Fit additive nonlinear consequence laws from body-local experienced transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

GAMFIT_VERSION = "0.1.259"
GAMFIT_SOURCE_COMMIT = "7c7eca8ac4826de95c8e743a20294bee132a9bcc"
FEATURES = (
    ("energy", "fraction"), ("fatigue", "fraction"),
    ("body_speed", "tanh(speed_m_s/2)"), ("support", "circuit_rate"),
    ("neural_activity", "mean_rate"), ("thrust", "command"),
    ("yaw", "command"), ("grip", "command"), ("oral", "command"),
    ("motor_magnitude", "mean_abs_command"),
    ("thrust_x_fatigue", "command*fraction"),
    ("yaw_x_speed", "command*tanh(speed_m_s/2)"),
)
OUTCOMES = (
    ("movement_response", "delta_tanh_speed_per_tick"),
    ("energy_cost", "energy_fraction_lost_per_tick"),
    ("fatigue_recovery", "fatigue_fraction_recovered_per_tick"),
)
FORMULA = "response ~ " + " + ".join(f"s({name}, k=9)" for name, _ in FEATURES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(episodes: list[Path], output: Path) -> None:
    columns, outcomes, units = [], [], []
    source_hashes = []
    for episode_index, path in enumerate(episodes):
        data = np.load(path)
        physiology = data["physiology"].astype(np.float64)
        action = data["executed_actions"].astype(np.float64)
        oral = data["oral_command"].astype(np.float64)
        neural = data["neural_readouts"].astype(np.float64)
        count = action.shape[1]
        state = physiology[:-1]
        row_features = np.stack((
            state[..., 0], state[..., 2], state[..., 3], state[..., 5],
            neural[:-1].mean(axis=-1), action[..., 0], action[..., 1],
            action[..., 3], oral, np.abs(action[..., :4]).mean(axis=-1),
            action[..., 0] * state[..., 2], action[..., 1] * state[..., 3],
        ), axis=-1)
        row_outcomes = np.stack((
            physiology[1:, :, 3] - state[..., 3],
            state[..., 0] - physiology[1:, :, 0],
            state[..., 2] - physiology[1:, :, 2],
        ), axis=-1)
        columns.append(row_features.reshape(-1, len(FEATURES)).astype(np.float32))
        outcomes.append(row_outcomes.reshape(-1, len(OUTCOMES)).astype(np.float32))
        # Three anonymous residents share one independently seeded physical world.
        world = np.arange(count, dtype=np.int32) // 3 + episode_index * (count // 3)
        units.append(np.broadcast_to(world, action.shape[:2]).reshape(-1))
        source_hashes.append(sha256(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=np.concatenate(columns), outcomes=np.concatenate(outcomes),
                       world_unit=np.concatenate(units), source_sha256=np.asarray(source_hashes))


def rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def fit(compact_path: Path, output_dir: Path) -> dict:
    import gamfit
    if gamfit.__version__ != GAMFIT_VERSION or not gamfit.build_info().get("available"):
        raise RuntimeError(f"requires native gamfit {GAMFIT_VERSION}")
    data = np.load(compact_path)
    x, y, units = data["features"].astype(float), data["outcomes"].astype(float), data["world_unit"]
    test = units % 4 == 0
    train = ~test
    means = x[train].mean(axis=0)
    scales = x[train].std(axis=0)
    scales[scales < 1e-9] = 1.0
    z = (x - means) / scales
    lower = np.quantile(x[train], 0.005, axis=0)
    upper = np.quantile(x[train], 0.995, axis=0)
    # Avoid duplicate bounds for nearly discrete channels.
    upper = np.maximum(upper, lower + scales * 1e-6)
    names = [name for name, _ in FEATURES]
    train_rows = [{name: float(z[row, col]) for col, name in enumerate(names)}
                  for row in np.flatnonzero(train)]
    test_rows = [{name: float(z[row, col]) for col, name in enumerate(names)}
                 for row in np.flatnonzero(test)]
    laws, metrics = [], {}
    started = time.perf_counter()
    for outcome_index, (outcome_name, unit) in enumerate(OUTCOMES):
        rows = [dict(row, response=float(y[index, outcome_index]))
                for row, index in zip(train_rows, np.flatnonzero(train), strict=True)]
        gamfit.validate_formula(rows, FORMULA)
        model = gamfit.fit(rows, FORMULA)
        direct_test = np.asarray(model.predict(test_rows), dtype=float)
        baseline_row = {name: 0.0 for name in names}
        intercept = float(model.predict([baseline_row])[0])
        terms = []
        for feature_index, name in enumerate(names):
            lo = (lower[feature_index] - means[feature_index]) / scales[feature_index]
            hi = (upper[feature_index] - means[feature_index]) / scales[feature_index]
            knots = np.linspace(lo, hi, 97)
            grid_rows = [dict(baseline_row, **{name: float(value)}) for value in knots]
            values = np.asarray(model.predict(grid_rows), dtype=float) - intercept
            terms.append({"feature": feature_index, "knots": knots.tolist(), "values": values.tolist()})
        approximate = np.full(test.sum(), intercept)
        for term in terms:
            approximate += np.interp(z[test, term["feature"]], term["knots"], term["values"])
        observed = y[test, outcome_index]
        train_mean = float(y[train, outcome_index].mean())
        law_rmse = rmse(direct_test, observed)
        interpolation_error = float(np.max(np.abs(approximate - direct_test)))
        in_domain = np.all((x[test] >= lower) & (x[test] <= upper), axis=1)
        in_domain_error = float(np.max(np.abs(approximate[in_domain] - direct_test[in_domain])))
        absolute_residual = np.abs(observed - direct_test)
        residual_bound = max(float(np.quantile(absolute_residual, 0.995)), 3.0 * law_rmse)
        laws.append({"name": outcome_name, "unit": unit, "intercept": intercept,
                     "residual_rmse": law_rmse,
                     "target_scale": float(y[train, outcome_index].std()),
                     "conservative_residual_bound": residual_bound, "terms": terms})
        metrics[outcome_name] = {
            "held_out_rmse": law_rmse,
            "held_out_mean_baseline_rmse": rmse(np.full(test.sum(), train_mean), observed),
            "held_out_zero_baseline_rmse": rmse(np.zeros(test.sum()), observed),
            "export_grid_max_abs_error": interpolation_error,
            "in_domain_export_grid_max_abs_error": in_domain_error,
            "in_domain_held_out_rows": int(in_domain.sum()),
            "held_out_abs_residual_q995": float(np.quantile(absolute_residual, 0.995)),
            "conservative_residual_bound": residual_bound,
            "target_training_standard_deviation": float(y[train, outcome_index].std()),
        }
    artifact = {
        "schema": "chreatures-gam-consequence-law-bank-v1",
        "source": {"model_library": "SauersML/gam", "model_version": GAMFIT_VERSION,
                   "model_source_commit": GAMFIT_SOURCE_COMMIT,
                   "telemetry_sha256": data["source_sha256"].tolist()},
        "features": [{"name": name, "unit": unit, "mean": float(means[i]),
                      "scale": float(scales[i]), "minimum": float(lower[i]),
                      "maximum": float(upper[i])} for i, (name, unit) in enumerate(FEATURES)],
        "laws": laws,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "body_consequence_laws.json"
    artifact_path.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n")
    report = {
        "schema": "chreatures-gam-consequence-fit-report-v1",
        "status": "conditional predictive fits; no causal claim",
        "formula": FORMULA, "fit_seconds": time.perf_counter() - started,
        "prediction_rows": {"train": int(train.sum()), "held_out": int(test.sum())},
        "independent_world_units": {"train": int(np.unique(units[train]).size),
                                    "held_out": int(np.unique(units[test]).size)},
        "split": "complete physical worlds with world_unit modulo 4 equal to zero held out",
        "artifact": {"path": artifact_path.name, "bytes": artifact_path.stat().st_size,
                     "sha256": sha256(artifact_path)}, "metrics": metrics,
        "limitations": [
            "The fits estimate conditional responses in experienced exploratory behavior, not interventions or causes.",
            "The source is the completed fresh-world sensorimotor-play collection used to bootstrap development; the later online run retained only aggregate update telemetry.",
            "Resident/world grouping is used only for evaluation splitting and is absent from runtime features.",
            "Grid inference is a measured piecewise-linear approximation of additive native GAM predictions.",
        ],
    }
    (output_dir / "fit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=Path, default=[])
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.episode:
        compact(args.episode, args.compact)
    if args.output_dir:
        print(json.dumps(fit(args.compact, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
