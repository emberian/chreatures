#!/usr/bin/env python3
"""Fit an offline native-GAM response surface for CNS dynamics V2.

This is an analyst calibration tool.  Sweep outcomes are supervised research
targets; neither the parameters nor their outcomes are resident observations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


GAMFIT_VERSION = "0.1.259"
GAMFIT_SOURCE_COMMIT = "7c7eca8ac4826de95c8e743a20294bee132a9bcc"
REPORT_FORMAT = "chreatures-cns-dynamics-v2-gam-response-v1"
EXPECTED_LEVELS = {
    "gain": (0.7, 1.05, 1.4),
    "tau": (0.04, 0.08, 0.16),
    "adaptation_gain": (0.05, 0.15, 0.35),
}
FEATURES = tuple(EXPECTED_LEVELS)
TARGETS = ("skill", "saturation", "recovery_memory")
REQUIRED_COLUMNS = (
    *FEATURES,
    "temporal_probe_mse",
    "skill",
    "persistence_mse",
    "recovery_memory",
    "saturation",
)
TENSOR_FORMULA = (
    "response ~ te(gain_scaled,tau_scaled,adaptation_gain_scaled,k=15)"
)
ADDITIVE_FORMULA = (
    "response ~ s(gain_scaled,k=3)+s(tau_scaled,k=3)"
    "+s(adaptation_gain_scaled,k=3)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _canonical_level(value: float, name: str) -> float:
    for level in EXPECTED_LEVELS[name]:
        if math.isclose(value, level, rel_tol=0.0, abs_tol=1e-10):
            return level
    raise ValueError(
        f"{name}={value!r} is outside the frozen levels {EXPECTED_LEVELS[name]}"
    )


def load_sweep(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Load and strictly validate the complete crossed-design sweep."""
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("sweep JSON root must be an object")
    source_format = document.get("format")
    if not isinstance(source_format, str) or not source_format:
        raise ValueError("sweep JSON requires a nonempty string `format`")
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("sweep JSON requires a `records` array")

    records: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for index, source in enumerate(raw_records):
        if not isinstance(source, dict):
            raise ValueError(f"records[{index}] must be an object")
        missing = [name for name in REQUIRED_COLUMNS if name not in source]
        if missing:
            raise ValueError(f"records[{index}] lacks required columns {missing}")
        record: dict[str, Any] = {
            name: _finite_float(source[name], f"records[{index}].{name}")
            for name in REQUIRED_COLUMNS
        }
        for name in FEATURES:
            record[name] = _canonical_level(record[name], name)
        if "run_id" in source:
            if not isinstance(source["run_id"], (str, int)):
                raise ValueError(f"records[{index}].run_id must be a string or integer")
            record["run_id"] = source["run_id"]
        if record["temporal_probe_mse"] < 0.0:
            raise ValueError(f"records[{index}].temporal_probe_mse must be nonnegative")
        if record["persistence_mse"] <= 0.0:
            raise ValueError(f"records[{index}].persistence_mse must be positive")
        if record["recovery_memory"] < 0.0:
            raise ValueError(f"records[{index}].recovery_memory must be nonnegative")
        if not 0.0 <= record["saturation"] <= 1.0:
            raise ValueError(f"records[{index}].saturation must lie in [0,1]")
        setting = tuple(record[name] for name in FEATURES)
        if setting in seen:
            raise ValueError(f"duplicate parameter configuration {setting}")
        seen.add(setting)
        records.append(record)

    expected = set(itertools.product(*(EXPECTED_LEVELS[name] for name in FEATURES)))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            "sweep must contain the complete 27-setting 3x3x3 design; "
            f"missing={missing}, extra={extra}"
        )
    records.sort(key=lambda row: tuple(row[name] for name in FEATURES))
    return source_format, records


def _require_native_gamfit() -> Any:
    try:
        import gamfit
    except ImportError as exc:
        raise RuntimeError(
            "native gamfit is absent from this Python environment. Run with "
            "`integrations/.venv/bin/python research/dynamics_v2/gam_fit.py ...`; "
            "to rebuild that isolated environment use `uv venv integrations/.venv "
            "&& uv pip install --python integrations/.venv/bin/python "
            "-r integrations/gamfit-requirements.txt`."
        ) from exc
    if gamfit.__version__ != GAMFIT_VERSION:
        raise RuntimeError(
            f"expected gamfit {GAMFIT_VERSION}, loaded {gamfit.__version__} "
            f"from {Path(gamfit.__file__).resolve()}"
        )
    build = gamfit.build_info()
    if not build.get("available"):
        raise RuntimeError(f"gamfit native extension is unavailable: {build}")
    return gamfit


def _scaled_parameters(record: dict[str, Any]) -> dict[str, float]:
    scaled: dict[str, float] = {}
    for name in FEATURES:
        levels = EXPECTED_LEVELS[name]
        midpoint = levels[1]
        half_range = (levels[-1] - levels[0]) / 2.0
        scaled[f"{name}_scaled"] = (float(record[name]) - midpoint) / half_range
    return scaled


def _fit_rows(records: list[dict[str, Any]], target: str) -> list[dict[str, float]]:
    return [{**_scaled_parameters(row), "response": float(row[target])} for row in records]


def _predict_rows(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [_scaled_parameters(row) for row in records]


def _capture_native_stderr(callback: Callable[[], Any]) -> tuple[Any, list[str]]:
    """Keep repeated REML diagnostics in the report instead of flooding stderr."""
    with tempfile.TemporaryFile() as capture:
        saved_stderr = os.dup(2)
        try:
            os.dup2(capture.fileno(), 2)
            result = callback()
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        capture.seek(0)
        lines = capture.read().decode(errors="replace").strip().splitlines()
    return result, lines


def _metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r_squared": float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0.0
        else 1.0,
    }


def _loo(
    gamfit: Any,
    records: list[dict[str, Any]],
    target: str,
    formula: str,
) -> tuple[dict[str, Any], list[str]]:
    observed = np.asarray([row[target] for row in records], dtype=float)
    predicted = np.empty(len(records), dtype=float)
    mean_baseline = np.empty(len(records), dtype=float)
    native_messages: list[str] = []
    for held_out in range(len(records)):
        training = [row for index, row in enumerate(records) if index != held_out]
        fit_rows = _fit_rows(training, target)
        gamfit.validate_formula(fit_rows, formula)
        model, messages = _capture_native_stderr(
            lambda fit_rows=fit_rows: gamfit.fit(
                fit_rows, formula, family="gaussian"
            )
        )
        native_messages.extend(messages)
        predicted[held_out] = float(
            model.predict([_scaled_parameters(records[held_out])])[0]
        )
        mean_baseline[held_out] = float(np.mean([row[target] for row in training]))
    configuration_predictions = []
    for row, value in zip(records, predicted, strict=True):
        configuration_predictions.append(
            {
                **{name: row[name] for name in FEATURES},
                "observed": row[target],
                "predicted": float(value),
            }
        )
    return (
        {
            "split": "leave one complete parameter configuration out per fold",
            "folds": len(records),
            "leaked_parameter_configurations": 0,
            "metrics": _metrics(predicted, observed),
            "training_mean_baseline": _metrics(mean_baseline, observed),
            "configuration_predictions": configuration_predictions,
        },
        native_messages,
    )


def _warning_summary(messages: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for message in messages:
        label = "native_message"
        if "[" in message and "]" in message:
            label = message.split("[", 1)[1].split("]", 1)[0]
        counts[label] = counts.get(label, 0) + 1
    return {
        "count": len(messages),
        "categories": dict(sorted(counts.items())),
        "first_messages": messages[:8],
    }


def _fit_target(
    gamfit: Any,
    records: list[dict[str, Any]],
    target: str,
    output_dir: Path,
) -> tuple[dict[str, Any], Any | None]:
    observed = np.asarray([row[target] for row in records], dtype=float)
    result: dict[str, Any] = {
        "target": target,
        "observed_min": float(observed.min()),
        "observed_max": float(observed.max()),
    }
    if float(np.ptp(observed)) <= 1e-12:
        result.update(
            {
                "status": "constant response; no GAM surface identifiable",
                "constant": float(observed[0]),
                "loo": {
                    "split": "leave one complete parameter configuration out per fold",
                    "folds": len(records),
                    "metrics": {"rmse": 0.0, "mae": 0.0, "r_squared": 1.0},
                    "training_mean_baseline": {
                        "rmse": 0.0,
                        "mae": 0.0,
                        "r_squared": 1.0,
                    },
                },
            }
        )
        return result, None

    loo, messages = _loo(gamfit, records, target, TENSOR_FORMULA)
    fit_rows = _fit_rows(records, target)
    gamfit.validate_formula(fit_rows, TENSOR_FORMULA)
    model, full_messages = _capture_native_stderr(
        lambda: gamfit.fit(fit_rows, TENSOR_FORMULA, family="gaussian")
    )
    messages.extend(full_messages)
    model_path = output_dir / f"{target}.gam"
    model.save(model_path)
    reloaded = gamfit.load(model_path)
    predictors = _predict_rows(records)
    direct = np.asarray(model.predict(predictors), dtype=float)
    restored = np.asarray(reloaded.predict(predictors), dtype=float)
    result.update(
        {
            "status": "fitted native GAM",
            "formula": TENSOR_FORMULA,
            "surface": "three-axis tensor smooth containing joint gain/tau/adaptation response",
            "loo": loo,
            "heldout_predictive_useful_against_training_mean": (
                loo["metrics"]["rmse"]
                < loo["training_mean_baseline"]["rmse"]
            ),
            "training_reconstruction": _metrics(direct, observed),
            "model": {
                "file": model_path.name,
                "bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
                "reload_max_abs_prediction_delta": float(
                    np.max(np.abs(direct - restored))
                ),
            },
            "native_diagnostics": _warning_summary(messages),
        }
    )
    return result, reloaded


def _fit_skill_additive_benchmark(
    gamfit: Any, records: list[dict[str, Any]], output_dir: Path
) -> tuple[dict[str, Any], Any | None]:
    observed = np.asarray([row["skill"] for row in records], dtype=float)
    if float(np.ptp(observed)) <= 1e-12:
        return {"status": "constant response; additive benchmark is identical"}, None
    loo, messages = _loo(gamfit, records, "skill", ADDITIVE_FORMULA)
    fit_rows = _fit_rows(records, "skill")
    gamfit.validate_formula(fit_rows, ADDITIVE_FORMULA)
    model, full_messages = _capture_native_stderr(
        lambda: gamfit.fit(fit_rows, ADDITIVE_FORMULA, family="gaussian")
    )
    messages.extend(full_messages)
    model_path = output_dir / "skill_additive.gam"
    model.save(model_path)
    return (
        {
            "status": "fitted native GAM benchmark",
            "formula": ADDITIVE_FORMULA,
            "loo": loo,
            "model": {
                "file": model_path.name,
                "bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
            },
            "native_diagnostics": _warning_summary(messages),
        },
        gamfit.load(model_path),
    )


def _model_predictions(
    model: Any | None,
    model_report: dict[str, Any],
    records: list[dict[str, Any]],
    target: str,
) -> np.ndarray:
    if model is None:
        return np.full(len(records), float(model_report["constant"]), dtype=float)
    predicted = np.asarray(model.predict(_predict_rows(records)), dtype=float)
    if target == "saturation":
        return np.clip(predicted, 0.0, 1.0)
    if target == "recovery_memory":
        return np.maximum(predicted, 0.0)
    return predicted


def _query_optimum(
    models: dict[str, Any | None],
    reports: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    grid_points: int,
    saturation_limit: float,
) -> dict[str, Any]:
    axes = {
        name: np.linspace(levels[0], levels[-1], grid_points)
        for name, levels in EXPECTED_LEVELS.items()
    }
    query = [
        dict(zip(FEATURES, values, strict=True))
        for values in itertools.product(*(axes[name] for name in FEATURES))
    ]
    predictions = {
        target: _model_predictions(models[target], reports[target], query, target)
        for target in TARGETS
    }
    eligible = np.flatnonzero(predictions["saturation"] <= saturation_limit)
    result: dict[str, Any] = {
        "method": "dense interpolation over the measured parameter box",
        "points_per_axis": grid_points,
        "queried_points": len(query),
        "objective": "maximize predicted skill",
        "constraint": f"predicted saturation <= {saturation_limit}",
        "recovery_memory_role": "reported diagnostic only; absent from objective",
        "eligible_points": int(eligible.size),
    }
    if not eligible.size:
        result["status"] = "no predicted point satisfies the saturation constraint"
        result["recommended"] = None
    else:
        best = int(eligible[np.argmax(predictions["skill"][eligible])])
        result["status"] = "candidate identified within observed parameter bounds"
        result["recommended"] = {
            **{name: float(query[best][name]) for name in FEATURES},
            "predicted_skill": float(predictions["skill"][best]),
            "predicted_saturation": float(predictions["saturation"][best]),
            "predicted_recovery_memory": float(predictions["recovery_memory"][best]),
        }

        # A second full-graph run is most informative at a genuinely unmeasured
        # interpolation, rather than at the already observed optimum or one
        # floating-point grid step away. Distances use the declared scaled axes.
        observed_scaled = np.asarray(
            [
                [
                    _scaled_parameters(row)[f"{name}_scaled"]
                    for name in FEATURES
                ]
                for row in records
            ],
            dtype=float,
        )
        query_scaled = np.asarray(
            [
                [
                    _scaled_parameters(row)[f"{name}_scaled"]
                    for name in FEATURES
                ]
                for row in query
            ],
            dtype=float,
        )
        nearest_distance = np.min(
            np.sqrt(np.sum((query_scaled[:, None, :] - observed_scaled[None, :, :]) ** 2, axis=2)),
            axis=1,
        )
        confirmation_pool = eligible[nearest_distance[eligible] >= 0.20]
        if confirmation_pool.size:
            confirmation = int(
                confirmation_pool[
                    np.argmax(predictions["skill"][confirmation_pool])
                ]
            )
            skill_rmse = float(reports["skill"]["loo"]["metrics"]["rmse"])
            skill_prediction = float(predictions["skill"][confirmation])
            result["confirmatory_unmeasured_setting"] = {
                **{name: float(query[confirmation][name]) for name in FEATURES},
                "predicted_skill": skill_prediction,
                "predicted_saturation": float(predictions["saturation"][confirmation]),
                "predicted_recovery_memory": float(
                    predictions["recovery_memory"][confirmation]
                ),
                "nearest_measured_scaled_distance": float(
                    nearest_distance[confirmation]
                ),
                "skill_uncertainty": {
                    "leave_one_configuration_out_rmse": skill_rmse,
                    "two_rmse_heuristic_range": [
                        skill_prediction - 2.0 * skill_rmse,
                        skill_prediction + 2.0 * skill_rmse,
                    ],
                    "interpretation": (
                        "Empirical prediction-error scale, not a calibrated confidence interval."
                    ),
                },
                "recovery_prediction_reliable_against_mean": bool(
                    reports["recovery_memory"].get(
                        "heldout_predictive_useful_against_training_mean", False
                    )
                ),
                "required_next_step": (
                    "Run this exact setting on the full graph; the fitted value does not "
                    "establish improvement over the best measured setting."
                ),
            }

    observed_eligible = [
        row for row in records if float(row["saturation"]) <= saturation_limit
    ]
    if observed_eligible:
        observed_best = max(observed_eligible, key=lambda row: float(row["skill"]))
        result["best_measured_setting"] = {
            **{name: observed_best[name] for name in FEATURES},
            **{target: observed_best[target] for target in TARGETS},
            "temporal_probe_mse": observed_best["temporal_probe_mse"],
            "persistence_mse": observed_best["persistence_mse"],
        }
    else:
        result["best_measured_setting"] = None
    return result


def fit_response_surface(
    input_path: Path,
    output_dir: Path,
    *,
    grid_points: int = 25,
    saturation_limit: float = 0.01,
) -> dict[str, Any]:
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    if not 0.0 <= saturation_limit <= 1.0:
        raise ValueError("saturation_limit must lie in [0,1]")
    source_format, records = load_sweep(input_path)
    gamfit = _require_native_gamfit()
    output_dir.mkdir(parents=True, exist_ok=True)
    reserved = [output_dir / f"{target}.gam" for target in TARGETS]
    reserved.extend((output_dir / "skill_additive.gam", output_dir / "fit_report.json"))
    occupied = [str(path) for path in reserved if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite existing fit outputs: {occupied}")

    started = time.perf_counter()
    reports: dict[str, dict[str, Any]] = {}
    models: dict[str, Any | None] = {}
    for target in TARGETS:
        reports[target], models[target] = _fit_target(
            gamfit, records, target, output_dir
        )
    additive, _ = _fit_skill_additive_benchmark(gamfit, records, output_dir)
    if reports["skill"]["status"] == "fitted native GAM":
        tensor_rmse = reports["skill"]["loo"]["metrics"]["rmse"]
        additive_rmse = additive["loo"]["metrics"]["rmse"]
        interaction_evidence = {
            "comparison": "three-axis tensor smooth versus additive smooths",
            "tensor_loo_rmse": tensor_rmse,
            "additive_loo_rmse": additive_rmse,
            "rmse_reduction_from_joint_surface": additive_rmse - tensor_rmse,
            "joint_surface_supported_by_heldout_prediction": tensor_rmse
            < additive_rmse,
            "interpretation": (
                "Predictive comparison only; a positive reduction supports using the joint "
                "surface for interpolation but does not identify a physiological interaction."
            ),
        }
    else:
        interaction_evidence = {
            "joint_surface_supported_by_heldout_prediction": False,
            "interpretation": "Skill did not vary, so an interaction is not identifiable.",
        }

    build = gamfit.build_info()
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "status": "complete",
        "interpretation": (
            "Offline descriptive interpolation of full-graph response metrics; no causal or "
            "physiological parameter claim and no resident controller input."
        ),
        "source": {
            "file": str(input_path.resolve()),
            "format": source_format,
            "sha256": _sha256(input_path),
            "records": len(records),
        },
        "library": {
            "name": "gamfit",
            "version": gamfit.__version__,
            "source_commit": GAMFIT_SOURCE_COMMIT,
            "native_extension_available": True,
            "native_crate": build.get("crate"),
            "native_engine_crate": build.get("engine_crate"),
            "abi": build.get("abi3"),
        },
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.system().lower(),
            "fit_seconds": time.perf_counter() - started,
        },
        "design": {
            "kind": "complete crossed 3x3x3 full-graph sweep",
            "levels": {name: list(levels) for name, levels in EXPECTED_LEVELS.items()},
            "normalization": {
                name: {
                    "center": levels[1],
                    "half_range": (levels[-1] - levels[0]) / 2.0,
                }
                for name, levels in EXPECTED_LEVELS.items()
            },
            "skill_definition": (
                "source sweep's mean three-channel 1 - temporal_probe_mse / "
                "persistence_mse; stored skill is authoritative because a ratio of means "
                "need not equal the mean of channel ratios"
            ),
            "recovery_memory_definition": (
                "normalized neural residual RMS one second after neutral input divided by "
                "last driven RMS"
            ),
            "saturation_definition": "fraction of sampled abs(r-r0) greater than 0.19",
        },
        "models": reports,
        "skill_additive_benchmark": additive,
        "interaction_evidence": interaction_evidence,
        "selection": _query_optimum(
            models, reports, records, grid_points, saturation_limit
        ),
        "limitations": [
            "Twenty-seven configurations support bounded interpolation, not a recovered biological law.",
            "Leave-one-configuration-out folds share stimulus construction and graph identity.",
            "The dense optimum must be rerun as a full-graph setting before adoption.",
            "Recovery memory is diagnostic persistence and is never maximized as task value.",
        ],
    }
    report_path = output_dir / "fit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="crossed-sweep JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-points", type=int, default=25)
    parser.add_argument(
        "--saturation-limit",
        type=float,
        default=0.01,
        help="maximum predicted saturated fraction for parameter selection",
    )
    args = parser.parse_args()
    report = fit_response_surface(
        args.input,
        args.output_dir,
        grid_points=args.grid_points,
        saturation_limit=args.saturation_limit,
    )
    print(json.dumps({
        "status": report["status"],
        "report": str((args.output_dir / "fit_report.json").resolve()),
        "selection": report["selection"],
        "interaction_evidence": report["interaction_evidence"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
