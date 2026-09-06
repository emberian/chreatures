#!/usr/bin/env python3
"""Fit an immutable, budget-constrained population response bank from whole units."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

GAMFIT_VERSION = "0.1.259"
GAMFIT_COMMIT = "7c7eca8ac4826de95c8e743a20294bee132a9bcc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(value: np.ndarray) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode("ascii") if isinstance(scalar, bytes) else str(scalar)


def inverse_target(value: np.ndarray, transform: dict) -> np.ndarray:
    kind = transform["kind"]
    if kind == "positive_softplus":
        ceiling = float(transform["ceiling"])
        clipped = np.clip(value, 1e-9, ceiling - 1e-9)
        return np.log(np.expm1(clipped))
    if kind == "signed_tanh":
        magnitude = float(transform["magnitude"])
        return np.arctanh(np.clip(value / magnitude, -0.999999, 0.999999))
    if kind == "budget_logit":
        return np.log(np.maximum(value, 1e-9))
    raise ValueError(f"unknown response transform {kind}")


def forward_target(value: np.ndarray, transform: dict) -> np.ndarray:
    kind = transform["kind"]
    if kind == "positive_softplus":
        return np.minimum(np.logaddexp(0.0, value), float(transform["ceiling"]))
    if kind == "signed_tanh":
        return float(transform["magnitude"]) * np.tanh(value)
    if kind == "budget_logit":
        return value
    raise ValueError(f"unknown response transform {kind}")


def records(x: np.ndarray, y: np.ndarray, indices: np.ndarray, names: list[str]) -> list[dict]:
    return [{**{name: float(x[row, col]) for col, name in enumerate(names)},
             "response": float(y[row])} for row in indices]


def complete_unit_metrics(predicted: np.ndarray, observed: np.ndarray, indices: np.ndarray,
                          units: dict[str, np.ndarray]) -> dict:
    """Score mean realized response across each whole candidate/episode/world unit."""
    grouped: dict[tuple[str, ...], list[int]] = {}
    for local, row in enumerate(indices):
        key = tuple(str(units[name][row]) for name in
                    ("lineage_unit", "environment_unit", "candidate_unit",
                     "episode_unit", "world_unit"))
        grouped.setdefault(key, []).append(local)
    predicted_means = np.asarray([predicted[rows].mean() for rows in grouped.values()])
    observed_means = np.asarray([observed[rows].mean() for rows in grouped.values()])
    return {
        "complete_candidate_episode_world_units": len(grouped),
        "aggregate_rmse": float(np.sqrt(np.mean((predicted_means - observed_means) ** 2))),
    }


def fit(data_path: Path, schema_path: Path, feature_contract_path: Path, output: Path) -> dict:
    import gamfit
    if gamfit.__version__ != GAMFIT_VERSION or not gamfit.build_info().get("available"):
        raise RuntimeError(f"native gamfit {GAMFIT_VERSION} is required")
    schema = json.loads(schema_path.read_text())
    feature_contract = json.loads(feature_contract_path.read_text())
    if schema.get("format") != "chreatures-population-response-fit-v1":
        raise ValueError("population response fit schema differs")
    if (feature_contract.get("format") != "chreatures-population-response-features-v1"
            or schema.get("feature_contract_sha256") != sha256(feature_contract_path)
            or schema.get("features") != feature_contract.get("features")):
        raise ValueError("population response feature contract or authenticated hash differs")
    data = np.load(data_path)
    x, target = data["features"].astype(float), data["targets"].astype(float)
    names = [item["name"] for item in schema["features"]]
    if x.shape[1] != len(names) or target.ndim != 2 or any(
            not 0 <= int(response["target_column"]) < target.shape[1]
            for response in schema["responses"]):
        raise ValueError("population response data dimensions differ from schema")
    unit_keys = ("lineage_unit", "environment_unit", "candidate_unit", "episode_unit", "world_unit")
    for key in unit_keys:
        if key not in data.files or len(data[key]) != len(x):
            raise ValueError(f"population response data lacks complete-unit axis {key}")
    unit_arrays = {key: np.asarray(data[key]) for key in unit_keys}
    lineage, environment = data["lineage_unit"], data["environment_unit"]
    candidate = data["candidate_unit"]
    test_lineages = np.asarray(schema["split"]["heldout_lineages"], dtype=lineage.dtype)
    test_environments = np.asarray(schema["split"]["heldout_environments"], dtype=environment.dtype)
    test_candidates = np.asarray(schema["split"].get("heldout_candidates", []), dtype=candidate.dtype)
    heldout_lineage = np.isin(lineage, test_lineages)
    heldout_environment = np.isin(environment, test_environments)
    heldout_candidate = np.isin(candidate, test_candidates)
    test = heldout_lineage | heldout_candidate | heldout_environment
    remaining = ~test
    validation = remaining & (data["world_unit"].astype(np.int64) % int(schema["split"].get("validation_world_mod", 5)) == 0)
    train_pool = remaining & ~validation
    stride = int(schema.get("training_tick_stride", 1))
    if "tick_unit" not in data.files or not 1 <= stride <= 64:
        raise ValueError("population response training sampling contract differs")
    train = train_pool & (np.asarray(data["tick_unit"], dtype=np.uint64) % stride == 0)
    if (not train.any() or not test.any()
            or set(test_lineages) & set(lineage[train])
            or set(test_environments) & set(environment[train])):
        raise ValueError("held-out lineage/environment split is empty or leaks units")
    mean, scale = x[train].mean(0), x[train].std(0); scale[scale < 1e-9] = 1.0
    active_features = [i for i in range(x.shape[1])
                       if np.unique(x[train, i]).size >= 3 and x[train, i].std() >= 1e-9]
    if not active_features:
        raise ValueError("population response training features are all constant")
    z = (x - mean) / scale
    lower, upper = np.quantile(x[train], .005, axis=0), np.quantile(x[train], .995, axis=0)
    upper = np.maximum(upper, lower + scale * 1e-6)
    output.mkdir(parents=True, exist_ok=True)
    laws, response_rules, model_report = [], [], {}
    started = time.perf_counter()
    for response_index, response in enumerate(schema["responses"]):
        requested = response.get("smooth_features", names)
        selected_features = [i for i in active_features if names[i] in requested]
        if set(requested) - set(names) or not selected_features:
            raise ValueError(f"smooth feature contract differs for {response['law']}")
        terms = " + ".join(f"s({names[i]}, k={int(schema.get('basis_size', 9))})"
                           for i in selected_features)
        target_column = int(response["target_column"])
        transformed = inverse_target(target[:, target_column], response["transform"])
        formula = f"response ~ {terms}"
        try:
            rows = records(z, transformed, np.flatnonzero(train), names)
            gamfit.validate_formula(rows, formula)
            model = gamfit.fit(rows, formula, family="gaussian")
            model_path = output / f"{response['law']}.gam"; model.save(model_path)
            baseline = {name: 0.0 for name in names}
            intercept = float(model.predict([baseline])[0]); exported = []
            for feature_index in selected_features:
                name = names[feature_index]
                lo = (lower[feature_index] - mean[feature_index]) / scale[feature_index]
                hi = (upper[feature_index] - mean[feature_index]) / scale[feature_index]
                knots = np.linspace(lo, hi, 97)
                curve = [{**baseline, name: float(value)} for value in knots]
                values = np.asarray(model.predict(curve), dtype=float) - intercept
                exported.append({"feature": feature_index, "knots": knots.tolist(), "values": values.tolist()})
            metrics = {}
            for split_name, mask in (("validation", validation),
                                     ("heldout_lineage", heldout_lineage),
                                     ("heldout_candidate", heldout_candidate),
                                     ("heldout_environment", heldout_environment),
                                     ("heldout_union", test)):
                if not mask.any(): continue
                direct_raw = np.asarray(model.predict([{name: float(z[row, col]) for col, name in enumerate(names)}
                    for row in np.flatnonzero(mask)]), dtype=float)
                if response["transform"]["kind"] == "budget_logit":
                    predicted, observed = direct_raw, transformed[mask]
                    baseline_value = np.full(len(observed), transformed[train].mean())
                    metric_scale = "latent log allocation before joint budget softmax"
                else:
                    predicted = forward_target(direct_raw, response["transform"])
                    observed = target[mask, target_column]
                    baseline_value = np.full(len(observed), target[train, target_column].mean())
                    metric_scale = response["unit"]
                metrics[split_name] = {"rows": int(mask.sum()),
                    "scale": metric_scale,
                    "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                    "mean_baseline_rmse": float(np.sqrt(np.mean((baseline_value - observed) ** 2))),
                    **complete_unit_metrics(predicted, observed, np.flatnonzero(mask), unit_arrays)}
            # Calibrate immutable runtime uncertainty on the separately held-out
            # validation worlds. Final candidate/environment holdouts remain
            # reporting-only and never alter a deployed artifact.
            residual = transformed[validation] - np.asarray(model.predict([{name: float(z[row, col]) for col, name in enumerate(names)}
                for row in np.flatnonzero(validation)]), dtype=float)
            laws.append({"name": response["law"], "unit": f"latent_for:{response['unit']}",
                "intercept": intercept, "residual_rmse": float(np.sqrt(np.mean(residual * residual))),
                "target_scale": float(transformed[train].std()),
                "conservative_residual_bound": float(max(np.quantile(np.abs(residual), .995), 3*np.sqrt(np.mean(residual*residual)))),
                "terms": exported})
            response_rules.append({key: response[key] for key in ("law", "mechanism", "unit", "transform")})
            model_report[response["law"]] = {"status": "fitted native GAM", "formula": formula,
                "model_sha256": sha256(model_path), "metrics": metrics,
                "features_excluded_from_smooths": [names[i] for i in range(len(names))
                    if i not in selected_features]}
        except Exception as exc:
            model_report[response["law"]] = {"status": "native fit failed; no law minted",
                "error": f"{type(exc).__name__}: {exc}"[:4000]}
    if len(laws) != len(schema["responses"]):
        report = {"format": "chreatures-population-response-fit-report-v1", "status": "failed; no bank minted",
            "models": model_report, "source_data_sha256": sha256(data_path)}
        (output / "fit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    fitted = {"schema": "chreatures-gam-consequence-law-bank-v1",
        "source": {"model_library": "SauersML/gam", "model_version": GAMFIT_VERSION,
            "model_source_commit": GAMFIT_COMMIT,
            "telemetry_sha256": [scalar_text(data["source_sha256"])],
            "contract": schema["source_contract"]},
        "features": [{**item, "mean": float(mean[i]), "scale": float(scale[i]),
            "minimum": float(lower[i]), "maximum": float(upper[i])} for i, item in enumerate(schema["features"])],
        "laws": laws}
    bank = {"schema": "chreatures-population-response-bank-v1",
        "feature_contract_sha256": schema["feature_contract_sha256"], "fitted": fitted,
        "responses": response_rules, "budgets": schema.get("budgets", []),
        "candidate_score": schema.get("candidate_score")}
    bank_path = output / "population_response_bank.json"
    bank_path.write_text(json.dumps(bank, separators=(",", ":"), allow_nan=False) + "\n")
    report = {"format": "chreatures-population-response-fit-report-v1",
        "status": "candidate for new genome births; no resident promotion", "fit_seconds": time.perf_counter()-started,
        "source_data_sha256": sha256(data_path), "schema_sha256": sha256(schema_path),
        "bank_sha256": sha256(bank_path), "models": model_report,
        "rows": {"train": int(train.sum()), "validation": int(validation.sum()), "heldout": int(test.sum())},
        "training_sampling": {"pool_rows": int(train_pool.sum()), "tick_stride": stride,
            "fitted_rows": int(train.sum()), "rule": "tick modulo stride equals zero per complete fit life"},
        "units": {key: int(np.unique(data[key]).size) for key in unit_keys}}
    (output / "fit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.data, args.schema, args.feature_contract, args.output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
