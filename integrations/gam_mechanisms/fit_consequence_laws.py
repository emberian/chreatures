#!/usr/bin/env python3
"""Fit and native-verify current body consequence laws from rich-v4 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


GAMFIT_VERSION = "0.1.259"
GAMFIT_SOURCE_COMMIT = "7c7eca8ac4826de95c8e743a20294bee132a9bcc"
FEATURES = (
    ("energy", "fraction"),
    ("fatigue", "fraction"),
    ("body_speed", "tanh(speed_m_s/2)"),
    ("support", "circuit_rate"),
    ("neural_activity", "mean_rate"),
    ("thrust", "command"),
    ("yaw", "command"),
    ("grip", "command"),
    ("oral", "eat_command"),
    ("motor_magnitude", "mean_abs_thrust_yaw_gaze_grip"),
    ("thrust_x_fatigue", "command*fraction"),
    ("yaw_x_speed", "command*tanh(speed_m_s/2)"),
)
OUTCOMES = (
    ("movement_response", "delta_tanh_speed_per_tick"),
    ("energy_cost", "energy_fraction_lost_per_tick"),
    ("fatigue_recovery", "fatigue_fraction_recovered_per_tick"),
)
FORMULA = "response ~ " + " + ".join(
    f"s(x{index}, k=9)" for index in range(len(FEATURES))
)
PARTITIONS = {"train": 0, "validation": 1, "holdout": 2}
EXPECTED_SLOTS = {
    "train": tuple(range(8)), "validation": (8,), "holdout": (9,),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(value: np.ndarray) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode("ascii") if isinstance(scalar, bytes) else str(scalar)


def write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        if compact:
            json.dump(value, handle, separators=(",", ":"), allow_nan=False)
        else:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def aggregate_unit_metrics(
    predicted: np.ndarray, observed: np.ndarray, unit: np.ndarray
) -> dict[str, Any]:
    keys = np.unique(unit)
    predicted_mean = np.asarray([predicted[unit == key].mean() for key in keys])
    observed_mean = np.asarray([observed[unit == key].mean() for key in keys])
    return {
        "episode_world_units": int(len(keys)),
        "unit_mean_rmse": rmse(predicted_mean, observed_mean),
    }


def exported_prediction(
    standardized: np.ndarray, intercept: float, terms: list[dict]
) -> np.ndarray:
    result = np.full(len(standardized), intercept, dtype=np.float64)
    for term in terms:
        result += np.interp(
            standardized[:, term["feature"]], term["knots"], term["values"]
        )
    return result


def split_metrics(
    *,
    direct: np.ndarray,
    observed: np.ndarray,
    approximate: np.ndarray,
    raw: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    units: np.ndarray,
    slots: np.ndarray,
    training_mean: float,
) -> dict[str, Any]:
    in_domain = np.all((raw >= lower) & (raw <= upper), axis=1)
    export_error = np.abs(approximate - direct)
    result = {
        "rows": int(len(observed)),
        "rmse": rmse(direct, observed),
        "mean_baseline_rmse": rmse(np.full(len(observed), training_mean), observed),
        "zero_baseline_rmse": rmse(np.zeros(len(observed)), observed),
        "in_domain_rows": int(in_domain.sum()),
        "in_domain_fraction": float(in_domain.mean()),
        "export_grid_max_abs_error_all_rows": float(export_error.max()),
        "export_grid_max_abs_error_in_domain": (
            float(export_error[in_domain].max()) if in_domain.any() else None
        ),
        "in_domain_rmse": (
            rmse(direct[in_domain], observed[in_domain]) if in_domain.any() else None
        ),
        **aggregate_unit_metrics(direct, observed, units),
    }
    result["world_slots"] = {
        str(int(slot)): {
            "rows": int(np.sum(slots == slot)),
            "rmse": rmse(direct[slots == slot], observed[slots == slot]),
            "in_domain_fraction": float(in_domain[slots == slot].mean()),
        }
        for slot in np.unique(slots)
    }
    return result


def validate_compact(data: np.lib.npyio.NpzFile) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    required = {
        "features", "outcomes", "partition", "world_unit", "world_slot",
        "resident_slot", "episode_unit", "tick_unit", "source_sha256",
        "source_contract",
    }
    if set(data.files) != required:
        raise ValueError(f"body-law compact arrays differ: {sorted(set(data.files) ^ required)}")
    arrays = {name: np.asarray(data[name]) for name in required if name != "source_contract"}
    x, y = arrays["features"], arrays["outcomes"]
    if x.shape != (655_360, len(FEATURES)) or y.shape != (655_360, len(OUTCOMES)):
        raise ValueError("body-law compact dimensions differ from the pinned rich-v4 campaign")
    row_arrays = (
        "partition", "world_unit", "world_slot", "resident_slot", "episode_unit",
        "tick_unit",
    )
    if any(arrays[name].shape != (len(x),) for name in row_arrays):
        raise ValueError("body-law compact audit axes differ")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("body-law compact contains non-finite model values")
    contract = json.loads(scalar_text(data["source_contract"]))
    if (
        contract.get("collection_format") != "chreatures-sensorimotor-play-rich-v4"
        or contract.get("feature_names") != [name for name, _ in FEATURES]
        or contract.get("target_names") != [name for name, _ in OUTCOMES]
        or contract.get("split", {}).get("train_world_slots") != list(range(8))
        or contract.get("split", {}).get("validation_world_slots") != [8]
        or contract.get("split", {}).get("heldout_world_slots") != [9]
    ):
        raise ValueError("body-law source contract differs from current LawBank seams")
    for name, code in PARTITIONS.items():
        mask = arrays["partition"] == code
        if not mask.any() or tuple(np.unique(arrays["world_slot"][mask])) != EXPECTED_SLOTS[name]:
            raise ValueError(f"{name} world partition differs")
        if np.any(arrays["partition"][np.isin(arrays["world_slot"], EXPECTED_SLOTS[name])] != code):
            raise ValueError(f"world slots leak across the {name} partition")
    if set(np.unique(arrays["partition"])) != set(PARTITIONS.values()):
        raise ValueError("unknown body-law partition code")
    expected_rows = {"train": 524_288, "validation": 65_536, "holdout": 65_536}
    for name, code in PARTITIONS.items():
        if int(np.sum(arrays["partition"] == code)) != expected_rows[name]:
            raise ValueError(f"{name} row count differs")
    return contract, arrays


def law_evaluation(artifact: dict, raw: np.ndarray) -> tuple[list[float], bool]:
    features = artifact["features"]
    standardized = np.asarray(
        [(float(value) - item["mean"]) / item["scale"] for value, item in zip(raw, features, strict=True)]
    )
    feature_ood = np.asarray(
        [float(value) < item["minimum"] or float(value) > item["maximum"]
         for value, item in zip(raw, features, strict=True)]
    )
    values = []
    any_ood = False
    for law in artifact["laws"]:
        expected = float(law["intercept"])
        law_ood = False
        for term in law["terms"]:
            feature = int(term["feature"])
            coordinate = standardized[feature]
            law_ood |= bool(feature_ood[feature])
            law_ood |= coordinate < term["knots"][0] or coordinate > term["knots"][-1]
            expected += float(np.interp(coordinate, term["knots"], term["values"]))
        values.append(expected)
        any_ood |= law_ood
    return [float(value) for value in values], bool(any_ood)


def native_cases(artifact: dict) -> dict[str, Any]:
    features = artifact["features"]
    center = np.asarray(
        [(item["minimum"] + item["maximum"]) / 2.0 for item in features],
        dtype=np.float64,
    )
    rows: list[tuple[str, np.ndarray]] = [("domain-center", center)]
    for index, item in enumerate(features):
        delta = max((item["maximum"] - item["minimum"]) * 1e-6, 1e-12)
        low, high, below, above = center.copy(), center.copy(), center.copy(), center.copy()
        low[index] = item["minimum"] + delta
        high[index] = item["maximum"] - delta
        below[index] = item["minimum"] - delta
        above[index] = item["maximum"] + delta
        rows.extend(((f"feature-{index}-domain-low", low),
                     (f"feature-{index}-domain-high", high),
                     (f"feature-{index}-below-domain", below),
                     (f"feature-{index}-above-domain", above)))
    cases = []
    for name, row in rows:
        expected, out_of_domain = law_evaluation(artifact, row)
        cases.append({
            "name": name,
            "features": row.tolist(),
            "expected": expected,
            "out_of_domain": out_of_domain,
        })
    return {
        "schema": "chreatures-gam-native-verification-cases-v1",
        "tolerance": 1e-10,
        "cases": cases,
    }


def failed_report(
    output_dir: Path,
    compact_path: Path,
    started: float,
    models: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    report = {
        "schema": "chreatures-gam-consequence-fit-report-v2",
        "status": "failed; no native LawBank minted",
        "reason": reason,
        "fit_seconds": time.perf_counter() - started,
        "source_compact_sha256": sha256(compact_path),
        "split": {
            "train_world_slots": list(EXPECTED_SLOTS["train"]),
            "validation_world_slots": list(EXPECTED_SLOTS["validation"]),
            "heldout_world_slots": list(EXPECTED_SLOTS["holdout"]),
            "rule": "whole episode/world units; eight co-resident rows never cross partitions",
        },
        "models": models,
    }
    write_json(output_dir / "fit_report.json", report)
    write_json(output_dir / "analyst_receipt.json", {
        "schema": "chreatures-rich-v4-body-law-analyst-receipt-v1",
        "status": report["status"],
        "reason": reason,
        "source_compact_sha256": report["source_compact_sha256"],
        "split": report["split"],
        "fit_report_sha256": sha256(output_dir / "fit_report.json"),
        "interpretation": "Failed native fits are retained as evidence; no fallback law is substituted.",
    })
    return report


def fit(compact_path: Path, output_dir: Path, native_evaluator: Path) -> dict[str, Any]:
    import gamfit

    build = gamfit.build_info()
    if gamfit.__version__ != GAMFIT_VERSION or not build.get("available"):
        raise RuntimeError(f"requires native gamfit {GAMFIT_VERSION}")
    if not native_evaluator.is_file() or not os.access(native_evaluator, os.X_OK):
        raise ValueError("native LawBank evaluator is missing or not executable")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("fit output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with np.load(compact_path, allow_pickle=False) as data:
        source_contract, arrays = validate_compact(data)
        source_hashes = [str(value) for value in data["source_sha256"].tolist()]
    if len(source_hashes) != 16 or any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in source_hashes
    ):
        raise ValueError("body-law compact source packet identities differ")
    x = arrays["features"].astype(np.float64)
    y = arrays["outcomes"].astype(np.float64)
    partition = arrays["partition"]
    train = partition == PARTITIONS["train"]
    validation = partition == PARTITIONS["validation"]
    holdout = partition == PARTITIONS["holdout"]
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    if np.any(scale < 1e-9):
        constant = [FEATURES[index][0] for index in np.flatnonzero(scale < 1e-9)]
        return failed_report(
            output_dir, compact_path, started, {},
            f"training features lack support: {constant}",
        )
    lower = np.quantile(x[train], 0.005, axis=0)
    upper = np.quantile(x[train], 0.995, axis=0)
    if np.any(upper <= lower):
        return failed_report(
            output_dir, compact_path, started, {},
            "training feature quantile domains are degenerate",
        )
    z = (x - mean) / scale
    laws: list[dict[str, Any]] = []
    model_report: dict[str, Any] = {}
    for outcome_index, (outcome_name, unit) in enumerate(OUTCOMES):
        model_path = output_dir / f"{outcome_name}.gam"
        try:
            model = gamfit.fit_array(
                np.ascontiguousarray(z[train]),
                np.ascontiguousarray(y[train, outcome_index]),
                FORMULA,
                family="gaussian",
            )
            model.save(model_path)
            baseline = np.zeros((1, len(FEATURES)), dtype=np.float64)
            intercept = float(np.asarray(model.predict(baseline))[0])
            terms = []
            for feature_index in range(len(FEATURES)):
                lo = (lower[feature_index] - mean[feature_index]) / scale[feature_index]
                hi = (upper[feature_index] - mean[feature_index]) / scale[feature_index]
                knots = np.linspace(lo, hi, 97)
                grid = np.zeros((len(knots), len(FEATURES)), dtype=np.float64)
                grid[:, feature_index] = knots
                values = np.asarray(model.predict(grid), dtype=np.float64) - intercept
                terms.append({
                    "feature": feature_index,
                    "knots": knots.tolist(),
                    "values": values.tolist(),
                })
            split_report = {}
            direct_by_split = {}
            for split_name, mask in (("validation", validation), ("holdout", holdout)):
                direct = np.asarray(model.predict(np.ascontiguousarray(z[mask])), dtype=np.float64)
                approximate = exported_prediction(z[mask], intercept, terms)
                direct_by_split[split_name] = direct
                split_report[split_name] = split_metrics(
                    direct=direct,
                    observed=y[mask, outcome_index],
                    approximate=approximate,
                    raw=x[mask],
                    lower=lower,
                    upper=upper,
                    units=arrays["world_unit"][mask],
                    slots=arrays["world_slot"][mask],
                    training_mean=float(y[train, outcome_index].mean()),
                )
            validation_residual = np.abs(
                y[validation, outcome_index] - direct_by_split["validation"]
            )
            validation_rmse = rmse(
                direct_by_split["validation"], y[validation, outcome_index]
            )
            target_scale = float(y[train, outcome_index].std())
            if target_scale <= 0.0:
                raise ValueError("training target has no variation")
            residual_bound = max(
                float(np.quantile(validation_residual, 0.995)), 3.0 * validation_rmse
            )
            laws.append({
                "name": outcome_name,
                "unit": unit,
                "intercept": intercept,
                "residual_rmse": validation_rmse,
                "target_scale": target_scale,
                "conservative_residual_bound": residual_bound,
                "terms": terms,
            })
            model_report[outcome_name] = {
                "status": "fitted with native gamfit",
                "formula": FORMULA,
                "model": {
                    "path": model_path.name,
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256(model_path),
                },
                "validation_abs_residual_q995": float(
                    np.quantile(validation_residual, 0.995)
                ),
                "conservative_residual_bound": residual_bound,
                "splits": split_report,
            }
        except Exception as exc:
            model_report[outcome_name] = {
                "status": "native fit failed; no law minted",
                "error": f"{type(exc).__name__}: {exc}"[:8000],
            }
    if len(laws) != len(OUTCOMES):
        return failed_report(
            output_dir, compact_path, started, model_report,
            "one or more native gamfit models failed certification",
        )

    artifact = {
        "schema": "chreatures-gam-consequence-law-bank-v1",
        "source": {
            "model_library": "SauersML/gam",
            "model_version": GAMFIT_VERSION,
            "model_source_commit": GAMFIT_SOURCE_COMMIT,
            "telemetry_sha256": source_hashes,
            "contract": source_contract,
        },
        "features": [
            {
                "name": name, "unit": unit, "mean": float(mean[index]),
                "scale": float(scale[index]), "minimum": float(lower[index]),
                "maximum": float(upper[index]),
            }
            for index, (name, unit) in enumerate(FEATURES)
        ],
        "laws": laws,
    }
    candidate_path = output_dir / "body_consequence_laws.candidate.json"
    cases_path = output_dir / "native_verification_cases.json"
    native_receipt_path = output_dir / "native_evaluation.json"
    write_json(candidate_path, artifact, compact=True)
    write_json(cases_path, native_cases(artifact))
    command = [
        str(native_evaluator.resolve()), "--verify", str(candidate_path.resolve()),
        str(cases_path.resolve()),
    ]
    native = subprocess.run(command, capture_output=True, text=True)
    if native.returncode != 0:
        write_json(native_receipt_path, {
            "schema": "chreatures-gam-native-evaluation-v1",
            "status": "failed; candidate retained and no LawBank minted",
            "returncode": native.returncode,
            "stdout": native.stdout[-4000:],
            "stderr": native.stderr[-4000:],
        })
        return failed_report(
            output_dir, compact_path, started, model_report,
            "native LawBank verification failed; candidate artifact retained",
        )
    try:
        native_receipt = json.loads(native.stdout)
    except json.JSONDecodeError as exc:
        write_json(native_receipt_path, {
            "schema": "chreatures-gam-native-evaluation-v1",
            "status": "failed; evaluator returned invalid JSON",
            "error": str(exc),
            "stdout": native.stdout[-4000:],
            "stderr": native.stderr[-4000:],
        })
        return failed_report(
            output_dir, compact_path, started, model_report,
            "native LawBank verification receipt was invalid",
        )
    if (
        native_receipt.get("schema") != "chreatures-gam-native-evaluation-v1"
        or native_receipt.get("artifact_sha256") != sha256(candidate_path)
        or native_receipt.get("cases_sha256") != sha256(cases_path)
        or native_receipt.get("cases") != 49
        or native_receipt.get("in_domain_cases") != 25
        or native_receipt.get("out_of_domain_cases") != 24
        or native_receipt.get("feature_contract_checked") is not True
        or float(native_receipt.get("maximum_absolute_error", float("inf")))
        > float(native_receipt.get("tolerance", 0.0))
    ):
        write_json(native_receipt_path, {
            "schema": "chreatures-gam-native-evaluation-v1",
            "status": "failed; evaluator receipt contents differ",
            "received": native_receipt,
        })
        return failed_report(
            output_dir, compact_path, started, model_report,
            "native LawBank verification receipt contents differed",
        )
    native_receipt["evaluator_sha256"] = sha256(native_evaluator)
    write_json(native_receipt_path, native_receipt)
    artifact_path = output_dir / "body_consequence_laws.json"
    os.replace(candidate_path, artifact_path)
    report = {
        "schema": "chreatures-gam-consequence-fit-report-v2",
        "status": "native-verified successor candidate for new births; not promoted",
        "fit_seconds": time.perf_counter() - started,
        "model_library": {
            "name": "SauersML/gam", "version": GAMFIT_VERSION,
            "source_commit": GAMFIT_SOURCE_COMMIT,
            "build": build,
            "fit_api": "gamfit.fit_array native Rust path",
        },
        "formula": FORMULA,
        "rows": {
            name: int(np.sum(partition == code)) for name, code in PARTITIONS.items()
        },
        "independent_episode_world_units": {
            name: int(np.unique(arrays["world_unit"][partition == code]).size)
            for name, code in PARTITIONS.items()
        },
        "split": "whole world slots 0-7 train, 8 validation, 9 final holdout across both episodes",
        "uncertainty_calibration": (
            "validation slot 8 only; slot 9 is final reporting and never changes the artifact"
        ),
        "cohort_dependence": (
            "eight residents share each physical world; partitions and aggregate metrics keep "
            "each episode/world unit intact"
        ),
        "source_compact_sha256": sha256(compact_path),
        "artifact": {
            "path": artifact_path.name, "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        },
        "native_evaluation": {
            "path": native_receipt_path.name,
            "sha256": sha256(native_receipt_path),
        },
        "models": model_report,
        "limitations": [
            "These are conditional associations in executed physical transitions, not interventions or causes.",
            "World and resident audit keys never enter the LawBank feature vector.",
            "A candidate outside any fitted feature domain receives no GAM selection adjustment.",
            "The fit creates a reusable successor artifact and does not change an existing resident.",
        ],
    }
    write_json(output_dir / "fit_report.json", report)
    analyst = {
        "schema": "chreatures-rich-v4-body-law-analyst-receipt-v1",
        "status": report["status"],
        "source": {
            "compact_sha256": report["source_compact_sha256"],
            "collection_identity_sha256": source_contract["collection_identity_sha256"],
            "source_revision": source_contract["source_revision"],
            "resident_artifact": source_contract.get("resident_artifact"),
            "founding_bank": source_contract.get("founding_bank"),
            "candidate_order_sha256": source_contract.get("candidate_order_sha256"),
            "neural_phenotypes_sha256": source_contract.get("neural_phenotypes_sha256"),
        },
        "feature_target_seams": {
            "features": [name for name, _ in FEATURES],
            "targets": [name for name, _ in OUTCOMES],
            "executed_action_mapping": {
                "thrust": 0, "yaw": 1, "grip": 4, "oral_eat": 8,
                "motor_magnitude": [0, 1, 2, 4],
            },
            "physiology_mapping": {
                "energy": 0, "fatigue": 2, "speed": 3, "neural_support": 5,
            },
        },
        "partitions": {
            "train_world_slots": list(EXPECTED_SLOTS["train"]),
            "validation_world_slots": list(EXPECTED_SLOTS["validation"]),
            "heldout_world_slots": list(EXPECTED_SLOTS["holdout"]),
            "rows": report["rows"],
            "episode_world_units": report["independent_episode_world_units"],
        },
        "domain_and_performance": {
            law: value["splits"] for law, value in model_report.items()
        },
        "files": {
            "law_bank": report["artifact"],
            "fit_report": {"path": "fit_report.json"},
            "native_evaluation": report["native_evaluation"],
            "native_verification_cases": {
                "path": cases_path.name, "sha256": sha256(cases_path),
            },
        },
        "interpretation": (
            "Outside analysis receipt only; it is not a controller input, score, personality, "
            "or causal claim. Root owns any source freeze and promotion."
        ),
    }
    # Write once, then fill the self-independent fit-report hash.
    analyst["files"]["fit_report"]["sha256"] = sha256(output_dir / "fit_report.json")
    write_json(output_dir / "analyst_receipt.json", analyst)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-evaluator", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        fit(args.compact, args.output_dir, args.native_evaluator),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
