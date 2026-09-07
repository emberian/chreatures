"""Fit and query a native GAM genotype-by-environment transfer atlas.

The atlas is an analyst artifact.  Its environmental descriptors and archive
identities must never enter a resident's sensory or controller inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

GAMFIT_VERSION = "0.1.259"
GAMFIT_COMMIT = "7c7eca8ac4826de95c8e743a20294bee132a9bcc"
SEARCH_FORMAT = "chreatures-population-search-v2"
RESULT_FORMAT = "chreatures-population-episode-evaluation-v1"
REQUEST_FORMAT = "chreatures-gxe-ranking-request-v1"

GENOTYPE_FEATURES = {
    "reserve_capacity": "somatic.reserve_capacity_gain",
    "maintenance": "somatic.maintenance_rate_gain",
    "leg_length": "body.leg_length_scale",
    "friction": "body.friction_scale",
    "secretion_rate": "somatic.secretion_rate_gain",
    "allocation_structure": "developmental.allocation.structure",
}
ENVIRONMENT_FEATURES = (
    "resource_density",
    "renewal_rate",
    "elevation_relief",
    "regional_scale",
    "connectivity",
)
RANK_FEATURES = tuple(GENOTYPE_FEATURES) + ENVIRONMENT_FEATURES
DIAGNOSTIC_FEATURES = (
    "mean_abs_thrust",
    "mean_abs_posture",
    "mean_abs_secrete",
    "mean_abs_allocate",
    "mean_fatigue",
    "mean_gut",
    "elevation_relief",
    "resource_density",
)

RESPONSES = {
    "energy_change": {
        "unit": "physiological_energy_over_102.4_seconds",
        "source": "trajectory_metrics.energy_change",
        "transform": "identity",
        "additive": (
            "response ~ s(reserve_capacity,k=4)+s(maintenance,k=4)"
            "+s(resource_density,k=4)+s(renewal_rate,k=4)"
        ),
        "interaction": "ti(reserve_capacity,resource_density,k=9)",
        "mechanism_hypothesis": "stored-energy capacity by resource density",
    },
    "contact_ticks": {
        "unit": "physical_ticks_with_contact",
        "source": "trajectory_metrics.contact_ticks",
        "transform": "log1p",
        "additive": (
            "response ~ s(leg_length,k=4)+s(friction,k=4)"
            "+s(elevation_relief,k=4)+s(regional_scale,k=4)"
        ),
        "interaction": "ti(leg_length,elevation_relief,k=9)",
        "mechanism_hypothesis": "leg scale by terrain relief",
    },
    "mechanical_work_rate": {
        "unit": "world_mechanical_work_per_second",
        "source": "trajectory_metrics.mechanical_work_sum / valid_time_seconds",
        "transform": "identity",
        "additive": (
            "response ~ s(leg_length,k=4)+s(friction,k=4)"
            "+s(elevation_relief,k=4)+s(regional_scale,k=4)"
        ),
        "interaction": "ti(leg_length,elevation_relief,k=9)",
        "mechanism_hypothesis": "leg scale by terrain relief",
    },
    "allocation_rate": {
        "unit": "allocated_material_mass_per_second",
        "source": "trajectory_metrics.allocation_mass_sum / valid_time_seconds",
        "transform": "identity",
        "additive": (
            "response ~ s(allocation_structure,k=4)+s(secretion_rate,k=4)"
            "+s(resource_density,k=4)+s(renewal_rate,k=4)"
        ),
        "interaction": "ti(reserve_capacity,resource_density,k=9)",
        "mechanism_hypothesis": "stored-energy capacity by resource density",
    },
}

DIAGNOSTIC_FORMULA = (
    "response ~ s(mean_abs_thrust,k=4)+s(mean_abs_posture,k=4)"
    "+s(mean_abs_secrete,k=4)+s(mean_abs_allocate,k=4)"
    "+s(mean_fatigue,k=4)+s(mean_gut,k=4)"
    "+s(elevation_relief,k=4)+s(resource_density,k=4)"
    "+ti(mean_abs_thrust,elevation_relief,k=9)"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_sha(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} file SHA-256 differs: {actual} != {expected}")


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _source_identities(principal_receipt: dict[str, Any], challenge_receipt: dict[str, Any]) -> dict[str, str]:
    return {
        "principal_result_file_sha256": principal_receipt["evidence"]["finalized_result"]["file_sha256"],
        "principal_result_content_sha256": principal_receipt["evidence"]["finalized_result"]["content_sha256"],
        "principal_receipt_content_sha256": principal_receipt["content_sha256"],
        "challenge_result_file_sha256": challenge_receipt["evidence"]["result"]["file_sha256"],
        "challenge_result_content_sha256": challenge_receipt["evidence"]["result"]["content_sha256"],
        "challenge_receipt_content_sha256": challenge_receipt["content_sha256"],
        "search_file_sha256": challenge_receipt["evidence"]["post_ingest_search_file_sha256"],
    }


def prepare_life_rows(
    search_path: Path,
    principal_path: Path,
    challenge_path: Path,
    principal_receipt_path: Path,
    challenge_receipt_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate and reduce two completed physical campaigns to 160 life rows."""
    search = _load_json(search_path)
    principal = _load_json(principal_path)
    challenge = _load_json(challenge_path)
    principal_receipt = _load_json(principal_receipt_path)
    challenge_receipt = _load_json(challenge_receipt_path)
    if principal_receipt.get("format") != "chreatures-population-principal-wave-receipt-v1":
        raise ValueError("principal receipt format differs")
    if challenge_receipt.get("format") != "chreatures-population-challenge-wave-receipt-v1":
        raise ValueError("challenge receipt format differs")
    identities = _source_identities(principal_receipt, challenge_receipt)
    identities["principal_receipt_file_sha256"] = file_sha256(principal_receipt_path)
    identities["challenge_receipt_file_sha256"] = file_sha256(challenge_receipt_path)
    _require_sha(search_path, identities["search_file_sha256"], "post-ingest search")
    _require_sha(principal_path, identities["principal_result_file_sha256"], "principal result")
    _require_sha(challenge_path, identities["challenge_result_file_sha256"], "challenge result")
    if search.get("format") != SEARCH_FORMAT:
        raise ValueError("population search format differs")
    from chreatures.population import content_sha256

    for candidate_id, genome in search.get("genomes", {}).items():
        if genome.get("sha256") != candidate_id or content_sha256(genome) != candidate_id:
            raise ValueError("candidate map key differs from authenticated identity")
    for environment_id, environment in search.get("environments", {}).items():
        if content_sha256(environment) != environment_id:
            raise ValueError("environment map key differs from authenticated identity")
    for label, result in (("principal", principal), ("challenge", challenge)):
        if result.get("format") != RESULT_FORMAT or result.get("status") != "completed":
            raise ValueError(f"{label} result is not a completed current evaluation")
        if len(result.get("lives", [])) != 80:
            raise ValueError(f"{label} result must contain exactly 80 completed lives")
        expected_content = identities[f"{label}_result_content_sha256"]
        if result.get("content_sha256") != expected_content:
            raise ValueError(f"{label} result declared content identity differs")
    if principal_receipt.get("status") != "completed" or challenge_receipt.get("status") != "completed-and-ingested":
        raise ValueError("campaign public receipts are not final")

    rows: list[dict[str, Any]] = []
    life_ids: set[str] = set()
    for wave, result in (("principal", principal), ("challenge", challenge)):
        for life in result["lives"]:
            life_id = str(life["life_id"])
            candidate_id = str(life["candidate_sha256"])
            environment_id = str(life["environment_sha256"])
            if life_id in life_ids:
                raise ValueError(f"duplicate life identity {life_id}")
            life_ids.add(life_id)
            if life.get("status") != "completed" or int(life.get("committed_ticks", -1)) != 2048:
                raise ValueError(f"life {life_id} is not a complete 2048-tick observation")
            genome = search.get("genomes", {}).get(candidate_id)
            environment = result.get("environments", {}).get(environment_id)
            if not genome or genome.get("sha256") != candidate_id:
                raise ValueError(f"life {life_id} lacks its authenticated candidate record")
            if not environment or environment.get("sha256") != environment_id:
                raise ValueError(f"life {life_id} lacks its authenticated environment record")
            search_environment = search.get("environments", {}).get(environment_id)
            if search_environment != environment:
                raise ValueError(f"life {life_id} environment differs from post-ingest search")
            values = genome.get("values", {})
            descriptors = environment.get("descriptors", {})
            if set(descriptors) != set(ENVIRONMENT_FEATURES):
                raise ValueError(f"environment descriptor contract differs for {environment_id}")
            metrics = life.get("trajectory_metrics", {})
            if metrics.get("format") != "chreatures-population-trajectory-v1":
                raise ValueError(f"life {life_id} trajectory contract differs")
            if int(metrics.get("valid_ticks", -1)) != 2048 or not metrics.get("has_valid_observation"):
                raise ValueError(f"life {life_id} lacks a full valid trajectory")
            action = dict(zip(metrics["executed_action_order"], metrics["executed_action_abs_mean"], strict=True))
            physiology = dict(zip(metrics["physiology_order"], metrics["physiology_mean"], strict=True))
            features = {name: _finite_float(values[source], source) for name, source in GENOTYPE_FEATURES.items()}
            features.update({name: _finite_float(descriptors[name], name) for name in ENVIRONMENT_FEATURES})
            diagnostic = {
                "mean_abs_thrust": _finite_float(action["thrust"], "mean_abs_thrust"),
                "mean_abs_posture": _finite_float(action["posture"], "mean_abs_posture"),
                "mean_abs_secrete": _finite_float(action["secrete"], "mean_abs_secrete"),
                "mean_abs_allocate": _finite_float(action["allocate"], "mean_abs_allocate"),
                "mean_fatigue": _finite_float(physiology["fatigue"], "mean_fatigue"),
                "mean_gut": _finite_float(physiology["gut"], "mean_gut"),
                "elevation_relief": features["elevation_relief"],
                "resource_density": features["resource_density"],
            }
            valid_seconds = _finite_float(metrics["valid_time_seconds"], "valid_time_seconds")
            if valid_seconds != 102.4:
                raise ValueError(f"life {life_id} observation duration differs")
            outcomes = {
                "energy_change": _finite_float(metrics["energy_change"], "energy_change"),
                "contact_ticks": _finite_float(metrics["contact_ticks"], "contact_ticks"),
                "mechanical_work_rate": _finite_float(metrics["mechanical_work_sum"], "mechanical_work_sum") / valid_seconds,
                "allocation_rate": _finite_float(metrics["allocation_mass_sum"], "allocation_mass_sum") / valid_seconds,
            }
            if outcomes["contact_ticks"] < 0 or outcomes["allocation_rate"] < 0 or outcomes["mechanical_work_rate"] < 0:
                raise ValueError(f"life {life_id} contains negative nonnegative outcome")
            rows.append({
                "life_id": life_id,
                "candidate_sha256": candidate_id,
                "environment_sha256": environment_id,
                "trajectory_sha256": str(life["trajectory_sha256"]),
                "wave": wave,
                "features": features,
                "realized_diagnostic_features": diagnostic,
                "outcomes": outcomes,
            })
    if len(rows) != 160:
        raise ValueError("combined campaign must contain exactly 160 independent life rows")
    transferred = sorted(candidate for candidate in {r["candidate_sha256"] for r in rows}
                         if sum(x["candidate_sha256"] == candidate for x in rows) > 1)
    environments = sorted({r["environment_sha256"] for r in rows})
    if len(transferred) != 10 or len(environments) != 10:
        raise ValueError("expected ten transfer genotypes and ten environments")
    split = {
        "heldout_candidates": transferred[-2:],
        "validation_candidates": transferred[-4:-2],
        "heldout_environments": environments[-2:],
        "validation_environments": environments[-4:-2],
        "rule": (
            "lexicographically last two repeated candidate hashes and environment hashes are reporting holdouts; "
            "preceding two of each are validation; a row belongs to the first applicable union"
        ),
    }
    for row in rows:
        if row["candidate_sha256"] in split["heldout_candidates"] or row["environment_sha256"] in split["heldout_environments"]:
            row["split"] = "heldout"
        elif row["candidate_sha256"] in split["validation_candidates"] or row["environment_sha256"] in split["validation_environments"]:
            row["split"] = "validation"
        else:
            row["split"] = "train"
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "heldout")}
    if min(counts.values()) < 1:
        raise ValueError("grouped split is empty")
    identity = {
        **identities,
        "principal_source_revision": principal_receipt["identity"]["source_revision"],
        "challenge_source_revision": challenge_receipt["identity"]["source_revision"],
        "controller_artifact_sha256": challenge_receipt["identity"]["controller"]["artifact_sha256"],
        "rows": len(rows),
        "unique_candidates": len({r["candidate_sha256"] for r in rows}),
        "transfer_candidates": len(transferred),
        "environments": len(environments),
        "split": split,
        "split_rows": counts,
        "genome_format": next(iter(search["genomes"].values()))["format"],
        "genome_value_keys": sorted(next(iter(search["genomes"].values()))["values"]),
        "genome_context": {
            name: next(iter(search["genomes"].values()))[name]
            for name in (
                "base_controller_sha256",
                "developmental_base_sha256",
                "graph_sha256",
                "organism_interface_sha256",
                "population_adapter_bank_sha256",
                "policy_adapter_count",
                "policy_adapter_rank",
            )
        },
        "environment_format": next(iter(search["environments"].values()))["format"],
        "environment_descriptor_keys": sorted(ENVIRONMENT_FEATURES),
        "environment_profile_sha256s": sorted(
            {row["profile_sha256"] for row in search["environments"].values()}
        ),
    }
    return rows, identity


def _response_value(row: dict[str, Any], name: str) -> float:
    value = float(row["outcomes"][name])
    return math.log1p(value) if RESPONSES[name]["transform"] == "log1p" else value


def _inverse_response(value: float, name: str) -> float:
    return max(0.0, math.expm1(value)) if RESPONSES[name]["transform"] == "log1p" else value


def _normalization(rows: list[dict[str, Any]], field: str, names: Iterable[str]) -> dict[str, dict[str, float]]:
    train = [row for row in rows if row["split"] == "train"]
    result = {}
    for name in names:
        values = np.asarray([row[field][name] for row in train], dtype=np.float64)
        scale = float(values.std())
        result[name] = {"mean": float(values.mean()), "scale": scale if scale >= 1e-12 else 1.0}
    return result


def _gam_rows(rows: list[dict[str, Any]], split: str, field: str, names: tuple[str, ...],
              normalization: dict[str, dict[str, float]], response: str) -> list[dict[str, float]]:
    result = []
    for row in rows:
        if row["split"] != split:
            continue
        item = {name: (float(row[field][name]) - normalization[name]["mean"]) / normalization[name]["scale"] for name in names}
        item["response"] = _response_value(row, response)
        result.append(item)
    return result


def _metrics(model: Any, rows: list[dict[str, Any]], split: str, field: str,
             names: tuple[str, ...], normalization: dict[str, dict[str, float]], response: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    table = _gam_rows(rows, split, field, names, normalization, response)
    observed_latent = np.asarray([item.pop("response") for item in table], dtype=np.float64)
    predicted_latent = np.asarray(model.predict(table), dtype=np.float64)
    observed = np.asarray([row["outcomes"][response] for row in selected], dtype=np.float64)
    predicted = np.asarray([_inverse_response(float(value), response) for value in predicted_latent])
    train = np.asarray([row["outcomes"][response] for row in rows if row["split"] == "train"], dtype=np.float64)
    baseline = np.full(len(observed), train.mean())
    by_candidate: dict[str, list[int]] = {}
    by_environment: dict[str, list[int]] = {}
    for index, row in enumerate(selected):
        by_candidate.setdefault(row["candidate_sha256"], []).append(index)
        by_environment.setdefault(row["environment_sha256"], []).append(index)

    def group_rmse(groups: dict[str, list[int]]) -> float:
        errors = [float(predicted[idx].mean() - observed[idx].mean()) for idx in groups.values()]
        return float(np.sqrt(np.mean(np.square(errors))))

    return {
        "life_observation_rows": len(selected),
        "statistical_independence_claim": False,
        "candidate_groups": len(by_candidate),
        "environment_groups": len(by_environment),
        "rmse": float(np.sqrt(np.mean(np.square(predicted - observed)))),
        "mean_baseline_rmse": float(np.sqrt(np.mean(np.square(baseline - observed)))),
        "candidate_group_mean_rmse": group_rmse(by_candidate),
        "environment_group_mean_rmse": group_rmse(by_environment),
        "latent_rmse": float(np.sqrt(np.mean(np.square(predicted_latent - observed_latent)))),
    }


def _support(rows: list[dict[str, Any]], normalization: dict[str, dict[str, float]], search: dict[str, Any]) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    matrix = np.asarray([[row["features"][name] for name in RANK_FEATURES] for row in train], dtype=np.float64)
    mean = np.asarray([normalization[name]["mean"] for name in RANK_FEATURES])
    scale = np.asarray([normalization[name]["scale"] for name in RANK_FEATURES])
    z = (matrix - mean) / scale
    unique_candidates = sorted({row["candidate_sha256"] for row in train})
    unique_environments = sorted({row["environment_sha256"] for row in train})
    candidate_matrix = np.asarray([[search["genomes"][candidate]["values"][GENOTYPE_FEATURES[name]]
                                    for name in GENOTYPE_FEATURES] for candidate in unique_candidates], dtype=np.float64)
    environment_matrix = np.asarray([[search["environments"][environment]["descriptors"][name]
                                      for name in ENVIRONMENT_FEATURES] for environment in unique_environments], dtype=np.float64)

    def nearest_threshold(values: np.ndarray, quantile: float = 0.95) -> float:
        if len(values) < 2:
            return 0.0
        distance = np.sqrt(np.sum(np.square(values[:, None, :] - values[None, :, :]), axis=2))
        np.fill_diagonal(distance, np.inf)
        return float(np.quantile(np.min(distance, axis=1), quantile))

    gz = (candidate_matrix - mean[:len(GENOTYPE_FEATURES)]) / scale[:len(GENOTYPE_FEATURES)]
    ez = (environment_matrix - mean[len(GENOTYPE_FEATURES):]) / scale[len(GENOTYPE_FEATURES):]
    return {
        "rule": "all full-schema values in training ranges and modeled candidate/environment nearest distances at or below training 95th-percentile leave-one-group-out distances",
        "modeled_bounds": {name: {"minimum": float(matrix[:, i].min()), "maximum": float(matrix[:, i].max())}
                           for i, name in enumerate(RANK_FEATURES)},
        "full_genome_value_bounds": {name: {"minimum": float(min(search["genomes"][row["candidate_sha256"]]["values"][name] for row in train)),
                                                    "maximum": float(max(search["genomes"][row["candidate_sha256"]]["values"][name] for row in train))}
                                     for name in sorted(next(iter(search["genomes"].values()))["values"])},
        "candidate_distance_threshold": nearest_threshold(gz),
        "environment_distance_threshold": nearest_threshold(ez),
        "train_candidate_vectors": gz.tolist(),
        "train_environment_vectors": ez.tolist(),
        "train_pair_vectors": z.tolist(),
        "observed_pairs": sorted({f'{row["candidate_sha256"]}:{row["environment_sha256"]}' for row in rows}),
    }


def fit_atlas(rows: list[dict[str, Any]], source: dict[str, Any], search_path: Path, output: Path) -> dict[str, Any]:
    import gamfit
    if gamfit.__version__ != GAMFIT_VERSION or not gamfit.build_info().get("available"):
        raise RuntimeError(f"native gamfit {GAMFIT_VERSION} is required")
    search = _load_json(search_path)
    normalization = _normalization(rows, "features", RANK_FEATURES)
    diagnostic_normalization = _normalization(rows, "realized_diagnostic_features", DIAGNOSTIC_FEATURES)
    output.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    started = time.perf_counter()
    for response, spec in RESPONSES.items():
        variants = {"additive": spec["additive"], "interaction": spec["additive"] + "+" + spec["interaction"]}
        comparison: dict[str, Any] = {}
        for variant, formula in variants.items():
            train = _gam_rows(rows, "train", "features", RANK_FEATURES, normalization, response)
            gamfit.validate_formula(train, formula)
            model = gamfit.fit(train, formula, family="gaussian")
            model_path = output / f"{response}-{variant}.gam"
            model.save(model_path)
            comparison[variant] = {
                "formula": formula,
                "native_model": model_path.name,
                "native_model_sha256": file_sha256(model_path),
                "validation": _metrics(model, rows, "validation", "features", RANK_FEATURES, normalization, response),
                "heldout_reporting_only": _metrics(model, rows, "heldout", "features", RANK_FEATURES, normalization, response),
            }
        selected = min(comparison, key=lambda key: comparison[key]["validation"]["rmse"])
        validation = comparison[selected]["validation"]
        reporting = comparison[selected]["heldout_reporting_only"]
        comparison["selection"] = {
            "chosen_on_validation": selected,
            "validation_beats_mean": validation["rmse"] < validation["mean_baseline_rmse"],
            "heldout_beats_mean": reporting["rmse"] < reporting["mean_baseline_rmse"],
            "usable_for_bounded_ranking": (
                validation["rmse"] < validation["mean_baseline_rmse"]
                and reporting["rmse"] < reporting["mean_baseline_rmse"]
            ),
            "rule": "variant chosen on validation RMSE; final holdout is reporting-only and may veto analyst ranking but never retunes the model",
        }
        train_outcome = np.asarray(
            [row["outcomes"][response] for row in rows if row["split"] == "train"],
            dtype=np.float64,
        )
        models[response] = {
            **spec,
            "train_outcome": {
                "mean": float(train_outcome.mean()),
                "scale": float(max(train_outcome.std(), 1e-12)),
            },
            "variants": comparison,
        }

    diagnostic_train = _gam_rows(rows, "train", "realized_diagnostic_features", DIAGNOSTIC_FEATURES,
                                 diagnostic_normalization, "energy_change")
    gamfit.validate_formula(diagnostic_train, DIAGNOSTIC_FORMULA)
    diagnostic_model = gamfit.fit(diagnostic_train, DIAGNOSTIC_FORMULA, family="gaussian")
    diagnostic_path = output / "energy-change-realized-diagnostic.gam"
    diagnostic_model.save(diagnostic_path)
    diagnostic = {
        "status": "descriptive after-the-fact mechanism fit; prohibited for unrun-pair ranking",
        "formula": DIAGNOSTIC_FORMULA,
        "features": list(DIAGNOSTIC_FEATURES),
        "normalization": diagnostic_normalization,
        "native_model": diagnostic_path.name,
        "native_model_sha256": file_sha256(diagnostic_path),
        "validation": _metrics(diagnostic_model, rows, "validation", "realized_diagnostic_features",
                               DIAGNOSTIC_FEATURES, diagnostic_normalization, "energy_change"),
        "heldout_reporting_only": _metrics(diagnostic_model, rows, "heldout", "realized_diagnostic_features",
                                            DIAGNOSTIC_FEATURES, diagnostic_normalization, "energy_change"),
    }
    rows_path = output / "life_rows.json"
    rows_payload = {"format": "chreatures-gxe-life-rows-v1", "rows": rows}
    rows_path.write_text(json.dumps(rows_payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    native_module = Path(__import__("gamfit._rust", fromlist=["_"]).__file__).resolve()
    atlas = {
        "format": "chreatures-native-gxe-transfer-atlas-v1",
        "status": "descriptive association atlas; analyst-only; no causal, feeding, survival, or resident-policy claim",
        "library": {"name": "sauersML/gam", "python_package": "gamfit", "version": GAMFIT_VERSION,
                    "source_commit": GAMFIT_COMMIT, "native_build": gamfit.build_info(),
                    "native_extension_sha256": file_sha256(native_module)},
        "source": {**source, "life_rows_file": rows_path.name, "life_rows_file_sha256": file_sha256(rows_path)},
        "observation_units": {
            "unit": "one completed 2048-tick physical life observation row",
            "rows": len(rows),
            "residents_per_shared_world": 8,
            "statistical_independence_claim": False,
            "warning": (
                "residents shared and could interact within each physical world; within-life ticks, "
                "contacts, and actions are also repeated measurements"
            ),
            "uncertainty_calibration_claim": "none",
        },
        "split": source["split"],
        "split_rows": source["split_rows"],
        "rank_features": list(RANK_FEATURES),
        "feature_sources": {**GENOTYPE_FEATURES, **{name: f"environment.descriptors.{name}" for name in ENVIRONMENT_FEATURES}},
        "normalization": normalization,
        "support": _support(rows, normalization, search),
        "models": models,
        "realized_action_physiology_diagnostic": diagnostic,
        "rank_contract": {
            "request_format": REQUEST_FORMAT,
            "output": "response vector plus per-axis additive/interaction comparison and support; never a scalar fitness",
            "unsupported": "new genome/environment schemas, values outside training bounds, or distant candidate/environment vectors return unranked",
            "resident_boundary": "environment descriptors, archive hashes, fit predictions, and support diagnostics are analyst-only",
        },
        "fit_seconds": time.perf_counter() - started,
    }
    atlas_path = output / "atlas.json"
    atlas_path.write_text(json.dumps(atlas, indent=2, sort_keys=True, allow_nan=False) + "\n")
    receipt = {
        "format": "chreatures-native-gxe-transfer-atlas-receipt-v1",
        "sha256": "",
        "atlas_file_sha256": file_sha256(atlas_path),
        "life_rows_file_sha256": file_sha256(rows_path),
        "native_models": {path.name: file_sha256(path) for path in sorted(output.glob("*.gam"))},
    }
    receipt["sha256"] = canonical_sha256(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return atlas


def _distance(vector: np.ndarray, rows: list[list[float]]) -> float:
    matrix = np.asarray(rows, dtype=np.float64)
    return float(np.sqrt(np.min(np.sum(np.square(matrix - vector[None, :]), axis=1))))


def rank_pairs(atlas_path: Path, search_path: Path, request_path: Path) -> dict[str, Any]:
    """Evaluate authenticated search records through saved native GAM models."""
    import gamfit
    atlas, search, request = _load_json(atlas_path), _load_json(search_path), _load_json(request_path)
    if atlas.get("format") != "chreatures-native-gxe-transfer-atlas-v1" or request.get("format") != REQUEST_FORMAT:
        raise ValueError("atlas or ranking request format differs")
    if gamfit.__version__ != atlas["library"]["version"] or not gamfit.build_info().get("available"):
        raise RuntimeError("pinned native gamfit runtime is unavailable")
    if search.get("format") != SEARCH_FORMAT:
        raise ValueError("search format differs")
    from chreatures.population import content_sha256
    normalization = atlas["normalization"]
    mean = np.asarray([normalization[name]["mean"] for name in RANK_FEATURES])
    scale = np.asarray([normalization[name]["scale"] for name in RANK_FEATURES])
    support = atlas["support"]
    results = []
    loaded: dict[str, Any] = {}
    for pair in request.get("pairs", []):
        candidate_id, environment_id = str(pair["candidate_sha256"]), str(pair["environment_sha256"])
        candidate, environment = search.get("genomes", {}).get(candidate_id), search.get("environments", {}).get(environment_id)
        reasons: list[str] = []
        if not candidate or candidate.get("sha256") != candidate_id:
            reasons.append("candidate_absent_from_authenticated_search")
        if not environment or environment.get("sha256") != environment_id:
            reasons.append("environment_absent_from_authenticated_search")
        if reasons:
            results.append({"candidate_sha256": candidate_id, "environment_sha256": environment_id,
                            "eligible_for_ranking": False, "support_reasons": reasons, "response_vector": None})
            continue
        if content_sha256(candidate) != candidate_id:
            reasons.append("candidate_content_hash_invalid")
        if content_sha256(environment) != environment_id:
            reasons.append("environment_content_hash_invalid")
        if candidate.get("format") != atlas["source"]["genome_format"] or sorted(candidate.get("values", {})) != atlas["source"]["genome_value_keys"]:
            reasons.append("genome_mechanism_schema_out_of_support")
        if any(
            candidate.get(name) != expected
            for name, expected in atlas["source"]["genome_context"].items()
        ):
            reasons.append("genome_controller_or_interface_out_of_support")
        if environment.get("format") != atlas["source"]["environment_format"] or sorted(environment.get("descriptors", {})) != atlas["source"]["environment_descriptor_keys"]:
            reasons.append("environment_mechanism_schema_out_of_support")
        if environment.get("profile_sha256") not in atlas["source"]["environment_profile_sha256s"]:
            reasons.append("environment_profile_out_of_support")
        values = candidate["values"]
        descriptors = environment["descriptors"]
        raw = np.asarray([values[GENOTYPE_FEATURES[name]] for name in GENOTYPE_FEATURES]
                         + [descriptors[name] for name in ENVIRONMENT_FEATURES], dtype=np.float64)
        if not np.isfinite(raw).all():
            reasons.append("nonfinite_modeled_feature")
        for name, bounds in support["full_genome_value_bounds"].items():
            value = float(values.get(name, math.nan))
            if not math.isfinite(value) or value < bounds["minimum"] or value > bounds["maximum"]:
                reasons.append(f"genome_value_out_of_support:{name}")
        for index, name in enumerate(RANK_FEATURES):
            bounds = support["modeled_bounds"][name]
            if raw[index] < bounds["minimum"] or raw[index] > bounds["maximum"]:
                reasons.append(f"modeled_feature_out_of_support:{name}")
        z = (raw - mean) / scale
        candidate_distance = _distance(z[:len(GENOTYPE_FEATURES)], support["train_candidate_vectors"])
        environment_distance = _distance(z[len(GENOTYPE_FEATURES):], support["train_environment_vectors"])
        if candidate_distance > support["candidate_distance_threshold"]:
            reasons.append("candidate_vector_distant_from_training_support")
        if environment_distance > support["environment_distance_threshold"]:
            reasons.append("environment_vector_distant_from_training_support")
        eligible = not reasons
        response_vector = None
        if eligible:
            item = {name: float(z[index]) for index, name in enumerate(RANK_FEATURES)}
            response_vector = {}
            for response, spec in atlas["models"].items():
                estimates = {}
                for variant in ("additive", "interaction"):
                    model_meta = spec["variants"][variant]
                    model_name = model_meta["native_model"]
                    model_path = atlas_path.parent / model_name
                    if model_name not in loaded:
                        if file_sha256(model_path) != model_meta["native_model_sha256"]:
                            raise ValueError(f"native model hash differs: {model_name}")
                        loaded[model_name] = gamfit.load(model_path)
                    model = loaded[model_name]
                    estimates[variant] = _inverse_response(float(model.predict([item])[0]), response)
                selection = spec["variants"]["selection"]
                response_vector[response] = {
                    "unit": spec["unit"],
                    "additive_estimate": estimates["additive"],
                    "interaction_estimate": estimates["interaction"],
                    "interaction_gap": estimates["interaction"] - estimates["additive"],
                    "selected_variant": selection["chosen_on_validation"],
                    "selected_estimate": estimates[selection["chosen_on_validation"]] if selection["usable_for_bounded_ranking"] else None,
                    "usable_for_bounded_ranking": selection["usable_for_bounded_ranking"],
                }
        results.append({
            "candidate_sha256": candidate_id,
            "environment_sha256": environment_id,
            "eligible_for_ranking": eligible,
            "support_reasons": sorted(set(reasons)),
            "support": {
                "candidate_distance": candidate_distance,
                "candidate_distance_threshold": support["candidate_distance_threshold"],
                "environment_distance": environment_distance,
                "environment_distance_threshold": support["environment_distance_threshold"],
                "pair_observed": f"{candidate_id}:{environment_id}" in support["observed_pairs"],
            },
            "response_vector": response_vector,
        })
    return {
        "format": "chreatures-gxe-ranking-response-v1",
        "atlas_file_sha256": file_sha256(atlas_path),
        "search_file_sha256": file_sha256(search_path),
        "interpretation": "analyst-side descriptive associations; response vector is not a scalar fitness or resident input",
        "pairs": results,
    }


def proposal_scores(
    ranking: dict[str, Any], atlas_path: Path, criterion: str, policy_output: Path,
    *, limit: int | None = None,
) -> dict[str, Any]:
    """Emit current native population-core scores for one declared criterion."""
    from chreatures.population import content_sha256

    atlas = _load_json(atlas_path)
    if criterion != "interaction_information" and criterion not in RESPONSES:
        raise ValueError(f"unknown proposal criterion {criterion}")
    scores: dict[str, float] = {}
    excluded_reasons: dict[str, int] = {}
    excluded_pairs = 0
    for pair in ranking["pairs"]:
        key = f'{pair["candidate_sha256"]}:{pair["environment_sha256"]}'
        if not pair["eligible_for_ranking"] or pair["response_vector"] is None:
            excluded_pairs += 1
            for reason in pair["support_reasons"]:
                excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
            continue
        vector = pair["response_vector"]
        if criterion == "interaction_information":
            supported = [
                abs(item["interaction_gap"])
                / atlas["models"][name]["train_outcome"]["scale"]
                for name, item in vector.items()
                if item["usable_for_bounded_ranking"]
            ]
            if not supported:
                excluded_pairs += 1
                reason = "no_response_passed_grouped_validation"
                excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
                continue
            score = math.tanh(max(supported))
        else:
            item = vector[criterion]
            if not item["usable_for_bounded_ranking"]:
                excluded_pairs += 1
                reason = f"response_failed_grouped_validation:{criterion}"
                excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
                continue
            stats = atlas["models"][criterion]["train_outcome"]
            score = math.tanh(
                (item["selected_estimate"] - stats["mean"]) / stats["scale"]
            )
        scores[key] = float(max(-1.0, min(1.0, score)))
    if not scores:
        raise ValueError("no supported pairs can receive proposal scores")
    eligible_pairs = len(scores)
    if limit is not None:
        if limit < 1:
            raise ValueError("proposal score limit must be positive")
        scores = dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit])
    policy = {
        "format": "chreatures-gxe-proposal-policy-v1",
        "atlas_file_sha256": file_sha256(atlas_path),
        "ranking_search_file_sha256": ranking["search_file_sha256"],
        "criterion": criterion,
        "direction": (
            "larger absolute validated interaction-versus-additive gap is more informative"
            if criterion == "interaction_information"
            else f"larger predicted {criterion} receives a larger experiment-selection score"
        ),
        "normalization": (
            "tanh of train-standardized response; interaction-information uses "
            "the maximum absolute standardized gap"
        ),
        "scope": (
            "campaign analyst experiment selection only; not scalar fitness, "
            "organism preference, or resident input"
        ),
        "scored_pairs": len(scores),
        "eligible_pairs_before_limit": eligible_pairs,
        "limit": limit,
        "excluded_pairs": excluded_pairs,
        "excluded_reason_counts": dict(sorted(excluded_reasons.items())),
    }
    policy_output.parent.mkdir(parents=True, exist_ok=True)
    policy_output.write_text(
        json.dumps(policy, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    proposal = {
        "format": "chreatures-population-challenge-scores-v1",
        "sha256": "",
        "artifact_sha256": file_sha256(policy_output),
        "scores": scores,
    }
    proposal["sha256"] = content_sha256(proposal)
    return proposal
