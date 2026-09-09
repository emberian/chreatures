#!/usr/bin/env python3
"""Fit native GAMs to committed construction, resource, and aperture responses."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.dynamics_v2.gam_fit import _capture_native_stderr, _require_native_gamfit
from research.fly_ecology_atlas.prepare_native import FEATURES, FORMAT, RANGES


ENV = (
    "initial_route_permeability",
    "initial_blocked_fraction",
    "initial_minimum_aperture",
)
TARGETS = (
    "committed_constructions",
    "construction_material",
    "colony_resource_change",
    "route_permeability",
    "route_permeability_change",
)
MODEL_FEATURES = (*FEATURES, *ENV)
FORMULAS = {
    "joint": "response ~ duchon(" + ",".join(MODEL_FEATURES) + ",centers=12)",
    "additive": "response ~ "
    + "+".join(f"s({name},k=3)" for name in MODEL_FEATURES),
}
REPORT_FORMAT = "chreatures-native-fly-ecology-gam-v3"


def sha(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def scores(prediction: object, truth: object) -> dict:
    prediction = np.asarray(prediction, np.float64)
    truth = np.asarray(truth, np.float64)
    error = prediction - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def canopy_metrics(structures: list[dict]) -> dict:
    if not structures:
        return {"height_mm": 0.0, "span_mm": 0.0, "retained_structures": 0}
    centers = np.asarray([item["position_m"] for item in structures], np.float64)
    radii = np.asarray([item["nominal_radius_m"] for item in structures], np.float64)
    half_lengths = 0.5 * np.asarray(
        [item["nominal_length_m"] for item in structures], np.float64
    )
    # Orientation is xyzw and the authored capsule axis is local +z.
    axes = []
    for item in structures:
        x, y, z, w = item["orientation_xyzw"]
        axes.append(
            [
                2 * (x * z + w * y),
                2 * (y * z - w * x),
                1 - 2 * (x * x + y * y),
            ]
        )
    extents = np.abs(np.asarray(axes)) * half_lengths[:, None] + radii[:, None]
    low = centers - extents
    high = centers + extents
    return {
        "height_mm": float(1000 * high[:, 2].max()),
        "span_mm": float(1000 * np.linalg.norm(high[:, :2].max(0) - low[:, :2].min(0))),
        "retained_structures": len(structures),
    }


def metrics(run: dict) -> dict:
    first = run["initial"]
    last = run["final"]
    first_growth = first["growth"]
    last_growth = last["growth"]
    if first["route_plan_sha256"] != last["route_plan_sha256"]:
        raise ValueError("route plan changed within one independent world")
    # The native host's tick-zero observer precedes its first physical aperture
    # refresh and therefore reports the core's neutral all-open initialization.
    # Use the earliest measured sample which still has the tick-zero topology
    # and construction count.  In the frozen campaign this is tick 100, before
    # the first development opportunity at tick 200.  Resource chronology still
    # starts at tick zero below.
    environment = next(
        (
            sample
            for sample in run["trace"][1:]
            if sample["topology_revision"] == first["topology_revision"]
            and sample["growth"]["committed_constructions"]
            == first_growth["committed_constructions"]
        ),
        None,
    )
    if environment is None or environment["route_plan_sha256"] != first["route_plan_sha256"]:
        raise ValueError("no measured preconstruction aperture sample")
    constructions = int(last_growth["committed_constructions"]) - int(
        first_growth["committed_constructions"]
    )
    initial_material = np.asarray(
        first_growth["construction_material_allocated"], np.float64
    )
    final_material = np.asarray(
        last_growth["construction_material_allocated"], np.float64
    )
    if initial_material.size == 0 and final_material.size:
        initial_material = np.zeros_like(final_material)
    if constructions < 0 or initial_material.shape != final_material.shape:
        raise ValueError("committed construction chronology differs")
    allocation = final_material - initial_material
    if (allocation < -1e-12).any():
        raise ValueError("construction allocation decreased")
    if constructions != last["new_committed_structure_count"]:
        raise ValueError("committed counter differs from retained physical structures")
    canopy = canopy_metrics(last["new_committed_structures"])
    if canopy["retained_structures"] < constructions:
        raise ValueError("retained committed geometry differs")
    result = {
        "environment_measurement_tick": int(environment["tick"]),
        "initial_route_permeability": environment["route"]["area_weighted_permeability"],
        "initial_blocked_fraction": environment["route"]["blocked_fraction"],
        "initial_minimum_aperture": environment["route"]["minimum_aperture"],
        "committed_constructions": float(constructions),
        "construction_material": float(allocation.sum()),
        "construction_material_by_pool": allocation.tolist(),
        "colony_resource_change": last["colony_resource"] - first["colony_resource"],
        "route_permeability": last["route"]["area_weighted_permeability"],
        "route_permeability_change": last["route"]["area_weighted_permeability"]
        - environment["route"]["area_weighted_permeability"],
        "captured_photon_energy": last["captured_photon_energy"]
        - first["captured_photon_energy"],
        "clearance_approved": int(last_growth["construction_clearance_approved"])
        - int(first_growth["construction_clearance_approved"]),
        "clearance_blocked": int(last_growth["construction_clearance_blocked"])
        - int(first_growth["construction_clearance_blocked"]),
        "resource_rejected": int(last_growth["construction_resource_rejected"])
        - int(first_growth["construction_resource_rejected"]),
        "canopy": canopy,
        "route_plan_sha256": last["route_plan_sha256"],
        "topology_revision_change": last["topology_revision"]
        - first["topology_revision"],
        "maximum_elemental_residual": last["maximum_elemental_residual"],
    }
    if not np.isfinite(
        [result[name] for name in (*ENV, *TARGETS, "captured_photon_energy")]
    ).all():
        raise ValueError("nonfinite ecology response")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("format") != FORMAT or plan.get("historical_aperture_rows_reused"):
        raise ValueError("current committed-response plan differs")
    plan_hash = sha(args.plan)
    records = []
    failures = []
    for unit in plan["units"]:
        path = args.results / f"{unit['unit_id']}.json"
        if not path.exists():
            raise ValueError(f"missing independent unit {unit['unit_id']}")
        run = json.loads(path.read_text())
        if (
            run.get("unit") != unit
            or run.get("provenance", {}).get("plan_sha256") != plan_hash
            or run.get("provenance", {}).get("runner_sha256") != plan["runner_sha256"]
            or run.get("provenance", {}).get("native_binary_sha256")
            != plan["native_binary_sha256"]
        ):
            raise ValueError(f"unit identity differs: {unit['unit_id']}")
        item = {
            "unit": unit,
            "sha256": sha(path),
            "error": run.get("error"),
        }
        if not run.get("completed"):
            failures.append(item)
        else:
            item["metrics"] = metrics(run)
            records.append(item)
    train = [item for item in records if item["unit"]["genotype"]["split"] == "fit"]
    holdout = [
        item
        for item in records
        if item["unit"]["genotype"]["split"] == "genotype-holdout"
    ]
    for layout_id in {item["unit"]["layout"]["layout_id"] for item in records}:
        route_plans = {
            item["metrics"]["route_plan_sha256"]
            for item in records
            if item["unit"]["layout"]["layout_id"] == layout_id
        }
        if len(route_plans) != 1:
            raise ValueError(f"route plan identity varies within layout {layout_id}")
    if len(train) < 30 or len({r["unit"]["genotype"]["genotype_id"] for r in train}) < 10:
        raise ValueError("insufficient independent fitting units")
    args.output.mkdir(parents=True, exist_ok=False)
    gam = _require_native_gamfit()
    messages = []
    models = {}
    diagnostics = {}

    def rows(items: list[dict], target: str) -> list[dict]:
        return [
            {
                **item["unit"]["genotype"]["genes"],
                **{name: item["metrics"][name] for name in ENV},
                "response": item["metrics"][target],
            }
            for item in items
        ]

    def fit(items: list[dict], target: str, formula: str):
        data = rows(items, target)
        gam.validate_formula(data, formula, family="gaussian")
        model, stderr = _capture_native_stderr(
            lambda: gam.fit(data, formula, family="gaussian")
        )
        messages.extend(stderr)
        return model

    for target in TARGETS:
        truth = np.asarray([item["metrics"][target] for item in train], np.float64)
        baseline = np.full(len(train), truth.mean())
        entry = {
            "training_independent_worlds": len(train),
            "genotype_holdout_worlds": len(holdout),
            "sd": float(truth.std()),
            "mean_baseline": scores(baseline, truth),
            "models": {},
        }
        if np.ptp(truth) <= 1e-12:
            entry["status"] = "constant-not-fitted"
            entry["constant"] = float(truth[0])
            diagnostics[target] = entry
            continue
        for label, formula in FORMULAS.items():
            try:
                model = fit(train, target, formula)
                artifact = args.output / f"{target}-{label}.gam"
                model.save(artifact)
                restored = gam.load(artifact)
                models[target, label] = restored
                fitted = np.asarray(restored.predict(rows(train, target))).reshape(-1)
                reloaded = np.asarray(gam.load(artifact).predict(rows(train, target))).reshape(-1)
                if not np.allclose(fitted, reloaded, rtol=0, atol=1e-10):
                    raise ValueError("serialized GAM reload differs")
                genotype_pred, genotype_truth, genotype_mean = [], [], []
                genotype_ids = sorted(
                    {item["unit"]["genotype"]["genotype_id"] for item in train}
                )
                for genotype_id in genotype_ids:
                    test = [
                        item
                        for item in train
                        if item["unit"]["genotype"]["genotype_id"] == genotype_id
                    ]
                    fold = [item for item in train if item not in test]
                    fold_model = fit(fold, target, formula)
                    genotype_pred.extend(
                        np.asarray(fold_model.predict(rows(test, target))).reshape(-1).tolist()
                    )
                    genotype_truth.extend(item["metrics"][target] for item in test)
                    genotype_mean.extend(
                        [np.mean([item["metrics"][target] for item in fold])] * len(test)
                    )
                layout_pred, layout_truth, layout_mean = [], [], []
                for layout_id in sorted({item["unit"]["layout"]["layout_id"] for item in train}):
                    test = [
                        item for item in train if item["unit"]["layout"]["layout_id"] == layout_id
                    ]
                    fold = [item for item in train if item not in test]
                    fold_model = fit(fold, target, formula)
                    layout_pred.extend(
                        np.asarray(fold_model.predict(rows(test, target))).reshape(-1).tolist()
                    )
                    layout_truth.extend(item["metrics"][target] for item in test)
                    layout_mean.extend(
                        [np.mean([item["metrics"][target] for item in fold])] * len(test)
                    )
                hold_truth = [item["metrics"][target] for item in holdout]
                entry["models"][label] = {
                    "status": "native-fit-serialized-reloaded",
                    "formula": formula,
                    "artifact_sha256": sha(artifact),
                    "reload_max_abs": float(np.max(np.abs(fitted - reloaded))),
                    "training": scores(fitted, truth),
                    "leave_genotype_out": scores(genotype_pred, genotype_truth),
                    "leave_genotype_out_mean": scores(genotype_mean, genotype_truth),
                    "leave_layout_out": scores(layout_pred, layout_truth),
                    "leave_layout_out_mean": scores(layout_mean, layout_truth),
                    "genotype_holdout": scores(
                        np.asarray(restored.predict(rows(holdout, target))).reshape(-1),
                        hold_truth,
                    )
                    if holdout
                    else None,
                    "genotype_holdout_mean": scores(
                        np.full(len(holdout), truth.mean()), hold_truth
                    )
                    if holdout
                    else None,
                }
            except Exception as exc:
                models.pop((target, label), None)
                entry["models"][label] = {
                    "status": "native-fit-or-validation-rejected",
                    "error": str(exc),
                }
        diagnostics[target] = entry

    selected_models = {}
    for target, entry in diagnostics.items():
        candidates = [
            (model["leave_genotype_out"]["rmse"], label)
            for label, model in entry.get("models", {}).items()
            if model["status"] == "native-fit-serialized-reloaded"
        ]
        if candidates:
            selected_models[target] = min(candidates)[1]

    # The GAM selects a diverse *set*: no scalar utility and no promoted winner.
    rng = np.random.default_rng(plan["seed"] + 701)
    gene_units = rng.uniform(0.025, 0.975, (768, len(FEATURES)))
    genes = [
        {
            name: float(lo + value[j] * (hi - lo))
            for j, (name, (lo, hi)) in enumerate(zip(FEATURES, RANGES))
        }
        for value in gene_units
    ]
    layout_env = []
    for layout_id in sorted({item["unit"]["layout"]["layout_id"] for item in train}):
        cells = [item for item in train if item["unit"]["layout"]["layout_id"] == layout_id]
        layout_env.append(
            {
                "layout_id": layout_id,
                **{name: float(np.median([item["metrics"][name] for item in cells])) for name in ENV},
            }
        )
    predictions = {}
    prediction_by_layout = {}
    for target in TARGETS:
        if target in selected_models:
            model = models[target, selected_models[target]]
            by_layout = []
            for environment in layout_env:
                query = [
                    {**gene, **{name: environment[name] for name in ENV}, "response": 0.0}
                    for gene in genes
                ]
                by_layout.append(np.asarray(model.predict(query)).reshape(-1))
            prediction_by_layout[target] = np.stack(by_layout, axis=1)
            predictions[target] = prediction_by_layout[target].mean(1)
        elif diagnostics[target].get("status") == "constant-not-fitted":
            predictions[target] = np.full(768, diagnostics[target]["constant"])
            prediction_by_layout[target] = np.repeat(
                predictions[target][:, None], len(layout_env), axis=1
            )

    proposal_targets = [
        name
        for name in TARGETS
        if name in predictions and np.ptp(predictions[name]) > 1e-12
    ]
    proposals = []
    if len(proposal_targets) >= 2:
        matrix = np.stack([predictions[name] for name in proposal_targets], axis=1)
        matrix = (matrix - matrix.mean(0)) / np.maximum(matrix.std(0), 1e-12)
        eligible = np.flatnonzero(predictions["committed_constructions"] >= 0.5)
        if len(eligible):
            selected = [int(eligible[np.argmax(np.linalg.norm(matrix[eligible], axis=1))])]
            while len(selected) < 4:
                response_distance = np.min(
                    np.linalg.norm(matrix[eligible, None, :] - matrix[selected][None, :, :], axis=2),
                    axis=1,
                )
                gene_distance = np.min(
                    np.linalg.norm(gene_units[eligible, None, :] - gene_units[selected][None, :, :], axis=2),
                    axis=1,
                )
                score = np.minimum(response_distance, gene_distance)
                score[np.isin(eligible, selected)] = -np.inf
                candidate = int(eligible[np.argmax(score)])
                if not np.isfinite(score.max()):
                    break
                selected.append(candidate)
            for rank, index in enumerate(selected):
                proposal = {
                    "genotype_id": f"confirmation-diversity-{rank:02d}",
                    "split": "confirmation",
                    "genes": genes[index],
                    "predicted_fit_layout_mean": {
                        name: float(predictions[name][index]) for name in predictions
                    },
                    "predicted_fit_layout_range": {
                        name: [
                            float(prediction_by_layout[name][index].min()),
                            float(prediction_by_layout[name][index].max()),
                        ]
                        for name in prediction_by_layout
                    },
                    "selection": "native-GAM maximin diversity in predicted joint construction/resource/aperture response and gene space; no scalar winner or promotion",
                }
                write(args.output / f"{proposal['genotype_id']}.json", proposal)
                proposals.append(proposal)

    environment_priorities = []
    if selected_models:
        for environment_index, environment in enumerate(layout_env):
            disagreements = []
            response_spreads = []
            for target, label in selected_models.items():
                scale = max(diagnostics[target]["sd"], 1e-12)
                selected = prediction_by_layout[target][:, environment_index]
                response_spreads.append(float(selected.std() / scale))
                alternate = "additive" if label == "joint" else "joint"
                alternate_model = models.get((target, alternate))
                if alternate_model is not None:
                    query = [
                        {
                            **gene,
                            **{name: environment[name] for name in ENV},
                            "response": 0.0,
                        }
                        for gene in genes
                    ]
                    other = np.asarray(alternate_model.predict(query)).reshape(-1)
                    disagreements.append(float(np.mean(np.abs(selected - other)) / scale))
            disagreement = float(np.mean(disagreements)) if disagreements else 0.0
            spread = float(np.mean(response_spreads)) if response_spreads else 0.0
            environment_priorities.append(
                {
                    **environment,
                    "normalized_model_disagreement": disagreement,
                    "normalized_predicted_response_spread": spread,
                    "priority_score": disagreement + 0.1 * spread,
                    "use": "compose a distinct physical obstruction/light layout toward this measured aperture signature, then accept it only after native initial aperture measurement",
                }
            )
        environment_priorities.sort(key=lambda item: item["priority_score"], reverse=True)

    report = {
        "format": REPORT_FORMAT,
        "status": "native-gam-diversity-confirmation-pending"
        if proposals
        else "native-fit-no-diverse-feasible-set",
        "plan_sha256": plan_hash,
        "source_sha256": sha(__file__),
        "native_gam": gam.build_info(),
        "native_version": gam.__version__,
        "formulas": FORMULAS,
        "experimental_unit": plan["experimental_unit"],
        "records": records,
        "failed_units": failures,
        "diagnostics": diagnostics,
        "selected_models": selected_models,
        "diversity_response_axes": proposal_targets,
        "confirmation_proposals": proposals,
        "next_environment_priorities": environment_priorities,
        "historical_aperture_rows_reused": False,
        "native_messages": messages,
        "claim_limit": plan["claim_limit"],
    }
    write(args.output / "report.json", report)
    write(
        args.output / "receipt.json",
        {
            "format": REPORT_FORMAT,
            "status": report["status"],
            "plan_sha256": plan_hash,
            "report_sha256": sha(args.output / "report.json"),
            "source_sha256": sha(__file__),
            "completed_independent_worlds": len(records),
            "failed_independent_worlds": len(failures),
            "native_gam": gam.build_info(),
            "selected_models": selected_models,
            "confirmation_proposals": proposals,
            "historical_aperture_rows_reused": False,
            "claim_limit": plan["claim_limit"],
        },
    )
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "sha256": sha(args.output / "report.json"),
                "proposals": len(proposals),
            }
        )
    )


if __name__ == "__main__":
    main()
