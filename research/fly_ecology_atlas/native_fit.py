#!/usr/bin/env python3
"""Fit native GAM response laws to measured native-host ecology histories."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.dynamics_v2.gam_fit import _require_native_gamfit, _capture_native_stderr
from research.fly_ecology_atlas.prepare_native import FEATURES, FORMAT, RANGES

ENV = ("initial_mean_aperture", "initial_blocked_fraction", "initial_minimum_aperture")
TARGETS = ("mean_route_aperture", "aperture_change", "resource_retention")
FORMULAS = {
    "joint": "response ~ duchon(" + ",".join((*FEATURES, *ENV)) + ",centers=12)",
    "additive": "response ~ "
    + "+".join(f"s({name},k=3)" for name in (*FEATURES, *ENV)),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def metrics(run):
    first, last = run["initial"], run["final"]
    return {
        "initial_mean_aperture": first["mean_route_aperture"],
        "initial_blocked_fraction": first["blocked_route_fraction"],
        "initial_minimum_aperture": first["minimum_route_aperture"],
        "mean_route_aperture": last["mean_route_aperture"],
        "aperture_change": last["mean_route_aperture"] - first["mean_route_aperture"],
        "resource_retention": last["colony_resource"] - first["colony_resource"],
    }


def score(pred, truth):
    error = np.asarray(pred) - np.asarray(truth)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs(error))),
        "max_abs": float(np.max(abs(error))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixed-proposals", type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    plan_hash = sha(a.plan)
    records = []
    for genotype in plan["genotypes"]:
        for layout in [x for x in plan["layouts"] if x["split"] == "fit-layout"]:
            path = a.results / f"{genotype['genotype_id']}--{layout['layout_id']}.json"
            run = json.loads(path.read_text())
            if (
                not run["completed"]
                or run["genotype"] != genotype
                or run["layout"] != layout
                or run["provenance"]["plan_sha256"] != plan_hash
            ):
                raise ValueError(f"invalid/missing sealed run {path}")
            records.append(
                {
                    "genotype": genotype,
                    "layout": layout["layout_id"],
                    "sha256": sha(path),
                    "metrics": metrics(run),
                }
            )
    train = [r for r in records if r["genotype"]["split"] == "fit"]
    holdout = [r for r in records if r["genotype"]["split"] == "genotype-holdout"]
    gam = _require_native_gamfit()
    a.output.mkdir(parents=True, exist_ok=False)
    warnings = []
    models = {}
    diagnostics = {}

    def rows(items, target):
        return [
            {
                **r["genotype"]["genes"],
                **{name: r["metrics"][name] for name in ENV},
                "response": r["metrics"][target],
            }
            for r in items
        ]

    def fit(items, target, formula):
        data = rows(items, target)
        gam.validate_formula(data, formula, family="gaussian")
        model, msg = _capture_native_stderr(
            lambda: gam.fit(data, formula, family="gaussian")
        )
        warnings.extend(msg)
        return model

    for target in TARGETS:
        truth = np.array([r["metrics"][target] for r in train])
        entry = {
            "training_rows": len(train),
            "genotype_holdout_rows": len(holdout),
            "sd": float(truth.std()),
            "models": {},
        }
        if np.ptp(truth) <= 1e-12:
            entry["status"] = "constant-not-fitted"
            diagnostics[target] = entry
            continue
        for label, formula in FORMULAS.items():
            try:
                model = fit(train, target, formula)
                artifact = a.output / f"{target}-{label}.gam"
                model.save(artifact)
                restored = gam.load(artifact)
                models[target, label] = restored
                train_pred = np.asarray(restored.predict(rows(train, target))).reshape(
                    -1
                )
                hold_pred = np.asarray(restored.predict(rows(holdout, target))).reshape(
                    -1
                )
                genotype_pred = []
                genotype_truth = []
                for genotype_id in sorted(
                    {r["genotype"]["genotype_id"] for r in train}
                ):
                    test = [
                        r for r in train if r["genotype"]["genotype_id"] == genotype_id
                    ]
                    fold = [
                        r for r in train if r["genotype"]["genotype_id"] != genotype_id
                    ]
                    genotype_pred.extend(
                        np.asarray(
                            fit(fold, target, formula).predict(rows(test, target))
                        ).reshape(-1)
                    )
                    genotype_truth.extend(r["metrics"][target] for r in test)
                layout_pred = []
                layout_truth = []
                for layout in sorted({r["layout"] for r in train}):
                    test = [r for r in train if r["layout"] == layout]
                    fold = [r for r in train if r["layout"] != layout]
                    layout_pred.extend(
                        np.asarray(
                            fit(fold, target, formula).predict(rows(test, target))
                        ).reshape(-1)
                    )
                    layout_truth.extend(r["metrics"][target] for r in test)
                entry["models"][label] = {
                    "status": "native-fit-serialized-reloaded",
                    "formula": formula,
                    "artifact_sha256": sha(artifact),
                    "train": score(train_pred, truth),
                    "fit_leave_genotype_out": score(genotype_pred, genotype_truth),
                    "validation_genotype_holdout": score(
                        hold_pred, [r["metrics"][target] for r in holdout]
                    ),
                    "fit_layout_holdout": score(layout_pred, layout_truth),
                }
            except Exception as error:
                entry["models"][label] = {"status": "rejected", "error": str(error)}
        diagnostics[target] = entry
    chosen = {}
    for target, entry in diagnostics.items():
        valid = [
            (
                data["fit_leave_genotype_out"]["rmse"]
                + data["fit_layout_holdout"]["rmse"],
                label,
            )
            for label, data in entry["models"].items()
            if data["status"].startswith("native-fit")
        ]
        if valid:
            chosen[target] = min(valid)[1]
    proposals = []
    if len(chosen) == len(TARGETS):
        rng = np.random.default_rng(plan["seed"] + 91)
        points = [
            {
                name: float(lo + u * (hi - lo))
                for name, (lo, hi), u in zip(FEATURES, RANGES, row)
            }
            for row in rng.random((512, len(FEATURES)))
        ]
        # Candidate selection integrates across the three independently measured layouts.
        prediction = {target: [] for target in TARGETS}
        environment = []
        for layout in sorted({r["layout"] for r in train}):
            record = next(r for r in train if r["layout"] == layout)
            environment.append({name: record["metrics"][name] for name in ENV})
        for point in points:
            for target in TARGETS:
                query = [{**point, **env, "response": 0.0} for env in environment]
                prediction[target].append(
                    float(
                        np.mean(
                            np.asarray(models[target, chosen[target]].predict(query))
                        )
                    )
                )
        normalized = {
            key: (np.asarray(value) - np.mean(value)) / max(np.std(value), 1e-12)
            for key, value in prediction.items()
        }
        objectives = {
            "open-resource": normalized["mean_route_aperture"]
            + normalized["resource_retention"],
            "aperture-stability": normalized["mean_route_aperture"]
            + normalized["aperture_change"],
        }
        used = []
        for intent, objective in objectives.items():
            index = next(
                int(i)
                for i in np.argsort(objective)[::-1]
                if all(
                    np.linalg.norm(
                        [
                            (points[i][k] - points[j][k]) / (hi - lo)
                            for k, (lo, hi) in zip(FEATURES, RANGES)
                        ]
                    )
                    > 0.25
                    for j in used
                )
            )
            used.append(index)
            proposals.append(
                {
                    "genotype_id": "confirmation-" + intent,
                    "split": "confirmation",
                    "genes": points[index],
                    "intent": intent,
                    "predicted_fit_layout_mean": {
                        target: prediction[target][index] for target in TARGETS
                    },
                }
            )
    if a.fixed_proposals:
        parent = json.loads(a.fixed_proposals.read_text())
        proposals = []
        for old in parent["proposals"]:
            point = old["genes"]
            predicted = {}
            for target in TARGETS:
                query = [{**point, **env, "response": 0.0} for env in environment]
                predicted[target] = float(
                    np.mean(np.asarray(models[target, chosen[target]].predict(query)))
                )
            proposals.append(
                {
                    "genotype_id": old["genotype_id"],
                    "split": "confirmation",
                    "genes": point,
                    "intent": old["intent"],
                    "predicted_fit_layout_mean": predicted,
                    "proposal_origin": "preserved-parent-analysis; branching objective disqualified",
                }
            )
    report = {
        "format": "chreatures-native-fly-ecology-gam-v2",
        "status": "confirmation-pending" if proposals else "no-promotable-candidate",
        "plan_sha256": plan_hash,
        "records": records,
        "targets": TARGETS,
        "predictors": (*FEATURES, *ENV),
        "formulas": FORMULAS,
        "diagnostics": diagnostics,
        "selected_models": chosen,
        "proposals": proposals,
        "native_gam": gam.build_info(),
        "native_messages": warnings,
        "claim_limit": plan["claim_limit"],
    }
    write(a.output / "report.json", report)
    confirmation = dict(plan)
    confirmation["format"] = "chreatures-native-fly-ecology-atlas-v2"
    confirmation["genotypes"] = proposals
    confirmation["parent_plan_sha256"] = plan_hash
    confirmation["runner"] = str((a.plan.parent / "frozen-native-run.py").resolve())
    write(a.output / "confirmation-plan.json", confirmation)
    print(
        json.dumps(
            {
                "report": str(a.output / "report.json"),
                "sha256": sha(a.output / "report.json"),
                "proposals": len(proposals),
            }
        )
    )


if __name__ == "__main__":
    main()
