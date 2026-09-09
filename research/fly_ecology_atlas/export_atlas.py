"""Publish recorded construction and conditional native GAM predictions.

This is an observation/export boundary. It never advances a world, fits a model,
or synthesizes a missing branch. Large experimental artifacts stay in paperbin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from integrations.export_live_cns_weave import blob, node, read_json, require, sha256
from research.dynamics_v2.gam_fit import _require_native_gamfit
from research.fly_ecology_atlas.native_fit import ENV, metrics
from research.fly_ecology_atlas.prepare_native import FEATURES, RANGES

ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_KEYS = (
    "id", "owner_id", "physics_binding", "active", "position_m",
    "orientation_xyzw", "nominal_length_m", "nominal_radius_m",
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def project_layout(layout: dict) -> dict:
    fixture_path = Path(layout["fixture"])
    fixture = read_json(fixture_path)
    require(sha256(fixture_path) == layout["fixture_sha256"], "layout fixture changed")
    habitat_path = fixture_path.parent / "habitat-plan.json"
    require(sha256(habitat_path) == fixture["habitat_plan_sha256"], "habitat geometry changed")
    habitat = read_json(habitat_path)
    return {
        **{key: layout[key] for key in ("layout_id", "description", "split", "fixture_sha256")},
        "habitat_plan_sha256": sha256(habitat_path),
        "bounds_mm": habitat["bounds_mm"],
        "geometries": [item for item in habitat["geometries"] if not item["dynamic"]],
        "geometry_scope": "Static authored habitat only; moving flies and resource packets are omitted.",
    }


def project_world(run: dict, path: Path, plan: dict, plan_hash: str) -> dict:
    provenance = run["provenance"]
    require(provenance["plan_sha256"] == plan_hash, "world plan differs")
    require(provenance["native_binary_sha256"] == plan["native_binary_sha256"], "world engine differs")
    require(provenance["runner_sha256"] == plan["runner_sha256"], "world runner differs")
    unit = run["unit"]
    require(unit in plan["units"], "world is outside declared campaign")
    result = {
        "unit_id": unit["unit_id"], "genotype_id": unit["genotype"]["genotype_id"],
        "layout_id": unit["layout"]["layout_id"], "split": unit["genotype"]["split"],
        "genes": unit["genotype"]["genes"], "sha256": sha256(path),
        "completed": run["completed"], "error": run["error"],
        "metrics": metrics(run) if run["completed"] else None,
        "predictions": {}, "samples": [],
    }
    result["baselines"] = {} if not run["completed"] else {
        "route_permeability": {"value": result["metrics"]["initial_route_permeability"], "label": "Initial measured aperture"},
        "route_permeability_change": {"value": 0.0, "label": "No aperture change"},
    }
    for sample in run["trace"]:
        growth = sample["growth"]
        result["samples"].append({
            "tick": sample["tick"], "time_s": sample["time_s"],
            "committed_constructions": growth["committed_constructions"],
            "construction_material": sum(growth["construction_material_allocated"]),
            "colony_resource": sample["colony_resource"],
            # Tick zero is the host's unsampled all-open placeholder, not a ray measurement.
            "route_permeability": None if sample["tick"] == 0 else sample["route"]["area_weighted_permeability"],
            "structures": [{key: item[key] for key in STRUCTURE_KEYS}
                           for item in sample["committed_structures"]],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--confirmation-results", type=Path)
    parser.add_argument("--public", type=Path, default=ROOT / "site/assets/native-growth-atlas.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cargo-target-dir", type=Path, required=True)
    args = parser.parse_args()
    require(bool(args.confirmation_plan) == bool(args.confirmation_results), "confirmation inputs must be paired")
    plan, fit = read_json(args.plan), read_json(args.fit / "report.json")
    plan_hash = sha256(args.plan)
    fit_hash = sha256(args.fit / "report.json")
    require(fit["plan_sha256"] == plan_hash, "GAM fit plan differs")
    require(read_json(args.fit / "receipt.json")["report_sha256"] == fit_hash, "GAM fit receipt differs")
    args.output.mkdir(parents=True, exist_ok=False)
    campaigns = [(args.plan, plan, args.results)]
    if args.confirmation_plan:
        confirmation = read_json(args.confirmation_plan)
        require(confirmation["native_binary_sha256"] == plan["native_binary_sha256"], "confirmation engine differs")
        require(confirmation["parent_plan_sha256"] == plan_hash and confirmation["fit_report_sha256"] == fit_hash,
                "confirmation proposal ancestry differs")
        campaigns.append((args.confirmation_plan, confirmation, args.confirmation_results))
    worlds, layouts, records = [], {}, []
    plan_id = f"committed-growth-plan:{plan_hash}"
    fit_id = f"committed-growth-gam:{fit_hash}"
    plan_parents = {}
    failed_launch = plan.get("supersedes_failed_launch")
    public_failure = None
    if failed_launch:
        failed_plan_path = Path(failed_launch["results"]).parent / "plan.json"
        require(sha256(failed_plan_path) == failed_launch["plan_sha256"], "failed launch ancestry changed")
        failed_plan = read_json(failed_plan_path)
        failures = []
        for unit in failed_plan["units"]:
            path = Path(failed_launch["results"]) / f"{unit['unit_id']}.json"
            run = read_json(path)
            require(not run["completed"] and run["unit"] == unit, "failed launch contains a different result")
            failures.append(blob(path, "failed_world", "application/json"))
        require(len(failures) == failed_launch["failed_units_preserved"], "failed world count differs")
        public_failure = {key: value for key, value in failed_launch.items() if key != "results"}
        failure_id = f"growth-failed-launch:{failed_launch['plan_sha256']}"
        records.append(node(failure_id, 0, "shared_engine_failure", "The preceding launch failed at topology recompilation; all assigned failed histories remain preserved.",
                            blobs=[blob(failed_plan_path, "failed_plan", "application/json"), *failures], fields=public_failure))
        plan_parents[failure_id] = "repaired_launch"
    records.append(node(plan_id, 0, "experimental_plan", "A frozen genotype by habitat experiment in material-funded physical construction.",
                        parents=plan_parents,
                        blobs=[blob(args.plan, "frozen_campaign", "application/json")],
                        fields={"worlds": len(plan["units"]), "independent_unit": plan["experimental_unit"]}))
    fit_world_ids = {}
    for plan_path, campaign, directory in campaigns:
        campaign_hash = sha256(plan_path)
        confirmation = campaign_hash != plan_hash
        campaign_id = f"committed-growth-plan:{campaign_hash}"
        if confirmation:
            records.append(node(campaign_id, 3, "gam_proposal_plan", "The fitted models proposed diverse inherited programs for an untouched physical layout.",
                                parents={plan_id: "parent_experiment", fit_id: "proposal_model"},
                                blobs=[blob(plan_path, "confirmation_plan", "application/json")],
                                fields={"worlds": len(campaign["units"]), "selection": "diverse response set; no scalar winner"}))
        for unit in campaign["units"]:
            layout = unit["layout"]
            if layout["layout_id"] not in layouts:
                layouts[layout["layout_id"]] = project_layout(layout)
            path = directory / f"{unit['unit_id']}.json"
            run = read_json(path)  # Missing worlds are not silently dropped.
            require(run["unit"] == unit, "independent world identity differs")
            row = project_world(run, path, campaign, campaign_hash)
            worlds.append(row)
            world_id = f"committed-growth-world:{row['sha256']}"
            if row["split"] == "fit" and row["completed"]:
                fit_world_ids[world_id] = "fitting_world"
            records.append(node(world_id, 4 if confirmation else 1, "recorded_physical_world",
                "A completed recorded native construction history." if row["completed"] else "A failed assigned world, retained as a failure.",
                parents={campaign_id: "declared_experiment"}, blobs=[blob(path, "world_record", "application/json")],
                fields={key: row[key] for key in ("unit_id", "split", "completed", "error", "metrics")}))
    fit_records = {item["unit"]["unit_id"]: item for item in fit["records"]}
    for world in worlds:
        if world["unit_id"] in fit_records:
            require(fit_records[world["unit_id"]]["sha256"] == world["sha256"], "GAM fitted world bytes changed")
    gam = _require_native_gamfit()
    completed = [world for world in worlds if world["completed"]]
    query = [{**world["genes"], **{name: world["metrics"][name] for name in ENV}, "response": 0.0}
             for world in completed]
    for target, label in fit["selected_models"].items():
        path = args.fit / f"{target}-{label}.gam"
        expected = fit["diagnostics"][target]["models"][label]["artifact_sha256"]
        require(sha256(path) == expected, "selected GAM artifact differs")
        prediction = np.asarray(gam.load(path).predict(query)).reshape(-1)
        require(prediction.shape == (len(completed),) and np.isfinite(prediction).all(), "GAM query result differs")
        for world, value in zip(completed, prediction):
            world["predictions"][target] = {
                "value": float(value), "model": label,
                "scope": "in-fit conditional prediction" if world["split"] == "fit" else "held-out conditional prediction",
                "conditioning": "Measured preconstruction route state; not an ex-ante forecast of an unseen layout.",
            }
    records.append(node(fit_id, 2, "native_gam_response_models",
        "Serialized native GAM models selected by whole-genotype leave-out error; no genotype is promoted.",
        parents=fit_world_ids, blobs=[blob(args.fit / "report.json", "gam_report", "application/json")],
        fields={"selected_models": fit["selected_models"], "diagnostics": fit["diagnostics"]}))
    for world in worlds:
        if world["split"] == "fit" or not world["completed"]:
            continue
        records.append(node(f"growth-comparison:{world['sha256']}", 5, "held_out_response_comparison",
            "Conditional model predictions compared with a withheld world's measured construction and resource outcomes.",
            parents={fit_id: "fitted_model", f"committed-growth-world:{world['sha256']}": "withheld_world"},
            fields={"predictions": world["predictions"], "measurements": world["metrics"], "split": world["split"]}))
    request = {"archive_id": f"committed-growth-{plan_hash[:16]}", "description": "Recorded construction, fitted hypotheses and withheld outcomes.", "evidence": records}
    request_path, weave_path = args.output / "request.json", args.output / "evidence.weave.json"
    write(request_path, request)
    process = subprocess.run(["cargo", "run", "--locked", "--release", "--target-dir", str(args.cargo_target_dir),
        "--manifest-path", str(ROOT / "integrations/weave/Cargo.toml"), "--", "--input", str(request_path), "--output", str(weave_path)],
        check=True, capture_output=True, text=True)
    weave = json.loads(process.stdout)
    require(weave["reload_equal"] and weave["validated_after_reload"] and weave["node_count"] == len(records), "native Weave export failed")
    public_weave = args.public.parent / "native-growth-evidence.weave.json"
    public_weave.write_bytes(weave_path.read_bytes())
    atlas = {
        "format": "chreatures-native-growth-atlas-v1",
        "scope": "Recorded material-funded colony construction in the actual native MuJoCo habitat. Fly controls were fixed at zero; this campaign does not run or train a CNS. No evolved strategy or learned behavior is claimed.",
        "provenance": {"plan_sha256": plan_hash, "fit_sha256": fit_hash, "native_binary_sha256": plan["native_binary_sha256"],
                       "exporter_sha256": sha256(Path(__file__)), "weave_sha256": sha256(weave_path)},
        "genes": [{"name": name, "min": bounds[0], "max": bounds[1]} for name, bounds in zip(FEATURES, RANGES)],
        "layouts": list(layouts.values()), "worlds": worlds,
        "gam": {key: fit[key] for key in ("selected_models", "diagnostics", "native_version", "confirmation_proposals", "next_environment_priorities")},
        "preserved_failure": public_failure,
        "weave": {key: weave[key] for key in ("node_count", "edge_count", "multi_parent_nodes", "reload_equal", "validated_after_reload", "library")},
        "recording": {"sample_interval_s": plan["sample_every_ticks"] * plan["control_dt"], "duration_s": plan["seconds"],
                      "interpolation": "none; each sample contains only already committed recorded structures", "resource_units": "synthetic model pools; colony_resource is a composite index, not joules or conserved mass"},
    }
    atlas["gam"]["formulas"] = fit["formulas"]
    atlas["gam"]["interpretation"] = (
        "Absolute route permeability is already mostly determined by its measured initial value; compare against persistence. "
        "Construction material is 0.08 synthetic units per committed branch in this campaign, so count and material are coupled responses, not independent diversity axes. "
        "Conditional prediction uses the recorded world's preconstruction route measurement; confirmation proposal forecasts were based on the three fitting layouts instead."
    )
    write(args.public, atlas)
    receipt = {"format": "chreatures-native-growth-atlas-export-v1", "atlas_sha256": sha256(args.public),
               "atlas_bytes": args.public.stat().st_size, "weave_sha256": sha256(weave_path),
               "world_count": len(worlds), "completed_worlds": len(completed), "sample_count": sum(len(world["samples"]) for world in worlds),
               "provenance": atlas["provenance"], "weave": atlas["weave"]}
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
