#!/usr/bin/env python3
"""Offline full MaleCNS V3 physiology experiment; never a resident tick engine."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import struct
import time
from pathlib import Path

import numpy as np

FORMAT = "chreatures-anatomical-cns-gam-dynamics-v1"
FEATURES = ("release_tau", "release_use", "mod_gain", "mod_tau")
JOINT = "response ~ te(release_tau,release_use,mod_gain,mod_tau,k=15)"
ADDITIVE = "response ~ " + "+".join(f"s({name},k=3)" for name in FEATURES)
HORIZON = 4
TRIM = 8
RIDGE = .01


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Immutable outputs: a new experiment directory is required to change inputs.
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def design():
    """Face-centered four-factor design: 16 corners, 8 axial, one center."""
    points = list(itertools.product((-1., 1.), repeat=4))
    for axis in range(4):
        for value in (-1., 1.):
            point = [0.] * 4
            point[axis] = value
            points.append(tuple(point))
    points.append((0.,) * 4)
    return [{"setting_id": f"setting-{i:02d}", **dict(zip(FEATURES, x))}
            for i, x in enumerate(points)]


def setting_parameters(point):
    for key in FEATURES:
        if not math.isfinite(float(point[key])) or abs(float(point[key])) > 1:
            raise ValueError(f"{key} is outside the frozen design box")
    return {"release_tau_logit_shift": math.log(2) * point["release_tau"],
            "release_use_logit_shift": math.log(2) * point["release_use"],
            "mod_gain_raw_scale": 1 + .5 * point["mod_gain"],
            "modulation_tau_logit_shift": math.log(2) * point["mod_tau"]}


def load_service(path):
    from chreatures.cns_adapter_contract import (
        ARRAY_SPECS, MAGIC, metadata_for, service_identity, validate_arrays,
    )
    path = Path(path).resolve()
    with path.open("rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError("requires a current CHCNS3 artifact")
        n = struct.unpack("<I", f.read(4))[0]
        if n > 4 << 20:
            raise ValueError("oversized artifact metadata")
        metadata = json.loads(f.read(n))
    offset, arrays = 12 + n, {}
    for name, dtype, shape in ARRAY_SPECS:
        arrays[name] = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=shape)
        offset += arrays[name].nbytes
    if path.stat().st_size != offset:
        raise ValueError("artifact byte count differs")
    validate_arrays(arrays)
    checked = metadata_for(arrays, **{key: metadata[key] for key in (
        "graph_sha256", "atlas_sha256", "anatomy_sha256", "training_status")},
        provenance=metadata.get("provenance"))
    if checked != metadata:
        raise ValueError("artifact tensor or metadata identity differs")
    return arrays, service_identity(metadata, sha(path)), metadata["training_status"]


def apply_setting(model, base, point):
    import torch
    p = setting_parameters(point)
    with torch.no_grad():
        for attr, key in (("dynamics_release_tau_raw", "release_tau_logit_shift"),
                          ("dynamics_release_use_raw", "release_use_logit_shift"),
                          ("dynamics_modulation_tau_raw", "modulation_tau_logit_shift")):
            getattr(model, attr).copy_(base[attr] + p[key])
        model.dynamics_mod_gain_raw.copy_(base["dynamics_mod_gain_raw"] * p["mod_gain_raw_scale"])


def sensory_targets(episodes):
    # Eight contiguous atlas-site groups, RGB mean per group. These are offline
    # targets, not a claim of geometrically uniform angular retinal sectors.
    target = []
    for ep in episodes:
        rgb = ep.optic_rgb.reshape(513, 3, 1771, 3)
        optic = np.concatenate([x.mean(2) for x in np.array_split(rgb, 8, axis=2)], -1)
        target.append(np.concatenate((optic, ep.body), -1))
    return np.stack(target)


def temporal_probe(z, target, fit_worlds, eval_worlds):
    """Fixed ridge probe: Z_t alone predicts normalized sensory delta at t+4."""
    stop = target.shape[1] - HORIZON
    current = target[:, TRIM:stop]
    future = target[:, TRIM + HORIZON:]
    scale = np.maximum(current[fit_worlds].reshape(-1, 134).std(0), .01)
    delta = ((future - current) / scale).astype(np.float64)
    features = z[:, TRIM:stop].astype(np.float64)
    xfit = features[fit_worlds].reshape(-1, 512)
    yfit = delta[fit_worlds].reshape(-1, 134)
    mean = xfit.mean(0)
    sd = np.maximum(xfit.std(0), 1e-6)
    xf = (xfit - mean) / sd
    ym = yfit.mean(0)
    matrix = xf.T @ xf / len(xf) + RIDGE * np.eye(512)
    weight = np.linalg.solve(matrix, xf.T @ (yfit - ym) / len(xf))
    predictions = ((features[eval_worlds] - mean) / sd) @ weight + ym
    truth = delta[eval_worlds]
    errors = (predictions - truth) ** 2
    result = {}
    for name, sl in (("optic", slice(0, 24)), ("body", slice(24, 134))):
        persistence = float(np.mean(truth[..., sl] ** 2))
        mse = float(np.mean(errors[..., sl]))
        baseline = float(np.mean((truth[..., sl] - ym[sl]) ** 2))
        result[name] = {"mse": mse, "persistence_mse": persistence,
                        "train_mean_delta_mse": baseline,
                        "skill": 1 - mse / max(persistence, 1e-12),
                        "active_features": int((yfit[:, sl].std(0) > 1e-5).sum())}
    result["balanced_persistence_ratio"] = float(np.mean([
        result[k]["mse"] / max(result[k]["persistence_mse"], 1e-12)
        for k in ("optic", "body")]))
    result["scope"] = "linear Z512-to-future sensory delta; no actions or raw current senses as probe inputs"
    return result


def evaluate(z, motors, episodes):
    target = np.stack([ep.delivered_motor for ep in episodes])
    sensory = sensory_targets(episodes)
    scale = np.ones(34); scale[24:26] = 2
    result = {}
    # The first two validation worlds are training worlds, never the outer test.
    for name, fit, test in (("inner", list(range(4)), [4, 5]),
                            ("heldout", list(range(6)), [6, 7])):
        prediction = motors[test, TRIM:]
        truth = target[test, TRIM:]
        residual = (prediction - truth) / scale
        mean_motor = target[fit, TRIM:].mean(axis=(0, 1, 2))
        mse = float(np.mean(residual ** 2))
        baseline = float(np.mean(((truth - mean_motor) / scale) ** 2))
        future = temporal_probe(z, sensory, fit, test)
        temporal_ratio = future["balanced_persistence_ratio"]
        result[name] = {
            "motor_mse": mse, "motor_train_mean_mse": baseline,
            "motor_skill": 1 - mse / max(baseline, 1e-12),
            "motor_mse_per_channel": np.mean(residual ** 2, axis=(0, 1, 2)).tolist(),
            "motor_mse_per_world": np.mean(residual ** 2, axis=(1, 2, 3)).tolist(),
            "predicted_antagonist_effort": float(np.mean(prediction[..., :24] ** 2)),
            "teacher_antagonist_effort": float(np.mean(truth[..., :24] ** 2)),
            "predicted_cocontraction": float(np.mean(prediction[..., :24:2] * prediction[..., 1:24:2])),
            "temporal": future,
            "objective": mse + .25 * temporal_ratio,
        }
    return result


def run(args):
    import torch
    from research.anatomical_cns.data import load_corpus
    from research.anatomical_cns.model import AnatomicalCNS
    if not 0 <= args.start < args.stop <= 25:
        raise ValueError("setting range must satisfy 0 <= start < stop <= 25")
    torch.set_num_threads(args.cpu_threads)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU is unavailable")
    if args.device != "cpu":
        torch.cuda.set_per_process_memory_fraction(.70)
    arrays, identity, training_status = load_service(args.service)
    if training_status != "trained":
        raise ValueError("parameter sensitivity requires a trained V3 artifact")
    train, test = load_corpus(args.corpus)
    episodes = train + test
    if [e.metadata["episode_index"] for e in episodes] != list(range(8)):
        raise ValueError("expected explicit episode index order 0..7")
    model = AnatomicalCNS(arrays, device=torch.device(args.device)).eval()
    base = {name: getattr(model, name).detach().clone() for name in (
        "dynamics_release_tau_raw", "dynamics_release_use_raw", "dynamics_mod_gain_raw",
        "dynamics_modulation_tau_raw")}
    source_files = (Path(__file__), Path("research/anatomical_cns/model.py"),
                    Path("research/anatomical_cns/data.py"), Path("chreatures/cns_adapter_contract.py"))
    source = {str(p): sha(p) for p in source_files}
    provenance = {"service": identity, "corpus_sha256": sha(args.corpus / "corpus.json"),
                  "episode_sha256": [e.sha256 for e in episodes], "source_sha256": source,
                  "torch": torch.__version__, "hip": torch.version.hip, "cpu_threads": args.cpu_threads,
                  "device": str(model.device), "world_batch": args.world_batch,
                  "horizon_ticks": HORIZON, "trim_ticks": TRIM, "ridge": RIDGE,
                  "input_contract": ["optic_rgb", "body110", "delivered_context12"],
                  "teacher_labels_are_targets_only": True}
    if args.confirmation:
        proposal = json.loads(args.confirmation.read_text())
        if proposal["provenance"] != provenance:
            raise ValueError("confirmation must use identical model, corpus, source and replay configuration")
        points = [proposal["setting"]]
    elif args.zero_edge:
        points = [{"setting_id": "zero-edge", **dict.fromkeys(FEATURES, 0.)}]
        with torch.no_grad():
            for name in ("fast_graph", "da_graph", "oa_graph", "ht_graph"):
                getattr(model, name).values().zero_()
    else:
        points = design()[args.start:args.stop]
    for point in points:
        path = args.output / (point["setting_id"] + ".json")
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["provenance"] != provenance or previous["setting"] != point:
                raise ValueError(f"cannot resume mismatching output {path}")
            continue
        started = time.monotonic()
        apply_setting(model, base, point)
        z = np.empty((8, 512, 3, 512), np.float32)
        motors = np.empty((8, 512, 3, 34), np.float32)
        with torch.inference_mode():
            for begin in range(0, 8, args.world_batch):
                group = episodes[begin:begin + args.world_batch]
                batch = len(group) * 3
                optic = torch.as_tensor(np.concatenate([e.optic_rgb[:512] for e in group], 1), device=model.device).reshape(512, batch, 1771, 3)
                body = torch.as_tensor(np.concatenate([e.body[:512] for e in group], 1), device=model.device)
                context = torch.as_tensor(np.concatenate([e.delivered_context for e in group], 1), device=model.device)
                state = model.initial_state(batch)
                for t in range(512):
                    latent, motor, state = model(optic[t], body[t], context[t], state, dt=.05)
                    z[begin:begin + len(group), t] = latent.cpu().numpy().reshape(len(group), 3, 512)
                    motors[begin:begin + len(group), t] = motor.cpu().numpy().reshape(len(group), 3, 34)
                if not all(torch.isfinite(x).all().item() for x in state.fields()):
                    raise ValueError("nonfinite neural state")
        if not np.isfinite(z).all() or not np.isfinite(motors).all():
            raise ValueError("nonfinite replay outputs")
        metrics = evaluate(z, motors, episodes)
        record = {"format": FORMAT, "completed": True, "setting": point,
                  "effective_intervention": setting_parameters(point), "provenance": provenance,
                  "zero_edge": bool(args.zero_edge), "metrics": metrics,
                  "elapsed_seconds": time.monotonic() - started,
                  "scope": "frozen fullgraph open-loop physical-sequence replay, no physical rollout or CNS fit",
                  "latent_per_channel_std_mean": float(z.std(axis=(0, 1, 2)).mean()),
                  "motor_per_channel_std_mean": float(motors.std(axis=(0, 1, 2)).mean())}
        write(path, record)
        print(json.dumps({"setting": point["setting_id"], "seconds": record["elapsed_seconds"],
                          "inner": metrics["inner"]["objective"], "heldout": metrics["heldout"]["objective"]}), flush=True)


def fit(args):
    from research.dynamics_v2.gam_fit import (
        _require_native_gamfit, _capture_native_stderr, _metrics,
        GAMFIT_VERSION, GAMFIT_SOURCE_COMMIT,
    )
    gam = _require_native_gamfit()
    records = [json.loads((args.results / (p["setting_id"] + ".json")).read_text()) for p in design()]
    provenance = records[0]["provenance"]
    for expected, row in zip(design(), records):
        if (row.get("format") != FORMAT or row.get("completed") is not True or
            row.get("setting") != expected or row.get("zero_edge") or row["provenance"] != provenance):
            raise ValueError("requires all 25 matched, completed, non-ablated settings")
    args.output.mkdir(parents=True, exist_ok=False)
    models, diagnostics, warnings = {}, {}, []
    def response(row, target):
        m = row["metrics"]["inner"]
        if target == "temporal_ratio":
            return m["temporal"]["balanced_persistence_ratio"]
        return m[target]
    def rows_for(subset, target):
        return [{**{k: r["setting"][k] for k in FEATURES}, "response": response(r, target)} for r in subset]
    for target in ("objective", "motor_mse", "temporal_ratio", "predicted_antagonist_effort"):
        observed = np.array([response(r, target) for r in records])
        if not np.isfinite(observed).all():
            raise ValueError("nonfinite observed target")
        if np.ptp(observed) <= 1e-12:
            diagnostics[target] = {"status": "constant-no-identifiable-response", "value": float(observed[0])}
            continue
        rows = rows_for(records, target)
        gam.validate_formula(rows, JOINT, family="gaussian")
        model, msgs = _capture_native_stderr(lambda: gam.fit(rows, JOINT, family="gaussian"))
        warnings.extend(msgs)
        model.save(args.output / (target + ".gam"))
        restored = gam.load(args.output / (target + ".gam"))
        if not np.allclose(model.predict(rows), restored.predict(rows), atol=1e-10, rtol=0):
            raise ValueError("native GAM reload prediction differs")
        models[target] = restored
        center = dict.fromkeys(FEATURES, 0.)
        center_prediction = float(np.asarray(restored.predict([center])).reshape(-1)[0])
        axis_slices = {}
        for name in FEATURES:
            points = [{**center, name: float(x)} for x in np.linspace(-1, 1, 9)]
            axis_slices[name] = [{"coordinate": point[name], "prediction": float(value)}
                for point, value in zip(points, np.asarray(restored.predict(points)).reshape(-1))]
        pair_interactions = {}
        for left, right in itertools.combinations(FEATURES, 2):
            interaction = []
            for a, b in itertools.product((-.75, 0., .75), repeat=2):
                points = [{**center, left: a, right: b}, {**center, left: a}, {**center, right: b}]
                values = np.asarray(restored.predict(points)).reshape(-1)
                interaction.append({left: a, right: b, "prediction": float(values[0]),
                    "nonadditive_effect": float(values[0] - values[1] - values[2] + center_prediction)})
            pair_interactions[left + ":" + right] = interaction
        diagnostics[target] = {"status": "native-gam-fitted",
            "training": _metrics(np.asarray(model.predict(rows)).reshape(-1), observed),
            "center_conditional_axis_slices": axis_slices,
            "center_conditional_pair_interactions": pair_interactions,
            "sensitivity_scope": "native GAM predictions conditional on other factors at zero; uncertainty governed by leave-setting-out evidence"}
        if target == "objective":
            for label, formula in (("joint", JOINT), ("additive", ADDITIVE)):
                prediction, baseline = [], []
                for i in range(len(records)):
                    subset = records[:i] + records[i+1:]
                    train_rows = rows_for(subset, target)
                    gam.validate_formula(train_rows, formula, family="gaussian")
                    fold, msgs = _capture_native_stderr(lambda: gam.fit(train_rows, formula, family="gaussian"))
                    warnings.extend(msgs)
                    prediction.append(float(np.asarray(fold.predict([rows[i]])).reshape(-1)[0]))
                    baseline.append(float(np.mean([response(r, target) for r in subset])))
                diagnostics[target][label + "_leave_setting_out"] = _metrics(np.array(prediction), observed)
                diagnostics[target][label + "_predictions"] = prediction
                diagnostics[target]["mean_baseline_leave_setting_out"] = _metrics(np.array(baseline), observed)
    proposal = None
    if "objective" in models:
        candidates = [dict(zip(FEATURES, x)) for x in itertools.product((-.75, -.25, .25, .75), repeat=4)]
        predictions = np.asarray(models["objective"].predict(candidates)).reshape(-1)
        point = candidates[int(np.argmin(predictions))]
        proposal = {"format": FORMAT, "provenance": provenance,
                    "setting": {"setting_id": "confirmation", **point},
                    "predicted_inner_objective": float(predictions.min()),
                    "selection": "minimum inner-world objective among 256 unmeasured interior points; heldout excluded",
                    "gam_model_sha256": sha(args.output / "objective.gam")}
        write(args.output / "confirmation-proposal.json", proposal)
    report = {"format": FORMAT, "status": "native-fit-complete-confirmation-pending" if proposal else "constant-no-confirmation-proposed",
              "provenance": provenance, "gamfit_version": GAMFIT_VERSION,
              "gamfit_source_commit": GAMFIT_SOURCE_COMMIT, "native_build": gam.build_info(),
              "formula": JOINT, "additive_formula": ADDITIVE, "diagnostics": diagnostics,
              "analysis_source_sha256": {str(Path(__file__)): sha(Path(__file__)),
                  "research/dynamics_v2/gam_fit.py": sha(Path("research/dynamics_v2/gam_fit.py"))},
              "records_sha256": {p.name: sha(p) for p in sorted(args.results.glob("setting-??.json"))},
              "observed_heldout": [{"setting": r["setting"], "metrics": r["metrics"]["heldout"]} for r in records],
              "native_messages": warnings, "confirmation": proposal,
              "claim_limit": "parameter sensitivity of one frozen trained artifact on one eight-world corpus; no causal physical competence claim"}
    write(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))


def confirm(args):
    proposal = json.loads((args.fit / "confirmation-proposal.json").read_text())
    record = json.loads((args.results / "confirmation.json").read_text())
    if record["provenance"] != proposal["provenance"] or record["setting"] != proposal["setting"] or record.get("zero_edge"):
        raise ValueError("confirmation replay differs from frozen proposal")
    actual = record["metrics"]["inner"]["objective"]
    report = json.loads((args.fit / "report.json").read_text())
    loo = report["diagnostics"]["objective"]["joint_leave_setting_out"]["rmse"]
    error = abs(actual - proposal["predicted_inner_objective"])
    zero = json.loads((args.results / "zero-edge.json").read_text())
    center = json.loads((args.results / "setting-24.json").read_text())
    if zero["provenance"] != record["provenance"] or zero.get("zero_edge") is not True:
        raise ValueError("requires a matched center zero-edge replay")
    write(args.fit / "confirmation-report.json", {"format": FORMAT, "status": "actual-fullgraph-confirmed",
          "center_metrics": center["metrics"], "zero_edge_metrics": zero["metrics"],
          "zero_edge_latent_std": zero["latent_per_channel_std_mean"],
          "zero_edge_motor_std": zero["motor_per_channel_std_mean"],
          "zero_edge_scope": "all fast and modulatory graph values zero; same nontrivial baseline, afferents and frozen motor decoder; temporal probe refit on ablated training Z",
          "predicted_inner_objective": proposal["predicted_inner_objective"], "observed_inner_objective": actual,
          "absolute_error": error, "within_loo_rmse": error <= loo,
          "joint_loo_rmse": loo, "heldout_metrics": record["metrics"]["heldout"],
          "proposal_sha256": sha(args.fit / "confirmation-proposal.json"),
          "replay_sha256": sha(args.results / "confirmation.json"),
          "claim_limit": "confirmation executed; accuracy flag may fail; no production artifact promotion"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("design")
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("run")
    q.add_argument("--service", type=Path, required=True)
    q.add_argument("--corpus", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--start", type=int, default=0)
    q.add_argument("--stop", type=int, default=25)
    q.add_argument("--world-batch", type=int, default=2, choices=range(1, 9))
    q.add_argument("--cpu-threads", type=int, default=4)
    q.add_argument("--device", default="cuda")
    group = q.add_mutually_exclusive_group()
    group.add_argument("--zero-edge", action="store_true")
    group.add_argument("--confirmation", type=Path)
    q = sub.add_parser("fit")
    q.add_argument("--results", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("confirm")
    q.add_argument("--results", type=Path, required=True)
    q.add_argument("--fit", type=Path, required=True)
    args = p.parse_args()
    if args.command == "design":
        write(args.output, {"format": FORMAT, "status": "planned-not-executed", "settings": design(),
                           "additional_runs": ["center-zero-edge", "GAM-selected-interior-confirmation"],
                           "total_fullgraph_runs": 27})
    else:
        {"run": run, "fit": fit, "confirm": confirm}[args.command](args)


if __name__ == "__main__":
    main()
