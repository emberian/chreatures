#!/usr/bin/env python3
"""Fit a descriptive developmental GAM atlas from completed rich online telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

GAMFIT_VERSION = "0.1.259"
GAMFIT_COMMIT = "7c7eca8ac4826de95c8e743a20294bee132a9bcc"
TELEMETRY_FORMAT = "chreatures-rich-development-transition-telemetry-v1"
FEATURES = (
    "energy", "gut", "fatigue", "speed", "angular_velocity", "support",
    "neural_mean", "neural_std", "neural_p90",
    "history_mean", "history_std", "history_rms",
    "previous_thrust", "previous_yaw", "previous_grip", "previous_oral",
    "thrust", "yaw", "grip", "posture", "oral", "motor_magnitude",
    "goal_distance", "goal_age", "peripheral_proximity", "foveal_proximity",
    "visual_contrast", "experience_tick", "motor_x_fatigue",
)
TARGETS = {
    "goal_progress": (
        "goal_progress_code_distance_per_tick",
        "response ~ s(goal_distance,k=9)+s(goal_age,k=8)+s(history_rms,k=8)"
        "+s(neural_mean,k=8)+s(peripheral_proximity,k=8)+s(foveal_proximity,k=8)"
        "+s(fatigue,k=8)+s(speed,k=8)+s(thrust,k=8)+s(yaw,k=8)"
        "+s(experience_tick,k=8)+ti(goal_distance,goal_age,k=16)",
    ),
    "effort": (
        "world_reported_effort_per_tick",
        "response ~ s(energy,k=8)+s(fatigue,k=8)+s(speed,k=8)+s(support,k=8)"
        "+s(neural_mean,k=8)+s(history_rms,k=8)+s(thrust,k=8)+s(yaw,k=8)"
        "+s(grip,k=8)+s(posture,k=8)+s(oral,k=8)+s(motor_magnitude,k=8)"
        "+s(motor_x_fatigue,k=8)+s(experience_tick,k=8)",
    ),
    "body_law_residual": (
        "mean_absolute_standardized_deployed_body_law_residual",
        "response ~ s(experience_tick,k=10)+s(goal_distance,k=8)+s(goal_age,k=8)"
        "+s(history_rms,k=8)+s(neural_mean,k=8)+s(neural_std,k=8)"
        "+s(peripheral_proximity,k=8)+s(foveal_proximity,k=8)"
        "+s(visual_contrast,k=8)+s(energy,k=8)+s(fatigue,k=8)"
        "+s(motor_magnitude,k=8)+ti(experience_tick,goal_distance,k=16)",
    ),
}
SURFACES = {
    "goal_progress": (("goal_distance", "goal_age"), ("goal_distance", "history_rms")),
    "effort": (("motor_magnitude", "fatigue"), ("thrust", "speed")),
    "body_law_residual": (("experience_tick", "goal_distance"),
                          ("experience_tick", "history_rms")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deployed_predictions(bank: dict, raw: np.ndarray) -> np.ndarray:
    result = np.empty((len(raw), len(bank["laws"])), dtype=np.float64)
    normalized = np.column_stack([
        (raw[:, i] - item["mean"]) / item["scale"]
        for i, item in enumerate(bank["features"])
    ])
    for law_index, law in enumerate(bank["laws"]):
        value = np.full(len(raw), law["intercept"], dtype=np.float64)
        for term in law["terms"]:
            value += np.interp(normalized[:, term["feature"]], term["knots"], term["values"])
        result[:, law_index] = value
    return result


def prepare(run: Path, bank_path: Path, compact_path: Path, *,
            allow_interrupted: bool = False, source_revision: str | None = None) -> dict:
    identity_path, result_path = run / "identity.json", run / "result.json"
    if not identity_path.exists():
        raise RuntimeError("rich run lacks identity.json")
    identity = json.loads(identity_path.read_text())
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    if identity.get("telemetry", {}).get("format") != TELEMETRY_FORMAT:
        raise RuntimeError("rich telemetry format differs")
    expected_steps = int(identity.get("arguments", {}).get("steps", -1))
    updates_path = run / "updates.jsonl"
    if result is None:
        if not allow_interrupted or not source_revision or not updates_path.exists():
            raise RuntimeError("interrupted run requires explicit allowance, source revision, and updates.jsonl")
        updates = [json.loads(line) for line in updates_path.read_text().splitlines() if line.strip()]
        if not updates:
            raise RuntimeError("interrupted rich run has no completed update receipt")
        last_update = updates[-1]
        observed_steps = int(last_update["physical_steps"])
    else:
        if expected_steps < 1 or int(result.get("physical_steps", -2)) != expected_steps:
            raise RuntimeError("rich result does not receipt the configured completed physical steps")
        last_update, observed_steps = None, expected_steps
    if identity["telemetry"].get("outcome_order") != ["nutrition", "contact", "distance",
            "effort", "mechanical_work", "ingested_mass", "mouth_material_contacts",
            "homeostatic_reward"]:
        raise RuntimeError("rich outcome order differs")
    packets = sorted(run.glob("telemetry-*.npz"))
    if not packets:
        raise RuntimeError("completed rich run contains no telemetry packets")
    bank = json.loads(bank_path.read_text())
    rows, targets, residual_valid_rows, episode_rows, world_rows, source_hashes = [], [], [], [], [], []
    for packet in packets:
        data = np.load(packet)
        required = {"episode", "tick", "world_slot", "resident_slot", "physiology",
                    "neural", "worker_hidden", "previous", "goal_distance_t",
                    "goal_attempt_age", "rich_summary", "executed_action", "oral",
                    "next_physiology", "outcomes", "goal_progress"}
        if required - set(data.files):
            raise RuntimeError(f"{packet.name} lacks {sorted(required - set(data.files))}")
        shape = data["episode"].shape
        flatten = lambda name: data[name].reshape((-1,) + data[name].shape[len(shape):])
        phys, neural, hidden = flatten("physiology"), flatten("neural"), flatten("worker_hidden")
        previous, action, oral = flatten("previous"), flatten("executed_action"), flatten("oral")
        rich, next_phys, outcomes = flatten("rich_summary"), flatten("next_physiology"), flatten("outcomes")
        goal_distance = flatten("goal_distance_t").reshape(-1)
        goal_age = flatten("goal_attempt_age").reshape(-1)
        tick = flatten("tick").reshape(-1)
        motor = np.abs(action[:, :4]).mean(axis=1)
        x = np.column_stack((
            phys, neural.mean(1), neural.std(1), np.quantile(neural, .9, axis=1),
            hidden.mean(1), hidden.std(1), np.sqrt(np.mean(hidden * hidden, axis=1)),
            previous[:, 0], previous[:, 1], previous[:, 3], previous[:, 8],
            action[:, 0], action[:, 1], action[:, 3], action[:, 7], oral, motor,
            goal_distance, np.log1p(goal_age), rich[:, 6], rich[:, 14],
            np.mean(rich[:, [3, 4, 5, 11, 12, 13]], axis=1), np.log1p(tick),
            motor * phys[:, 2],
        )).astype(np.float32)
        current_bank = np.column_stack((phys[:, 0], phys[:, 2], phys[:, 3], phys[:, 5],
            neural.mean(1), action[:, 0], action[:, 1], action[:, 3], oral, motor,
            action[:, 0] * phys[:, 2], action[:, 1] * phys[:, 3]))
        predicted = deployed_predictions(bank, current_bank)
        actual = np.column_stack((next_phys[:, 3] - phys[:, 3],
                                  phys[:, 0] - next_phys[:, 0],
                                  phys[:, 2] - next_phys[:, 2]))
        scales = np.asarray([law["target_scale"] for law in bank["laws"]])
        residual = np.mean(np.abs(actual - predicted) / scales, axis=1)
        in_domain = np.all(np.column_stack([
            (current_bank[:, i] >= feature["minimum"]) & (current_bank[:, i] <= feature["maximum"])
            for i, feature in enumerate(bank["features"])
        ]), axis=1)
        target = np.column_stack((flatten("goal_progress").reshape(-1), outcomes[:, 3], residual))
        rows.append(x); targets.append(target.astype(np.float32)); residual_valid_rows.append(in_domain)
        episode_rows.append(flatten("episode").reshape(-1).astype(np.int32))
        world_rows.append(flatten("world_slot").reshape(-1).astype(np.int16))
        source_hashes.append(sha256(packet))
    x, y = np.concatenate(rows), np.concatenate(targets)
    residual_valid = np.concatenate(residual_valid_rows)
    episode, world = np.concatenate(episode_rows), np.concatenate(world_rows)
    maximum_tick = int(max(np.load(packet)["tick"].max() for packet in packets))
    if maximum_tick + 1 != observed_steps:
        raise RuntimeError(f"telemetry ends at tick {maximum_tick}, receipt declares {observed_steps} steps")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("non-finite rich telemetry cannot enter an atlas")
    observation = {"kind": "completed-run" if result is not None else "bounded-interrupted-run",
                   "configured_steps": expected_steps, "observed_steps": observed_steps,
                   "maximum_tick": maximum_tick, "source_revision": source_revision,
                   "identity_sha256": sha256(identity_path),
                   "updates_sha256": sha256(updates_path) if updates_path.exists() else None,
                   "result_sha256": sha256(result_path) if result_path.exists() else None,
                   "last_completed_update": last_update,
                   "graph_sha256": identity.get("graph_sha256"),
                   "rich_profile_sha256": identity.get("rich_profile_sha256"),
                   "port_bundle_sha256": identity.get("port_bundle_sha256"),
                   "bootstrap_sha256": identity.get("bootstrap_sha256")}
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(compact_path, features=x, targets=y, episode=episode,
                        world_slot=world, residual_valid=residual_valid,
                        source_sha256=np.asarray(source_hashes),
                        observation_receipt=np.asarray(json.dumps(observation, sort_keys=True)))
    return {"rows": len(x), "episodes": int(np.unique(episode).size),
            "worlds_per_episode": int(np.unique(world).size), "packets": len(packets),
            "observation_receipt": observation,
            "deployed_bank_sha256": sha256(bank_path), "compact_sha256": sha256(compact_path),
            "body_law_in_domain_rows": int(residual_valid.sum())}


def _rows(x: np.ndarray, indices: np.ndarray, names: list[str], response: np.ndarray | None = None):
    result = [{name: float(x[row, col]) for col, name in enumerate(names)} for row in indices]
    if response is not None:
        for item, value in zip(result, response[indices], strict=True): item["response"] = float(value)
    return result


def fit(compact_path: Path, output: Path) -> dict:
    import gamfit
    if gamfit.__version__ != GAMFIT_VERSION or not gamfit.build_info().get("available"):
        raise RuntimeError(f"native gamfit {GAMFIT_VERSION} is required")
    data = np.load(compact_path)
    x, y, episode, world = data["features"].astype(float), data["targets"].astype(float), data["episode"], data["world_slot"]
    residual_valid = data["residual_valid"].astype(bool)
    if np.unique(episode).size >= 4:
        heldout_episode = int(np.max(episode))
        heldout_world = int(np.max(world))
        test = episode == heldout_episode
        validation = (~test) & (world == heldout_world)
        train = ~(test | validation)
        split_label = (f"episodes before {heldout_episode}, worlds 0..{heldout_world - 1} train; "
                       f"world {heldout_world} validates; complete episode {heldout_episode} held out")
    else:
        heldout_world = int(np.max(world))
        test = world == heldout_world
        validation = np.zeros(len(world), dtype=bool)
        train = ~test
        split_label = f"single observed episode: worlds 0..{heldout_world - 1} train; complete world {heldout_world} held out"
    if not train.any() or not test.any():
        raise RuntimeError("need at least one complete training and held-out physical unit")
    mean, scale = x[train].mean(0), x[train].std(0); scale[scale < 1e-9] = 1
    z = (x - mean) / scale
    names, indices = list(FEATURES), {name: i for i, name in enumerate(FEATURES)}
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "chreatures-developmental-gam-atlas-v1", "status": "descriptive conditional prediction; no causal claim",
              "source": {"gamfit_version": GAMFIT_VERSION, "gam_source_commit": GAMFIT_COMMIT,
                         "telemetry_sha256": data["source_sha256"].tolist(),
                         "observation_receipt": json.loads(str(data["observation_receipt"]))},
              "models": {},
              "split": split_label,
              "rows": {"train": int(train.sum()), "validation_worlds": int(validation.sum()), "held_out": int(test.sum())},
              "units": {"episodes_total": int(np.unique(episode).size),
                        "episodes_held_out": int(np.unique(episode[test]).size) if np.unique(episode).size >= 4 else 0,
                        "training_worlds": int(np.unique(world[train]).size),
                        "held_out_worlds": int(np.unique(world[test]).size),
                        "validation_episode_worlds": int(np.unique(np.column_stack((episode[validation], world[validation])), axis=0).shape[0])},
              "feature_normalization": {name: {"mean": float(mean[i]), "scale": float(scale[i])} for i, name in enumerate(names)}}
    started = time.perf_counter()
    for target_index, (name, (unit, formula)) in enumerate(TARGETS.items()):
        eligible = residual_valid if name == "body_law_residual" else np.ones(len(x), dtype=bool)
        target_train, target_validation, target_test = train & eligible, validation & eligible, test & eligible
        train_rows = _rows(z, np.flatnonzero(target_train), names, y[:, target_index])
        try:
            gamfit.validate_formula(train_rows, formula)
            # All three recorded responses are continuous. Declare this explicitly so
            # nonnegative effort/residual columns are not guessed as count families.
            model = gamfit.fit(train_rows, formula, family="gaussian")
        except Exception as exc:
            report["models"][name] = {"status": "native fit failed; no model minted",
                                      "unit": unit, "formula": formula,
                                      "train_rows": int(target_train.sum()),
                                      "error": f"{type(exc).__name__}: {exc}"[:4000]}
            continue
        model_path = output / f"{name}.gam"; model.save(model_path)
        metrics = {}
        for split_name, mask in (("validation_worlds", target_validation), ("held_out", target_test)):
            if not mask.any():
                continue
            observed = y[mask, target_index]
            predicted = np.asarray(model.predict(_rows(z, np.flatnonzero(mask), names)), dtype=float)
            baseline = np.full(len(observed), y[target_train, target_index].mean())
            metrics[split_name] = {"rows": len(observed), "rmse": float(np.sqrt(np.mean((predicted-observed)**2))),
                                   "mean_baseline_rmse": float(np.sqrt(np.mean((baseline-observed)**2)))}
        baseline_row = {feature: 0.0 for feature in names}; surfaces = []
        for first, second in SURFACES[name]:
            gx = np.linspace(np.quantile(z[train, indices[first]], .01), np.quantile(z[train, indices[first]], .99), 25)
            gy = np.linspace(np.quantile(z[train, indices[second]], .01), np.quantile(z[train, indices[second]], .99), 25)
            grid = [dict(baseline_row, **{first: float(a), second: float(b)}) for b in gy for a in gx]
            surfaces.append({"axes": [first, second], "x_standardized": gx.tolist(), "y_standardized": gy.tolist(),
                             "prediction": np.asarray(model.predict(grid), dtype=float).reshape(25, 25).tolist()})
        report["models"][name] = {"status": "fitted native GAM", "unit": unit, "formula": formula, "model_file": model_path.name,
                                  "model_sha256": sha256(model_path), "metrics": metrics, "surfaces": surfaces}
    report["fit_seconds"] = time.perf_counter() - started
    (output / "developmental_atlas.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--deployed-bank", type=Path, default=Path(__file__).with_name("artifacts") / "body_consequence_laws.json")
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-interrupted", action="store_true")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if args.run:
        print(json.dumps(prepare(args.run, args.deployed_bank, args.compact,
            allow_interrupted=args.allow_interrupted,
            source_revision=args.source_revision), indent=2, sort_keys=True))
    if args.output: print(json.dumps(fit(args.compact, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
