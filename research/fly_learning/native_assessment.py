"""Matched native physical diagnosis; observer metrics never enter the CNS."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .batch_collection import _write_receipt
from .data import sha256_file
from .native_host import NativeActualFlyWorld, TorchFullCNS
from .recovery import _variant_scene, build_recovery_plan

FORMAT = "chreatures-native-fly-zero-context-assessment-v2"
ARM_FORMAT = "chreatures-cns-v5-physical-assessment-arms-v1"
TICKS = 512
STATE_FIELDS = (
    "rates", "adaptation", "support", "release", "mod_da", "mod_oa", "mod_ht",
    "efficacy_deviation", "eligibility",
)


def _load_arms(path):
    manifest = json.loads(path.read_text())
    if manifest.get("format") != ARM_FORMAT:
        raise ValueError("assessment requires the current V5 arm manifest")
    arms = manifest.get("arms", [])
    if len(arms) < 2:
        raise ValueError("matched assessment requires at least two arms")
    names = set()
    for arm in arms:
        name = arm.get("name", "")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name) or name in names:
            raise ValueError("assessment arm names must be unique safe names")
        names.add(name)
        for key in ("service_sha256", "adapter_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", arm.get(key, "")):
                raise ValueError("assessment arm lacks exact service identity")
        service = (path.parent / arm["service"]).resolve()
        if sha256_file(service) != arm["service_sha256"]:
            raise ValueError("assessment service bytes differ before world allocation")
        arm["service"] = str(service)
    return arms


def _save_cns_state(path, state):
    if tuple(state.__dataclass_fields__) != STATE_FIELDS:
        raise ValueError("assessment requires all nine V5 private state fields")
    arrays = {name: getattr(state, name).detach().cpu().numpy() for name in STATE_FIELDS}
    for name, value in arrays.items():
        expected = (4184 if name in STATE_FIELDS[-2:] else 165122, 4)
        if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid private state {name}")
    d, e = arrays["efficacy_deviation"], arrays["eligibility"]
    if np.any(d < -.8) or np.any(d > 0) or np.any(e < 0) or np.any(e > 1):
        raise ValueError("private plasticity state outside V5 contract")
    return _save_trace(path, arrays)


def _stimulus(world, tick):
    event = tick // 64
    rgb = np.asarray([((event + channel * 3) % 5) / 4 for channel in range(12)], dtype="<f4")
    import hashlib
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    sound = {"position_mm": [0, 0, 2], "frequency_hz": [100, 630, 250, 1000][event % 4],
             "envelope": .65, "duration_s": .64}
    receipt = world._rpc("stimulus", sound=sound, screen={
        "width": 2, "height": 2, "rgb_f32_base64": base64.b64encode(rgb.tobytes()).decode(),
        "sha256": digest,
    })
    if receipt["screen_sha256"] != digest:
        raise RuntimeError("native stimulus acknowledgement differs")
    return {"tick": tick, "screen_sha256": digest, "sound": sound}


def _save_trace(path, values, *, require_finite=True):
    arrays = {name: np.asarray(rows) for name, rows in values.items()}
    if require_finite and any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("nonfinite physical assessment trace")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    return {"file": path.name, "sha256": sha256_file(path),
            "shapes": {name: list(value.shape) for name, value in arrays.items()}}


def _summarize(trace):
    position = np.asarray(trace["root_position"])
    upright = np.asarray(trace["upright"])
    feet = np.asarray(trace["feet_contact"])
    motor = np.asarray(trace["motor"])
    work = np.asarray(trace["mechanical_work"])
    rows = []
    for resident in range(4):
        support = (upright[:, resident] >= .65) & (feet[:, resident].sum(-1) >= 3)
        established = np.flatnonzero(support)
        losses = np.flatnonzero(~support & (np.arange(len(support)) > (established[0] if len(established) else len(support))))
        fallen = np.flatnonzero(upright[:, resident] < -.15)
        delta = position[-1, resident] - position[0, resident]
        rows.append({
            "resident": resident, "initially_supported": bool(support[0]),
            "first_support_tick": None if not len(established) else int(established[0]),
            "first_support_loss_after_establishment_tick": None if not len(losses) else int(losses[0]),
            "first_fall_tick": None if not len(fallen) else int(fallen[0]),
            "upright_duration_s": float((upright[1:, resident] > .5).sum() * .01),
            "support_duration_s": float(support[1:].sum() * .01),
            "net_displacement_mm": delta.tolist(), "net_displacement_norm_mm": float(np.linalg.norm(delta)),
            "root_path_mm": float(np.linalg.norm(np.diff(position[:, resident], axis=0), axis=-1).sum()),
            "mechanical_work_model_units": float(work[:, resident].sum()),
            "mean_abs_motor92": float(np.abs(motor[:, resident]).mean()),
            "servo_min": float(motor[:, resident, :84].min()), "servo_max": float(motor[:, resident, :84].max()),
            "activation_min": float(motor[:, resident, 84:].min()), "activation_max": float(motor[:, resident, 84:].max()),
        })
    return rows


def _sealed_prior(directory, identity, arms):
    """Authenticate completed conditions; never recover an unsealed life."""
    results = []
    for spec in arms:
        arm, service_sha = spec["name"], spec["service_sha256"]
        for index in (10, 11):
            path = directory / f"world{index:02d}-{arm}" / "receipt.json"
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            for key in ("format", "ticks", "control_dt_s", "context", "seed", "native_deployment_sha256", "layouts_manifest_sha256"):
                if row.get(key) != identity[key]:
                    raise ValueError(f"prior sealed assessment changed {key}")
            if (not row.get("completed") or row.get("arm") != arm or row.get("world_index") != index
                    or row.get("service_sha256") != service_sha
                    or row.get("adapter_sha256") != spec["adapter_sha256"]):
                raise ValueError("prior assessment condition differs")
            trace_name = row["trace"]["file"]
            if Path(trace_name).name != trace_name or sha256_file(path.parent / trace_name) != row["trace"]["sha256"]:
                raise ValueError("prior assessment trace bytes differ")
            if sha256_file(path.parent / "initial-world.bin") != row["initial_snapshot_sha256"]:
                raise ValueError("prior saved wholeworld bytes differ")
            if (row["trace"]["shapes"].get("motor") != [TICKS, 4, 92]
                    or row["trace"]["shapes"].get("root_position") != [TICKS + 1, 4, 3]):
                raise ValueError("prior assessment trace is incomplete")
            for key in ("initial_cns_state", "final_cns_state"):
                item = row[key]
                if Path(item["file"]).name != item["file"] or sha256_file(path.parent / item["file"]) != item["sha256"]:
                    raise ValueError("prior private CNS state differs")
            results.append({"receipt": str(path.resolve()), "sha256": sha256_file(path), **row})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scenes", "native-binary", "native-manifest", "arms", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sealed-prior", type=Path,
                   help="Authenticate completed conditions from a failed earlier run; fresh lives for missing conditions only")
    args = p.parse_args()
    arms = _load_arms(args.arms)
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    identity = {"format": FORMAT, "source_revision": args.source_revision,
                "assessment_source_sha256": sha256_file(Path(__file__)),
                "transport_source_sha256": sha256_file(Path(__file__).with_name("native_host.py")),
                "native_deployment_sha256": sha256_file(args.native_manifest),
                "layouts_manifest_sha256": sha256_file(args.scenes / "nursery-layouts.json"),
                "arms_manifest_sha256": sha256_file(args.arms),
                "expected_services": {a["name"]: a["service_sha256"] for a in arms}, "ticks": TICKS, "control_dt_s": .01,
                "context": "exact zero12", "world_indices": [10, 11], "seed": args.seed,
                "model_process_pid": os.getpid(),
                "initialization": "fresh physical world and nine private CNS state fields for each arm/layout"}
    _write_receipt(args.output / "launch-receipt.json", identity)
    starts = {}; results = []; model = None; world = None
    phase = "model-load"; arm = None; index = None; completed_ticks = 0; trace = None
    try:
        if args.sealed_prior is not None:
            results = _sealed_prior(args.sealed_prior, identity, arms)
            for row in results:
                previous = starts.setdefault(row["world_index"], row["initial_snapshot_sha256"])
                if previous != row["initial_snapshot_sha256"]:
                    raise ValueError("prior arms have different wholeworld starts")
            _write_receipt(args.output / "sealed-prior-amendment.json", {
                "format": "chreatures-native-assessment-writer-amendment-v1",
                "reason": "Preserve authenticated completed conditions from the prior run; collect only missing conditions in new physical lives.",
                "source_revision": args.source_revision,
                "prior_directory": str(args.sealed_prior.resolve()),
                "authenticated_conditions": [{"arm": r["arm"], "world_index": r["world_index"],
                                               "receipt": r["receipt"], "sha256": r["sha256"]} for r in results],
                "unsealed_prior_worlds_reused": False,
            })
        for spec in arms:
            arm, expected_sha = spec["name"], spec["service_sha256"]
            missing_worlds = [index for index in (10, 11)
                              if not any(r["arm"] == arm and r["world_index"] == index for r in results)]
            if not missing_worlds:
                continue
            service = Path(spec["service"])
            if sha256_file(service) != expected_sha:
                raise ValueError("assessment service checksum differs")
            phase = "model-load"
            load_start = time.monotonic()
            model = TorchFullCNS(service, args.device)
            if (model.metadata["cns_service_sha256"] != expected_sha
                    or model.metadata.get("format") != "chreatures-cns-service-v5"
                    or model.metadata["adapter_sha256"] != spec["adapter_sha256"]):
                raise ValueError("service changed during model loading")
            load_seconds = time.monotonic() - load_start
            for index in missing_worlds:
                completed_ticks = 0; trace = None; phase = "world-start"
                directory = args.output / f"world{index:02d}-{arm}"
                directory.mkdir()
                log_path = directory / "native-stderr.log"
                with log_path.open("xb") as log:
                    local = SimpleNamespace(**vars(args), scene=_variant_scene(args.scenes, index),
                                            native_stderr=log, native_working_directory=directory.resolve())
                    world = NativeActualFlyWorld(local, build_recovery_plan(index, base_seed=args.seed))
                    snapshot = base64.b64decode(world._rpc("snapshot")["snapshot_base64"], validate=True)
                    snapshot_path = directory / "initial-world.bin"
                    snapshot_path.write_bytes(snapshot)
                    snapshot_sha = sha256_file(snapshot_path)
                    if snapshot_sha != world.initial_snapshot_sha256():
                        raise RuntimeError("saved wholeworld differs from native READY")
                    if index in starts and starts[index] != snapshot_sha:
                        raise RuntimeError("assessment arms do not share identical full physical start")
                    starts[index] = snapshot_sha
                    if model.state is not None:
                        raise RuntimeError("private CNS state leaked across physical lives")
                    model.state = model.model.initial_state(4)
                    initial_cns = _save_cns_state(directory / "initial-cns.npz", model.state)
                    initial = world.sample()
                    trace = {"root_position": [initial.observer["thorax_position"].copy()],
                             "upright": [initial.observer["thorax_rotation"][:, 2, 2].copy()],
                             "feet_contact": [initial.ground_contact_raw[:, :, 0] > 0],
                             "body": [initial.body_afferents.copy()], "joint_position": [initial.joint_position.copy()],
                             "motor": [], "latent": [], "mechanical_work": [],
                             "efficacy_deviation_mean": [], "eligibility_mean": [],
                             "efficacy_deviation_min": [], "eligibility_max": []}
                    cues = []; run_start = time.monotonic()
                    sample = initial
                    phase = "physical-loop"
                    for tick in range(TICKS):
                        if tick % 64 == 0:
                            cues.append(_stimulus(world, tick))
                            sample = world.sample()
                        if log_path.stat().st_size or (directory / "MUJOCO_LOG.TXT").exists():
                            raise RuntimeError("native stderr output: warning/error is fatal")
                        latent, motor = model.step(sample.optic_rgb, sample.body_afferents, np.zeros((4, 12), np.float32))
                        if (not np.isfinite(latent).all() or not np.isfinite(motor).all() or np.any(np.abs(motor[:, :84]) > 1)
                                or np.any(motor[:, 84:] < 0) or np.any(motor[:, 84:] > 1)):
                            raise RuntimeError("invalid full-CNS motor output")
                        world.advance(motor, .01)
                        after = world.sample()
                        if log_path.stat().st_size or (directory / "MUJOCO_LOG.TXT").exists():
                            raise RuntimeError("native stderr output after mutation: no retry")
                        trace["root_position"].append(after.observer["thorax_position"].copy())
                        trace["upright"].append(after.observer["thorax_rotation"][:, 2, 2].copy())
                        trace["feet_contact"].append(after.ground_contact_raw[:, :, 0] > 0)
                        trace["body"].append(after.body_afferents.copy())
                        trace["joint_position"].append(after.joint_position.copy())
                        trace["motor"].append(motor.copy()); trace["latent"].append(latent.copy())
                        trace["mechanical_work"].append([row["work"] for row in after.observer["actuator_state"]])
                        for state_name in STATE_FIELDS[-2:]:
                            values = getattr(model.state, state_name)
                            trace[state_name + "_mean"].append(values.mean(0).cpu().numpy().copy())
                            operation = "min" if state_name == "efficacy_deviation" else "max"
                            trace[state_name + "_" + operation].append(getattr(values, operation)(0).values.cpu().numpy().copy())
                        completed_ticks = tick + 1
                        sample = after
                        if completed_ticks % 64 == 0:
                            print(json.dumps({"event": "assessment-progress", "arm": arm, "world": index,
                                              "ticks": completed_ticks, "elapsed_seconds": time.monotonic() - run_start}), flush=True)
                    phase = "world-seal"
                    result = {**identity, "completed": True, "arm": arm, "world_index": index,
                              "service_sha256": expected_sha, "adapter_sha256": model.metadata["adapter_sha256"],
                              **world.deployment_identity, "initial_snapshot_sha256": snapshot_sha,
                              "scene_sha256": world.ready["fixture_sha256"], "world_seed": build_recovery_plan(index, base_seed=args.seed).world_seed,
                              "model_load_seconds": load_seconds,
                              "service_training_status": model.metadata["training_status"],
                              "initial_cns_state": initial_cns,
                              "final_cns_state": _save_cns_state(directory / "final-cns.npz", model.state), "physical_loop_seconds": time.monotonic() - run_start,
                              "trace": _save_trace(directory / "trace.npz", trace), "stimulus_receipts": cues,
                              "residents": _summarize(trace), "native_stderr_sha256": sha256_file(log_path),
                              "native_warning_observation": "no stderr bytes, MUJOCO_LOG.TXT or invalid stdout protocol; numerical warning counters unavailable in frozen native binary",
                              "claim_limit": "Matched native physical zero-context comparison; no private-context control or general competence claim."}
                    world.close(); world = None
                    model.close()
                    _write_receipt(directory / "receipt.json", result)
                    results.append({"receipt": str(directory / "receipt.json"), "sha256": sha256_file(directory / "receipt.json"), **result})
            model.model = None; model = None
        result = {**identity, "completed": True, "conditions": results, "elapsed_seconds": time.monotonic() - began}
        _write_receipt(args.output / "assessment.json", result)
        print(json.dumps({"event": "assessment-sealed", "path": str(args.output / "assessment.json"),
                          "sha256": sha256_file(args.output / "assessment.json"), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    except BaseException as error:
        partial = None
        if trace is not None:
            partial = _save_trace(args.output / "aborted-trace.npz", trace, require_finite=False)
        _write_receipt(args.output / "abort.json", {**identity, "completed": False, "phase": phase,
                       "arm": arm, "world_index": index, "completed_ticks": completed_ticks,
                       "error_type": type(error).__name__, "error": str(error), "partial_trace": partial,
                       "sealed_conditions": [{"receipt": r["receipt"], "sha256": r["sha256"]} for r in results],
                       "elapsed_seconds": time.monotonic() - began, "mutation_retry_performed": False})
        raise
    finally:
        if world is not None:
            world.close()
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()
