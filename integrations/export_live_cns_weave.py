#!/usr/bin/env python3
"""Bind the executed live-CNS research wave into the native Universal Weave.

The source experiments stay in paperbin.  This exporter verifies their local
bytes and receipt-to-artifact identities, emits a small generic import request,
then invokes the pinned Rust Weave adapter and publishes its portable result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WEAVE = ROOT / "integrations" / "weave"
DEFAULT_PAPERBIN = Path.home() / "paperbin" / "chreatures" / "integration"
DEFAULT_OUTPUT = ROOT / "integrations" / "artifacts" / "live-cns-wave-v2"
DEFAULT_PUBLIC = ROOT / "site" / "assets" / "live-cns-evidence.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def require_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"{label} is not a lowercase SHA-256")
    return value


def blob(path: Path, role: str, media_type: str, expected: str | None = None) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular artifact: {path}")
    digest = sha256(path)
    if expected is not None:
        require(digest == require_hash(expected, role), f"{role} bytes differ from their receipt")
    return {
        "role": role,
        "uri": f"urn:sha256:{digest}",
        "sha256": digest,
        "bytes": path.stat().st_size,
        "media_type": media_type,
        "verification": "verified_local_sha256",
    }


def byte_blob(data: bytes, role: str, media_type: str) -> dict[str, Any]:
    digest = hashlib.sha256(data).hexdigest()
    return {
        "role": role,
        "uri": f"urn:sha256:{digest}",
        "sha256": digest,
        "bytes": len(data),
        "media_type": media_type,
        "verification": "derived_public_projection_sha256",
    }


def difference_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("samples", "signedMean", "rms", "min", "max", "maxAbs", "changedSamples", "positiveSamples", "negativeSamples")
    }


def verify_release(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    release_path = directory / "release.json"
    release = read_json(release_path)
    require(release.get("format") == "chreatures-browser-release-v1", "browser release format differs")
    files = release.get("files")
    require(isinstance(files, dict) and files, "browser release has no files")
    total = 0
    for name, expected in sorted(files.items()):
        require(isinstance(name, str) and Path(name).name == name, "browser release contains an unsafe file name")
        require(isinstance(expected, dict), f"browser release entry {name} differs")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"browser release file {name} is absent")
        require(path.stat().st_size == expected.get("bytes"), f"browser release size differs for {name}")
        require(sha256(path) == require_hash(expected.get("sha256"), name), f"browser release hash differs for {name}")
        total += path.stat().st_size
    require(total == release.get("totalBytes"), "browser release total byte count differs")
    return release, blob(release_path, "browser_release_manifest", "application/json")


def node(
    node_id: str,
    stage: int,
    record_type: str,
    text: str,
    *,
    parents: dict[str, str] | None = None,
    blobs: list[dict[str, Any]] | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parent_roles = parents or {}
    return {
        "id": node_id,
        "time": {"domain": "live_cns_evidence_stage", "value": stage},
        "record_type": record_type,
        "text": text,
        "artifact_uri": blobs[0]["uri"] if blobs else None,
        "blob_refs": blobs or [],
        "parent_ids": list(parent_roles),
        "fields": {"parent_roles": parent_roles, **(fields or {})},
    }


def metric_triplet(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": result["label"],
        "training_status": result["trainingStatus"],
        "resident_artifact_sha256": require_hash(result["residentArtifactSha256"], "rollout resident"),
        "net_displacement_meters": result["physical"]["netDisplacementMeters"],
        "path_length_meters": result["physical"]["pathLengthMeters"],
        "physical_stop_ticks_at_1cm_per_second": result["physical"]["physicalStopTicksAt1cmPerSecond"],
        "action_distinct_rows_rounded_1e3": result["action"]["distinctRowsRounded1e3"],
        "action_mean_channel_std": result["action"]["meanChannelStd"],
        "execution": result["execution"],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    crossed = args.paperbin / "dynamics-v2-crossed-20260907"
    temporal = args.paperbin / "dynamics-v2-temporal-fit-20260907"
    browser = args.paperbin / "browser-v2-temporal-pack-20260907"
    teacher = args.paperbin / "browser-v2-teacher-skills-20260907"
    screen = args.paperbin / "screen-response-v2-20260907"

    sweep_path, fit_path, confirm_path = crossed / "receipt.json", crossed / "gam" / "fit_report.json", crossed / "gam-confirm.receipt.json"
    sweep, fit, confirmation = map(read_json, (sweep_path, fit_path, confirm_path))
    require(sweep.get("format") == "chreatures-v2-crossed-fullgraph-regimes-v1" and sweep.get("completed") is True, "crossed sweep is incomplete")
    require(len(sweep.get("records", [])) == 27 and sweep.get("neurons") == 165122 and sweep.get("edges") == 25563197, "crossed sweep extent differs")
    require(fit.get("format") == "chreatures-cns-dynamics-v2-gam-response-v1" and fit.get("status") == "complete", "GAM fit is incomplete")
    require(fit.get("source", {}).get("sha256") == sha256(sweep_path), "GAM fit source hash differs")
    require(confirmation.get("format") == sweep["format"] and confirmation.get("completed") is True and len(confirmation.get("records", [])) == 1, "GAM confirmation is incomplete")
    prediction = fit["selection"]["confirmatory_unmeasured_setting"]
    observed = confirmation["records"][0]
    for key in ("gain", "tau", "adaptation_gain"):
        require(abs(float(prediction[key]) - float(observed[key])) < 1e-12, f"GAM confirmation changed {key}")
    gam_blobs = [blob(fit_path, "gam_fit_report", "application/json")]
    for name in ("skill.gam", "skill_additive.gam", "recovery_memory.gam"):
        model = crossed / "gam" / name
        expected = next((v["model"]["sha256"] for v in fit["models"].values() if isinstance(v, dict) and v.get("model", {}).get("file") == name), None)
        if name == "skill_additive.gam":
            expected = fit["skill_additive_benchmark"]["model"]["sha256"]
        gam_blobs.append(blob(model, f"native_gam_{name.removesuffix('.gam')}", "application/octet-stream", expected))

    training_path, audit_path = temporal / "training.receipt.json", temporal / "canonical-audit.receipt.json"
    service_receipt_path, service_path = temporal / "cns-service-v2-temporal-trained.bin.receipt.json", temporal / "cns-service-v2-temporal-trained.bin"
    training, audit, service_receipt = map(read_json, (training_path, audit_path, service_receipt_path))
    require(training.get("format") == "chreatures-dynamics-v2-temporal-research-fit-v1" and training.get("status") == "research-fit-not-controller", "temporal CNS training receipt differs")
    require(training.get("updates") == 192 and training.get("neurons") == 165122 and training.get("edges") == 25563197, "temporal CNS training extent differs")
    require(audit.get("format") == "chreatures-canonical-v2-joined-training-audit-v1", "canonical audit format differs")
    require(audit.get("scope", "").endswith("not behavior"), "canonical audit scope is missing")
    service_sha = require_hash(service_receipt.get("file_sha256"), "CNS service")
    service_blob = blob(service_path, "trained_cns_service", "application/octet-stream", service_sha)
    require(service_receipt.get("metadata", {}).get("training_status") == "trained", "CNS service is not the trained export")
    require(service_receipt["metadata"].get("graph_sha256") == "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625", "CNS graph identity differs")

    base_release, base_release_blob = verify_release(browser)
    require(base_release.get("serviceArtifactSha256") == service_sha, "browser pack uses another CNS service")
    dawn_path, joined_failure_path, joined_path = browser / "dawn-probe-tight.receipt.json", browser / "joined-runtime-bound.failure.json", browser / "joined-runtime-bound.receipt.json"
    dawn, joined_failure, joined = map(read_json, (dawn_path, joined_failure_path, joined_path))
    require(dawn.get("format") == "chreatures-cns-webgpu-v2-dawn-probe-v1" and dawn.get("byteExactRestore") == {"latent": True, "snapshot": True}, "Dawn export probe differs")
    require(joined_failure.get("status") == "failed", "joined failure receipt no longer records failure")
    require(joined.get("format") == "chreatures-live-joined-headless-v1" and joined.get("wholeLifeReplayExact") is True, "joined browser-equivalent run did not replay")

    collection_failure_path, collection_path = teacher / "failure.json", teacher / "collection-complete.json"
    result_path, native_path = teacher / "result.json", teacher / "native-comparison.json"
    unmatched_rollout_path, rollout_path = teacher / "matched-policy-rollout.receipt.json", teacher / "matched-policy-rollout-v2.receipt.json"
    collection_failure, collection, result, native, unmatched_rollout, rollout = map(read_json, (collection_failure_path, collection_path, result_path, native_path, unmatched_rollout_path, rollout_path))
    require(collection_failure.get("format") == "chreatures-browser-teacher-collection-failure-v1" and collection_failure.get("failed_phase") == "heldout-world-reinitialization", "teacher collection failure receipt differs")
    require(collection.get("format") == "chreatures-browser-teacher-collection-combined-receipt-v1" and len(collection.get("episodes", [])) == 2, "teacher collection differs")
    require(collection.get("action_source") == "privileged-physical-teacher" and collection.get("raw_geometry_retained") is False, "teacher boundary differs")
    episode_blobs = []
    for index, episode in enumerate(collection["episodes"]):
        episode_blobs.append(blob(Path(episode["file"]), f"teacher_episode_{index}", "application/x-npz", episode["sha256"]))
    require(result.get("format") == "chreatures-cns-resident-training-result-v1" and result.get("complete") is True and len(result.get("history", [])) == 128, "resident training did not complete 128 updates")
    trained = result["publication"]["resident"]
    sequence = result["publication"]["sequence_control"]
    trained_blob = blob(teacher / "cns-resident-trained.npz", "trained_resident", "application/x-npz", trained["file_sha256"])
    sequence_blob = blob(teacher / "sequence-control-trained.npz", "trained_sequence_control", "application/x-npz", sequence["file_sha256"])
    require(native.get("format") == "chreatures-cns-resident-native-comparison-v1" and native.get("passed") is True, "native controller comparison failed")
    require(native.get("resident_file_sha256") == trained_blob["sha256"], "native comparison used another trained resident")
    require(unmatched_rollout.get("format") == "chreatures-matched-resident-rollout-v1" and unmatched_rollout.get("initialWorldAndCnsStateMatched") is False, "first unmatched rollout receipt differs")
    require(rollout.get("format") == "chreatures-matched-resident-rollout-v2" and rollout.get("initialWorldAndCnsStateMatched") is True, "matched rollout did not share initial state")
    require(rollout.get("teacherGeometryUsedByPolicy") is False and len(rollout.get("results", [])) == 2, "matched rollout boundary differs")
    by_label = {entry["label"]: entry for entry in rollout["results"]}
    require(set(by_label) == {"parent", "trained"}, "matched rollout arms differ")
    require(by_label["trained"]["residentArtifactSha256"] == trained["artifact_sha256"], "matched rollout used another trained artifact")
    trained_release, trained_release_blob = verify_release(teacher / "trained-browser-model")
    require(trained_release.get("residentArtifactSha256") == trained["artifact_sha256"], "trained browser release used another resident")

    screen_receipt_path = screen / "bad-apple-vs-blank.receipt.json"
    screen_trace_path = screen / "bad-apple-vs-blank.traces.json.gz"
    screen_receipt = read_json(screen_receipt_path)
    screen_receipt_blob = blob(screen_receipt_path, "original_screen_response_receipt", "application/json")
    screen_trace_blob = blob(screen_trace_path, "screen_response_compact_traces", "application/gzip")
    require(screen_receipt_blob["sha256"] == "d94c7a8b48a55ca0b3396234f65d246350214c47e4f8a1cfe39b2fe75643cd62", "screen response receipt identity differs")
    require(screen_trace_blob["sha256"] == "c1018f24e20c0044374f8437db6b66148d62232f1a76540acf48a2137a2a9942", "screen response trace identity differs")
    require(screen_receipt.get("format") == "chreatures-screen-response-v1", "screen response receipt format differs")
    require(screen_receipt.get("sourceRevision") == "10b4b4c0c46d99866af623824e633ac1b04a61b5", "screen response source revision differs")
    require(screen_receipt.get("model", {}).get("cnsFormat") == "chreatures-cns-webgpu-v2", "screen response CNS format differs")
    require(screen_receipt["model"].get("cnsServiceArtifactSha256") == service_sha, "screen response used another CNS service")
    require(screen_receipt["model"].get("residentArtifactSha256") == base_release["residentArtifactSha256"], "screen response used another initialized resident")
    require(screen_receipt["model"].get("cnsTrainingStatus") == "trained" and screen_receipt["model"].get("residentTrainingStatus") == "initialized-untrained", "screen response model status differs")
    execution = screen_receipt.get("execution", {})
    initial = screen_receipt.get("matchedInitialState", {})
    require(execution.get("ticks") == 600 and execution.get("batch") == 3 and execution.get("dt") == 0.05 and execution.get("modelSeconds") == 30, "screen response execution extent differs")
    require(initial.get("exactAcrossConditions") is True, "screen response conditions did not share exact initial state")
    for key in ("physicalSha256", "cnsSha256", "residentSha256"):
        require_hash(initial.get(key), f"screen response {key}")
    require(screen_receipt.get("threshold", {}).get("absoluteDifference") == 1e-6, "screen response difference threshold differs")
    retina = screen_receipt.get("retina", {})
    neural = screen_receipt.get("neural", {})
    action = screen_receipt.get("action", {})
    physical = screen_receipt.get("physical", {})
    require(retina.get("capturedResident") == 0 and neural.get("capturedResident") == 0, "screen response neural/retinal capture is not resident 0")
    require(retina.get("anatomicalSites") == 1771 and retina.get("supportedSites") == 1486 and retina.get("capturesPerCondition") == 600, "screen response retina extent differs")
    require(neural.get("capturesPerCondition") == 600 and neural.get("afferent", {}).get("neurons") == 15340 and neural.get("nonafferent", {}).get("neurons") == 149782, "screen response neural extent differs")
    require(neural["nonafferent"].get("responsiveNeurons") == 140200, "screen response nonafferent count differs")
    require(action.get("valuesPerCondition") == 21600 and len(action.get("filmMinusBlankByChannel", [])) == 12, "screen response action extent differs")
    require(len(physical.get("filmPathLengthMeters", [])) == 3 and len(physical.get("blankPathLengthMeters", [])) == 3, "screen response physical batch extent differs")

    try:
        screen_trace = json.loads(gzip.decompress(screen_trace_path.read_bytes()))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("screen response trace is not valid gzip JSON") from exc
    require(isinstance(screen_trace, dict) and screen_trace.get("format") == "chreatures-screen-response-trace-v1" and screen_trace.get("dt") == 0.05, "screen response trace format differs")
    for key in ("filmTimeSeconds", "afferentDeltaRms", "nonafferentDeltaRms", "afferentDeltaSignedMean", "nonafferentDeltaSignedMean", "retinalChangedSites", "actionDeltaRms", "bodyDeltaRms"):
        require(isinstance(screen_trace.get(key), list) and len(screen_trace[key]) == 600, f"screen response trace {key} extent differs")
    for key in ("filmRootPositions", "blankRootPositions"):
        values = screen_trace.get(key)
        require(isinstance(values, list) and len(values) == 600 and all(isinstance(frame, list) and len(frame) == 3 and all(isinstance(row, list) and len(row) == 3 for row in frame) for frame in values), f"screen response trace {key} extent differs")
    for key in ("filmActions", "blankActions"):
        values = screen_trace.get(key)
        require(isinstance(values, list) and len(values) == 600 and all(isinstance(frame, list) and len(frame) == 36 for frame in values), f"screen response trace {key} extent differs")

    public_screen_receipt = {
        "format": "chreatures-public-screen-response-v1",
        "resultNodeId": "physical-screen-response-v2",
        "originalReceiptSha256": screen_receipt_blob["sha256"],
        "trace": {"file": "live-cns-screen-response.traces.json.gz", "sha256": screen_trace_blob["sha256"], "bytes": screen_trace_blob["bytes"], "format": screen_trace["format"], "samples": 600},
        "engineIdentity": require_hash(screen_receipt["engineIdentity"], "screen response engine"),
        "sourceRevision": screen_receipt["sourceRevision"],
        "adapter": screen_receipt["adapter"],
        "model": screen_receipt["model"],
        "stimulus": {
            "sourceAsset": "bad-apple-source-30s.mp4",
            "sourceSha256": require_hash(screen_receipt["stimulus"]["sourceSha256"], "screen source video"),
            "sourceDurationSeconds": screen_receipt["stimulus"]["sourceDurationSeconds"],
            "decodedWidth": screen_receipt["stimulus"]["decodedWidth"],
            "decodedHeight": screen_receipt["stimulus"]["decodedHeight"],
            "decodedFrames": screen_receipt["stimulus"]["decodedFrames"],
            "decodedRgbSha256": require_hash(screen_receipt["stimulus"]["decodedRgbSha256"], "decoded screen stimulus"),
            "comparison": screen_receipt["stimulus"]["comparison"],
        },
        "execution": execution,
        "matchedInitialState": initial,
        "threshold": screen_receipt["threshold"],
        "retina": {
            "anatomicalSites": retina["anatomicalSites"], "supportedSites": retina["supportedSites"], "capturedResident": retina["capturedResident"], "capturesPerCondition": retina["capturesPerCondition"],
            "filmMinusBlank": difference_summary(retina["filmMinusBlank"]), "everChangedSites": retina["everChangedSites"], "meanChangedSitesPerTick": retina["meanChangedSitesPerTick"], "maxChangedSitesPerTick": retina["maxChangedSitesPerTick"], "unsupportedSitesEverNonzero": retina["unsupportedSitesEverNonzero"],
        },
        "neural": {
            "capturedResident": neural["capturedResident"], "capturesPerCondition": neural["capturesPerCondition"],
            "afferent": {"neurons": neural["afferent"]["neurons"], "filmMinusBlank": difference_summary(neural["afferent"]["filmMinusBlank"]), "responsiveNeurons": neural["afferent"]["responsiveNeurons"]},
            "nonafferent": {"neurons": neural["nonafferent"]["neurons"], "filmMinusBlank": difference_summary(neural["nonafferent"]["filmMinusBlank"]), "responsiveNeurons": neural["nonafferent"]["responsiveNeurons"]},
        },
        "action": {"scope": "all three residents", "valuesPerCondition": action["valuesPerCondition"], "changedTicks": action["changedTicks"], "changedResidentRows": action["changedResidentRows"], "filmMinusBlank": difference_summary(action["filmMinusBlank"])},
        "physical": {"scope": "all three residents", "bodyPositionFilmMinusBlank": difference_summary(physical["bodyPositionFilmMinusBlank"]), "filmPathLengthMeters": physical["filmPathLengthMeters"], "blankPathLengthMeters": physical["blankPathLengthMeters"], "maxRootDivergenceMeters": physical["maxRootDivergenceMeters"], "finalRootDivergenceMeters": physical["finalRootDivergenceMeters"]},
        "temporalResponse": screen_receipt["temporalResponse"],
        "scope": screen_receipt["scope"],
        "interpretation": "Paired differences exceeded the declared threshold in retinal, neural, action, and physical streams during the executed 30-second comparison.",
        "limitations": [
            "Retinal and neural captures are for resident 0; action and physical summaries cover all three residents.",
            "Each stream has its own sampled or aggregated first-change time; the receipt does not establish one simultaneous response.",
            "Responsive means the neuron exceeded an absolute paired-condition difference of 1e-6 at least once across 600 ticks.",
            "The result does not establish recovered physiology, stimulus understanding, or learned motor competence.",
        ],
    }
    public_screen_bytes = json.dumps(public_screen_receipt, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    public_screen_blob = byte_blob(public_screen_bytes, "public_screen_response_receipt", "application/json")

    sweep_id = f"fullgraph-sweep:{sha256(sweep_path)}"
    gam_id = f"gam-fit:{sha256(fit_path)}"
    prediction_id = f"heldout-setting-prediction:{hashlib.sha256(canonical(prediction)).hexdigest()}"
    confirmation_id = f"executed-setting-confirmation:{sha256(confirm_path)}"
    temporal_id = f"temporal-cns-fit:{sha256(training_path)}"
    ablation_id = f"graph-edge-ablation:{sha256(audit_path)}"
    service_id = f"trained-cns-export:{service_sha}"
    browser_id = f"browser-export:{base_release_blob['sha256']}"
    dawn_id = f"browser-numeric-probe:{sha256(dawn_path)}"
    failure_id = f"browser-load-failure:{sha256(joined_failure_path)}"
    joined_id = f"joined-browser-equivalent:{sha256(joined_path)}"
    collection_failure_id = f"teacher-collection-failure:{sha256(collection_failure_path)}"
    teacher_id = f"teacher-collection:{sha256(collection_path)}"
    controller_id = f"controller-training:{trained['artifact_sha256']}"
    native_id = f"native-controller-check:{sha256(native_path)}"
    trained_browser_id = f"trained-browser-export:{trained_release_blob['sha256']}"
    unmatched_rollout_id = f"rollout-identity-diagnostic:{sha256(unmatched_rollout_path)}"
    rollout_id = f"matched-physical-rollout:{sha256(rollout_path)}"
    screen_intervention_id = f"matched-physical-screen-intervention:{screen_receipt_blob['sha256']}"
    screen_result_id = "physical-screen-response-v2"

    records = [
        node(sweep_id, 0, "fullgraph_parameter_sweep", "Executed 27-setting crossed dynamics sweep on all 165,122 neurons and 25,563,197 graph edges.", blobs=[blob(sweep_path, "fullgraph_sweep_receipt", "application/json")], fields={"settings": 27, "ticks_per_setting": sweep["ticks"], "train_streams": 8, "heldout_streams": 4, "metric": sweep["metric"], "scope": "procedural full-graph diagnostic; not behavior"}),
        node(gam_id, 1, "native_gam_fit", "Native GAM surfaces fit the crossed sweep; held-out skill prediction was useful while recovery-memory prediction was not.", parents={sweep_id: "fit_source"}, blobs=gam_blobs, fields={"library": fit["library"], "skill_loo": fit["models"]["skill"]["loo"]["metrics"], "recovery_loo": fit["models"]["recovery_memory"]["loo"]["metrics"], "skill_predictive_useful": True, "recovery_predictive_useful": False, "scope": fit["interpretation"]}),
        node(prediction_id, 2, "heldout_parameter_prediction", "The fitted surface proposed one previously unmeasured in-bounds setting for an explicit full-graph rerun.", parents={gam_id: "predictive_model"}, fields={"setting": {key: prediction[key] for key in ("gain", "tau", "adaptation_gain")}, "predicted_skill": prediction["predicted_skill"], "predicted_recovery_memory": prediction["predicted_recovery_memory"], "skill_error_scale_rmse": prediction["skill_uncertainty"]["leave_one_configuration_out_rmse"], "confidence_interval": False, "required_next_step": prediction["required_next_step"]}),
        node(confirmation_id, 3, "executed_parameter_confirmation", "The proposed setting was executed on the full graph; its skill was close to the prediction without exceeding the best measured sweep result.", parents={prediction_id: "predicted_setting", sweep_id: "comparison_sweep"}, blobs=[blob(confirm_path, "fullgraph_confirmation_receipt", "application/json")], fields={"observed_skill": observed["skill"], "predicted_skill": prediction["predicted_skill"], "absolute_prediction_error": abs(observed["skill"] - prediction["predicted_skill"]), "best_measured_skill": fit["selection"]["best_measured_setting"]["skill"], "improved_over_best_measured": observed["skill"] > fit["selection"]["best_measured_setting"]["skill"], "observed_recovery_memory": observed["recovery_memory"], "recovery_prediction_was_reliable": False}),
        node(temporal_id, 0, "temporal_cns_fit", "A 192-update ROCm research fit trained full-graph temporal parameters and a supervised probe head.", blobs=[blob(training_path, "temporal_training_receipt", "application/json")], fields={"updates": 192, "seconds": training["seconds"], "procedural_frequency_split_after_skill": training["after"]["skill"], "procedural_future_optic_skill_vs_persistence": training["after"]["skill"][3:5], "procedural_future_body_skill_vs_persistence": training["after"]["skill"][5], "status": training["status"], "limitations": training["limitations"]}),
        node(ablation_id, 1, "graph_edge_ablation", "The canonical audit retained strong trained temporal-probe skill with graph edges intact and reduced it to approximately zero when all graph edges were removed.", parents={temporal_id: "audited_fit"}, blobs=[blob(audit_path, "canonical_edge_ablation_receipt", "application/json")], fields={"trained_temporal_probe": {"intact_skill": audit["intact"]["skill"], "zero_edge_skill": audit["zero_edges"]["skill"], "scope": "trained temporal probe replay and graph-edge ablation"}, "untrained_broad_head_startup": {"loss": audit["canonical_metrics"]["loss"], "future_optic_skill_vs_persistence": audit["canonical_metrics"]["future_optic_skill_vs_persistence"], "future_body_skill_vs_persistence": audit["canonical_metrics"]["future_body_skill_vs_persistence"], "scope": "gradient and execution startup check of newly initialized broad sensory heads; not a trained-head evaluation"}, "audit_scope": audit["scope"], "claim": "edge dependence of trained temporal-probe outputs plus finite startup gradients; not biological validation or behavior"}),
        node(service_id, 2, "trained_cns_export", "The audited temporal fit was serialized as one hash-bound full MaleCNS service artifact.", parents={temporal_id: "trained_parameters", ablation_id: "edge_dependence_audit"}, blobs=[service_blob, blob(service_receipt_path, "cns_service_receipt", "application/json")], fields={"neurons": 165122, "edges": 25563197, "graph_sha256": service_receipt["metadata"]["graph_sha256"], "atlas_sha256": service_receipt["metadata"]["atlas_sha256"], "training_status": "trained", "controller_status": "not trained"}),
        node(browser_id, 3, "browser_model_export", "The trained CNS and initialized-untrained resident were packed into a complete hash-verified browser release.", parents={service_id: "cns_service"}, blobs=[base_release_blob], fields={"release_files": len(base_release["files"]), "release_bytes": base_release["totalBytes"], "source_revision": base_release["sourceRevision"], "cns_service_artifact_sha256": base_release["serviceArtifactSha256"], "resident_artifact_sha256": base_release["residentArtifactSha256"], "controller_status": "initialized-untrained"}),
        node(dawn_id, 4, "browser_numeric_confirmation", "The browser WebGPU kernels executed eight full-graph ticks within pinned binary16 regression bounds and restored state byte-exactly.", parents={browser_id: "browser_release"}, blobs=[blob(dawn_path, "dawn_webgpu_probe_receipt", "application/json")], fields={"ticks": dawn["ticks"], "adapter": dawn["adapter"]["name"], "limits": dawn["limits"], "final_max_abs": dawn["finalMaxAbs"], "byte_exact_restore": dawn["byteExactRestore"], "limit_basis": dawn["limitBasis"]}),
        node(failure_id, 4, "executed_integration_failure", "The first whole-life load attempt failed because a reused MuJoCo Emscripten options object had been mutated.", parents={browser_id: "browser_release"}, blobs=[blob(joined_failure_path, "joined_load_failure_receipt", "application/json")], fields={"operation": joined_failure["operation"], "error": joined_failure["error"], "repair": joined_failure["repair"], "superseded_result": joined_id}),
        node(joined_id, 5, "joined_execution_confirmation", "After the recorded options-lifetime repair, the full CNS, Rust resident and MuJoCo world advanced and replayed one grown life exactly.", parents={browser_id: "browser_release", dawn_id: "numeric_confirmation", failure_id: "repaired_failure"}, blobs=[blob(joined_path, "joined_execution_receipt", "application/json")], fields={key: joined[key] for key in ("neurons", "edges", "residents", "modelSeconds", "meanCompleteTickMs", "maxCompleteTickMs", "wholeLifeReplayExact", "fullNeuralReplayExact", "physicalReplayExact", "restoredGrownWorld", "modelStatus", "controllerStatus", "scope")}),
        node(screen_intervention_id, 6, "matched_physical_screen_intervention", "From an exact shared initial state, the current V2 runtime executed 600 ticks with the official film decoded onto its physical screen and 600 matched ticks with that screen black.", parents={service_id: "trained_cns_service", browser_id: "initialized_resident", joined_id: "executed_runtime_predecessor"}, blobs=[screen_receipt_blob, screen_trace_blob, public_screen_blob], fields={"engine_identity": screen_receipt["engineIdentity"], "source_revision": screen_receipt["sourceRevision"], "cns_artifact_sha256": screen_receipt["model"]["cnsArtifactSha256"], "cns_service_artifact_sha256": screen_receipt["model"]["cnsServiceArtifactSha256"], "resident_artifact_sha256": screen_receipt["model"]["residentArtifactSha256"], "cns_training_status": "trained", "resident_training_status": "initialized-untrained", "stimulus": public_screen_receipt["stimulus"], "execution": execution, "matched_initial_state": initial, "screen_only_intervention": True}),
        node(screen_result_id, 7, "physical_screen_response", "The paired film and blank executions differed in resident 0 retina and neural rates, and in actions and physical motion across the three-resident batch.", parents={screen_intervention_id: "paired_execution"}, blobs=[public_screen_blob, screen_trace_blob], fields={"plot_url": "assets/live-cns-screen-response.svg", "threshold_absolute_difference": 1e-6, "retinal_neural_capture_resident": 0, "action_physical_batch": 3, "retinal_sites_ever_changed": retina["everChangedSites"], "retinal_supported_sites": retina["supportedSites"], "afferent_neurons_ever_changed": neural["afferent"]["responsiveNeurons"], "afferent_neurons": neural["afferent"]["neurons"], "nonafferent_neurons_ever_changed": neural["nonafferent"]["responsiveNeurons"], "nonafferent_neurons": neural["nonafferent"]["neurons"], "action_changed_ticks": action["changedTicks"], "action_changed_resident_rows": action["changedResidentRows"], "film_path_length_meters": physical["filmPathLengthMeters"], "blank_path_length_meters": physical["blankPathLengthMeters"], "maximum_root_divergence_meters": physical["maxRootDivergenceMeters"], "temporal_response": screen_receipt["temporalResponse"], "response_semantics": "ever exceeded absolute paired-condition difference 1e-6 over 600 ticks; stream timings are not a simultaneous-response claim", "outcome": "measured differences", "claims_not_established": ["recovered physiology", "stimulus understanding", "learned motor competence"]}),
        node(collection_failure_id, 6, "executed_collection_failure", "The first teacher collection stopped when a mutated Emscripten options object prevented held-out world reinitialization.", parents={joined_id: "runtime_predecessor", browser_id: "collection_release"}, blobs=[blob(collection_failure_path, "teacher_collection_failure_receipt", "application/json")], fields={"completed_before_failure": collection_failure["completed"], "failed_phase": collection_failure["failed_phase"], "error": collection_failure["error"], "diagnosis": collection_failure["diagnosis"], "resolution": collection_failure["resolution"]}),
        node(teacher_id, 6, "physical_teacher_collection", "Two actual full-CNS physical episodes supplied 576 transitions of privileged offline teacher supervision across train and held-out starts.", parents={joined_id: "executed_runtime", browser_id: "collection_release", collection_failure_id: "repaired_attempt"}, blobs=[blob(collection_path, "teacher_collection_receipt", "application/json"), *episode_blobs], fields={"action_source": collection["action_source"], "transitions": sum(item["transitions"] * item["batch"] for item in collection["episodes"]), "episodes": [{key: item[key] for key in ("split", "transitions", "batch", "positiveRewardFraction", "commandRms", "rawGeometryRetained", "rawSensesRetained")} for item in collection["episodes"]], "heldout_contact_attempts": collection["executionMetrics"]["heldout"]["contactAttempts"], "limitations": collection["limitations"]}),
        node(controller_id, 7, "resident_controller_training", "The current teacher data drove 128 optimizer updates and produced a new resident plus sequence-control artifact.", parents={teacher_id: "training_data", browser_id: "parent_controller"}, blobs=[blob(result_path, "resident_training_result", "application/json"), trained_blob, sequence_blob], fields={"updates": len(result["history"]), "seconds": result["seconds"], "validation_before": result["validation"]["before"], "validation_after": result["validation"]["after"], "resident_artifact_sha256": trained["artifact_sha256"], "sequence_control_artifact_sha256": sequence["artifact_sha256"], "scope": "offline privileged-teacher fit; physical competence requires matched rollout"}),
        node(native_id, 8, "native_controller_confirmation", "The trained resident matched the Python reference at the native Rust boundary within the declared tolerance.", parents={controller_id: "trained_controller", teacher_id: "heldout_episode"}, blobs=[blob(native_path, "native_controller_comparison", "application/json")], fields={"passed": native["passed"], "residents": native["residents"], "tolerance": native["tolerance"], "errors": native["errors"]}),
        node(trained_browser_id, 8, "trained_browser_export", "The trained resident was repacked with the same CNS service into a second complete hash-verified browser release.", parents={controller_id: "trained_controller", service_id: "same_cns_service", native_id: "native_confirmation"}, blobs=[trained_release_blob], fields={"release_files": len(trained_release["files"]), "release_bytes": trained_release["totalBytes"], "source_revision": trained_release["sourceRevision"], "resident_artifact_sha256": trained_release["residentArtifactSha256"], "service_artifact_sha256": trained_release["serviceArtifactSha256"], "training_status": "trained"}),
        node(unmatched_rollout_id, 9, "comparison_identity_diagnostic", "An initial two-arm physical run executed with matching numeric CNS buffers; its snapshot envelope hashes differed only in the sourceRevision identity header, producing a diagnostic false negative.", parents={browser_id: "initialized_controller_arm", trained_browser_id: "trained_controller_arm"}, blobs=[blob(unmatched_rollout_path, "initial_rollout_identity_receipt", "application/json")], fields={"reported_initial_world_and_cns_state_matched": False, "numeric_cns_buffers_matched": unmatched_rollout["cnsNumericBuffersMatched"], "difference_scope": "sourceRevision identity header only; decoded GPU state bytes and canonical host state matched", "diagnostic_false_negative": True, "world_seed_matched": unmatched_rollout["worldSeedMatched"], "action_and_suffix_seeds_matched": unmatched_rollout["actionAndSuffixSeedsMatched"], "physical_run_executed": True}),
        node(rollout_id, 10, "matched_physical_rollout", "The refined matched-state receipt confirmed the numeric CNS and host state, then fresh-life rollouts executed both initialized and trained controllers; action diversity increased while physical outcomes differed by resident and did not establish competence.", parents={browser_id: "initialized_controller_arm", trained_browser_id: "trained_controller_arm", controller_id: "training_result", unmatched_rollout_id: "refined_identity_comparison"}, blobs=[blob(rollout_path, "matched_physical_rollout_receipt", "application/json")], fields={"initial_world_and_cns_state_matched": rollout["initialWorldAndCnsStateMatched"], "initial_cns_comparison": rollout["initialCnsComparison"], "world_seed_matched": rollout["worldSeedMatched"], "action_and_suffix_seeds_matched": rollout["actionAndSuffixSeedsMatched"], "teacher_geometry_used_by_policy": rollout["teacherGeometryUsedByPolicy"], "arms": [metric_triplet(by_label[name]) for name in ("parent", "trained")], "outcome": "mixed", "interpretation": rollout["interpretation"]}),
    ]

    request = {
        "archive_id": "live-cns-wave-v2-20260907",
        "description": "Actual live-CNS research chain: dynamics, temporal fit, browser export, paired physical-screen response, teacher fit, and matched physical rollout.",
        "evidence": records,
    }
    for record in records:
        require(set(record["parent_ids"]) == set(record["fields"]["parent_roles"]), f"parent-role mismatch at {record['id']}")
    request_bytes = canonical(request)
    json.loads(request_bytes)

    output = args.output_dir
    public = args.public
    public_weave = public.with_suffix(".weave.json")
    public_screen = public.parent / "live-cns-screen-response.json"
    public_screen_trace = public.parent / "live-cns-screen-response.traces.json.gz"
    targets = [output / "live-cns-evidence.request.json", output / "live-cns-evidence.weave.json", output / "live-cns-evidence.receipt.json", public, public_weave, public_screen, public_screen_trace]
    if not args.replace:
        existing = [path for path in targets if path.exists()]
        require(not existing, "refusing to replace existing evidence: " + ", ".join(map(str, existing)))
    output.mkdir(parents=True, exist_ok=True)
    public.parent.mkdir(parents=True, exist_ok=True)
    request_path, weave_path, receipt_path = targets[:3]
    request_path.write_bytes(json.dumps(request, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    completed = subprocess.run(
        ["cargo", "run", "--locked", "--release", "--manifest-path", str(WEAVE / "Cargo.toml"), "--", "--input", str(request_path), "--output", str(weave_path)],
        cwd=WEAVE,
        check=True,
        capture_output=True,
        text=True,
    )
    portable = json.loads(completed.stdout)
    require(portable.get("reload_equal") is True and portable.get("validated_after_reload") is True, "native Weave roundtrip failed")
    require(portable.get("node_count") == len(records), "native Weave node count differs")
    portable["artifact"] = public_weave.name
    portable["artifact_sha256"] = sha256(weave_path)
    portable["public_summary"] = {
        "status": "executed research chain with mixed outcomes",
        "fullgraph_settings": 27,
        "neurons": 165122,
        "edges": 25563197,
        "teacher_updates": 128,
        "screen_response": {
            "result_node_id": screen_result_id,
            "model_seconds_per_condition": execution["modelSeconds"],
            "retinal_neural_capture_resident": 0,
            "action_physical_batch": 3,
            "nonafferent_neurons_ever_changed": neural["nonafferent"]["responsiveNeurons"],
            "nonafferent_neurons": neural["nonafferent"]["neurons"],
            "threshold_absolute_difference": 1e-6,
        },
        "matched_rollout_outcome": "mixed",
        "claims_not_established": ["biological parameter recovery", "general optic/body prediction improvement beyond the procedural fit split", "simultaneous cross-stream response", "stimulus understanding", "general embodied competence"],
    }
    encoded = json.dumps(portable, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    public.write_bytes(encoded)
    shutil.copyfile(weave_path, public_weave)
    public_screen.write_bytes(public_screen_bytes)
    shutil.copyfile(screen_trace_path, public_screen_trace)
    require(sha256(public_screen) == public_screen_blob["sha256"], "public screen response receipt changed during write")
    require(sha256(public_screen_trace) == screen_trace_blob["sha256"], "public screen response trace changed during copy")
    receipt = {
        "format": "chreatures-live-cns-weave-export-v1",
        "request_sha256": sha256(request_path),
        "weave_sha256": sha256(weave_path),
        "portable_sha256": sha256(public),
        "node_count": portable["node_count"],
        "edge_count": portable["edge_count"],
        "multi_parent_nodes": portable["multi_parent_nodes"],
        "reload_equal": True,
        "validated_after_reload": True,
        "universal_weave": portable["library"],
        "public_files": [public.name, public_weave.name, public_screen.name, public_screen_trace.name],
        "public_screen_response_sha256": public_screen_blob["sha256"],
        "public_screen_trace_sha256": screen_trace_blob["sha256"],
        "original_screen_response_receipt_sha256": screen_receipt_blob["sha256"],
        "source_directories": [path.name for path in (crossed, temporal, browser, teacher, screen)],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paperbin", type=Path, default=DEFAULT_PAPERBIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--replace", action="store_true", help="explicitly replace this exact evidence projection")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
