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


def training_validation_summary(result: dict[str, Any]) -> dict[str, Any]:
    before = result["validation"]["before"]
    after = result["validation"]["after"]
    return {
        "before_overall": before["overall"],
        "after_overall": after["overall"],
        "total_by_skill": {
            skill: {"before": metrics["total"], "after": after["by_skill"][skill]["total"]}
            for skill, metrics in before["by_skill"].items()
        },
    }


def autonomous_arm(result: dict[str, Any]) -> dict[str, Any]:
    physical = result["physical"]
    action = result["action"]
    return {
        "label": result["label"],
        "training_status": result["trainingStatus"],
        "resident_artifact_sha256": require_hash(result["residentArtifactSha256"], "autonomous resident"),
        "assimilable_reserve_change": physical["assimilableReserveChange"],
        "fatigue_change": physical["fatigueChange"],
        "net_target_food_stock_loss": physical["netTargetFoodStockLoss"],
        "stop_ticks_below_1cm_per_second": physical["physicalStopTicksAt1cmPerSecond"],
        "target_proximity_ticks_below_34cm": physical["targetProximityTicksBelow34cm"],
        "path_length_meters": physical["pathLengthMeters"],
        "mean_squared_effort": action["meanSquaredEffort"],
        "distinct_action_rows_rounded_1e3": action["distinctRowsRounded1e3"],
        "execution": result["execution"],
    }


def mean(values: list[float | int]) -> float:
    require(bool(values), "cannot average an empty measurement")
    return sum(values) / len(values)


def aggregate_map_outcomes(reports: list[dict[str, Any]], label: str) -> dict[str, Any]:
    arms = []
    for report in reports:
        matches = [result for result in report["results"] if result["label"] == label]
        require(len(matches) == 1, f"MAP report must contain one {label} arm")
        arms.append(matches[0])
    reserve = [value for arm in arms for value in arm["physical"]["assimilableReserveChange"]]
    fatigue = [value for arm in arms for value in arm["physical"]["fatigueChange"]]
    stops = [value for arm in arms for value in arm["physical"]["physicalStopTicksAt1cmPerSecond"]]
    proximity = [value for arm in arms for value in arm["physical"]["targetProximityTicksBelow34cm"]]
    return {
        "resident_world_count": len(reserve),
        "mean_assimilable_reserve_change": mean(reserve),
        "mean_fatigue_change": mean(fatigue),
        "mean_stop_ticks_below_1cm_per_second": mean(stops),
        "mean_target_proximity_ticks_below_34cm": mean(proximity),
        "world_count": len(arms),
        "mean_world_total_net_target_food_stock_loss": mean([sum(arm["physical"]["netTargetFoodStockLoss"]) for arm in arms]),
        "mean_world_batch_squared_effort": mean([arm["action"]["meanSquaredEffort"] for arm in arms]),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    crossed = args.paperbin / "dynamics-v2-crossed-20260907"
    temporal = args.paperbin / "dynamics-v2-temporal-fit-20260907"
    browser = args.paperbin / "browser-v2-temporal-pack-20260907"
    teacher = args.paperbin / "browser-v2-teacher-skills-20260907"
    screen = args.paperbin / "screen-response-v2-20260907"
    curriculum = args.paperbin / "browser-v2-teacher-curriculum-512x8-20260907"

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

    curriculum_source_path = curriculum / "source" / "combined-receipt.json"
    curriculum_collection_path = curriculum / "corpus" / "collection-receipt.json"
    curriculum_corpus_path = curriculum / "corpus" / "corpus.json"
    curriculum_source = read_json(curriculum_source_path)
    curriculum_collection = read_json(curriculum_collection_path)
    curriculum_corpus = read_json(curriculum_corpus_path)
    curriculum_source_blob = blob(curriculum_source_path, "curriculum_collection_receipt", "application/json")
    require(curriculum_source_blob["sha256"] == sha256(curriculum_collection_path), "curriculum source and packed collection receipts differ")
    require(curriculum_source.get("format") == "chreatures-browser-teacher-collection-combined-receipt-v2" and curriculum_source.get("complete") is True, "curriculum collection is incomplete")
    require(curriculum_source.get("episode_count") == 8 and curriculum_source.get("transitions_per_episode") == 512 and curriculum_source.get("batch") == 3, "curriculum collection extent differs")
    require(curriculum_source.get("cns_service_artifact_sha256") == service_sha and curriculum_source.get("resident_artifact_sha256") == base_release["residentArtifactSha256"], "curriculum collection model identity differs")
    require(curriculum_source.get("action_source") == "privileged-physical-teacher" and curriculum_source.get("raw_geometry_retained") is False and curriculum_source.get("raw_senses_retained_by_resident_episodes") is False, "curriculum teacher boundary differs")
    require(curriculum_source.get("afferent_sidecars_excluded_from_resident_training") is True and curriculum_source.get("validation", {}).get("all_coverage_flags_passed") is True, "curriculum corpus boundary validation differs")
    source_episode_blobs = []
    for episode in curriculum_source.get("episodes", []):
        source_episode_blobs.append(blob(curriculum / "source" / Path(episode["file"]).name, f"curriculum_source_episode_{episode['episode_index']}", "application/x-npz", episode["sha256"]))
    require(len(source_episode_blobs) == 8, "curriculum source episode count differs")

    require(curriculum_corpus.get("format") == "chreatures-cns-resident-corpus-v1", "curriculum corpus format differs")
    require(curriculum_corpus.get("collection_receipt", {}).get("file_sha256") == curriculum_source_blob["sha256"], "curriculum corpus collection identity differs")
    require(curriculum_corpus.get("parent", {}).get("artifact_sha256") == base_release["residentArtifactSha256"], "curriculum corpus parent differs")
    require(curriculum_corpus.get("controller_input_fields") == ["cns_latent", "previous_delivered_command", "reset"], "curriculum controller inputs differ")
    require(curriculum_corpus.get("teacher_only_fields") == ["delivered_command", "physical_reward", "terminal", "teacher_bouts"], "curriculum teacher-only fields differ")
    require(len(curriculum_corpus.get("train", [])) == 6 and len(curriculum_corpus.get("heldout_worlds", [])) == 2, "curriculum split differs")
    corpus_episode_blobs = []
    for episode in curriculum_corpus.get("episodes", []):
        require(sha256(curriculum / "source" / Path(episode["source"]).name) == require_hash(episode["source_file_sha256"], "curriculum source episode"), "curriculum source-to-corpus identity differs")
        corpus_episode_blobs.append(blob(curriculum / "corpus" / episode["file"], f"curriculum_corpus_episode_{episode['episode_index']}", "application/x-npz", episode["file_sha256"]))
    require(len(corpus_episode_blobs) == 8 and len({entry["layout_identity"] for entry in curriculum_corpus["episodes"]}) == 8, "curriculum corpus episode identities differ")
    curriculum_corpus_blob = blob(curriculum_corpus_path, "curriculum_corpus_manifest", "application/json")

    first_fit_path = curriculum / "artifacts" / "result.json"
    first_native_path = curriculum / "artifacts" / "native-comparison.json"
    corrected_fit_path = curriculum / "artifacts-causal-currentkey" / "result.json"
    corrected_native_path = curriculum / "artifacts-causal-currentkey" / "native-comparison.json"
    first_fit, first_native, corrected_fit, corrected_native = map(read_json, (first_fit_path, first_native_path, corrected_fit_path, corrected_native_path))
    for label, fit_result in (("first curriculum", first_fit), ("corrected curriculum", corrected_fit)):
        require(fit_result.get("format") == "chreatures-cns-resident-training-result-v1" and fit_result.get("complete") is True and len(fit_result.get("history", [])) == 256, f"{label} fit is incomplete")
        require(fit_result.get("parent", {}).get("artifact_sha256") == base_release["residentArtifactSha256"], f"{label} parent differs")
        require(fit_result.get("corpus", {}).get("file_sha256") == first_fit["corpus"]["file_sha256"], f"{label} training corpus identity differs")
        require(fit_result.get("source", {}).get("data_sha256") == "b6e6f420294b7edf3ce79a310d8d971db16e9b4627d92bb111df2be91a2a453c" and fit_result.get("source", {}).get("trainer_sha256") == "a68ebfe22ba6ae4347bc538be96dfc8dc74c9340bdd29bb1bf0d86f074279f3d", f"{label} training source differs")
    require(first_fit["source"].get("model_sha256") == "13b74aee8f4ca27a634b7929c59addde47372d91cd8a56d9ce59720d01da1e0a", "first curriculum model source differs")
    require(corrected_fit["source"].get("model_sha256") == "84d1dbce05b76fc97c8169ad98fa904237d48ec36ebdfb1100f294956c04f8ab", "corrected curriculum model source differs")
    first_resident = first_fit["publication"]["resident"]
    first_sequence = first_fit["publication"]["sequence_control"]
    corrected_resident = corrected_fit["publication"]["resident"]
    corrected_sequence = corrected_fit["publication"]["sequence_control"]
    first_fit_blobs = [
        blob(first_fit_path, "future_key_contaminated_fit_receipt", "application/json"),
        blob(curriculum / "artifacts" / "cns-resident-trained.npz", "future_key_contaminated_resident", "application/x-npz", first_resident["file_sha256"]),
        blob(curriculum / "artifacts" / "sequence-control-trained.npz", "future_key_contaminated_sequence_control", "application/x-npz", first_sequence["file_sha256"]),
        blob(first_native_path, "future_key_contaminated_native_comparison", "application/json"),
    ]
    corrected_fit_blobs = [
        blob(corrected_fit_path, "current_key_fit_receipt", "application/json"),
        blob(curriculum / "artifacts-causal-currentkey" / "cns-resident-trained.npz", "current_key_resident", "application/x-npz", corrected_resident["file_sha256"]),
        blob(curriculum / "artifacts-causal-currentkey" / "sequence-control-trained.npz", "current_key_sequence_control", "application/x-npz", corrected_sequence["file_sha256"]),
        blob(corrected_native_path, "current_key_native_comparison", "application/json"),
    ]
    require(first_native.get("format") == "chreatures-cns-resident-native-comparison-v1" and first_native.get("passed") is True and first_native.get("resident_file_sha256") == first_resident["file_sha256"], "first curriculum native comparison differs")
    require(corrected_native.get("format") == "chreatures-cns-resident-native-comparison-v1" and corrected_native.get("passed") is True and corrected_native.get("resident_file_sha256") == corrected_resident["file_sha256"], "corrected curriculum native comparison differs")
    first_curriculum_release, first_curriculum_release_blob = verify_release(curriculum / "trained-browser-model")
    corrected_release, corrected_release_blob = verify_release(curriculum / "trained-browser-model-causal-currentkey")
    require(first_curriculum_release.get("residentArtifactSha256") == first_resident["artifact_sha256"], "first curriculum browser release differs")
    require(corrected_release.get("residentArtifactSha256") == corrected_resident["artifact_sha256"] and corrected_release.get("serviceArtifactSha256") == service_sha, "corrected curriculum browser release differs")

    candidate01_path = curriculum / "autonomous-outcome-assay-candidate01-512.json"
    candidate02_path = curriculum / "autonomous-outcome-assay-candidate02-512.json"
    map_assay_path = curriculum / "autonomous-outcome-assay-candidate02-map-512.json"
    candidate01, candidate02, map_assay = map(read_json, (candidate01_path, candidate02_path, map_assay_path))
    for label, assay in (("candidate 01", candidate01), ("candidate 02", candidate02), ("MAP intervention", map_assay)):
        require(assay.get("format") == "chreatures-autonomous-physical-outcome-assay-v1", f"{label} assay format differs")
        require(assay.get("initialWorldAndCnsStateMatched") is True and assay.get("worldSeedMatched") is True and assay.get("actionAndSuffixSeedsMatched") is True and assay.get("cnsNumericBuffersMatched") is True, f"{label} assay initial state differs")
        require(assay.get("teacherGeometryUsedByPolicy") is False and assay.get("policyInputContract") == ["cns_latent[512]", "previous_delivered_command[12]", "reset"], f"{label} policy boundary differs")
        require(assay.get("cnsServiceArtifactSha256") == service_sha and all(result.get("transitions") == 512 and result.get("execution", {}).get("failed") is False for result in assay.get("results", [])), f"{label} execution differs")
    candidate01_arms = {entry["label"]: entry for entry in candidate01["results"]}
    candidate02_arms = {entry["label"]: entry for entry in candidate02["results"]}
    require(candidate01_arms.get("trained", {}).get("residentArtifactSha256") == first_resident["artifact_sha256"], "candidate 01 assay used another fit")
    require(candidate02_arms.get("candidate01-future-key-contaminated", {}).get("residentArtifactSha256") == first_resident["artifact_sha256"] and candidate02_arms.get("trained", {}).get("residentArtifactSha256") == corrected_resident["artifact_sha256"], "candidate 02 assay arms differ")
    require(candidate02_arms.get("parent", {}).get("residentArtifactSha256") == base_release["residentArtifactSha256"] and "zero-command" in candidate02_arms, "candidate 02 baselines differ")

    corrected_manifest_path = curriculum / "trained-browser-model-causal-currentkey" / "resident-manifest.json"
    map_manifest_path = curriculum / "trained-browser-model-causal-currentkey-map" / "resident-manifest.json"
    corrected_manifest, map_manifest = map(read_json, (corrected_manifest_path, map_manifest_path))
    require(map_manifest.get("format") == "chreatures-browser-resident-v1" and map_manifest.get("parentArtifactSha256") == corrected_resident["artifact_sha256"], "MAP derivative parent differs")
    require(map_manifest.get("buffers") == corrected_manifest.get("buffers") and map_manifest.get("cnsServiceArtifactSha256") == service_sha, "MAP derivative changed immutable weights")
    require(map_manifest.get("actionModeIntervention", {}).get("action_mode") == "map" and map_manifest.get("config", {}).get("action_mode") == "map", "MAP derivative action mode differs")
    require({key: value for key, value in map_manifest["config"].items() if key != "action_mode"} == {key: value for key, value in corrected_manifest["config"].items() if key != "action_mode"}, "MAP derivative changed configuration beyond action mode")
    require(map_manifest.get("trainingStatus") == "trained-weights-map-action-research-intervention", "MAP derivative status differs")
    for buffer in map_manifest["buffers"].values():
        blob(curriculum / "trained-browser-model-causal-currentkey-map" / buffer["url"], "map_intervention_shared_weight", "application/gzip", buffer["transportSha256"])
    map_manifest_blob = blob(map_manifest_path, "map_action_derivative_manifest", "application/json")
    require(map_assay.get("referenceReportSha256") == sha256(candidate02_path), "MAP assay reference report differs")
    require(len(map_assay.get("results", [])) == 1 and map_assay["results"][0].get("residentArtifactSha256") == map_manifest["artifactSha256"], "MAP assay used another derivative")

    map_generalization = curriculum / "map-generalization-3worlds"
    map_recipe_path = map_generalization / "recipe.json"
    map_recipe = read_json(map_recipe_path)
    require(map_recipe.get("format") == "chreatures-map-generalization-recipe-v1" and map_recipe.get("ticks") == 512 and len(map_recipe.get("contexts", [])) == 3, "MAP generalization recipe differs")
    map_reports = []
    map_report_blobs = []
    for context in map_recipe["contexts"]:
        report_path = map_generalization / context["report"]
        report = read_json(report_path)
        require(report.get("format") == "chreatures-autonomous-physical-outcome-assay-v1" and report.get("worldSeed") == context["world_seed"] and report.get("variationSeed") == context["variation_seed"], "MAP generalization context differs")
        require(report.get("initialWorldAndCnsStateMatched") is True and report.get("worldSeedMatched") is True and report.get("actionAndSuffixSeedsMatched") is True and report.get("cnsNumericBuffersMatched") is True and report.get("teacherGeometryUsedByPolicy") is False, "MAP generalization state or policy boundary differs")
        report_arms = {entry["label"]: entry for entry in report.get("results", [])}
        require(report_arms.get("parent", {}).get("residentArtifactSha256") == base_release["residentArtifactSha256"] and report_arms.get("trained", {}).get("residentArtifactSha256") == map_manifest["artifactSha256"], "MAP generalization arms differ")
        require(all(entry.get("transitions") == 512 and entry.get("execution", {}).get("failed") is False for entry in report_arms.values()), "MAP generalization execution differs")
        map_reports.append(report)
        map_report_blobs.append(blob(report_path, f"map_generalization_world_{context['world_seed']}", "application/json"))
    parent_map_aggregate = aggregate_map_outcomes(map_reports, "parent")
    trained_map_aggregate = aggregate_map_outcomes(map_reports, "trained")

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
    curriculum_id = f"curriculum-corpus:{curriculum_corpus_blob['sha256']}"
    rejected_fit_id = f"rejected-future-key-fit:{sha256(first_fit_path)}"
    corrected_fit_id = f"current-key-controller-fit:{sha256(corrected_fit_path)}"
    autonomous_id = f"autonomous-controller-outcomes:{sha256(candidate02_path)}"
    map_intervention_id = f"map-action-intervention:{map_manifest_blob['sha256']}"
    map_generalization_id = f"map-generalization-3worlds:{hashlib.sha256(canonical([blob_ref['sha256'] for blob_ref in map_report_blobs])).hexdigest()}"

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
        node(
            curriculum_id,
            6,
            "resident_teacher_curriculum",
            "Eight fresh three-resident worlds supplied 12,288 physical transitions through the current CNS, split into six training worlds and two identity-disjoint held-out worlds.",
            parents={browser_id: "initialized_resident", joined_id: "executed_runtime_lineage", service_id: "same_cns_service"},
            blobs=[curriculum_source_blob, curriculum_corpus_blob, *source_episode_blobs, *corpus_episode_blobs],
            fields={
                "episodes": 8,
                "transitions_per_episode": 512,
                "resident_batch": 3,
                "physical_transitions": 12288,
                "train_worlds": 6,
                "heldout_worlds": 2,
                "curriculum_contract_sha256": curriculum_source["curriculum_contract_sha256"],
                "source_collection_sha256": curriculum_source_blob["sha256"],
                "rehydrated_corpus_manifest_sha256": curriculum_corpus_blob["sha256"],
                "training_input_corpus_sha256": first_fit["corpus"]["file_sha256"],
                "action_source": curriculum_source["action_source"],
                "controller_input_fields": curriculum_corpus["controller_input_fields"],
                "teacher_only_fields": curriculum_corpus["teacher_only_fields"],
                "raw_geometry_retained": False,
                "raw_senses_retained": False,
                "afferent_sidecars_excluded_from_training": True,
                "source_amendment_scopes": [entry["scope"] for entry in curriculum_source["source_amendments"]],
                "default_resident_artifact_sha256": base_release["residentArtifactSha256"],
            },
        ),
        node(
            rejected_fit_id,
            7,
            "rejected_controller_fit",
            "The first 256-update curriculum fit reduced recorded held-out losses, but its dynamics forecast received the future achieved key and was rejected as causally contaminated.",
            parents={curriculum_id: "training_corpus"},
            blobs=[*first_fit_blobs, first_curriculum_release_blob],
            fields={
                "updates": 256,
                "seconds": first_fit["seconds"],
                "validation": training_validation_summary(first_fit),
                "resident_artifact_sha256": first_resident["artifact_sha256"],
                "sequence_control_artifact_sha256": first_sequence["artifact_sha256"],
                "native_comparison": {"passed": first_native["passed"], "tolerance": first_native["tolerance"], "errors": first_native["errors"]},
                "forecast_input": "future achieved key",
                "rejection_reason": "future target information entered the dynamics forecast context",
                "hindsight_inverse_action_use_valid": True,
                "status": "executed fit rejected for forecasting contamination",
                "publication_status": "research only; not the default resident",
            },
        ),
        node(
            corrected_fit_id,
            8,
            "causal_controller_fit",
            "A second 256-update fit retained the held-out loss reductions while forecasting from the current experienced key; the future achieved key remained only in hindsight inverse-action supervision.",
            parents={curriculum_id: "same_training_corpus", rejected_fit_id: "causal_correction"},
            blobs=[*corrected_fit_blobs, corrected_release_blob],
            fields={
                "updates": 256,
                "seconds": corrected_fit["seconds"],
                "validation": training_validation_summary(corrected_fit),
                "resident_artifact_sha256": corrected_resident["artifact_sha256"],
                "sequence_control_artifact_sha256": corrected_sequence["artifact_sha256"],
                "native_comparison": {"passed": corrected_native["passed"], "tolerance": corrected_native["tolerance"], "errors": corrected_native["errors"]},
                "forecast_input": "current experienced key",
                "future_achieved_key_scope": "hindsight inverse-action supervision only",
                "browser_release_sha256": corrected_release_blob["sha256"],
                "publication_status": "research candidate only; not the default resident",
            },
        ),
        node(
            autonomous_id,
            9,
            "autonomous_controller_outcomes",
            "Matched 512-tick autonomous assays found that the corrected fit kept its offline loss gains but used more effort, lost more reserve, and produced less net target food-stock loss than the initialized parent in this world.",
            parents={browser_id: "initialized_parent_arm", rejected_fit_id: "rejected_candidate_arm", corrected_fit_id: "corrected_candidate_arm"},
            blobs=[blob(candidate01_path, "candidate_01_autonomous_assay", "application/json"), blob(candidate02_path, "candidate_02_autonomous_assay", "application/json")],
            fields={
                "world_seed": candidate02["worldSeed"],
                "variation_seed": candidate02["variationSeed"],
                "ticks": 512,
                "resident_batch": 3,
                "matched_initial_world_and_cns_state": True,
                "teacher_geometry_used_by_policy": False,
                "policy_input_contract": candidate02["policyInputContract"],
                "initialized_parent": autonomous_arm(candidate02_arms["parent"]),
                "future_key_contaminated_candidate": autonomous_arm(candidate02_arms["candidate01-future-key-contaminated"]),
                "corrected_current_key_candidate": autonomous_arm(candidate02_arms["trained"]),
                "zero_command_baseline": autonomous_arm(candidate02_arms["zero-command"]),
                "earlier_candidate_01_assay_trained_arm": autonomous_arm(candidate01_arms["trained"]),
                "outcome": "offline validation losses improved; autonomous physical cost regressed in this matched world",
                "claims_not_established": ["goal conditioning", "general embodied competence"],
            },
        ),
        node(
            map_intervention_id,
            10,
            "map_action_intervention",
            "A research-only derivative kept every corrected controller weight fixed and changed action selection from sampling to maximum a posteriori selection for an executed matched assay.",
            parents={corrected_fit_id: "identical_trained_weights", autonomous_id: "motivating_sampled_action_outcome"},
            blobs=[map_manifest_blob, blob(map_assay_path, "map_action_autonomous_assay", "application/json")],
            fields={
                "parent_resident_artifact_sha256": corrected_resident["artifact_sha256"],
                "derivative_resident_artifact_sha256": map_manifest["artifactSha256"],
                "changed_configuration": {"action_mode": {"from": corrected_manifest["config"]["action_mode"], "to": map_manifest["config"]["action_mode"]}},
                "immutable_weight_buffers_equal": True,
                "reference_autonomous_assay_sha256": map_assay["referenceReportSha256"],
                "single_world_outcome": autonomous_arm(map_assay["results"][0]),
                "status": "research intervention; not the default resident",
            },
        ),
        node(
            map_generalization_id,
            11,
            "map_action_generalization",
            "Across three new matched worlds, MAP selection increased stopping and proximity but had slightly worse mean reserve and fatigue changes, higher effort, and less net target food-stock loss than the initialized parent.",
            parents={map_intervention_id: "tested_configuration", autonomous_id: "prior_single_world_assay"},
            blobs=[blob(map_recipe_path, "map_generalization_recipe", "application/json"), *map_report_blobs],
            fields={
                "contexts": [{"world_seed": context["world_seed"], "variation_seed": context["variation_seed"], "report_sha256": map_report_blobs[index]["sha256"]} for index, context in enumerate(map_recipe["contexts"])],
                "ticks_per_world": 512,
                "resident_batch": 3,
                "initialized_parent": parent_map_aggregate,
                "map_action_candidate": trained_map_aggregate,
                "unit_of_analysis": "reserve, fatigue, stopping, and proximity means cover 9 resident-worlds; food-stock loss is a per-world resident-total mean; effort is a 3-world batch mean; adjacent ticks are not independent replicates",
                "outcome": "mixed, with worse physiology and net food-stock-loss measures; no promotion",
                "default_resident_unchanged": True,
                "claims_not_established": ["generalization beyond these three contexts", "goal conditioning", "general embodied competence"],
            },
        ),
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
        "description": "Actual live-CNS research chain: dynamics, temporal fit, browser export, paired physical-screen response, teacher curricula, and matched physical rollouts.",
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
        "curriculum_controller": {
            "worlds": 8,
            "physical_transitions": 12288,
            "updates": 256,
            "forecast_input": "current experienced key",
            "hindsight_future_key_scope": "inverse-action supervision only",
            "autonomous_outcome": "offline loss improvement with physical cost regression in the first matched world",
            "map_three_world_outcome": "mixed, with worse physiology and net food-stock-loss measures; no promotion",
            "default_resident_unchanged": True,
        },
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
        "claims_not_established": ["biological parameter recovery", "general optic/body prediction improvement beyond the procedural fit split", "simultaneous cross-stream response", "stimulus understanding", "goal conditioning", "general embodied competence"],
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
        "source_directories": [path.name for path in (crossed, temporal, browser, teacher, screen, curriculum)],
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
