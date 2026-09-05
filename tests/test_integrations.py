import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
GAM_SCRIPT = ROOT / "integrations" / "gamfit_regression.py"
OBSERVE_SCRIPT = ROOT / "integrations" / "observe_habitat.py"
WEAVE_CRATE = ROOT / "integrations" / "weave"


def _load_gam_integration():
    spec = importlib.util.spec_from_file_location("gamfit_regression", GAM_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_observe_integration():
    spec = importlib.util.spec_from_file_location("observe_habitat", OBSERVE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_response_and_split_are_deterministic() -> None:
    integration = _load_gam_integration()
    train, test, train_y = integration.make_dataset()

    assert len(train) == 120
    assert len(test) == 60
    assert len(train_y) == len(train)
    assert [row["x"] for row in test[:3]] == pytest.approx(
        [-3.0, -2.899441340782123, -2.798882681564246]
    )
    x = np.array([-1.0, 0.0, 1.0])
    assert integration.response_law(x) == pytest.approx(
        [0.65 - np.sin(2.2), 0.0, 0.65 + np.sin(2.2)]
    )


def test_checked_in_gamfit_run_is_native_and_beats_null() -> None:
    result_path = ROOT / "integrations" / "artifacts" / "gamfit" / "regression_result.json"
    model_path = ROOT / "integrations" / "artifacts" / "gamfit" / "nonlinear_response.gam"
    result = json.loads(result_path.read_text())

    assert result["library"]["version"] == "0.1.259"
    assert result["library"]["source_commit"] == "7c7eca8ac4826de95c8e743a20294bee132a9bcc"
    assert result["library"]["native_extension_available"] is True
    assert result["metrics"]["rmse_ratio"] < 0.2
    assert result["persistence"]["reload_max_abs_prediction_delta"] == 0.0
    assert model_path.stat().st_size == result["persistence"]["bytes"]
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == result["persistence"]["sha256"]


def test_checked_in_observatory_report_uses_habitat_records() -> None:
    artifact_dir = ROOT / "integrations" / "artifacts" / "observatory"
    report = json.loads((artifact_dir / "observatory_report.json").read_text())

    assert report["report_type"] == "chreatures-observatory"
    assert report["provenance"]["input_format"] == "chreatures-checkpoint-v1"
    assert report["provenance"]["checkpoint_verified"] is True
    assert report["provenance"]["telemetry_rows"] == 378
    assert report["provenance"]["rejected_telemetry_rows"] == 0
    excerpt = artifact_dir / report["provenance"]["telemetry_excerpt"]["file"]
    assert hashlib.sha256(excerpt.read_bytes()).hexdigest() == report["provenance"]["telemetry_excerpt"]["sha256"]
    assert report["gamfit"]["status"] == "complete"
    assert report["gamfit"]["interpretation"] == "descriptive only; no causal claim"
    assert any("RHO uncertainty" in line for line in report["gamfit"]["native_messages"])
    assert report["gamfit"]["persistence"]["reload_max_abs_prediction_delta"] == 0.0
    model = artifact_dir / report["gamfit"]["persistence"]["model_file"]
    assert hashlib.sha256(model.read_bytes()).hexdigest() == report["gamfit"]["persistence"]["sha256"]
    assert report["weave"]["status"] == "complete"
    assert report["weave"]["node_count"] == 5
    assert [record["record_type"] for record in report["weave"]["records"]].count("episode") == 4
    weave = artifact_dir / report["weave"]["artifact_file"]
    assert weave.stat().st_size == report["weave"]["bytes"]


def test_universal_weave_native_roundtrip(tmp_path: Path) -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is required for the native Universal Weave smoke test")

    artifact = tmp_path / "evidence.weave.json"
    request = tmp_path / "journal.json"
    request.write_text(
        json.dumps(
            {
                "habitat_id": "habitat-real-schema-test",
                "journal": [
                    {"id": "hab:0:0", "time": 0.0, "kind": "hatched", "text": "Hatched."},
                    {"id": "hab:8:1", "time": 0.4, "kind": "caregiver", "text": "Placed flower."},
                ],
                "evidence": [
                    {
                        "id": "fit:1",
                        "time": 0.4,
                        "text": "Descriptive fit.",
                        "artifact_uri": "file:activity.gam",
                        "parent_ids": ["hab:8:1"],
                    }
                ],
            }
        )
    )
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "--",
            "--input",
            str(request),
            "--output",
            str(artifact),
        ],
        cwd=WEAVE_CRATE,
        check=True,
        capture_output=True,
        env={**os.environ, "CARGO_TARGET_DIR": str(tmp_path / "cargo-target")},
        text=True,
        timeout=180,
    )
    receipt = json.loads(completed.stdout)

    assert receipt["library"]["version"] == "0.5.0"
    assert receipt["library"]["source_commit"] == "7a5a0dabb94885e44ad8a6c4355c015d7f38020f"
    assert receipt["node_count"] == 3
    assert receipt["reload_equal"] is True
    assert receipt["validated_after_reload"] is True
    assert artifact.stat().st_size == receipt["bytes"]
    records = {record["source_id"]: record for record in receipt["records"]}
    assert records["hab:8:1"]["time"] == 0.4
    assert records["hab:8:1"]["text"] == "Placed flower."
    assert records["fit:1"]["parents"] == [records["hab:8:1"]["node_id"]]


def test_habitat_checkpoint_extraction_and_sparse_fit(tmp_path: Path) -> None:
    observe = _load_observe_integration()
    state = {
        "id": "resident-test",
        "journal": [{"id": "resident-test:0:0", "time": 0.0, "kind": "hatched", "text": "Hatched."}],
        "history": {
            "mica": [
                {"time": 0.5, "energy": 0.9, "activity": 0.2},
                {"time": 1.0, "energy": 0.89, "activity": "not-a-number"},
            ]
        },
    }
    checkpoint = {
        "format": "chreatures-checkpoint-v1",
        "sha256": hashlib.sha256(observe._canonical(state)).hexdigest(),
        "state": state,
    }
    path = tmp_path / "residents.json"
    path.write_text(json.dumps(checkpoint))

    loaded = observe.load_records(path)
    assert loaded["provenance"]["checkpoint_verified"] is True
    assert loaded["provenance"]["telemetry_rows"] == 1
    assert loaded["provenance"]["rejected_telemetry_rows"] == 1
    assert loaded["journal"] == state["journal"]
    fit = observe.fit_activity(loaded["telemetry"], tmp_path)
    assert fit["status"] == "skipped"
    assert "need at least 60" in fit["reason"]


def test_observatory_report_keeps_sparse_data_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observe = _load_observe_integration()
    state = {
        "id": "sparse-resident",
        "journal": [
            {
                "id": "sparse-resident:0:0",
                "time": 0.0,
                "kind": "hatched",
                "text": "Hatched.",
            }
        ],
        "history": {
            "mica": [{"time": 0.5, "energy": 0.9, "activity": 0.2}]
        },
    }
    checkpoint = {
        "format": "chreatures-checkpoint-v1",
        "sha256": hashlib.sha256(observe._canonical(state)).hexdigest(),
        "state": state,
    }
    source = tmp_path / "residents.json"
    source.write_text(json.dumps(checkpoint))
    output = tmp_path / "observatory"

    def fake_weave(journal, habitat_id, fit, output_dir, evidence_parent=None):
        assert journal == state["journal"]
        assert habitat_id == state["id"]
        assert fit["status"] == "skipped"
        return {"status": "complete", "node_count": 1}

    monkeypatch.setattr(observe, "export_weave", fake_weave)
    report = observe.observe(source, output)

    assert report["gamfit"]["status"] == "skipped"
    assert report["weave"] == {"status": "complete", "node_count": 1}
    assert not (output / "habitat_activity.gam").exists()
    assert report["provenance"]["telemetry_excerpt"]["rows"] == 1
    assert json.loads((output / "observatory_report.json").read_text()) == report
