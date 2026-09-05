#!/usr/bin/env python3
"""Project a real habitat checkpoint or telemetry CSV into the observatory.

Journal entries become native Universal Weave episode nodes. When enough
finite history rows exist, native gamfit describes activity as smooth functions
of energy and model time. A fit is descriptive evidence, not a causal claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from integrations.gamfit_regression import GAMFIT_SOURCE_COMMIT, GAMFIT_VERSION
except ModuleNotFoundError:  # Direct execution puts integrations/ on sys.path.
    from gamfit_regression import GAMFIT_SOURCE_COMMIT, GAMFIT_VERSION


ROOT = Path(__file__).parents[1]
WEAVE_MANIFEST = Path(__file__).with_name("weave") / "Cargo.toml"
WEAVE_SOURCE_COMMIT = "7a5a0dabb94885e44ad8a6c4355c015d7f38020f"
MIN_FIT_ROWS = 60
FORMULA = "activity ~ s(energy, k=8) + s(time, k=8)"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _flatten_history(history: Any) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    if isinstance(history, dict):
        for resident_id, values in history.items():
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, dict):
                        candidates.append({"resident_id": str(resident_id), **value})
    elif isinstance(history, list):
        candidates.extend(value for value in history if isinstance(value, dict))

    rows = []
    for candidate in candidates:
        time_value = _number(candidate.get("time"))
        energy = _number(candidate.get("energy"))
        activity = _number(candidate.get("activity"))
        if time_value is None or energy is None or activity is None:
            continue
        row = dict(candidate)
        row.update({"time": time_value, "energy": energy, "activity": activity})
        row["resident_id"] = str(candidate.get("resident_id", "unknown"))
        rows.append(row)
    return rows, len(candidates) - len(rows)


def load_records(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    provenance: dict[str, Any] = {
        "input_file": str(path.resolve()),
        "input_bytes": len(raw),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
    }

    if path.suffix.lower() == ".csv":
        with path.open(newline="") as file:
            history = list(csv.DictReader(file))
        state: dict[str, Any] = {
            "id": path.stem,
            "journal": [],
            "history": history,
        }
        provenance["input_format"] = "telemetry-csv"
    else:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("JSON input must be an object")
        if payload.get("format") == "chreatures-checkpoint-v1":
            state = payload.get("state")
            if not isinstance(state, dict):
                raise ValueError("checkpoint state must be an object")
            actual = hashlib.sha256(_canonical(state)).hexdigest()
            if actual != payload.get("sha256"):
                raise ValueError("checkpoint checksum does not match")
            provenance.update(
                {
                    "input_format": "chreatures-checkpoint-v1",
                    "checkpoint_sha256": actual,
                    "checkpoint_verified": True,
                }
            )
        else:
            nested = payload.get("state")
            state = nested if isinstance(nested, dict) else payload
            provenance["input_format"] = "habitat-json"

    journal = state.get("journal", [])
    if not isinstance(journal, list) or not all(isinstance(row, dict) for row in journal):
        raise ValueError("habitat journal must be a list of objects")
    history = state.get("history", state.get("telemetry", state.get("records", [])))
    rows, rejected_rows = _flatten_history(history)
    provenance.update(
        {
            "habitat_id": state.get("id"),
            "journal_events": len(journal),
            "telemetry_candidates": len(rows) + rejected_rows,
            "telemetry_rows": len(rows),
            "rejected_telemetry_rows": rejected_rows,
            "resident_ids": sorted({row["resident_id"] for row in rows}),
            "time_range": (
                [min(row["time"] for row in rows), max(row["time"] for row in rows)]
                if rows
                else None
            ),
        }
    )
    return {"provenance": provenance, "journal": journal, "telemetry": rows}


def _fit_readiness(rows: list[dict[str, Any]]) -> str | None:
    if len(rows) < MIN_FIT_ROWS:
        return f"need at least {MIN_FIT_ROWS} finite telemetry rows; found {len(rows)}"
    for field, minimum_unique in (("time", 12), ("energy", 8), ("activity", 3)):
        count = len({row[field] for row in rows})
        if count < minimum_unique:
            return f"need at least {minimum_unique} distinct {field} values; found {count}"
    return None


def _capture_native_stderr(callback):
    """Capture diagnostics written directly to file descriptor 2 by Rust."""
    with tempfile.TemporaryFile() as capture:
        saved_stderr = os.dup(2)
        try:
            os.dup2(capture.fileno(), 2)
            result = callback()
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        capture.seek(0)
        messages = capture.read().decode(errors="replace").strip().splitlines()
    return result, messages


def freeze_telemetry(
    rows: list[dict[str, Any]], output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write and reread the exact compact table passed to native gamfit."""
    path = output_dir / "telemetry_used.csv"
    fields = ("resident_id", "time", "energy", "activity")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "resident_id": row["resident_id"],
                    "time": format(row["time"], ".10g"),
                    "energy": format(row["energy"], ".10g"),
                    "activity": format(row["activity"], ".10g"),
                }
            )
    with path.open(newline="") as file:
        frozen, rejected = _flatten_history(list(csv.DictReader(file)))
    if rejected or len(frozen) != len(rows):
        raise RuntimeError("telemetry excerpt did not round-trip")
    payload = path.read_bytes()
    return frozen, {
        "file": path.name,
        "rows": len(frozen),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "numeric_format": "10 significant decimal digits",
    }


def fit_activity(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    reason = _fit_readiness(rows)
    if reason:
        return {
            "status": "skipped",
            "reason": reason,
            "formula": FORMULA,
            "interpretation": "descriptive only; no causal claim",
        }

    try:
        import gamfit
    except ImportError:
        return {
            "status": "unavailable",
            "reason": f"adequate data found, but pinned gamfit {GAMFIT_VERSION} is not installed",
            "formula": FORMULA,
            "interpretation": "descriptive only; no causal claim",
        }
    if gamfit.__version__ != GAMFIT_VERSION:
        return {
            "status": "unavailable",
            "reason": f"expected gamfit {GAMFIT_VERSION}, loaded {gamfit.__version__}",
            "formula": FORMULA,
            "interpretation": "descriptive only; no causal claim",
        }
    build_info = gamfit.build_info()
    if not build_info.get("available"):
        return {
            "status": "unavailable",
            "reason": f"native extension unavailable: {build_info.get('reason', 'unknown')}",
            "formula": FORMULA,
            "interpretation": "descriptive only; no causal claim",
        }

    ordered = sorted(rows, key=lambda row: (row["time"], row["resident_id"]))
    held_out = np.arange(len(ordered)) % 5 == 0
    train = [row for index, row in enumerate(ordered) if not held_out[index]]
    test = [row for index, row in enumerate(ordered) if held_out[index]]
    try:
        gamfit.validate_formula(train, FORMULA)
        started = time.perf_counter()
        model, native_messages = _capture_native_stderr(
            lambda: gamfit.fit(train, FORMULA)
        )
        fit_seconds = time.perf_counter() - started
        predictors = [{"energy": row["energy"], "time": row["time"]} for row in test]
        observed = np.asarray([row["activity"] for row in test], dtype=float)
        predicted = np.asarray(model.predict(predictors), dtype=float)
        null = np.full(observed.shape, np.mean([row["activity"] for row in train]))

        model_path = output_dir / "habitat_activity.gam"
        model.save(model_path)
        model_bytes = model_path.read_bytes()
        reloaded = gamfit.load(model_path)
        reloaded_prediction = np.asarray(reloaded.predict(predictors), dtype=float)
        reload_delta = float(np.max(np.abs(predicted - reloaded_prediction)))
        if reload_delta != 0.0:
            raise RuntimeError(f"persisted prediction delta is {reload_delta}")
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "formula": FORMULA,
            "rows": len(rows),
            "interpretation": "descriptive only; no causal claim",
        }

    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    null_rmse = float(np.sqrt(np.mean((null - observed) ** 2)))
    return {
        "status": "complete",
        "formula": FORMULA,
        "interpretation": "descriptive only; no causal claim",
        "library": {
            "name": "gamfit",
            "version": gamfit.__version__,
            "source_commit": GAMFIT_SOURCE_COMMIT,
            "native_extension_available": True,
            "native_crate": build_info.get("crate"),
            "native_engine_crate": build_info.get("engine_crate"),
        },
        "data": {
            "rows": len(rows),
            "training_rows": len(train),
            "held_out_rows": len(test),
            "time_range": [ordered[0]["time"], ordered[-1]["time"]],
            "split": "every fifth row after ordering by model time and resident",
        },
        "metrics": {
            "held_out_rmse": rmse,
            "training_mean_null_rmse": null_rmse,
            "rmse_ratio": rmse / null_rmse if null_rmse else None,
        },
        "fit_seconds": fit_seconds,
        "native_messages": native_messages,
        "persistence": {
            "model_file": model_path.name,
            "bytes": len(model_bytes),
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
            "reload_max_abs_prediction_delta": reload_delta,
        },
    }


def export_weave(
    journal: list[dict[str, Any]],
    habitat_id: Any,
    fit: dict[str, Any],
    output_dir: Path,
    evidence_parent: str | None = None,
) -> dict[str, Any]:
    known_ids = {event.get("id") for event in journal}
    if evidence_parent is not None and evidence_parent not in known_ids:
        raise ValueError(f"evidence parent {evidence_parent!r} is absent from the journal")
    evidence = []
    if fit["status"] == "complete":
        evidence.append(
            {
                "id": "gamfit:habitat-activity:v1",
                "time": fit["data"]["time_range"][1],
                "text": (
                    f"Descriptive native GAM fit over {fit['data']['rows']} habitat telemetry rows; "
                    "this is not a causal claim."
                ),
                "artifact_uri": f"file:{fit['persistence']['model_file']}",
                "parent_ids": [evidence_parent] if evidence_parent else [],
            }
        )
    request = {
        "habitat_id": str(habitat_id) if habitat_id is not None else None,
        "journal": journal,
        "evidence": evidence,
    }
    request_path = output_dir / "weave_import_request.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    weave_path = output_dir / "habitat_journal.weave.json"

    if shutil.which("cargo") is None:
        return {
            "status": "unavailable",
            "reason": "cargo is required for native Universal Weave import",
            "source_commit": WEAVE_SOURCE_COMMIT,
        }
    with tempfile.TemporaryDirectory(prefix="chreatures-weave-build-") as target:
        command = [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(WEAVE_MANIFEST),
            "--",
            "--input",
            str(request_path),
            "--output",
            str(weave_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                env={**os.environ, "CARGO_TARGET_DIR": target},
                text=True,
                timeout=180,
            )
        except subprocess.CalledProcessError as exc:
            return {
                "status": "failed",
                "reason": (exc.stderr or str(exc)).strip()[-2000:],
                "source_commit": WEAVE_SOURCE_COMMIT,
                "request_file": request_path.name,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "reason": "native Universal Weave import exceeded 180 seconds",
                "source_commit": WEAVE_SOURCE_COMMIT,
                "request_file": request_path.name,
            }
    receipt = json.loads(completed.stdout)
    return {
        "status": "complete",
        "request_file": request_path.name,
        "artifact_file": weave_path.name,
        **receipt,
    }


def observe(
    input_path: Path, output_dir: Path, evidence_parent: str | None = None
) -> dict[str, Any]:
    records = load_records(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry, excerpt = freeze_telemetry(records["telemetry"], output_dir)
    records["provenance"]["telemetry_excerpt"] = excerpt
    fit = fit_activity(telemetry, output_dir)
    weave = export_weave(
        records["journal"],
        records["provenance"]["habitat_id"],
        fit,
        output_dir,
        evidence_parent,
    )
    report = {
        "schema_version": 1,
        "report_type": "chreatures-observatory",
        "provenance": records["provenance"],
        "gamfit": fit,
        "weave": weave,
        "limitations": [
            "The GAM is descriptive and does not establish a causal or biological mechanism.",
            "Bounded runtime history may omit earlier telemetry; the source checkpoint remains authoritative.",
            "Journal episodes are external evidence records, not the resident's internal memory.",
        ],
    }
    temporary = output_dir / "observatory_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_dir / "observatory_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=ROOT / "runs" / "residents.json",
        help="habitat checkpoint/JSON artifact or telemetry CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("artifacts") / "observatory",
    )
    parser.add_argument(
        "--evidence-parent",
        help="optional journal event id to make the GAM evidence node descend from",
    )
    args = parser.parse_args()
    print(json.dumps(observe(args.input, args.output_dir, args.evidence_parent), indent=2))


if __name__ == "__main__":
    main()
