#!/usr/bin/env python3
"""Fit, persist, and reload a native gamfit nonlinear regression.

The synthetic response has a known nonlinear law. Alternating points are held
out so the check measures interpolation rather than training reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


GAMFIT_VERSION = "0.1.259"
GAMFIT_SOURCE_COMMIT = "7c7eca8ac4826de95c8e743a20294bee132a9bcc"
SEED = 20260905


def response_law(x: np.ndarray) -> np.ndarray:
    """Noise-free synthetic law used to make the regression falsifiable."""
    return 0.65 * x**2 + np.sin(2.2 * x)


def make_dataset() -> tuple[list[dict[str, float]], list[dict[str, float]], np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = np.linspace(-3.0, 3.0, 180)
    y = response_law(x) + rng.normal(0.0, 0.08, x.size)
    test_mask = np.arange(x.size) % 3 == 0
    train = [
        {"x": float(xi), "y": float(yi)}
        for xi, yi in zip(x[~test_mask], y[~test_mask], strict=True)
    ]
    test = [
        {"x": float(xi), "y": float(yi)}
        for xi, yi in zip(x[test_mask], y[test_mask], strict=True)
    ]
    return train, test, y[~test_mask]


def _rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def run(output_dir: Path) -> dict[str, object]:
    try:
        import gamfit
    except ImportError as exc:
        raise SystemExit(
            "gamfit is absent; install the isolated integration dependencies with "
            "`uv venv integrations/.venv && uv pip install --python "
            "integrations/.venv/bin/python -r integrations/gamfit-requirements.txt`"
        ) from exc

    if gamfit.__version__ != GAMFIT_VERSION:
        raise RuntimeError(
            f"expected gamfit {GAMFIT_VERSION}, loaded {gamfit.__version__}"
        )
    build_info = gamfit.build_info()
    if not build_info.get("available"):
        raise RuntimeError(f"gamfit native extension unavailable: {build_info}")

    train, test, train_y = make_dataset()
    formula = "y ~ s(x, k=14)"
    gamfit.validate_formula(train, formula)

    started = time.perf_counter()
    model = gamfit.fit(train, formula)
    fit_seconds = time.perf_counter() - started

    test_x = [{"x": row["x"]} for row in test]
    observed = np.asarray([row["y"] for row in test], dtype=float)
    predicted = np.asarray(model.predict(test_x), dtype=float)
    null_prediction = np.full(observed.shape, float(np.mean(train_y)))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "nonlinear_response.gam"
    model.save(model_path)
    model_bytes = model_path.read_bytes()

    reloaded = gamfit.load(model_path)
    reloaded_prediction = np.asarray(reloaded.predict(test_x), dtype=float)
    max_reload_delta = float(np.max(np.abs(predicted - reloaded_prediction)))
    smooth_rmse = _rmse(predicted, observed)
    null_rmse = _rmse(null_prediction, observed)

    result: dict[str, object] = {
        "schema_version": 1,
        "integration": "native-gamfit-regression",
        "library": {
            "name": "gamfit",
            "version": gamfit.__version__,
            "source_commit": GAMFIT_SOURCE_COMMIT,
            "native_extension_available": bool(build_info["available"]),
            "native_crate": build_info.get("crate"),
            "native_engine_crate": build_info.get("engine_crate"),
            "abi": build_info.get("abi3"),
        },
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.system().lower(),
            "fit_seconds": fit_seconds,
        },
        "experiment": {
            "seed": SEED,
            "truth": "0.65*x**2 + sin(2.2*x)",
            "noise_standard_deviation": 0.08,
            "formula": formula,
            "train_rows": len(train),
            "held_out_rows": len(test),
            "split": "indices divisible by 3 held out",
            "null_model": "training-response mean",
        },
        "metrics": {
            "held_out_rmse": smooth_rmse,
            "null_rmse": null_rmse,
            "rmse_ratio": smooth_rmse / null_rmse,
        },
        "persistence": {
            "model_file": model_path.name,
            "bytes": len(model_bytes),
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
            "reload_max_abs_prediction_delta": max_reload_delta,
        },
    }

    if not math.isfinite(smooth_rmse) or smooth_rmse >= 0.2 * null_rmse:
        raise RuntimeError(f"nonlinear fit did not beat the declared null: {result['metrics']}")
    if max_reload_delta != 0.0:
        raise RuntimeError(f"persisted model predictions changed by {max_reload_delta}")

    result_path = output_dir / "regression_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("artifacts") / "gamfit",
        help="directory for the .gam model and JSON result",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

