#!/usr/bin/env python3
"""Export compact, auditable 25x25 surfaces from fitted native GAM models."""

from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(values: np.ndarray, transform: dict) -> np.ndarray:
    if transform["kind"] == "signed_tanh":
        return float(transform["magnitude"]) * np.tanh(values)
    if transform["kind"] == "positive_softplus":
        return np.minimum(np.logaddexp(0.0, values), float(transform["ceiling"]))
    raise ValueError("surface export supports signed_tanh and positive_softplus responses")


def main() -> None:
    import gamfit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank_path = args.fit / "population_response_bank.json"
    bank = json.loads(bank_path.read_text())
    axes = {"energy_state_delta": ("history_energy_mean", "thrust"),
            "fatigue_state_delta": ("history_fatigue_mean", "thrust"),
            "effort": ("fatigue", "thrust")}
    features = bank["fitted"]["features"]
    names = [feature["name"] for feature in features]
    laws = {law["name"]: law for law in bank["fitted"]["laws"]}
    rules = {rule["law"]: rule for rule in bank["responses"]}
    surfaces = []
    for law_name, (x_name, y_name) in axes.items():
        law, rule = laws[law_name], rules[law_name]
        xi, yi = names.index(x_name), names.index(y_name)
        xraw = np.linspace(features[xi]["minimum"], features[xi]["maximum"], 25)
        yraw = np.linspace(features[yi]["minimum"], features[yi]["maximum"], 25)
        rows = []
        for y in yraw:
            for x in xraw:
                raw = np.asarray([feature["mean"] for feature in features], dtype=float)
                raw[xi], raw[yi] = x, y
                rows.append({name: float((raw[i] - features[i]["mean"]) / features[i]["scale"])
                             for i, name in enumerate(names)})
        model_path = args.fit / f"{law_name}.gam"
        native_latent = np.asarray(gamfit.load(model_path).predict(rows), dtype=float)
        exported_latent = np.full(len(rows), float(law["intercept"]))
        for term in law["terms"]:
            feature = features[int(term["feature"])]
            coordinate = np.asarray([row[feature["name"]] for row in rows])
            exported_latent += np.interp(coordinate, term["knots"], term["values"])
        native = forward(native_latent, rule["transform"])
        exported = forward(exported_latent, rule["transform"])
        surfaces.append({"law": law_name, "unit": rule["unit"],
            "x": {"feature": x_name, "unit": features[xi]["unit"], "values": xraw.tolist()},
            "y": {"feature": y_name, "unit": features[yi]["unit"], "values": yraw.tolist()},
            "fixed_features": "training means", "shape": [25, 25],
            "native_gam_values_row_major": native.tolist(),
            "exported_rust_values_row_major": exported.tolist(),
            "whole_in_domain_grid_export_error": {
                "maximum_absolute": float(np.max(np.abs(native - exported))),
                "rmse": float(np.sqrt(np.mean((native - exported) ** 2)))},
            "model_sha256": digest(model_path)})
    artifact = {"format": "chreatures-population-gam-surfaces-v1",
        "interpretation": "conditional fitted associations with other inputs fixed at training means; not causal effects",
        "grid": "25x25 within the fitted 0.5%-99.5% training domain",
        "gamfit_version": gamfit.__version__, "bank_sha256": digest(bank_path), "surfaces": surfaces}
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "max_export_error": max(x["whole_in_domain_grid_export_error"]["maximum_absolute"] for x in surfaces)}, indent=2))


if __name__ == "__main__": main()
