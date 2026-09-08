#!/usr/bin/env python3
"""Render authenticated physical poses beside full-CNS observer state."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(encoded: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded, validate=True), dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--body-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tick", type=int, default=10)
    arguments = parser.parse_args()
    reports = sorted(arguments.capture.glob("heldout-10--*.json"))
    if len(reports) != 3:
        raise ValueError("observer rendering requires the exact three heldout-10 arms")
    rows = []
    all_delta = []
    for path in reports:
        report = json.loads(path.read_text())
        observer = report["observer_capture"]
        frame = next(item for item in observer["frames"] if item["tick"] == arguments.tick)
        positions = decode(observer["soma_positions_f32_base64"], "<f4").reshape(165122, 3)
        valid = decode(observer["soma_valid_u8_base64"], "u1").astype(bool)
        baseline = decode(observer["baseline_rate_f32_base64"], "<f4")
        rate = decode(frame["rate_state_f32_base64"], "<f4")
        if positions.shape != (165122, 3) or valid.shape != (165122,) or rate.shape != baseline.shape:
            raise ValueError("full-CNS observer tensor shape differs")
        delta = rate - baseline
        all_delta.append(np.abs(delta[valid]))
        body = arguments.body_directory / f"{report['arm']}-tick{arguments.tick}-body.jpg"
        rows.append((path, report, body, positions, valid, delta))
    color_limit = float(np.quantile(np.concatenate(all_delta), 0.995))
    if not np.isfinite(color_limit) or color_limit <= 0:
        raise ValueError("CNS display range is empty")

    labels = {
        "initialized-cns_initialized-private": "Initialized CNS + initialized private context",
        "trained-cns_zero-context": "Trained CNS + exact zero context",
        "trained-cns_learned-private": "Trained CNS + learned private context",
    }
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    scatter = None
    for row, (path, report, body, positions, valid, delta) in enumerate(rows):
        axes[row, 0].imshow(plt.imread(body))
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{labels[report['arm']]} — articulated world pose at 0.11 s", fontsize=10)
        points = positions[valid]
        scatter = axes[row, 1].scatter(points[:, 0], points[:, 1], c=delta[valid], s=.18,
            cmap="coolwarm", vmin=-color_limit, vmax=color_limit, linewidths=0, rasterized=True)
        axes[row, 1].set_aspect("equal")
        axes[row, 1].set_xlabel("atlas x")
        axes[row, 1].set_ylabel("atlas y")
        axes[row, 1].set_title("All valid soma: model rate state minus artifact baseline", fontsize=10)
    figure.colorbar(scatter, ax=axes[:, 1], shrink=.72,
        label=f"dimensionless model rate-state delta (display clipped at pooled 99.5% |delta| = {color_limit:.4g})")
    figure.suptitle("MaleCNS V4 physical instability diagnosis — heldout nursery 10, resident 0", fontsize=15)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    plt.close(figure)
    receipt = {
        "format": "chreatures-fly-cns-v4-assessment-observer-v1",
        "scope": "Observer rendering of executed Dawn/MuJoCo states; model rate state is dimensionless and is not measured firing.",
        "tick": arguments.tick,
        "model_time_s": (arguments.tick + 1) * .01,
        "resident": 0,
        "neural_field": "rate state minus immutable artifact baseline",
        "display_clip": {"method": "pooled valid-soma absolute 99.5 percentile", "limit": color_limit},
        "condition_receipts": {path.name: sha256(path) for path, *_ in rows},
        "body_renders": {body.name: sha256(body) for _, _, body, *_ in rows},
        "output_sha256": sha256(arguments.output),
        "renderer_sha256": sha256(Path(__file__)),
    }
    arguments.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
