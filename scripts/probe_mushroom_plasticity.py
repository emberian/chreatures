#!/usr/bin/env python3
"""Build and probe a measured gamma1pedc KC-to-MBON plasticity substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.malecns import MaleCNSGraph  # noqa: E402
from chreatures.mushroom_plasticity import (  # noqa: E402
    MushroomBodySubstrate,
    MushroomPlasticity,
    PlasticityConfig,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in value]


def max_state_delta(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    return max(
        float(
            np.max(
                np.abs(np.asarray(left[name], dtype=np.float64) - np.asarray(right[name], dtype=np.float64)),
                initial=0,
            )
        )
        for name in left
    )


def train_history(
    substrate: MushroomBodySubstrate,
    cues: dict[str, np.ndarray],
    paired: str,
    *,
    trials: int,
    config: PlasticityConfig,
) -> tuple[MushroomPlasticity, dict[str, Any]]:
    model = MushroomPlasticity(substrate, config=config)
    frozen = MushroomPlasticity(substrate, config=config, plasticity_enabled=False)
    zero = np.zeros(substrate.kc_count, dtype=np.float64)
    pre = {name: model.response(cue) for name, cue in cues.items()}
    frozen_pre = {name: frozen.response(cue) for name, cue in cues.items()}
    for trial in range(trials):
        order = ("A", "B") if trial % 2 == 0 else ("B", "A")
        for name in order:
            for candidate in (model, frozen):
                candidate.step(cues[name], 0.0, dt=0.2)
                candidate.step(zero, 1.0 if name == paired else 0.0, dt=0.1)
                candidate.washout(4.0)
    post = {name: model.response(cue) for name, cue in cues.items()}
    frozen_post = {name: frozen.response(cue) for name, cue in cues.items()}
    retention = {
        name: np.divide(
            post[name], pre[name], out=np.ones_like(post[name]), where=pre[name] > 0
        )
        for name in cues
    }
    frozen_delta = max(
        float(np.max(np.abs(frozen_post[name] - frozen_pre[name]), initial=0))
        for name in cues
    )
    return model, {
        "paired_cue": paired,
        "pre_response": {name: vector(value) for name, value in pre.items()},
        "post_response": {name: vector(value) for name, value in post.items()},
        "retention": {name: vector(value) for name, value in retention.items()},
        "mean_retention": {
            name: float(np.mean(value)) for name, value in retention.items()
        },
        "frozen_max_abs_response_delta": frozen_delta,
        "learned_edges": int(np.count_nonzero(model.efficacy_deviation)),
        "efficacy_deviation": {
            "minimum": float(model.efficacy_deviation.min()),
            "mean": float(model.efficacy_deviation.mean()),
            "maximum": float(model.efficacy_deviation.max()),
        },
    }


def uniform_control(
    cues: dict[str, np.ndarray], paired: str, *, trials: int, config: PlasticityConfig
) -> dict[str, Any]:
    """Symbolic two-cue lookup with no connectome identity or edge structure."""
    baseline = {name: float(np.mean(cue)) for name, cue in cues.items()}
    deviations = {name: 0.0 for name in cues}
    decrement = (
        config.depression_rate
        * 0.1
        * np.exp(-0.1 / config.eligibility_tau)
    )
    deviations[paired] = max(-config.maximum_depression, -trials * decrement)
    post = {name: baseline[name] * (1 + deviations[name]) for name in cues}
    return {
        "definition": "two symbolic cue-to-output weights with equal initial efficacy; no neuron IDs, edges, or synapse counts",
        "paired_cue": paired,
        "pre_response": baseline,
        "post_response": post,
        "retention": {
            name: post[name] / baseline[name] if baseline[name] else 1.0
            for name in cues
        },
    }


def snapshot_replay(
    model: MushroomPlasticity,
    cue: np.ndarray,
    snapshot_path: Path,
) -> dict[str, Any]:
    receipt = model.save_snapshot(snapshot_path)
    zero = np.zeros(model.substrate.kc_count, dtype=np.float64)
    first_cue = model.step(cue, 0.0, dt=0.2)
    first_modulator = model.step(zero, 1.0, dt=0.1)
    first_state = model.export_state()
    restored = MushroomPlasticity.load_snapshot(
        snapshot_path,
        model.substrate,
        expected_sha256=receipt["sha256"],
    )
    replay_cue = restored.step(cue, 0.0, dt=0.2)
    replay_modulator = restored.step(zero, 1.0, dt=0.1)
    replay_state = restored.export_state()
    return {
        "receipt": receipt,
        "state_max_abs_delta": max_state_delta(first_state, replay_state),
        "cue_response_max_abs_delta": float(
            np.max(np.abs(first_cue.response - replay_cue.response), initial=0)
        ),
        "modulator_response_max_abs_delta": float(
            np.max(
                np.abs(first_modulator.response - replay_modulator.response), initial=0
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=Path("/tank/chreatures/data/malecns/derived")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tank/chreatures/runs/mushroom-plasticity/gamma1pedc-v1"),
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--cue-size", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.trials < 1 or args.cue_size < 1:
        parser.error("trials and cue size must be positive")

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    graph = MaleCNSGraph.load(args.graph, mmap=True)
    substrate = MushroomBodySubstrate.from_graph(graph)
    if args.cue_size * 2 > substrate.kc_count:
        parser.error("two disjoint cues exceed connected KC population")
    bundle_path = args.output / "gamma1pedc-kc-mbon11-v1.npz"
    bundle_receipt = substrate.save(bundle_path)
    baseline_counts = substrate.synapse_counts.copy()
    baseline_weights = substrate.baseline_weights.copy()

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(substrate.kc_count, size=args.cue_size * 2, replace=False)
    cues = {
        "A": np.zeros(substrate.kc_count, dtype=np.float64),
        "B": np.zeros(substrate.kc_count, dtype=np.float64),
    }
    cues["A"][chosen[: args.cue_size]] = 1.0
    cues["B"][chosen[args.cue_size :]] = 1.0
    config = PlasticityConfig()
    histories: dict[str, Any] = {}
    models: dict[str, MushroomPlasticity] = {}
    for paired in ("A", "B"):
        models[paired], histories[f"pair_{paired}"] = train_history(
            substrate, cues, paired, trials=args.trials, config=config
        )
    controls = {
        f"pair_{paired}": uniform_control(
            cues, paired, trials=args.trials, config=config
        )
        for paired in ("A", "B")
    }
    replay = snapshot_replay(
        models["A"], cues["A"], args.output / "pair-A-state.npz"
    )
    immutable_counts = np.array_equal(baseline_counts, substrate.synapse_counts)
    immutable_weights = np.array_equal(baseline_weights, substrate.baseline_weights)
    crossover = all(
        histories[f"pair_{paired}"]["mean_retention"][paired]
        < histories[f"pair_{paired}"]["mean_retention"]["B" if paired == "A" else "A"]
        for paired in ("A", "B")
    )
    frozen_exact = all(
        histories[name]["frozen_max_abs_response_delta"] == 0 for name in histories
    )
    snapshot_exact = max(
        replay["state_max_abs_delta"],
        replay["cue_response_max_abs_delta"],
        replay["modulator_response_max_abs_delta"],
    ) == 0
    report = {
        "format": "chreatures-mushroom-plasticity-probe-v1",
        "generated_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "graph_sha256": graph.hash,
        "substrate": substrate.metadata,
        "bundle": bundle_receipt,
        "plasticity": {
            "config": asdict(config),
            "rule": "engineered cue-before-modulator eligibility-gated depression",
            "modulator_semantics": "synthetic dimensionless experimental pulse; no reward, punishment, food, or motor meaning",
        },
        "cues": {
            "definition": "synthetic disjoint sparse masks over connected KC body IDs",
            "seed": args.seed,
            "size": args.cue_size,
            "overlap": int(np.count_nonzero(cues["A"] * cues["B"])),
            "body_ids": {
                name: [int(value) for value in substrate.kc_body_ids[cue > 0]]
                for name, cue in cues.items()
            },
        },
        "trials": args.trials,
        "histories": histories,
        "nonconnectomic_uniform_control": controls,
        "snapshot_replay": replay,
        "invariants": {
            "raw_synapse_counts_immutable": immutable_counts,
            "normalized_baseline_immutable": immutable_weights,
            "counterbalanced_associative_crossover": crossover,
            "frozen_anatomy_response_exact": frozen_exact,
            "snapshot_replay_exact": snapshot_exact,
        },
        "literature": [
            {
                "citation": "Hige et al. (2015), Heterosynaptic Plasticity Underlies Aversive Olfactory Learning in Drosophila",
                "doi": "10.1016/j.neuron.2015.11.003",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/",
                "used_for": "gamma1pedc compartment choice, forward timing, odor-specific KC-MBON depression",
            },
            {
                "citation": "Aso et al. (2014), The neuronal architecture of the mushroom body provides a logic for associative learning",
                "doi": "10.7554/eLife.04577",
                "url": "https://elifesciences.org/articles/04577",
                "used_for": "compartmental KC-MBON-DAN organization",
            },
        ],
        "source_sha256": {
            "chreatures/mushroom_plasticity.py": sha256(
                ROOT / "chreatures/mushroom_plasticity.py"
            ),
            "scripts/probe_mushroom_plasticity.py": sha256(Path(__file__).resolve()),
        },
        "interpretation_limit": "The probe establishes history-dependent state in a measured edge scaffold. It does not establish the engineered rule, constants, cue masks, or synthetic modulator as MaleCNS physiology, and does not show superiority over the uniform associative control.",
    }
    output_path = args.output / "probe.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt = {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
        "invariants": report["invariants"],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not all(report["invariants"].values()):
        raise SystemExit("mushroom plasticity probe invariant failed")


if __name__ == "__main__":
    main()
