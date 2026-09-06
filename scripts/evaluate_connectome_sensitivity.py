#!/usr/bin/env python3
"""Evaluate one learned policy through canonical and matched-rewired MaleCNS graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.learning import PredictivePPOTrainer
from chreatures.malecns import MaleCNSGraph
from chreatures.neural_ports import NeuralPortBundle
from scripts.learn_affordances import FixedCohortBrain, evaluate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-graph", type=Path, required=True)
    parser.add_argument("--control-graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--learner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--macro-steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20350906)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if not 1 <= args.worlds <= 16 or args.steps % args.macro_steps:
        raise SystemExit("worlds must be in 1..16 and steps divisible by macro steps")
    canonical = MaleCNSGraph.load(args.canonical_graph, mmap=True)
    control = MaleCNSGraph.load(args.control_graph, mmap=True)
    if canonical.n != control.n or not np.array_equal(canonical.ids, control.ids):
        raise SystemExit("control graph does not preserve canonical neuron ordering")
    ports = NeuralPortBundle.load(args.port_bundle, canonical)
    trainer, _ = PredictivePPOTrainer.restore(args.learner_checkpoint, device=args.device)
    if trainer.config.feature_dim != len(ports.readout_names):
        raise SystemExit("learner and port feature dimensions differ")
    started = time.perf_counter()
    results = {}
    brain_metadata = {}
    for name, graph in (("canonical", canonical), ("matched_rewire", control)):
        brain = FixedCohortBrain(
            graph, ports, args.worlds * 3, device=args.device,
            backend="triton", microbatch_size=3,
        )
        results[name] = evaluate(
            brain, ports, args.genome, trainer.moments,
            worlds=args.worlds, steps=args.steps, macro_steps=args.macro_steps,
            workers=args.workers, seed=args.seed, silence_features=False,
        )
        brain_metadata[name] = brain.metadata()
    canonical_result = results["canonical"]
    control_result = results["matched_rewire"]
    differences = {
        key: float(canonical_result[key] - control_result[key])
        for key in canonical_result
        if isinstance(canonical_result[key], (int, float))
    }
    differences["homeostasis_mean_per_step"] = {
        key: float(
            canonical_result["homeostasis_mean_per_step"][key]
            - control_result["homeostasis_mean_per_step"][key]
        )
        for key in canonical_result["homeostasis_mean_per_step"]
    }
    record = {
        "format": "chreatures-connectome-sensitivity-v1",
        "interpretation": (
            "Same frozen learned policy, normalization, rich ports, held-out layouts and "
            "deterministic actions; only canonical versus degree-matched recurrent topology changes."
        ),
        "command": shlex.join([sys.executable, *sys.argv]),
        "elapsed_seconds": time.perf_counter() - started,
        "worlds": args.worlds, "residents": args.worlds * 3,
        "steps": args.steps, "seed": args.seed,
        "canonical_graph_sha256": canonical.hash,
        "control_graph_sha256": control.hash,
        "port_bundle_sha256": sha256(args.port_bundle),
        "genome_sha256": sha256(args.genome),
        "learner_checkpoint_sha256": sha256(args.learner_checkpoint),
        "source_sha256": {
            "scripts/evaluate_connectome_sensitivity.py": sha256(Path(__file__)),
            "scripts/learn_affordances.py": sha256(ROOT / "scripts/learn_affordances.py"),
            "chreatures/fast_circuit.py": sha256(ROOT / "chreatures/fast_circuit.py"),
        },
        "results": results,
        "canonical_minus_matched_rewire": differences,
        "brain": brain_metadata,
        "environment": {
            name: os.environ[name]
            for name in ("HSA_OVERRIDE_GFX_VERSION", "TRITON_CACHE_DIR")
            if name in os.environ
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
