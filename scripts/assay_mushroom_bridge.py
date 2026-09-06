#!/usr/bin/env python3
"""Build and assay the γ1pedc bridge against the canonical full graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.malecns import MaleCNSGraph  # noqa: E402
from chreatures.mushroom_plasticity import (  # noqa: E402
    MushroomBodySubstrate,
    MushroomFullGraphBridgeSpec,
)


RESIDENTS = ("pair-cue-a", "pair-cue-b", "no-modulation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_with_receipt(
    path: Path, loader: Any
) -> MushroomBodySubstrate | MushroomFullGraphBridgeSpec:
    expected = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return loader(path, expected_sha256=expected["sha256"])


def build_bridge(args: argparse.Namespace) -> None:
    graph = MaleCNSGraph.load(args.graph, mmap=True)
    substrate = load_with_receipt(args.substrate, MushroomBodySubstrate.load)
    bridge = MushroomFullGraphBridgeSpec.from_graph(graph, substrate)
    bridge_receipt = bridge.save(args.bridge)
    matrix = graph.matrix(normalized=True, signed=True, dtype=np.float32)
    observed = []
    for target_position, target in enumerate(bridge.target_neuron_indices):
        start, stop = int(matrix.indptr[target]), int(matrix.indptr[target + 1])
        row_sources = matrix.indices[start:stop]
        selected_sources = substrate.edge_source_neuron_indices[
            substrate.edge_target_positions == target_position
        ]
        positions = np.searchsorted(row_sources, selected_sources)
        if not np.array_equal(row_sources[positions], selected_sources):
            raise ValueError("bridge source is absent from the canonical recurrent row")
        observed.append(matrix.data[start:stop][positions])
    observed_weights = np.concatenate(observed)
    validation = {
        "format": "chreatures-mushroom-bridge-build-validation-v1",
        "generated_unix": time.time(),
        "graph_sha256": graph.hash,
        "substrate_sha256": substrate.substrate_hash,
        "bridge": bridge_receipt,
        "selected_neurons": bridge.selected_count,
        "edges": bridge.edge_count,
        "target_neuron_indices": bridge.target_neuron_indices.tolist(),
        "target_body_ids": bridge.target_body_ids.tolist(),
        "target_row_synapses": bridge.target_row_synapses.tolist(),
        "weights_max_abs_delta_vs_canonical_matrix": float(
            np.max(np.abs(observed_weights - bridge.full_row_weights), initial=0)
        ),
        "weights_bit_exact_vs_canonical_matrix": bool(
            np.array_equal(observed_weights, bridge.full_row_weights)
        ),
        "normalization": bridge.metadata["normalization"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "bridge-build-validation.json"
    output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt(output), indent=2, sort_keys=True))
    if not validation["weights_bit_exact_vs_canonical_matrix"]:
        raise SystemExit("bridge coefficients differ from canonical recurrence")


def cue_definitions() -> dict[str, dict[str, float]]:
    return {
        "A": {
            "odor/L/0": 1.0,
            "odor/L/1": 0.35,
            "retina/e02/a03/blue": 0.8,
            "retina/e02/a04/blue": 0.8,
        },
        "B": {
            "odor/R/2": 1.0,
            "odor/R/1": 0.35,
            "retina/e02/a11/red": 0.8,
            "retina/e02/a12/red": 0.8,
        },
    }


def request(
    circuit: MetalCircuit,
    senses: dict[str, float],
    pulses: dict[str, float] | None = None,
    *,
    dt: float,
) -> dict[str, dict[str, Any]]:
    pulses = pulses or {}
    rows = [
        {
            "id": resident,
            "senses": senses,
            "mushroom_modulator": float(pulses.get(resident, 0.0)),
        }
        for resident in RESIDENTS
    ]
    return {item["id"]: item for item in circuit.step(rows, dt)}


def run_history(
    circuit: MetalCircuit,
    cues: dict[str, dict[str, float]],
    *,
    dt: float,
    trials: int,
    cue_steps: int,
    pulse_steps: int,
    washout_steps: int,
) -> dict[str, float]:
    maximum_correction = 0.0

    def observe(result: dict[str, dict[str, Any]]) -> None:
        nonlocal maximum_correction
        maximum_correction = max(
            maximum_correction,
            *(
                abs(value)
                for item in result.values()
                for value in item["mushroom"]["target_recurrent_correction"]
            ),
        )

    for trial in range(trials):
        order = ("A", "B") if trial % 2 == 0 else ("B", "A")
        for cue_name in order:
            for _ in range(cue_steps):
                result = request(circuit, cues[cue_name], dt=dt)
                observe(result)
            pulses = {
                "pair-cue-a": 1.0 if cue_name == "A" else 0.0,
                "pair-cue-b": 1.0 if cue_name == "B" else 0.0,
            }
            for _ in range(pulse_steps):
                observe(request(circuit, {}, pulses, dt=dt))
            for _ in range(washout_steps):
                observe(request(circuit, {}, dt=dt))
    return {"maximum_abs_recurrent_correction": maximum_correction}


def summarize_result(item: dict[str, Any], kc_count: int) -> dict[str, Any]:
    selected = np.asarray(item["selected_rates"], dtype=np.float32)
    kc = selected[:kc_count]
    return {
        "mbon_rates": [float(value) for value in item["mushroom"]["actual_mbon_rates"]],
        "dan_rates": [float(value) for value in item["mushroom"]["actual_dan_rates"]],
        "target_recurrent_correction": [
            float(value)
            for value in item["mushroom"]["target_recurrent_correction"]
        ],
        "kc_rate_mean": float(kc.mean()),
        "kc_rate_maximum": float(kc.max(initial=0)),
        "active_kcs_above_0_01": int(np.count_nonzero(kc > 0.01)),
        "kc_rates_sha256": hashlib.sha256(kc.tobytes()).hexdigest(),
    }


def evaluate_from_snapshot(
    circuit: MetalCircuit,
    directory: Path,
    name: str,
    snapshot_receipt: dict[str, Any],
    cues: dict[str, dict[str, float]],
    *,
    dt: float,
    steps: int,
    kc_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result: dict[str, Any] = {}
    kc_rates: dict[str, dict[str, np.ndarray]] = {}
    for cue_name, senses in cues.items():
        circuit.restore(directory, name, snapshot_receipt["sha256"])
        observed = None
        for _ in range(steps):
            observed = request(circuit, senses, dt=dt)
        result[cue_name] = {
            resident: summarize_result(item, kc_count)
            for resident, item in observed.items()
        }
        kc_rates[cue_name] = {
            resident: np.asarray(item["selected_rates"][:kc_count], dtype=np.float32)
            for resident, item in observed.items()
        }
    separation: dict[str, Any] = {}
    for resident in RESIDENTS:
        left = kc_rates["A"][resident]
        right = kc_rates["B"][resident]
        difference = left - right
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        separation[resident] = {
            "mean_abs_delta": float(np.mean(np.abs(difference))),
            "rms_delta": float(np.sqrt(np.mean(np.square(difference)))),
            "cosine_similarity": (
                float(np.dot(left, right) / denominator) if denominator else 1.0
            ),
        }
    return result, separation


def max_result_delta(first: dict[str, Any], second: dict[str, Any]) -> float:
    fields = ("features", "selected_rates")
    delta = 0.0
    for resident in RESIDENTS:
        for field in fields:
            left = np.asarray(first[resident][field], dtype=np.float32)
            right = np.asarray(second[resident][field], dtype=np.float32)
            delta = max(delta, float(np.max(np.abs(left - right), initial=0)))
        left = np.asarray(
            first[resident]["mushroom"]["target_recurrent_correction"],
            dtype=np.float32,
        )
        right = np.asarray(
            second[resident]["mushroom"]["target_recurrent_correction"],
            dtype=np.float32,
        )
        delta = max(delta, float(np.max(np.abs(left - right), initial=0)))
    return delta


def response_changes(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cue in pre:
        result[cue] = {}
        for resident in RESIDENTS:
            before = np.asarray(pre[cue][resident]["mbon_rates"], dtype=np.float64)
            after = np.asarray(post[cue][resident]["mbon_rates"], dtype=np.float64)
            result[cue][resident] = {
                "pre": before.tolist(),
                "post": after.tolist(),
                "delta": (after - before).tolist(),
            }
    return result


def paired_contrasts(post: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cue, paired_resident in (("A", "pair-cue-a"), ("B", "pair-cue-b")):
        paired = np.asarray(post[cue][paired_resident]["mbon_rates"], dtype=np.float64)
        control = np.asarray(
            post[cue]["no-modulation"]["mbon_rates"], dtype=np.float64
        )
        result[cue] = {
            "paired_resident": paired_resident,
            "paired_minus_no_modulation": (paired - control).tolist(),
        }
        other = "pair-cue-b" if paired_resident == "pair-cue-a" else "pair-cue-a"
        cross_history = np.asarray(post[cue][other]["mbon_rates"], dtype=np.float64)
        result[cue]["paired_minus_other_history"] = (
            paired - cross_history
        ).tolist()
    return result


def run_circuit(
    args: argparse.Namespace,
    substrate: MushroomBodySubstrate,
    bridge: MushroomFullGraphBridgeSpec,
    *,
    frozen: bool,
) -> dict[str, Any]:
    from chreatures.metal_circuit import MetalCircuit

    label = "frozen" if frozen else "plastic"
    directory = args.output / "snapshots"
    with MetalCircuit(
        args.artifact,
        args.port_bundle,
        kernel=args.kernel,
        mushroom_substrate=substrate,
        mushroom_bridge=bridge,
        mushroom_plasticity_enabled=not frozen,
        mushroom_modulator_mode="synthetic",
    ) as circuit:
        circuit.add_residents(RESIDENTS)
        for _ in range(args.warmup_steps):
            request(circuit, {}, dt=args.dt)
        baseline_name = f"{label}-baseline"
        baseline_snapshot = circuit.snapshot(directory, baseline_name)
        cues = cue_definitions()
        pre, pre_cue_separation = evaluate_from_snapshot(
            circuit,
            directory,
            baseline_name,
            baseline_snapshot,
            cues,
            dt=args.dt,
            steps=args.evaluation_steps,
            kc_count=substrate.kc_count,
        )
        circuit.restore(directory, baseline_name, baseline_snapshot["sha256"])
        history = run_history(
            circuit,
            cues,
            dt=args.dt,
            trials=args.trials,
            cue_steps=args.cue_steps,
            pulse_steps=args.pulse_steps,
            washout_steps=args.washout_steps,
        )
        trained_name = f"{label}-trained"
        trained_snapshot = circuit.snapshot(directory, trained_name)
        post, post_cue_separation = evaluate_from_snapshot(
            circuit,
            directory,
            trained_name,
            trained_snapshot,
            cues,
            dt=args.dt,
            steps=args.evaluation_steps,
            kc_count=substrate.kc_count,
        )
        circuit.restore(directory, trained_name, trained_snapshot["sha256"])
        first = request(circuit, cues["A"], dt=args.dt)
        circuit.restore(directory, trained_name, trained_snapshot["sha256"])
        replay = request(circuit, cues["A"], dt=args.dt)
        return {
            "mode": label,
            "metadata": circuit.metadata()["research_mode"],
            "baseline_snapshot": baseline_snapshot,
            "trained_snapshot": trained_snapshot,
            "history": history,
            "pre": pre,
            "post": post,
            "pre_kc_cue_separation": pre_cue_separation,
            "post_kc_cue_separation": post_cue_separation,
            "response_changes": response_changes(pre, post),
            "paired_contrasts": paired_contrasts(post),
            "snapshot_replay_max_abs_delta": max_result_delta(first, replay),
        }


def assay(args: argparse.Namespace) -> None:
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    substrate = load_with_receipt(args.substrate, MushroomBodySubstrate.load)
    bridge = load_with_receipt(args.bridge, MushroomFullGraphBridgeSpec.load)
    if bridge.substrate_hash != substrate.substrate_hash:
        raise ValueError("bridge and substrate hashes differ")
    plastic = run_circuit(args, substrate, bridge, frozen=False)
    frozen = run_circuit(args, substrate, bridge, frozen=True)
    frozen_correction = max(
        abs(value)
        for cue in frozen["post"].values()
        for resident in cue.values()
        for value in resident["target_recurrent_correction"]
    )
    report = {
        "format": "chreatures-mushroom-fullgraph-assay-v1",
        "generated_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "graph_sha256": bridge.graph_hash,
        "substrate_sha256": substrate.substrate_hash,
        "bridge_sha256": bridge.bridge_hash,
        "artifact_sha256": sha256(args.artifact),
        "port_bundle_sha256": sha256(args.port_bundle),
        "parameters": {
            "dt": args.dt,
            "trials": args.trials,
            "cue_steps": args.cue_steps,
            "pulse_steps": args.pulse_steps,
            "washout_steps": args.washout_steps,
            "warmup_steps": args.warmup_steps,
            "evaluation_steps": args.evaluation_steps,
            "kernel": args.kernel,
        },
        "cues": {
            "definition": "engineered retinal-v1 and bilateral odor port patterns; KC activity is read from the actual recurrent graph",
            "channels": cue_definitions(),
        },
        "integration": {
            "equation": "correction_j(t+1)=sum_e(full_row_weight_e * private_deviation_e * actual_KC_rate_source(e,t))",
            "native_placement": "gain * (original_recurrent + lagged_correction)",
            "lag_steps": 1,
            "modulator": "explicit synthetic dimensionless pulse; no food, reward, punishment, pleasure, or motor meaning",
        },
        "plastic": plastic,
        "frozen": frozen,
        "invariants": {
            "plastic_snapshot_replay_exact": plastic["snapshot_replay_max_abs_delta"] == 0,
            "frozen_snapshot_replay_exact": frozen["snapshot_replay_max_abs_delta"] == 0,
            "plastic_correction_nonzero": plastic["history"]["maximum_abs_recurrent_correction"] > 0,
            "frozen_correction_zero": frozen_correction == 0,
        },
        "interpretation_limit": "The assay tests causal state transfer through measured full-graph rates and original normalized edges. Sensory cue definitions, the eligibility rule, constants, and synthetic modulator are engineered; the result does not assert DAN firing or behavioral valence.",
        "source_sha256": {
            "chreatures/mushroom_plasticity.py": sha256(
                ROOT / "chreatures/mushroom_plasticity.py"
            ),
            "scripts/assay_mushroom_bridge.py": sha256(Path(__file__).resolve()),
        },
    }
    output = args.output / "assay.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**receipt(output), "invariants": report["invariants"]}, indent=2, sort_keys=True))
    if not all(report["invariants"].values()):
        raise SystemExit("mushroom full-graph assay invariant failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-bridge", action="store_true")
    parser.add_argument(
        "--graph", type=Path, default=Path("/tank/chreatures/data/malecns/derived")
    )
    parser.add_argument(
        "--substrate",
        type=Path,
        default=ROOT / "data/mushroom/gamma1pedc-kc-mbon11-v1.npz",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=ROOT / "data/mushroom/gamma1pedc-fullgraph-bridge-v1.npz",
    )
    parser.add_argument(
        "--artifact", type=Path, default=ROOT / "data/metal-brain/metal-csr-v2.bin"
    )
    parser.add_argument(
        "--port-bundle", type=Path, default=ROOT / "data/ports/retinal-v1-maps.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/mushroom-fullgraph-assay"
    )
    parser.add_argument("--kernel", choices=("row", "simd"), default="simd")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--cue-steps", type=int, default=6)
    parser.add_argument("--pulse-steps", type=int, default=2)
    parser.add_argument("--washout-steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--evaluation-steps", type=int, default=8)
    args = parser.parse_args()
    if args.build_bridge:
        build_bridge(args)
    else:
        assay(args)


if __name__ == "__main__":
    main()
