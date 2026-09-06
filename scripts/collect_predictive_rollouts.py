#!/usr/bin/env python3
"""Collect anonymous continuous MaleCNS/body sequences for predictive learning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shlex
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if mp.current_process().name == "MainProcess":
    import mujoco
    import torch
    import triton
    from torch.distributions import Normal

    from chreatures.homeostasis import FiniteEnergyConfig, FiniteEnergyObjective
    from chreatures.learning import ACTIONS, PredictivePPOTrainer
    from chreatures.malecns import MaleCNSGraph
    from chreatures.neural_ports import NeuralPortBundle
    from chreatures.training_environment import EmbodiedTrainingProfile
    from scripts.learn_affordances import AffordanceCohort, FixedCohortBrain, sha256


PHYSIOLOGY_NAMES = (
    "energy", "gut", "fatigue", "tanh_speed_over_2",
    "tanh_angular_velocity_over_4", "neural_support",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--learner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--holdout-worlds", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=4_800)
    parser.add_argument("--macro-steps", type=int, default=5)
    parser.add_argument("--exploration-scale", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sample_action(
    trainer: PredictivePPOTrainer, features: np.ndarray,
    physiology: np.ndarray, exploration_scale: float,
) -> dict[str, np.ndarray]:
    """Sample the frozen policy, optionally widening only its Gaussian proposal."""
    with torch.no_grad():
        feature_tensor = torch.as_tensor(features, device=trainer.device)
        physiology_tensor = torch.as_tensor(physiology, device=trainer.device)
        context_tensor = torch.as_tensor(trainer.context, device=trainer.device)
        mean, value, hidden = trainer.model(
            feature_tensor, physiology_tensor, context_tensor
        )
        base = trainer.model.distribution(mean, hidden)
        distribution = Normal(mean, base.scale * exploration_scale)
        latent = distribution.sample()
        action = torch.tanh(latent)
        prediction = trainer.model.predictor(torch.cat((hidden, action), dim=-1))
        trainer.decision_count += 1
        return {
            "features": features.copy(), "physiology": physiology.copy(),
            "context": trainer.context.copy(), "latent": latent.cpu().numpy(),
            "action": action.cpu().numpy(),
            "log_prob": trainer.model.squashed_log_prob(distribution, latent).cpu().numpy(),
            "value": value.cpu().numpy(), "prediction": prediction.cpu().numpy(),
        }


def _open_arrays(directory: Path, rows: int, residents: int) -> dict[str, np.memmap]:
    shapes = {
        "features": (rows, residents, 384),
        "source_channels": (rows, residents, 351),
        "physiology": (rows, residents, 6),
        "action": (rows, residents, len(ACTIONS)),
        "reset": (rows, residents), "done": (rows, residents),
        "valid": (rows, residents), "behavior_mode": (rows, residents),
    }
    dtypes = {
        "features": np.float32, "source_channels": np.float32,
        "physiology": np.float32, "action": np.float32,
        "reset": np.bool_, "done": np.bool_, "valid": np.bool_,
        "behavior_mode": np.uint8,
    }
    return {
        name: np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=dtypes[name], shape=shape
        )
        for name, shape in shapes.items()
    }


def _flush(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def _file_receipts(directory: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(directory.glob("*.npy")):
        result[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


def main() -> int:
    args = arguments()
    if (
        args.worlds != 16 or not 0 < args.holdout_worlds < args.worlds
        or args.episodes < 1 or args.episode_steps % args.macro_steps
        or args.exploration_scale < 1.0
    ):
        raise SystemExit("invalid fixed-cohort collection configuration")
    args.output = args.output.resolve()
    if args.output.exists():
        raise SystemExit("dataset output already exists")
    args.output.mkdir(parents=True)

    graph = MaleCNSGraph.load(args.graph.resolve(), mmap=True)
    ports = NeuralPortBundle.load(args.port_bundle.resolve(), graph)
    if len(ports.input_names) != 351 or len(ports.readout_names) != 384:
        raise SystemExit("collector requires the body-v1 351-to-384 port bundle")
    profile = EmbodiedTrainingProfile.current_v2()
    brain = FixedCohortBrain(
        graph, ports, args.worlds * 3, device=args.device,
        backend="tiled", microbatch_size=3,
    )
    cohort = AffordanceCohort(
        brain, ports, args.worlds, args.worlds, args.seed,
        reward_objective=FiniteEnergyObjective(FiniteEnergyConfig.from_value(
            profile.component("homeostasis")
        )),
        training_profile=profile, physical_backend="fast",
        curriculum_start_stage=2,
    )
    trainer, _ = PredictivePPOTrainer.restore(
        args.learner.resolve(), device=args.device,
        expected_sha256=sha256(args.learner.resolve()),
    )
    if trainer.config.feature_dim != 384 or trainer.config.macro_steps != args.macro_steps:
        raise SystemExit("learner dimensions differ from collection layout")
    if trainer.resident_ids != cohort.resident_ids:
        raise SystemExit("learner cohort identities differ from fixed collection slots")
    trainer.model.eval()
    trainer.reset_private_state()
    moment_value = trainer.moments.snapshot()
    normalizer_bytes = json.dumps(
        moment_value, sort_keys=True, separators=(",", ":")
    ).encode()
    np.savez_compressed(
        args.output / "normalizer.npz", count=np.asarray(moment_value["count"]),
        mean=np.asarray(moment_value["mean"], dtype=np.float64),
        m2=np.asarray(moment_value["m2"], dtype=np.float64),
    )
    (args.output / "layouts.json").write_text(json.dumps({
        "feature_names": list(ports.readout_names),
        "source_channel_names": list(ports.input_names),
        "physiology_names": list(PHYSIOLOGY_NAMES), "action_names": list(ACTIONS),
        "behavior_modes": {"0": "frozen_policy", "1": "frozen_policy_std_scaled"},
    }, indent=2, sort_keys=True) + "\n")

    rows = args.episode_steps // args.macro_steps
    train_worlds = args.worlds - args.holdout_worlds
    holdout_residents = args.holdout_worlds * 3
    split_indices = {
        "holdout": np.arange(holdout_residents),
        "train": np.arange(holdout_residents, args.worlds * 3),
    }
    native_binaries = list(ROOT.glob("_world_kernels*.so"))
    if len(native_binaries) != 1:
        raise SystemExit("collector requires one project-local native world extension")
    native_binary = native_binaries[0]
    native_sources = [
        ROOT / "native/world-kernels" / name
        for name in ("Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs")
    ]
    manifest: dict[str, Any] = {
        "format": "chreatures-anonymous-predictive-sequences-v1",
        "completed": False, "started_unix": time.time(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "graph_sha256": graph.hash, "port_spec_sha256": ports.spec_hash,
        "port_bundle_sha256": sha256(args.port_bundle.resolve()),
        "learner": {"path": str(args.learner.resolve()),
                    "sha256": sha256(args.learner.resolve()),
                    "std_profile": trainer.config.std_profile,
                    "updates": trainer.update_count, "decisions": trainer.decision_count},
        "normalizer": {"sha256": hashlib.sha256(normalizer_bytes).hexdigest(),
                       "artifact_sha256": sha256(args.output / "normalizer.npz")},
        "profile": profile.to_value(), "curriculum_stage": 2,
        "physical_dt": 0.05, "macro_steps": args.macro_steps,
        "macro_dt": args.macro_steps * 0.05,
        "precision": "float32 exact runtime observations/actions; bool masks; uint8 mode",
        "layout": "episode-major, time-major, anonymous resident-major, channel-minor",
        "splits": {
            "train": {"whole_world_slots": [args.holdout_worlds, args.worlds],
                      "residents_per_episode": train_worlds * 3},
            "holdout": {"whole_world_slots": [0, args.holdout_worlds],
                        "residents_per_episode": args.holdout_worlds * 3},
        },
        "episodes_requested": args.episodes, "episode_steps": args.episode_steps,
        "rows_per_episode": rows, "resident_rows_total": rows * args.worlds * 3 * args.episodes,
        "world_seeds": {
            split: [[args.seed + episode * 1009 + world
                     for world in range(bounds[0], bounds[1])]
                    for episode in range(args.episodes)]
            for split, bounds in {
                "holdout": (0, args.holdout_worlds),
                "train": (args.holdout_worlds, args.worlds),
            }.items()
        },
        "world_seed_scope": "reproducibility metadata only; no seed or identity tensor",
        "alignment": (
            "row t stores observation[t] and the action actually executed through the next "
            "macro interval; each episode stores terminal observation[T] separately"
        ),
        "target_pairs_total": rows * args.worlds * 3 * args.episodes,
        "behavior_schedule": [
            {"episode": episode, "mode": "frozen_policy" if episode % 2 == 0
             else "frozen_policy_std_scaled",
             "std_scale": 1.0 if episode % 2 == 0 else args.exploration_scale}
            for episode in range(args.episodes)
        ],
        "episodes": [],
        "source_sha256": {
            "scripts/collect_predictive_rollouts.py": sha256(Path(__file__).resolve()),
            "scripts/learn_affordances.py": sha256(ROOT / "scripts/learn_affordances.py"),
            **{f"chreatures/{name}": sha256(ROOT / "chreatures" / name) for name in (
                "learning.py", "tiled_circuit.py", "fast_circuit.py", "training_environment.py",
                "physics.py", "physical_batch.py", "sensorium.py", "neural_ports.py",
            )},
            **{str(path.relative_to(ROOT)): sha256(path) for path in native_sources},
        },
        "device": brain.metadata()["device"],
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "build_abi": {
            "python": platform.python_version(), "implementation": platform.python_implementation(),
            "machine": platform.machine(), "platform": platform.platform(),
            "numpy": np.__version__, "mujoco": mujoco.__version__,
            "torch": torch.__version__, "hip": torch.version.hip,
            "triton": triton.__version__,
            "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
            "native_world_extension": {
                "filename": native_binary.name, "bytes": native_binary.stat().st_size,
                "sha256": sha256(native_binary),
            },
        },
    }
    _atomic_json(args.output / "manifest.json", manifest)

    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            if episode:
                cohort.reset(episode)
            trainer.reset_private_state()
            raw, physiology, _ = cohort.observe(0.05)
            normalized = trainer.normalize(raw, update=False)
            scale = 1.0 if episode % 2 == 0 else args.exploration_scale
            mode = 0 if scale == 1.0 else 1
            temporary = args.output / f"episode-{episode:03d}.tmp"
            final = args.output / f"episode-{episode:03d}"
            temporary.mkdir()
            split_arrays = {}
            for split, indices in split_indices.items():
                directory = temporary / split
                directory.mkdir(exist_ok=True)
                split_arrays[split] = _open_arrays(directory, rows, len(indices))

            episode_started = time.perf_counter()
            for row in range(rows):
                previous = _sample_action(trainer, normalized, physiology, scale)
                for split, indices in split_indices.items():
                    arrays = split_arrays[split]
                    arrays["features"][row] = normalized[indices]
                    arrays["source_channels"][row] = cohort.last_source_channels[indices]
                    arrays["physiology"][row] = physiology[indices]
                    arrays["action"][row] = previous["action"][indices]
                    arrays["reset"][row] = row == 0
                    arrays["done"][row] = row == rows - 1
                    arrays["valid"][row] = True
                    arrays["behavior_mode"][row] = mode
                for _ in range(args.macro_steps):
                    cohort.advance(previous["action"], 0.05)
                    raw, physiology, _ = cohort.observe(0.05)
                next_normalized = trainer.normalize(raw, update=False)
                done = np.full(args.worlds * 3, row == rows - 1, dtype=bool)
                trainer.finish_transition(
                    previous, next_normalized, np.zeros(args.worlds * 3, dtype=np.float32),
                    done, args.macro_steps * 0.05,
                )
                normalized = next_normalized
            for arrays in split_arrays.values():
                _flush(arrays)
            del split_arrays
            for split, indices in split_indices.items():
                directory = temporary / split
                np.save(directory / "terminal_features.npy", normalized[indices])
                np.save(
                    directory / "terminal_source_channels.npy",
                    cohort.last_source_channels[indices],
                )
                np.save(directory / "terminal_physiology.npy", physiology[indices])
            receipts = {
                split: _file_receipts(temporary / split) for split in split_indices
            }
            episode_receipt = {
                "episode": episode, "physical_steps": args.episode_steps,
                "macro_rows": rows, "behavior_mode": mode, "std_scale": scale,
                "elapsed_seconds": time.perf_counter() - episode_started,
                "files": receipts,
            }
            _atomic_json(temporary / "receipt.json", episode_receipt)
            os.replace(temporary, final)
            manifest["episodes"].append(episode_receipt)
            _atomic_json(args.output / "manifest.json", manifest)
            print(
                f"episode={episode + 1}/{args.episodes} rows={rows} "
                f"elapsed={episode_receipt['elapsed_seconds']:.3f}s", flush=True,
            )
    finally:
        cohort.close()
    manifest["completed"] = True
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["brain_timing_seconds"] = cohort.timings
    manifest["completed_unix"] = time.time()
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({
        "completed": True, "resident_rows": manifest["resident_rows_total"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
