#!/usr/bin/env python3
"""Collect rich 20 Hz achieved-history control data in fresh chemical worlds."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import sysconfig
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.motor_inheritance import ACTIONS, MotorArtifact, MotorOrgan
from research.sensorimotor_skills.data import DT_SECONDS

FORMAT = "chreatures-sensorimotor-play-rich-v2"
SCHEMA = ROOT / "research/sensorimotor_skills/trajectory-schema-rich-v2.json"
RICH_PROFILE_SHA256 = (
    "c71380718ba5535dbaebdeaf8aa2e88cc45cf218312a03e13507877f02a5554e"
)
RICH_CHANNEL_NAMES_SHA256 = (
    "b4c6b328116d820143e16ee922ccffd7b950dbe008efc580ad93056e01349bfa"
)
RICH_OBSERVATION_ORDER = (
    "rich_body_v1_4096", "canonical_channels_351", "physiology_6",
)
TRANSITION_OUTCOMES = (
    "nutrition", "contact", "distance", "effort", "mechanical_work",
    "ingested_mass", "mouth_material_contacts", "homeostatic_reward",
)
MIN_TRAINING_WORLDS = 2
MIN_TRAINING_STEPS = 44
PHYSIOLOGY = (
    "energy", "gut", "fatigue", "speed", "angular_velocity", "support",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_gzip_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode())
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--motor-organ", type=Path, required=True)
    parser.add_argument(
        "--chemical-habitat", type=Path,
        default=ROOT / "data/habitats/living-reef.json",
    )
    parser.add_argument(
        "--chemical-biosphere", type=Path,
        default=ROOT / "data/biosphere/living-reef.json",
    )
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=DT_SECONDS)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--brain-backend", choices=("tiled", "triton", "microbatch"),
        default="tiled",
    )
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--physical-backend", choices=("fast", "reference"), default="fast")
    parser.add_argument("--record-neural-readouts", action="store_true")
    parser.add_argument("--exploration-sigma", type=float, default=0.18)
    parser.add_argument("--exploration-timescale", type=float, default=0.35)
    parser.add_argument("--exploration-segment-steps", type=int, default=20)
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "collect a small mechanics check; its manifest is explicitly marked "
            "not training-ready"
        ),
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not 1 <= args.worlds <= 64 or not 1 <= args.episodes <= 100:
        raise SystemExit("worlds must be 1..64 and episodes 1..100")
    if not 1 <= args.steps <= 1_000_000:
        raise SystemExit("steps must be 1..1,000,000")
    if not args.smoke and args.worlds < MIN_TRAINING_WORLDS:
        raise SystemExit(
            f"training collections require at least {MIN_TRAINING_WORLDS} worlds; "
            "use --smoke only for a non-training mechanics check"
        )
    if not args.smoke and args.steps < MIN_TRAINING_STEPS:
        raise SystemExit(
            f"training collections require at least {MIN_TRAINING_STEPS} steps "
            "for complete 40-step future-goal windows; use --smoke only for a "
            "non-training mechanics check"
        )
    if not math.isfinite(args.dt) or args.dt != DT_SECONDS:
        raise SystemExit("rich collector v2 requires the 0.05 s physical interval")
    if (
        not math.isfinite(args.exploration_sigma)
        or not 0 <= args.exploration_sigma <= 0.5
        or not math.isfinite(args.exploration_timescale)
        or args.exploration_timescale < args.dt
        or not 1 <= args.exploration_segment_steps <= 1000
    ):
        raise SystemExit("invalid bounded exploration configuration")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be absent or empty")


def training_readiness(args: argparse.Namespace) -> dict[str, Any]:
    """Describe whether a completed invocation is eligible for worker training."""
    if args.smoke:
        ready = False
        reason = (
            "explicit --smoke collection validates collection mechanics only and "
            "must not be used to fit, select, or evaluate a worker"
        )
    else:
        ready = True
        reason = None
    return {
        "ready": ready,
        "mode": "smoke" if args.smoke else "training",
        "reason": reason,
        "minimum_worlds": MIN_TRAINING_WORLDS,
        "minimum_steps_per_episode": MIN_TRAINING_STEPS,
        "actual_worlds": args.worlds,
        "actual_steps_per_episode": args.steps,
    }


def physiology_rows(
    body_states: list[list[dict[str, Any]]], circuit: np.ndarray,
) -> np.ndarray:
    rows = []
    index = 0
    for bodies in body_states:
        for body in bodies:
            rows.append((
                body["energy"], body["gut"], body["fatigue"],
                math.tanh(float(body["speed"]) / 2),
                math.tanh(float(body["angular_velocity"]) / 4),
                float(circuit[index, 2]),
            ))
            index += 1
    return np.asarray(rows, dtype=np.float32)


def transition_outcome_rows(
    body_states: list[list[dict[str, Any]]],
    outcomes: list[Mapping[str, Mapping[str, Any]]],
) -> np.ndarray:
    rows = []
    for bodies, world_outcomes in zip(body_states, outcomes, strict=True):
        for body in bodies:
            outcome = world_outcomes[str(body["id"])]
            rows.append([float(outcome.get(name, 0.0)) for name in TRANSITION_OUTCOMES])
    result = np.ascontiguousarray(rows, dtype=np.float32)
    if result.shape != (sum(map(len, body_states)), len(TRANSITION_OUTCOMES)):
        raise RuntimeError("transition outcome cohort has the wrong shape")
    return result


class CorrelatedMotorPlay:
    """Seeded motor babble independent of world contents and resident identity."""

    def __init__(
        self, count: int, *, seed: int, sigma: float, timescale: float,
        segment_steps: int, dt: float,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.sigma = float(sigma)
        self.alpha = math.exp(-dt / timescale)
        self.segment_steps = int(segment_steps)
        self.state = np.zeros((count, len(ACTIONS)), dtype=np.float32)
        self.target = np.zeros_like(self.state)
        self.modes = np.ones(count, dtype=np.int64)
        self.previous = np.zeros_like(self.state)

    def apply(self, base: np.ndarray, step: int) -> np.ndarray:
        if step % self.segment_steps == 0:
            self.target = self.rng.normal(
                0.0, self.sigma, size=self.state.shape,
            ).astype(np.float32)
            self.modes = self.rng.integers(0, 4, size=len(base))
            # One content-independent segment in four practices stopping. The
            # smooth correction cancels locomotor axes only; it never depends
            # on a packet, outcome, coordinate, or sensory value.
            stopped = self.modes == 0
            self.target[stopped, 0:2] = -base[stopped, 0:2]
            # One segment in four observes the inherited policy unperturbed.
            self.target[self.modes == 1] = 0.0
        innovation = self.rng.normal(0.0, 1.0, size=self.state.shape).astype(np.float32)
        scale = self.sigma * math.sqrt(max(0.0, 1.0 - self.alpha**2))
        self.state = (
            self.alpha * self.state
            + (1.0 - self.alpha) * self.target
            + scale * innovation
        ).astype(np.float32)
        executed = np.clip(base + self.state, -1.0, 1.0).astype(np.float32)
        stopped = self.modes == 0
        executed[stopped, 0:2] = self.alpha * self.previous[stopped, 0:2]
        inherited_only = self.modes == 1
        executed[inherited_only] = base[inherited_only]
        executed[:, 3:7] = np.maximum(executed[:, 3:7], 0.0)
        self.previous = executed.copy()
        return executed

    def snapshot_value(self) -> dict[str, Any]:
        return {
            "format": "chreatures-correlated-motor-play-state-v1",
            "config": {
                "sigma": self.sigma,
                "alpha": self.alpha,
                "segment_steps": self.segment_steps,
            },
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "state": self.state.astype(float).tolist(),
            "target": self.target.astype(float).tolist(),
            "modes": self.modes.tolist(),
            "previous": self.previous.astype(float).tolist(),
        }


def observe(
    pool: ProcessWorldPool, brain: FixedCohortBrain, dt: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[dict[str, Any]]],
]:
    rich, senses, bodies = pool.observe_arrays()
    features, circuit, _ = brain.step_channels(senses, dt)
    physiology = physiology_rows(bodies, circuit)
    observation = np.ascontiguousarray(
        np.concatenate((rich, senses, physiology), axis=1), dtype=np.float32,
    )
    if observation.shape != (len(senses), 4453):
        raise RuntimeError("rich collector observation has the wrong dimensions")
    return observation, senses, physiology, features, bodies


def source_receipts(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    causal_modules = (
        "acoustics.py", "articulated.py", "biosphere.py", "circuit_blueprint.py",
        "ecological_exchange.py", "ecology.py", "fast_circuit.py", "fields.py",
        "growth.py", "homeostasis.py", "learning.py", "malecns.py",
        "material_objects.py", "metabolism.py", "motor_inheritance.py",
        "native_world.py", "neural_ports.py", "physical_batch.py", "physics.py",
        "remote_brain.py", "sensorium.py", "somatic.py", "tiled_circuit.py",
        "training_environment.py",
    )
    paths = {
        "graph_manifest": args.graph.resolve() / "manifest.json",
        "port_bundle": args.port_bundle.resolve(),
        "motor_organ": args.motor_organ.resolve(),
        "chemical_habitat": args.chemical_habitat.resolve(),
        "chemical_biosphere": args.chemical_biosphere.resolve(),
        "collector": Path(__file__).resolve(),
        "data_boundary": ROOT / "research/sensorimotor_skills/data.py",
        "world_pool_runner": ROOT / "scripts/learn_affordances.py",
        "schema": SCHEMA.resolve(),
        **{f"chreatures/{name}": ROOT / "chreatures" / name for name in causal_modules},
        "native/world-kernels/Cargo.toml": ROOT / "native/world-kernels/Cargo.toml",
        "native/world-kernels/Cargo.lock": ROOT / "native/world-kernels/Cargo.lock",
    }
    for path in sorted((ROOT / "native/world-kernels/src").glob("*")):
        if path.is_file():
            paths[f"native/world-kernels/src/{path.name}"] = path
    return {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in paths.items()
    }


def git_identity(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    relative = sorted({
        str(Path(value["path"]).resolve().relative_to(ROOT))
        for value in sources.values()
        if Path(value["path"]).resolve().is_relative_to(ROOT)
    })
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *relative], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout
    content = {name: value["sha256"] for name, value in sorted(sources.items())}
    return {
        "revision": revision,
        "targeted_sources_dirty": bool(status),
        "targeted_source_content_sha256": canonical_sha256(content),
    }


def main() -> int:
    args = arguments()
    validate_arguments(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # The collector remains inspectable (`--help`, schema validation) on the
    # Torch-free deployment laptop. These imports are required only in the
    # explicitly selected full-graph collection environment.
    from chreatures.neural_ports import NeuralPortBundle
    from chreatures.sensorium import (
        RICH_CHANNEL_NAMES_SHA256 as ACTIVE_RICH_CHANNEL_NAMES_SHA256,
        RICH_PROFILE_SHA256 as ACTIVE_RICH_PROFILE_SHA256,
        profile_identity,
    )
    from chreatures.training_environment import EmbodiedTrainingProfile
    from scripts.learn_affordances import (
        FixedCohortBrain,
        ProcessWorldPool,
        load_graph,
        native_extension_receipt,
        stable_brain_identity,
    )

    profile = EmbodiedTrainingProfile.chemical_nursery(
        args.chemical_habitat, args.chemical_biosphere,
    )
    if (
        ACTIVE_RICH_PROFILE_SHA256 != RICH_PROFILE_SHA256
        or ACTIVE_RICH_CHANNEL_NAMES_SHA256 != RICH_CHANNEL_NAMES_SHA256
        or profile.component("sensorium") != profile_identity()
    ):
        raise SystemExit("rich collector sensorium identity differs")
    graph = load_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    if len(ports.input_names) != 351 or len(ports.readout_names) != 384:
        raise SystemExit("rich collector v2 requires the pinned 351-input/384-readout ports")
    artifact = MotorArtifact.load(args.motor_organ)
    if int(artifact.config["feature_dim"]) != len(ports.readout_names):
        raise SystemExit("motor organ and neural readout dimensions differ")
    if int(artifact.config["macro_steps"]) != 5:
        raise SystemExit("rich collector v2 requires the inherited five-tick macro policy")
    provenance = artifact.metadata["training_provenance"]
    if provenance["graph_sha256"] != str(graph.hash):
        raise SystemExit("motor organ and recurrence graph identities differ")
    motor_port_transfer = None
    if (
        provenance["port_spec_sha256"] != ports.spec_hash
        or provenance.get("port_bundle_sha256") != sha256(args.port_bundle.resolve())
    ):
        source_path = ROOT / "data/ports/retinal-v1-maps.npz"
        source = NeuralPortBundle.load(source_path, graph)
        target_document = json.loads(
            (ROOT / "data/ports/retinal-v2.json").read_text(encoding="utf-8")
        )
        derivation = target_document["built_artifact"]["mapping_derivation"]
        matrices_equal = all(
            np.array_equal(left, right)
            for left, right in (
                (source.input_map.indptr, ports.input_map.indptr),
                (source.input_map.indices, ports.input_map.indices),
                (source.input_map.data, ports.input_map.data),
                (source.readout_map.indptr, ports.readout_map.indptr),
                (source.readout_map.indices, ports.readout_map.indices),
                (source.readout_map.data, ports.readout_map.data),
            )
        )
        if (
            source.spec_hash != provenance["port_spec_sha256"]
            or sha256(source_path) != provenance.get("port_bundle_sha256")
            or derivation.get("source_bundle_sha256") != sha256(source_path)
            or derivation.get("matrix_equality_required") is not True
            or source.input_names != ports.input_names
            or source.readout_names != ports.readout_names
            or not matrices_equal
        ):
            raise SystemExit("motor organ retinal-v1 to retinal-v2 transfer is not exact")
        motor_port_transfer = {
            "semantics": (
                "retinal-v1 inherited motor policy drives byte-identical sparse "
                "maps under retinal-v2 physical source semantics"
            ),
            "source_port_spec_sha256": source.spec_hash,
            "source_port_bundle_sha256": sha256(source_path),
            "target_port_spec_sha256": ports.spec_hash,
            "target_port_bundle_sha256": sha256(args.port_bundle.resolve()),
            "sparse_matrices_equal": True,
        }

    residents_per_world = len(profile.component("habitat")["bodies"])
    if residents_per_world <= 0:
        raise SystemExit("rich collector habitat has no residents")
    count = args.worlds * residents_per_world
    brain = FixedCohortBrain(
        graph, ports, count, device=args.device, backend=args.brain_backend,
        microbatch_size=args.microbatch_size,
    )
    pool = ProcessWorldPool(
        args.worlds, dict(ports.spec), profile.to_value(), args.physical_backend,
        residents_per_world=residents_per_world,
    )
    sources = source_receipts(args)
    software = {
        "python": sys.version.split()[0],
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "numpy": np.__version__,
    }
    for package in ("torch", "triton", "mujoco"):
        try:
            from importlib.metadata import version
            software[package] = version(package)
        except Exception:  # package inventory is diagnostic, never a dependency probe
            software[package] = None
    collection_identity = {
        "format": "chreatures-sensorimotor-play-rich-collection-identity-v2",
        "git": git_identity(sources),
        "sources": sources,
        "software": software,
        "device": stable_brain_identity(brain.metadata())["device"],
        "graph_sha256": str(graph.hash),
        "port_spec_sha256": ports.spec_hash,
        "port_bundle_sha256": sha256(args.port_bundle.resolve()),
        "motor_artifact_sha256": artifact.sha256,
        "profile_sha256": profile.sha256,
        "rich_profile_sha256": RICH_PROFILE_SHA256,
        "rich_channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
        "observation_order": list(RICH_OBSERVATION_ORDER),
        "motor_port_transfer": motor_port_transfer,
    }
    collection_identity["sha256"] = canonical_sha256(collection_identity)
    identity_receipt = atomic_json(output / "identity.json", collection_identity)
    packets = []
    births = []
    try:
        for episode in range(args.episodes):
            # The Living Reef profile preserves its complete birth-v3 material
            # exchange and has no feeder-placement curriculum.
            stage = 0
            seeds = [args.seed + episode * 1009 + index for index in range(args.worlds)]
            body_states = pool.call_all("reset", [
                {"seed": seed, "held_out": False, "stage": stage} for seed in seeds
            ])
            if any(len(bodies) != residents_per_world for bodies in body_states):
                raise RuntimeError("rich collector world resident count differs")
            partition_keys = [
                f"episode-{episode:03d}/world-{world:03d}/resident-{resident:02d}"
                for world in range(args.worlds)
                for resident in range(residents_per_world)
            ]
            if brain.resident_ids:
                brain.remove_residents(brain.resident_ids)
            brain.add_residents(partition_keys)
            organs = [
                MotorOrgan(artifact, seed=args.seed ^ (episode << 20) ^ index)
                for index in range(count)
            ]
            play = CorrelatedMotorPlay(
                count, seed=args.seed ^ 0x51A11 ^ episode,
                sigma=args.exploration_sigma,
                timescale=args.exploration_timescale,
                segment_steps=args.exploration_segment_steps, dt=args.dt,
            )

            # Publish the complete fresh private birth before the first sense
            # advances the neural circuit and before any physical mutation.
            birth_dir = output / f"episode-{episode:03d}-birth"
            worlds_receipt = atomic_gzip_json(
                birth_dir / "worlds.json.gz", pool.call_all("snapshot"),
            )
            neural_receipt = brain.snapshot(birth_dir, "neural")
            private_receipt = atomic_json(birth_dir / "motor-private.json", {
                "format": "chreatures-sensorimotor-play-private-birth-v1",
                "artifact_sha256": artifact.sha256,
                "partition_keys": partition_keys,
                "motor": [organ.snapshot_value(include_artifact=False) for organ in organs],
                "exploration": play.snapshot_value(),
            })
            birth_index = atomic_json(birth_dir / "index.json", {
                "format": "chreatures-sensorimotor-play-birth-v1",
                "episode": episode, "stage": stage, "world_seeds": seeds,
                "profile_sha256": profile.sha256,
                "collection_identity_sha256": collection_identity["sha256"],
                "collection_identity": identity_receipt,
                "worlds": worlds_receipt,
                "neural": {key: neural_receipt[key] for key in ("name", "bytes", "sha256")},
                "motor_private": private_receipt,
            })
            births.append({"episode": episode, **birth_index})

            observation, source_rows, physiology, neural, body_states = observe(
                pool, brain, args.dt,
            )
            observation_sequence = [observation]
            source_sequence = [source_rows]
            neural_sequence = [neural]
            action_sequence = []
            oral_sequence = []
            outcome_sequence = []
            reset_sequence = [np.ones(count, dtype=np.bool_)]
            aggregate = {"nutrition": 0.0, "ingested_mass": 0.0, "mouth_contacts": 0}
            for step in range(args.steps):
                base = np.asarray([
                    [action[name] for name in ACTIONS]
                    for organ, features, local in zip(
                        organs, neural, physiology, strict=True,
                    )
                    for action in (organ.tick(features, local, args.dt),)
                ], dtype=np.float32)
                executed = play.apply(base, step)
                actions_by_world = []
                oral = np.zeros(count, dtype=np.float32)
                for world in range(args.worlds):
                    actions = {}
                    for resident, body in enumerate(body_states[world]):
                        index = world * residents_per_world + resident
                        action = dict(zip(
                            ACTIONS, executed[index].astype(float).tolist(), strict=True,
                        ))
                        oral[index] = np.clip(
                            (1 - body["gut"]) * (1.1 - body["energy"]), 0, 1,
                        )
                        action["eat"] = float(oral[index])
                        actions[body["id"]] = action
                    actions_by_world.append(actions)
                advanced = pool.call_all("advance", [
                    {"actions": actions, "dt": args.dt} for actions in actions_by_world
                ])
                advanced_outcomes = [item[0] for item in advanced]
                advanced_bodies = [item[1] for item in advanced]
                outcome_sequence.append(transition_outcome_rows(
                    advanced_bodies, advanced_outcomes,
                ))
                for world_outcomes in advanced_outcomes:
                    for outcome in world_outcomes.values():
                        aggregate["nutrition"] += float(outcome["nutrition"])
                        aggregate["ingested_mass"] += float(outcome.get("ingested_mass", 0.0))
                        aggregate["mouth_contacts"] += int(
                            outcome.get("mouth_material_contacts", 0)
                        )
                action_sequence.append(executed)
                oral_sequence.append(oral)
                observation, source_rows, physiology, neural, body_states = observe(
                    pool, brain, args.dt,
                )
                observation_sequence.append(observation)
                source_sequence.append(source_rows)
                neural_sequence.append(neural)
                reset_sequence.append(np.zeros(count, dtype=np.bool_))

            arrays = {
                "observation": np.ascontiguousarray(
                    np.stack(observation_sequence), dtype=np.float32,
                ),
                "canonical_channels": np.ascontiguousarray(
                    np.stack(source_sequence), dtype=np.float32,
                ),
                "executed_actions": np.stack(action_sequence).astype(np.float32),
                "oral_command": np.stack(oral_sequence).astype(np.float32),
                "transition_outcomes": np.ascontiguousarray(
                    np.stack(outcome_sequence), dtype=np.float32,
                ),
                "reset": np.stack(reset_sequence).astype(np.bool_),
                "dt_seconds": np.asarray(args.dt, dtype=np.float64),
            }
            if not np.array_equal(
                arrays["canonical_channels"], arrays["observation"][..., 4096:4447],
            ):
                raise RuntimeError("canonical diagnostic differs from rich observation")
            if args.record_neural_readouts:
                arrays["neural_readouts"] = np.stack(neural_sequence).astype(np.float32)
            packet = atomic_npz(output / f"episode-{episode:03d}.npz", arrays)
            packets.append({
                "episode": episode, "stage": stage, **packet,
                "steps": args.steps, "resident_partitions": partition_keys,
                "model_array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                "diagnostic_outcomes": aggregate,
            })
    finally:
        pool.close()

    native = native_extension_receipt()
    manifest = {
        "format": FORMAT,
        "version": 2,
        "training_readiness": training_readiness(args),
        "scope": {
            "worlds": args.worlds, "residents_per_world": residents_per_world,
            "episodes": args.episodes, "steps_per_episode": args.steps,
            "dt_seconds": args.dt,
            "transition_outcome_order": list(TRANSITION_OUTCOMES),
            "model_columns_exclude": [
                "world coordinates", "headings", "distances", "bearings",
                "entity identities", "object labels", "body identities",
            ],
        },
        "schema": {
            "path": str(SCHEMA.relative_to(ROOT)),
            "bytes": SCHEMA.stat().st_size,
            "sha256": sha256(SCHEMA),
        },
        "profile": profile.to_value(),
        "rich_sensorium": {
            "profile_sha256": RICH_PROFILE_SHA256,
            "channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
            "observation_order": list(RICH_OBSERVATION_ORDER),
        },
        "transition_outcome_order": list(TRANSITION_OUTCOMES),
        "motor_port_transfer": motor_port_transfer,
        "motor_artifact_sha256": artifact.sha256,
        "graph_sha256": str(graph.hash),
        "port_spec_sha256": ports.spec_hash,
        "chemistry_sha256": profile.component("biosphere")["material_objects"][
            "chemistry_sha256"
        ],
        "brain": stable_brain_identity(brain.metadata()),
        "native_world": None if native is None else {
            key: native[key] for key in ("bytes", "sha256", "python_soabi", "cache_tag")
            if key in native
        },
        "exploration": {
            "format": "chreatures-correlated-motor-play-v1",
            "sigma": args.exploration_sigma,
            "timescale_seconds": args.exploration_timescale,
            "segment_steps": args.exploration_segment_steps,
            "segment_modes": (
                "one-quarter smooth locomotor decay to stop, one-quarter inherited-only, "
                "one-half bounded Gaussian target; content-independent"
            ),
        },
        "oral_command_law": "clip((1-gut)*(1.1-energy),0,1)",
        "bootstrap_context_limitation": (
            "The inherited organ updates private context with its proposed held macro "
            "action; downstream per-tick exploration is delivered physically but is not "
            "fed back into that organ context. executed_actions is authoritative."
        ),
        "seed": args.seed,
        "collection_identity": collection_identity,
        "collection_identity_receipt": identity_receipt,
        "birth_checkpoints": births,
        "packets": packets,
        "sources": sources,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output), "content_sha256": manifest["content_sha256"],
        "packets": len(packets), "transitions": args.episodes * args.steps * count,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
