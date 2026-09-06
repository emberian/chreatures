#!/usr/bin/env python3
"""Closed-loop research probe for an achieved-history sensorimotor worker."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLLECTION_REVISION = "78edd5604dd857bbcd341e20bc6fe9fd26083cb8"
COLLECTION_SOURCE_CONTENT = "dc2a589235b3e7c77aa6d188f8383663af64ffaf9a42c47fa8b88a18a82f2475"
COLLECTION_NATIVE = "cf4ab75a1f7e7f9bedff8b0bb36b2a12f44c3664b42483b7b4c62e75109f9cb4"
WORKER_FORMAT = "chreatures-sensorimotor-worker-research-v2"
CONDITIONS = (
    "goal_conditioned_matched", "goal_conditioned_alternate", "goal_free",
    "repeat_last", "zero_action", "inherited_4hz", "recorded_replay",
)
ACTIONS = (
    "thrust", "yaw", "gaze_pitch", "grip",
    "signal_low", "signal_mid", "signal_high", "posture",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def receipt_path(directory: Path, receipt: Mapping[str, Any], name: str) -> Path:
    relative = Path(str(receipt.get("path", "")))
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"invalid {name} receipt path")
    if path.stat().st_size != int(receipt.get("bytes", -1)) or sha256(path) != receipt.get("sha256"):
        raise ValueError(f"{name} receipt differs")
    return path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--conditioned-workers", type=Path, nargs="+", required=True)
    parser.add_argument("--goal-free-workers", type=Path, nargs="+", required=True)
    parser.add_argument("--motor-organ", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--world-slots", type=int, nargs="+", default=[14, 15])
    parser.add_argument("--resident-columns", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--prefixes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--brain-backend", choices=("tiled", "triton"), default="tiled")
    parser.add_argument("--physical-backend", choices=("fast", "reference"), default="fast")
    return parser.parse_args()


def quaternion_error(first: list[float], second: list[float]) -> float:
    dot = float(abs(np.dot(np.asarray(first), np.asarray(second))))
    return 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))


def body_pose(snapshot: Mapping[str, Any], resident: int) -> dict[str, Any]:
    body = snapshot["world"]["bodies"][resident]
    return {
        "position": [float(body[key]) for key in ("x", "y", "z")],
        "quaternion": [float(value) for value in body["quaternion"]],
    }


def physiology(bodies: list[dict[str, Any]], circuit: np.ndarray) -> np.ndarray:
    return np.asarray([
        [
            body["energy"], body["gut"], body["fatigue"],
            math.tanh(float(body["speed"]) / 2),
            math.tanh(float(body["angular_velocity"]) / 4),
            circuit[index, 2],
        ]
        for index, body in enumerate(bodies)
    ], dtype=np.float32)


def observe(pool, brain, normalizer):
    result = pool.call_all("observe")[0]
    source = np.asarray(result[0], dtype=np.float32)
    features, circuit, _ = brain.step_channels(source, 0.05)
    local = physiology(result[1], circuit)
    observation = np.concatenate((source, local), axis=-1).astype(np.float32)
    return source, local, features, result[1], normalizer.normalize(observation)


def load_neural_subset(
    path: Path, start: int, stop: int, graph_hash: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as value:
        metadata = json.loads(str(value["metadata"]))
        if metadata.get("graph_sha256") != graph_hash:
            raise ValueError("birth neural graph differs")
        resident_ids = list(metadata["resident_ids"])
        state = {
            name: np.ascontiguousarray(value[name][start:stop])
            for name in ("rates", "adaptation", "support", "times")
        }
    selected = resident_ids[start:stop]
    if len(selected) != 3 or any(
        f"/resident-{index:02d}" not in key for index, key in enumerate(selected)
    ):
        raise ValueError("birth neural resident ordering differs")
    receipt = {
        "source_neural_sha256": sha256(path),
        "source_batch_size": len(resident_ids),
        "resident_axis_indices": list(range(start, stop)),
        "partition_keys": selected,
        "subset_state_sha256": canonical_sha256({
            name: hashlib.sha256(array.tobytes()).hexdigest()
            for name, array in state.items()
        }),
    }
    return receipt, state, selected


def import_neural_subset(brain, state: Mapping[str, np.ndarray], selected: list[str]) -> None:
    brain.add_residents(selected)
    restore_neural_state(brain, state)


def validate_replay_sources(collection: Mapping[str, Any]) -> None:
    """Require the running physical stack to be the collector's frozen stack."""
    sources = collection.get("sources", {})
    checked = 0
    for name, receipt in sources.items():
        if name == "world_pool_runner":
            path = ROOT / "scripts/learn_affordances.py"
        elif name.startswith("chreatures/") or name.startswith("native/"):
            path = ROOT / name
        else:
            continue
        if not path.is_file() or sha256(path) != receipt.get("sha256"):
            raise SystemExit(f"frozen replay source differs: {name}")
        checked += 1
    if checked == 0:
        raise SystemExit("collection has no verifiable physical source receipts")


def copy_neural_state(brain) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value.copy())
        for name, value in brain.circuit.export_state().items()
    }


def restore_neural_state(brain, state: Mapping[str, np.ndarray]) -> None:
    brain.circuit.import_state({name: value.copy() for name, value in state.items()})


def oral_from_actual(body: Mapping[str, Any]) -> np.float32:
    return np.float32(np.clip(
        (1.0 - float(body["gut"])) * (1.1 - float(body["energy"])), 0.0, 1.0,
    ))


def action_mapping(action: np.ndarray, oral: float) -> dict[str, float]:
    action = np.asarray(action, dtype=np.float32)
    if (
        action.shape != (8,)
        or not np.isfinite(action).all()
        or np.any(action < -1)
        or np.any(action > 1)
        or np.any(action[[3, 4, 5, 6]] < 0)
    ):
        raise ValueError("executed motor action violates the recorded physical contract")
    oral = np.float32(oral)
    if not np.isfinite(oral) or oral < 0 or oral > 1:
        raise ValueError("oral command violates the supplied body-state law")
    result = dict(zip(ACTIONS, action.astype(float).tolist(), strict=True))
    result["eat"] = float(oral)
    return result


def goal_window(episode, normalizer, end: int, column: int) -> np.ndarray:
    if end < 3 or end >= len(episode.observations):
        raise ValueError("goal window lies outside recorded episode")
    return normalizer.normalize(episode.observations[end - 3 : end + 1, column])


def encode_worker_observation(worker, observation, previous, hidden, reset, torch, device):
    with torch.inference_mode():
        state, hidden = worker.encode_sequence(
            torch.as_tensor(observation[None, None], device=device),
            torch.as_tensor(previous[None, None], device=device),
            hidden,
            torch.as_tensor([[reset]], dtype=torch.bool, device=device),
        )
    return state[0, 0], hidden


def load_worker_checkpoint(path, expected_mode, dataset, torch, device, classes):
    GoalEncoder, SensorimotorSkillConfig, SensorimotorWorker = classes
    # TorchVersion is the only non-tensor global emitted by the research
    # checkpoint identity. Keep weights-only loading and allowlist that exact
    # inert string subclass rather than falling back to unrestricted pickle.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(
            path.resolve(), map_location=device, weights_only=True,
        )
    if checkpoint.get("format") != WORKER_FORMAT:
        raise SystemExit(f"worker checkpoint format differs: {path}")
    identity = checkpoint.get("identity", {})
    if checkpoint.get("artifact_mode") != expected_mode or identity.get(
        "artifact_mode"
    ) != expected_mode:
        raise SystemExit(f"worker mode differs: {path}")
    if identity.get("dataset") != dict(dataset.identity.file_sha256s):
        raise SystemExit(f"worker was trained from another dataset: {path}")
    for relative, expected in identity["source_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != expected:
            raise SystemExit(f"worker research source differs: {relative}")
    config = SensorimotorSkillConfig(**identity["config"])
    encoder = GoalEncoder(config).to(device)
    worker = SensorimotorWorker(config).to(device)
    encoder.load_state_dict(checkpoint["goal_encoder"], strict=True)
    worker.load_state_dict(checkpoint["worker"], strict=True)
    encoder.freeze()
    worker.eval()
    seed = int(identity["seed"])
    return {
        "path": str(path.resolve()), "sha256": sha256(path.resolve()),
        "mode": expected_mode, "seed": seed, "identity": identity,
        "encoder": encoder, "worker": worker,
    }


def alternate_goal_end(prefix: int, horizon: int, observations: int) -> int:
    later = prefix + 2 * horizon
    if later < observations:
        return later
    earlier = prefix - horizon
    if earlier >= 3:
        return earlier
    raise ValueError("no nonoverlapping same-resident alternate goal window")


def main() -> int:
    args = arguments()
    if len(args.conditioned_workers) != len(args.goal_free_workers):
        raise SystemExit("conditioned and goal-free checkpoint counts differ")
    if args.output.exists():
        raise SystemExit("probe output must not already exist")
    if any(value not in (14, 15) for value in args.world_slots):
        raise SystemExit("probe v1 is restricted to heldout world slots 14 and 15")
    if any(value not in (0, 1, 2) for value in args.resident_columns):
        raise SystemExit("resident columns must be 0, 1, or 2")
    if any(value not in (0, 1) for value in args.episodes):
        raise SystemExit("probe v1 episodes must be 0 or 1")
    if any(value <= 3 for value in args.prefixes) or any(value not in (10, 20, 40) for value in args.horizons):
        raise SystemExit("invalid prefix or horizon")

    import torch
    from chreatures.neural_ports import NeuralPortBundle
    from chreatures.motor_inheritance import MotorArtifact, MotorOrgan
    from chreatures.training_environment import EmbodiedTrainingProfile
    from research.sensorimotor_skills.data import Normalizer, PlayDataset
    from research.sensorimotor_skills.model import (
        GoalEncoder, SensorimotorSkillConfig, SensorimotorWorker,
    )
    from scripts.learn_affordances import FixedCohortBrain, ProcessWorldPool, load_graph, native_extension_receipt

    dataset = PlayDataset(args.dataset)
    manifest = dataset.manifest
    collection = manifest["collection_identity"]
    if (
        collection["git"]["revision"] != COLLECTION_REVISION
        or collection["git"]["targeted_source_content_sha256"] != COLLECTION_SOURCE_CONTENT
        or manifest["native_world"]["sha256"] != COLLECTION_NATIVE
    ):
        raise SystemExit("dataset was not collected by the frozen physical source/native build")
    validate_replay_sources(collection)
    graph = load_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    if str(graph.hash) != manifest["graph_sha256"] or ports.spec_hash != manifest["port_spec_sha256"]:
        raise SystemExit("probe graph or ports differ from collection")
    if sha256(args.port_bundle.resolve()) != collection.get("port_bundle_sha256"):
        raise SystemExit("probe port bundle bytes differ from collection")

    classes = (GoalEncoder, SensorimotorSkillConfig, SensorimotorWorker)
    conditioned = [load_worker_checkpoint(
        path, "goal-conditioned", dataset, torch, args.device, classes,
    ) for path in args.conditioned_workers]
    goal_free = [load_worker_checkpoint(
        path, "goal-free", dataset, torch, args.device, classes,
    ) for path in args.goal_free_workers]
    conditioned_by_seed = {item["seed"]: item for item in conditioned}
    goal_free_by_seed = {item["seed"]: item for item in goal_free}
    if set(conditioned_by_seed) != set(goal_free_by_seed) or len(
        conditioned_by_seed
    ) != len(conditioned):
        raise SystemExit("paired worker seed identities differ or repeat")
    normalizer_values = {
        canonical_sha256(item["identity"]["normalizer"])
        for item in conditioned + goal_free
    }
    if len(normalizer_values) != 1:
        raise SystemExit("paired worker normalizers differ")
    normalizer = Normalizer.from_value(conditioned[0]["identity"]["normalizer"])
    motor_artifact = MotorArtifact.load(args.motor_organ.resolve())
    if motor_artifact.sha256 != collection.get("motor_artifact_sha256"):
        raise SystemExit("inherited motor artifact differs from collection")
    profile = EmbodiedTrainingProfile.from_value(manifest["profile"])
    native = native_extension_receipt()
    if native["sha256"] != COLLECTION_NATIVE:
        raise SystemExit("loaded native extension differs from collection")

    cases = []
    subset_receipts = {}
    for episode_number in args.episodes:
        episode = dataset.episodes[episode_number]
        birth_dir = dataset.path / f"episode-{episode_number:03d}-birth"
        birth_index = json.loads((birth_dir / "index.json").read_text())
        worlds_path = receipt_path(birth_dir, birth_index["worlds"], "birth worlds")
        private_path = receipt_path(
            birth_dir, birth_index["motor_private"], "birth motor private state",
        )
        private_birth = json.loads(private_path.read_text())
        if (
            private_birth.get("artifact_sha256") != motor_artifact.sha256
            or len(private_birth.get("motor", [])) != len(episode.world_slots)
        ):
            raise ValueError("birth motor private state differs")
        neural_receipt = birth_index["neural"]
        neural_path = birth_dir / f"{neural_receipt['name']}.npz"
        if neural_path.stat().st_size != neural_receipt["bytes"] or sha256(neural_path) != neural_receipt["sha256"]:
            raise ValueError("birth neural receipt differs")
        with gzip.open(worlds_path, "rt", encoding="utf-8") as handle:
            birth_worlds = json.load(handle)
        with np.load(episode.packet_path, allow_pickle=False) as packet:
            if "neural_readouts" not in packet.files:
                raise ValueError("closed-loop probe requires recorded neural readouts")
            recorded_neural = np.asarray(packet["neural_readouts"], dtype=np.float32)

        for world_slot in args.world_slots:
            columns = np.arange(world_slot * 3, world_slot * 3 + 3)
            if not np.array_equal(episode.world_slots[columns], np.full(3, world_slot)):
                raise ValueError("dataset resident partition ordering differs")
            pool = ProcessWorldPool(1, dict(ports.spec), profile.to_value(), args.physical_backend)
            brain = FixedCohortBrain(
                graph, ports, 3, device=args.device, backend=args.brain_backend,
                microbatch_size=1,
            )
            try:
                pool.call_all("restore", [copy.deepcopy(birth_worlds[world_slot])])
                subset, birth_subset_state, selected_keys = load_neural_subset(
                    neural_path, world_slot * 3, world_slot * 3 + 3,
                    str(graph.hash),
                )
                import_neural_subset(brain, birth_subset_state, selected_keys)
                subset_receipts[f"episode-{episode_number}/world-{world_slot}"] = subset

                for resident in args.resident_columns:
                    selected_column = int(columns[resident])
                    for prefix in args.prefixes:
                        for horizon in args.horizons:
                            if prefix + horizon >= len(episode.observations):
                                raise ValueError("prefix plus horizon exceeds episode")
                            # Each case replays from an independently restored birth.
                            pool.call_all("restore", [copy.deepcopy(birth_worlds[world_slot])])
                            restore_neural_state(brain, birth_subset_state)
                            previous = np.zeros(9, dtype=np.float32)
                            recurrent = {
                                (item["mode"], item["seed"]): {"hidden": None, "state": None}
                                for item in conditioned + goal_free
                            }
                            inherited = MotorOrgan.restore_value(
                                private_birth["motor"][selected_column], motor_artifact,
                            )
                            replay_residual = {"source": 0.0, "physiology": 0.0, "neural": 0.0}
                            source, local, features, bodies, normalized = observe(pool, brain, normalizer)
                            inherited_action = inherited.tick(
                                features[resident], local[resident], 0.05,
                            )
                            for item in conditioned + goal_free:
                                state, hidden = encode_worker_observation(
                                    item["worker"], normalized[resident], previous,
                                    None, True, torch, args.device,
                                )
                                recurrent[(item["mode"], item["seed"])] = {
                                    "state": state, "hidden": hidden,
                                }
                            for step in range(prefix):
                                actions = {
                                    body["id"]: action_mapping(
                                        episode.actions[step, columns[index]],
                                        episode.oral[step, columns[index]],
                                    )
                                    for index, body in enumerate(bodies)
                                }
                                pool.call_all("advance", [{"actions": actions, "dt": 0.05}])
                                previous = np.concatenate((
                                    episode.actions[step, selected_column],
                                    episode.oral[step, selected_column : selected_column + 1],
                                )).astype(np.float32)
                                source, local, features, bodies, normalized = observe(pool, brain, normalizer)
                                inherited_action = inherited.tick(
                                    features[resident], local[resident], 0.05,
                                )
                                replay_residual["source"] = max(
                                    replay_residual["source"],
                                    float(np.max(np.abs(source - episode.observations[step + 1, columns, :351]))),
                                )
                                replay_residual["physiology"] = max(
                                    replay_residual["physiology"],
                                    float(np.max(np.abs(local - episode.observations[step + 1, columns, 351:]))),
                                )
                                replay_residual["neural"] = max(
                                    replay_residual["neural"],
                                    float(np.max(np.abs(features - recorded_neural[step + 1, columns]))),
                                )
                                for item in conditioned + goal_free:
                                    key = (item["mode"], item["seed"])
                                    state, hidden = encode_worker_observation(
                                        item["worker"], normalized[resident], previous,
                                        recurrent[key]["hidden"], False,
                                        torch, args.device,
                                    )
                                    recurrent[key] = {"state": state, "hidden": hidden}

                            prefix_world = pool.call_all("snapshot")[0]
                            prefix_neural = copy_neural_state(brain)
                            prefix_recurrent = {
                                key: {
                                    "hidden": value["hidden"].clone(),
                                    "state": value["state"].clone(),
                                }
                                for key, value in recurrent.items()
                            }
                            prefix_inherited = inherited.snapshot_value(include_artifact=False)
                            prefix_inherited_action = np.asarray([
                                inherited_action[name] for name in ACTIONS
                            ], dtype=np.float32)
                            prefix_previous = previous.copy()
                            prefix_normalized = normalized.copy()
                            matched_window = goal_window(
                                episode, normalizer, prefix + horizon, selected_column,
                            )
                            alternate_end = alternate_goal_end(
                                prefix, horizon, len(episode.observations),
                            )
                            alternate_window = goal_window(
                                episode, normalizer, alternate_end, selected_column,
                            )
                            seed_results = {}
                            for seed in sorted(conditioned_by_seed):
                                conditioned_item = conditioned_by_seed[seed]
                                free_item = goal_free_by_seed[seed]
                                with torch.no_grad():
                                    matched_code = conditioned_item["encoder"].encode(
                                        torch.as_tensor(matched_window[None], device=args.device)
                                    )[0]
                                    alternate_code = conditioned_item["encoder"].encode(
                                        torch.as_tensor(alternate_window[None], device=args.device)
                                    )[0]
                                    free_code = torch.zeros_like(matched_code)

                                branch_results = {}
                                final_poses = {}
                                for condition in CONDITIONS:
                                    bodies = pool.call_all(
                                        "restore", [copy.deepcopy(prefix_world)],
                                    )[0]
                                    restore_neural_state(brain, prefix_neural)
                                    inherited_branch = MotorOrgan.restore_value(
                                        prefix_inherited, motor_artifact,
                                    )
                                    inherited_next = prefix_inherited_action.copy()
                                    active = (
                                        free_item if condition == "goal_free"
                                        else conditioned_item
                                    )
                                    active_key = (active["mode"], seed)
                                    hidden = prefix_recurrent[active_key]["hidden"].clone()
                                    state = prefix_recurrent[active_key]["state"].clone()
                                    previous = prefix_previous.copy()
                                    actual_windows = [prefix_normalized[resident].copy()]
                                    initial_body = copy.deepcopy(bodies[resident])
                                    initial_position = np.asarray([
                                        initial_body[key] for key in ("x", "y", "z")
                                    ], dtype=np.float64)
                                    previous_position = initial_position.copy()
                                    path_length = 0.0
                                    speeds = []
                                    contact_active = False
                                    mouth_active = False
                                    totals = {
                                        "generic_contact_ticks": 0,
                                        "generic_contact_bouts": 0,
                                        "mouth_contact_ticks": 0,
                                        "mouth_contact_bouts": 0,
                                        "ingested_mass": 0.0,
                                        "nutrition": 0.0,
                                        "effort_time_integral": 0.0,
                                        "action_l2_from_recorded": 0.0,
                                    }
                                    branch_replay_residual = {
                                        "source": 0.0, "physiology": 0.0, "neural": 0.0,
                                    }
                                    policy_goal = (
                                        alternate_code
                                        if condition == "goal_conditioned_alternate"
                                        else free_code if condition == "goal_free"
                                        else matched_code
                                    )
                                    target_window = (
                                        alternate_window
                                        if condition == "goal_conditioned_alternate"
                                        else matched_window
                                    )
                                    for offset in range(horizon):
                                        recorded = episode.actions[
                                            prefix + offset, selected_column
                                        ]
                                        if condition == "recorded_replay":
                                            selected_action = recorded
                                        elif condition == "repeat_last":
                                            selected_action = prefix_previous[:8]
                                        elif condition == "zero_action":
                                            selected_action = np.zeros(8, dtype=np.float32)
                                        elif condition == "inherited_4hz":
                                            selected_action = inherited_next
                                        else:
                                            remaining = horizon - offset
                                            horizon_value = math.log1p(remaining) / math.log(41)
                                            with torch.no_grad():
                                                logits = active["worker"].policy(
                                                    state, policy_goal,
                                                    torch.as_tensor(
                                                        horizon_value, device=args.device,
                                                    ),
                                                    torch.as_tensor(
                                                        previous[:8], device=args.device,
                                                    ),
                                                )
                                                selected_action = active["worker"].decode(
                                                    logits, mode="mode",
                                                ).cpu().numpy()
                                        actions = {}
                                        for index, body in enumerate(bodies):
                                            column = int(columns[index])
                                            action = (
                                                selected_action if index == resident
                                                else episode.actions[prefix + offset, column]
                                            )
                                            oral = (
                                                episode.oral[prefix + offset, column]
                                                if index != resident
                                                or condition == "recorded_replay"
                                                else oral_from_actual(body)
                                            )
                                            actions[body["id"]] = action_mapping(action, oral)
                                        advanced = pool.call_all(
                                            "advance", [{"actions": actions, "dt": 0.05}],
                                        )[0][0]
                                        selected_outcome = advanced[bodies[resident]["id"]]
                                        contact_now = selected_outcome["contact"] > 0
                                        mouth_now = selected_outcome.get(
                                            "mouth_material_contacts", 0,
                                        ) > 0
                                        totals["generic_contact_ticks"] += int(contact_now)
                                        totals["mouth_contact_ticks"] += int(mouth_now)
                                        totals["generic_contact_bouts"] += int(
                                            contact_now and not contact_active
                                        )
                                        totals["mouth_contact_bouts"] += int(
                                            mouth_now and not mouth_active
                                        )
                                        contact_active, mouth_active = contact_now, mouth_now
                                        totals["ingested_mass"] += float(
                                            selected_outcome.get("ingested_mass", 0.0)
                                        )
                                        totals["nutrition"] += float(
                                            selected_outcome["nutrition"]
                                        )
                                        totals["effort_time_integral"] += 0.05 * float(
                                            selected_outcome["effort"]
                                        )
                                        totals["action_l2_from_recorded"] += float(
                                            np.linalg.norm(
                                                np.asarray(selected_action) - recorded
                                            )
                                        )
                                        oral_executed = actions[bodies[resident]["id"]]["eat"]
                                        previous = np.concatenate((
                                            np.asarray(selected_action, dtype=np.float32),
                                            np.asarray([oral_executed], dtype=np.float32),
                                        ))
                                        source, local, features, bodies, normalized = observe(
                                            pool, brain, normalizer,
                                        )
                                        inherited_value = inherited_branch.tick(
                                            features[resident], local[resident], 0.05,
                                        )
                                        inherited_next = np.asarray([
                                            inherited_value[name] for name in ACTIONS
                                        ], dtype=np.float32)
                                        position = np.asarray([
                                            bodies[resident][key] for key in ("x", "y", "z")
                                        ], dtype=np.float64)
                                        path_length += float(np.linalg.norm(
                                            position - previous_position
                                        ))
                                        previous_position = position
                                        speeds.append(float(bodies[resident]["speed"]))
                                        if condition == "recorded_replay":
                                            recorded_step = prefix + offset + 1
                                            for name, actual, expected in (
                                                ("source", source, episode.observations[
                                                    recorded_step, columns, :351
                                                ]),
                                                ("physiology", local, episode.observations[
                                                    recorded_step, columns, 351:
                                                ]),
                                                ("neural", features, recorded_neural[
                                                    recorded_step, columns
                                                ]),
                                            ):
                                                branch_replay_residual[name] = max(
                                                    branch_replay_residual[name],
                                                    float(np.max(np.abs(actual - expected))),
                                                )
                                        actual_windows.append(normalized[resident].copy())
                                        state, hidden = encode_worker_observation(
                                            active["worker"], normalized[resident], previous,
                                            hidden, False, torch, args.device,
                                        )
                                    actual_window = np.stack(actual_windows[-4:])
                                    with torch.no_grad():
                                        actual_code = conditioned_item["encoder"].encode(
                                            torch.as_tensor(
                                                actual_window[None], device=args.device,
                                            )
                                        )[0]
                                        scored_target_code = (
                                            alternate_code
                                            if condition == "goal_conditioned_alternate"
                                            else matched_code
                                        )
                                    final_snapshot = pool.call_all("snapshot")[0]
                                    final_poses[condition] = body_pose(
                                        final_snapshot, resident,
                                    )
                                    final_body = bodies[resident]
                                    branch_results[condition] = {
                                        **totals,
                                        "primary_normalized_goal_window_rmse": float(np.sqrt(
                                            np.mean((actual_window - target_window) ** 2)
                                        )),
                                        "primary_goal_code_l2": float(torch.linalg.vector_norm(
                                            actual_code - scored_target_code
                                        ).cpu()),
                                        "matched_normalized_goal_window_rmse": float(
                                            np.sqrt(np.mean(
                                                (actual_window - matched_window) ** 2
                                            ))
                                        ),
                                        "alternate_normalized_goal_window_rmse": float(
                                            np.sqrt(np.mean(
                                                (actual_window - alternate_window) ** 2
                                            ))
                                        ),
                                        "matched_goal_code_l2": float(
                                            torch.linalg.vector_norm(
                                                actual_code - matched_code
                                            ).cpu()
                                        ),
                                        "alternate_goal_code_l2": float(
                                            torch.linalg.vector_norm(
                                                actual_code - alternate_code
                                            ).cpu()
                                        ),
                                        "net_displacement_m": float(np.linalg.norm(
                                            previous_position - initial_position
                                        )),
                                        "path_length_m": path_length,
                                        "final_speed_m_per_s": speeds[-1],
                                        "terminal_four_tick_mean_speed_m_per_s": float(
                                            np.mean(speeds[-4:])
                                        ),
                                        "stopped_tick_fraction_speed_below_0_01": float(
                                            np.mean(np.asarray(speeds) < 0.01)
                                        ),
                                        "energy_delta": float(
                                            final_body["energy"] - initial_body["energy"]
                                        ),
                                        "gut_delta": float(
                                            final_body["gut"] - initial_body["gut"]
                                        ),
                                        "fatigue_recovery": float(
                                            initial_body["fatigue"] - final_body["fatigue"]
                                        ),
                                    }
                                    if condition == "recorded_replay":
                                        branch_results[condition][
                                            "closed_loop_replay_residual_max_abs"
                                        ] = branch_replay_residual
                                target_pose = final_poses["recorded_replay"]
                                for condition in CONDITIONS:
                                    pose = final_poses[condition]
                                    branch_results[condition][
                                        "analyst_position_error_m"
                                    ] = float(np.linalg.norm(
                                        np.asarray(pose["position"])
                                        - np.asarray(target_pose["position"])
                                    ))
                                    branch_results[condition][
                                        "analyst_orientation_error_rad"
                                    ] = quaternion_error(
                                        pose["quaternion"], target_pose["quaternion"],
                                    )
                                seed_results[str(seed)] = branch_results
                            cases.append({
                                "episode": episode_number,
                                "world_slot": world_slot,
                                "resident_column_within_world": resident,
                                "prefix": prefix,
                                "horizon": horizon,
                                "alternate_goal_recorded_end": alternate_end,
                                "prefix_replay_residual_max_abs": replay_residual,
                                "paired_seed_conditions": seed_results,
                            })
            finally:
                pool.close()

    result = {
        "format": "chreatures-sensorimotor-worker-closed-loop-probe-v2",
        "status": "research capability probe; no behavioral promotion",
        "protocol": {
            "conditions": list(CONDITIONS),
            "selected_oral": (
                "actual float32 clip((1-gut)*(1.1-energy),0,1), except stored oral "
                "in recorded replay"
            ),
            "other_bodies": "stored recorded eight-axis and oral commands",
            "selected_history": (
                "worker recurrence advances only from fresh counterfactual observations "
                "and the selected body's actually executed eight-axis plus oral command"
            ),
            "goal_semantics": (
                "frozen encoder of four recorded observations ending at t+k; probe target "
                "only, never experienced recurrent history or manager memory"
            ),
            "alternate_goal_semantics": (
                "same resident and episode, ending at t+2k when available (otherwise "
                "t-k), fixed for the branch"
            ),
            "goal_free_semantics": (
                "paired v2 goal-free worker receives an exact zero 64-vector; attainment "
                "is scored against the matched recorded goal by its paired conditioned encoder"
            ),
            "stopping_threshold_m_per_s": 0.01,
            "encounter_bout": (
                "one maximal contiguous run of positive per-tick contact; ticks are not meals"
            ),
            "inherited_4hz_context": (
                "restored collected birth state and advanced through source history; its "
                "private context follows inherited proposals, as in collection, while the "
                "recorded delivered exploration remains authoritative"
            ),
            "analyst_pose": "derived from snapshots only; never a model input",
        },
        "identity": {
            "dataset_content_sha256": manifest["content_sha256"],
            "collection_identity_sha256": dataset.identity.sha256,
            "paired_workers": [
                {
                    "seed": seed,
                    "goal_conditioned_sha256": conditioned_by_seed[seed]["sha256"],
                    "goal_free_sha256": goal_free_by_seed[seed]["sha256"],
                }
                for seed in sorted(conditioned_by_seed)
            ],
            "worker_contract": WORKER_FORMAT,
            "graph_sha256": str(graph.hash),
            "port_spec_sha256": ports.spec_hash,
            "port_bundle_sha256": sha256(args.port_bundle.resolve()),
            "motor_artifact_sha256": motor_artifact.sha256,
            "training_profile_sha256": profile.sha256,
            "native_sha256": native["sha256"],
            "frozen_physical_source_lineage": "78edd56+0d9b9dd (declared collection lineage)",
            "probe_source_sha256": sha256(Path(__file__)),
        },
        "neural_subsets": subset_receipts,
        "cases": cases,
        "limitations": [
            "B48 birth columns are independent and sliced into B3, but replay residuals rather than an assumed exact equivalence establish numerical fidelity.",
            "A recorded achieved goal may no longer be reachable from a counterfactual prefix and does not imply stopping competence.",
            "Other residents follow recorded open-loop commands and do not react to the selected resident's counterfactual behavior.",
            "Energy and gut capability, generalization, and long-term regulation are not established by this probe.",
        ],
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "cases": len(cases), "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
