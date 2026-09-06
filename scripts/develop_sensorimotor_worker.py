#!/usr/bin/env python3
"""Online developmental PPO for the research achieved-history worker.

This runner joins a fixed full-MaleCNS circuit, shared chemical 3-D worlds,
private recurrent worker state, and physical homeostasis.  It changes inherited
shared worker weights only at explicit rollout boundaries.  Goals are sampled
from each resident's own already-achieved four-frame sensory history.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.homeostasis import FiniteEnergyConfig, FiniteEnergyObjective
from research.sensorimotor_skills.data import Normalizer
from research.sensorimotor_skills.model import GoalEncoder, SensorimotorWorker
from research.sensorimotor_skills.online_model import SlowGoalManager

FORMAT = "chreatures-online-sensorimotor-development-v1"
ACTIONS = (
    "thrust",
    "yaw",
    "gaze_pitch",
    "grip",
    "signal_low",
    "signal_mid",
    "signal_high",
    "posture",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument(
        "--chemical-habitat",
        type=Path,
        default=ROOT / "data/habitats/chemical-reef-v3.json",
    )
    parser.add_argument(
        "--chemical-biosphere",
        type=Path,
        default=ROOT / "data/biosphere/chemical-reef-v3.json",
    )
    parser.add_argument(
        "--chemical-conditions",
        type=Path,
        default=ROOT / "data/training/chemical-resource-encounters-v1.json",
    )
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_480)
    parser.add_argument("--episode-steps", type=int, default=2_048)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--goal-sticky-steps", type=int, default=10)
    parser.add_argument("--goal-reservoir-size", type=int, default=128)
    parser.add_argument("--goal-progress-coefficient", type=float, default=0.01)
    parser.add_argument("--checkpoint-every-updates", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.002)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--brain-backend", choices=("tiled", "triton"), default="tiled")
    parser.add_argument(
        "--physical-backend", choices=("fast", "reference"), default="fast"
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not 1 <= args.worlds <= 16 or args.steps < args.rollout_steps:
        raise SystemExit("worlds must be 1..16 and steps at least one rollout")
    if args.steps % args.rollout_steps or args.episode_steps % args.rollout_steps:
        raise SystemExit("steps and episode steps must be divisible by rollout steps")
    if args.episode_steps < args.rollout_steps or not 1 <= args.ppo_epochs <= 16:
        raise SystemExit("invalid episode or PPO schedule")
    if args.goal_sticky_steps < 1 or args.goal_reservoir_size < 4:
        raise SystemExit("invalid private goal schedule")
    if not 0 <= args.goal_progress_coefficient <= 1:
        raise SystemExit("goal progress coefficient must be in [0,1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")


def load_bootstrap(path: Path, device: torch.device):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        value = torch.load(path, map_location=device, weights_only=True)
    if value.get("format") != "chreatures-sensorimotor-worker-research-v2":
        raise ValueError("bootstrap worker format differs")
    identity = value["identity"]
    encoder = GoalEncoder().to(device)
    worker = SensorimotorWorker().to(device)
    encoder.load_state_dict(value["goal_encoder"], strict=True)
    worker.load_state_dict(value["worker"], strict=True)
    encoder.freeze()
    normalizer = Normalizer.from_value(identity["normalizer"])
    return encoder, worker, normalizer, identity


def physiology(bodies, circuit: np.ndarray) -> np.ndarray:
    rows = []
    index = 0
    for world in bodies:
        for body in world:
            rows.append(
                (
                    body["energy"],
                    body["gut"],
                    body["fatigue"],
                    math.tanh(float(body["speed"]) / 2),
                    math.tanh(float(body["angular_velocity"]) / 4),
                    circuit[index, 2],
                )
            )
            index += 1
    return np.asarray(rows, dtype=np.float32)


def observe(pool, brain, normalizer):
    result = pool.call_all("observe")
    source = np.concatenate([item[0] for item in result]).astype(np.float32)
    bodies = [item[1] for item in result]
    neural, circuit, _ = brain.step_channels(source, 0.05)
    physical = physiology(bodies, circuit)
    raw = np.concatenate((source, physical), axis=1)
    return normalizer.normalize(raw).astype(np.float32), physical, neural, bodies


def oral(body) -> float:
    return float(np.clip((1.0 - body["gut"]) * (1.1 - body["energy"]), 0, 1))


def action_payload(actions: np.ndarray, bodies) -> list[dict[str, object]]:
    payloads = []
    row = 0
    for world in bodies:
        mapped = {}
        for body in world:
            values = dict(
                zip(ACTIONS, actions[row].astype(float).tolist(), strict=True)
            )
            values["eat"] = oral(body)
            mapped[str(body["id"])] = values
            row += 1
        payloads.append({"actions": mapped, "dt": 0.05})
    return payloads


def physical_reward(objective, before, after_bodies, outcomes):
    after = np.asarray(
        [
            (body["energy"], body["gut"], body["fatigue"])
            for world in after_bodies
            for body in world
        ],
        dtype=np.float32,
    )
    nutrition, effort = [], []
    for world_bodies, world_outcomes in zip(after_bodies, outcomes, strict=True):
        for body in world_bodies:
            outcome = world_outcomes[body["id"]]
            nutrition.append(outcome["nutrition"])
            effort.append(outcome["effort"])
    reward, terms = objective.transition(
        before[:, :3],
        after,
        nutrition=nutrition,
        effort=effort,
        dt=0.05,
    )
    return reward, terms


class ResidentGoals:
    """Private achieved-window reservoir and sticky-goal RNG for each resident."""

    def __init__(self, count: int, size: int, sticky: int, seed: int):
        self.size = size
        self.sticky = sticky
        self.rng = [
            np.random.default_rng(seed ^ (index * 0x9E3779B1)) for index in range(count)
        ]
        self.frames = [deque(maxlen=4) for _ in range(count)]
        self.reservoirs = [[] for _ in range(count)]
        self.seen = np.zeros(count, dtype=np.int64)
        self.codes = np.zeros((count, 64), dtype=np.float32)
        self.horizons = np.zeros((count, 1), dtype=np.float32)
        self.age = np.zeros(count, dtype=np.int64)

    @torch.no_grad()
    def record(self, observation: np.ndarray, encoder, device) -> np.ndarray:
        windows, indices = [], []
        for index, row in enumerate(observation):
            self.frames[index].append(row.copy())
            if len(self.frames[index]) == 4:
                windows.append(np.stack(self.frames[index]))
                indices.append(index)
        current = np.zeros((len(observation), 64), dtype=np.float32)
        if windows:
            encoded = (
                encoder.encode(torch.as_tensor(np.stack(windows), device=device))
                .cpu()
                .numpy()
            )
            for index, code in zip(indices, encoded, strict=True):
                current[index] = code
                self.seen[index] += 1
                record = (code.copy(), int(self.seen[index]))
                reservoir = self.reservoirs[index]
                if len(reservoir) < self.size:
                    reservoir.append(record)
                else:
                    slot = int(self.rng[index].integers(self.seen[index]))
                    if slot < self.size:
                        reservoir[slot] = record
            self.age += 1
        return current

    def manager_batch(self) -> tuple[np.ndarray, np.ndarray]:
        keys = np.zeros((len(self.reservoirs), self.size, 64), dtype=np.float32)
        valid = np.zeros(keys.shape[:2], dtype=bool)
        for resident, reservoir in enumerate(self.reservoirs):
            for slot, (code, _) in enumerate(reservoir):
                keys[resident, slot] = code
                valid[resident, slot] = True
        return keys, valid

    def apply_selection(self, selected: np.ndarray) -> None:
        for resident, slot in enumerate(selected):
            code, achieved = self.reservoirs[resident][int(slot)]
            self.codes[resident] = code
            lag = max(1, int(self.seen[resident]) - achieved)
            self.horizons[resident, 0] = math.log1p(lag) / math.log(41)

    def reset(self) -> None:
        for frames, reservoir in zip(self.frames, self.reservoirs, strict=True):
            frames.clear()
            reservoir.clear()
        self.seen.fill(0)
        self.codes.fill(0)
        self.horizons.fill(0)
        self.age.fill(0)

    def value(self) -> dict[str, object]:
        return {
            "rng": [copy.deepcopy(value.bit_generator.state) for value in self.rng],
            "frames": [[row.tolist() for row in value] for value in self.frames],
            "reservoirs": [
                [(code.tolist(), tick) for code, tick in value]
                for value in self.reservoirs
            ],
            "seen": self.seen.tolist(),
            "codes": self.codes.tolist(),
            "horizons": self.horizons.tolist(),
            "age": self.age.tolist(),
        }


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128 + 64 + 1, 256), nn.Tanh(), nn.Linear(256, 1)
        )

    def forward(self, state, goal, horizon):
        return self.network(torch.cat((state, goal, horizon), -1)).squeeze(-1)


def policy_terms(worker, logits, action):
    log_probability = -worker.action_nll(logits, action).sum(-1)
    signed = torch.distributions.Categorical(logits=logits["signed"]).entropy().sum(-1)
    active = torch.distributions.Bernoulli(logits=logits["active"])
    positive = torch.distributions.Categorical(logits=logits["positive"]).entropy()
    entropy = signed + active.entropy().sum(-1) + (active.probs * positive).sum(-1)
    return log_probability, entropy


def reset_worlds(pool, brain, worlds: int, seed: int, episode: int):
    bodies = pool.call_all(
        "reset",
        [
            {
                "seed": seed + episode * 1009 + index,
                "held_out": False,
                "stage": min(2, episode),
            }
            for index in range(worlds)
        ],
    )
    if brain.resident_ids:
        brain.remove_residents(brain.resident_ids)
    identities = [
        f"episode-{episode:04d}/world-{world:02d}/resident-{resident}"
        for world in range(worlds)
        for resident in range(3)
    ]
    brain.add_residents(identities)
    return bodies


def save_development(
    path,
    *,
    identity,
    worker,
    encoder,
    critic,
    manager,
    optimizer,
    hidden,
    previous,
    goals,
    manager_events,
    active_manager,
    pool,
    brain,
    episode,
    physical_steps,
    updates,
):
    """Atomically save coherent world, neural, optimizer, and private state."""
    value = {
        "format": FORMAT,
        "identity": identity,
        "updates": updates,
        "physical_steps": physical_steps,
        "episode": episode,
        "worker": worker.state_dict(),
        "goal_encoder": encoder.state_dict(),
        "critic": critic.state_dict(),
        "goal_manager": manager.state_dict(),
        "optimizer": optimizer.state_dict(),
        "private_worker_hidden": hidden.cpu(),
        "private_previous_action": previous,
        "private_goals": goals.value(),
        "private_manager_events": manager_events,
        "private_active_manager": active_manager,
        "worlds": pool.call_all("snapshot"),
        "neural": {
            key: value.copy() for key, value in brain.circuit.export_state().items()
        },
        "neural_resident_ids": list(brain.resident_ids),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "pending_rollout": {"length": 0},
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> int:
    args = arguments()
    validate(args)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    from chreatures.neural_ports import NeuralPortBundle
    from chreatures.training_environment import EmbodiedTrainingProfile
    from scripts.learn_affordances import FixedCohortBrain, ProcessWorldPool, load_graph

    encoder, worker, normalizer, bootstrap_identity = load_bootstrap(
        args.bootstrap_worker.resolve(),
        device,
    )
    profile = EmbodiedTrainingProfile.chemical_encounters(
        args.chemical_habitat,
        args.chemical_biosphere,
        args.chemical_conditions,
    )
    objective = FiniteEnergyObjective(
        FiniteEnergyConfig.from_value(profile.component("homeostasis"))
    )
    graph = load_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    if len(ports.input_names) != 351:
        raise ValueError("development v1 requires the pinned 351 sensory channels")
    count = args.worlds * 3
    brain = FixedCohortBrain(
        graph,
        ports,
        count,
        device=args.device,
        backend=args.brain_backend,
        microbatch_size=1,
    )
    pool = ProcessWorldPool(
        args.worlds,
        dict(ports.spec),
        profile.to_value(),
        args.physical_backend,
    )
    critic = Critic().to(device)
    manager = SlowGoalManager().to(device)
    optimizer = torch.optim.AdamW(
        [*worker.parameters(), *critic.parameters(), *manager.parameters()],
        lr=args.learning_rate,
    )
    goals = ResidentGoals(
        count,
        args.goal_reservoir_size,
        args.goal_sticky_steps,
        args.seed ^ 0x60A1,
    )
    identity = {
        "format": FORMAT,
        "seed": args.seed,
        "dt_seconds": 0.05,
        "action_order": list(ACTIONS),
        "oral_action": "supplied body physiology law",
        "graph_sha256": str(graph.hash),
        "port_spec_sha256": ports.spec_hash,
        "port_bundle_sha256": sha256(args.port_bundle.resolve()),
        "profile": profile.to_value(),
        "bootstrap_path": str(args.bootstrap_worker.resolve()),
        "bootstrap_sha256": sha256(args.bootstrap_worker.resolve()),
        "bootstrap_identity": bootstrap_identity,
        "arguments": vars(args)
        | {
            key: str(value)
            for key, value in vars(args).items()
            if isinstance(value, Path)
        },
    }
    atomic_json(args.output / "identity.json", identity)
    episode = 0
    reset_worlds(pool, brain, args.worlds, args.seed, episode)
    observation, physical, neural, bodies = observe(pool, brain, normalizer)
    previous = np.zeros((count, 9), dtype=np.float32)
    hidden = torch.zeros((1, count, 128), device=device)
    current_code = goals.record(observation, encoder, device)
    manager_events: list[dict[str, object]] = []
    active_manager = np.full(count, -1, dtype=np.int64)
    updates = []
    started = time.perf_counter()
    try:
        for rollout_start in range(0, args.steps, args.rollout_steps):
            arrays = {
                name: []
                for name in (
                    "observation",
                    "previous",
                    "reset",
                    "goal",
                    "horizon",
                    "action",
                    "old_log_probability",
                    "value",
                    "reward",
                    "physical_reward",
                    "goal_progress_reward",
                    "done",
                )
            }
            initial_hidden = hidden.detach().clone()
            for local in range(args.rollout_steps):
                global_step = rollout_start + local
                reset = np.zeros(count, dtype=bool)
                with torch.no_grad():
                    state, next_hidden = worker.encode_sequence(
                        torch.as_tensor(observation[None], device=device),
                        torch.as_tensor(previous[None], device=device),
                        hidden,
                        torch.as_tensor(reset[None], device=device),
                    )
                    # Slow selection uses the hidden state after encoding the
                    # current observation, plus current neural and body state.
                    # It occurs only after the preceding transition's credit.
                    if global_step % args.goal_sticky_steps == 0 and all(
                        goals.reservoirs
                    ):
                        keys, valid = goals.manager_batch()
                        manager_logits, manager_value = manager.policy(
                            next_hidden[0],
                            torch.as_tensor(neural, device=device),
                            torch.as_tensor(physical, device=device),
                            torch.as_tensor(keys, device=device),
                            torch.as_tensor(valid, device=device),
                        )
                        selected = torch.distributions.Categorical(
                            logits=manager_logits
                        ).sample()
                        selected_logp = F.log_softmax(manager_logits, -1).gather(
                            1, selected[:, None]
                        )[:, 0]
                        for resident in range(count):
                            if active_manager[resident] >= 0:
                                manager_events[active_manager[resident]]["complete"] = (
                                    True
                                )
                            manager_events.append(
                                {
                                    "resident": resident,
                                    "hidden": next_hidden[0, resident].cpu().numpy(),
                                    "neural": neural[resident].copy(),
                                    "physiology": physical[resident].copy(),
                                    "keys": keys[resident].copy(),
                                    "valid": valid[resident].copy(),
                                    "selected": int(selected[resident]),
                                    "old_logp": float(selected_logp[resident]),
                                    "old_value": float(manager_value[resident]),
                                    "return": 0.0,
                                    "discount": 1.0,
                                    "steps": 0,
                                    "complete": False,
                                }
                            )
                            active_manager[resident] = len(manager_events) - 1
                        goals.apply_selection(selected.cpu().numpy())
                    goal_tensor = torch.as_tensor(goals.codes, device=device)
                    horizon = torch.as_tensor(goals.horizons, device=device)
                    logits = worker.policy(
                        state[0],
                        goal_tensor,
                        horizon,
                        torch.as_tensor(previous[:, :8], device=device),
                    )
                    action = worker.decode(logits, mode="sample")
                    log_probability, _ = policy_terms(worker, logits, action)
                    value = critic(state[0], goal_tensor, horizon)
                action_np = action.cpu().numpy()
                executed_oral = np.asarray(
                    [oral(body) for world in bodies for body in world],
                    dtype=np.float32,
                )
                before = physical.copy()
                advanced = pool.call_all("advance", action_payload(action_np, bodies))
                outcomes = [item[0] for item in advanced]
                bodies = [item[1] for item in advanced]
                base_reward, _ = physical_reward(objective, before, bodies, outcomes)
                next_observation, physical, next_neural, bodies = observe(
                    pool, brain, normalizer
                )
                # Credit the whole transition against the pre-action sticky goal.
                frozen_goal = goals.codes.copy()
                next_code = goals.record(next_observation, encoder, device)
                old_distance = np.linalg.norm(current_code - frozen_goal, axis=1)
                new_distance = np.linalg.norm(next_code - frozen_goal, axis=1)
                progress = args.goal_progress_coefficient * (
                    old_distance - new_distance
                )
                reward = base_reward + progress.astype(np.float32)
                for resident, event_index in enumerate(active_manager):
                    if event_index >= 0:
                        event = manager_events[event_index]
                        event["return"] += event["discount"] * float(reward[resident])
                        event["discount"] *= args.discount
                        event["steps"] += 1
                done = np.full(count, (global_step + 1) % args.episode_steps == 0)
                for name, value_np in (
                    ("observation", observation),
                    ("previous", previous),
                    ("reset", reset),
                    ("goal", goals.codes.copy()),
                    ("horizon", goals.horizons.copy()),
                    ("action", action_np),
                    ("old_log_probability", log_probability.cpu().numpy()),
                    ("value", value.cpu().numpy()),
                    ("reward", reward),
                    ("physical_reward", base_reward),
                    ("goal_progress_reward", progress),
                    ("done", done),
                ):
                    arrays[name].append(np.asarray(value_np))
                previous = np.concatenate(
                    (
                        action_np,
                        executed_oral[:, None],
                    ),
                    axis=1,
                ).astype(np.float32)
                observation = next_observation
                neural = next_neural
                current_code = next_code
                hidden = next_hidden
                if done[0]:
                    episode += 1
                    reset_worlds(pool, brain, args.worlds, args.seed, episode)
                    goals.reset()
                    observation, physical, neural, bodies = observe(
                        pool, brain, normalizer
                    )
                    previous.fill(0)
                    hidden.zero_()
                    current_code = goals.record(observation, encoder, device)
                    for resident, event_index in enumerate(active_manager):
                        if event_index >= 0:
                            manager_events[event_index]["complete"] = True
                    active_manager.fill(-1)
            batch = {key: np.stack(value) for key, value in arrays.items()}
            with torch.no_grad():
                next_state, _ = worker.encode_sequence(
                    torch.as_tensor(observation[None], device=device),
                    torch.as_tensor(previous[None], device=device),
                    hidden,
                    torch.zeros((1, count), dtype=torch.bool, device=device),
                )
                bootstrap = (
                    critic(
                        next_state[0],
                        torch.as_tensor(goals.codes, device=device),
                        torch.as_tensor(goals.horizons, device=device),
                    )
                    .cpu()
                    .numpy()
                )
            advantage = np.zeros_like(batch["reward"])
            carry = np.zeros(count, dtype=np.float32)
            for index in reversed(range(args.rollout_steps)):
                next_value = (
                    bootstrap
                    if index + 1 == args.rollout_steps
                    else batch["value"][index + 1]
                )
                live = 1.0 - batch["done"][index].astype(np.float32)
                delta = (
                    batch["reward"][index]
                    + args.discount * next_value * live
                    - batch["value"][index]
                )
                carry = delta + args.discount * args.gae_lambda * live * carry
                advantage[index] = carry
            returns = advantage + batch["value"]
            advantage = (advantage - advantage.mean()) / max(
                float(advantage.std()), 1e-6
            )
            metrics = None
            worker.train()
            for _ in range(args.ppo_epochs):
                states, _ = worker.encode_sequence(
                    torch.as_tensor(batch["observation"], device=device),
                    torch.as_tensor(batch["previous"], device=device),
                    initial_hidden,
                    torch.as_tensor(batch["reset"], device=device),
                )
                goal_tensor = torch.as_tensor(batch["goal"], device=device)
                horizon = torch.as_tensor(batch["horizon"], device=device)
                logits = worker.policy(
                    states,
                    goal_tensor,
                    horizon,
                    torch.as_tensor(batch["previous"][..., :8], device=device),
                )
                new_logp, entropy = policy_terms(
                    worker,
                    logits,
                    torch.as_tensor(batch["action"], device=device),
                )
                old_logp = torch.as_tensor(batch["old_log_probability"], device=device)
                adv = torch.as_tensor(advantage, device=device)
                ratio = (new_logp - old_logp).exp()
                policy_loss = -torch.minimum(
                    ratio * adv,
                    ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio) * adv,
                ).mean()
                predicted_value = critic(states, goal_tensor, horizon)
                value_loss = F.mse_loss(
                    predicted_value, torch.as_tensor(returns, device=device)
                )
                loss = (
                    policy_loss
                    + args.value_coefficient * value_loss
                    - args.entropy_coefficient * entropy.mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [*worker.parameters(), *critic.parameters()], 1.0
                )
                optimizer.step()
                metrics = {
                    "loss": float(loss.detach()),
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.mean().detach()),
                }
            complete_events = [event for event in manager_events if event["complete"]]
            manager_metrics = {"manager_events": 0, "manager_policy_loss": 0.0}
            if complete_events:
                manager_logits, manager_values = manager.policy(
                    torch.as_tensor(
                        np.stack([event["hidden"] for event in complete_events]),
                        device=device,
                    ),
                    torch.as_tensor(
                        np.stack([event["neural"] for event in complete_events]),
                        device=device,
                    ),
                    torch.as_tensor(
                        np.stack([event["physiology"] for event in complete_events]),
                        device=device,
                    ),
                    torch.as_tensor(
                        np.stack([event["keys"] for event in complete_events]),
                        device=device,
                    ),
                    torch.as_tensor(
                        np.stack([event["valid"] for event in complete_events]),
                        device=device,
                    ),
                )
                chosen = torch.as_tensor(
                    [event["selected"] for event in complete_events], device=device
                )
                new_logp = F.log_softmax(manager_logits, -1).gather(1, chosen[:, None])[
                    :, 0
                ]
                old_logp = torch.as_tensor(
                    [event["old_logp"] for event in complete_events], device=device
                )
                manager_return = torch.as_tensor(
                    [event["return"] for event in complete_events], device=device
                )
                manager_advantage = manager_return - torch.as_tensor(
                    [event["old_value"] for event in complete_events], device=device
                )
                ratio = (new_logp - old_logp).exp()
                manager_policy = -torch.minimum(
                    ratio * manager_advantage,
                    ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio)
                    * manager_advantage,
                ).mean()
                manager_value_loss = F.mse_loss(manager_values, manager_return)
                manager_entropy = (
                    torch.distributions.Categorical(logits=manager_logits)
                    .entropy()
                    .mean()
                )
                manager_loss = (
                    manager_policy
                    + args.value_coefficient * manager_value_loss
                    - args.entropy_coefficient * manager_entropy
                )
                optimizer.zero_grad(set_to_none=True)
                manager_loss.backward()
                nn.utils.clip_grad_norm_(manager.parameters(), 1.0)
                optimizer.step()
                manager_metrics = {
                    "manager_events": len(complete_events),
                    "manager_policy_loss": float(manager_policy.detach()),
                    "manager_value_loss": float(manager_value_loss.detach()),
                    "manager_entropy": float(manager_entropy.detach()),
                }
                manager_events = [
                    event for event in manager_events if not event["complete"]
                ]
                active_manager.fill(-1)
                for index, event in enumerate(manager_events):
                    active_manager[int(event["resident"])] = index
            worker.eval()
            metrics.update(manager_metrics)
            metrics.update(
                {
                    "update": len(updates) + 1,
                    "physical_steps": rollout_start + args.rollout_steps,
                    "physical_reward_mean": float(batch["physical_reward"].mean()),
                    "goal_progress_reward_mean": float(
                        batch["goal_progress_reward"].mean()
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            updates.append(metrics)
            with (args.output / "updates.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if (
                args.checkpoint_every_updates
                and len(updates) % args.checkpoint_every_updates == 0
            ):
                save_development(
                    args.output / f"checkpoint-update-{len(updates):06d}.pt",
                    identity=identity,
                    worker=worker,
                    encoder=encoder,
                    critic=critic,
                    manager=manager,
                    optimizer=optimizer,
                    hidden=hidden,
                    previous=previous,
                    goals=goals,
                    manager_events=manager_events,
                    active_manager=active_manager,
                    pool=pool,
                    brain=brain,
                    episode=episode,
                    physical_steps=rollout_start + args.rollout_steps,
                    updates=len(updates),
                )
        save_development(
            args.output / "development.pt",
            identity=identity,
            worker=worker,
            encoder=encoder,
            critic=critic,
            manager=manager,
            optimizer=optimizer,
            hidden=hidden,
            previous=previous,
            goals=goals,
            manager_events=manager_events,
            active_manager=active_manager,
            pool=pool,
            brain=brain,
            episode=episode,
            physical_steps=args.steps,
            updates=len(updates),
        )
        atomic_json(
            args.output / "result.json",
            {
                "format": FORMAT,
                "status": "research joined-development run; no behavior claim",
                "physical_steps": args.steps,
                "updates": len(updates),
                "last_update": updates[-1],
                "elapsed_seconds": time.perf_counter() - started,
                "artifact_sha256": sha256(args.output / "development.pt"),
                "world_transport_timing": (
                    pool.timing_snapshot() if hasattr(pool, "timing_snapshot") else None
                ),
            },
        )
    finally:
        pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
