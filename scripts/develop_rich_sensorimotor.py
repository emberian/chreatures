#!/usr/bin/env python3
"""Online developmental PPO for the breaking rich achieved-history worker.

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
from chreatures.organism_interface import (
    ACTION_DIM,
    ACTION_NAMES,
    MAX_RESIDENTS,
    PHYSIOLOGY_DIM,
)
from research.sensorimotor_skills.rich_online import (
    SlowGoalManager,
    cold_inherit_v3_manager,
    sample_worker_actions,
)
from research.sensorimotor_skills.rich_data import RichNormalizer
from research.sensorimotor_skills.rich_model import (
    RICH_CHANNEL_NAMES_SHA256,
    RICH_PROFILE_SHA256,
    PopulationAdapterBank,
    RichSensorimotorModel,
    cold_inherit_v3_model,
)

FORMAT = "chreatures-rich-online-sensorimotor-development-v4"


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


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-worker", type=Path, required=True)
    parser.add_argument("--candidate-genomes", type=Path, required=True)
    parser.add_argument(
        "--neural-recipe",
        type=Path,
        default=ROOT / "data/ports/neural-variant-canonical-v1.json",
    )
    parser.add_argument("--cold-inherit-v3", action="store_true")
    parser.add_argument("--initialize-from-development", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument(
        "--chemical-habitat",
        type=Path,
        default=ROOT / "data/habitats/living-reef.json",
    )
    parser.add_argument(
        "--chemical-biosphere",
        type=Path,
        default=ROOT / "data/biosphere/living-reef.json",
    )
    parser.add_argument("--nursery-family-config", type=Path, required=True)
    parser.add_argument("--nursery-family-schedule", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--residents-per-world", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20_480)
    parser.add_argument("--episode-steps", type=int, default=2_048)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--goal-sticky-steps", type=int, default=10)
    parser.add_argument("--goal-reservoir-size", type=int, default=128)
    parser.add_argument("--goal-progress-coefficient", type=float, default=0.01)
    parser.add_argument("--checkpoint-every-updates", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--candidate-adapter-rank", type=int, default=8)
    parser.add_argument("--candidate-variation-scale", type=float, default=0.01)
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
    if (
        not 1 <= args.worlds <= 16
        or not 1 <= args.residents_per_world <= MAX_RESIDENTS
        or args.steps < args.rollout_steps
    ):
        raise SystemExit(
            "worlds must be 1..16, residents within the interface capacity, "
            "and steps at least one rollout"
        )
    if args.steps % args.rollout_steps or args.episode_steps % args.rollout_steps:
        raise SystemExit("steps and episode steps must be divisible by rollout steps")
    if args.episode_steps < args.rollout_steps or not 1 <= args.ppo_epochs <= 16:
        raise SystemExit("invalid episode or PPO schedule")
    if args.goal_sticky_steps < 1 or args.goal_reservoir_size < 4:
        raise SystemExit("invalid private goal schedule")
    if not 1 <= args.candidate_count <= args.worlds * args.residents_per_world:
        raise SystemExit("candidate count exceeds the resident cohort")
    if not 1 <= args.candidate_adapter_rank <= 32:
        raise SystemExit("candidate adapter rank must be 1..32")
    if not 0 <= args.candidate_variation_scale <= 0.1:
        raise SystemExit("candidate variation scale must be in [0,0.1]")
    if not 0 <= args.goal_progress_coefficient <= 1:
        raise SystemExit("goal progress coefficient must be in [0,1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be absent or empty")
    if (
        args.initialize_from_development is not None
        and not args.initialize_from_development.is_file()
    ):
        raise SystemExit("development initialization checkpoint does not exist")
    if not args.candidate_genomes.is_file():
        raise SystemExit("candidate genome plan does not exist")


def load_candidate_plan(args, graph_hash, port_spec_hash, profile_hash, bootstrap_hash):
    from chreatures.population import CandidateGenome, canonical_bytes

    path = args.candidate_genomes.resolve()
    value = json.loads(path.read_text())
    if (
        value.get("format")
        != "chreatures-torch-population-training-candidates-v1"
        or value.get("version") != 1
        or value.get("content_sha256")
        != hashlib.sha256(
            canonical_bytes(
                {key: item for key, item in value.items() if key != "content_sha256"}
            )
        ).hexdigest()
    ):
        raise ValueError("candidate training plan identity differs")
    count = args.worlds * args.residents_per_world
    candidates = [CandidateGenome(item) for item in value.get("candidates", ())]
    mapping = value.get("mapping")
    training = value.get("training", {})
    if (
        len(candidates) != count
        or not isinstance(mapping, list)
        or len(mapping) != count
        or value.get("controller_file_sha256") != bootstrap_hash
        or value.get("profile_sha256") != profile_hash
        or value.get("graph_sha256") != graph_hash
        or value.get("port_spec_sha256") != port_spec_hash
        or value.get("worlds") != args.worlds
        or value.get("residents_per_world") != args.residents_per_world
        or value.get("policy_adapter_count") != args.candidate_count
        or value.get("policy_adapter_rank") != args.candidate_adapter_rank
        or training.get("physical_steps") != args.steps
        or training.get("rollout_steps") != args.rollout_steps
        or training.get("episode_steps") != args.episode_steps
        or training.get("ppo_epochs") != args.ppo_epochs
        or training.get("updates") != args.steps // args.rollout_steps
    ):
        raise ValueError("candidate training plan substrate or schedule differs")
    adapter_indices = []
    for resident, (entry, candidate) in enumerate(
        zip(mapping, candidates, strict=True)
    ):
        adapter = candidate.controller_adapter()
        if (
            entry.get("resident_index") != resident
            or entry.get("candidate_sha256") != candidate.to_value()["sha256"]
            or entry.get("policy_adapter_index") != adapter["policy_adapter_index"]
        ):
            raise ValueError("candidate training assignment differs")
        adapter_indices.append(adapter["policy_adapter_index"])
    if sorted(adapter_indices) != [
        index
        for index in range(args.candidate_count)
        for _ in range(count // args.candidate_count)
    ]:
        raise ValueError("candidate adapter rows are not balanced")
    return value, candidates, np.asarray(adapter_indices, dtype=np.int64), path


class GoalAdapter:
    def __init__(self, model):
        self.model = model

    def encode(self, window):
        return self.model.encode_goal(window)

    def state_dict(self):
        return self.model.state_dict()


def load_bootstrap(path: Path, device: torch.device, *, cold_inherit_v3: bool):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        value = torch.load(path, map_location=device, weights_only=True)
    source_format = value.get("format")
    inherited_formats = {
        "chreatures-rich-sensorimotor-bootstrap-v1",
        "chreatures-rich-online-sensorimotor-development-v1",
    }
    if cold_inherit_v3 != (source_format in inherited_formats):
        raise ValueError(
            "bootstrap format requires an explicit cold inheritance choice"
        )
    if source_format not in {
        *inherited_formats,
        "chreatures-rich-sensorimotor-bootstrap-v4",
    }:
        raise ValueError("rich bootstrap format differs")
    identity = value["identity"]
    model = RichSensorimotorModel().to(device)
    inherited_manager = None
    inherited_adapters = None
    if cold_inherit_v3:
        model.load_state_dict(cold_inherit_v3_model(value["model"]), strict=True)
        normalizer = RichNormalizer.cold_inherit_v3(identity["normalizer"])
        identity = copy.deepcopy(identity)
        identity["cold_inheritance"] = {
            "source_format": source_format,
            "source_sha256": sha256(path),
            "new_action_axes": ["eat", "release", "secrete", "allocate"],
            "new_physiology_columns": 6,
        }
        if "goal_manager" in value:
            inherited_manager = cold_inherit_v3_manager(value["goal_manager"])
    else:
        model.load_state_dict(value["model"], strict=True)
        normalizer = RichNormalizer.from_value(identity["normalizer"])
        inherited_manager = value.get("goal_manager")
        inherited_adapters = value.get("candidate_adapters")
    for module in (model.visual, model.body, model.goal_encoder, model.goal_decoder):
        module.requires_grad_(False)
        module.eval()
    return (
        GoalAdapter(model),
        model,
        normalizer,
        identity,
        inherited_manager,
        inherited_adapters,
    )


def observe(pool, brain, normalizer):
    rich, canonical, bodies = pool.observe_arrays()
    neural, circuit, _ = brain.step_channels(canonical, 0.05)
    physical = pool.physiology_array(circuit[:, 2])
    raw = np.ascontiguousarray(
        np.concatenate((rich, canonical, physical), axis=1), dtype=np.float32
    )
    return normalizer.normalize(raw), physical, neural, bodies, raw


def rich_summary(raw: np.ndarray) -> np.ndarray:
    rays = raw[:, :4096].reshape(len(raw), 1024, 4)
    groups = (rays[:, :256], rays[:, 256:])
    values = []
    for group in groups:
        values.extend(
            (
                group[..., :3].mean(1),
                group[..., :3].std(1),
                group[..., 3:4].mean(1),
                group[..., 3:4].max(1),
            )
        )
    return np.concatenate(values, axis=1).astype(np.float32)


def action_payload(actions: np.ndarray, bodies) -> list[dict[str, object]]:
    payloads = []
    row = 0
    for world in bodies:
        mapped = {}
        for body in world:
            values = dict(
                zip(ACTION_NAMES, actions[row].astype(float).tolist(), strict=True)
            )
            mapped[str(body["id"])] = values
            row += 1
        payloads.append({"actions": mapped, "dt": 0.05})
    return payloads


def physical_reward(objective, before, after, outcomes, bodies):
    nutrition, effort = [], []
    for world_bodies, world_outcomes in zip(bodies, outcomes, strict=True):
        for body in world_bodies:
            outcome = world_outcomes[body["id"]]
            nutrition.append(outcome["nutrition"])
            effort.append(outcome["effort"])
    reward, terms = objective.transition(
        before[:, :3],
        after[:, :3],
        nutrition=nutrition,
        effort=effort,
        dt=0.05,
    )
    return reward, terms


def transition_outcome_rows(outcomes, bodies, reward) -> np.ndarray:
    rows = []
    index = 0
    for world_bodies, world_outcomes in zip(bodies, outcomes, strict=True):
        for body in world_bodies:
            value = world_outcomes[body["id"]]
            rows.append(
                (
                    value["nutrition"],
                    value["contact"],
                    value["distance"],
                    value["effort"],
                    value["mechanical_work"],
                    value.get("ingested_mass", 0.0),
                    value.get("mouth_material_contacts", 0.0),
                    reward[index],
                )
            )
            index += 1
    return np.asarray(rows, dtype=np.float32)


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
            code, _achieved_tick = self.reservoirs[resident][int(slot)]
            self.codes[resident] = code
            self.age[resident] = 0
            self.horizons[resident, 0] = math.log1p(self.sticky) / math.log(41)

    def advance_goal_age(self) -> None:
        """Advance attempt time; achieved-memory age remains in each record."""
        self.age += 1
        remaining = np.maximum(1, self.sticky - self.age)
        self.horizons[:, 0] = np.log1p(remaining) / math.log(41)

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


def apply_candidate_temperatures(logits, offsets):
    """Apply inherited per-axis log-temperature offsets to policy factors."""
    scale = torch.exp(-offsets)
    return {
        "signed": logits["signed"] * scale[..., :4, None],
        "active": logits["active"] * scale[..., 4:],
        "positive": logits["positive"] * scale[..., 4:, None],
    }


def reset_worlds(
    pool, brain, worlds: int, residents: int, seed: int, episode: int, candidates
):
    bodies = pool.reset(
        [
            {
                "seed": seed + episode * 1009 + index,
                "held_out": False,
                "candidates": [
                    item.to_value()
                    for item in candidates[index * residents : (index + 1) * residents]
                ],
            }
            for index in range(worlds)
        ],
    )
    identities = [
        f"episode-{episode:04d}/world-{world:02d}/resident-{resident}"
        for world in range(worlds)
        for resident in range(residents)
    ]
    brain.reset_residents(identities)
    return bodies


def save_development(
    path,
    *,
    identity,
    worker,
    encoder,
    critic,
    manager,
    adapters,
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
    observation,
    physical,
    neural_readout,
    raw_observation,
    current_code,
):
    """Atomically save coherent world, neural, optimizer, and private state."""
    value = {
        "format": FORMAT,
        "identity": identity,
        "updates": updates,
        "physical_steps": physical_steps,
        "episode": episode,
        "model": worker.state_dict(),
        "critic": critic.state_dict(),
        "goal_manager": manager.state_dict(),
        "candidate_adapters": adapters.state_dict(),
        "optimizer": optimizer.state_dict(),
        "private_worker_hidden": hidden.cpu(),
        "private_previous_action": previous,
        "private_goals": goals.value(),
        "private_manager_events": manager_events,
        "private_active_manager": active_manager,
        "worlds": pool.snapshot(),
        "neural": {
            key: value.copy() for key, value in brain.export_state().items()
        },
        "neural_resident_ids": list(brain.resident_ids),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "pending_rollout": {"length": 0},
        "boundary_observation": observation,
        "boundary_physiology": physical,
        "boundary_neural_readout": neural_readout,
        "boundary_raw_observation": raw_observation,
        "boundary_current_code": current_code,
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
    from chreatures.training_cohort import (
        TrainingCohortBrain,
        WorldTrainingPool,
        load_training_graph,
    )
    from chreatures.training_environment import EmbodiedTrainingProfile

    (
        encoder,
        worker,
        normalizer,
        bootstrap_identity,
        inherited_manager,
        inherited_adapters,
    ) = load_bootstrap(
        args.bootstrap_worker.resolve(),
        device,
        cold_inherit_v3=args.cold_inherit_v3,
    )
    profile = EmbodiedTrainingProfile.nursery_family(
        args.chemical_habitat,
        args.chemical_biosphere,
        args.nursery_family_config,
        args.nursery_family_schedule,
    )
    if int(profile.component("version")) != 6:
        raise ValueError("constructed environment profile version differs")
    objective = FiniteEnergyObjective(
        FiniteEnergyConfig.from_value(profile.component("homeostasis"))
    )
    graph = load_training_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    if len(ports.input_names) != 351:
        raise ValueError("development v4 requires the pinned 351 sensory channels")
    profile_residents = len(profile.component("habitat")["bodies"])
    if profile_residents != args.residents_per_world:
        raise ValueError("profile resident count differs from the requested cohort")
    declared_transport = profile.component("family")["transport"]
    actual_transport = {
        "residents": profile_residents,
        "rich": 4096,
        "physical": len(ports.input_names),
        "physiology": PHYSIOLOGY_DIM,
        "controller": worker.config.observation_dim,
        "readouts": len(ports.readout_names),
    }
    expected_transport = {
        "residents": args.residents_per_world,
        "rich": 4096,
        "physical": 351,
        "physiology": PHYSIOLOGY_DIM,
        "controller": worker.config.observation_dim,
        "readouts": 384,
    }
    if (
        declared_transport != expected_transport
        or actual_transport != expected_transport
    ):
        raise ValueError(
            "current nursery-family transport differs from the active rich interfaces"
        )
    bootstrap_hash = sha256(args.bootstrap_worker.resolve())
    (
        candidate_plan,
        candidates,
        adapter_assignment,
        candidate_plan_path,
    ) = load_candidate_plan(
        args,
        str(graph.hash),
        ports.spec_hash,
        profile.to_value()["sha256"],
        bootstrap_hash,
    )
    count = args.worlds * args.residents_per_world
    from chreatures.neural_genotype import (
        NeuralVariantRecipe,
        compile_population_phenotypes,
    )

    phenotypes = compile_population_phenotypes(
        candidates,
        NeuralVariantRecipe.load(args.neural_recipe),
        graph,
        ports,
        sha256(args.port_bundle.resolve()),
        bootstrap_hash,
    )
    brain = TrainingCohortBrain(
        graph,
        ports,
        count,
        device=args.device,
        backend=args.brain_backend,
    )
    brain.bind_phenotypes(phenotypes)
    pool = WorldTrainingPool(
        args.worlds,
        dict(ports.spec),
        profile.to_value(),
        args.physical_backend,
        args.residents_per_world,
    )
    critic = Critic().to(device)
    manager = SlowGoalManager().to(device)
    if inherited_manager is not None:
        manager.load_state_dict(inherited_manager, strict=True)
    adapters = PopulationAdapterBank(
        args.candidate_count, args.candidate_adapter_rank
    ).to(device)
    if inherited_adapters is not None:
        adapters.load_state_dict(inherited_adapters, strict=True)
    elif args.candidate_count > 1:
        worker.requires_grad_(False)
        adapters.vary(
            torch.arange(1, args.candidate_count, device=device),
            seed=args.seed ^ 0xADA7,
            scale=args.candidate_variation_scale,
        )
    if args.candidate_count > 1:
        worker.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            *[
                parameter
                for parameter in worker.parameters()
                if parameter.requires_grad
            ],
            *adapters.parameters(),
            *critic.parameters(),
            *manager.parameters(),
        ],
        lr=args.learning_rate,
    )
    initialization = None
    if args.initialize_from_development is not None:
        initialization_path = args.initialize_from_development.resolve()
        checkpoint = torch.load(
            initialization_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("format") != FORMAT:
            raise ValueError("development initialization format differs")
        parent_identity = checkpoint["identity"]
        if (
            parent_identity.get("rich_profile_sha256") != RICH_PROFILE_SHA256
            or parent_identity.get("rich_channel_names_sha256")
            != RICH_CHANNEL_NAMES_SHA256
            or parent_identity.get("graph_sha256") != str(graph.hash)
            or parent_identity.get("port_spec_sha256") != ports.spec_hash
            or parent_identity.get("bootstrap_sha256")
            != sha256(args.bootstrap_worker.resolve())
        ):
            raise ValueError("development initialization substrate differs")
        worker.load_state_dict(checkpoint["model"], strict=True)
        critic.load_state_dict(checkpoint["critic"], strict=True)
        manager.load_state_dict(checkpoint["goal_manager"], strict=True)
        adapters.load_state_dict(checkpoint["candidate_adapters"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        initialization = {
            "path": str(initialization_path),
            "sha256": sha256(initialization_path),
            "updates": int(checkpoint["updates"]),
            "physical_steps": int(checkpoint["physical_steps"]),
            "semantics": (
                "shared worker, critic, goal manager, and optimizer inherited; "
                "fresh world, neural, private goal/history, and RNG state"
            ),
        }
    goals = ResidentGoals(
        count,
        args.goal_reservoir_size,
        args.goal_sticky_steps,
        args.seed ^ 0x60A1,
    )
    identity = {
        "format": FORMAT,
        "controller_mode": "rich-achieved-goal",
        "normalizer": normalizer.to_value(),
        "rich_profile_sha256": RICH_PROFILE_SHA256,
        "rich_channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
        "seed": args.seed,
        "dt_seconds": 0.05,
        "action_order": list(ACTION_NAMES),
        "graph_sha256": str(graph.hash),
        "port_spec_sha256": ports.spec_hash,
        "port_bundle_sha256": sha256(args.port_bundle.resolve()),
        "profile": profile.to_value(),
        "bootstrap_path": str(args.bootstrap_worker.resolve()),
        "bootstrap_sha256": sha256(args.bootstrap_worker.resolve()),
        "bootstrap_identity": bootstrap_identity,
        "population_adapters": {
            "format": "chreatures-population-controller-adapters-v1",
            "candidates": args.candidate_count,
            "rank": args.candidate_adapter_rank,
            "variation_seed": args.seed ^ 0xADA7,
            "variation_scale": args.candidate_variation_scale,
            "candidate_assignment": "resident_index_mod_candidate_count",
            "shared_base_trainable": args.candidate_count == 1,
            "private_state_at_birth": "zero",
            "candidate_plan_path": str(candidate_plan_path),
            "candidate_plan_file_sha256": sha256(candidate_plan_path),
            "candidate_plan_content_sha256": candidate_plan["content_sha256"],
            "candidate_sha256s": [
                item["sha256"] for item in candidate_plan["candidates"]
            ],
            "applied_genome_loci": [
                "controller.policy_adapter_index",
                "controller.action_logit_temperature_offset.*",
            ],
            "physical_genome_loci": "applied by WorldTrainingPool reset",
            "neural_phenotype_sha256s": [item.sha256 for item in phenotypes],
            "neural_recipe_path": str(args.neural_recipe.resolve()),
            "neural_recipe_file_sha256": sha256(args.neural_recipe.resolve()),
            "unsupported_torch_controller_loci": [
                "controller.action_gain.*",
                "controller.recurrent_gain",
                "controller.learning_rate_gain",
            ],
        },
        "telemetry": {
            "format": "chreatures-rich-development-transition-telemetry-v1",
            "storage": "one compressed NPZ per PPO rollout; time-major then resident",
            "audit_axes": ["episode", "tick", "world_slot", "resident_slot"],
            "outcome_order": [
                "nutrition",
                "contact",
                "distance",
                "effort",
                "mechanical_work",
                "ingested_mass",
                "mouth_material_contacts",
                "homeostatic_reward",
            ],
            "fit_rule": "audit identities and outcomes are never controller inputs",
        },
        "arguments": vars(args)
        | {
            key: str(value)
            for key, value in vars(args).items()
            if isinstance(value, Path)
        },
        "development_initialization": initialization,
    }
    atomic_json(args.output / "identity.json", identity)
    episode = 0
    reset_worlds(
        pool,
        brain,
        args.worlds,
        args.residents_per_world,
        args.seed,
        episode,
        candidates,
    )
    observation, physical, neural, bodies, raw_observation = observe(
        pool, brain, normalizer
    )
    previous = np.zeros((count, ACTION_DIM), dtype=np.float32)
    candidate_index = torch.as_tensor(
        adapter_assignment, device=device
    )
    candidate_temperature = torch.as_tensor(
        np.asarray(
            [
                item.controller_adapter()["action_logit_temperature_offset"]
                for item in candidates
            ],
            dtype=np.float32,
        ),
        device=device,
    )
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
            telemetry = {
                name: []
                for name in (
                    "episode",
                    "tick",
                    "world_slot",
                    "resident_slot",
                    "physiology",
                    "neural",
                    "worker_hidden",
                    "previous",
                    "goal_distance_t",
                    "goal_distance_t1",
                    "goal_progress",
                    "goal_attempt_age",
                    "goal_switch",
                    "rich_summary",
                    "canonical",
                    "executed_action",
                    "next_physiology",
                    "outcomes",
                )
            }
            world_slot = np.repeat(
                np.arange(args.worlds, dtype=np.int16), args.residents_per_world
            )
            resident_slot = np.tile(
                np.arange(args.residents_per_world, dtype=np.int8), args.worlds
            )
            initial_hidden = hidden.detach().clone()
            for local in range(args.rollout_steps):
                global_step = rollout_start + local
                reset = np.zeros(count, dtype=bool)
                selected_this_tick = np.zeros(count, dtype=bool)
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
                        manager_probability = manager_logits.softmax(-1).cpu().numpy()
                        selected = torch.as_tensor(
                            [
                                goals.rng[index].choice(
                                    goals.size, p=manager_probability[index]
                                )
                                for index in range(count)
                            ],
                            device=device,
                        )
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
                        selected_this_tick.fill(True)
                    goal_age_at_action = goals.age.copy()
                    goal_tensor = torch.as_tensor(goals.codes, device=device)
                    horizon = torch.as_tensor(goals.horizons, device=device)
                    logits = worker.policy(
                        state[0],
                        goal_tensor,
                        horizon,
                        torch.as_tensor(previous, device=device),
                        adapters,
                        candidate_index,
                    )
                    logits = apply_candidate_temperatures(
                        logits, candidate_temperature
                    )
                    private_uniforms = np.stack(
                        [rng.random(20, dtype=np.float32) for rng in goals.rng]
                    )
                    action = sample_worker_actions(
                        worker,
                        logits,
                        torch.as_tensor(private_uniforms, device=device),
                    )
                    log_probability, _ = policy_terms(worker, logits, action)
                    value = critic(state[0], goal_tensor, horizon)
                action_np = action.cpu().numpy()
                before = physical.copy()
                advanced = pool.advance(action_payload(action_np, bodies))
                outcomes = [item[0] for item in advanced]
                bodies = [item[1] for item in advanced]
                next_observation, physical, next_neural, bodies, next_raw = observe(
                    pool, brain, normalizer
                )
                base_reward, _ = physical_reward(
                    objective, before, physical, outcomes, bodies
                )
                outcome_rows = transition_outcome_rows(outcomes, bodies, base_reward)
                # Credit the whole transition against the pre-action sticky goal.
                frozen_goal = goals.codes.copy()
                next_code = goals.record(next_observation, encoder, device)
                old_distance = np.linalg.norm(current_code - frozen_goal, axis=1)
                new_distance = np.linalg.norm(next_code - frozen_goal, axis=1)
                progress = args.goal_progress_coefficient * (
                    old_distance - new_distance
                )
                reward = base_reward + progress.astype(np.float32)
                goals.advance_goal_age()
                for name, value_np in (
                    ("episode", np.full(count, episode, dtype=np.int32)),
                    ("tick", np.full(count, global_step, dtype=np.int64)),
                    ("world_slot", world_slot),
                    ("resident_slot", resident_slot),
                    ("physiology", before),
                    ("neural", neural),
                    ("worker_hidden", state[0].cpu().numpy()),
                    ("previous", previous),
                    ("goal_distance_t", old_distance),
                    ("goal_distance_t1", new_distance),
                    ("goal_progress", old_distance - new_distance),
                    ("goal_attempt_age", goal_age_at_action),
                    ("goal_switch", selected_this_tick),
                    ("rich_summary", rich_summary(raw_observation)),
                    ("canonical", raw_observation[:, 4096:4447]),
                    ("executed_action", action_np),
                    ("next_physiology", physical),
                    ("outcomes", outcome_rows),
                ):
                    telemetry[name].append(np.asarray(value_np))
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
                previous = action_np.astype(np.float32, copy=True)
                observation = next_observation
                raw_observation = next_raw
                neural = next_neural
                current_code = next_code
                hidden = next_hidden
                if done[0]:
                    episode += 1
                    reset_worlds(
                        pool,
                        brain,
                        args.worlds,
                        args.residents_per_world,
                        args.seed,
                        episode,
                        candidates,
                    )
                    goals.reset()
                    observation, physical, neural, bodies, raw_observation = observe(
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
            atomic_npz(
                args.output / f"telemetry-{rollout_start:08d}.npz",
                {key: np.stack(value) for key, value in telemetry.items()},
            )
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
                    torch.as_tensor(batch["previous"], device=device),
                    adapters,
                    candidate_index.expand(args.rollout_steps, -1),
                )
                logits = apply_candidate_temperatures(
                    logits,
                    candidate_temperature.expand(args.rollout_steps, -1, -1),
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
                    [
                        *worker.parameters(),
                        *adapters.parameters(),
                        *critic.parameters(),
                    ],
                    1.0,
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
                    adapters=adapters,
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
                    observation=observation,
                    physical=physical,
                    neural_readout=neural,
                    raw_observation=raw_observation,
                    current_code=current_code,
                )
        save_development(
            args.output / "development.pt",
            identity=identity,
            worker=worker,
            encoder=encoder,
            critic=critic,
            manager=manager,
            adapters=adapters,
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
            observation=observation,
            physical=physical,
            neural_readout=neural,
            raw_observation=raw_observation,
            current_code=current_code,
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
