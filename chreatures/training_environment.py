"""Versioned embodied training worlds matching the current 3-D life stack."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .acoustics import Acoustics
from .ecology import Ecology
from .fields import FieldEnvironment
from .homeostasis import FiniteEnergyConfig, FiniteEnergyObjective
from .physical_batch import FastArticulatedSensoriumWorld
from .sensorium import ArticulatedSensoriumWorld, BODY_FRAME


PROFILE_FORMAT = "chreatures-embodied-training-profile-v1"
SNAPSHOT_FORMAT = "chreatures-embodied-training-world-v1"
ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_BACKENDS = {
    "reference": ArticulatedSensoriumWorld,
    "fast": FastArticulatedSensoriumWorld,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EmbodiedTrainingProfile:
    """Immutable-by-copy environment contract selected explicitly by a runner."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        raw = copy.deepcopy(dict(value))
        expected = {
            "format", "version", "name", "sensorium", "body", "fields",
            "resources", "acoustics", "homeostasis", "variation", "horizons", "sources",
        }
        if set(raw) != expected or raw.get("format") != PROFILE_FORMAT or raw.get("version") != 1:
            raise ValueError("unsupported embodied training profile")
        if raw["sensorium"] != {"frame": BODY_FRAME} or raw["body"] != "articulated":
            raise ValueError("embodied training v1 requires body-v1 articulated sensing")
        self._value = raw
        self.sha256 = hashlib.sha256(_canonical(raw)).hexdigest()
        # Constructing these validators catches malformed embedded configs
        # without retaining mutable simulation state.
        FiniteEnergyConfig.from_value(raw["homeostasis"])
        if not all(isinstance(raw[key], dict) for key in ("fields", "resources", "acoustics", "variation", "horizons", "sources")):
            raise ValueError("embodied training profile components must be mappings")

    @classmethod
    def current(cls) -> "EmbodiedTrainingProfile":
        """Build the explicit current-life profile from repository-owned assets."""
        habitat = ROOT / "data/habitats/terrarium-garden.json"
        resources = ROOT / "data/ecology/terrarium-orchard.json"
        acoustics = ROOT / "data/components/terrarium-play.json"
        # Persist normalized field defaults rather than relying on defaults at restore.
        field_config = FieldEnvironment(size=(12, 8, 3.5)).config
        paths = {
            "habitat": habitat, "resources": resources, "acoustics": acoustics,
            "physics": ROOT / "chreatures/physics.py",
            "articulated": ROOT / "chreatures/articulated.py",
            "sensorium": ROOT / "chreatures/sensorium.py",
            "fields": ROOT / "chreatures/fields.py",
            "ecology": ROOT / "chreatures/ecology.py",
            "acoustics_module": ROOT / "chreatures/acoustics.py",
            "homeostasis": ROOT / "chreatures/homeostasis.py",
            "training_environment": ROOT / "chreatures/training_environment.py",
        }
        return cls({
            "format": PROFILE_FORMAT,
            "version": 1,
            "name": "current-life-body-v1-diffusion-terrarium-v1",
            "sensorium": {"frame": BODY_FRAME},
            "body": "articulated",
            "fields": field_config,
            "resources": json.loads(resources.read_text()),
            "acoustics": json.loads(acoustics.read_text()),
            "homeostasis": FiniteEnergyConfig().to_value(),
            "variation": {
                "version": 1,
                "training_occluder_jitter_m": [0.35, 0.25],
                "heldout_occluder_jitter_m": [0.60, 0.40],
                "movable_jitter_m": 0.32,
                "body_heading_span_rad": math.pi,
                "heldout_seed_offset": 80_000_003,
                "early_food_distance_m": [0.28, 0.44],
            },
            "horizons": {
                "training_episode_steps": 24_000,
                "heldout_steps": 24_000,
                "telemetry_every_steps": 1_200,
                "checkpoint_every_steps": 4_800,
                "dt_seconds": 0.05,
                "rationale": "1200 s spans digestion/fatigue recovery and exceeds the observed ~1100 s inherited-policy collapse",
            },
            "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": _sha(path)}
                        for name, path in paths.items()},
        })

    def to_value(self) -> dict[str, Any]:
        return {"value": copy.deepcopy(self._value), "sha256": self.sha256}

    @classmethod
    def from_value(cls, encoded: Mapping[str, Any]) -> "EmbodiedTrainingProfile":
        if not isinstance(encoded, Mapping) or set(encoded) != {"value", "sha256"}:
            raise ValueError("invalid encoded embodied training profile")
        profile = cls(encoded["value"])
        if encoded["sha256"] != profile.sha256:
            raise ValueError("embodied training profile checksum differs")
        return profile

    def component(self, name: str) -> Any:
        return copy.deepcopy(self._value[name])

    @property
    def name(self) -> str:
        return str(self._value["name"])


def embodied_training_spec(
    seed: int, *, held_out: bool = False,
    profile: EmbodiedTrainingProfile | None = None,
    base_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a varied physical world; geometry never enters policy inputs."""
    profile = profile or EmbodiedTrainingProfile.current()
    spec = copy.deepcopy(dict(base_spec)) if base_spec is not None else json.loads(
        (ROOT / profile.component("sources")["habitat"]["path"]).read_text()
    )
    spec["sensorium"] = profile.component("sensorium")
    variation = profile.component("variation")
    chosen_seed = int(seed) + (int(variation["heldout_seed_offset"]) if held_out else 0)
    rng = np.random.default_rng(chosen_seed)
    width, height = map(float, spec["size"][:2])
    by_id = {entity["id"]: entity for entity in spec["entities"]}

    span = variation["heldout_occluder_jitter_m"] if held_out else variation["training_occluder_jitter_m"]
    for entity_id in ("garden-screen", "echo-arch"):
        entity = by_id[entity_id]
        entity["position"][0] = float(np.clip(entity["position"][0] + rng.uniform(-span[0], span[0]), .5, width - .5))
        entity["position"][1] = float(np.clip(entity["position"][1] + rng.uniform(-span[1], span[1]), .5, height - .5))

    movable = (
        "tone-ball", "cyan-ball", "rattle-block", "stack-block-a", "stack-block-b",
        "sun-berry", "shade-nectar", "screen-seed", "way-berry", "floor-nectar",
    )
    jitter = float(variation["movable_jitter_m"])
    for entity_id in movable:
        entity = by_id[entity_id]
        entity["position"][0] = float(np.clip(entity["position"][0] + rng.uniform(-jitter, jitter), .25, width - .25))
        entity["position"][1] = float(np.clip(entity["position"][1] + rng.uniform(-jitter, jitter), .25, height - .25))
    for body in spec["bodies"]:
        body["heading"] = float(rng.uniform(-variation["body_heading_span_rad"], variation["body_heading_span_rad"]))
        body["energy"] = float(rng.uniform(.72, .84))
        body["gut"] = float(rng.uniform(.06, .16))
        body["fatigue"] = float(rng.uniform(.02, .08))
    # One ordinary renewable food begins within early physical reach of each
    # resident. This is a world curriculum, not an identity-aware policy aid.
    nearby_food = ("way-berry", "screen-seed", "sun-berry")
    low, high = map(float, variation["early_food_distance_m"])
    for body, entity_id in zip(spec["bodies"], nearby_food, strict=True):
        angle = float(body["heading"] + rng.uniform(-.75, .75))
        distance = float(rng.uniform(low, high))
        entity = by_id[entity_id]
        entity["position"][0] = float(np.clip(body["position"][0] + math.cos(angle) * distance, .25, width - .25))
        entity["position"][1] = float(np.clip(body["position"][1] + math.sin(angle) * distance, .25, height - .25))
        entity["position"][2] = float(body["position"][2])
    spec["name"] = "embodied-current-life-heldout" if held_out else "embodied-current-life-training"
    spec["training_profile_sha256"] = profile.sha256
    spec["training_variant"] = {"seed": chosen_seed, "held_out": bool(held_out)}
    return spec


class EmbodiedTrainingWorld:
    """Drop-in worker world with body-v1, diffusion, resources and acoustics."""

    def __init__(
        self, seed: int, spec: dict[str, Any], profile: EmbodiedTrainingProfile,
        *, physical_backend: str = "reference",
    ) -> None:
        if not isinstance(profile, EmbodiedTrainingProfile):
            raise TypeError("profile must be an EmbodiedTrainingProfile")
        if spec.get("sensorium") != profile.component("sensorium"):
            raise ValueError("world spec sensorium differs from training profile")
        if spec.get("training_profile_sha256") != profile.sha256:
            raise ValueError("world spec does not identify its training profile")
        if physical_backend not in PHYSICAL_BACKENDS:
            raise ValueError(f"unknown embodied physical backend: {physical_backend!r}")
        self.seed = int(seed)
        self.profile = profile
        self.physical_backend = physical_backend
        self.world = PHYSICAL_BACKENDS[physical_backend](seed=self.seed, spec=copy.deepcopy(spec))
        self.field = FieldEnvironment.from_world(self.world, profile.component("fields"))
        self.resources = Ecology(self.world, profile.component("resources"), seed=self.seed ^ 0xEC0106)
        self.acoustics = Acoustics(self.world, profile.component("acoustics"))
        self.objective = FiniteEnergyObjective(
            FiniteEnergyConfig.from_value(profile.component("homeostasis"))
        )
        self.last_telemetry: dict[str, Any] = {}

    @property
    def bodies(self):
        return self.world.bodies

    def sense(self, body_id: str) -> dict[str, Any]:
        sensed = self.world.sense(body_id)
        positions = sensed.pop("antenna_position", None)
        if positions is None:
            raise RuntimeError("body-v1 sensorium omitted physical antenna positions")
        concentration = np.asarray(self.field.sample(positions), dtype=np.float64)[:, :3]
        sensed["odor"] = (-np.expm1(-concentration / .1)).tolist()
        return sensed

    def advance(self, actions: dict[str, dict[str, Any]], dt: float) -> dict[str, dict[str, Any]]:
        before = {
            body.id: np.asarray([body.energy, body.gut, body.fatigue], dtype=np.float64)
            for body in self.bodies
        }
        outcomes = self.world.advance(actions, dt)
        acoustic = self.acoustics.advance(dt)
        resources = self.resources.advance(dt)
        self.field.sync_dynamic_barriers(self.world.diffusion_barriers())
        field = self.field.advance(dt, sources=self.field.sources_from_world(self.world))
        rewards = []
        for body in self.bodies:
            outcome = outcomes[body.id]
            after = np.asarray([body.energy, body.gut, body.fatigue], dtype=np.float64)
            reward, terms = self.objective.transition(
                before[body.id], after, nutrition=outcome["nutrition"],
                effort=outcome["effort"], dt=dt,
            )
            outcome["homeostatic_reward"] = float(reward)
            outcome["homeostasis"] = {name: float(value) for name, value in terms.items()}
            rewards.append(float(reward))
        self.last_telemetry = {
            "time": float(self.world.time), "profile_sha256": self.profile.sha256,
            "nutrition": float(sum(item["nutrition"] for item in outcomes.values())),
            "distance": float(sum(item["distance"] for item in outcomes.values())),
            "contacts": int(sum(item["contact"] > 0 for item in outcomes.values())),
            "effort_mean": float(np.mean([item["effort"] for item in outcomes.values()])),
            "homeostatic_reward_sum": float(sum(rewards)),
            "field": copy.deepcopy(field),
            "resources": copy.deepcopy(resources),
            "acoustics": copy.deepcopy(acoustic),
        }
        return outcomes

    def snapshot(self) -> dict[str, Any]:
        return {
            "format": SNAPSHOT_FORMAT, "version": 1,
            "seed": self.seed, "profile": self.profile.to_value(),
            "world": self.world.snapshot(), "field": self.field.snapshot(),
            "resources": self.resources.snapshot(), "acoustics": self.acoustics.snapshot(),
            "last_telemetry": copy.deepcopy(self.last_telemetry),
        }

    @classmethod
    def restore(
        cls, snapshot: Mapping[str, Any],
        expected_profile: EmbodiedTrainingProfile | str | None = None,
        *, physical_backend: str = "reference",
    ) -> "EmbodiedTrainingWorld":
        if snapshot.get("format") != SNAPSHOT_FORMAT or snapshot.get("version") != 1:
            raise ValueError("unsupported embodied training world snapshot")
        profile = EmbodiedTrainingProfile.from_value(snapshot["profile"])
        expected_hash = expected_profile.sha256 if isinstance(expected_profile, EmbodiedTrainingProfile) else expected_profile
        if expected_hash is not None and str(expected_hash) != profile.sha256:
            raise ValueError("training checkpoint profile differs")
        if physical_backend not in PHYSICAL_BACKENDS:
            raise ValueError(f"unknown embodied physical backend: {physical_backend!r}")
        instance = cls.__new__(cls)
        instance.seed = int(snapshot["seed"])
        instance.profile = profile
        instance.physical_backend = physical_backend
        instance.world = PHYSICAL_BACKENDS[physical_backend].restore(snapshot["world"])
        if instance.world.spec.get("training_profile_sha256") != profile.sha256:
            raise ValueError("restored physical world profile differs")
        instance.field = FieldEnvironment.restore(snapshot["field"])
        instance.resources = Ecology.restore(instance.world, snapshot["resources"])
        instance.acoustics = Acoustics.restore(instance.world, snapshot["acoustics"])
        instance.objective = FiniteEnergyObjective(
            FiniteEnergyConfig.from_value(profile.component("homeostasis"))
        )
        instance.last_telemetry = copy.deepcopy(snapshot.get("last_telemetry", {}))
        times = (instance.world.time, instance.field.time, instance.resources.time, instance.acoustics.time)
        if max(times) - min(times) > 1e-9:
            raise ValueError("restored embodied environment clocks differ")
        return instance

    def close(self) -> None:
        self.acoustics.close()


__all__ = [
    "EmbodiedTrainingProfile", "EmbodiedTrainingWorld", "PHYSICAL_BACKENDS",
    "embodied_training_spec",
]
