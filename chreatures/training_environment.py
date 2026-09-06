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
PROFILE_FORMAT_V2 = "chreatures-embodied-training-profile-v2"
PROFILE_FORMAT_V3 = "chreatures-embodied-chemical-nursery-profile-v3"
SNAPSHOT_FORMAT = "chreatures-embodied-training-world-v1"
SNAPSHOT_FORMAT_V2 = "chreatures-embodied-training-world-v2"
SNAPSHOT_FORMAT_V3 = "chreatures-embodied-training-world-v3"
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
        legacy_expected = {
            "format", "version", "name", "sensorium", "body", "fields",
            "resources", "acoustics", "homeostasis", "variation", "horizons", "sources",
        }
        chemical_expected = legacy_expected | {"habitat", "biosphere", "physiology"}
        identity = (raw.get("format"), raw.get("version"))
        expected = chemical_expected if identity == (PROFILE_FORMAT_V3, 3) else legacy_expected
        if set(raw) != expected or identity not in (
            (PROFILE_FORMAT, 1), (PROFILE_FORMAT_V2, 2), (PROFILE_FORMAT_V3, 3)
        ):
            raise ValueError("unsupported embodied training profile")
        if raw["sensorium"] != {"frame": BODY_FRAME} or raw["body"] != "articulated":
            raise ValueError("embodied training v1 requires body-v1 articulated sensing")
        self._value = raw
        self.sha256 = hashlib.sha256(_canonical(raw)).hexdigest()
        # Constructing these validators catches malformed embedded configs
        # without retaining mutable simulation state.
        FiniteEnergyConfig.from_value(raw["homeostasis"])
        mappings = ("fields", "acoustics", "variation", "horizons", "sources")
        if not all(isinstance(raw[key], dict) for key in mappings):
            raise ValueError("embodied training profile components must be mappings")
        if raw["version"] < 3 and not isinstance(raw["resources"], dict):
            raise ValueError("legacy embodied profiles require ecology resources")
        if raw["version"] == 3:
            if raw["resources"] is not None or not all(
                isinstance(raw[key], dict) for key in ("habitat", "biosphere", "physiology")
            ):
                raise ValueError("chemical nursery requires a biosphere and no scalar ecology")
            if raw["physiology"] != {
                "energy": "normalized usable ATP plus 0.72 reserve against capacity",
                "gut": "normalized conserved chemical mass against gut capacity",
                "fatigue": "bounded actuator fatigue state",
                "transfer_baseline": "old-policy input range only; not calibrated to prior physiology",
            }:
                raise ValueError("chemical nursery physiology semantics differ")

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

    def with_full_bearings(self, *, training_half_span: float = math.pi) -> "EmbodiedTrainingProfile":
        """Derive a declared curriculum with unbiased held-out food bearings.

        Existing profile-v1 worlds keep their original draws. A teacher may
        choose a narrower training span at a new episode boundary, while probes
        always draw across the complete circle.
        """
        span = float(training_half_span)
        if not math.isfinite(span) or not 0 < span <= math.pi:
            raise ValueError("training food-bearing half-span must be in (0, pi]")
        value = copy.deepcopy(self._value)
        value["format"] = PROFILE_FORMAT_V2
        value["version"] = 2
        value["name"] = "current-life-full-bearing-v2"
        value["variation"].update({
            "version": 2,
            "parent_profile_sha256": self.sha256,
            "training_food_bearing_schedule_rad": [
                min(0.75, span), min(math.pi / 2.0, span), span,
            ],
            "heldout_food_bearing_half_span_rad": math.pi,
        })
        value["variation"].pop("training_food_bearing_half_span_rad", None)
        # V2 uses the three ecology-backed foods for its reachable curriculum.
        # Increased inflow, uptake, and growth let those ordinary producers
        # recover repeatedly over a 1,200 s episode; conservation remains in
        # Ecology's material/energy ledger.
        resources = value["resources"]
        resources["ambient"]["material_inflow_rate"] = 0.008
        for reservoir in resources["reservoirs"]:
            reservoir["uptake_rate"] = float(reservoir["uptake_rate"]) * 3.0
        for producer in resources["producers"]:
            producer["growth_rate"] = float(producer["growth_rate"]) * 2.5
        return type(self)(value)

    @classmethod
    def current_v2(cls) -> "EmbodiedTrainingProfile":
        """Build the three-stage full-bearing renewable nursery profile."""
        return cls.current().with_full_bearings(training_half_span=math.pi)

    @classmethod
    def chemical_nursery(
        cls, habitat: str | Path, biosphere: str | Path,
    ) -> "EmbodiedTrainingProfile":
        """Bind one fresh common-chemistry birth to the existing worker world."""
        habitat_path = Path(habitat).resolve()
        biosphere_path = Path(biosphere).resolve()
        habitat_value = json.loads(habitat_path.read_text())
        biosphere_value = json.loads(biosphere_path.read_text())
        size = habitat_value.get("size")
        if not isinstance(size, list) or len(size) != 3:
            raise ValueError("chemical nursery habitat needs a three-dimensional size")
        field_config = FieldEnvironment(size=tuple(size)).config
        source_paths = {
            "habitat": habitat_path, "biosphere_birth": biosphere_path,
            "physics": ROOT / "chreatures/physics.py",
            "articulated": ROOT / "chreatures/articulated.py",
            "sensorium": ROOT / "chreatures/sensorium.py",
            "physical_batch": ROOT / "chreatures/physical_batch.py",
            "fields": ROOT / "chreatures/fields.py",
            "acoustics_module": ROOT / "chreatures/acoustics.py",
            "biosphere_module": ROOT / "chreatures/biosphere.py",
            "somatic": ROOT / "chreatures/somatic.py",
            "material_objects": ROOT / "chreatures/material_objects.py",
            "metabolism": ROOT / "chreatures/metabolism.py",
            "growth": ROOT / "chreatures/growth.py",
            "training_environment": ROOT / "chreatures/training_environment.py",
            "native_cargo_lock": ROOT / "native/world-kernels/Cargo.lock",
            "native_cargo_manifest": ROOT / "native/world-kernels/Cargo.toml",
            "native_build": ROOT / "native/world-kernels/build.rs",
            "native_lib": ROOT / "native/world-kernels/src/lib.rs",
            "native_contacts": ROOT / "native/world-kernels/src/contacts.rs",
            "native_growth": ROOT / "native/world-kernels/src/growth.rs",
            "native_metabolism": ROOT / "native/world-kernels/src/metabolism.rs",
            "native_transport": ROOT / "native/world-kernels/src/transport.rs",
            "native_contact_shim": ROOT / "native/world-kernels/src/contact_shim.c",
        }
        return cls({
            "format": PROFILE_FORMAT_V3, "version": 3,
            "name": "common-chemistry-mobile-nursery-v3",
            "sensorium": {"frame": BODY_FRAME}, "body": "articulated",
            "habitat": habitat_value, "biosphere": biosphere_value,
            "fields": field_config, "resources": None,
            "acoustics": {"version": 1, "include_authored": True, "emitters": []},
            "homeostasis": FiniteEnergyConfig().to_value(),
            "physiology": {
                "energy": "normalized usable ATP plus 0.72 reserve against capacity",
                "gut": "normalized conserved chemical mass against gut capacity",
                "fatigue": "bounded actuator fatigue state",
                "transfer_baseline": "old-policy input range only; not calibrated to prior physiology",
            },
            "variation": {
                "version": 3, "heldout_seed_offset": 80_000_003,
                "body_heading_span_rad": math.pi,
                "fatigue_range": [0.02, 0.08],
            },
            "horizons": {
                "training_episode_steps": 1_200, "heldout_steps": 1_200,
                "telemetry_every_steps": 120, "checkpoint_every_steps": 600,
                "dt_seconds": 0.05,
                "rationale": "60 s exercises physical, somatic, metabolic and developmental boundaries",
            },
            "sources": {
                name: {"path": str(path), "sha256": _sha(path)}
                for name, path in source_paths.items()
            },
        })

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
    stage: int = 0,
    profile: EmbodiedTrainingProfile | None = None,
    base_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a varied physical world; geometry never enters policy inputs."""
    profile = profile or EmbodiedTrainingProfile.current()
    profile_version = int(profile.component("version"))
    spec = (
        copy.deepcopy(dict(base_spec)) if base_spec is not None
        else profile.component("habitat") if profile_version == 3
        else json.loads((ROOT / profile.component("sources")["habitat"]["path"]).read_text())
    )
    spec["sensorium"] = profile.component("sensorium")
    variation = profile.component("variation")
    if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)):
        raise ValueError("training stage must be an integer")
    stage = int(stage)
    if profile_version == 1 and stage != 0:
        raise ValueError("profile v1 has no staged bearing curriculum")
    chosen_seed = int(seed) + (int(variation["heldout_seed_offset"]) if held_out else 0)
    rng = np.random.default_rng(chosen_seed)
    if profile_version == 3:
        if stage != 0:
            raise ValueError("chemical nursery v3 has no staged curriculum")
        low, high = map(float, variation["fatigue_range"])
        for body in spec["bodies"]:
            body["heading"] = float(rng.uniform(
                -variation["body_heading_span_rad"], variation["body_heading_span_rad"]
            ))
            body["fatigue"] = float(rng.uniform(low, high))
        spec["name"] = (
            "common-chemistry-mobile-heldout" if held_out
            else "common-chemistry-mobile-training"
        )
        spec["training_profile_sha256"] = profile.sha256
        spec["training_variant"] = {
            "seed": chosen_seed, "held_out": bool(held_out), "stage": 0,
        }
        return spec
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
    nearby_food = (("way-berry", "screen-seed", "sun-berry") if profile_version == 1
                   else ("sun-berry", "shade-nectar", "screen-seed"))
    low, high = map(float, variation["early_food_distance_m"])
    if variation["version"] == 1:
        food_bearing_span = 0.75
    elif variation["version"] == 2:
        schedule = variation["training_food_bearing_schedule_rad"]
        if len(schedule) != 3 or stage not in range(len(schedule)):
            raise ValueError("profile-v2 training stage must be 0, 1, or 2")
        food_bearing_span = float(
            variation["heldout_food_bearing_half_span_rad"] if held_out else schedule[stage]
        )
        if (
            not math.isfinite(food_bearing_span)
            or not 0 < food_bearing_span <= math.pi
            or (held_out and food_bearing_span != math.pi)
        ):
            raise ValueError("Invalid version-2 food-bearing distribution")
    else:
        raise ValueError("Unsupported embodied environment variation")
    for body, entity_id in zip(spec["bodies"], nearby_food, strict=True):
        angle = float(body["heading"] + rng.uniform(-food_bearing_span, food_bearing_span))
        distance = float(rng.uniform(low, high))
        entity = by_id[entity_id]
        entity["position"][0] = float(np.clip(body["position"][0] + math.cos(angle) * distance, .25, width - .25))
        entity["position"][1] = float(np.clip(body["position"][1] + math.sin(angle) * distance, .25, height - .25))
        entity["position"][2] = float(body["position"][2])
    spec["name"] = "embodied-current-life-heldout" if held_out else "embodied-current-life-training"
    spec["training_profile_sha256"] = profile.sha256
    if profile_version == 1:
        # Preserve the exact v1 spec/checkpoint contract.
        spec["training_variant"] = {"seed": chosen_seed, "held_out": bool(held_out)}
    else:
        spec["training_variant"] = {
            "seed": chosen_seed, "held_out": bool(held_out), "stage": stage,
            "food_bearing_half_span_rad": food_bearing_span,
        }
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
        self.profile_version = int(profile.component("version"))
        variant = spec.get("training_variant", {})
        self.stage = int(variant.get("stage", 0))
        if self.profile_version == 2 and (
            self.stage not in range(3) or "food_bearing_half_span_rad" not in variant
        ):
            raise ValueError("profile-v2 world spec omits its curriculum stage")
        if self.profile_version == 3 and self.stage != 0:
            raise ValueError("chemical nursery v3 has no staged curriculum")
        self.physical_backend = physical_backend
        self.world = PHYSICAL_BACKENDS[physical_backend](seed=self.seed, spec=copy.deepcopy(spec))
        self.field = FieldEnvironment.from_world(self.world, profile.component("fields"))
        self.resources = None
        self.biosphere = None
        if self.profile_version == 3:
            from .biosphere import Biosphere

            self.biosphere = Biosphere.from_config(
                self.world, profile.component("biosphere")
            )
        else:
            self.resources = Ecology(
                self.world, profile.component("resources"), seed=self.seed ^ 0xEC0106
            )
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
        resources = self.resources.advance(dt) if self.resources is not None else None
        biosphere = self.biosphere.advance(dt) if self.biosphere is not None else None
        static = self.field.sync_static_geometry(self.world)
        self.field.sync_dynamic_barriers(self.world.diffusion_barriers())
        sources = self.field.sources_from_world(self.world)
        if self.biosphere is not None:
            sources.extend(self.biosphere.field_sources())
        field = self.field.advance(dt, sources=sources)
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
            "biosphere": copy.deepcopy(biosphere),
            "static_field_sync": copy.deepcopy(static),
            "acoustics": copy.deepcopy(acoustic),
            "physiology_semantics": (
                self.profile.component("physiology")
                if self.profile_version == 3 else {
                    "energy": "legacy scalar body reserve readout",
                    "gut": "legacy scalar gut fill readout",
                    "fatigue": "bounded actuator fatigue state",
                }
            ),
        }
        return outcomes

    def terminal_outcomes(self) -> dict[str, Any]:
        """Return bounded episode-end outcomes without world-state coordinates."""
        residents: dict[str, Any] = {}
        mobile = (
            self.biosphere.mobility.view()
            if self.biosphere is not None and self.biosphere.mobility is not None
            else None
        )
        for body in self.bodies:
            value = {
                "energy": float(body.energy), "gut": float(body.gut),
                "fatigue": float(body.fatigue), "speed": float(body.speed),
            }
            if mobile is not None:
                value.update(copy.deepcopy(mobile["residents"][body.id]))
            residents[body.id] = value
        return {
            "format": "chreatures-embodied-terminal-outcomes-v1",
            "time": float(self.world.time), "profile_sha256": self.profile.sha256,
            "physiology_semantics": (
                self.profile.component("physiology")
                if self.profile_version == 3 else {
                    "energy": "legacy scalar body reserve readout",
                    "gut": "legacy scalar gut fill readout",
                    "fatigue": "bounded actuator fatigue state",
                }
            ),
            "residents": residents,
            "biosphere_accounting": (
                self.biosphere.accounting() if self.biosphere is not None else None
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        value = {
            "format": SNAPSHOT_FORMAT, "version": 1,
            "seed": self.seed, "profile": self.profile.to_value(),
            "world": self.world.snapshot(), "field": self.field.snapshot(),
            "resources": (
                self.resources.snapshot() if self.resources is not None else None
            ),
            "acoustics": self.acoustics.snapshot(),
            "last_telemetry": copy.deepcopy(self.last_telemetry),
        }
        if self.profile_version == 2:
            value.update({"format": SNAPSHOT_FORMAT_V2, "version": 2, "stage": self.stage})
        elif self.profile_version == 3:
            value.update({
                "format": SNAPSHOT_FORMAT_V3, "version": 3, "stage": 0,
                "biosphere": self.biosphere.snapshot(),
            })
        return value

    @classmethod
    def restore(
        cls, snapshot: Mapping[str, Any],
        expected_profile: EmbodiedTrainingProfile | str | None = None,
        *, physical_backend: str = "reference",
    ) -> "EmbodiedTrainingWorld":
        identity = (snapshot.get("format"), snapshot.get("version"))
        if identity not in (
            (SNAPSHOT_FORMAT, 1), (SNAPSHOT_FORMAT_V2, 2), (SNAPSHOT_FORMAT_V3, 3)
        ):
            raise ValueError("unsupported embodied training world snapshot")
        profile = EmbodiedTrainingProfile.from_value(snapshot["profile"])
        if int(profile.component("version")) != int(snapshot["version"]):
            raise ValueError("training world/profile versions differ")
        expected_hash = expected_profile.sha256 if isinstance(expected_profile, EmbodiedTrainingProfile) else expected_profile
        if expected_hash is not None and str(expected_hash) != profile.sha256:
            raise ValueError("training checkpoint profile differs")
        if physical_backend not in PHYSICAL_BACKENDS:
            raise ValueError(f"unknown embodied physical backend: {physical_backend!r}")
        instance = cls.__new__(cls)
        instance.seed = int(snapshot["seed"])
        instance.profile = profile
        instance.profile_version = int(profile.component("version"))
        instance.stage = int(snapshot.get("stage", 0))
        if instance.profile_version == 2 and instance.stage not in range(3):
            raise ValueError("invalid restored curriculum stage")
        if instance.profile_version == 3 and instance.stage != 0:
            raise ValueError("invalid restored chemical nursery stage")
        instance.physical_backend = physical_backend
        instance.world = PHYSICAL_BACKENDS[physical_backend].restore(snapshot["world"])
        if instance.world.spec.get("training_profile_sha256") != profile.sha256:
            raise ValueError("restored physical world profile differs")
        if instance.profile_version == 2 and int(
            instance.world.spec.get("training_variant", {}).get("stage", -1)
        ) != instance.stage:
            raise ValueError("restored physical world curriculum stage differs")
        instance.field = FieldEnvironment.restore(snapshot["field"])
        instance.resources = None
        instance.biosphere = None
        if instance.profile_version == 3:
            if snapshot.get("resources") is not None or not isinstance(
                snapshot.get("biosphere"), Mapping
            ):
                raise ValueError("chemical nursery snapshot composition differs")
            from .biosphere import Biosphere

            instance.biosphere = Biosphere.restore(instance.world, snapshot["biosphere"])
        else:
            instance.resources = Ecology.restore(instance.world, snapshot["resources"])
        instance.acoustics = Acoustics.restore(instance.world, snapshot["acoustics"])
        instance.objective = FiniteEnergyObjective(
            FiniteEnergyConfig.from_value(profile.component("homeostasis"))
        )
        instance.last_telemetry = copy.deepcopy(snapshot.get("last_telemetry", {}))
        times = [instance.world.time, instance.field.time, instance.acoustics.time]
        times.append(
            instance.biosphere.web.time
            if instance.biosphere is not None else instance.resources.time
        )
        if max(times) - min(times) > 1e-9:
            raise ValueError("restored embodied environment clocks differ")
        return instance

    def close(self) -> None:
        self.acoustics.close()


__all__ = [
    "EmbodiedTrainingProfile", "EmbodiedTrainingWorld", "PHYSICAL_BACKENDS",
    "embodied_training_spec",
]
