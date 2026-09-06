"""Current native-family embodied training world."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .acoustics import Acoustics
from .fields import FieldEnvironment
from .homeostasis import FiniteEnergyConfig, FiniteEnergyObjective
from .physical_batch import FastArticulatedSensoriumWorld
from .sensorium import ArticulatedSensoriumWorld, profile_identity


PROFILE_FORMAT = "chreatures-embodied-nursery-family-profile-v5"
SNAPSHOT_FORMAT = "chreatures-embodied-training-world-v5"
BIOSPHERE_BIRTH_FORMAT = "chreatures-biosphere-birth-v5"
BIOSPHERE_SNAPSHOT_FORMAT = "chreatures-biosphere-v6"
ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_BACKENDS = {
    "reference": ArticulatedSensoriumWorld,
    "fast": FastArticulatedSensoriumWorld,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class EmbodiedTrainingProfile:
    """Immutable-by-copy environment contract selected explicitly by a runner."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        raw = copy.deepcopy(dict(value))
        expected = {
            "format", "version", "name", "sensorium", "body", "fields",
            "resources", "acoustics", "homeostasis", "variation", "horizons", "sources",
            "habitat", "biosphere", "physiology", "family",
        }
        if (
            set(raw) != expected
            or raw.get("format") != PROFILE_FORMAT
            or raw.get("version") != 5
        ):
            raise ValueError("unsupported current embodied training profile")
        if raw["sensorium"] != profile_identity() or raw["body"] != "articulated":
            raise ValueError("embodied training requires the current rich body sensorium")
        self._value = raw
        self.sha256 = hashlib.sha256(_canonical(raw)).hexdigest()
        # Constructing these validators catches malformed embedded configs
        # without retaining mutable simulation state.
        FiniteEnergyConfig.from_value(raw["homeostasis"])
        mappings = ("fields", "acoustics", "variation", "horizons", "sources")
        if not all(isinstance(raw[key], dict) for key in mappings):
            raise ValueError("embodied training profile components must be mappings")
        if raw["resources"] is not None or not all(
            isinstance(raw[key], dict) for key in ("habitat", "biosphere", "physiology")
        ):
            raise ValueError("current training requires a biosphere and no scalar ecology")
        birth = raw["biosphere"]
        if (
            birth.get("format") != BIOSPHERE_BIRTH_FORMAT
            or not isinstance(birth.get("illumination_cycle"), dict)
            or not isinstance(birth.get("mobile_phototrophy"), dict)
        ):
            raise ValueError("current training requires the mixotrophic birth-v5")
        illumination_sources = {
            "native_environment", "native_environment_shim", "native_illumination",
        }
        if not illumination_sources <= set(raw["sources"]):
            raise ValueError("current training omits native illumination provenance")
        if raw["physiology"] != {
            "energy": "normalized usable ATP plus 0.72 reserve against capacity",
            "gut": "normalized conserved chemical mass against gut capacity",
            "fatigue": "bounded actuator fatigue state",
            "transfer_baseline": "old-policy input range only; not calibrated to prior physiology",
        }:
            raise ValueError("current training physiology semantics differ")
        self._validate_family_identity(raw["family"])

    @staticmethod
    def _validate_family_schedule(value: Any) -> None:
        if (
            not isinstance(value, dict)
            or set(value) != {"format", "version", "selector", "training", "heldout"}
            or value.get("format") != "chreatures-nursery-family-schedule-v1"
            or value.get("version") != 1
            or value.get("selector") != "world-seed-modulo-split-v1"
        ):
            raise ValueError("invalid nursery-family schedule")
        seen: set[tuple[str, int]] = set()
        for split in ("training", "heldout"):
            variants = value[split]
            if not isinstance(variants, list) or not 3 <= len(variants) <= 64:
                raise ValueError("nursery-family splits require 3..64 variants")
            for variant in variants:
                if (
                    not isinstance(variant, dict)
                    or set(variant) != {"family", "seed"}
                    or not isinstance(variant["family"], str)
                    or not variant["family"]
                    or isinstance(variant["seed"], bool)
                    or not isinstance(variant["seed"], int)
                    or not 0 <= variant["seed"] < 2**64
                    or (variant["family"], variant["seed"]) in seen
                ):
                    raise ValueError("invalid or duplicate nursery-family variant")
                seen.add((variant["family"], variant["seed"]))

    @staticmethod
    def _validate_family_identity(value: Any) -> None:
        if (
            not isinstance(value, dict)
            or set(value) != {
                "format", "selector", "generator_config", "schedule", "transport",
                "variants",
            }
            or value.get("format") != "chreatures-nursery-family-training-identity-v1"
            or value.get("selector") != "world-seed-modulo-split-v1"
        ):
            raise ValueError("invalid nursery-family training identity")
        for key in ("generator_config", "schedule"):
            source = value[key]
            if (
                not isinstance(source, dict)
                or set(source) != {"path", "sha256"}
                or not isinstance(source["path"], str)
                or not source["path"]
                or not _valid_sha(source["sha256"])
            ):
                raise ValueError("invalid nursery-family identity source")
        if value["transport"] != {
            "residents": 6,
            "rich": 4096,
            "physical": 351,
            "physiology": 6,
            "controller": 4453,
            "readouts": 384,
        }:
            raise ValueError("nursery-family transport contract differs")
        variants = value["variants"]
        if not isinstance(variants, list) or len(variants) < 6:
            raise ValueError("nursery-family identity omits scheduled artifacts")
        seen: set[tuple[str, str, int]] = set()
        for variant in variants:
            expected = {
                "split", "index", "family", "seed", "habitat_sha256",
                "biosphere_sha256", "analyst_sha256",
            }
            identity = (
                variant.get("split"), variant.get("family"), variant.get("seed")
            ) if isinstance(variant, dict) else None
            if (
                not isinstance(variant, dict)
                or set(variant) != expected
                or variant["split"] not in {"training", "heldout"}
                or isinstance(variant["index"], bool)
                or not isinstance(variant["index"], int)
                or variant["index"] < 0
                or not isinstance(variant["family"], str)
                or not variant["family"]
                or isinstance(variant["seed"], bool)
                or not isinstance(variant["seed"], int)
                or not 0 <= variant["seed"] < 2**64
                or not all(_valid_sha(variant[key]) for key in (
                    "habitat_sha256", "biosphere_sha256", "analyst_sha256",
                ))
                or identity in seen
            ):
                raise ValueError("invalid nursery-family artifact identity")
            seen.add(identity)
        for split in ("training", "heldout"):
            indices = [item["index"] for item in variants if item["split"] == split]
            if len(indices) < 3 or indices != list(range(len(indices))):
                raise ValueError("nursery-family split indices are not contiguous")

    @classmethod
    def nursery_family(
        cls,
        habitat: str | Path,
        biosphere: str | Path,
        family_config: str | Path,
        schedule: str | Path,
    ) -> "EmbodiedTrainingProfile":
        """Pin generated nursery artifacts for cold episode construction."""
        habitat_path = Path(habitat).resolve()
        biosphere_path = Path(biosphere).resolve()
        config_path = Path(family_config).resolve()
        schedule_path = Path(schedule).resolve()
        habitat_text = habitat_path.read_text()
        biosphere_text = biosphere_path.read_text()
        config_text = config_path.read_text()
        schedule_value = json.loads(schedule_path.read_text())
        cls._validate_family_schedule(schedule_value)
        port_path = ROOT / "data/ports/retinal-v2.json"
        port = json.loads(port_path.read_text())
        if (
            port.get("physical_inputs", {}).get("count") != 351
            or port.get("readouts", {}).get("count") != 384
        ):
            raise ValueError("nursery-family retinal port dimensions differ")

        from .native_world import load_world_kernels

        native_type = getattr(load_world_kernels(), "HabitatFamily", None)
        if native_type is None:
            raise RuntimeError("native world kernels omit HabitatFamily")
        generator = native_type(config_text, _text_sha(config_text))
        allowed = set(generator.families())
        declared_training = [
            (str(variant["family"]), int(variant["seed"]))
            for variant in schedule_value["training"]
        ]
        if declared_training != list(generator.training_variants()):
            raise ValueError(
                "nursery training schedule differs from the generator manifest"
            )
        artifacts = []
        for split in ("training", "heldout"):
            for index, variant in enumerate(schedule_value[split]):
                family = str(variant["family"])
                seed = int(variant["seed"])
                if family not in allowed:
                    raise ValueError("nursery schedule names an unknown family")
                generated_habitat, generated_biosphere, analyst = generator.generate(
                    habitat_text, biosphere_text, seed, family,
                )
                # Biology and inherited resident traits are fixed across the
                # family. Only the physical arrangement varies.
                if json.loads(generated_biosphere) != json.loads(biosphere_text):
                    raise ValueError("nursery generator changed the pinned biosphere")
                generated = json.loads(generated_habitat)
                if len(generated.get("bodies", [])) != 6:
                    raise ValueError("nursery family must retain six resident bodies")
                metadata = json.loads(analyst)
                if metadata.get("runtime_visible") is not False:
                    raise ValueError("nursery analyst topology is not private")
                artifacts.append({
                    "split": split, "index": index, "family": family, "seed": seed,
                    "habitat_sha256": _text_sha(generated_habitat),
                    "biosphere_sha256": _text_sha(generated_biosphere),
                    "analyst_sha256": _text_sha(analyst),
                })

        habitat_value = json.loads(habitat_text)
        biosphere_value = json.loads(biosphere_text)
        size = habitat_value.get("size")
        if not isinstance(size, list) or len(size) != 3:
            raise ValueError("nursery-family habitat needs a three-dimensional size")
        source_paths = {
            "habitat": habitat_path,
            "biosphere_birth": biosphere_path,
            "nursery_family_config": config_path,
            "nursery_family_schedule": schedule_path,
            "physics": ROOT / "chreatures/physics.py",
            "articulated": ROOT / "chreatures/articulated.py",
            "sensorium": ROOT / "chreatures/sensorium.py",
            "sensorium_profile": ROOT / "data/sensorium/rich-body-v1.json",
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
            "native_acoustics": ROOT / "native/world-kernels/src/acoustics.rs",
            "native_acoustics_shim": ROOT / "native/world-kernels/src/acoustics_shim.c",
            "native_growth": ROOT / "native/world-kernels/src/growth.rs",
            "native_environment": ROOT / "native/world-kernels/src/environment.rs",
            "native_environment_shim": ROOT / "native/world-kernels/src/environment_shim.c",
            "native_illumination": ROOT / "native/world-kernels/src/illumination.rs",
            "native_metabolism": ROOT / "native/world-kernels/src/metabolism.rs",
            "native_sensorium": ROOT / "native/world-kernels/src/sensorium.rs",
            "native_sensorium_shim": ROOT / "native/world-kernels/src/sensorium_shim.c",
            "native_transport": ROOT / "native/world-kernels/src/transport.rs",
            "native_contact_shim": ROOT / "native/world-kernels/src/contact_shim.c",
            "native_habitat_family": ROOT / "native/world-kernels/src/habitat_family.rs",
            "retinal_port_spec": port_path,
        }
        value = {
            "format": PROFILE_FORMAT,
            "version": 5,
            "name": "common-chemistry-native-nursery-family-v5",
            "sensorium": profile_identity(),
            "body": "articulated",
            "habitat": habitat_value,
            "biosphere": biosphere_value,
            "fields": FieldEnvironment(size=tuple(size)).config,
            "resources": None,
            "acoustics": {"version": 1, "include_authored": True, "emitters": []},
            "homeostasis": FiniteEnergyConfig().to_value(),
            "physiology": {
                "energy": "normalized usable ATP plus 0.72 reserve against capacity",
                "gut": "normalized conserved chemical mass against gut capacity",
                "fatigue": "bounded actuator fatigue state",
                "transfer_baseline": "old-policy input range only; not calibrated to prior physiology",
            },
            "family": {
                "format": "chreatures-nursery-family-training-identity-v1",
                "selector": schedule_value["selector"],
                "generator_config": {
                    "path": str(config_path), "sha256": _sha(config_path),
                },
                "schedule": {
                    "path": str(schedule_path), "sha256": _sha(schedule_path),
                },
                "transport": {
                    "residents": 6,
                    "rich": 4096,
                    "physical": 351,
                    "physiology": 6,
                    "controller": 4453,
                    "readouts": 384,
                },
                "variants": artifacts,
            },
            "variation": {
                "version": 5, "heldout_seed_offset": 80_000_003,
                "body_heading_span_rad": math.pi,
                "fatigue_range": [0.02, 0.08],
            },
            "horizons": {
                "training_episode_steps": 1_200, "heldout_steps": 1_200,
                "telemetry_every_steps": 120, "checkpoint_every_steps": 600,
                "dt_seconds": 0.05,
                "rationale": (
                    "60 s cold episodes rotate connected nursery structures while "
                    "retaining finite chemistry and physical encounters"
                ),
            },
            "sources": {
                name: {"path": str(path), "sha256": _sha(path)}
                for name, path in source_paths.items()
            },
        }
        return cls(value)

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


def _generated_family_spec(
    profile: EmbodiedTrainingProfile, chosen_seed: int, held_out: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Regenerate and verify one declared cold-episode artifact."""
    identity = profile.component("family")
    split = "heldout" if held_out else "training"
    variants = [item for item in identity["variants"] if item["split"] == split]
    selected = variants[chosen_seed % len(variants)]
    sources = profile.component("sources")
    habitat_path = Path(sources["habitat"]["path"])
    biosphere_path = Path(sources["biosphere_birth"]["path"])
    config_path = Path(identity["generator_config"]["path"])
    schedule_path = Path(identity["schedule"]["path"])
    for path, expected in (
        (habitat_path, sources["habitat"]["sha256"]),
        (biosphere_path, sources["biosphere_birth"]["sha256"]),
        (config_path, identity["generator_config"]["sha256"]),
        (schedule_path, identity["schedule"]["sha256"]),
    ):
        if _sha(path) != expected:
            raise ValueError("nursery-family cold source checksum differs")
    habitat_text = habitat_path.read_text()
    biosphere_text = biosphere_path.read_text()
    config_text = config_path.read_text()
    from .native_world import load_world_kernels

    generator = load_world_kernels().HabitatFamily(
        config_text, identity["generator_config"]["sha256"],
    )
    habitat_output, biosphere_output, analyst_output = generator.generate(
        habitat_text, biosphere_text, selected["seed"], selected["family"],
    )
    actual = {
        "habitat_sha256": _text_sha(habitat_output),
        "biosphere_sha256": _text_sha(biosphere_output),
        "analyst_sha256": _text_sha(analyst_output),
    }
    if any(actual[key] != selected[key] for key in actual):
        raise ValueError("generated nursery artifact differs from training identity")
    if json.loads(biosphere_output) != profile.component("biosphere"):
        raise ValueError("generated nursery biosphere differs from training profile")
    variant = {
        "selector": identity["selector"],
        "split": split,
        "split_index": selected["index"],
        "family": selected["family"],
        "family_seed": selected["seed"],
        "family_output_sha256": actual,
    }
    return json.loads(habitat_output), variant


def _validate_selected_family(
    profile: EmbodiedTrainingProfile, variant: Mapping[str, Any],
) -> None:
    expected = {
        "seed", "held_out", "stage", "selector", "split", "split_index",
        "family", "family_seed", "family_output_sha256",
    }
    family = profile.component("family")
    if (
        not isinstance(variant, Mapping)
        or set(variant) != expected
        or variant["selector"] != family["selector"]
        or variant["split"] != ("heldout" if variant["held_out"] else "training")
    ):
        raise ValueError("nursery-family world omits its artifact identity")
    matched = next((item for item in family["variants"] if (
        item["split"] == variant["split"]
        and item["index"] == variant["split_index"]
        and item["family"] == variant["family"]
        and item["seed"] == variant["family_seed"]
    )), None)
    if matched is None or variant["family_output_sha256"] != {
        key: matched[key] for key in (
            "habitat_sha256", "biosphere_sha256", "analyst_sha256",
        )
    }:
        raise ValueError("nursery-family world artifact differs from profile")


def embodied_training_spec(
    seed: int, *, held_out: bool = False, stage: int = 0,
    profile: EmbodiedTrainingProfile,
) -> dict[str, Any]:
    """Generate one verified family world; geometry never enters policy inputs."""
    if not isinstance(profile, EmbodiedTrainingProfile):
        raise TypeError("profile must be the current EmbodiedTrainingProfile")
    if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)):
        raise ValueError("training stage must be an integer")
    if int(stage) != 0:
        raise ValueError("nursery-family training requires stage 0")
    variation = profile.component("variation")
    chosen_seed = int(seed) + (
        int(variation["heldout_seed_offset"]) if held_out else 0
    )
    spec, family_variant = _generated_family_spec(profile, chosen_seed, held_out)
    spec["sensorium"] = profile.component("sensorium")
    rng = np.random.default_rng(chosen_seed)
    low, high = map(float, variation["fatigue_range"])
    for body in spec["bodies"]:
        body["heading"] = float(rng.uniform(
            -variation["body_heading_span_rad"], variation["body_heading_span_rad"]
        ))
        body["fatigue"] = float(rng.uniform(low, high))
    spec["name"] = (
        "native-nursery-family-heldout"
        if held_out else "native-nursery-family-training"
    )
    spec["training_profile_sha256"] = profile.sha256
    spec["training_variant"] = {
        "seed": chosen_seed,
        "held_out": bool(held_out),
        "stage": 0,
        **family_variant,
    }
    return spec


class EmbodiedTrainingWorld:
    """Drop-in worker world with body-v1, diffusion and shared chemistry."""

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
        variant = spec.get("training_variant", {})
        self.stage = int(variant.get("stage", 0))
        if self.stage != 0:
            raise ValueError("nursery-family world requires stage 0")
        _validate_selected_family(profile, variant)
        self.physical_backend = physical_backend
        self.world = PHYSICAL_BACKENDS[physical_backend](
            seed=self.seed, spec=copy.deepcopy(spec)
        )
        if len(self.world.bodies) != profile.component(
            "family"
        )["transport"]["residents"]:
            raise ValueError("nursery-family physical resident count differs")
        self.field = FieldEnvironment.from_world(self.world, profile.component("fields"))
        from .biosphere import Biosphere

        self.biosphere = Biosphere.from_config(
            self.world, profile.component("biosphere")
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
        biosphere = self.biosphere.advance(dt)
        static = self.field.sync_static_geometry(self.world)
        self.field.sync_dynamic_barriers(self.world.diffusion_barriers())
        sources = self.field.sources_from_world(self.world)
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
            "family_variant": copy.deepcopy(self.world.spec["training_variant"]),
            "nutrition": float(sum(item["nutrition"] for item in outcomes.values())),
            "absorbed": float(sum(item["nutrition"] for item in outcomes.values())),
            "ingested_mass": float(sum(
                item.get("ingested_mass", 0.0) for item in outcomes.values()
            )),
            "mouth_material_contacts": int(sum(
                item.get("mouth_material_contacts", 0) for item in outcomes.values()
            )),
            "eat_requests": int(sum(
                float(actions.get(body.id, {}).get("eat", 0.0)) > 0
                for body in self.bodies
            )),
            "distance": float(sum(item["distance"] for item in outcomes.values())),
            "contacts": int(sum(item["contact"] > 0 for item in outcomes.values())),
            "effort_mean": float(np.mean([item["effort"] for item in outcomes.values()])),
            "homeostatic_reward_sum": float(sum(rewards)),
            "field": copy.deepcopy(field),
            "biosphere": copy.deepcopy(biosphere),
            "static_field_sync": copy.deepcopy(static),
            "acoustics": copy.deepcopy(acoustic),
            "physiology_semantics": self.profile.component("physiology"),
        }
        return outcomes

    def terminal_outcomes(self) -> dict[str, Any]:
        """Return bounded episode-end outcomes without world-state coordinates."""
        residents: dict[str, Any] = {}
        mobile = self.biosphere.mobility.view() if self.biosphere.mobility is not None else None
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
            "family_variant": copy.deepcopy(self.world.spec["training_variant"]),
            "physiology_semantics": self.profile.component("physiology"),
            "residents": residents,
            "biosphere_accounting": self.biosphere.accounting(),
        }

    def snapshot(self) -> dict[str, Any]:
        biosphere_snapshot = self.biosphere.snapshot()
        if biosphere_snapshot.get("format") != BIOSPHERE_SNAPSHOT_FORMAT:
            raise RuntimeError("world did not produce the current biosphere snapshot")
        return {
            "format": SNAPSHOT_FORMAT, "version": 5, "stage": 0,
            "seed": self.seed, "profile": self.profile.to_value(),
            "world": self.world.snapshot(), "field": self.field.snapshot(),
            "resources": None,
            "biosphere": biosphere_snapshot,
            "acoustics": self.acoustics.snapshot(),
            "last_telemetry": copy.deepcopy(self.last_telemetry),
            "family_variant": copy.deepcopy(self.world.spec["training_variant"]),
        }

    @classmethod
    def restore(
        cls, snapshot: Mapping[str, Any],
        expected_profile: EmbodiedTrainingProfile | str | None = None,
        *, physical_backend: str = "reference",
    ) -> "EmbodiedTrainingWorld":
        if (snapshot.get("format"), snapshot.get("version")) != (
            SNAPSHOT_FORMAT, 5,
        ):
            raise ValueError("unsupported current training world snapshot")
        profile = EmbodiedTrainingProfile.from_value(snapshot["profile"])
        expected_hash = (
            expected_profile.sha256
            if isinstance(expected_profile, EmbodiedTrainingProfile)
            else expected_profile
        )
        if expected_hash is not None and str(expected_hash) != profile.sha256:
            raise ValueError("training checkpoint profile differs")
        if physical_backend not in PHYSICAL_BACKENDS:
            raise ValueError(f"unknown embodied physical backend: {physical_backend!r}")
        instance = cls.__new__(cls)
        instance.seed = int(snapshot["seed"])
        instance.profile = profile
        instance.stage = int(snapshot.get("stage", 0))
        if instance.stage != 0:
            raise ValueError("invalid restored nursery-family stage")
        instance.physical_backend = physical_backend
        instance.world = PHYSICAL_BACKENDS[physical_backend].restore(snapshot["world"])
        if instance.world.spec.get("training_profile_sha256") != profile.sha256:
            raise ValueError("restored physical world profile differs")
        if snapshot.get("family_variant") != instance.world.spec.get("training_variant"):
            raise ValueError("restored nursery-family identity differs")
        _validate_selected_family(profile, instance.world.spec.get("training_variant", {}))
        instance.field = FieldEnvironment.restore(snapshot["field"])
        if snapshot.get("resources") is not None or not isinstance(
            snapshot.get("biosphere"), Mapping
        ) or snapshot["biosphere"].get("format") != BIOSPHERE_SNAPSHOT_FORMAT:
            raise ValueError("nursery-family snapshot composition differs")
        from .biosphere import Biosphere

        instance.biosphere = Biosphere.restore(instance.world, snapshot["biosphere"])
        instance.acoustics = Acoustics.restore(instance.world, snapshot["acoustics"])
        instance.objective = FiniteEnergyObjective(
            FiniteEnergyConfig.from_value(profile.component("homeostasis"))
        )
        instance.last_telemetry = copy.deepcopy(snapshot.get("last_telemetry", {}))
        times = [instance.world.time, instance.field.time, instance.acoustics.time]
        times.append(instance.biosphere.web.time)
        if max(times) - min(times) > 1e-9:
            raise ValueError("restored embodied environment clocks differ")
        return instance

    def close(self) -> None:
        self.acoustics.close()


__all__ = [
    "EmbodiedTrainingProfile", "EmbodiedTrainingWorld", "PHYSICAL_BACKENDS",
    "embodied_training_spec",
]
