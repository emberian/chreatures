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
from .organism_interface import (
    ACTION_DIM, MAX_RESIDENTS, OBSERVATION_DIM, PHYSIOLOGY_DIM,
    PHYSIOLOGY_NAMES, identity as organism_identity,
)


PROFILE_FORMAT = "chreatures-embodied-nursery-family-profile-v7"
PROFILE_VERSION = 7
SNAPSHOT_FORMAT = "chreatures-embodied-training-world-v8"
BIOSPHERE_BIRTH_FORMAT = "chreatures-biosphere-birth-v7"
BIOSPHERE_SNAPSHOT_FORMAT = "chreatures-biosphere-v9"
ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_BACKENDS = {
    "reference": ArticulatedSensoriumWorld,
    "fast": FastArticulatedSensoriumWorld,
}


def physiology_identity() -> dict[str, Any]:
    return {
        "ordered_names": list(PHYSIOLOGY_NAMES),
        "energy": "normalized usable ATP plus 0.72 reserve against capacity",
        "gut": "normalized conserved chemical mass against gut capacity",
        "fatigue": "bounded actuator fatigue state",
        "development": "private funded structure, gland and brood state",
        "neural_support": "private connectome support state",
        "units": "synthetic conserved material and chemical energy; not calibrated joules",
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

    def __init__(
        self, value: Mapping[str, Any], locators: Mapping[str, str]
    ) -> None:
        raw = copy.deepcopy(dict(value))
        expected = {
            "format", "version", "name", "sensorium", "body", "fields",
            "resources", "acoustics", "homeostasis", "variation", "horizons", "sources",
            "habitat", "biosphere", "physiology", "family", "organism_interface",
        }
        if (
            set(raw) != expected
            or raw.get("format") != PROFILE_FORMAT
            or raw.get("version") != PROFILE_VERSION
        ):
            raise ValueError("unsupported current embodied training profile")
        if raw["sensorium"] != profile_identity() or raw["body"] != "articulated":
            raise ValueError("embodied training requires the current rich body sensorium")
        self._value = raw
        self.sha256 = hashlib.sha256(_canonical(raw)).hexdigest()
        if raw["organism_interface"] != organism_identity():
            raise ValueError("training organism interface differs")
        # Constructing these validators catches malformed embedded configs
        # without retaining mutable simulation state.
        FiniteEnergyConfig.from_value(raw["homeostasis"])
        mappings = ("fields", "acoustics", "variation", "horizons", "sources")
        if not all(isinstance(raw[key], dict) for key in mappings):
            raise ValueError("embodied training profile components must be mappings")
        sources = raw["sources"]
        if not sources or any(
            not isinstance(name, str)
            or not name
            or not isinstance(source, dict)
            or set(source) != {"sha256"}
            or not _valid_sha(source["sha256"])
            for name, source in sources.items()
        ):
            raise ValueError("training profile source identities differ")
        locator_values = copy.deepcopy(dict(locators))
        if set(locator_values) != set(sources) or any(
            not isinstance(path, str) or not path or not Path(path).is_absolute()
            for path in locator_values.values()
        ):
            raise ValueError("training profile locators differ")
        self._locators = locator_values
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
            raise ValueError("current training requires the developmental birth-v6")
        illumination_sources = {
            "native_environment", "native_environment_shim", "native_illumination",
        }
        if not illumination_sources <= set(raw["sources"]):
            raise ValueError("current training omits native illumination provenance")
        if raw["physiology"] != physiology_identity():
            raise ValueError("current training physiology semantics differ")
        self._validate_family_identity(raw["family"], sources)

    @staticmethod
    def _validate_family_schedule(value: Any) -> None:
        expected = {
            "format", "version", "selector", "resident_count", "training", "heldout",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("format") != "chreatures-regional-environment-schedule-v1"
            or value.get("version") != 1
            or value.get("selector") != "environment-genome-round-robin-v1"
            or isinstance(value.get("resident_count"), bool)
            or not isinstance(value.get("resident_count"), int)
            or not 1 <= value["resident_count"] <= MAX_RESIDENTS
        ):
            raise ValueError("invalid current regional environment schedule")
        seen: set[tuple[str, int]] = set()
        for split in ("training", "heldout"):
            variants = value[split]
            if not isinstance(variants, list) or not 3 <= len(variants) <= 64:
                raise ValueError("regional splits require 3..64 genomes")
            for variant in variants:
                if (
                    not isinstance(variant, dict)
                    or set(variant) != {"archetype", "seed", "epoch", "profile_sha256"}
                    or not isinstance(variant["archetype"], str)
                    or not variant["archetype"]
                    or isinstance(variant["seed"], bool)
                    or not isinstance(variant["seed"], int)
                    or not 0 <= variant["seed"] < 2**64
                    or isinstance(variant["epoch"], bool)
                    or not isinstance(variant["epoch"], int)
                    or variant["epoch"] < 0
                    or not _valid_sha(variant["profile_sha256"])
                    or (variant["archetype"], variant["seed"]) in seen
                ):
                    raise ValueError("invalid or duplicate regional environment genome")
                seen.add((variant["archetype"], variant["seed"]))

    @staticmethod
    def _validate_family_identity(
        value: Any, sources: Mapping[str, Mapping[str, str]]
    ) -> None:
        if (
            not isinstance(value, dict)
            or set(value) != {
                "format", "selector", "generator_config", "schedule",
                "resident_bundle", "transport", "variants",
            }
            or value.get("format") != "chreatures-regional-training-identity-v4"
            or value.get("selector") != "environment-genome-round-robin-v1"
        ):
            raise ValueError("invalid regional training identity")
        for key in ("generator_config", "schedule", "resident_bundle"):
            source = value[key]
            if (
                not isinstance(source, dict)
                or set(source) != {"sha256"}
                or not _valid_sha(source["sha256"])
            ):
                raise ValueError("invalid regional identity source")
        for key, source_name in (
            ("generator_config", "regional_family_config"),
            ("schedule", "regional_schedule"),
            ("resident_bundle", "regional_residents"),
        ):
            if value[key]["sha256"] != sources.get(source_name, {}).get("sha256"):
                raise ValueError("regional identity source differs")
        transport = value["transport"]
        if transport != {
            "residents": transport.get("residents") if isinstance(transport, dict) else None,
            "max_residents": MAX_RESIDENTS,
            "rich": 4096,
            "physical": 351,
            "physiology": PHYSIOLOGY_DIM,
            "controller": OBSERVATION_DIM,
            "readouts": 384,
            "actions": ACTION_DIM,
        } or isinstance(transport["residents"], bool) or not isinstance(
            transport["residents"], int
        ) or not 1 <= transport["residents"] <= MAX_RESIDENTS:
            raise ValueError("regional transport contract differs")
        variants = value["variants"]
        if not isinstance(variants, list) or len(variants) < 6:
            raise ValueError("regional identity omits scheduled artifacts")
        seen: set[tuple[str, int]] = set()
        expected = {
            "split", "index", "archetype", "seed", "epoch", "genome_sha256",
            "environment_sha256", "habitat_sha256", "biosphere_sha256",
            "analyst_sha256", "dimensions_m", "resident_count", "environment_record",
        }
        for variant in variants:
            identity = (variant.get("split"), variant.get("index")) if isinstance(variant, dict) else None
            record = variant.get("environment_record") if isinstance(variant, dict) else None
            record_body = copy.deepcopy(record) if isinstance(record, dict) else {}
            parents = record.get("parents") if isinstance(record, dict) else None
            genome_parents = (
                record.get("genome_parents") if isinstance(record, dict) else None
            )
            variation = record.get("variation") if isinstance(record, dict) else None
            descriptors = record.get("descriptors") if isinstance(record, dict) else None
            generation_cost = (
                record.get("generation_cost") if isinstance(record, dict) else None
            )
            if record_body:
                record_body["sha256"] = ""
            if (
                not isinstance(variant, dict) or set(variant) != expected
                or variant["split"] not in {"training", "heldout"}
                or isinstance(variant["index"], bool) or not isinstance(variant["index"], int)
                or variant["index"] < 0 or not isinstance(variant["archetype"], str)
                or not variant["archetype"] or isinstance(variant["seed"], bool)
                or not isinstance(variant["seed"], int) or not 0 <= variant["seed"] < 2**64
                or isinstance(variant["epoch"], bool) or not isinstance(variant["epoch"], int)
                or variant["epoch"] < 0 or variant["resident_count"] != transport["residents"]
                or not isinstance(variant["environment_record"], dict)
                or variant["environment_record"].get("sha256") != variant["environment_sha256"]
                or set(record) != {
                    "format", "sha256", "genome_sha256", "genome_parents",
                    "parents", "variation", "topology_sha256",
                    "resource_sha256", "profile_sha256", "epoch", "descriptors",
                    "generation_cost",
                }
                or record.get("format") != "chreatures-environment-record-v3"
                or record.get("genome_sha256") != variant["genome_sha256"]
                or hashlib.sha256(_canonical(record_body)).hexdigest() != record["sha256"]
                or not isinstance(parents, list) or len(parents) > 2
                or not all(_valid_sha(parent) for parent in parents)
                or not isinstance(genome_parents, list) or len(genome_parents) > 2
                or not all(_valid_sha(parent) for parent in genome_parents)
                or len(genome_parents) != len(parents)
                or not isinstance(variation, dict)
                or set(variation) != {"operator", "seed", "recipe_sha256"}
                or not isinstance(variation["operator"], str) or not variation["operator"]
                or isinstance(variation["seed"], bool) or not isinstance(variation["seed"], int)
                or not 0 <= variation["seed"] < 2**64
                or not _valid_sha(variation["recipe_sha256"])
                or record.get("topology_sha256") != variant["habitat_sha256"]
                or record.get("resource_sha256") != variant["biosphere_sha256"]
                or not _valid_sha(record.get("profile_sha256"))
                or record.get("epoch") != variant["epoch"]
                or not isinstance(descriptors, dict)
                or set(descriptors) != {
                    "regional_scale", "elevation_relief", "resource_density",
                    "renewal_rate", "connectivity",
                }
                or not all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(item)
                    and 0.0 <= item <= 1.0
                    for item in descriptors.values()
                )
                or not isinstance(generation_cost, dict)
                or set(generation_cost) != {
                    "physical_geoms", "regions", "edges", "movables",
                    "compartments", "normalized",
                }
                or not all(
                    isinstance(generation_cost[key], int)
                    and not isinstance(generation_cost[key], bool)
                    and generation_cost[key] >= 0
                    for key in (
                        "physical_geoms", "regions", "edges", "movables",
                        "compartments",
                    )
                )
                or not isinstance(generation_cost["normalized"], (int, float))
                or isinstance(generation_cost["normalized"], bool)
                or not math.isfinite(generation_cost["normalized"])
                or not 0.0 <= generation_cost["normalized"] <= 1.0
                or not isinstance(variant["dimensions_m"], list) or len(variant["dimensions_m"]) != 3
                or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in variant["dimensions_m"])
                or not all(_valid_sha(variant[key]) for key in (
                    "genome_sha256", "environment_sha256", "habitat_sha256",
                    "biosphere_sha256", "analyst_sha256",
                )) or identity in seen
            ):
                raise ValueError("invalid regional artifact identity")
            seen.add(identity)
        for split in ("training", "heldout"):
            indices = [item["index"] for item in variants if item["split"] == split]
            if len(indices) < 3 or indices != list(range(len(indices))):
                raise ValueError("regional split indices are not contiguous")

    @classmethod
    def nursery_family(
        cls,
        habitat: str | Path,
        biosphere: str | Path,
        family_config: str | Path,
        schedule: str | Path,
    ) -> "EmbodiedTrainingProfile":
        """Pin current regional genomes and their generated physical ecology."""
        habitat_path = Path(habitat).resolve()
        biosphere_path = Path(biosphere).resolve()
        config_path = Path(family_config).resolve()
        schedule_path = Path(schedule).resolve()
        resident_path = config_path.with_name("regional-residents-v3.json")
        paths = (habitat_path, biosphere_path, config_path, schedule_path, resident_path)
        if any(not path.is_file() for path in paths):
            raise ValueError("regional profile source is absent")
        habitat_text = habitat_path.read_text()
        biosphere_text = biosphere_path.read_text()
        config_text = config_path.read_text()
        residents_text = resident_path.read_text()
        schedule_value = json.loads(schedule_path.read_text())
        residents_value = json.loads(residents_text)
        cls._validate_family_schedule(schedule_value)
        if (
            residents_value.get("format") != "chreatures-regional-residents-v3"
            or not isinstance(residents_value.get("residents"), list)
            or len(residents_value["residents"]) != MAX_RESIDENTS
        ):
            raise ValueError("regional resident founder bundle must declare capacity 32")
        campaign_residents = {
            "format": "chreatures-regional-residents-v3",
            "residents": residents_value["residents"][:schedule_value["resident_count"]],
        }
        campaign_residents_text = _canonical(campaign_residents).decode()
        port_path = ROOT / "data/ports/retinal-v2.json"
        port = json.loads(port_path.read_text())
        if port.get("physical_inputs", {}).get("count") != 351 or port.get(
            "readouts", {}
        ).get("count") != 384:
            raise ValueError("regional retinal port dimensions differ")

        from .native_world import load_world_kernels

        native_type = getattr(load_world_kernels(), "HabitatFamily", None)
        if native_type is None:
            raise RuntimeError("native world kernels omit HabitatFamily")
        generator = native_type(config_text, _text_sha(config_text))
        allowed = set(generator.archetypes())
        declared_training = [(
            str(item["archetype"]), int(item["seed"]), schedule_value["resident_count"],
            int(item["epoch"]), str(item["profile_sha256"]),
        ) for item in schedule_value["training"]]
        if declared_training != [tuple(value) for value in generator.training_genomes()]:
            raise ValueError("regional training schedule differs from generator founders")
        artifacts = []
        for split in ("training", "heldout"):
            for index, item in enumerate(schedule_value[split]):
                archetype = str(item["archetype"])
                if archetype not in allowed:
                    raise ValueError("regional schedule names an unknown archetype")
                genome_text = generator.initial_genome(
                    int(item["seed"]), archetype, schedule_value["resident_count"],
                    str(item["profile_sha256"]), int(item["epoch"]),
                )
                habitat_output, biosphere_output, analyst = generator.generate(
                    habitat_text, biosphere_text, genome_text, campaign_residents_text,
                )
                generated = json.loads(habitat_output)
                generated_biosphere = json.loads(biosphere_output)
                metadata = json.loads(analyst)
                record = metadata.get("environment_record", {})
                genome = json.loads(genome_text)
                if (
                    metadata.get("runtime_visible") is not False
                    or len(generated.get("bodies", [])) != schedule_value["resident_count"]
                    or len(generated_biosphere.get("mobiles", [])) != schedule_value["resident_count"]
                    or record.get("topology_sha256") != _text_sha(habitat_output)
                    or record.get("resource_sha256") != _text_sha(biosphere_output)
                ):
                    raise ValueError("regional generator violated its physical identity")
                artifacts.append({
                    "split": split, "index": index, "archetype": archetype,
                    "seed": int(item["seed"]), "epoch": int(item["epoch"]),
                    "genome_sha256": genome["sha256"],
                    "environment_sha256": record["sha256"],
                    "environment_record": copy.deepcopy(record),
                    "habitat_sha256": _text_sha(habitat_output),
                    "biosphere_sha256": _text_sha(biosphere_output),
                    "analyst_sha256": _text_sha(analyst),
                    "dimensions_m": list(generated["size"]),
                    "resident_count": schedule_value["resident_count"],
                })

        # Embed one concrete current runtime artifact in the profile. Episode
        # birth still regenerates its selected hash-pinned variant from the
        # immutable source templates recorded below.
        habitat_value = json.loads(habitat_output)
        biosphere_value = json.loads(biosphere_output)
        size = habitat_value.get("size")
        if not isinstance(size, list) or len(size) != 3:
            raise ValueError("regional template needs a three-dimensional size")
        source_paths = {
            "habitat": habitat_path,
            "biosphere_birth": biosphere_path,
            "regional_family_config": config_path,
            "regional_schedule": schedule_path,
            "regional_residents": resident_path,
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
            "habitat_family_host": ROOT / "chreatures/habitat_family.py",
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
        source_identities = {
            name: {"sha256": _sha(path)} for name, path in source_paths.items()
        }
        locators = {name: str(path) for name, path in source_paths.items()}
        value = {
            "format": PROFILE_FORMAT, "version": PROFILE_VERSION,
            "name": "common-chemistry-native-regional-population-v7",
            "organism_interface": organism_identity(), "sensorium": profile_identity(),
            "body": "articulated", "habitat": habitat_value,
            "biosphere": biosphere_value,
            "fields": FieldEnvironment(size=tuple(size)).config,
            "resources": None,
            "acoustics": {"version": 1, "include_authored": True, "emitters": []},
            "homeostasis": FiniteEnergyConfig().to_value(),
            "physiology": physiology_identity(),
            "family": {
                "format": "chreatures-regional-training-identity-v4",
                "selector": schedule_value["selector"],
                "generator_config": {"sha256": _sha(config_path)},
                "schedule": {"sha256": _sha(schedule_path)},
                "resident_bundle": {"sha256": _sha(resident_path)},
                "transport": {
                    "residents": schedule_value["resident_count"],
                    "max_residents": MAX_RESIDENTS, "rich": 4096, "physical": 351,
                    "physiology": PHYSIOLOGY_DIM, "controller": OBSERVATION_DIM,
                    "readouts": 384, "actions": ACTION_DIM,
                },
                "variants": artifacts,
            },
            "variation": {
                "version": PROFILE_VERSION, "heldout_seed_offset": 80_000_003,
                "body_heading_span_rad": math.pi, "fatigue_range": [0.02, 0.08],
            },
            "horizons": {
                "training_episode_steps": 1_200, "heldout_steps": 1_200,
                "telemetry_every_steps": 120, "checkpoint_every_steps": 600,
                "dt_seconds": 0.05,
                "rationale": "60 s cold episodes rotate inherited connected regional ecologies",
            },
            "sources": source_identities,
        }
        return cls(value, locators)

    def to_value(self) -> dict[str, Any]:
        """Encode semantic identity and host locators as distinct fields."""
        return {
            "value": self.semantic_value(),
            "sha256": self.sha256,
            "locators": self.locator_manifest(),
        }

    def semantic_value(self) -> dict[str, Any]:
        """Return the content-bound value authenticated by ``sha256``."""
        return copy.deepcopy(self._value)

    def locator_manifest(self) -> dict[str, str]:
        """Return host paths excluded from semantic profile identity."""
        return copy.deepcopy(self._locators)

    @classmethod
    def from_value(
        cls,
        encoded: Mapping[str, Any],
        *,
        locators: Mapping[str, str] | None = None,
    ) -> "EmbodiedTrainingProfile":
        if not isinstance(encoded, Mapping) or set(encoded) != {
            "value", "sha256", "locators",
        }:
            raise ValueError("invalid encoded embodied training profile")
        encoded_locators = encoded["locators"]
        semantic_sources = (
            encoded["value"].get("sources")
            if isinstance(encoded["value"], Mapping)
            else None
        )
        if (
            not isinstance(encoded_locators, Mapping)
            or not isinstance(semantic_sources, Mapping)
            or set(encoded_locators) != set(semantic_sources)
            or any(
                not isinstance(path, str) or not path or not Path(path).is_absolute()
                for path in encoded_locators.values()
            )
        ):
            raise ValueError("invalid encoded embodied training profile locators")
        selected_locators = encoded["locators"] if locators is None else locators
        if not isinstance(selected_locators, Mapping):
            raise ValueError("invalid embodied training profile locators")
        profile = cls(encoded["value"], selected_locators)
        if encoded["sha256"] != profile.sha256:
            raise ValueError("embodied training profile checksum differs")
        if locators is not None:
            for name in profile._locators:
                profile.source_path(name)
        return profile

    def component(self, name: str) -> Any:
        return copy.deepcopy(self._value[name])

    def source_path(self, name: str) -> Path:
        """Resolve one transport locator and authenticate its semantic bytes."""
        if name not in self._value["sources"]:
            raise KeyError(f"unknown training profile source: {name}")
        path = Path(self._locators[name])
        if not path.is_file() or _sha(path) != self._value["sources"][name]["sha256"]:
            raise ValueError(f"training profile source differs: {name}")
        return path

    @property
    def name(self) -> str:
        return str(self._value["name"])


def _generated_family_spec(
    profile: EmbodiedTrainingProfile, chosen_seed: int, held_out: bool,
    environment: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Regenerate and verify one pinned regional environment genome."""
    identity = profile.component("family")
    split = "heldout" if held_out else "training"
    variants = [item for item in identity["variants"] if item["split"] == split]
    if environment is None:
        selected = variants[chosen_seed % len(variants)]
    else:
        if (
            set(environment) != {"split", "index"} or environment["split"] != split
            or isinstance(environment["index"], bool)
            or not isinstance(environment["index"], int)
            or not 0 <= environment["index"] < len(variants)
        ):
            raise ValueError("explicit population environment is outside the pinned split")
        selected = variants[environment["index"]]
    sources = profile.component("sources")
    habitat_path = profile.source_path("habitat")
    biosphere_path = profile.source_path("biosphere_birth")
    config_path = profile.source_path("regional_family_config")
    resident_path = profile.source_path("regional_residents")
    schedule_path = profile.source_path("regional_schedule")
    for path, expected in (
        (habitat_path, sources["habitat"]["sha256"]),
        (biosphere_path, sources["biosphere_birth"]["sha256"]),
        (config_path, identity["generator_config"]["sha256"]),
        (resident_path, identity["resident_bundle"]["sha256"]),
        (schedule_path, identity["schedule"]["sha256"]),
    ):
        if _sha(path) != expected:
            raise ValueError("regional cold source checksum differs")
    habitat_text, biosphere_text = habitat_path.read_text(), biosphere_path.read_text()
    config_text = config_path.read_text()
    resident_source = json.loads(resident_path.read_text())
    residents_text = _canonical({
        "format": "chreatures-regional-residents-v3",
        "residents": resident_source["residents"][:identity["transport"]["residents"]],
    }).decode()
    from .native_world import load_world_kernels
    generator = load_world_kernels().HabitatFamily(config_text, _text_sha(config_text))
    schedule = json.loads(schedule_path.read_text())
    schedule_item = schedule[split][selected["index"]]
    genome_output = generator.initial_genome(
        selected["seed"], selected["archetype"], selected["resident_count"],
        schedule_item["profile_sha256"], selected["epoch"],
    )
    habitat_output, biosphere_output, analyst_output = generator.generate(
        habitat_text, biosphere_text, genome_output, residents_text,
    )
    metadata = json.loads(analyst_output)
    actual = {
        "genome_sha256": json.loads(genome_output)["sha256"],
        "environment_sha256": metadata["environment_record"]["sha256"],
        "habitat_sha256": _text_sha(habitat_output),
        "biosphere_sha256": _text_sha(biosphere_output),
        "analyst_sha256": _text_sha(analyst_output),
    }
    if any(actual[key] != selected[key] for key in actual):
        raise ValueError("generated regional artifact differs from training identity")
    variant = {
        "selector": identity["selector"], "split": split,
        "split_index": selected["index"], "archetype": selected["archetype"],
        "environment_seed": selected["seed"], "environment_epoch": selected["epoch"],
        "environment_genome_sha256": actual["genome_sha256"],
        "environment_sha256": actual["environment_sha256"],
        "environment_record": copy.deepcopy(metadata["environment_record"]),
        "family_output_sha256": {
            key: actual[key] for key in ("habitat_sha256", "biosphere_sha256", "analyst_sha256")
        },
    }
    return json.loads(habitat_output), json.loads(biosphere_output), variant


def _validate_selected_family(
    profile: EmbodiedTrainingProfile, variant: Mapping[str, Any],
) -> None:
    expected = {
        "seed", "held_out", "stage", "selector", "split", "split_index",
        "archetype", "environment_seed", "environment_epoch",
        "environment_genome_sha256", "environment_sha256", "environment_record",
        "family_output_sha256",
    }
    family = profile.component("family")
    if (
        not isinstance(variant, Mapping) or set(variant) != expected
        or variant["selector"] != family["selector"]
        or variant["split"] != ("heldout" if variant["held_out"] else "training")
    ):
        raise ValueError("regional world omits its environment identity")
    matched = next((item for item in family["variants"] if (
        item["split"] == variant["split"] and item["index"] == variant["split_index"]
        and item["archetype"] == variant["archetype"]
        and item["seed"] == variant["environment_seed"]
        and item["epoch"] == variant["environment_epoch"]
    )), None)
    if (
        matched is None
        or variant["environment_genome_sha256"] != matched["genome_sha256"]
        or variant["environment_sha256"] != matched["environment_sha256"]
        or variant["environment_record"] != matched["environment_record"]
        or variant["family_output_sha256"] != {
            key: matched[key]
            for key in ("habitat_sha256", "biosphere_sha256", "analyst_sha256")
        }
    ):
        raise ValueError("regional world artifact differs from profile")


def embodied_training_spec(
    seed: int, *, held_out: bool = False, stage: int = 0,
    profile: EmbodiedTrainingProfile,
    environment: Mapping[str, Any] | None = None,
    candidates: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate one verified family world; geometry never enters policy inputs."""
    if not isinstance(profile, EmbodiedTrainingProfile):
        raise TypeError("profile must be the current EmbodiedTrainingProfile")
    if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)):
        raise ValueError("training stage must be an integer")
    if int(stage) != 0:
        raise ValueError("regional training requires stage 0")
    variation = profile.component("variation")
    chosen_seed = int(seed) + (
        int(variation["heldout_seed_offset"]) if held_out else 0
    )
    spec, biosphere_birth, family_variant = _generated_family_spec(
        profile, chosen_seed, held_out, environment
    )
    if candidates is not None:
        from .population import compose_population_birth

        spec, biosphere_birth, receipt = compose_population_birth(
            spec, biosphere_birth, candidates
        )
        spec["population_birth"] = receipt
    spec["biosphere_birth"] = biosphere_birth
    spec["sensorium"] = profile.component("sensorium")
    rng = np.random.default_rng(chosen_seed)
    low, high = map(float, variation["fatigue_range"])
    for body in spec["bodies"]:
        body["heading"] = float(rng.uniform(
            -variation["body_heading_span_rad"], variation["body_heading_span_rad"]
        ))
        body["fatigue"] = float(rng.uniform(low, high))
    spec["name"] = (
        "native-regional-heldout"
        if held_out else "native-regional-training"
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
            raise ValueError("regional world requires stage 0")
        _validate_selected_family(profile, variant)
        self.physical_backend = physical_backend
        self.world = PHYSICAL_BACKENDS[physical_backend](
            seed=self.seed, spec=copy.deepcopy(spec)
        )
        if len(self.world.bodies) != profile.component(
            "family"
        )["transport"]["residents"]:
            raise ValueError("regional physical resident count differs")
        # Regional size belongs to the generated world, while transport and
        # chemical constants remain pinned in the profile.
        fields = profile.component("fields")
        base_size = np.asarray(profile.component("habitat")["size"], dtype=np.float64)
        spacing = base_size / np.asarray(fields["grid"], dtype=np.float64)
        fields["grid"] = np.maximum(4, np.ceil(np.asarray(spec["size"]) / spacing)).astype(int).tolist()
        self.field = FieldEnvironment.from_world(self.world, fields)
        from .biosphere import Biosphere

        self.biosphere = Biosphere.from_config(
            self.world, spec["biosphere_birth"]
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

    def physiology_rows(self) -> np.ndarray:
        """Private somatic organs, with neural support bound by the cohort host."""
        if self.biosphere.mobility is None:
            raise RuntimeError("population worlds require mobile developmental physiology")
        return np.ascontiguousarray([
            self.biosphere.mobility.normalized12(body.id, neural_support=1.0)
            for body in self.bodies
        ], dtype=np.float32)

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
            "format": SNAPSHOT_FORMAT, "version": 8, "stage": 0,
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
            SNAPSHOT_FORMAT, 8,
        ):
            raise ValueError("unsupported current training world snapshot")
        rebound_locators = (
            expected_profile.locator_manifest()
            if isinstance(expected_profile, EmbodiedTrainingProfile)
            else None
        )
        profile = EmbodiedTrainingProfile.from_value(
            snapshot["profile"], locators=rebound_locators
        )
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
            raise ValueError("invalid restored regional stage")
        instance.physical_backend = physical_backend
        instance.world = PHYSICAL_BACKENDS[physical_backend].restore(snapshot["world"])
        if instance.world.spec.get("training_profile_sha256") != profile.sha256:
            raise ValueError("restored physical world profile differs")
        if snapshot.get("family_variant") != instance.world.spec.get("training_variant"):
            raise ValueError("restored regional identity differs")
        _validate_selected_family(profile, instance.world.spec.get("training_variant", {}))
        instance.field = FieldEnvironment.restore(snapshot["field"])
        if snapshot.get("resources") is not None or not isinstance(
            snapshot.get("biosphere"), Mapping
        ) or snapshot["biosphere"].get("format") != BIOSPHERE_SNAPSHOT_FORMAT:
            raise ValueError("regional snapshot composition differs")
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
