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
from .sensorium import ArticulatedSensoriumWorld, profile_identity


PROFILE_FORMAT = "chreatures-embodied-training-profile-v1"
PROFILE_FORMAT_V2 = "chreatures-embodied-training-profile-v2"
PROFILE_FORMAT_V3 = "chreatures-embodied-chemical-nursery-profile-v3"
PROFILE_FORMAT_V4 = "chreatures-embodied-chemical-encounter-profile-v4"
PROFILE_FORMAT_V5 = "chreatures-embodied-nursery-family-profile-v5"
SNAPSHOT_FORMAT = "chreatures-embodied-training-world-v1"
SNAPSHOT_FORMAT_V2 = "chreatures-embodied-training-world-v2"
SNAPSHOT_FORMAT_V3 = "chreatures-embodied-training-world-v3"
SNAPSHOT_FORMAT_V4 = "chreatures-embodied-training-world-v4"
SNAPSHOT_FORMAT_V5 = "chreatures-embodied-training-world-v5"
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
        legacy_expected = {
            "format", "version", "name", "sensorium", "body", "fields",
            "resources", "acoustics", "homeostasis", "variation", "horizons", "sources",
        }
        chemical_expected = legacy_expected | {"habitat", "biosphere", "physiology"}
        encounter_expected = chemical_expected | {"conditions"}
        family_expected = chemical_expected | {"family"}
        identity = (raw.get("format"), raw.get("version"))
        expected = (
            family_expected if identity == (PROFILE_FORMAT_V5, 5)
            else encounter_expected if identity == (PROFILE_FORMAT_V4, 4)
            else chemical_expected if identity == (PROFILE_FORMAT_V3, 3)
            else legacy_expected
        )
        if set(raw) != expected or identity not in (
            (PROFILE_FORMAT, 1), (PROFILE_FORMAT_V2, 2),
            (PROFILE_FORMAT_V3, 3), (PROFILE_FORMAT_V4, 4),
            (PROFILE_FORMAT_V5, 5),
        ):
            raise ValueError("unsupported embodied training profile")
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
        if raw["version"] < 3 and not isinstance(raw["resources"], dict):
            raise ValueError("legacy embodied profiles require ecology resources")
        if raw["version"] in (3, 4, 5):
            if raw["resources"] is not None or not all(
                isinstance(raw[key], dict) for key in ("habitat", "biosphere", "physiology")
            ):
                raise ValueError("chemical nursery requires a biosphere and no scalar ecology")
            birth = raw["biosphere"]
            if (
                birth.get("format") != BIOSPHERE_BIRTH_FORMAT
                or not isinstance(birth.get("illumination_cycle"), dict)
                or not isinstance(birth.get("mobile_phototrophy"), dict)
            ):
                raise ValueError("chemical training requires the current mixotrophic birth-v5")
            illumination_sources = {
                "native_environment",
                "native_environment_shim",
                "native_illumination",
            }
            if not illumination_sources <= set(raw["sources"]):
                raise ValueError("chemical training omits native illumination provenance")
            if raw["physiology"] != {
                "energy": "normalized usable ATP plus 0.72 reserve against capacity",
                "gut": "normalized conserved chemical mass against gut capacity",
                "fatigue": "bounded actuator fatigue state",
                "transfer_baseline": "old-policy input range only; not calibrated to prior physiology",
            }:
                raise ValueError("chemical nursery physiology semantics differ")
        if raw["version"] == 4:
            self._validate_encounter_conditions(raw["conditions"])
        if raw["version"] == 5:
            self._validate_family_identity(raw["family"])

    @staticmethod
    def _validate_encounter_conditions(value: Any) -> None:
        expected = {
            "format", "version", "name", "resource_abundance",
            "initial_chemistry", "placement", "horizons",
        }
        if (
            not isinstance(value, dict) or set(value) != expected
            or value.get("format") != "chreatures-chemical-resource-encounter-conditions-v1"
            or value.get("version") != 1
        ):
            raise ValueError("invalid chemical encounter conditions")
        abundance = value["resource_abundance"]
        chemistry = value["initial_chemistry"]
        placement = value["placement"]
        horizons = value["horizons"]
        if set(abundance) != {"nearby_packet_count", "packet_stock_scale"}:
            raise ValueError("chemical resource abundance fields differ")
        if (
            isinstance(abundance["nearby_packet_count"], bool)
            or not isinstance(abundance["nearby_packet_count"], int)
            or abundance["nearby_packet_count"] < 1
            or not 0 < float(abundance["packet_stock_scale"]) <= 1
        ):
            raise ValueError("invalid chemical resource abundance")
        if set(chemistry) != {
            "body_atp_fraction", "body_reserve_fraction", "gut_pool_fraction",
        } or any(not 0 <= float(chemistry[key]) <= 1 for key in chemistry):
            raise ValueError("invalid chemical initial-state fractions")
        if set(placement) != {
            "maximum_build_attempts", "boundary_margin_m", "vertical_reach_m",
            "surface_clearance_m", "training_schedule", "heldout",
        } or not 1 <= int(placement["maximum_build_attempts"]) <= 128:
            raise ValueError("invalid chemical placement controls")
        if not 0.1 <= float(placement["boundary_margin_m"]) <= 1:
            raise ValueError("invalid chemical placement boundary")
        if not 0.05 <= float(placement["vertical_reach_m"]) <= 1:
            raise ValueError("invalid chemical placement vertical reach")
        if not 0 < float(placement["surface_clearance_m"]) <= 0.02:
            raise ValueError("invalid chemical placement surface clearance")
        schedule = placement["training_schedule"]
        if not isinstance(schedule, list) or len(schedule) != 3:
            raise ValueError("chemical encounter curriculum requires three stages")
        for distribution in [*schedule, placement["heldout"]]:
            if not isinstance(distribution, dict) or set(distribution) != {
                "radius_m", "bearing_half_span_rad",
            }:
                raise ValueError("invalid chemical placement distribution")
            radius = distribution["radius_m"]
            bearing = float(distribution["bearing_half_span_rad"])
            if (
                not isinstance(radius, list) or len(radius) != 2
                or not 0.2 <= float(radius[0]) < float(radius[1]) <= 2
                or not 0 < bearing <= math.pi
            ):
                raise ValueError("invalid chemical placement extent")
        required_horizons = {
            "training_episode_steps", "heldout_steps", "telemetry_every_steps",
            "checkpoint_every_steps", "dt_seconds",
        }
        if set(horizons) != required_horizons or any(
            float(horizons[key]) <= 0 for key in required_horizons
        ):
            raise ValueError("invalid chemical encounter horizons")

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
            "sensorium_profile": ROOT / "data/sensorium/rich-body-v1.json",
            "native_cargo_lock": ROOT / "native/world-kernels/Cargo.lock",
            "native_cargo_manifest": ROOT / "native/world-kernels/Cargo.toml",
            "native_build": ROOT / "native/world-kernels/build.rs",
            "native_lib": ROOT / "native/world-kernels/src/lib.rs",
            "native_sensorium": ROOT / "native/world-kernels/src/sensorium.rs",
            "native_sensorium_shim": ROOT / "native/world-kernels/src/sensorium_shim.c",
            "fields": ROOT / "chreatures/fields.py",
            "ecology": ROOT / "chreatures/ecology.py",
            "acoustics_module": ROOT / "chreatures/acoustics.py",
            "homeostasis": ROOT / "chreatures/homeostasis.py",
            "training_environment": ROOT / "chreatures/training_environment.py",
        }
        return cls({
            "format": PROFILE_FORMAT,
            "version": 1,
            "name": "current-life-rich-body-v1-diffusion-terrarium-v1",
            "sensorium": profile_identity(),
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
        }
        return cls({
            "format": PROFILE_FORMAT_V3, "version": 3,
            "name": "common-chemistry-mobile-nursery-v3",
            "sensorium": profile_identity(), "body": "articulated",
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
    def chemical_encounters(
        cls, habitat: str | Path, biosphere: str | Path,
        conditions: str | Path,
    ) -> "EmbodiedTrainingProfile":
        """Create the staged finite-packet encounter profile from data."""
        base = cls.chemical_nursery(habitat, biosphere)
        value = copy.deepcopy(base._value)
        conditions_path = Path(conditions).resolve()
        condition_value = json.loads(conditions_path.read_text())
        cls._validate_encounter_conditions(condition_value)
        birth = value["biosphere"]
        if (
            birth.get("format") != BIOSPHERE_BIRTH_FORMAT
            or birth.get("exchange") is not None
        ):
            raise ValueError("chemical encounter profile requires exchange=None")
        mobiles = birth.get("mobiles")
        materials = birth.get("material_objects")
        if not isinstance(mobiles, list) or not mobiles or not isinstance(materials, dict):
            raise ValueError("chemical encounter profile requires mobile bodies and materials")
        material_rows = [item["row"] for item in materials.get("objects", [])]
        nearby_count = int(condition_value["resource_abundance"]["nearby_packet_count"])
        if nearby_count > len(material_rows):
            raise ValueError("nearby packet count exceeds physical material objects")
        chemistry = condition_value["initial_chemistry"]
        compartments = birth["compartments"]
        for mobile in mobiles:
            body = compartments[mobile["body_row"]]
            gut = compartments[mobile["gut_row"]]
            body["atp"] = float(body["atp_capacity"]) * float(
                chemistry["body_atp_fraction"]
            )
            body["pools"]["reserve"] = float(mobile["reserve_capacity"]) * float(
                chemistry["body_reserve_fraction"]
            )
            gut_fraction = float(chemistry["gut_pool_fraction"])
            gut["pools"] = {
                name: float(amount) * gut_fraction
                for name, amount in gut["pools"].items()
            }
        stock_scale = float(condition_value["resource_abundance"]["packet_stock_scale"])
        for row in material_rows:
            compartments[row]["pools"] = {
                name: float(amount) * stock_scale
                for name, amount in compartments[row]["pools"].items()
            }
        value.update({
            "format": PROFILE_FORMAT_V4,
            "version": 4,
            "name": "common-chemistry-finite-packet-encounters-v4",
            "biosphere": birth,
            "conditions": condition_value,
            "variation": {
                "version": 4,
                "heldout_seed_offset": 80_000_003,
                "body_heading_span_rad": math.pi,
                "fatigue_range": [0.02, 0.08],
                "conditions_sha256": hashlib.sha256(_canonical(condition_value)).hexdigest(),
            },
            "horizons": {
                **condition_value["horizons"],
                "rationale": (
                    "250 s episodes expose acquisition and digestion while three stages "
                    "broaden physical packet range and bearing"
                ),
            },
        })
        value["sources"]["encounter_conditions"] = {
            "path": str(conditions_path), "sha256": _sha(conditions_path),
        }
        # Re-pin the compiler because chemical_nursery hashed it before this
        # constructor's source was fully interpreted.
        compiler = ROOT / "chreatures/training_environment.py"
        value["sources"]["training_environment"] = {
            "path": str(compiler), "sha256": _sha(compiler),
        }
        return cls(value)

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

        base = cls.chemical_nursery(habitat_path, biosphere_path)
        value = copy.deepcopy(base._value)
        value.update({
            "format": PROFILE_FORMAT_V5,
            "version": 5,
            "name": "common-chemistry-native-nursery-family-v5",
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
        })
        value["sources"].update({
            "nursery_family_config": {
                "path": str(config_path), "sha256": _sha(config_path),
            },
            "nursery_family_schedule": {
                "path": str(schedule_path), "sha256": _sha(schedule_path),
            },
            "native_habitat_family": {
                "path": str(ROOT / "native/world-kernels/src/habitat_family.rs"),
                "sha256": _sha(ROOT / "native/world-kernels/src/habitat_family.rs"),
            },
            "retinal_port_spec": {
                "path": str(port_path), "sha256": _sha(port_path),
            },
        })
        compiler = ROOT / "chreatures/training_environment.py"
        value["sources"]["training_environment"] = {
            "path": str(compiler), "sha256": _sha(compiler),
        }
        return cls(value)

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
    seed: int, *, held_out: bool = False,
    stage: int = 0,
    profile: EmbodiedTrainingProfile | None = None,
    base_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a varied physical world; geometry never enters policy inputs."""
    profile = profile or EmbodiedTrainingProfile.current()
    profile_version = int(profile.component("version"))
    variation = profile.component("variation")
    chosen_seed = int(seed) + (int(variation["heldout_seed_offset"]) if held_out else 0)
    family_variant = None
    if base_spec is not None:
        if profile_version == 5:
            raise ValueError("nursery-family worlds require native cold generation")
        spec = copy.deepcopy(dict(base_spec))
    elif profile_version == 5:
        spec, family_variant = _generated_family_spec(profile, chosen_seed, held_out)
    elif profile_version in (3, 4):
        spec = profile.component("habitat")
    else:
        spec = json.loads(
            (ROOT / profile.component("sources")["habitat"]["path"]).read_text()
        )
    spec["sensorium"] = profile.component("sensorium")
    if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)):
        raise ValueError("training stage must be an integer")
    stage = int(stage)
    if profile_version == 1 and stage != 0:
        raise ValueError("profile v1 has no staged bearing curriculum")
    rng = np.random.default_rng(chosen_seed)
    if profile_version in (3, 4, 5):
        if profile_version == 3 and stage != 0:
            raise ValueError("chemical nursery v3 has no staged curriculum")
        if profile_version == 4 and stage not in range(3):
            raise ValueError("chemical encounter profile stage must be 0, 1, or 2")
        if profile_version == 5 and stage != 0:
            raise ValueError("nursery-family profile requires stage 0")
        low, high = map(float, variation["fatigue_range"])
        for body in spec["bodies"]:
            body["heading"] = float(rng.uniform(
                -variation["body_heading_span_rad"], variation["body_heading_span_rad"]
            ))
            body["fatigue"] = float(rng.uniform(low, high))
        spec["name"] = (
            "native-nursery-family-heldout" if held_out and profile_version == 5
            else "native-nursery-family-training" if profile_version == 5
            else "common-chemistry-encounter-heldout" if held_out and profile_version == 4
            else "common-chemistry-encounter-training" if profile_version == 4
            else "common-chemistry-mobile-heldout" if held_out
            else "common-chemistry-mobile-training"
        )
        spec["training_profile_sha256"] = profile.sha256
        spec["training_variant"] = {
            "seed": chosen_seed, "held_out": bool(held_out), "stage": stage,
        }
        if family_variant is not None:
            spec["training_variant"].update(family_variant)
        if profile_version == 4:
            conditions = profile.component("conditions")
            distribution = (
                conditions["placement"]["heldout"] if held_out
                else conditions["placement"]["training_schedule"][stage]
            )
            spec["training_variant"].update({
                "placement_distribution": copy.deepcopy(distribution),
                "conditions_sha256": variation["conditions_sha256"],
            })
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
        if self.profile_version == 4 and (
            self.stage not in range(3) or "placement_distribution" not in variant
        ):
            raise ValueError("chemical encounter world omits its placement stage")
        if self.profile_version == 5:
            if self.stage != 0:
                raise ValueError("nursery-family world requires stage 0")
            _validate_selected_family(profile, variant)
        self.physical_backend = physical_backend
        if self.profile_version == 4:
            self.world = self._build_encounter_world(copy.deepcopy(spec))
        else:
            self.world = PHYSICAL_BACKENDS[physical_backend](
                seed=self.seed, spec=copy.deepcopy(spec)
            )
        if self.profile_version == 5 and len(self.world.bodies) != profile.component(
            "family"
        )["transport"]["residents"]:
            raise ValueError("nursery-family physical resident count differs")
        self.field = FieldEnvironment.from_world(self.world, profile.component("fields"))
        self.resources = None
        self.biosphere = None
        if self.profile_version in (3, 4, 5):
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

    def _build_encounter_world(self, spec: dict[str, Any]):
        """Place finite packets, rejecting contacts using the actual MuJoCo model."""
        conditions = self.profile.component("conditions")
        placement = conditions["placement"]
        count = int(conditions["resource_abundance"]["nearby_packet_count"])
        material_ids = [
            str(item["entity"])
            for item in self.profile.component("biosphere")["material_objects"]["objects"]
        ][:count]
        entities = {item["id"]: item for item in spec["entities"]}
        if len(material_ids) != count or any(item not in entities for item in material_ids):
            raise ValueError("physical packet entities differ from chemical materials")
        bodies = spec["bodies"]
        if not bodies:
            raise ValueError("chemical encounter world needs mobile residents")
        distribution = spec["training_variant"]["placement_distribution"]
        low, high = map(float, distribution["radius_m"])
        span = float(distribution["bearing_half_span_rad"])
        margin = float(placement["boundary_margin_m"])
        vertical_reach = float(placement["vertical_reach_m"])
        clearance = float(placement["surface_clearance_m"])
        width, height = map(float, spec["size"][:2])
        for attempt in range(int(placement["maximum_build_attempts"])):
            candidate = copy.deepcopy(spec)
            candidate_entities = {item["id"]: item for item in candidate["entities"]}
            rng = np.random.default_rng(
                int(spec["training_variant"]["seed"]) ^ 0xC4E11 ^ (attempt * 0x9E3779B1)
            )
            realized = []
            for index, entity_id in enumerate(material_ids):
                body = bodies[index % len(bodies)]
                radius = float(rng.uniform(low, high))
                offset = float(rng.uniform(-span, span))
                bearing = float(body["heading"] + offset)
                position = [
                    float(np.clip(
                        float(body["position"][0]) + radius * math.cos(bearing),
                        margin, width - margin,
                    )),
                    float(np.clip(
                        float(body["position"][1]) + radius * math.sin(bearing),
                        margin, height - margin,
                    )),
                    0.0,
                ]
                candidate_entities[entity_id]["position"] = position
                realized.append({
                    "entity": entity_id, "resident": str(body["id"]),
                    "radius_m": radius, "bearing_offset_rad": offset,
                    "position": position,
                })
            # Query the complete native scene for the actual supporting surface.
            # Material packets are omitted only from this placement probe so they
            # cannot occlude each other's downward rays.
            probe_spec = copy.deepcopy(candidate)
            probe_spec["entities"] = [
                item for item in probe_spec["entities"] if item["id"] not in material_ids
            ]
            probe = PHYSICAL_BACKENDS[self.physical_backend](
                seed=self.seed, spec=probe_spec
            )
            valid_surfaces = True
            try:
                for item in realized:
                    point = item["position"]
                    distance, geom_id = probe._ray(
                        np.asarray([point[0], point[1], probe.depth - 1e-4]),
                        np.asarray([0.0, 0.0, -1.0]), -1,
                    )
                    supporting_entity = probe._geom_entity.get(int(geom_id))
                    if distance < 0 or supporting_entity is None:
                        valid_surfaces = False
                        break
                    try:
                        support = probe._entity(supporting_entity)
                    except (KeyError, ValueError):
                        valid_surfaces = False
                        break
                    if support["mobility"] != "static":
                        valid_surfaces = False
                        break
                    surface_z = float(probe.depth - 1e-4 - distance)
                    resident = next(
                        body for body in bodies if body["id"] == item["resident"]
                    )
                    if abs(surface_z - float(resident["position"][2])) > vertical_reach:
                        valid_surfaces = False
                        break
                    packet = candidate_entities[item["entity"]]
                    shapes = packet.get("shapes", [])
                    if (
                        len(shapes) != 1 or shapes[0].get("type") != "sphere"
                        or len(shapes[0].get("size", [])) != 1
                    ):
                        raise ValueError("chemical encounter packets must be single spheres")
                    point[2] = surface_z + float(shapes[0]["size"][0]) + clearance
                    packet["position"] = point
                    item["supporting_entity"] = supporting_entity
                    item["support_surface_z"] = surface_z
            finally:
                if hasattr(probe, "close"):
                    probe.close()
            if not valid_surfaces:
                continue
            candidate["training_variant"]["placement_attempt"] = attempt
            candidate["training_variant"]["packet_placements"] = realized
            world = PHYSICAL_BACKENDS[self.physical_backend](
                seed=self.seed, spec=candidate
            )
            if not self._packets_have_initial_contacts(world, set(material_ids)):
                return world
            if hasattr(world, "close"):
                world.close()
        raise RuntimeError("could not place chemical packets without physical intersections")

    @staticmethod
    def _packets_have_initial_contacts(world: Any, packet_ids: set[str]) -> bool:
        for index in range(int(world.data.ncon)):
            contact = world.data.contact[index]
            first = world._geom_entity.get(int(contact.geom1))
            second = world._geom_entity.get(int(contact.geom2))
            if first in packet_ids or second in packet_ids:
                return True
        return False

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
            "family_variant": (
                copy.deepcopy(self.world.spec["training_variant"])
                if self.profile_version == 5 else None
            ),
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
            "resources": copy.deepcopy(resources),
            "biosphere": copy.deepcopy(biosphere),
            "static_field_sync": copy.deepcopy(static),
            "acoustics": copy.deepcopy(acoustic),
            "physiology_semantics": (
                self.profile.component("physiology")
                if self.profile_version in (3, 4, 5) else {
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
            "family_variant": (
                copy.deepcopy(self.world.spec["training_variant"])
                if self.profile_version == 5 else None
            ),
            "physiology_semantics": (
                self.profile.component("physiology")
                if self.profile_version in (3, 4, 5) else {
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
        biosphere_snapshot = None
        if self.profile_version in (3, 4, 5):
            biosphere_snapshot = self.biosphere.snapshot()
            if biosphere_snapshot.get("format") != BIOSPHERE_SNAPSHOT_FORMAT:
                raise RuntimeError(
                    "chemical world did not produce the current biosphere snapshot"
                )
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
                "biosphere": biosphere_snapshot,
            })
        elif self.profile_version == 4:
            value.update({
                "format": SNAPSHOT_FORMAT_V4, "version": 4, "stage": self.stage,
                "biosphere": biosphere_snapshot,
            })
        elif self.profile_version == 5:
            value.update({
                "format": SNAPSHOT_FORMAT_V5, "version": 5, "stage": 0,
                "biosphere": biosphere_snapshot,
                "family_variant": copy.deepcopy(
                    self.world.spec["training_variant"]
                ),
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
            (SNAPSHOT_FORMAT, 1), (SNAPSHOT_FORMAT_V2, 2),
            (SNAPSHOT_FORMAT_V3, 3), (SNAPSHOT_FORMAT_V4, 4),
            (SNAPSHOT_FORMAT_V5, 5),
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
        if instance.profile_version == 4 and instance.stage not in range(3):
            raise ValueError("invalid restored chemical encounter stage")
        if instance.profile_version == 5 and instance.stage != 0:
            raise ValueError("invalid restored nursery-family stage")
        instance.physical_backend = physical_backend
        instance.world = PHYSICAL_BACKENDS[physical_backend].restore(snapshot["world"])
        if instance.world.spec.get("training_profile_sha256") != profile.sha256:
            raise ValueError("restored physical world profile differs")
        if instance.profile_version in (2, 4) and int(
            instance.world.spec.get("training_variant", {}).get("stage", -1)
        ) != instance.stage:
            raise ValueError("restored physical world curriculum stage differs")
        if instance.profile_version == 4 and instance.world.spec.get(
            "training_variant", {}
        ).get("conditions_sha256") != profile.component("variation")["conditions_sha256"]:
            raise ValueError("restored physical world encounter conditions differ")
        if instance.profile_version == 5 and snapshot.get(
            "family_variant"
        ) != instance.world.spec.get("training_variant"):
            raise ValueError("restored nursery-family identity differs")
        if instance.profile_version == 5:
            _validate_selected_family(
                profile, instance.world.spec.get("training_variant", {})
            )
        instance.field = FieldEnvironment.restore(snapshot["field"])
        instance.resources = None
        instance.biosphere = None
        if instance.profile_version in (3, 4, 5):
            if snapshot.get("resources") is not None or not isinstance(
                snapshot.get("biosphere"), Mapping
            ) or snapshot["biosphere"].get("format") != BIOSPHERE_SNAPSHOT_FORMAT:
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
