"""Cold-build boundary for native inherited regional habitats."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .native_world import load_world_kernels


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@dataclass(frozen=True)
class GeneratedRegion:
    """Runtime inputs plus analyst-only provenance kept on the host side."""

    habitat: Mapping[str, Any]
    biosphere: Mapping[str, Any]
    analyst: Mapping[str, Any]
    habitat_sha256: str
    biosphere_sha256: str
    analyst_sha256: str


class RegionalHabitatFamily:
    """Pin templates and delegate all seeded regional generation to Rust."""

    def __init__(
        self,
        config: str | Path,
        habitat_template: str | Path,
        biosphere_template: str | Path,
    ) -> None:
        self.config_path = Path(config).resolve()
        self.habitat_path = Path(habitat_template).resolve()
        self.biosphere_path = Path(biosphere_template).resolve()
        self.config_text = self.config_path.read_text()
        self.habitat_text = self.habitat_path.read_text()
        self.biosphere_text = self.biosphere_path.read_text()
        config_value = json.loads(self.config_text)
        if config_value.get("habitat_template_sha256") != _sha_bytes(
            self.habitat_text.encode()
        ) or config_value.get("biosphere_template_sha256") != _sha_bytes(
            self.biosphere_text.encode()
        ):
            raise ValueError("regional source hashes differ from pinned templates")
        native = load_world_kernels()
        native_type = getattr(native, "HabitatFamily", None)
        if native_type is None:
            raise RuntimeError("native world kernels omit HabitatFamily")
        self._native = native_type(
            self.config_text, _sha_bytes(self.config_text.encode())
        )

    @property
    def archetypes(self) -> tuple[str, ...]:
        return tuple(self._native.archetypes())

    @property
    def training_genomes(self) -> tuple[tuple[str, int, int, int, str], ...]:
        return tuple(tuple(value) for value in self._native.training_genomes())

    def initial_genome(
        self,
        *,
        seed: int,
        archetype: str,
        resident_count: int,
        profile_sha256: str,
        epoch: int = 0,
    ) -> Mapping[str, Any]:
        return json.loads(
            self._native.initial_genome(
                seed, archetype, resident_count, profile_sha256, epoch
            )
        )

    def mutate_genome(
        self,
        parent: Mapping[str, Any],
        *,
        parent_environment_record_sha256: str,
        variation_seed: int,
    ) -> Mapping[str, Any]:
        return json.loads(
            self._native.mutate_genome(
                _canonical(parent).decode(),
                parent_environment_record_sha256,
                variation_seed,
            )
        )

    def generate(
        self,
        genome: Mapping[str, Any],
        residents: Mapping[str, Any],
    ) -> GeneratedRegion:
        resident_count = int(genome["parameters"]["resident_count"])
        resident_values = residents.get("residents")
        if (
            residents.get("format") != "chreatures-regional-residents-v3"
            or not isinstance(resident_values, list)
            or len(resident_values) < resident_count
        ):
            raise ValueError("regional resident bundle lacks requested capacity")
        selected_residents = {
            "format": "chreatures-regional-residents-v3",
            "residents": resident_values[:resident_count],
        }
        habitat_text, biosphere_text, analyst_text = self._native.generate(
            self.habitat_text,
            self.biosphere_text,
            _canonical(genome).decode(),
            _canonical(selected_residents).decode(),
        )
        habitat = json.loads(habitat_text)
        biosphere = json.loads(biosphere_text)
        analyst = json.loads(analyst_text)
        if (
            biosphere.get("format") != "chreatures-biosphere-birth-v7"
            or analyst.get("format") != "chreatures-regional-analyst-v4"
            or analyst.get("runtime_visible") is not False
        ):
            raise ValueError("regional analyst geometry must remain host-private")
        record = analyst.get("environment_record")
        if (
            not isinstance(record, dict)
            or record.get("format") != "chreatures-environment-record-v4"
            or record.get("genome_sha256") != genome.get("sha256")
        ):
            raise ValueError("generated region omits environment ancestry")
        if record.get("topology_sha256") != _sha_bytes(
            habitat_text.encode()
        ) or record.get("resource_sha256") != _sha_bytes(biosphere_text.encode()):
            raise ValueError("generated environment record hashes differ")
        matter = biosphere.get("regional_matter")
        matter_analyst = analyst.get("regional_matter")
        if (
            not isinstance(matter, dict)
            or matter.get("format") != "chreatures-regional-matter-v1"
            or not isinstance(matter_analyst, dict)
            or matter_analyst.get("sha256") != _sha_bytes(_canonical(matter))
        ):
            raise ValueError("generated regional matter identity differs")
        compartments = biosphere.get("compartments")
        regions = matter.get("regions")
        if (
            not isinstance(compartments, list)
            or not isinstance(regions, list)
            or len(regions) != len(analyst.get("graph", {}).get("nodes", []))
            or len(
                {region.get("row") for region in regions if isinstance(region, dict)}
            )
            != len(regions)
            or any(
                not isinstance(region, dict)
                or not isinstance(region.get("row"), int)
                or not 0 <= region["row"] < len(compartments)
                or not isinstance(region.get("position"), list)
                or len(region["position"]) != 3
                for region in regions
            )
            or matter.get("world_size_m") != genome["parameters"]["dimensions_m"]
        ):
            raise ValueError("generated regional matter rows differ")
        return GeneratedRegion(
            habitat=habitat,
            biosphere=biosphere,
            analyst=analyst,
            habitat_sha256=_sha_bytes(habitat_text.encode()),
            biosphere_sha256=_sha_bytes(biosphere_text.encode()),
            analyst_sha256=_sha_bytes(analyst_text.encode()),
        )
