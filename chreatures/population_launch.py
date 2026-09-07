"""Authenticated preparation contracts for current population collection."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .organism_interface import identity as organism_identity
from .population import (
    CandidateGenome,
    canonical_bytes,
    current_parameter_recipe,
)
from .resident_contract import (
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
)
from .training_environment import EmbodiedTrainingProfile

FOUNDING_BANK_FORMAT = "chreatures-rich-collection-founding-bank-v1"
FOUNDING_BANK_VERSION = 1
DESCRIPTOR_RECIPE = "physical-population-descriptor-v2"
QUALITY_RECIPE = "finite-life-quality-v2"
SPATIAL_CELL_SCALE = 256.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def valid_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def resident_artifact_identity(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one current native controller format without loading weights."""
    artifact_path = Path(path).resolve()
    with np.load(artifact_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    bank = metadata.get("population_adapters")
    training = metadata.get("training_identity")
    if (
        metadata.get("format") != NATIVE_POPULATION_FORMAT
        or metadata.get("version") != NATIVE_POPULATION_VERSION
        or metadata.get("execution") != NATIVE_EXECUTION
        or not isinstance(bank, dict)
        or not isinstance(bank.get("count"), int)
        or isinstance(bank.get("count"), bool)
        or not 1 <= bank["count"] <= 4096
        or not isinstance(bank.get("rank"), int)
        or isinstance(bank.get("rank"), bool)
        or bank["rank"] < 1
        or not isinstance(training, dict)
    ):
        raise ValueError("controller is not the current native population artifact")
    identity = {
        "file_sha256": file_sha256(artifact_path),
        "artifact_sha256": valid_sha256(
            metadata.get("artifact_sha256"), "controller artifact"
        ),
        "format": NATIVE_POPULATION_FORMAT,
        "version": NATIVE_POPULATION_VERSION,
        "execution": NATIVE_EXECUTION,
        "population_adapter_bank_sha256": valid_sha256(
            bank.get("identity"), "population adapter bank"
        ),
        "population_adapter_count": bank["count"],
        "population_adapter_rank": bank["rank"],
        "graph_sha256": valid_sha256(training.get("graph_sha256"), "training graph"),
        "port_spec_sha256": valid_sha256(
            training.get("port_spec_sha256"), "training ports"
        ),
        "port_bundle_sha256": valid_sha256(
            training.get("port_bundle_sha256"), "training port bundle"
        ),
    }
    if metadata.get("organism_interface") != organism_identity():
        raise ValueError("controller organism interface differs")
    return identity, metadata


def current_search_contract(
    profile: EmbodiedTrainingProfile,
    controller: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the one current search-v3 configuration shared by campaigns and banks."""
    variants = profile.component("family")["variants"]
    epochs = {int(item["environment_record"]["epoch"]) for item in variants}
    if len(epochs) != 1:
        raise ValueError("campaign environments span archive epochs")
    sources = profile.component("sources")
    developmental_base = valid_sha256(
        sources["biosphere_birth"]["sha256"], "biosphere birth"
    )
    specs, founder = current_parameter_recipe(
        policy_adapter_count=int(controller["population_adapter_count"]),
        heritable_policy_adapter_rows=False,
    )
    variation_receipt = {
        "operator": "bounded-genome-variation-v3",
        "parameters": specs,
        "policy_adapter_selection": "fixed-row-zero-v1",
    }
    probe_panel = {
        "format": "chreatures-population-probe-panel-v1",
        "controller_file_sha256": valid_sha256(
            controller["file_sha256"], "controller file"
        ),
        "action_mode": "sample",
        "fine_tuning": False,
    }
    search_config = {
        "graph_sha256": valid_sha256(controller["graph_sha256"], "training graph"),
        "port_spec_sha256": valid_sha256(
            controller["port_spec_sha256"], "training ports"
        ),
        "base_controller_sha256": controller["file_sha256"],
        "developmental_base_sha256": developmental_base,
        "population_adapter_bank_sha256": valid_sha256(
            controller["population_adapter_bank_sha256"], "population adapter bank"
        ),
        "organism_interface_sha256": value_sha256(organism_identity()),
        "policy_adapter_count": int(controller["population_adapter_count"]),
        "policy_adapter_rank": int(controller["population_adapter_rank"]),
        "parameter_specs": specs,
        "founder_values": founder,
        "descriptor_axes": [
            {"component": "mean_action_thrust", "low": -1.0, "high": 1.0, "bins": 8},
            {"component": "spatial_coverage", "low": 0.0, "high": 1.0, "bins": 8},
            {"component": "elevation_fraction", "low": 0.0, "high": 1.0, "bins": 6},
            {"component": "signal_activity_rate", "low": 0.0, "high": 2.0, "bins": 6},
            {"component": "allocated_mass_rate", "low": 0.0, "high": 0.02, "bins": 6},
        ],
        "quality_terms": [
            {"component": "mean_energy", "scale": 1.0, "weight": 0.5, "direction": 1.0},
            {"component": "energy_delta", "scale": 1.0, "weight": 0.3, "direction": 1.0},
            {"component": "mean_effort", "scale": 1.0, "weight": 0.2, "direction": -1.0},
        ],
        "archive_members_per_cell": 4,
        "variation_recipe_sha256": value_sha256(variation_receipt),
        "environment_probe_panel_sha256": value_sha256(probe_panel),
        "environment_epoch": epochs.pop(),
        "environment_novelty_weight": 0.25,
        "environment_cost_weight": 0.10,
    }
    return search_config, variation_receipt, probe_panel


def collection_worlds(
    profile: EmbodiedTrainingProfile,
    *,
    seed: int,
    worlds: int,
    validation_worlds: int,
    heldout_worlds: int,
) -> list[dict[str, Any]]:
    """Resolve the exact profile selectors used by every collection world slot."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**64
        or not 3 <= worlds <= 16
        or not 1 <= validation_worlds
        or not 1 <= heldout_worlds
        or validation_worlds + heldout_worlds >= worlds
    ):
        raise ValueError("invalid collection world split or seed")
    family = profile.component("family")
    offset = int(profile.component("variation")["heldout_seed_offset"])
    validation_start = worlds - validation_worlds - heldout_worlds
    heldout_start = worlds - heldout_worlds
    result = []
    for slot in range(worlds):
        partition = (
            "training"
            if slot < validation_start
            else "validation"
            if slot < heldout_start
            else "heldout"
        )
        family_split = "training" if partition == "training" else "heldout"
        variants = [item for item in family["variants"] if item["split"] == family_split]
        chosen_seed = seed + slot + (offset if family_split == "heldout" else 0)
        selected = variants[chosen_seed % len(variants)]
        result.append(
            {
                "world_slot": slot,
                "partition": partition,
                "seed": seed + slot,
                "held_out": family_split == "heldout",
                "environment": {"split": family_split, "index": selected["index"]},
                "environment_sha256": selected["environment_sha256"],
            }
        )
    return result


def regulation_diversity(candidates: Sequence[CandidateGenome]) -> dict[str, Any]:
    names = sorted(
        name
        for name in candidates[0].to_value()["values"]
        if name.startswith("metabolic.regulation.")
    )
    if not names:
        raise ValueError("current genome recipe omits inherited metabolic regulation")
    vectors = []
    unique_by_locus = {}
    for name in names:
        values = [float(candidate.to_value()["values"][name]) for candidate in candidates]
        unique_by_locus[name] = len(set(values))
    for candidate in candidates:
        values = candidate.to_value()["values"]
        vectors.append(value_sha256([float(values[name]) for name in names]))
    if len(set(vectors)) != len(candidates) or any(
        count < 2 for count in unique_by_locus.values()
    ):
        raise ValueError("founding bank lacks distinct inherited regulation laws")
    return {
        "loci": names,
        "unique_regulation_vectors": len(set(vectors)),
        "unique_values_by_locus": unique_by_locus,
    }


def validate_founding_bank(
    value: Mapping[str, Any],
    *,
    profile: EmbodiedTrainingProfile,
    controller: Mapping[str, Any],
    seed: int,
    worlds: int,
    validation_worlds: int,
    heldout_worlds: int,
    episodes: int,
    steps: int,
) -> list[CandidateGenome]:
    """Authenticate slot order, controller binding, and inherited diversity."""
    body = copy.deepcopy(dict(value))
    expected = {
        "format",
        "version",
        "sha256",
        "seed",
        "world_count",
        "residents_per_world",
        "candidate_count",
        "episode_count",
        "steps_per_episode",
        "profile_sha256",
        "controller",
        "search_config_sha256",
        "search_state_sha256",
        "episodes",
        "assignments",
        "candidates",
        "regulation_diversity",
        "inheritance",
    }
    family_residents = profile.component("family")["transport"]["residents"]
    if (
        set(body) != expected
        or body.get("format") != FOUNDING_BANK_FORMAT
        or body.get("version") != FOUNDING_BANK_VERSION
        or body.get("sha256") != value_sha256(body | {"sha256": ""})
        or body.get("seed") != seed
        or body.get("world_count") != worlds
        or body.get("residents_per_world") != family_residents
        or body.get("candidate_count") != worlds * family_residents
        or body.get("episode_count") != episodes
        or body.get("steps_per_episode") != steps
        or body.get("profile_sha256") != profile.sha256
        or body.get("controller") != dict(controller)
        or body.get("inheritance")
        != {
            "genome": "immutable public genome-v3 loci",
            "metabolic_regulation": "cold baseline and inherited response laws",
            "private_neural_state": "fresh per episode",
            "adult_memory": "never inherited",
        }
    ):
        raise ValueError("founding bank identity or dimensions differ")
    candidates_raw = body.get("candidates")
    assignments = body.get("assignments")
    if (
        not isinstance(candidates_raw, list)
        or len(candidates_raw) != body["candidate_count"]
        or not isinstance(assignments, list)
        or len(assignments) != body["candidate_count"]
    ):
        raise ValueError("founding bank candidate rows differ")
    candidates = [CandidateGenome(item) for item in candidates_raw]
    hashes = [candidate.sha256 for candidate in candidates]
    if len(set(hashes)) != len(hashes):
        raise ValueError("founding bank genomes are not distinct")
    for index, (assignment, candidate) in enumerate(zip(assignments, candidates, strict=True)):
        if assignment != {
            "candidate_sha256": candidate.sha256,
            "world_slot": index // family_residents,
            "resident_slot": index % family_residents,
            "search_environment_sha256": assignment.get("search_environment_sha256"),
            "search_phase": assignment.get("search_phase"),
        }:
            raise ValueError("founding bank slot assignment differs")
        valid_sha256(assignment["search_environment_sha256"], "search environment")
        if assignment["search_phase"] not in {"direct-transfer", "eligible-fine-tune"}:
            raise ValueError("founding bank search phase differs")
    expected_episodes = [
        {
            "episode": episode,
            "seed": int(body["seed"]) + episode * 1009,
            "worlds": collection_worlds(
                profile,
                seed=int(body["seed"]) + episode * 1009,
                worlds=worlds,
                validation_worlds=validation_worlds,
                heldout_worlds=heldout_worlds,
            ),
        }
        for episode in range(episodes)
    ]
    if body.get("episodes") != expected_episodes:
        raise ValueError("founding bank episode world selectors differ")
    diversity = regulation_diversity(candidates)
    if body.get("regulation_diversity") != diversity:
        raise ValueError("founding bank regulation diversity receipt differs")
    for candidate in candidates:
        raw = candidate.to_value()
        if (
            raw.get("base_controller_sha256") != controller["file_sha256"]
            or raw.get("population_adapter_bank_sha256")
            != controller["population_adapter_bank_sha256"]
            or raw.get("policy_adapter_count") != controller["population_adapter_count"]
            or raw.get("policy_adapter_rank") != controller["population_adapter_rank"]
        ):
            raise ValueError("founding genome controller binding differs")
    valid_sha256(body.get("search_config_sha256"), "search configuration")
    valid_sha256(body.get("search_state_sha256"), "search state")
    return candidates


__all__ = [
    "DESCRIPTOR_RECIPE",
    "FOUNDING_BANK_FORMAT",
    "FOUNDING_BANK_VERSION",
    "QUALITY_RECIPE",
    "SPATIAL_CELL_SCALE",
    "collection_worlds",
    "current_search_contract",
    "file_sha256",
    "regulation_diversity",
    "resident_artifact_identity",
    "valid_sha256",
    "validate_founding_bank",
    "value_sha256",
]
