import copy
import json
from pathlib import Path

import pytest

from chreatures.population import (
    CandidateGenome,
    compose_population_birth,
    content_sha256,
    current_parameter_recipe,
)
from chreatures.population_launch import (
    collection_worlds,
    regulation_diversity,
    validate_founding_bank,
    value_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SHA = "0" * 64


def candidate(index: int) -> CandidateGenome:
    specs, values = current_parameter_recipe(policy_adapter_count=1)
    values = copy.deepcopy(values)
    for spec in specs:
        if spec["name"].startswith("metabolic.regulation."):
            fraction = (index + 1) / 7.0
            values[spec["name"]] = spec["low"] + fraction * (
                spec["high"] - spec["low"]
            )
    value = {
        "format": "chreatures-population-genome-v3",
        "sha256": "",
        "parents": [],
        "graph_sha256": SHA,
        "port_spec_sha256": SHA,
        "base_controller_sha256": SHA,
        "developmental_base_sha256": SHA,
        "population_adapter_bank_sha256": SHA,
        "organism_interface_sha256": SHA,
        "policy_adapter_count": 1,
        "policy_adapter_rank": 1,
        "values": values,
        "variation": {
            "operator": "bounded-genome-variation-v3",
            "seed": index,
            "recipe_sha256": SHA,
            "mutated": [
                spec["name"]
                for spec in specs
                if spec["name"].startswith("metabolic.regulation.")
            ],
        },
    }
    value["sha256"] = content_sha256(value)
    return CandidateGenome(value)


def test_cold_birth_preserves_distinct_inherited_regulation_without_memory() -> None:
    habitat = json.loads((ROOT / "data/habitats/living-reef.json").read_text())
    biosphere = json.loads((ROOT / "data/biosphere/living-reef.json").read_text())
    resident_bundle = json.loads(
        (ROOT / "data/habitat-families/regional-residents-v3.json").read_text()
    )["residents"][:6]
    habitat["bodies"] = [copy.deepcopy(row["body"]) for row in resident_bundle]
    biosphere["compartments"] = []
    biosphere["mobiles"] = []
    for row in resident_bundle:
        body_row = len(biosphere["compartments"])
        biosphere["compartments"].append(copy.deepcopy(row["founders"]["body"]))
        gut_row = len(biosphere["compartments"])
        biosphere["compartments"].append(copy.deepcopy(row["founders"]["gut"]))
        biosphere["mobiles"].append(
            {
                "id": row["body"]["id"],
                "body_row": body_row,
                "gut_row": gut_row,
                **copy.deepcopy(row["mobile"]),
            }
        )
    candidates = [candidate(index) for index in range(len(resident_bundle))]

    _habitat, born, receipt = compose_population_birth(habitat, biosphere, candidates)

    diversity = regulation_diversity(candidates)
    assert diversity["unique_regulation_vectors"] == len(candidates)
    compartments = born["compartments"]
    compiled = []
    for mobile, candidate_value in zip(born["mobiles"], candidates, strict=True):
        loci = candidate_value.loci("metabolic")
        for row_name, compartment_name in (("body_row", "body"), ("gut_row", "gut")):
            row = compartments[mobile[row_name]]
            assert row["regulation"]["baseline"] == row["enzymes"]
            assert row["regulation"]["time_constant_seconds"] == pytest.approx(
                loci[f"regulation.time_constant_seconds.{compartment_name}"]
            )
            assert row["regulation"]["change_cost_atp_per_expression"] == pytest.approx(
                loci[f"regulation.expression_change_cost.{compartment_name}"]
            )
            compiled.append(json.dumps(row["regulation"], sort_keys=True))
    assert len(set(compiled[::2])) == len(candidates)
    assert len(set(compiled[1::2])) == len(candidates)
    assert receipt["semantics"] == (
        "fresh private rows and runtime state; immutable inherited loci only"
    )

    private = candidates[0].to_value()
    private["values"]["adult.memory"] = 1.0
    private["sha256"] = content_sha256(private)
    with pytest.raises(ValueError, match="private lifetime field"):
        CandidateGenome(private)


class ProfileStub:
    sha256 = "1" * 64

    def component(self, name: str):
        if name == "variation":
            return {"heldout_seed_offset": 80_000_003}
        if name == "family":
            return {
                "transport": {"residents": 2},
                "variants": [
                    {
                        "split": split,
                        "index": index,
                        "environment_sha256": str(digit) * 64,
                    }
                    for split, start in (("training", 2), ("heldout", 5))
                    for index, digit in enumerate(range(start, start + 3))
                ],
            }
        raise KeyError(name)


def test_founding_bank_pins_episode_worlds_and_controller() -> None:
    profile = ProfileStub()
    candidates = [candidate(index) for index in range(6)]
    controller = {
        "file_sha256": SHA,
        "artifact_sha256": "a" * 64,
        "format": "current-test",
        "version": 1,
        "execution": "current-test",
        "population_adapter_bank_sha256": SHA,
        "population_adapter_count": 1,
        "population_adapter_rank": 1,
        "graph_sha256": SHA,
        "port_spec_sha256": SHA,
        "port_bundle_sha256": SHA,
    }
    episodes = [
        {
            "episode": episode,
            "seed": 91 + episode * 1009,
            "worlds": collection_worlds(
                profile,
                seed=91 + episode * 1009,
                worlds=3,
                validation_worlds=1,
                heldout_worlds=1,
            ),
        }
        for episode in range(2)
    ]
    bank = {
        "format": "chreatures-rich-collection-founding-bank-v1",
        "version": 1,
        "sha256": "",
        "seed": 91,
        "world_count": 3,
        "residents_per_world": 2,
        "candidate_count": 6,
        "episode_count": 2,
        "steps_per_episode": 512,
        "profile_sha256": profile.sha256,
        "controller": controller,
        "search_config_sha256": SHA,
        "search_state_sha256": SHA,
        "episodes": episodes,
        "assignments": [
            {
                "candidate_sha256": item.sha256,
                "world_slot": index // 2,
                "resident_slot": index % 2,
                "search_environment_sha256": "2" * 64,
                "search_phase": "direct-transfer",
            }
            for index, item in enumerate(candidates)
        ],
        "candidates": [item.to_value() for item in candidates],
        "regulation_diversity": regulation_diversity(candidates),
        "inheritance": {
            "genome": "immutable public genome-v3 loci",
            "metabolic_regulation": "cold baseline and inherited response laws",
            "private_neural_state": "fresh per episode",
            "adult_memory": "never inherited",
        },
    }
    bank["sha256"] = value_sha256(bank)

    restored = validate_founding_bank(
        bank,
        profile=profile,
        controller=controller,
        seed=91,
        worlds=3,
        validation_worlds=1,
        heldout_worlds=1,
        episodes=2,
        steps=512,
    )

    assert [item.sha256 for item in restored] == [item.sha256 for item in candidates]
    assert [row["partition"] for row in bank["episodes"][0]["worlds"]] == [
        "training",
        "validation",
        "heldout",
    ]
