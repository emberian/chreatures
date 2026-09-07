#!/usr/bin/env python3
"""Materialize a pinned rich-v4 profile and search-v3 founding genome bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.neural_genotype import NeuralVariantRecipe
from chreatures.neural_ports import NeuralPortBundle
from chreatures.population import PopulationSearch
from chreatures.population_launch import (
    FOUNDING_BANK_FORMAT,
    FOUNDING_BANK_VERSION,
    collection_worlds,
    current_search_contract,
    file_sha256,
    regulation_diversity,
    resident_artifact_identity,
    validate_founding_bank,
    value_sha256,
)
from chreatures.training_cohort import load_training_graph
from chreatures.training_environment import EmbodiedTrainingProfile

FORMAT = "chreatures-rich-collection-launch-v1"
VERSION = 1


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
            + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_identity() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if dirty:
        raise RuntimeError("launch preparation requires a clean frozen source checkout")
    paths = (
        Path("scripts/prepare_rich_sensorimotor_collection.py"),
        Path("scripts/collect_rich_sensorimotor.py"),
        Path("chreatures/population.py"),
        Path("chreatures/population_launch.py"),
        Path("chreatures/training_environment.py"),
        Path("native/population-core/Cargo.toml"),
        Path("native/population-core/Cargo.lock"),
        Path("native/population-core/src/lib.rs"),
        Path("native/population-core/src/main.rs"),
    )
    return {
        "revision": revision,
        "dirty": False,
        "files": {
            str(path): {"bytes": (ROOT / path).stat().st_size, "sha256": file_sha256(ROOT / path)}
            for path in paths
        },
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resident-artifact", type=Path, required=True)
    parser.add_argument("--neural-recipe", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--chemical-habitat", type=Path, required=True)
    parser.add_argument("--chemical-biosphere", type=Path, required=True)
    parser.add_argument("--regional-family-config", type=Path, required=True)
    parser.add_argument("--regional-family-schedule", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=10)
    parser.add_argument("--validation-worlds", type=int, default=1)
    parser.add_argument("--heldout-worlds", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    destination = args.output.resolve()
    if destination.exists():
        raise SystemExit("launch output must be absent")
    if not args.graph.is_dir():
        raise SystemExit(f"graph directory does not exist: {args.graph}")
    for path in (
        args.resident_artifact,
        args.neural_recipe,
        args.port_bundle,
        args.chemical_habitat,
        args.chemical_biosphere,
        args.regional_family_config,
        args.regional_family_schedule,
    ):
        if not path.is_file():
            raise SystemExit(f"required artifact does not exist: {path}")
    if not 3 <= args.worlds <= 16 or (
        args.validation_worlds < 1
        or args.heldout_worlds < 1
        or args.validation_worlds + args.heldout_worlds >= args.worlds
    ):
        raise SystemExit("world split requires train, validation, and final holdout worlds")
    if isinstance(args.seed, bool) or not 0 <= args.seed < 2**64:
        raise SystemExit("seed must be an unsigned 64-bit integer")
    if not 1 <= args.episodes <= 64 or not 512 <= args.steps <= 100_000:
        raise SystemExit("episodes or steps outside bounded collection range")
    if args.steps % 512:
        raise SystemExit("steps must be divisible by the fixed 512-tick shard size")

    source = source_identity()
    controller, _metadata = resident_artifact_identity(args.resident_artifact)
    graph = load_training_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    NeuralVariantRecipe.load(args.neural_recipe)
    if (
        str(graph.hash) != controller["graph_sha256"]
        or ports.spec_hash != controller["port_spec_sha256"]
        or file_sha256(args.port_bundle) != controller["port_bundle_sha256"]
    ):
        raise ValueError("controller neural substrate differs from launch inputs")

    profile = EmbodiedTrainingProfile.nursery_family(
        args.chemical_habitat,
        args.chemical_biosphere,
        args.regional_family_config,
        args.regional_family_schedule,
    )
    # Verify every source byte through the same envelope the collector will load.
    encoded_profile = profile.to_value()
    profile = EmbodiedTrainingProfile.from_value(
        encoded_profile, locators=encoded_profile["locators"]
    )
    residents = int(profile.component("family")["transport"]["residents"])
    count = args.worlds * residents
    if count > 4096:
        raise ValueError("founding bank exceeds native search capacity")
    search_config, variation_receipt, probe_panel = current_search_contract(
        profile, controller
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.prepare-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    try:
        profile_path = stage / "profile.json"
        atomic_json(profile_path, profile.to_value())
        search = PopulationSearch.initialize(
            stage / "search.json", search_config, seed=args.seed
        )
        for variant in profile.component("family")["variants"]:
            search.register_environment(variant["environment_record"])
        native_assignments = search.ask(count)
        search.validate()
        candidates = [row["candidate"] for row in native_assignments]
        if len({candidate.sha256 for candidate in candidates}) != count:
            raise RuntimeError("native search did not create distinct founding genomes")
        episodes = [
            {
                "episode": episode,
                "seed": args.seed + episode * 1009,
                "worlds": collection_worlds(
                    profile,
                    seed=args.seed + episode * 1009,
                    worlds=args.worlds,
                    validation_worlds=args.validation_worlds,
                    heldout_worlds=args.heldout_worlds,
                ),
            }
            for episode in range(args.episodes)
        ]
        assignments = [
            {
                "candidate_sha256": row["candidate"].sha256,
                "world_slot": index // residents,
                "resident_slot": index % residents,
                "search_environment_sha256": row["environment_sha256"],
                "search_phase": row["phase"],
            }
            for index, row in enumerate(native_assignments)
        ]
        bank = {
            "format": FOUNDING_BANK_FORMAT,
            "version": FOUNDING_BANK_VERSION,
            "sha256": "",
            "seed": args.seed,
            "world_count": args.worlds,
            "residents_per_world": residents,
            "candidate_count": count,
            "episode_count": args.episodes,
            "steps_per_episode": args.steps,
            "profile_sha256": profile.sha256,
            "controller": controller,
            "search_config_sha256": value_sha256(search_config),
            "search_state_sha256": file_sha256(stage / "search.json"),
            "episodes": episodes,
            "assignments": assignments,
            "candidates": [candidate.to_value() for candidate in candidates],
            "regulation_diversity": regulation_diversity(candidates),
            "inheritance": {
                "genome": "immutable public genome-v3 loci",
                "metabolic_regulation": "cold baseline and inherited response laws",
                "private_neural_state": "fresh per episode",
                "adult_memory": "never inherited",
            },
        }
        bank["sha256"] = value_sha256(bank)
        validate_founding_bank(
            bank,
            profile=profile,
            controller=controller,
            seed=args.seed,
            worlds=args.worlds,
            validation_worlds=args.validation_worlds,
            heldout_worlds=args.heldout_worlds,
            episodes=args.episodes,
            steps=args.steps,
        )
        bank_path = stage / "founding-bank.json"
        atomic_json(bank_path, bank)
        inputs = {
            "resident_artifact": {
                "path": str(args.resident_artifact.resolve()),
                **controller,
            },
            "neural_recipe": {
                "path": str(args.neural_recipe.resolve()),
                "sha256": file_sha256(args.neural_recipe),
            },
            "graph": {
                "path": str(args.graph.resolve()),
                "sha256": str(graph.hash),
            },
            "port_bundle": {
                "path": str(args.port_bundle.resolve()),
                "sha256": file_sha256(args.port_bundle),
                "spec_sha256": ports.spec_hash,
            },
        }
        receipt = {
            "format": FORMAT,
            "version": VERSION,
            "sha256": "",
            "status": "prepared; collection not started",
            "source": source,
            "inputs": inputs,
            "profile": {
                "path": "profile.json",
                "semantic_sha256": profile.sha256,
                "file_sha256": file_sha256(profile_path),
            },
            "founding_bank": {
                "path": "founding-bank.json",
                "semantic_sha256": bank["sha256"],
                "file_sha256": file_sha256(bank_path),
                "candidate_count": count,
                "distinct_candidate_count": len({candidate.sha256 for candidate in candidates}),
                "distinct_regulation_count": bank["regulation_diversity"][
                    "unique_regulation_vectors"
                ],
            },
            "search": {
                "path": "search.json",
                "format": "chreatures-population-search-v3",
                "file_sha256": file_sha256(stage / "search.json"),
                "config_sha256": value_sha256(search_config),
                "variation_receipt": variation_receipt,
                "probe_panel": probe_panel,
                "pending_semantics": "generation provenance only; no physical evaluation was started",
            },
            "collection": {
                "worlds": args.worlds,
                "residents_per_world": residents,
                "episodes": args.episodes,
                "steps_per_episode": args.steps,
                "expected_transitions": count * args.episodes * args.steps,
                "partitions": {
                    "training": args.worlds - args.validation_worlds - args.heldout_worlds,
                    "validation": args.validation_worlds,
                    "heldout": args.heldout_worlds,
                },
            },
        }
        receipt["sha256"] = value_sha256(receipt)
        atomic_json(stage / "receipt.json", receipt)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(destination),
                "profile_sha256": profile.sha256,
                "founding_bank_sha256": bank["sha256"],
                "candidate_count": count,
                "transitions": count * args.episodes * args.steps,
                "status": "prepared; collection not started",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
