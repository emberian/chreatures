#!/usr/bin/env python3
"""Export one authenticated cold population birth without starting a world."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.neural_genotype import NeuralVariantRecipe, compile_population_phenotypes
from chreatures.neural_ports import NeuralPortBundle
from chreatures.population import CandidateGenome, canonical_bytes, content_sha256
from chreatures.resident_birth import FORMAT as BIRTH_FORMAT, validate_manifest, verify_controller
from chreatures.training_cohort import load_training_graph
from chreatures.resident_contract import (
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
)

FORMAT = "chreatures-population-birth-export-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def artifact_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata.get("format") != NATIVE_POPULATION_FORMAT:
        raise ValueError("resident artifact is not the current population format")
    if metadata.get("version") != NATIVE_POPULATION_VERSION:
        raise ValueError("resident artifact version differs")
    if metadata.get("execution") != NATIVE_EXECUTION:
        raise ValueError("resident artifact execution differs")
    return metadata


def selected_world(path: Path, index: int) -> tuple[dict[str, object], list[CandidateGenome]]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("worlds"), list):
        raise ValueError("assignment must contain worlds")
    declared = document.get("sha256")
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise ValueError("assignment lacks its native canonical identity")
    assignment_body = dict(document)
    assignment_body.pop("sha256")
    if hashlib.sha256(canonical_bytes(assignment_body)).hexdigest() != declared:
        raise ValueError("assignment content differs from its declared identity")
    if not 0 <= index < len(document["worlds"]):
        raise ValueError("world index is outside assignment")
    world = document["worlds"][index]
    if not isinstance(world, dict) or not isinstance(world.get("candidates"), list):
        raise ValueError("selected assignment world differs")
    candidates = [CandidateGenome(value) for value in world["candidates"]]
    if not candidates or any(
        content_sha256(item.to_value()) != item.sha256 for item in candidates
    ):
        raise ValueError("selected assignment candidates differ")
    return world, candidates


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--resident-artifact", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--neural-recipe", type=Path, required=True)
    parser.add_argument("--service-phenotype-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    from chreatures.training_environment import EmbodiedTrainingProfile, _generated_family_spec
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    inputs = (
        args.profile,
        args.assignments,
        args.resident_artifact,
        args.port_bundle,
        args.neural_recipe,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.graph.is_dir():
        raise NotADirectoryError(args.graph)

    profile = EmbodiedTrainingProfile.from_value(json.loads(args.profile.read_text()))
    world, candidates = selected_world(args.assignments, args.world_index)
    environment = world.get("environment")
    if not isinstance(environment, dict) or environment.get("split") not in {
        "training",
        "heldout",
    }:
        raise ValueError("selected environment differs")
    held_out = environment["split"] == "heldout"
    variation = profile.component("variation")
    chosen_seed = int(world["seed"]) + (
        int(variation["heldout_seed_offset"]) if held_out else 0
    )
    spec, biosphere, environment_receipt = _generated_family_spec(
        profile, chosen_seed, held_out, environment
    )

    resident_metadata = artifact_metadata(args.resident_artifact)
    controller_sha = file_sha256(args.resident_artifact)
    developmental_sha = profile.component("sources")["biosphere_birth"]["sha256"]
    bank = resident_metadata.get("population_adapters", {})
    for candidate in candidates:
        value = candidate.to_value()
        if (
            value["base_controller_sha256"] != controller_sha
            or value["developmental_base_sha256"] != developmental_sha
        ):
            raise ValueError("candidate base artifact identity differs")
        candidate_bank = (
            value["population_adapter_bank_sha256"],
            value["policy_adapter_count"],
            value["policy_adapter_rank"],
        )
        artifact_bank = (bank.get("identity"), bank.get("count"), bank.get("rank"))
        if candidate_bank != artifact_bank:
            raise ValueError("candidate policy adapter bank differs")

    graph = load_training_graph(args.graph)
    ports = NeuralPortBundle.load(args.port_bundle, graph)
    port_sha = file_sha256(args.port_bundle)
    recipe = NeuralVariantRecipe.load(args.neural_recipe)
    phenotypes = compile_population_phenotypes(
        candidates, recipe, graph, ports, port_sha, controller_sha
    )
    compatibility = {item.compatibility_group for item in phenotypes}
    if len(compatibility) != 1:
        raise ValueError("selected founders do not share a neural compatibility group")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    try:
        neural = staging / "neural"
        neural.mkdir()
        birth_rows, phenotype_receipts = [], []
        for index, (candidate, phenotype) in enumerate(zip(candidates, phenotypes, strict=True)):
            name = f"resident-{index:02d}-{phenotype.sha256}.npz"
            local_path = neural / name
            phenotype.save(local_path)
            artifact_sha = file_sha256(local_path)
            service_path = str(args.service_phenotype_root / name)
            identity = {
                "artifact_path": service_path, "artifact_sha256": artifact_sha,
                "phenotype_sha256": phenotype.sha256, "graph_sha256": str(graph.hash),
                "port_spec_sha256": ports.spec_hash, "port_bundle_sha256": port_sha,
            }
            birth_rows.append({"candidate": candidate.to_value(), "neural_phenotype": identity})
            phenotype_receipts.append(
                {
                    "index": index,
                    "candidate_sha256": candidate.sha256,
                    **identity,
                    "local_path": f"neural/{name}",
                }
            )
        birth = validate_manifest({"format": BIRTH_FORMAT, "residents": birth_rows})
        verify_controller(birth, args.resident_artifact)
        atomic_json(staging / "habitat.json", spec)
        atomic_json(staging / "biosphere.json", biosphere)
        atomic_json(staging / "resident-birth.json", birth)
        source = {
            "exporter": {
                "path": str(Path(__file__).resolve()),
                "file_sha256": file_sha256(Path(__file__).resolve()),
            },
            "profile": {
                "path": str(args.profile.resolve()),
                "file_sha256": file_sha256(args.profile),
                "sha256": profile.sha256,
            },
            "assignments": {
                "path": str(args.assignments.resolve()),
                "file_sha256": file_sha256(args.assignments),
                "world_index": args.world_index,
                "content_sha256": document_sha(args.assignments),
            },
            "resident_artifact": {
                "path": str(args.resident_artifact.resolve()),
                "file_sha256": controller_sha,
                "artifact_sha256": resident_metadata["artifact_sha256"],
            },
            "graph": {"path": str(args.graph.resolve()), "sha256": str(graph.hash)},
            "port_bundle": {
                "path": str(args.port_bundle.resolve()),
                "file_sha256": port_sha,
                "spec_sha256": ports.spec_hash,
            },
            "neural_recipe": {
                "path": str(args.neural_recipe.resolve()),
                "file_sha256": file_sha256(args.neural_recipe),
                "sha256": recipe.sha256,
            },
        }
        from chreatures.native_world import load_world_kernels

        native_path = Path(load_world_kernels().__file__).resolve()
        source["native_world"] = {
            "path": str(native_path),
            "file_sha256": file_sha256(native_path),
        }
        receipt = {
            "format": FORMAT,
            "source": source,
            "world": {
                "assignment_world_id": world.get("world_id"),
                "seed": world["seed"],
                "environment": environment,
                "environment_receipt": environment_receipt,
            },
            "service_phenotype_root": str(args.service_phenotype_root),
            "compatibility_group": compatibility.pop(),
            "phenotypes": phenotype_receipts,
            "outputs": {
                name: file_sha256(staging / name)
                for name in ("habitat.json", "biosphere.json", "resident-birth.json")
            },
        }
        receipt["sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        atomic_json(staging / "receipt.json", receipt)
        os.replace(staging, args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "receipt_sha256": receipt["sha256"],
                "residents": len(candidates),
            },
            sort_keys=True,
        )
    )
    return 0


def document_sha(path: Path) -> str | None:
    value = json.loads(path.read_text())
    return value.get("sha256") if isinstance(value, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
