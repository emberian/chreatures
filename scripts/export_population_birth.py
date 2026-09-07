#!/usr/bin/env python3
"""Export one authenticated CNS-only cold population birth bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.population import CandidateGenome, canonical_bytes, content_sha256
from chreatures.resident_birth import (
    FORMAT as BIRTH_FORMAT,
    GENOME_SCOPE,
    controller_identity,
    validate_manifest,
    verify_controller,
)
from chreatures.resident_contract import (
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
)

FORMAT = "chreatures-cns-only-population-birth-export-v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    if (
        metadata.get("format") != NATIVE_POPULATION_FORMAT
        or metadata.get("version") != NATIVE_POPULATION_VERSION
        or metadata.get("execution") != NATIVE_EXECUTION
    ):
        raise ValueError("resident artifact is not the current CNS-only population")
    return metadata


def selected_world(
    path: Path, index: int
) -> tuple[dict[str, object], list[CandidateGenome]]:
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
    if not candidates:
        raise ValueError("selected assignment has no candidates")
    return world, candidates


def rebase_controller(
    candidate: CandidateGenome, controller_file_sha256: str
) -> tuple[CandidateGenome, dict[str, str]]:
    """Bind an existing physical genome to the current shared CNS controller."""
    value = candidate.to_value()
    source_sha256 = candidate.sha256
    value["base_controller_sha256"] = controller_file_sha256
    value["sha256"] = content_sha256(value)
    rebased = CandidateGenome(value)
    return rebased, {
        "source_candidate_sha256": source_sha256,
        "birth_candidate_sha256": rebased.sha256,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--resident-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    from chreatures.training_environment import (
        EmbodiedTrainingProfile,
        _generated_family_spec,
    )

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    for path in (args.profile, args.assignments, args.resident_artifact):
        if not path.is_file():
            raise FileNotFoundError(path)

    encoded_profile = json.loads(args.profile.read_text())
    profile = EmbodiedTrainingProfile.from_value(
        encoded_profile, locators=encoded_profile["locators"]
    )
    world, source_candidates = selected_world(args.assignments, args.world_index)
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
    if len(source_candidates) != len(spec.get("bodies", [])):
        raise ValueError("selected candidate count differs from generated bodies")

    resident_metadata = artifact_metadata(args.resident_artifact)
    controller = controller_identity(args.resident_artifact)
    developmental_sha256 = profile.component("sources")["biosphere_birth"]["sha256"]
    candidates: list[CandidateGenome] = []
    candidate_receipts: list[dict[str, str]] = []
    for candidate in source_candidates:
        if candidate.to_value()["developmental_base_sha256"] != developmental_sha256:
            raise ValueError("candidate developmental base identity differs")
        rebased, receipt = rebase_controller(candidate, controller["file_sha256"])
        candidates.append(rebased)
        candidate_receipts.append(receipt)

    birth = validate_manifest(
        {
            "format": BIRTH_FORMAT,
            "controller": controller,
            "residents": [
                {
                    "candidate": candidate.to_value(),
                    "cns_adapter_sha256": controller["cns_adapter_sha256"],
                }
                for candidate in candidates
            ],
        }
    )
    verify_controller(birth, args.resident_artifact)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent)
    )
    try:
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
                "file_sha256": controller["file_sha256"],
                "artifact_sha256": resident_metadata["artifact_sha256"],
                "cns_service": copy.deepcopy(resident_metadata["cns_service"]),
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
            "genome_scope": GENOME_SCOPE,
            "candidate_rebase": candidate_receipts,
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
                "genome_scope": GENOME_SCOPE,
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
