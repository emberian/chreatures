#!/usr/bin/env python3
"""Build an authenticated matched-cohort plan for current Torch training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.population import (
    CandidateGenome,
    canonical_bytes,
    content_sha256,
    current_parameter_recipe,
)
from chreatures.resident_contract import (
    BOOTSTRAP_FORMAT,
    TORCH_POPULATION_PLAN_FORMAT,
    TORCH_POPULATION_PLAN_VERSION,
)
from chreatures.training_environment import EmbodiedTrainingProfile

FORMAT = TORCH_POPULATION_PLAN_FORMAT
VERSION = TORCH_POPULATION_PLAN_VERSION
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def hash_without(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def valid_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def validate_loci(candidate: CandidateGenome, specs: list[dict[str, Any]]) -> None:
    values = candidate.to_value()["values"]
    if set(values) != {item["name"] for item in specs}:
        raise ValueError("source candidate loci differ from the current recipe")
    for spec in specs:
        value = float(values[spec["name"]])
        if not spec["low"] <= value <= spec["high"]:
            raise ValueError(f"source locus outside current bound: {spec['name']}")
        if spec["integer"] and value != int(value):
            raise ValueError(f"source integer locus is fractional: {spec['name']}")
    groups = {item["group"] for item in specs if item["group"].startswith("simplex:")}
    for group in groups:
        total = sum(
            float(values[item["name"]]) for item in specs if item["group"] == group
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"source simplex does not sum to one: {group}")


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--bootstrap-identity", type=Path, required=True)
    parser.add_argument("--port-bundle", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-freeze", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.output.exists() or args.receipt.exists():
        raise SystemExit("output and receipt must not already exist")
    profile_envelope = load(args.profile)
    profile = EmbodiedTrainingProfile.from_value(profile_envelope)
    identity = load(args.bootstrap_identity)
    source = load(args.source_plan)
    if identity.get("content_sha256") != hash_without(identity, "content_sha256"):
        raise ValueError("bootstrap identity content hash differs")
    controller = identity.get("controller_file", {})
    adapters = identity.get("population_adapters", {})
    cold = identity.get("cold_inheritance", {})
    if (
        identity.get("controller_format") != BOOTSTRAP_FORMAT
        or file_sha256(args.bootstrap) != controller.get("sha256")
        or not isinstance(adapters.get("count"), int)
        or adapters["count"] < 1
        or not isinstance(adapters.get("rank"), int)
        or adapters["rank"] < 1
        or adapters.get("candidate_assignment") != "resident_index_mod_candidate_count"
        or adapters.get("heritable_policy_adapter_rows") is not True
    ):
        raise ValueError("bootstrap population contract differs")
    if source.get("content_sha256") != hash_without(source, "content_sha256"):
        raise ValueError("source matched-cohort plan identity differs")
    worlds = source.get("worlds")
    residents = source.get("residents_per_world")
    adapter_count = adapters["count"]
    adapter_rank = adapters["rank"]
    if (
        not isinstance(worlds, int)
        or worlds < 1
        or not isinstance(residents, int)
        or residents < 1
        or worlds * residents % adapter_count
    ):
        raise ValueError("source cohort dimensions cannot balance adapter rows")
    raw_candidates = source.get("candidates")
    if (
        not isinstance(raw_candidates, list)
        or len(raw_candidates) != worlds * residents
    ):
        raise ValueError("source candidates differ from its cohort dimensions")
    specs, _ = current_parameter_recipe(
        policy_adapter_count=adapter_count, heritable_policy_adapter_rows=True
    )
    source_candidates = [CandidateGenome(item) for item in raw_candidates]
    for candidate in source_candidates:
        validate_loci(candidate, specs)
    source_training = source.get("training")
    if not isinstance(source_training, dict):
        raise ValueError("source plan training schedule differs")
    schedule = {
        name: source_training.get(name)
        for name in (
            "updates",
            "physical_steps",
            "rollout_steps",
            "episode_steps",
            "ppo_epochs",
        )
    }
    if (
        any(not isinstance(value, int) or value < 1 for value in schedule.values())
        or schedule["updates"]
        != schedule["physical_steps"] // schedule["rollout_steps"]
        or schedule["physical_steps"] % schedule["rollout_steps"]
        or schedule["episode_steps"] % schedule["rollout_steps"]
    ):
        raise ValueError("source plan training schedule differs")
    source_record = cold.get("source", {})
    recipe = {
        "format": "chreatures-torch-population-candidate-variation-v2",
        "operation": "matched-loci-controller-profile-rebase-v1",
        "source_plan_file_sha256": file_sha256(args.source_plan),
        "source_plan_content_sha256": source["content_sha256"],
        "base_controller_sha256": controller["sha256"],
        "controller_identity_content_sha256": identity["content_sha256"],
        "profile_sha256": profile.sha256,
        "population_adapter_bank_sha256": valid_sha(
            adapters["identity"], "adapter bank"
        ),
        "policy_adapter_count": adapter_count,
        "policy_adapter_rank": adapter_rank,
        "policy_adapter_assignment": "resident_index_mod_candidate_count",
        "locus_rule": "copy every source candidate values entry exactly; no mutation or redraw",
        "candidate_parentage": "parents empty; source candidate is matched-loci provenance, not a genome parent",
        "cohort_semantics": "engineered matched research cohort; not native archive offspring or the same individuals",
        "cold_history": {
            "source_checkpoint_sha256": valid_sha(
                source_record["file_sha256"], "cold checkpoint"
            ),
            "source_identity_sha256": valid_sha(
                source_record["identity_sha256"], "cold identity"
            ),
            "physiology_adapter": "zero initialization",
            "new_actuator_active_logit": cold["new_axis_active_logit"],
            "new_axis_active_probability": cold["new_axis_active_probability"],
            "new_axis_positive_magnitudes": cold["new_axis_positive_magnitudes"],
            "shared_trainable_organs": cold["shared_trainable_organs"],
            "optimizer": "fresh",
            "private_state": "fresh",
        },
    }
    recipe_sha = hashlib.sha256(canonical_bytes(recipe)).hexdigest()
    candidates = []
    mapping = []
    for index, source_candidate in enumerate(source_candidates):
        old = source_candidate.to_value()
        candidate = copy.deepcopy(old)
        candidate.update(
            {
                "sha256": "",
                "parents": [],
                "base_controller_sha256": controller["sha256"],
                "population_adapter_bank_sha256": adapters["identity"],
                "policy_adapter_count": adapter_count,
                "policy_adapter_rank": adapter_rank,
            }
        )
        candidate["variation"]["recipe_sha256"] = recipe_sha
        row = index % adapter_count
        if int(candidate["values"]["controller.policy_adapter_index"]) != row:
            raise ValueError("source adapter rows are not balanced in resident order")
        candidate["sha256"] = content_sha256(candidate)
        CandidateGenome(candidate)
        candidates.append(candidate)
        mapping.append(
            {
                "candidate_sha256": candidate["sha256"],
                "matched_source_candidate_sha256": old["sha256"],
                "resident_index": index,
                "policy_adapter_index": row,
                "copied_loci": sorted(candidate["values"]),
                "changed_genome_fields": [
                    "base_controller_sha256",
                    "population_adapter_bank_sha256",
                    "variation.recipe_sha256",
                    "sha256",
                ],
                "parents": [],
            }
        )
    plan = {
        "format": FORMAT,
        "version": VERSION,
        "profile_sha256": profile.sha256,
        "profile_identity_scope": "semantic value only; host locators are excluded",
        "controller_file_sha256": controller["sha256"],
        "controller_identity_file_sha256": file_sha256(args.bootstrap_identity),
        "controller_identity_content_sha256": identity["content_sha256"],
        "graph_sha256": source["graph_sha256"],
        "port_spec_sha256": source["port_spec_sha256"],
        "port_bundle_sha256": file_sha256(args.port_bundle),
        "developmental_base_sha256": source["developmental_base_sha256"],
        "cold_source_checkpoint_sha256": source_record["file_sha256"],
        "cold_source_identity_sha256": source_record["identity_sha256"],
        "population_adapter_bank_sha256": adapters["identity"],
        "policy_adapter_count": adapter_count,
        "policy_adapter_rank": adapter_rank,
        "variation_recipe": recipe,
        "variation_recipe_sha256": recipe_sha,
        "candidate_count": len(candidates),
        "worlds": worlds,
        "residents_per_world": residents,
        "candidate_assignment": "flat resident index modulo policy adapter count; balanced exactly",
        "mapping": mapping,
        "candidates": candidates,
        "training": {
            "source_freeze": args.source_freeze,
            **schedule,
            "new_axis_exploration_probability": cold[
                "new_axis_active_probability"
            ],
            "status": "planned; no behavior claim",
        },
        "lineage_boundary": "separate Torch training lineage; no native campaign ask, pending assignment, archive selection, or individual continuation",
    }
    plan["content_sha256"] = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    atomic(args.output, plan)
    receipt = {
        "format": "chreatures-torch-population-training-plan-receipt-v1",
        "plan": {
            "path": str(args.output.resolve()),
            "file_sha256": file_sha256(args.output),
            "content_sha256": plan["content_sha256"],
        },
        "recipe_sha256": recipe_sha,
        "source_plan": {
            "path": str(args.source_plan.resolve()),
            "file_sha256": file_sha256(args.source_plan),
            "content_sha256": source["content_sha256"],
        },
        "profile_sha256": profile.sha256,
        "profile_source": {
            "path": str(args.profile.resolve()),
            "file_sha256": file_sha256(args.profile),
        },
        "controller_file_sha256": controller["sha256"],
        "candidate_sha256": [item["sha256"] for item in candidates],
        "adapter_row_counts": {
            str(row): len(candidates) // adapter_count for row in range(adapter_count)
        },
        "status": "constructed; training outcome unknown",
    }
    receipt["content_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    atomic(args.receipt, receipt)
    print(
        json.dumps(
            {
                "plan": receipt["plan"],
                "receipt": str(args.receipt.resolve()),
                "receipt_content_sha256": receipt["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
