#!/usr/bin/env python3
"""Convert one authenticated v3 development checkpoint into a v4 Torch birth."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.organism_interface import identity as organism_identity
from research.sensorimotor_skills.rich_data import RichNormalizer, canonical_sha256
from research.sensorimotor_skills.rich_model import (
    PopulationAdapterBank,
    cold_inherit_v3_model,
)
from research.sensorimotor_skills.rich_online import cold_inherit_v3_manager

SOURCE_FORMAT = "chreatures-rich-online-sensorimotor-development-v1"
FORMAT = "chreatures-rich-sensorimotor-bootstrap-v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-source", action="store_true")
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--variation-seed", type=int, required=True)
    parser.add_argument("--variation-scale", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if not args.trusted_source:
        raise SystemExit("Torch deserialization requires --trusted-source")
    source_path = args.source.expanduser().resolve()
    actual_sha256 = sha256(source_path)
    if actual_sha256 != args.source_sha256:
        raise ValueError("source checkpoint SHA-256 differs")
    if args.output.exists():
        raise SystemExit("output already exists")
    if not 1 <= args.candidate_count <= 4096 or not 1 <= args.adapter_rank <= 32:
        raise SystemExit("candidate adapter dimensions are outside their bound")
    if not 0 <= args.variation_scale <= 0.1:
        raise SystemExit("variation scale must be in [0,0.1]")

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("format") != SOURCE_FORMAT:
        raise ValueError("source development checkpoint format differs")
    source_identity = copy.deepcopy(source["identity"])
    model = cold_inherit_v3_model(source["model"])
    manager = cold_inherit_v3_manager(source["goal_manager"])
    normalizer = RichNormalizer.cold_inherit_v3(source_identity["normalizer"])
    adapters = PopulationAdapterBank(args.candidate_count, args.adapter_rank)
    if args.candidate_count > 1:
        adapters.vary(
            torch.arange(1, args.candidate_count),
            seed=args.variation_seed,
            scale=args.variation_scale,
        )
    conversion = {
        "format": "chreatures-v3-to-v4-torch-cold-birth-v1",
        "source": {
            "path": str(source_path),
            "file_sha256": actual_sha256,
            "format": SOURCE_FORMAT,
            "updates": int(source.get("updates", 0)),
            "physical_steps": int(source.get("physical_steps", 0)),
            "identity_sha256": canonical_sha256(source_identity),
        },
        "organism_interface": organism_identity(),
        "old_to_new_action_columns": [0, 1, 2, 4, 5, 6, 7, 3, 8],
        "new_physiology_normalization": "mean-zero-scale-one",
        "new_rectified_head_bias": -8.0,
        "candidate_adapters": {
            "count": args.candidate_count,
            "rank": args.adapter_rank,
            "variation_seed": args.variation_seed,
            "variation_scale": args.variation_scale,
            "candidate_zero": "exact inherited base",
        },
        "optimizer": "fresh at online birth",
        "critic": "fresh at online birth",
        "private_state": "fresh at online birth",
    }
    identity = {
        "format": FORMAT,
        "normalizer": normalizer.to_value(),
        "cold_inheritance": conversion,
        "source_identity": source_identity,
    }
    artifact = {
        "format": FORMAT,
        "identity": identity,
        "model": model,
        "goal_manager": manager,
        "candidate_adapters": adapters.state_dict(),
        "frozen_modules": ["visual", "body", "goal_encoder", "goal_decoder"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    torch.save(artifact, temporary)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "bytes": args.output.stat().st_size,
                "file_sha256": sha256(args.output),
                "conversion": conversion,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
