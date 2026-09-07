#!/usr/bin/env python3
"""Verify that current profile identity survives exact source relocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.training_environment import (  # noqa: E402
    EmbodiedTrainingProfile,
    _generated_family_spec,
)


DEFAULT_RECEIPT = ROOT / "data/training/profile-v7-relocation.receipt.json"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--habitat", type=Path, default=ROOT / "data/habitats/living-reef.json"
    )
    parser.add_argument(
        "--biosphere", type=Path, default=ROOT / "data/biosphere/living-reef.json"
    )
    parser.add_argument(
        "--family-config",
        type=Path,
        default=ROOT / "data/habitat-families/regional-v3.json",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=ROOT / "data/training/regional-environment-schedule-v1.json",
    )
    parser.add_argument("--staging-parent", type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--expect", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def manifest(value: Mapping[str, str]) -> dict[str, Any]:
    entries = {name: str(path) for name, path in sorted(value.items())}
    return {"entries": entries, "sha256": value_sha256(entries)}


def stable_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    profile = receipt["profile"]
    return {
        "profile": {
            key: profile[key]
            for key in ("format", "version", "semantic_sha256", "source_count")
        },
        "source_assets": receipt["source_assets"],
        "generated_environment": receipt["generated_environment"],
        "checks": receipt["checks"],
    }


def main() -> int:
    args = arguments()
    profile = EmbodiedTrainingProfile.nursery_family(
        args.habitat, args.biosphere, args.family_config, args.schedule
    )
    encoded = profile.to_value()
    first_locators = profile.locator_manifest()
    staging_parent = (
        str(args.staging_parent.resolve()) if args.staging_parent is not None else None
    )
    if args.staging_parent is not None:
        args.staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="chreatures-profile-v7-relocated-", dir=staging_parent
    ) as temporary:
        relocated_root = Path(temporary)
        second_locators: dict[str, str] = {}
        for index, (name, source_value) in enumerate(sorted(first_locators.items())):
            source = Path(source_value)
            relocated = relocated_root / f"{index:02d}-{name}{source.suffix}"
            shutil.copyfile(source, relocated)
            second_locators[name] = str(relocated.resolve())
        rebound = EmbodiedTrainingProfile.from_value(
            encoded, locators=second_locators
        )
        original = _generated_family_spec(profile, 0, False)
        relocated = _generated_family_spec(rebound, 0, False)
        checks = {
            "locator_manifests_differ": first_locators != second_locators,
            "semantic_identity_equal": profile.sha256 == rebound.sha256,
            "generated_habitat_equal": original[0] == relocated[0],
            "generated_biosphere_equal": original[1] == relocated[1],
            "generated_variant_equal": original[2] == relocated[2],
        }
        if not all(checks.values()):
            raise RuntimeError("profile relocation changed semantic or generated identity")
        semantic = profile.semantic_value()
        variant = original[2]
        receipt: dict[str, Any] = {
            "format": "chreatures-profile-v7-relocation-receipt-v1",
            "version": 1,
            "sha256": "",
            "profile": {
                "format": semantic["format"],
                "version": semantic["version"],
                "semantic_sha256": profile.sha256,
                "source_count": len(semantic["sources"]),
                "source_transport_sha256": value_sha256(encoded),
                "relocated_transport_sha256": value_sha256(rebound.to_value()),
            },
            "source_assets": {
                name: source["sha256"]
                for name, source in sorted(semantic["sources"].items())
            },
            "locator_manifests": {
                "source_checkout": manifest(first_locators),
                "relocated_ephemeral_tree": manifest(second_locators),
                "relocated_tree_retained": False,
            },
            "generated_environment": {
                "environment_sha256": variant["environment_sha256"],
                "environment_genome_sha256": variant["environment_genome_sha256"],
                **variant["family_output_sha256"],
            },
            "checks": checks,
            "execution": {
                "git_revision": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "probe_source_sha256": file_sha256(Path(__file__).resolve()),
                "selection": {"split": "training", "index": 0, "chosen_seed": 0},
                "method": (
                    "copy every profile source byte-for-byte to a second path tree; "
                    "explicitly rebind; regenerate the same native regional environment"
                ),
            },
        }
        receipt["sha256"] = value_sha256(receipt)

    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise SystemExit("output receipt already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    else:
        expected = json.loads(args.expect.read_text(encoding="utf-8"))
        unsigned = dict(expected)
        declared = unsigned.pop("sha256", None)
        unsigned["sha256"] = ""
        if declared != value_sha256(unsigned):
            raise ValueError("saved relocation receipt checksum differs")
        if stable_evidence(expected) != stable_evidence(receipt):
            raise ValueError("current relocation evidence differs from receipt")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
