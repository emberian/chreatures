#!/usr/bin/env python3
"""Materialize one current native regional environment genome headlessly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chreatures.native_world import load_world_kernels


OUTPUT_NAMES = (
    "environment.genome.json",
    "habitat.json",
    "biosphere.json",
    "analyst.json",
    "manifest.json",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _atomic_json(path: Path, value: Any) -> None:
    staging = path.with_suffix(path.suffix + ".part")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    staging.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    staging = path.with_suffix(path.suffix + ".part")
    staging.write_text(value, encoding="utf-8")
    staging.replace(path)


def _uint64(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if not 0 <= parsed < 2**64:
        raise argparse.ArgumentTypeError("value must be an unsigned 64-bit integer")
    return parsed


def _sha(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256 digest")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one concrete habitat/biosphere from the current regional grammar."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "data/habitat-families/regional-v3.json",
    )
    parser.add_argument(
        "--resident-bundle",
        type=Path,
        default=ROOT / "data/habitat-families/regional-residents-v2.json",
    )
    parser.add_argument(
        "--habitat-template",
        type=Path,
        default=ROOT / "data/habitats/living-reef.json",
    )
    parser.add_argument(
        "--biosphere-template",
        type=Path,
        default=ROOT / "data/biosphere/living-reef.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace this command's five known output files",
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    initial = modes.add_parser("initial", help="sample a founder environment genome")
    initial.add_argument("--archetype", required=True)
    initial.add_argument("--seed", type=_uint64, required=True)
    initial.add_argument("--residents", type=int, required=True, choices=range(1, 33))
    initial.add_argument("--profile-sha256", type=_sha, required=True)
    initial.add_argument("--epoch", type=_uint64, default=0)
    mutate = modes.add_parser("mutate", help="derive an inherited environment genome")
    mutate.add_argument("--parent-genome", type=Path, required=True)
    mutate.add_argument("--parent-analyst", type=Path, required=True)
    mutate.add_argument("--variation-seed", type=_uint64, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    sources = {
        "config": args.config.resolve(),
        "resident_bundle": args.resident_bundle.resolve(),
        "habitat_template": args.habitat_template.resolve(),
        "biosphere_template": args.biosphere_template.resolve(),
    }
    if any(not path.is_file() for path in sources.values()):
        raise SystemExit("one or more regional source files are absent")
    config_text = sources["config"].read_text()
    habitat_text = sources["habitat_template"].read_text()
    biosphere_text = sources["biosphere_template"].read_text()
    residents = json.loads(sources["resident_bundle"].read_text())
    native = load_world_kernels()
    family_type = getattr(native, "HabitatFamily", None)
    if family_type is None:
        raise SystemExit("native world kernels do not provide HabitatFamily")
    family = family_type(config_text, _sha256_bytes(config_text.encode()))
    if args.mode == "initial":
        genome_text = family.initial_genome(
            args.seed, args.archetype, args.residents, args.profile_sha256, args.epoch
        )
    else:
        parent = json.loads(args.parent_genome.resolve().read_text())
        parent_analyst = json.loads(args.parent_analyst.resolve().read_text())
        parent_record = parent_analyst.get("environment_record")
        if not isinstance(parent_record, dict) or not isinstance(
            parent_record.get("sha256"), str
        ):
            raise SystemExit("parent analyst omits its environment record")
        genome_text = family.mutate_genome(
            _canonical(parent).decode(), parent_record["sha256"], args.variation_seed
        )
    genome = json.loads(genome_text)
    resident_count = genome["parameters"]["resident_count"]
    resident_values = residents.get("residents")
    if (
        residents.get("format") != "chreatures-regional-residents-v2"
        or not isinstance(resident_values, list)
        or len(resident_values) < resident_count
    ):
        raise SystemExit("regional resident bundle lacks the requested capacity")
    selected_residents = _canonical(
        {
            "format": "chreatures-regional-residents-v2",
            "residents": resident_values[:resident_count],
        }
    ).decode()
    habitat_output, biosphere_output, analyst_output = family.generate(
        habitat_text,
        biosphere_text,
        _canonical(genome).decode(),
        selected_residents,
    )
    analyst = json.loads(analyst_output)
    record = analyst.get("environment_record")
    if (
        analyst.get("runtime_visible") is not False
        or not isinstance(record, dict)
        or record.get("topology_sha256") != _sha256_bytes(habitat_output.encode())
        or record.get("resource_sha256") != _sha256_bytes(biosphere_output.encode())
    ):
        raise RuntimeError("native regional output identity is inconsistent")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    occupied = [name for name in OUTPUT_NAMES if (output / name).exists()]
    if occupied and not args.replace:
        raise SystemExit(
            f"regional outputs already exist in {output}; pass --replace to replace them"
        )
    values = {
        "environment.genome.json": genome_text,
        "habitat.json": habitat_output,
        "biosphere.json": biosphere_output,
        "analyst.json": analyst_output,
    }
    for name, value in values.items():
        _atomic_text(output / name, value)

    native_path = Path(native.__file__).resolve()
    manifest = {
        "format": "chreatures-regional-family-export-v1",
        "sha256": "",
        "environment_genome_sha256": genome["sha256"],
        "environment_record": record,
        "resident_count": genome["parameters"]["resident_count"],
        "generator": {
            "class": "_world_kernels.HabitatFamily",
            "extension": str(native_path),
            "extension_sha256": _sha256(native_path),
            "rust_source_sha256": _sha256(
                ROOT / "native/world-kernels/src/habitat_family.rs"
            ),
            "python_source_sha256": _sha256(Path(__file__).resolve()),
        },
        "inputs": {
            name: {"path": _source_ref(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "files": {
            name: {"path": name, "sha256": _sha256(output / name)}
            for name in values
        },
    }
    manifest["sha256"] = _sha256_bytes(_canonical(manifest))
    _atomic_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "environment_genome_sha256": genome["sha256"],
                "environment_sha256": record["sha256"],
                "resident_count": genome["parameters"]["resident_count"],
                "manifest_sha256": manifest["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
