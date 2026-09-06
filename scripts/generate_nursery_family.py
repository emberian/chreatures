#!/usr/bin/env python3
"""Generate deterministic, build-time nursery habitats from pinned templates."""

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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return resolved.name


def _atomic_write(path: Path, value: str) -> None:
    staging = path.with_suffix(path.suffix + ".part")
    staging.write_text(value, encoding="utf-8")
    staging.replace(path)


def _variant(value: str) -> tuple[str, int]:
    try:
        family, raw_seed = value.rsplit(":", 1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("variant must be FAMILY:SEED") from exc
    if not family or not 0 <= seed < 2**64:
        raise argparse.ArgumentTypeError("variant family/seed is invalid")
    return family, seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pinned Living Reef nursery-family artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "data/habitat-families/nursery-v1.json",
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
        "--variant",
        action="append",
        type=_variant,
        help="FAMILY:SEED; repeat as needed (default: pinned training variants)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace a previously generated directory",
    )
    args = parser.parse_args()

    config_bytes = args.config.resolve().read_bytes()
    habitat_bytes = args.habitat_template.resolve().read_bytes()
    biosphere_bytes = args.biosphere_template.resolve().read_bytes()
    native = load_world_kernels()
    family_type = getattr(native, "HabitatFamily", None)
    if family_type is None:
        raise SystemExit("native world kernels do not provide HabitatFamily")
    generator = family_type(config_bytes.decode(), _sha256_bytes(config_bytes))
    variants = args.variant or list(generator.training_variants())
    if len(set(variants)) != len(variants):
        raise SystemExit("duplicate family/seed variant")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.replace:
        raise SystemExit(f"output is nonempty: {output}; pass --replace")
    output.mkdir(parents=True, exist_ok=True)
    if args.replace:
        for path in output.glob("nursery-*.*.json"):
            path.unlink()
        (output / "manifest.json").unlink(missing_ok=True)

    manifest: dict[str, Any] = {
        "format": "chreatures-nursery-training-family-v1",
        "generator": {
            "class": "_world_kernels.HabitatFamily",
            "config": _source_ref(args.config),
            "config_sha256": _sha256_bytes(config_bytes),
            "rust_source_sha256": _sha256(
                ROOT / "native/world-kernels/src/habitat_family.rs"
            ),
            "python_source_sha256": _sha256(Path(__file__).resolve()),
            "extension": Path(native.__file__).name,
            "extension_sha256": _sha256(Path(native.__file__).resolve()),
        },
        "inputs": {
            "habitat_template": _source_ref(args.habitat_template),
            "habitat_template_sha256": _sha256_bytes(habitat_bytes),
            "biosphere_template": _source_ref(args.biosphere_template),
            "biosphere_template_sha256": _sha256_bytes(biosphere_bytes),
        },
        "variants": [],
    }
    for family, seed in variants:
        habitat, biosphere, analyst = generator.generate(
            habitat_bytes.decode(), biosphere_bytes.decode(), seed, family
        )
        stem = f"nursery-{family}-{seed}"
        paths = {
            "habitat": output / f"{stem}.habitat.json",
            "biosphere": output / f"{stem}.biosphere.json",
            "analyst": output / f"{stem}.analyst.json",
        }
        values = {"habitat": habitat, "biosphere": biosphere, "analyst": analyst}
        for key, path in paths.items():
            _atomic_write(path, values[key])
        analyst_value = json.loads(analyst)
        manifest["variants"].append(
            {
                "family": family,
                "seed": seed,
                "files": {
                    key: {
                        "path": path.name,
                        "sha256": _sha256(path),
                    }
                    for key, path in paths.items()
                },
                "graph": {
                    "nodes": len(analyst_value["graph"]["nodes"]),
                    "edges": len(analyst_value["graph"]["edges"]),
                    "connected": analyst_value["graph"]["connected"],
                },
            }
        )
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest["content_sha256"] = _sha256_bytes(payload.encode())
    _atomic_write(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "variants": len(variants), "content_sha256": manifest["content_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
