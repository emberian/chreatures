#!/usr/bin/env python3
"""Build the immutable 3D/development observatory from actual run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.observatory import DEFAULT_OUTPUT, ObservatoryError, build_observatory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=ROOT / "runs" / "hollow-garden.json")
    parser.add_argument(
        "--development",
        type=Path,
        default=ROOT / "runs" / "development" / "initial-8x3-20260905",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = build_observatory(args.world, args.development, args.output)
    except ObservatoryError as exc:
        raise SystemExit(f"observatory: {exc}") from None
    manifest = result["manifest"]
    report = result["report"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_set_sha256": manifest["artifact_set_sha256"],
                "artifacts": len(manifest["artifacts"]),
                "native_weave_roundtrip": manifest["native_weave_roundtrip"],
                "gamfit_status": report["gamfit"]["status"],
                "held_out_worlds": report["development"]["whole_world_holdout"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
