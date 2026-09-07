#!/usr/bin/env python3
"""Fit the native GAM ecological transfer atlas from two completed campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.gam.transfer_atlas import fit_atlas, prepare_life_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--principal-result", type=Path, required=True)
    parser.add_argument("--challenge-result", type=Path, required=True)
    parser.add_argument("--principal-receipt", type=Path, required=True)
    parser.add_argument("--challenge-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, source = prepare_life_rows(
        args.search,
        args.principal_result,
        args.challenge_result,
        args.principal_receipt,
        args.challenge_receipt,
    )
    report = fit_atlas(rows, source, args.search, args.output)
    print(json.dumps({
        "atlas": str(args.output / "atlas.json"),
        "rows": len(rows),
        "split_rows": report["split_rows"],
        "fit_seconds": report["fit_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
