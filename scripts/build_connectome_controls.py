#!/usr/bin/env python3
"""Build a full sparse, matched-rewiring MaleCNS control artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chreatures.connectome_controls import MatchedRewireSpec, build_matched_control
from chreatures.malecns import MaleCNSGraph


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-destructive MaleCNS directed double-swap control. "
            "Large output and scratch paths should be on /tank."
        )
    )
    parser.add_argument(
        "--graph", type=Path, default=Path("/tank/chreatures/data/malecns/derived")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--name", default="matched-rewire-v1")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--category-field", default="superclasses")
    parser.add_argument("--transmitter-field", default="effective_nt")
    parser.add_argument("--quick-check-rows", type=int, default=4096)
    parser.add_argument(
        "--verify-source", action="store_true",
        help="hash the complete source artifact before building",
    )
    args = parser.parse_args()

    graph = MaleCNSGraph.load(args.graph, mmap=True, verify=args.verify_source)
    spec = MatchedRewireSpec(
        name=args.name,
        seed=args.seed,
        category_field=args.category_field,
        transmitter_field=args.transmitter_field,
    )
    receipt = build_matched_control(
        graph,
        args.output,
        spec,
        scratch_directory=args.scratch,
        quick_check_rows=args.quick_check_rows,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
