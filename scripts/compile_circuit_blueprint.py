#!/usr/bin/env python3
"""Compile a pinned circuit blueprint into bulk graph, ports, and native CSR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.circuit_blueprint import (
    CircuitBlueprint,
    compile_blueprint,
)
from chreatures.malecns import MaleCNSGraph
from chreatures.neural_ports import NeuralPortBundle


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blueprint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("/tank/chreatures/data/malecns/derived"),
    )
    parser.add_argument(
        "--ports", type=Path, default=ROOT / "data/ports/retinal-v1-maps.npz"
    )
    parser.add_argument(
        "--selector-root",
        type=Path,
        default=ROOT,
        help="base directory for selector artifact paths in the blueprint",
    )
    args = parser.parse_args()
    graph = MaleCNSGraph.load(args.graph, mmap=True)
    ports = NeuralPortBundle.load(args.ports, graph)
    receipt = compile_blueprint(
        graph,
        ports,
        CircuitBlueprint.load(args.blueprint),
        args.output,
        selector_root=args.selector_root,
        parent_port_sha256=sha256(args.ports),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
