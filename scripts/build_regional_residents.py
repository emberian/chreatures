#!/usr/bin/env python3
"""One-way conversion of an explicit archived v2 founder bundle into current v3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FORMAT = "chreatures-regional-residents-v2"
FORMAT = "chreatures-regional-residents-v3"


def canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def coefficient(seed: bytes, label: str, *, signed: bool) -> float:
    word = int.from_bytes(hashlib.sha256(seed + label.encode()).digest()[:2], "big")
    magnitude = 0.006 * (1 + word % 8)  # 0.006 .. 0.048, below the 0.06 bound.
    return (-magnitude if signed and word & 8 else magnitude)


def build(source: dict[str, object]) -> dict[str, object]:
    if source.get("format") != SOURCE_FORMAT or not isinstance(source.get("residents"), list):
        raise ValueError("regional resident source format differs")
    result = copy.deepcopy(source)
    result["format"] = FORMAT
    for index, resident in enumerate(result["residents"]):
        if not isinstance(resident, dict) or not isinstance(resident.get("founders"), dict):
            raise ValueError("regional resident founder differs")
        seed = hashlib.sha256(canonical({"index": index, "founder": resident})).digest()
        for compartment_index, (name, founder) in enumerate(resident["founders"].items()):
            enzymes = founder.get("enzymes")
            if not isinstance(enzymes, dict):
                raise ValueError("founder enzymes differ")
            allowed = tuple(enzymes)
            if name == "structure":
                allowed = tuple(key for key in allowed if key in {"soft_turnover", "tough_turnover"})
            elif name not in {"body", "gut"}:
                allowed = ()
            tau = (4.0, 12.0, 36.0, 96.0)[(index + compartment_index) % 4]
            cost = (0.04, 0.10, 0.24, 0.60)[(3 * index + compartment_index) % 4]
            founder["regulation"] = {
                "baseline": copy.deepcopy(enzymes),
                "substrate_response": {
                    reaction: coefficient(seed, f"{name}:substrate:{reaction}", signed=True)
                    for reaction in allowed
                },
                "atp_response": {
                    reaction: coefficient(seed, f"{name}:atp:{reaction}", signed=True)
                    for reaction in allowed
                },
                "time_constant_seconds": tau,
                "change_cost_atp_per_expression": cost,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, required=True,
        help="explicit archived v2 input; obtain it from the pinned source revision",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = build(json.loads(args.source.read_text()))
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=args.output.name + ".", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({
        "path": str(args.output),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "residents": len(value["residents"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
