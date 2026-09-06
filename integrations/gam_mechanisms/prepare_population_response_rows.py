#!/usr/bin/env python3
"""Prepare causal physiology12 population GAM rows from a completed evaluator trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "integrations/gam_mechanisms/population_feature_contract_v1.json"
HISTORY_WINDOW = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def history_features(pre: np.ndarray) -> np.ndarray:
    """Prior/current-state-only rolling summaries, shaped [ticks,residents,4]."""
    ticks, residents, _ = pre.shape
    result = np.zeros((ticks, residents, 4), dtype=np.float32)
    cumulative = np.concatenate(
        [np.zeros((1, residents, 2), dtype=np.float64),
         np.cumsum(pre[:, :, [0, 2]], axis=0, dtype=np.float64)], axis=0,
    )
    for tick in range(ticks):
        start = max(0, tick - HISTORY_WINDOW + 1)
        count = tick - start + 1
        result[tick, :, :2] = (cumulative[tick + 1] - cumulative[start]) / count
        result[tick, :, 2] = pre[tick, :, 6] - pre[start, :, 6]
        result[tick, :, 3] = pre[tick, :, 7] - pre[start, :, 7]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity_path = args.evaluation / "identity.json"
    result_path = args.evaluation / "result.json"
    if not identity_path.is_file() or not result_path.is_file():
        raise SystemExit("population evaluation must be complete before row preparation")
    identity = json.loads(identity_path.read_text())
    result = json.loads(result_path.read_text())
    if result.get("status") != "completed" or result.get("evaluation_identity_sha256") != identity.get("sha256"):
        raise ValueError("population evaluation result identity differs")
    chunks = sorted((args.evaluation / "gam_trace").glob("tick-*.npz"))
    if not chunks:
        raise ValueError("completed evaluation has no row-level GAM trace")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in
        ("pre_physiology12", "executed_action12", "post_physiology12", "outcomes8", "organ_flows3")}
    expected_tick = 0
    for path in chunks:
        with np.load(path, allow_pickle=False) as chunk:
            if str(chunk["format"]) != "chreatures-population-gam-trace-v1" or str(chunk["evaluation_identity_sha256"]) != identity["sha256"]:
                raise ValueError(f"trace identity differs: {path}")
            ticks = np.asarray(chunk["tick"], dtype=np.uint64)
            if len(ticks) == 0 or int(ticks[0]) != expected_tick or not np.array_equal(ticks, np.arange(expected_tick, expected_tick + len(ticks))):
                raise ValueError(f"trace ticks are incomplete or overlap: {path}")
            expected_tick += len(ticks)
            for key in arrays:
                arrays[key].append(np.asarray(chunk[key], dtype=np.float32))
    if expected_tick != int(result["completed_steps"]):
        raise ValueError("trace does not cover the complete physical episode")
    joined = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    pre, post = joined["pre_physiology12"], joined["post_physiology12"]
    actions, outcomes, flows = joined["executed_action12"], joined["outcomes8"], joined["organ_flows3"]
    if pre.shape != post.shape or pre.shape[2] != 12 or actions.shape != pre.shape or outcomes.shape[:2] != pre.shape[:2] or flows.shape[:2] != pre.shape[:2]:
        raise ValueError("population trace tensor contract differs")
    history = history_features(pre)
    features = np.concatenate([pre, history, actions], axis=2).reshape(-1, 28)
    targets = np.stack([
        post[:, :, 0] - pre[:, :, 0], post[:, :, 2] - pre[:, :, 2], outcomes[:, :, 3],
        outcomes[:, :, 5], outcomes[:, :, 1], flows[:, :, 0], flows[:, :, 1], flows[:, :, 2],
    ], axis=2).reshape(-1, 8)
    lives = identity["life_records"]
    residents = pre.shape[1]
    if len(lives) != residents:
        raise ValueError("trace resident order differs from sealed life records")
    repeat = pre.shape[0]
    def units(key: str) -> np.ndarray:
        return np.tile(np.asarray([str(row[key]) for row in lives]), repeat)
    lineage = []
    assignment = json.loads(Path(identity["assignments"]["path"]).read_text())
    for world in assignment["worlds"]:
        for candidate in world["candidates"]:
            ancestry = candidate.get("ancestry", {})
            lineage.append(str(ancestry.get("founder_sha256") or ancestry.get("parent_sha256") or candidate["sha256"]))
    source_hashes = [sha256(path) for path in chunks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, features=features.astype(np.float32), targets=targets.astype(np.float32),
        lineage_unit=np.tile(np.asarray(lineage), repeat), environment_unit=units("environment_sha256"),
        candidate_unit=units("candidate_sha256"), episode_unit=np.full(len(features), identity["sha256"]),
        world_unit=units("world_slot"), source_sha256=np.asarray(hashlib.sha256("".join(source_hashes).encode()).hexdigest()),
    )
    contract = json.loads(CONTRACT.read_text())
    target_names = ["energy_state_delta", "fatigue_state_delta", "effort", "ingested_mass", "contact",
                    "release_mass", "secretion_mass", "allocation_mass"]
    receipt = {"format":"chreatures-population-gam-rows-v1", "evaluation_identity_sha256":identity["sha256"],
        "evaluation_result_sha256":sha256(result_path), "feature_contract_sha256":sha256(CONTRACT),
        "history_window_ticks":HISTORY_WINDOW, "history_window_seconds":HISTORY_WINDOW * 0.05,
        "feature_order":[x["name"] for x in contract["features"]], "target_order":target_names,
        "rows":len(features), "physical_ticks":pre.shape[0], "lives":residents,
        "trace_chunk_sha256":dict(zip([x.name for x in chunks], source_hashes, strict=True)),
        "output_sha256":sha256(args.output)}
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
