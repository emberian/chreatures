#!/usr/bin/env python3
"""Build a minimal Universal Weave request for the executed G×E atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.gam.transfer_atlas import file_sha256


def blob(role: str, sha256: str, *, path: Path | None = None) -> dict:
    value = {"role": role, "uri": f"urn:sha256:{sha256}", "sha256": sha256}
    if path is not None:
        if file_sha256(path) != sha256:
            raise ValueError(f"{role} hash differs")
        value.update(bytes=path.stat().st_size, media_type="application/json")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--principal-receipt", type=Path, required=True)
    parser.add_argument("--challenge-receipt", type=Path, required=True)
    parser.add_argument("--recorded-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atlas = json.loads(args.atlas.read_text())
    execution = json.loads(args.execution_receipt.read_text())
    principal = json.loads(args.principal_receipt.read_text())
    challenge = json.loads(args.challenge_receipt.read_text())
    if atlas.get("format") != "chreatures-native-gxe-transfer-atlas-v1":
        raise ValueError("G×E atlas format differs")
    if execution.get("fit", {}).get("atlas_file_sha256") != file_sha256(args.atlas):
        raise ValueError("execution receipt does not authenticate atlas")
    if atlas["source"]["principal_receipt_file_sha256"] != file_sha256(args.principal_receipt):
        raise ValueError("atlas does not authenticate principal receipt")
    if atlas["source"]["challenge_receipt_file_sha256"] != file_sha256(args.challenge_receipt):
        raise ValueError("atlas does not authenticate challenge receipt")
    energy = atlas["models"]["energy_change"]["variants"]
    atlas_sha = file_sha256(args.atlas)
    principal_id = f'campaign:principal:{principal["content_sha256"]}'
    challenge_id = f'campaign:challenge:{challenge["content_sha256"]}'
    request = {
        "archive_id": f"gxe-atlas:{atlas_sha}",
        "description": "Authenticated campaigns and their analyst-only native GAM transfer atlas.",
        "evidence": [
            {
                "id": principal_id,
                "time": "2026-09-07T01:10:34.220146Z",
                "record_type": "evidence",
                "text": "Principal campaign completed 80 physical life observations across ten shared multi-resident worlds.",
                "artifact_uri": f'urn:sha256:{principal["content_sha256"]}',
                "blob_refs": [
                    blob("principal_campaign_receipt", file_sha256(args.principal_receipt), path=args.principal_receipt),
                    blob("principal_evaluation_result", atlas["source"]["principal_result_file_sha256"]),
                ],
                "parent_ids": [],
                "fields": {"life_observation_rows": 80, "shared_residents_per_world": 8,
                           "statistical_independence_claim": False},
            },
            {
                "id": challenge_id,
                "time": "2026-09-07T01:28:45Z",
                "record_type": "evidence",
                "text": "Challenge campaign completed 80 transfer observations of ten selected genotypes in a sparse environment matrix.",
                "artifact_uri": f'urn:sha256:{challenge["content_sha256"]}',
                "blob_refs": [
                    blob("challenge_campaign_receipt", file_sha256(args.challenge_receipt), path=args.challenge_receipt),
                    blob("challenge_evaluation_result", atlas["source"]["challenge_result_file_sha256"]),
                ],
                "parent_ids": [principal_id],
                "fields": {"life_observation_rows": 80, "transfer_genotypes": 10,
                           "shared_residents_per_world": 8, "statistical_independence_claim": False},
            },
            {
                "id": f"gam-gxe-atlas:{atlas_sha}",
                "time": args.recorded_at,
                "record_type": "gam_law_fit",
                "text": "Native GAM transfer atlas fit 160 life observations; all energy changes were negative and only a weak descriptive energy model passed grouped baseline comparisons.",
                "artifact_uri": f"urn:sha256:{atlas_sha}",
                "blob_refs": [
                    blob("native_gxe_atlas", atlas_sha, path=args.atlas),
                    blob("gxe_execution_receipt", file_sha256(args.execution_receipt), path=args.execution_receipt),
                ],
                "parent_ids": [principal_id, challenge_id],
                "fields": {
                    "observation_rows": 160,
                    "shared_world_interaction": True,
                    "uncertainty_claim": "none",
                    "energy_changes_all_negative": True,
                    "energy_validation_rmse": energy["additive"]["validation"]["rmse"],
                    "energy_validation_mean_baseline_rmse": energy["additive"]["validation"]["mean_baseline_rmse"],
                    "energy_heldout_rmse": energy["additive"]["heldout_reporting_only"]["rmse"],
                    "energy_heldout_mean_baseline_rmse": energy["additive"]["heldout_reporting_only"]["mean_baseline_rmse"],
                    "promoted_interactions": 0,
                    "scope": "analyst-side descriptive association; no causal, fitness, survival, feeding, or resident-input claim",
                },
            },
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
