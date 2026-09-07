#!/usr/bin/env python3
"""Query an authenticated population search through a fitted native GAM atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.gam.transfer_atlas import proposal_scores, rank_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--proposal-scores-output", type=Path)
    parser.add_argument(
        "--proposal-criterion",
        choices=("energy_change", "contact_ticks", "mechanical_work_rate", "allocation_rate", "interaction_information"),
    )
    parser.add_argument("--proposal-policy-output", type=Path)
    parser.add_argument("--proposal-limit", type=int)
    args = parser.parse_args()
    response = rank_pairs(args.atlas, args.search, args.request)
    text = json.dumps(response, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    requested = args.proposal_scores_output is not None
    if args.proposal_limit is not None and not requested:
        parser.error("proposal limit requires proposal score output")
    if requested != (
        args.proposal_criterion is not None and args.proposal_policy_output is not None
    ):
        parser.error(
            "proposal score output, criterion, and policy output must be provided together"
        )
    if requested:
        proposal = proposal_scores(
            response,
            args.atlas,
            args.proposal_criterion,
            args.proposal_policy_output,
            limit=args.proposal_limit,
        )
        args.proposal_scores_output.parent.mkdir(parents=True, exist_ok=True)
        args.proposal_scores_output.write_text(
            json.dumps(proposal, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
