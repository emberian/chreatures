#!/usr/bin/env python3
"""Export a trusted Torch training checkpoint as a pure NumPy motor organ."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.motor_inheritance import ACTIONS, ARTIFACT_FORMAT, artifact_identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--port-spec-sha256", required=True)
    parser.add_argument("--port-bundle-sha256")
    parser.add_argument("--run-record", type=Path)
    parser.add_argument("--cohort-checkpoint", type=Path,
                        help="full cohort JSON.gz used to derive the physical interface")
    parser.add_argument("--sensorium-interface")
    parser.add_argument("--body-interface")
    parser.add_argument("--chemical-model")
    parser.add_argument("--physical-spec-sha256")
    parser.add_argument("--trusted-checkpoint", action="store_true",
                        help="confirm the Torch/pickle checkpoint comes from a trusted training run")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if not args.trusted_checkpoint:
        raise SystemExit("refusing Torch deserialization without --trusted-checkpoint")
    for name in ("graph_sha256", "port_spec_sha256"):
        if len(getattr(args, name)) != 64:
            raise SystemExit(f"--{name.replace('_', '-')} must be a 64-character SHA-256")
    # Imported only in the trusted, Torch-equipped export environment. The
    # deployment artifact and its loader never import or deserialize Torch.
    from chreatures.learning import PredictivePPOTrainer

    checkpoint = args.checkpoint.resolve()
    trainer, extra = PredictivePPOTrainer.restore(checkpoint, device="cpu")
    state = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in trainer.model.state_dict().items()
    }
    moments = trainer.moments.snapshot()
    state.update({
        "normalizer_count": np.asarray(moments["count"], dtype=np.float64),
        "normalizer_mean": np.asarray(moments["mean"], dtype=np.float64),
        "normalizer_m2": np.asarray(moments["m2"], dtype=np.float64),
    })
    run = None
    if args.run_record:
        run = json.loads(args.run_record.read_text())
        for supplied, recorded in (
            (args.graph_sha256, run.get("graph_sha256")),
            (args.port_spec_sha256, run.get("port_spec_sha256")),
        ):
            if recorded is not None and supplied != recorded:
                raise SystemExit("supplied provenance differs from run record")
    interface = {
        "sensorium": args.sensorium_interface or "unknown",
        "body": args.body_interface or "unknown",
        "chemical_model": args.chemical_model or "unknown",
        "physical_spec_sha256": args.physical_spec_sha256 or "unknown",
        "physics_model_signature": "unknown",
        "source_physical_spec_sha256": "unknown",
        "source_physics_sha256": "unknown",
        "source_sensorium_sha256": "unknown",
        "derivation": "explicit CLI values where supplied; otherwise unavailable",
    }
    if run:
        source_hashes = run.get("source_sha256", {})
        interface["source_physical_spec_sha256"] = source_hashes.get(
            "data/habitats/hollow-garden.json", "unknown"
        )
        interface["source_physics_sha256"] = source_hashes.get("chreatures/physics.py", "unknown")
        interface["source_sensorium_sha256"] = source_hashes.get("chreatures/sensorium.py", "unknown")
    cohort_hash = None
    cohort_step = None
    if args.cohort_checkpoint:
        cohort_path = args.cohort_checkpoint.resolve()
        cohort_hash = sha256(cohort_path)
        with gzip.open(cohort_path, "rt", encoding="utf-8") as handle:
            cohort = json.load(handle)
        cohort_step = int(cohort["step"])
        if (
            cohort.get("graph_sha256") != args.graph_sha256
            or cohort.get("port_spec_sha256") != args.port_spec_sha256
        ):
            raise SystemExit("cohort checkpoint graph or port provenance differs")
        worlds = cohort.get("worlds", [])
        if not worlds or not isinstance(worlds[0].get("spec"), dict):
            raise SystemExit("cohort checkpoint does not contain a physical world spec")
        spec = worlds[0]["spec"]
        spec_hash = hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if args.physical_spec_sha256 and args.physical_spec_sha256 != spec_hash:
            raise SystemExit("supplied physical-spec hash differs from cohort checkpoint")
        interface["physical_spec_sha256"] = spec_hash
        interface["physics_model_signature"] = str(worlds[0].get("model_signature", "unknown"))
        body_spec = spec.get("articulated_body_spec")
        if not args.body_interface and isinstance(body_spec, dict):
            name = body_spec.get("name", "unnamed-articulated-body")
            version = body_spec.get("version", "unknown")
            interface["body"] = f"{name}:v{version}"
        if "sensorium" not in spec:
            if not args.sensorium_interface:
                interface["sensorium"] = "legacy-world-v0-default (spec.sensorium absent)"
            if not args.chemical_model:
                interface["chemical_model"] = "analytic-odor-default (spec.sensorium absent)"
        interface["derivation"] = (
            "physical spec and model signature from cohort checkpoint; explicit CLI values take precedence; "
            "absent spec.sensorium identifies the training code's legacy camera and analytic odor defaults"
        )
    provenance = {
        "checkpoint_sha256": sha256(checkpoint),
        "graph_sha256": args.graph_sha256,
        "port_spec_sha256": args.port_spec_sha256,
        "port_bundle_sha256": args.port_bundle_sha256,
        "run_record_sha256": sha256(args.run_record.resolve()) if args.run_record else None,
        "cohort_checkpoint_sha256": cohort_hash,
        "training_steps": (
            cohort_step if cohort_step is not None
            else (extra.get("step") if isinstance(extra, dict) else None)
        ),
        "training_interface": interface,
    }
    metadata = {
        "format": ARTIFACT_FORMAT, "version": 1, "actions": list(ACTIONS),
        "config": asdict(trainer.config), "updates": trainer.update_count,
        "decisions": trainer.decision_count,
        "training_provenance": provenance,
        "scope": "immutable inherited policy, value, predictor, transforms and frozen feature normalizer",
    }
    metadata["artifact_sha256"] = artifact_identity(metadata, state)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **state)
    os.replace(temporary, output)
    print(json.dumps({"path": str(output), "bytes": output.stat().st_size,
                      "file_sha256": sha256(output), **metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
