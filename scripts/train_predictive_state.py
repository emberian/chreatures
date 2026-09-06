#!/usr/bin/env python3
"""Train predictive state on audited anonymous episode shards."""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chreatures.predictive_state import (
    PredictiveNormalizer,
    PredictiveSequence,
    PredictiveStateConfig,
    PredictiveStateTrainer,
)  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk(s: PredictiveSequence, start: int, stop: int) -> PredictiveSequence:
    reset = s.reset[start:stop].copy()
    reset[0] = True
    return PredictiveSequence(
        s.features[start:stop],
        s.physiology[start:stop],
        s.actions[start:stop],
        reset,
        s.valid[start:stop],
    ).validated()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, nargs="+")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--chunk-rows", type=int, default=160)
    p.add_argument("--burn-in", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=96)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    manifest_path = a.dataset / "manifest.json"
    layouts_path = a.dataset / "layouts.json"
    manifest = json.loads(manifest_path.read_text())
    layouts = json.loads(layouts_path.read_text())
    available = sorted(
        int(x.name.split("-")[1])
        for x in a.dataset.glob("episode-*")
        if (x / "receipt.json").is_file()
    )
    episodes = a.episodes or available
    if not episodes or not set(episodes) <= set(available):
        raise SystemExit("requested episode shard is unavailable")
    train = [
        PredictiveSequence.from_episode(a.dataset / f"episode-{i:03d}", "train")
        for i in episodes
    ]
    holdout = [
        PredictiveSequence.from_episode(a.dataset / f"episode-{i:03d}", "holdout")
        for i in episodes
    ]
    identity = {
        "dataset_manifest_sha256": sha(manifest_path),
        "layouts_sha256": sha(layouts_path),
        "episodes": episodes,
        "split": manifest["splits"],
        "graph_sha256": manifest["graph_sha256"],
        "port_spec_sha256": manifest["port_spec_sha256"],
        "port_bundle_sha256": manifest["port_bundle_sha256"],
        "source_normalizer": manifest["normalizer"],
        "fit_policy": "train worlds only",
    }
    normalizer = PredictiveNormalizer.fit(train, identity)
    config = PredictiveStateConfig(
        feature_dim=train[0].features.shape[2],
        physiology_dim=6,
        action_dim=8,
        latent_dim=a.latent_dim,
    )
    trainer = PredictiveStateTrainer(config, normalizer, device=a.device)
    history = []
    for _ in range(a.epochs):
        for sequence in train:
            for start in range(0, len(sequence.features) - 1, a.chunk_rows):
                begin = max(0, start - a.burn_in)
                stop = min(len(sequence.features), start + a.chunk_rows + 8)
                history.append(
                    trainer.update(
                        chunk(sequence, begin, stop), loss_start=start - begin
                    )
                )
    heldout = []
    with torch.no_grad():
        for sequence in holdout:
            for start in range(0, len(sequence.features) - 1, a.chunk_rows):
                begin = max(0, start - a.burn_in)
                stop = min(len(sequence.features), start + a.chunk_rows + 8)
                _, metrics = trainer.loss(
                    chunk(sequence, begin, stop), loss_start=start - begin
                )
                heldout.append(metrics)
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint = trainer.checkpoint(a.output / "predictive-state.pt")
    immutable = trainer.export(
        a.output / "predictive-state-rust.npz",
        training_input_identity=identity,
        source_normalizer_path=a.dataset / "normalizer.npz",
        source_dataset_manifest_path=manifest_path,
    )
    keys = [
        k
        for k in heldout[0]
        if k.startswith(("feature_nll_h", "physiology_delta_nll_h"))
    ]
    report = {
        "format": "chreatures-predictive-training-v1",
        "architecture": "normalized-feature-absolute-physiology-delta-h1-h8",
        "episodes": episodes,
        "source_identity": identity,
        "normalizer": normalizer.metadata(),
        "engineered_loss": "0.5 feature Gaussian NLL + 0.5 physiology-delta Gaussian NLL",
        "updates": trainer.update_count,
        "last_training": history[-1],
        "heldout": {k: float(np.mean([row[k] for row in heldout])) for k in keys},
        "checkpoint": checkpoint,
        "immutable_export": immutable,
    }
    (a.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
