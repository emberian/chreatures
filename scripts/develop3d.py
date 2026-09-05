#!/usr/bin/env python3
"""Run a resumable, batched 3D developmental nursery on a full MaleCNS graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chreatures.development import DevelopmentConfig, DevelopmentNursery
from chreatures.remote_brain import RemoteBrain


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--simple-steps", type=int, default=768)
    parser.add_argument("--rich-steps", type=int, default=1280)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=512)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--inheritance-seed", type=int, default=7301)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--progress-every", type=int, default=64)
    return parser.parse_args()


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    fields = (
        "nutrition", "contacts", "distance", "effort", "energy", "activity",
        "advantage", "prediction_error",
    )
    return {
        f"{name}_{'total' if name in {'nutrition', 'contacts', 'distance'} else 'mean'}":
        float(np.sum([row[name] for row in rows]) if name in {"nutrition", "contacts", "distance"}
              else np.mean([row[name] for row in rows]))
        for name in fields
    }


def main() -> int:
    args = arguments()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worlds < 1 or args.worlds > 16:
        raise SystemExit("--worlds must be between 1 and 16")
    if args.simple_steps < 0 or args.rich_steps < 0 or args.simple_steps + args.rich_steps < 1:
        raise SystemExit("development steps must be nonnegative and total at least one")
    if not 0 < args.dt <= 0.2:
        raise SystemExit("--dt must be in (0, 0.2]")

    config = DevelopmentConfig(
        worlds=args.worlds,
        simple_steps=args.simple_steps,
        rich_steps=args.rich_steps,
        dt=args.dt,
        checkpoint_every=args.checkpoint_every,
        record_every=args.record_every,
        workers=args.workers,
        seed=args.seed,
        inheritance_seed=args.inheritance_seed,
    )
    started_wall = time.time()
    started = time.perf_counter()
    brain = RemoteBrain.from_malecns(
        args.graph,
        capacity=config.residents,
        device=args.device,
    )
    nursery = (
        DevelopmentNursery.restore(brain, args.resume, output_directory=args.output)
        if args.resume
        else DevelopmentNursery(brain, args.output, config)
    )
    if nursery.config != config:
        raise SystemExit("resume arguments differ from the checkpoint configuration")

    source_paths = [
        ROOT / "chreatures" / name
        for name in (
            "development.py", "remote_brain.py", "malecns.py", "cognition.py",
            "physics.py", "neural_client.py",
        )
    ] + [Path(__file__).resolve(), ROOT / "data/habitats/hollow-garden.json"]
    provenance = {
        "format": "chreatures-development-run-v1",
        "started_unix": started_wall,
        "argv": [sys.executable, *sys.argv],
        "command": shlex.join([sys.executable, *sys.argv]),
        "pid": os.getpid(),
        "environment": {
            name: os.environ[name]
            for name in ("HSA_OVERRIDE_GFX_VERSION", "CHREATURES_MALECNS_DIR")
            if name in os.environ
        },
        "python": sys.version,
        "torch": {"version": torch.__version__, "hip": torch.version.hip},
        "brain": brain.metadata(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
    }
    (args.output / "run.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    rows: list[dict[str, float]] = []
    checkpoints = []
    try:
        while nursery.step_index < config.total_steps and not stop:
            row = nursery.step()
            rows.append(row)
            if args.progress_every and nursery.step_index % args.progress_every == 0:
                rate = nursery.step_index / max(time.perf_counter() - started, 1e-6)
                print(
                    f"step={nursery.step_index}/{config.total_steps} phase={nursery.phase} "
                    f"rate={rate:.2f} steps/s energy={row['energy']:.4f} "
                    f"prediction_error={row['prediction_error']:.5f}",
                    flush=True,
                )
            if config.checkpoint_every and nursery.step_index % config.checkpoint_every == 0:
                checkpoints.append(nursery.save_checkpoint())
                nursery.save_trajectory()
        if not checkpoints or checkpoints[-1]["step"] != nursery.step_index:
            checkpoints.append(nursery.save_checkpoint())
        trajectory = nursery.save_trajectory()
        manifest = nursery.export()
    finally:
        nursery.close()

    elapsed = time.perf_counter() - started
    quarter = max(1, len(rows) // 4)
    summary = {
        "completed": nursery.step_index == config.total_steps,
        "stopped_by_signal": stop,
        "steps_at_start": nursery.step_index - len(rows),
        "steps_at_end": nursery.step_index,
        "steps_this_invocation": len(rows),
        "resident_steps_this_invocation": len(rows) * config.residents,
        "elapsed_seconds": elapsed,
        "steps_per_second": len(rows) / max(elapsed, 1e-9),
        "resident_steps_per_second": len(rows) * config.residents / max(elapsed, 1e-9),
        "first_quarter": aggregate(rows[:quarter]),
        "last_quarter": aggregate(rows[-quarter:]),
        "all": aggregate(rows),
        "checkpoints": checkpoints,
        "trajectory": {
            "path": trajectory.name,
            "bytes": trajectory.stat().st_size,
            "sha256": sha256(trajectory),
        },
        "egg": manifest,
        "finished_unix": time.time(),
        "final_device": brain.metadata()["device"],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["completed"] else 130


if __name__ == "__main__":
    raise SystemExit(main())
