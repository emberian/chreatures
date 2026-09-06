#!/usr/bin/env python3
"""Profile Biosphere overhead from an offline grown-world checkpoint copy.

The research-reference branch binds the previous optimized Python tissue scan
and turnover behavior to one disposable instance. It is never selectable by a
runtime. This script restores only the physical world and Biosphere; it does not
construct or load Habitat3D, restore neural state, or contact a service.

Copy a checkpoint away from a running world before invoking this probe, for
example::

    cp runs/recycling-garden.json /tmp/chreatures-recycling-profile.json
    .venv/bin/python scripts/probe_grown_biosphere.py \
        --checkpoint /tmp/chreatures-recycling-profile.json
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.biosphere import Biosphere
from chreatures.runtime3d import physical_world_type


def _install_research_reference(biosphere: Biosphere) -> None:
    """Bind the exact prior hot-path behavior to one disposable instance."""

    def resource_matrix(self, owned):
        names = self.web.chemistry.pools
        allowed = set(names)
        if not owned:
            return np.empty((0, len(names)), dtype=np.float64)
        rows = []
        for part in owned:
            resources = part["resources"]
            if not isinstance(resources, dict) or set(resources) - allowed:
                raise ValueError("unknown chemical resource")
            rows.append([resources.get(name, 0.0) for name in names])
        matrix = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
            raise ValueError("resource quantities must be finite and nonnegative")
        return matrix

    def check_structure(self):
        checked = {}
        for colony in self.config:
            owned = [
                part
                for part in self.parts.values()
                if part["colony"] == colony["id"]
            ]
            resources = resource_matrix(self, owned)
            total = np.zeros(len(self.web.chemistry.pools))
            for row in resources:
                total += row
            if not np.allclose(
                total,
                self.web.pools[colony["structure_row"]],
                rtol=1e-11,
                atol=1e-12,
            ):
                raise ValueError(
                    "physical structures and allocated chemical tissue disagree"
                )
            checked[colony["id"]] = (owned, resources)
        self._research_tissue_checked = checked

    def distribute_turnover(self, ledger):
        chemistry = self.web.chemistry
        for colony in self.config:
            owned, before = self._research_tissue_checked[colony["id"]]
            if not owned:
                continue
            after = before.copy()
            for reaction, extent in enumerate(
                ledger["extent"][colony["structure_row"]]
            ):
                if extent <= 0:
                    continue
                stoich = chemistry._arrays[0][reaction]
                consumed = np.flatnonzero(stoich < 0)
                if len(consumed) != 1:
                    raise RuntimeError("structure turnover must consume one pool")
                substrate = int(consumed[0])
                total = before[:, substrate].sum()
                if total <= 0:
                    raise RuntimeError("turnover consumed absent physical tissue")
                after += (
                    extent * before[:, substrate] / total
                )[:, None] * stoich[None, :]
            if np.any(after < -1e-12):
                raise RuntimeError("turnover over-consumed physical tissue")
            after = np.maximum(after, 0.0)
            for part, resources in zip(owned, after, strict=True):
                part["resources"] = dict(
                    zip(chemistry.pools, resources.tolist(), strict=True)
                )

    biosphere._check_structure = types.MethodType(check_structure, biosphere)
    biosphere._distribute_turnover = types.MethodType(
        distribute_turnover, biosphere
    )


def _profile_branch(
    state: dict[str, Any], *, reference: bool, steps: int, dt: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    world_class = physical_world_type(
        state["body_mode"], state["physics_backend"]
    )
    world = world_class.restore(state["world"])
    biosphere = Biosphere.restore(world, state["biosphere"])
    if reference:
        _install_research_reference(biosphere)

    profile = cProfile.Profile()
    physics_seconds = 0.0
    biosphere_seconds = 0.0
    profile.enable()
    for _ in range(steps):
        started = time.perf_counter()
        world.advance({}, dt)
        physics_seconds += time.perf_counter() - started
        started = time.perf_counter()
        biosphere.advance(dt)
        biosphere_seconds += time.perf_counter() - started
    profile.disable()

    stats = pstats.Stats(profile)
    selected: dict[str, float] = {}
    aliases = {
        "check_structure": "_check_structure",
        "_check_structure": "_check_structure",
        "distribute_turnover": "_distribute_turnover",
        "_distribute_turnover": "_distribute_turnover",
        "_develop": "_develop",
        "_part_resource_matrix": "_part_resource_matrix",
    }
    for (_, _, function_name), values in stats.stats.items():
        key = aliases.get(function_name)
        if key is not None:
            selected[key] = selected.get(key, 0.0) + float(values[3])
    return {
        "physics_seconds": physics_seconds,
        "biosphere_seconds": biosphere_seconds,
        "function_calls": stats.total_calls,
        "components": selected,
        "parts": len(biosphere.parts),
        "model_revision": world.model_revision,
    }, {"world": world.snapshot(), "biosphere": biosphere.snapshot()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="offline copy of a Habitat3D envelope",
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or not np.isfinite(args.dt) or not 0 < args.dt <= 1:
        raise ValueError("profile parameters are outside their bounds")

    envelope = json.loads(args.checkpoint.read_text())
    state = envelope["state"]
    if not all(key in state for key in ("world", "biosphere", "body_mode")):
        raise ValueError("checkpoint does not contain a world and Biosphere")
    reference, reference_state = _profile_branch(
        state, reference=True, steps=args.steps, dt=args.dt
    )
    optimized, optimized_state = _profile_branch(
        state, reference=False, steps=args.steps, dt=args.dt
    )
    exact = reference_state == optimized_state
    if not exact:
        raise RuntimeError("research-reference and optimized snapshots differ")

    report = {
        "format": "chreatures-grown-biosphere-profile-v1",
        "scope": (
            "Offline physical world plus Biosphere under cProfile; no Habitat3D, "
            "neural state, rendering, or live service."
        ),
        "research_reference": (
            "One disposable branch emulates the prior scans and unconditional "
            "deep copy. It is not production-selectable."
        ),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "tick": state.get("tick"),
            "initial_parts": len(state["biosphere"]["parts"]),
            "model_revision": state["world"].get("model_revision", 0),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "parameters": {"steps": args.steps, "dt_seconds": args.dt},
        "reference": reference,
        "optimized": optimized,
        "full_world_and_biosphere_snapshots_exact": exact,
        "interpretation": {
            "whole_runtime_speedup_claim": False,
            "reason": (
                "The measurement excludes neural inference, fields, rendering, "
                "server work, and other Habitat3D phases."
            ),
        },
        "sources": {
            path: _sha256(ROOT / path)
            for path in (
                "chreatures/biosphere.py",
                "chreatures/physical_batch.py",
                "chreatures/material_objects.py",
                "chreatures/ecological_exchange.py",
                "scripts/probe_grown_biosphere.py",
            )
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
