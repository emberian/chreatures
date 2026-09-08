"""Deterministic, counterbalanced curriculum plans for actual fly worlds.

This module schedules teacher and CNS-closed-loop collection. It never computes
muscle commands and is never imported by a deployed controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Final

import numpy as np


TICKS: Final = 1024
RESIDENTS: Final = 4
CONTEXT: Final = 12

PHASES: Final = (
    "safe-babble",
    "posture-support",
    "self-right-recovery",
    "forward-locomotion",
    "turning",
    "stopping",
    "terrain-transition",
    "slip-contact-recovery",
    "antenna-orient-contact",
    "mouth-reach-touch-withdraw",
    "chemical-gradient-forage",
    "gustatory-intake-pump-salivary",
    "light-acoustic-orient",
    "conspecific-antenna-contact",
    "movable-material-push",
    "free-consequence",
)
PHASE_INDEX: Final = {name: index for index, name in enumerate(PHASES)}

CONTROL_SOURCES: Final = (
    "offline-author-teacher",
    "cns-zero-context",
    "cns-context-ou",
    "cns-context-pulse",
    "cns-context-reversal",
)
CONTROL_INDEX: Final = {name: index for index, name in enumerate(CONTROL_SOURCES)}


@dataclass(frozen=True)
class Bout:
    start: int
    stop: int
    phase: str
    control_source: str
    terrain_variant: int
    difficulty: float
    intended_outcome: str


@dataclass(frozen=True)
class Plan:
    world_index: int
    world_seed: int
    variation_seed: int
    ticks: int
    residents: int
    bouts: tuple[tuple[Bout, ...], ...]
    context: np.ndarray

    def metadata(self) -> dict:
        payload = {
            "world_index": self.world_index,
            "world_seed": self.world_seed,
            "variation_seed": self.variation_seed,
            "ticks": self.ticks,
            "residents": self.residents,
            "bouts": [[asdict(bout) for bout in row] for row in self.bouts],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["plan_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload


def _smooth_context(rng: np.random.Generator, ticks: int, mode: str) -> np.ndarray:
    result = np.zeros((ticks, CONTEXT), np.float32)
    if mode == "cns-zero-context" or mode == "offline-author-teacher":
        return result
    state = np.zeros(CONTEXT, np.float64)
    target = rng.uniform(-0.55, 0.55, CONTEXT)
    for tick in range(ticks):
        if tick % 12 == 0:
            target = rng.uniform(-0.65, 0.65, CONTEXT)
        state += 0.12 * (target - state)
        result[tick] = state
    if mode == "cns-context-pulse":
        result[:] *= 0.2
        width = max(6, ticks // 4)
        start = (ticks - width) // 2
        result[start : start + width] += rng.choice((-1.0, 1.0), CONTEXT) * rng.uniform(0.25, 0.6, CONTEXT)
    elif mode == "cns-context-reversal":
        result[ticks // 2 :] *= -1
    return np.clip(result, -1, 1).astype(np.float32)


def build_plan(world_index: int, *, base_seed: int = 20260908) -> Plan:
    """Build one gapless plan with different orders for all four physical residents.

    Teacher and closed-loop bouts are interleaved so physical state, tone and
    task order cannot serve as a fixed label. Every task occurs under both
    successful and deliberately difficult initial conditions; success remains
    an observed outcome rather than a schedule assertion.
    """
    if not 0 <= world_index < 12:
        raise ValueError("the frozen whole-world split uses indices 0..11")
    seed_sequence = np.random.SeedSequence([base_seed, world_index])
    world_seed, variation_seed = (int(value) for value in seed_sequence.generate_state(2, dtype=np.uint32))
    resident_sequences = seed_sequence.spawn(RESIDENTS)
    bouts_by_resident: list[tuple[Bout, ...]] = []
    context = np.zeros((TICKS, RESIDENTS, CONTEXT), np.float32)
    # Sixteen equal bouts allow every resident a different Latin-like rotation
    # while retaining exact gapless boundaries and 0.64 s physical bouts.
    bout_ticks = TICKS // 16
    phase_base = list(PHASES)
    control_base = [
        "offline-author-teacher", "cns-context-ou", "offline-author-teacher", "cns-context-pulse",
        "offline-author-teacher", "cns-zero-context", "cns-context-reversal", "offline-author-teacher",
    ] * 2
    for resident, sequence in enumerate(resident_sequences):
        rng = np.random.default_rng(sequence)
        rotation = (world_index * 3 + resident * 5) % len(phase_base)
        phases = phase_base[rotation:] + phase_base[:rotation]
        controls = control_base[resident:] + control_base[:resident]
        row: list[Bout] = []
        for index, (phase, control) in enumerate(zip(phases, controls, strict=True)):
            start, stop = index * bout_ticks, (index + 1) * bout_ticks
            difficult = ((index + resident + world_index) % 3 == 0)
            row.append(Bout(
                start=start, stop=stop, phase=phase, control_source=control,
                terrain_variant=(world_index + index + resident) % 5,
                difficulty=0.75 if difficult else 0.35,
                intended_outcome="challenging-attempt" if difficult else "viable-attempt",
            ))
            context[start:stop, resident] = _smooth_context(rng, bout_ticks, control)
        bouts_by_resident.append(tuple(row))
    return Plan(
        world_index=world_index, world_seed=world_seed, variation_seed=variation_seed,
        ticks=TICKS, residents=RESIDENTS, bouts=tuple(bouts_by_resident), context=context,
    )


def split_for_world(world_index: int) -> str:
    if 0 <= world_index <= 7:
        return "train"
    if 8 <= world_index <= 9:
        return "validation-worlds"
    if 10 <= world_index <= 11:
        return "heldout-worlds"
    raise ValueError("world index must be 0..11")
