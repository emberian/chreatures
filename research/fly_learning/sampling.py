"""Balanced recurrent windows for the actual-fly developmental corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .curriculum import CONTROL_SOURCES, PHASES
from .data import Corpus, Episode


@dataclass(frozen=True)
class Window:
    episode_index: int
    resident: int
    start: int
    burn_in: int
    optimize: int
    phase: int
    control_source: int
    outcome_class: int  # 0 neutral/in-progress, 1 success, 2 failed attempt

    @property
    def history_start(self) -> int:
        return self.start - self.burn_in

    @property
    def stop(self) -> int:
        return self.start + self.optimize


def _outcome_class(episode: Episode, tick: int, resident: int) -> int:
    if episode.success[tick, resident]:
        return 1
    if episode.failure[tick, resident]:
        return 2
    return 0


class BalancedWindowSampler:
    """Samples task/outcome/control strata without exposing labels to models."""

    def __init__(
        self, corpus: Corpus, *, burn_in: int = 40, optimize: int = 64,
        seed: int = 20260908, split: str = "train",
    ) -> None:
        if burn_in < 1 or optimize < 1:
            raise ValueError("recurrent windows require positive burn-in and optimization lengths")
        if split == "train":
            episodes = corpus.train
        elif split == "validation-worlds":
            episodes = corpus.validation
        else:
            raise ValueError("heldout worlds cannot construct an optimization sampler")
        self.episodes = episodes
        self.burn_in, self.optimize = burn_in, optimize
        self.rng = np.random.default_rng(seed)
        strata: dict[tuple[int, int, int], list[Window]] = {}
        for episode_index, episode in enumerate(episodes):
            ticks, residents = episode.active.shape
            for resident in range(residents):
                for start in range(burn_in, ticks - optimize + 1):
                    scored = start + optimize // 2
                    if not episode.active[start : start + optimize, resident].all():
                        continue
                    phase = int(episode.curriculum_phase[scored, resident])
                    control = int(episode.control_source[scored, resident])
                    outcome = _outcome_class(episode, scored, resident)
                    window = Window(
                        episode_index, resident, start, burn_in, optimize,
                        phase, control, outcome,
                    )
                    strata.setdefault((phase, control, outcome), []).append(window)
        if not strata:
            raise ValueError("corpus contains no complete recurrent windows")
        self.strata = {key: tuple(value) for key, value in sorted(strata.items())}
        self.keys = tuple(self.strata)

    def coverage(self) -> dict:
        return {
            "burn_in": self.burn_in, "optimize": self.optimize,
            "strata": [
                {
                    "phase": PHASES[key[0]], "control_source": CONTROL_SOURCES[key[1]],
                    "outcome": ("neutral", "success", "failure")[key[2]], "windows": len(rows),
                }
                for key, rows in self.strata.items()
            ],
        }

    def sample(self, count: int) -> tuple[Window, ...]:
        if count < 1:
            raise ValueError("sample count must be positive")
        result = []
        # First choose a stratum uniformly, then a chronology within it. This
        # prevents long neutral locomotion regions from erasing rare recovery
        # and mouth/antenna outcomes while preserving real failed attempts.
        for _ in range(count):
            key = self.keys[int(self.rng.integers(len(self.keys)))]
            rows = self.strata[key]
            result.append(rows[int(self.rng.integers(len(rows)))])
        return tuple(result)

    def batches(self, batch_size: int) -> Iterator[tuple[Window, ...]]:
        while True:
            yield self.sample(batch_size)


def slice_window(episode: Episode, window: Window) -> dict[str, np.ndarray]:
    """Return chronological arrays with target-only labels separately named."""
    h0, t1, resident = window.history_start, window.stop, window.resident
    # T+1 state streams cover the final future target; transition streams do not.
    return {
        "optic_rgb": episode.optic_rgb[h0 : t1 + 1, resident],
        "body_afferents": episode.body_afferents[h0 : t1 + 1, resident],
        "collected_latent": episode.collected_latent[h0 : t1 + 1, resident],
        "delivered_context": episode.delivered_context[h0:t1, resident],
        "cns_motor": episode.cns_motor[h0:t1, resident],
        "delivered_motor": episode.delivered_motor[h0:t1, resident],
        "target_applied_body_control": episode.applied_body_control[h0:t1, resident],
        "reset": episode.reset[h0 : t1 + 1, resident],
        "target_body": episode.body_afferents[h0 : t1 + 1, resident],
        "target_reward": episode.reward[h0:t1, resident],
        "target_success": episode.success[h0:t1, resident],
        "target_failure": episode.failure[h0:t1, resident],
        "target_outcome": episode.outcome[h0:t1, resident],
        "teacher_motor": episode.teacher_motor[h0:t1, resident],
        "teacher_valid": episode.teacher_valid[h0:t1, resident],
    }
