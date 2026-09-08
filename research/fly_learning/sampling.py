"""Balanced recurrent windows for the actual-fly developmental corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .curriculum import CONTROL_SOURCES, PHASES
from .data import Corpus, Episode, RECOVERY_FORMAT, SUPPORT_ACQUISITION_FORMAT


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
        if burn_in < 0 or optimize < 1:
            raise ValueError("recurrent windows require nonnegative burn-in and positive optimization lengths")
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


class ResetPrefixSampler:
    """Whole causal prefixes beginning at the actual cold CNS/world reset."""

    def __init__(
        self, corpus: Corpus, *, optimize: int = 40,
        seed: int = 20260908, split: str = "train",
    ) -> None:
        if optimize < 1:
            raise ValueError("reset prefix requires a positive optimization length")
        if split == "train":
            episodes = corpus.train
        elif split == "validation-worlds":
            episodes = corpus.validation
        else:
            raise ValueError("heldout worlds cannot construct an optimization sampler")
        self.episodes = episodes
        self.burn_in, self.optimize = 0, optimize
        self.rng = np.random.default_rng(seed)
        support_indices = [
            index for index, episode in enumerate(episodes)
            if episode.metadata.get("support_acquisition_format")
            == SUPPORT_ACQUISITION_FORMAT
        ]
        recovery_indices = [
            index for index, episode in enumerate(episodes)
            if episode.metadata.get("recovery_format")
            == RECOVERY_FORMAT
        ]
        episode_indices = support_indices or recovery_indices or list(range(len(episodes)))
        rows: list[Window] = []
        for episode_index in episode_indices:
            episode = episodes[episode_index]
            ticks, residents = episode.active.shape
            if optimize > ticks:
                continue
            for resident in range(residents):
                if not episode.reset[0, resident] or not episode.active[:optimize, resident].all():
                    continue
                scored = optimize // 2
                rows.append(Window(
                    episode_index, resident, 0, 0, optimize,
                    int(episode.curriculum_phase[scored, resident]),
                    int(episode.control_source[scored, resident]),
                    _outcome_class(episode, scored, resident),
                ))
        if not rows:
            raise ValueError("corpus contains no complete cold-reset prefixes")
        self.rows = tuple(rows)
        self.support_only = bool(support_indices)
        self.recovery_only = bool(recovery_indices) and not self.support_only

    def coverage(self) -> dict:
        return {
            "burn_in": 0,
            "optimize": self.optimize,
            "cold_reset_prefixes": len(self.rows),
            "support_acquisition_corpus_only": self.support_only,
            "recovery_corpus_only": self.recovery_only,
        }

    def sample(self, count: int) -> tuple[Window, ...]:
        if count < 1:
            raise ValueError("sample count must be positive")
        return tuple(
            self.rows[int(self.rng.integers(len(self.rows)))] for _ in range(count)
        )


class MixedCausalSampler:
    """Mix cold-start batches with ordinary history-conditioned batches."""

    def __init__(
        self, corpus: Corpus, *, burn_in: int, optimize: int,
        reset_optimize: int, reset_fraction: float,
        seed: int, split: str,
    ) -> None:
        if not 0 <= reset_fraction <= 1:
            raise ValueError("reset prefix fraction must lie in [0,1]")
        self.ordinary = BalancedWindowSampler(
            corpus, burn_in=burn_in, optimize=optimize, seed=seed, split=split,
        )
        self.reset = ResetPrefixSampler(
            corpus, optimize=reset_optimize, seed=seed + 104729, split=split,
        )
        self.episodes = self.ordinary.episodes
        self.reset_fraction = reset_fraction
        self.rng = np.random.default_rng(seed + 130363)

    def coverage(self) -> dict:
        return {
            "reset_prefix_fraction": self.reset_fraction,
            "ordinary": self.ordinary.coverage(),
            "reset": self.reset.coverage(),
        }

    def sample(self, count: int) -> tuple[Window, ...]:
        sampler = self.reset if self.rng.random() < self.reset_fraction else self.ordinary
        return sampler.sample(count)


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
