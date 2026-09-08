"""Concurrent native worlds sharing one full-CNS GPU forward per control tick.

Episode decisions and tensors remain owned by collect_episode. This module only
coordinates research processes and the GPU batch; it defines no new policy,
physics, teacher, or outcome equation.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import sys
from types import SimpleNamespace
from typing import Any, Sequence

from .collect import collect_episode
from .curriculum import RESIDENTS, TICKS
from .data import sha256_file
from .native_host import BatchedTorchFullCNS


class CohortFailure(RuntimeError):
    pass


class _Barrier:
    def __init__(self, cns: BatchedTorchFullCNS, timeout: float) -> None:
        self.cns = cns
        self.timeout = timeout
        self.condition = threading.Condition()
        self.requests: dict[int, Any] = {}
        self.results: dict[int, Any] = {}
        self.generation = 0
        self.error: BaseException | None = None
        self.inference_seconds = 0.0
        self.wait_seconds = [0.0] * cns.worlds

    def abort(self, error: BaseException) -> None:
        with self.condition:
            if self.error is None:
                self.error = error
            self.condition.notify_all()

    def step(self, slot: int, generation: int, optic, body, context):
        began = time.monotonic()
        with self.condition:
            if self.error is not None:
                raise CohortFailure("cohort previously aborted") from self.error
            if generation != self.generation or slot in self.requests:
                error = CohortFailure("CNS cohort tick/slot chronology differs")
                self.abort(error)
                raise error
            self.requests[slot] = (optic, body, context)
            if len(self.requests) == self.cns.worlds:
                try:
                    ordered = [self.requests[i] for i in range(self.cns.worlds)]
                    start = time.monotonic()
                    values = self.cns.step_worlds(*zip(*ordered))
                    self.inference_seconds += time.monotonic() - start
                    self.results = dict(enumerate(values))
                    self.requests.clear()
                    self.generation += 1
                    self.condition.notify_all()
                except BaseException as error:
                    self.abort(error)
                    raise
            else:
                deadline = began + self.timeout
                while self.generation == generation and self.error is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.abort(CohortFailure("CNS cohort barrier timed out"))
                        break
                    self.condition.wait(remaining)
            if self.error is not None:
                raise CohortFailure("cohort aborted; physical mutations are not retried") from self.error
            self.wait_seconds[slot] += time.monotonic() - began
            return self.results.pop(slot)


class _WorldCNS:
    def __init__(self, barrier: _Barrier, slot: int) -> None:
        self.barrier, self.slot = barrier, slot
        self.metadata = barrier.cns.metadata
        self.calls = 0

    def step(self, optic, body, context):
        result = self.barrier.step(self.slot, self.calls, optic, body, context)
        self.calls += 1
        return result

    def close(self) -> None:
        # A member cannot release the shared neural state of its peers.
        pass


def collect_cohort(
    bundles: Sequence[Any], plans: Sequence[Any], outputs: Sequence[Path],
    shared_cns: BatchedTorchFullCNS, *, barrier_timeout: float = 120.0,
) -> dict[str, Any]:
    """Collect W episodes with existing semantics and a single GPU model.

    Factories must supply world/teacher/evaluator/metadata and no per-world CNS.
    This function owns and closes the supplied worlds and shared CNS. Each world
    has one worker, so its physical requests remain serial while worlds advance
    concurrently. On an exception all neural barriers abort and no tick retries.
    """
    width = shared_cns.worlds
    if len(bundles) != width or len(plans) != width or len(outputs) != width:
        raise ValueError("cohort members, plans and outputs must have identical width")
    if shared_cns.forward_calls != 0 or shared_cns.state is not None:
        raise ValueError("cohort requires freshly initialized private CNS state")
    if barrier_timeout <= 0:
        raise ValueError("barrier timeout must be positive")
    outputs = [Path(path) for path in outputs]
    if len({path.resolve() for path in outputs}) != width or any(path.exists() for path in outputs):
        raise ValueError("cohort outputs must be distinct new files")
    if len({id(bundle.world) for bundle in bundles}) != width:
        raise ValueError("each cohort member must own a distinct physical world")
    if any(getattr(bundle, "cns", None) is not None for bundle in bundles):
        raise ValueError("factory must not allocate per-world CNS models")
    barrier = _Barrier(shared_cns, barrier_timeout)
    mapping = [
        {"slot": i, "world_index": plan.world_index,
         "world_instance_identity": bundle.world.world_instance_identity(),
         "resident_columns": list(range(i * RESIDENTS, (i + 1) * RESIDENTS))}
        for i, (bundle, plan) in enumerate(zip(bundles, plans))
    ]
    metadata = {
        "format": "chreatures-native-world-cns-cohort-v1",
        "width": width, "residents_per_world": RESIDENTS,
        "row_mapping": mapping, "control_dt_s": .01,
        "cns_model_forwards_per_tick": 1,
        "terminal_observation_included": True,
        "cns_service_sha256": shared_cns.metadata["cns_service_sha256"],
        "cns_adapter_sha256": shared_cns.metadata["adapter_sha256"],
        "orchestrator_sha256": sha256_file(Path(__file__)),
        "transport_sha256": sha256_file(Path(__file__).with_name("native_host.py")),
    }
    proxies = [_WorldCNS(barrier, i) for i in range(width)]
    began = time.monotonic()

    def run(slot: int):
        bundle = bundles[slot]
        adapted = SimpleNamespace(
            world=bundle.world, cns=proxies[slot], teacher=bundle.teacher,
            evaluator=bundle.evaluator,
            metadata={**dict(bundle.metadata), "cns_cohort": {**metadata, "slot": slot}},
        )
        try:
            return asyncio.run(collect_episode(adapted, plans[slot], outputs[slot]))
        except BaseException as error:
            barrier.abort(error)
            raise

    try:
        with ThreadPoolExecutor(max_workers=width, thread_name_prefix="native-fly-world") as pool:
            futures = [pool.submit(run, i) for i in range(width)]
            results = [future.result() for future in futures]
        if shared_cns.forward_calls != TICKS + 1 or any(p.calls != TICKS + 1 for p in proxies):
            raise CohortFailure("cohort did not include every control and terminal CNS observation")
        return {
            **metadata, "completed": True, "episodes": results,
            "cns_forward_calls": shared_cns.forward_calls,
            "elapsed_seconds": time.monotonic() - began,
            "cns_inference_seconds": barrier.inference_seconds,
            "member_barrier_seconds": barrier.wait_seconds,
            "world_transport": [getattr(bundle.world, "transport_statistics", None)
                                for bundle in bundles],
        }
    finally:
        shared_cns.close()
        close_errors = []
        for bundle in bundles:
            try:
                bundle.world.close()
            except BaseException as error:
                close_errors.append(error)
        if close_errors and sys.exc_info()[0] is None:
            raise CohortFailure("one or more research worlds failed to close") from close_errors[0]
