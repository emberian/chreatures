"""Concurrent native worlds sharing one full-CNS GPU forward per control tick.

Episode decisions and tensors remain owned by collect_episode. This module only
coordinates research processes and the GPU batch; it defines no new policy,
physics, teacher, or outcome equation.
"""
from __future__ import annotations

import asyncio
import json
import os
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


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    if path.exists():
        raise FileExistsError(path)
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


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
    This function owns and closes the supplied worlds and private CNS state.
    Immutable weights remain resident after a sealed cohort. Each world
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
        "model_process_pid": os.getpid(),
        "cohort_index": shared_cns.sealed_cohorts,
        "private_state_initialization": "fresh service initial state for every world/resident column",
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

    shared_cns.begin_cohort()
    completed = False
    try:
        with ThreadPoolExecutor(max_workers=width, thread_name_prefix="native-fly-world") as pool:
            futures = [pool.submit(run, i) for i in range(width)]
            results = [future.result() for future in futures]
        if shared_cns.forward_calls != TICKS + 1 or any(p.calls != TICKS + 1 for p in proxies):
            raise CohortFailure("cohort did not include every control and terminal CNS observation")
        completed = True
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
        close_errors = []
        for bundle in bundles:
            try:
                bundle.world.close()
            except BaseException as error:
                close_errors.append(error)
        shared_cns.release_cohort(sealed=completed and not close_errors)
        if close_errors and sys.exc_info()[0] is None:
            raise CohortFailure("one or more research worlds failed to close") from close_errors[0]


def collect_campaign(arguments: Any, plan_factory, bundle_factory, widths: Sequence[int]) -> dict[str, Any]:
    """One model load, successive fresh physical cohorts, no failed-life reuse."""
    widths = tuple(widths)
    start = int(arguments.world_start)
    if not widths or any(width not in (2, 4) for width in widths) or start < 0 or start + sum(widths) > 12:
        raise ValueError("campaign widths must partition new worlds within 0..11")
    output = Path(arguments.output)
    expected = [output / f"episode-{index:02d}.npz" for index in range(start, start + sum(widths))]
    if any(path.exists() for path in expected):
        raise FileExistsError("campaign cannot overwrite or resume existing episode files")
    output.mkdir(parents=True, exist_ok=True)
    receipts = output / "cohorts"
    receipts.mkdir(exist_ok=True)
    if any(receipts.iterdir()) or (output / "campaign-receipt.json").exists() or (output / "campaign-abort.json").exists():
        raise FileExistsError("campaign requires new cohort and campaign receipts")
    began = time.monotonic()
    model_start = time.monotonic()
    shared = None
    expected_service_sha256 = None
    model_load_seconds = None
    phase = "model-load"
    results = []
    try:
        expected_service_sha256 = sha256_file(Path(arguments.service))
        shared = BatchedTorchFullCNS(arguments.service, arguments.device, widths[0])
        if shared.metadata["cns_service_sha256"] != expected_service_sha256:
            raise ValueError("service changed during model loading")
        model_load_seconds = time.monotonic() - model_start
        for width in widths:
            phase = "world-construction"
            shared.configure_cohort(width)
            plans = [plan_factory(index, base_seed=arguments.seed) for index in range(start, start + width)]
            paths = [output / f"episode-{plan.world_index:02d}.npz" for plan in plans]
            receipt_path = receipts / f"worlds{start:02d}-{start + width - 1:02d}.json"
            if receipt_path.exists():
                raise FileExistsError(receipt_path)
            bundles = []
            cohort_start = time.monotonic()
            try:
                for plan in plans:
                    bundles.append(bundle_factory(arguments, plan, service_metadata=dict(shared.metadata)))
                phase = "cohort-collection"
                result = collect_cohort(bundles, plans, paths, shared)
            except BaseException:
                # collect_cohort closes all members once it begins. A factory
                # or preflight failure still leaves constructed worlds here.
                if shared.cohort_status != "failed":
                    for bundle in bundles:
                        bundle.world.close()
                raise
            result["world_construction_through_close_seconds"] = time.monotonic() - cohort_start
            result["cns_private_state_status"] = shared.cohort_status
            phase = "cohort-receipt"
            _write_receipt(receipt_path, result)
            results.append({"receipt": str(receipt_path), "sha256": sha256_file(receipt_path), **result})
            print(json.dumps({"event": "cohort-sealed", "world_start": start, "width": width,
                              "receipt": str(receipt_path), "seconds": result["world_construction_through_close_seconds"]}), flush=True)
            start += width
        phase = "campaign-receipt"
        result = {"format": "chreatures-persistent-cns-collection-campaign-v1", "completed": True,
                "source_revision": arguments.source_revision, "model_process_pid": os.getpid(),
                "model_load_seconds": model_load_seconds, "elapsed_seconds": time.monotonic() - began,
                "cns_service_sha256": shared.metadata["cns_service_sha256"],
                "orchestrator_sha256": sha256_file(Path(__file__)),
                "native_deployment_sha256": sha256_file(arguments.native_manifest),
                "cohort_widths": list(widths), "cohorts": results}
        _write_receipt(output / "campaign-receipt.json", result)
        return result
    except BaseException as error:
        _write_receipt(output / "campaign-abort.json", {
            "format": "chreatures-persistent-cns-collection-abort-v1", "completed": False,
            "source_revision": arguments.source_revision, "model_process_pid": os.getpid(),
            "cns_service_sha256": expected_service_sha256,
            "cns_private_state_status": None if shared is None else shared.cohort_status,
            "phase": phase,
            "model_load_seconds": model_load_seconds if model_load_seconds is not None else time.monotonic() - model_start,
            "error_type": type(error).__name__, "active_world_start": start,
            "sealed_cohort_receipts": [{"receipt": row["receipt"], "sha256": row["sha256"]} for row in results],
            "elapsed_seconds": time.monotonic() - began,
            "mutation_retry_performed": False,
        })
        raise
    finally:
        if shared is not None:
            shared.close()
