#!/usr/bin/env python3
"""Collection orchestrator for actual B4 fly worlds and full CNS V4.

The concrete native-world host supplies the factory.  This file owns ordering,
information boundaries, episode tensors, and atomic sealing; it does not
implement physics or a substitute CNS.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import importlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

import numpy as np

from .curriculum import (
    CONTROL_INDEX,
    PHASE_INDEX,
    RESIDENTS,
    TICKS,
    Plan,
    build_plan,
    split_for_world,
)
from .data import (
    BODY_AFFERENTS,
    BODY_CONTROL,
    EPISODE_FORMAT,
    FEET,
    JOINTS,
    LATENT,
    MOTOR,
    OPTIC_SITES,
    OUTCOME_NAMES,
    OUTCOMES,
    SEGMENTS,
    SITES,
    load_episode,
    sha256_file,
)
from .teacher import FlyCurriculumTeacher, TeacherObservation


@dataclass(frozen=True)
class WorldSample:
    optic_rgb: np.ndarray  # [B,1771,3], only this and BODY807 enter CNS
    body_afferents: np.ndarray  # [B,807]
    joint_position: np.ndarray  # [B,126], target only
    joint_velocity: np.ndarray  # [B,126], target only
    segment_pose: np.ndarray  # [B,69,7], target only
    ground_contact_raw: np.ndarray  # [B,6,16], target only
    sensory_site_position: np.ndarray  # [B,5,3], target only
    mouth_contact_raw: np.ndarray  # [B,4] local normal3 plus contact fraction, target only
    applied_body_control: np.ndarray  # [B,90], observer receipt only
    teacher_observation: tuple[TeacherObservation, ...]  # privileged, never archived as model ingress
    observer: Mapping[str, Any]  # target/evaluator values, never model ingress


class ActualFlyWorld(Protocol):
    def sample(self) -> WorldSample: ...
    def advance(self, normalized_motor92: np.ndarray, dt: float = 0.01) -> None: ...
    def initial_snapshot_sha256(self) -> str: ...
    def world_instance_identity(self) -> str: ...
    def scene_layout_identity(self) -> str: ...
    def close(self) -> None: ...


class FullCNS(Protocol):
    metadata: Mapping[str, Any]
    def step(
        self, optic_rgb: np.ndarray, body_afferents: np.ndarray, delivered_context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def close(self) -> None: ...


class OutcomeEvaluator(Protocol):
    def transition(
        self,
        before: WorldSample,
        after: WorldSample,
        delivered_motor: np.ndarray,
        phase: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return outcome[B,16], reward[B], success[B], failure[B]."""
        ...


class CollectionBundle(Protocol):
    world: ActualFlyWorld
    cns: FullCNS
    teacher: FlyCurriculumTeacher
    evaluator: OutcomeEvaluator
    metadata: Mapping[str, Any]


def _require(value: np.ndarray, shape: tuple[int, ...], dtype: np.dtype, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}{shape}, got {result.dtype}{result.shape}")
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_sample(sample: WorldSample) -> None:
    _require(sample.optic_rgb, (RESIDENTS, OPTIC_SITES, 3), np.dtype("<f4"), "optic_rgb")
    _require(sample.body_afferents, (RESIDENTS, BODY_AFFERENTS), np.dtype("<f4"), "body_afferents")
    _require(sample.joint_position, (RESIDENTS, JOINTS), np.dtype("<f4"), "joint_position")
    _require(sample.joint_velocity, (RESIDENTS, JOINTS), np.dtype("<f4"), "joint_velocity")
    _require(sample.segment_pose, (RESIDENTS, SEGMENTS, 7), np.dtype("<f4"), "segment_pose")
    _require(sample.ground_contact_raw, (RESIDENTS, FEET, 16), np.dtype("<f4"), "ground_contact_raw")
    _require(sample.sensory_site_position, (RESIDENTS, SITES, 3), np.dtype("<f4"), "sensory_site_position")
    _require(sample.mouth_contact_raw, (RESIDENTS, 4), np.dtype("<f4"), "mouth_contact_raw")
    _require(sample.applied_body_control, (RESIDENTS, BODY_CONTROL), np.dtype("<f4"), "applied_body_control")
    if len(sample.teacher_observation) != RESIDENTS:
        raise ValueError("teacher observer rows differ from B4")


def _allocate() -> dict[str, np.ndarray]:
    return {
        "optic_rgb": np.empty((TICKS + 1, RESIDENTS, OPTIC_SITES, 3), "<f4"),
        "body_afferents": np.empty((TICKS + 1, RESIDENTS, BODY_AFFERENTS), "<f4"),
        "delivered_context": np.empty((TICKS, RESIDENTS, 12), "<f4"),
        "collected_latent": np.empty((TICKS + 1, RESIDENTS, LATENT), "<f4"),
        "delivered_motor": np.empty((TICKS, RESIDENTS, MOTOR), "<f4"),
        "cns_motor": np.empty((TICKS, RESIDENTS, MOTOR), "<f4"),
        "teacher_motor": np.empty((TICKS, RESIDENTS, MOTOR), "<f4"),
        "applied_body_control": np.empty((TICKS, RESIDENTS, BODY_CONTROL), "<f4"),
        "teacher_valid": np.empty((TICKS, RESIDENTS), "|b1"),
        "joint_position": np.empty((TICKS + 1, RESIDENTS, JOINTS), "<f4"),
        "joint_velocity": np.empty((TICKS + 1, RESIDENTS, JOINTS), "<f4"),
        "segment_pose": np.empty((TICKS + 1, RESIDENTS, SEGMENTS, 7), "<f4"),
        "ground_contact_raw": np.empty((TICKS + 1, RESIDENTS, FEET, 16), "<f4"),
        "sensory_site_position": np.empty((TICKS + 1, RESIDENTS, SITES, 3), "<f4"),
        "mouth_contact_raw": np.empty((TICKS + 1, RESIDENTS, 4), "<f4"),
        "outcome": np.empty((TICKS, RESIDENTS, OUTCOMES), "<f4"),
        "reward": np.empty((TICKS, RESIDENTS), "<f4"),
        "success": np.empty((TICKS, RESIDENTS), "|b1"),
        "failure": np.empty((TICKS, RESIDENTS), "|b1"),
        "reset": np.zeros((TICKS + 1, RESIDENTS), "|b1"),
        "active": np.ones((TICKS, RESIDENTS), "|b1"),
        "terminal": np.zeros((TICKS, RESIDENTS), "|b1"),
        "control_source": np.empty((TICKS, RESIDENTS), "|u1"),
        "curriculum_phase": np.empty((TICKS, RESIDENTS), "|u1"),
    }


def _write_sample(arrays: dict[str, np.ndarray], tick: int, sample: WorldSample) -> None:
    for name in (
        "optic_rgb", "body_afferents", "joint_position",
        "joint_velocity", "segment_pose", "ground_contact_raw",
        "sensory_site_position", "mouth_contact_raw",
    ):
        arrays[name][tick] = getattr(sample, name)


async def collect_episode(bundle: CollectionBundle, plan: Plan, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    arrays = _allocate()
    arrays["reset"][0] = True
    arrays["terminal"][-1] = True
    rngs = [np.random.default_rng([plan.variation_seed, resident]) for resident in range(RESIDENTS)]
    bout_index = np.zeros(RESIDENTS, np.int64)
    sample = bundle.world.sample()
    _validate_sample(sample)
    _write_sample(arrays, 0, sample)
    current_context = plan.context[0]
    latent, cns_motor = bundle.cns.step(sample.optic_rgb, sample.body_afferents, current_context)
    arrays["collected_latent"][0] = _require(latent, (RESIDENTS, LATENT), np.dtype("<f4"), "CNS latent")
    began = time.monotonic()

    for tick in range(TICKS):
        contexts = plan.context[tick]
        if tick:
            latent, cns_motor = bundle.cns.step(sample.optic_rgb, sample.body_afferents, contexts)
            arrays["collected_latent"][tick] = _require(latent, (RESIDENTS, LATENT), np.dtype("<f4"), "CNS latent")
        cns_motor = _require(cns_motor, (RESIDENTS, MOTOR), np.dtype("<f4"), "CNS motor")
        teacher = np.empty((RESIDENTS, MOTOR), np.float32)
        teacher_valid = np.zeros(RESIDENTS, bool)
        phases: list[str] = []
        source = np.empty(RESIDENTS, np.uint8)
        phase_id = np.empty(RESIDENTS, np.uint8)
        for resident in range(RESIDENTS):
            bouts = plan.bouts[resident]
            while tick >= bouts[bout_index[resident]].stop:
                bout_index[resident] += 1
            bout = bouts[bout_index[resident]]
            command = bundle.teacher.command(
                bout, tick - bout.start, sample.teacher_observation[resident], rngs[resident]
            )
            teacher[resident] = command.normalized_motor
            teacher_valid[resident] = command.valid
            phases.append(bout.phase)
            source[resident] = CONTROL_INDEX[bout.control_source]
            phase_id[resident] = PHASE_INDEX[bout.phase]
        teacher_rows = source == CONTROL_INDEX["offline-author-teacher"]
        delivered = cns_motor.copy()
        delivered[teacher_rows] = teacher[teacher_rows]

        arrays["delivered_context"][tick] = contexts
        arrays["cns_motor"][tick] = cns_motor
        arrays["teacher_motor"][tick] = teacher
        arrays["teacher_valid"][tick] = teacher_valid
        arrays["delivered_motor"][tick] = delivered
        arrays["control_source"][tick] = source
        arrays["curriculum_phase"][tick] = phase_id
        before = sample
        # The native world owns the only normalized-M92 to physical-M90 plus
        # pump/salivary conversion.  The collector never duplicates actuation.
        advanced = bundle.world.advance(delivered, 0.01)
        if inspect.isawaitable(advanced):
            await advanced
        sample = bundle.world.sample()
        _validate_sample(sample)
        arrays["applied_body_control"][tick] = sample.applied_body_control
        _write_sample(arrays, tick + 1, sample)
        outcome, reward, success, failure = bundle.evaluator.transition(
            before, sample, delivered, tuple(phases)
        )
        arrays["outcome"][tick] = _require(outcome, (RESIDENTS, OUTCOMES), np.dtype("<f4"), "outcome")
        arrays["reward"][tick] = _require(reward, (RESIDENTS,), np.dtype("<f4"), "reward")
        arrays["success"][tick] = _require(success, (RESIDENTS,), np.dtype("|b1"), "success")
        arrays["failure"][tick] = _require(failure, (RESIDENTS,), np.dtype("|b1"), "failure")
        if (tick + 1) % 64 == 0:
            print(json.dumps({
                "event": "collection-progress", "world_index": plan.world_index,
                "completed_ticks": tick + 1, "ticks": TICKS,
                "elapsed_seconds": time.monotonic() - began,
                "success_ticks": int(arrays["success"][: tick + 1].sum()),
                "failure_ticks": int(arrays["failure"][: tick + 1].sum()),
            }, sort_keys=True), flush=True)

    # One final CNS observation binds the T+1 cache. It repeats the last
    # acknowledged context without causing another physical transition.
    latent, _ = bundle.cns.step(
        sample.optic_rgb, sample.body_afferents, plan.context[-1]
    )
    arrays["collected_latent"][-1] = _require(
        latent, (RESIDENTS, LATENT), np.dtype("<f4"), "terminal CNS latent"
    )
    plan_metadata = plan.metadata()
    metadata = {
        **dict(bundle.metadata),
        "format": EPISODE_FORMAT,
        "collector_sha256": sha256_file(Path(__file__).resolve()),
        "world_index": plan.world_index,
        "world_seed": plan.world_seed,
        "variation_seed": plan.variation_seed,
        "split": split_for_world(plan.world_index),
        "ticks": TICKS,
        "residents": RESIDENTS,
        "body_afferent_dim": BODY_AFFERENTS,
        "motor_dim": MOTOR,
        "outcome_dim": OUTCOMES,
        "outcome_names": list(OUTCOME_NAMES),
        "sensory_dim": 6120,
        "body_afferent_rows": 11798,
        "motor_rows": 815,
        "physics_dt_s": 0.0001,
        "control_dt_s": 0.01,
        "control_substeps": 100,
        "cns_dt_s": 0.005,
        "cns_substeps": 2,
        "terminal_context_semantics": "repeat-last-delivered",
        "initial_snapshot_sha256": bundle.world.initial_snapshot_sha256(),
        "world_instance_identity": bundle.world.world_instance_identity(),
        "scene_layout_identity": bundle.world.scene_layout_identity(),
        "curriculum_plan_sha256": plan_metadata["plan_sha256"],
        "curriculum": plan_metadata,
        "raw_geometry_controller_access": False,
        "model_ingress": ["optic_rgb", "body_afferents", "delivered_context"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output)
    episode = load_episode(output)
    return {
        "path": str(output.resolve()), "sha256": episode.sha256,
        "world_index": plan.world_index, "split": episode.metadata["split"],
        "success_ticks": int(episode.success.sum()),
        "failure_ticks": int(episode.failure.sum()),
    }


def _factory(specification: str, arguments: argparse.Namespace, plan: Plan) -> CollectionBundle:
    if ":" not in specification:
        raise ValueError("factory must be module:function")
    module_name, function_name = specification.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    return function(arguments, plan)


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True, help="native host adapter module:function")
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--source-revision", required=True)
    # Passed through to the concrete host factory and identity-bound there.
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("native/browser-world/runtime.mjs"))
    parser.add_argument(
        "--core-wasm", type=Path,
        default=Path("native/browser-world/pkg/chreatures_browser_world_bg.wasm"),
    )
    parser.add_argument(
        "--body-schema", type=Path,
        default=Path("native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/schema.json"),
    )
    parser.add_argument(
        "--motor-atlas", type=Path,
        default=Path("research/fly_embodiment/fly-body-neural-atlas-v1.npz"),
    )
    parser.add_argument("--node", default="node")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    plan = build_plan(arguments.world_index, base_seed=arguments.seed)
    bundle = _factory(arguments.factory, arguments, plan)
    try:
        print(json.dumps(await collect_episode(bundle, plan, arguments.output), sort_keys=True))
    finally:
        for closed in (bundle.cns.close(), bundle.world.close()):
            if inspect.isawaitable(closed):
                await closed


if __name__ == "__main__":
    asyncio.run(_main())
