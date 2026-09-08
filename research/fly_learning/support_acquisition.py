"""Supplied support references and short CNS probes in fresh physical lives.

This is offline research orchestration. Contact/pose feedback reaches only the
supplied teacher and targets; every CNS probe traverses the complete CNS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

import numpy as np

from .batch_collection import collect_campaign
from .curriculum import Bout, Plan, CONTROL_INDEX, PHASE_INDEX, RESIDENTS, TICKS, split_for_world
from .data import load_episode, sha256_file
from .recovery import build_recovery_plan, create_native_bundle
from .teacher import FlyCurriculumTeacher, TeacherCommand, TeacherObservation

FORMAT: Final = "chreatures-fly-support-acquisition-corpus-v1"
COLD_TICKS: Final = 40
PROBE_TICKS: Final = 8
CORRECTION_TICKS: Final = 16
ACQUISITION_TICKS: Final = 16
CYCLES: Final = 24
SERVO_SLEW_RAD: Final = 0.04
AUTHOR_HZ: Final = 1.5
AUTHOR_MAGNITUDE: Final = 0.25
TASKS: Final = (
    "posture-support", "forward-locomotion", "stopping", "turning",
    "antenna-orient-contact", "posture-support",
)


class SupportPlan(Plan):
    def metadata(self) -> dict:
        value = super().metadata()
        value.pop("plan_sha256")
        value.update({
            "support_acquisition_format": FORMAT,
            "cold_support_ticks": COLD_TICKS,
            "probe_ticks": PROBE_TICKS,
            "correction_ticks": CORRECTION_TICKS,
            "acquisition_ticks": ACQUISITION_TICKS,
            "cycles": CYCLES,
            "reset_between_interventions": False,
            "fresh_world_per_episode": True,
            "observer_teacher_only": True,
            "teacher_servo_slew_rad_per_tick": SERVO_SLEW_RAD,
            "teacher_author_hz": AUTHOR_HZ,
            "teacher_author_magnitude": AUTHOR_MAGNITUDE,
            "teacher_support_gate": {"upright_min": 0.65, "contact_feet_min": 3},
            "context_sha256": hashlib.sha256(self.context.astype("<f4").tobytes()).hexdigest(),
        })
        value["plan_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value


def build_support_plan(world_index: int, *, base_seed: int = 20260908) -> SupportPlan:
    # Same seeds and authored layout as the corresponding recovery condition;
    # construction always allocates a new complete world and a fresh CNS state.
    matched = build_recovery_plan(world_index, base_seed=base_seed)
    context = matched.context.copy()
    context[:COLD_TICKS] = 0
    rows = []
    for resident in range(RESIDENTS):
        bouts = []
        def add(start: int, stop: int, phase: str, role: str) -> None:
            control = (matched.bouts[resident][0].control_source if role == "short-probe"
                       else "offline-author-teacher")
            bouts.append(Bout(start, stop, phase, control, world_index, 0.25, role))
        add(0, COLD_TICKS, "posture-support", "cold-support")
        for cycle in range(CYCLES):
            start = COLD_TICKS + cycle * 40
            phase = TASKS[(cycle + world_index + resident) % len(TASKS)]
            add(start, start + 8, phase, "short-probe")
            add(start + 8, start + 24, "stopping", "neutral-correction")
            add(start + 24, start + 40, phase, "contact-acquisition")
        add(1000, TICKS, "posture-support", "tail-support")
        rows.append(tuple(bouts))
    return SupportPlan(world_index, matched.world_seed, matched.variation_seed,
                       TICKS, RESIDENTS, tuple(rows), context)


class SupportTeacher(FlyCurriculumTeacher):
    """Research-only low-slew joint references; no deployed actor or motor filter."""

    def command(self, bout: Bout, tick_in_bout: int, observation: TeacherObservation,
                rng: np.random.Generator, *, resident: int) -> TeacherCommand:
        if not 0 <= resident < RESIDENTS:
            raise ValueError("teacher resident index differs from B4")
        contact = (np.asarray(observation.foot_contact) > 0).astype(np.float32)
        if observation.active_joint_position.shape != (84,) or contact.shape != (6,):
            raise ValueError("support teacher requires actual joint/contact observations")
        target = self.neutral.copy()
        adhesion = contact.copy()
        role = bout.intended_outcome
        supported = observation.thorax_up >= 0.65 and int(contact.sum()) >= 3
        if role in ("contact-acquisition", "short-probe") and supported:
            if bout.phase in ("forward-locomotion", "turning"):
                turn = (0.2 if (bout.start // 40 + resident) % 2 else -0.2) if bout.phase == "turning" else 0.0
                phase = ((bout.start + tick_in_bout) * 0.01 * AUTHOR_HZ) % 1.0
                walking, author_adhesion = self._author_walk(phase, turn, AUTHOR_MAGNITUDE)
                target[self.groups["walking"]] = walking
                adhesion *= author_adhesion
            elif bout.phase == "antenna-orient-contact":
                relative = np.asarray(observation.antenna_target_local)
                yaw = float(np.clip(np.arctan2(relative[1], max(abs(relative[0]), 1e-4)), -0.2, 0.2))
                target[self.groups["head"][0]] += yaw * 0.3
                target[self.groups["pedicels"]] += np.asarray([yaw, 0, 0, yaw, 0, 0], np.float32)
        previous = self.last_delivered[resident, :84]
        span = np.where(previous < 0, self.neutral - self.low, self.high - self.neutral)
        previous_rad = self.neutral + previous * span
        # The bound is in actual joint radians, including the first correction
        # after an unmodified CNS probe. Asymmetric normalization comes last.
        position = previous_rad + np.clip(target - previous_rad, -SERVO_SLEW_RAD, SERVO_SLEW_RAD)
        position = np.clip(position, self.low, self.high)
        normalized = np.r_[self._normalized(position), adhesion, np.zeros(2, np.float32)].astype(np.float32)
        return TeacherCommand(normalized, np.r_[position, adhesion].astype(np.float32), True,
                              "supplied-contact-gated-low-slew-support-reference")


def create_support_bundle(arguments: Any, plan: SupportPlan, *, service_metadata: dict[str, Any]):
    bundle = create_native_bundle(arguments, plan, service_metadata=service_metadata)
    try:
        bundle.teacher = SupportTeacher(Path(arguments.body_schema), bundle.teacher.author_steps)
        metadata = dict(bundle.metadata)
        metadata.pop("recovery_format")
        metadata.pop("recovery_source_sha256")
        metadata.update({
            "support_acquisition_format": FORMAT,
            "support_acquisition_source_sha256": sha256_file(Path(__file__)),
            "teacher_source_sha256": sha256_file(Path(__file__).with_name("teacher.py")),
            "collection_base_seed": int(arguments.seed),
            "matched_recovery_world_seed": plan.world_seed,
            "fresh_physical_life": True,
            "within_episode_snapshot_restore": False,
            "supplied_teacher_is_deployed_policy": False,
        })
        metadata["collection_life_identity"] = hashlib.sha256(json.dumps({
            "condition": FORMAT, "source_revision": arguments.source_revision,
            "world_instance_identity": bundle.world.world_instance_identity(),
        }, sort_keys=True).encode()).hexdigest()
        bundle.metadata = metadata
        return bundle
    except BaseException:
        bundle.world.close()
        raise


def seal_support(directory: Path, output: Path) -> dict:
    directory, output = directory.resolve(), output.resolve()
    if output.parent != directory or output.exists():
        raise ValueError("support manifest must be a new file beside the episodes")
    rows, identities = [], []
    for index in range(12):
        path = directory / f"episode-{index:02d}.npz"
        episode = load_episode(path)
        meta = episode.metadata
        plan = build_support_plan(index, base_seed=int(meta["collection_base_seed"]))
        if (meta.get("support_acquisition_format") != FORMAT
                or meta["curriculum"] != plan.metadata()
                or not meta.get("fresh_physical_life")
                or meta.get("within_episode_snapshot_restore") is not False):
            raise ValueError("support chronology/identity contract differs")
        for resident, bouts in enumerate(plan.bouts):
            for bout in bouts:
                sl = np.s_[bout.start:bout.stop, resident]
                teacher = bout.control_source == "offline-author-teacher"
                expected = episode.teacher_motor[sl] if teacher else episode.cns_motor[sl]
                if (not np.array_equal(episode.delivered_motor[sl], expected)
                        or not np.all(episode.control_source[sl] == CONTROL_INDEX[bout.control_source])
                        or not np.all(episode.curriculum_phase[sl] == PHASE_INDEX[bout.phase])):
                    raise ValueError("support delivered commands differ from recorded source")
        if not np.array_equal(episode.delivered_context, plan.context):
            raise ValueError("support CNS context chronology differs")
        teacher_rows = episode.control_source[1:] == CONTROL_INDEX["offline-author-teacher"]
        physical_slew = np.abs(np.diff(episode.applied_body_control[:, :, :84], axis=0))
        if np.any(physical_slew[teacher_rows] > SERVO_SLEW_RAD + 1e-5):
            raise ValueError("supplied teacher exceeded physical-radian servo slew bound")
        if np.any(episode.delivered_motor[:COLD_TICKS, :, :84]):
            raise ValueError("cold support did not deliver exact authored neutral servos")
        identities.append(meta)
        rows.append({"world_index": index, "split": split_for_world(index), "file": path.name,
                     "sha256": episode.sha256, "scene_layout_identity": meta["scene_layout_identity"],
                     "initial_snapshot_sha256": meta["initial_snapshot_sha256"],
                     "collection_life_identity": meta["collection_life_identity"]})
    for name in ("scene_layout_identity", "world_instance_identity", "initial_snapshot_sha256", "collection_life_identity"):
        if len({meta[name] for meta in identities}) != 12:
            raise ValueError(f"support condition repeats {name}")
    shared_keys = (
        "source_revision", "collector_sha256", "support_acquisition_source_sha256",
        "teacher_source_sha256", "collection_base_seed", "cns_format",
        "cns_service_sha256", "cns_adapter_sha256", "body_schema_sha256",
        "cns_body807_schema_sha256", "physical_sensory_schema_sha256",
        "morphology_sha256", "morphology_source_revision", "motor_atlas_sha256",
        "motor_calibration_sha256", "retina_mapping_sha256", "execution_backend",
        "native_runtime_sha256", "native_host_binary_sha256",
        "native_host_source_manifest_sha256", "mujoco_library_sha256", "native_world_engine",
        "author_trajectory_bank_sha256", "author_trajectory_manifest_sha256",
    )
    shared = {key: identities[0][key] for key in shared_keys}
    if any(meta[key] != value for meta in identities for key, value in shared.items()):
        raise ValueError("support corpus mixes CNS/source/schema/native deployment identities")
    result = {"format": FORMAT, "completed": True, "worlds": 12, "ticks": TICKS,
              "residents": RESIDENTS, "episodes": rows, "cold_support_ticks": COLD_TICKS,
              "probe_ticks": PROBE_TICKS, "reset_between_interventions": False,
              "observer_teacher_only": True, "source_service_sha256": identities[0]["cns_service_sha256"],
              "shared_identity": shared,
              "claim_limit": "Supplied support references and raw CNS probes; measured viability is not assumed."}
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(output)
    return {**result, "path": str(output), "sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--world-start", type=int, required=True)
    width = collect.add_mutually_exclusive_group()
    width.add_argument("--cohort-width", type=int, choices=(2, 4), default=2)
    width.add_argument("--cohort-widths", type=int, choices=(2, 4), nargs="+")
    for name in ("output", "scenes", "service", "native-binary", "native-manifest", "author-source", "body-schema", "motor-atlas"):
        collect.add_argument("--" + name, type=Path, required=True)
    collect.add_argument("--source-revision", required=True)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--seed", type=int, default=20260908)
    seal = commands.add_parser("seal")
    seal.add_argument("--directory", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        print(json.dumps(seal_support(args.directory, args.output), sort_keys=True)); return
    widths = args.cohort_widths or (args.cohort_width,)
    print(json.dumps(collect_campaign(args, build_support_plan, create_support_bundle, widths), sort_keys=True))


if __name__ == "__main__":
    main()
