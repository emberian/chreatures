"""Long supplied author continuations before rare CNS probes in fresh worlds.

This is an offline research curriculum. Actual physical observations are used by
its supplied teacher only; the full CNS still receives optic, BODY807 and zero
context on every tick. Failed physical lives continue without resets.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from .curriculum import Bout, Plan, RESIDENTS, TICKS
from .recovery import build_recovery_plan
from .teacher import FlyCurriculumTeacher, TeacherCommand, TeacherObservation

FORMAT: Final = "chreatures-fly-supported-continuation-corpus-v1"
COLD_TICKS: Final = 40
BOUT_TICKS: Final = 160
MODES: Final = ("continuation", "stop", "left", "stop", "right", "antenna")
AMPLITUDES: Final = (.15, .30, .45)
PHASES: Final = {"continuation": "forward-locomotion", "stop": "stopping",
                "left": "turning", "right": "turning", "antenna": "antenna-orient-contact"}


class ContinuationPlan(Plan):
    def metadata(self) -> dict:
        value = super().metadata()
        value.pop("plan_sha256")
        value.update({
            "supported_continuation_format": FORMAT,
            "cold_support_ticks": COLD_TICKS,
            "author_bout_ticks": BOUT_TICKS, "author_bouts": 6,
            "mode_order": list(MODES), "mode_rotation": "(world_index + resident) modulo 6",
            "requested_amplitudes": list(AMPLITUDES),
            "amplitude_assignment": "walking modes: (world_index + 2*resident + bout_index//2) modulo 3; stop/antenna retain the previous request",
            "amplitude_ramp_ticks": 64,
            "probe_start": 1000, "probe_ticks": 8, "post_probe_correction_ticks": 16,
            "probe_assignment": "(world_index + resident) even: probe; odd: teacher control",
            "teacher_servo_slew_rad_per_tick": .04,
            "teacher_adhesion_slew_per_tick": .25,
            "teacher_author_hz": 1.5,
            "teacher_support_gate": {"upright_min": .65, "contact_feet_min": 3},
            "reset_between_interventions": False, "fresh_world_per_episode": True,
            "observer_teacher_only": True, "context": "exact zero12 throughout",
            "context_sha256": hashlib.sha256(self.context.astype("<f4").tobytes()).hexdigest(),
            "claim_limit": "Supplied author motor/body/neural histories, including actual falls. Frozen antenna geometry; no touch or private-context competence claim.",
        })
        value["plan_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return value


def build_continuation_plan(world_index: int, *, base_seed: int = 20260908) -> ContinuationPlan:
    matched = build_recovery_plan(world_index, base_seed=base_seed)
    rows = []
    for resident in range(RESIDENTS):
        bouts = [Bout(0, 40, "posture-support", "offline-author-teacher", world_index, 0., "cold-support")]
        for index in range(6):
            mode = MODES[(index + world_index + resident) % 6]
            amplitude = (AMPLITUDES[(world_index + 2 * resident + index // 2) % 3]
                         if mode in ("continuation", "left", "right") else 0.)
            start = 40 + index * 160
            bouts.append(Bout(start, start + 160, PHASES[mode], "offline-author-teacher",
                              world_index, amplitude, mode))
        if (world_index + resident) % 2 == 0:
            bouts.extend((Bout(1000, 1008, "free-consequence", "cns-zero-context", world_index, 0., "tail-probe"),
                          Bout(1008, 1024, "stopping", "offline-author-teacher", world_index, 0., "post-probe-correction")))
        else:
            bouts.append(Bout(1000, 1024, "stopping", "offline-author-teacher", world_index, 0., "tail-teacher-control"))
        rows.append(tuple(bouts))
    return ContinuationPlan(world_index, matched.world_seed, matched.variation_seed,
                            TICKS, RESIDENTS, tuple(rows), np.zeros((TICKS, RESIDENTS, 12), np.float32))


def _author_module():
    import importlib.util
    from pathlib import Path
    import sys
    name = "chreatures_offline_supported_author"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "native/fly-body/tools/replay_author_teacher.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("offline author continuation source is unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class ContinuationTeacher(FlyCurriculumTeacher):
    """One state per resident, wrapping the supplied author's existing equations."""

    def __init__(self, schema_path: Path, author_steps):
        super().__init__(schema_path, author_steps)
        if author_steps is None:
            raise ValueError("supported continuation requires the sealed author bank")
        cls = _author_module().SupportedAuthorContinuation
        self.continuations = [cls(author_steps.joint, author_steps.adhesion, author_steps.neutral,
                                  self.neutral, np.stack((self.low, self.high), axis=1),
                                  frequency_hz=1.5, servo_slew_rad_per_tick=.04,
                                  adhesion_slew_per_tick=.25) for _ in range(RESIDENTS)]
        self.fitted = [False] * RESIDENTS
        self.execution = {"residents": [{"fit_events": [], "supported_teacher_ticks": 0,
                                          "unsupported_correction_ticks": 0,
                                          "requested_mode_ticks": {}, "amplitude_min": None,
                                          "amplitude_max": None} for _ in range(RESIDENTS)]}

    def command(self, bout: Bout, tick_in_bout: int, observation: TeacherObservation,
                rng: np.random.Generator, *, resident: int) -> TeacherCommand:
        if not 0 <= resident < RESIDENTS:
            raise ValueError("supported continuation requires a B4 resident")
        helper = self.continuations[resident]
        previous = self.last_delivered[resident]
        servo = self.neutral + np.where(previous[:84] < 0, previous[:84] * (self.neutral - self.low),
                                       previous[:84] * (self.high - self.neutral))
        contact = np.asarray(observation.foot_contact) > 0
        supported = observation.thorax_up >= .65 and int(contact.sum()) >= 3
        delivered_teacher = bout.control_source == "offline-author-teacher"
        mode = bout.intended_outcome
        diagnostic = self.execution["residents"][resident]
        if delivered_teacher:
            counts = diagnostic["requested_mode_ticks"]
            counts[mode] = counts.get(mode, 0) + 1
        if not supported:
            self.fitted[resident] = False
        result = None
        if mode in MODES and supported:
            if not self.fitted[resident] and delivered_teacher:
                fit = helper.fit(observation.active_joint_position, servo, contact, observation.thorax_up)
                diagnostic["fit_events"].append({"tick": bout.start + tick_in_bout, **asdict(fit)})
                self.fitted[resident] = bool(fit.eligible)
            if self.fitted[resident]:
                nonwalking = None
                if mode == "antenna":
                    target = super().command(bout, tick_in_bout, observation, rng, resident=resident)
                    nonwalking = target.physical_control[:84].copy()
                    nonwalking[:42] = self.neutral[:42]
                result = helper.command(observation.active_joint_position, servo, previous[84:90],
                                        contact, observation.thorax_up, control_dt_s=.01, mode=mode,
                                        requested_amplitude=bout.difficulty if mode in ("continuation", "left", "right") else None,
                                        nonwalking_target84_rad=nonwalking, advance_state=delivered_teacher)
        if result is None:
            result = helper.neutral_command(servo, previous[84:90], contact, advance_state=delivered_teacher)
            if delivered_teacher and not supported:
                diagnostic["unsupported_correction_ticks"] += 1
        elif delivered_teacher:
            diagnostic["supported_teacher_ticks"] += 1
            amplitude = float(result.amplitude)
            diagnostic["amplitude_min"] = amplitude if diagnostic["amplitude_min"] is None else min(diagnostic["amplitude_min"], amplitude)
            diagnostic["amplitude_max"] = amplitude if diagnostic["amplitude_max"] is None else max(diagnostic["amplitude_max"], amplitude)
        return TeacherCommand(np.asarray(result.normalized_motor92, np.float32),
                              np.r_[result.servo_targets_rad, result.adhesion].astype(np.float32),
                              True, "supplied-supported-author-or-bounded-neutral-correction")


def create_continuation_bundle(arguments: Any, plan: ContinuationPlan, *, service_metadata: dict[str, Any]):
    from .recovery import create_native_bundle
    from .data import sha256_file
    bundle = create_native_bundle(arguments, plan, service_metadata=service_metadata)
    try:
        bundle.teacher = ContinuationTeacher(Path(arguments.body_schema), bundle.teacher.author_steps)
        metadata = dict(bundle.metadata)
        metadata.pop("recovery_format"); metadata.pop("recovery_source_sha256")
        metadata.update({
            "supported_continuation_format": FORMAT,
            "supported_continuation_source_sha256": sha256_file(Path(__file__)),
            "teacher_source_sha256": sha256_file(Path(__file__).with_name("teacher.py")),
            "author_continuation_source_sha256": sha256_file(Path(_author_module().__file__)),
            "collection_base_seed": int(arguments.seed),
            "fresh_physical_life": True, "within_episode_snapshot_restore": False,
            "supplied_teacher_is_deployed_policy": False,
            "teacher_continuation_execution": bundle.teacher.execution,
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


SHARED_KEYS = (
    "source_revision", "collector_sha256", "supported_continuation_source_sha256",
    "teacher_source_sha256", "author_continuation_source_sha256", "collection_base_seed",
    "cns_format", "cns_service_sha256", "cns_adapter_sha256", "body_schema_sha256",
    "cns_body807_schema_sha256", "physical_sensory_schema_sha256", "morphology_sha256",
    "morphology_source_revision", "motor_atlas_sha256", "motor_calibration_sha256",
    "retina_mapping_sha256", "execution_backend", "native_runtime_sha256",
    "native_host_binary_sha256", "native_host_source_manifest_sha256", "mujoco_library_sha256",
    "native_world_engine", "author_trajectory_bank_sha256", "author_trajectory_manifest_sha256",
)


def validate_episode(episode, index: int) -> None:
    from .curriculum import CONTROL_INDEX, PHASE_INDEX
    meta = episode.metadata
    plan = build_continuation_plan(index, base_seed=int(meta["collection_base_seed"]))
    if (meta.get("supported_continuation_format") != FORMAT or meta["world_index"] != index
            or meta["curriculum"] != plan.metadata() or meta.get("fresh_physical_life") is not True
            or meta.get("within_episode_snapshot_restore") is not False):
        raise ValueError("supported continuation chronology/identity differs")
    if np.any(episode.delivered_context) or np.any(episode.delivered_motor[:40, :, :84]):
        raise ValueError("supported continuation requires zero context and exact cold neutral servos")
    for resident, bouts in enumerate(plan.bouts):
        for bout in bouts:
            region = np.s_[bout.start:bout.stop, resident]
            teacher = bout.control_source == "offline-author-teacher"
            expected = episode.teacher_motor[region] if teacher else episode.cns_motor[region]
            if (not np.array_equal(episode.delivered_motor[region], expected)
                    or not np.all(episode.control_source[region] == CONTROL_INDEX[bout.control_source])
                    or not np.all(episode.curriculum_phase[region] == PHASE_INDEX[bout.phase])):
                raise ValueError("supported continuation control source differs from delivered chronology")
    teacher_rows = episode.control_source[1:] == CONTROL_INDEX["offline-author-teacher"]
    delta = np.abs(np.diff(episode.applied_body_control, axis=0))
    if np.any(delta[:, :, :84][teacher_rows] > .04001) or np.any(delta[:, :, 84:][teacher_rows] > .25001):
        raise ValueError("supported continuation exceeds physical-radian or adhesion slew")
    if len(meta.get("teacher_continuation_execution", {}).get("residents", [])) != RESIDENTS:
        raise ValueError("supported continuation lacks its actual teacher execution trace")


def seal_continuation(directory: Path, output: Path) -> dict:
    from .batch_collection import _write_receipt
    from .curriculum import split_for_world
    from .data import load_episode, sha256_file
    directory, output = directory.resolve(), output.resolve()
    if output.parent != directory or output.exists():
        raise ValueError("continuation manifest must be new and beside its episodes")
    rows = []; identities = []
    for index in range(12):
        path = directory / f"episode-{index:02d}.npz"
        episode = load_episode(path)
        validate_episode(episode, index)
        meta = episode.metadata
        identities.append(meta)
        rows.append({"world_index": index, "split": split_for_world(index), "file": path.name,
                     "sha256": episode.sha256, "scene_layout_identity": meta["scene_layout_identity"],
                     "initial_snapshot_sha256": meta["initial_snapshot_sha256"],
                     "collection_life_identity": meta["collection_life_identity"]})
    for key in ("scene_layout_identity", "world_instance_identity", "initial_snapshot_sha256", "collection_life_identity"):
        if len({meta[key] for meta in identities}) != 12:
            raise ValueError(f"continuation worlds repeat {key}")
    shared = {key: identities[0][key] for key in SHARED_KEYS}
    if any(meta[key] != value for meta in identities for key, value in shared.items()):
        raise ValueError("continuation corpus mixes source/CNS/native/schema identities")
    result = {"format": FORMAT, "completed": True, "worlds": 12, "ticks": TICKS,
              "residents": RESIDENTS, "episodes": rows, "shared_identity": shared,
              "source_service_sha256": shared["cns_service_sha256"],
              "cold_support_ticks": 40, "author_bout_ticks": 160, "author_bouts": 6,
              "probe_start": 1000, "probe_ticks": 8, "probe_residents_per_world": 2,
              "context": "exact zero12 throughout", "reset_between_interventions": False,
              "observer_teacher_only": True,
              "claim_limit": "Actual supplied-author/CNS/body histories with retained failures. No private-context, autonomous locomotor, or antenna-touch competence claim."}
    _write_receipt(output, result)
    return {**result, "path": str(output), "sha256": sha256_file(output)}


def main():
    import argparse
    from .batch_collection import collect_campaign
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--world-start", type=int, required=True)
    widths = collect.add_mutually_exclusive_group()
    widths.add_argument("--cohort-width", type=int, choices=(2, 4), default=2)
    widths.add_argument("--cohort-widths", type=int, choices=(2, 4), nargs="+")
    for name in ("output", "scenes", "service", "native-binary", "native-manifest", "author-source", "body-schema", "motor-atlas"):
        collect.add_argument("--" + name, type=Path, required=True)
    collect.add_argument("--source-revision", required=True)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--seed", type=int, default=20260908)
    seal = commands.add_parser("seal")
    seal.add_argument("--directory", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "seal":
        print(json.dumps(seal_continuation(args.directory, args.output), sort_keys=True))
        return
    result = collect_campaign(args, build_continuation_plan, create_continuation_bundle,
                              args.cohort_widths or (args.cohort_width,))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
