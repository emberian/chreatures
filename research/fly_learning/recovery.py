"""On-policy CNS failure, physical correction, and release curriculum.

The child CNS always supplies actions in probe/release bouts.  The offline
author teacher takes over the same physical chronology only in corrective
bouts; no observer value is exposed to the CNS or private context policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np

from .batch_collection import collect_campaign
from .collect import CollectionBundle
from .curriculum import (
    Bout, CONTEXT, CONTROL_INDEX, PHASE_INDEX, Plan, RESIDENTS, TICKS,
    split_for_world,
)
from .data import load_episode, sha256_file
from .native_host import (
    ActualOutcomeEvaluator,
    NativeActualFlyWorld,
    SampledAuthorSteps,
)
from .teacher import FlyCurriculumTeacher


RECOVERY_FORMAT: Final = "chreatures-fly-on-policy-recovery-corpus-v1"
BOUT_TICKS: Final = 64

# Each teacher intervention follows CNS-driven physical state without a reset.
# The next release therefore tests the consequence of the correction within the
# same actual chronology.
_PHASES: Final = (
    ("stopping", "probe"),
    ("self-right-recovery", "teacher"),
    ("posture-support", "teacher"),
    ("stopping", "release"),
    ("forward-locomotion", "probe"),
    ("stopping", "teacher"),
    ("posture-support", "teacher"),
    ("forward-locomotion", "release"),
    ("turning", "probe"),
    ("self-right-recovery", "teacher"),
    ("posture-support", "teacher"),
    ("stopping", "release"),
    ("antenna-orient-contact", "probe"),
    ("antenna-orient-contact", "teacher"),
    ("mouth-reach-touch-withdraw", "teacher"),
    ("free-consequence", "release"),
)


class RecoveryPlan(Plan):
    def metadata(self) -> dict:
        value = super().metadata()
        value.pop("plan_sha256")
        value.update({
            "recovery_format": RECOVERY_FORMAT,
            "intervention_sequence": [list(row) for row in _PHASES],
            "same_state_teacher_takeover": True,
            "reset_between_probe_correction_release": False,
            "observer_teacher_only": True,
            "context_sha256": hashlib.sha256(
                np.ascontiguousarray(self.context, dtype="<f4").tobytes()
            ).hexdigest(),
        })
        value["plan_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value


def _context_schedule(variation_seed: int) -> np.ndarray:
    result = np.zeros((TICKS, RESIDENTS, CONTEXT), np.float32)
    rng = np.random.default_rng([variation_seed, 0x5245434F])
    ou = np.zeros(CONTEXT, np.float32)
    for tick in range(TICKS):
        # Resident 0 is the goal-free control.  Residents 1..3 receive bounded
        # exploratory DN interventions; every value still traverses the full CNS.
        ou = 0.94 * ou + rng.normal(0, 0.035, CONTEXT).astype(np.float32)
        result[tick, 1] = np.clip(ou, -0.35, 0.35)
        phase = (tick % BOUT_TICKS) / BOUT_TICKS
        envelope = np.sin(np.pi * phase)
        result[tick, 2] = 0.28 * envelope * np.where(np.arange(CONTEXT) % 2, -1, 1)
        sign = -1.0 if (tick // 16) % 2 else 1.0
        result[tick, 3] = sign * 0.2 * np.where(np.arange(CONTEXT) < 6, 1, -1)
    return result


def build_recovery_plan(world_index: int, *, base_seed: int = 20260908) -> RecoveryPlan:
    if not 0 <= world_index < 12:
        raise ValueError("recovery worlds use indices 0..11")
    sequence = np.random.SeedSequence([base_seed, world_index, 0x5245434F])
    world_seed, variation_seed = (
        int(value) for value in sequence.generate_state(2, dtype=np.uint32)
    )
    interventions = ("cns-zero-context", "cns-context-ou", "cns-context-pulse", "cns-context-reversal")
    resident_bouts = []
    for resident in range(RESIDENTS):
        rows = []
        for index, (phase, role) in enumerate(_PHASES):
            start, stop = index * BOUT_TICKS, (index + 1) * BOUT_TICKS
            rows.append(Bout(
                start=start,
                stop=stop,
                phase=phase,
                control_source=("offline-author-teacher" if role == "teacher" else interventions[resident]),
                terrain_variant=world_index,
                difficulty=0.4 + 0.1 * ((world_index + resident + index) % 4),
                intended_outcome=f"on-policy-{role}-within-shared-physical-chronology",
            ))
        resident_bouts.append(tuple(rows))
    return RecoveryPlan(
        world_index=world_index,
        world_seed=world_seed,
        variation_seed=variation_seed,
        ticks=TICKS,
        residents=RESIDENTS,
        bouts=tuple(resident_bouts),
        context=_context_schedule(variation_seed),
    )


def _variant_scene(root: Path, world_index: int) -> Path:
    manifest_path = root.resolve() / "nursery-layouts.json"
    manifest = json.loads(manifest_path.read_text())
    variants = manifest.get("variants")
    if not isinstance(variants, list) or len(variants) != 12:
        raise ValueError("recovery collection requires the twelve sealed nursery layouts")
    row = variants[world_index]
    directory = manifest_path.parent / str(row["directory"])
    scene = directory / "world.json"
    if sha256_file(scene) != row["world_fixture_sha256"]:
        raise ValueError("recovery world fixture checksum differs")
    return scene


def create_native_bundle(arguments: Any, plan: RecoveryPlan, *, service_metadata: dict[str, Any]) -> CollectionBundle:
    """Create one world/teacher/evaluator; cohort orchestration injects CNS."""
    scene = _variant_scene(Path(arguments.scenes), plan.world_index)
    scene_contract = json.loads(scene.read_text())
    author = SampledAuthorSteps(Path(arguments.author_source))
    teacher = FlyCurriculumTeacher(Path(arguments.body_schema), author)
    motor_atlas_sha = sha256_file(Path(arguments.motor_atlas).resolve())
    # The campaign passes metadata from the exact already-loaded shared model.
    # Do not reread a half-gigabyte service for every newly constructed world.
    service_sha = service_metadata["cns_service_sha256"]
    author_bank_sha = sha256_file(Path(arguments.author_source).resolve())
    author_manifest_sha = sha256_file(Path(arguments.author_source).with_name("manifest.json"))
    local = SimpleNamespace(**vars(arguments))
    local.scene = scene
    world = NativeActualFlyWorld(local, plan)
    try:
        fixture = world.ready["fixture"]
        # Native READY deliberately returns only the small runtime projection.
        # The full contract lives in the exact scene bytes READY authenticates.
        if sha256_file(scene) != world.ready["fixture_sha256"]:
            raise ValueError("native READY fixture identity differs from the scene bytes")
        ecology_ids = tuple(str(item["ecology_id"]) for item in fixture["bodies"])
        ready = world.ready
        deployment = world.deployment_identity
        metadata = {
            "source_revision": str(arguments.source_revision),
            "morphology_source_revision": scene_contract["source_revision"],
            "body_schema_sha256": scene_contract["body_schema_sha256"],
            "cns_body807_schema_sha256": scene_contract["sensory_schema_sha256"],
            "physical_sensory_schema_sha256": scene_contract["physical_sensory_schema_sha256"],
            "morphology_sha256": scene_contract["morphology_sha256"],
            "motor_atlas_sha256": motor_atlas_sha,
            "cns_service_sha256": service_sha,
            "cns_adapter_sha256": service_metadata["adapter_sha256"],
            "motor_calibration_sha256": service_metadata["motor_calibration_sha256"],
            "retina_mapping_sha256": service_metadata["atlas_sha256"],
            "scene_manifest_sha256": ready["fixture_sha256"],
            "scene_xml_sha256": ready["scene_xml_sha256"],
            "native_runtime_sha256": deployment["native_host_binary_sha256"],
            **deployment,
            "author_trajectory_bank_sha256": author_bank_sha,
            "author_trajectory_manifest_sha256": author_manifest_sha,
            "cns_format": service_metadata["format"],
            "native_world_engine": ready["engine"],
            "recovery_format": RECOVERY_FORMAT,
            "recovery_source_sha256": sha256_file(Path(__file__)),
            "observer_values": "teacher targets and physical labels only",
        }
        return SimpleNamespace(
            world=world, cns=None, teacher=teacher,
            evaluator=ActualOutcomeEvaluator(ecology_ids), metadata=metadata,
        )
    except BaseException:
        world.close()
        raise


def seal_recovery(directory: Path, output: Path) -> dict[str, Any]:
    directory, output = directory.resolve(), output.resolve()
    if output.parent != directory:
        raise ValueError("recovery manifest must be sealed beside its episodes")
    rows, episodes = [], []
    for index in range(12):
        path = directory / f"episode-{index:02d}.npz"
        episode = load_episode(path)
        if episode.metadata.get("recovery_format") != RECOVERY_FORMAT:
            raise ValueError("episode is not an on-policy recovery chronology")
        curriculum = episode.metadata["curriculum"]
        if (
            curriculum.get("intervention_sequence") != [list(item) for item in _PHASES]
            or curriculum.get("same_state_teacher_takeover") is not True
            or curriculum.get("reset_between_probe_correction_release") is not False
            or curriculum.get("observer_teacher_only") is not True
        ):
            raise ValueError("recovery curriculum provenance differs")
        for bout_index, (phase, role) in enumerate(_PHASES):
            start, stop = bout_index * BOUT_TICKS, (bout_index + 1) * BOUT_TICKS
            if not np.all(episode.curriculum_phase[start:stop] == PHASE_INDEX[phase]):
                raise ValueError("recovery phase chronology differs")
            teacher = role == "teacher"
            expected_sources = (
                np.full(RESIDENTS, CONTROL_INDEX["offline-author-teacher"], np.uint8)
                if teacher else np.asarray([
                    CONTROL_INDEX["cns-zero-context"], CONTROL_INDEX["cns-context-ou"],
                    CONTROL_INDEX["cns-context-pulse"], CONTROL_INDEX["cns-context-reversal"],
                ], np.uint8)
            )
            if not np.all(episode.control_source[start:stop] == expected_sources):
                raise ValueError("recovery control-source chronology differs")
            expected_motor = episode.teacher_motor[start:stop] if teacher else episode.cns_motor[start:stop]
            if not np.array_equal(episode.delivered_motor[start:stop], expected_motor):
                raise ValueError("recovery delivered motor does not match committed controller")
        if np.any(episode.delivered_context[:, 0]) or not all(
            np.any(episode.delivered_context[:, resident]) for resident in range(1, RESIDENTS)
        ):
            raise ValueError("recovery context intervention coverage differs")
        rows.append({
            "world_index": index, "split": split_for_world(index),
            "file": path.name, "sha256": episode.sha256,
            "scene_layout_identity": episode.metadata["scene_layout_identity"],
            "initial_snapshot_sha256": episode.metadata["initial_snapshot_sha256"],
        })
        episodes.append(episode)
    for key in ("scene_layout_identity", "initial_snapshot_sha256"):
        if len({episode.metadata[key] for episode in episodes}) != 12:
            raise ValueError(f"recovery {key} identities overlap")
    result = {
        "format": RECOVERY_FORMAT,
        "completed": True,
        "worlds": 12,
        "ticks": TICKS,
        "residents": RESIDENTS,
        "splits": {"train": list(range(8)), "validation-worlds": [8, 9], "heldout-worlds": [10, 11]},
        "episodes": rows,
        "source_service_sha256": episodes[0].metadata["cns_service_sha256"],
        "same_state_probe_correction_release": True,
        "raw_geometry_controller_access": False,
    }
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    return {**result, "path": str(output), "sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--world-start", type=int, required=True)
    width = collect.add_mutually_exclusive_group()
    width.add_argument("--cohort-width", type=int, choices=(2, 4), default=2)
    width.add_argument("--cohort-widths", type=int, choices=(2, 4), nargs="+")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--scenes", type=Path, required=True)
    collect.add_argument("--service", type=Path, required=True)
    collect.add_argument("--native-binary", type=Path, required=True)
    collect.add_argument("--native-manifest", type=Path, required=True)
    collect.add_argument("--author-source", type=Path, required=True)
    collect.add_argument("--body-schema", type=Path, required=True)
    collect.add_argument("--motor-atlas", type=Path, required=True)
    collect.add_argument("--source-revision", required=True)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--seed", type=int, default=20260908)
    seal = commands.add_parser("seal")
    seal.add_argument("--directory", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "seal":
        print(json.dumps(seal_recovery(arguments.directory, arguments.output), sort_keys=True))
        return
    widths = arguments.cohort_widths or (arguments.cohort_width,)
    print(json.dumps(collect_campaign(arguments, build_recovery_plan, create_native_bundle, widths), sort_keys=True))



if __name__ == "__main__":
    main()
