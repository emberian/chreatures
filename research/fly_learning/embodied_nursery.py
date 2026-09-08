#!/usr/bin/env python3
"""Opt-in multisensory nursery layouts, plans, and actual-world bundles.

This research entry point sits beside the frozen body-bootstrap collector.  It
never changes that collector's default plan, corpus format, or bridge process.
All stimuli enter through the physical screen or ``BrowserWorld.visitorSound``;
none are written into BODY807 or policy inputs directly.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Final, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from .collect import CollectionBundle, collect_episode
from .curriculum import Bout, CONTEXT, RESIDENTS, TICKS, split_for_world
from .data import MOTOR, sha256_file
from .native_host import (
    ActualOutcomeEvaluator,
    NodeActualFlyWorld,
    SampledAuthorSteps,
    TorchFullCNS,
    _hash_json,
)
from .teacher import FlyCurriculumTeacher


NURSERY_PLAN_FORMAT: Final = "chreatures-embodied-nursery-plan-v1"
NURSERY_LAYOUT_FORMAT: Final = "chreatures-embodied-nursery-layout-v1"
NURSERY_CORPUS_FORMAT: Final = "chreatures-embodied-nursery-corpus-v1"
BOUT_TICKS: Final = 64
TONE_DURATION_S: Final = BOUT_TICKS * 0.01
SCREEN_WIDTH: Final = 32
SCREEN_HEIGHT: Final = 32
DEFAULT_BASE_SCENE: Final = Path("native/fly-body/scenes/training-4/world.json")
DEFAULT_LAYOUT_ROOT: Final = (
    Path.home() / "paperbin/chreatures/research/embodied-nursery-20260908/layouts"
)

CAPABILITIES: Final = (
    "stopping",
    "forward-locomotion",
    "antenna-orient-contact",
    "mouth-reach-touch-withdraw",
)
TONE_HZ: Final = {
    "stopping": 83.651,
    "forward-locomotion": 136.798,
    "antenna-orient-contact": 223.711,
    "mouth-reach-touch-withdraw": 365.844,
}
SOUND_ANCHOR_MM: Final = {
    "stopping": (0.0, 3.0, 1.8),       # existing curved-bark platform
    "forward-locomotion": (2.2, 0.0, 0.8),  # existing central grain region
    "antenna-orient-contact": (-7.5, -5.5, 1.2),  # existing west leaf
    "mouth-reach-touch-withdraw": (7.5, 5.0, 1.0),  # existing east leaf
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _episode_trajectory_sha256(path: Path) -> str:
    """Hash every stored episode array in one declared order, excluding metadata."""
    members = (
        "optic_rgb", "body_afferents", "delivered_context", "collected_latent",
        "delivered_motor", "cns_motor", "teacher_motor", "applied_body_control",
        "teacher_valid", "joint_position", "joint_velocity", "segment_pose",
        "ground_contact_raw", "sensory_site_position", "mouth_contact_raw",
        "outcome", "reward", "success", "failure", "reset", "active", "terminal",
        "control_source", "curriculum_phase",
    )
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", *members}:
            raise ValueError(f"episode array members differ: {path.name}")
        for name in members:
            value = np.ascontiguousarray(archive[name])
            header = json.dumps(
                {"name": name, "dtype": value.dtype.str, "shape": value.shape},
                sort_keys=True, separators=(",", ":"),
            ).encode()
            digest.update(len(header).to_bytes(4, "little"))
            digest.update(header)
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("nursery layout contains non-finite coordinate")
    return f"{value:.10g}"


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _rotate_xy(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def _replace_vector(element: ET.Element, attribute: str, values: list[float]) -> None:
    element.set(attribute, " ".join(_format_number(value) for value in values))


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    element = root.find(f".//{tag}[@name='{name}']")
    if element is None:
        raise ValueError(f"base scene lacks {tag} {name}")
    return element


def _id_contract(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compiled_counts": fixture["compiled_counts"],
        "residents": [
            {
                "id": row["id"],
                "root_body_id": row["root_body_id"],
                "head_body_id": row["head_body_id"],
                "free_joint_id": row["free_joint_id"],
                "free_qpos_address": row["free_qpos_address"],
                "segment_body_ids": [item["body_id"] for item in row["segments69"]],
            }
            for row in fixture["residents"]
        ],
        "entities": [
            {"id": row["id"], "body": row["body"], "geoms": row["geoms"]}
            for row in fixture["entities"]
        ],
        "screen_geom": fixture["screen_geom"],
    }


def _link_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _layout_payload(
    fixture: Mapping[str, Any], qpos_tokens: list[str], variant: int, seed: int,
    root_changes: list[dict[str, Any]], grain_changes: list[dict[str, Any]],
    static_changes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": NURSERY_LAYOUT_FORMAT,
        "variant": variant,
        "seed": seed,
        "base_fixture_sha256": sha256_file(DEFAULT_BASE_SCENE.resolve())
        if DEFAULT_BASE_SCENE.resolve().exists() else None,
        "body_schema_sha256": fixture["body_schema_sha256"],
        "sensory_schema_sha256": fixture["sensory_schema_sha256"],
        "actuator_schema_sha256": fixture["actuator_schema_sha256"],
        "body_axis_values_sha256": _sha256_bytes(
            " ".join(
                token
                for resident in fixture["residents"]
                for token in qpos_tokens[
                    int(resident["free_qpos_address"]) + 7:
                    int(resident["free_qpos_address"]) + 133
                ]
            ).encode()
        ),
        "root_free_qpos": root_changes,
        "movable_grain_free_qpos": grain_changes,
        "static_existing_geom_changes": static_changes,
        "compiled_id_contract_sha256": _sha256_bytes(_canonical_bytes(_id_contract(fixture))),
    }


def compose_layouts(
    base_scene: Path = DEFAULT_BASE_SCENE,
    output_root: Path = DEFAULT_LAYOUT_ROOT,
    *,
    seed: int = 20260908,
) -> dict[str, Any]:
    """Compose twelve deterministic XML/fixture variants without adding elements."""
    base_scene = base_scene.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"nursery layout directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    base_directory = base_scene.parent
    base_fixture_bytes = base_scene.read_bytes()
    base_fixture = json.loads(base_fixture_bytes)
    base_xml_path = base_directory / str(base_fixture["scene_xml"])
    base_xml_bytes = base_xml_path.read_bytes()
    if sha256_file(base_xml_path) != base_fixture["scene_xml_sha256"]:
        raise ValueError("base fixture XML checksum differs")
    if len(base_fixture["residents"]) != RESIDENTS:
        raise ValueError("nursery requires the actual B4 fixture")

    variants: list[dict[str, Any]] = []
    base_id_hash = _sha256_bytes(_canonical_bytes(_id_contract(base_fixture)))
    for variant in range(12):
        variant_seed = int(np.random.SeedSequence([seed, variant, 0x4E555253]).generate_state(1)[0])
        rng = np.random.default_rng(variant_seed)
        root = ET.fromstring(base_xml_bytes)
        neutral = _find_named(root, "key", str(base_fixture["neutral_keyframe"]["name"]))
        qpos_tokens = neutral.attrib["qpos"].split()
        if len(qpos_tokens) != int(base_fixture["compiled_counts"]["nq"]):
            raise ValueError("neutral keyframe qpos length differs from compiled fixture")

        global_rotation = (variant - 5.5) * 0.027 + float(rng.uniform(-0.018, 0.018))
        root_changes: list[dict[str, Any]] = []
        for resident_index, resident in enumerate(base_fixture["residents"]):
            address = int(resident["free_qpos_address"])
            original = [float(value) for value in qpos_tokens[address:address + 7]]
            radius = math.hypot(original[0], original[1]) + float(rng.uniform(-0.28, 0.28))
            angle = math.atan2(original[1], original[0]) + global_rotation
            angle += (resident_index - 1.5) * 0.009 + float(rng.uniform(-0.012, 0.012))
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            heading = math.atan2(-y, -x) + float(rng.uniform(-0.16, 0.16))
            replacement = [x, y, original[2], *_yaw_quaternion(heading)]
            qpos_tokens[address:address + 7] = [_format_number(value) for value in replacement]
            root_changes.append({
                "resident": resident["id"], "qpos_address": address,
                "position_mm": replacement[:3], "heading_rad": heading,
            })

        resident_qpos_end = max(int(row["free_qpos_address"]) + 133 for row in base_fixture["residents"])
        grains = [row for row in base_fixture["entities"] if str(row["id"]).startswith("grain-")]
        if resident_qpos_end + len(grains) * 7 != len(qpos_tokens):
            raise ValueError("movable-grain free qpos block is not the expected suffix")
        grain_changes: list[dict[str, Any]] = []
        grain_rotation = -global_rotation * 1.7 + float(rng.uniform(-0.12, 0.12))
        for grain_index, grain in enumerate(grains):
            address = resident_qpos_end + grain_index * 7
            original = [float(value) for value in qpos_tokens[address:address + 7]]
            x, y = _rotate_xy(original[0], original[1], grain_rotation)
            x += float(rng.uniform(-0.38, 0.38))
            y += float(rng.uniform(-0.38, 0.38))
            yaw = grain_rotation + float(rng.uniform(-0.6, 0.6))
            replacement = [x, y, original[2], *_yaw_quaternion(yaw)]
            qpos_tokens[address:address + 7] = [_format_number(value) for value in replacement]
            grain_changes.append({
                "entity": grain["id"], "qpos_address": address,
                "position_mm": replacement[:3], "heading_rad": yaw,
            })
        neutral.set("qpos", " ".join(qpos_tokens))

        static_changes: list[dict[str, Any]] = []
        for name in ("ecology/leaf-west", "ecology/leaf-east"):
            geom = _find_named(root, "geom", name)
            pos = [float(value) for value in geom.attrib["pos"].split()]
            euler = [float(value) for value in geom.attrib["euler"].split()]
            pos[0] += float(rng.uniform(-0.35, 0.35))
            pos[1] += float(rng.uniform(-0.35, 0.35))
            euler[2] += float(rng.uniform(-0.14, 0.14))
            _replace_vector(geom, "pos", pos)
            _replace_vector(geom, "euler", euler)
            static_changes.append({"geom": name, "position_mm": pos, "euler_rad": euler})
        bark_index = variant % 7
        bark_name = f"ecology/curved-bark-{bark_index:02d}"
        bark = _find_named(root, "geom", bark_name)
        bark_pos = [float(value) for value in bark.attrib["pos"].split()]
        bark_euler = [float(value) for value in bark.attrib["euler"].split()]
        bark_pos[0] += float(rng.uniform(-0.22, 0.22))
        bark_pos[1] += float(rng.uniform(-0.18, 0.18))
        bark_euler[2] += float(rng.uniform(-0.09, 0.09))
        _replace_vector(bark, "pos", bark_pos)
        _replace_vector(bark, "euler", bark_euler)
        static_changes.append({"geom": bark_name, "position_mm": bark_pos, "euler_rad": bark_euler})

        xml_bytes = ET.tostring(root, encoding="utf-8", short_empty_elements=True) + b"\n"
        xml_sha = _sha256_bytes(xml_bytes)
        layout = _layout_payload(
            base_fixture, qpos_tokens, variant, variant_seed,
            root_changes, grain_changes, static_changes,
        )
        layout["base_fixture_sha256"] = _sha256_bytes(base_fixture_bytes)
        layout_sha = _sha256_bytes(_canonical_bytes(layout))
        layout["layout_sha256"] = layout_sha

        destination = output_root / f"variant-{variant:02d}"
        destination.mkdir()
        (destination / "scene.xml").write_bytes(xml_bytes)
        fixture_hashes: dict[str, str] = {}
        for fixture_name in ("physics.json", "world.json"):
            source_path = base_directory / fixture_name
            value = json.loads(source_path.read_text())
            value["source_mjcf_sha256"] = xml_sha
            value["scene_xml"] = "scene.xml"
            value["scene_xml_sha256"] = xml_sha
            value["nursery_layout"] = layout
            encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
            (destination / fixture_name).write_bytes(encoded)
            fixture_hashes[fixture_name] = _sha256_bytes(encoded)
        for asset in base_fixture["mesh_assets"]:
            relative = Path(str(asset["path"]))
            _link_asset(base_directory / relative, destination / relative)
            if sha256_file(destination / relative) != asset["sha256"]:
                raise ValueError(f"linked mesh changed: {relative}")
        license_path = base_directory / "FlyGym-LICENSE"
        if license_path.exists():
            _link_asset(license_path, destination / license_path.name)

        manifest = {
            "format": NURSERY_LAYOUT_FORMAT,
            "variant": variant,
            "seed": variant_seed,
            "base_fixture_sha256": _sha256_bytes(base_fixture_bytes),
            "base_xml_sha256": _sha256_bytes(base_xml_bytes),
            "scene_xml_sha256": xml_sha,
            "physics_fixture_sha256": fixture_hashes["physics.json"],
            "world_fixture_sha256": fixture_hashes["world.json"],
            "layout_sha256": layout_sha,
            "compiled_id_contract_sha256": base_id_hash,
            "compiled_ids_expected_unchanged": True,
            "actual_startup_validation": None,
            "layout": layout,
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
        (destination / "manifest.json").write_bytes(manifest_bytes)
        variants.append({
            "variant": variant,
            "directory": destination.name,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "scene_xml_sha256": xml_sha,
            "world_fixture_sha256": fixture_hashes["world.json"],
            "layout_sha256": layout_sha,
        })

    if len({row["layout_sha256"] for row in variants}) != 12:
        raise AssertionError("nursery composer did not produce twelve distinct layouts")
    top = {
        "format": NURSERY_LAYOUT_FORMAT,
        "variants": variants,
        "base_scene": str(base_scene),
        "base_fixture_sha256": _sha256_bytes(base_fixture_bytes),
        "base_xml_sha256": _sha256_bytes(base_xml_bytes),
        "compiled_id_contract_sha256": base_id_hash,
    }
    top_bytes = json.dumps(top, sort_keys=True, indent=2).encode() + b"\n"
    (output_root / "nursery-layouts.json").write_bytes(top_bytes)
    return {**top, "manifest_sha256": _sha256_bytes(top_bytes), "output": str(output_root)}


@dataclass(frozen=True)
class NurseryStimulus:
    start: int
    stop: int
    capability: str
    frequency_hz: float
    position_mm: tuple[float, float, float]
    envelope: float
    duration_s: float
    visual_cue: bool
    screen_pattern: str
    screen_frame_sha256: str


@dataclass(frozen=True)
class NurseryPlan:
    world_index: int
    world_seed: int
    variation_seed: int
    ticks: int
    residents: int
    bouts: tuple[tuple[Bout, ...], ...]
    context: np.ndarray
    stimuli: tuple[NurseryStimulus, ...]

    def metadata(self) -> dict[str, Any]:
        payload = {
            "format": NURSERY_PLAN_FORMAT,
            "world_index": self.world_index,
            "world_seed": self.world_seed,
            "variation_seed": self.variation_seed,
            "ticks": self.ticks,
            "residents": self.residents,
            "tone_capability_mapping_hz": TONE_HZ,
            "bouts": [[asdict(bout) for bout in row] for row in self.bouts],
            "stimuli": [asdict(stimulus) for stimulus in self.stimuli],
            "stimulus_boundary": "physical visitorSound and screen only; T+1 BODY/retina",
        }
        payload["plan_sha256"] = _sha256_bytes(_canonical_bytes(payload))
        return payload


def _screen_frame(pattern: str) -> np.ndarray:
    if pattern == "neutral-gray":
        return np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 0.18, dtype="<f4")
    try:
        capability_index = CAPABILITIES.index(pattern.removeprefix("grating/"))
    except ValueError as exc:
        raise ValueError(f"unknown nursery screen pattern {pattern}") from exc
    yy, xx = np.mgrid[:SCREEN_HEIGHT, :SCREEN_WIDTH]
    angle = capability_index * math.pi / 4
    coordinate = xx * math.cos(angle) + yy * math.sin(angle)
    stripes = (np.sin(2 * math.pi * coordinate / 8.0) >= 0).astype(np.float32)
    colors = np.asarray([
        [0.12, 0.22, 0.08], [0.22, 0.10, 0.04],
        [0.06, 0.16, 0.24], [0.24, 0.20, 0.05],
    ], np.float32)[capability_index]
    frame = 0.04 + stripes[..., None] * colors
    return np.ascontiguousarray(frame, dtype="<f4")


def _screen_sha(pattern: str) -> str:
    return _sha256_bytes(_screen_frame(pattern).tobytes())


def build_nursery_plan(world_index: int, *, base_seed: int = 20260908) -> NurseryPlan:
    """Build 16 global 0.64-s tone/action bouts with 12 teacher and four CNS probes."""
    if not 0 <= world_index < 12:
        raise ValueError("nursery worlds use indices 0..11")
    sequence = np.random.SeedSequence([base_seed, world_index, 0x4E555253])
    world_seed, variation_seed = (
        int(value) for value in sequence.generate_state(2, dtype=np.uint32)
    )
    rng = np.random.default_rng(variation_seed)
    latin = (
        (0, 1, 2, 3),
        (1, 3, 0, 2),
        (2, 0, 3, 1),
        (3, 2, 1, 0),
    )
    capability_order: list[str] = []
    for block in range(4):
        row = latin[(world_index + block) % 4]
        if (world_index // 4) % 2:
            row = tuple(reversed(row))
        capability_order.extend(CAPABILITIES[index] for index in row)
    occurrence = {capability: 0 for capability in CAPABILITIES}
    probe_occurrence = {
        capability: (world_index + index) % 4
        for index, capability in enumerate(CAPABILITIES)
    }
    row: list[Bout] = []
    stimuli: list[NurseryStimulus] = []
    context = np.zeros((TICKS, RESIDENTS, CONTEXT), np.float32)
    for bout_index, capability in enumerate(capability_order):
        start, stop = bout_index * BOUT_TICKS, (bout_index + 1) * BOUT_TICKS
        current_occurrence = occurrence[capability]
        probe = current_occurrence == probe_occurrence[capability]
        control_source = "cns-context-pulse" if probe else "offline-author-teacher"
        visual = ((current_occurrence + world_index + CAPABILITIES.index(capability)) % 2) == 0
        pattern = f"grating/{capability}" if visual else "neutral-gray"
        difficulty = 0.38 + 0.09 * ((world_index + bout_index) % 4)
        row.append(Bout(
            start=start,
            stop=stop,
            phase=capability,
            control_source=control_source,
            terrain_variant=world_index,
            difficulty=difficulty,
            intended_outcome="tone-paired-cns-probe" if probe else "tone-paired-teacher-demonstration",
        ))
        anchor = np.asarray(SOUND_ANCHOR_MM[capability], np.float64)
        anchor[:2] += rng.uniform(-0.35, 0.35, 2)
        stimuli.append(NurseryStimulus(
            start=start,
            stop=stop,
            capability=capability,
            frequency_hz=TONE_HZ[capability],
            position_mm=tuple(float(value) for value in anchor),
            envelope=0.72,
            duration_s=TONE_DURATION_S,
            visual_cue=visual,
            screen_pattern=pattern,
            screen_frame_sha256=_screen_sha(pattern),
        ))
        if probe:
            for resident in range(RESIDENTS):
                local_rng = np.random.default_rng([variation_seed, bout_index, resident])
                direction = local_rng.choice((-1.0, 1.0), CONTEXT)
                envelope = np.sin(np.linspace(0, math.pi, BOUT_TICKS, dtype=np.float32))[:, None]
                context[start:stop, resident] = 0.32 * envelope * direction
        occurrence[capability] += 1
    if sum(bout.control_source == "offline-author-teacher" for bout in row) != 12:
        raise AssertionError("nursery plan must contain twelve teacher bouts")
    if sum(stimulus.visual_cue for stimulus in stimuli) != 8:
        raise AssertionError("exactly half of nursery tone bouts must carry gratings")
    bouts = tuple(tuple(row) for _ in range(RESIDENTS))
    return NurseryPlan(
        world_index=world_index,
        world_seed=world_seed,
        variation_seed=variation_seed,
        ticks=TICKS,
        residents=RESIDENTS,
        bouts=bouts,
        context=context,
        stimuli=tuple(stimuli),
    )


class NurseryNodeActualFlyWorld(NodeActualFlyWorld):
    """Actual world using the separate, stimulus-aware research bridge."""

    def __init__(self, arguments: Any, plan: NurseryPlan) -> None:
        self.scene = Path(arguments.scene).resolve()
        self.runtime = Path(arguments.runtime).resolve()
        self.core_wasm = Path(arguments.core_wasm).resolve()
        bridge = Path(__file__).with_name("stimuli.mjs").resolve()
        command = [
            str(arguments.node), str(bridge), "--scene", str(self.scene),
            "--runtime", str(self.runtime), "--core-wasm", str(self.core_wasm),
            "--seed", str(plan.world_seed),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to create nursery world pipes")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"nursery world exited during startup ({self.process.poll()})")
        self.ready = json.loads(line)
        if (
            not self.ready.get("ok")
            or self.ready.get("event") != "ready"
            or self.ready.get("stimulus_bridge") != "physical-screen-visitor-sound-v1"
        ):
            raise RuntimeError(f"nursery world startup failed: {self.ready}")
        if int(self.ready["residents"]) != RESIDENTS:
            raise RuntimeError("nursery collector requires four actual residents")
        self._id = 0
        self._last = None
        self._layout = _hash_json({
            "fixture_sha256": self.ready["fixture_sha256"],
            "initial_snapshot_sha256": self.ready["initial_snapshot_sha256"],
            "world_seed": plan.world_seed,
            "variation_seed": plan.variation_seed,
        })


class StimulatedWorld:
    """Inject scheduled external surfaces before ordinary physical advances."""

    def __init__(
        self, world: NurseryNodeActualFlyWorld, plan: NurseryPlan,
        deliveries: list[dict[str, Any]],
    ) -> None:
        self.world = world
        self.plan = plan
        self.deliveries = deliveries
        self.tick = 0
        self.by_start = {stimulus.start: stimulus for stimulus in plan.stimuli}

    def sample(self):
        return self.world.sample()

    def advance(self, normalized_motor92: np.ndarray, dt: float = 0.01) -> None:
        stimulus = self.by_start.get(self.tick)
        if stimulus is not None:
            frame = _screen_frame(stimulus.screen_pattern)
            acknowledgement = self.world._rpc(
                "stimulus",
                sound={
                    "position_mm": stimulus.position_mm,
                    "frequency_hz": stimulus.frequency_hz,
                    "envelope": stimulus.envelope,
                    "duration_s": stimulus.duration_s,
                },
                screen={
                    "width": SCREEN_WIDTH,
                    "height": SCREEN_HEIGHT,
                    "rgb_f32_base64": base64.b64encode(frame.tobytes()).decode(),
                    "sha256": stimulus.screen_frame_sha256,
                    "pattern": stimulus.screen_pattern,
                },
            )
            self.deliveries.append({
                "tick": self.tick,
                "planned": asdict(stimulus),
                "acknowledged_world_time_s": float(acknowledgement["world_time_s"]),
                "bridge_screen_sha256": str(acknowledgement["screen_sha256"]),
                "bridge_sound": dict(acknowledgement["sound"]),
            })
        self.world.advance(normalized_motor92, dt)
        self.tick += 1

    def initial_snapshot_sha256(self) -> str:
        return self.world.initial_snapshot_sha256()

    def world_instance_identity(self) -> str:
        return self.world.world_instance_identity()

    def scene_layout_identity(self) -> str:
        return self.world.scene_layout_identity()

    def close(self) -> None:
        self.world.close()


def _variant_scene(arguments: Any, world_index: int) -> Path:
    root = Path(arguments.nursery_scenes).resolve()
    manifest_path = root / "nursery-layouts.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != NURSERY_LAYOUT_FORMAT or len(manifest.get("variants", [])) != 12:
        raise ValueError("nursery layout manifest differs")
    row = manifest["variants"][world_index]
    directory = root / str(row["directory"])
    variant_manifest = json.loads((directory / "manifest.json").read_text())
    if sha256_file(directory / "world.json") != row["world_fixture_sha256"]:
        raise ValueError("nursery world fixture checksum differs")
    if variant_manifest["layout_sha256"] != row["layout_sha256"]:
        raise ValueError("nursery variant manifest layout differs")
    return directory / "world.json"


def create_bundle(arguments: Any, plan: NurseryPlan) -> CollectionBundle:
    """Create an actual B4+CNS bundle with opt-in physical cue delivery."""
    if not isinstance(plan, NurseryPlan):
        raise TypeError("embodied nursery bundle requires NurseryPlan")
    scene = _variant_scene(arguments, plan.world_index)
    nursery_arguments = SimpleNamespace(**vars(arguments))
    nursery_arguments.scene = scene
    world = NurseryNodeActualFlyWorld(nursery_arguments, plan)
    if arguments.cns_backend == "dawn":
        # Imported lazily so the separate nursery entry point is the only new
        # consumer.  The running body-bootstrap collector remains untouched.
        from .dawn_host import DawnFullCNS

        cns = DawnFullCNS(Path(arguments.service), str(arguments.node))
    elif arguments.cns_backend == "torch":
        cns = TorchFullCNS(Path(arguments.service), str(arguments.device))
    else:  # argparse prevents this; retain a fail-closed programmatic API.
        raise ValueError(f"unknown CNS backend {arguments.cns_backend}")
    author = SampledAuthorSteps(Path(arguments.author_source))
    teacher = FlyCurriculumTeacher(Path(arguments.body_schema), author)
    fixture = world.ready["fixture"]
    ecology_ids = tuple(str(item["ecology_id"]) for item in fixture["bodies"])
    deliveries: list[dict[str, Any]] = []
    stimulated = StimulatedWorld(world, plan, deliveries)
    variant_manifest = json.loads(scene.with_name("manifest.json").read_text())
    metadata = {
        "source_revision": str(arguments.source_revision),
        "morphology_source_revision": fixture["source_revision"],
        "body_schema_sha256": fixture["body_schema_sha256"],
        "cns_body807_schema_sha256": fixture["sensory_schema_sha256"],
        "physical_sensory_schema_sha256": fixture["physical_sensory_schema_sha256"],
        "morphology_sha256": fixture["morphology_sha256"],
        "motor_atlas_sha256": sha256_file(Path(arguments.motor_atlas).resolve()),
        "cns_service_sha256": cns.metadata["cns_service_sha256"],
        "cns_adapter_sha256": cns.metadata["adapter_sha256"],
        "motor_calibration_sha256": cns.metadata["motor_calibration_sha256"],
        "retina_mapping_sha256": cns.metadata["atlas_sha256"],
        "scene_manifest_sha256": world.ready["fixture_sha256"],
        "scene_xml_sha256": world.ready["scene_xml_sha256"],
        "core_wasm_sha256": world.ready["core_wasm_sha256"],
        "native_runtime_sha256": sha256_file(Path(arguments.runtime).resolve()),
        "nursery_source_sha256": sha256_file(Path(__file__).resolve()),
        "nursery_stimulus_bridge_sha256": sha256_file(
            Path(__file__).with_name("stimuli.mjs")
        ),
        "author_trajectory_bank_sha256": sha256_file(Path(arguments.author_source).resolve()),
        "author_trajectory_manifest_sha256": sha256_file(Path(arguments.author_source).with_name("manifest.json")),
        "cns_format": cns.metadata["format"],
        "cns_backend": cns.metadata.get("cns_backend", "torch"),
        "native_world_engine": world.ready["engine"],
        "nursery_format": NURSERY_CORPUS_FORMAT,
        "nursery_layout_manifest_sha256": sha256_file(scene.with_name("manifest.json")),
        "nursery_layout_sha256": variant_manifest["layout_sha256"],
        "nursery_stimulus_schedule": [asdict(item) for item in plan.stimuli],
        "nursery_stimulus_schedule_sha256": _sha256_bytes(
            _canonical_bytes([asdict(item) for item in plan.stimuli])
        ),
        "nursery_stimulus_deliveries": deliveries,
        "nursery_raw_stimulus_controller_access": False,
    }
    if arguments.cns_backend == "dawn":
        # Preserve the authenticated Dawn identities exposed by DawnFullCNS;
        # these are not reconstructed from filenames or nursery configuration.
        metadata.update({
            "cns_webgpu_manifest_sha256": cns.metadata["webgpu_manifest_sha256"],
            "cns_source_revision": cns.metadata["cns_source_revision"],
        })
    try:
        return SimpleNamespace(
            world=stimulated,
            cns=cns,
            teacher=teacher,
            evaluator=ActualOutcomeEvaluator(ecology_ids),
            metadata=metadata,
        )
    except Exception:
        cns.close()
        world.close()
        raise


def seal_nursery_manifest(episode_directory: Path, output: Path) -> dict[str, Any]:
    """Seal episode identities while requiring twelve different physical layouts."""
    from .data import load_episode

    episode_directory = episode_directory.resolve()
    output = output.resolve()
    if output.parent != episode_directory:
        raise ValueError("nursery manifest must be sealed beside its episode files")
    rows = []
    for world_index in range(12):
        path = episode_directory / f"episode-{world_index:02d}.npz"
        episode = load_episode(path)
        if int(episode.metadata["world_index"]) != world_index:
            raise ValueError(f"episode {world_index} has wrong world identity")
        if episode.metadata.get("nursery_format") != NURSERY_CORPUS_FORMAT:
            raise ValueError(f"episode {world_index} is not an embodied nursery episode")
        deliveries = episode.metadata.get("nursery_stimulus_deliveries", [])
        if len(deliveries) != 16 or [row["tick"] for row in deliveries] != list(range(0, TICKS, BOUT_TICKS)):
            raise ValueError(f"episode {world_index} lacks its exact physical stimulus log")
        rows.append({
            "world_index": world_index,
            "split": split_for_world(world_index),
            "file": path.name,
            "sha256": episode.sha256,
            "scene_manifest_sha256": episode.metadata["scene_manifest_sha256"],
            "scene_layout_identity": episode.metadata["scene_layout_identity"],
            "nursery_layout_sha256": episode.metadata["nursery_layout_sha256"],
            "initial_snapshot_sha256": episode.metadata["initial_snapshot_sha256"],
            "stimulus_schedule_sha256": episode.metadata["nursery_stimulus_schedule_sha256"],
        })
    if len({row["scene_layout_identity"] for row in rows}) != 12:
        raise ValueError("nursery corpus must contain twelve distinct scene fixtures")
    if len({row["nursery_layout_sha256"] for row in rows}) != 12:
        raise ValueError("nursery corpus must contain twelve distinct layout payloads")
    historical = episode_directory.parent / "historical"
    supplemental_source = historical / "episode-00-pre-schema-scope-correction.npz"
    amendment_source = historical / "episode-00-pre-schema-scope-correction.amendment.json"
    if not supplemental_source.exists() or not amendment_source.exists():
        raise ValueError("nursery seal requires its preserved amended supplemental episode")
    amendment = json.loads(amendment_source.read_text())
    supplemental_sha = sha256_file(supplemental_source)
    if (
        amendment.get("sha256") != supplemental_sha
        or amendment.get("usable_for_training") is not True
        or amendment.get("metadata_change_only") is not True
        or amendment.get("numerical_dynamics_action_reward_contract_unchanged") is not True
    ):
        raise ValueError("supplemental nursery amendment contract differs")
    supplemental_file = "supplemental-episode-00-pre-schema-scope-correction.npz"
    amendment_file = "supplemental-episode-00-pre-schema-scope-correction.amendment.json"
    for source, name in (
        (supplemental_source, supplemental_file),
        (amendment_source, amendment_file),
    ):
        destination = episode_directory / name
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise ValueError(f"existing supplemental nursery file differs: {name}")
        else:
            _link_asset(source, destination)
    amendment_sha = sha256_file(episode_directory / amendment_file)
    supplemental_trajectory_sha = _episode_trajectory_sha256(
        episode_directory / supplemental_file
    )
    canonical_world00_trajectory_sha = _episode_trajectory_sha256(
        episode_directory / "episode-00.npz"
    )
    numerically_distinct = (
        supplemental_trajectory_sha != canonical_world00_trajectory_sha
    )
    manifest = {
        "format": NURSERY_CORPUS_FORMAT,
        "completed": True,
        "worlds": 12,
        "ticks": TICKS,
        "residents": RESIDENTS,
        "tone_capability_mapping_hz": TONE_HZ,
        "episodes": rows,
        "supplemental_train": [{
            "file": supplemental_file,
            "sha256": supplemental_sha,
            "amendment_file": amendment_file,
            "amendment_sha256": amendment_sha,
            "duplicate_layout_of_world": 0,
            "supplemental_numerical_trajectory_sha256": supplemental_trajectory_sha,
            "canonical_world00_numerical_trajectory_sha256": canonical_world00_trajectory_sha,
            "numerically_distinct": numerically_distinct,
            "usable_training_row": numerically_distinct,
            "replicate_label": (
                "same-seed-layout-replicate"
                if numerically_distinct else "numerically-identical-receipt-only"
            ),
        }],
        "numerical_trajectory_hash": (
            "sha256 over name/dtype/shape header and contiguous bytes for all 24 episode "
            "arrays in declared member order; scalar metadata excluded"
        ),
    }
    encoded = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return {**manifest, "sha256": _sha256_bytes(encoded), "path": str(output.resolve())}


def _collection_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--nursery-scenes", type=Path, default=DEFAULT_LAYOUT_ROOT)
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
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--cns-backend", choices=("dawn", "torch"), default="dawn",
        help="nursery CNS implementation; this opt-in entry point defaults to M2 Dawn",
    )


async def _collect(arguments: argparse.Namespace) -> None:
    plan = build_nursery_plan(arguments.world_index, base_seed=arguments.seed)
    bundle = create_bundle(arguments, plan)
    try:
        print(json.dumps(await collect_episode(bundle, plan, arguments.output), sort_keys=True))
    finally:
        bundle.cns.close()
        bundle.world.close()


def _main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    compose = subparsers.add_parser("compose")
    compose.add_argument("--base-scene", type=Path, default=DEFAULT_BASE_SCENE)
    compose.add_argument("--output-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    compose.add_argument("--seed", type=int, default=20260908)
    collect = subparsers.add_parser("collect")
    _collection_parser(collect)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--episodes", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "compose":
        print(json.dumps(compose_layouts(arguments.base_scene, arguments.output_root, seed=arguments.seed), sort_keys=True))
    elif arguments.command == "collect":
        asyncio.run(_collect(arguments))
    elif arguments.command == "seal":
        print(json.dumps(seal_nursery_manifest(arguments.episodes, arguments.output), sort_keys=True))


if __name__ == "__main__":
    _main()
