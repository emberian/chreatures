"""Concrete research host for actual MuJoCo/Wasm worlds and Torch CHCNS4.

The JSON pipe is deliberately narrow: policy-visible retina/BODY807 are sent to
the full CNS, while all MuJoCo/ecology observations stay in target-only
``WorldSample.observer`` and the offline teacher/evaluator.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from chreatures.cns_adapter_contract import load_service_artifact
from research.anatomical_cns.model import AnatomicalCNS, CNSState

from .collect import CollectionBundle, WorldSample
from .curriculum import Plan, RESIDENTS
from .data import BODY_AFFERENTS, LATENT, MOTOR, OPTIC_SITES, sha256_file
from .teacher import FlyCurriculumTeacher, TeacherObservation


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _decode(value: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    dtype = np.dtype(str(value["dtype"]))
    raw = base64.b64decode(value["base64"], validate=True)
    result = np.frombuffer(raw, dtype=dtype).copy()
    if result.size != int(value["length"]) or result.size != int(np.prod(shape)):
        raise RuntimeError(f"native bridge tensor length differs for {shape}")
    return result.reshape(shape)


class SampledAuthorSteps:
    """Periodic interpolation of the sealed FlyGym author trajectory bank."""

    def __init__(self, bank: Path) -> None:
        self.path = bank.resolve()
        manifest_path = self.path.with_name("manifest.json")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != "chreatures.author-step-bank.v1":
            raise ValueError("author trajectory bank format differs")
        if self.manifest["bank"]["sha256"] != sha256_file(self.path):
            raise ValueError("author trajectory bank checksum differs")
        with np.load(self.path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "phase_cycles", "joint_targets_rad", "joint_targets_normalized",
                "adhesion", "fixture_neutral_rad", "teacher_neutral_rad",
                "control_ranges_rad",
            }:
                raise ValueError("author trajectory bank members differ")
            self.phase = np.asarray(archive["phase_cycles"], np.float64)
            self.joint = np.asarray(archive["joint_targets_rad"], np.float64)
            self.adhesion = np.asarray(archive["adhesion"], np.float64)
            self.neutral = np.asarray(archive["teacher_neutral_rad"], np.float64)
        if self.phase.shape != (2048,) or self.joint.shape != (2048, 42):
            raise ValueError("author trajectory bank must be 2048xwalking42")
        if self.adhesion.shape != (2048, 6) or self.neutral.shape != (42,):
            raise ValueError("author trajectory adhesion/neutral order differs")

    def _interpolate(self, values: np.ndarray, phases: np.ndarray) -> np.ndarray:
        coordinate = np.mod(phases, 1.0) * self.phase.size
        left = np.floor(coordinate).astype(np.int64) % self.phase.size
        right = (left + 1) % self.phase.size
        fraction = coordinate - np.floor(coordinate)
        return values[left] * (1 - fraction[:, None]) + values[right] * fraction[:, None]

    def get_joint_angles_by_dof_order(
        self, phases: np.ndarray, magnitudes: np.ndarray
    ) -> np.ndarray:
        phases = np.asarray(phases, np.float64)
        magnitudes = np.asarray(magnitudes, np.float64)
        if phases.shape != (6,) or magnitudes.shape != (6,):
            raise ValueError("author query must contain six ordered legs")
        # The bank is leg-major, seven joints per leg. Each requested leg has
        # its own phase; magnitude scales only the author's neutral deviation.
        result = np.empty((6, 7), np.float64)
        for leg in range(6):
            row = self._interpolate(self.joint, phases[[leg]])[0, leg * 7 : (leg + 1) * 7]
            neutral = self.neutral[leg * 7 : (leg + 1) * 7]
            result[leg] = neutral + magnitudes[leg] * (row - neutral)
        return result.reshape(42).astype(np.float32)

    def get_adhesion_onoff_by_phase(self, phases: np.ndarray) -> np.ndarray:
        phases = np.asarray(phases, np.float64)
        if phases.shape != (6,):
            raise ValueError("adhesion query must contain six ordered legs")
        coordinate = np.floor(np.mod(phases, 1.0) * self.phase.size).astype(np.int64)
        return self.adhesion[coordinate, np.arange(6)].astype(np.float32)


class NodeActualFlyWorld:
    def __init__(self, arguments: Any, plan: Plan) -> None:
        self.scene = Path(arguments.scene).resolve()
        self.runtime = Path(arguments.runtime).resolve()
        self.core_wasm = Path(arguments.core_wasm).resolve()
        bridge = Path(__file__).with_name("world_bridge.mjs").resolve()
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
            raise RuntimeError("failed to create native world pipes")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"native world exited during startup ({self.process.poll()})")
        self.ready = json.loads(line)
        if not self.ready.get("ok") or self.ready.get("event") != "ready":
            raise RuntimeError(f"native world startup failed: {self.ready}")
        if int(self.ready["residents"]) != RESIDENTS:
            raise RuntimeError("collector requires four actual residents")
        self._id = 0
        self._last: WorldSample | None = None
        self._layout = _hash_json({
            "fixture_sha256": self.ready["fixture_sha256"],
            "initial_snapshot_sha256": self.ready["initial_snapshot_sha256"],
            "world_seed": plan.world_seed,
            "variation_seed": plan.variation_seed,
        })

    def _rpc(self, command: str, **payload: Any) -> Mapping[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"native world process exited ({self.process.returncode})")
        self._id += 1
        request = {"id": self._id, "command": command, **payload}
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"native world pipe closed ({self.process.poll()})")
        response = json.loads(line)
        if response.get("id") != self._id or not response.get("ok"):
            raise RuntimeError(f"native world request failed: {response}")
        return response

    @staticmethod
    def _world_site(
        positions: np.ndarray, rotations: np.ndarray, descriptor: Mapping[str, Any]
    ) -> np.ndarray:
        body = int(descriptor["body_id"])
        local = descriptor.get("position_local_mm", descriptor.get("position_tip_local_mm"))
        return positions[body] + rotations[body] @ np.asarray(local, np.float64)

    def _sample(self, packet: Mapping[str, Any]) -> WorldSample:
        optic = _decode(packet["optic"], (RESIDENTS, OPTIC_SITES, 3)).astype("<f4", copy=False)
        body = _decode(packet["body"], (RESIDENTS, BODY_AFFERENTS)).astype("<f4", copy=False)
        raw_map = packet["bodyMap"]
        residents = raw_map["residents"]
        qpos = _decode(packet["qpos"], (int(packet["qpos"]["length"]),))
        qvel = _decode(packet["qvel"], (int(packet["qvel"]["length"]),))
        positions = _decode(packet["bodyPositions"], (int(packet["bodyPositions"]["length"]) // 3, 3))
        quaternions = _decode(packet["bodyQuaternions"], (positions.shape[0], 4))
        rotations = _decode(packet["bodyRotations"], (positions.shape[0], 3, 3))
        sensor = _decode(packet["sensordata"], (int(packet["sensordata"]["length"]),))
        control = _decode(packet["ctrl"], (int(packet["ctrl"]["length"]),))
        entity = _decode(packet["entityPosition"], (int(packet["entityPosition"]["length"]) // 3, 3))
        joint_position = np.empty((RESIDENTS, 126), np.float32)
        joint_velocity = np.empty_like(joint_position)
        segment_pose = np.empty((RESIDENTS, 69, 7), np.float32)
        contacts = np.empty((RESIDENTS, 6, 16), np.float32)
        sites = np.empty((RESIDENTS, 5, 3), np.float32)
        applied = np.empty((RESIDENTS, 90), np.float32)
        observations: list[TeacherObservation] = []
        thorax_position = np.empty((RESIDENTS, 3), np.float32)
        thorax_rotation = np.empty((RESIDENTS, 3, 3), np.float32)
        # Geometry is observer-only. Include other residents so social phases
        # are not reduced to static entity layouts.
        entity_candidates = [entity]
        for resident in residents:
            entity_candidates.append(positions[[int(resident["root_body_id"])]])
        all_candidates = np.concatenate(entity_candidates, axis=0)
        for row, resident in enumerate(residents):
            joints = resident["joint_dofs126"]
            joint_position[row] = [qpos[int(item["qpos_address"])] for item in joints]
            joint_velocity[row] = [qvel[int(item["dof_address"])] for item in joints]
            segments = np.asarray([int(item["body_id"]) for item in resident["segments69"]])
            segment_pose[row, :, :3] = positions[segments]
            # MuJoCo is wxyz; the frozen learning trace is xyzw.
            segment_pose[row, :, 3:] = quaternions[segments][:, [1, 2, 3, 0]]
            for foot, descriptor in enumerate(resident["contact_sensors6x16"]):
                address = int(descriptor["data_address"])
                contacts[row, foot] = sensor[address : address + 16]
            anchors = [*resident["olfactory_anchors4"], resident["mouth"]]
            sites[row] = [self._world_site(positions, rotations, item) for item in anchors]
            applied[row] = control[[int(item["actuator_id"]) for item in resident["actuators90"]]]
            root = int(resident["root_body_id"])
            thorax_position[row] = positions[root]
            thorax_rotation[row] = rotations[root]
            actuator_to_qpos = {
                str(item["semantic_id"]): qpos[int(item["qpos_address"])] for item in joints
            }
            active = []
            for actuator in resident["actuators90"][:84]:
                semantic = str(actuator["semantic_id"])
                if not semantic.endswith("-position") or semantic[:-9] not in actuator_to_qpos:
                    raise RuntimeError(f"cannot map actuator to joint: {semantic}")
                active.append(actuator_to_qpos[semantic[:-9]])
            relative = all_candidates - positions[root]
            distance = np.linalg.norm(relative, axis=1)
            distance[distance < 1e-5] = np.inf
            target_world = relative[int(np.argmin(distance))]
            target_local = rotations[root].T @ target_world
            observations.append(TeacherObservation(
                active_joint_position=np.asarray(active, np.float32),
                thorax_up=float(rotations[root, 2, 2]),
                thorax_height=float(positions[root, 2]),
                thorax_linear_velocity=body[row, 62:65].copy(),
                thorax_angular_velocity=body[row, 65:68].copy(),
                foot_contact=(contacts[row, :, 0] > 0).astype(np.float32),
                foot_slip_speed=np.linalg.norm(body[row, 459:495].reshape(6, 6)[:, 3:], axis=1),
                antenna_target_local=target_local.astype(np.float32),
                mouth_target_local=target_local.astype(np.float32),
                mouth_contact=bool(body[row, 711] > 0),
            ))
        observer = {
            "time": float(packet["time"]),
            "thorax_position": thorax_position,
            "thorax_rotation": thorax_rotation,
            "entity_position": entity.astype(np.float32),
            "entity_ids": tuple(item.get("id", "") for item in self.ready["fixture"]["entities"]),
            "ecology": packet["ecology"],
            "actuator_state": packet["actuatorState"],
        }
        return WorldSample(
            optic, body, joint_position, joint_velocity, segment_pose, contacts,
            sites, body[:, 708:712].copy(), applied, tuple(observations), observer,
        )

    def sample(self) -> WorldSample:
        self._last = self._sample(self._rpc("sample")["sample"])
        return self._last

    def advance(self, normalized_motor92: np.ndarray, dt: float = 0.01) -> None:
        value = np.ascontiguousarray(normalized_motor92, dtype="<f4")
        if value.shape != (RESIDENTS, MOTOR) or abs(dt - 0.01) > 1e-12:
            raise ValueError("world advance requires exact B4xM92 at 0.01 seconds")
        self._rpc("advance", motor92_base64=base64.b64encode(value).decode())

    def initial_snapshot_sha256(self) -> str:
        return str(self.ready["initial_snapshot_sha256"])

    def world_instance_identity(self) -> str:
        return self._layout

    def scene_layout_identity(self) -> str:
        return str(self.ready["fixture_sha256"])

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._rpc("close")
            finally:
                self.process.wait(timeout=30)


class TorchFullCNS:
    def __init__(self, service: Path, device: str) -> None:
        self.path = service.resolve()
        arrays, metadata = load_service_artifact(self.path)
        self.metadata = dict(metadata)
        self.metadata["cns_service_sha256"] = sha256_file(self.path)
        self.device = torch.device(device)
        self.model = AnatomicalCNS(arrays, device=self.device).eval()
        self.state: CNSState | None = None

    @torch.inference_mode()
    def step(
        self, optic_rgb: np.ndarray, body_afferents: np.ndarray,
        delivered_context: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        optic = torch.as_tensor(np.ascontiguousarray(optic_rgb), device=self.device)
        body = torch.as_tensor(np.ascontiguousarray(body_afferents), device=self.device)
        context = torch.as_tensor(np.ascontiguousarray(delivered_context), device=self.device)
        # One public .01 step performs the frozen pair of internal dt/2 rate
        # integrations and one .01 slow-state update. Calling twice would
        # incorrectly advance adaptation/support/release twice.
        latent, motor, self.state = self.model(
            optic, body, context, self.state, dt=0.01
        )
        return (
            np.ascontiguousarray(latent.cpu().numpy(), dtype="<f4"),
            np.ascontiguousarray(motor.cpu().numpy(), dtype="<f4"),
        )

    def close(self) -> None:
        self.state = None


def _organism_reserve(ecology: Mapping[str, Any], organism_id: str) -> float:
    for organism in ecology.get("organisms", []):
        if organism.get("id") == organism_id:
            material = organism.get("material", {}).get("quantity", [])
            return float(organism.get("atp", 0.0)) + float(np.sum(material, dtype=np.float64))
    raise RuntimeError(f"ecology observer lacks organism {organism_id}")


class ActualOutcomeEvaluator:
    """Phase labels from actual committed mechanics and conserved ecology only."""

    def __init__(self, ecology_ids: tuple[str, ...]) -> None:
        self.ecology_ids = ecology_ids

    def transition(
        self, before: WorldSample, after: WorldSample, delivered_motor: np.ndarray,
        phase: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        out = np.zeros((RESIDENTS, 16), np.float32)
        reward = np.zeros(RESIDENTS, np.float32)
        success = np.zeros(RESIDENTS, bool)
        failure = np.zeros(RESIDENTS, bool)
        bp = before.observer["thorax_position"]
        ap = after.observer["thorax_position"]
        br = before.observer["thorax_rotation"]
        ar = after.observer["thorax_rotation"]
        delta = ap - bp
        entity_delta = np.linalg.norm(
            after.observer["entity_position"] - before.observer["entity_position"], axis=1
        )
        for row in range(RESIDENTS):
            forward = br[row, :, 0]
            old_heading = np.arctan2(br[row, 1, 0], br[row, 0, 0])
            new_heading = np.arctan2(ar[row, 1, 0], ar[row, 0, 0])
            yaw = np.arctan2(np.sin(new_heading - old_heading), np.cos(new_heading - old_heading))
            speed = float(np.linalg.norm(after.body_afferents[row, 62:65]))
            angular = float(np.linalg.norm(after.body_afferents[row, 65:68]))
            found = after.ground_contact_raw[row, :, 0] > 0
            slip = np.linalg.norm(after.body_afferents[row, 459:495].reshape(6, 6)[:, 3:], axis=1)
            targets = after.observer["entity_position"]
            antenna_distance = np.min(
                np.linalg.norm(after.sensory_site_position[row, :4, None] - targets[None], axis=2)
            )
            reserve_before = _organism_reserve(before.observer["ecology"], self.ecology_ids[row])
            reserve_after = _organism_reserve(after.observer["ecology"], self.ecology_ids[row])
            out[row] = (
                np.clip(ar[row, 2, 2], -1, 1), delta[row, 2],
                np.dot(delta[row], forward), yaw, speed, angular,
                found.mean(), slip.mean(), np.exp(-antenna_distance / 0.5),
                after.mouth_contact_raw[row, 3],
                np.max(after.body_afferents[row, :32]) - np.max(before.body_afferents[row, :32]),
                entity_delta.max(initial=0), after.body_afferents[row, 806],
                np.clip((ar[row, 2, 2] + 1) * 0.5 / (1 + angular), 0, 1),
                np.mean(np.abs(delivered_motor[row])), reserve_after - reserve_before,
            )
            name = phase[row]
            progress = {
                "posture-support": out[row, 0] + out[row, 6],
                "self-right-recovery": out[row, 0],
                "forward-locomotion": 10 * out[row, 2],
                "turning": 2 * abs(out[row, 3]),
                "stopping": -out[row, 4] - out[row, 5],
                "terrain-transition": out[row, 2] + out[row, 6],
                "slip-contact-recovery": out[row, 6] - out[row, 7],
                "antenna-orient-contact": out[row, 8],
                "conspecific-antenna-contact": out[row, 8],
                "mouth-reach-touch-withdraw": out[row, 9],
                "chemical-gradient-forage": 4 * out[row, 10] + out[row, 2],
                "gustatory-intake-pump-salivary": out[row, 9] + out[row, 12] + out[row, 15],
                "movable-material-push": 10 * out[row, 11],
            }.get(name, out[row, 13])
            reward[row] = float(progress - 0.08 * out[row, 14])
            failure[row] = bool(out[row, 0] < -0.15 or (name == "stopping" and out[row, 4] > 1.0))
            success[row] = bool(progress > 0.45 and out[row, 0] > 0 and not failure[row])
        return out, reward, success, failure


class _Bundle:
    def __init__(self, world: NodeActualFlyWorld, cns: TorchFullCNS,
                 teacher: FlyCurriculumTeacher, evaluator: ActualOutcomeEvaluator,
                 metadata: Mapping[str, Any]) -> None:
        self.world, self.cns, self.teacher, self.evaluator = world, cns, teacher, evaluator
        self.metadata = metadata


def create_bundle(arguments: Any, plan: Plan) -> CollectionBundle:
    """Build one fresh, identity-bound actual-world + full-CNS chronology."""
    world = NodeActualFlyWorld(arguments, plan)
    cns = TorchFullCNS(Path(arguments.service), str(arguments.device))
    author = SampledAuthorSteps(Path(arguments.author_source))
    teacher = FlyCurriculumTeacher(Path(arguments.body_schema), author)
    fixture = world.ready["fixture"]
    body_map = fixture["bodies"]
    ecology_ids = tuple(str(item["ecology_id"]) for item in body_map)
    metadata = {
        "source_revision": str(arguments.source_revision),
        "morphology_source_revision": fixture["source_revision"],
        "body_schema_sha256": fixture["sensory_schema_sha256"],
        "morphology_sha256": fixture["morphology_sha256"],
        "motor_atlas_sha256": sha256_file(Path(arguments.motor_atlas).resolve()),
        "cns_service_sha256": cns.metadata["cns_service_sha256"],
        "cns_adapter_sha256": cns.metadata["adapter_sha256"],
        "motor_calibration_sha256": cns.metadata["motor_calibration_sha256"],
        "retina_mapping_sha256": cns.metadata["atlas_sha256"],
        "scene_manifest_sha256": world.ready["fixture_sha256"],
        "scene_xml_sha256": world.ready["scene_xml_sha256"],
        "core_wasm_sha256": world.ready["core_wasm_sha256"],
        "author_trajectory_bank_sha256": sha256_file(Path(arguments.author_source).resolve()),
        "author_trajectory_manifest_sha256": sha256_file(Path(arguments.author_source).with_name("manifest.json")),
        "cns_format": cns.metadata["format"],
        "native_world_engine": world.ready["engine"],
    }
    try:
        return _Bundle(world, cns, teacher, ActualOutcomeEvaluator(ecology_ids), metadata)
    except Exception:
        cns.close(); world.close()
        raise
