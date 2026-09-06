"""One physical habitat coupled to a full connectome and native resident cohort."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import canonical, write_envelope
from .engine_identity import current_engine_identity
from .neural_client import NeuralClient
from .sensorimotor_worker_native import DevelopmentalResidentCohort
from .visitor_events import VisitorPerformances

MODEL_DT = 0.05
SOURCE_SENSE_DIM = 351
NEURAL_DIM = 384
PHYSIOLOGY_DIM = 6
RICH_RETINA_DIM = 4096
ACTION_NAMES = (
    "thrust",
    "yaw",
    "gaze_pitch",
    "grip",
    "signal_low",
    "signal_mid",
    "signal_high",
    "posture",
)
CHECKPOINT_FORMAT = "chreatures-developmental-habitat-checkpoint-v1"
INTERRUPTED_FORMAT = "chreatures-developmental-habitat-interrupted-v1"


def physical_world_type(body_mode: str, execution: str):
    from .sensorium import ArticulatedSensoriumWorld, SensoriumWorld

    if execution == "vectorized" and body_mode == "articulated":
        from .physical_batch import FastArticulatedSensoriumWorld

        return FastArticulatedSensoriumWorld
    if execution != "reference":
        raise ValueError("Vectorized physics requires an articulated body")
    return ArticulatedSensoriumWorld if body_mode == "articulated" else SensoriumWorld


def _finite_row(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    row = np.ascontiguousarray(value, dtype=np.float32)
    if row.shape != shape or not np.isfinite(row).all():
        raise ValueError(f"{name} has invalid shape or nonfinite values")
    return row


class Habitat3D:
    """A synchronous cohort: senses -> connectome -> native cognition -> physics."""

    def __init__(
        self,
        seed: int = 7,
        brain_url: str = "http://127.0.0.1:18765",
        spec: dict[str, Any] | None = None,
        body_mode: str = "articulated",
        ecology: str = "diffusion",
        resources: str | Path | None = None,
        biosphere: str | Path | None = None,
        acoustics: str | Path | None = None,
        resident_artifact: str | Path | None = None,
        physics_backend: str | None = None,
        visitor_materials: str | Path | None = None,
    ) -> None:
        from .acoustics import Acoustics
        from .ecology import Ecology
        from .fields import FieldEnvironment

        if resident_artifact is None:
            raise ValueError("A current --resident-artifact is required for a new life")
        self.engine_identity = current_engine_identity()
        if body_mode not in {"crawler", "articulated"}:
            raise ValueError("Unknown body model")
        if ecology not in {"analytic", "diffusion"}:
            raise ValueError("Unknown ecology model")
        if resources is not None and biosphere is not None:
            raise ValueError(
                "Resources and a developmental biosphere are mutually exclusive"
            )
        if spec is None:
            spec = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "data/habitats/hollow-garden.json"
                ).read_text()
            )
        spec = copy.deepcopy(spec)
        self.body_mode = body_mode
        self.physics_backend = physics_backend or (
            "vectorized" if body_mode == "articulated" else "reference"
        )
        world_type = physical_world_type(body_mode, self.physics_backend)
        self.world = world_type(seed=seed, spec=spec)
        self.field = (
            FieldEnvironment.from_world(self.world) if ecology == "diffusion" else None
        )
        self.resources = (
            Ecology(self.world, resources, seed=seed) if resources is not None else None
        )
        self.resource_state = None
        self.biosphere = None
        if biosphere is not None:
            from .biosphere import Biosphere

            self.biosphere = Biosphere.from_config(self.world, biosphere)
        self.visitor_materials = None
        if visitor_materials is not None:
            from .visitor_materials import VisitorMaterialSupply

            if self.biosphere is None:
                raise ValueError("Material offerings require a developmental biosphere")
            self.visitor_materials = VisitorMaterialSupply(
                self.biosphere, visitor_materials
            )
        self.acoustics = Acoustics(
            self.world,
            acoustics
            if acoustics is not None
            else {"version": 1, "include_authored": True, "emitters": []},
        )
        self.acoustic_state = None

        self.neural = NeuralClient(brain_url)
        self._validate_neural_interface()
        self.resident_artifact = str(Path(resident_artifact).expanduser().resolve())
        cohort_size = len(self.world.bodies)
        self.residents = DevelopmentalResidentCohort(
            self.resident_artifact,
            cohort_size,
            action_mode="sample",
            goal_seed=(seed * 1009 + 17) % 2**64,
            action_seed=(seed * 1009 + 31) % 2**64,
        )
        self._validate_resident_interface()

        self.id = str(uuid.uuid4())
        self.tick = 0
        self.paused = False
        self.speed = 1
        self.branch = "resident"
        self.saved_at = None
        self.error = None
        self.remote_ids = {
            body.id: f"{self.id}:{body.id}" for body in self.world.bodies
        }
        self.neural.create(list(self.remote_ids.values()))
        self.actual_previous = np.zeros((cohort_size, 9), dtype=np.float32)
        self.reset_rows = np.ones(cohort_size, dtype=np.bool_)
        self.outcomes = {body.id: {} for body in self.world.bodies}
        self.neural_state = {
            body.id: {"features": [0.0] * NEURAL_DIM, "activity": 0.0, "support": 1.0}
            for body in self.world.bodies
        }
        self.cognition_state = {
            body.id: self._empty_cognition(body.id) for body in self.world.bodies
        }
        self.last_senses: dict[str, Any] = {}
        self.sensed_at = 0.0
        self.journal = deque(maxlen=256)
        self.history = {body.id: deque(maxlen=360) for body in self.world.bodies}
        self.timings = deque(maxlen=120)
        self.phase_timings = deque(maxlen=120)
        self.pending_step = None
        self.visitor = VisitorPerformances()
        self.execution_migrations: list[dict[str, Any]] = []
        self.note(
            "hatched",
            f"{cohort_size} new residents entered with one native developmental cohort.",
            neurons=int(self.neural.graph["neurons"]),
            graph_sha256=self.neural.graph["sha256"],
            resident_artifact_sha256=self.residents.model_identity["artifact_sha256"],
        )

    def _validate_neural_interface(self) -> None:
        if len(self.neural.input_names) != SOURCE_SENSE_DIM:
            raise ValueError(
                "Current resident artifacts require exactly 351 physical neural inputs"
            )
        if len(self.neural.output_names) != NEURAL_DIM:
            raise ValueError(
                "Current resident artifacts require exactly 384 neural readouts"
            )

    def _validate_resident_interface(self) -> None:
        from .sensorium import RICH_CHANNEL_NAMES_SHA256, RICH_PROFILE_SHA256

        trained_neural = self.residents.neural_contract
        remote_ports = self.neural.metadata["brain"].get("ports", {})
        if (
            trained_neural["graph_sha256"] != self.neural.graph["sha256"]
            or trained_neural["port_spec_sha256"] != remote_ports.get("spec_hash")
        ):
            raise ValueError("Resident artifact was trained with a different graph or neural port")
        expected = {
            "format": "chreatures-rich-sensorimotor-observation-v1",
            "observation_dim": 4453,
            "rich_body_dim": RICH_RETINA_DIM,
            "rich_profile_sha256": RICH_PROFILE_SHA256,
            "rich_channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
            "observation_order": [
                "rich_body_v1_4096",
                "canonical_channels_351",
                "physiology_6",
            ],
            "source_sense_dim": SOURCE_SENSE_DIM,
            "physiology_dim": PHYSIOLOGY_DIM,
            "neural_readout_dim": NEURAL_DIM,
            "previous_action_plus_oral_dim": 9,
        }
        if self.residents.observation_contract != expected:
            raise ValueError(
                "Resident artifact observation contract differs from this host"
            )

    def _empty_cognition(self, body_id: str) -> dict[str, Any]:
        return {
            "controller": "native-developmental-resident",
            "resident": body_id,
            "memory_count": 0,
            "memory_inserted_slot": -1,
            "goal": {
                "valid": False,
                "changed": False,
                "slot": -1,
                "recorded_tick": 0,
                "recorded_time": 0.0,
                "remaining_ticks": 0,
            },
            "sampled_proposal": {name: 0.0 for name in ACTION_NAMES},
            "executed_action": {
                **{name: 0.0 for name in ACTION_NAMES},
                "oral": 0.0,
            },
            "outcome": {},
            "model_identity": copy.deepcopy(self.residents.model_identity),
        }

    def memory_count(self, body_id: str) -> int:
        return int(self.cognition_state[body_id]["memory_count"])

    def cognitive_view(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self.cognition_state)

    def sense(self) -> dict[str, dict[str, Any]]:
        """Sample physical transducers and replace odor with transported concentration."""
        sensed = {body.id: self.world.sense(body.id) for body in self.world.bodies}
        if self.field is not None:
            for body in self.world.bodies:
                values = sensed[body.id]
                positions = values.get("antenna_position")
                if positions is None:
                    rotation = self.world.data.xmat[
                        self.world._body_mj[body.id]
                    ].reshape(3, 3)
                    center = np.asarray([body.x, body.y, body.z])
                    positions = [
                        center + rotation @ np.asarray([0.105, side * 0.055, 0.035])
                        for side in (-1, 1)
                    ]
                concentration = np.asarray(self.field.sample(positions))[:, :3]
                values["odor"] = (-np.expm1(-concentration / 0.1)).tolist()
        return sensed

    def note(self, kind: str, text: str, **fields: Any) -> None:
        self.journal.append(
            {
                "id": f"{self.id}:{self.tick}:{len(self.journal)}",
                "time": self.world.time,
                "kind": kind,
                "text": text,
                **fields,
            }
        )

    def _source_rows(self, sensed: dict[str, dict[str, Any]]) -> np.ndarray:
        rows = []
        for body in self.world.bodies:
            encoded = self.neural.encode(sensed[body.id])
            if set(encoded) != set(self.neural.input_names):
                raise ValueError(
                    "Physical encoder channel identity differs from the neural service"
                )
            rows.append([encoded[name] for name in self.neural.input_names])
        values = _finite_row(rows, (len(rows), SOURCE_SENSE_DIM), "physical senses")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Physical senses exceed their declared [0,1] range")
        return values

    def _physiology_rows(self, responses: dict[str, dict[str, Any]]) -> np.ndarray:
        rows = []
        for body in self.world.bodies:
            response = responses[body.id]
            rows.append(
                (
                    body.energy,
                    body.gut,
                    body.fatigue,
                    math.tanh(float(body.speed) / 2),
                    math.tanh(float(body.angular_velocity) / 4),
                    response["support"],
                )
            )
        result = _finite_row(rows, (len(rows), PHYSIOLOGY_DIM), "physiology")
        if np.any((result[:, :3] < 0.0) | (result[:, :3] > 1.0)):
            raise ValueError("Physiology exceeds its declared range")
        return result

    def _resident_observations(
        self, source: np.ndarray, physiology: np.ndarray
    ) -> np.ndarray:
        rich = _finite_row(
            self.world.rich_retina_batch(),
            (len(self.world.bodies), RICH_RETINA_DIM),
            "rich body senses",
        )
        if np.any((rich < 0.0) | (rich > 1.0)):
            raise ValueError("Rich body senses exceed their declared [0,1] range")
        return np.ascontiguousarray(
            np.concatenate((rich, source, physiology), axis=1), dtype=np.float32
        )

    def _actions(
        self, proposed: np.ndarray, oral_command: np.ndarray
    ) -> tuple[dict[str, dict[str, float]], np.ndarray]:
        values = _finite_row(proposed, (len(self.world.bodies), 8), "native action")
        if np.any((values < -1.0) | (values > 1.0)) or np.any(values[:, 3:7] < 0.0):
            raise ValueError("Native action exceeds the physical actuator bounds")
        oral = _finite_row(oral_command, (len(values),), "native oral command")
        if np.any((oral < 0.0) | (oral > 1.0)):
            raise ValueError("Native oral command exceeds its physical bounds")
        actions = {}
        for index, body in enumerate(self.world.bodies):
            action = dict(
                zip(ACTION_NAMES, values[index].astype(float).tolist(), strict=True)
            )
            action["eat"] = float(oral[index])
            actions[body.id] = action
        return actions, oral

    def _record_cognition(
        self, result: dict[str, np.ndarray], actions: np.ndarray
    ) -> None:
        for index, body in enumerate(self.world.bodies):
            self.cognition_state[body.id] = {
                "controller": "native-developmental-resident",
                "resident": body.id,
                "memory_count": int(result["memory_count"][index]),
                "memory_inserted_slot": int(result["memory_inserted_slot"][index]),
                "goal": {
                    "valid": bool(result["goal_valid"][index]),
                    "changed": bool(result["goal_changed"][index]),
                    "slot": int(result["goal_slot"][index]),
                    "recorded_tick": int(result["goal_recorded_tick"][index]),
                    "recorded_time": float(result["goal_recorded_time"][index]),
                    "remaining_ticks": int(result["goal_remaining_ticks"][index]),
                },
                "sampled_proposal": dict(
                    zip(
                        ACTION_NAMES, actions[index].astype(float).tolist(), strict=True
                    )
                ),
                "consequence_refinement": {
                    "candidate_scores": result["candidate_scores"][index]
                    .astype(float)
                    .tolist(),
                    "candidate_out_of_domain": result["candidate_out_of_domain"][index]
                    .astype(bool)
                    .tolist(),
                    "selected_candidate": int(result["selected_candidate"][index]),
                    "selected_private_correction": result[
                        "selected_consequence_correction"
                    ][index]
                    .astype(float)
                    .tolist(),
                    "completed_private_updates_before_action": int(
                        result["personal_consequence_updates"][index]
                    ),
                    "meaning": "predicted body component of the remembered sensory goal",
                },
                "model_identity": copy.deepcopy(self.residents.model_identity),
            }

    def _record_execution(self, actions: np.ndarray, oral: np.ndarray) -> None:
        """Publish actions and outcomes only after their physical tick commits."""
        for index, body in enumerate(self.world.bodies):
            self.cognition_state[body.id]["executed_action"] = {
                **dict(
                    zip(
                        ACTION_NAMES,
                        actions[index].astype(float).tolist(),
                        strict=True,
                    )
                ),
                "oral": float(oral[index]),
            }
            self.cognition_state[body.id]["outcome"] = copy.deepcopy(
                self.outcomes[body.id]
            )

    def step(self, steps: int = 1) -> None:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.pending_step is not None:
            raise RuntimeError(
                "A previous world step is incomplete; restore its checkpoint"
            )
        for _ in range(steps):
            started = time.perf_counter()
            for event in self.visitor.advance(self.world, self.tick):
                self.note(
                    "visitor-event", "A scheduled physical stimulus occurred.", **event
                )
            sensed = self.sense()
            self.last_senses = sensed
            self.sensed_at = self.world.time
            source = self._source_rows(sensed)
            entries = [
                {
                    "id": self.remote_ids[body.id],
                    "senses": dict(
                        zip(
                            self.neural.input_names,
                            source[index].astype(float).tolist(),
                            strict=True,
                        )
                    ),
                }
                for index, body in enumerate(self.world.bodies)
            ]
            sensed_done = time.perf_counter()
            self.pending_step = {
                "tick": self.tick,
                "neural_seq": self.neural.next_seq,
                "phase": "neural",
            }
            remote = self.neural.step(entries, MODEL_DT)
            neural_done = time.perf_counter()
            inverse = {
                remote_id: body_id for body_id, remote_id in self.remote_ids.items()
            }
            try:
                response_by_body = {inverse[row["id"]]: row for row in remote}
            except KeyError as error:
                raise ValueError(
                    "Neural response contains an unknown resident"
                ) from error
            if set(response_by_body) != set(self.remote_ids):
                raise ValueError(
                    "Neural response cohort differs from the physical cohort"
                )
            neural = _finite_row(
                [response_by_body[body.id]["features"] for body in self.world.bodies],
                (len(self.world.bodies), NEURAL_DIM),
                "neural readouts",
            )
            physiology = self._physiology_rows(response_by_body)
            observation = self._resident_observations(source, physiology)
            self.pending_step["phase"] = "resident"
            native = self.residents.step(
                observation,
                neural,
                physiology,
                self.actual_previous,
                np.full(len(self.world.bodies), self.tick, dtype=np.uint64),
                np.full(len(self.world.bodies), self.world.time, dtype=np.float64),
                self.reset_rows,
            )
            if not np.array_equal(
                native["actual_previous_action"], self.actual_previous[:, :8]
            ):
                raise RuntimeError("Native resident previous-action accounting differs")
            if not np.array_equal(native["physiology"], physiology):
                raise RuntimeError("Native resident physiology accounting differs")
            proposed = _finite_row(
                native["proposed_action"],
                (len(self.world.bodies), 8),
                "native proposed action",
            )
            actions, oral = self._actions(proposed, native["oral_command"])
            self._record_cognition(native, proposed)
            for body in self.world.bodies:
                self.neural_state[body.id] = response_by_body[body.id]
            cognition_done = time.perf_counter()
            self.pending_step["phase"] = "physics"
            self.outcomes = self.world.advance(actions, MODEL_DT)
            physics_done = time.perf_counter()
            self.actual_previous = np.ascontiguousarray(
                np.concatenate((proposed, oral[:, None]), axis=1), dtype=np.float32
            )
            self.reset_rows.fill(False)
            self._record_execution(proposed, oral)
            if self.acoustics is not None:
                self.acoustic_state = self.acoustics.advance(MODEL_DT)
            acoustics_done = time.perf_counter()
            if self.resources is not None:
                self.resource_state = self.resources.advance(MODEL_DT)
            resources_done = time.perf_counter()
            if self.biosphere is not None:
                self.biosphere.advance(MODEL_DT)
            biosphere_done = time.perf_counter()
            if self.field is not None:
                self.field.sync_static_geometry(self.world)
                self.field.sync_dynamic_barriers(self.world.diffusion_barriers())
                self.field.advance(
                    MODEL_DT,
                    sources=self.field.sources_from_world(self.world)
                    + (
                        self.biosphere.field_sources()
                        if self.biosphere is not None
                        else []
                    ),
                )
            fields_done = time.perf_counter()
            self.pending_step["phase"] = "personal-consequences"
            self.residents.observe_consequences(
                np.full(len(self.world.bodies), self.tick, dtype=np.uint64),
                physiology,
                self._physiology_rows(response_by_body),
                self.actual_previous,
            )
            personal_done = time.perf_counter()
            self.phase_timings.append(
                {
                    "senses": (sensed_done - started) * 1000,
                    "neural": (neural_done - sensed_done) * 1000,
                    "cognition": (cognition_done - neural_done) * 1000,
                    "physics": (physics_done - cognition_done) * 1000,
                    "acoustics": (acoustics_done - physics_done) * 1000,
                    "resources": (resources_done - acoustics_done) * 1000,
                    "biosphere": (biosphere_done - resources_done) * 1000,
                    "fields": (fields_done - biosphere_done) * 1000,
                    "personal_learning": (personal_done - fields_done) * 1000,
                }
            )
            self.tick += 1
            self.pending_step = None
            for body in self.world.bodies:
                if (
                    self.outcomes[body.id].get("nutrition", 0.0) > 0.0
                    and self.tick % 20 == 0
                ):
                    self.note(
                        "feeding", f"{body.name} ingested a resource.", resident=body.id
                    )
                if self.tick % 10 == 0:
                    self.history[body.id].append(
                        {
                            "time": self.world.time,
                            "x": float(body.x),
                            "y": float(body.y),
                            "z": float(body.z),
                            "energy": body.energy,
                            "activity": self.neural_state[body.id]["activity"],
                            "memory": self.memory_count(body.id),
                        }
                    )
            self.timings.append((time.perf_counter() - started) * 1000)

    def command(self, command: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(command, dict):
            raise TypeError("Command must be an object")
        op = command.get("op")
        if op == "pause":
            if type(command.get("paused")) is not bool:
                raise ValueError("paused must be boolean")
            self.paused = command["paused"]
            return {"paused": self.paused}
        if op == "speed":
            if type(command.get("value")) is not int or command["value"] not in {
                1,
                2,
                4,
            }:
                raise ValueError("speed must be 1, 2 or 4")
            self.speed = command["value"]
            return {"speed": self.speed}
        if op == "bookmark":
            text = command.get("text", "A moment in the garden.")
            if not isinstance(text, str) or len(text) > 500:
                raise ValueError("Bookmark text must be at most 500 characters")
            self.note("observation", text, origin="caregiver")
            return {"bookmarked": True}
        if op == "offer_material":
            if self.visitor_materials is None:
                raise ValueError("This world has no outside material supply")
            result = self.visitor_materials.command(command)
            self.note(
                "caregiver",
                "A visitor placed a finite material offering.",
                command=copy.deepcopy(command),
                transfer=copy.deepcopy(result),
            )
            return result
        result = self.world.command(command)
        self.visitor.direct_command(command)
        self.note(
            "caregiver", f"Outside interaction: {op}.", command=copy.deepcopy(command)
        )
        return result

    def _public_senses(self) -> dict[str, Any]:
        from .sensorium import serialize_rich_retina

        source = self.last_senses or self.sense()
        return {key: serialize_rich_retina(value) for key, value in source.items()}

    def view(self) -> dict[str, Any]:
        view = self.world.view()
        view.update(
            {
                "id": self.id,
                "name": self.world.spec.get("name", "The hollow garden"),
                "available_presets": sorted(self.world.spec.get("presets", {})),
                "tick": self.tick,
                "branch": self.branch,
                "paused": self.paused,
                "speed": self.speed,
                "saved_at": self.saved_at,
                "error": self.error,
                "neural": copy.deepcopy(self.neural_state),
                "cognition": self.cognitive_view(),
                "outcomes": copy.deepcopy(self.outcomes),
                "senses": self._public_senses(),
                "sensed_at": self.sensed_at,
                "ecology": (
                    {
                        "kind": "diffusion",
                        "channels": self.field.channels,
                        "time": self.field.time,
                    }
                    if self.field is not None
                    else {"kind": "analytic"}
                ),
                "resources": copy.deepcopy(self.resource_state),
                "biosphere": self._biosphere_view(),
                "acoustics": copy.deepcopy(self.acoustic_state),
                "visitor": self.visitor.view(self.tick, self.paused),
                "visitor_materials": (
                    self.visitor_materials.view()
                    if self.visitor_materials is not None
                    else None
                ),
                "journal": list(self.journal)[-40:],
                "history": {key: list(value) for key, value in self.history.items()},
                "anatomy": {
                    "dataset": self.neural.metadata["brain"].get("dataset", "unknown"),
                    "neurons": self.neural.graph["neurons"],
                    "connections": self.neural.graph["edges"],
                    "sha256": self.neural.graph["sha256"],
                    "scope": self.neural.metadata["brain"].get(
                        "scope", "configured neural graph"
                    ),
                    "inputs": len(self.neural.input_names),
                    "readouts": len(self.neural.output_names),
                },
                "resident_controller": {
                    **copy.deepcopy(self.residents.model_identity),
                    "neural_contract": copy.deepcopy(self.residents.neural_contract),
                    "observation_contract": copy.deepcopy(
                        self.residents.observation_contract
                    ),
                    "rich_retina_available": hasattr(self.world, "rich_retina_batch"),
                },
                "engine_identity": copy.deepcopy(self.engine_identity),
                "performance": {
                    "step_ms": sum(self.timings) / max(1, len(self.timings)),
                    "dt": MODEL_DT,
                    "physics_backend": self.physics_backend,
                    "phase_ms": (
                        {
                            key: sum(row[key] for row in self.phase_timings)
                            / len(self.phase_timings)
                            for key in self.phase_timings[0]
                        }
                        if self.phase_timings
                        else {}
                    ),
                },
            }
        )
        return view

    def _biosphere_view(self) -> dict[str, Any] | None:
        if self.biosphere is None:
            return None
        report = self.biosphere.last_report
        developments = report.get("developments", [])
        return {
            "kind": "native-metabolism-development-v1",
            "config_sha256": self.biosphere.config_sha256,
            "time": report.get("time", self.biosphere.web.time),
            "colonies": len(self.biosphere.config),
            "active_colonies": sum(self.biosphere.active.values()),
            "parts": report.get("parts", len(self.biosphere.parts)),
            "captured_photons": report.get("captured_photons", 0.0),
            "illumination": copy.deepcopy(report.get("illumination", {})),
            "illumination_sha256": self.biosphere.illumination_sha256,
            "mobile_phototrophy": copy.deepcopy(report.get("mobile_phototrophy", [])),
            "mobile_phototrophy_sha256": self.biosphere.mobile_photo_sha256,
            "accounting": copy.deepcopy(
                report.get("accounting", self.biosphere.accounting())
            ),
            "developments": copy.deepcopy(developments[-16:]),
            "developments_truncated": max(0, len(developments) - 16),
            "resident_physiology_coupled": self.biosphere.mobility is not None,
            "mobile_physiology": (
                self.biosphere.mobility.view()
                if self.biosphere.mobility is not None
                else None
            ),
            "whole_food_web": False,
            "exchange": (
                self.biosphere.exchange.view()
                if getattr(self.biosphere, "exchange", None) is not None
                else None
            ),
        }

    def save(self, path: str | Path) -> str:
        if self.pending_step is not None:
            raise RuntimeError("Cannot checkpoint an incomplete distributed tick")
        state = {
            "version": 1,
            "kind": "chreatures-developmental-habitat",
            "id": self.id,
            "tick": self.tick,
            "branch": self.branch,
            "paused": self.paused,
            "speed": self.speed,
            "world": self.world.snapshot(),
            "body_mode": self.body_mode,
            "physics_backend": self.physics_backend,
            "engine_identity": copy.deepcopy(self.engine_identity),
            "execution_migrations": self.execution_migrations,
            "field": self.field.snapshot() if self.field is not None else None,
            "resources": self.resources.snapshot()
            if self.resources is not None
            else None,
            "resource_state": self.resource_state,
            "biosphere": self.biosphere.snapshot()
            if self.biosphere is not None
            else None,
            "acoustics": self.acoustics.snapshot()
            if self.acoustics is not None
            else None,
            "acoustic_state": self.acoustic_state,
            "visitor": self.visitor.snapshot(),
            "last_senses": self._public_senses() if self.last_senses else {},
            "sensed_at": self.sensed_at,
            "neural_identity": copy.deepcopy(self.neural.metadata["brain"]),
            "remote_ids": self.remote_ids,
            "brain_url": self.neural.url,
            "neural_state": self.neural_state,
            "outcomes": self.outcomes,
            "resident_controller": self.residents.snapshot_value(),
            "actual_previous_plus_oral": self.actual_previous.astype(float).tolist(),
            "reset_rows": self.reset_rows.astype(bool).tolist(),
            "cognition_state": self.cognition_state,
            "journal": list(self.journal),
            "history": {key: list(value) for key, value in self.history.items()},
        }
        if self.visitor_materials is not None:
            state["visitor_materials"] = self.visitor_materials.snapshot()
        request = {
            "name": f"world-{self.id}-{self.tick}",
            "resident_ids": list(self.remote_ids.values()),
            "seq": self.neural.next_seq,
            "service_incarnation": self.neural.service_incarnation,
        }
        try:
            state["neural_snapshot"] = self.neural.snapshot(
                request["name"], request["resident_ids"]
            )
        except Exception as error:
            destination = Path(path)
            interrupted = destination.with_name(
                f"{destination.stem}.interrupted-{self.tick}.json"
            )
            write_envelope(
                interrupted,
                {"world": state, "neural_request": request, "error": str(error)},
                format=INTERRUPTED_FORMAT,
            )
            raise
        digest = write_envelope(path, state, format=CHECKPOINT_FORMAT)
        self.saved_at = time.time()
        return digest

    @classmethod
    def load(
        cls,
        path: str | Path,
        brain_url: str | None = None,
        resident_artifact: str | Path | None = None,
    ) -> Habitat3D:
        from .acoustics import Acoustics
        from .ecology import Ecology
        from .fields import FieldEnvironment

        envelope = json.loads(Path(path).read_text())
        if envelope.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                "Unsupported 3D checkpoint; current lives require a developmental habitat checkpoint"
            )
        value = envelope.get("state")
        if not isinstance(value, dict) or hashlib.sha256(
            canonical(value)
        ).hexdigest() != envelope.get("sha256"):
            raise ValueError("3D checkpoint checksum mismatch")
        if (
            value.get("version") != 1
            or value.get("kind") != "chreatures-developmental-habitat"
        ):
            raise ValueError("Unsupported developmental habitat state")
        if resident_artifact is None:
            raise ValueError(
                "A current --resident-artifact is required to restore this life"
            )
        instance = cls.__new__(cls)
        instance.engine_identity = current_engine_identity()
        if value.get("engine_identity") != instance.engine_identity:
            raise ValueError(
                "This life requires its pinned source, native binaries and runtime; "
                "restore it from its deployment directory"
            )
        instance.body_mode = value["body_mode"]
        instance.physics_backend = value["physics_backend"]
        instance.execution_migrations = copy.deepcopy(value["execution_migrations"])
        world_type = physical_world_type(instance.body_mode, instance.physics_backend)
        instance.world = world_type.restore(value["world"])
        instance.visitor = VisitorPerformances.restore(value["visitor"], instance.world)
        instance.field = (
            FieldEnvironment.restore(value["field"])
            if value["field"] is not None
            else None
        )
        if value["resources"] is not None and value["biosphere"] is not None:
            raise ValueError("Saved world contains conflicting ecological mechanisms")
        instance.resources = (
            Ecology.restore(instance.world, value["resources"])
            if value["resources"] is not None
            else None
        )
        instance.resource_state = copy.deepcopy(value["resource_state"])
        instance.biosphere = None
        if value["biosphere"] is not None:
            from .biosphere import Biosphere

            instance.biosphere = Biosphere.restore(instance.world, value["biosphere"])
        instance.visitor_materials = None
        if value.get("visitor_materials") is not None:
            from .visitor_materials import VisitorMaterialSupply

            if instance.biosphere is None:
                raise ValueError("Saved offerings lack their shared biosphere")
            instance.visitor_materials = VisitorMaterialSupply.restore(
                instance.biosphere, value["visitor_materials"]
            )
        instance.acoustics = (
            Acoustics.restore(instance.world, value["acoustics"])
            if value["acoustics"] is not None
            else None
        )
        instance.acoustic_state = copy.deepcopy(value["acoustic_state"])
        instance.last_senses = copy.deepcopy(value["last_senses"])
        instance.sensed_at = float(value["sensed_at"])
        instance.neural = NeuralClient(brain_url or value["brain_url"])
        instance._validate_neural_interface()
        if instance.neural.metadata["brain"] != value["neural_identity"]:
            raise ValueError(
                "Remote neural graph, sources, or ports differ from this life"
            )
        instance.resident_artifact = str(Path(resident_artifact).expanduser().resolve())
        instance.residents = DevelopmentalResidentCohort.restore_value(
            value["resident_controller"], instance.resident_artifact
        )
        instance._validate_resident_interface()
        cohort_size = len(instance.world.bodies)
        if instance.residents.batch_size != cohort_size:
            raise ValueError(
                "Resident controller cohort differs from the physical cohort"
            )
        instance.actual_previous = _finite_row(
            value["actual_previous_plus_oral"],
            (cohort_size, 9),
            "saved previous action",
        )
        instance.reset_rows = np.asarray(value["reset_rows"], dtype=np.bool_)
        if instance.reset_rows.shape != (cohort_size,):
            raise ValueError("Saved reset boundary differs from the physical cohort")
        instance.neural.restore(value["neural_snapshot"])
        for key in (
            "id",
            "tick",
            "branch",
            "paused",
            "speed",
            "remote_ids",
            "neural_state",
            "outcomes",
            "cognition_state",
        ):
            setattr(instance, key, copy.deepcopy(value[key]))
        expected_ids = [body.id for body in instance.world.bodies]
        if set(instance.remote_ids) != set(expected_ids) or set(
            instance.cognition_state
        ) != set(expected_ids):
            raise ValueError(
                "Saved resident identities differ from the physical cohort"
            )
        instance.remote_ids = {
            body_id: value["remote_ids"][body_id] for body_id in expected_ids
        }
        instance.journal = deque(value["journal"], maxlen=256)
        instance.history = {
            body_id: deque(value["history"][body_id], maxlen=360)
            for body_id in expected_ids
        }
        instance.timings = deque(maxlen=120)
        instance.phase_timings = deque(maxlen=120)
        instance.error = None
        instance.pending_step = None
        instance.saved_at = time.time()
        return instance
