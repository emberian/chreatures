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
from .organism_interface import (
    ACTION_NAMES, ACTION_DIM, PHYSIOLOGY_DIM, NEURAL_DIM,
    OBSERVATION_DIM, OBSERVATION_ORDER, PREVIOUS_DIM, RECTIFIED_AXES,
    identity as organism_identity,
)
SOURCE_SENSE_DIM = 351
RICH_RETINA_DIM = 4096
CHECKPOINT_FORMAT = "chreatures-developmental-habitat-checkpoint-v3"
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


def _neural_model_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    # Occupancy belongs to the neural snapshot, not to the executable model.
    # Metadata is read before creating a new cohort and after it exists on reload.
    return copy.deepcopy(
        {key: value for key, value in metadata.items() if key != "residents"}
    )


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
        population_birth: str | Path | None = None,
        population_response_artifact: str | Path | None = None,
    ) -> None:
        from .acoustics import Acoustics
        from .ecology import Ecology
        from .fields import FieldEnvironment

        if resident_artifact is None:
            raise ValueError("A current --resident-artifact is required for a new life")
        if population_birth is None or biosphere is None:
            raise ValueError("A new population life requires --population-birth and --biosphere")
        from .resident_birth import (
            load_manifest, verify_controller, candidate_adapters, inherited_body_templates,
        )
        from .population import compose_population_birth

        self.birth_manifest = load_manifest(population_birth)
        verify_controller(self.birth_manifest, resident_artifact)
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
        spec, biosphere_value, birth_receipt = compose_population_birth(
            spec, json.loads(Path(biosphere).read_text()),
            [row["candidate"] for row in self.birth_manifest["residents"]],
        )
        spec["population_birth"] = birth_receipt
        self.birth_templates = inherited_body_templates(spec, biosphere_value)
        self.birth_retry_tick = 0
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

            self.biosphere = Biosphere.from_config(self.world, biosphere_value)
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
            candidate_adapters=candidate_adapters(self.birth_manifest),
            population_response_artifact=population_response_artifact,
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
        self.neural.create([
            {"id": self.remote_ids[body.id], "neural_phenotype": entry["neural_phenotype"]}
            for body, entry in zip(self.world.bodies, self.birth_manifest["residents"], strict=True)
        ])
        self.actual_previous = np.zeros((cohort_size, PREVIOUS_DIM), dtype=np.float32)
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
        self.journal_sequence = 0
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
        if trained_neural["graph_sha256"] != self.neural.graph[
            "sha256"
        ] or trained_neural["port_spec_sha256"] != remote_ports.get("spec_hash"):
            raise ValueError(
                "Resident artifact was trained with a different graph or neural port"
            )
        expected = organism_identity()
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
                "generation": 0,
                "remaining_ticks": 0,
            },
            "sampled_proposal": {name: 0.0 for name in ACTION_NAMES},
            "executed_action": {
                **{name: 0.0 for name in ACTION_NAMES},
            },
            "outcome": {},
            "personal_goal_learning": {
                "selected_bias": 0.0,
                "prediction": 0.0,
                "last_transition_reward": 0.0,
                "last_completed_return": 0.0,
                "completed": False,
                "attributed": False,
                "learned": False,
                "completed_total": 0,
                "learned_total": 0,
                "frozen_total": 0,
                "skipped_replaced_total": 0,
                "cancelled_total": 0,
                "learning_enabled": True,
            },
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
        self.journal_sequence += 1
        self.journal.append(
            {
                **fields,
                "id": f"{self.id}:{self.journal_sequence}",
                "sequence": self.journal_sequence,
                "tick": self.tick,
                "time": self.world.time,
                "kind": kind,
                "text": text,
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
        if self.biosphere is None or self.biosphere.mobility is None:
            raise RuntimeError("Current residents require funded developmental physiology")
        rows = [
            self.biosphere.mobility.normalized12(
                body.id, neural_support=responses[body.id]["support"]
            )
            for body in self.world.bodies
        ]
        result = _finite_row(rows, (len(rows), PHYSIOLOGY_DIM), "physiology")
        unsigned = result[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]]
        if np.any((unsigned < 0.0) | (unsigned > 1.0)) or np.any(abs(result[:, 3:5]) > 1):
            raise ValueError("Physiology exceeds its declared range")
        return result

    def _attempt_brood_birth(self) -> None:
        """Append one funded clonal offspring at a complete world boundary.

        The present within-world reproduction law is asexual. Population search
        also varies genomes, but an in-world birth never consults its archive.
        """
        from .organism_interface import MAX_RESIDENTS
        from .population import CandidateGenome
        from .physics import BirthSpaceOccupied

        if self.pending_step is not None or self.tick < self.birth_retry_tick:
            return
        self.birth_retry_tick = self.tick + 20
        if len(self.world.bodies) >= MAX_RESIDENTS:
            return
        offers = self.biosphere.hatch_offers()
        if not offers:
            return
        offer = offers[0]
        parent_id = offer["parent_id"]
        parent_index = next(index for index, body in enumerate(self.world.bodies) if body.id == parent_id)
        parent = self.world.bodies[parent_index]
        inherited = copy.deepcopy(self.birth_manifest["residents"][parent_index])
        child_id = "hatch-" + hashlib.sha256(offer["offer_id"].encode()).hexdigest()[:16]
        bundle = copy.deepcopy(self.birth_templates[parent_id])
        body = bundle["body"]
        body.update(id=child_id, name=f"{parent.name} offspring {offer['serial']}")
        # This is a declared local emergence rule. Collision checks determine
        # whether the parent's immediate surroundings can accommodate the body.
        transaction = None
        separation = max(0.35, 4.0 * float(parent.radius))
        for offset in range(8):
            angle = float(parent.heading) + offset * math.pi / 4
            body["position"] = [
                parent.x + separation * math.cos(angle),
                parent.y + separation * math.sin(angle), parent.z + 0.08,
            ]
            body["heading"] = float(parent.heading)
            try:
                transaction = self.world.prepare_resident_birth(body)
                break
            except BirthSpaceOccupied:
                continue
        if transaction is None:
            return
        proposal = self.biosphere.prepare_hatch(offer["offer_id"], inherited["candidate"])
        prepared = self.biosphere.prepare_newborn(transaction.candidate, proposal, bundle)
        seed = int(hashlib.sha256(f"{self.id}:{offer['offer_id']}".encode()).hexdigest()[:16], 16)
        expanded = self.residents.expanded(
            [CandidateGenome(inherited["candidate"]).controller_adapter()],
            goal_seed=seed, action_seed=seed ^ 0xAC7100,
        )
        remote_id = f"{self.id}:{child_id}"
        self.pending_step = {"tick": self.tick, "phase": "birth-neural", "offer_id": offer["offer_id"], "resident_id": child_id}
        self.neural.create([{"id": remote_id, "neural_phenotype": inherited["neural_phenotype"]}])
        self.pending_step["phase"] = "birth-physical"
        transaction.commit()
        self.biosphere = self.biosphere.commit_newborn(prepared)
        self.residents = expanded
        self.remote_ids[child_id] = remote_id
        self.birth_manifest["residents"].append(inherited)
        self.birth_templates[child_id] = copy.deepcopy(bundle)
        self.actual_previous = np.concatenate((self.actual_previous, np.zeros((1, PREVIOUS_DIM), dtype=np.float32)))
        self.reset_rows = np.concatenate((self.reset_rows, np.ones(1, dtype=np.bool_)))
        self.outcomes[child_id] = {}
        self.neural_state[child_id] = {"features": [0.0] * NEURAL_DIM, "activity": 0.0, "support": 1.0}
        self.cognition_state[child_id] = self._empty_cognition(child_id)
        self.history[child_id] = deque(maxlen=360)
        if self.visitor_materials is not None:
            self.visitor_materials.biosphere = self.biosphere
        self.pending_step = None
        self.note("born", f"A funded offspring of {parent.name} entered the world.",
                  resident=child_id, parent=parent_id, funding=prepared.funding,
                  genome_sha256=inherited["candidate"]["sha256"])

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

    def _actions(self, proposed: np.ndarray) -> dict[str, dict[str, float]]:
        values = _finite_row(proposed, (len(self.world.bodies), ACTION_DIM), "native action")
        if np.any((values < -1.0) | (values > 1.0)) or np.any(values[:, RECTIFIED_AXES] < 0):
            raise ValueError("Native action exceeds the physical actuator bounds")
        return {
            body.id: dict(zip(ACTION_NAMES, values[index].astype(float).tolist(), strict=True))
            for index, body in enumerate(self.world.bodies)
        }

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
                    "generation": int(result["goal_generation"][index]),
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
                "sensory_forecast": {
                    "candidate_progress": result["forecast_progress"][index]
                    .astype(float)
                    .tolist(),
                    "candidate_disagreement": result["forecast_disagreement"][index]
                    .astype(float)
                    .tolist(),
                    "candidate_input_clipped": result["forecast_input_clipped"][index]
                    .astype(bool)
                    .tolist(),
                    "candidate_logit_tilt": result["forecast_tilt"][index]
                    .astype(float)
                    .tolist(),
                    "empirical_goal_error_scale": float(result["forecast_goal_rms"]),
                    "meaning": "one-step predicted progress toward an achieved sensory goal; disagreement is not calibrated confidence",
                },
                "personal_goal_learning": {
                    "selected_bias": float(
                        result["personal_goal_selected_bias"][index]
                    ),
                    "prediction": float(result["personal_goal_prediction"][index]),
                    "last_transition_reward": float(
                        result["personal_goal_last_reward"][index]
                    ),
                    "last_completed_return": float(
                        result["personal_goal_last_return"][index]
                    ),
                    "completed": bool(result["personal_goal_completed"][index]),
                    "attributed": bool(result["personal_goal_attributed"][index]),
                    "learned": bool(result["personal_goal_learned"][index]),
                    "completed_total": int(
                        result["personal_goal_completed_total"][index]
                    ),
                    "learned_total": int(result["personal_goal_learned_total"][index]),
                    "frozen_total": int(result["personal_goal_frozen_total"][index]),
                    "skipped_replaced_total": int(
                        result["personal_goal_skipped_total"][index]
                    ),
                    "cancelled_total": int(
                        result["personal_goal_cancelled_total"][index]
                    ),
                    "learning_enabled": bool(result["personal_goal_learning_enabled"]),
                    "meaning": "private association between selected achieved goals and actual finite-energy transition returns",
                },
                "model_identity": copy.deepcopy(self.residents.model_identity),
            }

    def _record_execution(self, actions: np.ndarray) -> None:
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
            }
            self.cognition_state[body.id]["outcome"] = copy.deepcopy(
                self.outcomes[body.id]
            )

    def _record_goal_receipts(self, receipts: dict[str, np.ndarray]) -> None:
        for index, body in enumerate(self.world.bodies):
            state = self.cognition_state[body.id]["personal_goal_learning"]
            state["last_transition_reward"] = float(receipts["reward"][index])
            completed = bool(receipts["completed"][index])
            state["completed"] = completed
            state["attributed"] = bool(receipts["attributed"][index])
            state["learned"] = bool(receipts["learned"][index])
            state["completed_total"] = int(receipts["completed_total"][index])
            state["learned_total"] = int(receipts["learned_total"][index])
            state["frozen_total"] = int(receipts["frozen_total"][index])
            state["skipped_replaced_total"] = int(receipts["skipped_total"][index])
            state["cancelled_total"] = int(receipts["cancelled_total"][index])
            if completed:
                state["last_completed_return"] = float(receipts["summed_return"][index])
            if self.residents.population_response_artifact is not None:
                self.cognition_state[body.id]["population_response"] = {
                    "artifact_sha256": str(receipts["population_response_identity"]),
                    "feature_contract_sha256": str(
                        receipts["population_feature_contract_identity"]
                    ),
                    "executed_transition_in_domain": bool(
                        receipts["population_response_in_domain"][index]
                    ),
                    "in_domain_total": int(
                        receipts["population_response_in_domain_total"][index]
                    ),
                    "out_of_domain_total": int(
                        receipts["population_response_out_of_domain_total"][index]
                    ),
                    "meaning": "fitted-domain coverage of committed actions; not evidence that the model changed the selected action",
                }

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
                native["actual_previous_action"], self.actual_previous
            ):
                raise RuntimeError("Native resident previous-action accounting differs")
            if not np.array_equal(native["physiology"], physiology):
                raise RuntimeError("Native resident physiology accounting differs")
            proposed = _finite_row(
                native["proposed_action"],
                (len(self.world.bodies), ACTION_DIM),
                "native proposed action",
            )
            actions = self._actions(proposed)
            self._record_cognition(native, proposed)
            for body in self.world.bodies:
                self.neural_state[body.id] = response_by_body[body.id]
            cognition_done = time.perf_counter()
            self.pending_step["phase"] = "physics"
            self.outcomes = self.world.advance(actions, MODEL_DT)
            physics_done = time.perf_counter()
            self.actual_previous = np.ascontiguousarray(
                proposed, dtype=np.float32
            )
            self.reset_rows.fill(False)
            self._record_execution(proposed)
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
            goal_receipts = self.residents.observe_consequences(
                np.full(len(self.world.bodies), self.tick, dtype=np.uint64),
                physiology,
                self._physiology_rows(response_by_body),
                self.actual_previous,
                np.asarray(
                    [self.outcomes[body.id]["effort"] for body in self.world.bodies],
                    dtype=np.float32,
                ),
                dt=MODEL_DT,
            )
            self._record_goal_receipts(goal_receipts)
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
            self._attempt_brood_birth()
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
        if op == "personal_goal_learning":
            if type(command.get("enabled")) is not bool:
                raise ValueError("personal goal learning enabled must be boolean")
            enabled = command["enabled"]
            self.residents.set_personal_goal_learning(enabled)
            self.note(
                "research-intervention",
                "Private goal association learning changed.",
                personal_goal_learning_enabled=enabled,
            )
            return {"personal_goal_learning_enabled": enabled}
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
            "version": 3,
            "kind": "chreatures-developmental-habitat",
            "id": self.id,
            "birth_manifest": copy.deepcopy(self.birth_manifest),
            "birth_templates": copy.deepcopy(self.birth_templates),
            "birth_retry_tick": self.birth_retry_tick,
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
            "neural_identity": _neural_model_identity(self.neural.metadata["brain"]),
            "remote_ids": self.remote_ids,
            "brain_url": self.neural.url,
            "neural_state": self.neural_state,
            "outcomes": self.outcomes,
            "resident_controller": self.residents.snapshot_value(),
            "actual_previous": self.actual_previous.astype(float).tolist(),
            "reset_rows": self.reset_rows.astype(bool).tolist(),
            "cognition_state": self.cognition_state,
            "journal": list(self.journal),
            "journal_sequence": self.journal_sequence,
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
        population_response_artifact: str | Path | None = None,
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
            value.get("version") != 3
            or value.get("kind") != "chreatures-developmental-habitat"
        ):
            raise ValueError("Unsupported developmental habitat state")
        journal_sequence = value.get("journal_sequence")
        journal = value.get("journal")
        if (
            type(journal_sequence) is not int
            or journal_sequence < 0
            or not isinstance(journal, list)
            or len(journal) > 256
        ):
            raise ValueError("Invalid saved journal sequence")
        previous_sequence = -1
        for event in journal:
            sequence = event.get("sequence")
            if (
                type(sequence) is not int
                or not previous_sequence < sequence <= journal_sequence
                or event.get("id") != f"{value['id']}:{sequence}"
            ):
                raise ValueError("Saved journal identities differ from their sequence")
            previous_sequence = sequence
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
        if (
            _neural_model_identity(instance.neural.metadata["brain"])
            != value["neural_identity"]
        ):
            raise ValueError(
                "Remote neural graph, sources, or ports differ from this life"
            )
        instance.resident_artifact = str(Path(resident_artifact).expanduser().resolve())
        from .resident_birth import validate_manifest, verify_controller

        instance.birth_manifest = validate_manifest(value["birth_manifest"])
        verify_controller(instance.birth_manifest, instance.resident_artifact)
        instance.birth_templates = copy.deepcopy(value["birth_templates"])
        instance.birth_retry_tick = int(value["birth_retry_tick"])
        if set(instance.birth_templates) != {body.id for body in instance.world.bodies}:
            raise ValueError("saved birth templates differ from the resident population")
        instance.residents = DevelopmentalResidentCohort.restore_value(
            value["resident_controller"], instance.resident_artifact,
            population_response_artifact=population_response_artifact,
        )
        instance._validate_resident_interface()
        cohort_size = len(instance.world.bodies)
        if instance.residents.batch_size != cohort_size:
            raise ValueError(
                "Resident controller cohort differs from the physical cohort"
            )
        instance.actual_previous = _finite_row(
            value["actual_previous"],
            (cohort_size, PREVIOUS_DIM),
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
        instance.journal_sequence = journal_sequence
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
