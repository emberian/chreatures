"""A whole 3D world coupled to full MaleCNS state and personal learning organs.

The physical world is local; the full curated nervous systems persist on AMD.
A checkpoint references a checksum-verified server-side neural artifact and
contains physical integration state plus every personal cognitive parameter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np

from .cognition import AdaptiveOrgan
from .neural_client import NeuralClient
from .runtime import canonical
from .visitor_events import VisitorPerformances


def physical_world_type(body_mode, execution):
    from .sensorium import ArticulatedSensoriumWorld, SensoriumWorld

    if execution == "vectorized" and body_mode == "articulated":
        from .physical_batch import FastArticulatedSensoriumWorld

        return FastArticulatedSensoriumWorld
    if execution != "reference":
        raise ValueError("Vectorized physics requires an articulated body")
    return ArticulatedSensoriumWorld if body_mode == "articulated" else SensoriumWorld


class Habitat3D:
    def __init__(
        self,
        seed=7,
        brain_url="http://127.0.0.1:18765",
        spec=None,
        body_mode="articulated",
        ecology="diffusion",
        resources=None,
        biosphere=None,
        acoustics=None,
        motor_genome=None,
        personal_memory=False,
        perception_url=None,
        physics_backend=None,
        personal_plasticity=False,
    ):
        from .acoustics import Acoustics
        from .ecology import Ecology
        from .fields import FieldEnvironment

        if body_mode not in ("crawler", "articulated") or ecology not in (
            "analytic",
            "diffusion",
        ):
            raise ValueError("Unknown body or ecology model")
        if personal_memory and motor_genome is None:
            raise ValueError(
                "Personal contextual motor learning requires an inherited motor artifact"
            )
        if perception_url is not None and not personal_memory:
            raise ValueError("Native visual episodes require the personal motor organ")
        if personal_plasticity and not personal_memory:
            raise ValueError(
                "Personal motor plasticity requires a personal memory organ"
            )
        if resources is not None and biosphere is not None:
            raise ValueError(
                "Legacy resources and the developmental biosphere are mutually exclusive"
            )
        self.personal_memory = bool(personal_memory)
        self.personal_plasticity = bool(personal_plasticity)
        if spec is None:
            spec = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "data/habitats/hollow-garden.json"
                ).read_text()
            )
        # This constructor creates a new life. Restoring an older snapshot
        # follows its saved selector, including legacy worlds with no selector.
        spec = copy.deepcopy(spec)
        spec.setdefault("sensorium", {"frame": "body-v1"})
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
        self.acoustics = (
            Acoustics(self.world, acoustics) if acoustics is not None else None
        )
        self.acoustic_state = None
        self.last_senses = {}
        self.sensed_at = 0.0
        self.neural = NeuralClient(brain_url)
        self.motor_artifact = None
        self.motors = {}
        if motor_genome is not None:
            from .motor_inheritance import MotorArtifact, MotorOrgan

            self.motor_artifact = MotorArtifact.load(motor_genome)
            self._validate_motor_interface()
            motor_type = MotorOrgan
            if self.personal_memory:
                from .living_motor import LivingMotorOrgan

                motor_type = LivingMotorOrgan
            for i, body in enumerate(self.world.bodies):
                private_seed = seed * 1009 + i
                options = (
                    {"plasticity": self.personal_plasticity}
                    if self.personal_memory
                    else {}
                )
                if self.personal_plasticity:
                    from .personal_plasticity import PersonalPlasticityConfig

                    # New lives can learn both motor means and exploration.
                    # Restore never consults this birth-time default.
                    options["plasticity_config"] = PersonalPlasticityConfig(
                        seed=private_seed + 3301,
                        variance_adaptation="state-log-std-v2",
                    )
                self.motors[body.id] = motor_type(
                    self.motor_artifact, seed=private_seed, **options
                )
        self.id = str(uuid.uuid4())
        self.tick = 0
        self.paused = False
        self.speed = 1
        self.branch = "resident"
        self.saved_at = None
        self.error = None
        self.remote_ids = {b.id: f"{self.id}:{b.id}" for b in self.world.bodies}
        self.neural.create(list(self.remote_ids.values()))
        self.organs = (
            {}
            if self.motors
            else {
                b.id: AdaptiveOrgan(
                    feature_dim=len(self.neural.output_names), seed=seed
                )
                for b in self.world.bodies
            }
        )
        for i, organ in enumerate(self.organs.values()):
            # Shared inherited maps; separate stochastic lives.
            organ.rng = np.random.default_rng(seed * 1009 + i)
        self.outcomes = {b.id: {} for b in self.world.bodies}
        self.neural_state = {
            b.id: {
                "features": [0.0] * len(self.neural.output_names),
                "activity": 0.0,
                "support": 1.0,
            }
            for b in self.world.bodies
        }
        self.feature_mean = {
            b.id: np.zeros(len(self.neural.output_names), dtype=np.float32)
            for b in self.world.bodies
        }
        self.feature_variance = {
            b.id: np.ones(len(self.neural.output_names), dtype=np.float32) * 0.01
            for b in self.world.bodies
        }
        self.journal = deque(maxlen=256)
        self.history = {b.id: deque(maxlen=360) for b in self.world.bodies}
        self.timings = deque(maxlen=120)
        self.phase_timings = deque(maxlen=120)
        self.pending_step = None
        self.visitor = VisitorPerformances()
        self.execution_migrations = []
        self.vision = None
        if perception_url is not None:
            from .embodied_vision import EmbodiedVision

            self.vision = EmbodiedVision(
                perception_url, [b.id for b in self.world.bodies]
            )
        self.note(
            "hatched",
            f"{len(self.world.bodies)} new residents entered the habitat with full MaleCNS circuits.",
        )
        if self.motor_artifact is not None:
            self.note(
                "inheritance",
                "Inherited a population-trained motor interface and private working context.",
                artifact_sha256=self.motor_artifact.sha256,
                training=self.motor_artifact.metadata["training_provenance"],
            )

    def _validate_motor_interface(self):
        artifact = self.motor_artifact
        if artifact is None:
            return
        provenance = artifact.metadata["training_provenance"]
        if (
            provenance["graph_sha256"] != self.neural.graph["sha256"]
            or provenance["port_spec_sha256"]
            != self.neural.metadata["brain"].get("ports", {}).get("spec_hash")
            or artifact.config["feature_dim"] != len(self.neural.output_names)
        ):
            raise ValueError(
                "Inherited motor anatomy or neural interface differs from this world"
            )

    def memory_count(self, body_id):
        if self.personal_memory:
            return self.motors[body_id].memory.transition_count
        return len(self.organs[body_id].memory.records) if body_id in self.organs else 0

    def cognitive_view(self):
        if not self.motors:
            return {key: organ.view() for key, organ in self.organs.items()}
        if self.personal_memory:
            return {
                key: {
                    **organ.view(),
                    "controller": "inherited-with-personal-context",
                    "time": self.world.time,
                    "memory_count": self.memory_count(key),
                    "learning_enabled": organ.refiner.learning,
                    "context": organ.motor.context.tolist(),
                    "metrics": {
                        **organ.metrics,
                        "prediction_error": organ.motor.last_prediction_error or 0.0,
                        "learning_progress": 0.0,
                    },
                }
                for key, organ in self.motors.items()
            }
        return {
            key: {
                **motor.view(),
                "controller": "inherited-predictive-ppo",
                "time": self.world.time,
                "memory_count": 0,
                "learning": False,
                "metrics": {
                    "prediction_error": motor.last_prediction_error or 0.0,
                    "learning_progress": 0.0,
                },
            }
            for key, motor in self.motors.items()
        }

    def sense(self):
        """Sample the body, including local concentration from physical transport."""
        sensed = {b.id: self.world.sense(b.id) for b in self.world.bodies}
        if self.field is not None:
            for body in self.world.bodies:
                values = sensed[body.id]
                positions = values.get("antenna_position")
                if positions is None:
                    # Geometry stays at the transducer boundary. Only the two
                    # measured chemical concentrations reach cognition.
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

    def note(self, kind, text, **fields):
        self.journal.append(
            {
                "id": f"{self.id}:{self.tick}:{len(self.journal)}",
                "time": self.world.time,
                "kind": kind,
                "text": text,
                **fields,
            }
        )

    def step(self, steps=1):
        dt = 0.05
        if self.pending_step is not None:
            raise RuntimeError(
                "A previous world step is incomplete; restore its checkpoint before advancing"
            )
        for _ in range(steps):
            started = time.perf_counter()
            if self.vision is not None and not self.vision.poll(self.tick):
                break
            for event in self.visitor.advance(self.world, self.tick):
                self.note(
                    "visitor-event", "A scheduled physical stimulus occurred.", **event
                )
            before_physiology = (
                self.vision.physiology(self.world, self.neural_state)
                if self.vision is not None
                else None
            )
            sensed = self.sense()
            self.last_senses = sensed
            self.sensed_at = self.world.time
            entries = [
                {
                    "id": self.remote_ids[b.id],
                    "senses": self.neural.encode(sensed[b.id]),
                }
                for b in self.world.bodies
            ]
            sensed_done = time.perf_counter()
            self.pending_step = {"tick": self.tick, "neural_seq": self.neural.next_seq}
            responses = self.neural.step(entries, dt)
            neural_done = time.perf_counter()
            inverse = {v: k for k, v in self.remote_ids.items()}
            actions = {}
            for response in responses:
                body_id = inverse[response["id"]]
                body = next(b for b in self.world.bodies if b.id == body_id)
                features = np.asarray(response["features"], dtype=np.float32)
                local_body = {
                    key: float(getattr(body, key))
                    for key in ("energy", "gut", "fatigue", "speed", "angular_velocity")
                }
                local_body["support"] = float(response["support"])
                if self.motors:
                    if self.personal_memory:
                        maturation = self.motors[body_id].variance_maturation
                        maturation_queued = (
                            maturation is not None and maturation["status"] == "queued"
                        )
                        action = self.motors[body_id].tick(
                            features,
                            local_body,
                            self.outcomes[body_id] if self.tick else None,
                            dt,
                            candidate_evidence=self.vision.candidate_evidence(
                                body_id,
                                self.tick,
                                local_body,
                                self.motors[body_id].refiner.config,
                            )
                            if self.vision is not None
                            else None,
                        )
                        if maturation_queued and maturation["status"] == "applied":
                            self.note(
                                "development",
                                f"{body.name} acquired a private capacity to learn action variance.",
                                resident=body_id,
                                maturation=copy.deepcopy(maturation),
                            )
                    else:
                        action = self.motors[body_id].tick(features, local_body, dt)
                else:
                    mean, variance = (
                        self.feature_mean[body_id],
                        self.feature_variance[body_id],
                    )
                    delta = features - mean
                    mean += np.float32(dt / 20) * delta
                    variance += np.float32(dt / 20) * (delta * delta - variance)
                    normalized = np.clip(
                        delta / np.sqrt(np.maximum(variance, 1e-6)), -2, 2
                    )
                    action = self.organs[body_id].step(
                        normalized, local_body, self.outcomes[body_id], dt
                    )
                for channel in ("grip", "signal_low", "signal_mid", "signal_high"):
                    action[channel] = max(0.0, action[channel])
                # Ingestion is an embodied contact reflex, not an object/position
                # query. Movement, looking, grip and signaling are learned outputs.
                action["eat"] = float(
                    np.clip((1 - body.gut) * (1.1 - body.energy), 0, 1)
                )
                actions[body_id] = action
                self.neural_state[body_id] = response
            cognition_done = time.perf_counter()
            if self.vision is not None:
                self.vision.begin_step(
                    self.world, self.tick, actions, before_physiology
                )
            self.outcomes = self.world.advance(actions, dt)
            physics_done = time.perf_counter()
            if self.acoustics is not None:
                self.acoustic_state = self.acoustics.advance(dt)
            acoustics_done = time.perf_counter()
            if self.resources is not None:
                self.resource_state = self.resources.advance(dt)
            resources_done = time.perf_counter()
            if self.biosphere is not None:
                self.biosphere.advance(dt)
            biosphere_done = time.perf_counter()
            if self.field is not None:
                self.field.sync_static_geometry(self.world)
                self.field.sync_dynamic_barriers(self.world.diffusion_barriers())
                self.field.advance(
                    dt, sources=self.field.sources_from_world(self.world)
                )
            fields_done = time.perf_counter()
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
                }
            )
            self.tick += 1
            if self.vision is not None:
                self.vision.finish_step(
                    self.world,
                    self.tick,
                    self.outcomes,
                    self.vision.physiology(self.world, self.neural_state),
                )
            self.pending_step = None
            for b in self.world.bodies:
                if self.outcomes[b.id].get("nutrition", 0) > 0 and self.tick % 20 == 0:
                    self.note(
                        "feeding", f"{b.name} ingested a resource.", resident=b.id
                    )
                if self.tick % 10 == 0:
                    self.history[b.id].append(
                        {
                            "time": self.world.time,
                            "x": float(b.x),
                            "y": float(b.y),
                            "z": float(b.z),
                            "energy": b.energy,
                            "activity": self.neural_state[b.id]["activity"],
                            "memory": self.memory_count(b.id),
                        }
                    )
            self.timings.append((time.perf_counter() - started) * 1000)

    def command(self, command):
        if not isinstance(command, dict):
            raise ValueError("Command must be an object")
        op = command.get("op")
        if op == "pause":
            if type(command.get("paused")) is not bool:
                raise ValueError("paused must be boolean")
            self.paused = command["paused"]
            return {"paused": self.paused}
        if op == "speed":
            if type(command.get("value")) is not int or command["value"] not in (
                1,
                2,
                4,
            ):
                raise ValueError("speed must be 1, 2 or 4")
            self.speed = command["value"]
            return {"speed": self.speed}
        if op == "bookmark":
            text = command.get("text", "A moment in the garden.")
            if not isinstance(text, str) or len(text) > 500:
                raise ValueError("Bookmark text must be at most 500 characters")
            self.note("observation", text, origin="caregiver")
            return {"bookmarked": True}
        if op == "mature_variance":
            if set(command) - {"op", "residents"}:
                raise ValueError("Unknown variance maturation option")
            if not self.paused or not self.personal_memory:
                raise ValueError(
                    "Variance maturation requires a paused personal-learning world"
                )
            ids = command.get("residents", list(self.motors))
            if (
                not isinstance(ids, list)
                or not ids
                or any(
                    not isinstance(key, str) or key not in self.motors for key in ids
                )
                or len(ids) != len(set(ids))
            ):
                raise ValueError("Residents must be distinct members of this world")
            # Check the complete cohort before queuing any developmental change.
            for key in ids:
                motor = self.motors[key]
                if (
                    motor.personal_plasticity is None
                    or motor.variance_maturation is not None
                    or motor.personal_plasticity.config.variance_adaptation
                    != "fixed-inherited-v1"
                ):
                    raise ValueError(f"Resident {key} cannot queue this maturation")
            records = {key: self.motors[key].queue_state_log_std_v2() for key in ids}
            self.note(
                "development",
                "Queued private variance learning after each current motor action completes.",
                residents=ids,
                maturation=copy.deepcopy(records),
            )
            return {"queued": records}
        result = self.world.command(command)
        self.visitor.direct_command(command)
        self.note(
            "caregiver", f"Outside interaction: {op}.", command=copy.deepcopy(command)
        )
        return result

    def view(self):
        view = self.world.view()
        view.update(
            {
                "id": self.id,
                "name": self.world.spec.get("name", "The hollow garden"),
                "tick": self.tick,
                "branch": self.branch,
                "paused": self.paused,
                "speed": self.speed,
                "saved_at": self.saved_at,
                "error": self.error,
                "neural": copy.deepcopy(self.neural_state),
                "cognition": self.cognitive_view(),
                "senses": copy.deepcopy(self.last_senses or self.sense()),
                "sensed_at": self.sensed_at,
                "ecology": {
                    "kind": "diffusion",
                    "channels": self.field.channels,
                    "time": self.field.time,
                }
                if self.field is not None
                else {"kind": "analytic"},
                "resources": copy.deepcopy(self.resource_state),
                "biosphere": self._biosphere_view(),
                "acoustics": copy.deepcopy(self.acoustic_state),
                "visitor": self.visitor.view(self.tick, self.paused),
                "vision": self.vision.view(self.tick)
                if self.vision is not None
                else None,
                "journal": list(self.journal)[-40:],
                "history": {k: list(v) for k, v in self.history.items()},
                "anatomy": {
                    "dataset": "MaleCNS v1.0",
                    "neurons": self.neural.graph["neurons"],
                    "connections": self.neural.graph["edges"],
                    "sha256": self.neural.graph["sha256"],
                    "scope": "full traced curated brain and nerve cord",
                    "inputs": len(self.neural.input_names),
                    "readouts": len(self.neural.output_names),
                },
                "performance": {
                    "step_ms": sum(self.timings) / max(1, len(self.timings)),
                    "dt": 0.05,
                    "physics_backend": self.physics_backend,
                    "phase_ms": {
                        key: sum(t[key] for t in self.phase_timings)
                        / len(self.phase_timings)
                        for key in self.phase_timings[0]
                    }
                    if self.phase_timings
                    else {},
                },
            }
        )
        return view

    def _biosphere_view(self):
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
            "accounting": copy.deepcopy(
                report.get("accounting", self.biosphere.accounting())
            ),
            "developments": copy.deepcopy(developments[-16:]),
            "developments_truncated": max(0, len(developments) - 16),
            "resident_physiology_coupled": False,
            "whole_food_web": False,
        }

    def save(self, path):
        if self.pending_step is not None:
            raise RuntimeError("Cannot checkpoint an incomplete distributed tick")
        receipt = self.neural.snapshot(
            f"world-{self.id}-{self.tick}", list(self.remote_ids.values())
        )
        state = {
            "version": 1,
            "kind": "chreatures-3d",
            "id": self.id,
            "tick": self.tick,
            "branch": self.branch,
            "paused": self.paused,
            "speed": self.speed,
            "world": self.world.snapshot(),
            "body_mode": self.body_mode,
            "physics_backend": self.physics_backend,
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
            "vision": self.vision.snapshot() if self.vision is not None else None,
            "last_senses": self.last_senses,
            "sensed_at": self.sensed_at,
            "input_names": self.neural.input_names,
            "output_names": self.neural.output_names,
            "organs": {k: v.snapshot() for k, v in self.organs.items()},
            "motor_artifact": self.motor_artifact.to_value()
            if self.motor_artifact is not None
            else None,
            "personal_memory": self.personal_memory,
            "personal_plasticity": self.personal_plasticity,
            "motors": {k: v.snapshot_value() for k, v in self.motors.items()},
            "remote_ids": self.remote_ids,
            "brain_url": self.neural.url,
            "graph_sha256": self.neural.graph["sha256"],
            "neural_snapshot": receipt,
            "neural_state": self.neural_state,
            "outcomes": self.outcomes,
            "journal": list(self.journal),
            "history": {k: list(v) for k, v in self.history.items()},
            "feature_mean": {k: v.tolist() for k, v in self.feature_mean.items()},
            "feature_variance": {
                k: v.tolist() for k, v in self.feature_variance.items()
            },
        }
        digest = hashlib.sha256(canonical(state)).hexdigest()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(
                canonical(
                    {
                        "format": "chreatures-3d-checkpoint-v1",
                        "sha256": digest,
                        "state": state,
                    }
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.saved_at = time.time()
        return digest

    @classmethod
    def load(cls, path, brain_url=None):
        from .acoustics import Acoustics
        from .ecology import Ecology
        from .fields import FieldEnvironment

        envelope = json.loads(Path(path).read_text())
        if envelope.get("format") != "chreatures-3d-checkpoint-v1":
            raise ValueError("Unsupported 3D checkpoint")
        value = envelope["state"]
        if hashlib.sha256(canonical(value)).hexdigest() != envelope["sha256"]:
            raise ValueError("3D checkpoint checksum mismatch")
        instance = cls.__new__(cls)
        instance.body_mode = value.get("body_mode", "crawler")
        instance.physics_backend = value.get("physics_backend", "reference")
        instance.execution_migrations = copy.deepcopy(
            value.get("execution_migrations", [])
        )
        world_type = physical_world_type(instance.body_mode, instance.physics_backend)
        instance.world = world_type.restore(value["world"])
        instance.visitor = VisitorPerformances.restore(
            value.get("visitor"), instance.world
        )
        instance.vision = None
        if value.get("vision") is not None:
            from .embodied_vision import EmbodiedVision

            instance.vision = EmbodiedVision.restore(value["vision"])
        instance.field = (
            FieldEnvironment.restore(value["field"])
            if value.get("field") is not None
            else None
        )
        if value.get("resources") is not None and value.get("biosphere") is not None:
            raise ValueError(
                "Saved world contains both legacy resources and a developmental biosphere"
            )
        instance.resources = (
            Ecology.restore(instance.world, value["resources"])
            if value.get("resources") is not None
            else None
        )
        instance.resource_state = copy.deepcopy(value.get("resource_state"))
        instance.biosphere = None
        if value.get("biosphere") is not None:
            from .biosphere import Biosphere

            instance.biosphere = Biosphere.restore(instance.world, value["biosphere"])
        instance.acoustics = (
            Acoustics.restore(instance.world, value["acoustics"])
            if value.get("acoustics") is not None
            else None
        )
        instance.acoustic_state = copy.deepcopy(value.get("acoustic_state"))
        instance.last_senses = copy.deepcopy(value.get("last_senses", {}))
        instance.sensed_at = value.get("sensed_at", instance.world.time)
        instance.organs = {
            key: AdaptiveOrgan.restore(organ) for key, organ in value["organs"].items()
        }
        instance.motor_artifact = None
        instance.motors = {}
        instance.personal_memory = bool(value.get("personal_memory", False))
        instance.personal_plasticity = bool(value.get("personal_plasticity", False))
        if value.get("motor_artifact") is not None:
            from .motor_inheritance import MotorArtifact, MotorOrgan

            instance.motor_artifact = MotorArtifact.from_value(value["motor_artifact"])
            motor_type = MotorOrgan
            if instance.personal_memory:
                from .living_motor import LivingMotorOrgan

                motor_type = LivingMotorOrgan
            instance.motors = {
                key: motor_type.restore_value(state, instance.motor_artifact)
                for key, state in value["motors"].items()
            }
            if (
                set(instance.motors) != {body.id for body in instance.world.bodies}
                or instance.organs
            ):
                raise ValueError(
                    "Saved motor controllers do not match the physical cohort"
                )
            if instance.personal_memory and any(
                (motor.personal_plasticity is not None) != instance.personal_plasticity
                for motor in instance.motors.values()
            ):
                raise ValueError(
                    "Saved personal plasticity selector differs from its organs"
                )
        instance.neural = NeuralClient(brain_url or value["brain_url"])
        instance._validate_motor_interface()
        if instance.neural.graph["sha256"] != value["graph_sha256"]:
            raise ValueError("Remote anatomy differs from saved resident anatomy")
        if (
            value.get("input_names", instance.neural.input_names)
            != instance.neural.input_names
            or value.get("output_names", instance.neural.output_names)
            != instance.neural.output_names
        ):
            raise ValueError("Remote neural ports differ from saved resident interface")
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
        ):
            setattr(instance, key, copy.deepcopy(value[key]))
        # Canonical JSON sorts mapping keys; artifact row order follows the
        # physical cohort, so reload must not reorder future neural snapshots.
        instance.remote_ids = {
            body.id: value["remote_ids"][body.id] for body in instance.world.bodies
        }
        instance.feature_mean = {
            k: np.asarray(v, dtype=np.float32) for k, v in value["feature_mean"].items()
        }
        instance.feature_variance = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in value["feature_variance"].items()
        }
        instance.journal = deque(value["journal"], maxlen=256)
        instance.history = {
            k: deque(v, maxlen=360) for k, v in value["history"].items()
        }
        instance.timings = deque(maxlen=120)
        instance.phase_timings = deque(maxlen=120)
        instance.error = None
        instance.pending_step = None
        instance.saved_at = time.time()
        return instance
