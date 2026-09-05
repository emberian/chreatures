"""A whole 3D world coupled to full MaleCNS state and personal learning organs.

The physical world is local; the full curated nervous systems persist on AMD.
A checkpoint references a checksum-verified server-side neural artifact and
contains physical integration state plus every personal cognitive parameter.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .cognition import AdaptiveOrgan
from .neural_client import NeuralClient, sensory_channels
from .runtime import canonical


class Habitat3D:
    def __init__(self, seed=7, brain_url="http://127.0.0.1:18765", spec=None):
        from .physics import PhysicsWorld
        self.world = PhysicsWorld(seed=seed, spec=spec)
        self.neural = NeuralClient(brain_url)
        self.id = str(uuid.uuid4())
        self.tick = 0
        self.paused = False
        self.speed = 1
        self.branch = "resident"
        self.saved_at = None
        self.error = None
        self.remote_ids = {b.id: f"{self.id}:{b.id}" for b in self.world.bodies}
        self.neural.create(list(self.remote_ids.values()))
        self.organs = {b.id: AdaptiveOrgan(feature_dim=len(self.neural.output_names), seed=seed) for b in self.world.bodies}
        for i, organ in enumerate(self.organs.values()):
            # Shared inherited maps; separate stochastic lives.
            organ.rng = np.random.default_rng(seed * 1009 + i)
        self.outcomes = {b.id: {} for b in self.world.bodies}
        self.neural_state = {b.id: {"features": [0.0] * len(self.neural.output_names), "activity": 0.0, "support": 1.0} for b in self.world.bodies}
        self.feature_mean = {b.id: np.zeros(len(self.neural.output_names), dtype=np.float32) for b in self.world.bodies}
        self.feature_variance = {b.id: np.ones(len(self.neural.output_names), dtype=np.float32) * 0.01 for b in self.world.bodies}
        self.journal = deque(maxlen=256)
        self.history = {b.id: deque(maxlen=360) for b in self.world.bodies}
        self.timings = deque(maxlen=120)
        self.pending_step = None
        self.note("hatched", "Three new residents entered the hollow garden with full MaleCNS circuits.")

    def note(self, kind, text, **fields):
        self.journal.append({"id": f"{self.id}:{self.tick}:{len(self.journal)}", "time": self.world.time,
                             "kind": kind, "text": text, **fields})

    def step(self, steps=1):
        dt = 0.05
        if self.pending_step is not None:
            raise RuntimeError("A previous world step is incomplete; restore its checkpoint before advancing")
        for _ in range(steps):
            started = time.perf_counter()
            sensed = {b.id: self.world.sense(b.id) for b in self.world.bodies}
            entries = [{"id": self.remote_ids[b.id], "senses": sensory_channels(sensed[b.id])} for b in self.world.bodies]
            self.pending_step = {"tick": self.tick, "neural_seq": self.neural.next_seq}
            responses = self.neural.step(entries, dt)
            inverse = {v: k for k, v in self.remote_ids.items()}
            actions = {}
            for response in responses:
                body_id = inverse[response["id"]]
                body = next(b for b in self.world.bodies if b.id == body_id)
                features = np.asarray(response["features"], dtype=np.float32)
                mean, variance = self.feature_mean[body_id], self.feature_variance[body_id]
                delta = features - mean
                mean += np.float32(dt / 20) * delta
                variance += np.float32(dt / 20) * (delta * delta - variance)
                normalized = np.clip(delta / np.sqrt(np.maximum(variance, 1e-6)), -2, 2)
                local_body = {key: float(getattr(body, key)) for key in ("energy", "gut", "fatigue", "speed", "angular_velocity")}
                local_body["support"] = float(response["support"])
                action = self.organs[body_id].step(normalized, local_body, self.outcomes[body_id], dt)
                for channel in ("grip", "signal_low", "signal_mid", "signal_high"):
                    action[channel] = max(0.0, action[channel])
                # Ingestion is an embodied contact reflex, not an object/position
                # query. Movement, looking, grip and signaling are learned outputs.
                action["eat"] = float(np.clip((1 - body.gut) * (1.1 - body.energy), 0, 1))
                actions[body_id] = action
                self.neural_state[body_id] = response
            self.outcomes = self.world.advance(actions, dt)
            self.tick += 1
            self.pending_step = None
            for b in self.world.bodies:
                if self.outcomes[b.id].get("nutrition", 0) > 0 and self.tick % 20 == 0:
                    self.note("feeding", f"{b.name} ingested a resource.", resident=b.id)
                if self.tick % 10 == 0:
                    self.history[b.id].append({"time": self.world.time, "x": float(b.x), "y": float(b.y), "z": float(b.z),
                                               "energy": b.energy, "activity": self.neural_state[b.id]["activity"],
                                               "memory": len(self.organs[b.id].memory.records)})
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
            if type(command.get("value")) is not int or command["value"] not in (1, 2, 4):
                raise ValueError("speed must be 1, 2 or 4")
            self.speed = command["value"]
            return {"speed": self.speed}
        if op == "bookmark":
            text = command.get("text", "A moment in the garden.")
            if not isinstance(text, str) or len(text) > 500:
                raise ValueError("Bookmark text must be at most 500 characters")
            self.note("observation", text, origin="caregiver")
            return {"bookmarked": True}
        result = self.world.command(command)
        self.note("caregiver", f"Outside interaction: {op}.", command=copy.deepcopy(command))
        return result

    def view(self):
        view = self.world.view()
        view.update({"id": self.id, "tick": self.tick, "branch": self.branch, "paused": self.paused,
                     "speed": self.speed, "saved_at": self.saved_at, "error": self.error,
                     "neural": copy.deepcopy(self.neural_state),
                     "cognition": {key: organ.view() for key, organ in self.organs.items()},
                     "senses": {b.id: self.world.sense(b.id) for b in self.world.bodies},
                     "journal": list(self.journal)[-40:], "history": {k: list(v) for k, v in self.history.items()},
                     "anatomy": {"dataset": "MaleCNS v1.0", "neurons": self.neural.graph["neurons"],
                                 "connections": self.neural.graph["edges"], "sha256": self.neural.graph["sha256"],
                                 "scope": "full traced curated brain and nerve cord"},
                     "performance": {"step_ms": sum(self.timings) / max(1, len(self.timings)), "dt": 0.05}})
        return view

    def save(self, path):
        if self.pending_step is not None:
            raise RuntimeError("Cannot checkpoint an incomplete distributed tick")
        receipt = self.neural.snapshot(f"world-{self.id}-{self.tick}")
        state = {"version": 1, "kind": "chreatures-3d", "id": self.id, "tick": self.tick,
                 "branch": self.branch, "paused": self.paused, "speed": self.speed,
                 "world": self.world.snapshot(), "organs": {k: v.snapshot() for k, v in self.organs.items()},
                 "remote_ids": self.remote_ids, "brain_url": self.neural.url, "graph_sha256": self.neural.graph["sha256"],
                 "neural_snapshot": receipt, "neural_state": self.neural_state,
                 "outcomes": self.outcomes, "journal": list(self.journal),
                 "history": {k: list(v) for k, v in self.history.items()},
                 "feature_mean": {k: v.tolist() for k, v in self.feature_mean.items()},
                 "feature_variance": {k: v.tolist() for k, v in self.feature_variance.items()}}
        digest = hashlib.sha256(canonical(state)).hexdigest()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(canonical({"format": "chreatures-3d-checkpoint-v1", "sha256": digest, "state": state}))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.saved_at = time.time()
        return digest

    @classmethod
    def load(cls, path, brain_url=None):
        from .physics import PhysicsWorld
        envelope = json.loads(Path(path).read_text())
        if envelope.get("format") != "chreatures-3d-checkpoint-v1":
            raise ValueError("Unsupported 3D checkpoint")
        value = envelope["state"]
        if hashlib.sha256(canonical(value)).hexdigest() != envelope["sha256"]:
            raise ValueError("3D checkpoint checksum mismatch")
        instance = cls.__new__(cls)
        instance.world = PhysicsWorld.restore(value["world"])
        instance.organs = {key: AdaptiveOrgan.restore(organ) for key, organ in value["organs"].items()}
        instance.neural = NeuralClient(brain_url or value["brain_url"])
        if instance.neural.graph["sha256"] != value["graph_sha256"]:
            raise ValueError("Remote anatomy differs from saved resident anatomy")
        instance.neural.restore(value["neural_snapshot"])
        for key in ("id", "tick", "branch", "paused", "speed", "remote_ids", "neural_state", "outcomes"):
            setattr(instance, key, copy.deepcopy(value[key]))
        instance.feature_mean = {k: np.asarray(v, dtype=np.float32) for k, v in value["feature_mean"].items()}
        instance.feature_variance = {k: np.asarray(v, dtype=np.float32) for k, v in value["feature_variance"].items()}
        instance.journal = deque(value["journal"], maxlen=256)
        instance.history = {k: deque(v, maxlen=360) for k, v in value["history"].items()}
        instance.timings = deque(maxlen=120)
        instance.error = None
        instance.pending_step = None
        instance.saved_at = time.time()
        return instance
