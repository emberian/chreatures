"""One ordered authority for an interacting world and its individual brains."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path

from .brain import Brain
from .world import World, MODEL_DT


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Habitat:
    def __init__(self, seed=7, connectome_path=None):
        self.world = World(seed)
        self.brains = {body.id: Brain(connectome_path, seed=seed * 1009 + i) for i, body in enumerate(self.world.bodies)}
        self.id = str(uuid.uuid4())
        self.branch = "resident"
        self.parent = None
        self.tick = 0
        self.paused = False
        self.speed = 1
        self.rewards = {body.id: 0.0 for body in self.world.bodies}
        self.journal = deque(maxlen=160)
        self.history = {body.id: deque(maxlen=220) for body in self.world.bodies}
        self.last_modes = {body.id: "exploring" for body in self.world.bodies}
        self.timings = deque(maxlen=120)
        self.saved_at = None
        self.note("hatched", "Mica, Fern and Pip entered the nursery.")

    def note(self, kind, text, **extra):
        self.journal.append({"id": f"{self.id}:{self.tick}:{len(self.journal)}", "time": round(self.world.time, 3),
                             "kind": kind, "text": text, **extra})

    def step(self, steps=1):
        for _ in range(steps):
            start = time.perf_counter()
            actions = {}
            for body in self.world.bodies:
                senses = self.world.sense(body.id)
                physiology = {key: getattr(body, key) for key in ("energy", "gut", "fatigue")}
                actions[body.id] = self.brains[body.id].step(senses, physiology, MODEL_DT, self.rewards[body.id])
            outcomes = self.world.advance(actions, MODEL_DT)
            self.rewards = {key: outcome["nutrition"] for key, outcome in outcomes.items()}
            self.tick += 1
            for body in self.world.bodies:
                brain = self.brains[body.id]
                if brain.mode != self.last_modes[body.id]:
                    if brain.mode in ("feeding", "resting"):
                        self.note(brain.mode, f"{body.name} started {brain.mode}.", resident=body.id)
                    self.last_modes[body.id] = brain.mode
                if self.tick % 10 == 0:
                    self.history[body.id].append({"time": self.world.time, "energy": body.energy,
                                                  "activity": float(brain.rates.mean()),
                                                  "nutrition": brain.total_nutrition, "x": body.x, "y": body.y})
            self.timings.append((time.perf_counter() - start) * 1000)

    def command(self, command):
        if not isinstance(command, dict):
            raise ValueError("Expected an object")
        op = command.get("op")
        if op == "pause":
            if set(command) - {"op", "paused"} or not isinstance(command.get("paused"), bool):
                raise ValueError("pause requires boolean paused")
            self.paused = command["paused"]
            return {"paused": self.paused}
        if op == "speed":
            if set(command) - {"op", "value"} or type(command.get("value")) is not int or command["value"] not in (1, 2, 4):
                raise ValueError("speed must be 1, 2, or 4")
            self.speed = command["value"]
            return {"speed": self.speed}
        result = self.world.command(command)
        description = {"add": "Placed", "move": "Moved", "remove": "Removed", "signal": "Played"}.get(op, op)
        self.note("caregiver", f"{description} {result.get('kind', 'a tone')}.", command=copy.deepcopy(command))
        return result

    def view(self):
        data = self.world.view()
        data.update({"id": self.id, "branch": self.branch, "tick": self.tick, "paused": self.paused, "speed": self.speed,
                     "brains": {key: brain.view() for key, brain in self.brains.items()},
                     "senses": {body.id: self.world.sense(body.id) for body in self.world.bodies},
                     "journal": list(self.journal)[-30:], "history": {key: list(value) for key, value in self.history.items()},
                     "performance": {"step_ms": sum(self.timings) / max(1, len(self.timings)), "dt": MODEL_DT},
                     "saved_at": self.saved_at})
        return data

    def snapshot(self):
        return {"version": 1, "id": self.id, "branch": self.branch, "parent": self.parent, "tick": self.tick,
                "world": self.world.snapshot(), "brains": {key: value.snapshot() for key, value in self.brains.items()},
                "paused": self.paused, "speed": self.speed, "rewards": copy.deepcopy(self.rewards),
                "journal": list(self.journal), "history": {key: list(v) for key, v in self.history.items()},
                "last_modes": self.last_modes.copy()}

    @classmethod
    def restore(cls, snapshot, connectome_path=None):
        if snapshot.get("version") != 1:
            raise ValueError("Unsupported habitat checkpoint")
        canonical(snapshot)  # Reject all nonfinite values before constructing state.
        instance = cls.__new__(cls)
        instance.world = World.restore(snapshot["world"])
        instance.brains = {key: Brain.restore(value, connectome_path) for key, value in snapshot["brains"].items()}
        ids = {body.id for body in instance.world.bodies}
        if ids != set(instance.brains) or ids != set(snapshot["rewards"]):
            raise ValueError("World and brain identities differ")
        for key in ("id", "branch", "parent", "tick", "paused", "speed", "rewards", "last_modes"):
            setattr(instance, key, copy.deepcopy(snapshot[key]))
        instance.journal = deque(copy.deepcopy(snapshot["journal"]), maxlen=160)
        instance.history = {key: deque(copy.deepcopy(value), maxlen=220) for key, value in snapshot["history"].items()}
        instance.timings = deque(maxlen=120)
        instance.saved_at = None
        return instance

    def fork(self):
        child = self.restore(self.snapshot())
        child.parent = {"id": self.id, "tick": self.tick}
        child.id = str(uuid.uuid4())
        child.branch = "research"
        return child

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.snapshot()
        envelope = {"format": "chreatures-checkpoint-v1", "sha256": hashlib.sha256(canonical(state)).hexdigest(), "state": state}
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as file:
            file.write(canonical(envelope))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        self.saved_at = time.time()
        return envelope["sha256"]

    @classmethod
    def load(cls, path, connectome_path=None):
        value = json.loads(Path(path).read_text())
        if value.get("format") != "chreatures-checkpoint-v1":
            raise ValueError("Unknown checkpoint format")
        if hashlib.sha256(canonical(value["state"])).hexdigest() != value.get("sha256"):
            raise ValueError("Checkpoint checksum does not match")
        return cls.restore(value["state"], connectome_path)
