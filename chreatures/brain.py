"""A measured recurrent circuit, engineered sensory bridges and personal plasticity.

These are synthetic rate dynamics on real directed fly edges, not fitted fly
physiology. No controller function receives world coordinates or object IDs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
CHANNELS = ["odor L0", "odor L1", "odor L2", "odor R0", "odor R1", "odor R2",
            "obstacle left", "obstacle right", "red", "green", "blue",
            "tone low", "tone middle", "tone high", "shade", "contact"]


def stable_number(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


@dataclass
class Genome:
    neural_gain: float = 0.92
    neural_tau: float = 0.16
    learning_rate: float = 0.20
    curiosity: float = 0.65
    activity: float = 0.70
    support_recovery: float = 0.024


class Connectome:
    def __init__(self, path: str | Path):
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        self.ids = data["ids"].astype(str)
        self.n = len(self.ids)
        self.pre = data["pre"].astype(np.int32)
        self.post = data["post"].astype(np.int32)
        self.count = data["count"].astype(np.float32)
        self.sign = data["sign"].astype(np.float32)
        if len(self.sign) != self.n or not np.all(np.isfinite(self.count)) or np.any(self.count <= 0):
            raise ValueError("Malformed anatomical scaffold")
        if self.pre.min(initial=0) < 0 or self.post.min(initial=0) < 0 or max(self.pre.max(initial=0), self.post.max(initial=0)) >= self.n:
            raise ValueError("Connectome index out of bounds")
        self.labels = data["labels"].astype(str) if "labels" in data else self.ids.copy()
        self.types = data["type"].astype(str) if "type" in data else self.labels.copy()
        self.sides = data["side"].astype(str) if "side" in data else np.full(self.n, "")
        self.groups = data["group"].astype(str) if "group" in data else self.types.copy()
        self.manifest = json.loads(path.with_name("manifest.json").read_text()) if path.with_name("manifest.json").exists() else {}
        self.hash = hashlib.sha256(path.read_bytes()).hexdigest()
        incoming = np.bincount(self.post, weights=self.count, minlength=self.n)
        # Per-cell input normalization and NT sign are explicit modeling choices.
        self.weights = (self.count * self.sign[self.pre] / np.maximum(incoming[self.post], 1)).astype(np.float32)
        self.matrix = sparse.csr_matrix((self.weights, (self.post, self.pre)), shape=(self.n, self.n))
        self.input_map, self.input_cells = self._make_inputs()
        self.output_cells = self._outputs()
        self.baseline, self.decoder, self.calibration_error = self._calibrate()
        self.display_indices = np.unique(np.linspace(0, self.n - 1, min(self.n, 180), dtype=int))

    def _make_inputs(self):
        annotation = np.char.lower(np.char.add(np.char.add(self.groups, " "), self.types))
        olfactory = np.array([("projection" in s or "olfactory_pn" in s or s.startswith("pn") or "alpn" in s) for s in annotation])
        if olfactory.sum() < 12:
            olfactory = np.array([("pn" in s and "mbon" not in s) for s in annotation])
        if olfactory.sum() < 12:
            raise ValueError("Circuit needs annotated projection neurons; cannot silently invent sensory anatomy")
        candidates = np.flatnonzero(olfactory)
        inputs = np.zeros((self.n, len(CHANNELS)), dtype=np.float32)
        # Assignment of fictional odor identities to real PNs is engineered.
        # Keep hemispheres where annotated; split deterministically if unassigned.
        for idx in candidates:
            h = stable_number(self.types[idx])
            side = str(self.sides[idx]).lower()
            lr = 0 if side.startswith("l") else 1 if side.startswith("r") else stable_number(self.ids[idx]) % 2
            inputs[idx, lr * 3 + h % 3] = 0.80
        # Our synthetic species routes additional modalities through separate PNs
        # when the selected subgraph has no peripheral visual/auditory neurons.
        # This is a bridge, not a claimed biological sensory projection.
        order = sorted(candidates.tolist(), key=lambda i: stable_number(self.ids[i] + "bridge"))
        for channel in range(6, len(CHANNELS)):
            cells = order[(channel - 6)::len(CHANNELS) - 6]
            inputs[cells, channel] = 0.55
        return inputs, candidates

    def _outputs(self):
        candidates = np.setdiff1d(np.arange(self.n), self.input_cells)
        strength = np.asarray(abs(self.matrix[:, self.input_cells]).sum(axis=1)).ravel()
        # Include second-order recipients as well as directly reached cells.
        strength += np.asarray(abs(self.matrix) @ strength).ravel()
        candidates = candidates[np.argsort(strength[candidates])[-min(1400, len(candidates)):]]
        if not len(candidates) or strength[candidates].max(initial=0) == 0:
            raise ValueError("No measured downstream paths from selected sensory cells")
        return candidates

    def _calibrate(self):
        """Fit a fixed linear bridge from downstream rates to sensory coordinates.

        Basis stimulation only; no food outcomes or live resident histories.
        Anatomy drives every response used by the bridge. General mixtures and
        transient responses are not guaranteed to reconstruct inputs exactly.
        """
        c = len(CHANNELS)
        r = np.zeros((self.n, c + 1), dtype=np.float32)
        drive = np.zeros((self.n, c + 1), dtype=np.float32)
        drive[:, 1:] = self.input_map * 0.65
        for _ in range(72):
            r += 0.22 * (np.maximum(0, np.tanh(0.005 + drive + 0.92 * (self.matrix @ r))) - r)
        baseline = r[:, 0].copy()
        x = (r[self.output_cells, 1:] - baseline[self.output_cells, None]).T.astype(np.float64)
        gram = x @ x.T
        ridge = max(float(np.trace(gram)) / c * 1e-4, 1e-9)
        decoder = (x.T @ np.linalg.solve(gram + np.eye(c) * ridge, np.eye(c) * 0.65)).astype(np.float32)
        error = float(np.abs(x @ decoder - np.eye(c) * 0.65).mean())
        return baseline, decoder, error

    def summary(self):
        return {"neurons": self.n, "connections": len(self.pre), "synapses": int(self.count.sum()),
                "sha256": self.hash, "calibration_mae": self.calibration_error,
                "input_cells": len(self.input_cells), "readout_cells": len(self.output_cells),
                "source": self.manifest, "channels": CHANNELS}


@lru_cache(maxsize=3)
def _cached_connectome(path: str, content_hash: str) -> Connectome:
    return Connectome(path)


def load_connectome(path: str = str(ROOT / "data/connectome/circuit.npz")) -> Connectome:
    # A path can be replaced while this process is alive. Identity follows bytes.
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return _cached_connectome(str(path), digest)


class Brain:
    def __init__(self, connectome_path: str | Path | None = None, seed: int = 0,
                 genome: dict | None = None):
        self.graph = load_connectome(str(connectome_path or ROOT / "data/connectome/circuit.npz"))
        self.genome = Genome(**(genome or {}))
        self.rng = np.random.default_rng(seed)
        n = self.graph.n
        self.rates = self.graph.baseline.copy()
        self.adaptation = np.zeros(n, dtype=np.float32)
        self.support = np.ones(n, dtype=np.float32)
        self.context = np.zeros(len(CHANNELS), dtype=np.float32)
        self.eligibility = np.zeros(3, dtype=np.float32)
        self.values = np.array([0.22, 0.22, 0.12], dtype=np.float32)
        self.sound_memory = np.zeros((3, 3), dtype=np.float32)
        self.sound_trace = np.zeros(3, dtype=np.float32)
        self.episodes: list[dict] = []
        self.time = 0.0
        self.exploration = float(self.rng.normal(0, 0.20))
        self.last_odor = np.zeros(3, dtype=np.float32)
        self.last_prediction = 0.0
        self.novelty = 1.0
        self.modulator = 0.0
        self.total_nutrition = 0.0
        self.learning = True
        self.silenced = False
        self.mode = "exploring"
        self.last_action = {"forward": 0.5, "turn": 0.0, "eat": 0.0, "signal": 0.0}
        self.last_encoded = np.zeros(len(CHANNELS), dtype=np.float32)
        self.last_decoded = np.zeros(len(CHANNELS), dtype=np.float32)
        self.last_episode_time = -10.0

    @staticmethod
    def encode(senses: dict) -> np.ndarray:
        odor = np.asarray(senses["odor"], dtype=np.float32).reshape(2, 3)
        vision = np.asarray(senses["vision"], dtype=np.float32).reshape(16, 4)
        sound = np.asarray(senses.get("sound", [0, 0, 0]), dtype=np.float32)
        # Left is the negative-angle half of the retinal fan.
        # This species starts with a warm-color affordance prior. Fruit-colored
        # surfaces invite contact; dark/blue surfaces elicit looming avoidance.
        # Only measured RGB samples enter this engineered peripheral transform.
        warm = np.clip((vision[:, 0] - vision[:, 2]) * 2.5, 0, 0.95)
        looming = vision[:, 3] ** 3 * (1 - warm)
        left = float(looming[:8].max(initial=0))
        right = float(looming[8:].max(initial=0))
        rgb = vision[:, :3].mean(axis=0)
        touch = float(np.max(senses.get("touch", [0, 0])))
        value = np.concatenate((odor.ravel(), [left, right], rgb, sound,
                                [senses.get("shade", 0), touch]))
        if not np.all(np.isfinite(value)):
            raise ValueError("Non-finite sensation")
        return np.clip(value, 0, 1).astype(np.float32)

    def step(self, senses: dict, physiology: dict, dt: float, reward: float = 0.0) -> dict:
        if not np.isfinite(dt) or not 0 < dt <= 0.2 or not np.isfinite(reward):
            raise ValueError("Invalid neural timestep or outcome")
        self.time += dt
        encoded = self.encode(senses)
        self.last_encoded = encoded
        drive = self.graph.input_map @ encoded
        # Two-timescale rate integration; row-normalized recurrent gain < 1.
        for _ in range(2):
            recurrent = self.graph.matrix @ self.rates if not self.silenced else np.zeros_like(self.rates)
            target = np.maximum(0, np.tanh(0.005 + drive + self.genome.neural_gain * recurrent - 0.10 * self.adaptation))
            self.rates += min(1.0, dt / 2 / self.genome.neural_tau) * (target * self.support - self.rates)
        self.adaptation += dt / 5 * (self.rates - self.adaptation)
        self.support += dt * (self.genome.support_recovery * (1 - self.support) - 0.003 * self.rates)
        np.clip(self.support, 0.65, 1, out=self.support)
        decoded = (self.rates[self.graph.output_cells] - self.graph.baseline[self.graph.output_cells]) @ self.graph.decoder
        decoded = np.clip(decoded, 0, 1).astype(np.float32)
        self.last_decoded = decoded
        self.context += dt / 3.0 * (decoded - self.context)
        odor = decoded[:6].reshape(2, 3)
        odor_mean = odor.mean(axis=0)
        self.eligibility = (self.eligibility * np.exp(-dt / 4) + odor_mean * dt / 4).astype(np.float32)
        self.sound_trace = (self.sound_trace * np.exp(-dt / 3) + decoded[11:14] * dt / 3).astype(np.float32)
        prediction = float(np.dot(self.values, odor_mean))
        # Nutrition is a body outcome; a delayed trace assigns it to recently
        # perceived cues. This is engineered three-factor associative plasticity.
        positive = max(0.0, float(reward))
        self.modulator += dt / 0.8 * (min(1.0, positive / dt * 30) - self.modulator)
        self.total_nutrition += positive
        if self.learning:
            if positive > 0:
                self.values += self.genome.learning_rate * positive * 120 * self.eligibility * (1.0 - self.values)
                self.sound_memory += positive * 25 * np.outer(self.sound_trace, self.eligibility)
            # Very slow extinction only where a strong cue is repeatedly sampled.
            else:
                self.values -= dt * 0.0008 * self.eligibility * np.maximum(0, self.values - 0.12)
            np.clip(self.values, 0.05, 1.0, out=self.values)
            np.clip(self.sound_memory, 0, 1, out=self.sound_memory)
        # Habituation is local to perceived configurations, not world positions.
        if self.time - self.last_episode_time >= 2.0:
            distances = [float(np.linalg.norm(decoded - np.asarray(e["features"]))) for e in self.episodes]
            distance = min(distances, default=1.0)
            self.novelty = min(1.0, distance * 2)
            if distance > 0.22 or positive > 0:
                self.episodes.append({"time": round(self.time, 3), "features": decoded.tolist(),
                                      "nutrition": positive, "origin": "experienced"})
                self.episodes = self.episodes[-160:]
            self.last_episode_time = self.time
        energy = float(physiology.get("energy", 0.7))
        gut = float(physiology.get("gut", 0.0))
        fatigue = float(physiology.get("fatigue", 0.0))
        hunger = float(np.clip(1.0 - energy + 0.15 - gut * 0.45, 0, 1))
        cue_values = self.values + decoded[11:14] @ self.sound_memory * 0.2
        attraction = float(np.dot(odor[1] - odor[0], cue_values))
        concentration = float(np.dot(odor_mean, cue_values))
        contact = max(float(senses.get("touch", [0, 0])[0]), float(senses.get("touch", [0, 0])[1]))
        # Body contact is a fast local reflex. All distance/odor decisions pass
        # through recurrent circuit responses and the calibrated output bridge.
        obstacles = decoded[6:8]
        avoidance = float(obstacles[0] - obstacles[1])
        self.exploration += dt * (-self.exploration * 0.6) + np.sqrt(dt) * float(self.rng.normal(0, 0.38))
        turning = 38.0 * attraction * (0.4 + hunger) + 2.1 * avoidance
        turning += self.exploration * (0.35 + self.genome.curiosity * (0.4 + self.novelty)) * max(0.15, 1 - concentration * 4)
        if contact > 0.2 and positive <= 0:
            turning += 1.0 if senses.get("touch", [0, 0])[0] >= senses.get("touch", [0, 0])[1] else -1.0
        rest = fatigue * (0.35 + decoded[14] * 0.8) * (1 - hunger * 0.7)
        forward = (0.30 + self.genome.activity * 0.38 + hunger * 0.22) * (1 - rest)
        forward *= 1 - min(0.65, float(max(obstacles)) * 0.65)
        # Slow down in a cue patch so contact ingestion has time to matter.
        forward *= 1 - min(0.65, concentration * 1.7)
        if positive > 0:
            forward *= 0.06
            turning *= 0.12
        eating = float(np.clip(hunger * 1.8 + concentration * 0.3 - gut * 0.5, 0, 1))
        signal = float(np.clip(float(decoded[11:14].max(initial=0)) * float(self.sound_memory.max(initial=0)), 0, 1))
        self.mode = "resting" if rest > 0.45 else "feeding" if positive > 0 else "investigating" if concentration > 0.12 else "exploring"
        self.last_action = {"forward": float(np.clip(forward, 0, 1)), "turn": float(np.clip(turning, -1, 1)),
                            "eat": eating, "signal": signal}
        self.last_odor = odor_mean.copy()
        self.last_prediction = prediction
        return self.last_action.copy()

    def view(self):
        display = self.graph.display_indices
        return {"mode": self.mode, "time": self.time, "activity": float(self.rates.mean()),
                "peak_activity": float(self.rates.max(initial=0)), "support": float(self.support.mean()),
                "novelty": self.novelty, "modulator": self.modulator, "values": self.values.tolist(),
                "episodes": len(self.episodes), "learning": self.learning, "silenced": self.silenced,
                "total_nutrition": self.total_nutrition, "action": self.last_action,
                "sensory": self.last_encoded.tolist(), "decoded": self.last_decoded.tolist(),
                "sound_memory": self.sound_memory.tolist(),
                "neural": {"ids": self.graph.ids[display].tolist(), "rates": self.rates[display].tolist(),
                           "labels": self.graph.labels[display].tolist()}}

    def snapshot(self):
        array_fields = ["rates", "adaptation", "support", "context", "eligibility", "values", "sound_memory",
                        "sound_trace", "last_odor", "last_encoded", "last_decoded"]
        scalar_fields = ["time", "exploration", "last_prediction", "novelty", "modulator", "total_nutrition",
                         "learning", "silenced", "mode", "last_action", "last_episode_time", "episodes"]
        return {"version": 1, "connectome_sha256": self.graph.hash, "genome": vars(self.genome),
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "arrays": {k: getattr(self, k).tolist() for k in array_fields},
                "state": {k: copy.deepcopy(getattr(self, k)) for k in scalar_fields}}

    @classmethod
    def restore(cls, snapshot, connectome_path=None):
        if snapshot.get("version") != 1:
            raise ValueError("Unsupported brain snapshot version")
        brain = cls(connectome_path=connectome_path, genome=snapshot["genome"])
        if brain.graph.hash != snapshot["connectome_sha256"]:
            raise ValueError("Checkpoint uses different anatomical scaffold")
        expected = brain.snapshot()
        if set(snapshot["arrays"]) != set(expected["arrays"]) or set(snapshot["state"]) != set(expected["state"]):
            raise ValueError("Incomplete brain checkpoint")
        for key, value in snapshot["arrays"].items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != getattr(brain, key).shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid brain state: {key}")
            setattr(brain, key, array)
        for key, value in snapshot["state"].items():
            if key in ("time", "exploration", "last_prediction", "novelty", "modulator", "total_nutrition", "last_episode_time"):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"Invalid brain scalar: {key}")
            elif key in ("learning", "silenced"):
                if type(value) is not bool:
                    raise ValueError(f"Invalid brain switch: {key}")
            elif key == "mode":
                if value not in ("exploring", "resting", "feeding", "investigating"):
                    raise ValueError("Invalid behavioral mode")
            elif key == "episodes":
                if not isinstance(value, list) or len(value) > 160:
                    raise ValueError("Invalid episode memory")
                for episode in value:
                    if not isinstance(episode, dict) or episode.get("origin") != "experienced":
                        raise ValueError("Invalid episode origin")
                    features = np.asarray(episode.get("features"), dtype=float)
                    if features.shape != (len(CHANNELS),) or not np.isfinite(features).all():
                        raise ValueError("Invalid episode features")
            elif key == "last_action":
                if not isinstance(value, dict) or set(value) != {"forward", "turn", "eat", "signal"}:
                    raise ValueError("Invalid saved motor action")
                for action, amount in value.items():
                    if not isinstance(amount, (int, float)) or not np.isfinite(amount) or not (-1 if action == "turn" else 0) <= amount <= 1:
                        raise ValueError("Invalid saved motor value")
            setattr(brain, key, copy.deepcopy(value))
        brain.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        return brain
