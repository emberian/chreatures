"""Finite-energy, physically triggered acoustic transducers for 3-D worlds.

This module supplies mechanisms, not social behaviors. Contact impulses and
hinge damping work can charge three-tone oscillators. Their stored energy then
radiates and decays, while listeners receive only a distance-attenuated,
ray-occluded local signal through the world's existing ``sound`` sense.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


_LEDGER_FIELDS = (
    "contact_work_observed", "hinge_work_extracted", "captured_energy",
    "transduction_loss", "capacity_rejected", "radiated_energy",
    "oscillator_loss",
)


def _number(value: Any, name: str, low: float = 0.0, high: float = 1e9) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 96 or not value[0].isalpha() or any(
        not (character.isalnum() or character in "_.-") for character in value
    ):
        raise ValueError(f"invalid {name}")
    return value


def _vector(value: Any, length: int, name: str, low: float, high: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [_number(item, name, low, high) for item in value]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _rotation(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = map(float, quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Acoustics:
    """Optional physical-to-acoustic energy conversion attached to one world."""

    def __init__(self, world: Any, config: dict[str, Any] | str | Path | None = None):
        required = (
            "attach_acoustics", "acoustic_components", "acoustic_entity_state",
            "apply_acoustic_hinge_torque", "acoustic_visibility",
        )
        if any(not callable(getattr(world, method, None)) for method in required):
            raise TypeError("world does not provide the acoustic integration hooks")
        self.world = world
        raw = self._load(config)
        self.config, self._emitters = self._resolve(raw)
        self._by_entity = {emitter["entity"]: emitter_id for emitter_id, emitter in self._emitters.items()}
        self._states: dict[str, dict[str, Any]] = {}
        for emitter_id, emitter in self._emitters.items():
            weights = np.asarray(emitter["tones"], dtype=float)
            weights /= weights.sum()
            self._states[emitter_id] = {
                "energy": (weights * emitter["initial_energy"]).tolist(),
                "cooldown": 0.0, "event_count": 0,
            }
        self._ledger = {name: 0.0 for name in _LEDGER_FIELDS}
        self._initial_energy = self._stored_energy()
        self.time = 0.0
        self._native_revision = -1
        self._bind_native()
        topology = {
            "config": self.config,
            "bindings": sorted((key, value["entity"], value["drive"]) for key, value in self._emitters.items()),
        }
        self._signature = hashlib.sha256(_canonical(topology)).hexdigest()
        self.world.attach_acoustics(self)

    def _bind_native(self) -> None:
        """Resolve model-local addresses while preserving oscillator state."""
        from .native_world import load_world_kernels

        values = list(self._emitters.values())
        bodies, dofs = [], []
        for emitter in values:
            entity = emitter["entity"]
            bodies.append(int(self.world._entity_mj[entity]))
            dofs.append(
                int(self.world.model.jnt_dofadr[self.world._entity_joint[entity]])
                if emitter["drive"] in {"hinge", "both"} else -1
            )
        tones = []
        energy = []
        for emitter_id, emitter in self._emitters.items():
            weights = np.asarray(emitter["tones"], dtype=float)
            weights /= weights.sum()
            tones.extend(weights.tolist())
            energy.extend(self._states[emitter_id]["energy"])
        self._native = load_world_kernels().AcousticEngine(
            bodies, dofs,
            [v for emitter in values for v in emitter["source_offset"]], tones,
            [v["energy_capacity"] for v in values],
            [v["capture_efficiency"] for v in values],
            [v["impact_threshold"] for v in values],
            [v["min_impact_speed"] for v in values],
            [v["cooldown"] for v in values], [v["decay_time"] for v in values],
            [v["radiative_fraction"] for v in values], [v["gain"] for v in values],
            [v["reference_energy"] for v in values], [v["range"] for v in values],
            [v["occlusion"] for v in values], [v["hinge_damping"] for v in values],
            [v["max_hinge_torque"] for v in values],
            [v["drive"] in {"contact", "both"} for v in values], energy,
        )
        self._native.restore_state(
            energy,
            [self._states[key]["cooldown"] for key in self._emitters],
            [self._states[key]["event_count"] for key in self._emitters],
            [self._ledger[key] for key in _LEDGER_FIELDS],
        )
        self._native_revision = self.world.model_revision

    def _ensure_native_binding(self) -> None:
        if self._native_revision != self.world.model_revision:
            self._sync_native_state()
            self._bind_native()

    def _sync_native_state(self) -> None:
        energy, cooldown, events, ledger = self._native.state()
        energy = np.asarray(energy).reshape((-1, 3))
        for index, emitter_id in enumerate(self._emitters):
            self._states[emitter_id] = {
                "energy": energy[index].astype(float).tolist(),
                "cooldown": float(cooldown[index]), "event_count": int(events[index]),
            }
        self._ledger = dict(zip(_LEDGER_FIELDS, map(float, ledger), strict=True))

    @staticmethod
    def _load(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        if config is None:
            return {"version": 1}
        if isinstance(config, dict):
            return copy.deepcopy(config)
        return json.loads(Path(config).read_text())

    def _resolve(self, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported acoustic specification")
        unknown = set(raw) - {"version", "include_authored", "emitters"}
        if unknown:
            raise ValueError(f"unknown acoustic specification fields: {sorted(unknown)}")
        include_authored = raw.get("include_authored", True)
        if not isinstance(include_authored, bool):
            raise ValueError("include_authored must be boolean")
        values: list[dict[str, Any]] = []
        if include_authored:
            for record in self.world.acoustic_components():
                values.extend({**component, "entity": record["entity"]} for component in record["components"])
        external = raw.get("emitters", [])
        if not isinstance(external, list) or any(not isinstance(value, dict) for value in external):
            raise ValueError("acoustic emitters must be a list of mappings")
        values.extend(copy.deepcopy(external))
        if len(values) > 64:
            raise ValueError("acoustic emitter capacity exceeded")

        emitters: dict[str, dict[str, Any]] = {}
        entities: set[str] = set()
        for raw_emitter in values:
            allowed = {
                "type", "id", "entity", "drive", "tones", "energy_capacity",
                "initial_energy", "capture_efficiency", "impact_threshold",
                "min_impact_speed", "cooldown", "decay_time", "radiative_fraction",
                "reference_energy", "gain", "range", "occlusion", "hinge_damping",
                "max_hinge_torque", "source_offset",
            }
            unknown = set(raw_emitter) - allowed
            if unknown:
                raise ValueError(f"unknown acoustic emitter fields: {sorted(unknown)}")
            entity = _identifier(raw_emitter.get("entity"), "acoustic entity")
            emitter_id = _identifier(raw_emitter.get("id", entity), "acoustic emitter id")
            if emitter_id in emitters or entity in entities:
                raise ValueError("acoustic emitter ids and physical bindings must be unique")
            physical = self.world.acoustic_entity_state(entity)
            drive = raw_emitter.get("drive", "contact")
            if drive not in {"contact", "hinge", "both"}:
                raise ValueError("acoustic drive must be contact, hinge, or both")
            if drive in {"hinge", "both"} and physical["mobility"] != "hinge":
                raise ValueError("hinge acoustic drive requires a hinged entity")
            tones = _vector(raw_emitter.get("tones", [1, 0, 0]), 3, "tone weights", 0.0, 1.0)
            if sum(tones) <= 0.0:
                raise ValueError("tone weights cannot all be zero")
            capacity = _number(raw_emitter.get("energy_capacity", 0.05), "oscillator capacity", 1e-8, 100.0)
            emitter = {
                "id": emitter_id, "entity": entity, "drive": drive, "tones": tones,
                "energy_capacity": capacity,
                "initial_energy": _number(raw_emitter.get("initial_energy", 0.0), "initial oscillator energy", 0.0, capacity),
                "capture_efficiency": _number(raw_emitter.get("capture_efficiency", 0.25), "capture efficiency", 0.0, 1.0),
                "impact_threshold": _number(raw_emitter.get("impact_threshold", 1e-5), "impact threshold", 0.0, 10.0),
                "min_impact_speed": _number(raw_emitter.get("min_impact_speed", 0.02), "minimum impact speed", 0.0, 20.0),
                "cooldown": _number(raw_emitter.get("cooldown", 0.08), "emitter cooldown", 0.0, 10.0),
                "decay_time": _number(raw_emitter.get("decay_time", 0.7), "oscillator decay time", 0.01, 100.0),
                "radiative_fraction": _number(raw_emitter.get("radiative_fraction", 0.65), "radiative fraction", 0.0, 1.0),
                "reference_energy": _number(raw_emitter.get("reference_energy", 0.003), "reference energy", 1e-9, 100.0),
                "gain": _number(raw_emitter.get("gain", 1.0), "acoustic gain", 0.0, 10.0),
                "range": _number(raw_emitter.get("range", 1.8), "acoustic range", 0.05, 50.0),
                "occlusion": _number(raw_emitter.get("occlusion", 0.12), "occlusion transmission", 0.0, 1.0),
                "hinge_damping": _number(raw_emitter.get("hinge_damping", 0.002), "hinge damping", 0.0, 100.0),
                "max_hinge_torque": _number(raw_emitter.get("max_hinge_torque", 0.03), "hinge torque", 0.0, 100.0),
                "source_offset": _vector(raw_emitter.get("source_offset", [0, 0, 0]), 3, "source offset", -5.0, 5.0),
            }
            emitters[emitter_id] = emitter
            entities.add(entity)
        normalized = {
            "version": 1, "include_authored": False,
            "emitters": sorted((copy.deepcopy(value) for value in emitters.values()), key=lambda value: value["id"]),
        }
        return normalized, {emitter_id: emitters[emitter_id] for emitter_id in sorted(emitters)}

    def close(self) -> None:
        if getattr(self.world, "_acoustics", None) is self:
            self.world.attach_acoustics(None)

    def handles(self, entity_id: str) -> bool:
        return entity_id in self._by_entity

    def _stored_energy(self) -> float:
        return sum(sum(state["energy"]) for state in self._states.values())

    def ingest_contacts(self, events: list[dict[str, Any]]) -> None:
        self._ensure_native_binding()
        indices, work, speed = [], [], []
        ids = {key: index for index, key in enumerate(self._emitters)}
        for event in events:
            emitter_id = self._by_entity.get(event.get("entity"))
            if emitter_id is None:
                continue
            indices.append(ids[emitter_id])
            work.append(_number(event.get("impact_work"), "impact work", 0.0, 5.0))
            speed.append(_number(event.get("relative_normal_speed"), "impact speed", 0.0, 100.0))
        self._native.ingest_contacts(
            np.asarray(indices, dtype=np.int32), np.asarray(work), np.asarray(speed)
        )

    def before_substep(self, dt: float) -> None:
        """Apply energy-removing hinge loads before one MuJoCo substep."""
        step = _number(dt, "acoustic substep", 1e-7, 0.1)
        self._ensure_native_binding()
        self._native.before_substep(
            int(self.world.model._address), int(self.world.data._address), step
        )

    def advance(self, dt: float) -> dict[str, Any]:
        """Radiate and decay oscillator energy over a completed world interval."""
        duration = _number(dt, "acoustic dt", 1e-6, 60.0)
        self._ensure_native_binding()
        self._native.advance(duration)
        self._sync_native_state()
        self.time += duration
        return self.view()

    def sample(self, listener: np.ndarray, exclude_body: int) -> list[float]:
        """Sample current three-tone pressure amplitudes at a local point."""
        self._ensure_native_binding()
        point = np.ascontiguousarray(listener, dtype=np.float64)
        return np.asarray(self._native.sample(
            int(self.world.model._address), int(self.world.data._address), point,
            int(exclude_body),
        )).astype(float).tolist()

    def _energy_residual(self) -> float:
        expected = self._initial_energy + self._ledger["captured_energy"] - self._ledger["radiated_energy"] - self._ledger["oscillator_loss"]
        return self._stored_energy() - expected

    def _mechanical_residual(self) -> float:
        observed = self._ledger["contact_work_observed"] + self._ledger["hinge_work_extracted"]
        return observed - self._ledger["captured_energy"] - self._ledger["transduction_loss"]

    def view(self) -> dict[str, Any]:
        self._sync_native_state()
        sources = []
        for emitter_id, emitter in self._emitters.items():
            physical = self.world.acoustic_entity_state(emitter["entity"])
            source = np.asarray(physical["position"], dtype=float) + _rotation(physical["quaternion"]) @ np.asarray(emitter["source_offset"], dtype=float)
            sources.append({
                "id": emitter_id, "position": source.tolist(),
                "energy": list(map(float, self._states[emitter_id]["energy"])),
                "active": bool(sum(self._states[emitter_id]["energy"]) > 1e-12),
            })
        return {
            "time": self.time, "sources": sources, "ledger": copy.deepcopy(self._ledger),
            "energy_residual": self._energy_residual(), "mechanical_residual": self._mechanical_residual(),
        }

    def snapshot(self) -> dict[str, Any]:
        self._sync_native_state()
        return {
            "version": 1, "signature": self._signature, "config": copy.deepcopy(self.config),
            "time": self.time, "states": copy.deepcopy(self._states), "ledger": self._ledger.copy(),
            "initial_energy": self._initial_energy,
        }

    @classmethod
    def restore(cls, world: Any, snapshot: dict[str, Any]) -> "Acoustics":
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("unsupported acoustic snapshot")
        expected = {"version", "signature", "config", "time", "states", "ledger", "initial_energy"}
        if set(snapshot) != expected:
            raise ValueError("invalid acoustic snapshot fields")
        engine = cls(world, snapshot.get("config"))
        try:
            if snapshot.get("signature") != engine._signature:
                raise ValueError("acoustic configuration or physical bindings differ")
            if not isinstance(snapshot.get("states"), dict) or set(snapshot["states"]) != set(engine._states):
                raise ValueError("acoustic snapshot identities differ")
            restored = {}
            for emitter_id, baseline in engine._states.items():
                raw = snapshot["states"][emitter_id]
                if not isinstance(raw, dict) or set(raw) != set(baseline):
                    raise ValueError("invalid acoustic emitter state")
                energy = _vector(raw["energy"], 3, "oscillator energy", 0.0, engine._emitters[emitter_id]["energy_capacity"])
                if sum(energy) > engine._emitters[emitter_id]["energy_capacity"] + 1e-10:
                    raise ValueError("oscillator energy exceeds capacity")
                event_count = raw["event_count"]
                if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 0:
                    raise ValueError("invalid acoustic event count")
                restored[emitter_id] = {
                    "energy": energy,
                    "cooldown": _number(raw["cooldown"], "acoustic cooldown", 0.0, engine._emitters[emitter_id]["cooldown"]),
                    "event_count": event_count,
                }
            ledger = snapshot.get("ledger")
            if not isinstance(ledger, dict) or set(ledger) != set(_LEDGER_FIELDS):
                raise ValueError("invalid acoustic ledger")
            engine._states = restored
            engine._ledger = {key: _number(ledger[key], f"acoustic ledger {key}", 0.0, 1e15) for key in _LEDGER_FIELDS}
            engine._initial_energy = _number(snapshot.get("initial_energy"), "initial acoustic energy", 0.0, 1e9)
            engine.time = _number(snapshot.get("time"), "acoustic time", 0.0, 1e12)
            engine._bind_native()
            if abs(engine._energy_residual()) > 1e-8 or abs(engine._mechanical_residual()) > 1e-8:
                raise ValueError("acoustic snapshot violates its energy ledger")
            return engine
        except Exception:
            engine.close()
            raise


def demo() -> dict[str, Any]:
    from .physics import PhysicsWorld

    root = Path(__file__).resolve().parent.parent
    world = PhysicsWorld(seed=23)
    acoustics = Acoustics(world, root / "data/components/acoustic-play.json")
    world.command({"op": "impulse", "id": "pendulum", "impulse": [0.0, 1.0, 0.0]})
    for _ in range(30):
        world.advance({}, 0.05)
        acoustics.advance(0.05)
    return {"sound": world.sense("pip")["sound"], **acoustics.view()}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))


__all__ = ["Acoustics", "demo"]
