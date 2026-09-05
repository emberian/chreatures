"""Optional mass- and energy-accounted resource production for physical worlds.

Ecology is environment machinery, not a policy.  It receives physical poses and
anonymous light samples from the world, and only updates ordinary edible
components plus a narrowly bounded growth visualization.  Organism observations
never receive producer identities, reservoir contents, or bookkeeping values.
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
    "external_material_in", "external_food_in", "external_food_energy_in", "photon_energy_in",
    "consumed_mass_out", "consumed_energy_out", "turnover_mass_out",
    "turnover_energy_out", "conversion_energy_loss", "maintenance_energy_loss",
)


def _number(value: Any, name: str, low: float = 0.0, high: float = 1e9) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _vector(value: Any, length: int, name: str, low: float, high: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [_number(item, name, low, high) for item in value]


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 96 or not value[0].isalpha() or any(
        not (char.isalnum() or char in "_.-") for char in value
    ):
        raise ValueError(f"invalid {name}")
    return value


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


class Ecology:
    """Conservative production, turnover and renewal around a physical world."""

    def __init__(
        self,
        world: Any,
        config: dict[str, Any] | str | Path | None = None,
        *,
        seed: int | None = None,
    ):
        required = ("ecology_components", "ecology_food_amount", "sample_environment", "apply_growth_visual")
        if any(not callable(getattr(world, name, None)) for name in required):
            raise TypeError("world does not provide the ecology integration hooks")
        self.world = world
        raw = self._load_config(config)
        self.config, self._reservoirs, self._producers = self._resolve(raw)
        chosen_seed = getattr(world, "seed", 0) ^ 0xEC0106 if seed is None else seed
        if isinstance(chosen_seed, bool) or not isinstance(chosen_seed, (int, np.integer)):
            raise ValueError("ecology seed must be an integer")
        self.seed = int(chosen_seed)
        self.rng = np.random.default_rng(self.seed)
        ambient = self.config["ambient"]
        self._ambient_material = float(ambient["material"])
        self._light_noise = 0.0
        self.time = 0.0
        self._ledger = {name: 0.0 for name in _LEDGER_FIELDS}
        self._last_food = {
            producer_id: self.world.ecology_food_amount(producer["entity"])
            for producer_id, producer in self._producers.items()
        }
        self._visual_scale = {producer_id: 1.0 for producer_id in self._producers}
        self._initial_mass = self._system_mass()
        self._initial_energy = self._system_energy()
        topology = {
            "config": self.config,
            "producers": sorted((key, value["entity"], value["reservoir"]) for key, value in self._producers.items()),
        }
        self._signature = hashlib.sha256(_canonical(topology)).hexdigest()

    @staticmethod
    def _load_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        if config is None:
            return {"version": 1}
        if isinstance(config, dict):
            return copy.deepcopy(config)
        return json.loads(Path(config).read_text())

    def _resolve(self, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported ecology specification")
        include_authored = raw.get("include_authored", True)
        if not isinstance(include_authored, bool):
            raise ValueError("include_authored must be boolean")
        reservoir_specs: list[dict[str, Any]] = []
        producer_specs: list[dict[str, Any]] = []
        food_components: dict[str, dict[str, Any]] = {}
        shape_counts: dict[str, int] = {}
        authored = self.world.ecology_components()
        for record in authored:
            shape_counts[record["entity"]] = int(record["shape_count"])
            for component in record["components"]:
                if component["type"] == "food":
                    food_components[record["entity"]] = component
                elif include_authored and component["type"] == "reservoir":
                    reservoir_specs.append({**component, "entity": record["entity"]})
                elif include_authored and component["type"] == "producer":
                    producer_specs.append({**component, "entity": record["entity"]})
        for name, target in (("reservoirs", reservoir_specs), ("producers", producer_specs)):
            values = raw.get(name, [])
            if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
                raise ValueError(f"ecology {name} must be a list of mappings")
            target.extend(copy.deepcopy(values))

        reservoirs: dict[str, dict[str, float]] = {}
        normalized_reservoirs = []
        for item in reservoir_specs:
            entity = _identifier(item.get("entity"), "reservoir entity")
            reservoir_id = _identifier(item.get("id", entity), "reservoir id")
            if reservoir_id in reservoirs:
                raise ValueError("duplicate reservoir id")
            material = _number(item.get("material", 0.0), "reservoir material", 0.0, 1e6)
            material_capacity = _number(item.get("material_capacity", material), "reservoir material capacity", material, 1e6)
            energy = _number(item.get("energy", 0.0), "reservoir energy", 0.0, 1e6)
            energy_capacity = _number(item.get("energy_capacity", energy), "reservoir energy capacity", energy, 1e6)
            uptake_rate = _number(item.get("uptake_rate", 0.0), "reservoir uptake rate", 0.0, 1e3)
            value = {
                "material": material, "material_capacity": material_capacity,
                "energy": energy, "energy_capacity": energy_capacity,
                "uptake_rate": uptake_rate,
            }
            reservoirs[reservoir_id] = value.copy()
            normalized_reservoirs.append({"id": reservoir_id, "entity": entity, **value})

        producers: dict[str, dict[str, Any]] = {}
        normalized_producers = []
        entities: set[str] = set()
        for item in producer_specs:
            entity = _identifier(item.get("entity"), "producer entity")
            producer_id = _identifier(item.get("id", entity), "producer id")
            reservoir_id = _identifier(item.get("reservoir", entity), "producer reservoir")
            if producer_id in producers or entity in entities:
                raise ValueError("each producer and food entity must be unique")
            if reservoir_id not in reservoirs:
                raise ValueError(f"producer {producer_id} refers to an unknown reservoir")
            if entity not in food_components:
                # External configurations can target any existing food entity.
                try:
                    amount = self.world.ecology_food_amount(entity)
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"producer {producer_id} requires an edible physical entity") from exc
                food_components[entity] = {"amount": amount}
            food = food_components[entity]
            current_food = self.world.ecology_food_amount(entity)
            food_capacity = _number(item.get("food_capacity", food.get("capacity", max(1.0, current_food))), "food capacity", current_food, 100.0)
            energy_cost = _number(item.get("energy_cost", 1.0), "growth energy cost", 1e-6, 1e4)
            visual = self._visual_spec(item.get("visual"))
            if visual is not None and any(index >= shape_counts.get(entity, 0) for index in visual["shape_indices"]):
                raise ValueError("visual shape index does not exist on its physical entity")
            producer: dict[str, Any] = {
                "id": producer_id, "entity": entity, "reservoir": reservoir_id,
                "growth_rate": _number(item.get("growth_rate", 0.01), "growth rate", 0.0, 100.0),
                "maintenance_rate": _number(item.get("maintenance_rate", 0.0), "maintenance rate", 0.0, 100.0),
                "capture_area": _number(item.get("capture_area", 0.1), "capture area", 0.0, 100.0),
                "efficiency": _number(item.get("efficiency", 0.25), "capture efficiency", 0.0, 1.0),
                "light_half_saturation": _number(item.get("light_half_saturation", 0.3), "light half saturation", 1e-6, 10.0),
                "energy_cost": energy_cost,
                "energy_content": _number(item.get("energy_content", 0.8), "food energy content", 0.0, energy_cost),
                "turnover_rate": _number(item.get("turnover_rate", 0.0), "turnover rate", 0.0, 100.0),
                "recycle_fraction": _number(item.get("recycle_fraction", 0.0), "recycle fraction", 0.0, 1.0),
                "sample_offset": _vector(item.get("sample_offset", [0, 0, 0.12]), 3, "sample offset", -5.0, 5.0),
                "food_capacity": food_capacity,
                "visual": visual,
            }
            producers[producer_id] = producer
            entities.add(entity)
            normalized_producers.append(copy.deepcopy(producer))

        ambient_raw = raw.get("ambient", {})
        if not isinstance(ambient_raw, dict):
            raise ValueError("ambient ecology settings must be a mapping")
        material = _number(ambient_raw.get("material", 0.0), "ambient material", 0.0, 1e9)
        capacity = _number(ambient_raw.get("material_capacity", material), "ambient material capacity", material, 1e9)
        ambient = {
            "material": material, "material_capacity": capacity,
            "material_inflow_rate": _number(ambient_raw.get("material_inflow_rate", 0.0), "ambient material inflow", 0.0, 1e5),
            "photon_flux": _number(ambient_raw.get("photon_flux", 1.0), "ambient photon flux", 0.0, 10.0),
            "light_variability": _number(ambient_raw.get("light_variability", 0.0), "light variability", 0.0, 0.5),
            "light_correlation_time": _number(ambient_raw.get("light_correlation_time", 60.0), "light correlation time", 0.01, 1e6),
            "max_step": _number(ambient_raw.get("max_step", 0.5), "ecology max step", 0.001, 10.0),
        }
        normalized = {
            "version": 1, "include_authored": False, "ambient": ambient,
            "reservoirs": sorted(normalized_reservoirs, key=lambda value: value["id"]),
            "producers": sorted(normalized_producers, key=lambda value: value["id"]),
        }
        return normalized, reservoirs, producers

    @staticmethod
    def _visual_spec(raw: Any) -> dict[str, Any] | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("producer visual must be a mapping")
        indices = raw.get("shape_indices", [0])
        if not isinstance(indices, list) or not indices or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in indices
        ) or len(indices) != len(set(indices)) or any(value < 0 for value in indices):
            raise ValueError("visual shape indices are invalid")
        scale_range = _vector(raw.get("scale_range", [0.94, 1.06]), 2, "visual scale range", 0.85, 1.15)
        if scale_range[0] > scale_range[1]:
            raise ValueError("visual scale range must be increasing")
        result: dict[str, Any] = {
            "shape_indices": indices, "scale_range": scale_range,
            "max_scale_rate": _number(raw.get("max_scale_rate", 0.02), "visual scale rate", 0.0, 0.2),
            "exclusive_material": raw.get("exclusive_material", False),
        }
        if not isinstance(result["exclusive_material"], bool):
            raise ValueError("exclusive_material must be boolean")
        for key in ("empty_color", "full_color"):
            if key in raw:
                result[key] = _vector(raw[key], 3, key, 0.0, 1.0)
        if ("empty_color" in result) != ("full_color" in result):
            raise ValueError("growth visual requires both endpoint colors")
        if "empty_color" in result and not result["exclusive_material"]:
            raise ValueError("growth color requires exclusive_material")
        return result

    def _system_mass(self) -> float:
        return self._ambient_material + sum(value["material"] for value in self._reservoirs.values()) + sum(
            self.world.ecology_food_amount(producer["entity"]) for producer in self._producers.values()
        )

    def _system_energy(self) -> float:
        stored = sum(value["energy"] for value in self._reservoirs.values())
        food = sum(
            self.world.ecology_food_amount(producer["entity"]) * producer["energy_content"]
            for producer in self._producers.values()
        )
        return stored + food

    def _mass_residual(self) -> float:
        expected = (
            self._initial_mass + self._ledger["external_material_in"] + self._ledger["external_food_in"]
            - self._ledger["consumed_mass_out"] - self._ledger["turnover_mass_out"]
        )
        return self._system_mass() - expected

    def _energy_residual(self) -> float:
        expected = (
            self._initial_energy + self._ledger["photon_energy_in"]
            + self._ledger["external_food_energy_in"]
            - self._ledger["consumed_energy_out"] - self._ledger["turnover_energy_out"]
            - self._ledger["conversion_energy_loss"] - self._ledger["maintenance_energy_loss"]
        )
        return self._system_energy() - expected

    def _poses(self) -> dict[str, dict[str, Any]]:
        return {record["entity"]: record for record in self.world.ecology_components()}

    def _detect_food_changes(self) -> None:
        for producer_id, producer in self._producers.items():
            current = self.world.ecology_food_amount(producer["entity"])
            previous = self._last_food[producer_id]
            delta = current - previous
            if delta < -1e-12:
                removed = -delta
                self._ledger["consumed_mass_out"] += removed
                self._ledger["consumed_energy_out"] += removed * producer["energy_content"]
            elif delta > 1e-12:
                self._ledger["external_food_in"] += delta
                self._ledger["external_food_energy_in"] += delta * producer["energy_content"]
            self._last_food[producer_id] = current

    def _advance_ambient(self, step: float) -> None:
        ambient = self.config["ambient"]
        inflow = min(ambient["material_capacity"] - self._ambient_material, ambient["material_inflow_rate"] * step)
        self._ambient_material += inflow
        self._ledger["external_material_in"] += inflow
        capacity = ambient["material_capacity"]
        concentration = 0.0 if capacity <= 0.0 else self._ambient_material / capacity
        demands = {
            reservoir_id: value["uptake_rate"] * concentration
            * max(0.0, 1.0 - value["material"] / max(value["material_capacity"], 1e-12)) * step
            for reservoir_id, value in self._reservoirs.items()
        }
        total = sum(demands.values())
        fraction = 0.0 if total <= 0.0 else min(1.0, self._ambient_material / total)
        for reservoir_id in sorted(demands):
            transfer = demands[reservoir_id] * fraction
            self._ambient_material -= transfer
            self._reservoirs[reservoir_id]["material"] += transfer

    def _environment(self) -> dict[str, dict[str, float]]:
        poses = self._poses()
        points = []
        ids = []
        for producer_id in sorted(self._producers):
            producer = self._producers[producer_id]
            pose = poses.get(producer["entity"])
            if pose is None:
                raise RuntimeError("physical producer entity disappeared")
            rotation = _rotation(pose.get("quaternion", [1, 0, 0, 0]))
            point = np.asarray(pose["position"], dtype=float) + rotation @ np.asarray(producer["sample_offset"], dtype=float)
            point = np.clip(point, [0.0, 0.0, 0.0], [self.world.width, self.world.height, self.world.depth])
            points.append(point.tolist())
            ids.append(producer_id)
        return dict(zip(ids, self.world.sample_environment(points), strict=True))

    def _advance_producers(self, step: float, environment: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        ambient = self.config["ambient"]
        rho = math.exp(-step / ambient["light_correlation_time"])
        self._light_noise = float(np.clip(
            rho * self._light_noise + math.sqrt(max(0.0, 1.0 - rho * rho)) * self.rng.normal(), -3.0, 3.0
        ))
        variation = max(0.0, 1.0 + ambient["light_variability"] * self._light_noise)
        reports: dict[str, dict[str, float]] = {}
        for producer_id in sorted(self._producers):
            producer = self._producers[producer_id]
            reservoir = self._reservoirs[producer["reservoir"]]
            light = environment[producer_id]["illumination"] * ambient["photon_flux"] * variation
            capture = min(
                reservoir["energy_capacity"] - reservoir["energy"],
                light * producer["capture_area"] * producer["efficiency"] * step,
            )
            reservoir["energy"] += capture
            self._ledger["photon_energy_in"] += capture
            maintenance = min(reservoir["energy"], producer["maintenance_rate"] * step)
            reservoir["energy"] -= maintenance
            self._ledger["maintenance_energy_loss"] += maintenance

            food = self.world.ecology_food_amount(producer["entity"])
            saturation = light / (producer["light_half_saturation"] + light)
            potential = (
                producer["growth_rate"] * saturation
                * max(0.0, 1.0 - food / producer["food_capacity"]) * step
            )
            growth = min(
                potential, producer["food_capacity"] - food, reservoir["material"],
                reservoir["energy"] / producer["energy_cost"],
            )
            reservoir["material"] -= growth
            reservoir["energy"] -= growth * producer["energy_cost"]
            food += growth
            self._ledger["conversion_energy_loss"] += growth * (producer["energy_cost"] - producer["energy_content"])

            turnover = food * -math.expm1(-producer["turnover_rate"] * step)
            food -= turnover
            recycled = min(
                turnover * producer["recycle_fraction"],
                reservoir["material_capacity"] - reservoir["material"],
            )
            reservoir["material"] += recycled
            self._ledger["turnover_mass_out"] += turnover - recycled
            self._ledger["turnover_energy_out"] += turnover * producer["energy_content"]
            self.world.ecology_food_amount(producer["entity"], food)
            self._last_food[producer_id] = food
            reports[producer_id] = {
                "illumination": float(environment[producer_id]["illumination"]),
                "sky_exposure": float(environment[producer_id]["sky_exposure"]),
                "captured_energy": capture, "growth": growth, "turnover": turnover,
                "food": food, "reservoir_material": reservoir["material"],
                "reservoir_energy": reservoir["energy"],
            }
        return reports

    def _advance_visuals(self, step: float) -> None:
        for producer_id, producer in self._producers.items():
            visual = producer["visual"]
            if visual is None:
                continue
            fraction = self._last_food[producer_id] / producer["food_capacity"]
            low, high = visual["scale_range"]
            target = low + (high - low) * fraction
            current = self._visual_scale[producer_id]
            delta = np.clip(target - current, -visual["max_scale_rate"] * step, visual["max_scale_rate"] * step)
            scale = float(current + delta)
            color = None
            if "empty_color" in visual:
                color = (
                    np.asarray(visual["empty_color"]) * (1.0 - fraction)
                    + np.asarray(visual["full_color"]) * fraction
                ).tolist()
            self.world.apply_growth_visual(
                producer["entity"], visual["shape_indices"], scale, color,
                exclusive_material=visual["exclusive_material"],
            )
            self._visual_scale[producer_id] = scale

    def advance(self, dt: float) -> dict[str, Any]:
        """Advance ecology after physical consumption for the same interval."""
        duration = _number(dt, "ecology dt", 1e-6, 60.0)
        self._detect_food_changes()
        remaining = duration
        combined: dict[str, dict[str, float]] = {}
        while remaining > 1e-12:
            step = min(remaining, self.config["ambient"]["max_step"])
            self._advance_ambient(step)
            environment = self._environment()
            reports = self._advance_producers(step, environment)
            for producer_id, report in reports.items():
                if producer_id not in combined:
                    combined[producer_id] = {**report, "captured_energy": 0.0, "growth": 0.0, "turnover": 0.0}
                for key in ("captured_energy", "growth", "turnover"):
                    combined[producer_id][key] += report[key]
                for key in ("illumination", "sky_exposure", "food", "reservoir_material", "reservoir_energy"):
                    combined[producer_id][key] = report[key]
            self._advance_visuals(step)
            self.time += step
            remaining -= step
        return {
            "time": self.time, "ambient_material": self._ambient_material,
            "producers": combined, "ledger": copy.deepcopy(self._ledger),
            "mass_residual": self._mass_residual(), "energy_residual": self._energy_residual(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 1, "signature": self._signature, "seed": self.seed,
            "config": copy.deepcopy(self.config), "time": self.time,
            "ambient_material": self._ambient_material, "light_noise": self._light_noise,
            "reservoirs": copy.deepcopy(self._reservoirs), "last_food": self._last_food.copy(),
            "visual_scale": self._visual_scale.copy(), "ledger": self._ledger.copy(),
            "initial_mass": self._initial_mass, "initial_energy": self._initial_energy,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    @classmethod
    def restore(cls, world: Any, snapshot: dict[str, Any]) -> "Ecology":
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("unsupported ecology snapshot")
        ecology = cls(world, snapshot.get("config"), seed=snapshot.get("seed"))
        if snapshot.get("signature") != ecology._signature:
            raise ValueError("ecology configuration or physical bindings differ")
        reservoir_ids = set(ecology._reservoirs)
        producer_ids = set(ecology._producers)
        if set(snapshot.get("reservoirs", {})) != reservoir_ids or set(snapshot.get("last_food", {})) != producer_ids or set(snapshot.get("visual_scale", {})) != producer_ids:
            raise ValueError("ecology snapshot identities differ")
        restored_reservoirs: dict[str, dict[str, float]] = {}
        for reservoir_id, baseline in ecology._reservoirs.items():
            raw = snapshot["reservoirs"][reservoir_id]
            if not isinstance(raw, dict) or set(raw) != set(baseline):
                raise ValueError("invalid reservoir snapshot")
            restored = {key: _number(raw[key], f"reservoir {key}", 0.0, 1e9) for key in raw}
            if restored["material"] > restored["material_capacity"] or restored["energy"] > restored["energy_capacity"]:
                raise ValueError("reservoir snapshot exceeds capacity")
            if any(abs(restored[key] - baseline[key]) > 1e-12 for key in ("material_capacity", "energy_capacity", "uptake_rate")):
                raise ValueError("reservoir parameters changed")
            restored_reservoirs[reservoir_id] = restored
        ecology._reservoirs = restored_reservoirs
        ecology._ambient_material = _number(snapshot.get("ambient_material"), "ambient material", 0.0, ecology.config["ambient"]["material_capacity"])
        ecology._light_noise = _number(snapshot.get("light_noise"), "light noise", -3.0, 3.0)
        ecology.time = _number(snapshot.get("time"), "ecology time", 0.0, 1e12)
        ecology._initial_mass = _number(snapshot.get("initial_mass"), "initial mass", 0.0, 1e12)
        ecology._initial_energy = _number(snapshot.get("initial_energy"), "initial energy", 0.0, 1e12)
        ledger = snapshot.get("ledger")
        if not isinstance(ledger, dict) or set(ledger) != set(_LEDGER_FIELDS):
            raise ValueError("invalid ecology ledger")
        ecology._ledger = {key: _number(ledger[key], f"ledger {key}", 0.0, 1e15) for key in _LEDGER_FIELDS}
        for producer_id, producer in ecology._producers.items():
            amount = _number(snapshot["last_food"][producer_id], "last food", 0.0, producer["food_capacity"])
            if abs(world.ecology_food_amount(producer["entity"]) - amount) > 1e-9:
                raise ValueError("world food state does not match ecology snapshot")
            ecology._last_food[producer_id] = amount
            visual = producer["visual"]
            scale = _number(snapshot["visual_scale"][producer_id], "visual scale", 0.85, 1.15)
            ecology._visual_scale[producer_id] = scale
            if visual is not None:
                fraction = amount / producer["food_capacity"]
                color = None
                if "empty_color" in visual:
                    color = (
                        np.asarray(visual["empty_color"]) * (1.0 - fraction)
                        + np.asarray(visual["full_color"]) * fraction
                    ).tolist()
                world.apply_growth_visual(
                    producer["entity"], visual["shape_indices"], scale, color,
                    exclusive_material=visual["exclusive_material"],
                )
        try:
            ecology.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid ecology RNG state") from exc
        if abs(ecology._mass_residual()) > 1e-7 or abs(ecology._energy_residual()) > 1e-7:
            raise ValueError("ecology snapshot violates its conservation ledger")
        return ecology


def demo() -> dict[str, Any]:
    """Run a short renewable-food demonstration against the stock physics."""
    from .physics import PhysicsWorld

    root = Path(__file__).resolve().parent.parent
    world = PhysicsWorld(seed=31)
    ecology = Ecology(world, root / "data/ecology/portable-orchard.json", seed=31)
    before = world.ecology_food_amount("berry-a")
    for _ in range(120):
        ecology.advance(0.25)
    after = world.ecology_food_amount("berry-a")
    return {"food_before": before, "food_after": after, **ecology.advance(0.25)}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))


__all__ = ["Ecology", "demo"]
