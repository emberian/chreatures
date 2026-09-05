"""MuJoCo-backed three-dimensional habitat.

This module is deliberately parallel to :mod:`chreatures.world`.  MuJoCo owns
pose and motion; Python owns physiology, sensory transduction, scent/audio
fields, and the declarative entity components.  The prototype crawler's ciliary
traction is an explicitly supplied motor capability, isolated in
``_apply_crawler_forces`` so a later articulated body can replace it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = ROOT / "data/habitats/hollow-garden.json"
MODEL_DT = 0.05
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")
_SHAPES = {"box", "sphere", "capsule", "cylinder", "ellipsoid"}
_MOBILITY = {"static", "free", "hinge"}
_MUTABLE_MODEL_FIELDS = (
    "geom_size", "geom_pos", "geom_quat", "geom_rgba", "geom_friction",
    "geom_contype", "geom_conaffinity", "mat_rgba", "light_pos", "light_diffuse",
    "body_mass", "body_inertia", "eq_solref", "eq_solimp",
)


def _number(value: Any, name: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _vector(value: Any, length: int, name: str, bound: float = 1e5) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [_number(v, name, -bound, bound) for v in value]


def _unit_quaternion(value: Any, name: str = "quaternion") -> list[float]:
    quat = np.asarray(_vector(value, 4, name), dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        raise ValueError(f"{name} cannot be zero")
    return (quat / norm).tolist()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hex(rgba: list[float]) -> str:
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02x}" for c in rgba[:3])


@dataclass
class PhysicsBody:
    id: str
    name: str
    x: float
    y: float
    z: float
    heading: float
    radius: float = 0.10
    energy: float = 0.78
    gut: float = 0.12
    fatigue: float = 0.05
    speed: float = 0.0
    angular_velocity: float = 0.0
    age: float = 0.0
    color: str = "#ffffff"
    quaternion: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity3d: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gaze_pitch: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalObject:
    id: str
    kind: str
    x: float
    y: float
    z: float
    radius: float
    color: str
    odor: int | None = None
    amount: float = 1.0
    movable: bool = False
    quaternion: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    shape: str = "compound"
    components: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicsSignal:
    id: str
    x: float
    y: float
    z: float
    tone: int
    strength: float = 1.0
    remaining: float = 1.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhysicsWorld:
    """A z-up, meter-scale MuJoCo world with local creature sensing."""

    def __init__(self, seed: int = 7, spec: dict[str, Any] | str | Path | None = None):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.spec = self._load_spec(spec)
        self._validate_spec(self.spec)
        self.width, self.height, self.depth = map(float, self.spec["size"])
        self.time = 0.0
        self.signals: list[PhysicsSignal] = []
        self._next_entity_id = 1
        self._next_signal_id = 1
        self._touch: dict[str, list[float]] = {}
        self._contact_normals: dict[str, list[list[float]]] = {}
        self._signal_cooldown: dict[str, float] = {}
        self._grips: dict[str, str | None] = {}
        self._hand: dict[str, Any] | None = None
        self._acoustics: Any | None = None
        self._light = {"position": [6.0, 4.0, 3.0], "intensity": 0.0, "remaining": 0.0, "color": [1.0, 0.94, 0.78]}
        self._compile_model()
        self.bodies = self._make_bodies()
        self._components = {
            entity["id"]: copy.deepcopy(entity.get("components", [])) for entity in self._entities
        }
        self._resonance = {entity["id"]: 0.0 for entity in self._entities}
        for body in self.bodies:
            self._touch[body.id] = [0.0, 0.0]
            self._contact_normals[body.id] = []
            self._signal_cooldown[body.id] = 0.0
            self._grips[body.id] = None
        mujoco.mj_forward(self.model, self.data)
        self._sync_public_state()

    @staticmethod
    def _load_spec(spec: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        if spec is None:
            return json.loads(DEFAULT_SPEC.read_text())
        if isinstance(spec, dict):
            return copy.deepcopy(spec)
        return json.loads(Path(spec).read_text())

    @staticmethod
    def _validate_spec(spec: dict[str, Any]) -> None:
        if not isinstance(spec, dict) or spec.get("version") != 1:
            raise ValueError("unsupported habitat specification")
        size = _vector(spec.get("size"), 3, "size")
        if any(value <= 0.0 for value in size):
            raise ValueError("habitat size must be positive")
        _vector(spec.get("gravity"), 3, "gravity")
        _number(spec.get("physics_timestep"), "physics_timestep", 0.0001, 0.02)
        materials = spec.get("materials")
        physical = spec.get("physical_materials")
        if not isinstance(materials, dict) or not isinstance(physical, dict):
            raise ValueError("materials and physical_materials are required")
        compiler = spec.get("compiler", {})
        if not isinstance(compiler, dict):
            raise ValueError("compiler settings must be a mapping")
        if "material_order" in compiler:
            order = compiler["material_order"]
            if not isinstance(order, list) or not all(isinstance(name, str) for name in order) or len(order) != len(set(order)) or set(order) != set(materials):
                raise ValueError("compiler material_order must list every material exactly once")
        for name, material in materials.items():
            if not _ID.match(name) or not isinstance(material, dict):
                raise ValueError("invalid visual material")
            rgba = _vector(material.get("rgba"), 4, f"material {name} rgba", 2.0)
            if any(not 0.0 <= value <= 1.0 for value in rgba):
                raise ValueError("material rgba must be in [0, 1]")
        for name, material in physical.items():
            if not _ID.match(name) or not isinstance(material, dict):
                raise ValueError("invalid physical material")
            _number(material.get("density"), f"material {name} density", 1.0, 30_000.0)
            friction = _vector(material.get("friction"), 3, f"material {name} friction", 10.0)
            if any(value < 0.0 for value in friction):
                raise ValueError("friction cannot be negative")
        if not isinstance(spec.get("presets", {}), dict) or not isinstance(spec.get("entities"), list):
            raise ValueError("presets and entities must be declarative collections")
        body_ids: set[str] = set()
        for body in spec.get("bodies", []):
            if not isinstance(body, dict) or not _ID.match(str(body.get("id", ""))) or body["id"] in body_ids:
                raise ValueError("body ids must be unique")
            body_ids.add(body["id"])
            _vector(body.get("position"), 3, "body position")
            _number(body.get("heading", 0.0), "body heading", -1e4, 1e4)
        entity_ids: set[str] = set()
        presets = spec.get("presets", {})
        for raw in spec["entities"]:
            if not isinstance(raw, dict) or not _ID.match(str(raw.get("id", ""))) or raw["id"] in entity_ids or raw["id"] in body_ids:
                raise ValueError("entity ids must be globally unique")
            entity_ids.add(raw["id"])
            entity = PhysicsWorld._expand_entity(raw, presets)
            if entity.get("mobility") not in _MOBILITY:
                raise ValueError(f"invalid mobility for {entity['id']}")
            if entity.get("material") not in materials or entity.get("physical_material") not in physical:
                raise ValueError(f"unknown material on {entity['id']}")
            _vector(entity.get("position"), 3, f"{entity['id']} position")
            if "quaternion" in entity:
                _unit_quaternion(entity["quaternion"])
            if entity["mobility"] == "hinge":
                joint = entity.get("joint", {})
                if not isinstance(joint, dict):
                    raise ValueError(f"invalid hinge joint on {entity['id']}")
                axis = np.asarray(_vector(joint.get("axis", [0, 1, 0]), 3, "hinge axis"), dtype=float)
                if float(np.linalg.norm(axis)) < 1e-9:
                    raise ValueError("hinge axis cannot be zero")
                limits = _vector(joint.get("range", [-45, 45]), 2, "hinge range", 360.0)
                if limits[0] >= limits[1]:
                    raise ValueError("hinge range must be increasing")
                _number(joint.get("damping", 0.05), "hinge damping", 0.0, 1e4)
                _number(joint.get("initial", 0.0), "hinge initial", limits[0], limits[1])
            shapes = entity.get("shapes")
            if not isinstance(shapes, list) or not shapes:
                raise ValueError(f"{entity['id']} requires at least one shape")
            for shape in shapes:
                if not isinstance(shape, dict) or shape.get("type") not in _SHAPES:
                    raise ValueError(f"unsupported shape on {entity['id']}")
                size = shape.get("size")
                expected = {"sphere": (1,), "box": (3,), "ellipsoid": (3,), "cylinder": (2,)}
                if shape["type"] == "capsule":
                    expected_lengths = (1,) if "fromto" in shape else (2,)
                else:
                    expected_lengths = expected[shape["type"]]
                if not isinstance(size, list) or len(size) not in expected_lengths or any(_number(v, "shape size", 0.002, 20.0) <= 0 for v in size):
                    raise ValueError(f"invalid shape size on {entity['id']}")
                if "position" in shape:
                    _vector(shape["position"], 3, "shape position")
                if "quaternion" in shape:
                    _unit_quaternion(shape["quaternion"])
                if "fromto" in shape:
                    _vector(shape["fromto"], 6, "shape fromto")
            components = entity.get("components", [])
            if not isinstance(components, list) or any(not isinstance(c, dict) or not isinstance(c.get("type"), str) for c in components):
                raise ValueError(f"invalid components on {entity['id']}")
            for component in components:
                component_type = component["type"]
                if component_type == "food":
                    _number(component.get("amount"), "food amount", 0.0, 100.0)
                    _number(component.get("nutrition", 1.0), "food nutrition", 0.0, 10.0)
                    if "capacity" in component:
                        _number(component["capacity"], "food capacity", component["amount"], 100.0)
                elif component_type == "scent":
                    odor = component.get("odor")
                    if isinstance(odor, bool) or not isinstance(odor, int) or odor not in (0, 1, 2):
                        raise ValueError("scent odor must be 0, 1, or 2")
                    _number(component.get("strength", 1.0), "scent strength", 0.0, 10.0)
                elif component_type == "resonator":
                    tone = component.get("tone")
                    if isinstance(tone, bool) or not isinstance(tone, int) or tone not in (0, 1, 2):
                        raise ValueError("resonator tone must be 0, 1, or 2")
                    _number(component.get("gain", 1.0), "resonator gain", 0.0, 10.0)
                elif component_type == "shade":
                    _number(component.get("radius", 1.0), "shade radius", 0.01, 20.0)
                    _number(component.get("strength", 1.0), "shade strength", 0.0, 1.0)
                elif component_type == "light":
                    _vector(component.get("position", [0, 0, 0]), 3, "light position", 20.0)
                    direction = np.asarray(_vector(component.get("direction", [0, 0, -1]), 3, "light direction"), dtype=float)
                    if float(np.linalg.norm(direction)) < 1e-9:
                        raise ValueError("light direction cannot be zero")
                    color = _vector(component.get("color", [1, 1, 1]), 3, "light color", 1.0)
                    if any(value < 0.0 for value in color):
                        raise ValueError("light color cannot be negative")
                    _number(component.get("intensity", 1.0), "light intensity", 0.0, 1.0)
                    _number(component.get("radius", 2.0), "light radius", 0.05, 20.0)
                elif component_type == "reservoir":
                    reservoir_id = component.get("id", entity["id"])
                    if not isinstance(reservoir_id, str) or not _ID.match(reservoir_id):
                        raise ValueError("reservoir id is invalid")
                    material = _number(component.get("material", 0.0), "reservoir material", 0.0, 1e6)
                    material_capacity = _number(component.get("material_capacity", material), "reservoir material capacity", material, 1e6)
                    energy = _number(component.get("energy", 0.0), "reservoir energy", 0.0, 1e6)
                    _number(component.get("energy_capacity", energy), "reservoir energy capacity", energy, 1e6)
                    _number(component.get("uptake_rate", 0.0), "reservoir uptake rate", 0.0, 1e3)
                elif component_type == "producer":
                    producer_id = component.get("id", entity["id"])
                    reservoir_id = component.get("reservoir", entity["id"])
                    if not isinstance(producer_id, str) or not _ID.match(producer_id):
                        raise ValueError("producer id is invalid")
                    if not isinstance(reservoir_id, str) or not _ID.match(reservoir_id):
                        raise ValueError("producer reservoir id is invalid")
                    _number(component.get("growth_rate", 0.01), "producer growth rate", 0.0, 100.0)
                    _number(component.get("maintenance_rate", 0.0), "producer maintenance rate", 0.0, 100.0)
                    _number(component.get("capture_area", 0.1), "producer capture area", 0.0, 100.0)
                    _number(component.get("efficiency", 0.25), "producer efficiency", 0.0, 1.0)
                    _number(component.get("light_half_saturation", 0.3), "light half saturation", 1e-6, 10.0)
                    energy_cost = _number(component.get("energy_cost", 1.0), "growth energy cost", 1e-6, 1e4)
                    _number(component.get("energy_content", 0.8), "food energy content", 0.0, energy_cost)
                    _number(component.get("turnover_rate", 0.0), "producer turnover rate", 0.0, 100.0)
                    _number(component.get("recycle_fraction", 0.0), "producer recycle fraction", 0.0, 1.0)
                    _vector(component.get("sample_offset", [0, 0, 0.12]), 3, "producer sample offset", 5.0)
                    if "food_capacity" in component:
                        _number(component["food_capacity"], "producer food capacity", 1e-6, 100.0)
                    visual = component.get("visual")
                    if visual is not None:
                        if not isinstance(visual, dict):
                            raise ValueError("producer visual must be a mapping")
                        indices = visual.get("shape_indices", [0])
                        if not isinstance(indices, list) or not indices or not all(isinstance(i, int) and not isinstance(i, bool) for i in indices) or len(set(indices)) != len(indices) or any(i < 0 or i >= len(shapes) for i in indices):
                            raise ValueError("producer visual shape indices are invalid")
                        scale = _vector(visual.get("scale_range", [0.94, 1.06]), 2, "producer scale range", 1.15)
                        if scale[0] < 0.85 or scale[0] > scale[1] or scale[1] > 1.15:
                            raise ValueError("producer scale range must stay within [0.85, 1.15]")
                        _number(visual.get("max_scale_rate", 0.02), "producer visual rate", 0.0, 0.2)
                        for key in ("empty_color", "full_color"):
                            if key in visual:
                                color = _vector(visual[key], 3, f"producer {key}", 1.0)
                                if any(value < 0.0 for value in color):
                                    raise ValueError("producer visual colors cannot be negative")
                        if "exclusive_material" in visual and not isinstance(visual["exclusive_material"], bool):
                            raise ValueError("exclusive_material must be boolean")
                        has_empty, has_full = "empty_color" in visual, "full_color" in visual
                        if has_empty != has_full or (has_empty and not visual.get("exclusive_material", False)):
                            raise ValueError("producer growth color requires two endpoints and an exclusive material")
                elif component_type == "acoustic_resonator":
                    emitter_id = component.get("id", entity["id"])
                    if not isinstance(emitter_id, str) or not _ID.match(emitter_id):
                        raise ValueError("acoustic emitter id is invalid")
                    drive = component.get("drive", "contact")
                    if drive not in {"contact", "hinge", "both"}:
                        raise ValueError("acoustic drive must be contact, hinge, or both")
                    if drive in {"hinge", "both"} and entity["mobility"] != "hinge":
                        raise ValueError("acoustic hinge drive requires a hinged entity")
                    tones = _vector(component.get("tones", [1, 0, 0]), 3, "acoustic tones", 1.0)
                    if any(value < 0.0 for value in tones) or sum(tones) <= 0.0:
                        raise ValueError("acoustic tones require positive weight")
                    capacity = _number(component.get("energy_capacity", 0.05), "acoustic capacity", 1e-8, 100.0)
                    _number(component.get("initial_energy", 0.0), "acoustic initial energy", 0.0, capacity)
                    _number(component.get("capture_efficiency", 0.25), "acoustic capture efficiency", 0.0, 1.0)
                    _number(component.get("impact_threshold", 1e-5), "acoustic impact threshold", 0.0, 10.0)
                    _number(component.get("min_impact_speed", 0.02), "acoustic impact speed", 0.0, 20.0)
                    _number(component.get("cooldown", 0.08), "acoustic cooldown", 0.0, 10.0)
                    _number(component.get("decay_time", 0.7), "acoustic decay time", 0.01, 100.0)
                    _number(component.get("radiative_fraction", 0.65), "acoustic radiative fraction", 0.0, 1.0)
                    _number(component.get("reference_energy", 0.003), "acoustic reference energy", 1e-9, 100.0)
                    _number(component.get("gain", 1.0), "acoustic gain", 0.0, 10.0)
                    _number(component.get("range", 1.8), "acoustic range", 0.05, 50.0)
                    _number(component.get("occlusion", 0.12), "acoustic occlusion", 0.0, 1.0)
                    _number(component.get("hinge_damping", 0.002), "acoustic hinge damping", 0.0, 100.0)
                    _number(component.get("max_hinge_torque", 0.03), "acoustic hinge torque", 0.0, 100.0)
                    _vector(component.get("source_offset", [0, 0, 0]), 3, "acoustic source offset", 5.0)
        limits = spec.get("limits", {})
        if not isinstance(limits, dict):
            raise ValueError("habitat limits must be a mapping")
        _number(limits.get("acoustic_impulse", 10.0), "acoustic impulse limit", 1e-6, 10.0)
        _number(limits.get("acoustic_work", 5.0), "acoustic work limit", 1e-8, 5.0)
        if len(spec["entities"]) > int(limits.get("entities", 96)):
            raise ValueError("entity capacity exceeded")

    @staticmethod
    def _expand_entity(raw: dict[str, Any], presets: dict[str, Any]) -> dict[str, Any]:
        if "preset" not in raw:
            return copy.deepcopy(raw)
        preset = raw["preset"]
        if preset not in presets:
            raise ValueError(f"unknown entity preset: {preset}")
        result = copy.deepcopy(presets[preset])
        result.update(copy.deepcopy(raw))
        result.pop("preset", None)
        return result

    @staticmethod
    def _attrs(values: dict[str, Any]) -> str:
        parts = []
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, np.ndarray)):
                value = " ".join(f"{float(v):.10g}" for v in value)
            parts.append(f'{key}="{html.escape(str(value), quote=True)}"')
        return " ".join(parts)

    def build_body(self, config: dict[str, Any]) -> str:
        """Construct one crawler; override this seam for an articulated body."""
        prototype = self.spec["body_prototype"]
        body_id = config["id"]
        heading = float(config.get("heading", 0.0))
        quat = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
        physical = self.spec["physical_materials"][prototype["physical_material"]]
        shape = prototype["shape"]
        body_attrs = self._attrs({"name": f"resident:{body_id}", "pos": config["position"], "quat": quat})
        geom_attrs = self._attrs({
            "name": f"resident:{body_id}:shell", "type": shape["type"], "size": shape["size"],
            "material": f"mat:{config['material']}", "density": physical["density"],
            "friction": physical["friction"], "condim": 4,
        })
        return f'<body {body_attrs}><freejoint name="resident:{body_id}:free"/><geom {geom_attrs}/></body>'

    def _entity_xml(self, entity: dict[str, Any]) -> str:
        entity_id = entity["id"]
        body_attrs = {"name": f"entity:{entity_id}", "pos": entity["position"]}
        if "quaternion" in entity:
            body_attrs["quat"] = _unit_quaternion(entity["quaternion"])
        pieces = [f"<body {self._attrs(body_attrs)}>"]
        mobility = entity["mobility"]
        if mobility == "free":
            pieces.append(f'<freejoint name="entity:{entity_id}:free"/>')
        elif mobility == "hinge":
            joint = entity.get("joint", {})
            pieces.append(
                f'<joint {self._attrs({"name": f"entity:{entity_id}:hinge", "type": "hinge", "axis": joint.get("axis", [0, 1, 0]), "range": joint.get("range", [-45, 45]), "limited": "true", "damping": joint.get("damping", 0.05), "armature": 0.001})}/>'
            )
        physical = self.spec["physical_materials"][entity["physical_material"]]
        for index, shape in enumerate(entity["shapes"]):
            material = shape.get("material", entity["material"])
            attrs = {
                "name": f"entity:{entity_id}:geom:{index}", "type": shape["type"], "size": shape["size"],
                "pos": shape.get("position"), "quat": _unit_quaternion(shape["quaternion"]) if "quaternion" in shape else None,
                "fromto": shape.get("fromto"), "material": f"mat:{material}", "density": physical["density"],
                "friction": physical["friction"], "condim": 4,
            }
            pieces.append(f"<geom {self._attrs(attrs)}/>")
        for index, component in enumerate(entity.get("components", [])):
            if component.get("type") != "light":
                continue
            intensity = float(component.get("intensity", 1.0))
            color = np.asarray(component.get("color", [1.0, 1.0, 1.0]), dtype=float) * intensity
            direction = np.asarray(component.get("direction", [0, 0, -1]), dtype=float)
            direction /= np.linalg.norm(direction)
            pieces.append(f'<light {self._attrs({"name": f"entity:{entity_id}:light:{index}", "mode": "fixed", "pos": component.get("position", [0, 0, 0]), "dir": direction, "diffuse": color, "specular": color * 0.15, "directional": "false", "castshadow": "true", "cutoff": 70, "exponent": 1.0})}/>')
        pieces.append("</body>")
        return "".join(pieces)

    def _compile_model(self) -> None:
        self._entities = [self._expand_entity(raw, self.spec.get("presets", {})) for raw in self.spec["entities"]]
        assets = []
        material_order = self.spec.get("compiler", {}).get("material_order", self.spec["materials"])
        for name in material_order:
            material = self.spec["materials"][name]
            assets.append(f'<material {self._attrs({"name": f"mat:{name}", "rgba": material["rgba"]})}/>')
        world = [
            '<light name="caregiver-light" pos="6 4 3" dir="0 0 -1" diffuse="0 0 0" specular="0 0 0" directional="false"/>',
            '<body name="caregiver-hand" mocap="true" pos="0 0 -10"><geom name="caregiver-hand:geom" type="sphere" size="0.085" rgba="0.95 0.88 0.68 0.75" contype="0" conaffinity="0"/></body>',
        ]
        world.extend(self.build_body(body) for body in self.spec["bodies"])
        world.extend(self._entity_xml(entity) for entity in self._entities)
        xml = (
            '<mujoco model="hollow-garden"><compiler angle="degree" coordinate="local"/>'
            f'<option timestep="{float(self.spec["physics_timestep"]):.10g}" gravity="{self._attrs_value(self.spec["gravity"])}" integrator="implicitfast" cone="elliptic"/>'
            '<size njmax="3000" nconmax="800"/><visual><map znear="0.01"/></visual>'
            f'<asset>{"".join(assets)}</asset><worldbody>{"".join(world)}</worldbody></mujoco>'
        )
        self._xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._model_signature = hashlib.sha256(_canonical({
            "spec": self.spec, "mujoco": mujoco.__version__,
            "compiled_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        })).hexdigest()
        self._body_mj = {
            item["id"]: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"resident:{item['id']}")
            for item in self.spec["bodies"]
        }
        self._body_joint = {
            item["id"]: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"resident:{item['id']}:free")
            for item in self.spec["bodies"]
        }
        self._entity_mj = {
            entity["id"]: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"entity:{entity['id']}")
            for entity in self._entities
        }
        self._entity_joint = {}
        for entity in self._entities:
            if entity["mobility"] != "static":
                suffix = "free" if entity["mobility"] == "free" else "hinge"
                self._entity_joint[entity["id"]] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"entity:{entity['id']}:{suffix}"
                )
        for entity in self._entities:
            if entity["mobility"] == "hinge":
                initial = float(entity.get("joint", {}).get("initial", 0.0))
                joint_id = self._entity_joint[entity["id"]]
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = math.radians(initial)
        self._geom_entity: dict[int, str] = {}
        self._geom_resident: dict[int, str] = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("entity:"):
                self._geom_entity[geom_id] = name.split(":", 2)[1]
            elif name.startswith("resident:"):
                self._geom_resident[geom_id] = name.split(":", 2)[1]

    @staticmethod
    def _attrs_value(value: list[float]) -> str:
        return " ".join(f"{float(v):.10g}" for v in value)

    def _make_bodies(self) -> list[PhysicsBody]:
        result = []
        prototype = self.spec["body_prototype"]
        for config in self.spec["bodies"]:
            rgba = self.spec["materials"][config["material"]]["rgba"]
            result.append(PhysicsBody(
                id=config["id"], name=config["name"], x=float(config["position"][0]),
                y=float(config["position"][1]), z=float(config["position"][2]),
                heading=float(config.get("heading", 0.0)), radius=float(prototype.get("radius", 0.1)),
                energy=float(config.get("energy", 0.78)), gut=float(config.get("gut", 0.12)),
                fatigue=float(config.get("fatigue", 0.05)), color=_hex(rgba),
            ))
        return result

    def _body(self, body_id: str) -> PhysicsBody:
        for body in self.bodies:
            if body.id == body_id:
                return body
        raise KeyError(f"unknown body id: {body_id}")

    def _entity(self, entity_id: str) -> dict[str, Any]:
        for entity in self._entities:
            if entity["id"] == entity_id:
                return entity
        raise KeyError(f"unknown entity id: {entity_id}")

    def _pose(self, entity_id: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = self._entity_mj[entity_id]
        return self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()

    def _velocity(self, mj_body_id: int) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, mj_body_id, velocity, 0)
        return velocity[3:].copy(), velocity[:3].copy()

    def _sync_public_state(self) -> None:
        for body in self.bodies:
            mj_body = self._body_mj[body.id]
            pos, quat = self.data.xpos[mj_body], self.data.xquat[mj_body]
            rotation = self.data.xmat[mj_body].reshape(3, 3)
            linear, angular = self._velocity(mj_body)
            body.x, body.y, body.z = map(float, pos)
            body.quaternion = quat.astype(float).tolist()
            body.heading = float(math.atan2(rotation[1, 0], rotation[0, 0]) % (2 * math.pi))
            body.linear_velocity = linear.astype(float).tolist()
            body.angular_velocity3d = angular.astype(float).tolist()
            body.speed = float(np.linalg.norm(linear[:2]))
            body.angular_velocity = float(angular[2])
        self.objects = [self._public_object(entity) for entity in self._entities]

    def _public_object(self, entity: dict[str, Any]) -> PhysicalObject:
        pos, quat = self._pose(entity["id"])
        components = copy.deepcopy(self._components.get(entity["id"], entity.get("components", [])))
        scent = next((c for c in components if c.get("type") == "scent"), None)
        food = next((c for c in components if c.get("type") == "food"), None)
        sizes = [max(map(float, shape["size"])) for shape in entity["shapes"]]
        kind = "food" if food else "toy" if entity["mobility"] == "free" else "structure"
        return PhysicalObject(
            id=entity["id"], kind=kind, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
            radius=max(sizes), color=_hex(self.spec["materials"][entity["material"]]["rgba"]),
            odor=None if scent is None else int(scent["odor"]),
            amount=1.0 if food is None else float(food["amount"]), movable=entity["mobility"] == "free",
            quaternion=quat.astype(float).tolist(),
            shape=entity["shapes"][0]["type"] if len(entity["shapes"]) == 1 else "compound",
            components=components,
        )

    def ecology_components(self) -> list[dict[str, Any]]:
        """Return authored ecology components for an optional environment layer.

        This integration view is deliberately separate from ``sense``: it is
        world machinery and is never included in an organism observation.
        """
        self._sync_public_state()
        result = []
        for entity in self._entities:
            components = [
                copy.deepcopy(component) for component in self._components[entity["id"]]
                if component.get("type") in {"food", "producer", "reservoir"}
            ]
            if components:
                position, quaternion = self._pose(entity["id"])
                result.append({
                    "entity": entity["id"], "position": position.astype(float).tolist(),
                    "quaternion": quaternion.astype(float).tolist(), "shape_count": len(entity["shapes"]),
                    "components": components,
                })
        return result

    def ecology_food_amount(self, entity_id: str, value: float | None = None) -> float:
        """Read or safely update the ordinary edible component on an entity."""
        if not isinstance(entity_id, str) or entity_id not in self._components:
            raise ValueError("unknown ecology entity")
        food = next((component for component in self._components[entity_id] if component.get("type") == "food"), None)
        if food is None:
            raise ValueError("ecology entity has no food component")
        if value is not None:
            capacity = float(food.get("capacity", 100.0))
            food["amount"] = _number(value, "food amount", 0.0, capacity)
            self._sync_public_state()
        return float(food["amount"])

    def apply_growth_visual(
        self,
        entity_id: str,
        shape_indices: list[int],
        scale: float,
        color: list[float] | None = None,
        *,
        exclusive_material: bool = False,
    ) -> None:
        """Apply a tightly bounded, absolute growth visualization.

        Absolute scaling from authored dimensions avoids accumulating numeric
        drift. The narrow scale range prevents ecology updates from creating a
        sudden large collider. Recoloring is allowed only for materials unused
        by any other entity.
        """
        entity = self._entity(entity_id)
        factor = _number(scale, "growth visual scale", 0.85, 1.15)
        if not isinstance(shape_indices, list) or not shape_indices or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(entity["shapes"])
            for index in shape_indices
        ):
            raise ValueError("growth visual shape indices are invalid")
        geom_ids: list[int] = []
        for index in shape_indices:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"entity:{entity_id}:geom:{index}")
            geom_ids.append(geom_id)
        rgb: np.ndarray | None = None
        selected_materials: set[int] = set()
        if color is not None:
            rgb = np.asarray(_vector(color, 3, "growth visual color", 1.0), dtype=float)
            if np.any(rgb < 0.0) or not exclusive_material:
                raise ValueError("growth color requires a nonnegative exclusive material")
            selected_materials = {int(self.model.geom_matid[geom_id]) for geom_id in geom_ids}
            if -1 in selected_materials:
                raise ValueError("growth color requires an authored material")
            for material_id in selected_materials:
                users = np.flatnonzero(self.model.geom_matid == material_id)
                if any(self._geom_entity.get(int(geom_id)) != entity_id for geom_id in users):
                    raise ValueError("growth material is shared by another entity")
        for index, geom_id in zip(shape_indices, geom_ids, strict=True):
            authored_size = np.asarray(entity["shapes"][index]["size"], dtype=float)
            self.model.geom_size[geom_id, : len(authored_size)] = authored_size * factor
        if rgb is not None:
            for material_id in selected_materials:
                self.model.mat_rgba[material_id, :3] = rgb
        mujoco.mj_forward(self.model, self.data)
        self._sync_public_state()

    def acoustic_components(self) -> list[dict[str, Any]]:
        """Return authored acoustic transducers outside organism observation."""
        result = []
        for entity in self._entities:
            components = [
                copy.deepcopy(component) for component in self._components[entity["id"]]
                if component.get("type") == "acoustic_resonator"
            ]
            if components:
                result.append({"entity": entity["id"], "components": components})
        return result

    def attach_acoustics(self, engine: Any | None) -> None:
        """Attach one optional local acoustic transducer engine."""
        if engine is not None and (
            not callable(getattr(engine, "ingest_contact", None))
            or not callable(getattr(engine, "before_substep", None))
            or not callable(getattr(engine, "sample", None))
            or not callable(getattr(engine, "handles", None))
        ):
            raise TypeError("acoustic engine does not implement the integration protocol")
        if engine is not None and self._acoustics is not None and self._acoustics is not engine:
            raise RuntimeError("an acoustic engine is already attached")
        self._acoustics = engine

    def acoustic_entity_state(self, entity_id: str) -> dict[str, Any]:
        """Expose physical source pose and hinge motion to environment machinery."""
        entity = self._entity(entity_id)
        position, quaternion = self._pose(entity_id)
        value: dict[str, Any] = {
            "mobility": entity["mobility"], "position": position.astype(float).tolist(),
            "quaternion": quaternion.astype(float).tolist(),
        }
        if entity["mobility"] == "hinge":
            joint_id = self._entity_joint[entity_id]
            dof = int(self.model.jnt_dofadr[joint_id])
            velocity = float(self.data.qvel[dof])
            value.update({
                "joint_velocity": velocity, "joint_inertia": float(self.model.dof_M0[dof]),
                "joint_energy": 0.5 * float(self.model.dof_M0[dof]) * velocity * velocity,
            })
        return value

    def apply_acoustic_hinge_torque(self, entity_id: str, torque: float) -> None:
        entity = self._entity(entity_id)
        if entity["mobility"] != "hinge":
            raise ValueError("acoustic torque requires a hinged entity")
        value = _number(torque, "acoustic hinge torque", -100.0, 100.0)
        joint_id = self._entity_joint[entity_id]
        self.data.qfrc_applied[self.model.jnt_dofadr[joint_id]] += value

    def acoustic_visibility(
        self,
        listener: list[float] | np.ndarray,
        source: list[float] | np.ndarray,
        source_entity: str | None,
        exclude_body: int,
        transmission: float,
    ) -> float:
        """Return direct-path sound transmission through current geometry."""
        start = np.asarray(listener, dtype=float)
        end = np.asarray(source, dtype=float)
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return 1.0
        ray_distance, geom_id = self._ray(start, delta / distance, exclude_body)
        hit_entity = self._geom_entity.get(geom_id)
        if ray_distance < 0.0 or ray_distance >= distance - 0.07 or (
            source_entity is not None and hit_entity == source_entity
        ):
            return 1.0
        return _number(transmission, "acoustic transmission", 0.0, 1.0)

    def sense(self, body_id: str) -> dict[str, Any]:
        body = self._body(body_id)
        vision = self._vision(body)
        rotation = self.data.xmat[self._body_mj[body.id]].reshape(3, 3)
        local_linear = rotation.T @ np.asarray(body.linear_velocity)
        local_angular = rotation.T @ np.asarray(body.angular_velocity3d)
        return {
            "odor": self._odor(body), "vision": vision, "retina3d": self._retina3d(body),
            "touch": list(self._touch[body.id]), "contact_normals": copy.deepcopy(self._contact_normals[body.id]),
            "sound": self._sound(body), "shade": self._shade(body),
            "illumination": self._illumination(body), "speed": body.speed,
            "angular_velocity": body.angular_velocity,
            "linear_velocity": local_linear.astype(float).tolist(),
            "angular_velocity3d": local_angular.astype(float).tolist(),
            "energy": body.energy, "gut": body.gut, "fatigue": body.fatigue,
        }

    def _odor(self, body: PhysicsBody) -> list[list[float]]:
        sigma = 0.82
        h = np.array([math.cos(body.heading), math.sin(body.heading), 0.0])
        right = np.array([-h[1], h[0], 0.0])
        center = np.array([body.x, body.y, body.z]) + h * 0.105 + np.array([0.0, 0.0, 0.035])
        antennae = (center - right * 0.055, center + right * 0.055)
        result = np.zeros((2, 3), dtype=float)
        for entity in self._entities:
            scent = next((c for c in self._components[entity["id"]] if c.get("type") == "scent"), None)
            if scent is None:
                continue
            food = next((c for c in self._components[entity["id"]] if c.get("type") == "food"), None)
            availability = max(0.0, float(food["amount"])) if food else 1.0
            if availability <= 0.0:
                continue
            source, _ = self._pose(entity["id"])
            for side, antenna in enumerate(antennae):
                d2 = float(np.dot(source - antenna, source - antenna))
                if d2 <= (4 * sigma) ** 2:
                    result[side, int(scent["odor"])] += float(scent.get("strength", 1.0)) * availability * math.exp(-d2 / (2 * sigma * sigma))
        return np.clip(result, 0.0, 4.0).tolist()

    def _ray(self, origin: np.ndarray, direction: np.ndarray, exclude_body: int) -> tuple[float, int]:
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, origin, direction, None, 1, exclude_body, geom_id)
        return float(distance), int(geom_id[0])

    def _geom_rgb(self, geom_id: int) -> list[float]:
        material = int(self.model.geom_matid[geom_id])
        rgba = self.model.mat_rgba[material] if material >= 0 else self.model.geom_rgba[geom_id]
        return [float(v) for v in rgba[:3]]

    def _vision(self, body: PhysicsBody, pitch_offset: float = 0.0) -> list[list[float]]:
        max_range = 3.2
        fan = math.radians(150)
        pitch = float(np.clip(body.gaze_pitch * 0.62 + pitch_offset, -1.15, 1.15))
        origin = np.array([body.x, body.y, body.z + 0.035], dtype=float)
        illumination = self._illumination(body)
        rows = []
        for offset in np.linspace(-fan / 2, fan / 2, 16):
            yaw = body.heading + float(offset)
            direction = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
            distance, geom_id = self._ray(origin, direction, self._body_mj[body.id])
            if distance < 0.0 or distance > max_range or geom_id < 0:
                rows.append([0.0, 0.0, 0.0, 0.0])
            else:
                rgb = self._geom_rgb(geom_id)
                rgb = [min(1.0, value * (0.45 + 0.55 * illumination)) for value in rgb]
                rows.append(rgb + [max(0.0, 1.0 - distance / max_range)])
        return rows

    def _retina3d(self, body: PhysicsBody) -> list[list[list[float]]]:
        """Five elevation bands of the same occluded physical ray fan."""
        return [self._vision(body, offset) for offset in (-0.42, -0.21, 0.0, 0.21, 0.42)]

    def _sound(self, body: PhysicsBody) -> list[float]:
        result = np.zeros(3, dtype=float)
        point = np.array([body.x, body.y, body.z])
        listener_body = self._body_mj[body.id]
        for signal in self.signals:
            source = np.array([signal.x, signal.y, signal.z])
            distance = float(np.linalg.norm(point - source))
            envelope = min(1.0, signal.remaining / 0.3)
            visibility = 1.0 if self._acoustics is None else self.acoustic_visibility(
                point, source, None, listener_body, 0.16,
            )
            result[signal.tone] += visibility * signal.strength * envelope / (1.0 + (distance / 1.4) ** 2)
        for entity in self._entities:
            resonator = next((c for c in self._components[entity["id"]] if c.get("type") == "resonator"), None)
            if resonator is None or self._resonance.get(entity["id"], 0.0) <= 0.0:
                continue
            if self._acoustics is not None and self._acoustics.handles(entity["id"]):
                continue
            position, _ = self._pose(entity["id"])
            distance = float(np.linalg.norm(point - position))
            visibility = 1.0 if self._acoustics is None else self.acoustic_visibility(
                point, position, entity["id"], listener_body, 0.16,
            )
            result[int(resonator["tone"])] += visibility * self._resonance[entity["id"]] * float(resonator.get("gain", 1.0)) / (1.0 + (distance / 1.4) ** 2)
        if self._acoustics is not None:
            sample = np.asarray(self._acoustics.sample(point, listener_body), dtype=float)
            if sample.shape != (3,) or not np.isfinite(sample).all() or np.any(sample < 0.0):
                raise RuntimeError("acoustic engine returned an invalid sample")
            result += sample
        return np.clip(result, 0.0, 2.0).tolist()

    def _scene_lights(self) -> list[dict[str, Any]]:
        """Resolve declarative body-local lights into current world coordinates."""
        result: list[dict[str, Any]] = []
        for entity in self._entities:
            body_id = self._entity_mj[entity["id"]]
            rotation = self.data.xmat[body_id].reshape(3, 3)
            origin = self.data.xpos[body_id]
            for index, component in enumerate(self._components[entity["id"]]):
                if component.get("type") != "light":
                    continue
                local_position = np.asarray(component.get("position", [0, 0, 0]), dtype=float)
                local_direction = np.asarray(component.get("direction", [0, 0, -1]), dtype=float)
                direction = rotation @ (local_direction / np.linalg.norm(local_direction))
                result.append({
                    "id": f"{entity['id']}:{index}", "entity": entity["id"],
                    "position": (origin + rotation @ local_position).astype(float).tolist(),
                    "direction": direction.astype(float).tolist(),
                    "color": list(map(float, component.get("color", [1, 1, 1]))),
                    "intensity": float(component.get("intensity", 1.0)),
                    "radius": float(component.get("radius", 2.0)),
                })
        return result

    def _environment_at(self, point: np.ndarray, exclude_body: int = -1) -> dict[str, float]:
        upward_distance, upward_geom = self._ray(point + np.array([0.0, 0.0, 1e-4]), np.array([0.0, 0.0, 1.0]), exclude_body)
        sky_exposure = 1.0 if upward_geom < 0 or upward_distance < 0.0 else 0.08
        value = 0.42 * sky_exposure
        for light in self._scene_lights():
            source = np.asarray(light["position"], dtype=float)
            delta = point - source
            distance = float(np.linalg.norm(delta))
            if distance < 1e-8:
                visibility = cone = 1.0
            else:
                ray_distance, geom_id = self._ray(point, -delta / distance, exclude_body)
                hit_entity = self._geom_entity.get(geom_id)
                visibility = 1.0 if ray_distance < 0.0 or ray_distance >= distance - 0.07 or hit_entity == light["entity"] else 0.10
                cone = max(0.0, float(np.dot(delta / distance, np.asarray(light["direction"])))) ** 0.35
            value += visibility * cone * float(light["intensity"]) / (1.0 + (distance / float(light["radius"])) ** 2)
        if self._light["remaining"] > 0.0:
            distance = float(np.linalg.norm(point - np.asarray(self._light["position"])))
            value += self._light["intensity"] / (1.0 + (distance / 1.8) ** 2)
        return {"illumination": float(min(1.0, value)), "sky_exposure": sky_exposure}

    def sample_environment(self, points: Any) -> list[dict[str, float]]:
        """Sample anonymous physical light conditions for ecology integrations."""
        if not isinstance(points, (list, tuple)) or len(points) > 128:
            raise ValueError("points must be a sequence of at most 128 positions")
        clean = [_vector(point, 3, "environment sample point", max(self.width, self.height, self.depth) * 2) for point in points]
        if any(not (0.0 <= p[0] <= self.width and 0.0 <= p[1] <= self.height and 0.0 <= p[2] <= self.depth) for p in clean):
            raise ValueError("environment sample point is outside habitat")
        return [self._environment_at(np.asarray(point, dtype=float)) for point in clean]

    def _illumination(self, body: PhysicsBody) -> float:
        point = np.array([body.x, body.y, body.z + 0.035], dtype=float)
        return self._environment_at(point, self._body_mj[body.id])["illumination"]

    def _shade(self, body: PhysicsBody) -> float:
        origin = np.array([body.x, body.y, body.z + 0.06])
        distance, geom_id = self._ray(origin, np.array([0.0, 0.0, 1.0]), self._body_mj[body.id])
        shade = 0.0 if geom_id < 0 or distance < 0 or distance > 3.0 else max(0.0, 1.0 - distance / 3.0)
        point = np.array([body.x, body.y, body.z])
        for entity in self._entities:
            component = next((c for c in self._components[entity["id"]] if c.get("type") == "shade"), None)
            if component:
                position, _ = self._pose(entity["id"])
                distance_xy = float(np.linalg.norm((point - position)[:2]))
                radius = float(component.get("radius", 1.0))
                shade = max(shade, float(component.get("strength", 1.0)) * max(0.0, 1.0 - distance_xy / radius))
        return float(np.clip(shade * (1.15 - 0.35 * self._illumination(body)), 0.0, 1.0))

    def _validate_actions(self, actions: Any, dt: Any) -> tuple[dict[str, dict[str, Any]], float]:
        if not isinstance(actions, dict):
            raise ValueError("actions must be a mapping")
        step = _number(dt, "dt", 0.0001, 0.2)
        ids = {body.id for body in self.bodies}
        if set(actions) - ids:
            raise ValueError("action refers to an unknown body")
        limits = {
            "forward": (-1.0, 1.0), "thrust": (-1.0, 1.0),
            "turn": (-1.0, 1.0), "yaw": (-1.0, 1.0), "eat": (0.0, 1.0),
            "grip": (0.0, 1.0), "gaze_pitch": (-1.0, 1.0),
            "lift": (-1.0, 1.0), "vertical": (-1.0, 1.0), "posture": (-1.0, 1.0),
            "signal_low": (0.0, 1.0), "signal_mid": (0.0, 1.0), "signal_high": (0.0, 1.0),
        }
        clean: dict[str, dict[str, Any]] = {}
        for body_id, action in actions.items():
            if not isinstance(action, dict) or set(action) - (set(limits) | {"signal"}):
                raise ValueError("invalid action schema")
            clean[body_id] = {}
            for name, value in action.items():
                if name == "signal":
                    if isinstance(value, (list, tuple)):
                        clean[body_id][name] = [
                            _number(v, f"signal[{index}]", 0.0, 1.0)
                            for index, v in enumerate(_vector(value, 3, "signal", 1.0))
                        ]
                    else:
                        clean[body_id][name] = _number(value, name, 0.0, 1.0)
                    continue
                low, high = limits[name]
                clean[body_id][name] = _number(value, name, low, high)
            if "lift" in action and "vertical" in action:
                raise ValueError("use either lift or vertical, not both")
            if "forward" in action and "thrust" in action:
                raise ValueError("use either forward or thrust, not both")
            if "turn" in action and "yaw" in action:
                raise ValueError("use either turn or yaw, not both")
            if "signal" in action and set(action) & {"signal_low", "signal_mid", "signal_high"}:
                raise ValueError("use either signal vector or named signal channels")
        return clean, step

    def advance(self, actions: dict[str, dict[str, Any]], dt: float = MODEL_DT) -> dict[str, dict[str, float]]:
        clean, step = self._validate_actions(actions, dt)
        starts = {body.id: np.array([body.x, body.y, body.z]) for body in self.bodies}
        outcomes = {
            body.id: {"nutrition": 0.0, "contact": 0.0, "distance": 0.0, "effort": 0.0}
            for body in self.bodies
        }
        self._touch = {body.id: [0.0, 0.0] for body in self.bodies}
        self._contact_normals = {body.id: [] for body in self.bodies}
        contacted_entities = {body.id: set() for body in self.bodies}
        self.signals = [signal for signal in self.signals if self._age_signal(signal, step)]
        for body in self.bodies:
            self._signal_cooldown[body.id] = max(0.0, self._signal_cooldown[body.id] - step)
            action = clean.get(body.id, {})
            body.gaze_pitch = action.get("gaze_pitch", body.gaze_pitch)
            self._update_grip(body.id, action.get("grip", 0.0))
            tones = self._action_tones(action, self.bodies.index(body) % 3)
            if tones and self._signal_cooldown[body.id] <= 1e-12:
                for tone, strength in tones:
                    if len(self.signals) >= self._signal_limit:
                        break
                    self._emit_signal(body.x, body.y, body.z + 0.08, tone, strength)
                self._signal_cooldown[body.id] = 0.5
        self._light["remaining"] = max(0.0, self._light["remaining"] - step)
        self._sync_light_model()
        for entity_id in self._resonance:
            self._resonance[entity_id] *= math.exp(-step / 0.7)

        motor_noise = {body.id: self.rng.normal(0.0, 1.0, 2) for body in self.bodies}
        substeps = max(1, int(math.ceil(step / float(self.spec["physics_timestep"]))))
        previous_timestep = float(self.model.opt.timestep)
        self.model.opt.timestep = step / substeps
        try:
            for _ in range(substeps):
                self.data.xfrc_applied[:] = 0.0
                self.data.qfrc_applied[:] = 0.0
                for body in self.bodies:
                    self._apply_crawler_forces(body, clean.get(body.id, {}), motor_noise[body.id])
                self._apply_hand_force()
                self._apply_grip_forces()
                if self._acoustics is not None:
                    self._acoustics.before_substep(self.model.opt.timestep)
                mujoco.mj_step(self.model, self.data)
                self._collect_contacts(contacted_entities)
                self._excite_hinges()
        finally:
            self.model.opt.timestep = previous_timestep
        # mj_step advances qpos after computing the kinematic fields used by
        # rendering and sensors. Refresh them so the public pose and checkpoint
        # describe the same instant as the integration state.
        mujoco.mj_forward(self.model, self.data)
        self._sync_public_state()

        for body in self.bodies:
            action = clean.get(body.id, {})
            for entity_id in contacted_entities[body.id]:
                food = next((c for c in self._components[entity_id] if c.get("type") == "food"), None)
                if food is None or action.get("eat", 0.0) <= 0.0 or float(food["amount"]) <= 0.0:
                    continue
                bite = min(float(food["amount"]), 0.34 * action["eat"] * step, 1.0 - body.gut)
                food["amount"] = float(food["amount"]) - bite
                body.gut += bite
                outcomes[body.id]["nutrition"] += bite * float(food.get("nutrition", 1.0))
            digestion = min(body.gut, 0.032 * step, max(0.0, (1.0 - body.energy) / 0.84))
            body.gut -= digestion
            body.energy += digestion * 0.84
            thrust = abs(action.get("forward", action.get("thrust", 0.0)))
            turn = abs(action.get("turn", action.get("yaw", 0.0)))
            vertical = abs(action.get("lift", action.get("vertical", action.get("posture", 0.0))))
            effort = float(np.clip(
                0.45 * thrust + 0.18 * turn + 0.22 * vertical + 0.15 * action.get("grip", 0.0)
                + 0.25 * min(1.0, abs(body.linear_velocity[2]) / 1.2), 0.0, 1.0
            ))
            body.energy = float(np.clip(body.energy - step * (0.0007 + 0.0042 * effort), 0.0, 1.0))
            body.fatigue = float(np.clip(body.fatigue + step * (0.07 * effort - 0.026 * (1.0 - min(1.0, effort))), 0.0, 1.0))
            body.gut = float(np.clip(body.gut, 0.0, 1.0))
            body.age += step
            outcomes[body.id]["contact"] = float(max(self._touch[body.id]))
            outcomes[body.id]["distance"] = float(np.linalg.norm(np.array([body.x, body.y, body.z]) - starts[body.id]))
            outcomes[body.id]["effort"] = effort
        self.time = float(self.data.time)
        self._sync_public_state()
        return outcomes

    @staticmethod
    def _action_tones(action: dict[str, Any], default_tone: int) -> list[tuple[int, float]]:
        signal = action.get("signal")
        if isinstance(signal, list):
            return [(tone, float(strength)) for tone, strength in enumerate(signal) if strength > 0.0]
        if isinstance(signal, (int, float)) and signal > 0.0:
            return [(default_tone, float(signal))]
        names = ("signal_low", "signal_mid", "signal_high")
        return [(tone, float(action.get(name, 0.0))) for tone, name in enumerate(names) if action.get(name, 0.0) > 0.0]

    @property
    def _signal_limit(self) -> int:
        return int(self.spec.get("limits", {}).get("signals", 256))

    def _age_signal(self, signal: PhysicsSignal, step: float) -> bool:
        signal.remaining -= step
        return signal.remaining > 0.0

    def _emit_signal(self, x: float, y: float, z: float, tone: int, strength: float) -> PhysicsSignal:
        signal = PhysicsSignal(f"signal3-{self._next_signal_id}", x, y, z, tone, strength)
        self._next_signal_id += 1
        self.signals.append(signal)
        return signal

    def _has_support_contact(self, resident_id: str) -> bool:
        resident_geoms = {geom for geom, owner in self._geom_resident.items() if owner == resident_id}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if int(contact.geom1) in resident_geoms or int(contact.geom2) in resident_geoms:
                return True
        return False

    def _apply_crawler_forces(self, body: PhysicsBody, action: dict[str, Any], noise: np.ndarray) -> None:
        mj_body = self._body_mj[body.id]
        rotation = self.data.xmat[mj_body].reshape(3, 3)
        linear, angular = self._velocity(mj_body)
        prototype = self.spec["body_prototype"]
        fatigue = 1.0 - 0.72 * body.fatigue
        vitality = 0.18 + 0.82 * body.energy
        supported = self._has_support_contact(body.id)
        if supported:
            forward = action.get("forward", action.get("thrust", 0.0))
            force = float(prototype["traction_force"]) * forward * fatigue * vitality * (1.0 + 0.015 * noise[0])
            heading = rotation[:, 0].copy()
            heading[2] = 0.0
            heading /= max(float(np.linalg.norm(heading)), 1e-9)
            self.data.xfrc_applied[mj_body, :3] += heading * force - linear * 1.35
            turn = action.get("turn", action.get("yaw", 0.0))
            self.data.xfrc_applied[mj_body, 5] += float(prototype["turn_torque"]) * turn * fatigue + 0.0004 * noise[1] * abs(turn)
            # The learned eight-axis adapter calls this coordinate ``posture``;
            # direct physics clients may name the same vertical traction ``lift``.
            lift = action.get("lift", action.get("vertical", action.get("posture", 0.0)))
            self.data.xfrc_applied[mj_body, 2] += float(prototype["vertical_force"]) * lift
        z_axis = rotation[:, 2]
        posture = action.get("posture")
        posture_gain = 1.0 if posture is None else 0.5 + 0.5 * posture
        stabilizing = np.cross(z_axis, np.array([0.0, 0.0, 1.0])) * float(prototype["posture_torque"]) * posture_gain
        self.data.xfrc_applied[mj_body, 3:6] += stabilizing - angular * 0.012

    def _update_grip(self, body_id: str, grip: float) -> None:
        if grip <= 0.1:
            self._grips[body_id] = None
            return
        if self._grips[body_id] is not None:
            return
        resident_geoms = {geom for geom, owner in self._geom_resident.items() if owner == body_id}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            other = second if first in resident_geoms else first if second in resident_geoms else -1
            entity_id = self._geom_entity.get(other)
            if entity_id and self._entity(entity_id)["mobility"] == "free":
                self._grips[body_id] = entity_id
                return

    def _apply_grip_forces(self) -> None:
        for body in self.bodies:
            entity_id = self._grips.get(body.id)
            if not entity_id or entity_id not in self._entity_mj:
                continue
            object_body = self._entity_mj[entity_id]
            creature_body = self._body_mj[body.id]
            rotation = self.data.xmat[creature_body].reshape(3, 3)
            target = self.data.xpos[creature_body] + rotation[:, 0] * 0.17 + np.array([0.0, 0.0, 0.04])
            position = self.data.xpos[object_body]
            velocity, _ = self._velocity(object_body)
            body_velocity, _ = self._velocity(creature_body)
            force = (target - position) * 15.0 - (velocity - body_velocity) * 1.1
            norm = float(np.linalg.norm(force))
            if norm > 8.0:
                force *= 8.0 / norm
            self.data.xfrc_applied[object_body, :3] += force
            self.data.xfrc_applied[creature_body, :3] -= force

    def _apply_hand_force(self) -> None:
        if not self._hand:
            return
        entity_id = self._hand["entity_id"]
        if entity_id not in self._entity_mj or self._entity(entity_id)["mobility"] != "free":
            self._hand = None
            return
        mj_body = self._entity_mj[entity_id]
        position = self.data.xpos[mj_body]
        velocity, _ = self._velocity(mj_body)
        force = (np.asarray(self._hand["target"]) - position) * self._hand["stiffness"] - velocity * self._hand["damping"]
        # The hand supports the object's weight, then the bounded spring supplies
        # the force that moves it toward the cursor. Heavy objects can still
        # exceed the cap and remain draggable rather than becoming weightless.
        force -= float(self.model.body_mass[mj_body]) * np.asarray(self.model.opt.gravity)
        limit = float(self.spec["limits"]["hand_force"])
        norm = float(np.linalg.norm(force))
        if norm > limit:
            force *= limit / norm
        self.data.xfrc_applied[mj_body, :3] += force

    def _collect_contacts(self, contacted_entities: dict[str, set[str]]) -> None:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            world_normal = np.asarray(contact.frame[:3], dtype=float)
            if self._acoustics is not None:
                force = np.zeros(6, dtype=float)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                point = np.asarray(contact.pos, dtype=float)
                velocities = []
                for geom_id in (first, second):
                    body_id = int(self.model.geom_bodyid[geom_id])
                    linear, angular = self._velocity(body_id)
                    velocities.append(linear + np.cross(angular, point - self.data.xpos[body_id]))
                relative_speed = abs(float(np.dot(velocities[1] - velocities[0], world_normal)))
                impulse = min(
                    float(self.spec.get("limits", {}).get("acoustic_impulse", 10.0)),
                    abs(float(force[0])) * float(self.model.opt.timestep),
                )
                impact_work = min(
                    float(self.spec.get("limits", {}).get("acoustic_work", 5.0)),
                    0.5 * impulse * relative_speed,
                )
                for entity_id in {
                    value for value in (self._geom_entity.get(first), self._geom_entity.get(second))
                    if value is not None
                }:
                    self._acoustics.ingest_contact({
                        "entity": entity_id, "position": point.astype(float).tolist(),
                        "normal_impulse": impulse, "relative_normal_speed": relative_speed,
                        "impact_work": impact_work,
                    })
            participants: list[tuple[str, int, np.ndarray]] = []
            if first in self._geom_resident:
                participants.append((self._geom_resident[first], second, world_normal))
            if second in self._geom_resident:
                participants.append((self._geom_resident[second], first, -world_normal))
            for resident_id, other, normal in participants:
                entity_id = self._geom_entity.get(other)
                if entity_id:
                    contacted_entities[resident_id].add(entity_id)
                body = self._body(resident_id)
                mj_body = self._body_mj[resident_id]
                point = np.asarray(contact.pos, dtype=float)
                # Ground-like support is used for traction but excluded from the
                # bilateral obstacle touch channel.
                if abs(float(normal[2])) > 0.72 and point[2] < body.z:
                    continue
                force = np.zeros(6, dtype=float)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                strength = min(1.0, 0.18 + float(np.linalg.norm(force[:3])) / 3.0)
                rotation = self.data.xmat[mj_body].reshape(3, 3)
                delta = point - self.data.xpos[mj_body]
                side = 1 if float(np.dot(delta, rotation[:, 1])) >= 0 else 0
                self._touch[resident_id][side] = max(self._touch[resident_id][side], strength)
                if len(self._contact_normals[resident_id]) < 8:
                    self._contact_normals[resident_id].append((rotation.T @ normal).astype(float).tolist())
                if entity_id:
                    self._resonance[entity_id] = max(self._resonance.get(entity_id, 0.0), strength)

    def _excite_hinges(self) -> None:
        for entity_id, joint_id in self._entity_joint.items():
            if self._entity(entity_id)["mobility"] != "hinge":
                continue
            velocity = abs(float(self.data.qvel[self.model.jnt_dofadr[joint_id]]))
            self._resonance[entity_id] = max(self._resonance.get(entity_id, 0.0), min(1.0, velocity * 0.22))

    def command(self, command: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(command, dict) or not isinstance(command.get("op"), str):
            raise ValueError("command requires string op")
        op = command["op"]
        if op == "add":
            self._reject_extra(command, {"op", "preset", "x", "y", "z", "id"})
            preset = command.get("preset")
            if not isinstance(preset, str) or preset not in self.spec.get("presets", {}):
                raise ValueError("add requires a declared preset")
            limit = int(self.spec["limits"]["entities"])
            if len(self.spec["entities"]) >= limit:
                raise ValueError("entity capacity exceeded")
            position = self._command_position(command)
            entity_id = command.get("id", f"{preset}-user-{self._next_entity_id}")
            if not isinstance(entity_id, str) or not _ID.match(entity_id) or entity_id in self._entity_mj or any(b.id == entity_id for b in self.bodies):
                raise ValueError("invalid or duplicate entity id")
            new_entity = {"id": entity_id, "preset": preset, "position": position}
            self.spec["entities"].append(new_entity)
            try:
                self._rebuild_preserving()
            except Exception:
                self.spec["entities"].pop()
                raise
            self._next_entity_id += 1
            return next(obj.to_dict() for obj in self.objects if obj.id == entity_id)
        if op in ("move", "hand"):
            self._reject_extra(command, {"op", "id", "x", "y", "z", "stiffness", "damping"})
            entity_id = command.get("id")
            if not isinstance(entity_id, str) or entity_id not in self._entity_mj or self._entity(entity_id)["mobility"] != "free":
                raise ValueError("hand target must be a free physical entity")
            target = self._command_position(command)
            stiffness = _number(command.get("stiffness", 18.0), "stiffness", 0.1, 80.0)
            damping = _number(command.get("damping", 2.5), "damping", 0.0, 20.0)
            self._hand = {"entity_id": entity_id, "target": target, "stiffness": stiffness, "damping": damping}
            self._sync_hand_mocap()
            mujoco.mj_forward(self.model, self.data)
            return copy.deepcopy(self._hand)
        if op == "release":
            self._reject_extra(command, {"op"})
            previous = copy.deepcopy(self._hand)
            self._hand = None
            self._sync_hand_mocap()
            mujoco.mj_forward(self.model, self.data)
            return {"released": None if previous is None else previous["entity_id"]}
        if op == "impulse":
            self._reject_extra(command, {"op", "id", "impulse"})
            entity_id = command.get("id")
            if not isinstance(entity_id, str) or entity_id not in self._entity_joint:
                raise ValueError("impulse target must be dynamic")
            impulse = np.asarray(_vector(command.get("impulse"), 3, "impulse", 20.0), dtype=float)
            limit = float(self.spec["limits"]["impulse"])
            if float(np.linalg.norm(impulse)) > limit:
                raise ValueError("impulse exceeds physical bound")
            entity = self._entity(entity_id)
            joint_id = self._entity_joint[entity_id]
            dof = int(self.model.jnt_dofadr[joint_id])
            if entity["mobility"] == "free":
                mass = float(self.model.body_mass[self._entity_mj[entity_id]])
                self.data.qvel[dof : dof + 3] += impulse / max(mass, 1e-8)
            else:
                axis = np.asarray(entity.get("joint", {}).get("axis", [0, 1, 0]), dtype=float)
                inertia = max(float(self.model.dof_M0[dof]), 1e-6)
                self.data.qvel[dof] += float(np.dot(impulse, axis)) / inertia
            mujoco.mj_forward(self.model, self.data)
            self._sync_public_state()
            return {"id": entity_id, "impulse": impulse.tolist()}
        if op == "signal":
            self._reject_extra(command, {"op", "x", "y", "z", "tone", "strength"})
            if len(self.signals) >= self._signal_limit:
                raise ValueError("signal capacity exceeded")
            position = self._command_position(command)
            tone = command.get("tone", 0)
            if isinstance(tone, bool) or not isinstance(tone, int) or tone not in (0, 1, 2):
                raise ValueError("tone must be 0, 1, or 2")
            strength = _number(command.get("strength", 1.0), "strength", 0.001, 1.0)
            return self._emit_signal(*position, tone, strength).to_dict()
        if op == "light":
            self._reject_extra(command, {"op", "x", "y", "z", "intensity", "duration", "color"})
            position = self._command_position(command)
            intensity = _number(command.get("intensity", 1.0), "intensity", 0.0, 1.0)
            duration = _number(command.get("duration", 2.0), "duration", 0.01, 30.0)
            color = _vector(command.get("color", [1.0, 0.94, 0.78]), 3, "light color", 1.0)
            if any(value < 0 for value in color):
                raise ValueError("light color cannot be negative")
            self._light = {"position": position, "intensity": intensity, "remaining": duration, "color": color}
            self._sync_light_model()
            return copy.deepcopy(self._light)
        raise ValueError(f"unknown command: {op}")

    @staticmethod
    def _reject_extra(command: dict[str, Any], allowed: set[str]) -> None:
        if set(command) - allowed:
            raise ValueError("unknown command field")

    def _command_position(self, command: dict[str, Any]) -> list[float]:
        if not all(key in command for key in ("x", "y", "z")):
            raise ValueError("command requires x, y, and z")
        position = [_number(command[key], key) for key in ("x", "y", "z")]
        if not (0.0 <= position[0] <= self.width and 0.0 <= position[1] <= self.height and 0.0 <= position[2] <= self.depth):
            raise ValueError("position is outside habitat")
        return position

    def _sync_light_model(self) -> None:
        light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "caregiver-light")
        self.model.light_pos[light_id] = self._light["position"]
        value = self._light["intensity"] if self._light["remaining"] > 0 else 0.0
        self.model.light_diffuse[light_id] = np.asarray(self._light["color"]) * value

    def _sync_hand_mocap(self) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "caregiver-hand")
        mocap_id = int(self.model.body_mocapid[body_id])
        self.data.mocap_pos[mocap_id] = self._hand["target"] if self._hand else [0.0, 0.0, -10.0]
        self.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]

    def _rebuild_preserving(self) -> None:
        joint_state: dict[str, tuple[list[float], list[float]]] = {}
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_type = int(self.model.jnt_type[joint_id])
            qn = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            vn = 6 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            qa, da = int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])
            joint_state[name] = (self.data.qpos[qa : qa + qn].tolist(), self.data.qvel[da : da + vn].tolist())
        old_time = float(self.data.time)
        old_bodies = {body.id: body.to_dict() for body in self.bodies}
        old_components = copy.deepcopy(self._components)
        old_resonance = self._resonance.copy()
        self._validate_spec(self.spec)
        self._compile_model()
        for name, (qpos, qvel) in joint_state.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                continue
            qa, da = int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qa : qa + len(qpos)] = qpos
            self.data.qvel[da : da + len(qvel)] = qvel
        self.data.time = old_time
        self.bodies = self._make_bodies()
        for body in self.bodies:
            if body.id in old_bodies:
                for key in ("energy", "gut", "fatigue", "age", "gaze_pitch"):
                    setattr(body, key, old_bodies[body.id][key])
        self._components = {entity["id"]: copy.deepcopy(entity.get("components", [])) for entity in self._entities}
        for entity_id in self._components.keys() & old_components.keys():
            self._components[entity_id] = old_components[entity_id]
        self._resonance = {entity["id"]: old_resonance.get(entity["id"], 0.0) for entity in self._entities}
        self._sync_light_model()
        self._sync_hand_mocap()
        mujoco.mj_forward(self.model, self.data)
        self._sync_public_state()

    def view(self) -> dict[str, Any]:
        self._sync_public_state()
        entities = []
        for entity, public in zip(self._entities, self.objects):
            shapes = []
            for index, shape in enumerate(entity["shapes"]):
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"entity:{entity['id']}:geom:{index}")
                geom_quat = np.empty(4, dtype=float)
                mujoco.mju_mat2Quat(geom_quat, self.data.geom_xmat[geom_id])
                size_count = 3 if shape["type"] in ("box", "ellipsoid") else 1 if shape["type"] == "sphere" else 2
                shapes.append({
                    "type": shape["type"], "size": self.model.geom_size[geom_id, :size_count].astype(float).tolist(),
                    "position": self.data.geom_xpos[geom_id].astype(float).tolist(),
                    "quaternion": geom_quat.tolist(),
                    "color": _hex([*self._geom_rgb(geom_id), 1.0]),
                })
            value = public.to_dict()
            value.update({"mobility": entity["mobility"], "material": entity["material"],
                          "physical_material": entity["physical_material"], "shapes": shapes})
            if entity["id"] in self._entity_joint:
                joint_id = self._entity_joint[entity["id"]]
                qadr, dadr = int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])
                if entity["mobility"] == "hinge":
                    value["joint"] = {"position": float(self.data.qpos[qadr]), "velocity": float(self.data.qvel[dadr])}
            entities.append(value)
        body_views = []
        prototype_shape = self.spec["body_prototype"]["shape"]
        for body, config in zip(self.bodies, self.spec["bodies"]):
            value = body.to_dict()
            value.update({"shape": prototype_shape["type"], "size": list(map(float, prototype_shape["size"])),
                          "material": config["material"]})
            body_views.append(value)
        return {
            "dimension": 3, "width": self.width, "height": self.height, "depth": self.depth,
            "time": self.time, "bodies": body_views,
            "objects": [obj.to_dict() for obj in self.objects], "entities": entities,
            "signals": [signal.to_dict() for signal in self.signals], "hand": copy.deepcopy(self._hand),
            "light": copy.deepcopy(self._light), "lights": self._scene_lights(),
            "engine": {"name": "MuJoCo", "version": mujoco.__version__},
        }

    def snapshot(self) -> dict[str, Any]:
        state = np.empty(mujoco.mj_stateSize(self.model, STATE_SPEC), dtype=float)
        mujoco.mj_getState(self.model, self.data, state, STATE_SPEC)
        return {
            "version": 1, "dimension": 3, "engine": {"name": "MuJoCo", "version": mujoco.__version__},
            "model_signature": self._model_signature, "seed": self.seed, "spec": copy.deepcopy(self.spec),
            "mj_state_spec": int(STATE_SPEC), "mj_state": state.tolist(),
            "model_mutable": {name: getattr(self.model, name).tolist() for name in _MUTABLE_MODEL_FIELDS},
            "bodies": [body.to_dict() for body in self.bodies], "components": copy.deepcopy(self._components),
            "resonance": self._resonance.copy(), "signals": [signal.to_dict() for signal in self.signals],
            "touch": copy.deepcopy(self._touch), "contact_normals": copy.deepcopy(self._contact_normals),
            "signal_cooldown": self._signal_cooldown.copy(), "grips": self._grips.copy(),
            "hand": copy.deepcopy(self._hand), "light": copy.deepcopy(self._light),
            "next_entity_id": self._next_entity_id, "next_signal_id": self._next_signal_id,
            "rng_state": _json_value(copy.deepcopy(self.rng.bit_generator.state)),
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> "PhysicsWorld":
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1 or snapshot.get("dimension") != 3:
            raise ValueError("unsupported 3D world snapshot")
        engine = snapshot.get("engine", {})
        if engine != {"name": "MuJoCo", "version": mujoco.__version__}:
            raise ValueError("snapshot requires a different MuJoCo engine version")
        raw_spec = snapshot.get("spec")
        if not isinstance(raw_spec, dict):
            raise ValueError("snapshot habitat specification is invalid")
        restore_spec = copy.deepcopy(raw_spec)
        # Early v1 checkpoints were written through a canonical JSON encoder
        # that sorted mapping keys, while the original compiler emitted visual
        # materials in declaration order. Recover that order from the exact
        # saved MuJoCo material table. New scenes carry material_order explicitly.
        compiler = restore_spec.get("compiler", {})
        if isinstance(compiler, dict) and "material_order" not in compiler:
            mutable = snapshot.get("model_mutable", {})
            try:
                saved_rgba = np.asarray(mutable.get("mat_rgba", []) if isinstance(mutable, dict) else [], dtype=float)
            except (TypeError, ValueError):
                saved_rgba = np.empty((0, 4), dtype=float)
            materials = restore_spec.get("materials", {})
            valid_materials = isinstance(materials, dict) and all(
                isinstance(value, dict) and isinstance(value.get("rgba"), list)
                for value in materials.values()
            )
            if valid_materials and saved_rgba.shape == (len(materials), 4):
                unmatched = set(materials)
                recovered: list[str] = []
                for row in saved_rgba:
                    matches = [
                        name for name in unmatched
                        if np.allclose(row, np.asarray(materials[name].get("rgba"), dtype=float), rtol=0.0, atol=1e-6)
                    ]
                    if len(matches) != 1:
                        recovered = []
                        break
                    recovered.append(matches[0])
                    unmatched.remove(matches[0])
                if recovered and not unmatched:
                    restore_spec["materials"] = {name: materials[name] for name in recovered}
        world = cls(seed=snapshot["seed"], spec=restore_spec)
        if snapshot.get("model_signature") != world._model_signature or snapshot.get("mj_state_spec") != int(STATE_SPEC):
            raise ValueError("snapshot model or integration state contract differs")
        state = np.asarray(snapshot.get("mj_state"), dtype=float)
        expected = mujoco.mj_stateSize(world.model, STATE_SPEC)
        if state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError("invalid MuJoCo integration state")
        body_data = snapshot.get("bodies")
        if not isinstance(body_data, list) or {b.get("id") for b in body_data if isinstance(b, dict)} != {b.id for b in world.bodies}:
            raise ValueError("snapshot body identities differ")
        try:
            world.bodies = [PhysicsBody(**item) for item in body_data]
            world._components = copy.deepcopy(snapshot["components"])
            world._resonance = {str(k): _number(v, "resonance", 0.0, 2.0) for k, v in snapshot["resonance"].items()}
            world.signals = [PhysicsSignal(**item) for item in snapshot["signals"]]
            world._touch = copy.deepcopy(snapshot["touch"])
            world._contact_normals = copy.deepcopy(snapshot["contact_normals"])
            world._signal_cooldown = {str(k): _number(v, "signal cooldown", 0.0, 0.5) for k, v in snapshot["signal_cooldown"].items()}
            world._grips = copy.deepcopy(snapshot["grips"])
            world._hand = copy.deepcopy(snapshot["hand"])
            world._light = copy.deepcopy(snapshot["light"])
            world._next_entity_id = int(snapshot["next_entity_id"])
            world._next_signal_id = int(snapshot["next_signal_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid 3D component state") from exc
        ids = {body.id for body in world.bodies}
        entity_ids = {entity["id"] for entity in world._entities}
        if set(world._touch) != ids or set(world._contact_normals) != ids or set(world._signal_cooldown) != ids or set(world._grips) != ids:
            raise ValueError("snapshot resident state identities differ")
        if set(world._components) != entity_ids or set(world._resonance) != entity_ids:
            raise ValueError("snapshot entity component identities differ")
        for body in world.bodies:
            if any(not 0.0 <= _number(getattr(body, field), field) <= 1.0 for field in ("energy", "gut", "fatigue")):
                raise ValueError("snapshot physiology is outside [0, 1]")
            _number(body.age, "age", 0.0, 1e12)
            _number(body.gaze_pitch, "gaze pitch", -1.0, 1.0)
            touch = world._touch[body.id]
            normals = world._contact_normals.get(body.id)
            if not isinstance(touch, list) or len(touch) != 2 or any(not 0.0 <= _number(v, "touch") <= 1.0 for v in touch):
                raise ValueError("invalid tactile snapshot state")
            if not isinstance(normals, list) or len(normals) > 8:
                raise ValueError("invalid contact-normal snapshot state")
            for normal in normals:
                _vector(normal, 3, "contact normal", 2.0)
            grip = world._grips[body.id]
            if grip is not None and (grip not in entity_ids or world._entity(grip)["mobility"] != "free"):
                raise ValueError("invalid grip target")
        for entity in world._entities:
            components = world._components[entity["id"]]
            expected_types = [value["type"] for value in entity.get("components", [])]
            if not isinstance(components, list) or not all(isinstance(value, dict) for value in components) or [value.get("type") for value in components] != expected_types:
                raise ValueError("snapshot changed attached component topology")
            for component in components:
                if component["type"] == "food":
                    _number(component.get("amount"), "food amount", 0.0, 100.0)
        if len(world.signals) > world._signal_limit or len({signal.id for signal in world.signals}) != len(world.signals):
            raise ValueError("invalid signal collection")
        for signal in world.signals:
            _vector([signal.x, signal.y, signal.z], 3, "signal position")
            if isinstance(signal.tone, bool) or signal.tone not in (0, 1, 2):
                raise ValueError("invalid signal tone")
            _number(signal.strength, "signal strength", 0.0, 1.0)
            _number(signal.remaining, "signal lifetime", 0.0, 1.25)
        if world._hand is not None:
            if world._hand.get("entity_id") not in entity_ids or world._entity(world._hand["entity_id"])["mobility"] != "free":
                raise ValueError("invalid hand target")
            _vector(world._hand.get("target"), 3, "hand target")
            _number(world._hand.get("stiffness"), "hand stiffness", 0.1, 80.0)
            _number(world._hand.get("damping"), "hand damping", 0.0, 20.0)
        light_position = _vector(world._light.get("position"), 3, "light position")
        if not (0.0 <= light_position[0] <= world.width and 0.0 <= light_position[1] <= world.height and 0.0 <= light_position[2] <= world.depth):
            raise ValueError("light is outside habitat")
        _number(world._light.get("intensity"), "light intensity", 0.0, 1.0)
        _number(world._light.get("remaining"), "light lifetime", 0.0, 30.0)
        _vector(world._light.get("color"), 3, "light color", 1.0)
        if world._next_entity_id < 1 or world._next_signal_id < 1:
            raise ValueError("invalid snapshot counters")
        mutable = snapshot.get("model_mutable", {})
        for name in _MUTABLE_MODEL_FIELDS:
            target = getattr(world.model, name)
            value = np.asarray(mutable.get(name), dtype=float)
            if value.size != target.size or not np.isfinite(value).all():
                raise ValueError(f"invalid mutable model field: {name}")
            target[:] = value.reshape(target.shape)
        try:
            mujoco.mj_setState(world.model, world.data, state, STATE_SPEC)
            world.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid integration or RNG state") from exc
        mujoco.mj_forward(world.model, world.data)
        world.time = float(world.data.time)
        world._sync_public_state()
        return world


Body3D = PhysicsBody
Object3D = PhysicalObject
World3D = PhysicsWorld

__all__ = [
    "MODEL_DT", "STATE_SPEC", "PhysicsBody", "PhysicalObject", "PhysicsSignal",
    "PhysicsWorld", "Body3D", "Object3D", "World3D",
]
