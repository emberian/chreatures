"""Material-funded development of persistent physical habitat structures.

This owns environmental transduction and developmental transactions, not a
resident's action policy. The first colonies have finite founder endowments;
mobile feeding, reproduction and spatial mineral transport are separate joins.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .growth import GrowthSystem
from .metabolism import Chemistry, MetabolicWeb, canonical
from .native_world import load_world_kernels

FORMAT = "chreatures-biosphere-v8"
KINDS = ("branch", "root", "leaf")


@dataclass
class PreparedNewborn:
    """Private chemical owner aligned to one prepared physical candidate."""

    source_sha256: str
    proposal_sha256: str
    resident_id: str
    candidate: Biosphere
    funding: dict[str, Any]
    _committed: bool = False


def _illumination_cycle(value: Any, world: Any) -> dict[str, Any]:
    """Validate the one current native solar-cycle boundary."""
    required = {
        "version", "light_entity", "period_seconds", "phase_offset_cycles",
        "path_azimuth_degrees", "peak_irradiance", "diffuse_fraction",
        "twilight_degrees", "orbit_radius_m", "center_m", "color",
    }
    if not isinstance(value, dict) or set(value) != required or value["version"] != 1:
        raise ValueError("illumination_cycle requires the exact version 1 schema")
    entity_id = value["light_entity"]
    if not isinstance(entity_id, str):
        raise TypeError("solar light entity must be a physical identity")
    entity = world._entity(entity_id)
    if entity["mobility"] != "static" or entity.get("quaternion", [1, 0, 0, 0]) != [1, 0, 0, 0]:
        raise ValueError("solar light requires a static world-aligned entity frame")
    lights = [
        component for component in world._components[entity_id]
        if component.get("type") == "light"
    ]
    if len(lights) != 1 or lights[0].get("directional") is not True:
        raise ValueError("solar light entity requires one directional light")
    for key in (
        "period_seconds", "phase_offset_cycles", "path_azimuth_degrees",
        "peak_irradiance", "diffuse_fraction", "twilight_degrees",
        "orbit_radius_m",
    ):
        scalar = value[key]
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)) or not np.isfinite(scalar):
            raise ValueError(f"solar {key} must be a finite number")
    for key in ("center_m", "color"):
        vector = np.asarray(value[key], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"solar {key} must have three finite values")
    color = np.asarray(value["color"], dtype=float)
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise ValueError("solar color must be in [0, 1]")
    return copy.deepcopy(value)


def _light_sampling(value: Any) -> dict[str, Any]:
    """Validate one immutable physical radiance sampling profile."""
    if not isinstance(value, dict) or set(value) != {
        "frame",
        "directions",
        "weights",
        "occluded_transmission",
    }:
        raise ValueError("invalid developmental light sampling profile")
    if value["frame"] != "world-v1":
        raise ValueError("developmental light rays require the world-v1 frame")
    directions = value["directions"]
    weights = value["weights"]
    if (
        not isinstance(directions, list)
        or not 1 <= len(directions) <= 64
        or not isinstance(weights, list)
        or len(weights) != len(directions)
    ):
        raise ValueError("developmental light ray dimensions differ")
    for direction in directions:
        vector = np.asarray(direction, dtype=float)
        if (
            vector.shape != (3,)
            or not np.isfinite(vector).all()
            or vector[2] < 0.0
            or np.linalg.norm(vector) <= 1e-12
        ):
            raise ValueError("developmental light rays must occupy the upper hemisphere")
    weight = np.asarray(weights, dtype=float)
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("developmental light weights must be finite and positive")
    transmission = value["occluded_transmission"]
    if (
        not isinstance(transmission, (int, float))
        or not np.isfinite(transmission)
        or not 0.0 <= transmission <= 1.0
    ):
        raise ValueError("developmental light transmission must be in [0, 1]")
    return copy.deepcopy(value)


def _mobile_phototrophy(
    value: Any, world: Any, mobiles: Any,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Validate shared radiance law and inherited dorsal surface traits."""
    if (
        not isinstance(value, dict)
        or set(value) != {
            "version", "photon_flux_per_square_meter_second", "light_sampling",
        }
        or value["version"] != 1
    ):
        raise ValueError("mobile_phototrophy requires the exact version 1 schema")
    flux = value["photon_flux_per_square_meter_second"]
    if (
        isinstance(flux, bool)
        or not isinstance(flux, (int, float))
        or not np.isfinite(flux)
        or not 0.0 < flux <= 1.0e9
    ):
        raise ValueError("mobile photon flux density must be finite and positive")
    result = copy.deepcopy(value)
    result["light_sampling"] = _light_sampling(value["light_sampling"])
    if not isinstance(mobiles, list) or not mobiles:
        raise ValueError("mobile phototrophy requires resident physiology")
    mobile_ids = [spec.get("id") for spec in mobiles if isinstance(spec, dict)]
    body_by_id = {body["id"]: body for body in world.spec["bodies"]}
    if len(mobile_ids) != len(mobiles) or set(mobile_ids) != set(body_by_id):
        raise ValueError("mobile phototrophy and physical resident identities differ")
    traits = []
    for resident_id in mobile_ids:
        inherited = body_by_id[resident_id].get("metabolic_traits")
        if not isinstance(inherited, dict) or set(inherited) != {
            "phototrophic_absorptivity", "dorsal_capture_fraction",
        }:
            raise ValueError("resident metabolic traits differ from current schema")
        clean = {}
        for key in ("phototrophic_absorptivity", "dorsal_capture_fraction"):
            scalar = inherited[key]
            if (
                isinstance(scalar, bool)
                or not isinstance(scalar, (int, float))
                or not np.isfinite(scalar)
                or not 0.0 <= scalar <= 1.0
            ):
                raise ValueError("resident metabolic traits must be in [0, 1]")
            clean[key] = float(scalar)
        traits.append(clean)
    return result, traits


class Biosphere:
    """One chemical web and multiple resource-funded developmental colonies.

    Each colony has separate body and allocated-structure compartments. Built
    tissue remains in the latter until a physical removal transfers it. The
    placeholder attachment structures are an explicitly supplied scaffold;
    accounting here covers additional developed tissue, not that scaffold.
    """

    def __init__(
        self,
        world: Any,
        web: MetabolicWeb,
        colonies: list[dict[str, Any]],
        *,
        illumination_cycle: dict[str, Any],
        mobile_phototrophy: dict[str, Any],
        mobiles=None,
    ):
        self.world = world
        self.web = web
        self.config = copy.deepcopy(colonies)
        self.config_sha256 = hashlib.sha256(canonical(self.config)).hexdigest()
        self.illumination_config = _illumination_cycle(illumination_cycle, world)
        self.illumination_sha256 = hashlib.sha256(
            canonical(self.illumination_config)
        ).hexdigest()
        self.mobile_photo_config, self._mobile_traits = _mobile_phototrophy(
            mobile_phototrophy, world, mobiles,
        )
        self.mobile_photo_sha256 = hashlib.sha256(
            canonical({
                "config": self.mobile_photo_config,
                "traits": self._mobile_traits,
            })
        ).hexdigest()
        self._mobile_specs = copy.deepcopy(mobiles)
        self.growth: dict[str, GrowthSystem] = {}
        self.parts: dict[str, dict[str, Any]] = {}
        self.active: dict[str, bool] = {}
        self._light_specs: dict[str, dict[str, Any]] = {}
        used_rows: set[int] = set()
        used_entities: set[str] = set()
        for colony in self.config:
            required = {
                "id",
                "body_row",
                "structure_row",
                "grammar",
                "seed",
                "bindings",
                "seed_capture_area",
                "photon_flux",
                "mineral_half_saturation",
                "light_sampling",
            }
            if (
                not required <= set(colony)
                or set(colony) - required - {"genome_sha256"}
                or not isinstance(colony["id"], str)
            ):
                raise ValueError("invalid developmental colony specification")
            name = colony["id"]
            if not name or name in self.growth:
                raise ValueError("colony identities must be unique")
            for key in ("body_row", "structure_row"):
                row = colony[key]
                if (
                    isinstance(row, bool)
                    or not isinstance(row, int)
                    or not 0 <= row < web.count
                ):
                    raise ValueError("invalid colony compartment")
                if row in used_rows:
                    raise ValueError("colony compartments must be private")
                used_rows.add(row)
            bindings = colony["bindings"]
            if set(bindings) != set(KINDS) or len(set(bindings.values())) != 3:
                raise ValueError("colony needs distinct branch/root/leaf structures")
            poses = []
            for entity_id in bindings.values():
                if entity_id in used_entities:
                    raise ValueError("structure belongs to another colony")
                entity = world._entity(entity_id)
                if entity["mobility"] != "static":
                    raise ValueError(
                        "current developmental structures require static attachment"
                    )
                poses.append(
                    (entity["position"], entity.get("quaternion", [1, 0, 0, 0]))
                )
                used_entities.add(entity_id)
            if any(pose != poses[0] for pose in poses[1:]):
                raise ValueError("developmental structure frames must agree")
            for key in ("seed_capture_area", "photon_flux", "mineral_half_saturation"):
                value = colony[key]
                if (
                    not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("colony acquisition parameters must be positive")
            growth = GrowthSystem(colony["grammar"], colony["seed"])
            if set(growth.resource_names) - set(web.chemistry.pools):
                raise ValueError("growth requests chemistry absent from this web")
            self.growth[name] = growth
            self.active[name] = True
            self._light_specs[name] = _light_sampling(colony["light_sampling"])
            structure_activity = web.enzyme_activity[colony["structure_row"]]
            supported = {"soft_turnover", "tough_turnover"}
            if any(
                rate != 0 and reaction not in supported
                for reaction, rate in zip(
                    web.chemistry.reactions, structure_activity, strict=True
                )
            ):
                raise ValueError(
                    "allocated geometry currently supports only tissue turnover"
                )
        self.initial_totals = web.totals()
        self.initial_ledger = self._ledger().tolist()
        self.last_report: dict[str, Any] = {}
        self.mobility = None
        self.materials = None
        self.exchange = None
        self._tissue = load_world_kernels().BiosphereTissue(
            list(self.web.chemistry.pools),
            list(self.growth),
            [colony["structure_row"] for colony in self.config],
            self.web.chemistry._arrays[0],
        )
        profiles: list[dict[str, Any]] = []
        profile_index: dict[bytes, int] = {}
        self._light_profile_ids: dict[str, int] = {}
        for name, spec in self._light_specs.items():
            key = canonical(spec)
            if key not in profile_index:
                profile_index[key] = len(profiles)
                profiles.append(spec)
            self._light_profile_ids[name] = profile_index[key]
        mobile_profile = self.mobile_photo_config["light_sampling"]
        mobile_key = canonical(mobile_profile)
        if mobile_key not in profile_index:
            profile_index[mobile_key] = len(profiles)
            profiles.append(mobile_profile)
        self._mobile_light_profile = profile_index[mobile_key]
        directions: list[list[float]] = []
        weights: list[float] = []
        offsets = [0]
        transmission = []
        for profile in profiles:
            directions.extend(profile["directions"])
            weights.extend(profile["weights"])
            offsets.append(len(directions))
            transmission.append(profile["occluded_transmission"])
        self._environment = load_world_kernels().LightEnvironment(
            len(self.config), directions, weights, offsets, transmission
        )
        cycle = self.illumination_config
        self._solar = load_world_kernels().SolarCycle(
            cycle["period_seconds"], cycle["phase_offset_cycles"],
            cycle["path_azimuth_degrees"], cycle["peak_irradiance"],
            cycle["diffuse_fraction"], cycle["twilight_degrees"],
            cycle["orbit_radius_m"], cycle["center_m"],
        )
        self._solar_state: dict[str, Any] = {}
        self._solar_revision = -1
        self._solar_light_id = -1
        self._environment_revision = -1
        self._environment_lights: tuple[np.ndarray, ...] | None = None
        self._development_light: dict[str, list[tuple[dict[str, Any], float]]] = {}
        self._development_surface: dict[str, list[dict[str, Any]]] = {}
        self._growth_evidence: list[dict[str, Any]] = []
        self._sync_solar_state(self._solar.state())
        if mobiles is not None:
            from .somatic import SomaticPhysiology

            self.mobility = SomaticPhysiology(self, mobiles)
            world.bind_physiology(self.mobility)

    @classmethod
    def from_config(cls, world: Any, config: Any) -> Biosphere:
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        if not isinstance(config, dict):
            raise TypeError("biosphere birth configuration must be an object")
        config = copy.deepcopy(config)
        if config.get("format") != "chreatures-biosphere-birth-v6" or set(config) != {
            "format",
            "chemistry",
            "compartments",
            "bulk",
            "colonies",
            "mobiles",
            "material_objects",
            "exchange",
            "illumination_cycle",
            "mobile_phototrophy",
        }:
            raise ValueError("invalid biosphere birth configuration")
        compartments = config["compartments"]
        if not isinstance(compartments, list) or not compartments:
            raise ValueError("biosphere requires physical material compartments")
        if any(
            set(row) != {"enzymes", "pools", "atp", "atp_capacity"}
            for row in compartments
        ):
            raise ValueError("invalid compartment founder")
        chemistry = Chemistry(config["chemistry"])
        web = MetabolicWeb(
            chemistry,
            [row["enzymes"] for row in compartments],
            [row["pools"] for row in compartments],
            [row["atp"] for row in compartments],
            [row["atp_capacity"] for row in compartments],
            bulk=config["bulk"],
        )
        instance = cls(
            world, web, config["colonies"],
            illumination_cycle=config["illumination_cycle"],
            mobile_phototrophy=config["mobile_phototrophy"],
            mobiles=config["mobiles"],
        )
        if instance.mobility is not None:
            instance.mobility.sync_bodies()
        if config["material_objects"] is not None:
            from .material_objects import MaterialObjects

            instance.materials = MaterialObjects(
                world, instance, config["material_objects"]
            )
            instance._sync_material_cues()
        if config["exchange"] is not None:
            from .ecological_exchange import EcologicalExchange

            instance.exchange = EcologicalExchange(instance, config["exchange"])
        instance._bind_tissue()
        return instance

    def _ledger(self) -> np.ndarray:
        return np.asarray(self.web._native.cumulative_ledger).sum(axis=0)

    def _colony(self, colony_id: str) -> dict[str, Any]:
        return next(colony for colony in self.config if colony["id"] == colony_id)

    @staticmethod
    def _matrix(values: list[Any], columns: int, dtype=float) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape((-1, columns))

    def _sync_solar_state(self, raw: Any) -> None:
        """Apply one native sun state to the declarative and MuJoCo light."""
        if not isinstance(raw, tuple) or len(raw) != 6:
            raise RuntimeError("native solar cycle returned an invalid state")
        clock, position, toward_sun, irradiance, direct, diffuse = raw
        position = np.asarray(position, dtype=float)
        toward_sun = np.asarray(toward_sun, dtype=float)
        scalars = np.asarray([clock, irradiance, direct, diffuse], dtype=float)
        if (
            position.shape != (3,)
            or toward_sun.shape != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite(toward_sun).all()
            or not np.isfinite(scalars).all()
            or abs(float(np.linalg.norm(toward_sun)) - 1.0) > 1e-10
            or np.any(scalars < 0.0)
        ):
            raise RuntimeError("native solar cycle returned a nonphysical state")
        entity_id = self.illumination_config["light_entity"]
        components = self.world._components[entity_id]
        light_index, component = next(
            (index, component)
            for index, component in enumerate(components)
            if component.get("type") == "light"
        )
        if self._solar_revision != self.world.model_revision:
            self._solar_light_id = mujoco.mj_name2id(
                self.world.model, mujoco.mjtObj.mjOBJ_LIGHT,
                f"entity:{entity_id}:light:{light_index}",
            )
            if self._solar_light_id < 0:
                raise RuntimeError("physical solar light is absent from MuJoCo")
            self._solar_revision = self.world.model_revision
        body_id = self.world._entity_mj[entity_id]
        rotation = self.world.data.xmat[body_id].reshape(3, 3)
        origin = self.world.data.xpos[body_id]
        local_position = rotation.T @ (position - origin)
        local_direction = rotation.T @ -toward_sun
        component["position"] = local_position.astype(float).tolist()
        component["direction"] = local_direction.astype(float).tolist()
        component["intensity"] = float(direct)
        component["ambient_intensity"] = float(diffuse)
        light_id = self._solar_light_id
        color = np.asarray(self.illumination_config["color"], dtype=float)
        self.world.model.light_pos[light_id] = local_position
        self.world.model.light_dir[light_id] = local_direction
        self.world.model.light_diffuse[light_id] = color * direct
        self.world.data.light_xpos[light_id] = position
        self.world.data.light_xdir[light_id] = -toward_sun
        self._solar_state = {
            "clock_seconds": float(clock),
            "position_m": position.astype(float).tolist(),
            "toward_sun": toward_sun.astype(float).tolist(),
            "irradiance": float(irradiance),
            "direct_irradiance": float(direct),
            "diffuse_irradiance": float(diffuse),
        }

    def _bind_environment(self) -> None:
        """Cache attachment topology; MuJoCo poses and tissue remain live inputs."""
        bodies: list[int] = []
        local: list[list[float]] = []
        world_offset: list[list[float]] = []
        profiles: list[int] = []
        colonies: list[int] = []
        areas: list[float] = []
        part_ids: list[str | None] = []
        initial_tissue: list[float] = []
        for colony_index, colony in enumerate(self.config):
            profile = self._light_profile_ids[colony["id"]]
            bodies.append(self.world._entity_mj[colony["bindings"]["branch"]])
            local.append([0.0, 0.0, 0.025])
            world_offset.append([0.0, 0.0, 0.0])
            profiles.append(profile)
            colonies.append(colony_index)
            areas.append(float(colony["seed_capture_area"]))
            part_ids.append(None)
            initial_tissue.append(0.0)
            for part_id, part in self.parts.items():
                if part["colony"] != colony["id"] or part["kind"] != "leaf":
                    continue
                shape = part["shape"]
                original = float(part["initial_resources"].get("soft_tissue", 0.0))
                if original <= 0.0:
                    raise ValueError("photosynthetic leaf lacks initial soft tissue")
                bodies.append(self.world._entity_mj[part["entity"]])
                local.append(list(map(float, shape["position"])))
                world_offset.append([0.0, 0.0, max(map(float, shape["size"])) + 1e-4])
                profiles.append(profile)
                colonies.append(colony_index)
                areas.append(float(part["area"]))
                part_ids.append(part_id)
                initial_tissue.append(original)
        self._environment.bind_capture(
            self._tissue,
            np.asarray(bodies, dtype=np.int32),
            self._matrix(local, 3),
            self._matrix(world_offset, 3),
            np.asarray(profiles, dtype=np.int32),
            np.asarray(colonies, dtype=np.int32),
            np.asarray(areas, dtype=np.float64),
            part_ids,
            np.asarray(initial_tissue, dtype=np.float64),
        )
        mobile_bodies: list[int] = []
        mobile_local: list[list[float]] = []
        mobile_area: list[float] = []
        mobile_absorptivity: list[float] = []
        for spec, traits in zip(self._mobile_specs, self._mobile_traits, strict=True):
            geom_id = mujoco.mj_name2id(
                self.world.model, mujoco.mjtObj.mjOBJ_GEOM,
                f"resident:{spec['id']}:geom:thorax",
            )
            body_id = self.world._body_mj[spec["id"]]
            if (
                geom_id < 0
                or int(self.world.model.geom_type[geom_id])
                != int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
                or int(self.world.model.geom_bodyid[geom_id]) != body_id
            ):
                raise ValueError("mobile phototrophy requires a physical thorax ellipsoid")
            size = np.asarray(self.world.model.geom_size[geom_id], dtype=float)
            local = np.asarray(self.world.model.geom_pos[geom_id], dtype=float)
            mobile_bodies.append(body_id)
            mobile_local.append((local + [0.0, 0.0, size[2] + 1e-4]).tolist())
            mobile_area.append(
                float(np.pi * size[0] * size[1] * traits["dorsal_capture_fraction"])
            )
            mobile_absorptivity.append(traits["phototrophic_absorptivity"])
        self._mobile_capture_area = np.asarray(mobile_area, dtype=np.float64)
        self._environment.bind_mobile(
            np.asarray(mobile_bodies, dtype=np.int32),
            self._matrix(mobile_local, 3),
            np.tile(
                np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
                (len(mobile_bodies), 1),
            ),
            np.full(len(mobile_bodies), self._mobile_light_profile, dtype=np.int32),
            self._mobile_capture_area,
            np.asarray(mobile_absorptivity, dtype=np.float64),
            self.mobile_photo_config["photon_flux_per_square_meter_second"],
        )
        light_bodies: list[int] = []
        light_positions: list[list[float]] = []
        light_directions: list[list[float]] = []
        light_intensity: list[float] = []
        light_radius: list[float] = []
        for entity in self.world._entities:
            if entity["id"] == self.illumination_config["light_entity"]:
                continue
            for component in self.world._components[entity["id"]]:
                if component.get("type") != "light":
                    continue
                direction = np.asarray(
                    component.get("direction", [0.0, 0.0, -1.0]), dtype=float
                )
                direction /= np.linalg.norm(direction)
                light_bodies.append(self.world._entity_mj[entity["id"]])
                light_positions.append(
                    list(map(float, component.get("position", [0.0, 0.0, 0.0])))
                )
                light_directions.append(direction.tolist())
                light_intensity.append(float(component.get("intensity", 1.0)))
                light_radius.append(float(component.get("radius", 2.0)))
        self._environment_lights = (
            np.asarray(light_bodies, dtype=np.int32),
            self._matrix(light_positions, 3),
            self._matrix(light_directions, 3),
            np.asarray(light_intensity, dtype=np.float64),
            np.asarray(light_radius, dtype=np.float64),
        )
        self._environment_revision = self.world.model_revision
        self._sync_solar_state(self._solar.state())

    def _sample_environment(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if self._environment_revision != self.world.model_revision:
            self._bind_environment()
        bud_bodies: list[int] = []
        bud_local: list[list[float]] = []
        bud_profiles: list[int] = []
        ordered: list[tuple[str, dict[str, Any]]] = []
        for colony in self.config:
            name = colony["id"]
            if not self.active[name] or not self.growth[name].is_due:
                continue
            body = self.world._entity_mj[colony["bindings"]["branch"]]
            profile = self._light_profile_ids[name]
            for bud in self.growth[name].buds():
                local = np.asarray(bud["position"], dtype=float) + 0.025 * np.asarray(
                    bud["forward"], dtype=float
                )
                bud_bodies.append(body)
                bud_local.append(local.tolist())
                bud_profiles.append(profile)
                ordered.append((name, bud))
        if self._environment_lights is None:
            raise RuntimeError("physical light environment is not bound")
        light = self.world._light
        capture, bud_light, mobile_photons = self._environment.sample(
            self._tissue,
            int(self.world.model._address),
            int(self.world.data._address),
            np.asarray(bud_bodies, dtype=np.int32),
            self._matrix(bud_local, 3),
            np.asarray(bud_profiles, dtype=np.int32),
            dt,
            self._solar_state["toward_sun"],
            self._solar_state["direct_irradiance"],
            self._solar_state["diffuse_irradiance"],
            *self._environment_lights,
            list(map(float, light["position"])),
            float(light["intensity"]),
            bool(light["remaining"] > 0.0),
            [float(self.world.width), float(self.world.height), float(self.world.depth)],
        )
        self._development_light = {colony["id"]: [] for colony in self.config}
        for (name, bud), value in zip(ordered, np.asarray(bud_light), strict=True):
            self._development_light[name].append((bud, float(value)))
        self._development_surface = {colony["id"]: [] for colony in self.config}
        if ordered:
            positions = []
            rotations = []
            colony_slots = []
            for name, bud in ordered:
                colony = self._colony(name)
                body = self.world._entity_mj[colony["bindings"]["branch"]]
                rotation = np.asarray(self.world.data.xmat[body]).reshape(3, 3)
                rotations.append(rotation)
                positions.append(
                    np.asarray(self.world.data.xpos[body])
                    + rotation @ np.asarray(bud["position"], dtype=float)
                )
                colony_slots.append(list(self.growth).index(name))
            binding_owner = {
                entity: index
                for index, colony in enumerate(self.config)
                for entity in colony["bindings"].values()
            }
            enabled = np.zeros(self.world.model.ngeom, dtype=np.bool_)
            geom_colonies = np.full(self.world.model.ngeom, -1, dtype=np.int32)
            for geom, entity_id in self.world._geom_entity.items():
                entity = self.world._entity(entity_id)
                enabled[geom] = entity["mobility"] == "static"
                geom_colonies[geom] = binding_owner.get(entity_id, -1)
            maximum_distance = max(
                max(
                    rule["guidance"]["surface_reach"],
                    rule["guidance"]["clearance_distance"],
                )
                for growth in self.growth.values()
                for rule in growth.grammar["rules"].values()
            )
            distances, directions, hits = load_world_kernels().developmental_surface_cues(
                np.ascontiguousarray(positions, dtype=np.float64),
                np.asarray(colony_slots, dtype=np.int32),
                np.asarray(self.world.model.geom_type, dtype=np.int32),
                np.asarray(self.world.data.geom_xpos, dtype=np.float64),
                np.asarray(self.world.data.geom_xmat, dtype=np.float64),
                np.asarray(self.world.model.geom_size, dtype=np.float64),
                np.asarray(self.world.model.geom_rbound, dtype=np.float64),
                enabled,
                geom_colonies,
                maximum_distance,
            )
            for (name, bud), rotation, distance, direction, hit in zip(
                ordered,
                rotations,
                np.asarray(distances),
                np.asarray(directions),
                np.asarray(hits),
                strict=True,
            ):
                local_direction = np.clip(rotation.T @ direction, -1.0, 1.0)
                local_up = np.clip(
                    rotation.T @ np.asarray([0.0, 0.0, 1.0]), -1.0, 1.0
                )
                self._development_surface[name].append(
                    {
                        "bud_id": bud["bud_id"],
                        "surface_direction": local_direction.astype(float).tolist(),
                        "world_up": local_up.astype(float).tolist(),
                        "surface_distance": float(distance),
                        "surface_geom": int(hit),
                    }
                )
        mobile_photons = np.asarray(mobile_photons, dtype=np.float64)
        if (
            mobile_photons.shape != (len(self._mobile_specs),)
            or not np.isfinite(mobile_photons).all()
            or np.any(mobile_photons < 0.0)
        ):
            raise RuntimeError("native mobile photon supply is invalid")
        return np.asarray(capture, dtype=np.float64), mobile_photons

    def _signals(self, colony: Mapping[str, Any]) -> list[dict[str, Any]]:
        mineral = float(
            self.web.pools[
                colony["body_row"], self.web.chemistry.pools.index("mineral")
            ]
        )
        nutrient = mineral / (mineral + colony["mineral_half_saturation"])
        result = []
        surfaces = {
            value["bud_id"]: value
            for value in self._development_surface.get(colony["id"], [])
        }
        clearance_distance = max(
            rule["guidance"]["clearance_distance"]
            for rule in self.growth[colony["id"]].grammar["rules"].values()
        )
        for bud, light in self._development_light.get(colony["id"], []):
            surface = surfaces[bud["bud_id"]]
            parent = bud["parent_part"]
            parent_record = (
                self.parts.get(f"{colony['id']}:{parent}") if parent is not None else None
            )
            support = 1.0
            if parent_record is not None:
                original = sum(parent_record["initial_resources"].values())
                remaining = sum(parent_record["resources"].values())
                support = min(1.0, remaining / original) if original > 0.0 else 0.0
                attachment = parent_record.get("attachment")
                if attachment is not None and not attachment["active"]:
                    support *= 0.5
            competition = max(
                0.0, 1.0 - surface["surface_distance"] / clearance_distance
            )
            result.append(
                {
                    "bud_id": bud["bud_id"],
                    "light": light,
                    "nutrient": nutrient,
                    "support": support,
                    "competition": competition,
                    **surface,
                }
            )
        return result

    def _mobile_photo_report(
        self, supplied: np.ndarray, photon_used: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows = np.asarray(
            [spec["body_row"] for spec in self._mobile_specs], dtype=np.int64,
        )
        resources = self.web.pools[rows]
        body_mass = resources @ self.web.chemistry._arrays[1].sum(axis=1)
        stored_energy = resources @ self.web.chemistry._arrays[2] + self.web.atp[rows]
        return [
            {
                "id": spec["id"],
                "capture_area_square_meters": float(self._mobile_capture_area[index]),
                "absorptivity": self._mobile_traits[index]["phototrophic_absorptivity"],
                "photons_supplied": float(supplied[index]),
                "photons_used": float(photon_used[spec["body_row"]]),
                "body_material_mass": float(body_mass[index]),
                "body_stored_energy": float(stored_energy[index]),
            }
            for index, spec in enumerate(self._mobile_specs)
        ]

    def advance(self, dt: float) -> dict[str, Any]:
        """Advance chemistry and development after the caller advances physics."""
        if not np.isfinite(dt) or not 0 < dt <= 1.0:
            raise ValueError("biosphere step must be in (0, 1] seconds")
        self._check_structure()
        if self.exchange is not None:
            self.exchange.before_reactions(dt)
        for colony in self.config:
            if self.active[colony["id"]]:
                self.growth[colony["id"]].elapse(dt)
        self._sync_solar_state(self._solar.advance(dt))
        capture, mobile_photons = self._sample_environment(dt)
        photons = np.zeros(self.web.count, dtype=np.float64)
        for colony_index, colony in enumerate(self.config):
            if self.active[colony["id"]]:
                photons[colony["body_row"]] = (
                    dt * colony["photon_flux"] * capture[colony_index]
                )
        for spec, supplied in zip(self._mobile_specs, mobile_photons, strict=True):
            photons[spec["body_row"]] = supplied
        if self.mobility is not None:
            self.mobility.before_reactions(dt)
        ledger = self.web.step(dt, photons, np.zeros(self.web.count))
        if self.exchange is not None:
            self.exchange.after_reactions(dt)
        if self.mobility is not None:
            self.mobility.after_reactions(dt)
        self._distribute_turnover(ledger)
        reports = self._develop()
        if self.materials is not None:
            self.materials.sync_geometry()
            self._sync_material_cues()
        self.last_report = {
            "time": self.web.time,
            "captured_photons": float(ledger["photon_used"].sum()),
            "developments": reports,
            "evidence_events": copy.deepcopy(self._growth_evidence)
            + ([] if self.exchange is None else self.exchange.step_events),
            "parts": len(self.parts),
            "illumination": copy.deepcopy(self._solar_state),
            "mobile_phototrophy": self._mobile_photo_report(
                mobile_photons, ledger["photon_used"],
            ),
            "accounting": self.accounting(),
            "exchange": self.exchange.view() if self.exchange is not None else None,
            "hatch_offers": self.hatch_offers(),
        }
        self._growth_evidence.clear()
        return copy.deepcopy(self.last_report)

    def _sync_material_cues(self) -> None:
        # Cues transduce present chemistry into sensed light/tracer emission.
        # The odor fields remain signal tracers, not conserved nutrient pools.
        import mujoco

        for cue in self.materials.surface_cues():
            entity = self.world._entity(cue["entity"])
            material = mujoco.mj_name2id(
                self.world.model,
                mujoco.mjtObj.mjOBJ_MATERIAL,
                f"mat:{entity['material']}",
            )
            if material < 0:
                raise ValueError("chemical surface requires an authored material")
            users = np.flatnonzero(self.world.model.geom_matid == material)
            if any(
                self.world._geom_entity.get(int(geom)) != entity["id"] for geom in users
            ):
                raise ValueError(
                    "chemical surface color requires its own physical material"
                )
            self.world.model.mat_rgba[material, :3] = cue["rgb"]

    def field_sources(self) -> list[dict[str, Any]]:
        sources = []
        if self.materials is not None:
            for cue in self.materials.surface_cues():
                body = self.world._entity_mj[cue["entity"]]
                position = self.world.data.xpos[body].astype(float).tolist()
                for channel, strength in enumerate(cue["odor"]):
                    if strength > 0:
                        sources.append(
                            {
                                "key": f"chemical-surface:{cue['entity']}:{channel}",
                                "position": position,
                                "channel": channel,
                                "rate": 0.018 * strength,
                                "spread": 0.04,
                            }
                        )
        if self.mobility is not None:
            sources.extend(self.mobility.field_sources())
        return sources

    def hatch_offers(self) -> list[dict[str, Any]]:
        return [] if self.mobility is None else self.mobility.hatch_offers()

    def prepare_hatch(
        self, offer_id: str, child_genome: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self.mobility is None:
            raise RuntimeError("biosphere has no mobile lifecycle")
        return self.mobility.prepare_hatch(offer_id, child_genome)

    def prepare_newborn(
        self,
        candidate_world: Any,
        hatch_proposal: Mapping[str, Any],
        newborn: Mapping[str, Any],
    ) -> PreparedNewborn:
        """Build a private B+1 chemical owner without mutating this biosphere."""
        if self.mobility is None or self.exchange is None:
            raise RuntimeError("resident birth requires mobile exchange physiology")
        if len(self.world.bodies) >= 32:
            raise ValueError("resident capacity exceeded")
        if not isinstance(newborn, Mapping) or set(newborn) != {
            "body", "mobile", "exchange", "founders",
        }:
            raise ValueError("newborn bundle fields differ")
        if (
            len(candidate_world.bodies) != len(self.world.bodies) + 1
            or [body.id for body in candidate_world.bodies[:-1]]
            != [body.id for body in self.world.bodies]
            or candidate_world.spec["bodies"][-1] != newborn["body"]
        ):
            raise ValueError("newborn physical candidate differs")
        resident_id = candidate_world.bodies[-1].id
        if resident_id in self.mobility.residents:
            raise ValueError("newborn identity already exists")

        founders = newborn["founders"]
        if not isinstance(founders, Mapping) or set(founders) != {
            "body", "gut", "structure", "gland", "brood",
        }:
            raise ValueError("newborn requires five private founders")
        founder_rows = []
        for name in ("body", "gut", "structure", "gland", "brood"):
            row = founders[name]
            if not isinstance(row, Mapping) or set(row) != {
                "enzymes", "pools", "atp", "atp_capacity",
            }:
                raise ValueError("newborn founder fields differ")
            if row["pools"] or float(row["atp"]) != 0.0:
                raise ValueError("newborn rows must be empty before brood funding")
            founder_rows.append(copy.deepcopy(dict(row)))

        proposal = copy.deepcopy(dict(hatch_proposal))
        proposal_sha256 = proposal.pop("proposal_sha256", None)
        if (
            proposal.get("format") != "chreatures-hatch-proposal-v1"
            or proposal_sha256 != hashlib.sha256(canonical(proposal)).hexdigest()
            or proposal["web_state_sha256"]
            != hashlib.sha256(canonical(self.web.snapshot())).hexdigest()
        ):
            raise ValueError("newborn hatch proposal is stale or altered")
        offer = next(
            (
                value
                for value in self.hatch_offers()
                if value["offer_id"] == proposal["offer"]["offer_id"]
            ),
            None,
        )
        if offer != proposal["offer"]:
            raise ValueError("newborn hatch offer differs")

        base = self.web.count
        expanded_web = self.web.expanded(
            [row["enzymes"] for row in founder_rows],
            [row["atp_capacity"] for row in founder_rows],
        )
        assigned = {
            f"{name}_row": base + index
            for index, name in enumerate(("body", "gut", "structure", "gland", "brood"))
        }
        mobile = copy.deepcopy(dict(newborn["mobile"]))
        if "id" in mobile or set(mobile) & set(assigned):
            raise ValueError("generator, not genotype, assigns newborn row identities")
        mobile.update(id=resident_id, **assigned)
        mobile_config = [*copy.deepcopy(self.mobility.config), mobile]

        stock = expanded_web.pools[offer["brood_row"]]
        stock_mass = float(stock @ self.mobility.mass_weights)
        factor = min(1.0, offer["material_target"] / stock_mass) if stock_mass > 0 else 0.0
        funding_receipt = expanded_web.transfer_batch(
            [offer["brood_row"]], [assigned["body_row"]],
            [dict(zip(
                expanded_web.chemistry.pools, (stock * factor).tolist(), strict=True
            ))],
            [offer["energy_target"]],
        )
        funded_resources = np.asarray(
            funding_receipt["moved_resources"][0], dtype=np.float64
        )
        funded_atp = float(funding_receipt["moved_atp"][0])
        if (
            float(funded_resources @ self.mobility.mass_weights) + 1e-12
            < offer["material_target"]
            or funded_atp + 1e-12 < offer["energy_target"]
        ):
            raise ValueError("newborn body capacity cannot accept prepared brood funding")

        candidate = type(self)(
            candidate_world, expanded_web, self.config,
            illumination_cycle=self.illumination_config,
            mobile_phototrophy=self.mobile_photo_config,
            mobiles=mobile_config,
        )
        saved = self.snapshot()
        candidate._sync_solar_state(
            candidate._solar.restore_clock(
                saved["illumination_cycle"]["clock_seconds"]
            )
        )
        candidate.growth = {
            key: GrowthSystem.restore(candidate.growth[key].grammar, value)
            for key, value in saved["growth"].items()
        }
        candidate.parts = {
            key: copy.deepcopy(saved["parts"][key]) for key in saved["part_order"]
        }
        candidate.active = copy.deepcopy(saved["active"])
        candidate.initial_totals = copy.deepcopy(saved["initial_totals"])
        candidate.initial_ledger = copy.deepcopy(saved["initial_ledger"])
        candidate.last_report = copy.deepcopy(saved["last_report"])
        candidate._growth_evidence = copy.deepcopy(saved["growth_evidence"])
        candidate._bind_tissue()

        from .ecological_exchange import EcologicalExchange
        from .material_objects import MaterialObjects
        from .somatic import SomaticPhysiology

        candidate.mobility = SomaticPhysiology.expanded_from(
            self.mobility, candidate, mobile_config
        )
        candidate_world.bind_physiology(candidate.mobility)
        if saved["material_objects"] is not None:
            candidate.materials = MaterialObjects.restore(
                candidate_world, candidate, saved["material_objects"]
            )
        exchange_config = copy.deepcopy(self.exchange.config)
        exchange_mobile = copy.deepcopy(dict(newborn["exchange"]))
        if "id" in exchange_mobile:
            raise ValueError("generator, not genotype, assigns exchange identity")
        exchange_mobile["id"] = resident_id
        exchange_config["mobiles"].append(exchange_mobile)
        candidate.exchange = EcologicalExchange.expanded_from(
            self.exchange, candidate, exchange_config
        )
        candidate.mobility._lifecycle.commit(
            np.asarray([offer["resident_index"]], dtype=np.int32),
            np.asarray([offer["serial"]], dtype=np.int64),
        )
        candidate.mobility._maturity[offer["resident_index"]] = 0.0
        candidate.mobility.sync_bodies()
        if candidate.last_report:
            candidate.last_report["hatch_offers"] = candidate.hatch_offers()
        source_sha256 = hashlib.sha256(canonical(saved)).hexdigest()
        funding = {
            "format": "chreatures-hatch-funding-v1",
            "offer_id": offer["offer_id"], "parent_id": offer["parent_id"],
            "resident_id": resident_id,
            "child_genome": proposal["child_genome"],
            "child_genome_sha256": proposal["child_genome_sha256"],
            "resources": dict(zip(
                expanded_web.chemistry.pools, funded_resources.tolist(), strict=True
            )),
            "atp": funded_atp, "assigned_rows": assigned,
        }
        return PreparedNewborn(
            source_sha256, str(proposal_sha256), resident_id, candidate, funding
        )

    def commit_newborn(self, prepared: PreparedNewborn) -> Biosphere:
        """Adopt a prepared chemical owner after its physical transaction commits."""
        if not isinstance(prepared, PreparedNewborn) or prepared._committed:
            raise ValueError("newborn preparation is absent or already committed")
        current = hashlib.sha256(canonical(self.snapshot())).hexdigest()
        if current != prepared.source_sha256:
            raise RuntimeError("newborn preparation is stale after biosphere change")
        if (
            len(self.world.bodies) != len(self.mobility.residents) + 1
            or self.world.bodies[-1].id != prepared.resident_id
        ):
            raise RuntimeError("physical newborn was not committed at the appended slot")
        candidate = prepared.candidate
        if candidate.world.model is not self.world.model:
            raise RuntimeError("committed physical world differs from prepared newborn")
        candidate.world = self.world
        if candidate.materials is not None:
            candidate.materials.world = self.world
        self.world.bind_physiology(candidate.mobility)
        candidate.mobility.sync_bodies()
        prepared._committed = True
        return candidate

    def _bind_tissue(self) -> None:
        """Rebind native numeric tissue after a committed topology change."""
        self._tissue.bind(self.parts, self.web.pools)
        self._bind_environment()

    def _check_structure(self) -> None:
        self._tissue.validate(self.web.pools)
        for part_id, part in self.parts.items():
            attachment = part.get("attachment")
            if attachment is None or not attachment["active"]:
                continue
            geom = mujoco.mj_name2id(
                self.world.model, mujoco.mjtObj.mjOBJ_GEOM, attachment["geom"]
            )
            target = (
                self.world._entity(attachment["entity"])
                if attachment["entity"] in self.world._entity_mj
                else None
            )
            shape_index = attachment["shape_index"]
            shape_valid = target is not None and (
                0 <= shape_index < len(target["shapes"])
                and hashlib.sha256(canonical(target["shapes"][shape_index])).hexdigest()
                == attachment["shape_sha256"]
            )
            if (
                geom < 0
                or self.world._geom_entity.get(geom) != attachment["entity"]
                or not shape_valid
            ):
                attachment["active"] = False
                self._growth_evidence.append(
                    {
                        "kind": "developmental-attachment-invalidated",
                        "actors": {
                            "bodies": [],
                            "entities": [part["entity"], attachment["entity"]],
                        },
                        "quantities": [],
                        "details": {
                            "part_id": part_id,
                            "entity_roles": {
                                part["entity"]: "constructed",
                                attachment["entity"]: "attachment",
                            },
                        },
                        "source": {"stream": "biosphere-growth"},
                    }
                )

    def _distribute_turnover(self, ledger: Mapping[str, Any]) -> None:
        # Turnover changes live tissue to detritus without removing its material
        # or geometry. Dead scaffold can persist; removal is a separate transfer.
        self._tissue.turnover(self.parts, ledger["extent"])

    def _develop(self) -> list[dict[str, Any]]:
        operations = []
        staged: list[tuple[dict[str, Any], dict[str, Any], GrowthSystem]] = []
        pending: list[tuple[str, str]] = []
        # Existing part records are immutable here. Allocate a new mapping only
        # after a proposal has passed its resource checks; _record_parts appends
        # fresh records and never edits the values shared with self.parts.
        next_parts: dict[str, dict[str, Any]] | None = None
        try:
            for colony in self.config:
                name = colony["id"]
                if not self.active[name]:
                    continue
                growth = self.growth[name]
                if not growth.is_due:
                    continue
                pool = self.web.pools[colony["body_row"]]
                budget = sum(
                    pool[self.web.chemistry.pools.index(key)]
                    for key in growth.resource_names
                )
                proposal = growth.propose(self._signals(colony), float(budget))
                if proposal is None:
                    continue
                pending.append((name, proposal["token"]))
                self._resolve_attachments(colony, proposal)
                request = proposal["request"]
                vector = self.web.chemistry.resources(request["resources"])
                if (
                    np.any(vector > pool)
                    or request["atp"] > self.web.atp[colony["body_row"]]
                ):
                    growth.reject(proposal["token"])
                    continue
                candidate = GrowthSystem.restore(growth.grammar, growth.snapshot())
                # Prevalidate the exact receipt on unexposed staged state. It is
                # published only after the actual physical commit succeeds.
                candidate.commit(
                    proposal["token"],
                    request["resources"],
                    request["atp"],
                    physical_committed=True,
                )
                ops = growth.physical_operations(proposal, colony["bindings"])
                operations.extend(ops)
                if next_parts is None:
                    next_parts = self.parts.copy()
                self._record_parts(colony, proposal, ops, next_parts)
                staged.append((colony, proposal, candidate))
            if not staged:
                return []
            transaction = self.world.prepare_topology_batch(operations)
            before = self.web.snapshot()
            try:
                for colony, proposal, _ in staged:
                    request = proposal["request"]
                    self.web.transfer(
                        colony["body_row"],
                        colony["structure_row"],
                        request["resources"],
                    )
                    self.web.pay_work(colony["body_row"], request["atp"])
                transaction.commit()
            except Exception:
                self.web = MetabolicWeb.restore(before)
                raise
            if next_parts is None:
                raise RuntimeError("staged development has no physical part records")
            self.parts = next_parts
            self._bind_tissue()
            for colony, _, candidate in staged:
                self.growth[colony["id"]] = candidate
            reports = [
                {
                    "colony": colony["id"],
                    "request": proposal["request"],
                    "token": proposal["token"],
                    "parts": [
                        f"{colony['id']}:{part['id']}"
                        for part in (
                            *proposal["geometry"]["segments"],
                            *proposal["geometry"]["leaves"],
                        )
                    ],
                }
                for colony, proposal, _ in staged
            ]
            for (colony, proposal, _), report in zip(staged, reports, strict=True):
                entity_roles = {
                    colony["bindings"][part["kind"]]: "constructed"
                    for part in proposal["geometry"]["segments"]
                }
                attachments = []
                parents = []
                for part in proposal["geometry"]["segments"]:
                    if part["parent_part"] is not None:
                        parents.append(f"{colony['id']}:{part['parent_part']}")
                    if part["attachment"] is not None:
                        attachment = part["attachment"]
                        entity_roles[attachment["entity"]] = "attachment"
                        attachments.append(copy.deepcopy(attachment))
                self._growth_evidence.append(
                    {
                        "kind": "developmental-growth-committed",
                        "actors": {
                            "bodies": [],
                            "entities": sorted(entity_roles),
                        },
                        "quantities": [
                            {
                                "name": name,
                                "value": float(amount),
                                "unit": "synthetic_pool_quantity",
                            }
                            for name, amount in proposal["request"]["resources"].items()
                        ]
                        + [
                            {
                                "name": "ATP",
                                "value": float(proposal["request"]["atp"]),
                                "unit": "synthetic_energy_quantity",
                            },
                            {
                                "name": "biomass",
                                "value": float(proposal["request"]["biomass"]),
                                "unit": "synthetic_mass",
                            },
                        ],
                        "details": {
                            "colony": colony["id"],
                            "token": proposal["token"],
                            "new_part_ids": report["parts"],
                            "removed_part_ids": [],
                            "parent_part_ids": parents,
                            "attachments": attachments,
                            "entity_roles": entity_roles,
                        },
                        "source": {
                            "stream": "biosphere-growth",
                            "grammar_sha256": self.growth[colony["id"]].grammar_hash,
                        },
                    }
                )
            return reports
        except Exception:
            # Preparation/payment failures consume neither a developmental RNG
            # draw nor a bud. Unexpected physical failures propagate to the
            # runtime's existing failed-tick pause boundary.
            for name, token in pending:
                growth = self.growth[name]
                if growth.snapshot()["pending"] is not None:
                    growth.reject(token)
            raise

    def _resolve_attachments(self, colony, proposal) -> None:
        """Resolve transient geom indices to stable identities before compilation."""
        entity = colony["bindings"]["branch"]
        body = self.world._entity_mj[entity]
        rotation = np.asarray(self.world.data.xmat[body]).reshape(3, 3)
        origin = np.asarray(self.world.data.xpos[body])
        for part in proposal["geometry"]["segments"]:
            geom = part.pop("attachment_geom")
            local_point = np.asarray(part.pop("attachment_point"), dtype=float)
            if geom < 0:
                part["attachment"] = None
                continue
            name = mujoco.mj_id2name(self.world.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            target = self.world._geom_entity.get(geom)
            if not name or target is None or self.world._entity(target)["mobility"] != "static":
                raise RuntimeError("developmental attachment target became invalid")
            shape_index = int(name.rsplit(":", 1)[1])
            target_shape = self.world._entity(target)["shapes"][shape_index]
            part["attachment"] = {
                "geom": name,
                "entity": target,
                "shape_index": shape_index,
                "shape_sha256": hashlib.sha256(canonical(target_shape)).hexdigest(),
                "point_world": (origin + rotation @ local_point).astype(float).tolist(),
                "active": True,
            }

    def _record_parts(self, colony, proposal, operations, destination):
        geometry = proposal["geometry"]
        composition = self.growth[colony["id"]].grammar["resources"]["composition"]
        names = self.growth[colony["id"]].resource_names
        for op in operations:
            entity_id = op["id"]
            kind = next(
                key for key, value in colony["bindings"].items() if value == entity_id
            )
            generated = (
                geometry["leaves"]
                if kind == "leaf"
                else [part for part in geometry["segments"] if part["kind"] == kind]
            )
            start_index = len(self.world._entity(entity_id)["shapes"])
            for offset, (part, shape) in enumerate(
                zip(generated, op["shapes"], strict=True)
            ):
                part_id = f"{colony['id']}:{part['id']}"
                if part_id in destination:
                    raise ValueError("developmental part identity already exists")
                destination[part_id] = {
                    "colony": colony["id"],
                    "kind": kind,
                    "entity": entity_id,
                    "shape_index": start_index + offset,
                    "shape": copy.deepcopy(shape),
                    "area": float(part.get("area", 0.0)),
                    "resources": {
                        name: float(part["biomass"] * composition[kind][i])
                        for i, name in enumerate(names)
                    },
                    "born": self.web.time,
                    "parent_part": (
                        f"{colony['id']}:{part['parent_part']}"
                        if part.get("parent_part") is not None
                        else None
                    ),
                    "transport_resistance": float(
                        part.get("transport_resistance", 0.0)
                    ),
                    "attachment": copy.deepcopy(part.get("attachment")),
                }
                if kind == "leaf":
                    parent_segment = next(
                        (
                            segment
                            for segment in geometry["segments"]
                            if segment["parent_bud"] == part["parent_bud"]
                        ),
                        None,
                    )
                    if parent_segment is not None:
                        destination[part_id]["parent_part"] = (
                            f"{colony['id']}:{parent_segment['id']}"
                        )
                        destination[part_id]["transport_resistance"] = float(
                            parent_segment["transport_resistance"]
                        )
                destination[part_id]["initial_resources"] = destination[part_id][
                    "resources"
                ].copy()

    def release_parts(
        self, part_ids: list[str], receiver: int | None
    ) -> dict[str, Any]:
        """Environmental removal with exact tissue transfer, no resource deletion.

        This is an explicit intervention seam. A future consumer must establish
        physical access before calling it; it is not a remote feeding action.
        """
        if not part_ids or len(set(part_ids)) != len(part_ids):
            raise ValueError("release requires distinct existing parts")
        receipt = self.transfer_parts(dict.fromkeys(part_ids, receiver))
        receipt["receiver"] = receiver
        return receipt

    def transfer_parts(self, receivers: Mapping[str, int | None]) -> dict[str, Any]:
        """Remove a physically selected set in one topology/material transaction."""
        part_ids = list(receivers)
        if not part_ids or any(key not in self.parts for key in part_ids):
            raise ValueError("transfer requires existing physical parts")
        for key, receiver in receivers.items():
            if receiver is not None and (
                isinstance(receiver, bool)
                or not isinstance(receiver, int)
                or not 0 <= receiver < self.web.count
                or receiver in {c["structure_row"] for c in self.config}
            ):
                raise ValueError("invalid receiving compartment")
        removed = set(part_ids)
        dangling = {
            key
            for key, part in self.parts.items()
            if key not in removed and part.get("parent_part") in removed
        }
        if dangling:
            raise ValueError(
                "developmental removal must include directly dependent parts"
            )
        entities = {self.parts[key]["entity"] for key in removed}
        operations = []
        next_parts = copy.deepcopy(self.parts)
        for entity_id in sorted(entities):
            entity = copy.deepcopy(self.world._entity(entity_id))
            indices = {
                self.parts[key]["shape_index"]
                for key in removed
                if self.parts[key]["entity"] == entity_id
            }
            index_map = {}
            kept = []
            for index, shape in enumerate(entity["shapes"]):
                if index not in indices:
                    index_map[index] = len(kept)
                    kept.append(shape)
            entity["shapes"] = kept
            operations.append({"op": "replace", "id": entity_id, "entity": entity})
            for key, part in next_parts.items():
                if key not in removed and part["entity"] == entity_id:
                    part["shape_index"] = index_map[part["shape_index"]]
        for key in removed:
            del next_parts[key]
        transaction = self.world.prepare_topology_batch(operations)
        before = self.web.snapshot()
        totals = np.zeros(len(self.web.chemistry.pools))
        try:
            for key in part_ids:
                part = self.parts[key]
                colony = self._colony(part["colony"])
                self.web.transfer(
                    colony["structure_row"], receivers[key], part["resources"]
                )
                totals += self.web.chemistry.resources(part["resources"])
            transaction.commit()
        except Exception:
            self.web = MetabolicWeb.restore(before)
            raise
        self.parts = next_parts
        invalidated_buds = {}
        for colony in self.config:
            prefix = f"{colony['id']}:"
            local = [key[len(prefix) :] for key in removed if key.startswith(prefix)]
            invalidated_buds[colony["id"]] = self.growth[colony["id"]].invalidate_parts(
                local
            )
        self._bind_tissue()
        receipt = {
            "parts": part_ids,
            "receivers": dict(receivers),
            "resources": dict(zip(self.web.chemistry.pools, totals.tolist())),
            "invalidated_buds": invalidated_buds,
        }
        entities = sorted(entities)
        self._growth_evidence.append(
            {
                "kind": "developmental-parts-removed",
                "actors": {"bodies": [], "entities": entities},
                "quantities": [
                    {
                        "name": name,
                        "value": float(value),
                        "unit": "synthetic_pool_quantity",
                    }
                    for name, value in receipt["resources"].items()
                ],
                "details": {
                    "new_part_ids": [],
                    "removed_part_ids": part_ids,
                    "invalidated_buds": invalidated_buds,
                    "entity_roles": dict.fromkeys(entities, "constructed"),
                },
                "source": {"stream": "biosphere-growth"},
            }
        )
        return receipt

    def suspend_development(self, colony_id: str) -> None:
        """Stop new construction; existing structures and chemistry remain."""
        if colony_id not in self.active:
            raise ValueError("unknown colony")
        self.active[colony_id] = False

    def accounting(self) -> dict[str, Any]:
        totals = self.web.totals()
        ledger = self._ledger() - np.asarray(self.initial_ledger)
        elements = {
            name: amount - self.initial_totals["elements"][name]
            for name, amount in totals["elements"].items()
        }
        return {
            **totals,
            "elemental_residual": elements,
            "captured_photons": float(ledger[0]),
            "heat": float(ledger[2] + ledger[3]),
            "exported_work": float(ledger[4]),
            "energy_residual": float(
                totals["stored_energy"]
                - self.initial_totals["stored_energy"]
                - ledger[0]
                + ledger[2]
                + ledger[3]
                + ledger[4]
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "config": copy.deepcopy(self.config),
            "config_sha256": self.config_sha256,
            "illumination_cycle": {
                "config": copy.deepcopy(self.illumination_config),
                "config_sha256": self.illumination_sha256,
                "clock_seconds": float(self._solar.clock_seconds()),
            },
            "mobile_phototrophy": {
                "config": copy.deepcopy(self.mobile_photo_config),
                "config_sha256": self.mobile_photo_sha256,
            },
            "web": self.web.snapshot(),
            "growth": {key: value.snapshot() for key, value in self.growth.items()},
            "parts": copy.deepcopy(self.parts),
            # JSON canonicalization sorts mapping keys. Part insertion order
            # is causal state because floating-point reductions use it.
            "part_order": list(self.parts),
            "active": self.active.copy(),
            "initial_totals": copy.deepcopy(self.initial_totals),
            "initial_ledger": self.initial_ledger.copy(),
            "last_report": copy.deepcopy(self.last_report),
            "growth_evidence": copy.deepcopy(self._growth_evidence),
            "material_objects": self.materials.snapshot()
            if self.materials is not None
            else None,
            "mobility": self.mobility.snapshot() if self.mobility is not None else None,
            "exchange": self.exchange.snapshot() if self.exchange is not None else None,
        }

    @classmethod
    def restore(cls, world: Any, snapshot: Mapping[str, Any]) -> Biosphere:
        snapshot = copy.deepcopy(snapshot)
        if snapshot.get("format") != FORMAT:
            raise ValueError("unsupported biosphere snapshot")
        illumination = snapshot.get("illumination_cycle")
        if not isinstance(illumination, dict) or set(illumination) != {
            "config", "config_sha256", "clock_seconds",
        }:
            raise ValueError("invalid saved solar cycle")
        mobile_photo = snapshot.get("mobile_phototrophy")
        if not isinstance(mobile_photo, dict) or set(mobile_photo) != {
            "config", "config_sha256",
        }:
            raise ValueError("invalid saved mobile phototrophy")
        mobile_state = snapshot["mobility"]
        instance = cls(
            world,
            MetabolicWeb.restore(snapshot["web"]),
            snapshot["config"],
            illumination_cycle=illumination["config"],
            mobile_phototrophy=mobile_photo["config"],
            mobiles=mobile_state["config"] if mobile_state is not None else None,
        )
        if instance.config_sha256 != snapshot["config_sha256"]:
            raise ValueError("developmental colony configuration differs")
        if instance.illumination_sha256 != illumination["config_sha256"]:
            raise ValueError("solar cycle configuration differs")
        if instance.mobile_photo_sha256 != mobile_photo["config_sha256"]:
            raise ValueError("mobile phototrophy configuration differs")
        instance._sync_solar_state(
            instance._solar.restore_clock(illumination["clock_seconds"])
        )
        if set(snapshot["growth"]) != set(instance.growth) or set(
            snapshot["active"]
        ) != set(instance.growth):
            raise ValueError("developmental identity set differs")
        instance.growth = {
            key: GrowthSystem.restore(instance.growth[key].grammar, value)
            for key, value in snapshot["growth"].items()
        }
        part_order = snapshot.get("part_order")
        if (
            not isinstance(part_order, list)
            or any(not isinstance(key, str) for key in part_order)
            or len(part_order) != len(set(part_order))
            or set(part_order) != set(snapshot["parts"])
        ):
            raise ValueError("developmental part order differs from saved tissue")
        instance.parts = {
            key: copy.deepcopy(snapshot["parts"][key]) for key in part_order
        }
        for part in instance.parts.values():
            if (
                part["colony"] not in instance.growth
                or world._entity(part["entity"])["shapes"][part["shape_index"]]
                != part["shape"]
            ):
                raise ValueError("developed tissue and physical shape differ")
        instance.active = copy.deepcopy(snapshot["active"])
        instance.initial_totals = copy.deepcopy(snapshot["initial_totals"])
        instance.initial_ledger = copy.deepcopy(snapshot["initial_ledger"])
        instance.last_report = copy.deepcopy(snapshot["last_report"])
        instance._growth_evidence = copy.deepcopy(snapshot["growth_evidence"])
        instance._bind_tissue()
        if instance.mobility is not None:
            instance.mobility.restore_state(mobile_state)
        if snapshot["material_objects"] is not None:
            from .material_objects import MaterialObjects

            instance.materials = MaterialObjects.restore(
                world, instance, snapshot["material_objects"]
            )
        if snapshot["exchange"] is not None:
            from .ecological_exchange import EcologicalExchange

            instance.exchange = EcologicalExchange.restore(
                instance, snapshot["exchange"]
            )
        return instance


__all__ = ["Biosphere", "PreparedNewborn"]
