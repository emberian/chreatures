"""Finite chemical material bound to ordinary physical entities.

The metabolic web is the sole inventory.  This adapter proves local physical
access, prepares exact named-pool transfers, and changes physical geometry only
when declared content boundaries are crossed.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .metabolism import MetabolicWeb, canonical


FORMAT = "chreatures-material-objects-v1"
PROPOSAL_FORMAT = "chreatures-material-transfer-v1"
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")


def _number(
    value: Any, name: str, low: float | None = None, high: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or (low is not None and result < low)
        or (high is not None and result > high)
    ):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _vector(value: Any, length: int, name: str, low: float, high: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [_number(item, name, low, high) for item in value]


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


class MaterialObjects:
    """Bind finite web compartments to physical packets and stores.

    ``web_access`` may be a :class:`MetabolicWeb`, an owner whose ``web``
    attribute is a web, or a zero-argument callable returning one.  Resolving it
    for every operation keeps the binding valid when an owning Biosphere rolls
    its web back by replacement.
    """

    MAX_OBJECTS = 64
    MAX_BOUNDARIES = 16
    MAX_BATCH = 256

    def __init__(self, world: Any, web_access: Any, spec: Any):
        self.world = world
        self._web_access = web_access
        self.config = self._normalize_spec(spec)
        self.config_sha256 = hashlib.sha256(canonical(self.config)).hexdigest()
        self._validate_web_bindings()
        self._prepare_runtime_cache()
        self._base_entities: dict[str, dict[str, Any]] = {}
        self._state: dict[str, dict[str, Any]] = {}
        for item in self.config["objects"]:
            entity_id = item["entity"]
            dormant = item.get("dormant_template")
            entity = copy.deepcopy(
                dormant if dormant is not None else self._world_entity(entity_id)
            )
            self._validate_physical_binding(item, entity, dormant is not None)
            boundary = self._boundary(item, self._web().pools[item["row"]])
            self._base_entities[entity_id] = entity
            if dormant is not None:
                if entity_id in self._existing_ids() or boundary is not None:
                    raise ValueError("dormant material slot must begin empty and absent")
                self.world.prepare_topology_batch([{"op": "add", "entity": entity}])
                self._state[entity_id] = {"active": False, "boundary": None}
            else:
                if boundary != 0:
                    raise ValueError("founder material must occupy the first geometry boundary")
                self._state[entity_id] = {"active": True, "boundary": 0}
        self.transfer_count = 0
        self.cumulative_withdrawn = [0.0] * len(self._web().chemistry.pools)
        self.cumulative_deposited = [0.0] * len(self._web().chemistry.pools)
        self.geometry_syncs = 0
        self.last_receipt: dict[str, Any] | None = None
        self.last_geometry_sync: list[dict[str, Any]] = []
        self._check_physical_state()

    def _prepare_runtime_cache(self) -> None:
        """Cache immutable bindings and one compact material-row read."""
        web = self._web()
        names = web.chemistry.pools
        pool_indices = {name: index for index, name in enumerate(names)}
        self._items_by_entity = {
            item["entity"]: item for item in self.config["objects"]
        }
        self._donor_rows = {
            item["entity"]: item["row"] for item in self.config["objects"]
        }
        self._row_indices = np.asarray(
            [item["row"] for item in self.config["objects"]], dtype=np.intp
        )
        self._capacity_matrix = np.asarray(
            [
                [item["capacities"].get(name, 0.0) for name in names]
                for item in self.config["objects"]
            ],
            dtype=np.float64,
        )
        self._content_terms = {
            item["entity"]: tuple(
                (pool_indices[name], weight)
                for name, weight in item["content_weights"].items()
            )
            for item in self.config["objects"]
        }
        self._surface_terms = {
            item["entity"]: (
                tuple(
                    (pool_indices[name], np.asarray(coefficient, dtype=np.float64))
                    for name, coefficient in item["surface"]["rgb_coefficients"].items()
                ),
                tuple(
                    (pool_indices[name], np.asarray(coefficient, dtype=np.float64))
                    for name, coefficient in item["surface"]["odor_coefficients"].items()
                ),
            )
            for item in self.config["objects"]
        }
        self._expected_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._physical_check_revision: int | None = None
        self._inventory_cache = np.asarray(web.pools[self._row_indices]).copy()
        self._surface_inventory_cache: np.ndarray | None = None
        self._surface_cache: list[dict[str, Any]] | None = None

    def _mark_inventory_current(self) -> None:
        self._inventory_cache = np.asarray(
            self._web().pools[self._row_indices]
        ).copy()

    def _web(self) -> MetabolicWeb:
        candidate = self._web_access() if callable(self._web_access) else self._web_access
        if not isinstance(candidate, MetabolicWeb):
            candidate = getattr(candidate, "web", None)
        if not isinstance(candidate, MetabolicWeb):
            raise TypeError("material objects require MetabolicWeb access")
        return candidate

    def _world_entity(self, entity_id: str) -> dict[str, Any]:
        getter = getattr(self.world, "_entity", None)
        if not callable(getter):
            raise TypeError("material objects require physical entity lookup")
        try:
            return getter(entity_id)
        except StopIteration as exc:
            raise ValueError(f"material entity {entity_id!r} is absent") from exc

    def _validate_physical_binding(
        self, item: Mapping[str, Any], entity: Mapping[str, Any], dormant: bool
    ) -> None:
        if entity.get("id") != item["entity"]:
            raise ValueError("material entity and dormant template identities differ")
        allowed_mobility = {"free"} if dormant else {"free", "static", "hinge", "slide"}
        if entity.get("mobility") not in allowed_mobility:
            raise ValueError("material entity has unsupported mobility")
        if any(
            component.get("type") in {"food", "scent"}
            for component in entity.get("components", [])
        ):
            raise ValueError(
                "material entities cannot mirror chemistry in food or scent components"
            )
        if item["boundaries"][0]["scale"] != 1.0:
            raise ValueError("first material boundary must use authored scale 1")
        first_material = item["boundaries"][0].get("material")
        if first_material is not None and first_material != entity.get("material"):
            raise ValueError("first material boundary must preserve authored material")
        material_names = set(self.world.spec.get("materials", {}))
        if any(
            boundary.get("material") not in material_names
            for boundary in item["boundaries"]
            if "material" in boundary
        ):
            raise ValueError("material boundary references an unavailable material")
        if dormant and not item["remove_when_empty"]:
            raise ValueError("dormant material slots must deactivate when empty")

    def _normalize_spec(self, value: Any) -> dict[str, Any]:
        if isinstance(value, (str, Path)):
            value = json.loads(Path(value).read_text())
        raw = _mapping(value, "material object specification")
        if set(raw) != {"format", "chemistry_sha256", "max_transfer", "objects"}:
            raise ValueError("invalid material object specification fields")
        if raw["format"] != FORMAT:
            raise ValueError("unsupported material object specification")
        chemistry_sha256 = raw["chemistry_sha256"]
        if (
            not isinstance(chemistry_sha256, str)
            or len(chemistry_sha256) != 64
            or any(character not in "0123456789abcdef" for character in chemistry_sha256)
        ):
            raise ValueError("invalid material chemistry identity")
        objects = raw["objects"]
        if not isinstance(objects, list) or not 1 <= len(objects) <= self.MAX_OBJECTS:
            raise ValueError(f"material objects must contain 1..{self.MAX_OBJECTS} entries")
        normalized: list[dict[str, Any]] = []
        entities: set[str] = set()
        rows: set[int] = set()
        for index, value in enumerate(objects):
            item = _mapping(value, f"material object {index}")
            required_item = {
                "entity", "row", "capacities", "content_weights",
                "remove_when_empty", "boundaries", "surface",
            }
            if not required_item <= set(item) or set(item) - required_item - {
                "dormant_template"
            }:
                raise ValueError("invalid material object fields")
            entity = _identifier(item["entity"], "material entity")
            row = item["row"]
            if entity in entities:
                raise ValueError("material entity bindings must be unique")
            if isinstance(row, bool) or not isinstance(row, int) or row < 0 or row in rows:
                raise ValueError("material rows must be distinct nonnegative integers")
            entities.add(entity)
            rows.add(row)
            capacities = {
                _identifier(str(name), "capacity pool"): _number(
                    amount, "material capacity", 0.0, 1e12
                )
                for name, amount in _mapping(item["capacities"], "material capacities").items()
            }
            weights = {
                _identifier(str(name), "content pool"): _number(
                    amount, "content weight", 0.0, 1e9
                )
                for name, amount in _mapping(item["content_weights"], "content weights").items()
            }
            if not weights or not any(amount > 0 for amount in weights.values()):
                raise ValueError("material content needs at least one positive pool weight")
            boundaries = item["boundaries"]
            if not isinstance(boundaries, list) or not 1 <= len(boundaries) <= self.MAX_BOUNDARIES:
                raise ValueError(f"material boundaries must contain 1..{self.MAX_BOUNDARIES} entries")
            normalized_boundaries = []
            previous = math.inf
            for boundary in boundaries:
                boundary = _mapping(boundary, "material boundary")
                if set(boundary) - {"minimum_content", "scale", "material"} or not {
                    "minimum_content", "scale"
                } <= set(boundary):
                    raise ValueError("invalid material boundary fields")
                minimum = _number(boundary["minimum_content"], "minimum content", 0.0, 1e12)
                if minimum >= previous:
                    raise ValueError("material boundaries must descend by minimum content")
                previous = minimum
                normalized_boundary = {
                    "minimum_content": minimum,
                    "scale": _number(boundary["scale"], "material scale", 0.05, 1.0),
                }
                if "material" in boundary:
                    normalized_boundary["material"] = _identifier(
                        boundary["material"], "boundary material"
                    )
                normalized_boundaries.append(normalized_boundary)
            if normalized_boundaries[-1]["minimum_content"] != 0.0:
                raise ValueError("last material boundary must start at zero content")
            remove_when_empty = item["remove_when_empty"]
            if not isinstance(remove_when_empty, bool):
                raise ValueError("remove_when_empty must be boolean")
            surface = _mapping(item["surface"], "material surface")
            if set(surface) != {"rgb_bias", "rgb_coefficients", "odor_coefficients"}:
                raise ValueError("invalid material surface fields")
            rgb_coefficients = {
                _identifier(str(name), "surface pool"): _vector(
                    coefficient, 3, "RGB coefficient", -10.0, 10.0
                )
                for name, coefficient in _mapping(
                    surface["rgb_coefficients"], "RGB coefficients"
                ).items()
            }
            odor_coefficients = {
                _identifier(str(name), "surface pool"): _vector(
                    coefficient, 3, "odor coefficient", 0.0, 10.0
                )
                for name, coefficient in _mapping(
                    surface["odor_coefficients"], "odor coefficients"
                ).items()
            }
            normalized_item = {
                "entity": entity,
                "row": row,
                "capacities": capacities,
                "content_weights": weights,
                "remove_when_empty": remove_when_empty,
                "boundaries": normalized_boundaries,
                "surface": {
                    "rgb_bias": _vector(surface["rgb_bias"], 3, "RGB bias", 0.0, 1.0),
                    "rgb_coefficients": rgb_coefficients,
                    "odor_coefficients": odor_coefficients,
                },
            }
            if "dormant_template" in item:
                template = _mapping(item["dormant_template"], "dormant material template")
                try:
                    encoded = canonical(template)
                except (TypeError, ValueError) as exc:
                    raise ValueError("dormant material template must be finite JSON") from exc
                if len(encoded) > 100_000:
                    raise ValueError("dormant material template exceeds size bound")
                normalized_item["dormant_template"] = copy.deepcopy(template)
            normalized.append(normalized_item)
        return {
            "format": FORMAT,
            "chemistry_sha256": chemistry_sha256,
            "max_transfer": _number(raw["max_transfer"], "maximum material transfer", 1e-12, 1e9),
            "objects": normalized,
        }

    def _validate_web_bindings(self) -> None:
        web = self._web()
        if web.chemistry.sha256 != self.config["chemistry_sha256"]:
            raise ValueError("material and web chemistry identities differ")
        pool_names = set(web.chemistry.pools)
        reserved: set[int] = set()
        structure_rows: set[int] = set()
        owner = self._web_access if not callable(self._web_access) else None

        def collect_rows(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    if (
                        isinstance(key, str)
                        and key.endswith("_row")
                        and isinstance(nested, int)
                        and not isinstance(nested, bool)
                    ):
                        reserved.add(nested)
                    else:
                        collect_rows(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    collect_rows(nested)

        collect_rows(getattr(owner, "config", []))
        for colony in getattr(owner, "config", []):
            if isinstance(colony, Mapping):
                row = colony.get("structure_row")
                if isinstance(row, int) and not isinstance(row, bool):
                    structure_rows.add(row)
        mobility = getattr(owner, "mobility", None)
        collect_rows(getattr(mobility, "residents", []))
        collect_rows(getattr(mobility, "config", []))
        collect_rows(getattr(owner, "mobiles", []))
        self._structure_rows = structure_rows
        enzyme_activity = web.enzyme_activity
        atp_capacity = web.atp_capacity
        for item in self.config["objects"]:
            row = item["row"]
            if row >= web.count or row in reserved:
                raise ValueError("material row is absent or reserved by the biosphere")
            declared = set(item["capacities"])
            referenced = (
                set(item["content_weights"])
                | set(item["surface"]["rgb_coefficients"])
                | set(item["surface"]["odor_coefficients"])
            )
            if declared - pool_names or referenced - declared:
                raise ValueError("material specification references unavailable pool capacity")
            if item.get("dormant_template") is not None and declared != pool_names:
                raise ValueError("dormant material capacity must cover every chemical pool")
            capacity = np.asarray(
                [item["capacities"].get(name, 0.0) for name in web.chemistry.pools]
            )
            if np.any(web.pools[row] < 0.0) or np.any(web.pools[row] > capacity):
                raise ValueError("founder material exceeds its declared pool capacities")
            if np.any(enzyme_activity[row] != 0.0):
                raise ValueError("material compartments must be chemically inert")
            if web.atp[row] != 0.0 or atp_capacity[row] != 0.0:
                raise ValueError("material compartments cannot hide ATP inventory")

    def _item(self, entity_id: str) -> dict[str, Any]:
        if not isinstance(entity_id, str):
            raise ValueError("material entity must be a string")
        try:
            return self._items_by_entity[entity_id]
        except KeyError as exc:
            raise ValueError("unknown material entity") from exc

    @property
    def donor_rows(self) -> dict[str, int]:
        """Environment-only entity-to-compartment bindings."""
        return self._donor_rows.copy()

    def _content(self, item: Mapping[str, Any], pools: np.ndarray) -> float:
        return float(sum(
            pools[index] * weight
            for index, weight in self._content_terms[item["entity"]]
        ))

    def _empty(self, pools: np.ndarray) -> bool:
        return bool(np.all(pools == 0.0))

    def _boundary(self, item: Mapping[str, Any], pools: np.ndarray) -> int | None:
        if item["remove_when_empty"] and self._empty(pools):
            return None
        content = self._content(item, pools)
        for index, boundary in enumerate(item["boundaries"]):
            if content >= boundary["minimum_content"]:
                return index
        raise RuntimeError("zero material boundary is unreachable")

    @staticmethod
    def _scaled_entity(base: Mapping[str, Any], boundary: Mapping[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(dict(base))
        scale = boundary["scale"]
        material = boundary.get("material")
        if material is not None:
            result["material"] = material
        for shape in result["shapes"]:
            shape["size"] = [float(value) * scale for value in shape["size"]]
            if "position" in shape:
                shape["position"] = [float(value) * scale for value in shape["position"]]
            if "fromto" in shape:
                shape["fromto"] = [float(value) * scale for value in shape["fromto"]]
            if material is not None:
                shape["material"] = material
        return result

    def _expected_entity(self, entity_id: str, boundary_index: int) -> dict[str, Any]:
        key = (entity_id, boundary_index)
        expected = self._expected_cache.get(key)
        if expected is None:
            item = self._item(entity_id)
            expected = self._scaled_entity(
                self._base_entities[entity_id], item["boundaries"][boundary_index]
            )
            self._expected_cache[key] = expected
        return expected

    def _deposit_position(self, value: Any) -> list[float]:
        position = _vector(value, 3, "material deposit position", -1e5, 1e5)
        if not (
            0.0 <= position[0] <= float(self.world.width)
            and 0.0 <= position[1] <= float(self.world.height)
            and 0.0 <= position[2] <= float(self.world.depth)
        ):
            raise ValueError("material deposit position is outside habitat")
        return position

    def _existing_ids(self) -> set[str]:
        return set(self.world._entity_mj)

    def _check_physical_state(self) -> None:
        revision = int(self.world.model_revision)
        if self._physical_check_revision == revision:
            return
        existing = self._existing_ids()
        for entity_id, state in self._state.items():
            if state["active"]:
                if entity_id not in existing or self._world_entity(entity_id) != self._expected_entity(
                    entity_id, state["boundary"]
                ):
                    raise ValueError("material state and physical geometry differ")
            elif entity_id in existing:
                raise ValueError("exhausted material entity remains physical")
        self._physical_check_revision = revision

    def contact_entities(self, resident_id: str, contact_samples: Any) -> tuple[str, ...]:
        """Return material entities physically touched by one resident this step."""
        _identifier(resident_id, "resident id")
        if not isinstance(contact_samples, list) or len(contact_samples) > 4096:
            raise ValueError("contact samples must be a bounded list")
        material_ids = set(self.donor_rows)
        touched: set[str] = set()
        for sample in contact_samples:
            sample = _mapping(sample, "contact sample")
            participants = sample.get("participant_resident_ids")
            entities = sample.get("entity_ids")
            if (
                not isinstance(participants, list)
                or not all(isinstance(value, str) for value in participants)
                or not isinstance(entities, list)
                or len(entities) != 2
                or not all(value is None or isinstance(value, str) for value in entities)
            ):
                raise ValueError("invalid physical contact sample")
            if resident_id in participants:
                touched.update(value for value in entities if value in material_ids)
        return tuple(sorted(touched))

    def acquisition_proposals(
        self,
        resident_id: str,
        receiver_row: int,
        contact_samples: Any,
        per_pool_limits: Mapping[str, float] | None = None,
        *,
        maximum_mass: float | None = None,
        receiver_capacity: float | None = None,
    ) -> list[dict[str, Any]]:
        """Prepare exact withdrawals only from entities in physical contact.

        Physiology normally supplies ``maximum_mass`` (credited bite mass) and
        ``receiver_capacity`` (free gut mass).  The proposal preserves the
        source mixture.  ``per_pool_limits`` remains available to engineered
        transfer machinery that has already computed an exact pool vector.
        """
        if per_pool_limits is not None:
            if maximum_mass is not None or receiver_capacity is not None:
                raise ValueError("choose pool limits or a physiological mass limit")
            limits = self._resource_vector(per_pool_limits, "acquisition limits")
            if limits.sum() > self.config["max_transfer"]:
                raise ValueError("acquisition limits exceed maximum material transfer")
            mass_budget = None
        else:
            if maximum_mass is None or receiver_capacity is None:
                raise ValueError("physiological acquisition needs bite and receiver capacity")
            mass_budget = min(
                _number(maximum_mass, "maximum acquisition mass", 0.0, 1e12),
                _number(receiver_capacity, "receiver free capacity", 0.0, 1e12),
            )
            if mass_budget <= 0.0:
                return []
            limits = None
        result = []
        web = self._web()
        mass_weights = web.chemistry._arrays[1].sum(axis=1)
        for entity_id in self.contact_entities(resident_id, contact_samples):
            row = self._item(entity_id)["row"]
            available = web.pools[row]
            if limits is not None:
                request = np.minimum(available, limits)
            else:
                available_mass = float(available @ mass_weights)
                if available_mass <= 0.0 or mass_budget <= 0.0:
                    continue
                fraction = min(1.0, mass_budget / available_mass)
                request = available.copy() if fraction == 1.0 else available * fraction
                if request.sum() > self.config["max_transfer"]:
                    request *= self.config["max_transfer"] / request.sum()
                mass_budget -= float(request @ mass_weights)
            if np.any(request > 0.0):
                proposal = self._prepare("withdraw", entity_id, receiver_row, request)
                proposal["contact_resident"] = resident_id
                proposal["token"] = self._proposal_token(proposal)
                result.append(proposal)
        return result

    def _resource_vector(self, resources: Mapping[str, float], name: str) -> np.ndarray:
        if not isinstance(resources, Mapping):
            raise ValueError(f"{name} must be a mapping")
        vector = self._web().chemistry.resources(resources)
        if not np.any(vector > 0.0):
            raise ValueError(f"{name} must request positive material")
        return vector

    def prepare_withdraw(
        self, entity_id: str, receiver_row: int, resources: Mapping[str, float]
    ) -> dict[str, Any]:
        return self._prepare("withdraw", entity_id, receiver_row, self._resource_vector(resources, "withdrawal"))

    def prepare_deposit(
        self, entity_id: str, donor_row: int, resources: Mapping[str, float]
    ) -> dict[str, Any]:
        return self._prepare("deposit", entity_id, donor_row, self._resource_vector(resources, "deposit"))

    def _row(self, value: Any, name: str) -> int:
        web = self._web()
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < web.count:
            raise ValueError(f"{name} is invalid")
        return value

    def _prepare(
        self, direction: str, entity_id: str, other_row: int, resources: np.ndarray
    ) -> dict[str, Any]:
        item = self._item(entity_id)
        state = self._state[entity_id]
        if not state["active"] and direction != "deposit":
            raise ValueError("material entity is not physically present")
        if (
            not state["active"]
            and direction == "deposit"
            and item.get("dormant_template") is not None
        ):
            raise ValueError("dormant material activation requires positioned deposit_batch")
        web = self._web()
        other_row = self._row(other_row, "transfer compartment")
        object_row = item["row"]
        if other_row == object_row:
            raise ValueError("material transfer endpoints must differ")
        if resources.shape != (len(web.chemistry.pools),) or resources.sum() > self.config["max_transfer"]:
            raise ValueError("material request exceeds transfer bound")
        before_object = web.pools[object_row].copy()
        before_other = web.pools[other_row].copy()
        capacity = np.asarray(
            [item["capacities"].get(name, 0.0) for name in web.chemistry.pools]
        )
        if direction == "withdraw":
            if np.any(resources > before_object):
                raise ValueError("withdrawal exceeds material availability")
            after_object = before_object - resources
        elif direction == "deposit":
            if np.any(resources > before_other):
                raise ValueError("deposit exceeds donor availability")
            after_object = before_object + resources
            if np.any(after_object > capacity):
                raise ValueError("deposit exceeds material capacity")
        else:
            raise ValueError("invalid material transfer direction")
        after_boundary = self._boundary(item, after_object)
        proposal = {
            "format": PROPOSAL_FORMAT,
            "direction": direction,
            "entity": entity_id,
            "object_row": object_row,
            "other_row": other_row,
            "resources": {
                name: float(resources[index])
                for index, name in enumerate(web.chemistry.pools)
                if resources[index] > 0.0
            },
            "object_before": before_object.tolist(),
            "object_after": after_object.tolist(),
            "other_before": before_other.tolist(),
            "boundary_before": state["boundary"],
            "boundary_after": after_boundary,
            "world_revision": int(self.world.model_revision),
            "config_sha256": self.config_sha256,
        }
        proposal["token"] = self._proposal_token(proposal)
        return proposal

    @staticmethod
    def _proposal_token(proposal: Mapping[str, Any]) -> str:
        value = {key: copy.deepcopy(item) for key, item in proposal.items() if key != "token"}
        return hashlib.sha256(canonical(value)).hexdigest()

    def _topology_operation(
        self, entity_id: str, before: int | None, after: int | None
    ) -> dict[str, Any] | None:
        if before == after:
            return None
        if after is None:
            return {"op": "remove", "id": entity_id}
        entity = self._expected_entity(entity_id, after)
        if before is None:
            return {"op": "add", "entity": entity}
        return {"op": "replace", "id": entity_id, "entity": entity}

    def _physical_mass(self, entity_id: str) -> float:
        if entity_id not in self._existing_ids():
            return 0.0
        body_id = self.world._entity_mj[entity_id]
        return float(self.world.model.body_mass[body_id])

    @staticmethod
    def _restore_native(web: MetabolicWeb, snapshot: Mapping[str, Any]) -> None:
        web._native.restore(base64.b64decode(snapshot["native_base64"], validate=True))

    def commit(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        """Commit one still-current transfer and its prepared physical boundary."""
        value = _mapping(proposal, "material proposal")
        allowed = {
            "format", "direction", "entity", "object_row", "other_row", "resources",
            "object_before", "object_after", "other_before", "boundary_before",
            "boundary_after", "world_revision", "config_sha256", "token",
            "contact_resident",
        }
        required = allowed - {"contact_resident"}
        if not required <= set(value) <= allowed:
            raise ValueError("invalid material proposal fields")
        if value["format"] != PROPOSAL_FORMAT or value["config_sha256"] != self.config_sha256:
            raise ValueError("material proposal identity differs")
        if value["world_revision"] != self.world.model_revision:
            raise RuntimeError("material proposal is stale after physical topology change")
        request = self._resource_vector(value["resources"], "proposal resources")
        expected = self._prepare(value["direction"], value["entity"], value["other_row"], request)
        if "contact_resident" in value:
            expected["contact_resident"] = _identifier(value["contact_resident"], "contact resident")
            expected["token"] = self._proposal_token(expected)
        if expected != value:
            raise RuntimeError("material proposal is stale or altered")

        entity_id = value["entity"]
        operation = self._topology_operation(
            entity_id, value["boundary_before"], value["boundary_after"]
        )
        transaction = self.world.prepare_topology_batch([operation]) if operation else None
        web = self._web()
        web_before = web.snapshot()
        totals_before = web.totals()
        mass_before = self._physical_mass(entity_id)
        try:
            if value["direction"] == "withdraw":
                web.transfer(value["object_row"], value["other_row"], value["resources"])
            else:
                web.transfer(value["other_row"], value["object_row"], value["resources"])
            totals_after = web.totals()
            elemental_residual = {
                name: totals_after["elements"][name] - amount
                for name, amount in totals_before["elements"].items()
            }
            energy_residual = totals_after["stored_energy"] - totals_before["stored_energy"]
            if max(map(abs, elemental_residual.values()), default=0.0) > 1e-10 or abs(
                energy_residual
            ) > 1e-10:
                raise RuntimeError("native material transfer violated chemical conservation")
            if transaction is not None:
                transaction.commit()
        except Exception:
            self._restore_native(web, web_before)
            raise
        self._state[entity_id] = {
            "active": value["boundary_after"] is not None,
            "boundary": value["boundary_after"],
        }
        vector = web.chemistry.resources(value["resources"])
        ledger = (
            self.cumulative_withdrawn
            if value["direction"] == "withdraw"
            else self.cumulative_deposited
        )
        updated = np.asarray(ledger, dtype=np.float64) + vector
        if value["direction"] == "withdraw":
            self.cumulative_withdrawn = updated.tolist()
        else:
            self.cumulative_deposited = updated.tolist()
        self.transfer_count += 1
        receipt = {
            "token": value["token"],
            "direction": value["direction"],
            "entity": entity_id,
            "resources": copy.deepcopy(value["resources"]),
            "boundary_before": value["boundary_before"],
            "boundary_after": value["boundary_after"],
            "physical_mass_before": mass_before,
            "physical_mass_after": self._physical_mass(entity_id),
            "model_revision": int(self.world.model_revision),
            "elemental_residual": elemental_residual,
            "stored_energy_residual": energy_residual,
        }
        self.last_receipt = copy.deepcopy(receipt)
        self._mark_inventory_current()
        self._check_physical_state()
        return receipt

    def withdraw_batch(self, requests: Any) -> dict[str, Any]:
        """Fairly execute simultaneous, already-authorized withdrawals.

        Contact and mouth checks belong to resident physiology. Native batch
        transfer computes all donor scarcity factors from the same pre-state,
        so request order cannot decide which receiver gets a shared object's
        last material.
        """
        if not isinstance(requests, list) or len(requests) > self.MAX_BATCH:
            raise ValueError(f"material batch must contain at most {self.MAX_BATCH} requests")
        web = self._web()
        if not requests:
            return {
                "direction": "withdraw_batch", "requests": [],
                "pools": list(web.chemistry.pools), "moved_resources": [],
                "resource_limiter": [], "changes": [],
                "model_revision": int(self.world.model_revision),
                "elemental_residual": {
                    name: 0.0 for name in web.chemistry.elements
                },
                "stored_energy_residual": 0.0,
            }
        material_rows = set(self.donor_rows.values())
        normalized = []
        donors = []
        receivers = []
        for index, raw in enumerate(requests):
            request = _mapping(raw, f"material request {index}")
            if set(request) != {"entity", "receiver_row", "resources"}:
                raise ValueError("invalid material withdrawal request fields")
            entity_id = _identifier(request["entity"], "material entity")
            item = self._item(entity_id)
            if not self._state[entity_id]["active"]:
                raise ValueError("material entity is not physically present")
            receiver = self._row(request["receiver_row"], "receiver compartment")
            if receiver in material_rows:
                raise ValueError("withdrawal receiver must not be another material object")
            vector = self._resource_vector(request["resources"], "batch withdrawal")
            if vector.sum() > self.config["max_transfer"]:
                raise ValueError("batch request exceeds maximum material transfer")
            resources = {
                name: float(vector[pool])
                for pool, name in enumerate(web.chemistry.pools)
                if vector[pool] > 0.0
            }
            normalized.append({
                "entity": entity_id,
                "receiver_row": receiver,
                "resources": resources,
            })
            donors.append(item["row"])
            receivers.append(receiver)

        # Derive scarcity allocation and final physical boundaries without
        # mutating either authoritative owner.
        stage = MetabolicWeb.restore(web.snapshot())
        staged = stage.transfer_batch(
            donors, receivers, [item["resources"] for item in normalized],
            [0.0] * len(normalized),
        )
        moved = np.asarray(staged["moved_resources"], dtype=np.float64)
        if moved.shape != (len(normalized), len(web.chemistry.pools)):
            raise RuntimeError("native material batch returned an invalid shape")

        changes = []
        operations = []
        mass_before = {}
        requested_entities = set(item["entity"] for item in normalized)
        for item in self.config["objects"]:
            entity_id = item["entity"]
            if entity_id not in requested_entities:
                continue
            state = self._state[entity_id]
            boundary = self._boundary(item, stage.pools[item["row"]])
            operation = self._topology_operation(entity_id, state["boundary"], boundary)
            if operation is not None:
                operations.append(operation)
                mass_before[entity_id] = self._physical_mass(entity_id)
                changes.append({
                    "entity": entity_id,
                    "boundary_before": state["boundary"],
                    "boundary_after": boundary,
                })
        transaction = self.world.prepare_topology_batch(operations) if operations else None

        before = web.snapshot()
        totals_before = web.totals()
        try:
            applied = web.transfer_batch(
                donors, receivers, [item["resources"] for item in normalized],
                [0.0] * len(normalized),
            )
            applied_moved = np.asarray(applied["moved_resources"], dtype=np.float64)
            if not np.array_equal(applied_moved, moved):
                raise RuntimeError("staged and authoritative material allocations differ")
            totals_after = web.totals()
            elemental_residual = {
                name: totals_after["elements"][name] - amount
                for name, amount in totals_before["elements"].items()
            }
            energy_residual = totals_after["stored_energy"] - totals_before["stored_energy"]
            if max(map(abs, elemental_residual.values()), default=0.0) > 1e-10 or abs(
                energy_residual
            ) > 1e-10:
                raise RuntimeError("native material batch violated chemical conservation")
            if transaction is not None:
                transaction.commit()
        except Exception:
            self._restore_native(web, before)
            raise

        for change in changes:
            self._state[change["entity"]] = {
                "active": change["boundary_after"] is not None,
                "boundary": change["boundary_after"],
            }
            change["physical_mass_before"] = mass_before[change["entity"]]
            change["physical_mass_after"] = self._physical_mass(change["entity"])
        moved_total = moved.sum(axis=0)
        self.cumulative_withdrawn = (
            np.asarray(self.cumulative_withdrawn, dtype=np.float64) + moved_total
        ).tolist()
        moved_count = int(np.count_nonzero(np.any(moved > 0.0, axis=1)))
        self.transfer_count += moved_count
        receipt = {
            "direction": "withdraw_batch",
            "requests": copy.deepcopy(normalized),
            "pools": list(web.chemistry.pools),
            "moved_resources": moved.tolist(),
            "resource_limiter": np.asarray(
                applied["resource_limiter"], dtype=np.float64
            ).tolist(),
            "changes": changes,
            "model_revision": int(self.world.model_revision),
            "elemental_residual": elemental_residual,
            "stored_energy_residual": energy_residual,
        }
        self.last_receipt = copy.deepcopy(receipt)
        self._mark_inventory_current()
        self._check_physical_state()
        return receipt

    def deposit_batch(self, requests: Any) -> dict[str, Any]:
        """Fairly deposit finite chemistry into active objects or dormant slots.

        Donor access and egestion budgets are established by the caller. An
        inactive opt-in slot requires a world position; active objects reject
        positions and retain their physical trajectory.
        """
        if not isinstance(requests, list) or len(requests) > self.MAX_BATCH:
            raise ValueError(f"material batch must contain at most {self.MAX_BATCH} requests")
        web = self._web()
        if not requests:
            return {
                "direction": "deposit_batch", "requests": [],
                "pools": list(web.chemistry.pools), "moved_resources": [],
                "blocked_resources": [], "capacity_blocked_resources": [],
                "donor_blocked_resources": [], "receiver_limiter": [],
                "resource_limiter": [], "changes": [],
                "model_revision": int(self.world.model_revision),
                "elemental_residual": {
                    name: 0.0 for name in web.chemistry.elements
                },
                "stored_energy_residual": 0.0,
            }

        material_rows = set(self.donor_rows.values())
        normalized = []
        requested = []
        donors = []
        receivers = []
        spawn_positions: dict[str, list[float]] = {}
        for index, raw in enumerate(requests):
            request = _mapping(raw, f"material deposit request {index}")
            required = {"entity", "donor_row", "resources"}
            if not required <= set(request) or set(request) - required - {"position"}:
                raise ValueError("invalid material deposit request fields")
            entity_id = _identifier(request["entity"], "material entity")
            item = self._item(entity_id)
            state = self._state[entity_id]
            donor = self._row(request["donor_row"], "material donor compartment")
            if donor in material_rows or donor in self._structure_rows:
                raise ValueError("material deposits cannot debit material or structure rows")
            vector = self._resource_vector(request["resources"], "batch deposit")
            if vector.sum() > self.config["max_transfer"]:
                raise ValueError("batch request exceeds maximum material transfer")
            if state["active"]:
                if "position" in request:
                    raise ValueError("deposit cannot reposition an active material object")
            else:
                if item.get("dormant_template") is None or "position" not in request:
                    raise ValueError("inactive dormant deposit requires a spawn position")
                position = self._deposit_position(request["position"])
                previous = spawn_positions.setdefault(entity_id, position)
                if previous != position:
                    raise ValueError("one dormant slot cannot receive different spawn positions")
            resources = {
                name: float(vector[pool])
                for pool, name in enumerate(web.chemistry.pools)
                if vector[pool] > 0.0
            }
            entry = {"entity": entity_id, "donor_row": donor, "resources": resources}
            if "position" in request:
                entry["position"] = position
            normalized.append(entry)
            requested.append(vector)
            donors.append(donor)
            receivers.append(item["row"])

        requested_array = np.asarray(requested, dtype=np.float64)
        receiver_limiter = np.ones_like(requested_array)
        for item in self.config["objects"]:
            row = item["row"]
            edges = [edge for edge, receiver in enumerate(receivers) if receiver == row]
            if not edges:
                continue
            demand = requested_array[edges].sum(axis=0)
            capacity = np.asarray(
                [item["capacities"].get(name, 0.0) for name in web.chemistry.pools],
                dtype=np.float64,
            )
            free = np.maximum(0.0, capacity - web.pools[row])
            if np.any(web.pools[row] > capacity):
                raise ValueError("material inventory exceeds declared capacity")
            factors = np.ones_like(free)
            positive = demand > 0.0
            factors[positive] = np.minimum(1.0, free[positive] / demand[positive])
            limited = positive & (factors < 1.0) & (factors > 0.0)
            factors[limited] = np.nextafter(factors[limited], 0.0)
            receiver_limiter[edges] = factors
        capacity_limited = requested_array * receiver_limiter
        effective = [
            {
                name: float(row[pool])
                for pool, name in enumerate(web.chemistry.pools)
                if row[pool] > 0.0
            }
            for row in capacity_limited
        ]

        stage = MetabolicWeb.restore(web.snapshot())
        staged = stage.transfer_batch(
            donors, receivers, effective, [0.0] * len(normalized)
        )
        moved = np.asarray(staged["moved_resources"], dtype=np.float64)
        if moved.shape != requested_array.shape:
            raise RuntimeError("native material deposit returned an invalid shape")
        for item in self.config["objects"]:
            capacity = np.asarray(
                [item["capacities"].get(name, 0.0) for name in web.chemistry.pools]
            )
            if np.any(stage.pools[item["row"]] > capacity):
                raise RuntimeError("staged material deposit exceeded receiver capacity")

        changes = []
        operations = []
        mass_before = {}
        new_bases: dict[str, dict[str, Any]] = {}
        requested_entities = set(item["entity"] for item in normalized)
        for item in self.config["objects"]:
            entity_id = item["entity"]
            if entity_id not in requested_entities:
                continue
            state = self._state[entity_id]
            boundary = self._boundary(item, stage.pools[item["row"]])
            if boundary == state["boundary"]:
                continue
            mass_before[entity_id] = self._physical_mass(entity_id)
            if state["boundary"] is None and boundary is not None:
                base = copy.deepcopy(self._base_entities[entity_id])
                base["position"] = spawn_positions[entity_id]
                new_bases[entity_id] = base
                operations.append({
                    "op": "add",
                    "entity": self._scaled_entity(base, item["boundaries"][boundary]),
                })
            else:
                operation = self._topology_operation(
                    entity_id, state["boundary"], boundary
                )
                if operation is not None:
                    operations.append(operation)
            change = {
                "entity": entity_id,
                "boundary_before": state["boundary"],
                "boundary_after": boundary,
            }
            if entity_id in new_bases:
                change["spawn_position"] = spawn_positions[entity_id]
            changes.append(change)
        transaction = self.world.prepare_topology_batch(operations) if operations else None

        before = web.snapshot()
        totals_before = web.totals()
        try:
            applied = web.transfer_batch(
                donors, receivers, effective, [0.0] * len(normalized)
            )
            applied_moved = np.asarray(applied["moved_resources"], dtype=np.float64)
            if not np.array_equal(applied_moved, moved):
                raise RuntimeError("staged and authoritative material deposits differ")
            totals_after = web.totals()
            elemental_residual = {
                name: totals_after["elements"][name] - amount
                for name, amount in totals_before["elements"].items()
            }
            energy_residual = totals_after["stored_energy"] - totals_before["stored_energy"]
            if max(map(abs, elemental_residual.values()), default=0.0) > 1e-10 or abs(
                energy_residual
            ) > 1e-10:
                raise RuntimeError("native material deposit violated chemical conservation")
            if transaction is not None:
                transaction.commit()
        except Exception:
            self._restore_native(web, before)
            raise

        for change in changes:
            entity_id = change["entity"]
            if entity_id in new_bases:
                self._base_entities[entity_id] = new_bases[entity_id]
                self._expected_cache.clear()
            self._state[entity_id] = {
                "active": change["boundary_after"] is not None,
                "boundary": change["boundary_after"],
            }
            change["physical_mass_before"] = mass_before[entity_id]
            change["physical_mass_after"] = self._physical_mass(entity_id)
        moved_total = moved.sum(axis=0)
        self.cumulative_deposited = (
            np.asarray(self.cumulative_deposited, dtype=np.float64) + moved_total
        ).tolist()
        self.transfer_count += int(np.count_nonzero(np.any(moved > 0.0, axis=1)))
        capacity_blocked = requested_array - capacity_limited
        donor_blocked = capacity_limited - moved
        receipt = {
            "direction": "deposit_batch",
            "requests": copy.deepcopy(normalized),
            "pools": list(web.chemistry.pools),
            "moved_resources": moved.tolist(),
            "blocked_resources": (requested_array - moved).tolist(),
            "capacity_blocked_resources": capacity_blocked.tolist(),
            "donor_blocked_resources": donor_blocked.tolist(),
            "receiver_limiter": receiver_limiter.tolist(),
            "resource_limiter": np.asarray(
                applied["resource_limiter"], dtype=np.float64
            ).tolist(),
            "changes": changes,
            "model_revision": int(self.world.model_revision),
            "elemental_residual": elemental_residual,
            "stored_energy_residual": energy_residual,
        }
        self.last_receipt = copy.deepcopy(receipt)
        self._mark_inventory_current()
        self._check_physical_state()
        return receipt

    def sync_geometry(self) -> list[dict[str, Any]]:
        """Apply boundary changes caused by other operations on the shared web."""
        web = self._web()
        material_pools = np.asarray(web.pools[self._row_indices])
        if np.any(material_pools > self._capacity_matrix):
            raise ValueError("material inventory exceeds declared capacity")
        if np.array_equal(material_pools, self._inventory_cache):
            self._check_physical_state()
            return []
        operations = []
        changes = []
        for index, item in enumerate(self.config["objects"]):
            entity_id = item["entity"]
            state = self._state[entity_id]
            boundary = self._boundary(item, material_pools[index])
            if (
                state["boundary"] is None
                and boundary is not None
                and item.get("dormant_template") is not None
            ):
                raise ValueError("dormant material activation requires positioned deposit_batch")
            operation = self._topology_operation(entity_id, state["boundary"], boundary)
            if operation is not None:
                operations.append(operation)
                changes.append({
                    "entity": entity_id,
                    "boundary_before": state["boundary"],
                    "boundary_after": boundary,
                })
        if not operations:
            self._inventory_cache = material_pools.copy()
            self._check_physical_state()
            return []
        transaction = self.world.prepare_topology_batch(operations)
        transaction.commit()
        for change in changes:
            self._state[change["entity"]] = {
                "active": change["boundary_after"] is not None,
                "boundary": change["boundary_after"],
            }
        self.geometry_syncs += 1
        self.last_geometry_sync = copy.deepcopy(changes)
        self._inventory_cache = material_pools.copy()
        self._check_physical_state()
        return changes

    def surface_cues(self) -> list[dict[str, Any]]:
        """Return chemistry-derived cues for environment transduction.

        The machinery-side entity id is used to place the cue. Controllers must
        receive only the resulting ray RGB and local odor samples.
        """
        web = self._web()
        material_pools = np.asarray(web.pools[self._row_indices])
        if (
            self._surface_cache is not None
            and self._surface_inventory_cache is not None
            and np.array_equal(material_pools, self._surface_inventory_cache)
        ):
            return [
                {
                    "entity": cue["entity"],
                    "rgb": cue["rgb"].copy(),
                    "odor": cue["odor"].copy(),
                }
                for cue in self._surface_cache
            ]
        result = []
        for index, item in enumerate(self.config["objects"]):
            if not self._state[item["entity"]]["active"]:
                continue
            pools = material_pools[index]
            surface = item["surface"]
            rgb = np.asarray(surface["rgb_bias"], dtype=np.float64)
            odor = np.zeros(3, dtype=np.float64)
            rgb_terms, odor_terms = self._surface_terms[item["entity"]]
            for pool, coefficient in rgb_terms:
                rgb += pools[pool] * coefficient
            for pool, coefficient in odor_terms:
                odor += pools[pool] * coefficient
            result.append({
                "entity": item["entity"],
                "rgb": np.clip(rgb, 0.0, 1.0).tolist(),
                "odor": np.clip(odor, 0.0, 4.0).tolist(),
            })
        self._surface_inventory_cache = material_pools.copy()
        self._surface_cache = copy.deepcopy(result)
        return result

    def snapshot(self) -> dict[str, Any]:
        """Snapshot bindings and physical derivation state, never web inventory."""
        base_entities = copy.deepcopy(self._base_entities)
        return {
            "format": FORMAT,
            "config": copy.deepcopy(self.config),
            "config_sha256": self.config_sha256,
            "chemistry_sha256": self._web().chemistry.sha256,
            "base_entities": base_entities,
            "base_entities_sha256": hashlib.sha256(canonical(base_entities)).hexdigest(),
            "state": copy.deepcopy(self._state),
            "transfer_count": self.transfer_count,
            "cumulative_withdrawn": self.cumulative_withdrawn.copy(),
            "cumulative_deposited": self.cumulative_deposited.copy(),
            "geometry_syncs": self.geometry_syncs,
            "last_receipt": copy.deepcopy(self.last_receipt),
            "last_geometry_sync": copy.deepcopy(self.last_geometry_sync),
        }

    @classmethod
    def restore(
        cls, world: Any, web_access: Any, snapshot: Mapping[str, Any]
    ) -> "MaterialObjects":
        value = _mapping(snapshot, "material object snapshot")
        if set(value) != {
            "format", "config", "config_sha256", "chemistry_sha256",
            "base_entities", "base_entities_sha256", "state", "transfer_count",
            "cumulative_withdrawn", "cumulative_deposited", "geometry_syncs",
            "last_receipt", "last_geometry_sync",
        }:
            raise ValueError("invalid material object snapshot fields")
        if value.get("format") != FORMAT:
            raise ValueError("unsupported material object snapshot")
        instance = cls.__new__(cls)
        instance.world = world
        instance._web_access = web_access
        instance.config = instance._normalize_spec(value["config"])
        instance.config_sha256 = hashlib.sha256(canonical(instance.config)).hexdigest()
        if (
            instance.config_sha256 != value.get("config_sha256")
            or value.get("chemistry_sha256") != instance._web().chemistry.sha256
        ):
            raise ValueError("material snapshot identity differs")
        instance._validate_web_bindings()
        instance._prepare_runtime_cache()
        base_entities = value.get("base_entities")
        state = value.get("state")
        ids = {item["entity"] for item in instance.config["objects"]}
        if not isinstance(base_entities, dict) or set(base_entities) != ids:
            raise ValueError("material base entity identities differ")
        if hashlib.sha256(canonical(base_entities)).hexdigest() != value.get(
            "base_entities_sha256"
        ):
            raise ValueError("material base entity checksum differs")
        if not isinstance(state, dict) or set(state) != ids:
            raise ValueError("material state identities differ")
        instance._base_entities = copy.deepcopy(base_entities)
        instance._state = {}
        for item in instance.config["objects"]:
            entity_id = item["entity"]
            entry = _mapping(state[entity_id], "material state")
            if set(entry) != {"active", "boundary"} or not isinstance(entry["active"], bool):
                raise ValueError("invalid material state")
            boundary = entry["boundary"]
            if boundary is not None and (
                isinstance(boundary, bool)
                or not isinstance(boundary, int)
                or not 0 <= boundary < len(item["boundaries"])
            ):
                raise ValueError("invalid material boundary state")
            derived = instance._boundary(item, instance._web().pools[item["row"]])
            if entry["active"] != (boundary is not None) or boundary != derived:
                raise ValueError("material state differs from web inventory")
            instance._state[entity_id] = copy.deepcopy(entry)
        count = value.get("transfer_count")
        syncs = value.get("geometry_syncs")
        if any(
            isinstance(number, bool) or not isinstance(number, int) or number < 0
            for number in (count, syncs)
        ):
            raise ValueError("invalid material event counters")
        instance.transfer_count = count
        instance.geometry_syncs = syncs
        for key in ("cumulative_withdrawn", "cumulative_deposited"):
            vector = np.asarray(value.get(key), dtype=np.float64)
            if (
                vector.shape != (len(instance._web().chemistry.pools),)
                or not np.isfinite(vector).all()
                or np.any(vector < 0.0)
            ):
                raise ValueError("invalid material transfer ledger")
            setattr(instance, key, vector.tolist())
        last_receipt = value.get("last_receipt")
        if last_receipt is not None and not isinstance(last_receipt, dict):
            raise ValueError("invalid last material receipt")
        last_sync = value.get("last_geometry_sync")
        if not isinstance(last_sync, list) or len(last_sync) > cls.MAX_OBJECTS:
            raise ValueError("invalid last material geometry sync")
        instance.last_receipt = copy.deepcopy(last_receipt)
        instance.last_geometry_sync = copy.deepcopy(last_sync)
        try:
            if len(canonical({
                "last_receipt": instance.last_receipt,
                "last_geometry_sync": instance.last_geometry_sync,
            })) > 1_000_000:
                raise ValueError("material diagnostics exceed snapshot bound")
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid material snapshot diagnostics") from exc
        inactive = [
            {"op": "add", "entity": copy.deepcopy(instance._base_entities[entity_id])}
            for entity_id, entry in instance._state.items()
            if not entry["active"]
        ]
        if inactive:
            # Compilation validates dormant geometry needed by a future deposit
            # without adopting the candidate or changing the restored world.
            instance.world.prepare_topology_batch(inactive)
        instance._check_physical_state()
        return instance


__all__ = ["MaterialObjects"]
