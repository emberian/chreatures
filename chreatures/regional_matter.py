"""Thin host boundary for finite material flow through physical regions.

All pool quantities remain in the owning :class:`MetabolicWeb`. The native
kernel computes same-prestate flow proposals and owns only clocks and ledgers;
this adapter applies those proposals through existing authoritative transfer
and physical material transactions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .metabolism import MetabolicWeb, canonical
from .native_world import load_world_kernels
from .evidence_events import _payload as _evidence_payload


FORMAT = "chreatures-regional-matter-v1"
SNAPSHOT_FORMAT = "chreatures-regional-matter-snapshot-v1"
VIEW_FORMAT = "chreatures-regional-matter-view-v1"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return copy.deepcopy(dict(value))


def _resources(names: list[str], values: np.ndarray) -> dict[str, float]:
    return {
        name: float(values[index])
        for index, name in enumerate(names)
        if values[index] > 0.0
    }


def _event(
    kind: str,
    *,
    entities: list[str],
    resources: Mapping[str, float],
    details: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "actors": {"bodies": [], "entities": list(entities)},
        "quantities": [
            {
                "name": name,
                "value": float(amount),
                "unit": "synthetic-chemical-amount",
            }
            for name, amount in resources.items()
            if amount > 0.0
        ],
        "details": copy.deepcopy(dict(details)),
        "source": {
            "stream": "regional-matter",
            "config_sha256": config_sha256,
        },
        "blob_refs": [],
    }


class RegionalMatter:
    """Conservative material routing among ordinary MetabolicWeb rows."""

    def __init__(self, biosphere: Any, config: Any):
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        self.biosphere = biosphere
        self.config = _mapping(config, "regional matter configuration")
        if set(self.config) != {
            "format",
            "chemistry_sha256",
            "world_size_m",
            "regions",
            "routes",
            "exit_faces",
            "outlets",
        } or self.config.get("format") != FORMAT:
            raise ValueError("unsupported regional matter configuration")
        encoded = canonical(self.config)
        self.config_sha256 = hashlib.sha256(encoded).hexdigest()
        native_type = getattr(load_world_kernels(), "RegionalMatter", None)
        if native_type is None:
            raise RuntimeError("native world kernels omit RegionalMatter")
        self._native = native_type(
            encoded.decode(), self.config_sha256, list(self.web.chemistry.pools)
        )
        self._validate_bindings()
        self._revision = -1
        self._carrier_bodies = np.empty(0, dtype=np.int32)
        self.last_events: list[dict[str, Any]] = []
        self._bind_physical()

    @classmethod
    def from_config(cls, biosphere: Any, config: Any) -> "RegionalMatter":
        return cls(biosphere, config)

    @property
    def world(self) -> Any:
        return self.biosphere.world

    @property
    def web(self) -> MetabolicWeb:
        web = self.biosphere.web
        if not isinstance(web, MetabolicWeb):
            raise TypeError("regional matter requires the biosphere MetabolicWeb")
        return web

    @property
    def materials(self) -> Any:
        materials = self.biosphere.materials
        if materials is None:
            raise ValueError("regional matter requires physical material objects")
        return materials

    def _validate_bindings(self) -> None:
        if self.config["chemistry_sha256"] != self.web.chemistry.sha256:
            raise ValueError("regional matter chemistry identity differs")
        expected_size = [
            float(self.world.width),
            float(self.world.height),
            float(self.world.depth),
        ]
        if self.config["world_size_m"] != expected_size:
            raise ValueError("regional matter and physical world dimensions differ")

        rows = [int(value) for value in self._native.region_rows()]
        if len(rows) != len(set(rows)) or any(
            row < 0 or row >= self.web.count for row in rows
        ):
            raise ValueError("regional matter rows are absent or duplicated")
        reserved = set(self.materials.donor_rows.values())
        for colony in self.biosphere.config:
            reserved.update((colony["body_row"], colony["structure_row"]))
        for mobile in self.biosphere._mobile_specs or []:
            reserved.update(
                value
                for key, value in mobile.items()
                if key.endswith("_row") and isinstance(value, int)
            )
        if set(rows) & reserved:
            raise ValueError("regional matter rows must be private from physical owners")
        if (
            np.any(self.web.enzyme_activity[rows] != 0.0)
            or np.any(self.web.atp[rows] != 0.0)
            or np.any(self.web.atp_capacity[rows] != 0.0)
            or any(
                any(
                    float(value) != 0.0
                    for field in ("baseline", "substrate_response", "atp_response")
                    for value in self.web._regulation[row][field].values()
                )
                for row in rows
            )
        ):
            raise ValueError("regional matter rows must be chemically passive")

        pool_names = list(self.web.chemistry.pools)
        for _, row, _, capacities in self._native.metadata():
            capacity = np.asarray(capacities, dtype=np.float64)
            if capacity.shape != (len(pool_names),) or np.any(
                self.web.pools[int(row)] > capacity
            ):
                raise ValueError("regional founder inventory exceeds capacity")

        material_items = {
            item["entity"]: item for item in self.materials.config["objects"]
        }
        maximum_transfer = float(self.materials.config["max_transfer"])
        for outlet in self.config["outlets"]:
            maximum = np.asarray(
                [outlet["maximum_release"][name] for name in pool_names],
                dtype=np.float64,
            )
            if maximum.sum() > maximum_transfer:
                raise ValueError("regional outlet exceeds material transfer bound")
            for slot in outlet["slots"]:
                item = material_items.get(slot)
                if item is None or item.get("dormant_template") is None:
                    raise ValueError("regional outlets require dormant material slots")
                capacity = np.asarray(
                    [item["capacities"].get(name, 0.0) for name in pool_names],
                    dtype=np.float64,
                )
                if np.any(maximum > capacity):
                    raise ValueError("regional outlet exceeds dormant slot capacity")

    def _bind_physical(self) -> None:
        carriers = []
        for entity_id in self._native.route_carriers():
            if entity_id is None:
                carriers.append(-1)
                continue
            if entity_id not in self.world._entity_mj:
                raise ValueError("regional route carrier is absent")
            entity = self.world._entity(entity_id)
            if entity["mobility"] not in {"static", "hinge", "slide"}:
                raise ValueError("regional route carrier has unsupported mobility")
            carriers.append(int(self.world._entity_mj[entity_id]))
        self._carrier_bodies = np.ascontiguousarray(carriers, dtype=np.int32)
        self._revision = int(self.world.model_revision)

    def _ensure_physical(self) -> None:
        if self._revision != int(self.world.model_revision):
            self._bind_physical()

    def before_reactions(self, dt: float) -> list[dict[str, Any]]:
        """Retire complete escaped packets before they can emit field sources."""
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("regional matter interval must be finite and positive")
        self.last_events = []
        entity_ids = self.materials.active_free_entities
        if not entity_ids:
            return []
        positions = np.ascontiguousarray(
            [
                np.asarray(
                    self.world.data.xpos[self.world._entity_mj[entity_id]],
                    dtype=np.float64,
                )
                for entity_id in entity_ids
            ],
            dtype=np.float64,
        )
        faces = np.asarray(self._native.detect_exits(positions), dtype=np.int32)
        metadata = self._native.face_metadata()
        requests = []
        for entity_id, face_index in zip(entity_ids, faces, strict=True):
            if face_index < 0:
                continue
            face_id, region_index, capacities = metadata[int(face_index)]
            row = int(self._native.region_rows()[int(region_index)])
            requests.append({
                "entity": entity_id,
                "receiver_row": row,
                "receiver_capacities": dict(
                    zip(self.web.chemistry.pools, map(float, capacities), strict=True)
                ),
                "face": str(face_id),
                "cause": "declared-world-face",
            })
        if not requests:
            return []
        receipt = self.materials.exit_batch(requests)
        for request, moved in zip(
            receipt["requests"], receipt["moved_resources"], strict=True
        ):
            resources = _resources(
                list(self.web.chemistry.pools), np.asarray(moved, dtype=np.float64)
            )
            self.last_events.append(
                _event(
                    "physical-material-entered-region",
                    entities=[request["entity"]],
                    resources=resources,
                    details={
                        "face": request["face"],
                        "receiver_row": request["receiver_row"],
                    },
                    config_sha256=self.config_sha256,
                )
            )
        return copy.deepcopy(self.last_events)

    def after_reactions(self, dt: float) -> list[dict[str, Any]]:
        """Apply native same-prestate route flows and due physical outlets."""
        if (
            isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not math.isfinite(dt)
            or dt <= 0.0
        ):
            raise ValueError("regional matter interval must be finite and positive")
        self._ensure_physical()
        rows = np.asarray(self._native.region_rows(), dtype=np.int64)
        pools = np.ascontiguousarray(self.web.pools[rows], dtype=np.float64)
        accessibility = np.asarray(
            self._native.route_accessibility(
                int(self.world.model._address),
                int(self.world.data._address),
                self._carrier_bodies,
            ),
            dtype=np.float64,
        )
        outlet_slots = self._native.outlet_slots()
        active = set(self.materials.active_entities)
        selected_slots = [
            next((slot for slot in slots if slot not in active), None)
            for slots in outlet_slots
        ]
        proposal = self._native.propose(
            float(dt), pools, accessibility, [slot is not None for slot in selected_slots]
        )
        token = str(proposal["token"])
        route_source = np.asarray(proposal["route_source"], dtype=np.int64)
        route_target = np.asarray(proposal["route_target"], dtype=np.int64)
        route_resources = np.asarray(proposal["route_resources"], dtype=np.float64)
        outlet_resources = np.asarray(proposal["outlet_resources"], dtype=np.float64)
        pool_names = list(self.web.chemistry.pools)
        before = self.web.snapshot()
        actual_route = np.zeros_like(route_resources)
        route_requests: list[tuple[int, int]] = []
        route_donors: list[int] = []
        route_receivers: list[int] = []
        route_amounts: list[dict[str, float]] = []
        for edge in range(route_resources.shape[0]):
            for pool, name in enumerate(pool_names):
                amount = float(route_resources[edge, pool])
                if amount <= 0.0:
                    continue
                route_requests.append((edge, pool))
                route_donors.append(int(rows[route_source[edge, pool]]))
                route_receivers.append(int(rows[route_target[edge, pool]]))
                route_amounts.append({name: amount})
        try:
            if route_requests:
                route_result = self.web.transfer_batch(
                    route_donors,
                    route_receivers,
                    route_amounts,
                    [0.0] * len(route_requests),
                )
                moved = np.asarray(route_result["moved_resources"], dtype=np.float64)
                for (edge, pool), values in zip(route_requests, moved, strict=True):
                    actual_route[edge, pool] = values[pool]
            if not np.array_equal(actual_route, route_resources):
                raise RuntimeError("authoritative regional route application differs")
        except Exception:
            self.materials._restore_native(self.web, before)
            self.materials._mark_inventory_current()
            self._native.abort(token)
            raise

        outlet_metadata = self._native.outlet_metadata()
        metadata = self._native.metadata()
        requested_outlets = []
        requested_indices = []
        for index, (slot, values) in enumerate(
            zip(selected_slots, outlet_resources, strict=True)
        ):
            if slot is None or not np.any(values > 0.0):
                continue
            _, region_index, _, position, _ = outlet_metadata[index]
            requested_indices.append(index)
            requested_outlets.append({
                "entity": slot,
                "donor_row": int(rows[int(region_index)]),
                "resources": _resources(pool_names, values),
                "position": list(map(float, position)),
            })
        actual_outlet = np.zeros_like(outlet_resources)
        try:
            if requested_outlets:
                outlet_receipt = self.materials.deposit_batch(requested_outlets)
                moved = np.asarray(
                    outlet_receipt["moved_resources"], dtype=np.float64
                )
                for index, values in zip(requested_indices, moved, strict=True):
                    actual_outlet[index] = values
        except Exception:
            self.materials._restore_native(self.web, before)
            self.materials._mark_inventory_current()
            self._native.abort(token)
            raise

        self._native.commit(token, actual_route, actual_outlet)
        route_metadata = self._native.route_metadata()
        for edge, moved in enumerate(actual_route):
            resources = _resources(pool_names, moved)
            if not resources:
                continue
            route_id, a, b, carrier = route_metadata[edge]
            entities = [] if carrier is None else [str(carrier)]
            directions = {
                name: {
                    "source": str(metadata[int(route_source[edge, pool])][0]),
                    "target": str(metadata[int(route_target[edge, pool])][0]),
                }
                for pool, name in enumerate(pool_names)
                if moved[pool] > 0.0
            }
            self.last_events.append(
                _event(
                    "regional-material-flow",
                    entities=entities,
                    resources=resources,
                    details={
                        "route": str(route_id),
                        "accessibility": float(accessibility[edge]),
                        "endpoints": [str(metadata[int(a)][0]), str(metadata[int(b)][0])],
                        "directions": directions,
                    },
                    config_sha256=self.config_sha256,
                )
            )
        for index, moved in enumerate(actual_outlet):
            resources = _resources(pool_names, moved)
            if not resources:
                continue
            outlet_id, region_index, _, position, _ = outlet_metadata[index]
            self.last_events.append(
                _event(
                    "regional-material-outlet",
                    entities=[str(selected_slots[index])],
                    resources=resources,
                    details={
                        "outlet": str(outlet_id),
                        "source_row": int(rows[int(region_index)]),
                        "position": list(map(float, position)),
                    },
                    config_sha256=self.config_sha256,
                )
            )
        return copy.deepcopy(self.last_events)

    def view(self) -> dict[str, Any]:
        state = self._native.state()
        rows = np.asarray(self._native.region_rows(), dtype=np.int64)
        pool_names = list(self.web.chemistry.pools)
        metadata = self._native.metadata()
        nodes = []
        for region_id, row, position, _ in metadata:
            nodes.append({
                "id": str(region_id),
                "row": int(row),
                "position": list(map(float, position)),
                "pools": {
                    name: float(self.web.pools[int(row), pool])
                    for pool, name in enumerate(pool_names)
                },
            })
        route_last = np.asarray(state["last_route"], dtype=np.float64)
        route_total = np.asarray(state["route_cumulative"], dtype=np.float64)
        access = np.asarray(state["last_accessibility"], dtype=np.float64)
        edges = []
        for index, (route_id, a, b, _) in enumerate(self._native.route_metadata()):
            edges.append({
                "id": str(route_id),
                "source": str(metadata[int(a)][0]),
                "target": str(metadata[int(b)][0]),
                "accessibility": float(access[index]),
                "last_moved_resources": _resources(pool_names, route_last[index]),
                "cumulative_moved_resources": _resources(
                    pool_names, route_total[index]
                ),
            })
        outlet_last = np.asarray(state["last_outlet"], dtype=np.float64)
        outlet_total = np.asarray(state["outlet_cumulative"], dtype=np.float64)
        outlet_credit = np.asarray(state["outlet_credit"], dtype=np.float64)
        active = set(self.materials.active_entities)
        outlets = []
        for index, (outlet_id, region_index, slots, position, interval) in enumerate(
            self._native.outlet_metadata()
        ):
            outlets.append({
                "id": str(outlet_id),
                "region": str(metadata[int(region_index)][0]),
                "position": list(map(float, position)),
                "interval_seconds": float(interval),
                "credit_seconds": float(outlet_credit[index]),
                "slots": list(map(str, slots)),
                "available_slots": sum(slot not in active for slot in slots),
                "last_moved_resources": _resources(pool_names, outlet_last[index]),
                "cumulative_moved_resources": _resources(
                    pool_names, outlet_total[index]
                ),
            })
        return {
            "format": VIEW_FORMAT,
            "config_sha256": self.config_sha256,
            "time": float(state["time"]),
            "step_index": int(state["step_index"]),
            "nodes": nodes,
            "edges": edges,
            "outlets": outlets,
            "last_events": copy.deepcopy(self.last_events),
        }

    def snapshot(self) -> dict[str, Any]:
        native = str(self._native.snapshot())
        return {
            "format": SNAPSHOT_FORMAT,
            "config": copy.deepcopy(self.config),
            "config_sha256": self.config_sha256,
            "chemistry_sha256": self.web.chemistry.sha256,
            "native": native,
            "native_sha256": hashlib.sha256(native.encode()).hexdigest(),
            "last_events": copy.deepcopy(self.last_events),
        }

    @classmethod
    def restore(cls, biosphere: Any, snapshot: Any) -> "RegionalMatter":
        value = _mapping(snapshot, "regional matter snapshot")
        if set(value) != {
            "format",
            "config",
            "config_sha256",
            "chemistry_sha256",
            "native",
            "native_sha256",
            "last_events",
        } or value.get("format") != SNAPSHOT_FORMAT:
            raise ValueError("unsupported regional matter snapshot")
        result = cls(biosphere, value["config"])
        native = value.get("native")
        if (
            result.config_sha256 != value.get("config_sha256")
            or result.web.chemistry.sha256 != value.get("chemistry_sha256")
            or not isinstance(native, str)
            or hashlib.sha256(native.encode()).hexdigest()
            != value.get("native_sha256")
        ):
            raise ValueError("regional matter snapshot identity differs")
        events = value.get("last_events")
        if not isinstance(events, list) or len(events) > 2048:
            raise ValueError("regional matter snapshot events differ")
        events = [_evidence_payload(event) for event in events]
        result._native.restore(native)
        result.last_events = events
        return result


__all__ = ["RegionalMatter"]
