"""Finite outside material offerings into the shared chemical world.

An offering transfers conserved pools from a private, inert source compartment
to an ordinary dormant :class:`MaterialObjects` slot.  The source compartment,
object compartment, and every later gut/body transfer live in one MetabolicWeb;
there is no scalar food inventory or material created by the visitor command.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .metabolism import canonical


FORMAT = "chreatures-visitor-material-supply-v1"
SNAPSHOT_FORMAT = "chreatures-visitor-material-supply-state-v1"
RECEIPT_FORMAT = "chreatures-visitor-material-offer-v1"
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


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


class VisitorMaterialSupply:
    """Move bounded external reserves into physical, chemically finite packets.

    Choice and slot identifiers stay in the environment implementation. A
    resident senses only the spawned object's ordinary geometry, contact,
    chemistry-derived color, and chemistry-derived odor.
    """

    MAX_CHOICES = 16
    MAX_SLOTS = 64
    MAX_DIAGNOSTIC_BYTES = 1_000_000

    def __init__(self, biosphere: Any, spec: Any):
        self.biosphere = biosphere
        self.config = self._normalize(spec)
        self.config_sha256 = hashlib.sha256(canonical(self.config)).hexdigest()
        self._prepare_bindings(require_initial=True)
        self.offer_count = 0
        self.last_receipt: dict[str, Any] | None = None

    @property
    def world(self):
        return self.biosphere.world

    @property
    def web(self):
        return self.biosphere.web

    @property
    def materials(self):
        materials = self.biosphere.materials
        if materials is None:
            raise RuntimeError("visitor material supply requires MaterialObjects")
        return materials

    def _normalize(self, value: Any) -> dict[str, Any]:
        if isinstance(value, (str, Path)):
            value = json.loads(Path(value).read_text())
        if not isinstance(value, Mapping):
            raise ValueError("visitor material supply must be an object")
        value = dict(value)
        if set(value) != {"format", "chemistry_sha256", "choices"}:
            raise ValueError("invalid visitor material supply fields")
        if value["format"] != FORMAT:
            raise ValueError("unsupported visitor material supply")
        chemistry_sha256 = value["chemistry_sha256"]
        if (
            not isinstance(chemistry_sha256, str)
            or len(chemistry_sha256) != 64
            or any(character not in "0123456789abcdef" for character in chemistry_sha256)
        ):
            raise ValueError("invalid visitor material chemistry identity")
        choices = value["choices"]
        if not isinstance(choices, list) or not 1 <= len(choices) <= self.MAX_CHOICES:
            raise ValueError(
                f"visitor material choices must contain 1..{self.MAX_CHOICES} entries"
            )
        normalized = []
        choice_ids: set[str] = set()
        source_rows: set[int] = set()
        slots: set[str] = set()
        for index, raw in enumerate(choices):
            if not isinstance(raw, Mapping):
                raise ValueError(f"visitor material choice {index} must be an object")
            raw = dict(raw)
            if set(raw) != {
                "id",
                "source_row",
                "initial_resources",
                "portion",
                "minimum_fraction",
                "slots",
            }:
                raise ValueError("invalid visitor material choice fields")
            choice_id = _identifier(raw["id"], "visitor material choice")
            row = raw["source_row"]
            if choice_id in choice_ids:
                raise ValueError("visitor material choice identities must be unique")
            if (
                isinstance(row, bool)
                or not isinstance(row, int)
                or row < 0
                or row in source_rows
            ):
                raise ValueError("visitor material source rows must be distinct")
            choice_ids.add(choice_id)
            source_rows.add(row)
            initial = self._resource_mapping(
                raw["initial_resources"], "initial visitor material"
            )
            portion = self._resource_mapping(raw["portion"], "visitor portion")
            if not any(amount > 0.0 for amount in initial.values()):
                raise ValueError("visitor source must contain finite material")
            if not any(amount > 0.0 for amount in portion.values()):
                raise ValueError("visitor portion must contain finite material")
            if any(portion.get(name, 0.0) > initial.get(name, 0.0) for name in portion):
                raise ValueError("visitor portion exceeds its initial source mixture")
            choice_slots = raw["slots"]
            if (
                not isinstance(choice_slots, list)
                or not choice_slots
                or len(choice_slots) > self.MAX_SLOTS
            ):
                raise ValueError("visitor choice needs a bounded slot list")
            normalized_slots = [
                _identifier(slot, "visitor material slot") for slot in choice_slots
            ]
            if (
                len(normalized_slots) != len(set(normalized_slots))
                or any(slot in slots for slot in normalized_slots)
            ):
                raise ValueError("visitor material slots must be globally unique")
            slots.update(normalized_slots)
            if len(slots) > self.MAX_SLOTS:
                raise ValueError("visitor material supply exceeds total slot capacity")
            normalized.append(
                {
                    "id": choice_id,
                    "source_row": row,
                    "initial_resources": initial,
                    "portion": portion,
                    "minimum_fraction": _number(
                        raw["minimum_fraction"],
                        "minimum visitor portion fraction",
                        0.001,
                        1.0,
                    ),
                    "slots": normalized_slots,
                }
            )
        return {
            "format": FORMAT,
            "chemistry_sha256": chemistry_sha256,
            "choices": normalized,
        }

    @staticmethod
    def _resource_mapping(value: Any, name: str) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object")
        return {
            _identifier(key, "chemical pool"): _number(
                amount, name, 0.0, 1e12
            )
            for key, amount in value.items()
        }

    def _prepare_bindings(self, *, require_initial: bool) -> None:
        if self.biosphere.materials is None:
            raise ValueError("visitor material supply requires MaterialObjects")
        if self.config["chemistry_sha256"] != self.web.chemistry.sha256:
            raise ValueError("visitor supply and shared-web chemistry differ")
        names = set(self.web.chemistry.pools)
        material_items = {
            item["entity"]: item for item in self.materials.config["objects"]
        }
        material_rows = set(self.materials.donor_rows.values())
        reserved_rows = set(material_rows)
        for colony in self.biosphere.config:
            reserved_rows.update((colony["body_row"], colony["structure_row"]))
        mobility = self.biosphere.mobility
        if mobility is not None:
            for resident in mobility.residents.values():
                reserved_rows.update((resident["body_row"], resident["gut_row"]))
        reserved_slots = set()
        exchange = self.biosphere.exchange
        if exchange is not None:
            reserved_slots.update(exchange.config["deposit_slots"])
            for emitter in exchange.emitters.values():
                reserved_rows.add(emitter["donor_row"])
                reserved_slots.update(emitter["deposit_slots"])

        self._choices = {}
        for choice in self.config["choices"]:
            if (
                set(choice["initial_resources"]) - names
                or set(choice["portion"]) - names
            ):
                raise ValueError("visitor supply references unknown chemical pools")
            row = choice["source_row"]
            if row >= self.web.count or row in reserved_rows:
                raise ValueError("visitor source row is absent or privately reserved")
            if (
                np.any(self.web.enzyme_activity[row] != 0.0)
                or self.web.atp[row] != 0.0
                or self.web.atp_capacity[row] != 0.0
            ):
                raise ValueError("visitor source rows must be chemically inert and ATP-free")
            initial = self.web.chemistry.resources(choice["initial_resources"])
            current = self.web.pools[row]
            if require_initial:
                if not np.array_equal(current, initial):
                    raise ValueError("visitor source row differs from its initial reserve")
            elif np.any(current < 0.0) or np.any(current > initial):
                raise ValueError("visitor source row exceeds its finite reserve")
            for slot in choice["slots"]:
                if slot in reserved_slots:
                    raise ValueError("visitor supply slots overlap ecological exchange")
                item = material_items.get(slot)
                if item is None or item.get("dormant_template") is None:
                    raise ValueError("visitor supply slots must be dormant MaterialObjects")
                capacity = self.web.chemistry.resources(item["capacities"])
                portion = self.web.chemistry.resources(choice["portion"])
                if (
                    np.any(portion > capacity)
                    or portion.sum() > self.materials.config["max_transfer"]
                ):
                    raise ValueError("visitor portion exceeds its physical slot capacity")
            self._choices[choice["id"]] = choice

    def _available_slot(self, choice: Mapping[str, Any]) -> str:
        existing = set(self.world._entity_mj)
        try:
            return next(slot for slot in choice["slots"] if slot not in existing)
        except StopIteration as exc:
            raise ValueError("all physical slots for this material are occupied") from exc

    def offer(self, choice_id: Any, position: Any) -> dict[str, Any]:
        """Place one finite portion through MaterialObjects atomically."""
        choice_id = _identifier(choice_id, "visitor material choice")
        try:
            choice = self._choices[choice_id]
        except KeyError as exc:
            raise ValueError("unknown visitor material choice") from exc
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError("visitor material position must contain three numbers")
        position = [
            _number(value, "visitor material position", -1e5, 1e5)
            for value in position
        ]
        if not (
            0.0 <= position[0] <= float(self.world.width)
            and 0.0 <= position[1] <= float(self.world.height)
            and 0.0 <= position[2] <= float(self.world.depth)
        ):
            raise ValueError("visitor material position is outside habitat")

        row = choice["source_row"]
        nominal = self.web.chemistry.resources(choice["portion"])
        positive = nominal > 0.0
        available = self.web.pools[row].copy()
        fraction = min(1.0, float(np.min(available[positive] / nominal[positive])))
        if fraction < choice["minimum_fraction"]:
            raise ValueError("finite visitor material reserve is exhausted")
        requested = nominal if fraction == 1.0 else nominal * fraction
        resources = {
            name: float(requested[index])
            for index, name in enumerate(self.web.chemistry.pools)
            if requested[index] > 0.0
        }
        slot = self._available_slot(choice)
        material_receipt = self.materials.deposit_batch(
            [
                {
                    "entity": slot,
                    "donor_row": row,
                    "resources": resources,
                    "position": position,
                }
            ]
        )
        moved = np.asarray(material_receipt["moved_resources"], dtype=np.float64)
        if moved.shape != (1, len(self.web.chemistry.pools)) or not np.any(moved[0] > 0):
            raise RuntimeError("visitor material deposit moved no chemistry")
        moved = moved[0]
        elements = moved @ self.web.chemistry._arrays[1]
        energy = float(moved @ self.web.chemistry._arrays[2])
        self.offer_count += 1
        receipt = {
            "format": RECEIPT_FORMAT,
            "choice": choice_id,
            "slot": slot,
            "position": position,
            "fraction": fraction,
            "pools": list(self.web.chemistry.pools),
            "moved_resources": moved.tolist(),
            "remaining_source_resources": self.web.pools[row].tolist(),
            "outside_boundary": {
                "elements": dict(
                    zip(self.web.chemistry.elements, elements.tolist(), strict=True)
                ),
                "chemical_energy": energy,
            },
            "offer_count": self.offer_count,
            "material_receipt": material_receipt,
        }
        self.last_receipt = copy.deepcopy(receipt)
        return receipt

    def command(self, command: Any) -> dict[str, Any]:
        """Validate the narrow Habitat3D command seam before any mutation."""
        if not isinstance(command, Mapping):
            raise ValueError("visitor material command must be an object")
        command = dict(command)
        if set(command) != {"op", "material", "x", "y", "z"}:
            raise ValueError("invalid visitor material command fields")
        if command["op"] != "offer_material":
            raise ValueError("unknown visitor material command")
        return self.offer(
            command["material"], [command["x"], command["y"], command["z"]]
        )

    def accounting(self) -> dict[str, Any]:
        """Report finite reserve and cumulative outside-to-habitat transfer."""
        result = {}
        for choice in self.config["choices"]:
            initial = self.web.chemistry.resources(choice["initial_resources"])
            remaining = self.web.pools[choice["source_row"]].copy()
            supplied = initial - remaining
            elements = supplied @ self.web.chemistry._arrays[1]
            portion = self.web.chemistry.resources(choice["portion"])
            positive = portion > 0.0
            remaining_portions = float(np.min(remaining[positive] / portion[positive]))
            available_slots = sum(
                slot not in self.world._entity_mj for slot in choice["slots"]
            )
            result[choice["id"]] = {
                "pools": list(self.web.chemistry.pools),
                "initial_resources": initial.tolist(),
                "remaining_resources": remaining.tolist(),
                "supplied_resources": supplied.tolist(),
                "supplied_elements": dict(
                    zip(self.web.chemistry.elements, elements.tolist(), strict=True)
                ),
                "supplied_chemical_energy": float(
                    supplied @ self.web.chemistry._arrays[2]
                ),
                "remaining_portions": remaining_portions,
                "available_slots": available_slots,
                "available": (
                    available_slots > 0
                    and remaining_portions >= choice["minimum_fraction"]
                ),
            }
        return result

    def view(self) -> dict[str, Any]:
        accounting = self.accounting()
        return {
            "format": FORMAT,
            "offer_count": self.offer_count,
            "choices": {
                choice["id"]: {
                    "remaining_resources": accounting[choice["id"]][
                        "remaining_resources"
                    ],
                    "pools": accounting[choice["id"]]["pools"],
                    "remaining_portions": accounting[choice["id"]][
                        "remaining_portions"
                    ],
                    "available_slots": accounting[choice["id"]]["available_slots"],
                    "available": accounting[choice["id"]]["available"],
                }
                for choice in self.config["choices"]
            },
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "format": SNAPSHOT_FORMAT,
            "config": copy.deepcopy(self.config),
            "config_sha256": self.config_sha256,
            "offer_count": self.offer_count,
            "last_receipt": copy.deepcopy(self.last_receipt),
        }

    @classmethod
    def restore(cls, biosphere: Any, snapshot: Any) -> "VisitorMaterialSupply":
        if not isinstance(snapshot, Mapping):
            raise ValueError("visitor material snapshot must be an object")
        snapshot = dict(snapshot)
        if set(snapshot) != {
            "format",
            "config",
            "config_sha256",
            "offer_count",
            "last_receipt",
        }:
            raise ValueError("invalid visitor material snapshot fields")
        if snapshot["format"] != SNAPSHOT_FORMAT:
            raise ValueError("unsupported visitor material snapshot")
        instance = cls.__new__(cls)
        instance.biosphere = biosphere
        instance.config = instance._normalize(snapshot["config"])
        instance.config_sha256 = hashlib.sha256(canonical(instance.config)).hexdigest()
        if instance.config_sha256 != snapshot["config_sha256"]:
            raise ValueError("visitor material snapshot identity differs")
        instance._prepare_bindings(require_initial=False)
        count = snapshot["offer_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid visitor material offer count")
        last = snapshot["last_receipt"]
        if last is not None and not isinstance(last, Mapping):
            raise ValueError("invalid last visitor material receipt")
        if count == 0 and last is not None:
            raise ValueError("visitor material receipt exists without an offer")
        try:
            if len(canonical(last)) > cls.MAX_DIAGNOSTIC_BYTES:
                raise ValueError("visitor material receipt exceeds snapshot bound")
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid last visitor material receipt") from exc
        instance.offer_count = count
        instance.last_receipt = copy.deepcopy(last)
        return instance


__all__ = ["VisitorMaterialSupply"]
