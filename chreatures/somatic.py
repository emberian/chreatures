"""Funded private physiology and development in the shared chemical web."""

from __future__ import annotations

import base64
import copy
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .metabolism import canonical
from .native_world import load_world_kernels
from .organism_interface import ACTION_DIM, ACTION_NAMES, PHYSIOLOGY_NAMES

FORMAT = "chreatures-somatic-physiology-v3"
ROW_NAMES = ("body_row", "gut_row", "structure_row", "gland_row", "brood_row")
SCALAR_PARAMETERS = {
    "gut_capacity",
    "reserve_capacity",
    "maintenance_rate",
    "activation_rate",
    "absorption_rate",
    "digestive_atp_rate",
    "bite_rate",
    "maximum_bite",
    "mouth_radius",
    "fatigue_rise",
    "fatigue_recovery",
    "structural_capacity",
    "gland_capacity",
    "gland_synthesis_rate",
    "brood_capacity",
    "secretion_rate",
    "release_rate",
    "allocation_rate",
    "brood_maturation_rate",
    "brood_material_target",
    "brood_energy_target",
    "release_radius",
    "exchange_load_decay_rate",
    "eat_activity_cost",
    "secrete_activity_cost",
    "allocate_activity_cost",
}
ALLOCATION_KEYS = ("structure", "gland", "brood")
LAST_KEYS = (
    "absorbed",
    "ingested_mass",
    "released_mass",
    "secreted_mass",
    "allocated_mass",
    "mechanical_work",
    "funded_scale",
    "mouth_material_contacts",
)


def _positive(value: Any, name: str, *, zero: bool = False) -> float:
    invalid = (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or (value < 0 if zero else value <= 0)
    )
    if invalid:
        raise ValueError(f"invalid mobile {name}")
    return float(value)


def _native_bytes(value: Any, name: str) -> dict[str, str]:
    data = bytes(value)
    return {
        f"{name}_sha256": hashlib.sha256(data).hexdigest(),
        f"{name}_base64": base64.b64encode(data).decode("ascii"),
    }


class SomaticPhysiology:
    """Own recurring somatic accounting for one stable physical body order."""

    def __init__(self, biosphere: Any, config: list[dict[str, Any]]):
        self.biosphere = biosphere
        self.config = copy.deepcopy(config)
        if not isinstance(config, list) or not 1 <= len(config) <= 32:
            raise ValueError("mobile physiology requires 1..32 residents")
        identities = [body.id for body in self.world.bodies]
        self.residents: dict[str, dict[str, Any]] = {}
        used_rows = {
            colony[key]
            for colony in biosphere.config
            for key in ("body_row", "structure_row")
        }
        required = {
            "id",
            *ROW_NAMES,
            *SCALAR_PARAMETERS,
            "allocation_weights",
            "secretion_profile",
        }
        for spec in self.config:
            if not isinstance(spec, dict) or set(spec) != required:
                raise ValueError("mobile physiology fields differ")
            identity = spec["id"]
            if identity not in identities or identity in self.residents:
                raise ValueError("mobile physiology identities differ")
            for key in ROW_NAMES:
                row = spec[key]
                if (
                    isinstance(row, bool)
                    or not isinstance(row, int)
                    or not 0 <= row < self.web.count
                    or row in used_rows
                ):
                    raise ValueError("mobile compartments must be valid and private")
                used_rows.add(row)
            for name in SCALAR_PARAMETERS:
                spec[name] = _positive(spec[name], name, zero=name.endswith("_cost"))
            if spec["maximum_bite"] > spec["gut_capacity"]:
                raise ValueError("single bite exceeds gut capacity")
            weights = spec["allocation_weights"]
            if not isinstance(weights, dict) or set(weights) != set(ALLOCATION_KEYS):
                raise ValueError("allocation weights differ")
            clean_weights = {
                key: _positive(weights[key], f"allocation {key}", zero=True)
                for key in ALLOCATION_KEYS
            }
            total = sum(clean_weights.values())
            if total <= 0:
                raise ValueError("allocation weights require positive mass")
            spec["allocation_weights"] = {
                key: value / total for key, value in clean_weights.items()
            }
            profile = np.asarray(spec["secretion_profile"], dtype=np.float64)
            if (
                profile.shape != (3,)
                or not np.isfinite(profile).all()
                or np.any(profile < 0)
                or profile.sum() <= 0
            ):
                raise ValueError(
                    "secretion profile must contain three nonnegative channels"
                )
            spec["secretion_profile"] = (profile / profile.sum()).tolist()
            self.residents[identity] = spec
        if list(self.residents) != identities:
            raise ValueError("mobile configuration must follow physical resident order")
        self.sha256 = hashlib.sha256(canonical(self.config)).hexdigest()

        self.mass_weights = np.asarray(
            self.web.chemistry._arrays[1].sum(axis=1), dtype=np.float64
        )
        self.reserve_index = self.web.chemistry.pools.index("reserve")
        traits = np.asarray(
            [self._trait_row(spec) for spec in self.residents.values()],
            dtype=np.float64,
        )
        native = load_world_kernels()
        self._native = native.SomaticCohort(
            self.sha256,
            traits,
            np.asarray([body.fatigue for body in self.world.bodies], dtype=np.float64),
        )
        specs = list(self.residents.values())
        self._lifecycle = native.LifecycleCohort(
            self.sha256,
            np.asarray(
                [spec["brood_material_target"] for spec in specs], dtype=np.float64
            ),
            np.asarray(
                [spec["brood_energy_target"] for spec in specs], dtype=np.float64
            ),
            np.asarray(
                [spec["brood_maturation_rate"] for spec in specs], dtype=np.float64
            ),
        )
        self.bite_credit = dict.fromkeys(self.residents, 0.0)
        self.paid = {
            key: {"maintenance": 0.0, "activation": 0.0, "unmet": 0.0}
            for key in self.residents
        }
        self.last = {key: self._empty_report() for key in self.residents}
        self.totals = {key: self._empty_report() for key in self.residents}
        self.contacts_since = float(self.web.time)
        self.pending_dt: float | None = None
        self._pending_actions: np.ndarray | None = None
        self._pending_allocation = np.zeros((len(self.residents), 3), dtype=np.float64)
        self._pending_secretion = np.zeros(len(self.residents), dtype=np.float64)
        self._outcomes: dict[str, dict[str, Any]] | None = None
        self._state = np.zeros((len(self.residents), 6), dtype=np.float64)
        self._state[:, 0] = [body.fatigue for body in self.world.bodies]
        self._state[:, 1] = 1.0
        for index, spec in enumerate(self.residents.values()):
            self._state[index, 2] = np.clip(
                self._mass(spec["structure_row"]) / spec["structural_capacity"], 0, 1
            )
            self._state[index, 3] = np.clip(
                self._mass(spec["gland_row"]) / spec["gland_capacity"], 0, 1
            )
            self._state[index, 4] = np.clip(
                self._mass(spec["brood_row"]) / spec["brood_capacity"], 0, 1
            )
        self._maturity = np.zeros(len(self.residents), dtype=np.float64)
        self._locomotion = np.asarray(
            [[body.speed, body.angular_velocity] for body in self.world.bodies],
            dtype=np.float64,
        )
        self._secretion_pulses: list[dict[str, Any]] = []
        self._next_pulse = 1

    @staticmethod
    def _empty_report() -> dict[str, float | int]:
        return {
            name: (0 if name == "mouth_material_contacts" else 0.0)
            for name in LAST_KEYS
        }

    @staticmethod
    def _trait_row(spec: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                spec["maintenance_rate"],
                spec["activation_rate"],
                spec["fatigue_rise"],
                spec["fatigue_recovery"],
                spec["allocation_rate"],
                spec["secretion_rate"],
                spec["release_rate"],
                spec["structural_capacity"],
                spec["gland_capacity"],
                spec["brood_capacity"],
                spec["release_radius"],
                *(spec["allocation_weights"][key] for key in ALLOCATION_KEYS),
                spec["exchange_load_decay_rate"],
                spec["eat_activity_cost"],
                spec["secrete_activity_cost"],
                spec["allocate_activity_cost"],
            ],
            dtype=np.float64,
        )

    @classmethod
    def expanded_from(
        cls,
        previous: SomaticPhysiology,
        biosphere: Any,
        config: list[dict[str, Any]],
    ) -> SomaticPhysiology:
        """Build a private B+1 owner while retaining all prior native state."""
        candidate = cls(biosphere, config)
        old_ids = list(previous.residents)
        new_ids = list(candidate.residents)
        if new_ids[:-1] != old_ids or len(new_ids) != len(old_ids) + 1:
            raise ValueError("somatic expansion requires one appended resident")
        newborn_id = new_ids[-1]
        newborn = candidate.residents[newborn_id]
        candidate._native = previous._native.expanded(
            candidate.sha256,
            cls._trait_row(newborn),
            float(candidate.world._body(newborn_id).fatigue),
        )
        candidate._lifecycle = previous._lifecycle.expanded(
            candidate.sha256,
            newborn["brood_material_target"],
            newborn["brood_energy_target"],
            newborn["brood_maturation_rate"],
        )
        candidate.bite_credit = {**previous.bite_credit, newborn_id: 0.0}
        candidate.paid = {
            **copy.deepcopy(previous.paid),
            newborn_id: {"maintenance": 0.0, "activation": 0.0, "unmet": 0.0},
        }
        candidate.last = {
            **copy.deepcopy(previous.last),
            newborn_id: cls._empty_report(),
        }
        candidate.totals = {
            **copy.deepcopy(previous.totals),
            newborn_id: cls._empty_report(),
        }
        candidate.contacts_since = previous.contacts_since
        candidate._state[:-1] = previous._state
        candidate._maturity[:-1] = previous._maturity
        candidate._locomotion[:-1] = previous._locomotion
        candidate._secretion_pulses = copy.deepcopy(previous._secretion_pulses)
        candidate._next_pulse = previous._next_pulse
        candidate._refresh_compartment_state()
        candidate.sync_bodies()
        return candidate

    @property
    def world(self):
        return self.biosphere.world

    @property
    def web(self):
        return self.biosphere.web

    def _mass(self, row: int) -> float:
        return float(self.web.pools[row] @ self.mass_weights)

    def _refresh_compartment_state(self) -> None:
        native_state = self._native.state()
        peaks = np.asarray(native_state["peak_structure_fraction"], dtype=np.float64)
        for index, spec in enumerate(self.residents.values()):
            self._state[index, 2] = np.clip(
                self._mass(spec["structure_row"]) / spec["structural_capacity"], 0, 1
            )
            self._state[index, 3] = np.clip(
                self._mass(spec["gland_row"]) / spec["gland_capacity"], 0, 1
            )
            self._state[index, 4] = np.clip(
                self._mass(spec["brood_row"]) / spec["brood_capacity"], 0, 1
            )
            peak = max(peaks[index], self._state[index, 2])
            self._state[index, 1] = (
                1.0 if peak <= 1e-12 else min(1.0, self._state[index, 2] / peak)
            )

    def normalized(self, resident_id: str) -> dict[str, float]:
        index = list(self.residents).index(resident_id)
        spec = self.residents[resident_id]
        reserve = float(self.web.pools[spec["body_row"], self.reserve_index])
        usable = float(self.web.atp[spec["body_row"]]) + 0.72 * reserve
        reference = (
            float(self.web.atp_capacity[spec["body_row"]])
            + 0.72 * spec["reserve_capacity"]
        )
        return {
            "energy": float(np.clip(usable / reference, 0.0, 1.0)),
            "gut": float(
                np.clip(self._mass(spec["gut_row"]) / spec["gut_capacity"], 0.0, 1.0)
            ),
            "fatigue": float(self._state[index, 0]),
            "structural_integrity": float(self._state[index, 1]),
            "development_fraction": float(self._state[index, 2]),
            "gland_fill": float(self._state[index, 3]),
            "brood_fill": float(self._state[index, 4]),
            "reproductive_maturity": float(self._maturity[index]),
            "exchange_load": float(self._state[index, 5]),
        }

    def normalized12(self, resident_id: str, neural_support: float) -> np.ndarray:
        support = _positive(neural_support, "neural support", zero=True)
        if support > 1.0:
            raise ValueError("neural support must be in [0,1]")
        state = self.normalized(resident_id)
        values = (
            state["energy"],
            state["gut"],
            state["fatigue"],
            float(
                np.tanh(
                    self._locomotion[list(self.residents).index(resident_id), 0] / 2.0
                )
            ),
            float(
                np.tanh(
                    self._locomotion[list(self.residents).index(resident_id), 1] / 4.0
                )
            ),
            support,
            state["structural_integrity"],
            state["development_fraction"],
            state["gland_fill"],
            state["brood_fill"],
            state["reproductive_maturity"],
            state["exchange_load"],
        )
        if len(values) != len(PHYSIOLOGY_NAMES):
            raise RuntimeError("physiology interface dimensions differ")
        return np.asarray(values, dtype=np.float32)

    def sync_bodies(self, *, validate: bool = False) -> None:
        for body in self.world.bodies:
            for key, value in self.normalized(body.id).items():
                if validate and abs(float(getattr(body, key)) - value) > 1e-12:
                    raise ValueError(
                        f"saved body readout disagrees with private chemistry: {body.id}.{key}"
                    )
                setattr(body, key, value)

    @staticmethod
    def _actions(actions: Any, residents: int) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if (
            values.shape != (residents, ACTION_DIM)
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
        ):
            raise ValueError("somatic actions must be one contiguous action12 cohort")
        if (
            np.any(np.abs(values[:, :4]) > 1.0)
            or np.any(values[:, 4:] < 0.0)
            or np.any(values[:, 4:] > 1.0)
        ):
            raise ValueError("somatic actions violate signed/rectified bounds")
        return values

    def begin_step(self, actions: Any, dt: float) -> np.ndarray:
        if self.pending_dt is not None:
            raise RuntimeError("mobile chemistry has an unfinished physical step")
        matrix = self._actions(actions, len(self.residents))
        payments, scales = self._native.begin(
            matrix,
            np.ascontiguousarray(
                [self.web.atp[spec["body_row"]] for spec in self.residents.values()],
                dtype=np.float64,
            ),
            float(dt),
        )
        payments = np.asarray(payments, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float32)
        for index, (key, spec) in enumerate(self.residents.items()):
            maintenance, activation = map(float, payments[index])
            self.web.pay_work(spec["body_row"], maintenance + activation)
            requested = dt * spec["maintenance_rate"]
            self.paid[key] = {
                "maintenance": maintenance,
                "activation": activation,
                "unmet": 0.0 if requested <= 0 else 1.0 - maintenance / requested,
            }
            self.last[key] = self._empty_report()
            self.last[key]["funded_scale"] = float(scales[index])
            self.bite_credit[key] = min(
                spec["maximum_bite"],
                self.bite_credit[key]
                + dt
                * spec["bite_rate"]
                * float(matrix[index, 8])
                * float(scales[index]),
            )
            for name, amount in (
                ("maintenance", maintenance),
                ("activation", activation),
            ):
                self.totals[key][name] = self.totals[key].get(name, 0.0) + amount
        self.pending_dt = float(dt)
        self._pending_actions = matrix.copy()
        return scales

    def _consume_contacts(
        self,
        actions: np.ndarray,
        contacts: tuple[np.ndarray, ...],
        positions: np.ndarray,
        rotations: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        resident, _counterpart, entity_slot, shape_index, point, *_ = contacts
        ids = list(self.residents)
        entity_ids: Sequence[str] = self.world._contact_entity_ids
        part_index = {
            (part["entity"], part["shape_index"]): name
            for name, part in self.biosphere.parts.items()
        }
        capacity = {
            key: max(0.0, spec["gut_capacity"] - self._mass(spec["gut_row"]))
            for key, spec in self.residents.items()
        }
        candidates: list[tuple[float, int, str]] = []
        material_contacts: set[tuple[int, str]] = set()
        for sample in range(len(resident)):
            owner = int(resident[sample])
            if owner < 0 or actions[owner, 8] <= 0.0:
                continue
            key = ids[owner]
            mouth = positions[owner] + rotations[owner, :, 0] * radii[owner]
            distance = float(np.linalg.norm(point[sample] - mouth))
            if distance > self.residents[key]["mouth_radius"]:
                continue
            slot = int(entity_slot[sample])
            if slot < 0:
                continue
            entity = entity_ids[slot]
            part = part_index.get((entity, int(shape_index[sample])))
            if part is not None:
                candidates.append((distance, owner, part))
            if (
                self.biosphere.materials is not None
                and entity in self.biosphere.materials.donor_rows
            ):
                material_contacts.add((owner, entity))
        selected: dict[str, int] = {}
        for _distance, owner, part_id in sorted(set(candidates)):
            if part_id in selected:
                continue
            key = ids[owner]
            resources = self.web.chemistry.resources(
                self.biosphere.parts[part_id]["resources"]
            )
            mass = float(resources @ self.mass_weights)
            if mass <= 0.0 or mass > min(capacity[key], self.bite_credit[key]):
                continue
            selected[part_id] = self.residents[key]["gut_row"]
            capacity[key] -= mass
            self.bite_credit[key] -= mass
            self.last[key]["ingested_mass"] += mass
        if selected:
            self.biosphere.transfer_parts(selected)
        if material_contacts:
            requests, owners = [], []
            for owner, entity in sorted(material_contacts):
                key = ids[owner]
                available = min(capacity[key], self.bite_credit[key])
                stock = self.web.pools[self.biosphere.materials.donor_rows[entity]]
                mass = float(stock @ self.mass_weights)
                amount = min(available, mass)
                if amount <= 0.0:
                    continue
                requests.append(
                    {
                        "entity": entity,
                        "receiver_row": self.residents[key]["gut_row"],
                        "resources": dict(
                            zip(
                                self.web.chemistry.pools,
                                (stock * amount / mass).tolist(),
                                strict=True,
                            )
                        ),
                    }
                )
                owners.append(key)
                capacity[key] -= amount
            if requests:
                receipt = self.biosphere.materials.withdraw_batch(requests)
                for key, vector in zip(owners, receipt["moved_resources"], strict=True):
                    mass = float(
                        np.asarray(vector, dtype=np.float64) @ self.mass_weights
                    )
                    self.bite_credit[key] = max(0.0, self.bite_credit[key] - mass)
                    self.last[key]["ingested_mass"] += mass
        counts = np.zeros(len(self.residents), dtype=np.float64)
        for owner, _entity in material_contacts:
            counts[owner] += 1.0
        return counts

    def _direct_release(
        self,
        budgets: np.ndarray,
        contacts: tuple[np.ndarray, ...],
        positions: np.ndarray,
        rotations: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        resident, counterpart, _entity, _shape, point, *_ = contacts
        ids = list(self.residents)
        direct = np.zeros(len(ids), dtype=np.float64)
        pairs: dict[int, tuple[float, int]] = {}
        for sample in range(len(resident)):
            donor, receiver = int(resident[sample]), int(counterpart[sample])
            if donor < 0 or receiver < 0 or donor == receiver or budgets[donor] <= 0.0:
                continue
            outlet = positions[donor] - rotations[donor, :, 0] * radii[donor]
            distance = float(np.linalg.norm(point[sample] - outlet))
            if distance <= self.residents[ids[donor]]["release_radius"]:
                pairs[donor] = min(
                    pairs.get(donor, (float("inf"), receiver)), (distance, receiver)
                )
        donors, receivers, resources, owner_rows = [], [], [], []
        for donor, (_distance, receiver) in sorted(pairs.items()):
            donor_spec = self.residents[ids[donor]]
            receiver_spec = self.residents[ids[receiver]]
            stock = self.web.pools[donor_spec["gut_row"]]
            stock_mass = float(stock @ self.mass_weights)
            free = max(
                0.0,
                receiver_spec["gut_capacity"] - self._mass(receiver_spec["gut_row"]),
            )
            amount = min(float(budgets[donor]), stock_mass, free)
            if amount <= 0.0:
                continue
            donors.append(donor_spec["gut_row"])
            receivers.append(receiver_spec["gut_row"])
            resources.append(
                dict(
                    zip(
                        self.web.chemistry.pools,
                        (stock * amount / stock_mass).tolist(),
                        strict=True,
                    )
                )
            )
            owner_rows.append(donor)
        if resources:
            receipt = self.web.transfer_batch(
                donors, receivers, resources, [0.0] * len(resources)
            )
            for donor, vector in zip(
                owner_rows, receipt["moved_resources"], strict=True
            ):
                direct[donor] += float(
                    np.asarray(vector, dtype=np.float64) @ self.mass_weights
                )
        return direct

    def finish_step(
        self,
        actions: Any,
        outcomes: dict[str, dict[str, Any]],
        *,
        contacts: tuple[np.ndarray, ...],
        kinematics: tuple[np.ndarray, ...],
        dt: float,
    ) -> None:
        matrix = self._actions(actions, len(self.residents))
        if (
            self.pending_dt != dt
            or self._pending_actions is None
            or not np.array_equal(matrix, self._pending_actions)
        ):
            raise RuntimeError("mobile physical/chemical action or timestep differs")
        if not isinstance(contacts, tuple) or len(contacts) != 8:
            raise ValueError("somatic contact cohort differs")
        position, rotation, radius, forward_speed, turn = [
            np.asarray(value) for value in kinematics
        ]
        count = len(contacts[0])
        expected = (
            (count,),
            (count,),
            (count,),
            (count,),
            (count, 3),
            (count,),
            (count,),
            (count,),
        )
        if tuple(np.asarray(value).shape for value in contacts) != expected:
            raise ValueError("somatic packed contact dimensions differ")
        residents = len(self.residents)
        if (
            position.shape != (residents, 3)
            or rotation.shape != (residents, 3, 3)
            or radius.shape != (residents,)
            or forward_speed.shape != (residents,)
            or turn.shape != (residents,)
        ):
            raise ValueError("somatic kinematic dimensions differ")
        self._locomotion[:] = np.column_stack((forward_speed, turn))
        mouth_counts = self._consume_contacts(
            matrix, contacts, position, rotation, radius
        )
        effort = np.asarray(
            [outcomes[key]["effort"] for key in self.residents], dtype=np.float64
        )
        result = self._native.finish(
            effort,
            np.asarray(
                [self._mass(s["structure_row"]) for s in self.residents.values()]
            ),
            np.asarray([self._mass(s["gland_row"]) for s in self.residents.values()]),
            np.asarray([self._mass(s["brood_row"]) for s in self.residents.values()]),
            float(dt),
        )
        self._pending_allocation = np.asarray(
            result["allocation_mass"], dtype=np.float64
        )
        for index, spec in enumerate(self.residents.values()):
            self._pending_allocation[index, 1] = min(
                self._pending_allocation[index, 1],
                dt
                * spec["gland_synthesis_rate"]
                * float(self.last[spec["id"]]["funded_scale"]),
            )
        self._pending_secretion = np.asarray(result["secretion_mass"], dtype=np.float64)
        release_budget = np.asarray(result["release_mass"], dtype=np.float64)
        self._state = np.asarray(result["state"], dtype=np.float64)
        direct = self._direct_release(
            release_budget, contacts, position, rotation, radius
        )
        if self.biosphere.exchange is not None:
            self.biosphere.exchange.stage_mobile_release(
                {
                    key: max(0.0, float(release_budget[index] - direct[index]))
                    for index, key in enumerate(self.residents)
                }
            )
        for index, (key, body) in enumerate(
            zip(self.residents, self.world.bodies, strict=True)
        ):
            body.fatigue = float(self._state[index, 0])
            work = float(outcomes[key].get("mechanical_work", 0.0))
            self.last[key]["mechanical_work"] = work
            self.last[key]["mouth_material_contacts"] = int(mouth_counts[index])
            self.last[key]["released_mass"] = float(direct[index])
            for name in (
                "mechanical_work",
                "mouth_material_contacts",
                "ingested_mass",
                "released_mass",
            ):
                self.totals[key][name] += self.last[key][name]
            outcomes[key].update(
                nutrition=0.0,
                ingested_mass=self.last[key]["ingested_mass"],
                released_mass=self.last[key]["released_mass"],
                secreted_mass=0.0,
                allocated_mass=0.0,
                mouth_material_contacts=int(mouth_counts[index]),
            )
        self._outcomes = outcomes
        self.contacts_since = float(self.web.time)

    def _allocate(self) -> np.ndarray:
        actual = np.zeros((len(self.residents), 3), dtype=np.float64)
        requests, donors, receivers, atp, owners = [], [], [], [], []
        for owner, spec in enumerate(self.residents.values()):
            body_stock = self.web.pools[spec["body_row"]]
            body_mass = float(body_stock @ self.mass_weights)
            remaining = body_mass
            for target, name in enumerate(ALLOCATION_KEYS):
                amount = min(float(self._pending_allocation[owner, target]), remaining)
                if amount <= 0.0 or body_mass <= 0.0:
                    continue
                vector = body_stock * (amount / body_mass)
                energy = 0.0
                if name == "brood":
                    ratio = spec["brood_energy_target"] / spec["brood_material_target"]
                    energy = min(float(self.web.atp[spec["body_row"]]), amount * ratio)
                requests.append(
                    dict(zip(self.web.chemistry.pools, vector.tolist(), strict=True))
                )
                donors.append(spec["body_row"])
                receivers.append(spec[f"{name}_row"])
                atp.append(energy)
                owners.append((owner, target))
                remaining -= amount
        if requests:
            receipt = self.web.transfer_batch(donors, receivers, requests, atp)
            for (owner, target), vector in zip(
                owners, receipt["moved_resources"], strict=True
            ):
                actual[owner, target] += float(
                    np.asarray(vector, dtype=np.float64) @ self.mass_weights
                )
        return actual

    def _secrete(self) -> np.ndarray:
        actual = np.zeros(len(self.residents), dtype=np.float64)
        for owner, spec in enumerate(self.residents.values()):
            stock = self.web.pools[spec["gland_row"]]
            mass = float(stock @ self.mass_weights)
            amount = min(float(self._pending_secretion[owner]), mass)
            if amount <= 0.0:
                continue
            vector = stock * (amount / mass)
            receipt = self.web.transfer_batch(
                [spec["gland_row"]],
                [None],
                [dict(zip(self.web.chemistry.pools, vector.tolist(), strict=True))],
                [0.0],
            )
            moved = np.asarray(receipt["moved_resources"][0], dtype=np.float64)
            actual[owner] = float(moved @ self.mass_weights)
            body = self.world._body(spec["id"])
            duration = min(30.0, max(0.05, 1.0 / spec["exchange_load_decay_rate"]))
            self._secretion_pulses.append(
                {
                    "serial": self._next_pulse,
                    "position": [body.x, body.y, body.z],
                    "profile": list(spec["secretion_profile"]),
                    "remaining": duration,
                    "duration": duration,
                    "mass": actual[owner],
                }
            )
            self._next_pulse += 1
        return actual

    def before_reactions(self, dt: float) -> None:
        if self.pending_dt != dt or self._outcomes is None:
            raise RuntimeError("mobile chemistry requires one completed physical step")
        specs = list(self.residents.values())
        self.web.transfer_batch(
            [s["body_row"] for s in specs],
            [s["gut_row"] for s in specs],
            [{} for _ in specs],
            [dt * s["digestive_atp_rate"] for s in specs],
        )
        allocated = self._allocate()
        secreted = self._secrete()
        for index, key in enumerate(self.residents):
            self.last[key]["allocated_mass"] = float(allocated[index].sum())
            self.last[key]["secreted_mass"] = float(secreted[index])
            self._outcomes[key]["allocated_mass"] = self.last[key]["allocated_mass"]
            self._outcomes[key]["secreted_mass"] = self.last[key]["secreted_mass"]
            for name in ("allocated_mass", "secreted_mass"):
                self.totals[key][name] += self.last[key][name]

    def after_reactions(self, dt: float) -> None:
        specs = list(self.residents.values())
        requests = []
        for spec in specs:
            reserve = max(
                0.0,
                spec["reserve_capacity"]
                - self.web.pools[spec["body_row"], self.reserve_index],
            )
            requests.append({"reserve": min(dt * spec["absorption_rate"], reserve)})
        receipt = self.web.transfer_batch(
            [s["gut_row"] for s in specs],
            [s["body_row"] for s in specs],
            requests,
            [0.0] * len(specs),
        )
        exchanged = np.zeros(len(specs), dtype=np.float64)
        for index, (spec, moved) in enumerate(
            zip(specs, receipt["moved_resources"], strict=True)
        ):
            key = spec["id"]
            absorbed = float(moved[self.reserve_index])
            self.last[key]["absorbed"] = absorbed
            self.totals[key]["absorbed"] += absorbed
            self._outcomes[key]["nutrition"] = absorbed
            exchanged[index] = (
                self.last[key]["ingested_mass"]
                + self.last[key]["released_mass"]
                + self.last[key]["secreted_mass"]
            )
        if self.biosphere.exchange is not None:
            released = self.biosphere.exchange.take_mobile_release_receipt()
            for index, key in enumerate(self.residents):
                amount = float(released.get(key, 0.0))
                self.last[key]["released_mass"] += amount
                self.totals[key]["released_mass"] += amount
                self._outcomes[key]["released_mass"] += amount
                exchanged[index] += amount
        self._native.record_exchange(np.ascontiguousarray(exchanged), float(dt))
        native_state = self._native.state()
        self._state[:, 0] = np.asarray(native_state["fatigue"])
        self._state[:, 5] = np.asarray(native_state["exchange_load"])
        self._refresh_compartment_state()
        lifecycle = self._lifecycle.advance(
            np.asarray([self._mass(s["brood_row"]) for s in specs]),
            np.asarray([self.web.atp[s["brood_row"]] for s in specs]),
            float(dt),
        )
        self._maturity = np.asarray(lifecycle["maturity"], dtype=np.float64)
        for pulse in self._secretion_pulses:
            pulse["remaining"] = max(0.0, pulse["remaining"] - dt)
        self._secretion_pulses = [
            pulse for pulse in self._secretion_pulses if pulse["remaining"] > 0.0
        ]
        self.sync_bodies()
        self.pending_dt = None
        self._pending_actions = None
        self._outcomes = None

    def field_sources(self) -> list[dict[str, Any]]:
        sources = []
        for pulse in self._secretion_pulses:
            fade = pulse["remaining"] / pulse["duration"]
            for channel, fraction in enumerate(pulse["profile"]):
                if fraction > 0.0:
                    sources.append(
                        {
                            "key": f"anonymous-secretion:{pulse['serial']}:{channel}",
                            "position": list(pulse["position"]),
                            "channel": channel,
                            "rate": pulse["mass"] * fraction * fade / pulse["duration"],
                            "spread": 0.04,
                        }
                    )
        return sources

    def organ_flows(self) -> np.ndarray:
        """Return completed donor-side release, secretion, and allocation mass."""
        if self.pending_dt is not None:
            raise RuntimeError("organ flows are unavailable inside a chemical boundary")
        return np.ascontiguousarray(
            [
                [row["released_mass"], row["secreted_mass"], row["allocated_mass"]]
                for row in self.last.values()
            ],
            dtype=np.float64,
        )

    def hatch_offers(self) -> list[dict[str, Any]]:
        raw = self._lifecycle.offers()
        offers = []
        for index, serial in zip(raw["resident_index"], raw["serial"], strict=True):
            owner = int(index)
            key = list(self.residents)[owner]
            spec = self.residents[key]
            body = self.world._body(key)
            offers.append(
                {
                    "offer_id": f"{self.sha256[:12]}:{int(serial)}",
                    "parent_id": key,
                    "resident_index": owner,
                    "serial": int(serial),
                    "brood_row": spec["brood_row"],
                    "material_target": spec["brood_material_target"],
                    "energy_target": spec["brood_energy_target"],
                    "position": [body.x, body.y, body.z],
                }
            )
        return offers

    def prepare_hatch(
        self, offer_id: str, child_genome: Mapping[str, Any]
    ) -> dict[str, Any]:
        offer = next(
            (item for item in self.hatch_offers() if item["offer_id"] == offer_id), None
        )
        if offer is None or not isinstance(child_genome, Mapping):
            raise ValueError("hatch offer or child genome identity is invalid")
        genome = copy.deepcopy(dict(child_genome))
        web_state = self.web.snapshot()
        proposal = {
            "format": "chreatures-hatch-proposal-v1",
            "offer": offer,
            "child_genome": genome,
            "child_genome_sha256": hashlib.sha256(canonical(genome)).hexdigest(),
            "web_state_sha256": hashlib.sha256(canonical(web_state)).hexdigest(),
            "rollback_snapshot": self.biosphere.snapshot(),
        }
        proposal["proposal_sha256"] = hashlib.sha256(canonical(proposal)).hexdigest()
        return proposal

    def snapshot(self) -> dict[str, Any]:
        if self.pending_dt is not None:
            raise RuntimeError(
                "cannot save mobile physiology inside a physical/chemical step"
            )
        value = {
            "format": FORMAT,
            "config": copy.deepcopy(self.config),
            "sha256": self.sha256,
            "bite_credit": self.bite_credit.copy(),
            "paid": copy.deepcopy(self.paid),
            "last": copy.deepcopy(self.last),
            "totals": copy.deepcopy(self.totals),
            "contacts_since": self.contacts_since,
            "secretion_pulses": copy.deepcopy(self._secretion_pulses),
            "next_pulse": self._next_pulse,
            "locomotion": self._locomotion.tolist(),
        }
        value.update(_native_bytes(self._native.snapshot(), "somatic_native"))
        value.update(_native_bytes(self._lifecycle.snapshot(), "lifecycle_native"))
        return value

    def restore_state(self, snapshot: Mapping[str, Any]) -> None:
        snapshot = copy.deepcopy(snapshot)
        required = {
            "format",
            "config",
            "sha256",
            "bite_credit",
            "paid",
            "last",
            "totals",
            "contacts_since",
            "secretion_pulses",
            "next_pulse",
            "locomotion",
            "somatic_native_sha256",
            "somatic_native_base64",
            "lifecycle_native_sha256",
            "lifecycle_native_base64",
        }
        if (
            set(snapshot) != required
            or snapshot["format"] != FORMAT
            or snapshot["sha256"] != self.sha256
            or snapshot["config"] != self.config
        ):
            raise ValueError("mobile physiology snapshot identity differs")
        for name, native in (
            ("somatic_native", self._native),
            ("lifecycle_native", self._lifecycle),
        ):
            data = base64.b64decode(snapshot[f"{name}_base64"], validate=True)
            if hashlib.sha256(data).hexdigest() != snapshot[f"{name}_sha256"]:
                raise ValueError("mobile native state checksum differs")
            native.restore(data)
        for name in ("bite_credit", "paid", "last", "totals"):
            if not isinstance(snapshot[name], dict) or set(snapshot[name]) != set(
                self.residents
            ):
                raise ValueError("mobile private state identities differ")
            setattr(self, name, copy.deepcopy(snapshot[name]))
        self.contacts_since = _positive(
            snapshot["contacts_since"], "contact boundary", zero=True
        )
        self._secretion_pulses = copy.deepcopy(snapshot["secretion_pulses"])
        self._next_pulse = int(snapshot["next_pulse"])
        locomotion = np.asarray(snapshot["locomotion"], dtype=np.float64)
        if (
            locomotion.shape != (len(self.residents), 2)
            or not np.isfinite(locomotion).all()
        ):
            raise ValueError("mobile locomotion snapshot differs")
        self._locomotion = locomotion
        somatic_state = self._native.state()
        lifecycle_state = self._lifecycle.state()
        self._state[:, 0] = np.asarray(somatic_state["fatigue"])
        self._state[:, 5] = np.asarray(somatic_state["exchange_load"])
        self._refresh_compartment_state()
        self._maturity = np.asarray(lifecycle_state["maturity"], dtype=np.float64)
        self.sync_bodies(validate=True)

    def view(self) -> dict[str, Any]:
        return {
            "kind": FORMAT,
            "sha256": self.sha256,
            "actions": list(ACTION_NAMES),
            "physiology": list(PHYSIOLOGY_NAMES),
            "contacts_since": self.contacts_since,
            "hatch_offers": self.hatch_offers(),
            "residents": {
                key: {
                    **self.normalized(key),
                    **self.last[key],
                    "totals": copy.deepcopy(self.totals[key]),
                }
                for key in self.residents
            },
            "actuator_budget": "engineered_activation_law",
            "mechanical_work": "separate_positive_active_force_estimate",
        }


__all__ = ["FORMAT", "SomaticPhysiology"]
