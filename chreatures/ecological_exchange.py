"""Spatial return and acquisition of material in the shared chemical web.

These are supplied transport laws, not organism preferences. Egestion leaves a
finite physical deposit at the body's rear; root uptake requires an actual
contact with constructed root geometry. All inventory remains in MetabolicWeb.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping

import mujoco
import numpy as np

from .metabolism import canonical
from .native_world import load_world_kernels

FORMAT = "chreatures-ecological-exchange-v4"


def _positive(value, name, *, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError(f"invalid ecological {name}")
    return float(value)


class EcologicalExchange:
    def __init__(self, biosphere, config):
        self.biosphere = biosphere
        config = copy.deepcopy(config)
        version = config.get("format") if isinstance(config, dict) else None
        expected = {"format", "deposit_slots", "mobiles", "roots", "emitters"}
        if (
            not isinstance(config, dict)
            or set(config) != expected
            or version != FORMAT
            or biosphere.materials is None
        ):
            raise ValueError("ecological exchange requires material object bindings")
        self.config = copy.deepcopy(config)
        slots = config["deposit_slots"]
        if not isinstance(slots, list) or len(set(slots)) != len(slots):
            raise ValueError("deposit slot identities must be unique")
        items = {item["entity"]: item for item in biosphere.materials.config["objects"]}
        if any(
            slot not in items or "dormant_template" not in items[slot] for slot in slots
        ):
            raise ValueError("egestion requires reusable dormant material slots")
        self.mobiles = {}
        self.roots = {}
        self.emitters = {}
        names = self.web.chemistry.pools
        for spec in config["mobiles"]:
            if set(spec) != {
                "id",
                "interval",
                "minimum_mass",
                "maximum_mass",
                "gut_rates",
                "body_rates",
                "offset_radii",
            }:
                raise ValueError("invalid egestion parameters")
            identity = spec["id"]
            if (
                biosphere.mobility is None
                or identity not in biosphere.mobility.residents
                or identity in self.mobiles
                or not slots
            ):
                raise ValueError("egestion needs a distinct physiological resident")
            for key in ("interval", "minimum_mass", "maximum_mass"):
                _positive(spec[key], key)
            if (
                not 0.05 <= spec["interval"] <= 3600
                or spec["minimum_mass"] > spec["maximum_mass"]
            ):
                raise ValueError("invalid egestion interval or packet bounds")
            offset = np.asarray(spec["offset_radii"], dtype=float)
            if (
                offset.shape != (3,)
                or not np.isfinite(offset).all()
                or np.max(np.abs(offset)) > 4
            ):
                raise ValueError("egestion offset must be bounded body-local radii")
            if offset[0] >= -1:
                raise ValueError("egestion outlet must lie behind the body")
            for key in ("gut_rates", "body_rates"):
                self._pool_parameters(spec[key], names)
            self.mobiles[identity] = spec
        colonies = {colony["id"]: colony for colony in biosphere.config}
        for spec in config["roots"]:
            if set(spec) != {"colony", "rates_per_area", "capacities"}:
                raise ValueError("invalid root acquisition parameters")
            identity = spec["colony"]
            if identity not in colonies or identity in self.roots:
                raise ValueError("root acquisition requires a distinct colony")
            self._pool_parameters(spec["rates_per_area"], names)
            self._pool_parameters(spec["capacities"], names)
            if set(spec["rates_per_area"]) != set(spec["capacities"]):
                raise ValueError("every acquired resource needs a root capacity")
            self.roots[identity] = spec
        used_slots = set(slots)
        used_donor_pools = set()
        structure_rows = {colony["structure_row"] for colony in biosphere.config}
        material_rows = set(biosphere.materials.donor_rows.values())
        for spec in config.get("emitters", []):
            if set(spec) != {
                "id",
                "donor_row",
                "attachment_entity",
                "local_offset",
                "interval",
                "minimum_mass",
                "maximum_mass",
                "rates",
                "reserve_floors",
                "deposit_slots",
            }:
                raise ValueError("invalid material emitter parameters")
            identity = spec["id"]
            donor = spec["donor_row"]
            emitter_slots = spec["deposit_slots"]
            if (
                not isinstance(identity, str)
                or not identity
                or identity in self.emitters
                or isinstance(donor, bool)
                or not isinstance(donor, int)
                or not 0 <= donor < len(self.web.pools)
                or donor in structure_rows
                or donor in material_rows
            ):
                raise ValueError("emitter requires a distinct non-structural donor")
            if (
                not isinstance(emitter_slots, list)
                or not emitter_slots
                or len(set(emitter_slots)) != len(emitter_slots)
                or set(emitter_slots) & used_slots
                or any(
                    slot not in items or "dormant_template" not in items[slot]
                    for slot in emitter_slots
                )
            ):
                raise ValueError("emitter dormant slots must be disjoint and reusable")
            used_slots.update(emitter_slots)
            if (
                spec["attachment_entity"] not in self.world._entity_mj
                and spec["attachment_entity"] not in self.world._body_mj
            ):
                raise ValueError("emitter attachment entity is absent")
            offset = np.asarray(spec["local_offset"], dtype=float)
            if (
                offset.shape != (3,)
                or not np.isfinite(offset).all()
                or np.max(np.abs(offset)) > 4
            ):
                raise ValueError("emitter local offset is invalid")
            for key in ("interval", "minimum_mass", "maximum_mass"):
                _positive(spec[key], key)
            if (
                not 0.05 <= spec["interval"] <= 3600
                or spec["minimum_mass"] > spec["maximum_mass"]
            ):
                raise ValueError("invalid emitter interval or packet bounds")
            self._pool_parameters(spec["rates"], names)
            self._pool_parameters(spec["reserve_floors"], names)
            if set(spec["rates"]) != set(spec["reserve_floors"]):
                raise ValueError("every emitted pool needs a protected reserve floor")
            for pool in spec["rates"]:
                key = (donor, pool)
                if key in used_donor_pools:
                    raise ValueError("emitters cannot share one donor pool")
                used_donor_pools.add(key)
            self.emitters[identity] = spec
        self.sha256 = hashlib.sha256(canonical(self.config)).hexdigest()
        self.elapsed = dict.fromkeys(self.mobiles, 0.0)
        self.release_credit = dict.fromkeys(self.mobiles, 0.0)
        self._staged_release: dict[str, float] | None = None
        self.release_receipt = dict.fromkeys(self.mobiles, 0.0)
        self.egested = {key: [0.0] * len(names) for key in self.mobiles}
        self.acquired = {key: [0.0] * len(names) for key in self.roots}
        self.capacity_blocked = dict.fromkeys(self.mobiles, 0)
        self.emitter_elapsed = dict.fromkeys(self.emitters, 0.0)
        self.emitted = {key: [0.0] * len(names) for key in self.emitters}
        self.emitter_capacity_blocked = dict.fromkeys(self.emitters, 0)
        self.emitter_attachment_unavailable = dict.fromkeys(self.emitters, 0)
        self.emitter_cursor = dict.fromkeys(self.emitters, 0)
        self.last = {
            "step_index": 0,
            "chemical_time_seconds": float(self.web.time),
            "pool_names": list(names),
            "resource_units": "synthetic_pool_quantity",
            "mass_units": "synthetic_element_sum",
            "evidence_events": [],
            "deposits": [],
            "root_transfers": [],
            "emitter_deposits": [],
        }
        self.step_index = 0
        self.mass_weights = self.web.chemistry._arrays[1].sum(axis=1)
        mobile_specs = list(self.mobiles.values())
        self._mobile_rows = np.asarray(
            [
                [
                    self.biosphere.mobility.residents[spec["id"]][f"{name}_row"]
                    for name in ("gut", "body")
                ]
                for spec in mobile_specs
            ],
            dtype=np.int64,
        ).reshape(len(mobile_specs), 2)
        self._mobile_rates = np.asarray(
            [
                [
                    [spec[f"{compartment}_rates"].get(name, 0.0) for name in names]
                    for compartment in ("gut", "body")
                ]
                for spec in mobile_specs
            ],
            dtype=np.float64,
        ).reshape(len(mobile_specs), 2, len(names))
        self._mobile_intervals = np.asarray(
            [spec["interval"] for spec in mobile_specs], dtype=np.float64,
        )
        self._mobile_minimum = np.asarray(
            [spec["minimum_mass"] for spec in mobile_specs], dtype=np.float64,
        )
        self._mobile_maximum = np.asarray(
            [spec["maximum_mass"] for spec in mobile_specs], dtype=np.float64,
        )
        self._native_mobile_candidates = load_world_kernels().mobile_release_candidates

    @classmethod
    def expanded_from(cls, previous, biosphere, config):
        """Build a B+1 exchange owner while retaining completed old ledgers."""
        candidate = cls(biosphere, config)
        old_ids = list(previous.mobiles)
        new_ids = list(candidate.mobiles)
        if new_ids[:-1] != old_ids or len(new_ids) != len(old_ids) + 1:
            raise ValueError("ecological expansion requires one appended mobile")
        child = new_ids[-1]
        candidate.elapsed = {**previous.elapsed, child: 0.0}
        candidate.release_credit = {**previous.release_credit, child: 0.0}
        candidate.release_receipt = {key: 0.0 for key in new_ids}
        candidate.egested = {
            **copy.deepcopy(previous.egested),
            child: [0.0] * len(candidate.web.chemistry.pools),
        }
        candidate.capacity_blocked = {**previous.capacity_blocked, child: 0}
        for name in (
            "acquired",
            "emitter_elapsed",
            "emitted",
            "emitter_capacity_blocked",
            "emitter_attachment_unavailable",
            "emitter_cursor",
        ):
            if set(getattr(previous, name)) != set(getattr(candidate, name)):
                raise ValueError("ecological nonmobile identity changed during birth")
            setattr(candidate, name, copy.deepcopy(getattr(previous, name)))
        candidate.last = copy.deepcopy(previous.last)
        candidate.step_index = previous.step_index
        return candidate

    @staticmethod
    def _pool_parameters(values, names):
        if not isinstance(values, dict) or set(values) - set(names):
            raise ValueError("unknown ecological resource")
        for key, value in values.items():
            _positive(value, key, zero=True)

    @property
    def world(self):
        return self.biosphere.world

    @property
    def web(self):
        return self.biosphere.web

    @property
    def step_events(self):
        """Committed material transfers from the most recent chemical step."""
        return copy.deepcopy(self.last["evidence_events"])

    def _record_event(self, kind, *, bodies=(), entities=(), resources, details):
        vector = np.asarray(resources, dtype=np.float64)
        if not np.any(vector > 0.0):
            return
        quantities = [
            {"name": f"pool:{name}", "value": float(vector[index]), "unit": "pool_quantity"}
            for index, name in enumerate(self.web.chemistry.pools)
            if vector[index] > 0.0
        ]
        quantities.append({
            "name": "element_weighted_mass",
            "value": float(vector @ self.mass_weights),
            "unit": "synthetic_element_sum",
        })
        self.last["evidence_events"].append({
            "kind": kind,
            "actors": {"bodies": list(bodies), "entities": list(entities)},
            "quantities": quantities,
            "details": copy.deepcopy(details),
            "source": {
                "stream": "ecological-exchange",
                "step_index": self.step_index + 1,
                "chemical_time_seconds": float(self.web.time),
            },
        })

    def _root_contacts(self):
        # MuJoCo contact geometry, including static roots against free deposits.
        # No founder placeholder can acquire resources: only funded root parts.
        parts = {
            (part["entity"], part["shape_index"]): (identity, part)
            for identity, part in self.biosphere.parts.items()
            if part["kind"] == "root" and part["colony"] in self.roots
        }
        materials = self.biosphere.materials.donor_rows
        contacts = {}
        for contact in self.world.data.contact:
            if contact.dist > 1e-5:
                continue
            geoms = [int(value) for value in contact.geom]
            for first, second in (geoms, geoms[::-1]):
                source = self.world._geom_entity.get(first)
                target = self.world._geom_entity.get(second)
                if source not in materials or target is None:
                    continue
                name = mujoco.mj_id2name(
                    self.world.model, mujoco.mjtObj.mjOBJ_GEOM, second
                )
                part = parts.get((target, int(name.rsplit(":", 1)[1])))
                if part is None:
                    continue
                identity, record = part
                shape = record["shape"]
                if shape["type"] != "capsule":
                    raise ValueError(
                        "root surface calculation requires capsule geometry"
                    )
                radius, half_length = self.world.model.geom_size[second, :2]
                area = 4 * math.pi * radius * (half_length + radius)
                tissue = ("soft_tissue", "tough_tissue")
                initial = sum(
                    record["initial_resources"].get(pool, 0.0) for pool in tissue
                )
                remaining = sum(record["resources"].get(pool, 0.0) for pool in tissue)
                area *= min(1.0, remaining / initial) if initial > 0 else 0.0
                if area <= 0:
                    continue
                contacts.setdefault((record["colony"], source), {})[identity] = area
        return contacts

    def before_reactions(self, dt):
        self.last = {
            "step_index": self.step_index + 1,
            "chemical_time_seconds": float(self.web.time + dt),
            "pool_names": list(self.web.chemistry.pools),
            "resource_units": "synthetic_pool_quantity",
            "mass_units": "synthetic_element_sum",
            "evidence_events": [],
            "deposits": [],
            "root_transfers": [],
            "emitter_deposits": [],
        }
        contacts = self._root_contacts()
        requests, recipients = [], []
        pools = self.web.pools
        for colony, spec in self.roots.items():
            available = {
                source: parts
                for (key, source), parts in contacts.items()
                if key == colony
            }
            if not available:
                continue
            unique = {
                part: area
                for parts in available.values()
                for part, area in parts.items()
            }
            area = sum(unique.values())
            weights = {
                source: sum(parts.values()) for source, parts in available.items()
            }
            weight_sum = sum(weights.values())
            row = self.biosphere._colony(colony)["body_row"]
            budget = {
                name: min(
                    dt * rate * area,
                    max(
                        0.0,
                        spec["capacities"][name]
                        - pools[row, self.web.chemistry.pools.index(name)],
                    ),
                )
                for name, rate in spec["rates_per_area"].items()
            }
            for source in sorted(available):
                requests.append(
                    {
                        "entity": source,
                        "receiver_row": row,
                        "resources": {
                            name: value * weights[source] / weight_sum
                            for name, value in budget.items()
                        },
                    }
                )
                recipients.append((colony, source))
        if requests:
            receipt = self.biosphere.materials.withdraw_batch(requests)
            for (colony, source), moved in zip(
                recipients, receipt["moved_resources"], strict=True
            ):
                self.acquired[colony] = (
                    np.asarray(self.acquired[colony]) + moved
                ).tolist()
                self.last["root_transfers"].append(
                    {"colony": colony, "source": source, "resources": moved}
                )
                self._record_event(
                    "root-material-acquisition", entities=(source,), resources=moved,
                    details={"entity_roles": {source: "physical-donor"}, "colony": colony},
                )

    def stage_mobile_release(self, budgets: Mapping[str, float]) -> None:
        """Accept one action-funded bolus budget for the next chemical boundary."""
        if self._staged_release is not None or set(budgets) != set(self.mobiles):
            raise RuntimeError("ecological mobile release boundary differs")
        clean = {
            key: _positive(value, "mobile release budget", zero=True)
            for key, value in budgets.items()
        }
        self._staged_release = clean

    def take_mobile_release_receipt(self) -> dict[str, float]:
        """Consume actual donor mass moved by the completed release boundary."""
        receipt = self.release_receipt.copy()
        self.release_receipt = dict.fromkeys(self.mobiles, 0.0)
        return receipt

    def after_reactions(self, dt):
        if self._staged_release is None:
            self._staged_release = dict.fromkeys(self.mobiles, 0.0)
        names = self.web.chemistry.pools
        free = [
            slot
            for slot in self.config["deposit_slots"]
            if slot not in self.world._entity_mj
        ]
        keys = tuple(self.mobiles)
        if keys:
            pools = self.web.pools[self._mobile_rows]
            vectors, elapsed, credit, masses = self._native_mobile_candidates(
                float(dt),
                np.asarray([self.elapsed[key] for key in keys], dtype=np.float64),
                np.asarray([self.release_credit[key] for key in keys], dtype=np.float64),
                np.asarray([self._staged_release[key] for key in keys], dtype=np.float64),
                np.ascontiguousarray(pools), self._mobile_rates,
                self._mobile_intervals, self._mobile_minimum, self._mobile_maximum,
                np.ascontiguousarray(self.mass_weights),
            )
            vectors = np.asarray(vectors)
            for index, key in enumerate(keys):
                self.elapsed[key] = float(elapsed[index])
                self.release_credit[key] = float(credit[index])
        else:
            vectors = np.empty((0, 2, len(names)), dtype=np.float64)
            masses = np.empty(0, dtype=np.float64)
        requests, recipients = [], []
        for index, (key, spec) in enumerate(self.mobiles.items()):
            if float(masses[index]) <= 0.0:
                continue
            if not free:
                self.capacity_blocked[key] += 1
                continue
            slot = free.pop(0)
            body = self.world._body(key)
            physical = self.world._body_mj[key]
            position = self.world.data.xpos[physical] + self.world.data.xmat[
                physical
            ].reshape(3, 3) @ (np.asarray(spec["offset_radii"]) * body.radius)
            # Outside the world, retain material; do not teleport the outlet.
            if np.any(position < 0) or np.any(
                position > [self.world.width, self.world.height, self.world.depth]
            ):
                self.capacity_blocked[key] += 1
                free.insert(0, slot)
                continue
            for row, vector in zip(self._mobile_rows[index], vectors[index], strict=True):
                if np.any(vector > 0):
                    requests.append(
                        {
                            "entity": slot,
                            "donor_row": int(row),
                            "resources": dict(zip(names, vector.tolist())),
                            "position": position.tolist(),
                        }
                    )
                    recipients.append((key, slot))
        if requests:
            receipt = self.biosphere.materials.deposit_batch(requests)
            blocked: set[tuple[str, str]] = set()
            for (key, slot), moved in zip(
                recipients, receipt["moved_resources"], strict=True
            ):
                mass = float(np.asarray(moved, dtype=np.float64) @ self.mass_weights)
                if mass <= 0.0:
                    if (key, slot) not in blocked:
                        self.capacity_blocked[key] += 1
                        blocked.add((key, slot))
                    continue
                self.egested[key] = (np.asarray(self.egested[key]) + moved).tolist()
                self.release_receipt[key] += mass
                self.release_credit[key] = max(0.0, self.release_credit[key] - mass)
                self.elapsed[key] = 0.0
                self.last["deposits"].append(
                    {"resident": key, "entity": slot, "resources": moved}
                )
                self._record_event(
                    "mobile-material-release", bodies=(key,), entities=(slot,),
                    resources=moved,
                    details={
                        "body_roles": {key: "donor"},
                        "entity_roles": {slot: "physical-packet"},
                    },
                )
            self.biosphere.mobility.sync_bodies()
        self._staged_release = None
        self._emit_material(dt)
        self.step_index += 1

    def _emit_material(self, dt):
        """Release donor-funded packets into emitter-reserved dormant slots."""
        if not self.emitters:
            return
        names = self.web.chemistry.pools
        requests, recipients = [], []
        for identity, spec in self.emitters.items():
            self.emitter_elapsed[identity] = min(
                spec["interval"],
                self.emitter_elapsed[identity] + dt,
            )
            elapsed = self.emitter_elapsed[identity]
            if elapsed + 1e-12 < spec["interval"]:
                continue
            donor = spec["donor_row"]
            available = np.maximum(
                self.web.pools[donor]
                - np.asarray([spec["reserve_floors"].get(name, 0.0) for name in names]),
                0.0,
            )
            requested = np.minimum(
                available,
                elapsed * np.asarray([spec["rates"].get(name, 0.0) for name in names]),
            )
            mass = float(requested @ self.mass_weights)
            if mass < spec["minimum_mass"]:
                continue
            slots = spec["deposit_slots"]
            start = self.emitter_cursor[identity] % len(slots)
            chosen = next(
                (
                    slots[(start + offset) % len(slots)]
                    for offset in range(len(slots))
                    if slots[(start + offset) % len(slots)] not in self.world._entity_mj
                ),
                None,
            )
            if chosen is None:
                self.emitter_capacity_blocked[identity] += 1
                continue
            attachment = spec["attachment_entity"]
            if (
                attachment not in self.world._entity_mj
                and attachment not in self.world._body_mj
            ):
                self.emitter_attachment_unavailable[identity] += 1
                continue
            entity = self.world._entity_mj.get(
                attachment,
                self.world._body_mj.get(attachment),
            )
            position = self.world.data.xpos[entity] + self.world.data.xmat[
                entity
            ].reshape(3, 3) @ np.asarray(spec["local_offset"], dtype=float)
            if np.any(position < 0) or np.any(
                position > [self.world.width, self.world.height, self.world.depth]
            ):
                self.emitter_capacity_blocked[identity] += 1
                continue
            requested *= min(1.0, spec["maximum_mass"] / mass)
            resources = {
                name: float(requested[index])
                for index, name in enumerate(names)
                if requested[index] > 0.0
            }
            requests.append(
                {
                    "entity": chosen,
                    "donor_row": donor,
                    "resources": resources,
                    "position": position.tolist(),
                }
            )
            recipients.append((identity, chosen))
        if not requests:
            return
        receipt = self.biosphere.materials.deposit_batch(requests)
        transferred = False
        for (identity, slot), moved in zip(
            recipients,
            receipt["moved_resources"],
            strict=True,
        ):
            vector = np.asarray(moved, dtype=np.float64)
            if np.any(vector > 0):
                transferred = True
                self.emitter_elapsed[identity] = 0.0
                slots = self.emitters[identity]["deposit_slots"]
                self.emitter_cursor[identity] = (slots.index(slot) + 1) % len(slots)
                self.emitted[identity] = (
                    np.asarray(self.emitted[identity]) + vector
                ).tolist()
                self.last["emitter_deposits"].append(
                    {
                        "emitter": identity,
                        "entity": slot,
                        "resources": moved,
                    }
                )
                self._record_event(
                    "colony-material-emission", entities=(
                        self.emitters[identity]["attachment_entity"], slot,
                    ), resources=moved,
                    details={
                        "emitter": identity,
                        "entity_roles": {
                            self.emitters[identity]["attachment_entity"]: "donor",
                            slot: "physical-packet",
                        },
                    },
                )
        if transferred and self.biosphere.mobility is not None:
            self.biosphere.mobility.sync_bodies()

    def snapshot(self):
        if self._staged_release is not None:
            raise RuntimeError(
                "cannot snapshot ecological exchange inside a release boundary"
            )
        value = {
            "format": self.config["format"],
            "config": copy.deepcopy(self.config),
            "sha256": self.sha256,
            "elapsed": self.elapsed.copy(),
            "egested": copy.deepcopy(self.egested),
            "acquired": copy.deepcopy(self.acquired),
            "capacity_blocked": self.capacity_blocked.copy(),
            "release_credit": self.release_credit.copy(),
            "release_receipt": self.release_receipt.copy(),
            "step_index": self.step_index,
            "last": copy.deepcopy(self.last),
        }
        value.update(
            {
                "emitter_elapsed": self.emitter_elapsed.copy(),
                "emitted": copy.deepcopy(self.emitted),
                "emitter_capacity_blocked": self.emitter_capacity_blocked.copy(),
                "emitter_attachment_unavailable": self.emitter_attachment_unavailable.copy(),
                "emitter_cursor": self.emitter_cursor.copy(),
            }
        )
        return value

    @classmethod
    def restore(cls, biosphere, snapshot: Mapping):
        source_format = snapshot.get("format")
        common = {
            "format",
            "config",
            "sha256",
            "elapsed",
            "egested",
            "acquired",
            "capacity_blocked",
            "release_credit",
            "release_receipt",
            "last",
            "step_index",
        }
        emitter_fields = {
            "emitter_elapsed",
            "emitted",
            "emitter_capacity_blocked",
            "emitter_attachment_unavailable",
            "emitter_cursor",
        }
        if source_format != FORMAT or set(snapshot) != common | emitter_fields:
            raise ValueError("invalid ecological exchange snapshot fields")
        if snapshot.get("config", {}).get("format") != source_format:
            raise ValueError("ecological exchange snapshot/config formats differ")
        source_sha256 = hashlib.sha256(canonical(snapshot["config"])).hexdigest()
        if snapshot["sha256"] != source_sha256:
            raise ValueError("ecological exchange identity differs")
        instance = cls(biosphere, snapshot["config"])
        if snapshot["sha256"] != instance.sha256:
            raise ValueError("ecological exchange identity differs")
        keys = [
            "elapsed",
            "egested",
            "acquired",
            "capacity_blocked",
            "release_credit",
            "release_receipt",
            "emitter_elapsed",
            "emitted",
            "emitter_capacity_blocked",
            "emitter_attachment_unavailable",
            "emitter_cursor",
        ]
        for key in keys:
            value = snapshot[key]
            expected = getattr(instance, key)
            if not isinstance(value, dict) or set(value) != set(expected):
                raise ValueError("ecological private state identities differ")
            for identity, row in value.items():
                if isinstance(expected[identity], list):
                    if not isinstance(row, list) or len(row) != len(expected[identity]):
                        raise ValueError("ecological resource vector differs")
                    for item in row:
                        _positive(item, key, zero=True)
                else:
                    _positive(row, key, zero=True)
                    if key in {
                        "capacity_blocked",
                        "emitter_capacity_blocked",
                        "emitter_attachment_unavailable",
                        "emitter_cursor",
                    } and not isinstance(row, int):
                        raise ValueError("ecological event count must be integral")
                    if (
                        key == "elapsed"
                        and row > instance.mobiles[identity]["interval"]
                    ):
                        raise ValueError("ecological elapsed phase exceeds interval")
                    if (
                        key == "emitter_elapsed"
                        and row > instance.emitters[identity]["interval"]
                    ):
                        raise ValueError("emitter elapsed phase exceeds interval")
                    if key == "emitter_cursor" and row >= len(
                        instance.emitters[identity]["deposit_slots"]
                    ):
                        raise ValueError("emitter slot cursor is invalid")
            setattr(instance, key, copy.deepcopy(value))
        canonical(snapshot["last"])
        instance.last = copy.deepcopy(snapshot["last"])
        step_index = snapshot["step_index"]
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError("ecological exchange step index is invalid")
        instance.step_index = step_index
        return instance

    def view(self):
        return {
            "kind": self.config["format"],
            "sha256": self.sha256,
            "step_index": self.step_index,
            "pools": list(self.web.chemistry.pools),
            "egested": copy.deepcopy(self.egested),
            "acquired": copy.deepcopy(self.acquired),
            "capacity_blocked": self.capacity_blocked.copy(),
            "release_credit": self.release_credit.copy(),
            "last_release_mass": self.release_receipt.copy(),
            "emitted": copy.deepcopy(self.emitted),
            "emitter_capacity_blocked": self.emitter_capacity_blocked.copy(),
            "emitter_attachment_unavailable": self.emitter_attachment_unavailable.copy(),
            "emitter_cursor": self.emitter_cursor.copy(),
            "last": copy.deepcopy(self.last),
        }
