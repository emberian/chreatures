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

FORMAT_V1 = "chreatures-ecological-exchange-v1"
FORMAT = "chreatures-ecological-exchange-v2"


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
        expected = (
            {"format", "deposit_slots", "mobiles", "roots"}
            if version == FORMAT_V1
            else {"format", "deposit_slots", "mobiles", "roots", "emitters"}
        )
        if (
            not isinstance(config, dict)
            or set(config) != expected
            or version not in {FORMAT_V1, FORMAT}
            or biosphere.materials is None
        ):
            raise ValueError("ecological exchange requires material object bindings")
        if version == FORMAT_V1:
            config["format"] = FORMAT
            config["emitters"] = []
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
                "id", "donor_row", "attachment_entity", "local_offset",
                "interval", "minimum_mass", "maximum_mass", "rates",
                "reserve_floors", "deposit_slots",
            }:
                raise ValueError("invalid material emitter parameters")
            identity = spec["id"]
            donor = spec["donor_row"]
            emitter_slots = spec["deposit_slots"]
            if (
                not isinstance(identity, str) or not identity
                or identity in self.emitters
                or isinstance(donor, bool) or not isinstance(donor, int)
                or not 0 <= donor < len(self.web.pools)
                or donor in structure_rows or donor in material_rows
            ):
                raise ValueError("emitter requires a distinct non-structural donor")
            if (
                not isinstance(emitter_slots, list) or not emitter_slots
                or len(set(emitter_slots)) != len(emitter_slots)
                or set(emitter_slots) & used_slots
                or any(slot not in items or "dormant_template" not in items[slot]
                       for slot in emitter_slots)
            ):
                raise ValueError("emitter dormant slots must be disjoint and reusable")
            used_slots.update(emitter_slots)
            if (
                spec["attachment_entity"] not in self.world._entity_mj
                and spec["attachment_entity"] not in self.world._body_mj
            ):
                raise ValueError("emitter attachment entity is absent")
            offset = np.asarray(spec["local_offset"], dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all() or np.max(np.abs(offset)) > 4:
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
        self.egested = {key: [0.0] * len(names) for key in self.mobiles}
        self.acquired = {key: [0.0] * len(names) for key in self.roots}
        self.capacity_blocked = dict.fromkeys(self.mobiles, 0)
        self.emitter_elapsed = dict.fromkeys(self.emitters, 0.0)
        self.emitted = {key: [0.0] * len(names) for key in self.emitters}
        self.emitter_capacity_blocked = dict.fromkeys(self.emitters, 0)
        self.emitter_attachment_unavailable = dict.fromkeys(self.emitters, 0)
        self.emitter_cursor = dict.fromkeys(self.emitters, 0)
        self.last = {
            "deposits": [], "root_transfers": [], "emitter_deposits": [],
        }
        self.mass_weights = self.web.chemistry._arrays[1].sum(axis=1)

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
            "deposits": [], "root_transfers": [], "emitter_deposits": [],
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

    def after_reactions(self, dt):
        names = self.web.chemistry.pools
        free = [
            slot
            for slot in self.config["deposit_slots"]
            if slot not in self.world._entity_mj
        ]
        requests, recipients = [], []
        for key, spec in self.mobiles.items():
            self.elapsed[key] += dt
            if self.elapsed[key] + 1e-12 < spec["interval"]:
                continue
            elapsed = self.elapsed[key]
            self.elapsed[key] = 0.0
            private = self.biosphere.mobility.residents[key]
            scale = self.biosphere.mobility.last[key]["funded_scale"]
            vectors, rows = [], []
            for compartment in ("gut", "body"):
                row = private[f"{compartment}_row"]
                rates = np.asarray(
                    [spec[f"{compartment}_rates"].get(name, 0) for name in names]
                )
                vectors.append(
                    self.web.pools[row] * (-np.expm1(-elapsed * scale * rates))
                )
                rows.append(row)
            mass = sum(float(vector @ self.mass_weights) for vector in vectors)
            if mass < spec["minimum_mass"]:
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
            factor = min(1.0, spec["maximum_mass"] / mass)
            for row, vector in zip(rows, vectors, strict=True):
                if np.any(vector > 0):
                    requests.append(
                        {
                            "entity": slot,
                            "donor_row": row,
                            "resources": dict(zip(names, (factor * vector).tolist())),
                            "position": position.tolist(),
                        }
                    )
                    recipients.append((key, slot))
        if requests:
            receipt = self.biosphere.materials.deposit_batch(requests)
            for (key, slot), moved in zip(
                recipients, receipt["moved_resources"], strict=True
            ):
                self.egested[key] = (np.asarray(self.egested[key]) + moved).tolist()
                self.last["deposits"].append(
                    {"resident": key, "entity": slot, "resources": moved}
                )
            self.biosphere.mobility.sync_bodies()
        self._emit_material(dt)

    def _emit_material(self, dt):
        """Release donor-funded packets into emitter-reserved dormant slots."""
        if not self.emitters:
            return
        names = self.web.chemistry.pools
        requests, recipients = [], []
        for identity, spec in self.emitters.items():
            self.emitter_elapsed[identity] = min(
                spec["interval"], self.emitter_elapsed[identity] + dt,
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
                (slots[(start + offset) % len(slots)] for offset in range(len(slots))
                 if slots[(start + offset) % len(slots)] not in self.world._entity_mj),
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
                attachment, self.world._body_mj.get(attachment),
            )
            position = self.world.data.xpos[entity] + self.world.data.xmat[entity].reshape(
                3, 3
            ) @ np.asarray(spec["local_offset"], dtype=float)
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
            requests.append({
                "entity": chosen, "donor_row": donor,
                "resources": resources, "position": position.tolist(),
            })
            recipients.append((identity, chosen))
        if not requests:
            return
        receipt = self.biosphere.materials.deposit_batch(requests)
        transferred = False
        for (identity, slot), moved in zip(
            recipients, receipt["moved_resources"], strict=True,
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
                self.last["emitter_deposits"].append({
                    "emitter": identity, "entity": slot, "resources": moved,
                })
        if transferred and self.biosphere.mobility is not None:
            self.biosphere.mobility.sync_bodies()

    def snapshot(self):
        value = {
            "format": self.config["format"],
            "config": copy.deepcopy(self.config),
            "sha256": self.sha256,
            "elapsed": self.elapsed.copy(),
            "egested": copy.deepcopy(self.egested),
            "acquired": copy.deepcopy(self.acquired),
            "capacity_blocked": self.capacity_blocked.copy(),
            "last": copy.deepcopy(self.last),
        }
        value.update({
            "emitter_elapsed": self.emitter_elapsed.copy(),
            "emitted": copy.deepcopy(self.emitted),
            "emitter_capacity_blocked": self.emitter_capacity_blocked.copy(),
            "emitter_attachment_unavailable": self.emitter_attachment_unavailable.copy(),
            "emitter_cursor": self.emitter_cursor.copy(),
        })
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
            "last",
        }
        emitter_fields = {
            "emitter_elapsed", "emitted", "emitter_capacity_blocked",
            "emitter_attachment_unavailable", "emitter_cursor",
        }
        if (
            source_format not in {FORMAT_V1, FORMAT}
            or set(snapshot) != common | (emitter_fields if source_format == FORMAT else set())
        ):
            raise ValueError("invalid ecological exchange snapshot fields")
        if snapshot.get("config", {}).get("format") != source_format:
            raise ValueError("ecological exchange snapshot/config formats differ")
        source_sha256 = hashlib.sha256(canonical(snapshot["config"])).hexdigest()
        if snapshot["sha256"] != source_sha256:
            raise ValueError("ecological exchange identity differs")
        instance = cls(biosphere, snapshot["config"])
        if source_format == FORMAT and snapshot["sha256"] != instance.sha256:
            raise ValueError("ecological exchange identity differs")
        keys = ["elapsed", "egested", "acquired", "capacity_blocked"]
        if source_format == FORMAT:
            keys.extend((
                "emitter_elapsed", "emitted", "emitter_capacity_blocked",
                "emitter_attachment_unavailable", "emitter_cursor",
            ))
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
                        "capacity_blocked", "emitter_capacity_blocked",
                        "emitter_attachment_unavailable", "emitter_cursor",
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
                    if (
                        key == "emitter_cursor"
                        and row >= len(instance.emitters[identity]["deposit_slots"])
                    ):
                        raise ValueError("emitter slot cursor is invalid")
            setattr(instance, key, copy.deepcopy(value))
        canonical(snapshot["last"])
        instance.last = copy.deepcopy(snapshot["last"])
        if source_format == FORMAT_V1:
            instance.last["emitter_deposits"] = []
        return instance

    def view(self):
        return {
            "kind": self.config["format"],
            "sha256": self.sha256,
            "pools": list(self.web.chemistry.pools),
            "egested": copy.deepcopy(self.egested),
            "acquired": copy.deepcopy(self.acquired),
            "capacity_blocked": self.capacity_blocked.copy(),
            "emitted": copy.deepcopy(self.emitted),
            "emitter_capacity_blocked": self.emitter_capacity_blocked.copy(),
            "emitter_attachment_unavailable": self.emitter_attachment_unavailable.copy(),
            "emitter_cursor": self.emitter_cursor.copy(),
            "last": copy.deepcopy(self.last),
        }
