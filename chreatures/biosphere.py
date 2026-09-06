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
from pathlib import Path
from typing import Any

import numpy as np

from .growth import GrowthSystem
from .metabolism import Chemistry, MetabolicWeb, canonical

FORMAT = "chreatures-biosphere-v3"
KINDS = ("branch", "root", "leaf")


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
        mobiles=None,
    ):
        self.world = world
        self.web = web
        self.config = copy.deepcopy(colonies)
        self.config_sha256 = hashlib.sha256(canonical(self.config)).hexdigest()
        self.growth: dict[str, GrowthSystem] = {}
        self.parts: dict[str, dict[str, Any]] = {}
        self.active: dict[str, bool] = {}
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
        # One-way data import: old saved colonies have no mobile coupling.
        if config.get("format") == "chreatures-biosphere-birth-v1":
            config["format"] = "chreatures-biosphere-birth-v2"
            config["mobiles"] = None
            config["material_objects"] = None
        if config.get("format") == "chreatures-biosphere-birth-v2":
            config["format"] = "chreatures-biosphere-birth-v3"
            config["exchange"] = None
        if config.get("format") != "chreatures-biosphere-birth-v3" or set(config) != {
            "format",
            "chemistry",
            "compartments",
            "bulk",
            "colonies",
            "mobiles",
            "material_objects",
            "exchange",
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
        instance = cls(world, web, config["colonies"], mobiles=config["mobiles"])
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
        instance._check_structure()
        return instance

    def _ledger(self) -> np.ndarray:
        return np.asarray(self.web._native.cumulative_ledger).sum(axis=0)

    def _colony(self, colony_id: str) -> dict[str, Any]:
        return next(colony for colony in self.config if colony["id"] == colony_id)

    def _point(self, colony: Mapping[str, Any], point: Any) -> np.ndarray:
        body_id = self.world._entity_mj[colony["bindings"]["branch"]]
        return (
            self.world.data.xpos[body_id]
            + self.world.data.xmat[body_id].reshape(3, 3) @ point
        )

    def _illumination(self, point: Any) -> float:
        p = np.asarray(point, dtype=float)
        size = np.asarray([self.world.width, self.world.height, self.world.depth])
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > size):
            return 0.0
        return float(self.world.sample_environment([p.tolist()])[0]["illumination"])

    def _photon_budget(self, colony: Mapping[str, Any], dt: float) -> float:
        # Founder photosynthetic surface has an explicit area and attachment.
        point = self._point(colony, np.asarray([0.0, 0.0, 0.025]))
        capture = colony["seed_capture_area"] * self._illumination(point)
        for part in self.parts.values():
            if part["colony"] != colony["id"] or part["kind"] != "leaf":
                continue
            shape = part["shape"]
            # The upper bounding surface avoids self-intersection. The current
            # effective area is supplied by the grammar, not a radiative solver.
            center = self._point(colony, np.asarray(shape["position"]))
            center[2] += max(shape["size"]) + 1e-4
            original = part["initial_resources"].get("soft_tissue", 0.0)
            active_fraction = (
                part["resources"].get("soft_tissue", 0.0) / original
                if original > 0
                else 0.0
            )
            capture += (
                part["area"]
                * max(0.0, min(1.0, active_fraction))
                * self._illumination(center)
            )
        return dt * colony["photon_flux"] * capture

    def _signals(self, colony: Mapping[str, Any]) -> list[dict[str, Any]]:
        growth = self.growth[colony["id"]]
        mineral = float(
            self.web.pools[
                colony["body_row"], self.web.chemistry.pools.index("mineral")
            ]
        )
        nutrient = mineral / (mineral + colony["mineral_half_saturation"])
        result = []
        for bud in growth.buds():
            # Buds are attached by construction; support is structural ancestry,
            # not a claim that static branches have a fitted stress model.
            local = np.asarray(bud["position"]) + 0.025 * np.asarray(bud["forward"])
            light = self._illumination(self._point(colony, local))
            result.append(
                {
                    "bud_id": bud["bud_id"],
                    "light": light,
                    "nutrient": nutrient,
                    "support": 1.0,
                    "competition": 0.0,
                }
            )
        return result

    def advance(self, dt: float) -> dict[str, Any]:
        """Advance chemistry and development after the caller advances physics."""
        if not np.isfinite(dt) or not 0 < dt <= 1.0:
            raise ValueError("biosphere step must be in (0, 1] seconds")
        self._check_structure()
        if self.exchange is not None:
            self.exchange.before_reactions(dt)
        photons = np.zeros(self.web.count, dtype=np.float64)
        for colony in self.config:
            if self.active[colony["id"]]:
                photons[colony["body_row"]] = self._photon_budget(colony, dt)
                self.growth[colony["id"]].elapse(dt)
        if self.mobility is not None:
            self.mobility.before_reactions(dt)
        ledger = self.web.step(dt, photons, np.zeros(self.web.count))
        if self.mobility is not None:
            self.mobility.after_reactions(dt)
        if self.exchange is not None:
            self.exchange.after_reactions(dt)
        self._distribute_turnover(ledger)
        reports = self._develop()
        if self.materials is not None:
            self.materials.sync_geometry()
            self._sync_material_cues()
        self.last_report = {
            "time": self.web.time,
            "captured_photons": float(ledger["photon_used"].sum()),
            "developments": reports,
            "parts": len(self.parts),
            "accounting": self.accounting(),
            "exchange": self.exchange.view() if self.exchange is not None else None,
        }
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
        if self.materials is None:
            return []
        sources = []
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
        return sources

    def _check_structure(self) -> None:
        for colony in self.config:
            total = np.zeros(len(self.web.chemistry.pools))
            for part in self.parts.values():
                if part["colony"] == colony["id"]:
                    total += self.web.chemistry.resources(part["resources"])
            if not np.allclose(
                total, self.web.pools[colony["structure_row"]], rtol=1e-11, atol=1e-12
            ):
                raise ValueError(
                    "physical structures and allocated chemical tissue disagree"
                )

    def _distribute_turnover(self, ledger: Mapping[str, Any]) -> None:
        # Turnover changes live tissue to detritus without removing its material
        # or geometry. Dead scaffold can persist; removal is a separate transfer.
        chemistry = self.web.chemistry
        for colony in self.config:
            owned = [
                part for part in self.parts.values() if part["colony"] == colony["id"]
            ]
            if not owned:
                continue
            before = np.asarray(
                [chemistry.resources(part["resources"]) for part in owned]
            )
            after = before.copy()
            for reaction, extent in enumerate(
                ledger["extent"][colony["structure_row"]]
            ):
                if extent <= 0:
                    continue
                stoich = chemistry._arrays[0][reaction]
                consumed = np.flatnonzero(stoich < 0)
                if len(consumed) != 1:
                    raise RuntimeError(
                        "structural reaction has unsupported geometric bookkeeping"
                    )
                substrate = int(consumed[0])
                total = before[:, substrate].sum()
                if total <= 0:
                    raise RuntimeError(
                        "structural reaction consumed unallocated substrate"
                    )
                after += (extent * before[:, substrate] / total)[:, None] * stoich[
                    None, :
                ]
            if np.any(after < -1e-12):
                raise RuntimeError("structural turnover produced negative material")
            after = np.maximum(after, 0.0)
            for part, resources in zip(owned, after, strict=True):
                part["resources"] = dict(zip(chemistry.pools, resources.tolist()))

    def _develop(self) -> list[dict[str, Any]]:
        operations = []
        staged: list[tuple[dict[str, Any], dict[str, Any], GrowthSystem]] = []
        pending: list[tuple[str, str]] = []
        next_parts = copy.deepcopy(self.parts)
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
            self.parts = next_parts
            for colony, _, candidate in staged:
                self.growth[colony["id"]] = candidate
            return [
                {
                    "colony": colony["id"],
                    "request": proposal["request"],
                    "token": proposal["token"],
                }
                for colony, proposal, _ in staged
            ]
        except Exception:
            # Preparation/payment failures consume neither a developmental RNG
            # draw nor a bud. Unexpected physical failures propagate to the
            # runtime's existing failed-tick pause boundary.
            for name, token in pending:
                growth = self.growth[name]
                if growth.snapshot()["pending"] is not None:
                    growth.reject(token)
            raise

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
                }
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
        return {
            "parts": part_ids,
            "receivers": dict(receivers),
            "resources": dict(zip(self.web.chemistry.pools, totals.tolist())),
        }

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
            "web": self.web.snapshot(),
            "growth": {key: value.snapshot() for key, value in self.growth.items()},
            "parts": copy.deepcopy(self.parts),
            "active": self.active.copy(),
            "initial_totals": copy.deepcopy(self.initial_totals),
            "initial_ledger": self.initial_ledger.copy(),
            "last_report": copy.deepcopy(self.last_report),
            "material_objects": self.materials.snapshot()
            if self.materials is not None
            else None,
            "mobility": self.mobility.snapshot() if self.mobility is not None else None,
            "exchange": self.exchange.snapshot() if self.exchange is not None else None,
        }

    @classmethod
    def restore(cls, world: Any, snapshot: Mapping[str, Any]) -> Biosphere:
        snapshot = copy.deepcopy(snapshot)
        if snapshot.get("format") == "chreatures-biosphere-v1":
            snapshot["format"] = "chreatures-biosphere-v2"
            snapshot["mobility"] = None
            snapshot["material_objects"] = None
        if snapshot.get("format") == "chreatures-biosphere-v2":
            snapshot["format"] = FORMAT
            snapshot["exchange"] = None
        if snapshot.get("format") != FORMAT:
            raise ValueError("unsupported biosphere snapshot")
        mobile_state = snapshot["mobility"]
        instance = cls(
            world,
            MetabolicWeb.restore(snapshot["web"]),
            snapshot["config"],
            mobiles=mobile_state["config"] if mobile_state is not None else None,
        )
        if instance.config_sha256 != snapshot["config_sha256"]:
            raise ValueError("developmental colony configuration differs")
        if set(snapshot["growth"]) != set(instance.growth) or set(
            snapshot["active"]
        ) != set(instance.growth):
            raise ValueError("developmental identity set differs")
        instance.growth = {
            key: GrowthSystem.restore(instance.growth[key].grammar, value)
            for key, value in snapshot["growth"].items()
        }
        instance.parts = copy.deepcopy(snapshot["parts"])
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
        instance._check_structure()
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


__all__ = ["Biosphere"]
