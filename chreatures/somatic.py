"""Private mobile physiology within the same chemistry as constructed tissue.

The active-force budget is an engineered ATP activation law. Measured mechanical
work is reported separately; no equivalence to a calibrated muscle is implied.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np

from .metabolism import canonical

FORMAT = "chreatures-somatic-physiology-v1"
PARAMETERS = {
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
}


class SomaticPhysiology:
    """Transduce physical encounters into private chemical state and back."""

    def __init__(self, biosphere: Any, config: list[dict[str, Any]]):
        self.biosphere = biosphere
        self.config = copy.deepcopy(config)
        self.sha256 = hashlib.sha256(canonical(self.config)).hexdigest()
        self.residents = {}
        owned_rows = {
            c[k] for c in biosphere.config for k in ("body_row", "structure_row")
        }
        identities = {b.id for b in self.world.bodies}
        for spec in self.config:
            if set(spec) != {"id", "body_row", "gut_row", *PARAMETERS}:
                raise ValueError("mobile physiology fields differ")
            if spec["id"] not in identities or spec["id"] in self.residents:
                raise ValueError(
                    "mobile physiology must identify each physical resident once"
                )
            for key in ("body_row", "gut_row"):
                row = spec[key]
                if (
                    isinstance(row, bool)
                    or not isinstance(row, int)
                    or not 0 <= row < self.web.count
                    or row in owned_rows
                ):
                    raise ValueError("mobile compartments must be valid and private")
                owned_rows.add(row)
            for name in PARAMETERS:
                value = spec[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("mobile physiological parameters must be positive")
            if spec["maximum_bite"] > spec["gut_capacity"]:
                raise ValueError("single bite exceeds gut capacity")
            self.residents[spec["id"]] = spec
        if set(self.residents) != identities:
            raise ValueError("one physiological owner must cover every mobile resident")
        chemistry = self.web.chemistry
        # Abstract material mass is the sum of conserved elemental equivalents.
        self.mass_weights = chemistry._arrays[1].sum(axis=1)
        self.reserve_index = chemistry.pools.index("reserve")
        self.bite_credit = dict.fromkeys(self.residents, 0.0)
        self.paid = {
            key: {"maintenance": 0.0, "activation": 0.0, "unmet": 0.0}
            for key in self.residents
        }
        self.last = {
            key: {
                "absorbed": 0.0,
                "ingested_mass": 0.0,
                "mechanical_work": 0.0,
                "funded_scale": 0.0,
            }
            for key in self.residents
        }
        self.totals = {
            key: {
                "maintenance": 0.0,
                "activation": 0.0,
                "mechanical_work": 0.0,
                "ingested_mass": 0.0,
                "absorbed": 0.0,
            }
            for key in self.residents
        }
        self.pending_dt: float | None = None
        self._outcomes = None

    @property
    def world(self):
        return self.biosphere.world

    @property
    def web(self):
        # A failed prepared topology transaction can restore the web owner.
        return self.biosphere.web

    def normalized(self, resident_id):
        spec = self.residents[resident_id]
        row, gut = spec["body_row"], spec["gut_row"]
        capacity = float(self.web.atp_capacity[row])
        reserve = float(self.web.pools[row, self.reserve_index])
        usable = float(self.web.atp[row]) + 0.72 * reserve
        reference = capacity + 0.72 * spec["reserve_capacity"]
        return {
            "energy": float(np.clip(usable / reference, 0, 1)),
            "gut": float(
                np.clip(
                    self.web.pools[gut] @ self.mass_weights / spec["gut_capacity"], 0, 1
                )
            ),
        }

    def sync_bodies(self, *, validate=False):
        for body in self.world.bodies:
            for key, value in self.normalized(body.id).items():
                if validate and abs(float(getattr(body, key)) - value) > 1e-12:
                    raise ValueError(
                        "saved body readout disagrees with its private chemistry"
                    )
                setattr(body, key, value)

    def begin_step(self, actions, dt):
        if self.pending_dt is not None:
            raise RuntimeError("mobile chemistry has an unfinished physical step")
        if not np.isfinite(dt) or not 0 < dt <= 1:
            raise ValueError("mobile physiological step must be in (0, 1]")
        scales = {}
        for body in self.world.bodies:
            key, spec = body.id, self.residents[body.id]
            action = actions.get(key, {})
            # A quiet articulated servo still exerts posture/holding forces.
            activity = 0.08 + 0.45 * abs(action.get("forward", action.get("thrust", 0)))
            activity += 0.18 * abs(action.get("turn", action.get("yaw", 0)))
            activity += 0.22 * abs(
                action.get("posture", action.get("lift", action.get("vertical", 0)))
            )
            activity += 0.15 * max(0, action.get("grip", 0))
            signal = action.get(
                "signal",
                [action.get(n, 0) for n in ("signal_low", "signal_mid", "signal_high")],
            )
            activity += 0.12 * sum(
                signal if isinstance(signal, (list, tuple)) else [signal]
            )
            requested_maintenance = dt * spec["maintenance_rate"]
            requested_activation = dt * spec["activation_rate"] * activity
            row = spec["body_row"]
            maintenance = min(requested_maintenance, float(self.web.atp[row]))
            self.web.pay_work(row, maintenance)
            activation = min(requested_activation, float(self.web.atp[row]))
            self.web.pay_work(row, activation)
            scale = activation / requested_activation
            scales[key] = scale
            self.paid[key] = {
                "maintenance": maintenance,
                "activation": activation,
                "unmet": (requested_maintenance - maintenance) / requested_maintenance,
            }
            self.last[key] = {
                "absorbed": 0.0,
                "ingested_mass": 0.0,
                "mechanical_work": 0.0,
                "funded_scale": scale,
            }
            for name, amount in (
                ("maintenance", maintenance),
                ("activation", activation),
            ):
                self.totals[key][name] += amount
            eat = max(0, min(1, action.get("eat", 0)))
            self.bite_credit[key] = min(
                spec["maximum_bite"],
                self.bite_credit[key] + dt * spec["bite_rate"] * eat * scale,
            )
        self.pending_dt = float(dt)
        return scales

    def _mouth(self, resident_id):
        geom = mujoco.mj_name2id(
            self.world.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"resident:{resident_id}:geom:head",
        )
        if geom >= 0:
            rotation = self.world.data.geom_xmat[geom].reshape(3, 3)
            return self.world.data.geom_xpos[geom] + rotation[:, 0] * float(
                self.world.model.geom_size[geom, 0]
            )
        body = self.world._body_mj[resident_id]
        rotation = self.world.data.xmat[body].reshape(3, 3)
        return (
            self.world.data.xpos[body]
            + rotation[:, 0] * self.world._body(resident_id).radius
        )

    def finish_step(self, actions, outcomes, contact_samples, dt):
        if self.pending_dt != dt:
            raise RuntimeError("mobile physical/chemical timestep differs")
        self._outcomes = outcomes
        candidates = []
        material_contacts = {}
        part_index = {
            (p["entity"], p["shape_index"]): name
            for name, p in self.biosphere.parts.items()
        }
        for sample in contact_samples:
            key = sample["resident_id"]
            if key not in self.residents or actions.get(key, {}).get("eat", 0) <= 0:
                continue
            distance = float(
                np.linalg.norm(np.asarray(sample["point"]) - self._mouth(key))
            )
            if distance > self.residents[key]["mouth_radius"]:
                continue
            for entity, shape in zip(
                sample["entity_ids"], sample["entity_shape_indices"], strict=True
            ):
                part_id = part_index.get((entity, shape))
                if part_id is not None:
                    candidates.append((distance, key, part_id))
                if (
                    self.biosphere.materials is not None
                    and entity in self.biosphere.materials.donor_rows
                ):
                    contact_key = (key, entity)
                    material_contacts[contact_key] = min(
                        distance, material_contacts.get(contact_key, float("inf"))
                    )
        selected = {}
        capacity = {
            key: max(
                0.0,
                spec["gut_capacity"]
                - float(self.web.pools[spec["gut_row"]] @ self.mass_weights),
            )
            for key, spec in self.residents.items()
        }
        for _, key, part_id in sorted(set(candidates)):
            if part_id in selected:
                continue
            resources = self.web.chemistry.resources(
                self.biosphere.parts[part_id]["resources"]
            )
            mass = float(resources @ self.mass_weights)
            if mass <= 0 or mass > min(capacity[key], self.bite_credit[key]):
                continue
            selected[part_id] = self.residents[key]["gut_row"]
            capacity[key] -= mass
            self.bite_credit[key] -= mass
            self.last[key]["ingested_mass"] += mass
        if selected:
            self.biosphere.transfer_parts(selected)
        if material_contacts:
            requests, recipients = [], []
            remaining = {
                key: min(capacity[key], self.bite_credit[key]) for key in capacity
            }
            for (key, entity), _ in sorted(
                material_contacts.items(), key=lambda pair: (pair[1], pair[0])
            ):
                row = self.biosphere.materials.donor_rows[entity]
                stock = self.web.pools[row]
                mass = float(stock @ self.mass_weights)
                amount = min(mass, remaining[key])
                if amount <= 0:
                    continue
                vector = stock * (amount / mass)
                requests.append(
                    {
                        "entity": entity,
                        "receiver_row": self.residents[key]["gut_row"],
                        "resources": dict(
                            zip(self.web.chemistry.pools, vector.tolist())
                        ),
                    }
                )
                recipients.append(key)
                remaining[key] -= amount
            if requests:
                receipt = self.biosphere.materials.withdraw_batch(requests)
                for key, vector in zip(
                    recipients, receipt["moved_resources"], strict=True
                ):
                    mass = float(np.asarray(vector) @ self.mass_weights)
                    self.bite_credit[key] = max(0.0, self.bite_credit[key] - mass)
                    self.last[key]["ingested_mass"] += mass
        for body in self.world.bodies:
            key, spec = body.id, self.residents[body.id]
            work = float(outcomes[key].get("mechanical_work", 0))
            self.last[key]["mechanical_work"] = work
            self.totals[key]["mechanical_work"] += work
            self.totals[key]["ingested_mass"] += self.last[key]["ingested_mass"]
            drive = max(
                float(outcomes[key]["effort"]) * self.last[key]["funded_scale"],
                self.paid[key]["unmet"],
            )
            body.fatigue = float(
                np.clip(
                    body.fatigue
                    + dt
                    * (
                        spec["fatigue_rise"] * drive
                        - spec["fatigue_recovery"] * (1 - min(1, drive))
                    ),
                    0,
                    1,
                )
            )
            outcomes[key]["nutrition"] = 0.0
            outcomes[key]["ingested_mass"] = self.last[key]["ingested_mass"]
        # Digestion and absorption execute once with the shared web in advance.

    def before_reactions(self, dt):
        if self.pending_dt != dt or self._outcomes is None:
            raise RuntimeError("mobile chemistry requires one completed physical step")
        specs = list(self.residents.values())
        self.web.transfer_batch(
            [s["body_row"] for s in specs],
            [s["gut_row"] for s in specs],
            [{} for _ in specs],
            atp=[dt * s["digestive_atp_rate"] for s in specs],
        )

    def after_reactions(self, dt):
        specs = list(self.residents.values())
        requests = []
        pools = self.web.pools
        for spec in specs:
            reserve = max(
                0.0,
                spec["reserve_capacity"] - pools[spec["body_row"], self.reserve_index],
            )
            requests.append({"reserve": min(dt * spec["absorption_rate"], reserve)})
        receipt = self.web.transfer_batch(
            [s["gut_row"] for s in specs],
            [s["body_row"] for s in specs],
            requests,
            atp=[0.0] * len(specs),
        )
        for spec, moved in zip(specs, receipt["moved_resources"], strict=True):
            key = spec["id"]
            amount = float(moved[self.reserve_index])
            self.last[key]["absorbed"] = amount
            self.totals[key]["absorbed"] += amount
            # Reward evidence is actual absorbed chemical energy, not contact.
            self._outcomes[key]["nutrition"] = amount
        self.sync_bodies()
        self.pending_dt = None
        self._outcomes = None

    def snapshot(self):
        if self.pending_dt is not None:
            raise RuntimeError(
                "cannot save mobile physiology between physical and chemical boundaries"
            )
        return {
            "format": FORMAT,
            "config": copy.deepcopy(self.config),
            "sha256": self.sha256,
            "bite_credit": self.bite_credit.copy(),
            "paid": copy.deepcopy(self.paid),
            "last": copy.deepcopy(self.last),
            "totals": copy.deepcopy(self.totals),
        }

    def restore_state(self, snapshot: Mapping[str, Any]):
        if snapshot.get("format") != FORMAT or snapshot.get("sha256") != self.sha256:
            raise ValueError("mobile physiology identity differs")
        for name in ("bite_credit", "paid", "last", "totals"):
            value = snapshot[name]
            if set(value) != set(self.residents):
                raise ValueError("mobile private state identities differ")
            if name != "bite_credit":
                expected = getattr(self, name)
                if any(
                    not isinstance(row, dict) or set(row) != set(expected[key])
                    for key, row in value.items()
                ):
                    raise ValueError("mobile private state fields differ")
            flat = (
                list(value.values())
                if name == "bite_credit"
                else [v for row in value.values() for v in row.values()]
            )
            if not np.isfinite(flat).all() or min(flat, default=0) < 0:
                raise ValueError("invalid mobile physiological state")
            setattr(self, name, copy.deepcopy(value))
        if any(
            value > self.residents[key]["maximum_bite"]
            for key, value in self.bite_credit.items()
        ):
            raise ValueError("saved bite capacity exceeded")
        if any(row["unmet"] > 1 for row in self.paid.values()) or any(
            row["funded_scale"] > 1 for row in self.last.values()
        ):
            raise ValueError("saved physiological fraction exceeds one")
        self.sync_bodies(validate=True)

    def view(self):
        return {
            "kind": FORMAT,
            "sha256": self.sha256,
            "residents": {
                key: {
                    **self.normalized(key),
                    **self.last[key],
                    "totals": self.totals[key].copy(),
                }
                for key in self.residents
            },
            "actuator_budget": "engineered_activation_law",
            "mechanical_work": "separate_positive_active_force_estimate",
        }
