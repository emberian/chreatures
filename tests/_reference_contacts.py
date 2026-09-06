"""Pinned scalar contact equations retained only for native conformance tests."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def collect_contacts(world: Any, contacted_entities: dict[str, set[str]]) -> None:
    for index in range(world.data.ncon):
        contact = world.data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        world_normal = np.asarray(contact.frame[:3], dtype=float)
        point = np.asarray(contact.pos, dtype=float)
        if world._acoustics is not None:
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(world.model, world.data, index, force)
            velocities = []
            for geom_id in (first, second):
                body_id = int(world.model.geom_bodyid[geom_id])
                linear, angular = world._velocity(body_id)
                radius = point - world.data.xpos[body_id]
                velocities.append(linear + np.asarray([
                    angular[1] * radius[2] - angular[2] * radius[1],
                    angular[2] * radius[0] - angular[0] * radius[2],
                    angular[0] * radius[1] - angular[1] * radius[0],
                ]))
            relative_speed = abs(float(np.dot(velocities[1] - velocities[0], world_normal)))
            impulse = min(
                float(world.spec.get("limits", {}).get("acoustic_impulse", 10.0)),
                abs(float(force[0])) * float(world.model.opt.timestep),
            )
            impact_work = min(
                float(world.spec.get("limits", {}).get("acoustic_work", 5.0)),
                0.5 * impulse * relative_speed,
            )
            for entity_id in {
                value for value in (world._geom_entity.get(first), world._geom_entity.get(second))
                if value is not None
            }:
                world._acoustics.ingest_contact({
                    "entity": entity_id, "position": point.astype(float).tolist(),
                    "normal_impulse": impulse, "relative_normal_speed": relative_speed,
                    "impact_work": impact_work,
                })
        participants = []
        if first in world._geom_resident:
            participants.append((world._geom_resident[first], second, world_normal))
        if second in world._geom_resident:
            participants.append((world._geom_resident[second], first, -world_normal))
        for resident_id, other, normal in participants:
            entity_id = world._geom_entity.get(other)
            if entity_id:
                contacted_entities[resident_id].add(entity_id)
            body = world._body(resident_id)
            mj_body = world._body_mj[resident_id]
            if abs(float(normal[2])) > 0.72 and point[2] < body.z:
                continue
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(world.model, world.data, index, force)
            strength = min(1.0, 0.18 + float(np.linalg.norm(force[:3])) / 3.0)
            rotation = world.data.xmat[mj_body].reshape(3, 3)
            delta = point - world.data.xpos[mj_body]
            side = 1 if float(np.dot(delta, rotation[:, 1])) >= 0 else 0
            world._touch[resident_id][side] = max(world._touch[resident_id][side], strength)
            if len(world._contact_normals[resident_id]) < 8:
                world._contact_normals[resident_id].append((rotation.T @ normal).astype(float).tolist())
            if entity_id:
                world._resonance[entity_id] = max(world._resonance.get(entity_id, 0.0), strength)
