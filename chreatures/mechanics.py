"""Validation and compilation for passive one-axis mechanical assemblies.

Assemblies connect existing physical joint coordinates. They introduce no
controller action or semantic trigger: MuJoCo contact forces, gravity and the
constraint solver determine their motion.
"""

from __future__ import annotations

import copy
import html
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")
_FIELDS = {
    "id", "type", "joint_a", "joint_b", "offset", "ratio", "solref", "solimp",
}


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _vector(value: Any, length: int, name: str, low: float, high: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [_number(item, name, low, high) for item in value]


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


def normalize_assemblies(raw: Any, entities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a strict normalized list of passive joint couplings."""
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 32 or any(not isinstance(item, dict) for item in raw):
        raise ValueError("assemblies must be a list of at most 32 mappings")
    by_id = {str(entity.get("id")): entity for entity in entities}
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    participants: set[str] = set()
    for item in raw:
        unknown = set(item) - _FIELDS
        if unknown:
            raise ValueError(f"unknown assembly fields: {sorted(unknown)}")
        assembly_id = _identifier(item.get("id"), "assembly id")
        if assembly_id in ids:
            raise ValueError("assembly ids must be unique")
        if item.get("type") != "joint_coupling":
            raise ValueError("assembly type must be joint_coupling")
        joint_a = _identifier(item.get("joint_a"), "assembly joint_a")
        joint_b = _identifier(item.get("joint_b"), "assembly joint_b")
        if joint_a == joint_b:
            raise ValueError("an assembly must connect two distinct joints")
        for entity_id in (joint_a, joint_b):
            entity = by_id.get(entity_id)
            if entity is None or entity.get("mobility") not in {"hinge", "slide"}:
                raise ValueError(f"assembly joint {entity_id!r} must name a hinge or slide entity")
            if entity_id in participants:
                raise ValueError("a joint can participate in only one passive coupling")
        offset = _number(item.get("offset", 0.0), "assembly offset", -20.0, 20.0)
        ratio = _number(item.get("ratio", 1.0), "assembly ratio", -100.0, 100.0)
        if abs(ratio) < 1e-8:
            raise ValueError("assembly ratio cannot be zero")
        initial_a = float(by_id[joint_a].get("joint", {}).get("initial", 0.0))
        initial_b = float(by_id[joint_b].get("joint", {}).get("initial", 0.0))
        if by_id[joint_a]["mobility"] == "hinge":
            initial_a = math.radians(initial_a)
        if by_id[joint_b]["mobility"] == "hinge":
            initial_b = math.radians(initial_b)
        if abs(initial_a - (offset + ratio * initial_b)) > 1e-8:
            raise ValueError("assembly joint initial positions violate the coupling")
        solref = _vector(item.get("solref", [0.01, 1.0]), 2, "assembly solref", 1e-6, 100.0)
        solimp = _vector(item.get("solimp", [0.95, 0.99, 0.001, 0.5, 2.0]), 5, "assembly solimp", 0.0, 100.0)
        if not 0.0 <= solimp[0] < solimp[1] <= 1.0 or solimp[2] <= 0.0 or not 0.0 <= solimp[3] <= 1.0 or solimp[4] < 1.0:
            raise ValueError("assembly solimp is outside MuJoCo's stable parameter domain")
        result.append({
            "id": assembly_id, "type": "joint_coupling", "joint_a": joint_a,
            "joint_b": joint_b, "offset": offset, "ratio": ratio,
            "solref": solref, "solimp": solimp,
        })
        ids.add(assembly_id)
        participants.update((joint_a, joint_b))
    return result


def equality_xml(
    assemblies: Sequence[Mapping[str, Any]], mobilities: Mapping[str, str],
) -> str:
    """Compile normalized assemblies to MuJoCo equality constraints."""
    if not assemblies:
        return ""
    records = []
    for assembly in assemblies:
        values = {
            "name": f"assembly:{assembly['id']}",
            "joint1": f"entity:{assembly['joint_a']}:{mobilities[assembly['joint_a']]}",
            "joint2": f"entity:{assembly['joint_b']}:{mobilities[assembly['joint_b']]}",
            "polycoef": [assembly["offset"], assembly["ratio"], 0.0, 0.0, 0.0],
            "solref": assembly["solref"], "solimp": assembly["solimp"],
        }
        attrs = " ".join(
            f'{key}="{html.escape(_attribute(value), quote=True)}"' for key, value in values.items()
        )
        records.append(f"<joint {attrs}/>")
    return f"<equality>{''.join(records)}</equality>"


def _attribute(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(f"{float(item):.10g}" for item in value)
    return str(value)


def assembly_view(
    assemblies: Sequence[Mapping[str, Any]], model: Any, data: Any,
    joint_ids: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Expose physical assembly coordinates for external rendering/debugging."""
    result = []
    for assembly in assemblies:
        coordinates = {}
        for field in ("joint_a", "joint_b"):
            entity_id = str(assembly[field])
            joint_id = joint_ids[entity_id]
            qadr = int(model.jnt_qposadr[joint_id])
            dadr = int(model.jnt_dofadr[joint_id])
            coordinates[field] = {
                "entity": entity_id,
                "position": float(data.qpos[qadr]),
                "velocity": float(data.qvel[dadr]),
            }
        error = coordinates["joint_a"]["position"] - (
            float(assembly["offset"]) + float(assembly["ratio"]) * coordinates["joint_b"]["position"]
        )
        result.append({
            **copy.deepcopy(dict(assembly)), "coordinates": coordinates,
            "constraint_error": error,
        })
    return result


__all__ = ["normalize_assemblies", "equality_xml", "assembly_view"]
