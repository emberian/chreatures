"""Bounded inherited morphology for the fixed six-limb articulated interface."""

from __future__ import annotations

import copy
import math
from typing import Any

TRAIT_BOUNDS = {
    "body_scale": (0.80, 1.25),
    "axial_scale": (0.80, 1.25),
    "radial_scale": (0.80, 1.25),
    "leg_length_scale": (0.75, 1.30),
    "leg_radius_scale": (0.80, 1.25),
    "antenna_length_scale": (0.70, 1.50),
    "antenna_spread_scale": (0.70, 1.45),
    "density_scale": (0.75, 1.30),
    "friction_scale": (0.70, 1.40),
    "torque_scale": (0.75, 1.30),
    "sweep_scale": (0.75, 1.25),
    "frequency_scale": (0.75, 1.25),
}


def resolve_articulation(base: dict[str, Any], traits: Any) -> dict[str, Any]:
    if traits is None:
        traits = {}
    if not isinstance(traits, dict) or set(traits) - set(TRAIT_BOUNDS):
        raise ValueError("articulated_traits contains unknown fields")
    values = {}
    for name, (low, high) in TRAIT_BOUNDS.items():
        value = traits.get(name, 1.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"articulated trait {name} must be finite")
        if not low <= float(value) <= high:
            raise ValueError(f"articulated trait {name} is outside [{low}, {high}]")
        values[name] = float(value)
    result = copy.deepcopy(base)
    body, axial, radial = (
        values["body_scale"],
        values["axial_scale"],
        values["radial_scale"],
    )
    trunk = result["trunk"]
    for part in ("thorax", "head", "abdomen"):
        size = trunk[f"{part}_size"]
        trunk[f"{part}_size"] = [
            size[0] * body * axial,
            size[1] * body * radial,
            size[2] * body * radial,
        ]
    for key in ("head_position", "abdomen_position"):
        trunk[key] = [
            trunk[key][0] * body * axial,
            trunk[key][1] * body,
            trunk[key][2] * body,
        ]
    trunk["density"] *= values["density_scale"]
    trunk["friction"] = [
        value * values["friction_scale"] for value in trunk["friction"]
    ]
    legs = result["legs"]
    for key in ("upper_lateral", "upper_drop", "lower_lateral", "lower_drop"):
        legs[key] *= body * values["leg_length_scale"]
    for key in ("upper_radius", "lower_radius", "tarsus_radius"):
        legs[key] *= body * values["leg_radius_scale"]
    legs["density"] *= values["density_scale"]
    legs["tarsus_density"] *= values["density_scale"]
    legs["tarsus_friction"] = [
        value * values["friction_scale"] for value in legs["tarsus_friction"]
    ]
    for leg in legs["layout"]:
        leg["hip_position"] = [
            leg["hip_position"][0] * body * axial,
            leg["hip_position"][1] * body * radial,
            leg["hip_position"][2] * body,
        ]
    antennae = result["antennae"]
    base_position = antennae["base_position"]
    antennae["base_position"] = [value * body for value in base_position]
    for tip in antennae["tip_positions"]:
        delta = [tip[i] - base_position[i] for i in range(3)]
        tip[:] = [
            antennae["base_position"][0]
            + delta[0] * body * values["antenna_length_scale"],
            antennae["base_position"][1]
            + delta[1] * body * values["antenna_spread_scale"],
            antennae["base_position"][2]
            + delta[2] * body * values["antenna_length_scale"],
        ]
    antennae["radius"] *= body * values["leg_radius_scale"]
    controller = result["controller"]
    controller["max_joint_torque"] *= (
        values["torque_scale"] * values["density_scale"] * body**2
    )
    controller["max_posture_torque"] *= (
        values["torque_scale"] * values["density_scale"] * body**2
    )
    controller["hip_sweep_degrees"] *= values["sweep_scale"]
    controller["frequency_hz"] *= min(
        values["frequency_scale"], values["leg_length_scale"] ** -0.5 * 1.15
    )
    result["inherited_traits"] = values
    return result


__all__ = ["TRAIT_BOUNDS", "resolve_articulation"]
