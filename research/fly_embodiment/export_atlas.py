#!/usr/bin/env python3
"""Export the MaleCNS-to-NeuroMechFly anatomical support atlas.

This build-time tool joins author annotations to author body identities.  It
does not estimate decoder weights, muscle recruitment, or baseline firing
rates.  Its masks only say which learned connections are anatomically allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MOTOR_SUPERCLASSES = ("cb_motor", "vnc_motor")
MOTOR_SUBCLASSES = ("ad", "am", "fl", "hl", "hm", "ml", "nm", "pm", "rm", "wm", "xm")
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_SUBCLASS = {"lf": "fl", "lm": "ml", "lh": "hl", "rf": "fl", "rm": "ml", "rh": "hl"}
LEG_SIDE = {leg: ("L" if leg[0] == "l" else "R") for leg in LEGS}
UINT32_SENTINEL = np.iinfo(np.uint32).max
BODY_IDENTITY = "neuromechfly-2.1.0-ca65a510-ypr"
ATLAS_SCHEMA = "chreatures.fly-body-neural-atlas.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def side_mask(sides: np.ndarray, side: str | None) -> np.ndarray:
    return np.ones(sides.shape, dtype=bool) if side is None else sides == side


def actuator_leg(target: str) -> str | None:
    for leg in LEGS:
        if target.startswith(f"{leg}_") or f"-{leg}_" in target:
            return leg
    return None


def actuator_side(target: str) -> str | None:
    if "-l_" in target or target.startswith("l_"):
        return "L"
    if "-r_" in target or target.startswith("r_"):
        return "R"
    return None


def motor_support_mask(
    actuators: list[dict[str, object]],
    subclasses: np.ndarray,
    sides: np.ndarray,
    motor_types: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    mask = np.zeros((len(actuators), len(subclasses)), dtype=np.float32)
    grades: list[str] = []
    bases: list[str] = []
    for row, actuator in enumerate(actuators):
        group = str(actuator["control_group"])
        target = str(actuator["target"])
        if group == "walking":
            leg = actuator_leg(target)
            if leg is None:
                raise ValueError(f"cannot locate leg in walking target {target}")
            selected = (subclasses == LEG_SUBCLASS[leg]) & (sides == LEG_SIDE[leg])
            basis = f"same anatomical leg cohort {leg}"
        elif group == "head":
            selected = subclasses == "nm"
            basis = "neck-motor family to central head axes"
        elif group == "pedicels":
            side = actuator_side(target)
            selected = (subclasses == "am") & side_mask(sides, side)
            basis = f"antenna-motor family, side {side}"
        elif group == "proboscis":
            # Separate author-identified pump and salivary neurons from the
            # external position servos.  Systematic/unknown pm rows stay here:
            # they have a known body family but no finer defensible target.
            internal = np.isin(motor_types, ("MN10", "MN11D", "MN11V", "MN12D", "MN13"))
            selected = (subclasses == "pm") & ~internal
            basis = "external or finer-unresolved proboscis-motor family to central mouthpart axes"
        elif group == "abdomen":
            selected = subclasses == "ad"
            basis = "abdominal-motor family to central abdominal axes"
        elif group == "wings":
            side = actuator_side(target)
            selected = (subclasses == "wm") & side_mask(sides, side)
            basis = f"wing-motor family, side {side}"
        elif group == "halteres":
            side = actuator_side(target)
            selected = (subclasses == "hm") & side_mask(sides, side)
            basis = f"haltere-motor family, side {side}"
        elif group == "adhesion":
            leg = actuator_leg(target)
            if leg is None:
                raise ValueError(f"cannot locate leg in adhesion target {target}")
            selected = (subclasses == LEG_SUBCLASS[leg]) & (sides == LEG_SIDE[leg])
            basis = f"engineered adhesion control from same anatomical leg cohort {leg}"
        elif group == "pharyngeal_pump":
            selected = (subclasses == "pm") & np.isin(motor_types, ("MN10", "MN11D", "MN11V", "MN12D"))
            basis = "author-identified pharyngeal pump motor types"
        elif group == "salivary_drive":
            selected = (subclasses == "pm") & (motor_types == "MN13")
            basis = "author-identified salivary motor type MN13"
        else:
            raise ValueError(f"unknown actuator group {group}")
        mask[row, selected] = 1.0
        grades.append("engineered" if group == "adhesion" else "inter_animal_inferred")
        bases.append(basis)
    return mask, grades, bases


def is_descriptive_motor_type(subclass: str, manc_type: str) -> bool:
    if not manc_type or manc_type.startswith("MN"):
        return False
    if subclass in {"fl", "ml", "hl", "wm"}:
        return True
    return subclass == "hm" and manc_type in {"hDVM MN", "hi1 MN", "hi2 MN", "hiii2 MN"}


def fine_motor_cohorts(
    subclasses: np.ndarray, sides: np.ndarray, manc_types: np.ndarray
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    keys: list[tuple[str, str, str]] = []
    for leg in LEGS:
        sub, side = LEG_SUBCLASS[leg], LEG_SIDE[leg]
        types = sorted(
            {
                str(t)
                for s, sd, t in zip(subclasses, sides, manc_types, strict=True)
                if s == sub and sd == side and is_descriptive_motor_type(str(s), str(t))
            }
        )
        keys.extend((leg, side, motor_type) for motor_type in types)
    for side in ("L", "R"):
        types = sorted(
            {
                str(t)
                for s, sd, t in zip(subclasses, sides, manc_types, strict=True)
                if s == "wm" and sd == side and is_descriptive_motor_type("wm", str(t))
            }
        )
        keys.extend(("wing", side, motor_type) for motor_type in types)
    for side in ("L", "R"):
        types = sorted(
            {
                str(t)
                for s, sd, t in zip(subclasses, sides, manc_types, strict=True)
                if s == "hm" and sd == side and is_descriptive_motor_type("hm", str(t))
            }
        )
        keys.extend(("haltere", side, motor_type) for motor_type in types)

    names = [f"{part}/{side}/{motor_type}" for part, side, motor_type in keys]
    targets = [motor_type for _, _, motor_type in keys]
    mask = np.zeros((len(keys), len(subclasses)), dtype=np.uint8)
    row_index = np.full(len(subclasses), UINT32_SENTINEL, dtype=np.uint32)
    for cohort, (part, side, motor_type) in enumerate(keys):
        sub = LEG_SUBCLASS[part] if part in LEGS else ("wm" if part == "wing" else "hm")
        selected = (subclasses == sub) & (sides == side) & (manc_types == motor_type)
        mask[cohort, selected] = 1
        row_index[selected] = cohort
    return names, targets, mask, row_index


def sensory_rows(superclasses: np.ndarray) -> np.ndarray:
    # ol_sensory contains optic-lobe visual neurons.  Other sensory superclasses
    # include both peripheral and ascending afferents and must stay available.
    return np.flatnonzero(
        np.char.find(superclasses.astype("U"), "sensory") >= 0
    )[superclasses[np.char.find(superclasses.astype("U"), "sensory") >= 0] != "ol_sensory"].astype(np.uint32)


def build_afferent_ports(
    classes: np.ndarray,
    subclasses: np.ndarray,
    nerves: np.ndarray,
    sides: np.ndarray,
) -> tuple[list[str], list[str], list[str], np.ndarray]:
    specs: list[tuple[str, str, str, np.ndarray]] = []

    def add(name: str, modality: str, grade: str, selected: np.ndarray) -> None:
        specs.append((name, modality, grade, selected))

    for side, label in (("L", "left"), ("R", "right")):
        sided = sides == side
        antenna = nerves == "AN"
        palp = nerves == "MxLbN"
        pharynx = np.isin(nerves, ("PhN", "aPhN"))
        for modality, class_name in (
            ("olfaction", "olfactory"),
            ("humidity", "hygrosensory"),
            ("temperature", "thermosensory"),
        ):
            add(f"antenna/{label}/{modality}", modality, "measured", antenna & sided & (classes == class_name))
        add("antenna/%s/audition" % label, "audition", "measured", antenna & sided & (subclasses == "auditory"))
        add("antenna/%s/wind_gravity" % label, "mechanosensation", "measured", antenna & sided & (subclasses == "wind_gravity"))
        add("antenna/%s/contact" % label, "mechanosensation", "inter_animal_inferred", antenna & sided & np.char.startswith(classes, "mechanosensory"))
        add("maxillary_palp/%s/olfaction" % label, "olfaction", "measured", palp & sided & (classes == "olfactory"))
        add("proboscis/%s/taste" % label, "gustation", "measured", palp & sided & np.isin(classes, ("gustatory", "chemosensory")))
        add("proboscis/%s/contact" % label, "mechanosensation", "inter_animal_inferred", palp & sided & np.char.startswith(classes, "mechanosensory"))
        add("pharynx/%s/taste" % label, "gustation", "measured", pharynx & sided & np.isin(classes, ("gustatory", "chemosensory")))
        add("pharynx/%s/contact" % label, "mechanosensation", "inter_animal_inferred", pharynx & sided & np.char.startswith(classes, "mechanosensory"))

    leg_nerves = {"lf": "ProLN", "lm": "MesoLN", "lh": "MetaLN", "rf": "ProLN", "rm": "MesoLN", "rh": "MetaLN"}
    for leg in LEGS:
        selected_leg = (nerves == leg_nerves[leg]) & (sides == LEG_SIDE[leg])
        add(f"leg/{leg}/chordotonal", "joint_proprioception", "measured", selected_leg & (subclasses == "chordotonal organ"))
        add(f"leg/{leg}/hair_plate", "joint_proprioception", "measured", selected_leg & (subclasses == "hair plate"))
        add(f"leg/{leg}/campaniform", "load_proprioception", "measured", selected_leg & (subclasses == "campaniform sensilla"))
        add(f"leg/{leg}/contact", "touch", "inter_animal_inferred", selected_leg & np.isin(subclasses, ("leg", "leg bristle", "mechanosensory bristle")))
        add(f"leg/{leg}/taste", "gustation", "inter_animal_inferred", selected_leg & np.isin(classes, ("gustatory", "chemosensory")))
    for side, label in (("L", "left"), ("R", "right")):
        sided = sides == side
        add(f"wing/{label}/mechanosensation", "mechanosensation", "inter_animal_inferred", sided & np.isin(subclasses, ("wing", "wing bristle")))
        add(f"haltere/{label}/mechanosensation", "mechanosensation", "inter_animal_inferred", sided & (subclasses == "haltere"))
        add(f"neck/{label}/proprioception", "joint_proprioception", "inter_animal_inferred", sided & ((subclasses == "neck") | (nerves == "PrN")))
        add(f"abdomen/{label}/mechanosensation", "mechanosensation", "inter_animal_inferred", sided & (subclasses == "abdomen"))

    mask = np.stack([spec[3] for spec in specs], axis=0).astype(np.uint8)
    return [s[0] for s in specs], [s[1] for s in specs], [s[2] for s in specs], mask


def joint_afferent_masks(
    joints: list[dict[str, object]],
    classes: np.ndarray,
    subclasses: np.ndarray,
    nerves: np.ndarray,
    sides: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    position = np.zeros((len(joints), len(classes)), dtype=np.uint8)
    velocity = np.zeros_like(position)
    load = np.zeros_like(position)
    grades: list[str] = []
    leg_nerves = {"lf": "ProLN", "lm": "MesoLN", "lh": "MetaLN", "rf": "ProLN", "rm": "MesoLN", "rh": "MetaLN"}
    for index, joint in enumerate(joints):
        joint_id = str(joint["id"])
        child = str(joint["child"])
        leg = actuator_leg(joint_id)
        if leg is not None:
            cohort = (nerves == leg_nerves[leg]) & (sides == LEG_SIDE[leg])
            chord = cohort & (subclasses == "chordotonal organ")
            hair = cohort & (subclasses == "hair plate")
            position[index] = chord | hair
            velocity[index] = chord
            load[index] = cohort & (subclasses == "campaniform sensilla")
            grades.append("inter_animal_inferred")
        elif child == "c_head":
            cohort = (subclasses == "neck") | (nerves == "PrN")
            position[index] = cohort
            velocity[index] = cohort
            grades.append("inter_animal_inferred")
        elif "pedicel" in child or "funiculus" in child or "arista" in child:
            side = "L" if child.startswith("l_") else "R"
            cohort = (nerves == "AN") & (sides == side) & np.char.startswith(classes, "mechanosensory")
            position[index] = cohort
            velocity[index] = cohort
            grades.append("inter_animal_inferred")
        elif child in {"c_rostrum", "c_haustellum"}:
            cohort = np.isin(nerves, ("MxLbN", "PhN", "aPhN")) & np.char.startswith(classes, "mechanosensory")
            position[index] = cohort
            velocity[index] = cohort
            grades.append("inter_animal_inferred")
        elif child.startswith("c_abdomen"):
            cohort = subclasses == "abdomen"
            position[index] = cohort
            velocity[index] = cohort
            load[index] = cohort & (classes == "mechanosensory_proprioceptive")
            grades.append("inter_animal_inferred")
        elif child.endswith("_wing"):
            side = "L" if child.startswith("l_") else "R"
            cohort = (sides == side) & np.isin(subclasses, ("wing", "wing bristle"))
            position[index] = cohort
            velocity[index] = cohort
            grades.append("inter_animal_inferred")
        elif child.endswith("_haltere"):
            side = "L" if child.startswith("l_") else "R"
            cohort = (sides == side) & (subclasses == "haltere")
            position[index] = cohort
            velocity[index] = cohort
            grades.append("inter_animal_inferred")
        else:
            # Eye rotations have no identified proprioceptor mapping in this
            # atlas.  Keeping an all-zero row is an explicit unsupported claim.
            grades.append("unsupported")
    return position, velocity, load, grades


def body807(
    body: dict[str, object],
    actuators: list[dict[str, object]],
    classes: np.ndarray,
    subclasses: np.ndarray,
    nerves: np.ndarray,
    sides: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    joint_load: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Build the frozen physical BODY807 channel order and afferent support."""
    channels: list[dict[str, object]] = []
    masks: list[np.ndarray] = []

    def add(
        name: str,
        unit: str,
        raw_range: tuple[float | None, float | None],
        evidence: str,
        basis: str,
        selected: np.ndarray,
    ) -> None:
        channels.append(
            {
                "index": len(channels),
                "name": name,
                "unit": unit,
                "raw_range": list(raw_range),
                "normalized_range": [-1.0, 1.0],
                "evidence_grade": evidence,
                "basis": basis,
                "supported_afferent_rows": int(np.count_nonzero(selected)),
            }
        )
        masks.append(selected.astype(np.uint8))

    olfactory_sites = (
        ("antenna_l", (nerves == "AN") & (sides == "L") & (classes == "olfactory")),
        ("antenna_r", (nerves == "AN") & (sides == "R") & (classes == "olfactory")),
        ("maxillary_palp_l", (nerves == "MxLbN") & (sides == "L") & (classes == "olfactory")),
        ("maxillary_palp_r", (nerves == "MxLbN") & (sides == "R") & (classes == "olfactory")),
    )
    for site, selected in olfactory_sites:
        for feature in range(8):
            add(
                f"olfaction/{site}/chemical_{feature}",
                "effective_concentration",
                (0.0, None),
                "engineered",
                "local physical chemical feature projected to all author-annotated olfactory afferents at the same organ and side",
                selected,
            )

    mouth_gustatory = np.isin(nerves, ("MxLbN", "PhN", "aPhN")) & np.isin(classes, ("gustatory", "chemosensory"))
    for feature in range(8):
        add(
            f"gustation/mouth/chemical_{feature}",
            "effective_concentration",
            (0.0, None),
            "engineered",
            "mouth-local chemical feature projected to author-annotated proboscis and pharyngeal taste afferents",
            mouth_gustatory,
        )

    for side, label in (("L", "left"), ("R", "right")):
        wind = (nerves == "AN") & (sides == side) & (subclasses == "wind_gravity")
        for axis in "xyz":
            add(
                f"airflow/antenna_{label}/{axis}",
                "mm/s",
                (None, None),
                "inter_animal_inferred",
                "antenna-local airflow projected to same-side wind/gravity afferents",
                wind,
            )

    auditory = (nerves == "AN") & (subclasses == "auditory")
    band_edges = np.geomspace(40.0, 1600.0, 16)
    for index, hz in enumerate(band_edges):
        add(
            f"audition/effective_band_{index:02d}_{hz:.3f}hz",
            "effective_amplitude",
            (0.0, None),
            "engineered",
            "effective acoustic band projected to all author-annotated auditory afferents; no fabricated frequency tuning per row",
            auditory,
        )

    inertial = (subclasses == "haltere") | (subclasses == "campaniform sensilla")
    for name, unit in (
        ("thorax/linear_velocity_x", "mm/s"),
        ("thorax/linear_velocity_y", "mm/s"),
        ("thorax/linear_velocity_z", "mm/s"),
        ("thorax/angular_velocity_x", "rad/s"),
        ("thorax/angular_velocity_y", "rad/s"),
        ("thorax/angular_velocity_z", "rad/s"),
    ):
        add(name, unit, (None, None), "engineered", "thorax-local inertial transduction to haltere and campaniform cohorts", inertial)
    add(
        "thorax/local_irradiance",
        "effective_irradiance",
        (0.0, None),
        "unsupported",
        "nonvisual afferent table has no defensible irradiance receptor mapping; optic input is a separate interface",
        np.zeros(len(classes), dtype=bool),
    )

    internal_names = (
        "atp_fraction",
        "gut_fullness",
        "carbon_reserve_fraction",
        "nitrogen_reserve_fraction",
        "hydration",
        "structural_fraction",
        "salivary_reservoir_fraction",
        "oxygen_availability",
        "maintenance_shortfall",
        "acclimation_effort",
        "reproductive_investment_fraction",
        "internal_concentration_deviation",
    )
    unknown_internal = classes == "unknown_sensory"
    for name in internal_names:
        add(
            f"interoception/{name}",
            "fraction" if "fraction" in name or name in {"gut_fullness", "hydration", "oxygen_availability"} else "normalized_effective_state",
            (0.0, 1.0) if name != "internal_concentration_deviation" else (None, None),
            "engineered",
            "engineered physiology transduction to unresolved sensory rows; no receptor identity claimed",
            unknown_internal,
        )

    joints = body["joint_dofs"]
    for matrix, signal, unit in (
        (joint_position, "position", "rad"),
        (joint_velocity, "angular_velocity", "rad/s"),
        (joint_load, "generalized_load", "model_force_or_torque"),
    ):
        for joint, selected in zip(joints, matrix, strict=True):
            supported = bool(np.any(selected))
            add(
                f"joint/{joint['id']}/{signal}",
                unit,
                (None, None),
                "inter_animal_inferred" if supported else "unsupported",
                "body-axis signal projected to the anatomically matched afferent cohort" if supported else "no defensible annotated afferent cohort for this axis and modality",
                selected.astype(bool),
            )

    leg_nerves = {"lf": "ProLN", "lm": "MesoLN", "lh": "MetaLN", "rf": "ProLN", "rm": "MesoLN", "rh": "MetaLN"}
    for leg in LEGS:
        cohort = (nerves == leg_nerves[leg]) & (sides == LEG_SIDE[leg])
        contact = cohort & (np.char.startswith(classes, "mechanosensory") | (subclasses == "leg") | (subclasses == "leg bristle"))
        force = contact | (cohort & (subclasses == "campaniform sensilla"))
        for axis in "xyz":
            add(f"foot/{leg}/contact_force_{axis}", "model_force", (None, None), "engineered", "measured simulated foot force projected to same-leg touch/load afferents", force)
        for axis in "xyz":
            add(f"foot/{leg}/slip_velocity_{axis}", "mm/s", (None, None), "engineered", "surface-relative slip projected to same-leg tactile afferents", contact)

    def segment_cohort(segment: str) -> np.ndarray:
        leg = actuator_leg(segment)
        if leg is not None:
            cohort = (nerves == leg_nerves[leg]) & (sides == LEG_SIDE[leg])
            return cohort & np.char.startswith(classes, "mechanosensory")
        if segment == "c_thorax":
            return (subclasses == "notum") | (subclasses == "campaniform sensilla")
        if segment == "c_head":
            return (subclasses == "mechanosensory bristle") & np.isin(nerves, ("AN", "MxLbN", "PrN"))
        if segment in {"c_rostrum", "c_haustellum"}:
            return np.isin(nerves, ("MxLbN", "PhN", "aPhN")) & np.char.startswith(classes, "mechanosensory")
        if segment.startswith("c_abdomen"):
            return subclasses == "abdomen"
        if segment.endswith("_wing"):
            side = "L" if segment.startswith("l_") else "R"
            return (sides == side) & np.isin(subclasses, ("wing", "wing bristle"))
        if segment.endswith("_haltere"):
            side = "L" if segment.startswith("l_") else "R"
            return (sides == side) & (subclasses == "haltere")
        if any(token in segment for token in ("pedicel", "funiculus", "arista")):
            side = "L" if segment.startswith("l_") else "R"
            return (nerves == "AN") & (sides == side) & np.char.startswith(classes, "mechanosensory")
        return np.zeros(len(classes), dtype=bool)

    for segment in body["body_segments"]:
        selected = segment_cohort(str(segment["id"]))
        for axis in "xyz":
            supported = bool(np.any(selected))
            add(
                f"segment/{segment['id']}/contact_force_{axis}",
                "model_force",
                (None, None),
                "engineered" if supported else "unsupported",
                "simulated segment contact projected to anatomically matched tactile/load afferents" if supported else "no defensible nonvisual contact-afferent cohort",
                selected,
            )

    for side, label in (("L", "left"), ("R", "right")):
        selected = (sides == side) & (subclasses == "haltere")
        for axis in "xyz":
            add(f"haltere/{label}/inertial_load_{axis}", "effective_inertial_load", (None, None), "engineered", "same-side effective haltere load transduction", selected)

    mouth_mech = np.isin(nerves, ("MxLbN", "PhN", "aPhN")) & np.char.startswith(classes, "mechanosensory")
    for axis in "xyz":
        add(f"mouth/contact_normal_{axis}", "unitless", (-1.0, 1.0), "engineered", "mouth-local simulated contact normal projected to mouth mechanosensory afferents", mouth_mech)
    add("mouth/contact_fraction", "fraction", (0.0, 1.0), "engineered", "measured simulated mouth contact fraction projected to mouth mechanosensory afferents", mouth_mech)

    for actuator in actuators:
        group, target = str(actuator["control_group"]), str(actuator["target"])
        leg = actuator_leg(target)
        if leg is not None:
            selected = (nerves == leg_nerves[leg]) & (sides == LEG_SIDE[leg]) & (classes == "mechanosensory_proprioceptive")
        elif group == "head":
            selected = (subclasses == "neck") | (nerves == "PrN")
        elif group == "pedicels":
            selected = (nerves == "AN") & side_mask(sides, actuator_side(target)) & np.char.startswith(classes, "mechanosensory")
        elif group in {"proboscis", "pharyngeal_pump", "salivary_drive"}:
            selected = mouth_mech
        elif group == "abdomen":
            selected = subclasses == "abdomen"
        elif group == "wings":
            selected = side_mask(sides, actuator_side(target)) & np.isin(subclasses, ("wing", "wing bristle"))
        elif group == "halteres":
            selected = side_mask(sides, actuator_side(target)) & (subclasses == "haltere")
        else:
            selected = np.zeros(len(classes), dtype=bool)
        add(f"fatigue/{actuator['id']}", "fraction", (0.0, 1.0), "engineered", "persistent effective actuator fatigue projected to same-body-part proprioceptive cohort", selected)

    pharyngeal = np.isin(nerves, ("PhN", "aPhN")) & np.char.startswith(classes, "mechanosensory")
    add("pump/phase_sine", "unitless", (-1.0, 1.0), "engineered", "native pump phase transduction to pharyngeal mechanosensory afferents", pharyngeal)
    add("pump/phase_cosine", "unitless", (-1.0, 1.0), "engineered", "native pump phase transduction to pharyngeal mechanosensory afferents", pharyngeal)
    add("pump/transferred_flow", "model_material_per_control_interval", (0.0, None), "engineered", "actual transferred flow transduction to pharyngeal mechanosensory afferents", pharyngeal)

    if len(channels) != 807:
        raise RuntimeError(f"BODY807 construction yielded {len(channels)} channels")
    return channels, np.stack(masks, axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=Path, required=True)
    parser.add_argument("--body-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--channel-schema", type=Path)
    parser.add_argument("--motor-schema", type=Path)
    parser.add_argument("--physical-fixture", type=Path)
    args = parser.parse_args()

    body = json.loads(args.body_schema.read_text())
    if body.get("identity") != BODY_IDENTITY:
        raise SystemExit(f"expected body identity {BODY_IDENTITY}, found {body.get('identity')}")
    actuators = list(body["actuators"])
    joints = body["joint_dofs"]
    if len(actuators) != 90 or len(joints) != 126:
        raise SystemExit(f"expected M90 and 126 joints, found {len(actuators)} and {len(joints)}")
    if [a["index"] for a in actuators] != list(range(90)):
        raise SystemExit("body actuator indices are not canonical")
    actuators.extend(
        [
            {
                "index": 90,
                "id": "pharyngeal-pump-drive",
                "target": "pharyngeal_pump",
                "kind": "physiology_drive",
                "control_group": "pharyngeal_pump",
                "control_range": [0.0, 1.0],
            },
            {
                "index": 91,
                "id": "salivary-drive",
                "target": "salivary_gland",
                "kind": "physiology_drive",
                "control_group": "salivary_drive",
                "control_range": [0.0, 1.0],
            },
        ]
    )
    motor_schema_path = (
        args.motor_schema or Path(__file__).resolve().parent / "motor92-channel-schema.json"
    ).resolve()
    motor_schema = {
        "schema": "chreatures.motor-output-channels.v1",
        "identity": "MOTOR92",
        "length": 92,
        "body_identity": BODY_IDENTITY,
        "body_schema_sha256": sha256(args.body_schema),
        "blocks": [
            {"range": [0, 84], "name": "position_target", "activation": "tanh", "normalized_range": [-1.0, 1.0]},
            {"range": [84, 90], "name": "adhesion", "activation": "sigmoid", "normalized_range": [0.0, 1.0]},
            {"range": [90, 91], "name": "pharyngeal_pump", "activation": "sigmoid", "normalized_range": [0.0, 1.0]},
            {"range": [91, 92], "name": "salivary_drive", "activation": "sigmoid", "normalized_range": [0.0, 1.0]},
        ],
        "channels": [
            {
                "index": int(actuator["index"]),
                "id": actuator["id"],
                "target": actuator["target"],
                "group": actuator["control_group"],
                "kind": actuator["kind"],
                "normalized_range": [-1.0, 1.0] if int(actuator["index"]) < 84 else [0.0, 1.0],
                "physical_range": actuator["control_range"],
                "activation": "tanh" if int(actuator["index"]) < 84 else "sigmoid",
                "physical_boundary": "MJCF" if int(actuator["index"]) < 90 else "native_physiology",
            }
            for actuator in actuators
        ],
    }
    motor_schema_path.write_text(json.dumps(motor_schema, indent=2) + "\n")
    cross_schema_identity: dict[str, dict[str, str]] = {}

    with np.load(args.neurons, allow_pickle=False) as neurons:
        superclasses = neurons["superclasses"]
        global_motor_rows = np.flatnonzero(np.isin(superclasses, MOTOR_SUPERCLASSES)).astype(np.uint32)
        if len(global_motor_rows) != 815:
            raise SystemExit(f"expected 815 MaleCNS motors, found {len(global_motor_rows)}")
        motor_subclasses = neurons["subclasses"][global_motor_rows]
        motor_sides = neurons["sides"][global_motor_rows]
        if set(motor_subclasses) != set(MOTOR_SUBCLASSES):
            raise SystemExit(f"unexpected motor subclasses {sorted(set(motor_subclasses))}")
        motor_manc_types = neurons["manc_types"][global_motor_rows]
        motor_types = neurons["types"][global_motor_rows]
        motor_mask, motor_grades, motor_bases = motor_support_mask(
            actuators, motor_subclasses, motor_sides, motor_types
        )
        cohort_names, cohort_targets, cohort_mask, cohort_index = fine_motor_cohorts(
            motor_subclasses, motor_sides, motor_manc_types
        )
        if cohort_mask.shape != (156, 815) or int(cohort_mask.sum()) != 400:
            raise SystemExit(f"fine cohort invariant failed: shape={cohort_mask.shape}, sum={cohort_mask.sum()}")

        global_afferent_rows = sensory_rows(superclasses)
        if len(global_afferent_rows) != 11798:
            raise SystemExit(f"expected 11798 nonvisual afferents, found {len(global_afferent_rows)}")
        aff_classes = neurons["classes"][global_afferent_rows]
        aff_subclasses = neurons["subclasses"][global_afferent_rows]
        aff_nerves = neurons["entry_nerves"][global_afferent_rows]
        aff_sides = neurons["sides"][global_afferent_rows]
        port_names, port_modalities, port_grades, port_mask = build_afferent_ports(
            aff_classes, aff_subclasses, aff_nerves, aff_sides
        )
        joint_position, joint_velocity, joint_load, joint_grades = joint_afferent_masks(
            joints, aff_classes, aff_subclasses, aff_nerves, aff_sides
        )
        body_channels, body_mask = body807(
            body,
            actuators,
            aff_classes,
            aff_subclasses,
            aff_nerves,
            aff_sides,
            joint_position,
            joint_velocity,
            joint_load,
        )

        channel_schema_path = (
            args.channel_schema or Path(__file__).resolve().parent / "body807-channel-schema.json"
        ).resolve()
        channel_schema = {
            "schema": "chreatures.body-afferent-channels.v1",
            "identity": "BODY807",
            "length": 807,
            "normalization": "Fit per-channel center and positive scale on training worlds and freeze them in the CNS artifact. Host raw values and identities never enter the private controller directly.",
            "inputs": {
                "body_identity": BODY_IDENTITY,
                "body_schema_sha256": sha256(args.body_schema),
                "malecns_neurons_sha256": sha256(args.neurons),
            },
            "blocks": [
                {"range": [0, 32], "name": "olfaction"},
                {"range": [32, 40], "name": "gustation"},
                {"range": [40, 46], "name": "airflow"},
                {"range": [46, 62], "name": "audition"},
                {"range": [62, 68], "name": "thorax_velocity"},
                {"range": [68, 69], "name": "irradiance"},
                {"range": [69, 81], "name": "interoception"},
                {"range": [81, 207], "name": "joint_position"},
                {"range": [207, 333], "name": "joint_velocity"},
                {"range": [333, 459], "name": "joint_load"},
                {"range": [459, 495], "name": "foot_contact_and_slip"},
                {"range": [495, 702], "name": "segment_contact"},
                {"range": [702, 708], "name": "haltere_inertial_load"},
                {"range": [708, 712], "name": "mouth_contact"},
                {"range": [712, 804], "name": "actuator_fatigue"},
                {"range": [804, 807], "name": "pump_state"},
            ],
            "channels": body_channels,
        }
        channel_schema_path.write_text(json.dumps(channel_schema, indent=2) + "\n")
        if args.physical_fixture is not None:
            physical = json.loads(args.physical_fixture.read_text())
            cns_sensory_sha = sha256(channel_schema_path)
            cns_actuator_sha = sha256(motor_schema_path)
            if physical.get("body_schema_sha256") != sha256(args.body_schema):
                raise SystemExit("physical fixture and canonical body schema identities differ")
            if physical.get("cns_sensory_schema_sha256") != cns_sensory_sha:
                raise SystemExit("physical fixture and BODY807 schema identities differ")
            if physical.get("cns_actuator_schema_sha256") != cns_actuator_sha:
                raise SystemExit("physical fixture and MOTOR92 schema identities differ")
            if physical.get("sensory_schema_sha256") != cns_sensory_sha:
                raise SystemExit("physical fixture sensory compatibility identity differs")
            if physical.get("actuator_schema_sha256") != cns_actuator_sha:
                raise SystemExit("physical fixture actuator compatibility identity differs")
            cross_schema_identity = {
                "body_schema_sha256": {
                    "sha256": sha256(args.body_schema),
                    "scope": "canonical author-derived morphology, 126-DOF order and physical M90 schema file",
                },
                "morphology_asset_set_sha256": {
                    "sha256": str(physical["morphology_asset_set_sha256"]),
                    "scope": "canonical hash of model asset file paths and content hashes used by the compiled physical fixture",
                },
                "cns_sensory_schema_sha256": {
                    "sha256": cns_sensory_sha,
                    "scope": "canonical BODY807 neural channel names, ranges, units and evidence",
                },
                "physical_sensory_schema_sha256": {
                    "sha256": str(physical["physical_sensory_schema_sha256"]),
                    "scope": "compiled contact schema, compound-eye source, physical eye/olfactory/mouth anchors and engineered optic calibration",
                },
                "cns_actuator_schema_sha256": {
                    "sha256": cns_actuator_sha,
                    "scope": "canonical MOTOR92 semantic order, ranges, activation and MJCF/native boundary",
                },
                "physical_actuator_schema_sha256": {
                    "sha256": str(physical["physical_actuator_schema_sha256"]),
                    "scope": "compiled resident M90 numeric actuator/control map",
                },
                "optic_calibration_schema_sha256": {
                    "sha256": str(physical["optic_calibration"]["schema_sha256"]),
                    "scope": "engineered 1771-ray equidistant projection from author camera field of view",
                },
            }

        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "atlas.schema": np.asarray(ATLAS_SCHEMA),
            "atlas.body_identity": np.asarray(BODY_IDENTITY),
            "atlas.neurons_sha256": np.asarray(sha256(args.neurons)),
            "atlas.body_schema_sha256": np.asarray(sha256(args.body_schema)),
            "atlas.body_channel_schema_sha256": np.asarray(sha256(channel_schema_path)),
            "atlas.motor_rows": global_motor_rows,
            "atlas.motor_ids": neurons["ids"][global_motor_rows],
            "atlas.motor_body_ids": neurons["body_ids"][global_motor_rows],
            "atlas.motor_subclasses": motor_subclasses,
            "atlas.motor_sides": motor_sides,
            "atlas.motor_types": motor_types,
            "atlas.motor_manc_types": motor_manc_types,
            "atlas.motor_mask": motor_mask,
            "atlas.motor_row_supported": np.any(motor_mask != 0, axis=0),
            "atlas.motor_cohort_index": cohort_index,
            "atlas.fine_cohort_names": np.asarray(cohort_names),
            "atlas.fine_cohort_targets": np.asarray(cohort_targets),
            "atlas.fine_cohort_mask": cohort_mask,
            "atlas.actuator_ids": np.asarray([a["id"] for a in actuators]),
            "atlas.actuator_targets": np.asarray([a["target"] for a in actuators]),
            "atlas.actuator_groups": np.asarray([a["control_group"] for a in actuators]),
            "atlas.actuator_kinds": np.asarray([a["kind"] for a in actuators]),
            "atlas.actuator_evidence_grades": np.asarray(motor_grades),
            "atlas.actuator_mapping_bases": np.asarray(motor_bases),
            "atlas.actuator_control_ranges": np.asarray([a["control_range"] for a in actuators], dtype=np.float32),
            "atlas.afferent_rows": global_afferent_rows,
            "atlas.afferent_ids": neurons["ids"][global_afferent_rows],
            "atlas.afferent_body_ids": neurons["body_ids"][global_afferent_rows],
            "atlas.afferent_classes": aff_classes,
            "atlas.afferent_subclasses": aff_subclasses,
            "atlas.afferent_entry_nerves": aff_nerves,
            "atlas.afferent_sides": aff_sides,
            "atlas.afferent_port_names": np.asarray(port_names),
            "atlas.afferent_port_modalities": np.asarray(port_modalities),
            "atlas.afferent_port_evidence_grades": np.asarray(port_grades),
            "atlas.afferent_port_mask": port_mask,
            "atlas.joint_ids": np.asarray([j["id"] for j in joints]),
            "atlas.joint_evidence_grades": np.asarray(joint_grades),
            "atlas.joint_position_afferent_mask": joint_position,
            "atlas.joint_velocity_afferent_mask": joint_velocity,
            "atlas.joint_load_afferent_mask": joint_load,
            "atlas.body_rows": global_afferent_rows,
            "atlas.body_names": np.asarray([channel["name"] for channel in body_channels]),
            "atlas.body_evidence_grades": np.asarray([channel["evidence_grade"] for channel in body_channels]),
            "atlas.body_mask": body_mask,
        }
        np.savez_compressed(output, **arrays)

    unsupported = np.flatnonzero(~np.any(motor_mask != 0, axis=0))
    manifest = {
        "schema": ATLAS_SCHEMA,
        "artifact": output.name,
        "artifact_sha256": sha256(output),
        "inputs": {
            "neurons_npz": {"name": args.neurons.name, "sha256": sha256(args.neurons)},
            "body_schema": {"name": args.body_schema.name, "sha256": sha256(args.body_schema)},
            "body807_channel_schema": {"name": channel_schema_path.name, "sha256": sha256(channel_schema_path)},
            "motor92_channel_schema": {"name": motor_schema_path.name, "sha256": sha256(motor_schema_path)},
        },
        "cross_schema_identity": cross_schema_identity,
        "counts": {
            "motor_rows": 815,
            "motor_rows_with_m92_support": int(np.any(motor_mask != 0, axis=0).sum()),
            "unsupported_motor_rows": int(len(unsupported)),
            "motor_outputs": 92,
            "mjcf_actuators": 90,
            "fine_cohorts": len(cohort_names),
            "fine_cohort_motor_rows": int(cohort_mask.sum()),
            "nonvisual_afferent_rows": int(len(global_afferent_rows)),
            "nonvisual_afferent_rows_with_body807_support": int(np.any(body_mask != 0, axis=1).sum()),
            "unsupported_nonvisual_afferent_rows": int(np.count_nonzero(~np.any(body_mask != 0, axis=1))),
            "afferent_ports": len(port_names),
            "body_joint_dofs": len(joints),
            "body_channels": len(body_channels),
            "body_channels_with_afferent_support": int(np.any(body_mask != 0, axis=0).sum()),
            "unsupported_body_channels": int(np.count_nonzero(~np.any(body_mask != 0, axis=0))),
        },
        "arrays": {
            "atlas.motor_rows": "uint32[815], global canonical MaleCNS row indices",
            "atlas.motor_mask": "float32[92,815], 1 permits a learned weight; 0 structurally forbids it",
            "atlas.motor_cohort_index": f"uint32[815], optional fine-cohort index; {UINT32_SENTINEL} means unresolved",
            "atlas.fine_cohort_mask": "uint8[156,815], diagnostic author target cohorts; not a runtime bottleneck",
            "atlas.afferent_rows": "uint32[11798], nonvisual peripheral and ascending sensory canonical rows",
            "atlas.afferent_port_mask": f"uint8[{len(port_names)},11798], author annotation support for named ports",
            "atlas.joint_position_afferent_mask": "uint8[126,11798], inferred anatomical support; zero means unsupported",
            "atlas.joint_velocity_afferent_mask": "uint8[126,11798], inferred anatomical support; zero means unsupported",
            "atlas.joint_load_afferent_mask": "uint8[126,11798], inferred anatomical support; zero means unsupported",
            "atlas.body_rows": "uint32[11798], identical to afferent_rows and in canonical MaleCNS order",
            "atlas.body_mask": "float32[11798,807], 1 permits physical BODY807 channel injection to an afferent row",
        },
        "array_sha256": {
            name: hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()
            for name in (
                "atlas.body_rows",
                "atlas.body_mask",
                "atlas.motor_rows",
                "atlas.motor_mask",
                "atlas.motor_cohort_index",
                "atlas.fine_cohort_mask",
                "atlas.actuator_ids",
                "atlas.body_names",
            )
        },
        "decoder_contract": {
            "input": "actual post-recurrence MaleCNS motor state, ordered by motor_rows",
            "reference_rate": "separate float32[815] calibration tensor measured from an actual CNS calibration corpus; absent from this anatomical artifact",
            "normalization": "center and reference-scale each motor row before decoding",
            "weights": "learned signed weights multiplied by motor_mask at every forward pass",
            "output": "bounded M92: body M90 in exact actuator order, then pharyngeal pump and salivary drive",
            "ranges": "position rows 0:84 use body-schema bounds and signed centered outputs; adhesion and physiology rows 84:92 are [0,1]",
            "activation_dynamics": "owned by the runtime ABI; this atlas does not fabricate muscle activation time constants",
            "muscle_warning": "MOTOR92 is a servo/physiology target interface, not muscle recruitment; mask signs and servo directions are not anatomical claims",
        },
        "unsupported_motor": [
            {
                "motor_index": int(i),
                "canonical_row": int(global_motor_rows[i]),
                "id": str(np.load(args.neurons, allow_pickle=False)["ids"][global_motor_rows[i]]),
                "subclass": str(motor_subclasses[i]),
                "type": str(np.load(args.neurons, allow_pickle=False)["types"][global_motor_rows[i]]),
                "reason": "rm eye motor with passive eye axes" if motor_subclasses[i] == "rm" else "xm target identity unsupported",
            }
            for i in unsupported
        ],
        "evidence_terms": {
            "measured": "author dataset identifies the peripheral modality/site within MaleCNS",
            "inter_animal_inferred": "author annotation or Chreatures transfer joins different animals, datasets, or body axes",
            "engineered": "simulation control/sensor mechanism with no claimed biological identity",
            "unsupported": "no defensible mapping is available; represented by a zero mask",
        },
    }
    manifest_path = (args.manifest or output.with_suffix(".manifest.json")).resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"artifact": str(output), "manifest": str(manifest_path), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
