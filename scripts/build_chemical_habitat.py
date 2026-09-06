#!/usr/bin/env python3
"""Compile a fresh chemical ecology from supplied founders and physical terrain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(*, recycling=False):
    birth = json.loads((ROOT / "data/biosphere/reef-founders-v1.json").read_text())
    habitat = json.loads((ROOT / "data/habitats/reef-garden.json").read_text())
    birth["format"] = "chreatures-biosphere-birth-v2"
    birth["mobiles"] = []
    birth["material_objects"] = None
    # Fresh founding state. Never applied to existing saved residents.
    for body in habitat["bodies"]:
        soma, gut = len(birth["compartments"]), len(birth["compartments"]) + 1
        birth["compartments"].extend(
            [
                {
                    "enzymes": {"respiration": 0.012},
                    "pools": {
                        "reserve": 1.2,
                        "soft_tissue": 0.08,
                        "tough_tissue": 0.04,
                    },
                    "atp": 0.8,
                    "atp_capacity": 1.0,
                },
                {
                    "enzymes": {
                        "soft_digestion": 0.04,
                        "tough_digestion": 0.008,
                        "detritus_digestion": 0.014,
                    },
                    "pools": {"soft_tissue": 0.018},
                    "atp": 0.08,
                    "atp_capacity": 0.16,
                },
            ]
        )
        birth["mobiles"].append(
            {
                "id": body["id"],
                "body_row": soma,
                "gut_row": gut,
                "gut_capacity": 0.7,
                "reserve_capacity": 1.4,
                "maintenance_rate": 0.0008,
                "activation_rate": 0.006,
                "absorption_rate": 0.035,
                "digestive_atp_rate": 0.008,
                "bite_rate": 0.08,
                "maximum_bite": 0.045,
                "mouth_radius": 0.045,
                "fatigue_rise": 0.045,
                "fatigue_recovery": 0.025,
            }
        )
    # Remove the legacy scalar food stock. Material packets are added separately.
    habitat["entities"] = [e for e in habitat["entities"] if e["id"] != "food-west"]
    habitat["name"] = "chemical-reef"
    chemistry_hash = hashlib.sha256(
        json.dumps(
            birth["chemistry"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    materials = {
        "format": "chreatures-material-objects-v1",
        "chemistry_sha256": chemistry_hash,
        "max_transfer": 1.0,
        "objects": [],
    }
    for index, position in enumerate(
        ([1.4, 1.4, 0.08], [5.4, 6.0, 0.08], [9.8, 1.45, 0.08], [6.4, 4.4, 0.97])
    ):
        name = f"chemical-packet-{index}"
        habitat["materials"][name] = {"rgba": [0.65, 0.2, 0.12, 1.0]}
        habitat["compiler"]["material_order"].append(name)
        habitat["entities"].append(
            {
                "id": name,
                "mobility": "free",
                "material": name,
                "physical_material": "light",
                "position": list(position),
                "shapes": [{"type": "sphere", "size": [0.07]}],
                "components": [],
            }
        )
        row = len(birth["compartments"])
        stock = {"reserve": 0.32, "soft_tissue": 0.08, "detritus": 0.04}
        birth["compartments"].append(
            {"enzymes": {}, "pools": stock, "atp": 0.0, "atp_capacity": 0.0}
        )
        materials["objects"].append(
            {
                "entity": name,
                "row": row,
                "capacities": {"reserve": 0.5, "soft_tissue": 0.1, "detritus": 0.05},
                "content_weights": {
                    "reserve": 1.0,
                    "soft_tissue": 5.0,
                    "detritus": 5.0,
                },
                "remove_when_empty": True,
                "boundaries": [
                    {"minimum_content": 0.55, "scale": 1.0},
                    {"minimum_content": 0.25, "scale": 0.75},
                    {"minimum_content": 0.0, "scale": 0.5},
                ],
                "surface": {
                    "rgb_bias": [0.10, 0.08, 0.03],
                    "rgb_coefficients": {
                        "reserve": [1.4, 0.2, 0.05],
                        "soft_tissue": [0.3, 0.7, 0.2],
                    },
                    "odor_coefficients": {
                        "reserve": [2.0, 0.05, 0.0],
                        "soft_tissue": [0.3, 1.0, 0.0],
                        "detritus": [0.0, 0.3, 2.0],
                    },
                },
            }
        )
    birth["material_objects"] = materials
    if recycling:
        _add_recycling(habitat, birth)
    return copy.deepcopy(habitat), copy.deepcopy(birth)


def configure_visitor_material_supply(habitat, birth, *, slots_per_choice=4):
    """Add finite outside reserves and reusable physical offering slots.

    This mutates one fresh birth specification. Existing worlds and the default
    ``configure()`` result retain their prior chemistry and topology.
    """
    if (
        isinstance(slots_per_choice, bool)
        or not isinstance(slots_per_choice, int)
        or not 1 <= slots_per_choice <= 8
    ):
        raise ValueError("visitor supply needs 1..8 slots per choice")
    materials = birth.get("material_objects")
    if not isinstance(materials, dict):
        raise ValueError("visitor supply requires configured MaterialObjects")
    names = [pool["name"] for pool in birth["chemistry"]["pools"]]
    mass = {
        pool["name"]: sum(pool["composition"])
        for pool in birth["chemistry"]["pools"]
    }
    choices = (
        {
            "id": "reserve-fruit",
            "initial": {"reserve": 2.64},
            "portion": {"reserve": 0.22},
            "shape": {"type": "sphere", "size": [0.075]},
            "rgba": [0.72, 0.18, 0.08, 1.0],
            "surface": {
                "rgb_bias": [0.08, 0.02, 0.01],
                "rgb_coefficients": {"reserve": [2.6, 0.45, 0.1]},
                "odor_coefficients": {"reserve": [2.2, 0.05, 0.0]},
            },
        },
        {
            "id": "soft-fruit",
            "initial": {"reserve": 1.44, "soft_tissue": 0.42},
            "portion": {"reserve": 0.12, "soft_tissue": 0.035},
            "shape": {"type": "ellipsoid", "size": [0.08, 0.06, 0.055]},
            "rgba": [0.48, 0.38, 0.08, 1.0],
            "surface": {
                "rgb_bias": [0.06, 0.05, 0.01],
                "rgb_coefficients": {
                    "reserve": [1.8, 0.3, 0.08],
                    "soft_tissue": [0.4, 2.4, 0.5],
                },
                "odor_coefficients": {
                    "reserve": [1.4, 0.08, 0.0],
                    "soft_tissue": [0.2, 1.8, 0.05],
                },
            },
        },
        {
            "id": "detrital-cake",
            "initial": {"mineral": 0.12, "reserve": 0.72, "detritus": 0.48},
            "portion": {"mineral": 0.01, "reserve": 0.06, "detritus": 0.04},
            "shape": {"type": "box", "size": [0.065, 0.055, 0.045]},
            "rgba": [0.30, 0.20, 0.07, 1.0],
            "surface": {
                "rgb_bias": [0.04, 0.025, 0.01],
                "rgb_coefficients": {
                    "mineral": [0.2, 1.0, 0.25],
                    "reserve": [1.4, 0.25, 0.05],
                    "detritus": [0.3, 0.18, 0.06],
                },
                "odor_coefficients": {
                    "reserve": [0.8, 0.05, 0.0],
                    "detritus": [0.0, 0.35, 2.2],
                },
            },
        },
    )
    supply = {
        "format": "chreatures-visitor-material-supply-v1",
        "chemistry_sha256": materials["chemistry_sha256"],
        "choices": [],
    }
    for choice in choices:
        source_row = len(birth["compartments"])
        birth["compartments"].append(
            {
                "enzymes": {},
                "pools": choice["initial"],
                "atp": 0.0,
                "atp_capacity": 0.0,
            }
        )
        slots = []
        for index in range(slots_per_choice):
            entity_id = f"visitor-{choice['id']}-{index}"
            slots.append(entity_id)
            habitat["materials"][entity_id] = {"rgba": choice["rgba"]}
            habitat["compiler"]["material_order"].append(entity_id)
            row = len(birth["compartments"])
            birth["compartments"].append(
                {"enzymes": {}, "pools": {}, "atp": 0.0, "atp_capacity": 0.0}
            )
            materials["objects"].append(
                {
                    "entity": entity_id,
                    "row": row,
                    "capacities": dict.fromkeys(names, 0.5),
                    "content_weights": mass.copy(),
                    "remove_when_empty": True,
                    "boundaries": [
                        {"minimum_content": 0.12, "scale": 1.0},
                        {"minimum_content": 0.04, "scale": 0.76},
                        {"minimum_content": 0.0, "scale": 0.52},
                    ],
                    "dormant_template": {
                        "id": entity_id,
                        "mobility": "free",
                        "material": entity_id,
                        "physical_material": "light",
                        "position": [0.5, 0.5, 0.1],
                        "shapes": [copy.deepcopy(choice["shape"])],
                        "components": [],
                    },
                    "surface": copy.deepcopy(choice["surface"]),
                }
            )
        supply["choices"].append(
            {
                "id": choice["id"],
                "source_row": source_row,
                "initial_resources": copy.deepcopy(choice["initial"]),
                "portion": copy.deepcopy(choice["portion"]),
                "minimum_fraction": 0.25,
                "slots": slots,
            }
        )
    return supply


def _add_recycling(habitat, birth):
    """Physical shared deposit capacity and chemical root acquisition laws."""
    habitat["name"] = "recycling-reef"
    birth["format"] = "chreatures-biosphere-birth-v3"
    names = [pool["name"] for pool in birth["chemistry"]["pools"]]
    mass = {
        pool["name"]: sum(pool["composition"]) for pool in birth["chemistry"]["pools"]
    }
    slots = []
    for index in range(24):
        name = f"deposit-{index}"
        slots.append(name)
        habitat["materials"][name] = {"rgba": [0.3, 0.25, 0.12, 1.0]}
        habitat["compiler"]["material_order"].append(name)
        row = len(birth["compartments"])
        birth["compartments"].append(
            {"enzymes": {}, "pools": {}, "atp": 0.0, "atp_capacity": 0.0}
        )
        birth["material_objects"]["objects"].append(
            {
                "entity": name,
                "row": row,
                "capacities": dict.fromkeys(names, 0.5),
                "content_weights": mass.copy(),
                "remove_when_empty": True,
                "boundaries": [
                    {"minimum_content": 0.02, "scale": 1.0},
                    {"minimum_content": 0.008, "scale": 0.7},
                    {"minimum_content": 0.0, "scale": 0.45},
                ],
                "dormant_template": {
                    "id": name,
                    "mobility": "free",
                    "material": name,
                    "physical_material": "light",
                    "position": [0.5, 0.5, 0.1],
                    "shapes": [{"type": "sphere", "size": [0.05]}],
                    "components": [],
                },
                "surface": {
                    "rgb_bias": [0.12, 0.08, 0.04],
                    "rgb_coefficients": {
                        "mineral": [0.4, 2.0, 0.4],
                        "reserve": [2.0, 0.4, 0.1],
                    },
                    "odor_coefficients": {
                        "reserve": [2.0, 0.1, 0.0],
                        "detritus": [0.0, 0.3, 2.0],
                    },
                },
            }
        )
    emitter_slots = []
    slot_template = copy.deepcopy(birth["material_objects"]["objects"][-1])
    for index in range(12):
        name = f"exudate-{index}"
        emitter_slots.append(name)
        habitat["materials"][name] = {"rgba": [0.48, 0.36, 0.10, 1.0]}
        habitat["compiler"]["material_order"].append(name)
        row = len(birth["compartments"])
        birth["compartments"].append(
            {"enzymes": {}, "pools": {}, "atp": 0.0, "atp_capacity": 0.0}
        )
        item = copy.deepcopy(slot_template)
        item["entity"] = name
        item["row"] = row
        item["dormant_template"]["id"] = name
        item["dormant_template"]["material"] = name
        birth["material_objects"]["objects"].append(item)
    birth["exchange"] = {
        "format": "chreatures-ecological-exchange-v2",
        "deposit_slots": slots,
        "mobiles": [
            {
                "id": body["id"],
                "interval": 2.0,
                "minimum_mass": 0.0015,
                "maximum_mass": 0.03,
                "offset_radii": [-1.7, 0.0, -0.2],
                "gut_rates": {
                    "mineral": 0.2,
                    "inorganic_carbon": 0.2,
                    "soft_tissue": 0.006,
                    "tough_tissue": 0.006,
                    "detritus": 0.04,
                    "reserve": 0.003,
                },
                "body_rates": {"inorganic_carbon": 0.2},
            }
            for body in habitat["bodies"]
        ],
        "roots": [
            {
                "colony": colony["id"],
                "rates_per_area": {"mineral": 0.8, "inorganic_carbon": 1.2},
                "capacities": {"mineral": 10.0, "inorganic_carbon": 80.0},
            }
            for colony in birth["colonies"]
        ],
        "emitters": [
            {
                "id": f"{colony['id']}-reserve-exudate",
                "donor_row": colony["body_row"],
                "attachment_entity": colony["bindings"]["branch"],
                "local_offset": [0.08, 0.0, 0.08],
                "interval": 2.0,
                "minimum_mass": 0.004,
                "maximum_mass": 0.03,
                "rates": {"reserve": 0.006},
                "reserve_floors": {"reserve": 42.0},
                "deposit_slots": emitter_slots[index * 4 : (index + 1) * 4],
            }
            for index, colony in enumerate(birth["colonies"])
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/chemical-reef-birth"
    )
    parser.add_argument(
        "--recycling",
        action="store_true",
        help="Enable physical egestion and root acquisition",
    )
    parser.add_argument(
        "--visitor-supply",
        action="store_true",
        help="Add finite outside reserves and dormant offering slots",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    habitat, birth = configure(recycling=args.recycling)
    visitor_supply = (
        configure_visitor_material_supply(habitat, birth)
        if args.visitor_supply
        else None
    )
    (args.output / "habitat.json").write_text(json.dumps(habitat, indent=2) + "\n")
    (args.output / "biosphere.json").write_text(json.dumps(birth, indent=2) + "\n")
    if visitor_supply is not None:
        (args.output / "visitor-materials.json").write_text(
            json.dumps(visitor_supply, indent=2) + "\n"
        )
    print(args.output)


if __name__ == "__main__":
    main()
