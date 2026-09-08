"""Author a finite material ecology around an already compiled actual fly scene.

Offline composition only. All recurring chemistry, sensation and body control
are native. This does not load or migrate any historical resident.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AERODYNAMICS_MODEL = ROOT / "native/fly-aerodynamics/assets/aerodynamics-model-v1.json"


def load_wing_aerodynamics() -> dict:
    payload = json.loads(AERODYNAMICS_MODEL.read_text())
    expected = payload.get("aerodynamic_schema_sha256")
    hashed = copy.deepcopy(payload)
    hashed.pop("aerodynamic_schema_sha256", None)
    actual = hashlib.sha256(
        json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError(
            f"wing aerodynamics identity mismatch: expected {expected}, derived {actual}"
        )
    return payload


def assemble(physics: dict, recipe: dict, habitat: dict, habitat_sha256: str) -> dict:
    if habitat.get("format") != "chreatures.fly-habitat-plan.v1":
        raise ValueError("invalid current fly habitat plan")
    if physics.get("habitat_plan_sha256") != habitat_sha256:
        raise ValueError("compiled physics and habitat plan identities differ")
    if habitat.get("parameters", {}).get("residents") != len(physics.get("bodies", [])):
        raise ValueError("habitat plan and compiled resident counts differ")
    fixture = copy.deepcopy(physics)
    fixture["wing_aerodynamics"] = load_wing_aerodynamics()
    for body, resident in zip(fixture["bodies"], fixture["residents"], strict=True):
        segments = {s["semantic_id"]: s["body_id"] for s in resident["segments69"]}
        body["wings"] = [segments["l_wing"], segments["r_wing"]]
        body["wing_centroid_local_mm"] = [[-0.1255656,1.1510138,0.1189536],[-0.1255656,-1.1510138,0.1189536]]
        body["wing_source_gain"] = [1.0,1.0]
    fixture["acoustics"] = dict(sample_hz=1000.0,highpass_hz=20.0,
        lowpass_fraction_of_sample_hz=0.4,spectrum_window_s=0.128,
        dipole_radius_mm=0.55,coupling_scale=0.12,body_band_reference_mm_s=1.0)
    fixture["acoustics_notice"] = "Engineering near-field dipole proxy from actual thorax-relative wing motion; no song oscillator, lift, measured fluid calibration or input anti-alias guarantee. At 1kHz sampling, endogenous bands above400Hz are zero."

    chemistry = copy.deepcopy(recipe["ecology"])
    chemistry["seed"] = habitat["seed"]
    chemistry["coordinate_contract"]["world_min_m"] = [v * 0.001 for v in habitat["bounds_mm"][0]]
    chemistry["coordinate_contract"]["world_max_m"] = [v * 0.001 for v in habitat["bounds_mm"][1]]
    chemistry["regions"] = [
        {key: value for key, value in region.items() if key != "geometry_id"}
        for region in habitat["regions"]
    ]
    chemistry["routes"] = copy.deepcopy(habitat["routes"])
    source_colony = copy.deepcopy(chemistry["organisms"][0])
    source_fly = copy.deepcopy(chemistry["organisms"][1])
    chemistry["organisms"] = []
    for index, body in enumerate(fixture["bodies"]):
        fly = copy.deepcopy(source_fly)
        fly.update(id=body["ecology_id"],physics_binding=body["id"],anchored_region=None,
                   internal_volume_m3=1e-9,capacity=[2.0,2.0,1.0,0.5,1.0,2.0,1.0,0.5],
                   initial=[1.7,1.4,0.6,0.3,0.7,0.05,0.95,0.0],atp=1.7,atp_capacity=2.0)
        genotype = fly["genotype"]
        genotype.update(lineage_id=f"fly-founder-{index}",enzyme_baseline=[0.65,0.2,0.015,0.0,0.025,0.02],
                        enzyme_substrate_response=[0.15,0.12,0.02,0.0,0.01,0.01],enzyme_atp_response=[-0.1,0.08,0,0,0,0],
                        enzyme_total_budget=2.0,membrane_permeability=[0.05,0,0,0,0.8,0.8,0,0.3],
                        development=None,reproduction=None,maintenance_atp_s=0.002)
        chemistry["organisms"].append(fly)

    entities = {entity["id"]: entity for entity in fixture["entities"]}
    bindings = []
    colony_geometries = set()
    for index, site in enumerate(habitat["colony_sites"]):
        entity = entities[site["geometry_id"]]
        colony_geometries.add(site["geometry_id"])
        colony = copy.deepcopy(source_colony)
        colony.update(id=site["id"],physics_binding=site["geometry_id"],anchored_region=site["anchored_region"],
                      initial=site["initial"],atp=site["atp"])
        colony["genotype"]["lineage_id"] = f"habitat-colony-{index}"
        colony["genotype"]["development"].update(
            interval_s=2.0,maximum_structures=24,decay_time_constant_s=40.0,
            branch_angle_rad=0.55+0.15*(index%3),lateral_probability=0.22+0.08*(index%4),
            phototropism=0.8-0.12*(index%3),contact_avoidance=1.2,directional_persistence=0.8)
        colony["genotype"]["reproduction"].update(interval_s=16.0,maximum_descendants=4)
        chemistry["organisms"].append(colony)
        bindings.append(dict(body=entity["body"],geoms=entity["geoms"],
                             store={"kind":"organism","id":site["id"]},exposed=True))

    chemistry["packets"] = []
    for packet_spec in habitat["packets"]:
        entity = entities[packet_spec["geometry_id"]]
        packet = {key: value for key, value in packet_spec.items() if key != "geometry_id"}
        packet["physics_binding"] = packet_spec["geometry_id"]
        chemistry["packets"].append(packet)
        bindings.append(dict(body=entity["body"],geoms=entity["geoms"],
                             store={"kind":"packet","id":packet_spec["id"]},exposed=True))

    for region in habitat["regions"]:
        if region["geometry_id"] in colony_geometries:
            continue
        entity = entities[region["geometry_id"]]
        bindings.append(dict(body=entity["body"],geoms=entity["geoms"],
                             store={"kind":"region","id":region["id"]},exposed=True))

    extent = [habitat["bounds_mm"][1][i] - habitat["bounds_mm"][0][i] for i in range(3)]
    recipe_identity = {"base_recipe": recipe, "habitat_plan_sha256": habitat_sha256}
    fixture.update(ecology=chemistry,material_bindings=bindings,interoception=recipe["fly_interoception"],
                   ecology_capacity=dict(max_colonies=64,max_geoms=2048),
                   illumination=dict(sky_direction_world=[0.0,0.0,1.0],sky_intensity=0.8,
                                     screen_intensity=0.2,photon_energy_per_second=0.3),
                   airflow_mm_s=[0.8,0.2,0.0],volatile_fraction=[0.01,0,0,0,1,1,0,1],
                   ray_distance_mm=max(extent)*1.6,atp_per_model_work=0.001,world_size=extent,
                   physiology_notice="Engineered finite eight-pool physiology with five conserved material axes; model mass/work units are not asserted to be SI mass/joules.",
                   habitat_plan_sha256=habitat_sha256,
                   ecology_recipe_sha256=hashlib.sha256(json.dumps(recipe_identity,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics",type=Path,default=Path(__file__).parent/"fixtures/fly-ecology/physics.json")
    parser.add_argument("--recipe",type=Path,default=ROOT/"native/ecology-core/fixtures/finite-garden-v2.json")
    parser.add_argument("--habitat-plan",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    output = args.output or args.physics.with_name("world.json")
    habitat_bytes = args.habitat_plan.read_bytes()
    habitat_sha256 = hashlib.sha256(habitat_bytes).hexdigest()
    fixture = assemble(
        json.loads(args.physics.read_text()),
        json.loads(args.recipe.read_text()),
        json.loads(habitat_bytes),
        habitat_sha256,
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
    print(json.dumps(dict(path=str(output),residents=len(fixture["bodies"]),regions=len(fixture["ecology"]["regions"]),
                          routes=len(fixture["ecology"]["routes"]),sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == "__main__": main()
