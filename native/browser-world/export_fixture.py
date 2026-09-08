"""Author a finite material ecology around an already compiled actual fly scene.

Offline composition only. All recurring chemistry, sensation and body control
are native. This does not load or migrate any historical resident.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def assemble(physics: dict, recipe: dict, seed: int) -> dict:
    fixture = copy.deepcopy(physics)
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
    chemistry["seed"] = seed
    chemistry["coordinate_contract"]["world_min_m"] = [-0.025, -0.020, -0.002]
    chemistry["coordinate_contract"]["world_max_m"] = [0.025, 0.020, 0.014]
    source_colony = copy.deepcopy(chemistry["organisms"][0])
    source_fly = copy.deepcopy(chemistry["organisms"][1])
    regions = []
    dims = (5, 4, 3)
    spacing = (0.010, 0.010, 0.004)
    volume = math.prod(spacing)
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                variation = 0.5 + 0.5 * math.sin(seed * 0.013 + x * 1.7 + y * 2.3)
                initial = [2.0 if z == 0 else 0.1, 0.1 * variation if z == 0 else 0.0,
                           0.03 if z == 0 else 0.0, 0.1 if z == 0 else 0.0,
                           6.0, 0.3, 0.0, 0.003 * variation]
                regions.append(dict(id=f"region-{x}-{y}-{z}", center_m=[-0.020+x*spacing[0], -0.015+y*spacing[1], z*spacing[2]+0.001],
                                    volume_m3=volume, capacity=[20.0]*8, initial=initial))
    chemistry["regions"] = regions
    routes = []
    def region_id(x, y, z): return f"region-{x}-{y}-{z}"
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                here = (x,y,z)
                for axis in range(3):
                    there = list(here); there[axis] += 1
                    if there[axis] >= dims[axis]: continue
                    routes.append(dict(id=f"route-{x}-{y}-{z}-{axis}",a=region_id(*here),b=region_id(*there),
                                       length_m=spacing[axis],cross_section_m2=volume/spacing[axis],
                                       hydraulic_capacity_m3_s=volume*0.05,base_open_fraction=1.0))
    chemistry["routes"] = routes
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
    entities = {e["id"]: e for e in fixture["entities"]}
    for index, (entity_id, at) in enumerate([("leaf-west",[-0.008,-0.006,0.0003]),("leaf-east",[0.008,0.005,0.0004])]):
        colony = copy.deepcopy(source_colony)
        region = min(regions,key=lambda r:sum((r["center_m"][k]-at[k])**2 for k in range(3)))
        colony.update(id=f"colony-{index}",physics_binding=entity_id,anchored_region=region["id"],
                      initial=[1.5,1.2,0.8,0.6,0.8,0.6,1.6,0.05],atp=0.8)
        colony["genotype"]["lineage_id"] = f"colony-founder-{index}"
        colony["genotype"]["development"].update(
            interval_s=2.0,maximum_structures=24,decay_time_constant_s=40.0,
            branch_angle_rad=0.55+0.25*index,lateral_probability=0.22+0.16*index,
            phototropism=0.8-0.35*index,contact_avoidance=1.2,directional_persistence=0.8)
        colony["genotype"]["reproduction"].update(interval_s=16.0,maximum_descendants=4)
        chemistry["organisms"].append(colony)
    packets = []
    bindings = []
    for entity_id, entity in entities.items():
        if entity_id.startswith("moist-patch") or entity_id.startswith("grain"):
            wet = entity_id.startswith("moist")
            quantity = [1.6,0.8,0.35,0.05,0.02,0,0.05,0.05] if wet else [0.2,0.5,0.3,0.02,0.01,0,0.3,0.02]
            packets.append(dict(id=entity_id,physics_binding=entity_id,volume_m3=1e-11 if wet else 8e-11,
                                capacity=[3.0]*8,initial=quantity))
            bindings.append(dict(body=entity["body"],geoms=entity["geoms"],store={"kind":"packet","id":entity_id},exposed=True))
        elif entity_id in {"leaf-west","leaf-east"}:
            index = 0 if entity_id == "leaf-west" else 1
            bindings.append(dict(body=entity["body"],geoms=entity["geoms"],store={"kind":"organism","id":f"colony-{index}"},exposed=True))
    chemistry["packets"] = packets
    fixture.update(ecology=chemistry,material_bindings=bindings,interoception=recipe["fly_interoception"],
                   illumination=dict(sky_direction_world=[0.0,0.0,1.0],sky_intensity=0.8,
                                     screen_intensity=0.2,photon_energy_per_second=0.3),
                   airflow_mm_s=[0.8,0.2,0.0],volatile_fraction=[0.01,0,0,0,1,1,0,1],
                   ray_distance_mm=120.0,atp_per_model_work=0.001,world_size=[50,40,16],
                   physiology_notice="Engineered finite eight-pool physiology with five conserved material axes; model mass/work units are not asserted to be SI mass/joules.",
                   ecology_recipe_sha256=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics",type=Path,default=Path(__file__).parent/"fixtures/fly-ecology/physics.json")
    parser.add_argument("--recipe",type=Path,default=ROOT/"native/ecology-core/fixtures/finite-garden-v2.json")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--seed",type=int,default=20260908)
    args = parser.parse_args()
    output = args.output or args.physics.with_name("world.json")
    fixture = assemble(json.loads(args.physics.read_text()),json.loads(args.recipe.read_text()),args.seed)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
    print(json.dumps(dict(path=str(output),residents=len(fixture["bodies"]),regions=len(fixture["ecology"]["regions"]),
                          routes=len(fixture["ecology"]["routes"]),sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == "__main__": main()
