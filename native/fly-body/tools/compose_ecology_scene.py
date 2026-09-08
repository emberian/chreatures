#!/usr/bin/env python3
"""Compose a deterministic multi-resident, fly-scale MuJoCo ecology fixture.

This is an offline asset compiler.  It clones the already-exported pinned
NeuroMechFly XML, prefixes every resident-owned MuJoCo object, adds primitive
terrain, and emits numeric compiled-ID maps.  It does not import FlyGym and it
does not supply locomotor commands.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

FIXTURE_ID = "neuromechfly-2.1.0-ca65a510-ypr"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
GEOM_DYNAMICS = {
    "margin": "0.001",
    "solref": "0.0002",
    "solimp": "0.98 0.99 1e-05 0.5 3",
    "friction": "1 1 0.02",
}


def floats(values: list[float] | np.ndarray) -> list[float]:
    return [float(x) for x in values]


def fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{x:.9g}" for x in values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def renamed(value: str, prefix: str) -> str:
    if value == "fly":
        return f"{prefix}/free"
    if value.startswith("fly/"):
        return f"{prefix}/{value[4:]}"
    return value


def clone_named_tree(element: ET.Element, prefix: str) -> ET.Element:
    clone = copy.deepcopy(element)
    for item in clone.iter():
        if "name" in item.attrib:
            item.set("name", renamed(item.get("name", ""), prefix))
        for attr in ("joint", "body", "subtree1", "subtree2", "site"):
            if attr in item.attrib:
                item.set(attr, renamed(item.get(attr, ""), prefix))
    return clone


def add_environment_assets(asset: ET.Element) -> None:
    materials = {
        "ecology/soil": "0.18 0.10 0.045 1",
        "ecology/leaf": "0.18 0.48 0.12 1",
        "ecology/stem": "0.28 0.22 0.08 1",
        "ecology/bark": "0.31 0.16 0.07 1",
        "ecology/moist": "0.07 0.25 0.31 0.72",
        "ecology/grain": "0.72 0.56 0.25 1",
        "ecology/pod": "0.58 0.24 0.16 1",
    }
    for name, rgba in materials.items():
        ET.SubElement(asset, "material", name=name, rgba=rgba)


def collision_geom(parent: ET.Element, **attributes: str) -> ET.Element:
    attributes.update(GEOM_DYNAMICS)
    attributes.update(contype="1", conaffinity="1")
    if parent.tag == "worldbody" and attributes.get("name", "").startswith("ecology/"):
        parent = ET.SubElement(parent, "body", name=f"{attributes['name']}/body")
    return ET.SubElement(parent, "geom", attributes)


def add_environment_from_plan(
    worldbody: ET.Element, plan: dict
) -> list[tuple[str, list[float], list[float]]]:
    dynamic_poses = []
    for spec in plan["geometries"]:
        name = f"ecology/{spec['id']}"
        attributes = {
            "name": name,
            "type": spec["shape"],
            "size": fmt(spec["size_mm"]),
            "material": f"ecology/{spec['material']}",
        }
        if spec["fromto_mm"] is not None:
            attributes["fromto"] = fmt(spec["fromto_mm"])
        if spec["position_mm"] is not None:
            attributes["pos"] = fmt(spec["position_mm"])
        if spec["euler_rad"] is not None:
            attributes["euler"] = fmt(spec["euler_rad"])
        if spec["dynamic"]:
            position = spec["position_mm"]
            quaternion = spec["quaternion_wxyz"]
            body = ET.SubElement(worldbody, "body", name=name, pos=fmt(position), quat=fmt(quaternion))
            ET.SubElement(body, "freejoint", name=f"{name}/free")
            attributes["name"] = f"{name}/geom"
            attributes.pop("pos", None)
            attributes.pop("euler", None)
            attributes["mass"] = f"{spec['mass_model_units']:.9g}"
            collision_geom(body, **attributes)
            dynamic_poses.append((name, position, quaternion))
        else:
            collision_geom(worldbody, **attributes)
    return dynamic_poses


def apply_collisions(
    body: ET.Element,
    prefix: str,
    resident_index: int,
    residents: int,
    contactable_suffixes: set[str],
) -> None:
    own_bit = 1 << (resident_index + 1)
    resident_mask = sum(1 << (i + 1) for i in range(residents))
    affinity = 1 | (resident_mask ^ own_bit)
    for geom in body.iter("geom"):
        suffix = geom.get("name", "").removeprefix(prefix + "/")
        if suffix in contactable_suffixes:
            geom.attrib.update(GEOM_DYNAMICS)
            geom.set("contype", str(own_bit))
            geom.set("conaffinity", str(affinity))
        else:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def add_antenna_contact_proxies(
    body: ET.Element,
    prefix: str,
    resident_index: int,
    residents: int,
    proxies: dict,
) -> None:
    own_bit = 1 << (resident_index + 1)
    resident_mask = sum(1 << (i + 1) for i in range(residents))
    affinity = 1 | (resident_mask ^ own_bit)
    bodies = {item.get("name"): item for item in body.iter("body")}
    contact = proxies["physics_contract"]["contact_parameters"]
    tactile_dynamics = {
        "margin": f"{contact['margin_mm']:.9g}",
        "solref": f"{contact['solref_time_constant_s']:.9g}",
        "solimp": fmt(contact["solimp"]),
        "friction": fmt(contact["friction"]),
    }
    for proxy in proxies["proxies"]:
        segment_name = f"{prefix}/{proxy['segment']}"
        segment = bodies.get(segment_name)
        if segment is None:
            raise RuntimeError(f"antenna proxy segment is absent: {segment_name}")
        attributes = {
            **tactile_dynamics,
            "name": f"{prefix}/tactile/{proxy['id']}",
            "type": proxy["shape"],
            "size": fmt(proxy["size_mm"]),
            "mass": "0",
            "rgba": "0 0 0 0",
            "group": "3",
            "contype": str(own_bit),
            "conaffinity": str(affinity),
        }
        if "position_mm" in proxy:
            attributes["pos"] = fmt(proxy["position_mm"])
        if "fromto_mm" in proxy:
            attributes["fromto"] = fmt(proxy["fromto_mm"])
        ET.SubElement(segment, "geom", attributes)


def compose(
    base_xml: Path,
    output: Path,
    resident_count: int,
    habitat: dict,
    antenna_proxies: dict,
) -> list[str]:
    if not 1 <= resident_count <= 16:
        raise SystemExit("--residents must be between 1 and 16")
    if habitat.get("format") != "chreatures.fly-habitat-plan.v1":
        raise SystemExit("--habitat-plan must use chreatures.fly-habitat-plan.v1")
    if habitat.get("parameters", {}).get("residents") != resident_count:
        raise SystemExit("habitat resident count differs from --residents")
    if antenna_proxies.get("format") != "chreatures-antenna-contact-proxies-v1":
        raise SystemExit("invalid antenna contact proxy artifact")
    expected_proxy_segments = {
        f"{side}_{segment}" for side in "lr" for segment in ("pedicel", "funiculus", "arista")
    }
    if {proxy.get("segment") for proxy in antenna_proxies.get("proxies", [])} != expected_proxy_segments:
        raise SystemExit("antenna contact proxy segment set differs")
    tree = ET.parse(base_xml)
    root = tree.getroot()
    root.set("model", f"chreatures_ecology_{resident_count}x")
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    sensor = root.find("sensor")
    contact = root.find("contact")
    keyframe = root.find("keyframe")
    asset = root.find("asset")
    assert all(x is not None for x in (worldbody, actuator, sensor, contact, keyframe, asset))

    source_body = next(x for x in worldbody if x.tag == "body" and x.get("name") == "fly/c_thorax")
    source_actuators = list(actuator)
    source_sensors = list(sensor)
    source_key = keyframe.find("key")
    assert source_key is not None
    source_qpos = [float(x) for x in source_key.get("qpos", "").split()]
    source_ctrl = [float(x) for x in source_key.get("ctrl", "").split()]
    contactable_suffixes = {
        pair.get("geom2", "").removeprefix("fly/") for pair in contact
    }
    # The base author preset only declares walking contact bodies.  The ecology
    # derivative adds the actual haustellum mesh so the native mouth-tip law can
    # require measured physical contact instead of a proximity-only feeding cue.
    contactable_suffixes.add("c_haustellum")

    worldbody.clear()
    actuator.clear()
    sensor.clear()
    contact.clear()
    keyframe.clear()
    add_environment_assets(asset)

    prefixes = [f"resident{index:02d}" for index in range(resident_count)]
    qpos: list[float] = []
    ctrl: list[float] = []
    spawns = sorted(habitat["spawns"], key=lambda spawn: spawn["resident"])
    if [spawn["resident"] for spawn in spawns] != list(range(resident_count)):
        raise SystemExit("habitat plan has incomplete resident spawn order")
    for index, (prefix, spawn) in enumerate(zip(prefixes, spawns)):
        x, y, z = spawn["position_mm"]
        yaw = spawn["yaw_rad"]
        clone = clone_named_tree(source_body, prefix)
        quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        clone.set("pos", fmt((x, y, z)))
        clone.set("quat", fmt(quat))
        apply_collisions(clone, prefix, index, resident_count, contactable_suffixes)
        add_antenna_contact_proxies(clone, prefix, index, resident_count, antenna_proxies)
        worldbody.append(clone)
        ET.SubElement(worldbody, "site", name=f"{prefix}/spawn", pos=fmt((x, y, 0.0)))
        for source in source_actuators:
            actuator.append(clone_named_tree(source, prefix))
        for source in source_sensors:
            cloned_sensor = clone_named_tree(source, prefix)
            cloned_sensor.set("name", f"{prefix}/{source.get('name')}")
            cloned_sensor.attrib.pop("geom2", None)
            sensor.append(cloned_sensor)
        qpos.extend([x, y, z, *quat, *source_qpos[7:]])
        ctrl.extend(source_ctrl)

    dynamic_poses = add_environment_from_plan(worldbody, habitat)
    for _, position, quaternion in dynamic_poses:
        qpos.extend([*position, *quaternion])
    ET.SubElement(keyframe, "key", name="neutral", qpos=fmt(qpos), ctrl=fmt(ctrl))

    resident_mask = sum(1 << (i + 1) for i in range(resident_count))
    for geom in worldbody.iter("geom"):
        if geom.get("name", "").startswith("ecology/"):
            geom.set("contype", "1")
            geom.set("conaffinity", str(1 | resident_mask))

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode", xml_declaration=False)
    return prefixes


def numeric_id(model: mj.MjModel, object_type: mj.mjtObj, name: str) -> int:
    result = int(mj.mj_name2id(model, object_type, name))
    if result < 0:
        raise RuntimeError(f"compiled {object_type.name} missing: {name}")
    return result


def relative_camera_pose(data: mj.MjData, body_id: int, camera_id: int) -> tuple[list[float], list[list[float]]]:
    body_rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    camera_rotation = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
    position = body_rotation.T @ (np.asarray(data.cam_xpos[camera_id]) - np.asarray(data.xpos[body_id]))
    rotation = body_rotation.T @ camera_rotation
    return floats(position), [[float(x) for x in row] for row in rotation]


def nearest_segment(model: mj.MjModel, body_id: int, segments: dict[int, str]) -> str | None:
    while body_id > 0:
        if body_id in segments:
            return segments[body_id]
        body_id = int(model.body_parentid[body_id])
    return None


def engineered_retina(optic_atlas: Path) -> dict:
    atlas = np.load(optic_atlas)
    sites = np.asarray(atlas["site_side_hex"], dtype=np.int16)
    if sites.shape != (1771, 3):
        raise RuntimeError(f"optic atlas site_side_hex must be [1771,3], got {sites.shape}")
    supported = np.zeros(1771, dtype=bool)
    supported[np.asarray(atlas["photoreceptor_site_indices"], dtype=np.int64)] = True
    aspect = 450.0 / 512.0
    half_fov = math.radians(78.5)
    directions = np.empty((1771, 3), dtype=np.float64)
    calibration = []
    for side in (1, 2):
        mask = sites[:, 0] == side
        q = sites[mask, 1].astype(np.float64)
        r = sites[mask, 2].astype(np.float64)
        cq = 0.5 * (q.min() + q.max())
        cr = 0.5 * (r.min() + r.max())
        x = q + 0.5 * r
        y = math.sqrt(3.0) * 0.5 * r
        center_x = cq + 0.5 * cr
        center_y = math.sqrt(3.0) * 0.5 * cr
        sx = float(np.max(np.abs(x - center_x)))
        sy = float(np.max(np.abs(y - center_y)))
        u = (x - center_x) / sx * aspect
        v = -(y - center_y) / sy
        rho = np.hypot(u, v)
        theta = rho * half_fov
        nonzero = rho > 0
        local = np.zeros((len(q), 3), dtype=np.float64)
        local[:, 2] = -np.cos(theta)
        local[nonzero, 0] = np.sin(theta[nonzero]) * u[nonzero] / rho[nonzero]
        local[nonzero, 1] = np.sin(theta[nonzero]) * v[nonzero] / rho[nonzero]
        directions[mask] = local
        calibration.append(
            [cq, cr, aspect / sx, 0.5 * aspect / sx, 0.0, -math.sqrt(3.0) / (2.0 * sy), half_fov, 1.0]
        )
    payload = {
        "version": "engineered-equidistant-author-fov-v1",
        "evidence_grade": "ENGINEERED_EQUIDISTANT_FROM_AUTHOR_FOV",
        "author_camera_vertical_fov_deg": 157.0,
        "author_camera_pixel_aspect_width_over_height": aspect,
        "camera_frame": "+X image-right, +Y image-up, -Z forward",
        "equation": "hex x=q+.5r,y=sqrt(3)/2*r; normalized u,v; rho=hypot(u,v); theta=rho*78.5deg; d=[sin(theta)u/rho,sin(theta)v/rho,-cos(theta)]",
        "calibration_rows2x8": calibration,
        "directions_camera_local": directions.tolist(),
    }
    payload["schema_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {
        "atlas_sha256": sha256(optic_atlas),
        "anatomical_sites": sites.tolist(),
        "supported_sites": supported.tolist(),
        "retinal_directions": directions.tolist(),
        "calibration": payload,
    }


def compiled_manifest(
    xml_path: Path,
    schema: dict,
    prefixes: list[str],
    steps: int,
    optic_atlas: Path,
    habitat: dict,
    habitat_path: Path,
    antenna_proxies: dict,
    antenna_proxy_path: Path,
) -> dict:
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    key_id = numeric_id(model, mj.mjtObj.mjOBJ_KEY, "neutral")
    mj.mj_resetDataKeyframe(model, data, key_id)
    mj.mj_forward(model, data)
    neutral_qpos = np.array(data.qpos, copy=True)
    neutral_ctrl = np.array(data.ctrl, copy=True)
    for _ in range(steps):
        mj.mj_step(model, data)
    if not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.sensordata)):
        raise RuntimeError("non-finite state during ecology startup")

    residents = []
    all_segment_ids: dict[int, str] = {}
    for prefix in prefixes:
        segments = []
        for segment in schema["body_segments"]:
            name = f"{prefix}/{segment['id']}"
            body_id = numeric_id(model, mj.mjtObj.mjOBJ_BODY, name)
            segments.append({"semantic_id": segment["id"], "name": name, "body_id": body_id})
            all_segment_ids[body_id] = f"{prefix}/{segment['id']}"
        joints = []
        for joint in schema["joint_dofs"]:
            name = f"{prefix}/{joint['id']}"
            joint_id = numeric_id(model, mj.mjtObj.mjOBJ_JOINT, name)
            joints.append(
                {
                    "semantic_id": joint["id"],
                    "name": name,
                    "joint_id": joint_id,
                    "qpos_address": int(model.jnt_qposadr[joint_id]),
                    "dof_address": int(model.jnt_dofadr[joint_id]),
                    "axis_semantic": joint["axis_semantic"],
                    "actuated": joint["actuated"],
                }
            )
        actuators = []
        for actuator_spec in schema["actuators"]:
            name = f"{prefix}/{actuator_spec['id']}"
            actuator_id = numeric_id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            lo, hi = (float(x) for x in model.actuator_ctrlrange[actuator_id])
            neutral = float(neutral_ctrl[actuator_id])
            normalized = (
                {
                    "input_range": [-1.0, 1.0],
                    "formula": "neutral + u*(neutral-low) if u<0 else neutral + u*(high-neutral)",
                    "negative_scale": neutral - lo,
                    "positive_scale": hi - neutral,
                }
                if actuator_spec["kind"] == "position_target"
                else {"input_range": [0.0, 1.0], "formula": "identity"}
            )
            actuators.append(
                {
                    "semantic_id": actuator_spec["id"],
                    "name": name,
                    "actuator_id": actuator_id,
                    "kind": actuator_spec["kind"],
                    "control_group": actuator_spec["control_group"],
                    "control_range": [lo, hi],
                    "neutral": neutral,
                    "normalized": normalized,
                }
            )
        sensors = []
        for source in schema["contact_sensors"]:
            name = f"{prefix}/{source['id']}"
            sensor_id = numeric_id(model, mj.mjtObj.mjOBJ_SENSOR, name)
            sensors.append(
                {
                    "semantic_id": source["id"],
                    "sensor_id": sensor_id,
                    "data_address": int(model.sensor_adr[sensor_id]),
                    "dimension": int(model.sensor_dim[sensor_id]),
                    "channels": schema["interfaces"]["cns_eligible_body_measurements"]["ground_contact_raw"]["channels"],
                }
            )
        eyes = []
        for side in ("l", "r"):
            body_id = numeric_id(model, mj.mjtObj.mjOBJ_BODY, f"{prefix}/{side}_eye")
            camera_id = numeric_id(model, mj.mjtObj.mjOBJ_CAMERA, f"{prefix}/{side}_eye_cam_camera")
            position, rotation = relative_camera_pose(data, body_id, camera_id)
            eyes.append(
                {
                    "side": "left" if side == "l" else "right",
                    "body_id": body_id,
                    "camera_id": camera_id,
                    "position_local_mm": position,
                    "rotation_local_3x3": rotation,
                    "author_ommatidia": 721,
                }
            )
        def body_ref(segment: str) -> int:
            return numeric_id(model, mj.mjtObj.mjOBJ_BODY, f"{prefix}/{segment}")

        olfactory = [
            {"id": "left_funiculus", "body_id": body_ref("l_funiculus"), "position_local_mm": [0.0084393, -0.0128549, -0.102897]},
            {"id": "right_funiculus", "body_id": body_ref("r_funiculus"), "position_local_mm": [0.0084393, 0.0128549, -0.102897]},
            {"id": "left_arista", "body_id": body_ref("l_arista"), "position_local_mm": [0.0136261, 0.117421, 0.0133472]},
            {"id": "right_arista", "body_id": body_ref("r_arista"), "position_local_mm": [0.0136261, -0.117421, 0.0133472]},
        ]
        root_id = body_ref("c_thorax")
        free_joint_id = numeric_id(model, mj.mjtObj.mjOBJ_JOINT, f"{prefix}/free")
        residents.append(
            {
                "id": prefix,
                "root_body_id": root_id,
                "head_body_id": body_ref("c_head"),
                "free_joint_id": free_joint_id,
                "free_qpos_address": int(model.jnt_qposadr[free_joint_id]),
                "segments69": segments,
                "joint_dofs126": joints,
                "actuators90": actuators,
                "neutral_servo84": [x["neutral"] for x in actuators[:84]],
                "control_ranges84": [x["control_range"] for x in actuators[:84]],
                "eye_anchors2": eyes,
                "olfactory_anchors4": olfactory,
                "mouth": {
                    "body_id": body_ref("c_haustellum"),
                    "position_tip_local_mm": [0.30, 0.0, -0.12],
                    "contact_radius_mm": 0.06,
                    "evidence_grade": "engineered_from_mesh_extent",
                },
                "feet6": [
                    {
                        "leg": leg,
                        "body_ids": [body_ref(f"{leg}_tarsus{index}") for index in range(1, 6)],
                    }
                    for leg in LEGS
                ],
                "halteres2": [body_ref("l_haltere"), body_ref("r_haltere")],
                "contact_sensors6x16": sensors,
                "neutral_joint_qpos126": [float(neutral_qpos[x["qpos_address"]]) for x in joints],
            }
        )

    geoms = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        geoms.append(
            {
                "geom_id": geom_id,
                "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id),
                "body_id": body_id,
                "resident_segment": nearest_segment(model, body_id, all_segment_ids),
                "geom_type": int(model.geom_type[geom_id]),
                "size": floats(model.geom_size[geom_id]),
                "mesh_id": int(model.geom_dataid[geom_id]) if model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_MESH else -1,
            }
        )
    meshes = []
    for mesh_id in range(model.nmesh):
        meshes.append(
            {
                "mesh_id": mesh_id,
                "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH, mesh_id),
                "vertex_address": int(model.mesh_vertadr[mesh_id]),
                "vertex_count": int(model.mesh_vertnum[mesh_id]),
                "face_address": int(model.mesh_faceadr[mesh_id]),
                "face_count": int(model.mesh_facenum[mesh_id]),
                "scale": floats(model.mesh_scale[mesh_id]),
                "position": floats(model.mesh_pos[mesh_id]),
                "quaternion_wxyz": floats(model.mesh_quat[mesh_id]),
            }
        )
    sites = []
    for site_id in range(model.nsite):
        sites.append(
            {
                "site_id": site_id,
                "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, site_id),
                "body_id": int(model.site_bodyid[site_id]),
                "position_local_mm": floats(model.site_pos[site_id]),
                "quaternion_local_wxyz": floats(model.site_quat[site_id]),
                "size_mm": floats(model.site_size[site_id]),
            }
        )

    compiled_tactile_proxies = []
    resident_mask = sum(1 << (i + 1) for i in range(len(prefixes)))
    for resident_index, prefix in enumerate(prefixes):
        own_bit = 1 << (resident_index + 1)
        expected_affinity = 1 | (resident_mask ^ own_bit)
        for proxy in antenna_proxies["proxies"]:
            geom_name = f"{prefix}/tactile/{proxy['id']}"
            geom_id = numeric_id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
            body_id = numeric_id(model, mj.mjtObj.mjOBJ_BODY, f"{prefix}/{proxy['segment']}")
            if int(model.geom_bodyid[geom_id]) != body_id:
                raise RuntimeError(f"antenna proxy body differs: {geom_name}")
            if int(model.geom_contype[geom_id]) != own_bit or int(model.geom_conaffinity[geom_id]) != expected_affinity:
                raise RuntimeError(f"antenna proxy collision mask differs: {geom_name}")
            compiled_tactile_proxies.append(
                {
                    "resident": prefix,
                    "proxy_id": proxy["id"],
                    "segment": proxy["segment"],
                    "body_id": body_id,
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "shape": proxy["shape"],
                    "evidence_grade": proxy["evidence_grade"],
                    "contype": int(model.geom_contype[geom_id]),
                    "conaffinity": int(model.geom_conaffinity[geom_id]),
                    "margin_mm": float(model.geom_margin[geom_id]),
                    "solref": floats(model.geom_solref[geom_id]),
                    "solimp": floats(model.geom_solimp[geom_id]),
                    "friction": floats(model.geom_friction[geom_id]),
                }
            )

    morphology_files = {k: v for k, v in schema["files"].items() if k.startswith("model/")}
    morphology_hash = hashlib.sha256(json.dumps(morphology_files, sort_keys=True).encode()).hexdigest()
    retina = engineered_retina(optic_atlas)
    sensory_payload = {
        "contact_channels": schema["interfaces"]["cns_eligible_body_measurements"]["ground_contact_raw"],
        "compound_eye_sha256": schema["files"]["author-source/compound_eye.npz"],
        "eye_anchors": [resident["eye_anchors2"] for resident in residents],
        "olfactory_anchors": [resident["olfactory_anchors4"] for resident in residents],
        "mouth": [resident["mouth"] for resident in residents],
        "antenna_contact_proxy_artifact_sha256": sha256(antenna_proxy_path),
        "antenna_contact_proxies": compiled_tactile_proxies,
        "retina_calibration": retina["calibration"],
    }
    compact_bodies = []
    for resident in residents:
        geom_ids_by_body: dict[int, list[int]] = {}
        for geom in geoms:
            geom_ids_by_body.setdefault(geom["body_id"], []).append(geom["geom_id"])
        segment_ids = [segment["body_id"] for segment in resident["segments69"]]
        qpos_addresses = [joint["qpos_address"] for joint in resident["joint_dofs126"]]
        free_qpos_address = resident["free_qpos_address"]
        compact_bodies.append(
            {
                "id": resident["id"],
                "root": resident["root_body_id"],
                "head": resident["head_body_id"],
                "qpos": qpos_addresses,
                "dofs": [joint["dof_address"] for joint in resident["joint_dofs126"]],
                "segments": segment_ids,
                "segment_geom_ids": [geom_ids_by_body.get(body_id, []) for body_id in segment_ids],
                "actuators": [actuator["actuator_id"] for actuator in resident["actuators90"]],
                "neutral": resident["neutral_servo84"],
                "control_ranges": resident["control_ranges84"],
                "normalized_actuators": [actuator["normalized"] for actuator in resident["actuators90"]],
                "eyes": [
                    {
                        "body": eye["body_id"],
                        "position": eye["position_local_mm"],
                        "rotation": sum(eye["rotation_local_3x3"], []),
                    }
                    for eye in resident["eye_anchors2"]
                ],
                "olfactory_sites": [
                    {"body": site["body_id"], "position": site["position_local_mm"]}
                    for site in resident["olfactory_anchors4"]
                ],
                "mouth": {
                    "body": resident["mouth"]["body_id"],
                    "position": resident["mouth"]["position_tip_local_mm"],
                    "contact_radius_mm": resident["mouth"]["contact_radius_mm"],
                },
                "mouth_geom_ids": geom_ids_by_body.get(resident["mouth"]["body_id"], []),
                "feet": [foot["body_ids"] for foot in resident["feet6"]],
                "halteres": resident["halteres2"],
                "ecology_id": resident["id"],
                "neutral_qpos": floats(neutral_qpos[free_qpos_address:free_qpos_address + 7])
                + [float(neutral_qpos[address]) for address in qpos_addresses],
            }
        )
    runtime_geoms = [
        {"id": geom["geom_id"], "body": geom["body_id"], "size": geom["size"]}
        for geom in geoms
    ]
    mesh_assets = [
        {"path": Path(path).name, "sha256": digest}
        for path, digest in sorted(schema["files"].items())
        if path.startswith("model/") and path.endswith(".stl")
    ]
    environment_geoms = [geom for geom in geoms if geom["name"].startswith("ecology/")]
    planned_dynamic = {
        item["id"]: bool(item["dynamic"])
        for item in habitat["geometries"]
    }
    entities = []
    for geom in environment_geoms:
        entity_id = geom["name"].removeprefix("ecology/").removesuffix("/geom")
        dynamic = planned_dynamic[entity_id]
        entity = {
            "id": entity_id,
            "body": geom["body_id"],
            "geoms": [geom["geom_id"]],
            "free": dynamic,
        }
        if dynamic:
            joint_id = numeric_id(model, mj.mjtObj.mjOBJ_JOINT, f"ecology/{entity_id}/free")
            entity["free_qpos_address"] = int(model.jnt_qposadr[joint_id])
            entity["free_dof_address"] = int(model.jnt_dofadr[joint_id])
        entities.append(entity)
    actuator_payload = [resident["actuators90"] for resident in residents]
    actuator_hash = hashlib.sha256(json.dumps(actuator_payload, sort_keys=True).encode()).hexdigest()
    scene_hash = sha256(xml_path)
    planned_names = {
        f"ecology/{item['id']}/geom" if item["dynamic"] else f"ecology/{item['id']}"
        for item in habitat["geometries"]
    }
    compiled_names = {geom["name"] for geom in environment_geoms}
    if planned_names != compiled_names:
        raise RuntimeError(
            f"habitat geometry compilation differs: missing={sorted(planned_names-compiled_names)}, "
            f"extra={sorted(compiled_names-planned_names)}"
        )
    screen_geom = numeric_id(model, mj.mjtObj.mjOBJ_GEOM, "ecology/visual-screen")
    return {
        "format": "chreatures-fly-ecology-compiled-v1",
        "engine": "mujoco-3.12.0-neuromechfly-cns-v4",
        "source_mjcf_sha256": scene_hash,
        "atlas_sha256": retina["atlas_sha256"],
        "schema": "chreatures.fly-ecology-compiled.v1",
        "fixture_id": FIXTURE_ID,
        "scene_xml": xml_path.name,
        "scene_xml_sha256": scene_hash,
        "source_revision": schema["source"]["revision"],
        "antenna_contact_proxies": {
            "format": antenna_proxies["format"],
            "artifact": antenna_proxy_path.name,
            "artifact_sha256": sha256(antenna_proxy_path),
            "source": antenna_proxies["source"],
            "fit": antenna_proxies["fit"],
            "physics_contract": antenna_proxies["physics_contract"],
            "compiled": compiled_tactile_proxies,
        },
        "habitat_plan_sha256": sha256(habitat_path),
        "habitat": {
            "format": habitat["format"],
            "generator": habitat["generator"],
            "seed": habitat["seed"],
            "bounds_mm": habitat["bounds_mm"],
            "parameters": habitat["parameters"],
            "validation": habitat["validation"],
            "information_boundary": habitat["information_boundary"],
        },
        "source_body_mjcf_sha256": schema["files"]["model/model.xml"],
        "body_schema_sha256": "d8c3ff3d22b7f68ec8fb752ba210689ce6531820df57edc96c3bd305766a2d5a",
        "morphology_sha256": morphology_hash,
        "morphology_asset_set_sha256": morphology_hash,
        "physical_sensory_schema_sha256": hashlib.sha256(json.dumps(sensory_payload, sort_keys=True).encode()).hexdigest(),
        "sensory_schema_sha256": "97b48925c5580c883e1e06bac2d14ad458d84c8dec8ed8b25b143975dc0b1786",
        "cns_sensory_schema_sha256": "97b48925c5580c883e1e06bac2d14ad458d84c8dec8ed8b25b143975dc0b1786",
        "physical_actuator_schema_sha256": actuator_hash,
        "actuator_schema_sha256": "00a4a98097f90f6a511e9ee72741e2ee403a8017cfd9d0c0b8c474e195940928",
        "cns_actuator_schema_sha256": "00a4a98097f90f6a511e9ee72741e2ee403a8017cfd9d0c0b8c474e195940928",
        "anatomical_sites": retina["anatomical_sites"],
        "supported_sites": retina["supported_sites"],
        "retinal_directions": retina["retinal_directions"],
        "optic_calibration": retina["calibration"],
        "physics_dt": float(model.opt.timestep),
        "control_dt": 0.01,
        "units": {
            "length": schema["fixture"]["length_unit"],
            "time": schema["fixture"]["time_unit"],
            "angle": schema["fixture"]["angle_unit"],
            "mass": schema["fixture"]["mass_unit_status"],
        },
        "timing": {"physics_dt_s": float(model.opt.timestep), "control_dt_s": 0.01, "substeps_per_control": 100},
        "compiled_counts": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "nbody": int(model.nbody),
            "njnt": int(model.njnt), "ngeom": int(model.ngeom), "nmesh": int(model.nmesh),
            "nsite": int(model.nsite), "nsensor": int(model.nsensor), "nsensordata": int(model.nsensordata),
        },
        "neutral_keyframe": {"key_id": key_id, "name": "neutral"},
        "bodies": compact_bodies,
        "entities": entities,
        "geoms": runtime_geoms,
        "meshes": meshes,
        "mesh_assets": mesh_assets,
        "screen_geom": screen_geom,
        "residents": residents,
        "geom_map": geoms,
        "raw_mujoco": {"meshes": meshes, "sites": sites},
        "motor_contract": {
            "mjcf_motor90": {"range": [0, 90], "servos84": [0, 84], "adhesion6": [84, 90]},
            "native_motor92": {
                "range": [0, 92], "pharyngeal_pump": 90, "salivary_drive": 91,
                "note": "indices 90 and 91 are native fluid/metabolism actions, not MJCF joints or actuators",
            },
        },
        "information_boundary": {
            "resident_body_measurements": "must enter policy and memory through the CNS",
            "body_and_world_kinematics": "observer-only labels, rewards, diagnostics and rendering",
        },
        "validation": {"startup_steps": steps, "time_s": float(data.time), "finite": True},
        "known_limit": "Author compound-eye reduction has 721 ommatidia per eye (1442 total). The 1771 neural-port rays retain atlas site order but use the separately labeled engineered equidistant projection from the author camera FOV.",
    }


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    default_asset = here / "assets" / FIXTURE_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-xml", type=Path, default=default_asset / "model" / "model.xml")
    parser.add_argument("--base-schema", type=Path, default=default_asset / "schema.json")
    parser.add_argument("--optic-atlas", type=Path, default=here.parents[1] / "data" / "ports" / "optic-anatomy-audit-v1.npz")
    parser.add_argument(
        "--antenna-contact-proxies",
        type=Path,
        default=here / "assets" / "antenna-contact-proxies-v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--residents", type=int, default=2)
    parser.add_argument("--habitat-plan", type=Path, required=True)
    parser.add_argument("--startup-steps", type=int, default=10)
    args = parser.parse_args()
    base_xml = args.base_xml.resolve()
    output = args.output.resolve()
    habitat_path = args.habitat_plan.resolve()
    habitat = json.loads(habitat_path.read_text())
    antenna_proxy_path = args.antenna_contact_proxies.resolve()
    antenna_proxies = json.loads(antenna_proxy_path.read_text())
    if output.parent != base_xml.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
        for mesh in base_xml.parent.glob("*.stl"):
            destination = output.parent / mesh.name
            if not destination.exists():
                try:
                    destination.hardlink_to(mesh)
                except OSError:
                    shutil.copy2(mesh, destination)
        license_path = args.base_schema.parent / "author-source" / "FlyGym-LICENSE"
        license_destination = output.parent / license_path.name
        if not license_destination.exists():
            try:
                license_destination.hardlink_to(license_path)
            except OSError:
                shutil.copy2(license_path, license_destination)
        proxy_destination = output.parent / antenna_proxy_path.name
        if proxy_destination.resolve() != antenna_proxy_path:
            if proxy_destination.exists() and sha256(proxy_destination) != sha256(antenna_proxy_path):
                proxy_destination.unlink()
            if not proxy_destination.exists():
                try:
                    proxy_destination.hardlink_to(antenna_proxy_path)
                except OSError:
                    shutil.copy2(antenna_proxy_path, proxy_destination)
    prefixes = compose(base_xml, output, args.residents, habitat, antenna_proxies)
    schema = json.loads(args.base_schema.read_text())
    manifest = compiled_manifest(
        output,
        schema,
        prefixes,
        args.startup_steps,
        args.optic_atlas.resolve(),
        habitat,
        habitat_path,
        antenna_proxies,
        antenna_proxy_path,
    )
    manifest_path = args.manifest.resolve() if args.manifest else output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"xml": str(output), "manifest": str(manifest_path), **manifest["compiled_counts"], "finite": True}, indent=2))


if __name__ == "__main__":
    main()
