"""Lightweight articulated MuJoCo embodiment for the 3-D habitat.

The controller in this module is body mechanics: it converts continuous
``forward``/``turn`` commands into a tripod stance reflex and bounded hinge
torques.  It contains no target, resource, reward, or position policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .physics import PhysicsBody, PhysicsWorld, _canonical, _hex


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BODY_SPEC = ROOT / "data/bodies/hexapod.json"


class ArticulatedWorld(PhysicsWorld):
    """A :class:`PhysicsWorld` whose residents have twelve driven hinges.

    Six hip hinges sweep the feet fore/aft and six knee hinges lift them for
    recovery.  Locomotion comes from the resulting tarsus/terrain contacts;
    this class never applies translational traction to the trunk.
    """

    def __init__(
        self,
        seed: int = 7,
        spec: dict[str, Any] | str | Path | None = None,
        body_spec: dict[str, Any] | str | Path | None = None,
    ):
        habitat = PhysicsWorld._load_spec(spec)
        if body_spec is None:
            body_spec = habitat.get("articulated_body_spec", DEFAULT_BODY_SPEC)
        self.articulation_spec = (
            copy.deepcopy(body_spec) if isinstance(body_spec, dict)
            else json.loads(Path(body_spec).read_text())
        )
        self._validate_articulation_spec(self.articulation_spec)
        # Embed the exact mechanism definition in ordinary world snapshots so
        # inherited restore can reject drift or reconstruct a custom body.
        habitat["articulated_body_spec"] = copy.deepcopy(self.articulation_spec)
        super().__init__(seed=seed, spec=habitat)

    @staticmethod
    def _validate_articulation_spec(spec: dict[str, Any]) -> None:
        if not isinstance(spec, dict) or spec.get("version") != 1:
            raise ValueError("unsupported articulated body specification")
        if spec.get("units") != "meters":
            raise ValueError("articulated body specification must use meters")
        required = {"trunk", "legs", "antennae", "controller"}
        if not required <= set(spec):
            raise ValueError("articulated body specification is incomplete")
        layout = spec["legs"].get("layout")
        if not isinstance(layout, list) or len(layout) != 6:
            raise ValueError("articulated body requires six legs")
        names = {leg.get("name") for leg in layout if isinstance(leg, dict)}
        if names != {"lf", "lm", "lh", "rf", "rm", "rh"}:
            raise ValueError("articulated leg names must be lf/lm/lh/rf/rm/rh")
        for leg in layout:
            if leg.get("side") not in (-1, 1):
                raise ValueError("leg side must be -1 or 1")
            position = leg.get("hip_position")
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError("each leg requires a three-dimensional hip position")
            phase = leg.get("phase")
            if not isinstance(phase, (int, float)) or not 0.0 <= float(phase) < 1.0:
                raise ValueError("leg phase must be in [0, 1)")

    def build_body(self, config: dict[str, Any]) -> str:
        body_id = config["id"]
        material = f"mat:{config['material']}"
        heading = float(config.get("heading", 0.0))
        quat = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
        trunk = self.articulation_spec["trunk"]
        legs = self.articulation_spec["legs"]
        antennae = self.articulation_spec["antennae"]

        pieces = [
            f'<body {self._attrs({"name": f"resident:{body_id}", "pos": config["position"], "quat": quat})}>',
            f'<freejoint name="resident:{body_id}:free"/>',
        ]
        for part, position in (("thorax", [0.0, 0.0, 0.0]),
                               ("head", trunk["head_position"]),
                               ("abdomen", trunk["abdomen_position"])):
            attrs = {
                "name": f"resident:{body_id}:geom:{part}", "type": "ellipsoid",
                "size": trunk[f"{part}_size"], "pos": position,
                "material": material, "density": trunk["density"],
                "friction": trunk["friction"], "condim": 4,
            }
            pieces.append(f"<geom {self._attrs(attrs)}/>")

        base = antennae["base_position"]
        for index, tip in enumerate(antennae["tip_positions"]):
            side = "left" if index == 0 else "right"
            geom = {
                "name": f"resident:{body_id}:geom:antenna:{side}", "type": "capsule",
                "size": [antennae["radius"]], "fromto": [*base, *tip],
                "material": material, "density": 45.0, "contype": 0, "conaffinity": 0,
            }
            site = {
                "name": f"resident:{body_id}:site:antenna:{side}", "type": "sphere",
                "size": [antennae["radius"] * 1.6], "pos": tip, "rgba": [1.0, 1.0, 1.0, 0.0],
            }
            pieces.extend((f"<geom {self._attrs(geom)}/>", f"<site {self._attrs(site)}/>"))

        upper_end_z = -float(legs["upper_drop"])
        lower_end_z = -float(legs["lower_drop"])
        for leg in legs["layout"]:
            name, side = leg["name"], int(leg["side"])
            upper_end = [0.0, side * float(legs["upper_lateral"]), upper_end_z]
            lower_end = [0.0, side * float(legs["lower_lateral"]), lower_end_z]
            pieces.append(f'<body {self._attrs({"name": f"resident:{body_id}:link:{name}:upper", "pos": leg["hip_position"]})}>')
            pieces.append(f'<joint {self._attrs({"name": f"resident:{body_id}:joint:{name}:hip", "type": "hinge", "axis": [0, 0, 1], "range": legs["hip_range_degrees"], "limited": "true", "damping": legs["hip_damping"], "armature": legs["armature"]})}/>')
            pieces.append(f'<geom {self._attrs({"name": f"resident:{body_id}:geom:{name}:upper", "type": "capsule", "size": [legs["upper_radius"]], "fromto": [0, 0, 0, *upper_end], "material": material, "density": legs["density"], "friction": trunk["friction"], "condim": 4})}/>')
            pieces.append(f'<body {self._attrs({"name": f"resident:{body_id}:link:{name}:lower", "pos": upper_end})}>')
            pieces.append(f'<joint {self._attrs({"name": f"resident:{body_id}:joint:{name}:knee", "type": "hinge", "axis": [1, 0, 0], "range": legs["knee_range_degrees"], "limited": "true", "damping": legs["knee_damping"], "armature": legs["armature"]})}/>')
            pieces.append(f'<geom {self._attrs({"name": f"resident:{body_id}:geom:{name}:lower", "type": "capsule", "size": [legs["lower_radius"]], "fromto": [0, 0, 0, *lower_end], "material": material, "density": legs["density"], "friction": trunk["friction"], "condim": 4})}/>')
            pieces.append(f'<geom {self._attrs({"name": f"resident:{body_id}:geom:{name}:tarsus", "type": "sphere", "size": [legs["tarsus_radius"]], "pos": lower_end, "material": material, "density": legs["tarsus_density"], "friction": legs["tarsus_friction"], "condim": 4})}/>')
            pieces.append(f'<site {self._attrs({"name": f"resident:{body_id}:site:{name}:tarsus", "type": "sphere", "size": [legs["tarsus_radius"] * 0.45], "pos": lower_end, "rgba": [1.0, 1.0, 1.0, 0.0]})}/>')
            pieces.extend(("</body>", "</body>"))
        pieces.append("</body>")
        return "".join(pieces)

    def _compile_model(self) -> None:
        super()._compile_model()
        self._model_signature = hashlib.sha256(_canonical({
            "habitat": self.spec,
            "articulation": self.articulation_spec,
            "mujoco": mujoco.__version__,
            "compiled_xml_sha256": hashlib.sha256(self._xml.encode()).hexdigest(),
        })).hexdigest()
        self._leg_joints: dict[str, dict[str, dict[str, int]]] = {}
        self._tarsus_geoms: dict[str, dict[str, int]] = {}
        self._articulated_links: dict[str, dict[str, int]] = {}
        self._articulated_sites: dict[str, dict[str, int]] = {}
        for resident in self.spec["bodies"]:
            resident_id = resident["id"]
            joints: dict[str, dict[str, int]] = {}
            tarsi: dict[str, int] = {}
            links = {"trunk": self._body_mj[resident_id]}
            sites: dict[str, int] = {}
            for leg in self.articulation_spec["legs"]["layout"]:
                name = leg["name"]
                joints[name] = {
                    kind: mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT,
                        f"resident:{resident_id}:joint:{name}:{kind}",
                    ) for kind in ("hip", "knee")
                }
                for segment in ("upper", "lower"):
                    links[f"{name}:{segment}"] = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY,
                        f"resident:{resident_id}:link:{name}:{segment}",
                    )
                tarsi[name] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM,
                    f"resident:{resident_id}:geom:{name}:tarsus",
                )
                sites[f"{name}:tarsus"] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE,
                    f"resident:{resident_id}:site:{name}:tarsus",
                )
            for side in ("left", "right"):
                sites[f"antenna:{side}"] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE,
                    f"resident:{resident_id}:site:antenna:{side}",
                )
            self._leg_joints[resident_id] = joints
            self._tarsus_geoms[resident_id] = tarsi
            self._articulated_links[resident_id] = links
            self._articulated_sites[resident_id] = sites

    def _apply_crawler_forces(self, body: PhysicsBody, action: dict[str, Any], noise: np.ndarray) -> None:
        """Apply joint servos and a bounded roll/pitch stance reflex.

        The inherited method's trunk traction is intentionally replaced.  The
        only route from forward/yaw commands to translation is articulated foot
        motion followed by MuJoCo contact and friction.
        """
        controller = self.articulation_spec["controller"]
        layout = self.articulation_spec["legs"]["layout"]
        forward = float(action.get("forward", action.get("thrust", 0.0)))
        turn = float(action.get("turn", action.get("yaw", 0.0)))
        activity = max(abs(forward), abs(turn))
        strength = (1.0 - 0.72 * body.fatigue) * (0.18 + 0.82 * body.energy)
        frequency = float(controller["frequency_hz"]) * (0.32 + 0.68 * activity)
        stance_fraction = float(controller["stance_fraction"])
        hip_amplitude = math.radians(float(controller["hip_sweep_degrees"]))
        knee_stance = math.radians(float(controller["knee_stance_degrees"]))
        knee_swing = math.radians(float(controller["knee_swing_degrees"]))
        idle_knee = math.radians(float(controller["idle_knee_degrees"]))
        torque_limit = float(controller["max_joint_torque"]) * strength

        for leg in layout:
            name, side = leg["name"], int(leg["side"])
            side_drive = float(np.clip(
                forward + side * float(controller["turn_gain"]) * turn, -1.0, 1.0
            ))
            if activity < 1e-4 or abs(side_drive) < 1e-4:
                hip_target, knee_target = 0.0, side * idle_knee
            else:
                cycle = (self.time * frequency + float(leg["phase"])) % 1.0
                stride = 0.30 + 0.70 * abs(side_drive)
                if cycle < stance_fraction:
                    progress = cycle / stance_fraction
                    sweep = 1.0 - 2.0 * progress
                    knee_target = side * knee_stance
                else:
                    progress = (cycle - stance_fraction) / (1.0 - stance_fraction)
                    sweep = -1.0 + 2.0 * progress
                    knee_target = side * knee_swing
                hip_target = -side * math.copysign(1.0, side_drive) * hip_amplitude * stride * sweep
            targets = {"hip": hip_target, "knee": knee_target}
            for kind, target in targets.items():
                joint_id = self._leg_joints[body.id][name][kind]
                qadr = int(self.model.jnt_qposadr[joint_id])
                dadr = int(self.model.jnt_dofadr[joint_id])
                kp = float(controller[f"{kind}_kp"])
                kd = float(controller[f"{kind}_kd"])
                torque = kp * (target - float(self.data.qpos[qadr])) - kd * float(self.data.qvel[dadr])
                self.data.qfrc_applied[dadr] = float(np.clip(torque, -torque_limit, torque_limit))

        # A low-gain vestibular stance reflex keeps the trunk over the support
        # polygon.  It supplies torque only; it cannot translate or set height.
        root = self._body_mj[body.id]
        rotation = self.data.xmat[root].reshape(3, 3)
        _, angular = self._velocity(root)
        correction = (
            np.cross(rotation[:, 2], np.array([0.0, 0.0, 1.0]))
            * float(controller["posture_kp"])
            - angular * float(controller["posture_kd"])
        )
        correction[2] = 0.0
        limit = float(controller["max_posture_torque"])
        norm = float(np.linalg.norm(correction))
        if norm > limit:
            correction *= limit / norm
        self.data.xfrc_applied[root, 3:6] += correction

    def sense(self, body_id: str) -> dict[str, Any]:
        values = super().sense(body_id)
        values["tarsal_contact"] = self._tarsal_contacts(body_id)
        joint_position, joint_velocity = [], []
        for leg in self.articulation_spec["legs"]["layout"]:
            for kind in ("hip", "knee"):
                joint_id = self._leg_joints[body_id][leg["name"]][kind]
                joint_position.append(float(self.data.qpos[self.model.jnt_qposadr[joint_id]]))
                joint_velocity.append(float(self.data.qvel[self.model.jnt_dofadr[joint_id]]))
        values["joint_position"] = joint_position
        values["joint_velocity"] = joint_velocity
        values["antenna_position"] = [
            self.data.site_xpos[self._articulated_sites[body_id][f"antenna:{side}"]].astype(float).tolist()
            for side in ("left", "right")
        ]
        return values

    def _odor(self, body: PhysicsBody) -> list[list[float]]:
        """Sample the habitat odor field at the two modeled antenna tips."""
        sigma = 0.82
        antennae = [
            self.data.site_xpos[self._articulated_sites[body.id][f"antenna:{side}"]]
            for side in ("left", "right")
        ]
        result = np.zeros((2, 3), dtype=float)
        for entity in self._entities:
            scent = next((c for c in self._components[entity["id"]] if c.get("type") == "scent"), None)
            if scent is None:
                continue
            food = next((c for c in self._components[entity["id"]] if c.get("type") == "food"), None)
            availability = max(0.0, float(food["amount"])) if food else 1.0
            if availability <= 0.0:
                continue
            source, _ = self._pose(entity["id"])
            for side, antenna in enumerate(antennae):
                distance2 = float(np.dot(source - antenna, source - antenna))
                if distance2 <= (4.0 * sigma) ** 2:
                    result[side, int(scent["odor"])] += (
                        float(scent.get("strength", 1.0)) * availability
                        * math.exp(-distance2 / (2.0 * sigma * sigma))
                    )
        return np.clip(result, 0.0, 4.0).tolist()

    def _tarsal_contacts(self, body_id: str) -> list[float]:
        strength = {name: 0.0 for name in self._tarsus_geoms[body_id]}
        by_geom = {geom: name for name, geom in self._tarsus_geoms[body_id].items()}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            foot = by_geom.get(int(contact.geom1), by_geom.get(int(contact.geom2)))
            if foot is None:
                continue
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            strength[foot] = max(strength[foot], min(1.0, float(np.linalg.norm(force[:3])) / 1.2))
        return [strength[leg["name"]] for leg in self.articulation_spec["legs"]["layout"]]

    @staticmethod
    def _world_quaternion(matrix: np.ndarray) -> list[float]:
        quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quat, matrix)
        return quat.tolist()

    def view(self) -> dict[str, Any]:
        view = super().view()
        articulations = []
        for body_view in view["bodies"]:
            resident_id = body_view["id"]
            links = []
            for name, body_id in self._articulated_links[resident_id].items():
                links.append({
                    "name": name,
                    "position": self.data.xpos[body_id].astype(float).tolist(),
                    "quaternion": self.data.xquat[body_id].astype(float).tolist(),
                })
            geoms = []
            for geom_id, owner in self._geom_resident.items():
                if owner != resident_id:
                    continue
                full_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                kind = int(self.model.geom_type[geom_id])
                type_name = {
                    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
                    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
                }.get(kind, "unknown")
                size_count = 3 if type_name == "ellipsoid" else 1 if type_name == "sphere" else 2
                link_id = int(self.model.geom_bodyid[geom_id])
                link_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, link_id)
                geoms.append({
                    "name": full_name.split(f"resident:{resident_id}:", 1)[-1],
                    "link": "trunk" if link_id == self._body_mj[resident_id] else link_name.split(":link:", 1)[-1],
                    "type": type_name,
                    "size": self.model.geom_size[geom_id, :size_count].astype(float).tolist(),
                    "position": self.data.geom_xpos[geom_id].astype(float).tolist(),
                    "quaternion": self._world_quaternion(self.data.geom_xmat[geom_id]),
                    "color": _hex([*self._geom_rgb(geom_id), 1.0]),
                })
            joints = []
            for leg in self.articulation_spec["legs"]["layout"]:
                for kind in ("hip", "knee"):
                    joint_id = self._leg_joints[resident_id][leg["name"]][kind]
                    joints.append({
                        "name": f"{leg['name']}:{kind}",
                        "position": float(self.data.qpos[self.model.jnt_qposadr[joint_id]]),
                        "velocity": float(self.data.qvel[self.model.jnt_dofadr[joint_id]]),
                        "anchor": self.data.xanchor[joint_id].astype(float).tolist(),
                        "axis": self.data.xaxis[joint_id].astype(float).tolist(),
                        "range": self.model.jnt_range[joint_id].astype(float).tolist(),
                    })
            sites = []
            for name, site_id in self._articulated_sites[resident_id].items():
                sites.append({
                    "name": name,
                    "position": self.data.site_xpos[site_id].astype(float).tolist(),
                    "quaternion": self._world_quaternion(self.data.site_xmat[site_id]),
                })
            articulation = {"id": resident_id, "links": links, "geoms": geoms, "joints": joints, "sites": sites}
            articulations.append(articulation)
            body_view.update({
                "shape": "compound", "size": list(map(float, self.articulation_spec["trunk"]["thorax_size"])),
                "shapes": geoms, "articulation": articulation,
            })
        view["body_model"] = {
            "name": self.articulation_spec["name"],
            "kind": "engineered_hexapod",
            "joint_count_per_resident": 12,
        }
        view["articulations"] = articulations
        return view


__all__ = ["DEFAULT_BODY_SPEC", "ArticulatedWorld"]
