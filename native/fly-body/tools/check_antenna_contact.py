#!/usr/bin/env python3
"""Run one joined native BODY807 antenna-contact and snapshot-replay check."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

import mujoco as mj


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(tensor: dict) -> tuple[float, ...]:
    kind = "f" if tensor["dtype"] == "<f4" else "d"
    return struct.unpack(f"<{tensor['length']}{kind}", base64.b64decode(tensor["base64"]))


def tensor_sha256(values: tuple[float, ...], kind: str) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}{kind}", *values)).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--native-host", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity", default="grain-14")
    parser.add_argument("--resident", default="resident00")
    parser.add_argument("--segment", default="l_arista")
    parser.add_argument("--hold-z", type=float, default=0.01962)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--body-schema",
        type=Path,
        default=root / "research/fly_embodiment/body807-channel-schema.json",
    )
    parser.add_argument(
        "--author-model",
        type=Path,
        default=root / "native/fly-body/assets/neuromechfly-2.1.0-ca65a510-ypr/model/model.xml",
    )
    args = parser.parse_args()
    world_path = args.world.resolve()
    world = json.loads(world_path.read_text())
    scene_path = world_path.parent / world["scene_xml"]
    channel_schema = json.loads(args.body_schema.read_text())
    channel_index = {item["name"]: i for i, item in enumerate(channel_schema["channels"])}
    axes = ("yaw", "pitch", "roll")
    q_indices = [channel_index[f"joint/l_funiculus-l_arista-{axis}/position"] for axis in axes]
    qdot_indices = [channel_index[f"joint/l_funiculus-l_arista-{axis}/angular_velocity"] for axis in axes]
    load_indices = [channel_index[f"joint/l_funiculus-l_arista-{axis}/generalized_load"] for axis in axes]
    contact_indices = [channel_index[f"segment/{args.segment}/contact_force_{axis}"] for axis in "xyz"]
    entity = next(item for item in world["entities"] if item["id"] == args.entity)
    proxy = next(
        item
        for item in world["antenna_contact_proxies"]["compiled"]
        if item["resident"] == args.resident and item["segment"] == args.segment
    )
    resident_row = next(i for i, body in enumerate(world["bodies"]) if body["id"] == args.resident)
    resident_offset = resident_row * 807
    q_indices = [resident_offset + i for i in q_indices]
    qdot_indices = [resident_offset + i for i in qdot_indices]
    load_indices = [resident_offset + i for i in load_indices]
    contact_indices = [resident_offset + i for i in contact_indices]

    process = subprocess.Popen(
        [str(args.native_host.resolve()), "--scene", str(world_path), "--seed", str(args.seed)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def request(payload: dict) -> dict:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        if not response.get("ok"):
            raise RuntimeError(response)
        return response

    assert process.stdout is not None
    ready = json.loads(process.stdout.readline())
    motor = base64.b64encode(struct.pack(f"<{len(world['bodies']) * 92}f", *([0.0] * (len(world["bodies"]) * 92)))).decode()
    hold = [0.0, 0.0, args.hold_z]

    def sample() -> tuple[dict[str, tuple[float, ...]], dict]:
        raw = request({"id": "sample", "command": "sample"})["sample"]
        return ({key: decode(raw[key]) for key in ("body", "qpos", "qvel", "sensordata")}, raw)

    request({"id": "hold-pre", "command": "visitor_force", "entity_id": args.entity, "force": hold})
    request({"id": "advance-pre", "command": "advance", "motor92_base64": motor})
    pre, _ = sample()
    snapshot = request({"id": "snapshot", "command": "snapshot"})["snapshot_base64"]
    request({"id": "hold-contact", "command": "visitor_force", "entity_id": args.entity, "force": hold})
    request({"id": "advance-contact", "command": "advance", "motor92_base64": motor})
    post, post_raw = sample()
    request({"id": "restore", "command": "restore", "snapshot_base64": snapshot})
    request({"id": "hold-replay", "command": "visitor_force", "entity_id": args.entity, "force": hold})
    request({"id": "advance-replay", "command": "advance", "motor92_base64": motor})
    replay, _ = sample()
    request({"id": "close", "command": "close"})
    process.wait()

    qpos_address = entity["free_qpos_address"]
    dof_address = entity["free_dof_address"]
    author = mj.MjModel.from_xml_path(str(args.author_model.resolve()))
    compiled = mj.MjModel.from_xml_path(str(scene_path))
    dynamics = []
    for side in "lr":
        for segment in ("pedicel", "funiculus", "arista"):
            name = f"{side}_{segment}"
            source_id = mj.mj_name2id(author, mj.mjtObj.mjOBJ_BODY, f"fly/{name}")
            compiled_id = mj.mj_name2id(compiled, mj.mjtObj.mjOBJ_BODY, f"{args.resident}/{name}")
            exact = (
                float(author.body_mass[source_id]) == float(compiled.body_mass[compiled_id])
                and list(author.body_inertia[source_id]) == list(compiled.body_inertia[compiled_id])
            )
            dynamics.append(
                {
                    "segment": name,
                    "author_mass_model_units": float(author.body_mass[source_id]),
                    "compiled_mass_model_units": float(compiled.body_mass[compiled_id]),
                    "author_inertia_model_units": [float(x) for x in author.body_inertia[source_id]],
                    "compiled_inertia_model_units": [float(x) for x in compiled.body_inertia[compiled_id]],
                    "exact": exact,
                }
            )

    def select(values: tuple[float, ...], indices: list[int]) -> list[float]:
        return [values[i] for i in indices]

    contact = select(post["body"], contact_indices)
    hashes = {
        key: tensor_sha256(value, "f" if key == "body" else "d") for key, value in post.items()
    }
    report = {
        "format": "chreatures-fly-antenna-touch-receipt-v1",
        "scope": "Joined native MuJoCo host, zero MOTOR92, vertical hold on an existing movable grain, actual distal antenna proxy contact, BODY807 transduction, and exact snapshot replay.",
        "identity": {
            "world_sha256": sha256(world_path),
            "scene_xml_sha256": sha256(scene_path),
            "physics_manifest_sha256": sha256(world_path.with_name("physics.json")),
            "habitat_plan_sha256": sha256(world_path.with_name("habitat-plan.json")),
            "antenna_contact_proxy_artifact_sha256": world["antenna_contact_proxies"]["artifact_sha256"],
            "physical_sensory_schema_sha256": world["physical_sensory_schema_sha256"],
            "body807_schema_sha256": world["sensory_schema_sha256"],
            "native_host_binary_sha256": sha256(args.native_host.resolve()),
            "initial_snapshot_sha256": ready["initial_snapshot_sha256"],
        },
        "compiled_counts": world["compiled_counts"],
        "control_dt_s": world["control_dt"],
        "physics_dt_s": world["physics_dt"],
        "motor92": "all_zero",
        "visitor_force_model_units": hold,
        "visitor_force_note": "Vertical force balances configured grain model weight; the model mass scale is inferred and this is not an SI-newton claim.",
        "contact_dynamics": world["antenna_contact_proxies"]["physics_contract"]["contact_parameters"],
        "compiled_indices": {
            "antenna_body": proxy["body_id"],
            "antenna_proxy_geom": proxy["geom_id"],
            "movable_body": entity["body"],
            "movable_geoms": entity["geoms"],
            "movable_free_qpos_address": qpos_address,
            "movable_free_dof_address": dof_address,
        },
        "body807_indices": {"q": q_indices, "qdot": qdot_indices, "generalized_load": load_indices, "segment_contact_force": contact_indices},
        "pre_contact": {
            "time_s": world["control_dt"],
            "antenna_q": select(pre["body"], q_indices),
            "antenna_qdot_rad_s": select(pre["body"], qdot_indices),
            "antenna_generalized_load_model_units": select(pre["body"], load_indices),
            "antenna_segment_contact_force_model_units": select(pre["body"], contact_indices),
            "grain_position_mm": list(pre["qpos"][qpos_address : qpos_address + 3]),
            "grain_linear_velocity_mm_s": list(pre["qvel"][dof_address : dof_address + 3]),
        },
        "contact_tick": {
            "time_s": post_raw["time"],
            "antenna_q": select(post["body"], q_indices),
            "antenna_qdot_rad_s": select(post["body"], qdot_indices),
            "antenna_generalized_load_model_units": select(post["body"], load_indices),
            "antenna_segment_contact_force_model_units": contact,
            "antenna_segment_contact_force_norm_model_units": math.sqrt(sum(x * x for x in contact)),
            "grain_position_mm": list(post["qpos"][qpos_address : qpos_address + 3]),
            "grain_linear_velocity_mm_s": list(post["qvel"][dof_address : dof_address + 3]),
        },
        "delta": {
            "antenna_q": [post["body"][i] - pre["body"][i] for i in q_indices],
            "antenna_qdot_rad_s": [post["body"][i] - pre["body"][i] for i in qdot_indices],
            "grain_position_mm": [post["qpos"][qpos_address + i] - pre["qpos"][qpos_address + i] for i in range(3)],
            "grain_linear_velocity_mm_s": [post["qvel"][dof_address + i] - pre["qvel"][dof_address + i] for i in range(3)],
        },
        "author_body_preservation": {"all_exact": all(row["exact"] for row in dynamics), "segments": dynamics},
        "replay": {"exact": all(post[key] == replay[key] for key in post), "post_hashes": hashes, "replay_hashes": {key: tensor_sha256(value, "f" if key == "body" else "d") for key, value in replay.items()}},
        "claims": {
            "contact": "Measured through the current BODY807 segment-contact path.",
            "deflection": "Measured motion of the author's passive distal arista joints.",
            "reciprocity": "The free grain gains lateral position and velocity under a vertical-only applied hold during resolved contact.",
            "biology": "The proxy and compliance are engineered from author mesh geometry; no measured sensillum force calibration or behavioral-response claim.",
        },
    }
    if report["contact_tick"]["antenna_segment_contact_force_norm_model_units"] <= 0.0:
        raise RuntimeError("antenna contact did not reach BODY807")
    if not report["author_body_preservation"]["all_exact"] or not report["replay"]["exact"]:
        raise RuntimeError("antenna dynamics preservation or snapshot replay differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "contact_force_norm_model_units": report["contact_tick"]["antenna_segment_contact_force_norm_model_units"], "snapshot_replay_exact": True}))


if __name__ == "__main__":
    main()
