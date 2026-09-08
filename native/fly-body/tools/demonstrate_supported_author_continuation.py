#!/usr/bin/env python3
"""Branch a failed recorded child probe into a supported author continuation.

This is an offline physical demonstration tool. It preserves the failed branch,
restores the last actually supported MuJoCo snapshot, and emits a distinct
sibling branch made only of legal M92 actions. It does not restore CNS state or
claim self-righting; later collection must recompute private CNS history from
the checkpoint before using a sibling as learning evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from replay_author_teacher import SupportedAuthorContinuation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(tensor: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(tensor["dtype"])
    return np.frombuffer(base64.b64decode(tensor["base64"], validate=True), dtype=dtype).copy()


class NativeWorld:
    def __init__(self, binary: Path, world: Path, seed: int) -> None:
        self.process = subprocess.Popen(
            [str(binary.resolve()), "--scene", str(world.resolve()), "--seed", str(seed)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        self.ready = json.loads(self.process.stdout.readline())
        if not self.ready.get("ok"):
            raise RuntimeError(self.ready)
        self.serial = 0

    def request(self, command: str, **values: Any) -> dict[str, Any]:
        self.serial += 1
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps({"id": self.serial, "command": command, **values}, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if not response.get("ok"):
            raise RuntimeError(response)
        return response

    def sample(self) -> dict[str, Any]:
        return self.request("sample")["sample"]

    def advance(self, motor92: np.ndarray) -> None:
        value = np.ascontiguousarray(motor92, dtype="<f4")
        encoded = base64.b64encode(value.tobytes()).decode()
        self.request("advance", motor92_base64=encoded)

    def snapshot(self) -> bytes:
        encoded = self.request("snapshot")["snapshot_base64"]
        return base64.b64decode(encoded, validate=True)

    def restore(self, snapshot: bytes) -> None:
        self.request("restore", snapshot_base64=base64.b64encode(snapshot).decode())

    def close(self) -> str:
        if self.process.poll() is None:
            self.request("close")
        self.process.wait(timeout=30)
        assert self.process.stderr is not None
        return self.process.stderr.read()


def extract_state(packet: dict[str, Any], world: dict[str, Any]) -> dict[str, np.ndarray]:
    qpos = decode(packet["qpos"])
    rotations = decode(packet["bodyRotations"]).reshape(-1, 3, 3)
    positions = decode(packet["bodyPositions"]).reshape(-1, 3)
    sensors = decode(packet["sensordata"])
    control = decode(packet["ctrl"])
    residents = len(world["bodies"])
    joint = np.empty((residents, 84), np.float64)
    applied = np.empty((residents, 90), np.float64)
    contact = np.empty((residents, 6), bool)
    root_up = np.empty(residents, np.float64)
    root_position = np.empty((residents, 3), np.float64)
    for row, (body, detail) in enumerate(zip(world["bodies"], world["residents"])):
        q_by_name = {item["semantic_id"]: qpos[int(item["qpos_address"])] for item in detail["joint_dofs126"]}
        names = [item["semantic_id"] for item in detail["actuators90"][:84]]
        if any(not name.endswith("-position") or name[:-9] not in q_by_name for name in names):
            raise RuntimeError("compiled active-servo order cannot be mapped to qpos")
        joint[row] = [q_by_name[name[:-9]] for name in names]
        applied[row] = control[np.asarray(body["actuators"], dtype=np.int64)]
        contact[row] = [sensors[int(item["data_address"])] > 0.0 for item in detail["contact_sensors6x16"]]
        root = int(body["root"])
        root_up[row] = rotations[root, 2, 2]
        root_position[row] = positions[root]
    return {
        "joint84_rad": joint,
        "applied90": applied,
        "foot_contact": contact,
        "root_up": root_up,
        "root_position_mm": root_position,
        "qpos": qpos,
        "qvel": decode(packet["qvel"]),
    }


def supported(state: dict[str, np.ndarray], resident: int) -> bool:
    return bool(state["root_up"][resident] >= 0.65 and state["foot_contact"][resident].sum() >= 3)


def append_trace(trace: dict[str, list[np.ndarray]], state: dict[str, np.ndarray]) -> None:
    for key in ("joint84_rad", "applied90", "foot_contact", "root_up", "root_position_mm"):
        trace.setdefault(key, []).append(np.asarray(state[key]).copy())


def trace_arrays(trace: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(values) for key, values in trace.items()}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--native-host", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=root / "native/fly-body/assets/author-step-bank-v1/trajectory-bank.npz")
    parser.add_argument("--bank-manifest", type=Path, default=root / "native/fly-body/assets/author-step-bank-v1/manifest.json")
    parser.add_argument("--settle-ticks", type=int, default=40)
    parser.add_argument("--continuation-ticks", type=int, default=128)
    parser.add_argument("--preferred-resident", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    world_path = args.world.resolve()
    world = json.loads(world_path.read_text())
    if len(world["bodies"]) != 4 or world["control_dt"] != 0.01 or world["physics_dt"] != 0.0001:
        raise RuntimeError("demonstrator requires the frozen B4 100 Hz/0.1 ms physical contract")
    probe_receipt = json.loads(args.probe_receipt.read_text())
    if sha256(args.probe) != probe_receipt["probe_sha256"]:
        raise RuntimeError("recorded child-probe artifact hash differs")
    probe = np.load(args.probe, allow_pickle=False)
    probe_motor = np.asarray(probe["motor92"], dtype=np.float32)
    if probe_motor.ndim != 3 or probe_motor.shape[1:] != (4, 92):
        raise ValueError("probe must be an exact Tx4x92 recorded CNS action block")
    if np.any(probe_motor[:, :, :84] < -1.0) or np.any(probe_motor[:, :, :84] > 1.0):
        raise ValueError("probe servo actions leave normalized physical bounds")
    if np.any(probe_motor[:, :, 84:92] < 0.0) or np.any(probe_motor[:, :, 84:92] > 1.0):
        raise ValueError("probe adhesion/physiology actions leave normalized bounds")

    bank_manifest = json.loads(args.bank_manifest.read_text())
    if sha256(args.bank) != bank_manifest["bank"]["sha256"]:
        raise RuntimeError("sealed author trajectory bank hash differs")
    bank = np.load(args.bank, allow_pickle=False)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    physical = NativeWorld(args.native_host, world_path, args.seed)
    residents = len(world["bodies"])
    hold = np.zeros((residents, 92), np.float32)
    hold[:, 84:90] = 1.0
    branch_trace: dict[str, list[np.ndarray]] = {}
    continuation_trace: dict[str, list[np.ndarray]] = {}
    last_supported: list[tuple[int, bytes, dict[str, np.ndarray]] | None] = [None] * residents
    try:
        for _ in range(args.settle_ticks):
            physical.advance(hold)
        state = extract_state(physical.sample(), world)
        append_trace(branch_trace, state)
        initial_supported = [supported(state, row) for row in range(residents)]
        snapshot = physical.snapshot()
        for row, is_supported in enumerate(initial_supported):
            if is_supported:
                last_supported[row] = (0, snapshot, state)

        lost_support: list[int] = []
        for index, motor in enumerate(probe_motor, start=1):
            physical.advance(motor)
            state = extract_state(physical.sample(), world)
            append_trace(branch_trace, state)
            now_supported = [supported(state, row) for row in range(residents)]
            snap = physical.snapshot() if any(now_supported) else b""
            for row in range(residents):
                before = last_supported[row]
                if now_supported[row] and row not in lost_support:
                    last_supported[row] = (index, snap, state)
                elif before is not None and row not in lost_support:
                    lost_support.append(row)

        candidates = [args.preferred_resident, *range(residents)]
        selected = next((row for row in candidates if row in lost_support), None)
        if selected is None or last_supported[selected] is None:
            raise RuntimeError("recorded probe did not leave the author continuation domain in this scene")
        restore_tick, restore_snapshot, restore_source = last_supported[selected]
        snapshot_path = output / "last-supported-physical.snapshot"
        snapshot_path.write_bytes(restore_snapshot)
        physical.restore(restore_snapshot)
        restored_packet = physical.sample()
        restored = extract_state(restored_packet, world)
        restore_exact = all(
            np.array_equal(restored[key], restore_source[key])
            for key in ("qpos", "qvel", "joint84_rad", "applied90", "foot_contact", "root_up", "root_position_mm")
        )
        if not restore_exact or not supported(restored, selected):
            raise RuntimeError("last-supported physical snapshot did not restore exactly")

        body = world["bodies"][selected]
        continuation = SupportedAuthorContinuation(
            bank["joint_targets_rad"], bank["adhesion"], bank["teacher_neutral_rad"],
            np.asarray(body["neutral"], np.float64), np.asarray(body["control_ranges"], np.float64),
        )
        fit = continuation.fit(
            restored["joint84_rad"][selected], restored["applied90"][selected, :84],
            restored["foot_contact"][selected], float(restored["root_up"][selected]),
        )
        if not fit.eligible:
            raise RuntimeError("restored state unexpectedly lies outside author continuation domain")
        append_trace(continuation_trace, restored)
        delivered = hold.copy()
        delivered[selected, :84] = 0.0
        delivered[selected, 84:90] = restored["applied90"][selected, 84:90]
        bridge_complete_tick: int | None = None
        continuation_lost_tick: int | None = None
        command_rows = []
        for tick in range(args.continuation_ticks):
            command = continuation.command(
                restored["joint84_rad"][selected], restored["applied90"][selected, :84],
                restored["applied90"][selected, 84:90], restored["foot_contact"][selected],
                float(restored["root_up"][selected]), control_dt_s=float(world["control_dt"]),
            )
            if command is None:
                continuation_lost_tick = tick
                break
            delivered[selected] = command.normalized_motor92
            command_rows.append(np.r_[
                command.servo_targets_rad, command.adhesion,
                command.phase_cycles, command.amplitude, float(command.bridge_complete),
                command.max_servo_delta_rad, command.max_adhesion_delta,
            ])
            if command.bridge_complete and bridge_complete_tick is None:
                bridge_complete_tick = tick
            physical.advance(delivered)
            restored = extract_state(physical.sample(), world)
            append_trace(continuation_trace, restored)

        stderr = physical.close()
    except BaseException:
        if physical.process.poll() is None:
            physical.process.terminate()
            physical.process.wait(timeout=30)
        raise

    branch = trace_arrays(branch_trace)
    sibling = trace_arrays(continuation_trace)
    trace_path = output / "paired-branches.npz"
    np.savez_compressed(
        trace_path,
        failed_probe_motor92=probe_motor,
        continuation_commands=np.asarray(command_rows, dtype=np.float64),
        **{f"failed_{key}": value for key, value in branch.items()},
        **{f"sibling_{key}": value for key, value in sibling.items()},
    )
    failed_end = len(probe_motor)
    receipt = {
        "format": "chreatures-supported-author-continuation-demonstration-v1",
        "status": "executed",
        "scope": "Paired offline branches in the joined native MuJoCo host; no CNS or runtime policy bypass.",
        "identity": {
            "world": str(world_path), "world_sha256": sha256(world_path),
            "scene_xml_sha256": world["source_mjcf_sha256"],
            "native_host": str(args.native_host.resolve()), "native_host_sha256": sha256(args.native_host.resolve()),
            "author_bank_sha256": sha256(args.bank), "author_source_revision": bank_manifest["source"]["git_revision"],
            "recorded_probe_sha256": sha256(args.probe),
            "recorded_probe_source_episode_sha256": probe_receipt["source_episode_sha256"],
            "last_supported_snapshot_sha256": sha256(snapshot_path),
        },
        "timing": {"physics_dt_s": world["physics_dt"], "control_dt_s": world["control_dt"], "settle_ticks": args.settle_ticks},
        "failed_branch": {
            "recorded_source_ticks": probe_receipt["source_ticks"], "selected_resident": selected,
            "initial_supported": initial_supported, "residents_losing_support": lost_support,
            "selected_last_supported_probe_tick": restore_tick,
            "selected_end_root_up": float(branch["root_up"][failed_end, selected]),
            "selected_end_contact_feet": int(branch["foot_contact"][failed_end, selected].sum()),
            "preserved": True,
        },
        "restored_sibling": {
            "provenance": "exact native physical snapshot at the last support-gated state on the failed branch",
            "restore_exact": restore_exact,
            "private_cns_state_included": False,
            "fit": fit.__dict__,
            "requested_ticks": args.continuation_ticks,
            "delivered_ticks": len(command_rows),
            "bridge_complete_tick": bridge_complete_tick,
            "lost_support_tick": continuation_lost_tick,
            "end_root_up": float(sibling["root_up"][-1, selected]),
            "end_contact_feet": int(sibling["foot_contact"][-1, selected].sum()),
            "max_servo_slew_rad_per_tick": float(max((row[-2] for row in command_rows), default=0.0)),
            "max_adhesion_slew_unitless_per_tick": float(max((row[-1] for row in command_rows), default=0.0)),
        },
        "trace": {"file": trace_path.name, "sha256": sha256(trace_path)},
        "native_stderr": stderr.strip(),
        "limits": [
            "Eligibility requires measured upright support and at least three contacting feet; excluded fallen states are not self-righting cases.",
            "The fitted phase and amplitude are an engineered state match to the sealed author trajectory, not a biological recovery law.",
            "The sibling contains physical state only. Any later learning branch must recompute private CNS history from its checkpoint without a sensory discontinuity.",
        ],
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path),
        "trace_sha256": sha256(trace_path), "selected_resident": selected,
        "lost_support": lost_support, "continuation_ticks": len(command_rows),
        "continuation_lost_tick": continuation_lost_tick,
    }, indent=2))


if __name__ == "__main__":
    main()
