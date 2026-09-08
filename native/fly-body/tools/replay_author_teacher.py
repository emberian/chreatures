#!/usr/bin/env python3
"""Replay the sealed author gait bank in the actual B4 MuJoCo scene.

This is a one-off physical feasibility diagnostic.  It is not a production
controller and does not pass world state around the CNS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
MODES = ("forward", "turn_left", "stop", "turn_right")
TRIPOD_PHASE = np.asarray([0.0, 0.5, 0.0, 0.5, 0.0, 0.5], dtype=np.float64)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def periodic_leg_sample(samples: np.ndarray, phases: np.ndarray) -> np.ndarray:
    count = len(samples)
    result = np.empty(42, dtype=np.float64)
    for leg, phase in enumerate(phases):
        scaled = (phase % 1.0) * count
        lower = int(math.floor(scaled))
        upper = (lower + 1) % count
        fraction = scaled - lower
        section = slice(leg * 7, (leg + 1) * 7)
        result[section] = samples[lower, section] * (1.0 - fraction) + samples[upper, section] * fraction
    return result


def periodic_adhesion(samples: np.ndarray, phases: np.ndarray) -> np.ndarray:
    indices = np.floor(np.mod(phases, 1.0) * len(samples)).astype(np.int64)
    return np.asarray([samples[index, leg] for leg, index in enumerate(indices)], dtype=np.float64)


@dataclass(frozen=True)
class ContinuationFit:
    """A support-gated fit to the already sealed FlyGym trajectory bank."""

    eligible: bool
    reason: str
    phase_cycles: float | None
    amplitude: float | None
    score: float | None
    joint_rms_normalized: float | None
    delivered_target_rms_normalized: float | None
    contacting_feet_released_by_candidate: int | None


@dataclass(frozen=True)
class ContinuationCommand:
    """One legal physical command and diagnostics from the offline teacher."""

    servo_targets_rad: np.ndarray
    adhesion: np.ndarray
    normalized_motor92: np.ndarray
    phase_cycles: float
    amplitude: float
    requested_amplitude: float
    mode: str
    bridge_complete: bool
    max_servo_delta_rad: float
    max_adhesion_delta: float


class SupportedAuthorContinuation:
    """Continue the sealed author step from an actually supported state.

    This class is an offline demonstration source. It observes physical support,
    joint angles and the command that was actually delivered at takeover. It is
    neither a deployed controller nor a self-righting mechanism.
    """

    def __init__(
        self,
        joint_targets_rad: np.ndarray,
        adhesion: np.ndarray,
        teacher_neutral_rad: np.ndarray,
        fixture_neutral84_rad: np.ndarray,
        control_ranges84_rad: np.ndarray,
        *,
        frequency_hz: float = 1.5,
        amplitude_bounds: tuple[float, float] = (0.05, 0.50),
        servo_slew_rad_per_tick: float = 0.04,
        adhesion_slew_per_tick: float = 0.25,
        upright_min: float = 0.65,
        contact_feet_min: int = 3,
        delivered_target_weight: float = 1.0,
        contact_release_penalty: float = 0.25,
        bridge_tolerance_rad: float = 0.04,
        bridge_stable_ticks: int = 2,
        amplitude_ramp_supported_ticks: int = 64,
    ) -> None:
        self.targets = np.asarray(joint_targets_rad, dtype=np.float64)
        self.adhesion_bank = np.asarray(adhesion, dtype=np.float64)
        self.teacher_neutral = np.asarray(teacher_neutral_rad, dtype=np.float64)
        self.neutral = np.asarray(fixture_neutral84_rad, dtype=np.float64)
        self.ranges = np.asarray(control_ranges84_rad, dtype=np.float64)
        if self.targets.ndim != 2 or self.targets.shape[1] != 42:
            raise ValueError("author trajectory must have 42 walking coordinates")
        if self.adhesion_bank.shape != (len(self.targets), 6):
            raise ValueError("author adhesion must align with the trajectory bank")
        if self.teacher_neutral.shape != (42,) or self.neutral.shape != (84,):
            raise ValueError("continuation requires teacher42 and fixture84 neutral angles")
        if self.ranges.shape != (84, 2) or np.any(self.ranges[:, 1] <= self.ranges[:, 0]):
            raise ValueError("continuation requires ordered 84-servo physical ranges")
        if not (0.0 <= amplitude_bounds[0] <= amplitude_bounds[1] <= 1.0):
            raise ValueError("amplitude bounds must be inside the sealed author's scale")
        if frequency_hz <= 0.0 or servo_slew_rad_per_tick <= 0.0:
            raise ValueError("frequency and physical-radian slew must be positive")
        if not 0.0 < adhesion_slew_per_tick <= 1.0:
            raise ValueError("adhesion slew must be in unitless activation per tick")
        self.frequency_hz = float(frequency_hz)
        self.amplitude_bounds = tuple(float(x) for x in amplitude_bounds)
        self.servo_slew = float(servo_slew_rad_per_tick)
        self.adhesion_slew = float(adhesion_slew_per_tick)
        self.upright_min = float(upright_min)
        self.contact_feet_min = int(contact_feet_min)
        self.delivered_weight = float(delivered_target_weight)
        self.release_penalty = float(contact_release_penalty)
        self.bridge_tolerance = float(bridge_tolerance_rad)
        self.bridge_stable_ticks = int(bridge_stable_ticks)
        if amplitude_ramp_supported_ticks <= 0:
            raise ValueError("amplitude ramp duration must be positive")
        self.amplitude_ramp_ticks = int(amplitude_ramp_supported_ticks)
        self._fit: ContinuationFit | None = None
        self._phase = 0.0
        self._bridge_stable = 0
        self._phase_running = False
        self._amplitude = 0.0
        self._ramp_start = 0.0
        self._ramp_target = 0.0
        self._ramp_progress = 0

    def _supported(self, thorax_up: float, foot_contact: np.ndarray) -> bool:
        contact = np.asarray(foot_contact, dtype=bool)
        return bool(thorax_up >= self.upright_min and contact.shape == (6,) and contact.sum() >= self.contact_feet_min)

    def _desired_walking(self, phase: float, amplitude: float) -> np.ndarray:
        raw = periodic_leg_sample(self.targets, TRIPOD_PHASE + phase)
        return self.neutral[:42] + amplitude * (raw - self.teacher_neutral)

    def fit(
        self,
        actual_joint84_rad: np.ndarray,
        last_delivered_servo84_rad: np.ndarray,
        foot_contact: np.ndarray,
        thorax_up: float,
    ) -> ContinuationFit:
        q = np.asarray(actual_joint84_rad, dtype=np.float64)
        delivered = np.asarray(last_delivered_servo84_rad, dtype=np.float64)
        contact = np.asarray(foot_contact, dtype=bool)
        if q.shape != (84,) or delivered.shape != (84,) or contact.shape != (6,):
            raise ValueError("fit requires q84, delivered-servo84 and foot-contact6")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(delivered)):
            raise ValueError("fit inputs must be finite model-space radians")
        if not self._supported(thorax_up, contact):
            result = ContinuationFit(False, "outside-supported-author-domain", None, None, None, None, None, None)
            self._fit = result
            return result

        # Fit in model-space radians, normalized only by each physical servo's
        # full range so a broad coxa axis cannot dominate a narrow tarsus axis.
        span = self.ranges[:42, 1] - self.ranges[:42, 0]
        q_residual = (q[:42] - self.neutral[:42]) / span
        delivered_residual = (delivered[:42] - self.neutral[:42]) / span
        best: tuple[float, float, float, float, float, int] | None = None
        for index in range(len(self.targets)):
            phase = index / len(self.targets)
            raw = periodic_leg_sample(self.targets, TRIPOD_PHASE + phase)
            direction = (raw - self.teacher_neutral) / span
            denominator = float((1.0 + self.delivered_weight) * np.dot(direction, direction))
            if denominator <= 1e-18:
                amplitude = self.amplitude_bounds[0]
            else:
                numerator = float(np.dot(direction, q_residual) + self.delivered_weight * np.dot(direction, delivered_residual))
                amplitude = float(np.clip(numerator / denominator, *self.amplitude_bounds))
            desired_norm = amplitude * direction
            joint_rms = float(np.sqrt(np.mean(np.square(q_residual - desired_norm))))
            delivered_rms = float(np.sqrt(np.mean(np.square(delivered_residual - desired_norm))))
            candidate_adhesion = periodic_adhesion(self.adhesion_bank, TRIPOD_PHASE + phase) >= 0.5
            released = int(np.logical_and(contact, ~candidate_adhesion).sum())
            score = joint_rms * joint_rms + self.delivered_weight * delivered_rms * delivered_rms + self.release_penalty * released
            candidate = (score, phase, amplitude, joint_rms, delivered_rms, released)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        score, phase, amplitude, joint_rms, delivered_rms, released = best
        result = ContinuationFit(True, "supported-fit", phase, amplitude, score, joint_rms, delivered_rms, released)
        self._fit = result
        self._phase = phase
        self._bridge_stable = 0
        self._phase_running = False
        self._amplitude = amplitude
        self._ramp_start = amplitude
        self._ramp_target = amplitude
        self._ramp_progress = self.amplitude_ramp_ticks
        return result

    def _amplitude_preview(self, requested: float | None) -> tuple[float, float, float, int]:
        # None keeps the existing request so stop/antenna bouts can carry the
        # same supported-tick ramp without restarting or cancelling it.
        target = self._ramp_target if requested is None else float(requested)
        if not self.amplitude_bounds[0] <= target <= self.amplitude_bounds[1]:
            raise ValueError("requested amplitude leaves configured author-bank bounds")
        if target != self._ramp_target:
            start, progress = self._amplitude, 0
        else:
            start, progress = self._ramp_start, self._ramp_progress
        progress = min(progress + 1, self.amplitude_ramp_ticks)
        amplitude = start + (target - start) * progress / self.amplitude_ramp_ticks
        return amplitude, start, target, progress

    def _bounded_command(
        self,
        desired_servo84_rad: np.ndarray,
        adhesion_goal: np.ndarray,
        previous_servo84_rad: np.ndarray,
        previous_adhesion: np.ndarray,
        *,
        phase_cycles: float,
        amplitude: float,
        requested_amplitude: float,
        mode: str,
        bridge_complete: bool,
    ) -> ContinuationCommand:
        desired = np.asarray(desired_servo84_rad, dtype=np.float64)
        previous = np.asarray(previous_servo84_rad, dtype=np.float64)
        adhesion_goal = np.asarray(adhesion_goal, dtype=np.float64)
        previous_adhesion = np.asarray(previous_adhesion, dtype=np.float64)
        if desired.shape != (84,) or previous.shape != (84,):
            raise ValueError("physical servo command requires desired84 and previous84 radians")
        if adhesion_goal.shape != (6,) or previous_adhesion.shape != (6,):
            raise ValueError("physical adhesion command requires desired6 and previous6")
        servo = previous + np.clip(desired - previous, -self.servo_slew, self.servo_slew)
        servo = np.clip(servo, self.ranges[:, 0], self.ranges[:, 1])
        adhesion = previous_adhesion + np.clip(
            adhesion_goal - previous_adhesion, -self.adhesion_slew, self.adhesion_slew
        )
        adhesion = np.clip(adhesion, 0.0, 1.0)
        normalized = np.zeros(92, dtype=np.float32)
        above = servo >= self.neutral
        positive_span = self.ranges[:, 1] - self.neutral
        negative_span = self.neutral - self.ranges[:, 0]
        normalized[:84] = np.where(
            above,
            (servo - self.neutral) / positive_span,
            (servo - self.neutral) / negative_span,
        ).astype(np.float32)
        normalized[:84] = np.clip(normalized[:84], -1.0, 1.0)
        normalized[84:90] = adhesion.astype(np.float32)
        if not np.all(np.isfinite(normalized)):
            raise ValueError("physical normalization produced a non-finite action")
        return ContinuationCommand(
            servo_targets_rad=servo,
            adhesion=adhesion,
            normalized_motor92=normalized,
            phase_cycles=phase_cycles,
            amplitude=amplitude,
            requested_amplitude=requested_amplitude,
            mode=mode,
            bridge_complete=bridge_complete,
            max_servo_delta_rad=float(np.max(np.abs(servo - previous))),
            max_adhesion_delta=float(np.max(np.abs(adhesion - previous_adhesion))),
        )

    def command(
        self,
        actual_joint84_rad: np.ndarray,
        last_delivered_servo84_rad: np.ndarray,
        last_delivered_adhesion: np.ndarray,
        foot_contact: np.ndarray,
        thorax_up: float,
        *,
        control_dt_s: float = 0.01,
        mode: str = "continuation",
        requested_amplitude: float | None = None,
        nonwalking_target84_rad: np.ndarray | None = None,
        advance_state: bool = True,
    ) -> ContinuationCommand | None:
        """Return the next command, or None after support leaves the author domain."""
        if self._fit is None or not self._fit.eligible:
            raise RuntimeError("fit an eligible supported state before requesting continuation")
        q = np.asarray(actual_joint84_rad, dtype=np.float64)
        previous = np.asarray(last_delivered_servo84_rad, dtype=np.float64)
        previous_adhesion = np.asarray(last_delivered_adhesion, dtype=np.float64)
        contact = np.asarray(foot_contact, dtype=bool)
        if q.shape != (84,) or previous.shape != (84,) or previous_adhesion.shape != (6,) or contact.shape != (6,):
            raise ValueError("command requires q84, delivered-servo84, adhesion6 and contact6")
        if not self._supported(thorax_up, contact):
            return None
        if control_dt_s <= 0.0:
            raise ValueError("control interval must be positive")
        if mode not in ("continuation", "stop", "left", "right", "antenna"):
            raise ValueError(f"unsupported author continuation mode: {mode}")
        assert self._fit.amplitude is not None
        amplitude, ramp_start, ramp_target, ramp_progress = self._amplitude_preview(requested_amplitude)
        desired = self.neutral.copy()
        if mode in ("continuation", "left", "right"):
            raw = periodic_leg_sample(self.targets, TRIPOD_PHASE + self._phase)
            leg_scale = np.ones(6, np.float64)
            if mode == "left":
                leg_scale[:3] = 0.35
            elif mode == "right":
                leg_scale[3:] = 0.35
            desired[:42] += np.repeat(leg_scale, 7) * amplitude * (raw - self.teacher_neutral)
        if nonwalking_target84_rad is not None:
            nonwalking = np.asarray(nonwalking_target84_rad, dtype=np.float64)
            if mode != "antenna" or nonwalking.shape != (84,):
                raise ValueError("nonwalking target84 is accepted only in antenna mode")
            if not np.allclose(nonwalking[:42], self.neutral[:42], rtol=0.0, atol=1e-12):
                raise ValueError("antenna target must keep walking42 at fixture neutral")
            desired[42:] = nonwalking[42:]
        elif mode == "antenna":
            desired[42:] = self.neutral[42:]

        # The bridge completes when the command itself has reached the fitted
        # trajectory without a discontinuity. Physical tracking remains an
        # observed outcome and is not used to pretend a fallen state recovered.
        servo_preview = previous + np.clip(desired - previous, -self.servo_slew, self.servo_slew)
        servo_preview = np.clip(servo_preview, self.ranges[:, 0], self.ranges[:, 1])
        command_error = float(np.max(np.abs(servo_preview - desired)))
        if command_error <= self.bridge_tolerance:
            bridge_stable = self._bridge_stable + 1
        else:
            bridge_stable = 0
        bridge_complete = self._phase_running or bridge_stable >= self.bridge_stable_ticks
        author_adhesion = periodic_adhesion(self.adhesion_bank, TRIPOD_PHASE + self._phase)
        command_phase = self._phase
        if mode in ("stop", "antenna"):
            adhesion_goal = contact.astype(np.float64)
        else:
            adhesion_goal = author_adhesion if bridge_complete else np.maximum(
                author_adhesion, contact.astype(np.float64)
            )
        result = self._bounded_command(
            desired, adhesion_goal, previous, previous_adhesion,
            phase_cycles=command_phase, amplitude=amplitude,
            requested_amplitude=ramp_target, mode=mode, bridge_complete=bridge_complete,
        )
        if advance_state:
            self._bridge_stable = bridge_stable
            self._phase_running = bridge_complete
            self._amplitude = amplitude
            self._ramp_start = ramp_start
            self._ramp_target = ramp_target
            self._ramp_progress = ramp_progress
        if bridge_complete and advance_state:
            self._phase = (self._phase + self.frequency_hz * control_dt_s) % 1.0
        return result

    def neutral_command(
        self,
        last_delivered_servo84_rad: np.ndarray,
        last_delivered_adhesion: np.ndarray,
        foot_contact: np.ndarray,
        *,
        advance_state: bool = True,
    ) -> ContinuationCommand:
        """Bounded neutral/contact-hold action, valid outside the author domain.

        ``advance_state`` is accepted for the same teacher call boundary as
        :meth:`command`; neutral correction never changes author phase, bridge,
        or amplitude state.
        """
        del advance_state
        contact = np.asarray(foot_contact, dtype=bool)
        if contact.shape != (6,):
            raise ValueError("neutral correction requires foot-contact6")
        amplitude = self._amplitude if self._fit is not None and self._fit.eligible else 0.0
        return self._bounded_command(
            self.neutral, contact.astype(np.float64),
            last_delivered_servo84_rad, last_delivered_adhesion,
            phase_cycles=self._phase, amplitude=amplitude,
            requested_amplitude=(
                self._ramp_target if self._fit is not None and self._fit.eligible else amplitude
            ),
            mode="neutral-correction", bridge_complete=self._phase_running,
        )


def yaw(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation).reshape(3, 3)
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def main() -> None:
    import mujoco as mj

    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=here / "scenes/training-4/scene.xml")
    parser.add_argument("--physics", type=Path, default=here / "scenes/training-4/physics.json")
    parser.add_argument("--bank", type=Path, default=here / "assets/author-step-bank-v1/trajectory-bank.npz")
    parser.add_argument("--bank-manifest", type=Path, default=here / "assets/author-step-bank-v1/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--settle", type=float, default=0.1)
    parser.add_argument("--frequency", type=float, default=12.0)
    parser.add_argument("--adhesion-gain", type=float, default=40.0)
    args = parser.parse_args()

    scene = args.scene.resolve()
    physics = json.loads(args.physics.read_text())
    bank_manifest = json.loads(args.bank_manifest.read_text())
    if sha256(scene) != physics["source_mjcf_sha256"]:
        raise RuntimeError("scene hash differs from compiled physics manifest")
    if sha256(args.bank) != bank_manifest["bank"]["sha256"]:
        raise RuntimeError("teacher trajectory bank hash differs")
    bank = np.load(args.bank, allow_pickle=False)
    targets = np.asarray(bank["joint_targets_rad"], dtype=np.float64)
    teacher_neutral = np.asarray(bank["teacher_neutral_rad"], dtype=np.float64)

    model = mj.MjModel.from_xml_path(str(scene))
    if len(physics["bodies"]) != 4 or model.nu != 360:
        raise RuntimeError("the feasibility demonstration requires the pinned B4 scene")
    if abs(float(model.opt.timestep) - 0.0001) > 1e-15:
        raise RuntimeError("unexpected physics timestep")
    substeps = 100
    control_dt = substeps * float(model.opt.timestep)
    ticks = round(args.duration / control_dt)
    settle_ticks = round(args.settle / control_dt)
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)

    bodies = physics["bodies"]
    detailed = physics["residents"]
    for body in bodies:
        adhesion_ids = body["actuators"][84:90]
        model.actuator_gainprm[adhesion_ids, 0] = args.adhesion_gain
        data.ctrl[body["actuators"][:84]] = body["neutral"]
        data.ctrl[adhesion_ids] = 1.0
    for _ in range(settle_ticks * substeps):
        mj.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        raise RuntimeError("non-finite state during neutral settling")

    initial_position = np.asarray([data.xpos[body["root"]].copy() for body in bodies])
    initial_rotation = np.asarray([data.xmat[body["root"]].copy().reshape(3, 3) for body in bodies])
    initial_yaw = np.asarray([yaw(matrix) for matrix in initial_rotation])
    control_trace = np.empty((ticks, 4, 90), dtype=np.float64)
    walking_trace = np.empty((ticks, 4, 42), dtype=np.float64)
    contact_trace = np.empty((ticks, 4, 6, 16), dtype=np.float64)
    root_pose = np.empty((ticks + 1, 4, 7), dtype=np.float64)
    joint_qpos = np.empty((ticks + 1, 4, 126), dtype=np.float64)

    def capture(index: int) -> None:
        for resident, body in enumerate(bodies):
            root_pose[index, resident, :3] = data.xpos[body["root"]]
            root_pose[index, resident, 3:] = data.xquat[body["root"]]
            joint_qpos[index, resident] = data.qpos[body["qpos"]]

    capture(0)
    tripod = np.asarray([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    magnitudes = (
        np.ones(6),
        np.asarray([0.35, 0.35, 0.35, 1.0, 1.0, 1.0]),
        np.zeros(6),
        np.asarray([1.0, 1.0, 1.0, 0.35, 0.35, 0.35]),
    )
    for tick in range(ticks):
        phases = tripod + args.frequency * tick * control_dt
        blend = min(1.0, (tick + 1) * control_dt / 0.25)
        sampled = periodic_leg_sample(targets, phases)
        sampled_adhesion = periodic_adhesion(bank["adhesion"], phases)
        for resident, (body, mode, magnitude) in enumerate(zip(bodies, MODES, magnitudes)):
            controls = np.zeros(90, dtype=np.float64)
            controls[:84] = body["neutral"]
            if mode == "stop":
                walking = np.asarray(body["neutral"][:42], dtype=np.float64)
                adhesion = np.ones(6, dtype=np.float64)
            else:
                expanded = np.repeat(magnitude, 7)
                desired = teacher_neutral + expanded * (sampled - teacher_neutral)
                walking = np.asarray(body["neutral"][:42]) + blend * (desired - np.asarray(body["neutral"][:42]))
                adhesion = sampled_adhesion if blend >= 1.0 else np.ones(6, dtype=np.float64)
            controls[:42] = walking
            controls[84:90] = adhesion
            actuator_ids = body["actuators"]
            ranges = np.asarray([model.actuator_ctrlrange[index] for index in actuator_ids])
            if np.any(controls < ranges[:, 0]) or np.any(controls > ranges[:, 1]):
                raise RuntimeError(f"{mode} target outside physical control range")
            data.ctrl[actuator_ids] = controls
            control_trace[tick, resident] = controls
            walking_trace[tick, resident] = walking
        for _ in range(substeps):
            mj.mj_step(model, data)
        if not all(np.all(np.isfinite(array)) for array in (data.qpos, data.qvel, data.sensordata)):
            raise RuntimeError(f"non-finite state after control tick {tick}")
        for resident, sensors in enumerate(detailed):
            for leg, sensor in enumerate(sensors["contact_sensors6x16"]):
                address = sensor["data_address"]
                contact_trace[tick, resident, leg] = data.sensordata[address:address + 16]
        capture(tick + 1)

    final_position = root_pose[-1, :, :3]
    final_yaw = np.asarray([yaw(data.xmat[body["root"]]) for body in bodies])
    outcomes = []
    for resident, mode in enumerate(MODES):
        displacement = final_position[resident] - initial_position[resident]
        forward_axis = initial_rotation[resident, :, 0]
        left_axis = initial_rotation[resident, :, 1]
        outcomes.append(
            {
                "resident": resident,
                "mode": mode,
                "displacement_world_mm": [float(x) for x in displacement],
                "forward_displacement_mm": float(displacement @ forward_axis),
                "lateral_displacement_mm": float(displacement @ left_axis),
                "yaw_change_rad": float(math.atan2(math.sin(final_yaw[resident] - initial_yaw[resident]), math.cos(final_yaw[resident] - initial_yaw[resident]))),
                "root_height_range_mm": [float(root_pose[:, resident, 2].min()), float(root_pose[:, resident, 2].max())],
                "leg_contact_found_fraction": [float(x) for x in (contact_trace[:, resident, :, 0] > 0).mean(axis=0)],
                "mean_contact_force_model_units": float(np.linalg.norm(contact_trace[:, resident, :, 1:4], axis=-1).mean()),
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "teacher-replay-trace.npz"
    np.savez_compressed(
        trace_path,
        applied_controls=control_trace,
        walking_targets_rad=walking_trace,
        raw_contact_sensors=contact_trace,
        root_pose_world=root_pose,
        joint_qpos_rad=joint_qpos,
    )
    receipt = {
        "format": "chreatures.author-teacher-physical-feasibility.v1",
        "status": "executed",
        "engine": "mujoco-3.12.0-native",
        "scene": str(args.scene),
        "scene_sha256": physics["source_mjcf_sha256"],
        "teacher_bank_sha256": bank_manifest["bank"]["sha256"],
        "source_revision": bank_manifest["source"]["git_revision"],
        "body_schema_sha256": physics["body_schema_sha256"],
        "residents": 4,
        "modes": list(MODES),
        "frequency_hz": args.frequency,
        "duration_s": args.duration,
        "settle_s": args.settle,
        "physics_dt_s": float(model.opt.timestep),
        "control_dt_s": control_dt,
        "substeps_per_control": substeps,
        "control_ticks": ticks,
        "fixture_servo_kp": 50.0,
        "adhesion_gain_in_memory": args.adhesion_gain,
        "nonwalking_servo_command": "all 42 remaining servos held at per-resident fixture neutral",
        "trace": {
            "file": trace_path.name,
            "sha256": sha256(trace_path),
            "applied_controls_shape": list(control_trace.shape),
            "walking_targets_shape": list(walking_trace.shape),
            "raw_contact_shape": list(contact_trace.shape),
            "joint_qpos_shape": list(joint_qpos.shape),
        },
        "outcomes": outcomes,
        "observed_checks": {
            "forward_displacement_positive": outcomes[0]["forward_displacement_mm"] > 0.0,
            "stop_translation_mm": float(np.linalg.norm(final_position[2] - initial_position[2])),
            "asymmetric_turn_commands_produced_opposite_yaw_signs": outcomes[1]["yaw_change_rad"] * outcomes[3]["yaw_change_rad"] < 0.0,
        },
        "finite": True,
        "interpretation": "Research-only author teacher replay. Displacement and contact show behavior of this engineered model/controller pairing; they do not establish faithful fly locomotion or learned CNS control.",
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
