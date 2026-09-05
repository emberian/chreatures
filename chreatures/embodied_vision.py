"""Sparse, delayed native vision acquired through an actual body's camera.

Each episode spans exactly one five-tick inherited motor action. A pair of
images is processed together; delivery has a fixed model-tick deadline, and
the world waits at that boundary if inference is still running. Transport time
never silently changes which action first has access to a representation.
"""

from __future__ import annotations

import base64
import copy

from .contextual_motor import MOTOR_ACTIONS
from .perception_client import AsyncPerceptionClient


PHYSIOLOGY = ("energy", "gut", "fatigue", "speed", "angular_velocity", "support")
OUTCOMES = ("nutrition", "contact", "distance", "effort")


class EmbodiedVision:
    def __init__(
        self, endpoint, resident_ids, *, interval_ticks=200, delivery_delay_ticks=20
    ):
        from .visual_episodes import VisualEpisodeMemory

        if (
            interval_ticks < 10
            or interval_ticks % 5
            or not 1 <= delivery_delay_ticks < interval_ticks - 5
        ):
            raise ValueError(
                "visual cadence must separate five-tick captures and their delivery"
            )
        self.ids = list(resident_ids)
        self.client = AsyncPerceptionClient(endpoint, min_interval_seconds=0)
        self.memories = {key: VisualEpisodeMemory() for key in self.ids}
        self.latest = {}
        self.frames = {}
        self.interval_ticks = int(interval_ticks)
        self.delivery_delay_ticks = int(delivery_delay_ticks)
        self.next_capture_tick = 0
        self.round_robin = 0
        self.pair = None
        self.waiting = None
        self.completed_pairs = 0
        self.renderer = None
        self.render_model = None

    @staticmethod
    def physiology(world, neural_state):
        return {
            b.id: {
                **{k: float(getattr(b, k)) for k in PHYSIOLOGY if k != "support"},
                "support": float(neural_state[b.id]["support"]),
            }
            for b in world.bodies
        }

    def _frame(self, world, resident, tick):
        from .retinal_render import RetinalRenderer

        if self.render_model is not world.model:
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = RetinalRenderer(world, width=256, height=192)
            self.render_model = world.model
        frame = self.renderer.render(resident)
        # The raster contains no names or researcher overlays. Geometry never
        # enters the native feature/memory interface as privileged coordinates.
        return {
            "sensor_id": resident,
            "world_sequence": tick,
            "model_time": frame.model_time,
            "captured_at": frame.captured_at,
            "provenance": "resident_fov",
            "png_base64": base64.b64encode(frame.png()).decode(),
        }

    def begin_step(self, world, tick, actions, physiology):
        if tick == self.next_capture_tick:
            if self.pair is not None:
                raise RuntimeError("Previous visual episode was not delivered")
            resident = self.ids[self.round_robin % len(self.ids)]
            self.round_robin += 1
            self.pair = {
                "resident": resident,
                "start_tick": tick,
                "start": self._frame(world, resident, tick),
                "steps": [],
                "request_id": None,
            }
            self.next_capture_tick += self.interval_ticks
        if self.pair is not None and self.pair["request_id"] is None:
            resident = self.pair["resident"]
            self.pair["before"] = copy.deepcopy(physiology[resident])
            self.pair["action"] = {
                key: float(actions[resident].get(key, 0)) for key in MOTOR_ACTIONS
            }

    def finish_step(self, world, tick, outcomes, physiology):
        pair = self.pair
        if pair is None or pair["request_id"] is not None:
            return
        resident = pair["resident"]
        pair["steps"].append(
            {
                "from_tick": tick - 1,
                "to_tick": tick,
                "dt": 0.05,
                "action": pair.pop("action"),
                "outcome": {k: float(outcomes[resident].get(k, 0)) for k in OUTCOMES},
                "physiology_before": pair.pop("before"),
                "physiology_after": copy.deepcopy(physiology[resident]),
            }
        )
        if len(pair["steps"]) != 5:
            return
        pair["end"] = self._frame(world, resident, tick)
        rows = []
        for frame in (pair["start"], pair["end"]):
            row = {key: value for key, value in frame.items() if key != "png_base64"}
            row["png"] = base64.b64decode(frame["png_base64"])
            rows.append(row)
        request_id = f"vision-{resident}-{pair['start_tick']}"
        result = self.client.submit_cohort(
            request_id, rows, delivery_tick=tick + self.delivery_delay_ticks
        )
        if result["status"] != "accepted":
            raise RuntimeError(f"Visual pair was not accepted: {result['status']}")
        pair["request_id"] = request_id

    @staticmethod
    def _capture(observation, cohort):
        return {
            "feature": observation["feature"]["values"],
            "feature_sha256": observation["feature"]["sha256"],
            "frame_sha256": observation["frame_sha256"],
            "response_sha256": cohort["response_sha256"],
            "capture_tick": observation["source"]["world_sequence"],
            "delivery_tick": cohort["delivery_tick"],
            "model_time": observation["source"]["model_time"],
            "model_revision": cohort["model"]["revision"],
            "pooling_version": cohort["pooling"]["version"],
        }

    def poll(self, tick):
        result = self.client.take_scheduled(tick, {key: tick for key in self.ids})
        if result["status"] == "missed_delivery":
            raise RuntimeError("Missed the native vision delivery boundary")
        self.waiting = result if result["status"] == "awaiting" else None
        if self.waiting is not None:
            # Retry only failed read-only inference over the identical frozen
            # raster. Neural/physical state does not advance while waiting.
            for request in result["requests"]:
                if request.get("retryable"):
                    self.client.retry_failed(request["request_id"])
            return False
        for cohort in result["cohorts"]:
            if self.pair is None or cohort["request_id"] != self.pair["request_id"]:
                raise RuntimeError(
                    "Native response does not belong to the pending personal episode"
                )
            captures = [self._capture(row, cohort) for row in cohort["observations"]]
            resident = self.pair["resident"]
            self.memories[resident].bind_interval(
                captures[0], captures[1], self.pair["steps"]
            )
            self.latest[resident] = captures[1]
            self.frames[resident] = self.pair["end"]["png_base64"]
            self.completed_pairs += 1
            self.pair = None
            self.client.prune_delivered(keep=1)
        return True

    def candidate_evidence(self, resident, tick, physiology, utility_config):
        capture = self.latest.get(resident)
        if capture is None:
            return None
        memory = self.memories[resident]
        # The callback consumes a delayed observation with an age-dependent
        # bound. It cannot query a fresh image, the world or another resident.
        return memory.contextual_candidate_evidence(
            capture,
            current_tick=tick,
            current_physiology=physiology,
            utility_config=utility_config,
        )

    def view(self, tick):
        return {
            "kind": "native-body-vision-episodes-v1",
            "waiting": self.waiting,
            "completed_pairs": self.completed_pairs,
            "next_capture_tick": self.next_capture_tick,
            "interval_ticks": self.interval_ticks,
            "delivery_delay_ticks": self.delivery_delay_ticks,
            "residents": {
                key: {
                    "episodes": memory.record_count,
                    "has_frame": key in self.frames,
                    "capture_tick": self.latest.get(key, {}).get("capture_tick"),
                    "age_ticks": tick - self.latest[key]["capture_tick"]
                    if key in self.latest
                    else None,
                }
                for key, memory in self.memories.items()
            },
        }

    def snapshot(self):
        return {
            "format": "chreatures-embodied-vision-v1",
            "ids": self.ids,
            "client": self.client.snapshot(),
            "memories": {k: v.snapshot() for k, v in self.memories.items()},
            "latest": copy.deepcopy(self.latest),
            "frames": copy.deepcopy(self.frames),
            "pair": copy.deepcopy(self.pair),
            "interval_ticks": self.interval_ticks,
            "delivery_delay_ticks": self.delivery_delay_ticks,
            "next_capture_tick": self.next_capture_tick,
            "round_robin": self.round_robin,
            "completed_pairs": self.completed_pairs,
        }

    @classmethod
    def restore(cls, value):
        from .visual_episodes import VisualEpisodeMemory

        if value.get("format") != "chreatures-embodied-vision-v1":
            raise ValueError("Unsupported embodied vision checkpoint")
        instance = cls.__new__(cls)
        instance.client = AsyncPerceptionClient.restore(value["client"])
        instance.memories = {
            k: VisualEpisodeMemory.restore(v) for k, v in value["memories"].items()
        }
        instance.frames = copy.deepcopy(value.get("frames", {}))
        for key in (
            "ids",
            "latest",
            "pair",
            "interval_ticks",
            "delivery_delay_ticks",
            "next_capture_tick",
            "round_robin",
            "completed_pairs",
        ):
            setattr(instance, key, copy.deepcopy(value[key]))
        instance.waiting = None
        instance.renderer = None
        instance.render_model = None
        return instance

    def close(self):
        self.client.close()
        if self.renderer is not None:
            self.renderer.close()
