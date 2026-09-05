"""Model-time performances of ordinary physical commands.

Names belong to the visitor's notebook. Only validated physical commands reach
the world, ordered at integer integration boundaries and saved with its state.
"""

from __future__ import annotations

import copy
import math


DT = 0.05


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return float(value)


def ticks(seconds):
    return math.ceil(seconds / DT - 1e-9)


def validate_command(world, command):
    if not isinstance(command, dict):
        raise ValueError("event command must be an object")
    allowed = {
        "signal": {"op", "x", "y", "z", "tone", "strength"},
        "light": {"op", "x", "y", "z", "intensity", "duration", "color"},
        "hand": {"op", "id", "x", "y", "z", "stiffness", "damping"},
        "release": {"op"},
    }
    op = command.get("op")
    if op not in allowed or set(command) - allowed[op]:
        raise ValueError("performances support only signal, light, hand and release")
    result = copy.deepcopy(command)
    if op != "release":
        for key, bound in zip(
            ("x", "y", "z"), (world.width, world.height, world.depth)
        ):
            result[key] = number(command.get(key), key, 0, bound)
    if op == "signal":
        tone = command.get("tone", 0)
        if type(tone) is not int or tone not in (0, 1, 2):
            raise ValueError("tone must be 0, 1 or 2")
        result.update(
            tone=tone, strength=number(command.get("strength", 1), "strength", 0.001, 1)
        )
    elif op == "light":
        result.update(
            intensity=number(command.get("intensity", 1), "intensity", 0, 1),
            duration=number(command.get("duration", 2), "duration", 0.01, 30),
        )
        color = command.get("color", [1, 0.94, 0.78])
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError("color must have three channels")
        result["color"] = [number(x, "color", 0, 1) for x in color]
    elif op == "hand":
        target = command.get("id")
        if (
            not isinstance(target, str)
            or target not in world._entity_mj
            or world._entity(target)["mobility"] != "free"
        ):
            raise ValueError("hand target must be a free physical entity")
        result.update(
            stiffness=number(command.get("stiffness", 18), "stiffness", 0.1, 80),
            damping=number(command.get("damping", 2.5), "damping", 0, 20),
        )
    return result


class VisitorPerformances:
    def __init__(self):
        self.motifs = {}
        self.queue = []
        self.revision = 0
        self.next_id = 1
        self.hand_owner = None

    def _id(self, prefix):
        identifier = f"{prefix}-{self.next_id}"
        self.next_id += 1
        return identifier

    def _motif(self, world, value):
        if not isinstance(value, dict) or set(value) - {"name", "duration", "events"}:
            raise ValueError("motif requires name, duration and events")
        name = value.get("name", "A visitor's performance")
        if not isinstance(name, str) or not 1 <= len(name) <= 100:
            raise ValueError("motif name must have 1..100 characters")
        duration = number(value.get("duration"), "duration", DT, 120)
        rows = value.get("events")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
            raise ValueError("motif must have 1..64 events")
        events = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"at", "command"}:
                raise ValueError("event requires at and command")
            at = ticks(number(row["at"], "event offset", 0, duration))
            events.append(
                {"offset_tick": at, "command": validate_command(world, row["command"])}
            )
        events.sort(key=lambda event: event["offset_tick"])
        return {"name": name, "duration_ticks": ticks(duration), "events": events}

    def add_motif(self, world, value):
        if len(self.motifs) >= 32:
            raise ValueError("motif capacity reached")
        motif = self._motif(world, value)
        motif["id"] = self._id("motif")
        self.motifs[motif["id"]] = motif
        self.revision += 1
        return self._motif_view(motif)

    def schedule(self, world, tick, value):
        if not isinstance(value, dict):
            raise ValueError("schedule must be an object")
        if sum(row["status"] in ("queued", "playing") for row in self.queue) >= 16:
            raise ValueError("pending performance capacity reached")
        delay = ticks(number(value.get("start_in", 0), "start_in", 0, 120))
        if "motif_id" in value:
            if set(value) - {"motif_id", "start_in"}:
                raise ValueError("unknown schedule field")
            motif = copy.deepcopy(self.motifs[value["motif_id"]])
        else:
            motif = self._motif(
                world, {k: v for k, v in value.items() if k != "start_in"}
            )
            motif["id"] = None
        # Revalidate current physical targets before reserving a schedule.
        for event in motif["events"]:
            validate_command(world, event["command"])
        item = {
            **motif,
            "motif_id": motif["id"],
            "id": self._id("performance"),
            "start_tick": tick + delay,
            "status": "queued",
            "cursor": 0,
            "failure": None,
        }
        # Completed rows are bounded notebook entries; active entries survive.
        if len(self.queue) >= 64:
            self.queue = [x for x in self.queue if x["status"] in ("queued", "playing")]
        self.queue.append(item)
        self.revision += 1
        return self._schedule_view(item)

    def direct_command(self, command):
        if command.get("op") in ("hand", "move", "release"):
            self.hand_owner = None

    def cancel(self, world, identifier):
        item = next(x for x in self.queue if x["id"] == identifier)
        if item["status"] in ("queued", "playing"):
            item["status"] = "cancelled"
            if self.hand_owner == identifier:
                world.command({"op": "release"})
                self.hand_owner = None
            self.revision += 1
        return self._schedule_view(item)

    def advance(self, world, tick):
        executed = []
        due = []
        for item in self.queue:
            if item["status"] not in ("queued", "playing") or tick < item["start_tick"]:
                continue
            item["status"] = "playing"
            for index in range(item["cursor"], len(item["events"])):
                event = item["events"][index]
                deadline = item["start_tick"] + event["offset_tick"]
                if deadline > tick:
                    break
                due.append(
                    (deadline, int(item["id"].split("-")[-1]), index, item, event)
                )
        for deadline, _, index, item, event in sorted(due, key=lambda x: x[:3]):
            if item["status"] == "failed":
                continue
            command = event["command"]
            try:
                if command["op"] != "release" or self.hand_owner == item["id"]:
                    world.command(command)
                if command["op"] == "hand":
                    self.hand_owner = item["id"]
                elif command["op"] == "release" and self.hand_owner == item["id"]:
                    self.hand_owner = None
                item["cursor"] = index + 1
                executed.append(
                    {
                        "performance": item["id"],
                        "tick": tick,
                        "scheduled_tick": deadline,
                        "command": copy.deepcopy(command),
                    }
                )
            except (ValueError, KeyError) as error:
                item.update(status="failed", failure=str(error))
                if self.hand_owner == item["id"]:
                    world.command({"op": "release"})
                    self.hand_owner = None
            self.revision += 1
        for item in self.queue:
            if (
                item["status"] == "playing"
                and tick >= item["start_tick"] + item["duration_ticks"]
            ):
                item["status"] = "completed"
                if self.hand_owner == item["id"]:
                    world.command({"op": "release"})
                    self.hand_owner = None
                self.revision += 1
        return executed

    @staticmethod
    def _motif_view(motif):
        return {
            "id": motif["id"],
            "name": motif["name"],
            "duration": motif["duration_ticks"] * DT,
            "event_count": len(motif["events"]),
        }

    def _schedule_view(self, item):
        return {
            **self._motif_view(item),
            "motif_id": item["motif_id"],
            "start_time": item["start_tick"] * DT,
            "status": item["status"],
            "failure": item["failure"],
            "delivered": item["cursor"],
            "events": [
                {"at": x["offset_tick"] * DT, "kind": x["command"]["op"]}
                for x in item["events"]
            ],
        }

    def view(self, tick, paused):
        return {
            "model_time": tick * DT,
            "paused": paused,
            "revision": self.revision,
            "motifs": [self._motif_view(x) for x in self.motifs.values()],
            "queue": [self._schedule_view(x) for x in self.queue],
        }

    def snapshot(self):
        return copy.deepcopy(
            {
                "format": "chreatures-visitor-v1",
                "motifs": self.motifs,
                "queue": self.queue,
                "revision": self.revision,
                "next_id": self.next_id,
                "hand_owner": self.hand_owner,
            }
        )

    @classmethod
    def restore(cls, value, world):
        instance = cls()
        if value is None:
            return instance
        if value.get("format") != "chreatures-visitor-v1":
            raise ValueError("unsupported visitor checkpoint")
        for key in ("motifs", "queue", "revision", "next_id", "hand_owner"):
            setattr(instance, key, copy.deepcopy(value[key]))
        if len(instance.motifs) > 32 or len(instance.queue) > 64:
            raise ValueError("visitor checkpoint exceeds capacity")
        for item in [*instance.motifs.values(), *instance.queue]:
            for event in item["events"]:
                validate_command(world, event["command"])
        return instance
