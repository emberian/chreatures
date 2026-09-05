"""The authoritative physical habitat for Chreatures.

The world deliberately contains no goal or action-selection logic.  It turns
motor commands into physics and exposes only signals available at a creature's
body: antennae, a small retina, touch, sound, shade, and proprioception.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Any

import numpy as np


MODEL_DT = 0.05
DT = MODEL_DT
MAX_OBJECTS = 128
MAX_SIGNALS = 256
_KINDS = {"food", "stone", "shelter", "toy", "flower", "beacon"}
_SOLID_KINDS = {"stone", "toy", "flower", "beacon"}


@dataclass
class Body:
    id: str
    name: str
    x: float
    y: float
    heading: float
    radius: float = 9.0
    energy: float = 0.78
    gut: float = 0.12
    fatigue: float = 0.05
    speed: float = 0.0
    angular_velocity: float = 0.0
    age: float = 0.0
    color: str = "#ffffff"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Object:
    id: str
    kind: str
    x: float
    y: float
    radius: float
    color: str
    odor: int | None = None
    amount: float = 1.0
    movable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Signal:
    id: str
    x: float
    y: float
    tone: int
    strength: float = 1.0
    remaining: float = 1.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _rgb(color: str) -> tuple[float, float, float]:
    """Parse our own render colors, falling back to visible neutral gray."""
    if isinstance(color, str) and len(color) == 7 and color.startswith("#"):
        try:
            return tuple(int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))  # type: ignore[return-value]
        except ValueError:
            pass
    return (0.55, 0.55, 0.55)


class World:
    """Seeded continuous-time terrarium state and local sensor model."""

    width = 1200.0
    height = 800.0

    def __init__(self, seed: int = 7):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.seed = int(seed)
        self.time = 0.0
        self.rng = np.random.default_rng(self.seed)
        self.bodies: list[Body] = [
            Body("mica", "Mica", 245.0, 335.0, 0.12, color="#ef9a72"),
            Body("fern", "Fern", 605.0, 205.0, 1.62, energy=0.74, gut=0.18, color="#79c68b"),
            Body("pip", "Pip", 905.0, 535.0, 3.18, energy=0.82, gut=0.08, color="#87aee8"),
        ]
        # The fixed layout leaves long sight lines as well as occluded pockets.
        # Small seed-derived offsets give different nurseries distinct histories.
        j = self.rng.uniform(-12.0, 12.0, size=(6, 2))
        self.objects: list[Object] = [
            Object("food-1", "food", 365 + j[0, 0], 300 + j[0, 1], 8, "#e64f67", 0, 1.0),
            Object("food-2", "food", 775 + j[1, 0], 180 + j[1, 1], 10, "#f2b64d", 1, 1.0),
            Object("food-3", "food", 690 + j[2, 0], 625 + j[2, 1], 9, "#d95f8d", 0, 0.8),
            Object("food-4", "food", 1020 + j[3, 0], 360 + j[3, 1], 9, "#efd45d", 1, 0.9),
            Object("stone-1", "stone", 480, 385, 42, "#718096"),
            Object("stone-2", "stone", 835, 390, 55, "#526579"),
            Object("stone-3", "stone", 170, 610, 29, "#91a0aa"),
            Object("shelter-1", "shelter", 1040, 665, 92, "#405365", amount=1.0),
            Object("toy-1", "toy", 535 + j[4, 0], 585 + j[4, 1], 14, "#a974d6", 2, 1.0, True),
            Object("toy-2", "toy", 965 + j[5, 0], 245 + j[5, 1], 12, "#52bfc0", 1, 1.0, True),
            Object("flower-1", "flower", 315, 675, 16, "#c76fe3", 2, 1.0),
            Object("flower-2", "flower", 715, 365, 18, "#63c98d", 2, 1.0),
            Object("beacon-1", "beacon", 110, 115, 13, "#65c9ff", amount=0.55),
        ]
        self.signals: list[Signal] = []
        self._touch: dict[str, list[float]] = {body.id: [0.0, 0.0] for body in self.bodies}
        self._signal_cooldown: dict[str, float] = {body.id: 0.0 for body in self.bodies}
        self._next_object_id = 1
        self._next_signal_id = 1

    def _body(self, body_id: str) -> Body:
        for body in self.bodies:
            if body.id == body_id:
                return body
        raise KeyError(f"unknown body id: {body_id}")

    def sense(self, body_id: str) -> dict[str, Any]:
        body = self._body(body_id)
        return {
            "odor": self._odor(body),
            "vision": self._vision(body),
            "touch": list(self._touch.get(body.id, (0.0, 0.0))),
            "sound": self._sound(body),
            "shade": self._shade(body),
            "speed": float(body.speed),
            "angular_velocity": float(body.angular_velocity),
            "energy": float(body.energy),
            "gut": float(body.gut),
            "fatigue": float(body.fatigue),
        }

    def _odor(self, body: Body) -> list[list[float]]:
        # A 2-D Gaussian plume is a useful local approximation for still air.
        # Antennae sit forward and to either side of the body.  sigma controls
        # diffusion length; sources beyond four sigma are negligible and skipped.
        sigma = 92.0
        h = (math.cos(body.heading), math.sin(body.heading))
        right = (-h[1], h[0])
        antennae = [
            (body.x + h[0] * 11.5 - right[0] * 7.0, body.y + h[1] * 11.5 - right[1] * 7.0),
            (body.x + h[0] * 11.5 + right[0] * 7.0, body.y + h[1] * 11.5 + right[1] * 7.0),
        ]
        result = np.zeros((2, 3), dtype=float)
        cutoff2 = (4.0 * sigma) ** 2
        for obj in self.objects:
            if obj.odor is None or obj.amount <= 0.0:
                continue
            strength = obj.amount
            if obj.kind == "food":
                strength *= 1.2
            elif obj.kind == "flower":
                strength *= 0.7
            elif obj.kind == "toy":
                strength *= 0.6
            for side, (ax, ay) in enumerate(antennae):
                d2 = (obj.x - ax) ** 2 + (obj.y - ay) ** 2
                if d2 <= cutoff2:
                    result[side, obj.odor] += strength * math.exp(-d2 / (2.0 * sigma * sigma))
        return np.clip(result, 0.0, 4.0).tolist()

    @staticmethod
    def _ray_circle(
        ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, radius: float
    ) -> float | None:
        rx, ry = ox - cx, oy - cy
        b = rx * dx + ry * dy
        c = rx * rx + ry * ry - radius * radius
        disc = b * b - c
        if disc < 0.0:
            return None
        root = math.sqrt(disc)
        near, far = -b - root, -b + root
        if near >= 0.0:
            return near
        if far >= 0.0:
            return far
        return None

    def _wall_distance(self, x: float, y: float, dx: float, dy: float) -> float:
        hits: list[float] = []
        if dx > 1e-12:
            hits.append((self.width - x) / dx)
        elif dx < -1e-12:
            hits.append(-x / dx)
        if dy > 1e-12:
            hits.append((self.height - y) / dy)
        elif dy < -1e-12:
            hits.append(-y / dy)
        return min(t for t in hits if t >= 0.0)

    def _vision(self, body: Body) -> list[list[float]]:
        max_range = 260.0
        fan = math.radians(150.0)
        rows: list[list[float]] = []
        for angle in np.linspace(-fan / 2.0, fan / 2.0, 16):
            theta = body.heading + float(angle)
            dx, dy = math.cos(theta), math.sin(theta)
            nearest = self._wall_distance(body.x, body.y, dx, dy)
            color = (0.20, 0.24, 0.29)
            for obj in self.objects:
                if obj.amount <= 0.0 and obj.kind == "food":
                    continue
                hit = self._ray_circle(body.x, body.y, dx, dy, obj.x, obj.y, obj.radius)
                if hit is not None and hit < nearest:
                    nearest, color = hit, _rgb(obj.color)
            for other in self.bodies:
                if other.id == body.id:
                    continue
                hit = self._ray_circle(body.x, body.y, dx, dy, other.x, other.y, other.radius)
                if hit is not None and hit < nearest:
                    nearest, color = hit, _rgb(other.color)
            proximity = max(0.0, 1.0 - nearest / max_range) if nearest <= max_range else 0.0
            if nearest > max_range:
                color = (0.0, 0.0, 0.0)
            rows.append([float(color[0]), float(color[1]), float(color[2]), float(proximity)])
        return rows

    def _sound(self, body: Body) -> list[float]:
        sound = [0.0, 0.0, 0.0]
        for obj in self.objects:
            if obj.kind != "beacon" or obj.amount <= 0.0:
                continue
            distance = math.hypot(obj.x - body.x, obj.y - body.y)
            tone = 0 if obj.odor is None else obj.odor
            sound[tone] += obj.amount / (1.0 + (distance / 150.0) ** 2)
        for signal in self.signals:
            distance = math.hypot(signal.x - body.x, signal.y - body.y)
            envelope = max(0.0, min(1.0, signal.remaining / 0.35))
            sound[signal.tone] += signal.strength * envelope / (1.0 + (distance / 125.0) ** 2)
        return [float(min(2.0, value)) for value in sound]

    def _shade(self, body: Body) -> float:
        shade = 0.0
        for obj in self.objects:
            if obj.kind == "shelter":
                d = math.hypot(obj.x - body.x, obj.y - body.y)
                shade = max(shade, min(1.0, max(0.0, (obj.radius + body.radius - d) / (body.radius * 2.0))))
        return float(shade)

    def _validate_actions(self, actions: dict[str, dict[str, Any]], dt: Any) -> tuple[dict[str, dict[str, float]], float]:
        if not isinstance(actions, dict):
            raise ValueError("actions must be a mapping")
        step = _finite_number(dt, "dt")
        if step <= 0.0 or step > 1.0:
            raise ValueError("dt must be in (0, 1]")
        body_ids = {body.id for body in self.bodies}
        unknown = set(actions) - body_ids
        if unknown:
            raise ValueError(f"unknown body id: {sorted(unknown)[0]}")
        limits = {"forward": (0.0, 1.0), "turn": (-1.0, 1.0), "eat": (0.0, 1.0), "signal": (0.0, 1.0)}
        clean: dict[str, dict[str, float]] = {}
        for body_id, raw in actions.items():
            if not isinstance(raw, dict):
                raise ValueError(f"action for {body_id} must be a mapping")
            extra = set(raw) - set(limits)
            if extra:
                raise ValueError(f"unknown action field: {sorted(extra)[0]}")
            clean[body_id] = {}
            for field, value in raw.items():
                number = _finite_number(value, field)
                low, high = limits[field]
                if not low <= number <= high:
                    raise ValueError(f"{field} must be in [{low:g}, {high:g}]")
                clean[body_id][field] = number
        return clean, step

    def advance(self, actions: dict[str, dict[str, Any]], dt: float = MODEL_DT) -> dict[str, dict[str, float]]:
        # Validation is complete before RNG, clocks, bodies, or events mutate.
        clean, step = self._validate_actions(actions, dt)
        self._touch = {body.id: [0.0, 0.0] for body in self.bodies}
        self._signal_cooldown = {
            body.id: max(0.0, self._signal_cooldown.get(body.id, 0.0) - step)
            for body in self.bodies
        }
        outcomes = {body.id: {"nutrition": 0.0, "contact": 0.0, "distance": 0.0} for body in self.bodies}

        for signal in self.signals:
            signal.remaining -= step
        self.signals = [signal for signal in self.signals if signal.remaining > 0.0]

        old_positions = {body.id: (body.x, body.y) for body in self.bodies}
        for body_index, body in enumerate(self.bodies):
            action = clean.get(body.id, {})
            forward = action.get("forward", 0.0)
            turn = action.get("turn", 0.0)
            # Tiny seeded motor variation makes individuals physical rather than
            # clockwork, while zero commands cannot cause a body to wander.
            speed_noise, turn_noise = self.rng.normal(0.0, 1.0, size=2)
            vitality = 0.18 + 0.82 * body.energy
            fatigue_factor = 1.0 - 0.72 * body.fatigue
            desired_speed = 76.0 * forward * vitality * fatigue_factor * (1.0 + 0.018 * speed_noise * forward)
            desired_turn = 2.8 * turn * fatigue_factor + 0.025 * turn_noise * abs(turn)
            speed_alpha = 1.0 - math.exp(-step / 0.24)
            turn_alpha = 1.0 - math.exp(-step / 0.18)
            body.speed += (desired_speed - body.speed) * speed_alpha
            body.angular_velocity += (desired_turn - body.angular_velocity) * turn_alpha
            body.heading = (body.heading + body.angular_velocity * step) % (2.0 * math.pi)
            body.x += math.cos(body.heading) * body.speed * step
            body.y += math.sin(body.heading) * body.speed * step
            self._resolve_boundaries(body)
            self._resolve_objects(body, old_positions[body.id])

            if (
                action.get("signal", 0.0) > 0.0
                and self._signal_cooldown[body.id] <= 1e-12
                and len(self.signals) < MAX_SIGNALS
            ):
                strength = action["signal"]
                self.signals.append(
                    Signal(
                        f"signal-{self._next_signal_id}", body.x, body.y, body_index % 3, strength
                    )
                )
                self._next_signal_id += 1
                self._signal_cooldown[body.id] = 0.5

        self._resolve_body_collisions()

        for body in self.bodies:
            action = clean.get(body.id, {})
            distance = math.hypot(body.x - old_positions[body.id][0], body.y - old_positions[body.id][1])
            outcomes[body.id]["distance"] = float(distance)
            outcomes[body.id]["contact"] = float(max(self._touch[body.id]))
            eat = action.get("eat", 0.0)
            if eat > 0.0:
                for obj in self.objects:
                    if obj.kind != "food" or obj.amount <= 0.0:
                        continue
                    if math.hypot(obj.x - body.x, obj.y - body.y) <= body.radius + obj.radius + 1.0:
                        bite = min(obj.amount, 0.48 * eat * step, 1.0 - body.gut)
                        if bite > 0.0:
                            obj.amount -= bite
                            body.gut += bite
                            outcomes[body.id]["nutrition"] += bite
                            self._mark_touch(body, obj.x - body.x, obj.y - body.y, 1.0)
            digestion = min(body.gut, 0.038 * step, max(0.0, (1.0 - body.energy) / 0.82))
            body.gut -= digestion
            body.energy += digestion * 0.82
            effort = min(1.0, abs(body.speed) / 76.0) + 0.35 * min(1.0, abs(body.angular_velocity) / 2.8)
            body.energy = float(np.clip(body.energy - step * (0.0007 + 0.0038 * effort), 0.0, 1.0))
            fatigue_delta = step * (0.065 * effort - 0.027 * (1.0 - min(1.0, effort)))
            body.fatigue = float(np.clip(body.fatigue + fatigue_delta, 0.0, 1.0))
            body.gut = float(np.clip(body.gut, 0.0, 1.0))
            body.age += step
            outcomes[body.id]["contact"] = float(max(self._touch[body.id]))
        self.time += step
        return outcomes

    def _mark_touch(self, body: Body, dx: float, dy: float, strength: float) -> None:
        right_x, right_y = -math.sin(body.heading), math.cos(body.heading)
        lateral = dx * right_x + dy * right_y
        index = 1 if lateral >= 0.0 else 0
        self._touch[body.id][index] = max(self._touch[body.id][index], float(np.clip(strength, 0.0, 1.0)))

    def _resolve_boundaries(self, body: Body) -> None:
        old_x, old_y = body.x, body.y
        nx = min(max(body.x, body.radius), self.width - body.radius)
        ny = min(max(body.y, body.radius), self.height - body.radius)
        if nx != body.x or ny != body.y:
            # Tactile direction points from the body toward the contacted wall.
            self._mark_touch(body, old_x - nx, old_y - ny, 1.0)
            body.x, body.y = nx, ny
            body.speed *= 0.25

    def _toy_position_valid(self, toy: Object, x: float, y: float) -> bool:
        if not (toy.radius <= x <= self.width - toy.radius and toy.radius <= y <= self.height - toy.radius):
            return False
        for other in self.objects:
            if other.id == toy.id or other.kind not in _SOLID_KINDS:
                continue
            if math.hypot(other.x - x, other.y - y) < other.radius + toy.radius:
                return False
        return True

    def _resolve_objects(self, body: Body, old_position: tuple[float, float]) -> None:
        for obj in self.objects:
            if obj.kind not in _SOLID_KINDS:
                continue
            dx, dy = body.x - obj.x, body.y - obj.y
            distance = math.hypot(dx, dy)
            minimum = body.radius + obj.radius
            if distance >= minimum:
                continue
            if distance < 1e-9:
                dx, dy, distance = math.cos(body.heading), math.sin(body.heading), 1.0
            nx, ny = dx / distance, dy / distance  # from object toward body
            penetration = minimum - distance
            self._mark_touch(body, -nx, -ny, min(1.0, 0.35 + penetration / body.radius))
            if obj.kind == "toy" and obj.movable:
                travel_x, travel_y = body.x - old_position[0], body.y - old_position[1]
                forward_push = max(0.0, travel_x * -nx + travel_y * -ny)
                shift = penetration + 0.72 * forward_push
                tx, ty = obj.x - nx * shift, obj.y - ny * shift
                if self._toy_position_valid(obj, tx, ty):
                    obj.x, obj.y = tx, ty
                    body.x = obj.x + nx * minimum
                    body.y = obj.y + ny * minimum
                    body.speed *= 0.72
                    continue
            body.x = obj.x + nx * minimum
            body.y = obj.y + ny * minimum
            body.speed *= 0.22

    def _resolve_body_collisions(self) -> None:
        for i, first in enumerate(self.bodies):
            for second in self.bodies[i + 1 :]:
                dx, dy = second.x - first.x, second.y - first.y
                distance = math.hypot(dx, dy)
                minimum = first.radius + second.radius
                if distance >= minimum:
                    continue
                if distance < 1e-9:
                    dx, dy, distance = 1.0, 0.0, 1.0
                nx, ny = dx / distance, dy / distance
                correction = (minimum - distance) / 2.0
                first.x -= nx * correction
                first.y -= ny * correction
                second.x += nx * correction
                second.y += ny * correction
                self._resolve_boundaries(first)
                self._resolve_boundaries(second)
                self._mark_touch(first, dx, dy, 1.0)
                self._mark_touch(second, -dx, -dy, 1.0)
                first.speed *= 0.55
                second.speed *= 0.55

    def command(self, command: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(command, dict):
            raise ValueError("command must be a mapping")
        op = command.get("op", command.get("command"))
        if not isinstance(op, str):
            raise ValueError("command requires string 'op'")

        if op == "add":
            allowed = {"op", "command", "kind", "x", "y", "odor", "amount", "radius", "color", "movable"}
            self._reject_extra(command, allowed)
            kind = command.get("kind")
            if kind not in _KINDS:
                raise ValueError(f"kind must be one of {sorted(_KINDS)}")
            if len(self.objects) >= MAX_OBJECTS:
                raise ValueError(f"world cannot contain more than {MAX_OBJECTS} objects")
            x, y = self._validated_position(command)
            odor = command.get("odor")
            if odor is not None and (isinstance(odor, bool) or not isinstance(odor, (int, np.integer)) or int(odor) not in (0, 1, 2)):
                raise ValueError("odor must be 0, 1, or 2")
            defaults = {
                "food": (9.0, "#e96970", True), "stone": (28.0, "#718096", False),
                "shelter": (72.0, "#405365", False), "toy": (13.0, "#a974d6", True),
                "flower": (15.0, "#d67be6", False), "beacon": (12.0, "#65c9ff", False),
            }
            default_radius, default_color, default_movable = defaults[kind]
            radius = _finite_number(command.get("radius", default_radius), "radius")
            amount = _finite_number(command.get("amount", 1.0), "amount")
            if not 1.0 <= radius <= 150.0:
                raise ValueError("radius must be in [1, 150]")
            if not 0.0 <= amount <= 100.0:
                raise ValueError("amount must be in [0, 100]")
            color = command.get("color", default_color)
            if not isinstance(color, str) or len(color) > 32:
                raise ValueError("color must be a short string")
            movable = command.get("movable", default_movable)
            if not isinstance(movable, bool):
                raise ValueError("movable must be boolean")
            object_id = f"{kind}-user-{self._next_object_id}"
            new_object = Object(object_id, kind, x, y, radius, color, None if odor is None else int(odor), amount, movable)
            self.objects.append(new_object)
            self._next_object_id += 1
            return new_object.to_dict()

        if op == "move":
            self._reject_extra(command, {"op", "command", "id", "x", "y"})
            object_id = command.get("id")
            if not isinstance(object_id, str):
                raise ValueError("move requires string id")
            target = next((obj for obj in self.objects if obj.id == object_id), None)
            if target is None:
                raise ValueError(f"unknown object id: {object_id}")
            x, y = self._validated_position(command)
            target.x, target.y = x, y
            return target.to_dict()

        if op == "remove":
            self._reject_extra(command, {"op", "command", "id"})
            object_id = command.get("id")
            if not isinstance(object_id, str):
                raise ValueError("remove requires string id")
            index = next((i for i, obj in enumerate(self.objects) if obj.id == object_id), None)
            if index is None:
                raise ValueError(f"unknown object id: {object_id}")
            removed = self.objects.pop(index)
            return removed.to_dict()

        if op == "signal":
            self._reject_extra(command, {"op", "command", "x", "y", "tone", "strength"})
            x, y = self._validated_position(command)
            tone = command.get("tone", 0)
            if isinstance(tone, bool) or not isinstance(tone, (int, np.integer)) or int(tone) not in (0, 1, 2):
                raise ValueError("tone must be 0, 1, or 2")
            strength = _finite_number(command.get("strength", 1.0), "strength")
            if not 0.0 < strength <= 1.0:
                raise ValueError("strength must be in (0, 1]")
            if len(self.signals) >= MAX_SIGNALS:
                raise ValueError(f"world cannot contain more than {MAX_SIGNALS} signals")
            signal = Signal(f"signal-{self._next_signal_id}", x, y, int(tone), strength)
            self.signals.append(signal)
            self._next_signal_id += 1
            return signal.to_dict()

        raise ValueError(f"unknown command: {op}")

    @staticmethod
    def _reject_extra(command: dict[str, Any], allowed: set[str]) -> None:
        extra = set(command) - allowed
        if extra:
            raise ValueError(f"unknown command field: {sorted(extra)[0]}")

    def _validated_position(self, command: dict[str, Any]) -> tuple[float, float]:
        if "x" not in command or "y" not in command:
            raise ValueError("command requires x and y")
        x, y = _finite_number(command["x"], "x"), _finite_number(command["y"], "y")
        if not 0.0 <= x <= self.width or not 0.0 <= y <= self.height:
            raise ValueError("position is outside habitat")
        return x, y

    def view(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "time": self.time,
            "bodies": [body.to_dict() for body in self.bodies],
            "objects": [obj.to_dict() for obj in self.objects],
            "signals": [signal.to_dict() for signal in self.signals],
        }

    def snapshot(self) -> dict[str, Any]:
        # PCG64's state is JSON-compatible today; conversion keeps this true for
        # bit generators that use NumPy scalar or array fields in the future.
        return {
            "version": 1,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "time": self.time,
            "bodies": [body.to_dict() for body in self.bodies],
            "objects": [obj.to_dict() for obj in self.objects],
            "signals": [signal.to_dict() for signal in self.signals],
            "touch": copy.deepcopy(self._touch),
            "signal_cooldown": copy.deepcopy(self._signal_cooldown),
            "next_object_id": self._next_object_id,
            "next_signal_id": self._next_signal_id,
            "rng_state": self._json_value(copy.deepcopy(self.rng.bit_generator.state)),
        }

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): World._json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [World._json_value(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> "World":
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("unsupported world snapshot")
        seed = snapshot.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("snapshot seed must be an integer")
        world = cls(seed=seed)
        width = _finite_number(snapshot["width"], "width")
        height = _finite_number(snapshot["height"], "height")
        time = _finite_number(snapshot["time"], "time")
        if width <= 0.0 or height <= 0.0 or width > 100_000.0 or height > 100_000.0 or time < 0.0:
            raise ValueError("invalid world dimensions or time")
        bodies_data = snapshot.get("bodies")
        objects_data = snapshot.get("objects")
        signals_data = snapshot.get("signals", [])
        if not isinstance(bodies_data, list) or not 1 <= len(bodies_data) <= 64:
            raise ValueError("snapshot must contain 1 to 64 bodies")
        if not isinstance(objects_data, list) or len(objects_data) > MAX_OBJECTS:
            raise ValueError(f"snapshot may contain at most {MAX_OBJECTS} objects")
        if not isinstance(signals_data, list) or len(signals_data) > MAX_SIGNALS:
            raise ValueError(f"snapshot may contain at most {MAX_SIGNALS} signals")
        world.width, world.height, world.time = width, height, time
        try:
            world.bodies = [Body(**item) for item in bodies_data]
            world.objects = [Object(**item) for item in objects_data]
            world.signals = [Signal(**item) for item in signals_data]
        except (TypeError, KeyError) as exc:
            raise ValueError("malformed entity in world snapshot") from exc
        world._validate_restored_entities()
        touch_data = snapshot.get("touch", {})
        if not isinstance(touch_data, dict) or set(touch_data) - {body.id for body in world.bodies}:
            raise ValueError("snapshot touch state refers to an unknown body")
        try:
            world._touch = {
                str(k): [
                    _finite_number(v[0], "touch"),
                    _finite_number(v[1], "touch"),
                ]
                for k, v in touch_data.items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            }
        except (IndexError, TypeError) as exc:
            raise ValueError("malformed touch state") from exc
        if len(world._touch) != len(touch_data) or any(not 0.0 <= x <= 1.0 for v in world._touch.values() for x in v):
            raise ValueError("malformed touch state")
        for body in world.bodies:
            world._touch.setdefault(body.id, [0.0, 0.0])
        cooldown_data = snapshot.get("signal_cooldown", {})
        if not isinstance(cooldown_data, dict) or set(cooldown_data) - {body.id for body in world.bodies}:
            raise ValueError("snapshot signal cooldown refers to an unknown body")
        world._signal_cooldown = {}
        for body in world.bodies:
            cooldown = _finite_number(cooldown_data.get(body.id, 0.0), "signal cooldown")
            if not 0.0 <= cooldown <= 0.5 + 1e-9:
                raise ValueError("signal cooldown must be in [0, 0.5]")
            world._signal_cooldown[body.id] = cooldown
        next_object_id = snapshot.get("next_object_id", 1)
        next_signal_id = snapshot.get("next_signal_id", 1)
        if (
            isinstance(next_object_id, bool)
            or not isinstance(next_object_id, int)
            or next_object_id < 1
            or isinstance(next_signal_id, bool)
            or not isinstance(next_signal_id, int)
            or next_signal_id < 1
        ):
            raise ValueError("snapshot counters must be positive integers")
        world._next_object_id = next_object_id
        world._next_signal_id = next_signal_id
        try:
            world.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid RNG state in world snapshot") from exc
        return world

    def _validate_restored_entities(self) -> None:
        body_ids: set[str] = set()
        for body in self.bodies:
            if not isinstance(body.id, str) or not body.id or body.id in body_ids:
                raise ValueError("body ids must be unique nonempty strings")
            body_ids.add(body.id)
            if not isinstance(body.name, str) or not isinstance(body.color, str):
                raise ValueError("body name and color must be strings")
            numeric = {
                "body x": body.x, "body y": body.y, "body heading": body.heading,
                "body radius": body.radius, "body energy": body.energy, "body gut": body.gut,
                "body fatigue": body.fatigue, "body speed": body.speed,
                "body angular velocity": body.angular_velocity, "body age": body.age,
            }
            clean = {name: _finite_number(value, name) for name, value in numeric.items()}
            if not 1.0 <= clean["body radius"] <= 150.0:
                raise ValueError("body radius is out of bounds")
            if not clean["body radius"] <= clean["body x"] <= self.width - clean["body radius"]:
                raise ValueError("body x is outside habitat")
            if not clean["body radius"] <= clean["body y"] <= self.height - clean["body radius"]:
                raise ValueError("body y is outside habitat")
            if any(not 0.0 <= clean[name] <= 1.0 for name in ("body energy", "body gut", "body fatigue")):
                raise ValueError("body physiology is out of bounds")
            if clean["body age"] < 0.0 or abs(clean["body speed"]) > 500.0 or abs(clean["body angular velocity"]) > 100.0:
                raise ValueError("body motion or age is out of bounds")

        object_ids: set[str] = set()
        for obj in self.objects:
            if not isinstance(obj.id, str) or not obj.id or obj.id in object_ids or obj.id in body_ids:
                raise ValueError("object ids must be unique and distinct from body ids")
            object_ids.add(obj.id)
            if obj.kind not in _KINDS or not isinstance(obj.color, str) or not isinstance(obj.movable, bool):
                raise ValueError("object kind, color, or movable flag is invalid")
            x, y = _finite_number(obj.x, "object x"), _finite_number(obj.y, "object y")
            radius, amount = _finite_number(obj.radius, "object radius"), _finite_number(obj.amount, "object amount")
            if not 0.0 <= x <= self.width or not 0.0 <= y <= self.height:
                raise ValueError("object is outside habitat")
            if not 1.0 <= radius <= 150.0 or not 0.0 <= amount <= 100.0:
                raise ValueError("object radius or amount is out of bounds")
            if obj.odor is not None and (isinstance(obj.odor, bool) or not isinstance(obj.odor, int) or obj.odor not in (0, 1, 2)):
                raise ValueError("object odor is invalid")

        signal_ids: set[str] = set()
        for signal in self.signals:
            if not isinstance(signal.id, str) or not signal.id or signal.id in signal_ids or signal.id in object_ids or signal.id in body_ids:
                raise ValueError("signal ids must be globally unique nonempty strings")
            signal_ids.add(signal.id)
            x, y = _finite_number(signal.x, "signal x"), _finite_number(signal.y, "signal y")
            strength = _finite_number(signal.strength, "signal strength")
            remaining = _finite_number(signal.remaining, "signal remaining")
            if not 0.0 <= x <= self.width or not 0.0 <= y <= self.height:
                raise ValueError("signal is outside habitat")
            if isinstance(signal.tone, bool) or not isinstance(signal.tone, int) or signal.tone not in (0, 1, 2):
                raise ValueError("signal tone is invalid")
            if not 0.0 < strength <= 1.0 or not 0.0 < remaining <= 1.25:
                raise ValueError("signal strength or lifetime is invalid")


WorldObject = Object

__all__ = ["MODEL_DT", "DT", "MAX_OBJECTS", "MAX_SIGNALS", "Body", "Object", "WorldObject", "Signal", "World"]
