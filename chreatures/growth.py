"""Resource-gated parametric growth over the native world kernel.

The wrapper validates and canonically hashes immutable grammar data, translates
local receptor signals, and formats accepted geometry for PhysicsWorld's batched
topology transaction. Bud rewriting and all developmental randomness remain in
native private state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping

from .native_world import load_world_kernels


_KINDS = ("branch", "root", "leaf")
_SIGNAL_NAMES = ("light", "nutrient", "support")


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _vector(value: Any, length: int, name: str, low: float, high: float) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [_number(item, name, low, high) for item in value]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields differ: expected {sorted(expected)}")


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 80
        or not value[0].isalpha()
        or any(not (character.isalnum() or character in "_.-") for character in value)
    ):
        raise ValueError(f"invalid {name}")
    return value


def _normalize_grammar(raw: Any) -> dict[str, Any]:
    source = _mapping(copy.deepcopy(raw), "growth grammar")
    _keys(source, {
        "version", "name", "cadence_seconds", "initial_delay_seconds",
        "max_buds", "max_shapes_per_batch", "resources", "variation",
        "rules", "axiom",
    }, "growth grammar")
    if source["version"] != 1:
        raise ValueError("unsupported growth grammar version")
    name = _identifier(source["name"], "grammar name")
    cadence = _number(source["cadence_seconds"], "cadence_seconds", 0.05, 1e9)
    initial_delay = _number(source["initial_delay_seconds"], "initial_delay_seconds", 0.0, 1e9)
    max_buds = _integer(source["max_buds"], "max_buds", 1, 16_384)
    max_shapes = _integer(source["max_shapes_per_batch"], "max_shapes_per_batch", 1, 4096)

    resources = _mapping(source["resources"], "resources")
    _keys(resources, {"names", "composition", "atp_per_biomass"}, "resources")
    names = resources["names"]
    if not isinstance(names, list) or not 1 <= len(names) <= 64:
        raise ValueError("resource names must contain 1..64 entries")
    names = [_identifier(value, "resource name") for value in names]
    if len(names) != len(set(names)):
        raise ValueError("resource names must be unique")
    composition = _mapping(resources["composition"], "resource composition")
    atp = _mapping(resources["atp_per_biomass"], "ATP costs")
    _keys(composition, set(_KINDS), "resource composition")
    _keys(atp, set(_KINDS), "ATP costs")
    normalized_composition: dict[str, list[float]] = {}
    normalized_atp: dict[str, float] = {}
    for kind in _KINDS:
        values = _vector(composition[kind], len(names), f"{kind} resource composition", 0.0, 1.0)
        if abs(sum(values) - 1.0) > 1e-12:
            raise ValueError(f"{kind} resource composition must sum to one")
        normalized_composition[kind] = values
        normalized_atp[kind] = _number(atp[kind], f"{kind} ATP cost", 0.0, 1e9)

    variation = _mapping(source["variation"], "variation")
    _keys(variation, {"length_log_sigma", "angle_sigma_degrees", "leaf_log_sigma"}, "variation")
    normalized_variation = {
        "length_log_sigma": _number(variation["length_log_sigma"], "length variation", 0.0, 2.0),
        "angle_sigma_degrees": _number(variation["angle_sigma_degrees"], "angle variation", 0.0, 114.591559026),
        "leaf_log_sigma": _number(variation["leaf_log_sigma"], "leaf variation", 0.0, 2.0),
    }

    rules = _mapping(source["rules"], "rules")
    if not 1 <= len(rules) <= 64:
        raise ValueError("rules must contain 1..64 symbols")
    rule_names = [_identifier(value, "rule symbol") for value in rules]
    normalized_rules: dict[str, Any] = {}
    for symbol in sorted(rule_names):
        rule = _mapping(rules[symbol], f"rule {symbol}")
        expected = {"role", "segment", "activation", "successors", "leaf"}
        _keys(rule, expected, f"rule {symbol}")
        role = rule["role"]
        if role not in {"branch", "root"}:
            raise ValueError("growth rule role must be branch or root")
        segment = _mapping(rule["segment"], f"rule {symbol} segment")
        _keys(segment, {"length", "radius", "density"}, f"rule {symbol} segment")
        segment = {
            "length": _number(segment["length"], "segment length", 0.001, 10.0),
            "radius": _number(segment["radius"], "segment radius", 0.0005, 1.0),
            "density": _number(segment["density"], "segment density", 0.001, 30_000.0),
        }
        activation = _mapping(rule["activation"], f"rule {symbol} activation")
        _keys(activation, {"minimum", "weights", "competition_gain"}, f"rule {symbol} activation")
        minimum = _mapping(activation["minimum"], "activation minimum")
        weights = _mapping(activation["weights"], "activation weights")
        _keys(minimum, set(_SIGNAL_NAMES), "activation minimum")
        _keys(weights, set(_SIGNAL_NAMES), "activation weights")
        minimum_values = {key: _number(minimum[key], f"minimum {key}", 0.0, 1.0) for key in _SIGNAL_NAMES}
        weight_values = {key: _number(weights[key], f"weight {key}", 0.0, 1.0) for key in _SIGNAL_NAMES}
        if sum(weight_values.values()) <= 0.0:
            raise ValueError("activation weights require positive mass")
        normalized_activation = {
            "minimum": minimum_values,
            "weights": weight_values,
            "competition_gain": _number(activation["competition_gain"], "competition gain", 0.0, 20.0),
        }
        successors = rule["successors"]
        if not isinstance(successors, list) or len(successors) > 16:
            raise ValueError("successors must be a list with at most 16 entries")
        normalized_successors = []
        for successor in successors:
            successor = _mapping(successor, "successor")
            _keys(successor, {
                "symbol", "angle_degrees", "azimuth_degrees",
                "generation_phase_degrees", "scale", "probability",
            }, "successor")
            target = _identifier(successor["symbol"], "successor symbol")
            if target not in rules:
                raise ValueError("successor names an unknown rule")
            normalized_successors.append({
                "symbol": target,
                "angle_degrees": _number(successor["angle_degrees"], "branch angle", -180.0, 180.0),
                "azimuth_degrees": _number(successor["azimuth_degrees"], "branch azimuth", -57_295.0, 57_295.0),
                "generation_phase_degrees": _number(successor["generation_phase_degrees"], "generation phase", -57_295.0, 57_295.0),
                "scale": _number(successor["scale"], "successor scale", 0.05, 4.0),
                "probability": _number(successor["probability"], "successor probability", 0.0, 1.0),
            })
        leaf = rule["leaf"]
        if leaf is not None:
            leaf = _mapping(leaf, "leaf rule")
            _keys(leaf, {"probability", "area", "aspect", "thickness", "areal_density"}, "leaf rule")
            leaf = {
                "probability": _number(leaf["probability"], "leaf probability", 0.0, 1.0),
                "area": _number(leaf["area"], "leaf area", 1e-7, 10.0),
                "aspect": _number(leaf["aspect"], "leaf aspect", 0.05, 20.0),
                # Physics shape sizes are half-extents and require >= 2 mm.
                "thickness": _number(leaf["thickness"], "leaf thickness", 0.004, 0.2),
                "areal_density": _number(leaf["areal_density"], "leaf areal density", 0.0001, 10_000.0),
            }
        normalized_rules[symbol] = {
            "role": role, "segment": segment, "activation": normalized_activation,
            "successors": normalized_successors, "leaf": leaf,
        }

    axiom = source["axiom"]
    if not isinstance(axiom, list) or not 1 <= len(axiom) <= max_buds:
        raise ValueError("axiom must contain 1..max_buds initial buds")
    normalized_axiom = []
    for bud in axiom:
        bud = _mapping(bud, "axiom bud")
        _keys(bud, {"symbol", "position", "forward", "up", "scale"}, "axiom bud")
        symbol = _identifier(bud["symbol"], "axiom symbol")
        if symbol not in rules:
            raise ValueError("axiom names an unknown rule")
        forward = _vector(bud["forward"], 3, "bud forward", -1e6, 1e6)
        up = _vector(bud["up"], 3, "bud up", -1e6, 1e6)
        cross = [forward[1] * up[2] - forward[2] * up[1], forward[2] * up[0] - forward[0] * up[2], forward[0] * up[1] - forward[1] * up[0]]
        if sum(value * value for value in cross) < 1e-18:
            raise ValueError("bud forward and up cannot be parallel")
        normalized_axiom.append({
            "symbol": symbol,
            "position": _vector(bud["position"], 3, "bud position", -1e6, 1e6),
            "forward": forward, "up": up,
            "scale": _number(bud["scale"], "bud scale", 0.01, 100.0),
        })
    return {
        "version": 1, "name": name, "cadence_seconds": cadence,
        "initial_delay_seconds": initial_delay, "max_buds": max_buds,
        "max_shapes_per_batch": max_shapes,
        "resources": {"names": names, "composition": normalized_composition, "atp_per_biomass": normalized_atp},
        "variation": normalized_variation, "rules": normalized_rules,
        "axiom": normalized_axiom,
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class GrowthSystem:
    """One private developmental state sharing an immutable grammar."""

    VERSION = 1

    def __init__(self, grammar: dict[str, Any] | str | Path, seed: int = 1):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
            raise ValueError("growth seed must be an unsigned 64-bit integer")
        seed = int(seed)
        raw = json.loads(Path(grammar).read_text()) if isinstance(grammar, (str, Path)) else grammar
        self._grammar = _normalize_grammar(raw)
        self._canonical = _canonical(self._grammar)
        self.grammar_hash = hashlib.sha256(self._canonical.encode()).hexdigest()
        self.seed = seed
        self.resource_names = tuple(self._grammar["resources"]["names"])
        self.max_shapes_per_batch = self._grammar["max_shapes_per_batch"]
        rules = []
        for symbol, rule in self._grammar["rules"].items():
            segment, activation = rule["segment"], rule["activation"]
            successors = [(
                item["symbol"], math.radians(item["angle_degrees"]),
                math.radians(item["azimuth_degrees"]),
                math.radians(item["generation_phase_degrees"]),
                item["scale"], item["probability"],
            ) for item in rule["successors"]]
            leaf = rule["leaf"]
            leaf_tuple = None if leaf is None else (
                leaf["probability"], leaf["area"], leaf["aspect"],
                leaf["thickness"], leaf["areal_density"],
            )
            rules.append((
                symbol, rule["role"], segment["length"], segment["radius"],
                segment["density"],
                [activation["minimum"][key] for key in _SIGNAL_NAMES],
                [activation["weights"][key] for key in _SIGNAL_NAMES],
                activation["competition_gain"], successors, leaf_tuple,
            ))
        axiom = [(
            item["symbol"], item["position"], item["forward"], item["up"], item["scale"],
        ) for item in self._grammar["axiom"]]
        composition = self._grammar["resources"]["composition"]
        resource_rows = [[composition[kind][index] for kind in _KINDS] for index in range(len(self.resource_names))]
        atp = [self._grammar["resources"]["atp_per_biomass"][kind] for kind in _KINDS]
        variation = self._grammar["variation"]
        native = load_world_kernels()
        kernel_type = getattr(native, "GrowthKernel", None)
        if kernel_type is None:
            raise RuntimeError(
                "installed _world_kernels predates GrowthKernel; rebuild native/world-kernels"
            )
        self._kernel = kernel_type(
            self._canonical, self.grammar_hash, seed, rules, axiom,
            list(self.resource_names), resource_rows, atp,
            [variation["length_log_sigma"], math.radians(variation["angle_sigma_degrees"]), variation["leaf_log_sigma"]],
            self._grammar["cadence_seconds"], self._grammar["initial_delay_seconds"],
            self._grammar["max_buds"],
        )
        self._pending: dict[str, Any] | None = None

    @property
    def grammar(self) -> dict[str, Any]:
        return copy.deepcopy(self._grammar)

    def elapse(self, dt: float) -> float:
        return float(self._kernel.elapse(_number(dt, "growth dt", 0.0, 1e6)))

    def buds(self) -> list[dict[str, Any]]:
        """Return local receptor poses for environmental sampling machinery."""
        return [{
            "bud_id": int(item[0]), "symbol": item[1], "role": item[2],
            "generation": int(item[3]), "scale": float(item[4]),
            "position": list(item[5]), "forward": list(item[6]),
            "up": list(item[7]), "right": list(item[8]),
        } for item in self._kernel.buds()]

    @staticmethod
    def _signals(signals: Any) -> list[tuple[int, list[float]]]:
        if not isinstance(signals, list):
            raise ValueError("bud signals must be a list")
        result = []
        seen: set[int] = set()
        for value in signals:
            value = _mapping(value, "bud signal")
            _keys(value, {"bud_id", "light", "nutrient", "support", "competition"}, "bud signal")
            bud_id = _integer(value["bud_id"], "bud id", 1, 2**64 - 1)
            if bud_id in seen:
                raise ValueError("bud signals must be unique")
            result.append((bud_id, [
                _number(value["light"], "bud light", 0.0, 1.0),
                _number(value["nutrient"], "bud nutrient", 0.0, 1.0),
                _number(value["support"], "bud support", 0.0, 1.0),
                _number(value["competition"], "bud competition", 0.0, 1.0),
            ]))
            seen.add(bud_id)
        return sorted(result)

    def propose(self, signals: list[dict[str, Any]], structural_budget: float) -> dict[str, Any] | None:
        if self._pending is not None:
            raise RuntimeError("a growth proposal is already pending")
        normalized_signals = self._signals(signals)
        budget = _number(structural_budget, "structural budget", 0.0, 1e9)
        raw = self._kernel.propose(normalized_signals, budget, self.max_shapes_per_batch)
        if raw is None:
            return None
        proposal = {
            "token": raw[0], "grammar_hash": self.grammar_hash,
            "generation": int(raw[1]), "activated_buds": [int(value) for value in raw[2]],
            "request": {
                "biomass": float(raw[3]),
                "kind_biomass": {kind: float(raw[4][index]) for index, kind in enumerate(_KINDS)},
                "resources": {name: float(raw[5][index]) for index, name in enumerate(self.resource_names)},
                "resource_vector": [float(value) for value in raw[5]],
                "atp": float(raw[6]),
            },
            "geometry": {
                "segments": [{
                    "id": item[0], "kind": item[1], "parent_bud": int(item[2]),
                    "from": list(item[3]), "to": list(item[4]),
                    "radius": float(item[5]), "biomass": float(item[6]),
                } for item in raw[7]],
                "leaves": [{
                    "id": item[0], "kind": "leaf", "parent_bud": int(item[1]),
                    "position": list(item[2]), "quaternion": list(item[3]),
                    "size": list(item[4]), "area": float(item[5]), "biomass": float(item[6]),
                } for item in raw[8]],
            },
        }
        self._pending = {
            "signals": [{"bud_id": item[0], "light": item[1][0], "nutrient": item[1][1], "support": item[1][2], "competition": item[1][3]} for item in normalized_signals],
            "structural_budget": budget,
            "proposal": copy.deepcopy(proposal),
        }
        return proposal

    def reject(self, token: str) -> None:
        if self._pending is None or token != self._pending["proposal"]["token"]:
            raise ValueError("growth transaction token differs")
        self._kernel.reject(token)
        self._pending = None

    def commit(
        self, token: str, accepted_resources: Mapping[str, float],
        accepted_atp: float, *, physical_committed: bool,
    ) -> dict[str, Any]:
        if self._pending is None or token != self._pending["proposal"]["token"]:
            raise ValueError("growth transaction token differs")
        if not isinstance(accepted_resources, Mapping) or set(accepted_resources) != set(self.resource_names):
            raise ValueError("accepted resource receipt names differ")
        vector = [_number(accepted_resources[name], f"accepted {name}", 0.0, 1e9) for name in self.resource_names]
        atp = _number(accepted_atp, "accepted ATP", 0.0, 1e9)
        if not isinstance(physical_committed, bool):
            raise ValueError("physical_committed must be boolean")
        raw = self._kernel.commit(token, vector, atp, physical_committed)
        receipt = {
            "token": raw[0],
            "requested_resources": {name: float(raw[1][i]) for i, name in enumerate(self.resource_names)},
            "requested_atp": float(raw[2]),
            "accepted_resources": {name: float(raw[3][i]) for i, name in enumerate(self.resource_names)},
            "accepted_atp": float(raw[4]),
            "physical_committed": True,
        }
        self._pending = None
        return receipt

    @staticmethod
    def physical_operations(proposal: Mapping[str, Any], bindings: Mapping[str, str]) -> list[dict[str, Any]]:
        """Format one accepted candidate for a pre-existing three-entity plant."""
        geometry = _mapping(proposal.get("geometry"), "proposal geometry")
        _keys(geometry, {"segments", "leaves"}, "proposal geometry")
        if not isinstance(bindings, Mapping):
            raise ValueError("growth bindings must be a mapping")
        grouped: dict[str, list[dict[str, Any]]] = {kind: [] for kind in _KINDS}
        for segment in geometry["segments"]:
            kind = segment["kind"]
            if kind not in {"branch", "root"}:
                raise ValueError("invalid segment kind")
            grouped[kind].append({
                "type": "capsule", "size": [float(segment["radius"])],
                "fromto": [*map(float, segment["from"]), *map(float, segment["to"])],
            })
        for leaf in geometry["leaves"]:
            grouped["leaf"].append({
                "type": "ellipsoid", "size": list(map(float, leaf["size"])),
                "position": list(map(float, leaf["position"])),
                "quaternion": list(map(float, leaf["quaternion"])),
            })
        present = [kind for kind in _KINDS if grouped[kind]]
        if not set(present).issubset(bindings) or set(bindings) - set(_KINDS):
            raise ValueError("growth bindings do not cover the proposed geometry kinds")
        ids = [_identifier(bindings[kind], f"{kind} binding") for kind in present]
        if len(ids) != len(set(ids)):
            raise ValueError("each growth geometry kind requires a distinct entity")
        return [{"op": "append_shapes", "id": bindings[kind], "shapes": grouped[kind]} for kind in present]

    def snapshot(self) -> dict[str, Any]:
        raw = self._kernel.state()
        state = {
            "rng_state": int(raw[0]), "next_bud": int(raw[1]),
            "next_part": int(raw[2]), "generation": int(raw[3]),
            "clock": float(raw[4]), "next_due": float(raw[5]),
            "genotype": list(raw[6]),
            "buds": [{
                "bud_id": int(item[0]), "symbol": item[1], "role": item[2],
                "generation": int(item[3]), "scale": float(item[4]),
                "position": list(item[5]), "forward": list(item[6]),
                "up": list(item[7]), "right": list(item[8]),
            } for item in raw[7]],
        }
        return {
            "version": self.VERSION, "grammar_hash": self.grammar_hash,
            "seed": self.seed, "state": state, "pending": copy.deepcopy(self._pending),
        }

    @classmethod
    def restore(cls, grammar: dict[str, Any] | str | Path, snapshot: dict[str, Any]) -> "GrowthSystem":
        if not isinstance(snapshot, dict) or snapshot.get("version") != cls.VERSION:
            raise ValueError("unsupported growth snapshot")
        _keys(snapshot, {"version", "grammar_hash", "seed", "state", "pending"}, "growth snapshot")
        instance = cls(grammar, snapshot["seed"])
        if snapshot["grammar_hash"] != instance.grammar_hash:
            raise ValueError("growth snapshot grammar hash differs")
        state = _mapping(snapshot["state"], "growth state")
        _keys(state, {"rng_state", "next_bud", "next_part", "generation", "clock", "next_due", "genotype", "buds"}, "growth state")
        buds = state["buds"]
        if not isinstance(buds, list):
            raise ValueError("growth state buds must be a list")
        bud_values = []
        for bud in buds:
            bud = _mapping(bud, "growth bud")
            _keys(bud, {"bud_id", "symbol", "role", "generation", "scale", "position", "forward", "up", "right"}, "growth bud")
            bud_values.append((
                _integer(bud["bud_id"], "bud id", 1, 2**64 - 1),
                _identifier(bud["symbol"], "bud symbol"), bud["role"],
                _integer(bud["generation"], "bud generation", 0, 2**32 - 1),
                _number(bud["scale"], "bud scale", 0.001, 100.0),
                _vector(bud["position"], 3, "bud position", -1e6, 1e6),
                _vector(bud["forward"], 3, "bud forward", -1.0, 1.0),
                _vector(bud["up"], 3, "bud up", -1.0, 1.0),
                _vector(bud["right"], 3, "bud right", -1.0, 1.0),
            ))
        instance._kernel.restore_state(
            _integer(state["rng_state"], "growth RNG", 1, 2**64 - 1),
            _integer(state["next_bud"], "next bud", 1, 2**64 - 1),
            _integer(state["next_part"], "next part", 1, 2**64 - 1),
            _integer(state["generation"], "growth generation", 0, 2**32 - 1),
            _number(state["clock"], "growth clock", 0.0, 1e15),
            _number(state["next_due"], "next growth time", 0.0, 1e15),
            _vector(state["genotype"], 3, "growth genotype", -1e6, 1e6),
            bud_values,
        )
        pending = snapshot["pending"]
        if pending is not None:
            pending = _mapping(pending, "pending growth transaction")
            _keys(pending, {"signals", "structural_budget", "proposal"}, "pending growth transaction")
            replay = instance.propose(pending["signals"], pending["structural_budget"])
            if replay != pending["proposal"]:
                raise ValueError("pending growth proposal does not replay exactly")
        return instance


__all__ = ["GrowthSystem"]
