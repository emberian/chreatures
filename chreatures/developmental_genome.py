"""Immutable developmental inheritance compiled into existing native systems.

The genome contains bounded expression parameters and source identities.  It
never contains a living organism's metabolic pools, growth state, neural rates,
memories, or optimizer state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .growth import GrowthSystem
from .metabolism import Chemistry, MetabolicWeb


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHEMISTRY = ROOT / "data" / "metabolism" / "common-chemistry.json"
DEFAULT_GROWTH_GRAMMAR = ROOT / "data" / "growth" / "nursery-plant.json"
DEFAULT_PORT_BUNDLE = ROOT / "data" / "ports" / "retinal-v1-maps.npz"

FORMAT = "chreatures-developmental-genome-v1"
MUTATION_OPERATOR = "bounded-related-offspring-v1"
COMPARTMENTS = ("soma", "gut", "allocated_structure")
ALLOCATIONS = ("photosynthesis", "digestion", "structure", "geometry")
ALLOCATED_LOCI = {
    "photosynthesis": ("soma/carbon_fixation",),
    "digestion": (
        "gut/soft_digestion", "gut/tough_digestion", "gut/detritus_digestion",
    ),
    "structure": ("soma/soft_growth", "soma/tough_growth"),
}
HOUSEKEEPING_LOCI = (
    "soma/respiration", "gut/respiration",
    "allocated_structure/soft_turnover", "allocated_structure/tough_turnover",
)
PRIVATE_KEYS = {
    "state", "snapshot", "memory", "memories", "context", "optimizer",
    "rates", "adaptation", "support", "pools", "atp", "rng_state",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields differ: expected {sorted(expected)}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} is outside [{low}, {high}]")
    return result


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str) or not value or len(value) > 128
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"invalid {name}")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(value))
    body.pop("sha256", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _reject_private_state(value: Any, path: str = "genome") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in PRIVATE_KEYS:
                raise ValueError(f"private runtime field is forbidden in a genome: {path}.{key}")
            _reject_private_state(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_state(child, f"{path}[{index}]")


def _normalize_weights(values: Mapping[str, float], names: Sequence[str]) -> dict[str, float]:
    weights = np.asarray([values[name] for name in names], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("each allocated enzyme category needs positive expression mass")
    return {name: float(value / total) for name, value in zip(names, weights, strict=True)}


def _load_port_identity(path: Path) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {"metadata", "input_names", "readout_names", "input_shape", "readout_shape"}
        if not required.issubset(bundle.files):
            raise ValueError("neural port bundle lacks identity arrays")
        metadata = json.loads(str(bundle["metadata"]))
        inputs = tuple(str(value) for value in bundle["input_names"].tolist())
        readouts = tuple(str(value) for value in bundle["readout_names"].tolist())
        input_shape = tuple(int(value) for value in bundle["input_shape"])
        readout_shape = tuple(int(value) for value in bundle["readout_shape"])
    if (
        len(set(inputs)) != len(inputs) or len(set(readouts)) != len(readouts)
        or input_shape[1:] != (len(inputs),) or readout_shape[:1] != (len(readouts),)
        or input_shape[0] != readout_shape[1]
    ):
        raise ValueError("neural port bundle dimensions or names differ")
    return metadata, inputs, readouts


@dataclass(frozen=True)
class DevelopmentalPhenotype:
    """Compiled immutable expression with constructors for real subsystems."""

    genome_sha256: str
    chemistry: Chemistry
    base_growth_grammar_hash: str
    compiled_growth_grammar_hash: str
    _growth_grammar: dict[str, Any]
    _enzyme_rows: tuple[dict[str, float], ...]
    port_bundle_sha256: str
    graph_sha256: str
    port_spec_sha256: str
    input_names: tuple[str, ...]
    readout_names: tuple[str, ...]
    _input_gains: np.ndarray
    _readout_gains: np.ndarray
    allocation: tuple[float, ...]
    _expression_receipt: dict[str, Any]

    @property
    def compartment_names(self) -> tuple[str, ...]:
        return COMPARTMENTS

    @property
    def growth_grammar(self) -> dict[str, Any]:
        return copy.deepcopy(self._growth_grammar)

    @property
    def expression_receipt(self) -> dict[str, Any]:
        return copy.deepcopy(self._expression_receipt)

    @property
    def input_gains(self) -> np.ndarray:
        value = self._input_gains.view()
        value.flags.writeable = False
        return value

    @property
    def readout_gains(self) -> np.ndarray:
        value = self._readout_gains.view()
        value.flags.writeable = False
        return value

    def enzyme_rows(self) -> tuple[dict[str, float], ...]:
        """Return soma, gut, and allocated-structure rows for a shared web."""
        return tuple(copy.deepcopy(row) for row in self._enzyme_rows)

    def new_growth(self, seed: int) -> GrowthSystem:
        return GrowthSystem(self._growth_grammar, seed=seed)

    def new_metabolism(
        self, pools: Sequence[Mapping[str, float]], atp: Sequence[float],
        atp_capacity: Sequence[float], *, bulk: Mapping[str, float] | None = None,
        bulk_atp: float = 0.0,
    ) -> MetabolicWeb:
        """Construct a three-compartment native web for an isolated organism."""
        if len(pools) != len(COMPARTMENTS):
            raise ValueError("metabolic state must provide soma, gut, and allocated_structure rows")
        return MetabolicWeb(
            self.chemistry, self.enzyme_rows(), pools, atp, atp_capacity,
            bulk=bulk, bulk_atp=bulk_atp,
        )

    def apply_input_gains(self, channels: Any) -> np.ndarray:
        values = np.asarray(channels, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] != len(self.input_names) or not np.isfinite(values).all():
            raise ValueError("input channels must be finite with the compiled port width")
        return np.asarray(values * self._input_gains, dtype=np.float32)

    def apply_readout_gains(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] != len(self.readout_names) or not np.isfinite(values).all():
            raise ValueError("readouts must be finite with the compiled port width")
        return np.asarray(values * self._readout_gains, dtype=np.float32)


class DevelopmentalGenome:
    """Validated declarative genotype with deterministic compilation and descent."""

    def __init__(self, value: Mapping[str, Any]):
        raw = copy.deepcopy(_mapping(value, "developmental genome"))
        _reject_private_state(raw)
        _keys(raw, {
            "format", "name", "generation", "ancestry", "sources", "allocation",
            "metabolism", "growth", "neural", "sha256",
        }, "developmental genome")
        if raw["format"] != FORMAT:
            raise ValueError("unsupported developmental genome")
        raw["name"] = _identifier(raw["name"], "genome name")
        generation = raw["generation"]
        if isinstance(generation, bool) or not isinstance(generation, Integral) or not 0 <= generation < 2**31:
            raise ValueError("generation must be a nonnegative integer")
        raw["generation"] = int(generation)
        self._validate_ancestry(raw["ancestry"], raw["generation"])
        self._validate_sources(raw["sources"])
        self._validate_allocation(raw["allocation"])
        self._validate_metabolism(raw["metabolism"])
        self._validate_growth(raw["growth"])
        self._validate_neural(raw["neural"])
        expected = _digest(raw)
        if raw["sha256"] not in (None, expected):
            raise ValueError("developmental genome checksum differs")
        raw["sha256"] = expected
        self._value = raw
        self.sha256 = expected

    @staticmethod
    def _validate_ancestry(value: Any, generation: int) -> None:
        value = _mapping(value, "ancestry")
        _keys(value, {"parent_sha256", "founder_sha256", "mutation"}, "ancestry")
        if generation == 0:
            if any(value[key] is not None for key in value):
                raise ValueError("a founder cannot name a parent, founder, or mutation")
            return
        for key in ("parent_sha256", "founder_sha256"):
            item = value[key]
            if not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item):
                raise ValueError(f"invalid ancestry {key}")
        mutation = _mapping(value["mutation"], "ancestry mutation")
        _keys(mutation, {"operator", "seed", "scale"}, "ancestry mutation")
        if mutation["operator"] != MUTATION_OPERATOR:
            raise ValueError("unsupported offspring mutation operator")
        seed = mutation["seed"]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
            raise ValueError("mutation seed must be unsigned 64-bit")
        _number(mutation["scale"], "mutation scale", 1e-6, 0.5)

    @staticmethod
    def _validate_sources(value: Any) -> None:
        value = _mapping(value, "sources")
        _keys(value, {
            "chemistry_sha256", "chemistry_file_sha256",
            "growth_grammar_sha256", "growth_grammar_file_sha256",
            "graph_sha256", "port_spec_sha256", "port_bundle_sha256",
        }, "sources")
        for key, item in value.items():
            if not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item):
                raise ValueError(f"invalid source digest {key}")

    @staticmethod
    def _validate_allocation(value: Any) -> None:
        value = _mapping(value, "allocation")
        _keys(value, {"fractions", "enzyme_activity_budget", "geometry_reference_fraction"}, "allocation")
        fractions = _mapping(value["fractions"], "allocation fractions")
        _keys(fractions, set(ALLOCATIONS), "allocation fractions")
        values = [_number(fractions[name], f"allocation {name}", 0.05, 0.85) for name in ALLOCATIONS]
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("allocation fractions must sum to one")
        _number(value["enzyme_activity_budget"], "enzyme activity budget", 0.1, 16.0)
        _number(value["geometry_reference_fraction"], "geometry reference", 0.05, 0.85)

    @staticmethod
    def _validate_metabolism(value: Any) -> None:
        value = _mapping(value, "metabolism")
        _keys(value, {"allocated_expression", "housekeeping_expression", "housekeeping_ceiling"}, "metabolism")
        allocated = _mapping(value["allocated_expression"], "allocated enzyme expression")
        _keys(allocated, set(ALLOCATED_LOCI), "allocated enzyme expression")
        for category, allowed_loci in ALLOCATED_LOCI.items():
            loci = _mapping(allocated[category], f"{category} expression")
            if not set(loci).issubset(allowed_loci) or not loci:
                raise ValueError(f"{category} expression contains invalid or no loci")
            values = {
                name: _number(level, f"enzyme expression {name}", 0.0, 1.0)
                for name, level in loci.items()
            }
            if sum(values.values()) <= 0.0:
                raise ValueError(f"{category} expression needs positive mass")
        housekeeping = _mapping(value["housekeeping_expression"], "housekeeping expression")
        if not set(housekeeping).issubset(HOUSEKEEPING_LOCI):
            raise ValueError("housekeeping expression contains an invalid locus")
        for name, level in housekeeping.items():
            _number(level, f"housekeeping expression {name}", 0.0, 1.0)
        _number(value["housekeeping_ceiling"], "housekeeping ceiling", 0.0, 4.0)

    @staticmethod
    def _validate_growth(value: Any) -> None:
        value = _mapping(value, "growth")
        _keys(value, {"length", "radius", "leaf_area", "leaf_thickness"}, "growth expression")
        for name, level in value.items():
            low = 1.0 if name == "leaf_thickness" else 0.75
            _number(level, f"growth expression {name}", low, 1.25)

    @staticmethod
    def _validate_neural(value: Any) -> None:
        value = _mapping(value, "neural")
        _keys(value, {"input_gain_loci", "readout_gain_loci"}, "neural expression")
        for field in ("input_gain_loci", "readout_gain_loci"):
            loci = value[field]
            if not isinstance(loci, list) or len(loci) > 128:
                raise ValueError(f"{field} must be a sparse list with at most 128 loci")
            names = []
            for locus in loci:
                locus = _mapping(locus, "neural gain locus")
                _keys(locus, {"name", "gain"}, "neural gain locus")
                names.append(_identifier(locus["name"], "neural port name"))
                _number(locus["gain"], "neural port gain", 0.5, 1.5)
            if len(names) != len(set(names)):
                raise ValueError(f"{field} contains duplicate names")

    @classmethod
    def load(cls, path: str | Path) -> "DevelopmentalGenome":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def generation(self) -> int:
        return int(self._value["generation"])

    def to_value(self) -> dict[str, Any]:
        return copy.deepcopy(self._value)

    def compile(
        self, *, chemistry: str | Path = DEFAULT_CHEMISTRY,
        growth_grammar: str | Path = DEFAULT_GROWTH_GRAMMAR,
        port_bundle: str | Path = DEFAULT_PORT_BUNDLE,
    ) -> DevelopmentalPhenotype:
        chemistry_path, growth_path, port_path = map(
            lambda value: Path(value).resolve(), (chemistry, growth_grammar, port_bundle)
        )
        sources = self._value["sources"]
        chemistry_value = Chemistry.load(chemistry_path)
        if (
            chemistry_value.sha256 != sources["chemistry_sha256"]
            or _sha256_file(chemistry_path) != sources["chemistry_file_sha256"]
        ):
            raise ValueError("chemistry source identity differs from genome")
        base_growth = GrowthSystem(growth_path, seed=0)
        if (
            base_growth.grammar_hash != sources["growth_grammar_sha256"]
            or _sha256_file(growth_path) != sources["growth_grammar_file_sha256"]
        ):
            raise ValueError("growth source identity differs from genome")
        metadata, input_names, readout_names = _load_port_identity(port_path)
        if (
            _sha256_file(port_path) != sources["port_bundle_sha256"]
            or metadata.get("graph_hash") != sources["graph_sha256"]
            or metadata.get("spec_hash") != sources["port_spec_sha256"]
        ):
            raise ValueError("neural graph or port identity differs from genome")

        allocation = self._value["allocation"]
        fractions = allocation["fractions"]
        enzyme_budget = float(allocation["enzyme_activity_budget"])
        rows = {name: {} for name in COMPARTMENTS}
        allocated_receipt: dict[str, float] = {}
        for category, allowed_loci in ALLOCATED_LOCI.items():
            loci = self._value["metabolism"]["allocated_expression"][category]
            weights = _normalize_weights(loci, tuple(sorted(loci)))
            category_budget = enzyme_budget * float(fractions[category])
            for locus, weight in weights.items():
                compartment, reaction = locus.split("/", 1)
                if locus not in allowed_loci:
                    raise AssertionError("validated enzyme compartment changed")
                activity = category_budget * weight
                rows[compartment][reaction] = activity
                allocated_receipt[locus] = activity
        housekeeping_ceiling = float(self._value["metabolism"]["housekeeping_ceiling"])
        housekeeping_receipt: dict[str, float] = {}
        for locus, expression in self._value["metabolism"]["housekeeping_expression"].items():
            compartment, reaction = locus.split("/", 1)
            activity = housekeeping_ceiling * float(expression)
            rows[compartment][reaction] = activity
            housekeeping_receipt[locus] = activity
        known_reactions = set(chemistry_value.reactions)
        if any(set(row) - known_reactions for row in rows.values()):
            raise ValueError("genome enzyme locus is absent from chemistry")
        # Validate the exact emitted rows through Chemistry before constructing a web.
        chemistry_value.enzymes([rows[name] for name in COMPARTMENTS])

        geometry_fraction = float(fractions["geometry"])
        geometry_reference = float(allocation["geometry_reference_fraction"])
        shared_factor = float(np.clip(math.sqrt(geometry_fraction / geometry_reference), 0.75, 1.25))
        genes = self._value["growth"]
        multipliers = {
            "length": float(np.clip(shared_factor * genes["length"], 0.65, 1.35)),
            "radius": float(np.clip(math.sqrt(shared_factor) * genes["radius"], 0.65, 1.35)),
            "leaf_area": float(np.clip(shared_factor * shared_factor * genes["leaf_area"], 0.65, 1.35)),
            # The current physical grammar sits at GrowthSystem's 4 mm leaf
            # half-extent floor, so descendants may thicken but not shrink it.
            "leaf_thickness": float(np.clip(
                math.sqrt(shared_factor) * genes["leaf_thickness"], 1.0, 1.35
            )),
        }
        grammar = base_growth.grammar
        for rule in grammar["rules"].values():
            rule["segment"]["length"] *= multipliers["length"]
            rule["segment"]["radius"] *= multipliers["radius"]
            if rule["leaf"] is not None:
                rule["leaf"]["area"] *= multipliers["leaf_area"]
                rule["leaf"]["thickness"] *= multipliers["leaf_thickness"]
        compiled_growth = GrowthSystem(grammar, seed=0)

        input_gains = np.ones(len(input_names), dtype=np.float32)
        readout_gains = np.ones(len(readout_names), dtype=np.float32)
        for field, names, gains in (
            ("input_gain_loci", input_names, input_gains),
            ("readout_gain_loci", readout_names, readout_gains),
        ):
            positions = {name: index for index, name in enumerate(names)}
            for locus in self._value["neural"][field]:
                if locus["name"] not in positions:
                    raise ValueError(f"neural gain locus is absent from exact port bundle: {locus['name']}")
                gains[positions[locus["name"]]] = np.float32(locus["gain"])
        input_gains.flags.writeable = False
        readout_gains.flags.writeable = False
        allocation_tuple = tuple(float(fractions[name]) for name in ALLOCATIONS)
        receipt = {
            "allocation": dict(zip(ALLOCATIONS, allocation_tuple, strict=True)),
            "allocated_enzyme_activity": allocated_receipt,
            "housekeeping_enzyme_activity": housekeeping_receipt,
            "enzyme_rows_sha256": hashlib.sha256(_canonical(
                [{"compartment": name, "enzymes": rows[name]} for name in COMPARTMENTS]
            )).hexdigest(),
            "geometry_shared_factor": shared_factor,
            "growth_multipliers": multipliers,
            "compiled_growth_grammar_sha256": compiled_growth.grammar_hash,
            "neural_gain_counts": {
                "input": len(self._value["neural"]["input_gain_loci"]),
                "readout": len(self._value["neural"]["readout_gain_loci"]),
            },
            "input_gains_sha256": hashlib.sha256(input_gains.tobytes()).hexdigest(),
            "readout_gains_sha256": hashlib.sha256(readout_gains.tobytes()).hexdigest(),
            "engineering_scope": (
                "engineered bounded inheritance constraint; not a measured developmental tradeoff"
            ),
        }
        return DevelopmentalPhenotype(
            genome_sha256=self.sha256, chemistry=chemistry_value,
            base_growth_grammar_hash=base_growth.grammar_hash,
            compiled_growth_grammar_hash=compiled_growth.grammar_hash,
            _growth_grammar=compiled_growth.grammar,
            _enzyme_rows=tuple(rows[name] for name in COMPARTMENTS),
            port_bundle_sha256=sources["port_bundle_sha256"],
            graph_sha256=sources["graph_sha256"],
            port_spec_sha256=sources["port_spec_sha256"],
            input_names=input_names, readout_names=readout_names,
            _input_gains=input_gains, _readout_gains=readout_gains,
            allocation=allocation_tuple, _expression_receipt=receipt,
        )

    def offspring(self, seed: int, mutation_scale: float = 0.08) -> "DevelopmentalGenome":
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
            raise ValueError("offspring seed must be an unsigned 64-bit integer")
        scale = _number(mutation_scale, "mutation scale", 1e-6, 0.5)
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        child = self.to_value()
        child["name"] = f"related-offspring-g{self.generation + 1}-{int(seed):016x}"
        child["generation"] = self.generation + 1
        founder = self.sha256 if self.generation == 0 else self._value["ancestry"]["founder_sha256"]
        child["ancestry"] = {
            "parent_sha256": self.sha256, "founder_sha256": founder,
            "mutation": {"operator": MUTATION_OPERATOR, "seed": int(seed), "scale": scale},
        }
        fractions = child["allocation"]["fractions"]
        logits = np.log(np.asarray([fractions[name] for name in ALLOCATIONS]))
        logits += rng.normal(0.0, scale, len(ALLOCATIONS))
        proportions = np.exp(logits - logits.max())
        proportions /= proportions.sum()
        proportions = 0.05 + 0.80 * proportions
        for name, value in zip(ALLOCATIONS, proportions, strict=True):
            fractions[name] = float(value)

        def mutate_unit(value: float) -> float:
            bounded = float(np.clip(value, 1e-6, 1.0 - 1e-6))
            logit = math.log(bounded / (1.0 - bounded)) + float(rng.normal(0.0, scale))
            return float(1.0 / (1.0 + math.exp(-logit)))

        metabolism = child["metabolism"]
        for loci in metabolism["allocated_expression"].values():
            for name in sorted(loci):
                loci[name] = mutate_unit(float(loci[name]))
        for name in sorted(metabolism["housekeeping_expression"]):
            metabolism["housekeeping_expression"][name] = mutate_unit(
                float(metabolism["housekeeping_expression"][name])
            )
        for name in sorted(child["growth"]):
            low = 1.0 if name == "leaf_thickness" else 0.75
            child["growth"][name] = float(np.clip(
                float(child["growth"][name]) * math.exp(float(rng.normal(0.0, scale))),
                low, 1.25,
            ))
        for field in ("input_gain_loci", "readout_gain_loci"):
            for locus in child["neural"][field]:
                locus["gain"] = float(np.clip(
                    float(locus["gain"]) * math.exp(float(rng.normal(0.0, scale))),
                    0.5, 1.5,
                ))
        child["sha256"] = None
        result = DevelopmentalGenome(child)
        if result.sha256 == self.sha256:
            raise RuntimeError("positive mutation scale did not produce a distinct offspring")
        return result


__all__ = [
    "ALLOCATIONS", "COMPARTMENTS", "DevelopmentalGenome",
    "DevelopmentalPhenotype", "FORMAT", "MUTATION_OPERATOR",
]
