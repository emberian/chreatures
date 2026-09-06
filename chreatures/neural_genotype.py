"""Heritable, annotation-grounded modulation of a full MaleCNS circuit.

The genotype stores immutable bounded parameters and source identities.  The
compiled phenotype contains no rates, support, adaptation, memories, optimizer
state, or other state belonging to a living resident.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .neural_ports import NeuralPortBundle


RECIPE_FORMAT = "chreatures-neural-variant-recipe-v1"
GENOTYPE_FORMAT = "chreatures-neural-variant-genotype-v1"
PHENOTYPE_FORMAT = "chreatures-neural-variant-phenotype-v1"
STRUCTURAL_KIND = "circuit_blueprint"

NEURON_PARAMETERS = (
    "excitability_gain",
    "recurrent_source_gain",
    "recurrent_target_gain",
    "learning_rate_gain",
    "modulator_gain",
)
NEURAL_VARIANT_ARRAYS = (
    "input_gain",
    "readout_gain",
    "excitability_gain",
    "recurrent_source_gain",
    "recurrent_target_gain",
    "learning_rate_gain",
    "modulator_gain",
)
POPULATION_LOCI = (
    "input_gain",
    "readout_gain",
    "excitability",
    "recurrent_gain",
    "learning_rate_gain",
    "modulator_gain",
)

_PRIVATE_KEYS = {
    "rates",
    "support",
    "adaptation",
    "memory",
    "optimizer",
    "snapshot",
    "state",
    "rng_state",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(value))
    body.pop("sha256", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields differ: expected {sorted(expected)}")


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _name(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"invalid {name}")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} is outside [{low},{high}]")
    return result


def _reject_private(value: Any, path: str = "neural genotype") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _PRIVATE_KEYS:
                raise ValueError(f"private runtime field is forbidden: {path}.{key}")
            _reject_private(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private(child, f"{path}[{index}]")


def _readonly(value: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(value, dtype=np.float32)
    value.flags.writeable = False
    return value


@dataclass(frozen=True)
class NeuralVariantRecipe:
    document: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "NeuralVariantRecipe":
        return cls.from_value(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "NeuralVariantRecipe":
        document = json.loads(_canonical(dict(value)))
        _keys(
            document,
            {
                "format",
                "name",
                "source",
                "overlap_policy",
                "bounds",
                "neuron_groups",
                "input_groups",
                "readout_groups",
                "structural_templates",
                "model_boundary",
                "sha256",
            },
            "neural variant recipe",
        )
        if document["format"] != RECIPE_FORMAT:
            raise ValueError("unsupported neural variant recipe")
        _name(document["name"], "recipe name")
        if document["overlap_policy"] != "multiply_then_clip":
            raise ValueError("unsupported neural group overlap policy")
        source = _mapping(document["source"], "recipe source")
        _keys(
            source,
            {
                "canonical_graph_sha256",
                "port_bundle_sha256",
                "port_spec_sha256",
                "input_count",
                "readout_count",
            },
            "recipe source",
        )
        for field in (
            "canonical_graph_sha256",
            "port_bundle_sha256",
            "port_spec_sha256",
        ):
            _sha(source[field], f"source.{field}")
        if source["input_count"] != 351 or source["readout_count"] != 384:
            raise ValueError("recipe must use the current 351/384 neural ports")
        bounds = _mapping(document["bounds"], "recipe bounds")
        _keys(
            bounds,
            {"parameters", "combined", "mutation_log_sigma_max"},
            "recipe bounds",
        )
        parameter_bounds = _mapping(bounds["parameters"], "parameter bounds")
        expected_parameters = {*NEURON_PARAMETERS, "input_gain", "readout_gain"}
        if set(parameter_bounds) != expected_parameters:
            raise ValueError("neural recipe parameter bounds differ")
        for parameter, interval in parameter_bounds.items():
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"bounds for {parameter} must be [low,high]")
            low = _number(interval[0], f"{parameter} lower bound", 0.05, 4.0)
            high = _number(interval[1], f"{parameter} upper bound", 0.05, 4.0)
            if not low < 1.0 < high:
                raise ValueError(f"bounds for {parameter} must straddle one")
        combined = _mapping(bounds["combined"], "combined bounds")
        if set(combined) != expected_parameters:
            raise ValueError("combined neural bounds differ")
        for parameter, interval in combined.items():
            low, high = map(float, interval)
            own_low, own_high = map(float, parameter_bounds[parameter])
            if not 0.05 <= low <= own_low or not own_high <= high <= 4.0:
                raise ValueError(f"combined bounds for {parameter} are invalid")
        _number(bounds["mutation_log_sigma_max"], "mutation sigma", 0.01, 0.5)
        cls._validate_groups(document["neuron_groups"], "neuron", set(NEURON_PARAMETERS))
        cls._validate_groups(document["input_groups"], "input", {"input_gain"})
        cls._validate_groups(document["readout_groups"], "readout", {"readout_gain"})
        templates = document["structural_templates"]
        if not isinstance(templates, list):
            raise TypeError("structural_templates must be a list")
        template_names: set[str] = set()
        for item in templates:
            item = _mapping(item, "structural template")
            _keys(item, {"name", "template", "bounds"}, "structural template")
            name = _name(item["name"], "structural template name")
            if name in template_names:
                raise ValueError("duplicate structural template name")
            template_names.add(name)
            if not isinstance(item["template"], dict) or not isinstance(item["bounds"], dict):
                raise TypeError("structural template and bounds must be objects")
        boundary = _mapping(document["model_boundary"], "model boundary")
        _keys(boundary, {"measured", "engineered", "excluded"}, "model boundary")
        expected = _digest(document)
        if document["sha256"] not in (None, expected):
            raise ValueError("neural variant recipe checksum differs")
        document["sha256"] = expected
        return cls(document)

    @staticmethod
    def _validate_groups(groups: Any, kind: str, allowed_parameters: set[str]) -> None:
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"{kind}_groups must be nonempty")
        names: set[str] = set()
        for group in groups:
            group = _mapping(group, f"{kind} group")
            expected = (
                {"name", "selectors", "parameters", "basis"}
                if kind == "neuron"
                else {"name", "match", "parameters", "basis"}
            )
            _keys(group, expected, f"{kind} group")
            name = _name(group["name"], f"{kind} group name")
            if name in names:
                raise ValueError(f"duplicate {kind} group name")
            names.add(name)
            parameters = group["parameters"]
            if not isinstance(parameters, list) or not parameters or not set(parameters) <= allowed_parameters:
                raise ValueError(f"{kind} group parameters differ")
            if len(parameters) != len(set(parameters)):
                raise ValueError(f"duplicate parameter in {kind} group")
            if kind == "neuron":
                selectors = group["selectors"]
                if not isinstance(selectors, list) or not selectors:
                    raise ValueError("neuron selectors must be a nonempty OR-list")
                if any(not isinstance(selector, dict) or not selector for selector in selectors):
                    raise ValueError("each neuron selector must be a nonempty exact-field object")
            elif not isinstance(group["match"], dict) or not group["match"]:
                raise ValueError(f"{kind} match must be a nonempty exact metadata object")

    @property
    def sha256(self) -> str:
        return self.document["sha256"]

    def to_value(self) -> dict[str, Any]:
        return copy.deepcopy(self.document)


class NeuralVariantGenotype:
    def __init__(self, value: Mapping[str, Any], recipe: NeuralVariantRecipe):
        document = json.loads(_canonical(dict(value)))
        _reject_private(document)
        _keys(
            document,
            {
                "format",
                "name",
                "generation",
                "parents",
                "variation",
                "sources",
                "loci",
                "structure",
                "sha256",
            },
            "neural genotype",
        )
        if document["format"] != GENOTYPE_FORMAT:
            raise ValueError("unsupported neural genotype")
        _name(document["name"], "neural genotype name")
        generation = document["generation"]
        if isinstance(generation, bool) or not isinstance(generation, Integral) or generation < 0:
            raise ValueError("neural genotype generation must be nonnegative")
        parents = document["parents"]
        if not isinstance(parents, list) or len(parents) > 2:
            raise ValueError("neural genotype must have zero, one, or two parents")
        for parent in parents:
            _sha(parent, "parent genotype")
        if generation == 0 and parents or generation > 0 and not parents:
            raise ValueError("neural genotype generation and parents disagree")
        variation = _mapping(document["variation"], "neural variation")
        _keys(
            variation,
            {"recipe_sha256", "seed", "mutation_scale", "population_loci"},
            "neural variation",
        )
        if variation["recipe_sha256"] != recipe.sha256:
            raise ValueError("neural genotype belongs to another recipe")
        seed = variation["seed"]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
            raise ValueError("neural variation seed must be uint64")
        _number(variation["mutation_scale"], "mutation scale", 0.0, 1.0)
        population_loci = _mapping(variation["population_loci"], "population neural loci")
        if set(population_loci) != set(POPULATION_LOCI):
            raise ValueError("population neural loci differ")
        for locus in POPULATION_LOCI:
            _number(population_loci[locus], f"population neural {locus}", 0.70, 1.35)
        self._validate_sources(document["sources"], recipe, document["structure"])
        self._validate_loci(document["loci"], recipe)
        expected = _digest(document)
        if document["sha256"] not in (None, expected):
            raise ValueError("neural genotype checksum differs")
        document["sha256"] = expected
        self._value = document
        self.sha256 = expected
        self.recipe = recipe

    @classmethod
    def load(cls, path: str | Path, recipe: NeuralVariantRecipe) -> "NeuralVariantGenotype":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), recipe)

    @staticmethod
    def _validate_sources(value: Any, recipe: NeuralVariantRecipe, structure: Any) -> None:
        source = _mapping(value, "neural genotype sources")
        _keys(
            source,
            {
                "canonical_graph_sha256",
                "active_graph_sha256",
                "port_bundle_sha256",
                "port_spec_sha256",
                "base_controller_sha256",
            },
            "neural genotype sources",
        )
        canonical = recipe.document["source"]
        if source["canonical_graph_sha256"] != canonical["canonical_graph_sha256"]:
            raise ValueError("canonical neural ancestor differs")
        for field in source:
            _sha(source[field], f"neural source {field}")
        if structure is None:
            if (
                source["active_graph_sha256"] != source["canonical_graph_sha256"]
                or source["port_bundle_sha256"] != canonical["port_bundle_sha256"]
                or source["port_spec_sha256"] != canonical["port_spec_sha256"]
            ):
                raise ValueError("unstructured genotype must use canonical graph and ports")
            return
        structure = _mapping(structure, "neural structure")
        _keys(
            structure,
            {
                "kind",
                "parent_graph_sha256",
                "active_graph_sha256",
                "blueprint_sha256",
                "manifest_sha256",
                "port_bundle_sha256",
                "port_spec_sha256",
            },
            "neural structure",
        )
        if structure["kind"] != STRUCTURAL_KIND:
            raise ValueError("unsupported neural structural kind")
        for field in structure:
            if field != "kind":
                _sha(structure[field], f"structure {field}")
        if structure["parent_graph_sha256"] != source["canonical_graph_sha256"]:
            raise ValueError("structural variant parent differs from canonical ancestor")
        for field in ("active_graph_sha256", "port_bundle_sha256", "port_spec_sha256"):
            if structure[field] != source[field]:
                raise ValueError(f"structural source {field} differs")

    @staticmethod
    def _validate_loci(value: Any, recipe: NeuralVariantRecipe) -> None:
        loci = _mapping(value, "neural loci")
        _keys(loci, {"neuron_groups", "input_groups", "readout_groups"}, "neural loci")
        bounds = recipe.document["bounds"]["parameters"]
        for kind in ("neuron", "input", "readout"):
            recipe_groups = {group["name"]: group for group in recipe.document[f"{kind}_groups"]}
            values = _mapping(loci[f"{kind}_groups"], f"{kind} loci")
            if set(values) != set(recipe_groups):
                raise ValueError(f"{kind} genotype groups differ from recipe")
            for name, group_values in values.items():
                group_values = _mapping(group_values, f"{kind} group {name}")
                expected = set(recipe_groups[name]["parameters"])
                if set(group_values) != expected:
                    raise ValueError(f"{kind} group {name} parameters differ")
                for parameter, value in group_values.items():
                    low, high = bounds[parameter]
                    _number(value, f"{name}.{parameter}", float(low), float(high))

    def to_value(self) -> dict[str, Any]:
        return copy.deepcopy(self._value)

    @property
    def structure(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._value["structure"])


@dataclass(frozen=True)
class NeuralPhenotype:
    genotype_sha256: str
    recipe_sha256: str
    canonical_graph_sha256: str
    active_graph_sha256: str
    port_bundle_sha256: str
    port_spec_sha256: str
    base_controller_sha256: str
    compatibility_group: str
    structural_blueprint_sha256: str | None
    input_names: tuple[str, ...]
    readout_names: tuple[str, ...]
    input_gains: np.ndarray
    readout_gains: np.ndarray
    excitability_gains: np.ndarray
    recurrent_source_gains: np.ndarray
    recurrent_target_gains: np.ndarray
    learning_rate_gains: np.ndarray
    modulator_gains: np.ndarray
    receipt: dict[str, Any]
    sha256: str

    def __post_init__(self) -> None:
        for field in (
            "input_gains",
            "readout_gains",
            "excitability_gains",
            "recurrent_source_gains",
            "recurrent_target_gains",
            "learning_rate_gains",
            "modulator_gains",
        ):
            object.__setattr__(self, field, _readonly(getattr(self, field)))

    def runtime_arrays(self) -> dict[str, np.ndarray]:
        return {
            "input_gain": self.input_gains,
            "readout_gain": self.readout_gains,
            "excitability_gain": self.excitability_gains,
            "recurrent_source_gain": self.recurrent_source_gains,
            "recurrent_target_gain": self.recurrent_target_gains,
            "learning_rate_gain": self.learning_rate_gains,
            "modulator_gain": self.modulator_gains,
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        """Atomically persist a single-resident phenotype without pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": PHENOTYPE_FORMAT,
            "phenotype_sha256": self.sha256,
            "genotype_sha256": self.genotype_sha256,
            "recipe_sha256": self.recipe_sha256,
            "canonical_graph_sha256": self.canonical_graph_sha256,
            "active_graph_sha256": self.active_graph_sha256,
            "port_bundle_sha256": self.port_bundle_sha256,
            "port_spec_sha256": self.port_spec_sha256,
            "base_controller_sha256": self.base_controller_sha256,
            "compatibility_group": self.compatibility_group,
            "structural_blueprint_sha256": self.structural_blueprint_sha256,
            "input_names": list(self.input_names),
            "readout_names": list(self.readout_names),
            "receipt": self.receipt,
        }
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(_canonical(metadata).decode()),
                **self.runtime_arrays(),
            )
        os.replace(temporary, path)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": _file_hash(path)}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        graph: Any,
        ports: NeuralPortBundle,
        recipe: NeuralVariantRecipe,
        port_bundle_sha256: str,
        base_controller_sha256: str,
        expected_file_sha256: str | None = None,
    ) -> "NeuralPhenotype":
        path = Path(path)
        if expected_file_sha256 is not None and _file_hash(path) != _sha(
            expected_file_sha256, "phenotype file"
        ):
            raise ValueError("neural phenotype file checksum differs")
        with np.load(path, allow_pickle=False) as value:
            if set(value.files) != {"metadata", *NEURAL_VARIANT_ARRAYS}:
                raise ValueError("neural phenotype artifact arrays differ")
            metadata = json.loads(str(value["metadata"]))
            arrays = {name: np.asarray(value[name]) for name in NEURAL_VARIANT_ARRAYS}
        if (
            metadata.get("format") != PHENOTYPE_FORMAT
            or metadata.get("recipe_sha256") != recipe.sha256
            or metadata.get("canonical_graph_sha256")
            != recipe.document["source"]["canonical_graph_sha256"]
            or metadata.get("active_graph_sha256") != graph.hash
            or metadata.get("port_spec_sha256") != ports.spec_hash
            or metadata.get("port_bundle_sha256") != port_bundle_sha256
            or metadata.get("base_controller_sha256") != base_controller_sha256
            or metadata.get("input_names") != list(ports.input_names)
            or metadata.get("readout_names") != list(ports.readout_names)
        ):
            raise ValueError("neural phenotype artifact identity differs")
        expected_shapes = {
            "input_gain": (351,),
            "readout_gain": (384,),
            **{name: (graph.n,) for name in NEURAL_VARIANT_ARRAYS[2:]},
        }
        for name, array in arrays.items():
            if (
                array.dtype != np.float32
                or array.shape != expected_shapes[name]
                or not array.flags.c_contiguous
                or not np.isfinite(array).all()
                or np.any((array < 0.05) | (array > 4.0))
            ):
                raise ValueError(f"neural phenotype array {name} differs")
        receipt = _mapping(metadata.get("receipt"), "phenotype receipt")
        observed_hashes = {name: _hash_array(array) for name, array in arrays.items()}
        if (
            receipt.get("array_sha256") != observed_hashes
            or receipt.get("compatibility_group") != metadata.get("compatibility_group")
        ):
            raise ValueError("neural phenotype array receipts differ")
        phenotype_sha = hashlib.sha256(_canonical(receipt)).hexdigest()
        if phenotype_sha != metadata.get("phenotype_sha256"):
            raise ValueError("neural phenotype receipt checksum differs")
        return cls(
            genotype_sha256=_sha(metadata.get("genotype_sha256"), "phenotype genotype"),
            recipe_sha256=recipe.sha256,
            canonical_graph_sha256=_sha(
                metadata.get("canonical_graph_sha256"), "phenotype canonical graph"
            ),
            active_graph_sha256=graph.hash,
            port_bundle_sha256=port_bundle_sha256,
            port_spec_sha256=ports.spec_hash,
            base_controller_sha256=base_controller_sha256,
            compatibility_group=_sha(
                metadata.get("compatibility_group"), "phenotype compatibility group"
            ),
            structural_blueprint_sha256=(
                None
                if metadata.get("structural_blueprint_sha256") is None
                else _sha(
                    metadata.get("structural_blueprint_sha256"),
                    "phenotype structural blueprint",
                )
            ),
            input_names=tuple(ports.input_names),
            readout_names=tuple(ports.readout_names),
            input_gains=arrays["input_gain"],
            readout_gains=arrays["readout_gain"],
            excitability_gains=arrays["excitability_gain"],
            recurrent_source_gains=arrays["recurrent_source_gain"],
            recurrent_target_gains=arrays["recurrent_target_gain"],
            learning_rate_gains=arrays["learning_rate_gain"],
            modulator_gains=arrays["modulator_gain"],
            receipt=receipt,
            sha256=phenotype_sha,
        )


def batch_neural_phenotypes(
    phenotypes: Sequence[NeuralPhenotype],
) -> tuple[dict[str, np.ndarray], list[str], str]:
    """Pack immutable phenotype vectors into the circuit's `[value,resident]` layout."""
    if not phenotypes:
        raise ValueError("at least one neural phenotype is required")
    compatibility = phenotypes[0].compatibility_group
    if any(phenotype.compatibility_group != compatibility for phenotype in phenotypes):
        raise ValueError("neural phenotypes require separate graph-compatible cohorts")
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "input_gain",
        "readout_gain",
        "excitability_gain",
        "recurrent_source_gain",
        "recurrent_target_gain",
        "learning_rate_gain",
        "modulator_gain",
    ):
        values = [phenotype.runtime_arrays()[name] for phenotype in phenotypes]
        arrays[name] = np.ascontiguousarray(np.stack(values, axis=1), dtype=np.float32)
    return arrays, [phenotype.sha256 for phenotype in phenotypes], compatibility


def compile_population_phenotypes(
    candidates: Sequence[Any],
    recipe: NeuralVariantRecipe,
    graph: Any,
    ports: NeuralPortBundle,
    port_bundle_sha256: str,
    base_controller_sha256: str,
) -> list[NeuralPhenotype]:
    """Compile canonical, mutation-free annotation gains for candidate births.

    ``CandidateGenome.neural_population_loci`` is the authoritative boundary.
    Repeated immutable candidate identities share one compiled object; a
    repeated identity with different birth inputs is rejected.
    """
    controller_sha = _sha(base_controller_sha256, "base controller")
    canonical = recipe.document["source"]
    if (
        graph.hash != canonical["canonical_graph_sha256"]
        or ports.graph_hash != graph.hash
        or ports.spec_hash != canonical["port_spec_sha256"]
        or _sha(port_bundle_sha256, "port bundle")
        != canonical["port_bundle_sha256"]
    ):
        raise ValueError(
            "population phenotype compiler requires the pinned canonical graph and ports"
        )

    compiled: dict[str, tuple[bytes, NeuralPhenotype]] = {}
    result: list[NeuralPhenotype] = []
    expected_birth_fields = {
        "candidate_sha256",
        "base_controller_sha256",
        "variation_seed",
        "neural_seed",
        "population_loci",
    }
    for candidate in candidates:
        candidate_sha = _sha(getattr(candidate, "sha256", None), "candidate")
        method = getattr(candidate, "neural_population_loci", None)
        if not callable(method):
            raise TypeError("candidate must expose neural_population_loci()")
        birth = _mapping(method(), "candidate neural population loci")
        _keys(birth, expected_birth_fields, "candidate neural population loci")
        if (
            birth["candidate_sha256"] != candidate_sha
            or birth["base_controller_sha256"] != controller_sha
        ):
            raise ValueError("candidate neural birth identity differs")
        variation_seed = birth["variation_seed"]
        neural_seed = birth["neural_seed"]
        if (
            isinstance(variation_seed, bool)
            or not isinstance(variation_seed, Integral)
            or not 0 <= variation_seed < 2**64
            or isinstance(neural_seed, bool)
            or not isinstance(neural_seed, Integral)
            or neural_seed != (int(variation_seed) ^ 0x4E455552414C5631)
        ):
            raise ValueError("candidate neural seed derivation differs")
        # Canonical bytes also make duplicate-candidate validation independent
        # of object identity or dictionary insertion order.
        birth_bytes = _canonical(birth)
        prior = compiled.get(candidate_sha)
        if prior is not None:
            if prior[0] != birth_bytes:
                raise ValueError("duplicate candidate identity has different neural loci")
            result.append(prior[1])
            continue
        genotype = breed_neural_genotype(
            recipe,
            name=f"candidate-{candidate_sha}-neural",
            seed=int(neural_seed),
            base_controller_sha256=controller_sha,
            population_loci=_mapping(birth["population_loci"], "population neural loci"),
            mutation_scale=0.0,
        )
        phenotype = compile_neural_phenotype(
            graph,
            ports,
            genotype,
            port_bundle_sha256=port_bundle_sha256,
            base_controller_sha256=controller_sha,
        )
        compiled[candidate_sha] = (birth_bytes, phenotype)
        result.append(phenotype)
    return result


def structural_identity(artifact_directory: str | Path) -> dict[str, Any]:
    """Return the exact genotype identity of a compiled CircuitBlueprint graph."""
    root = Path(artifact_directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    native_path = root / "native-csr-v2.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    native = json.loads(native_path.read_text(encoding="utf-8"))
    derivation = _mapping(manifest.get("derivation"), "derived graph receipt")
    source = _mapping(manifest.get("source_graph"), "derived graph source")
    ports = _mapping(manifest.get("derived_artifacts"), "derived artifacts").get("ports.npz")
    ports = _mapping(ports, "derived port artifact")
    result = {
        "kind": STRUCTURAL_KIND,
        "parent_graph_sha256": _sha(source.get("dataset_hash"), "structural parent graph"),
        "active_graph_sha256": _sha(manifest.get("dataset_hash"), "structural active graph"),
        "blueprint_sha256": _sha(derivation.get("blueprint_sha256"), "structural blueprint"),
        "manifest_sha256": _file_hash(manifest_path),
        "port_bundle_sha256": _sha(ports.get("sha256"), "structural port bundle"),
        "port_spec_sha256": _sha(native.get("port_spec_sha256"), "structural port spec"),
    }
    if native.get("graph_sha256") != result["active_graph_sha256"]:
        raise ValueError("native structural graph receipt differs")
    return result


def _parent_value(
    parents: Sequence[NeuralVariantGenotype], kind: str, group: str, parameter: str,
    rng: np.random.Generator,
) -> float:
    if not parents:
        return 1.0
    parent = parents[int(rng.integers(0, len(parents)))]
    return float(parent._value["loci"][f"{kind}_groups"][group][parameter])


def breed_neural_genotype(
    recipe: NeuralVariantRecipe,
    *,
    name: str,
    seed: int,
    base_controller_sha256: str,
    population_loci: Mapping[str, float],
    parents: Sequence[NeuralVariantGenotype] = (),
    mutation_scale: float = 0.0,
    structure: Mapping[str, Any] | None = None,
) -> NeuralVariantGenotype:
    """Recombine zero, one, or two immutable parents and add bounded variation."""
    _name(name, "neural genotype name")
    _sha(base_controller_sha256, "base controller")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
        raise ValueError("neural variation seed must be uint64")
    if len(parents) > 2 or any(parent.recipe.sha256 != recipe.sha256 for parent in parents):
        raise ValueError("neural parents must share this recipe")
    population_loci = dict(population_loci)
    if set(population_loci) != set(POPULATION_LOCI):
        raise ValueError(f"population_loci must contain {list(POPULATION_LOCI)}")
    population_loci = {
        locus: _number(population_loci[locus], f"population neural {locus}", 0.70, 1.35)
        for locus in POPULATION_LOCI
    }
    scale = _number(mutation_scale, "mutation scale", 0.0, 1.0)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    sigma = float(recipe.document["bounds"]["mutation_log_sigma_max"]) * scale
    bounds = recipe.document["bounds"]["parameters"]
    loci: dict[str, dict[str, dict[str, float]]] = {}
    for kind in ("neuron", "input", "readout"):
        loci[f"{kind}_groups"] = {}
        for group in recipe.document[f"{kind}_groups"]:
            values: dict[str, float] = {}
            for parameter in group["parameters"]:
                inherited = _parent_value(parents, kind, group["name"], parameter, rng)
                low, high = map(float, bounds[parameter])
                mutated = inherited * math.exp(float(rng.normal(0.0, sigma)))
                values[parameter] = float(np.clip(mutated, low, high))
            loci[f"{kind}_groups"][group["name"]] = values
    canonical = recipe.document["source"]
    structure_value = copy.deepcopy(dict(structure)) if structure is not None else None
    active_graph = (
        structure_value["active_graph_sha256"] if structure_value else canonical["canonical_graph_sha256"]
    )
    port_bundle = (
        structure_value["port_bundle_sha256"] if structure_value else canonical["port_bundle_sha256"]
    )
    port_spec = structure_value["port_spec_sha256"] if structure_value else canonical["port_spec_sha256"]
    generation = 0 if not parents else max(parent._value["generation"] for parent in parents) + 1
    return NeuralVariantGenotype(
        {
            "format": GENOTYPE_FORMAT,
            "name": name,
            "generation": generation,
            "parents": [parent.sha256 for parent in parents],
            "variation": {
                "recipe_sha256": recipe.sha256,
                "seed": int(seed),
                "mutation_scale": scale,
                "population_loci": population_loci,
            },
            "sources": {
                "canonical_graph_sha256": canonical["canonical_graph_sha256"],
                "active_graph_sha256": active_graph,
                "port_bundle_sha256": port_bundle,
                "port_spec_sha256": port_spec,
                "base_controller_sha256": base_controller_sha256,
            },
            "loci": loci,
            "structure": structure_value,
            "sha256": None,
        },
        recipe,
    )


def _neuron_indices(graph: Any, selectors: Sequence[Mapping[str, Any]]) -> np.ndarray:
    parts = [graph.select(**selector) for selector in selectors]
    return np.unique(np.concatenate(parts)).astype(np.int32, copy=False)


def _port_indices(records: Sequence[Mapping[str, Any]], match: Mapping[str, Any]) -> np.ndarray:
    indices = [
        index
        for index, record in enumerate(records)
        if all(record.get(field) == expected for field, expected in match.items())
    ]
    return np.asarray(indices, dtype=np.int32)


def _apply_group_gains(
    target: np.ndarray,
    indices: np.ndarray,
    value: float,
) -> None:
    target[indices] *= value


def _categorical_receipt(graph: Any, indices: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(len(indices)),
        "indices_sha256": _hash_array(indices.astype("<i4", copy=False)),
        "body_ids_sha256": _hash_array(np.asarray(graph.body_ids[indices], dtype="<i8")),
        "classes": dict(sorted(Counter(str(value) or "unavailable" for value in graph.classes[indices]).items())),
        "superclasses": dict(
            sorted(Counter(str(value) or "unavailable" for value in graph.superclasses[indices]).items())
        ),
        "sides": dict(sorted(Counter(str(value) or "unavailable" for value in graph.sides[indices]).items())),
    }


def compile_neural_phenotype(
    graph: Any,
    ports: NeuralPortBundle,
    genotype: NeuralVariantGenotype,
    *,
    port_bundle_sha256: str,
    base_controller_sha256: str,
) -> NeuralPhenotype:
    """Compile immutable candidate arrays without changing the sparse graph."""
    source = genotype._value["sources"]
    if graph.hash != source["active_graph_sha256"] or ports.graph_hash != graph.hash:
        raise ValueError("active graph identity differs from neural genotype")
    if ports.spec_hash != source["port_spec_sha256"]:
        raise ValueError("active port spec differs from neural genotype")
    if _sha(port_bundle_sha256, "port bundle") != source["port_bundle_sha256"]:
        raise ValueError("active port bundle differs from neural genotype")
    if _sha(base_controller_sha256, "base controller") != source["base_controller_sha256"]:
        raise ValueError("base controller differs from neural genotype")
    if len(ports.input_names) != 351 or len(ports.readout_names) != 384:
        raise ValueError("neural phenotype requires current 351/384 ports")
    structure = genotype._value["structure"]
    if structure is not None:
        derivation = graph.manifest.get("derivation", {})
        manifest_path = Path(graph.path) / "manifest.json"
        if (
            graph.manifest.get("format") != "chreatures-derived-circuit-v1"
            or derivation.get("blueprint_sha256") != structure["blueprint_sha256"]
            or derivation.get("blueprint", {}).get("parent", {}).get("graph_sha256")
            != structure["parent_graph_sha256"]
            or _file_hash(manifest_path) != structure["manifest_sha256"]
        ):
            raise ValueError("compiled structural graph receipt differs")

    population_loci = genotype._value["variation"]["population_loci"]
    recurrent_root = math.sqrt(population_loci["recurrent_gain"])
    arrays = {
        "input_gain": np.full(351, population_loci["input_gain"], dtype=np.float64),
        "readout_gain": np.full(384, population_loci["readout_gain"], dtype=np.float64),
        "excitability_gain": np.full(
            graph.n, population_loci["excitability"], dtype=np.float64
        ),
        "recurrent_source_gain": np.full(graph.n, recurrent_root, dtype=np.float64),
        "recurrent_target_gain": np.full(graph.n, recurrent_root, dtype=np.float64),
        "learning_rate_gain": np.full(
            graph.n, population_loci["learning_rate_gain"], dtype=np.float64
        ),
        "modulator_gain": np.full(
            graph.n, population_loci["modulator_gain"], dtype=np.float64
        ),
    }
    group_receipts: dict[str, list[dict[str, Any]]] = {
        "neuron_groups": [],
        "input_groups": [],
        "readout_groups": [],
    }
    recipe = genotype.recipe.document
    loci = genotype._value["loci"]
    for group in recipe["neuron_groups"]:
        indices = _neuron_indices(graph, group["selectors"])
        if not len(indices):
            raise ValueError(f"neural group {group['name']!r} is empty on active graph")
        for parameter, value in loci["neuron_groups"][group["name"]].items():
            _apply_group_gains(arrays[parameter], indices, value)
        group_receipts["neuron_groups"].append(
            {"name": group["name"], "selectors": group["selectors"], **_categorical_receipt(graph, indices)}
        )
    for kind, names, records, parameter in (
        ("input", ports.input_names, ports.input_ports, "input_gain"),
        ("readout", ports.readout_names, ports.readout_ports, "readout_gain"),
    ):
        for group in recipe[f"{kind}_groups"]:
            indices = _port_indices(records, group["match"])
            if not len(indices):
                raise ValueError(f"{kind} group {group['name']!r} is empty")
            value = loci[f"{kind}_groups"][group["name"]][parameter]
            _apply_group_gains(arrays[parameter], indices, value)
            group_receipts[f"{kind}_groups"].append(
                {
                    "name": group["name"],
                    "match": group["match"],
                    "count": int(len(indices)),
                    "indices_sha256": _hash_array(indices.astype("<i4", copy=False)),
                    "names_sha256": hashlib.sha256(
                        _canonical([names[index] for index in indices])
                    ).hexdigest(),
                }
            )
    for parameter, values in arrays.items():
        low, high = map(float, recipe["bounds"]["combined"][parameter])
        np.clip(values, low, high, out=values)
        arrays[parameter] = _readonly(values)

    structural_blueprint = structure["blueprint_sha256"] if structure else None
    compatibility_record = {
        "active_graph_sha256": graph.hash,
        "port_bundle_sha256": source["port_bundle_sha256"],
        "port_spec_sha256": ports.spec_hash,
        "base_controller_sha256": base_controller_sha256,
        "structural_blueprint_sha256": structural_blueprint,
        "runtime_contract": PHENOTYPE_FORMAT,
    }
    compatibility_group = hashlib.sha256(_canonical(compatibility_record)).hexdigest()
    array_hashes = {name: _hash_array(value) for name, value in arrays.items()}
    receipt = {
        "format": PHENOTYPE_FORMAT,
        "genotype_sha256": genotype.sha256,
        "recipe_sha256": genotype.recipe.sha256,
        "compatibility": compatibility_record,
        "compatibility_group": compatibility_group,
        "shape": {"neurons": graph.n, "inputs": 351, "readouts": 384},
        "groups": group_receipts,
        "array_sha256": array_hashes,
        "population_loci": population_loci,
        "equations": {
            "input": "x_prime[c,b] = input_gain[c,b] * x[c,b]",
            "recurrence": "q[i,b] = recurrent_target_gain[i,b] * sum_j(W[i,j] * recurrent_source_gain[j,b] * rate[j,b])",
            "activation": "a[i,b] = 0.005 + excitability_gain[i,b] * (drive[i,b] + base_gain*q[i,b]) - 0.10*adaptation[i,b]",
            "readout": "y_prime[o,b] = readout_gain[o,b] * sum_i(R[o,i] * rate[i,b])",
            "learning": "declared_local_delta[i,b] *= learning_rate_gain[i,b] only where an explicit plastic rule owns that delta",
            "modulator": "declared_modulator[i,b] *= modulator_gain[i,b] only for an explicit synthetic or measured path",
        },
        "private_state_excluded": sorted(_PRIVATE_KEYS),
    }
    phenotype_sha = hashlib.sha256(_canonical(receipt)).hexdigest()
    return NeuralPhenotype(
        genotype_sha256=genotype.sha256,
        recipe_sha256=genotype.recipe.sha256,
        canonical_graph_sha256=source["canonical_graph_sha256"],
        active_graph_sha256=graph.hash,
        port_bundle_sha256=source["port_bundle_sha256"],
        port_spec_sha256=ports.spec_hash,
        base_controller_sha256=base_controller_sha256,
        compatibility_group=compatibility_group,
        structural_blueprint_sha256=structural_blueprint,
        input_names=tuple(ports.input_names),
        readout_names=tuple(ports.readout_names),
        input_gains=arrays["input_gain"],
        readout_gains=arrays["readout_gain"],
        excitability_gains=arrays["excitability_gain"],
        recurrent_source_gains=arrays["recurrent_source_gain"],
        recurrent_target_gains=arrays["recurrent_target_gain"],
        learning_rate_gains=arrays["learning_rate_gain"],
        modulator_gains=arrays["modulator_gain"],
        receipt=receipt,
        sha256=phenotype_sha,
    )
