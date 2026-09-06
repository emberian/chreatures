"""Thin artifact and process boundary for the native population search engine."""
from __future__ import annotations

import hashlib
import json
import math
import os
import copy
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

FORMAT = "chreatures-population-search-v1"
GENOME_FORMAT = "chreatures-population-genome-v1"
ACTION_ORDER = (
    "thrust", "yaw", "gaze_pitch", "posture", "grip", "signal_low",
    "signal_mid", "signal_high", "eat", "release", "secrete", "allocate",
)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "native" / "population-core" / "Cargo.toml"
PRIVATE_WORDS = {"state", "memory", "optimizer", "rng", "rates", "support", "pools", "atp", "context", "history"}


def current_parameter_recipe(
    *, policy_adapter_count: int, heritable_policy_adapter_rows: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Return bounded v4 loci and a neutral founder; no lifetime values."""
    from .body_genome import TRAIT_BOUNDS

    specs: list[dict[str, Any]] = []
    founder: dict[str, float] = {}

    if isinstance(policy_adapter_count, bool) or not 1 <= policy_adapter_count <= 4096:
        raise ValueError("policy_adapter_count must be in 1..4096")

    def add(name: str, group: str, low: float, high: float, value: float, sigma: float) -> None:
        specs.append({
            "name": name, "group": group, "low": low, "high": high,
            "mutation_sigma": sigma,
            "integer": False,
        })
        founder[name] = value

    for name, (low, high) in TRAIT_BOUNDS.items():
        add(f"body.{name}", "body", low, high, 1.0, 0.06 * (high - low))
    somatic = (
        "gut_capacity", "reserve_capacity", "maintenance_rate", "activation_rate",
        "absorption_rate", "digestive_atp_rate", "bite_rate", "maximum_bite",
        "mouth_radius", "fatigue_rise", "fatigue_recovery", "structural_capacity",
        "gland_capacity", "brood_capacity", "secretion_rate", "release_rate",
        "gland_synthesis_rate", "allocation_rate", "brood_maturation_rate", "brood_material_target",
        "brood_energy_target", "release_radius", "exchange_load_decay_rate",
        "eat_activity_cost", "secrete_activity_cost", "allocate_activity_cost",
        "phototrophic_absorptivity", "dorsal_capture_fraction",
    )
    for name in somatic:
        add(f"somatic.{name}_gain", "somatic", 0.65, 1.45, 1.0, 0.045)
    for index, value in enumerate((0.34, 0.33, 0.33)):
        add(
            f"somatic.secretion_profile.{index}",
            "simplex:secretion_profile", 0.0, 1.0, value, 0.04,
        )
    for name, value in (("photosynthesis", 0.40), ("digestion", 0.60)):
        add(f"metabolic.allocation.{name}", "simplex:metabolic_allocation", 0.0, 1.0, value, 0.04)
    for name, value in (("structure", 0.45), ("gland", 0.20), ("brood", 0.35)):
        add(f"developmental.allocation.{name}", "simplex:developmental_allocation", 0.0, 1.0, value, 0.04)
    for name in (
        "input_gain", "readout_gain", "excitability", "recurrent_gain",
        "learning_rate_gain", "modulator_gain",
    ):
        add(f"neural.{name}", "neural", 0.70, 1.35, 1.0, 0.04)
    add("controller.recurrent_gain", "controller", 0.80, 1.20, 1.0, 0.025)
    add("controller.learning_rate_gain", "controller", 0.50, 1.50, 1.0, 0.06)
    for channel in ACTION_ORDER:
        add(f"controller.action_gain.{channel}", "controller", 0.75, 1.25, 1.0, 0.03)
        add(
            f"controller.action_logit_temperature_offset.{channel}",
            "controller", -0.50, 0.50, 0.0, 0.06,
        )
    specs.append({
        "name": "controller.policy_adapter_index", "group": "controller",
        "low": 0.0,
        "high": float(policy_adapter_count - 1 if heritable_policy_adapter_rows else 0),
        "mutation_sigma": 1.0, "integer": True,
    })
    founder["controller.policy_adapter_index"] = 0.0
    return specs, founder


def _serde_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("nonfinite values are not canonical JSON")
    encoded = repr(value)
    absolute = abs(value)
    if "e" in encoded and 1e-5 <= absolute < 1e16:
        mantissa, exponent_text = encoded.split("e")
        exponent = int(exponent_text)
        sign = ""
        if mantissa.startswith("-"):
            sign, mantissa = "-", mantissa[1:]
        whole, _, fraction = mantissa.partition(".")
        digits = whole + fraction
        point = len(whole) + exponent
        if point <= 0:
            return sign + "0." + "0" * (-point) + digits
        if point >= len(digits):
            return sign + digits + "0" * (point - len(digits)) + ".0"
        return sign + digits[:point] + "." + digits[point:]
    if "e" in encoded:
        mantissa, exponent = encoded.split("e")
        encoded = f"{mantissa}e{int(exponent):+d}"
    return encoded


class _SerdeCanonicalEncoder(json.JSONEncoder):
    def iterencode(self, value: Any, _one_shot: bool = False) -> Any:
        markers = {} if self.check_circular else None
        iterator = json.encoder._make_iterencode(
            markers,
            self.default,
            json.encoder.encode_basestring,
            self.indent,
            _serde_float,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return iterator(value, 0)


def canonical_bytes(value: Any) -> bytes:
    return _SerdeCanonicalEncoder(
        sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    ).encode(value).encode()


def content_sha256(value: Mapping[str, Any], *, identity_field: str = "sha256") -> str:
    body = dict(value)
    body[identity_field] = ""
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def _digest(value: str, name: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _run(arguments: Sequence[str]) -> str:
    executable = os.environ.get("CHREATURES_POPULATION_CORE")
    if executable:
        command = [executable, *arguments]
    else:
        command = [
            "cargo", "run", "--quiet", "--manifest-path", str(DEFAULT_MANIFEST),
            "--", *arguments,
        ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "native population command failed")
    return result.stdout.strip()


class CandidateGenome:
    """Immutable public candidate loci; never a resident's lifetime state."""

    def __init__(self, value: Mapping[str, Any]):
        self._value = json.loads(json.dumps(value))
        if self._value.get("format") != GENOME_FORMAT:
            raise ValueError("candidate genome format differs")
        _digest(self._value.get("sha256", ""), "candidate identity")
        if content_sha256(self._value) != self._value["sha256"]:
            raise ValueError("candidate genome content hash differs")
        values = self._value.get("values")
        if not isinstance(values, dict) or not values:
            raise ValueError("candidate values are absent")
        for name, value in values.items():
            words = str(name).lower().replace("/", ".").replace("_", ".").split(".")
            if PRIVATE_WORDS.intersection(words) and not str(name).endswith("_gain"):
                raise ValueError(f"private lifetime field in candidate: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"candidate locus {name} is nonfinite")

    @property
    def sha256(self) -> str:
        return self._value["sha256"]

    def to_value(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._value))

    def loci(self, group: str) -> dict[str, float]:
        prefix = group + "."
        return {
            key[len(prefix):]: float(value)
            for key, value in self._value["values"].items() if key.startswith(prefix)
        }

    def controller_adapter(self) -> dict[str, Any]:
        loci = self.loci("controller")
        adapter_index = int(loci["policy_adapter_index"])
        if adapter_index != loci["policy_adapter_index"] or not 0 <= adapter_index < int(
            self._value["policy_adapter_count"]
        ):
            raise ValueError("candidate policy adapter index differs")
        gains = [loci[f"action_gain.{name}"] for name in ACTION_ORDER]
        temperatures = [
            loci[f"action_logit_temperature_offset.{name}"] for name in ACTION_ORDER
        ]
        body = {
            "policy_adapter_index": adapter_index,
            "population_adapter_bank_sha256": self._value["population_adapter_bank_sha256"],
            "policy_adapter_count": int(self._value["policy_adapter_count"]),
            "policy_adapter_rank": int(self._value["policy_adapter_rank"]),
            "organism_interface_sha256": self._value["organism_interface_sha256"],
            "recurrent_gain": loci["recurrent_gain"],
            "learning_rate_gain": loci["learning_rate_gain"],
            "action_gain": gains,
            "action_logit_temperature_offset": temperatures,
        }
        return {
            "candidate_sha256": self.sha256,
            "loci_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
            **body,
        }

    def neural_population_loci(self) -> dict[str, Any]:
        """Inputs for breed_neural_genotype; compiled arrays live elsewhere."""
        loci = self.loci("neural")
        expected = {
            "input_gain", "readout_gain", "excitability", "recurrent_gain",
            "learning_rate_gain", "modulator_gain",
        }
        if set(loci) != expected:
            raise ValueError("candidate neural population loci differ")
        return {
            "candidate_sha256": self.sha256,
            "base_controller_sha256": self._value["base_controller_sha256"],
            "variation_seed": int(self._value["variation"]["seed"]),
            "neural_seed": int(self._value["variation"]["seed"]) ^ 0x4E455552414C5631,
            "population_loci": loci,
        }


class PopulationSearch:
    """Durable ask/tell interface; mutation, selection and archive stay native."""

    def __init__(self, state_path: str | Path):
        self.path = Path(state_path).resolve()

    @classmethod
    def initialize(
        cls, state_path: str | Path, config: Mapping[str, Any], *, seed: int,
    ) -> "PopulationSearch":
        path = Path(state_path).resolve()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as stream:
            stream.write(canonical_bytes(config))
            config_path = Path(stream.name)
        try:
            _run(("init", str(config_path), str(path), str(seed)))
        finally:
            config_path.unlink(missing_ok=True)
        return cls(path)

    def validate(self) -> None:
        _run(("validate", str(self.path)))

    def register_environment(self, value: Mapping[str, Any]) -> None:
        body = dict(value)
        body["sha256"] = body.get("sha256") or content_sha256(body)
        _digest(body["sha256"], "environment identity")
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=self.path.parent) as stream:
            stream.write(canonical_bytes(body))
            temporary = Path(stream.name)
        try:
            _run(("register-environment", str(self.path), str(temporary)))
        finally:
            temporary.unlink(missing_ok=True)

    def ask(self, count: int) -> list[dict[str, Any]]:
        values = json.loads(_run(("ask", str(self.path), str(count))))
        for assignment in values:
            assignment["candidate"] = CandidateGenome(assignment["candidate"])
        return values

    def ask_transfers(self, count: int) -> list[dict[str, Any]]:
        values = json.loads(_run(("ask-transfers", str(self.path), str(count))))
        for assignment in values:
            assignment["candidate"] = CandidateGenome(assignment["candidate"])
        return values

    def tell(self, result: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(result)
        required = {"life_id", "evaluation_seed", "committed_ticks", "trajectory_sha256"}
        if not required.issubset(result):
            raise ValueError(f"evaluation lacks identity fields: {sorted(required - set(result))}")
        result["evaluation_sha256"] = result.get("evaluation_sha256") or content_sha256(
            result, identity_field="evaluation_sha256"
        )
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=self.path.parent) as stream:
            stream.write(canonical_bytes(result))
            temporary = Path(stream.name)
        try:
            return json.loads(_run(("tell", str(self.path), str(temporary))))
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot_value(self) -> dict[str, Any]:
        value = json.loads(self.path.read_text())
        if value.get("format") != FORMAT:
            raise ValueError("population state format differs")
        return value


def compose_population_birth(
    habitat: Mapping[str, Any],
    biosphere: Mapping[str, Any],
    candidates: Sequence[CandidateGenome | Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile immutable candidates into fresh physical founder rows, once."""
    habitat_value = copy.deepcopy(dict(habitat))
    biosphere_value = copy.deepcopy(dict(biosphere))
    bodies = habitat_value.get("bodies")
    mobiles = biosphere_value.get("mobiles")
    compartments = biosphere_value.get("compartments")
    if not isinstance(bodies, list) or not isinstance(mobiles, list) or not isinstance(compartments, list):
        raise ValueError("population birth requires habitat bodies and biosphere mobile rows")
    genomes = [item if isinstance(item, CandidateGenome) else CandidateGenome(item) for item in candidates]
    if len(genomes) != len(bodies) or len(mobiles) != len(bodies):
        raise ValueError("candidate, physical body, and mobile counts differ")
    mobile_by_id = {item.get("id"): item for item in mobiles if isinstance(item, dict)}
    if len(mobile_by_id) != len(mobiles):
        raise ValueError("mobile founder identities differ")
    receipts = []
    rate_fields = (
        "gut_capacity", "reserve_capacity", "maintenance_rate", "activation_rate",
        "absorption_rate", "digestive_atp_rate", "bite_rate", "maximum_bite",
        "mouth_radius", "fatigue_rise", "fatigue_recovery", "structural_capacity",
        "gland_capacity", "brood_capacity", "secretion_rate", "release_rate",
        "gland_synthesis_rate", "allocation_rate", "brood_maturation_rate", "brood_material_target",
        "brood_energy_target", "release_radius", "exchange_load_decay_rate",
        "eat_activity_cost", "secrete_activity_cost", "allocate_activity_cost",
    )
    for body, candidate in zip(bodies, genomes, strict=True):
        if not isinstance(body, dict) or body.get("id") not in mobile_by_id:
            raise ValueError("candidate body lacks its private mobile founder")
        mobile = mobile_by_id[body["id"]]
        somatic = candidate.loci("somatic")
        body["articulated_traits"] = candidate.loci("body")
        traits = body.get("metabolic_traits")
        if not isinstance(traits, dict):
            raise ValueError("body lacks metabolic_traits")
        for field in ("phototrophic_absorptivity", "dorsal_capture_fraction"):
            gain = somatic[f"{field}_gain"]
            traits[field] = float(traits[field]) * gain
        for field in rate_fields:
            if field not in mobile:
                raise ValueError(f"mobile founder lacks current field {field}")
            mobile[field] = float(mobile[field]) * somatic[f"{field}_gain"]
        allocation = candidate.loci("developmental")
        mobile["allocation_weights"] = {
            name: allocation[f"allocation.{name}"]
            for name in ("structure", "gland", "brood")
        }
        secretion = somatic.get("secretion_profile.0")
        if secretion is None:
            raise ValueError("candidate lacks secretion profile simplex")
        mobile["secretion_profile"] = [somatic[f"secretion_profile.{index}"] for index in range(3)]
        metabolism = candidate.loci("metabolic")
        for row_name, fraction, neutral, enzyme_rule in (
            ("body_row", metabolism["allocation.photosynthesis"], 0.4, lambda name: name == "carbon_fixation"),
            ("gut_row", metabolism["allocation.digestion"], 0.6, lambda name: name.endswith("_digestion")),
        ):
            row = mobile.get(row_name)
            if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(compartments):
                raise ValueError(f"mobile {row_name} is invalid")
            enzymes = compartments[row].get("enzymes")
            if not isinstance(enzymes, dict):
                raise ValueError("private founder enzymes differ")
            for name in tuple(enzymes):
                if enzyme_rule(name):
                    enzymes[name] = float(enzymes[name]) * fraction / neutral
        physical = {
            "body_id": body["id"],
            "body_traits": body["articulated_traits"],
            "somatic": {key: mobile[key] for key in (*rate_fields, "allocation_weights", "secretion_profile")},
            "metabolic_traits": traits,
            "metabolic_allocation": metabolism,
        }
        receipts.append({
            "candidate_sha256": candidate.sha256,
            "physical_phenotype_sha256": hashlib.sha256(canonical_bytes(physical)).hexdigest(),
            "controller_adapter": candidate.controller_adapter(),
            "neural_population_loci": candidate.neural_population_loci(),
        })
    receipt_body = {
        "format": "chreatures-population-cold-birth-v1",
        "sha256": "",
        "base_habitat_sha256": hashlib.sha256(canonical_bytes(habitat)).hexdigest(),
        "base_biosphere_sha256": hashlib.sha256(canonical_bytes(biosphere)).hexdigest(),
        "candidate_order": [candidate.sha256 for candidate in genomes],
        "phenotypes": receipts,
        "semantics": "fresh private rows and runtime state; immutable inherited loci only",
    }
    receipt = receipt_body | {"sha256": hashlib.sha256(canonical_bytes(receipt_body)).hexdigest()}
    return habitat_value, biosphere_value, receipt


__all__ = [
    "ACTION_ORDER", "CandidateGenome", "PopulationSearch", "content_sha256",
    "current_parameter_recipe",
    "compose_population_birth",
]
