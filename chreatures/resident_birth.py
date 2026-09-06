"""Authenticated cold birth manifest for a heterogeneous resident cohort.

This file references inherited artifacts only. Runtime continuations use whole
world checkpoints and never reinterpret this manifest as an adult save state.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .organism_interface import MAX_RESIDENTS
from .population import CandidateGenome

FORMAT = "chreatures-resident-birth-manifest-v1"
PHENOTYPE_FIELDS = {
    "artifact_path", "artifact_sha256", "phenotype_sha256", "graph_sha256",
    "port_spec_sha256", "port_bundle_sha256",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"format", "residents"} or value["format"] != FORMAT:
        raise ValueError("resident birth manifest format differs")
    rows = value["residents"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_RESIDENTS:
        raise ValueError("a resident birth cohort contains 1..32 founders")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"candidate", "neural_phenotype"}:
            raise ValueError("a birth row requires its genome and neural phenotype")
        candidate = CandidateGenome(row["candidate"])
        phenotype = row["neural_phenotype"]
        if not isinstance(phenotype, dict) or set(phenotype) != PHENOTYPE_FIELDS:
            raise ValueError("birth neural phenotype artifact identity differs")
        if not isinstance(phenotype["artifact_path"], str) or not phenotype["artifact_path"]:
            raise ValueError("birth neural phenotype requires a service-local artifact path")
        for key in PHENOTYPE_FIELDS - {"artifact_path"}:
            digest = phenotype[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"birth neural phenotype requires SHA-256: {key}")
        for key in ("graph_sha256", "port_spec_sha256"):
            if phenotype[key] != candidate.to_value()[key]:
                raise ValueError(f"birth neural phenotype and inherited genome differ: {key}")
    return copy.deepcopy(dict(value))


def load_manifest(path: str | Path) -> dict[str, Any]:
    return validate_manifest(json.loads(Path(path).read_text()))


def candidate_adapters(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = validate_manifest(value)
    return [CandidateGenome(row["candidate"]).controller_adapter() for row in manifest["residents"]]


def verify_controller(value: Mapping[str, Any], artifact: str | Path) -> None:
    actual = file_sha256(artifact)
    if any(row["candidate"]["base_controller_sha256"] != actual for row in value["residents"]):
        raise ValueError("birth genomes do not inherit this controller artifact")


def inherited_body_templates(habitat: Mapping[str, Any], biosphere: Mapping[str, Any]) -> dict[str, Any]:
    """Retain constitutive birth parameters without copying any adult chemistry."""
    mobiles = {row["id"]: row for row in biosphere["mobiles"]}
    exchange = {row["id"]: row for row in biosphere["exchange"]["mobiles"]}
    result = {}
    for body in habitat["bodies"]:
        mobile = mobiles[body["id"]]
        founders = {}
        for compartment in ("body", "gut", "structure", "gland", "brood"):
            source = biosphere["compartments"][mobile[f"{compartment}_row"]]
            founders[compartment] = {
                "enzymes": copy.deepcopy(source["enzymes"]),
                "pools": {}, "atp": 0.0, "atp_capacity": source["atp_capacity"],
            }
        result[body["id"]] = {
            "body": copy.deepcopy(body),
            "mobile": {key: copy.deepcopy(value) for key, value in mobile.items() if key != "id" and not key.endswith("_row")},
            "exchange": {key: copy.deepcopy(value) for key, value in exchange[body["id"]].items() if key != "id"},
            "founders": founders,
        }
    return result
