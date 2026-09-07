"""Authenticated cold birth boundary for a CNS-only resident cohort.

The candidate genome remains the source of physical body and metabolic
inheritance.  Neural adapter and controller weights are immutable shared
artifacts; old per-genome policy adapters and phenotype gain files are not part
of a current birth.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cns_adapter_contract import CONTROLLER_FORMAT
from .organism_interface import MAX_RESIDENTS
from .population import CandidateGenome
from .resident_contract import (
    NATIVE_EXECUTION,
    NATIVE_POPULATION_FORMAT,
    NATIVE_POPULATION_VERSION,
)
from .sequence_control import valid_sha256

FORMAT = "chreatures-cns-only-resident-birth-v2"
GENOME_SCOPE = "physical-body-and-metabolism-only"
CONTROLLER_FIELDS = {
    "artifact_sha256",
    "file_sha256",
    "controller_input",
    "cns_adapter_sha256",
    "genome_scope",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"format", "controller", "residents"}
        or value["format"] != FORMAT
    ):
        raise ValueError("CNS resident birth manifest format differs")
    controller = value["controller"]
    if not isinstance(controller, Mapping) or set(controller) != CONTROLLER_FIELDS:
        raise ValueError("birth controller identity fields differ")
    if (
        controller["controller_input"] != CONTROLLER_FORMAT
        or controller["genome_scope"] != GENOME_SCOPE
    ):
        raise ValueError("birth controller contract differs")
    for name in ("artifact_sha256", "file_sha256", "cns_adapter_sha256"):
        if not valid_sha256(controller[name]):
            raise ValueError(f"birth controller requires SHA-256: {name}")
    rows = value["residents"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_RESIDENTS:
        raise ValueError(f"a resident birth cohort contains 1..{MAX_RESIDENTS} founders")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"candidate", "cns_adapter_sha256"}:
            raise ValueError("a birth row requires its physical genome and CNS adapter")
        candidate = CandidateGenome(row["candidate"])
        if row["cns_adapter_sha256"] != controller["cns_adapter_sha256"]:
            raise ValueError("birth residents must use the authenticated shared CNS adapter")
        if candidate.to_value()["base_controller_sha256"] != controller["file_sha256"]:
            raise ValueError("birth genome base controller identity differs")
    return copy.deepcopy(dict(value))


def load_manifest(path: str | Path) -> dict[str, Any]:
    return validate_manifest(json.loads(Path(path).read_text()))


def controller_identity(artifact: str | Path) -> dict[str, str]:
    path = Path(artifact).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("resident controller artifact has no metadata")
        metadata = json.loads(str(archive["metadata"].item()))
    cns = metadata.get("cns_service", {})
    if (
        metadata.get("format") != NATIVE_POPULATION_FORMAT
        or metadata.get("version") != NATIVE_POPULATION_VERSION
        or metadata.get("execution") != NATIVE_EXECUTION
        or metadata.get("controller_input") != CONTROLLER_FORMAT
        or not valid_sha256(metadata.get("artifact_sha256"))
        or not valid_sha256(cns.get("adapter_sha256"))
    ):
        raise ValueError("resident controller artifact contract differs")
    return {
        "artifact_sha256": metadata["artifact_sha256"],
        "file_sha256": file_sha256(path),
        "controller_input": CONTROLLER_FORMAT,
        "cns_adapter_sha256": cns["adapter_sha256"],
        "genome_scope": GENOME_SCOPE,
    }


def verify_controller(value: Mapping[str, Any], artifact: str | Path) -> None:
    manifest = validate_manifest(value)
    if manifest["controller"] != controller_identity(artifact):
        raise ValueError("birth manifest names a different resident controller")


def cns_service_birth_records(
    value: Mapping[str, Any], resident_ids: Sequence[str]
) -> list[dict[str, str]]:
    manifest = validate_manifest(value)
    if (
        not isinstance(resident_ids, Sequence)
        or isinstance(resident_ids, (str, bytes))
        or len(resident_ids) != len(manifest["residents"])
        or any(not isinstance(item, str) or not item for item in resident_ids)
        or len(set(resident_ids)) != len(resident_ids)
    ):
        raise ValueError("CNS service birth requires one distinct ID per founder")
    return [
        {"id": resident_id, "cns_adapter_sha256": row["cns_adapter_sha256"]}
        for resident_id, row in zip(resident_ids, manifest["residents"], strict=True)
    ]


def inherited_body_templates(
    habitat: Mapping[str, Any], biosphere: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain constitutive physical inheritance without adult chemistry/state."""
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
                "pools": {},
                "atp": 0.0,
                "atp_capacity": source["atp_capacity"],
            }
        result[body["id"]] = {
            "body": copy.deepcopy(body),
            "mobile": {
                key: copy.deepcopy(item)
                for key, item in mobile.items()
                if key != "id" and not key.endswith("_row")
            },
            "exchange": {
                key: copy.deepcopy(item)
                for key, item in exchange[body["id"]].items()
                if key != "id"
            },
            "founders": founders,
        }
    return result


__all__ = [
    "FORMAT",
    "GENOME_SCOPE",
    "cns_service_birth_records",
    "controller_identity",
    "file_sha256",
    "inherited_body_templates",
    "load_manifest",
    "validate_manifest",
    "verify_controller",
]
