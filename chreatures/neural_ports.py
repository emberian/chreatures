"""Versioned sparse sensory and readout ports for the full MaleCNS graph.

This module keeps the physical observation schema, its engineered neural
routing, and annotation-derived readouts explicit.  It never constructs a
dense neuron-by-neuron or neuron-by-channel array.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT_SPEC = ROOT / "data" / "ports" / "retinal-v2.json"
DEFAULT_SENSORIUM_PROFILE = ROOT / "data" / "sensorium" / "rich-body-v1.json"

RETINA_ELEVATIONS = 5
RETINA_AZIMUTHS = 16
RETINA_COMPONENTS = ("red", "green", "blue", "proximity")
AXES = ("x", "y", "z")
SIGNS = ("positive", "negative")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoding_sha256(spec: Mapping[str, Any]) -> str:
    """Hash every declarative field consumed by physical-sense preprocessing."""
    semantic = {
        "schema_version": spec.get("schema_version"),
        "name": spec.get("name"),
        "physical_inputs": spec.get("physical_inputs"),
    }
    return hashlib.sha256(_canonical_json(semantic).encode()).hexdigest()


def _names() -> list[str]:
    names = [
        f"retina/e{elevation:02d}/a{azimuth:02d}/{component}"
        for elevation in range(RETINA_ELEVATIONS)
        for azimuth in range(RETINA_AZIMUTHS)
        for component in RETINA_COMPONENTS
    ]
    names.extend(
        f"odor/{side}/{odor}" for side in ("L", "R") for odor in range(3)
    )
    names.extend(
        f"proprio/linear/{axis}/{sign}" for axis in AXES for sign in SIGNS
    )
    names.extend(
        f"proprio/angular/{axis}/{sign}" for axis in AXES for sign in SIGNS
    )
    names.extend(
        f"contact/normal/{axis}/{sign}" for axis in AXES for sign in SIGNS
    )
    names.append("contact/count")
    names.extend(("touch/L", "touch/R"))
    names.extend(f"sound/{tone}" for tone in range(3))
    names.append("shade")
    return names


BASE_INPUT_NAMES = _names()


def load_port_spec(path: str | Path = DEFAULT_PORT_SPEC) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    # Build observations document the generated bundle but do not recursively
    # become part of the semantic spec embedded in that same bundle.
    value.pop("built_artifact", None)
    required = {
        "schema_version", "name", "graph", "physical_inputs", "routing",
        "readouts", "provenance",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"neural port spec needs keys {sorted(required)}")
    if (
        value["schema_version"] != 2
        or value.get("name") != "retinal-v2"
        or value["physical_inputs"]["count"] != len(BASE_INPUT_NAMES)
    ):
        raise ValueError("unsupported or malformed neural port schema")
    if value["physical_inputs"]["ordered_names"] != BASE_INPUT_NAMES:
        raise ValueError("neural port channel order differs from retinal-v2")
    source = value["physical_inputs"].get("retina_source", {})
    profile_path = DEFAULT_SENSORIUM_PROFILE
    if (
        source.get("profile") != "rich-body-v1"
        or source.get("profile_sha256") != _sha256(profile_path)
        or source.get("projection")
        != "peripheral-area-pool-5x16-elevation-2/1/2/1/2-azimuth-pairs-v1"
        or source.get("measured_rays") != 1024
        or source.get("projection_collision_rays") != 0
    ):
        raise ValueError("retinal-v2 does not match the current native sensorium")
    return value


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _opponents(value: np.ndarray, scale: float) -> list[float]:
    normalized = np.clip(value / scale, -1.0, 1.0)
    output: list[float] = []
    for scalar in normalized:
        output.extend((float(max(scalar, 0.0)), float(max(-scalar, 0.0))))
    return output


def encode_physical_senses(
    senses: Mapping[str, Any],
    spec: Mapping[str, Any] | None = None,
    *,
    feature_values: Mapping[str, float] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Convert one local 3D observation into the explicit port vector.

    Optional pretrained features must be passed by their declared port names;
    this function never resizes or projects an opaque feature vector.
    """
    spec = dict(spec or load_port_spec())
    scaling = spec["physical_inputs"]["scaling"]
    retina = _finite_array(
        senses.get("retina3d"),
        (RETINA_ELEVATIONS, RETINA_AZIMUTHS, len(RETINA_COMPONENTS)),
        "retina3d",
    )
    if np.any((retina < 0) | (retina > 1)):
        raise ValueError("retina3d values must be in [0, 1]")
    odor = _finite_array(senses.get("odor"), (2, 3), "odor")
    linear = _finite_array(senses.get("linear_velocity"), (3,), "linear_velocity")
    angular = _finite_array(
        senses.get("angular_velocity3d"), (3,), "angular_velocity3d"
    )
    touch = _finite_array(senses.get("touch"), (2,), "touch")
    sound = _finite_array(senses.get("sound"), (3,), "sound")
    shade = float(senses.get("shade", 0.0))
    if not np.isfinite(shade):
        raise ValueError("shade must be finite")

    raw_normals = senses.get("contact_normals", [])
    if not isinstance(raw_normals, Sequence) or isinstance(raw_normals, (str, bytes)):
        raise ValueError("contact_normals must be a sequence")
    if len(raw_normals) > int(scaling["maximum_contacts"]):
        raise ValueError("contact_normals exceeds the declared contact capacity")
    if raw_normals:
        normals = _finite_array(raw_normals, (len(raw_normals), 3), "contact_normals")
        if np.any(np.linalg.norm(normals, axis=1) > 1.0001):
            raise ValueError("contact normal magnitude exceeds 1")
        positive = np.maximum(normals, 0).max(axis=0)
        negative = np.maximum(-normals, 0).max(axis=0)
    else:
        positive = negative = np.zeros(3, dtype=np.float32)

    vector = retina.reshape(-1).astype(np.float32).tolist()
    vector.extend(np.clip(odor / float(scaling["odor_max"]), 0, 1).reshape(-1).tolist())
    vector.extend(_opponents(linear, float(scaling["linear_velocity_abs_max"])))
    vector.extend(_opponents(angular, float(scaling["angular_velocity_abs_max"])))
    for axis in range(3):
        vector.extend((float(positive[axis]), float(negative[axis])))
    vector.append(min(1.0, len(raw_normals) / float(scaling["maximum_contacts"])))
    vector.extend(np.clip(touch, 0, 1).tolist())
    vector.extend(np.clip(sound / float(scaling["sound_max"]), 0, 1).tolist())
    vector.append(float(np.clip(shade, 0, 1)))
    names = BASE_INPUT_NAMES.copy()

    declared_features = spec["physical_inputs"].get("feature_ports", [])
    supplied = dict(feature_values or {})
    expected = [str(item["name"]) for item in declared_features]
    if set(supplied) != set(expected):
        raise ValueError(
            f"feature values must exactly match declared ports: {expected}"
        )
    for item in declared_features:
        name = str(item["name"])
        value = float(supplied[name])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"feature port {name!r} must be finite in [0, 1]")
        names.append(name)
        vector.append(value)
    array = np.asarray(vector, dtype=np.float32)
    if len(names) != len(array) or not np.isfinite(array).all():
        raise RuntimeError("neural port encoder produced malformed output")
    return names, array


def sensory_channel_dict(
    senses: Mapping[str, Any],
    spec: Mapping[str, Any] | None = None,
    *,
    feature_values: Mapping[str, float] | None = None,
) -> dict[str, float]:
    names, values = encode_physical_senses(
        senses, spec, feature_values=feature_values
    )
    return dict(zip(names, values.astype(float).tolist(), strict=True))


def _read_hex_annotations(graph: Any, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - extraction-host dependency
        raise RuntimeError(
            "building retinal-v2 needs pyarrow and the pinned MaleCNS annotation Feather; "
            "loading an already serialized port bundle does not"
        ) from exc

    expected = graph.manifest["sources"]["annotations"]
    if _sha256(path) != expected["sha256"]:
        raise ValueError("retinal annotation source does not match the graph manifest")
    required = ("bodyId", "status", "assignedOlHex1", "assignedOlHex2")
    ids: list[np.ndarray] = []
    hex1: list[np.ndarray] = []
    hex2: list[np.ndarray] = []
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        if not set(required).issubset(reader.schema.names):
            raise ValueError("MaleCNS annotation source lacks optic-lobe hex fields")
        positions = [reader.schema.get_field_index(name) for name in required]
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            body, status, first, second = [batch.column(pos) for pos in positions]
            keep = pc.and_(
                pc.fill_null(pc.equal(status, "Traced"), False),
                pc.and_(pc.is_valid(first), pc.is_valid(second)),
            )
            ids.append(body.filter(keep).to_numpy(zero_copy_only=False))
            hex1.append(first.filter(keep).to_numpy(zero_copy_only=False))
            hex2.append(second.filter(keep).to_numpy(zero_copy_only=False))
    body_ids = np.concatenate(ids).astype(np.int64, copy=False)
    local = np.searchsorted(graph.body_ids, body_ids)
    valid = (
        (local < graph.n)
        & (graph.body_ids[np.minimum(local, graph.n - 1)] == body_ids)
    )
    if not valid.all():
        raise ValueError("optic-lobe annotations are not a subset of the loaded graph")
    return local.astype(np.int32), np.concatenate(hex1), np.concatenate(hex2)


def _counts(values: np.ndarray) -> dict[str, int]:
    return dict(sorted(Counter(str(item) if str(item) else "unavailable" for item in values).items()))


def _source_record(graph: Any, indices: np.ndarray) -> dict[str, Any]:
    return {
        "neurons": int(len(indices)),
        "superclasses": _counts(graph.superclasses[indices]),
        "classes": _counts(graph.classes[indices]),
        "types": _counts(graph.types[indices]),
        "sides": _counts(graph.sides[indices]),
        "soma_neuromeres": _counts(graph.soma_neuromeres[indices]),
        "entry_nerves": _counts(graph.entry_nerves[indices]),
        "exit_nerves": _counts(graph.exit_nerves[indices]),
    }


def _type_bucket_indices(
    graph: Any, candidates: np.ndarray, bucket: int, buckets: int
) -> np.ndarray:
    types = sorted(set(graph.types[candidates]) - {""})
    selected_types = {value for position, value in enumerate(types) if position % buckets == bucket}
    selected = candidates[np.isin(graph.types[candidates], list(selected_types))]
    # Untyped cells receive all related fictional channels rather than acquiring
    # a fake identity from an arbitrary neuron ordering.
    untyped = candidates[graph.types[candidates] == ""]
    return np.unique(np.concatenate((selected, untyped))).astype(np.int32)


def _base_input_assignments(
    graph: Any, spec: Mapping[str, Any], annotation_path: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    assignments: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    local, first, second = _read_hex_annotations(graph, annotation_path)
    routing = spec["routing"]["retina"]
    first_bins = np.clip(
        ((first - float(routing["hex1_min"])) * RETINA_AZIMUTHS
         / (float(routing["hex1_max"]) - float(routing["hex1_min"]) + 1)),
        0, RETINA_AZIMUTHS - 1,
    ).astype(np.int32)
    second_bins = np.clip(
        ((second - float(routing["hex2_min"])) * RETINA_ELEVATIONS
         / (float(routing["hex2_max"]) - float(routing["hex2_min"]) + 1)),
        0, RETINA_ELEVATIONS - 1,
    ).astype(np.int32)
    cohorts = routing["component_type_cohorts"]
    for elevation in range(RETINA_ELEVATIONS):
        for azimuth in range(RETINA_AZIMUTHS):
            pixel = (first_bins == azimuth) & (second_bins == elevation)
            for component in RETINA_COMPONENTS:
                name = f"retina/e{elevation:02d}/a{azimuth:02d}/{component}"
                component_mask = np.isin(graph.types[local], cohorts[component])
                indices = np.unique(local[pixel & component_mask]).astype(np.int32)
                nearest_fallback = False
                if not len(indices):
                    # The optic-lobe lattice has empty corners when projected to
                    # a rectangular raster. Use the nearest measured coordinate
                    # from the same declared type cohort and record the fallback.
                    target_first = float(routing["hex1_min"]) + (
                        azimuth + 0.5
                    ) * (
                        float(routing["hex1_max"])
                        - float(routing["hex1_min"])
                        + 1
                    ) / RETINA_AZIMUTHS
                    target_second = float(routing["hex2_min"]) + (
                        elevation + 0.5
                    ) * (
                        float(routing["hex2_max"])
                        - float(routing["hex2_min"])
                        + 1
                    ) / RETINA_ELEVATIONS
                    distance = (
                        (first - target_first) ** 2
                        + (second - target_second) ** 2
                    )
                    nearest = np.min(distance[component_mask])
                    indices = np.unique(
                        local[component_mask & np.isclose(distance, nearest)]
                    ).astype(np.int32)
                    nearest_fallback = True
                assignments[name] = indices
                records.append({
                    "name": name,
                    "family": "retina",
                    "routing": "measured assignedOlHex coordinates binned to the 5x16 canonical neural raster; sensor values are an engineered area pool of measured rich-body-v1 peripheral rays, and RGB/proximity type cohorts plus camera-axis alignment are engineered",
                    "hex_bin": {"hex1_to_azimuth": azimuth, "hex2_to_elevation": elevation},
                    "component": component,
                    "nearest_hex_fallback": nearest_fallback,
                    "source": _source_record(graph, indices),
                })

    afferent = graph.population_indices("afferent")
    classes = graph.classes
    sides = graph.sides
    olfactory = afferent[classes[afferent] == "olfactory"]
    for side in ("L", "R"):
        # Neurons without a resolved side join both channels. Assigning them to
        # one side by ID would manufacture laterality that the source lacks.
        side_candidates = olfactory[
            (sides[olfactory] == side) | ~np.isin(sides[olfactory], ["L", "R"])
        ]
        for odor in range(3):
            name = f"odor/{side}/{odor}"
            indices = _type_bucket_indices(graph, side_candidates, odor, 3)
            assignments[name] = indices
            records.append({
                "name": name, "family": "odor",
                "routing": "anatomical olfactory afferents and side; fictional odor identity is an engineered exact-type partition",
                "source": _source_record(graph, indices),
            })

    proprio = afferent[classes[afferent] == "mechanosensory_proprioceptive"]
    tactile = afferent[np.isin(
        classes[afferent], ["mechanosensory", "mechanosensory_tactile", "mechanosensory_tbc"]
    )]
    # Peripheral sensory bodies have no soma-neuromere annotation. Their entry
    # nerves provide the measured segmental grouping used by this interface.
    nerve_for_axis = dict(zip(AXES, ("ProLN", "MesoLN", "MetaLN"), strict=True))
    side_for_sign = {"positive": "R", "negative": "L"}
    for family, candidates in (("proprio/linear", proprio), ("proprio/angular", proprio)):
        for axis_index, axis in enumerate(AXES):
            for sign_index, sign in enumerate(SIGNS):
                mask = (
                    (graph.entry_nerves[candidates] == nerve_for_axis[axis])
                    & (graph.sides[candidates] == side_for_sign[sign])
                )
                candidates_for_port = candidates[mask]
                bucket = 0 if family.endswith("linear") else 1
                indices = _type_bucket_indices(graph, candidates_for_port, bucket, 2)
                name = f"{family}/{axis}/{sign}"
                assignments[name] = indices
                records.append({
                    "name": name, "family": family,
                    "routing": "proprioceptive afferent class with engineered axis/sign to entry-nerve/side/type cohort",
                    "source": _source_record(graph, indices),
                })

    for axis, nerve in nerve_for_axis.items():
        for sign in SIGNS:
            candidates = tactile[
                (graph.entry_nerves[tactile] == nerve)
                & (graph.sides[tactile] == side_for_sign[sign])
            ]
            name = f"contact/normal/{axis}/{sign}"
            assignments[name] = candidates.astype(np.int32)
            records.append({
                "name": name, "family": "contact_normal",
                "routing": "tactile/mechanosensory afferents with engineered local normal axis/sign to entry-nerve/side",
                "source": _source_record(graph, assignments[name]),
            })
    assignments["contact/count"] = tactile.astype(np.int32)
    records.append({
        "name": "contact/count", "family": "contact",
        "routing": "all annotated tactile/mechanosensory afferents",
        "source": _source_record(graph, assignments["contact/count"]),
    })
    for side in ("L", "R"):
        name = f"touch/{side}"
        assignments[name] = tactile[graph.sides[tactile] == side].astype(np.int32)
        records.append({
            "name": name, "family": "touch",
            "routing": "annotated tactile/mechanosensory afferents by side",
            "source": _source_record(graph, assignments[name]),
        })

    auditory = afferent[graph.subclasses[afferent] == "auditory"]
    for tone in range(3):
        name = f"sound/{tone}"
        assignments[name] = _type_bucket_indices(graph, auditory, tone, 3)
        records.append({
            "name": name, "family": "sound",
            "routing": "annotated auditory afferents; fictional tone identity is an engineered exact-type partition",
            "source": _source_record(graph, assignments[name]),
        })
    shade = afferent[np.isin(classes[afferent], ["hygrosensory", "thermosensory"])]
    assignments["shade"] = shade.astype(np.int32)
    records.append({
        "name": "shade", "family": "shade",
        "routing": "annotated hygro/thermosensory afferents; shade combination is engineered",
        "source": _source_record(graph, assignments["shade"]),
    })
    if list(assignments) != BASE_INPUT_NAMES:
        raise RuntimeError("retinal-v2 input builder changed declared channel order")
    if any(not len(indices) for indices in assignments.values()):
        empty = [name for name, indices in assignments.items() if not len(indices)]
        raise ValueError(f"neural input ports have no source neurons: {empty}")
    return assignments, records


def _domain_masks(graph: Any) -> dict[str, np.ndarray]:
    return {
        "visual": (
            np.isin(graph.superclasses, [
                "ol_intrinsic", "ol_sensory", "visual_projection",
                "visual_projection_tbc", "visual_centrifugal",
            ])
            | (graph.classes == "visual")
        ),
        "mushroom_body": np.isin(
            graph.classes,
            ["ALPN", "ALLN", "ALIN", "ALON", "Kenyon_Cell", "MBON", "DAN"],
        ),
        "navigation": (
            (graph.classes == "CX")
            | np.isin(graph.superclasses, [
                "ascending_neuron", "descending_neuron",
                "sensory_ascending", "sensory_descending",
            ])
        ),
        "efferent": np.asarray([
            ("_motor" in value)
            or ("_efferent" in value)
            or value.startswith("efferent_")
            for value in graph.superclasses
        ]),
    }


def _readout_assignments(
    graph: Any, spec: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    assignments: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    masks = _domain_masks(graph)
    for domain, quota_value in spec["readouts"]["domain_quotas"].items():
        quota = int(quota_value)
        candidates = np.flatnonzero(masks[domain]).astype(np.int32)
        strata: dict[str, list[int]] = {}
        for index in candidates:
            cell_type = (
                graph.types[index]
                or graph.classes[index]
                or graph.subclasses[index]
                or graph.superclasses[index]
                or "unassigned"
            )
            side = graph.sides[index] or "unknown"
            region = (
                graph.soma_neuromeres[index]
                or graph.exit_nerves[index]
                or "none"
            )
            signature = f"type={cell_type}|side={side}|region={region}"
            strata.setdefault(signature, []).append(int(index))
        if len(strata) < quota:
            raise ValueError(f"{domain} provides {len(strata)} strata, fewer than quota {quota}")
        ranked = sorted(strata, key=lambda key: (-len(strata[key]), key))
        for signature in ranked[: quota - 1]:
            name = f"readout/{domain}/{signature}"
            indices = np.asarray(strata[signature], dtype=np.int32)
            assignments[name] = indices
            records.append({
                "name": name, "domain": domain, "stratum": signature,
                "source": _source_record(graph, indices),
            })
        other = np.asarray(
            sorted(index for signature in ranked[quota - 1:] for index in strata[signature]),
            dtype=np.int32,
        )
        name = f"readout/{domain}/other"
        assignments[name] = other
        records.append({
            "name": name, "domain": domain,
            "stratum": f"aggregate of {len(ranked) - quota + 1} less-populous exact strata",
            "source": _source_record(graph, other),
        })
    expected = int(spec["readouts"]["count"])
    if len(assignments) != expected or any(not len(value) for value in assignments.values()):
        raise RuntimeError("readout builder did not produce the declared nonempty ports")
    return assignments, records


@dataclass
class NeuralPortBundle:
    spec: dict[str, Any]
    graph_hash: str
    input_names: list[str]
    input_map: sparse.csr_matrix
    readout_names: list[str]
    readout_map: sparse.csr_matrix
    input_ports: list[dict[str, Any]]
    readout_ports: list[dict[str, Any]]

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.spec).encode()).hexdigest()

    def remote_brain_kwargs(self) -> dict[str, tuple[list[str], sparse.csr_matrix]]:
        """Arguments accepted directly by ``RemoteBrain(..., **value)``."""
        return {
            "input_map": (self.input_names, self.input_map),
            "readout_map": (self.readout_names, self.readout_map),
        }

    def encode(
        self, senses: Mapping[str, Any], *,
        feature_values: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        names, values = encode_physical_senses(
            senses, self.spec, feature_values=feature_values
        )
        if names != self.input_names:
            raise ValueError("encoded sensory names differ from bundle inputs")
        return values

    def channel_dict(
        self, senses: Mapping[str, Any], *,
        feature_values: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        return dict(zip(
            self.input_names,
            self.encode(senses, feature_values=feature_values).astype(float).tolist(),
            strict=True,
        ))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.spec["name"],
            "graph_hash": self.graph_hash,
            "spec_hash": self.spec_hash,
            "inputs": {
                "shape": list(self.input_map.shape), "nnz": int(self.input_map.nnz),
                "names": self.input_names, "ports": self.input_ports,
            },
            "readouts": {
                "shape": list(self.readout_map.shape), "nnz": int(self.readout_map.nnz),
                "names": self.readout_names, "ports": self.readout_ports,
            },
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        """Serialize both sparse maps and their provenance without pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        inputs = self.input_map.tocsr().astype(np.float32, copy=False)
        readouts = self.readout_map.tocsr().astype(np.float32, copy=False)
        metadata = self.metadata()
        np.savez_compressed(
            path,
            metadata=np.asarray(_canonical_json(metadata)),
            spec=np.asarray(_canonical_json(self.spec)),
            input_names=np.asarray(self.input_names),
            input_indptr=inputs.indptr.astype(np.int64, copy=False),
            input_indices=inputs.indices.astype(np.int32, copy=False),
            input_data=inputs.data,
            input_shape=np.asarray(inputs.shape, dtype=np.int64),
            readout_names=np.asarray(self.readout_names),
            readout_indptr=readouts.indptr.astype(np.int64, copy=False),
            readout_indices=readouts.indices.astype(np.int32, copy=False),
            readout_data=readouts.data,
            readout_shape=np.asarray(readouts.shape, dtype=np.int64),
        )
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}

    @classmethod
    def load(cls, path: str | Path, graph: Any) -> "NeuralPortBundle":
        with np.load(path, allow_pickle=False) as value:
            metadata = json.loads(str(value["metadata"]))
            spec = json.loads(str(value["spec"]))
            if metadata["graph_hash"] != graph.hash:
                raise ValueError("serialized neural port belongs to a different graph")
            input_shape = tuple(map(int, value["input_shape"]))
            readout_shape = tuple(map(int, value["readout_shape"]))
            inputs = sparse.csr_matrix(
                (value["input_data"], value["input_indices"], value["input_indptr"]),
                shape=input_shape,
            )
            readouts = sparse.csr_matrix(
                (value["readout_data"], value["readout_indices"], value["readout_indptr"]),
                shape=readout_shape,
            )
            bundle = cls(
                spec=spec,
                graph_hash=metadata["graph_hash"],
                input_names=value["input_names"].astype(str).tolist(),
                input_map=inputs,
                readout_names=value["readout_names"].astype(str).tolist(),
                readout_map=readouts,
                input_ports=metadata["inputs"]["ports"],
                readout_ports=metadata["readouts"]["ports"],
            )
        if bundle.spec_hash != metadata["spec_hash"]:
            raise ValueError("serialized neural port spec hash mismatch")
        return bundle


def build_neural_port(
    graph: Any,
    spec_path: str | Path = DEFAULT_PORT_SPEC,
    *,
    annotation_path: str | Path | None = None,
    feature_ports: Mapping[str, Any] | None = None,
) -> NeuralPortBundle:
    """Build the retinal-v2 sparse interface from real MaleCNS annotations."""
    spec = load_port_spec(spec_path)
    expected_graph = spec["graph"].get("dataset_hash")
    if expected_graph and graph.hash != expected_graph:
        raise ValueError("neural port spec is pinned to a different MaleCNS graph")
    if annotation_path is None:
        annotation_path = spec["routing"]["retina"]["annotation_path"]
    assignments, input_records = _base_input_assignments(
        graph, spec, Path(annotation_path)
    )
    declared_features = spec["physical_inputs"].get("feature_ports", [])
    if feature_ports is None:
        feature_ports = {}
    if [item["name"] for item in declared_features] != list(feature_ports):
        raise ValueError("feature selectors must exactly match feature_ports in the spec")
    for item in declared_features:
        name = str(item["name"])
        indices = graph._selector_indices(feature_ports[name])
        if not len(indices):
            raise ValueError(f"feature port {name!r} has no target neurons")
        assignments[name] = indices
        input_records.append({
            "name": name, "family": "pretrained_feature",
            "routing": item["routing"], "source": _source_record(graph, indices),
        })
    input_names, input_map = graph.build_input_map(
        assignments, gains=float(spec["routing"]["input_gain"])
    )
    readout_assignments, readout_records = _readout_assignments(graph, spec)
    readout_names, readout_map = graph.build_readout_map(
        readout_assignments, normalize=True
    )
    return NeuralPortBundle(
        spec=spec,
        graph_hash=graph.hash,
        input_names=input_names,
        input_map=input_map,
        readout_names=readout_names,
        readout_map=readout_map,
        input_ports=input_records,
        readout_ports=readout_records,
    )
