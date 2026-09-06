"""Persistent local Metal backend for the full retinal-v2 MaleCNS circuit."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .neural_genotype import NEURAL_VARIANT_ARRAYS, PHENOTYPE_FORMAT

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
CANONICAL_ARTIFACT_SHA256 = (
    "4a2df4b62208cb4021c6abe1e33c02f008f13d8964c90eebe8255a68a9b88df0"
)
CANONICAL_GRAPH_SHA256 = (
    "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625"
)
CANONICAL_PORT_SPEC_SHA256 = (
    "a3182cc5c546fac164774e56cfcf3d4f185c2feab5a994fe3d2a37cc8604302e"
)
CANONICAL_PORT_BUNDLE_SHA256 = (
    "933b871fdd11dafa8c43afceb9862101984bf0950592af84631a4d7aa9bebe53"
)
MIN_CAPACITY = 6
MAX_CAPACITY = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class MetalCircuit:
    """RemoteBrain-compatible fixed-capacity circuit backed by one native process."""

    def __init__(
        self,
        artifact: str | Path,
        port_bundle: str | Path,
        *,
        capacity: int = 6,
        binary: str | Path | None = None,
        tau: float = 0.16,
        gain: float = 0.92,
        support_recovery: float = 0.024,
        kernel: str = "row",
        manifest: str | Path | None = None,
        mushroom_substrate: Any | None = None,
        mushroom_bridge: Any | None = None,
        mushroom_modulator_mode: str = "synthetic",
        mushroom_plasticity_enabled: bool = True,
        mushroom_config: Any | None = None,
    ):
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError("capacity must be an integer")
        if not MIN_CAPACITY <= capacity <= MAX_CAPACITY:
            raise ValueError(
                f"capacity must be in {MIN_CAPACITY}..{MAX_CAPACITY} for the current Metal cohort"
            )
        self.capacity = capacity
        self.tau = tau
        self.gain = gain
        self.support_recovery = support_recovery
        if (tau, gain, support_recovery) != (0.16, 0.92, 0.024):
            raise ValueError(
                "native dynamics are fixed at tau=.16, gain=.92, recovery=.024"
            )
        port_bundle = Path(port_bundle)
        with np.load(port_bundle, allow_pickle=False) as z:
            self.input_names = z["input_names"].astype(str).tolist()
            self.readout_names = z["readout_names"].astype(str).tolist()
            meta = json.loads(str(z["metadata"]))
        self.graph_hash = str(meta["graph_hash"])
        self.port_spec_hash = str(meta["spec_hash"])
        self.n = 165122
        self.edge_count = 25563197
        if (mushroom_substrate is None) != (mushroom_bridge is None):
            raise ValueError("mushroom substrate and bridge must be supplied together")
        self._mushroom_spec = mushroom_bridge
        self._mushroom_substrate = mushroom_substrate
        self._mushroom_mode = mushroom_bridge is not None
        self._mushroom_modulator_mode = mushroom_modulator_mode
        self._mushroom_plasticity_enabled = bool(mushroom_plasticity_enabled)
        self._mushroom_cohort = None
        if self._mushroom_mode:
            from .mushroom_plasticity import WholeBrainMushroomCohort

            if mushroom_bridge.graph_hash != self.graph_hash:
                raise ValueError("mushroom bridge belongs to a different graph")
            self._mushroom_cohort = WholeBrainMushroomCohort(
                mushroom_substrate,
                mushroom_bridge,
                capacity=capacity,
                config=mushroom_config,
                plasticity_enabled=self._mushroom_plasticity_enabled,
                modulator_mode=mushroom_modulator_mode,
            )
        self._input_position = {n: i for i, n in enumerate(self.input_names)}
        self._slots = {}
        self._resident_for_slot = [None] * capacity
        self._phenotype_for_slot = [None] * capacity
        self.times = np.zeros(capacity, dtype=np.float64)
        artifact = Path(artifact)
        manifest_path = (
            Path(manifest)
            if manifest is not None
            else artifact.with_suffix(".manifest.json")
        )
        self.artifact_manifest_sha256 = _sha256(manifest_path)
        receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 2,
            "format": "metal-csr-v2",
            "recipe": "normalized-signed-float32-recurrence+retinal-v2-csr",
            "artifact_sha256": CANONICAL_ARTIFACT_SHA256,
            "artifact_bytes": 207261844,
            "graph_sha256": CANONICAL_GRAPH_SHA256,
            "port_spec_sha256": CANONICAL_PORT_SPEC_SHA256,
            "neurons": self.n,
            "edges": self.edge_count,
            "inputs": 351,
            "readouts": 384,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Metal artifact manifest is not the pinned canonical v2 recipe"
            )
        if (
            self.graph_hash != receipt["graph_sha256"]
            or self.port_spec_hash != receipt["port_spec_sha256"]
        ):
            raise ValueError(
                "port bundle identity differs from the Metal artifact manifest"
            )
        if (
            receipt.get("port_bundle_sha256") != CANONICAL_PORT_BUNDLE_SHA256
            or _sha256(port_bundle) != CANONICAL_PORT_BUNDLE_SHA256
        ):
            raise ValueError(
                "port bundle checksum differs from the Metal artifact manifest"
            )
        if artifact.stat().st_size != receipt["artifact_bytes"]:
            raise ValueError("Metal artifact byte size differs from its manifest")
        self.artifact_sha256 = _sha256(artifact)
        if self.artifact_sha256 != receipt["artifact_sha256"]:
            raise ValueError("Metal artifact checksum differs from its manifest")
        self.artifact_manifest = receipt
        if binary is None:
            binary = (
                Path(__file__).resolve().parent.parent
                / "native/metal-brain/target/release/metal-brain-server"
            )
        if kernel not in {"row", "simd"}:
            raise ValueError("kernel must be row or simd")
        self.kernel = kernel
        self._process = subprocess.Popen(
            [str(binary), str(artifact), kernel, str(capacity)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.RLock()
        ready = json.loads(self._process.stdout.readline())
        if (
            not ready.get("ok")
            or ready.get("inputs") != 351
            or ready.get("readouts") != 384
            or ready.get("kernel") != kernel
            or ready.get("capacity") != capacity
            or ready.get("storage_tiles") != (capacity + 3) // 4
            or ready.get("phenotype_binding") != "neural-variant-metal-v1"
        ):
            self.close()
            raise RuntimeError(f"Metal backend failed startup: {ready}")
        self.device_name = ready["device"]
        self.native_startup = ready

    @property
    def resident_ids(self):
        return [x for x in self._resident_for_slot if x is not None]

    def _call(self, value):
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("Metal backend exited")
            self._process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            result = json.loads(self._process.stdout.readline())
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "native request failed"))
            return result

    def _load_phenotype(self, reference: Mapping[str, Any]):
        required = {
            "artifact_path",
            "artifact_sha256",
            "phenotype_sha256",
            "graph_sha256",
            "port_spec_sha256",
            "port_bundle_sha256",
        }
        if not isinstance(reference, Mapping) or set(reference) != required:
            raise ValueError(f"neural phenotype reference fields must be {sorted(required)}")
        path = Path(str(reference["artifact_path"])).expanduser().resolve()
        file_sha = str(reference["artifact_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", file_sha) or _sha256(path) != file_sha:
            raise ValueError("neural phenotype artifact checksum differs")
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != {"metadata", *NEURAL_VARIANT_ARRAYS}:
                raise ValueError("neural phenotype artifact arrays differ")
            metadata = json.loads(str(artifact["metadata"]))
            arrays = {
                name: np.ascontiguousarray(artifact[name])
                for name in NEURAL_VARIANT_ARRAYS
            }
        identities = {
            "phenotype_sha256": metadata.get("phenotype_sha256"),
            "graph_sha256": metadata.get("active_graph_sha256"),
            "port_spec_sha256": metadata.get("port_spec_sha256"),
            "port_bundle_sha256": metadata.get("port_bundle_sha256"),
        }
        if any(str(reference[name]) != value for name, value in identities.items()):
            raise ValueError("neural phenotype reference identity differs from artifact")
        if (
            metadata.get("format") != PHENOTYPE_FORMAT
            or identities["graph_sha256"] != self.graph_hash
            or identities["port_spec_sha256"] != self.port_spec_hash
            or identities["port_bundle_sha256"] != CANONICAL_PORT_BUNDLE_SHA256
            or metadata.get("input_names") != self.input_names
            or metadata.get("readout_names") != self.readout_names
        ):
            raise ValueError("neural phenotype belongs to a different graph or port interface")
        expected_shapes = {
            "input_gain": (351,),
            "readout_gain": (384,),
            **{name: (self.n,) for name in NEURAL_VARIANT_ARRAYS[2:]},
        }
        for name, array in arrays.items():
            if (
                array.dtype != np.float32
                or array.shape != expected_shapes[name]
                or not np.isfinite(array).all()
                or np.any((array < 0.05) | (array > 4.0))
            ):
                raise ValueError(f"neural phenotype array {name} differs")
        receipt = metadata.get("receipt")
        if not isinstance(receipt, dict) or (
            receipt.get("array_sha256")
            != {name: _array_sha256(value) for name, value in arrays.items()}
            or hashlib.sha256(_canonical(receipt)).hexdigest()
            != identities["phenotype_sha256"]
        ):
            raise ValueError("neural phenotype receipt differs")
        stored = {
            "artifact_path": str(path),
            "artifact_sha256": file_sha,
            **identities,
            "compatibility_group": metadata.get("compatibility_group"),
            "genotype_sha256": metadata.get("genotype_sha256"),
            "learning_rate_gain": "inactive:no_native_learning_delta_path",
            "modulator_gain": "inactive:no_native_modulator_path",
        }
        return stored, arrays

    def _bind_phenotypes(self, slots, arrays):
        packed = {}
        for name in NEURAL_VARIANT_ARRAYS:
            packed[name] = np.ascontiguousarray(
                np.stack([item[name] for item in arrays], axis=1), dtype=np.float32
            ).ravel().tolist()
        self._call({"op": "bind_phenotypes", "slots": slots, **packed})

    def add_residents(self, residents: Sequence[Mapping[str, Any]]):
        if not isinstance(residents, Sequence) or isinstance(residents, (str, bytes)):
            raise TypeError("residents must be a sequence of birth records")
        records = list(residents)
        if any(not isinstance(item, Mapping) or set(item) != {"id", "neural_phenotype"} for item in records):
            raise ValueError("each birth record requires exactly id and neural_phenotype")
        ids = [str(x["id"]) for x in records]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not x or len(x) > 128 for x in ids)
        ):
            raise ValueError(
                "resident IDs must be nonempty unique strings of at most 128 characters"
            )
        if any(x in self._slots for x in ids):
            raise ValueError("resident already exists")
        free = [i for i, x in enumerate(self._resident_for_slot) if x is None]
        if len(ids) > len(free):
            raise ValueError("resident capacity exceeded")
        assigned = dict(zip(ids, free[: len(ids)], strict=True))
        mask = 0
        loaded = [self._load_phenotype(item["neural_phenotype"]) for item in records]
        slots = list(assigned.values())
        self._bind_phenotypes(slots, [value[1] for value in loaded])
        try:
            for slot in slots:
                mask |= 1 << slot
            self._call({"op": "reset", "mask": mask})
        except Exception:
            self._call({"op": "unbind_phenotypes", "mask": mask})
            raise
        for (rid, slot), (identity, _) in zip(assigned.items(), loaded, strict=True):
            self._slots[rid] = slot
            self._resident_for_slot[slot] = rid
            self._phenotype_for_slot[slot] = identity
            self.times[slot] = 0
        if self._mushroom_mode:
            self._mushroom_cohort.reset_slots(mask)
        return assigned

    def remove_residents(self, resident_ids: Sequence[str]):
        ids = [str(x) for x in resident_ids]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("resident IDs must be nonempty and unique")
        if any(x not in self._slots for x in ids):
            raise KeyError("resident does not exist")
        mask = 0
        for rid in ids:
            slot = self._slots[rid]
            mask |= 1 << slot
        self._call({"op": "reset", "mask": mask})
        self._call({"op": "unbind_phenotypes", "mask": mask})
        for rid in ids:
            slot = self._slots.pop(rid)
            self._resident_for_slot[slot] = None
            self._phenotype_for_slot[slot] = None
            self.times[slot] = 0
        if self._mushroom_mode:
            self._mushroom_cohort.reset_slots(mask)

    def step(self, residents: Sequence[Mapping[str, Any]], dt: float):
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        if not residents or len(residents) > self.capacity:
            raise ValueError(f"step must contain 1..{self.capacity} residents")
        ids = [str(x.get("id", "")) for x in residents]
        if len(set(ids)) != len(ids) or any(x not in self._slots for x in ids):
            raise ValueError("step resident IDs must be unique and allocated")
        channels = np.zeros((351, self.capacity), dtype=np.float32)
        mask = 0
        for item, rid in zip(residents, ids, strict=True):
            senses = item.get("senses")
            if not isinstance(senses, Mapping):
                raise TypeError("each resident needs a senses object")
            unknown = set(senses) - set(self._input_position)
            if unknown:
                raise ValueError(f"unknown sensory channels: {sorted(unknown)}")
            slot = self._slots[rid]
            mask |= 1 << slot
            for name, value in senses.items():
                v = float(value)
                if not np.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError("sensory values must be finite in [0, 1]")
                channels[self._input_position[name], slot] = v
        request = {
            "op": "step",
            "dt": dt,
            "active_mask": mask,
            "channels": channels.ravel().tolist(),
        }
        mushroom_modulator = None
        if self._mushroom_mode:
            corrections = self._mushroom_cohort.pending_correction
            if self._mushroom_cohort.modulator_mode == "synthetic":
                mushroom_modulator = np.zeros((2, self.capacity), dtype=np.float32)
                for item, rid in zip(residents, ids, strict=True):
                    value = np.asarray(item.get("mushroom_modulator"), dtype=np.float32)
                    if value.ndim == 0:
                        value = np.full(2, value, dtype=np.float32)
                    if (
                        value.shape != (2,)
                        or not np.isfinite(value).all()
                        or np.any((value < 0) | (value > 1))
                    ):
                        raise ValueError(
                            "mushroom_modulator must be a finite scalar or bilateral pair in [0, 1]"
                        )
                    mushroom_modulator[:, self._slots[rid]] = value
            elif any(item.get("mushroom_modulator") is not None for item in residents):
                raise ValueError(
                    "actual_ppl101_rate mode does not accept mushroom_modulator"
                )
            request.update(
                selected_neuron_indices=self._mushroom_spec.selected_neuron_indices.tolist(),
                target_neuron_indices=self._mushroom_spec.target_neuron_indices.tolist(),
                target_recurrent_correction=corrections.ravel().tolist(),
            )
        response = self._call(request)
        combined = np.asarray(response["combined"], dtype=np.float32).reshape(
            387, self.capacity
        )
        selected = None
        if self._mushroom_mode:
            selected = np.asarray(response["selected_rates"], dtype=np.float32).reshape(
                self._mushroom_spec.selected_count, self.capacity
            )
            bridge_steps = self._mushroom_cohort.advance(
                selected, mushroom_modulator, mask, dt=dt
            )
        out = []
        for item, rid in zip(residents, ids, strict=True):
            slot = self._slots[rid]
            self.times[slot] += dt
            vector = combined[:384, slot].tolist()
            result = {
                "id": rid,
                "neural_phenotype_sha256": self._phenotype_for_slot[slot][
                    "phenotype_sha256"
                ],
                "time": float(self.times[slot]),
                "features": vector,
                "readout_vector": vector,
                "readouts": dict(zip(self.readout_names, vector, strict=True)),
                "activity": float(combined[384, slot]),
                "activity_mean": float(combined[384, slot]),
                "activity_peak": float(combined[385, slot]),
                "support": float(combined[386, slot]),
                "support_mean": float(combined[386, slot]),
                "gpu_ms": float(response["gpu_ms"]),
            }
            if self._mushroom_mode:
                bridge_step = bridge_steps[slot]
                result["selected_rates"] = selected[:, slot].tolist()
                result["mushroom"] = {
                    "target_recurrent_correction": bridge_step.target_recurrent_correction.tolist(),
                    "actual_dan_rates": bridge_step.actual_dan_rates.tolist(),
                    "actual_mbon_rates": bridge_step.actual_mbon_rates.tolist(),
                }
            out.append(result)
        return out

    def metadata(self):
        result = {
            "graph": {
                "sha256": self.graph_hash,
                "neurons": self.n,
                "edges": self.edge_count,
                "artifact_sha256": self.artifact_sha256,
                "artifact_manifest_sha256": self.artifact_manifest_sha256,
                "artifact_format": "metal-csr-v2",
            },
            "device": {
                "type": "metal",
                "name": self.device_name,
                "kernel": self.kernel,
                "storage_tiles": self.native_startup["storage_tiles"],
            },
            "capacity": self.capacity,
            "residents": self.resident_ids,
            "neural_phenotypes": {
                rid: self._phenotype_for_slot[self._slots[rid]]
                for rid in self.resident_ids
            },
            "dynamics": {
                "tau": self.tau,
                "gain": self.gain,
                "support_recovery": self.support_recovery,
                "substeps": 2,
                "kernel": self.kernel,
            },
            "neural_modulation": {
                "format": PHENOTYPE_FORMAT,
                "binding": "immutable_per_resident_at_birth",
                "input_gain": "active",
                "readout_gain": "active",
                "excitability_gain": "active",
                "recurrent_source_gain": "active",
                "recurrent_target_gain": "active",
                "learning_rate_gain": "inactive:no_native_learning_delta_path",
                "modulator_gain": "inactive:no_native_modulator_path",
            },
            "inputs": self.input_names,
            "readouts": self.readout_names,
            "ports": {
                "mode": "versioned_bundle",
                "name": "retinal-v2",
                "spec_hash": self.port_spec_hash,
                "input_count": 351,
                "readout_count": 384,
            },
        }
        if self._mushroom_mode:
            result["research_mode"] = {
                "format": "chreatures-metal-mushroom-research-v1",
                "bridge_sha256": self._mushroom_spec.bridge_hash,
                "selected_neuron_indices": self._mushroom_spec.selected_neuron_indices.tolist(),
                "selected_body_ids": self._mushroom_spec.selected_body_ids.tolist(),
                "target_neuron_indices": self._mushroom_spec.target_neuron_indices.tolist(),
                "target_body_ids": self._mushroom_spec.target_body_ids.tolist(),
                "modulator_mode": self._mushroom_cohort.modulator_mode,
                "lag_steps": 1,
                "plasticity_enabled": self._mushroom_cohort.plasticity_enabled,
                "config": vars(self._mushroom_cohort.config),
                "native_interface": "metal-mushroom-recurrent-correction-v1",
            }
        return result

    def _path(self, directory, name):
        if not SAFE_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError("snapshot name must be a safe filename")
        return Path(directory).resolve() / f"{name}.npz"

    def snapshot(self, directory, name, resident_ids=None):
        residents = (
            self.resident_ids
            if resident_ids is None
            else [str(x) for x in resident_ids]
        )
        if residents != self.resident_ids:
            raise ValueError(
                "Metal snapshots require the complete active cohort in slot order"
            )
        path = self._path(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "version": 6,
                "state_layout": "neuron-major-float4-tiles-v1",
                "capacity": self.capacity,
                "graph_sha256": self.graph_hash,
                "artifact_sha256": self.artifact_sha256,
                "artifact_manifest_sha256": self.artifact_manifest_sha256,
                "port_spec_hash": self.port_spec_hash,
                "kernel": self.kernel,
                "resident_ids": self._resident_for_slot,
                "neural_phenotypes": self._phenotype_for_slot,
                "times": self.times.tolist(),
                "input_names": self.input_names,
                "readout_names": self.readout_names,
                **(
                    {"mushroom_research": self._mushroom_cohort.snapshot()}
                    if self._mushroom_mode
                    else {}
                ),
            },
            separators=(",", ":"),
        )
        self._call({"op": "snapshot", "path": str(path), "metadata": metadata})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "name": name,
            "sha256": digest,
            "artifact_sha256": self.artifact_sha256,
            "bytes": path.stat().st_size,
            "scope": "all",
            "residents": residents,
        }

    def restore(self, directory, name, expected_sha256=None, resident_ids=None):
        path = self._path(directory, name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("snapshot checksum does not match")
        with path.open("rb") as handle:
            if handle.read(8) != b"MBST1\0\0\0":
                raise ValueError("snapshot has invalid Metal state header")
            length = struct.unpack("<Q", handle.read(8))[0]
            if length > 2_000_000:
                raise ValueError("snapshot metadata is too large")
            meta = json.loads(handle.read(length))
        if (
            meta.get("version") != 6
            or meta.get("state_layout") != "neuron-major-float4-tiles-v1"
            or meta.get("capacity") != self.capacity
            or ("mushroom_research" in meta) != self._mushroom_mode
            or meta.get("artifact_sha256") != self.artifact_sha256
            or meta.get("artifact_manifest_sha256")
            != self.artifact_manifest_sha256
            or meta.get("graph_sha256") != self.graph_hash
            or meta.get("port_spec_hash") != self.port_spec_hash
            or meta.get("kernel") != self.kernel
            or meta.get("input_names") != self.input_names
            or meta.get("readout_names") != self.readout_names
        ):
            raise ValueError(
                "snapshot is incompatible with this artifact, graph, port interface, or Metal kernel"
            )
        restored_cohort = None
        if self._mushroom_mode:
            from .mushroom_plasticity import WholeBrainMushroomCohort

            restored_cohort = WholeBrainMushroomCohort(
                self._mushroom_substrate,
                self._mushroom_spec,
                capacity=self.capacity,
                config=self._mushroom_cohort.config,
                plasticity_enabled=self._mushroom_cohort.plasticity_enabled,
                modulator_mode=self._mushroom_cohort.modulator_mode,
            )
            restored_cohort.restore(meta.get("mushroom_research", {}))
        slots = meta["resident_ids"]
        phenotype_slots = meta.get("neural_phenotypes")
        if (
            not isinstance(slots, list)
            or len(slots) != self.capacity
            or not isinstance(phenotype_slots, list)
            or len(phenotype_slots) != self.capacity
            or any(
                (rid is None) != (phenotype is None)
                for rid, phenotype in zip(slots, phenotype_slots, strict=True)
            )
        ):
            raise ValueError("snapshot resident and phenotype slots differ")
        checkpoint_residents = [x for x in slots if x is not None]
        if (
            any(
                not isinstance(rid, str) or not rid or len(rid) > 128
                for rid in checkpoint_residents
            )
            or len(set(checkpoint_residents)) != len(checkpoint_residents)
            or not isinstance(meta.get("times"), list)
            or len(meta["times"]) != self.capacity
            or not np.isfinite(np.asarray(meta["times"], dtype=np.float64)).all()
        ):
            raise ValueError("snapshot resident identity or time state differs")
        requested = (
            checkpoint_residents
            if resident_ids is None
            else [str(value) for value in resident_ids]
        )
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(rid not in checkpoint_residents for rid in requested)
        ):
            raise ValueError("restore resident IDs must be a nonempty checkpoint subset")
        checkpoint_slot = {
            rid: slot for slot, rid in enumerate(slots) if rid is not None
        }
        loaded_by_slot = {}
        new_slots = []
        for rid in requested:
            slot = checkpoint_slot[rid]
            identity = phenotype_slots[slot]
            current_rid = self._resident_for_slot[slot]
            if rid in self._slots and self._slots[rid] != slot:
                raise ValueError("resident occupies a different slot than its checkpoint")
            if current_rid not in (None, rid):
                raise ValueError("checkpoint slot is occupied by another resident")
            if current_rid == rid:
                current_identity = dict(self._phenotype_for_slot[slot])
                checkpoint_identity = dict(identity)
                current_identity.pop("artifact_path", None)
                checkpoint_identity.pop("artifact_path", None)
                if current_identity != checkpoint_identity:
                    raise ValueError("active resident phenotype differs from its checkpoint")
            reference = {
                name: identity[name]
                for name in (
                    "artifact_path",
                    "artifact_sha256",
                    "phenotype_sha256",
                    "graph_sha256",
                    "port_spec_sha256",
                    "port_bundle_sha256",
                )
            }
            observed, arrays = self._load_phenotype(reference)
            if observed != identity:
                raise ValueError("snapshot neural phenotype identity differs from artifact")
            loaded_by_slot[slot] = arrays
            if current_rid is None:
                new_slots.append(slot)
        mask = sum(1 << checkpoint_slot[rid] for rid in requested)
        new_mask = sum(1 << slot for slot in new_slots)
        if new_slots:
            self._bind_phenotypes(
                new_slots, [loaded_by_slot[slot] for slot in new_slots]
            )
        try:
            restored = json.loads(
                self._call({"op": "restore", "path": str(path), "mask": mask})[
                    "metadata"
                ]
            )
            if restored != meta:
                raise ValueError("native snapshot metadata changed during restore")
        except Exception:
            if new_slots:
                self._call({"op": "unbind_phenotypes", "mask": new_mask})
            raise
        for rid in requested:
            slot = checkpoint_slot[rid]
            self._resident_for_slot[slot] = rid
            self._phenotype_for_slot[slot] = phenotype_slots[slot]
            self._slots[rid] = slot
            self.times[slot] = float(meta["times"][slot])
        if restored_cohort is not None:
            for rid in requested:
                slot = checkpoint_slot[rid]
                self._mushroom_cohort._slots[slot] = restored_cohort._slots[slot]
        return {
            "name": name,
            "sha256": digest,
            "artifact_sha256": self.artifact_sha256,
            "bytes": path.stat().st_size,
            "scope": "selected",
            "residents": requested,
        }

    def close(self):
        process = getattr(self, "_process", None)
        if process and process.poll() is None:
            try:
                self._call({"op": "shutdown"})
            except (
                BrokenPipeError,
                OSError,
                RuntimeError,
                ValueError,
                json.JSONDecodeError,
            ):
                process.terminate()
            process.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
