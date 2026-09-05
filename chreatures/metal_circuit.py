"""Persistent local Metal backend for the full retinal-v1 MaleCNS circuit."""

from __future__ import annotations
import hashlib, json, os, re, subprocess, threading
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
CANONICAL_ARTIFACT_SHA256 = "4a2df4b62208cb4021c6abe1e33c02f008f13d8964c90eebe8255a68a9b88df0"
CANONICAL_GRAPH_SHA256 = "48ce8c8f643b8b533172a84814da2a08e8b5fbf060e1cb6b4f8beaca5073d625"
CANONICAL_PORT_SPEC_SHA256 = "fffb48c65bdb5bc2503ff8ad7c65b4419e12aa9ef5b58b9f36bc910f64dadb6f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MetalCircuit:
    """RemoteBrain-compatible fixed-capacity circuit backed by one native process."""

    def __init__(
        self,
        artifact: str | Path,
        port_bundle: str | Path,
        *,
        capacity: int = 3,
        binary: str | Path | None = None,
        tau: float = 0.16,
        gain: float = 0.92,
        support_recovery: float = 0.024,
        kernel: str = "row",
        manifest: str | Path | None = None,
    ):
        if capacity != 3:
            raise ValueError("experimental Metal backend has fixed capacity 3")
        self.capacity = capacity
        self.tau = tau
        self.gain = gain
        self.support_recovery = support_recovery
        if (tau, gain, support_recovery) != (0.16, 0.92, 0.024):
            raise ValueError(
                "native v1 dynamics are fixed at tau=.16, gain=.92, recovery=.024"
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
        self._input_position = {n: i for i, n in enumerate(self.input_names)}
        self._slots = {}
        self._resident_for_slot = [None] * 3
        self.times = np.zeros(3, dtype=np.float64)
        artifact = Path(artifact)
        manifest_path = (
            Path(manifest) if manifest is not None else artifact.with_suffix(".manifest.json")
        )
        receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "format": "metal-csr-v2",
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
            raise ValueError("Metal artifact manifest is not the pinned canonical v2 recipe")
        if (
            self.graph_hash != receipt["graph_sha256"]
            or self.port_spec_hash != receipt["port_spec_sha256"]
        ):
            raise ValueError("port bundle identity differs from the Metal artifact manifest")
        if _sha256(port_bundle) != receipt.get("port_bundle_sha256"):
            raise ValueError("port bundle checksum differs from the Metal artifact manifest")
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
            [str(binary), str(artifact), kernel],
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
        ):
            self.close()
            raise RuntimeError(f"Metal backend failed startup: {ready}")
        self.device_name = ready["device"]

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

    def add_residents(self, resident_ids: Sequence[str]):
        ids = [str(x) for x in resident_ids]
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
        for rid, slot in assigned.items():
            self._slots[rid] = slot
            self._resident_for_slot[slot] = rid
            self.times[slot] = 0
            mask |= 1 << slot
        self._call({"op": "reset", "mask": mask})
        return assigned

    def remove_residents(self, resident_ids: Sequence[str]):
        ids = [str(x) for x in resident_ids]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("resident IDs must be nonempty and unique")
        if any(x not in self._slots for x in ids):
            raise KeyError("resident does not exist")
        mask = 0
        for rid in ids:
            slot = self._slots.pop(rid)
            self._resident_for_slot[slot] = None
            self.times[slot] = 0
            mask |= 1 << slot
        self._call({"op": "reset", "mask": mask})

    def step(self, residents: Sequence[Mapping[str, Any]], dt: float):
        if not np.isfinite(dt) or not 0 < dt <= 0.2:
            raise ValueError("dt must be finite and in (0, 0.2]")
        if not residents or len(residents) > 3:
            raise ValueError("step must contain 1..3 residents")
        ids = [str(x.get("id", "")) for x in residents]
        if len(set(ids)) != len(ids) or any(x not in self._slots for x in ids):
            raise ValueError("step resident IDs must be unique and allocated")
        channels = np.zeros((351, 3), dtype=np.float32)
        mask = 0
        for item, rid in zip(residents, ids, strict=True):
            senses = item.get("senses")
            if not isinstance(senses, Mapping):
                raise ValueError("each resident needs a senses object")
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
        response = self._call(
            {
                "op": "step",
                "dt": dt,
                "active_mask": mask,
                "channels": channels.ravel().tolist(),
            }
        )
        combined = np.asarray(response["combined"], dtype=np.float32).reshape(387, 3)
        out = []
        for rid in ids:
            slot = self._slots[rid]
            self.times[slot] += dt
            vector = combined[:384, slot].tolist()
            out.append(
                {
                    "id": rid,
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
            )
        return out

    def metadata(self):
        return {
            "graph": {
                "sha256": self.graph_hash,
                "neurons": self.n,
                "edges": self.edge_count,
                "artifact_sha256": self.artifact_sha256,
                "artifact_format": "metal-csr-v2",
            },
            "device": {
                "type": "metal",
                "name": self.device_name,
                "kernel": self.kernel,
            },
            "capacity": 3,
            "residents": self.resident_ids,
            "dynamics": {
                "tau": self.tau,
                "gain": self.gain,
                "support_recovery": self.support_recovery,
                "substeps": 2,
                "kernel": self.kernel,
            },
            "inputs": self.input_names,
            "readouts": self.readout_names,
            "ports": {
                "mode": "versioned_bundle",
                "name": "retinal-v1",
                "spec_hash": self.port_spec_hash,
                "input_count": 351,
                "readout_count": 384,
            },
        }

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
                "Metal v1 snapshots require the complete active cohort in slot order"
            )
        path = self._path(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "version": 3,
                "graph_sha256": self.graph_hash,
                "artifact_sha256": self.artifact_sha256,
                "port_spec_hash": self.port_spec_hash,
                "kernel": self.kernel,
                "resident_ids": self._resident_for_slot,
                "times": self.times.tolist(),
                "input_names": self.input_names,
                "readout_names": self.readout_names,
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
        version = meta.get("version")
        legacy_verified = (
            version == 2
            and self.artifact_sha256 == CANONICAL_ARTIFACT_SHA256
            and self.graph_hash == CANONICAL_GRAPH_SHA256
            and self.port_spec_hash == CANONICAL_PORT_SPEC_SHA256
        )
        if (
            version not in {2, 3}
            or (version == 3 and meta.get("artifact_sha256") != self.artifact_sha256)
            or (version == 2 and not legacy_verified)
            or meta.get("graph_sha256") != self.graph_hash
            or meta.get("port_spec_hash") != self.port_spec_hash
            or meta.get("kernel") != self.kernel
            or meta.get("input_names") != self.input_names
            or meta.get("readout_names") != self.readout_names
        ):
            raise ValueError(
                "snapshot is incompatible with this artifact, graph, port interface, or Metal kernel"
            )
        slots = meta["resident_ids"]
        active = [x for x in slots if x is not None]
        if resident_ids is not None and list(map(str, resident_ids)) != active:
            raise ValueError("restore resident IDs differ")
        restored = json.loads(
            self._call({"op": "restore", "path": str(path)})["metadata"]
        )
        if restored != meta:
            raise ValueError("native snapshot metadata changed during restore")
        self._resident_for_slot = slots
        self._slots = {rid: i for i, rid in enumerate(slots) if rid is not None}
        self.times = np.asarray(meta["times"], dtype=np.float64)
        return {
            "name": name,
            "sha256": digest,
            "artifact_sha256": self.artifact_sha256,
            "bytes": path.stat().st_size,
            "scope": "all",
            "residents": active,
        }

    def close(self):
        process = getattr(self, "_process", None)
        if process and process.poll() is None:
            try:
                self._call({"op": "shutdown"})
            except Exception:
                process.terminate()
            process.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
