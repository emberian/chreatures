"""Thin owner of one native, trainable-afferent full-MaleCNS Metal process.

All sensory projection, neural integration and learned readout run natively.
The host owns transport, resident names and checkpoint file receipts only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import threading
from pathlib import Path

import numpy as np

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
IDENTITY_KEYS = ("graph_sha256", "atlas_sha256", "readout_mask_sha256", "adapter_sha256")
INPUT_COUNT, LATENT_COUNT, NEURON_COUNT = 5356, 512, 165122


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_names():
    return [f"optic.{site}.{rgb}" for site in range(1771) for rgb in "rgb"] + [f"body.{i}" for i in range(43)]


class MetalCircuit:
    """One immutable service artifact, with private state in each native slot."""

    def __init__(self, artifact, *, capacity=8, binary=None, kernel="simd"):
        if type(capacity) is not int or not 1 <= capacity <= 32:
            raise ValueError("CNS capacity must be an integer in 1..32")
        if kernel not in {"row", "simd"}:
            raise ValueError("unknown sparse Metal kernel")
        self.capacity, self.kernel = capacity, kernel
        self.artifact = Path(artifact).resolve()
        self.artifact_sha256 = _sha256(self.artifact)
        binary = Path(binary) if binary else Path(__file__).resolve().parents[1] / "native/metal-brain/target/release/metal-brain-server"
        self.execution_identity = {"binary_sha256": _sha256(binary), "kernel": kernel, "clock": "float64 declared interval; float32 neural integration"}
        self._process = subprocess.Popen([str(binary), str(self.artifact), kernel, str(capacity)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._lock = threading.RLock()
        self.uncertain = False
        try:
            ready = json.loads(self._process.stdout.readline())
            if (ready.get("ok") is not True or ready.get("inputs") != INPUT_COUNT
                    or ready.get("readouts") != LATENT_COUNT or ready.get("neurons") != NEURON_COUNT
                    or ready.get("capacity") != capacity):
                raise ValueError("native CNS dimensions differ from current interface")
            model = ready.get("cns_adapter", ready.get("metadata", ready))
            self.cns_identity = {key: model[key] for key in IDENTITY_KEYS}
            if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in self.cns_identity.values()):
                raise ValueError("native CNS artifact lacks authenticated model identities")
            self.cns_identity.update(format="chreatures-cns-service-v2", service_artifact_sha256=self.artifact_sha256,
                                     sensory_dim=INPUT_COUNT, latent_dim=LATENT_COUNT)
            self.native_startup = ready
        except Exception:
            self.close()
            raise
        self.graph_hash = self.cns_identity["graph_sha256"]
        self.n, self.edge_count = NEURON_COUNT, int(ready.get("edges", 25563197))
        self.input_names = input_names()
        self.readout_names = [f"cns.latent.{i}" for i in range(LATENT_COUNT)]
        self._resident_for_slot = [None] * capacity
        self._slots = {}
        self.times = np.zeros(capacity, dtype=np.float64)

    @property
    def resident_ids(self):
        return [rid for rid in self._resident_for_slot if rid is not None]

    def _call(self, value, *, mutation=True):
        with self._lock:
            if self.uncertain and mutation:
                raise RuntimeError("native CNS mutation outcome unknown; restore a coherent world")
            try:
                if self._process.poll() is not None:
                    raise RuntimeError("native CNS process exited")
                self._process.stdin.write(_canonical(value) + "\n")
                self._process.stdin.flush()
                result = json.loads(self._process.stdout.readline())
                if result.get("ok") is not True:
                    raise RuntimeError(result.get("error", "native CNS request failed"))
                return result
            except Exception:
                if mutation:
                    self.uncertain = True
                raise

    def add_residents(self, residents):
        with self._lock:
            if not isinstance(residents, list) or not residents:
                raise ValueError("fresh CNS births require a nonempty list")
            for row in residents:
                if (not isinstance(row, dict) or set(row) != {"id", "cns_adapter_sha256"}
                        or row["cns_adapter_sha256"] != self.cns_identity["adapter_sha256"]
                        or not isinstance(row["id"], str) or not 1 <= len(row["id"]) <= 128):
                    raise ValueError("CNS birth must bind the current learned adapter identity")
            ids = [row["id"] for row in residents]
            free = [i for i, rid in enumerate(self._resident_for_slot) if rid is None]
            if len(set(ids)) != len(ids) or any(rid in self._slots for rid in ids) or len(ids) > len(free):
                raise ValueError("duplicate resident or insufficient CNS capacity")
            slots = free[:len(ids)]
            self._call({"op": "reset", "mask": sum(1 << slot for slot in slots)})
            for rid, slot in zip(ids, slots, strict=True):
                self._slots[rid] = slot
                self._resident_for_slot[slot] = rid
                self.times[slot] = 0
            return dict(zip(ids, slots, strict=True))

    def remove_residents(self, resident_ids):
        with self._lock:
            ids = list(resident_ids)
            if not ids or len(set(ids)) != len(ids) or any(rid not in self._slots for rid in ids):
                raise ValueError("unknown or duplicate CNS resident removal")
            slots = [self._slots[rid] for rid in ids]
            self._call({"op": "reset", "mask": sum(1 << slot for slot in slots)})
            for rid, slot in zip(ids, slots, strict=True):
                del self._slots[rid]
                self._resident_for_slot[slot] = None
                self.times[slot] = 0

    def step(self, entries, dt, *, selected_neuron_indices=None):
        with self._lock:
            if not math.isfinite(dt) or dt <= 0 or dt > 0.1:
                raise ValueError("CNS step duration must be in (0,.1]")
            ids = [entry["id"] for entry in entries]
            if not ids or len(set(ids)) != len(ids) or any(rid not in self._slots for rid in ids):
                raise ValueError("CNS step resident set is invalid")
            sensory = np.zeros((INPUT_COUNT, self.capacity), dtype=np.float32)
            for entry in entries:
                row = np.asarray(entry["sensory"], dtype=np.float32)
                if row.shape != (INPUT_COUNT,) or not np.isfinite(row).all():
                    raise ValueError("CNS sensory row must have 5356 finite values")
                sensory[:, self._slots[entry["id"]]] = row
            request = {"op": "step", "dt": dt, "active_mask": sum(1 << self._slots[rid] for rid in ids),
                       "sensory": sensory.reshape(-1).tolist()}
            if selected_neuron_indices is not None:
                indices = list(selected_neuron_indices)
                if any(type(i) is not int or not 0 <= i < self.n for i in indices) or len(set(indices)) != len(indices):
                    raise ValueError("observer neuron indices differ from graph")
                request["selected_neuron_indices"] = indices
            result = self._call(request)
            try:
                latent = np.asarray(result["latent"], dtype=np.float32).reshape(LATENT_COUNT, self.capacity)
                stats = np.asarray(result["physiology"], dtype=np.float32).reshape(3, self.capacity)
                times = np.asarray(result["times"], dtype=np.float64)
                if times.shape != (self.capacity,) or not np.isfinite(latent).all() or not np.isfinite(stats).all() or not np.isfinite(times).all():
                    raise ValueError("native CNS returned invalid state")
                self.times[:] = times
                selected = None if selected_neuron_indices is None else np.asarray(result["selected_rates"], dtype=np.float32).reshape(len(indices), self.capacity)
                output = []
                for rid in ids:
                    slot = self._slots[rid]
                    row = {"id": rid, "time": float(times[slot]), "features": latent[:, slot].tolist(),
                           "activity": float(stats[0, slot]), "activity_peak": float(stats[1, slot]), "support": float(stats[2, slot]),
                           "cns_adapter_sha256": self.cns_identity["adapter_sha256"]}
                    if selected is not None:
                        row["selected_rates"] = selected[:, slot].tolist()
                    output.append(row)
                return output
            except Exception:
                self.uncertain = True
                raise

    def capture_rates(self, directory, name, resident_id):
        """Record actual graph state through an observer-only binary path."""
        with self._lock:
            if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or resident_id not in self._slots:
                raise ValueError("invalid observer capture identity")
            path = Path(directory).resolve() / "rates" / (name + ".f32")
            path.parent.mkdir(parents=True, exist_ok=True)
            slot = self._slots[resident_id]
            self._call({"op": "capture_rates", "path": str(path), "slot": slot}, mutation=False)
            if path.stat().st_size != self.n * 4:
                raise ValueError("native observer capture size differs")
            return {"name": name, "path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size,
                    "neurons": self.n, "dtype": "<f4", "resident_id": resident_id,
                    "time": float(self.times[slot]), "cns_identity": copy.deepcopy(self.cns_identity)}

    def metadata(self):
        return {"dataset": "MaleCNS v1.0", "scope": "full curated traced brain and ventral nerve cord",
                "graph": {"sha256": self.graph_hash, "neurons": self.n, "edges": self.edge_count},
                "capacity": self.capacity, "residents": self.resident_ids,
                "device": {"type": "metal", "name": self.native_startup.get("device"), "kernel": self.kernel},
                "inputs": self.input_names, "readouts": self.readout_names,
                "cns_adapter": copy.deepcopy(self.cns_identity), "execution": copy.deepcopy(self.execution_identity)}

    def _path(self, directory, name):
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise ValueError("invalid CNS snapshot name")
        return Path(directory).resolve() / (name + ".cnsstate")

    def snapshot(self, directory, name, resident_ids=None):
        with self._lock:
            if resident_ids is not None and resident_ids != self.resident_ids:
                raise ValueError("CNS snapshots require the complete ordered cohort")
            path = self._path(directory, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {"format": "chreatures-cns-host-state-v1", "identity": self.cns_identity,
                        "capacity": self.capacity, "resident_slots": self._resident_for_slot,
                        "execution": self.execution_identity}
            self._call({"op": "snapshot", "path": str(path), "metadata": _canonical(metadata)}, mutation=False)
            return {"name": name, "sha256": _sha256(path), "bytes": path.stat().st_size,
                    "scope": "all", "residents": self.resident_ids, "cns_identity": copy.deepcopy(self.cns_identity)}

    def restore(self, directory, name, expected_sha256=None, resident_ids=None):
        with self._lock:
            path = self._path(directory, name)
            if expected_sha256 is None or _sha256(path) != expected_sha256:
                raise ValueError("CNS snapshot checksum differs")
            inspected = self._call({"op": "inspect_snapshot", "path": str(path)}, mutation=False)
            metadata = json.loads(inspected["metadata"])
            slots = metadata.get("resident_slots")
            if (metadata.get("format") != "chreatures-cns-host-state-v1" or metadata.get("identity") != self.cns_identity
                    or metadata.get("execution") != self.execution_identity
                    or metadata.get("capacity") != self.capacity or not isinstance(slots, list) or len(slots) != self.capacity):
                raise ValueError("CNS snapshot host identity differs")
            ids = [rid for rid in slots if rid is not None]
            if (not ids or len(set(ids)) != len(ids) or any(not isinstance(rid, str) or not 1 <= len(rid) <= 128 for rid in ids)
                    or (resident_ids is not None and list(resident_ids) != ids)):
                raise ValueError("CNS snapshot resident set differs")
            if self.resident_ids and slots != self._resident_for_slot:
                raise ValueError("restore cannot overwrite another active CNS cohort")
            # A whole-world restore replaces all slots, including inactive ones.
            # Clear uncertainty only for the explicit authenticated restoration.
            self.uncertain = False
            restored = self._call({"op": "restore", "path": str(path), "mask": (1 << self.capacity) - 1})
            if json.loads(restored["metadata"]) != metadata:
                self.uncertain = True
                raise RuntimeError("native CNS restore metadata differs")
            self._resident_for_slot = slots
            self._slots = {rid: slot for slot, rid in enumerate(slots) if rid is not None}
            self.times[:] = np.asarray(inspected["times"], dtype=np.float64)
            return {"name": name, "sha256": expected_sha256, "residents": ids}

    def close(self):
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
