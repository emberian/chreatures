"""Authenticated thin pipe to the same Rust ResidentRuntime used in production."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import select
import subprocess

import numpy as np

from chreatures.sensorimotor_worker_native import _load_resident, _packed
from chreatures.sequence_control import CORE_ORDER, PREDICTOR_ORDER, EMBEDDED_ORDER
from .batch_collection import _write_receipt
from .data import sha256_file


def prepare_pack(resident_path: Path, expected_sha: str, service_sha: str, output: Path):
    if sha256_file(resident_path) != expected_sha:
        raise ValueError("Private resident file identity differs")
    metadata, arrays, control = _load_resident(resident_path)
    if metadata["cns_service"]["service_artifact_sha256"] != service_sha:
        raise ValueError("Private resident belongs to a different CNS service")
    components = metadata["controller_components"]
    output.mkdir(exist_ok=False)
    buffers = {}
    for name, order in (("core", CORE_ORDER), ("predictor", PREDICTOR_ORDER), ("sequence", EMBEDDED_ORDER)):
        path = output / (name + ".f32")
        path.write_bytes(_packed(arrays, order).astype("<f4", copy=False).tobytes())
        path.chmod(0o444)
        buffers[name] = {"file": path.name, "sha256": sha256_file(path)}
    pack = {
        "format": "chreatures-research-resident-pack-v1", "resident_file_sha256": expected_sha,
        "artifact_sha256": metadata["artifact_sha256"], "cns_service_sha256": service_sha,
        "training_status": metadata["initialization"]["training_status"], "buffers": buffers,
        "config": dict(batch=4, action_mode="sample", action_seed=314159, suffix_seed=271828,
                       tick_seconds=.01, context_policy_version="signed-context12-v1",
                       private_learning_version="context-consequence-v1",
                       core_sha256=components["core_packed_sha256"],
                       predictor_sha256=components["predictor_packed_sha256"],
                       sequence_control_version=control.version, sequence_control_sha256=control.sha256,
                       research_training=False),
    }
    path = output / "resident-pack.json"
    _write_receipt(path, pack)
    path.chmod(0o444)
    return path, pack


class ProductionResident:
    def __init__(self, node: Path, runtime: Path, pack: Path, stderr):
        self.failed = False
        self.serial = 0
        self.process = subprocess.Popen(
            [str(node.resolve()), str(Path(__file__).with_suffix(".mjs")), str(runtime.resolve()), str(pack.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
        )
        try:
            self.ready = self._read()
            if not self.ready.get("ready"):
                raise RuntimeError("Resident runtime did not initialize")
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _read(self):
        if not select.select([self.process.stdout], [], [], 60)[0]:
            raise TimeoutError("Resident boundary timed out; no mutation retry")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Resident boundary exited")
        return json.loads(line)

    def _rpc(self, op, **fields):
        if self.failed:
            raise RuntimeError("Resident mutation failure is latched")
        self.serial += 1
        try:
            self.process.stdin.write(json.dumps(dict(id=self.serial, op=op, **fields)) + "\n")
            self.process.stdin.flush()
            reply = self._read()
            if reply.get("id") != self.serial or not reply.get("ok"):
                raise RuntimeError(f"Resident operation failed: {reply}")
            return reply
        except BaseException:
            self.failed = True
            raise

    @staticmethod
    def _encode(value, shape):
        value = np.asarray(value, dtype="<f4")
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Resident input tensor differs")
        return base64.b64encode(value.tobytes()).decode()

    def acknowledge(self, tick, context):
        reply = self._rpc("acknowledge", tick=tick, context=self._encode(context, (4, 12)))
        if reply["accepted"] != [1, 1, 1, 1]:
            self.failed = True
            raise RuntimeError("Actual delivered context was not acknowledged")

    def step(self, tick, latent, delivered):
        reply = self._rpc("step", tick=tick, latent=self._encode(latent, (4, 512)),
                          context=self._encode(delivered, (4, 12)))
        context = np.frombuffer(base64.b64decode(reply["context"], validate=True), dtype="<f4").copy().reshape(4, 12)
        if not np.isfinite(context).all() or np.any(np.abs(context) > 1):
            self.failed = True
            raise RuntimeError("Private context proposal is invalid")
        return context, reply["diagnostics"]

    def snapshot(self, path, *, verify=False):
        reply = self._rpc("snapshot", verify=verify)
        state = base64.b64decode(reply["state"], validate=True)
        with path.open("xb") as stream:
            stream.write(state)
            stream.flush(); os.fsync(stream.fileno())
        return {"file": path.name, "sha256": sha256_file(path), "bytes": len(state), "exact_restore_verified": reply["exact_restore_verified"]}

    def close(self):
        if self.process.poll() is None:
            try:
                if not self.failed:
                    self._rpc("close")
            finally:
                if self.failed:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
