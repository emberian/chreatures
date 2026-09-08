"""M2 Dawn host for actual-world collection through the full CHCNS4 graph."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from .collect import CollectionBundle
from .curriculum import Plan, RESIDENTS
from .data import BODY_AFFERENTS, LATENT, MOTOR, OPTIC_SITES, sha256_file
from .native_host import (
    ActualOutcomeEvaluator,
    FlyCurriculumTeacher,
    NodeActualFlyWorld,
    SampledAuthorSteps,
)


class DawnFullCNS:
    """Synchronous FullCNS adapter over one fail-closed Node/Dawn process."""

    def __init__(self, model_directory: Path, node: str = "node") -> None:
        self.model_directory = model_directory.resolve()
        bridge = Path(__file__).with_name("cns_bridge.mjs").resolve()
        self.process = subprocess.Popen(
            [node, str(bridge), "--model", str(self.model_directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to create Dawn CNS pipes")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Dawn CNS exited during startup ({self.process.poll()})")
        ready = json.loads(line)
        if not ready.get("ok") or ready.get("event") != "ready":
            raise RuntimeError(f"Dawn CNS startup failed: {ready}")
        if int(ready["capacity"]) != RESIDENTS:
            raise RuntimeError("Dawn CNS must expose exactly four resident lanes")
        manifest = ready["manifest"]
        if (
            manifest.get("format") != "chreatures-cns-webgpu-v4"
            or manifest.get("version") != 4
            or manifest.get("controlDt") != 0.01
            or manifest.get("substeps") != 2
        ):
            raise RuntimeError("Dawn CNS manifest contract differs")
        self.metadata = {
            "format": "chreatures-cns-service-v4",
            "cns_service_sha256": str(ready["service_sha256"]),
            "adapter_sha256": str(ready["adapter_sha256"]),
            "motor_calibration_sha256": manifest["identity"]["motorCalibration"],
            "atlas_sha256": manifest["identity"]["atlas"],
            "webgpu_manifest_sha256": str(ready["manifest_sha256"]),
            "cns_backend": str(ready["backend"]),
            "cns_source_revision": str(manifest["sourceRevision"]),
        }
        self._id = 0

    def _rpc(self, command: str, **payload: Any) -> Mapping[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"Dawn CNS process exited ({self.process.returncode})")
        self._id += 1
        request = {"id": self._id, "command": command, **payload}
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Dawn CNS pipe closed ({self.process.poll()})")
        response = json.loads(line)
        if response.get("id") != self._id or not response.get("ok"):
            raise RuntimeError(f"Dawn CNS request failed: {response}")
        return response

    @staticmethod
    def _encode(value: np.ndarray, shape: tuple[int, ...], name: str) -> str:
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.dtype("<f4"):
            raise ValueError(f"{name} must be float32{shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains nonfinite values")
        return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()

    @staticmethod
    def _decode(value: str, shape: tuple[int, ...], name: str) -> np.ndarray:
        raw = base64.b64decode(value, validate=True)
        array = np.frombuffer(raw, dtype="<f4").copy()
        if array.size != int(np.prod(shape)):
            raise RuntimeError(f"Dawn CNS {name} length differs")
        result = array.reshape(shape)
        if not np.isfinite(result).all():
            raise RuntimeError(f"Dawn CNS {name} contains nonfinite values")
        return result

    def step(
        self,
        optic_rgb: np.ndarray,
        body_afferents: np.ndarray,
        delivered_context: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # One public .01 step contains the frozen two .005 neural substeps.
        response = self._rpc(
            "step",
            optic_base64=self._encode(
                optic_rgb, (RESIDENTS, OPTIC_SITES, 3), "optic_rgb"
            ),
            body_base64=self._encode(
                body_afferents, (RESIDENTS, BODY_AFFERENTS), "body_afferents"
            ),
            context_base64=self._encode(
                delivered_context, (RESIDENTS, 12), "delivered_context"
            ),
        )
        return (
            self._decode(response["latent_base64"], (RESIDENTS, LATENT), "latent"),
            self._decode(response["motor_base64"], (RESIDENTS, MOTOR), "motor"),
        )

    def reset(self) -> None:
        self._rpc("reset")

    def snapshot(self) -> bytes:
        return base64.b64decode(self._rpc("snapshot")["snapshot_base64"], validate=True)

    def restore(self, snapshot: bytes) -> None:
        self._rpc("restore", snapshot_base64=base64.b64encode(snapshot).decode())

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._rpc("close")
            finally:
                self.process.wait(timeout=30)


class _Bundle:
    def __init__(self, world, cns, teacher, evaluator, metadata) -> None:
        self.world = world
        self.cns = cns
        self.teacher = teacher
        self.evaluator = evaluator
        self.metadata = metadata


def create_bundle(arguments: Any, plan: Plan) -> CollectionBundle:
    """Build one actual MuJoCo world with the M2 full-graph Dawn CNS."""
    world = NodeActualFlyWorld(arguments, plan)
    # The existing --service pass-through names the authenticated WebGPU pack
    # for this factory; no collector CLI or data contract change is required.
    cns = DawnFullCNS(Path(arguments.service), str(arguments.node))
    author = SampledAuthorSteps(Path(arguments.author_source))
    teacher = FlyCurriculumTeacher(Path(arguments.body_schema), author)
    fixture = world.ready["fixture"]
    ecology_ids = tuple(str(item["ecology_id"]) for item in fixture["bodies"])
    metadata = {
        "source_revision": str(arguments.source_revision),
        "morphology_source_revision": fixture["source_revision"],
        "body_schema_sha256": fixture["sensory_schema_sha256"],
        "morphology_sha256": fixture["morphology_sha256"],
        "motor_atlas_sha256": sha256_file(Path(arguments.motor_atlas).resolve()),
        "cns_service_sha256": cns.metadata["cns_service_sha256"],
        "cns_adapter_sha256": cns.metadata["adapter_sha256"],
        "motor_calibration_sha256": cns.metadata["motor_calibration_sha256"],
        "retina_mapping_sha256": cns.metadata["atlas_sha256"],
        "scene_manifest_sha256": world.ready["fixture_sha256"],
        "scene_xml_sha256": world.ready["scene_xml_sha256"],
        "core_wasm_sha256": world.ready["core_wasm_sha256"],
        "author_trajectory_bank_sha256": sha256_file(Path(arguments.author_source).resolve()),
        "author_trajectory_manifest_sha256": sha256_file(
            Path(arguments.author_source).with_name("manifest.json")
        ),
        "cns_format": cns.metadata["format"],
        "cns_backend": cns.metadata["cns_backend"],
        "cns_webgpu_manifest_sha256": cns.metadata["webgpu_manifest_sha256"],
        "cns_source_revision": cns.metadata["cns_source_revision"],
        "native_world_engine": world.ready["engine"],
    }
    try:
        return _Bundle(
            world,
            cns,
            teacher,
            ActualOutcomeEvaluator(ecology_ids),
            metadata,
        )
    except Exception:
        cns.close()
        world.close()
        raise
