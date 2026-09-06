#!/usr/bin/env python3
"""Read-only acquisition and paired continuation of the frozen 8a801fc life."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import sys
import threading
import time
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPLOYMENT = Path("/Users/ember/paperbin/chreatures/deployments/population-v4-8a801fc")
SOURCE = DEPLOYMENT / "source"
AUTHORITATIVE_WORLD = DEPLOYMENT / "run/world/checkpoint.json"
AUTHORITATIVE_BRAIN = DEPLOYMENT / "run/brain/snapshots"
ARTIFACT = Path("/Users/ember/dev/chreatures/data/metal-brain/metal-csr-retinal-v2.bin")
PORT_BUNDLE = Path("/Users/ember/dev/chreatures/data/ports/retinal-v2-maps.npz")
RESIDENT_ARTIFACT = DEPLOYMENT / "data/genomes/developmental-resident-population-v4.npz"
BINARY = DEPLOYMENT / "native/metal-brain/metal-brain-server"
PORT = 19782
STEPS = 32

sys.path.insert(0, str(SOURCE))
from chreatures.checkpoint import canonical  # noqa: E402
from chreatures.metal_circuit import MetalCircuit  # noqa: E402
from chreatures.runtime3d import Habitat3D  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def clone(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        os.clonefile(source, temporary)
    except AttributeError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def acquire_pair() -> tuple[Path, Path, dict]:
    target = ROOT / "source-pair"
    target.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        world = target / "checkpoint.acquired.json"
        clone(AUTHORITATIVE_WORLD, world)
        envelope = json.loads(world.read_text())
        state = envelope.get("state")
        if (
            envelope.get("format") != "chreatures-developmental-habitat-checkpoint-v3"
            or not isinstance(state, dict)
            or hashlib.sha256(canonical(state)).hexdigest() != envelope.get("sha256")
        ):
            continue
        reference = state.get("neural_snapshot", {})
        name = reference.get("name")
        if not isinstance(name, str) or not name.startswith("world-"):
            continue
        source_brain = AUTHORITATIVE_BRAIN / f"{name}.npz"
        if not source_brain.is_file():
            time.sleep(0.2)
            continue
        brain = target / f"{name}.npz"
        clone(source_brain, brain)
        if (
            sha256(brain) == reference.get("sha256")
            and brain.stat().st_size == reference.get("bytes")
        ):
            return world, brain, {
                "attempt": attempt,
                "tick": state["tick"],
                "world_file_sha256": sha256(world),
                "world_state_sha256": envelope["sha256"],
                "brain_file_sha256": reference["sha256"],
                "brain_bytes": reference["bytes"],
                "source_world": str(AUTHORITATIVE_WORLD),
                "source_brain": str(source_brain),
            }
    raise RuntimeError("could not acquire a coherent world/neural checkpoint pair")


def load_service_module():
    path = SOURCE / "scripts/serve_metal.py"
    spec = importlib.util.spec_from_file_location("frozen_serve_metal", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_port_free() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", PORT))


def run_once(label: str, checkpoint: Path, brain_source: Path, service_module) -> dict:
    run_root = ROOT / f"run-{label}"
    snapshots = run_root / "brain/snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    brain_copy = snapshots / brain_source.name
    clone(brain_source, brain_copy)
    assert sha256(brain_copy) == sha256(brain_source)
    assert_port_free()
    brain = MetalCircuit(
        ARTIFACT,
        PORT_BUNDLE,
        capacity=32,
        binary=BINARY,
        kernel="simd",
    )
    state = service_module.Sequenced(brain, snapshots)
    server = ThreadingHTTPServer(
        ("127.0.0.1", PORT), service_module.handler_type(state)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fork_id = f"research-fork-{label}-{uuid.uuid4().hex}"
    started = time.perf_counter()
    try:
        habitat = Habitat3D.load(
            checkpoint,
            brain_url=f"http://127.0.0.1:{PORT}",
            resident_artifact=RESIDENT_ARTIFACT,
        )
        source_world_id = habitat.id
        habitat.branch = "research-fork:population-v4-continuity-check"
        habitat.step(STEPS)
        final_path = run_root / "final-checkpoint.json"
        state_sha256 = habitat.save(final_path)
        final = json.loads(final_path.read_text())
        final_brain = snapshots / f"{final['state']['neural_snapshot']['name']}.npz"
        if sha256(final_brain) != final["state"]["neural_snapshot"]["sha256"]:
            raise ValueError("final neural snapshot checksum differs")
        return {
            "fork_id": fork_id,
            "source_world_id": source_world_id,
            "runtime_world_id_semantics": "inherited checkpoint identifier; not an authoritative live-world claim",
            "service_incarnation": state.incarnation,
            "elapsed_seconds": time.perf_counter() - started,
            "final_checkpoint": str(final_path),
            "final_checkpoint_file_sha256": sha256(final_path),
            "final_state_sha256": state_sha256,
            "final_neural_snapshot": str(final_brain),
            "final_neural_sha256": sha256(final_brain),
            "final_neural_bytes": final_brain.stat().st_size,
            "tick": final["state"]["tick"],
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        brain.close()


def first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return {"path": path, "left_type": type(left).__name__, "right_type": type(right).__name__}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {"path": path, "left_keys": sorted(left), "right_keys": sorted(right)}
        for key in sorted(left):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return {"path": path, "left": repr(left)[:240], "right": repr(right)[:240]}
    return None


def main() -> None:
    checkpoint, brain_source, acquisition = acquire_pair()
    deployment = json.loads((DEPLOYMENT / "deployment-receipt.json").read_text())
    frozen_hashes = {
        "deployment_receipt_sha256": sha256(DEPLOYMENT / "deployment-receipt.json"),
        "source_revision": deployment["source"]["git_revision"],
        "metal_brain_server_sha256": sha256(BINARY),
        "metal_graph_artifact_sha256": sha256(ARTIFACT),
        "port_bundle_sha256": sha256(PORT_BUNDLE),
        "resident_artifact_sha256": sha256(RESIDENT_ARTIFACT),
        "cognitive_core_sha256": sha256(SOURCE / "_cognitive_core.cpython-312-darwin.so"),
        "world_kernels_sha256": sha256(SOURCE / "_world_kernels.cpython-312-darwin.so"),
    }
    service_module = load_service_module()
    runs = [
        run_once("a", checkpoint, brain_source, service_module),
        run_once("b", checkpoint, brain_source, service_module),
    ]
    left = json.loads(Path(runs[0]["final_checkpoint"]).read_text())
    right = json.loads(Path(runs[1]["final_checkpoint"]).read_text())
    difference = first_difference(left, right)
    exact_world = difference is None
    exact_neural = runs[0]["final_neural_sha256"] == runs[1]["final_neural_sha256"]
    receipt = {
        "format": "chreatures-population-v4-coupled-continuation-check-v1",
        "status": "passed" if exact_world and exact_neural else "failed",
        "research_scope": {
            "kind": "isolated_research_fork",
            "authoritative_world_mutated": False,
            "source_runtime_advanced": False,
            "continuation_steps": STEPS,
            "research_port": PORT,
            "execution": "two sequential runs; one Python process plus one native Metal child per run",
        },
        "acquisition": acquisition,
        "frozen_runtime": frozen_hashes,
        "runs": runs,
        "comparison": {
            "complete_world_checkpoint_exact": exact_world,
            "complete_neural_snapshot_exact": exact_neural,
            "first_world_difference": difference,
            "excluded_from_comparison": [
                "receipt-only research fork IDs",
                "receipt-only service incarnation IDs",
                "elapsed wall-clock durations",
            ],
            "runtime_state_fields_excluded": [],
        },
        "command": (
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="
            f"{SOURCE} /Users/ember/dev/chreatures/.venv/bin/python {Path(__file__).resolve()}"
        ),
        "limitations": [
            "This demonstrates deterministic 32-tick continuation for one frozen eight-resident checkpoint on one Apple M2 Max runtime.",
            "The runtime world UUID is necessarily preserved to address its neural snapshot; receipt-level fork IDs distinguish both research executions.",
            "The check does not compare against a different engine, machine, graph, or checkpoint.",
        ],
    }
    atomic_json(ROOT / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(ROOT / "receipt.json"), "comparison": receipt["comparison"]}, sort_keys=True))
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
