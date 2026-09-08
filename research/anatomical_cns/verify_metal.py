#!/usr/bin/env python3
"""Run the canonical three-tick CNS V3 Torch/Metal parity experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

STATE_NAMES = ("rates", "adaptation", "support", "release", "mod_da", "mod_oa", "mod_ht")
TOLERANCE = 2e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_snapshot(path: Path):
    with path.open("rb") as stream:
        if stream.read(9) != b"CNSSTATE3":
            raise ValueError("snapshot is not CNSSTATE3")
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
        times = np.frombuffer(stream.read(8 * header["capacity"]), "<f8").copy()
        raw = np.frombuffer(stream.read(), "<f4")
    stride = 165_122 * header["storage_tiles"] * 4
    if raw.size != 7 * stride or not np.isfinite(raw).all() or not np.isfinite(times).all():
        raise ValueError("snapshot state has the wrong extent or contains nonfinite values")
    fields = [raw[i * stride : (i + 1) * stride].reshape(165_122, header["storage_tiles"], 4)[:, 0, 0] for i in range(7)]
    return header, times, fields


def finite_max_error(actual, expected, name: str) -> float:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"{name} shape differs: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{name} contains a nonfinite value")
    error = float(np.max(np.abs(actual - expected)))
    if not np.isfinite(error):
        raise ValueError(f"{name} produced a nonfinite maximum error")
    return error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    fixture = np.load(args.fixture)
    metadata = json.loads(str(fixture["metadata_json"]))
    ticks = fixture["sensory"].shape[0]
    if ticks != 3 or fixture["sensory"].shape != (3, 1, 5423) or fixture["context"].shape != (3, 1, 12):
        raise ValueError("canonical fixture must contain three capacity-one V3 ticks")

    errors = {}
    with tempfile.TemporaryDirectory(prefix="chreatures-metal-v3-") as directory:
        snapshot = Path(directory) / "state.bin"
        restored = Path(directory) / "restored.bin"
        process = subprocess.Popen(
            [str(args.binary), str(args.service), "row", "1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

        def receive():
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Metal server exited without a reply")
            reply = json.loads(line)
            if not reply.get("ok"):
                raise RuntimeError(reply)
            return reply

        def send(request):
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
            return receive()

        try:
            ready = receive()
            for tick in range(ticks):
                reply = send({
                    "op": "step", "dt": float(metadata.get("dt", 0.05)), "active_mask": 1,
                    "sensory": fixture["sensory"][tick, 0].tolist(),
                    "context": fixture["context"][tick, 0].tolist(),
                })
                snapshot.unlink(missing_ok=True)
                send({"op": "snapshot", "path": str(snapshot), "metadata": f"parity-tick-{tick}"})
                _, _, fields = read_snapshot(snapshot)
                for name, actual in zip(STATE_NAMES, fields):
                    errors[f"t{tick + 1}.{name}"] = finite_max_error(actual, fixture[f"state.{name}"][tick, :, 0], name)
                errors[f"t{tick + 1}.latent"] = finite_max_error(reply["latent"], fixture["latent"][tick, 0], "latent")
                errors[f"t{tick + 1}.motor"] = finite_max_error(reply["motor"], fixture["motor"][tick, 0], "motor")
            snapshot.unlink(missing_ok=True)
            send({"op": "snapshot", "path": str(snapshot), "metadata": "restore-exact"})
            send({"op": "reset", "mask": 1})
            send({"op": "restore", "path": str(snapshot), "mask": 1})
            send({"op": "snapshot", "path": str(restored), "metadata": "restore-exact"})
            snapshot_hash, restored_hash = sha256(snapshot), sha256(restored)
            exact = snapshot_hash == restored_hash
            send({"op": "shutdown"})
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    overall = max(errors.values())
    report = {
        "service": str(args.service.resolve()), "service_sha256": sha256(args.service),
        "fixture": str(args.fixture.resolve()), "fixture_sha256": sha256(args.fixture),
        "binary": str(args.binary.resolve()), "ready": ready, "ticks": ticks,
        "max_abs_errors": errors, "overall_max_abs": overall, "tolerance": TOLERANCE,
        "snapshot_restore_byte_exact": exact, "snapshot_sha256": snapshot_hash,
        "restored_snapshot_sha256": restored_hash,
    }
    if not np.isfinite(overall) or overall >= TOLERANCE or not exact:
        report["passed"] = False
    else:
        report["passed"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
