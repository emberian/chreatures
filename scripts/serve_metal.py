#!/usr/bin/env python3
"""Serve the persistent local Metal MaleCNS backend over the existing API."""

from __future__ import annotations
import argparse, json, os, socket, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chreatures.metal_circuit import MetalCircuit


class Sequenced:
    def __init__(self, brain, snapshots):
        self.brain = brain
        self.snapshots = Path(snapshots)
        self.next_sequence = 0
        self.lock = threading.RLock()

    def mutate(self, seq, operation):
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ValueError("seq must be an integer")
        with self.lock:
            if seq != self.next_sequence:
                raise ValueError(f"expected seq {self.next_sequence}, received {seq}")
            result = operation()
            self.next_sequence += 1
            return seq, result


def handler_type(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ChreaturesMetalCNS/1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(30)

        def body(self):
            if (
                self.headers.get("Content-Type", "").split(";", 1)[0]
                != "application/json"
            ):
                raise ValueError("Content-Type must be application/json")
            n = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= n <= 1 << 20:
                raise ValueError("invalid request size")
            value = json.loads(self.rfile.read(n))
            if not isinstance(value, dict):
                raise ValueError("request must be an object")
            return value

        def send_json(self, status, value):
            raw = (
                json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            with state.lock:
                if self.path == "/v1/health":
                    self.send_json(
                        200,
                        {
                            "status": "ok",
                            "next_seq": state.next_sequence,
                            "residents": state.brain.resident_ids,
                        },
                    )
                elif self.path == "/v1/metadata":
                    self.send_json(
                        200,
                        {
                            "backend": "metal-local-v1",
                            "next_seq": state.next_sequence,
                            "brain": state.brain.metadata(),
                        },
                    )
                else:
                    self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            shutdown = False
            try:
                q = self.body()
                seq = q.get("seq")
                if self.path == "/v1/residents/create":
                    s, r = state.mutate(
                        seq,
                        lambda: state.brain.add_residents(q.get("resident_ids", [])),
                    )
                    answer = {"seq": s, "slots": r}
                elif self.path == "/v1/residents/remove":
                    ids = q.get("resident_ids", [])
                    s, _ = state.mutate(seq, lambda: state.brain.remove_residents(ids))
                    answer = {"seq": s, "removed": ids}
                elif self.path == "/v1/step":
                    s, r = state.mutate(
                        seq,
                        lambda: state.brain.step(
                            q.get("residents", []), float(q.get("dt", 0))
                        ),
                    )
                    answer = {
                        "seq": s,
                        "graph_sha256": state.brain.graph_hash,
                        "feature_names": state.brain.readout_names,
                        "residents": r,
                    }
                    if q.get("compact") is True:
                        keys = (
                            "id",
                            "time",
                            "features",
                            "activity",
                            "activity_peak",
                            "support",
                        )
                        answer["residents"] = [{k: x[k] for k in keys} for x in r]
                        answer.pop("feature_names")
                elif self.path == "/v1/snapshot":
                    s, r = state.mutate(
                        seq,
                        lambda: state.brain.snapshot(
                            state.snapshots,
                            str(q.get("name", "")),
                            q.get("resident_ids"),
                        ),
                    )
                    answer = {"seq": s, "snapshot": r}
                elif self.path == "/v1/restore":
                    s, r = state.mutate(
                        seq,
                        lambda: state.brain.restore(
                            state.snapshots,
                            str(q.get("name", "")),
                            q.get("sha256"),
                            q.get("resident_ids"),
                        ),
                    )
                    answer = {"seq": s, "snapshot": r}
                elif self.path == "/v1/shutdown":
                    s, _ = state.mutate(seq, lambda: None)
                    answer = {"seq": s, "status": "shutting down"}
                    shutdown = True
                else:
                    self.send_json(404, {"error": "unknown endpoint"})
                    return
                self.send_json(200, answer)
                if shutdown:
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
            except (
                FileNotFoundError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                json.JSONDecodeError,
            ) as e:
                self.send_json(
                    400,
                    {
                        "error": type(e).__name__,
                        "message": str(e),
                        "next_seq": state.next_sequence,
                    },
                )

        def log_message(self, fmt, *args):
            if self.path == "/v1/step" and len(args) > 1 and str(args[1]) == "200":
                return
            print(
                f"{self.log_date_time_string()} {fmt % args}",
                file=sys.stderr,
                flush=True,
            )

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifact", type=Path, default=ROOT / "data/metal-brain/metal-csr-v2.bin"
    )
    p.add_argument(
        "--port-bundle", type=Path, default=ROOT / "data/ports/retinal-v1-maps.npz"
    )
    p.add_argument("--snapshot-dir", type=Path, required=True)
    p.add_argument("--pid-file", type=Path, required=True)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--kernel", choices=("row", "simd"), default="row")
    a = p.parse_args()
    if a.bind not in {"127.0.0.1", "localhost"}:
        p.error("only loopback binding is supported")
    a.pid_file.parent.mkdir(parents=True, exist_ok=True)
    if a.pid_file.exists():
        try:
            os.kill(int(a.pid_file.read_text()), 0)
        except (ValueError, ProcessLookupError):
            pass
        else:
            raise RuntimeError(f"PID file belongs to a live process: {a.pid_file}")
    a.pid_file.write_text(f"{os.getpid()}\n")
    try:
        with MetalCircuit(a.artifact, a.port_bundle, kernel=a.kernel) as brain:
            ThreadingHTTPServer(
                (a.bind, a.port), handler_type(Sequenced(brain, a.snapshot_dir))
            ).serve_forever()
    finally:
        a.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
