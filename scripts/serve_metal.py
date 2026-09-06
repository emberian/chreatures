#!/usr/bin/env python3
"""Serve the persistent local Metal MaleCNS backend over the existing API."""

from __future__ import annotations
import argparse, hashlib, json, os, re, socket, sys, threading, uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chreatures.metal_circuit import MetalCircuit
from chreatures.mushroom_plasticity import (
    MushroomBodySubstrate,
    MushroomFullGraphBridgeSpec,
)


HASH = re.compile(r"[0-9a-f]{64}\Z")


def request_sha256(path, body):
    unsigned = dict(body)
    unsigned.pop("request_sha256", None)
    canonical = json.dumps(
        {"method": "POST", "path": path, "body": unsigned},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class Sequenced:
    def __init__(self, brain, snapshots):
        self.brain = brain
        self.snapshots = Path(snapshots)
        self.next_sequence = 0
        self.lock = threading.RLock()
        self.receipt_lock = threading.Lock()
        self.incarnation = uuid.uuid4().hex
        self.receipts = OrderedDict()

    def _receipt(self, key, value):
        with self.receipt_lock:
            self.receipts[key] = value
            self.receipts.move_to_end(key)
            while len(self.receipts) > 64:
                self.receipts.popitem(last=False)

    def mutate(self, seq, request_hash, operation):
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ValueError("seq must be an integer")
        with self.lock:
            if seq != self.next_sequence:
                raise ValueError(f"expected seq {self.next_sequence}, received {seq}")
            key = (seq, request_hash)
            base = {
                "service_incarnation": self.incarnation,
                "seq": seq,
                "request_sha256": request_hash,
            }
            self._receipt(key, {**base, "status": "in_progress", "next_seq": seq})
            try:
                result = operation()
            except Exception as error:
                self._receipt(
                    key,
                    {
                        **base,
                        "status": "failed",
                        "next_seq": self.next_sequence,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                )
                raise
            self.next_sequence += 1
            response = {
                **result,
                "service_incarnation": self.incarnation,
                "request_sha256": request_hash,
            }
            self._receipt(
                key,
                {
                    **base,
                    "status": "committed",
                    "next_seq": self.next_sequence,
                    "response": response,
                },
            )
            return response

    def receipt(self, incarnation, seq, request_hash):
        base = {
            "service_incarnation": self.incarnation,
            "seq": seq,
            "request_sha256": request_hash,
            "next_seq": self.next_sequence,
        }
        if incarnation != self.incarnation:
            return {**base, "status": "stale_incarnation"}
        with self.receipt_lock:
            found = self.receipts.get((seq, request_hash))
            return (
                {**found, "next_seq": self.next_sequence}
                if found is not None
                else {**base, "status": "unknown"}
            )


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
            parsed = urlparse(self.path)
            if parsed.path == "/v1/receipt":
                try:
                    query = parse_qs(parsed.query, strict_parsing=True)
                    if set(query) != {"incarnation", "seq", "request_sha256"} or any(
                        len(value) != 1 for value in query.values()
                    ):
                        raise ValueError("receipt query fields differ")
                    incarnation = query["incarnation"][0]
                    request_hash = query["request_sha256"][0]
                    seq_text = query["seq"][0]
                    if not re.fullmatch(r"[0-9a-f]{32}", incarnation) or not HASH.fullmatch(
                        request_hash
                    ) or not re.fullmatch(r"0|[1-9][0-9]*", seq_text):
                        raise ValueError("receipt query identity is malformed")
                    self.send_json(
                        200, state.receipt(incarnation, int(seq_text), request_hash)
                    )
                except ValueError as error:
                    self.send_json(400, {"error": "ValueError", "message": str(error)})
                return
            with state.lock:
                if parsed.path == "/v1/health":
                    self.send_json(
                        200,
                        {
                            "status": "ok",
                            "next_seq": state.next_sequence,
                            "residents": state.brain.resident_ids,
                            "service_incarnation": state.incarnation,
                        },
                    )
                elif parsed.path == "/v1/metadata":
                    self.send_json(
                        200,
                        {
                            "backend": "metal-local-v1",
                            "next_seq": state.next_sequence,
                            "brain": state.brain.metadata(),
                            "service_incarnation": state.incarnation,
                            "receipt_protocol": "chreatures-request-receipt-v1",
                            "receipt_details": {
                                "cache_entries": 64,
                                "hash": "sha256 canonical JSON of method/path/body_without_request_sha256",
                                "query": "/v1/receipt",
                                "failed_semantics": "does_not_certify_no_mutation",
                            },
                        },
                    )
                else:
                    self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            shutdown = False
            try:
                q = self.body()
                seq = q.get("seq")
                known = {
                    "/v1/residents/create",
                    "/v1/residents/remove",
                    "/v1/step",
                    "/v1/snapshot",
                    "/v1/restore",
                    "/v1/shutdown",
                }
                if self.path not in known:
                    self.send_json(404, {"error": "unknown endpoint"})
                    return
                expected_hash = request_sha256(self.path, q)
                supplied_hash = q.get("request_sha256")
                if supplied_hash is not None and (
                    not isinstance(supplied_hash, str) or not HASH.fullmatch(supplied_hash)
                ):
                    raise ValueError("request_sha256 must be 64 lowercase hexadecimal characters")
                if supplied_hash is not None and supplied_hash != expected_hash:
                    raise ValueError("request_sha256 differs from canonical request")
                supplied_hash = expected_hash
                if self.path == "/v1/residents/create":
                    answer = state.mutate(
                        seq,
                        supplied_hash,
                        lambda: {
                            "seq": seq,
                            "slots": state.brain.add_residents(q.get("resident_ids", [])),
                        },
                    )
                elif self.path == "/v1/residents/remove":
                    ids = q.get("resident_ids", [])
                    answer = state.mutate(
                        seq,
                        supplied_hash,
                        lambda: (
                            state.brain.remove_residents(ids),
                            {"seq": seq, "removed": ids},
                        )[1],
                    )
                elif self.path == "/v1/step":
                    def step():
                        residents = state.brain.step(
                            q.get("residents", []), float(q.get("dt", 0))
                        )
                        result = {
                            "seq": seq,
                            "graph_sha256": state.brain.graph_hash,
                            "feature_names": state.brain.readout_names,
                            "residents": residents,
                        }
                        if q.get("compact") is True:
                            keys = ("id", "time", "features", "activity", "activity_peak", "support")
                            result["residents"] = [{k: x[k] for k in keys} for x in residents]
                            result.pop("feature_names")
                        return result

                    answer = state.mutate(seq, supplied_hash, step)
                elif self.path == "/v1/snapshot":
                    answer = state.mutate(
                        seq,
                        supplied_hash,
                        lambda: {
                            "seq": seq,
                            "snapshot": state.brain.snapshot(
                                state.snapshots,
                                str(q.get("name", "")),
                                q.get("resident_ids"),
                            ),
                        },
                    )
                elif self.path == "/v1/restore":
                    answer = state.mutate(
                        seq,
                        supplied_hash,
                        lambda: {
                            "seq": seq,
                            "snapshot": state.brain.restore(
                                state.snapshots,
                                str(q.get("name", "")),
                                q.get("sha256"),
                                q.get("resident_ids"),
                            ),
                        },
                    )
                elif self.path == "/v1/shutdown":
                    answer = state.mutate(
                        seq,
                        supplied_hash,
                        lambda: {"seq": seq, "status": "shutting down"},
                    )
                    shutdown = True
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
                        "service_incarnation": state.incarnation,
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
    p.add_argument("--mushroom-substrate", type=Path)
    p.add_argument("--mushroom-bridge", type=Path)
    p.add_argument(
        "--mushroom-modulator-mode",
        choices=("synthetic", "actual_ppl101_rate"),
        default="synthetic",
    )
    p.add_argument("--mushroom-frozen", action="store_true")
    a = p.parse_args()
    if (a.mushroom_substrate is None) != (a.mushroom_bridge is None):
        p.error("--mushroom-substrate and --mushroom-bridge must be supplied together")
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
        substrate = None
        bridge = None
        if a.mushroom_substrate is not None:
            substrate_receipt = json.loads(
                a.mushroom_substrate.with_suffix(".json").read_text(encoding="utf-8")
            )
            bridge_receipt = json.loads(
                a.mushroom_bridge.with_suffix(".json").read_text(encoding="utf-8")
            )
            substrate = MushroomBodySubstrate.load(
                a.mushroom_substrate,
                expected_sha256=substrate_receipt["sha256"],
            )
            bridge = MushroomFullGraphBridgeSpec.load(
                a.mushroom_bridge,
                expected_sha256=bridge_receipt["sha256"],
            )
        with MetalCircuit(
            a.artifact,
            a.port_bundle,
            kernel=a.kernel,
            mushroom_substrate=substrate,
            mushroom_bridge=bridge,
            mushroom_modulator_mode=a.mushroom_modulator_mode,
            mushroom_plasticity_enabled=not a.mushroom_frozen,
        ) as brain:
            ThreadingHTTPServer(
                (a.bind, a.port), handler_type(Sequenced(brain, a.snapshot_dir))
            ).serve_forever()
    finally:
        a.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
