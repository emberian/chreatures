#!/usr/bin/env python3
"""Serve a persistent full-graph MaleCNS brain over localhost HTTP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from chreatures.remote_brain import RemoteBrain, SequencedBrain


MAX_REQUEST_BYTES = 1 << 20


def mapping_config(graph: Any, path: Path | None):
    if path is None:
        return None, None
    config = json.loads(path.read_text())
    allowed = {"inputs", "input_gains", "readouts", "readout_gains"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError(f"mapping JSON keys must be a subset of {sorted(allowed)}")
    if "inputs" not in config or "readouts" not in config:
        raise ValueError("mapping JSON needs inputs and readouts")
    inputs = graph.build_input_map(
        config["inputs"], gains=config.get("input_gains", 1.0)
    )
    readouts = graph.build_readout_map(
        config["readouts"], gains=config.get("readout_gains", 1.0)
    )
    return inputs, readouts


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def port_bundle_config(graph: Any, bundle_path: Path, spec_path: Path | None):
    from chreatures.neural_ports import NeuralPortBundle, load_port_spec

    bundle = NeuralPortBundle.load(bundle_path, graph)
    bundle_sha256 = _sha256(bundle_path)
    if spec_path is not None:
        semantic_spec = load_port_spec(spec_path)
        semantic_hash = hashlib.sha256(
            json.dumps(
                semantic_spec, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if semantic_hash != bundle.spec_hash:
            raise ValueError("port bundle does not match the requested port spec")
        document = json.loads(spec_path.read_text(encoding="utf-8"))
        built = document.get("built_artifact", {})
        if built:
            if built.get("sha256") != bundle_sha256:
                raise ValueError("port bundle checksum differs from the port spec")
            if int(built.get("bytes", -1)) != bundle_path.stat().st_size:
                raise ValueError("port bundle size differs from the port spec")
    metadata = {
        "mode": "versioned_bundle",
        "name": bundle.spec["name"],
        "spec_hash": bundle.spec_hash,
        "bundle_sha256": bundle_sha256,
        "bundle_bytes": bundle_path.stat().st_size,
    }
    kwargs = bundle.remote_brain_kwargs()
    return kwargs["input_map"], kwargs["readout_map"], metadata


def handler_type(
    sequenced: SequencedBrain, server_metadata: dict[str, Any]
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ChreaturesMaleCNS/1"
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(30)

        def _json(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
            if content_type != "application/json":
                raise ValueError("Content-Type must be application/json")
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body is too large")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _respond(self, status: HTTPStatus, value: Any) -> None:
            body = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path == "/v1/health":
                with sequenced.lock:
                    self._respond(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "next_seq": sequenced.next_sequence,
                            "residents": sequenced.brain.resident_ids,
                        },
                    )
            elif self.path == "/v1/metadata":
                with sequenced.lock:
                    self._respond(
                        HTTPStatus.OK,
                        {
                            **server_metadata,
                            "next_seq": sequenced.next_sequence,
                            "brain": sequenced.brain.metadata(),
                        },
                    )
            else:
                self._respond(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            shutdown = False
            try:
                request = self._json()
                sequence = request.get("seq")
                if self.path == "/v1/residents/create":
                    seq, result = sequenced.mutate(
                        sequence,
                        lambda: sequenced.brain.add_residents(request.get("resident_ids", [])),
                    )
                    response = {"seq": seq, "slots": result}
                elif self.path == "/v1/residents/remove":
                    ids = request.get("resident_ids", [])
                    seq, _ = sequenced.mutate(
                        sequence, lambda: sequenced.brain.remove_residents(ids)
                    )
                    response = {"seq": seq, "removed": ids}
                elif self.path == "/v1/step":
                    seq, result = sequenced.mutate(
                        sequence,
                        lambda: sequenced.brain.step(
                            request.get("residents", []), float(request.get("dt", 0))
                        ),
                    )
                    response = {
                        "seq": seq,
                        "graph_sha256": sequenced.brain.graph_hash,
                        "feature_names": sequenced.brain.readout_names,
                        "residents": result,
                    }
                    if request.get("compact") is True:
                        keys = ("id", "time", "features", "activity", "activity_peak", "support")
                        response["residents"] = [{key: entry[key] for key in keys} for entry in result]
                        response.pop("feature_names")
                elif self.path == "/v1/snapshot":
                    seq, result = sequenced.mutate(
                        sequence,
                        lambda: sequenced.brain.snapshot(
                            sequenced.snapshot_directory,
                            str(request.get("name", "")),
                            request.get("resident_ids"),
                        ),
                    )
                    response = {"seq": seq, "snapshot": result}
                elif self.path == "/v1/restore":
                    seq, result = sequenced.mutate(
                        sequence,
                        lambda: sequenced.brain.restore(
                            sequenced.snapshot_directory,
                            str(request.get("name", "")),
                            request.get("sha256"),
                            request.get("resident_ids"),
                        ),
                    )
                    response = {"seq": seq, "snapshot": result}
                elif self.path == "/v1/shutdown":
                    seq, _ = sequenced.mutate(sequence, lambda: None)
                    response = {"seq": seq, "status": "shutting down"}
                    shutdown = True
                else:
                    self._respond(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                    return
                self._respond(HTTPStatus.OK, response)
                if shutdown:
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._respond(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": type(error).__name__,
                        "message": str(error),
                        "next_seq": sequenced.next_sequence,
                    },
                )
            except FileNotFoundError as error:
                self._respond(
                    HTTPStatus.NOT_FOUND,
                    {"error": type(error).__name__, "message": str(error)},
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(
                f"{self.log_date_time_string()} {self.client_address[0]} {format % args}",
                file=sys.stderr,
                flush=True,
            )

    return Handler


def write_pid(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = int(path.read_text().strip())
            os.kill(existing, 0)
        except (ValueError, ProcessLookupError):
            pass
        else:
            raise RuntimeError(f"PID file belongs to live process {existing}: {path}")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{os.getpid()}\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--mapping-json", type=Path)
    parser.add_argument("--port-bundle", type=Path)
    parser.add_argument("--port-spec", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.16)
    parser.add_argument("--gain", type=float, default=0.92)
    parser.add_argument("--support-recovery", type=float, default=0.024)
    parser.add_argument("--feature-count", type=int)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    args = parser.parse_args()
    if args.bind not in {"127.0.0.1", "localhost"}:
        parser.error("non-loopback binding is disabled; use an SSH tunnel")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if args.mapping_json is not None and args.port_bundle is not None:
        parser.error("--mapping-json and --port-bundle are mutually exclusive")
    if args.port_spec is not None and args.port_bundle is None:
        parser.error("--port-spec requires --port-bundle")

    from chreatures.malecns import MaleCNSGraph

    graph = MaleCNSGraph.load(args.graph, mmap=True)
    if args.port_bundle is not None:
        inputs, readouts, port_metadata = port_bundle_config(
            graph, args.port_bundle, args.port_spec
        )
    else:
        inputs, readouts = mapping_config(graph, args.mapping_json)
        port_metadata = {
            "mode": "selector_config" if args.mapping_json else "default",
            "name": args.mapping_json.stem if args.mapping_json else "malecns-default-16x48",
        }
    brain = RemoteBrain(
        graph,
        capacity=args.capacity,
        device=args.device,
        tau=args.tau,
        gain=args.gain,
        support_recovery=args.support_recovery,
        input_map=inputs,
        readout_map=readouts,
        port_metadata=port_metadata,
    )
    expected_features = args.feature_count
    if expected_features is None:
        expected_features = 384 if args.port_bundle is not None else 48
    if len(brain.readout_names) != expected_features:
        raise ValueError(
            f"readout map has {len(brain.readout_names)} features; "
            f"controller requires {expected_features}"
        )
    sequenced = SequencedBrain(brain, args.snapshot_dir)
    server_metadata = {
        "pid": os.getpid(),
        "bind": args.bind,
        "port": args.port,
        "snapshot_directory": str(args.snapshot_dir.resolve()),
        "pid_file": str(args.pid_file.resolve()),
    }
    server = ThreadingHTTPServer((args.bind, args.port), handler_type(sequenced, server_metadata))
    try:
        write_pid(args.pid_file)
    except Exception:
        server.server_close()
        raise

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps({**server_metadata, "brain": brain.metadata()}, sort_keys=True), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        try:
            if args.pid_file.read_text().strip() == str(os.getpid()):
                args.pid_file.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
