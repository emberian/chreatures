#!/usr/bin/env python3
"""Serve bounded creature-field-of-view perception over loopback HTTP."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from chreatures.perception import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PerceptionService,
    SmolVLMBackend,
    UnavailableBackend,
)


MAX_REQUEST_BYTES = 5_500_000


def build_backend(args: argparse.Namespace):
    if args.backend == "off":
        return UnavailableBackend("perception backend explicitly disabled", "off")
    if args.model_path is None:
        return UnavailableBackend("--model-path is required for smolvlm2", "smolvlm2")
    try:
        return SmolVLMBackend(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as error:
        return UnavailableBackend(
            f"model initialization failed: {type(error).__name__}: {error}",
            "smolvlm2",
        )


def handler_type(
    perception: PerceptionService, request_timeout: float
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ChreaturesPerception/1"

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
            body = (
                json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path == "/v1/health":
                self._respond(HTTPStatus.OK, perception.metadata())
            else:
                self._respond(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/v1/perceive":
                self._respond(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                response = perception.submit(self._json()).result(timeout=request_timeout)
                self._respond(HTTPStatus.OK, response)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                self._respond(
                    HTTPStatus.BAD_REQUEST,
                    {"error": type(error).__name__, "message": str(error)},
                )
            except FutureTimeout:
                self._respond(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    {
                        "error": "inference_timeout",
                        "message": f"perception exceeded {request_timeout} seconds",
                    },
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("off", "smolvlm2"), default="off")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-pending", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--pid-file", type=Path, default=Path("runs/perception.pid"))
    args = parser.parse_args()
    if args.bind not in {"127.0.0.1", "localhost"}:
        parser.error("non-loopback binding is disabled; use an SSH tunnel")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if not 16 <= args.max_new_tokens <= 512:
        parser.error("max-new-tokens must be in 16..512")
    if args.request_timeout <= 0:
        parser.error("request-timeout must be positive")

    backend = build_backend(args)
    perception = PerceptionService(
        backend, max_workers=args.max_workers, max_pending=args.max_pending
    )
    write_pid(args.pid_file)
    server = ThreadingHTTPServer(
        (args.bind, args.port), handler_type(perception, args.request_timeout)
    )

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "bind": args.bind,
                "port": args.port,
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "perception": perception.metadata(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        perception.close()
        try:
            if args.pid_file.read_text().strip() == str(os.getpid()):
                args.pid_file.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
