"""Nonblocking client for persistent, body-view native feature inference."""

from __future__ import annotations

import base64
import binascii
from concurrent.futures import Future, ThreadPoolExecutor
import copy
import hashlib
import json
import math
import re
import threading
import time
from typing import Any, Iterable
import urllib.error
import urllib.request

from .perception import (
    DENSE_POOLING_VERSION,
    EmbedRequest,
    MAX_FRAME_BYTES,
    MAX_FRAMES,
    MAX_TOTAL_FRAME_BYTES,
    MODEL_REVISION,
    SENSOR_ID,
)


SAFE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
SAFE_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _causal_values(value: Any, name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 32:
        raise ValueError(f"{name} must be a mapping with at most 32 values")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not SAFE_FIELD.fullmatch(key):
            raise ValueError(f"invalid {name} field")
        result[key] = _finite(item, f"{name}.{key}")
    return result


class AsyncPerceptionClient:
    """One-cohort background client whose full delivery state is checkpointable.

    `submit_cohort` never waits for the network or model. Completed observations
    remain capture-time facts; `take_completed` labels them current only when
    their captured world sequence exactly matches the caller's current sequence.
    """

    VERSION = 3

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 180.0,
        min_interval_seconds: float = 1.0,
        min_interval_ticks: int = 0,
        queue_capacity: int = 1,
    ):
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an HTTP(S) URL")
        if _finite(timeout_seconds, "timeout_seconds") <= 0:
            raise ValueError("timeout_seconds must be positive")
        if _finite(min_interval_seconds, "min_interval_seconds") < 0:
            raise ValueError("min_interval_seconds must be nonnegative")
        if (
            not isinstance(min_interval_ticks, int)
            or isinstance(min_interval_ticks, bool)
            or min_interval_ticks < 0
        ):
            raise ValueError("min_interval_ticks must be a nonnegative integer")
        if queue_capacity != 1:
            raise ValueError("perception queue_capacity is fixed at one cohort")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.min_interval_seconds = float(min_interval_seconds)
        self.min_interval_ticks = min_interval_ticks
        self.queue_capacity = queue_capacity
        self.records: dict[str, dict[str, Any]] = {}
        self.last_accepted_at = -math.inf
        self.last_accepted_tick: int | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="perception-client"
        )
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._closed = False

    def submit_cohort(
        self,
        request_id: str,
        observations: Iterable[dict[str, Any]],
        *,
        delivery_tick: int | None = None,
    ) -> dict[str, Any]:
        """Freeze and queue one body-view cohort without blocking the caller."""

        if not isinstance(request_id, str) or not SAFE_ID.fullmatch(request_id):
            raise ValueError("request_id must be a safe identifier")
        rows = list(observations)
        if not 1 <= len(rows) <= MAX_FRAMES:
            raise ValueError(f"cohort must contain 1-{MAX_FRAMES} observations")
        payload_rows, causal_rows = [], []
        total_bytes = 0
        capture_ids = set()
        capture_ticks = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each cohort observation must be a mapping")
            allowed = {
                "sensor_id", "world_sequence", "model_time", "captured_at",
                "provenance", "png", "action", "outcome",
            }
            if set(row) - allowed:
                raise ValueError(f"unsupported observation keys: {sorted(set(row)-allowed)}")
            sensor_id = row.get("sensor_id")
            if not isinstance(sensor_id, str) or not SENSOR_ID.fullmatch(sensor_id):
                raise ValueError("sensor_id must be a safe identifier")
            sequence = row.get("world_sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
                raise ValueError("world_sequence must be a nonnegative integer")
            capture_id = (sensor_id, sequence)
            if capture_id in capture_ids:
                raise ValueError("sensor/tick pairs must be unique within a cohort")
            capture_ids.add(capture_id)
            capture_ticks.append(sequence)
            png = row.get("png")
            if not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("png must contain PNG bytes")
            if len(png) > MAX_FRAME_BYTES:
                raise ValueError("png exceeds per-frame byte limit")
            total_bytes += len(png)
            digest = hashlib.sha256(png).hexdigest()
            source = {
                "sensor_id": sensor_id,
                "world_sequence": sequence,
                "model_time": _finite(row.get("model_time"), "model_time"),
                "captured_at": _finite(row.get("captured_at"), "captured_at"),
                "provenance": row.get("provenance", "resident_fov"),
            }
            if source["provenance"] != "resident_fov":
                raise ValueError("running-world client accepts resident_fov only")
            payload_rows.append(
                {
                    "source": source,
                    "frame": {
                        "mime_type": "image/png",
                        "data_base64": base64.b64encode(png).decode(),
                        "sha256": digest,
                    },
                }
            )
            causal_rows.append(
                {
                    "sensor_id": sensor_id,
                    "world_sequence": sequence,
                    "frame_sha256": digest,
                    "action": _causal_values(row.get("action"), "action"),
                    "outcome": _causal_values(row.get("outcome"), "outcome"),
                }
            )
        if total_bytes > MAX_TOTAL_FRAME_BYTES:
            raise ValueError("cohort exceeds total frame byte limit")
        if delivery_tick is not None:
            if (
                not isinstance(delivery_tick, int)
                or isinstance(delivery_tick, bool)
                or delivery_tick < max(capture_ticks)
            ):
                raise ValueError("delivery_tick must be at or after every capture tick")
        capture_tick = max(capture_ticks)
        now = time.time()
        with self._lock:
            if self._closed:
                return {"status": "unavailable", "reason": "client is closed"}
            if request_id in self.records:
                return {"status": "duplicate", "request_id": request_id}
            if any(record["state"] == "pending" for record in self.records.values()):
                return {"status": "busy", "reason": "one cohort is already pending"}
            if (
                self.min_interval_ticks
                and self.last_accepted_tick is not None
                and capture_tick - self.last_accepted_tick < self.min_interval_ticks
            ):
                return {
                    "status": "rate_limited",
                    "basis": "model_tick",
                    "next_capture_tick": self.last_accepted_tick
                    + self.min_interval_ticks,
                }
            remaining = self.min_interval_seconds - (now - self.last_accepted_at)
            if delivery_tick is None and remaining > 0:
                return {"status": "rate_limited", "retry_after_seconds": remaining}
            record = {
                "request_id": request_id,
                "state": "pending",
                "queued_at": now,
                "attempts": 0,
                "capture_tick": capture_tick,
                "delivery_tick": delivery_tick,
                "payload": {"request_id": request_id, "observations": payload_rows},
                "causality": causal_rows,
                "response": None,
                "response_sha256": None,
                "response_bytes_base64": None,
            }
            self.records[request_id] = record
            self.last_accepted_at = now
            self.last_accepted_tick = capture_tick
            self._dispatch(request_id)
        return {
            "status": "accepted",
            "request_id": request_id,
            "cohort_size": len(rows),
            "delivery_tick": delivery_tick,
        }

    def record_outcome(
        self,
        request_id: str,
        sensor_id: str,
        outcome: dict[str, Any],
        *,
        world_sequence: int | None = None,
    ) -> None:
        """Attach the later bodily consequence to its frozen capture/action."""

        values = _causal_values(outcome, "outcome")
        with self._lock:
            record = self.records.get(request_id)
            if record is None:
                raise KeyError(request_id)
            if record["state"] == "delivered":
                raise ValueError("cannot change causality after delivery")
            matches = [
                row for row in record["causality"]
                if row["sensor_id"] == sensor_id
                and (world_sequence is None or row["world_sequence"] == world_sequence)
            ]
            if len(matches) != 1:
                raise KeyError((sensor_id, world_sequence))
            matches[0]["outcome"] = values

    def retry_failed(self, request_id: str) -> dict[str, Any]:
        """Reissue a frozen scheduled request that produced no usable feature."""

        with self._lock:
            record = self.records.get(request_id)
            if record is None:
                raise KeyError(request_id)
            if record["delivery_tick"] is None:
                raise ValueError("retry_failed is limited to scheduled requests")
            if (
                record["state"] != "completed"
                or (record.get("response") or {}).get("status") == "ok"
            ):
                raise ValueError("scheduled request has no completed failure to retry")
            if any(item["state"] == "pending" for item in self.records.values()):
                return {"status": "busy", "reason": "one cohort is already pending"}
            record["state"] = "pending"
            record["response"] = None
            record["response_sha256"] = None
            record["response_bytes_base64"] = None
            self._dispatch(request_id)
            return {
                "status": "accepted",
                "request_id": request_id,
                "delivery_tick": record["delivery_tick"],
                "attempt": record["attempts"],
            }

    def _dispatch(self, request_id: str) -> None:
        record = self.records[request_id]
        record["attempts"] += 1
        future = self._executor.submit(self._post, copy.deepcopy(record["payload"]))
        self._futures[request_id] = future
        future.add_done_callback(
            lambda completed, identifier=request_id: self._finish(identifier, completed)
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint + "/v1/embed",
            data=_canonical(payload),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = {"message": str(error)}
            return {"status": "transport_error", "http_status": error.code, "detail": detail}
        except (OSError, TimeoutError) as error:
            return {"status": "transport_error", "reason": f"{type(error).__name__}: {error}"}
        if not isinstance(value, dict):
            return {"status": "transport_error", "reason": "embed response was not an object"}
        return value

    def _finish(self, request_id: str, future: Future[dict[str, Any]]) -> None:
        try:
            response = future.result()
            self._validate_response(request_id, response)
        except Exception as error:
            response = {"status": "invalid_response", "reason": f"{type(error).__name__}: {error}"}
        with self._lock:
            record = self.records.get(request_id)
            if record is not None and record["state"] == "pending":
                response_bytes = _canonical(response)
                record["response"] = response
                record["response_sha256"] = hashlib.sha256(response_bytes).hexdigest()
                record["response_bytes_base64"] = base64.b64encode(
                    response_bytes
                ).decode()
                record["state"] = "completed"
            self._futures.pop(request_id, None)

    def _validate_response(self, request_id: str, response: dict[str, Any]) -> None:
        if response.get("status") != "ok":
            return
        if response.get("request_id") != request_id:
            raise ValueError("response request_id differs")
        if response.get("model", {}).get("revision") != MODEL_REVISION:
            raise ValueError("response model revision differs")
        pooling = response.get("pooling")
        if not isinstance(pooling, dict) or pooling.get("version") != DENSE_POOLING_VERSION:
            raise ValueError("response pooling contract differs")
        record = self.records[request_id]
        expected = record["payload"]["observations"]
        actual = response.get("observations")
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("response cohort differs")
        for expected_row, actual_row in zip(expected, actual):
            feature = actual_row.get("feature") if isinstance(actual_row, dict) else None
            if (
                actual_row.get("source") != expected_row["source"]
                or actual_row.get("frame_sha256") != expected_row["frame"]["sha256"]
                or not isinstance(feature, dict)
                or feature.get("dimension") != 960
                or feature.get("dtype") != "float32-le"
                or not isinstance(feature.get("values"), list)
                or len(feature["values"]) != 960
            ):
                raise ValueError("response observation contract differs")
            import numpy as np

            vector = np.asarray(feature["values"], dtype="<f4")
            if not np.isfinite(vector).all():
                raise ValueError("response feature is nonfinite")
            if hashlib.sha256(vector.tobytes()).hexdigest() != feature.get("sha256"):
                raise ValueError("response feature hash differs")

    def _deliver(
        self, record: dict[str, Any], current_sequences: dict[str, int]
    ) -> dict[str, Any]:
        response = record["response"]
        result = {
            "request_id": record["request_id"],
            "capture_tick": record["capture_tick"],
            "delivery_tick": record["delivery_tick"],
            "response_sha256": record["response_sha256"],
            "status": response.get("status"),
            "completed_at": response.get("completed_at"),
            "latency_seconds": response.get("latency_seconds"),
            "model": copy.deepcopy(response.get("model")),
            "pooling": copy.deepcopy(response.get("pooling")),
            "observations": [],
        }
        for observation, causal in zip(
            response.get("observations", []), record["causality"]
        ):
            source = observation["source"]
            current = current_sequences.get(source["sensor_id"])
            if current is None:
                relation = "unresolved"
            elif source["world_sequence"] == current:
                relation = "current"
            elif source["world_sequence"] < current:
                relation = "historical"
            else:
                relation = "future"
            result["observations"].append(
                {
                    **copy.deepcopy(observation),
                    "temporal_relation": relation,
                    "usable_as_current_perception": relation == "current",
                    "historical_experience": relation == "historical",
                    "causal_input": copy.deepcopy(causal["action"]),
                    "causal_outcome": copy.deepcopy(causal["outcome"]),
                }
            )
        record["state"] = "delivered"
        record["delivered_at"] = time.time()
        return result

    def take_completed(
        self, current_sequences: dict[str, int]
    ) -> list[dict[str, Any]]:
        """Take unscheduled completions; scheduled cohorts require their slot."""

        if not isinstance(current_sequences, dict):
            raise ValueError("current_sequences must be a sensor-to-sequence mapping")
        with self._lock:
            return [
                self._deliver(record, current_sequences)
                for record in self.records.values()
                if record["state"] == "completed" and record["delivery_tick"] is None
            ]

    def take_scheduled(
        self, current_tick: int, current_sequences: dict[str, int]
    ) -> dict[str, Any]:
        """Poll a deterministic delivery slot without waiting inside this client."""

        if (
            not isinstance(current_tick, int)
            or isinstance(current_tick, bool)
            or current_tick < 0
        ):
            raise ValueError("current_tick must be a nonnegative integer")
        if not isinstance(current_sequences, dict):
            raise ValueError("current_sequences must be a sensor-to-sequence mapping")
        with self._lock:
            scheduled = [
                record for record in self.records.values()
                if record["delivery_tick"] is not None
                and record["state"] != "delivered"
            ]
            missed = [
                record for record in scheduled
                if record["delivery_tick"] < current_tick
            ]
            if missed:
                return {
                    "status": "missed_delivery",
                    "current_tick": current_tick,
                    "requests": [
                        {
                            "request_id": record["request_id"],
                            "delivery_tick": record["delivery_tick"],
                            "inference_state": record["state"],
                        }
                        for record in missed
                    ],
                    "cohorts": [],
                }
            due = [record for record in scheduled if record["delivery_tick"] == current_tick]
            waiting = [
                record for record in due
                if record["state"] == "pending"
                or (record.get("response") or {}).get("status") != "ok"
            ]
            if waiting:
                return {
                    "status": "awaiting",
                    "current_tick": current_tick,
                    "delivery_tick": current_tick,
                    "requests": [
                        {
                            "request_id": record["request_id"],
                            "inference_state": record["state"],
                            "response_status": (
                                (record.get("response") or {}).get("status")
                            ),
                            "retryable": record["state"] == "completed",
                        }
                        for record in waiting
                    ],
                    "cohorts": [],
                }
            if due:
                cohorts = [self._deliver(record, current_sequences) for record in due]
                return {
                    "status": "delivered",
                    "current_tick": current_tick,
                    "delivery_tick": current_tick,
                    "cohorts": cohorts,
                }
            future_ticks = sorted(
                {record["delivery_tick"] for record in scheduled
                 if record["delivery_tick"] > current_tick}
            )
            return {
                "status": "not_due" if future_ticks else "idle",
                "current_tick": current_tick,
                "next_delivery_tick": future_ticks[0] if future_ticks else None,
                "cohorts": [],
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts = {state: 0 for state in ("pending", "completed", "delivered")}
            for record in self.records.values():
                counts[record["state"]] += 1
            return {"closed": self._closed, **counts}

    def prune_delivered(self, keep: int = 1) -> None:
        """Bound transport history after its features enter private memory.

        Pending and completed-but-undelivered responses are never discarded.
        The organism must preserve its own episode and provenance first.
        """
        if type(keep) is not int or keep < 0:
            raise ValueError("keep must be a nonnegative integer")
        with self._lock:
            delivered = [key for key, row in self.records.items() if row["state"] == "delivered"]
            for key in delivered[:max(0, len(delivered) - keep)]:
                del self.records[key]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.VERSION,
                "endpoint": self.endpoint,
                "timeout_seconds": self.timeout_seconds,
                "min_interval_seconds": self.min_interval_seconds,
                "min_interval_ticks": self.min_interval_ticks,
                "queue_capacity": self.queue_capacity,
                "last_accepted_at": (
                    self.last_accepted_at if math.isfinite(self.last_accepted_at) else None
                ),
                "last_accepted_tick": self.last_accepted_tick,
                "records": copy.deepcopy(list(self.records.values())),
            }

    @classmethod
    def restore(cls, value: Any, *, reissue_pending: bool = True) -> "AsyncPerceptionClient":
        if not isinstance(value, dict) or value.get("version") not in {2, cls.VERSION}:
            raise ValueError("unsupported perception client checkpoint")
        legacy = value["version"] == 2
        client = cls(
            value["endpoint"],
            timeout_seconds=value["timeout_seconds"],
            min_interval_seconds=value["min_interval_seconds"],
            min_interval_ticks=value.get("min_interval_ticks", 0),
            queue_capacity=value["queue_capacity"],
        )
        if value["last_accepted_at"] is not None:
            client.last_accepted_at = _finite(
                value["last_accepted_at"], "last_accepted_at"
            )
        last_tick = value.get("last_accepted_tick")
        if last_tick is not None:
            if not isinstance(last_tick, int) or isinstance(last_tick, bool) or last_tick < 0:
                raise ValueError("last_accepted_tick must be a nonnegative integer")
            client.last_accepted_tick = last_tick
        records = value.get("records")
        if not isinstance(records, list):
            raise ValueError("perception client records must be a list")
        for record in records:
            if not isinstance(record, dict) or record.get("state") not in {
                "pending", "completed", "delivered"
            }:
                raise ValueError("invalid perception client record")
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not SAFE_ID.fullmatch(request_id):
                raise ValueError("invalid checkpoint request_id")
            if request_id in client.records:
                raise ValueError("duplicate checkpoint request_id")
            payload = record.get("payload")
            # Server-side parsing authenticates the full frozen payload again
            # when pending work is reissued.
            if not isinstance(payload, dict) or payload.get("request_id") != request_id:
                raise ValueError("invalid checkpoint embed payload")
            parsed = EmbedRequest.from_mapping(payload)
            capture_tick = max(row.world_sequence for row in parsed.observations)
            if legacy:
                record["capture_tick"] = capture_tick
                record["delivery_tick"] = None
            if record.get("capture_tick") != capture_tick:
                raise ValueError("checkpoint capture tick differs from frozen cohort")
            delivery_tick = record.get("delivery_tick")
            if delivery_tick is not None and (
                not isinstance(delivery_tick, int)
                or isinstance(delivery_tick, bool)
                or delivery_tick < capture_tick
            ):
                raise ValueError("invalid checkpoint delivery tick")
            causality = record.get("causality")
            if not isinstance(causality, list) or len(causality) != len(parsed.observations):
                raise ValueError("checkpoint causality differs from frozen cohort")
            for causal, observation in zip(causality, parsed.observations):
                if legacy:
                    causal["world_sequence"] = observation.world_sequence
                if (
                    not isinstance(causal, dict)
                    or causal.get("sensor_id") != observation.sensor_id
                    or causal.get("world_sequence") != observation.world_sequence
                    or causal.get("frame_sha256") != observation.frame_sha256
                ):
                    raise ValueError("checkpoint causality source differs")
                causal["action"] = _causal_values(causal.get("action"), "action")
                causal["outcome"] = _causal_values(causal.get("outcome"), "outcome")
            if record["state"] in {"completed", "delivered"}:
                response = record.get("response")
                if not isinstance(response, dict):
                    raise ValueError("completed checkpoint has no response")
                expected_hash = hashlib.sha256(_canonical(response)).hexdigest()
                if record.get("response_sha256") != expected_hash:
                    raise ValueError("checkpoint response hash differs")
                response_bytes = _canonical(response)
                encoded = record.get("response_bytes_base64")
                if legacy:
                    encoded = base64.b64encode(response_bytes).decode()
                    record["response_bytes_base64"] = encoded
                try:
                    decoded = base64.b64decode(encoded, validate=True)
                except (binascii.Error, TypeError, ValueError) as error:
                    raise ValueError("invalid checkpoint response bytes") from error
                if decoded != response_bytes:
                    raise ValueError("checkpoint response bytes differ")
            client.records[request_id] = copy.deepcopy(record)
            if record["state"] in {"completed", "delivered"}:
                client._validate_response(request_id, record["response"])
        pending = [key for key, record in client.records.items() if record["state"] == "pending"]
        if client.last_accepted_tick is None and client.records:
            client.last_accepted_tick = max(
                record["capture_tick"] for record in client.records.values()
            )
        if len(pending) > client.queue_capacity:
            raise ValueError("checkpoint exceeds perception queue capacity")
        if reissue_pending:
            for request_id in pending:
                client._dispatch(request_id)
        return client

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["AsyncPerceptionClient"]
