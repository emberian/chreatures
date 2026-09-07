"""Ordered client for a dedicated persistent neural service over an SSH tunnel."""
from __future__ import annotations

import hashlib
import http.client
import json
import time
from urllib.parse import urlencode, urlsplit

from .metal_circuit import IDENTITY_KEYS, input_names


class NeuralServiceError(RuntimeError):
    pass


class NeuralClient:
    def __init__(self, url="http://127.0.0.1:18765", timeout=10, snapshot_timeout=60):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("Neural service must be reached through a localhost connection/SSH tunnel")
        self.url = url.rstrip("/")
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.timeout = timeout
        self.snapshot_timeout = snapshot_timeout
        self.uncertain = False
        self._connection = None
        self._last_request = 0.0
        self.metadata = self._request("GET", "/v1/metadata")
        self.receipt_protocol = self.metadata.get("receipt_protocol")
        self.service_incarnation = self.metadata.get("service_incarnation")
        if self.receipt_protocol == "chreatures-request-receipt-v1" and (
            not isinstance(self.service_incarnation, str)
            or len(self.service_incarnation) != 32
            or any(c not in "0123456789abcdef" for c in self.service_incarnation)
        ):
            raise ValueError("Neural receipt protocol lacks a valid service identity")
        self.next_seq = self.metadata["next_seq"]
        self.graph = self.metadata["brain"]["graph"]
        self.input_names = self.metadata["brain"]["inputs"]
        self.output_names = self.metadata["brain"]["readouts"]
        self.cns_identity = self.metadata["brain"].get("cns_adapter")
        if (self.input_names != input_names()
                or self.output_names != [f"cns.latent.{i}" for i in range(512)]
                or not isinstance(self.cns_identity, dict)
                or self.cns_identity.get("format") != "chreatures-cns-service-v1"
                or self.cns_identity.get("sensory_dim") != 5356
                or self.cns_identity.get("latent_dim") != 512):
            raise ValueError("Neural service does not implement the current CNS-only interface")
        for key in (*IDENTITY_KEYS, "service_artifact_sha256"):
            value = self.cns_identity.get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError(f"Neural service has invalid {key}")
        if self.cns_identity["graph_sha256"] != self.graph["sha256"]:
            raise ValueError("Neural service graph and learned adapter disagree")
        if self.receipt_protocol != "chreatures-request-receipt-v1":
            raise ValueError("Current neural service requires authenticated mutation receipts")

    def _request(self, method, path, value=None, *, timeout=None):
        # Reuse the SSH-forwarded transport while actively stepping, but discard
        # idle connections before a new mutation. Never retry a sent mutation.
        if self._connection is not None and time.monotonic() - self._last_request > 5:
            self._connection.close()
            self._connection = None
        request_timeout = self.timeout if timeout is None else timeout
        if self._connection is None:
            self._connection = http.client.HTTPConnection(self.host, self.port, timeout=request_timeout)
        connection = self._connection
        connection.timeout = request_timeout
        if connection.sock is not None:
            connection.sock.settimeout(request_timeout)
        payload = None if value is None else json.dumps(value, allow_nan=False).encode()
        try:
            connection.request(method, path, body=payload, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = json.loads(response.read())
            if not isinstance(data, dict):
                raise NeuralServiceError("Neural response must be a JSON object")
            self._last_request = time.monotonic()
            if response.status >= 400:
                raise NeuralServiceError(data.get("message", data.get("error", "Remote neural request rejected")))
            return data
        except Exception:
            connection.close()
            self._connection = None
            raise

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def available_resident_slots(self) -> int | None:
        """Read capacity before a funded birth; never allocate or resize a brain."""
        metadata = self._request("GET", "/v1/metadata")
        if metadata.get("service_incarnation") != self.service_incarnation:
            raise NeuralServiceError("Neural service changed before resident birth")
        brain = metadata["brain"]
        capacity = brain.get("capacity")
        if capacity is None:
            return None  # Dynamically allocated backends have no fixed slot pool.
        residents = brain.get("residents")
        if (type(capacity) is not int or capacity < 0
                or not isinstance(residents, list)
                or len(residents) > capacity):
            raise NeuralServiceError("Neural resident capacity metadata differs")
        return capacity - len(residents)

    def mutate(self, path, **value):
        if self.uncertain:
            raise NeuralServiceError("An earlier neural request has unconfirmed outcome; restore the last whole-world checkpoint")
        body = {"seq": self.next_seq, **value}
        digest = None
        if self.receipt_protocol == "chreatures-request-receipt-v1":
            digest = hashlib.sha256(json.dumps(
                {"method": "POST", "path": path, "body": body},
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                allow_nan=False,
            ).encode()).hexdigest()
            body["request_sha256"] = digest
        try:
            response = self._request(
                "POST", path, body,
                timeout=self.snapshot_timeout
                if path in {"/v1/snapshot", "/v1/restore"} else self.timeout,
            )
        except (OSError, http.client.HTTPException, json.JSONDecodeError, NeuralServiceError) as exc:
            self.uncertain = True
            # Read the original committed response; never repeat the mutation.
            # A failed operation may have changed state. Only a matching commit
            # in this same service incarnation is sufficient to continue.
            response = self._committed_receipt(digest) if digest is not None else None
            if response is None:
                raise NeuralServiceError("Lost neural response; world must pause to preserve causal order") from exc
        if (
            type(response.get("seq")) is not int
            or response["seq"] != self.next_seq
            or (digest is not None and (
                response.get("service_incarnation") != self.service_incarnation
                or response.get("request_sha256") != digest
            ))
        ):
            self.uncertain = True
            raise NeuralServiceError("Neural response request identity mismatch")
        self.uncertain = False
        self.next_seq += 1
        return response

    def _committed_receipt(self, digest):
        query = urlencode({
            "incarnation": self.service_incarnation,
            "seq": self.next_seq,
            "request_sha256": digest,
        })
        try:
            receipt = self._request("GET", f"/v1/receipt?{query}")
        except (OSError, http.client.HTTPException, json.JSONDecodeError, NeuralServiceError):
            return None
        if (
            receipt.get("status") != "committed"
            or receipt.get("service_incarnation") != self.service_incarnation
            or type(receipt.get("seq")) is not int
            or receipt["seq"] != self.next_seq
            or receipt.get("request_sha256") != digest
            or type(receipt.get("next_seq")) is not int
            or receipt["next_seq"] != self.next_seq + 1
            or not isinstance(receipt.get("response"), dict)
        ):
            return None
        return receipt["response"]

    def create(self, residents):
        """Bind fresh private neural state to its immutable learned CNS adapter."""
        if not isinstance(residents, list) or not residents or any(
            not isinstance(row, dict) or set(row) != {"id", "cns_adapter_sha256"}
            for row in residents
        ):
            raise ValueError("neural birth requires resident identities and a learned CNS adapter")
        return self.mutate("/v1/residents/create", residents=residents)

    def step(self, entries, dt):
        return self.mutate("/v1/step", residents=entries, dt=dt, compact=True)["residents"]

    def capture_rates(self, name, resident_id):
        return self.mutate("/v1/observer/capture-rates", name=name, resident_id=resident_id)["capture"]

    def snapshot(self, name, ids=None):
        values = {"name": name}
        if ids is not None and self.metadata["brain"].get("cns_adapter"):
            values["resident_ids"] = ids
        return self.mutate("/v1/snapshot", **values)["snapshot"]

    def restore(self, receipt):
        values = {"name": receipt["name"], "sha256": receipt["sha256"]}
        if receipt.get("scope") == "cohort":
            values["resident_ids"] = receipt["residents"]
        return self.mutate("/v1/restore", **values)
