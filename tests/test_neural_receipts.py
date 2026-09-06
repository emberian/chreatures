"""Lost replies must recover the committed response, never repeat neural work."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from chreatures.neural_client import NeuralClient, NeuralServiceError


def client_with_lost_reply(change=None, *, protocol=True):
    client = NeuralClient.__new__(NeuralClient)
    client.next_seq = 17
    client.uncertain = False
    client.timeout = 0.1
    client.snapshot_timeout = 60
    client.receipt_protocol = "chreatures-request-receipt-v1" if protocol else None
    client.service_incarnation = "a" * 32 if protocol else None
    calls = []
    original = {}

    def request(method, path, value=None, *, timeout=None):
        calls.append((method, path, value, timeout))
        if method == "POST":
            if protocol:
                body = {key: val for key, val in value.items() if key != "request_sha256"}
                digest = hashlib.sha256(json.dumps(
                    {"method": method, "path": path, "body": body},
                    sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode()).hexdigest()
                assert value["request_sha256"] == digest
                original.update({
                    "seq": 17, "request_sha256": digest,
                    "service_incarnation": client.service_incarnation,
                    "snapshot": {"name": body["name"], "sha256": "b" * 64},
                })
            raise TimeoutError("Committed, but response was lost")
        query = parse_qs(urlsplit(path).query)
        assert query["seq"] == ["17"]
        assert query["request_sha256"] == [original["request_sha256"]]
        receipt = {
            "status": "committed", "next_seq": 18,
            "seq": 17, "service_incarnation": client.service_incarnation,
            "request_sha256": original["request_sha256"],
            "response": dict(original),
        }
        if change:
            change(receipt)
        return receipt

    client._request = request
    return client, calls, original


def test_lost_snapshot_response_recovers_without_second_post():
    client, calls, original = client_with_lost_reply()
    answer = client.mutate("/v1/snapshot", name="world-test")
    assert answer == original
    assert [call[0] for call in calls] == ["POST", "GET"]
    assert calls[0][3] == 60
    assert client.next_seq == 18
    assert not client.uncertain


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="in_progress"),
    lambda r: r.update(status="failed"),
    lambda r: r.update(status="unknown"),
    lambda r: r.update(status="stale_incarnation"),
    lambda r: r.update(service_incarnation="c" * 32),
    lambda r: r.update(request_sha256="d" * 64),
    lambda r: r.update(seq=16),
    lambda r: r.update(next_seq=19),
    lambda r: r["response"].update(seq=16),
    lambda r: r["response"].update(service_incarnation="c" * 32),
    lambda r: r["response"].update(request_sha256="d" * 64),
])
def test_ambiguous_or_different_commit_stays_paused(change):
    client, calls, _ = client_with_lost_reply(change)
    with pytest.raises(NeuralServiceError):
        client.mutate("/v1/snapshot", name="world-test")
    assert client.next_seq == 17
    assert client.uncertain
    with pytest.raises(NeuralServiceError, match="unconfirmed"):
        client.mutate("/v1/snapshot", name="world-test")
    assert [call[0] for call in calls] == ["POST", "GET"]


def test_service_without_receipts_never_guesses_after_timeout():
    client, calls, _ = client_with_lost_reply(protocol=False)
    with pytest.raises(NeuralServiceError):
        client.mutate("/v1/snapshot", name="world-test")
    assert client.uncertain
    assert client.next_seq == 17
    assert [call[0] for call in calls] == ["POST"]
