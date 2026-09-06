import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode

from scripts.serve_metal import Sequenced, handler_type, request_sha256


class FakeBrain:
    graph_hash = "graph"

    def __init__(self):
        self.resident_ids = []
        self.readout_names = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.snapshot_calls = 0

    def metadata(self):
        return {"fake": True}

    def add_residents(self, resident_ids):
        self.resident_ids.extend(resident_ids)
        return {name: index for index, name in enumerate(resident_ids)}

    def snapshot(self, _directory, name, _resident_ids):
        self.snapshot_calls += 1
        if name == "failed":
            raise ValueError("synthetic failure")
        self.started.set()
        self.release.wait(timeout=2)
        return {"name": name, "sha256": "receipt", "bytes": 7}


def request(port, method, path, body=None, timeout=2):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    raw = None if body is None else json.dumps(body, separators=(",", ":"))
    headers = {} if raw is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=raw, headers=headers)
    response = connection.getresponse()
    value = json.loads(response.read())
    connection.close()
    return response.status, value


def receipt_path(state, seq, digest):
    return "/v1/receipt?" + urlencode(
        {
            "incarnation": state.incarnation,
            "seq": seq,
            "request_sha256": digest,
        }
    )


def test_lost_response_receipt_is_queryable_without_reexecution(tmp_path):
    brain = FakeBrain()
    state = Sequenced(brain, tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type(state))
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    port = server.server_address[1]
    try:
        status, metadata = request(port, "GET", "/v1/metadata")
        assert status == 200
        assert metadata["receipt_protocol"] == "chreatures-request-receipt-v1"
        assert metadata["service_incarnation"] == state.incarnation

        # Legacy callers remain accepted and receive the server-computed identity.
        status, created = request(
            port, "POST", "/v1/residents/create", {"seq": 0, "resident_ids": ["a"]}
        )
        assert status == 200 and created["seq"] == 0
        assert len(created["request_sha256"]) == 64

        body = {"seq": 1, "name": "slow", "resident_ids": ["a"]}
        digest = request_sha256("/v1/snapshot", body)
        body["request_sha256"] = digest
        lost = []

        def timed_out_request():
            try:
                request(port, "POST", "/v1/snapshot", body, timeout=0.03)
            except TimeoutError:
                lost.append(True)

        caller = threading.Thread(target=timed_out_request)
        caller.start()
        assert brain.started.wait(timeout=1)
        status, pending = request(port, "GET", receipt_path(state, 1, digest))
        assert status == 200 and pending["status"] == "in_progress"
        time.sleep(0.05)
        caller.join(timeout=1)
        assert lost == [True]
        brain.release.set()
        time.sleep(0.03)

        status, committed = request(port, "GET", receipt_path(state, 1, digest))
        assert status == 200
        assert committed["status"] == "committed"
        assert committed["next_seq"] == 2
        assert committed["response"]["snapshot"]["name"] == "slow"
        assert committed["response"]["request_sha256"] == digest
        assert brain.snapshot_calls == 1

        _, older = request(
            port, "GET", receipt_path(state, 0, created["request_sha256"])
        )
        assert older["status"] == "committed"
        assert older["next_seq"] == 2
        assert older["response"] == created

        failed_body = {"seq": 2, "name": "failed", "resident_ids": ["a"]}
        failed_hash = request_sha256("/v1/snapshot", failed_body)
        failed_body["request_sha256"] = failed_hash
        status, failure = request(port, "POST", "/v1/snapshot", failed_body)
        assert status == 400 and failure["next_seq"] == 2
        _, failed = request(port, "GET", receipt_path(state, 2, failed_hash))
        assert failed["status"] == "failed"
        assert failed["next_seq"] == 2

        unknown_hash = "0" * 64
        _, unknown = request(port, "GET", receipt_path(state, 2, unknown_hash))
        assert unknown["status"] == "unknown"
        stale_path = receipt_path(state, 1, digest).replace(state.incarnation, "0" * 32)
        _, stale = request(port, "GET", stale_path)
        assert stale["status"] == "stale_incarnation"
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)
