"""Ordered client for a dedicated persistent neural service over an SSH tunnel."""
from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

import numpy as np

from .brain import CHANNELS


class NeuralServiceError(RuntimeError):
    pass


class NeuralClient:
    def __init__(self, url="http://127.0.0.1:18765", timeout=10):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("Neural service must be reached through a localhost connection/SSH tunnel")
        self.url = url.rstrip("/")
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.timeout = timeout
        self.uncertain = False
        self.metadata = self._request("GET", "/v1/metadata")
        self.next_seq = self.metadata["next_seq"]
        self.graph = self.metadata["brain"]["graph"]
        self.input_names = self.metadata["brain"]["inputs"]
        self.output_names = self.metadata["brain"]["readouts"]
        if self.input_names != CHANNELS:
            raise ValueError("Remote sensory map differs from this explicitly versioned interface")

    def _request(self, method, path, value=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        payload = None if value is None else json.dumps(value, allow_nan=False).encode()
        try:
            connection.request(method, path, body=payload, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = json.loads(response.read())
            if response.status >= 400:
                raise NeuralServiceError(data.get("message", data.get("error", "Remote neural request rejected")))
            return data
        finally:
            connection.close()

    def mutate(self, path, **value):
        if self.uncertain:
            raise NeuralServiceError("An earlier neural request has unconfirmed outcome; restore the last whole-world checkpoint")
        try:
            response = self._request("POST", path, {"seq": self.next_seq, **value})
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            self.uncertain = True
            raise NeuralServiceError("Lost neural response; world must pause to preserve causal order") from exc
        if response.get("seq") != self.next_seq:
            self.uncertain = True
            raise NeuralServiceError("Neural response sequence mismatch")
        self.next_seq += 1
        return response

    def create(self, ids):
        return self.mutate("/v1/residents/create", resident_ids=ids)

    def step(self, entries, dt):
        return self.mutate("/v1/step", residents=entries, dt=dt)["residents"]

    def snapshot(self, name):
        return self.mutate("/v1/snapshot", name=name)["snapshot"]

    def restore(self, receipt):
        return self.mutate("/v1/restore", name=receipt["name"], sha256=receipt["sha256"])


def sensory_channels(senses):
    """Sensory transduction without object identities or a food-color prior."""
    odor = np.asarray(senses["odor"], dtype=np.float32).reshape(2, 3)
    retina = np.asarray(senses["vision"], dtype=np.float32).reshape(16, 4)
    # Proximity is a raw sensory coordinate here; an adaptive policy learns its
    # consequences. There is no reflex that labels all seen objects obstacles.
    values = np.concatenate((odor.ravel(), [retina[:8, 3].max(initial=0), retina[8:, 3].max(initial=0)],
                             retina[:, :3].mean(axis=0), senses.get("sound", [0, 0, 0]),
                             [senses.get("shade", 0), np.max(senses.get("touch", [0, 0]))]))
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("Invalid physical sensory observations")
    return dict(zip(CHANNELS, np.clip(values, 0, 1).astype(float).tolist()))
