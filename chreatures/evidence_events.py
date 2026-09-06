"""Bounded publication of committed mechanism receipts, outside organism memory.

This is metadata at the world/checkpoint boundary. Simulation and numerical
state remain in their existing native owners. Sequence gaps are explicit so a
recorder cannot mistake a bounded view for an exhaustive lifetime archive.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import deque

from .checkpoint import canonical

FORMAT = "chreatures-committed-evidence-events-v1"
CAPACITY = 2048
ZERO_HASH = "0" * 64


def _payload(value):
    if not isinstance(value, dict) or set(value) - {
        "kind", "actors", "quantities", "details", "source", "blob_refs",
    }:
        raise ValueError("invalid committed event payload")
    if not isinstance(value.get("kind"), str) or not 1 <= len(value["kind"]) <= 80:
        raise ValueError("committed event requires a mechanism kind")
    actors = value.get("actors", {"bodies": [], "entities": []})
    if not isinstance(actors, dict) or set(actors) != {"bodies", "entities"}:
        raise ValueError("event actors must separate bodies and entities")
    for ids in actors.values():
        if not isinstance(ids, list) or len(ids) > 256 or any(
            not isinstance(item, str) or not 1 <= len(item) <= 256 for item in ids
        ) or len(set(ids)) != len(ids):
            raise ValueError("invalid event actor identities")
    quantities = value.get("quantities", [])
    if not isinstance(quantities, list) or len(quantities) > 128:
        raise ValueError("invalid event quantities")
    for quantity in quantities:
        if not isinstance(quantity, dict) or set(quantity) != {"name", "value", "unit"}:
            raise ValueError("event quantity requires name, value and unit")
        if any(not isinstance(quantity[key], str) or not quantity[key] for key in ("name", "unit")):
            raise ValueError("event quantity requires explicit units")
        scalar = quantity["value"]
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)) or not math.isfinite(scalar):
            raise ValueError("event quantity must be finite")
    source = value.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("stream"), str) or not source["stream"]:
        raise ValueError("event requires a source mechanism")
    result = {"kind": value["kind"], "actors": actors, "quantities": quantities,
              "details": value.get("details", {}), "source": source,
              "blob_refs": value.get("blob_refs", [])}
    if not isinstance(result["details"], dict) or not isinstance(result["blob_refs"], list):
        raise ValueError("event details and blob references must be structured metadata")
    # The public boundary accepts JSON only; prohibit NaNs and unbounded blobs.
    import json
    encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode()) > 32768:
        raise ValueError("event metadata exceeds the public receipt budget")
    return copy.deepcopy(result)


class CommittedEvents:
    def __init__(self, world_id):
        if not isinstance(world_id, str) or not world_id:
            raise ValueError("event stream requires a world identity")
        self.world_id = world_id
        self.sequence = 0
        self.head = ZERO_HASH
        self.rows = deque(maxlen=CAPACITY)

    def append(self, payloads, *, tick, model_time):
        if type(tick) is not int or tick < 0 or not math.isfinite(model_time) or model_time < 0:
            raise ValueError("invalid committed event time")
        if self.rows and (tick < self.rows[-1]["tick"] or model_time < self.rows[-1]["model_time"]):
            raise ValueError("event time cannot precede committed history")
        clean = [_payload(value) for value in payloads]
        if len(clean) > CAPACITY:
            raise ValueError("one committed event batch exceeds journal capacity")
        # Validate the whole batch before advancing its sequence or hash chain.
        prepared = []
        sequence, previous = self.sequence, self.head
        for payload in clean:
            sequence += 1
            row = {**payload, "event_id": f"{self.world_id}:event:{sequence}",
                   "sequence": sequence, "tick": tick, "model_time": float(model_time),
                   "previous_sha256": previous}
            row["sha256"] = hashlib.sha256(canonical(row)).hexdigest()
            previous = row["sha256"]
            prepared.append(row)
        self.rows.extend(prepared)
        self.sequence, self.head = sequence, previous
        return copy.deepcopy(prepared)

    def view(self):
        return {"format": FORMAT, "world_id": self.world_id,
                "first_sequence": self.rows[0]["sequence"] if self.rows else self.sequence + 1,
                "last_sequence": self.sequence, "retired_events": max(0, self.sequence - len(self.rows)),
                "head_sha256": self.head, "capacity": CAPACITY,
                "events": copy.deepcopy(list(self.rows))}

    def snapshot(self):
        return self.view()

    @classmethod
    def restore(cls, value, world_id):
        if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("world_id") != world_id:
            raise ValueError("event snapshot identity differs")
        rows = value.get("events")
        sequence = value.get("last_sequence")
        if not isinstance(rows, list) or type(sequence) is not int or sequence < 0 or len(rows) != min(CAPACITY, sequence):
            raise ValueError("invalid event snapshot sequence")
        result = cls(world_id)
        previous = None
        for index, original in enumerate(rows):
            row = copy.deepcopy(original)
            digest = row.pop("sha256", None)
            if hashlib.sha256(canonical(row)).hexdigest() != digest:
                raise ValueError("event snapshot hash differs")
            expected = sequence - len(rows) + index + 1
            if row.get("sequence") != expected or row.get("event_id") != f"{world_id}:event:{expected}":
                raise ValueError("event snapshot has a sequence gap")
            _payload({key: row[key] for key in ("kind", "actors", "quantities", "details", "source", "blob_refs")})
            if type(row.get("tick")) is not int or row["tick"] < 0 or not math.isfinite(row.get("model_time", math.nan)) or row["model_time"] < 0:
                raise ValueError("invalid event snapshot time")
            if previous and (row["previous_sha256"] != previous["sha256"] or row["tick"] < previous["tick"] or row["model_time"] < previous["model_time"]):
                raise ValueError("event snapshot chain or time differs")
            row["sha256"] = digest
            previous = row
            result.rows.append(row)
        result.sequence = sequence
        result.head = rows[-1]["sha256"] if rows else ZERO_HASH
        if result.view() != value:
            raise ValueError("event snapshot metadata differs")
        return result
