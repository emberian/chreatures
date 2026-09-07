#!/usr/bin/env python3
"""Create an explicit one-way research continuation from a whole-world checkpoint.

This is deliberately separate from Habitat3D.restore: normal restore continues to
require an identical engine.  The utility authenticates a source checkpoint and
its MBST1 neural snapshot, copies the complete state under a new world identity,
pins a supplied target engine identity, and emits a migration receipt.  Native
neural arrays are copied without interpretation or numerical conversion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import uuid

from chreatures.checkpoint import canonical
from chreatures.evidence_events import CommittedEvents


CHECKPOINT_FORMAT = "chreatures-developmental-habitat-checkpoint-v4"
MIGRATION_FORMAT = "chreatures-research-continuation-migration-v1"
MAGIC = b"MBST1\0\0\0"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_identity(value: object, old: str, new: str) -> object:
    """Replace the authenticated world namespace in JSON identity strings."""
    if isinstance(value, dict):
        return {key: _replace_identity(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_identity(item, old, new) for item in value]
    if isinstance(value, str) and old in value:
        if not (value == old or value.startswith(f"{old}:") or f"world-{old}-" in value):
            raise ValueError(f"world identity appears in an unsupported string: {value!r}")
        return value.replace(old, new)
    return value


def migrate(
    source_checkpoint: Path,
    source_snapshot_dir: Path,
    output_checkpoint: Path,
    output_snapshot_dir: Path,
    target_engine_path: Path,
    target_brain_url: str,
    target_world_id: str,
    from_revision: str,
    to_revision: str,
    reason: str,
) -> dict[str, object]:
    if output_checkpoint.exists():
        raise ValueError("output checkpoint already exists")
    try:
        parsed_world_id = str(uuid.UUID(target_world_id))
    except ValueError as error:
        raise ValueError("target world ID must be a canonical UUID") from error
    if parsed_world_id != target_world_id:
        raise ValueError("target world ID must use canonical UUID spelling")

    envelope_bytes = source_checkpoint.read_bytes()
    envelope = json.loads(envelope_bytes)
    state = envelope.get("state")
    if envelope.get("format") != CHECKPOINT_FORMAT or not isinstance(state, dict):
        raise ValueError("source is not a current whole-world checkpoint")
    source_state_sha = _sha(canonical(state))
    if source_state_sha != envelope.get("sha256"):
        raise ValueError("source checkpoint checksum mismatch")
    if state.get("pending_step") is not None:
        raise ValueError("source checkpoint contains an incomplete distributed tick")
    old_world_id = state.get("id")
    if not isinstance(old_world_id, str) or old_world_id == target_world_id:
        raise ValueError("source and target world identities must be distinct")
    source_body_ids = list(state.get("remote_ids", {}).values())
    if not source_body_ids or any(
        not isinstance(body_id, str) or not body_id.startswith(f"{old_world_id}:")
        for body_id in source_body_ids
    ):
        raise ValueError("source body identities do not use the world namespace")
    body_mapping = {
        body_id: body_id.replace(old_world_id, target_world_id, 1)
        for body_id in source_body_ids
    }

    snapshot_receipt = state.get("neural_snapshot")
    if not isinstance(snapshot_receipt, dict):
        raise ValueError("source checkpoint has no neural snapshot receipt")
    source_snapshot = source_snapshot_dir / f"{snapshot_receipt.get('name')}.npz"
    native = source_snapshot.read_bytes()
    if _sha(native) != snapshot_receipt.get("sha256"):
        raise ValueError("source neural snapshot checksum mismatch")
    if len(native) < 16 or native[:8] != MAGIC:
        raise ValueError("source neural snapshot is not MBST1")
    metadata_len = struct.unpack("<Q", native[8:16])[0]
    if metadata_len > 2_000_000 or 16 + metadata_len > len(native):
        raise ValueError("source neural metadata length is invalid")
    metadata = json.loads(native[16 : 16 + metadata_len])
    payload = native[16 + metadata_len :]
    if [item for item in metadata.get("resident_ids", []) if item is not None] != snapshot_receipt.get("residents"):
        raise ValueError("source neural resident order differs from its receipt")

    target_engine = json.loads(target_engine_path.read_text())
    target_engine_body = copy.deepcopy(target_engine)
    target_engine_sha = target_engine_body.pop("sha256", None)
    if (
        target_engine.get("format") != "chreatures-local-engine-identity-v1"
        or _sha(canonical(target_engine_body)) != target_engine_sha
    ):
        raise ValueError("target engine identity is not self-authenticating")

    changed = _replace_identity(copy.deepcopy(state), old_world_id, target_world_id)
    assert isinstance(changed, dict)
    changed["id"] = target_world_id
    changed["branch"] = f"research-continuation:{old_world_id}:tick-{state['tick']}"
    changed["paused"] = True
    changed["brain_url"] = target_brain_url
    changed["engine_identity"] = target_engine
    # Historical migrations describe the source life and retain their original
    # identities.  Only this new edge belongs to the target branch.
    changed["execution_migrations"] = copy.deepcopy(
        state.get("execution_migrations", [])
    )

    source_evidence = state["evidence_events"]
    observer = CommittedEvents(target_world_id)
    observer.append(
        [
            {
                "kind": "research_continuation",
                "actors": {"bodies": [], "entities": []},
                "quantities": [],
                "details": {
                    "source_tick": state["tick"],
                    "source_checkpoint_state_sha256": source_state_sha,
                    "source_event_head_sha256": source_evidence["head_sha256"],
                },
                "source": {"stream": "explicit-research-migration"},
                "blob_refs": [],
            }
        ],
        tick=state["tick"],
        model_time=float(state["world"]["mj_state"][0]),
    )
    changed["evidence_events"] = observer.snapshot()

    new_metadata = _replace_identity(metadata, old_world_id, target_world_id)
    assert isinstance(new_metadata, dict)
    encoded_metadata = canonical(new_metadata)
    migrated_native = MAGIC + struct.pack("<Q", len(encoded_metadata)) + encoded_metadata + payload
    target_snapshot_name = f"world-{target_world_id}-{state['tick']}-migrated"
    target_snapshot = output_snapshot_dir / f"{target_snapshot_name}.npz"
    _write_new(target_snapshot, migrated_native)
    changed["neural_snapshot"] = {
        **changed["neural_snapshot"],
        "name": target_snapshot_name,
        "sha256": _sha(migrated_native),
        "bytes": len(migrated_native),
    }
    target_residents = list(changed["remote_ids"].values())
    metadata_residents = [
        item for item in new_metadata["resident_ids"] if item is not None
    ]
    if len(metadata_residents) != len(target_residents) or set(metadata_residents) != set(target_residents):
        raise ValueError("migrated neural residents do not exactly cover world remote IDs")

    migration = {
        "format": MIGRATION_FORMAT,
        "from_world_id": old_world_id,
        "to_world_id": target_world_id,
        "tick": state["tick"],
        "from_revision": from_revision,
        "to_revision": to_revision,
        "from_engine_sha256": state["engine_identity"]["sha256"],
        "to_engine_sha256": target_engine_sha,
        "source_checkpoint_file_sha256": _sha(envelope_bytes),
        "source_checkpoint_state_sha256": source_state_sha,
        "source_neural_file_sha256": snapshot_receipt["sha256"],
        "source_neural_payload_sha256": _sha(payload),
        "source_event_snapshot_sha256": _sha(canonical(source_evidence)),
        "source_event_head_sha256": source_evidence["head_sha256"],
        "source_execution_migration_count": len(
            state.get("execution_migrations", [])
        ),
        "target_neural_file_sha256": _sha(migrated_native),
        "target_neural_payload_sha256": _sha(payload),
        "reason": reason,
        "body_identity_mapping": body_mapping,
        "no_model_advance_during_migration": True,
        "state_changes": [
            "world and namespaced resident/event identities rewritten to a new research branch",
            "engine identity changed to the fixed implementation",
            "brain endpoint changed to the isolated continuation service",
            "branch provenance appended and continuation starts paused",
            "MBST1 metadata resident identities rewritten; native neural payload unchanged",
            "observer event stream restarted with one authenticated research-continuation event",
            "journal restarted with one research-continuation entry",
        ],
        "future_numerics": "future execution uses a different cognitive engine; this is not an unchanged-engine continuation",
    }
    retained_components = (
        "world",
        "field",
        "resources",
        "resource_state",
        "biosphere",
        "acoustics",
        "acoustic_state",
        "visitor",
        "visitor_materials",
        "last_senses",
        "sensed_at",
        "neural_state",
        "outcomes",
        "resident_controller",
        "actual_previous",
        "reset_rows",
        "cognition_state",
        "contact_event_state",
        "history",
    )
    migration["retained_component_sha256"] = {}
    for key in retained_components:
        if key in state:
            source_digest = _sha(canonical(state[key]))
            normalized = _replace_identity(
                changed[key], target_world_id, old_world_id
            )
            target_digest = _sha(canonical(normalized))
            if source_digest != target_digest:
                raise ValueError(f"private/numerical component changed: {key}")
            migration["retained_component_sha256"][key] = source_digest
    changed.setdefault("execution_migrations", []).append(migration)
    changed["journal_sequence"] = 1
    changed["journal"] = [
        {
            "id": f"{target_world_id}:1",
            "sequence": 1,
            "time": float(changed["world"]["mj_state"][0]),
            "kind": "implementation",
            "text": "Copied into an explicit research continuation under a fixed cognitive engine.",
            "migration": migration,
        }
    ]
    CommittedEvents.restore(changed["evidence_events"], target_world_id)
    output_state_sha = _sha(canonical(changed))
    output_envelope = canonical(
        {"format": CHECKPOINT_FORMAT, "sha256": output_state_sha, "state": changed}
    )
    _write_new(output_checkpoint, output_envelope)

    receipt = {
        **migration,
        "output_checkpoint_file_sha256": _sha(output_envelope),
        "output_checkpoint_state_sha256": output_state_sha,
        "output_neural_bytes": len(migrated_native),
    }
    receipt_body = canonical(receipt)
    receipt["sha256"] = _sha(receipt_body)
    receipt_path = output_checkpoint.with_name("migration-receipt.json")
    _write_new(receipt_path, canonical(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-snapshot-dir", type=Path, required=True)
    parser.add_argument("--target-engine", dest="target_engine_path", type=Path, required=True)
    parser.add_argument("--target-brain-url", required=True)
    parser.add_argument("--target-world-id", required=True)
    parser.add_argument("--from-revision", required=True)
    parser.add_argument("--to-revision", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    result = migrate(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
