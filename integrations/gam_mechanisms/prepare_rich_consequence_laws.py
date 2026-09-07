#!/usr/bin/env python3
"""Extract the native body-law contract from a completed rich-v4 collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.sensorimotor_skills.rich_data import RichPlayDataset
from chreatures.organism_interface import ACTION_NAMES, PHYSIOLOGY_NAMES


COLLECTION_FORMAT = "chreatures-sensorimotor-play-rich-v4"
FEATURE_NAMES = (
    "energy", "fatigue", "body_speed", "support", "neural_activity",
    "thrust", "yaw", "grip", "oral", "motor_magnitude",
    "thrust_x_fatigue", "yaw_x_speed",
)
TARGET_NAMES = ("movement_response", "energy_cost", "fatigue_recovery")
ACTION_ORDER = (
    "thrust", "yaw", "gaze_pitch", "posture", "grip", "signal_low",
    "signal_mid", "signal_high", "eat", "release", "secrete", "allocate",
)
PHYSIOLOGY_ORDER = (
    "energy", "gut", "fatigue", "speed", "turn", "neural_support",
    "structural_integrity", "development_fraction", "gland_fill", "brood_fill",
    "reproductive_maturity", "exchange_load",
)
TRAIN_SLOTS = tuple(range(8))
VALIDATION_SLOTS = (8,)
HOLDOUT_SLOTS = (9,)
EXPECTED_TRANSITIONS = 655_360
PARTITION_CODES = {"train": 0, "validation": 1, "holdout": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_manifest(collection: Path, expected_revision: str | None) -> tuple[dict, Path]:
    if tuple(ACTION_NAMES) != ACTION_ORDER or tuple(PHYSIOLOGY_NAMES) != PHYSIOLOGY_ORDER:
        raise ValueError("current organism-interface names differ from body-law extraction")
    manifest_path = collection / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != COLLECTION_FORMAT or manifest.get("completed") is not True:
        raise ValueError("body-law extraction requires a completed rich-v4 manifest")
    claimed_content = manifest.get("content_sha256")
    unhashed = dict(manifest)
    unhashed.pop("content_sha256", None)
    if claimed_content != canonical_hash(unhashed):
        raise ValueError("collection manifest content hash differs")

    identity = manifest.get("collection_identity")
    if not isinstance(identity, dict):
        raise ValueError("collection manifest lacks its immutable identity")
    identity_body = dict(identity)
    embedded_identity_hash = identity_body.pop("sha256", None)
    if (
        embedded_identity_hash != canonical_hash(identity_body)
        or manifest.get("collection_identity_sha256") != embedded_identity_hash
    ):
        raise ValueError("collection identity hash differs")
    source = identity.get("source", {})
    revision = source.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("collection identity lacks a full source revision")
    if source.get("dirty") is not False:
        raise ValueError("body-law fitting refuses a dirty collection source")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            f"collection revision differs: expected {expected_revision}, got {revision}"
        )
    founding_bank = identity.get("founding_bank")
    resident_artifact = identity.get("resident_artifact")
    candidate_order = identity.get("candidate_order")
    neural_phenotypes = identity.get("neural_phenotypes")
    if (
        not isinstance(founding_bank, dict)
        or any(
            not isinstance(founding_bank.get(name), str)
            or len(founding_bank[name]) != 64
            for name in ("sha256", "file_sha256")
        )
        or not isinstance(candidate_order, list)
        or len(candidate_order) != 80
        or not isinstance(neural_phenotypes, list)
        or len(neural_phenotypes) != 80
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in candidate_order + neural_phenotypes
        )
    ):
        raise ValueError("collection identity lacks the current founding bank cohort")
    if (
        not isinstance(resident_artifact, dict)
        or resident_artifact.get("format")
        != "chreatures-native-developmental-resident-population-v8"
        or resident_artifact.get("version") != 8
        or resident_artifact.get("execution")
        != "developmental-resident-native-population-v8"
        or any(
            resident_artifact.get(name) != identity.get(name)
            for name in ("graph_sha256", "port_spec_sha256", "port_bundle_sha256")
        )
    ):
        raise ValueError("collection identity is not the current native-v8 controller seam")
    profile = manifest.get("profile")
    if (
        identity.get("profile") != profile
        or not isinstance(profile, dict)
        or not isinstance(profile.get("sha256"), str)
        or len(profile["sha256"]) != 64
    ):
        raise ValueError("collection world profile identity differs")

    scope = manifest.get("scope", {})
    split = (
        tuple(scope.get("train_world_slots", [])),
        tuple(scope.get("validation_world_slots", [])),
        tuple(scope.get("heldout_world_slots", [])),
    )
    if split != (TRAIN_SLOTS, VALIDATION_SLOTS, HOLDOUT_SLOTS):
        raise ValueError(f"rich-v4 split differs from pinned 0-7/8/9 contract: {split}")
    if (
        int(scope.get("worlds", -1)) != 10
        or int(scope.get("residents_per_world", -1)) != 8
        or int(scope.get("episodes", -1)) != 2
        or int(scope.get("steps_per_episode", -1)) != 4096
        or int(scope.get("shard_steps", -1)) != 512
        or float(scope.get("dt_seconds", -1.0)) != 0.05
        or int(manifest.get("transitions", -1)) != EXPECTED_TRANSITIONS
    ):
        raise ValueError("collection scope differs from W10/R8/E2/T4096 rich-v4 campaign")
    if identity.get("split") != {
        "train_world_slots": list(TRAIN_SLOTS),
        "validation_world_slots": list(VALIDATION_SLOTS),
        "heldout_world_slots": list(HOLDOUT_SLOTS),
    }:
        raise ValueError("collection identity split differs from manifest scope")
    if identity.get("observation_order") != [
        "rich_body_v1_4096", "canonical_channels_351", "physiology_12",
    ]:
        raise ValueError("rich-v4 observation order differs")
    if manifest.get("rich_sensorium", {}).get("observation_order") != identity.get(
        "observation_order"
    ):
        raise ValueError("rich sensorium observation identity differs")
    return manifest, manifest_path


def packet_schedule(manifest: dict) -> list[dict]:
    scope = manifest["scope"]
    episodes = int(scope["episodes"])
    steps = int(scope["steps_per_episode"])
    shard_steps = int(scope["shard_steps"])
    expected = [
        (episode, shard, start, start + shard_steps)
        for episode in range(episodes)
        for shard, start in enumerate(range(0, steps, shard_steps))
    ]
    packets = sorted(
        manifest.get("packets", []),
        key=lambda item: (int(item["episode"]), int(item["start_tick"])),
    )
    actual = [
        (int(item.get("episode", -1)), int(item.get("shard", -1)),
         int(item.get("start_tick", -1)), int(item.get("stop_tick", -1)))
        for item in packets
    ]
    if actual != expected:
        raise ValueError("manifest packet schedule is incomplete, duplicated, or non-contiguous")
    return packets


def extract_arrays(
    observation: np.ndarray,
    action: np.ndarray,
    neural: np.ndarray | None,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    observation = np.asarray(observation)
    action = np.asarray(action)
    if neural is None:
        raise ValueError("body-law fitting requires canonical MaleCNS neural readouts")
    neural = np.asarray(neural)
    ticks, residents = action.shape[:2] if action.ndim == 3 else (-1, -1)
    if (
        observation.shape != (ticks + 1, residents, 4459)
        or action.shape != (ticks, residents, 12)
        or neural.shape != (ticks + 1, residents, 384)
        or float(dt_seconds) != 0.05
    ):
        raise ValueError("rich-v4 packet array shapes or timestep differ")
    physiology = observation[..., 4447:4459].astype(np.float64)
    state = physiology[:-1]
    action64 = action.astype(np.float64)
    # LawBank receives thrust, yaw, gaze pitch, and grip as its first four
    # motor channels. Rich-v4 posture occupies executed action index 3, so an
    # actions[..., :4] shortcut would silently learn the wrong magnitude.
    motor = np.abs(action64[..., (0, 1, 2, 4)]).mean(axis=-1)
    features = np.stack(
        (
            state[..., 0], state[..., 2], state[..., 3], state[..., 5],
            neural[:-1].astype(np.float64).mean(axis=-1),
            action64[..., 0], action64[..., 1], action64[..., 4],
            action64[..., 8], motor,
            action64[..., 0] * state[..., 2],
            action64[..., 1] * state[..., 3],
        ),
        axis=-1,
    )
    outcomes = np.stack(
        (
            physiology[1:, :, 3] - state[..., 3],
            state[..., 0] - physiology[1:, :, 0],
            state[..., 2] - physiology[1:, :, 2],
        ),
        axis=-1,
    )
    if not np.isfinite(features).all() or not np.isfinite(outcomes).all():
        raise ValueError("rich-v4 body-law rows contain non-finite values")
    return features.astype(np.float32), outcomes.astype(np.float32)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("compact output already exists; use a fresh path")
    manifest, manifest_path = require_manifest(args.collection, args.source_revision)
    packets = packet_schedule(manifest)
    dataset = RichPlayDataset(args.collection)
    if dataset.manifest != manifest or len(dataset.packets) != len(packets):
        raise ValueError("authenticated rich-v4 loader differs from manifest")
    scope = manifest["scope"]
    worlds = int(scope["worlds"])
    residents_per_world = int(scope["residents_per_world"])
    count = worlds * residents_per_world
    total = int(manifest["transitions"])
    features = np.empty((total, len(FEATURE_NAMES)), dtype=np.float32)
    outcomes = np.empty((total, len(TARGET_NAMES)), dtype=np.float32)
    world_unit = np.empty(total, dtype=np.uint8)
    world_slot = np.empty(total, dtype=np.uint8)
    resident_slot = np.empty(total, dtype=np.uint8)
    episode_unit = np.empty(total, dtype=np.uint8)
    tick_unit = np.empty(total, dtype=np.uint16)
    partition = np.empty(total, dtype=np.uint8)
    packet_hashes: list[str] = []
    cursor = 0
    slot_row = np.arange(count, dtype=np.uint8) // residents_per_world
    resident_row = np.arange(count, dtype=np.uint8) % residents_per_world
    partition_row = np.select(
        (np.isin(slot_row, TRAIN_SLOTS), np.isin(slot_row, VALIDATION_SLOTS),
         np.isin(slot_row, HOLDOUT_SLOTS)),
        (PARTITION_CODES["train"], PARTITION_CODES["validation"],
         PARTITION_CODES["holdout"]),
        default=255,
    ).astype(np.uint8)
    if np.any(partition_row == 255):
        raise AssertionError("world slot escaped the pinned split")

    for episode_data in dataset.iter_contiguous_shards():
        packet_features, packet_outcomes = extract_arrays(
            episode_data.observation,
            episode_data.actions,
            episode_data.neural,
            dataset.dt_seconds,
        )
        ticks = episode_data.stop_tick - episode_data.start_tick
        if packet_features.shape != (ticks, count, len(FEATURE_NAMES)):
            raise ValueError(f"resident layout differs for {episode_data.packet_path.name}")
        rows = ticks * count
        destination = slice(cursor, cursor + rows)
        features[destination] = packet_features.reshape(rows, len(FEATURE_NAMES))
        outcomes[destination] = packet_outcomes.reshape(rows, len(TARGET_NAMES))
        world_slot[destination] = np.broadcast_to(slot_row, (ticks, count)).reshape(-1)
        resident_slot[destination] = np.broadcast_to(resident_row, (ticks, count)).reshape(-1)
        episode = episode_data.episode
        episode_unit[destination] = episode
        world_unit[destination] = np.broadcast_to(
            episode * worlds + slot_row, (ticks, count)
        ).reshape(-1)
        tick_unit[destination] = np.broadcast_to(
            np.arange(episode_data.start_tick, episode_data.stop_tick,
                      dtype=np.uint16)[:, None],
            (ticks, count),
        ).reshape(-1)
        partition[destination] = np.broadcast_to(partition_row, (ticks, count)).reshape(-1)
        cursor += rows
        packet_hashes.append(episode_data.packet_sha256)
    if cursor != total:
        raise ValueError(f"extracted {cursor} rows, expected {total}")

    identity = manifest["collection_identity"]
    contract = {
        "status": "successor candidate for new births; no resident promotion",
        "collection_format": COLLECTION_FORMAT,
        "source_revision": identity["source"]["revision"],
        "manifest_sha256": sha256(manifest_path),
        "manifest_content_sha256": manifest["content_sha256"],
        "collection_identity_sha256": manifest["collection_identity_sha256"],
        "graph_sha256": identity.get("graph_sha256"),
        "port_spec_sha256": identity.get("port_spec_sha256"),
        "port_bundle_sha256": identity.get("port_bundle_sha256"),
        "resident_artifact": identity.get("resident_artifact"),
        "founding_bank": identity.get("founding_bank"),
        "candidate_order_sha256": canonical_hash(identity.get("candidate_order")),
        "neural_phenotypes_sha256": canonical_hash(identity.get("neural_phenotypes")),
        "world_profile_sha256": manifest.get("profile", {}).get("sha256"),
        "rich_profile_sha256": manifest.get("rich_sensorium", {}).get("profile_sha256"),
        "rich_channel_names_sha256": manifest.get("rich_sensorium", {}).get(
            "channel_names_sha256"
        ),
        "dt_seconds": float(scope["dt_seconds"]),
        "action_order": list(ACTION_ORDER),
        "physiology_order": list(PHYSIOLOGY_ORDER),
        "feature_names": list(FEATURE_NAMES),
        "feature_contract": (
            "native LawBank::fitted_features v1 from organism-interface-v4 actual "
            "executed actions; grip index4, oral/eat index8, motor indices0,1,2,4"
        ),
        "target_names": list(TARGET_NAMES),
        "target_contract": [
            "next_speed-now_speed", "now_energy-next_energy",
            "now_fatigue-next_fatigue",
        ],
        "split": {
            "train_world_slots": list(TRAIN_SLOTS),
            "validation_world_slots": list(VALIDATION_SLOTS),
            "heldout_world_slots": list(HOLDOUT_SLOTS),
            "rule": "whole episode/world units; residents sharing a world never cross partitions",
        },
        "scope": {
            "rows": total, "worlds": worlds,
            "residents_per_world": residents_per_world,
            "episodes": int(scope["episodes"]),
            "steps_per_episode": int(scope["steps_per_episode"]),
        },
    }
    atomic_npz(
        args.output,
        {
            "features": features, "outcomes": outcomes, "partition": partition,
            "world_unit": world_unit, "world_slot": world_slot,
            "resident_slot": resident_slot, "episode_unit": episode_unit,
            "tick_unit": tick_unit, "source_sha256": np.asarray(packet_hashes),
            "source_contract": np.asarray(json.dumps(contract, sort_keys=True)),
        },
    )
    print(json.dumps({
        "schema": "chreatures-rich-v4-body-law-compact-receipt-v1",
        "rows": total,
        "packets": len(packets),
        "partition_rows": {
            name: int(np.sum(partition == code))
            for name, code in PARTITION_CODES.items()
        },
        "independent_episode_world_units": {
            name: int(np.unique(world_unit[partition == code]).size)
            for name, code in PARTITION_CODES.items()
        },
        "output_sha256": sha256(args.output),
        "source": contract,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
