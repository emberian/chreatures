"""Current rich family training transport and full-connectome cohort backend."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Mapping
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
from .organism_interface import (
    ACTION_NAMES, MAX_RESIDENTS, PHYSIOLOGY_DIM, RECTIFIED_AXES,
)

RICH_OBSERVATION_CHANNELS = 4096
RICH_SENSORIUM_PROFILE_SHA256 = (
    "c71380718ba5535dbaebdeaf8aa2e88cc45cf218312a03e13507877f02a5554e"
)
RICH_CHANNEL_NAMES_SHA256 = (
    "b4c6b328116d820143e16ee922ccffd7b950dbe008efc580ad93056e01349bfa"
)
ACTION_FIELDS = ACTION_NAMES
BODY_SCALARS = (
    "x",
    "y",
    "z",
    "heading",
    "radius",
    "energy",
    "gut",
    "fatigue",
    "speed",
    "angular_velocity",
    "age",
    "gaze_pitch",
)
BODY_VECTORS = (("quaternion", 4), ("linear_velocity", 3), ("angular_velocity3d", 3))
OUTCOME_FIELDS = (
    "nutrition",
    "contact",
    "distance",
    "effort",
    "mechanical_work",
    "ingested_mass",
    "mouth_material_contacts",
    "homeostatic_reward",
)
HOMEOSTASIS_FIELDS = (
    "potential_delta_energy",
    "effort_cost_energy",
    "nutrition_observed",
    "hunger_gate",
    "reward",
    "before_reserve_energy",
    "before_reserve_shortfall_energy",
    "before_fatigue_cost_energy",
    "before_gut_overload_cost_energy",
    "before_potential_energy",
    "after_reserve_energy",
    "after_reserve_shortfall_energy",
    "after_fatigue_cost_energy",
    "after_gut_overload_cost_energy",
    "after_potential_energy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_graph(path: Path):
    """Load canonical or explicitly derived anatomy through its strict verifier."""
    from .circuit_blueprint import GRAPH_FORMAT, DerivedCircuitGraph
    from .malecns import MaleCNSGraph

    path = path.resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") == GRAPH_FORMAT:
        return DerivedCircuitGraph.load(path, mmap=True, verify=True)
    return MaleCNSGraph.load(path, mmap=True, verify=True)


def _shared_array_layout(
    worlds: int,
    channels: int,
    residents_per_world: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Lay out one cache-line-aligned fixed cohort block."""
    definitions = (
        ("observations", np.dtype("<f4"), (worlds, residents_per_world, channels)),
        ("physiology", np.dtype("<f4"), (worlds, residents_per_world, PHYSIOLOGY_DIM)),
        ("organ_flows", np.dtype("<f8"), (worlds, residents_per_world, 3)),
        (
            "rich_observations",
            np.dtype("<f4"),
            (
                worlds,
                residents_per_world,
                RICH_OBSERVATION_CHANNELS,
            ),
        ),
        (
            "bodies",
            np.dtype("<f8"),
            (
                worlds,
                residents_per_world,
                len(BODY_SCALARS) + sum(size for _name, size in BODY_VECTORS),
            ),
        ),
        (
            "actions",
            np.dtype("<f4"),
            (
                worlds,
                residents_per_world,
                len(ACTION_FIELDS),
            ),
        ),
        (
            "outcomes",
            np.dtype("<f8"),
            (
                worlds,
                residents_per_world,
                len(OUTCOME_FIELDS),
            ),
        ),
        (
            "homeostasis",
            np.dtype("<f8"),
            (
                worlds,
                residents_per_world,
                len(HOMEOSTASIS_FIELDS),
            ),
        ),
        ("intervals", np.dtype("<f8"), (worlds,)),
        ("completed", np.dtype("<i8"), (worlds,)),
        ("worker_seconds", np.dtype("<f8"), (worlds, 2)),
    )
    layout: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, dtype, shape in definitions:
        offset = (offset + 63) // 64 * 64
        layout[name] = {"dtype": dtype.str, "shape": shape, "offset": offset}
        offset += int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    return layout, (offset + 63) // 64 * 64


def _shared_array_views(
    memory: shared_memory.SharedMemory,
    layout: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        name: np.ndarray(
            tuple(specification["shape"]),
            dtype=np.dtype(specification["dtype"]),
            buffer=memory.buf,
            offset=int(specification["offset"]),
        )
        for name, specification in layout.items()
    }


def _body_values(bodies: list[Any], residents_per_world: int) -> np.ndarray:
    if len(bodies) != residents_per_world:
        raise ValueError("shared world body count differs from its fixed cohort row")
    rows = np.empty(
        (
            residents_per_world,
            len(BODY_SCALARS) + sum(size for _name, size in BODY_VECTORS),
        ),
        dtype=np.float64,
    )
    for row, body in enumerate(bodies):
        body = body.to_dict() if hasattr(body, "to_dict") else body
        values = [float(body[name]) for name in BODY_SCALARS]
        for name, size in BODY_VECTORS:
            vector = np.asarray(body[name], dtype=np.float64)
            if vector.shape != (size,):
                raise ValueError(f"body field {name} has the wrong fixed shape")
            values.extend(vector.astype(float).tolist())
        rows[row] = values
    if not np.isfinite(rows).all():
        raise ValueError("body shared state contains nonfinite values")
    return rows


def _write_outcomes(
    target: np.ndarray,
    homeostasis_target: np.ndarray,
    outcomes: Mapping[str, Mapping[str, Any]],
    bodies: list[Any],
) -> None:
    target.fill(0)
    homeostasis_target.fill(0)
    for row, body in enumerate(bodies):
        body_id = body.id if hasattr(body, "id") else str(body["id"])
        outcome = outcomes[body_id]
        homeostasis = outcome.get("homeostasis", {})
        # Actual material flows have their own native shared buffer. Contact
        # identities and emitted signal records are observer metadata, consumed
        # by the interactive runtime's evidence journal. This transport projects
        # only numeric outcomes; it must not turn entity IDs into policy inputs.
        unknown = set(outcome) - set(OUTCOME_FIELDS) - {
            "homeostasis", "released_mass", "secreted_mass", "allocated_mass",
            "contacted_entities", "emitted_signals",
        }
        unknown_homeostasis = set(homeostasis) - set(HOMEOSTASIS_FIELDS)
        if unknown or unknown_homeostasis:
            raise ValueError(
                "shared outcome schema differs: "
                f"fields={sorted(unknown)} homeostasis={sorted(unknown_homeostasis)}"
            )
        contacts = outcome.get("contacted_entities", [])
        signals = outcome.get("emitted_signals", [])
        if not isinstance(contacts, list) or any(not isinstance(x, str) for x in contacts):
            raise ValueError("contacted_entities observer metadata must be a list of IDs")
        if not isinstance(signals, list) or any(not isinstance(x, Mapping) for x in signals):
            raise ValueError("emitted_signals observer metadata must be a list of records")
        target[row] = [float(outcome.get(name, 0.0)) for name in OUTCOME_FIELDS]
        homeostasis_target[row] = [
            float(homeostasis.get(name, 0.0)) for name in HOMEOSTASIS_FIELDS
        ]
    if not np.isfinite(target).all() or not np.isfinite(homeostasis_target).all():
        raise ValueError("outcome shared state contains nonfinite values")


class SharedWorldCohort:
    """Parent-owned fixed buffers with disjoint resident cohort chunks."""

    def __init__(self, worlds: int, channels: int, residents_per_world: int) -> None:
        if worlds < 1 or channels < 1 or not 1 <= residents_per_world <= MAX_RESIDENTS:
            raise ValueError("invalid population transport dimensions")
        self.layout, size = _shared_array_layout(
            worlds,
            channels,
            residents_per_world,
        )
        self.memory = shared_memory.SharedMemory(create=True, size=size)
        self.arrays = _shared_array_views(self.memory, self.layout)
        for array in self.arrays.values():
            array.fill(0)

    def descriptor(self) -> dict[str, Any]:
        return {"name": self.memory.name, "layout": self.layout}

    def close(self) -> None:
        self.arrays.clear()
        self.memory.close()
        self.memory.unlink()


def _world_worker(
    connection,
    port_spec: dict[str, Any],
    profile_value: dict[str, Any],
    physical_backend: str,
    shared_descriptor: Mapping[str, Any],
    world_index: int,
) -> None:
    """Own one MuJoCo instance so native and Python work spans CPU cores."""
    from chreatures.neural_ports import encode_physical_senses
    from chreatures.sensorium import (
        RICH_CHANNEL_NAMES_SHA256 as ACTIVE_RICH_CHANNEL_NAMES_SHA256,
    )
    from chreatures.sensorium import (
        RICH_PROFILE_SHA256 as ACTIVE_RICH_PROFILE_SHA256,
    )

    if (
        ACTIVE_RICH_PROFILE_SHA256 != RICH_SENSORIUM_PROFILE_SHA256
        or ACTIVE_RICH_CHANNEL_NAMES_SHA256 != RICH_CHANNEL_NAMES_SHA256
    ):
        raise RuntimeError("shared transport rich sensorium identity differs")
    from chreatures.training_environment import (
        EmbodiedTrainingProfile,
        EmbodiedTrainingWorld,
        PROFILE_VERSION,
        embodied_training_spec,
    )

    profile = EmbodiedTrainingProfile.from_value(profile_value)
    if int(profile.component("version")) != PROFILE_VERSION:
        raise ValueError(
            "world training transport requires the current regional profile"
        )
    memory = shared_memory.SharedMemory(name=str(shared_descriptor["name"]))
    shared = _shared_array_views(memory, shared_descriptor["layout"])
    world = None
    try:
        while True:
            try:
                operation, payload = connection.recv()
            except EOFError:
                if world is not None and hasattr(world, "close"):
                    world.close()
                return
            if operation == "close":
                if world is not None and hasattr(world, "close"):
                    world.close()
                connection.send((True, None))
                return
            if operation == "observe_shared":
                started = time.perf_counter()
                sequence = int(payload)
                if world is None:
                    raise RuntimeError("world must be reset before observation")
                physical_world = getattr(world, "world", world)
                try:
                    rich = np.asarray(
                        physical_world.rich_retina_batch(refresh=True),
                        dtype=np.float32,
                    )
                except ValueError as error:
                    # Preserve the complete bounded native precondition state.
                    # A failed cohort barrier closes the pool, so this diagnostic
                    # is the only chance to distinguish a pointer rebind from a
                    # malformed body-local input without retrying a world step.
                    import mujoco

                    roots = [
                        mujoco.mj_name2id(
                            physical_world.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            f"resident:{body.id}:geom:thorax",
                        )
                        for body in physical_world.bodies
                    ]
                    heads = [
                        mujoco.mj_name2id(
                            physical_world.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            f"resident:{body.id}:geom:head",
                        )
                        for body in physical_world.bodies
                    ]
                    gaze = [float(body.gaze_pitch) for body in physical_world.bodies]
                    illumination = [
                        float(physical_world._illumination(body))
                        for body in physical_world.bodies
                    ]
                    native = physical_world._native_retina
                    raise RuntimeError(
                        "native rich retina rejected its bound cohort: "
                        f"world_index={world_index}, "
                        f"model_address={int(physical_world.model._address)}, "
                        f"data_address={int(physical_world.data._address)}, "
                        f"roots={roots}, heads={heads}, gaze={gaze}, "
                        f"illumination={illumination}, "
                        f"native_residents={native.residents}, "
                        f"native_rays={native.rays_per_resident}, "
                        f"native_profile_sha256={native.profile_sha256}"
                    ) from error
                vectors = [
                    encode_physical_senses(world.sense(body.id), port_spec)[1]
                    for body in world.bodies
                ]
                observations = np.asarray(vectors, dtype=np.float32)
                expected = shared["observations"][world_index].shape
                if (
                    observations.shape != expected
                    or not np.isfinite(observations).all()
                ):
                    raise ValueError("encoded shared observations have the wrong shape")
                rich_expected = shared["rich_observations"][world_index].shape
                if (
                    rich.shape != rich_expected
                    or not rich.flags.c_contiguous
                    or not np.isfinite(rich).all()
                    or np.any((rich < 0.0) | (rich > 1.0))
                ):
                    raise ValueError("native rich observations have the wrong layout")
                shared["observations"][world_index] = observations
                shared["rich_observations"][world_index] = rich
                shared["bodies"][world_index] = _body_values(
                    world.bodies,
                    shared["bodies"].shape[1],
                )
                shared["physiology"][world_index] = world.physiology_rows()
                shared["organ_flows"][world_index] = world.biosphere.mobility.organ_flows()
                shared["worker_seconds"][world_index, 0] += (
                    time.perf_counter() - started
                )
                shared["completed"][world_index] = sequence
                connection.send((True, sequence))
                continue
            if operation == "advance_shared":
                started = time.perf_counter()
                sequence = int(payload)
                if world is None:
                    raise RuntimeError("world must be reset before advance")
                action_rows = shared["actions"][world_index]
                if not np.isfinite(action_rows).all():
                    raise ValueError("shared actions contain nonfinite values")
                actions = {
                    body.id: {
                        name: float(action_rows[row, column])
                        for column, name in enumerate(ACTION_FIELDS)
                    }
                    for row, body in enumerate(world.bodies)
                }
                try:
                    outcome = world.advance(
                        actions, float(shared["intervals"][world_index])
                    )
                except ValueError as error:
                    if (
                        str(error) == "source is outside field bounds"
                        and getattr(world, "biosphere", None) is not None
                    ):
                        sources = world.field.sources_from_world(world.world)
                        sources.extend(world.biosphere.field_sources())
                        bounds = np.asarray(world.field.size, dtype=np.float64)
                        outside = []
                        for source_index, source in enumerate(sources):
                            position = np.asarray(
                                source.get("position"), dtype=np.float64
                            )
                            if position.shape == (3,) and (
                                np.any(position < 0.0) or np.any(position > bounds)
                            ):
                                outside.append(
                                    {
                                        "index": source_index,
                                        "key": source.get("key"),
                                        "position": position.astype(float).tolist(),
                                        "channel": source.get("channel"),
                                        "rate": source.get("rate"),
                                        "spread": source.get("spread"),
                                    }
                                )
                        raise RuntimeError(
                            "chemical field source left the declared bounds: "
                            f"world_index={world_index}, "
                            f"world_time={float(world.world.time)}, "
                            f"field_size={bounds.astype(float).tolist()}, "
                            f"outside_sources={outside}"
                        ) from error
                    raise
                shared["bodies"][world_index] = _body_values(
                    world.bodies,
                    shared["bodies"].shape[1],
                )
                shared["physiology"][world_index] = world.physiology_rows()
                shared["organ_flows"][world_index] = world.biosphere.mobility.organ_flows()
                _write_outcomes(
                    shared["outcomes"][world_index],
                    shared["homeostasis"][world_index],
                    outcome,
                    world.bodies,
                )
                shared["worker_seconds"][world_index, 1] += (
                    time.perf_counter() - started
                )
                shared["completed"][world_index] = sequence
                connection.send((True, sequence))
                continue
            if operation == "reset":
                if world is not None and hasattr(world, "close"):
                    world.close()
                spec = embodied_training_spec(
                    payload["seed"],
                    held_out=payload.get("held_out", False),
                    stage=0,
                    profile=profile,
                    environment=payload.get("environment"),
                    candidates=payload.get("candidates"),
                )
                world = EmbodiedTrainingWorld(
                    payload["seed"],
                    spec,
                    profile,
                    physical_backend=physical_backend,
                )
                result = [body.to_dict() for body in world.bodies]
            elif operation == "restore":
                if world is not None and hasattr(world, "close"):
                    world.close()
                world = EmbodiedTrainingWorld.restore(
                    payload,
                    expected_profile=profile,
                    physical_backend=physical_backend,
                )
                result = [body.to_dict() for body in world.bodies]
            elif operation == "snapshot":
                result = world.snapshot()
            elif operation == "terminal_outcomes":
                result = world.terminal_outcomes()
            else:
                raise ValueError(f"unknown world worker operation {operation}")
            connection.send((True, result))
    except Exception as exc:  # noqa: BLE001 - worker must return any failure to parent
        try:
            connection.send(
                (False, {"error": repr(exc), "traceback": traceback.format_exc()})
            )
        except (BrokenPipeError, EOFError):
            pass
    finally:
        memory.close()
        connection.close()


class WorldTrainingPool:
    """Fixed-shape numeric transport and rare structured world commands.

    Each worker owns one disjoint world row.  Numeric calls complete through a
    cohort-wide sequence barrier before the parent reads any shared buffer.
    """

    def __init__(
        self,
        count: int,
        port_spec: dict[str, Any],
        profile_value: dict[str, Any],
        physical_backend: str = "fast",
        residents_per_world: int | None = None,
    ) -> None:
        context = mp.get_context("spawn")
        declared_residents = profile_value["value"]["family"]["transport"]["residents"]
        if residents_per_world is None:
            residents_per_world = declared_residents
        if residents_per_world != declared_residents:
            raise ValueError("transport resident count differs from regional profile")
        channels = int(port_spec["physical_inputs"]["count"])
        ordered_names = port_spec["physical_inputs"]["ordered_names"]
        if channels != len(ordered_names) or count <= 0 or not 1 <= residents_per_world <= MAX_RESIDENTS:
            raise ValueError("invalid fixed world cohort dimensions")
        self.residents_per_world = int(residents_per_world)
        self.shared = SharedWorldCohort(count, channels, self.residents_per_world)
        self._sequence = 0
        self._closed = False
        self._hot_calls = {"observe": 0, "advance": 0}
        self._hot_wall_seconds = {"observe": 0.0, "advance": 0.0}
        self._body_templates: list[list[dict[str, Any]]] | None = None
        self.connections = []
        self.processes = []
        try:
            descriptor = self.shared.descriptor()
            for world_index in range(count):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_world_worker,
                    args=(
                        child,
                        port_spec,
                        profile_value,
                        physical_backend,
                        descriptor,
                        world_index,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
        except Exception:
            self._abort()
            raise

    def _abort(self) -> None:
        """Close this pool after any broken barrier; its buffers are unsafe."""
        if self._closed:
            return
        self._closed = True
        for connection in self.connections:
            try:
                connection.close()
            except OSError:
                pass
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        self.shared.close()

    def _barrier(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError("world worker pool is closed")
        self._sequence += 1
        sequence = self._sequence
        try:
            for connection in self.connections:
                connection.send((operation, sequence))
            for connection in self.connections:
                if not connection.poll(300):
                    raise TimeoutError(f"world worker timed out during {operation}")
                ok, value = connection.recv()
                if not ok:
                    raise RuntimeError(
                        f"world worker failed: {value['error']}\n{value['traceback']}"
                    )
                if int(value) != sequence:
                    raise RuntimeError("world worker acknowledged the wrong sequence")
            if not np.all(self.shared.arrays["completed"] == sequence):
                raise RuntimeError("world cohort barrier completed with stale rows")
        except Exception:
            self._abort()
            raise

    def _bodies(self) -> list[list[dict[str, Any]]]:
        if self._body_templates is None:
            raise RuntimeError("worlds must be reset before reading shared body state")
        result = copy.deepcopy(self._body_templates)
        values = self.shared.arrays["bodies"]
        scalar_count = len(BODY_SCALARS)
        for world_index, bodies in enumerate(result):
            for resident_index, body in enumerate(bodies):
                row = values[world_index, resident_index]
                for column, name in enumerate(BODY_SCALARS):
                    body[name] = float(row[column])
                column = scalar_count
                for name, size in BODY_VECTORS:
                    body[name] = row[column : column + size].astype(float).tolist()
                    column += size
        return result

    def _outcomes(self, bodies: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        values = self.shared.arrays["outcomes"]
        homeostasis = self.shared.arrays["homeostasis"]
        for world_index, world_bodies in enumerate(bodies):
            by_resident = {}
            for resident_index, body in enumerate(world_bodies):
                outcome = {
                    name: (
                        int(values[world_index, resident_index, column])
                        if name == "mouth_material_contacts"
                        else float(values[world_index, resident_index, column])
                    )
                    for column, name in enumerate(OUTCOME_FIELDS)
                }
                outcome["homeostasis"] = {
                    name: float(homeostasis[world_index, resident_index, column])
                    for column, name in enumerate(HOMEOSTASIS_FIELDS)
                }
                by_resident[str(body["id"])] = outcome
            result.append(by_resident)
        return result

    def _structured_call(self, operation: str, payloads: list[Any]) -> list[Any]:
        if self._closed:
            raise RuntimeError("world worker pool is closed")
        try:
            for connection, payload in zip(self.connections, payloads, strict=True):
                connection.send((operation, payload))
            results = []
            for connection in self.connections:
                if not connection.poll(300):
                    raise TimeoutError(f"world worker timed out during {operation}")
                ok, value = connection.recv()
                if not ok:
                    raise RuntimeError(
                        f"world worker failed: {value['error']}\n{value['traceback']}"
                    )
                results.append(value)
            return results
        except Exception:
            self._abort()
            raise

    def timing_snapshot(self) -> dict[str, Any]:
        """Return bounded transport costs without synchronizing a GPU."""
        worker = self.shared.arrays["worker_seconds"].copy()
        return {
            "format": "chreatures-shared-world-transport-timing-v1",
            "buffer_bytes": int(self.shared.memory.size),
            "worlds": len(self.connections),
            "residents_per_world": self.residents_per_world,
            "observation_channels": int(self.shared.arrays["observations"].shape[-1]),
            "rich_observation_channels": RICH_OBSERVATION_CHANNELS,
            "rich_sensorium_profile_sha256": RICH_SENSORIUM_PROFILE_SHA256,
            "rich_channel_names_sha256": RICH_CHANNEL_NAMES_SHA256,
            "numeric_layout": {
                name: {
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                }
                for name, array in self.shared.arrays.items()
                if name not in {"completed", "worker_seconds"}
            },
            "hot_calls": copy.deepcopy(self._hot_calls),
            "parent_wall_seconds": copy.deepcopy(self._hot_wall_seconds),
            "worker_cpu_seconds": {
                "observe_sum": float(worker[:, 0].sum()),
                "observe_max_world": float(worker[:, 0].max(initial=0.0)),
                "advance_sum": float(worker[:, 1].sum()),
                "advance_max_world": float(worker[:, 1].max(initial=0.0)),
            },
        }

    def observe_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
        """Return rich and canonical cohort arrays after one complete barrier."""
        if self._closed:
            raise RuntimeError("world worker pool is closed")
        started = time.perf_counter()
        self._barrier("observe_shared")
        rich = (
            self.shared.arrays["rich_observations"]
            .copy()
            .reshape(
                -1,
                RICH_OBSERVATION_CHANNELS,
            )
        )
        observations = (
            self.shared.arrays["observations"]
            .copy()
            .reshape(
                -1,
                self.shared.arrays["observations"].shape[-1],
            )
        )
        bodies = self._bodies()
        self._hot_calls["observe"] += 1
        self._hot_wall_seconds["observe"] += time.perf_counter() - started
        return rich, observations, bodies

    def reset(self, payloads: list[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
        """Create one current family world per worker and return its bodies."""
        if len(payloads) != len(self.connections):
            raise ValueError("world worker payload count differs")
        results = self._structured_call("reset", list(payloads))
        self._body_templates = copy.deepcopy(results)
        return results

    def physiology_array(self, neural_support: np.ndarray) -> np.ndarray:
        """Copy private body readouts and bind current private neural support.

        The worker computes organ state from its actual chemistry. This array
        exposes no positions, entity labels or other residents' private state.
        Call after a completed observe/advance barrier.
        """
        if self._closed or self._body_templates is None:
            raise RuntimeError("worlds must be active before physiology access")
        result = self.shared.arrays["physiology"].reshape(-1, PHYSIOLOGY_DIM).copy()
        support = np.asarray(neural_support, dtype=np.float32)
        if support.shape != (len(result),) or not np.isfinite(support).all():
            raise ValueError("neural support must identify every population row")
        if np.any((support < 0) | (support > 1)):
            raise ValueError("neural support is outside [0,1]")
        result[:, 5] = support
        if not np.isfinite(result).all():
            raise RuntimeError("private body physiology is nonfinite")
        return result

    def organ_flows_array(self) -> np.ndarray:
        """Actual donor-side release, secretion and allocation after the barrier."""
        if self._closed or self._body_templates is None:
            raise RuntimeError("worlds must be active before organ-flow access")
        return self.shared.arrays["organ_flows"].reshape(-1, 3).copy()

    def terminal_outcomes(self) -> list[dict[str, Any]]:
        if self._body_templates is None:
            raise RuntimeError("worlds must be active before outcome access")
        return self._structured_call("terminal_outcomes", [None] * len(self.connections))

    def restore(self, snapshots: list[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
        """Restore exact current family worlds into all worker slots."""
        if len(snapshots) != len(self.connections):
            raise ValueError("world snapshot count differs from worker count")
        results = self._structured_call("restore", list(snapshots))
        self._body_templates = copy.deepcopy(results)
        return results

    def snapshot(self) -> list[dict[str, Any]]:
        """Return complete current family state for every worker."""
        if self._body_templates is None:
            raise RuntimeError("worlds must be reset before snapshot")
        return self._structured_call("snapshot", [None] * len(self.connections))

    def advance(
        self, payloads: list[Mapping[str, Any]]
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Commit one numeric action batch through the shared cohort barrier."""
        if self._body_templates is None:
            raise RuntimeError("worlds must be reset before advance")
        if len(payloads) != len(self.connections):
            raise ValueError("world action payload count differs")
        started = time.perf_counter()
        action_buffer = self.shared.arrays["actions"]
        intervals = self.shared.arrays["intervals"]
        for world_index, (payload, bodies) in enumerate(
            zip(payloads, self._body_templates, strict=True)
        ):
            if not isinstance(payload, Mapping) or set(payload) != {"actions", "dt"}:
                raise ValueError("advance payload must contain only actions and dt")
            interval = float(payload["dt"])
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError("advance dt must be finite and positive")
            intervals[world_index] = interval
            actions = payload["actions"]
            if set(actions) != {str(body["id"]) for body in bodies}:
                raise ValueError("advance action resident IDs differ from world")
            for resident_index, body in enumerate(bodies):
                action = actions[str(body["id"])]
                unknown = set(action) - set(ACTION_FIELDS)
                if unknown:
                    raise ValueError(f"unknown action fields: {sorted(unknown)}")
                action_buffer[world_index, resident_index] = [
                    float(action.get(name, 0.0)) for name in ACTION_FIELDS
                ]
        if (
            not np.isfinite(action_buffer).all()
            or np.any((action_buffer < -1) | (action_buffer > 1))
            or np.any(action_buffer[..., RECTIFIED_AXES] < 0)
        ):
            raise ValueError("shared action cohort exceeds its physical bounds")
        self._barrier("advance_shared")
        bodies = self._bodies()
        outcomes = self._outcomes(bodies)
        self._hot_calls["advance"] += 1
        self._hot_wall_seconds["advance"] += time.perf_counter() - started
        return list(zip(outcomes, bodies, strict=True))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for connection in self.connections:
            try:
                if connection.poll(5):
                    connection.recv()
            except (BrokenPipeError, EOFError):
                pass
            connection.close()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.shared.close()


class TrainingCohortBrain:
    """Current fixed-cohort shell around the measured sparse GPU circuit."""

    def __init__(
        self,
        graph: Any,
        ports: Any,
        batch_size: int,
        *,
        device: str,
        backend: str = "tiled",
    ) -> None:
        import torch

        from .fast_circuit import TritonFusedCircuit
        from .tiled_circuit import MaleCNSEdgeTiledCircuit

        if backend not in {"tiled", "triton"}:
            raise ValueError("training brain backend must be tiled or triton")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("training brain batch size must be a positive integer")
        self._torch = torch
        self.graph = graph
        self.ports = ports
        self.graph_hash = str(graph.hash)
        self.device = torch.device(device)
        self.capacity = batch_size
        self.resident_ids: list[str] = []
        kwargs = {
            "device": device,
            "input_map": (ports.input_names, ports.input_map),
            "readout_map": (ports.readout_names, ports.readout_map),
        }
        self.backend = backend
        circuit_type = (
            MaleCNSEdgeTiledCircuit if backend == "tiled" else TritonFusedCircuit
        )
        self.circuit = circuit_type(graph, batch_size, **kwargs)

    def reset_residents(self, resident_ids: list[str]) -> None:
        """Reset neural state and bind one complete ordered training cohort."""
        clean = [str(value) for value in resident_ids]
        if not clean or len(clean) > self.capacity or len(set(clean)) != len(clean):
            raise ValueError(
                "fixed circuit requires one unique prefix cohort within capacity"
            )
        self.resident_ids = clean
        self.circuit.reset()

    def bind_phenotypes(self, phenotypes: list[Any]) -> None:
        """Install inherited neural arrays once, before a new cohort advances.

        Binding deliberately changes model identity. A continuation must bind
        the identical phenotypes before restoring its private dynamic state.
        """
        from .neural_genotype import batch_neural_phenotypes

        if len(phenotypes) != self.capacity:
            raise ValueError("neural phenotype count differs from cohort capacity")
        if np.any(self.circuit.times != 0):
            raise RuntimeError("neural inheritance is only bound before advancing a life")
        if any(item.active_graph_sha256 != self.graph_hash for item in phenotypes):
            raise ValueError("neural phenotype graph differs from loaded anatomy")
        arrays, identities, group = batch_neural_phenotypes(phenotypes)
        self.circuit.bind_neural_phenotypes(
            arrays, phenotype_sha256=identities, compatibility_group=group
        )

    def step_channels(
        self, channels: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
        values = np.asarray(channels, dtype=np.float32)
        active = len(self.resident_ids)
        if values.shape != (active, self.circuit.input_count):
            raise ValueError("fixed circuit channel batch has the wrong shape")
        if active < self.capacity:
            padded = np.zeros(
                (self.circuit.input_count, self.capacity), dtype=np.float32
            )
            padded[:, :active] = values.T
            device_input = padded
        else:
            device_input = np.ascontiguousarray(values.T)
        result = self.circuit.step_numpy(device_input, dt)
        physiology = result.physiology[:active]
        neural = [
            {
                "activity": float(row[0]),
                "activity_peak": float(row[1]),
                "support": float(row[2]),
            }
            for row in physiology
        ]
        return result.features[:active].copy(), physiology.copy(), neural

    def export_state(self) -> dict[str, np.ndarray]:
        """Copy the complete current neural cohort state for a training checkpoint."""
        return {key: value.copy() for key, value in self.circuit.export_state().items()}

    def snapshot(self, directory: Path, name: str) -> dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.npz"
        temporary = path.with_name(path.name + ".tmp")
        state = self.circuit.export_state()
        metadata = {
            "format": "chreatures-training-cohort-neural-state-v2",
            "backend": self.backend,
            "capacity": self.capacity,
            "graph_sha256": self.graph_hash,
            "resident_ids": self.resident_ids,
            "circuit": self.circuit.metadata(),
        }
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, metadata=np.asarray(json.dumps(metadata)), **state
            )
        os.replace(temporary, path)
        return {"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}

    def restore(
        self, directory: Path, name: str, expected_sha256: str | None = None
    ) -> dict[str, Any]:
        path = Path(directory) / f"{name}.npz"
        digest = _sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("fixed circuit snapshot checksum differs")
        with np.load(path, allow_pickle=False) as value:
            metadata = json.loads(str(value["metadata"]))
            if (
                metadata.get("format") != "chreatures-training-cohort-neural-state-v2"
                or metadata.get("backend") != self.backend
                or metadata.get("capacity") != self.capacity
                or metadata.get("graph_sha256") != self.graph_hash
                or metadata.get("circuit") != self.circuit.metadata()
            ):
                raise ValueError("training circuit snapshot identity differs")
            residents = [str(item) for item in metadata["resident_ids"]]
            state = {
                key: np.asarray(value[key])
                for key in (
                    "rates", "adaptation", "support", "times",
                    "neural_variant_state_identity",
                )
            }
        if not residents or len(residents) > self.capacity:
            raise ValueError("fixed circuit snapshot cohort size differs")
        self.resident_ids = residents
        self.circuit.import_state(state)
        return {
            "name": name,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "residents": residents,
        }

    def metadata(self) -> dict[str, Any]:
        value = self.circuit.metadata()
        value["device"] = {
            "type": self.device.type,
            "name": (
                self._torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else "cpu"
            ),
            "memory_allocated_bytes": (
                self._torch.cuda.memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
        }
        value["residents"] = self.resident_ids
        return value
