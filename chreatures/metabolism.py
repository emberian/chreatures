"""Named chemistry definitions at the boundary of the native metabolic core.

Names describe bookkeeping columns, never targets supplied to a controller.
Rows are compartments: an organism may have separate gut, body and structural
rows with different enzyme expression under the same reaction program.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .native_world import load_world_kernels

CHEMISTRY_FORMAT = "chreatures-common-chemistry-v1"
SNAPSHOT_FORMAT = "chreatures-metabolic-web-v2"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


class Chemistry:
    """An immutable-by-copy reaction program with checked named columns."""

    def __init__(self, value: Mapping[str, Any]):
        self._value = copy.deepcopy(dict(value))
        if self._value.get("format") != CHEMISTRY_FORMAT:
            raise ValueError("unsupported common chemistry")
        self.elements = tuple(self._value["elements"])
        self.pools = tuple(item["name"] for item in self._value["pools"])
        self.reactions = tuple(item["name"] for item in self._value["reactions"])
        budget = self._value.get("enzyme_budget")
        if budget is None:
            self.enzyme_maximum = None
            self.enzyme_row_sum = None
        else:
            if not isinstance(budget, dict) or set(budget) != {
                "maximum_expression", "row_sum",
            }:
                raise ValueError("chemistry enzyme budget is invalid")
            maximum, row_sum = budget["maximum_expression"], budget["row_sum"]
            if (
                isinstance(maximum, bool) or isinstance(row_sum, bool)
                or not isinstance(maximum, (int, float, np.number))
                or not isinstance(row_sum, (int, float, np.number))
                or not np.isfinite(maximum) or not np.isfinite(row_sum)
                or not 0 < float(maximum) <= float(row_sum)
            ):
                raise ValueError("chemistry enzyme budget is invalid")
            self.enzyme_maximum = float(budget["maximum_expression"])
            self.enzyme_row_sum = float(budget["row_sum"])
        for names in (self.elements, self.pools, self.reactions):
            if (
                not names
                or len(set(names)) != len(names)
                or any(
                    not isinstance(name, str) or not name or len(name) > 96
                    for name in names
                )
            ):
                raise ValueError("chemistry names must be unique nonempty strings")
        self.sha256 = hashlib.sha256(canonical(self._value)).hexdigest()
        r, k = len(self.reactions), len(self.pools)
        stoich = np.zeros((r, k), np.float64)
        half = np.zeros((r, k), np.float64)
        for row, reaction in enumerate(self._value["reactions"]):
            for field, sign in (("consume", -1), ("produce", 1)):
                for name, amount in reaction[field].items():
                    if (
                        name not in self.pools
                        or not np.isfinite(amount)
                        or float(amount) < 0
                    ):
                        raise ValueError("invalid named reaction quantity")
                    stoich[row, self.pools.index(name)] += sign * float(amount)
            half[row, stoich[row] < 0] = float(reaction["half_saturation"])
        self._arrays = (
            stoich,
            np.asarray([p["composition"] for p in self._value["pools"]], np.float64),
            np.asarray(
                [p["chemical_energy"] for p in self._value["pools"]], np.float64
            ),
            *[
                np.asarray(
                    [item[name] for item in self._value["reactions"]], np.float64
                )
                for name in ("atp_cost", "atp_yield", "photon_cost")
            ],
            half,
            np.ones(r, np.float64),
        )
        for array in self._arrays:
            if not np.isfinite(array).all():
                raise ValueError("chemistry arrays must be finite")
            array.flags.writeable = False

    @classmethod
    def load(cls, path: str | Path) -> Chemistry:
        return cls(json.loads(Path(path).read_text()))

    def to_value(self) -> dict[str, Any]:
        return copy.deepcopy(self._value)

    def resources(self, values: Mapping[str, float]) -> np.ndarray:
        if set(values) - set(self.pools):
            raise ValueError("unknown chemical resource")
        vector = np.asarray([values.get(name, 0.0) for name in self.pools], np.float64)
        if not np.isfinite(vector).all() or np.any(vector < 0):
            raise ValueError("resource quantities must be finite and nonnegative")
        return vector

    def enzymes(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        if any(set(row) - set(self.reactions) for row in rows):
            raise ValueError("unknown reaction expression")
        values = np.asarray(
            [[row.get(name, 0.0) for name in self.reactions] for row in rows],
            np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("enzyme expression must be finite and nonnegative")
        if self.enzyme_maximum is not None and (
            np.any(values > self.enzyme_maximum)
            or np.any(values.sum(axis=1) > self.enzyme_row_sum)
        ):
            raise ValueError("enzyme expression exceeds the chemistry allocation budget")
        return values


class MetabolicWeb:
    """Thin owner of private compartments and a conserved external bulk pool."""

    def __init__(
        self,
        chemistry: Chemistry,
        enzymes: Sequence[Mapping[str, float]],
        pools: Sequence[Mapping[str, float]],
        atp: Sequence[float],
        atp_capacity: Sequence[float],
        *,
        regulation: Sequence[Mapping[str, Any]],
        bulk: Mapping[str, float] | None = None,
        bulk_atp: float = 0.0,
    ):
        self.chemistry = chemistry
        self.count = len(enzymes)
        if len(pools) != self.count:
            raise ValueError("enzyme and pool row counts differ")
        enzyme_array = chemistry.enzymes(enzymes)
        self._native = load_world_kernels().MetabolicCohort(
            *chemistry._arrays,
            enzyme_array,
            np.ascontiguousarray([chemistry.resources(row) for row in pools]),
            np.ascontiguousarray(atp, dtype=np.float64),
            np.ascontiguousarray(atp_capacity, dtype=np.float64),
            chemistry.resources(bulk or {}),
            float(bulk_atp),
        )
        self._regulation, arrays = self._regulation_arrays(regulation)
        if not np.array_equal(arrays[0], enzyme_array):
            raise ValueError("regulation baseline must equal initial enzyme expression")
        self._native.enable_regulation(
            *arrays, chemistry.enzyme_maximum, chemistry.enzyme_row_sum
        )
        self.program_sha256 = str(self._native.program_sha256)

    def _regulation_arrays(
        self, rows: Sequence[Mapping[str, Any]], *, expected_count: int | None = None
    ) -> tuple[list[dict[str, Any]], tuple[np.ndarray, ...]]:
        expected = self.count if expected_count is None else expected_count
        if self.chemistry.enzyme_maximum is None or len(rows) != expected:
            raise ValueError("regulated metabolism requires an enzyme budget and one rule per row")
        normalized: list[dict[str, Any]] = []
        for value in rows:
            row = copy.deepcopy(dict(value))
            if set(row) != {
                "baseline", "substrate_response", "atp_response",
                "time_constant_seconds", "change_cost_atp_per_expression",
            }:
                raise ValueError("metabolic regulation rule fields differ")
            for field in ("baseline", "substrate_response", "atp_response"):
                if not isinstance(row[field], dict) or set(row[field]) - set(self.chemistry.reactions):
                    raise ValueError("metabolic regulation reaction names differ")
            normalized.append(row)
        baseline = self.chemistry.enzymes([row["baseline"] for row in normalized])
        named = []
        for field in ("substrate_response", "atp_response"):
            values = np.asarray(
                [[row[field].get(name, 0.0) for name in self.chemistry.reactions] for row in normalized],
                np.float64,
            )
            if not np.isfinite(values).all() or np.any(np.abs(values) > self.chemistry.enzyme_maximum):
                raise ValueError("metabolic regulation response is invalid")
            named.append(np.ascontiguousarray(values))
        tau = np.asarray([row["time_constant_seconds"] for row in normalized], np.float64)
        cost = np.asarray([row["change_cost_atp_per_expression"] for row in normalized], np.float64)
        if (not np.isfinite(tau).all() or np.any((tau < 0.5) | (tau > 120.0))
                or not np.isfinite(cost).all() or np.any((cost < 0.01) | (cost > 2.0))):
            raise ValueError("metabolic regulation timing or cost is invalid")
        return normalized, (np.ascontiguousarray(baseline), *named, tau, cost)

    @property
    def pools(self) -> np.ndarray:
        return np.asarray(self._native.pools)

    @property
    def atp(self) -> np.ndarray:
        return np.asarray(self._native.atp)

    @property
    def enzyme_activity(self) -> np.ndarray:
        return np.asarray(self._native.enzyme_activity)

    @property
    def atp_capacity(self) -> np.ndarray:
        return np.asarray(self._native.atp_capacity)

    @property
    def time(self) -> float:
        return float(self._native.time)

    def totals(self) -> dict[str, Any]:
        resources = self.pools.sum(axis=0) + np.asarray(self._native.bulk_pool)
        elements = resources @ self.chemistry._arrays[1]
        energy = (
            resources @ self.chemistry._arrays[2]
            + self.atp.sum()
            + self._native.bulk_atp
        )
        return {
            "elements": dict(zip(self.chemistry.elements, elements.tolist())),
            "stored_energy": float(energy),
        }

    def regulation_view(self) -> dict[str, Any]:
        """Return whole-web audit state; this is not an organism sensory input."""
        return {
            "reaction_order": list(self.chemistry.reactions),
            "expression": self.enzyme_activity.tolist(),
            "expression_unit": "synthetic_enzyme_expression",
            "maximum_expression": self.chemistry.enzyme_maximum,
            "total_expression_budget": self.chemistry.enzyme_row_sum,
            "time_constant_seconds": [
                float(row["time_constant_seconds"]) for row in self._regulation
            ],
            "cumulative_atp_cost": np.asarray(
                self._native.cumulative_regulation_atp
            ).tolist(),
            "atp_cost_unit": "synthetic_ATP",
        }

    def step(self, dt: float, photons: Any, work: Any) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(value)
            for name, value in self._native.step(
                float(dt),
                np.ascontiguousarray(photons, dtype=np.float64),
                np.ascontiguousarray(work, dtype=np.float64),
            ).items()
        }

    def transfer(
        self,
        donor: int | None,
        receiver: int | None,
        resources: Mapping[str, float],
        *,
        atp: float = 0.0,
    ) -> None:
        self._native.transfer(
            donor, receiver, self.chemistry.resources(resources), float(atp)
        )

    def transfer_batch(
        self,
        donors: Sequence[int | None],
        receivers: Sequence[int | None],
        resources: Sequence[Mapping[str, float]],
        atp: Sequence[float],
    ) -> dict[str, np.ndarray]:
        def endpoint_rows(values: Sequence[int | None]) -> np.ndarray:
            rows = []
            for value in values:
                if value is None:
                    rows.append(-1)
                elif isinstance(value, Integral) and not isinstance(value, bool):
                    rows.append(int(value))
                else:
                    raise ValueError("transfer endpoints must be integer rows or None")
            return np.ascontiguousarray(rows, dtype=np.int64)

        resource_rows = np.ascontiguousarray(
            [self.chemistry.resources(value) for value in resources],
            dtype=np.float64,
        )
        if resource_rows.size == 0:
            resource_rows = np.empty(
                (0, len(self.chemistry.pools)), dtype=np.float64
            )
        result = self._native.transfer_batch(
            endpoint_rows(donors),
            endpoint_rows(receivers),
            resource_rows,
            np.ascontiguousarray(atp, dtype=np.float64),
        )
        return {name: np.asarray(value) for name, value in result.items()}

    def split(self, parent: int, child: int, fraction: float) -> None:
        self._native.split(parent, child, float(fraction))

    def expanded(
        self,
        enzymes: Sequence[Mapping[str, float]],
        atp_capacity: Sequence[float],
        regulation: Sequence[Mapping[str, Any]],
    ) -> MetabolicWeb:
        """Return a private owner with appended empty rows and exact old state."""
        if not 1 <= len(enzymes) <= 5 or len(atp_capacity) != len(enzymes):
            raise ValueError("metabolic expansion requires 1..5 aligned rows")
        capacities = np.ascontiguousarray(atp_capacity, dtype=np.float64)
        if not np.isfinite(capacities).all() or np.any(capacities < 0):
            raise ValueError("expanded ATP capacities must be finite and nonnegative")
        instance = object.__new__(type(self))
        instance.chemistry = self.chemistry
        instance.count = self.count + len(enzymes)
        instance._regulation = self._regulation + copy.deepcopy(list(regulation))
        _, arrays = instance._regulation_arrays(regulation, expected_count=len(regulation))
        new_enzymes = self.chemistry.enzymes(enzymes)
        if not np.array_equal(arrays[0], new_enzymes):
            raise ValueError("newborn regulation baseline must equal enzyme expression")
        instance._native = self._native.expanded(
            arrays[0], capacities, *arrays[1:]
        )
        instance.program_sha256 = str(instance._native.program_sha256)
        if instance.program_sha256 != self.program_sha256:
            raise RuntimeError("metabolic program changed during row expansion")
        return instance

    def pay_work(self, row: int, amount: float) -> None:
        self._native.pay_work(row, float(amount))

    def pay_work_batch(self, rows: Any, amounts: Any) -> None:
        """Atomically debit one packed cohort after native prevalidation."""
        self._native.pay_work_batch(
            np.ascontiguousarray(rows, dtype=np.int64),
            np.ascontiguousarray(amounts, dtype=np.float64),
        )

    def snapshot(self) -> dict[str, Any]:
        data = bytes(self._native.snapshot())
        return {
            "format": SNAPSHOT_FORMAT,
            "chemistry": self.chemistry.to_value(),
            "chemistry_sha256": self.chemistry.sha256,
            "program_sha256": self.program_sha256,
            "compartments": self.count,
            "regulation": copy.deepcopy(self._regulation),
            "regulation_sha256": hashlib.sha256(canonical(self._regulation)).hexdigest(),
            "native_sha256": hashlib.sha256(data).hexdigest(),
            "native_base64": base64.b64encode(data).decode("ascii"),
        }

    @classmethod
    def restore(cls, value: Mapping[str, Any]) -> MetabolicWeb:
        if value.get("format") != SNAPSHOT_FORMAT:
            raise ValueError("unsupported metabolic web snapshot")
        chemistry = Chemistry(value["chemistry"])
        if chemistry.sha256 != value.get("chemistry_sha256"):
            raise ValueError("chemistry identity differs")
        count = int(value["compartments"])
        if not 1 <= count <= 4096:
            raise ValueError("invalid compartment count")
        native = base64.b64decode(value["native_base64"], validate=True)
        if hashlib.sha256(native).hexdigest() != value["native_sha256"]:
            raise ValueError("metabolic state checksum differs")
        regulation = value["regulation"]
        if hashlib.sha256(canonical(regulation)).hexdigest() != value["regulation_sha256"]:
            raise ValueError("metabolic regulation identity differs")
        instance = cls(
            chemistry,
            [row["baseline"] for row in regulation],
            [{}] * count,
            [0.0] * count,
            [0.0] * count,
            regulation=regulation,
        )
        if instance.program_sha256 != value["program_sha256"]:
            raise ValueError("metabolic program identity differs")
        instance._native.restore(native)
        return instance


__all__ = ["Chemistry", "MetabolicWeb"]
