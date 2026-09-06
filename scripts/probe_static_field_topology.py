#!/usr/bin/env python3
"""Focused static-topology field conservation and transport probe."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from chreatures.fields import FieldEnvironment
from chreatures.physics import PhysicsWorld


ROOT = Path(__file__).resolve().parents[1]


def small_world() -> PhysicsWorld:
    spec = json.loads((ROOT / "data/habitats/hollow-garden.json").read_text())
    spec["name"] = "Static field topology probe"
    spec["size"] = [4.0, 2.0, 1.0]
    spec["bodies"] = []
    spec["entities"] = [{
        "id": "floor", "mobility": "static", "material": "soil",
        "physical_material": "earth", "position": [2.0, 1.0, -0.08],
        "shapes": [{"type": "box", "size": [2.0, 1.0, 0.05]}],
        "components": [],
    }]
    spec.pop("assemblies", None)
    return PhysicsWorld(seed=3, spec=spec)


def main() -> None:
    world = small_world()
    field = FieldEnvironment.from_world(world, {
        "grid": [40, 20, 10],
        "channels": [{"name": "trace", "diffusion": 0.045, "decay": 0.0, "uptake": 0.0}],
        "integration_dt": 0.005,
    })
    assert field.sync_static_geometry(world) is None
    field.deposit([1.72, 1.0, 0.5], "trace", 1.0, spread=0.08)
    field.advance(0.32)

    wall = {
        "id": "grown-wall", "mobility": "static", "material": "wood",
        "physical_material": "timber", "position": [2.0, 1.0, 0.5],
        "shapes": [{"type": "box", "size": [0.06, 1.0, 0.5]}],
        "components": [],
    }
    world.prepare_topology_batch([{"op": "add", "entity": wall}]).commit()
    before_build = field.total_mass.copy()
    build = field.sync_static_geometry(world)
    after_build = field.total_mass.copy()
    assert build is not None and build["new_solid_cells"] > 0
    assert build["displaced_mass"][0] > 0.0
    assert np.allclose(after_build, before_build, rtol=0.0, atol=2e-14)
    assert np.all(field.concentration[:, field.solid] == 0.0)

    right = np.arange(field.nx)[None, None, :] >= 21
    right = np.broadcast_to(right, field.grid_shape)
    right_before_closed = float(field.concentration[0, right].sum() * field.cell_volume)
    field.advance(0.7)
    right_after_closed = float(field.concentration[0, right].sum() * field.cell_volume)

    world.prepare_topology_batch([{"op": "remove", "id": "grown-wall"}]).commit()
    before_remove = field.total_mass.copy()
    reopened = field.sync_static_geometry(world)
    assert reopened is not None and reopened["reopened_cells"] > 0
    assert np.all(field.permeability[reopened_mask(field, 19, 20)] == 1.0)
    assert np.array_equal(field.total_mass, before_remove)
    right_before_open = float(field.concentration[0, right].sum() * field.cell_volume)
    field.advance(0.7)
    right_after_open = float(field.concentration[0, right].sum() * field.cell_volume)
    assert abs(right_after_closed - right_before_closed) < 2e-14
    assert right_after_open > right_before_open + 1e-5

    snapshot = field.snapshot()
    restored = FieldEnvironment.restore(copy.deepcopy(snapshot))
    assert restored.snapshot() == snapshot
    assert restored.sync_static_geometry(world) is None

    # A failed all-solid candidate cannot discard an existing field mass.
    class SolidView:
        model_revision = world.model_revision + 1
        @staticmethod
        def view():
            return {
                "dimension": 3, "width": 4.0, "height": 2.0, "depth": 1.0,
                "entities": [{
                    "mobility": "static", "shapes": [{
                        "type": "box", "size": [2.0, 1.0, 0.5],
                        "position": [2.0, 1.0, 0.5],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                    }],
                }],
            }
    before_failed = field.snapshot()
    try:
        field.sync_static_geometry(SolidView())
    except ValueError as exc:
        assert "all solid" in str(exc)
    else:
        raise AssertionError("mass-bearing all-solid topology was accepted")
    assert field.snapshot() == before_failed

    print(json.dumps({
        "build": build,
        "reopened": reopened,
        "right_mass_closed_before_after": [right_before_closed, right_after_closed],
        "right_mass_open_before_after": [right_before_open, right_after_open],
        "total_mass": field.total_mass.tolist(),
        "snapshot_restore_exact": True,
        "all_solid_rejected_atomically": True,
    }, indent=2))


def reopened_mask(field: FieldEnvironment, *x_indices: int) -> np.ndarray:
    mask = np.zeros(field.grid_shape, dtype=bool)
    mask[:, :, list(x_indices)] = True
    return mask


if __name__ == "__main__":
    main()
