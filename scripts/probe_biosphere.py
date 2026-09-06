"""Grow real inherited colonies, retain dead substrate, transfer and digest it.

This is a developmental/physical integration experiment, not a neural policy
or a demonstration of evolved ecology. It does not access any resident service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chreatures.biosphere import Biosphere
from chreatures.checkpoint import canonical
from chreatures.developmental_genome import DevelopmentalGenome
from chreatures.fields import FieldEnvironment
from chreatures.growth import GrowthSystem
from chreatures.physics import PhysicsWorld


def birth_configuration():
    grammar_path = ROOT / "data/growth/reef-builder.json"
    base = DevelopmentalGenome.load(
        ROOT / "data/development/common-ancestor-v1.json"
    ).to_value()
    base["name"] = "reef-ancestor-v1"
    # Supplied slow-development founder expression, expressed per model second.
    # The generic constructor fixture's fast turnover otherwise consumes a
    # photosynthetic surface before it can fund another developmental cycle.
    base["allocation"]["enzyme_activity_budget"] = 0.1
    base["metabolism"]["housekeeping_ceiling"] = 0.05
    base["metabolism"]["housekeeping_expression"][
        "allocated_structure/soft_turnover"
    ] = 0.015
    base["metabolism"]["housekeeping_expression"][
        "allocated_structure/tough_turnover"
    ] = 0.008
    base["sources"]["growth_grammar_file_sha256"] = hashlib.sha256(
        grammar_path.read_bytes()
    ).hexdigest()
    base["sources"]["growth_grammar_sha256"] = GrowthSystem(grammar_path).grammar_hash
    base["sha256"] = None
    founder = DevelopmentalGenome(base)
    genomes = [founder, founder.offspring(37), founder.offspring(81)]
    phenotypes = [genome.compile(growth_grammar=grammar_path) for genome in genomes]
    compartments, colonies = [], []
    for index, (label, genome, phenotype) in enumerate(
        zip("abc", genomes, phenotypes, strict=True)
    ):
        for local, enzymes in enumerate(phenotype.enzyme_rows()):
            compartments.append(
                {
                    "enzymes": enzymes,
                    "pools": {
                        "mineral": 8.0,
                        "inorganic_carbon": 45.0,
                        "reserve": 45.0,
                        "soft_tissue": 0.6,
                        "tough_tissue": 0.8,
                    }
                    if local == 0
                    else {},
                    "atp": 8.0 if local != 2 else 0.0,
                    "atp_capacity": 20.0 if local != 2 else 0.0,
                }
            )
        colonies.append(
            {
                "id": f"colony-{label}",
                "body_row": index * 3,
                "structure_row": index * 3 + 2,
                "grammar": phenotype.growth_grammar,
                "seed": index + 2,
                "bindings": {
                    kind: f"reef-{label}-{plural}"
                    for kind, plural in (
                        ("branch", "branches"),
                        ("root", "roots"),
                        ("leaf", "leaves"),
                    )
                },
                "seed_capture_area": 5.026548245743669e-5,
                "photon_flux": 2000.0,
                "mineral_half_saturation": 0.5,
                "genome_sha256": genome.sha256,
            }
        )
    return {
        "format": "chreatures-biosphere-birth-v1",
        "chemistry": phenotypes[0].chemistry.to_value(),
        "compartments": compartments,
        "bulk": {},
        "colonies": colonies,
    }, [genome.to_value() for genome in genomes]


def run(seconds: float, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    config, genomes = birth_configuration()
    (output / "birth.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "genomes.json").write_text(json.dumps(genomes, indent=2) + "\n")
    spec = json.loads((ROOT / "data/habitats/reef-garden.json").read_text())
    # Physical resident models aren't secretly driven by a procedural policy.
    # This assay isolates substrate development and field response.
    spec["bodies"] = []
    world = PhysicsWorld(seed=102, spec=spec)
    field = FieldEnvironment.from_world(world)
    sphere = Biosphere.from_config(world, config)
    initial_view = world.view()
    events, topology = [], []
    started = time.perf_counter()
    ticks = round(seconds / 0.05)
    for tick in range(ticks):
        world.advance({}, 0.05)
        result = sphere.advance(0.05)
        update = field.sync_static_geometry(world)
        if update:
            topology.append(update)
        field.advance(0.05)
        events.extend(result["developments"])
        if tick and tick % 400 == 0:
            print(
                json.dumps({"time": world.time, "parts": len(sphere.parts)}), flush=True
            )
    elapsed = time.perf_counter() - started
    checkpoint = {
        "world": world.snapshot(),
        "field": field.snapshot(),
        "biosphere": sphere.snapshot(),
    }
    (output / "grown-checkpoint.json").write_text(json.dumps(checkpoint) + "\n")
    # Exercise the real canonical JSON boundary, which sorts dictionary keys.
    checkpoint = json.loads(canonical(checkpoint))
    restored_world = PhysicsWorld.restore(checkpoint["world"])
    restored_field = FieldEnvironment.restore(checkpoint["field"])
    restored = Biosphere.restore(restored_world, checkpoint["biosphere"])
    assert restored.snapshot() == sphere.snapshot()
    assert restored_field.sync_static_geometry(restored_world) is None
    for _ in range(4):
        for w, f, b in (
            (world, field, sphere),
            (restored_world, restored_field, restored),
        ):
            w.advance({}, 0.05)
            b.advance(0.05)
            f.sync_static_geometry(w)
            f.advance(0.05)
    assert restored.snapshot() == sphere.snapshot()
    assert restored_world.snapshot() == world.snapshot()
    assert restored_field.snapshot() == field.snapshot()
    developed_view = world.view()
    # A measured intervention: remove a real built leaf, allocate its tissue to
    # another lineage's gut, then let that lineage's native digestion act.
    leaf = next(
        (key for key, part in sphere.parts.items() if part["kind"] == "leaf"), None
    )
    transfer = None
    if leaf is not None:
        donor = sphere.parts[leaf]["colony"]
        receiver = next(colony for colony in sphere.config if colony["id"] != donor)
        row = receiver["body_row"] + 1
        before = sphere.accounting()
        transfer = sphere.release_parts([leaf], row)
        field.sync_static_geometry(world)
        after = sphere.accounting()
        assert np.allclose(
            list(before["elements"].values()),
            list(after["elements"].values()),
            rtol=0,
            atol=1e-12,
        )
        assert abs(before["stored_energy"] - after["stored_energy"]) < 1e-12
        gut_before = sphere.web.pools[row].copy()
        for _ in range(80):
            world.advance({}, 0.05)
            sphere.advance(0.05)
            field.sync_static_geometry(world)
            field.advance(0.05)
        transfer["gut_before"] = gut_before.tolist()
        transfer["gut_after"] = sphere.web.pools[row].tolist()
        transfer["chemical_columns"] = list(sphere.web.chemistry.pools)
    accounting = sphere.accounting()
    assert max(map(abs, accounting["elemental_residual"].values())) < 1e-9
    assert abs(accounting["energy_residual"]) < 1e-9
    if not events:
        raise RuntimeError("No resource-funded physical development occurred")
    report = {
        "format": "chreatures-biosphere-probe-v1",
        "seconds": seconds,
        "wall_seconds": elapsed,
        "native_graph_executed": False,
        "founders_engineered": True,
        "config_sha256": sphere.config_sha256,
        "program_sha256": sphere.web.program_sha256,
        "events": events,
        "physical_parts": len(sphere.parts),
        "topology_updates": topology,
        "accounting": accounting,
        "same_runtime_continuation_exact": True,
        "experimental_tissue_transfer": transfer,
        "scope": "Finite founder material plus captured photons; physical growth, dead scaffold, intervention-mediated digestion. No autonomous predation or population selection.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "views.json").write_text(
        json.dumps(
            {
                "initial": initial_view,
                "developed": developed_view,
                "final": world.view(),
            }
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"events", "topology_updates"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/biosphere-probe")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 4:
        parser.error("--seconds must be finite and at least four")
    run(args.seconds, args.output)
